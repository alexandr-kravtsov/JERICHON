"""JERICHON end-to-end live smoke test.

Tests without touching production parquet:
1. provider import + mock trades with equal timestamps;
2. lossless timestamp handling;
3. dollar-bar construction with several bars at one timestamp;
4. CUSUM event generation;
5. optional real model/artifact loading for a ticker;
6. feature -> model -> position path using synthetic post-warmup trades.

Run from project root:
    python tools/smoke_test_live.py --ticker GAZP
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import tempfile
import json
import pickle

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "data"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.for_ml.providers.algopack_provider import AlgoPackProvider
from data.for_ml.builders.dollar_bars import InformationDrivenBars
from data.for_ml.labeling.cusum import cusum_filter


def check_provider() -> None:
    provider = AlgoPackProvider(mock=True)
    raw = pd.DataFrame({
        "TRADETIME": ["2026-08-31 10:00:00.000", "2026-08-31 10:00:00.000"],
        "TRADENO": [101, 102], "TRADEPRICE": [100.0, 100.1],
        "QUANTITY": [100.0, 100.0], "VALUE": [10000.0, 10010.0], "BUYSELL": ["B", "S"],
    })
    out = provider._normalize_trades(raw, "GAZP", "2026-08-31")
    assert len(out) == 2, "Provider removed a trade row"
    assert out["ts"].duplicated(keep=False).all(), "Synthetic equal timestamp was not preserved"
    assert out["trade_id"].tolist() == [101, 102], "TRADENO was not preserved"
    print("[PASS] provider normalization preserves equal timestamps + TRADENO")


def check_bars_and_cusum() -> None:
    ts = pd.Timestamp("2026-08-31 10:00:00", tz="Europe/Moscow")
    trades = pd.DataFrame({
        "ts": [ts, ts, ts, ts],
        "price": [100.0, 101.0, 99.0, 100.5],
        "quantity": [100_000.0] * 4,
    })
    bars = InformationDrivenBars().build_dollar(trades, threshold=10_000_000)
    assert len(bars) >= 3, f"Expected several bars, got {len(bars)}"
    assert bars["close_time"].duplicated().any(), "Test did not create same-timestamp bars"
    print(f"[PASS] dollar bars preserve same-timestamp bars: {len(bars)} bars")

    closes = pd.Series([100.0, 101.0, 102.5, 99.0, 97.0, 100.0],
                       index=pd.date_range(ts, periods=6, freq="min"))
    events = cusum_filter(closes, threshold=0.0045)
    assert len(events) > 0, "CUSUM produced no events on a deliberately moving series"
    print(f"[PASS] CUSUM generated {len(events)} event(s)")


def find_local_trades(ticker: str) -> pd.DataFrame:
    base = ROOT / "data" / "for_ml" / "storage" / "history" / "realtime" / ticker
    files = sorted(base.glob("*/trades.parquet"), reverse=True)
    frames = []
    total_value = 0.0
    threshold = None
    params_path = ROOT / "models" / "params" / f"{ticker}_params.json"
    if params_path.exists():
        try:
            threshold = float(json.loads(params_path.read_text(encoding="utf-8")).get("dollar_threshold", 10_000_000))
        except Exception:
            pass
    threshold = threshold or 10_000_000
    for path in files:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        frames.append(df)
        total_value += float(pd.to_numeric(df.get("VALUE"), errors="coerce").sum())
        if total_value >= threshold * 60:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def check_real_pipeline(ticker: str) -> None:
    models = ROOT / "models" / "params"
    model_files = [p for p in models.glob(f"{ticker}_*.pkl") if "_features" not in p.stem and "_params" not in p.stem]
    params = models / f"{ticker}_params.json"
    features = models / f"{ticker}_features.pkl"
    if not model_files or not params.exists() or not features.exists():
        print(f"[SKIP] real artifacts for {ticker} not found")
        return

    try:
        from live.live_pipeline import LiveStrategy
    except Exception as exc:
        print(f"[SKIP] live feature-builder imports unavailable in this environment: {exc}")
        return
    strategy = LiveStrategy(ticker, sorted(model_files)[0], params, features, 100_000.0)
    raw = find_local_trades(ticker)
    if raw.empty:
        print(f"[SKIP] no local raw trades for {ticker}")
        return

    # Normalize through the same public provider contract used by Terminal.
    provider = AlgoPackProvider(mock=True)
    trades = provider._normalize_trades(raw, ticker)
    warm = strategy.warmup(trades)
    print(f"[PASS] real artifacts load: bars={warm['bars']} events={len(strategy.cusum_events)} ready={warm['ready']}")
    if not strategy.warmup_done:
        print("[WARN] local history is not yet sufficient for the model feature state")
        return

    # Synthetic post-warmup stream: one threshold-sized trade per bar. This
    # exercises the causal path without modifying parquet files.
    start = strategy.last_processed_time + pd.Timedelta(seconds=1)
    price = 100.0
    before_events = len(strategy.cusum_events)
    before_trades = len(strategy.trades)
    threshold = strategy.dollar_threshold
    for i in range(max(strategy.min_bars_for_features + 5, 60)):
        price *= 1.006 if i % 2 == 0 else 0.994
        value = threshold * 1.01
        t = start + pd.Timedelta(seconds=i)
        strategy.process_trade({"ts": t, "price": price, "quantity": value / price,
                                "value": value, "trade_id": f"SMOKE-{i}"})
    new_events = len(strategy.cusum_events) - before_events
    new_trades = len(strategy.trades) - before_trades
    print(f"[RESULT] post-warmup bars={len(strategy.dollar_bars)} new_events={new_events} new_closed_trades={new_trades} open_position={bool(strategy.position)}")
    if new_events == 0:
        raise AssertionError("Post-warmup synthetic stream generated no CUSUM events")
    if strategy.last_compute_error:
        print(f"[WARN] last_compute_error={strategy.last_compute_error}")
    if strategy.last_compute_missing:
        print(f"[WARN] missing_features={strategy.last_compute_missing}")
    if new_trades == 0 and not strategy.position:
        print("[WARN] no position was opened; inspect feature/model compatibility above")
    else:
        print("[PASS] CUSUM -> features -> model -> position path reached trading state")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="GAZP")
    args = ap.parse_args()
    check_provider()
    check_bars_and_cusum()
    check_real_pipeline(args.ticker)
    print("\nSMOKE TEST FINISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
