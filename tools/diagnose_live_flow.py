"""JERICHON live-flow audit.

Runs the SAME LiveStrategy pipeline on historical data:
previous local days -> warmup -> today's complete tape as replay.
It does not send broker orders. It reports exactly where events disappear.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[2] / "models").exists() else Path.cwd()
DATA = ROOT / "data"
MODELS = ROOT / "models" / "params"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DATA))

from live.live_pipeline import LiveStrategy


def load_day(ticker: str, date: str) -> pd.DataFrame:
    p = DATA / "for_ml" / "storage" / "history" / "realtime" / ticker / date / "trades.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for canonical, variants in {
        "ts": ["ts", "tradetime", "TRADETIME"],
        "price": ["price", "TRADEPRICE"],
        "quantity": ["quantity", "QUANTITY"],
        "value": ["value", "VALUE"],
        "trade_id": ["trade_id", "TRADENO"],
        "side": ["side", "BUYSELL"],
    }.items():
        for v in variants:
            if v.lower() in cols:
                rename[cols[v.lower()]] = canonical
                break
    df = df.rename(columns=rename)
    if "ts" not in df or "price" not in df or "quantity" not in df:
        return pd.DataFrame()
    if "value" not in df:
        df["value"] = pd.to_numeric(df["price"], errors="coerce") * pd.to_numeric(df["quantity"], errors="coerce")
    df["ts"] = pd.to_datetime(df["ts"], format="mixed", errors="coerce")
    if getattr(df["ts"].dt, "tz", None) is None:
        df["ts"] = df["ts"].dt.tz_localize("Europe/Moscow")
    else:
        df["ts"] = df["ts"].dt.tz_convert("Europe/Moscow")
    return df.dropna(subset=["ts", "price", "quantity", "value"]).sort_values("ts", kind="mergesort")


def available_dates(ticker: str) -> list[str]:
    base = DATA / "for_ml" / "storage" / "history" / "realtime" / ticker
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []


def audit(ticker: str, today: str, warmup_days: int) -> dict:
    params = MODELS / f"{ticker}_params.json"
    features = MODELS / f"{ticker}_features.pkl"
    models = [p for p in MODELS.glob(f"{ticker}_*.pkl") if "_features" not in p.stem]
    if not params.exists() or not features.exists() or not models:
        raise FileNotFoundError(f"Missing model artifacts for {ticker} in {MODELS}")

    strategy = LiveStrategy(ticker, sorted(models)[0], params, features, initial_capital=50_000.0)
    dates = [d for d in available_dates(ticker) if d < today]
    warm_dates = dates[-warmup_days:] if warmup_days > 0 else dates
    warm = pd.concat([load_day(ticker, d) for d in warm_dates], ignore_index=True) if warm_dates else pd.DataFrame()
    w = strategy.warmup(warm)
    day = load_day(ticker, today)
    strategy.start_live(snapshot=[])
    processed = strategy.process_trades(day.to_dict("records")) if not day.empty else 0
    d = strategy.get_debug_info()
    d["audit"] = {
        "ticker": ticker,
        "today": today,
        "warmup_dates": warm_dates,
        "today_rows": len(day),
        "today_processed": processed,
        "warmup": w,
    }
    # JSON-safe summary
    d["last_processed_time"] = str(d.get("last_processed_time"))
    d["live_cutover_time"] = str(d.get("live_cutover_time"))
    d["last_event"] = str(d.get("last_event"))
    if d.get("last_prediction"):
        d["last_prediction"]["time"] = str(d["last_prediction"].get("time"))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="+", help="e.g. ROSN RUAL")
    ap.add_argument("--today", default=datetime.now().astimezone().strftime("%Y-%m-%d"))
    ap.add_argument("--warmup-days", type=int, default=5)
    ap.add_argument("--out", default="live_flow_audit.json")
    args = ap.parse_args()

    result = {}
    for ticker in args.tickers:
        try:
            result[ticker] = audit(ticker, args.today, args.warmup_days)
            q = result[ticker]["stats"]
            print(f"\n=== {ticker} ===")
            for k in ["trades_accepted", "trades_duplicate", "bars_closed", "cusum_events", "predictions", "pred_long", "pred_short", "same_signal", "opens", "reversals", "barrier_closes", "closed_trades", "feature_errors", "prediction_errors"]:
                print(f"{k:35s}: {q.get(k, 0)}")
            print("position:", result[ticker]["position"])
            print("last prediction:", result[ticker]["last_prediction"])
        except Exception as exc:
            result[ticker] = {"error": repr(exc)}
            print(f"\n=== {ticker} ERROR ===\n{exc}")

    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nAudit written to: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
