"""Small regression test for the no-trade-dedup invariant."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.for_ml.builders.dollar_bars import InformationDrivenBars
from data.for_ml.labeling.cusum import cusum_filter


def main() -> int:
    ts = pd.Timestamp("2026-08-28 12:00:00", tz="Europe/Moscow")
    trades = pd.DataFrame({
        "ts": [ts, ts, ts, ts + pd.Timedelta(seconds=1)],
        "price": [100.0, 100.1, 100.2, 100.3],
        "quantity": [100.0, 100.0, 100.0, 100.0],
        "trade_id": [101, 102, 103, 104],
    })
    bars = InformationDrivenBars().build_dollar(trades, threshold=10_000)
    assert len(trades) == 4
    assert trades["ts"].duplicated().sum() == 2
    assert int(bars["count"].sum()) == 4, "A trade disappeared while building bars"
    assert abs(float(bars["value"].sum()) - float((trades.price * trades.quantity).sum())) < 1e-9

    closes = pd.Series(
        [100.0, 100.2, 99.8, 100.5],
        index=[ts, ts, ts + pd.Timedelta(seconds=1), ts + pd.Timedelta(seconds=2)],
    )
    events = cusum_filter(closes, threshold=0.001)
    assert len(events) >= 1
    print("PASS: duplicate timestamps are preserved; dollar-bar turnover/count is conserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
