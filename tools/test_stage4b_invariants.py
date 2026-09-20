"""Stage 4B focused invariants: no warmup/live replay and no timestamp dedup."""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live.live_pipeline import LiveStrategy

def main():
    s = LiveStrategy.__new__(LiveStrategy)
    s.ticker = "TEST"
    s.initial_capital = 100000.0
    s.capital = 100000.0
    s.position = None
    s.trades = []
    s.live_started = False
    s.live_cutover_time = None
    s.live_cutover_trade_count = 0
    s._seen_trade_ids = set()
    s._seen_fallback_keys = set()
    s.last_processed_time = None
    s.last_input_ts = None
    s._bar_buffer = []
    s._bar_value = 0.0
    s.dollar_threshold = 1000.0
    s.lot_size = 1.0
    s.last_signal = 0
    s.warmup_done = True
    s.logs = []
    s._log = lambda msg: None

    snapshot = [
        {"ts": "2026-09-03 12:00:00+03:00", "price": 100, "quantity": 10, "value": 1000, "trade_id": "1"},
        {"ts": "2026-09-03 12:00:00+03:00", "price": 101, "quantity": 10, "value": 1010, "trade_id": "2"},
    ]
    s.start_live(snapshot=snapshot, cutover_time="2026-09-03 12:00:01+03:00")

    assert s.position is None
    assert s.capital == s.initial_capital
    assert s.trades == []
    assert len(s._seen_trade_ids) == 2

    # Same timestamp is legitimate; a new TRADENO must be accepted.
    accepted = s.process_trade({
        "ts": "2026-09-03 12:00:00+03:00",
        "price": 102,
        "quantity": 10,
        "value": 1020,
        "trade_id": "3",
    })
    assert accepted is True

    # Same TRADENO is a delivery duplicate and must be ignored.
    accepted2 = s.process_trade({
        "ts": "2026-09-03 12:00:00+03:00",
        "price": 999,
        "quantity": 1,
        "value": 999,
        "trade_id": "3",
    })
    assert accepted2 is False

    print("[PASS] live cutover does not replay snapshot")
    print("[PASS] warmup accounting remains clean")
    print("[PASS] equal timestamps do not deduplicate")
    print("[PASS] TRADENO deduplication works")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
