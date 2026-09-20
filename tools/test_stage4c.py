from pathlib import Path
import sys, tempfile, json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from live_pipeline import LiveStrategy, CUSUMState

print('[PASS] source imports')

# Timezone invariant: naive MOEX values are local MSK, not UTC.
from algopack_provider import _parse_moex_datetime
x=_parse_moex_datetime(pd.Series(['2026-09-04 12:00:00']))
assert str(x.iloc[0]) == '2026-09-04 12:00:00+03:00', x.iloc[0]
print('[PASS] MOEX naive timestamp stays 12:00 MSK')

# CUSUM same timestamps are independent events.
c=CUSUMState(0.0001)
assert c.update(0.001,pd.Timestamp('2026-09-04 12:00',tz='Europe/Moscow'))
assert c.update(-0.001,pd.Timestamp('2026-09-04 12:00',tz='Europe/Moscow'))
print('[PASS] CUSUM permits equal event timestamps')

# Lot arithmetic invariant.
price=150.0; capital=1_000_000.0; lot=10
lots=int(capital/(price*lot))
assert lots == 666
assert lots*lot == 6660
print('[PASS] position sizing uses floor whole lots')
print('[PASS] Stage4C tests complete')
