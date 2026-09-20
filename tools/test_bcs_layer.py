import os
from live.bcs_executor import BCSExecutor
from live.bcs_models import BCSOrder

os.environ["BCS_EXECUTION_MODE"] = "paper"
result = BCSExecutor().submit(
    BCSOrder(ticker="GAZP", side="BUY", lots=55, shares=550, price=90.25, reason="TEST")
)
assert result.ok and result.status == "paper"
print("BCS paper execution test: OK")
print(result.raw)
