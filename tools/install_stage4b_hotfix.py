from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
FILES={
 "live/live_pipeline.py":"live/live_pipeline.py",
 "data/for_ml/providers/algopack_provider.py":"data/for_ml/providers/algopack_provider.py",
 "app/pages/03_Terminal.py":"app/pages/03_Terminal.py",
}
SRC=Path(__file__).resolve().parents[1]
def backup(p):
 b=p.with_suffix(p.suffix+".stage4b_hotfix.bak")
 if not b.exists(): shutil.copy2(p,b); print("[BACKUP]",b)
for rel in FILES:
 p=ROOT/rel; src=SRC/rel
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): backup(p)
 shutil.copy2(src,p); print("[INSTALLED]",p)
print("[DONE] Stage4B hotfix installed")
