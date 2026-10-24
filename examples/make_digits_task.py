"""Build a self-contained smoke task (sklearn digits) to test the runner and the two Bots.

  python examples/make_digits_task.py /workspace/smoke
  cd /workspace/smoke && python referee/runner.py init && python referee/runner.py run runs/001
"""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_digits

KIT = Path(__file__).resolve().parent.parent
root = Path(sys.argv[1] if len(sys.argv) > 1 else "smoke").resolve()
if root.exists():
    sys.exit(f"{root} already exists; pick a new folder")

X, y = load_digits(return_X_y=True)
df = pd.DataFrame(X, columns=[f"px{i}" for i in range(X.shape[1])])
df.insert(0, "label", y.astype(str))
df.insert(0, "id", [f"d{i:04d}" for i in range(len(df))])
test, train = df.iloc[:200], df.iloc[200:]

(root / "task").mkdir(parents=True)
train.to_csv(root / "task/train.csv", index=False)
test.drop(columns=["label"]).to_csv(root / "task/test.csv", index=False)
pd.DataFrame({"id": test["id"], "label": "0"}).to_csv(root / "task/sample_submission.csv", index=False)
(root / "task/SUMMARY.md").write_text("Smoke task: sklearn digits, 10 classes, 8x8 images as 64 pixel columns.\n")

shutil.copytree(KIT / "referee", root / "referee", ignore=shutil.ignore_patterns("frozen", ".lock", "contract.json"))
contract = json.loads((KIT / "referee/contract.example.json").read_text())
contract.update(task_id="smoke-digits", metric="macro_f1", exploration_deadline=None)
(root / "referee/contract.json").write_text(json.dumps(contract, indent=2))

run = root / "runs/001"
run.mkdir(parents=True)
shutil.copy(KIT / "templates/train_template.py", run / "train.py")
(run / "plan.md").write_text("parent: none\nhypothesis: baseline - standardized pixels + logistic regression\n")
print(f"smoke task ready at {root}")
