"""Candidate on cached features: copy to runs/NNN/train.py, edit FEATURE_SETS and make_model().

Writes what the runner expects in threshold mode:
  eval_preds.csv  id, fold, proba_FALSE, proba_TRUE   (out-of-fold: fit on fold != f, predict fold == f)
  test_proba.csv  id, proba_FALSE, proba_TRUE         (average of the fold models)
The runner picks the threshold and writes submission.csv itself; do not threshold here.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RUN = Path(os.environ["RUN_DIR"])
ROOT = Path(os.environ["TASK_ROOT"])
SEED = int(os.environ.get("SEED", 42))
ID, LABEL = os.environ["ID_COL"], os.environ["LABEL_COL"]
classes = json.loads(Path(os.environ["CLASSES_JSON"]).read_text())

FEATURE_SETS = ["diffmed_v1"]          # ===== EDIT: cache/<name>_{train,test}.csv to use


def load(split, ids):
    out = pd.DataFrame({"id": ids})
    for name in FEATURE_SETS:
        f = pd.read_csv(ROOT / "cache" / f"{name}_{split}.csv", dtype={"id": str})
        f = f.rename(columns={c: f"{name}:{c}" for c in f.columns if c != "id"})
        out = out.merge(f, on="id", how="left")
    if out.isna().any().any():
        raise SystemExit(f"missing cached features for some {split} ids")
    return out.drop(columns="id").to_numpy(dtype=np.float32)


def make_model():                       # ===== EDIT: model; learned preprocessing goes inside this pipeline =====
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)


def proba_in_class_order(model, X):
    P = model.predict_proba(X)
    return P[:, [list(model.classes_).index(c) for c in classes]]


train = pd.read_csv(os.environ["TRAIN_CSV"], dtype={ID: str, LABEL: str}, usecols=[ID, LABEL])
split = pd.read_csv(os.environ["SPLIT_CSV"], dtype={"id": str}).rename(columns={"id": ID})
train = train.merge(split, on=ID, how="inner")
X, y, fold = load("train", train[ID]), train[LABEL].to_numpy(), train["fold"].to_numpy()

(RUN / "models").mkdir(exist_ok=True)
parts = []
for f in sorted(set(fold[fold >= 0])):
    tr, va = fold != f, fold == f
    model = make_model().fit(X[tr], y[tr])
    joblib.dump(model, RUN / "models" / f"fold_{f}.joblib")
    P = proba_in_class_order(model, X[va])
    out = pd.DataFrame({"id": train.loc[va, ID].to_numpy(), "fold": f})
    for j, c in enumerate(classes):
        out[f"proba_{c}"] = P[:, j]
    parts.append(out)
    print(f"fold {f}: fit {tr.sum()} / predict {va.sum()}", flush=True)
pd.concat(parts).to_csv(RUN / "eval_preds.csv", index=False)

test = pd.read_csv(os.environ["TEST_CSV"], dtype={ID: str}, usecols=[ID])
Xt = load("test", test[ID])
P = np.mean([proba_in_class_order(joblib.load(m), Xt) for m in sorted((RUN / "models").glob("fold_*.joblib"))], axis=0)
tp = pd.DataFrame({"id": test[ID]})
for j, c in enumerate(classes):
    tp[f"proba_{c}"] = P[:, j]
tp.to_csv(RUN / "test_proba.csv", index=False)
print(f"test_proba: {len(tp)} rows", flush=True)
