"""Candidate template: copy to runs/NNN/train.py and edit only the two EDIT sections.

What the runner expects (it checks all of this):
  * For every fold f >= 0 in SPLIT_CSV: fit on rows with fold != f, predict rows with fold == f.
    (Rows with fold == -1 are training-only in a holdout split; they are never scored.)
  * RUN_DIR/eval_preds.csv with columns: id, fold, pred, proba_<class> for every class (class order from CLASSES_JSON).
  * Every fitted fold pipeline saved under RUN_DIR/models/.
  * If TEST_CSV is set: average the fold models' probabilities, take the argmax class, write RUN_DIR/submission.csv.
All learned preprocessing (scaling, PCA, TF-IDF, feature selection, ...) must live inside make_model() so it is fit per fold.
Features from a frozen pretrained encoder may be computed once for all rows and cached in CACHE_DIR.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RUN = Path(os.environ["RUN_DIR"])
SEED = int(os.environ.get("SEED", 42))
ID, LABEL = os.environ["ID_COL"], os.environ["LABEL_COL"]
classes = json.loads(Path(os.environ["CLASSES_JSON"]).read_text())

train = pd.read_csv(os.environ["TRAIN_CSV"], dtype={ID: str, LABEL: str})
split = pd.read_csv(os.environ["SPLIT_CSV"], dtype={"id": str}).rename(columns={"id": ID})
train = train.merge(split, on=ID, how="inner")
NON_FEATURES = {ID, LABEL, "fold", os.environ.get("GROUP_COL") or "__none__"}


# ===== EDIT 1: turn a dataframe into a feature matrix (same function for train and test) =====
def features(df):
    return df[[c for c in df.columns if c not in NON_FEATURES]].to_numpy(dtype=float)


# ===== EDIT 2: the candidate model; keep all learned preprocessing inside this pipeline =====
def make_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def proba_in_class_order(model, X):
    P = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in classes]  # fails loudly if a fold lacks a class
    return P[:, order]


X, y, fold = features(train), train[LABEL].to_numpy(), train["fold"].to_numpy()
(RUN / "models").mkdir(exist_ok=True)
parts = []
for f in sorted(set(fold[fold >= 0])):
    tr, va = fold != f, fold == f
    model = make_model().fit(X[tr], y[tr])
    joblib.dump(model, RUN / "models" / f"fold_{f}.joblib")
    P = proba_in_class_order(model, X[va])
    out = pd.DataFrame({"id": train.loc[va, ID].to_numpy(), "fold": f, "pred": np.array(classes)[P.argmax(1)]})
    for j, c in enumerate(classes):
        out[f"proba_{c}"] = P[:, j]
    parts.append(out)
    print(f"fold {f}: trained on {tr.sum()} rows, predicted {va.sum()} rows", flush=True)
pd.concat(parts).to_csv(RUN / "eval_preds.csv", index=False)

if os.environ.get("TEST_CSV"):
    test = pd.read_csv(os.environ["TEST_CSV"], dtype={ID: str})
    Xt = features(test)
    models = sorted((RUN / "models").glob("fold_*.joblib"))
    P = np.mean([proba_in_class_order(joblib.load(m), Xt) for m in models], axis=0)  # soft voting over folds
    sub = pd.DataFrame({os.environ["SUB_ID_COL"]: test[ID], os.environ["SUB_PRED_COL"]: np.array(classes)[P.argmax(1)]})
    sub.to_csv(RUN / "submission.csv", index=False)
    print(f"submission: {len(sub)} rows from {len(models)} fold models", flush=True)
