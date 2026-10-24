"""run 004: frozen ImageNet resnet18 frame embeddings (r18emb_v1) -> per-fold StandardScaler->PCA(128)->LogReg on frames
with weak sequence label; sequence proba_TRUE = max over 10 frames. Test = mean of fold models. Threshold mode.
Env PREFLIGHT_FOLDS limits folds and skips test (scratch use only)."""
import json, os, resource, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

RUN = Path(os.environ["RUN_DIR"]); ROOT = Path(os.environ["TASK_ROOT"])
SEED = int(os.environ.get("SEED", 42)); ID, LABEL = os.environ["ID_COL"], os.environ["LABEL_COL"]
classes = json.loads(Path(os.environ["CLASSES_JSON"]).read_text())
PRE = os.environ.get("PREFLIGHT_FOLDS"); T, E = 10, 512; T0 = time.time()
def mem(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
def log(m): print(f"[{time.time()-T0:7.1f}s peakRSS {mem():6.0f}MiB] {m}", flush=True)

def load(split, ids):
    cols = [f"f{t}_e{k}" for t in range(T) for k in range(E)]
    f = pd.read_csv(ROOT / "cache" / f"r18emb_v1_{split}.csv", dtype={"id": str, **{c: np.float32 for c in cols}}, engine="c")
    return f.set_index("id").loc[list(ids)][cols].to_numpy(np.float32).reshape(-1, T, E)

train = pd.read_csv(os.environ["TRAIN_CSV"], dtype={ID: str, LABEL: str}, usecols=[ID, LABEL])
split = pd.read_csv(os.environ["SPLIT_CSV"], dtype={"id": str}).rename(columns={"id": ID})
train = train.merge(split, on=ID, how="inner")
X = load("train", train[ID]); y = (train[LABEL].to_numpy() == "TRUE").astype(int); fold = train["fold"].to_numpy()
log(f"loaded train emb {X.shape}")
Xt = None
if not PRE:
    test = pd.read_csv(os.environ["TEST_CSV"], dtype={ID: str}, usecols=[ID]); Xt = load("test", test[ID]); log(f"loaded test emb {Xt.shape}")
folds = sorted(set(fold[fold >= 0])) if not PRE else [int(v) for v in PRE.split(",")]
ti, fi = classes.index("TRUE"), classes.index("FALSE")
def seqp(m, A): return m.predict_proba(A.reshape(-1, E))[:, 1].reshape(-1, T).max(1)
parts, tps = [], []
for f in folds:
    tr = np.where((fold != f) & (fold >= 0))[0]; va = np.where(fold == f)[0]; t = time.time()
    m = make_pipeline(StandardScaler(), PCA(128, random_state=SEED), LogisticRegression(C=1.0, max_iter=2000))
    m.fit(X[tr].reshape(-1, E), np.repeat(y[tr], T)); p = seqp(m, X[va])
    out = pd.DataFrame({"id": train[ID].to_numpy()[va], "fold": f}); P = np.zeros((len(va), len(classes)))
    P[:, ti] = p; P[:, fi] = 1 - p
    for j, c in enumerate(classes): out[f"proba_{c}"] = P[:, j]
    parts.append(out); log(f"fold {f}: fit {len(tr)*T} frames, predict {len(va)} seqs {time.time()-t:.1f}s")
    if Xt is not None: tps.append(seqp(m, Xt))
pd.concat(parts).to_csv(RUN / "eval_preds.csv", index=False)
if Xt is not None:
    p = np.mean(tps, 0); tp = pd.DataFrame({"id": test[ID]})
    for c in classes: tp[f"proba_{c}"] = p if c == "TRUE" else 1 - p
    tp.to_csv(RUN / "test_proba.csv", index=False); print(f"test_proba: {len(tp)} rows", flush=True)
log("done")
