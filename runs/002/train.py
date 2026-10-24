"""run 002: per-camera (KMeans cluster) normal reference from thumb_v1, with reference/classifier separation (see notes.md).
Writes eval_preds.csv / test_proba.csv (threshold mode). Env PREFLIGHT_FOLDS=0 limits folds and skips test (scratch use only)."""
import json, os, resource, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier

RUN = Path(os.environ["RUN_DIR"]); ROOT = Path(os.environ["TASK_ROOT"])
SEED = int(os.environ.get("SEED", 42)); ID, LABEL = os.environ["ID_COL"], os.environ["LABEL_COL"]
classes = json.loads(Path(os.environ["CLASSES_JSON"]).read_text())
PRE = os.environ.get("PREFLIGHT_FOLDS")
K, H, W, T = 10, 14, 24, 10
T0 = time.time()

def mem():  # peak RSS of this process in MiB (macOS ru_maxrss is bytes)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

def log(msg):
    print(f"[{time.time()-T0:7.1f}s peakRSS {mem():6.0f}MiB] {msg}", flush=True)

def load(split, ids):
    f = pd.read_csv(ROOT / "cache" / f"thumb_v1_{split}.csv", dtype={"id": str}, engine="c")
    f = f.set_index("id").loc[list(ids)]
    cols = [f"f{t}_p{k}" for t in range(T) for k in range(H * W)]
    return f[cols].to_numpy(dtype=np.float32).reshape(len(f), T, H * W) / 255.0

rows = np.array_split(np.arange(H), 4); colsg = np.array_split(np.arange(W), 4)

def seq_features(X, km, med, mad):
    n = len(X); fr = X.reshape(n * T, H * W)
    c = km.predict(fr); dist = np.sqrt(((fr - km.cluster_centers_[c]) ** 2).sum(1))
    z = np.abs(fr - med[c]) / mad[c]
    zi = z.reshape(-1, H, W)
    grid = np.max([zi[:, r][:, :, cc].mean(axis=(1, 2)) for r in rows for cc in colsg], axis=0)
    pf = np.stack([z.mean(1), np.percentile(z, 95, axis=1), grid, (z > 3).mean(1), dist], 1).reshape(n, T, 5)
    ncam = np.array([len(set(r)) for r in c.reshape(n, T)])
    F = np.concatenate([pf.max(1), pf.mean(1), ncam[:, None]], 1)
    return F, c.reshape(n, T)

train = pd.read_csv(os.environ["TRAIN_CSV"], dtype={ID: str, LABEL: str}, usecols=[ID, LABEL])
split = pd.read_csv(os.environ["SPLIT_CSV"], dtype={"id": str}).rename(columns={"id": ID})
train = train.merge(split, on=ID, how="inner")
X = load("train", train[ID]); y = train[LABEL].to_numpy(); fold = train["fold"].to_numpy()
log(f"loaded train thumbs {X.shape}")
Xt = None
if not PRE:
    test = pd.read_csv(os.environ["TEST_CSV"], dtype={ID: str}, usecols=[ID])
    Xt = load("test", test[ID]); log(f"loaded test thumbs {Xt.shape}")

folds = sorted(set(fold[fold >= 0])) if not PRE else [int(v) for v in PRE.split(",")]
parts, tps, clus = [], [], []
rng = np.random.RandomState(SEED)
for f in folds:
    tr = np.where(fold != f)[0]; va = np.where(fold == f)[0]
    fal = tr[y[tr] == "FALSE"].copy(); np.random.RandomState(SEED + f).shuffle(fal)
    ref = np.sort(fal[: len(fal) // 2]); clf_idx = np.setdiff1d(tr, ref)
    t = time.time(); R = X[ref].reshape(-1, H * W)
    km = KMeans(n_clusters=K, n_init=5, random_state=SEED).fit(R); log(f"fold {f}: KMeans on {len(R)} ref frames ({len(ref)} FALSE seqs) {time.time()-t:.1f}s")
    t = time.time(); lab = km.labels_
    med = np.stack([np.median(R[lab == k], 0) for k in range(K)])
    mad = np.stack([np.median(np.abs(R[lab == k] - med[k]), 0) for k in range(K)]) + 0.02
    sizes = np.bincount(lab, minlength=K).tolist(); log(f"fold {f}: reference median/MAD {time.time()-t:.1f}s cluster sizes {sizes}")
    if f == folds[0]:
        np.save(RUN / "fold0_ref_frames.npy", (R * 255).astype(np.uint8)); np.save(RUN / "fold0_ref_labels.npy", lab)
    t = time.time(); Ftr, _ = seq_features(X[clf_idx], km, med, mad); Fva, cva = seq_features(X[va], km, med, mad)
    log(f"fold {f}: features train {len(clf_idx)} / val {len(va)} {time.time()-t:.1f}s")
    t = time.time(); model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED).fit(Ftr, y[clf_idx])
    idx = [list(model.classes_).index(c) for c in classes]
    P = model.predict_proba(Fva)[:, idx]; log(f"fold {f}: classifier fit+predict {time.time()-t:.1f}s")
    out = pd.DataFrame({"id": train[ID].to_numpy()[va], "fold": f})
    for j, c in enumerate(classes): out[f"proba_{c}"] = P[:, j]
    parts.append(out)
    clus.append(pd.DataFrame({"id": out["id"], "fold": f, "main_cluster": [np.bincount(r).argmax() for r in cva],
                              "n_cams": [len(set(r)) for r in cva]}))
    if Xt is not None:
        t = time.time(); Ft, _ = seq_features(Xt, km, med, mad); tps.append(model.predict_proba(Ft)[:, idx]); log(f"fold {f}: test features+predict {time.time()-t:.1f}s")
    print(f"fold {f}: ref {len(ref)} / fit {len(clf_idx)} / predict {len(va)}", flush=True)
pd.concat(parts).to_csv(RUN / "eval_preds.csv", index=False)
pd.concat(clus).to_csv(RUN / "val_clusters.csv", index=False)
if Xt is not None:
    P = np.mean(tps, 0); tp = pd.DataFrame({"id": test[ID]})
    for j, c in enumerate(classes): tp[f"proba_{c}"] = P[:, j]
    tp.to_csv(RUN / "test_proba.csv", index=False); print(f"test_proba: {len(tp)} rows", flush=True)
log("done")
