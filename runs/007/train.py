"""run 007: per-fold fine-tune ResNet18 (layer4 + linear head; up to layer3 frozen), 160x96, MIL max BCE, 1 epoch.
Sequence proba_TRUE = sigmoid(max frame logit); test = mean of fold models. Env GATE_A=N: time 1 epoch on N train seqs + inference on N, print projection, write nothing."""
import json, os, sys, time, random
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd, torch, torchvision
from PIL import Image
ROOT = Path(os.environ["TASK_ROOT"]); RUN = Path(os.environ["RUN_DIR"]); SEED = int(os.environ.get("SEED", 42))
ID, LABEL = os.environ["ID_COL"], os.environ["LABEL_COL"]; classes = json.loads(Path(os.environ["CLASSES_JSON"]).read_text())
W, H, T, BS = 160, 96, 10, 64; SPS = BS // T  # sequences per batch (6 seqs = 60 frames)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
def load_seq(d):
    out = np.empty((T, 3, H, W), np.float32)
    for t in range(T):
        with Image.open(Path(d) / f"frame_{t}.jpg") as im:
            im.draft("RGB", (2 * W, 2 * H)); x = np.asarray(im.convert("RGB").resize((W, H), Image.BILINEAR), np.float32) / 255
        out[t] = ((x - MEAN) / STD).transpose(2, 0, 1)
    return out
def net():
    torch.manual_seed(SEED); m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Linear(512, 1)
    for n, p in m.named_parameters(): p.requires_grad = n.startswith(("layer4", "fc"))
    return m.to(DEV)
def batches(dirs, pool):  # yields (index slice, decoded array) in order, decode prefetched by pool
    chunks = [list(range(i, min(i + SPS, len(dirs)))) for i in range(0, len(dirs), SPS)]
    it = pool.imap(load_seq, dirs, chunksize=2); 
    for c in chunks: yield c, np.stack([next(it) for _ in c])
def train(dirs, y, pool):
    m = net(); m.train(); [mod.eval() for n, mod in m.named_modules() if not n.startswith(("layer4", "fc")) and n]
    opt = torch.optim.AdamW([{"params": m.layer4.parameters(), "lr": 1e-4}, {"params": m.fc.parameters(), "lr": 1e-3}])
    order = np.random.RandomState(SEED).permutation(len(dirs)); dirs = [dirs[i] for i in order]; y = y[order]
    for c, X in batches(dirs, pool):
        logit = m(torch.from_numpy(X.reshape(-1, 3, H, W)).to(DEV)).view(len(c), T).max(1).values
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(y[c], dtype=torch.float32, device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
    return m
@torch.inference_mode()
def predict(m, dirs, pool):
    m.eval(); out = []
    for c, X in batches(dirs, pool): out.append(torch.sigmoid(m(torch.from_numpy(X.reshape(-1, 3, H, W)).to(DEV)).view(len(c), T).max(1).values).cpu().numpy())
    return np.concatenate(out)
if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED); pool = Pool(int(os.environ.get("FEAT_WORKERS", 6)))
    tr_df = pd.read_csv(os.environ["TRAIN_CSV"], dtype=str); split = pd.read_csv(os.environ["SPLIT_CSV"], dtype={"id": str}).rename(columns={"id": ID})
    tr_df = tr_df.merge(split, on=ID); dirs = tr_df["seq_dir"].str.strip().tolist(); y = (tr_df[LABEL] == "TRUE").to_numpy(np.float32); fold = tr_df["fold"].to_numpy()
    G = os.environ.get("GATE_A")
    if G:
        n = int(G); t = time.time(); m = train(dirs[:n], y[:n], pool); tt = time.time() - t
        t = time.time(); predict(m, dirs[n:2 * n], pool); ti = time.time() - t
        proj = 5 * (5600 * tt / n + 9400 * ti / n)
        print(f"GATE_A train {n} seqs {tt:.1f}s, infer {n} seqs {ti:.1f}s -> projection {proj:.0f}s ({proj/60:.1f} min)"); sys.exit(0)
    te = pd.read_csv(os.environ["TEST_CSV"], dtype=str); tdirs = te["seq_dir"].str.strip().tolist(); parts, tps = [], []
    ti, fi = classes.index("TRUE"), classes.index("FALSE")
    for f in sorted(set(fold[fold >= 0])):
        tr = np.where((fold != f) & (fold >= 0))[0]; va = np.where(fold == f)[0]
        m = train([dirs[i] for i in tr], y[tr], pool); p = predict(m, [dirs[i] for i in va], pool)
        P = np.zeros((len(va), len(classes))); P[:, ti] = p; P[:, fi] = 1 - p
        out = pd.DataFrame({"id": tr_df[ID].to_numpy()[va], "fold": f})
        for j, c in enumerate(classes): out[f"proba_{c}"] = P[:, j]
        parts.append(out); tps.append(predict(m, tdirs, pool)); log(f"fold {f} done")
    pd.concat(parts).to_csv(RUN / "eval_preds.csv", index=False)
    p = np.mean(tps, 0); tp = pd.DataFrame({"id": te[ID]})
    for c in classes: tp[f"proba_{c}"] = p if c == "TRUE" else 1 - p
    tp.to_csv(RUN / "test_proba.csv", index=False); log("done")
