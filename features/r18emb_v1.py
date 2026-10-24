"""Label-free frozen ImageNet resnet18 per-frame embeddings (runs/004/plan.md).
  python features/r18emb_v1.py 200   # timing on first 200 train sequences, no cache written
  python features/r18emb_v1.py       # full: cache/r18emb_v1_{train,test}.csv, meta.json last
Columns f{t}_e{k}, t=0..9, k=0..511. FEAT_WORKERS = DataLoader decode workers; one model process.
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torchvision
from PIL import Image
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent.parent
NAME = Path(__file__).stem
SIZE = tuple(int(v) for v in os.environ.get("R18_SIZE", "224x128").split("x"))   # (w, h); 160x96 = declared fallback
WORKERS = int(os.environ.get("FEAT_WORKERS", 6)); T = 10; BS = 256
HALF = os.environ.get("R18_FP16", "1") == "1"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def load_seq(d):
    out = np.empty((T, 3, SIZE[1], SIZE[0]), np.float32)
    for t in range(T):
        with Image.open(Path(d) / f"frame_{t}.jpg") as im:
            im.draft("RGB", (2 * SIZE[0], 2 * SIZE[1]))
            x = np.asarray(im.convert("RGB").resize(SIZE, Image.BILINEAR), np.float32) / 255
        out[t] = ((x - MEAN) / STD).transpose(2, 0, 1)
    return out

def model():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity(); m = m.eval().to(DEV)
    return m.half() if HALF else m

def decode(dirs, pool):
    return np.stack(pool.map(load_seq, list(dirs), chunksize=8)).reshape(-1, 3, SIZE[1], SIZE[0])

def embed(X, m):
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), BS):
            b = torch.from_numpy(X[i:i + BS]).to(DEV)
            if HALF: b = b.half()
            assert b.device.type == DEV
            out.append(m(b).float().cpu().numpy())
    return np.concatenate(out).reshape(-1, T * 512)

def extract(dirs, m, pool, chunk=500):
    dirs = list(dirs); res = []
    for i in range(0, len(dirs), chunk):
        res.append(embed(decode(dirs[i:i + chunk], pool), m))
    return np.concatenate(res)

if __name__ == "__main__":
    torch.set_num_threads(max(1, (os.cpu_count() or 8) - WORKERS))
    m = model(); cache = ROOT / "cache"; pool = Pool(WORKERS)
    if len(sys.argv) > 1:
        n = int(sys.argv[1]); dirs = pd.read_csv(ROOT / "task" / "train.csv", usecols=["seq_dir"])["seq_dir"].head(n)
        extract(dirs.head(8), m, pool)                      # warm-up (MPS kernels) not timed
        t0 = time.time(); X = decode(dirs, pool); td = time.time() - t0
        t1 = time.time(); E = embed(X, m); tm = time.time() - t1; dt = td + tm
        print(f"decode {td:.1f}s model {tm:.1f}s fp16={HALF}")
        print(f"{n} sequences ({n*T} frames) in {dt:.1f}s on {DEV} size {SIZE} workers {WORKERS} -> projected "
              f"{dt / n * 15000 / 60:.1f} min for 15000 sequences; emb shape {E.shape}, finite {np.isfinite(E).all()}")
        sys.exit(0)
    marker = cache / f"{NAME}.meta.json"
    if marker.exists(): sys.exit(f"feature set {NAME} is complete: never overwrite")
    targets = [cache / f"{NAME}_{s}.csv" for s in ("train", "test")]
    for t in targets:
        if t.exists(): print(f"discarding incomplete {t.name}", flush=True); t.unlink()
    meta = {"name": NAME, "workers": WORKERS, "size": SIZE, "device": DEV, "fp16": HALF, "batch": BS, "torch": torch.__version__,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "seconds": {}}
    cols = [f"f{t}_e{k}" for t in range(T) for k in range(512)]
    for split, target in zip(("train", "test"), targets):
        df = pd.read_csv(ROOT / "task" / f"{split}.csv", usecols=["sequence_id", "seq_dir"])
        t0 = time.time(); E = extract(df["seq_dir"], m, pool); te = time.time() - t0
        out = pd.DataFrame(np.round(E, 4).astype(np.float32), columns=cols); out.insert(0, "id", df["sequence_id"].astype(str))
        out.to_csv(str(target) + ".tmp", index=False, float_format="%.4f"); os.replace(str(target) + ".tmp", target)
        meta["seconds"][split] = round(time.time() - t0, 1)
        print(f"{NAME}_{split}: {len(out)} rows x {len(cols)} in {meta['seconds'][split]}s (embed {te:.1f}s)", flush=True)
    marker.write_text(json.dumps(meta, indent=2))
