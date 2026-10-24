"""Label-free feature extraction, run once per feature set OUTSIDE the runner.

  python features/<name>.py 200    # 1) time it on 200 sequences, prints the projected full time
  python features/<name>.py        # 2) full run: train + test

Contract:
  * Reads only sequence_id + seq_dir from task/train.csv and task/test.csv. Never reads labels.
  * Writes cache/<name>_train.csv and cache/<name>_test.csv (column `id` + numeric features) and
    cache/<name>.meta.json (time taken, workers, script hash) LAST, as the completion marker.
    A complete feature set (meta.json present) is never overwritten: a new idea = a new file name.
    An incomplete one (csv without meta.json, e.g. a crash or timeout) is discarded and redone on the next run.
  * Same compute() for train and test. Anything fitted on data (scalers, PCA, per-camera references learned
    from labeled data, ...) belongs in train.py inside the per-fold pipeline, not here.
  * Workers: FEAT_WORKERS env (default 6 of 8 cores; free RAM is limited). If compute() uses torch, call
    torch.set_num_threads(1) inside the worker.
The example below is deliberately simple (how much each frame differs from the sequence's median frame).
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")             # one thread per worker process; parallelism comes from the Pool
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent          # /workspace/comp
NAME = Path(__file__).stem                             # feature-set name = file name
SIZE = (112, 64)                                       # (w, h) after resize; small = fast
WORKERS = int(os.environ.get("FEAT_WORKERS", min(6, os.cpu_count() or 1)))


def load_frames(seq_dir, mode="L"):
    files = sorted(Path(seq_dir).glob("frame_*"), key=lambda p: int(p.stem.split("_")[1]))
    frames = []
    for f in files:
        with Image.open(f) as im:
            im.draft(mode, (SIZE[0] * 2, SIZE[1] * 2))   # JPEG: decode at reduced scale directly (much faster)
            frames.append(np.asarray(im.convert(mode).resize(SIZE), dtype=np.float32) / 255)
    return np.stack(frames)                             # (T, H, W) for "L", (T, H, W, 3) for "RGB"


# ===== EDIT: one sequence directory -> dict of numeric features =====
def compute(seq_dir):
    x = load_frames(seq_dir)                           # (T, H, W) grayscale in [0, 1]
    med = np.median(x, axis=0)
    d = np.abs(x - med)                                # per-frame deviation from the sequence's median frame
    per_frame_mean = d.mean(axis=(1, 2))
    per_frame_big = (d > 0.15).mean(axis=(1, 2))
    step = np.abs(np.diff(x, axis=0)).mean(axis=(1, 2))
    return {"dev_mean_max": per_frame_mean.max(), "dev_mean_mean": per_frame_mean.mean(),
            "dev_big_max": per_frame_big.max(), "dev_big_mean": per_frame_big.mean(),
            "step_max": step.max(), "step_mean": step.mean(), "brightness": x.mean()}


def extract(dirs):
    dirs = list(dirs)
    chunk = max(1, min(16, len(dirs) // (WORKERS * 4)))  # small inputs still keep every worker busy
    with Pool(WORKERS) as pool:
        return pool.map(compute, dirs, chunksize=chunk)


if __name__ == "__main__":
    cache = ROOT / "cache"
    if len(sys.argv) > 1:                              # timing mode
        n = int(sys.argv[1])
        dirs = pd.read_csv(ROOT / "task" / "train.csv", usecols=["seq_dir"])["seq_dir"].head(n)
        t0 = time.time()
        extract(dirs)
        dt = time.time() - t0
        total = sum(len(pd.read_csv(ROOT / "task" / f"{s}.csv", usecols=["sequence_id"])) for s in ("train", "test"))
        print(f"{n} sequences in {dt:.1f}s with {WORKERS} workers -> projected {dt / n * total / 60:.1f} min "
              f"for all {total} sequences")
        sys.exit(0)

    targets = [cache / f"{NAME}_{s}.csv" for s in ("train", "test")]
    marker = cache / f"{NAME}.meta.json"
    if marker.exists():
        sys.exit(f"feature set {NAME} is complete (cache/{NAME}.meta.json): never overwrite it, use a new file name")
    for t in targets:
        if t.exists():
            print(f"discarding incomplete {t.name} from an unfinished earlier run", flush=True)
            t.unlink()
    cache.mkdir(exist_ok=True)
    meta = {"name": NAME, "workers": WORKERS, "size": SIZE,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "seconds": {}}
    for split, target in zip(("train", "test"), targets):
        df = pd.read_csv(ROOT / "task" / f"{split}.csv", usecols=["sequence_id", "seq_dir"])
        t0 = time.time()
        out = pd.DataFrame(extract(df["seq_dir"]))
        out.insert(0, "id", df["sequence_id"].astype(str))
        out.to_csv(str(target) + ".tmp", index=False)
        os.replace(str(target) + ".tmp", target)
        meta["seconds"][split] = round(time.time() - t0, 1)
        print(f"{NAME}_{split}: {len(out)} rows x {out.shape[1] - 1} features in {meta['seconds'][split]}s", flush=True)
    marker.write_text(json.dumps(meta, indent=2))   # written last = feature set complete
