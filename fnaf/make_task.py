"""Convert the starter-kit data layout into the runner's task files, with completeness checks.

  python fnaf/make_task.py /workspace/fnaf/data /workspace/comp/task            # expects 7000 train / 8000 test
  python fnaf/make_task.py DATA OUT --expect-train 500 --expect-test 120         # other sizes (tests)

Checks (fails loudly): label ids == train folder ids, expected sequence counts, frame_0..frame_9 in every
sequence, the first frame of every sequence opens as an image.
Writes task/train.csv (sequence_id, has_anomaly TRUE/FALSE, seq_dir), task/test.csv (sequence_id, seq_dir),
task/sample_submission.csv (every test id, FALSE).
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--expect-train", type=int, default=7000)
ap.add_argument("--expect-test", type=int, default=8000)
ap.add_argument("--frames", type=int, default=10)
a = ap.parse_args()
data, out = Path(a.data).resolve(), Path(a.out).resolve()
TRUE, FALSE = {"true", "t", "yes", "y", "1"}, {"false", "f", "no", "n", "0"}
problems = []


def seq_dirs(split):
    return sorted(p for p in (data / split).iterdir() if p.is_dir())


labels = {}
with open(data / "train" / "labels.csv", newline="") as fh:
    for row in csv.DictReader(fh):
        sid, v = row["sequence_id"].strip(), row["has_anomaly"].strip().lower()
        if sid in labels:
            problems.append(f"duplicate label row {sid}")
        labels[sid] = "TRUE" if v in TRUE else "FALSE" if v in FALSE else None
problems += [f"unreadable label for {k}" for k, v in labels.items() if v is None][:5]

train_dirs, test_dirs = seq_dirs("train"), seq_dirs("test")
dir_ids = {d.name for d in train_dirs}
if dir_ids != set(labels):
    problems.append(f"labels without folder: {len(set(labels) - dir_ids)}, folders without label: "
                    f"{len(dir_ids - set(labels))} (incomplete download/unzip?)")
if len(train_dirs) != a.expect_train:
    problems.append(f"train has {len(train_dirs)} sequences, expected {a.expect_train}")
if len(test_dirs) != a.expect_test:
    problems.append(f"test has {len(test_dirs)} sequences, expected {a.expect_test}")
want = {f"frame_{i}" for i in range(a.frames)}
sizes = Counter()
for d in train_dirs + test_dirs:
    stems = {p.stem for p in d.iterdir() if p.suffix.lower() in (".jpg", ".png")}
    if stems != want:
        problems.append(f"{d.name}: frames {sorted(stems)[:3]}... != frame_0..frame_{a.frames - 1}")
        continue
    first = next(p for p in d.iterdir() if p.stem == "frame_0")
    try:
        with Image.open(first) as im:
            sizes[im.size] += 1
    except Exception as e:  # noqa: BLE001
        problems.append(f"{d.name}: cannot open {first.name}: {e}")
if problems:
    print("DATA CHECK FAILED:")
    for p in problems[:20]:
        print("  -", p)
    sys.exit(1)

out.mkdir(parents=True, exist_ok=True)
with open(out / "train.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["sequence_id", "has_anomaly", "seq_dir"])
    for d in train_dirs:
        w.writerow([d.name, labels[d.name], str(d)])
with open(out / "test.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["sequence_id", "seq_dir"])
    for d in test_dirs:
        w.writerow([d.name, str(d)])
with open(out / "sample_submission.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["sequence_id", "has_anomaly"])
    for d in test_dirs:
        w.writerow([d.name, "FALSE"])

n_pos = sum(labels[d.name] == "TRUE" for d in train_dirs)
prev = n_pos / len(train_dirs)
print(f"DATA CHECK OK. train: {len(train_dirs)} sequences, {n_pos} anomalous ({prev:.1%}); test: {len(test_dirs)}")
print(f"frame sizes (w, h) of frame_0: {dict(sizes)}")
print(f"all-TRUE F1 baseline = {2 * prev / (1 + prev):.3f}  (any real model must beat this)")
print(f"wrote {out}/train.csv, test.csv, sample_submission.csv")
