# Five Nights with AI: Track 1 submission

Anomaly detection on 10-frame security-camera sequences from ACM AI's Fall 2026 competition *Five Nights with AI* (UCSD).
A sequence is `TRUE` if any of its frames is anomalous. The metric is F1 on the `TRUE` class.

**Final model: run 006. Out-of-fold F1 = 0.8865 (OOF AUC 0.960). Official leaderboard: 0.8899.**
The final upload is [`best/submission.csv`](best/submission.csv).

## Approach (final pipeline)
1. **Frozen ResNet18 frame embeddings** ([`features/r18emb_v1.py`](features/r18emb_v1.py)): ImageNet ResNet18, 224x128 input, a 512-d embedding per frame. No labels are used.
2. **Camera-aware "distance from normal"** ([`runs/006/train.py`](runs/006/train.py)): scaler + PCA(128) inside each fold. Half of the normal (`FALSE`) training sequences become a reference set. For every frame we add its k=5 nearest-neighbour offset and distance to normal frames, which works as a per-camera normal model.
3. **Nonlinear frame scorer**: HistGradientBoosting on 257 features per frame. A sequence's score is the max over its frames.
4. **Threshold** chosen on pooled out-of-fold probabilities (0.83), then applied to the mean of the 5 fold models on test.

## Experiment log
Every run is a folder under [`runs/`](runs/) with `plan.md` (hypothesis, written before any code), `train.py`, `notes.md`, `review.md` and the runner output. The full table is in [`report.md`](report.md).

| run | change | OOF F1 | decision |
|---|---|---|---|
| 001 | median-deviation pixel stats + HGB | 0.6742 | keep |
| 002 | per-camera normal pixel reference | 0.7279 | keep |
| 003 | higher-resolution thumbnails | 0.7283 | reject |
| 004 | pivot: frozen ResNet18 embeddings + LR | 0.8526 | keep |
| 005 | + kNN offset from normal frames | 0.8731 | keep |
| 006 | LR -> HistGradientBoosting | **0.8865** | **final** |
| 007 | fine-tune layer4 | - | not run (time budget) |

(Predicting `TRUE` for every sequence gives F1 = 0.667.)

## How it was built
An Engineer + Critic agent workflow with a referee ([`PROTOCOL.md`](PROTOCOL.md), [`bots/`](bots/)).
[`referee/runner.py`](referee/runner.py) is the only component that scores runs. It enforces a frozen 5-fold split ([`referee/frozen/split.csv`](referee/frozen/split.csv)), a per-run time budget and a deadline, and it alone decides KEEP/REJECT, writing [`best/`](best/) and [`events.jsonl`](events.jsonl). The Chinese guide is [`README_zh.md`](README_zh.md).

## Reproduce
Requires the official starter kit and data: https://github.com/acmucsd/Five-Nights-With-AI-Starter-Kit

```bash
python3 -m venv venv && venv/bin/pip install numpy pandas pillow scikit-learn joblib matplotlib torch torchvision
# starter kit at ../fnaf with the data in ../fnaf/data/{train,test}
venv/bin/python fnaf/make_task.py               # writes task/*.csv (labels are not committed)
venv/bin/python features/r18emb_v1.py           # caches embeddings in cache/ (not committed, ~1 GB)
venv/bin/python referee/runner.py run runs/006
```
Scripts reference absolute paths from the original machine (`/Users/admin/workspace/...`). Adjust them if your layout differs.
