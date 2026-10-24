parent: 002
hypothesis: Anomalies are semantic (an animatronic doing something it shouldn't), so frozen pretrained ImageNet CNN embeddings per frame plus a supervised frame scorer max-pooled over the sequence will separate classes far better than pixel deviation from a background reference.

## Strategic review (why PIVOT, not another local step)
Observed:
- 001-003 all share one assumption: anomaly = pixel deviation (from sequence median, then from a per-cluster median). 002 gave the only real gain (AUC ~0.80); 003 (4x pixels) left AUC unchanged (0.7977 vs 0.7983) and only moved along the PR curve.
- Competition README: "A sequence is anomalous if an animatronic does something it shouldn't in any of its frames"; metric = F1 of the anomalous class, same as ours (so the scale is comparable, but our number is local OOF, not an official score; we have no official score yet).
- The official starter baseline already uses a pretrained ResNet18 per frame with max over the sequence; leaderboard top scores are ~0.96-0.99 (monitor, unverified by Critic). Used only as a signal that the semantic approach class exists, not for selection.
- Our features keep only grayscale low-res deviation magnitude; they discard color, shape and identity of the figure. Animatronics appear in both classes in normal positions (seen in 001/002 image sheets), so "something is there / differs" cannot tell normal from anomalous.
- Labels are only used as sequence-level max; 002/003 never learned what an anomalous frame looks like.
Hypothesis-level conclusion: the representation is the bottleneck, not resolution, reference or threshold.

## Change
1. New label-free feature set `features: r18emb_v1` (features/r18emb_v1.py): torchvision resnet18 ImageNet weights, frozen, eval mode, on MPS if available else CPU. Each frame RGB resized to 224x128 (keep aspect), ImageNet normalization, global-average-pooled 512-d embedding, stored as float16-rounded values (4 decimals). Columns f{t}_e{k}, t=0..9, k=0..511 (5120 cols). Batching over frames; FEAT_WORKERS=6 only for JPEG decoding DataLoader workers; one model process.
   This is a frozen pretrained encoder: allowed to be cached once (PROTOCOL).
2. runs/004/train.py (copy of 002 structure, replaces its features; NO background reference, NO KMeans):
   per fold f, fit on fold != f only:
   - Frame-level training set: every frame of every training sequence, label = sequence label (multiple-instance weak label).
   - Pipeline: StandardScaler -> PCA(128, random_state=SEED) -> LogisticRegression(C=1.0, max_iter=2000), fit on fold-train frames.
   - Sequence proba_TRUE = max over the 10 frames of the frame probability (no second-stage model).
   - Test: average of fold models. Threshold left to the runner.
   No internal search.
3. Unchanged: frozen split, runner, epsilon, deadline, FEAT_WORKERS=6.

## Cost and gates (deadline 15:00 PDT; leave runner run done by 14:55)
- Gate 0 (Engineer): torch/torchvision importable in a venv on the Mac (install CPU/MPS wheels if missing; weights download ~45 MB); report memory_pressure. If not ready by 14:35 -> abandon 004, HOLD.
- Gate 1: `python features/r18emb_v1.py 200` (2000 frames). Projection for 150k frames must be <= 12 min, else reduce to 160x96; if still > 12 min -> abandon.
- Full extraction under timeout; cache ~150k x 512 -> ~0.6-1 GB CSV (if write is slow, store float with 3 decimals).
- Training: PCA + LR on ~56k x 5120->128 per fold, est. 1-3 min total; peak RAM ~3-4 GB. One-fold preflight first.
If any gate fails: do not run partial folds; record and HOLD with 002 as best.

## Success / failure
- Success: OOF AUC clearly > 0.80 and F1 > 0.7279 + 0.002 with flagged share not exceeding ~60% (real separation, not threshold movement).
- Failure: AUC <= 0.80 -> frozen ImageNet features don't capture the anomaly; next phase would need fine-tuning (not feasible today).
