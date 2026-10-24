parent: 001
hypothesis: Comparing each frame to a per-camera normal reference (learned per fold from training FALSE sequences) instead of the sequence's own median will separate anomalous sequences, because frames within a sequence come from different cameras.

## Evidence (runs/001/review.md)
- diffmed_v1 features have nearly identical per-class medians; model flags 96% (F1 0.674 vs all-TRUE 0.667).
- Inspected frames: sequences mix several camera views, so the sequence median is a blend of unrelated scenes and diff/step features measure camera switches in both classes.

## Change (one main factor: the reference image)
1. New label-free feature set `features: thumb_v1` (features/thumb_v1.py, from the template / diffmed_v1 loader):
   per frame t=0..9, grayscale, resized to 24x14 (w x h), stored as uint8 0-255 integers.
   Columns: f{t}_p{k}, k=0..335  -> 3360 columns + id. No statistics computed, no labels read.
   If a sequence has != 10 frames: pad by repeating the last frame and add column n_frames.
2. runs/002/train.py (copy of 001) – all learned steps inside the fold loop, fit on fold != f only:
   a. Reference frames = all frames of fold-train sequences with label FALSE (all their frames are normal).
   b. Camera assignment: KMeans(n_clusters=10, n_init=5, random_state=SEED) on those reference frame thumbnails (pixels/255). Fixed k=10 = number of cameras in the task description; no search.
   c. Per-camera normal reference: pixel-wise median and MAD (+0.02 floor) of reference frames in each cluster.
   d. For every frame of fold-train, fold-val and test sequences: assign nearest centroid; z = |x - median_c| / MAD_c; per-frame features: mean z, 95th pct z, max over 4x4 grid cells (6x3.5 px blocks) of mean z, fraction z>3, distance to centroid.
   e. Sequence features: max and mean over the 10 frames of each per-frame feature (10 features) + number of distinct cameras in the sequence.
   f. Classifier unchanged: HistGradientBoostingClassifier(max_iter=300, lr=0.05, random_state=SEED). diffmed_v1 NOT added (keep the single change clean).
   Test probabilities: each fold model is applied to test features computed with that fold's own references, then averaged (as in 001).
   Note: fold-train sequences also get features from references that include themselves (FALSE ones); acceptable bias, equal for all folds; val/test never in the reference.
3. Unchanged: frozen 5-fold split (SPLIT_CSV), runner threshold selection, output files, FEAT_WORKERS=6, seed.

## Cost
- Extraction: decode only (same draft-JPEG loader as diffmed_v1, smaller size) -> expected <= diffmed_v1 time. Engineer first runs `python features/thumb_v1.py 200`; full run under `timeout` with limit = min(projection x 1.5, 30 min). Cache ~ 15000 x 3360 small ints (~150 MB csv).
- Training: per fold KMeans on ~28k frames x 336 dims + median/MAD + HGB on 11 features: est. 1-3 min total, well under 900 s. If close to the limit: subsample reference frames to 10k per fold (seeded); this is not a hypothesis change.
- If extraction projection does not fit before 14:15 PDT, reduce to 16x9 thumbnails.

## Expected effect
Clear separation beyond all-TRUE: OOF F1 > 0.70, predicted-positive share well below 96%. If F1 stays ~0.67, the "anomaly = deviation from camera normal" assumption is wrong at this resolution and next step is pretrained visual features.
