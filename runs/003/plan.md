parent: 002
hypothesis: Doubling thumbnail resolution from 24x14 to 48x28 lets the per-camera normal reference detect the small, low-contrast figures in dark views that 002 misses, raising recall without losing precision.

## Evidence (runs/002/review.md addendum)
- 002 KEEP, F1 0.7279, P 0.712, R 0.744; 896 FN.
- Most confident FN (p<0.05) are dark hallway/door views where the anomaly candidate is a small figure covering ~1-2 pixels at 24x14; confident TP are large close figures.
- Merged-cluster error rate is not worse than other clusters, so cluster quality is not the main FN source.

## Change (one factor: thumbnail resolution)
1. New label-free feature set `features: thumb_v2` (features/thumb_v2.py = copy of thumb_v1.py with SIZE=(48,28)). Same grayscale, same uint8 storage, same padding rule. Columns f{t}_p{k}, k=0..1343 -> 13440 columns + id. thumb_v1 cache is not modified.
2. runs/003/train.py = copy of runs/002/train.py (including the human's amendments from runs/002/notes.md: per-fold 50/50 seeded reference/classifier split of FALSE, KMeans k=10 + per-cluster median/MAD fit on reference half only, val/test use the fold's frozen reference). Only changes:
   - FEATURE_SETS = ["thumb_v2"], image shape 28x48.
   - The grid for "max cell mean z" becomes 8x8 cells (each cell 6x3.5 px, the same pixel size as in 002), so a cell now covers 1/64 of the frame instead of 1/16.
   - KMeans still runs on the downsampled 24x14 version of each frame (average-pool 2x2) so camera assignment is identical in method to 002; only the deviation maps use 48x28.
   All other per-frame/sequence features, MAD floor 0.02, classifier HGB(max_iter=300, lr=0.05, seed) unchanged. No internal search.
3. Unchanged: frozen 5-fold split, runner threshold selection, FEAT_WORKERS=6, seed.
4. Keep producing the per-cluster error table (errors.md) and a cluster check image for fold 0.

## Cost
- Extraction: decode dominates; 002's thumb_v1 took 130 s. Expect ~150-250 s; CSV ~4x larger (~600 MB). Engineer first runs `python features/thumb_v2.py 200`; full run under `timeout 600`. If the projection exceeds 10 min or disk/RAM is a problem, use 40x22 instead (same hypothesis, declared fallback).
- Training: 002 took 64.8 s; median/MAD and z-maps scale ~4x in pixels -> estimate 2-4 min; one-fold preflight first (time + peak memory, 002 peak was 1.68 GiB). Run the runner only if projected <= 600 s and finished before 14:30 PDT.

## Expected effect
Recall above 0.744 at similar precision; OOF F1 > 0.7279 + epsilon. Specifically fewer confident FN in dark views. If F1 does not improve, resolution is not the bottleneck and the next idea is a pretrained-embedding distance or contrast normalization for dark scenes.
