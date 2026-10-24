# review 001 (Critic)

Hypothesis (baseline diffmed_v1 + HGB > 0.667): formally supported (OOF F1 0.6742, KEEP), but only +0.008 over all-TRUE. At threshold 0.15 the model flags 96.1% (TP 3447, FP 3278, FN 53, TN 222); at 0.5 accuracy is 0.590. It is essentially an all-TRUE predictor.

## Observed evidence (files / images)
- Per-class medians of all 7 diffmed_v1 features are nearly identical (e.g. dev_mean_max 0.0631 vs 0.0631, step_max 0.0876 vs 0.0870, dev_big_max 0.0961 vs 0.0963). Only brightness differs slightly (0.1164 FALSE vs 0.1087 TRUE).
- Opened frames of 8 sequences (2 confident FP, 2 confident FN, 2 confident TP, 2 confident TN; per-sequence frame strips, ids not recorded in plans). In most of them the 10 frames come from DIFFERENT cameras: the view switches within a sequence (stage, dining hall, curtain, hallway, office door...). So the sequence's median frame is a blend of unrelated scenes, and |frame - median| and frame-to-frame steps mostly measure camera switches, in both classes.
- Visual content that could be anomalous (animatronic figures at varying positions/sizes, e.g. a close-up purple figure) appears in both FP and TP rows; nothing in the global features separates them.
- Rough camera clustering of first frames (k-means on 16x9 thumbnails, 1500 train seqs, exploratory only) shows no single cluster dominated by one class (positive rate 0.35-0.59).

## Not yet verified
- That an anomaly = a frame that deviates from that camera's normal appearance (requires a per-camera normal reference; not tested).
- Whether anomalies are local (small region) vs whole-frame.

## Judgement
The main failure is not dilution by global statistics but a wrong reference: "the sequence's own median" is meaningless when frames come from mixed cameras (and would also fail for persistent anomalies). Next step: compare every frame to a NORMAL reference of its own camera, learned per fold from training FALSE sequences only. Proposed as 002.
