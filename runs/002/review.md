# review 002 (Critic)

Hypothesis (per-camera normal reference instead of sequence median): SUPPORTED. OOF F1 0.7279 vs 0.6742 (001), KEEP, threshold 0.63 (result.json).
- Real discrimination now: precision 0.712 vs 0.513, flagged 52.2% vs 96.1%, F1 at 0.5 = 0.725 (001: 0.584). FP on FALSE fell from 3278/3500 to 1051/3500.
- Cost: recall 0.985 -> 0.744; 896 FN. Fold F1 0.706-0.761 (std 0.024), all above all-TRUE 0.667.
- Implemented with the human's amendment (50/50 reference/classifier split of FALSE per fold, notes.md); classifier saw only half the FALSE data.

## Remaining errors (Engineer report + errors.md; not re-derived by Critic)
- Per-cluster error 15-41%, no single cluster dominates. Cluster ~ camera only roughly: each fold has one ~2.7k-frame cluster that merges two scenes (fold0 c3), so its reference median is again a blend.
- Not yet checked: what the 896 FN look like (small/local anomaly at 24x14? anomaly in a merged cluster?).

## Candidate next steps (one at a time)
1. Fix the merged-cluster reference (e.g. nearest-k reference frames instead of a cluster median) - targets the one observed implementation flaw.
2. Higher-resolution or pretrained-embedding reference distance - only if FN images show small anomalies lost at 24x14.
Decision on 003 pending human; inspect FN frames first.

## Addendum (Critic, 13:40): FN inspection
- Opened 8 TRUE sequences (4 most confident FN p<0.05, 2 near-threshold FN p~0.40, 2 confident TP), saved as runs/002/critic_fn_sheet.jpg.
- Confident FN are mostly very dark hallway/door views where the visible anomaly candidate is a small or low-contrast figure (a few % of the frame); at 24x14 such a figure covers ~1-2 thumbnail pixels. Confident TP show large, close figures.
- errors.md: the merged ~2.7k-frame cluster (fold0 c3) has error 0.261, not worse than the other clusters (0.15-0.41). So the merged-cluster flaw is NOT the main FN source; option 1 above is dropped.
- Next: higher spatial resolution of the same reference method (003).
