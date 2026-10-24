# FNAF Track 1 - Engineer + Critic report

Development scores = pooled OOF F1 on the frozen 5-fold split (runner); not official results.

| run | parent | change | OOF F1 | decision |
|---|---|---|---|---|
| 001 | - | diffmed_v1 (7 global median-deviation stats) + HGB | 0.6742 | KEEP |
| 002 | 001 | per-cluster (camera) normal pixel reference, ref/classifier split | 0.7279 | KEEP |
| 003 | 002 | 48x28 thumbnails | 0.7283 | REJECT |
| 004 | 002 | PIVOT: frozen ResNet18 frame embeddings + LR, max over frames | 0.8526 | KEEP |
| 005 | 004 | + kNN offset from normal frames in embedding space | 0.8731 | KEEP |
| 006 | 005 | LR -> HGB frame scorer | 0.8865 | KEEP (final best) |
| 007 | 006 | fine-tune layer4 | - | not run (time gate failed) |

What worked: seeing that frames mix cameras (002); recognizing after 003 that pixel-deviation was the wrong representation and pivoting to semantic features (004, largest gain); re-applying the camera-normal idea in embedding space (005); a nonlinear scorer (006). AUC rose 0.80 -> 0.96.
What did not: resolution increase (003, same AUC); end-to-end fine-tuning did not fit the time budget on an M1 (007). Official leaderboard (monitor, unverified by Critic): 0.8899 after a later upload vs leaders ~0.997.

Human interventions (Jack): amended 002 (reference/classifier split, cluster check, one-fold preflight); moved the exploration deadline 15:00 -> 15:30 -> 15:50 with FREEZE hash updates; raised the 004 extraction projection limit to 20 min; requested the strategic review and continued iteration; did all uploads.
