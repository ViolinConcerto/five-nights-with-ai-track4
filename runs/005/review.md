# review 005 (Critic)

Hypothesis (per-view normal offset in 004 embedding space helps the frame scorer): SUPPORTED. KEEP, OOF F1 0.8731 vs 0.8526, threshold 0.84 (result.json). Precision 0.861->0.876, recall 0.845->0.870, flagged 49.6%; Engineer reports AUC 0.946 vs 0.931.
Lesson: the 002 idea (camera-normal reference) was right but limited by the pixel representation; in semantic space it adds a real gain.
Remaining weakness: frozen ImageNet encoder + linear scorer. Next would be end-to-end fine-tuning (needs > 1 h with 5-fold OOF on M1).
