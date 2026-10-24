# review 004 (Critic)

Hypothesis (semantic frozen ResNet18 frame embeddings + supervised frame scorer, max over frames, beat pixel-deviation features): SUPPORTED. KEEP, OOF F1 0.8526 vs 0.7279 (002), threshold 0.69 (result.json).
- Precision 0.712 -> 0.861 and recall 0.744 -> 0.845 both rose and the flagged share fell from 52.2% to 49.1%. Engineer reports OOF AUC 0.931 vs 0.798. This is real separation, not a threshold move.
- Strategic lesson: runs 001-003 were local optimization inside the wrong representation (pixel-deviation magnitude). The pivot to semantic features produced a larger gain than all earlier rounds combined.
- Process notes: the first extraction attempt failed its speed gate because of the loader, not the model, and the fix kept the hypothesis unchanged. Jack authorized moving the deadline to 15:30, raising the projection limit to 20 min, and updating the FREEZE hash (runs/004/notes.md).
- Remaining gap: local OOF 0.85 vs leaderboard leaders reported at ~0.96-0.99 (not directly comparable). The likely next step is fine-tuning the encoder end to end, which does not fit before 15:30.
