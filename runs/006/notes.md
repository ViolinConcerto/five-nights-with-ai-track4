# run 006 notes
- Authorization (Jack, Critics chat 15:00 PDT): "please continue modify it until catch up with 0.99". referee/ and FREEZE.json untouched.
- train.py = runs/005/train.py with only the frame scorer changed: LR -> HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=SEED) (diff: docstring, import, model line).
- Preflight fold 0 (scratch006, 15:13): AUC 0.955 (005 fold0 0.940), 20.2s wall, peak RSS 1.37 GB. Projected full run ~2 min.
- Runner 15:13-15:15: RESULT fnaf-track1/006: decision=KEEP f1=0.8865 (fold std 0.0068) threshold=0.83 at0.5=0.6667 P=0.888 R=0.885 flagged=49.8% all_TRUE_f1=0.667 best_before=0.8731 (005) export=ok runtime=98.4s; OOF AUC 0.9601 (005: 0.9464). check_submission: VALID - 8,000 sequences, 3,845 flagged as anomalous (48%). Safe to upload.
