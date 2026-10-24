# run 005 notes
- Authorization (Jack, Critics chat 15:00 PDT): "please continue modify it until catch up with 0.99" -> continue experiments. Deadline unchanged 15:30 PDT; referee/ and FREEZE.json untouched.
- train.py per plan: reuses cache r18emb_v1 (no extraction); scaler+PCA128 on fold-train frames; FALSE 50/50 split seed SEED+f; kNN k=5 euclidean bank; features [z, z-nn, ||z-nn||] (257); LR C=1; max over frames; test mean of folds.
- Preflight fold 0 (scratch005, 15:02): AUC 0.940 (004 fold0 0.922), 11.2s wall, peak RSS 1.29 GB. Projected full run ~1 min.
- Runner 15:09-15:11: RESULT fnaf-track1/005: decision=KEEP f1=0.8731 (fold std 0.0141) threshold=0.84 at0.5=0.6667 P=0.876 R=0.870 flagged=49.6% all_TRUE_f1=0.667 best_before=0.8526 (004) export=ok runtime=66.3s; OOF AUC 0.9464 (004: 0.9313). check_submission: VALID - 8,000 sequences, 3,812 flagged as anomalous (48%). Safe to upload.
