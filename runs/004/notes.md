
## Deadline change (2026-10-24 14:27:39 PDT)
- exploration_deadline: "2026-10-24T15:00:00-07:00" -> "2026-10-24T15:30:00-07:00" (only change in referee/contract.json; FREEZE.json and runner.py untouched)
- Authorization (Jack, Critics card): "我授权把探索截止改到 15:30，Engineer 的时间门相应顺延 30 分钟"
- new contract.json sha256: bc449fa674cf3620916dcdc496ea6b112b48976cd0d1fa8a874620f6c71a75ee
- Gates shifted: torch by 15:05, projection <=12 min, runner done by 15:25 PDT

## Gate results
- Gate 0 PASS 14:25:59 PDT: torch 2.8.0 / torchvision 0.23.0, MPS available; memory free 85%, swap 0 used.
- Gate 1 FAIL: 224x128 -> 43.2s/2000 frames, projected 53.9 min (14:27:14); 160x96 -> 39.1s, projected 48.9 min (14:28:34). Limit 12 min. Abandoned per plan; no extraction, no folds, runner not started. HOLD, 002 remains best.
- Resolution barely matters (-9%), so bottleneck is likely JPEG decode / DataLoader, not the CNN.

## Implementation fix pass (Critic, 14:29) - same encoder/features/model
- Decode: multiprocessing Pool(6), Image.draft('RGB',(2w,2h)); model: MPS, batch 256, torch.inference_mode(), fp16 (finite, stable), device asserted mps.
- 160x96 fp16 on 2000 frames: decode 1.6s, model 3.1s, total 4.8s -> projected 6.0 min.
- 224x128 fp16 (plan size): decode 2.3s, model 4.7s, total 7.1s -> projected 8.8 min (14:30:33). Old DataLoader version: 53.9 min.
- Authorization (Jack, 14:29 PDT, Critics card): "I authorize raising the 004 extraction projection limit to 20 minutes". (Projection 8.8 min also met the original 12 min limit.)
- Full extraction started at 224x128 fp16 under perl alarm 1500s.
- Preflight fold 0 (scratch004, 14:41): AUC 0.922, 9.0s, peak RSS 1.26 GB. Train cache 7000 rows: embed 271.8s, total 500.5s (CSV write slow).
- Full extraction done 14:45: train 7000 / test 8000 rows complete (ids match, finite); 882s wall, peak RSS 4.96 GB.
- Runner started ~14:45:40 -> refused: 'ERROR: frozen files do not match the freeze: changed referee/contract.json -> ask the human'. Stopped per instruction (no re-init, FREEZE.json untouched, no bypass). No eval/test outputs produced by runner.

## FREEZE.json update (14:55:07 PDT)
- Authorization (Jack, 14:54 PDT, Critics transcript): "I authorize Engineer to update the contract.json hash in FREEZE.json to the new value bc449fa6…75ee (that change only) and run 004"
- old: 322b9c30f989bb6b57e043bc9ea78ef996f8b5b0eaa173039b1826a522d40bd9 -> new: bc449fa674cf3620916dcdc496ea6b112b48976cd0d1fa8a874620f6c71a75ee (backup runs/004/FREEZE.json.bak_1454)
```
5c5
<     "referee/contract.json": "322b9c30f989bb6b57e043bc9ea78ef996f8b5b0eaa173039b1826a522d40bd9",
---
>     "referee/contract.json": "bc449fa674cf3620916dcdc496ea6b112b48976cd0d1fa8a874620f6c71a75ee",
```
- Runner 14:55 -> RESULT decision=KEEP f1=0.8526 thr 0.69 P 0.861 R 0.845 flagged 49.1% runtime 17.6s; promoted. check_submission: VALID (8000, 3882 flagged).
