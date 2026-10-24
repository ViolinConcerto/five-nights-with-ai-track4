# run 007 notes
## Contract change (15:21:25 PDT)
- Authorization (Jack, Critics card 15:20 PDT): "I authorize moving the exploration deadline to 15:50 and updating the FREEZE hash, for one fine-tuning attempt"
- exploration_deadline 2026-10-24T15:30:00-07:00 -> 2026-10-24T15:50:00-07:00
- contract.json sha256 bc449fa674cf3620916dcdc496ea6b112b48976cd0d1fa8a874620f6c71a75ee -> a24a4df3fad8d24d64a06558db1a6b711d0de7815a04ee911173f1ea6fc316c3 (FREEZE backup runs/007/FREEZE.json.bak_1520)
```
15c15
<   "exploration_deadline": "2026-10-24T15:30:00-07:00",
---
>   "exploration_deadline": "2026-10-24T15:50:00-07:00",
5c5
<     "referee/contract.json": "bc449fa674cf3620916dcdc496ea6b112b48976cd0d1fa8a874620f6c71a75ee",
---
>     "referee/contract.json": "a24a4df3fad8d24d64a06558db1a6b711d0de7815a04ee911173f1ea6fc316c3",
```
## Gate A (15:22 PDT) - FAIL, 007 abandoned
- 1 epoch train on 200 seqs: 16.6s; inference on 200 seqs: 4.1s (160x96, batch 64 = 6 seqs, MPS, Pool(6) draft decode; peak RSS 0.88 GB).
- Projection 5 x (5600 train + 9400 infer) = 3271s (54.5 min) > 18 min and > 900s max_run_seconds; could not finish by 15:45.
- No runner run, no partial folds, no holdout substitute. 006 stays best. Contract deadline/FREEZE edits (above) remain in place.
