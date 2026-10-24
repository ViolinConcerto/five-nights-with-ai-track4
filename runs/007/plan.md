parent: 006
hypothesis: Fine-tuning the last ResNet18 block (layer4) per fold on frames with the weak sequence label adapts the ImageNet features to FNAF animatronics and lifts AUC beyond the frozen-encoder ceiling of 006.

## Evidence
- 004-006: every gain came from better use of frozen ImageNet embeddings (AUC 0.931 -> 0.946 -> 0.960). review 006: the frozen encoder is the remaining limit; the official starter baseline also fine-tunes ResNet18 with max over frames.

## Change
New train.py (Engineer writes it; does not reuse the r18emb cache for the learned part):
- Per outer fold f, fit on fold != f only: ResNet18 ImageNet weights, layers up to layer3 frozen, layer4 + new linear head trained. Input 160x96 RGB, ImageNet norm, PIL draft decode in Pool(6) as in r18emb_v1, MPS, batch 64, AdamW lr 1e-4 (head 1e-3), 1 epoch, frame loss = BCE with sequence label on the max frame logit per sequence (MIL max, like the starter baseline). Seed fixed.
- Sequence proba_TRUE = sigmoid(max frame logit). Test = mean of fold models. No 005/006 kNN features (one change: learned encoder vs frozen).
- No internal search.

## Gates (Jack authorized deadline 15:50 PDT, one attempt)
- Gate A: time 1 epoch on 200 training sequences (+ inference on 200). Project 5 folds x (5600 train + 1400 val + 8000 test sequences). Go only if projection <= 18 min AND runner can finish by 15:45. Otherwise abandon: 006 stays best. No partial folds, no single-holdout substitute.
- max_run_seconds 900 still applies; if the projection exceeds 900 s, abandon.

## Success / failure
Success: KEEP over 0.8865 with AUC > 0.960. Failure or gate fail: record and STOP; 006 is final.
