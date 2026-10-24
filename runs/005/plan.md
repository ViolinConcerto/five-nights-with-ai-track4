parent: 004
hypothesis: Adding each frame's offset from its nearest normal frames of the same view (in the 004 embedding space) lets the frame scorer judge "is this figure where it should be for this camera", which a camera-agnostic linear scorer on raw embeddings cannot.

## Evidence
- 004 (KEEP, F1 0.8526, AUC ~0.93): semantic features break the pixel ceiling, but the scorer is a single linear model over all cameras. Anomaly per README = "an animatronic does something it shouldn't"; in 001/002 image sheets animatronics also appear in normal sequences, so "figure present" is not enough and what counts as normal depends on the view.
- 002 showed that a per-view normal reference helps (AUC 0.70 -> 0.80 on pixels); 005 applies that same idea in the semantic space that 004 validated.
- No other changes: encoder, cache r18emb_v1 (no new extraction), and aggregation stay as in 004.

## Change (inside train.py per fold, fit on fold != f only)
1. Same as 004: StandardScaler -> PCA(128, seed) fit on fold-train frames.
2. Reference/classifier split (Jack's 002 amendment): split fold-train FALSE sequences 50/50 with a fixed seed. The reference half's frames form the normal bank; the classifier trains on the other FALSE half plus all TRUE.
3. For every frame (classifier-train, val, test): z = PCA embedding; nn = mean of its k=5 nearest neighbors in the normal bank (sklearn NearestNeighbors, euclidean); features = [z, z - nn, ||z - nn||] (257 dims).
4. LogisticRegression(C=1.0, max_iter=2000) on these frame features with the weak sequence label; sequence proba_TRUE = max over the 10 frames (same as 004). Test = mean of fold models.
No internal search. k=5 fixed.

## Cost and gates (deadline 15:30 PDT)
- No new extraction. kNN: ~28k bank x ~70k queries x 128 dims per fold, ~5-15 s with brute force / ball tree; total run est. 1-3 min (004 took 17.6 s).
- One-fold preflight first (time + memory). Runner must finish by 15:25 PDT; if preflight projects beyond that, abandon 005 and keep 004.

## Success / failure
- Success: KEEP (F1 > 0.8526 + 0.002) with AUC above 004's, precision and recall not traded off.
- Failure: no gain -> per-view normality isn't the missing factor at the frozen-encoder level; next phase = fine-tuning (needs a new authorized phase).
