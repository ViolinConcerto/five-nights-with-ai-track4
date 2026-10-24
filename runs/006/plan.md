parent: 005
hypothesis: Replacing the linear frame scorer with a nonlinear one (HistGradientBoosting) on the same 257 frame features captures interactions between "which view / which figure" and "offset from normal" that a single linear boundary across all cameras cannot.

## Evidence
- 005 KEEP (F1 0.8731, AUC ~0.946). review 005: remaining weakness = frozen encoder + linear scorer. Fine-tuning does not fit before 15:30; the scorer is the only cheap lever.
- Normality depends on the camera (002/005), so one global linear weight vector over [z, z-nn, dist] is a strong restriction.

## Change (one factor: frame scorer)
Everything identical to runs/005/train.py (r18emb_v1 cache, scaler+PCA128, 50/50 reference/classifier split of FALSE, k=5 kNN offset features, max over frames, mean of fold models for test), except:
LogisticRegression -> HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=SEED). No internal search.

## Cost and gates
No extraction. HGB on ~42k frames x 257 dims per fold: est. 10-30 s per fold. One-fold preflight first; runner must finish by 15:25 PDT, else abandon and keep 005.

## Success / failure
Success: KEEP (F1 > 0.8731 + 0.002) with AUC > 0.946. Failure: no gain -> scorer capacity is not the limit; the encoder is (needs fine-tuning phase).
