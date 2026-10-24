parent: none
hypothesis: baseline - diffmed_v1 features + HistGradientBoosting

evidence: none (first run); reference is all-TRUE F1 = 0.667.
change: template train_features_template.py unchanged; FEATURE_SETS=["diffmed_v1"] (7 label-free median-deviation/step features), HistGradientBoostingClassifier(max_iter=300, lr=0.05, seed 42). No internal search.
expected effect: F1 above 0.667 baseline.
