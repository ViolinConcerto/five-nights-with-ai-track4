# run 001 error analysis (OOF, threshold 0.15 chosen by runner)

Confusion: TP=3447 FP=3278 FN=53 TN=222 (n=7000)

proba_TRUE by true label:
```
        count   mean    std    min    25%    50%    75%    max
y                                                             
False  3500.0  0.453  0.187  0.007  0.331  0.451  0.579  0.987
True   3500.0  0.549  0.200  0.016  0.405  0.537  0.690  0.998
```

Accuracy at 0.5 (info only): 0.590

Takeaway: class distributions of proba_TRUE overlap heavily; the F1-optimal threshold 0.15 flags 96% of sequences, so the model is nearly all-TRUE (F1 0.674 vs 0.667). The 7 global median-deviation features carry little signal. Next ideas need localized/per-camera features (e.g. per-camera reference computed inside the fold pipeline, max local patch deviation).
