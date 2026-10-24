# run 002 engineer notes (amendments approved by Jack 13:25 PDT, recorded before implementation)
plan.md is unchanged. Implementation amendments:
1. Reference/classifier separation: inside each outer training fold, FALSE sequences are split with fixed seed (SEED) 50/50 into a REFERENCE subset and the rest. KMeans(n_clusters=10, n_init=5, random_state=SEED) and per-cluster median/MAD (+0.02 floor) are fit only on frames of the reference subset. The classifier trains only on sequences not used for the reference (remaining FALSE + all TRUE of that fold). Val and test are transformed with that fold's frozen reference. No validation labels in any fit. Frozen split/runner/contract/deadline untouched. (This removes the self-reference bias noted in plan.md 2.d.)
2. Cluster check: cluster != camera is not assumed; runs/002/cluster_check.png shows random reference frames per cluster (fold 0); sizes and mixed clusters recorded below.
3. Budget: thumb_v1 timed on 200, full extraction under perl alarm with FEAT_WORKERS=6, one-fold preflight outside the runner (scratch dir /Users/admin/workspace/scratch002) before runner run.

## Results of checks (13:33 PDT)
- thumb_v1: 200-seq timing 3.6s (proj 4.5 min); full run under perl alarm 420s, FEAT_WORKERS=6: train 60.9s + test 68.9s = 130s; 7000/8000 rows, 3360 cols, 0 NaN, unique ids, values 0-246.
- Preflight fold 0 (scratch /Users/admin/workspace/scratch002, outside runner): KMeans 1.4s, median/MAD 0.2s, features 0.9s, HGB 8.2s; total 16.2s wall, peak RSS 1.22 GiB -> 5-fold projection ~80s. Preflight scores not used.
- Cluster check (runs/002/cluster_check.png, fold 0): sizes [1241,1333,1211,2765,1339,1225,1188,1297,1094,1307]. 9 clusters look like single scenes. Cluster 3 (2765 frames, ~2x others) is MIXED: a dark room with small lights plus dark/figure/bright-spot frames, i.e. likely 2 views merged. Cluster 9 contains a few frames from a different (diagonal structure) view. Every fold has one ~2.6-2.8k cluster, so the 10-camera assumption is only approximately met by KMeans k=10.
- Runner attempt 1 OK: 64.8s of 900s.
