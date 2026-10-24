# run 003 engineer notes (13:45 PDT, before implementation)
plan.md unchanged. Carried-over amendments from runs/002/notes.md (already referenced in plan.md 2):
- per-fold seeded (SEED+fold) 50/50 split of fold-train FALSE sequences into reference half / classifier half; KMeans(k=10, n_init=5) + per-cluster median/MAD (+0.02) fit on reference half only; classifier on remaining FALSE + all TRUE; val/test use the fold's frozen reference.
- KMeans and cluster assignment on 2x2 average-pooled 24x14 frames (same method as 002); median/MAD and z maps at 48x28; max-cell grid 8x8.
Implementation detail (no hypothesis change): thumb_v2 CSV read with uint8 dtype and features computed in chunks to limit memory (swap was ~10 GB used before the session).
