# Team protocol: Engineer + Critic

Goal: maximize the frozen metric on the task in /workspace/comp before the exploration deadline in referee/contract.json, and leave a valid best/submission.csv.

Files are the source of truth, chat is only for handoffs. `referee/runner.py` is the only thing that scores, decides KEEP/REJECT, and writes best/ and events.jsonl. Nobody edits referee/, best/, events.jsonl, result.json or error_analysis.csv. If something in referee/ looks wrong, ask the human.

## One round = one folder runs/NNN/ (next free 3-digit number)
1. Critic writes runs/NNN/plan.md BEFORE any code. First two lines exactly:
   `parent: <run>` and `hypothesis: <one line>`, then: evidence from error analysis, the concrete change, expected effect. plan.md is never edited afterwards.
2. Engineer copies the parent's train.py into runs/NNN/, implements only that change, then runs
   `python referee/runner.py run runs/NNN`
   The runner enforces the time budget, attempt limit, frozen split and deadline, scores against frozen labels, and prints one RESULT line.
   On FAILED / TIMEOUT / INVALID / EXPORT_FAILED with retry_allowed=True the Engineer owns the round: it fixes the bug and reruns the same folder (time and attempts are shared) while the Critic waits. A fix must not change the hypothesis. If it cannot, it posts `GIVE_UP NNN: <reason>`.
3. Engineer posts the RESULT line verbatim to the Critic.
4. Critic reads runs/NNN/result.json and error_analysis.csv, writes runs/NNN/review.md (was the hypothesis supported, what errors remain), then posts the next PROPOSAL or STOP.

## FNAF Track 1 specifics
- Task: 10-frame security-camera sequences, TRUE if any frame is anomalous. Metric: F1 of TRUE. 10 fixed cameras.
- Threshold mode: train.py writes probabilities only (eval_preds.csv with proba_FALSE/proba_TRUE, and test_proba.csv). The runner picks the threshold on out-of-fold probabilities and writes submission.csv itself. Never threshold in train.py.
- Feature extraction is a separate, label-free step: `python features/<name>.py` (start from templates/featurize_template.py; time it first with `python features/<name>.py 50`). It reads only sequence_id + seq_dir, never labels, and writes cache/<name>_{train,test}.csv. A new feature set = a new file name; never overwrite an existing cache file. The Critic names the feature set in plan.md; the Engineer builds it, then runs the candidate (start from templates/train_features_template.py).
- Feature-extraction budget: always time a new feature set on 200 sequences first (`python features/<name>.py 200` prints the projection for all 15,000 sequences), and run the full extraction under `timeout`. If the projection does not fit the time left before the deadline (leave 45 minutes for finalizing), simplify the idea. Extraction time is part of the cost of a proposal; cache/<name>.meta.json records it.
- Training data is balanced (50% TRUE), so predicting TRUE everywhere already gives F1 = 0.667. The RESULT line shows precision, recall and the flagged share; judge progress against that baseline.
- Use /workspace/venv/bin/python for everything.
- Official uploads (max 200, last valid one counts) are done by the human from best/submission.csv only. The public leaderboard is never used to choose between candidates.

## Rules
- One main hypothesis per run. Any internal search (e.g. trying 3 values of C) must be declared in plan.md.
- Fit on fold != f, predict fold == f, exactly as SPLIT_CSV says. All learned preprocessing goes inside the per-fold pipeline. Features from a frozen pretrained encoder may be cached once under CACHE_DIR.
- Never hard-code validation ids or labels. Never tune on the public leaderboard.
- One training job at a time (the runner holds a lock).
- Messages, nothing else: `PROPOSAL NNN (parent PPP): <hypothesis>` / the runner's `RESULT ...` line / `GIVE_UP NNN: <reason>` / `WAITING_FOR_APPROVAL: <what>` (to the human) / `STOP: <reason>` (ends the whole experiment). Do not reply to a message that needs no action.
- STOP when the deadline passes, after 2 REJECTs in a row with no new evidence-based idea, or when the human says stop. After STOP nobody sends further messages.
