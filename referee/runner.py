#!/usr/bin/env python3
"""Fixed runner + evaluator for the two-Bot ML loop (classification).

Layout (ROOT = the folder that contains referee/):
  ROOT/referee/runner.py, contract.json, frozen/   <- frozen after `init`
  ROOT/task/...                                    <- official data
  ROOT/runs/NNN/{plan.md, train.py, ...}           <- one directory per candidate
  ROOT/best/current.json, best/submission.csv      <- written only by this runner
  ROOT/events.jsonl                                <- append-only ledger, written only by this runner

Commands:
  python referee/runner.py init              build frozen split + labels from contract, record hashes
  python referee/runner.py verify            check frozen files against recorded hashes
  python referee/runner.py run runs/003      run runs/003/train.py under budget, score, decide, promote
  python referee/runner.py rescore runs/003  recompute a run's score from its saved predictions (read-only)
  python referee/runner.py summary           markdown table (+ summary.png if matplotlib is available)
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REF = Path(__file__).resolve().parent
ROOT = REF.parent
CONTRACT = REF / "contract.json"
FROZEN = REF / "frozen"
FREEZE = FROZEN / "FREEZE.json"
RUNS = ROOT / "runs"
BEST = ROOT / "best"
EVENTS = ROOT / "events.jsonl"

DIRECTION = {"accuracy": "max", "f1": "max", "macro_f1": "max", "weighted_f1": "max", "roc_auc": "max",
             "log_loss": "min"}
NEEDS_PROBA = {"roc_auc", "log_loss"}
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
SETTLED = {"KEEP", "REJECT"}  # EXPORT_FAILED may be fixed and re-run in the same folder


# ---------------------------------------------------------------- helpers
def now():
    return datetime.now().astimezone()


def die(msg, code=2):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def load_contract():
    if not CONTRACT.exists():
        die(f"{CONTRACT} not found (copy contract.example.json and fill it in)")
    c = json.loads(CONTRACT.read_text())
    for k in ("task_id", "train_csv", "id_col", "label_col", "split", "metric", "epsilon",
              "max_run_seconds", "max_attempts_per_run"):
        if c.get(k) in (None, ""):
            die(f"contract.json: '{k}' must be set")
    if c["metric"] not in DIRECTION:
        die(f"contract.json: metric must be one of {sorted(DIRECTION)}")
    return c


def frozen_paths():
    return [CONTRACT, Path(__file__).resolve(), FROZEN / "split.csv", FROZEN / "labels.csv", FROZEN / "classes.json"]


def verify_frozen():
    """Return a list of problems; empty list means everything matches the freeze."""
    if not FREEZE.exists():
        return ["not frozen yet: run `runner.py init` after the human approves the contract"]
    recorded = json.loads(FREEZE.read_text())["sha256"]
    problems = []
    for p in frozen_paths():
        rel = str(p.relative_to(ROOT))
        if not p.exists():
            problems.append(f"missing {rel}")
        elif sha256(p) != recorded.get(rel):
            problems.append(f"changed {rel}")
    return problems


def append_event(ev):
    ev = {"ts": now().isoformat(timespec="seconds"), **ev}
    with open(EVENTS, "a") as f:
        f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")


def read_frozen():
    split = pd.read_csv(FROZEN / "split.csv", dtype={"id": str})
    labels = pd.read_csv(FROZEN / "labels.csv", dtype={"id": str, "label": str})
    classes = json.loads((FROZEN / "classes.json").read_text())
    return split, labels, classes


def parse_plan(run_dir):
    text = (run_dir / "plan.md").read_text() if (run_dir / "plan.md").exists() else ""
    parent = re.search(r"^\s*parent\s*:\s*(\S+)", text, re.I | re.M)
    hyp = re.search(r"^\s*hypothesis\s*:\s*(.+)$", text, re.I | re.M)
    return (parent.group(1) if parent else None), (hyp.group(1).strip() if hyp else None)


# ---------------------------------------------------------------- init
def cmd_init():
    c = load_contract()
    if FREEZE.exists():
        die("already frozen. A deliberate rule change = new phase: copy the kit into a NEW folder and init there.")
    from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                         StratifiedKFold, StratifiedShuffleSplit)
    idc, lab, grp = c["id_col"], c["label_col"], c.get("group_col")
    cols = [idc, lab] + ([grp] if grp else [])
    df = pd.read_csv(ROOT / c["train_csv"], usecols=cols, dtype=str)
    if df[[idc, lab]].isna().any().any():
        die("train_csv has empty ids or labels")
    if df[idc].duplicated().any():
        die("train_csv has duplicate ids")
    classes = sorted(df[lab].unique().tolist())
    if len(classes) < 2:
        die("need at least 2 classes")
    s = c["split"]
    seed = int(s.get("seed", 42))
    y, groups = df[lab].to_numpy(), (df[grp].to_numpy() if grp else None)
    fold = np.full(len(df), -1)  # -1 = training only, never scored
    if s["scheme"] == "kfold":
        n = int(s["n_splits"])
        kf = (StratifiedGroupKFold(n, shuffle=True, random_state=seed) if grp
              else StratifiedKFold(n, shuffle=True, random_state=seed))
        for f, (_, va) in enumerate(kf.split(np.zeros(len(df)), y, groups)):
            fold[va] = f
    elif s["scheme"] == "holdout":
        frac = float(s["val_fraction"])
        sp = (GroupShuffleSplit(1, test_size=frac, random_state=seed) if grp
              else StratifiedShuffleSplit(1, test_size=frac, random_state=seed))
        _, va = next(sp.split(np.zeros(len(df)), y, groups))
        fold[va] = 0
    else:
        die("split.scheme must be 'kfold' or 'holdout'")
    FROZEN.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df[idc], "fold": fold}).to_csv(FROZEN / "split.csv", index=False)
    pd.DataFrame({"id": df[idc], "label": df[lab]}).to_csv(FROZEN / "labels.csv", index=False)
    (FROZEN / "classes.json").write_text(json.dumps(classes, ensure_ascii=False))
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in frozen_paths()}
    write_json(FREEZE, {"task_id": c["task_id"], "frozen_at": now().isoformat(timespec="seconds"), "sha256": hashes})
    RUNS.mkdir(exist_ok=True)
    append_event({"event": "FROZEN", "task_id": c["task_id"], "sha256": hashes})

    print(f"FROZEN task={c['task_id']} rows={len(df)} classes={len(classes)} metric={c['metric']}")
    print("class counts:", df[lab].value_counts().to_dict())
    print("fold sizes:  ", pd.Series(fold).value_counts().sort_index().to_dict(), "(-1 = train only)")
    print("Keep these hashes on your own machine:")
    for k, v in hashes.items():
        print(f"  {v}  {k}")


# ---------------------------------------------------------------- scoring
def metric_value(name, y_true, y_pred, P, classes, positive):
    from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
    if name == "accuracy":
        return accuracy_score(y_true, y_pred)
    if name == "f1":  # binary F1 of the positive class
        return f1_score(y_true, y_pred, pos_label=positive or classes[1], average="binary", zero_division=0)
    if name in ("macro_f1", "weighted_f1"):
        return f1_score(y_true, y_pred, labels=classes, average=name.split("_")[0], zero_division=0)
    if name == "roc_auc":
        if len(classes) == 2:
            pos = positive or classes[1]
            return roc_auc_score((y_true == pos).astype(int), P[:, classes.index(pos)])
        return roc_auc_score(y_true, P, multi_class="ovr", labels=classes)
    if name == "log_loss":
        return log_loss(y_true, P, labels=classes)
    raise ValueError(name)


def score_run(run_dir, c, write=True):
    split, labels, classes = read_frozen()
    path = run_dir / "eval_preds.csv"
    if not path.exists():
        return {"valid": False, "errors": ["eval_preds.csv not written"]}
    try:
        p = pd.read_csv(path, dtype={"id": str, "pred": str})
    except Exception as e:  # noqa: BLE001
        return {"valid": False, "errors": [f"cannot read eval_preds.csv: {e}"]}

    errors = []
    thr_mode = c.get("threshold")  # None = use the candidate's pred column; "oof_optimal" or a number = runner thresholds
    if thr_mode is not None and len(classes) != 2:
        return {"valid": False, "errors": ["contract.threshold needs a binary task"]}
    if thr_mode is not None and "pred" not in p.columns:
        p["pred"] = classes[0]  # placeholder; the runner sets the real labels from probabilities below
    missing_cols = [k for k in ("id", "fold", "pred") if k not in p.columns]
    if missing_cols:
        return {"valid": False, "errors": [f"eval_preds.csv missing columns {missing_cols}"]}
    expected = split[split["fold"] >= 0]
    if p["id"].duplicated().any():
        errors.append(f"{int(p['id'].duplicated().sum())} duplicate ids")
    miss = set(expected["id"]) - set(p["id"])
    extra = set(p["id"]) - set(expected["id"])
    if miss:
        errors.append(f"{len(miss)} required ids missing (e.g. {sorted(miss)[:3]})")
    if extra:
        errors.append(f"{len(extra)} unexpected ids, e.g. training-only or unknown rows ({sorted(extra)[:3]})")
    m = p.merge(expected, on="id", how="inner", suffixes=("", "_frozen"))
    if (pd.to_numeric(m["fold"], errors="coerce") != m["fold_frozen"]).any():
        errors.append("fold column does not match the frozen split")
    bad_pred = ~p["pred"].isin(classes)
    if thr_mode is None and bad_pred.any():
        errors.append(f"{int(bad_pred.sum())} preds not in classes {classes[:10]}")

    pcols = [f"proba_{k}" for k in classes]
    has_any = [col for col in p.columns if col.startswith("proba_")]
    P = None
    if has_any or c["metric"] in NEEDS_PROBA or thr_mode is not None:
        unknown = sorted(set(has_any) - set(pcols))
        lacking = [col for col in pcols if col not in p.columns]
        if unknown:
            errors.append(f"unknown probability columns {unknown[:5]}")
        if lacking:
            errors.append(f"missing probability columns {lacking[:5]} (need proba_<class> for every class)")
        if not unknown and not lacking:
            vals = p[pcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)  # text -> NaN -> INVALID
            if not np.isfinite(vals).all() or (vals < 0).any() or (vals > 1).any():
                errors.append("probabilities must be finite and within [0, 1]")
            elif np.abs(vals.sum(1) - 1).max() > 1e-3:
                errors.append("probability rows must sum to 1")
    if errors:
        return {"valid": False, "errors": errors}

    m = m.merge(labels, on="id", how="left")
    if pcols[0] in m.columns:
        P = m[pcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    threshold, extra = None, {}
    if thr_mode is not None:
        pos = c.get("positive_class") or classes[1]
        neg = [k for k in classes if k != pos][0]
        p_pos = P[:, classes.index(pos)]
        y_all = m["label"].to_numpy()

        def labels_at(t, probs):
            return np.where(probs >= t, pos, neg)

        if thr_mode == "oof_optimal":
            f1s = np.array([metric_value(c["metric"], y_all, labels_at(t, p_pos), P, classes, pos)
                            for t in THRESHOLD_GRID])
            best_ts = THRESHOLD_GRID[f1s >= f1s.max() - 1e-12]
            threshold = float(best_ts[len(best_ts) // 2])  # middle of the tied optimum: more stable
            extra["score_at_0.5_info_only"] = float(metric_value(c["metric"], y_all, labels_at(0.5, p_pos), P,
                                                                  classes, pos))
        else:
            threshold = float(thr_mode)
        m["pred"] = labels_at(threshold, p_pos)
    fold_scores = {}
    try:
        for f in sorted(m["fold_frozen"].unique()):
            sel = (m["fold_frozen"] == f).to_numpy()
            fold_scores[int(f)] = float(metric_value(c["metric"], m["label"].to_numpy()[sel], m["pred"].to_numpy()[sel],
                                                     None if P is None else P[sel], classes, c.get("positive_class")))
        pooled = float(metric_value(c["metric"], m["label"].to_numpy(), m["pred"].to_numpy(), P, classes,
                                    c.get("positive_class")))
    except Exception as e:  # noqa: BLE001
        return {"valid": False, "errors": [f"scoring failed: {e}"]}
    vals = list(fold_scores.values())
    if write:
        ea = m[["id", "fold_frozen", "label", "pred"] + ([*pcols] if P is not None else [])].rename(
            columns={"fold_frozen": "fold", "label": "y_true"})
        ea.insert(4, "correct", ea["y_true"] == ea["pred"])
        ea.to_csv(run_dir / "error_analysis.csv", index=False)
    tuned = thr_mode == "oof_optimal"
    if c["metric"] == "f1":  # context for F1: precision/recall, how much is flagged, and the all-positive baseline
        pos = c.get("positive_class") or classes[1]
        yt, yp = m["label"].to_numpy() == pos, m["pred"].to_numpy() == pos
        tp = float((yt & yp).sum())
        prev = float(yt.mean())
        extra.update(precision=tp / yp.sum() if yp.sum() else 0.0, recall=tp / yt.sum() if yt.sum() else 0.0,
                     predicted_positive_rate=float(yp.mean()), all_positive_f1_baseline=2 * prev / (1 + prev))
    return {
        "valid": True,
        "selection_score": pooled if tuned else float(np.mean(vals)),
        "aggregation": ("pooled OOF score at the OOF-selected threshold (threshold chosen on the same predictions: "
                        "possible selection bias, not an independent test score)" if tuned
                        else "mean_of_fold_scores"),
        "threshold": threshold,
        **extra,
        "fold_scores": fold_scores,
        "fold_std_descriptive_only": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
        "pooled_score_info_only": pooled,
        "n_scored": int(len(m)),
    }


def build_submission(run_dir, c, classes, threshold):
    """Threshold mode: turn the candidate's test_proba.csv into submission.csv with the run's frozen-rule threshold."""
    sub_cfg = c.get("submission") or {}
    src = run_dir / "test_proba.csv"
    if not src.exists():
        return ["test_proba.csv not written (columns: id, proba_<positive class>)"]
    pos = c.get("positive_class") or classes[1]
    neg = [k for k in classes if k != pos][0]
    t = pd.read_csv(src, dtype={"id": str})
    col = f"proba_{pos}"
    if "id" not in t.columns or col not in t.columns:
        return [f"test_proba.csv needs columns id and {col}"]
    v = pd.to_numeric(t[col], errors="coerce")
    if v.isna().any() or (v < 0).any() or (v > 1).any() or t["id"].duplicated().any():
        return ["test_proba.csv has duplicate ids or probabilities outside [0, 1]"]
    out = pd.DataFrame({sub_cfg.get("id_col") or c["id_col"]: t["id"],
                        sub_cfg.get("pred_col") or c["label_col"]: np.where(v >= threshold, pos, neg)})
    out.to_csv(run_dir / "submission.csv", index=False)
    return []


def check_submission(run_dir, c, classes):
    sub_cfg = c.get("submission") or {}
    if not sub_cfg.get("sample_csv"):
        return "pending_inputs", []
    sample = pd.read_csv(ROOT / sub_cfg["sample_csv"], dtype=str)
    path = run_dir / "submission.csv"
    if not path.exists():
        return "failed", ["submission.csv not written"]
    d = pd.read_csv(path, dtype=str)
    errs = []
    if list(d.columns) != list(sample.columns):
        errs.append(f"columns {list(d.columns)} != sample {list(sample.columns)}")
    idc = sub_cfg.get("id_col", sample.columns[0])
    if idc in d.columns:
        if d[idc].duplicated().any():
            errs.append("duplicate ids in submission")
        if set(d[idc]) != set(sample[idc]):
            errs.append("submission ids differ from sample submission ids")
    if d.isna().any().any():
        errs.append("empty cells in submission")
    pc, kind = sub_cfg.get("pred_col"), sub_cfg.get("pred_values", "class")
    if pc and pc in d.columns:
        if kind == "class" and not d[pc].isin(classes).all():
            errs.append(f"{pc} has values outside the class list")
        if kind == "probability":
            v = pd.to_numeric(d[pc], errors="coerce")
            if v.isna().any() or (v < 0).any() or (v > 1).any():
                errs.append(f"{pc} must be probabilities in [0, 1]")
    return ("ok" if not errs else "failed"), errs


# ---------------------------------------------------------------- run
def cmd_run(run_arg):
    lock = open(REF / ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        die("another run is in progress (one training job at a time)")
    c = load_contract()
    problems = verify_frozen()
    if problems:
        append_event({"event": "REFUSED", "reason": "frozen_mismatch", "details": problems})
        die("frozen files do not match the freeze: " + "; ".join(problems) + " -> ask the human")
    if c.get("exploration_deadline") and now() > datetime.fromisoformat(c["exploration_deadline"]):
        die(f"exploration deadline {c['exploration_deadline']} has passed: no new runs; finalize the current best")

    run_dir = Path(run_arg).resolve()
    if run_dir.parent != RUNS.resolve():
        die(f"run directory must be directly inside {RUNS}")
    run = run_dir.name
    for need in ("plan.md", "train.py"):
        if not (run_dir / need).exists():
            die(f"{run}/{need} is missing (plan.md must exist before running)")
    prev = run_dir / "result.json"
    if prev.exists() and json.loads(prev.read_text()).get("decision") in SETTLED:
        die(f"{run} is already settled; create a new run directory for a new idea")

    adir = run_dir / "attempts"
    adir.mkdir(exist_ok=True)
    state_path = adir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"attempts": 0, "elapsed_s": 0.0}
    if state["attempts"] >= int(c["max_attempts_per_run"]):
        die(f"{run}: attempt limit {c['max_attempts_per_run']} reached; give up on this run or ask the human")
    remaining = float(c["max_run_seconds"]) - state["elapsed_s"]
    if remaining <= 1:
        die(f"{run}: time budget {c['max_run_seconds']}s used up")

    attempt = state["attempts"] + 1
    for stale in ("eval_preds.csv", "test_proba.csv", "submission.csv", "error_analysis.csv"):  # never score leftovers
        if (run_dir / stale).exists():
            shutil.move(str(run_dir / stale), str(adir / f"{attempt - 1:02d}_{stale}"))

    sub_cfg = c.get("submission") or {}
    env = dict(os.environ,
               TASK_ROOT=str(ROOT), RUN_DIR=str(run_dir), TRAIN_CSV=str(ROOT / c["train_csv"]),
               TEST_CSV=str(ROOT / c["test_csv"]) if c.get("test_csv") else "",
               SPLIT_CSV=str(FROZEN / "split.csv"), CLASSES_JSON=str(FROZEN / "classes.json"),
               ID_COL=c["id_col"], LABEL_COL=c["label_col"], GROUP_COL=c.get("group_col") or "",
               SUB_ID_COL=sub_cfg.get("id_col") or c["id_col"], SUB_PRED_COL=sub_cfg.get("pred_col") or c["label_col"],
               SEED=str(c["split"].get("seed", 42)), PYTHONHASHSEED="0", CACHE_DIR=str(ROOT / "cache"))
    log_path = adir / f"{attempt:02d}.log"
    t0 = time.time()
    status = "ok"
    with open(log_path, "w") as log:
        proc = subprocess.Popen([sys.executable, "train.py"], cwd=run_dir, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)  # kill the whole process group, children included
            proc.wait()
            rc, status = None, "TIMEOUT"
    elapsed = time.time() - t0
    state = {"attempts": attempt, "elapsed_s": state["elapsed_s"] + elapsed}
    write_json(state_path, state)

    parent, hypothesis = parse_plan(run_dir)
    best = json.loads((BEST / "current.json").read_text()) if (BEST / "current.json").exists() else None
    res = {"task_id": c["task_id"], "run": run, "parent": parent, "hypothesis": hypothesis, "attempt": attempt,
           "metric": c["metric"], "direction": DIRECTION[c["metric"]], "epsilon": c["epsilon"],
           "runtime_s": round(elapsed, 1), "cumulative_s": round(state["elapsed_s"], 1),
           "budget_s": c["max_run_seconds"], "log": str(log_path.relative_to(ROOT)),
           "best_before": {"run": best["run"], "score": best["score"]} if best else None}

    after = verify_frozen()
    if after:
        status, res["errors"] = "INVALID", ["frozen files changed during the run: " + "; ".join(after)]
    elif status == "ok" and rc != 0:
        tail = log_path.read_text(errors="replace").splitlines()[-15:]
        status, res["errors"] = "FAILED", [f"train.py exited with code {rc}"] + tail
    elif status == "TIMEOUT":
        res["errors"] = [f"killed after {remaining:.0f}s (remaining budget)"]

    if status == "ok":
        try:
            sc = score_run(run_dir, c)
        except Exception as e:  # noqa: BLE001  any unexpected scoring error -> INVALID with a reason, never a crash
            sc = {"valid": False, "errors": [f"scoring error: {type(e).__name__}: {e}"]}
        if not sc["valid"]:
            status, res["errors"] = "INVALID", sc["errors"]
        else:
            res.update(sc)

    if status != "ok":
        res["status"], res["decision"] = status, status
        left = int(c["max_attempts_per_run"]) - attempt
        res["retry_allowed"] = left > 0 and state["elapsed_s"] < float(c["max_run_seconds"]) - 1
    else:
        _, _, classes = read_frozen()
        build_errs = []
        try:
            if c.get("threshold") is not None and (c.get("submission") or {}).get("sample_csv"):
                if (run_dir / "submission.csv").exists():  # in threshold mode only the runner writes submission.csv
                    shutil.move(str(run_dir / "submission.csv"), str(adir / f"{attempt:02d}_candidate_submission.csv"))
                build_errs = build_submission(run_dir, c, classes, res["threshold"])
            export, export_errs = check_submission(run_dir, c, classes)
        except Exception as e:  # noqa: BLE001  empty/garbled test output -> EXPORT_FAILED with a reason, never a crash
            export, export_errs = "failed", [f"export error: {type(e).__name__}: {e}"]
        if build_errs:
            export, export_errs = "failed", build_errs + export_errs
        res["export_status"], res["export_errors"] = export, export_errs
        sign = 1 if DIRECTION[c["metric"]] == "max" else -1
        gain = None if best is None else sign * (res["selection_score"] - best["score"])
        res["improvement_vs_best"] = gain
        if best is None or gain > float(c["epsilon"]):
            decision = "KEEP" if export in ("ok", "pending_inputs") else "EXPORT_FAILED"
        else:
            decision = "REJECT"
        res["status"], res["decision"] = "ok", decision
        if decision == "EXPORT_FAILED":
            res["retry_allowed"] = (attempt < int(c["max_attempts_per_run"])
                                    and state["elapsed_s"] < float(c["max_run_seconds"]) - 1)
        if decision == "KEEP":
            BEST.mkdir(exist_ok=True)
            if (run_dir / "submission.csv").exists() and export == "ok":
                shutil.copy2(run_dir / "submission.csv", BEST / "submission.tmp")
                os.replace(BEST / "submission.tmp", BEST / "submission.csv")
            write_json(BEST / "current.json", {"run": run, "score": res["selection_score"], "metric": c["metric"],
                                               "threshold": res.get("threshold"),
                                               "export_status": export,
                                               "promoted_at": now().isoformat(timespec="seconds")})
    res["scored_at"] = now().isoformat(timespec="seconds")
    write_json(run_dir / "result.json", res)
    append_event({"event": "RUN", "run": run, "parent": parent, "attempt": attempt, "status": res["status"],
                  "decision": res["decision"], "score": res.get("selection_score"),
                  "best_after": (json.loads((BEST / "current.json").read_text())["run"]
                                 if (BEST / "current.json").exists() else None)})

    # one line the Engineer pastes to the Critic
    if res["status"] == "ok":
        bb = res["best_before"]
        before = f"{bb['score']:.4f} ({bb['run']})" if bb else "none"
        std = res["fold_std_descriptive_only"]
        print(f"RESULT {c['task_id']}/{run}: decision={res['decision']} {c['metric']}={res['selection_score']:.4f}"
              + (f" (fold std {std:.4f})" if std is not None else "")
              + (f" threshold={res['threshold']:.2f}" if res.get("threshold") is not None else "")
              + (f" at0.5={res['score_at_0.5_info_only']:.4f}" if "score_at_0.5_info_only" in res else "")
              + (f" P={res['precision']:.3f} R={res['recall']:.3f} flagged={res['predicted_positive_rate']:.1%}"
                 f" all_TRUE_f1={res['all_positive_f1_baseline']:.3f}" if "precision" in res else "")
              + f" best_before={before} export={res['export_status']} runtime={res['runtime_s']}s"
              + (f" retry_allowed={res['retry_allowed']}" if res["decision"] == "EXPORT_FAILED" else ""))
        for e in (res.get("export_errors") or [])[:5]:
            print("    export:", e)
    else:
        print(f"RESULT {c['task_id']}/{run}: decision={res['decision']} retry_allowed={res['retry_allowed']} "
              f"attempt={attempt}/{c['max_attempts_per_run']} used={res['cumulative_s']}s/{c['max_run_seconds']}s")
        for e in res.get("errors", [])[:8]:
            print("   ", e)
    sys.exit(0 if res["decision"] in SETTLED else 1)


# ---------------------------------------------------------------- read-only commands
def cmd_verify():
    problems = verify_frozen()
    print("OK: frozen files match FREEZE.json" if not problems else "MISMATCH: " + "; ".join(problems))
    sys.exit(0 if not problems else 1)


def cmd_rescore(run_arg):
    c = load_contract()
    problems = verify_frozen()
    if problems:
        print("WARNING frozen mismatch:", "; ".join(problems))
    sc = score_run(Path(run_arg).resolve(), c, write=False)
    print(json.dumps(sc, indent=2))


def cmd_summary():
    if not EVENTS.exists():
        die("no events yet")
    last = {}
    for line in EVENTS.read_text().splitlines():
        ev = json.loads(line)
        if ev.get("event") == "RUN":
            last[ev["run"]] = ev
    c = load_contract()
    sign = 1 if DIRECTION[c["metric"]] == "max" else -1
    rows, best = [], None
    for run in sorted(last):
        ev = last[run]
        _, hyp = parse_plan(RUNS / run)
        if ev["decision"] == "KEEP":
            best = ev["score"]
        rows.append((run, ev.get("parent") or "-", (hyp or "")[:70], ev["decision"],
                     "" if ev["score"] is None else f"{ev['score']:.4f}", "" if best is None else f"{best:.4f}"))
    md = [f"Metric: {c['metric']} ({'higher' if sign > 0 else 'lower'} is better), development score on the frozen split",
          "", "| run | parent | hypothesis | decision | score | best so far |", "|---|---|---|---|---|---|"]
    md += [f"| {' | '.join(r)} |" for r in rows]
    text = "\n".join(md)
    (ROOT / "summary.md").write_text(text + "\n")
    print(text)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = list(range(len(rows)))
        fig, ax = plt.subplots(figsize=(7, 3.5))
        kept = [(i, float(r[4])) for i, r in enumerate(rows) if r[4] and r[3] == "KEEP"]
        other = [(i, float(r[4])) for i, r in enumerate(rows) if r[4] and r[3] != "KEEP"]
        failed = [i for i, r in enumerate(rows) if not r[4]]
        bsf = [(i, float(r[5])) for i, r in enumerate(rows) if r[5]]
        if bsf:
            ax.step(*zip(*bsf), where="post", color="#2a6fdb", label="best so far")
        if kept:
            ax.scatter(*zip(*kept), color="#2a6fdb", zorder=3, label="kept")
        if other:
            ax.scatter(*zip(*other), facecolors="white", edgecolors="#2a6fdb", zorder=3, label="rejected")
        scored = kept + other
        if failed and scored:
            lo, hi = min(v for _, v in scored), max(v for _, v in scored)
            ax.scatter(failed, [lo - 0.08 * ((hi - lo) or 0.01)] * len(failed), marker="x", color="gray",
                       label="failed / invalid / timeout")
        ax.set_xticks(x, [r[0] for r in rows])
        ax.set_xlabel("run")
        ax.set_ylabel(c["metric"])
        ax.legend()
        fig.tight_layout()
        fig.savefig(ROOT / "summary.png", dpi=150)
        print(f"\nwrote {ROOT / 'summary.md'} and {ROOT / 'summary.png'}")
    except ImportError:
        print(f"\nwrote {ROOT / 'summary.md'} (matplotlib not installed, no chart)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    sub.add_parser("verify")
    r = sub.add_parser("run")
    r.add_argument("run_dir")
    rs = sub.add_parser("rescore")
    rs.add_argument("run_dir")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "init":
        cmd_init()
    elif a.cmd == "verify":
        cmd_verify()
    elif a.cmd == "run":
        cmd_run(a.run_dir)
    elif a.cmd == "rescore":
        cmd_rescore(a.run_dir)
    else:
        cmd_summary()


if __name__ == "__main__":
    main()
