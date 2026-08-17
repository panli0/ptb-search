#!/usr/bin/env python3
"""MATM ALFWorld natural terminal-run audit, corrected adapter v0.2.

Important v0.2 fixes after blinded/manual label audit:
1) MATM trajectory rows pair action_i with PRE-action observation_i; the result of
   action_i is observation_{i+1}. Failure/success checks are therefore shifted.
2) ALFWorld uses `move OBJ N to RECEPTACLE M`; successful moves to the goal
   receptacle resolve that object-subgoal and must be excluded from the
   "original direction remains unresolved" primary.
3) pick-two target class is read literally from the goal (`two <object>`), not
   inferred by substring matching over all entities.
UNKNOWN/ambiguous workstream steps are excluded rather than guessed.
"""
from __future__ import annotations

import argparse, ast, json, math, re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SEED = 20260817
BOOT_N = 100_000
K_LABELS = ["1", "2", "3", "4", "5", "6+"]
NO_RETRIEVAL = {"none", "no_retrieval", "no-retrieval", "no retrieval", "baseline"}
ENTITY_RE = re.compile(r"\b([a-z][a-z0-9_-]*)\s+(\d+)\b", re.I)
FAIL_RE = re.compile(
    r"\b(invalid|not a valid|cannot|can't|could not|unable|failed|failure|parse error|"
    r"structured parsing failed|nothing happens|not holding)\b",
    re.I,
)


def norm(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip().lower())


def parse_obj(x: Any) -> Any:
    if isinstance(x, (dict, list, tuple)):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    for f in (json.loads, ast.literal_eval):
        try:
            return f(s)
        except Exception:
            pass
    return x


def load_rows(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suf == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(obj if isinstance(obj, list) else obj.get("rows", obj.get("data", [])))
    if suf == ".csv":
        return pd.read_csv(path)
    raise SystemExit(f"Unsupported input: {path}")


def steps_from_row(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("trajectory", "steps", "history", "episode"):
        if key not in row or row[key] is None:
            continue
        obj = parse_obj(row[key])
        if isinstance(obj, dict):
            for k2 in ("trajectory", "steps", "history"):
                if isinstance(obj.get(k2), list):
                    obj = obj[k2]
                    break
        if isinstance(obj, (list, tuple)):
            return [parse_obj(x) for x in obj if isinstance(parse_obj(x), dict)]
    return []


def field(step: Dict[str, Any], *keys: str) -> str:
    for k in keys:
        if k in step and step[k] not in (None, ""):
            v = step[k]
            return norm(json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v)
    return ""


def numeric(step: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k in step:
            try:
                v = float(step[k])
                if math.isfinite(v):
                    return v
            except Exception:
                pass
    return None


def result_text(steps: List[Dict[str, Any]], i: int) -> str:
    """MATM stores action_i result as observation_{i+1}."""
    if i + 1 < len(steps):
        return field(steps[i + 1], "observation", "obs", "result")
    return ""


def action_failed(steps: List[Dict[str, Any]], i: int) -> bool:
    reason = field(steps[i], "reasoning", "explanation", "thought")
    return bool(FAIL_RE.search(reason) or FAIL_RE.search(result_text(steps, i)))


def is_completed(step: Dict[str, Any]) -> bool:
    for k in ("isCompleted", "is_completed", "done", "completed"):
        if k in step:
            v = step[k]
            if isinstance(v, str):
                return v.strip().lower() in {"1", "true", "yes", "done", "completed"}
            return bool(v)
    return False


def kbin(k: int) -> str:
    return "6+" if k >= 6 else str(k)


def exact_pick_target(goal: str) -> Optional[str]:
    m = re.search(r"\btwo\s+([a-z][a-z_-]*)\b", norm(goal))
    return m.group(1).rstrip("s") if m else None


def exact_goal_receptacle(goal: str) -> Optional[str]:
    m = re.search(r"\b(?:in|into|to)\s+([a-z][a-z_-]*)\.?$", norm(goal))
    return m.group(1).rstrip("s") if m else None


def target_ids(text: str, target: str) -> List[str]:
    out = []
    for base, num in ENTITY_RE.findall(norm(text)):
        if base.rstrip("s") == target.rstrip("s"):
            out.append(f"{target.rstrip('s')} {num}")
    return sorted(set(out))


def successful_take(action: str, result: str, target: str) -> Optional[str]:
    m = re.search(r"\b(?:take|pick up)\s+([a-z][a-z0-9_-]*)\s+(\d+)\b", norm(action))
    if not m:
        return None
    base, num = m.groups()
    if base.rstrip("s") != target.rstrip("s"):
        return None
    ok = f"you pick up the {base} {num}" in norm(result)
    return f"{target.rstrip('s')} {num}" if ok else None


def successful_move(action: str, result: str, target: str) -> Optional[Tuple[str, str]]:
    m = re.search(
        r"\bmove\s+([a-z][a-z0-9_-]*)\s+(\d+)\s+to\s+([a-z][a-z0-9_-]*)\s+(\d+)\b",
        norm(action),
    )
    if not m:
        return None
    ob, on, db, dn = m.groups()
    if ob.rstrip("s") != target.rstrip("s"):
        return None
    if f"you move the {ob} {on} to the {db} {dn}" not in norm(result):
        return None
    return f"{target.rstrip('s')} {on}", db.rstrip("s")


def pick_two_labels(goal: str, steps: List[Dict[str, Any]]) -> Tuple[List[Optional[str]], List[bool], List[bool], List[bool], Dict[str, Any]]:
    target = exact_pick_target(goal)
    dest = exact_goal_receptacle(goal)
    labels: List[Optional[str]] = [None] * len(steps)
    resolved_now = [False] * len(steps)
    failures = [False] * len(steps)
    unresolved_before = [False] * len(steps)
    audit: Dict[str, Any] = {"target_class": target, "goal_receptacle": dest, "ambiguous": 0}
    if not target:
        return labels, resolved_now, failures, unresolved_before, audit

    holding: Optional[str] = None
    placed: set[str] = set()
    for i, st in enumerate(steps):
        # Apply the previous action outcome to the state visible at decision i.
        if i > 0:
            pa = field(steps[i - 1], "action")
            pres = field(st, "observation", "obs", "result")
            picked = successful_take(pa, pres, target)
            if picked:
                holding = picked
                if dest and re.search(rf"\bfrom\s+{re.escape(dest)}\s+\d+\b", pa):
                    placed.discard(picked)
            moved = successful_move(pa, pres, target)
            if moved:
                obj, moved_dest = moved
                if holding == obj:
                    holding = None
                if dest and moved_dest == dest.rstrip("s"):
                    placed.add(obj)

        act = field(st, "action")
        reason = field(st, "reasoning", "explanation", "thought")
        aids = target_ids(act, target)
        rids = target_ids(reason, target)
        label: Optional[str] = None
        if len(aids) == 1:
            label = aids[0]
        elif holding:
            label = holding
        elif len(rids) == 1:
            label = rids[0]
        elif len(set(aids + rids)) > 1:
            audit["ambiguous"] += 1
        labels[i] = label
        unresolved_before[i] = bool(label and label not in placed)
        failures[i] = action_failed(steps, i)
        moved_now = successful_move(act, result_text(steps, i), target)
        if moved_now and dest and moved_now[1] == dest.rstrip("s"):
            resolved_now[i] = True

    audit["placed_objects_detected"] = len(placed)
    return labels, resolved_now, failures, unresolved_before, audit


def infer_light_target(goal: str) -> Optional[str]:
    g = norm(goal)
    for p in (
        r"look at (?:the )?([a-z][a-z_-]*) (?:under|with)",
        r"examine (?:the )?([a-z][a-z_-]*) (?:under|with)",
    ):
        m = re.search(p, g)
        if m:
            return m.group(1)
    return None


def light_labels(goal: str, steps: List[Dict[str, Any]]) -> Tuple[List[Optional[str]], List[bool], List[bool], List[bool], Dict[str, Any]]:
    """Secondary family only; ambiguous steps remain UNKNOWN."""
    target = infer_light_target(goal)
    labels: List[Optional[str]] = []
    resolved = [False] * len(steps)
    failures = [action_failed(steps, i) for i in range(len(steps))]
    unresolved = [True] * len(steps)
    audit = {"target_class": target, "ambiguous": 0}
    for i, st in enumerate(steps):
        act = field(st, "action")
        reason = field(st, "reasoning", "explanation", "thought")
        lamp = bool(re.search(r"\b(?:desklamp|lamp)\s*\d*\b", act))
        targ = bool(target and re.search(rf"\b{re.escape(target)}\s*\d*\b", act))
        label = None
        if lamp and not targ:
            label = "LAMP"
        elif targ and not lamp:
            label = "TARGET_OBJECT"
        else:
            # Only use the final planned sentence when the action itself is not diagnostic.
            tail = reason.split(".")[-2] if "." in reason else reason
            lp = bool(re.search(r"\b(?:desklamp|lamp)\b", tail))
            tp = bool(target and re.search(rf"\b{re.escape(target)}\b", tail))
            if lp ^ tp:
                label = "LAMP" if lp else "TARGET_OBJECT"
            elif lp or tp or lamp or targ:
                audit["ambiguous"] += 1
        labels.append(label)
        if label == "LAMP" and re.search(r"\b(?:use|turn on)\b", act) and "lamp" in act and not failures[i]:
            resolved[i] = True
        if is_completed(st):
            resolved[i] = True
    return labels, resolved, failures, unresolved, audit


def cumulative_progress(steps: List[Dict[str, Any]]) -> List[float]:
    out: List[float] = []
    cur = 0.0
    for st in steps:
        x = numeric(st, "score", "cumulative_score", "task_score")
        if x is not None:
            cur = max(cur, x)
        out.append(cur)
    return out


def extract_decisions(row: Dict[str, Any], uid: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    task_type = norm(row.get("task_type") or "")
    goal = str(row.get("goal") or "")
    steps = steps_from_row(row)
    if task_type == "pick_two_obj_and_place":
        labels, resolved, failures, unresolved, audit = pick_two_labels(goal, steps)
        family = "A_pick_two"
    elif task_type == "look_at_obj_in_light":
        labels, resolved, failures, unresolved, audit = light_labels(goal, steps)
        family = "B_light"
    else:
        return [], {"skip": "unsupported_task_type", "task_type": task_type, "n_steps": len(steps)}

    prog = cumulative_progress(steps)
    decisions = []
    for i in range(len(steps) - 1):
        cur, nxt = labels[i], labels[i + 1]
        if not cur or not nxt:
            continue
        k = 1
        j = i - 1
        while j >= 0 and labels[j] == cur:
            k += 1
            j -= 1
        start = i - k + 1
        recent_failed = any(failures[start : i + 1])
        clean = (not recent_failed) and (not resolved[i]) and bool(unresolved[i])
        end = min(len(steps) - 1, i + 5)
        future_progress = max(prog[i + 1 : end + 1], default=prog[i]) > prog[i] + 1e-12
        decisions.append(
            {
                "trajectory_id": uid,
                "task_id": str(row.get("task_id") or "UNKNOWN_TASK"),
                "model": str(row.get("model") or "UNKNOWN_MODEL"),
                "task_type": task_type,
                "family": family,
                "goal": goal,
                "step_index": i,
                "workstream": cur,
                "next_workstream": nxt,
                "K": k,
                "K_bin": kbin(k),
                "switch_next": int(cur != nxt),
                "clean_unresolved": int(clean),
                "recent_failed": int(recent_failed),
                "current_just_resolved": int(resolved[i]),
                "current_unresolved_before": int(unresolved[i]),
                "future5_any_progress": int(future_progress),
                "current_progress": float(prog[i]),
            }
        )
    audit.update(
        {
            "family": family,
            "task_type": task_type,
            "n_steps": len(steps),
            "labeled_steps": sum(x is not None for x in labels),
            "decisions": len(decisions),
        }
    )
    return decisions, audit


def two_level_bootstrap(dec: pd.DataFrame, n_boot: int, seed: int) -> Dict[str, Any]:
    d = dec[dec.clean_unresolved == 1].copy()
    out: Dict[str, Any] = {
        "n_decisions": int(len(d)),
        "n_trajectories": int(d.trajectory_id.nunique()),
        "n_tasks": int(d.task_id.nunique()),
        "n_models": int(d.model.nunique()),
    }
    out["by_K"] = {}
    for kb in K_LABELS:
        z = d[d.K_bin == kb]
        out["by_K"][kb] = {
            "switch_rate": float(z.switch_next.mean()) if len(z) else None,
            "decisions": int(len(z)),
            "switches": int(z.switch_next.sum()) if len(z) else 0,
            "trajectories": int(z.trajectory_id.nunique()),
            "tasks": int(z.task_id.nunique()),
        }

    def rates(kb: str) -> Dict[str, float]:
        z = d[d.K_bin == kb]
        if z.empty:
            return {}
        r = z.groupby(["task_id", "trajectory_id"], sort=False).switch_next.mean().reset_index()
        return r.groupby("task_id", sort=False).switch_next.mean().to_dict()

    r1, r6 = rates("1"), rates("6+")
    common = sorted(set(r1) & set(r6))
    out["paired_task_count_K1_K6"] = len(common)
    out["effect_K6_minus_K1_task_equal"] = float(np.mean([r6[t] - r1[t] for t in common])) if common else None
    if not common or n_boot <= 0:
        out["bootstrap_95_ci"] = None
        return out

    per: Dict[str, Dict[str, Dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    for (task, traj, kb), g in d[d.K_bin.isin(["1", "6+"])].groupby(["task_id", "trajectory_id", "K_bin"]):
        per[task][kb][traj] = g.switch_next.to_numpy(dtype=float)
    eligible = [t for t in common if per[t]["1"] and per[t]["6+"]]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ts = rng.choice(eligible, size=len(eligible), replace=True)
        diffs = []
        for t in ts:
            vals = {}
            for kb in ("1", "6+"):
                trajs = list(per[t][kb])
                sampled = rng.choice(trajs, size=len(trajs), replace=True)
                vals[kb] = float(np.mean([per[t][kb][tr].mean() for tr in sampled]))
            diffs.append(vals["6+"] - vals["1"])
        boots[b] = float(np.mean(diffs))
    out["bootstrap_95_ci"] = [float(x) for x in np.quantile(boots, [0.025, 0.975])]
    out["bootstrap_n"] = n_boot
    out["bootstrap_seed"] = seed
    return out


def consequence(dec: pd.DataFrame) -> Dict[str, Any]:
    d = dec[dec.clean_unresolved == 1]
    out = {}
    for sw in (0, 1):
        z = d[d.switch_next == sw]
        out["switch" if sw else "continue"] = {
            "n": int(len(z)),
            "future5_any_progress": float(z.future5_any_progress.mean()) if len(z) else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, default=Path("matm_natural_run_out"))
    ap.add_argument("--bootstrap", type=int, default=BOOT_N)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--retrieval", choices=["no_retrieval", "all"], default="no_retrieval")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_rows(args.input)
    source_n = len(df)
    filters: Dict[str, int] = {"source_rows": source_n}
    if "environment" in df.columns:
        df = df[df.environment.astype(str).str.lower().str.contains("alfworld", na=False)]
        filters["after_environment"] = len(df)
    if "source_type" in df.columns:
        df = df[df.source_type.astype(str).str.lower().eq("eval")]
        filters["after_source_type_eval"] = len(df)
    if args.retrieval == "no_retrieval" and "retrieval_strategy" in df.columns:
        df = df[df.retrieval_strategy.astype(str).str.lower().str.strip().isin(NO_RETRIEVAL)]
        filters["after_no_retrieval"] = len(df)

    decisions: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    for pos, (_, row) in enumerate(df.iterrows()):
        uid = f"row_{pos}"
        dec, aud = extract_decisions(row.to_dict(), uid)
        decisions.extend(dec)
        aud["trajectory_id"] = uid
        aud["task_id"] = str(row.get("task_id") or "UNKNOWN_TASK")
        audits.append(aud)

    D = pd.DataFrame(decisions)
    pd.DataFrame(audits).to_csv(args.out / "LABEL_RESOLUTION_AUDIT.csv", index=False)
    if D.empty:
        summary = {"status": "DATA_GATE_DIRECTION_RESOLUTION_OR_NO_SUPPORTED_TASKS", "filters": filters}
        (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    D.to_csv(args.out / "DECISIONS.csv", index=False)
    summary: Dict[str, Any] = {"status": "ANALYZED_V0_2", "adapter_version": "0.2", "filters": filters, "families": {}}
    for fam, g in D.groupby("family"):
        s = two_level_bootstrap(g, args.bootstrap, args.seed)
        s["future5_by_switch"] = consequence(g)
        k1 = s["by_K"]["1"]["trajectories"]
        k6 = s["by_K"]["6+"]["trajectories"]
        s["data_gate"] = "PASS" if s["n_trajectories"] >= 30 and k1 >= 20 and k6 >= 20 else "DATA_GATE_LOW_N_K6_OR_TRAJECTORIES"
        summary["families"][fam] = s

    (args.out / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# MATM 自然 terminal-run 验证 v0.2", "", f"输入 rows={source_n}; 过滤后 rows={len(df)}", ""]
    for fam, s in summary["families"].items():
        lines += [
            f"## {fam}",
            f"- clean decisions={s['n_decisions']}; trajectories={s['n_trajectories']}; tasks={s['n_tasks']}; models={s['n_models']}",
            f"- K6+-K1={s['effect_K6_minus_K1_task_equal']}; 95% CI={s['bootstrap_95_ci']}; gate={s['data_gate']}",
            "- K curve: " + ", ".join(f"{k}:{s['by_K'][k]['switch_rate']} (switch={s['by_K'][k]['switches']}, n={s['by_K'][k]['decisions']}, traj={s['by_K'][k]['trajectories']})" for k in K_LABELS),
            f"- future5: {json.dumps(s['future5_by_switch'], ensure_ascii=False)}",
            "",
        ]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
