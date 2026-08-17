#!/usr/bin/env python3
"""Natural run-length / switching audit for MATM ALFWorld trajectories.

Frozen adapter v0.1. The script is intentionally conservative: UNKNOWN workstream
labels are excluded rather than guessed. It supports JSON/JSONL/CSV directly and
Parquet when a parquet engine (pyarrow/fastparquet) is available.
"""
from __future__ import annotations

import argparse, ast, json, math, re, sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

SEED = 20260817
BOOT_N = 100_000
K_LABELS = ["1", "2", "3", "4", "5", "6+"]
NO_RETRIEVAL = {"none", "no_retrieval", "no-retrieval", "no retrieval", "baseline"}
ERROR_PATTERNS = re.compile(
    r"\b(invalid|not a valid|cannot|can't|could not|unable|failed|failure|parse error|structured parsing failed)\b",
    re.I,
)
ENTITY_RE = re.compile(r"\b([a-z][a-z0-9_-]*)\s+(\d+)\b", re.I)

RECEPTACLE_WORDS = {
    "countertop","coffeetable","diningtable","sidetable","desk","dresser","drawer","cabinet",
    "garbagecan","toilet","microwave","fridge","refrigerator","sinkbasin","bathtubbasin",
    "shelf","safe","bed","armchair","sofa","couch","table","counter","chair"
}


def norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


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
        try:
            return pd.read_parquet(path)
        except ImportError as e:
            raise SystemExit("Parquet engine missing. Install pyarrow/fastparquet or convert once to JSONL.") from e
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
        if key in row and row[key] is not None:
            obj = parse_obj(row[key])
            if isinstance(obj, dict):
                for k2 in ("trajectory", "steps", "history"):
                    if isinstance(obj.get(k2), list):
                        obj = obj[k2]; break
            if isinstance(obj, (list, tuple)):
                out=[]
                for x in obj:
                    x=parse_obj(x)
                    if isinstance(x, dict): out.append(x)
                return out
    return []


def field(step: Dict[str, Any], *keys: str) -> str:
    for k in keys:
        if k in step and step[k] not in (None, ""):
            v = step[k]
            if isinstance(v, (dict, list, tuple)):
                return norm(json.dumps(v, ensure_ascii=False))
            return norm(v)
    return ""


def numeric(step: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k in step:
            try:
                v=float(step[k])
                if math.isfinite(v): return v
            except Exception: pass
    return None


def is_failed_step(step: Dict[str, Any]) -> bool:
    for k in ("actionSuccess", "action_success", "success"):
        if k in step and isinstance(step[k], (bool, np.bool_)):
            return not bool(step[k])
    txt = " ".join([field(step,"observation","obs","result"), field(step,"reasoning","explanation","thought")])
    return bool(ERROR_PATTERNS.search(txt))


def is_completed(step: Dict[str, Any]) -> bool:
    for k in ("isCompleted","is_completed","done","completed"):
        if k in step:
            v=step[k]
            if isinstance(v, str): return v.strip().lower() in {"1","true","yes","done","completed"}
            return bool(v)
    return False


def entities(text: str) -> List[Tuple[str,str]]:
    return [(m.group(1).lower(), m.group(2)) for m in ENTITY_RE.finditer(text or "")]


def infer_pick_two_target_class(goal: str, steps: List[Dict[str,Any]]) -> Optional[str]:
    goaln = norm(goal).replace(" ", "")
    ids=defaultdict(set)
    for st in steps:
        txt=" ".join([field(st,"action"),field(st,"observation","obs","result"),field(st,"reasoning","explanation","thought"),field(st,"inventory")])
        for base,num in entities(txt): ids[base].add(num)
    candidates=[]
    for base, nums in ids.items():
        b=base.replace("_","").replace("-","")
        if len(nums)>=2 and b in goaln and base not in RECEPTACLE_WORDS:
            candidates.append((base,len(nums)))
    if not candidates:
        m=re.search(r"\btwo\s+([a-z][a-z_-]+)", norm(goal))
        if m: return m.group(1).rstrip("s")
        return None
    candidates.sort(key=lambda x:(-x[1], x[0]))
    return candidates[0][0]


def pick_two_labels(goal: str, steps: List[Dict[str,Any]]) -> Tuple[List[Optional[str]],List[bool],Dict[str,Any]]:
    target=infer_pick_two_target_class(goal,steps)
    labels=[]; just_completed=[]; holding:Optional[str]=None
    placed=set(); audit={"target_class":target,"ambiguous":0}
    if not target:
        return [None]*len(steps), [False]*len(steps), audit
    variants={target, target.rstrip("s"), target.rstrip("s")+"s"}
    def target_ids(txt:str)->List[str]:
        out=[]
        for b,n in entities(txt):
            if b in variants or b.rstrip("s")==target.rstrip("s"):
                out.append(f"{b.rstrip('s')} {n}")
        return sorted(set(out))
    for st in steps:
        act=field(st,"action")
        reason=field(st,"reasoning","explanation","thought")
        inv=field(st,"inventory")
        act_ids=target_ids(act); inv_ids=target_ids(inv); reason_ids=target_ids(reason)
        label=None
        if len(act_ids)==1: label=act_ids[0]
        elif len(inv_ids)==1: label=inv_ids[0]
        elif holding: label=holding
        elif len(reason_ids)==1: label=reason_ids[0]
        else:
            if len(set(act_ids+inv_ids+reason_ids))>1: audit["ambiguous"]+=1
        if label and re.search(r"\b(pick up|take|get)\b",act): holding=label
        complete=False
        if label and re.search(r"\b(put|place|drop)\b",act) and not is_failed_step(st):
            placed.add(label); complete=True
            if holding==label: holding=None
        labels.append(label); just_completed.append(complete)
    audit["placed_objects_detected"]=len(placed)
    return labels,just_completed,audit


def infer_light_target(goal:str)->Optional[str]:
    g=norm(goal)
    pats=[r"look at (?:the )?([a-z][a-z_-]*) (?:under|with)",r"examine (?:the )?([a-z][a-z_-]*) (?:under|with)"]
    for p in pats:
        m=re.search(p,g)
        if m: return m.group(1)
    m=re.search(r"\b(?:look at|examine)\s+(?:the |a |an )?([a-z][a-z_-]*)",g)
    return m.group(1) if m else None


def light_labels(goal:str,steps:List[Dict[str,Any]])->Tuple[List[Optional[str]],List[bool],Dict[str,Any]]:
    target=infer_light_target(goal)
    labels=[]; completed=[]; lamp_done=False; audit={"target_class":target,"ties":0}
    for st in steps:
        act=field(st,"action")
        reason=field(st,"reasoning","explanation","thought")
        r=reason
        r=re.sub(r"^(let'?s think step by step\.\s*)", "", r)
        r=re.sub(r"^the goal is [^.]*\.\s*", "", r)
        lamp=0; targ=0
        if "desklamp" in act or re.search(r"\b(lamp)\s+\d+\b",act): lamp+=4
        if target and target in act: targ+=4
        if re.search(r"\b(find|locate|search|go|get|retrieve|turn on|use|inspect).*\b(desklamp|lamp)\b",r): lamp+=2
        if target and re.search(rf"\b(find|locate|search|go|get|retrieve|inspect|examine|look).*\b{re.escape(target)}\b",r): targ+=2
        if "desklamp" in r: lamp+=1
        if target and target in r: targ+=1
        label=None
        if lamp>targ: label="LAMP"
        elif targ>lamp: label="TARGET_OBJECT"
        elif lamp or targ: audit["ties"]+=1
        comp=False
        obs=field(st,"observation","obs","result")
        if label=="LAMP" and not is_failed_step(st) and (re.search(r"\b(use|turn on)\b",act) and "lamp" in act):
            comp=True; lamp_done=True
        if is_completed(st): comp=True
        labels.append(label); completed.append(comp)
    audit["lamp_completed_detected"]=lamp_done
    return labels,completed,audit


def cumulative_progress_series(task_type:str,steps:List[Dict[str,Any]],just_completed:List[bool])->List[float]:
    scores=[numeric(st,"score","cumulative_score","task_score") for st in steps]
    vals=[x for x in scores if x is not None]
    if len(vals)>=2:
        ok=True; last=-float("inf")
        for x in scores:
            if x is None: continue
            if x+1e-12<last: ok=False; break
            last=x
        if ok:
            out=[]; cur=0.0
            for x in scores:
                if x is not None: cur=x
                out.append(cur)
            return out
    cur=0.0; out=[]
    for st,comp in zip(steps,just_completed):
        if comp: cur+=1.0
        if is_completed(st): cur=max(cur,2.0 if task_type=="pick_two_obj_and_place" else 1.0)
        out.append(cur)
    return out


def kbin(k:int)->str: return "6+" if k>=6 else str(k)


def extract_decisions(row:Dict[str,Any],traj_uid:str)->Tuple[List[Dict[str,Any]],Dict[str,Any]]:
    task_type=norm(row.get("task_type") or row.get("task_name") or "")
    goal=str(row.get("goal") or row.get("task_goal") or row.get("task_description") or "")
    steps=steps_from_row(row)
    if task_type=="pick_two_obj_and_place":
        labels,just_completed,audit=pick_two_labels(goal,steps); family="A_pick_two"
    elif task_type=="look_at_obj_in_light":
        labels,just_completed,audit=light_labels(goal,steps); family="B_light"
    else:
        return [],{"skip":"unsupported_task_type","task_type":task_type,"n_steps":len(steps)}
    prog=cumulative_progress_series(task_type,steps,just_completed)
    decisions=[]
    for i in range(len(steps)-1):
        cur=labels[i]; nxt=labels[i+1]
        if not cur or not nxt: continue
        if is_completed(steps[i]): continue
        k=1; j=i-1
        while j>=0 and labels[j]==cur:
            k+=1; j-=1
        run_start=i-k+1
        recent_failed=any(is_failed_step(steps[t]) for t in range(run_start,i+1))
        resolved=bool(just_completed[i])
        clean=(not recent_failed) and (not resolved)
        switch=(nxt!=cur)
        end=min(len(steps)-1,i+5)
        p0=prog[i]; p1=max(prog[i+1:end+1]) if end>=i+1 else p0
        future_completed=any(is_completed(steps[t]) for t in range(i+1,end+1))
        decisions.append({
            "trajectory_id":traj_uid,
            "task_id":str(row.get("task_id") or row.get("variation_id") or row.get("game_id") or "UNKNOWN_TASK"),
            "model":str(row.get("model") or row.get("agent") or row.get("agent_type") or "UNKNOWN_MODEL"),
            "task_type":task_type,"family":family,"goal":goal,"step_index":i,
            "workstream":cur,"next_workstream":nxt,"K":k,"K_bin":kbin(k),
            "switch_next":int(switch),"clean_unresolved":int(clean),"recent_failed":int(recent_failed),
            "current_just_completed":int(resolved),"future5_any_progress":int((p1>p0+1e-12) or future_completed),
            "future5_progress_count":float(max(0.0,p1-p0)),"future5_completed":int(future_completed),
            "current_progress":float(p0),"n_steps":len(steps),
        })
    audit.update({"family":family,"task_type":task_type,"n_steps":len(steps),"labeled_steps":sum(x is not None for x in labels),"decisions":len(decisions)})
    return decisions,audit


def two_level_bootstrap(dec:pd.DataFrame,n_boot:int=BOOT_N,seed:int=SEED)->Dict[str,Any]:
    d=dec[dec.clean_unresolved==1].copy()
    out={"n_decisions":int(len(d)),"n_trajectories":int(d.trajectory_id.nunique()),"n_tasks":int(d.task_id.nunique()),"n_models":int(d.model.nunique())}
    rates={}
    for kb in K_LABELS:
        z=d[d.K_bin==kb]
        rates[kb]={"switch_rate":float(z.switch_next.mean()) if len(z) else None,"decisions":int(len(z)),"trajectories":int(z.trajectory_id.nunique()),"tasks":int(z.task_id.nunique())}
    out["by_K"]=rates
    def task_rates(data:pd.DataFrame,kb:str)->Dict[str,float]:
        z=data[data.K_bin==kb]
        if z.empty: return {}
        rr=z.groupby(["task_id","trajectory_id"],sort=False).switch_next.mean().reset_index()
        return rr.groupby("task_id",sort=False).switch_next.mean().to_dict()
    r1=task_rates(d,"1"); r6=task_rates(d,"6+")
    common=sorted(set(r1)&set(r6))
    out["paired_task_count_K1_K6"]=len(common)
    out["effect_K6_minus_K1_task_equal"]=(float(np.mean([r6[t]-r1[t] for t in common])) if common else None)
    if not common or n_boot<=0:
        out["bootstrap_95_ci"]=None; return out
    per=defaultdict(lambda:defaultdict(dict))
    for (task,traj,kb),g in d[d.K_bin.isin(["1","6+"])].groupby(["task_id","trajectory_id","K_bin"]):
        per[task][kb][traj]=g.switch_next.to_numpy(dtype=float)
    eligible=[t for t in common if per[t]["1"] and per[t]["6+"]]
    if not eligible:
        out["bootstrap_95_ci"]=None; return out
    rng=np.random.default_rng(seed); boots=np.empty(n_boot,dtype=float)
    for b in range(n_boot):
        ts=rng.choice(eligible,size=len(eligible),replace=True)
        diffs=[]
        for t in ts:
            vals={}
            for kb in ("1","6+"):
                trajs=list(per[t][kb])
                sampled=rng.choice(trajs,size=len(trajs),replace=True)
                trates=[]
                for tr in sampled:
                    arr=per[t][kb][tr]
                    trates.append(float(arr.mean()))
                vals[kb]=float(np.mean(trates))
            diffs.append(vals["6+"]-vals["1"])
        boots[b]=float(np.mean(diffs))
    out["bootstrap_95_ci"]=[float(x) for x in np.quantile(boots,[.025,.975])]
    out["bootstrap_n"]=n_boot; out["bootstrap_seed"]=seed
    return out


def consequence_summary(dec:pd.DataFrame)->Dict[str,Any]:
    d=dec[dec.clean_unresolved==1].copy()
    ans={}
    for sw in (0,1):
        z=d[d.switch_next==sw]
        ans["switch" if sw else "continue"]={
            "n":int(len(z)),"future5_any_progress":float(z.future5_any_progress.mean()) if len(z) else None,
            "future5_progress_count":float(z.future5_progress_count.mean()) if len(z) else None,
            "future5_completed":float(z.future5_completed.mean()) if len(z) else None,
        }
    return ans


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input",type=Path)
    ap.add_argument("--out",type=Path,default=Path("matm_natural_run_out"))
    ap.add_argument("--bootstrap",type=int,default=BOOT_N)
    ap.add_argument("--seed",type=int,default=SEED)
    ap.add_argument("--retrieval",default="no_retrieval",choices=["no_retrieval","all"])
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    df=load_rows(args.input)
    source_n=len(df); filters={"source_rows":source_n}
    if "environment" in df.columns:
        df=df[df.environment.astype(str).str.lower().str.contains("alfworld",na=False)]; filters["after_environment"]=len(df)
    if "source_type" in df.columns:
        df=df[df.source_type.astype(str).str.lower().eq("eval")]; filters["after_source_type_eval"]=len(df)
    if args.retrieval=="no_retrieval" and "retrieval_strategy" in df.columns:
        m=df.retrieval_strategy.astype(str).str.lower().str.strip().isin(NO_RETRIEVAL)
        df=df[m]; filters["after_no_retrieval"]=len(df)
    all_dec=[]; audits=[]
    for idx,row in df.iterrows():
        r=row.to_dict(); uid=str(r.get("trajectory_id") or r.get("run_id") or r.get("episode_id") or f"row_{idx}")
        dec,aud=extract_decisions(r,uid); all_dec.extend(dec); aud["trajectory_id"]=uid; audits.append(aud)
    D=pd.DataFrame(all_dec)
    A=pd.DataFrame(audits)
    A.to_csv(args.out/"LABEL_RESOLUTION_AUDIT.csv",index=False)
    if D.empty:
        summary={"status":"DATA_GATE_DIRECTION_RESOLUTION_OR_NO_SUPPORTED_TASKS","filters":filters,"audits":audits[:20]}
        (args.out/"SUMMARY.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
        print(json.dumps(summary,indent=2,ensure_ascii=False)); return
    D.to_csv(args.out/"DECISIONS.csv",index=False)
    summary={"status":"ANALYZED","filters":filters,"families":{}}
    for fam,g in D.groupby("family"):
        s=two_level_bootstrap(g,args.bootstrap,args.seed)
        s["future5_by_switch"]=consequence_summary(g)
        k1=s["by_K"]["1"]["trajectories"]; k6=s["by_K"]["6+"]["trajectories"]
        s["data_gate"]="PASS" if s["n_trajectories"]>=30 and k1>=20 and k6>=20 else "DATA_GATE_LOW_N_K6_OR_TRAJECTORIES"
        summary["families"][fam]=s
    (args.out/"SUMMARY.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    lines=["# MATM 自然轨迹验证结果\n",f"输入 rows: {source_n}; 过滤后 rows: {len(df)}\n"]
    for fam,s in summary["families"].items():
        lines.append(f"## {fam}\n")
        lines.append(f"- clean decisions={s['n_decisions']}; trajectories={s['n_trajectories']}; tasks={s['n_tasks']}; models={s['n_models']}\n")
        lines.append(f"- K6+-K1={s['effect_K6_minus_K1_task_equal']}; 95% CI={s['bootstrap_95_ci']}; gate={s['data_gate']}\n")
        lines.append("- K curve: "+", ".join(f"{k}:{s['by_K'][k]['switch_rate']} (n={s['by_K'][k]['decisions']})" for k in K_LABELS)+"\n")
        lines.append(f"- future5 by switch: {json.dumps(s['future5_by_switch'],ensure_ascii=False)}\n")
    (args.out/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=="__main__": main()
