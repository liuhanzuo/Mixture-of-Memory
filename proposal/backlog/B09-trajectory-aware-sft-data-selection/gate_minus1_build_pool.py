#!/opt/conda/envs/torch-base/bin/python
"""B09 GATE -1 step 2: emit the FLAT CANDIDATE POOL with PROPOSAL section 1's columns.

Decomposition is the one PRE-REGISTERED in B09_GATE_MINUS1_PREREG.md sec 3, which was
written BEFORE this parquet was parsed:

    GROUP (partition/cap unit)   g(i) = traj_id       -- one agent rollout
    ROW   (selectable unit)      one DECISION TURN extracted from `messages`
    PARENT TASK (split unit)     instance_id          -- 1..21 rollouts, measured
    HOLDOUT UNIT                 repo_family = instance_id.split('.',1)[0]

WHY the row is a TURN and not a parquet row: traj_id is unique in all 24100 parquet rows
(measured, 0 duplicates). If g(i)=traj_id AND row=parquet row, every group has size 1,
which makes the parent-multiplicity cap vacuous, sibling redundancy undefined, and
trajectory-stratified constraint-matched random identical to plain random -- i.e.
kill_criteria #1 would fire for a purely structural reason.

PROPOSAL section 1 required columns:
  g(i)=traj_id | instance_id | t(i)=message_type | step index | resolved | tool_name |
  c_i = assistant target-token count
plus, per the PREREG: repo_family, task_strategy, model, and n_turns for the group.

c_i (target-token count): the corpus ships no tokenizer, and installing one is a
different dependency decision. So TWO surrogates are emitted and BOTH are labelled:
  c_i_chars  = len of the assistant turn's serialised loss-bearing text (exact, no dep)
  c_i_tokest = c_i_chars / 3.6 (a CHARS-PER-TOKEN HEURISTIC, NOT a tokenizer count)
Anything that needs a real token budget must re-derive c_i with the student's tokenizer.
This is recorded as a named limitation, not silently substituted.

Output: parquet (pyarrow) + a summary JSON. Reads only the 8 shards; 0 GPU.
"""
import json, sys, glob, collections, re, datetime
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
SHARDS = sorted(glob.glob(f"{ROOT}/data/agent_traj/tool-*.parquet"))
OUT_POOL = sys.argv[1]
OUT_SUM = sys.argv[2]
assert len(SHARDS) == 8

CHARS_PER_TOKEN = 3.6      # heuristic, declared

def repo_family(iid):
    """PRE-REGISTERED: everything before the FIRST '.'; whole string if no '.'."""
    return iid.split(".", 1)[0] if "." in iid else iid

def task_strategy(iid):
    """CORRECTED 2026-08-17 after measurement; see STATUS.json prereg_rule_correction_20260817.

    The PREREG (sec 2.3) guessed `instance_id` = `<repo>.<strategy-suffix>` and read the
    strategy from the segment after the FIRST dot. MEASURED: the real form is
    `<owner>__<repo>.<commit8>.<strategy>` -- 5177/5215 ids have exactly 2 dots, 38 have 3.
    The blind rule therefore returned "other" for 9116/9116 trajectories, i.e. it was a
    CONSTANT and the secondary axis was degenerate. The strategy lives in segment [2].

    repo_family (the PRIMARY axis, sec 2.1) is UNAFFECTED: split-on-first-dot yields
    `owner__repo`, which is exactly the intended unit (128 families measured).
    """
    parts = iid.split(".")
    seg = parts[2] if len(parts) > 2 else ""
    for k in ("lm_rewrite", "func_pm", "func_basic", "combine_file", "combine_module"):
        if seg.startswith(k):
            return k
    if seg.startswith("pr_"):
        return "pr"
    if seg.isdigit():
        return "numeric"
    return "other:" + seg[:20]

cols = {k: [] for k in ("traj_id", "instance_id", "repo_family", "task_strategy",
                        "model", "resolved", "step_index", "action_ordinal",
                        "message_type", "tool_name", "n_tool_calls",
                        "c_i_chars", "c_i_tokest", "has_thought",
                        "n_turns_in_traj", "n_actions_in_traj",
                        "position_frac", "is_final_action")}

fam_ct = collections.Counter()
strat_ct = collections.Counter()
other_suffixes = collections.Counter()
nodot = 0
tool_ct = collections.Counter()
mt_ct = collections.Counter()
turns_per_traj = []
n_traj = 0
n_rows_seen = 0
n_bare = 0
fam_traj = collections.defaultdict(set)
fam_inst = collections.defaultdict(set)
inst_roll = collections.Counter()
inst_resolved_mix = collections.defaultdict(set)

for sh in SHARDS:
    for b in pq.ParquetFile(sh).iter_batches(
            batch_size=128,
            columns=["messages", "instance_id", "resolved", "model", "traj_id"]):
        for m, iid, rv, md, tid in zip(b.column("messages").to_pylist(),
                                       b.column("instance_id").to_pylist(),
                                       b.column("resolved").to_pylist(),
                                       b.column("model").to_pylist(),
                                       b.column("traj_id").to_pylist()):
            n_rows_seen += 1
            turns = json.loads(m)
            structured = any(isinstance(t, dict) and "message_type" in t for t in turns) and \
                         any(isinstance(t, dict) and "tool_calls" in t for t in turns)
            if not structured:
                n_bare += 1
                continue
            n_traj += 1
            fam = repo_family(iid)
            strat = task_strategy(iid)
            if "." not in iid:
                nodot += 1
            if strat == "other" and "." in iid:
                other_suffixes[iid.split(".", 1)[1][:24]] += 1
            fam_ct[fam] += 1
            strat_ct[strat] += 1
            fam_traj[fam].add(tid)
            fam_inst[fam].add(iid)
            inst_roll[iid] += 1
            inst_resolved_mix[iid].add(bool(rv))

            actions = [(i, t) for i, t in enumerate(turns)
                       if isinstance(t, dict) and t.get("message_type") == "action"]
            turns_per_traj.append(len(turns))
            n_act = len(actions)
            for ordinal, (i, t) in enumerate(actions):
                tcs = t.get("tool_calls") or []
                names = []
                for c in tcs:
                    if isinstance(c, dict):
                        fn = c.get("function") or {}
                        if isinstance(fn, dict) and fn.get("name"):
                            names.append(fn["name"])
                tn = names[0] if names else ""
                if tn:
                    tool_ct[tn] += 1
                mt_ct[t.get("message_type")] += 1
                # loss-bearing text of the assistant turn: thought + content + call args
                parts = []
                for k in ("thought", "content", "action"):
                    v = t.get(k)
                    if isinstance(v, str):
                        parts.append(v)
                for c in tcs:
                    if isinstance(c, dict):
                        fn = c.get("function") or {}
                        if isinstance(fn, dict):
                            a = fn.get("arguments")
                            if isinstance(a, str):
                                parts.append(a)
                            elif a is not None:
                                parts.append(json.dumps(a))
                chars = sum(len(p) for p in parts)
                cols["traj_id"].append(tid)
                cols["instance_id"].append(iid)
                cols["repo_family"].append(fam)
                cols["task_strategy"].append(strat)
                cols["model"].append(md)
                cols["resolved"].append(bool(rv))
                cols["step_index"].append(i)
                cols["action_ordinal"].append(ordinal)
                cols["message_type"].append(t.get("message_type"))
                cols["tool_name"].append(tn)
                cols["n_tool_calls"].append(len(tcs))
                cols["c_i_chars"].append(chars)
                cols["c_i_tokest"].append(round(chars / CHARS_PER_TOKEN, 2))
                cols["has_thought"].append(bool(t.get("thought")))
                cols["n_turns_in_traj"].append(len(turns))
                cols["n_actions_in_traj"].append(n_act)
                cols["position_frac"].append(round((ordinal + 1) / n_act, 6) if n_act else 0.0)
                cols["is_final_action"].append(ordinal == n_act - 1)

# PRE-REGISTERED K=5 fold assignment (sec 2.2): greedy by descending trajectory count,
# seed 0. Materialised as a COLUMN so no downstream consumer can silently re-derive a
# different partition -- the holdout is part of the artifact, not of a convention.
_fam_traj_n = {f: len(v) for f, v in fam_traj.items()}
_K = 5
_folds = [[] for _ in range(_K)]
_load = [0] * _K
for _f, _sz in sorted(_fam_traj_n.items(), key=lambda kv: (-kv[1], kv[0])):
    _j = _load.index(min(_load))
    _folds[_j].append(_f)
    _load[_j] += _sz
_fam2fold = {f: j for j, fs in enumerate(_folds) for f in fs}
cols["prereg_fold"] = [_fam2fold[f] for f in cols["repo_family"]]

tbl = pa.table({k: pa.array(v) for k, v in cols.items()})
pq.write_table(tbl, OUT_POOL, compression="zstd")

U = tbl.num_rows
G = n_traj
# rollout siblings with DIFFERING resolved outcome -- what section 5 branch-verified credit needs
mixed_inst = sum(1 for v in inst_resolved_mix.values() if len(v) > 1)
multi_inst = sum(1 for v in inst_roll.values() if v > 1)

# Gini over rows-per-group (next_gate[1] wants this; computing it here is free)
def gini(xs):
    xs = sorted(xs)
    n = len(xs)
    s = sum(xs)
    if n == 0 or s == 0:
        return None
    cum = 0.0
    for i, x in enumerate(xs, 1):
        cum += i * x
    return round((2 * cum) / (n * s) - (n + 1) / n, 6)

rows_per_traj = collections.Counter()
for t in cols["traj_id"]:
    rows_per_traj[t] += 1
rpt = list(rows_per_traj.values())

fam_sizes = _fam_traj_n
top = sorted(fam_sizes.items(), key=lambda kv: -kv[1])
K, folds, load = _K, _folds, _load

summary = {
 "_what": "B09 GATE -1 next_action item 2: the FLAT CANDIDATE POOL, emitted with PROPOSAL section 1's columns under the decomposition PRE-REGISTERED in B09_GATE_MINUS1_PREREG.md sec 3.",
 "computed_at": datetime.datetime.now().astimezone().isoformat(),
 "pool_path": OUT_POOL,
 "pool_rows_U": U,
 "pool_groups_G_traj_id": G,
 "parent_tasks_instance_id": len(inst_roll),
 "repo_families": len(fam_sizes),
 "columns": list(cols.keys()),
 "source_rows_read": n_rows_seen,
 "source_rows_excluded_bare": n_bare,
 "exclusion_rate_pct": round(100 * n_bare / n_rows_seen, 4),
 "rows_per_group": {"min": min(rpt), "max": max(rpt),
                    "mean": round(sum(rpt) / len(rpt), 4),
                    "gini": gini(rpt)},
 "turns_per_traj": {"min": min(turns_per_traj), "max": max(turns_per_traj),
                    "mean": round(sum(turns_per_traj) / len(turns_per_traj), 4)},
 "instance_rollouts": {"n_instances": len(inst_roll),
                       "n_with_more_than_1_rollout": multi_inst,
                       "max": max(inst_roll.values()),
                       "n_with_MIXED_resolved_outcome": mixed_inst},
 "tool_name_counts": dict(tool_ct.most_common()),
 "message_type_counts": dict(mt_ct),
 "task_strategy_traj_counts": dict(strat_ct.most_common()),
 "instance_ids_without_a_dot": nodot,
 "other_suffix_samples": dict(other_suffixes.most_common(10)),
 "repo_family_top20_by_traj": top[:20],
 "repo_family_size_hist": dict(sorted(collections.Counter(fam_sizes.values()).items())),
 "prereg_K5_folds": {f"fold{i}": {"n_families": len(folds[i]), "n_traj": load[i]}
                     for i in range(K)},
 "prereg_K5_fold_members_head": {f"fold{i}": sorted(folds[i])[:8] for i in range(K)},
 "c_i_note": ("c_i_chars is an EXACT character count of the assistant turn's loss-bearing text "
              "(thought + content + action + tool-call arguments). c_i_tokest = c_i_chars/3.6 is a "
              "HEURISTIC, NOT a tokenizer count. Any fixed-target-token budget must re-derive c_i "
              "with the student model's own tokenizer."),
 "c_i_chars_summary": {"min": min(cols["c_i_chars"]), "max": max(cols["c_i_chars"]),
                       "mean": round(sum(cols["c_i_chars"]) / U, 2),
                       "total": sum(cols["c_i_chars"])},
 "gpu_used": "none",
}
with open(OUT_SUM, "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(json.dumps({k: summary[k] for k in
                  ("pool_rows_U", "pool_groups_G_traj_id", "parent_tasks_instance_id",
                   "repo_families", "exclusion_rate_pct", "rows_per_group",
                   "instance_rollouts", "tool_name_counts", "task_strategy_traj_counts",
                   "prereg_K5_folds", "repo_family_size_hist", "c_i_chars_summary")},
                 indent=2))
