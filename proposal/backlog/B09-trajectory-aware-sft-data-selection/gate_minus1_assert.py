#!/opt/conda/envs/torch-base/bin/python
"""B09 GATE -1 step 1: ASSERT the three provenance numbers on the DOWNLOADED parquet.

next_action item 1 (GATE_MINUS1_FEASIBILITY_20260816.md sec 6):
  n_rows == 24100 ; resolved true/false == 9427/14673 ;
  rows with BOTH message_type and tool_calls == 9116.
  "If any differs, stop -- the HF index and the parquet disagree."

The item does NOT say which method is canonical for the 9116. The HF figure came from
`/filter "messages" LIKE '%tool_calls%'`, i.e. a SUBSTRING test on the raw JSON string.
`message_type` and `tool_calls` are NOT columns -- /info features are exactly 6
(messages, instance_id, resolved, model, traj_id, patch) and `messages` is a JSON-encoded
STRING. So this script computes the count BOTH ways and reports both:

  substring : marker in the raw messages string   (reproduces the HF LIKE semantics)
  parsed    : json.loads, then any(marker in turn) for a dict turn

A disagreement is itself a result and is reported, not hidden. Writes JSON to stdout path.
"""
import json, sys, glob, collections
import pyarrow.parquet as pq

SHARDS = sorted(glob.glob(
    "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/data/agent_traj/tool-*.parquet"))
OUT = sys.argv[1]

assert len(SHARDS) == 8, SHARDS

n_rows = 0
resolved_ct = collections.Counter()
model_ct = collections.Counter()
sub_mt = sub_tc = sub_both = sub_thought = 0
par_mt = par_tc = par_both = par_thought = 0
disagree_mt = disagree_tc = 0
parse_fail = 0
traj_ct = collections.Counter()
inst_ct = collections.Counter()
# structure x model / x resolved crosstabs (PREREG sec 4 requires these)
struct_by_model = collections.Counter()
struct_by_resolved = collections.Counter()
n_msgs_hist = collections.Counter()
turns_structured = 0          # decision turns available in the structured subset
mt_values = collections.Counter()
role_values = collections.Counter()
tool_names = collections.Counter()
per_shard = []
schema_seen = None

for sh in SHARDS:
    f = pq.ParquetFile(sh)
    if schema_seen is None:
        schema_seen = [f.schema_arrow.field(i).name + ":" + str(f.schema_arrow.field(i).type)
                       for i in range(len(f.schema_arrow))]
    sr = 0
    for batch in f.iter_batches(batch_size=256,
                                columns=["messages", "instance_id", "resolved",
                                         "model", "traj_id"]):
        msgs = batch.column("messages").to_pylist()
        iids = batch.column("instance_id").to_pylist()
        res = batch.column("resolved").to_pylist()
        mdl = batch.column("model").to_pylist()
        tids = batch.column("traj_id").to_pylist()
        for m, iid, rv, md, tid in zip(msgs, iids, res, mdl, tids):
            n_rows += 1
            sr += 1
            resolved_ct[bool(rv)] += 1
            model_ct[md] += 1
            traj_ct[tid] += 1
            inst_ct[iid] += 1
            s = m if isinstance(m, str) else json.dumps(m)
            s_mt = "message_type" in s
            s_tc = "tool_calls" in s
            s_th = '"thought"' in s
            sub_mt += s_mt
            sub_tc += s_tc
            sub_th_ = s_th
            sub_thought += s_th
            sub_both += (s_mt and s_tc)
            # parsed
            try:
                turns = json.loads(m) if isinstance(m, str) else m
            except Exception:
                parse_fail += 1
                continue
            if not isinstance(turns, list):
                parse_fail += 1
                continue
            p_mt = any(isinstance(t, dict) and "message_type" in t for t in turns)
            p_tc = any(isinstance(t, dict) and "tool_calls" in t for t in turns)
            p_th = any(isinstance(t, dict) and "thought" in t for t in turns)
            par_mt += p_mt
            par_tc += p_tc
            par_thought += p_th
            par_both += (p_mt and p_tc)
            disagree_mt += (s_mt != p_mt)
            disagree_tc += (s_tc != p_tc)
            n_msgs_hist[len(turns)] += 1
            if p_mt and p_tc:
                struct_by_model[md] += 1
                struct_by_resolved[bool(rv)] += 1
                for t in turns:
                    if not isinstance(t, dict):
                        continue
                    mtv = t.get("message_type")
                    if mtv is not None:
                        mt_values[mtv] += 1
                    rl = t.get("role")
                    if rl is not None:
                        role_values[rl] += 1
                    tc = t.get("tool_calls")
                    if isinstance(tc, list):
                        for c in tc:
                            if isinstance(c, dict):
                                fn = (c.get("function") or {})
                                nm = fn.get("name") if isinstance(fn, dict) else None
                                if nm:
                                    tool_names[nm] += 1
                    # a decision turn = an assistant action turn
                    if mtv == "action":
                        turns_structured += 1
    per_shard.append({"shard": sh.split("/")[-1], "rows": sr})

out = {
  "_what": "B09 GATE -1 next_action item 1: the three provenance assertions, recomputed on the parquet actually on disk.",
  "computed_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
  "shards": per_shard,
  "parquet_schema": schema_seen,
  "n_rows": n_rows,
  "resolved_true": resolved_ct[True],
  "resolved_false": resolved_ct[False],
  "model_counts": dict(model_ct),
  "substring_method": {
    "_semantics": "marker present in the raw `messages` JSON string; reproduces the HF /filter LIKE test",
    "message_type": sub_mt, "tool_calls": sub_tc, "both": sub_both, "thought_quoted": sub_thought},
  "parsed_method": {
    "_semantics": "json.loads, then any(marker is a KEY of a dict turn)",
    "message_type": par_mt, "tool_calls": par_tc, "both": par_both, "thought": par_thought},
  "method_disagreements": {"message_type": disagree_mt, "tool_calls": disagree_tc},
  "parse_failures": parse_fail,
  "traj_id": {"distinct": len(traj_ct), "max_rows_per_traj_id": max(traj_ct.values()),
              "n_traj_id_with_more_than_1_row": sum(1 for v in traj_ct.values() if v > 1)},
  "instance_id": {"distinct": len(inst_ct), "max_rollouts": max(inst_ct.values()),
                  "min_rollouts": min(inst_ct.values()),
                  "rollout_count_hist": dict(sorted(collections.Counter(inst_ct.values()).items()))},
  "structured_subset": {
    "n": par_both,
    "by_model": dict(struct_by_model),
    "by_resolved": {str(k): v for k, v in struct_by_resolved.items()},
    "message_type_values": dict(mt_values),
    "role_values": dict(role_values),
    "tool_names": dict(tool_names.most_common(30)),
    "action_turns_total": turns_structured},
  "n_messages_summary": {
    "min": min(n_msgs_hist), "max": max(n_msgs_hist),
    "mean": sum(k * v for k, v in n_msgs_hist.items()) / sum(n_msgs_hist.values()),
    "n_distinct_lengths": len(n_msgs_hist)},
}

# the three assertions, evaluated but NOT raised -- recorded so a failure is visible
out["ASSERTIONS"] = {
  "n_rows==24100": (n_rows == 24100, n_rows, 24100),
  "resolved_true==9427": (resolved_ct[True] == 9427, resolved_ct[True], 9427),
  "resolved_false==14673": (resolved_ct[False] == 14673, resolved_ct[False], 14673),
  "both_substring==9116": (sub_both == 9116, sub_both, 9116),
  "both_parsed==9116": (par_both == 9116, par_both, 9116),
}
out["ALL_PASS"] = all(v[0] for v in out["ASSERTIONS"].values())

with open(OUT, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ("n_rows", "resolved_true", "resolved_false",
                                      "substring_method", "parsed_method",
                                      "method_disagreements", "parse_failures",
                                      "ASSERTIONS", "ALL_PASS")}, indent=2))
