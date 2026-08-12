#!/usr/bin/env python3
"""A05 closeout §2 -- recompute the surviving cost claim on ALL axes, incl. the AR control.

Everything here is recomputed from per-item rows. Nothing is read from a summary table.
"""
import glob, json, statistics as st

Z = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft"
Z104 = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104"

def load(pat, expected, label):
    rows = []
    for f in sorted(glob.glob(pat)):
        for l in open(f):
            if l.strip(): rows.append(json.loads(l))
    ids = [r["task_id"] for r in rows]
    assert len(rows) == expected, (label, len(rows), expected)
    assert len(set(ids)) == expected, (label, "dups")
    return rows

def summ(v):
    v = sorted(v)
    n = len(v)
    return {"mean": st.mean(v), "median": st.median(v), "total": sum(v),
            "p90": v[int(0.9*n)-1], "max": max(v),
            "top15pct_share": sum(v[int(0.85*n):]) / sum(v) if sum(v) else 0.0}

out = {}

# ---------- DreamOn (K1 cells, canvas=32) ----------
for ds, cell, exp in [("HE+", "he_c32", 164), ("MBPP+", "mbpp_c32", 378),
                      ("HE+_c8", "he_c8", 164), ("MBPP+_c8", "mbpp_c8", 378),
                      ("HE+_c128", "he_c128", 164)]:
    rows = load(f"{Z}/runs/a05_k1/{cell}/metrics.rank*.jsonl", exp, cell)
    p = [r["process"] for r in rows]
    out[f"dreamon_{cell}"] = {
        "benchmark": ds, "n": exp,
        "nfe": summ([x["nfe"] for x in p]),
        "tokens_fed_effective": summ([x["tokens_fed_effective"] for x in p]),
        "tokens_fed_padded": summ([x["tokens_fed_padded"] for x in p]),
    }

# ---------- Scaffold Medium: WZC1-ONLY, computed on LOCAL, not here ----------
for ds, run, exp in []:
    rows = load(f"{Z}/runs/{run}/metrics.rank*.jsonl", exp, run)
    p = [(r.get("process") or r.get("failure_process")) for r in rows]
    assert all(x is not None for x in p), run
    out[f"scaffold_{run}"] = {
        "benchmark": ds, "n": exp,
        "nfe": summ([x["nfe"] for x in p]),
        "tokens_fed_effective": summ([x["cumulative_model_tokens"] for x in p]),
        "note": "tokens_fed == cumulative_model_tokens (sum of canvas width over model calls); "
                "no KV cache, so attended_context_sum == tokens_fed",
    }

# ---------- AR control (report.json carries measured_matches_analytic) ----------
for ds, dsdir, exp in [("HE+", "humaneval", 164), ("MBPP+", "mbpp", 378)]:
    for run in ["ar_qwen25coder7b_base_greedy", "ar_qwen25coder7b_base"]:
        rows = load(f"{Z104}/outputs/{run}/{dsdir}/shards/metrics.rank*.jsonl", exp, run)
        got = []
        for r in rows:
            pr = r.get("process") or {}
            c = pr.get("cost") or {}
            got.append(c)
        assert all(c for c in got), (run, "missing cost")
        out[f"ar_{run}_{dsdir}"] = {
            "benchmark": ds, "n": exp,
            "nfe": summ([c["forward_passes"] for c in got]),
            "tokens_fed_effective": summ([c["tokens_fed"] for c in got]),
            "attended_context_sum": summ([c["attended_context_sum"] for c in got]),
        }

print(json.dumps(out, indent=1, default=float))
with open("/tmp/a05_cost_raw.json", "w") as h:
    json.dump(out, h, indent=2, default=float)
