#!/usr/bin/env python3
"""Paper B task #181 — keep14+fresh2 seed variance (seed42 vs seed1234) @ step200000.

Reads the `*_sv181*` result dirs produced by
scripts/_run_paperB_keep14_seedvar_local.sh (both arms, one driver, one node)
and emits the per-axis delta table with exact McNemar + paired bootstrap.

Integrity is asserted, never assumed:
  * every summary.json must exist and report n_shards == 8 (except OOD, 1 shard);
  * per-task n_scored must equal the expected count and n_nan must be 0;
  * per-item files must pair 1:1 on item_id with no duplicates;
  * MMLU must have exactly 14042 items.
Any violation raises -- we never emit a number from a partial set.

Also re-checks the archived seed-42 rows (produced on the retired .252 node with
$WD/.venv) against this battery's seed-42 re-run, which quantifies the
node/interpreter term separately from the seed term.

CPU only. No GPU.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter

import numpy as np

ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
os.chdir(ROOT)

SUF = "_sv181"
CORE6 = ["hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande", "openbookqa"]
KNOW5 = ["mmlu", "lambada_openai", "boolq", "commonsense_qa", "social_iqa"]
EXPECTED_N = {
    "hellaswag": 10042, "arc_challenge": 1172, "arc_easy": 2376,
    "piqa": 1838, "winogrande": 1267, "openbookqa": 500,
    "mmlu": 14042, "lambada_openai": 5153, "boolq": 3270,
    "commonsense_qa": 1221, "social_iqa": 1954,
}
# core6 macro convention from paperB tab_policy_endpoint: acc_norm, except
# winogrande where acc_norm == acc by construction (partial-cloze scoring).
N_BOOT = 10000
BOOT_SEED = 42


def die(msg):
    raise SystemExit(f"FATAL: {msg}")


def load_json(p):
    if not os.path.exists(p):
        die(f"missing {p}")
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------- statistics
def mcnemar_exact_p(b, c):
    """Two-sided exact binomial (sign) test on discordant pairs. Log-space to
    survive n>1000 (see fix 324a44f)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X<=k) under Bin(n,0.5), doubled, capped at 1
    log_terms = []
    for i in range(k + 1):
        log_terms.append(math.lgamma(n + 1) - math.lgamma(i + 1)
                         - math.lgamma(n - i + 1) - n * math.log(2.0))
    m = max(log_terms)
    tail = m + math.log(sum(math.exp(t - m) for t in log_terms))
    return min(1.0, 2.0 * math.exp(tail))


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=BOOT_SEED):
    """CI95 on mean(a)-mean(b) resampling ITEMS (pairs kept together)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        die(f"paired_bootstrap shape mismatch {a.shape} vs {b.shape}")
    rng = np.random.default_rng(seed)
    n = a.size
    d = a - b
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


# ---------------------------------------------------------------- loaders
def load_ppl(name, root="olmo2_ppl_results", n_shards_expected=8):
    s = load_json(f"{root}/{name}/summary.json")
    if s["n_shards"] != n_shards_expected:
        die(f"{root}/{name}: n_shards={s['n_shards']} != {n_shards_expected}")
    if s["n_tokens"] <= 0:
        die(f"{root}/{name}: n_tokens={s['n_tokens']}")
    return s


def load_downstream(name):
    s = load_json(f"olmo2_downstream_results/{name}/summary.json")
    if s["n_shards"] != 8:
        die(f"{name}: n_shards={s['n_shards']} != 8")
    if s.get("add_bos") is not False:
        die(f"{name}: add_bos={s.get('add_bos')} -- base protocol requires False")
    for t, v in s["tasks"].items():
        exp = EXPECTED_N.get(t)
        if exp is not None and v["n"] != exp:
            die(f"{name}/{t}: n={v['n']} != expected {exp}")
        if v["n_nan"] != 0:
            die(f"{name}/{t}: n_nan={v['n_nan']} != 0")
        if v["n_scored"] != v["n"] - v["n_nan"]:
            die(f"{name}/{t}: n_scored={v['n_scored']} inconsistent")
    return s


def load_perex_downstream(name, task):
    """-> dict item_id -> 1/0 correctness under the paper's metric for that task."""
    p = f"olmo2_downstream_results/{name}/per_example_{task}.jsonl"
    if not os.path.exists(p):
        die(f"missing per-item file {p}")
    out = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get("nan"):
                die(f"{p}: nan row item_id={r.get('item_id')}")
            iid = r["item_id"]
            if iid in out:
                die(f"{p}: duplicate item_id {iid}")
            # acc_norm_score is the char-length-normalised hit; `correct` is raw acc.
            # paperB core6 macro uses acc_norm (== acc for winogrande).
            out[iid] = r
    exp = EXPECTED_N.get(task)
    if exp is not None and len(out) != exp:
        die(f"{p}: {len(out)} rows != expected {exp}")
    return out


def perex_vector(recs, task, metric):
    """metric in {'acc','acc_norm'} -> (sorted ids, 0/1 hit vector).

    Verified empirically on the archived seed-42 dumps: `correct` is the raw-acc
    hit and `acc_norm_score` is a 0/1 char-length-normalised hit whose mean
    equals summary acc_norm to 6 dp (arc_challenge: .413823 / .437713 both ways).
    Any other value set would be a harness change -> we assert rather than guess.
    """
    ids = sorted(recs)
    if metric == "acc":
        return ids, np.array([1.0 if recs[i]["correct"] else 0.0 for i in ids])
    vals = []
    for i in ids:
        r = recs[i]
        if "acc_norm_score" not in r:
            die(f"{task}: per-item row {i} has no acc_norm_score")
        v = float(r["acc_norm_score"])
        if v not in (0.0, 1.0):
            die(f"{task}: acc_norm_score={v} at item {i} is not a 0/1 hit -- "
                "harness semantics changed, refusing to compute McNemar")
        vals.append(v)
    return ids, np.array(vals, dtype=np.float64)


def load_mmlu(name):
    s = load_json(f"olmo2_mmlu_content_results/{name}/summary.json")
    if s["n"] != 14042:
        die(f"{name}: mmlu n={s['n']} != 14042")
    if s["n_nan"] != 0:
        die(f"{name}: mmlu n_nan={s['n_nan']} != 0")
    if s["n_valid"] != 14042:
        die(f"{name}: mmlu n_valid={s['n_valid']} != 14042")
    return s


def load_mmlu_perex(name):
    p = f"olmo2_mmlu_content_results/{name}/per_example_mmlu.jsonl"
    if not os.path.exists(p):
        die(f"missing {p}")
    out = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get("nan"):
                die(f"{p}: nan row {r.get('item_id')}")
            iid = r["item_id"]
            if iid in out:
                die(f"{p}: duplicate item_id {iid}")
            out[iid] = r
    if len(out) != 14042:
        die(f"{p}: {len(out)} rows != 14042")
    return out


def paired_axis(name, a_vec, b_vec, ids_a, ids_b):
    if ids_a != ids_b:
        die(f"{name}: item_id sets differ between arms")
    b_only = int(((a_vec == 1) & (b_vec == 0)).sum())   # s42 right, s1234 wrong
    c_only = int(((a_vec == 0) & (b_vec == 1)).sum())   # s1234 right, s42 wrong
    p = mcnemar_exact_p(b_only, c_only)
    d, lo, hi = paired_bootstrap(a_vec, b_vec)
    return {
        "n": int(a_vec.size),
        "acc_s42": float(a_vec.mean()),
        "acc_s1234": float(b_vec.mean()),
        "delta_pp": 100.0 * d,
        "boot_ci95_pp": [100.0 * lo, 100.0 * hi],
        "n_flips": b_only + c_only,
        "s42_only_correct": b_only,
        "s1234_only_correct": c_only,
        "mcnemar_exact_p": p,
    }


def main():
    rep = {"axes": {}, "integrity": {}, "provenance_check": {}}

    # ---------------- axis 1: in-domain PPL ----------------
    a = load_ppl(f"keep14_s42_step200000{SUF}")
    b = load_ppl(f"keep14_s1234_step200000{SUF}")
    if a["n_tokens"] != b["n_tokens"]:
        die(f"ppl n_tokens mismatch {a['n_tokens']} vs {b['n_tokens']}")
    rep["axes"]["ppl_dolmino"] = {
        "n_tokens": a["n_tokens"], "n_windows": a["n_windows"],
        "ppl_s42": a["ppl"], "ppl_s1234": b["ppl"],
        "delta_ppl": a["ppl"] - b["ppl"],
        "avg_nll_s42": a["avg_nll"], "avg_nll_s1234": b["avg_nll"],
        "delta_avg_nll": a["avg_nll"] - b["avg_nll"],
    }
    # archived-vs-rerun provenance for seed42
    arch = load_ppl("7B_keep14_step200000")
    rep["provenance_check"]["ppl_dolmino_s42"] = {
        "archived_252_venv": arch["ppl"], "rerun_local_conda": a["ppl"],
        "delta": arch["ppl"] - a["ppl"],
        "identical": abs(arch["ppl"] - a["ppl"]) < 1e-12,
    }

    # ---------------- axes 5,6: OOD PPL ----------------
    for corpus, archived in [("wikitext103", "keep14_step200000_wikitext103"),
                             ("pg19", "keep14_step200000_pg19")]:
        a = load_ppl(f"keep14_s42_step200000{SUF}_{corpus}", "ood_ppl_results", 1)
        b = load_ppl(f"keep14_s1234_step200000{SUF}_{corpus}", "ood_ppl_results", 1)
        if a["n_tokens"] != b["n_tokens"]:
            die(f"{corpus}: n_tokens mismatch")
        rep["axes"][f"ppl_{corpus}"] = {
            "n_tokens": a["n_tokens"],
            "ppl_s42": a["ppl"], "ppl_s1234": b["ppl"],
            "delta_ppl": a["ppl"] - b["ppl"],
            "delta_avg_nll": a["avg_nll"] - b["avg_nll"],
        }
        ar = load_ppl(archived, "ood_ppl_results", 1)
        rep["provenance_check"][f"ppl_{corpus}_s42"] = {
            "archived": ar["ppl"], "rerun": a["ppl"],
            "delta": ar["ppl"] - a["ppl"],
            "identical": abs(ar["ppl"] - a["ppl"]) < 1e-12,
        }

    # ---------------- axes 2,3: downstream ----------------
    for leg, tasks, sfx in [("core6", CORE6, ""), ("know5", KNOW5, "_know")]:
        sa = load_downstream(f"keep14_s42_step200000{SUF}{sfx}")
        sb = load_downstream(f"keep14_s1234_step200000{SUF}{sfx}")
        rep["integrity"][leg] = {
            t: {"n_scored_s42": sa["tasks"][t]["n_scored"],
                "n_scored_s1234": sb["tasks"][t]["n_scored"],
                "expected": EXPECTED_N[t],
                "n_nan_s42": sa["tasks"][t]["n_nan"],
                "n_nan_s1234": sb["tasks"][t]["n_nan"]}
            for t in tasks
        }
        for t in tasks:
            ra = load_perex_downstream(f"keep14_s42_step200000{SUF}{sfx}", t)
            rb = load_perex_downstream(f"keep14_s1234_step200000{SUF}{sfx}", t)
            ida, va = perex_vector(ra, t, "acc")
            idb, vb = perex_vector(rb, t, "acc")
            res = paired_axis(f"{leg}/{t}/acc", va, vb, ida, idb)
            # cross-check per-item mean vs summary acc
            for tag, vec, s in (("s42", va, sa), ("s1234", vb, sb)):
                if abs(vec.mean() - s["tasks"][t]["acc"]) > 1e-9:
                    die(f"{leg}/{t}/{tag}: per-item acc {vec.mean()} != summary "
                        f"{s['tasks'][t]['acc']}")
            rep["axes"][f"{leg}_{t}_acc"] = res
            # acc_norm: paired per-item test too (paperB core6 macro uses acc_norm)
            idan, van = perex_vector(ra, t, "acc_norm")
            idbn, vbn = perex_vector(rb, t, "acc_norm")
            for tag, vec, s in (("s42", van, sa), ("s1234", vbn, sb)):
                if abs(vec.mean() - s["tasks"][t]["acc_norm"]) > 1e-9:
                    die(f"{leg}/{t}/{tag}: per-item acc_norm {vec.mean()} != "
                        f"summary {s['tasks'][t]['acc_norm']}")
            rep["axes"][f"{leg}_{t}_accnorm"] = paired_axis(
                f"{leg}/{t}/acc_norm", van, vbn, idan, idbn)
        # macro aggregates in the paper's convention
        def macro(s, tl):
            vals = []
            for t in tl:
                v = s["tasks"][t]
                vals.append(v["acc"] if t == "winogrande" else v["acc_norm"])
            return float(np.mean(vals))
        if leg == "core6":
            rep["axes"]["core6_macro_accnorm"] = {
                "s42": macro(sa, CORE6), "s1234": macro(sb, CORE6),
                "delta_pp": 100.0 * (macro(sa, CORE6) - macro(sb, CORE6)),
                "convention": "acc_norm, winogrande acc (acc_norm==acc there)",
            }
        else:
            aux = lambda s: float(np.mean([s["tasks"][t]["acc"] for t in KNOW5]))
            rep["axes"]["aux5_raw_acc"] = {
                "s42": aux(sa), "s1234": aux(sb),
                "delta_pp": 100.0 * (aux(sa) - aux(sb)),
                "convention": "acc-based mean over know5 (paperB aux5_raw; NOT "
                              "a knowledge-recovery claim)",
            }
        # provenance: archived seed42 vs rerun seed42
        arch_name = ("7B_keep14_step200000" if leg == "core6"
                     else "7B_keep14_step200000_know")
        sar = load_downstream(arch_name)
        rep["provenance_check"][f"{leg}_s42"] = {
            t: {"archived_acc": sar["tasks"][t]["acc"],
                "rerun_acc": sa["tasks"][t]["acc"],
                "delta_pp": 100.0 * (sar["tasks"][t]["acc"] - sa["tasks"][t]["acc"]),
                "archived_accnorm": sar["tasks"][t]["acc_norm"],
                "rerun_accnorm": sa["tasks"][t]["acc_norm"]}
            for t in tasks
        }

    # ---------------- axis 4: MMLU letter+content ----------------
    ma = load_mmlu(f"keep14_s42_step200000{SUF}")
    mb = load_mmlu(f"keep14_s1234_step200000{SUF}")
    pa = load_mmlu_perex(f"keep14_s42_step200000{SUF}")
    pb = load_mmlu_perex(f"keep14_s1234_step200000{SUF}")
    ids = sorted(pa)
    if ids != sorted(pb):
        die("mmlu item_id sets differ")
    for proto, key in [("letter", "letter_acc"),
                       ("content_raw", "content_raw_acc"),
                       ("content_norm", "content_norm_acc")]:
        va = np.array([1.0 if pa[i][proto]["correct"] else 0.0 for i in ids])
        vb = np.array([1.0 if pb[i][proto]["correct"] else 0.0 for i in ids])
        if abs(va.mean() - ma[key]) > 1e-9:
            die(f"mmlu {proto} s42 per-item {va.mean()} != summary {ma[key]}")
        if abs(vb.mean() - mb[key]) > 1e-9:
            die(f"mmlu {proto} s1234 per-item {vb.mean()} != summary {mb[key]}")
        rep["axes"][f"mmlu_{proto}"] = paired_axis(f"mmlu/{proto}", va, vb, ids, ids)
    rep["integrity"]["mmlu"] = {"n_s42": ma["n"], "n_s1234": mb["n"],
                                "expected": 14042,
                                "n_nan_s42": ma["n_nan"], "n_nan_s1234": mb["n_nan"]}
    march = load_mmlu("7B_keep14_step200000")
    rep["provenance_check"]["mmlu_s42"] = {
        k: {"archived": march[k], "rerun": ma[k],
            "delta_pp": 100.0 * (march[k] - ma[k])}
        for k in ("letter_acc", "content_raw_acc", "content_norm_acc")
    }

    # ---------------- df=1 caveat, computed not asserted ----------------
    # With n=2 draws, s^2 has 1 dof. A chi2_1 CI on sigma is
    #   [ s*sqrt(df/chi2_{0.975,df}), s*sqrt(df/chi2_{0.025,df}) ]
    # chi2_{0.975,1}=5.023886, chi2_{0.025,1}=0.000982
    rep["df1_caveat"] = {
        "n_runs": 2, "df": 1,
        "chi2_0975_df1": 5.023886, "chi2_0025_df1": 0.000982,
        "sigma_ci_multiplier_lo": math.sqrt(1 / 5.023886),
        "sigma_ci_multiplier_hi": math.sqrt(1 / 0.000982),
        "note": "any sd from 2 draws must be multiplied by [0.446, 31.9] to get a "
                "95% CI on sigma_run -- a ~72x wide interval. NOT reportable as "
                "sigma_run. Do NOT pool with A03 (1B keep7/keep12 triviaqa, df=5).",
    }

    out = "paperB/SEEDVAR_KEEP14_RESULTS.json"
    with open(out, "w") as f:
        json.dump(rep, f, indent=2, sort_keys=True)
    print(json.dumps(rep, indent=2, sort_keys=True))
    print(f"\nwrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
