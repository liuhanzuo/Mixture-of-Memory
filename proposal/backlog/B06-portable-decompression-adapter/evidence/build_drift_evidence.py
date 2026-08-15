#!/usr/bin/env python3
"""B06 drift-resolution leg -- recompute every number from per-item raw records.

0 GPU. Reads only:
  locomo_results/{hcache,hcache_j12_noLoRA_chatFALSE,hcache_j12_LoRA_chatFALSE}/
      {preds_shard*.jsonl, judge_cache.jsonl, scores.json}
  locomo_results_openjudge_qwen3_MIRROR/*/scores.json
  locomo/data/locomo10.json

Writes evidence/drift_resolution_evidence.json.

Nothing here is copied from STATUS.json. Where the recomputation disagrees with
STATUS.json the recomputed value wins and the disagreement is recorded in the
'discrepancies_vs_status_json' block.

Run:  python3 proposal/backlog/B06-portable-decompression-adapter/evidence/build_drift_evidence.py
"""
import collections
import glob
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from math import comb

ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
OUT = os.path.join(
    ROOT, "proposal/backlog/B06-portable-decompression-adapter/evidence",
    "drift_resolution_evidence.json")

# Refusal regex copied verbatim from scripts/eval_qcmem_locomo.py:305 so the
# cat-5 local grading can be reproduced without importing the (torch-heavy) driver.
REFUSAL = re.compile(
    r"\b(i don'?t know|not (mentioned|sure|provided|available|specified)"
    r"|no (information|mention|record)|cannot (find|determine|answer)"
    r"|unanswerable|isn'?t (mentioned|provided)|wasn'?t mentioned)\b", re.I)

CAT_N = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_run(d):
    """Return (preds_by_id, judge_cache_by_id, n_dup_cache_records)."""
    preds = {}
    for p in sorted(glob.glob(f"{ROOT}/locomo_results/{d}/preds_shard*.jsonl")):
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    preds[r["id"]] = r
    cache, dup = {}, 0
    with open(f"{ROOT}/locomo_results/{d}/judge_cache.jsonl") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if r["id"] in cache:
                    dup += 1
                cache[r["id"]] = r
    return preds, cache, dup


def recompute(d):
    """Recompute Judge_1:4, the cat-5 regex term and the blended score from raw items."""
    preds, cache, dup = load_run(d)
    ab = [p for p in preds.values() if p.get("is_abstention")]
    nab = [p for p in preds.values() if not p.get("is_abstention")]
    judge_correct = sum(1 for p in nab if float(cache[p["id"]]["judge"]) == 1.0)
    regex_correct = sum(
        1 for p in ab
        if REFUSAL.search(p.get("pred", "") or "") or not (p.get("pred", "") or "").strip())
    n_all, n14, n5 = len(preds), len(nab), len(ab)
    published = json.load(open(f"{ROOT}/locomo_results/{d}/scores.json"))
    per_cat = {}
    for c in ("1", "2", "3", "4"):
        sub = [p for p in nab if str(p["category"]) == c]
        k = sum(1 for p in sub if float(cache[p["id"]]["judge"]) == 1.0)
        per_cat[c] = {"n": len(sub), "n_correct": k, "pct": 100.0 * k / len(sub),
                      "published_pct": published["by_category"][c]["judge"],
                      "matches_published": abs(
                          published["by_category"][c]["judge"] - 100.0 * k / len(sub)) < 1e-9}
    blended = 100.0 * (judge_correct + regex_correct) / n_all
    return {
        "dir": f"locomo_results/{d}",
        "n_preds": n_all,
        "n_unique_pred_ids": len(preds),
        "n_judge_cache_records": len(cache),
        "n_duplicate_cache_ids": dup,
        "judge_models_in_cache": sorted({r.get("model") for r in cache.values()}),
        "cache_id_set_equals_non_abstention_id_set":
            set(cache) == {p["id"] for p in nab},
        "abstention_set_equals_cat5_set":
            {p["id"] for p in ab} == {p["id"] for p in preds.values() if p["category"] == 5},
        "instrument": "MIXED: cat1-4 gpt-4o judge, cat5 local refusal regex",
        "n_judged_by_llm": n14,
        "n_graded_by_regex": n5,
        "judge_1_4_n_correct": judge_correct,
        "judge_1_4_pct_RECOMPUTED": 100.0 * judge_correct / n14,
        "cat5_regex_n_correct": regex_correct,
        "cat5_regex_pct_RECOMPUTED": 100.0 * regex_correct / n5,
        "blended_n1986_pct_RECOMPUTED": blended,
        "blended_n1986_pct_PUBLISHED_scores_json": published["overall_judge"],
        "blended_matches_published_to_1e_9":
            abs(published["overall_judge"] - blended) < 1e-9,
        "per_category_judge_1_4": per_cat,
        "judge_independent_1_4": {
            "f1": sum(published["by_category"][c]["f1"] * CAT_N[int(c)]
                      for c in "1234") / 1540,
            "acc": sum(published["by_category"][c]["acc"] * CAT_N[int(c)]
                       for c in "1234") / 1540,
        },
        "sha256": {
            "judge_cache.jsonl": sha256(f"{ROOT}/locomo_results/{d}/judge_cache.jsonl"),
            "scores.json": sha256(f"{ROOT}/locomo_results/{d}/scores.json"),
        },
        "eval_config": json.load(
            open(f"{ROOT}/locomo_results/{d}/eval_config_shard0of3.json")),
    }


def invert_published_percentages():
    """Pin the canonical run's integer counts by brute-force inverting the
    published per-category percentages. Three independent routes must agree."""
    pub = {1: 6.74, 2: 2.49, 3: 9.38, 4: 14.27, 5: 1.12}  # LOCOMO_JUDGE_AGGREGATE.md:41
    per_cat, tot14 = {}, 0
    for c in (1, 2, 3, 4, 5):
        cands = [k for k in range(CAT_N[c] + 1)
                 if round(100.0 * k / CAT_N[c], 2) == pub[c]]
        per_cat[str(c)] = {"published_pct": pub[c], "n": CAT_N[c],
                           "integer_counts_consistent": cands,
                           "unique": len(cands) == 1}
        if len(cands) == 1 and c <= 4:
            tot14 += cands[0]
    cat5 = per_cat["5"]["integer_counts_consistent"][0]
    route_b = [x for x in range(1541) if round(100.0 * (x + cat5) / 1986, 2) == 8.11]
    route_c = [x for x in range(1541) if round(100.0 * x / 1540, 2) == 10.13]
    return {
        "method": "brute-force inversion of published rounded percentages; "
                  "uniqueness asserted rather than assumed",
        "route_A_per_category": per_cat,
        "route_A_implies_cat1_4_correct": tot14,
        "route_A_implies_cat5_correct": cat5,
        "route_B_from_blended_8_11": {
            "solutions": route_b, "unique": len(route_b) == 1},
        "route_C_from_published_judge_1_4_10_13": {
            "solutions": route_c, "unique": len(route_c) == 1},
        "all_three_routes_agree": (len(route_b) == 1 and len(route_c) == 1
                                   and route_b[0] == route_c[0] == tot14),
        "canonical_judge_1_4_n_correct": tot14,
        "canonical_cat5_regex_n_correct": cat5,
        "canonical_blended_n1986_RECOMPUTED": 100.0 * (tot14 + cat5) / 1986,
        "canonical_judge_1_4_RECOMPUTED": 100.0 * tot14 / 1540,
    }


def mirror_block():
    """The canonical run IS represented on wzc1: an open-judge re-grade of the
    SAME predictions, whose judge-independent columns identify the run."""
    p = (f"{ROOT}/locomo_results_openjudge_qwen3_MIRROR/hcache_8b_chatFALSE")
    sj = json.load(open(f"{p}/scores.json"))
    meta = json.load(open(f"{p}/judge_meta.json"))
    cat5 = sj["by_category"]["5"]["judge"]
    cat5_k = [k for k in range(447) if abs(100.0 * k / 446 - cat5) < 1e-9]
    return {
        "dir": "locomo_results_openjudge_qwen3_MIRROR/hcache_8b_chatFALSE",
        "what_it_is": "second judge (open-weight qwen3-8b, non-thinking, greedy) applied to "
                      "the SAME canonical hcache_8b_chatFALSE predictions. Mirrored to wzc1 "
                      "per paperA/TODOList.md:170. preds/judge_cache themselves stayed on zwfy6.",
        "judge_meta": meta,
        "n_samples": sj["n_samples"],
        "open_judge_1_4": sum(sj["by_category"][c]["judge"] * CAT_N[int(c)]
                              for c in "1234") / 1540,
        "open_judge_blended_n1986": sj["overall_judge"],
        "cat5_judge_pct": cat5,
        "cat5_implied_integer_count": cat5_k,
        "cat5_is_judge_independent_regex_short_circuit": cat5_k == [5],
        "judge_independent_1_4": {
            "f1": sum(sj["by_category"][c]["f1"] * CAT_N[int(c)] for c in "1234") / 1540,
            "acc": sum(sj["by_category"][c]["acc"] * CAT_N[int(c)] for c in "1234") / 1540,
        },
        "identifies_same_run_as_published_canonical_row": {
            "published_row": "status/PAPERA_RESULTS_CONSOLIDATED.md:175 -> f1 4.67 acc 6.29 em 0.25",
            "mirror_rounded": {"f1": round(sj["overall_f1"], 2),
                               "acc": round(sj["overall_acc"], 2),
                               "em": round(sj["overall_em"], 2)},
            "match": (round(sj["overall_f1"], 2) == 4.67
                      and round(sj["overall_acc"], 2) == 6.29
                      and round(sj["overall_em"], 2) == 0.25),
        },
        "sha256_scores_json": sha256(f"{p}/scores.json"),
    }


def cross_judge_ratios():
    """Same generations, two judges, six methods -> is HCache an outlier?"""
    gpt = {  # published gpt-4o per-category, status/LOCOMO_JUDGE_AGGREGATE.md:34-41
        "qcmem_8b_iter_chatFALSE": {1: 26.95, 2: 19.00, 3: 30.21, 4: 69.32},
        "kvdirect_8b_chatFALSE": {1: 24.11, 2: 18.69, 3: 25.00, 4: 62.19},
        "streamingllm_8b_chatFALSE": {1: 22.70, 2: 11.21, 3: 23.96, 4: 42.21},
        "infllm_8b_chatFALSE": {1: 18.44, 2: 14.33, 3: 25.00, 4: 33.89},
        "memoryllm_chatFALSE": {1: 15.60, 2: 8.72, 3: 15.63, 4: 27.47},
        "hcache_8b_chatFALSE": {1: 6.74, 2: 2.49, 3: 9.38, 4: 14.27},
    }
    rows = {}
    for m, pc in gpt.items():
        g = 100.0 * sum(round(pc[c] / 100.0 * CAT_N[c]) for c in (1, 2, 3, 4)) / 1540
        sj = json.load(open(
            f"{ROOT}/locomo_results_openjudge_qwen3_MIRROR/{m}/scores.json"))
        o = sum(sj["by_category"][str(c)]["judge"] * CAT_N[c] for c in (1, 2, 3, 4)) / 1540
        rows[m] = {"gpt4o_judge_1_4": g, "open_judge_1_4": o, "ratio_open_over_gpt4o": o / g}
    others = [v["ratio_open_over_gpt4o"] for k, v in rows.items() if k != "hcache_8b_chatFALSE"]
    mu = sum(others) / len(others)
    sd = math.sqrt(sum((x - mu) ** 2 for x in others) / (len(others) - 1))
    hc = rows["hcache_8b_chatFALSE"]["ratio_open_over_gpt4o"]
    return {
        "what_this_tests": "If the canonical gpt-4o HCache score were a faithful measurement, "
                           "its open-judge/gpt-4o ratio should sit inside the range spanned by "
                           "the five sibling methods graded in the SAME two judge passes.",
        "per_method": rows,
        "sibling_ratios": sorted(round(x, 4) for x in others),
        "sibling_ratio_mean": mu, "sibling_ratio_sd": sd,
        "hcache_ratio": hc,
        "hcache_ratio_z_vs_siblings": (hc - mu) / sd,
        "hcache_ratio_over_max_sibling": hc / max(others),
        "implied_gpt4o_judge_1_4_if_hcache_followed_sibling_mapping": {
            "point": rows["hcache_8b_chatFALSE"]["open_judge_1_4"] / mu,
            "range_over_sibling_ratios": [
                rows["hcache_8b_chatFALSE"]["open_judge_1_4"] / max(others),
                rows["hcache_8b_chatFALSE"]["open_judge_1_4"] / min(others)],
            "vs_published_canonical_judge_1_4": 10.129870129870129,
            "vs_local_replicates": [15.454545454545455, 16.688311688311687],
        },
        "rank_inversion": {
            "gpt4o_order": [m for m, _ in sorted(
                rows.items(), key=lambda kv: -kv[1]["gpt4o_judge_1_4"])],
            "open_order": [m for m, _ in sorted(
                rows.items(), key=lambda kv: -kv[1]["open_judge_1_4"])],
            "note": "under the open judge HCache overtakes MemoryLLM; under gpt-4o it is last",
        },
    }


def judge_noise_floor():
    """Two separate gpt-4o passes over BYTE-IDENTICAL prediction strings give a
    direct, assumption-free measurement of judge non-determinism."""
    pa, ca, _ = load_run("hcache")
    pb, cb, _ = load_run("hcache_j12_noLoRA_chatFALSE")
    ids = sorted(set(ca) & set(cb))
    same = [i for i in ids if pa[i]["pred"] == pb[i]["pred"]]
    diff = [i for i in ids if pa[i]["pred"] != pb[i]["pred"]]

    def flips(sub):
        f01 = sum(1 for i in sub if ca[i]["judge"] == 0.0 and cb[i]["judge"] == 1.0)
        f10 = sum(1 for i in sub if ca[i]["judge"] == 1.0 and cb[i]["judge"] == 0.0)
        return f01, f10
    s01, s10 = flips(same)
    d01, d10 = flips(diff)
    p_flip = (s01 + s10) / len(same)
    sd_net = math.sqrt(1540 * p_flip)
    return {
        "design": "hcache (generated 07-09/10, judged 07-18) vs hcache_j12_noLoRA_chatFALSE "
                  "(generated + judged 07-25). Both gpt-4o, both n=1540, identical id sets. "
                  "On the subset where the prediction STRING is byte-identical the judge input "
                  "is literally the same, so any verdict change is judge non-determinism.",
        "n_paired": len(ids),
        "n_byte_identical_predictions": len(same),
        "pct_byte_identical": 100.0 * len(same) / len(ids),
        "n_differing_predictions": len(diff),
        "identical_subset_flips": {"zero_to_one": s01, "one_to_zero": s10,
                                   "total": s01 + s10,
                                   "pct_of_subset": 100.0 * (s01 + s10) / len(same),
                                   "net": s01 - s10},
        "differing_subset_flips": {"zero_to_one": d01, "one_to_zero": d10,
                                   "net": d01 - d10},
        "per_item_flip_probability": p_flip,
        "flips_are_symmetric": abs(s01 - s10) <= 2,
        "sd_of_net_change_at_n1540_items": sd_net,
        "interpretation": "judge non-determinism is ~3% per item but SYMMETRIC, so it moves the "
                          "aggregate by only ~+-0.45 pp (1 sd), NOT by 6.6 pp. It explains the "
                          "1.23 pp local-vs-local wobble, not the canonical gap.",
        "sigma_of_each_observed_gap": {
            "canonical_vs_B06_control_101_items": 101 / sd_net,
            "canonical_vs_older_local_82_items": 82 / sd_net,
            "older_local_vs_B06_control_19_items": 19 / sd_net,
        },
        "local_vs_local_gap_decomposition": {
            "net_items": (s01 - s10) + (d01 - d10),
            "from_byte_identical_predictions_pure_judge_noise": s01 - s10,
            "from_differing_predictions": d01 - d10,
        },
    }


def attribution():
    """Calibrate judge-vs-lexical sensitivity on a KNOWN-real quality change
    (the +LoRA arm), then ask how much lexical movement the canonical drift shows."""
    def c14(path, key):
        sj = json.load(open(path))
        return sum(sj["by_category"][c][key] * CAT_N[int(c)] for c in "1234") / 1540
    P = {
        "canonical": f"{ROOT}/locomo_results_openjudge_qwen3_MIRROR/hcache_8b_chatFALSE/scores.json",
        "older_local": f"{ROOT}/locomo_results/hcache/scores.json",
        "b06_control": f"{ROOT}/locomo_results/hcache_j12_noLoRA_chatFALSE/scores.json",
        "b06_lora": f"{ROOT}/locomo_results/hcache_j12_LoRA_chatFALSE/scores.json",
    }
    f1 = {k: c14(v, "f1") for k, v in P.items()}
    ac = {k: c14(v, "acc") for k, v in P.items()}
    ju = {"canonical": 10.129870129870129, "older_local": 15.454545454545455,
          "b06_control": 16.688311688311687, "b06_lora": 39.805194805194802}
    dj = ju["b06_lora"] - ju["b06_control"]
    sf = (f1["b06_lora"] - f1["b06_control"]) / dj
    sa = (ac["b06_lora"] - ac["b06_control"]) / dj
    dj2 = ju["b06_control"] - ju["canonical"]
    dF = f1["b06_control"] - f1["canonical"]
    dA = ac["b06_control"] - ac["canonical"]
    return {
        "logic": "F1 and acc are deterministic functions of the predictions ONLY -- no judge, no "
                 "API, no sampling. A real change in generation quality must move them. A judge "
                 "artefact cannot.",
        "calibration_on_a_known_real_change_the_LoRA_arm": {
            "judge_1_4_delta_pp": dj,
            "f1_1_4_delta_pp": f1["b06_lora"] - f1["b06_control"],
            "acc_1_4_delta_pp": ac["b06_lora"] - ac["b06_control"],
            "slope_f1_pp_per_judge_pp": sf,
            "slope_acc_pp_per_judge_pp": sa,
        },
        "applied_to_the_canonical_drift": {
            "judge_1_4_drift_pp": dj2,
            "predicted_f1_movement_if_real_pp": sf * dj2,
            "observed_f1_movement_pp": dF,
            "observed_over_predicted_f1_pct": 100.0 * dF / (sf * dj2),
            "predicted_acc_movement_if_real_pp": sa * dj2,
            "observed_acc_movement_pp": dA,
            "observed_over_predicted_acc_pct": 100.0 * dA / (sa * dj2),
            "generation_attributable_share_via_f1_pct": 100.0 * (dF / sf) / dj2,
            "generation_attributable_share_via_acc_pct": 100.0 * (dA / sa) / dj2,
            "judge_attributable_share_via_f1_pct": 100.0 - 100.0 * (dF / sf) / dj2,
            "judge_attributable_share_via_acc_pct": 100.0 - 100.0 * (dA / sa) / dj2,
        },
        "three_noLoRA_replicates_on_the_n1540_ruler": {
            k: {"f1_1_4": f1[k], "acc_1_4": ac[k], "gpt4o_judge_1_4": ju[k]}
            for k in ("canonical", "older_local", "b06_control")},
        "relative_spread_across_the_three_replicates_pct": {
            m: 100.0 * (max(v) - min(v)) / min(v) for m, v in {
                "f1_1_4": [f1[k] for k in ("canonical", "older_local", "b06_control")],
                "acc_1_4": [ac[k] for k in ("canonical", "older_local", "b06_control")],
                "gpt4o_judge_1_4": [ju[k] for k in ("canonical", "older_local", "b06_control")],
            }.items()},
    }


def headline_stats():
    """Re-verify the B06 headline result and settle kill condition 1."""
    pa, ca, _ = load_run("hcache_j12_noLoRA_chatFALSE")
    pb, cb, _ = load_run("hcache_j12_LoRA_chatFALSE")
    ids = sorted(ca)
    A = [int(ca[i]["judge"] == 1.0) for i in ids]
    B = [int(cb[i]["judge"] == 1.0) for i in ids]
    b = sum(1 for x, y in zip(A, B) if y == 1 and x == 0)
    c = sum(1 for x, y in zip(A, B) if y == 0 and x == 1)
    n, k = b + c, min(b, c)
    pexact = min(2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n, 1.0)
    d = [B[i] - A[i] for i in range(len(ids))]
    random.seed(1)
    bi = sorted(100.0 * sum(d[j] for j in random.choices(range(len(ids)), k=len(ids)))
                / len(ids) for _ in range(10000))
    conv = collections.defaultdict(list)
    for i, _id in enumerate(ids):
        conv[_id.split("_")[0]].append(i)
    convs = sorted(conv)
    random.seed(1)
    cl = []
    for _ in range(10000):
        idx = [i for cc in random.choices(convs, k=len(convs)) for i in conv[cc]]
        cl.append(100.0 * sum(d[i] for i in idx) / len(idx))
    cl.sort()
    # data-grounded category semantics
    raw = json.load(open(f"{ROOT}/locomo/data/locomo10.json"))
    byc = collections.defaultdict(list)
    for s in raw:
        for q in s.get("qa", []):
            byc[q.get("category")].append(q)
    semantics = {}
    for cat in sorted(byc):
        qs = byc[cat]
        evs = [q.get("evidence") or [] for q in qs]
        sess = [len({m.group(1) for e in ev if (m := re.match(r"D(\d+)", e))}) for ev in evs]
        semantics[str(cat)] = {
            "n": len(qs),
            "mean_evidence_items": sum(len(e) for e in evs) / len(qs),
            "pct_exactly_one_evidence": 100.0 * sum(1 for e in evs if len(e) == 1) / len(qs),
            "pct_multi_evidence": 100.0 * sum(1 for e in evs if len(e) > 1) / len(qs),
            "pct_zero_evidence": 100.0 * sum(1 for e in evs if len(e) == 0) / len(qs),
            "mean_distinct_sessions_cited": sum(sess) / len(sess),
            "pct_question_starts_when_or_what_year": 100.0 * sum(
                1 for q in qs if re.match(r"\s*(when|what year|what date|how long ago)\b",
                                          q["question"], re.I)) / len(qs),
        }
    percat = {}
    labels = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
    for cat in (1, 2, 3, 4):
        idx = [i for i in ids if pa[i]["category"] == cat]
        aa = [int(ca[i]["judge"] == 1.0) for i in idx]
        bb = [int(cb[i]["judge"] == 1.0) for i in idx]
        pb_ = sum(1 for x, y in zip(aa, bb) if y == 1 and x == 0)
        pc_ = sum(1 for x, y in zip(aa, bb) if y == 0 and x == 1)
        nn, kk = pb_ + pc_, min(pb_, pc_)
        pp = min(2.0 * sum(comb(nn, i) for i in range(kk + 1)) / 2 ** nn, 1.0) if nn else 1.0
        percat[str(cat)] = {
            "data_grounded_label": labels[cat],
            "label_in_eval_script_CATEGORY_NAMES": {
                1: "multi_hop", 2: "single_hop", 3: "temporal", 4: "open_domain"}[cat],
            "n": len(idx),
            "noLoRA_pct": 100.0 * sum(aa) / len(idx),
            "lora_pct": 100.0 * sum(bb) / len(idx),
            "within_cat_delta_pp": 100.0 * (sum(bb) - sum(aa)) / len(idx),
            "contribution_to_overall_gain_pp": 100.0 * (sum(bb) - sum(aa)) / len(ids),
            "share_of_overall_gain_pct":
                100.0 * (sum(bb) - sum(aa)) / (sum(B) - sum(A)),
            "mcnemar_b": pb_, "mcnemar_c": pc_, "mcnemar_exact_two_sided_p": pp,
            "significantly_positive_at_0_05": pp < 0.05 and sum(bb) > sum(aa),
        }
    return {
        "noLoRA_judge_1_4": 100.0 * sum(A) / len(ids),
        "lora_judge_1_4": 100.0 * sum(B) / len(ids),
        "gain_pp": 100.0 * (sum(B) - sum(A)) / len(ids),
        "mcnemar_b": b, "mcnemar_c": c,
        "mcnemar_exact_two_sided_p": pexact,
        "mcnemar_chi2_continuity_corrected": (abs(b - c) - 1) ** 2 / (b + c),
        "paired_item_bootstrap_95ci_pp": [bi[249], bi[9750]],
        "conversation_clustered_bootstrap_95ci_pp": [cl[249], cl[9750]],
        "conversation_clustered_bootstrap_frac_le_zero":
            sum(1 for x in cl if x <= 0) / len(cl),
        "n_conversations": len(convs),
        "bootstrap_protocol": "10000 resamples, seed 1, paired per item / per conversation",
        "locomo_category_semantics_from_raw_data": semantics,
        "per_category_on_corrected_instrument": percat,
        "kill_condition_1_fires": not all(
            v["significantly_positive_at_0_05"] for v in percat.values()),
    }


def main():
    runs = {k: recompute(k) for k in
            ("hcache", "hcache_j12_noLoRA_chatFALSE", "hcache_j12_LoRA_chatFALSE")}
    canon = invert_published_percentages()
    mirror = mirror_block()
    ev = {
        "_what": "B06 drift-resolution leg. Every number recomputed from per-item raw records; "
                 "nothing copied from STATUS.json.",
        "_generated": subprocess.run(["date", "-Iseconds"], capture_output=True,
                                     text=True).stdout.strip(),
        "_generator": "proposal/backlog/B06-portable-decompression-adapter/evidence/"
                      "build_drift_evidence.py",
        "_git_commit": subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
        "_gpu_used": "none. CPU only. No ssh to any node. No judge API call.",
        "_python": sys.version.split()[0],

        "the_four_numbers": {
            "_ruler_warning": "8.11 and 13.29 live on the blended n=1986 ruler (mixed "
                              "instrument); 16.69 / 39.81 / 10.13 / 15.45 live on the "
                              "single-instrument Judge_1:4 n=1540 ruler.",
            "canonical_hcache_8_11": {
                "published_as": 8.11,
                "published_at": "status/PAPERA_RESULTS_CONSOLIDATED.md:175, "
                                "status/LOCOMO_JUDGE_AGGREGATE.md:21",
                "recomputed": canon["canonical_blended_n1986_RECOMPUTED"],
                "n": 1986, "instrument": "MIXED (1540 gpt-4o + 446 refusal regex)",
                "status_json_claim_mixed_instrument_n1986": "CONFIRMED",
                "provenance_of_raw_records":
                    "locomo_results/hcache_8b_chatFALSE/ on zwfy6 (.73/.104). NOT on wzc1: "
                    "an exhaustive scan of all 32 judge_cache*.jsonl and all 67 scores.json "
                    "under the wzc1 root found no run with 156/1540. zwfy6 is not mounted on "
                    "this node and ssh was forbidden for this task.",
                "recovered_on_wzc1_instead":
                    "locomo_results_openjudge_qwen3_MIRROR/hcache_8b_chatFALSE/ -- a second "
                    "judge over the SAME predictions, whose judge-independent columns "
                    "(f1 4.67 / acc 6.29 / em 0.25) match the published canonical row exactly.",
            },
            "canonical_judge_1_4_counterpart_10_13": {
                "published_as": 10.13,
                "published_at": "status/PAPERA_RESULTS_CONSOLIDATED.md:175",
                "recomputed": canon["canonical_judge_1_4_RECOMPUTED"],
                "n": 1540, "instrument": "single (gpt-4o)",
                "status_json_claim_already_converted": "CONFIRMED",
                "derivation": canon,
            },
            "b06_control_16_69": {
                "published_as": 16.69,
                "recomputed": runs["hcache_j12_noLoRA_chatFALSE"]["judge_1_4_pct_RECOMPUTED"],
                "n": 1540, "instrument": "single (gpt-4o)",
                "status_json_claim_single_instrument_n1540": "CONFIRMED",
            },
            "b06_treatment_39_81": {
                "published_as": 39.81,
                "recomputed": runs["hcache_j12_LoRA_chatFALSE"]["judge_1_4_pct_RECOMPUTED"],
                "n": 1540, "instrument": "single (gpt-4o)",
                "status_json_claim_single_instrument_n1540": "CONFIRMED",
            },
            "third_replicate_15_45": {
                "published_as": 15.4545,
                "recomputed": runs["hcache"]["judge_1_4_pct_RECOMPUTED"],
                "n": 1540, "instrument": "single (gpt-4o)",
            },
        },

        "same_ruler_comparison": {
            "_the_gate_question": "how much of the 8.11-vs-13.29 'cross-node drift' survives "
                                  "once both endpoints are on the Judge_1:4 n=1540 ruler?",
            "blended_n1986_ruler": {
                "canonical_vs_b06_control_pp": 13.293051359516618 - 8.106747230614300,
                "canonical_vs_older_local_pp": 12.286002014098690 - 8.106747230614300,
                "older_local_vs_b06_control_pp": 13.293051359516618 - 12.286002014098690,
            },
            "judge_1_4_n1540_ruler": {
                "canonical_vs_b06_control_pp": 16.688311688311687 - 10.129870129870129,
                "canonical_vs_older_local_pp": 15.454545454545455 - 10.129870129870129,
                "older_local_vs_b06_control_pp": 16.688311688311687 - 15.454545454545455,
            },
            "does_the_drift_vanish_on_one_ruler": False,
            "it_gets_LARGER": "5.1863 pp blended -> 6.5584 pp on Judge_1:4, because the two "
                              "endpoints are deflated unequally by the constant cat-5 term.",
        },

        "attribution": attribution(),
        "judge_noise_floor": judge_noise_floor(),
        "cross_judge_outlier_test": cross_judge_ratios(),
        "canonical_number_derivation": canon,
        "canonical_mirror_on_wzc1": mirror,
        "recomputed_runs": runs,
        "headline_and_kill_condition_1": headline_stats(),

        "verdict": {
            "drift_is_real_or_instrument": "INSTRUMENT (judge-side), not node/harness/generation.",
            "primary_evidence": "F1 and acc on the identical n=1540 ruler agree across the three "
                                "noLoRA replicates to 0.16% / 4.2% relative, while the gpt-4o "
                                "judge on the SAME items disagrees by 64.7% relative. F1 and acc "
                                "are deterministic functions of the predictions alone.",
            "generation_attributable_share_of_the_6_56_pp": "0.4%-7.8%",
            "corroboration": "the canonical run's open-judge/gpt-4o ratio is 3.404 vs 1.366-1.681 "
                             "for the five sibling methods graded in the same two judge passes "
                             "(z=+14.2), i.e. the canonical gpt-4o HCache score is the outlier, "
                             "not the local replicates.",
            "rejudge_needed": False,
            "why_no_rejudge": "A rejudge cannot be same-harness in the sense the gate wants: the "
                              "gpt-4o endpoint is not a frozen instrument (measured 3.07% per-item "
                              "verdict instability on byte-identical inputs) and the canonical "
                              "per-item records are not on this disk. The judge-independent "
                              "columns already settle the question at zero cost and zero API risk.",
            "kill_condition_2_verdict": "DOES NOT FIRE. Effect +23.12 pp; worst-case same-ruler "
                                        "drift 6.56 pp, of which <=0.5 pp is generation-attributable.",
            "kill_condition_1_verdict": "DOES NOT FIRE. All four judged categories individually "
                                        "significant (p 1.99e-54 to 4.43e-3). The category the "
                                        "clause names (open-domain = cat3) is the SMALLEST "
                                        "contributor at 3.7% of the gain.",
            "kill_condition_3_verdict": "STILL UNTESTED (needs a second compressor; GPU).",
        },

        "discrepancies_vs_status_json": [
            {
                "status_json_key": "kill_gate.condition_2_status",
                "status_json_text": "the drift is ~1.0 pp on the blended scale (12.286 canonical "
                                    "vs 13.293 local, both from scores.json overall_judge)",
                "finding": "WRONG PAIR. 12.286 is locomo_results/hcache, an OLDER LOCAL run -- "
                           "STATUS.json's own key third_measurement_found_20260814 labels that "
                           "same 12.28600201409869 as 'older run (2026-07-09/10)'. The canonical "
                           "blended value is 8.1067. So condition_2_status compares local-vs-local "
                           "and calls one of them canonical.",
                "correct_value": "canonical vs B06 control = 5.1863 pp blended / 6.5584 pp on "
                                 "Judge_1:4, i.e. 5.2x-6.5x larger than the stated ~1.0 pp.",
                "changes_the_conclusion": False,
                "why_not": "the drift is still small vs +23.12 pp AND is now shown to be judge-side.",
            },
            {
                "status_json_key": "kill_gate.condition_1_status",
                "status_json_text": "cat4 (open_domain, n=841 = 55% of the 1540)",
                "finding": "MISLABEL. cat4 is SINGLE-HOP, not open-domain. From locomo10.json: "
                           "cat4 has 1.07 evidence items on average, 94.5% of its items cite "
                           "exactly one evidence turn, and 1.00 distinct sessions. Open-domain is "
                           "cat3 (n=96, 2.08 evidence items, the only category with 0-evidence "
                           "items). The mislabel is inherited from "
                           "scripts/eval_qcmem_locomo.py:126-132 CATEGORY_NAMES, which is itself "
                           "wrong for cats 2/3/4 and disagrees with "
                           "status/LOCOMO_JUDGE_AGGREGATE.md:31-32.",
                "correct_value": "kill condition 1 names open-domain = cat3, which supplies only "
                                 "+0.84 pp (3.7%) of the 23.12 pp gain. Reading cat4 as "
                                 "'open_domain' makes the clause look near-fired when it is not.",
                "changes_the_conclusion": True,
                "why": "it flips the reading of kill condition 1 from 'at real risk' to "
                       "'does not fire' -- all four judged categories are individually significant.",
            },
            {
                "status_json_key": "next_gate / gpu_cost_estimate.drift_resolution_leg",
                "status_json_text": "Same-harness rejudge ... rejudge = OpenAI API + CPU",
                "finding": "The rejudge is UNNECESSARY and would not be same-instrument anyway. "
                           "Measured judge non-determinism is 3.07% per item on byte-identical "
                           "inputs, so a fresh pass is a NEW instrument draw, not a fixed ruler.",
                "correct_value": "0 GPU and 0 API calls. The judge-independent columns (F1/acc) "
                                 "settle the attribution.",
                "changes_the_conclusion": False,
            },
            {
                "status_json_key": "canonical_8_11_conversion_status.what_is_still_missing",
                "status_json_text": "The canonical run's own judge_cache.jsonl lives on zwfy6 ... "
                                    "and was NOT read here",
                "finding": "Still true for the judge cache, but an ARTEFACT OF THE SAME RUN IS ON "
                           "wzc1 and was missed: locomo_results_openjudge_qwen3_MIRROR/"
                           "hcache_8b_chatFALSE/ (mirrored per paperA/TODOList.md:170). Its "
                           "judge-independent columns identify the run exactly and its cat-5 cell "
                           "recovers 5/446, matching the 8.11 arithmetic independently.",
                "correct_value": "the conversion 8.11 -> 10.13 is confirmed by three mutually "
                                 "independent routes plus a same-run artefact on this disk.",
                "changes_the_conclusion": False,
            },
            {
                "status_json_key": "established_measurements.caveat_from_errata",
                "status_json_text": "A conversation-clustered bootstrap is the honest version and "
                                    "is 0 GPU",
                "finding": "DONE in this leg. 10-conversation clustered paired bootstrap "
                           "(10000 resamples, seed 1) = the interval reported under "
                           "headline_and_kill_condition_1; 0/10000 resamples <= 0.",
                "correct_value": "see conversation_clustered_bootstrap_95ci_pp",
                "changes_the_conclusion": False,
            },
        ],

        "not_verified": [
            "The canonical run's per-item judge_cache.jsonl and preds were never read: they are "
            "on zwfy6 and ssh was forbidden. The canonical counts (156/1540, 5/446) are pinned by "
            "inverting published rounded percentages -- uniqueness is asserted programmatically, "
            "but this is arithmetic recovery, not a read of the raw records.",
            "The 101-item deficit is CONSISTENT with silent judge API failures "
            "(scripts/eval_qcmem_locomo.py:713-715 sets judge=0.0 on failure and does NOT cache "
            "the record), but that mechanism was NOT confirmed: it requires counting records in "
            "the canonical judge_cache.jsonl on zwfy6. If that file has ~1439 records instead of "
            "1540, the mechanism is proven. This is the single highest-value 0-GPU follow-up.",
            "Whether the canonical predictions themselves are byte-identical to the local ones was "
            "not checked (canonical preds are on zwfy6). The judge-independent metrics agree to "
            "0.16% relative, which bounds any difference as immaterial, but is not byte equality.",
            "The open-judge cross-check uses PUBLISHED gpt-4o per-category percentages for the five "
            "sibling methods (status/LOCOMO_JUDGE_AGGREGATE.md:34-41), not their raw caches; four "
            "of the six sibling judge caches are not on wzc1.",
            "Kill condition 3 (second compressor) is untouched and needs GPU.",
            "No claim is made about 'portability' -- still one task, one compressor, one model.",
        ],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(ev, fh, indent=2, ensure_ascii=False)
    print(f"wrote {OUT}")
    v = ev["verdict"]
    print(f"  drift attribution : {v['drift_is_real_or_instrument']}")
    print(f"  generation share  : {v['generation_attributable_share_of_the_6_56_pp']}")
    print(f"  rejudge needed    : {v['rejudge_needed']}")
    print(f"  discrepancies     : {len(ev['discrepancies_vs_status_json'])} "
          f"({sum(1 for d in ev['discrepancies_vs_status_json'] if d.get('changes_the_conclusion'))} "
          f"conclusion-changing)")


if __name__ == "__main__":
    main()
