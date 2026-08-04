#!/usr/bin/env python
"""Paper A A-P1.2(a) — clean-subset re-scoring across benchmarks (CPU only).

Given the CLEAN-subset id manifest produced by
``scripts/audit_ap1_2_contamination.py`` (``clean_subset_ids.json``) and the
EXISTING per-example prediction files already on this wzc1 node, recompute each
benchmark's official metric on ONLY the documents that are NOT contaminated by the
PG-19 training corpus — no model is run.

The cross-benchmark audit found that the ONLY materially contaminated benchmark is
LongBench ``narrativeqa`` (Project-Gutenberg books shared with PG-19: 96/200 records
at containment ~1.0). LoCoMo (synthetic dialogue) and LongEval (synthetic lines) are
clean, so their clean subset == full set and the re-score is an identity check.
InfiniteBench was audited by P0.14; its raw predictions live on an off-limits GPU
node, so it is reported NO_PREDICTIONS here (see bench_results/p0_14_contamination/).

Scorers are reused VERBATIM from the drivers that produced the paper numbers:
  * LongBench : scripts.eval_longbench_mem_space.compute_f1_multi / compute_em_multi
                (run_scoring口径: dedup preds by ``index``, macro-avg F1/EM*100).
  * LoCoMo    : scripts.eval_qcmem_locomo.score_sample (dedup by ``id``; overall
                F1/EM/acc*100, plus LLM-judge from judge_cache.jsonl when present).
  * InfiniteBench : scripts.eval_qcmem_infbench.score_one (P0.14 pattern).

USAGE
-----
  CUDA_VISIBLE_DEVICES=-1 python scripts/recompute_ap1_2_clean_subset.py \
      --clean_ids bench_results/ap1_2_contamination/clean_subset_ids.json \
      --longbench_root longbench_results \
      --locomo_arms qcmem_8b_zs_j9_iter_chatFALSE=locomo_results/qcmem_8b_zs_j9_iter_chatFALSE \
                    kvdirect=locomo_results/kvdirect \
      --out bench_results/ap1_2_contamination/clean_subset_recomputed.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scripts.eval_longbench_mem_space as lb  # noqa: E402
import scripts.eval_qcmem_locomo as lc  # noqa: E402


# --------------------------------------------------------------------------- #
# LongBench
# --------------------------------------------------------------------------- #
def _lb_load_task(pred_dir: str, task: str):
    """Merge {task}_*.jsonl, dedup by index — identical to lb.run_scoring."""
    files = sorted(glob.glob(os.path.join(pred_dir, f"{task}_*.jsonl")))
    files = [f for f in files if "eval_config" not in os.path.basename(f)]
    if not files:
        return None
    preds, seen = {}, set()
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                idx = item.get("index", len(preds))
                if idx in seen:
                    continue
                seen.add(idx)
                preds[int(idx)] = item
    return preds


def _lb_score(preds: dict, keep_idx=None):
    rows = ([preds[i] for i in preds if i in keep_idx] if keep_idx is not None
            else list(preds.values()))
    if not rows:
        return None, None, 0
    f1s, ems = [], []
    for r in rows:
        ans = r.get("answers", [])
        if isinstance(ans, str):
            try:
                ans = json.loads(ans)
            except json.JSONDecodeError:
                ans = [ans]
        if not isinstance(ans, list):
            ans = [ans]
        f1s.append(lb.compute_f1_multi(r.get("pred", ""), ans))
        ems.append(lb.compute_em_multi(r.get("pred", ""), ans))
    return (100.0 * sum(f1s) / len(f1s), 100.0 * sum(ems) / len(ems), len(rows))


def recompute_longbench(clean_ids: dict, longbench_root: str, arms):
    by_task_clean = clean_ids["longbench"]["by_task"]
    out = []
    for arm_name, pred_dir in arms:
        if not os.path.isdir(pred_dir):
            out.append({"arm": arm_name, "pred_dir": pred_dir,
                        "status": "NO_DIR"})
            continue
        per_task = {}
        macro_full, macro_clean = [], []
        for task, cinfo in by_task_clean.items():
            preds = _lb_load_task(pred_dir, task)
            if preds is None:
                continue
            keep = set(int(i) for i in cinfo.get("indices", []))
            f_f1, f_em, f_n = _lb_score(preds, None)
            c_f1, c_em, c_n = _lb_score(preds, keep)
            per_task[task] = {
                "full_f1": round(f_f1, 2) if f_f1 is not None else None,
                "full_em": round(f_em, 2) if f_em is not None else None,
                "full_n": f_n,
                "clean_f1": round(c_f1, 2) if c_f1 is not None else None,
                "clean_em": round(c_em, 2) if c_em is not None else None,
                "clean_n": c_n,
                "n_dropped": f_n - c_n,
            }
            if f_f1 is not None:
                macro_full.append(f_f1)
            if c_f1 is not None:
                macro_clean.append(c_f1)
        if not per_task:
            out.append({"arm": arm_name, "pred_dir": pred_dir,
                        "status": "NO_PREDICTIONS"})
            continue
        out.append({
            "arm": arm_name, "pred_dir": pred_dir, "status": "OK",
            "macro_f1_full": round(sum(macro_full) / len(macro_full), 2) if macro_full else None,
            "macro_f1_clean": round(sum(macro_clean) / len(macro_clean), 2) if macro_clean else None,
            "n_tasks": len(per_task),
            "per_task": per_task,
        })
    return out


# --------------------------------------------------------------------------- #
# LoCoMo
# --------------------------------------------------------------------------- #
def _locomo_load(pred_dir: str):
    files = sorted(glob.glob(os.path.join(pred_dir, "preds*.jsonl")))
    if not files:
        return None
    preds, seen = {}, set()
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                rid = item.get("id")
                if rid is None or rid in seen:
                    continue
                seen.add(rid)
                preds[rid] = item
    # attach cached LLM judge if available
    jc = os.path.join(pred_dir, "judge_cache.jsonl")
    if os.path.exists(jc):
        with open(jc) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                j = json.loads(line)
                if j.get("id") in preds and "judge" in j:
                    preds[j["id"]]["judge"] = float(j["judge"])
    return preds


def _locomo_score(preds: dict, keep_ids=None):
    ids = [i for i in preds if (keep_ids is None or i in keep_ids)]
    if not ids:
        return None
    f1 = em = acc = 0.0
    judge_vals = []
    for i in ids:
        sc = lc.score_sample(preds[i])
        f1 += sc["f1"]
        em += sc["em"]
        acc += sc["acc"]
        if "judge" in preds[i]:
            judge_vals.append(float(preds[i]["judge"]))
    n = len(ids)
    res = {"f1": round(100.0 * f1 / n, 2), "em": round(100.0 * em / n, 2),
           "acc": round(100.0 * acc / n, 2), "n": n}
    if judge_vals:
        res["judge"] = round(100.0 * sum(judge_vals) / len(judge_vals), 2)
        res["judge_n"] = len(judge_vals)
    return res


def recompute_locomo(clean_ids: dict, arms):
    clean_qa = set(clean_ids["locomo"].get("qa_ids", []))
    out = []
    for arm_name, pred_dir in arms:
        if not os.path.isdir(pred_dir):
            out.append({"arm": arm_name, "pred_dir": pred_dir, "status": "NO_DIR"})
            continue
        preds = _locomo_load(pred_dir)
        if not preds:
            out.append({"arm": arm_name, "pred_dir": pred_dir,
                        "status": "NO_PREDICTIONS"})
            continue
        full = _locomo_score(preds, None)
        clean = _locomo_score(preds, clean_qa)
        out.append({"arm": arm_name, "pred_dir": pred_dir, "status": "OK",
                    "full": full, "clean": clean,
                    "n_dropped": full["n"] - clean["n"]})
    return out


# --------------------------------------------------------------------------- #
def _parse_arms(items):
    arms = []
    for it in items or []:
        if "=" in it:
            name, d = it.split("=", 1)
        else:
            name, d = os.path.basename(it.rstrip("/")), it
        arms.append((name, d))
    return arms


def _auto_longbench_arms(root: str):
    """Every arm dir under `root` that has a narrativeqa prediction file (the only
    contaminated task) — those are the arms whose LongBench numbers can be inflated."""
    arms = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "narrativeqa_*.jsonl")):
            arms.append((os.path.basename(d), d))
    return arms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean_ids",
                    default="bench_results/ap1_2_contamination/clean_subset_ids.json")
    ap.add_argument("--longbench_root", default="longbench_results",
                    help="auto-discover every arm dir with narrativeqa preds.")
    ap.add_argument("--longbench_arms", nargs="*", default=None,
                    help="explicit name=dir list (overrides auto-discovery).")
    ap.add_argument("--locomo_arms", nargs="*", default=None,
                    help="name=dir list of LoCoMo prediction dirs.")
    ap.add_argument("--infb_note", action="store_true", default=True)
    ap.add_argument("--out",
                    default="bench_results/ap1_2_contamination/clean_subset_recomputed.json")
    args = ap.parse_args()

    clean_ids = json.load(open(args.clean_ids))

    lb_arms = (_parse_arms(args.longbench_arms) if args.longbench_arms
               else _auto_longbench_arms(args.longbench_root))
    loc_arms = _parse_arms(args.locomo_arms) if args.locomo_arms else [
        ("qcmem_8b_zs_j9_iter_chatFALSE", "locomo_results/qcmem_8b_zs_j9_iter_chatFALSE"),
        ("kvdirect", "locomo_results/kvdirect"),
    ]

    out = {
        "clean_ids_source": os.path.abspath(args.clean_ids),
        "note": "Only LongBench narrativeqa is materially contaminated vs PG-19; "
                "LoCoMo/LongEval are clean (clean==full identity check); "
                "InfiniteBench predictions are off-node (see p0_14_contamination).",
        "longbench": recompute_longbench(clean_ids, args.longbench_root, lb_arms),
        "locomo": recompute_locomo(clean_ids, loc_arms),
        "infinitebench": {
            "status": "NO_PREDICTIONS_ON_THIS_NODE",
            "clean_subset_size": {k: len(v) for k, v in
                                  clean_ids["infinitebench"]["by_task"].items()},
            "note": "raw predictions live on an off-limits GPU node; the clean-subset "
                    "recompute is in bench_results/p0_14_contamination/clean_subset_recomputed.json",
        },
        "longeval": {"status": "CLEAN_BY_CONSTRUCTION",
                     "note": "synthetic lines-retrieval; no contamination possible; "
                             "clean subset == full set."},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ---- human-readable log ---- #
    print("\n===== A-P1.2(a) CLEAN-SUBSET RE-SCORE =====")
    print("\n[LongBench] (only narrativeqa changes; macro-F1 over each arm's tasks)")
    for a in out["longbench"]:
        if a.get("status") != "OK":
            print(f"  {a['arm']:34s} {a.get('status')}")
            continue
        nq = a["per_task"].get("narrativeqa", {})
        print(f"  {a['arm']:34s} macroF1 full={a['macro_f1_full']} -> clean={a['macro_f1_clean']}"
              f"  | narrativeqa F1 full={nq.get('full_f1')} (n={nq.get('full_n')}) "
              f"-> clean={nq.get('clean_f1')} (n={nq.get('clean_n')})")
    print("\n[LoCoMo] (fully clean -> full == clean)")
    for a in out["locomo"]:
        if a.get("status") != "OK":
            print(f"  {a['arm']:34s} {a.get('status')}")
            continue
        fu, cl = a["full"], a["clean"]
        jg = (f"  judge full={fu.get('judge')} -> clean={cl.get('judge')}"
              if "judge" in fu else "")
        print(f"  {a['arm']:34s} F1 full={fu['f1']} -> clean={cl['f1']}  "
              f"acc full={fu['acc']} -> clean={cl['acc']} (n {fu['n']}->{cl['n']}){jg}")
    print(f"\n[InfiniteBench] {out['infinitebench']['status']} "
          f"(clean subset {out['infinitebench']['clean_subset_size']})")
    print(f"[LongEval] {out['longeval']['status']}")
    print(f"\n[recompute] wrote {args.out}")


if __name__ == "__main__":
    main()
