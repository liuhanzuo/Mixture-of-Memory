#!/usr/bin/env python
"""Paper B P1.2 — clean-subset re-scoring of OLMo-2 closed-book QA predictions.

CPU-ONLY aggregation. NO model run. Given EXISTING per-example prediction dumps
from ``scripts/eval_olmo2_closedbook_qa.py`` (``per_example_{task}[_shard*of*].jsonl``
with fields ``item_id``/``pred``/``gold``/``em``/``contains``/``f1``) and the
Dolmino contamination CLEAN-id list produced by
``scripts/audit_olmo2_dolmino_contamination.py`` (``clean_subset_ids.json``),
recompute PopQA / TriviaQA / NQ-open accuracy on ONLY the decontaminated
(``keep_for_recompute`` = all-except-CONTAMINATED) examples.

This is the B-P1.2 remediation step: the closed-book knowledge table must be
reported on questions that are NOT contaminated by Dolmino (OLMo-2's mid-train
corpus). Numbers are produced with the EXACT same scorer that made the paper
numbers (``eval_olmo2_closedbook_qa.score_prediction``, imported when available;
otherwise the per-example ``em/contains/f1`` fields — which are that scorer's own
stored output — are aggregated). No fabrication, no new normalise/match logic.

ID JOIN (critical)
------------------
Per-example ``item_id`` is the GLOBAL positional index into the eval's loaded
example list (``item_id = shard_index + li*num_shards``).
The audit ``qid`` is:
  * triviaqa / nq_open : ``enumerate`` index over the SAME raw dataset (no
    empty-answer filtering happened -> line counts match exactly), so
    ``item_id == qid`` and the join is exact.
  * popqa : the NATIVE PopQA ``ex["id"]`` (large ids), which is NOT the
    positional ``item_id``. Those two id spaces are disjoint, so a reliable join
    is impossible. Per the task spec, PopQA is reported ``clean == full`` with
    ``"popqa_join": "unmatched_report_full"`` (also numerically exact here since
    PopQA has 0 contaminated ids). The join is detected generically: if the
    prediction/clean id overlap is tiny, we fall back to full.

USAGE
-----
  CUDA_VISIBLE_DEVICES=-1 python scripts/recompute_closedbook_clean_subset.py \
      --clean_ids   bench_results/olmo2_dolmino_contamination/clean_subset_ids.json \
      --results_root olmo2_closedbook_results \
      --ngram n13 \
      --out bench_results/olmo2_dolmino_contamination/closedbook_clean_subset_recomputed.json

Auto-discovers every arm dir under ``--results_root``: ``<arm>/`` supplies popqa
& triviaqa, ``<arm>_nqopen/`` supplies nq_open. Only arms with predictions
present are scored.
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

# Reuse the EXACT closed-book scorer (normalise + max-over-alias em/contains/f1).
# Import is optional: it pulls the torch/transformers chain (never touches a GPU
# at import), so if it fails we fall back to the per-example em/contains/f1
# fields, which are this same scorer's stored output.
try:
    from eval_olmo2_closedbook_qa import score_prediction as _score_prediction
except Exception:  # noqa: BLE001
    _score_prediction = None

# Headline metric per dataset (matches how the eval / paper reports them):
#   popqa -> "contains" (PopQA's own accuracy convention),
#   triviaqa / nq_open -> "em".
HEADLINE = {"popqa": "contains", "triviaqa": "em", "nq_open": "em"}
BASE_TASKS = ["popqa", "triviaqa"]
NQ_TASK = "nq_open"


def load_preds(pred_dir: str, task: str):
    """Load per-example records keyed by item_id. Prefer the merged
    per_example_{task}.jsonl; else merge/dedup the shard files."""
    if not pred_dir or not os.path.isdir(pred_dir):
        return None
    merged = os.path.join(pred_dir, f"per_example_{task}.jsonl")
    files = [merged] if os.path.isfile(merged) else sorted(
        glob.glob(os.path.join(pred_dir, f"per_example_{task}_shard*of*.jsonl")))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None
    preds = {}
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                iid = r.get("item_id")
                if iid is None or iid in preds:
                    continue
                preds[int(iid)] = r
    return preds or None


def _metrics(r: dict) -> tuple[int, int, float]:
    """(em, contains, f1) for one record. Recompute via the imported scorer when
    pred+gold present (faithful); else use the stored fields."""
    if _score_prediction is not None and "pred" in r and "gold" in r:
        s = _score_prediction(r.get("pred", ""), list(r.get("gold", [])))
        return int(s["em"]), int(s["contains"]), float(s["f1"])
    return int(r.get("em", 0)), int(r.get("contains", 0)), float(r.get("f1", 0.0))


def _agg(rows) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "em": None, "contains": None, "f1": None}
    em = sum(m[0] for m in rows)
    co = sum(m[1] for m in rows)
    f1 = sum(m[2] for m in rows)
    return {"n": n,
            "em": round(100.0 * em / n, 3),
            "contains": round(100.0 * co / n, 3),
            "f1": round(100.0 * f1 / n, 3)}


def score_arm_task(preds: dict, task: str, clean_ids):
    """Return a result row: full metrics, clean metrics, headline, join status."""
    all_metrics = {iid: _metrics(r) for iid, r in preds.items()}
    full = _agg(list(all_metrics.values()))

    pred_ids = set(all_metrics)
    clean_set = set(int(i) for i in clean_ids)
    overlap = len(pred_ids & clean_set)
    # Generic join-validity check: if the clean ids barely intersect the
    # prediction item_ids, the id spaces differ (PopQA native-id case) -> we
    # cannot reliably filter, so report clean == full.
    join_ok = (len(pred_ids) > 0 and overlap >= 0.5 * len(pred_ids))

    hd = HEADLINE[task]
    if not join_ok:
        clean = dict(full)  # clean == full
        join = "unmatched_report_full"
    else:
        clean_rows = [all_metrics[i] for i in pred_ids if i in clean_set]
        clean = _agg(clean_rows)
        join = "matched"

    row = {
        "task": task,
        "headline_metric": hd,
        "full_score": full[hd], "full_n": full["n"],
        "clean_score": clean[hd], "clean_n": clean["n"],
        "full_em": full["em"], "full_contains": full["contains"], "full_f1": full["f1"],
        "clean_em": clean["em"], "clean_contains": clean["contains"], "clean_f1": clean["f1"],
        "n_dropped": full["n"] - clean["n"],
        "n_clean_ids": len(clean_set),
        "id_overlap": overlap,
        "join": join,
    }
    if task == "popqa":
        row["popqa_join"] = join
    return row


def discover_arms(results_root: str):
    """Return {arm: {"base": <dir or None>, "nq": <dir or None>}}."""
    arms: dict[str, dict] = {}
    if not os.path.isdir(results_root):
        return arms
    for d in sorted(os.listdir(results_root)):
        full = os.path.join(results_root, d)
        if not os.path.isdir(full):
            continue
        if d.endswith("_nqopen"):
            arm = d[: -len("_nqopen")]
            arms.setdefault(arm, {"base": None, "nq": None})["nq"] = full
        else:
            arms.setdefault(d, {"base": None, "nq": None})["base"] = full
    return arms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean_ids",
                    default="bench_results/olmo2_dolmino_contamination/clean_subset_ids.json")
    ap.add_argument("--results_root", default="olmo2_closedbook_results")
    ap.add_argument("--ngram", default="n13",
                    help="which n-gram audit slice of clean ids to use (n13/n8)")
    ap.add_argument("--out",
                    default="bench_results/olmo2_dolmino_contamination/"
                            "closedbook_clean_subset_recomputed.json")
    args = ap.parse_args()

    clean_all = json.load(open(args.clean_ids))
    # keep_for_recompute = all qids EXCEPT contaminated (standard decontaminated).
    clean_ids = {ds: clean_all[ds][args.ngram]["keep_for_recompute_ids"]
                 for ds in ("popqa", "triviaqa", "nq_open") if ds in clean_all}

    arms = discover_arms(args.results_root)
    out = {
        "clean_ids_source": os.path.abspath(args.clean_ids),
        "ngram": args.ngram,
        "clean_subset_note": "clean = keep_for_recompute (all qids EXCEPT CONTAMINATED)",
        "results_root": os.path.abspath(args.results_root),
        "scorer": "eval_olmo2_closedbook_qa.score_prediction"
                  if _score_prediction is not None else "per_example_fields_fallback",
        "clean_subset_size": {ds: len(v) for ds, v in clean_ids.items()},
        "n_contaminated": {ds: len(clean_all[ds][args.ngram]["contaminated_ids"])
                           for ds in clean_ids},
        "results": [],
    }

    for arm in sorted(arms):
        dirs = arms[arm]
        plan = [(t, dirs["base"]) for t in BASE_TASKS] + [(NQ_TASK, dirs["nq"])]
        for task, d in plan:
            if task not in clean_ids:
                continue
            preds = load_preds(d, task)
            if preds is None:
                out["results"].append({"arm": arm, "task": task,
                                       "pred_dir": d, "status": "NO_PREDICTIONS"})
                print(f"[recompute] {arm} / {task}: no predictions -> skip")
                continue
            row = {"arm": arm}
            row.update(score_arm_task(preds, task, clean_ids[task]))
            row["pred_dir"] = d
            out["results"].append(row)
            print(f"[recompute] {arm} / {task}: "
                  f"full {row['headline_metric']}={row['full_score']} (n={row['full_n']})"
                  f" -> CLEAN={row['clean_score']} (n={row['clean_n']}) [{row['join']}]")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[recompute] wrote {args.out}  ({len(out['results'])} arm×dataset rows)")


if __name__ == "__main__":
    main()
