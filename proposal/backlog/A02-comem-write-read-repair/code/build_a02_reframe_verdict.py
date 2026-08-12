#!/usr/bin/env python
"""A02 reframe gate — build the verdict payload from measured cost + phase-1 quality.

Consumes:
  * `bench_results/a02_storage_readcompute/a02_storage_readcompute_aggregate.json`
    (this gate's measured cost/storage, produced by
    `bench_a02_storage_readcompute.py --mode aggregate`)
  * `proposal/active/A02-comem-write-read-repair/evidence/phase1_full_summary.json`
    (the phase-1 quality legs, already on disk)

Emits ONE json under the proposal's `evidence/` holding every number the verdict
markdown quotes, so no headline exists only in prose. (A03 was just caught with a
+0.48pp headline that lived only in two .md files; this file exists so A02 cannot
repeat that.)

DECISION RULE (fixed before the numbers were known)
---------------------------------------------------
The reframe is "CoMem is a storage / read-compute method for high-reuse
workloads". It SURVIVES only if, against the arm phase-1 actually used as C1:
  (a) CoMem's per-query cost is strictly lower (so a finite N* exists), AND
  (b) N* <= 1e5 queries per corpus (reachable in a real high-reuse deployment), AND
  (c) the storage premium is bounded enough to still call it a "storage" method.
Failing (a) or (b) => DEAD. Passing (a)+(b) but with a large storage premium =>
SURVIVES ONLY AS READ-COMPUTE, not as storage.

QUALITY IS HELD ALONGSIDE, NEVER POOLED
---------------------------------------
"Same quality, better cost" is the claim, so a cost win at degraded quality is
not a win. Phase-1 quality is therefore reported PER CELL. The pooled BABILong
-17.89pp and pooled LongEval +2.00pp are deliberately NOT surfaced as headline
numbers: each averages over cells with opposite true signs (BABILong: 4
significantly negative, 2 significantly POSITIVE; LongEval: -30pp at 8k vs +68pp
at 128k where C1 scores literally 0.0). This script classifies every cell by CI
sign and reports the counts instead of a mean.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PREREG_NSTAR_MAX = 1e5

# Clause (c) threshold. Pre-registered 2026-08-10 when the unexecuted-clause bug
# was found (the rule was stated in the docstring above but never evaluated in
# code, so this bar is being set AFTER the ratios were measured -- that is
# disclosed here rather than hidden).
#
# JUSTIFICATION, set on principle rather than to fit the measurement: the word
# "storage" in "CoMem is a storage method" is only meaningful if CoMem's store is
# within a small constant factor of what the baseline stores. RAG stores raw text
# at ~4 B/token. A 100x premium is already generous -- it admits ~400 B/token,
# roughly a fp16 vector of dim 200 per token. Anything that admits the measured
# 2048x (h12 = 8192 B/token) would make the claim vacuous, since at that point the
# method stores three orders of magnitude more than the thing it replaces and the
# only honest framing is read-compute.
PREREG_STORAGE_PREMIUM_MAX = 100.0


def _sign(cell):
    lo, hi = cell["ci_lo"], cell["ci_hi"]
    if lo > 0 and hi > 0:
        return "c2_better"
    if lo < 0 and hi < 0:
        return "c1_better"
    return "ns"


def summarise_quality(phase1):
    """Per-cell CI-sign classification. No pooling across sign-discordant cells."""
    out = {}
    for bench, rec in phase1["benchmarks"].items():
        by = None
        for k in ("by_cell", "by_length", "by_dataset"):
            if k in rec:
                by = rec[k]
                break
        cells = {}
        if by:
            for name, c in by.items():
                cells[name] = {"c1": c["c1"], "c2": c["c2"],
                               "diff_pt": c["diff_pt"],
                               "ci": [c["ci_lo"], c["ci_hi"]],
                               "verdict": _sign(c)}
        counts = {"c1_better": 0, "c2_better": 0, "ns": 0}
        for c in cells.values():
            counts[c["verdict"]] += 1
        out[bench] = {
            "per_cell": cells,
            "counts": counts,
            "n_pairs": rec.get("n_pairs"),
            "pooled_diff_pt_DO_NOT_QUOTE": rec.get("diff_pt"),
            "pooled_is_sign_discordant": counts["c1_better"] > 0
                                         and counts["c2_better"] > 0,
        }
    return out


def build(agg, phase1):
    q = summarise_quality(phase1)

    # ---- the decisive comparison: comem vs the phase-1 C1 (pack-all) --------
    decisive = {}
    for cell_key, cell in agg["cells"].items():
        per_g = {}
        for G, xo in cell["crossover"].items():
            main = xo.get("comem_vs_c1_all", {})
            ctrl = xo.get("comem_vs_j0_top12", {})
            retr = xo.get("j0_top12_vs_c1_all", {})
            per_g[G] = {
                "comem_vs_c1_all": {
                    "n_star": main.get("n_star"),
                    "n_star_infinite": main.get("n_star_infinite"),
                    "reachable_within_prereg": main.get("reachable_within_prereg"),
                    "per_query_s": main.get("per_query_s"),
                    "one_time_s": main.get("one_time_s"),
                },
                "comem_vs_j0_top12_matched_pack": {
                    "n_star": ctrl.get("n_star"),
                    "n_star_infinite": ctrl.get("n_star_infinite"),
                    "reachable_within_prereg": ctrl.get("reachable_within_prereg"),
                    "per_query_s": ctrl.get("per_query_s"),
                },
                "j0_top12_vs_c1_all_retrieval_only": {
                    "n_star": retr.get("n_star"),
                    "per_query_s": retr.get("per_query_s"),
                },
                "latency_decomposition": cell["per_G"][G],
            }
        decisive[cell_key] = {
            "store_L_tokens": cell["store_L_tokens"],
            "n_store_chunks": cell["n_store_chunks"],
            "n_procs": cell["n_procs"],
            "one_time_s": cell["one_time"],
            "select_s": cell["select_s"],
            "storage": cell["storage"],
            "peak_gb": cell["peak_gb"],
            "c1_all_status": cell.get("c1_all_status"),
            "per_G": per_g,
        }

    # ---- verdict ------------------------------------------------------------
    # Gate on the DECISIVE pairing (vs the phase-1 C1) at every measured cell.
    # THREE outcomes, never conflated:
    #   finite   : c1_all ran and CoMem is cheaper per query -> N* exists.
    #   infinite : c1_all ran and CoMem is NOT cheaper -> no N* amortises.
    #   absent   : c1_all could not run at all (OOM). This is evidence ABOUT
    #              c1_all (it does not fit), not a missing measurement, and it is
    #              reported separately so it is never scored as a CoMem win by
    #              default nor silently dropped.
    finite, reachable, infinite, absent = [], [], [], []
    for ck, c in decisive.items():
        for G, v in c["per_G"].items():
            m = v["comem_vs_c1_all"]
            tag = f"{ck}|G={G}"
            if m.get("reason") or m["n_star"] is None and not m.get("n_star_infinite"):
                absent.append({"cell": tag, "c1_all_status": c["c1_all_status"]})
            elif m["n_star_infinite"]:
                infinite.append(tag)
            else:
                finite.append((tag, m["n_star"]))
                if m["reachable_within_prereg"]:
                    reachable.append((tag, m["n_star"]))

    ratios = [c["storage"]["ratio_h12_over_raw"] for c in decisive.values()
              if c["storage"].get("ratio_h12_over_raw")]
    tot_ratios = [c["storage"]["ratio_comem_total_over_rag_total"]
                  for c in decisive.values()
                  if c["storage"].get("ratio_comem_total_over_rag_total")]

    cost_survives = len(reachable) > 0

    # --- clause (c): the storage-premium bound ---------------------------------
    # BUG FIXED 2026-08-10. Clause (c) is stated in this module's decision rule and
    # its inputs were computed just above (`ratios`, `tot_ratios`), but the verdict
    # was gated on `cost_survives` ALONE. The gate therefore emitted a bare
    # "SURVIVES" while the storage premium went unchecked -- asserting the OPPOSITE
    # of the human verdict in A02_STORAGE_READCOMPUTE_VERDICT.md ("STORAGE FORM:
    # DEAD"). Clause (c) is now actually evaluated, and every clause records the
    # value tested plus its threshold so a skipped clause cannot recur silently.
    max_ratio = max(ratios) if ratios else None
    storage_survives = (max_ratio is not None
                        and max_ratio <= PREREG_STORAGE_PREMIUM_MAX)

    if not cost_survives:
        verdict = "DEAD"
    elif storage_survives:
        verdict = "SURVIVES"
    else:
        # Exactly the case the docstring already describes in words:
        # "Passing (a)+(b) but with a large storage premium => SURVIVES ONLY AS
        # READ-COMPUTE, not as storage."
        verdict = "SURVIVES_AS_READ_COMPUTE_ONLY"

    return {
        "gate": "A02_storage_readcompute_reframe",
        "decision_rule": {
            "survives_iff": "finite N* vs phase-1 C1 AND N* <= 1e5 AND bounded "
                            "storage premium",
            "prereg_nstar_max": PREREG_NSTAR_MAX,
            "prereg_storage_premium_max": PREREG_STORAGE_PREMIUM_MAX,
        },
        "verdict": verdict,
        "clause_evaluation": {
            "a_finite_nstar": {
                "passed": len(finite) > 0,
                "value_tested": len(finite),
                "threshold": "> 0 cells with finite N* vs phase-1 C1",
            },
            "b_nstar_reachable": {
                "passed": cost_survives,
                "value_tested": len(reachable),
                "threshold": f"> 0 cells with N* <= {PREREG_NSTAR_MAX:g}",
            },
            "c_storage_premium_bounded": {
                "passed": storage_survives,
                "value_tested": max_ratio,
                "threshold": f"max(h12/raw ratio) <= "
                             f"{PREREG_STORAGE_PREMIUM_MAX:g}",
                "note": "Computed but NOT gated on before the 2026-08-10 fix; "
                        "see the block comment in build_a02_reframe_verdict.py.",
            },
        },
        "verdict_basis": {
            "n_cells_with_finite_nstar_vs_c1": len(finite),
            "n_cells_reachable_within_prereg": len(reachable),
            "n_cells_infinite_nstar": len(infinite),
            "n_cells_c1_arm_absent": len(absent),
            "infinite_cells": infinite,
            "c1_absent_cells": absent,
            "finite_nstar_values": dict(finite),
            "reachable_nstar_values": dict(reachable),
            "storage_ratio_h12_over_raw_range": ([min(ratios), max(ratios)]
                                                 if ratios else None),
            "storage_ratio_total_range": ([min(tot_ratios), max(tot_ratios)]
                                          if tot_ratios else None),
        },
        "cost": decisive,
        "quality_phase1_per_cell": q,
        "quality_note": "Pooled BABILong/LongEval means are recorded under "
                        "`pooled_diff_pt_DO_NOT_QUOTE` ONLY to make their "
                        "exclusion auditable; they average over cells with "
                        "opposite true signs and must never be quoted.",
        "provenance": {
            "cost_source": "bench_results/a02_storage_readcompute/"
                           "a02_storage_readcompute_aggregate.json",
            "quality_source": "proposal/active/A02-comem-write-read-repair/"
                              "evidence/phase1_full_summary.json",
            "driver": "proposal/active/A02-comem-write-read-repair/code/"
                      "bench_a02_storage_readcompute.py",
            "n_cost_files": agg.get("n_files"),
            "partial_cells": agg.get("partial_cells"),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", required=True)
    ap.add_argument("--phase1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.aggregate) as f:
        agg = json.load(f)
    with open(a.phase1) as f:
        phase1 = json.load(f)
    payload = build(agg, phase1)
    Path(os.path.dirname(a.out)).mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=2)
    vb = payload["verdict_basis"]
    print(f"[a02][verdict] {payload['verdict']}")
    print(f"  finite N* cells        : {vb['n_cells_with_finite_nstar_vs_c1']}")
    print(f"  reachable (<=1e5) cells: {vb['n_cells_reachable_within_prereg']}")
    print(f"  infinite N* cells      : {vb['n_cells_infinite_nstar']}")
    print(f"  c1_all-absent cells    : {vb['n_cells_c1_arm_absent']} "
          f"(arm could not run; NOT scored as a CoMem win)")
    print(f"  storage h12/raw range  : {vb['storage_ratio_h12_over_raw_range']}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
