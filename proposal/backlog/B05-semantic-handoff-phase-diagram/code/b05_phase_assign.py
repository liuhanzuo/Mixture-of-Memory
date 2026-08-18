#!/usr/bin/env python
"""B05 PHASE ASSIGNMENT + KILL-GATE ADJUDICATOR.  Pure CPU, 0 GPU.

This file IS the operational form of `PHASE_SEPARATION_PREREG.md` sections 3.1
(CELL->RUNG), 3.2 (KILL GATE), 3.3 (monotonicity guard) and 2.1 (FLOOR).  It is
written PRE-DATA: at authoring time zero B05 cells exist on disk, so nothing
here can have been tuned to a result.

WHY THIS FILE EXISTS AT ALL
---------------------------
The first B05 gate draft was refuted 3/3 by adversarial review, and TWO lenses
returned `decidable=False`.  Both defects were the same species: the gate was
prose that no program could evaluate.

  * lens `decidability`: "a rung has no defined phase label when its 4 cells
    disagree."  -> fixed by `rung_label()`: >=3 of 4 cells, else MIXED.
  * lens `falsifiability`: the cross-task clause had no monotonicity guard, so
    "a non-monotone scatter [could] be read as separation."  -> fixed by
    `cell_is_monotone()` + `SEVERITY`, over the ordered index list
    `RUNG_INDEX` = [BELOW, 6, 9, 12, 18] = 0..4.

A THIRD defect, which no lens raised and which this file also fixes:
`FLOOR = 4.0 pp + bootstrap-percentile CI excluding 0` was calibrated on a
comparator cell that the bootstrap called SIGNIFICANT and an EXACT test calls
not significant.  See `selftest_floor_calibration()` for the arithmetic.  The
floor is therefore raised to 6.0 pp with an exact test.  This is a
*directional* change and it is declared: it makes Phase I easier to populate
and thus makes ONE kill route (Phase-I-empty) harder to trigger, while the
monotonicity guard adds a strictly new one.  Both changes are pre-data.

STATISTICS: exact McNemar, NOT the paired bootstrap
---------------------------------------------------
Two independent reasons, both measured:

1. ANTI-CONSERVATIVE ON SPARSE DISCORDANCE.  Every one of the 20 comparator
   cells in A02's per-item file has `n10 == 0` (the arm never beats the anchor
   on an item the anchor missed).  On a 4-vs-0 discordant table the exact
   two-sided binomial p is 0.125, but A02's percentile bootstrap reported
   CI95 = [-8, -1], i.e. SIG.  A floor derived from that call inherits a
   false positive.
2. NODE-DEPENDENT.  `memory/numpy-version-split-breaks-cross-node-bootstrap`
   records three numpy versions across the five nodes (LOCAL 2.3.5 / .82 2.4.6
   / rest 2.5.1) and a same-seed `multinomial` divergence.  A bootstrap-derived
   phase label could therefore FLIP with the node the analyzer happens to run
   on.  `math.comb` is exact integer arithmetic: node-invariant by construction.

The bootstrap CI is still COMPUTED and EMITTED for continuity with A02's table,
but it is never an input to a phase label.  `stat_decides_label` records which
test decided, in every cell, so the substitution is auditable.

USAGE
  python b05_phase_assign.py --selftest          # no data needed; run this now
  python b05_phase_assign.py --out <evidence_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb
from pathlib import Path

# --------------------------------------------------------------------------- #
# PRE-REGISTERED CONSTANTS.  Changing any of these after the first B05 result
# file exists on disk voids the prereg (PHASE_SEPARATION_PREREG.md sec 4).
# --------------------------------------------------------------------------- #

FLOOR_PP = 6.0          # prereg 2.1 -- minimum detectable one-directional
                        # effect of the exact test at n=100 (see selftest)
ALPHA = 0.05            # exact two-sided McNemar
N_EXPECT = 100          # items per cell
R_DENOM_MIN_PP = 10.0   # prereg 2.2 -- recovery-fraction denominator guard

# prereg 3.2 / lens `falsifiability`: the ORDERED rung index.  Index 0 is the
# j=0 anchor position, named BELOW because it sits below the shallowest B05
# rung.  It is Phase I BY CONSTRUCTION (A0 vs A0 = 0.0 pp; A1 vs A0 = 0 flips
# out of 400 paired items, A02 GATE 0) and is EXCLUDED from every
# phase-non-emptiness test -- it exists only to anchor the monotonicity check,
# which is undefined without a known-Phase-I left endpoint.
RUNG_INDEX = [("BELOW", 0), ("6", 6), ("9", 9), ("12", 12), ("18", 18)]
LADDER_J = [6, 9, 12, 18]          # the rungs that may satisfy non-emptiness
BELOW_IDX = 0

PRIMARY_CELLS = [
    ("niah_multikey_1", "16k"),
    ("niah_multikey_1", "32k"),
    ("variable_tracking", "16k"),
    ("variable_tracking", "32k"),
]
TASKS = ["niah_multikey_1", "variable_tracking"]

# Phase severity.  Monotone == severity NON-DECREASING as rung index grows.
# MIXED / UNDEFINED carry no severity and break the chain (they are not
# silently skipped -- see cell_is_monotone).
SEVERITY = {"I": 1, "II": 2, "III": 3}

# arm -> (rung index label, comparator arm in A02, ruler subdir)
NATIVE_ARMS = {
    "N6":  ("6",  "A2", "b05_native_ruler_N6_j6"),
    "N9":  ("9",  "A3", "b05_native_ruler_N9_j9"),
    "N12": ("12", "A4", "b05_native_ruler_N12_j12"),
    "N18": ("18", "A5", "b05_native_ruler_N18_j18"),
}
COMPARATOR_DIRS = {
    "A0": "a02_dvr_ruler_j0_top12",
    "A2": "a02_rtax_ruler_A2_j6",
    "A3": "a02_rtax_ruler_A3_j9",
    "A4": "a02_ruler_c2_j12_readlora",
    "A5": "a02_rtax_ruler_A5_j18",
}


# ------------------------------------------------------------- statistics -- #
def mcnemar_exact(anchor, arm):
    """Exact two-sided McNemar on paired 0/1 correctness vectors.

    Returns (delta_pp, n01, n10, p_two_sided).  delta_pp = arm - anchor in
    percentage points, i.e. NEGATIVE means the arm is worse than the anchor.
    Pure integer arithmetic -> identical on every node (see module docstring).
    """
    assert len(anchor) == len(arm), f"unpaired: {len(anchor)} vs {len(arm)}"
    n = len(anchor)
    assert n > 0, "empty paired vector"
    n01 = sum(1 for a, b in zip(anchor, arm) if a == 1 and b == 0)  # arm lost
    n10 = sum(1 for a, b in zip(anchor, arm) if a == 0 and b == 1)  # arm won
    nd = n01 + n10
    if nd == 0:
        p = 1.0
    else:
        k = min(n01, n10)
        p = min(1.0, 2.0 * sum(comb(nd, i) for i in range(k + 1)) / 2 ** nd)
    delta_pp = 100.0 * (n10 - n01) / n
    return delta_pp, n01, n10, p


def classify_vs_anchor(anchor, arm):
    """Prereg 2.1: is `arm` BELOW the anchor, or INDISTINGUISHABLE from it?

    Returns ("BELOW"|"SAME", detail).  BOTH conditions are required for BELOW,
    not either: |delta| >= FLOOR_PP *and* exact p < ALPHA.  A delta below the
    floor is declared indistinguishable regardless of its p-value, and a
    significant p on a sub-floor delta is likewise declared indistinguishable.
    """
    delta_pp, n01, n10, p = mcnemar_exact(anchor, arm)
    below = (delta_pp <= -FLOOR_PP) and (p < ALPHA)
    detail = {
        "delta_pp": round(delta_pp, 2), "n01": n01, "n10": n10,
        "exact_p": p, "n_paired": len(anchor),
        "floor_pp": FLOOR_PP, "alpha": ALPHA,
        "meets_floor": bool(delta_pp <= -FLOOR_PP),
        "meets_alpha": bool(p < ALPHA),
        "stat_decides_label": "exact_mcnemar",
        # at_ceiling: prereg 2.2 -- a positive delta against a 100.0 anchor is
        # unmeasurable, so such a cell may support "indistinguishable" but may
        # never support "better than".
        "at_ceiling_positive_censored": bool(n01 == 0 and n10 == 0),
    }
    return ("BELOW" if below else "SAME"), detail


# ------------------------------------------------------------- prereg 3.1 -- #
def cell_label(anchor, native, lora):
    """Phase label for ONE cell at ONE rung.  Prereg 3.1, unchanged in spirit.

        I   -- native SAME as anchor, LoRA SAME as anchor, native SAME as LoRA
        II  -- LoRA SAME as anchor, native BELOW anchor
        III -- BOTH native and LoRA BELOW anchor
        MIXED_CELL -- anything else (e.g. LoRA below while native is not, which
                      is not a named phase; it is recorded, not renamed)
    """
    nat, nat_d = classify_vs_anchor(anchor, native)
    lor, lor_d = classify_vs_anchor(anchor, lora)
    nvl, nvl_d = classify_vs_anchor(lora, native)   # native vs LoRA directly
    if nat == "SAME" and lor == "SAME" and nvl == "SAME":
        lab = "I"
    elif lor == "SAME" and nat == "BELOW":
        lab = "II"
    elif lor == "BELOW" and nat == "BELOW":
        lab = "III"
    else:
        lab = "MIXED_CELL"
    return lab, {"native_vs_anchor": nat_d, "lora_vs_anchor": lor_d,
                 "native_vs_lora": nvl_d,
                 "native_verdict": nat, "lora_verdict": lor}


def rung_label(cell_labels):
    """★ THE FIX FOR LENS `decidability`.  Prereg 3.1(a), verbatim:

    "rung j is assigned phase P iff >= 3 of its 4 cells carry label P;
     otherwise the rung is MIXED and counts toward neither phase."

    `cell_labels` is a dict cell_key -> label.  Missing cells count as
    disagreement (they cannot vote), so a dropped comparator cannot silently
    lower the bar from 3-of-4 to 3-of-3.
    """
    n_slots = len(PRIMARY_CELLS)
    counts = {}
    for lab in cell_labels.values():
        counts[lab] = counts.get(lab, 0) + 1
    for phase in ("I", "II", "III"):
        if counts.get(phase, 0) >= 3:
            return phase, {"counts": counts, "n_slots": n_slots,
                           "n_reported": len(cell_labels), "rule": ">=3 of 4"}
    return "MIXED", {"counts": counts, "n_slots": n_slots,
                     "n_reported": len(cell_labels), "rule": ">=3 of 4"}


# ------------------------------------------------------------- prereg 3.3 -- #
def cell_is_monotone(seq):
    """★ THE FIX FOR LENS `falsifiability`.

    `seq` is the list of this cell's labels along RUNG_INDEX order, i.e.
    indices 0..4 == [BELOW, 6, 9, 12, 18].  Monotone == phase SEVERITY is
    non-decreasing in the index.  Index 0 is Phase I by construction.

    A label with no severity (MIXED_CELL / UNDEFINED) makes the cell
    NON-MONOTONE rather than being skipped -- skipping it would let a hole in
    the ladder pass as a clean ordering, which is exactly the "non-monotone
    scatter read as separation" failure the lens named.
    """
    sev = []
    for lab in seq:
        if lab not in SEVERITY:
            return False, {"reason": f"unrankable label {lab!r} in ladder",
                           "labels": list(seq)}
        sev.append(SEVERITY[lab])
    ok = all(sev[i] <= sev[i + 1] for i in range(len(sev) - 1))
    return ok, {"reason": "severity non-decreasing" if ok
                else "severity DECREASES somewhere -> scatter, not a ladder",
                "labels": list(seq), "severity": sev}


def cell_boundary_index(seq):
    """First index at which this cell LEAVES Phase I, in RUNG_INDEX units.

    Defined ONLY for a monotone cell (guarded by the caller).  Returns None if
    the cell never leaves Phase I -- which is itself informative: it means the
    I/II boundary is deeper than j=18 and hence OUTSIDE the measured ladder,
    so no boundary was located.
    """
    for i, lab in enumerate(seq):
        if lab in ("II", "III"):
            return i
    return None


# ------------------------------------------------------------- prereg 2.2 -- #
def recovery_fraction(anchor, native, lora):
    """R_j = (LoRA - native) / (anchor - native), with the denominator guard.

    DENOMINATOR GUARD (prereg 2.2): R_j is computed ONLY IF the denominator
    (anchor - native) is >= R_DENOM_MIN_PP AND that denominator is itself
    significant by the exact test.  Otherwise R_j is ill-defined and MUST be
    emitted as null with a reason -- never as a large or negative ratio.  This
    bites hardest at j=6, where native may sit at the ceiling and the
    denominator -> 0.
    """
    d_pp, _, _, d_p = mcnemar_exact(native, anchor)   # anchor - native
    if d_pp < R_DENOM_MIN_PP or d_p >= ALPHA:
        return None, {"recovery_fraction": None,
                      "reason": (f"denominator {d_pp:.2f}pp < {R_DENOM_MIN_PP}pp "
                                 f"or exact p={d_p:.4g} >= {ALPHA} "
                                 f"-> Delta ill-defined"),
                      "denominator_pp": round(d_pp, 2), "denominator_p": d_p}
    n_pp, _, _, _ = mcnemar_exact(native, lora)       # lora - native
    return n_pp / d_pp, {"recovery_fraction": round(n_pp / d_pp, 4),
                         "numerator_pp": round(n_pp, 2),
                         "denominator_pp": round(d_pp, 2),
                         "denominator_p": d_p}


# --------------------------------------------------------------- the gate -- #
def adjudicate(per_cell_vectors):
    """Run the whole prereg-3.2 kill gate.  Returns a decision dict.

    `per_cell_vectors[(task, length)][rung_key][arm]` -> list of 0/1, where
    rung_key in {"BELOW","6","9","12","18"} and arm in {"anchor","native","lora"}.

    THREE INDEPENDENT KILL CLAUSES.  The gate FIRES if ANY of them trips:
      K1 COUNT     : fewer than 3 of {I, II, III} non-empty over j in LADDER_J
      K2 CROSSTASK : the two tasks' I/II boundary indices differ by >= 2
      K3 SCATTER   : fewer than 3 of the 4 primary cells are monotone
    K3 is new; it is lens `falsifiability`'s guard promoted to a clause, because
    a guard that only invalidates K2 would let a total scatter pass K1 by
    accident.
    """
    out = {"prereg": "PHASE_SEPARATION_PREREG.md sec 3.1/3.2/3.3",
           "floor_pp": FLOOR_PP, "alpha": ALPHA, "test": "exact_mcnemar",
           "rung_index": [k for k, _ in RUNG_INDEX],
           "cells": {}, "rungs": {}, "monotonicity": {}, "boundaries": {},
           "recovery_fraction": {}}

    # ---- per-cell labels along the ladder ----
    cell_seq = {}
    for task, length in PRIMARY_CELLS:
        ck = f"ruler|{task}|{length}"
        seq = []
        out["cells"][ck] = {}
        for rk, _j in RUNG_INDEX:
            v = per_cell_vectors.get((task, length), {}).get(rk)
            if not v:
                seq.append("UNDEFINED")
                out["cells"][ck][rk] = {"label": "UNDEFINED",
                                        "reason": "no paired vectors on disk"}
                continue
            lab, det = cell_label(v["anchor"], v["native"], v["lora"])
            seq.append(lab)
            out["cells"][ck][rk] = {"label": lab, **det}
            if rk != "BELOW":
                r, rdet = recovery_fraction(v["anchor"], v["native"], v["lora"])
                out["recovery_fraction"].setdefault(ck, {})[rk] = rdet
        cell_seq[ck] = seq

    # ---- K3: monotonicity, per cell ----
    n_monotone = 0
    for ck, seq in cell_seq.items():
        ok, det = cell_is_monotone(seq)
        out["monotonicity"][ck] = {"monotone": bool(ok), **det}
        n_monotone += int(ok)
    out["n_monotone_cells"] = n_monotone
    k3 = n_monotone < 3

    # ---- K1: rung labels by >=3-of-4 majority, then phase non-emptiness ----
    nonempty = {"I": [], "II": [], "III": []}
    for rk, j in RUNG_INDEX:
        labs = {ck: out["cells"][ck][rk]["label"] for ck in cell_seq
                if out["cells"][ck][rk]["label"] != "UNDEFINED"}
        lab, det = rung_label(labs)
        out["rungs"][rk] = {"label": lab, "j": j, **det}
        if lab in nonempty and j in LADDER_J:      # BELOW never counts
            nonempty[lab].append(j)
    out["phase_nonempty"] = nonempty
    n_nonempty = sum(1 for p in nonempty if nonempty[p])
    out["n_phases_nonempty"] = n_nonempty
    k1 = n_nonempty < 3

    # ---- K2: cross-task I/II boundary, monotone cells only ----
    task_b = {}
    for task in TASKS:
        idxs = []
        for length in ("16k", "32k"):
            ck = f"ruler|{task}|{length}"
            if not out["monotonicity"][ck]["monotone"]:
                continue                            # a scatter votes on nothing
            b = cell_boundary_index(cell_seq[ck])
            idxs.append(b)
        out["boundaries"][task] = {"per_cell_index": idxs}
        usable = [i for i in idxs if i is not None]
        if not idxs:
            task_b[task] = None
            out["boundaries"][task]["task_index"] = None
            out["boundaries"][task]["reason"] = "no monotone cell for this task"
        elif not usable:
            task_b[task] = None
            out["boundaries"][task]["task_index"] = None
            out["boundaries"][task]["reason"] = (
                "cell(s) never leave Phase I -> boundary deeper than j=18, "
                "i.e. NOT LOCATED inside the measured ladder")
        else:
            # shallowest usable index; pre-registered so that a task whose two
            # lengths disagree cannot be silently averaged into a half-rung.
            task_b[task] = min(usable)
            out["boundaries"][task]["task_index"] = min(usable)
            out["boundaries"][task]["rule"] = "min over that task's monotone cells"
    ba, bb = task_b.get(TASKS[0]), task_b.get(TASKS[1])
    if ba is None or bb is None:
        k2 = True
        gap = None
        k2_why = ("at least one task has NO located I/II boundary -> the "
                  "cross-task agreement claim is not evaluable, which the "
                  "prereg treats as FAILING, not as passing by default")
    else:
        gap = abs(ba - bb)
        k2 = gap >= 2
        k2_why = f"|{ba} - {bb}| = {gap} rung indices"
    out["crosstask_gap_rung_indices"] = gap

    fired = []
    if k1:
        fired.append(f"K1_COUNT: only {n_nonempty} of 3 phases non-empty "
                     f"over j in {LADDER_J} (need >= 3)")
    if k2:
        fired.append(f"K2_CROSSTASK: {k2_why} (need <= 1)")
    if k3:
        fired.append(f"K3_SCATTER: only {n_monotone} of 4 primary cells are "
                     f"monotone in rung index (need >= 3)")
    out["clauses"] = {"K1_count": bool(k1), "K2_crosstask": bool(k2),
                      "K3_scatter": bool(k3)}
    out["kill_gate_fired"] = bool(k1 or k2 or k3)
    out["kill_gate_reasons"] = fired
    out["verdict"] = "KILL -> exit clause: fold into Paper A/B mechanism " \
                     "subsection" if out["kill_gate_fired"] else \
                     "PASS -> first measured phase boundary; eligible for " \
                     "promotion review"
    return out


# --------------------------------------------------------------- selftest -- #
def _vec(n_correct, n=N_EXPECT):
    """A 0/1 vector with `n_correct` leading 1s -- NESTED, so pairs against any
    other _vec are worst-case one-directional (n10 == 0), which is what the
    comparator actually shows in 20/20 cells."""
    return [1] * n_correct + [0] * (n - n_correct)


def selftest_floor_calibration():
    """Show WHY the floor moved from 4.0 pp (bootstrap) to 6.0 pp (exact)."""
    print("-- floor calibration (exact two-sided McNemar, n=100, n10=0) --")
    mdes = None
    for k in range(0, 12):
        _, _, _, p = mcnemar_exact(_vec(100), _vec(100 - k))
        flag = "SIG" if p < ALPHA else "ns"
        if mdes is None and p < ALPHA:
            mdes = k
        print(f"   |delta| = {k:2d} pp  exact p = {p:.5f}  {flag}")
    assert mdes == 6, f"expected MDE 6 pp, got {mdes}"
    # The A02 cell the OLD floor was calibrated on: niah_multikey_1 32k, A3,
    # delta -4.0 pp, 4 discordant pairs all one direction.
    _, n01, n10, p = mcnemar_exact(_vec(99), _vec(95))
    assert (n01, n10) == (4, 0), (n01, n10)
    assert p == 0.125, p
    print(f"   A02 niah_multikey_1|32k A3: n01=4 n10=0 exact p={p} = ns, "
          f"but A02's percentile bootstrap reported CI95 [-8,-1] = SIG.")
    print(f"   => FLOOR_PP = {FLOOR_PP} (was 4.0, calibrated on that call).")
    print("   OK")


def selftest_gate():
    """Construct results that MUST kill, and one that MUST pass."""
    A = {"niah_multikey_1": {"16k": 100, "32k": 99},
         "variable_tracking": {"16k": 100, "32k": 100}}
    LORA = {"6": {"niah_multikey_1": (99, 99), "variable_tracking": (99, 100)},
            "9": {"niah_multikey_1": (99, 95), "variable_tracking": (99, 100)},
            "12": {"niah_multikey_1": (90, 96), "variable_tracking": (88, 89)},
            "18": {"niah_multikey_1": (32, 42), "variable_tracking": (4, 5)}}

    def build(native):
        """native[rung][task] = (acc16k, acc32k); LoRA/anchor from A02 on-disk."""
        pv = {}
        for task, length in PRIMARY_CELLS:
            li = 0 if length == "16k" else 1
            d = {}
            anc = A[task][length]
            # index 0 (BELOW) == j=0: anchor vs itself, and A1 vs A0 = 0 flips.
            d["BELOW"] = {"anchor": _vec(anc), "native": _vec(anc),
                          "lora": _vec(anc)}
            for rk in ("6", "9", "12", "18"):
                d[rk] = {"anchor": _vec(anc),
                         "native": _vec(native[rk][task][li]),
                         "lora": _vec(LORA[rk][task][li])}
            pv[(task, length)] = d
        return pv

    # ---- KILL route 1: native already dead at j=6 -> Phase I EMPTY ----
    r1 = adjudicate(build({
        "6":  {"niah_multikey_1": (78, 85), "variable_tracking": (61, 58)},
        "9":  {"niah_multikey_1": (44, 59), "variable_tracking": (18, 16)},
        "12": {"niah_multikey_1": (12, 20), "variable_tracking": (3, 4)},
        "18": {"niah_multikey_1": (1, 2),   "variable_tracking": (0, 1)}}))
    assert r1["kill_gate_fired"], r1["kill_gate_reasons"]
    assert r1["clauses"]["K1_count"], r1["clauses"]
    assert r1["n_phases_nonempty"] == 2, r1["n_phases_nonempty"]
    print("-- KILL route 1 (Phase I empty):", r1["kill_gate_reasons"][0])

    # ---- KILL route 2: tasks put the I/II boundary >= 2 rung indices apart ----
    r2 = adjudicate(build({
        "6":  {"niah_multikey_1": (99, 99), "variable_tracking": (62, 59)},
        "9":  {"niah_multikey_1": (99, 96), "variable_tracking": (18, 16)},
        "12": {"niah_multikey_1": (60, 58), "variable_tracking": (3, 4)},
        "18": {"niah_multikey_1": (10, 12), "variable_tracking": (0, 1)}}))
    assert r2["kill_gate_fired"], r2["kill_gate_reasons"]
    assert r2["clauses"]["K2_crosstask"], r2
    assert r2["crosstask_gap_rung_indices"] >= 2, r2["boundaries"]
    print("-- KILL route 2 (cross-task):", [x for x in r2["kill_gate_reasons"]
                                            if x.startswith("K2")][0])

    # ---- KILL route 3: non-monotone scatter (native recovers at j=9) ----
    r3 = adjudicate(build({
        "6":  {"niah_multikey_1": (40, 42), "variable_tracking": (38, 41)},
        "9":  {"niah_multikey_1": (99, 99), "variable_tracking": (99, 100)},
        "12": {"niah_multikey_1": (55, 57), "variable_tracking": (52, 54)},
        "18": {"niah_multikey_1": (1, 2),   "variable_tracking": (0, 1)}}))
    assert r3["kill_gate_fired"], r3["kill_gate_reasons"]
    assert r3["clauses"]["K3_scatter"], r3["monotonicity"]
    print("-- KILL route 3 (scatter):", [x for x in r3["kill_gate_reasons"]
                                         if x.startswith("K3")][0])

    # ---- PASS: a clean 3-phase ladder with cross-task boundary agreement ----
    rp = adjudicate(build({
        "6":  {"niah_multikey_1": (99, 99),  "variable_tracking": (99, 100)},
        "9":  {"niah_multikey_1": (44, 59),  "variable_tracking": (60, 62)},
        "12": {"niah_multikey_1": (12, 20),  "variable_tracking": (3, 4)},
        "18": {"niah_multikey_1": (1, 2),    "variable_tracking": (0, 1)}}))
    assert not rp["kill_gate_fired"], rp["kill_gate_reasons"]
    assert rp["n_phases_nonempty"] == 3, rp["phase_nonempty"]
    assert rp["crosstask_gap_rung_indices"] == 0, rp["boundaries"]
    print("-- PASS case: phases", rp["phase_nonempty"],
          "gap", rp["crosstask_gap_rung_indices"])

    # ---- the CELL->RUNG rule itself: a 2-2 tie must be MIXED, not a phase ----
    lab, det = rung_label({"a": "I", "b": "I", "c": "II", "d": "II"})
    assert lab == "MIXED", (lab, det)
    lab, det = rung_label({"a": "I", "b": "I", "c": "I", "d": "II"})
    assert lab == "I", (lab, det)
    # a dropped cell cannot lower the bar from 3-of-4 to 2-of-3
    lab, det = rung_label({"a": "I", "b": "I", "c": "II"})
    assert lab == "MIXED", (lab, det)
    print("-- CELL->RUNG rule: 2-2 tie -> MIXED; 3-1 -> phase; "
          "3 reported with 2-1 -> MIXED.  OK")

    # ---- denominator guard: at-ceiling native must yield null, not a ratio ----
    r, det = recovery_fraction(_vec(100), _vec(100), _vec(100))
    assert r is None and det["recovery_fraction"] is None, det
    r, det = recovery_fraction(_vec(100), _vec(98), _vec(100))
    assert r is None, det          # 2 pp denominator: below the 10 pp guard
    r, det = recovery_fraction(_vec(100), _vec(40), _vec(98))
    assert r is not None and 0.9 < r < 1.0, det
    print("-- denominator guard: null at ceiling and at 2pp; "
          f"R={det['recovery_fraction']} at a 60pp denominator.  OK")


# ====================================================================== #
# THE DATA PATH.  Added 2026-08-17, still PRE-DATA (verified at write
# time: `ls -d ruler_results/b05_native_ruler_*` -> No such file, 0 cells).
#
# WHY THIS SECTION EXISTS
# -----------------------
# Everything above was written pre-data and its --selftest passed, so this
# file LOOKED finished.  It was not: it had no way to read a byte off disk.
# Measured on the 2026-08-15 revision:
#     grep -n 'open(\|json.dump\|write_text\|mkdir'  -> rc=1, ZERO matches
#     grep -c 'args.out'                             -> 0 (declared, never used)
#     awk 'NR>=524,NR<=566' | grep adjudicate        -> rc=1 (main never called it)
# main() unconditionally printed "FAIL: no B05 result cells exist yet" and
# returned 3, for ANY input.  So the prereg's read-out point -- "exactly one
# invocation of code/b05_phase_assign.py --out <evidence_dir>" -- was
# UNEXECUTABLE: it would have returned 3 and written nothing even with all 16
# cells present.  Discovering that after the GPU run would have forced the
# loader to be written POST-DATA, which by the prereg's own void_condition
# makes the whole analysis descriptive only.  Hence: written now, pre-data.
#
# GATE C2 and GATE D were likewise named as PASS conditions by prereg sec 4
# items 2-3 but had no implementation here (grep counts in the 2026-08-15
# revision: sha256=0, selector=0, iter_bm25=0, lora_adapter=0).  Both are
# implemented below, fail-closed.
# ====================================================================== #

N_SHARD_EXPECT = 8          # prereg sec 1; must match the A02 comparators

# prereg sec 1 / GATE D: the protocol invariants every cell must satisfy.
EXPECT_SELECTOR = "iter_bm25"
EXPECT_TOPK = 12
EXPECT_ITER_HOP_TOPK = 4
EXPECT_CHUNK = 512

# arm -> expected resume_j.  The four native arms plus the five comparators
# form the 9-arm set prereg sec 4 item 2 requires C2 pairing across.
ARM_RESUME_J = {"N6": 6, "N9": 9, "N12": 12, "N18": 18,
                "A0": 0, "A2": 6, "A3": 9, "A4": 12, "A5": 18}
# Comparator arms carry a Read-LoRA; native arms must carry NONE.  That single
# difference IS the experiment, so it is asserted rather than assumed.
ARM_WANTS_ADAPTER = {"N6": False, "N9": False, "N12": False, "N18": False,
                     "A0": False,          # j=0 ceiling anchor, no adapter
                     "A2": True, "A3": True, "A4": True, "A5": True}
ARM_ORDER_9 = ["A0", "N6", "A2", "N9", "A3", "N12", "A4", "N18", "A5"]

# native arm -> its paired comparator, from NATIVE_ARMS above (single source).
NATIVE_TO_COMPARATOR = {k: v[1] for k, v in NATIVE_ARMS.items()}
RUNG_TO_NATIVE = {v[0]: k for k, v in NATIVE_ARMS.items()}   # "6"->"N6" ...


def _reference_rul_cell_items(arm_dir, task, length, nshard=N_SHARD_EXPECT):
    """Reference reader, used ONLY by the fixture selftest when the canonical
    A02 loader is not importable (see GATE E note in load_per_cell_vectors).

    Deliberately a line-for-line transcription of
    analyze_a02_depth_vs_retrieval.rul_cell_items so that any divergence is a
    visible diff rather than a silent metric reimplementation.  It is NEVER
    used on the real read-out path -- that path requires the canonical import.
    """
    import glob as _glob
    files = sorted(_glob.glob(str(Path(arm_dir) /
                   f"{task}_{length}_shard*of{nshard}.records.json")))
    if len(files) != nshard:
        return None, None, f"G1_SHARD_INCOMPLETE {len(files)}/{nshard}"
    out, cfg = {}, None
    for fp in files:
        with open(fp) as fh:
            d = json.load(fh)
        cfg = d
        for r in d.get("records", []):
            out[int(r["sample_index"])] = (int(r["correct"]),
                                           r.get("input_ids_sha256"))
    return out, cfg, None


def check_cfg(arm, cfg, errs, where):
    """GATE D (prereg sec 4 item 3): config identity, read from the level each
    field actually lives at.

    A02 recorded the structural fact this depends on: RULER stores its config
    FLAT at the top of *.records.json and carries NO chat_template (that lives
    in the sibling summary *.json).  So chat_template is NOT checked here; it is
    checked in check_summary_chat_template() against the sibling file, and the
    comparison is `is not False` -- never `is not True`, which would pass on a
    None from the wrong nesting level.
    """
    if not cfg:
        errs.append(f"{where}/{arm}: no records cfg")
        return
    if cfg.get("resume_j") != ARM_RESUME_J[arm]:
        errs.append(f"{where}/{arm}: resume_j={cfg.get('resume_j')!r} "
                    f"!= {ARM_RESUME_J[arm]}")
    if cfg.get("selector") != EXPECT_SELECTOR:
        errs.append(f"{where}/{arm}: selector={cfg.get('selector')!r} "
                    f"!= {EXPECT_SELECTOR!r}")
    if cfg.get("topk") != EXPECT_TOPK:
        errs.append(f"{where}/{arm}: topk={cfg.get('topk')!r} != {EXPECT_TOPK}")
    if cfg.get("iter_hop_topk") != EXPECT_ITER_HOP_TOPK:
        errs.append(f"{where}/{arm}: iter_hop_topk={cfg.get('iter_hop_topk')!r} "
                    f"!= {EXPECT_ITER_HOP_TOPK}")
    if cfg.get("chunk_size") != EXPECT_CHUNK:
        errs.append(f"{where}/{arm}: chunk_size={cfg.get('chunk_size')!r} "
                    f"!= {EXPECT_CHUNK}")
    if cfg.get("baseline") not in (None, "none"):
        errs.append(f"{where}/{arm}: baseline={cfg.get('baseline')!r} != none")
    if cfg.get("no_retrieval") is True:
        errs.append(f"{where}/{arm}: no_retrieval=True (must retrieve)")
    lora = cfg.get("lora_adapter")
    if ARM_WANTS_ADAPTER[arm]:
        if not lora:
            errs.append(f"{where}/{arm}: expected a Read-LoRA, got {lora!r}")
    else:
        # ★ THE SINGLE VARIABLE.  A native arm with an adapter is not a native
        # readout; it is a second copy of the comparator, and every delta
        # computed from it would be ~0 for a purely procedural reason.
        if lora:
            errs.append(f"{where}/{arm}: NATIVE arm carries adapter {lora!r} "
                        f"-- voids the single-variable design")


def check_summary_chat_template(arm, summary, errs, where):
    """GATE D, chat-template half.  `chat_template` and `enable_thinking` live
    in the sibling summary json, not in records.json (A02's recorded finding).
    Absent summary is an ERROR, not a pass: silently skipping it is how a
    chat=True cell would enter a chat=False table."""
    if not summary:
        errs.append(f"{where}/{arm}: no summary json (cannot verify chat_template)")
        return
    if summary.get("chat_template") is not False:
        errs.append(f"{where}/{arm}: chat_template="
                    f"{summary.get('chat_template')!r} must be False")
    if summary.get("enable_thinking") is not False:
        errs.append(f"{where}/{arm}: enable_thinking="
                    f"{summary.get('enable_thinking')!r} must be False")


def _summary_for(arm_dir, task, length, nshard=N_SHARD_EXPECT):
    import glob as _glob
    fs = sorted(_glob.glob(str(Path(arm_dir) /
                f"{task}_{length}_shard0of{nshard}.json")))
    if not fs:
        return {}
    with open(fs[0]) as fh:
        return json.load(fh)


def gate_c2_sha_pairing(items_by_arm, arms, where, errs):
    """GATE C2 (prereg sec 4 item 2): input_ids_sha256 equality across all arms,
    joined by sample_index.

    FAIL-CLOSED IN BOTH DIRECTIONS, and the second one matters:
      * a MISMATCH means two arms saw different prompts -> unpaired, fail.
      * a MISSING sha means the identity was never recorded -> the gate cannot
        be evaluated, which is ALSO a failure.  "No evidence of mismatch" is
        not "evidence of match".  This is what catches an attempt to source the
        comparator half from A02's derived per-item vectors file, which is
        verified to contain no sha256 at all.
    """
    common = None
    for a in arms:
        ks = set(items_by_arm[a])
        common = ks if common is None else (common & ks)
    common = sorted(common or [])
    n_mismatch, n_missing, examples = 0, 0, []
    for i in common:
        shas = {}
        for a in arms:
            s = items_by_arm[a][i][1]
            if not s:
                n_missing += 1
                if len(examples) < 5:
                    examples.append({"cell": where, "sample_index": i,
                                     "arm": a, "problem": "sha256 ABSENT"})
                continue
            shas[a] = s
        if len(set(shas.values())) > 1:
            n_mismatch += 1
            if len(examples) < 5:
                examples.append({"cell": where, "sample_index": i,
                                 "problem": "sha256 MISMATCH", **shas})
    if n_mismatch:
        errs.append(f"{where}: GATE_C2_SHA_MISMATCH on {n_mismatch}/{len(common)} "
                    f"paired items")
    if n_missing:
        errs.append(f"{where}: GATE_C2_SHA_ABSENT {n_missing} (arm,item) pairs "
                    f"carry no input_ids_sha256 -> pairing UNVERIFIABLE")
    return {"n_common": len(common), "n_sha_mismatch": n_mismatch,
            "n_sha_absent": n_missing, "examples": examples,
            "passed": bool(not n_mismatch and not n_missing and common)}


def load_per_cell_vectors(base, nshard=N_SHARD_EXPECT, reader=None,
                          require_canonical=True):
    """★ THE MISSING DATA PATH.  Build adjudicate()'s input from disk.

    Returns (per_cell_vectors, report).  per_cell_vectors is exactly the shape
    adjudicate() documents:
        per_cell_vectors[(task, length)][rung_key][arm] -> list of 0/1
    with rung_key in {"BELOW","6","9","12","18"} and arm in
    {"anchor","native","lora"}.

    PAIRING IS PER CELL AND PER RUNG, on the intersection of sample_index
    present in all three of (anchor A0, native Nx, comparator Ax).  Vectors are
    emitted in the sorted-common-index order, so position k is the same item in
    all three -- which is what makes the paired McNemar legitimate.

    THE "BELOW" RUNG is A0 in all three roles.  That is Phase I by construction
    and the prereg says so at RUNG_INDEX (sec 3.0): it is EXCLUDED from every
    phase-non-emptiness test and exists only to anchor the monotonicity check,
    which is undefined without a known-Phase-I left endpoint.  It is not
    evidence and must never be reported as a result.

    GATE E (prereg sec 4 item 4): the per-item scorer is IMPORTED from
    analyze_a02_depth_vs_retrieval, never reimplemented.  `require_canonical`
    exists ONLY for the fixture selftest, which may run on a node where that
    module's dependencies (pandas, datasets) are absent; the real read-out path
    calls this with require_canonical=True and refuses to proceed otherwise.
    """
    report = {"errors": [], "refused": [], "gate_c2": {}, "cells_loaded": [],
              "gate_e": None, "base": str(base)}
    if reader is None:
        code = Path(base) / "proposal/backlog/A02-comem-write-read-repair/code"
        for p in (str(base), str(code), str(Path(base) / "scripts")):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            import analyze_a02_depth_vs_retrieval as dvr
            reader = dvr.rul_cell_items
            report["gate_e"] = f"PASS canonical loader imported from {code}"
        except Exception as e:                      # noqa: BLE001
            if require_canonical:
                report["errors"].append(
                    f"GATE_E_FAIL cannot import the canonical A02 loaders from "
                    f"{code}: {type(e).__name__}: {e}")
                return None, report
            reader = _reference_rul_cell_items
            report["gate_e"] = (f"SUBSTITUTE reference reader "
                                f"({type(e).__name__}: {e}) -- fixture only")

    base = Path(base)
    pcv = {}
    for task, length in PRIMARY_CELLS:
        ck = f"ruler|{task}|{length}"
        # --- load every arm this cell needs, refusing incomplete ones --------
        items, cfgs, sums, bad = {}, {}, {}, []
        for arm in ARM_ORDER_9:
            sub = (NATIVE_ARMS[arm][2] if arm in NATIVE_ARMS
                   else COMPARATOR_DIRS[arm])
            d = base / "ruler_results" / sub
            it, cfg, err = reader(d, task, length, nshard)
            if err or it is None:
                bad.append(f"{arm}({sub}): {err or 'no items'}")
                continue
            if len(it) != N_EXPECT:
                bad.append(f"{arm}({sub}): n={len(it)} != {N_EXPECT}")
                continue
            nonbin = [k for k, v in it.items() if int(v[0]) not in (0, 1)]
            if nonbin:
                bad.append(f"{arm}({sub}): {len(nonbin)} non-binary correct")
                continue
            items[arm], cfgs[arm] = it, cfg
            sums[arm] = _summary_for(d, task, length, nshard)
        if bad:
            report["refused"].append({"cell": ck, "reasons": bad})
            report["errors"].append(f"{ck}: GATE_C_REFUSED -- " + "; ".join(bad))
            continue

        # --- GATE D on every arm of this cell -------------------------------
        for arm in ARM_ORDER_9:
            check_cfg(arm, cfgs[arm], report["errors"], ck)
            check_summary_chat_template(arm, sums[arm], report["errors"], ck)

        # --- GATE C2 across all 9 arms of this cell -------------------------
        report["gate_c2"][ck] = gate_c2_sha_pairing(
            items, ARM_ORDER_9, ck, report["errors"])

        # --- build the ladder ----------------------------------------------
        d_cell = {}
        # BELOW: A0 in all three roles (Phase I by construction; see docstring)
        a0 = items["A0"]
        idx0 = sorted(a0)
        v0 = [int(a0[i][0]) for i in idx0]
        d_cell["BELOW"] = {"anchor": v0, "native": list(v0), "lora": list(v0)}
        for rk in ("6", "9", "12", "18"):
            nat_arm = RUNG_TO_NATIVE[rk]
            lor_arm = NATIVE_TO_COMPARATOR[nat_arm]
            common = sorted(set(items["A0"]) & set(items[nat_arm])
                            & set(items[lor_arm]))
            if len(common) != N_EXPECT:
                report["errors"].append(
                    f"{ck}/{rk}: paired intersection {len(common)} != "
                    f"{N_EXPECT} across (A0, {nat_arm}, {lor_arm})")
            d_cell[rk] = {
                "anchor": [int(items["A0"][i][0]) for i in common],
                "native": [int(items[nat_arm][i][0]) for i in common],
                "lora": [int(items[lor_arm][i][0]) for i in common],
            }
        pcv[(task, length)] = d_cell
        report["cells_loaded"].append(ck)
    return pcv, report


def write_evidence(out_dir, decision, report, meta):
    """★ THE MISSING WRITER.  prereg sec 4: the read-out is `--out <dir>`.

    Emits the machine-readable evidence JSON plus a short verdict .md RENDERED
    FROM THAT JSON (never typed by hand), so the prose cannot drift from the
    numbers.  Returns the two paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev = out / "b05_phase_diagram.json"
    payload = {"meta": meta, "load_report": report, "decision": decision}
    with open(ev, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)

    L = ["# B05 PHASE DIAGRAM — VERDICT",
         "",
         "Rendered FROM `b05_phase_diagram.json` by "
         "`b05_phase_assign.py:write_evidence`. Do not hand-edit.",
         "",
         f"- prereg: `{decision.get('prereg')}`",
         f"- test: {decision.get('test')}  floor={decision.get('floor_pp')} pp  "
         f"alpha={decision.get('alpha')}",
         f"- cells loaded: {len(report.get('cells_loaded', []))} / "
         f"{len(PRIMARY_CELLS)}",
         "",
         "## Kill gate",
         "",
         f"- **kill_gate_fired: {decision.get('kill_gate_fired')}**",
         f"- verdict: {decision.get('verdict')}",
         f"- clauses: {decision.get('clauses')}",
         ""]
    for r in decision.get("kill_gate_reasons", []):
        L.append(f"  - FIRED: {r}")
    L += ["",
          "## Rung labels (>=3 of 4 cells)",
          "",
          "| rung | j | label | counts |",
          "|---|---|---|---|"]
    for rk, v in decision.get("rungs", {}).items():
        L.append(f"| {rk} | {v.get('j')} | {v.get('label')} | {v.get('counts')} |")
    L += ["",
          f"- phases non-empty over j in {LADDER_J}: "
          f"{decision.get('phase_nonempty')}",
          f"- monotone cells: {decision.get('n_monotone_cells')} / "
          f"{len(PRIMARY_CELLS)}",
          f"- cross-task boundary gap (rung indices): "
          f"{decision.get('crosstask_gap_rung_indices')}",
          "",
          "## Scope limits carried from the prereg (read before quoting this)",
          "",
          "- The `BELOW` rung is A0 vs A0, i.e. **Phase I by construction**. It "
          "anchors the monotonicity check and is excluded from every "
          "phase-non-emptiness test. It is not a result.",
          "- prereg sec 3.4 / STATUS honest_power_disclosure: Phase III cannot "
          "realistically fail, because the comparator arm A5 (j=18) is already "
          "near-dead in 4/4 primary cells. A passing K1 is therefore **not** "
          "three independently earned phases; the discriminating question is "
          "whether Phase I survives at j=6.",
          ""]
    md = out / "B05_PHASE_DIAGRAM_VERDICT.md"
    md.write_text("\n".join(L))
    return ev, md


def selftest_loader_against_real_files(base):
    """★ FIXTURE SELFTEST over REAL on-disk bytes.  0 GPU.  Returns an rc.

    WHY THIS IS SEPARATE FROM --selftest, and why --selftest was not enough:
    `--selftest` passes (rc=0) over four hand-built `_vec()` dicts.  It passed
    every day while ZERO disk bytes could reach adjudicate(), because the loader
    did not exist.  Per
    memory/selftest-over-invented-inputs-proves-nothing-about-the-pipeline, a
    gate needs a test that walks real I/O -- and one that injects data which
    LOOKS complete but is subtly wrong.

    Three legs, each an assertion about the loader, not about the statistics:

      LEG 1  REAL COMPARATOR FILE.  A02's per-item vectors file is on wzc1 and
             is the anchor/LoRA half of all 16 cells.  We assert its structure
             matches what the loader needs AND that it carries NO
             input_ids_sha256 -- so a C2 implementation keyed on sha in THIS
             file would be unimplementable.  That is the point: C2 must read
             the raw records.json, never this derived file.

      LEG 2  SYNTHETIC records.json IN THE REAL HARNESS SCHEMA, written to a
             temp dir and read back THROUGH the loader.  Correctness values are
             taken from the REAL A02 vectors file, so the bytes are real data in
             a real schema; only the directory is synthetic.  This is the leg
             that would have caught "the loader does not exist".

      LEG 3  NEGATIVE CONTROLS.  Each must FAIL:
               3a  a native arm carrying a lora_adapter   -> GATE D fires
               3b  one shard deleted (7/8)                -> GATE C refuses
               3c  a mutated input_ids_sha256 in one arm  -> GATE C2 fires
               3d  sha256 stripped entirely               -> GATE C2 fires
                   (ABSENT is a failure, not a pass -- this is the trap that
                    catches sourcing the pairing from the derived file)
    """
    import glob as _glob
    import shutil
    import tempfile

    base = Path(base)
    rc = 0
    print("== B05 LOADER FIXTURE SELFTEST (real bytes, 0 GPU) ==")

    # ---------------- LEG 1: the real A02 per-item vectors file -------------
    vec_p = (Path("/apdcephfs_wzc1/share_304376610/pighzliu_code/"
                  "Mixture-of-Memory") /
             "proposal/backlog/A02-comem-write-read-repair/evidence/"
             "read_tax_ruler/a02_read_tax_per_item_vectors.json")
    if not vec_p.exists():
        print(f"LEG 1 SKIP: {vec_p} not on this disk")
        return 1
    with open(vec_p) as fh:
        vecs = json.load(fh)["ruler"]
    print(f"LEG 1: read {vec_p.name} ({vec_p.stat().st_size} B)")
    real_correct = {}
    for task, length in PRIMARY_CELLS:
        ck = f"ruler|{task}|{length}"
        assert ck in vecs, f"LEG 1: {ck} missing from the real vectors file"
        e = vecs[ck]
        idx, pac = e["common_idx"], e["per_arm_correct"]
        assert len(idx) == N_EXPECT, f"LEG 1 {ck}: {len(idx)} != {N_EXPECT}"
        for a in ("A0", "A2", "A3", "A4", "A5"):
            assert a in pac and len(pac[a]) == N_EXPECT, f"LEG 1 {ck}: bad {a}"
        real_correct[(task, length)] = (idx, pac)
        print(f"   {ck}: n={len(idx)} "
              + " ".join(f"{a}={sum(pac[a])}" for a in
                         ("A0", "A2", "A3", "A4", "A5")))
    # THE TRAP, asserted rather than assumed:
    has_sha = "sha256" in json.dumps(vecs)
    assert not has_sha, "LEG 1: expected NO sha256 in the derived vectors file"
    print("   ASSERTED: the derived vectors file carries NO input_ids_sha256, "
          "so GATE C2 CANNOT be satisfied from it -- C2 must read records.json.")

    # ---------------- LEG 2: real schema, read back through the loader ------
    tmp = Path(tempfile.mkdtemp(prefix="b05_fixture_"))
    try:
        def _write_arm(root, sub, arm, mutate=None, drop_shard=None,
                       strip_sha=False, lora=None):
            """Write 4 cells x nshard records.json in the harness's own schema
            (fields copied from scripts/eval_ruler_qcmem.py:756-771)."""
            d = root / "ruler_results" / sub
            d.mkdir(parents=True, exist_ok=True)
            for task, length in PRIMARY_CELLS:
                idx, pac = real_correct[(task, length)]
                src_arm = "A0" if arm in ("A0",) else (
                    arm if arm in pac else
                    {"N6": "A2", "N9": "A3", "N12": "A4",
                     "N18": "A5"}.get(arm, "A0"))
                for s in range(N_SHARD_EXPECT):
                    if drop_shard is not None and s == drop_shard:
                        continue
                    recs = []
                    for pos, i in enumerate(idx):
                        if pos % N_SHARD_EXPECT != s:
                            continue
                        sha = None if strip_sha else f"sha_{task}_{length}_{i}"
                        if mutate is not None and mutate == (task, length, i):
                            sha = "sha_MUTATED"
                        recs.append({
                            "sample_index": int(i),
                            "input_ids_sha256": sha,
                            "target": "x", "output": "x",
                            "recall": float(pac[src_arm][pos]),
                            "correct": int(pac[src_arm][pos]),
                            "n_tok": 6657,
                        })
                    tag = f"_shard{s}of{N_SHARD_EXPECT}"
                    with open(d / f"{task}_{length}{tag}.records.json", "w") as f:
                        json.dump({
                            "task": task, "length": length,
                            "sharding": {"num_shards": N_SHARD_EXPECT,
                                         "shard_index": s},
                            "resume_j": ARM_RESUME_J[arm],
                            "selector": EXPECT_SELECTOR,
                            "topk": EXPECT_TOPK,
                            "iter_hop_topk": EXPECT_ITER_HOP_TOPK,
                            "iter_rounds": 4,
                            "chunk_size": EXPECT_CHUNK,
                            "lora_adapter": (
                                lora if lora is not None else
                                ("outputs/fixture_adapter/final"
                                 if ARM_WANTS_ADAPTER[arm] else None)),
                            "baseline": "none", "seed": 42,
                            "pythonhashseed": "0",
                            "records": recs,
                        }, f)
                    with open(d / f"{task}_{length}{tag}.json", "w") as f:
                        json.dump({"status": "completed",
                                   "task": task, "length": length,
                                   "chat_template": False,
                                   "enable_thinking": False,
                                   "score": 0.0}, f)

        def _build(root, **kw):
            for arm in ARM_ORDER_9:
                sub = (NATIVE_ARMS[arm][2] if arm in NATIVE_ARMS
                       else COMPARATOR_DIRS[arm])
                _write_arm(root, sub, arm, **(kw.get(arm) or {}))

        good = tmp / "good"
        _build(good)
        n_files = len(_glob.glob(str(good / "ruler_results" / "*" / "*.records.json")))
        print(f"LEG 2: wrote {n_files} records.json in the harness schema "
              f"(9 arms x 4 cells x {N_SHARD_EXPECT} shards)")
        pcv, rep = load_per_cell_vectors(good, require_canonical=False)
        assert pcv is not None, rep["errors"]
        assert not rep["errors"], f"LEG 2 unexpected errors: {rep['errors'][:5]}"
        assert len(rep["cells_loaded"]) == len(PRIMARY_CELLS), rep["cells_loaded"]
        for ck, g in rep["gate_c2"].items():
            assert g["passed"], (ck, g)
            assert g["n_common"] == N_EXPECT, (ck, g)
        # the loaded vectors must equal the REAL per-arm counts
        for task, length in PRIMARY_CELLS:
            idx, pac = real_correct[(task, length)]
            d = pcv[(task, length)]
            assert sum(d["BELOW"]["anchor"]) == sum(pac["A0"]), (task, length)
            for rk, comp in (("6", "A2"), ("9", "A3"),
                             ("12", "A4"), ("18", "A5")):
                assert len(d[rk]["native"]) == N_EXPECT, (rk, len(d[rk]["native"]))
                assert sum(d[rk]["lora"]) == sum(pac[comp]), (task, length, rk)
                assert sum(d[rk]["native"]) == sum(pac[comp]), (task, length, rk)
        print("   loader returned 4/4 cells, 5 rungs each, GATE C2 pass, "
              "and per-arm sums equal the real A02 counts.  OK")

        # adjudicate() must accept the loader's output shape.  This is the join
        # that was previously untested: adjudicate had only ever been fed
        # hand-built dicts.
        dec = adjudicate(pcv)
        assert "kill_gate_fired" in dec and "rungs" in dec, dec.keys()
        assert len(dec["rungs"]) == len(RUNG_INDEX), dec["rungs"].keys()
        print(f"   adjudicate() accepted the loader output: "
              f"kill_gate_fired={dec['kill_gate_fired']}, "
              f"phases={dec['phase_nonempty']}")

        # and the writer must actually produce files
        outd = tmp / "evidence_out"
        ev, md = write_evidence(outd, dec, rep, {"fixture": True})
        assert ev.exists() and ev.stat().st_size > 0, ev
        assert md.exists() and md.stat().st_size > 0, md
        with open(ev) as fh:
            back = json.load(fh)
        assert back["decision"]["kill_gate_fired"] == dec["kill_gate_fired"]
        print(f"   writer emitted {ev.name} ({ev.stat().st_size} B) + "
              f"{md.name} ({md.stat().st_size} B), and the JSON round-trips.  OK")

        # ---------------- LEG 3: negative controls -------------------------
        def _expect_fail(name, kw, needle):
            root = tmp / ("bad_" + name)
            _build(root, **kw)
            p, r = load_per_cell_vectors(root, require_canonical=False)
            hit = [e for e in r["errors"] if needle in e]
            if not hit:
                print(f"   3{name} DID NOT FAIL -- guard is blind. "
                      f"errors={r['errors'][:3]}")
                return False
            print(f"   3{name} correctly REFUSED: {hit[0][:110]}")
            return True

        ok = True
        ok &= _expect_fail("a", {"N9": {"lora": "outputs/oops/final"}},
                           "NATIVE arm carries adapter")
        ok &= _expect_fail("b", {"A3": {"drop_shard": 5}},
                           "G1_SHARD_INCOMPLETE")
        mut = ("niah_multikey_1", "16k", real_correct[
            ("niah_multikey_1", "16k")][0][0])
        ok &= _expect_fail("c", {"N12": {"mutate": mut}},
                           "GATE_C2_SHA_MISMATCH")
        ok &= _expect_fail("d", {"N6": {"strip_sha": True}},
                           "GATE_C2_SHA_ABSENT")
        if not ok:
            print("LEG 3 FAILED: at least one guard cannot fire.")
            rc = 1
        else:
            print("LEG 3: all four negative controls fired.  OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nFIXTURE SELFTEST " + ("PASS" if rc == 0 else "FAIL") +
          " -- the loader was exercised against real correctness data in the "
          "real harness schema, adjudicate() accepted its output, the writer "
          "produced files, and each provenance guard was shown to fire.")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="run the pre-data gate selftest (needs no B05 data)")
    ap.add_argument("--fixture-selftest", action="store_true",
                    help="exercise the LOADER against REAL on-disk bytes "
                         "(0 GPU). Distinct from --selftest, which only uses "
                         "hand-built dicts and therefore proves nothing about "
                         "the disk path.")
    ap.add_argument("--out", default=None, help="evidence dir to write into")
    ap.add_argument("--nshard", type=int, default=N_SHARD_EXPECT)
    ap.add_argument("--base", default=os.environ.get(
        "B05_BASE", "/apdcephfs_zwfy6/share_304376610/pighzliu_code/"
                    "Mixture-of-Memory"))
    args = ap.parse_args()

    if args.selftest:
        selftest_floor_calibration()
        print()
        selftest_gate()
        print("\nALL SELFTESTS PASS -- the gate is decidable and each of the "
              "three kill clauses has a demonstrated triggering result.")
        return 0

    if args.fixture_selftest:
        return selftest_loader_against_real_files(args.base)

    if not args.out:
        print("FAIL: --out <evidence_dir> is required. prereg sec 4 makes the "
              "read-out exactly one invocation that WRITES an evidence dir; an "
              "invocation that prints and writes nothing is not a read-out.",
              file=sys.stderr)
        return 4

    # ---- prereg sec 4: load, gate, adjudicate ONCE, write. ------------------
    # GATE E is enforced inside load_per_cell_vectors (require_canonical=True):
    # the per-item scorer is IMPORTED from analyze_a02_depth_vs_retrieval, and
    # if that import fails we refuse rather than falling back to a local
    # reimplementation. On wzc1 the import fails for a MISSING-DEPENDENCY
    # reason (no pandas / no datasets, measured 2026-08-17), which is exactly
    # why prereg sec 5 pins the analyzer to a zwfy6 H20 node.
    pcv, report = load_per_cell_vectors(args.base, nshard=args.nshard,
                                        require_canonical=True)
    if pcv is None:
        for e in report["errors"]:
            print(f"FAIL: {e}", file=sys.stderr)
        print("This analyzer must run on a zwfy6 node (.73/.82/.104) where the "
              "comparator dirs and the canonical loaders live. See prereg "
              "sec 5.", file=sys.stderr)
        return 2

    if not report["cells_loaded"]:
        print("FAIL: zero B05 primary cells could be loaded. Run the driver "
              "(code/run_b05_native_eval.sh) first; by prereg sec 4 this "
              "analyzer is the READ-OUT and executes exactly once, after all "
              "16 cells pass the completeness gates.", file=sys.stderr)
        for e in report["errors"][:20]:
            print(f"  {e}", file=sys.stderr)
        return 3

    # A partial grid must not be adjudicated: the >=3-of-4 CELL->RUNG rule and
    # the K1 non-emptiness count are both defined over the FULL 4-cell slate,
    # and a missing cell silently lowers what "3 of 4" means.
    if len(report["cells_loaded"]) != len(PRIMARY_CELLS):
        print(f"FAIL: only {len(report['cells_loaded'])}/{len(PRIMARY_CELLS)} "
              f"primary cells loaded; the gate is defined over the full slate.",
              file=sys.stderr)
        for e in report["errors"][:20]:
            print(f"  {e}", file=sys.stderr)
        return 3

    gate_errors = list(report["errors"])
    if gate_errors:
        # Fail-closed: prereg sec 4 requires conditions 1-5 to PASS *before*
        # the phase assignment is produced. Emitting a verdict alongside failed
        # provenance gates is how an unpaired or chat=True cell would end up in
        # a headline.
        print(f"FAIL: {len(gate_errors)} completeness/provenance gate error(s); "
              f"refusing to adjudicate.", file=sys.stderr)
        for e in gate_errors[:40]:
            print(f"  {e}", file=sys.stderr)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "b05_gate_failure.json", "w") as fh:
            json.dump({"gate_errors": gate_errors, "load_report": report},
                      fh, indent=2)
        print(f"wrote {out / 'b05_gate_failure.json'}", file=sys.stderr)
        return 6

    decision = adjudicate(pcv)          # ★ called EXACTLY ONCE (prereg sec 4)
    meta = {
        "produced_by": "code/b05_phase_assign.py --out",
        "prereg": "PHASE_SEPARATION_PREREG.md sec 4 (one-shot read-out)",
        "base": str(args.base),
        "nshard": args.nshard,
        "n_expect_per_cell": N_EXPECT,
        "arms_paired_for_gate_c2": ARM_ORDER_9,
        "native_arms": {k: v[2] for k, v in NATIVE_ARMS.items()},
        "comparator_dirs": COMPARATOR_DIRS,
        "adjudicate_invocations": 1,
    }
    ev, md = write_evidence(args.out, decision, report, meta)
    print(f"cells loaded : {len(report['cells_loaded'])}/{len(PRIMARY_CELLS)}")
    print(f"GATE E       : {report['gate_e']}")
    for ck, g in report["gate_c2"].items():
        print(f"GATE C2 {ck}: n_common={g['n_common']} "
              f"mismatch={g['n_sha_mismatch']} absent={g['n_sha_absent']} "
              f"passed={g['passed']}")
    print(f"kill_gate_fired = {decision['kill_gate_fired']}")
    for r in decision["kill_gate_reasons"]:
        print(f"  FIRED: {r}")
    print(f"verdict: {decision['verdict']}")
    print(f"wrote {ev}")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
