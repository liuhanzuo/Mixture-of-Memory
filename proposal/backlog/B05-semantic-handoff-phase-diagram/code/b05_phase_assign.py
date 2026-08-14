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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="run the pre-data gate selftest (needs no B05 data)")
    ap.add_argument("--out", default=None, help="evidence dir to write into")
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

    # Loading real cells requires the A02 loaders, which live on zwfy6 next to
    # the comparator result dirs.  GATE E: import them, never reimplement.
    base = Path(args.base)
    code = base / "proposal/backlog/A02-comem-write-read-repair/code"
    for p in (str(base), str(code), str(base / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import analyze_a02_depth_vs_retrieval as dvr  # noqa: F401
    except Exception as e:
        print(f"FAIL: cannot import the canonical A02 loaders from {code}: {e}",
              file=sys.stderr)
        print("This analyzer must run on a zwfy6 node (.73/.82/.104) where the "
              "comparator dirs and loaders live.  See prereg sec 5.",
              file=sys.stderr)
        return 2
    print("FAIL: no B05 result cells exist yet.  Run the driver first; this "
          "analyzer is the READ-OUT and by prereg sec 4 it executes exactly "
          "once, after all 16 cells pass the completeness gates.",
          file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main())
