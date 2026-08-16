#!/usr/bin/env python3
"""GATE: the designated-damaged arm set the manuscript declares must equal the arm set
the evidence rollups actually count, and every quoted denominator's cardinality must
equal arms x benchmarks.

Motivation (paperC round_04, R1-01): a hardcoded `DAMAGED_OLMO = {"keep8","keep10",
"keep12"}` in code/mmlu_pro_power_nulls.py silently defined BOTH headline denominators
(MMLU-Pro 14/15 and off-MMLU 0/15) while 04_experiments.tex:8 designates five OLMo-2
arms. The same list also made depth-14 damaged for three families and not for OLMo-2.
No existing check compared the declared set with the counted set, so a reviewer found
it instead. This gate closes the CLASS, not the instance: it re-derives the declared
set from the .tex, re-derives the counted set from every rollup on disk, and fails on
any disagreement in either direction.

This is a WRITER only if --json_out is passed. With no flags it is a pure probe.

0 GPU. Exit 0 = pass, non-zero = fail.

Negative control:
    python3 code/gate_designated_denominator.py --selftest_negative_control
runs the gate against a mutated in-memory copy with one arm deleted from the rollup
and asserts the gate FAILS. "Added an assertion" and "the assertion can stop it" are
different claims; this proves the second.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECTIONS = os.path.join(HERE, "sections")
EV = os.path.join(HERE, "evidence")

# The five small evidence benchmarks; winogrande is the declared negative control and
# is excluded from every denominator by 04_experiments.tex:5.
SMALL5 = ["arc_challenge", "arc_easy", "openbookqa", "commonsense_qa", "piqa"]
NEG_CONTROL = "winogrande"

# Maps the on-disk arm directory names to the short rung names used in the prose.
OLMO_DIR_TO_RUNG = {
    "7B_base": "base",
    "7B_shortgpt16_step200000": "shortgpt16",
    "7B_keep14_step200000": "keep14",
    "7B_keep12_step124000": "keep12",
    "7B_keep10_step83500": "keep10",
    "7B_keep8_step121000": "keep8",
}
# Retained-layer count per rung, needed for the cross-family parity assertion.
RETAINED = {"shortgpt16": 16, "keep14": 14, "keep12": 12, "keep10": 10, "keep8": 8,
            "k14": 14, "k12": 12, "k10": 10, "k8": 8}

MMLUPRO = os.path.join(EV, "mmlu_scale_power", "mmlu_pro_power_nulls_v2.json")
OFFMMLU_OLMO = os.path.join(EV, "second_mc_benchmark", "gate2_letter_content_nulls.json")
OFFMMLU_XF = os.path.join(EV, "second_mc_benchmark_crossfamily",
                          "gate2_crossfamily_nulls.json")


def load(p):
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------------ declared set
def declared_olmo2_damaged_rungs() -> set:
    """Parse the OLMo-2 arm list out of the Models-and-structural-damage paragraph.

    The manuscript sentence is of the form
        'OLMo-2 arms are prune-then-heal checkpoints, including \\texttt{keep8},
         \\texttt{keep10}, \\texttt{keep12}, \\texttt{keep14}, and \\texttt{shortgpt16}.'
    Every \\texttt{...} token in that sentence is a designated damaged rung. We do NOT
    hardcode the expected list here -- that would reproduce the very defect. We read
    whatever the .tex currently declares.
    """
    path = os.path.join(SECTIONS, "04_experiments.tex")
    with open(path) as f:
        tex = f.read()
    m = re.search(r"OLMo-2 arms are prune-then-heal checkpoints[^.]*\.", tex)
    if not m:
        raise SystemExit("GATE ERROR: could not locate the OLMo-2 arm declaration "
                         "sentence in sections/04_experiments.tex. If that sentence "
                         "was reworded, update this gate deliberately rather than "
                         "letting the denominator go unchecked.")
    rungs = set(re.findall(r"\\texttt\{([A-Za-z0-9_\\]+)\}", m.group(0)))
    rungs = {r.replace("\\_", "_") for r in rungs}
    return rungs


# ------------------------------------------------------------------- counted sets
def counted_mmlupro(doc) -> dict:
    """Return {family: set(rungs counted as damaged)} from the rollup blocks."""
    out = {}
    for k, v in doc.get("rollup", {}).items():
        if not isinstance(v, dict) or "damaged_rungs" not in v:
            continue
        fam = k.replace("_naive_chance", "")
        out.setdefault(fam, set()).update(v["damaged_rungs"])
    return out


def counted_offmmlu_olmo2(doc) -> set:
    """The arms present in the OLMo-2 off-MMLU rollup, minus the intact base."""
    arms = set()
    for bench, bv in doc.get("rollup_letter_floor", {}).items():
        if bench not in SMALL5:
            continue
        for d in bv.get("per_arm_verdict", {}):
            arms.add(OLMO_DIR_TO_RUNG.get(d, d))
    arms.discard("base")
    return arms


def counted_offmmlu_nonolmo(doc) -> dict:
    out = {}
    for fam, fv in doc.get("rollup_letter_floor_by_family", {}).items():
        s = set()
        for bench, bv in fv.items():
            if bench not in SMALL5:
                continue
            s.update(bv.get("per_rung", {}))
        s.discard("base")
        out[fam] = s
    return out


# ------------------------------------------------------------------------- checks
def run_gate(mmlupro_doc, off_olmo_doc, off_xf_doc, verbose=True):
    fails, notes = [], []

    def say(s):
        if verbose:
            print(s)

    declared = declared_olmo2_damaged_rungs()
    say(f"[declared]  sections/04_experiments.tex OLMo-2 damaged rungs: "
        f"{sorted(declared)}  (n={len(declared)})")

    # ---- CHECK 1: MMLU-Pro rollup counts every declared OLMo-2 rung -------------
    cm = counted_mmlupro(mmlupro_doc)
    counted_olmo = cm.get("olmo2_7b", set())
    say(f"[counted]   MMLU-Pro rollup.olmo2_7b.damaged_rungs: {sorted(counted_olmo)}"
        f"  (n={len(counted_olmo)})")
    missing = declared - counted_olmo
    extra = counted_olmo - declared
    if missing:
        fails.append(f"CHECK 1 FAIL: MMLU-Pro rollup omits declared damaged OLMo-2 "
                     f"rungs {sorted(missing)}. A ratio quoted over this rollup is "
                     f"computed over an undisclosed subset, which "
                     f"sections/09a_relocated.tex:24 forbids.")
    if extra:
        fails.append(f"CHECK 1 FAIL: MMLU-Pro rollup counts {sorted(extra)} as damaged "
                     f"but sections/04_experiments.tex does not declare them.")
    if not missing and not extra:
        notes.append("CHECK 1 pass: MMLU-Pro OLMo-2 declared set == counted set.")

    # ---- CHECK 2: off-MMLU OLMo-2 rollup counts every declared rung -------------
    counted_off = counted_offmmlu_olmo2(off_olmo_doc)
    say(f"[counted]   off-MMLU OLMo-2 arms in rollup: {sorted(counted_off)}"
        f"  (n={len(counted_off)})")
    missing2 = declared - counted_off
    if missing2:
        fails.append(f"CHECK 2 FAIL: off-MMLU OLMo-2 rollup omits declared damaged "
                     f"rungs {sorted(missing2)}. This is the denominator behind the "
                     f"0/15 and 10/15 counts.")
    else:
        notes.append("CHECK 2 pass: off-MMLU OLMo-2 rollup contains every declared rung.")

    # ---- CHECK 3: cross-family depth parity ------------------------------------
    # A retained depth may not be 'damaged' in one family and 'not damaged' in
    # another. 04_experiments.tex:23 says the designation is fixed by construction.
    xf = counted_mmlupro(mmlupro_doc)
    depth_status = {}
    for fam, rungs in xf.items():
        for r in rungs:
            d = RETAINED.get(r)
            if d is None:
                continue
            depth_status.setdefault(d, {})[fam] = True
    all_fams = set(xf)
    for d, fams in sorted(depth_status.items()):
        absent = all_fams - set(fams)
        # a family legitimately has no arm at this depth; only flag when the family
        # HAS an arm at that depth on disk but does not count it.
        for fam in absent:
            fam_rungs = _rungs_on_disk(mmlupro_doc, fam)
            has = {r for r in fam_rungs if RETAINED.get(r) == d}
            if has:
                fails.append(
                    f"CHECK 3 FAIL: retained depth {d} is counted as designated "
                    f"damaged for {sorted(set(fams))} but family '{fam}' has arm(s) "
                    f"{sorted(has)} at the same retained depth and does NOT count "
                    f"them. sections/04_experiments.tex:23 states the designation is "
                    f"'fixed by construction, never by a measured score', so the same "
                    f"construction cannot be damaged in one family and intact in "
                    f"another.")
    if not any(f.startswith("CHECK 3") for f in fails):
        notes.append("CHECK 3 pass: retained-depth designation is consistent across families.")

    # ---- CHECK 4: cardinality == arms x benchmarks ------------------------------
    off_xf_counted = counted_offmmlu_nonolmo(off_xf_doc)
    n_off = len(counted_off) * len(SMALL5) + sum(len(v) for v in off_xf_counted.values()) * len(SMALL5)
    say(f"[cardinality] off-MMLU designated cells = "
        f"({len(counted_off)} OLMo-2 arms + "
        f"{sum(len(v) for v in off_xf_counted.values())} non-OLMo arms) x "
        f"{len(SMALL5)} benchmarks = {n_off}")
    for fam, rungs in sorted(off_xf_counted.items()):
        for bench, bv in off_xf_doc["rollup_letter_floor_by_family"][fam].items():
            if bench not in SMALL5:
                continue
            got = set(bv.get("per_rung", {})) - {"base"}
            if got != rungs:
                fails.append(f"CHECK 4 FAIL: {fam}/{bench} counts {sorted(got)} but "
                             f"the family's arm set is {sorted(rungs)}; the "
                             f"denominator is not arms x benchmarks.")
    if not any(f.startswith("CHECK 4") for f in fails):
        notes.append(f"CHECK 4 pass: off-MMLU denominator is rectangular "
                     f"(arms x benchmarks = {n_off}).")

    # ---- CHECK 5: negative control is excluded, and only it ---------------------
    declared_ctrl = _declared_negative_controls()
    if declared_ctrl != {NEG_CONTROL}:
        fails.append(f"CHECK 5 FAIL: the manuscript declares negative control(s) "
                     f"{sorted(declared_ctrl)} but this gate excludes "
                     f"'{NEG_CONTROL}'. Update both together.")
    else:
        notes.append(f"CHECK 5 pass: '{NEG_CONTROL}' is the sole declared negative "
                     f"control and is excluded from every denominator.")

    return fails, notes, dict(declared_olmo2=sorted(declared),
                              counted_mmlupro=({k: sorted(v) for k, v in cm.items()}),
                              counted_offmmlu_olmo2=sorted(counted_off),
                              counted_offmmlu_nonolmo={k: sorted(v) for k, v in off_xf_counted.items()},
                              offmmlu_cardinality=n_off)


def _rungs_on_disk(doc, fam):
    if fam == "olmo2_7b":
        return set(doc["olmo2"]["rungs"]) - {"base"}
    return set(doc["crossfamily"].get(fam, {}).get("rungs", {})) - {"base"}


def _declared_negative_controls():
    with open(os.path.join(SECTIONS, "04_experiments.tex")) as f:
        tex = f.read()
    out = set()
    for m in re.finditer(r"(\w[\w-]*)\s+as a negative control", tex):
        out.add(m.group(1).lower())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_out", default=None,
                    help="optional; the gate is a pure probe without it")
    ap.add_argument("--selftest_negative_control", action="store_true",
                    help="mutate an in-memory copy and assert the gate FAILS")
    a = ap.parse_args()

    mp = load(MMLUPRO)
    oo = load(OFFMMLU_OLMO)
    ox = load(OFFMMLU_XF)

    if a.selftest_negative_control:
        print("=" * 78)
        print("NEGATIVE CONTROL: deleting one arm from each rollup on an IN-MEMORY copy")
        print("(the files on disk are not touched)")
        print("=" * 78)
        rc = 0
        # NC-1: drop keep12 from the MMLU-Pro rollup
        import copy
        m1 = copy.deepcopy(mp)
        for k, v in m1["rollup"].items():
            if isinstance(v, dict) and "damaged_rungs" in v and k.startswith("olmo2"):
                v["damaged_rungs"] = [r for r in v["damaged_rungs"] if r != "keep12"]
        f1, _, _ = run_gate(m1, oo, ox, verbose=False)
        hit = [x for x in f1 if "CHECK 1" in x and "keep12" in x]
        print(f"NC-1 drop keep12 from MMLU-Pro rollup -> "
              f"{'CAUGHT' if hit else 'NOT CAUGHT <-- GATE IS BLIND'}")
        for x in hit:
            print("     ", x[:150])
        rc |= 0 if hit else 1

        # NC-2: drop an arm from the off-MMLU OLMo-2 rollup
        o2 = copy.deepcopy(oo)
        for bench, bv in o2["rollup_letter_floor"].items():
            bv.get("per_arm_verdict", {}).pop("7B_keep10_step83500", None)
        f2, _, _ = run_gate(mp, o2, ox, verbose=False)
        hit2 = [x for x in f2 if "CHECK 2" in x]
        print(f"NC-2 drop keep10 from off-MMLU rollup -> "
              f"{'CAUGHT' if hit2 else 'NOT CAUGHT <-- GATE IS BLIND'}")
        for x in hit2:
            print("     ", x[:150])
        rc |= 0 if hit2 else 1

        # NC-3: make one non-OLMo family drop k14 (breaks rectangularity)
        o3 = copy.deepcopy(ox)
        o3["rollup_letter_floor_by_family"]["llama2_7b"]["arc_easy"]["per_rung"].pop("k14", None)
        f3, _, _ = run_gate(mp, oo, o3, verbose=False)
        hit3 = [x for x in f3 if "CHECK 4" in x]
        print(f"NC-3 drop k14 from one family/bench cell -> "
              f"{'CAUGHT' if hit3 else 'NOT CAUGHT <-- GATE IS BLIND'}")
        for x in hit3:
            print("     ", x[:150])
        rc |= 0 if hit3 else 1

        # NC-4: positive control -- unmutated input must NOT trip these three checks
        f4, _, _ = run_gate(mp, oo, ox, verbose=False)
        spurious = [x for x in f4 if "CHECK 4" in x or "CHECK 5" in x]
        print(f"NC-4 positive control (unmutated) -> CHECK 4/5 spurious failures: "
              f"{len(spurious)} {'OK' if not spurious else '<-- FALSE POSITIVE'}")
        rc |= 1 if spurious else 0

        print()
        print("NEGATIVE CONTROL " + ("PASSED" if rc == 0 else "FAILED"))
        return rc

    fails, notes, payload = run_gate(mp, oo, ox)
    print()
    for n in notes:
        print("  " + n)
    for f in fails:
        print("  " + f)
    print()
    verdict = "PASS" if not fails else "FAIL"
    print(f"GATE designated_denominator: {verdict} "
          f"({len(notes)} checks passed, {len(fails)} failed)")

    if a.json_out:
        payload.update(dict(schema_version="1.0.0", verdict=verdict,
                            n_pass=len(notes), n_fail=len(fails),
                            passed=notes, failures=fails, gpu_used="NONE"))
        with open(a.json_out, "w") as f:
            json.dump(payload, f, indent=1)
        print("wrote", a.json_out)

    return 0 if not fails else 2


if __name__ == "__main__":
    sys.exit(main())
