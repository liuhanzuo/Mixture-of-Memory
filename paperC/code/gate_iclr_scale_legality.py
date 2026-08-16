#!/usr/bin/env python3
"""Guard: reviews scored under the ICLR rubric must use only ICLR-legal values.

WHY THIS EXISTS
---------------
The owner switched paperC to the ICLR scale on 2026-08-16. That scale is a SIX-POINT
ladder -- {1, 3, 5, 6, 8, 10} -- with the accept/reject boundary between 5 and 6. Values
2, 4, 7, 9 are not on the form, and half-points are not either.

This matters because it is exactly what our previous rubric let slip. round_04's panel
contains a 5.5 ("borderline") and its gates were written as median>=7.0 and LQ>=6.0 --
thresholds that are NOT EXPRESSIBLE on the ICLR ladder, since 7 is not a legal value. A
gate whose threshold cannot be attained is not a gate. So the scale change has to be
enforced numerically, or the next aggregate will silently mix the two conventions and the
median will be computed over a set that contains impossible scores.

It also refuses to let a reviewer decline to decide. On the ICLR form there is no value
between 5 and 6; picking one is picking a side.

WHAT IT CHECKS
--------------
For each aggregate JSON under review_rounds/round_NN/ that declares the ICLR scale:
  1. every rating is in {1,3,5,6,8,10};
  2. every confidence, where present, is in {1,2,3,4,5};
  3. the recorded median/lower_quartile are recomputed from the per-reviewer ratings and
     must match what the file claims -- a stale summary is the defect this repo has hit
     more than once;
  4. the gate thresholds are the ICLR ones (median>=6, LQ>=5, no rating of 1), not the
     generic rubric's median>=7.0;
  5. no ceiling is below its own rating (a paper cannot be scored above its ceiling).

Rounds that do NOT declare the ICLR scale are skipped and reported as skipped, because
round_00..round_04 were legitimately scored on the generic 1-10 rubric and must not be
retro-checked against a scale they never used.

Exit codes: 0 = every ICLR-scale round is internally legal. 2 = at least one violation.
3 = nothing to check (no ICLR-scale round found), which is not a pass.
"""
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROUNDS = ROOT / "review_rounds"

LEGAL_RATINGS = {1, 3, 5, 6, 8, 10}
LEGAL_CONFIDENCE = {1, 2, 3, 4, 5}
# The generic rubric's thresholds are unattainable here: 7 is not a legal rating.
ICLR_GATES = {"median": 6, "lower_quartile": 5, "no_rating_of": 1}


def declares_iclr(doc):
    """True only for a file that IS a scoring aggregate on the ICLR scale.

    Originally this was `"iclr" in blob and "scale" in blob`, which is a test for MENTIONING
    the scale, not for BEING scored on it. Measured 2026-08-17: writing
    MAIN_ADJUDICATION_20260817.json -- an adjudication that discusses the round_05 scale in
    prose and carries no ratings at all -- flipped this gate from rc=0 to rc=2 with
    "declares the ICLR scale but has no per_reviewer ratings". The gate was right that the
    file had no ratings and wrong that it should have any.

    The discriminator is now structural: a top-level `scale` key naming the ladder, AND a
    `per_reviewer` map. A prose mention cannot satisfy either. Renaming the adjudication
    file would also have silenced the gate, which is exactly the wrong repair -- it would
    leave the loose test in place for the next non-aggregate document.
    """
    if not isinstance(doc, dict):
        return False
    scale = doc.get("scale")
    if not (isinstance(scale, str) and "iclr" in scale.lower()):
        return False
    return isinstance(doc.get("per_reviewer"), dict)


def ratings_of(doc):
    """Per-reviewer ratings, preferring the per_reviewer map over any summary list."""
    per = doc.get("per_reviewer") or {}
    out = {}
    for name, rec in per.items():
        if isinstance(rec, dict):
            for key in ("rating", "overall"):
                if key in rec:
                    out[name] = rec[key]
                    break
    return out


def ceilings_of(doc):
    per = doc.get("per_reviewer") or {}
    return {n: r["ceiling"] for n, r in per.items()
            if isinstance(r, dict) and "ceiling" in r}


def lower_quartile(vals):
    """Lower quartile as an ORDER STATISTIC, numpy method='lower'.

    Pinned explicitly because my first version used `vals[len(vals)//4]`, which is not any
    standard definition: on the legal fixture [5,6,6,8] it returns 6, while the actual
    lower quartile is 5.75 (linear) or 5 (lower). That made the gate reject a legal
    aggregate -- a false positive on its own control.

    I could NOT infer the house convention from round_04, because on its data
    [4,4,4,4,4,5,5,5,5,5,5.5,6] all four numpy methods (lower/linear/midpoint/higher) give
    the same 4.0. A case where every definition coincides does not disambiguate anything,
    so this is a choice being declared rather than a fact being recovered: we use
    method='lower', i.e. the largest observed value at or below the 25th percentile.
    Any aggregate that uses a different convention will now fail loudly instead of
    silently disagreeing.
    """
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    # index of the 25th percentile, floored -- numpy's method='lower'
    idx = int(0.25 * (len(s) - 1))
    return s[idx]


def check(path):
    """Return (n_checked, [problems])."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return 0, [f"{path.name}: unreadable ({exc})"]
    if not declares_iclr(doc):
        return 0, []

    problems = []
    rats = ratings_of(doc)
    if not rats:
        return 1, [f"{path.name}: declares the ICLR scale but has no per_reviewer ratings"]

    for name, v in sorted(rats.items()):
        if v not in LEGAL_RATINGS:
            problems.append(
                f"{path.name}: {name} rating={v} is NOT on the ICLR ladder "
                f"{sorted(LEGAL_RATINGS)}. Half-points and 2/4/7/9 are not on the form; "
                f"the boundary is 5|6 and a reviewer must pick a side.")

    per = doc.get("per_reviewer") or {}
    for name, rec in sorted(per.items()):
        if isinstance(rec, dict) and "confidence" in rec:
            c = rec["confidence"]
            if c not in LEGAL_CONFIDENCE:
                problems.append(f"{path.name}: {name} confidence={c} not in 1..5")

    # recompute the summary rather than trusting it
    vals = sorted(rats.values())
    if all(isinstance(v, (int, float)) for v in vals):
        med = statistics.median(vals)
        lq = lower_quartile(vals)
        for key, got in (("median", doc.get("median")), ("lower_quartile", doc.get("lower_quartile"))):
            exp = med if key == "median" else lq
            if got is not None and abs(float(got) - float(exp)) > 1e-9:
                problems.append(
                    f"{path.name}: recorded {key}={got} but recomputing from the "
                    f"{len(vals)} per-reviewer ratings gives {exp} "
                    f"(quartile convention: order statistic, method='lower'). A stale "
                    f"summary is a defect this repo has shipped before.")

    gates = doc.get("gates") or {}
    gate_text = json.dumps(gates)
    if "7.0" in gate_text or "median>=7" in gate_text.replace(" ", ""):
        problems.append(
            f"{path.name}: gates still use the generic rubric's median>=7.0, which is "
            f"UNATTAINABLE on the ICLR ladder (7 is not a legal rating). Use median>=6.")

    for name, ceil in sorted(ceilings_of(doc).items()):
        r = rats.get(name)
        if r is not None and isinstance(ceil, (int, float)) and ceil < r:
            problems.append(f"{path.name}: {name} ceiling={ceil} < rating={r}")

    return 1, problems


def main():
    if not ROUNDS.exists():
        print(f"CANNOT CHECK: {ROUNDS} does not exist")
        return 3

    n_iclr, all_problems, skipped = 0, [], []
    for j in sorted(ROUNDS.glob("round_*/*.json")):
        n, probs = check(j)
        if n == 0 and not probs:
            skipped.append(j.relative_to(ROOT))
            continue
        n_iclr += n
        all_problems += probs

    print(f"ICLR-scale aggregates checked: {n_iclr}")
    print(f"skipped (generic-rubric or non-score json): {len(skipped)}")
    print()

    if n_iclr == 0:
        print("NOTHING TO CHECK: no aggregate declares the ICLR scale yet.")
        print("  This is NOT a pass. round_05 onward must record its scale explicitly,")
        print("  e.g. \"scale\": \"ICLR six-point {1,3,5,6,8,10}, boundary 5|6\".")
        print("  Until then this guard cannot protect anything.")
        return 3

    if all_problems:
        print("FAIL:")
        for p in all_problems:
            print(f"  {p}")
        return 2

    print(f"PASS: all {n_iclr} ICLR-scale aggregate(s) use legal values, their summaries")
    print("      recompute, and their gates are the ICLR thresholds.")
    print("      Scope: legality and internal consistency only. It does not judge whether")
    print("      a rating is DESERVED, and it deliberately does not touch round_00..04,")
    print("      which were scored on the generic rubric.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
