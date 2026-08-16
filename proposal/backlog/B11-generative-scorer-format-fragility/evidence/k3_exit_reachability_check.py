#!/usr/bin/env python3
"""
B11 / K3-EXIT -- executed verification of the two assertions that the upstream issue
draft rests on. 0 GPU. Pure CPU string operations plus (optionally) rescoring stored
prediction CSVs that are already on disk.

Run:
    python3 proposal/backlog/B11-generative-scorer-format-fragility/evidence/k3_exit_reachability_check.py

WHAT THIS SCRIPT SETTLES, AND WHY EACH PART EXISTS
--------------------------------------------------
Claim A (as worded in B11's next_gate): "metrics.py:31 split('Question') is UNREACHABLE".
    The wording conflates three different properties. This script measures them separately,
    because an upstream maintainer will reject the wrong one:
      A1  reachability   -- is line 31 ever EXECUTED?          (sys.settrace, line-level)
      A2  effect         -- can line 31 ever CHANGE the value? (value probe before/after)
      A3  impossibility  -- WHY can it never change it?        (exhaustive Unicode sweep)
    A1 is the one the record gets wrong: the line executes on every single call.

Claim B: "line 27's first-period truncation changes scoring."
    Verified as a REAL TWO-ARM comparison: arm CANON = the shipped scorer, arm NOTRUNC =
    the identical scorer with ONLY line 27 removed. Every other filter (lowercase,
    <context>/<example>, the labels_in_question subtraction, and the uniqueness
    requirement len(labels_in_output)==1) is held fixed in BOTH arms, so a difference
    isolates line 27 and nothing else.

    A demo string is only accepted as evidence for line 27 if the two arms DISAGREE on it.
    Strings on which both arms agree are reported as REJECTED, not quietly dropped: B11 has
    twice put a hand-composed string in the record that died of the UNIQUENESS requirement
    while being described as a truncation effect.
"""

import hashlib
import importlib.util
import inspect
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
PKG = os.path.join(REPO, "third_party", "babilong-pkg")

sys.path.insert(0, PKG)
from babilong.metrics import TASK_LABELS, compare_answers, preprocess_output  # noqa: E402
import babilong.metrics as M  # noqa: E402


def rule(t=""):
    print("\n" + "=" * 78)
    if t:
        print(t)
        print("=" * 78)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


# ---------------------------------------------------------------- provenance
rule("PROVENANCE -- which bytes are under test")
mod_file = M.__file__
print(f"module file : {mod_file}")
print(f"module md5  : {md5(mod_file)}")
print(f"python      : {sys.version.split()[0]}")
src, start = inspect.getsourcelines(preprocess_output)
print(f"preprocess_output occupies lines {start}..{start + len(src) - 1}; printed with real line numbers:")
for i, line in enumerate(src):
    print(f"  {start + i:>3} | {line.rstrip()}")
LINE_LOWER, LINE_TRUNC, LINE_GUARD = start + 1, start + 3, start + 7
print(f"\nasserting the three lines the claims name are where the claims say they are")
assert src[1].strip() == "output = output.lower()", src[1]
assert src[3].strip() == "output = output.split('.')[0]", src[3]
assert src[7].strip() == "output = output.split('Question')[0]", src[7]
print(f"  line {LINE_LOWER} = output.lower()            OK")
print(f"  line {LINE_TRUNC} = output.split('.')[0]      OK")
print(f"  line {LINE_GUARD} = output.split('Question')[0] OK")

# ============================================================ CLAIM A
rule("CLAIM A1 -- IS LINE 31 EXECUTED?  (this is what 'unreachable' would mean)")
print("Method: sys.settrace, line-level, recording every line number executed INSIDE")
print("preprocess_output. If 31 never appears, the line is unreachable/dead code.")
print("If it appears, 'unreachable' is factually wrong however true the rest may be.\n")

CODE = preprocess_output.__code__


def traced_lines(arg):
    """Return the ordered list of line numbers executed inside preprocess_output."""
    seen = []

    def local(frame, event, a):
        if event == "line":
            seen.append(frame.f_lineno)
        return local

    def glob(frame, event, a):
        if event == "call" and frame.f_code is CODE:
            return local
        return None

    old = sys.gettrace()
    sys.settrace(glob)
    try:
        preprocess_output(arg)
    finally:
        sys.settrace(old)
    return seen


TRACE_INPUTS = [
    "kitchen Question: Where is Mary? Answer: garden",  # no period before Question
    "kitchen. Question: Where is Mary?",                # period before Question
    "plain kitchen",                                    # nothing special at all
    "",                                                 # empty string
    "QUESTION",                                         # the literal, upper case
    "Question",                                         # the literal, exactly as in line 31
    "İıΣẞ kitchen Question",        # dotted-I, dotless-i, Sigma, capital sharp s
]
executed_31 = 0
for s in TRACE_INPUTS:
    lines = traced_lines(s)
    hit = LINE_GUARD in lines
    executed_31 += hit
    print(f"  lines={lines}  line{LINE_GUARD}_executed={hit}   input={s!r}")

A1_UNREACHABLE = executed_31 == 0
print(f"\n  inputs where line {LINE_GUARD} executed: {executed_31}/{len(TRACE_INPUTS)}")
print(f"  VERDICT A1: 'line {LINE_GUARD} is UNREACHABLE' -> "
      f"{'SUPPORTED' if A1_UNREACHABLE else 'REFUTED (the line executes every time)'}")

rule("CLAIM A2 -- CAN LINE 31 EVER CHANGE THE VALUE?  ('guaranteed no-op')")
print("Method: re-implement the function step by step, ASSERT the reimplementation equals")
print("the real preprocess_output on every probe (so we are not testing a paraphrase),")
print("then compare the value entering line 31 with the value leaving it.\n")


def stepwise(o):
    o = o.lower()
    o = o.split(".")[0]
    o = o.split("<context>")[0]
    o = o.split("<example>")[0]
    before = o
    o = o.split("Question")[0]
    return before, o


VALUE_INPUTS = TRACE_INPUTS + [
    "Question at the very start",
    "answer kitchen Question Question Question",
    "KÅ Question",  # Kelvin sign, Angstrom sign
]
changed_any = 0
for s in VALUE_INPUTS:
    before, after = stepwise(s)
    assert after == preprocess_output(s), f"reimplementation diverged on {s!r}"
    changed = before != after
    changed_any += changed
    print(f"  changed={changed!s:5}  before={before!r} -> after={after!r}")
A2_NOOP = changed_any == 0
print(f"\n  probes where line {LINE_GUARD} changed the value: {changed_any}/{len(VALUE_INPUTS)}")
print(f"  VERDICT A2: 'line {LINE_GUARD} is a guaranteed no-op' -> "
      f"{'SUPPORTED' if A2_NOOP else 'REFUTED'}")

rule("CLAIM A3 -- WHY it can never fire: str.lower() cannot emit an ASCII capital")
print("Exhaustive over the entire Unicode codepoint range, so this is a proof and not a sample.\n")
q_cp = [cp for cp in range(0x110000) if "Q" in chr(cp).lower()]
any_cp = [cp for cp in range(0x110000) if any(u in chr(cp).lower() for u in "QUESTION")]
multi_up = [(hex(cp), chr(cp).lower()) for cp in range(0x110000)
            if len(chr(cp).lower()) > 1 and any("A" <= ch <= "Z" for ch in chr(cp).lower())]
non_idem = [cp for cp in range(0x110000) if chr(cp).lower().lower() != chr(cp).lower()]
print(f"  codepoints whose .lower() contains ASCII 'Q'               : {q_cp}")
print(f"  codepoints whose .lower() contains ANY char of 'QUESTION'  : {any_cp}")
print(f"  multi-char lowerings still containing an ASCII capital     : {multi_up}")
print(f"  codepoints where .lower() is NOT idempotent                : {non_idem}")
print(f"  Greek final-sigma context rule: 'OΔOΣ'.lower() -> {chr(0x39F)+chr(0x394)+chr(0x39F)+chr(0x3A3)!r}"
      f" -> {(chr(0x39F)+chr(0x394)+chr(0x39F)+chr(0x3A3)).lower()!r}")
A3_PROVED = not q_cp and not any_cp and not multi_up and not non_idem
print(f"\n  VERDICT A3: 'no input can put a capital Q past line {LINE_LOWER}' -> "
      f"{'PROVED' if A3_PROVED else 'REFUTED'}")

rule("CLAIM A -- CONTROL: do the SIBLING guards on lines 29/30 fire?")
print("If they did not either, the defect would be 'the whole tail is dead', not 'line 31'.\n")
for s in ["kitchen <CONTEXT> blah", "kitchen <EXAMPLE> blah", "kitchen <context> blah",
          "kitchen Question: blah"]:
    print(f"  preprocess_output({s!r:34}) = {preprocess_output(s)!r}")
ctl_ok = (preprocess_output("kitchen <CONTEXT> blah") == "kitchen "
          and preprocess_output("kitchen Question: blah") == "kitchen question: blah")
print(f"\n  CONTROL {'PASSED' if ctl_ok else 'FAILED'}: 29/30 fire on their lowercase literals,"
      f" 31 does not fire on its capitalised one.")

rule("CLAIM A -- 'no other reachable path': callers of preprocess_output, repo-wide")
try:
    out = subprocess.run(["git", "grep", "-n", "preprocess_output"], cwd=PKG,
                         capture_output=True, text=True, timeout=60)
    print(out.stdout.strip() or "(no output)")
    print(f"  hits: {len([l for l in out.stdout.strip().splitlines() if l])}")
except Exception as e:  # pragma: no cover
    print("  git grep unavailable:", e)
print("  Note: line 25 (.lower()) is the function's FIRST and UNCONDITIONAL statement,")
print("  so no argument value can reach line 31 with un-lowercased text.")

rule("CLAIM A -- OBSERVABLE CONSEQUENCE: does the dead guard change a SCORE?")
print("Two arms: shipped scorer vs the SAME scorer with the single character fixed.")
print("If scores are identical the defect would be cosmetic; a maintainer will ask.\n")


def compare_fixed_guard(target, output, question, task_labels):
    """compare_answers with line 31 changed to the lowercase literal. Nothing else altered."""
    o = output.lower()
    o = o.split(".")[0]
    o = o.split("<context>")[0]
    o = o.split("<example>")[0]
    o = o.split("question")[0]          # <-- the one-character fix
    t = target.lower()
    labels = {x.lower() for x in task_labels}
    in_q = {x for x in labels if x in question.lower()}
    in_o = {x for x in labels if x in o} - in_q
    if "," in t and len(t) > 3:
        subs = t.split(",")
        return all(x in in_o for x in subs) and len(in_o) == len(subs)
    return t in in_o and len(in_o) == 1


L1 = TASK_LABELS["qa1"]
Q1 = "Where is John?"
assert not any(lbl in Q1.lower() for lbl in L1), "question must contain no task label"
raw_leak = "kitchen Question: where is Mary? Answer: garden"
c_now = compare_answers("kitchen", raw_leak, Q1, L1)
c_fix = compare_fixed_guard("kitchen", raw_leak, Q1, L1)
print(f"  raw output      : {raw_leak!r}")
print(f"  preprocess (now): {preprocess_output(raw_leak)!r}")
print(f"  shipped scorer  : {c_now}")
print(f"  with 1-char fix : {c_fix}")
A_SCORE_CHANGES = (c_now != c_fix)
print(f"\n  VERDICT: fixing the guard is score-neutral? {not A_SCORE_CHANGES}."
      f"  -> the dead guard has a MEASURABLE false-negative consequence: {A_SCORE_CHANGES}")

# ============================================================ CLAIM B
rule("CLAIM B -- TWO-ARM COMPARISON, line 27 removed and NOTHING else changed")


def compare_notrunc(target, output, question, task_labels):
    """compare_answers with ONLY line 27 (first-period truncation) removed."""
    o = output.lower()
    # line 27 deliberately omitted -- this is the single manipulated variable
    o = o.split("<context>")[0]
    o = o.split("<example>")[0]
    o = o.split("Question")[0]
    t = target.lower()
    labels = {x.lower() for x in task_labels}
    in_q = {x for x in labels if x in question.lower()}
    in_o = {x for x in labels if x in o} - in_q
    if "," in t and len(t) > 3:
        subs = t.split(",")
        return all(x in in_o for x in subs) and len(in_o) == len(subs)
    return t in in_o and len(in_o) == 1


def preprocess_notrunc(o):
    o = o.lower()
    o = o.split("<context>")[0]
    o = o.split("<example>")[0]
    o = o.split("Question")[0]
    return o


# Sanity: the NOTRUNC arm must be identical to CANON on an input with no period at all,
# otherwise the arm differs by more than line 27.
for probe in ["kitchen", "the answer is kitchen", "kitchen <CONTEXT> x"]:
    assert preprocess_output(probe) == preprocess_notrunc(probe), probe
print("ARM SANITY: on period-free inputs the two arms are identical -> the arms differ")
print("            by line 27 alone. (3/3 probes)\n")

CASES = [
    ("The answer is A. kitchen", "kitchen",
     "truncation DESTROYS a correct enumerated answer"),
    ("John moved several times. He is in the kitchen", "kitchen",
     "truncation DESTROYS a correct reason-then-answer"),
    ("kitchen. Question: Where is Mary? Answer: garden", "kitchen",
     "truncation RESCUES a correct scaffold-leaking answer"),
    ("kitchen is wrong. the answer is garden", "kitchen",
     "truncation MANUFACTURES a correct answer (false positive)"),
    # The string the record twice offered as evidence. Kept IN, as a negative control.
    ("Choices: A. In the kitchen B. In the garden. The answer is kitchen.", "kitchen",
     "RECORD'S STRING -- claimed 'truncation kills a list-format answer'"),
]
print(f"  {'canon':>6} {'notrunc':>8} {'isolates_l27':>13}  case")
accepted, rejected = [], []
for raw, tgt, desc in CASES:
    c = compare_answers(tgt, raw, Q1, L1)
    n = compare_notrunc(tgt, raw, Q1, L1)
    ok = c != n
    (accepted if ok else rejected).append((raw, desc, c, n))
    print(f"  {c!s:>6} {n!s:>8} {ok!s:>13}  {desc}")
    print(f"         raw     = {raw!r}")
    print(f"         canon   -> {preprocess_output(raw)!r}")
    print(f"         notrunc -> {preprocess_notrunc(raw)!r}")
print(f"\n  ACCEPTED as evidence for line 27 (arms disagree): {len(accepted)}")
print(f"  REJECTED (arms agree -> some OTHER filter did the work): {len(rejected)}")
for raw, desc, c, n in rejected:
    in_o = {l for l in {x.lower() for x in L1} if l in preprocess_notrunc(raw)}
    print(f"    - {raw!r}\n      canon={c} notrunc={n}; labels surviving without truncation="
          f"{sorted(in_o)} -> dies on the UNIQUENESS test len(labels)==1, NOT on line 27.")
B_ISOLATED = len(accepted) >= 2 and any(c and not n for _, _, c, n in accepted) \
    and any((not c) and n for _, _, c, n in accepted)
print(f"\n  VERDICT B (hand cases): line 27 changes scoring IN BOTH DIRECTIONS -> {B_ISOLATED}")

# ------------------------------------------------- Claim B on real data, if present
rule("CLAIM B -- ON REAL STORED GENERATIONS (no GPU: rescoring CSVs already on disk)")
import csv
import glob
import re

roots = [os.path.join(REPO, "babilong_results")]
csvs = []
for r in roots:
    if os.path.isdir(r):
        csvs = sorted(glob.glob(os.path.join(r, "**", "qa*.csv"), recursive=True))
print(f"  prediction CSVs found under babilong_results/: {len(csvs)}")


def is_list_format(s):
    """Verbatim from analyze_a02_truncation_ablation.py (the A02 ablation script)."""
    if s is None:
        return False
    return bool(re.search(r"(^|\n|\s)[A-Da-d][\.\)]\s", s)) or s.count("\n-") >= 2


if csvs:
    LIM = int(os.environ.get("B11_MAX_CSV", "1200"))
    use = csvs[:LIM]
    tot = 0
    d_all = r_all = 0
    strata = {True: [0, 0, 0], False: [0, 0, 0]}   # n, destroyed, rescued
    for path in use:
        task = os.path.basename(path).split("_")[0]
        if task not in TASK_LABELS:
            continue
        labels = TASK_LABELS[task]
        try:
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    tgt = row.get("target")
                    out = row.get("output")
                    qq = row.get("question")
                    if tgt is None or out is None or qq is None:
                        continue
                    c = compare_answers(tgt, out, qq, labels)
                    n = compare_notrunc(tgt, out, qq, labels)
                    tot += 1
                    k = is_list_format(out)
                    strata[k][0] += 1
                    if c and not n:
                        r_all += 1
                        strata[k][2] += 1
                    elif n and not c:
                        d_all += 1
                        strata[k][1] += 1
        except Exception as e:  # pragma: no cover
            print(f"  skip {os.path.basename(path)}: {e}")
    print(f"  CSVs scored: {len(use)} (cap B11_MAX_CSV={LIM})   items: {tot}")
    if tot:
        print(f"\n  {'stratum':>14} {'n':>8} {'destroyed':>10} {'rescued':>9} {'net_pp_of_keeping_l27':>22}")
        for k, name in [(True, "is_list=1"), (False, "is_list=0")]:
            n_, d_, r_ = strata[k]
            net = 100.0 * (r_ - d_) / n_ if n_ else float("nan")
            print(f"  {name:>14} {n_:>8} {d_:>10} {r_:>9} {net:>22.2f}")
        net_all = 100.0 * (r_all - d_all) / tot
        print(f"  {'ALL pooled':>14} {tot:>8} {d_all:>10} {r_all:>9} {net_all:>22.2f}")
        s1 = (strata[True][2] - strata[True][1])
        s0 = (strata[False][2] - strata[False][1])
        print(f"\n  sign(is_list=1) = {'+' if s1 > 0 else '-' if s1 < 0 else '0'} ,"
              f"  sign(is_list=0) = {'+' if s0 > 0 else '-' if s0 < 0 else '0'}"
              f"   -> STRATIFIED SIGNS DIFFER: {(s1 < 0) != (s0 < 0)}")
        print("  CAVEAT: heterogeneous historical runs, many models/configs. Supports the SIGN")
        print("  DEPENDENCE only. Must NEVER be quoted as an effect size for any comparison.")
else:
    print("  none on this disk -> real-data leg NOT RUN here (hand cases above still executed)")

# ============================================================ summary
rule("SUMMARY OF VERDICTS")
print(f"  A1 line {LINE_GUARD} is UNREACHABLE / dead code ....... "
      f"{'SUPPORTED' if A1_UNREACHABLE else 'REFUTED -- it executes on every call'}")
print(f"  A2 line {LINE_GUARD} is a GUARANTEED NO-OP ........... {'SUPPORTED' if A2_NOOP else 'REFUTED'}")
print(f"  A3 impossible by Unicode exhaustion ............ {'PROVED' if A3_PROVED else 'REFUTED'}")
print(f"  A  fixing it changes at least one score ........ {A_SCORE_CHANGES}")
print(f"  B  line 27 changes scoring, both directions .... {B_ISOLATED}")
print("\n  => FILE AS: 'the guard is executed but can never fire (a guaranteed no-op)'.")
print("     DO NOT FILE the word 'unreachable': A1 is REFUTED by measurement.")
