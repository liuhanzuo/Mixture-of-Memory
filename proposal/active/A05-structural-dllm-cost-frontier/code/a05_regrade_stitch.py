#!/usr/bin/env python3
"""A05 closeout -- re-grade the HE+ cells with the CORRECTED stitch.

This is a CORRECTED CELL, not a replacement. The generation is byte-identical to
the K1 run (same raw_output rows on disk, no model is loaded, zero GPU); only the
post-processing that turns raw_output into a gradeable program is fixed.

The bug (verified independently here, see A05_CLOSEOUT_VERDICT.md):
  combine_humaneval_prompt() called extract_python() FIRST. extract_python ends in
  `.strip()`, which removes leading whitespace from the FIRST line only. DreamOn
  emits an already-4-space-indented function body, so after extraction line 1 sits
  at column 0 while lines 2..n keep their original depth. textwrap.indent(...,"    ")
  then shifts everything by 4 -> line 1 at 4, line 2 at 8 -> "unexpected indent".
  NOTE: a dedent applied AFTER extract_python is a NO-OP (the common prefix is
  already 0 because line 1 was stripped). The dedent must happen BEFORE extraction.

Grader is evalplus.eval.untrusted_check with the same mandatory self-test as
a05_k1_merge_and_grade.py (canonical PASSES, stub FAILS) -- invariant 1.
"""
from __future__ import annotations
import ast, glob, json, os, re, statistics, sys, textwrap
from pathlib import Path

R = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft"
EXPECTED = 164

def die(m):
    print(f"FATAL: {m}", file=sys.stderr); raise SystemExit(2)

def extract_python(text):
    fences = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fences: text = max(fences, key=len)
    else:
        u = re.search(r"```(?:python)?\s*\n?(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
        if u: text = u.group(1)
    text = text.strip()
    st = [m.start() for m in re.finditer(r"(?m)^(?:async\s+def|def|from|import|@)\s*", text)]
    if st: text = text[min(st):]
    return text.rstrip() + ("\n" if text else "")

def combine_as_run(prompt, generated):
    e = extract_python(generated)
    if re.search(r"(?m)^(?:async\s+def|def)\s+", e): return e
    body = e.strip() or "pass"
    return prompt.rstrip() + "\n" + textwrap.indent(body, "    ") + "\n"

def combine_fixed(prompt, generated):
    e = extract_python(generated)
    if re.search(r"(?m)^(?:async\s+def|def)\s+", e): return e
    body = textwrap.dedent(generated.replace("\t", "    ")).strip("\n").rstrip()
    if not body.strip(): body = "pass"
    return prompt.rstrip() + "\n" + textwrap.indent(body, "    ") + "\n"

def parseable(t):
    try: ast.parse(t); return True
    except Exception: return False

def main():
    os.environ.setdefault("HOME", "/tmp/a05_k1_grade")
    Path("/tmp/a05_k1_grade").mkdir(parents=True, exist_ok=True)
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.evaluate import get_groundtruth
    from evalplus.eval import untrusted_check

    problems = get_human_eval_plus()
    expected_output = get_groundtruth(problems, get_human_eval_plus_hash(), [])

    def check(task_id, code):
        p = problems[task_id]; ex = expected_output[task_id]; out = {}
        for kind, inputs, ref in (("base", p["base_input"], ex["base"]),
                                  ("plus", p.get("plus_input", []), ex.get("plus", []))):
            if not inputs: out[kind] = "pass"; continue
            status, _ = untrusted_check(
                "humaneval", code, inputs, p["entry_point"], ref, p["atol"],
                ex["base_time"] if kind == "base" else ex["plus_time"],
                fast_check=True, min_time_limit=1.0, gt_time_limit_factor=4.0)
            out[kind] = status
        return out

    # ---- mandatory grader self-test (invariant 1) ----
    probe = "HumanEval/0"
    canonical = problems[probe]["prompt"] + problems[probe]["canonical_solution"]
    stub = problems[probe]["prompt"] + "    pass\n"
    good, bad = check(probe, canonical), check(probe, stub)
    selftest = {"probe_task": probe, "canonical_base": good["base"], "canonical_plus": good["plus"],
                "stub_base": bad["base"], "stub_plus": bad["plus"]}
    if good["base"] != "pass" or good["plus"] != "pass": die(f"self-test: canonical failed {selftest}")
    if bad["base"] == "pass": die(f"self-test: stub passed {selftest}")
    print("grader self-test OK:", selftest, flush=True)

    prompts = {}
    for l in open(f"{R}/data/evalplus/HumanEvalPlus-v0.1.10.jsonl"):
        if l.strip():
            d = json.loads(l); prompts[d["task_id"]] = d["prompt"]

    results = {}
    for cell in ["he_c8", "he_c32", "he_c128"]:
        rows = []
        for f in sorted(glob.glob(f"{R}/runs/a05_k1/{cell}/metrics.rank*.jsonl")):
            for l in open(f):
                if l.strip(): rows.append(json.loads(l))
        ids = [r["task_id"] for r in rows]
        if len(rows) != EXPECTED: die(f"{cell}: {len(rows)} items != {EXPECTED}")
        if len(set(ids)) != EXPECTED: die(f"{cell}: duplicate task_id")
        if set(ids) != set(prompts): die(f"{cell}: task_id set mismatch vs dataset")
        if sum(1 for r in rows if r.get("error")): die(f"{cell}: generation errors present")

        base_ok = plus_ok = 0
        as_run_base = as_run_plus = 0
        par_fixed = par_asrun = 0
        per_item = {}
        flips_gain, flips_loss = [], []
        for r in rows:
            raw = r.get("raw_output") or ""
            p = prompts[r["task_id"]]
            A, B = combine_as_run(p, raw), combine_fixed(p, raw)
            par_asrun += parseable(A); par_fixed += parseable(B)
            ra = check(r["task_id"], A)
            rb = check(r["task_id"], B)
            ab, ap = ra["base"] == "pass", (ra["base"] == "pass" and ra["plus"] == "pass")
            bb, bp = rb["base"] == "pass", (rb["base"] == "pass" and rb["plus"] == "pass")
            as_run_base += ab; as_run_plus += ap
            base_ok += bb; plus_ok += bp
            per_item[r["task_id"]] = {"base": bb, "plus": bp, "as_run_plus": ap}
            if bp and not ap: flips_gain.append(r["task_id"])
            if ap and not bp: flips_loss.append(r["task_id"])
        results[cell] = {
            "n": EXPECTED,
            "parseability_as_run": round(par_asrun / EXPECTED, 4),
            "parseability_corrected": round(par_fixed / EXPECTED, 4),
            "pass_at_1_base_as_run": round(as_run_base / EXPECTED, 4),
            "pass_at_1_plus_as_run": round(as_run_plus / EXPECTED, 4),
            "pass_at_1_base_corrected": round(base_ok / EXPECTED, 4),
            "pass_at_1_plus_corrected": round(plus_ok / EXPECTED, 4),
            "n_pass_plus_as_run": as_run_plus,
            "n_pass_plus_corrected": plus_ok,
            "delta_pp_plus": round(100 * (plus_ok - as_run_plus) / EXPECTED, 2),
            "items_gained": flips_gain,
            "items_lost": flips_loss,
            "grader_self_test": selftest,
            "per_item_pass": per_item,
        }
        r_ = results[cell]
        print(f"{cell}: plus {r_['pass_at_1_plus_as_run']} -> {r_['pass_at_1_plus_corrected']} "
              f"({r_['delta_pp_plus']:+} pp)  parseability {r_['parseability_as_run']} -> "
              f"{r_['parseability_corrected']}  gained={len(flips_gain)} lost={len(flips_loss)}",
              flush=True)

    payload = {
        "what": "A05 closeout -- HE+ cells re-graded with the CORRECTED stitch (defect c).",
        "corrected_not_replacement": ("generation is byte-identical to K1 (same raw_output rows, "
                                      "no model loaded, 0 GPU); only post-processing changed"),
        "bug": ("combine_humaneval_prompt ran extract_python FIRST; its .strip() de-indents line 1 "
                "only, so textwrap.indent then double-indents lines 2..n. A dedent applied AFTER "
                "extract_python is a no-op -- it must be applied BEFORE extraction."),
        "grader": "evalplus.eval.untrusted_check, self-tested (canonical PASS / stub FAIL)",
        "cells": results,
    }
    outp = f"{R}/runs/a05_k1/a05_closeout_stitch_regrade.json"
    with open(outp, "w") as h: json.dump(payload, h, indent=2)
    print("wrote", outp)

if __name__ == "__main__":
    main()
