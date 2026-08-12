#!/usr/bin/env python3
"""A05 closeout follow-up -- BLAST RADIUS of the HE+ stitch defect (defect c).

Tests the falsification conditions F1/F2/F3 registered in
proposal/archive/A05-structural-dllm-cost-frontier/A05_BLAST_RADIUS_PREREGISTRATION.md
BEFORE this script was written.

What this does, per arm:
  1. reconstructs the arm's AS-RUN gradeable program from what the arm actually stored
     (raw_output + the arm's own post-processing, or the stored final `solution`);
  2. reconstructs the CORRECTED program under the dedent-before-extract fix;
  3. grades BOTH with evalplus.eval.untrusted_check, with a mandatory per-invocation
     grader self-test (canonical PASSES / stub FAILS) -- invariant 1, never hand-rolled;
  4. reports pass@1 base/plus as-run vs corrected, item-level flips, and parseability.

0 GPU. No model is loaded. Generation is byte-identical to the original runs -- this is a
CORRECTED grading of stored outputs, never a "replacement" run.

Key point for scope: `combine_humaneval_prompt` (the buggy stitch) only applies
textwrap.indent on the branch where extract_python() finds NO top-level def. Arms whose
raw output already contains `def` return early and are structurally inert. Arms that never
call the stitch at all (Scaffold: `solution = result.text`; AR/Dream: extract_python of
prompt+continuation) cannot be affected. This script proves that empirically rather than
by reading alone.
"""
from __future__ import annotations
import ast, glob, json, os, re, sys, textwrap
from pathlib import Path

BUNDLE = "/tmp/a05br/bundle"
OUT = "/tmp/a05br/a05_blast_radius.json"


def die(m):
    print(f"FATAL: {m}", file=sys.stderr)
    raise SystemExit(2)


# ---- the arms' own post-processing, copied verbatim in behaviour ----
def extract_python(text: str) -> str:
    fences = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fences:
        text = max(fences, key=len)
    else:
        u = re.search(r"```(?:python)?\s*\n?(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
        if u:
            text = u.group(1)
    text = text.strip()
    st = [m.start() for m in re.finditer(r"(?m)^(?:async\s+def|def|from|import|@)\s*", text)]
    if st:
        text = text[min(st):]
    return text.rstrip() + ("\n" if text else "")


def combine_as_run(prompt: str, generated: str) -> str:
    """generate_evalplus_dreamon.py::combine_humaneval_prompt -- the BUGGY stitch."""
    e = extract_python(generated)
    if re.search(r"(?m)^(?:async\s+def|def)\s+", e):
        return e
    body = e.strip() or "pass"
    return prompt.rstrip() + "\n" + textwrap.indent(body, "    ") + "\n"


def combine_fixed(prompt: str, generated: str) -> str:
    """Dedent BEFORE extraction (the fix). Post-extraction dedent is a no-op."""
    e = extract_python(generated)
    if re.search(r"(?m)^(?:async\s+def|def)\s+", e):
        return e
    body = textwrap.dedent(generated.replace("\t", "    ")).strip("\n").rstrip()
    if not body.strip():
        body = "pass"
    return prompt.rstrip() + "\n" + textwrap.indent(body, "    ") + "\n"


def parseable(t: str) -> bool:
    try:
        ast.parse(t)
        return True
    except Exception:
        return False


def load_rows(arm_dir: str):
    rows = []
    for f in sorted(glob.glob(f"{arm_dir}/metrics.rank*.jsonl")) or sorted(
        glob.glob(f"{arm_dir}/metrics.jsonl")
    ):
        for l in open(f, errors="replace"):
            if l.strip():
                try:
                    rows.append(json.loads(l))
                except json.JSONDecodeError:
                    pass
    return rows


def load_solutions(arm_dir: str):
    p = f"{arm_dir}/solutions.jsonl"
    if not os.path.exists(p):
        return None
    out = {}
    for l in open(p, errors="replace"):
        if l.strip():
            try:
                r = json.loads(l)
                out[r["task_id"]] = r.get("solution") or ""
            except json.JSONDecodeError:
                pass
    return out


def main():
    os.environ.setdefault("HOME", "/tmp/a05br_grade")
    Path("/tmp/a05br_grade").mkdir(parents=True, exist_ok=True)
    from evalplus.data import (
        get_human_eval_plus, get_human_eval_plus_hash,
        get_mbpp_plus, get_mbpp_plus_hash,
    )
    from evalplus.evaluate import get_groundtruth
    from evalplus.eval import untrusted_check

    DS = {}
    he = get_human_eval_plus()
    DS["humaneval"] = (he, get_groundtruth(he, get_human_eval_plus_hash(), []), 164)
    mb = get_mbpp_plus()
    DS["mbpp"] = (
        mb,
        get_groundtruth(mb, get_mbpp_plus_hash(), ["mbpp/1", "mbpp/2", "mbpp/3"]),
        378,
    )

    def check(ds, task_id, code):
        problems, expected, _ = DS[ds]
        p = problems[task_id]
        ex = expected[task_id]
        out = {}
        for kind, inputs, ref in (
            ("base", p["base_input"], ex["base"]),
            ("plus", p.get("plus_input", []), ex.get("plus", [])),
        ):
            if not inputs:
                out[kind] = "pass"
                continue
            status, _ = untrusted_check(
                ds, code, inputs, p["entry_point"], ref, p["atol"],
                ex["base_time"] if kind == "base" else ex["plus_time"],
                fast_check=True, min_time_limit=1.0, gt_time_limit_factor=4.0,
            )
            out[kind] = status
        return out

    # ---- mandatory grader self-test per dataset (invariant 1) ----
    selftests = {}
    for ds, probe in (("humaneval", "HumanEval/0"), ("mbpp", "mbpp/4")):
        problems = DS[ds][0]
        if probe not in problems:
            probe = sorted(problems)[0]
        canonical = problems[probe]["prompt"] + problems[probe]["canonical_solution"]
        stub = problems[probe]["prompt"] + "    pass\n"
        good, bad = check(ds, probe, canonical), check(ds, probe, stub)
        st = {"probe": probe, "canonical": good, "stub": bad}
        if good["base"] != "pass" or good["plus"] != "pass":
            die(f"self-test {ds}: canonical FAILED {st}")
        if bad["base"] == "pass":
            die(f"self-test {ds}: stub PASSED {st}")
        selftests[ds] = st
        print(f"grader self-test OK [{ds}]: {st}", flush=True)

    prompts = {ds: {t: DS[ds][0][t]["prompt"] for t in DS[ds][0]} for ds in DS}

    # arm -> (dataset, mode)
    #   mode "dreamon_stitch": raw_output exists AND arm used combine_humaneval_prompt
    #   mode "raw_extract"   : raw_output exists, arm used extract_python only (no stitch)
    #   mode "final_only"    : only the final stored solution exists (no raw_output)
    ARMS = [
        # ---- the arm the defect was found on (positive control) ----
        ("dreamon_heplus_r1", "humaneval", "dreamon_stitch"),
        ("dreamon_heplus_r2", "humaneval", "dreamon_stitch"),
        # ---- same driver, other benchmark: stitch NOT applied (dataset!=humaneval) ----
        ("dreamon_mbppplus_r1", "mbpp", "raw_extract"),
        ("dreamon_mbppplus_r2", "mbpp", "raw_extract"),
        # ---- other published diffusion arms, different driver ----
        ("dream_coder_instruct_heplus_r2", "humaneval", "raw_extract"),
        ("dream_coder_instruct_mbppplus_r2", "mbpp", "raw_extract"),
        ("dream_coder_base_heplus", "humaneval", "raw_extract"),
        ("dream_coder_base_mbppplus", "mbpp", "raw_extract"),
        # ---- AR control ----
        ("ar_qwen25coder7b_base", "humaneval", "final_only"),
        # ---- Scaffold, ALL tiers (F3: is .177/.354 itself understated?) ----
        ("scaffold_tiny_heplus", "humaneval", "final_only"),
        ("scaffold_small_heplus", "humaneval", "final_only"),
        ("scaffold_medium_heplus", "humaneval", "final_only"),
        ("scaffold_large_heplus", "humaneval", "final_only"),
        ("scaffold_tiny_mbppplus", "mbpp", "final_only"),
        ("scaffold_small_mbppplus", "mbpp", "final_only"),
        ("scaffold_medium_mbppplus", "mbpp", "final_only"),
        ("scaffold_large_mbppplus", "mbpp", "final_only"),
    ]

    results = {}
    for arm, ds, mode in ARMS:
        d = f"{BUNDLE}/{arm}"
        if not os.path.isdir(d):
            print(f"SKIP {arm}: not staged")
            continue
        expected_n = DS[ds][2]
        rows = load_rows(d)
        sols = load_solutions(d)

        recs = []  # (task_id, as_run_program, corrected_program, hits_indent_branch)
        if mode in ("dreamon_stitch", "raw_extract"):
            if not rows:
                print(f"SKIP {arm}: no metrics rows")
                continue
            ids = [r["task_id"] for r in rows]
            if len(rows) != expected_n:
                die(f"{arm}: {len(rows)} rows != {expected_n}")
            if len(set(ids)) != expected_n:
                die(f"{arm}: duplicate task_id")
            if any(r.get("error") for r in rows):
                die(f"{arm}: generation errors present")
            if sum(1 for r in rows if r.get("raw_output") is None):
                die(f"{arm}: raw_output missing on some rows")
            for r in rows:
                raw = r.get("raw_output") or ""
                tid = r["task_id"]
                p = prompts[ds][tid]
                if mode == "dreamon_stitch":
                    A, B = combine_as_run(p, raw), combine_fixed(p, raw)
                    hits = not re.search(
                        r"(?m)^(?:async\s+def|def)\s+", extract_python(raw)
                    )
                else:
                    # driver used extract_python only; the buggy stitch is not in this path
                    A = B = extract_python(raw)
                    hits = False
                recs.append((tid, A, B, hits))
        else:  # final_only
            if not sols:
                print(f"SKIP {arm}: no solutions.jsonl")
                continue
            if len(sols) != expected_n:
                die(f"{arm}: {len(sols)} solutions != {expected_n}")
            for tid, s in sols.items():
                # the stored final program IS what was graded; there is no stitch to undo,
                # and no raw_output to re-stitch from -> as-run == corrected by construction.
                recs.append((tid, s, s, False))

        n = len(recs)
        ab = ap = bb = bp = 0
        par_a = par_b = 0
        hit_branch = 0
        gained, lost = [], []
        for tid, A, B, hits in recs:
            hit_branch += bool(hits)
            par_a += parseable(A)
            par_b += parseable(B)
            ra = check(ds, tid, A)
            rb = ra if A == B else check(ds, tid, B)
            a_b = ra["base"] == "pass"
            a_p = a_b and ra["plus"] == "pass"
            b_b = rb["base"] == "pass"
            b_p = b_b and rb["plus"] == "pass"
            ab += a_b; ap += a_p; bb += b_b; bp += b_p
            if b_p and not a_p:
                gained.append(tid)
            if a_p and not b_p:
                lost.append(tid)
        results[arm] = {
            "dataset": ds,
            "mode": mode,
            "n": n,
            "expected_n": expected_n,
            "items_reaching_indent_branch": hit_branch,
            "text_changed_by_fix": sum(1 for _, A, B, _ in recs if A != B),
            "pass_at_1_base_as_run": round(ab / n, 4),
            "pass_at_1_plus_as_run": round(ap / n, 4),
            "pass_at_1_base_corrected": round(bb / n, 4),
            "pass_at_1_plus_corrected": round(bp / n, 4),
            "n_pass_plus_as_run": ap,
            "n_pass_plus_corrected": bp,
            "delta_pp_plus": round(100 * (bp - ap) / n, 2),
            "parseability_as_run": round(par_a / n, 4),
            "parseability_corrected": round(par_b / n, 4),
            "items_gained": gained,
            "items_lost": lost,
        }
        r_ = results[arm]
        print(
            f"{arm:36s} [{ds:9s}/{mode:14s}] n={n:4d} "
            f"plus {r_['pass_at_1_plus_as_run']:.4f} -> {r_['pass_at_1_plus_corrected']:.4f} "
            f"({r_['delta_pp_plus']:+.2f} pp)  indent_branch={hit_branch:4d} "
            f"changed={r_['text_changed_by_fix']:4d}  par {r_['parseability_as_run']:.3f}"
            f"->{r_['parseability_corrected']:.3f}",
            flush=True,
        )

    payload = {
        "what": "Blast radius of the A05 HE+ stitch defect (defect c) across every arm with "
                "recoverable outputs, on the wzc1 dllm_draft run tree.",
        "gpu_cost": "0 (no model loaded; stored outputs re-graded)",
        "corrected_not_replacement": True,
        "grader": "evalplus.eval.untrusted_check, per-dataset self-test (canonical PASS / stub FAIL)",
        "grader_self_tests": selftests,
        "prereg": "A05_BLAST_RADIUS_PREREGISTRATION.md (F1/F2/F3), committed before this ran",
        "arms": results,
    }
    json.dump(payload, open(OUT, "w"), indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
