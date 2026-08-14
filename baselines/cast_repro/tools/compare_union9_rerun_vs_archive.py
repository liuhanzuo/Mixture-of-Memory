#!/usr/bin/env python
"""Compare a union-9 re-run against an ARCHIVED union-9 run, task by task and doc by doc.

Used as the admission gate for the rebuilt pinned harness (lm_eval 0.4.8 +
transformers 4.57.6 in $ROOT/venv_union9) after the 2026-08-13 restart destroyed
the original stack. See scripts/_union9_harness_rebuild_control.sh for context.

WHY BOTH LEVELS ARE CHECKED
---------------------------
A per-task accuracy match is necessary but NOT sufficient. Two runs can land on
the same accuracy while scoring different documents (a permuted split, a
re-versioned dataset, a changed prompt template) -- and the piqa cell in this
control is deliberately loaded from a SUBSTITUTED source, so "same number" would
be the easiest possible way to fool ourselves. Therefore:

  LEVEL 1  per-task acc / acc_norm exact equality.
           Bar is 0 flips, not "within noise": memory
           [[same-harness-runs-bit-identical]] measured same-arch/same-disk/
           same-harness re-runs as BYTE-IDENTICAL. A nonzero delta is real drift
           and is reported with magnitude in pp.

  LEVEL 2  per-doc doc_hash / prompt_hash / target_hash set equality, from the
           samples_*.jsonl. This is what proves the same items were scored with
           the same prompts. It is the check that makes the substituted piqa cell
           admissible rather than merely non-empty.

Exit codes: 0 PASS, 30 per-task drift, 31 doc/prompt drift, 32 structural problem
(missing task, wrong doc count). Nonzero => the rebuilt harness must NOT be used
to add a row to the union-9 table.
"""
import argparse
import glob
import json
import os
import sys

TASKS = ["boolq", "rte", "hellaswag", "race", "piqa", "winogrande",
         "arc_easy", "arc_challenge", "openbookqa"]

# The union-9 table's own protocol counts. Hard-coded so a silently truncated
# split cannot pass by matching a truncated archive.
EXPECT_N = {"boolq": 3270, "rte": 277, "hellaswag": 10042, "race": 1045,
            "piqa": 1838, "winogrande": 1267, "arc_easy": 2376,
            "arc_challenge": 1172, "openbookqa": 500}


def load_results(d):
    fs = sorted(glob.glob(os.path.join(d, "results_*.json")))
    if not fs:
        sys.exit("no results_*.json in %s" % d)
    return json.load(open(fs[-1])), fs[-1]


def load_samples(d, task):
    fs = sorted(glob.glob(os.path.join(d, "samples_%s_*.jsonl" % task)))
    if not fs:
        return None, None
    out = {}
    with open(fs[-1]) as f:
        for line in f:
            r = json.loads(line)
            out[r["doc_id"]] = (r.get("doc_hash"), r.get("prompt_hash"), r.get("target_hash"))
    return out, fs[-1]


def load_logprobs(d, task):
    """Per-doc list of continuation logprobs from filtered_resps.

    These are the RAW model outputs, before argmax. They are what makes a
    hardware/stack difference attributable: two runs can agree on every accuracy
    while disagreeing in the 4th decimal of every logprob (different kernels,
    different reduction order, different batch shape), and conversely a genuine
    numerical difference shows up here long before it flips a single answer.
    """
    fs = sorted(glob.glob(os.path.join(d, "samples_%s_*.jsonl" % task)))
    if not fs:
        return None
    out = {}
    with open(fs[-1]) as f:
        for line in f:
            r = json.loads(line)
            fr = r.get("filtered_resps")
            if not fr:
                continue
            vals = []
            for item in fr:
                # loglikelihood tasks: [logprob, is_greedy]; be tolerant of shape
                if isinstance(item, (list, tuple)) and item:
                    try:
                        vals.append(float(item[0]))
                    except (TypeError, ValueError):
                        vals.append(None)
                else:
                    try:
                        vals.append(float(item))
                    except (TypeError, ValueError):
                        vals.append(None)
            out[r["doc_id"]] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-dir", required=True)
    ap.add_argument("--rerun-dir", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    arch, arch_f = load_results(a.archive_dir)
    # the rerun writes into a model-named subdir
    rerun_dir = a.rerun_dir
    if not glob.glob(os.path.join(rerun_dir, "results_*.json")):
        subs = [d for d in glob.glob(os.path.join(rerun_dir, "*")) if os.path.isdir(d)]
        cands = [d for d in subs if glob.glob(os.path.join(d, "results_*.json"))]
        if len(cands) != 1:
            sys.exit("expected exactly 1 results dir under %s, found %d: %s"
                     % (rerun_dir, len(cands), cands))
        rerun_dir = cands[0]
    rer, rer_f = load_results(rerun_dir)

    print("archive results : %s" % arch_f)
    print("rerun   results : %s" % rer_f)
    print("archive env     : tf=%s torch-ver-in-env-info, git=%s"
          % (arch.get("transformers_version"), arch.get("git_hash")))
    print("rerun   env     : tf=%s git=%s"
          % (rer.get("transformers_version"), rer.get("git_hash")))
    print("archive bs      : %s -> %s" % (arch["config"]["batch_size"], arch["config"]["batch_sizes"]))
    print("rerun   bs      : %s -> %s" % (rer["config"]["batch_size"], rer["config"]["batch_sizes"]))

    report = {"archive_results": arch_f, "rerun_results": rer_f,
              "archive_transformers": arch.get("transformers_version"),
              "rerun_transformers": rer.get("transformers_version"),
              "archive_git_hash": arch.get("git_hash"),
              "rerun_git_hash": rer.get("git_hash"),
              "archive_batch_sizes": arch["config"]["batch_sizes"],
              "rerun_batch_sizes": rer["config"]["batch_sizes"],
              "per_task": {}, "flips": 0, "structural_errors": [],
              "doc_hash_mismatches": {}}

    struct = []
    print("\n%-14s %-10s %14s %14s %12s   %14s %14s %12s" %
          ("task", "n", "archive_acc", "rerun_acc", "d_acc_pp", "archive_accn", "rerun_accn", "d_accn_pp"))
    flips = 0
    for t in TASKS:
        if t not in arch["results"]:
            struct.append("archive missing task %s" % t)
            continue
        if t not in rer["results"]:
            struct.append("RERUN MISSING TASK %s -- 9-task row impossible" % t)
            continue
        ar, rr = arch["results"][t], rer["results"][t]
        an = arch["n-samples"][t]["effective"]
        rn = rer["n-samples"][t]["effective"]
        if an != EXPECT_N[t]:
            struct.append("archive %s n=%d != protocol %d" % (t, an, EXPECT_N[t]))
        if rn != EXPECT_N[t]:
            struct.append("RERUN %s n=%d != protocol %d" % (t, rn, EXPECT_N[t]))
        if an != rn:
            struct.append("%s n differs archive=%d rerun=%d" % (t, an, rn))

        e = {"n_archive": an, "n_rerun": rn}
        row = [t, "%d" % rn]
        for m, key in (("acc", "acc,none"), ("acc_norm", "acc_norm,none")):
            av, rv = ar.get(key), rr.get(key)
            if av is None and rv is None:
                e[m] = None
                row += ["-", "-", "-"]
                continue
            if av is None or rv is None:
                struct.append("%s %s present in only one run (archive=%s rerun=%s)"
                              % (t, m, av, rv))
                row += [str(av), str(rv), "N/A"]
                continue
            d = (rv - av) * 100.0
            e[m] = {"archive": av, "rerun": rv, "delta_pp": d, "identical": av == rv}
            if av != rv:
                flips += 1
            row += ["%.10f" % av, "%.10f" % rv, "%+.6f" % d]
        report["per_task"][t] = e
        print("%-14s %-10s %14s %14s %12s   %14s %14s %12s" % tuple(row))

    # LEVEL 2 -- same docs, same prompts.
    print("\nper-doc hash comparison (doc_hash / prompt_hash / target_hash):")
    for t in TASKS:
        asamp, af = load_samples(a.archive_dir, t)
        rsamp, rf = load_samples(rerun_dir, t)
        if asamp is None or rsamp is None:
            print("  %-14s SKIP (samples missing: archive=%s rerun=%s)"
                  % (t, asamp is not None, rsamp is not None))
            continue
        common = set(asamp) & set(rsamp)
        only_a, only_r = set(asamp) - set(rsamp), set(rsamp) - set(asamp)
        bad = [i for i in sorted(common) if asamp[i] != rsamp[i]]
        report["doc_hash_mismatches"][t] = {
            "n_archive": len(asamp), "n_rerun": len(rsamp),
            "n_common": len(common), "only_archive": len(only_a),
            "only_rerun": len(only_r), "n_hash_mismatch": len(bad),
            "examples": bad[:5]}
        tag = "OK" if not bad and not only_a and not only_r else "MISMATCH"
        print("  %-14s %-9s archive=%d rerun=%d common=%d hash_mismatch=%d"
              % (t, tag, len(asamp), len(rsamp), len(common), len(bad)))
        if bad:
            i = bad[0]
            print("      first: doc %d archive=%s rerun=%s" % (i, asamp[i], rsamp[i]))

    hash_bad = sum(v["n_hash_mismatch"] + v["only_archive"] + v["only_rerun"]
                   for v in report["doc_hash_mismatches"].values())

    # task_hashes is lm_eval's own cumulative digest over every sample's
    # doc_hash+prompt_hash+target_hash (loggers/evaluation_tracker.py:219). It is a
    # single value per task that certifies "same items, same prompts, same targets",
    # independent of my own per-doc comparison above -- so it cross-checks that
    # comparison rather than restating it. It is also the field that would betray a
    # substituted piqa source, which is precisely why it is asserted here.
    print("\ntask_hashes (lm_eval's own cumulative per-task digest):")
    ath, rth = arch.get("task_hashes", {}), rer.get("task_hashes", {})
    report["task_hashes"] = {}
    th_bad = 0
    for t in TASKS:
        av, rv = ath.get(t), rth.get(t)
        same = (av is not None and av == rv)
        if not same:
            th_bad += 1
        report["task_hashes"][t] = {"archive": av, "rerun": rv, "identical": same}
        print("  %-14s %-9s archive=%s rerun=%s"
              % (t, "OK" if same else "MISMATCH",
                 (av or "None")[:16], (rv or "None")[:16]))
    report["task_hash_mismatches"] = th_bad

    # LEVEL 3 -- raw logprob agreement. DIAGNOSTIC ONLY, never a pass/fail gate:
    # its job is to say WHY a flip happened (bit-identical kernels vs. tiny
    # numerical drift that tipped a near-tie), so a BLOCKED verdict comes with an
    # attributable magnitude instead of just a count.
    print("\nraw logprob agreement (diagnostic; explains any flip above):")
    report["logprob_agreement"] = {}
    for t in TASKS:
        al, rl = load_logprobs(a.archive_dir, t), load_logprobs(rerun_dir, t)
        if not al or not rl:
            continue
        common = sorted(set(al) & set(rl))
        n_exact, n_cmp, worst, worst_doc = 0, 0, 0.0, None
        for i in common:
            va, vr = al[i], rl[i]
            if len(va) != len(vr):
                continue
            for x, y in zip(va, vr):
                if x is None or y is None:
                    continue
                n_cmp += 1
                if x == y:
                    n_exact += 1
                else:
                    d = abs(x - y)
                    if d > worst:
                        worst, worst_doc = d, i
        if not n_cmp:
            continue
        frac = n_exact / n_cmp
        report["logprob_agreement"][t] = {
            "n_compared": n_cmp, "n_bit_identical": n_exact,
            "frac_bit_identical": frac, "max_abs_diff": worst,
            "worst_doc_id": worst_doc}
        print("  %-14s bit-identical %6d/%6d (%6.2f%%)  max|d|=%.3e%s"
              % (t, n_exact, n_cmp, frac * 100.0, worst,
                 "" if worst_doc is None else " @doc %d" % worst_doc))

    report["flips"] = flips
    report["total_doc_hash_problems"] = hash_bad
    report["structural_errors"] = struct

    if struct:
        rc, verdict = 32, "BLOCKED_STRUCTURAL"
    elif hash_bad or th_bad:
        rc, verdict = 31, "BLOCKED_DOC_DRIFT"
    elif flips:
        rc, verdict = 30, "BLOCKED_SCORE_DRIFT"
    else:
        rc, verdict = 0, "PASS"
    report["verdict"] = verdict

    print("\nstructural errors      : %d" % len(struct))
    for s in struct:
        print("   ! %s" % s)
    print("per-task metric flips  : %d  (bar is 0)" % flips)
    print("doc/prompt hash issues : %d  (bar is 0)" % hash_bad)
    print("task_hash mismatches   : %d  (bar is 0)" % th_bad)
    print("VERDICT: %s (rc=%d)" % (verdict, rc))
    if rc:
        print("=> the rebuilt harness MUST NOT be used to add a union-9 row.")
    else:
        print("=> rebuilt harness reproduces the archive exactly; admissible.")

    with open(a.output, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote %s" % a.output)
    return rc


if __name__ == "__main__":
    sys.exit(main())
