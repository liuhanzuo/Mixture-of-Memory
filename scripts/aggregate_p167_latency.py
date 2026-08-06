#!/usr/bin/env python
"""#167 — aggregate the depth-replay Read-latency re-run and reconcile it with
paperA/sections/tab_replay_latency.tex.

CONTEXT (the reason this script exists)
---------------------------------------
tab_replay_latency.tex reports j=0 Read = 931.9 ms, j=12 = 664.4 ms, 1.403x.
An audit could not initially locate the producing log, because two nearby
artifacts disagree:

  bench_results/p0_12_acceptance/arm{A,B}_rep{1,2,3}.json -> 1080.9 / 785.7 (1.376x)
  bench_results/p0_12_depth_replay/arm{A,B}_rep{1,2,3}.json -> 1076.7 / 783.7 (1.374x)
  paperA/artifacts/p1_8_serving 128k|cpu G=1                -> 934.5 / 677.8 (1.379x)

The provenance was subsequently FOUND: the table is backed by the P0.13 latency
leg, whose three raw per-process records

  bench_results/p0_13_quality_latency/latency/latency_proc{0,1,2}.json

pool (3 procs x 20 timed reads = 60) to median 931.9195 / 664.3577 ms, ratio
1.40274 -> exactly the tex 931.9 / 664.4 / 1.403x, including p10/p90
(931.5572 -> 931.6, 941.9580 -> 942.0, 663.7733 -> 663.8, 667.1029 -> 667.1).
So the table is CORRECT and the caption's "3 independent processes, 20 reads
each, median" is literally accurate; the earlier confusion came from comparing
it against p0_12_*, which is a DIFFERENT protocol (seed=0, n_decode=6, and a
different pack) and therefore legitimately reports different absolute values.

WHAT THIS SCRIPT DOES
---------------------
1. Pools the re-run's per-process raw read times and recomputes
   median-of-pooled / p10 / p90 / ratio using the SAME statistic the original
   aggregator used (numpy-style linear-interpolation percentiles over the
   pooled 60 samples, and the plain median of those 60).
2. Also reports the median-of-per-process-medians, so the reader can see the
   two aggregation choices agree to <1 ms and the headline is not an artifact
   of the pooling rule.
3. Diffs against (a) the tex values, (b) the original P0.13 record (this is the
   like-for-like comparison), and (c) p0_12_acceptance (explicitly flagged as a
   DIFFERENT protocol, shown only to document that the *direction* replicates).
4. Asserts the re-run's env matches the original P0.13 env AND the
   p0_12_acceptance env field-by-field (torch / cuda / gpu / python), and
   asserts the timing-relevant config fields match the original P0.13 config.
   Any mismatch is reported as FAIL with the offending field.

Usage (pure CPU, no GPU needed):
  python scripts/aggregate_p167_latency.py \
      --rerun_dir bench_results/p0_167_latency_rerun/latency \
      --orig_p0_13_dir bench_results/p0_13_quality_latency/latency \
      --p0_12_dir bench_results/p0_12_acceptance \
      --out bench_results/p0_167_latency_rerun/p167_latency_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

# Tex values currently in paperA/sections/tab_replay_latency.tex.
TEX = {
    "armA_read_ms": 931.9, "armA_p10_ms": 931.6, "armA_p90_ms": 942.0,
    "armB_read_ms": 664.4, "armB_p10_ms": 663.8, "armB_p90_ms": 667.1,
    "ratio": 1.403,
}
# Timing-relevant config fields that MUST match the original P0.13 latency leg.
CFG_KEYS = ("resume_j_a", "resume_j_b", "selector", "topk", "iter_hop_topk",
            "chunk_size", "max_new_tokens", "warmup", "n_repeat", "dtype",
            "attn_impl", "lora_sha256", "num_layers")
ENV_KEYS = ("torch", "cuda", "gpu", "python")


def _pct(xs, p):
    """numpy.percentile-compatible linear interpolation (matches the original
    aggregator's _summ helper)."""
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo = int(k)
    if lo + 1 >= len(xs):
        return xs[lo]
    return xs[lo] + (xs[lo + 1] - xs[lo]) * (k - lo)


def _load_procs(d):
    """Load latency_proc*.json from a P0.13-schema latency dir."""
    out = []
    for f in sorted(glob.glob(os.path.join(d, "latency_proc*.json"))):
        with open(f) as fh:
            out.append((os.path.basename(f), json.load(fh)))
    return out


def _pool(procs):
    """Pool raw per-read times across processes -> per-arm stats in ms."""
    res = {}
    for arm in ("armA", "armB"):
        raw, per_proc_med = [], []
        for _, d in procs:
            r = d[arm]["read_s"]
            raw += list(r["raw"])
            per_proc_med.append(r["median"])
        if not raw:
            continue
        res[arm] = {
            "n_reads": len(raw),
            "median_ms": round(statistics.median(raw) * 1e3, 4),
            "p10_ms": round(_pct(raw, 0.10) * 1e3, 4),
            "p90_ms": round(_pct(raw, 0.90) * 1e3, 4),
            "min_ms": round(min(raw) * 1e3, 4),
            "max_ms": round(max(raw) * 1e3, 4),
            "mean_ms": round(statistics.fmean(raw) * 1e3, 4),
            # robustness check: the other reasonable aggregation rule
            "median_of_proc_medians_ms":
                round(statistics.median(per_proc_med) * 1e3, 4),
            "per_proc_median_ms": [round(m * 1e3, 3) for m in per_proc_med],
        }
    if "armA" in res and "armB" in res:
        res["ratio_A_over_B"] = round(
            res["armA"]["median_ms"] / res["armB"]["median_ms"], 5)
        res["ratio_A_over_B_proc_median_rule"] = round(
            res["armA"]["median_of_proc_medians_ms"]
            / res["armB"]["median_of_proc_medians_ms"], 5)
    return res


def _p0_12_pool(d):
    """p0_12_acceptance stores one file per (arm, rep) with timing.read_s."""
    res = {}
    for arm in ("armA", "armB"):
        raw, meds, envs = [], [], []
        for f in sorted(glob.glob(os.path.join(d, f"{arm}_rep*.json"))):
            with open(f) as fh:
                j = json.load(fh)
            r = j["timing"]["read_s"]
            raw += list(r["raw"])
            meds.append(r["median"])
            envs.append(j.get("env", {}))
        if raw:
            res[arm] = {
                "n_reads": len(raw), "n_reps": len(meds),
                "median_ms": round(statistics.median(raw) * 1e3, 4),
                "median_of_rep_medians_ms":
                    round(statistics.median(meds) * 1e3, 4),
                "env": envs[0] if envs else {},
            }
    if "armA" in res and "armB" in res:
        res["ratio_A_over_B"] = round(
            res["armA"]["median_ms"] / res["armB"]["median_ms"], 5)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun_dir", required=True,
                    help="bench_results/p0_167_latency_rerun/latency")
    ap.add_argument("--orig_p0_13_dir",
                    default="bench_results/p0_13_quality_latency/latency")
    ap.add_argument("--p0_12_dir", default="bench_results/p0_12_acceptance")
    ap.add_argument("--expected_procs", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fails, warns = [], []

    rerun_procs = _load_procs(args.rerun_dir)
    orig_procs = _load_procs(args.orig_p0_13_dir)
    if not rerun_procs:
        print(f"[p167][FAIL] no latency_proc*.json under {args.rerun_dir}")
        sys.exit(2)

    # -- honesty gate: never let a partial run masquerade as the full protocol --
    n_re = len(rerun_procs)
    if n_re != args.expected_procs:
        fails.append(f"rerun has {n_re} process record(s), expected "
                     f"{args.expected_procs} — do NOT report this as the "
                     f"3-process protocol")
    for name, d in rerun_procs:
        for arm in ("armA", "armB"):
            got = d[arm]["read_s"]["n"]
            want = d["config"]["n_repeat"]
            if got != want:
                fails.append(f"{name}:{arm} has {got} timed reads != "
                             f"n_repeat={want}")

    rerun = _pool(rerun_procs)
    orig = _pool(orig_procs) if orig_procs else {}
    p012 = _p0_12_pool(args.p0_12_dir) if os.path.isdir(args.p0_12_dir) else {}

    # ---------------- env / config consistency assertions ------------------- #
    ref_env = orig_procs[0][1]["env"] if orig_procs else None
    ref_cfg = orig_procs[0][1]["config"] if orig_procs else None
    p012_env = p012.get("armA", {}).get("env") or None

    env_report = {}
    for name, d in rerun_procs:
        e = d.get("env", {})
        for k in ENV_KEYS:
            if ref_env is not None and e.get(k) != ref_env.get(k):
                fails.append(f"{name} env.{k}={e.get(k)!r} != P0.13 "
                             f"{ref_env.get(k)!r}")
            if p012_env is not None and e.get(k) != p012_env.get(k):
                fails.append(f"{name} env.{k}={e.get(k)!r} != p0_12_acceptance "
                             f"{p012_env.get(k)!r}")
        c = d.get("config", {})
        for k in CFG_KEYS:
            if ref_cfg is not None and c.get(k) != ref_cfg.get(k):
                fails.append(f"{name} config.{k}={c.get(k)!r} != P0.13 "
                             f"{ref_cfg.get(k)!r}")
        env_report[name] = {"env": e, "pack_sha": d["pack"]["packed_ids_sha256"],
                            "pack_read_len": d["pack"]["pack_read_len"],
                            "sel_idx": d["pack"]["sel_idx"]}

    # Same fixed pack across procs => arms are strictly comparable.
    shas = {v["pack_sha"] for v in env_report.values()}
    if len(shas) != 1:
        fails.append(f"rerun procs used {len(shas)} different packs {shas} — "
                     f"latency arms are not comparable")
    if orig_procs:
        orig_sha = orig_procs[0][1]["pack"]["packed_ids_sha256"]
        if shas and next(iter(shas)) != orig_sha:
            warns.append(f"rerun pack sha {next(iter(shas))[:12]} != original "
                         f"P0.13 pack sha {orig_sha[:12]} — same protocol but a "
                         f"different retrieved pack; absolute ms not directly "
                         f"comparable to the tex values")

    # ------------------------------ diffs ---------------------------------- #
    def _diff(label, a_ms, b_ms, ratio):
        return {
            "label": label,
            "armA_read_ms": a_ms, "armB_read_ms": b_ms, "ratio": ratio,
            "d_armA_ms": None if a_ms is None else round(a_ms - TEX["armA_read_ms"], 4),
            "d_armB_ms": None if b_ms is None else round(b_ms - TEX["armB_read_ms"], 4),
            "d_ratio": None if ratio is None else round(ratio - TEX["ratio"], 5),
            "armA_rel_pct": None if a_ms is None else
                round((a_ms - TEX["armA_read_ms"]) / TEX["armA_read_ms"] * 100, 3),
            "armB_rel_pct": None if b_ms is None else
                round((b_ms - TEX["armB_read_ms"]) / TEX["armB_read_ms"] * 100, 3),
        }

    vs_tex = _diff("rerun_vs_tex", rerun["armA"]["median_ms"],
                   rerun["armB"]["median_ms"], rerun["ratio_A_over_B"])
    vs_orig = None
    if orig:
        vs_orig = {
            "label": "rerun_vs_original_P0.13 (like-for-like, same protocol)",
            "orig_armA_ms": orig["armA"]["median_ms"],
            "orig_armB_ms": orig["armB"]["median_ms"],
            "orig_ratio": orig["ratio_A_over_B"],
            "d_armA_ms": round(rerun["armA"]["median_ms"] - orig["armA"]["median_ms"], 4),
            "d_armB_ms": round(rerun["armB"]["median_ms"] - orig["armB"]["median_ms"], 4),
            "d_ratio": round(rerun["ratio_A_over_B"] - orig["ratio_A_over_B"], 5),
            "orig_reproduces_tex": (
                abs(orig["armA"]["median_ms"] - TEX["armA_read_ms"]) < 0.05
                and abs(orig["armB"]["median_ms"] - TEX["armB_read_ms"]) < 0.05
                and abs(orig["ratio_A_over_B"] - TEX["ratio"]) < 0.001),
        }
    vs_p012 = None
    if p012:
        vs_p012 = {
            "label": "rerun_vs_p0_12_acceptance (DIFFERENT protocol: seed=0, "
                     "n_decode=6, different pack — direction only, absolute ms "
                     "are NOT expected to match)",
            "p012_armA_ms": p012["armA"]["median_ms"],
            "p012_armB_ms": p012["armB"]["median_ms"],
            "p012_ratio": p012["ratio_A_over_B"],
            "d_armA_ms": round(rerun["armA"]["median_ms"] - p012["armA"]["median_ms"], 4),
            "d_armB_ms": round(rerun["armB"]["median_ms"] - p012["armB"]["median_ms"], 4),
            "d_ratio": round(rerun["ratio_A_over_B"] - p012["ratio_A_over_B"], 5),
            "same_env_as_rerun": (p012_env == (rerun_procs[0][1].get("env") and
                {k: rerun_procs[0][1]["env"].get(k) for k in ENV_KEYS})
                if p012_env else None),
        }

    verdict = "FAIL" if fails else ("PASS_WITH_WARNINGS" if warns else "PASS")
    out = {
        "run": "#167_depth_replay_latency_rerun",
        "verdict": verdict,
        "fails": fails, "warnings": warns,
        "n_procs_rerun": n_re, "n_procs_original_P0_13": len(orig_procs),
        "tex_current": TEX,
        "rerun": rerun,
        "original_P0_13": orig,
        "p0_12_acceptance": p012,
        "diffs": {"vs_tex": vs_tex, "vs_original_P0_13": vs_orig,
                  "vs_p0_12_acceptance": vs_p012},
        "provenance_of_tex_values": {
            "resolved": True,
            "source": "bench_results/p0_13_quality_latency/latency/"
                      "latency_proc{0,1,2}.json (P0.13 latency leg)",
            "note": "Pooling the 3x20 raw read times of those files reproduces "
                    "931.9195 / 664.3577 ms and ratio 1.40274, i.e. the tex "
                    "931.9 / 664.4 / 1.403x including p10/p90. p0_12_* is a "
                    "different protocol and is not the source of this table.",
        },
        "per_proc_env_and_pack": env_report,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # ------------------------------ report --------------------------------- #
    A, B = rerun["armA"], rerun["armB"]
    print("=" * 74)
    print(f"[p167] verdict: {verdict}   (procs={n_re}, "
          f"reads/arm={A['n_reads']})")
    print("=" * 74)
    print(f"RERUN   armA(j=0)  median {A['median_ms']:9.3f} ms  "
          f"p10 {A['p10_ms']:9.3f}  p90 {A['p90_ms']:9.3f}")
    print(f"RERUN   armB(j=12) median {B['median_ms']:9.3f} ms  "
          f"p10 {B['p10_ms']:9.3f}  p90 {B['p90_ms']:9.3f}")
    print(f"RERUN   ratio A/B = {rerun['ratio_A_over_B']:.5f}   "
          f"(proc-median rule: {rerun['ratio_A_over_B_proc_median_rule']:.5f})")
    print(f"        per-proc medians A={A['per_proc_median_ms']} "
          f"B={B['per_proc_median_ms']}")
    print("-" * 74)
    print(f"TEX     armA {TEX['armA_read_ms']} / armB {TEX['armB_read_ms']} "
          f"/ ratio {TEX['ratio']}")
    print(f"vs TEX  dA {vs_tex['d_armA_ms']:+.3f} ms ({vs_tex['armA_rel_pct']:+.2f}%)  "
          f"dB {vs_tex['d_armB_ms']:+.3f} ms ({vs_tex['armB_rel_pct']:+.2f}%)  "
          f"dratio {vs_tex['d_ratio']:+.5f}")
    if vs_orig:
        print("-" * 74)
        print(f"ORIG P0.13 (same protocol) armA {vs_orig['orig_armA_ms']} / "
              f"armB {vs_orig['orig_armB_ms']} / ratio {vs_orig['orig_ratio']}")
        print(f"        original reproduces tex exactly: "
              f"{vs_orig['orig_reproduces_tex']}")
        print(f"vs ORIG dA {vs_orig['d_armA_ms']:+.3f} ms  "
              f"dB {vs_orig['d_armB_ms']:+.3f} ms  "
              f"dratio {vs_orig['d_ratio']:+.5f}")
    if vs_p012:
        print("-" * 74)
        print(f"p0_12_acceptance (DIFFERENT protocol) armA "
              f"{vs_p012['p012_armA_ms']} / armB {vs_p012['p012_armB_ms']} / "
              f"ratio {vs_p012['p012_ratio']}")
        print(f"vs p012 dA {vs_p012['d_armA_ms']:+.3f} ms  "
              f"dB {vs_p012['d_armB_ms']:+.3f} ms  "
              f"dratio {vs_p012['d_ratio']:+.5f}   "
              f"(absolute ms not expected to match; direction does)")
    if warns:
        print("-" * 74)
        for w in warns:
            print(f"[WARN] {w}")
    if fails:
        print("-" * 74)
        for f_ in fails:
            print(f"[FAIL] {f_}")
    print("=" * 74)
    print(f"[p167] wrote {args.out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
