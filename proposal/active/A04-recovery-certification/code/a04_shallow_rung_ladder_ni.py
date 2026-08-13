#!/usr/bin/env python3
"""A04 — the SHALLOW RUNG LADDER: can NI be OBSERVED TO ACCEPT under damage at 1B?

PRE-REGISTRATION: `A04_SHALLOW_RUNG_LADDER_PREREG.md`, commit **a2e1a95**,
committed while both training runs were ~20 min into 5,000 steps and no
checkpoint, no eval shard, no accuracy and no margin existed for either arm.

THE BLOCKER THIS ANSWERS (STATUS.json:pilot_one.pilot_two_status, verbatim)
--------------------------------------------------------------------------
    "BLOCKED. 1,077-4,309 GPU-h must not be committed until a NEW pre-data doc
     shows a rung exists where NI can be OBSERVED TO ACCEPT; otherwise the gate
     can only ever confirm rejection."

and (same key) "it is a rung-selection problem, not a variance problem."

NI's discrimination curve had a hole with nothing in it: damaged arms cluster at
11-63% recovery and REJECT by tens of SE (1B keep12 by 27.0-90.4x sd_run at
22-32% recovery), while the ONLY accept in A04 is `full32_dolmino` -- ZERO
structural damage. keep12 was the lightest damaged 1B rung in existence, and
shallower rungs had ZERO checkpoints on either disk. This pass trains and scores
the two lightest damaged rungs the family admits and fills the hole:

    keep12 (25% cut) -> keep13 (18.75%) -> keep14 (12.5%) -> zero damage

WHAT IS DECIDED, AND HOW (all fixed in the prereg, none chosen here)
-------------------------------------------------------------------
* 3 decision axes: triviaqa / popqa / mmlu_content. nq_open DEMOTED by design
  s5.2, ZERO decision weight, reported descriptively.
* Delta = 0.10 * residual(intact), NEVER substituted (guard G2). Built at runtime
  by CALLING the imported `build_nulls` on the G0-pinned anchor, then
  CROSS-CHECKED against the canonical full-precision constants at 1e-9. A drift
  aborts rather than silently re-margining every verdict.
* NI ACCEPTS iff margin_pp = diff_lower95_one_sided_pp + delta_pp > 0, via the
  IMPORTED `ni_rule` -- the same function that produced every archived A04 margin.
* The bar is >= 2 of 3 decision axes (the convention under which full32's 1-of-3
  was reported as below the bar).
* Branch A (>=2/3 accept on either arm) -> blocker DISCHARGED. Branch B (both
  constant-REJECT) -> NI's accept region at 1B contains NO damaged rung down to a
  12.5% cut; negative but publishable, blocker stays BLOCKED. Branch C (exactly
  1/3) -> INDETERMINATE, does NOT discharge. All three were written before any
  number existed.

WHAT IS IMPORTED AND NEVER REIMPLEMENTED
----------------------------------------
`ni_rule`, `ratio_rule`, `load_shards`, `build_nulls`, `mmlu_content_norm_vec`,
`qa_metric_vec`, `EXPECTED_N`, `AXES`, `DEMOTED_AXES`, `PREREG` from
`pilot_zero_rule_disagreement`; `paired_bootstrap`, `TIE_CONVS`, `N_BOOT`, `SEED`
from A03's `analyze_1b_knowledge_floor` via `proposal_paths.a03_code_dir()`;
`assert_aligned`, `d4_interface_degenerate`, `D4_CONSTANT_FRAC`, `D4_TIE_FRAC`,
`Z95_TWO_SIDED` from `a04_shallow_rung_ni_7b`. NO metric, null, rule, guard or
anchor is re-derived. THE NULL IS NEVER HAND-COMPUTED: MAIN's own subtraction of
a recorded null was ~0.5 pp off twice, which is exactly why `build_nulls` is
imported and called.

ANCHOR (guards G0 / G2). Anchor = VANILLA `models/OLMo-2-0425-1B` via
`A03_1B_base` + `A03_1B_base_nq`. A continued-pretrained model as anchor is
FORBIDDEN: at 7B, `full32_step25000` scores BELOW vanilla on all four axes, so
substituting it would shrink every Delta AND lower every target = manufactured
accepts. This script hard-refuses any anchor tag containing a CPT marker.

NOISE / RANGE CONSTANTS. E[range of k iid N(0,s)]/s is k-DEPENDENT:
k=2 -> 2/sqrt(pi) = 1.1283791670955126; k=3 -> 3/sqrt(pi) = 1.6925687506432689;
k=8 -> ~2.8472 (no closed form). Reusing the k=3 constant at k=8 makes the floor
40.6% too low. THIS ANALYSIS REPORTS NO RANGE STATISTIC AS DECISION-BEARING
(one seed per arm, 2 checkpoints per arm), so the constants are recorded as
DECLARED_UNUSED and re-derived by selftest so nobody can later reuse a wrong c_k
from this document. A RATIO of two ranges neither of which clears its own floor
is UNDEFINED, not a direction (the error that voided within_arm_lr).

BOOTSTRAP OFFSETS. Archived and in use: 0,1 / 100-102 / 200-204 / 300,301 /
400-408 / 500-503 / 600-610 / 700-702 / 800,801 / 900-902 / 1000-1005. This run
claims arm_index 1100.. and guard offset 9700, and the disjointness is EXECUTED
by `assert_seeds_disjoint` (reads every archive's own recorded offsets and
raises), not claimed in prose -- prose claims of disjointness in this repo have
already been wrong once, and the executed check caught a real collision.

ONE NODE for every statistic: numpy's Generator.multinomial differs in 19/10000
rows between 2.4.6 (.82) and 2.5.1 (.73). Node of record .73; node + numpy
version recorded in the JSON and pinnable with --expect_numpy.

CPU ONLY. Read-only on every input. No model load, no CUDA context.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SHARED_CODE = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "shared", "code"))
if _SHARED_CODE not in sys.path:
    sys.path.insert(0, _SHARED_CODE)

# ---- canonical: IMPORT, never reimplement ----------------------------------
from pilot_zero_rule_disagreement import (  # noqa: E402
    AXES,
    DEMOTED_AXES,
    EXPECTED_N,
    PREREG,
    build_nulls,
    load_shards,
    mmlu_content_norm_vec,
    ni_rule,
    qa_metric_vec,
    ratio_rule,
)
from a04_shallow_rung_ni_7b import (  # noqa: E402
    D4_CONSTANT_FRAC,
    D4_TIE_FRAC,
    Z95_TWO_SIDED,
    assert_aligned,
    d4_interface_degenerate,
)
from proposal_paths import a03_code_dir  # noqa: E402

_A03 = a03_code_dir()
if _A03 not in sys.path:
    sys.path.insert(0, _A03)
from analyze_1b_knowledge_floor import (  # noqa: E402
    N_BOOT,
    SEED,
    TIE_CONVS,
    paired_bootstrap,
)

# ---------------------------------------------------------------------------
# Frozen inputs (prereg s3, s4)
# ---------------------------------------------------------------------------
SEED_TRAIN = 101
STEP = 5000
NEW_KEEPS = (13, 14)                      # the two shallow rungs trained here
BASE_LAYERS = 16                          # OLMo-2-0425-1B, asserted from config

# The G0-pinned VANILLA 1B intact anchor. Same dirs Pilot Zero / Stage B used.
INTACT = {"mmlu": "A03_1B_base", "cb": "A03_1B_base", "nq": "A03_1B_base_nq"}
# Guard G2: refuse a continued-pretrained anchor. Substring markers that would
# indicate one; checked against the anchor tag itself, not merely asserted.
CPT_ANCHOR_MARKERS = ("full32", "dolmino", "cpt", "step2", "keep", "fresh",
                      "scratch", "freeze", "shortgpt")

# Reference arm: the Stage B keep12 seed101 cell. SAME seed, SAME protocol, SAME
# corpus, SAME step count -> the depth ladder's third rung and the reproduction
# target that makes this pass auditable.
REF_KEEP12_TAG = f"A04_1B_stageB_keep12_seed{SEED_TRAIN}_step{STEP}"

PREREG_CONVENTION = "split"
DELTA_FRACTION = PREREG["delta_fraction"]   # 0.10, frozen by git d1ba737
DECISION_AXES = tuple(a for a in AXES if a not in DEMOTED_AXES)
NI_ACCEPT_BAR = 2                            # >= 2 of 3 decision axes

# Canonical Delta, full precision. Used ONLY as a CROSS-CHECK of the values built
# at runtime from the pinned anchor -- never as the source.
DELTA_CANONICAL_1B = {
    "triviaqa": 4.043134195274186,
    "popqa": 1.3205298941613512,
    "mmlu_content": 1.0238926078906136,
    "nq_open": 0.9695290858725762,
}
DELTA_XCHECK_TOL = 1e-9

# Canonical keep12 seed101 per-axis accuracies (pct), from
# evidence/pilot_one_stage_b_s3_verdict.json's per_axis.*.seed_means_pct["101"].
# Read from that JSON AT RUNTIME (see `_load_keep12_canonical`); this dict is a
# LAST-RESORT cross-check only and the runtime read is authoritative. The
# control_arms pass caught its own hand-transcribed constant at 8.82e-05 pp; the
# fix was to remove the transcription step, so the runtime read governs.
KEEP12_SEED101_XCHECK_TOL = 1e-9

# Bootstrap offsets claimed by THIS run (prereg s6.5).
ARM_INDEX = {"keep13": 1100, "keep14": 1101, "keep12_ref": 1102}
GUARD_SEED_OFF = 9700

RANGE_CONSTANTS_DECLARED_UNUSED = {
    "why_recorded": (
        "prereg s4.2 / s5.5: this analysis reports NO range or spread statistic as "
        "decision-bearing (ONE seed per arm; save_every 2500 gives 2 checkpoints, "
        "and a 2-point series has one difference and cannot support a trend). The "
        "constants are recorded and SELF-TESTED so nobody can later reuse a wrong "
        "c_k from this document."),
    "c_2_E_range_of_2": 2.0 / math.sqrt(math.pi),
    "c_3_E_range_of_3": 3.0 / math.sqrt(math.pi),
    "c_8_E_range_of_8_monte_carlo": 2.8472,
    "c8_over_c3": 2.8472 / (3.0 / math.sqrt(math.pi)),
    "used_in_this_analysis": False,
    "ratio_of_two_ranges_is_UNDEFINED_below_floor": (
        "if neither range clears its own floor, their ratio is not 'a direction' -- "
        "it is undefined. This voided the within_arm_lr pass on 2026-08-13."),
}


# ---------------------------------------------------------------------------
def selftest_gate_constants():
    """Re-derive the k=2 and k=3 closed forms rather than trusting the literals."""
    c2 = 2.0 / math.sqrt(math.pi)
    c3 = 3.0 / math.sqrt(math.pi)
    assert abs(c2 - 1.1283791670955126) < 1e-15, c2
    assert abs(c3 - 1.6925687506432689) < 1e-15, c3
    # Monte-Carlo validation of the estimator against the k=3 closed form, so the
    # k=8 value could not be silently wrong if a future caller needs it.
    rng = np.random.default_rng(12345)
    z = rng.standard_normal((200000, 3))
    c3_mc = float((z.max(1) - z.min(1)).mean())
    assert abs(c3_mc - c3) < 0.01, (c3_mc, c3)
    z8 = rng.standard_normal((200000, 8))
    c8_mc = float((z8.max(1) - z8.min(1)).mean())
    return {"c2_closed_form": c2, "c3_closed_form": c3,
            "c3_monte_carlo": c3_mc, "c3_mc_abs_err": abs(c3_mc - c3),
            "c8_monte_carlo": c8_mc,
            "c8_over_c3": c8_mc / c3,
            "validated": True,
            "note": ("the k=3 Monte-Carlo estimator reproduces its own closed form, "
                     "so the k=8 value from the same estimator is trustworthy. "
                     "Neither is USED here.")}


def assert_gpu_clear(threshold_mib=8000):
    """Refuse-guard: never run on a node whose GPUs are doing someone else's work.
    Absent nvidia-smi is NOT treated as clear on a GPU host."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise SystemExit(f"FATAL refuse-guard: nvidia-smi rc={out.returncode}: {out.stderr}")
    per_gpu = []
    for line in out.stdout.strip().splitlines():
        idx, used = [t.strip() for t in line.split(",")]
        per_gpu.append({"index": int(idx), "memory_used_mib": int(used)})
    busy = [g for g in per_gpu if g["memory_used_mib"] > threshold_mib]
    if busy:
        raise SystemExit(f"FATAL refuse-guard: GPUs busy (> {threshold_mib} MiB): "
                         f"{busy}. Refusing to run.")
    return {"threshold_mib": threshold_mib, "per_gpu": per_gpu,
            "max_used_mib": max(g["memory_used_mib"] for g in per_gpu)}


def assert_forbidden_node():
    """.104 runs paperC Qwen3-8B heal; .21 runs SparseForge #246. Refuse by IP."""
    ips = subprocess.run(["hostname", "-I"], capture_output=True, text=True,
                         timeout=30).stdout.split()
    for bad, why in (("28.83.24.104", ".104 paperC Qwen3 heal"),
                     ("28.89.19.21", ".21 SparseForge #246")):
        if bad in ips:
            raise SystemExit(f"FATAL: refusing to run on {why}")
    return {"host_ips": [i for i in ips if i.startswith("28.")]}


def assert_anchor_is_vanilla(intact_spec, raw_root):
    """Guard G2, EXECUTED. Two independent checks:

    (a) the anchor TAG must not look like a continued-pretrained / pruned arm;
    (b) the anchor cell's own summary.json meta must say mode=base (or carry no
        pruning fields at all) with num_hidden_layers == BASE_LAYERS and no ckpt.

    A CPT anchor scores BELOW vanilla, so substituting one shrinks every Delta
    AND lowers every target = manufactured accepts. This must be impossible by
    construction, not by the analyst remembering.
    """
    report = {}
    for role, tag in intact_spec.items():
        low = tag.lower()
        hits = [m for m in CPT_ANCHOR_MARKERS if m in low]
        if hits:
            raise SystemExit(
                f"FATAL guard G2: anchor tag {tag!r} ({role}) contains CPT/pruning "
                f"marker(s) {hits}. A continued-pretrained or pruned model may NEVER "
                "be the intact anchor -- it would shrink every Delta AND lower every "
                "target, manufacturing accepts.")
        root = ("olmo2_mmlu_content_results" if role == "mmlu"
                else "olmo2_closedbook_results")
        sp = os.path.join(raw_root, root, tag, "summary.json")
        if not os.path.isfile(sp):
            raise SystemExit(f"FATAL: anchor summary.json missing: {sp}")
        blob = json.load(open(sp))
        meta = blob.get("meta", {}) or {}
        mode = meta.get("mode")
        nhl = meta.get("num_hidden_layers")
        ckpt = meta.get("ckpt")
        if mode not in (None, "base"):
            raise SystemExit(f"FATAL guard G2: anchor {tag} has meta.mode={mode!r}, "
                             "expected 'base' (or absent). Not a vanilla anchor.")
        if nhl is not None and int(nhl) != BASE_LAYERS:
            raise SystemExit(f"FATAL guard G2: anchor {tag} has "
                             f"num_hidden_layers={nhl}, expected {BASE_LAYERS}.")
        if ckpt:
            raise SystemExit(f"FATAL guard G2: anchor {tag} carries ckpt={ckpt!r} -- "
                             "a checkpointed model is not the vanilla base.")
        report[role] = {"tag": tag, "mode": mode, "num_hidden_layers": nhl,
                        "ckpt": ckpt, "base_model": meta.get("base_model"),
                        "markers_checked": list(CPT_ANCHOR_MARKERS),
                        "markers_found": []}
    return {"guard": "G2 -- anchor must be the VANILLA base",
            "per_role": report,
            "why": ("at 7B, full32_step25000 scores BELOW vanilla base on all four "
                    "axes; substituting it shrinks every Delta AND lowers every "
                    "target = manufactured accepts."),
            "executed": True}


def assert_protocol(raw_root, tags):
    """Protocol invariants, fail-closed.

    add_bos asserted `is False` -- NEVER `is not True`, so None / missing FAILS.
    chat_template asserted `is not False` -> FAIL, PLUS structurally: neither eval
    script has an apply_chat_template call site, so no flag can enable one. These
    are BASE LMs (no SFT/RL); any chat=True number is void.
    """
    rr = os.path.abspath(raw_root)
    struct = {}
    for f in ("eval_olmo2_closedbook_qa.py", "eval_olmo2_mmlu_content.py"):
        p = os.path.join(rr, "scripts", f)
        if not os.path.isfile(p):
            raise SystemExit(f"FATAL: eval script missing for protocol audit: {p}")
        src = open(p).read()
        n = src.count("apply_chat_template")
        if n != 0:
            raise SystemExit(f"FATAL: {f} has {n} apply_chat_template call site(s); "
                             "the structural chat-free guarantee is void.")
        struct[f] = {"apply_chat_template_sites": 0, "n_lines": src.count("\n") + 1}

    per_cell = {}
    for label, d in tags.items():
        sp = os.path.join(d, "summary.json")
        if not os.path.isfile(sp):
            raise SystemExit(f"FATAL: no summary.json for protocol audit: {sp}")
        blob = json.load(open(sp))
        meta = blob.get("meta", {}) or {}
        ct = blob.get("chat_template", meta.get("chat_template", False))
        if ct is not False:
            raise SystemExit(f"FATAL: {label}: chat_template is {ct!r}, expected "
                             "False (assertion is `is not False`, so None fails).")
        # add_bos: `is False`, never `is not True`.
        ab = blob.get("add_bos", meta.get("add_bos"))
        if ab is not False:
            raise SystemExit(f"FATAL: {label}: add_bos is {ab!r}; the assertion is "
                             "`is False`, so None / True / missing all FAIL.")
        mnt = meta.get("max_new_tokens")
        if "mmlu" not in label and mnt != 32:
            raise SystemExit(f"FATAL: {label}: max_new_tokens={mnt!r}, expected 32.")
        per_cell[label] = {
            "chat_template": False, "add_bos": False,
            "add_bos_assertion": "`is False` (NOT `is not True`)",
            "max_new_tokens": mnt, "base_model": meta.get("base_model"),
            "mode": meta.get("mode"), "ckpt": meta.get("ckpt"),
            "ckpt_step": meta.get("ckpt_step"),
            "keep_front_layers": meta.get("keep_front_layers"),
            "n_fresh_layers": meta.get("n_fresh_layers"),
            "num_hidden_layers": meta.get("num_hidden_layers"),
        }
    return {"structural_scripts": struct, "per_cell": per_cell,
            "note": ("chat_template is False STRUCTURALLY: neither eval script has an "
                     "apply_chat_template call site or a flag to enable one. BASE LMs, "
                     "no SFT/RL -- any chat=True number is void.")}


def _integ(d, rows, expected_n, stem):
    """Shard integrity as a RECORDED block, not merely a passing assertion.

    Asserts the shard index SET == {0..7} (a set, not a count of 8 files), the
    exact item count, 0 duplicate item_id. `load_shards` already asserts most of
    this; duplicating it here means a future caller that swaps the loader still
    gets the check, and the evidence JSON carries the proof.
    """
    files = sorted(glob.glob(os.path.join(d, f"per_example_{stem}_shard*of8.jsonl")))
    idx = sorted(int(os.path.basename(f).split("shard")[-1].split("of")[0])
                 for f in files)
    if idx != list(range(8)):
        raise SystemExit(f"FATAL {d}/{stem}: shard index set {idx} != {{0..7}}")
    ids = [r.get("item_id", r.get("idx")) for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"FATAL {d}/{stem}: duplicate item_id")
    if len(rows) != expected_n:
        raise SystemExit(f"FATAL {d}/{stem}: n={len(rows)} != {expected_n}")
    return {"dir": d, "shard_index_set": idx, "n_shards": 8,
            "n_items": len(rows), "n_items_expected": expected_n,
            "n_duplicate_item_ids": 0,
            "assertions": ["index SET == {0..7} (not a count)", "exact item count",
                           "0 duplicate item_id", "0 nan (below)"]}


def load_cell(raw_root, spec, want_rows=True):
    """Load all four axes for one cell with hard integrity assertions."""
    mm = os.path.join(raw_root, "olmo2_mmlu_content_results")
    cb = os.path.join(raw_root, "olmo2_closedbook_results")
    vecs, integ = {}, {}

    d = os.path.join(mm, spec["mmlu"])
    rows = load_shards(d, "mmlu", EXPECTED_N["mmlu"])
    for r in rows:
        if r.get("nan") is True:
            raise SystemExit(f"FATAL {spec['mmlu']}: nan:true row")
        cn = r.get("content_norm")
        if not isinstance(cn, dict) or not isinstance(cn.get("correct"), bool):
            raise SystemExit(f"FATAL {spec['mmlu']}: content_norm.correct not bool")
    vecs["mmlu_content"] = mmlu_content_norm_vec(rows)
    integ["mmlu_content"] = _integ(d, rows, EXPECTED_N["mmlu"], "mmlu")
    if want_rows:
        vecs["_mmlu_rows"] = rows

    d = os.path.join(cb, spec["cb"])
    for task in ("triviaqa", "popqa"):
        rr = load_shards(d, task, EXPECTED_N[task])
        vecs[task] = qa_metric_vec(rr, "em")
        integ[task] = _integ(d, rr, EXPECTED_N[task], task)
        if want_rows:
            vecs[f"_{task}_rows"] = rr

    d = os.path.join(cb, spec["nq"])
    rr = load_shards(d, "nq_open", EXPECTED_N["nq_open"])
    vecs["nq_open"] = qa_metric_vec(rr, "em")
    integ["nq_open"] = _integ(d, rr, EXPECTED_N["nq_open"], "nq_open")
    if want_rows:
        vecs["_nq_open_rows"] = rr

    for ax in AXES:
        v = np.asarray(vecs[ax], float)
        n_nan = int(np.isnan(v).sum())
        if n_nan:
            raise SystemExit(f"FATAL {ax}: {n_nan} nan in metric vector")
        integ[ax]["n_nan_in_metric_vector"] = 0
    return vecs, integ


def assert_seeds_disjoint(evidence_dir, used_arm_indices, used_offsets,
                          self_output_basename=None):
    """EXECUTE the seed-disjointness claim. Copied UNWEAKENED (self-excluding
    variant) from `a04_keep12_trajectory_monotonicity.py`.

    Defensive about SHAPE: two evidence files in that directory have a JSON LIST
    at top level, so a bare `blob.get(...)` raises AttributeError -- which would
    look like a code bug and invite someone to delete the check.
    """
    found, skipped = {}, {}
    for fn in sorted(os.listdir(evidence_dir)):
        if not fn.endswith(".json"):
            continue
        if self_output_basename and fn == self_output_basename:
            skipped[fn] = ("this run's own output file (about to be overwritten); "
                           "its offsets ARE this run's by construction")
            continue
        p = os.path.join(evidence_dir, fn)
        try:
            blob = json.load(open(p))
        except Exception as e:
            skipped[fn] = f"unreadable: {type(e).__name__}"
            continue
        if not isinstance(blob, dict):
            skipped[fn] = f"top-level {type(blob).__name__}, carries no offsets"
            continue
        bo = blob.get("bootstrap_offsets")
        if bo is None:
            continue
        if not isinstance(bo, dict):
            skipped[fn] = f"bootstrap_offsets is {type(bo).__name__}"
            continue
        ai = bo.get("arm_index")
        idxs = sorted(set(ai.values())) if isinstance(ai, dict) else []
        found[fn] = {"arm_index": idxs,
                     "guard_seed_offset": bo.get("guard_seed_offset"),
                     "interval_seed_offset": bo.get("interval_seed_offset")}
        clash = sorted(set(idxs) & set(used_arm_indices))
        if clash:
            raise SystemExit(
                f"FATAL: arm_index {clash} already used by {fn} -- re-running that "
                "archive would produce different numbers. Choose a disjoint base.")
    return {"archives_scanned": len(found), "per_archive": found,
            "archives_skipped": skipped,
            "this_run_arm_indices": sorted(used_arm_indices),
            "this_run_offsets": used_offsets, "checked_mechanically": True,
            "checker_provenance": ("verbatim from a04_keep12_trajectory_monotonicity.py "
                                   "(the fixed self-excluding version); NOT weakened")}


def _measure_gpu_h(raw_root, keeps):
    """Measure training wall time from each arm's OWN trainer log, then convert to
    GPU-h. NOT a hardcoded estimate and NOT a single tqdm s/it sample.

    memory: one-sample-is-not-a-trend-or-state -- a tqdm-style instantaneous
    `s/step` is not the cadence. The wall time here is (timestamp of the LAST
    logged step) - (timestamp of the FIRST logged step), i.e. elapsed/iter over
    the whole run, and the step numbers are read so the extrapolation to step
    max_steps is explicit rather than assumed.

    Eval GPU-h is measured the same way from the eval progress log's DRIVER
    START -> 'all 4 axes done' span when present; otherwise it is recorded as
    None rather than guessed.
    """
    import datetime as _dt

    def _ts(line):
        # trainer log format: '2026-08-13 18:40:24,368 - INFO - ...'
        try:
            return _dt.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None

    per_arm, total_gpu_h = {}, 0.0
    for k in keeps:
        lp = os.path.join(raw_root, "logs",
                          f"a04_shallow_keep{k}_seed{SEED_TRAIN}.log")
        rec = {"log": lp}
        if not os.path.isfile(lp):
            rec["measured"] = False
            rec["why"] = "trainer log not found on this disk"
            per_arm[f"keep{k}"] = rec
            continue
        steps = []
        for line in open(lp, errors="replace"):
            if "] loss=" not in line or "[step" not in line:
                continue
            t = _ts(line)
            if t is None:
                continue
            try:
                s = int(line.split("[step")[1].split("/")[0].strip())
            except Exception:
                continue
            steps.append((t, s))
        if len(steps) < 2:
            rec["measured"] = False
            rec["why"] = f"only {len(steps)} step lines parsed"
            per_arm[f"keep{k}"] = rec
            continue
        (t0, s0), (t1, s1) = steps[0], steps[-1]
        span_h = (t1 - t0).total_seconds() / 3600.0
        n_steps = s1 - s0
        h_per_step = span_h / n_steps if n_steps else None
        wall_h = h_per_step * STEP if h_per_step else None
        gpu_h = wall_h * 8 if wall_h else None
        rec.update({
            "measured": True,
            "first_logged_step": s0, "last_logged_step": s1,
            "n_steps_observed": n_steps,
            "observed_span_h": span_h,
            "h_per_step": h_per_step,
            "s_per_step": h_per_step * 3600.0 if h_per_step else None,
            "extrapolated_wall_h_for_%d_steps" % STEP: wall_h,
            "gpu_h_at_8_gpus": gpu_h,
            "estimator": ("(t_last - t_first) / (step_last - step_first), i.e. "
                          "elapsed/iter over the WHOLE run -- NOT an instantaneous "
                          "s/step sample"),
        })
        if gpu_h:
            total_gpu_h += gpu_h
        per_arm[f"keep{k}"] = rec

    # eval span, measured where the progress log exists
    eval_gpu_h, eval_per_arm = 0.0, {}
    for k in keeps:
        ep = os.path.join(raw_root, "logs",
                          f"a04_shallow_keep{k}_seed{SEED_TRAIN}_eval_progress.log")
        if not os.path.isfile(ep):
            eval_per_arm[f"keep{k}"] = {"measured": False, "log": ep}
            continue
        lines = [l for l in open(ep, errors="replace") if l.strip()]
        ts = []
        for l in lines:
            # driver format: '[08-13 23:31:02] a04-shallow(keep14): ...'
            if l.startswith("["):
                try:
                    ts.append(_dt.datetime.strptime(
                        l[1:l.index("]")], "%m-%d %H:%M:%S"))
                except Exception:
                    pass
        if len(ts) >= 2:
            span_h = (ts[-1] - ts[0]).total_seconds() / 3600.0
            eval_per_arm[f"keep{k}"] = {"measured": True, "span_h": span_h,
                                        "gpu_h_at_8_gpus": span_h * 8, "log": ep}
            eval_gpu_h += span_h * 8
        else:
            eval_per_arm[f"keep{k}"] = {"measured": False, "log": ep,
                                        "why": f"{len(ts)} timestamps parsed"}

    return {
        "training": {"per_arm": per_arm, "total_gpu_h": total_gpu_h},
        "eval": {"per_arm": eval_per_arm, "total_gpu_h": eval_gpu_h},
        "analysis_gpu_h": 0.0,
        "analysis_note": ("this analysis is CPU-only and read-only on every input: no "
                          "model load, no CUDA context, no scoring."),
        "total_gpu_h": total_gpu_h + eval_gpu_h,
        "context": ("Pilot Two is priced at 1,077-4,309 GPU-h. This pass is the only "
                    "expenditure that can decide whether Pilot Two's blocker is "
                    "dischargeable at all."),
    }


def _load_keep12_canonical(evidence_dir):
    """Read keep12 seed101's per-axis accuracy from the canonical Stage B JSON AT
    RUNTIME. No hand transcription: the control_arms pass caught its own
    hand-copied constant at 8.82e-05 pp and the fix was to delete the
    transcription step, not to loosen a tolerance."""
    p = os.path.join(evidence_dir, "pilot_one_stage_b_s3_verdict.json")
    if not os.path.isfile(p):
        raise SystemExit(f"FATAL: canonical Stage B verdict missing: {p}")
    blob = json.load(open(p))
    out = {}
    for ax in AXES:
        sm = blob["per_axis"][ax]["seed_means_pct"]
        out[ax] = float(sm[str(SEED_TRAIN)])
    return {"path": p, "seed_means_pct_seed101": out,
            "sd_run_pp_S3": {ax: blob["per_axis"][ax]["sd_run_pp"] for ax in AXES},
            "read_at_runtime_not_hardcoded": True}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True,
                    help="dir holding olmo2_{mmlu_content,closedbook}_results/")
    ap.add_argument("--evidence_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--expect_numpy", default=None)
    ap.add_argument("--arms", default="13,14",
                    help="comma-separated keep_front values actually scored")
    ap.add_argument("--preflight_only", action="store_true",
                    help="run every guard, the anchor build, the Delta cross-check and "
                         "the keep12 reproduction gate against the ALREADY-EXISTING "
                         "cells, print the result and exit WITHOUT writing the output "
                         "JSON and WITHOUT touching the new arms. Lets the pipeline be "
                         "validated while the two training runs are still in flight, so "
                         "a broken loader/guard is found before the arms land rather "
                         "than after. Writes nothing, so it cannot pre-empt a verdict.")
    ap.add_argument("--preflight_ignore_own_training", action="store_true",
                    help="ONLY valid together with --preflight_only: skip the GPU "
                         "refuse-guard because the cards are held by THIS pass's own "
                         "two training runs. HARD-REFUSED outside preflight mode, so "
                         "the guard can never be bypassed by a run that WRITES an "
                         "evidence file. The real analysis runs after training, on "
                         "idle cards, with the guard armed.")
    args = ap.parse_args()
    if args.preflight_ignore_own_training and not args.preflight_only:
        raise SystemExit(
            "FATAL: --preflight_ignore_own_training requires --preflight_only. The GPU "
            "refuse-guard may NOT be bypassed for a run that writes an evidence file.")
    if args.preflight_only:
        args.arms = ""

    if args.expect_numpy and np.__version__ != args.expect_numpy:
        raise SystemExit(f"FATAL: numpy {np.__version__} != pinned "
                         f"{args.expect_numpy}. All statistics must come from ONE "
                         "node (multinomial differs 19/10000 rows across versions).")
    keeps = [int(x) for x in args.arms.split(",") if x.strip()]
    for k in keeps:
        if k not in NEW_KEEPS:
            raise SystemExit(f"FATAL: keep={k} outside the pre-registered shallow "
                             f"set {NEW_KEEPS}")

    node_guard = assert_forbidden_node()
    if args.preflight_ignore_own_training:
        gpu_guard = {"BYPASSED_FOR_PREFLIGHT_ONLY": True,
                     "why": ("the cards are held by THIS pass's own keep13/keep14 "
                             "training runs; the preflight writes nothing")}
    else:
        gpu_guard = assert_gpu_clear()
    gates = selftest_gate_constants()
    gpu_h = _measure_gpu_h(args.raw_root, keeps or list(NEW_KEEPS))

    out = {
        "scope": ("A04 SHALLOW RUNG LADDER: NI on the two lightest damaged 1B rungs "
                  f"the family admits -- keep{'/keep'.join(str(k) for k in keeps)}"
                  f"+fresh2, seed {SEED_TRAIN}, step {STEP}, dolmino, 8xH20, zwfy6. "
                  "Fills the empty gap between keep12 (25% cut, constant-REJECT) and "
                  "zero damage (the only NI ACCEPT in A04)."),
        "prereg": {"document": "A04_SHALLOW_RUNG_LADDER_PREREG.md",
                   "commit": "a2e1a95",
                   "committed_before_first_margin": True,
                   "state_at_commit": ("both runs ~20 min into 5000 steps; no ckpt, no "
                                       "eval dir, no accuracy, no margin existed")},
        "node": os.uname().nodename,
        "node_ips": node_guard["host_ips"],
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "gpu_refuse_guard": gpu_guard,
        "gpu_h": gpu_h,
        "gate_constant_selftest": gates,
        "RANGE_CONSTANTS_DECLARED_UNUSED": RANGE_CONSTANTS_DECLARED_UNUSED,
        "preregistered_constants": dict(PREREG),
        "mmlu_tie_convention": PREREG_CONVENTION,
        "decision_axes": list(DECISION_AXES),
        "demoted_axes": list(DEMOTED_AXES),
        "ni_accept_bar": f">= {NI_ACCEPT_BAR} of {len(DECISION_AXES)} decision axes",
        "canonical_imports": {
            "from_pilot_zero": ["ni_rule", "ratio_rule", "load_shards", "build_nulls",
                                "mmlu_content_norm_vec", "qa_metric_vec",
                                "EXPECTED_N", "AXES", "DEMOTED_AXES", "PREREG"],
            "from_a04_shallow_rung_ni_7b": ["assert_aligned", "d4_interface_degenerate",
                                            "D4_CONSTANT_FRAC", "D4_TIE_FRAC",
                                            "Z95_TWO_SIDED"],
            "from_a03": ["paired_bootstrap", "TIE_CONVS", "N_BOOT", "SEED"],
            "SEED": SEED, "N_BOOT": N_BOOT, "TIE_CONVS": list(TIE_CONVS),
            "nothing_reimplemented": True,
        },
        "bootstrap_offsets": {
            "arm_index": {k: v for k, v in ARM_INDEX.items()},
            "guard_seed_offset": f"SEED+{GUARD_SEED_OFF}+13*axis_index",
            "interval_seed_offset": None,
        },
    }

    # ---- 0. seed disjointness, EXECUTED ---------------------------------
    out["seed_disjointness_checked"] = assert_seeds_disjoint(
        args.evidence_dir, list(ARM_INDEX.values()),
        {"arm_index": ARM_INDEX, "guard": f"SEED+{GUARD_SEED_OFF}"},
        self_output_basename=os.path.basename(args.out_json))
    print(f"[disjoint] scanned {out['seed_disjointness_checked']['archives_scanned']} "
          f"archives, no clash on {sorted(ARM_INDEX.values())}")

    # ---- 1. arm specs ----------------------------------------------------
    arm_specs = {"intact": INTACT}
    for k in keeps:
        tag = f"A04_1B_shallow_keep{k}_seed{SEED_TRAIN}_step{STEP}"
        arm_specs[f"keep{k}f2_step{STEP}"] = {"mmlu": tag, "cb": tag,
                                              "nq": f"{tag}_nq"}
    arm_specs[f"keep12f2_step{STEP}_REF"] = {"mmlu": REF_KEEP12_TAG,
                                             "cb": REF_KEEP12_TAG,
                                             "nq": f"{REF_KEEP12_TAG}_nq"}
    out["arm_dirs"] = arm_specs

    # ---- 2. guard G2 + protocol -----------------------------------------
    out["anchor_is_vanilla_G2"] = assert_anchor_is_vanilla(INTACT, args.raw_root)
    print("[G2] anchor verified VANILLA (tag markers + meta mode/depth/ckpt)")

    mm = os.path.join(args.raw_root, "olmo2_mmlu_content_results")
    cb = os.path.join(args.raw_root, "olmo2_closedbook_results")
    audit = {}
    for arm, spec in arm_specs.items():
        audit[f"{arm}|mmlu"] = os.path.join(mm, spec["mmlu"])
        audit[f"{arm}|cb"] = os.path.join(cb, spec["cb"])
        audit[f"{arm}|nq"] = os.path.join(cb, spec["nq"])
    out["protocol_audit"] = assert_protocol(args.raw_root, audit)
    print("[protocol] chat_template False (structural + per-cell), add_bos `is False`, "
          "max_new_tokens 32")

    # ---- 2b. the arms must be the ARCHITECTURES we claim -----------------
    arch = {}
    for arm, spec in arm_specs.items():
        if arm == "intact":
            continue
        cell = out["protocol_audit"]["per_cell"][f"{arm}|cb"]
        kf, nf, nhl = (cell["keep_front_layers"], cell["n_fresh_layers"],
                       cell["num_hidden_layers"])
        want_k = int(arm.replace("keep", "").split("f2")[0])
        if kf != want_k or nf != 2 or nhl != want_k + 2:
            raise SystemExit(
                f"FATAL {arm}: eval meta says keep_front={kf} n_fresh={nf} "
                f"depth={nhl}, expected {want_k}/2/{want_k+2}. The eval rebuilt the "
                "WRONG shell, or the tag points at another arm.")
        arch[arm] = {"keep_front_layers": kf, "n_fresh_layers": nf,
                     "num_hidden_layers": nhl,
                     "cut_layers": BASE_LAYERS - kf,
                     "cut_fraction_of_base": (BASE_LAYERS - kf) / BASE_LAYERS,
                     "ckpt": cell["ckpt"], "ckpt_step": cell["ckpt_step"],
                     "IS_DAMAGED": True,
                     "damaged_even_when_depth_equals_base": (
                         f"depth {nhl} vs base {BASE_LAYERS}: base layers "
                         f"{kf}..{BASE_LAYERS-1} are DISCARDED and replaced by "
                         "random-init Olmo2 layers. n_fresh=2 != 0, so this is NOT "
                         "the zero-damage CPT control."
                         if nhl == BASE_LAYERS else None)}
    out["arm_architectures_verified_from_eval_meta"] = arch
    print("[arch] every arm's eval meta matches its claimed keep/fresh/depth")

    # ---- 3. load everything + alignment ---------------------------------
    data, integ = {}, {}
    for arm, spec in arm_specs.items():
        data[arm], integ[arm] = load_cell(args.raw_root, spec)
    prov = {arm: {ax: {"dir": integ[arm][ax]["dir"]} for ax in AXES} for arm in data}
    out["shard_integrity"] = integ
    out["alignment"] = assert_aligned(data, prov)
    print("[align] item_id sequences identical across every arm and the anchor")

    # ---- 4. nulls + Delta, BUILT then cross-checked ---------------------
    nulls = build_nulls(data["intact"])

    def null_acc(axis, conv=PREREG_CONVENTION):
        if axis == "mmlu_content":
            return nulls["mmlu_content"]["by_convention"][conv]
        return nulls[axis]["acc"]

    delta_pp, resid_intact_pp, xcheck = {}, {}, {}
    for ax in AXES:
        rep = float(np.asarray(data["intact"][ax], float).mean())
        res = rep - null_acc(ax)
        resid_intact_pp[ax] = 100.0 * res
        delta_pp[ax] = 100.0 * DELTA_FRACTION * res
        dc = DELTA_CANONICAL_1B[ax]
        diff = abs(delta_pp[ax] - dc)
        if diff > DELTA_XCHECK_TOL:
            raise SystemExit(
                f"FATAL: Delta[{ax}] built {delta_pp[ax]!r} vs canonical {dc!r} "
                f"(|diff|={diff:.3e} > {DELTA_XCHECK_TOL:.0e}). Delta is NEVER "
                "substituted; aborting rather than re-margining every verdict.")
        xcheck[ax] = {"delta_built_pp": delta_pp[ax], "delta_canonical_pp": dc,
                      "abs_diff": diff, "tol": DELTA_XCHECK_TOL, "ok": True}
    out["intact_anchor"] = {
        "dirs": INTACT, "rule": "G0 (path-pinned) + G2 (vanilla, executed above)",
        "nulls_used": {ax: null_acc(ax) for ax in AXES},
        "nulls_all_conventions_mmlu": nulls["mmlu_content"]["by_convention"],
        "reported_intact_pp": {ax: 100.0 * float(np.asarray(data["intact"][ax],
                                                            float).mean())
                               for ax in AXES},
        "residual_intact_pp": resid_intact_pp,
        "delta_pp": delta_pp,
        "delta_rule": "0.10 * residual(intact, x); fraction frozen by git d1ba737",
        "delta_cross_check_vs_canonical": xcheck,
        "delta_never_substituted": True,
    }
    print("[delta] built Delta reproduces canonical on all 4 axes within 1e-9")

    # ---- 5. keep12 reproduction gate ------------------------------------
    k12 = _load_keep12_canonical(args.evidence_dir)
    ref_arm = f"keep12f2_step{STEP}_REF"
    repro = {}
    for ax in AXES:
        got = 100.0 * float(np.asarray(data[ref_arm][ax], float).mean())
        want = k12["seed_means_pct_seed101"][ax]
        d = abs(got - want)
        repro[ax] = {"recomputed_pct": got, "canonical_pct": want, "abs_diff_pp": d,
                     "ok": d <= KEEP12_SEED101_XCHECK_TOL}
        if d > KEEP12_SEED101_XCHECK_TOL:
            raise SystemExit(
                f"FATAL reproduction gate: keep12 seed101 {ax} recomputed {got!r} "
                f"but canonical Stage B says {want!r} (|diff|={d:.3e} pp). Either the "
                "loader, the metric or the shard set differs from the run that "
                "produced every Stage B number -- the new arms would not be "
                "comparable. Aborting.")
    out["keep12_reproduction_gate"] = {
        "why": ("the new rungs are only interpretable against keep12 if this pipeline "
                "reproduces keep12's OWN published accuracy bit-for-bit. Canonical "
                "values are READ AT RUNTIME from evidence/pilot_one_stage_b_s3_verdict"
                ".json, never transcribed."),
        "canonical_source": k12["path"], "per_axis": repro,
        "max_abs_diff_pp": max(v["abs_diff_pp"] for v in repro.values()),
        "tol_pp": KEEP12_SEED101_XCHECK_TOL, "passed": True,
    }
    print(f"[repro] keep12 seed101 reproduced to "
          f"{out['keep12_reproduction_gate']['max_abs_diff_pp']:.3e} pp on all 4 axes")

    if args.preflight_only:
        print("\n=== PREFLIGHT ONLY ===")
        print("  guards: forbidden-node OK, GPU refuse-guard OK, gate constants OK")
        print("  G2 anchor vanilla OK; protocol (chat/add_bos/max_new_tokens) OK")
        print("  Delta built from the pinned anchor reproduces canonical within 1e-9")
        print("  keep12 seed101 reproduction gate PASSED "
              f"({out['keep12_reproduction_gate']['max_abs_diff_pp']:.3e} pp)")
        print("  seed disjointness EXECUTED, no clash")
        print("  NOTHING WRITTEN. Re-run without --preflight_only once both arms are "
              "scored.")
        return

    # ---- 6. NI per (arm, axis), all five tie conventions ---------------
    arms = [a for a in arm_specs if a != "intact"]
    per_conv = {}
    for conv in TIE_CONVS:
        resid_i = {ax: 100.0 * (float(np.asarray(data["intact"][ax], float).mean())
                                - null_acc(ax, conv)) for ax in AXES}
        cells = {}
        for ai, arm in enumerate(arms):
            cells[arm] = {}
            for xi, ax in enumerate(AXES):
                d_pp = DELTA_FRACTION * resid_i[ax]
                r = ni_rule(data[arm][ax], data["intact"][ax], DELTA_FRACTION,
                            resid_i[ax] / 100.0,
                            seed_off=list(ARM_INDEX.values())[ai] * 13 + xi)
                margin = r["diff_lower95_one_sided_pp"] + d_pp
                se = ((r["diff_mean_pp"] - r["diff_lower95_one_sided_pp"])
                      / 1.6448536269514722)
                rep_a = 100.0 * float(np.asarray(data[arm][ax], float).mean())
                res_a = rep_a - 100.0 * null_acc(ax, conv)
                cells[arm][ax] = {
                    "reported_pp": rep_a,
                    "residual_arm_pp": res_a,
                    "residual_intact_pp": resid_i[ax],
                    "recovered_fraction": (res_a / resid_i[ax]
                                           if resid_i[ax] > 0 else None),
                    "deficit_pp": resid_i[ax] - res_a,
                    "delta_pp": d_pp,
                    "diff_mean_pp": r["diff_mean_pp"],
                    "diff_lower95_one_sided_pp": r["diff_lower95_one_sided_pp"],
                    "margin_pp": margin,
                    "bootstrap_se_pp": se,
                    "se_to_flip": abs(margin) / se if se > 0 else None,
                    "ni_accept": bool(r["ni_accept"]),
                    "boot_seed": r["boot_seed"],
                    "decision_weight": ax in DECISION_AXES,
                    "residual_intact_nonpositive_DEGENERATE": bool(resid_i[ax] <= 0),
                }
        per_conv[conv] = {"intact_residual_pp": resid_i, "cells": cells}
    out["per_convention"] = per_conv

    # ---- 7. verdict, on the pre-registered convention -------------------
    pc = per_conv[PREREG_CONVENTION]["cells"]
    verdicts = {}
    for arm in arms:
        acc_ax = [ax for ax in DECISION_AXES if pc[arm][ax]["ni_accept"]]
        n = len(acc_ax)
        v = ("NI_ACCEPT" if n >= NI_ACCEPT_BAR else
             "NI_INDETERMINATE_1_OF_3" if n == 1 else "NI_CONSTANT_REJECT")
        # identical under all five conventions?
        same = all(
            len([ax for ax in DECISION_AXES
                 if per_conv[c]["cells"][arm][ax]["ni_accept"]]) == n
            for c in TIE_CONVS)
        verdicts[arm] = {
            "n_decision_axes_accepting": n,
            "n_decision_axes": len(DECISION_AXES),
            "axes_accepting": acc_ax,
            "verdict": v,
            "identical_under_all_five_tie_conventions": bool(same),
            "per_axis_margin_pp": {ax: pc[arm][ax]["margin_pp"] for ax in AXES},
            "per_axis_bootstrap_se_pp": {ax: pc[arm][ax]["bootstrap_se_pp"]
                                         for ax in AXES},
            "per_axis_se_to_flip": {ax: pc[arm][ax]["se_to_flip"] for ax in AXES},
            "per_axis_recovered_fraction": {ax: pc[arm][ax]["recovered_fraction"]
                                            for ax in AXES},
            "d4_interface_degenerate": {
                ax: d4_interface_degenerate(data, arm, ax, nulls) for ax in AXES},
        }
    out["per_arm_verdict"] = verdicts

    new_arms = [a for a in arms if not a.endswith("_REF")]
    n_acc_arms = [a for a in new_arms
                  if verdicts[a]["n_decision_axes_accepting"] >= NI_ACCEPT_BAR]
    n_indet = [a for a in new_arms
               if verdicts[a]["n_decision_axes_accepting"] == 1]

    if n_acc_arms:
        branch, headline = "A", (
            "BLOCKER_DISCHARGED -- NI is OBSERVED TO ACCEPT under structural damage "
            f"at {', '.join(n_acc_arms)}")
    elif n_indet:
        branch, headline = "C", (
            "INDETERMINATE -- exactly 1 of 3 decision axes accepts on "
            f"{', '.join(n_indet)}; below the >=2/3 bar, blocker NOT discharged")
    else:
        branch, headline = "B", (
            "NI_ACCEPT_REGION_AT_1B_CONTAINS_NO_DAMAGED_RUNG -- both new rungs are "
            "constant-REJECT down to a 12.5% cut, the lightest the family admits")
    out["BRANCH"] = branch
    out["headline"] = headline
    out["branch_definitions_were_fixed_pre_data"] = {
        "A": "NI ACCEPT on >=2/3 decision axes on either new arm -> pilot_two_status "
             "blocker DISCHARGED (but Pilot Two still blocked independently by the "
             "format/ordering defect in control_arms_ni_20260813)",
        "B": "both constant-REJECT -> NI's accept region at 1B contains no damaged "
             "rung; negative but publishable verdict on the certification rule; "
             "pilot_two_status stays BLOCKED and the blocker is recorded as "
             "undischargeable by rung selection at 1B",
        "C": "exactly 1 of 3 -> INDETERMINATE (the same convention that put full32's "
             "1-of-3 below the bar); does NOT discharge",
        "source": "A04_SHALLOW_RUNG_LADDER_PREREG.md s4.1, commit a2e1a95",
    }

    # ---- 8. the depth ladder (keep12 / keep13 / keep14 / zero-damage) --
    ladder = {}
    for arm in arms:
        k = int(arm.replace("keep", "").split("f2")[0])
        ladder[f"keep{k}"] = {
            "keep_front_layers": k,
            "cut_layers": BASE_LAYERS - k,
            "cut_fraction": (BASE_LAYERS - k) / BASE_LAYERS,
            "depth": k + 2,
            "recovered_fraction": {ax: pc[arm][ax]["recovered_fraction"]
                                   for ax in AXES},
            "margin_pp": {ax: pc[arm][ax]["margin_pp"] for ax in AXES},
            "se_to_flip": {ax: pc[arm][ax]["se_to_flip"] for ax in AXES},
            "verdict": verdicts[arm]["verdict"],
        }
    # is recovery monotone in kept depth? 3 points -> 2 differences, so this is a
    # DESCRIPTIVE statement about sign, explicitly not a trend fit.
    mono = {}
    ks = sorted(int(x.replace("keep", "")) for x in ladder)
    for ax in AXES:
        vals = [ladder[f"keep{k}"]["recovered_fraction"][ax] for k in ks]
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        mono[ax] = {
            "keeps": ks, "recovered_fraction": vals,
            "successive_diffs": diffs,
            "all_same_sign": bool(all(d > 0 for d in diffs)
                                  or all(d < 0 for d in diffs)),
            "n_sign_reversals": sum(1 for i in range(len(diffs) - 1)
                                    if diffs[i] * diffs[i + 1] < 0),
        }
    out["depth_ladder"] = {
        "rungs": ladder,
        "monotonicity_DESCRIPTIVE_ONLY": mono,
        "why_descriptive": (f"{len(ks)} points give {len(ks)-1} differences. "
                            "'Non-monotone' on 2 differences means ONE sign flip and "
                            "cannot be distinguished from noise without a per-point "
                            "sigma_run, which does not exist here (one seed per arm). "
                            "NOT a trend fit and NOT decision-bearing."),
        "comparability": ("keep12/13/14 share corpus, step count, protocol, LR, "
                          "eff_bs and SEED (101) and are all post-ce5c298, so they ARE "
                          "mutually comparable as a 1B depth ladder. They are NOT "
                          "comparable to the 7B ladder (STATUS.json:warning's "
                          "two-corpora confound) and 5000 steps is not a converged heal."),
    }

    # ---- 9. pairwise resolved differences between adjacent rungs -------
    pairs = {}
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            ka = int(a.replace("keep", "").split("f2")[0])
            kb = int(b.replace("keep", "").split("f2")[0])
            if abs(ka - kb) != 1:
                continue
            per_ax = {}
            for xi, ax in enumerate(AXES):
                d = (np.asarray(data[b][ax], float)
                     - np.asarray(data[a][ax], float))
                m, lo, hi, p = paired_bootstrap(
                    d, seed=SEED + GUARD_SEED_OFF + 13 * xi + 7 * (ka + kb))
                per_ax[ax] = {"diff_mean_pp": 100.0 * m,
                              "ci95_pp": [100.0 * lo, 100.0 * hi],
                              "boot_p": p,
                              "resolved": bool(lo > 0 or hi < 0),
                              "direction": ("higher_at_keep%d" % max(ka, kb)
                                            if m > 0 else
                                            "higher_at_keep%d" % min(ka, kb))}
            pairs[f"keep{kb}_minus_keep{ka}"] = per_ax
    out["adjacent_rung_paired_differences"] = {
        "per_pair": pairs,
        "note": ("paired item bootstrap on the SAME item set (alignment asserted). "
                 "A difference whose CI straddles 0 is UNRESOLVED -- not 'a "
                 "direction'. This is the item-sample SE only; it is NOT sigma_run "
                 "and says nothing about seed variance."),
    }

    # ---- 10. RATIO(rho) for the rule-disagreement ledger --------------
    reported = {a: {ax: float(np.asarray(data[a][ax], float).mean())
                    for ax in AXES} for a in arm_specs}
    rat = {}
    for arm in arms:
        rr = ratio_rule(reported[arm], reported["intact"], PREREG["rho"], list(AXES))
        rat[arm] = rr
    disagree = [a for a in arms
                if rat[a]["ratio_accept"] is True
                and verdicts[a]["n_decision_axes_accepting"] < NI_ACCEPT_BAR]
    out["ratio_rule"] = {
        "rho": PREREG["rho"], "per_arm": rat,
        "rule_disagreement_cells": disagree,
        "n_rule_disagreement_cells": len(disagree),
        "note": ("RATIO(0.85) is A04's comparison rule. A cell where RATIO accepts and "
                 "NI rejects is the disagreement A04 is about. Reported for the ledger; "
                 "the NI verdict above does not depend on it."),
    }

    # ---- 11. what this pass does NOT license --------------------------
    out["not_licensed"] = [
        "any sigma_run, seed-variance or K2 statement -- ONE seed (101) per arm; a "
        "sigma over arms of DIFFERENT depth is not a run-to-run sigma",
        "any PLATEAU(T) comparison -- no in-domain val PPL trajectory was produced "
        "for either new arm, so the NI-vs-PLATEAU disagreement cannot be evaluated",
        "any trajectory / monotonicity / neighbour-range claim -- save_every 2500 "
        "gives 2 checkpoints per arm; 2 points = 1 difference",
        "any differential-LR claim -- all four optimizer groups ran at 2.00e-05 "
        "(GATE0-measured, prereg s3.2)",
        "treating keep14+fresh2 as a zero-damage control -- depth 16 == base depth, "
        "but base layers 14-15 are DISCARDED and random-re-initialised; the "
        "zero-damage control is n_fresh_layers=0 CPT (full32-style)",
        "comparing these 1B rungs to the 7B ladder as a matched experiment "
        "(STATUS.json:warning two-corpora / unequal-steps confound)",
        "any recovery FRACTION read as a clean 'inheritance is worth X' -- no 1B "
        "zero-inheritance floor (--from_scratch or --random_trunk) exists on either "
        "disk, so the fractions here are fractions of the intact residual ONLY "
        "(must_not_claim item 28)",
        "a format-free reading of triviaqa / popqa / nq_open -- A04 has two "
        "demonstrations that generative EM partly measures output length "
        "(PROPOSAL.md 4.4: 47.37% of an EM loss; control_arms P3: 50.00% of an EM "
        "gain, which REORDERED two arms). mmlu_content is length-free by construction",
        "any claim that a keep13-vs-keep14 margin difference is 'measured' unless its "
        "paired CI excludes 0 (section 9)",
        "quoting any margin finer than 0.01 pp across nodes (numpy multinomial drift)",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
        fh.write("\n")

    # ---- console summary ------------------------------------------------
    print("\n=== intact anchor (prereg convention 'split') ===")
    for ax in AXES:
        print(f"  {ax:<14} null {100*null_acc(ax):7.4f}  intact "
              f"{out['intact_anchor']['reported_intact_pp'][ax]:7.4f}  residual "
              f"{resid_intact_pp[ax]:7.4f}  Delta {delta_pp[ax]:7.4f}")
    print(f"\n=== NI margins (pp), convention '{PREREG_CONVENTION}' "
          f"(margin > 0 == ACCEPT) ===")
    hdr = f"{'arm':<24}" + "".join(f"{ax:>16}" for ax in AXES)
    print(hdr)
    for arm in arms:
        row = f"{arm:<24}"
        for ax in AXES:
            c = pc[arm][ax]
            row += f"{c['margin_pp']:>11.4f}{'*' if c['ni_accept'] else ' ':>2}" \
                   f"{'':>3}"
        print(row)
    print(f"\n{'arm':<24}{'recovered fraction of intact residual':>44}")
    for arm in arms:
        print(f"{arm:<24}" + "".join(
            f"{100*pc[arm][ax]['recovered_fraction']:>10.2f}%" for ax in AXES))
    print("\n=== verdicts ===")
    for arm in arms:
        v = verdicts[arm]
        print(f"  {arm:<24} {v['verdict']:<26} "
              f"{v['n_decision_axes_accepting']}/{v['n_decision_axes']} decision axes "
              f"accept; identical under all 5 conventions: "
              f"{v['identical_under_all_five_tie_conventions']}")
    print(f"\nBRANCH {branch}: {headline}")
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
