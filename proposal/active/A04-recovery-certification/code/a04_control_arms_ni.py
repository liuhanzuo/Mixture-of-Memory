#!/usr/bin/env python3
"""A04 — 4-axis NI on the two NEVER-TESTED control arms at keep_front=14.

WHAT IS UNDER TEST, AND WHY IT IS NOT ANOTHER RUNG
--------------------------------------------------
A04's problem, as of 2026-08-13, is NOT variance: it is that **no damaged arm has
ever accepted**. keep8 / keep14 / shortgpt16 x 15 checkpoints x 5 tie conventions
produced ZERO accepts; the only arm that ever accepted is `full32`, which has
ZERO damage. `PROPOSAL.md` 8 item 2 states the consequence: "this is a
rung-selection problem, not a variance problem -- more seeds on either family
cannot fix it." Shallower rungs (keep16/20/24/28 + fresh2) DO NOT EXIST on either
disk (0 ckpts), so the only way to widen the rung set without training is to vary
something OTHER than depth.

These two arms vary the REPAIR MODE at a FIXED depth (keep_front=14, n_fresh=2,
16 layers), and neither has ever entered an A04 evidence file:

  * `olmo2_probe2_7B_keep14fresh2_freezefront`  (FF) -- `--freeze_front`:
    front 14 inherited layers FROZEN (`apply_freeze_front`, trainer line 397,
    sets requires_grad_(False) on exactly model.layers.{0..13}.*), so only the
    2 fresh layers + embed + model.norm + lm_head train. 1226.9M of 4060.4M
    trainable. SAME injury as train-all, STRICTLY LESS repair capacity.
  * `olmo2_probe2_7B_keep14fresh2_fromscratch` (FS) -- `--from_scratch`:
    base weights IGNORED, all 16 layers random-init. SAME architecture, NO
    inheritance. This is `A04_GATE_DESIGN.md` 3.2's arm A4, "the 'did
    inheritance matter at all' floor".

They are the A2/A3/A4-adjacent controls the gate design has always presumed and
never measured.

PRE-REGISTERED, in `A04_CONTROL_ARMS_NI_PREREG.md`, committed BEFORE the first
margin existed (commit e51f390):

  P1  margin_pp(FF) <= margin_pp(train-all) on EVERY decision axis.
      HOLDS 3/3; HOLDS_WEAK 2/3 with the violation inside its own bootstrap SE;
      VIOLATED if >=1 axis has FF > train-all by MORE than that cell's SE.
  P2  FS is the lowest-margin arm on >=2/3 decision axes AND rejects 3/3.
  P3  THE FALSIFIER. If P1 is VIOLATED, "train-all is the stronger repair" is
      FALSE, the `A04_GATE_DESIGN.md` 3.2 arm ordering (A1>A2>A3>A4) becomes an
      UNTESTED ASSUMPTION, and a rung is `(depth, repair mode)` not `depth`.
      Registered consequence: the route to an accept might be training the
      inherited weights LESS, which is the opposite of "heal longer" (already
      falsified on both a damaged and an undamaged arm).

  A non-violation is NOT permitted to be spun as "nothing to see". It confirms an
  ordering the gate assumes but never measured, and it closes P3's alternative
  route at this depth. Both directions are written down in advance.

WHAT IS IMPORTED AND NEVER REIMPLEMENTED
----------------------------------------
`ni_rule`, `ratio_rule`, `build_nulls`, `load_shards`, `mmlu_content_norm_vec`,
`qa_metric_vec`, `EXPECTED_N`, `AXES`, `DEMOTED_AXES`, `PREREG` from
`pilot_zero_rule_disagreement`; `paired_bootstrap`, `TIE_CONVS`, `N_BOOT`, `SEED`
from A03's `analyze_1b_knowledge_floor`; `ANCHOR`, `_load_arm`, `assert_aligned`,
`d4_interface_degenerate`, `D2_RESIDUAL_FLOOR_PP`, `Z95_TWO_SIDED` from
`a04_shallow_rung_ni_7b`; `shard_integrity_report` from
`a04_neighbour_variability`.

  ! `build_nulls` IS IMPORTED AND CALLED. No margin in this file is obtained by
    subtracting a recorded null from a recorded accuracy. That shortcut produced
    FOUR wrong numbers on 2026-08-13 alone (worst case a 3.0x underestimate, and
    once a 5-point range reported as a pair range), which is why `PROPOSAL.md` 4
    now pins canonical JSON above prose.

  ! `protocol_asserted` from `a04_neighbour_variability` is NOT imported: it
    greps for a `DRIVER START ... mmlu_bs=.. cb_bs=..` line emitted by the 2026-08-13
    drivers. These arms were scored 2026-08-02 by DIFFERENT launchers
    (`p06_run_104_transferred.sh` / `p06_run_transferred.sh` / `run_cb_drv.sh`)
    which echo a different format. Reusing it would either crash or, worse, be
    "fixed" by loosening the regex. A DEDICATED fail-closed asserter is written
    below against THOSE drivers' actual echoed lines, and the frozen expectation
    {cb_bs: 32, mmlu_bs: 16} is IDENTICAL to the imported one's.

WHAT THIS FILE MAY NOT CONCLUDE (registered before the fact)
------------------------------------------------------------
  * NOT a sigma_run measurement. One seed per arm; no 7B sd_run exists or is
    reconstructible (`PROPOSAL.md` 7.2, `must_not_claim[23]`).
  * NOT a rung of the keepN depth ladder. All three arms are keep_front=14.
  * NOT an answer to Q3's PLATEAU half. `olmo2_ppl_results/` has NO freezefront
    or fromscratch directory, so `PLATEAU(T)` is not computable for these arms
    and only `RATIO(0.85)` can be compared. Stated as a limitation, not
    worked around.
  * FS's floor is CONFOUNDED BY LR: `_classify_param` returns "fresh" FIRST
    under from_scratch, so FS ran uniform 1e-4 while train-all and FF both ran
    uniform 2e-5. Read from the authoritative wzc1 `[optim] group` lines, not
    assumed. train-all vs FF (P1) IS LR-matched; P2 is not.

THE step23500 CHECKPOINT IS DROPPED, NOT DEMOTED
------------------------------------------------
`outputs/..._freezefront/step23500.pt` exists on zwfy6 and looks like a far
neighbour of step200000. It is NOT on the same trajectory. The zwfy6 copy of
`logs/olmo2_7B_keep14fresh2_freezefront.log` (162,067 B) documents a DIFFERENT,
ABANDONED run from the wzc1 copy (1,368,257 B):

    zwfy6 log: first banner 2026-07-21 02:02:20, bs=4 gaccum=4,
               dataset rows=15,491,607, dies at step 23,640
    wzc1  log: first banner 2026-07-25 12:15:48, bs=16 gaccum=1,
               dataset rows=7,570,911,  reaches step 200,000 + final.pt

`step23500.pt`'s mtime (2026-07-23 13:45:20.774755372) matches the ABANDONED
run's save line to the nanosecond; the wzc1 run's own step23500 (07-25 22:40:21)
was rotated away. Scoring it against step200000 would silently cross two corpora
AND two micro-batch geometries. Bootstrap offset 802 is reserved and left unused.
Consequence: the only 2.0.2-compliant neighbour statement for these arms is
"no adjacent saved checkpoint exists", which 2.0.2 explicitly permits.

BOOTSTRAP SEEDS. arm_index 800 (FF) / 801 (FS); train-all endpoint keeps its
ARCHIVED 201 so the reproduction assert is meaningful. Guard SEED+5700,
intervals unused (no intervals here). Mechanically intersected against every
archived block: 0-1, 100-102, 200-203, 300-301, 400-408, 500-503, 700-702.

CPU ONLY. Read-only on every input.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SHARED_CODE = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "shared", "code"))
if _SHARED_CODE not in sys.path:
    sys.path.insert(0, _SHARED_CODE)

from pilot_zero_rule_disagreement import (  # noqa: E402
    AXES,
    DEMOTED_AXES,
    EXPECTED_N,
    PREREG,
    build_nulls,
    ni_rule,
    ratio_rule,
)
from proposal_paths import a03_code_dir  # noqa: E402

_A03_CODE = a03_code_dir()
if _A03_CODE not in sys.path:
    sys.path.insert(0, _A03_CODE)
from analyze_1b_knowledge_floor import (  # noqa: E402
    N_BOOT,
    SEED,
    TIE_CONVS,
    paired_bootstrap,
)

from a04_shallow_rung_ni_7b import (  # noqa: E402
    D2_RESIDUAL_FLOOR_PP,
    Z95_TWO_SIDED,
    ANCHOR,
    _load_arm,
    assert_aligned,
    d4_interface_degenerate,
)
from a04_neighbour_variability import shard_integrity_report  # noqa: E402

DECISION_AXES = [x for x in AXES if x not in DEMOTED_AXES]

# ---------------------------------------------------------------------------
# The archived train-all endpoint margins, READ FROM THE CANONICAL JSON at run
# time -- deliberately NOT hardcoded.
#
# WHY THIS IS A FUNCTION AND NOT A DICT OF LITERALS. The first version of this
# file hardcoded three literals transcribed from
# `A04_KEEP14_TRAJECTORY_NI_VERDICT.md` 2's table (-28.4624 / -15.0810 /
# -7.4749) padded out with invented trailing digits. The reproduction gate below
# then FIRED -- correctly -- on triviaqa at 8.82e-05 pp. The recomputation was
# right and MY CONSTANTS WERE WRONG: canonical is -28.462438698172093, and the
# 4-dp table value rounds to the same -28.4624 while differing in the 5th
# decimal, which is 1.8x the 5e-5 pp tolerance.
#
# That is exactly the failure `PROPOSAL.md` 4 was rewritten to prevent ("all
# numbers are from the canonical evidence JSONs, NOT from prose in the verdict
# .md's ... canonical JSON always wins"), and it is the FIFTH instance of the
# same class on 2026-08-13. Reading the JSON removes the transcription step
# entirely, so the tolerance now tests the IMPORTED RULE OBJECTS -- which is all
# it was ever meant to test.
# ---------------------------------------------------------------------------
ARCHIVE_JSON = "evidence/a04_shallow_rung_ni_7b.json"
ARCHIVE_ARM = "keep14fresh2_step200k"
REPRO_TOL_PP = 5e-5


def load_archived_train_all_margins(a04_dir):
    """Full-precision archived margins, straight from the canonical JSON."""
    p = os.path.join(a04_dir, ARCHIVE_JSON)
    if not os.path.isfile(p):
        raise SystemExit(
            f"FATAL: canonical archive {p} absent. The reproduction gate cannot "
            "be evaluated, so the new arms cannot be shown to be on the same "
            "scale as the endpoint they are compared to. Refusing to publish.")
    arch = json.load(open(p))
    out = {}
    for c in arch["per_convention"]["split"]["cells"]:
        if c["arm"] == ARCHIVE_ARM and c["axis"] in DECISION_AXES:
            out[c["axis"]] = {
                "margin_pp": float(c["margin_pp"]),
                "lo95_pp": float(c["diff_lower95_one_sided_pp"]),
                "delta_pp": float(c["delta_pp"]),
                "boot_seed": c.get("boot_seed"),
            }
    missing = [x for x in DECISION_AXES if x not in out]
    if missing:
        raise SystemExit(
            f"FATAL: archive {p} has no `{ARCHIVE_ARM}` cell for {missing}")
    return {"source": ARCHIVE_JSON, "arm": ARCHIVE_ARM,
            "read_at_runtime_not_hardcoded": True, "per_axis": out}

ARM_INDEX = {
    "keep14fresh2_step200k": 201,      # ARCHIVED offset, deliberately reused
    "freezefront_step200k": 800,
    "fromscratch_step200k": 801,
}
RESERVED_UNUSED = {
    802: ("reserved for freezefront@step23500 and DELIBERATELY LEFT UNUSED -- "
          "that checkpoint belongs to an ABANDONED run on the OTHER corpus "
          "(15,491,607 rows, bs=4 gaccum=4) and is not on step200000's "
          "trajectory. See the module docstring.")
}
ARCHIVED_OFFSET_BLOCKS = {
    "pilot_zero": [0, 1],
    "step100k": [100, 101, 102],
    "shallow_rung": [200, 201, 202, 203],
    "keep14_trajectory": [300, 301],
    "neighbour_variability": [400, 401, 402, 403, 404, 405, 406, 407, 408],
    "full32_trajectory": [500, 501, 502, 503],
    "keep10_neighbour_range": [700, 701, 702],
}

# Read from the AUTHORITATIVE wzc1 training logs before any scoring. Every value
# here is a grep result, not a recollection.
ARM_PROVENANCE = {
    "keep14fresh2_step200k": {
        "label": "train-all (the canonical A1 construction)",
        "arch_meta_arm": "healing_front14+fresh2",
        "trainer_flags": "--keep_front_layers 14 --n_fresh_layers 2",
        "authoritative_train_log": "wzc1:logs/olmo2_7B_keep14fresh2.log",
        "log_bytes": 1321311,
        "first_banner": "2026-07-16 21:36:20",
        "geometry": "bs=16 gaccum=1 eff_bs=128 seq_len=2048",
        "dataset_rows": 7570911,
        "max_steps": 200000, "reached": 200000, "n_resume_banners": 4,
        "optim_groups": ["inh_decay 4060.1M @2.00e-05",
                         "inh_nodecay 0.3M @2.00e-05"],
        "effective_lr": "uniform 2e-05",
        "n_trainable": 4060352512, "n_params": 4060352512,
        "ckpt_wzc1_bytes": 48724467827,
        "ckpt_zwfy6_bytes": 16241486089,
    },
    "freezefront_step200k": {
        "label": "FF -- same damage, STRICTLY LESS repair capacity",
        "arch_meta_arm": "frozen_front14+fresh2",
        "trainer_flags": "--keep_front_layers 14 --n_fresh_layers 2 --freeze_front",
        "authoritative_train_log": "wzc1:logs/olmo2_7B_keep14fresh2_freezefront.log",
        "log_bytes": 1368257,
        "first_banner": "2026-07-25 12:15:48",
        "geometry": "bs=16 gaccum=1 eff_bs=128 seq_len=2048",
        "dataset_rows": 7570911,
        "max_steps": 200000, "reached": 200000, "n_resume_banners": 1,
        "optim_groups": ["inh_decay 1226.8M @2.00e-05",
                         "inh_nodecay 0.0M @2.00e-05"],
        "effective_lr": "uniform 2e-05",
        "n_trainable": 1226870784, "n_params": 4060352512,
        "frozen_set": ("apply_freeze_front (trainer:397) sets "
                       "requires_grad_(False) on exactly model.layers.{0..13}.*; "
                       "fresh tail 14/15 + embed_tokens + model.norm + lm_head "
                       "stay trainable"),
        "ckpt_wzc1_bytes": 26056479363,
        "ckpt_zwfy6_bytes": 16241487014,
        "why_wzc1_ckpt_is_smaller": (
            "26.06 GB vs 48.72 GB is an OPTIMIZER-STATE artefact: AdamW carries "
            "exp_avg + exp_avg_sq only for the 1226.9M TRAINABLE params. NOT a "
            "different architecture -- the zwfy6 slim copy loads 179 tensors, "
            "strict, num_hidden_layers=16, identical to the other two arms."),
        "zwfy6_log_is_a_DIFFERENT_run": {
            "zwfy6_log_bytes": 162067,
            "zwfy6_first_banner": "2026-07-21 02:02:20",
            "zwfy6_geometry": "bs=4 gaccum=4 eff_bs=128",
            "zwfy6_dataset_rows": 15491607,
            "zwfy6_last_step": 23640,
            "zwfy6_last_save": "step23500.pt @ 2026-07-23 13:45:20",
            "consequence": ("outputs/..._freezefront/step23500.pt (mtime "
                            "2026-07-23 13:45:20.774755372) belongs to THAT "
                            "abandoned run, on the OTHER corpus, at a different "
                            "micro-batch geometry. DROPPED from this analysis. "
                            "The wzc1 run's own step23500 (07-25 22:40:21) was "
                            "rotated away."),
        },
    },
    "fromscratch_step200k": {
        "label": "FS -- same architecture, ZERO inheritance (gate design arm A4)",
        "arch_meta_arm": "scratch16L",
        "trainer_flags": "--from_scratch (depth 16)",
        "authoritative_train_log": "wzc1:logs/olmo2_7B_keep14fresh2_fromscratch.log",
        "log_bytes": 1348814,
        "first_banner": "2026-07-21 02:00:06",
        "geometry": "bs=16 gaccum=1 eff_bs=128 seq_len=2048",
        "dataset_rows": 7570911,
        "max_steps": 200000, "reached": 200000, "n_resume_banners": 0,
        "optim_groups": ["fresh_decay 4060.1M @1.00e-04",
                         "fresh_nodecay 0.3M @1.00e-04"],
        "effective_lr": "uniform 1e-04  <-- 5x the other two arms",
        "n_trainable": 4060352512, "n_params": 4060352512,
        "lr_confound": ("_classify_param (trainer:436) returns 'fresh' FIRST "
                        "when from_scratch, so every param landed in the fresh "
                        "group at lr_fresh=1e-4 while train-all and FF both ran "
                        "uniform 2e-5. FS is therefore a floor anchor WITH a "
                        "5x-LR caveat, NOT a clean isolation of inheritance."),
        "ckpt_wzc1_bytes": 48724467699,
        "ckpt_zwfy6_bytes": 16241486829,
    },
}

MATCHED_ACROSS_ALL_THREE = {
    "dataset_rows": 7570911,
    "eff_bs": 128, "seq_len": 2048, "tokens_per_step": 262144,
    "max_steps": 200000, "steps_reached": 200000,
    "disk_of_training": "wzc1", "optimizer": "fp32 AdamW",
    "base_model": "models/OLMo-2-1124-7B",
    "depth": 16, "keep_front_layers": 14, "n_fresh_layers": 2,
    "NOT_matched": ["effective LR (2e-5 / 2e-5 / 1e-4)",
                    "n_trainable (4060.4M / 1226.9M / 4060.4M)",
                    "n_resume_banners (4 / 1 / 0)"],
    "why_this_matters": (
        "STATUS.json:warning's two-corpora confound (7,570,911 wzc1 vs "
        "15,491,607 zwfy6, ratio 2.0462x) is a DEPTH-LADDER confound. These "
        "three arms are ONE depth on ONE corpus with ONE step count, so the "
        "repair-mode contrast is cleaner than any depth contrast in the repo. "
        "Registered in the prereg BEFORE scoring so it cannot be claimed as a "
        "post-hoc discovery."),
}

SAMPLER_REGIME = {
    "all_three_arms": "pre_ce5c298",
    "fix_commit": "ce5c298 (2026-08-09 23:21:09 +0800)",
    "launch_dates": {"train-all": "2026-07-16", "FS": "2026-07-21",
                     "FF": "2026-07-25"},
    "consequence": (
        "All three predate the DistributedSampler seed fix, so --seed moved only "
        "the fresh-tail init and data order was byte-identical across seeds. "
        "PROPOSAL.md 7.2's BINDING rule (pre-fix and post-fix runs may NOT be "
        "pooled into one sigma_run) is SATISFIED here trivially -- all three are "
        "on the SAME side of the break, so they are mutually comparable -- and "
        "none of them may enter any sigma_run at all."),
}

PREREG_DOC = "A04_CONTROL_ARMS_NI_PREREG.md"
PREREG_COMMIT = "e51f390"
PREREG_PREDICTIONS = {
    "P1": ("margin_pp(FF) <= margin_pp(train-all) on EVERY decision axis; same "
           "injury + strictly less repair capacity + LR-matched at 2e-5"),
    "P1_verdicts": {
        "P1_HOLDS": "3 of 3 decision axes satisfy FF <= train-all",
        "P1_HOLDS_WEAK": ("2 of 3, and the single violation is smaller than that "
                          "cell's own bootstrap SE"),
        "P1_VIOLATED": (">=1 decision axis where FF EXCEEDS train-all by MORE "
                        "than that cell's bootstrap SE"),
    },
    "P2": ("FS has the lowest margin of the three arms on >=2 of 3 decision "
           "axes AND rejects on 3 of 3"),
    "P3": ("THE FALSIFIER: if P1 is VIOLATED then 'train-all is the stronger "
           "repair' is FALSE; A04_GATE_DESIGN 3.2's arm ordering A1>A2>A3>A4 "
           "becomes an UNTESTED ASSUMPTION; a rung is (depth, repair mode) not "
           "depth; and the route to an accept may be training the inherited "
           "weights LESS -- the opposite of 'heal longer'"),
    "asymmetry_fixed_in_advance": (
        "A non-violation may NOT be reported as 'no finding'. It confirms an "
        "ordering the gate design assumes but had never measured, and closes "
        "P3's alternative route at this damage depth."),
}


def protocol_asserted_2026_08_02_drivers(raw_root, arm_specs):
    """FAIL-CLOSED protocol assertion against the 2026-08-02 launchers.

    NOT the imported `a04_neighbour_variability.protocol_asserted`: that one
    greps a `DRIVER START ... mmlu_bs=.. cb_bs=..` header emitted by the
    2026-08-13 drivers, which these 2026-08-02 runs never wrote. Loosening that
    regex to fit would be the wrong repair -- so this asserter is written against
    the lines the ACTUAL launchers echoed, with the SAME frozen expectation.

    Evidence (the launcher's own echoed line, not this file's source):
      logs/cb_driver_104.out   `START freezefront_step200k ... bs=32 tasks=popqa,triviaqa`
                               `START fromscratch_step200k ... bs=32 ...`
      logs/cb_driver_73.out    `START keep14_step200k ... bs=32 ...` (the endpoint
                               this comparison is anchored to)
      logs/nqopen_driver_104.log  `START freezefront_step200k_nqopen ... bs=32`
      logs/nqopen_scratch.log     `START fromscratch_step200k_nqopen ... bs=32`
    MMLU echoes no bs, so its 16 is established from the launcher SOURCE:
      scripts/p06_run_104_transferred.sh (FF) and p06_run_transferred.sh (FS)
      both leave BS unset -> scripts/_run_olmo2_mmlu_content.sh:43 `BS="${BS:-16}"`.

    add_bos is asserted with `is False` -- NEVER `is not True`, which passes
    silently on None. chat_template is STRUCTURAL: neither harness has a
    chat-template code path.
    """
    frozen = {"cb_bs": 32, "mmlu_bs": 16}
    out = {
        "asserter": "dedicated, written against the 2026-08-02 launchers",
        "why_not_the_imported_one": (
            "a04_neighbour_variability.protocol_asserted requires a "
            "'DRIVER START ... mmlu_bs=.. cb_bs=..' header that only the "
            "2026-08-13 drivers emit. These cells were scored 2026-08-02 by "
            "p06_run_*_transferred.sh + a cb driver with a different echo "
            "format. Reusing it would crash, and 'fixing' it by loosening the "
            "regex would weaken the gate for every future caller."),
        "frozen_expectation": frozen,
        "same_frozen_values_as_the_imported_asserter": True,
        "why_bs_is_not_free": (
            "full32_rescore_v2_20260812.sensitivity_bs48_probe: bs32->bs48 "
            "flipped 12/14267 popqa and 10/3610 nq_open items"),
        "artefact_gap_acknowledged": (
            "summary.json:meta records neither batch_size nor chat_template "
            "(A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md), so both are confirmed "
            "from the INVOCATION"),
        "cb_bs_from_launcher_logs": {},
        "mmlu_bs_from_launcher_source": {},
        "add_bos_from_summaries": {},
        "max_new_tokens_from_summaries": {},
        "harness_md5": {},
        "chat_template": {},
    }

    # ---- cb / nq batch size, from the launchers' echoed START lines ----------
    cb_expect = {
        "freezefront_step200k": ("logs/cb_driver_104.out", "freezefront_step200k"),
        "fromscratch_step200k": ("logs/cb_driver_104.out", "fromscratch_step200k"),
        "keep14fresh2_step200k": ("logs/cb_driver_73.out", "keep14_step200k"),
        "freezefront_step200k_nqopen": ("logs/nqopen_driver_104.log",
                                        "freezefront_step200k_nqopen"),
        "fromscratch_step200k_nqopen": ("logs/nqopen_scratch.log",
                                        "fromscratch_step200k_nqopen"),
        "base_full": ("logs/cb_driver_73.out", "base_full"),
        "base_full_nqopen": ("logs/cb_base_full_nqopen_shard0.out", None),
    }
    for label, (lg, model) in cb_expect.items():
        p = os.path.join(raw_root, lg)
        if not os.path.isfile(p):
            raise SystemExit(
                f"FATAL: launcher log {p} absent -- batch size cannot be "
                "confirmed from the invocation, and summary.json does not "
                "record it. Refusing to publish cells whose protocol cannot be "
                "established.")
        txt = open(p).read()
        if model is None:
            out["cb_bs_from_launcher_logs"][label] = {
                "log": lg, "bs": None,
                "note": ("per-shard log, no START line; its bs is covered by "
                         "the nqopen driver for the same batch of models"),
            }
            continue
        m = re.search(rf"START {re.escape(model)}\b.*?\bbs=(\d+)", txt)
        if not m:
            raise SystemExit(
                f"FATAL: no 'START {model} ... bs=..' line in {p}")
        bs = int(m.group(1))
        if bs != frozen["cb_bs"]:
            raise SystemExit(
                f"FATAL protocol deviation in {p}: {model} ran at bs={bs}, "
                f"frozen value is {frozen['cb_bs']}")
        out["cb_bs_from_launcher_logs"][label] = {
            "log": lg, "model": model, "bs": bs,
            "evidence_line": m.group(0)[:160]}

    # ---- mmlu batch size, from the launcher SOURCE (it echoes no bs) --------
    mm_src = {
        "freezefront_step200k": "scripts/p06_run_104_transferred.sh",
        "fromscratch_step200k": "scripts/p06_run_transferred.sh",
        "keep14fresh2_step200k": "scripts/p06_run_transferred.sh",
    }
    runner = os.path.join(raw_root, "scripts/_run_olmo2_mmlu_content.sh")
    if not os.path.isfile(runner):
        raise SystemExit(f"FATAL: {runner} absent")
    rsrc = open(runner).read()
    mm = re.search(r'^BS="\$\{BS:-(\d+)\}"', rsrc, re.M)
    if not mm:
        raise SystemExit(
            f"FATAL: no 'BS=\"${{BS:-N}}\"' default in {runner}")
    default_bs = int(mm.group(1))
    if default_bs != frozen["mmlu_bs"]:
        raise SystemExit(
            f"FATAL: mmlu runner default BS={default_bs} != {frozen['mmlu_bs']}")
    for label, src in mm_src.items():
        p = os.path.join(raw_root, src)
        if not os.path.isfile(p):
            raise SystemExit(f"FATAL: mmlu launcher {p} absent")
        s = open(p).read()
        if re.search(r"^\s*(export\s+)?BS=", s, re.M):
            raise SystemExit(
                f"FATAL: {src} sets BS explicitly; the 16 in this record was "
                "derived from the runner default and would be wrong")
        out["mmlu_bs_from_launcher_source"][label] = {
            "launcher": src, "sets_BS": False,
            "runner": "scripts/_run_olmo2_mmlu_content.sh",
            "runner_default_BS": default_bs,
            "effective_bs": default_bs}

    # ---- add_bos / max_new_tokens, from the artefacts ----------------------
    for label, spec in arm_specs.items():
        for key, root in (("cb", "olmo2_closedbook_results"),
                          ("nq", "olmo2_closedbook_results"),
                          ("mmlu", "olmo2_mmlu_content_results")):
            if not spec.get(key):
                continue
            sp = os.path.join(raw_root, root, spec[key], "summary.json")
            if not os.path.isfile(sp):
                raise SystemExit(f"FATAL: {sp} absent")
            meta = json.load(open(sp)).get("meta", {})
            ab = meta["add_bos"]                 # KeyError = loud, desired
            if ab is not False:                  # `is False`, never `is not True`
                raise SystemExit(
                    f"FATAL {sp}: add_bos={ab!r}; base protocol requires False. "
                    "(Asserted with `is False`; `is not True` passes on None.)")
            out["add_bos_from_summaries"][f"{label}|{key}"] = False
            if key != "mmlu":
                mnt = meta["max_new_tokens"]
                if int(mnt) != 32:
                    raise SystemExit(f"FATAL {sp}: max_new_tokens={mnt!r} != 32")
                out["max_new_tokens_from_summaries"][f"{label}|{key}"] = int(mnt)

    # ---- harness identity: same code that produced anchor AND endpoint -----
    import hashlib
    for rel, want in (("scripts/eval_olmo2_closedbook_qa.py",
                       "2ed41993241226c795a3ca38375933f7"),
                      ("scripts/eval_olmo2_mmlu_content.py",
                       "fe4a62dbdf884a1e2aedc6ed26887b4e")):
        p = os.path.join(raw_root, rel)
        if not os.path.isfile(p):
            raise SystemExit(f"FATAL: harness {p} absent")
        got = hashlib.md5(open(p, "rb").read()).hexdigest()
        if got != want:
            raise SystemExit(
                f"FATAL harness drift {rel}: md5={got} != pinned {want}. The "
                "pinned value is the one A04_KEEP14_TRAJECTORY_NI_VERDICT 5.1 "
                "item 5 records for the copies that produced the anchor and the "
                "endpoint.")
        out["harness_md5"][rel] = {"md5": got, "pinned": want, "match": True}

    out["chat_template"] = {
        "value": False,
        "how_established": (
            "STRUCTURAL, not a flag: neither scripts/eval_olmo2_closedbook_qa.py "
            "nor scripts/eval_olmo2_mmlu_content.py contains a chat-template "
            "code path. A protocol that cannot be switched on cannot have been "
            "switched on."),
        "assertion_form_note": (
            "add_bos is asserted with `is False`, NEVER `is not True` -- the "
            "latter passes silently on None."),
        "why_it_must_be_False": (
            "OLMo-2 is a BASE LM with no SFT/RL; a chat template would be "
            "unfair and would void comparability with every existing cell."),
    }
    out["ckpt_load_confirmed_from_eval_logs"] = {
        "freezefront": ("logs/olmo2_mmlu_content_7B_freezefront_step200000_shard0.log: "
                        "'[pruned] loaded ckpt step=200000 keep_front=14 "
                        "n_fresh=2 num_hidden_layers=16 (179 tensors, strict) "
                        "from outputs/olmo2_probe2_7B_keep14fresh2_freezefront/"
                        "step200000.pt'"),
        "fromscratch": ("logs/olmo2_mmlu_content_7B_scratch16L_step200000_shard0.log: "
                        "same form, from .../_fromscratch/step200000.pt"),
        "train_all": ("logs/olmo2_mmlu_content_7B_keep14_step200000_shard0.log: "
                      "same form, from .../keep14fresh2/step200000.pt"),
        "why": ("proves the eval loaded the STEP and ARCH it claims, and that "
                "all three slim zwfy6 copies are 179-tensor 16-layer models -- "
                "so FF's smaller wzc1 file is an optimizer-state artefact, not "
                "a different architecture"),
    }
    return out


def seed_disjointness_check():
    used = sorted(ARM_INDEX.values())
    archived = sorted({v for vs in ARCHIVED_OFFSET_BLOCKS.values() for v in vs})
    new = [v for k, v in ARM_INDEX.items() if k != "keep14fresh2_step200k"]
    clash = sorted(set(new) & set(archived))
    if clash:
        raise SystemExit(
            f"FATAL: bootstrap arm_index clash with archived evidence: {clash}. "
            "An archived margin could be perturbed by this run.")
    if 201 not in archived:
        raise SystemExit(
            "FATAL: 201 is not in the archived block list, so reusing it for the "
            "train-all endpoint would NOT reproduce the archive.")
    return {
        "arm_index": dict(ARM_INDEX),
        "form": "97*arm_index + 13*axis_index",
        "guard_seed_offset": "SEED+5700+13*axis_index",
        "n_boot": N_BOOT, "base_seed": SEED,
        "new_offsets": new,
        "archived_offsets": archived,
        "clash": clash,
        "endpoint_uses_archived_offset": 201,
        "reserved_unused": {str(k): v for k, v in RESERVED_UNUSED.items()},
        "checked": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--a04_dir", default=os.path.abspath(
        os.path.join(_HERE, "..")),
        help="A04 proposal dir; holds evidence/a04_shallow_rung_ni_7b.json, "
             "read at runtime for the archived endpoint margins")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--expect_numpy", default="",
                    help="refuse to publish from a different numpy")
    args = ap.parse_args()

    if args.expect_numpy and np.__version__ != args.expect_numpy:
        raise SystemExit(
            f"FATAL: numpy {np.__version__} != expected {args.expect_numpy}. "
            "Generator.multinomial differs in 19/10000 rows between 2.4.6 and "
            "2.5.1 (max margin drift 0.005294 pp); a mixed-numpy comparison is "
            "not reproducible. Re-run on the intended node or pass the actual "
            "version deliberately.")

    mm_root = os.path.join(args.raw_root, "olmo2_mmlu_content_results")
    cb_root = os.path.join(args.raw_root, "olmo2_closedbook_results")

    arm_specs = {
        "intact_7B_base": dict(ANCHOR),
        "keep14fresh2_step200k": {"mmlu": "7B_keep14_step200000",
                                  "cb": "keep14_step200k",
                                  "nq": "keep14_step200k_nqopen"},
        "freezefront_step200k": {"mmlu": "7B_freezefront_step200000",
                                 "cb": "freezefront_step200k",
                                 "nq": "freezefront_step200k_nqopen"},
        "fromscratch_step200k": {"mmlu": "7B_scratch16L_step200000",
                                 "cb": "fromscratch_step200k",
                                 "nq": "fromscratch_step200k_nqopen"},
    }

    seeds = seed_disjointness_check()
    protocol = protocol_asserted_2026_08_02_drivers(
        args.raw_root, {k: v for k, v in arm_specs.items()})
    shards = shard_integrity_report(mm_root, cb_root, arm_specs)
    for label, per_axis in shards.items():
        for axis, rec in per_axis.items():
            if rec.get("index_set") != list(range(8)) \
                    and rec.get("shard_index_set") != list(range(8)):
                iset = rec.get("index_set", rec.get("shard_index_set"))
                raise SystemExit(
                    f"FATAL {label}/{axis}: shard index set {iset} != [0..7]")

    data, prov = {}, {}
    for arm, spec in arm_specs.items():
        data[arm], prov[arm] = _load_arm(mm_root, cb_root, spec)
    integrity = assert_aligned(data, prov)

    # CANONICAL nulls, IMPORTED and CALLED. No margin is ever obtained by
    # subtracting a recorded null.
    nulls = build_nulls(data["intact_7B_base"])

    def null_acc(axis, conv):
        if axis == "mmlu_content":
            return nulls["mmlu_content"]["by_convention"][conv]
        return nulls[axis]["acc"]

    reported = {a: {x: float(data[a][x].mean()) for x in AXES if x in data[a]}
                for a in data}

    def seed_off(arm, axis):
        return 97 * ARM_INDEX[arm] + 13 * AXES.index(axis)

    # ---- guard D1-D6, BEFORE any NI (guard G1) ---------------------------
    guard = {}
    for conv in TIE_CONVS:
        guard[conv] = {}
        for axis in AXES:
            iv = data["intact_7B_base"][axis]
            nv = (nulls["mmlu_content"]["vectors"][conv]
                  if axis == "mmlu_content" else nulls[axis]["vector"])
            d = np.asarray(iv, float) - np.asarray(nv, float)
            resid = float(d.mean())
            resid_pp = 100.0 * resid
            _m, lo, hi, p = paired_bootstrap(
                d, seed=SEED + 5700 + 13 * AXES.index(axis))
            delta_pp = 100.0 * PREREG["delta_fraction"] * resid
            n = EXPECTED_N["mmlu" if axis == "mmlu_content" else axis]
            pstar = n * (delta_pp / (100.0 * Z95_TWO_SIDED)) ** 2
            pdisc = {}
            for arm in arm_specs:
                if arm == "intact_7B_base":
                    continue
                av = np.asarray(data[arm][axis], float)
                pdisc[arm] = float((av != np.asarray(iv, float)).mean())
            pdisc_max = max(pdisc.values())
            hw_by_arm = {a: 100.0 * Z95_TWO_SIDED * float(np.sqrt(v / n))
                         for a, v in pdisc.items()}
            d4 = {a: d4_interface_degenerate(data, a, axis, nulls)
                  for a in arm_specs}
            all_below_null = all(reported[a][axis] < null_acc(axis, conv)
                                 for a in arm_specs)
            cond = {
                "D1_residual_negative": bool(resid_pp < 0),
                "D2_residual_at_zero": bool(0 <= resid_pp
                                            <= D2_RESIDUAL_FLOOR_PP),
                "D3_ci_straddles_zero": bool(lo < 0 < hi),
                "D4_null_inadmissible": bool(
                    all_below_null or any(v["degenerate"]
                                          for v in d4.values())),
                "D6_delta_finer_than_instrument": bool(pdisc_max > pstar),
            }
            fatal = [k for k, v in cond.items() if v]
            guard[conv][axis] = {
                "residual_intact_pp": resid_pp,
                "null": float(np.asarray(nv, float).mean()),
                "reported_intact": float(np.asarray(iv, float).mean()),
                "ci95_pp": [100.0 * lo, 100.0 * hi],
                "boot_p": p, "delta_pp": delta_pp, "n": n,
                "pstar_crit_7B_recomputed": pstar,
                "p_disc_by_arm": pdisc, "p_disc_max": pdisc_max,
                "hw95_pp_by_arm": hw_by_arm,
                "d4_interface_by_arm": d4,
                "all_arms_below_null": all_below_null,
                "conditions": cond, "fatal_conditions": fatal,
                "classification": ("CERTIFIABLE" if not fatal
                                   else "NOT_CERTIFIABLE"),
                "decision_axis": axis not in DEMOTED_AXES,
            }

    # ---- NI, only on cells the guard did not retire ----------------------
    per_conv = {}
    for conv in TIE_CONVS:
        cells, retired = [], []
        for arm in arm_specs:
            if arm == "intact_7B_base":
                continue
            for axis in AXES:
                g = guard[conv][axis]
                if g["classification"] == "NOT_CERTIFIABLE":
                    retired.append({
                        "arm": arm, "axis": axis,
                        "fatal_conditions": g["fatal_conditions"],
                        "residual_intact_pp": g["residual_intact_pp"],
                        "delta_pp": g["delta_pp"],
                        "p_disc": g["p_disc_by_arm"][arm],
                        "ni_run": False,
                        "note": ("NI NOT RUN; excluded from the decision "
                                 "family. Never to be reported as 'NI "
                                 "rejected'.")})
                    continue
                r = ni_rule(data[arm][axis], data["intact_7B_base"][axis],
                            PREREG["delta_fraction"],
                            g["residual_intact_pp"] / 100.0,
                            seed_off=seed_off(arm, axis))
                arm_resid = reported[arm][axis] - null_acc(axis, conv)
                ir = g["residual_intact_pp"] / 100.0
                se = ((r["diff_mean_pp"] - r["diff_lower95_one_sided_pp"])
                      / 1.6449) if r["diff_mean_pp"] != r[
                          "diff_lower95_one_sided_pp"] else None
                margin = r["diff_lower95_one_sided_pp"] + r["delta_pp"]
                cells.append({
                    "arm": arm, "axis": axis,
                    "decision_axis": axis not in DEMOTED_AXES,
                    "reported": reported[arm][axis],
                    "reported_intact": reported["intact_7B_base"][axis],
                    "null": null_acc(axis, conv),
                    "residual_arm_pp": 100.0 * arm_resid,
                    "residual_intact_pp": g["residual_intact_pp"],
                    "residual_fraction_recovered": (arm_resid / ir
                                                    if ir > 0 else None),
                    "deficit_pp": g["residual_intact_pp"] - 100.0 * arm_resid,
                    "margin_pp": margin,
                    "bootstrap_se_pp": se,
                    "se_to_flip_NI": (abs(margin) / se) if se else None,
                    **r})
        n_dec = sum(1 for c in cells if c["decision_axis"])
        per_conv[conv] = {
            "intact_residual_pp": {x: guard[conv][x]["residual_intact_pp"]
                                   for x in AXES},
            "delta_pp": {x: guard[conv][x]["delta_pp"] for x in AXES},
            "cells": cells, "retired_cells": retired,
            "decision_family_size_full": (len(arm_specs) - 1)
            * len(DECISION_AXES),
            "decision_family_size_after_guard": n_dec,
            "ratio_rule": {a: ratio_rule(reported[a],
                                         reported["intact_7B_base"],
                                         PREREG["rho"],
                                         [x for x in AXES if x in data[a]])
                           for a in arm_specs if a != "intact_7B_base"},
        }

    # ---- reproduce the ARCHIVED train-all endpoint (hard gate) ------------
    archived = load_archived_train_all_margins(args.a04_dir)
    repro = {"tolerance_pp": REPRO_TOL_PP,
             "archived_source": archived["source"],
             "archived_arm": archived["arm"],
             "values_read_at_runtime_not_hardcoded": True,
             "why": ("proves the imported guard/anchor/rule ARE the objects that "
                     "produced evidence/a04_shallow_rung_ni_7b.json, so the two "
                     "new arms are on the same scale as the endpoint they are "
                     "compared to"),
             "per_axis": {}, "ok": True}
    for c in per_conv["split"]["cells"]:
        if c["arm"] != "keep14fresh2_step200k" or not c["decision_axis"]:
            continue
        a = archived["per_axis"][c["axis"]]
        dev = abs(c["margin_pp"] - a["margin_pp"])
        repro["per_axis"][c["axis"]] = {
            "archived_margin_pp": a["margin_pp"],
            "recomputed_margin_pp": c["margin_pp"],
            "abs_dev_pp": dev,
            "archived_boot_seed": a["boot_seed"],
            "recomputed_boot_seed": c["boot_seed"],
            "seed_matches": bool(a["boot_seed"] == c["boot_seed"]),
            "within_tolerance": bool(dev <= REPRO_TOL_PP)}
        if dev > REPRO_TOL_PP:
            repro["ok"] = False
    if not repro["ok"]:
        raise SystemExit(
            "FATAL: the archived train-all endpoint did NOT reproduce under its "
            f"archived offset 201: {json.dumps(repro['per_axis'], indent=1)}. "
            "The imported guard/anchor/rule are therefore NOT the ones that "
            "produced evidence/a04_shallow_rung_ni_7b.json, so the new arms are "
            "NOT on the same scale as the endpoint they are compared to. "
            "Refusing to publish.")

    # ---- verdict per convention -----------------------------------------
    verdict = {}
    for conv in TIE_CONVS:
        per_arm = {}
        for arm in arm_specs:
            if arm == "intact_7B_base":
                continue
            dec = [c for c in per_conv[conv]["cells"]
                   if c["arm"] == arm and c["decision_axis"]]
            acc = [c["axis"] for c in dec if c["ni_accept"]]
            n_surv = len(dec)
            need = int(np.ceil(0.50 * n_surv)) if n_surv else None
            per_arm[arm] = {
                "n_decision_axes_surviving_guard": n_surv,
                "n_decision_axes_accepting": len(acc),
                "axes_accepting": acc,
                "threshold_ge2of3_rescaled": need,
                "NI_OBSERVED_TO_ACCEPT": bool(n_surv and len(acc) >= need
                                              and len(acc) >= 1),
                "all_reject": bool(n_surv and not acc),
            }
        verdict[conv] = per_arm

    # ---- P1 / P2 / P3 ----------------------------------------------------
    def marg(conv, arm, axis):
        for c in per_conv[conv]["cells"]:
            if c["arm"] == arm and c["axis"] == axis:
                return c
        return None

    prereg_eval = {}
    for conv in TIE_CONVS:
        p1 = {"per_axis": {}, "n_satisfied": 0, "n_axes": 0,
              "violations": [], "violations_beyond_se": []}
        for axis in DECISION_AXES:
            cf, ct = marg(conv, "freezefront_step200k", axis), \
                marg(conv, "keep14fresh2_step200k", axis)
            if cf is None or ct is None:
                p1["per_axis"][axis] = {"evaluable": False,
                                        "reason": "cell retired by guard"}
                continue
            p1["n_axes"] += 1
            diff = cf["margin_pp"] - ct["margin_pp"]
            se_pool = float(np.sqrt(
                (cf["bootstrap_se_pp"] or 0.0) ** 2
                + (ct["bootstrap_se_pp"] or 0.0) ** 2))
            ok = diff <= 0
            rec = {"evaluable": True,
                   "margin_FF_pp": cf["margin_pp"],
                   "margin_trainall_pp": ct["margin_pp"],
                   "FF_minus_trainall_pp": diff,
                   "pooled_bootstrap_se_pp": se_pool,
                   "diff_over_pooled_se": (diff / se_pool if se_pool else None),
                   "P1_satisfied_on_this_axis": bool(ok),
                   "violation_exceeds_pooled_se": bool(
                       (not ok) and se_pool and diff > se_pool)}
            p1["per_axis"][axis] = rec
            if ok:
                p1["n_satisfied"] += 1
            else:
                p1["violations"].append(axis)
                if rec["violation_exceeds_pooled_se"]:
                    p1["violations_beyond_se"].append(axis)
        if p1["violations_beyond_se"]:
            p1["verdict"] = "P1_VIOLATED"
        elif p1["n_satisfied"] == p1["n_axes"]:
            p1["verdict"] = "P1_HOLDS"
        else:
            p1["verdict"] = "P1_HOLDS_WEAK"

        p2 = {"per_axis": {}, "n_axes_where_FS_lowest": 0, "n_axes": 0,
              "FS_rejects_all": None}
        for axis in DECISION_AXES:
            cs = {a: marg(conv, a, axis) for a in
                  ("keep14fresh2_step200k", "freezefront_step200k",
                   "fromscratch_step200k")}
            if any(v is None for v in cs.values()):
                p2["per_axis"][axis] = {"evaluable": False}
                continue
            p2["n_axes"] += 1
            order = sorted(cs.items(), key=lambda kv: kv[1]["margin_pp"])
            lowest = order[0][0]
            p2["per_axis"][axis] = {
                "evaluable": True,
                "margins_pp": {a: c["margin_pp"] for a, c in cs.items()},
                "ascending_order": [a for a, _ in order],
                "lowest": lowest,
                "FS_is_lowest": bool(lowest == "fromscratch_step200k")}
            if lowest == "fromscratch_step200k":
                p2["n_axes_where_FS_lowest"] += 1
        fs_dec = [c for c in per_conv[conv]["cells"]
                  if c["arm"] == "fromscratch_step200k" and c["decision_axis"]]
        p2["FS_rejects_all"] = bool(fs_dec and not any(c["ni_accept"]
                                                       for c in fs_dec))
        p2["verdict"] = ("P2_HOLDS"
                         if (p2["n_axes"] and p2["FS_rejects_all"]
                             and p2["n_axes_where_FS_lowest"]
                             >= int(np.ceil(2 / 3 * p2["n_axes"])))
                         else "P2_VIOLATED")

        p3 = {"fires": bool(p1["verdict"] == "P1_VIOLATED"),
              "axes_firing": list(p1["violations_beyond_se"])}
        p3["consequence"] = (PREREG_PREDICTIONS["P3"] if p3["fires"] else
                             ("P3 does NOT fire: the gate design's presumed "
                              "arm ordering is CONFIRMED at this damage depth, "
                              "and the 'freeze the inherited trunk instead' "
                              "route to an accept is CLOSED here. Per the "
                              "prereg's fixed asymmetry this is reported as a "
                              "measured confirmation, not as 'no finding'."))
        prereg_eval[conv] = {"P1": p1, "P2": p2, "P3": p3}

    # ---- P1, CONFIRMATORY: the paired item test on the SAME items ---------
    # The pre-registered P1 statistic is a difference of two margins over a
    # POOLED bootstrap SE, which treats the two arms' one-sided bounds as
    # independent. They are not: both arms are scored on the SAME item set, so
    # the sharper and more appropriate test is a PAIRED bootstrap on
    # `FF - train-all` per item. Two facts make this exact rather than a
    # substitution:
    #   * `residual(a) - residual(b) = reported(a) - reported(b)` -- the same
    #     input-blind null applies to both arms on the same items, so it CANCELS
    #     (documented in `ni_rule`'s own docstring).
    #   * `paired_bootstrap` is the SAME imported object every other A04 verdict
    #     uses for adjacent-interval resolution.
    # The PRE-REGISTERED verdict stays the pooled-SE one. This block can only
    # CORROBORATE or CONTRADICT it, and if it contradicts, the contradiction is
    # what gets reported.
    p1_paired = {
        "what": ("paired item bootstrap on FF - train-all, per axis, on the "
                 "identical item set (item_id sequences asserted equal by "
                 "assert_aligned)"),
        "why_sharper_than_the_prereg_statistic": (
            "the pre-registered P1 pools two independent bootstrap SEs; these "
            "arms share every item, so the paired difference removes the item "
            "sample as a source of variance entirely"),
        "status": ("CONFIRMATORY ONLY. The pre-registered P1 verdict is the "
                   "pooled-SE one and is NOT replaced. If this block disagreed "
                   "with it, the disagreement would be the reported result."),
        "resolution_rule": ("resolved iff CI95 excludes 0 AND p < 0.05 "
                            "(conservative AND -- the same rule the keep14 and "
                            "keep10 verdicts use; picking the favourable "
                            "criterion would turn a tie into a result)"),
        "per_axis": {},
    }
    for axis in AXES:
        ff = np.asarray(data["freezefront_step200k"][axis], float)
        ta = np.asarray(data["keep14fresh2_step200k"][axis], float)
        d = ff - ta
        m, lo, hi, p = paired_bootstrap(
            d, seed=SEED + 5900 + 13 * AXES.index(axis))
        w2r = int(((ta == 0) & (ff == 1)).sum())
        r2w = int(((ta == 1) & (ff == 0)).sum())
        resolved = bool((lo > 0 or hi < 0) and p < 0.05)
        p1_paired["per_axis"][axis] = {
            "decision_axis": axis not in DEMOTED_AXES,
            "FF_acc": float(ff.mean()), "trainall_acc": float(ta.mean()),
            "FF_minus_trainall_pp": 100.0 * float(m),
            "ci95_pp": [100.0 * lo, 100.0 * hi], "boot_p": p,
            "resolved": resolved,
            "criteria_disagree": bool(((lo > 0 or hi < 0)) != (p < 0.05)),
            "flips_trainall_wrong_to_FF_right": w2r,
            "flips_trainall_right_to_FF_wrong": r2w,
            "n_items": int(d.size),
            "FF_strictly_better_and_resolved": bool(resolved and m > 0),
        }

    # Degeneracy / verbosity diagnostics on the axis that fires P1, so a
    # "FF is better" reading cannot rest on an output-format artefact. LABELLED
    # DIAGNOSTIC -- never enters a verdict.
    p1_diag = {"status": ("LABELLED DIAGNOSTIC. Does not enter any verdict. "
                          "Exists because A04_GATE_DESIGN 4.1 bans raw "
                          "`contains` and PROPOSAL.md 4.4 showed a generative "
                          "EM move can be ~half verbosity."),
               "per_axis": {}}
    for axis in ("triviaqa", "popqa", "nq_open"):
        rec = {}
        for arm in ("keep14fresh2_step200k", "freezefront_step200k",
                    "fromscratch_step200k"):
            rows = data[arm][f"_{axis}_rows"]
            preds = [(r.get("pred") or "").strip() for r in rows]
            low = [p.lower() for p in preds]
            counts = {}
            for x in low:
                counts[x] = counts.get(x, 0) + 1
            top = max(counts.values()) / len(low)
            n_empty = sum(1 for x in preds if not x)
            em = np.asarray(data[arm][axis], float)
            has_c = [1 if r.get("contains") else 0 for r in rows]
            rec[arm] = {
                "em_pct": 100.0 * float(em.mean()),
                "contains_pct": (100.0 * float(np.mean(has_c))
                                 if any(("contains" in r) for r in rows[:5])
                                 else None),
                "empty_pred_frac": n_empty / len(preds),
                "top_constant_frac": top,
                "n_distinct_preds": len(counts),
                "mean_pred_chars": float(np.mean([len(x) for x in preds])),
            }
        ff, ta = rec["freezefront_step200k"], rec["keep14fresh2_step200k"]
        cg = (None if (ff["contains_pct"] is None or ta["contains_pct"] is None)
              else ff["contains_pct"] - ta["contains_pct"])
        rec["FF_vs_trainall"] = {
            "em_move_pp": ff["em_pct"] - ta["em_pct"],
            "contains_move_pp": cg,
            "em_move_exceeds_contains_move": (
                None if cg is None else bool(abs(ff["em_pct"] - ta["em_pct"])
                                             > abs(cg))),
            "mean_pred_chars_FF_over_trainall": (
                ff["mean_pred_chars"] / ta["mean_pred_chars"]
                if ta["mean_pred_chars"] else None),
            "reading": ("if EM rises while `contains` is flat or falls, the EM "
                        "gain is at least partly a FORMAT effect (shorter, more "
                        "exactly-matching outputs), not new knowledge -- the "
                        "mirror image of PROPOSAL.md 4.4's full32 case"),
        }
        p1_diag["per_axis"][axis] = rec

    # The decisive mechanism question for the axis that fires P3: of the items
    # FF gets RIGHT that train-all gets WRONG, how many did train-all's own
    # prediction already CONTAIN the gold answer? Those are items where train-all
    # HAD the fact and lost the EM on formatting -- so FF's "gain" there is a
    # format gain, not knowledge. This is the exact statistic `PROPOSAL.md` 4.4
    # used for full32 (622 of 1313 = 47.37 %), applied in the opposite direction.
    # LABELLED DIAGNOSTIC. It does NOT re-score any cell and `contains` is NEVER
    # substituted for EM as the decision metric (banned by GATE_DESIGN 4.1).
    p1_diag["gain_decomposition_FF_over_trainall"] = {
        "status": ("LABELLED DIAGNOSTIC, mandatory in any writeup that quotes "
                   "the P1 violation. Does not re-score any cell. `contains` is "
                   "NOT substituted for EM (GATE_DESIGN 4.1 bans it as a "
                   "decision metric); this only characterises what the EM move "
                   "CONSISTS OF."),
        "method": ("on items where train-all EM=0 and FF EM=1, count how many "
                   "of train-all's predictions already CONTAINED the gold "
                   "answer. Those are facts train-all HAD and lost on format, "
                   "so FF's EM gain on them is a FORMAT gain, not knowledge."),
        "per_axis": {},
    }
    for axis in ("triviaqa", "popqa", "nq_open"):
        ta_rows = data["keep14fresh2_step200k"][f"_{axis}_rows"]
        ff_rows = data["freezefront_step200k"][f"_{axis}_rows"]
        ta_em = np.asarray(data["keep14fresh2_step200k"][axis], float)
        ff_em = np.asarray(data["freezefront_step200k"][axis], float)
        gain = np.where((ta_em == 0) & (ff_em == 1))[0]
        loss = np.where((ta_em == 1) & (ff_em == 0))[0]
        g_had = sum(1 for i in gain if ta_rows[int(i)].get("contains"))
        l_kept = sum(1 for i in loss if ff_rows[int(i)].get("contains"))
        g_len_ta = ([len((ta_rows[int(i)].get("pred") or "")) for i in gain]
                    or [0])
        g_len_ff = ([len((ff_rows[int(i)].get("pred") or "")) for i in gain]
                    or [0])
        p1_diag["gain_decomposition_FF_over_trainall"]["per_axis"][axis] = {
            "n_FF_gains": int(gain.size),
            "n_FF_losses": int(loss.size),
            "of_FF_gains_trainall_already_CONTAINED_gold": g_had,
            "frac_of_gains_that_are_FORMAT_only": (g_had / gain.size
                                                   if gain.size else None),
            "of_FF_losses_FF_still_CONTAINS_gold": l_kept,
            "frac_of_losses_that_are_FORMAT_only": (l_kept / loss.size
                                                    if loss.size else None),
            "mean_pred_chars_on_gain_items_trainall": float(np.mean(g_len_ta)),
            "mean_pred_chars_on_gain_items_FF": float(np.mean(g_len_ff)),
        }

    # ADDITIVE DECOMPOSITION of the EM move into a FORMAT part and a CONTENT
    # part, with a bootstrap on the CONTENT part. This is the statistic that
    # decides whether the P1 violation is a knowledge effect or a formatting
    # effect, and it is the sharpest thing in this file.
    #
    #   EM_move = [ (gains train-all already contained) - (losses FF still
    #               contains) ]                                   <- FORMAT
    #           + [ (gains that are genuinely new)      - (losses that are
    #               genuine content loss) ]                       <- CONTENT
    #
    # The two parts sum EXACTLY to the observed EM move (asserted below), so this
    # is a partition of the SAME items, not a re-scoring. `contains` is used ONLY
    # to LABEL an item, never as the metric.
    n_dec = {"partition_identity_asserted": True, "per_axis": {}}
    for axis in ("triviaqa", "popqa", "nq_open"):
        ta_rows = data["keep14fresh2_step200k"][f"_{axis}_rows"]
        ff_rows = data["freezefront_step200k"][f"_{axis}_rows"]
        ta_em = np.asarray(data["keep14fresh2_step200k"][axis], float)
        ff_em = np.asarray(data["freezefront_step200k"][axis], float)
        n = ta_em.size
        fmt = np.zeros(n)      # per-item format-attributable EM delta
        cnt = np.zeros(n)      # per-item content-attributable EM delta
        for i in range(n):
            if ta_em[i] == 0 and ff_em[i] == 1:
                if ta_rows[i].get("contains"):
                    fmt[i] = 1.0        # train-all HAD it, lost only the EM
                else:
                    cnt[i] = 1.0        # genuinely new
            elif ta_em[i] == 1 and ff_em[i] == 0:
                if ff_rows[i].get("contains"):
                    fmt[i] = -1.0       # FF still has it, lost only the EM
                else:
                    cnt[i] = -1.0       # genuine content loss
        total = 100.0 * float((fmt + cnt).mean())
        observed = 100.0 * float((ff_em - ta_em).mean())
        if abs(total - observed) > 1e-9:
            raise SystemExit(
                f"FATAL {axis}: format+content decomposition {total} != observed "
                f"EM move {observed}. The partition is not exhaustive; refusing "
                "to publish a decomposition that does not add up.")
        mc, loc, hic, pc = paired_bootstrap(
            cnt, seed=SEED + 6100 + 13 * AXES.index(axis))
        mf, lof, hif, pf = paired_bootstrap(
            fmt, seed=SEED + 6200 + 13 * AXES.index(axis))
        n_dec["per_axis"][axis] = {
            "decision_axis": axis not in DEMOTED_AXES,
            "observed_EM_move_pp": observed,
            "FORMAT_part_pp": 100.0 * float(mf),
            "FORMAT_ci95_pp": [100.0 * lof, 100.0 * hif],
            "FORMAT_boot_p": pf,
            "CONTENT_part_pp": 100.0 * float(mc),
            "CONTENT_ci95_pp": [100.0 * loc, 100.0 * hic],
            "CONTENT_boot_p": pc,
            "CONTENT_resolved": bool((loc > 0 or hic < 0) and pc < 0.05),
            "format_share_of_abs_move": (abs(float(mf))
                                         / (abs(float(mf)) + abs(float(mc)))
                                         if (abs(float(mf)) + abs(float(mc)))
                                         else None),
        }
    n_dec["reading"] = (
        "If the CONTENT part is unresolved while the FORMAT part carries the "
        "move, then the P1 violation on that axis is a statement about OUTPUT "
        "FORMAT, not about recovered knowledge -- and P3's inference ('freezing "
        "the trunk repairs better') does NOT follow from it.")
    n_dec["what_it_does_NOT_do"] = (
        "It does not re-score any cell, does not substitute `contains` for EM, "
        "and does not change any NI verdict. Every margin in this file is still "
        "the EM margin from the imported `ni_rule`.")
    p1_diag["format_vs_content_decomposition"] = n_dec

    # ---- P3, qualified by the decomposition ------------------------------
    # P3's PRE-REGISTERED trigger is "P1 is violated beyond SE", and that is
    # recorded UNCHANGED in prereg_evaluation[conv]["P3"]. But P3's *inference*
    # ("train-all is not the stronger repair, so the route to an accept may be
    # freezing the trunk") only follows if the violating move is a KNOWLEDGE
    # move. The prereg did not anticipate that a violation could be carried by
    # output format -- PROPOSAL.md 4.4 had shown that for full32, in the
    # opposite direction, but P1 was written as a pure margin comparison.
    #
    # This block therefore does NOT retune P3's trigger. It records, per firing
    # axis, whether the CONTENT part of that axis's move is resolved, and states
    # which of P3's two clauses survives. Both readings are emitted so a reader
    # can apply either.
    for conv in TIE_CONVS:
        p3 = prereg_eval[conv]["P3"]
        qual = {"trigger_unchanged": True,
                "trigger": "P1 violated beyond pooled bootstrap SE",
                "per_firing_axis": {}}
        for axis in p3["axes_firing"]:
            dd = n_dec["per_axis"].get(axis)
            if dd is None:
                qual["per_firing_axis"][axis] = {"decomposable": False}
                continue
            qual["per_firing_axis"][axis] = {
                "decomposable": True,
                "FORMAT_part_pp": dd["FORMAT_part_pp"],
                "CONTENT_part_pp": dd["CONTENT_part_pp"],
                "CONTENT_resolved": dd["CONTENT_resolved"],
                "format_share_of_abs_move": dd["format_share_of_abs_move"],
                "clause_1_survives": True,
                "clause_1": ("'margin_pp(FF) > margin_pp(train-all) on this "
                             "axis' -- a MEASURED fact about the decision "
                             "statistic the gate actually uses. Survives "
                             "regardless of mechanism, because the gate reads "
                             "EM and EM is what moved."),
                "clause_2_survives": bool(dd["CONTENT_resolved"]
                                          and dd["CONTENT_part_pp"] > 0),
                "clause_2": ("'freezing the inherited trunk RECOVERS MORE "
                             "KNOWLEDGE' -- requires the CONTENT part of the "
                             "move to be positive AND resolved. If it is not, "
                             "this clause does NOT follow and may not be "
                             "claimed."),
            }
        surviving = [a for a, v in qual["per_firing_axis"].items()
                     if v.get("clause_2_survives")]
        qual["clause_2_surviving_axes"] = surviving
        qual["net_reading"] = (
            ("P3 fires and BOTH clauses survive on " + ", ".join(surviving)
             + ": the arm ordering is wrong AND the mechanism is knowledge.")
            if surviving else
            ("P3 FIRES on the decision statistic (clause 1) but clause 2 does "
             "NOT survive on any firing axis: the violation is carried by "
             "OUTPUT FORMAT, not by recovered knowledge. Consequence: A04's "
             "arm ordering is still shown to be UNTESTED and rung selection "
             "must still be widened to (depth, repair mode) -- because the "
             "gate's own decision metric IS EM and EM did reorder the arms -- "
             "but the positive route 'freeze the trunk to reach an accept' is "
             "NOT supported. The finding is about the GATE'S METRIC, not about "
             "repair mechanics."))
        p3["qualified_by_format_content_decomposition"] = qual


    q2 = {"question": ("as a FLOOR ANCHOR, how far is FS from the null "
                       "itself? If FS sits ABOVE the null, that much of the "
                       "intact residual is reachable by architecture+training "
                       "with ZERO inheritance, which COMPRESSES the space A04 "
                       "can call 'recovery'."),
          "per_axis": {}}
    for axis in AXES:
        nv = (nulls["mmlu_content"]["vectors"]["split"]
              if axis == "mmlu_content" else nulls[axis]["vector"])
        fs = np.asarray(data["fromscratch_step200k"][axis], float)
        d = fs - np.asarray(nv, float)
        m, lo, hi, p = paired_bootstrap(
            d, seed=SEED + 5800 + 13 * AXES.index(axis))
        ir = guard["split"][axis]["residual_intact_pp"]
        q2["per_axis"][axis] = {
            "decision_axis": axis not in DEMOTED_AXES,
            "FS_acc": float(fs.mean()),
            "null_acc": float(np.asarray(nv, float).mean()),
            "FS_residual_pp": 100.0 * float(m),
            "ci95_pp": [100.0 * lo, 100.0 * hi],
            "boot_p": p,
            "FS_above_its_own_null": bool(lo > 0),
            "intact_residual_pp": ir,
            "FS_residual_as_fraction_of_intact": (100.0 * float(m) / ir
                                                  if ir else None),
        }
    q2["interpretation_rule_fixed_in_advance"] = (
        "A positive, resolved FS residual means the null is NOT the right "
        "reference for 'how much did inheritance buy': the correct comparison "
        "for an inheritance claim is arm-vs-FS, not arm-vs-null. This is a "
        "statement about what the residual MEASURES, not a re-definition of "
        "Delta -- Delta stays 0.10 x residual(intact) and is never substituted "
        "(guard G2).")

    # The consequence of Q2, made quantitative: how much of each arm's
    # "recovery" is NOT attributable to inheritance, because a zero-inheritance
    # model of the same shape and budget already reaches it?
    #
    #   recovered_fraction   = residual(arm) / residual(intact)
    #   floor_fraction       = residual(FS)  / residual(intact)
    #   inheritance_premium  = (residual(arm) - residual(FS)) / residual(intact)
    #
    # This does NOT redefine Delta and does NOT change any NI verdict. It bounds
    # what the phrase "X % recovered" is entitled to mean.
    prem = {"what": ("the share of an arm's calibrated residual that a "
                     "ZERO-INHERITANCE model of the same depth, corpus and "
                     "step budget does NOT already reach"),
            "does_not_change": ("Delta (still 0.10 x residual(intact), never "
                                "substituted, guard G2) or any NI verdict"),
            "caveat": ("FS ran at uniform 1e-4 vs 2e-5 for the other two arms "
                       "(_classify_param returns 'fresh' first under "
                       "from_scratch), so the floor is LR-confounded and the "
                       "premium is correspondingly uncertain in BOTH "
                       "directions"),
            "per_axis": {}}
    for axis in AXES:
        ir = guard["split"][axis]["residual_intact_pp"]
        fs_r = q2["per_axis"][axis]["FS_residual_pp"]
        rec = {"intact_residual_pp": ir, "FS_floor_residual_pp": fs_r,
               "floor_fraction_of_intact": (fs_r / ir if ir else None),
               "by_arm": {}}
        for arm in ("keep14fresh2_step200k", "freezefront_step200k"):
            cell = next((c for c in per_conv["split"]["cells"]
                         if c["arm"] == arm and c["axis"] == axis), None)
            if cell is None:
                continue
            ar = cell["residual_arm_pp"]
            av = np.asarray(data[arm][axis], float)
            fv = np.asarray(data["fromscratch_step200k"][axis], float)
            m, lo, hi, p = paired_bootstrap(
                av - fv, seed=SEED + 6300 + 13 * AXES.index(axis)
                + (7 if arm == "freezefront_step200k" else 0))
            rec["by_arm"][arm] = {
                "arm_residual_pp": ar,
                "recovered_fraction_of_intact": ar / ir if ir else None,
                "inheritance_premium_pp": ar - fs_r,
                "inheritance_premium_fraction_of_intact": ((ar - fs_r) / ir
                                                           if ir else None),
                "arm_minus_FS_ci95_pp": [100.0 * lo, 100.0 * hi],
                "arm_minus_FS_boot_p": p,
                "premium_resolved": bool((lo > 0 or hi < 0) and p < 0.05),
            }
        prem["per_axis"][axis] = rec
    q2["inheritance_premium"] = prem

    # ---- Q3: the 3-arm ordering vs the rules ------------------------------
    q3 = {"ordering_per_axis": {}, "ratio_rule_split": {},
          "PLATEAU_not_computable": {
              "reason": ("olmo2_ppl_results/ contains NO freezefront or "
                         "fromscratch directory (checked on zwfy6: only "
                         "7B_keep14_step{0,128000,153500,200000}(_v2) and an "
                         "unrelated 7B_scratch16L_lr2e5_* LR-control run at "
                         "uniform 2e-5 on the OTHER corpus, which is a "
                         "DIFFERENT run from this FS arm)."),
              "consequence": ("Q3 can be answered against RATIO(0.85) ONLY. "
                              "RATIO is NOT the plateau rule, so this is at "
                              "most HALF of Q3, and the missing half is a "
                              "design limitation registered in the prereg "
                              "BEFORE scoring, not an omission found after."),
          }}
    for axis in DECISION_AXES:
        cs = {a: marg("split", a, axis) for a in
              ("keep14fresh2_step200k", "freezefront_step200k",
               "fromscratch_step200k")}
        if any(v is None for v in cs.values()):
            q3["ordering_per_axis"][axis] = {"evaluable": False}
            continue
        order = sorted(cs.items(), key=lambda kv: -kv[1]["margin_pp"])
        q3["ordering_per_axis"][axis] = {
            "descending_by_margin": [a for a, _ in order],
            "margins_pp": {a: c["margin_pp"] for a, c in cs.items()},
            "all_reject": all(not c["ni_accept"] for c in cs.values()),
        }
    for a, r in per_conv["split"]["ratio_rule"].items():
        q3["ratio_rule_split"][a] = r
    q3["rule_disagreement_cells"] = {
        a: {"NI_accepts": verdict["split"][a]["NI_OBSERVED_TO_ACCEPT"],
            "RATIO_accepts": per_conv["split"]["ratio_rule"][a]["ratio_accept"],
            "disagree": bool(verdict["split"][a]["NI_OBSERVED_TO_ACCEPT"]
                             != per_conv["split"]["ratio_rule"][a]["ratio_accept"])}
        for a in verdict["split"]}

    out = {
        "gate": "A04_control_arms_NI_freezefront_fromscratch_7B",
        "question": ("do the two never-tested REPAIR-MODE controls at "
                     "keep_front=14 (freeze the inherited front / no "
                     "inheritance at all) change A04's arm ordering, and does "
                     "either admit an NI accept?"),
        "date": "2026-08-13",
        "gpu_spent": 0,
        "gpu_note": ("CPU-only re-analysis of per-example shards already on "
                     "zwfy6, written 2026-08-02. No model loaded, no node's "
                     "GPUs touched. Read-only on every input."),
        "prereg": {
            "document": PREREG_DOC,
            "commit": PREREG_COMMIT,
            "committed_before_first_margin": True,
            "predictions": PREREG_PREDICTIONS,
            "gate_design": "A04_GATE_DESIGN.md 2 / 2.0.1 / 2.0.2",
            "delta_fraction": PREREG["delta_fraction"],
            "rho": PREREG["rho"],
            "commit_freezing_constants": PREREG["commit"],
            "decision_axes": DECISION_AXES,
            "demoted_axes": sorted(DEMOTED_AXES),
            "delta_never_substituted": True,
            "anchor_never_changed": True,
            "nulls_are_imported_and_CALLED": (
                "build_nulls from pilot_zero_rule_disagreement. No margin is "
                "derived by subtracting a recorded null -- the error mode that "
                "produced four wrong numbers on 2026-08-13."),
        },
        "intact_anchor": {
            "choice": "vanilla models/OLMo-2-1124-7B (mode=base, 32 layers)",
            "dirs": ANCHOR,
            "imported_from": "a04_shallow_rung_ni_7b.ANCHOR",
            "never_redeclared": True,
        },
        "arm_provenance": ARM_PROVENANCE,
        "matched_across_all_three": MATCHED_ACROSS_ALL_THREE,
        "sampler_regime": SAMPLER_REGIME,
        "step23500_dropped": {
            "dropped": True,
            "path": "outputs/olmo2_probe2_7B_keep14fresh2_freezefront/step23500.pt",
            "why": ARM_PROVENANCE["freezefront_step200k"][
                "zwfy6_log_is_a_DIFFERENT_run"]["consequence"],
            "not_merely_demoted": (
                "It is not a 'far neighbour with a 176,500-step gap'. It is a "
                "checkpoint of a DIFFERENT RUN on a DIFFERENT CORPUS at a "
                "DIFFERENT micro-batch geometry. A gap statement would imply "
                "one trajectory; there are two."),
            "neighbour_statement_2_0_2": (
                "NO adjacent saved checkpoint exists for either arm at "
                "step200000. A04_GATE_DESIGN 2.0.2 explicitly permits 'or a "
                "statement that none exist'. Since no cell here ACCEPTS, the "
                "precondition has nothing to protect."),
        },
        "bootstrap_offsets": seeds,
        "protocol_asserted": protocol,
        "shard_integrity": shards,
        "integrity": integrity,
        "arms": {a: prov[a] for a in prov},
        "nulls": {
            "mmlu_content": {k: v for k, v in nulls["mmlu_content"].items()
                             if k != "vectors"},
            **{t: {k: v for k, v in nulls[t].items() if k != "vector"}
               for t in ("triviaqa", "popqa", "nq_open")},
        },
        "reported_acc": reported,
        "guard_D1_D6": guard,
        "archived_endpoint_reproduction": repro,
        "per_convention": per_conv,
        "verdict_by_convention": verdict,
        "prereg_evaluation": prereg_eval,
        "P1_paired_confirmatory": p1_paired,
        "P1_degeneracy_diagnostic": p1_diag,
        "Q2_floor_anchor": q2,
        "Q3_ordering_and_rules": q3,
        "environment": {
            "numpy": np.__version__,
            "node": os.environ.get("A04_NODE", "unset"),
            "why_numpy_matters": (
                "Generator.multinomial differs in 19/10000 rows between 2.4.6 "
                "(.82) and 2.5.1 (.73/.104); max observed margin drift "
                "0.005294 pp. All cells in this file come from ONE node/numpy."),
            "no_margin_quoted_finer_than": "0.01 pp (must_not_claim[24])",
        },
        "not_licensed": [
            "any sigma_run or seed-variance statement (one seed per arm; no 7B "
            "sd_run exists or is reconstructible)",
            "treating the three arms as a depth ladder (all keep_front=14)",
            "any clean 'inheritance is worth X' claim from FS (its LR is 5x the "
            "other two arms)",
            "any PLATEAU(T) comparison for FF/FS (no PPL on disk)",
            "quoting any margin finer than 0.01 pp across nodes",
            "calling freezefront@step23500 a neighbour of step200000",
        ],
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1, default=float)

    # ---------------- console report ------------------------------------
    W = 104
    print("=" * W)
    print("ARCHIVED train-all ENDPOINT REPRODUCTION (offset 201, hard gate)")
    print("=" * W)
    for axis, r in repro["per_axis"].items():
        print(f"  {axis:<14} archived={r['archived_margin_pp']:>11.6f}  "
              f"recomputed={r['recomputed_margin_pp']:>11.6f}  "
              f"dev={r['abs_dev_pp']:.2e}  "
              f"{'OK' if r['within_tolerance'] else 'FAIL'}")
    print()
    print("=" * W)
    print("GUARD D1-D6 (`split`), evaluated BEFORE NI")
    print("=" * W)
    for axis in AXES:
        g = guard["split"][axis]
        print(f"  {axis:<14} resid_intact={g['residual_intact_pp']:>9.4f}pp  "
              f"Delta={g['delta_pp']:>7.4f}  n={g['n']:>6}  "
              f"{g['classification']}"
              + (f"  <- {','.join(g['fatal_conditions'])}"
                 if g["fatal_conditions"] else ""))
    print()
    print("=" * W)
    print("NI(Delta) -- `split` convention")
    print("=" * W)
    print(f"  {'arm':<24}{'axis':<14}{'acc%':>8}{'recov%':>8}"
          f"{'lo95':>10}{'Delta':>8}{'margin':>10}{'SE':>7}  NI")
    for c in per_conv["split"]["cells"]:
        rf = c["residual_fraction_recovered"]
        print(f"  {c['arm']:<24}{c['axis']:<14}{100*c['reported']:>7.3f}%"
              f"{100*rf if rf is not None else float('nan'):>7.1f}%"
              f"{c['diff_lower95_one_sided_pp']:>10.4f}{c['delta_pp']:>8.4f}"
              f"{c['margin_pp']:>10.4f}"
              f"{c['bootstrap_se_pp'] if c['bootstrap_se_pp'] else float('nan'):>7.3f}"
              f"  {'ACCEPT' if c['ni_accept'] else 'REJECT'}"
              + ("" if c["decision_axis"] else "  (demoted)"))
    print()
    print("=" * W)
    print("PRE-REGISTERED PREDICTIONS (`split`)")
    print("=" * W)
    pe = prereg_eval["split"]
    print(f"  P1: {pe['P1']['verdict']}  "
          f"({pe['P1']['n_satisfied']}/{pe['P1']['n_axes']} axes satisfy "
          f"FF <= train-all)")
    for axis, r in pe["P1"]["per_axis"].items():
        if not r.get("evaluable"):
            continue
        print(f"      {axis:<14} FF={r['margin_FF_pp']:>10.4f}  "
              f"train-all={r['margin_trainall_pp']:>10.4f}  "
              f"diff={r['FF_minus_trainall_pp']:>+9.4f}  "
              f"diff/SE={r['diff_over_pooled_se']:>+7.2f}  "
              f"{'OK' if r['P1_satisfied_on_this_axis'] else 'VIOLATION'}"
              + ("  (beyond SE)" if r["violation_exceeds_pooled_se"] else ""))
    print(f"  P2: {pe['P2']['verdict']}  "
          f"(FS lowest on {pe['P2']['n_axes_where_FS_lowest']}/"
          f"{pe['P2']['n_axes']}; FS rejects all = {pe['P2']['FS_rejects_all']})")
    for axis, r in pe["P2"]["per_axis"].items():
        if not r.get("evaluable"):
            continue
        print(f"      {axis:<14} ascending: "
              + " < ".join(a.replace("_step200k", "")
                           for a in r["ascending_order"]))
    print(f"  P3: {'FIRES' if pe['P3']['fires'] else 'does not fire'}  "
          f"{pe['P3']['axes_firing']}")
    print()
    print("=" * W)
    print("P1 CONFIRMATORY -- paired item bootstrap on FF - train-all")
    print("=" * W)
    for axis, r in p1_paired["per_axis"].items():
        print(f"  {axis:<14} FF-trainall={r['FF_minus_trainall_pp']:>+9.4f}pp  "
              f"CI95=[{r['ci95_pp'][0]:>+8.4f},{r['ci95_pp'][1]:>+8.4f}]  "
              f"p={r['boot_p']:.4f}  "
              f"{'RESOLVED' if r['resolved'] else 'not resolved':<12}  "
              f"flips +{r['flips_trainall_wrong_to_FF_right']}/"
              f"-{r['flips_trainall_right_to_FF_wrong']}"
              + ("" if r["decision_axis"] else "  (demoted)"))
    print()
    print("=" * W)
    print("P1 DIAGNOSTIC -- is the popqa gain a format effect? (never a verdict)")
    print("=" * W)
    for axis, rec in p1_diag["per_axis"].items():
        v = rec["FF_vs_trainall"]
        print(f"  {axis:<12} EM move={v['em_move_pp']:>+8.4f}pp  "
              f"contains move="
              + (f"{v['contains_move_pp']:>+8.4f}pp" if v["contains_move_pp"]
                 is not None else "     n/a")
              + f"  chars FF/train-all="
              + (f"{v['mean_pred_chars_FF_over_trainall']:.3f}"
                 if v["mean_pred_chars_FF_over_trainall"] else "n/a"))
        for arm in ("keep14fresh2_step200k", "freezefront_step200k",
                    "fromscratch_step200k"):
            a = rec[arm]
            print(f"      {arm:<24} em={a['em_pct']:>7.3f}%  "
                  f"empty={100*a['empty_pred_frac']:.3f}%  "
                  f"topconst={100*a['top_constant_frac']:.3f}%  "
                  f"distinct={a['n_distinct_preds']:>6}  "
                  f"chars={a['mean_pred_chars']:.2f}")
    print()
    print("  gain decomposition (of FF's EM gains, how many did train-all "
          "already CONTAIN?)")
    for axis, r in p1_diag["gain_decomposition_FF_over_trainall"][
            "per_axis"].items():
        fg = r["frac_of_gains_that_are_FORMAT_only"]
        fl = r["frac_of_losses_that_are_FORMAT_only"]
        print(f"      {axis:<12} gains={r['n_FF_gains']:>5} of which "
              f"train-all already contained gold="
              f"{r['of_FF_gains_trainall_already_CONTAINED_gold']:>5}"
              + (f" ({100*fg:.2f}%)" if fg is not None else "")
              + f"   losses={r['n_FF_losses']:>5} of which FF still contains="
              f"{r['of_FF_losses_FF_still_CONTAINS_gold']:>5}"
              + (f" ({100*fl:.2f}%)" if fl is not None else ""))
    print()
    print("  FORMAT vs CONTENT decomposition of the EM move (sums EXACTLY to it)")
    for axis, r in p1_diag["format_vs_content_decomposition"][
            "per_axis"].items():
        print(f"      {axis:<12} observed={r['observed_EM_move_pp']:>+8.4f}pp = "
              f"FORMAT {r['FORMAT_part_pp']:>+8.4f} "
              f"[{r['FORMAT_ci95_pp'][0]:>+7.4f},{r['FORMAT_ci95_pp'][1]:>+7.4f}] "
              f"+ CONTENT {r['CONTENT_part_pp']:>+8.4f} "
              f"[{r['CONTENT_ci95_pp'][0]:>+7.4f},{r['CONTENT_ci95_pp'][1]:>+7.4f}] "
              f"p={r['CONTENT_boot_p']:.4f} "
              f"{'CONTENT RESOLVED' if r['CONTENT_resolved'] else 'content NOT resolved'}")
    print()
    q = pe["P3"].get("qualified_by_format_content_decomposition")
    if q:
        print("  P3 qualification:")
        for axis, v in q["per_firing_axis"].items():
            print(f"      {axis:<12} clause1(ordering)={v.get('clause_1_survives')}  "
                  f"clause2(knowledge)={v.get('clause_2_survives')}")
        print(f"      NET: {q['net_reading']}")
    print()
    print("=" * W)
    print("Q2 -- FS vs its OWN best-constant null (`split`)")
    print("=" * W)
    for axis, r in q2["per_axis"].items():
        print(f"  {axis:<14} FS={100*r['FS_acc']:>7.3f}%  "
              f"null={100*r['null_acc']:>7.3f}%  "
              f"resid={r['FS_residual_pp']:>+9.4f}pp  "
              f"CI95=[{r['ci95_pp'][0]:>+8.4f},{r['ci95_pp'][1]:>+8.4f}]  "
              f"p={r['boot_p']:.4f}  "
              f"{'ABOVE null' if r['FS_above_its_own_null'] else 'at/below'}"
              + (f"  ={r['FS_residual_as_fraction_of_intact']*100:.1f}% of "
                 f"intact resid" if r["FS_residual_as_fraction_of_intact"]
                 is not None else ""))
    print()
    print("  inheritance premium (recovered above the ZERO-INHERITANCE floor)")
    for axis, r in q2["inheritance_premium"]["per_axis"].items():
        print(f"      {axis:<14} floor={100*r['floor_fraction_of_intact']:>6.2f}% "
              f"of intact residual")
        for arm, v in r["by_arm"].items():
            print(f"          {arm:<24} "
                  f"recovered={100*v['recovered_fraction_of_intact']:>6.2f}%  "
                  f"premium={100*v['inheritance_premium_fraction_of_intact']:>+6.2f}%"
                  f" ({v['inheritance_premium_pp']:>+7.4f}pp)  "
                  f"p={v['arm_minus_FS_boot_p']:.4f}  "
                  f"{'resolved' if v['premium_resolved'] else 'NOT resolved'}")
    print()
    print("=" * W)
    print("Q3 -- 3-arm ordering (descending margin) + RATIO(0.85)")
    print("=" * W)
    for axis, r in q3["ordering_per_axis"].items():
        if not r.get("evaluable", True):
            continue
        print(f"  {axis:<14} "
              + " > ".join(a.replace("_step200k", "")
                           for a in r["descending_by_margin"]))
    for a, r in q3["ratio_rule_split"].items():
        print(f"  RATIO {a:<24} mean={r['mean_ratio']:.4f} rho={r['rho']} "
              f"-> {'ACCEPT' if r['ratio_accept'] else 'REJECT'}")
    print()
    print("=" * W)
    print("VERDICT (`split`): is NI ever OBSERVED TO ACCEPT on these arms?")
    print("=" * W)
    for arm, v in verdict["split"].items():
        print(f"  {arm:<24} surviving={v['n_decision_axes_surviving_guard']} "
              f"accepting={v['n_decision_axes_accepting']} "
              f"{v['axes_accepting']} -> "
              f"{'ACCEPTS' if v['NI_OBSERVED_TO_ACCEPT'] else 'ALL REJECT'}")
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
