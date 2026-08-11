#!/usr/bin/env python3
"""Recompute A03 CPT-trajectory paired-diff bootstrap CIs from per-item shards.

Covers Arm 3 / Arm 4 / Arm 6 (4 dose points each) plus the data-order
replication arms (seed 43/44/45, step220000 only -- see DATAORDER_PREREG.md).

The +0.48pp SIG triviaqa headline lived only in two .md files; the one bootstrap
JSON on .82:/tmp held MMLU for step205/210k ONLY and never contained triviaqa at
all. This regenerates every cell from the 8-shard per-item records and writes a
persistent evidence JSON.

Protocol matches what the verdict files claim: per-item paired difference,
bootstrap n_boot=5000, seed=42, CI95 percentile. SIG = CI excludes 0.
UNCHANGED by the data-order extension -- do not retune.

Shard integrity: a result dir that is ABSENT yields a {"pending": ...} cell (the
arm simply has not run yet). A result dir that EXISTS but holds fewer than 8
shards is a hard SystemExit for both closedbook and MMLU. That distinction is
load-bearing: a silently-merged 5/8 shard set has ruined results in this repo
before.

MMLU loader defect, fixed 2026-08-10 (see load_mmlu docstring): the previous
`load_mmlu` guessed FLAT key names (`letter_correct` / `content_norm_correct`
/ `em`) that the eval harness never writes -- the real records are NESTED
(`letter.correct`, `content_norm.correct`). Every lookup fell through to a None
default and the caller's `is not None` guard then dropped the cell with no
marker, so ALL 12 MMLU cells (arm3/arm4/arm6 x 4 dose points) were missing from
the canonical evidence JSON while four .md files asserted MMLU was flat. The
loader now reads the nested keys and hard-fails on any schema surprise.

LOADERS RELOCATED 2026-08-11 -- no behavioural change.
`load_cb` / `load_mmlu` / `paired` / `NotRunYet` and the ROOT/CB/MM/N_BOOT/SEED/
NSHARD/N_MMLU constants now live in `proposal/shared/code/canonical_eval_loaders.py`
and are imported below. Their bodies were moved BYTE-FOR-BYTE; the assertions
(8/8 shards, exact item count, duplicate item_id, nan rejection) and the
bootstrap protocol (n_boot=5000, seed=42, CI95 percentile) are unchanged, and the
A04 Stage-A/Stage-B verdict JSONs were re-derived after the move and compared
field-by-field to the archived copies before the A03 directory was moved. The
reason for the lift: A03 is archived, while A04's numbers and this script's own
seed-45 recompute both depend on these loaders -- leaving them inside an archived
proposal is what blocked the move. A04 also used to obtain them by reading THIS
file's source text and exec-ing everything before the `BASE = ` line (this module
has no __main__ guard); that textual coupling is now gone.
"""
import json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared" / "code"))
from canonical_eval_loaders import (  # noqa: E402
    CB, MM, N_BOOT, N_MMLU, NSHARD, ROOT, SEED,
    NotRunYet, load_cb, load_mmlu, paired,
)


BASE = "A03_1B_keep7_step200k"
ARMS = {
    "arm3_cosine_tail": [("step205000", "A03_1B_arm3_cpt_step205000"),
                         ("step210000", "A03_1B_arm3_cpt_step210000"),
                         ("step215000", "A03_1B_arm3_cpt_step215000"),
                         ("step220000", "A03_1B_arm3_cpt_step220000")],
    "arm4_peaklr":      [("step205000", "A03_1B_arm4_peaklr_step205000"),
                         ("step210000", "A03_1B_arm4_peaklr_step210000"),
                         ("step215000", "A03_1B_arm4_peaklr_step215000"),
                         ("step220000", "A03_1B_arm4_peaklr_step220000")],
    "arm6_lowerband":   [("step205000", "A03_1B_arm6_lowerband_step205000"),
                         ("step210000", "A03_1B_arm6_lowerband_step210000"),
                         ("step215000", "A03_1B_arm6_lowerband_step215000"),
                         ("step220000", "A03_1B_arm6_lowerband_step220000")],
    # --- data-order replication (DATAORDER_PREREG.md) -----------------------
    # Arm 3's EXACT config, varying only --seed, with the ce5c298
    # `seed=args.seed` fix now present on zwfy6 so --seed actually reaches
    # DistributedSampler. step220000 ONLY -- the single pre-registered decision
    # point. Cells stay absent (not zero, not partial) until each run lands.
    "dataorder_seed43": [("step220000", "A03_1B_dataorder_seed43_step220000")],
    "dataorder_seed44": [("step220000", "A03_1B_dataorder_seed44_step220000")],
    "dataorder_seed45": [("step220000", "A03_1B_dataorder_seed45_step220000")],
}

out = {
    "protocol": f"per-item paired diff bootstrap n_boot={N_BOOT} seed={SEED}, CI95 percentile; SIG = CI excludes 0",
    "baseline": BASE,
    "regenerated": "2026-08-10 from 8/8 per-item shards. Supersedes (a) the volatile "
                   ".82:/tmp/a03_arm3_cpt_trajectory_paired.json (md5 37149d4d59bf941c1dbc05f17260f0b2), "
                   "which held MMLU step205/210k only and never contained triviaqa, and "
                   "(b) evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json (2026-08-09), "
                   "which had NO mmlu key on any of its 12 cells because load_mmlu read "
                   "flat key names the harness never writes. Closed-book cells are "
                   "byte-identical to (b); the mmlu axis is newly recovered.",
    "shard_integrity": f"every cell asserts {NSHARD}/{NSHARD} shards; script exits non-zero otherwise",
    "mmlu_axis": f"letter + content_norm, n={N_MMLU} asserted per arm, nan rows rejected; "
                 "recovered 2026-08-10 after a silent loader defect had dropped it entirely",
    "arms": {},
}

# Every cell must carry every axis. The defect this script was fixed for was a
# MISSING key, not a wrong number, so absence is what gets asserted.
EXPECTED_AXES = ("popqa", "triviaqa", "nq_open", "mmlu")

for arm, ckpts in ARMS.items():
    out["arms"][arm] = {}
    for label, d in ckpts:
        cell = {}
        for task in ("popqa", "triviaqa", "nq_open"):
            src = d if task != "nq_open" else d + "_nq"
            try:
                bb = load_cb(BASE if task != "nq_open" else BASE + "_nq", task)
                aa = load_cb(src, task)
            except NotRunYet as e:
                cell[task] = {"pending": str(e)}
                continue
            except SystemExit as e:
                cell[task] = {"error": str(e)}
                continue
            idx = sorted(set(bb) & set(aa))
            if not idx:
                cell[task] = {"error": "no overlapping item_ids"}
                continue
            cell[task] = {
                "em":       paired({i: bb[i][0] for i in idx}, {i: aa[i][0] for i in idx}, idx),
                "contains": paired({i: bb[i][1] for i in idx}, {i: aa[i][1] for i in idx}, idx),
                "f1":       paired({i: bb[i][2] for i in idx}, {i: aa[i][2] for i in idx}, idx),
            }
        # --- MMLU (letter + content_norm) -----------------------------------
        # Both interfaces are ALWAYS emitted when the dirs exist. The old code
        # wrapped these in `if idx and all(... is not None ...)` guards, which
        # -- combined with the None-filled loader -- deleted the cell key
        # entirely and left no trace in the JSON. Any failure now either raises
        # (load_mmlu) or lands as an explicit "pending"/"error" marker.
        try:
            mb, ma = load_mmlu(BASE), load_mmlu(d)
        except NotRunYet as e:
            cell["mmlu"] = {"pending": str(e)}
        except SystemExit as e:
            cell["mmlu"] = {"error": str(e)}
        else:
            idx = sorted(set(mb) & set(ma))
            if len(idx) != N_MMLU:
                raise SystemExit(
                    f"FATAL {d}/mmlu: only {len(idx)}/{N_MMLU} item_ids overlap the "
                    f"baseline {BASE} -- paired MMLU requires the identical item set")
            cell["mmlu"] = {
                "letter": paired({i: mb[i][0] for i in idx},
                                 {i: ma[i][0] for i in idx}, idx),
                "content_norm": paired({i: mb[i][1] for i in idx},
                                       {i: ma[i][1] for i in idx}, idx),
            }
        out["arms"][arm][label] = cell
        missing = [a for a in EXPECTED_AXES if a not in cell]
        if missing:
            raise SystemExit(f"FATAL {arm}/{label}: axes {missing} produced NO key at all. "
                             "This is exactly the 2026-08-10 mmlu defect -- an axis must "
                             "always land as a result, a 'pending', or an 'error'.")

dest = sys.argv[1] if len(sys.argv) > 1 else "/tmp/a03_cpt_trajectory_paired_full.json"
Path(dest).write_text(json.dumps(out, indent=2))
print(f"wrote {dest}")
for arm, cks in out["arms"].items():
    for label, cell in cks.items():
        for task, m in cell.items():
            if "pending" in m:
                print(f"  {arm} {label} {task}: PENDING ({m['pending']})")
                continue
            if "error" in m:
                print(f"  {arm} {label} {task}: {m['error'][:70]}")
                continue
            bits = " ".join(
                f"{k}={v['delta_pp']:+.2f}{'*' if v['verdict']=='SIG' else ''}"
                for k, v in m.items())
            print(f"  {arm} {label} {task}: {bits}")
