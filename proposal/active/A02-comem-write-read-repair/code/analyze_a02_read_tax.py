#!/usr/bin/env python
"""A02 READ-TAX analyzer — the read-out PREREG §2.6 specifies.  Pure CPU.

WHAT THIS COMPUTES
------------------
The read tax at depth j = the PAIRED per-cell delta of each arm vs **A0**
(j=0, no adapter — which GATE 0 established *is* the optimally-distilled j=0
adapter, because at j=0 teacher==student so the distillation optimum is the
identity).

    arm  j   adapter                              params
    A0   0   none (= optimal j=0 adapter)          0        <- the ANCHOR
    A1   0   a02_j0control_lora_r32               87.29 M   <- literal control
    A2   6   qcmem_distill_qwen_j6_r32_4k         72.74 M
    A3   9   qcmem_distill_qwen_j9_r32_4k         65.47 M
    A4  12   qcmem_distill_qwen_j12_r32_4k        58.20 M   <- flagship
    A5  18   qcmem_distill_qwen_j18_r32_4k        43.65 M
    A6  12   a02_j12_capmatch_r40                 72.74 M   <- cap-match to A2

PRIMARY = RULER (niah_multikey_1, variable_tracking x 16k/32k). dvr measured
retrieval recall@12 = 99-100 % there, so retrieval is CLOSED and the delta is
attributable to the depth/adapter axis.

SECONDARY = BABILong (qa1/qa2/qa5 x 16k/32k), CONTRAST ONLY. dvr measured those
cells at recall@12 = 22.9-63.2 %, i.e. retrieval-DOMINATED, so they cannot
support depth inference. Reported per-cell to show the curve is interpretable
only where retrieval is closed.

AGGREGATION HYGIENE (PREREG §2.6): per-cell ONLY. This script contains no
pooled-accuracy computation across BABILong/LongEval cells; the banned pooled
figures (-17.89 pp / +2.00 pp) are not computable from its output by design.
A single cross-cell mean is emitted for RULER ONLY, explicitly labelled as the
mean over the 4 retrieval-closed cells, never mixed with BABILong.

CANONICAL SCORERS ARE IMPORTED, NEVER REIMPLEMENTED (PREREG GATE E). This module
imports the dvr analyzer's own loaders, which in turn import:
  * BABILong : babilong.metrics.{TASK_LABELS, compare_answers}
  * RULER    : the per-item `correct` the harness wrote into *.records.json
  * CI       : paired-difference bootstrap n_boot=5000 seed=42 (A03 protocol)

FAIL-CLOSED GATES
  GATE C shard completeness: a cell is refused unless EVERY arm has exactly
     `nshard` shards. No silent 5-of-8 merge. Additionally asserts per-cell
     n == 100, zero duplicate sample ids, and zero NaN/missing `correct`.
  GATE C2 pairing: RULER asserts input_ids_sha256 equality across all arms for
     every shared sample index (so the paired delta is truly paired).
  GATE D chat_template=False asserted in every emitted cell config, plus
     selector=iter_bm25 / topk=12 / chunk_size=512 / the expected resume_j and
     the expected adapter path per arm.

Usage:
  python analyze_a02_read_tax.py --out <evidence_dir>
  python analyze_a02_read_tax.py --selftest_gate
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get(
    "A02_BASE", "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"))
_CODE = BASE / "proposal/active/A02-comem-write-read-repair/code"
for _p in (str(BASE), str(_CODE), str(BASE / "scripts"),
           str(BASE / "third_party" / "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the dvr analyzer VERBATIM for loading + scoring + CI. Nothing is
# reimplemented here; this module only adds the A0-anchored delta table.
import analyze_a02_depth_vs_retrieval as dvr  # noqa: E402

bab_cell_items = dvr.bab_cell_items
bab_cell_cfg = dvr.bab_cell_cfg
rul_cell_items = dvr.rul_cell_items
bootstrap_diff_ci = dvr.bootstrap_diff_ci
wilson_ci = dvr.wilson_ci
sig = dvr.sig

NSHARD = 8
TOPK = 12
CHUNK = 512
N_EXPECT = 100          # --limit 100 per cell

# arm -> (ruler_subdir, babilong_subdir, resume_j, adapter_substr_or_None, params_M)
ARMS = {
    "A0": ("a02_dvr_ruler_j0_top12", "a02_dvr_babilong_j0_top12",
           0, None, 0.0),
    "A1": ("a02_rtax_ruler_A1_j0control", "a02_rtax_babilong_A1_j0control",
           0, "outputs/a02_j0control_lora_r32/final", 87.29),
    "A2": ("a02_rtax_ruler_A2_j6", "a02_rtax_babilong_A2_j6",
           6, "outputs/qcmem_distill_qwen_j6_r32_4k/final", 72.74),
    "A3": ("a02_rtax_ruler_A3_j9", "a02_rtax_babilong_A3_j9",
           9, "outputs/qcmem_distill_qwen_j9_r32_4k/final", 65.47),
    "A4": ("a02_ruler_c2_j12_readlora", "a02_babilong_c2_j12_readlora",
           12, "outputs/qcmem_distill_qwen_j12_r32_4k/final", 58.20),
    "A5": ("a02_rtax_ruler_A5_j18", "a02_rtax_babilong_A5_j18",
           18, "outputs/qcmem_distill_qwen_j18_r32_4k/final", 43.65),
    "A6": ("a02_rtax_ruler_A6_j12_r40", "a02_rtax_babilong_A6_j12_r40",
           12, "outputs/a02_j12_capmatch_r40/final", 72.74),
}
ARM_ORDER = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
ANCHOR = "A0"

RUL_TASKS = ("niah_multikey_1", "variable_tracking")
RUL_LENS = ("16k", "32k")
BAB_TASKS = ("qa1", "qa2", "qa5")
BAB_LENS = ("16k", "32k")


# ------------------------------------------------------------ GATE D cfg --- #
def check_cfg_ruler(arm, cfg, errs, where, summary=None):
    """GATE D for a RULER cell.

    NOTE on where each field lives (verified against the on-disk artefacts, not
    assumed): the per-shard `*.records.json` stores the QCMem config FLAT at the
    top level (resume_j/selector/topk/iter_hop_topk/chunk_size/lora_adapter/
    baseline/seed) and does NOT carry chat_template. `chat_template` and
    `enable_thinking` live in the sibling summary `*.json`, nested alongside a
    `qcmem` sub-dict. So GATE D reads the flat records cfg for the retrieval/
    depth/adapter identity and the summary for the chat-template invariant.
    """
    if not cfg:
        errs.append(f"{where}/{arm}: no records cfg")
        return
    j, want_lora = ARMS[arm][2], ARMS[arm][3]
    if summary is not None:
        if not summary:
            errs.append(f"{where}/{arm}: no summary json (cannot verify chat_template)")
        else:
            if summary.get("chat_template") is not False:
                errs.append(f"{where}/{arm}: chat_template="
                            f"{summary.get('chat_template')!r} must be False")
            if summary.get("enable_thinking") is not False:
                errs.append(f"{where}/{arm}: enable_thinking="
                            f"{summary.get('enable_thinking')!r} must be False")
    q = cfg
    if q.get("selector") != "iter_bm25":
        errs.append(f"{where}/{arm}: selector={q.get('selector')!r} != iter_bm25")
    if q.get("topk") != TOPK:
        errs.append(f"{where}/{arm}: topk={q.get('topk')} != {TOPK}")
    if q.get("chunk_size") != CHUNK:
        errs.append(f"{where}/{arm}: chunk_size={q.get('chunk_size')} != {CHUNK}")
    if q.get("resume_j") != j:
        errs.append(f"{where}/{arm}: resume_j={q.get('resume_j')} != {j}")
    if cfg.get("baseline") not in (None, "none"):
        errs.append(f"{where}/{arm}: baseline={cfg.get('baseline')!r} != none")
    if cfg.get("no_retrieval") is True:
        errs.append(f"{where}/{arm}: no_retrieval=True (must retrieve)")
    lora = q.get("lora_adapter")
    if want_lora is None:
        if lora:
            errs.append(f"{where}/{arm}: expected NO adapter, got {lora!r}")
    else:
        if not lora or want_lora not in str(lora):
            errs.append(f"{where}/{arm}: expected adapter {want_lora!r}, got {lora!r}")


def check_cfg_bab(arm, cfg, errs, where):
    """GATE D for a BABILong cell config (per-shard json).

    Structure verified on disk (not assumed): QCMem config is nested under
    `qcmem`, and `chat_template` is nested under `prompt` (NOT top level as in
    the RULER summary json). Reading it from the wrong level would silently
    "pass" the chat-template invariant on a None, so both are pinned here.
    """
    if not cfg:
        errs.append(f"{where}/{arm}: no cell config json")
        return
    j, want_lora = ARMS[arm][2], ARMS[arm][3]
    q = cfg.get("qcmem", cfg) or {}
    ct = (cfg.get("prompt") or {}).get("chat_template", cfg.get("chat_template"))
    if ct is not False:
        errs.append(f"{where}/{arm}: chat_template={ct!r} must be False")
    if cfg.get("no_retrieval") is True:
        errs.append(f"{where}/{arm}: no_retrieval=True (must retrieve)")
    if cfg.get("baseline") not in (None, "none"):
        errs.append(f"{where}/{arm}: baseline={cfg.get('baseline')!r} != none")
    if q.get("selector") != "iter_bm25":
        errs.append(f"{where}/{arm}: selector={q.get('selector')!r} != iter_bm25")
    if q.get("topk") != TOPK:
        errs.append(f"{where}/{arm}: topk={q.get('topk')} != {TOPK}")
    if q.get("chunk_size") != CHUNK:
        errs.append(f"{where}/{arm}: chunk_size={q.get('chunk_size')} != {CHUNK}")
    if q.get("resume_j") != j:
        errs.append(f"{where}/{arm}: resume_j={q.get('resume_j')} != {j}")
    lora = q.get("lora_adapter")
    if want_lora is None:
        if lora:
            errs.append(f"{where}/{arm}: expected NO adapter, got {lora!r}")
    else:
        if not lora or want_lora not in str(lora):
            errs.append(f"{where}/{arm}: expected adapter {want_lora!r}, got {lora!r}")


# ------------------------------------------------------------ GATE C ------- #
def assert_cell_integrity(arm, items, where, errs, n_expect=N_EXPECT):
    """n == n_expect, no dup ids (dict keys are unique by construction, so we
    additionally verify the raw shard files contribute exactly n_expect rows),
    no NaN/None `correct`."""
    n = len(items)
    if n != n_expect:
        errs.append(f"{where}/{arm}: n={n} != expected {n_expect}")
    bad = [k for k, v in items.items()
           if v[0] is None or (isinstance(v[0], float) and np.isnan(v[0]))
           or int(v[0]) not in (0, 1)]
    if bad:
        errs.append(f"{where}/{arm}: {len(bad)} items with non-binary/NaN correct")
    return n == n_expect and not bad


def rul_cell_summary(arm_dir: Path, task: str, length: str, nshard=NSHARD):
    """The sibling summary json, which is where chat_template/enable_thinking live."""
    fs = sorted(glob.glob(str(arm_dir / f"{task}_{length}_shard0of{nshard}.json")))
    if not fs:
        return {}
    return json.load(open(fs[0]))


def count_shard_rows_ruler(arm_dir, task, length, nshard=NSHARD):
    """Independent duplicate check: sum of per-shard record counts must equal
    the size of the merged dict (i.e. no sample_index appeared twice)."""
    files = sorted(glob.glob(str(arm_dir / f"{task}_{length}_shard*of{nshard}.records.json")))
    total, ids = 0, []
    for fp in files:
        recs = json.load(open(fp)).get("records", [])
        total += len(recs)
        ids += [int(r["sample_index"]) for r in recs]
    return total, len(set(ids)), len(files)


# ------------------------------------------------------------- analysis ---- #
def paired_table(items_by_arm, arms_present):
    """Per-cell accuracy for each arm + paired delta vs ANCHOR, on the exact
    intersection of sample indices present in ALL arms of this cell."""
    common = None
    for a in arms_present:
        ks = set(items_by_arm[a].keys())
        common = ks if common is None else (common & ks)
    common = sorted(common or [])
    vec = {a: [int(items_by_arm[a][i][0]) for i in common] for a in arms_present}
    out = {"n_paired": len(common), "acc": {}, "delta_vs_anchor": {}}
    for a in arms_present:
        k = int(sum(vec[a]))
        p, lo, hi = wilson_ci(k, len(common))
        out["acc"][a] = {"acc_pct": p, "k": k, "n": len(common),
                         "wilson_lo": lo, "wilson_hi": hi}
    if ANCHOR in arms_present:
        for a in arms_present:
            if a == ANCHOR:
                continue
            d, lo, hi, n = bootstrap_diff_ci(vec[ANCHOR], vec[a])
            out["delta_vs_anchor"][a] = {
                "delta_pp": round(100 * d, 2),
                "ci95_lo_pp": round(100 * lo, 2),
                "ci95_hi_pp": round(100 * hi, 2),
                "n_paired": n,
                "sig": sig(100 * lo, 100 * hi),
            }
    out["_vectors"] = vec
    out["_common_idx"] = common
    return out


def analyze(out_dir: Path, nshard=NSHARD):
    errs, refused = [], []
    ruler_cells, bab_cells = {}, {}
    sha_mismatch = []

    # ------------------------------- PRIMARY: RULER ------------------------ #
    for task in RUL_TASKS:
        for length in RUL_LENS:
            key = f"ruler|{task}|{length}"
            items, cfgs, bad = {}, {}, None
            for arm in ARM_ORDER:
                d = BASE / "ruler_results" / ARMS[arm][0]
                it, cfg, e = rul_cell_items(d, task, length, nshard)
                if e:
                    bad = f"{arm}: {e}"
                    break
                tot, uniq, nf = count_shard_rows_ruler(d, task, length, nshard)
                if nf != nshard:
                    bad = f"{arm}: GATE_C_SHARDFILES {nf}/{nshard}"
                    break
                if tot != uniq:
                    bad = f"{arm}: GATE_C_DUPLICATE_IDS {tot} rows / {uniq} unique"
                    break
                if tot != len(it):
                    bad = f"{arm}: GATE_C_MERGE_LOSS {tot} rows -> {len(it)} merged"
                    break
                items[arm] = it
                cfgs[arm] = cfg
                check_cfg_ruler(arm, cfg, errs, key,
                                summary=rul_cell_summary(d, task, length, nshard))
                assert_cell_integrity(arm, it, key, errs)
            if bad:
                refused.append({"cell": key, "reason": bad})
                continue
            # GATE C2: sha256 pairing across arms
            common = sorted(set.intersection(*[set(items[a]) for a in ARM_ORDER]))
            for i in common:
                shas = {a: items[a][i][1] for a in ARM_ORDER if items[a][i][1]}
                if len(set(shas.values())) > 1:
                    sha_mismatch.append({"cell": key, "sample_index": i, **shas})
            tab = paired_table(items, ARM_ORDER)
            ruler_cells[key] = tab

    # ------------------------------ SECONDARY: BABILong -------------------- #
    for task in BAB_TASKS:
        for length in BAB_LENS:
            key = f"babilong|{task}|{length}"
            items, bad = {}, None
            for arm in ARM_ORDER:
                d = BASE / "babilong_results" / ARMS[arm][1]
                it, e = bab_cell_items(d, task, length, nshard)
                if e:
                    bad = f"{arm}: {e}"
                    break
                items[arm] = it
                check_cfg_bab(arm, bab_cell_cfg(d, task, length), errs, key)
                assert_cell_integrity(arm, it, key, errs)
            if bad:
                refused.append({"cell": key, "reason": bad})
                continue
            # G2 pairing: identical (question,target) across arms
            common = sorted(set.intersection(*[set(items[a]) for a in ARM_ORDER]))
            for i in common:
                qt = {(items[a][i][1], items[a][i][2]) for a in ARM_ORDER}
                if len(qt) > 1:
                    errs.append(f"{key}: sample {i} question/target differs across arms")
                    break
            bab_cells[key] = paired_table(items, ARM_ORDER)

    # -------- RULER-ONLY cross-cell mean (explicitly NOT a pooled BABILong) - #
    ruler_mean = {}
    if ruler_cells:
        for arm in ARM_ORDER:
            accs = [c["acc"][arm]["acc_pct"] for c in ruler_cells.values()
                    if arm in c["acc"] and c["acc"][arm]["acc_pct"] is not None]
            ruler_mean[arm] = round(float(np.mean(accs)), 2) if accs else None
        # macro delta = mean over cells of the per-cell paired delta
        ruler_mean_delta = {}
        for arm in ARM_ORDER:
            if arm == ANCHOR:
                continue
            ds = [c["delta_vs_anchor"][arm]["delta_pp"] for c in ruler_cells.values()
                  if arm in c.get("delta_vs_anchor", {})]
            ruler_mean_delta[arm] = round(float(np.mean(ds)), 2) if ds else None
    else:
        ruler_mean_delta = {}

    # ------------------------------- monotonicity (pred 3) ----------------- #
    # depth ladder at matched r=32: A0(j0,noada) A2(j6) A3(j9) A4(j12) A5(j18)
    ladder = ["A0", "A2", "A3", "A4", "A5"]
    mono = {}
    if ruler_cells:
        seq = [(ARMS[a][2], ruler_mean.get(a)) for a in ladder
               if ruler_mean.get(a) is not None]
        if len(seq) >= 3:
            js = [s[0] for s in seq]
            vs = [s[1] for s in seq]
            diffs = [round(vs[i + 1] - vs[i], 2) for i in range(len(vs) - 1)]
            rho = float(np.corrcoef(js, vs)[0, 1]) if len(set(vs)) > 1 else None
            mono = {"ladder_j": js, "ruler_mean_acc": vs,
                    "successive_diffs_pp": diffs,
                    "n_nonincreasing_steps": sum(1 for d in diffs if d <= 0),
                    "n_steps": len(diffs),
                    "pearson_r_acc_vs_j": round(rho, 4) if rho is not None else None,
                    "note": "monotone-ish == accuracy non-increasing as j grows "
                            "(tax grows with depth); r<0 is the predicted sign"}

    out = {
        "gate_status": {
            "GATE_C_shard_completeness_and_counts": "PASS" if not refused else "REFUSED_CELLS",
            "GATE_C2_ruler_sha_pairing": "PASS" if not sha_mismatch else "FAIL",
            "GATE_D_config_identity": "PASS" if not errs else "FAIL",
            "GATE_E_scorers": "imported from analyze_a02_depth_vs_retrieval "
                              "(babilong.metrics.compare_answers / RULER harness "
                              "per-item correct); nothing reimplemented",
            "n_expected_per_cell": N_EXPECT,
            "nshard": nshard,
        },
        "config_errors": errs,
        "refused_cells": refused,
        "ruler_sha_pairing_failures": sha_mismatch,
        "arms": {a: {"resume_j": ARMS[a][2], "adapter": ARMS[a][3],
                     "lora_params_M": ARMS[a][4],
                     "ruler_dir": ARMS[a][0], "babilong_dir": ARMS[a][1]}
                 for a in ARM_ORDER},
        "anchor": ANCHOR,
        "PRIMARY_ruler_per_cell": ruler_cells,
        "PRIMARY_ruler_mean_over_4_retrieval_closed_cells": ruler_mean,
        "PRIMARY_ruler_mean_delta_vs_anchor_pp": ruler_mean_delta,
        "monotonicity_pred3": mono,
        "SECONDARY_babilong_per_cell_CONTRAST_ONLY": bab_cells,
        "aggregation_hygiene": (
            "PER-CELL ONLY. No pooled BABILong/LongEval figure is computed here; "
            "the only cross-cell mean is over the 4 RULER retrieval-closed cells "
            "and is labelled as such. The banned pooled numbers (-17.89 pp / "
            "+2.00 pp) are not derivable from this output."),
    }
    # strip bulky per-item vectors from the emitted JSON but keep them separately
    vectors = {"ruler": {}, "babilong": {}}
    for k, v in ruler_cells.items():
        vectors["ruler"][k] = {"common_idx": v.pop("_common_idx"),
                               "per_arm_correct": v.pop("_vectors")}
    for k, v in bab_cells.items():
        vectors["babilong"][k] = {"common_idx": v.pop("_common_idx"),
                                  "per_arm_correct": v.pop("_vectors")}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a02_read_tax_ruler.json").write_text(json.dumps(out, indent=1))
    (out_dir / "a02_read_tax_per_item_vectors.json").write_text(
        json.dumps(vectors, indent=1))
    return out


def selftest_gate():
    """Negative test: delete one shard from a scratch copy and confirm GATE C
    refuses the cell instead of merging 7/8."""
    src = BASE / "ruler_results" / ARMS["A0"][0]
    tmp = Path(tempfile.mkdtemp())
    dst = tmp / "arm"
    shutil.copytree(src, dst)
    it, cfg, e = rul_cell_items(dst, "niah_multikey_1", "16k", NSHARD)
    assert e is None, f"clean copy should load, got {e}"
    print(f"SELFTEST clean copy: n={len(it)} err={e}")
    victim = sorted(glob.glob(str(dst / "niah_multikey_1_16k_shard*of8.records.json")))[0]
    os.remove(victim)
    it2, cfg2, e2 = rul_cell_items(dst, "niah_multikey_1", "16k", NSHARD)
    assert e2 and "SHARD_INCOMPLETE" in e2, f"GATE C did NOT fire: {e2}"
    print(f"SELFTEST after deleting 1 shard: GATE C FIRED -> {e2}")
    shutil.rmtree(tmp)
    print("SELFTEST PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default=str(BASE / "proposal/active/A02-comem-write-read-repair"
                                      "/evidence/read_tax_ruler"))
    ap.add_argument("--nshard", type=int, default=NSHARD)
    ap.add_argument("--selftest_gate", action="store_true")
    a = ap.parse_args()
    if a.selftest_gate:
        selftest_gate()
        sys.exit(0)
    res = analyze(Path(a.out), a.nshard)
    print(json.dumps({
        "gates": res["gate_status"],
        "config_errors": res["config_errors"][:20],
        "refused": res["refused_cells"],
        "ruler_mean": res["PRIMARY_ruler_mean_over_4_retrieval_closed_cells"],
        "ruler_mean_delta_vs_A0": res["PRIMARY_ruler_mean_delta_vs_anchor_pp"],
        "mono": res["monotonicity_pred3"],
    }, indent=1))
