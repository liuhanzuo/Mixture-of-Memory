#!/usr/bin/env python3
"""Append ONE key to A04's STATUS.json: `shallow_ladder_2_0_2_disposition_20260814`.

WHY A SECOND KEY RATHER THAN AN EDIT TO THE FIRST.
`shallow_rung_ladder_20260813` was appended by
`a04_status_add_shallow_ladder_key.py` and pins the pre-registered analysis's
sha256. The §2.0.2 disposition + the per-range noise floors are a SEPARATE
artefact (`a04_shallow_ladder_neighbour_disposition.py`, run after the
admissibility document landed), so they get their own key. STATUS.json is
append-only: no existing key -- including the one added minutes earlier -- may be
rewritten.

THE APPEND-ONLY GUARANTEE IS ENFORCED AT THE TEXT LEVEL, verbatim the mechanism
in `a04_status_add_shallow_ladder_key.py`:

    THE NEW FILE IS A BYTE-PREFIX EXTENSION OF THE OLD FILE.

Every byte up to and including the last key's closing `  }` is preserved
literally, then `",\\n" + <new key> + "\\n}"` is appended. No existing byte is
re-serialised, so indent / key order / float formatting of existing keys CANNOT
change. This is the only check that catches a REFORMAT: on 2026-08-13 an
`indent=1` write against this `indent=2` file rewrote all 2,643 lines while every
per-key semantic check correctly passed.

NO HARDCODED KEY COUNT -- the relation `new == old + 1` is asserted and the
observed count recorded in the appended key.

CPU only. Refuses if the key already exists. Restores from backup on any failure.

Usage: a04_status_add_shallow_ladder_disposition_key.py STATUS.json DISPOSITION.json SHA256
"""
from __future__ import annotations

import json
import os
import shutil
import sys

NEW_KEY = "shallow_ladder_2_0_2_disposition_20260814"


def _f(x):
    try:
        return float(x)
    except Exception:
        return str(x)


def build_entry(disp_path, sha, old_n):
    D = json.load(open(disp_path))
    s = D["section_2_0_2_disposition"]
    rd = s["reconciliation_document"]
    rc = D["range_constants_used"]

    ranges = {}
    for grp, per_ax in D["range_disclosures"].items():
        any_ax = next(iter(per_ax))
        ranges[grp] = {
            "k": per_ax[any_ax]["k"],
            "c_k": per_ax[any_ax]["c_k"],
            "c_k_closed_form_expr": per_ax[any_ax]["c_k_closed_form_expr"],
            "k_matches_n_cells": per_ax[any_ax]["k_matches_n_cells"],
            "sigma_recipe": per_ax[any_ax]["sigma_recipe"],
            "NOT_DECISION_BEARING": True,
            "why_not_decision_bearing": per_ax[any_ax]["why_not_decision_bearing"],
            "per_axis": {ax: {"range_pp": v["range_pp"],
                              "noise_floor_pp": v["noise_floor_pp"],
                              "range_over_floor": v["range_over_floor"],
                              "CLEARS_ITS_OWN_FLOOR": v["CLEARS_ITS_OWN_FLOOR"],
                              "sigma_pp": v["sigma_pp"],
                              "if_wrong_c_k_had_been_used":
                                  v["if_wrong_c_k_had_been_used"]}
                         for ax, v in per_ax.items()},
        }

    return {
        "date": "2026-08-14",
        "gate": "A04_shallow_rung_ladder__GATE_DESIGN_2_0_2_disposition",
        "companion_to_key": "shallow_rung_ladder_20260813",
        "companion_to_evidence": D["companion_to"],
        "verdict": (
            "GATE_DESIGN 2.0.2 IS NOT TRIGGERED. It binds this ladder (it is scoped to "
            "'any NI(Delta) accept reported by this gate', and prereg 5.5 renounces a "
            "CLAIM while 2.0.2 imposes a DISCLOSURE duty -- renouncing a claim cannot "
            "discharge a duty), but it gates ACCEPTS ONLY and this pass has ZERO "
            "accepting decision-axis cells (Branch B). The precondition is NOT "
            "vacuously satisfied -- it is NOT TRIGGERED. No neighbour check was run "
            "because none was owed."),
        "gpu_h_spent": D["gpu_h"],
        "gpu_h_note": ("0 GPU. This artefact is float arithmetic over SEs already "
                       "computed by the pinned analysis: no RNG, no model load, no "
                       "CUDA context. It is therefore node-INDEPENDENT, unlike the "
                       "bootstrap (pinned to .73 for the numpy multinomial drift)."),
        "prereg": {
            "document": rd["file"], "commit": rd["commit"],
            "written_PRE_DATA": rd["written_PRE_DATA"],
            "pre_data_evidence": rd["pre_data_evidence"],
            "also_governed_by": ("A04_SHALLOW_RUNG_LADDER_PREREG.md 4.2 / 5.5 "
                                 "(commit a2e1a95) -- no range or spread statistic is "
                                 "decision-bearing in this pass"),
        },
        "verdict_doc": ("A04_SHALLOW_RUNG_LADDER_VERDICT.md 4.1 and 5.1, RENDERED from "
                        "the evidence JSONs by code/a04_render_shallow_ladder_verdict.py "
                        "-- no number in that .md is hand-transcribed"),
        "evidence": "evidence/a04_shallow_ladder_neighbour_disposition.json",
        "evidence_sha256": sha,
        "code": [
            "code/a04_shallow_ladder_neighbour_disposition.py",
            "code/a04_render_shallow_ladder_verdict.py (extended to render 4.1/5.1 "
            "from the companion; the original was committed at 21:41 on 2026-08-13, "
            "BEFORE the admissibility document landed at 22:09, so it could not)",
            "code/_a04_shallow_integrity_probe.py (independent shard-completeness "
            "witness, run BEFORE the analysis)",
            "code/a04_status_add_shallow_ladder_disposition_key.py",
        ],
        "key_count_note": (
            f"this key is the {old_n + 1}th; the file held {old_n} when it was "
            "appended. No count is hardcoded -- only the relation new == old+1, plus "
            "byte-prefix identity of the entire old file, which is the only check that "
            "catches a reformat."),
        "TRIGGERED": s["TRIGGERED"],
        "n_accepting_decision_axis_cells": s["n_accepting_decision_axis_cells"],
        "binds_this_ladder": s["binds_this_ladder"],
        "why_it_binds": s["why_it_binds"],
        "disposition": s["disposition"],
        "CERTIFIED_is_structurally_unreachable":
            s["CERTIFIED_is_structurally_unreachable"],
        "label_rule_evaluated_not_skipped": {
            "accepting_cells_per_arm": s["accepting_cells_per_arm"],
            "label_rule_output": s["label_rule_evaluated_per_accepting_cell"],
            "why_empty_is_correct": s["why_the_empty_dict_is_the_correct_output"],
        },
        "neighbour_inventory_per_arm": s["neighbour_inventory_per_arm"],
        "lower_neighbour_exists_but_was_NOT_scored": (
            "step2500.pt is on disk for BOTH arms (keep13 17,013,823,232 B; keep14 "
            "17,819,242,212 B) and NEITHER was scored -- no step2500 eval dir exists on "
            "either disk, and the admissibility document 6.5 authorises no GPU for one. "
            "Per 6.5 that is the WEAKER of the two available disclosures and it is the "
            "honest one, and WHICH one applies is decided by whether the eval was run, "
            "NOT by what the step5000 numbers turned out to be. Under Branch B the "
            "disclosure is not owed at all."),
        "upper_neighbour_cannot_exist": (
            "max_steps=5000 and the save condition is `step % save_every == 0 and step "
            "> 0` plus a terminal _save(..., final=True). final.pt is named by "
            "\"final\" if final else f\"step{step}\" at the SAME step value, so it is "
            "the SAME point as step5000.pt, not a checkpoint beyond it. Hence CERTIFIED "
            "is structurally unreachable for this ladder -- decided PRE-DATA, no datum "
            "can change it."),
        "2500_steps_is_NOT_a_neighbourhood": s["2500_steps_is_NOT_a_neighbourhood"],
        "one_process_provenance_no_resume_seam":
            s["one_process_provenance_no_resume_seam"],
        "range_constants_used": rc,
        "range_disclosures_each_against_ITS_OWN_floor": ranges,
        "no_ratio_of_ranges_is_formed": D["no_ratio_of_ranges_is_formed"],
        "nothing_here_is_decision_bearing": D["nothing_here_is_decision_bearing"],
        "generalisable_lesson": (
            "a range that clears an ITEM-noise floor is NOT thereby resolved against "
            "RUN-TO-RUN variance. All 8 ranges here clear their own floors, but there "
            "is ONE seed (101) per arm, so no sd_run exists at these rungs and none of "
            "them may be read as a seed-robust effect. The floors are stated so that "
            "the reverse error -- quoting a sub-floor range as 'a direction' -- is also "
            "closed off. Each range records its own k AND what the wrong c_k would have "
            "done, because c_3 at k=8 makes a floor 40.6 % too low (that manufactured a "
            "finding once) and c_3 at k=2 inflates one by 50.0 %."),
    }


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} STATUS.json DISPOSITION.json SHA256")
    status_path, disp_path, sha = sys.argv[1], sys.argv[2], sys.argv[3]

    old_bytes = open(status_path, "rb").read()
    old = json.loads(old_bytes.decode("utf-8"))
    old_keys = list(old.keys())
    old_n = len(old_keys)
    if NEW_KEY in old:
        raise SystemExit(f"FATAL: key {NEW_KEY} already present; refusing.")
    print(f"[status] observed {old_n} existing keys (NO count is hardcoded)")

    entry = build_entry(disp_path, sha, old_n)

    txt = old_bytes.decode("utf-8")
    if not txt.rstrip().endswith("}"):
        raise SystemExit("FATAL: STATUS.json does not end with '}'")
    i_root = txt.rindex("}")
    i_last_key_close = txt.rindex("}", 0, i_root)
    prefix = txt[:i_last_key_close + 1]

    body = json.dumps({NEW_KEY: entry}, indent=2, ensure_ascii=False, default=_f)
    inner = body[body.index("\n") + 1: body.rindex("\n")]
    new_bytes = (prefix + ",\n" + inner + "\n}").encode("utf-8")

    pb = prefix.encode("utf-8")
    if new_bytes[:len(pb)] != pb:
        raise SystemExit("FATAL: byte prefix changed; refusing to write.")

    bak = status_path + ".bak_shallow_disposition"
    shutil.copy2(status_path, bak)
    tmp = status_path + ".tmp_shallow_disposition"
    with open(tmp, "wb") as f:
        f.write(new_bytes)

    def fail(msg):
        os.remove(tmp)
        shutil.copy2(bak, status_path)
        raise SystemExit(f"FATAL: {msg} -- restored from {bak}")

    new = json.loads(open(tmp, encoding="utf-8").read())
    if len(new) != old_n + 1:
        fail(f"new key count {len(new)} != {old_n + 1}")
    for k in old_keys:
        if new[k] != old[k]:
            fail(f"existing key {k} changed value")
    if list(new.keys()) != old_keys + [NEW_KEY]:
        fail("key order changed or the new key is not last")

    os.replace(tmp, status_path)
    final = open(status_path, "rb").read()
    if final[:len(pb)] != pb:
        shutil.copy2(bak, status_path)
        raise SystemExit("FATAL: post-write byte-prefix check failed; restored.")

    print(f"[status] appended {NEW_KEY}: {old_n} -> {len(new)} keys")
    print(f"[status] byte-prefix identity held over {len(pb)} of {len(old_bytes)} "
          f"old bytes ({100.0*len(pb)/len(old_bytes):.4f}%)")
    print(f"[status] backup at {bak}")
    print(f"[status] TRIGGERED = {new[NEW_KEY]['TRIGGERED']}, "
          f"gpu_h = {new[NEW_KEY]['gpu_h_spent']}")


if __name__ == "__main__":
    main()
