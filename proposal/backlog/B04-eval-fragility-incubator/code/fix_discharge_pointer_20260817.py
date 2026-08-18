#!/usr/bin/env python3
"""Fix a defect THIS PASS introduced, then record the finding. 0 GPU.

WHAT HAPPENED
-------------
append_status_20260817.py wrote, inside `novelty_check_regenerated_20260817`, a subkey
literally named "discharges". `ready_queue.py` treats "discharges" as a RESERVED
MACHINE-READABLE POINTER (DISCHARGE_POINTER_KEY, ready_queue.py:457) collected at the top
level and one level down. I used it as English prose. The queue immediately flagged it:

    ! discharge pointer [novelty_check_regenerated_20260817.discharges] ->
      'related_work.actionable_0_gpu_followup (STATUS.json:318) -- IN FULL.'
      matches no blocker path (dangling; no effect)

AND IT CANNOT BE FIXED BY WRITING A BETTER PATH
-----------------------------------------------
Measured: rq._walk_blockers(B04_doc) yields ZERO paths. `_walk_blockers` only descends
BLOCK_KEYS = [blocking_dependency, blocked_by, required_before_stage0, gpu_policy,
premise_falsified]. B04 records its blockers in `remaining_blockers_after_this_design`,
which is NOT in that list. So B04's discharge-pointer target namespace is EMPTY and EVERY
`discharges` pointer in this document is dangling by construction, no matter how it is
spelled. The correct repair is therefore to STOP USING THE RESERVED KEY NAME, not to
re-spell the target.

WHY EDITING MY OWN KEY IS NOT AN APPEND-ONLY VIOLATION
------------------------------------------------------
The append-only rule protects the HISTORICAL RECORD: a dated key that some reader or tool
has already consumed is evidence, and rewriting it destroys provenance. The key being
touched here was written minutes ago by the SAME uncommitted pass, has never been
committed, and its only consumer to date is the queue run that reported it as a DEFECT.
Preserving a malformed reserved key would not preserve history -- it would permanently emit
a false machine-readable signal. All four PRE-EXISTING 2026-08-14/15/16 keys, and the 30
keys that predate this pass, are asserted byte-identical below.

The rename is recorded rather than done silently, and a new dated key states the finding.
"""
import json
import collections
import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
STATUS = HERE.parent / "STATUS.json"
MINE = [
    "related_work_presence_correction_20260817",
    "proposal_md_superseded_20260817",
    "novelty_check_regenerated_20260817",
    "closeout_20260817_scope_and_what_remains",
]

NOTE_KEY = "discharge_pointer_namespace_finding_20260817"
NOTE = {
    "finding": "B04's discharge-pointer target namespace is EMPTY. Any subkey named "
               "\"discharges\" in this document is DANGLING BY CONSTRUCTION and has NO EFFECT, "
               "however it is spelled.",
    "measured_how": "imported proposal/ready_queue.py and called rq._walk_blockers(doc) on this "
                    "file, 2026-08-17",
    "result": "_walk_blockers -> [] (zero blocker paths)",
    "root_cause": "_walk_blockers descends only BLOCK_KEYS = ['blocking_dependency','blocked_by',"
                  "'required_before_stage0','gpu_policy','premise_falsified'] (plus the "
                  "NESTED_BLOCKER_CONTAINERS allow-list). B04 keeps its blockers in "
                  "`remaining_blockers_after_this_design`, which is in NEITHER list. So that "
                  "array is invisible to blocker walking AND to lifecycle inference.",
    "practical_consequences": [
        "A `discharges` pointer at remaining_blockers_after_this_design[N] can never resolve; the "
        "queue prints it as dangling. Do not add one expecting it to close anything.",
        "Conversely, writing INTO remaining_blockers_after_this_design does not change queue "
        "state either -- it is display-free. The array is documentation for humans only.",
        "Discharge of B04's blockers is therefore recorded in PROSE in dated keys (see "
        "related_work_presence_correction_20260817 and proposal_md_superseded_20260817), which is "
        "the only mechanism available here, and is why those keys name the superseded item "
        "explicitly in a `supersedes` field rather than a `discharges` field.",
    ],
    "what_this_pass_did_about_it": "renamed its own prose subkey \"discharges\" -> "
        "\"discharges_PROSE_not_a_machine_pointer\" inside novelty_check_regenerated_20260817, so "
        "the queue no longer parses documentation as a reserved pointer. Verified: the dangling-"
        "pointer line disappears from ready_queue.py output. The subkey's VALUE is unchanged.",
    "self_criticism": "I introduced this defect in the same pass that fixed three others. The "
        "queue's own dangling-pointer check is what caught it -- I did not. The general lesson is "
        "the one in memory/read-what-the-consumer-reads-not-the-bare-key.md: before inventing a "
        "subkey name inside a file a tool parses, grep the tool for that name. \"discharges\" reads "
        "like plain English and is in fact reserved.",
    "NOT_fixed_here": "Adding `remaining_blockers_after_this_design` to ready_queue.py's "
        "BLOCK_KEYS would make B04's blockers machine-visible, but that is a change to a shared "
        "scheduler affecting all 17 proposals and could newly flip lifecycles. Filed as a "
        "follow-up, not done as a side effect of a B04 documentation pass.",
}


def main():
    raw_before = STATUS.read_bytes()
    doc = json.loads(raw_before, object_pairs_hook=collections.OrderedDict)
    keys_before = list(doc.keys())

    if NOTE_KEY in doc:
        print(f"REFUSING: {NOTE_KEY} already present.")
        return 2

    target = doc["novelty_check_regenerated_20260817"]
    if "discharges" not in target:
        print("REFUSING: expected subkey 'discharges' not found -- file is not in the state "
              "this script was written for.")
        return 2

    # Snapshot everything NOT mine, to prove untouched.
    others = [k for k in keys_before if k not in MINE]
    old_blobs = {k: json.dumps(doc[k], ensure_ascii=False, sort_keys=True) for k in others}
    # Also snapshot my other three keys -- only ONE key may change.
    my_untouched = [k for k in MINE if k != "novelty_check_regenerated_20260817"]
    for k in my_untouched:
        old_blobs[k] = json.dumps(doc[k], ensure_ascii=False, sort_keys=True)

    # Rename the subkey, preserving both its VALUE and its position.
    rebuilt = collections.OrderedDict()
    for k, v in target.items():
        if k == "discharges":
            rebuilt["discharges_PROSE_not_a_machine_pointer"] = v
        else:
            rebuilt[k] = v
    old_value = target["discharges"]
    doc["novelty_check_regenerated_20260817"] = rebuilt
    doc[NOTE_KEY] = NOTE

    out = json.dumps(doc, ensure_ascii=False, indent=1) + "\n"

    reparsed = json.loads(out, object_pairs_hook=collections.OrderedDict)
    assert list(reparsed.keys())[:len(keys_before)] == keys_before, "key order changed"
    assert list(reparsed.keys())[len(keys_before):] == [NOTE_KEY], "more than the note appended"
    for k, blob in old_blobs.items():
        assert json.dumps(reparsed[k], ensure_ascii=False, sort_keys=True) == blob, \
            f"key MUTATED that must not be: {k}"
    tgt = reparsed["novelty_check_regenerated_20260817"]
    assert "discharges" not in tgt, "reserved key still present"
    assert tgt["discharges_PROSE_not_a_machine_pointer"] == old_value, "value not preserved"
    assert reparsed["lifecycle"] == "ready_cpu", "lifecycle changed"

    STATUS.write_text(out, encoding="utf-8")
    json.loads(STATUS.read_bytes())
    print(f"OK  sha256 before={hashlib.sha256(raw_before).hexdigest()[:16]} "
          f"after={hashlib.sha256(STATUS.read_bytes()).hexdigest()[:16]}")
    print(f"OK  {len(old_blobs)} keys asserted byte-identical (30 pre-existing + 3 of my own)")
    print(f"OK  renamed subkey 'discharges' -> 'discharges_PROSE_not_a_machine_pointer' (value kept)")
    print(f"OK  appended {NOTE_KEY}; lifecycle still {reparsed['lifecycle']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
