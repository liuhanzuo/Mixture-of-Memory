#!/usr/bin/env python3
"""Correct a FALSE sentence I wrote into novelty_verdict_20260817 (2026-08-17).

WHAT IS WRONG, AND HOW I CAUGHT IT
==================================
``novelty_verdict_20260817.verdict_string_engineering_MEASURED`` ends with:

    "That probe reports PENDING for the whole-key blob (other fields of this key
    legitimately discuss the narrowing), so if a future reader ever blobs the
    whole key, it will regress -- recorded here rather than left as a landmine."

**That is false.** The writer PRINTED the probe result at write time and it said
``CLEARED``, not ``PENDING``:

    [probe] if a future reader blobbed the WHOLE key it would say: CLEARED

I wrote the sentence predicting PENDING, the probe measured CLEARED, and I shipped
the prediction instead of the measurement -- while the contradicting line was in my
own tool's output. That is precisely
``memory/a-declared-lifecycle-is-not-an-adjudicated-one.md`` and
``memory/a-writers-hardcoded-destination-defines-what-it-destroys.md``: I read the
PASS line and skipped the line the tool printed above it.

Re-measured independently just now over the on-disk key: PENDING tokens = NONE;
CLEARED tokens = ['gate cleared', 'clear', 'pass', 'not preempted']. The reason my
prediction failed is that the paraphrase which fixed the ``verdict`` field also
removed the underscored token from EVERY field -- the whole key now says
"narrowing" only in prose forms that no token matches. So the key is MORE robust
than I claimed, not less.

WHY THE SUBSTANTIVE POINT SURVIVES ANYWAY
=========================================
The fragility is real; I attributed it to the wrong mechanism. A whole-key blob
would not regress on the PENDING token -- it would go CLEARED, but for a BAD
reason: the bare substring ``"pass"`` matches inside ordinary words (e.g. "passes",
"a judge pass"), and ``"clear"`` matches inside "unclear". So a future reader that
blobs the whole key gets CLEARED almost unconditionally, i.e. it would report a
cleared gate even for a key whose verdict was pending. That is a WEAKER guard, not
a stricter one, and it is a real hazard -- just the opposite failure direction from
the one I described.

MECHANICS
=========
STATUS.json is append-only, so the false sentence is NOT edited. This adds ONE new
dated key that names it and states the measured value. Byte-prefix append, 31
pre-existing keys asserted byte-identical.

0 GPU, 0 ssh. Run once.
"""
import hashlib
import json
import os
import sys
from collections import OrderedDict

ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
STATUS = os.path.join(ROOT, "proposal/backlog/B08-memory-applications/STATUS.json")

BYTES_BEFORE = 95257
N_KEYS_BEFORE = 31

VERDICT_CLEARED = ("hold_in_backlog", "gate cleared", "clear", "pass",
                   "audited", "no candidate preempts", "not preempted")
VERDICT_PENDING = ("needs_narrowing", "unchecked", "not_checked", "todo")

NEW_KEYS = OrderedDict()

NEW_KEYS["novelty_verdict_20260817_probe_correction"] = {
    "supersedes": (
        "The FINAL SENTENCE of "
        "novelty_verdict_20260817.verdict_string_engineering_MEASURED, which "
        "reads 'That probe reports PENDING for the whole-key blob ... so if a "
        "future reader ever blobs the whole key, it will regress'. That sentence "
        "is FALSE. Everything else in that field is correct and stands, including "
        "the aborted-first-draft account, which is what matters operationally. "
        "NOT EDITED: append-only, and a wrong sentence that shipped is provenance."
    ),
    "when": "2026-08-17",
    "gpu_spent": "ZERO.",
    "the_false_claim": (
        "I predicted that classifying the WHOLE novelty_verdict_20260817 key as a "
        "single blob would yield PENDING, because other fields of the key discuss "
        "the narrowing."
    ),
    "the_measured_value": (
        "CLEARED. Measured twice, independently: (a) the writer PRINTED it at "
        "write time -- '[probe] if a future reader blobbed the WHOLE key it would "
        "say: CLEARED' -- and (b) re-measured from the on-disk key afterwards: "
        "VERDICT_PENDING tokens present = NONE; VERDICT_CLEARED tokens present = "
        "['gate cleared', 'clear', 'pass', 'not preempted']."
    ),
    "how_the_error_happened_and_it_is_the_instructive_part": (
        "The contradicting evidence was in my OWN tool's stdout, one line above "
        "the line I did read. I wrote the sentence as a PREDICTION before running "
        "the writer, the writer MEASURED the opposite, and I shipped the "
        "prediction. Cf. memory/a-declared-lifecycle-is-not-an-adjudicated-one.md "
        "(I reported two things as verified that the artefacts contradicted) and "
        "memory/a-writers-hardcoded-destination-defines-what-it-destroys.md (I "
        "read PASS and skipped the destination line the tool printed). Root cause: "
        "the probe was added as belt-and-braces, so I treated its output as "
        "decoration rather than as a result that could falsify the prose next to "
        "it. A printed measurement is a measurement."
    ),
    "why_the_prediction_failed": (
        "The paraphrase that fixed the `verdict` field (after the pre-write guard "
        "aborted the first draft) also removed the underscored not-yet-narrowed "
        "token from EVERY field of the key, not just from `verdict`. The key now "
        "refers to the narrowing only in prose forms that match no token. So the "
        "key is MORE robust to a future whole-key reader than I claimed."
    ),
    "the_fragility_is_real_but_points_the_OTHER_WAY": (
        "Correcting the direction, not deleting the concern. A whole-key blob "
        "would NOT regress toward PENDING; it would return CLEARED almost "
        "unconditionally, because VERDICT_CLEARED contains the bare substrings "
        "'pass' and 'clear', which are NOT word-bounded: 'pass' matches inside "
        "'passes' / 'a judge pass' / 'bypass', and 'clear' matches inside "
        "'unclear'. Any B08 key that mentions a judge pass would therefore read as "
        "a cleared novelty gate. So a whole-key reader would be a WEAKER guard, "
        "and the safe failure mode I imagined is actually the unsafe one. Today's "
        "reader is fine: ready_queue.py:864-867 reads only the 'verdict' (or "
        "'status') field."
    ),
    "not_a_ready_queue_bug_report": (
        "This is NOT a request to change VERDICT_CLEARED. ready_queue.py is the "
        "shared scheduler for 17-18 proposals; word-bounding 'pass' or 'clear' "
        "could flip other proposals' novelty status and is a separate reviewed "
        "task requiring a before/after queue diff. Recorded here as a hazard, not "
        "patched as a drive-by."
    ),
    "what_this_does_NOT_change": (
        "The verdict itself, the four applied edits, the lifecycle, and the "
        "gpu_authorisation statement are all unaffected. B08 remains ready_cpu / "
        "needs_prior_gate and remains NOT authorised for a GPU."
    ),
}


def main():
    with open(STATUS, "rb") as f:
        raw_before_b = f.read()
    raw_before = raw_before_b.decode("utf-8")
    print(f"[pre]  {len(raw_before_b)} bytes, sha256 "
          f"{hashlib.sha256(raw_before_b).hexdigest()}")
    if len(raw_before_b) != BYTES_BEFORE:
        sys.exit(f"ABORT: expected {BYTES_BEFORE} B, got {len(raw_before_b)}")

    before = json.loads(raw_before, object_pairs_hook=OrderedDict)
    keys_before = list(before.keys())
    if len(keys_before) != N_KEYS_BEFORE:
        sys.exit(f"ABORT: expected {N_KEYS_BEFORE} keys, got {len(keys_before)}")
    for k in NEW_KEYS:
        if k in before:
            sys.exit(f"ABORT: key exists, would OVERWRITE: {k}")

    # Confirm the sentence being corrected is really on disk, verbatim.
    target = before["novelty_verdict_20260817"]["verdict_string_engineering_MEASURED"]
    needle = "That probe reports PENDING for the whole-key blob"
    print(f"[pre]  false sentence present verbatim: {needle in target}")
    if needle not in target:
        sys.exit("ABORT: the sentence this key corrects is not on disk as quoted. "
                 "Re-audit before appending a correction to something else.")

    # Re-measure, so the correction states a value measured HERE, not copied.
    blob = json.dumps(before["novelty_verdict_20260817"], ensure_ascii=False).lower()
    p = [s for s in VERDICT_PENDING if s in blob]
    c = [s for s in VERDICT_CLEARED if s in blob]
    result = "PENDING" if p else ("CLEARED" if c else "UNPARSED")
    print(f"[pre]  re-measured whole-key blob: PENDING={p or 'NONE'} "
          f"CLEARED={c} -> {result}")
    if result != "CLEARED":
        sys.exit(f"ABORT: re-measurement says {result}, but this correction "
                 f"asserts CLEARED. Do not write a correction that is itself "
                 f"unverified.")

    # And show the word-boundary hazard is real, not asserted.
    for probe in ("the judge pass completed", "the situation is unclear"):
        hit = [s for s in VERDICT_CLEARED if s in probe.lower()]
        print(f"[pre]  substring hazard probe {probe!r} -> CLEARED tokens {hit}")

    frozen = {k: json.dumps(before[k], sort_keys=True, ensure_ascii=False)
              for k in keys_before}

    tail = raw_before.rstrip()
    if not tail.endswith("}"):
        sys.exit("ABORT: file does not end with '}'")
    body = tail[:-1].rstrip()
    if not body.endswith("}"):
        sys.exit("ABORT: unexpected byte before closing brace")

    chunks = []
    for k, v in NEW_KEYS.items():
        b = json.dumps({k: v}, indent=2, ensure_ascii=False)
        chunks.append(b[b.index("\n") + 1:b.rindex("\n")])
    out = body + ",\n\n" + ",\n\n".join(chunks) + "\n}\n"

    with open(STATUS, "w", encoding="utf-8") as f:
        f.write(out)

    with open(STATUS, "rb") as f:
        raw_after_b = f.read()
    prefix = raw_before_b[:len(body.encode("utf-8"))]
    if not raw_after_b.startswith(prefix):
        sys.exit("ABORT: byte-prefix broken -- restore from git")
    print(f"\n[post] byte-prefix of {len(prefix)} bytes preserved VERBATIM "
          f"({len(raw_before_b)} -> {len(raw_after_b)} bytes)")

    with open(STATUS, encoding="utf-8") as f:
        rb = json.load(f, object_pairs_hook=OrderedDict)
    ok = len(rb) == N_KEYS_BEFORE + len(NEW_KEYS)
    print(f"[post] {len(rb)} keys (expect {N_KEYS_BEFORE + len(NEW_KEYS)}): "
          f"{'OK' if ok else 'FAIL'}")

    rb_keys = list(rb.keys())
    n_bad = sum(1 for i, k in enumerate(keys_before)
                if not (rb_keys[i] == k and
                        json.dumps(rb[k], sort_keys=True,
                                   ensure_ascii=False) == frozen[k]))
    print(f"[post] {len(keys_before) - n_bad}/{len(keys_before)} pre-existing "
          f"keys identical in name, order and value")
    if n_bad:
        ok = False

    # The false sentence must STILL be there. Correcting != erasing.
    still = needle in rb["novelty_verdict_20260817"][
        "verdict_string_engineering_MEASURED"]
    print(f"[post] false sentence STILL on disk (provenance kept): {still}")
    if not still:
        ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL - RESTORE FROM GIT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
