#!/usr/bin/env python3
"""Guard: a proposal must not assert that a file is absent when the file is on disk.

WHY THIS EXISTS
---------------
On 2026-08-16 MAIN dispatched a B08 novelty-adjudication agent whose prompt said
RELATED_WORK.md was absent and had to be written. It had been on disk since 2026-08-15
(committed 463dca4, 39,604 B at the time, 59,799 B now). MAIN had not invented this: it is
what B08's own STATUS.json says, in three separate places, including

    "1. RELATED_WORK.md (leg-1-only) does not exist."

The records are append-only, so a sentinel written when it WAS true stays there forever
next to the later entry that corrects it. The same record already contained

    "blocker 1 said RELATED_WORK.md does not exist. IT DOES."

so the file was self-contradicting, and a reader who stops at the first match gets the
stale half. The cost: an agent spent part of a 33-minute run re-deriving that the premise
was wrong, and the dispatch could have asked for work that was already done.

Sweeping every proposal found this is not a B08 quirk -- EIGHT proposals assert a file is
absent that is present:

    A01 SOURCES.md          B01 RELATED_WORK.md    B03 RELATED_WORK.md
    B04 RELATED_WORK.md     B06 RELATED_WORK.md    B07 RELATED_WORK.md
    B08 RELATED_WORK.md     B12 RELATED_WORK.md

Every one of them is a live tripwire for the next agent that reads a blocker list and
believes it.

WHAT THIS GUARD DOES NOT CLAIM
------------------------------
Presence is not sufficiency. A RELATED_WORK.md can exist and still be inadequate, and
several of these proposals are legitimately blocked on the QUALITY of that file or on a
separate novelty adjudication. This checker only says: stop asserting the file is missing
when it is not. Fix the sentence, keep the blocker if the blocker is real.

It also does not edit anything. Rewriting an append-only research record is the
maintainer's call, not a script's -- the right repair is usually a new dated key that
supersedes, not a mutation of the old one.

Exit codes: 0 = no stale absence assertion. 1 = at least one (informational is NOT an
option here; see memory/an-informational-nonzero-rc-hides-real-defects.md -- a non-zero rc
that everyone learns to ignore is how the paperC rounding defect survived).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# "<file>.md ... does not exist" and friends. Bounded lookahead so we do not span
# sentences and match an unrelated absence claim later in the same blob.
ABSENCE = re.compile(
    r'([A-Za-z0-9_./-]+\.(?:md|py|json|tsv|csv))'
    r'[^"]{0,90}?(?:does not exist|do not exist|is absent|are absent|'
    r'is missing|not on disk|does NOT exist)',
    re.IGNORECASE)

# A sentence that QUOTES a stale claim in order to refute it is the repair, not the defect.
# This guard's own recommended fix -- "add a dated superseding key" -- necessarily restates
# the old sentence, so without this the checker penalises the only correct repair. Measured:
# of the 8 rows the first version reported, A01/SOURCES.md and B01/RELATED_WORK.md are
# refutation-only, i.e. pure false positives; B08 is mixed (2 bare sentinels + 1 correction);
# the other 5 are bare claims.
REFUTED_NEARBY = re.compile(
    r'\b(IT DOES|IT EXISTS|THEY DO|STALE|SUPERSED\w*|was stale|no longer (?:true|blocking)|'
    r'PREMISE_OF_THE_TASK_WAS_STALE|已存在|实际存在|已经存在)\b',
    re.IGNORECASE)
REFUTE_WINDOW = 260


def scan(status_path):
    """Return [(claimed_file, resolved_path, size)] for absences that are false.

    A hit is dropped when a refutation sits within REFUTE_WINDOW characters on either
    side, because that is a correction quoting its own stale sentinel.
    """
    try:
        text = status_path.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for m in ABSENCE.finditer(text):
        raw = m.group(1)
        name = raw.split("/")[-1]
        # Resolve against the proposal directory: these records name files relative to
        # their own proposal, and a bare basename is the common case.
        cand = status_path.parent / name
        if not (cand.exists() and cand.is_file()):
            continue
        before = text[max(0, m.start() - REFUTE_WINDOW):m.start()]
        after = text[m.end():m.end() + REFUTE_WINDOW]
        if REFUTED_NEARBY.search(before) or REFUTED_NEARBY.search(after):
            continue  # a correction, not a live stale claim
        out.append((raw, cand, cand.stat().st_size))
    return out


def main():
    stale = []
    n_scanned = 0
    for st in sorted(ROOT.glob("*/*/STATUS.json")):
        n_scanned += 1
        for raw, path, size in scan(st):
            stale.append((st.parent.name, raw, path.relative_to(ROOT), size))

    # de-dup: the same file is often asserted absent several times in one record
    seen, rows = set(), []
    for prop, raw, rel, size in stale:
        key = (prop, rel)
        if key in seen:
            continue
        seen.add(key)
        rows.append((prop, raw, rel, size))

    print(f"proposals scanned: {n_scanned}")
    print(f"stale absence assertions: {len(rows)}")
    print()
    if rows:
        print(f"{'proposal':46}{'claimed absent':22}{'bytes':>10}")
        for prop, raw, rel, size in rows:
            print(f"{prop[:46]:46}{raw.split('/')[-1][:22]:22}{size:10d}")
        print()
        print("FAIL: each row above tells the next agent to produce a file that exists.")
        print("      MAIN relayed exactly one of these (B08 RELATED_WORK.md) into a")
        print("      dispatch on 2026-08-16 and the agent had to spend part of its run")
        print("      disproving the premise.")
        print("      Repair by ADDING a dated superseding key, not by editing the old")
        print("      sentinel -- these records are append-only and the history is the point.")
        print("      Presence is not sufficiency: keep the blocker if it is really about")
        print("      the file's CONTENT, just stop saying the file is missing.")
        return 1

    print("PASS: no proposal asserts the absence of a file that is on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
