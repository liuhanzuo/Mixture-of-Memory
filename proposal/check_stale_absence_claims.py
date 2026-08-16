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
#
# The character class EXCLUDES anything that looks like another filename, because the
# 90-char lookahead otherwise bridges straight past an intervening file and pins the
# absence on the wrong subject. Measured false positive (2026-08-16), B03:
#
#   "...and GATE_PREREGISTRATION.md 9.1/11 cite the anchor as bare
#    'logs/olmo2_1B_keep7fresh2_1node.log'. THAT PATH DOES NOT EXIST ON wzc1."
#
# The absent thing is the .log. GATE_PREREGISTRATION.md is merely the file doing the
# citing, and it is present -- so the guard reported a stale claim where the record was
# CORRECT (verified: `ls logs/olmo2_1B_keep7fresh2_1node.log` -> rc=2 on wzc1; the file
# is zwfy6-only, exactly as B03 says). Two separate mistakes had to coincide:
#   (a) the lookahead crossed a filename boundary, and
#   (b) `.log` is not in the extension list, so the REAL subject was invisible and could
#       not out-compete the wrong one.
# (a) is the fix below. (b) is deliberately left alone as a SUBJECT: adding `.log` to the
# capture group would make the guard resolve run logs against the proposal directory, where
# they never live, so every such row would be a fresh false positive in the other direction.
# This checker's scope is proposal-local documents; cross-disk log provenance is a different
# question and `memory/two-disk-rule-applies-to-main-too.md` is where that lives. But `.log`
# and friends DO appear in BARRIER_EXT below, because a filename we refuse to accuse must
# still be able to stop the lookahead from reaching past it.
#
# First attempt at (a) excluded only quote and slash characters from the gap. That passed the
# B03 case -- but for the wrong reason: the log path there happens to be single-quoted. A
# bare filename in the gap was still crossed, which the fixture
#   "see foo.json; separately bar.md does not exist"
# caught by pinning the absence on foo.json. So the barrier has to be the filename TOKEN,
# not the punctuation that sometimes surrounds it.
BARRIER_EXT = ("md|py|json|jsonl|tsv|csv|log|txt|sh|npy|pt|bin|safetensors|"
               "yaml|yml|tex|bib|ini|cfg|out|err|pdf|aux")
# A position where some other filename begins. Extensions are enumerated rather than matched
# as [a-z]{2,6} so that "e.g.", "i.e." and version strings like "9.1/11" are not mistaken for
# files -- treating those as barriers would suppress genuine stale claims (a false NEGATIVE,
# which is the direction that actually costs a dispatch).
_FILE_AHEAD = r'(?![A-Za-z0-9_.-]*\.(?:' + BARRIER_EXT + r')\b)'
ABSENCE = re.compile(
    r'([A-Za-z0-9_./-]+\.(?:md|py|json|tsv|csv))'
    r'(?![A-Za-z0-9_.-])'
    r'(?:' + _FILE_AHEAD + r'[^"])' + r'{0,90}?'
    r'(?:does not exist|do not exist|is absent|are absent|'
    r'is missing|not on disk|does NOT exist)',
    re.IGNORECASE)
# KNOWN LIMITATION, measured not assumed: on "RELATED_WORK.md and SOURCES.md do not exist"
# only SOURCES.md is captured, because the barrier stops RELATED_WORK.md from reaching the
# predicate. That under-reports a genuine double claim. Verified absent from the live tree
# (see the conjunction probe in this file's control block), and the failure direction is
# under-reporting, so it is left documented rather than patched with a wider net.

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
