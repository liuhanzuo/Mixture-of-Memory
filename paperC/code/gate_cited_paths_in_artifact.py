#!/usr/bin/env python3
"""Guard: every evidence/code path the manuscript names must exist in the frozen artifact.

WHY THIS EXISTS
---------------
Round_04's freeze shipped 2 evidence records while the manuscript's Table 12 resolved its
evidence identifiers to ~10 files. Four of six reviewers independently made it a major
issue, and one made it the leading reason for its score:

    "the frozen artifact does not contain the evidence it repeatedly claims to publish"

They were right, and the cause was an interface, not the research: freeze_round.py took
evidence as a repeatable whitelist, so the freeze packaged exactly the two paths a human
typed, and then reported `missing_dependencies: []` -- actively signalling completeness.
`submission_complete/` was the repair, and it did restore those ten.

But the repair was frozen before the calibration fix landed, so it enumerated the
SUPERSEDED record and the manuscript now cites the SUPERSEDING one:

    shipped     evidence/floor_winners_curse_calibration.json     2026-08-14 22:26
    cited as    evidence/construct_nulls_legality_aware.json      2026-08-16 14:13   ABSENT
    also cited  code/recompute_legality_aware_nulls.py            2026-08-16 14:13   ABSENT

E-CAL is the corrected legality-aware calibration -- the paper's central fix, cited as its
single machine-readable source. A reviewer following that pointer in the frozen artifact
finds the record the paper says is wrong, and does not find the record that replaced it.

So the lesson is not "restore ten files". A hand-enumerated manifest goes stale every time
the evidence changes, and the second failure looks exactly like the first. What is needed is
a check that reads the manuscript and compares it against the artifact, which is what this
gate does.

WHAT THIS GUARD CHECKS
----------------------
It extracts every `\\texttt{evidence/...}` and `\\texttt{code/...}` path from sections/*.tex,
rejoins the ones the LaTeX line-break style splits across adjacent \\texttt groups, and
asserts each resolves inside the frozen submission directory.

The rejoin step matters: a naive extraction reports fragments like `evidence/mmlu_scale_`
as missing, and four of six "missing" paths in the first run of this check were my own
regex artefacts rather than real absences. A gate that cries wolf on its own parsing gets
ignored, so fragments ending in `_` or `/` are rejoined before any verdict.

Exit codes: 0 = every cited path is present. 2 = at least one genuine absence.
3 = the submission directory does not exist.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "sections"
# Which frozen round to judge. Defaults to the newest round_NN/submission_complete that
# exists, so the gate follows the re-freeze instead of being pinned to a stale round --
# round_04 was the one that shipped WITHOUT E-CAL, and a gate hard-pinned to it would keep
# reporting that failure forever after the artifact was repaired, or worse, be edited to
# point at the new round and silently stop checking the old finding.
# Override with PAPERC_SUBMISSION_DIR for a control run.
def _newest_submission():
    import os
    override = os.environ.get("PAPERC_SUBMISSION_DIR")
    if override:
        return Path(override)
    rounds = sorted((ROOT / "review_rounds").glob("round_*/submission_complete"))
    return rounds[-1] if rounds else ROOT / "review_rounds" / "round_04" / "submission_complete"


SUBMISSION = _newest_submission()

TEXTTT = re.compile(r"\\texttt\{((?:evidence|code)/[^}]*)\}")
# a \texttt group that continues in the next one: paperC breaks long paths at _ or /
CONTINUES = re.compile(r"[_/]$")


def unescape(s):
    return s.replace("\\_", "_").replace("\\", "")


def cited_paths():
    """Every evidence/ or code/ path named in the prose, with split groups rejoined."""
    found = set()
    for tex in sorted(SECTIONS.glob("*.tex")):
        flat = " ".join(tex.read_text(encoding="utf-8").split())
        # walk the \texttt groups in order so a fragment can absorb its continuation
        groups = list(re.finditer(r"\\texttt\{([^}]*)\}", flat))
        for i, m in enumerate(groups):
            body = m.group(1)
            if not body.startswith(("evidence/", "code/")):
                continue
            path = unescape(body)
            # absorb following groups while the path is still dangling
            j = i
            while CONTINUES.search(path) and j + 1 < len(groups):
                nxt = unescape(groups[j + 1].group(1))
                # only absorb if the groups are adjacent in the flattened text
                between = flat[groups[j].end():groups[j + 1].start()]
                if between.strip():
                    break
                path += nxt
                j += 1
            found.add(path)
    return found


def main():
    if not SUBMISSION.exists():
        print(f"CANNOT CHECK: {SUBMISSION} does not exist")
        return 3

    paths = cited_paths()
    missing, present, fragments = [], [], []
    for p in sorted(paths):
        if CONTINUES.search(p) and not (SUBMISSION / p.rstrip("/")).exists():
            # still dangling after rejoin AND not a real directory -> parsing artefact
            fragments.append(p)
            continue
        target = SUBMISSION / p.rstrip("/")
        (present if target.exists() else missing).append(p)

    print(f"submission: {SUBMISSION.relative_to(ROOT)}")
    print(f"cited paths: {len(paths)}  present: {len(present)}  "
          f"missing: {len(missing)}  unresolved fragments: {len(fragments)}")
    print()
    for p in present:
        print(f"  ok      {p}")
    for p in fragments:
        print(f"  frag    {p}   (line-break fragment, not counted)")
    for p in missing:
        print(f"  MISSING {p}")
    print()

    if missing:
        print("FAIL: the manuscript names paths the frozen artifact does not contain.")
        print("      Four of six round_04 reviewers made exactly this a major issue, and a")
        print("      hand-enumerated freeze manifest reproduces it every time the evidence")
        print("      set changes. Re-freeze with the full evidence tree rather than adding")
        print("      these paths one at a time.")
        for p in missing:
            live = ROOT / p
            note = "exists in the live tree" if live.exists() else "ABSENT LIVE TOO"
            print(f"        {p}  ({note})")
        return 2

    print(f"PASS: all {len(present)} cited paths resolve inside the frozen artifact.")
    print("      This checks PRESENCE, not content: a stale file at the right path passes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
