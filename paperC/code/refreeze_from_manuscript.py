#!/usr/bin/env python3
"""Re-freeze the submission snapshot, with the file list DERIVED FROM THE MANUSCRIPT.

WHY THIS EXISTS
---------------
Round_04's `submission_complete/MANIFEST.json` reports

    "freeze_gate_pass": true,
    "named_artifacts_missing": [],
    "missing_dependencies": [],

while the snapshot is missing the two artifacts the paper cites for its central fix:

    code/recompute_legality_aware_nulls.py     cited, ABSENT from the snapshot
    evidence/construct_nulls_legality_aware.json  cited, ABSENT from the snapshot

and while shipping their SUPERSEDED predecessor,
`evidence/floor_winners_curse_calibration.json`. Four of six round_04 reviewers made
"the frozen artifact does not contain the evidence it repeatedly claims to publish" a
major issue, one of them the leading reason for its score.

The manifest is not lying: `named_artifacts_present` holds ELEVEN hand-typed paths, so
the universe it checks against was itself hand-typed. Two artifacts created after that
list was written are outside its target space, and `[]` means "none of the eleven I know
about are missing". That is the same shape as the checker that reported 91/91 PASS while
three prose defects sat in the manuscript: a green field that describes its own
enumeration rather than the property a reader cares about.

So the fix is not "add two paths". It is to stop hand-enumerating: this script reads
sections/*.tex, extracts every evidence/ and code/ path the prose actually cites, and
copies THAT set. When the evidence set changes, the manifest changes with it.

WHAT IT DOES NOT DO
-------------------
It does not touch `round_04/submission_complete/`. That directory's `snapshot_sha256`
(ffd5fd7d...) is a record of what was frozen at that time; overwriting it would destroy
provenance even though nothing cites it yet. This writes a NEW round_05 directory.

Exit codes: 0 = snapshot written and every cited path is present in it.
2 = a cited path does not exist in the live tree either (nothing to copy) -- the snapshot
is still written, but the manifest records the gap honestly instead of reporting [].
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ROOT / "sections"
ROUNDS = ROOT / "review_rounds"

# A REVIEWED round is immutable. Deciding the destination is therefore not a constant.
#
# This was `DEST = ROOT / "review_rounds" / "round_05" / "submission_complete"`, and the
# docstring above still says "This writes a NEW round_05 directory" -- true when written,
# false the moment round_05 was reviewed. Measured 2026-08-17: MAIN ran this tool intending
# to create round_06 and it silently overwrote round_05's snapshot AFTER six reviewers had
# read it -- 5 manuscript files modified, 2 evidence files added, snapshot_sha256
# 4a2235e8 -> 16171eef. Recovered from git (f694741) because the snapshot was committed;
# had it not been, the artifact those six reviews were written against would be gone.
#
# The destination is now COMPUTED, and a round that already contains reviews is refused.
# A hardcoded path in a writer silently defines what gets destroyed -- cf.
# memory/a-hardcoded-list-in-an-emitter-silently-defines-a-headline.md.


def _has_reviews(round_dir):
    """True if this round has been reviewed, i.e. is immutable."""
    if any(round_dir.glob("reviews*/*.json")):
        return True
    return any(round_dir.glob("PANEL_AGGREGATE*.json"))


def resolve_dest():
    """The round to write. Newest unreviewed round, else the next round number.

    Override with PAPERC_FREEZE_DEST for a deliberate target. Even then a reviewed round
    is refused: the override exists to choose among unreviewed destinations, not to permit
    overwriting evidence a review was written against.
    """
    import os
    override = os.environ.get("PAPERC_FREEZE_DEST")
    if override:
        d = Path(override)
        rd = d.parent if d.name == "submission_complete" else d
        if rd.exists() and _has_reviews(rd):
            raise SystemExit(
                f"REFUSING to write {rd.name}: it contains reviews. A reviewed snapshot is "
                f"the artifact those reviews were written against and must not change. "
                f"Unset PAPERC_FREEZE_DEST to let the tool pick the next round.")
        return rd / "submission_complete"

    existing = sorted(ROUNDS.glob("round_[0-9][0-9]"))
    if existing and not _has_reviews(existing[-1]):
        return existing[-1] / "submission_complete"
    nxt = (int(existing[-1].name.split("_")[1]) + 1) if existing else 0
    return ROUNDS / f"round_{nxt:02d}" / "submission_complete"


DEST = resolve_dest()

CONTINUES = re.compile(r"[_/]$")


def unescape(s):
    return s.replace("\\_", "_").replace("\\", "")


def cited_paths():
    """Every evidence/ or code/ path named in the prose OR in the claim map.

    Same extraction as gate_cited_paths_in_artifact.py, deliberately: the gate that
    judges the snapshot and the tool that builds it must agree on what "cited" means,
    or the gate will fail its own artifact.

    UPDATED 2026-08-16, and this is why the agreement clause above matters. A round_05
    reviewer found two files -- s2_03_symmetric_inference.json (claim map row H-02, the
    source for "the flip is 3/12 vs 1/12") and s2_02_stratified_ordering.json (row H-04)
    -- present in the live tree and ABSENT from the frozen snapshot, while the gate
    reported "PASS: all 17 cited paths resolve". Both tools extracted only
    \\texttt{evidence/...} from sections/*.tex, so a file cited ONLY by
    claim_evidence_map.tsv was invisible to both. Fixing the gate alone would have made
    the freezer produce artifacts its own gate rejects, so the claim-map source is added
    to BOTH in the same commit. See memory/fix-the-class-not-the-instance.md.
    """
    found = set()
    for tex in sorted(SECTIONS.glob("*.tex")):
        flat = " ".join(tex.read_text(encoding="utf-8").split())
        groups = list(re.finditer(r"\\texttt\{([^}]*)\}", flat))
        for i, m in enumerate(groups):
            body = m.group(1)
            if not body.startswith(("evidence/", "code/")):
                continue
            path = unescape(body)
            j = i
            while CONTINUES.search(path) and j + 1 < len(groups):
                nxt = unescape(groups[j + 1].group(1))
                if flat[groups[j].end():groups[j + 1].start()].strip():
                    break
                path += nxt
                j += 1
            found.add(path.rstrip("/"))
    # The claim map is itself a SHIPPED artifact: a reviewer reads it, follows the
    # pointer, and must find the file. A dangling pointer there is worse than an
    # uncited file, because the map promises the number is verifiable.
    cmap = ROOT / "evidence" / "claim_evidence_map.tsv"
    if cmap.exists():
        for line in cmap.read_text(encoding="utf-8").splitlines():
            for tok in re.findall(r"(?:evidence|code)/[A-Za-z0-9_./-]+", line):
                found.add(tok.rstrip(".,;").rstrip("/"))
    return sorted(found)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Review-process language that must never reach a blind reviewer. Anchored on words that
# name the PROCESS (which round, whose verdict), not on words a paper legitimately uses
# about its own statistics -- "revision" alone would fire on prose about revising a bound.
#
# ⚠️ NO \b AROUND THE NOUNS. `_` is a word character, so `\breviewer\b` does NOT match
# `"reviewer_counterexample_reproduced"` -- and a JSON KEY NAME is the single most common
# shape these leaks take. Measured 2026-08-17: my first version of this screen reported 5
# leaks in s2_02 where grep reported 6 lines; the missed line was exactly that key, at
# line 8. A screen that under-counts relative to a one-line grep is not a screen.
#
# The boundary is therefore hand-built: a preceding non-letter, and a trailing character
# that is anything EXCEPT a letter -- `_`, `"`, `:`, space and end-of-line all count as
# boundaries, so `reviewer_x` and `"reviewer":` fire, while `interviewer` (letter before)
# and `refereed` (letter after) do not. Both look-alikes are in the controls; `refereed`
# was a real false positive in the version before this one, caught by the negative control
# rather than by inspection, which is the argument for having the negative control.
_BLIND_FATAL = re.compile(
    r"round_0[0-9]"
    r"|(?<![A-Za-z])(?:reviewer|referee|rebuttal)(?![A-Za-z])"
    r"|NEEDS_REVISION"
    r"|(?<![A-Za-z])meta.review(?![A-Za-z])"
    r"|(?<![A-Za-z])blind.review(?![A-Za-z])", re.I)

# SECOND PATTERN: the leak's SHAPE, not its vocabulary.
#
# The pattern above matches the *words* of the review process. It therefore misses any
# PARAPHRASE that discloses the same thing without them. Measured 2026-08-17, and the
# demonstration came from the de-attribution work itself: the first replacement wording
# written for construct_nulls_legality_aware.json was
#
#     "an audit found, in several independent readings, that ..."
#
# which still tells a blind reviewer that several independent parties read the paper. The
# vocabulary screen PASSED it. So did every one of these, all of which disclose a panel
# size or a vote split:
#
#     "four of six independent readers flagged this"
#     "multiple independent assessors agreed"
#     "three of the five reports raised it"
#
# That is the same failure as the round 00-02 breach in kind: I checked for the words of
# the process instead of for the disclosure. A count-of-persons construction is the
# disclosure, whatever noun it uses. See
# memory/blindness-check-must-grep-writer-steering-not-just-panel-words.md.
#
# So this matches "<N|word> of <the>? <N|word> <person-ish plural>" and a small set of
# panel nouns that are not review jargon. It is deliberately narrow on the NOUN side --
# `readers/readings/assessors/referees/reports/evaluators/respondents/panellists` -- so
# that ordinary research prose ("four of six cells", "three of the five arms", "two of
# twelve items") does not fire. Those are in the negative controls.
_NUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
_PANEL_NOUN = (r"(?:readers?|readings?|assessors?|evaluators?|respondents?|panell?ists?"
               r"|reports?|critiques?|referees?|opinions?)")
_BLIND_SHAPE = re.compile(
    # "four of six independent readers", "three of the five reports"
    rf"(?<![A-Za-z]){_NUM}\s+of\s+(?:the\s+)?{_NUM}\s+(?:\w+\s+){{0,2}}?{_PANEL_NOUN}(?![A-Za-z])"
    # "several/multiple independent readings", "N independent assessors"
    rf"|(?<![A-Za-z])(?:several|multiple|numerous|various|{_NUM})\s+independent\s+"
    rf"(?:\w+\s+){{0,1}}?{_PANEL_NOUN}(?![A-Za-z])",
    re.I)


def screen_blind(path, rel):
    """Return the review-process leaks in one file, as (lineno, text) pairs.

    WHY THIS EXISTS -- this tool created the defect it now blocks.
    ------------------------------------------------------------
    Measured 2026-08-17 on round_06/submission_complete/evidence/. This freezer shipped
    `s2_02_stratified_ordering.json` and `s2_03_symmetric_inference.json` BYTE-IDENTICAL
    to the live tree, and those files contain, in plain text:

        "round_01_review_findings": { ... }
        "reviewer_verdict": "NEEDS_REVISION -- the retraction is arithmetically exact..."
        "reproduced from integer counts independently by the round_01 reviewer"

    A blind reviewer handed that artifact learns that a previous round demanded revision
    and what it said -- the same class of breach that VOIDED the round 00-02 scores (see
    memory/blindness-check-must-grep-writer-steering-not-just-panel-words.md).

    The causal chain is two of my own repairs composing into a new defect:
      1. round_04 shipped only 2 of 29 evidence files -> five reviewers docked the paper
         for "the artifact does not contain the evidence it claims to publish".
      2. Fix A (freeze_round.py): pack evidence by default instead of by hand-enumerated
         whitelist. Fix B (this file + gate_cited_paths_in_artifact.py): also treat
         claim_evidence_map.tsv rows as citations, because a round_05 reviewer found
         H-02/H-04 cited there and absent from the snapshot.
      3. Together they taught this tool to DEMAND exactly the two files whose contents
         quote a reviewer -- and this tool, unlike freeze_round.py, had no blindness
         screen at all (grep for screen_fatal/BLIND/LEAK_CONTENT returned 0 hits).

    So the missing-evidence defect was converted into a blindness-leak defect, and the
    gate that was supposed to certify the artifact was the thing requesting the leak.

    FAIL CLOSED. A leak is not a warning: an artifact that discloses the review process
    cannot be un-read once a reviewer has it, whereas a refused freeze costs one command.
    The repair is to cite a de-attributed record (`*_shippable.json`) from the claim map,
    NOT to add an exclusion here -- an exclusion would silently drop evidence the prose
    promises, which is defect (1) again.
    """
    if path.suffix.lower() not in (".json", ".tsv", ".csv", ".md", ".txt", ".py", ".tex"):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        # FAIL CLOSED on an unreadable file. Returning [] here means "no leaks found",
        # which for a screen is the *permissive* answer -- so a missing or unreadable
        # file would be reported as clean and shipped.
        #
        # Measured 2026-08-17: my own dry-check passed a repo-relative path while CWD was
        # the repo root (the real code passes ROOT/rel), read_text raised, and the screen
        # printed "clean" for BOTH files that in fact carry 6 and 2 leaks. I nearly
        # concluded the guard did not work; the guard was fine and the probe was wrong --
        # but a screen must not be able to say "clean" because it could not look.
        return [(rel, 0, "UNREADABLE",
                 f"cannot read for blindness screening ({type(e).__name__}: {e}) -- "
                 f"treated as a LEAK because a screen that cannot read a file must not "
                 f"pass it")]
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        m = _BLIND_FATAL.search(line)
        if m:
            hits.append((rel, i, m.group(0), line.strip()[:120]))
            continue                      # one hit per line is enough to block
        m = _BLIND_SHAPE.search(line)
        if m:
            hits.append((rel, i, f"SHAPE:{m.group(0)}", line.strip()[:120]))
    return hits


def copy_tree(src, dst):
    """Copy a file or a directory, returning the list of files written."""
    written = []
    if src.is_dir():
        for f in sorted(src.rglob("*")):
            if f.is_file():
                rel = f.relative_to(src)
                target = dst / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                written.append(target)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        written.append(dst)
    return written


def screen_sources(cited):
    """Screen the SOURCE files a freeze would copy, before DEST is touched.

    WHY THIS RUNS FIRST -- a refusal must not be destructive.
    --------------------------------------------------------
    Measured 2026-08-17. main() used to rmtree(DEST) and only screen afterwards, so a
    refusal DELETED the existing snapshot it declined to replace: round_06's
    submission_complete (51 files, snapshot 16171eef...) was wiped by a refusal caused by
    a leak in a file the new freeze had not even reached yet. It was recoverable only
    because those 52 files happened to be committed at ca33a17.

    A guard whose failure path destroys the artifact it protects is worse than no guard on
    the day it fires. So: screen the sources, refuse before any mutation, and leave the
    previous snapshot exactly as it was.

    Expands cited DIRECTORIES the same way copy_tree does. That matters -- my own
    pre-flight check screened only the top-level cited paths and reported "clean", while
    the real leak sat in evidence/mmlu_scale_power/mmlu_pro_power_nulls_v2.json, a file
    INSIDE a cited directory. The screen must walk what the copier walks.
    """
    leaks = []
    for rel in cited:
        src = ROOT / rel
        if not src.exists():
            continue                      # absent -> reported separately by main()
        if src.is_dir():
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    leaks += screen_blind(f, str(Path(rel) / f.relative_to(src)))
        else:
            leaks += screen_blind(src, rel)
    return leaks


def main():
    cited = cited_paths()
    if not cited:
        print("CANNOT FREEZE: no evidence/ or code/ paths extracted from sections/*.tex")
        return 3

    # NON-DESTRUCTIVE REFUSAL: screen before DEST is touched. See screen_sources.
    leaks = screen_sources(cited)
    if leaks:
        print("REFUSING TO FREEZE: review-process language in artifacts a blind "
              "reviewer would receive.")
        for rel, ln, tok, line in leaks[:20]:
            print(f"  {rel}:{ln}  [{tok}]  {line}")
        if len(leaks) > 20:
            print(f"  ... and {len(leaks) - 20} more")
        print("\nRepair: point the citation (claim_evidence_map.tsv row, or the prose)\n"
              "at a de-attributed record -- e.g. evidence/<name>_shippable.json -- which\n"
              "keeps every number and drops only 'who raised it in which round'.\n"
              "Do NOT excuse the file here: dropping it recreates the round_04 defect\n"
              "where the artifact lacked evidence the paper promises.\n"
              f"\nThe existing snapshot at {DEST} was NOT modified.")
        return 4

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    written, absent = [], []
    for rel in cited:
        src = ROOT / rel
        if not src.exists():
            absent.append(rel)
            continue
        written += copy_tree(src, DEST / rel)

    # BACKSTOP SCREEN on what was actually COPIED. screen_sources() above already
    # refused any leak before DEST was touched, so this should never fire -- it exists
    # because the two screens walk different things (source tree vs written set) and a
    # divergence between them is itself a bug worth catching loudly.
    #
    # The rmtree here is safe in a way the old pre-screen one was NOT: by this point DEST
    # contains only files this run just wrote, so removing it cannot destroy a previous
    # snapshot. That is exactly the distinction the 2026-08-17 incident turned on.
    leaks = []
    for p in written:
        if p.is_file():
            leaks += screen_blind(p, str(p.relative_to(DEST)))
    if leaks:
        shutil.rmtree(DEST)          # only newly-written files -- see note above
        print("REFUSING TO FREEZE: review-process language in artifacts a blind "
              "reviewer would receive. (Reached the BACKSTOP screen, which means "
              "screen_sources missed it -- the two screens disagree and that is a bug.)")
        for rel, ln, tok, line in leaks[:20]:
            print(f"  {rel}:{ln}  [{tok}]  {line}")
        if len(leaks) > 20:
            print(f"  ... and {len(leaks) - 20} more")
        print("\nRepair: point the citation (claim_evidence_map.tsv row, or the prose)\n"
              "at a de-attributed record -- e.g. evidence/<name>_shippable.json -- which\n"
              "keeps every number and drops only 'who raised it in which round'.\n"
              "Do NOT excuse the file here: dropping it recreates the round_04 defect\n"
              "where the artifact lacked evidence the paper promises.")
        return 4

    # the manuscript itself, so a reader can follow the citations
    man = DEST / "manuscript"
    man.mkdir(exist_ok=True)
    for f in sorted(SECTIONS.glob("*.tex")):
        shutil.copy2(f, man / f.name)
    for extra in ("main.tex", "main.pdf", "refs.bib"):
        p = ROOT / extra
        if p.exists():
            shutil.copy2(p, man / extra)
            written.append(man / extra)
    written += [man / f.name for f in sorted(SECTIONS.glob("*.tex"))]

    files = {}
    for p in sorted(set(written)):
        if p.is_file():
            files[str(p.relative_to(DEST))] = sha256(p)

    digest = hashlib.sha256()
    for k in sorted(files):
        digest.update(k.encode())
        digest.update(files[k].encode())

    manifest = {
        "schema_version": 2,
        # DERIVED from DEST, not typed. It read `"round": 5` while the tool was writing
        # into round_06/ -- residue of the same hardcoded-destination bug recorded at
        # resolve_dest() above. A manifest that misreports which round it belongs to
        # breaks exactly the provenance the manifest exists to provide.
        "round": int(DEST.parent.name.split("_")[1])
        if DEST.parent.name.startswith("round_") else None,
        "round_dir": DEST.parent.name,
        "paper": "paperC",
        "snapshot_sha256": digest.hexdigest(),
        "n_files": len(files),
        # The field that round_04 got wrong. It is now DERIVED, not typed.
        "cited_path_source": "sections/*.tex, \\texttt{evidence/...} and \\texttt{code/...}",
        "n_cited_paths": len(cited),
        "cited_paths": cited,
        "named_artifacts_missing": absent,
        "freeze_gate_pass": not absent,
        "provenance_note": (
            "round_04/submission_complete is NOT modified by this tool. Its "
            "snapshot_sha256 ffd5fd7d8c3d8b30d44c... records what was frozen then, "
            "including the fact that it lacked E-CAL. Overwriting a hash that has "
            "already been emitted destroys provenance."
        ),
        "known_limitation": (
            "This checks PRESENCE of every cited path, not that its CONTENT is current. "
            "A stale file at a cited path passes."
        ),
        "files": files,
    }
    (DEST / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")

    print(f"snapshot: {DEST.relative_to(ROOT)}")
    print(f"cited paths: {len(cited)}  copied: {len(cited) - len(absent)}  "
          f"absent in live tree: {len(absent)}")
    print(f"files written: {len(files)}  snapshot_sha256: {manifest['snapshot_sha256'][:16]}...")
    if absent:
        print()
        print("ABSENT IN THE LIVE TREE (recorded in the manifest, not silently dropped):")
        for a in absent:
            print(f"  {a}")
        return 2
    print()
    print("PASS: every path the manuscript cites is present in the new snapshot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
