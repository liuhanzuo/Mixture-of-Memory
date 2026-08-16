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
DEST = ROOT / "review_rounds" / "round_05" / "submission_complete"

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


def main():
    cited = cited_paths()
    if not cited:
        print("CANNOT FREEZE: no evidence/ or code/ paths extracted from sections/*.tex")
        return 3

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
        "round": 5,
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
