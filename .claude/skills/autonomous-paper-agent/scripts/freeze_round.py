#!/usr/bin/env python3
"""freeze_round.py — freeze an immutable, hashed blind-review snapshot.

Why not just use the two scripts that already exist
--------------------------------------------------
`scripts/freeze_paper_version.py` is the better of the two: it follows `\\input`
and `\\includegraphics` to build a real dependency closure. But it **hardcodes
the venue** at lines 32-33 -- `acl.sty` / `acl_natbib.bst`, plus a bib whitelist
of only `qcmem.bib` / `paperB.bib`. Measured: paperA now uses
`colm2026_conference.sty` and paperC uses `iclr2026_conference.sty` + `refs.bib`,
so pointing that script at paperC raises FileNotFoundError on a file that is not
supposed to be there. It also names snapshots `vN_source_<stamp>`, which the
upstream `select_best_round.py` cannot see (it matches `round_(\\d+)$`).

Upstream `make_review_snapshot.py` has the opposite problem: it takes an explicit
`--include` list and a prebuilt PDF, so it cannot discover the dependency closure
and will happily freeze a snapshot that omits a section.

This script keeps the good half of each: **discover the closure** (from
freeze_paper_version) and **emit `round_NN/` with a hashed MANIFEST** (from
make_review_snapshot), with the venue style discovered rather than assumed.

What a blind snapshot MUST NOT contain (SKILL.md:300)
----------------------------------------------------
previous reviews, previous scores, response letters, hidden author notes, or the
target score. This script copies ONLY the dependency closure plus explicitly
named evidence, and it REFUSES to copy anything under a `review_rounds/`,
`review_history/`, or `tcodex_out/` path -- those carry prior-round material and
writer notes, and leaking them would silently destroy the blindness the whole
protocol rests on.

Usage:
  python freeze_round.py paperC --round 0 \
      --evidence paperC/tcodex_out/EVIDENCE_PACK.md \
      --evidence paperC/build/build_record.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]

# Paths that must never enter a blind snapshot.
BLIND_EXCLUDE = ("review_rounds", "review_history", "tcodex_out",
                 "SCORE_HISTORY", "review_prompts", "WRITER_NOTES")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def closure(main: Path) -> tuple[set[Path], list[str]]:
    """Dependency closure of a LaTeX main file. Returns (files, missing)."""
    root = main.parent
    files: set[Path] = set()
    missing: list[str] = []
    stack = [main]
    while stack:
        cur = stack.pop()
        if cur in files or not cur.is_file():
            continue
        files.add(cur)
        text = re.sub(r"(?<!\\)%.*", "",
                      cur.read_text(encoding="utf-8", errors="replace"))
        for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", text):
            t = m.group(1).strip()
            cand = root / t
            if cand.suffix != ".tex":
                cand = root / (t + ".tex")
            (stack.append(cand) if cand.is_file() else missing.append(t))
        for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            t = m.group(1).strip()
            for ext in ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps"):
                c = root / (t + ext)
                if c.is_file():
                    files.add(c)
                    break
            else:
                missing.append(t)
    # venue style + bib + class helpers, DISCOVERED not assumed
    for pat in ("*.sty", "*.bst", "*.bib", "*.cls", "*.clo"):
        files.update(root.glob(pat))
    return files, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--main", default="main.tex")
    ap.add_argument("--evidence", action="append", default=[],
                    help="reviewer-safe evidence file to include; repeat")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    pd = Path(a.paper) if Path(a.paper).is_absolute() else REPO / a.paper
    main = pd / a.main
    if not main.is_file():
        print(f"error: {main} not found", file=sys.stderr)
        return 2

    rd = pd / "review_rounds" / f"round_{a.round:02d}"
    dest = rd / "submission"
    if dest.exists():
        if not a.force:
            print(f"error: {dest} exists; use --force", file=sys.stderr)
            return 2
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    files, missing = closure(main)
    records = []
    for f in sorted(files):
        rel = f.relative_to(pd)
        if any(x in str(rel) for x in BLIND_EXCLUDE):
            continue
        tgt = dest / "manuscript" / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, tgt)
        records.append({"snapshot_path": str(tgt.relative_to(dest)),
                        "source_path": str(f), "sha256": sha256(tgt),
                        "size_bytes": tgt.stat().st_size})

    # compiled PDF, if the build gate produced one
    pdf = pd / (Path(a.main).stem + ".pdf")
    if pdf.is_file():
        tgt = dest / "manuscript" / pdf.name
        shutil.copy2(pdf, tgt)
        records.append({"snapshot_path": str(tgt.relative_to(dest)),
                        "source_path": str(pdf), "sha256": sha256(tgt),
                        "size_bytes": tgt.stat().st_size})

    for e in a.evidence:
        src = Path(e) if Path(e).is_absolute() else REPO / e
        if not src.exists():
            print(f"error: evidence not found: {src}", file=sys.stderr)
            return 2
        srcs = [src] if src.is_file() else [x for x in src.rglob("*") if x.is_file()]
        for s in srcs:
            rel = s.name if src.is_file() else str(s.relative_to(src))
            tgt = dest / "evidence" / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, tgt)
            records.append({"snapshot_path": str(tgt.relative_to(dest)),
                            "source_path": str(s), "sha256": sha256(tgt),
                            "size_bytes": tgt.stat().st_size})

    digest = hashlib.sha256()
    for r in sorted(records, key=lambda x: x["snapshot_path"]):
        digest.update(r["snapshot_path"].encode())
        digest.update(b"\0")
        digest.update(r["sha256"].encode())
        digest.update(b"\n")

    manifest = {
        "schema_version": "1.0.0",
        "round": a.round,
        "paper": str(pd.relative_to(REPO)) if str(pd).startswith(str(REPO)) else str(pd),
        "snapshot_sha256": digest.hexdigest(),
        "n_files": len(records),
        "missing_dependencies": missing,
        "excluded_by_blindness_rule": list(BLIND_EXCLUDE),
        "blindness_note":
            "This snapshot deliberately contains NO previous reviews, scores, "
            "response letters, writer notes, or target thresholds. Reviewers "
            "must see only this directory and the rubric.",
        "files": sorted(records, key=lambda x: x["snapshot_path"]),
    }
    (dest / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in manifest.items() if k != "files"},
                     indent=2, ensure_ascii=False))
    print(f"\n[freeze_round] {len(records)} files -> {dest}", file=sys.stderr)
    if missing:
        # A missing dependency means the frozen snapshot is not the paper.
        print(f"[freeze_round] WARNING: {len(missing)} missing dependency/ies: "
              f"{missing[:5]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
