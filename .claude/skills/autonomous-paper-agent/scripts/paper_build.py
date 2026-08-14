#!/usr/bin/env python3
"""paper_build.py — compile a paper and emit a machine-readable build_record.json.

Why this exists
---------------
The `autonomous-paper-agent` skill has an integrity gate that says "the LaTeX
package compiles" and "the PDF has been visually inspected". A gate you cannot
execute is not a gate -- it silently reads as passing, which is exactly the
failure mode `proposal/LIFECYCLE_SCHEMA.md:151` warns about (a detailed process
file that returned empty for two months and therefore "looked fine").

So this script makes that gate real, and it is fail-closed: it reports
`compiled: false` with the reason rather than letting a caller assume success.

★ THE TOOLCHAIN IS PRESENT. IT IS JUST NOT ON $PATH.
------------------------------------------------------
`paperA/V6_ITERATION_NOTES.md:82` records "The local environment lacked a
`latexmk` executable". That was true of `$PATH` and false of the disk. Measured
2026-08-14: `.texlive/2026/bin/x86_64-linux/` holds **142 binaries** including
`pdflatex`, `lualatex`, `xelatex`, `latexmk`, `bibtex`, `biber`, and the real
29-page paperA compiles from it with rc=0. `scripts/freeze_paper_version.py:23-28`
already prepends this directory -- the knowledge existed but was not centralised.

This is the same class of error as the paperC E1 gap ("no python has pyarrow"
-- true of the interpreters checked, false of `venv_union9`). Generalised
lesson: **"tool X is unavailable" requires enumerating locations, not just
asking the shell.**

Per-paper engine/style reality (measured, do not guess)
------------------------------------------------------
  paperA -> colm2026_conference.sty   (BUILD.md says latexmk -pdf; v6 used XeTeX)
  paperB -> acl.sty + acl_natbib.bst
  paperC -> iclr2026_conference.sty + iclr2026_conference.bst + refs.bib
`scripts/freeze_paper_version.py:32-33` hardcodes `acl.sty`/`acl_natbib.bst` and
a bib whitelist of only `qcmem.bib`/`paperB.bib`, so it raises FileNotFoundError
on paperC. This script discovers style/bib files instead of assuming them.

What it checks (all counted, none inferred)
-------------------------------------------
  * rc of the build
  * PDF produced, its byte size, page count, SHA-256
  * undefined references / undefined citations  (LaTeX warnings)
  * overfull / underfull boxes
  * missing \input targets  (a static pre-check, because a missing section can
    silently vanish from the PDF with only a warning)
  * unresolved \cite keys against the .bib
  * whether every `\input` target actually exists on disk

Usage:
  python paper_build.py paperC
  python paper_build.py paperC --engine pdflatex --out paperC/build/build_record.json
  python paper_build.py paperC --check-only     # static checks, no compile
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TEXBIN = REPO / ".texlive" / "2026" / "bin" / "x86_64-linux"


def texenv():
    """PATH with the project's TeX Live tree prepended. This is the whole trick."""
    env = dict(os.environ)
    env["PATH"] = f"{TEXBIN}:{env.get('PATH', '')}"
    # keep builds reproducible + non-interactive
    env["SOURCE_DATE_EPOCH"] = env.get("SOURCE_DATE_EPOCH", "1600000000")
    return env


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def toolchain_report() -> dict:
    env = texenv()
    found = {}
    for tool in ("pdflatex", "lualatex", "xelatex", "latexmk", "bibtex", "biber"):
        found[tool] = shutil.which(tool, path=env["PATH"]) or None
    return {
        "texlive_dir": str(TEXBIN),
        "texlive_dir_exists": TEXBIN.is_dir(),
        "n_binaries": len(list(TEXBIN.iterdir())) if TEXBIN.is_dir() else 0,
        "tools": found,
    }


def find_inputs(main: Path) -> tuple[list[str], list[str]]:
    """Return (declared \\input/\\include targets, ones missing on disk)."""
    root = main.parent
    declared, missing = [], []
    seen = set()
    stack = [main]
    while stack:
        cur = stack.pop()
        if cur in seen or not cur.is_file():
            continue
        seen.add(cur)
        text = cur.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"(?<!\\)%.*", "", text)          # strip comments
        for m in re.finditer(r"\\(?:input|include)\{([^}]+)\}", text):
            target = m.group(1).strip()
            declared.append(target)
            cand = root / target
            if cand.suffix != ".tex":
                cand = root / (target + ".tex")
            if cand.is_file():
                stack.append(cand)
            else:
                missing.append(target)
    return declared, missing


def cite_keys(main: Path) -> tuple[list[str], list[str]]:
    """Return (cited keys, keys with no @entry in any .bib next to the paper)."""
    root = main.parent
    text = ""
    for tex in list(root.rglob("*.tex")):
        text += re.sub(r"(?<!\\)%.*", "", tex.read_text(encoding="utf-8", errors="replace"))
    cited = set()
    for m in re.finditer(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\s*\{([^}]+)\}", text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                cited.add(k)
    defined = set()
    for bib in root.glob("*.bib"):
        btext = bib.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)", btext):
            defined.add(m.group(1).strip())
    return sorted(cited), sorted(cited - defined)


def parse_log(log: str) -> dict:
    undef_ref = re.findall(r"Reference `([^']+)' on page \d+ undefined", log)
    undef_cit = re.findall(r"Citation `([^']+)' on page \d+ undefined", log)
    return {
        "undefined_references": sorted(set(undef_ref)),
        "n_undefined_references": len(undef_ref),
        "undefined_citations": sorted(set(undef_cit)),
        "n_undefined_citations": len(undef_cit),
        "n_overfull_hbox": len(re.findall(r"Overfull \\hbox", log)),
        "n_overfull_vbox": len(re.findall(r"Overfull \\vbox", log)),
        "n_underfull_hbox": len(re.findall(r"Underfull \\hbox", log)),
        "n_latex_errors": len(re.findall(r"^! ", log, re.M)),
        "first_errors": re.findall(r"^! .*$", log, re.M)[:5],
        "n_missing_file": len(re.findall(r"File `[^']+' not found", log)),
    }


def page_count(pdf: Path, log_text: str = "") -> int | None:
    """Page count. Prefers the TeX log, which states it authoritatively.

    Do NOT grep the PDF for `/Type /Page`: modern pdfTeX writes page objects
    into COMPRESSED object streams (`/ObjStm`), so those tokens are not present
    as plaintext. Measured on paperC 2026-08-14: the regex found 0 pages in a
    genuine 16-page PDF, and `/Count` was absent too. The first version of this
    function reported `pdf_pages: null` for a perfectly good build -- a silent
    null in a gate is exactly the failure this script exists to prevent, so the
    log is now the primary source and a raw-scan is only the fallback.
    """
    m = re.findall(r"Output written on \S+ \((\d+) pages?", log_text)
    if m:
        return int(m[-1])
    try:
        data = pdf.read_bytes()
    except OSError:
        return None
    n = len(re.findall(rb"/Type\s*/Page[^s]", data))
    if n:
        return n
    counts = re.findall(rb"/Count\s+(\d+)", data)
    if counts:
        return int(max(counts, key=lambda x: int(x)))
    return None   # honest null: uncompressed scan failed and no log line


def build(paper_dir: Path, engine: str, main_name: str, check_only: bool) -> dict:
    main = paper_dir / main_name
    rec: dict = {
        "schema_version": "1.0.0",
        "paper_dir": str(paper_dir.relative_to(REPO)) if str(paper_dir).startswith(str(REPO)) else str(paper_dir),
        "main_tex": main_name,
        "engine": engine,
        "toolchain": toolchain_report(),
    }
    if not main.is_file():
        rec.update(compiled=False, reason=f"main tex not found: {main}")
        return rec

    # --- static checks first: these catch things a compile can hide ----------
    declared, missing_inputs = find_inputs(main)
    cited, unresolved = cite_keys(main)
    styles = sorted(p.name for p in paper_dir.glob("*.sty"))
    bibs = sorted(p.name for p in paper_dir.glob("*.bib"))
    rec["static"] = {
        "n_input_targets": len(declared),
        "missing_input_targets": missing_inputs,
        "n_cite_keys": len(cited),
        "unresolved_cite_keys": unresolved,
        "style_files_present": styles,
        "bib_files_present": bibs,
    }

    if check_only:
        rec.update(compiled=None, reason="check-only: no compile attempted")
        return rec

    tool = shutil.which(engine, path=texenv()["PATH"])
    if not tool:
        rec.update(compiled=False,
                   reason=f"{engine} not found even with .texlive on PATH")
        return rec

    if engine == "latexmk":
        # -gg forces a full rebuild so a stale .aux cannot mask a broken source.
        cmd = [tool, "-pdf", "-bibtex", "-norc", "-gg",
               "-interaction=nonstopmode", main_name]
    else:
        cmd = [tool, "-interaction=nonstopmode", main_name]

    rec["command"] = " ".join(cmd)
    passes = []
    try:
        n_pass = 1 if engine == "latexmk" else 3   # plain engines need 3 for refs
        for i in range(n_pass):
            p = subprocess.run(cmd, cwd=paper_dir, env=texenv(),
                               capture_output=True, text=True, timeout=900)
            passes.append({"pass": i + 1, "rc": p.returncode})
            if engine != "latexmk" and i == 0:
                bib = shutil.which("bibtex", path=texenv()["PATH"])
                if bib and bibs:
                    b = subprocess.run([bib, Path(main_name).stem], cwd=paper_dir,
                                       env=texenv(), capture_output=True,
                                       text=True, timeout=300)
                    passes.append({"pass": "bibtex", "rc": b.returncode})
    except subprocess.TimeoutExpired:
        rec.update(compiled=False, reason="build timed out (900s)", passes=passes)
        return rec

    rec["passes"] = passes
    logf = paper_dir / (Path(main_name).stem + ".log")
    log_text = logf.read_text(encoding="utf-8", errors="replace") if logf.is_file() else ""
    rec["diagnostics"] = parse_log(log_text) if log_text else {"note": "no .log produced"}

    pdf = paper_dir / (Path(main_name).stem + ".pdf")
    if pdf.is_file():
        rec.update(
            compiled=True,
            pdf=str(pdf.relative_to(paper_dir)),
            pdf_bytes=pdf.stat().st_size,
            pdf_sha256=sha256(pdf),
            pdf_pages=page_count(pdf, log_text),
        )
    else:
        rec.update(compiled=False, reason="no PDF produced")

    # --- the gate verdict, computed not narrated ----------------------------
    d = rec.get("diagnostics", {})
    blockers = []
    if not rec.get("compiled"):
        blockers.append("did not compile")
    if missing_inputs:
        blockers.append(f"{len(missing_inputs)} missing \\input target(s)")
    if unresolved:
        blockers.append(f"{len(unresolved)} unresolved \\cite key(s)")
    if d.get("n_undefined_references"):
        blockers.append(f"{d['n_undefined_references']} undefined reference(s)")
    if d.get("n_undefined_citations"):
        blockers.append(f"{d['n_undefined_citations']} undefined citation(s)")
    if d.get("n_latex_errors"):
        blockers.append(f"{d['n_latex_errors']} LaTeX error(s)")
    rec["build_gate_pass"] = not blockers
    rec["build_gate_blockers"] = blockers
    # Visual inspection cannot be done by this script. Say so rather than
    # letting the caller read a green build as "PDF inspected".
    rec["pdf_visually_inspected"] = False
    rec["pdf_visual_inspection_note"] = (
        "NOT DONE BY THIS SCRIPT. No pdftoppm/gs/mutool/PyMuPDF on this host "
        "(measured 2026-08-14), so page images cannot be rendered here. The "
        "skill's integrity gate item 'PDF visually inspected' therefore remains "
        "OPEN unless a human or a PDF-capable host confirms it."
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper", help="paper dir, e.g. paperC (relative to repo root) or a path")
    ap.add_argument("--engine", default="latexmk",
                    choices=["latexmk", "pdflatex", "xelatex", "lualatex"])
    ap.add_argument("--main", default="main.tex")
    ap.add_argument("--out", default=None, help="build_record.json path")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    pd = Path(a.paper)
    if not pd.is_absolute():
        pd = (REPO / a.paper) if (REPO / a.paper).is_dir() else Path(a.paper).resolve()
    if not pd.is_dir():
        print(f"error: not a directory: {pd}", file=sys.stderr)
        return 2

    rec = build(pd, a.engine, a.main, a.check_only)
    out = Path(a.out) if a.out else pd / "build" / "build_record.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(rec, indent=2, ensure_ascii=False))
    print(f"\n[paper_build] wrote {out}", file=sys.stderr)
    if rec.get("build_gate_pass") is True:
        print("[paper_build] BUILD GATE: PASS "
              "(note: pdf_visually_inspected is still false)", file=sys.stderr)
    else:
        print(f"[paper_build] BUILD GATE: FAIL -> {rec.get('build_gate_blockers') or rec.get('reason')}",
              file=sys.stderr)
    # exit 1 on gate failure so a caller cannot ignore it by accident
    return 0 if rec.get("build_gate_pass") or a.check_only else 1


if __name__ == "__main__":
    sys.exit(main())
