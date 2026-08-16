#!/usr/bin/env python3
"""Measure paperC main-text page cost of section / table removals, by BUILDING PDFs.

DIAGNOSTIC ONLY. Never writes into paperC/ itself: every variant is built in an
isolated scratch copy. Nothing here modifies the real paper.

Requires:
  * the in-repo TeX Live 2026 (LaTeX is NOT on $PATH -- see paperC/COMPILE_ENV_NOTE.md)
  * PyMuPDF, importable from /opt/conda/envs/torch-base/bin/python3

Usage:
  /opt/conda/envs/torch-base/bin/python3 paperC/code/measure_page_budget.py \
      --scratch /tmp/pcbudget --out paperC/state/page_budget_sweep.json

Measurement definition (identical to paperC/PAGE_LIMIT_ACTION_20260816.md):
  main-text pages = the 1-based PDF page on which the "REFERENCES" heading appears
  extent (pages)  = ((ref_page - 1) * H + (ref_y - 85.6)) / H,   H = 646.7 pt
  where 85.6 pt is the top of the body column and H the usable body column height,
  both measured off a full body page of this very document.
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
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paperC"
TEXBIN = REPO / ".texlive" / "2026" / "bin" / "x86_64-linux"

BODY_TOP_PT = 85.6      # y of first body line on a full page, measured
BODY_H_PT = 646.7       # usable body column height, measured (732.3 - 85.6)


# --------------------------------------------------------------------------- #
# build + measure
# --------------------------------------------------------------------------- #
def build(paper_dir: Path, timeout: int = 900) -> dict:
    """Full rebuild with -gg so no stale .aux can mask a broken source."""
    env = dict(os.environ)
    env["PATH"] = f"{TEXBIN}:{env.get('PATH', '')}"
    cmd = ["latexmk", "-pdf", "-bibtex", "-norc", "-gg",
           "-interaction=nonstopmode", "main.tex"]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=paper_dir, env=env, capture_output=True,
                       text=True, errors="replace", timeout=timeout)
    log = (paper_dir / "main.log")
    logtxt = log.read_text(errors="replace") if log.exists() else ""
    return {
        "rc": p.returncode,
        "secs": round(time.time() - t0, 1),
        "pdf_exists": (paper_dir / "main.pdf").exists(),
        "n_latex_error": len(re.findall(r"^! ", logtxt, re.M)),
        "n_undef_ref": len(re.findall(r"LaTeX Warning: Reference .* undefined", logtxt)),
        "n_undef_cite": len(re.findall(r"LaTeX Warning: Citation .* undefined", logtxt)),
        "n_overfull_hbox": len(re.findall(r"Overfull \\hbox", logtxt)),
        "n_overfull_vbox": len(re.findall(r"Overfull \\vbox", logtxt)),
        "first_error": (re.findall(r"^! .*$", logtxt, re.M) or [""])[0][:300],
        "stderr_tail": p.stderr[-500:],
    }


def measure(pdf: Path) -> dict:
    """Locate the References heading and report both page measures.

    Two integer measures are reported, because they differ by one exactly when
    References lands at the very top of a page:

      main_pages_refpage  = the 1-based page the References heading sits on.
                            This is the conservative reading and the one the
                            earlier artifact used.
      main_pages_occupied = the number of pages that actually carry at least one
                            block of main text, i.e. ref_page - 1 when no body
                            block on ref_page precedes the References heading.
                            This is the defensible reading: if References is the
                            first thing on page 10, then pages 1-9 hold all the
                            main text and page 10 holds none of it.

    Header ("Under review as a conference paper at ICLR 2026", y~27.8) and the
    page-number footer (y~752.2) are excluded from "body" by the 40 < y < 745
    window. The ICLR style also prints a LINE-NUMBER GUTTER at x0~73.1 (width
    ~13 pt); the real text column starts at x0~108.0. Counting the gutter as
    body text makes every page look occupied, so body blocks additionally
    require x0 >= 100.0. Both windows were measured on this document.
    """
    import pymupdf
    doc = pymupdf.open(pdf)
    ref_page = ref_y = None
    preceding_body = None
    for i, page in enumerate(doc):
        for blk in page.get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                txt = "".join(s["text"] for s in line["spans"]).strip().upper()
                if txt == "REFERENCES":
                    ref_page, ref_y = i + 1, line["bbox"][1]
                    break
            if ref_page:
                break
        if ref_page:
            # any TEXT-COLUMN block on this page starting above the heading?
            preceding_body = sum(
                1 for b in page.get_text("dict")["blocks"]
                if b.get("lines") and 40 < b["bbox"][1] < 745
                and b["bbox"][0] >= 100.0
                and b["bbox"][1] < ref_y - 1.0)
            break
    total = doc.page_count
    doc.close()
    if ref_page is None:
        return {"total_pdf_pages": total, "ref_page": None, "ref_y": None,
                "main_pages": None, "extent": None}
    extent = ((ref_page - 1) * BODY_H_PT + (ref_y - BODY_TOP_PT)) / BODY_H_PT
    occupied = ref_page if preceding_body else ref_page - 1
    return {"total_pdf_pages": total, "ref_page": ref_page,
            "ref_y": round(ref_y, 1),
            "n_body_blocks_before_refs_on_refpage": preceding_body,
            "main_pages": ref_page,               # conservative (legacy name)
            "main_pages_refpage": ref_page,
            "main_pages_occupied": occupied,
            "extent": round(extent, 3),
            "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]}


# --------------------------------------------------------------------------- #
# variant construction
# --------------------------------------------------------------------------- #
APPENDIX_STUB = r"""
\section{Relocated floats (page-budget probe)}
\label{app:probe-relocated}
%s
"""


def _appendix_file_set(paper_dir: Path) -> set:
    """Section files reachable from 09_appendix.tex, transitively, plus itself.

    Needed because "is this table in the appendix?" cannot be answered from a filename.
    09_appendix.tex \\input's 03b_nulls.tex and 09a_relocated.tex, so a table living in
    either of those is appendix content even though neither name starts with 09_appendix.
    Following the \\input graph is the only way to get this right, and getting it wrong
    produced five variants that reported a false null (see the note at the move loop).
    """
    sections = paper_dir / "sections"
    root = "09_appendix.tex"
    seen, queue = set(), [root]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        f = sections / name
        if not f.exists():
            continue
        for m in re.findall(r'\\input\{sections/([^}]+)\}', f.read_text()):
            child = m if m.endswith(".tex") else m + ".tex"
            if child not in seen:
                queue.append(child)
    return seen


def excise(paper_dir: Path, spec: list, notes: list) -> str:
    """Cut inclusive 1-based line ranges out of section files, replacing each with
    an optional stub, and return the excised material as appendix body text.

    spec entries: (relpath, first_line, last_line, stub_text_or_None, app_title)
    Line numbers refer to the ORIGINAL on-disk file; multiple ranges in the same
    file are applied bottom-up so earlier indices stay valid.
    """
    chunks = []
    by_file: dict[str, list] = {}
    for e in spec:
        by_file.setdefault(e[0], []).append(e)
    for rel, entries in by_file.items():
        f = paper_dir / rel
        lines = f.read_text().split("\n")
        for rel_, a, b, stub, title in sorted(entries, key=lambda e: -e[1]):
            cut = lines[a - 1:b]
            chunks.append((title, "\n".join(cut)))
            lines[a - 1:b] = ([stub] if stub else [])
            notes.append(f"{rel}: excised lines {a}-{b} "
                         f"({sum(len(c) for c in cut)} bytes) -> appendix "
                         f"'{title}'; stub={'yes' if stub else 'no'}"
                         f"{f' ({len(stub)} bytes)' if stub else ''}")
        f.write_text("\n".join(lines))
    out = []
    for title, body in reversed(chunks):
        out.append("\\subsection{%s}\n\\FloatBarrier\n%s\n\\FloatBarrier\n"
                   % (title, body))
    return "\n".join(out)


def apply_variant(paper_dir: Path, drop_sections: list[str],
                  move_tables: list[str], extra_edits=None,
                  relocate_sections=(), excise_spec=()) -> list[str]:
    """Mutate the scratch copy in place. Returns a list of applied-change notes."""
    notes = []
    app_extra = []
    already_applied = []

    # whole-section relocation: \input moves from main body to the appendix,
    # optionally replaced in place by a short summary stub.
    for sec, stub in relocate_sections:
        main_p = paper_dir / "main.tex"
        t = main_p.read_text()
        pat = "\\input{sections/%s}" % sec
        if pat not in t:
            # MEASURED 2026-08-17: this used to `raise SystemExit`, which killed the
            # WHOLE sweep on the FIRST variant naming a relocated section -- so C4
            # through C16 (every variant carrying relocate_sections=[("03b_nulls",...)])
            # were never evaluated, and the tool exited rc=1 after printing only the
            # four `drop_*` baselines. The cause is not a missing file: `03b_nulls` was
            # ALREADY MOVED to the appendix (that was option C4, and it was applied), so
            # main.tex now inputs `03b_nulls_summary`. The variant's premise is satisfied,
            # not violated.
            #
            # A hard exit is right for a typo and wrong for an applied change, and the
            # tool cannot tell them apart from the name alone -- so check the appendix.
            # If the section is already there, this variant measures nothing new: record
            # it as already-applied and skip it, so the remaining variants still run.
            app_txt = (paper_dir / "sections" / "09_appendix.tex").read_text()
            secfile = paper_dir / "sections" / (sec + ".tex")
            if pat in app_txt or ("sections/%s}" % sec) in app_txt:
                already_applied.append(sec)
                notes.append(f"SKIPPED: sections/{sec} is already in the appendix; "
                             f"this variant's saving is present in the baseline")
                continue
            if not secfile.exists():
                raise SystemExit(f"relocate: sections/{sec}.tex does not exist "
                                 f"(genuine typo in the variant definition, not an "
                                 f"applied relocation)")
            raise SystemExit(
                f"relocate: {pat} is not in main.tex, and sections/{sec}.tex exists but "
                f"is not \\input from 09_appendix.tex either. The section is orphaned: "
                f"resolve by hand rather than guessing.")
        repl = ("%% PROBE-RELOCATED " + pat +
                ("\n" + stub if stub else ""))
        main_p.write_text(t.replace(pat, repl))
        # demote the relocated file's \section to \subsection so it nests
        f = paper_dir / "sections" / (sec + ".tex")
        s = f.read_text()
        f.write_text(s.replace("\\section{", "\\subsection{", 1))
        app_extra.append(pat)
        notes.append(f"main.tex: relocated \\input{{sections/{sec}}} to appendix"
                     f"; stub={'yes' if stub else 'no'}"
                     f"{f' ({len(stub)} bytes)' if stub else ''}")

    if excise_spec:
        app_extra.append(excise(paper_dir, list(excise_spec), notes))

    if app_extra:
        app = paper_dir / "sections" / "09_appendix.tex"
        app.write_text(app.read_text() + APPENDIX_STUB % "\n".join(app_extra))

    main = paper_dir / "main.tex"
    txt = main.read_text()

    for sec in drop_sections:
        pat = "\\input{sections/%s}" % sec
        if pat not in txt:
            raise SystemExit(f"section input not found in main.tex: {pat}")
        txt = txt.replace(pat, "%% PROBE-DROPPED " + pat)
        notes.append(f"main.tex: commented out \\input{{sections/{sec}}}")
    main.write_text(txt)

    moved = []
    # MEASURED 2026-08-17: the appendix test below used to be
    # `not f.name.startswith("09_appendix")`, i.e. only the appendix ROOT file counted
    # as appendix. But the appendix root \input's OTHER section files
    # (03b_nulls.tex and 09a_relocated.tex), and all four MAIN_TABLES now live in those.
    # So every movetab_* variant "succeeded" while moving a table from one appendix
    # location to another -- and reported extent 9.48, byte-identical to baseline, for
    # all five table variants including movetab_all4. That reads as "relocating tables
    # buys nothing", a real-looking null, when in fact no main-text table was moved.
    # Resolve by computing the appendix file set transitively from the appendix root,
    # rather than pattern-matching one filename.
    appendix_files = _appendix_file_set(paper_dir)
    for tab in move_tables:
        pat = "\\input{sections/%s}" % tab
        hit = None
        skipped_in_appendix = []
        for f in sorted((paper_dir / "sections").glob("*.tex")):
            s = f.read_text()
            if pat not in s:
                continue
            if f.name in appendix_files:
                skipped_in_appendix.append(f.name)
                continue
            f.write_text(s.replace(pat, "%% PROBE-MOVED " + pat))
            hit = f.name
            break
        if hit is None:
            if skipped_in_appendix:
                raise SystemExit(
                    f"table {tab} is ALREADY in the appendix (found in "
                    f"{', '.join(skipped_in_appendix)}, reachable from 09_appendix.tex). "
                    f"Moving it cannot save main-text pages, so this variant would report "
                    f"a false null equal to baseline. Drop it from MAIN_TABLES.")
            raise SystemExit(f"table input not found anywhere: {pat}")
        moved.append(tab)
        notes.append(f"{hit}: removed \\input{{sections/{tab}}} -> appendix")

    if moved:
        body = "\n".join(r"\FloatBarrier" "\n" + "\\input{sections/%s}" % t
                         for t in moved) + "\n\\FloatBarrier"
        app = paper_dir / "sections" / "09_appendix.tex"
        app.write_text(app.read_text() + APPENDIX_STUB % body)

    for f_rel, old, new in (extra_edits or []):
        f = paper_dir / f_rel
        s = f.read_text()
        if old not in s:
            raise SystemExit(f"extra_edit anchor not found in {f_rel}: {old[:80]!r}")
        f.write_text(s.replace(old, new, 1))
        notes.append(f"{f_rel}: replaced {old[:60]!r} -> {new[:60]!r}")
    return notes


def run_variant(name: str, scratch: Path, drop_sections=(), move_tables=(),
                extra_edits=None, relocate_sections=(), excise_spec=(),
                desc: str = "") -> dict:
    wd = scratch / name
    if wd.exists():
        shutil.rmtree(wd)
    shutil.copytree(PAPER, wd,
                    ignore=shutil.ignore_patterns(
                        "review_rounds", "tcodex_out", "evidence", "state",
                        "gate", "*.pdf", "*.aux", "*.log", "*.fls",
                        "*.fdb_latexmk", "*.out", "*.bbl", "*.blg"))
    notes = apply_variant(wd, list(drop_sections), list(move_tables), extra_edits,
                          relocate_sections, excise_spec)
    b = build(wd)
    m = measure(wd / "main.pdf") if b["pdf_exists"] else {}
    rec = {"variant": name, "desc": desc,
           "drop_sections": list(drop_sections),
           "move_tables": list(move_tables),
           "applied": notes, "build": b, **m}
    print(json.dumps({k: rec[k] for k in
                      ("variant", "main_pages", "extent") if k in rec}
                     | {"rc": b["rc"], "err": b["n_latex_error"],
                        "undef": b["n_undef_ref"], "secs": b["secs"]}),
          flush=True)
    return rec


# --------------------------------------------------------------------------- #
# Candidate combinations.
#
# Every entry relocates material to the appendix (ICLR-legal: appendices sit
# after the bibliography and are unlimited) and leaves a real in-text stub, so
# no number, table cell, or claim is deleted. The stubs below are the actual
# strings that were compiled -- a stub costs lines too, so measuring with a
# fake short stub would understate the page count.
# --------------------------------------------------------------------------- #
# 2026-08-17: EMPTIED. Was ["tab_nulls", "tab_conventions", "tab_two_nulls", "tab_power"].
# All four are now \input from 03b_nulls.tex / 09a_relocated.tex, both of which the
# appendix root \input's -- i.e. all four are ALREADY in the appendix (verified with
# _appendix_file_set, 6/6 controls). The movetab_* variants were therefore relocating
# appendix tables to the appendix, and reported extent 9.48 for all five variants
# including movetab_all4 -- byte-identical to baseline. Read naively that says "moving
# main-text tables buys nothing", which is a false null about a lever that does not exist.
# There is no main-text table left to move. Re-populate this list only with tables that
# _appendix_file_set says are OUTSIDE the appendix.
MAIN_TABLES = []

STUB_METHOD_PRIORART = (
    r"\paragraph{Option-count-aware chance correction is decades old; only the "
    r"varying-$k$ stratification is ours.}" "\n"
    r"That a chance term should depend on how many options an item offers long "
    r"predates this work \citep{bennett1954communications,brennan1981kappa,"
    r"frary1988formula,brenner1996weightedkappa,devries2008pooledkappa}, and we "
    r"claim none of it: not correcting for the number of options, not choosing "
    r"$p_e$ as a methodological decision, not noticing that $\kappa$-family "
    r"statistics depend on $k$. What we are not aware of in that literature is a "
    r"null in which $k$ \emph{varies item to item} "
    r"($\texttt{n\_opt}\in\{3,\dots,10\}$, eight strata here), with the "
    r"permutation confined within strata so that a letter illegal for an item can "
    r"never be credited to it. Appendix~\ref{app:probe-relocated} states that "
    r"boundary claim by claim; its measured footprint is the $36$ items below."
)

STUB_METHOD_ORDERING = (
    r"The two nulls must not be conflated, and their ordering is weaker than it "
    r"first appears. Unstratified, $\sum_L p_{L}m_{L}\leq\max_Lm_{L}$ places the "
    r"v2 null at or below the v1 best-constant floor; our within-stratum "
    r"permutation only gives "
    r"$\widehat{\mathrm{acc}}\leq\sum_s w_s\max_L m_{s,L}$, whose right-hand side "
    r"\emph{dominates} $f_{\mathrm{const}}$ --- on MMLU-Pro by exactly $36$ items, "
    r"or $0.299$ pp, because the per-stratum argmax is not always-A. The ordering "
    r"holds empirically in all 27 cells of Table~\ref{tab:v2-full} but becomes a "
    r"theorem only under a regularity condition we can state and not assume, and "
    r"an \texttt{n\_opt}-conditional emitter legal on every item attains "
    r"$f_{\mathrm{const}}+0.299$ pp. Appendix~\ref{app:probe-relocated} gives the "
    r"per-stratum argmaxes, why the 27-cell count overstates the evidence, the "
    r"tightest cell's attainable violation ($0.085$ pp), and the condition. "
    r"Passing v2 must never be described as clearing the paper's floor."
)

STUB_ANALYSIS_MULT = (
    r"Two qualifications belong with this two-reference comparison, both carried "
    r"in Appendix~\ref{app:probe-relocated}. The exact McNemar instrument does not "
    r"transfer to the chance reference, which is not a $0/1$ predictor; and under "
    r"Benjamini--Hochberg at $q=0.05$ --- or Bonferroni --- \emph{neither} "
    r"reference retains a single cell (0/12 both sides), so the corrected "
    r"comparison is undefined rather than 3/12 versus 1/12. What survives "
    r"correction is the count itself: observing 3 or more rejections out of 12 has "
    r"binomial probability $0.0196$ under the global null. We therefore read this "
    r"comparison as evidence that the chance side is not uniformly null, not as a "
    r"set of three simultaneously valid per-cell claims."
)

STUB_ANALYSIS_AUDIT = (
    r"\paragraph{Auditing the audit.}" "\n"
    r"The analysis uncovered and repaired three defects without changing the "
    r"substantive verdicts: a doubled-tail bootstrap formula that produced an "
    r"illegal $p=1.042$ (re-emission changed 0/24 verdicts and moved 0/30 "
    r"$p$-values across 0.05), a sequence cap validated on the wrong tokenizer "
    r"that silently left-truncated 10/15 MMLU-Pro cells (0/14 verdicts changed, at "
    r"most 0.0083 points, but benignity was unknowable in advance), and an "
    r"out-of-memory failure on 5/8 intact-Llama-2 shards that the merge guard "
    r"correctly refused. All reported cross-family MMLU-Pro results use the "
    r"corrected launch. Table~\ref{tab:integrity} and "
    r"Appendix~\ref{app:probe-relocated} give each defect with its measured impact."
)

STUB_DISC_PRIDE = (
    r"Predictor-side debiasing is complementary rather than redundant: PriDe-style "
    r"option permutations remove a model's selection bias "
    r"\citep{zheng2024selectors} but a corrected predictor can still be judged "
    r"against the wrong reference, and a floor test does not repair a biased "
    r"predictor (Appendix~\ref{app:probe-relocated})."
)

STUB_DISC_LEDGER = (
    r"\paragraph{Current claim versus retracted claims.}" "\n"
    r"The surviving claim is narrower and stronger than the one we set out with: "
    r"under structural damage, the letter interface frequently fails its own "
    r"arm-independent best-constant floor while remaining above a conventional "
    r"chance line. It is 14/15 rather than universal at MMLU-Pro, is not "
    r"equivalent to literal constant emission, and does not by itself identify "
    r"healing or family as a cause. The flat ledger in "
    r"Appendix~\ref{app:ledger} (Table~\ref{tab:claims}) records the "
    r"self-falsification history, and Appendix~\ref{app:probe-relocated} the "
    r"status of the prospective healed arm."
)

STUB_NULLS_SECTION = (
    r"\subsection{Four under-specifications of the reference}" "\n"
    r"A content-side floor is not a dataset constant. Four choices usually left "
    r"implicit each move it by more than the effects under study, and they are "
    r"therefore part of the measured construct and must be printed with the score "
    r"(Table~\ref{tab:conventions}): the \emph{tie convention} (on MMLU-Pro, one "
    r"dataset and one tokenizer give 0.125914 under \texttt{wrong} and 0.532164 "
    r"under \texttt{credit}, a 40.6-point span whose upper end exceeds the intact "
    r"base model's own \texttt{content\_norm} score by 32.5 points, and on "
    r"ARC-Challenge \texttt{credit} alone moves five of six OLMo-2 arms from "
    r"significantly above the floor to significantly below it); the \emph{length "
    r"unit}, characters or continuation tokens (OpenBookQA \texttt{credit} is "
    r"0.416 versus 0.644; ARC-Challenge is 18.60\% tied-longest under characters "
    r"and 50.85\% under tokens); the \emph{tokenizer}, which moves \texttt{credit} "
    r"by up to 10.6 points on four-way tasks and 9.26 points on MMLU-Pro, "
    r"non-monotonically in vocabulary size "
    r"\citep{oostermeijer2026length}; and what \emph{``chance''} means when "
    r"\texttt{n\_opt} ranges from three to ten, where naive $1/10$ gives 0.100000 "
    r"and the item average $\mathrm{mean}(1/\texttt{n\_opt})$ gives 0.110877, so "
    r"the gap to the 0.116606 always-A floor is either $+1.661$ points and "
    r"$1.1661\times$ or $+0.573$ points and $1.0517\times$. The letter floor is "
    r"exempt from all four: it is a pure property of the item set's gold labels, "
    r"invariant across all 15 cross-family arms and bit-identical across all 21 "
    r"MMLU-Pro cells. Appendix~\ref{app:probe-relocated} develops each "
    r"under-specification with Tables~\ref{tab:nulls} and~\ref{tab:conventions}."
)

# (relpath, first_line, last_line, stub_or_None, appendix_subsection_title)
EX_METHOD_PRIORART = ("sections/03_method.tex", 50, 66, STUB_METHOD_PRIORART,
                      "Prior art on option-count-aware chance correction")
EX_METHOD_ORDERING = ("sections/03_method.tex", 80, 80, STUB_METHOD_ORDERING,
                      "Why the v1/v2 ordering is not an identity")
EX_ANALYSIS_MULT = ("sections/05_analysis.tex", 24, 24, STUB_ANALYSIS_MULT,
                    "Multiplicity in the two-reference comparison")
EX_ANALYSIS_AUDIT = ("sections/05_analysis.tex", 54, 59, STUB_ANALYSIS_AUDIT,
                     "Auditing the audit: defects found and repaired")
EX_DISC_PRIDE = ("sections/06_discussion.tex", 8, 9, STUB_DISC_PRIDE,
                 "Relationship to prediction debiasing")
EX_DISC_LEDGER = ("sections/06_discussion.tex", 14, 18, STUB_DISC_LEDGER,
                  "Claim history and the prospective healed arm")

STUB_EXP_MULT = (
    r"\paragraph{Multiplicity.} We report per-cell $\alpha=0.05$ decisions across "
    r"many cells without a family-wise correction, so the headline counts are "
    r"counts of per-cell decisions rather than simultaneously valid claims. The "
    r"cells share items, nest arms, and share a null, so an off-the-shelf "
    r"correction would not have a defensible family definition here; the "
    r"conclusions rest on the direction and near-unanimity of the aggregate (for "
    r"instance 0/60 damaged cells clearing their floor), not on any single cell "
    r"crossing $\alpha$. Appendix~\ref{app:probe-relocated} states the "
    r"limitation in full, together with the definition of a \emph{designated "
    r"damaged} cell that fixes every denominator quoted below."
)

STUB_ANALYSIS_OFFMMLU = (
    r"The most vivid small-benchmark cases are literal constant emitters: sixteen "
    r"damaged cross-family cells have accuracy equal to the marginal of the "
    r"emitted letter to machine precision, and two OpenBookQA cells emit A on "
    r"every item, score exactly 0.276000, and land on the optimal constant with "
    r"$\Delta=0.000$ points and CI95 $[0,0]$ --- yet read 2.6 points above chance "
    r"0.25. Off MMLU, the designated OLMo-2 subset gives 10/15 above chance and "
    r"0/15 above floor; across the broader non-OLMo replication 0/60 damaged cells "
    r"clear their floor while 25/60 read above chance, and only 7/60 are "
    r"significantly below because 52/60 are underpowered for the MMLU reference "
    r"effect. That is a power limit, not a small effect: ARC-Challenge's median "
    r"damaged effect ($-3.840$ points) is larger than MMLU's ($-3.603$), but its "
    r"half-width is 3.92 rather than 1.18 points "
    r"(Appendix~\ref{app:probe-relocated})."
)

EX_EXP_MULT = ("sections/04_experiments.tex", 24, 26, STUB_EXP_MULT,
               "Designated damaged cells, and multiplicity")
EX_ANALYSIS_OFFMMLU = ("sections/05_analysis.tex", 28, 30, STUB_ANALYSIS_OFFMMLU,
                       "Constant emitters and the off-MMLU replication")

STUB_ANALYSIS_READOUT = (
    r"\paragraph{Read-out versus retained knowledge.}" "\n"
    r"A floor-level letter score does not imply that all task-relevant competence "
    r"is gone. On ARC-Easy, OLMo-2 \texttt{keep8} is statistically at its "
    r"0.266414 letter floor (0.2584) while \texttt{content\_norm} reaches 0.6460, "
    r"a paired gap of $+38.76$ points with McNemar "
    r"$p=9.8\times10^{-148}$: the arm can rank answer contents but cannot express "
    r"that knowledge through the damaged letter interface. Healthy models often "
    r"favour letters, so ``content is the fair interface'' is not a general "
    r"conclusion, and the residual-fraction correction is construct-specific --- "
    r"using chance rather than the token-longest floor inflates OpenBookQA's "
    r"residual by $2.11\times$ but moves PIQA and ARC-Easy to $0.90\times$ and "
    r"$0.98\times$. Null calibration can increase or decrease the residual "
    r"(Appendix~\ref{app:probe-relocated})."
)

STUB_ANALYSIS_FP32 = (
    r"\paragraph{Full precision falsifies the numerical-tie mechanism.}" "\n"
    r"For OLMo-2 \texttt{keep8} on MMLU, fp32 removes all 4{,}303 bf16 exact "
    r"top-two ties and changes 2{,}532 of 14{,}042 letter argmax decisions "
    r"(18.03\%), yet letter accuracy changes by only $-0.0015$, CI95 "
    r"$[-0.0064,+0.0033]$, exact McNemar $p=0.5702$, and the arm is \emph{more} "
    r"significantly below its floor in fp32 ($-1.538$ points, $p=0.0060$, versus "
    r"$-1.389$, $p=0.0190$). Higher precision reshuffles ambiguous decisions; it "
    r"cannot create item-level information that is absent "
    r"(Appendix~\ref{app:probe-relocated})."
)

EX_ANALYSIS_READOUT = ("sections/05_analysis.tex", 32, 36, STUB_ANALYSIS_READOUT,
                       "Read-out versus retained knowledge")
EX_ANALYSIS_FP32 = ("sections/05_analysis.tex", 50, 52, STUB_ANALYSIS_FP32,
                    "Full precision and the numerical-tie mechanism")

CANDIDATES = {
    # C1: prose relocation only, tables stay in the main text.
    "C1_prose_only": dict(
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT],
        desc="4 prose blocks (2 method, 2 analysis) -> appendix w/ stubs; "
             "all 4 tables stay in main text"),
    # C2: C1 + every remaining main-text table relocated.
    "C2_prose_plus_all4tabs": dict(
        move_tables=MAIN_TABLES,
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT],
        desc="C1 + all 4 remaining main-text tables -> appendix"),
    # C3: C2 + discussion de-duplication against 02_related / 07_limitations.
    "C3_C2_plus_discussion_dedup": dict(
        move_tables=MAIN_TABLES,
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT,
                     EX_DISC_PRIDE, EX_DISC_LEDGER],
        desc="C2 + the two Discussion paragraphs that restate 02_related and "
             "07_limitations -> appendix"),
    # C4: whole 03b_nulls body to the appendix, tables travel with it.
    "C4_nulls_section_relocated": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        desc="03b_nulls body (incl. tab:nulls, tab:conventions) -> appendix with "
             "a one-paragraph in-text summary; tab_two_nulls + tab_power moved too"),
    # C5: C4 + the analysis/method prose relocations.
    "C5_nulls_plus_prose": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_ORDERING, EX_ANALYSIS_AUDIT],
        desc="C4 + method ordering paragraph + Auditing-the-audit -> appendix"),
    # C6: C3 without the tab:nulls move (does the central table have to go?).
    "C6_C3_keep_tab_nulls": dict(
        move_tables=["tab_conventions", "tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT,
                     EX_DISC_PRIDE, EX_DISC_LEDGER],
        desc="C3 but tab:nulls stays in the main text"),
    # C7: C3 + two more prose relocations (experiments multiplicity, off-MMLU).
    "C7_C3_plus_two_more": dict(
        move_tables=MAIN_TABLES,
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT,
                     EX_DISC_PRIDE, EX_DISC_LEDGER,
                     EX_EXP_MULT, EX_ANALYSIS_OFFMMLU],
        desc="C3 + the Experiments multiplicity/designated-damaged paragraphs and "
             "the off-MMLU replication paragraph -> appendix"),
    # C8: 03b_nulls section relocation combined with all of C7's prose moves.
    "C8_nulls_section_plus_C7prose": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT,
                     EX_DISC_PRIDE, EX_DISC_LEDGER,
                     EX_EXP_MULT, EX_ANALYSIS_OFFMMLU],
        desc="03b_nulls body -> appendix with a one-paragraph summary, plus all "
             "eight prose relocations and both remaining tables"),
    # C9: C8 without the two Discussion moves -- is Discussion needed for 9?
    "C9_C8_minus_discussion": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT,
                     EX_EXP_MULT, EX_ANALYSIS_OFFMMLU],
        desc="C8 with the Discussion left untouched"),
    # C10: C8 but tab_power stays in the main text next to the power argument.
    "C10_C8_keep_tab_power": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls"],
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT,
                     EX_DISC_PRIDE, EX_DISC_LEDGER,
                     EX_EXP_MULT, EX_ANALYSIS_OFFMMLU],
        desc="C8 but tab:power stays in the main text"),
    # --- minimality probes: strip pieces off C9 until 9 pages is lost --------- #
    "C11_C9_minus_offmmlu": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT, EX_EXP_MULT],
        desc="C9 minus the off-MMLU/constant-emitter relocation"),
    "C12_C9_minus_expmult": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT,
                     EX_ANALYSIS_OFFMMLU],
        desc="C9 minus the Experiments multiplicity relocation"),
    "C13_C9_minus_priorart": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_ORDERING, EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT,
                     EX_EXP_MULT, EX_ANALYSIS_OFFMMLU],
        desc="C9 minus the prior-art relocation (that paragraph answers a "
             "novelty reviewer and is cheap to keep)"),
    "C14_C9_minus_analysis_mult": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING, EX_ANALYSIS_AUDIT,
                     EX_EXP_MULT, EX_ANALYSIS_OFFMMLU],
        desc="C9 minus the analysis multiplicity relocation"),
    "C15_C9_minus_audit": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING, EX_ANALYSIS_MULT,
                     EX_EXP_MULT, EX_ANALYSIS_OFFMMLU],
        desc="C9 minus the Auditing-the-audit relocation"),
    "C16_C9_minus_ordering": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_PRIORART, EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT,
                     EX_EXP_MULT, EX_ANALYSIS_OFFMMLU],
        desc="C9 minus the v1/v2-ordering relocation"),
    # --- leave-TWO-out: how far can C9 be trimmed and still land on 9? ------- #
    "C17_C9_minus_priorart_analysismult": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_ORDERING, EX_ANALYSIS_AUDIT,
                     EX_EXP_MULT, EX_ANALYSIS_OFFMMLU],
        desc="C9 keeping BOTH the prior-art and the analysis-multiplicity "
             "paragraphs in the main text"),
    "C18_C9_minus_audit_expmult": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_OFFMMLU],
        desc="C9 keeping Auditing-the-audit and the Experiments multiplicity "
             "paragraph in the main text"),
    "C19_C9_minus_offmmlu_expmult": dict(
        relocate_sections=[("03b_nulls", STUB_NULLS_SECTION)],
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT],
        desc="C9 keeping the off-MMLU replication and the Experiments "
             "multiplicity paragraph in the main text (== C5 plus priorart "
             "and analysis-multiplicity)"),
    # --- alternative axis: keep 03b_nulls whole, relocate 05_analysis tail --- #
    "C20_keep_nulls_relocate_analysis_tail": dict(
        move_tables=MAIN_TABLES,
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT,
                     EX_DISC_PRIDE, EX_DISC_LEDGER,
                     EX_EXP_MULT, EX_ANALYSIS_OFFMMLU,
                     EX_ANALYSIS_READOUT, EX_ANALYSIS_FP32],
        desc="03b_nulls stays whole in the main text; instead relocate the "
             "read-out-vs-retained-knowledge and full-precision subsections plus "
             "all of C7's prose and all 4 tables"),
    "C21_C20_minus_two_tabs": dict(
        move_tables=["tab_two_nulls", "tab_power"],
        excise_spec=[EX_METHOD_PRIORART, EX_METHOD_ORDERING,
                     EX_ANALYSIS_MULT, EX_ANALYSIS_AUDIT,
                     EX_DISC_PRIDE, EX_DISC_LEDGER,
                     EX_EXP_MULT, EX_ANALYSIS_OFFMMLU,
                     EX_ANALYSIS_READOUT, EX_ANALYSIS_FP32],
        desc="C20 but tab:nulls and tab:conventions stay in the main text with "
             "their (unrelocated) 03b_nulls section"),
}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/tmp/pcbudget")
    ap.add_argument("--out", default=str(PAPER / "state" / "page_budget_sweep.json"))
    ap.add_argument("--only", default="", help="comma-separated variant names")
    ap.add_argument("--mode", default="all", choices=["all", "sweep", "candidates"])
    args = ap.parse_args()
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    # 2026-08-17: was "03b_nulls". main.tex has inputted `03b_nulls_summary` since the
    # C4 relocation was applied (the full nulls body now lives in the appendix), so the
    # old name made the sweep die at `drop_03b_nulls` before reaching any C-variant.
    # The measurable main-text section is the SUMMARY -- that is what still costs pages.
    SECTIONS = ["00_abstract", "01_introduction", "02_related", "03_method",
                "03b_nulls_summary", "04_experiments", "05_analysis", "06_discussion",
                "07_limitations"]

    variants = []
    if args.mode in ("all", "sweep"):
        variants.append(("baseline", dict(desc="on-disk state, unmodified")))
        for s in SECTIONS:
            variants.append((f"drop_{s}",
                             dict(drop_sections=[s],
                                  desc=f"section {s} \\input removed")))
        for t in MAIN_TABLES:
            variants.append((f"movetab_{t}",
                             dict(move_tables=[t],
                                  desc=f"{t} relocated to appendix")))
        # Guarded: with MAIN_TABLES empty this variant would move nothing and report
        # extent == baseline, i.e. a measurement-shaped no-op. Emit it only if there is
        # actually something to move.
        if MAIN_TABLES:
            variants.append((f"movetab_all{len(MAIN_TABLES)}",
                             dict(move_tables=MAIN_TABLES,
                                  desc=f"all {len(MAIN_TABLES)} remaining main-text "
                                       f"tables to appendix")))
    if args.mode in ("all", "candidates"):
        variants.extend(CANDIDATES.items())

    keep = set(x for x in args.only.split(",") if x)
    out = []
    for name, kw in variants:
        if keep and name not in keep:
            continue
        out.append(run_variant(name, scratch, **kw))

    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(op.read_text()) if op.exists() else []
    by = {r["variant"]: r for r in prev}
    by.update({r["variant"]: r for r in out})
    # Drop records for variants this build no longer defines. `by.update()` alone MERGES,
    # so a retired variant keeps its last number forever: measured 2026-08-17, the file
    # still carried movetab_tab_{nulls,conventions,two_nulls,power} at extent 9.48 after
    # those variants had been removed for being no-ops, and the summary line said
    # "15 variants" while only 11 had run. A stale row is worse than a missing one --
    # it reads as a current measurement. Only prune on a FULL sweep: with --only the
    # non-selected variants were never attempted, so their prior records are still valid.
    dropped = []
    if not keep:
        defined = {name for name, _ in variants}
        dropped = sorted(set(by) - defined)
        for name in dropped:
            del by[name]
    op.write_text(json.dumps(sorted(by.values(), key=lambda r: r["variant"]),
                             indent=2))
    print(f"wrote {op}  ({len(by)} variants"
          + (f"; pruned {len(dropped)} retired: {', '.join(dropped)}" if dropped else "")
          + ")", file=sys.stderr)


if __name__ == "__main__":
    main()
