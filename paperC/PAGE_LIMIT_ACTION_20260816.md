# paperC — ICLR 9-page compliance: measured action record (2026-08-16)

**Headline: the two-table lever was applied and measured. It did NOT reach 9 pages.
Main text is still 12 pages (References begins on page 12). Residual overflow = 3 pages.
Two tables buy 0.515 pages of *extent*, not a single integer page.**

This file reports only what was built and measured. Nothing here is predicted.

---

## 1. The build command

TeX Live 2026 is installed **inside the repo** and is **not on `$PATH`**. `command -v pdflatex`
returns empty; that does not mean latex is absent (see `paperC/COMPILE_ENV_NOTE.md`).

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
export PATH="$PWD/.texlive/2026/bin/x86_64-linux:$PATH"
cd paperC
latexmk -pdf -bibtex -norc -gg -interaction=nonstopmode main.tex
```

- Engine, from `main.log` line 1: `pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)`.
- `-gg` forces a full rebuild so a stale `.aux` cannot mask a broken source.
- **Result: rc=0, 0 LaTeX errors, 0 undefined references, 0 undefined citations,
  0 overfull hbox, 0 overfull vbox** (before and after the change).
- Equivalent wrapper, also run and also rc=0:
  `python3 .claude/skills/autonomous-paper-agent/scripts/paper_build.py paperC`
  → `paperC/gate/build_record.json`, `build_gate_pass: true`, `build_gate_blockers: []`.

## 2. How the References boundary was determined

The ICLR 2026 limit is **9 pages of main text**; references and appendices are excluded and
appendices must sit after the bibliography (verbatim quotes and provenance already recorded in
`paperC/gate/venue_page_limit.json`, fetched HTTP 200 from the ICLR 2026 Author Guide). So the
number that matters is *the page on which the `References` heading appears*.

Earlier notes in this repo state that no PDF tooling exists on this host
(`build_record.json: pdf_visual_inspection_note`, and the `no_pdf_tooling_note` in
`venue_page_limit.json`). **That is measured-false.** `PyMuPDF 1.28.2` and `pypdf 6.16.1` both
import from `/opt/conda/envs/torch-base/bin/python3`. The boundary was therefore read directly
out of the PDF geometry rather than inferred from `main.log`/`main.aux`:

```python
import pymupdf
doc = pymupdf.open("paperC/main.pdf")
for i, page in enumerate(doc):
    for blk in page.get_text("dict")["blocks"]:
        for line in blk.get("lines", []):
            if "".join(s["text"] for s in line["spans"]).strip().upper() == "REFERENCES":
                print(i + 1, line["bbox"][1])       # page number, y offset in points
```

Body text on a full page runs `y = 85.6 .. 732.3` pt (measured on page 5), i.e. a usable column
height of **H = 646.7 pt**. "Extent" below = `(ref_page - 1) * H + (ref_y - 85.6)`, divided by H.
It is a continuous measure of how much main text there is; the *integer* page count is what ICLR
enforces.

## 3. Measured before / after

| | total PDF pages | References page | References y | **main-text pages** | extent (pages) |
|---|---|---|---|---|---|
| **BEFORE** | 21 | 12 | 613.2 pt (77.4% down) | **12** | 11.816 |
| **AFTER** | 22 | 12 | 280.1 pt (35.4% down) | **12** | 11.301 |

- `pdf_sha256` AFTER: `56a376e128c358e9f09f2daeee7f543a3717c5cab318e113d28c4102a56e2485` (22 pages).
- BEFORE PDF preserved at `/tmp/paperC_BEFORE.pdf` (sha256
  `20545b200dbbc04a0bf785530c74fa2ce433c58381304e2d61197cbf5acc051d`, matches the
  `gate/build_record.json` that was on disk, so BEFORE is the genuine prior state and not a
  re-derivation).
- Total PDF grew 21 → 22 because the appendix absorbed the two tables. That is expected and
  irrelevant: appendix pages are unlimited.

**Measured saving: 0.515 pages of extent. Integer main-text pages: 12 → 12. The boundary did
not cross.** References moved from 77.4% down page 12 to 35.4% down page 12.

**Residual overflow vs the 9-page limit: 3 integer pages (2.301 pages of extent).**

## 4. Exactly what was moved

Two `table*` floats relocated from the main text into a new first appendix subsection.
**Both table `.tex` files are byte-identical — `git diff --stat` on them is empty.** No number
was retyped, rounded, or deleted.

| table | label | was `\input` at | now `\input` at | lands on PDF page |
|---|---|---|---|---|
| `sections/tab_mmlupro.tex` | `tab:mmlupro` | `sections/05_analysis.tex:32` | `sections/09_appendix.tex` (new §A.1) | 14 |
| `sections/tab_v2_resort.tex` | `tab:v2-resort` | `sections/05_analysis.tex:52` | `sections/09_appendix.tex` (new §A.1) | 15 |

Both are now **after** `\bibliography` (References p12, Appendix heading p14), which is what the
ICLR guide requires of appendix material.

New appendix subsection added, with `\label{app:mmlupro-and-resort}` and two `\FloatBarrier`s
matching the existing appendix convention. It states that every value is unchanged from what the
main-text subsections discuss.

### Prose sentences repaired

No sentence in paperC ever said "the table below" — a grep for
`table below|table above|following table|shown below|listed below|next table` across
`sections/` and `main.tex` returns **zero hits**; every reference is a `Table~\ref{}`. So no
positional-language breakage existed. Four edits were still needed, all in
`sections/05_analysis.tex`:

1. **line 5** — added the destination so the reader is not sent hunting:
   `MMLU-Pro provides the cleanest test (Table~\ref{tab:mmlupro}).`
   → `... (per-cell values in Table~\ref{tab:mmlupro}, Appendix~\ref{app:mmlupro-and-resort}).`
2. **line 44** — same treatment:
   `V2 is not uniformly conservative (Table~\ref{tab:v2-resort}).`
   → `V2 is not uniformly conservative (Table~\ref{tab:v2-resort}, Appendix~\ref{app:mmlupro-and-resort}).`
3. **line 24, opening** — `Two qualifications belong with this table.`
   → `Two qualifications belong with this two-reference comparison.`
4. **line 24, closing** — `We therefore read this table as evidence that ...`
   → `We therefore read this comparison as evidence that ...`

Edits 3 and 4 matter and are easy to miss. "This table" in that paragraph refers to the
**unnumbered inline `tabular`** at `05_analysis.tex:11-20` (the chance-vs-floor
point-estimate/CI comparison, which stays in the main text), *not* to `tab:mmlupro`. With
`tab:mmlupro` no longer sitting a few lines below, a reader would have re-bound "this table" to
the wrong referent. The referent is now named explicitly.

### What was deliberately NOT done

No font-size change, no margin change, no `\vspace` insertion, no `\tabcolsep`/`\arraystretch`
edit, no change to `iclr2026_conference.sty` (still byte-identical to the official zip, md5
`a040392a6cfee8dc54aae17eb2635282`, per `gate/venue_page_limit.json`). ICLR treats style-file
fighting as its own desk-reject trigger.

### Integrity checkers re-run (both rc=0, unchanged from before the move)

```
code/check_prose_vs_evidence.py  -> n_checked=81 n_ok=81 n_mismatch=0 n_uncovered=0   (rc=0)
code/validate_tex_static.py      -> all files OK                                       (rc=0)
```

## 5. `paper_build.py` — the bug is REAL, and it was not where it was described

Location: `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/.claude/skills/autonomous-paper-agent/scripts/paper_build.py`

The report said it "decodes log/source output as strict UTF-8". **The `read_text` calls were
already safe** — lines 111, 131, 140, 255, 323 all pass `errors="replace"` and were left alone.
Reporting only those would have been a false all-clear.

The real strict decode was in the **subprocess** calls: `text=True` with no `errors=` uses
strict UTF-8 on the child's stdout/stderr. Verified empirically, not assumed:

```
subprocess.run(["printf", "\xff\xfe bad"], capture_output=True, text=True)
  -> UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0
subprocess.run([...], capture_output=True, text=True, errors="replace")
  -> OK, '�� bad'
```

This crashes the whole gate rather than degrading, because the only handler around the loop is
`except subprocess.TimeoutExpired` (line 249) — a `UnicodeDecodeError` propagates out and no
`build_record.json` is written at all. A single non-UTF-8 byte from `pdflatex` (e.g. a Latin-1
byte echoed back from a `.bib` or a font path) takes down the build gate.

**Fix — minimal, `errors="replace"` added on the two `subprocess.run` calls:**

- **line 239-241** — the main latexmk/pdflatex pass
- **line 245-247** — the bibtex pass

```python
p = subprocess.run(cmd, cwd=paper_dir, env=texenv(),
                   capture_output=True, text=True,
                   errors="replace", timeout=900)
```

Verified after the fix: `ast.parse` clean, and the script ran end-to-end on paperC producing
`build_gate_pass: true`, `pdf_pages: 22`, `build_gate_blockers: []`.

### Separate, NOT fixed (out of scope, flagged for the next agent)

`paper_build.py` emits a hardcoded `pdf_visual_inspection_note` claiming "No
pdftoppm/gs/mutool/PyMuPDF on this host (measured 2026-08-14)". **PyMuPDF 1.28.2 and pypdf 6.16.1
are both importable from the interpreter that runs the script.** The `pdftoppm`/`gs`/`mutool`
half is correct (all absent from `$PATH` and from the in-repo TeX Live bin dir). The note is
therefore partly false and is why this repo twice concluded page counts could only be read from
`main.log`. Rendering page images *is* possible via `page.get_pixmap()`, so the skill's "PDF
visually inspected" gate is mechanizable on this host.

## 6. Residual overflow, and the next levers ranked BY MEASUREMENT

**3 integer pages still to cut.** Every row below was produced by copying paperC to `/tmp/pcprobe`,
applying exactly one change, running the full `latexmk -gg` build, and reading the References
boundary out of the resulting PDF. **These are measurements of page cost, not recommendations** —
several would destroy content and are listed so the cost of each is known. The probe reproduces
the on-disk state exactly (`main_pages=12, extent=11.301`), so the deltas are trustworthy.

### 6a. Remaining main-text tables (content-preserving; relocation only)

| lever | main pages | extent | Δ extent | reaches 9? |
|---|---|---|---|---|
| *(current state)* | 12 | 11.301 | — | no |
| move `tab:nulls` | **11** | 10.816 | −0.485 | no |
| move `tab:conventions` | 12 | 11.057 | −0.244 | no |
| move `tab:two-nulls` | 12 | 11.057 | −0.244 | no |
| move `tab:power` | 12 | 11.057 | −0.244 | no |
| `tab:nulls` + `tab:conventions` | **11** | 10.492 | −0.809 | no |
| `tab:nulls` + `tab:power` | **11** | 10.581 | −0.720 | no |
| **all 4 remaining tables** | **11** | 10.057 | −1.244 | **no** |

**Decisive negative result: relocating every remaining table still leaves 11 main-text pages.**
Table relocation alone cannot reach 9. `tab:nulls` is the single best remaining relocation
(−0.485 extent, and it is the one that flips 12 → 11); it is also the paper's central
construct-floor table, so moving it has real cost to the reader.

Note the three tables that each show −0.244 with no integer change: those are float-packing
artifacts, not three independent quarter-pages. Do not add them up — the combination rows above
are the measured truth.

### 6b. Prose bulk, for scale only (each row DELETES a whole section — do not do this)

Reported so the next agent knows where the mass actually is, and can judge how much *condensing*
(not deleting) each section would need to contribute.

| section removed entirely | main pages | extent | Δ extent |
|---|---|---|---|
| `05_analysis` | 10 | 9.231 | −2.070 |
| `03b_nulls` | 10 | 9.644 | −1.657 |
| `02_related` | 11 | 10.403 | −0.898 |
| `07_limitations` | 11 | 10.726 | −0.575 |
| `06_discussion` | 11 | 10.652 | −0.649 |

Reading: **not even deleting the entire Results and Analysis section reaches 9 pages on its own**
(9.231 extent, 10 pages). Getting to 9 requires roughly 2.3 pages of extent removed, i.e. the
combined effect of relocating `tab:nulls` (−0.485) *plus* condensing on the order of 1.8 pages of
prose. That is a substantive editorial decision about what the paper claims, which is why it was
not attempted here.

### Recommended next step (not executed — needs a judgement call, not a measurement)

1. Relocate `tab:nulls` to the appendix: **measured −0.485 extent and 12 → 11 pages**, zero data
   loss, one `\Cref` retarget. Cheapest remaining integer page.
2. Then condense prose ~1.8 pages of extent. The measured mass is in `05_analysis` (−2.070 for
   the whole section) and `03b_nulls` (−1.657). Candidate: `05_analysis` currently spends a long
   paragraph (line 24) on multiplicity caveats and another (line 30) on off-MMLU counts; both
   restate material the appendix tables carry. **Any such cut must be re-measured, not estimated.**
3. Free wins already flagged in `gate/venue_page_limit.json` and already structurally in place:
   Ethics and Reproducibility statements are emitted *after* `\bibliography` (main.tex:33-38), so
   they correctly do not consume main-text pages. Verified: Ethics p13, Reproducibility p13,
   both past the References boundary on p12.

## 7. Honest gaps

- **Not verified: that a program-committee page-counter agrees the main text is "12 pages".**
  References begins 35.4% down page 12, so pages 1-11 are pure main text and page 12 is shared
  between the tail of Limitations and the start of the bibliography. Under the strictest reading
  the count is 12; under a generous one an author might argue 11. Either way it exceeds 9, so the
  ambiguity does not change the required action. This is the same ambiguity flagged as
  "AMBIGUITY 3" in `gate/venue_page_limit.json`.
- **Not verified: rendered visual correctness of the moved tables.** `n_overfull_hbox = 0` and
  `n_overfull_vbox = 0` from the log, plus the captions were located on pages 14 and 15 by text
  extraction. Page *images* were not compared before/after, though §5 establishes PyMuPDF could
  do it.
- **zwfy6 disk: paperC does not exist there.** Checked, not assumed:
  `ls /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/paperC`
  → `No such file or directory`. So paperC is wzc1-only and there is no second copy to keep in
  sync. All work was on wzc1 (LOCAL).
- **GPU: zero used.** All work was pdflatex + stdlib/PyMuPDF python on the login host.
