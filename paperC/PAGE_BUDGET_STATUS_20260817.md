# paperC page budget — re-measured 2026-08-17

**Main text ends on page 10. The ICLR 2026 limit is 9. Overflow = 0.48 pages of extent
(`extent` 9.48), i.e. under half a page, not the 3 pages the previous record states.**

`PAGE_LIMIT_ACTION_20260816.md` is **stale**: its headline says "main text is still 12
pages ... residual overflow = 3 pages". That was true when written; compression landed
since. It is left byte-unchanged as a dated record — this file supersedes its numbers.

## What was measured, three independent ways

All three agree, which is why I trust the number:

| method | result |
|---|---|
| LaTeX's own count (`main.log`) | `Output written on main.pdf (27 pages, 387071 bytes)` |
| `\label` probe inserted immediately before `\bibliography{refs}` (on a COPY of main.tex) | last countable main-text page = **10** |
| `code/measure_page_budget.py` baseline | `main_pages: 10, extent: 9.48` |
| PyMuPDF, `REFERENCES` heading located by reconstructed line text + font size | page **10** |

The boundary is correct per the venue rule: `gate/venue_page_limit.json` (VERIFIED, http 200)
says the limit covers main text only, and that Ethics + Reproducibility are excluded *provided
they do not sit inside the main text*. `main.tex` emits `\bibliography` first, then Ethics,
Reproducibility, then `\appendix` — so countable main text ends at `\bibliography`.

### Two false measurements I nearly reported

- `pdftotext` and `pdfinfo` are **not installed** on this host. My first per-page grep loop
  "found" References on page 7 — it was reading `/tmp/pg.txt` left behind by a session on
  **Aug 16 23:59**. The command had failed silently and `2>/dev/null` hid it. Stale scratch
  files in `/tmp` are a measurement hazard; the run that produced them is long gone.
- TeX Live is at `.texlive/2026/bin/x86_64-linux`, not `.texlive/bin/...`. `command -v
  pdflatex` returning empty does not mean LaTeX is absent (see `COMPILE_ENV_NOTE.md`).

## Three defects fixed in `code/measure_page_budget.py`

The tool exited **rc=1 after 4 of 15 variants** and could not have informed any decision:

1. **`SECTIONS` named `03b_nulls`**, which main.tex no longer inputs (it inputs
   `03b_nulls_summary`; the full body moved to the appendix when option C4 was applied).
   The sweep died there, so **C4–C16 were never evaluated**. Fixed to the real input name;
   the relocation path now distinguishes "already applied" (skip, keep going) from
   "genuine typo" (still a hard error).
2. **All four `MAIN_TABLES` were already in the appendix.** `tab_nulls` / `tab_conventions`
   live in `03b_nulls.tex`, `tab_two_nulls` / `tab_power` in `09a_relocated.tex`, and the
   appendix root `\input`s both. The mover's appendix test was `not
   f.name.startswith("09_appendix")` — one filename, not the `\input` graph — so it happily
   moved tables from one appendix location to another and reported **extent 9.48 for all
   five table variants, byte-identical to baseline**. That reads as "relocating tables buys
   nothing": a real-looking null about a lever that does not exist. Now computed
   transitively (`_appendix_file_set`, 6/6 controls), the variant refuses with a diagnosis,
   and `MAIN_TABLES` is empty because **there is no main-text table left to move**.
3. **The results file merged instead of replacing.** After the retired variants were
   removed, `state/page_budget_sweep.json` still carried their 9.48 rows and the summary
   line still said "15 variants" while 11 had run. Retired records are now pruned on a full
   sweep (never under `--only`, where unselected variants were simply not attempted), and
   the prune is reported: `pruned 5 retired: movetab_all4, movetab_tab_*`.

## Where the 0.48 pages can come from (measured, not predicted)

From the repaired sweep (`state/page_budget_sweep.json`, 10 variants, rc=0):

| variant | main_pages | extent | Δextent vs baseline |
|---|---|---|---|
| baseline | 10 | 9.480 | — |
| drop_05_analysis | 8 | 7.231 | −2.249 |
| drop_03_method | 8 | 7.644 | −1.836 |
| drop_01_introduction | 9 | 8.231 | −1.249 |
| drop_03b_nulls_summary | 10 | 9.126 | −0.354 |
| drop_02_related | 9 | 8.480 | −1.000 |
| drop_04_experiments | 9 | 8.581 | −0.899 |
| drop_00_abstract | 9 | 8.740 | −0.740 |
| drop_06_discussion | 9 | 8.795 | −0.685 |
| drop_07_limitations | 9 | 8.919 | −0.561 |

These are **whole-section deletions**, i.e. upper bounds on what trimming each section could
buy — not proposals. Needing only 0.48 means no section has to go: a ~0.5-page prose trim
spread over the two longest sections suffices. `drop_03b_nulls_summary` buying just 0.354
confirms the summary stub is already compact.

**Not done here:** choosing what prose to cut. Per
`memory/numeric-census-misses-scoping-sentences.md`, compression must be verified sentence
by sentence — scoping and epistemic-limitation sentences carry no digits and so survive a
numeric census while being exactly the content that must not be silently dropped.


---

## ★ SUPERSEDED 07:10 — the overflow is CLOSED (10 -> 9 pages)

The 0.48-page overflow described above was closed by commit `ac59854` (subagent) plus
`623fefd` (MAIN, build record). **Main text now ends on page 9; `extent` 8.996.** The numbers
above remain the correct record of the *pre-compression* state.

MAIN verified independently, from a clean `git archive` of the commit rather than the agent's
working tree:

| check | result |
|---|---|
| `\label` probe before `\bibliography` (the designated method) | `main.aux` -> **page 9** |
| `main.log` | 26 pages, `latexmk` rc=0, 0 errors, 0 undefined |
| PyMuPDF block scan above the REFERENCES heading on p10 | only line numbers + ICLR header; **no prose** |
| numeric literals, per-file sorted multiset | **481 across 9 files, identical** — not one digit moved |
| 5 must-survive hedges | byte-identical |
| "loophole-closing construction" sentence | present (the one a prior compression silently dropped) |
| `gate_count_claims` / `check_prose_vs_evidence` | 17/17 rc=0 / 93-93-0 rc=0 |

**Two things the agent got wrong, both caught by re-checking rather than relaying:**

1. It called `gate_build_record_matches_pdf` **pre-existing**. It is not: rc=0 on `ac59854^`,
   rc=2 on `ac59854`. Fixed in `623fefd`. Its sibling `gate2_crossfamily_nulls` *is* genuinely
   pre-existing (rc=2 on both; needs argparse positionals).
2. It declined to refresh the build record, citing the reviewed-snapshot-provenance rule. That
   rule protects `review/round_*/` snapshots; a *build* record must describe the artefact beside
   it, and the stale sha256 appears in no snapshot manifest at all.

**Method finding worth keeping** (the agent's, verified by its own build trajectory): word cuts
do not reduce `extent` — a paragraph only shrinks when a trim pushes its **last line** off.
Four builds registered exactly 0.000 change despite real cuts. Per-paragraph last-line *slack*
has to be measured from the PDF to aim trims; `paperC/code/_measure_now.py` is the reusable
probe. Trajectory: 9.480 -> 9.455 -> 9.231 -> (3 builds at 0.000) -> 9.126 -> 8.996.
