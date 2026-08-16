# TABLE4_PROVENANCE_20260817.md — evidence paths for the three re-measured Table 4 rows

**Verdict: provenance ESTABLISHED.** All 33 cells of the keep8 / keep10 / keep12 rows of
`paperB/sections/tab_downstream.tex` (Table~\ref{tab:downstream}) reproduce from files on disk.
No number in `tab_downstream.tex` is changed by this audit.

Written 2026-08-17 by MAIN. 0 GPU used (JSON reads on CPU only). This file exists because
commit `6d15049` ("#192 option A+") rewrote those three rows to a single-pinned-protocol
re-measurement but **named no evidence path**, and the re-measured summaries are not under
`outputs/` or `evals/` — so a reviewer following the paper could not reach `.694`.

---

## 1. Why the value looked missing

`grep -rl '0\.6936' --include='*.json' outputs evals` returns nothing on **either** disk.
That is a **search-scope artifact, not a missing file**. The evaluation harness writes to
repo-root sibling trees, not into `outputs/`:

```
olmo2_downstream_results/     olmo2_ppl_results/       olmo2_closedbook_results/
olmo2_mmlu_content_results/   olmo2_mc_letter_content_results/   results/
```

`evals/` **does not exist** in this repo at all. The lead that resolved it is
`paperB/TODOList.md` L292, which cites `.82:olmo2_downstream_results/...` for ShortGPT.

## 2. The three rows draw on TWO directory families

Per the caption, the MMLU column and the ten non-MMLU columns have different sources.
Both were verified independently.

| Table 4 columns | source tree | disk |
|---|---|---|
| HS, ARC-C, ARC-E, PIQA, WinoG, OBQA, LAMB., BoolQ, CSQA, SIQA (10) | `olmo2_downstream_results/7B_keep{8,10,12}_step{121000,83500,124000}_v2{,_know}/` | **zwfy6 ONLY** |
| MMLU (1) | `olmo2_mmlu_content_results/7B_keep{8,10,12}_step{121000,83500,124000}/` | **both disks, byte-identical** |

### 2a. Ten non-MMLU columns — zwfy6 only

Absolute paths, disk **zwfy6** (`.73` / `.82` / `.104`), root
`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`:

| Table 4 row | core-6 summary | know-5 summary |
|---|---|---|
| keep8 (10L) @121k | `olmo2_downstream_results/7B_keep8_step121000_v2/summary.json` | `..._v2_know/summary.json` |
| keep10 (12L) @83.5k | `olmo2_downstream_results/7B_keep10_step83500_v2/summary.json` | `..._v2_know/summary.json` |
| keep12 (14L) @124k | `olmo2_downstream_results/7B_keep12_step124000_v2/summary.json` | `..._v2_know/summary.json` |

`sha256sum` (zwfy6, 2026-08-17):

```
369520409ee291e51d7b8f15cde2b4499c7ff2e8c5cec68b51568da98367aa9a  7B_keep8_step121000_v2/summary.json
20ddca7f349313c23eba3a3eca92670de8dba9366e82df0d33d57e21772aba33  7B_keep8_step121000_v2_know/summary.json
01d3a8699294ea93ec1b33e1e1fb2301ddcca0f720dadfe4eefa293a8a2885df  7B_keep10_step83500_v2/summary.json
ac948ac7851647c132bfe0538b1990cfcb05c4315d454d3a4415f2038ba48aaa  7B_keep10_step83500_v2_know/summary.json
4ee33849c6e61c9ca9af404e52e7b243dc5ec8888948281177d4b0964fd03f35  7B_keep12_step124000_v2/summary.json
f3911efdd8c162d979d54ae09956211b10ce2fa659f8b01b453ee60bda310e68  7B_keep12_step124000_v2_know/summary.json
```

Each dir also holds `shard{0..7}of8.json` and merged + per-shard `per_example_<task>.jsonl`.
`meta.ckpt` fields point at `outputs/olmo2_probe2_7B_keep{8,10,12}fresh2/step{121000,83500,124000}.pt`,
matching the row's step column. Written 2026-08-08 03:06–03:10 (shards) / 13:52 (merges).

> ⚠️ **DISK-LOCAL, NOT PORTABLE.** `find . -maxdepth 6 -type d -name '*keep{8,10,12}_step*_v2*'`
> on **wzc1** returns EMPTY. Only `7B_full32_base_wzc1_v2` and `7B_keep14_step200000_wzc1_v2`
> exist there. Per the distinction in `paperB/data/README.md`, these six files are
> **disk-local artifacts on zwfy6**, not part of the portable bundle. See §5.

### 2b. MMLU column — both disks, byte-identical

Relative to **either** repo root (verified identical on wzc1 and zwfy6):

| Table 4 row | path | sha256 |
|---|---|---|
| keep8 | `olmo2_mmlu_content_results/7B_keep8_step121000/summary.json` | `97cf8f5f5c68c0323344f82140fdfcca2b812b120b0d01df0643c31e0df07a7f` |
| keep10 | `olmo2_mmlu_content_results/7B_keep10_step83500/summary.json` | `fabf8705dcc4a47993dcc94b20ccc7dcb238c56ff6dc09c866aa109c3872f4ee` |
| keep12 | `olmo2_mmlu_content_results/7B_keep12_step124000/summary.json` | `2e3ed46807d7629b3389830ce30b0ead92df5565c04a3ed1b8e35487ff889ebd` |

Metric is the **`letter_acc`** field (`protocol = "letter+content(dual)"`). The same files also
carry `content_raw_acc` (.3222/.3232/.3407) and `content_norm_acc` (.3427/.3448/.3624), which are
much larger and must **not** be read as the Table 4 MMLU column.

> ⚠️ **Four decoy directories exist and look plausible. Do not use them.**
>
> | decoy | letter_acc | why wrong |
> |---|---|---|
> | `7B_keep8_step121000_wzc1` | .2546 | different re-measurement |
> | `7B_keep10_step83500_wzc1` | .2717 | different re-measurement |
> | `7B_keep12_step111500_wzc1` | .2713 | **also wrong step** (111500, not 124000) |
> | zwfy6 `7B_keep{8,10,12}_..._v2` (mmlu tree) | .2543 / .2707 / .2724 | the `_v2` MMLU tree is NOT the Table 4 basis |

## 3. Verification performed

Three independent recomputations. Every one captured its own exit code as
`cmd > file; echo $?` (never through a pipe).

**(a) Against `summary.json`, all 33 cells — `PY_RC=0`.** Per-cell asserts
`n_scored == n == expected`, `n_nan == 0`, `n_shards == 8`, `add_bos == false`,
`meta.ckpt_step ==` the row's step. Expected counts, all matched:

```
HS 10042  ARC-C 1172  ARC-E 2376  PIQA 1838  WinoG 1267  OBQA 500
MMLU 14042  LAMBADA 5153  BoolQ 3270  CSQA 1221  SIQA 1954
```

**(b) From the raw shards, bypassing `summary.json` — `PY_RC=0`.** This is the check that
matters, because `summary.json`'s `n_shards` counter is precisely what hid the old 6/8 merge.
Recomputed each cell as `sum(n_correct_{acc|accnorm}) / sum(n)` over `shard{0..7}of8.json`,
asserting the shard_index set equals `{0..7}`. All 30 non-MMLU cells reproduce.
The headline cell: **keep12 ARC-Easy = 1648 / 2376 = 0.693603 → `.694`**, from 8/8 shards
(297 items each). Also cross-checked `per_example_*.jsonl` line counts sum to `n_scored`.

**(c) MMLU from the 14,042 per-item records — `PY_RC=0`.** The `mmlu_content` dirs retain only
the merged `per_example_mmlu.jsonl` (shard files cleaned up), so recomputed as
`count(letter.correct) / count(lines)`:

| row | letter correct / n | value | printed |
|---|---|---|---|
| keep8 | 3581 / 14042 | 0.2550206523 | `.2550` |
| keep10 | 3820 / 14042 | 0.2720410198 | `.2720` |
| keep12 | 3830 / 14042 | 0.2727531691 | `.2728` |

`n_items == unique item_id == 14042`, zero `nan`, for all three.

**Cells that failed to reproduce: NONE (0 of 33).**

### 3a. The caption's single-source MMLU claim also holds literally

The caption asserts "all endpoint MMLU comparisons therefore share one set of 14,042 per-item
records, so the MMLU column is single-source across all nine rows." Tested by recomputing all
**nine** rows and hashing each row's `(item_id, subject, gold, n_opt)` stream: **one** distinct
signature `17cf44e95acde711` across all nine. 9/9 cells reproduce
(`.6054 .5877 .2550 .2720 .2728 .3184 .4742 .2624 .2470`), zero nan — `PY_RC=0`.
Dirs, all wzc1 `olmo2_mmlu_content_results/`: `7B_base`, `7B_full32_step25000`,
`7B_keep8_step121000`, `7B_keep10_step83500`, `7B_keep12_step124000`, `7B_keep14_step200000`,
`7B_shortgpt16_step200000`, `7B_freezefront_step200000`, `7B_scratch16L_step200000`.

Benign, pre-existing, and not a defect in any cell: BoolQ carries `n_trunc = 2` in all three
arms (items scored anyway, `n_nan = 0`).

## 4. Confirmation of the defect `6d15049` repaired

The old, superseded keep12 file is still on wzc1 inside the portable bundle:
`paperB/data/raw/olmo2_downstream_results/7B_keep12_step124000/summary.json`.

```
arc_easy: acc_norm 0.689113,  n = n_scored = 1782  (= 6 x 297),  n_nan = 0,  n_shards = 8
```

`n_shards` reads **8** while only six shards' worth of items are present — this is exactly the
failure mode the commit message describes, and it is verifiable in the file. The repair is real:
`.689` @ 1782 → `.694` @ 2376.

## 5. Reproducibility caveats a reviewer should be told

1. **The ten non-MMLU columns of these three rows are zwfy6-only** (§2a). They are not on wzc1
   and not in `paperB/data/raw/`. Under `paperB/data/README.md`'s own distinction they are
   disk-local, not portable.
2. **The portable bundle is now behind Table 4 for these three rows.**
   - `paperB/data/raw/olmo2_downstream_results/7B_keep8_step121000{,_know}/` **does not exist** —
     the bundle has keep8 only at 10k / 25k / 44k, so the keep8 row has **no** bundle counterpart.
   - The bundle's `7B_keep12_step124000/` is the **defective 6/8** file (§4).
   - Bundled `7B_keep10_step83500/` is a *different* (pre-`_v2`) measurement of the same
     checkpoint: max |delta| **0.40 pp**, e.g. OBQA .356 vs .352, ARC-E .6444 vs .6481.
     For keep12 the max |delta| is **0.49 pp** (CSQA .4963 vs .4914).
     Both are inside the caption's disclosed ≤0.5 pt same-architecture cross-stack bound.
3. **`paperB/scripts/build_appendix_artifacts.py` cannot validate these rows.** It resolves
   everything through `RAW = ROOT/"data"/"raw"` and its `close()` assertions still target the
   older rungs (`7B_keep12_step111500`, `7B_keep10_step10000`, `7B_keep8_step44000`). It does
   **not** read any `_v2` path, so a green run of that script is not evidence for Table 4's
   shallow rows.
4. `paperB/data/raw/` is **gitignored** (`.gitignore:61 data/`), so materializing the six `_v2`
   summaries into the bundle would not itself make them reviewable through git. Doing so is the
   recommended next step but is a deliberate artifact decision, left to the author.
5. `PAPER_B_DATA.md` §3(b) and §8.1–8.2 — the sections `tab_downstream.tex` cites in its header
   comment — are still the **2026-07-27 snapshot** (keep12@111500, keep8@44000, keep10@10000).
   They were not updated by `6d15049` and do not describe the rows now printed. This file is the
   current record for those three rows.

## 6. What was searched vs. what was inferred

**Searched (commands run, both disks):**

- `git show 6d15049` / `--stat` — touches only `paperB/*.md` + `paperB/sections/*.tex`.
- wzc1: `ls -d olmo2_*results/*_v2*`, `ls -d olmo2_*results/*keep{8,10,12}*`,
  `find . -maxdepth 6 -type d -name '*keep{8,10,12}_step*_v2*'` (empty), `ls paperB/data/raw/**`.
- zwfy6 via `.73`: `ls -d olmo2_*results/*keep{8,10,12}*_v2*` (found), `ls -la` of the six dirs,
  `sha256sum`, and three Python recomputations under `/opt/conda/envs/torch-base/bin/python`.
- `grep -n` over `paperB/TODOList.md`, `paperB/PAPER_B_DATA.md`, `paperB/P0_7_AGGREGATE_AUDIT.md`,
  `paperB/data/README.md`, `paperB/scripts/build_appendix_artifacts.py`.
- `find /` was **NOT** run (CephFS traversal hangs); all searches stayed inside the two repo roots.

**Inferred, not directly proven:**

- That the harness code writing these dirs pinned `torch` 2.13 / eval batch 8 — the caption's
  stack claim. The summaries record `n_shards`, `add_bos`, `meta.*` but **no torch version or
  batch size field**, so that specific claim rests on `6d15049`'s commit message and
  `paperB/TODOList.md` L277/L283, not on the JSON. Everything numeric above is from the files.
- The two self-inflicted false alarms during this audit are recorded in
  `paperB/_prov_worklog_20260817.md` (a leading-zero string compare, and a probe reading a
  nonexistent per-shard key) so that log's raw first runs are not mistaken for real failures.

---

## 7. Incidental pre-existing defect found while rebuilding (NOT caused by this audit)

`latexmk -pdf -bibtex -norc -gg main.tex` exits **rc=0**, 20 pages, 0 overfull/underfull boxes,
but reports two unresolved references:

```
Reference `tab:app-protocol-controls' on page 14 undefined on input line 21
```

Cause: `paperB/sections/app_tab_protocol_controls.tex` defines
`\label{tab:app-protocol-controls}` at line 35, but that file is **not `\input` anywhere** —
`sections/08_appendix.tex` inputs `app_tab_metric_sensitivity` and others, never
`app_tab_protocol_controls`. The dangling `\ref` is in
`sections/app_tab_metric_sensitivity.tex`'s caption, which tells the reader the
character-normalized metric is "distinct from the token-normalized complete-option MMLU protocol
in Table~\ref{tab:app-protocol-controls}" — pointing at a table the PDF does not contain.

**This is pre-existing, not introduced here.** Confirmed by `git stash`-ing the only `.tex` edit
of this audit and rebuilding from the clean HEAD tree: the same two undefined references appear
(`HEAD_BUILD_RC=0`, same message, same line). Note that `6d15049` *edited*
`app_tab_protocol_controls.tex` (+64 lines, correcting the false uniform 200k labels), so that
commit's careful appendix fix currently has **no effect on the built PDF**. Flagged for the
author; not fixed here, because resolving it means either adding an `\input` (which changes
pagination and the appendix table numbering) or rewording a caption — both are content
decisions, not provenance.

## 8. Records updated by this audit

Numbers in `paperB/sections/tab_downstream.tex` are **unchanged** — `git diff --numstat` reports
`11 insertions, 0 deletions`, and filtering the diff for non-comment lines returns nothing.

| file | change |
|---|---|
| `paperB/TABLE4_PROVENANCE_20260817.md` | this file (new) |
| `paperB/sections/tab_downstream.tex` | header **comment only**: stale-source warning + evidence paths |
| `paperB/P0_7_AGGREGATE_AUDIT.md` | §5 Provenance: dated addendum, since its keep8/10/12 paths are no longer Table 4's source |
| `paperB/data/README.md` | bundle-coverage caveat (disk-local vs portable), per its own existing distinction |
| `paperB/scripts/build_appendix_artifacts.py` | docstring scope limit: it cannot validate these three rows |
| `paperB/_prov_worklog_20260817.md` | incremental search log, incl. two self-inflicted false alarms |

> Note on §8: `paperB/data/README.md` is **gitignored** (`.gitignore:61 data/`) and untracked,
> so its added caveat lives on the wzc1 disk only and is not in the commit. That is the same
> reason the six `_v2` summaries cannot simply be checked into the bundle (§5.4).
>
> Also on §8: `paperB/scripts/build_appendix_artifacts.py` was **not tracked in git** before this
> commit, even though `paperB/data/README.md` names it as the headline-cell validator and its
> sibling `generate_appendix_tables.py` *is* tracked. It is not gitignored — it was simply never
> added. It is committed here (160 lines, no credentials) so that the scope limit recorded in its
> docstring is itself reviewable.
