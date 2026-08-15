---
task: what paperC still needs before it can be submitted as a paper
date: 2026-08-14
compute: CPU + web only. **ZERO GPU.** `.104` (paperC training, pid 3343485), LOCAL and
         `.212` (SparseForge) were read-only-inspected via `nvidia-smi` and never touched.
verdict: **NOT SUBMITTABLE TODAY — but the blocker is the MANUSCRIPT, not the science.**
         The methodology contribution is evidence-complete and independently
         recomputed here from raw records. There is no `.tex`, no figure, and no
         target venue. Estimated 2-3 weeks of writing, 0 additional GPU-hours
         required for the headline claim.
---

# paperC — submission gap audit

## 0. The one-paragraph answer

paperC is **a well-evidenced result with no manuscript**. Its headline
methodology claim ("report every construct against its construct-appropriate
best-constant null, not against chance") is supported by evidence that I
**re-derived from the raw per-item records in this session** — 9/9 construct
nulls reproduce, 7 of them to the last printed digit from records on this disk.
Its novelty position survives a fresh literature pass (0 preempting works; 1
missing citation cluster found). What does not exist is any part of a paper:
no `.tex`, no `sections/*.tex`, no figure, no table file, no venue, no author
list, no page budget. `paperC/sections/` is an **empty directory** created
2026-08-11 and never filled.

The user's stated facts were checked and are **all correct**. One clarification
and one correction to the framing are in §4.

---

## 1. MANUSCRIPT GAPS

### 1.1 BLOCKING

| # | gap | evidence | fix |
|---|---|---|---|
| M1 | **No `.tex` file of any kind.** `ls paperC/*.tex` → `No such file`. `paperC/sections/` is empty (0 files, created 08-11 20:55). | verified | Write it. This is the single largest gap. |
| M2 | **No target venue, therefore no page budget and no style file.** paperA pins `colm2026_conference.sty`; paperB pins `acl.sty` + `\documentclass[11pt]{article}`. paperC pins nothing. | verified | Decide venue **first** — it determines whether the retraction ledger (§2.4) fits. |
| M3 | **No figures.** `find paperC -iname '*.pdf' -o -iname '*.png'` → **zero hits**, and there is no figure-generating script. Both sibling papers ship figures. | verified | The paper needs ≥2: (a) the wrong-null flip (arm accuracy vs chance line vs floor line, per construct); (b) the v1-vs-v2 re-sort of the 27 cells. Both are pure-CPU plots off existing JSON. |
| M4 | **No table `.tex` files.** paperA has 30+ `tab_*.tex`, paperB has 20+ `app_tab_*.tex`. paperC has **0**, though the *numbers* exist in `evidence/*.csv`. | verified | Mechanical: `evidence/mmlu_pro_power_nulls_v2.csv` (231 rows) and `second_mc_benchmark_crossfamily/gate2_crossfamily_nulls.csv` (1122 rows) → booktabs. Write a `csv2tex` step so the tables cannot drift from evidence. |
| M5 | **No abstract / intro / method / experiments / related / limitations.** All six are absent, i.e. 100% of the prose. | verified | See §1.3 for the reusable skeleton. |

### 1.2 应做 (should do)

| # | gap | note |
|---|---|---|
| M6 | **The README is a *ledger*, not a paper.** It is organised as claim → retraction → citation obligation. A paper needs the inverse order (contribution → method → evidence → limitation). The retraction ledger is genuinely part of the contribution (self-falsification narrative) but cannot be the body text. | Budget a full re-narration pass. Do **not** paste the README. |
| M7 | **The evidence lives in 3 places and only 1 is `paperC/`.** `GATE3_VERDICT.md` (the fp32 mechanism result, a headline), `NOVELTY_CHECK.md`, and `gate3_dtype_runs/` are **only** under `proposal/active/A01-.../`. paperC's README cites them by name as if local. | Fine for provenance (CLAUDE.md requires A01 stay authoritative), but the paper's own artefact list must name the real paths. Verified: A01's `GATE3_VERDICT.md` exists and its `7B_keep8_step121000_dtype_summary.json` contains the exact headline numbers (`letter_argmax_changed_rate = 0.18031619`, `letter_acc_delta = −0.0014955`, `letter_mcnemar_p = 0.5702`). |
| M8 | **No author list / affiliation / ack / ethics / repro statement.** | Copy the block from `paperA/main.tex:32-49` (Tsinghua + Tencent). |
| M9 | **No `BUILD.md`.** Both siblings have one. | Copy pattern. |

### 1.3 Can the sibling skeletons be reused? **YES — paperB's, not paperA's.**

Inspected both. **Reuse paperB.**

* `paperB/acl.sty` is present and `paperB/main.tex` is `\documentclass[11pt]{article}` + `acl.sty` — the ACL-family layout, which is the right family for a **measurement/methodology** paper. paperA is COLM-workshop-specific (`colm2026_conference.sty`).
* paperB's section list is already almost paperC's outline:
  `00_abstract / 01_introduction / 02_related / 03_method / 03_measurement / 04_experiments / 05_analysis / 04b_discussion / 05_conclusion / 06_limitations / 07_ethics / 08_appendix`.
  paperC maps onto it with **one insertion**: a `03b_nulls.tex` defining the four
  under-specifications (tie convention / length unit / tokenizer / `mean(1/n_opt)`),
  which is paperC's sharpest owned material.
* paperB also already ships `app_tab_integrity.tex` and `app_tab_crossfamily.tex` —
  paperC needs exactly those two shapes and can copy their column structure.
* ⚠️ paperB's `sections/` are on **the same disk** and paperB is a *different*
  paper about layer pruning. Copy the **skeleton**, never the prose — several
  paperC arms (`keep8/keep10/keep12`) are literally paperB's arms, so a copy-paste
  would silently double-report the same runs across two submissions.

**Concrete first command** (not run — writing is not in scope for this audit):
`cp paperB/acl.sty paperB/BUILD.md paperC/ && cp paperB/main.tex paperC/main.tex`
then strip and re-`\input`.

---

## 2. EVIDENCE GAPS

### 2.1 What the existing evidence DOES support (recomputed first-hand today)

Every one of these was recomputed **from raw per-item records**, not read off the README:

| README claim | supported? | how verified today |
|---|---|---|
| MMLU letter floor `always-D 0.2689` | ✅ | recomputed `0.2689075630` from `olmo2_mmlu_content_results/7B_base/per_example_mmlu.jsonl`, n=14042. Bit-matches A01's stored `0.2689075630252101`. |
| MMLU content longest-option `split 0.2845` (token unit) | ✅ | recomputed `0.284450` from `cont_tokens`, tie rate `34.2188%` — matches A01's printed 34.22%. |
| BoolQ `always-B 0.6217` (2033 B / 1237 A) | ✅ | recomputed `2033/3270 = 0.6217125382` from `olmo2_downstream_results/.../per_example_boolq.jsonl`. Counts exact. |
| 5 non-MMLU letter floors | ✅ | all 5 recomputed exactly (§2.3 table). |
| MMLU-Pro `always-A 0.116606` + `mean(1/n_opt) 0.110877` + `n_opt` histogram | ✅ | recomputed on `.73` from 8 shards, n=12032, 12032 unique ids, `n_opt = {3:21,4:606,5:52,6:93,7:158,8:320,9:801,10:9981}` — **matches the pre-registration exactly**. |
| fp32-vs-bf16 mechanism (gate-3) | ✅ | all headline numbers present in `7B_keep8_step121000_dtype_summary.json`. |
| the 27-cell v2 permutation read-out | ✅ | `evidence/heal_readout_v2_permutation_null.json` present, with `selftest_collapse_invariance` and `G4_null_admissible` recorded. |
| per-item records exist on both disks as claimed | ✅ | wzc1: `mc_lc_crossfamily_results` 130M/15 arms, `olmo2_mc_letter_content_results` 52M/6, `olmo2_mmlu_content_results` 1.1G. zwfy6 (`.73`): `mmlu_pro_lc_crossfamily_results_fix` 310M/15, `mmlu_pro_letter_content_results` 122M/6, `mmlu_pro_lc_paperC_heal_results` 122M/6, `results/a01_gate3/dtype_runs` 52M/6, pinned ckpts 284G/9. **Nothing claimed missing is missing.** |

### 2.2 BLOCKING evidence gaps

| # | gap | severity |
|---|---|---|
| **E1** | ✅ **CLOSED 2026-08-14 by MAIN, 0 GPU.** `paperC/code/construct_nulls_length_unit.py` → `paperC/evidence/construct_nulls_length_unit.json`. The blocker below ("no python on any of the 5 nodes has pyarrow/datasets") was true when written and is now **false**: `pighzliu_code/venv_union9/bin/python` — built 2026-08-14 on wzc1 for the union-9 pinned harness — carries **pyarrow 25.0.1 + datasets 5.0.1**. The char unit is recomputed from the raw parquet under the *published* definition, `len(option_text)` per `scripts/eval_olmo2_mc_letter_content.py:331` with the continuation's leading space excluded (`char_with_space` is emitted alongside so the convention is visible rather than assumed; it is identical at `0.363500`, a uniform shift being unable to change which option is longest). **The self-test is 12/12, not 1/1**: all **six** character-unit values *and* all **six** token-unit values recorded in A01 `STATUS.json:NEW_degree_of_freedom_length_unit` reproduce exactly — obqa `0.363500`/`0.368000`, arc_ch `0.274104`/`0.283902`, arc_easy `0.255296`/`0.238054`, csqa `0.221977`/`0.201775`, piqa `0.475245`/`0.465452`, winogrande `0.491713`/`0.501184` — and the script **raises rather than writing** on any mismatch. The token leg is recomputed from `cont_tokens` in the per-item records, which is what **proves both units describe the same items**; prose could assert that but not establish it. ⚠️ **New finding the paper should carry**: the `credit` convention is drastically more unit-sensitive than `split`. OBQA `credit` is `0.416` (char) vs `0.644` (token) — a **22.8 pp swing on identical items** — against `0.45 pp` for `split`. So "print the convention AND the unit" is not pedantry: under `credit`, the length unit alone moves the floor by more than most reported arm effects. <br><br>*Original finding, preserved for provenance:* **The OBQA character-unit null `0.3635` is the ONE number in the README's headline table that is NOT recomputable from anything on disk.** `paperC/code/gate2_crossfamily_nulls.py:33-36` says so in its own docstring: *"The per-item records store `cont_tokens` only … The character unit is NOT recoverable from these records and is therefore not reported here rather than being guessed."* I confirmed: no record carries option **text**; `grep -rn '0.3635'` over all `*.py`/`*.json`/`*.csv`/`*.log` finds it **only in prose** (README, A01 `STATUS.json`, `PROPOSAL.md`) — never in a machine-readable evidence file and never in code output. The raw OBQA parquet **is** on disk (`.hf_cache/hub/datasets--allenai--openbookqa/.../test-00000-of-00001.parquet`) so it is *recomputable in principle*, but **no python on any of the 5 nodes has `pyarrow` or `datasets`** (checked conda, `.venv`, `python3.11`, `python3`), so I could not close it. **This is a 30-minute, 0-GPU fix that must happen before submission**: install pyarrow (or parse the arrow file), recompute, and emit it into an evidence JSON like every other null. |
| **E2** | ✅ **CLOSED 2026-08-15 by MAIN, 0 GPU — and its premise was FALSE.** `evidence/floor_winners_curse_calibration.json` **already is** the single machine-readable source: it carries exactly the **9 rows** of the README headline table with `construct`/`n`/`k`/`floor`/`chance` at full precision, plus `seed=20260814`, `n_draws=200000`, `method`, `schema_version=1.0.0`. MAIN self-test: **9/9 rows reproduce** the README values (checked at 3/4/6-dp and full precision), so the table is derivable from this file alone. Assertion emitted to `evidence/e2_single_source_assertion.json`. ⚠️ **The two numbers this row called homeless are not homeless**: `0.2689` is `rows[2].floor = 0.268908` and `0.6217` is `rows[8].floor = 0.621713`. ⚠️ **This is the SECOND instance of the same error in this very row** — E2 already carried a self-correction saying a naive grep for the README's *rounded* form misses values stored at full precision (that correction rescued `0.276`/`0.504897`). The identical reasoning was simply never re-applied to the remaining two. Per `memory/fix-the-class-not-the-instance.md`, fixing one instance of a lookup bug obliges re-running the **whole key family**. <br><br>*Original text, preserved for provenance:* **No single machine-readable source for the README's 9-row headline table.** The 9 values are *scattered* across `second_mc_benchmark/*.json` (9 rows carry `0.276`), `second_mc_benchmark_crossfamily/*.json` (17 rows carry `0.5048966268`), `mmlu_scale_power/*.csv`, and — for MMLU/BoolQ/OBQA-char — **prose only** in `A01/STATUS.json` and `A01/PROPOSAL.md`. So the README table is hand-assembled from four sources plus prose, and 2 of its 10 numbers (`0.2689` MMLU, `0.6217` BoolQ) have no paperC-side machine-readable home at all. Emit one `evidence/construct_nulls_table.json` that is the single source the paper's table is generated from. ⚠️ **Correction to my own first pass:** I initially recorded `0.276000` and `0.504897` as *absent* from the evidence JSONs. That was wrong — a naive `grep` misses them because they are stored at **full precision** (`0.5048966267682263`), not at the README's rounded form. Both **are** present. The gap is fragmentation, not absence. |
| **E3** | ✅ **CLOSED 2026-08-15, 0 GPU.** The binding now exists in both directions. **(a) Generated:** `paperC/code/emit_tab_construct_nulls.py` → `paperC/sections/tab_construct_nulls.tex` (26 lines, `\input` at `sections/09_appendix.tex` under a new subsection "Generated construct-null manifest", `\label{tab:app-construct-nulls}`). Every cell is a dict lookup into `evidence/floor_winners_curse_calibration.json`; the caption carries `schema_version`/`seed`/`n_draws`/`method`/`sha256` from that same file. **AST-verified: 0 hardcoded floor or chance literals** — all 17 float constants in the emitter are tolerances or arithmetic factors (`100.0`, `0.5`, `1e-9`). It **raises rather than writing** (E1's pattern) on 8 checks: T1 provenance fields, T2 row identity vs `e2_single_source_assertion.json` (a *two-file* agreement, not a hardcoded expectation), T3 `gap_pp == round(100*(floor-chance),3)`, T4 `n*floor` is an integer count, T5 `floor >= 1/k`, T6 `survives` ↔ `p_one_sided` ↔ the file's own survives/inside_noise partition, **T7 the already-shipped hand-typed `sections/tab_nulls.tex` round-trips (9/9 rows × 8 columns)** — this is the concrete risk this row names — and T8 its own output round-trips. Output is byte-deterministic across runs. **(b) Auditing:** `paperC/code/check_prose_vs_evidence.py` → `paperC/evidence/prose_vs_evidence_check.json`. Scans `README.md` + all 22 `sections/*.tex` against 23 targets from 5 evidence files; **measured `n_checked=81 / n_ok=81 / n_mismatch=0 / n_uncovered=0`, rc=0** (also rc=0 under `--strict`). Comparison is **multi-precision half-ulp, never string equality** — directly inheriting E2's documented error class (a naive grep for the README's *rounded* form misses full-precision storage: `0.2689`=`0.268908`, `0.6217`=`0.621713`). Two tiers, both needed and both measured: `near_miss` (63 ok) catches single-digit typos; `declaration` (18 ok) matches the *syntax* of an assertion (``always-D =``, ``floor is``) and so fails at **any** magnitude — verified necessary, because mistyping BoolQ `0.6217`→`0.6127` escapes tier 1 entirely. **Mutation-tested, not merely run:** 6 injected typos (large, small, last-digit, and a PIQA↔Winogrande value swap) all produce rc=1 with the correct file:line; 6 injected evidence/`.tex` inconsistencies all make the emitter refuse to write. **LaTeX: 0 errors, 0 undefined references, 0 undefined citations, 0 overfull boxes, `build_gate_pass: true`, 17→19 pages** — the 0-error/0-undefined baseline is preserved, verified both from `gate/build_record.json` and by grepping `main.log` directly. Toolchain: `.texlive/2026/bin/x86_64-linux/latexmk` (present on disk, not on `$PATH`). ⚠️ **New provenance finding, reported not repaired:** the **MMLU *content* null `0.2845`** (quoted `README.md:23`) has **no paperC-side machine-readable home** — a walk of every float in every `paperC/evidence/**/*.json` finds nothing within 5e-6 of it (16 files at the time of the check; the only present-day hit is the copy this task's own `prose_vs_evidence_check.json` now records). Its only machine-readable home is `proposal/active/A01-null-calibration-methodology/evidence/gate3_content_null_conventions.json` (`0.28445022076627263`, invariant across all 6 arms). The **value is correct**, so this is fragmentation of the same class E2 described, for a value E2's 9-row table does not cover; it is recorded under `provenance_gaps` in the emitted JSON and **E2 is not reopened**. Relocating evidence into `paperC/evidence/` is an editorial call, not an agent's. <br><br>*Original text, preserved for provenance:* **No `.tex`-level integrity binding between evidence and prose.** paperB solved this with `app_tab_integrity.tex`. paperC's numbers currently live in prose in a 478-line README with 30+ inline retractions. Given this direction has already shipped 10 truncated cells with summaries written (README:401-420), a hand-typed table is a real risk. |

### 2.3 The 9 construct nulls — provenance, one by one

Requested explicitly. `recomputed today` = I ran the arithmetic in this session
from raw per-item records or raw dataset, not from any summary.

| # | construct | README value | recomputed today | verdict |
|---|---|---|---|---|
| 1 | MMLU letter | `always-D 0.2689` | **`0.2689075630`** (n=14042, from `per_example_mmlu.jsonl`) | ✅ **recomputable**, and stored at full precision in A01 evidence (`0.2689075630252101`) |
| 2 | MMLU content (longest-opt, token) | `0.2845` | **`0.284450`** (ties 4805/14042 = 34.2188%) | ✅ **recomputable**; also in `gate3_content_null_conventions.csv` as `0.28445` |
| 3 | BoolQ | `always-B 0.6217` (2033/1237) | **`0.6217125382`** (2033 B, 1237 A, n=3270 — counts exact) | ✅ **recomputable** ⚠️ but see the trap below |
| 4 | OpenBookQA content | `0.3635` **char** / `0.3680` token | token: **`0.368000`** exactly (OLMo-2 tokenizer, ties 242/500). char: **`0.363500`** ✅ **RECOMPUTED 2026-08-14** (`evidence/construct_nulls_length_unit.json`, raw parquet, 12/12 self-test) | ✅ **both units recomputable — E1 CLOSED**. Note the token null is *tokenizer-valued*: I got **`0.375833`** on the Llama-2 records for the same 500 items, so `0.3680` must always be quoted as "OLMo-2 tokenizer". The char null, by contrast, is **tokenizer-free** — a pure dataset property — which is a reason to prefer it when comparing across families. |
| 5 | ARC-Challenge letter | `always-B 0.265358` | **`0.265358`** (n=1172, argmax B) | ✅ **recomputable** |
| 6 | ARC-Easy letter | `always-C 0.266414` | **`0.266414`** (n=2376, argmax C) | ✅ **recomputable** |
| 7 | OpenBookQA letter | `always-A 0.276000` | **`0.276000`** (n=500, gold A=138/500, argmax A) | ✅ **recomputable**; present in evidence JSON (9 rows) — see E2 correction |
| 8 | CommonsenseQA letter | `always-B 0.208845` | **`0.208845`** (n=1221, argmax B) | ✅ **recomputable** |
| 9 | PIQA letter | `always-B 0.504897` | **`0.504897`** (n=1838, argmax B) | ✅ **recomputable**; in evidence JSON at full precision `0.5048966267682263` |
| (+) | MMLU-Pro letter | `always-A 0.116606` | **`0.116606`** (n=12032, 12032 unique ids, argmax A) | ✅ **recomputable** on zwfy6 |
| (+) | Winogrande (neg. control) | `always-B 0.504341` | **`0.504341`** (n=1267, argmax B) | ✅ **recomputable** |

**Score at audit time: 10 of 11 fully recomputable; 1 (OBQA character unit) not, and its own
code said why.**

> ✅ **UPDATED 2026-08-14 — the score is now 11 of 11.** The character unit was closed
> the same day (`paperC/code/construct_nulls_length_unit.py` →
> `evidence/construct_nulls_length_unit.json`, 0 GPU). The blocker was never the data —
> the parquet was always on disk, as §2.2 E1 itself noted — it was the **interpreter
> census being incomplete**: the audit checked conda / `.venv` / `python3.11` / `python3`
> but not `pighzliu_code/venv_union9`, which had `pyarrow` + `datasets` the whole time.
> Generalisable lesson, same shape as the two-disk rule: **"no tool on any node has X"
> requires enumerating the environments, not just the nodes.** A project-disk venv built
> for one purpose is a general-purpose interpreter for every later purpose.

> ⚠️⚠️ **A trap I hit, recorded so nobody repeats it.** `grep -rn 0.6217` inside A01's
> evidence "finds" `0.6217930330316742` in `null_calibration_p1_nperm2000.json` and
> `null_calibration_obs4_nperm2000.json`. **That is NOT BoolQ.** It is
> `/c3_cka/per_pair[18]/null_mean` — the CKA permutation null for the pair
> `llama2_7b:openllama_3b`, a numeric coincidence agreeing with BoolQ's floor to 4
> decimals. BoolQ's real value is `2033/3270 = 0.62171254`; the CKA number is
> `0.62179303`. **Anyone citing "BoolQ 0.6217, see `null_calibration_*.json`" would be
> citing a representation-similarity null for a question-answering floor.** Verify by
> counting golds, not by grepping the value.
>
> Same shape of hazard in `A01/STATUS.json:75`'s character-unit list: I checked whether
> those values (`0.274104` / `0.255296` / `0.221977` / `0.475245` / `0.491713`) appear in
> the crossfamily CSV as `null`s — they do **not** (0 rows each), and `0.491713`
> appears **15 times as an `acc`** (llama3/k12/winogrande/content_raw). So those
> character-unit numbers are unbacked by any evidence file *and* one of them collides
> with an arm accuracy. Same fix as E1.

### 2.4 应做

| # | gap |
|---|---|
| E4 | **The retraction ledger is 30+ items and partly self-superseding.** README's scope-discipline list contains retractions-of-retractions (e.g. "significantly below floor is MMLU-specific" was retracted by #252, then **narrowed again** on 08-13 by v2). A reviewer reading this will not be able to tell what is currently claimed. Needs a single flat "what we claim / what we retract" table with one row per claim and a current-status column. |
| E5 | **`llama2_7b` fails its own G6 anchor** (`recovery_fraction = 0.0545` intact on MMLU-Pro). The paper's four-family framing is therefore **3.5 families** on the MMLU-Pro leg. Must be stated in the experiments section, not buried in a pre-registration §5c. |
| E6 | **paperC has no `NOVELTY_CHECK.md` of its own** and inherits A01's, which `VENUE_AND_NOVELTY_VERIFICATION.md:692` already flags as needing softening ("apparently unclaimed" → "not previously computed for an input-blind null"). That edit was written as replacement text but the softening is **not spliced into A01's `NOVELTY_CHECK.md` §3** — I checked. |
| E7 | **Cho et al. ICLR 2026 camera-ready was never diffed** (OpenReview `/pdf` bot-challenged from this network; read at arXiv v4, 14 days pre-camera-ready). Self-declared as an open gap in `VENUE_AND_NOVELTY_VERIFICATION.md:270-279`. Retry before submission. |

---

## 3. NOVELTY

### 3.1 Verdict: **the core claim survives. 0 preempting works. 1 missing citation cluster found (应做, not blocking).**

I ran a fresh arXiv pass (14 queries) independent of the existing
`VENUE_AND_NOVELTY_VERIFICATION.md`, targeting exactly the claim
"best-constant/input-blind null instead of chance, per construct, as a
precondition on arm comparison". **Nothing preempts it.** The existing 9-candidate
full-text audit is, as far as I can check, sound and unusually thorough — I did
not find a candidate it missed *on its own axis*.

Sources queried (all through `hy-proxy.woa.com:3128`):
* **arXiv API** (`export.arxiv.org/api/query`) — 14 distinct queries:
  `"majority class baseline"+"multiple choice"`, `"input-blind"+benchmark`,
  `"null model"+benchmark+LLM`, `"best constant predictor"`, `"trivial baseline"+LM`,
  `"construct validity"+"language model"`, `"chance level"+MCQA+LLM`,
  `"label bias"+MCQA`, `"label marginal"+benchmark`, `"permutation test"+MCQA+LLM`,
  `"selection bias"+"multiple choice"+LLM`, `"answer position"+bias+MC`,
  `"layer pruning"+MC+eval`, `ti:"null calibration"`.
* **DBLP** `search/publ/api` — venue authority for the ACL-family hit.
* **OpenReview API2** `notes/search` — venue authority for the ICLR-family hit.

### 3.2 The closest new hit, and why it does not preempt

**arXiv:2602.02182v2, "Evaluating Metalinguistic Knowledge in LLMs across the World's
Languages"** (2026-02-02, updated 2026-08-14). This is the *closest* thing I found to
paperC's headline flip, and it is worth reading before writing the intro. Its abstract
states the flip almost verbatim:

> "comparing to chance and majority-class baselines … **Although all models perform
> above chance, they fail to outperform the majority-class baseline**"

**Venue, checked at the correct authority:** **DBLP returns `CoRR 2026`, type
`Informal and Other Publications`** — i.e. **preprint, no venue**. arXiv carries **no
`journal_ref`, no `doi`, no `comment`**. So it is not a published preemption. It is
also 6 months old (2026-02), which under `memory/prior-work-differentiate-dont-abandon`
is **outside** the 2-3 month concurrency window, so it should be **cited**, not dismissed
as concurrent.

**Why it does not preempt, per paperC's own Q1/Q2/Q3 test:**
* **Q1 (input-blind null): YES** — it computes a majority-class baseline. Same as
  Balepur, which paperC already concedes.
* **Q2 (per-construct, as a precondition on arm comparison): NO** — it is a
  *descriptive finding on one benchmark it built itself* (WALS→MCQ). It reports
  "models are above chance but below majority" as a **result about model knowledge**,
  not as a **protocol** that disqualifies a number from entering a comparison. No gate,
  no floor test applied to other people's benchmarks, no per-arm validity certificate.
* **Q3 (letter-vs-content interface): NO.**
* No damage/pruning regime, no tie convention, no length unit, no tokenizer axis, no
  `mean(1/n_opt)` ambiguity, no permutation null.

**Its real value to paperC is as ammunition, not threat:** it is an *independent 2026
instance of exactly the failure paperC's protocol is designed to catch*, found in the
wild by authors who noticed it only because they happened to print both baselines. Cite
it in the intro as evidence the problem is live and recurring. That is a **stronger**
opening than a purely methodological argument.

### 3.3 ★ The one real gap: the selection-bias / token-bias literature is entirely uncited

`grep -rn "selection bias\|token bias\|PriDe\|2309.03882" paperC/ proposal/active/A01-*/`
returns **nothing**. That is a citation hole a reviewer in this area will find immediately,
because paperC's central mechanism — *a damaged model collapses onto one letter, and the
letter's marginal decides the verdict* — is the **measurement consequence** of the
phenomenon this literature named.

| paper | venue (verified at correct authority) | what it owns | why paperC is still distinct |
|---|---|---|---|
| **Zheng et al., "LLMs Are Not Robust Multiple Choice Selectors", arXiv:2309.03882** | **ICLR 2024 Spotlight** — OpenReview `venue = "ICLR 2024 spotlight"`, **`venueid = ICLR.cc/2024/Conference`** (the real note, not the `dblp.org/conf/ICLR/2024` mirror). arXiv `comment` also self-declares "ICLR 2024 Spotlight". | **"selection bias"** and **"token bias"**: LLMs "a priori assign more probabilistic mass to specific option ID tokens (e.g. A/B/C/D)". Debias method **PriDe**, which estimates the option-ID prior by permuting option contents. | They diagnose *why* a model prefers a letter and **fix the model's prediction**. paperC measures *what the benchmark's own gold-letter marginal does to the reference line* and **fixes the null**. Their prior is over **predictions**; paperC's floor is over **golds**. Crucially: PriDe would not change any paperC verdict, because paperC's floor is arm-independent by construction (`READOUT_V2 §3`). |
| **arXiv:2410.14248, "Addressing Blind Guessing"** (VLM MCQA, calibration method BOLD) | arXiv only — no `journal_ref`/`doi`/`comment` → **preprint** | selection-bias calibration in video-LM MCQA | different modality; a post-processing debias, not a null |
| **arXiv:2603.21016, PA-GRPO** | self-declares **"Accepted to ACL 2026 Main"** in arXiv `comment`. ⚠️ **NOT independently confirmable yet**: DBLP returns **0 hits**, and ACL Anthology has no `2026.acl-long` entry for it (proceedings not yet posted). Per `memory/venue-verify-acl-family-needs-anthology`, an arXiv self-declaration is **not** the authority — record as *claimed ACL 2026, unverified*. | trains selection-bias away via permutation-consistent RL | orthogonal: a training method, not a measurement protocol |

**Consequences for paperC, stated concretely:**
1. **Must cite Zheng et al. ICLR 2024** for "selection bias / token bias" and must not
   describe letter-preference as a new observation. paperC's README currently narrates
   letter collapse as if the phenomenon were its own to name.
2. **`READOUT_V2 §4d` needs a citation.** It rejects "balanced / permuted-option
   controls" as the fix on *cost* grounds ("needs new GPU"). That control is
   **PriDe's core mechanism** (estimate the prior by permuting option contents on a
   small subsample — explicitly designed to be cheap). The rejection is still
   defensible, but it must be argued **against the known cheap method**, not as if the
   option were merely expensive.
3. **This is a differentiation opportunity, not a kill** (per
   `memory/prior-work-differentiate-dont-abandon`). The follow-up paperC can claim:
   the selection-bias literature debiases the **model** and then reports accuracy
   against **chance** — so it fixes the predictor while leaving the reference line
   wrong. paperC's floor is the missing half. That is a clean, citable correction to a
   Spotlight paper's evaluation protocol, of the same shape as the two corrections
   `VENUE_AND_NOVELTY_VERIFICATION.md` already licenses against Balepur and OLMES.

### 3.4 应做

| # | gap |
|---|---|
| N1 | Cite **Zheng et al. ICLR 2024 Spotlight** (selection bias / token bias / PriDe). Re-argue `READOUT_V2 §4d` against PriDe specifically. |
| N2 | Cite **arXiv:2602.02182** as an in-the-wild instance of the flip. Mark as preprint (DBLP: `CoRR 2026`, informal). |
| N3 | Retry the **Cho et al. ICLR 2026 camera-ready** diff (E7). |
| N4 | Splice the already-written softening of `NOVELTY_CHECK.md §3` (E6). |
| N5 | Note that **two Zhengs** are now cited (Zheng et al. ICLR 2025 Oral "null model" ≠ Zheng et al. ICLR 2024 Spotlight "selection bias"). Disambiguate in the bib or a reviewer will read it as a duplicate. |

---

## 4. THE STATUS OF step121000 — the direct answer

### Verdict: **step121000 is a SIDE BRANCH. The main conclusion does not need it. And its own pre-registered question has already dissolved.**

This is not my inference. It is **stated in paperC's own documents**, in four places:

1. `HEAL_CONFOUND_PREREGISTRATION.md §9.4`, written 08-12 **before any GPU**:
   > "**It says nothing about the null-calibration methodology claim, which is
   > paperC's actual contribution and does not depend on this at all.** This closes a
   > *scope* defect in one empirical leg."
2. `README.md:457-459` repeats it verbatim.
3. `READOUT_V2_PREREGISTRATION.md §9.5`:
   > "**v2 says nothing about instrument validity.** paperC's headline claim
   > ('report against the best-constant null, not chance') is a v1 statement and is
   > untouched."
4. `READOUT_V2_PREREGISTRATION.md §8` — and this is the sharper point:
   **the arm's pre-registered question is no longer askable.** §8's `H_heal` requires
   the un-healed twin `qwen3_8b_base/k8` to sit *below* its floor. Under the
   collapse-proof permutation null it does **not** (−0.139 pp, p = 0.0964, n.s.).
   > "**The antecedent of the criterion is false.** So `H_heal` as literally
   > pre-registered is not merely unmeasured — its contrast has dissolved, and **no
   > amount of further training makes it measurable, because the failure is in the
   > *comparator*, not the arm.**"
   `H_family` is equally unavailable: it needs the healed arm to be significantly
   below its permutation null, and **0 of 27 cells** are.

**So what does step121000 still buy?** Exactly one narrowed, well-posed thing, per
v2 §8: *does a healed front-8 Qwen3 at 121k show material `ITEM_LEVEL_SIGNAL` on
MMLU-Pro, or stay at `NO_ITEM_LEVEL_SIGNAL` like every damaged cell so far?* It is a
genuine question and worth reading out — but it is **one appendix cell**, it resolves a
*scope caveat in one empirical leg*, and the existing data already predicts the answer:
OLMo-2's own `keep8` is `NO_SIGNAL` at both **45000** (+0.322 pp, p=0.086) and
**121000** (+0.229 pp, p=0.312), i.e. **OLMo-2 keep8 never acquires material signal at
any heal budget measured**, and the Qwen3 trajectory is **flat** across 5000→7000
(−0.100 … +0.004 pp, all p > 0.49).

**Live status (read-only, 0 GPU, verified today):** `.104` step **29700**/200000,
loss 2.74, ppl 15.4, 8/8 GPUs 100%, maxmem 77.5 GB, healthy, no kill condition met.
Realised rate from log timestamps (`elapsed/iter`, per
`memory/one-sample-is-not-a-trend-or-state` — **not** the instantaneous `5.74 s/step`
field): 48.4 h / 29700 steps = **5.873 s/step** ⇒ **6.2 days** to step 121000,
**ETA ≈ 2026-08-20**. This confirms the user's "~6 days".

**⚠️ One correction to the user's framing.** The user asked whether missing step121000
"would make the main conclusion fail". The premise is slightly off: the risk is not that
the read-out is *missing*, but that a reader could think the pre-registered **P1/P2
binary** is what will be reported. It will not be — it was withdrawn on 08-13, at step
~8220, **before** the numbers existed. Any manuscript text must present the arm as
answering v2 §8's narrowed question, and must **not** print the `H_heal`/`H_family`
dichotomy as though it had been tested.

---

## 5. TWO SUBMISSION ROUTES

| | **Route A — submit without step121000** | **Route B — wait ~6 days** |
|---|---|---|
| **Claim** | Null calibration is a **necessary precondition** on construct validity in MC evaluation. Demonstrated on **10 constructs across 4 model families**, with the null's **four under-specifications** (tie convention, length unit, tokenizer, `mean(1/n_opt)`) measured and shown to reverse verdicts. Headline: under structural damage the letter interface degenerates to at/below its own best-constant floor while reading "above chance" — **14/15** cross-family MMLU-Pro cells, **0/60** damaged non-OLMo cells clearing their floor, **25/60** reading "above chance". Plus the **v2 permutation null** as a collapse-proof companion, with `Delta_perm ≡ 0` for any constant emitter proved as an **algebraic identity**. | Route A **plus** one appendix cell: whether a healed front-8 Qwen3 at 121k heal steps shows material item-level signal on MMLU-Pro. |
| **What must be dropped / re-scoped** | The heal-vs-no-heal leg becomes an **acknowledged open confound** (regime and family are perfectly collinear across all 21 cells). Already flagged in 3 README places, so this is a re-wording, not a retraction. | Nothing extra dropped. |
| **What is gained** | Nothing lost on the headline — the methodology claim is **independent by construction** (§4, quoted from paperC's own pre-registration, written pre-GPU). | Converts one caveat from "open" to "measured at n=1 family, 1 depth". |
| **What is NOT gained even in B** | — | `HEAL_CONFOUND_PREREGISTRATION §9`: n=1 family at 1 depth; Llama-2/Llama-3 and k10/k12/k14 stay confounded; **corpus stays unmatched** (5.72 epochs of 5.541B SlimPajama vs OLMo-2's 1.0 epoch of 31.7B Dolmino — unfixable, raw Dolmino text is on **neither disk**); relative depth untested; `H_heal`/`H_family` **still unavailable** (§4). |
| **Cost** | 0 GPU-h. Writing only. | ~6 days wall (already-sunk GPU) + ~1 GPU-h scoring + a **known one-line family-dispatch fix** (`eval_olmo2_mc_letter_content.py` imports `load_pruned_model` from the OLMo-2 module, which hardcodes `Olmo2Config`; `HEAL_CONFOUND_PREREGISTRATION §8` calls this "known, small, CPU-testable"). |
| **Critical-path reality** | **Writing is the critical path in BOTH routes.** M1-M5 (no tex, no figures, no tables, no venue) is a 2-3 week job. step121000 lands in 6 days — i.e. **inside the writing window either way.** |

### Recommendation: **Route A framing, Route B timing.**

Start writing **now** against Route A's claim set, because (a) the methodology claim
does not depend on the arm and paperC says so pre-GPU, (b) the writing gap is 2-3× the
training gap, and (c) if the arm is written in as load-bearing and then lands
`NO_ITEM_LEVEL_SIGNAL` — which the existing trajectory predicts — the paper would need
restructuring at the worst moment. The arm arrives on 2026-08-20, well inside the
drafting window, and can be dropped into the appendix whichever way it lands. **Do not
gate the draft on it.**

---

## 6. PRIORITISED WORK ORDER

| order | item | class | GPU | why first |
|---|---|---|---|---|
| 1 | Pick the venue (M2) | BLOCKING | 0 | Determines page budget, which determines how much of the 30-item retraction ledger survives. Everything else depends on it. |
| 2 | `evidence/construct_nulls_table.json` — one machine-readable source for the README's 9-row table (E2), and **fix the OBQA character-unit null** (E1: install pyarrow, recompute from the parquet already on disk, emit it) | BLOCKING | 0 | The only actual *evidence* hole (E1) plus the fragmentation fix (E2). ~1 h. Also kills the `0.6217`-vs-CKA and `0.491713`-vs-acc collision hazards (§2.3), and the rounded-vs-full-precision grep trap that fooled my own first pass. |
| 3 | Skeleton: `cp paperB/{acl.sty,BUILD.md,main.tex} paperC/`, then 12 stub sections + `03b_nulls.tex` (M1, M5) | BLOCKING | 0 | Unblocks all prose work. |
| 4 | `csv2tex` for the two big evidence CSVs → `tab_*.tex` (M4) | BLOCKING | 0 | Mechanical; prevents hand-typed drift. |
| 5 | Flat claim/retraction status table (E4) | 应做 | 0 | Needed before prose, or the intro will contradict the appendix. |
| 6 | Add Zheng et al. ICLR 2024 + re-argue `READOUT_V2 §4d` vs PriDe (N1); cite 2602.02182 (N2) | 应做 | 0 | Cheapest reviewer-risk reduction available. |
| 7 | 2 figures (M3) | BLOCKING | 0 | CPU plots off existing JSON. |
| 8 | Land step121000 read-out into the appendix (Route B) | 可选 | ~1 GPU-h | Arrives 08-20 on its own; do not block on it. |
| 9 | Cho camera-ready re-diff (N3), splice `NOVELTY_CHECK §3` softening (N4) | 应做 | 0 | Pre-submission hygiene. |

---

## 7. RAN / READ

### RAN (this session, first-hand, **0 GPU**)

| what | where | result |
|---|---|---|
| `ls paperC/*.tex`, `ls -la paperC/sections/` | LOCAL | **no tex; sections/ empty** (0 files, mtime 08-11 20:55) |
| `find paperC -iname '*.pdf' -o -iname '*.png'` | LOCAL | **0 figures** |
| Recompute 6 non-MMLU letter nulls from raw records | LOCAL CPU | **6/6 exact** (§2.3) |
| Recompute MMLU letter null + content split null + tie rate | LOCAL CPU | `0.2689075630`, `0.284450`, `34.2188%` — all exact |
| Recompute BoolQ null from `per_example_boolq.jsonl` | LOCAL CPU | `2033/3270 = 0.6217125382`, counts exact |
| Recompute OBQA token null, two tokenizers | LOCAL CPU | OLMo-2 `0.368000` (exact); Llama-2 `0.375833` (**tokenizer-valued**) |
| Attempt OBQA **character** null | LOCAL CPU | **FAILED at audit time** — no `pyarrow`/`datasets` on conda, `.venv`, `python3.11`, `python3`. E1. ✅ **CLOSED later the same day** with `venv_union9/bin/python` (pyarrow 25.0.1 + datasets 5.0.1), which the audit had not checked: `0.363500`, and 11 more values as self-test. |
| Recompute MMLU-Pro floor + `mean(1/n_opt)` + `n_opt` histogram from 8 shards | `.73` CPU | `0.116606`, `0.110877`, `{3:21,4:606,5:52,6:93,7:158,8:320,9:801,10:9981}` — **all exact**, n=12032, 12032 unique ids |
| Verify zwfy6-only record dirs + pinned ckpts | `.73` | all 5 present (310M/122M/122M/52M/284G) |
| Verify gate-3 fp32 headline numbers | LOCAL | present in `7B_keep8_step121000_dtype_summary.json` |
| Trap check: does `0.6217` in A01 evidence mean BoolQ? | LOCAL CPU | **NO** — it is `/c3_cka/per_pair[18]/null_mean` (§2.3) |
| Trap check: are A01's char-unit values nulls or accs in the CSV? | LOCAL CPU | 0 rows as null; `0.491713` appears **15× as `acc`** |
| Trap check: full-precision presence of `0.276`/`0.504897` in evidence JSON | LOCAL CPU | **both present** (`0.5048966267682263`) — corrected my own E2 first pass |
| arXiv API novelty pass, 14 queries | web via proxy | 0 preempting; 1 close preprint; 1 missing citation cluster |
| DBLP venue check ×2 | web | 2602.02182 = `CoRR 2026` informal; PA-GRPO = **0 hits** |
| OpenReview API2 venue check | web | 2309.03882 = `venueid ICLR.cc/2024/Conference`, ICLR 2024 **spotlight** |
| `nvidia-smi` read-only | `.73`, `.104` | `.73` 8×0 MiB/0%; `.104` 8×100%/78775 MiB (paperC training, **untouched**) |
| Training rate from log timestamps | `.104` read-only | step **29700**, **5.873 s/step** realised ⇒ ETA step121000 ≈ **2026-08-20** |

### READ (pre-existing)

`paperC/README.md`, `READOUT_V2_PREREGISTRATION.md`, `HEAL_CONFOUND_PREREGISTRATION.md`,
`HEAL_TRAJECTORY_READOUT_1.md`, `VENUE_AND_NOVELTY_VERIFICATION.md`,
`HEAL_CONFOUND_LAUNCH_RECORD.md`, all 6 `paperC/evidence/` JSONs + 4 CSVs + 4 verdict
`.md`s, `paperC/code/gate2_crossfamily_nulls.py` (the character-unit non-recoverability
docstring), `A01/STATUS.json`, `A01/NOVELTY_CHECK.md`, `A01/PROPOSAL.md`,
`A01/evidence/*` (incl. `gate3_dtype_runs/`), `paperA/main.tex`, `paperB/main.tex` +
both `sections/` listings.

**Not done, deliberately:** any GPU job; any write to `.104`/LOCAL/`.212`; any edit to
`paperC/README.md` or `A01/` (this audit only *reports* the splices needed); the
step-121000 read-out; writing any part of the manuscript.
