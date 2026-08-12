# Paper C — Null Calibration for Construct Validity

Promoted from `proposal/active/A01-null-calibration-methodology/` on **2026-08-11**
per the user's instruction 「可以 那就晋升」 and the revised promotion rule in
`proposal/README.md` §晋升规则 (a verified finding is sufficient; a preset checklist
is not required).

> ⚠️ **The proposal directory is NOT deleted and remains the authoritative entry
> point for evidence and decision history**, per `CLAUDE.md`
> 「proposal 目录不删，它是该方向的证据与决策历史入口」. Every number cited here must
> trace back to `proposal/active/A01-null-calibration-methodology/`.

## The claim, at the scope it actually survives

Report every construct against a **construct-appropriate null** — the best-constant
(input-blind) predictor — **not** against chance, before comparing arms.

Where it has been demonstrated:

| construct | correct null | what chance would have said |
|---|---|---|
| MMLU letter | always-D = `0.2689` | `0.25` |
| MMLU content | longest-option split = `0.2845` (token unit) | `0.25` |
| BoolQ | always-B = `0.6217` (2033 B / 1237 A) | `0.50` |
| OpenBookQA | longest-option = `0.3635` **(character unit)**; `0.3680` token unit | `0.25` |
| ARC-Challenge letter | always-B = `0.265358` | `0.250156` |
| ARC-Easy letter | always-C = `0.266414` | `0.250161` |
| OpenBookQA letter | always-A = `0.276000` | `0.25` |
| CommonsenseQA letter (5-way) | always-B = `0.208845` | `0.20` |
| PIQA letter | always-B = `0.504897` | `0.50` |
| MMLU-Pro letter (**10-way**, `n_opt` varies) | always-A = `0.116606` | `0.100000` naive, or `0.110877` = `mean(1/n_opt)` |

> ⚠️ The longest-option null is under-specified in **three** ways, not one: the tie
> convention (`split`/`first`/`last`/`credit`/`wrong`), the **length unit**
> (characters vs continuation tokens), and — within the token unit — the
> **tokenizer**. Always print all three. The character-vs-token choice moves the
> `split` null by up to 2.0 pp and the `credit` null by up to 35 pp; the tokenizer
> moves `split` by up to 1.5 pp and `credit` by up to **10.6 pp** across
> Llama-2 / Llama-3 / Qwen3 on the identical items (larger BPE vocabularies tie more
> option lengths). So a content floor is a property of (dataset, convention, unit,
> **tokenizer**), not of the dataset alone — unlike the letter floor, which *is* a
> pure dataset property. See `evidence/SECOND_MC_BENCHMARK_VERDICT.md` §5a and
> `evidence/GATE2_CROSSFAMILY_VERDICT.md` §2b.

**Headline empirical result.** Under structural damage the letter MC interface
degenerates to at or below its own best-constant floor. Four families, n=14042/arm
(MMLU); all four now also verified on five **non-MMLU** MC benchmarks:

| family | damaged-arm letter acc | vs always-D floor `0.2689` |
|---|---|---|
| OLMo-2-7B (healed keep8) | `0.2550` | below |
| Llama-2-7B | `0.2295`–`0.2415` | below |
| Llama-3-8B | `0.2329`–`0.2527` | below |
| Qwen3-8B-Base | `0.2286`–`0.2301` | below |

On MMLU all **9/9** damaged non-OLMo arms are *significantly* below floor (boot
p ≤ 0.0008; recomputed 2026-08-11 with the R-7 mid-p estimators). Off MMLU the
verdict holds — **0 of 60** damaged non-OLMo arm×benchmark cells clear their own
floor, and **25 of 60 read "above chance"** under the naive null — but only 7/60
reach p<0.05 because those benchmarks are 6–28× smaller than MMLU; see
`evidence/GATE2_CROSSFAMILY_VERDICT.md`.

**Why this is not a numerical bug** (gate-3, `GATE3_VERDICT.md`, verdict
`MECHANISM_FALSIFIED`): full-fp32 forward removes **100%** of bf16 exact ties and
changes **18.03%** of the damaged arm's letter argmax decisions, yet letter accuracy
does not move — Δ = **−0.0015**, CI95 `[−0.0064, +0.0033]`, exact McNemar
**p = 0.570** — and the arm sits below its floor *more* significantly in fp32
(−1.54 pp, boot p = 0.0062) than in bf16 (−1.39 pp, p = 0.0192).

> A defect that fp32 can fix is an engineering note. One it cannot fix is a
> construct-validity problem. Reshuffling 2,532 coin-flips does not recover
> information that is not there.

## Scope discipline — what must NOT be claimed

These are retractions and narrowings this direction already made against itself.
They are load-bearing; the self-falsification narrative is part of the contribution.

- ✗ "letter is a family-general **step function** / sharp phase transition" —
  RETRACTED 2026-08-10 (external audit). Keep per-family descriptive form only.
- ✗ "damage turns letter into a **constant** predictor" — NARROWED to "drives letter
  to or below its best-constant floor". Modal share and floor verdict are DECOUPLED
  (modal share is only 43–45% on several below-floor arms).
- ✗ "**exact ties are the mechanism**" as a family-general causal claim —
  quadruple-falsified. Mechanism is family-specific: OLMo-2 via bf16 exact ties
  (30.64% on keep8) resolved by argmax index bias; Llama/Qwen via direct modal
  collapse (Llama-2 k12 = 100.0% modal / 0.00% ties).
- ✗ "letter MC is **generally** an unreliable instrument" — false on intact models.
  On three of four HEALTHY strong models letter is **+13 to +23 pp BETTER** than
  content, so "content is the fair interface" is not a general statement either.
- ✗ acc-vs-acc_norm length sensitivity as **ours** — preempted by Oostermeijer,
  ICML 2026 (arXiv:2607.12767). Reframe OBQA sign flips as replication under damage.
- ✗ "interface swap does not rescue a damaged arm: content_norm sits within **±3 pp**
  of letter on every damaged arm" — RE-SCOPED TO MMLU 2026-08-11. Off MMLU the gap is
  large: `keep8` content−letter is **+38.76 pp** on ARC-Easy, +22.77 on CommonsenseQA,
  +19.15 on PIQA. See `evidence/SECOND_MC_BENCHMARK_VERDICT.md` §4.
- ✗ "the damaged letter interface is **significantly** below its floor on benchmarks
  other than MMLU" — only ARC-Easy (`keep10`, p = 0.029) shows this on OLMo-2, and
  cross-family only **7 of 60** damaged non-OLMo cells reach p<0.05 while **52 of 60
  are underpowered to have detected MMLU's own −1.389 pp**. The point estimates are
  NOT smaller (arc_challenge's median damaged effect is −3.840 pp vs MMLU's −3.603),
  so this is a power limit, not a null result. Cite the power table with any null
  there, or use the pooled n=7107 construction
  (`GATE2_CROSSFAMILY_VERDICT.md` §4: 12/12 negative, 4/12 significant).
- ✗ "**k14 is the last arm that still clears its floor**" as a family-general
  ordering — ADDED 2026-08-11 (#250). True of OLMo-2's *healed* arms; **false in all
  three non-OLMo families**, whose k14 is already at/below floor. Their damaged
  ladder is a **cliff, not a gradient** (k14→k8 spread 0.46–7.00 pp, monotone in
  2/15 ladders) so **no depth curve may be fitted to those rungs**. ⚠️ confounded
  with heal-vs-no-heal; do not report it as a family effect.
- ✗ a **pooled** across-benchmark floor verdict quoted as a per-benchmark verdict —
  the pooled floor `0.318700` mixes five floors spanning `0.2088`–`0.5049`.
- ✗ a longest-option **content** floor quoted without its **tokenizer** — the token
  unit is tokenizer-dependent (`credit` moves up to 10.6 pp across families on
  identical items). The **letter** floor, by contrast, IS a pure dataset property and
  is asserted invariant across all 15 arms.
- ✗ "**larger BPE vocabularies tie more** option lengths" as the *mechanism* for that
  tokenizer dependence — RETRACTED 2026-08-12 (#252). On MMLU-Pro's 10-way items the
  32k Llama-2 (49.7% tied) and 152k Qwen3 (51.9%) tie **less** than the 100k OLMo-2
  (56.24%) and 128k Llama-3 (56.23%), so it is **not monotone in vocab size**. The
  tokenizer *dependence* stands; the explanation does not.
- ✗ "significantly below floor is **MMLU-specific**" — an intermediate #251 reading
  from the OLMo-2 leg alone, RETRACTED by #252: MMLU-Pro's `llama2_7b/k8`
  (p = 0.0168) and `qwen3_8b_base/k8` (p = 0.0362) are significantly below floor.
  The separating variable looks like **heal vs no-heal**, not benchmark.
  > ⚠️ **NARROWED 2026-08-13** (`READOUT_V2_PREREGISTRATION.md`). These two cells
  > are below the **arm-independent best-constant** floor, which is the correct
  > null for the *instrument-validity* claim and that claim stands. But both are
  > below it **because they collapse onto a letter other than `A`**, and `always-A`
  > is the floor by construction. Against an arm-conditional permutation null
  > neither is below: `qwen3/k8` **−0.139 pp, p = 0.0964**; `llama2/k8` **−0.416 pp,
  > p = 0.1002**. So "below floor" here is a statement about the **interface**, not
  > evidence that the arm is *worse than input-blind*. Do not use these cells to
  > argue heal-vs-no-heal: under the collapse-proof null the P1 asymmetry vanishes.
- ✗ "damage drives letter to or below its floor" as a **universal** — it is **14/15**
  on the cross-family MMLU-Pro cells; `qwen3_8b_base/k14` is significantly **above**
  its floor (+0.233 pp, p = 0.0192) at hw 0.191 pp, i.e. a real exception, not noise.
  > ⚠️ **The `k14` exception is WEAKER than this reads, 2026-08-13.** It is
  > significant against the permutation null too (+0.267 pp, p = 0.0066) but at
  > `recovery_fraction = 0.049`, i.e. **9.1%** of the same family's intact anchor —
  > under A04's 10% materiality bar — while emitting `A` on **94.6%** of 12032
  > items. v2 labels it `TRACE_SIGNAL`, not a capability. Report it as "a real but
  > immaterial exception", never as "k14 retains MMLU-Pro competence".
- ✗ **any number from the FIRST cross-family MMLU-Pro launch**
  (`mmlu_pro_lc_crossfamily_results/`, `MAXLEN=1536`) — 10 of its 15 cells were
  scored with the labelled option body partly left-truncated and `llama2_7b_base`
  never completed. Use `mmlu_pro_lc_crossfamily_results_fix/` and
  `mmlu_pro_power_nulls_v2.json` only.
- ✗ "clearing a floor is **sufficient**" — Feng et al. requires the opposite: state
  explicitly that it is necessary, not sufficient.
- ✗ the numbers in `STATUS.json:must_not_resurrect` (`4.8x`, `0.2822` as an
  arithmetic error, `58/91`, `0.25` as the MMLU null, `0.50` as the BoolQ null,
  `45.74pp` as Qwen3's transition jump).

## Citation obligations inherited from the novelty check

`NOVELTY_CHECK.md`, verdict `kill_clause_3_DOES_NOT_FIRE`, confidence medium-high.

- **Balepur et al., ACL 2024 main** (arXiv:2402.12483) — origin of "use stronger than
  chance baselines in MCQA". Do not claim it.
- **Zheng et al., ICLR 2025 Oral** (arXiv:2410.07137) — origin of "null model" and
  "a constant predictor can top a benchmark". Do not claim it.
- **Oostermeijer, ICML 2026** (arXiv:2607.12767) — drop the acc/acc_norm sub-claim.
- **OLMES, Findings of NAACL 2025** (arXiv:2406.08446, `2025.findings-naacl.282`) —
  origin of the letter(MCF)/cloze(CF) interface split. ⚠️ **Do NOT describe OLMES's
  rule as SIZE-keyed** — corrected 2026-08-12 after a full-text read of the Anthology
  camera-ready (verified independently by MAIN, not just by the audit subagent).
  OLMES's actual rule is **max-over-interfaces, per task per model**: "we standardize
  to evaluate each model using both the MCF and CF formulations, and the best
  performing one is used" (p. 5026); Table 7's `max` column is defined as "taking the
  best of MCF and CF for each task" (p. 5038). Model size is only their *narrative*
  for why the max lands where it does, not the selection key. The correctly-stated
  defect, which is a *stronger* claim: OLMES's only reference line for "is MCF
  meaningful for this model" is **"random"** — the string "chance" occurs **0 times**
  in the paper, and "majority" / "marginal" occur **0 times** as well — and its Part-1
  discussion asserts *without measuring* that a model which "might highly prefer a
  given label (like B)" "would not be much better than random" because "the benchmarks
  in OLMES are generally balanced" (p. 5035). paperC measures exactly that quantity and
  it is **not** the chance line: always-B on ARC-Challenge is `0.265358` vs `0.250156`,
  always-A on OpenBookQA is `0.276000` vs `0.25`, always-A on MMLU-Pro is `0.116606`
  vs `0.100000` (**1.1661×**). Position the floor test as (i) supplying the null
  OLMES's own robustness argument presupposes, and (ii) noting that max-over-interfaces
  is an *uncalibrated* selection — it can report a number from an interface that does
  not clear its own floor, because no floor is ever computed. Also: OLMES dismisses the
  tokenizer objection to per-token normalisation ("this does not seem like a relevant
  argument", p. 5037) — that dismissal is **valid in its own scope** (ranking answer
  choices "keeping model and tokenizer fixed") and does **not** cover paperC's finding,
  which is that a *content floor compared across models* is tokenizer-dependent. Say so
  explicitly rather than presenting OLMES as simply wrong.
- **Cho et al. ICLR 2026** (arXiv:2502.18798) + **Bean et al. NeurIPS 2025 D&B**
  (arXiv:2511.04703) — framing/parallel literature. Do not claim "MC interface
  validity is unexamined".
- **Layer-order null**: downgrade from "first" to "we are not aware of a prior
  layer-order null for layer correspondence".
- ~~Three venues still **UNVERIFIED** at the Anthology/OpenReview level.~~
  **CLOSED 2026-08-12** — all three verified at the family-correct authority; see
  [`VENUE_AND_NOVELTY_VERIFICATION.md`](VENUE_AND_NOVELTY_VERIFICATION.md) §1.
  **Ding et al.** = NeurIPS 2021 Poster, *Advances in NeurIPS* 34, pp. 1556–1568,
  OpenReview `venueid = NeurIPS.cc/2021/Conference` + DBLP `conf/nips/DingDS21` +
  the official proceedings page. ⚠️ **The camera-ready title differs from arXiv's:**
  cite "Grounding Representation Similarity **Through** Statistical Testing"
  (arXiv:2108.01661 says "**with**"). **Hewitt & Liang** = EMNLP-IJCNLP 2019 **main**,
  Anthology `D19-1275`, DOI `10.18653/v1/D19-1275`, pp. 2733–2743, DBLP
  `conf/emnlp/HewittL19` (`booktitle = EMNLP/IJCNLP (1)` = main volume, not Findings —
  Findings did not exist in 2019). **Feng et al.** = ACL 2019 **main**, Anthology
  `P19-1554`, DOI `10.18653/v1/P19-1554`, pp. 5533–5538, DBLP `conf/acl/FengWB19`
  (`booktitle = ACL (1)` = main). All three are peer-reviewed main-track; none is a
  workshop or preprint.
- ~~No full-text PDF pass has been done on any candidate.~~ **CLOSED 2026-08-12** —
  all nine candidates read in full (camera-ready where obtainable); see
  [`VENUE_AND_NOVELTY_VERIFICATION.md`](VENUE_AND_NOVELTY_VERIFICATION.md) §2.
  **No candidate preempts.** 0 of 9 computes a best-constant/input-blind null
  per-construct as a *precondition on arm comparison*; **none** reports any of
  paperC's floors (`0.2689` / `0.2845` / `0.6217` / `0.3635` / `0.116606` appear in
  zero candidate PDFs); only OLMES touches BoolQ and only Oostermeijer touches OBQA.
  Two of the mandatory citations contain **defects paperC can correct**: Balepur et al.
  impute `0.25` ("random guessing") for invalid outputs (ACL p. 10310) inside the very
  experiment that argues chance is the wrong reference — under their own MMLU letter
  marginal the correct imputation is `0.2689`; and OLMES's interface diagnostic is
  stated against "random", never against a label-marginal floor (see the corrected
  OLMES bullet above). One residual gap: **Cho et al.'s ICLR camera-ready PDF could
  not be fetched** (OpenReview `/pdf` is behind a bot challenge from this network), so
  the full-text read is of **arXiv v4 (2026-01-12)**, two weeks before the
  camera-ready `pdate` 2026-01-26. arXiv-vs-camera-ready was diffed successfully for
  Balepur (no substantive change).
- Venue/full-text provenance: `paperC/VENUE_AND_NOVELTY_VERIFICATION.md` (the
  authority actually queried is named per paper; the PDFs were read, not the abstracts).

## Provenance

| what | where |
|---|---|
| decision history, gate verdicts, retraction ledger | `proposal/active/A01-null-calibration-methodology/` |
| external audit + our response | `.../TCODEX_AUDIT_RESPONSE.md` |
| recompute code | `.../code/` |
| gate-3 per-example records (6 arms × 8 shards) | **zwfy6** `results/a01_gate3/dtype_runs/` — reachable from `.73`/`.82`/`.104` only |
| MMLU letter/content per-item records | **zwfy6** `olmo2_mmlu_content_results/` |
| non-MMLU letter/content per-item records (6 arms × 6 tasks × 8 shards) | **BOTH DISKS** `olmo2_mc_letter_content_results/` (52 MB) |
| second-MC-benchmark nulls + stats | `paperC/evidence/second_mc_benchmark/gate2_letter_content_nulls.{json,csv}` |
| **cross-family** letter/content per-item records (15 arms × 6 tasks × 8 shards) | **BOTH DISKS** `mc_lc_crossfamily_results/` (130 MB) |
| **cross-family** nulls + stats (1122 rows) | **BOTH DISKS** `paperC/evidence/second_mc_benchmark_crossfamily/gate2_crossfamily_nulls.{json,csv}` |
| **cross-family** MMLU per-item records (`gate1_*`, incl. the 12 recomputed cells) | **wzc1 ONLY** `olmo2_mmlu_content_results/gate1_*` (190 MB) |
| cross-family recompute code (CPU, 0 GPU) | `paperC/code/gate2_crossfamily_nulls.py` |
| cross-family driver | `scripts/_run_mc_letter_content_crossfamily_8gpu.sh` |
| **MMLU-Pro** nulls + stats, all 21 cells (231 rows) | **BOTH DISKS** `paperC/evidence/mmlu_scale_power/mmlu_pro_power_nulls_v2.{json,csv}` (the `_v2`-less pair is the OLMo-2-only 66-row predecessor) |
| **MMLU-Pro** per-item records | **zwfy6 ONLY** `mmlu_pro_letter_content_results/` (6 OLMo-2 arms, ~1.1 GB) + `mmlu_pro_lc_crossfamily_results_fix/` (15 non-OLMo arms). ⛔ `mmlu_pro_lc_crossfamily_results/` (no `_fix`) is the **defective** first launch — BEFORE side only |
| **MMLU-Pro** recompute + audit code (CPU, 0 GPU) | `paperC/code/mmlu_pro_power_nulls.py`, `paperC/code/mmlu_pro_trunc_audit.py`, `paperC/code/mmlu_pro_trunc_fix_compare.py` |
| **MMLU-Pro** driver | `scripts/_run_mmlu_pro_letter_content_8gpu.sh` (`MODE=olmo2\|crossfamily`, `MAXLEN` default 2048) |
| Qwen3-8B-**Base** weights | wzc1 `../models/Qwen3-8B-Base`; copied to zwfy6 same name. ⚠️ zwfy6's pre-existing `models/Qwen--Qwen3-8b` and the `models/Qwen3-8b-local` symlink are **Qwen3-8B-Instruct** (`eos 151645`, has `chat_template`) — **not** a valid base arm |

⚠️ Two-disk rule (`memory/cluster-two-disks-not-shared`): this repo copy is on
**wzc1**; the bulk gate-3 and MMLU evidence lives on **zwfy6**. A file is "missing"
only after both disks have been searched.

## Open defects (carried over, must be closed before submission)

*(both items closed 2026-08-11; the residual to-dos they created are listed at the end of item 2)*

1. ~~`evidence/gate3_dtype_runs/7B_base_dtype_summary.json:letter_acc_diff_boot_p = 1.042`
   — an **illegal p-value** (>1).~~ **CLOSED 2026-08-11** — see
   [`evidence/R7_BOOTSTRAP_P_FIX.md`](evidence/R7_BOOTSTRAP_P_FIX.md).
   The root cause was *not* a missing clamp: `2 * min((bs<=0).mean(), (bs>=0).mean())`
   **double-counts** the resamples whose mean is exactly 0 (`<=` and `>=` both
   include them), so the two tails sum to `1 + P(bs==0)` rather than 1. `d` is a
   difference of two 0/1 correctness vectors, so on the base arm 13986/14042 items
   have `d = 0` and **5.44%** of bootstrap means land exactly on 0 — enough to push
   the doubled smaller tail to 1.042. Fixed by a shared `two_sided_boot_p()` that
   splits the zero atom evenly between the tails (mid-p), making the `p ≤ 1` bound
   *structural*; `paired_bootstrap()` (which feeds every `*_vs_null_boot_p` and had
   the same bug, clamped only from below) now uses it too. All six dtype summaries
   re-emitted from the on-disk shards, **0 GPU**, all 8/8 shards / n=14042 / nan=0.
   **Impact on conclusions: none.** Base arm 1.042 → **0.9876**; 0 of 24 verdicts
   changed, 0 of 30 p-values crossed α=0.05, every non-p field byte-identical.
   The keep8 below-floor headline is unchanged (p = 0.0190 bf16 / 0.0060 fp32).
2. ~~Full second-MC-benchmark replication. BoolQ/OBQA are point evidence today;
   gate-2's interface contrast is raw sum-LL vs length-normalised acc_norm, which is
   *analogous to but not identical with* MMLU's letter-vs-content.~~
   **CLOSED 2026-08-11** (task #248) — see
   [`evidence/SECOND_MC_BENCHMARK_VERDICT.md`](evidence/SECOND_MC_BENCHMARK_VERDICT.md),
   verdict `REPLICATES_PARTIALLY_AND_NARROWS_THE_CLAIM`.
   MMLU's *exact* letter-vs-content contrast (letter prompt = content prompt with the
   labelled `A./B./C./D.` body spliced in before `\nAnswer:`) now exists on five
   non-MMLU MC benchmarks — **arc_challenge** (always-B `0.265358`), **arc_easy**
   (always-C `0.266414`), **openbookqa** (always-A `0.276000`), **commonsense_qa**
   (5-way, always-B `0.208845`), **piqa** (always-B `0.504897`) — plus **winogrande**
   (always-B `0.504341`) as the negative control, across the same six arms, 36/36
   cells shard-complete with `n_nan = 0`. **Replicates:** of the 15 damaged arm×task
   cells, **10 read "above chance" and 0 clear their own best-constant floor**; the
   arm ordering and floor-arrival point hold on 5/5; the `credit` tie convention
   flips 5/6 arms below the content floor on arc_challenge, as on MMLU.
   **Does NOT replicate:** the *significant* below-floor letter verdict — only
   arc_easy yields one (`keep10`, −2.694 pp, p = 0.029), and **four of the five
   tasks are underpowered to have detected MMLU's own −1.389 pp** (CI95 half-width
   1.31–6.40 pp vs MMLU's 1.15 pp). That power table must accompany any citation of a
   null result here. Two self-falsifications fell out: the longest-option null has a
   **second** under-specification (character vs **token** length unit; OBQA `0.3635`
   is the *character* null, token is `0.3680`), and
   `confirmed_general[2]`'s "content_norm within ±3 pp of letter on every damaged
   arm" is **false off MMLU** (arc_easy `keep8` is at its letter floor `0.2584` while
   scoring `0.6460` on content, +38.76 pp, McNemar p = 1e-147).

   **CROSS-FAMILY EXTENSION 2026-08-11** (task #250) — see
   [`evidence/GATE2_CROSSFAMILY_VERDICT.md`](evidence/GATE2_CROSSFAMILY_VERDICT.md),
   verdict `REPLICATES_IN_DIRECTION_ACROSS_FAMILIES_BUT_THE_LADDER_DOES_NOT`.
   #248 closed the *contrast* but left the whole second-benchmark leg inside ONE
   family, while MMLU's headline is four-family. That asymmetry is now closed: the
   same harness, unchanged, ran on **Llama-2-7B / Llama-3-8B / Qwen3-8B-Base** ×
   {intact, k14, k12, k10, k8} × the same six tasks — 90 cells, 8/8 shards,
   `n_nan = 0` everywhere, 12.5 min on `.73` (damage is an **eval-time**
   front-N truncation, no fresh block, no heal, so no training was needed).
   **Replicates:** **0 of 60** damaged non-OLMo arm×task cells clear their own
   best-constant letter floor, in any family on any benchmark, and **25 of 60 read
   "above chance"** under the naive null — the wrong-null flip, now in three more
   families off MMLU. 51/60 point estimates negative. Recomputing the archived MMLU
   cross-family cells with the R-7 mid-p estimators gives **9/9 significantly below
   floor**, so `STATUS.json:gate1_third_model_family_DAMAGED`'s point deltas survive
   proper statistics.
   **Does NOT replicate (two separate narrowings).** (a) *Significance*: only 7/60
   non-MMLU cells reach p<0.05, and **52/60 are underpowered to have detected MMLU's
   own −1.389 pp**. This is purely n, not effect size — arc_challenge's median
   damaged effect is **−3.840 pp, larger than MMLU's −3.603 pp**, yet n.s. because
   its CI95 half-width is 3.92 pp vs MMLU's 1.18. Pooling the five **disjoint**
   benchmarks into one n=7107 paired sample (same estimator, no new assumption)
   recovers part of it: 12/12 damaged arms negative, **4/12 significant**; a pooled
   verdict must never be quoted per-benchmark. (b) **#248's "k14 is the last arm
   above its floor" is NOT family-general** — all three non-OLMo k14 arms are
   *already* at/below their floors, and the damaged ladder is a **cliff, not a
   gradient** (k14→k8 spread 0.46–7.00 pp, monotone in only 2/15 ladders), so **no
   depth curve may be fitted to these rungs**. ⚠️ (b) is confounded with
   **heal vs no-heal** (OLMo-2's arms had 121k–200k heal steps, these have none) and
   this run cannot separate regime from family.
   **Third self-falsification of the longest-option null:** beyond the tie
   convention and the character-vs-token unit, *within* the token unit the null is
   **TOKENIZER-dependent** — arc_challenge `split` moves `0.268871` (Llama-2) →
   `0.283902` (Llama-3), and the `credit` convention moves up to **10.6 pp** across
   families, because larger BPE vocabularies tie more option lengths (tied-longest
   fraction 33.8% → 48.2% on OBQA, 48.4% → **86.5%** on winogrande). **A content
   floor must be quoted with its tokenizer.** Also newly measured: damaged
   `content_norm` is below its own `split` floor on **12/12** OBQA cells, and 16
   damaged cells are *literal* constant emitters whose accuracy equals the marginal
   of the emitted letter to machine precision — two of them landing exactly on the
   **optimal** constant (`0.276000`, Δ = 0.000 pp, CI95 `[0,0]`), which is invisible
   against chance. Modal share and floor verdict stay **DECOUPLED** (99.91% modal at
   p = 0.0499 vs 96.46% modal at p = 0.0015).
   **Residual to-do — TRAINING NOW RUNNING 2026-08-12:** healed non-OLMo arms are
   needed to separate the heal-vs-no-heal confound in (b). A healed
   **Qwen3-8B-Base front8+fresh2** arm is now training on `.104`
   (pre-registered: [`HEAL_CONFOUND_PREREGISTRATION.md`](HEAL_CONFOUND_PREREGISTRATION.md);
   launch evidence: [`HEAL_CONFOUND_LAUNCH_RECORD.md`](HEAL_CONFOUND_LAUNCH_RECORD.md)).
   ⚠️ It will **not** rehabilitate (b) as phrased: (b) is about the *ladder shape*
   over k14→k8 in three families, and **one arm at one depth in one family cannot
   fit a depth curve** — the "no depth curve may be fitted to these rungs"
   prohibition stands regardless of how this arm lands. What it addresses is the
   narrower question in the POWER WALL residual below.

   **POWER WALL CLOSED 2026-08-11/12** (tasks #251 + #252) — see
   [`evidence/POWER_WALL_VERDICT.md`](evidence/POWER_WALL_VERDICT.md), verdict
   `POWER_WALL_CLEARED_AND_THE_MMLU_EFFECT_DOES_NOT_REPLICATE` (on *healed* arms).
   The "52/60 underpowered" residual above is the defect this closes, and it could
   **not** be closed by adding more small benchmarks — openbookqa would need
   ~10615 items and its entire test set is 500. It needed MMLU-scale `n`.
   **MMLU-Pro** (`n = 12032`, up to **10-way**) delivers it: across **21 cells**
   (6 OLMo-2 healed arms + 15 non-OLMo arms × the same eval-time truncation
   ladder) the CI95 half-width on the letter-vs-floor test is **0.083–0.968 pp**,
   median **0.727** — **21/21 powered** against MMLU's own −1.389 pp, versus 1/6
   of #248's tasks and 8/60 of #250's cells. This is the **first non-MMLU
   benchmark in the paper with the resolution to answer the question at all**.
   **Replicates:** the wrong-null flip, now at full power in **four families** —
   **10/12** damaged non-OLMo cells read "above chance" under `mean(1/n_opt)`
   (**12/12** under naive `0.10`) while **1/12** clears its own best-constant
   floor `always-A 0.116606`; plus **3/3** on OLMo-2. The letter floor is asserted
   **bit-identical across all 21 cells**. Modal/floor **decoupling** is confirmed a
   third time and now at *both* extremes: 99.13% and 98.83% modal cells sit **AT**
   the floor while a **29.12%** modal cell sits **BELOW** it.
   **Does NOT replicate, and one reversal.** (a) On OLMo-2's healed arms MMLU's
   *significant* below-floor headline **fails with power**: `keep8` is −0.116 pp,
   CI95 **[−0.698, +0.465]**, an interval that **excludes −1.389 pp** — a positive
   exclusion, not a blind spot. (b) ⚠️ But below-floor is **NOT MMLU-specific**:
   on the same benchmark the un-healed `llama2_7b/k8` (−0.914 pp, p = 0.0168) and
   `qwen3_8b_base/k8` (−0.881 pp, p = 0.0362) **are** significantly below floor.
   The live explanation is **heal vs no-heal** — the same confound as (b) above —
   not benchmark identity. (c) A genuine counterexample to any universal
   "damage ⇒ at-or-below floor" phrasing: `qwen3_8b_base/k14` is significantly
   **above** its floor (+0.233 pp, p = 0.0192, hw 0.191). The honest form is
   **14/15**.
   **Fourth under-specification, on the LETTER side:** when `n_opt` varies,
   "chance" itself is ambiguous — naive `1/10 = 0.100000` gives floor − chance =
   **+1.661 pp** (1.1661×, the paper's strongest *relative* misstatement) while
   `mean(1/n_opt) = 0.110877` gives **+0.573 pp** (mid-pack in absolute pp).
   MMLU-Pro's `n_opt` is **not** constant (9981 items are 10-way, 606 are 4-way),
   so the naive reading is misleading. **Report both, and the ratio alongside the
   pp gap.** The letter *floor* remains a pure dataset property; the *gap to
   chance* is not.
   **`credit`-convention reductio gets its best number:** `0.532164` on MMLU-Pro
   — a pure length heuristic with oracle tie-breaking that **beats the intact
   OLMo-2 base model's content_norm (`0.207613`) by 32.5 pp**. Also: the 10-way
   content floor **is** tokenizer-dependent (`credit` spans 9.26 pp across the four
   tokenizers) but ⚠️ **#250's vocab-size explanation is retracted** — it is not
   monotone in vocab size (32k Llama-2 and 152k Qwen3 both tie *less* than 100k
   OLMo-2 and 128k Llama-3). Keep the fact, drop the mechanism.
   **Two integrity defects of the first cross-family launch, fixed in #252**
   (§6g): (i) `MAXLEN=1536` was measured on the **OLMo-2** tokenizer (max 1226 tok)
   and is wrong for Llama-2 (1678) and Qwen3 (1660), so **10 of 15 cells** were
   scored with part of the labelled option body **left-truncated** — and since the
   overflow set is tokenizer-specific, the table was **not item-matched**. Fixed by
   raising the cap to **2048** (not by dropping items, which would have broken the
   full-`n` match with the archived MMLU cells) and re-running: `n_trunc = 0` on
   15/15. Measured impact **0/14 verdicts changed, max letter-acc change 0.0083 pp
   (one item), all 9 argmax flips confined to the affected items, 0 flips
   elsewhere** — the defect was real but benign, which was **not** knowable in
   advance (40/12032 = 0.332% vs 0.1–0.9 pp effects). (ii) `llama2_7b_base`
   produced **nothing** (OOM on 5/8 shards; the guard correctly refused a 3/8
   merge) because Llama-2 has **no GQA** (`num_kv_heads = 32` × 32 layers =
   **72.0 GiB** fp32 KV at B=48/L=1536, vs 18.0/20.2 for the GQA families) — fixed
   with `use_cache=False`, which is unused in a single teacher-forced pass;
   94 GiB → 41–50 GiB. The missing intact-Llama-2 cell is now scored and is
   striking: only **+1.538 pp** above the floor, vs Llama-3's +21.3 and Qwen3's
   +34.5. **Hardening: `n_trunc` was a WARNING and is now a hard per-shard
   assert** — that weakness is exactly why 10 truncated cells shipped with
   summaries written.
   **Residual to-do — TRAINING NOW RUNNING 2026-08-12, read-out ETA ≈ 2026-08-20:**
   the heal-vs-no-heal confound is the single biggest open question in the
   direction — it is what separates (a) from (b) — and it needs healed non-OLMo
   arms, i.e. real training. **That training is now running.** A healed
   **Qwen3-8B-Base `keep_front=8` + `n_fresh=2`** arm (10 layers, 3.1741B params,
   eff_bs 128, cosine horizon 200k, **read-out pre-registered at step 121000** =
   `olmo2_7b/keep8`'s own scored step) started 2026-08-12 14:18 on `.104`,
   measured **5.718 s/step** ⇒ ≈ **8.0 days** to the read-out. See
   [`HEAL_CONFOUND_PREREGISTRATION.md`](HEAL_CONFOUND_PREREGISTRATION.md) (written
   and committed **before** any GPU, so the arm cannot be re-chosen post-hoc) and
   [`HEAL_CONFOUND_LAUNCH_RECORD.md`](HEAL_CONFOUND_LAUNCH_RECORD.md).

   **Design choice, stated so it is not mistaken for a relative-depth match:** the
   arm matches OLMo-2's **absolute** keep-depth (front-**8**), *not* its depth
   fraction. Qwen3-8B has **36** layers vs OLMo-2-7B's **32**, so `keep8` is 22.2%
   vs 25.0% and no integer depth is both absolute- and fraction-matched. Absolute
   was chosen because all 15 existing non-OLMo cells are literal front-N slices
   (`load_truncated_any_family`), so front-8 keeps the healed arm paired with the
   **`qwen3_8b_base/k8`** cell whose verdict disagrees with `olmo2_7b/keep8`; a
   front-9 arm would have traded the heal confound for a **depth** confound.

   **What it will resolve:** the within-family contrast `qwen3_8b_base/k8`
   (un-healed, −0.881 pp, p = 0.0362, **BELOW** floor) vs the same arch **healed**,
   holding family / tokenizer / benchmark / floor / keep-depth fixed and varying
   **only heal**. That identifies whether "significantly below floor" is a property
   of the **un-healed regime** (which would explain why healed `keep8` reads AT
   floor with a CI that excludes −1.389 pp) or a **family** property.

   **What it will NOT resolve, even if it succeeds** (full list in the
   pre-registration §9): it is **n = 1 family at 1 depth** — Llama-2/Llama-3 and
   k10/k12/k14 stay confounded; the **corpus stays unmatched** (Qwen3 cannot
   consume OLMo-2-token Dolmino and raw Dolmino text is on **neither disk**, so the
   arm heals on SlimPajama — **5.72 epochs of 5.541B** tok vs OLMo-2's 1.0 epoch of
   31.7B; the corpus was re-tokenized from all 48 shards with the correct **Base**
   EOS 151643 and the arm restarted onto it at step 240, which *shrinks* this
   caveat but does not remove it); **relative depth stays untested**; and it says
   **nothing** about the
   null-calibration methodology claim, which is paperC's actual contribution and
   does not depend on this leg at all.

   ⚠️ **Do not describe either side of this comparison as using differential LR.**
   Verified in the logs: `keep8`/`keep14` and the new Qwen3 arm all log **only**
   `inh_decay` + `inh_nodecay` at a **uniform 2e-5** with **no `fresh_*` group**,
   because `_classify_param` lacks the `module.` prefix strip while
   `build_param_groups` runs *after* the DDP wrap — so `--lr 1e-4` is a **no-op** on
   both sides. This bug-for-bug match is what makes them comparable, but the claim
   "differential LR" would be false. (`keep14fresh2_seed1234`, launched after the
   fix, *does* show a `fresh_decay` group and is therefore **not** comparable on
   this axis.)

   ⚠️ **`models/Qwen--Qwen3-8b` is Qwen3-8B-*Instruct*** (`eos 151645`
   `<|im_end|>`, ctx 40960), **not** base — reconfirming line 184. All five
   pre-existing `qwen3_minarch_*` arms record it as `base_model_path`, so they are
   invalid under paperC's `chat_template=False` protocol; this, not just their short
   7.5k/19k-step budgets, is why a fresh arm was required. The new arm uses
   `models/Qwen3-8B-Base` (`eos 151643`, ctx 32768) and the launcher **hard-refuses
   to start** if `eos_token_id != 151643`.
