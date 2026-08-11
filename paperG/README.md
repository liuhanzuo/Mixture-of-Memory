# Paper G — Null Calibration for Construct Validity

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
- **OLMES, Findings of NAACL 2025** (arXiv:2406.08446) — origin of the letter/cloze
  interface split. Position the floor test as a FIX to a defect in OLMES's
  SIZE-keyed interface-selection rule.
- **Cho et al. ICLR 2026** (arXiv:2502.18798) + **Bean et al. NeurIPS 2025 D&B**
  (arXiv:2511.04703) — framing/parallel literature. Do not claim "MC interface
  validity is unexamined".
- **Layer-order null**: downgrade from "first" to "we are not aware of a prior
  layer-order null for layer correspondence".
- ⚠️ Three venues still **UNVERIFIED** at the Anthology/OpenReview level
  (Ding et al. NeurIPS 2021, Hewitt & Liang EMNLP 2019, Feng et al. ACL 2019) —
  verify per `memory/venue-verify-acl-family-needs-anthology` before submission.
  S2 and DBLP were both DOWN (HTTP 429 / 500) during the novelty check.
- No full-text PDF pass has been done on any candidate. All overlap judgements are
  from title + abstract + venue metadata.

## Provenance

| what | where |
|---|---|
| decision history, gate verdicts, retraction ledger | `proposal/active/A01-null-calibration-methodology/` |
| external audit + our response | `.../TCODEX_AUDIT_RESPONSE.md` |
| recompute code | `.../code/` |
| gate-3 per-example records (6 arms × 8 shards) | **zwfy6** `results/a01_gate3/dtype_runs/` — reachable from `.73`/`.82`/`.104` only |
| MMLU letter/content per-item records | **zwfy6** `olmo2_mmlu_content_results/` |
| non-MMLU letter/content per-item records (6 arms × 6 tasks × 8 shards) | **BOTH DISKS** `olmo2_mc_letter_content_results/` (52 MB) |
| second-MC-benchmark nulls + stats | `paperG/evidence/second_mc_benchmark/gate2_letter_content_nulls.{json,csv}` |
| **cross-family** letter/content per-item records (15 arms × 6 tasks × 8 shards) | **BOTH DISKS** `mc_lc_crossfamily_results/` (130 MB) |
| **cross-family** nulls + stats (1122 rows) | **BOTH DISKS** `paperG/evidence/second_mc_benchmark_crossfamily/gate2_crossfamily_nulls.{json,csv}` |
| **cross-family** MMLU per-item records (`gate1_*`, incl. the 12 recomputed cells) | **wzc1 ONLY** `olmo2_mmlu_content_results/gate1_*` (190 MB) |
| cross-family recompute code (CPU, 0 GPU) | `paperG/code/gate2_crossfamily_nulls.py` |
| cross-family driver | `scripts/_run_mc_letter_content_crossfamily_8gpu.sh` |
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
   **Residual to-do:** healed non-OLMo arms would be needed to separate the
   heal-vs-no-heal confound in (b); that is real training and is not in scope.
