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
| MMLU content | longest-option split = `0.2845` | `0.25` |
| BoolQ | always-B = `0.6217` (2033 B / 1237 A) | `0.50` |
| OpenBookQA | longest-option = `0.3635` | `0.25` |

**Headline empirical result.** Under structural damage the letter MC interface
degenerates to at or below its own best-constant floor. Four families, n=14042/arm:

| family | damaged-arm letter acc | vs always-D floor `0.2689` |
|---|---|---|
| OLMo-2-7B (healed keep8) | `0.2550` | below |
| Llama-2-7B | `0.2295`–`0.2415` | below |
| Llama-3-8B | `0.2329`–`0.2527` | below |
| Qwen3-8B-Base | `0.2286`–`0.2301` | below |

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

⚠️ Two-disk rule (`memory/cluster-two-disks-not-shared`): this repo copy is on
**wzc1**; the bulk gate-3 and MMLU evidence lives on **zwfy6**. A file is "missing"
only after both disks have been searched.

## Open defects (carried over, must be closed before submission)

1. `evidence/gate3_dtype_runs/7B_base_dtype_summary.json:letter_acc_diff_boot_p = 1.042`
   — an **illegal p-value** (>1). The construction
   `2 * min((bs<=0).mean(), (bs>=0).mean())` in `code/a01_gate3_fp32_vs_bf16.py:345`
   is unclamped; ties in the bootstrap distribution let both tails exceed 0.5.
   Note `paired_bootstrap()` at line 393 already clamps *below* (`max(p, 1/n_boot)`)
   but not above. All six dtype summaries need re-emitting. **0 GPU** — the
   per-example shards are on disk.
2. Full second-MC-benchmark replication. BoolQ/OBQA are point evidence today;
   gate-2's winogrande is a structural degenerate (both options share the
   continuation → identical norm_lens, acc == acc_norm exactly, 100% tie rate) and
   is a **negative control only**. Also: gate-2's interface contrast is raw sum-LL
   vs length-normalised acc_norm, which is *analogous to but not identical with*
   MMLU's letter-vs-content — that difference must be stated wherever it is used.
