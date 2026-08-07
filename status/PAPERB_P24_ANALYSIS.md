# Paper B P2.4 SFT repairability — full pre/post-SFT assembly with per-item pairing

**Date**: 2026-08-08 CST. **Author**: sub-agent (dispatched by MAIN, all three arms landed).
**GPU cost**: 0 (all batteries produced earlier; this doc is pure analysis).
**Cross-reference**: `PAPERB_P24_SFT_REPAIRABILITY.md` (P2.4 setup), `PAPERB_P24_KEEP14_EVAL.md`,
`PAPERB_P24_FULL32_ARM.md`, `PAPERB_P24_SHORTGPT16_ARM.md`, `PAPERB_P24_WZC1_EVAL.md`,
`PAPERB_CORE6_CROSSARCH_FLOOR.md`. All are owned by MAIN — do not edit.

---

## Headline

**No arm shows SFT-driven recovery on core knowledge.** All three arms *lose* PPL, *lose*
core6 macro, and *lose* TriviaQA under Tulu-3-general SFT. MMLU letter-protocol
accuracy moves in different directions per arm (full32 down, ShortGPT-16 and keep14 up),
but MMLU content-normalised accuracy is either flat or slightly down for all three.
**Deleting the 45 contaminated MMLU items changes every arm's Δ by ≤3e-4** — the SFT
effect on MMLU is not carried by contamination.

Verdict, mapped to P2.4's pre-registered questions (`PAPERB_P24_SFT_REPAIRABILITY.md`
§7):
1. "MMLU up **and** independent closed-book knowledge up" — **FALSE** (PopQA/TriviaQA
   both flat/down).
2. "Only MMLU up" — **partially true for shortgpt16 letter-only, not for content**.
   Interpretation: cosmetic gain on letter interface, no gain in content-scored
   knowledge.
3. "keep14 remains far behind ShortGPT" — **remains true post-SFT**: keep14 MMLU
   letter `.328` vs shortgpt16 `.495`; content `.375` vs `.396`.
4. "Full base gets a similar gain" — full32 letter *decreases* (−1.0 pp, statistically
   significant), content stays flat. So the shortgpt16-letter gain is **not** a
   generic SFT effect that would apply equally to any starting point.

## Pairing convention (each arm within one disk)

Per `PAPERB_CORE6_CROSSARCH_FLOOR.md` §"Action items": pair pre vs post on the same
architecture. Executed as:

| arm | pre run | post run | disk | architecture |
|---|---|---|---|---|
| full32 | `7B_full32_base_wzc1` | `7B_p24_sft_full32_final` | wzc1 | L20A cc10.0 |
| ShortGPT-16 | `7B_shortgpt16_step200000_wzc1` | `7B_p24_sft_shortgpt16_final` | wzc1 | L20A cc10.0 |
| keep14+fresh2 | `7B_keep14_step200000` | `7B_p24_sft_keep14fresh2_final` | zwfy6 | H20 cc9.0 |

Held-out Dolmino PPL pre-values reproduce paper anchors: full32 `7.398071` vs paper
`7.398`; keep14 `10.561151` vs paper `10.561`; ShortGPT-16 `9.780042` vs paper `9.780`.

## Statistical machinery

- **McNemar exact two-sided**: over per-item paired binary correctness for {pre,post},
  `p = min(1, 2·Binomial(n=b+c, k=min(b,c), p=0.5).cdf(min(b,c)))` in log-space (avoids
  the `math.comb` overflow flagged in `PAPERB_P24_SFT_REPAIRABILITY.md` §8.4). **Explicitly
  two-sided** — the sidedness memory (`ratio-quoted-unlabelled` → treat as one-sided)
  was for a different context; here every reported `p` is two-sided.
- **Paired bootstrap**: 10,000 percentile resamples on the per-item difference
  `post_correct − pre_correct ∈ {−1, 0, +1}`; seed 0; numpy-vectorised.

## Full pre/post-SFT table (all three arms)

Δ is `post − pre` in the natural accuracy metric (proportion), and PPL Δ is absolute.
"95% CI" is paired-bootstrap percentile. "McN p" is two-sided McNemar exact.

### full32 (intact base + Tulu-3 SFT, wzc1)

| axis | n | pre | post | Δ | 95% CI on Δ | McN p |
|---|---:|---:|---:|---:|---:|---:|
| Held-out Dolmino PPL | 8,384,512 tok | 7.3981 | 7.7277 | **+0.3297** | — | — |
| MMLU letter | 14,042 | .6063 | .5961 | **−0.0101** | [−0.0160, −0.0043] | **6.9e-4** |
| MMLU content-norm | 14,042 | .4703 | .4738 | +0.0035 | [−0.0006, +0.0075] | 0.101 |
| **MMLU letter (clean, drop 45)** | **13,997** | .6073 | .5975 | **−0.0099** | [−0.0157, −0.0041] | **9.6e-4** |
| **MMLU content-norm (clean)** | **13,997** | .4707 | .4740 | +0.0034 | [−0.0008, +0.0075] | 0.115 |
| core6 macro | — | .7040 | .6951 | **−0.0089** | — | — |
| core6/hellaswag | 10,042 | .8048 | .7906 | −0.0142 | [−0.0184, −0.0100] | **3.2e-11** |
| core6/arc_challenge | 1,172 | .5725 | .5683 | −0.0043 | [−0.0222, +0.0137] | 0.707 |
| core6/arc_easy | 2,376 | .8283 | .8211 | −0.0072 | [−0.0173, +0.0029] | 0.199 |
| core6/piqa | 1,838 | .8107 | .8063 | −0.0044 | [−0.0141, +0.0054] | 0.445 |
| core6/openbookqa | 500 | .4620 | .4600 | −0.0020 | [−0.0220, +0.0180] | 1.000 |
| core6/winogrande | 1,267 | .7459 | .7245 | −0.0213 | [−0.0395, −0.0032] | 0.026 |
| PopQA (contains) | 14,267 | .2577 | .2558 | −0.0019 | [−0.0061, +0.0023] | 0.399 |
| TriviaQA (EM) | 17,944 | .6350 | .6195 | **−0.0155** | [−0.0209, −0.0104] | **5.8e-9** |

### ShortGPT-16 200k + Tulu-3 SFT (wzc1)

| axis | n | pre | post | Δ | 95% CI on Δ | McN p |
|---|---:|---:|---:|---:|---:|---:|
| Held-out Dolmino PPL | 8,384,512 tok | 9.7800 | 10.6125 | **+0.8325** | — | — |
| MMLU letter | 14,042 | .4735 | .4945 | **+0.0210** | [+0.0131, +0.0288] | **1.2e-7** |
| MMLU content-norm | 14,042 | .4013 | .3958 | −0.0055 | [−0.0103, −0.0006] | 0.030 |
| **MMLU letter (clean, drop 45)** | **13,997** | .4741 | .4954 | **+0.0213** | [+0.0134, +0.0290] | **8.7e-8** |
| **MMLU content-norm (clean)** | **13,997** | .4016 | .3963 | −0.0053 | [−0.0102, −0.0004] | 0.037 |
| core6 macro | — | .6219 | .6097 | **−0.0122** | — | — |
| core6/hellaswag | 10,042 | .6850 | .6720 | −0.0130 | [−0.0182, −0.0080] | **6.6e-7** |
| core6/arc_challenge | 1,172 | .4744 | .4480 | −0.0265 | [−0.0452, −0.0077] | **8.0e-3** |
| core6/arc_easy | 2,376 | .7449 | .7176 | −0.0274 | [−0.0387, −0.0156] | **5.9e-6** |
| core6/piqa | 1,838 | .7606 | .7601 | −0.0005 | [−0.0120, +0.0109] | 1.000 |
| core6/openbookqa | 500 | .4100 | .4080 | −0.0020 | [−0.0240, +0.0200] | 1.000 |
| core6/winogrande | 1,267 | .6567 | .6527 | −0.0039 | [−0.0237, +0.0158] | 0.756 |
| PopQA (contains) | 14,267 | .1578 | .1534 | −0.0043 | [−0.0083, −0.0004] | 0.036 |
| TriviaQA (EM) | 17,944 | .3300 | .2905 | **−0.0395** | [−0.0444, −0.0346] | **2.2e-55** |

### keep14+fresh2 200k + Tulu-3 SFT (zwfy6)

| axis | n | pre | post | Δ | 95% CI on Δ | McN p |
|---|---:|---:|---:|---:|---:|---:|
| Held-out Dolmino PPL | 8,384,512 tok | 10.5612 | 11.5569 | **+0.9957** | — | — |
| MMLU letter | 14,042 | .3184 | .3281 | **+0.0097** | [+0.0011, +0.0182] | **0.025** |
| MMLU content-norm | 14,042 | .3832 | .3745 | −0.0087 | [−0.0135, −0.0040] | **4.3e-4** |
| **MMLU letter (clean, drop 45)** | **13,997** | .3191 | .3284 | **+0.0094** | [+0.0007, +0.0179] | **0.031** |
| **MMLU content-norm (clean)** | **13,997** | .3835 | .3748 | −0.0087 | [−0.0134, −0.0039] | **4.2e-4** |
| core6 macro | — | .5953 | .5832 | **−0.0121** | — | — |
| core6/hellaswag | 10,042 | .6439 | .6342 | −0.0097 | [−0.0144, −0.0048] | **1.2e-4** |
| core6/arc_challenge | 1,172 | .4420 | .4266 | −0.0154 | [−0.0333, +0.0026] | 0.111 |
| core6/arc_easy | 2,376 | .7029 | .6755 | −0.0274 | [−0.0400, −0.0152] | **1.6e-5** |
| core6/piqa | 1,838 | .7470 | .7486 | +0.0016 | [−0.0082, +0.0114] | 0.828 |
| core6/openbookqa | 500 | .4040 | .3900 | −0.0140 | [−0.0360, +0.0080] | 0.281 |
| core6/winogrande | 1,267 | .6322 | .6243 | −0.0079 | [−0.0284, +0.0126] | 0.500 |
| PopQA (contains) | 14,267 | .1422 | .1424 | +0.0002 | [−0.0036, +0.0040] | 0.942 |
| TriviaQA (EM) | 17,944 | .2939 | .2648 | **−0.0290** | [−0.0336, −0.0244] | **5.8e-36** |

### Reader-friendly summary (post − pre in pp)

| axis | full32 | ShortGPT-16 | keep14+fresh2 |
|---|---:|---:|---:|
| PPL (absolute Δ, ↓ better) | +0.33 | +0.83 | +1.00 |
| MMLU letter (pp) | **−1.0** | **+2.1** | **+1.0** |
| MMLU content-norm (pp) | +0.3 | −0.5 | −0.9 |
| MMLU letter clean (pp) | **−1.0** | **+2.1** | **+0.9** |
| MMLU content-norm clean (pp) | +0.3 | −0.5 | −0.9 |
| core6 macro (pp) | −0.9 | **−1.2** | **−1.2** |
| PopQA contains (pp) | −0.2 | −0.4 | +0.0 |
| TriviaQA EM (pp) | **−1.5** | **−4.0** | **−2.9** |

## D3.5 — MMLU contamination-clean rescore

45 MMLU test items (0.32% of 14,042) had exact-substring overlap with the Tulu-3
SFT corpus (subject histogram: `high_school_mathematics=36, high_school_statistics=3,
college_chemistry=2, nutrition=2, international_law=1, logical_fallacies=1`). IDs
regenerated locally on zwfy6 with the same fast-audit identity used originally; count
matches the published 45. Full ID list preserved at
`data/olmo2_sft/tulu3_general_contam_mmlu_ids.json` on zwfy6 (recopy to wzc1 optional
— tiny file).

**The 45-item filter changes every arm's MMLU Δ by ≤3e-4** (letter: ±3e-4; content-norm:
±2e-4). Every McNemar `p` moves in the fourth significant digit. **The MMLU-letter
"improvement" observed on shortgpt16/keep14 and MMLU-letter "regression" on full32 are
NOT contamination-carried.** They are distributional shifts in the SFT model's
answer-letter preferences (see below).

If the paper wants to be maximally conservative, the "clean" columns are the ones to
quote. But qualitatively the P2.4 verdict is unchanged.

## Mechanistic note — why MMLU letter goes up but content-norm goes down on ShortGPT-16

The MMLU-letter interface asks the model to name a letter; the model's letter-token
biases are shaped by whatever letter distribution the SFT data has. Tulu-3 general-clean
does not contain multiple-choice format (the deny-filter removed MC/QA sources) but does
contain instruction-following (math, chat, code) that biases letter-token probabilities.
The observation that letter improves while content-norm drops is consistent with an
**answer-letter prior shift**, not a knowledge gain. This matches the P2.4 pre-registered
"only MMLU up = interface adaptation" branch.

The observation that this letter-prior effect **helps** the two 16L arms
(ShortGPT-16 +2.1 pp, keep14 +1.0 pp) but **hurts** the intact full32 (−1.0 pp) is
suggestive: the 16L arms had a worse pre-SFT letter distribution (higher letter tie
rates — `PAPERB_P24_SFT_REPAIRABILITY.md` §6, base tie=.0013 vs keep14 tie=.2547), so
any prior sharpening helps them more; the intact full32 already has a well-shaped
letter distribution and any shift can only degrade it. Not a knowledge story — a
calibration story.

## Concordant/discordant pair counts (for the paper's methods section)

Let `a = both correct`, `b = pre-only`, `c = post-only`, `d = both wrong`. For MMLU letter:

| arm | a | b | c | d | n |
|---|---:|---:|---:|---:|---:|
| full32 | 7,573 | 935 | 793 | 4,741 | 14,042 |
| ShortGPT-16 | 5,254 | 1,397 | 1,692 | 5,699 | 14,042 |
| keep14 | 2,585 | 1,750 | 1,886 | 7,821 | 14,042 |

The keep14 row shows the noise level in this experiment: **discordant pair count is
3,636 = 26% of all items**. Only 249 net differ. This is a *very* noisy per-item signal
even at n=14k, which is why the CI on Δ is wide even where the statistic is
"significant."

## Provenance

- **Per-example predictions**:
  - wzc1: `olmo2_mmlu_content_results/{7B_full32_base_wzc1,7B_p24_sft_full32_final,7B_shortgpt16_step200000_wzc1,7B_p24_sft_shortgpt16_final}/per_example_mmlu.jsonl`
  - wzc1: `olmo2_downstream_results/…/per_example_{hellaswag,arc_challenge,arc_easy,piqa,openbookqa,winogrande}.jsonl`
  - wzc1: `olmo2_closedbook_results/…/per_example_{popqa,triviaqa}.jsonl`
  - zwfy6 (fetched to `/tmp/keep14_*`): same paths under `7B_keep14_step200000` and `7B_p24_sft_keep14fresh2_final`.
- **PPL**: `olmo2_ppl_results/{...}/summary.json` on both disks.
- **Contamination ID list**: regenerated on zwfy6 with the identity from `/tmp/fast_audit.py`; script `/tmp/get_contam_ids.py` (Python 3.11+datasets, ~50 s runtime, no GPU).
- **Analysis code**: `/tmp/p24_analysis/run_analysis.py` (Python 3.11 + numpy 2.3, 10 s runtime).
- **Numeric outputs**: `/tmp/p24_analysis/results.json` (all statistics).
- **Chat template**: `chat_template=False`, `add_bos=false` throughout — matches paper's
  base-LM protocol per memory `paper-eval-chat-false-mandatory`.
