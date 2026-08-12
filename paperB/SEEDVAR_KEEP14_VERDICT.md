# Paper B — keep14+fresh2 seed variance at 200k (task #181): RESULTS & VERDICT

**Date**: 2026-08-12 CST. **Node**: LOCAL (8×L20A, wzc1) only. **Wall clock**: 33 min (01:06→01:39).
**Driver**: `scripts/_run_paperB_keep14_seedvar_local.sh` · **Analysis**: `scripts/analyze_paperB_keep14_seedvar.py`
**Machine-readable**: `paperB/SEEDVAR_KEEP14_RESULTS.json` · **Protocol pin**: `paperB/SEEDVAR_KEEP14_PROTOCOL.md`

## TL;DR

Two keep14+fresh2 runs (seed 42, seed 1234) trained with the identical recipe to step 200000
differ by **up to 1.2 pp on MMLU-letter** and **2.0 pp on BoolQ** while their **held-out PPL is
essentially identical (10.5613 vs 10.5673, Δ0.006 = 0.06%)**. Under Holm-Bonferroni over the
13-axis primary family **only BoolQ survives** (p=0.0035 < 0.00385); MMLU-L (p=0.0092) and
ARC-E (p=0.032) do not. Signs are **mixed** (7 axes favour seed 42, 5 favour seed 1234, 1 tie),
i.e. this is dispersion, not one run being better.

**Consequence for Paper B**: the paper's own caveat "Every trained row is one run"
(`tab_main_results.tex`) is now quantified rather than asserted, and it is **load-bearing**.
Cross-arm gaps in the paper that are **smaller than ~1.2 pp on MMLU-L or ~2 pp on BoolQ cannot
be attributed to the construction** — most importantly **ShortGPT − keep14 = +1.8 pp on MMLU-C**
(`tab_policy_endpoint.tex`) sits inside the same envelope as this seed pair's MMLU-L swing, while
the **+15.6 pp MMLU-L gap is ~13× the seed swing and stands comfortably**.

⚠️ **n=2 → df=1. The sd of two draws is NOT a reportable σ_run** (see §5).

## 1. Provenance check — the archived seed-42 numbers are portable (and one axis is not)

Before measuring the seed, the battery re-ran seed 42 on LOCAL/conda to isolate the
node+interpreter term. The archived rows were made on the now-retired **.252** with
`$WD/.venv/bin/python`; LOCAL's `.venv` has had no torch since 2026-08-04, so only
`/opt/conda/envs/torch-base/bin/python` (torch 2.13.0) is available.

| axis | archived (.252 / `.venv`) | re-run (LOCAL / conda) | Δ | verdict |
|---|---|---|---|---|
| PPL Dolmino | 10.561295076299 (`sum_nll` 19763937.398438) | 10.561295076299 (same `sum_nll`) | **0** | **bit-identical** |
| PPL WikiText-103 | 11.556173 | 11.556173 | **0** | bit-identical |
| PPL PG-19 | 15.426295 | 15.426295 | **0** | bit-identical |
| core6, all 6 tasks (acc **and** acc_norm) | — | — | **0.000000 pp** max | bit-identical |
| know5, all 5 tasks (acc **and** acc_norm) | — | — | **0.000000 pp** max | bit-identical |
| MMLU-content `letter_acc` | 0.318402 | 0.318544 | **−0.0142 pp** | **DRIFTED** |
| MMLU-content `content_raw_acc` | 0.354793 | 0.354650 | +0.0142 pp | DRIFTED |
| MMLU-content `content_norm_acc` | 0.383208 | 0.384133 | **−0.0926 pp** | **DRIFTED** |

**Reading.** For 14 of 17 measured quantities the archive reproduces to the last printed digit
across a node change *and* an interpreter change — the Paper B PPL/core6/know5 rows are portable.
The **MMLU-content harness is the exception**: `scripts/eval_olmo2_mmlu_content.py` received two
commits (`36ddb1e`, `7ac9653`, both 2026-08-08) after the archived 2026-08-02 run, and the drift
(0.09 pp on content_norm, 1 item on `letter_only`, 10 on `content_only`) is that driver boundary.
This is exactly the mechanism `status/PAPERB_WITHIN_DISK_FLOOR_V3.md` identified: *a systematic
bias between eval revisions, not zero-mean noise.*

**This is why both arms were re-run here.** Every seed delta below compares two arms produced by
**one driver, one commit (`b3626c3`), one interpreter, one node, back-to-back**, so the MMLU
driver drift cancels instead of contaminating the seed estimate.

## 2. Integrity (asserted by the analysis script, which raises rather than reporting a partial set)

| harness | dirs | shards | `n_scored` vs expected | `n_nan` |
|---|---|---|---|---|
| in-domain PPL | 2 | **8/8** each | 4096 windows / 8,384,512 tok, both arms equal | n/a |
| core6 downstream | 2 | **8/8** each | 10042 / 1172 / 2376 / 1838 / 1267 / 500 — **all exact, both arms** | **0** |
| know5 downstream | 2 | **8/8** each | 14042 / 5153 / 3270 / 1221 / 1954 — **all exact, both arms** | **0** |
| MMLU letter+content | 2 | **8/8** each | 14042 = `n` = `n_valid`, both arms | **0** |
| OOD PPL ×2 corpora | 4 | **1/1** each (archived protocol) | 288,627 / 2,456,400 tok, arms equal | n/a |

- Failure-syntax grep `Traceback \(most recent call last\)|CUDA out of memory|loss=nan` over all
  `logs/sv181_*.log`: **0 matches**. (Deliberately not the loose `grep -icE 'traceback|nan'`,
  which matches the *passing* check `✓ No NaN/Inf in model parameters`.)
- Per-item pairing: item_id sets identical between arms on every task; no duplicate item_ids;
  per-item means re-derive each summary `acc`/`acc_norm` to <1e-9 (else the script raises).
- `add_bos=false` recorded in every summary → **chat_template=False** base protocol held.
- McNemar implementation validated against the archived MMLU merge: reproduces
  `6.187518104901592e-33` at **ratio 1.000000**.

## 3. Per-axis delta table (seed 42 − seed 1234)

Exact McNemar on discordant pairs; paired bootstrap `n_boot=10000`, seed 42, resampling items.

### 3a. Likelihood axes (no per-item test; token-weighted)

| axis | tokens | seed 42 | seed 1234 | Δ PPL | Δ avg_nll |
|---|---:|---:|---:|---:|---:|
| PPL Dolmino held-out | 8,384,512 | **10.5613** | **10.5673** | −0.0060 | −0.000566 |
| PPL WikiText-103 (OOD) | 288,627 | 11.5562 | 11.5375 | +0.0186 | +0.001615 |
| PPL PG-19 (OOD) | 2,456,400 | 15.4263 | 15.4654 | −0.0391 | −0.002531 |

PPL agrees to **0.06 % / 0.16 % / 0.25 %** — and the sign even flips across corpora. **PPL is
seed-stable at this operating point.**

### 3b. Accuracy axes

| axis | n | seed 42 % | seed 1234 % | Δ pp | boot CI95 pp | flips | 42-only | 1234-only | McNemar p |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| **MMLU-L (letter)** | 14042 | 31.8544 | **33.0437** | **−1.189** | [−2.065, −0.299] | 4061 | 1947 | 2114 | **0.0092** |
| MMLU-C (content_norm) | 14042 | 38.4133 | 38.2353 | +0.178 | [−0.335, +0.684] | 1337 | 681 | 656 | 0.512 |
| MMLU content_raw | 14042 | 35.4650 | 35.4722 | −0.007 | [−0.463, +0.456] | 1101 | 550 | 551 | 1.000 |
| MMLU (know5 leg, acc) | 14042 | 31.9114 | 33.0081 | −1.097 | [−1.973, −0.199] | 4064 | 1955 | 2109 | 0.0164 |
| **BoolQ** acc_norm | 3270 | **68.8685** | 66.8196 | **+2.049** | [+0.734, +3.394] | 513 | 290 | 223 | **0.0035** |
| BoolQ acc | 3270 | 63.8226 | 61.5902 | +2.232 | [+0.642, +3.884] | 701 | 387 | 314 | 0.0065 |
| ARC-E acc_norm | 2376 | 70.4966 | 71.8855 | −1.389 | [−2.652, −0.168] | 223 | 95 | 128 | 0.0319 |
| SocialIQA acc | 1954 | 43.3982 | 42.0676 | +1.331 | [+0.102, +2.559] | 148 | 87 | 61 | 0.0395 |
| PIQA acc | 1838 | 75.1904 | 74.0479 | +1.143 | [+0.000, +2.285] | 113 | 67 | 46 | 0.0594 |
| CSQA acc | 1221 | 49.8771 | 51.1057 | −1.229 | [−3.112, +0.573] | 133 | 59 | 74 | 0.225 |
| ARC-C acc_norm | 1172 | 43.7713 | 44.8805 | −1.109 | [−2.986, +0.683] | 123 | 55 | 68 | 0.279 |
| PIQA acc_norm | 1838 | 74.5375 | 75.1360 | −0.598 | [−1.687, +0.490] | 105 | 47 | 58 | 0.329 |
| WinoGrande acc | 1267 | 62.5888 | 63.2202 | −0.631 | [−2.762, +1.500] | 186 | 89 | 97 | 0.608 |
| LAMBADA acc | 5153 | 57.7334 | 58.1215 | −0.388 | [−1.281, +0.505] | 568 | 274 | 294 | 0.425 |
| SocialIQA acc_norm | 1954 | 47.4411 | 47.1341 | +0.307 | [−0.870, +1.484] | 140 | 73 | 67 | 0.673 |
| HellaSwag acc_norm | 10042 | 64.4593 | 64.4394 | +0.020 | [−0.428, +0.478] | 552 | 277 | 275 | 0.966 |
| OpenBookQA acc_norm | 500 | 40.4000 | 40.4000 | **0.000** | [−2.200, +2.200] | 30 | 15 | 15 | 1.000 |

**Macro aggregates** (Paper B conventions):

| aggregate | seed 42 | seed 1234 | Δ pp |
|---|---:|---:|---:|
| core6 macro (acc_norm; WinoGrande acc) | .593756 | .599936 | **−0.618** |
| aux5_raw (acc mean over know5; *not* a knowledge-recovery claim) | .493485 | .491786 | +0.170 |

### 3c. Multiplicity

Holm-Bonferroni over the 13-axis primary family (one metric per task + 3 MMLU protocols):

| rank | axis | p | Holm threshold | decision |
|---:|---|---:|---:|---|
| 1 | BoolQ acc_norm | 0.00353 | 0.00385 | **REJECT null** |
| 2 | MMLU-L | 0.00918 | 0.00417 | retain |
| 3 | ARC-E acc_norm | 0.03189 | 0.00455 | retain |
| 4-13 | all others | ≥0.225 | — | retain |

3/13 axes are nominally p<0.05 versus 0.65 expected under a global null, so there is **real
seed-driven signal**, but after correction **only BoolQ is individually established**. The
scientifically useful quantity is not "which axis is significant" — it is the **magnitude
envelope**: two identical recipes land ~1.2 pp apart on MMLU-L and ~2 pp apart on BoolQ.

Note the **interface asymmetry**: MMLU-**letter** flips 4061/14042 items (28.9 %) between seeds
while MMLU-**content_norm** flips 1337 (9.5 %) and moves only +0.18 pp. The letter interface is
~3× more seed-labile. This *reinforces* Paper B's existing interface-dependence finding
(`tab_interface_audit`, `rebuttal_snippets/tab_letter_headroom.tex`): the letter protocol is the
fragile one, on this axis too.

## 4. What this does to specific Paper B claims

| claim | gap | vs seed envelope | verdict |
|---|---|---|---|
| `tab_main_results` keep14 PPL 10.561 | — | seed Δ 0.006 (0.06 %) | **safe**; quote 3 dp at most |
| `tab_policy_endpoint` ShortGPT − keep14 **MMLU-L +15.6 pp** | 15.6 pp | ~13× the 1.19 pp seed swing | **safe** |
| `tab_policy_endpoint` ShortGPT − keep14 **MMLU-C +1.8 pp** | 1.8 pp | **same order** as the 1.19 pp MMLU-L / 2.05 pp BoolQ swings | ⚠️ **cannot be attributed to construction on n=1 per arm** |
| `tab_policy_endpoint` ShortGPT − keep14 **core6 +2.8 pp** | 2.8 pp | ~4.5× the 0.62 pp core6-macro seed swing | **probably safe**, but state it is n=1 per arm |
| `tab_policy_endpoint` **BoolQ +9.1 pp** | 9.1 pp | ~4.4× the 2.05 pp BoolQ seed swing | **safe** |
| `tab_policy_endpoint` **ARC-C +3.8 pp** | 3.8 pp | ~3.4× the 1.11 pp ARC-C seed swing | probably safe |
| `tab_policy_endpoint` **HellaSwag +4.1 pp** | 4.1 pp | ~200× the 0.02 pp HellaSwag seed swing | **safe** |
| `tab_ood_audit` ordering Base < ShortGPT < keep14 | ≥0.78 PPL | seed Δ ≤0.039 PPL | **safe** — ordering cannot be seed-flipped |
| `tab_main_results` footnote "Every trained row is one run" | — | — | **keep it, and now cite this measurement** |

Recommended paper text (one sentence, no new table needed):

> A second keep14+fresh2 run with an independent seed reaches PPL 10.567 (vs 10.561) but MMLU-L
> 33.0 (vs 31.9) and BoolQ 66.8 (vs 68.9), so single-run target scores carry roughly a
> point-scale run-to-run spread even where perplexity is reproduced to 0.06 %; comparisons of
> a magnitude below about two points should not be read as construction effects.

## 5. ⚠️ Statistical caveat: n=2 → df=1, no reportable σ_run

The sd of two draws has **1 degree of freedom**. A 95 % χ²₁ interval on σ multiplies the sample
sd by `[sqrt(1/5.023886), sqrt(1/0.000982)] = [0.446, 31.91]` — a **~72× wide** interval:

| axis | two draws (pp) | sample sd (pp) | 95 % CI on σ_run (pp) |
|---|---|---:|---|
| MMLU-L | 31.8544 / 33.0437 | 0.841 | **[0.375, 26.84]** |
| core6 macro | 59.3756 / 59.9936 | 0.437 | [0.195, 13.95] |
| BoolQ acc_norm | 68.8685 / 66.8196 | 1.449 | [0.646, 46.23] |

So: **report the deltas, do not report a σ.** The deltas themselves are per-item paired
measurements on 14042 / 3270 / … items and are properly bounded by the McNemar p and bootstrap
CI above; it is only the *variance across runs* that is unestimable at n=2. A third seed would
take df to 2 and shrink the σ interval to roughly [0.52, 3.7]× — still wide; a genuine σ_run
needs ~5 seeds (each ~76 h on 8×L20A, i.e. ~3 weeks of one node).

⚠️ **Do not pool with A03.** A03's pooled σ (df=5, 0.3666 pp, χ² CI [0.229, 0.899]) is
**OLMo-2 1B keep7/keep12 on TriviaQA** — different model, scale, task, and harness. Mixing them
would be a category error.

## 6. Axes deliberately NOT measured

Closed-book **PopQA / TriviaQA / NQ-open** (Paper B reports keep14 = .142 / .294 / .060) are
excluded: the seed-42 baseline for keep14 exists **only on zwfy6**
(`olmo2_closedbook_results/7B_keep14_step200000{,_v2}`), wzc1 has no keep14 closed-book dir, and
wzc1's HF cache lacks `nq_open`. Measuring them here would put seed 42 on H20 and seed 1234 on
L20A, confounding seed with architecture. To close these axes, re-run **both** arms' closed-book
leg on one node. Also excluded: `olmo2_mc_letter_content_results` (a paperC gate-2 artefact from
.73, not a Paper B keep14 row).

## 7. Reproduce

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
setsid nohup bash scripts/_run_paperB_keep14_seedvar_local.sh > logs/sv181_main.log 2>&1 &   # 8 GPU, 33 min
/opt/conda/envs/torch-base/bin/python scripts/analyze_paperB_keep14_seedvar.py               # CPU, writes the JSON
```
The driver hard-fails on a pre-existing output dir (never clobbers an archived row) and refuses
to merge fewer than 8/8 shards.
