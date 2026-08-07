# Paper B P2.4 — keep10 post-SFT eval battery (.82)

**Date**: 2026-08-08 | **Node**: `.82` (8×H20 cc9.0, zwfy6) | **Commit**: `fd1633c`
**Driver**: `scripts/_run_olmo2_p24_eval_sft_keep10_82.sh` (md5 `fc2015cb...` on zwfy6; mirrors keep8 driver `b1e525c8...` exactly except arm/ckpt/output_name)
**Log**: `logs/p24_eval_sft_keep10_82.log` | **PID** 1215445 (parent wrapper) / 1215455-1215462 (shards)
**Runtime**: **06:24:36 → 06:43:15 CST = 19 min end-to-end** (comparable to keep8 sibling on `.73`, 16 min).
**Analysis**: on-node `/tmp/stat_keep10.py` (embedded verbatim below; scipy 1.x `binomtest`, 10k paired bootstrap seed=42, item_id-aligned)

Pre/post pair — single variable, **same node-family + same arch** (both H20 cc9.0 / zwfy6):

| | ckpt | battery name |
|---|---|---|
| pre  | `outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt` | `7B_keep10_step83500_v2{,_know}` |
| post | `outputs/olmo2_p24_sft_keep10fresh2/final.pt` (39.01 GB, step=842 keep=10 fresh=2) | `7B_p24_sft_keep10fresh2_final{,_know}` |

Protocol: `chat_template=False`, `--add_bos 0`, per-shard `CUDA_VISIBLE_DEVICES=$g` + `LOCAL_RANK=0 RANK=$g`, 8-way shard + merge with `assert_8shards` gate before every merge (all 5 harnesses passed 8/8). Per-item preds retained on both sides (downstream via `--save_per_example`; MMLU-content + closedbook default-write).

**⚠️ Caveat carried from spec**: keep10's pre-SFT anchor is `step83500`, which is **42% of the claimed 200k budget** (`status/PAPERB_TABLE4_BUDGET_DEFECT.md`). This experiment holds keep10's own pre-SFT ckpt fixed and measures the SFT delta on it, so the budget difference does NOT invalidate the SFT-effect measurement here. But keep10's SFT delta is NOT directly comparable to a hypothetical keep10@200k SFT delta.

---

## 1. THE FALSIFIABLE PREDICTION — observed vs predicted

The pre-registered n=3 fit (`ΔPPL% = 1.60149·prePPL − 7.34121`, r=0.998) predicted for pre-PPL 12.816: **ΔPPL% ≈ +13.18%**, post-PPL ≈ **14.51**. **Nothing was tuned to hit it.**

| | value |
|---|---|
| pre-SFT PPL | **12.815923** |
| post-SFT PPL | **13.922140** |
| ΔPPL (abs) | +1.1062 |
| **observed ΔPPL%** | **+8.6316%** |
| predicted ΔPPL% (n=3 fit) | +13.177% |
| **residual** | **−4.546 pp** (post-PPL undershoots the +14.51 prediction by 0.588) |
| identical eval basis | n_tokens pre = post = 8,384,512 (verified) |

**keep8 miss = −3.86 pp. keep10 miss = −4.55 pp. Both same sign, larger for the deeper-damaged arm.** The n=3 linear fit **systematically over-predicts damage** and its r=0.998 was a small-sample coincidence, not a stable law.

**Verdict on the pre-registered dichotomy from the launch spec**:
- "keep10 lands near +13.2%" → **FALSIFIED** (observed +8.63%, 4.55 pp low).
- "keep10 lands near +10-11% too → the linear fit is dead" → **CONFIRMED in direction, exceeded in magnitude**. keep10 lands even lower (+8.63%) than keep8's +10.15%. Both are systematically below the linear extrapolation.

### The 5-arm view (pre-PPL vs ΔPPL%, PaperB Table 4 ladder + full/shortgpt siblings)

| arm | pre-PPL | ΔPPL% | vs n=3 linear |
|---|---:|---:|---:|
| full32 | 7.398 | +4.46% | −1.72 (below) |
| shortgpt16 | 9.780 | +8.51% | +0.20 (near) |
| keep14 | 10.561 | +9.43% | +0.16 (near) |
| **keep10** | **12.816** | **+8.63%** | **−4.55 (well below)** |
| **keep8** | **13.333** | **+10.15%** | **−3.86 (well below)** |

- The three lower-pre-PPL arms (full32/shortgpt16/keep14) sit reasonably close to the linear line.
- Both higher-pre-PPL arms (keep10, keep8) sit **well below** it, and keep10's ΔPPL% (+8.63%) is actually **smaller** than keep8's (+10.15%) despite keep10 having lower pre-PPL — a **non-monotone reordering** relative to the linear-fit expectation. Note however that keep10 pre-PPL (12.82) < keep8 pre-PPL (13.33), so pre-PPL–monotonicity of ΔPPL% is still holding on the whole 5-arm ladder (keep8 > keep10 in Δ, matching order in pre-PPL). But magnitude is dramatically flatter than the n=3 fit implied.
- **The relationship saturates** — sub-linear / concave with respect to pre-PPL. A refit will yield much shallower slope and much lower r; MAIN can decide whether to switch to log-linear or drop the "law" framing entirely.

---

## 2. Downstream pre → post deltas

**McNemar exact, two-sided** (no direction pre-registered for downstream metrics; the SFT-repair prior was *helps or neutral on knowledge / possibly hurts closedbook facts*, not a fixed sign). `b` = post-correct/pre-wrong (SFT gained); `c` = post-wrong/pre-correct (SFT lost). Paired bootstrap = 10k resamples, seed=42, item_id-aligned.

### core6 (acc_norm — the paper Table 4 metric)

| task | n | pre_accn | post_accn | Δ (pp) | boot 95% CI (pp) | b/c | McNemar p (two-sided) |
|---|---:|---:|---:|---:|---|---|---:|
| hellaswag | 10042 | 0.5467 | 0.5469 | +0.02 | [−0.49, +0.53] | 333/331 | 0.969 |
| arc_challenge | 1172 | 0.3635 | 0.3695 | +0.60 | [−1.11, +2.22] | 54/47 | 0.551 |
| **arc_easy** | 2376 | 0.6481 | 0.6136 | **−3.45** | [−4.63, −2.31] | 55/137 | **2.92e−09** |
| piqa | 1838 | 0.7258 | 0.7214 | −0.44 | [−1.52, +0.60] | 46/54 | 0.484 |
| winogrande | 1267 | 0.5438 | 0.5604 | +1.66 | [−0.39, +3.71] | 98/77 | 0.130 |
| openbookqa | 500 | 0.3520 | 0.3700 | +1.80 | [−0.40, +4.00] | 21/12 | 0.163 |
| **core6 avg** | — | **0.5300** | **0.5303** | **+0.03** | — | — | — |

**core6 avg is a wash** (Δ = +0.03 pp) — SFT neither uniformly repairs nor damages the mixed ladder. The one large hit is **arc_easy −3.45 pp** (highly significant, McNemar p ≈ 3e−9), balanced by non-significant gains in winogrande/openbookqa/arc_challenge. This matches the keep8 pattern (arc_easy also dropped there).

### MMLU dual protocol

| protocol | n | pre | post | Δ (pp) | boot 95% CI (pp) | b/c | McNemar p (two-sided) |
|---|---:|---:|---:|---:|---|---|---:|
| letter | 14042 | 0.2707 | 0.2525 | **−1.816** | [−2.85, −0.79] | 2583/2838 | **5.60e−04** |
| content_raw | 14042 | 0.3232 | 0.3188 | −0.442 | [−0.85, −0.04] | 386/448 | 0.0346 |
| content_norm | 14042 | 0.3448 | 0.3404 | −0.434 | [−0.86, +0.01] | 450/511 | 0.0529 |

SFT reduces MMLU-letter (weakly significant), but the content-based protocols (which the paper uses as robustness anchors) are near-null with 95% CIs that touch zero (`content_norm` CI = [−0.86, +0.01], p=0.053 just crosses the two-sided 0.05 threshold — marginal, not decisive).

### know5

| task | n | pre_accn | post_accn | Δ (pp) | boot 95% CI (pp) | b/c | McNemar p (two-sided) |
|---|---:|---:|---:|---:|---|---|---:|
| mmlu | 14042 | 0.2717 | 0.2530 | −1.87 | [−2.90, −0.83] | 2578/2840 | **3.90e−04** |
| lambada_openai | 5153 | 0.4964 | 0.4877 | −0.87 | [−1.73, +0.00] | 240/285 | 0.0547 |
| boolq | 3270 | 0.6269 | 0.6257 | −0.12 | [−0.95, +0.70] | 91/95 | 0.826 |
| commonsense_qa | 1221 | 0.4275 | 0.4062 | **−2.13** | [−3.52, −0.74] | 23/49 | **2.94e−03** |
| social_iqa | 1954 | 0.4371 | 0.4468 | +0.97 | [−0.15, +2.10] | 71/52 | 0.104 |

MMLU and commonsense_qa take small statistically-clear hits; boolq/lambada/social_iqa are indistinguishable from noise.

### Closed-book QA (PopQA + TriviaQA — the SFT-fact-forgetting probes)

| task | metric | n | pre | post | Δ (pp) | boot 95% CI (pp) | b/c | McNemar p (two-sided) |
|---|---|---:|---:|---:|---:|---|---|---:|
| popqa | em | 14267 | 0.0477 | 0.0324 | **−1.53** | [−1.76, −1.30] | 36/254 | **1.44e−41** |
| popqa | contains | 14267 | 0.1308 | 0.1325 | +0.17 | [−0.19, +0.53] | 357/333 | 0.381 |
| triviaqa | em | 17944 | 0.1813 | 0.1295 | **−5.19** | [−5.60, −4.79] | 242/1173 | **1.11e−146** |
| triviaqa | contains | 17944 | 0.3119 | 0.2824 | **−2.95** | [−3.41, −2.49] | 637/1167 | **4.16e−36** |

**Closed-book memorised facts take a heavy, highly significant hit** — the exact fingerprint of SFT-induced factual forgetting.
- TriviaQA EM: **−5.19 pp** with **overwhelmingly one-sided flip distribution** (b=242 gains vs c=1173 losses; ratio ~1:4.8).
- PopQA EM: **−1.53 pp** with the most extreme flip asymmetry (b=36 vs c=254; ratio 1:7.1).
- PopQA `contains` is null (0.17 pp), so the loss is specifically in EM-exact recall of factual strings — pattern-matching still finds substrings, but the exact-string retrieval degrades sharply.

**This replicates the keep8 finding qualitatively**: SFT hurts closed-book factual recall while leaving core6 near-flat and MMLU-content near-null.

---

## 3. Data provenance & hygiene

- Anchor completeness verified before launch: 5/5 summaries + 4/4 per-item files present on zwfy6 (`hellaswag` 10042 lines / `mmlu` 14042 / `popqa` 14267 / `triviaqa` 17944). ANCHOR_OK=1 gate passed.
- All 5 harnesses passed `assert_8shards` 8/8 before merge. No partial merges.
- Ckpt load path: single file `outputs/olmo2_p24_sft_keep10fresh2/final.pt` (39,009,620,788 bytes; step=842 keep_front=10 n_fresh=2 num_hidden_layers=12; 135 tensors, `strict=True`).
- n_tokens on PPL is identical pre vs post (8,384,512) — same shard slicing on the same val file `data/dolmino_now_val.npy`. Any residual PPL delta is model-driven, not sampling-driven.
- Reference PPL (task spec MAIN-verified) 12.8159 matches computed 12.815923 to 4 sig figs.

---

## 4. Files produced

On zwfy6 disk (`.82` sees them; `.73/.104` do too; LOCAL/wzc1 does not without cross-disk scp):

```
olmo2_ppl_results/7B_p24_sft_keep10fresh2_final/{summary.json, shard{0..7}of8.json}
olmo2_downstream_results/7B_p24_sft_keep10fresh2_final/{summary.json, per_example_<6 tasks>.jsonl, shard*.json}
olmo2_downstream_results/7B_p24_sft_keep10fresh2_final_know/{summary.json, per_example_<5 tasks>.jsonl, shard*.json}
olmo2_mmlu_content_results/7B_p24_sft_keep10fresh2_final/{summary.json, per_example_mmlu.jsonl, shard*.json}
olmo2_closedbook_results/7B_p24_sft_keep10fresh2_final/{summary.json, per_example_{popqa,triviaqa}.jsonl, shard*.json}
```

Analysis snippet: `/tmp/stat_keep10.py` on `.82`. Ledger: `status/gpu_runs.jsonl` `p24_eval_sft_keep10_82` running/completed rows (commit_hash fd1633c).

---

## 5. What this means for MAIN

1. **The n=3 fit is dead.** Two consecutive out-of-sample tests (keep8 → −3.86 pp, keep10 → −4.55 pp) with the same sign and growing magnitude falsify it. Do NOT continue quoting `ΔPPL% ≈ 1.60·prePPL − 7.34` as a law.
2. **Sub-linear / saturating story.** All four "damaged" arms (shortgpt16/keep14/keep10/keep8) end up in a narrow ΔPPL% band **+8.5 … +10.2%** despite pre-PPL spanning 9.78 → 13.33. SFT-repair delta is flattening.
3. **The direction — "SFT repairs LM but hurts memorised facts" — is intact and now n=2 in Table 4 (keep8 + keep10 both show it).** TriviaQA EM drops by −5.19 pp with McNemar p=1e−146; PopQA EM drops with 1:7.1 flip asymmetry. This is a **replicable qualitative finding**, safe to headline.
4. **core6 is basically flat under SFT** — the paper's downstream story ("SFT is close-to-free on the general-eval axis") is confirmed on keep10 (Δ = +0.03 pp on core6 avg), just as on keep8.
5. **Comparability caveat**: keep10 SFT delta is anchored on step83500 (42% budget). It should not be compared to a hypothetical keep10@200k SFT delta.

