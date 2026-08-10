# Paper B — 200k THREE-ARM comparison (all evals DONE 2026-07-28)

Base-only protocol, OLMo-2-7B, identical Dolmino heal (200k steps, eff_bs 128,
seq 2048). All three arms are 16-layer models (keep14+2fresh / freeze-front14+2fresh / random-front14+2fresh).

## Headline: perplexity–knowledge dissociation across the three inheritance/adaptation arms

| arm | PPL | PPL tax | MMLU | MMLU above-chance recovery |
|---|---:|---:|---:|---:|
| base full (32L) | 7.398 | 1.000× | .6053 | 100% |
| **ShortGPT-16 policy** (16 inherited, non-contiguous [0-12,16,17,31], 0 fresh) | **9.780** | **1.322×** | **.4739** | **63.0%** |
| **keep14 train-all** | **10.561** | 1.428× | **.3191** | **19.4%** |
| **freeze-front** (frozen inherited) | **12.797** | 1.730× | **.2628** | **3.6%** |
| **random-front** (no inheritance) | 11.498 | 1.554× | .2461 | ~0% (chance) |

MMLU recovery = (arm − 0.25) / (0.6053 − 0.25).

**★ ShortGPT-policy dominates all three contiguous-truncation arms on BOTH axes** (best PPL tax 1.322× AND best MMLU recovery 63.0%; step200000, same 200k heal, same base protocol chat=False/no-BOS/LL-MC, `datasets 5.0.0` — verified by MAIN from JSON). Two confounded reasons, both pointing the same way: (a) it inherits **16 pretrained layers vs keep14's 14** (+2 more real blocks), and (b) crucially it **retains the pretrained final layer 31 and its native readout** rather than discarding the top and bolting on 2 randomly-initialized fresh layers. Interpretation for Paper B: *which* layers you keep (influence-ranked, keep the readout layer) matters far more than the contiguous-front assumption of the depth ladder — non-contiguous keeps recover >3× the knowledge at lower PPL tax. This is a headline-strength cross-policy result; ⚠️ it is NOT a same-inherited-layer-count comparison (16 vs 14 inherited), so the two effects (layer count vs selection policy) are not yet disentangled — a keep16-inherited/0-fresh contiguous control would isolate policy. Full core6/know5 breakdown below in §ShortGPT.

## What the three arms establish

1. **Inherited blocks must be *adapted*, not merely *reused*.**
   freeze-front (inherited front frozen, only fresh+head trained) reaches only
   3.6% MMLU recovery — barely above the random-front chance floor — versus
   keep14 train-all's 19.5%. Simply *having* pretrained decoder blocks in the
   stack is not enough; they must be updated during healing for their stored
   knowledge to become readable through the fresh tail.

2. **PPL orders oppositely to what "inheritance is always good" would predict.**
   freeze-front PPL (12.797) is the *worst* — worse than random-front (11.498),
   which fully retrains a randomly-initialized front. A frozen inherited front
   cannot adapt to the pruned topology, so it actively hurts next-token fit
   relative to learning a fresh front. Inheritance helps only when it is trainable.

3. **A sliver of knowledge survives without adaptation.**
   freeze-front MMLU (.2628) is 1.7 pp above random-front (.2461), a small but
   real edge showing the frozen pretrained representations carry *some*
   directly-readable signal. But this is dwarfed by the +5.6 pp gain from
   adapting those same blocks (keep14 .3191). The bulk of stored knowledge is
   *adaptation-gated*, not *presence-gated*.

4. **The dissociation is robust across all three arms.** PPL and MMLU do not
   track each other: keep14 has the best PPL *and* the best MMLU; freeze-front
   has the worst PPL but only the second-worst MMLU; random-front is middle-PPL
   but chance-MMLU. No arm recovers MMLU proportionally to its PPL recovery.

## Provenance
- keep14@200k: `status/PAPERB_KEEP14_200K_EVAL.md` (PPL `olmo2_ppl_results/7B_keep14_step200000/`, core6+know5 `olmo2_downstream_results/7B_keep14_step200000*`)
- freeze-front@200k: PPL `olmo2_ppl_results/7B_freezefront_step200000/summary.json` (ppl=12.797); core6 `olmo2_downstream_results/7B_freezefront_step200000/` (HS .595, WG .646); know5 `olmo2_downstream_results/7B_freezefront_step200000_know/` (MMLU .2628)
- random-front@200k: prior eval (PAPER_B_DATA.md §8)
- driver: `scripts/_run_olmo2_eval_freezefront_s200000.sh`; chain log `logs/freezefront_s200000_eval.out` (DONE 21:37:09)

## full-32L continued-pretrain CONTROL (#100 / P1.1) — plateau endpoint DONE (2026-08-02)

The full-32-layer continued-pretraining control (no pruning, `keep_front_layers=32 n_fresh_layers=0`,
identical Dolmino recipe/base-protocol) **reached a training plateau**: held-out ppl locked in 8.1–8.4
for 17k+ steps (step10k=8.19 / 20k=8.11 / 27.7k=8.41, no downward trend). Per user directive "到平台期就可以了",
stopped at step27740 (SIGTERM, clean 8-rank exit) and `step25000.pt` recorded as the **200k-equivalent endpoint**.

**step25000 base-protocol eval (chat=False / no-BOS / LL-MC; MAIN-verified from JSON):**
- held-out dolmino_now_val PPL = **7.6699** (avg_nll 2.0373, 8-shard merge, n_tokens 8.38M)
- **core6 = .6968** — hellaswag .785 · arc_c .5512 · arc_e .8245 · piqa .8085 (acc_norm) · winogrande .7419 · openbookqa .47
- **aux5 = .6536** — mmlu **.5867** · lambada_openai .7145 · boolq .8165 · commonsense_qa .6552 · social_iqa .4949 (acc)
- MMLU above-chance recovery vs base full (32L, healed) .6053 → **94.8%** ((.5867−.25)/(.6053−.25)).
- Provenance: `olmo2_ppl_results/7B_full32_step25000/summary.json` + `olmo2_downstream_results/7B_full32_step25000{,_know}/summary.json`. Driver `scripts/_run_olmo2_eval_full32_plateau.sh`, chain log `logs/full32_plateau_eval.out` (ALL DONE 17:24:07). Env `.venv` (LOCAL 8×B200 wzc1).
- **Interpretation**: continued-pretraining the intact 32L stack on the same 15B-token Dolmino mix recovers to ppl 7.67 / MMLU .5867 — essentially back to the healthy base full-32L reference (7.398 / .6053), i.e. the Dolmino heal itself is near-lossless for capability when NO layers are pruned. This is the clean upper anchor for the depth-ladder / ShortGPT policy comparison: any capability gap in the pruned arms is attributable to pruning+re-heal, not to the continued-pretraining data or recipe.


## ShortGPT external baseline (#98) — step200000 DONE (2026-08-02)

ShortGPT-influence layer selection keeps 16 non-contiguous layers [0-12,16,17,31]
(NO fresh tail; `keep_front_layers=16 n_fresh_layers=0`; ckpt saved compacted to
positions 0..15), same 200k Dolmino heal recipe as the ladder. Heal COMPLETE
(`logs/olmo2_7B_shortgpt16.log` last line step200000 loss 2.1785 ppl 8.83).

**step200000 base-protocol eval (chat=False / no-BOS / LL-MC / datasets 5.0.0; MAIN-verified from JSON):**
- held-out dolmino_now_val PPL = **9.7803** (avg_nll 2.2803, 8-shard merge)
- **core6 = .6215** — hellaswag .6851 · arc_c .4761 · arc_e .7462 · piqa .7584 · openbookqa .408 (acc_norm) · winogrande .6551 (acc)
- **know5 = .5596** — mmlu .4739 · lambada_openai .6194 · boolq .7287 · commonsense_qa .5340 · social_iqa .4422 (acc)
  - **P0.7 审计（2026-08-02）**：此 `know5`（acc-based）= 论文口径 **`aux5_raw`**（异质任务描述性均值，**禁称 knowledge recovery**；MMLU 单列）。`.5596` 经 raw-JSON 核对数值正确，仅改名。详见 `paperB/P0_7_AGGREGATE_AUDIT.md`。
- Provenance: `.82:olmo2_ppl_results/7B_shortgpt16_step200000/summary.json` + `.82:olmo2_downstream_results/7B_shortgpt16_step200000{,_know}/summary.json`. Driver `scripts/_run_shortgpt_downstream_only.sh` (PPL earlier via `_run_olmo2_eval_shortgpt.sh`). Env: `.82:$DB/olmo2_venv` (elsa torch2.7 + transformers 5.5.4 + datasets 5.0.0, built on diskB to survive .82 conda resets).

See headline table above — this endpoint **dominates all three 16L contiguous arms on both PPL and MMLU**. The finding answers the "Next" question: the perplexity–knowledge dissociation is NOT universal across pruning policies — a policy that keeps influence-ranked layers *including the readout layer* recovers far more knowledge (63% vs 19%) at lower PPL tax.

## Remaining ShortGPT trajectory points (deferred)
step0 already evaled (`7B_shortgpt_step0`, ppl~401 degenerate). Midpoints
128000/153500 would need a fresh 46GB scp LOCAL→.82 + eval each; deferred pending
whether the paper needs a ShortGPT healing trajectory vs just the headline endpoint.
