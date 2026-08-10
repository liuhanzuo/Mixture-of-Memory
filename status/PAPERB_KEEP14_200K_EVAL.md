# Paper B — keep14 train-all @ step200000 FULL eval (P0 #1, DONE 2026-07-28)

**Node**: B200 .252 (wzc1), chained after 32B LoCoMo. ckpt `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt` (48.7GB, train-all, DONE at 200k on 07-21).
**Protocol**: base-only, identical harness to the 153.5k eval (PPL raw 2048-tok NTP windows; downstream likelihood-MC, acc_norm). 8-GPU sharded, .venv torch2.13.

## Results — keep14@200k vs 153.5k vs base

| metric | base full | keep14@128k (apex) | keep14@153.5k | **keep14@200k** | 153.5k→200k Δ |
|---|---:|---:|---:|---:|---:|
| held-out PPL | 7.398 | 10.827 | 10.693 | **10.561** | −0.132 |
| PPL tax | 1.000× | 1.463× | 1.446× | **1.428×** | (still slowly improving) |
| MMLU | .6053 | .3012 | .3124 | **.3191** | +0.67pp |
| MMLU above-chance recovery | — | 14.4% | 17.6% | **19.5%** | +1.9pp |
| HellaSwag | .805 | .631 | .643 | .6446 | ~flat |
| ARC-Challenge | .571 | .426 | .442 | .4377 | ~flat |
| ARC-Easy | .829 | .702 | .705 | .705 | flat |
| PIQA | .811 | .747 | .745 | .7454 | flat |
| WinoGrande | .744 | .630 | .633 | .6259 | −0.7pp |
| OpenBookQA | .462 | .402 | .406 | .404 | flat |
| LAMBADA | .732 | .575 | .570 | .5773 | +0.7pp |
| BoolQ | .815 | .639 | .606 | .6887 | +8.3pp |
| CommonsenseQA | .665 | .505 | .506 | .4758 | −3.0pp |
| SocialIQA | .502 | .423 | .441 | .4744 | +3.3pp |

## Key takeaways (load-bearing for the paper)

1. **Perplexity–knowledge dissociation holds at 200k.** Another 46.5k steps past 153.5k
   drops PPL 10.693→10.561 (tax 1.446×→1.428×) but MMLU only .3124→.3191 (+0.67pp).
   The LM objective keeps improving; stored knowledge barely moves. This *strengthens*
   the headline — the dissociation is not a transient of under-healing at 153.5k.
2. **200k is a clean plateau point for the main arm.** PPL tax 1.428× at half decoder
   depth; MMLU recovery 19.5% of base above-chance. Use 200k as the canonical keep14
   number; 128k/153.5k become the trajectory.
3. **Most non-knowledge tasks are flat 153.5k→200k** (HS/ARC/PIQA/OBQA/LAMBADA within
   ±1pp). BoolQ +8pp and CSQA −3pp are the noisier in-context/tasks; WinoGrande −0.7pp.
   No broad rollback — post-apex healing is benign on language tasks, just not on
   closed-book knowledge.

## Provenance
- PPL: `olmo2_ppl_results/7B_keep14_step200000/summary.json` (ppl=10.561, n_tok=8384512)
- core6: `olmo2_downstream_results/7B_keep14_step200000/summary.json`
- know5: `olmo2_downstream_results/7B_keep14_step200000_know/summary.json`
- driver: `scripts/_run_olmo2_eval_keep14_s200000_b200.sh`
- chain log: `logs/keep14_s200000_chain.out` (DONE 14:20:54)

## Next (P0 #2)
freeze_front arm still training (LOCAL, ~92% at this eval's completion). Once it hits
200k (~3-4h), run the same 3-component eval → completes the 200k three-arm comparison
(inherited+train-all / inherited+freeze-front / random-init+train-all).
