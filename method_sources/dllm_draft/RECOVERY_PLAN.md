# SFT Recovery Experiment

This study isolates why the matched SFT checkpoints underperform the official
Dream-Coder Instruct checkpoint.

## Controlled arms

| Label | Initialization | Epochs | LR | Mask distribution |
|---|---|---:|---:|---|
| `base_raw` | Dream-Coder Base | 0 | — | — |
| `base_plain1` | Dream-Coder Base | 1 | 2e-6 | uniform |
| `base_plain5` | Dream-Coder Base | 5 | 1e-5 | uniform (completed control) |
| `instruct_raw` | Dream-Coder Instruct | 0 | — | — |
| `instruct_plain1` | Dream-Coder Instruct | 1 | 2e-6 | uniform |
| `instruct_highnoise1` | Dream-Coder Instruct | 1 | 2e-6 | 20% all-mask + 30% t∈[0.8,1] + 50% uniform |

All new SFT arms use the same 114,363 examples, global batch 128,
length bucketing, micro-batch 16/GPU, and one epoch.

A 5,000-state deterministic audit of the high-noise collator measured:

```text
exact all-mask rate       19.10%
t >= 0.8 rate             60.46%
mean t                    0.7232
observed mask fraction    0.7227
```

This matches the intended 20% all-mask + 30% high-noise + 50% uniform mixture
(the uniform component itself contributes about 10% more states above 0.8).

## Questions answered

1. `base_plain1 - base_raw`: does any matched SFT improve the exact Base
   checkpoint?
2. `base_plain5 - base_plain1`: did five epochs / LR 1e-5 over-train or forget?
3. `instruct_plain1 - instruct_raw`: does narrow SFT damage an already aligned
   Instruct model even at low LR and one epoch?
4. `instruct_highnoise1 - instruct_plain1`: does direct prompt-only/high-noise
   supervision improve full all-mask generation?

## Evaluation and routing

Every arm uses the same 512-NFE HumanEval harness. The final report includes
aggregate pass/parse, paired bootstrap intervals, exact McNemar tests, and
controlled deltas. MBPP is intentionally not used for selecting the recovery
recipe; it can be run only after HumanEval identifies a promising arm.
