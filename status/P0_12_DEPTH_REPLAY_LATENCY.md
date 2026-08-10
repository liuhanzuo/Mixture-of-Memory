# P0.12 — Same-Pack, Same-LoRA Replay-Start Latency Control

## Status

**DONE.** Core timing, strict provenance, component timing, finite-value checks, and natural-text output-consistency checks are complete. This result measures a model-side read-path control; it is not an end-to-end serving or strict equal-quality claim.

## Question

Holding the Qwen3-8B checkpoint, LoRA artifact, packed token IDs, retrieval order, query, hardware, dtype, and attention backend fixed, how does model-side `qc.read()` latency change when replay begins at layer 0 versus layer 12?

The two arms are not the same computation graph. At `j=0`, the packed sequence traverses lower layers with global causal attention; at `j=12`, each chunk is independently written to residual state $h_{12}$ and lower-layer computation is reused. The control therefore isolates the measured consequence of replay start under a fixed adapter and token pack, not bit-identical semantics.

## Authoritative artifacts

- Timing and component breakdown: `bench_results/p0_12_acceptance/`
- Synthetic-pack consistency: `bench_results/p0_12_acceptance/consistency.json`
- Natural-text consistency: `bench_results/p0_12_naturaltext/summary.json`
- Original independently replicated timing cohort: `bench_results/p0_12_depth_replay/`
- Acceptance benchmark: `scripts/bench_p0_12_acceptance.py`
- Natural-text check: `scripts/bench_p0_12_naturaltext_consistency.py`
- Launcher: `scripts/launch_p0_12_acceptance_82.sh`

The older `bench_results/p012/` cohort is superseded and is not used for paper numbers.

## Fixed configuration and provenance

- Hardware: NVIDIA H20, bf16, SDPA
- Backbone: Qwen/Qwen3-8B, 36 layers
- LoRA: identical flagship rank-32 artifact in both arms; SHA-256 recorded in every JSON
- Mounted adapter modules: 168, all enumerated in the acceptance artifacts and restricted to layers 12–35
- Pack: identical 6,657 packed token IDs, selected chunk IDs, ordering, and SHA-256 in both arms
- Repetitions: three independent processes per arm, three warmups and 20 raw timed reads per process
- Environment: Python, PyTorch, CUDA, driver, Transformers, PEFT, GPU, dtype, attention implementation, and git commit recorded in each artifact
- Finite-value audit: recursive check over the 17 current P0.12 JSON artifacts found zero NaN or non-finite values

## Timing result

Median of the three process medians:

| Arm | `qc.read()` | Upper-transformer forward | Final norm + LM head | Peak GPU memory |
|---|---:|---:|---:|---:|
| `j=0` + same LoRA | 1.08085 s | 1015.86 ms | ~59 ms | ~17.66 GB |
| `j=12` + same LoRA | 0.78707 s | 726.66 ms | ~59 ms | ~17.66 GB |

Derived effects:

- Model-side `qc.read()` path: **1.373× speedup**, **27.18% latency reduction**
- Upper-transformer forward: **1.398× speedup**
- Final norm + LM head: effectively unchanged
- All three independent process pairs agree in direction

This result supersedes the earlier rounded `1.374×` cohort and is consistent with it.

## Output-consistency result

Synthetic fixed pack:

- Last-position logit cosine: **0.9784**
- Next-token top-1: identical
- Greedy 16-step agreement: **15/16**
- KL divergence: approximately **0.04–0.08**

Three natural-text packs:

- Last-position logit cosine: **0.9758**
- Next-token top-1: **3/3 identical**
- Query-tail top-1 agreement: **84.8%**
- Greedy 16-step token agreement: **85.4%**
- Mean decode-logit cosine: **0.965**
- KL divergence: approximately **0.13**

These checks support near-equivalent output behavior for the tested packs, not bit identity or benchmark-level quality equivalence.

## Allowed paper claim

> Holding the adapter and packed token IDs fixed, starting model-side replay at layer 12 rather than layer 0 reduces the measured `qc.read()` path from 1.081 to 0.787 seconds on H20, a 1.373× speedup across three independent processes. The reduction arises primarily in upper-transformer forward time (1.398×), while final normalization and LM-head time remain unchanged. On three natural-text packs, both paths preserve the next-token top-1 prediction and reach 0.965 mean decode-logit cosine similarity.

Required caveats:

1. Lower-layer computation is moved to reusable store/query write rather than eliminated.
2. Retrieval, index construction, store/query write, and persistent-store I/O are excluded from `qc.read()`.
3. The two arms enter layer 12 with different hidden-state construction and are not bit-identical.
4. Output-consistency checks do not establish benchmark-level equal quality.
5. This is not an end-to-end serving speedup.
