# core6 is not hardware-portable: 28 items flip between L20A and H20 on bit-identical weights

**Date**: 2026-08-08 CST. **Verified by**: MAIN, by re-deriving both sides and hashing the weights.
**GPU cost**: 0 (both eval batteries already existed / were produced by the P2.4 eval agent).

## What happened

Agent `aefe8b20c71af8672`, running the P2.4 keep14 eval on `.73`, found the pre-SFT anchor for PPL and
core6 existed only on **wzc1**, so it re-ran those two on **zwfy6** to get a same-disk pre/post pair.
It reported HellaSwag `0.64390` vs the wzc1 anchor `0.64459` and attributed the `7e-4` gap to
"different shard ordering across a distributed 8-way MC eval."

**That attribution is wrong, and the effect is larger than one task.** Shard order cannot change a
discrete per-item accuracy: each item is scored independently and the merge is a sum. A changed count
means items actually flipped.

## The numbers (same ckpt, same harness, chat=False, add_bos=false, n_nan=0)

| task | metric | wzc1 (L20A cc10.0) | zwfy6 (H20 cc9.0) | net flip | n |
|---|---|---:|---:|---:|---:|
| hellaswag | acc_norm | .64459 (6473) | .64390 (6466) | **+7** | 10042 |
| arc_challenge | acc_norm | .43771 (513) | .44198 (518) | **−5** | 1172 |
| arc_easy | acc_norm | .70497 (1675) | .70286 (1670) | **+5** | 2376 |
| piqa | acc_norm | .74538 (1370) | .74701 (1373) | **−3** | 1838 |
| openbookqa | acc_norm | .40400 (202) | .40400 (202) | 0 | 500 |
| winogrande | acc | .62589 (793) | .63220 (801) | **−8** | 1267 |
| **core6 avg** | | **.59376** | **.59532** | **+0.156 pp** | |

28 net-flipped items. Signs differ per task (wzc1 wins 3, zwfy6 wins 3), i.e. **symmetric noise, not a
directional hardware advantage** — the same structure as the dLLM/AR cross-architecture result.
PPL moves far less: `10.561295` vs `10.561151` (1.4e-4), because summed NLL averages the jitter
instead of thresholding it.

## Ruled out: the two disks holding different weights

This was the serious possibility, and the file sizes *looked* alarming — wzc1 `48,724,473,850` B vs
zwfy6 `16,241,486,089` B, a 3× gap. Explanation: the wzc1 copy carries optimizer state
(weights + Adam m,v ≈ 3 × 4.06 B params × 4 B), the zwfy6 copy is weights-only (`has_optimizer=False`).

**The weights are bit-identical.** `model_state` SHA-256 over all 179 tensors, keys sorted, raw bytes:

```
wzc1 : 069c3e73a75a47c0cf1f0c00ca6d893c601f685ae1dde700e5b82ba9d47caa6c
zwfy6: 069c3e73a75a47c0cf1f0c00ca6d893c601f685ae1dde700e5b82ba9d47caa6c
```

(Per-tensor fp32 sums agree to ~1e-7 relative, consistent with identical bytes; the hash is the
authority.) So the flips are entirely attributable to **GPU architecture** — bf16 kernel/reduction
differences between cc10.0 and cc9.0.

## Why this matters for Paper B

Paper B Table 4's core6 column is `.5938` for keep14 — that is the **wzc1** number. `.59532` is
equally valid, measured on the same weights with the same harness. **core6 has a ~0.16 pp
cross-architecture floor.** Consequences:

1. **Never mix nodes within a comparison.** Table 4's ladder (keep8 `.5238` / keep10 `.5303` /
   keep12 `.5669` / keep14 `.5938`) is only sound if every rung was measured on the same
   architecture. Adjacent rungs differ by 2.7–3.7 pp, so a 0.16 pp floor does not threaten the
   ordering — **but it must be checked, not assumed.** Any rung silently scored on the other disk is
   a defect.
2. **Do not report core6 to 4 decimals** as if exact. Differences below ~0.2 pp are instrument.
3. **P2.4's pre/post pairing must be same-architecture.** The agent's instinct to re-run pre-SFT on
   zwfy6 was therefore *correct* — for a better reason than it gave. Pairing a wzc1 pre against a
   zwfy6 post would have put a 0.16 pp hardware artifact straight into the SFT effect. Per-item
   McNemar across disks would be structurally invalid.
4. This is the **third independent instance** of the same phenomenon today: dLLM HE+ ±2.44 pt, AR HE+
   ∓2.44 pt, and now OLMo-2 core6 ±0.16 pp. Cross-architecture bf16 irreproducibility is generic
   across model family, task type, and metric — not a dLLM property.

## Action items

- [ ] Audit which architecture produced each rung of Table 4's core6 column; re-run any that came
      from the other disk.
- [ ] State the measurement node in the paper's protocol section, and report the 0.16 pp
      cross-architecture floor alongside the seed floor.
- [ ] P2.4 pre/post pairing: use the zwfy6 pre-SFT anchor (agent's PART A) with the zwfy6 post-SFT,
      not the wzc1 anchor. Same for the wzc1-side arms (full32, shortgpt16) — pair within disk.

## Provenance

- wzc1: `olmo2_downstream_results/7B_keep14_step200000/summary.json`, `olmo2_ppl_results/7B_keep14_step200000/summary.json`
- zwfy6: same relative paths under `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`
- ckpt both disks: `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt`
- Related: `status/DLLM_G1_GATE_RESULT.md`, `dllm_draft/SAMPLER_VARIANCE_DECOMPOSITION.md` §3
