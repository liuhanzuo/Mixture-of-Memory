# core6 is not hardware-portable: 28 items flip between L20A and H20 on bit-identical weights

> **⚠️ PARTIAL REVISION (2026-08-08 ~08:2x CST): the "28 flips = cross-arch" number here was
> against a v1 (old-harness) H20 measurement. With MATCHED new-harness on both sides, the
> corresponding rung has been re-measured; refer to `status/PAPERB_HARNESS_DRIFT_REVISION.md`
> for the honest cross-arch flip counts (ShortGPT-16: 7, keep10: 23, keep8: 29, with
> winogrande dominating). The cross-arch signal is real but modest and does NOT scale
> monotonically with damage. Left visible rather than silently edited.**


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

---

# REPLICATION on a second checkpoint (MAIN, 2026-08-08 ~02:1x CST)

The `.252` eval agent's leg-1 result gives an **independent second instance**, on the *vanilla base*
model rather than a healed pruned one. Same HF checkpoint
(`models/OLMo-2-1124-7B`), same harness, `chat=False`, `add_bos=False`, `mode=base`,
`num_hidden_layers=32`.

| task | L20A cc10.0 (wzc1) | H20 cc9.0 (zwfy6) | net flip | n |
|---|---:|---:|---:|---:|
| hellaswag | .80482 (8082) | .80522 (8086) | −4 | 10042 |
| arc_challenge | .57253 (671) | .57082 (669) | +2 | 1172 |
| arc_easy | .82828 (1968) | .82912 (1970) | −2 | 2376 |
| piqa | .81066 (1490) | .81066 (1490) | 0 | 1838 |
| openbookqa | .46200 (231) | .46200 (231) | 0 | 500 |
| winogrande | .74586 (945) | .74428 (943) | +2 | 1267 |
| **core6 avg** | **.70402** | **.70368** | **+0.034 pp** | |

Sources: `wzc1:olmo2_downstream_results/7B_full32_base_wzc1/`, `zwfy6:olmo2_downstream_results/7B_base_full/`.

## Paper provenance, now pinned

> ### ⛔ CORRECTION (MAIN, ~04:3x CST) — the paragraph below originally said the wrong thing
>
> The text as first written claimed the keep14 row was "**also H20 — `zwfy6` anchor**" and that base
> and keep14 were therefore "*consistent with each other* in architecture… the good case."
> **That was false, and it contradicted this document's own earlier section.** Caught by the synthesis
> agent (`a0dca402`) and re-verified by MAIN:
>
> | Table 4 row | paper value | matching source | arch |
> |---|---|---|---|
> | base full-32L | `.7037` | `zwfy6:7B_base_full` = `.70368` | **H20 cc9.0** |
> | keep14 | `.5938` | `wzc1:7B_keep14_step200000` = `.59376` | **L20A cc10.0** |
>
> The zwfy6 keep14 measurement is `.59532`, which rounds to `.5953` — **not** the paper's `.5938`.
> So the two rows come from **different architectures**, which is the *bad* case, not the good one.
> This is the same conclusion MAIN reached at ~01:2x CST; the 02:1x addendum then restated it
> backwards. The earlier finding stands; this paragraph was the error.

**Table 4's base row `0.7037` is the H20 number** (`zwfy6:7B_base_full` = `.70368`); the wzc1/L20A
measurement of the same checkpoint is `.70402`. **Table 4's keep14 row `0.5938` is the L20A number**
(`wzc1` = `.59376`); the H20 measurement of the same weights is `.59532`. The two rows are therefore
**measured on different GPU architectures**, and the full per-rung attribution is in
`status/PAPERB_TABLE4_ARCH_AUDIT.md` (task #189): base/keep10/keep12/ShortGPT-16 = H20,
keep8/keep14 = L20A. Six rows, two architectures, mixed.

PPL is unaffected: full32 pre-SFT PPL reproduced as **7.398071** against the paper's **7.398** — exact.
Consistent with the mechanism: summed-NLL averages the bf16 jitter, whereas core6 thresholds it
through an argmax over options, so only core6 flips.

## A new observation: flip count scales with how damaged the model is

| checkpoint | net flips | core6 delta |
|---|---:|---:|
| full-32L vanilla base (undamaged) | **10** | +0.034 pp |
| keep14+fresh2 @200k (pruned, healed) | **28** | +0.156 pp |

The healed pruned model is **~3x more numerically fragile** than the intact base under the same
hardware swap. This is mechanistically plausible — a 16L shell healed to a 1.43x PPL tax sits closer
to its decision boundaries, so more items are within bf16 reordering distance of flipping — but with
n=2 checkpoints it is an **observation, not a result**. Do not put it in the paper as a claim.

If it holds across the ladder it is worth a sentence, because it means **the cross-architecture floor is
not a constant** and the more-pruned rungs carry the larger instrument noise — precisely the rungs the
paper's headline depends on. The Table 4 audit (#189) can test this for free: it is already reading
every rung's per-task counts, and the keep8/keep10/keep12 rungs would extend this table to n=5.
