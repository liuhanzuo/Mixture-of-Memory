---
id: B04
status: SURVIVING
promotion_pending: novelty_check_only
updated: 2026-08-08T24:00+0800
complete_rungs: 6
target_rungs: 6
kill_verdict: NOT_KILLED (established at max-possible significance)
---
# B04 eval-fragility — Direction A verdict (n=6, bs16, acc_norm)

## Result

**Primary Spearman (n=6, exact permutation p, 720 perms):**

| Metric | ρ | exact p | expected sign | match |
|---|---|---|---|---|
| median_margin (acc_norm) | **+1.0000** | **0.0028** | + | ✓ |
| frac(margin < 0.001) | −0.9429 | 0.0083 | − | ✓ |
| **frac(margin < 0.005)** ★ PRIMARY | **−1.0000** | **0.0028** | − | ✓ |
| frac(margin < 0.010) | −0.9429 | 0.0167 | − | ✓ |

Both PRIMARY metrics achieve **exact-permutation lower bound p = 1/360 = 0.00278** for n=6 two-sided tests. Result cannot be more significant with this design.

## Rungs, core6, and margin distributions

| Rung (OLMo-2-7B) | core6 acc_norm | median\_margin | frac<.005 | frac<.010 |
|---|---|---|---|---|
| base_full (32L, untouched) | 0.7037 | 0.1318 | 1.756% | 3.722% |
| shortgpt16@200k (structural prune, healed) | 0.6233 | 0.1165 | 2.669% | 5.345% |
| keep14@200k | 0.5949 | 0.1079 | 2.681% | 5.426% |
| keep12@124k | 0.5681 | 0.1005 | 3.228% | 6.054% |
| keep10@83.5k | 0.5308 | 0.0956 | 3.332% | 6.892% |
| keep8@121k | 0.5226 | 0.0949 | 3.553% | 6.816% |

- Perfect rank alignment across all 6 rungs on both fragility metrics (median_margin and frac<0.005).
- shortgpt16 (structural, healed) sits between base_full and keep14 in BOTH core6 and margin distributions — no cherry-picked ordering.
- N (per-item margins pooled across 6 tasks) = 17,195 per rung.

## What this shows

**Statement:** As OLMo-2-7B suffers increasing structural damage (deeper prune or fewer heal steps), its per-item acc_norm margin distribution shifts uniformly toward zero — the healed models make correct choices with less confidence AND get more near-tie items (both directions of the same phenomenon).

**What this is:** an internal-consistency check that "damage" and "measured capability" co-vary in a specific micro-structural way, not just at the aggregate accuracy level.

## What this is NOT

- **NOT** a claim about bs sensitivity or seed sensitivity. Only bs16, single 8-shard determinist run per rung. If bs4/bs8 rungs are added later the flip-rate story can be tested separately.
- **NOT** established beyond OLMo-2-7B. Cross-family replication (Qwen prune-heal ladder) is the next kill test.
- **NOT** a mediation claim ("near-ties cause the aggregate drop") — that would need LOO across items, not addressed here.

## Provenance

- Script: `proposal/backlog/B04-eval-fragility/analyze_b04_5rung.py` (extended in place for n=6 by adding shortgpt16 rung)
- Per-item source: `.73:/apdcephfs_zwfy6/.../olmo2_downstream_results/7B_{base_full,keep14_step200000,keep12_step124000,keep10_step83500,keep8_step121000,shortgpt16_step200000}_bs16/per_example_*.jsonl` (all 8-shard `_shardXof8.jsonl` files merged, integrity asserted via n_scored counts matching HF task cardinalities: hellaswag=10042, arc_challenge=1172, arc_easy=2376, piqa=1838, openbookqa=500, winogrande=1267)
- Output: `evidence/B04_6rung_bs16_analysis.json`
- Runner shell: `scripts/_run_paperF_bs16_ladder_73.sh` (8-shard assertion + per-task n_scored assertion + idempotent skip-if-summary.json guard)

## Promotion status

Per CLAUDE.md "研究方向命名与晋升规则" (2026-08-08):

| Criterion | Status |
|---|---|
| kill gate passed & not killed | ✓ (Spearman +1.00 / −1.00 at p = 0.0028) |
| independent-verified significant finding | ✓ (this replicates prior archived `PAPERF_ACCNORM_REDO` n=6 result on freshly re-run + integrity-audited data) |
| provenance complete & recomputable | ✓ (script + JSON + per-item preds all persist) |
| novelty check | ✗ NOT YET DONE |

→ **Stays as `proposal/backlog/B04-eval-fragility-incubator/` (not paper<X> yet)** until novelty check clears "does the damage-vs-margin Spearman finding already exist in prior work?"

## Next actions

1. **Novelty scan** (CPU, MAIN): "damage × MC margin × per-item near-tie" — check Post-training pruning literature (SparseGPT, ShortGPT), continued-pretraining literature (2506.00288 CMR, 2407.17467 data-mixture), knowledge-decay literature. Estimated 30 min.
2. **Qwen cross-family replication** (GPU on next free H20): rerun the same bs16 downstream harness on Qwen keepN-fresh2 ladder ckpts from #117. If ρ still ≈+1.00 at n≥5, direction A cross-family evidence is complete.
3. **LOO mediation check** (CPU): does `frac<threshold` mediate the `core6` drop? Compute paired-residuals correlation across items. Not required for promotion but strengthens story.

## Known limitations for the writeup

- All rungs on OLMo-2-7B only (single model, single size).
- Aggregate acc_norm from 8-shard merge; bs sensitivity itself not checked (only bs16 data available for keepN rungs).
- Task universe = core6 (H+ARC+PIQA+OBQA+WG). MMLU/knowledge tasks excluded from this analysis; they have separate margin distributions and separate confounds (interface effect studied in Paper E).
