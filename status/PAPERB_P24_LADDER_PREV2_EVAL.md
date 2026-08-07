# PAPERB P2.4 Ladder pre-SFT re-eval on `.73` (task #189 flip-count audit, n=5)

**Started**: 2026-08-08 02:28:05 CST
**Node**: `.73` (28.85.35.73:36000), 8×H20 cc9.0, zwfy6 disk, `/opt/conda/envs/torch-base/bin/python` torch 2.13.0
**Driver PID**: `2871872` (setsid nohup, disowned)
**Log**: `logs/p24_eval_ladder_prev2_73.log`
**Driver**: `scripts/_run_olmo2_p24_eval_ladder_prev2_73.sh` md5=`7e56545268874b180ef682e7e473734a` (byte-identical wzc1↔zwfy6)
**ETA**: ~4-6 h → 2026-08-08 06:30-08:30 CST
**Commit**: (driver + this report + ledger, hash filled after commit)

## Purpose

Extend the L20A (cc10.0, wzc1) vs H20 (cc9.0, zwfy6) **flip-count-scales-with-pruning-damage** observation from **n=2** (full32 base = 10 net flips / +0.034 pp core6; keep14 @200k = 28 net flips / +0.156 pp) to **n=5** by producing H20 per-item preds for the four remaining Table 4 ladder rungs. Preserves per-item files for downstream McNemar / paired bootstrap.

Sibling `.252` battery running full32 + shortgpt16 pre/post-SFT on L20A/wzc1 in parallel (PID 3032751, started 02:02:07). No overlap on `.73` outputs — see `_v2` naming below.

## Path correction — task text vs. what actually exists

The task specified `step200000.pt` for **all four rungs**. That file exists only for `shortgpt16`. For keep8/keep10/keep12, training never reached step200000 — the headline ckpts per `paperB/P0_7_AGGREGATE_AUDIT.md` §2 (the canonical Table 4 source) are:

| rung | task-text ckpt | actually-exists / audit-headline ckpt | audit `core6` | audit `PPL` |
|---|---|---|---:|---:|
| keep8 | `step200000.pt` (**does not exist**) | `step121000.pt` (11.4 GB) | 0.5238 | 13.333 |
| keep10 | `step200000.pt` (**does not exist**) | `step83500.pt` (39.0 GB) | 0.5303 | 12.816 |
| keep12 | `step200000.pt` (**does not exist**) | `step124000.pt` (43.9 GB) | 0.5669 | 11.443 |
| shortgpt16 | `step200000.pt` ✓ | `step200000.pt` (48.7 GB) | 0.6215 | 9.780 |

I proceeded with the **audit-headline ckpts** because (a) they are what Table 4 actually cites, per P0.7, and (b) they are the only ckpts present on either disk. Output names encode the actual step number (`7B_keep8_step121000_v2` etc.) — this keeps the `_v2` naming diffable against the existing Table-4-source summaries at the same base name. If Table 4 in the paper is actually citing different steps than P0.7 audited, that is a separate defect I cannot detect from this eval alone.

## Ckpts (verified present on zwfy6, `.73`)

| leg | ckpt | size (B) | output_name |
|---|---|---:|---|
| keep8 | `outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt` | 11,384,060,758 | `7B_keep8_step121000_v2` |
| keep10 | `outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt` | 39,009,621,151 | `7B_keep10_step83500_v2` |
| keep12 | `outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt` | 43,867,047,810 | `7B_keep12_step124000_v2` |
| shortgpt16 | `outputs/olmo2_probe2_7B_shortgpt16/step200000.pt` | 48,724,473,978 | `7B_shortgpt16_step200000_v2` |

`arch_meta.json` verified for keep8/10/12 (`arm: healing_frontN+fresh2`, `keep_front_layers=N`, `n_fresh_layers=2`). shortgpt16 has no `arch_meta.json` — pruning params (`keep_front_layers=16 n_fresh_layers=0 keep_layer_indices=[0..12,16,17,31]`) read from ckpt state itself by the eval loader. Verified via direct `torch.load`.

## Harnesses (per leg, mirroring keep14 sibling exactly)

For each rung, five harnesses run serially:
1. **PPL** — held-out Dolmino (`data/dolmino_now_val.npy`), 8-shard × batch=4 → `olmo2_ppl_results/<NAME>/summary.json`
2. **core6** — hellaswag/arc_challenge/arc_easy/piqa/winogrande/openbookqa, `--save_per_example`, 8-shard × batch=8 → `olmo2_downstream_results/<NAME>/summary.json`
3. **know5 (aux5_raw components)** — mmlu/lambada_openai/boolq/commonsense_qa/social_iqa, `--save_per_example` → `olmo2_downstream_results/<NAME>_know/summary.json`
4. **MMLU dual (letter+content)** — `eval_olmo2_mmlu_content.py`, per-item default → `olmo2_mmlu_content_results/<NAME>/summary.json`
5. **closedbook** — PopQA + TriviaQA, per-item default → `olmo2_closedbook_results/<NAME>/summary.json`

Config invariants (identical to keep14 sibling):
- `chat_template=False`, `--add_bos 0` (project memory `paper-eval-chat-false-mandatory`)
- bf16 forward (torch.amp.autocast `dtype=torch.bfloat16` in both PPL and downstream loaders; matches keep14 sibling)
- `--save_per_example` on both downstream calls (per-item load-bearing)
- `LOCAL_RANK=0 RANK=$g CUDA_VISIBLE_DEVICES=$g` per shard
- `assert_8shards` gate before every merge — 8/8 shard files required, else abort merge (silent 5/8 partial-merge corruption is the known failure mode, project memory `kill-remote-gpu-job-by-pid-not-pkill`)

## Early stepping proof (as of 02:31 CST)

- pre-flight: 4/4 ckpts present, pass
- keep8 PPL merged 02:30:23 → **PPL=13.3329** vs audit-quote **13.333**: matches to 1e-4 → zero harness drift
- All 8 downstream shards spawned 02:30:24 for `7B_keep8_step121000_v2` core6
- nvidia-smi 02:31: 8/8 cards @ 17.6-18.5 GiB, 66-98 % util → healthy
- shard0 log confirms `[pruned] loaded ckpt step=121000 keep_front=8 n_fresh=2 num_hidden_layers=10 (113 tensors, strict)`

## Load-bearing diff comparison (to be executed at completion)

For each rung `X`, compute
```
delta_core6(X) = core6_v2_H20(X) − core6_paper_table4(X)
delta_ppl(X)  = ppl_v2_H20(X)   − ppl_paper_table4(X)
```
against the P0.7-audit values in the table above.

**Interpretation matrix**:
- `|delta_core6| ≤ 0.02 pp` on all four: the paper's Table 4 core6 for those rungs was H20 (or was zero-flip) → task #189 audit passes for those rungs.
- `|delta_core6| ≈ 0.16 pp` for some rung: that rung was L20A-scored in Table 4 — **cross-arch contamination**; must relabel or re-run for same-arch consistency.
- `|delta_ppl| ≤ 5e-3`: expected; PPL is largely arch-invariant per `PAPERB_CORE6_CROSSARCH_FLOOR.md` (1.4e-4 seen on keep14; already exact on keep8 by 02:30).
- Any larger PPL deviation on any rung → harness drift or ckpt corruption; report loudly.

The n=5 dataset (10, ?, ?, ?, ? net flips vs pruning depth) will slot into `PAPERB_CORE6_CROSSARCH_FLOOR.md` §"flip count scales with damage" as the promised extension.

## Deliverables at completion

- `olmo2_ppl_results/{name}_v2/summary.json` × 4
- `olmo2_downstream_results/{name}_v2{,_know}/summary.json` × 4 pairs (with `per_example_<task>_shard{0..7}of8.jsonl` + merged `per_example_<task>.jsonl` for every task)
- `olmo2_mmlu_content_results/{name}_v2/summary.json` × 4 (with per-item)
- `olmo2_closedbook_results/{name}_v2/summary.json` × 4 (with per-item)
- Total 20 harness output dirs with per-item preds preserved

## Do NOT touch

- `.252`'s parallel battery outputs (`7B_full32_base_wzc1`, `7B_p24_sft_full32_final`, `7B_shortgpt16_step200000_wzc1`, `7B_p24_sft_shortgpt16_final`)
- Existing `.73` outputs from the keep14 sibling (`7B_keep14_step200000{,_know}`, `7B_p24_sft_keep14fresh2_final{,_know}`)
- Any base-named `7B_keepN_step*/*` summaries — `_v2` is the strict discriminator
- `.tex`, `versions/*.md`, `paperB/TODOList.md`, `paperB/P0_7_AGGREGATE_AUDIT.md`, sibling `PAPERB_P24_*.md` (per task text)

## Follow-up (post-completion, for MAIN)

- Diff `core6_v2` vs `core6_paper_table4` for each rung; report loudly per interpretation matrix
- Extend `PAPERB_CORE6_CROSSARCH_FLOOR.md` flip-count table to n=5 (needs the sibling `.252` shortgpt16 pre-SFT `7B_shortgpt16_step200000_wzc1` to complete the pair for shortgpt16)
- Update GPU_STATUS.md `.73` block to "done" and free the node
