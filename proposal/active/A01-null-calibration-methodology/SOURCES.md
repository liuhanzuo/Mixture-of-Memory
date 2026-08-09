# Sources

## Canonical evidence now stored in this proposal

- `evidence/P1_four_constructs.md`
- `evidence/C5_self_falsification.md`
- `evidence/mmlu_interface_initial_dossier.md`
- `evidence/null_calibration_p1_nperm2000.json`
- `evidence/null_calibration_obs4_nperm2000.json`
- `evidence/a01_gate1_third_family.json` — also the source for the **per-family**
  `longest_option_split_tie_null` (Llama-2 0.2757 / Qwen3 0.2833 / Llama-3 0.2847 /
  OLMo-2 0.2845)
- `evidence/gate4_c4_prereg.json`
- `evidence/gate3_dtype_runs/*_dtype_summary.json` — six OLMo-2-7B fp32-vs-bf16 summaries
  (copied from zwfy6; the per-item records stay on zwfy6, see below)
- `evidence/gate3_content_null_conventions.json` / `.csv` — content interface × five
  longest-option null conventions, 120 rows (added 2026-08-09)

## Regenerators

- `code/build_null_calibration_table.py` — the four-construct table
- `code/a01_gate1_verdict.py` — gate-1 intact-family verdict
- `code/summarise_gate1_depth.py` — gate-1 depth curves
- `code/a01_gate3_fp32_vs_bf16.py` — gate-3 dtype contrast harness (GPU)
- `code/a01_gate3_tie_baseline.py`
- `code/a01_gate3_content_conventions.py` — gate-3b convention table (CPU only)
- `code/a01_gate4_c4_prereg.py` — gate-4 C4 aggregation pre-registration

## Shared representation evidence

- `../../shared/representation/R4_repr_alignment_finding.md`
- `../../shared/representation/repr_alignment_results.json`
- `../../shared/representation/cka_matrices/`
- `../../shared/representation/functional_transfer/`

## External raw inputs retained in the main project

### On wzc1 (LOCAL / .21)

- `olmo2_mmlu_content_results/*/per_example_mmlu.jsonl`
- `olmo2_mmlu_content_results/gate1_dmg_llama2_7b_*/summary.json` (Llama-2 depth curve)
- `olmo2_mmlu_content_results/gate1_dmg_{llama2_7b,llama3_8b,qwen3_8b}_k{8,12}/summary.json`
  (the 6/6 damaged-arm result)
- `olmo2_mmlu_content_results/a01_*_intact_base_*/summary.json` (intact cross-scale grid)
- `status/scout_21/lane2_a01_gate2.md` — **gate-2's only writeup; wzc1 ONLY, not on zwfy6**
- `paperC_squad_results/`
- `data/squad_val.jsonl`
- `results/p1_2/p1_2_summary.json`

### On zwfy6 (.73 / .82 / .104) — NOT on wzc1

- `results/a01_gate3/dtype_runs/*_dtype/per_example_dtype_shard{0..7}of8.jsonl` —
  per-item fp32/bf16 logits for the six arms; the inputs to
  `code/a01_gate3_content_conventions.py`
- `olmo2_mmlu_content_results/gate1_dmg_{qwen3_8b,llama3_8b,olmo2_7b}_depth*_k*/summary.json` —
  the depth curves for three of four families (Llama-2's are on wzc1)

> ⚠️ Two-disk rule: a file is not missing until it has been looked for on **both**
> `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory` and
> `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory`.

## Prior-art / novelty

- `NOVELTY_CHECK.md` — per-candidate citation, **verified** venue + the authority that
  verified it, overlap, gap, preemption call. Kill clause 3 verdict.


