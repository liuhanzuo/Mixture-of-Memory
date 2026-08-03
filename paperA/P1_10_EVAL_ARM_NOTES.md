# P1.10 eval arm — BBWL (trained deployable Write) in the P0.18 E4 2x2 harness

**Added:** 2026-08-04. One new arm `BBWL` in `scripts/eval_p018_e4_2x2_writecontrol.py`.
Pure increment — gated behind `--write_lora_ckpt`; without that flag the harness is
bit-identical to before.

## What BBWL is
`BBWL` = Arm **BB** (chunk-local Write, local-pos Read, the *deployable* 92.5 config)
run VERBATIM through `p017._run_arm`, but with the P1.10-trained **WRITE LoRA**
enabled on layers `0..resume_j-1` (=0..11) during the write/read/decode. It quantifies
how much of the BB (92.5) → E0 (document-contextual Write, 100) gap a *trained
chunk-local* Write recovers — WITHOUT giving the lower band the whole document.

- Factor-1 = `chunk_local_trained_write`, factor-2 = `local_pos` (BB's deployable Read).
- Contrast of interest: `diff_BBWL_minus_BB` (trained-write gain) and
  `diff_E0_minus_BBWL` (residual gap to the non-deployable doc-contextual Write).

## Why A / BB / E0 / X / Y stay bit-identical
Two-adapter design over **disjoint** layer sets:
- flagship **READ** LoRA lives on layers `12..35`, loaded as the LIVE peft adapter
  `"default"` (exactly as `_load` does; never merged);
- trained **WRITE** LoRA lives on layers `0..11`, loaded as a SECOND adapter `"write"`.

Every layer holds exactly ONE adapter in its `lora` ModuleDict (12..35→only `default`,
0..11→only `write`). Post-load active set is `"default"`, so each layer-0..11
`lora.Linear` falls through to `base_layer(x)` == the original `nn.Linear` →
BIT-IDENTICAL to a load that never saw the write adapter. A/BB/E0/X/Y all run with
active==`"default"`. The WRITE LoRA fires ONLY inside the `_write_lora_enabled`
context (`set_adapter(["default","write"])`) that wraps the BBWL arm run and its h12
state metric; the `finally` restores `set_adapter("default")`. Because BBWL reuses the
SAME live-READ realization as BB, `BBWL vs BB` is a clean paired comparison (only the
lower-band write differs).

This reproduces the P1.10 trainer's student forward: there the READ LoRA was merged
into the base and the WRITE LoRA sat on 0..11; since layers 0..11 never carried READ
either way, the lower-band write is identical here.

## Fail-closed guards
`_load_with_write_lora` aborts unless the WRITE adapter's `layers_to_transform`
== `[0..resume_j-1]` and is disjoint from the READ layers (12..35). `run_manifest`
expects `EXPECTED_LORA_MODULE_COUNT + resume_j*7` (=168+84=252) LoRA modules when the
write ckpt is supplied, verifies WRITE layers + disjointness, and records
`write_lora_sha256` / `write_lora_layers` / `lora_adapter_names` in `strict_fixes`.

## Eval conventions (unchanged)
chat_template=False (base LM), selector `iter_bm25` (hardcoded), enable_thinking=False,
add_bos=0 — inherited verbatim from the existing harness.

## Run (per-ckpt; step500..step4000)
Run on the node that produced the write ckpt (the P1.10 node — see
`paperA/P1_10_WRITEPATH_NOTES.md` §Run; `.104`, diskB, torch-base) or any node sharing
its ceph. Set `PROJECT_ROOT` to that node's project root (same one the P1.10 launcher
used) so `--write_lora_ckpt outputs/qcmem_writepath_distill_qwen_j12_r32/stepN` resolves.
```
PROJECT_ROOT=<diskB project root that holds outputs/qcmem_writepath_distill_qwen_j12_r32/> \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python
cd $PROJECT_ROOT
# manifest (strict-fix gate; verifies READ+WRITE provenance) — GPU:
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/eval_p018_e4_2x2_writecontrol.py \
  --mode manifest --output_dir bench_results/p0_18_e4_bbwl_step500 \
  --write_lora_ckpt outputs/qcmem_writepath_distill_qwen_j12_r32/step500
# quality (one (task,length) per call; shard across GPUs as usual) — GPU:
CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN scripts/eval_p018_e4_2x2_writecontrol.py \
  --mode quality --task niah_multikey_1 --length 8k --limit 100 \
  --output_dir bench_results/p0_18_e4_bbwl_step500 \
  --write_lora_ckpt outputs/qcmem_writepath_distill_qwen_j12_r32/step500
# aggregate (CPU): paired CI + McNemar; adds BBWL macro + diff_BBWL_minus_BB.
$PYTHON_BIN scripts/eval_p018_e4_2x2_writecontrol.py \
  --mode aggregate --output_dir bench_results/p0_18_e4_bbwl_step500
```
Sweep `step500 step1000 … step4000` (own `--output_dir` each) to trace the trained-Write
accuracy vs training steps.
