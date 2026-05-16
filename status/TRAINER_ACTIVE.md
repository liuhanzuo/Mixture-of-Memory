# TRAINER_ACTIVE.md — Active Training Runs
## Updated 2026-05-16 21:30 CST (heartbeat)

## Active Runs

### Local H20 (8× H20, 97.8 GiB) — P11 8B FSDP
- **Experiment**: Phase-1B P11 — Llama-3-8B-Instruct + L1+L3, 8B backbone
- **Status**: RUNNING
- **Script**: `scripts/train_mem_space_babilong.py --use_fsdp --gradient_checkpointing`
- **Output dir**: `outputs/babilong_sft_phase11_fsdp_full/`
- **Log**: `logs/p11_fsdp_full_20260516_181417.log`
- **PID**: 898860 (torchrun launcher)
- **Progress**: ~step 2190/5000 (44%), nf=0, BABI lm_loss ~0.006–1.1, PG19 lm_loss ~2.8–3.5
- **Ckpts saved**: step 500/1000/1500/2000
- **Config**: babilong_tasks=qa1,qa2,qa5; lengths=1k,2k,4k; lr=2e-5; num_slots=512; top_k=64; use_dual_gate; use_l3_summary; pg19_mix=0.2
- **ETA**: ~05:00 CST 2026-05-17
- **Notes**: FSDP needed to fix cuBLAS workspace OOM. Commit a6dcda3 fixes 3 FSDP bugs (scalar params, no top-level wrap, manual ckpt path).

### Remote 28.59.80.196 (8× H20, 97.8 GiB) — 1B v4 L1+L2+L3 FSDP
- **Experiment**: Phase-1B v4 — Llama-3.2-1B-Instruct + L1+L2+L3 full architecture
- **Status**: RUNNING
- **Script**: `scripts/launch_phase1b_v4_l1l2l3.sh full`
- **Output dir**: `outputs/babilong_sft_phase1b_v4_l1l2l3/`
- **Log**: `logs/phase1b_v4_20260516_2049.log`
- **PID**: 234624
- **Progress**: ~step 2010/5000 (40%), nf=0, BABI lm_loss ~0.05–0.9, aux ~10.6
- **Ckpts saved**: step 500/1000/1500/2000
- **Config**: model=Llama-3.2-1B-Instruct; babilong_tasks=qa1,qa2,qa5; lengths=1k,2k,4k; lr=2e-5; use_l2; l2_compress_ratio=16; l2_d_c=512; use_l3_summary; use_fsdp; gradient_checkpointing
- **GPU**: 27-33 GiB/card (~30% of 97.8 GiB)
- **ETA**: ~03:30 CST 2026-05-17
- **Notes**: FSDP + fix for L2Compressor forward() needed (commit 0349264). Baseline comparison: v2 (L1+L3 only, 37.29 mean). This adds L2 to check if token-compressed KV helps.

## Node Status Summary
| Node | GPU | Status |
|------|-----|--------|
| Local H20 | 8× H20 97.8 GiB | P11 8B FSDP RUNNING |
| 28.59.80.196 | 8× H20 97.8 GiB | 1B v4 L1+L2+L3 FSDP RUNNING |
| b200-1..4 (28.89.17.x) | 8× L20A 183 GiB | SSH Connection refused / Permission denied — unavailable |
| b200-5..8 (ephemeral) | 8× L20A 183 GiB | SSH Permission denied — unavailable |
| h20-2..4 (28.85.x / 28.59.x / 28.83.x) | 8× H20 | SSH Permission denied — unavailable |

## Recent Completed (2026-05-16)
- **1B v1** (500 steps): mean=35.19, ≥8K qa1/qa2/qa5 = 35.7/24.3/55.7 vs LM2 paper 19.0/8.0/36.5
- **1B v2** (10k steps): mean=37.29, marginal +2.1 over v1 — 500 steps captures 94% of value
- **v2 multi-ckpt eval**: step1000/step5000/final scored — results in `outputs/eval_phase1b_v2_*/`
- **8B P8** (reference): mean=59.14, qa1≥8K=47.7 — 1B v4+L2 target is to approach this at ≥8K
