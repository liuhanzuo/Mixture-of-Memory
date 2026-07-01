#!/usr/bin/env bash
# swateacherA (W2-teacher) FINAL step2000 W0 eval — local 8xH20
# Completes the W2-teacher long-train degradation trajectory: step500/1000 on .196, step2000 here.
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export WANDB_MODE=offline
export RUN_PREFIX=swaA_seed42_W0
export CKPT_FILES="outputs/mem_space_selfstudy_swateacherA_chunk512/full_model.pt"
export CK_NAMES="swaA_seed42_step2000"
export ADAPTER_CONFIG=outputs/mem_space_selfstudy_swateacherA_chunk512/adapter_config.json
export MODEL=models/Meta-Llama-3-8B
export PROJECT_ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export PYTHON_BIN=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/.venv/bin/python
bash scripts/_eval_taskpool_2group.sh
