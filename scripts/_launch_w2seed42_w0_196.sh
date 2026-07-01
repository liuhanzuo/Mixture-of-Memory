#!/usr/bin/env bash
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export WANDB_MODE=offline
export RUN_PREFIX=swaA_seed42_W0
export CKPT_FILES="outputs/mem_space_selfstudy_swateacherA_chunk512/full_model_step000500.pt outputs/mem_space_selfstudy_swateacherA_chunk512/full_model_step001000.pt"
export CK_NAMES="swaA_seed42_step500 swaA_seed42_step1000"
export ADAPTER_CONFIG=outputs/mem_space_selfstudy_swateacherA_chunk512/adapter_config.json
export MODEL=models/Meta-Llama-3-8B
export PROJECT_ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export PYTHON_BIN=/opt/conda/envs/torch-base/bin/python
bash scripts/_eval_taskpool_2group.sh
