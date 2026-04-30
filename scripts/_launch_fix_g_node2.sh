#!/bin/bash
export NODE_IDX=2
export PROJECT_DIR=/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$PROJECT_DIR"
source /opt/conda/etc/profile.d/conda.sh
conda activate torch-base
exec bash "$PROJECT_DIR/scripts/_run_fix_g_ablation.sh"
