#!/bin/bash
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory || exit 1
rm -rf src/memory/mem_space/__pycache__
export WANDB_API_KEY='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'
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
nohup /opt/conda/envs/torch-base/bin/python -u -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=29720 \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_name models/Meta-Llama-3-8B-Instruct \
    --output_dir outputs/p1v3_chunk_query \
    --num_slots 128 \
    --selector_dim 256 \
    --top_k 8 \
    --num_mem_tokens 5 \
    --n_ctx 4 \
    --chunk_len 1024 \
    --total_steps 2000 \
    --lr 1e-3 \
    --batch_size 1 \
    --routing_pool_mode chunk_query \
    --log_every 10 \
    --eval_every 500 \
    --seed 44 \
    > logs/p1v3_chunk_query.log 2>&1 &
echo "PID=$!"
