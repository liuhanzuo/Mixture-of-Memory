#!/bin/bash
cd /root/Mixture-of-Memory
tmux kill-session -t train_sparse 2>/dev/null
sleep 2
tmux new-session -d -s train_sparse
tmux send-keys -t train_sparse 'export MASTER_PORT=29602 && CUDA_VISIBLE_DEVICES=0-7 torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --redirects 1 --log_dir /root/Mixture-of-Memory/logs/slp_selective_256 scripts/train_sparse_memory.py --model_path /root/Mixture-of-Memory/models/Llama--Llama2-7b/ --data_path /root/Mixture-of-Memory/data/slimpajama_chunks_4096.npy --output_dir /root/Mixture-of-Memory/outputs/slp_selective_256 --memory_slots 256 --top_k 16 --write_top_k 16 --sliding_window 256 --lr 2e-5 --batch_size 2 --grad_accumulation_steps 4 --max_steps 5000 --log_every 50 --save_every 1000 --mixed_precision bf16 --seq_len 4096 2>&1 | tee /root/Mixture-of-Memory/logs/train_slp_selective_256.log' Enter
