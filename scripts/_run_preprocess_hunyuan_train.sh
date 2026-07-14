#!/usr/bin/env bash
# One-shot wrapper: tokenize SlimPajama-6B (6 train shards) with the Hunyuan-A13B
# tokenizer into data/slimpajama_chunks_2048_hunyuan.npy. Runs from the project root
# so relative paths resolve regardless of the (non-interactive) SSH login cwd.
set -euo pipefail
cd /volume/haru/Mixture-of-Memory
mkdir -p logs
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec .venv_hy3/bin/python scripts/preprocess_slimpajama.py \
  --input_dir data/slimpajama-6b/data \
  --output data/slimpajama_chunks_2048_hunyuan.npy \
  --tokenizer models/Hunyuan-A13B-Pretrain \
  --chunk_size 2048 --trust_remote_code --eos_token_id 127960 \
  --skip_val --num_proc 16 --num_train_shards 6
