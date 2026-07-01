#!/bin/bash
# P1 verdict: BABILong eval norecon vs recon final adapters, 8 local H20
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export PYTHONUNBUFFERED=1
export PYTHONPATH="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory:/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/third_party/babilong-pkg"
PYTHON=.venv/bin/python
MODEL=models/Meta-Llama-3-8B
LENGTHS="0k 1k 2k 4k 8k 16k 32k"
RESULTS=babilong_results/p1_verdict
mkdir -p $RESULTS logs

# arm: name:adapter_dir
ARMS=(
  "norecon:outputs/dolmino_norecon_local_v2"
  "recon:outputs/dolmino_recon_diskA"
)
# split tasks across GPU pairs to parallelize: GPU0/1=arm tasks A, GPU2/3=arm tasks B per arm
GPU=0
for entry in "${ARMS[@]}"; do
  NAME="${entry%%:*}"
  DIR="${entry##*:}"
  ADAPTER="${DIR}/mem_space_adapter.pt"
  CFG="${DIR}/adapter_config.json"
  # split tasks into 2 GPUs per arm
  for TG in "qa1 qa2 qa3" "qa4 qa5"; do
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_babilong_mem_space.py \
      --model_path $MODEL --checkpoint $ADAPTER --adapter_config $CFG \
      --results_folder $RESULTS --output_name p1_${NAME}_g${GPU} \
      --tasks $TG --lengths $LENGTHS --chunk_size 1024 \
      > logs/eval_p1_${NAME}_g${GPU}.log 2>&1 &
    GPU=$((GPU+1))
  done
done
echo "Launched on GPUs 0-$((GPU-1))"
wait
echo "[$(date)] All P1 eval jobs done"
