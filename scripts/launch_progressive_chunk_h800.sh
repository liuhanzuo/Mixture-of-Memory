#!/usr/bin/env bash
# Progressive chunk_size warm-start chain (H800, single node, 8-GPU DDP).
#
# Idea (user intent): grow chunk_size in stages so the memory architecture first
# learns to compress short contexts, then is progressively challenged with longer
# ones — acquiring stronger long-range compression. We do NOT vary chunk_size
# inside one run (would perturb RoPE / injection count / optimizer state). Instead
# we chain runs via --init_checkpoint warm-start:
#   stage1 chunk128 (scratch) -> stage2 chunk256 -> stage3 chunk512 -> stage4 chunk1024
# Each stage inits from the PREVIOUS stage's step000600 adapter (early ckpt, taken
# before the known ~step1000 loss-spike collapse).
#
# Hyperparams mirror launch_mem_space_p8_nullsink_chunk512_diskB.sh EXACTLY; only
# chunk_size, output_dir, total_steps, save_interval, init_checkpoint differ.
#   - 8 GPUs x bs1 x ga4 = eff batch 32 (matches all sibling arms)
#   - total_steps 800 / save_interval 200 / eval_interval 0 (inline eval -> NCCL crash)
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_jn2/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TOTAL_STEPS="${TOTAL_STEPS:-800}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
# Early ckpt used to chain to the next stage (before the ~step1000 spike).
CHAIN_STEP="${CHAIN_STEP:-600}"
CHAIN_CKPT_NAME="mem_space_adapter_step$(printf '%06d' "$CHAIN_STEP").pt"

OUT_BASE="outputs/progressive_chunk_h800"
mkdir -p logs "$OUT_BASE"

# Shared P8 null-sink config (copied verbatim from chunk512_diskB; chunk_size omitted).
run_stage () {
  local stage_name="$1"   # e.g. stage1_c128
  local chunk="$2"        # chunk_size
  local port="$3"         # master_port (unique per stage)
  local init_ckpt="$4"    # "" for scratch, else path to prior adapter
  local out_dir="$OUT_BASE/$stage_name"
  local log="logs/progressive_chunk_h800_${stage_name}.log"
  mkdir -p "$out_dir"

  local init_arg=""
  if [[ -n "$init_ckpt" ]]; then
    if [[ ! -f "$init_ckpt" ]]; then
      echo "[FATAL] $stage_name: init_checkpoint $init_ckpt missing -> abort chain" | tee -a "$log"
      exit 1
    fi
    init_arg="--init_checkpoint $init_ckpt"
    echo "[$stage_name] warm-start from $init_ckpt" | tee -a "$log"
  else
    echo "[$stage_name] from scratch (no init_checkpoint)" | tee -a "$log"
  fi

  echo "[$stage_name] chunk_size=$chunk total_steps=$TOTAL_STEPS out=$out_dir port=$port" | tee -a "$log"

  CUDA_VISIBLE_DEVICES="$GPUS" "$PYBIN" -m torch.distributed.run --nproc_per_node=8 --master_port="$port" \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_path models/Meta-Llama-3-8B \
    --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
    --output_dir "$out_dir" --total_steps "$TOTAL_STEPS" --lr 1e-4 --warmup_steps 100 \
    --chunk_size "$chunk" --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
    --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
    --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
    --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
    --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
    --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
    --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
    --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 4 \
    --curriculum 0:3 --bptt_window 2 --no_detach_slots_in_selector \
    --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
    --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
    --l_recon_weight 0.0 --route_aux_weight 1.0 \
    --use_memory_xattn --memory_xattn_gate_init 0.4 \
    --save_interval "$SAVE_INTERVAL" --eval_interval 0 --eval_samples 30 --log_interval 5 \
    --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
    --wandb_run_name "progressive_chunk_h800_${stage_name}" --dtype bfloat16 --attn_impl sdpa --seed 42 \
    $init_arg \
    >>"$log" 2>&1

  # Verify the chain checkpoint was produced.
  local chain_ckpt="$out_dir/$CHAIN_CKPT_NAME"
  local final_ckpt="$out_dir/mem_space_adapter.pt"
  if [[ -f "$chain_ckpt" ]]; then
    echo "[$stage_name] DONE. chain ckpt = $chain_ckpt" | tee -a "$log"
    echo "$chain_ckpt"
  elif [[ -f "$final_ckpt" ]]; then
    echo "[$stage_name] DONE (no step$CHAIN_STEP ckpt; using final). chain ckpt = $final_ckpt" | tee -a "$log"
    echo "$final_ckpt"
  else
    echo "[FATAL] $stage_name produced no adapter.pt in $out_dir -> abort chain" | tee -a "$log"
    exit 1
  fi
}

echo "=== Progressive chunk_size chain start $(date) ==="

# Stage 1: chunk128, from scratch.
CKPT1=$(run_stage "stage1_c128" 128 29810 "" | tail -1)
# Stage 2: chunk256, warm-start from stage1.
CKPT2=$(run_stage "stage2_c256" 256 29811 "$CKPT1" | tail -1)
# Stage 3: chunk512, warm-start from stage2.
CKPT3=$(run_stage "stage3_c512" 512 29812 "$CKPT2" | tail -1)
# Stage 4: chunk1024, warm-start from stage3.
CKPT4=$(run_stage "stage4_c1024" 1024 29813 "$CKPT3" | tail -1)

echo "=== Progressive chunk_size chain DONE $(date) ==="
echo "stage1=$CKPT1"
echo "stage2=$CKPT2"
echo "stage3=$CKPT3"
echo "stage4=$CKPT4"
