#!/usr/bin/env bash
# Progressive chunk_size warm-start chain V2 (diskB, SINGLE-NODE 1x8 = 8-GPU DDP).
#
# V2 = V1 (launch_progressive_chunk_diskB.sh) + per-stage REVERSE-SCALED warmup
# and gradient_accumulation. Everything else (P11 stable recipe, slot config,
# warm-start chain, loss_spike_skip, offline env) is byte-for-byte identical to v1.
#
# Motivation (status/research_notes/small_chunk_training_and_slot_capacity_20260607.md):
# under per_doc tbptt the per-step gradient token count = (n_ctx+1)*chunk_size, so
# chunk128=512 vs chunk1024=4096 -> small chunk has ~4x larger single-step gradient
# variance. v1 used a GLOBAL fixed warmup=300 + grad_accum=2 for all stages, which
# (a) under-warms small chunks (300 steps @ chunk128 ~= 0.6M token vs ~5M @ chunk1024)
# and (b) leaves small-chunk gradient variance uncompensated. v2 scales BOTH knobs
# inversely with chunk_size so warmup-token and effective-gradient-token per step stay
# roughly constant across stages:
#   stage1 c128  -> warmup 800, grad_accum 8
#   stage2 c256  -> warmup 500, grad_accum 4
#   stage3 c512  -> warmup 300, grad_accum 2   (== v1)
#   stage4 c1024 -> warmup 200, grad_accum 1
# Research note tags both changes [high, 可直接采用], zero-risk.
#
# Idea (user intent): grow chunk_size in stages so the memory architecture first
# learns to compress short contexts, then is progressively challenged with longer
# ones. We do NOT vary chunk_size inside one run (would perturb RoPE / injection
# count / optimizer state). Instead we chain runs via --init_checkpoint warm-start:
#   stage1 chunk128 (scratch) -> stage2 chunk256 -> stage3 chunk512 -> stage4 chunk1024
# Each stage inits from the PREVIOUS stage's step000600 adapter.
#
# STABLE RECIPE (same as v1): bakes in the empirically-most-stable P11 configuration
# (status/research_notes/training_stability_20260607.md): ST-Gumbel topk OFF +
# delta-rule writeback + normalized readout (--use_delta_rule_writeback
# --normalize_readout), and the --loss_spike_skip guard. Everything else
# (num_slots / top_k / selector_temperature / loss-free balance / route_aux /
# memory xattn / dual-gate / l3 summary ...) mirrors the h800 P8-nullsink recipe.
#
# SINGLE-NODE USAGE (diskB .76 = 28.49.57.76, 8x H20, shares FS with .249):
#   bash scripts/launch_progressive_chunk_diskB_v2.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

# --- Single-node DDP topology (no cross-node IB / DMABUF flags) ---
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export WANDB_API_KEY="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"
export WANDB_MODE="offline"
# FULLY OFFLINE babilong: cache is pre-warmed (0k-32k) on diskB shared FS, so
# every rank's load_dataset reads straight from the local Arrow cache instead of
# doing a slow online builder-resolve. No network / proxy needed on diskB.
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TOTAL_STEPS="${TOTAL_STEPS:-800}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
LOSS_SPIKE_SIGMA="${LOSS_SPIKE_SIGMA:-3.0}"
# Early ckpt used to chain to the next stage.
CHAIN_STEP="${CHAIN_STEP:-600}"
CHAIN_CKPT_NAME="mem_space_adapter_step$(printf '%06d' "$CHAIN_STEP").pt"

OUT_BASE="${OUT_BASE:-outputs/progressive_chunk_diskB_v2}"
mkdir -p logs "$OUT_BASE"

# Shared stable (P11) config; chunk_size / output_dir / total_steps /
# init_checkpoint / master_port / warmup_steps / gradient_accumulation_steps
# differ per stage. (v2: warmup + grad_accum are now per-stage args.)
run_stage () {
  local stage_name="$1"   # e.g. stage1_c128
  local chunk="$2"        # chunk_size
  local port="$3"         # master_port (unique per stage)
  local init_ckpt="$4"    # "" for scratch, else path to prior adapter
  local warmup="$5"       # v2: per-stage warmup_steps (reverse-scaled with chunk)
  local grad_accum="$6"   # v2: per-stage gradient_accumulation_steps (reverse-scaled)
  local out_dir="$OUT_BASE/$stage_name"
  local log="logs/progressive_chunk_diskB_v2_${stage_name}.log"
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

  echo "[$stage_name] chunk_size=$chunk total_steps=$TOTAL_STEPS warmup=$warmup grad_accum=$grad_accum out=$out_dir port=$port" | tee -a "$log"

  CUDA_VISIBLE_DEVICES="$GPUS" "$PYBIN" -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="$port" \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_path models/Meta-Llama-3-8B \
    --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
    --output_dir "$out_dir" --total_steps "$TOTAL_STEPS" --lr 1e-4 --warmup_steps "$warmup" \
    --chunk_size "$chunk" --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
    --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
    --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
    --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
    --use_delta_rule_writeback --normalize_readout \
    --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
    --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
    --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
    --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps "$grad_accum" \
    --curriculum 0:3 --bptt_window 2 --no_detach_slots_in_selector \
    --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
    --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
    --l_recon_weight 0.0 --route_aux_weight 1.0 \
    --use_memory_xattn --memory_xattn_gate_init 0.4 \
    --loss_spike_skip --loss_spike_sigma "$LOSS_SPIKE_SIGMA" \
    --save_interval "$SAVE_INTERVAL" --eval_interval 0 --eval_samples 30 --log_interval 5 \
    --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
    --wandb_run_name "progressive_chunk_diskB_v2_${stage_name}" --dtype bfloat16 --attn_impl sdpa --seed 42 \
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

echo "=== Progressive chunk_size chain V2 (diskB, per-stage warmup/grad_accum) start $(date) ==="

# Per-stage REVERSE-SCALED warmup + grad_accum (small chunk -> larger warmup/accum).
# Stage 1: chunk128, from scratch.   warmup=800 grad_accum=8
CKPT1=$(run_stage "stage1_c128"  128  29850 ""        800 8 | tail -1)
# Stage 2: chunk256, warm-start from stage1.  warmup=500 grad_accum=4
CKPT2=$(run_stage "stage2_c256"  256  29851 "$CKPT1"  500 4 | tail -1)
# Stage 3: chunk512, warm-start from stage2.  warmup=300 grad_accum=2 (== v1)
CKPT3=$(run_stage "stage3_c512"  512  29852 "$CKPT2"  300 2 | tail -1)
# Stage 4: chunk1024, warm-start from stage3.  warmup=200 grad_accum=1
CKPT4=$(run_stage "stage4_c1024" 1024 29853 "$CKPT3"  200 1 | tail -1)

echo "=== Progressive chunk_size chain V2 (diskB) DONE $(date) ==="
echo "stage1=$CKPT1"
echo "stage2=$CKPT2"
echo "stage3=$CKPT3"
echo "stage4=$CKPT4"
