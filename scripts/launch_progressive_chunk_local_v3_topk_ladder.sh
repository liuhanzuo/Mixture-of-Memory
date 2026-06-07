#!/usr/bin/env bash
# Progressive chunk_size warm-start chain V3 -- top_k LADDER arm
# (LOCAL diskA, SINGLE-NODE 1x8 = 8-GPU DDP).
#
# This is a SINGLE-VARIABLE ablation arm of the canonical F1 v3 improved chain
# (scripts/launch_progressive_chunk_diskB_v3_improved.sh). Everything is byte-for-byte
# identical to the diskB v3 canonical EXCEPT:
#   (a) paths -> local diskA (share_303098609) + local .venv + local HF cache,
#   (b) top_k is laddered with chunk_size (the ONLY trained-config change),
#   (c) output_dir / run_name / log / master_port are made independent so this arm
#       can run in parallel with the diskB canonical without colliding.
#
# --- The single experimental variable: top_k laddered with chunk_size ---
# canonical diskB v3 keeps a FIXED --top_k 16 across all 3 stages. This arm tests the
# research-note hypothesis (status/research_notes/small_chunk_training_and_slot_capacity_20260607.md,
# confidence=high) that "larger chunk needs more activated slots":
#   stage1_c256  -> --top_k 16   (>=16 floor; RUN_REGISTRY §3.4 showed top_k8 hurts retrieval)
#   stage2_c512  -> --top_k 24
#   stage3_c1024 -> --top_k 32
# top_k is a RUNTIME selection count (not a parameter shape), so laddering it across
# stages does NOT break warm-start state_dict shape matching. num_slots / slot_dim /
# selector_dim are UNCHANGED (--num_slots 128 --selector_dim 128) to keep ckpt shapes
# identical across stages -- this is what makes the chain warm-start cleanly.
#
# All other v3 per-stage knobs are kept IDENTICAL to the canonical:
#   warmup ~= round(300*1024/chunk):  c256=1200, c512=600, c1024=300
#   accum  ~= round(2*1024/chunk):    c256=8,    c512=4,   c1024=2
#   loss_spike_sigma relaxed small:   c256=4.0,  c512=3.5, c1024=3.0
# Chain is 3 stages: stage1_c256(scratch) -> stage2_c512 -> stage3_c1024.
#
# SINGLE-NODE USAGE (local diskA, 8x H20):
#   bash scripts/launch_progressive_chunk_local_v3_topk_ladder.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

# --- Single-node DDP topology (no cross-node IB / DMABUF flags) ---
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export WANDB_API_KEY="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"
# Set offline to avoid any wandb-connect stall; flip to online manually if desired.
export WANDB_MODE="offline"
# Local diskA HF cache. diskA has internet, so HF_HUB_OFFLINE is left default-off,
# but babilong is NOT used during training here (eval_interval=0) so no dataset
# resolve happens on the train path anyway.
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# Default 5000 so the chain trains strong ckpts; env can override.
TOTAL_STEPS="${TOTAL_STEPS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
# Early ckpt used to chain to the next stage. MUST be a multiple of SAVE_INTERVAL
# (else that step's ckpt is never written -> chain breaks). 500 is a multiple of 500.
CHAIN_STEP="${CHAIN_STEP:-500}"
CHAIN_CKPT_NAME="mem_space_adapter_step$(printf '%06d' "$CHAIN_STEP").pt"

# Guard: CHAIN_STEP must be an integer multiple of SAVE_INTERVAL.
if (( CHAIN_STEP % SAVE_INTERVAL != 0 )); then
  echo "[FATAL] CHAIN_STEP ($CHAIN_STEP) must be a multiple of SAVE_INTERVAL ($SAVE_INTERVAL)" >&2
  exit 1
fi

OUT_BASE="${OUT_BASE:-outputs/progressive_chunk_local_v3_topk_ladder}"
mkdir -p logs "$OUT_BASE"

# Shared stable (P11) config; chunk_size / top_k / output_dir / init_checkpoint /
# master_port / warmup_steps / gradient_accumulation_steps / loss_spike_sigma
# differ per stage. top_k is the single experimental variable in this arm.
run_stage () {
  local stage_name="$1"   # e.g. stage1_c256
  local chunk="$2"        # chunk_size
  local port="$3"         # master_port (unique per stage)
  local init_ckpt="$4"    # "" for scratch, else path to prior adapter
  local warmup="$5"       # warmup_steps scaled inversely with chunk (same as canonical)
  local accum="$6"        # gradient_accumulation_steps scaled inversely with chunk (same as canonical)
  local sigma="$7"        # loss_spike_sigma relaxed for small chunk (same as canonical)
  local topk="$8"         # LADDER: top_k scaled with chunk (THE experimental variable)
  local out_dir="$OUT_BASE/$stage_name"
  local log="logs/progressive_chunk_local_v3_topk_ladder_${stage_name}.log"
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

  echo "[$stage_name] chunk_size=$chunk top_k=$topk total_steps=$TOTAL_STEPS warmup=$warmup accum=$accum sigma=$sigma out=$out_dir port=$port" | tee -a "$log"

  CUDA_VISIBLE_DEVICES="$GPUS" "$PYBIN" -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="$port" \
    scripts/train_mem_space_dolmino_cpt.py \
    --model_path models/Meta-Llama-3-8B \
    --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
    --output_dir "$out_dir" --total_steps "$TOTAL_STEPS" --lr 1e-4 --warmup_steps "$warmup" \
    --chunk_size "$chunk" --batch_size 1 --num_slots 128 --top_k "$topk" --selector_dim 128 \
    --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
    --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
    --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
    --use_delta_rule_writeback --normalize_readout \
    --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
    --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
    --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
    --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps "$accum" \
    --curriculum 0:3 --bptt_window 2 --no_detach_slots_in_selector \
    --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
    --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
    --l_recon_weight 0.0 --route_aux_weight 1.0 \
    --use_memory_xattn --memory_xattn_gate_init 0.4 \
    --loss_spike_skip --loss_spike_sigma "$sigma" \
    --save_interval "$SAVE_INTERVAL" --eval_interval 0 --eval_samples 30 --log_interval 5 \
    --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
    --wandb_run_name "progressive_chunk_local_v3_topk_ladder_${stage_name}" --dtype bfloat16 --attn_impl sdpa --seed 42 \
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

echo "=== Progressive chunk_size chain V3 top_k-ladder (local diskA) start $(date) ==="

# V3 per-stage knobs (warmup/accum/sigma IDENTICAL to canonical diskB v3); top_k LADDER
# is the single experimental variable. Ports use the free 29840 region (29840-29842),
# avoiding the canonical diskB v3 ports (29830-29832) and all other active ports.
# Stage 1: chunk256, from scratch.   warmup=1200 accum=8 sigma=4.0 top_k=16
CKPT1=$(run_stage "stage1_c256" 256 29840 "" 1200 8 4.0 16 | tail -1)
# Stage 2: chunk512, warm-start from stage1.  warmup=600 accum=4 sigma=3.5 top_k=24
CKPT2=$(run_stage "stage2_c512" 512 29841 "$CKPT1" 600 4 3.5 24 | tail -1)
# Stage 3: chunk1024, warm-start from stage2.  warmup=300 accum=2 sigma=3.0 top_k=32
CKPT3=$(run_stage "stage3_c1024" 1024 29842 "$CKPT2" 300 2 3.0 32 | tail -1)

echo "=== Progressive chunk_size chain V3 top_k-ladder (local diskA) DONE $(date) ==="
echo "stage1=$CKPT1"
echo "stage2=$CKPT2"
echo "stage3=$CKPT3"
