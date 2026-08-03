#!/usr/bin/env bash
# ============================================================================
# Paper B P2.2 — causal-layer restoration / activation patching, 8-GPU driver.
# PURE FORWARD INFERENCE — NO TRAINING. Runs scripts/eval_olmo2_activation_patching.py.
#
# For every intervention point it fans out 8 GPU shards of BOTH ppl and mmlu and
# merges (TODOList hard rule: report PPL *and* MMLU, never one alone).
#
# Two interventions (see paperB/P2_2_actpatch_NOTES.md for the design):
#   GRAFT   : base layer-L output residual injected at keep14 fresh-tail input,
#             scanned over $GRAFT_LAYERS. Loads BOTH base(32L) + keep14(16L) per
#             GPU (~44 GB fp32 on H20 97.8 GB) -> small batch sizes.
#   RESTORE : keep14 front-14 + k restored base upper layers (+ keep14 fresh tail
#             or base head), scanned over $RESTORE_KS. Single composed model/GPU.
#
# ── env knobs (all overridable; defaults = diskB .104 H20 layout) ─────────────
#   PROJECT_ROOT   default /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
#   PY             default /opt/conda/envs/torch-base/bin/python
#   BASE           pretrained OLMo-2-7B (32L) path
#   KEEP14         healed keep14 ckpt (.pt)  [MUST be present on the run node's FS]
#   DO_GRAFT       1 (default) run graft scan; 0 skip
#   DO_RESTORE     1 (default) run restore scan; 0 skip
#   GRAFT_LAYERS   base layer-output indices to graft (default "13 16 20 24 28 31")
#   RESTORE_KS     #upper layers to restore (default "0 2 4 6 9 12 18")
#   RESTORE_READOUT tail_keep14 (default) | base_head
#   NGPU           default 8
#   GRAFT_BS_PPL / GRAFT_BS_MMLU / RESTORE_BS_PPL / RESTORE_BS_MMLU
#   VAL            held-out PPL windows npy (default data/dolmino_now_val.npy)
#   N_BOOT         mmlu paired-bootstrap resamples (default 10000)
# BASE PROTOCOL (hard): add_bos=0, chat_template=False, zero-shot, MMLU LL-MC 14042.
# ============================================================================
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT" || exit 1
PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
KEEP14="${KEEP14:-outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt}"
DO_GRAFT="${DO_GRAFT:-1}"
DO_RESTORE="${DO_RESTORE:-1}"
GRAFT_LAYERS="${GRAFT_LAYERS:-13 16 20 24 28 31}"
RESTORE_KS="${RESTORE_KS:-0 2 4 6 9 12 18}"
RESTORE_READOUT="${RESTORE_READOUT:-tail_keep14}"
NGPU="${NGPU:-8}"
GRAFT_BS_PPL="${GRAFT_BS_PPL:-2}"
GRAFT_BS_MMLU="${GRAFT_BS_MMLU:-4}"
RESTORE_BS_PPL="${RESTORE_BS_PPL:-2}"
RESTORE_BS_MMLU="${RESTORE_BS_MMLU:-8}"
VAL="${VAL:-data/dolmino_now_val.npy}"
N_BOOT="${N_BOOT:-10000}"
ADD_BOS="${ADD_BOS:-0}"          # base protocol: no BOS
CONTENT_DESC="${CONTENT_DESC:-full}"
SCRIPT=scripts/eval_olmo2_activation_patching.py
PPL_ROOT=olmo2_actpatch_ppl_results
MMLU_ROOT=olmo2_actpatch_mmlu_results

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE="$PROJECT_ROOT/data/hf_datasets_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs "$PPL_ROOT" "$MMLU_ROOT" "$HF_DATASETS_CACHE"

[ -f "$KEEP14" ] || { echo "FATAL: KEEP14 ckpt not found: $KEEP14 (rsync it to this node's FS first)"; exit 2; }
[ -d "$BASE" ]   || { echo "FATAL: BASE model dir not found: $BASE"; exit 2; }

# args ($1..) shared by both tasks describe ONE intervention point.
run_ppl() {   # $1=output_name  $2..=mode/point args
  local NAME="$1"; shift
  echo "[$(date '+%F %T')] --- PPL $NAME ---"
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY $SCRIPT --task ppl "$@" \
      --base_model "$BASE" --keep14_ckpt "$KEEP14" --val_path "$VAL" \
      --num_shards $NGPU --shard_index $g \
      --results_root "$PPL_ROOT" --output_name "$NAME" \
      > "logs/actpatch_ppl_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  $PY $SCRIPT --merge --task ppl --results_root "$PPL_ROOT" --output_name "$NAME" 2>&1
}

run_mmlu() {  # $1=output_name  $2..=mode/point args
  local NAME="$1"; shift
  echo "[$(date '+%F %T')] --- MMLU $NAME ---"
  # cache cais/mmlu ONCE (CPU, proxy) to avoid an 8-way download race
  $PY $SCRIPT --prepare_data --content_desc "$CONTENT_DESC" \
      > "logs/actpatch_mmlu_${NAME}_prepare.log" 2>&1
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY $SCRIPT --task mmlu "$@" \
      --base_model "$BASE" --keep14_ckpt "$KEEP14" \
      --content_desc "$CONTENT_DESC" --add_bos "$ADD_BOS" \
      --num_shards $NGPU --shard_index $g \
      --results_root "$MMLU_ROOT" --output_name "$NAME" \
      > "logs/actpatch_mmlu_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  $PY $SCRIPT --merge --task mmlu --results_root "$MMLU_ROOT" \
      --output_name "$NAME" --n_boot "$N_BOOT" 2>&1
}

# ---------------------------------------------------------------------------
# GRAFT scan
# ---------------------------------------------------------------------------
if [ "$DO_GRAFT" = "1" ]; then
  for L in $GRAFT_LAYERS; do
    NAME="graft_baseL${L}_injTail"
    run_ppl  "$NAME" --mode graft --graft_layer "$L" --batch_size "$GRAFT_BS_PPL"
    run_mmlu "$NAME" --mode graft --graft_layer "$L" --batch_size "$GRAFT_BS_MMLU"
  done
fi

# ---------------------------------------------------------------------------
# RESTORE scan  (k=0 is a built-in identity check vs plain keep14)
# ---------------------------------------------------------------------------
if [ "$DO_RESTORE" = "1" ]; then
  for K in $RESTORE_KS; do
    NAME="restore_k${K}_${RESTORE_READOUT}"
    run_ppl  "$NAME" --mode restore --restore_k "$K" --restore_readout "$RESTORE_READOUT" --batch_size "$RESTORE_BS_PPL"
    run_mmlu "$NAME" --mode restore --restore_k "$K" --restore_readout "$RESTORE_READOUT" --batch_size "$RESTORE_BS_MMLU"
  done
fi

echo "[$(date '+%F %T')] ===== P2.2 activation-patching ALL DONE ====="
echo "PPL summaries: $PPL_ROOT/*/summary.json ; MMLU summaries: $MMLU_ROOT/*/summary.json"
