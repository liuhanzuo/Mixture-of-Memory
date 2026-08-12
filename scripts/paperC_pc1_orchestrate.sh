#!/usr/bin/env bash
# Paper C P-C1 overnight orchestrator (run ON .104, setsid nohup). Chains the
# remaining arms A3 (from-scratch) -> A1 (full-FT 32L) -> A2 (LoRA r=160) on ALL
# 8 cards sequentially (each eff_bs=128, no contention), waiting first for the
# already-running A4 (hero) to finish. Then runs the SQuAD dev EM/F1 headline eval
# on every finished arm + the untuned base reference. Each stage is fault-tolerant:
# a failed/OOM arm is logged and the chain continues.
#
# Usage (on .104):
#   cd <project_root>
#   setsid nohup bash scripts/paperC_pc1_orchestrate.sh > logs/paperC_pc1_orch.log 2>&1 &
set -uo pipefail

ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$ROOT"
PY=/opt/conda/envs/torch-base/bin/python
BASE=/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B
GPUS=0,1,2,3,4,5,6,7
STEPS=1000
export WANDB_MODE=offline TOKENIZERS_PARALLELISM=false

log(){ echo "[orch $(date '+%F %T')] $*"; }

wait_for_final(){  # $1=out_dir  $2=timeout_sec
  local d="$1" to="${2:-21600}" waited=0
  while [ ! -f "$d/final.pt" ] && [ ! -d "$d/merged" ]; do
    sleep 60; waited=$((waited+60))
    if [ "$waited" -ge "$to" ]; then log "TIMEOUT waiting $d/final.pt"; return 1; fi
    # bail early if no python training proc is alive AND still no final -> crashed
    if ! pgrep -f 'train_olmo2_(arch_probe2|lora_sft)' >/dev/null 2>&1; then
      sleep 30
      if [ ! -f "$d/final.pt" ] && [ ! -d "$d/merged" ]; then
        log "no trainer alive and no final in $d -> assume crashed"; return 1
      fi
    fi
  done
  log "final present in $d"; return 0
}

# ---- 0. wait for the in-flight A4 hero ----
log "waiting for A4 (hero) final.pt ..."
wait_for_final "$ROOT/outputs/paperC_pc1_squad_A4" 18000 || log "A4 wait ended (may have crashed)"

# ---- 1. A3 from-scratch (depth-matched 16L, fp32 AdamW, 8 cards) ----
log "=== launch A3 from-scratch ==="
ARM=A3 GPUS=$GPUS PORT=29552 MAX_STEPS=$STEPS BS=4 GA=4 FOREGROUND=1 bash scripts/run_paperC_pc1.sh
wait_for_final "$ROOT/outputs/paperC_pc1_squad_A3" 3600 || log "A3 no final (check log)"

# ---- 2. A1 full-FT 32L (7B all-params -> bnb 8-bit Adam, BS2/GA8 eff_bs128) ----
log "=== launch A1 full-FT 32L (bnb8bit) ==="
ARM=A1 GPUS=$GPUS PORT=29553 MAX_STEPS=$STEPS BS=2 GA=8 OPT=bnb8bit FOREGROUND=1 bash scripts/run_paperC_pc1.sh
if ! wait_for_final "$ROOT/outputs/paperC_pc1_squad_A1" 3600; then
  log "A1 no final -> likely OOM on H20; retry BS1/GA16"
  ARM=A1 GPUS=$GPUS PORT=29553 MAX_STEPS=$STEPS BS=1 GA=16 OPT=bnb8bit FOREGROUND=1 bash scripts/run_paperC_pc1.sh
  wait_for_final "$ROOT/outputs/paperC_pc1_squad_A1" 5400 || log "A1 STILL no final -> NEEDS B200"
fi

# ---- 3. A2 LoRA r=160 param-matched (frozen bf16 base + fp32 adapters) ----
log "=== launch A2 LoRA r=160 ==="
GPUS=$GPUS PORT=29554 R=160 MAX_STEPS=$STEPS BS=4 GA=4 FOREGROUND=1 bash scripts/run_paperC_pc1_lora.sh
wait_for_final "$ROOT/outputs/paperC_pc1_squad_A2_lora_r160" 3600 || log "A2 no merged (check log)"

# ---- 4. SQuAD dev EM/F1 headline eval on every finished arm + base ref ----
log "=== SQuAD EM/F1 evals ==="
eval_ckpt(){  # $1=name $2=ckpt-or-empty $3=base-or-merged
  local name="$1" ckpt="$2" bm="$3"
  log "eval $name"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_paperC_squad_emf1.py \
    ${ckpt:+--ckpt "$ckpt"} --base_model "$bm" --tokenizer "$BASE" \
    --val_path data/squad_val.jsonl --output_name "$name" --batch_size 32 \
    >> "logs/paperC_pc1_eval_${name}.log" 2>&1
  $PY scripts/eval_paperC_squad_emf1.py --merge --output_name "$name" \
    >> "logs/paperC_pc1_eval_${name}.log" 2>&1
}
[ -f outputs/paperC_pc1_squad_A4/final.pt ] && eval_ckpt A4_hero        outputs/paperC_pc1_squad_A4/final.pt "$BASE"
[ -f outputs/paperC_pc1_squad_A3/final.pt ] && eval_ckpt A3_fromscratch outputs/paperC_pc1_squad_A3/final.pt "$BASE"
[ -f outputs/paperC_pc1_squad_A1/final.pt ] && eval_ckpt A1_fullft      outputs/paperC_pc1_squad_A1/final.pt "$BASE"
[ -d outputs/paperC_pc1_squad_A2_lora_r160/merged ] && eval_ckpt A2_lora_r160 "" outputs/paperC_pc1_squad_A2_lora_r160/merged
eval_ckpt BASE_ref "" "$BASE"

# ---- 5. summary table ----
log "=== P-C1 SQuAD EM/F1 SUMMARY ==="
$PY - <<'PY'
import glob, json, os
rows=[]
for d in sorted(glob.glob("evidence_squad_label_prior/*/summary.json")):
    s=json.load(open(d))
    rows.append((os.path.basename(os.path.dirname(d)), s.get("em"), s.get("f1"), s.get("n")))
print(f"{'arm':22s} {'EM':>8s} {'F1':>8s} {'n':>6s}")
for name,em,f1,n in rows:
    print(f"{name:22s} {em:8.4f} {f1:8.4f} {n:6d}")
PY
log "=== ORCHESTRATOR DONE ==="
