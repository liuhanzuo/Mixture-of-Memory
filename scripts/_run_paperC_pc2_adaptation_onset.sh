#!/usr/bin/env bash
# Paper C P-C2 adaptation-onset probe launcher.
#
# Runs scripts/probe_paperC_adaptation_onset.py sequentially on all 6
# (base, ft) pairs using 8x H20 DDP each, forward-only.
#
# Pairs:
#   1. A4_hero       = paperC_pc1_squad_A4/final.pt          (keep14+fresh2, freeze-graft SQuAD-FT)
#   2. A3_fromscratch= paperC_pc1_squad_A3/final.pt          (keep14+fresh2, from-scratch)
#   3. A2_lora_r160  = paperC_pc1_squad_A2_lora_r160/merged  (full 32L + LoRA r=160 merged)
#   4. A4_keep20     = paperC_pc1_squad_A4_keep20fresh2/final.pt
#   5. A4_keep24     = paperC_pc1_squad_A4_keep24fresh2/final.pt
#   6. A4_keep28     = paperC_pc1_squad_A4_keep28fresh2/final.pt
#
# Plus a base-vs-base sanity check (pair 0).
#
# Usage:
#   cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
#   setsid nohup bash scripts/_run_paperC_pc2_adaptation_onset.sh \
#     > logs/paperC_pc2_onset.log 2>&1 &
#
# Node: .82 (8xH20). Python: /opt/conda/envs/torch-base/bin/python (.venv broken).
# Disk: zwfy6 (all inputs/outputs there).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"

BASE_PATH="${BASE_PATH:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/squad_val.jsonl}"
N_WIN="${N_WIN:-512}"
SEQ_LEN="${SEQ_LEN:-2048}"
SEED="${SEED:-42}"
PBS="${PBS:-16}"                 # per-batch positions kept (fp64 pool)
BATCH_SIZE="${BATCH_SIZE:-1}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
PORT_BASE="${PORT_BASE:-29509}"

OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/paperC_probe_results}"
mkdir -p "$OUT_ROOT" "$PROJECT_ROOT/logs"
mkdir -p "$OUT_ROOT/onset_baseself" "$OUT_ROOT/onset_A4" "$OUT_ROOT/onset_A3" \
         "$OUT_ROOT/onset_A2" "$OUT_ROOT/onset_A4_keep20" \
         "$OUT_ROOT/onset_A4_keep24" "$OUT_ROOT/onset_A4_keep28"

log() { echo "[$(date '+%F %T')] $*"; }

run_pair() {
  local tag="$1"; local ft_mode="$2"; local ft_path="$3"; local out_sub="$4"
  local port="$5"
  log "==== pair: $tag (mode=$ft_mode) ==== ft=$ft_path"
  CUDA_VISIBLE_DEVICES="$GPUS" \
  "$PYTHON_BIN" -m torch.distributed.run \
      --nproc_per_node="$NPROC" --master_port="$port" \
      scripts/probe_paperC_adaptation_onset.py \
      --base_path "$BASE_PATH" \
      --ft_path "$ft_path" \
      --ft_mode "$ft_mode" \
      --data_path "$DATA_PATH" \
      --tok_path "$BASE_PATH" \
      --n_windows "$N_WIN" --seq_len "$SEQ_LEN" --seed "$SEED" \
      --batch_size "$BATCH_SIZE" --per_batch_subsample "$PBS" \
      --out_dir "$OUT_ROOT/$out_sub" --tag "$tag" \
    2>&1 | tee "$PROJECT_ROOT/logs/paperC_pc2_onset_${tag}.log"
  log "==== done: $tag ===="
  # cool-down before next pair
  sleep 2
}

# ---- pair 0: base-vs-base self-consistency ---------------------------------
run_pair "baseself" "base"  "$BASE_PATH" "onset_baseself" "$((PORT_BASE+0))"

# early sanity gate: if base-vs-base fails, stop everything
python3 - <<PY || { echo "[FATAL] base-vs-base sanity gate failed"; exit 2; }
import json
r = json.load(open("$OUT_ROOT/onset_baseself/cka_per_layer.json"))
s = r["sanity"]
print("[gate] base-vs-base min CKA =", s["base_vs_base_min_cka"], "pass:", s["base_vs_base_pass"])
assert s["base_vs_base_pass"], "base-vs-base min CKA < 0.999 -- forward nondeterminism too high, HALT"
PY

# ---- pair 1: A4 hero (keep14+fresh2 freeze-graft) ---------------------------
run_pair "A4_hero" "pruned" \
  "$PROJECT_ROOT/outputs/paperC_pc1_squad_A4/final.pt" \
  "onset_A4" "$((PORT_BASE+1))"

# ---- pair 2: A3 from-scratch (16L) -----------------------------------------
run_pair "A3_fromscratch" "pruned" \
  "$PROJECT_ROOT/outputs/paperC_pc1_squad_A3/final.pt" \
  "onset_A3" "$((PORT_BASE+2))"

# ---- pair 3: A2 LoRA r=160 (merged, full 32L) ------------------------------
run_pair "A2_lora_r160" "hf_dir" \
  "$PROJECT_ROOT/outputs/paperC_pc1_squad_A2_lora_r160/merged" \
  "onset_A2" "$((PORT_BASE+3))"

# ---- pairs 4-6: A4 depth-sweep (keep20/24/28) -------------------------------
run_pair "A4_keep20" "pruned" \
  "$PROJECT_ROOT/outputs/paperC_pc1_squad_A4_keep20fresh2/final.pt" \
  "onset_A4_keep20" "$((PORT_BASE+4))"

run_pair "A4_keep24" "pruned" \
  "$PROJECT_ROOT/outputs/paperC_pc1_squad_A4_keep24fresh2/final.pt" \
  "onset_A4_keep24" "$((PORT_BASE+5))"

run_pair "A4_keep28" "pruned" \
  "$PROJECT_ROOT/outputs/paperC_pc1_squad_A4_keep28fresh2/final.pt" \
  "onset_A4_keep28" "$((PORT_BASE+6))"

# ---- aggregate ---------------------------------------------------------------
"$PYTHON_BIN" - <<PY
import json, os, glob
root = "$OUT_ROOT"
all_curves = {}
for sub in ["onset_baseself","onset_A4","onset_A3","onset_A2",
            "onset_A4_keep20","onset_A4_keep24","onset_A4_keep28"]:
    p = os.path.join(root, sub, "cka_per_layer.json")
    if not os.path.exists(p):
        print("[warn] missing", p); continue
    r = json.load(open(p))
    all_curves[r["tag"]] = {
        "ft_mode": r["ft_mode"],
        "ft_meta": r["ft_meta"],
        "config": r["config"],
        "sanity": r.get("sanity", {}),
        "per_layer": r["per_layer"],
    }
out = os.path.join(root, "all_curves.json")
with open(out,"w") as f:
    json.dump(all_curves, f, indent=2)
print("[aggregate] wrote", out)
PY

log "ALL DONE."
