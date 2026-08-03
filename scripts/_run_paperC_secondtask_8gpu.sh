#!/usr/bin/env bash
# ============================================================================
# Paper C P-C1 follow-up (task #132): SECOND-TASK capability eval on 8 GPUs.
#
# WHY: SQuAD EM (the P-C1 headline) is format-dominated. This harness scores the
# SAME 4 arms on CAPABILITY-sensitive benchmarks (zero-shot MC knowledge/reasoning
# + closed-book QA) so we can tell a real capability gap from a formatting artefact.
#
# Arms (default, override with ARMS="..."):
#   A4    freeze-graft keep14+fresh2 (16L)  -> raw .pt   outputs/paperC_pc1_squad_A4/final.pt
#   A3    from-scratch  keep14+fresh2 (16L)  -> raw .pt   outputs/paperC_pc1_squad_A3/final.pt
#   A2    full-32L LoRA r=160 (MERGED HF dir)-> --base_model outputs/paperC_pc1_squad_A2_lora_r160/merged
#   BASE  untuned OLMo-2-1124-7B reference   -> --base_model $BASE  (raw 32L, no ckpt)
#   A1    (optional) full-FT 32L raw .pt     -> outputs/paperC_pc1_squad_A1/final.pt  (add "A1" to ARMS)
#
#   *** Per-arm LOAD LOGIC differs (verified against the eval scripts) ***
#     - A4/A3/A1 are raw state_dict .pt from train_olmo2_arch_probe2.py. We pass
#       --ckpt <path> --base_model $BASE. load_pruned_model() reads keep_front /
#       n_fresh FROM THE CKPT META (all these ckpts store them), rebuilds the exact
#       (keep+fresh)-layer Olmo2 shell, then STRICT-loads -> zero arch drift. We do
#       NOT pass --keep_front_layers/--n_fresh_layers (meta drives it; a CLI value
#       that disagrees with meta is a HARD ERROR, so leaving it to meta is safest).
#     - A2 is a merged HF dir (LoRA folded into the base) -> --base_model <merged>,
#       NO --ckpt (load_base_model path). Tokenizer is inside the merged dir.
#     - BASE is the raw pretrained 7B -> --base_model $BASE, NO --ckpt.
#
# BASE PROTOCOL (Paper B/C red line, enforced here): chat_template=False (these
# scripts NEVER apply a chat template -- raw "Question:/Answer:" framing),
# add_special_tokens=False i.e. NO BOS (--add_bos 0), LL-based MC argmax (mode=mc,
# greedy only for lambada last-word), greedy decode for the closed-book QA,
# zero-shot, NO retrieval. Nothing to toggle -- these are the script defaults; we
# pass --add_bos 0 explicitly for the record.
#
# SCHEDULING: a shared task-pool of work units = (arm x eval_type x shard). Each
# eval script strides examples[shard_index::num_shards] PER TASK, so NSHARD shards
# = the whole benchmark split. NSHARD defaults to the #GPUs; 8 background workers
# (one pinned GPU each) atomically flock-pop units -> dynamic load balance. After
# the pool drains we --merge each (arm,eval_type) and print an arm x task table.
# FAULT-TOLERANT: a failed unit/arm/benchmark is logged and the chain continues.
#
# NODE / PYTHON: target is .104 (8xH20). Use conda torch-base there (its .venv is
# the broken py3.14 symlink). B200 (.venv torch2.10) also works -- override PYBIN.
#
# USAGE (on .104):
#   cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
#   PYBIN=/opt/conda/envs/torch-base/bin/python \
#   setsid nohup bash scripts/_run_paperC_secondtask_8gpu.sh \
#       > logs/paperC_secondtask_sched.out 2>&1 &
#   # optional overrides:
#   #   ARMS="A4 A3 A2 BASE A1"  GPUS="0 1 2 3 4 5 6 7"  NSHARD=8
#   #   DOWNSTREAM_TASKS="hellaswag arc_challenge ... mmlu boolq"
#   #   CLOSEDBOOK_TASKS="popqa triviaqa nq_open"   EVAL_TYPES="downstream closedbook"
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYBIN:-${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}}"
BASE="${BASE:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"

# --- HF dataset access (downstream/closedbook load_dataset from the hub) --------
# proxy on (needed for first-time downloads); do NOT force HF_HUB_OFFLINE so cache
# misses can still fetch. Override HF_HOME to persist the cache on the project disk.
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export all_proxy="${all_proxy:-http://hy-proxy.woa.com:3128}"
export no_proxy="${no_proxy:-mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# --- config -------------------------------------------------------------------
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a ARMS <<< "${ARMS:-A4 A3 A2 BASE}"
read -r -a EVAL_TYPES <<< "${EVAL_TYPES:-downstream closedbook}"
NSHARD="${NSHARD:-${#GPUS[@]}}"
ADD_BOS="${ADD_BOS:-0}"                      # base protocol: 0 (no BOS)
# downstream MC (LL-based, no generation). mmlu_pro is heavy -> off by default.
DOWNSTREAM_TASKS="${DOWNSTREAM_TASKS:-hellaswag arc_challenge arc_easy piqa winogrande openbookqa mmlu lambada_openai boolq commonsense_qa social_iqa}"
CLOSEDBOOK_TASKS="${CLOSEDBOOK_TASKS:-popqa triviaqa nq_open}"
MC_BATCH="${MC_BATCH:-16}"
MC_MAXLEN="${MC_MAXLEN:-1024}"
GEN_BATCH="${GEN_BATCH:-16}"
GEN_MAXNEW="${GEN_MAXNEW:-32}"
GEN_MAXCTX="${GEN_MAXCTX:-512}"

RESULTS_ROOT="${RESULTS_ROOT:-paperC_secondtask_results}"
LOGDIR="${LOGDIR:-logs}"
mkdir -p "$LOGDIR" "$RESULTS_ROOT"
SUMMARY_LOG="$LOGDIR/paperC_secondtask_summary.log"
QDIR="$LOGDIR/paperC_secondtask_pool"
mkdir -p "$QDIR"
QUEUE="$QDIR/queue.txt"
LOCK="$QDIR/queue.lock"

log(){ echo "[secondtask $(date '+%F %T')] $*"; }

# --- per-arm loader flags (the only place arm-specific load logic lives) ------
DS_TASKS_CSV=$(echo "$DOWNSTREAM_TASKS" | tr ' ' ',')
CB_TASKS_CSV=$(echo "$CLOSEDBOOK_TASKS" | tr ' ' ',')

arm_ckpt(){ # echo the ckpt .pt path for a raw-state_dict arm, or "" for base/merged arms
  case "$1" in
    A4) echo "$PROJECT_ROOT/outputs/paperC_pc1_squad_A4/final.pt" ;;
    A3) echo "$PROJECT_ROOT/outputs/paperC_pc1_squad_A3/final.pt" ;;
    A1) echo "$PROJECT_ROOT/outputs/paperC_pc1_squad_A1/final.pt" ;;
    *)  echo "" ;;
  esac
}
arm_base(){ # echo the --base_model dir (cfg+tokenizer for ckpt arms; the model for base/merged arms)
  case "$1" in
    A2)   echo "$PROJECT_ROOT/outputs/paperC_pc1_squad_A2_lora_r160/merged" ;;
    BASE) echo "$BASE" ;;
    A4|A3|A1) echo "$BASE" ;;   # ckpt arms: base is the cfg+tokenizer source
    *)    echo "" ;;
  esac
}
arm_load_flags(){ # -> "--ckpt X --base_model Y"  or  "--base_model Y"
  local arm="$1" ck; ck="$(arm_ckpt "$arm")"
  local bm; bm="$(arm_base "$arm")"
  if [ -n "$ck" ]; then echo "--ckpt $ck --base_model $bm"; else echo "--base_model $bm"; fi
}
arm_available(){ # 0 if the arm's artefact exists on disk
  local arm="$1" ck; ck="$(arm_ckpt "$arm")"
  if [ -n "$ck" ]; then [ -f "$ck" ]; return; fi
  local bm; bm="$(arm_base "$arm")"; [ -d "$bm" ] || [ -f "$bm/config.json" ]
}

# keep only arms whose artefact is present (fault-tolerant: log + drop the rest)
PRESENT_ARMS=()
for arm in "${ARMS[@]}"; do
  if arm_available "$arm"; then
    PRESENT_ARMS+=("$arm")
  else
    log "WARNING arm=$arm artefact missing (ckpt=$(arm_ckpt "$arm") base=$(arm_base "$arm")) -> SKIP"
  fi
done
if [ "${#PRESENT_ARMS[@]}" -eq 0 ]; then
  log "FATAL no arm artefacts found under $PROJECT_ROOT/outputs -- nothing to eval"; exit 1
fi
log "arms=${PRESENT_ARMS[*]} eval_types=${EVAL_TYPES[*]} nshard=$NSHARD gpus=${GPUS[*]} pybin=$PYBIN"
log "downstream_tasks=$DS_TASKS_CSV"
log "closedbook_tasks=$CB_TASKS_CSV"

# --- 0. warm the HF dataset cache ONCE per eval_type (avoid an 8-way download race)
#        prepare_data returns before the CUDA check -> run on CPU, fault-tolerant.
for et in "${EVAL_TYPES[@]}"; do
  case "$et" in
    downstream) CUDA_VISIBLE_DEVICES="" "$PYBIN" scripts/eval_olmo2_probe2_downstream.py \
                  --prepare_data --tasks "$DS_TASKS_CSV" --base_model "$BASE" \
                  >>"$LOGDIR/paperC_secondtask_prepare.log" 2>&1 || log "prepare downstream failed (cache may exist; continue)" ;;
    closedbook) CUDA_VISIBLE_DEVICES="" "$PYBIN" scripts/eval_olmo2_closedbook_qa.py \
                  --prepare_data --tasks "$CB_TASKS_CSV" --base_model "$BASE" \
                  >>"$LOGDIR/paperC_secondtask_prepare.log" 2>&1 || log "prepare closedbook failed (cache may exist; continue)" ;;
  esac
done

# --- 1. build the shared task-pool: one line per (arm eval_type shard) --------
: > "$QUEUE"
for arm in "${PRESENT_ARMS[@]}"; do
  for et in "${EVAL_TYPES[@]}"; do
    for s in $(seq 0 $((NSHARD-1))); do
      echo "$arm $et $s" >> "$QUEUE"
    done
  done
done
NTASKS=$(wc -l < "$QUEUE")
log "task-pool built: $NTASKS units (${#PRESENT_ARMS[@]} arm x ${#EVAL_TYPES[@]} evaltype x $NSHARD shard)"

pop_task(){ # atomic pop the first queue line (flock)
  local line=""
  exec 9>"$LOCK"; flock 9
  line="$(head -n 1 "$QUEUE")"
  if [ -n "$line" ]; then tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; fi
  flock -u 9; exec 9>&-
  echo "$line"
}

run_unit(){ # $1=gpu  $2=arm  $3=eval_type  $4=shard
  local gpu="$1" arm="$2" et="$3" shard="$4"
  local flags; flags="$(arm_load_flags "$arm")"
  local armlog="$LOGDIR/paperC_secondtask_${arm}.log"
  local oname="${arm}_${et}"
  if [ "$et" = "downstream" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" scripts/eval_olmo2_probe2_downstream.py \
      $flags --tasks "$DS_TASKS_CSV" --add_bos "$ADD_BOS" \
      --num_shards "$NSHARD" --shard_index "$shard" \
      --batch_size "$MC_BATCH" --max_len "$MC_MAXLEN" \
      --output_name "$oname" --results_root "$RESULTS_ROOT" \
      >>"$armlog" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" scripts/eval_olmo2_closedbook_qa.py \
      $flags --tasks "$CB_TASKS_CSV" --add_bos "$ADD_BOS" \
      --num_shards "$NSHARD" --shard_index "$shard" \
      --batch_size "$GEN_BATCH" --max_new_tokens "$GEN_MAXNEW" --max_ctx_len "$GEN_MAXCTX" \
      --output_name "$oname" --results_root "$RESULTS_ROOT" \
      >>"$armlog" 2>&1
  fi
}

worker(){ # $1=gpu : loop popping units until the pool is empty
  local gpu="$1" line arm et shard
  while :; do
    line="$(pop_task)"
    [ -z "$line" ] && break
    read -r arm et shard <<< "$line"
    log "[gpu$gpu] START $arm/$et shard $shard/$NSHARD"
    if run_unit "$gpu" "$arm" "$et" "$shard"; then
      log "[gpu$gpu] DONE  $arm/$et shard $shard"
    else
      log "[gpu$gpu] FAIL  $arm/$et shard $shard (see logs/paperC_secondtask_${arm}.log) -> continue"
    fi
  done
  log "[gpu$gpu] worker drained"
}

# --- 2. fan out one worker per GPU -------------------------------------------
pids=()
for gpu in "${GPUS[@]}"; do
  worker "$gpu" &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p"; done
log "all workers drained -> merging"

# --- 3. merge each (arm, eval_type) ------------------------------------------
for arm in "${PRESENT_ARMS[@]}"; do
  armlog="$LOGDIR/paperC_secondtask_${arm}.log"
  for et in "${EVAL_TYPES[@]}"; do
    oname="${arm}_${et}"
    rdir="$RESULTS_ROOT/$oname"
    if ! ls "$rdir"/shard*of*.json >/dev/null 2>&1; then
      log "WARNING no shard files for $oname -> skip merge"; continue
    fi
    if [ "$et" = "downstream" ]; then
      "$PYBIN" scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$oname" \
        --results_root "$RESULTS_ROOT" >>"$armlog" 2>&1 || log "merge $oname failed"
    else
      "$PYBIN" scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$oname" \
        --results_root "$RESULTS_ROOT" >>"$armlog" 2>&1 || log "merge $oname failed"
    fi
    log "merged $oname"
  done
done

# --- 4. arm x task summary table (for MAIN to backfill) ----------------------
log "=== P-C1 SECOND-TASK SUMMARY -> $SUMMARY_LOG ==="
RESULTS_ROOT="$RESULTS_ROOT" "$PYBIN" - <<'PY' | tee "$SUMMARY_LOG"
import glob, json, os
root = os.environ.get("RESULTS_ROOT", "paperC_secondtask_results")
rows = {}   # arm -> {task -> metricstr}
tasks_seen = []
for sm in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
    name = os.path.basename(os.path.dirname(sm))
    # name = <arm>_<downstream|closedbook>
    if name.endswith("_downstream"): arm, et = name[:-len("_downstream")], "downstream"
    elif name.endswith("_closedbook"): arm, et = name[:-len("_closedbook")], "closedbook"
    else: arm, et = name, "?"
    try:
        s = json.load(open(sm))
    except Exception as e:
        print(f"[warn] cannot read {sm}: {e}"); continue
    arow = rows.setdefault(arm, {})
    for task, t in (s.get("tasks") or {}).items():
        if t.get("skipped"):
            val = "SKIP"
        elif et == "downstream":
            acc = t.get("acc"); accn = t.get("acc_norm")
            val = f"acc={acc:.4f}/accn={accn:.4f}" if acc is not None else "NA"
        else:
            em = t.get("em"); f1 = t.get("f1"); cs = t.get("contains")
            val = f"em={em:.4f}/f1={f1:.4f}/cont={cs:.4f}" if em is not None else "NA"
        col = f"{et[:2]}:{task}"
        arow[col] = val
        if col not in tasks_seen: tasks_seen.append(col)
print(f"[secondtask summary] root={root}\n")
for arm in sorted(rows):
    print(f"### arm={arm}")
    for col in tasks_seen:
        if col in rows[arm]:
            print(f"    {col:28s} {rows[arm][col]}")
    print()
PY
log "=== SECOND-TASK DONE ==="
