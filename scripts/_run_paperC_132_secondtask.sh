#!/usr/bin/env bash
# ============================================================================
# Paper C P-C1 task #132 — SECOND-TASK CAPABILITY EVAL (runner, node .73 8xH20)
#
# WHY: the P-C1 headline (SQuAD dev EM: A4 0.2930 vs A3 0.2605, p=7.3e-4) rests on a
# SINGLE format-dominated benchmark. Format-SFT can lift SQuAD EM without restoring
# any knowledge/reasoning. This scores the SAME 4 arms on CAPABILITY-sensitive
# benchmarks (zero-shot LL-MC + closed-book parametric-knowledge QA) so we can tell a
# real capability gap from a formatting artefact, i.e. does A4 > A3 SURVIVE off SQuAD?
#
# Derived from scripts/_run_paperC_secondtask_8gpu.sh (commit cd0f527, never run) with
# three corrections for .73:
#   1. do NOT override HF_HOME. The project-disk .hf_cache does NOT contain mmlu /
#      hellaswag / arc / popqa / triviaqa / nq_open, but .73's DEFAULT cache
#      (/root/.cache/huggingface) does (verified: all 14 splits load, 0 skips). Pointing
#      HF_HOME at the project disk would force 8 workers into a re-download race.
#   2. --save_per_example on the MC harness -> per_example_{task}.jsonl, which unlocks
#      the paired McNemar / paired bootstrap on A4-vs-A3 that the SQuAD slice used
#      (the closed-book harness always dumps per-example rows).
#   3. post-run arch verification: assert each arm's LOADED num_hidden_layers matches
#      its construction (A4/A3 = 16L, A2/BASE = 32L) straight out of the shard metas,
#      so an identical-looking row can never come from a silently-wrong shell.
#
# ARMS (artefact-gated; a missing arm is logged and dropped, never substituted)
#   A4    freeze-graft keep14 frozen + fresh2, 16L  -> --ckpt outputs/paperC_pc1_squad_A4/final.pt
#   A3    from-scratch 16L random-init, all-train   -> --ckpt outputs/paperC_pc1_squad_A3/final.pt
#   A2    full 32L + LoRA r=160 param-matched       -> --base_model .../A2_lora_r160/merged
#   BASE  raw OLMo-2-1124-7B 32L, NO SFT (ref)      -> --base_model $BASE
#   A1    full-FT 32L was NEVER trained (H20 OOM; only arch_meta.json on disk) -> absent.
#
#   *** per-arm LOAD LOGIC (verified against eval_olmo2_probe2_ppl.load_* ) ***
#     - A4/A3 are raw {'model_state':...} .pt from train_olmo2_arch_probe2.py. Pass
#       --ckpt <pt> --base_model $BASE; load_pruned_model() reads keep_front/n_fresh
#       FROM THE CKPT META, rebuilds that exact (keep+fresh)-layer Olmo2 shell, then
#       STRICT-loads (missing/unexpected == [] asserted) -> zero arch drift. We do NOT
#       pass --keep_front_layers/--n_fresh_layers: meta drives it and a disagreeing CLI
#       value is a hard error, so letting meta drive is strictly safer.
#     - A2 is a merged HF dir (LoRA folded into the 32L base) -> --base_model <merged>,
#       NO --ckpt (load_base_model path).
#     - BASE is the raw pretrained 32L 7B -> --base_model $BASE, NO --ckpt.
#     - Tokenizer always comes from --base_model. VERIFIED on .73 that the A2-merged
#       tokenizer and the base tokenizer encode identically (same ids on the MMLU
#       stem / " A".." D" letters / "Question:..Answer:" prompt, same vocab 100278,
#       same bos/eos/pad) although the re-serialised tokenizer.json md5 differs.
#
# BASE PROTOCOL (project red line, enforced + recorded here)
#   chat_template=False  — these harnesses NEVER apply a chat template (raw flan-style
#     MC stems and a fixed "Question: {q}\nAnswer:" QA prompt). OLMo-2 is a BASE LM
#     with no SFT/RL, so a chat template would be unfair; ALL paper numbers are chat=False.
#   add_bos=0            — add_special_tokens=False, no BOS (OLMo-2 lm-eval convention).
#   MC = likelihood-based — sum log-prob of the teacher-forced continuation, argmax over
#     candidates (acc) + char-length-normalised (acc_norm). NO generation. Matches
#     Paper B's base-protocol MC convention. lambada_openai is the one greedy last-word
#     task (is_greedy), by harness design.
#   closed-book QA = greedy (do_sample=False, num_beams=1), zero-shot, NO retrieval,
#     first line of the completion, SQuAD-normalised em/f1 + PopQA-style `contains`.
#   IDENTICAL flags for all 4 arms -> the cross-arm comparison is valid.
#
# SCHEDULING: shared flock task-pool of (arm x eval_type x shard) units. Each harness
# strides examples[shard_index::num_shards] PER TASK, so NSHARD shards == the whole
# split. 8 workers (one pinned GPU each) atomically pop units -> dynamic load balance,
# no idle card. Closed-book units are queued FIRST (longest-processing-time-first: they
# are generation-bound, the MC units are short) so the tail fills with cheap work.
# Fault-tolerant: a failed unit is logged and the pool keeps draining.
#
# USAGE (on .73; PROJECT_ROOT is the zwfy6 disk that holds the P-C1 ckpts):
#   cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
#   setsid nohup bash scripts/_run_paperC_132_secondtask.sh \
#       > logs/paperC_132_secondtask.log 2>&1 &
#   # overrides: ARMS="A4 A3 A2 BASE" GPUS="0 1 2 3 4 5 6 7" NSHARD=8
#   #            DOWNSTREAM_TASKS="..." CLOSEDBOOK_TASKS="..." EVAL_TYPES="..."
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT" || { echo "FATAL cannot cd $PROJECT_ROOT"; exit 1; }
PYBIN="${PYBIN:-${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}}"
BASE="${BASE:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"

# --- HF hub access. Datasets were pre-warmed into the DEFAULT cache; do NOT move
#     HF_HOME (see header note 1). Proxy stays on for any residual cache miss.
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export no_proxy="${no_proxy:-mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# --- config -------------------------------------------------------------------
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a ARMS <<< "${ARMS:-A4 A3 A2 BASE}"
read -r -a EVAL_TYPES <<< "${EVAL_TYPES:-closedbook downstream}"   # slow first (LPT)
NSHARD="${NSHARD:-${#GPUS[@]}}"
ADD_BOS="${ADD_BOS:-0}"                       # base protocol: no BOS
# LL-based MC. mmlu = the knowledge headline; mmlu_pro left off (heavy, low marginal value).
DOWNSTREAM_TASKS="${DOWNSTREAM_TASKS:-mmlu hellaswag arc_challenge arc_easy piqa winogrande openbookqa lambada_openai boolq commonsense_qa social_iqa}"
CLOSEDBOOK_TASKS="${CLOSEDBOOK_TASKS:-popqa triviaqa nq_open}"
MC_BATCH="${MC_BATCH:-16}"                    # == Paper B's tested H20 setting
MC_MAXLEN="${MC_MAXLEN:-1024}"
GEN_BATCH="${GEN_BATCH:-16}"
GEN_MAXNEW="${GEN_MAXNEW:-32}"
GEN_MAXCTX="${GEN_MAXCTX:-512}"
SAVE_PER_EXAMPLE="${SAVE_PER_EXAMPLE:-1}"     # paired McNemar / bootstrap on A4-vs-A3

RESULTS_ROOT="${RESULTS_ROOT:-paperC_secondtask_results}"
LOGDIR="${LOGDIR:-logs}"
mkdir -p "$LOGDIR" "$RESULTS_ROOT"
SUMMARY_LOG="$LOGDIR/paperC_132_secondtask_summary.log"
QDIR="$LOGDIR/paperC_132_pool"
mkdir -p "$QDIR"
QUEUE="$QDIR/queue.txt"
LOCK="$QDIR/queue.lock"

log(){ echo "[pc132 $(date '+%F %T')] $*"; }

DS_TASKS_CSV=$(echo "$DOWNSTREAM_TASKS" | tr ' ' ',')
CB_TASKS_CSV=$(echo "$CLOSEDBOOK_TASKS" | tr ' ' ',')

# --- per-arm loader flags (the ONLY place arm-specific load logic lives) -------
arm_ckpt(){   # raw-state_dict .pt path, or "" for base/merged arms
  case "$1" in
    A4) echo "$PROJECT_ROOT/outputs/paperC_pc1_squad_A4/final.pt" ;;
    A3) echo "$PROJECT_ROOT/outputs/paperC_pc1_squad_A3/final.pt" ;;
    A1) echo "$PROJECT_ROOT/outputs/paperC_pc1_squad_A1/final.pt" ;;
    *)  echo "" ;;
  esac
}
arm_base(){   # --base_model dir (cfg+tokenizer for ckpt arms; the model for base/merged)
  case "$1" in
    A2)       echo "$PROJECT_ROOT/outputs/paperC_pc1_squad_A2_lora_r160/merged" ;;
    BASE)     echo "$BASE" ;;
    A4|A3|A1) echo "$BASE" ;;
    *)        echo "" ;;
  esac
}
arm_expect_layers(){ case "$1" in A4|A3) echo 16 ;; A2|BASE|A1) echo 32 ;; *) echo 0 ;; esac; }
arm_load_flags(){
  local arm="$1" ck bm; ck="$(arm_ckpt "$arm")"; bm="$(arm_base "$arm")"
  if [ -n "$ck" ]; then echo "--ckpt $ck --base_model $bm"; else echo "--base_model $bm"; fi
}
arm_available(){
  local arm="$1" ck bm; ck="$(arm_ckpt "$arm")"
  if [ -n "$ck" ]; then [ -f "$ck" ]; return; fi
  bm="$(arm_base "$arm")"; [ -f "$bm/config.json" ]
}

PRESENT_ARMS=()
for arm in "${ARMS[@]}"; do
  if arm_available "$arm"; then PRESENT_ARMS+=("$arm")
  else log "WARNING arm=$arm artefact MISSING (ckpt=$(arm_ckpt "$arm") base=$(arm_base "$arm")) -> SKIP (never substituted)"; fi
done
[ "${#PRESENT_ARMS[@]}" -eq 0 ] && { log "FATAL no arm artefacts under $PROJECT_ROOT/outputs"; exit 1; }

log "node=$(hostname) root=$PROJECT_ROOT pybin=$PYBIN"
log "arms=${PRESENT_ARMS[*]} eval_types=${EVAL_TYPES[*]} nshard=$NSHARD gpus=${GPUS[*]}"
log "protocol: chat_template=False add_bos=$ADD_BOS MC=likelihood(acc/acc_norm) QA=greedy zero-shot no-retrieval"
log "mc: batch=$MC_BATCH max_len=$MC_MAXLEN tasks=$DS_TASKS_CSV"
log "gen: batch=$GEN_BATCH max_new=$GEN_MAXNEW max_ctx=$GEN_MAXCTX tasks=$CB_TASKS_CSV"
log "results_root=$RESULTS_ROOT save_per_example=$SAVE_PER_EXAMPLE"

# --- build the shared pool: one line per (arm eval_type shard) ----------------
: > "$QUEUE"
for et in "${EVAL_TYPES[@]}"; do
  for arm in "${PRESENT_ARMS[@]}"; do
    for s in $(seq 0 $((NSHARD-1))); do echo "$arm $et $s" >> "$QUEUE"; done
  done
done
log "task-pool: $(wc -l < "$QUEUE") units (${#PRESENT_ARMS[@]} arm x ${#EVAL_TYPES[@]} evaltype x $NSHARD shard)"

pop_task(){
  local line=""
  exec 9>"$LOCK"; flock 9
  line="$(head -n 1 "$QUEUE")"
  [ -n "$line" ] && { tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; }
  flock -u 9; exec 9>&-
  echo "$line"
}

run_unit(){ # $1=gpu $2=arm $3=eval_type $4=shard
  local gpu="$1" arm="$2" et="$3" shard="$4"
  local flags; flags="$(arm_load_flags "$arm")"
  local armlog="$LOGDIR/paperC_132_${arm}_${et}.log"
  local oname="${arm}_${et}"
  if [ "$et" = "downstream" ]; then
    local pe=""; [ "$SAVE_PER_EXAMPLE" = "1" ] && pe="--save_per_example"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" scripts/eval_olmo2_probe2_downstream.py \
      $flags --tasks "$DS_TASKS_CSV" --add_bos "$ADD_BOS" \
      --num_shards "$NSHARD" --shard_index "$shard" \
      --batch_size "$MC_BATCH" --max_len "$MC_MAXLEN" $pe \
      --output_name "$oname" --results_root "$RESULTS_ROOT" >>"$armlog" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" scripts/eval_olmo2_closedbook_qa.py \
      $flags --tasks "$CB_TASKS_CSV" --add_bos "$ADD_BOS" \
      --num_shards "$NSHARD" --shard_index "$shard" \
      --batch_size "$GEN_BATCH" --max_new_tokens "$GEN_MAXNEW" --max_ctx_len "$GEN_MAXCTX" \
      --output_name "$oname" --results_root "$RESULTS_ROOT" >>"$armlog" 2>&1
  fi
}

worker(){
  local gpu="$1" line arm et shard
  while :; do
    line="$(pop_task)"; [ -z "$line" ] && break
    read -r arm et shard <<< "$line"
    log "[gpu$gpu] START $arm/$et shard$shard/$NSHARD"
    if run_unit "$gpu" "$arm" "$et" "$shard"; then log "[gpu$gpu] DONE  $arm/$et shard$shard"
    else log "[gpu$gpu] FAIL  $arm/$et shard$shard (see $LOGDIR/paperC_132_${arm}_${et}.log) -> continue"; fi
  done
  log "[gpu$gpu] drained"
}

pids=()
for gpu in "${GPUS[@]}"; do worker "$gpu" & pids+=("$!"); done
for p in "${pids[@]}"; do wait "$p"; done
log "all workers drained -> merge"

# --- merge each (arm, eval_type) ---------------------------------------------
for arm in "${PRESENT_ARMS[@]}"; do
  for et in "${EVAL_TYPES[@]}"; do
    oname="${arm}_${et}"; rdir="$RESULTS_ROOT/$oname"
    armlog="$LOGDIR/paperC_132_${arm}_${et}.log"
    ls "$rdir"/shard*of*.json >/dev/null 2>&1 || { log "WARNING no shards for $oname -> skip merge"; continue; }
    if [ "$et" = "downstream" ]; then
      "$PYBIN" scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$oname" \
        --results_root "$RESULTS_ROOT" >>"$armlog" 2>&1 || log "merge $oname FAILED"
    else
      "$PYBIN" scripts/eval_olmo2_closedbook_qa.py --merge --output_name "$oname" \
        --results_root "$RESULTS_ROOT" >>"$armlog" 2>&1 || log "merge $oname FAILED"
    fi
    log "merged $oname"
  done
done

# --- arch verification + arm x task summary ----------------------------------
log "=== P-C1 SECOND-TASK SUMMARY -> $SUMMARY_LOG ==="
EXPECT_LAYERS=""
for arm in "${PRESENT_ARMS[@]}"; do EXPECT_LAYERS="$EXPECT_LAYERS$arm=$(arm_expect_layers "$arm") "; done
RESULTS_ROOT="$RESULTS_ROOT" EXPECT_LAYERS="$EXPECT_LAYERS" "$PYBIN" - <<'PY' | tee "$SUMMARY_LOG"
import glob, json, os
root = os.environ.get("RESULTS_ROOT", "paperC_secondtask_results")
expect = {}
for tok in os.environ.get("EXPECT_LAYERS", "").split():
    if "=" in tok:
        k, v = tok.split("=", 1)
        expect[k] = int(v)

rows, cols = {}, []
arch = {}
for sm in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
    name = os.path.basename(os.path.dirname(sm))
    for suf in ("downstream", "closedbook"):
        if name.endswith("_" + suf):
            arm, et = name[: -(len(suf) + 1)], suf
            break
    else:
        arm, et = name, "?"
    try:
        s = json.load(open(sm))
    except Exception as e:
        print(f"[warn] unreadable {sm}: {e}"); continue
    meta = s.get("meta") or {}
    arch.setdefault(arm, []).append((et, meta.get("num_hidden_layers"), meta.get("mode"),
                                     s.get("n_shards"), s.get("add_bos")))
    arow = rows.setdefault(arm, {})
    for task, t in (s.get("tasks") or {}).items():
        if t.get("skipped"):
            val, n = "SKIPPED", 0
        elif et == "downstream":
            a, an = t.get("acc"), t.get("acc_norm")
            val = f"acc={a:.4f} accn={an:.4f}" if a is not None else "NA"
            n = t.get("n", 0)
        else:
            em, f1, ct = t.get("em"), t.get("f1"), t.get("contains")
            val = f"em={em:.4f} f1={f1:.4f} cont={ct:.4f}" if em is not None else "NA"
            n = t.get("n", 0)
        col = f"{et[:2]}:{task}"
        arow[col] = (val, n)
        if col not in cols: cols.append(col)

print(f"[pc132] results_root={root}\n")
print("=== ARCH / PROTOCOL VERIFICATION (loaded layers must match construction) ===")
bad = []
for arm in sorted(arch):
    for et, nl, mode, nsh, bos in sorted(arch[arm]):
        exp = expect.get(arm)
        okl = "OK" if (exp is None or nl == exp) else f"*** MISMATCH expect {exp} ***"
        if exp is not None and nl != exp: bad.append((arm, et, nl, exp))
        print(f"  {arm:5s} {et:10s} loaded_layers={nl} expect={exp} {okl}  "
              f"mode={mode} n_shards={nsh} add_bos={bos}")
print("  ARCH_CHECK: " + ("ALL OK" if not bad else f"FAILED {bad}"))

print("\n=== ARM x BENCHMARK ===")
for arm in sorted(rows):
    print(f"### {arm}")
    for col in cols:
        if col in rows[arm]:
            val, n = rows[arm][col]
            print(f"    {col:24s} n={n:<6d} {val}")
    print()

# identical-row red flag: two architecturally different arms scoring bit-identical
print("=== DUPLICATE-ROW RED-FLAG CHECK (identical scores across arms => ckpt not loading) ===")
arms = sorted(rows)
dup = False
for i in range(len(arms)):
    for j in range(i + 1, len(arms)):
        a, b = arms[i], arms[j]
        shared = [c for c in cols if c in rows[a] and c in rows[b]]
        if shared and all(rows[a][c][0] == rows[b][c][0] for c in shared):
            print(f"  *** {a} and {b} identical on all {len(shared)} shared cells ***"); dup = True
print("  DUP_CHECK: " + ("clean (arms differ)" if not dup else "SUSPICIOUS"))
PY
log "=== PC132 DONE ==="
