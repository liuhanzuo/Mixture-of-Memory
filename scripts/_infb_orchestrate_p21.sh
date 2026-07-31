#!/bin/bash
# P2.1 orchestration on .73: finish Dense qa -> kill wasteful Dense choice-generate
# -> run QCMem + Dense choice LL-MC passes -> score all. Autonomous; logs to
# logs/infb_orchestrate.log; touches /tmp/infb_orch_DONE at the end.
set -u
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python
export PYTHONHASHSEED=0
LOG=logs/infb_orchestrate.log
QADIR=infbench_results/kvdirect_8b
QDIR=infbench_results/qcmem_8b_j12_lora_llmc
DDIR=infbench_results/kvdirect_8b_llmc
rm -f /tmp/infb_orch_DONE
echo "[orch $(date)] START pid=$$" >> $LOG

MATCH="eval_qcmem_infbench.py"

# --- Stage 1: wait for Dense qa to finish (all 8 qa metrics) or procs die / timeout ---
dead=0; waited=0; MAXWAIT=10800   # 3h hard cap
while true; do
  nqa=$(ls $QADIR/longbook_qa_eng_shard*of8_metrics.json 2>/dev/null | wc -l)
  nproc=$(pgrep -fc "$MATCH" || true)
  echo "[orch $(date)] S1 qa_metrics=$nqa live_procs=$nproc waited=${waited}s" >> $LOG
  if [ "$nqa" -ge 8 ]; then echo "[orch] S1 all qa done" >> $LOG; break; fi
  if [ "$nproc" -eq 0 ]; then dead=$((dead+1)); [ "$dead" -ge 2 ] && { echo "[orch] S1 procs gone nqa=$nqa" >> $LOG; break; }; else dead=0; fi
  [ "$waited" -ge "$MAXWAIT" ] && { echo "[orch] S1 TIMEOUT nqa=$nqa" >> $LOG; break; }
  sleep 60; waited=$((waited+60))
done

# --- Stage 2: kill remaining Dense procs (they'd be into the wasteful choice-generate) ---
pkill -9 -f "$MATCH" 2>/dev/null; sleep 10
echo "[orch $(date)] S2 killed remaining $MATCH procs" >> $LOG
$PY scripts/eval_qcmem_infbench.py --score_only --tasks longbook_qa_eng --output_dir $QADIR >> $LOG 2>&1
echo "[orch $(date)] S2 scored Dense qa" >> $LOG

launch_ll () {  # $1=outdir  $2=extra-args (LoRA/baseline)
  local outdir="$1"; shift
  for k in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$k PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    setsid nohup $PY scripts/eval_qcmem_infbench.py \
      --model_path models/Qwen3-8b-local \
      --selector iter_bm25 --topk 12 --iter_rounds 0 --iter_hop_topk 4 \
      --sink_tokens bos --chunk_size 512 --dtype bfloat16 --attn_impl sdpa --seed 42 \
      --tasks longbook_choice_eng --mc_ll --data_dir data/infinitebench \
      --output_dir "$outdir" --num_shards 8 --shard_index $k --device cuda:0 "$@" \
      >logs/$(basename "$outdir")_shard$k.out 2>&1 &
  done
}

wait_ll () {  # $1=outdir  $2=sleep
  local outdir="$1"; local slp="$2"; local w=0; local d=0
  while true; do
    local n=$(ls "$outdir"/longbook_choice_eng_shard*of8_metrics.json 2>/dev/null | wc -l)
    local np=$(pgrep -fc "$MATCH" || true)
    echo "[orch $(date)] wait $outdir metrics=$n live=$np waited=${w}s" >> $LOG
    [ "$n" -ge 8 ] && break
    if [ "$np" -eq 0 ]; then d=$((d+1)); [ "$d" -ge 2 ] && { echo "[orch] $outdir procs gone n=$n" >> $LOG; break; }; else d=0; fi
    [ "$w" -ge 7200 ] && { echo "[orch] $outdir TIMEOUT n=$n" >> $LOG; break; }
    sleep "$slp"; w=$((w+slp))
  done
}

# --- Stage 3: QCMem choice LL-MC (LoRA, resume_j 12, retrieval) ---
echo "[orch $(date)] S3 launch QCMem choice LL" >> $LOG
launch_ll "$QDIR" --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final --resume_j 12
sleep 20; wait_ll "$QDIR" 30
$PY scripts/eval_qcmem_infbench.py --score_only --tasks longbook_choice_eng --output_dir "$QDIR" >> $LOG 2>&1
echo "[orch $(date)] S3 scored QCMem choice LL" >> $LOG

# --- Stage 4: Dense (KV-Direct) choice LL-MC (no LoRA, resume_j 0, no retrieval) ---
echo "[orch $(date)] S4 launch Dense choice LL" >> $LOG
launch_ll "$DDIR" --baseline kvdirect
sleep 20; wait_ll "$DDIR" 45
$PY scripts/eval_qcmem_infbench.py --score_only --tasks longbook_choice_eng --output_dir "$DDIR" >> $LOG 2>&1
echo "[orch $(date)] S4 scored Dense choice LL" >> $LOG

echo "[orch $(date)] ALL DONE" >> $LOG
touch /tmp/infb_orch_DONE
