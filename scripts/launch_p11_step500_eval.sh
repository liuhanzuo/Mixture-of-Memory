#!/usr/bin/env bash
# Launch the P11 step-500 ckpt BABILong eval as a 6-way parallel fanout
# (qa{1,2,5} × {short=0k,1k,2k,4k, long=8k,16k,32k}) on 6 GPUs of the remote node.
#
# Caller is expected to have already cd'd into this repo root via the SSH wrapper.
# Same recipe / args as the step4500 eval that produced
# outputs/eval_phase1b_p11_step4500_20260517_040951/.

set -u

TS="${TS:-$(date +%Y%m%d_%H%M)}"
CKPT="outputs/babilong_sft_phase11_fsdp_full/mem_space_adapter_step000500.pt"
ACFG="outputs/babilong_sft_phase11_fsdp_full/adapter_config.json"
MODEL="models/Meta-Llama-3-8B-Instruct"
RESULTS="outputs/eval_p11_step500"
LOGDIR="logs"

mkdir -p "$RESULTS" "$LOGDIR"

run_cell () {
  local task="$1"          # qa1 / qa2 / qa5
  local split="$2"         # short / long
  local lengths="$3"       # "0k 1k 2k 4k" or "8k 16k 32k"
  local gpu="$4"           # cuda:N
  local out="${task}_${split}"
  local log="${LOGDIR}/eval_p11step500_${task}_${split}_${TS}.log"

  CUDA_VISIBLE_DEVICES="$gpu" \
  python scripts/run_babilong_mem_space.py \
      --model_path        "$MODEL" \
      --checkpoint        "$CKPT" \
      --adapter_config    "$ACFG" \
      --results_folder    "$RESULTS" \
      --output_name       "p11step500_${out}" \
      --tasks             "$task" \
      --lengths           $lengths \
      --chunk_size        4096 \
      --max_new_tokens    20 \
      --limit             100 \
      --device            cuda:0 \
      --use_chat_template \
      > "$log" 2>&1 &

  echo "[launcher] task=${task} split=${split} gpu=${gpu} pid=$! log=${log}"
}

# Six parallel workers, one GPU each (GPUs 0..5)
run_cell qa1 short "0k 1k 2k 4k" 0
run_cell qa2 short "0k 1k 2k 4k" 1
run_cell qa5 short "0k 1k 2k 4k" 2
run_cell qa1 long  "8k 16k 32k"  3
run_cell qa2 long  "8k 16k 32k"  4
run_cell qa5 long  "8k 16k 32k"  5

wait
echo "[launcher] all 6 workers exited"
