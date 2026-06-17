#!/usr/bin/env bash
# Full sliding-window PPL matrix: {pg19,proofpile,codeparrot} x {base,mem_space}
# Run on one GPU sequentially. Each combo writes a JSON under ppl_results/.
set -u
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export WANDB_MODE=offline
PY=/opt/conda/envs/torch-base/bin/python
MODEL=models/Meta-Llama-3-8B
ADCFG=outputs/mem_space_p11_chunk1024_deltarule_normreadout/adapter_config.json
CKPT=outputs/mem_space_p11_chunk1024_deltarule_normreadout/mem_space_adapter.pt
SEQ=32768
MAXTOK=${MAXTOK:-1000000}
OUT=ppl_results
mkdir -p $OUT logs/sliding_ppl

declare -A DATAPATH=( [pg19]=data/pg19_real_llama3_noeos.npy [proofpile]=data/proofpile_llama3_noeos.npy [codeparrot]=data/codeparrot_llama3_noeos.npy )
declare -A SKIP=( [pg19]=0 [proofpile]=0 [codeparrot]=0 )

FILTER='tvm_ffi|UserWarning|info = core|WARNING. Field|duplicates an ancestor|Loading weights'

run_one() {
  local data=$1 mode=$2
  local dp=${DATAPATH[$data]} sk=${SKIP[$data]}
  local log=logs/sliding_ppl/${data}_${mode}.log
  local js=$OUT/${data}_${mode}.json
  echo "===== RUN data=$data mode=$mode skip=$sk =====" | tee -a $log
  if [ "$mode" = "base" ]; then
    CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_sliding_ppl.py \
      --data $data --data_path $dp --model_path $MODEL \
      --seq_length $SEQ --window 8192 --stride 4096 \
      --skip_tokens $sk --max_tokens $MAXTOK --gpu 0 \
      --output_json $js 2>&1 | grep -vE "$FILTER" | tee -a $log
  else
    CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_sliding_ppl.py \
      --data $data --data_path $dp --model_path $MODEL \
      --adapter_config $ADCFG --checkpoint $CKPT \
      --seq_length $SEQ --chunk_size 1024 \
      --skip_tokens $sk --max_tokens $MAXTOK --gpu 0 \
      --output_json $js 2>&1 | grep -vE "$FILTER" | tee -a $log
  fi
}

for data in proofpile codeparrot pg19; do
  for mode in base mem_space; do
    run_one $data $mode
  done
done
echo "[matrix] ALL DONE"
