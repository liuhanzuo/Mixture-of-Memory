#!/usr/bin/env bash
# S5 auto-passkey driver. The 2k training occupies BOTH Group-A nodes (16 GPU)
# until step 2000, so passkey eval (8-shard on local node) waits until training
# fully exits + GPUs free, then evals step1000/2000 sequentially (NOT step2000
# only — judge at step1000/2000 per overtraining rule; here 2000 is the cap).
# Uses scripts/eval_landmark_S2_passkey.sh (checkpoint-agnostic, S0/S2 protocol).
# NOTE: the eval loads the ckpt via LlamaForCausalLM.from_pretrained which reads
# the ckpt's OWN llama_mem.py? No — eval imports from external/landmark-attention.
# S5 single_layer_mem is saved in ckpt config.json; the eval's llama_mem must
# support single_layer_mem. The S5 ckpts carry config.single_layer_mem=16, and
# the eval-side llama_mem (S4b tree) does NOT have that arg -> would ignore it.
# So we eval using the S5 TREE's llama_mem by pointing PYTHONPATH at it.
set -uo pipefail
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
CKDIR=external/landmark_ckpts/landmark_S5_L16_singlelayer

echo "=== [$(date '+%F %T')] waiting for S5 2k training to finish (checkpoint-2000 + GPUs free) ==="
while true; do
  IDX="$CKDIR/checkpoint-2000/pytorch_model.bin.index.json"
  nproc_train=$(pgrep -f "landmark_venv/bin/python3 -u train.py" | wc -l)
  if [ -f "$IDX" ] && [ "$nproc_train" -eq 0 ]; then
    last=$(ls "$CKDIR/checkpoint-2000"/pytorch_model-*-of-*.bin 2>/dev/null | sort | tail -1)
    if [ -n "$last" ]; then
      s1=$(stat -c %s "$last" 2>/dev/null); sleep 20; s2=$(stat -c %s "$last" 2>/dev/null)
      if [ "$s1" = "$s2" ] && [ -n "$s1" ] && [ "$s1" -gt 1000000 ]; then
        echo "=== [$(date '+%F %T')] S5 training done, ckpt-2000 stable, GPUs free ==="
        break
      fi
    fi
  fi
  sleep 60
done

for st in 1000 2000; do
  CK="$CKDIR/checkpoint-$st"
  if [ ! -f "$CK/pytorch_model.bin.index.json" ]; then
    echo "=== [$(date '+%F %T')] WARN: $CK missing, skip ==="
    continue
  fi
  echo "=== [$(date '+%F %T')] eval S5 step$st (single_layer_mem=16 via S5-tree llama_mem) ==="
  CKPT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/$CK" \
    CK_NAME="s5_L16_singlelayer_step$st" \
    bash scripts/eval_landmark_S5_passkey.sh 2>&1 | tail -25
done
echo "=== [$(date '+%F %T')] ALL S5 PASSKEY EVAL DONE ==="
