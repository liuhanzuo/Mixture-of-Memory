#!/usr/bin/env bash
# S4b auto-passkey driver. The 3k training occupies BOTH Group-A nodes (16 GPU)
# until step 3000, so passkey eval (8-shard on local node) must wait until
# training fully exits and frees the GPUs. Then eval all 3 ckpts sequentially.
# Uses scripts/eval_landmark_S2_passkey.sh (checkpoint-agnostic, S0/S2 protocol).
# learned_block_gate is saved in each ckpt's config.json so from_pretrained
# rebuilds + loads the trained gate.
set -uo pipefail
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
CKDIR=external/landmark_ckpts/landmark_S4b_learnedgate

echo "=== [$(date '+%F %T')] waiting for 3k training to finish (checkpoint-3000 + GPUs free) ==="
while true; do
  # training done when checkpoint-3000 fully written AND no train.py procs hold GPUs
  IDX="$CKDIR/checkpoint-3000/pytorch_model.bin.index.json"
  nproc_train=$(pgrep -f "landmark_venv/bin/python3 -u train.py" | wc -l)
  if [ -f "$IDX" ] && [ "$nproc_train" -eq 0 ]; then
    # stability check on last shard
    last=$(ls "$CKDIR/checkpoint-3000"/pytorch_model-*-of-*.bin 2>/dev/null | sort | tail -1)
    if [ -n "$last" ]; then
      s1=$(stat -c %s "$last" 2>/dev/null); sleep 20; s2=$(stat -c %s "$last" 2>/dev/null)
      if [ "$s1" = "$s2" ] && [ -n "$s1" ] && [ "$s1" -gt 1000000 ]; then
        echo "=== [$(date '+%F %T')] training done, ckpt-3000 stable, GPUs free ==="
        break
      fi
    fi
  fi
  sleep 60
done

for st in 1000 2000 3000; do
  CK="$CKDIR/checkpoint-$st"
  if [ ! -f "$CK/pytorch_model.bin.index.json" ]; then
    echo "=== [$(date '+%F %T')] WARN: $CK missing, skip ==="
    continue
  fi
  echo "=== [$(date '+%F %T')] eval S4b step$st ==="
  CKPT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/$CK" \
    CK_NAME="s4b_learnedgate_step$st" \
    bash scripts/eval_landmark_S2_passkey.sh 2>&1 | tail -20
done
echo "=== [$(date '+%F %T')] ALL S4b PASSKEY EVAL DONE ==="
