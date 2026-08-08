#!/bin/bash
echo "[DISABLED 2026-08-08] .21 已改分配给 SparseForge (用户改优先级 STAND DOWN); Paper B keep10 在 .82 正常跑, keep12 在 .104 正常跑, 无需迁移. 如需恢复请人工删除此 guard." >&2
exit 1
# Self-restarting wrapper for step124000.pt transfer
# Retries until MD5_OK is confirmed on .21

PASS_73=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_h20_853573.txt
PASS_21=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_b200_19021.txt
LOG=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/logs/transfer_ckpt124k_to_21.log
CKPT_ZWFY6=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt
CKPT_WZC1=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt

echo "=== step124000.pt self-restarting wrapper started at $(date) ===" | tee $LOG

SRC_SIZE=$(sshpass -f $PASS_73 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.85.35.73 \
  "stat -c %s $CKPT_ZWFY6 2>/dev/null || echo 0" 2>/dev/null || echo 0)
echo "Source size: $SRC_SIZE bytes" | tee -a $LOG

MAX_ATTEMPTS=20
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "Attempt $ATTEMPT at $(date)" | tee -a $LOG

  # Remove partial
  sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
    "rm -f $CKPT_WZC1 2>/dev/null; echo cleared" >> $LOG 2>&1

  # Stream
  sshpass -f $PASS_73 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.85.35.73 \
    "cat $CKPT_ZWFY6" | \
  sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
    "cat > $CKPT_WZC1"
  EXIT=$?
  echo "stream exit=$EXIT at $(date)" | tee -a $LOG

  FINAL_SIZE=$(sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
    "stat -c %s $CKPT_WZC1 2>/dev/null || echo 0" 2>/dev/null || echo 0)

  if [ "$FINAL_SIZE" = "$SRC_SIZE" ]; then
    echo "SIZE_OK ($FINAL_SIZE bytes)" | tee -a $LOG
    # MD5 check
    echo "Computing md5 checksums..." | tee -a $LOG
    SRC_MD5=$(sshpass -f $PASS_73 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.85.35.73 \
      "md5sum $CKPT_ZWFY6 | awk '{print \$1}'")
    DST_MD5=$(sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
      "md5sum $CKPT_WZC1 | awk '{print \$1}'")
    echo "src_md5=$SRC_MD5 dst_md5=$DST_MD5" | tee -a $LOG
    if [ "$SRC_MD5" = "$DST_MD5" ]; then
      echo "MD5_OK - transfer verified!" | tee -a $LOG
      exit 0
    else
      echo "MD5_MISMATCH - will retry" | tee -a $LOG
    fi
  else
    echo "INCOMPLETE ($FINAL_SIZE / $SRC_SIZE), retrying in 30s..." | tee -a $LOG
  fi

  sleep 30
done

echo "Max attempts reached" | tee -a $LOG
exit 1
