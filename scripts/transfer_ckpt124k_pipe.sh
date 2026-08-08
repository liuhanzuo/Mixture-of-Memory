#!/bin/bash
# Transfer step124000.pt from .73 (zwfy6) to .21 (wzc1) via local pipe
set -e

PASS_73=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_h20_853573.txt
PASS_21=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_b200_19021.txt
LOG=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/logs/transfer_ckpt124k_to_21.log
CKPT_ZWFY6=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt
CKPT_WZC1=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt

echo "=== step124000.pt pipe transfer started at $(date) ===" | tee $LOG
echo "Source: .73:$CKPT_ZWFY6 -> Dest: .21:$CKPT_WZC1" | tee -a $LOG

# Check source size on .73
SRC_SIZE=$(sshpass -f $PASS_73 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.85.35.73 \
  "stat -c %s $CKPT_ZWFY6 2>/dev/null || echo 0")
echo "source size: $SRC_SIZE bytes" | tee -a $LOG

# Stream via pipe
sshpass -f $PASS_73 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.85.35.73 \
  "cat $CKPT_ZWFY6" | \
sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
  "cat > $CKPT_WZC1"
EXIT=$?
echo "pipe exit=$EXIT at $(date)" | tee -a $LOG

if [ $EXIT -eq 0 ]; then
  REMOTE_SIZE=$(sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
    "stat -c %s $CKPT_WZC1 2>/dev/null || echo 0")
  echo "size check: src=$SRC_SIZE dst=$REMOTE_SIZE" | tee -a $LOG
  if [ "$REMOTE_SIZE" = "$SRC_SIZE" ]; then
    echo "SIZE_OK" | tee -a $LOG
    # Compute md5 for final verification
    echo "Computing md5 checksums..." | tee -a $LOG
    LOCAL_MD5=$(sshpass -f $PASS_73 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.85.35.73 \
      "md5sum $CKPT_ZWFY6 | awk '{print \$1}'")
    REMOTE_MD5=$(sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
      "md5sum $CKPT_WZC1 | awk '{print \$1}'")
    echo "src_md5=$LOCAL_MD5 dst_md5=$REMOTE_MD5" | tee -a $LOG
    if [ "$LOCAL_MD5" = "$REMOTE_MD5" ]; then
      echo "MD5_OK - transfer verified" | tee -a $LOG
    else
      echo "MD5_MISMATCH - needs retransfer" | tee -a $LOG
      exit 1
    fi
  else
    echo "SIZE_MISMATCH - needs retransfer" | tee -a $LOG
    exit 1
  fi
else
  echo "Transfer failed with exit=$EXIT" | tee -a $LOG
  exit 1
fi

echo "=== step124000.pt transfer complete at $(date) ===" | tee -a $LOG
