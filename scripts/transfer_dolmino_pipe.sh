#!/bin/bash
# Transfer dolmino_now15b.npy from .73 to .21 via local pipe
# No intermediate storage needed - streams directly
set -e

PASS_73=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_h20_853573.txt
PASS_21=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_b200_19021.txt
LOG=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/logs/transfer_dolmino_to_21.log
EXPECTED_SIZE=126907244672

echo "=== dolmino pipe transfer started at $(date) ===" | tee $LOG
echo "Source: .73:/dev/shm/dolmino_now15b.npy -> Dest: .21:/dev/shm/dolmino_now15b_zwfy6.npy" | tee -a $LOG

# Stream via pipe
sshpass -f $PASS_73 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.85.35.73 \
  "cat /dev/shm/dolmino_now15b.npy" | \
sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
  "cat > /dev/shm/dolmino_now15b_zwfy6.npy"
EXIT=$?
echo "pipe exit=$EXIT at $(date)" | tee -a $LOG

if [ $EXIT -eq 0 ]; then
  # Verify file size
  REMOTE_SIZE=$(sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
    "stat -c %s /dev/shm/dolmino_now15b_zwfy6.npy 2>/dev/null || echo 0")
  echo "size check: expected=$EXPECTED_SIZE actual=$REMOTE_SIZE" | tee -a $LOG
  if [ "$REMOTE_SIZE" = "$EXPECTED_SIZE" ]; then
    echo "SIZE_OK" | tee -a $LOG
  else
    echo "SIZE_MISMATCH - needs retransfer" | tee -a $LOG
    exit 1
  fi
else
  echo "Transfer failed with exit=$EXIT" | tee -a $LOG
  exit 1
fi

echo "=== dolmino transfer complete at $(date) ===" | tee -a $LOG
