#!/bin/bash
# Self-restarting wrapper for dolmino transfer
# Retries until SIZE_OK is confirmed on .21
# USER INSTRUCTION 2026-08-08: B200用于resume; 此传输为用户授权

PASS_73=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_h20_853573.txt
PASS_21=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_b200_19021.txt
LOG=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/logs/transfer_dolmino_to_21.log
EXPECTED_SIZE=126907244672

echo "=== dolmino self-restarting wrapper started at $(date) ===" | tee $LOG

MAX_ATTEMPTS=20
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "Attempt $ATTEMPT at $(date)" | tee -a $LOG

  # Remove any blocker directory
  export PATH=/opt/conda/bin:$PATH
  sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
    "rm -rf /dev/shm/dolmino_now15b_zwfy6.npy 2>/dev/null; echo cleared" >> $LOG 2>&1

  # Check current remote size (for resume awareness)
  CURRENT_SIZE=$(sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
    "stat -c %s /dev/shm/dolmino_now15b_zwfy6.npy 2>/dev/null || echo 0" 2>/dev/null || echo 0)

  if [ "$CURRENT_SIZE" = "$EXPECTED_SIZE" ]; then
    echo "File already complete ($CURRENT_SIZE bytes) - done" | tee -a $LOG
    exit 0
  fi

  echo "Current size on .21: $CURRENT_SIZE / $EXPECTED_SIZE" | tee -a $LOG

  # Start streaming
  sshpass -f $PASS_73 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.85.35.73 \
    "cat /dev/shm/dolmino_now15b.npy" | \
  sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
    "cat > /dev/shm/dolmino_now15b_zwfy6.npy"
  EXIT=$?
  echo "stream exit=$EXIT at $(date)" | tee -a $LOG

  # Check final size
  FINAL_SIZE=$(sshpass -f $PASS_21 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password root@28.89.19.21 \
    "stat -c %s /dev/shm/dolmino_now15b_zwfy6.npy 2>/dev/null || echo 0" 2>/dev/null || echo 0)

  echo "Final size: $FINAL_SIZE" | tee -a $LOG

  if [ "$FINAL_SIZE" = "$EXPECTED_SIZE" ]; then
    echo "SIZE_OK - transfer complete!" | tee -a $LOG
    exit 0
  fi

  echo "Incomplete ($FINAL_SIZE / $EXPECTED_SIZE), waiting 30s before retry..." | tee -a $LOG
  sleep 30
done

echo "Max attempts reached, giving up" | tee -a $LOG
exit 1
