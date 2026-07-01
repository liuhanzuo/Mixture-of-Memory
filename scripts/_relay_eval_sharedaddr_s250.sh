#!/usr/bin/env bash
# Two-hop relay sharedaddr step250 ckpt: wzc1(.53) -> main staging -> diskB(.174), then launch MECH eval.
set -euo pipefail
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory

SRC_NODE=28.88.184.53
SRC_PW=configs/password_b200_53.txt
SRC_DIR=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/distill_pg19_nctx63_sharedaddr
DST_NODE=28.58.245.174
DST_PW=configs/password_h20_returned.txt
DST_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
DST_DIR=$DST_ROOT/outputs/distill_pg19_nctx63_sharedaddr
STAGE=staging/sharedaddr_s250

echo "[$(date)] HOP1 wzc1(.53) -> main staging"
rsync -az --partial \
  -e "sshpass -f $SRC_PW ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o PreferredAuthentications=password" \
  root@$SRC_NODE:$SRC_DIR/mem_space_adapter_step000250.pt \
  root@$SRC_NODE:$SRC_DIR/adapter_config.json \
  $STAGE/
echo "[$(date)] HOP1 done; staging contents:"; ls -la $STAGE/

echo "[$(date)] HOP2 main staging -> diskB(.174)"
rsync -az --partial \
  -e "sshpass -f $DST_PW ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o PreferredAuthentications=password" \
  $STAGE/mem_space_adapter_step000250.pt \
  $STAGE/adapter_config.json \
  root@$DST_NODE:$DST_DIR/
echo "[$(date)] HOP2 done"

echo "[$(date)] LAUNCH MECH eval on .174"
sshpass -f $DST_PW ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o PreferredAuthentications=password root@$DST_NODE "
cd $DST_ROOT
RUN_PREFIX=mech_sharedaddr_s250 \
CKPT_FILES='outputs/distill_pg19_nctx63_sharedaddr/mem_space_adapter_step000250.pt' \
CK_NAMES='mech_sharedaddr_s250' \
ADAPTER_CONFIG=outputs/distill_pg19_nctx63_sharedaddr/adapter_config.json \
PROJECT_ROOT=$DST_ROOT PYTHON_BIN=$DST_ROOT/.venv/bin/python \
setsid nohup bash scripts/_eval_taskpool_2group.sh >logs/eval_mech_sharedaddr_s250_sched.out 2>&1 &
echo launched eval pid \$!
"
echo "[$(date)] RELAY+EVAL dispatch complete"
