#!/usr/bin/env bash
# Detached orchestrator: resume-rsync N192c step500 ckpt to B200 (.188, wzc1),
# scp the eval sched, then ssh-launch it detached on B200's 8 L20A GPUs.
set -uo pipefail
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
PW=configs/password_b200_188.txt
B200=root@28.89.18.188
RROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
SSH="sshpass -f $PW ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o PreferredAuthentications=password"
LOG=logs/_orchestrate_N192c_b200.log
exec >>"$LOG" 2>&1
echo "==== [$(date)] orchestrator start ===="

$SSH $B200 "mkdir -p $RROOT/outputs/expR1cN192c_cum_slots192 $RROOT/scripts $RROOT/logs"

echo "[$(date)] rsync ckpt (resume --partial)"
rsync -az --partial \
  -e "sshpass -f $PW ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password" \
  outputs/expR1cN192c_cum_slots192/adapter_config.json \
  outputs/expR1cN192c_cum_slots192/mem_space_adapter_step000500.pt \
  $B200:$RROOT/outputs/expR1cN192c_cum_slots192/
RC=$?
echo "[$(date)] rsync rc=$RC"
[ $RC -ne 0 ] && { echo "RSYNC FAILED, abort"; exit 1; }

echo "[$(date)] scp eval sched"
sshpass -f $PW scp -o StrictHostKeyChecking=no -o PreferredAuthentications=password \
  scripts/_expR1cN192c_eval_sched.sh $B200:$RROOT/scripts/

echo "[$(date)] launch eval detached on B200"
$SSH $B200 "cd $RROOT && setsid nohup bash scripts/_expR1cN192c_eval_sched.sh </dev/null >logs/eval_expR1cN192c_sched.log 2>&1 & echo launched pid \$!"

echo "==== [$(date)] orchestrator done ===="
