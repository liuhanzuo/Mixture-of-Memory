#!/bin/bash
# PaperC research runner WITH AUTO-RETRY.
# $1=tag $2=promptfile
# CRITICAL: do NOT pass any -c flag to tcodex exec -- it clobbers tcodex's
# injected model_providers.tencent block and silently reverts to provider=openai,
# which then dials unreachable wss://api.openai.com and times out forever.
# Reasoning effort (max) + web_search come from $CODEX_HOME/config.toml instead.
TAG="$1"; PF="$2"
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT" || exit 9
export CODEX_HOME=/tmp/tcx/ch
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

: > /tmp/tcx/${TAG}.meta
for ATTEMPT in 1 2 3 4; do
  echo "ATTEMPT $ATTEMPT START $(date -Is) tag=$TAG" >> /tmp/tcx/${TAG}.meta
  S=$(date +%s)
  tcodex exec --skip-git-repo-check \
    -o /tmp/tcx/${TAG}.final.md \
    "$(cat "$PF")" > /tmp/tcx/${TAG}.a${ATTEMPT}.log 2>&1 < /dev/null
  RC=$?
  E=$(date +%s)
  echo "ATTEMPT $ATTEMPT END $(date -Is) rc=$RC elapsed_s=$((E-S))" >> /tmp/tcx/${TAG}.meta
  if [ -s /tmp/tcx/${TAG}.final.md ]; then
    echo "SUCCESS attempt=$ATTEMPT" >> /tmp/tcx/${TAG}.meta
    cp /tmp/tcx/${TAG}.a${ATTEMPT}.log /tmp/tcx/${TAG}.log
    exit 0
  fi
  echo "RETRYING (no final output)" >> /tmp/tcx/${TAG}.meta
  sleep 10
done
echo "ALL_ATTEMPTS_FAILED" >> /tmp/tcx/${TAG}.meta
