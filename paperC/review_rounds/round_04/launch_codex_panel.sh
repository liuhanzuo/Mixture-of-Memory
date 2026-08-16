#!/usr/bin/env bash
# 6 tcodex blind reviewers on paperC round_04, ALL on gpt-5.6-sol at effort=max.
#
# Why one model, not six: MEASURED 2026-08-16 -- `effort=max` is only accepted by the gpt-5.6-*
# family. gpt-5.5 / gpt-5.4 / gpt-5.3-codex all return HTTP 400
# "Unsupported value: 'max' is not supported with the '<model>' model. Supported values are:
# none, low, medium, high, xhigh". config.toml sets model_reasoning_effort="max" globally, so any
# non-5.6 model dies at the first request. Reviewer diversity therefore comes from the six role
# prompts, not from mixing models.
#
# NEVER pass -c: it clears the injected model_providers.tencent and the run dies dialling a
# hardcoded openai.com endpoint. All config belongs in $CODEX_HOME/config.toml.
# `-o <file>` is mandatory: an unrelated tcodex start SIGKILLs running gateways and without -o the
# output is lost entirely.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
RD=$ROOT/paperC/review_rounds/round_04
MODEL=gpt-5.6-sol
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1"; export NO_PROXY="$no_proxy"
export CODEX_HOME=$ROOT/.codex

launch () {
  id=$1
  ( cd "$ROOT" && timeout 5400 tcodex exec --skip-git-repo-check -m "$MODEL" \
      -o "$RD/raw/$id.md" "$(cat "$RD/prompts/$id.txt")" \
      > "$RD/raw/$id.tcodex.log" 2>&1 < /dev/null
    printf 'rc=%s id=%s model=%s\n' "$?" "$id" "$MODEL" >> "$RD/raw/_exit_codes.txt" ) &
  printf 'launched %s on %s (pid %s)\n' "$id" "$MODEL" "$!"
}

for id in "$@"; do launch "$id"; done
wait
printf 'PANEL BATCH FINISHED\n'
cat "$RD/raw/_exit_codes.txt"
