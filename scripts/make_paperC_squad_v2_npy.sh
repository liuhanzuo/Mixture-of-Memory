#!/usr/bin/env bash
# ============================================================================
# Regenerate the Paper C CLEAN SQuAD SFT training shards (.npy) from the P0-4
# rebuilt jsonl tiers. Idempotent; skips a tier whose .npy already exists.
#
# WHY THIS SCRIPT EXISTS: `data/` is in .gitignore, so the .npy files do NOT
# travel with the repo. The zwfy6 nodes (.73/.82) are a separate checkout on a
# separate physical disk and will NOT have them -- run this there (or `scp -O`
# the .npy across and verify md5) before launching run_paperC_depthsweep.sh.
#
# WHY THE OLD .npy IS VOID: `data/squad_sft_olmo2_2048_train.npy` was VERIFIED
# on 2026-08-06 to be byte-identical (md5 44f46c1d595e8f20d54ed9dcb6a9e34c) to
# re-tokenising the CONTAMINATED legacy `data/squad_train.jsonl` -- whose val
# counterpart has a 49.85% input-blind constant-refusal floor ABOVE every
# measured arm. See versions/paperC_scoping.md and the forensics docstring of
# scripts/build_paperC_squad_eval.py. This script NEVER touches that file.
#
# The tokeniser (scripts/tokenize_squad_olmo2_sft.py) is used UNMODIFIED, so the
# clean shards are packed exactly like the legacy one (uint32 [N,2048], EOS 100257
# after each SFT example, full-LM loss, <seq_len tail dropped) -- the only thing
# that changes is the source jsonl.
#
# Expected output (2026-08-06, OLMo-2-1124-7B tokenizer, seq_len 2048):
#   train_refusal00_olmo2_2048.npy  (1073, 2048)  2.20M tok
#   train_refusal25_olmo2_2048.npy  (1064, 2048)  2.18M tok   <- launcher default
#   train_refusal50_olmo2_2048.npy  (1054, 2048)  2.16M tok
#
# USAGE
#   bash scripts/make_paperC_squad_v2_npy.sh                       # all 3 tiers
#   TIERS=25 bash scripts/make_paperC_squad_v2_npy.sh              # one tier
#   PROJECT_ROOT=/apdcephfs_zwfy6/... PYBIN=/opt/conda/envs/torch-base/bin/python \
#     BASE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B \
#     bash scripts/make_paperC_squad_v2_npy.sh                     # on .73/.82
#   FORCE=1 ... # re-tokenise even if the .npy exists
# ============================================================================
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT" || { echo "FATAL: cannot cd $PROJECT_ROOT"; exit 1; }
PYBIN="${PYBIN:-${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}}"
BASE="${BASE:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
SEQ_LEN="${SEQ_LEN:-2048}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/data/paperC_squad_v2}"
FORCE="${FORCE:-0}"
read -r -a TIERS <<< "${TIERS:-00 25 50}"

log(){ echo "[mk-npy $(date '+%F %T')] $*"; }

pf=0
[ -x "$PYBIN" ] || { log "PREFLIGHT: PYBIN not executable: $PYBIN"; pf=1; }
[ -e "$BASE/tokenizer.json" ] || { log "PREFLIGHT MISSING tokenizer: $BASE/tokenizer.json"; pf=1; }
[ -e "$PROJECT_ROOT/scripts/tokenize_squad_olmo2_sft.py" ] || { log "PREFLIGHT MISSING tokeniser script"; pf=1; }
for t in "${TIERS[@]}"; do
  [ -e "$OUT_DIR/train_refusal${t}.jsonl" ] || {
    log "PREFLIGHT MISSING $OUT_DIR/train_refusal${t}.jsonl"
    log "  -> build it first: $PYBIN scripts/build_paperC_squad_eval.py"
    pf=1; }
done
[ "$pf" = 0 ] || { log "FATAL preflight failed"; exit 1; }
log "PROJECT_ROOT=$PROJECT_ROOT  PYBIN=$PYBIN  BASE=$BASE  seq_len=$SEQ_LEN  tiers=${TIERS[*]}"

rc_all=0
for t in "${TIERS[@]}"; do
  IN="$OUT_DIR/train_refusal${t}.jsonl"
  OUT="$OUT_DIR/train_refusal${t}_olmo2_2048.npy"
  if [ -f "$OUT" ] && [ "$FORCE" != "1" ]; then
    log "SKIP tier $t (exists: $OUT, md5 $(md5sum "$OUT" | cut -d' ' -f1)); FORCE=1 to redo"
    continue
  fi
  log "tokenise tier $t: $IN -> $OUT"
  "$PYBIN" scripts/tokenize_squad_olmo2_sft.py \
      --in_jsonl "$IN" --out_npy "$OUT" --tokenizer "$BASE" --seq_len "$SEQ_LEN"
  rc=$?
  if [ "$rc" -ne 0 ] || [ ! -f "$OUT" ]; then
    log "FAIL tier $t (exit $rc)"; rc_all=1; continue
  fi
  log "OK tier $t md5=$(md5sum "$OUT" | cut -d' ' -f1)"
done

log "=== inventory ==="
ls -la "$OUT_DIR"/train_refusal*_olmo2_2048.npy 2>/dev/null
log "NOTE data/squad_sft_olmo2_2048_train.npy is left untouched (VOID legacy, kept only to reproduce #133)."
exit "$rc_all"
