#!/usr/bin/env bash
# ============================================================================
# UNION-9 HARNESS REBUILD — SAME-ARM REPRODUCTION CONTROL
#
#   bash scripts/_union9_harness_rebuild_control.sh
#
# WHY
# ---
# The 2026-08-13 node restart destroyed the pinned union-9 harness stack
# (lm_eval 0.4.8 + transformers 4.57.6) on every node, AND wiped
# /root/.cache/huggingface/datasets/ on LOCAL and .212. A replacement venv was
# rebuilt at $ROOT/venv_union9. Version STRINGS now match the archive, but a
# matching version string is not a matching harness: the whole reason this
# project has a pinned stack is the retracted cross-harness comparison whose
# AST-7 offset was -0.346 pp (baselines/cast_repro/SPARSEFORGE_SAME_HARNESS.md
# CORRECTION block). So before any NEW arm is scored on the rebuilt stack, the
# rebuild must be shown to reproduce an ALREADY-ARCHIVED arm.
#
# WHAT IS CONTROLLED
# ------------------
# dense_ref (= models/Llama--Llama2-7b, unmodified Llama-2-7B) is re-scored on
# all 9 tasks and compared per-task against its archived results_*.json. It is
# chosen because it is the one arm whose weights are provably unchanged:
# sha256 of both safetensors shards is identical on wzc1 and zwfy6
#   4ec71fd53e99766de38f24753b30c9e8942630e9e576a1ba27b0ec531e87be41  shard1
#   41780b5dac322ac35598737e99208d90bdc632a1ba3389ebedbb46a1d8385a7f  shard2
# so a per-task delta cannot be blamed on the checkpoint. A pruned arm
# (wanda/sparsegpt) would confound stack drift with export drift.
#
# PASS CRITERION — 0 flips, per memory [[same-harness-runs-bit-identical]]:
# same arch + same disk + same harness re-run was measured BYTE-IDENTICAL
# (0 flips). So the bar here is exact equality of every per-task acc/acc_norm,
# not "within noise". Anything else is reported as a magnitude and BLOCKS.
#
# ⚠️ NODE CHOICE IS PART OF THE CONTROL, NOT A CONVENIENCE
# --------------------------------------------------------
# All 37 archived results_*.json in this repo were scored on `NVIDIA L20A`
# (= B200 sm_100, driver 580.105.08). There is NO H20-scored union-9 precedent.
# H20 is sm_90 with a different driver (535.247.01), so scoring the control on
# .73/.82 would confound "did the software rebuild change the numbers" with
# "does sm_90 vs sm_100 change the numbers" — and a FAIL would be
# uninterpretable. The control therefore runs on a B200, which is also where
# the watchers will score. GPU pressure is handled by taking ONE GPU and
# waiting for it, never by moving to the wrong hardware.
#
# PIQA — a real gap, recovered and proven, not waved through
# ---------------------------------------------------------
# Upstream piqa.yaml uses `dataset_path: piqa`, a loading SCRIPT. datasets 5.0.1
# hard-refuses it ("Dataset scripts are no longer supported, but found piqa.py")
# so piqa cannot load on ANY node today. The archive did not hit this because on
# 2026-08-11 its hub lookup failed and it silently fell back to a script-built
# cache (`ybisk___piqa/plain_text/1.1.0/6c611c1a...`) that no longer exists
# anywhere. See union9_taskoverride/piqa.yaml for the full forensics.
# The override feeds the SAME 1838 docs from parquet, proven doc-for-doc
# identical to the archived samples file by
#   baselines/cast_repro/tools/check_piqa_source_matches_archive.py
# and this script additionally asserts the per-doc doc_hash/prompt_hash match
# the archive, which is what makes the piqa cell comparable rather than merely
# present.
#
# ⚠️ `--include_path` DOES NOT WORK for overriding a built-in task.
# lm_eval/tasks/__init__.py:83 merges `task_index = {**tasks, **task_index}`, so
# the DEFAULT registry wins over include_path for an existing name. Verified:
# with --include_path the run still raised
#   ConnectionError: Couldn't reach 'piqa' on the Hub
# The working route is __main__.py:349 — a --tasks entry that is a FILE PATH is
# loaded as a raw config and bypasses the registry. Hence piqa is passed as a
# yaml path while the other 8 stay plain names.
# ============================================================================
set -u

ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code
MOM=$ROOT/Mixture-of-Memory
PY=${PYTHON_BIN:-$ROOT/venv_union9/bin/python}
LM_EVAL=${LM_EVAL_BIN:-$ROOT/venv_union9/bin/lm_eval}

MODEL=$ROOT/models/Llama--Llama2-7b
# ⚠️ NEITHER `--include_path` NOR a --tasks YAML PATH CAN OVERRIDE A BUILT-IN TASK.
# Both were tried and both failed:
#   --include_path  : lm_eval/tasks/__init__.py:83 merges
#                     `task_index = {**tasks, **task_index}`, so the DEFAULT
#                     registry WINS for an existing name. Run still raised
#                     ConnectionError: Couldn't reach 'piqa' on the Hub.
#   --tasks <path>  : __main__.py:349 does load a file path as a raw config, but
#                     :353 then computes `task_missing` by testing the raw STRING
#                     against a list that now holds dicts, so the path is always
#                     "missing" -> ValueError: Tasks not found: <...>/piqa.yaml.
# Copying all 9 yamls into one dir (the __main__.py:337 directory route) is also
# wrong here: hellaswag/race/winogrande use `!function utils.process_docs` etc.,
# which resolve relative to their own package dir and would silently break.
#
# So the override is INSTALLED OVER the venv's own piqa.yaml, with the upstream
# file preserved beside it as piqa.yaml.upstream-0.4.8.bak. This venv exists only
# to score union-9, so a pinned task definition is appropriate; and the change is
# auditable as a 2-key diff:
#   dataset_path: piqa -> parquet, plus dataset_kwargs.data_files
# Everything that determines prompts/targets/metrics is byte-identical, which is
# why doc_hash/prompt_hash/target_hash and hence results_*.json's
# task_hashes["piqa"] (evaluation_tracker.py:219 hashes exactly those three) must
# still match the archive. verify_piqa_override_hashes.py checks that directly:
# 0/1838 mismatches on all three hashes.
PIQA_INSTALLED=$ROOT/venv_union9/lib/python3.14/site-packages/lm_eval/tasks/piqa/piqa.yaml
PIQA_BAK=$PIQA_INSTALLED.upstream-0.4.8.bak
OUT=$ROOT/outputs/union9_harness_rebuild_control/dense_ref
ARCHIVE=$ROOT/outputs/cast_eval_spec_union9/dense_ref/lm_eval_out/__apdcephfs_wzc1__share_304376610__pighzliu_code__models__Llama--Llama2-7b/results_2026-08-11T11-58-45.812255.json
ARCHIVE_DIR=$(dirname "$ARCHIVE")
PROG=$MOM/logs/union9_harness_rebuild_control.log

# Byte-identical to _sparseforge_same_harness_21.sh:53. piqa resolves through the
# INSTALLED-yaml override (see PIQA note in the header), so the task list itself
# is unchanged from the archive's -- which is what keeps the row comparable.
TASKS=boolq,rte,hellaswag,race,piqa,winogrande,arc_easy,arc_challenge,openbookqa

GPUS="${GPUS:-0,1,2,3}"
POLL_S="${POLL_S:-300}"
MAX_WAIT_H="${MAX_WAIT_H:-24}"

export HF_HUB_OFFLINE=0
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export no_proxy='mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local'

mkdir -p "$OUT/lm_eval_out" "$MOM/logs"
note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

note "=============================================================="
note "UNION-9 HARNESS REBUILD CONTROL -- arm=dense_ref"
note "model=$MODEL"
note "archive=$ARCHIVE"
note "out=$OUT   gpus=$GPUS"
note "=============================================================="

for f in "$MODEL/model.safetensors.index.json" "$PIQA_INSTALLED" "$PIQA_BAK" "$ARCHIVE"; do
  [ -e "$f" ] || { note "FATAL missing asset: $f"; exit 3; }
done
[ -x "$PY" ] || { note "FATAL python not executable: $PY"; exit 3; }
[ -x "$LM_EVAL" ] || { note "FATAL lm_eval not executable: $LM_EVAL"; exit 3; }

# Fail closed if the installed piqa.yaml is not the intended override, so a
# reverted/reinstalled venv cannot silently score the wrong piqa (or crash 40 min in).
if ! grep -q "union9_piqa_parquet" "$PIQA_INSTALLED"; then
  note "FATAL installed piqa.yaml is not the union-9 override (no union9_piqa_parquet path)."
  note "      reinstall it from baselines/cast_repro/union9_taskoverride/piqa.yaml"
  exit 3
fi
note "preflight: piqa override installed at $PIQA_INSTALLED"

# Prove the override still yields the archive's exact docs/prompts BEFORE burning
# GPU time. Cheap (CPU, no model) and it is the assumption the piqa cell rests on.
CUDA_VISIBLE_DEVICES="" "$PY" "$MOM/baselines/cast_repro/tools/verify_piqa_override_hashes.py" \
  2>/dev/null | tail -6 | tee -a "$PROG"
piqa_rc=${PIPESTATUS[0]}
[ "$piqa_rc" -eq 0 ] || { note "FATAL piqa override is not hash-identical to the archive (rc=$piqa_rc)"; exit 3; }

# Same harness-identity assertion the watcher uses, so the control cannot pass
# on a stack the watcher would reject (or vice versa).
"$PY" - <<'PYEOF' 2>&1 | tee -a "$PROG"
import importlib.metadata as m
want = {"lm_eval": "0.4.8", "transformers": "4.57.6"}
bad = []
for pkg, exp in want.items():
    try:
        got = m.version(pkg)
    except Exception as e:
        got = f"<missing: {e.__class__.__name__}>"
    print(f"[harness] {'OK ' if got == exp else 'DRIFT'} {pkg}: got {got} expected {exp}")
    if got != exp:
        bad.append(pkg)
print("[harness] VERDICT: " + ("MATCH" if not bad else f"MISMATCH on {bad}"))
raise SystemExit(0 if not bad else 21)
PYEOF
[ "${PIPESTATUS[0]}" -eq 0 ] || { note "FATAL harness stack mismatch"; exit 21; }

# Hardware guard: refuse to produce a control on non-archive hardware, because a
# FAIL would not be attributable (see header).
"$PY" - <<'PYEOF' 2>&1 | tee -a "$PROG"
import torch
p = torch.cuda.get_device_properties(0)
print(f"[hw] name={p.name} cc=sm_{p.major}{p.minor} SMs={p.multi_processor_count} mem={p.total_memory/1e9:.1f}GB")
if (p.major, p.minor) != (10, 0):
    print("[hw] REFUSE: archive arms were all scored on sm_100 (B200). Scoring the")
    print("[hw]         control elsewhere confounds stack drift with hardware drift.")
    raise SystemExit(22)
print("[hw] VERDICT: sm_100, matches the archive's hardware")
raise SystemExit(0)
PYEOF
[ "${PIPESTATUS[0]}" -eq 0 ] || { note "FATAL wrong hardware for a control"; exit 22; }

# Wait for enough FREE memory on my GPUs. NOTE this is deliberately a
# "free memory" test, not the watcher's "used <= threshold" test: the two ±SLoRB
# training arms legitimately hold ~111-120 GB/card on this box for hours, and the
# control must coexist with them rather than wait for them to end. A dense
# Llama-2-7B in bf16 is ~13.5 GB of weights, sharded over 4 cards by
# parallelize=True, plus activations at batch 64.
#
# Sampled REPEATEDLY, not once: an instantaneous reading cannot distinguish
# "steady state" from "mid-allocation dip" (memory [[one-sample-is-not-a-trend-or-state]]).
MIN_FREE_MIB="${MIN_FREE_MIB:-40000}"
FREE_SAMPLES="${FREE_SAMPLES:-3}"
FREE_SAMPLE_GAP_S="${FREE_SAMPLE_GAP_S:-20}"

min_free_mib() {
  nvidia-smi -i "$GPUS" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' 'BEGIN{m=1e9} {f=$1-$2; if (f<m) m=f} END {print (m==1e9?0:m)}'
}
gpus_have_room() {   # every sample in the window must clear the bar
  local i f
  for i in $(seq 1 "$FREE_SAMPLES"); do
    f=$(min_free_mib)
    note "  free-mem sample $i/$FREE_SAMPLES: min free across $GPUS = ${f}MiB (need >= ${MIN_FREE_MIB})"
    [ "$f" -ge "$MIN_FREE_MIB" ] || return 1
    [ "$i" -eq "$FREE_SAMPLES" ] || sleep "$FREE_SAMPLE_GAP_S"
  done
  return 0
}
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
note "waiting for >=${MIN_FREE_MIB}MiB free on EACH of $GPUS, stable over ${FREE_SAMPLES} samples"
while :; do
  if gpus_have_room; then
    note "GPUs $GPUS have stable headroom; proceeding"
    break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    note "FATAL wait budget ${MAX_WAIT_H}h exhausted. NOT scoring."
    exit 9
  fi
  sleep "$POLL_S"
done

# ⚠️ BATCH SIZE IS PINNED TO 64, NOT `auto`, AND THAT IS THE COMPARABLE CHOICE.
# The archive ran `--batch_size auto` on an IDLE box and auto RESOLVED to 64
# (every archived results_*.json: batch_size="auto", batch_sizes=[64]).
# `auto` re-probes by allocating until OOM (huggingface.py:745,
# find_executable_batch_size(starting_batch_size=64)); on this box, where a
# trainer already holds ~111-120 GB/card, that probe could (a) resolve LOWER than
# 64 and silently make the control a different invocation, or (b) OOM the
# neighbouring trainer. Pinning 64 reproduces the archive's EFFECTIVE batch while
# removing the probe. The comparator prints archive vs rerun batch_sizes so the
# equivalence is visible, not assumed.
BATCH_SIZE="${BATCH_SIZE:-64}"

# Invocation byte-identical to _sparseforge_same_harness_21.sh run_one() except
# `pretrained`, the piqa yaml path, and the pinned batch size documented above.
note "=== lm_eval union-9 (9 tasks), batch_size=$BATCH_SIZE ==="
CUDA_VISIBLE_DEVICES="$GPUS" HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1 \
"$LM_EVAL" \
  --model hf \
  --model_args "pretrained=$MODEL,dtype=bfloat16,parallelize=True,trust_remote_code=True,add_bos_token=False" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --num_fewshot 0 \
  --output_path "$OUT/lm_eval_out" \
  --seed 0 \
  --trust_remote_code \
  --log_samples 2>&1 | tee "$OUT/lm_eval.log"
rc=${PIPESTATUS[0]}
note "=== lm_eval rc=$rc ==="
[ "$rc" -eq 0 ] || { note "FATAL lm_eval failed rc=$rc -- no verdict"; exit "$rc"; }

note "=== COMPARE vs archive ==="
"$PY" "$MOM/baselines/cast_repro/tools/compare_union9_rerun_vs_archive.py" \
    --archive-dir "$ARCHIVE_DIR" \
    --rerun-dir "$OUT/lm_eval_out" \
    --output "$OUT/harness_rebuild_control.json" 2>&1 | tee -a "$PROG"
cmp_rc=${PIPESTATUS[0]}
note "=== COMPARE rc=$cmp_rc (0=PASS 30=per-task drift 31=doc/prompt drift) ==="
exit "$cmp_rc"
