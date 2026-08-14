# LAUNCH RANKING — .21 + .82, night of 2026-08-08

Author: ranking lane (read-only scout). All claims below were re-verified by me directly;
where a lane's claim did not survive verification I say so explicitly.

---

## 0. TL;DR

| node | disk | recommendation | why |
|---|---|---|---|
| **.21** | wzc1 | **B04 cross-family replication — Qwen3 prune-heal ladder, core6, 6 rungs** (task #208) | Only tonight-ready gate that can KILL a scientific claim. 107 GB of ckpts are wzc1-only → zero transfer. Needs a 6-line harness patch first. |
| **.82** | zwfy6 | **A03 floor gate for HotpotQA-multi-evidence + CounterFact-conflict** (3 arms × 5 conditions) | The 1B ckpts are zwfy6-only (22.6 GiB), .82 has scipy (.21 does NOT). Can retire individual axes and set A03's scope. |

**Two free wins with no GPU at all** (do these first, they are minutes of CPU):
1. **A01 gate-2 is already answered** — BoolQ + OpenBookQA null-calibration. I re-ran lane2's
   headline statistic myself and it reproduced **exactly**. Just needs the driver script persisted.
2. **A01 gate-1 already FINISHED on .21 at 21:41** for all three families. Only the
   per-family floor recomputation is left, and I did that too (numbers in §3).

---

## 1. Verification pass — what changed vs. the lane reports

### 1.1 ★ .21 IS FREE. Lane1's central blocker is STALE.

Lane1's headline blocker was "★ .21 IS ALREADY RUNNING THIS EXACT GATE … PIDs 41778-41785 …
Launching anything on .21 will OOM it." **That run completed at 21:41:52.** Verified:

```
$ nvidia-smi on 28.89.19.21     → all 8 GPUs "1 MiB, 0 %"
$ nvidia-smi --query-compute-apps → EMPTY
$ ps                            → only PID 25999 prepare_dolmino_llama2.py (CPU+network)
```

`logs/a01_gate1_driver.log` last line: `[21:41:52] ALL DONE`. **.21 is available.** MAIN should
not hold it back on lane1's advice.

### 1.2 ★ A01 gate-1 is DONE for all three families — do not re-launch it

Not one family but **three** completed, each n=14042 / n_nan=0 / 8-of-8 shards:

| arm | dir | arch | letter | content_norm |
|---|---|---|---|---|
| Llama-2-7B | `olmo2_mmlu_content_results/gate1_llama2_7b` | LlamaForCausalLM 32L | 0.4100 | 0.4135 |
| Llama-3-8B | `olmo2_mmlu_content_results/gate1_llama3_8b` | LlamaForCausalLM 32L | 0.6220 | 0.4624 |
| Qwen3-8B-Base | `olmo2_mmlu_content_results/gate1_qwen3_8b_base` | Qwen3ForCausalLM 36L | 0.7464 | 0.5173 |

Total GPU time: **~5 minutes** (21:36:54 → 21:41:52). Lane1's proposed 1.5-3 h Llama-3 launch on
.82 would **redo work already on disk**.

### 1.3 Lane2's BoolQ headline — I reproduced it exactly, independently

I recomputed from `olmo2_downstream_results/*_know/per_example_boolq.jsonl` with
`scipy.stats.binomtest`:

```
boolq n=3270  gold {A:1237, B:2033}  best-constant always-B = 0.6217  (naive chance 0.5000)

arm            acc   accnorm | raw-LL vs always-B          | acc_norm vs always-B
base         0.8156  0.7526  | +0.1939 p=6.26e-93 ABOVE    | +0.1309 p=2.96e-88 ABOVE
keep14       0.6382  0.6887  | +0.0165 p=0.203    n.s.     | +0.0670 p=1.62e-17 ABOVE
keep12       0.6101  0.6541  | -0.0116 p=0.362    below    | +0.0324 p=2.23e-06 ABOVE   <- FLIP
keep10       0.6086  0.6287  | -0.0131 p=0.217    below    | +0.0070 p=0.144    n.s.
keep8        0.5948  0.6269  | -0.0269 p=0.0065   BELOW*   | +0.0052 p=0.196    n.s.
shortgpt16   0.7297  0.6972  | +0.1080 p=1.88e-24 ABOVE    | +0.0755 p=2.46e-39 ABOVE
```

Matches lane2 to 4 decimals on every cell. The 12.2 pp null understatement and the keep12
interface flip are **real**. Lane2's `NOT_A_GPU_JOB` verdict is correct and valuable.

### 1.4 Lane4's blocker is real, but its FIX is wrong — there is a much better one

Lane4 correctly found that `eval_qwen3_probe2_downstream.py` does not emit
`norm_lens`/`norm_scores`, so the B04 analyzer silently falls back to raw un-length-normalized
`option_scores`. Verified — the OLMo harness has them at lines 470-475, the Qwen harness does not.

**But lane4's proposed fix (post-hoc `enrich_per_example_normscores.py`) is the inferior path.**
I read the Qwen harness: at the write point (line ~458) **both `norm_lens` and `norm_lls` are
already local variables in scope**. So the fix is a **6-line in-harness patch**, which:

* needs **no proxy** (lane4's enrich path needs `http_proxy` because the HF dataset resolve
  goes to network — lane4 admits this);
* **mutates no canonical result file** (the enrich path rewrites `per_example_*.jsonl` in place);
* produces the field on first write, so the data is right the first time.

Lane4 also stated the wzc1 HF cache lacks these datasets. **That is wrong for wzc1.** I ran
`load_task_examples()` for all six core6 tasks fully offline (`HF_DATASETS_OFFLINE=1
HF_HUB_OFFLINE=1`) and all six loaded from `data/hf_datasets_cache/`:

```
OK hellaswag     n=10042  norm_lens0=[37, 27, 26, 36]
OK arc_challenge n=1172   norm_lens0=[32, 35, 35, 39]
OK arc_easy      n=2376   norm_lens0=[59, 54, 50, 43]
OK piqa          n=1838   norm_lens0=[164, 166]
OK winogrande    n=1267   norm_lens0=[28, 28]
OK openbookqa    n=500    norm_lens0=[21, 21, 28, 23]
```

Lane4's proxy blocker applied to `/root/.cache/huggingface`, not to the project's own
`data/hf_datasets_cache`. **No proxy is needed. The launch can run fully offline.**

### 1.5 Comparability check lane4 did NOT do (and it passes)

For a cross-family Spearman to be meaningful the two harnesses must use the same numeric path.
I checked: both use **fp32 weights + bf16 autocast forward** (`eval_olmo2_probe2_downstream.py:383`
and `eval_qwen3_probe2_downstream.py:392`, identical `torch.amp.autocast("cuda",
dtype=torch.bfloat16)`), and `_safe_lp` exists in both. Same `item_id = shard_index + ei *
num_shards`. **Comparable.** Good news, but it was an unverified assumption in lane4's report.

### 1.6 Qwen ckpts are wzc1-ONLY — .82 is not an option for B04

Lane4 said .82 would need "107 GB" of transfer. I verified per-arm on .82:

| arm | wzc1 | zwfy6 (.82) |
|---|---|---|
| `qwen3_minarch_armB_f12k2_200k/final.pt` | 45G | present |
| `qwen3_minarch_armB_f12k2_20k/final.pt` | 15G | **ABSENT** |
| `qwen3_minarch_armB_f12k4/final.pt` | 17G | **ABSENT** |
| `qwen3_minarch_armB_f12k2/final.pt` | 15G | **ABSENT** |
| `qwen3_minarch_scratch_f12k2/final.pt` | 15G | **ABSENT** |

62 GB missing on zwfy6 → ~1.4 h (4-stream) to 4.3 h (single-stream). **B04 must go to .21.**
This is the disk-match decision, and it is decisive.

### 1.7 Lane4's n=6 claim needs one correction

Lane4 says including base full-36 "restores n=6 so p can reach 0.0028". Two-sided exact
permutation at n=6 has floor **2/720 = 0.00278**; the OLMo verdict reports it as "1/360". Same
number, both fine. But note the analyzer's `exact_p_two_sided` iterates `permutations(y)`
generically, so it handles n=6 with **no code change** — lane4's STEP 8 "fix the exact-permutation
p for n=6" is unnecessary; only the docstring says n=5.

### 1.8 Lane3 (A02) — confirmed: the runnable version already ran

Verified on .82: `bench_results/p0_18_e4_{2x2,bbwl_step1000,bbwl_step1500,bbwl_step2000}` all
exist. `p0_18_e4_bbwl_step2000/summary.json` has both cells at 8k and 16k with
`diff_BBWL_minus_BB = 6.0` at each. Write-LoRA `qcmem_writepath_distill_qwen_j12_r32/` = 556M
with step500..2500. Lane3's own verdict — "re-running p018 reproduces a known number and DECIDES
NOTHING" — is right. **Do not launch A02 tonight.**

### 1.9 Lane5 (A03) — assets verified

.82 confirmed 8/8 GPUs at 0 MiB. `outputs/olmo2_probe2_1B_keep7fresh2_16card/` has
step{50000,100000,150000,200000}.pt + final.pt at 12181310078 B each; `..._keep7fresh2/step500.pt`
= 12181308233 B. `scripts/eval_paperC_squad_emf1.py` (9792 B) and `_run_a03_1b_floor_82.sh`
(5010 B) present. `scipy 1.18.0` + `pyarrow 25.0.0` on .82. All lane5 claims hold.

---

## 2. RECOMMENDED LAUNCH — `.21` (wzc1, 8× L20A 183 GB)

### B04 cross-family replication: Qwen3-8B prune-heal ladder, core6 acc_norm margins

**What it decides.** `DIRECTION_A_VERDICT.md` states in its own "What this is NOT" section:
*"**NOT** established beyond OLMo-2-7B. Cross-family replication (Qwen prune-heal ladder) is the
next kill test."* Status is `SURVIVING`, `promotion_pending: novelty_check_only`. So:

* **Signs replicate** (median_margin ρ>0, frac<.005 ρ<0) → B04's generality holds, and with the
  CPU novelty check (#207) B04 is promotable to `paper<X>`.
* **Sign flips or ρ collapses** → "structural damage compresses per-item decision margins" is
  **OLMo-2-specific** and B04's headline must be narrowed to a single-model observation. That is a
  real kill of the general claim.

Either way MAIN writes something different. This is the only tonight-ready gate with that property.

**Why .21 and not .82:** 62 GB of the 5 ckpts are absent on zwfy6 (§1.6). Zero transfer on wzc1.
Also the 45 GB ckpt load benefits from .21's 2013 GB RAM / 256 cores.

**Why the 183 GB cards are justified:** fp32 weights for a 14-16L Qwen3-8B shell plus bf16
autocast activations at bs16 — and critically, the base full-36 rung is a full 8B fp32 load. This
is the fp32-7B/8B-forward case that is the canonical L20A-only workload.

### STEP 1 — the 6-line harness patch (MANDATORY, do before launching)

Without this the analyzer silently uses raw un-length-normalized `option_scores` and the
cross-family comparison is **invalid**. In
`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/scripts/eval_qwen3_probe2_downstream.py`,
in the `save_per_example` block at **line ~458-467** (the non-nan branch), after
`"acc_norm_score": 1.0 if pred_norm == gold else 0.0,` and `"nan": False,` add:

```python
                # NEW: length-norm fields needed for acc_norm margin analysis (B04).
                # Mirrors eval_olmo2_probe2_downstream.py:470-475 exactly.
                # norm_lens[k] = raw candidate char count (= c[2] in load_task_examples).
                # norm_scores[k] = option_scores[k] / max(norm_lens[k], 1).
                "norm_lens": {_LETTERS[k]: norm_lens[k]
                              for k in range(len(norm_lens))},
                "norm_scores": {_LETTERS[k]: _safe_lp(norm_lls[k])
                                for k in range(len(norm_lls))},
```

`norm_lens` (line 418) and `norm_lls` (line 445) are **already in scope** — no other change needed.
Do NOT touch the nan branch (line 424-432): `norm_lls` is undefined there.

Verify:
```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory && \
  grep -c "norm_scores" scripts/eval_qwen3_probe2_downstream.py   # expect 1
```

### STEP 2 — pre-flight asset assertion

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory && \
for p in /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b/model.safetensors.index.json \
         outputs/qwen3_minarch_armB_f12k2_200k/final.pt \
         outputs/qwen3_minarch_armB_f12k2_20k/final.pt \
         outputs/qwen3_minarch_armB_f12k4/final.pt \
         outputs/qwen3_minarch_armB_f12k2/final.pt \
         outputs/qwen3_minarch_scratch_f12k2/final.pt \
         scripts/eval_qwen3_probe2_downstream.py; do
  [ -e "$p" ] && echo "EXISTS $(du -sh "$p"|cut -f1)  $p" || echo "MISSING  $p"; done
```
All 7 must print EXISTS (I verified this passes).

### STEP 3 — write the driver

```bash
cat > /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/scripts/_run_b04_qwen_crossfamily_21.sh <<'DRIVER_EOF'
#!/usr/bin/env bash
# B04 Direction-A CROSS-FAMILY replication: Qwen3-8B prune-heal ladder, core6, bs16.
# Node .21 (8x L20A, wzc1). All assets wzc1-local -> zero cross-disk transfer.
# Protocol: BASE LM, chat_template=False (harness never applies one), --add_bos 0,
#           fp32 weights + bf16 autocast (identical to the OLMo harness), 8 shards.
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$ROOT" || exit 3
PY=/opt/conda/envs/torch-base/bin/python
BASE=/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b
HARNESS=scripts/eval_qwen3_probe2_downstream.py
RR=qwen3_probe2_downstream_results
TASKS="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
BS=16

# core6 is fully cached on wzc1 -> stay offline so no shard can hang on a network resolve
export HF_DATASETS_CACHE="$ROOT/data/hf_datasets_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset http_proxy https_proxy all_proxy
mkdir -p logs "$RR"
PROG=logs/b04qwen_progress.log
note() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$PROG"; }

assert_8shards () {
  local D="$RR/$1" MISS=0
  for g in 0 1 2 3 4 5 6 7; do
    [ -f "$D/shard${g}of8.json" ] || { note "[SHARD MISSING] $D/shard${g}of8.json"; MISS=$((MISS+1)); }
  done
  [ $MISS -eq 0 ] && return 0
  note "[ABORT] $MISS/8 shards missing for $1 -- NOT merging"; return 1
}

assert_nscored () {
  $PY - "$RR/$1/summary.json" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1])); tasks = s.get("tasks", {})
expected = {"hellaswag":10042,"arc_challenge":1172,"arc_easy":2376,
            "piqa":1838,"openbookqa":500,"winogrande":1267}
fail = 0
for t, n in expected.items():
    got = tasks.get(t, {}).get("n_scored", -1)
    if got != n: print(f"[N_SCORED MISMATCH] {t}: expected {n}, got {got}"); fail += 1
    else:        print(f"[N_SCORED OK] {t}: {got}")
    if tasks.get(t, {}).get("n_nan", 0): print(f"[N_NAN NONZERO] {t}"); fail += 1
sys.exit(1 if fail else 0)
PYEOF
}

assert_normscores () {
  $PY - "$RR/$1" <<'PYEOF'
import json, os, sys
rd = sys.argv[1]; bad = 0
for t in ["hellaswag","arc_challenge","arc_easy","piqa","winogrande","openbookqa"]:
    p = os.path.join(rd, f"per_example_{t}.jsonl")
    r = json.loads(open(p).readline())
    if "norm_scores" in r and r["norm_scores"]:
        print(f"[NORM_SCORES OK] {t}")
    else:
        print(f"[NORM_SCORES MISSING] {t} -- harness patch not applied!"); bad += 1
sys.exit(1 if bad else 0)
PYEOF
}

note "prepare_data (offline)"
$PY "$HARNESS" --prepare_data --tasks "$TASKS" --results_root "$RR" \
  > logs/b04qwen_prepare.log 2>&1
tail -3 logs/b04qwen_prepare.log

# "NAME|keep_front|n_fresh|ckpt"   (empty ckpt => base full-36 mode)
# Damage order FROZEN here, BEFORE any result is seen (no order-shopping).
# rung 0 = base full-36 (free, no ckpt) -> n=6 so exact two-sided p can reach 2/720=0.00278
CONFIGS=(
  "qwen3_base_full36_bs16|||"
  "qwen3_f12k2_s200000_bs16|12|2|outputs/qwen3_minarch_armB_f12k2_200k/final.pt"
  "qwen3_f12k2_s20000_bs16|12|2|outputs/qwen3_minarch_armB_f12k2_20k/final.pt"
  "qwen3_f12k4_s2000_bs16|12|4|outputs/qwen3_minarch_armB_f12k4/final.pt"
  "qwen3_f12k2_s2000_bs16|12|2|outputs/qwen3_minarch_armB_f12k2/final.pt"
  "qwen3_scratch14L_s2000_bs16|12|2|outputs/qwen3_minarch_scratch_f12k2/final.pt"
)

for row in "${CONFIGS[@]}"; do
  IFS='|' read -r NAME KFL NFL CKPT <<< "$row"
  note "=== CONFIG $NAME kfl=${KFL:-BASE} nfl=${NFL:-BASE} ckpt=${CKPT:-NONE} ==="
  if [ -f "$RR/$NAME/summary.json" ]; then note "$NAME ALREADY DONE -- SKIP"; continue; fi
  if [ -n "$CKPT" ] && [ ! -f "$CKPT" ]; then note "[FATAL] ckpt missing: $CKPT"; continue; fi
  if [ -z "$CKPT" ]; then ARCH_ARGS=""
  else ARCH_ARGS="--ckpt $CKPT --keep_front_layers $KFL --n_fresh_layers $NFL"; fi

  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY "$HARNESS" \
      --base_model "$BASE" $ARCH_ARGS \
      --tasks "$TASKS" --num_shards 8 --shard_index $g \
      --batch_size $BS --add_bos 0 --save_per_example \
      --results_root "$RR" --output_name "$NAME" \
      > "logs/b04qwen_${NAME}_shard${g}.log" 2>&1 &
  done
  wait

  note "shards done for $NAME; asserting 8/8"
  assert_8shards "$NAME" || { note "[FATAL] merge aborted for $NAME"; continue; }
  note "merging $NAME"
  $PY "$HARNESS" --merge --results_root "$RR" --output_name "$NAME" 2>&1 | tail -5
  note "asserting n_scored for $NAME"
  assert_nscored "$NAME" || note "[FATAL] n_scored assertion FAILED for $NAME"
  note "asserting norm_scores present for $NAME"
  assert_normscores "$NAME" || note "[FATAL] norm_scores MISSING for $NAME"
done
note "ALL RUNGS DONE"
DRIVER_EOF
chmod +x /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/scripts/_run_b04_qwen_crossfamily_21.sh && echo WROTE_DRIVER
```

### STEP 4 — confirm .21 idle, then launch

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory && \
sshpass -f configs/password_b200_19021.txt ssh -o StrictHostKeyChecking=no \
  -o ConnectTimeout=12 -o PreferredAuthentications=password root@28.89.19.21 \
  'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; \
   echo "--- must be EMPTY ---"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'
```

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory && \
sshpass -f configs/password_b200_19021.txt ssh -o StrictHostKeyChecking=no \
  -o ConnectTimeout=12 -o PreferredAuthentications=password root@28.89.19.21 \
  'cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory && \
   setsid nohup bash scripts/_run_b04_qwen_crossfamily_21.sh \
     > logs/b04qwen_driver.out 2>&1 < /dev/null & echo LAUNCHED_PID=$!'
```

### STEP 5 — monitor (runs on LOCAL; wzc1 is shared, no ssh needed)

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory && \
  tail -30 logs/b04qwen_progress.log && ls -d qwen3_probe2_downstream_results/*/summary.json 2>/dev/null
```
6 rungs × (START / shards / merge / 3 asserts). A >45 min gap inside one rung = presume dead.

### STEP 6 — analysis (CPU, after ALL RUNGS DONE)

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory && \
sed -e 's#ROOT = Path("olmo2_downstream_results")#ROOT = Path("qwen3_probe2_downstream_results")#' \
    -e 's#^RUNGS = \[#RUNGS = [\n    ("base_full36",   "qwen3_base_full36_bs16"),\n    ("f12k2@200k",    "qwen3_f12k2_s200000_bs16"),\n    ("f12k2@20k",     "qwen3_f12k2_s20000_bs16"),\n    ("f12k4@2k",      "qwen3_f12k4_s2000_bs16"),\n    ("f12k2@2k",      "qwen3_f12k2_s2000_bs16"),\n    ("scratch14L@2k", "qwen3_scratch14L_s2000_bs16"),\n]\n_OLD_RUNGS = [#' \
    -e 's#olmo2_downstream_results/B04_5rung_bs16_analysis.json#qwen3_probe2_downstream_results/B04_qwen_crossfamily_bs16_analysis.json#' \
    proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_5rung.py \
  > proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_qwen_crossfamily.py && \
/opt/conda/envs/torch-base/bin/python proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_qwen_crossfamily.py 2>&1 | tail -30
```

`exact_p_two_sided` iterates `permutations(y)` generically, so n=6 needs no code change (only the
docstring says n=5). Two-sided exact floor at n=6 = **2/720 = 0.00278**.

### Interpretation guard (read before writing anything down)

* **SURVIVES** → `Spearman(core6, frac<0.005)` strongly NEGATIVE *and*
  `Spearman(core6, median_margin)` strongly POSITIVE, same signs as OLMo.
* **KILLED** → sign flips or ρ collapses toward 0 → claim is OLMo-2-specific; narrow the headline.
* Report as an **n=6 cross-family replication**. Do not claim a second p=0.00278 unless all 6
  rungs merged *and* every `n_scored` / `norm_scores` assert passed.
* **State plainly** that the Qwen damage axis is mostly **heal-steps at fixed depth**
  (2000/20000/200000 @ keep12+fresh2) plus one depth variant (f12k4 = 16L) and one from-scratch
  control — it is **not** the OLMo keep{8,10,12,14} depth ladder. Verified from each
  `outputs/*/arch_meta.json`.

**GPUs:** 8. **Est. wall:** 4-6 h (dominated by 15-45 GB fp32 ckpt loads from cephfs; measured
core6 forward is only ~50 s/shard per `logs/olmo2_downstream_7B_keep14_step200000_wzc1_v2_shard0.log`).

**Idempotency:** three layers. (1) skip-if-`summary.json`-exists → never overwrites, safe to
re-invoke; (2) merge gated on 8/8 shard-file assert **plus** per-task `n_scored` assert
(hellaswag 10042 / arc_challenge 1172 / arc_easy 2376 / piqa 1838 / openbookqa 500 /
winogrande 1267) **plus** `n_nan==0` → a silent 5-of-8 merge is impossible; (3) a `norm_scores`
presence assert catches a forgotten STEP-1 patch **before** the analyzer can silently fall back to
raw scores. Results go to a brand-new root `qwen3_probe2_downstream_results/` (verified absent), so
nothing existing can be clobbered. To redo one rung, `rm -rf` just that arm dir.

---

## 3. RECOMMENDED LAUNCH — `.82` (zwfy6, 8× H20 97.8 GB)

### A03 floor gate: multi-evidence (HotpotQA) + updated/conflicting facts (CounterFact)

**What it decides — stated honestly.** It **cannot kill A03 as a whole**: A03's kill condition is
conjunctive ("所有知识指标均处于 floor") and the old-parametric axis already cleared floor on
2026-08-08. What it CAN do:

* **retire an individual axis** the way MMLU-letter was retired (pruned+healed arm at/below its own
  construct-appropriate null while the intact arm is above it) — that prevents the 6-arm build from
  producing an arm ranking that ranks *degeneracy* rather than knowledge;
* if **both** measurable axes retire, A03's 6-axis design collapses to one axis, which is a
  **design kill** for A03's central claim that the optimal split "取决于知识是旧/新/更新/多证据";
* if they clear, it **licenses** the 6-arm build on 3 axes instead of 1.

That is scope-setting, not direction-killing. Ranked #2 for exactly that reason.

**Why .82 and not .21 — two independent reasons, both verified by me:**
1. The 1B ckpts are **zwfy6-only** (`step200000.pt` 12181310078 B, `step500.pt` 12181308233 B;
   `ls outputs/olmo2_probe2_1B_keep7fresh2*` on wzc1 → No such file or directory). Using .21 costs
   a 22.6 GiB cross-disk copy (~17-31 min).
2. **.21 has no scipy.** `analyze_1b_knowledge_floor.py:64-67` falls back to `binomtest = None` and
   `mcnemar_exact()` then returns `float('nan')` **silently**, dropping a required statistic. .82
   has scipy 1.18.0 (verified).

**Why H20 is the right card:** this is a 1B model at ≤3584 tokens. It does not need 183 GB. Per
priority rule #3, the L20A cards must go to the fp32-8B job (B04).

**Full commands:** lane5's `launch_plan.commands` are correct and complete — use them verbatim
from `status/scout_21/lane5_a03_next_axes.md`, in order:
`STEP 0` (idle check) → `1a` (fetch 2 parquets, 27 MiB, ~25 s via
`http_proxy=http://hy-proxy.woa.com:3128`) → `1b` (write + run `build_a03_newaxes.py`; must print
`7405/7405/7405/2191/2191`) → `2` (write `_run_a03_newaxes_82.sh`) → `2b` (launch) → `2c` (watch)
→ `3` (analysis).

Interpreter on .82: `/opt/conda/envs/torch-base/bin/python`.

**Two things MAIN must not skip:**
* **Stratify HotpotQA by `question_type`.** 458/7405 golds are literally yes/no, so best-constant
  EM is **0.1567 on `comparison` (n=1487)** vs **0.0035 on `bridge` (n=5918)**, 0.0315 pooled. An
  unstratified floor lets the comparison subset's yes/no prior masquerade as multi-hop skill.
* **Keep the `step500` arm.** It is the load-bearing at-floor control — it is what proves "at floor"
  is *detectable* on these new axes.

**GPUs:** 8. **Est. wall:** 0.4-0.7 h (anchor: `logs/a03_1b_floor_progress.log` shows the passed
gate did 3 arms × 46k items in 7 min on this node, 20:48→20:55 rc=0).

**Idempotency:** result dirs keyed `<arm>_<axis>` under a **new** root `a03_newaxes_results/` — no
collision with `olmo2_closedbook_results/` or `olmo2_mmlu_content_results/`. A re-run overwrites
shard files in place and rebuilds `per_example.jsonl` from scratch, so a partial earlier run cannot
contaminate a later one. Because it asserts `len(shards)==8` **and** `summary n==expected` **and**
per-example rows==expected **and** `item_id` uniqueness, a partial run **fails loudly** instead of
silently merging 5-of-8. The parquet fetches are filename-addressed (re-curl is a 27 MiB no-op).

**Note:** lane5's driver adds the per-example merge and the exact-n assert that
`eval_paperC_squad_emf1.py` lacks (its `merge()` only sums n/em_hits/f1_sum and globs whatever
`shard*of*.json` exists — no shard-count assert). That gap is real and the driver closes it.

---

## 4. NO GPU NEEDED — do these immediately, in this order

### 4.1 ★ A01 gate-2 — ALREADY ANSWERED, just needs persisting (~10 min CPU)

Lane2's `NOT_A_GPU_JOB` is the best outcome in the whole set: **gate-2 is answered and it does not
fail.** I independently reproduced the BoolQ headline to 4 decimals (§1.3). Two non-MMLU MC
benchmarks reproduce the interface/floor failure with conclusion-changing force:

* **BoolQ** (n=3270): construct-appropriate null **0.6217** (always-B) vs 0.5000 naive chance =
  **12.2 pp** understatement. keep12/keep10/keep8 all sit **below** the floor on raw sum-LL while
  keep14 is n.s. (p=0.203) — 4 of 6 arms indistinguishable from a constant predictor despite
  nominally "passing" accuracies. keep12 **flips verdict across interfaces** (below floor under raw
  LL, significantly above under acc_norm, p=2.2e-06).
* **OpenBookQA** (n=500): longest-option-split null **0.3635** vs 0.2500 = **11.4 pp**
  understatement; drops base's residual fraction 45.9% → 21.3% (**2.15×** inflation); keep10 goes
  **below** floor; 3/15 acc-vs-acc_norm sign flips.

**Run:** lane2's STEP 2 + STEP 3 + STEP 4 from `status/scout_21/lane2_a01_gate2.md`, on **LOCAL**
(scipy 1.18.0 present; `.21` has none). **Skip lane2's STEP 1** — it mutates canonical
`per_example_*.jsonl` in place and needs the proxy; lane2's own STEP 2 derives `norm_lens`
in-memory and never touches the jsonl. Take the snapshot (STEP 0) only if you do choose STEP 1.

Two caveats to carry into the writeup:
* **Exclude winogrande** — structurally degenerate. `load_task_examples` builds a *shared*
  continuation, so both options get identical `norm_lens` (I verified: `[28, 28]`), making
  length-norm a no-op (`acc == acc_norm` exactly, longest-option null exactly .5000 at 100% tie
  rate). Report as a structural negative control only.
* CommonsenseQA (5-way, null .2088) and SocialIQA (3-way, null .3362) are the
  **calibration-survives** counter-cases — all 6 arms clear both. Good for "not everything
  collapses"; do not lead with them.

### 4.2 ★ A01 gate-1 — per-family floor recomputation (I already did it; just persist)

Gate-1's GPU work is done (§1.2). The only remaining step is applying the *correct* nulls, because
the harness's own `above_chance` fields use CHANCE=0.25 which A01 explicitly forbids. I computed
this from the per-example files:

| family | n | always-D | longest-split | letter resid (frac) | content resid (frac) |
|---|---|---|---|---|---|
| OLMo-2-7B (`7B_base`) | 14042 | 0.2689 | **0.2845** | +0.3365 (55.6%) | +0.1861 (39.6%) |
| Llama-2-7B | 14042 | 0.2689 | **0.2757** | +0.1411 (34.4%) | +0.1378 (33.3%) |
| Llama-3-8B | 14042 | 0.2689 | **0.2847** | +0.3531 (56.8%) | +0.1777 (38.4%) |
| Qwen3-8B-Base | 14042 | 0.2689 | **0.2833** | +0.4775 (64.0%) | +0.2340 (45.2%) |

Two facts MAIN must record:
* **always-D = 0.2689 is family-invariant** (it is a property of the gold label marginals).
* **The content longest-option split-tie null is TOKENIZER-DEPENDENT**: .2845 / .2757 / .2847 /
  .2833. Reusing OLMo's .2845 for the Llama-2 arm overstates its floor by 0.88 pp. Each family's
  floor must be recomputed from its own `cont_tokens` (the field is already emitted per item).

**Scope honestly:** all three families are **INTACT bases**, so this reproduces Obs-1 (healthy
model ⇒ letter interface works) on 3 new families. It **cannot** test A01's load-bearing headline
(letter falls *below* .2689 on *damaged* arms) — no damaged third-family arm exists on either disk
(`grep -l "llama\|mistral\|gemma\|pythia" outputs/*/arch_meta.json` → zero; every prune-heal
`arch_meta` is olmo2/qwen3/hunyuan/hy_v3). A damaged arm needs `--truncate_layers` (verified absent:
`grep -c truncate_layers` on both harnesses = 0). Defer that.

Note also `build_null_calibration_table.py` hardcodes `MMLU_DIR='olmo2_mmlu_content_results'`
(line 76) and a fixed 9-arm `MMLU_ARMS` list (85-95) with no path CLI args, so a third-family
analysis needs a new/parameterised script, not just a rerun.

### 4.3 B04 novelty check (task #207) — CPU/literature only

`DIRECTION_A_VERDICT.md`'s promotion table has exactly one ✗: novelty. If .21's replication comes
back SURVIVES, this is the *only* remaining barrier to promoting B04 to `paper<X>`. Doing it in
parallel with the GPU run means the promotion decision lands the same night.

### 4.4 A02 canonical Read-LoRA provenance (~20 min grep)

Lane3 resolved lane4's `outputs/lora_best_ref` red herring: it **does not exist on wzc1 at all**
(not even a dangling symlink; `readlink -f` echoes the path unchanged). The real Read adapter is
`outputs/qcmem_distill_qwen_j12_r32_4k/final/` (adapter_model.safetensors 232829168 B), which is the
argparse default `--lora_adapter` of eval_p016 (line 1033) / p017 (1156) / p018 (1363) and is named
by A02's `SOURCES.md`, with **identical md5 on both disks**. Worth pinning in writing so the next
agent does not re-chase it.

---

## 5. DO NOT LAUNCH — with reasons, so this is not re-litigated

| gate | reason |
|---|---|
| **A01 gate-1 (any family)** | **Already finished on .21 at 21:41:52** — Llama-2-7B, Llama-3-8B, Qwen3-8B-Base all n=14042 / n_nan=0 / 8-of-8. Lane1's ".21 is occupied" blocker is stale and its Llama-3-on-.82 plan would redo work on disk. Only CPU floor recomputation remains (§4.2). |
| **A01 gate-1 "damaged third-family arm"** | Needs code that does not exist: `grep -c truncate_layers` = 0 in both harnesses; `load_pruned_model`/`build_pruned_shell` are OLMo-locked (Olmo2Config → Olmo2ForCausalLM → strict load) and the new `--any_family` flag explicitly **refuses** `--ckpt`. ~15 new lines. Not tonight. |
| **A01 gate-2** | `NOT_A_GPU_JOB` and **already answered**. Occupying a node for it would be pure waste. See §4.1. |
| **A02 stage-1 (p018 BBWL)** | The zero-code runnable version **already ran on .82 on Aug 4** — `bench_results/p0_18_e4_bbwl_step{1000,1500,2000}` verified present, task #150 completed, distilled into `paperA/artifacts/writepath_distill_150/summary.json`. Re-running reproduces a known number. The 32k extension is decoration. |
| **A02 stage-1 (the version that CAN kill)** | `BLOCKED_NEEDS_CODE`. `grep write_lora_ckpt` = **0** in all five natural-task harnesses (longeval/babilong/locomo/longbench, eval_ruler_qcmem). Only `eval_p018_e4_2x2_writecontrol.py` is write-capable and it is RULER-synthetic-only. Porting `_load_with_write_lora` is a 1-2 day coder task across 4-5 files. Also p018 raises SystemExit if `--lora_adapter` is empty, so A02 config 4 as written is not even expressible. |
| **A03 remaining axes via LongMemEval** | Length-infeasible at 1B: OLMo-2-0425-1B `max_position_embeddings=4096`, but knowledge-update items are p50 ~6780 est. tokens — **0 of 78 fit** even at 3500. n=78 also thin for the BH family. Use CounterFact instead (lane5's plan does). |
| **A03 via LongBench multihop** | Same problem: hotpotqa/2wikimqa/musique are on both disks (200 items each) but packed to p50 10145/4218/11388 tokens; only 10/200 hotpot and 1/200 musique fit ≤3500. n=200 too thin. |
| **A03 "new facts" axis** | `BLOCKED_MISSING_ASSET`. No FreshQA/TempLAMA/post-cutoff set on **either** disk (searched data/, .hf_cache/, and both `/root/.cache/huggingface/hub`). Needs a generated synthetic-injection set. Must not gate the other two axes. |
| **B04 on `.82`** | Wrong disk. 62 GB of the 5 Qwen ckpts are **absent on zwfy6** (verified per-arm, §1.6) → 1.4-4.3 h of transfer for zero benefit. B04 belongs on .21. |
| **A03 on `.21`** | Wrong disk **and** wrong environment: 22.6 GiB cross-disk copy, plus `.21` has no scipy → `mcnemar_exact()` silently returns nan, dropping a required statistic. |
| **Re-running B04 Direction A (OLMo)** | Established at the exact-permutation lower bound (ρ=+1.00 / −1.00, p=0.0028, n=6). `DIRECTION_A_VERDICT.md`: "Result cannot be more significant with this design." |
| **Anything on `.73` / `.104` / LOCAL** | .73 = A01 gate-3; .104 = keep12 training; LOCAL = keep14 seed1234 training. Out of scope. |

**Do not kill `.21` PID 25999** (`prepare_dolmino_llama2.py --stage download`, CPU+network only,
elapsed 1:20). It does not conflict with GPU work. Do not compete for its bandwidth — which is
another reason the B04 launch is configured **fully offline** (§1.4).

---

## 6. DOCUMENTATION BUGS — concrete fixes

### 6.1 B02 still carries a premise MAIN already falsified (2 files)

`proposal/MINIMAL_VALIDATION_PLAN.md` §3 falsified the "reuse existing sweeps" premise: the 8 T21
configs each used **different 50 samples**, question-md5 intersection **0/50** for all 7 pairs vs
j3. Both B02 files still tell the next agent to walk that dead path. I verified both texts; I did
**not** edit them.

**FIX 1 — `proposal/backlog/B02-adaptive-depth-and-read-budget/PROPOSAL.md`, §Stage 0:**
change `使用现有 sweep 计算 per-example oracle action 和 regret：`
to `⚠️ 修正 (2026-08-08)：不能复用现有 sweep —— 8 个 T21 config 各用不同的 50 条样本，question-md5 交集对 j3 的 7 个配对全为 0/50（见 proposal/MINIMAL_VALIDATION_PLAN.md §3）。oracle headroom 需要一次新的 GPU run，在同一批样本上跑全部候选 config：`

**FIX 2 — `proposal/backlog/B02-adaptive-depth-and-read-budget/STATUS.json`:**
change `"next_gate": "measure per-example oracle headroom from existing sweeps"`
to `"next_gate": "NEW GPU run: paired per-example oracle headroom on ONE shared sample set (existing sweeps unusable, 0/50 question overlap)"`
and add `"gate_class": "needs_new_gpu_run"`.

### 6.2 A01 gate spec: mark gate-1's GPU leg done, and scope it honestly

`proposal/active/A01-null-calibration-methodology/STATUS.json` still lists
`"third model family"` as an open `next_gate`, but three families completed at 21:41 (§1.2).

**FIX 3 —** change the `next_gate` entry `"third model family"` to
`"third model family: INTACT-BASE leg DONE 2026-08-08 (gate1_llama2_7b / gate1_llama3_8b / gate1_qwen3_8b_base, n=14042 each); REMAINING = damaged-arm leg, needs --truncate_layers (~15 lines, absent from harness)"`.

**FIX 4 —** in the same file (or A01's `PROPOSAL.md` §不得复活的旧数字), record the tokenizer
dependence: `content longest-option split-tie null 是 tokenizer 相关的：OLMo-2 .2845 / Qwen3-8B .2833 / Llama-3-8B .2847 / Llama-2-7B .2757。always-D = .2689 是 family-invariant。不得跨家族复用 .2845。`
This is the same class of error as the already-retracted `.2822`-vs-`.2845` mistake.

### 6.3 CLAUDE.md is stale about `models/Qwen3-8b-local`

CLAUDE.md's pitfall list says paperB/TODOList's `models/Qwen3-8b-local` "在 zwfy6 不存在". Lane3
verified on .82 that `models/Qwen3-8b-local -> /apdcephfs_zwfy6/.../models/Qwen--Qwen3-8b` **exists
and resolves** (mtime Jul 11 17:21), and the same symlink exists on wzc1.

**FIX 5 — CLAUDE.md**, the `⚠️ Qwen base 路径名错` bullet: change
`paperB/TODOList 写的 models/Qwen3-8b-local 在 zwfy6 不存在`
to `⚠️ 已过期 (2026-08-08 实测纠正)：models/Qwen3-8b-local 在 zwfy6 和 wzc1 上都存在，是指向 models/Qwen--Qwen3-8b 的 symlink，harness 默认 --model_path 可直接用，无需 override。注意 512 字节的 du 结果是 symlink 大小，不代表占位文件。`

### 6.4 B04 analyzer docstring says n=5 but the verdict is n=6

`proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_5rung.py` docstring says
"Exact permutation p at n=5 (5! = 120 perms)" and `RUNGS` lists 5, but
`DIRECTION_A_VERDICT.md` reports n=6 with 720 perms (shortgpt16 added). The code is generic
(`permutations(y)`), so this is documentation-only — but it misleads.

**FIX 6 —** update the docstring to `Exact permutation p at n=len(RUNGS) (n=6 -> 720 perms, two-sided floor 2/720 = 0.00278)` and add the shortgpt16 rung to the `RUNGS` literal so the file matches the reported result.

### 6.5 Long-term harness fix (not a doc bug, but same root cause)

The `norm_lens`/`norm_scores` addition of 2026-08-08 landed **only** in
`eval_olmo2_probe2_downstream.py` (lines 470-475), not in `eval_qwen3_probe2_downstream.py`. That
asymmetry is a silent-wrong-answer hazard for **any** acc_norm margin analysis on Qwen, because the
B04 analyzer falls back to raw `option_scores` without warning. §2 STEP 1 fixes it at the source.
Recommend committing that patch regardless of whether B04 launches tonight.

---

## 7. Residual risks on my own recommendation

* **B04 batch size on .21 is untested.** I did not run a GPU probe (read-only scope). bs16 is the
  OLMo ladder's setting and the Qwen shells are 14-16L (smaller than OLMo's 32L base), but the
  base full-36 rung is a full fp32 8B load with vocab 151936. If rung 0 OOMs, drop to `BS=8` — the
  skip-if-`summary.json` guard makes a partial-then-resume safe.
* **Qwen damage axis is not depth-matched to OLMo's** (verified from `arch_meta.json`: mostly
  heal-steps at keep12+fresh2, one depth variant, one from-scratch). A sign replication is still
  informative, but this is a **different damage axis**, and the writeup must say so.
* **A03's HotpotQA yes/no prior** (best-constant EM .1567 on comparison vs .0035 on bridge) will
  silently inflate the apparent floor-clearing if not stratified. Flagged in §3.
* I did **not** independently re-verify lane2's OpenBookQA longest-option null (0.3635) or lane5's
  measured token-length distributions. BoolQ reproduced exactly, which raises my confidence in
  lane2's method, but the OBQA number is `unsure` at my own verification level.
