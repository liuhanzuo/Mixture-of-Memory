# B04 read-out eval-fill — READY TO LAUNCH. 2026-08-16, 0 GPU, PRE-DATA

> **Headline.** The 6 missing read-out arms are ready to evaluate in one command. The driver
> `scripts/_run_b04_readout_evalfill.sh` replicates the protocol of record byte-for-byte,
> asserts `n_scored` per task before calling any rung complete, and its `--dry-run` passes
> (exit 0, pasted verbatim in §3).
>
> **The one number I disagree with.** The artifact's **1.61 GPU-h** rests on a **warm-page-cache**
> anchor. `logs/sv181_main.log:5-6` measures 121 s for a stage whose checkpoint had already been
> read off CephFS by the PPL stage 6 minutes earlier. Every one of the 6 new rungs is a **first
> touch** of a 48.7 GB file. Re-derived from cold-load anchors in the same log:
> **≈4.83 occupancy GPU-h** (8 cards × ~36 min), of which only **≈0.44 GPU-h** is scoring
> compute; the rest is IO. The task is still cheap and still worth doing — but it is **3.0×**
> the artifact's figure, and `PREFETCH=1` can move ~1.7 GPU-h of it off the GPU clock. §4.
>
> **Scope.** This document prepares the launch. It does **not** modify the gate, `STATUS.json`'s
> verdict, or `DECIDABILITY_FIX_20260816.md`. The verdict today remains `READOUT_ABSENT`.

---

## 1. The protocol of record, with provenance

The single margin-computable read-out arm is `olmo2_downstream_results/keep14_s1234_step200000_sv181`
(`median_margin` 0.108500, n = 17195). It was produced by **stage (2) core6 downstream** of
`scripts/_run_paperB_keep14_seedvar_local.sh`. Verified at **lines 112-127** — the artifact cites
"116-125", which is where the invocation sits; the guard at :114 and the merge at :126-127 are part
of the same stage and matter, so I quote the wider block.

```
scripts/_run_paperB_keep14_seedvar_local.sh:112-127
  112  # ---------------- (2) core6 downstream ----------------
  113  NAME="keep14_${TAG}_step200000${SUFFIX}"
  114  guard olmo2_downstream_results "$NAME"
  116  $PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$CORE_TASKS"
  118  for g in $(seq 0 $((NGPU-1))); do
  119    CUDA_VISIBLE_DEVICES=$g LOCAL_RANK=0 RANK=$g $PY scripts/eval_olmo2_probe2_downstream.py \
  120      --base_model "$BASE" --ckpt "$CKPT" --tasks "$CORE_TASKS" \
  121      --num_shards 8 --shard_index $g --batch_size 8 \
  122      --save_per_example --output_name "$NAME" \
  124  done; wait
  126  assert_shards "olmo2_downstream_results/$NAME"
  127  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME"
```

with `PY=/opt/conda/envs/torch-base/bin/python` (:48), `BASE=../models/OLMo-2-1124-7B` (:49),
`NGPU=8` (:53), `CORE_TASKS=hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa` (:63),
and env `HF_DATASETS_OFFLINE=1` / `HF_HUB_OFFLINE=1` (:57-58).

### Fields that must be byte-identical across all 7 rungs

If any of these differs, the rungs are not same-harness and φ is not a measurement of budget
response — it is partly a measurement of the harness.

| field | value | why it is load-bearing |
|---|---|---|
| harness | `scripts/eval_olmo2_probe2_downstream.py` @ **git `a163a89`** (2026-08-08) | **The one confound with measured teeth.** `status/PAPERB_WITHIN_DISK_FLOOR_V3.md:37-38`: same-driver same-disk re-runs are **byte-identical (0 flips)**; the *only* non-zero within-disk comparison **crosses a driver boundary**. Driver drift is a systematic bias, not zero-mean noise, so it cannot be averaged away. |
| base_model | `../models/OLMo-2-1124-7B` (32 layers) | wrong base ⇒ wrong transplant |
| tasks + order | the 6 core tasks above | order fixes `item_id` assignment |
| `--num_shards` | **8** | `item_id = shard_index + ei*num_shards` (harness :412). Changing shard count **renumbers every item**, silently breaking any per-item pairing. |
| `--shard_index` | 0..7, one per GPU | — |
| `--batch_size` | **8** | batching changes padding ⇒ can change log-probs at the last digit |
| `--max_len` | 1024 — **default, not passed** | passing it explicitly is fine numerically, but I keep the record's exact flag set |
| `--add_bos` | 0 — **default, not passed**; asserted `add_bos=False` in `summary.json` | OLMo-2 published numbers are made without BOS |
| `--save_per_example` | **REQUIRED** | without it `median_margin` is not computable; the run becomes another `7B_keep8_step100000` (correct core6, primary metric absent) |
| `--keep_front_layers` / `--n_fresh_layers` | **NOT passed** — read from ckpt meta (14/2) | the record omits them for the *downstream* stage (passes them only to PPL/MMLU). `load_pruned_model` (`eval_olmo2_probe2_ppl.py:84-97`) reads them from the ckpt and *raises* if a CLI value disagrees, so omitting is both faithful and safe. |
| chat template | **absent by construction** | `grep -nE "chat_template|apply_chat_template" scripts/eval_olmo2_probe2_downstream.py` → **zero hits**. Not a flag that could be flipped: the base-LM protocol is structural. OLMo-2-1124-7B is a **BASE** LM. |
| arch | **sm_100 only** (LOCAL or `.21`) | comparator ladder + σ̂ are sm_100/wzc1; an H20 arm confounds run-to-run with hardware |
| determinism | no seeds needed | `grep -nE "manual_seed|use_deterministic|cudnn"` on both harness files → **zero hits**, consistent with FLOOR_V3's 0-flip finding: this eval is deterministic given fixed driver + arch. |

### The margin definition, confirmed

`proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py:172-201` — confirmed
at the cited location:

```python
sc  = o["norm_scores"]                     # length-normalised, from per_example
g   = o["gold_letter"]
oth = [v for k, v in sc.items() if k != g and v is not None]
out.append(abs(sc[g] - max(oth)))          # |score(gold) - max(other)|
...
if n != EXPECTED_N[t]:  sys.exit(f"PROTOCOL_VIOLATION: ...")
if len(out) != EXPECTED_POOLED: sys.exit(f"PROTOCOL_VIOLATION: ...")
```

So the read-out needs `norm_scores`, `norm_lens`, `gold_letter`, `item_id` per row — exactly what
`--save_per_example` writes (harness :456-474, added in `a163a89`). Confirmed present in the
existing step200000 dir.

---

## 2. `n_scored` expectations per task

`EXPECTED_N` at `analyze_b04_wzc1_floor.py:85-87`; independently confirmed against both the
existing `summary.json` and its `per_example_*.jsonl` line counts.

| task | expected `n_scored` | `summary.json` (step200000) | `per_example` rows |
|---|---:|---:|---:|
| hellaswag | 10042 | 10042 | 10042 |
| arc_challenge | 1172 | 1172 | 1172 |
| arc_easy | 2376 | 2376 | 2376 |
| piqa | 1838 | 1838 | 1838 |
| winogrande | 1267 | 1267 | 1267 |
| openbookqa | 500 | 500 | 500 |
| **pooled** | **17195** | **17195** | **17195** |

Plus `n_nan == 0` on every task, `n_shards == 8`, `add_bos == False`, `meta.keep/fresh == 14/2`.

### The completeness oracle, and proof it fails loudly

A rung counts as complete only if **all** of the above hold. The same predicate decides
skip-on-resume and accepts a freshly finished rung, so the two cannot disagree. It returns
rc 0 = complete, 1 = absent/incomplete, **2 = present but MALFORMED (hard stop, never reused,
never clobbered)**.

I exercised it against six synthetic corruptions of a copy of the real dir (throwaway tmpdir,
removed afterwards; nothing was written to `olmo2_downstream_results/`):

| injected fault | result |
|---|---|
| pristine copy | `rc=0 COMPLETE ckpt_step=200000 pooled=17195` |
| **only 5 of 8 shard files** (the historical silent-merge disaster) | `rc=1 INCOMPLETE 5/8 shard files` |
| 8 shards but `summary.n_shards=5` (partial merge that *looks* complete) | `rc=2 MALFORMED summary.n_shards=5 != 8 -- PARTIAL MERGE` |
| `per_example_hellaswag.jsonl` truncated by 100 rows | `rc=2 MALFORMED ... has 9942 rows, expected 10042` |
| `norm_scores` stripped from openbookqa | `rc=2 MALFORMED ... lacks 'norm_scores' -- margin not computable` |
| right dir name, **wrong checkpoint** (seed42 substituted) | `rc=2 MALFORMED meta.ckpt=...keep14fresh2/step200000.pt is not a keep14fresh2_seed1234 checkpoint` |
| ask for step 175000, dir holds 200000 | `rc=2 MALFORMED meta.ckpt_step=200000 != expected 175000` |

The last two matter more than they look: a dir with the right *name* but the wrong *checkpoint*
would feed a wrong `y` into φ and nothing downstream would notice. The oracle pins
`meta.ckpt` to `keep14fresh2_seed1234` **and** `meta.ckpt_step` to the step the filename claims.

---

## 3. Launch command + the real `--dry-run` output

### One-command launch, when a sm_100 node frees

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory && \
setsid nohup bash scripts/_run_b04_readout_evalfill.sh \
  > logs/b04_evalfill_launch_$(date +%Y%m%d_%H%M%S).out 2>&1 &
```

Validate first (no GPU, ~4 s): `bash scripts/_run_b04_readout_evalfill.sh --dry-run`

Useful overrides: `PREFETCH=1` (move the cold read off the GPU clock, §4),
`INCLUDE_200K=1` (add a 7th rung that re-derives the archived point under this driver — turns
"same-harness" from assumption into measurement), `STEPS="100000 128000 153500 175000"`
(GRID_I only, 4 rungs), `SUFFIX=...`, `PY=...`, `NGPU=...`.

The driver **refuses to launch** if any GPU has >2 GiB allocated (exit 5), so it cannot be
accidentally dropped on top of a live training run. `--dry-run` only warns.

### Pasted real output (executed 2026-08-16 11:21, LOCAL, 0 GPU touched, **exit 0**)

```
[2026-08-16 11:21:04] ===== B04 read-out eval-fill  run_id=b04_evalfill_20260816_112104  dry_run=1 =====
[2026-08-16 11:21:04] WD=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
[2026-08-16 11:21:04] PY=/opt/conda/envs/torch-base/bin/python
[2026-08-16 11:21:04] --- 1. interpreter (tested by execution, not assumed) ---
  [ ok ] /opt/conda/envs/torch-base/bin/python -> py 3.14.6 torch 2.13.0 transformers 5.15.0 arch_list=True
[2026-08-16 11:21:06] --- 2. harness + analyzer files ---
  [ ok ] scripts/eval_olmo2_probe2_downstream.py
  [ ok ] scripts/eval_olmo2_probe2_ppl.py
  [ ok ] proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py
  [ ok ] harness last commit: a163a89 2026-08-08
  [ ok ] harness is clean vs git (no driver boundary)
[2026-08-16 11:21:07] --- 3. base model ---
  [ ok ] ../models/OLMo-2-1124-7B (num_hidden_layers=32)
[2026-08-16 11:21:07] --- 4. HF dataset caches (offline mode is ON) ---
  [ ok ] cache Rowan___hellaswag
  [ ok ] cache allenai___ai2_arc
  [ ok ] cache ybisk___piqa
  [ ok ] cache allenai___winogrande
  [ ok ] cache allenai___openbookqa
[2026-08-16 11:21:07] --- 5. checkpoints: existence + EXPECTED BYTE SIZE ---
  [ ok ] step25000 outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step25000.pt size 48724473567 == expected
  [ ok ] step50000 outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step50000.pt size 48724473567 == expected
  [ ok ] step100000 outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step100000.pt size 48724474298 == expected
  [ ok ] step128000 outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step128000.pt size 48724474298 == expected
  [ ok ] step153500 outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step153500.pt size 48724474298 == expected
  [ ok ] step175000 outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step175000.pt size 48724474298 == expected
[2026-08-16 11:21:07] --- 6. output dirs: writability + per-rung completeness ---
  [ ok ] olmo2_downstream_results is writable
  [ ok ] logs/ is writable
  [ ok ] step25000 -> keep14_s1234_step25000_b04fill to be evaluated (INCOMPLETE dir does not exist)
  [ ok ] step50000 -> keep14_s1234_step50000_b04fill to be evaluated (INCOMPLETE dir does not exist)
  [ ok ] step100000 -> keep14_s1234_step100000_b04fill to be evaluated (INCOMPLETE dir does not exist)
  [ ok ] step128000 -> keep14_s1234_step128000_b04fill to be evaluated (INCOMPLETE dir does not exist)
  [ ok ] step153500 -> keep14_s1234_step153500_b04fill to be evaluated (INCOMPLETE dir does not exist)
  [ ok ] step175000 -> keep14_s1234_step175000_b04fill to be evaluated (INCOMPLETE dir does not exist)
[2026-08-16 11:21:07] --- 7. reference rung (the protocol of record) ---
  [ ok ] keep14_s1234_step200000_sv181 COMPLETE ckpt_step=200000 pooled=17195 (same oracle passes on the archived rung)
[2026-08-16 11:21:07] --- 8. GPUs (read-only query; no CUDA context created) ---
  [ ok ] 8 GPUs visible (need 8)
  [ ok ] device name string: NVIDIA L20A  (NOTE: 'L20A' on LOCAL/.21 is a name-string
  [ ok ]   display bug; real hardware is B200/sm_100. Judge by capability, not name.)
  [WARN] 8/8 GPUs currently have >2 GiB allocated.
         This driver needs all 8 idle. Do NOT launch on top of a live run.
  [ ok ] compute_cap 10.0 = sm_100 (Blackwell/B200) -> OK
[2026-08-16 11:21:07] --- 9. host resources ---
  [ ok ] MemAvailable 1598 GiB (need ~180 GiB for 8x fp32 4.06B + page cache)
[2026-08-16 11:21:07] --- 10. PLAN ---
[2026-08-16 11:21:07]   protocol : eval_olmo2_probe2_downstream.py, 8 shards, batch_size 8,
[2026-08-16 11:21:07]              --save_per_example, add_bos 0 (default), max_len 1024 (default),
[2026-08-16 11:21:07]              chat_template absent by construction, keep/fresh from ckpt meta
[2026-08-16 11:21:07]   tasks    : hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa
[2026-08-16 11:21:07]   expect   : per-task n_scored hellaswag=10042 arc_challenge=1172 arc_easy=2376 piqa=1838 winogrande=1267 openbookqa=500
[2026-08-16 11:21:08]              pooled 17195, n_nan 0, 8/8 shards
[2026-08-16 11:21:08]   to run   : 25000 50000 100000 128000 153500 175000   (6 rungs)
[2026-08-16 11:21:08]   skipping : (none)   (0 already complete)
[2026-08-16 11:21:08]   outputs  : olmo2_downstream_results/keep14_s1234_step<STEP>_b04fill/
[2026-08-16 11:21:08]     step25000: outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step25000.pt -> olmo2_downstream_results/keep14_s1234_step25000_b04fill
[2026-08-16 11:21:08]     step50000: outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step50000.pt -> olmo2_downstream_results/keep14_s1234_step50000_b04fill
[2026-08-16 11:21:08]     step100000: outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step100000.pt -> olmo2_downstream_results/keep14_s1234_step100000_b04fill
[2026-08-16 11:21:08]     step128000: outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step128000.pt -> olmo2_downstream_results/keep14_s1234_step128000_b04fill
[2026-08-16 11:21:08]     step153500: outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step153500.pt -> olmo2_downstream_results/keep14_s1234_step153500_b04fill
[2026-08-16 11:21:08]     step175000: outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step175000.pt -> olmo2_downstream_results/keep14_s1234_step175000_b04fill
[2026-08-16 11:21:08]   cost     : ~362s/rung wall = cold ckpt read ~325s + scoring ~33s + merge ~4s
[2026-08-16 11:21:08]              MEASURED anchors (EVAL_FILL_READY_20260816.md sec 4), NOT copied:
[2026-08-16 11:21:08]                COLD load  logs/sv181_main.log:2->  + sv181_ppl_*_shard*.log = 338-348s
[2026-08-16 11:21:08]                WARM load  logs/sv181_main.log:5-6 stage 121s, of which 89s load + 30s score
[2026-08-16 11:21:08]                The artifact's 121s/rung anchor is the WARM stage: the PPL stage
[2026-08-16 11:21:08]                had already read that same ckpt minutes earlier. Each of the 6 new
[2026-08-16 11:21:08]                rungs is a FIRST touch, so 121s understates it ~3x.
[2026-08-16 11:21:08]              6 rungs -> ~36 min wall, ~4.83 occupancy GPU-h (8 cards held)
[2026-08-16 11:21:08]              compute-only (scoring, excluding the IO-bound read): ~0.44 GPU-h
[2026-08-16 11:21:08]              set PREFETCH=1 to warm the page cache per rung on CPU first; that
[2026-08-16 11:21:08]              moves the 325s off the GPU clock (~1.68 GPU-h) but does not change the total wall.
[2026-08-16 11:21:08] SELF-CHECK PASSED (6 to run, 0 complete)
[2026-08-16 11:21:08] --dry-run: validation only. No GPU touched, nothing written to olmo2_downstream_results/.
=== EXIT CODE: 0 ===
```

Note the `[WARN]` is the correct behaviour, not a defect: all 40 cards are mid-training today, so
the busy-detector fired. In a real launch that same condition exits 5 instead of warning.

---

## 4. Cost, re-derived from a named measured anchor — **I disagree with 1.61 GPU-h**

### The artifact's basis, and why it is the wrong stage

`DECIDABILITY_FIX_20260816.md:402-406` anchors on `logs/sv181_main.log` lines 5-6:

```
logs/sv181_main.log:5  [2026-08-12 01:12:18] (2) core6 downstream -> keep14_s42_step200000_sv181
logs/sv181_main.log:6  [2026-08-12 01:14:19] OK 8/8 shards in .../keep14_s42_step200000_sv181
```

121 s × 8 GPUs / 3600 = 0.268889 GPU-h/rung × 6 = **1.613 GPU-h**. The arithmetic is right and the
lines say what the artifact claims. **The problem is which stage those lines describe.**

That core6 stage was **stage (2)**. **Stage (1)** — in-domain PPL, `logs/sv181_main.log:2-4` — had
already loaded *the same 48.7 GB checkpoint* starting at 01:06:05, finishing 01:12:14. So by
01:12:18 the checkpoint was **hot in the host page cache**. The 121 s is a warm-cache figure.

### Decomposition, measured from the per-shard logs

Per-shard logs timestamp the `loaded ckpt` line, which splits load from scoring:

| quantity | s42 | s1234 | source |
|---|---:|---:|---|
| **COLD** ckpt load (stage 1, first touch, 8 procs) | **348 s** | **338 s** | `sv181_main.log:2`/`:31` → max over `sv181_ppl_*_shard{0..7}.log` `loaded ckpt` |
| **WARM** ckpt load (stage 2, same ckpt cached) | **89 s** | **89 s** | `sv181_main.log:5`/`:34` → max over `sv181_core_*_shard{0..7}.log` `loaded ckpt` |
| core6 scoring after load | **30 s** | **32 s** | last `loaded ckpt` → last `wrote shard` |
| merge (incl. per-example concat) | ~3-4 s | ~4 s | `sv181_main.log:6→7`, `:35→36` |

Cold ≈ 343 s vs warm 89 s — a **3.9× IO difference**, and it dominates the stage.

Independent corroboration that this is IO, not a one-off: single-stream cold read of an
**unevaluated** rung measures **0.97 GB/s** (`dd iflag=direct`, 3 GB, ×3 files, 973/976/968 MB/s),
so 48.7 GB ≈ 50 s at full single-stream rate; 8 concurrent O_DIRECT streams aggregate only
**2.57 GB/s** (16 GB in 6.22 s), i.e. **~19 s of pure bandwidth** — the remaining ~320 s is
`torch.load` unpickling ~179 fp32 tensors plus CephFS latency across 8 competing processes. And
the 48.7 GB file does **not** stay resident: a re-read of a 3 GB slice right after a buffered read
returns 4.8 GB/s (cached), but re-reading slices of a fully-read 48.7 GB file returns 1.1 GB/s —
it is evicted. So **each rung pays the cold price**; there is no cross-rung reuse to exploit.

### My figure

Per rung: 325 s (cold load, conservative midpoint of 338/348 minus the ~20 s of scoring overlap
already counted) + 33 s scoring + 4 s merge ≈ **362 s**.

```
6 rungs × 362 s            = 2172 s ≈ 36.2 min wall
× 8 GPUs / 3600            = 4.83 occupancy GPU-h     <- what the node actually costs
scoring compute only:
6 × 33 s × 8 / 3600        = 0.44 GPU-h               <- what the GPUs actually compute
```

| basis | GPU-h | comment |
|---|---:|---|
| artifact (warm anchor, 121 s) | 1.61 | understates: anchors on a pre-warmed stage |
| **mine — occupancy, cold** | **4.83** | **the honest booking figure; 3.0× the artifact** |
| mine — occupancy with `PREFETCH=1` | ~1.68 | cold read moved to CPU; GPUs held only ~126 s/rung |
| mine — scoring compute only | 0.44 | the artifact's 1.61 sits *between* compute and occupancy, matching neither |

**I agree with the artifact's conclusion and disagree with its number.** The direction is
unchanged — this is a trivially cheap fill, ~36 minutes of one node either way, and nothing about
the decision to do it depends on 1.61 vs 4.83. But per
`memory/ckpt-interval-rate-is-not-compute-rate.md` ("report must say whether it is compute or
amortised") the booking figure should be **4.83 occupancy GPU-h**, or **~1.68** with `PREFETCH=1`.

Wall-clock either way: **~36 min on 8 idle sm_100 cards.**

---

## 5. Contradictions and gaps found while reading

Reported per instruction; **nothing here rewrites the pre-registration.**

### 5.1 The cost anchor is a warm-cache measurement (§4) — **material**
1.61 GPU-h understates occupancy by 3.0×. Both numbers are small; the correction matters for
booking honesty, not for the go/no-go.

### 5.2 ⚠️ Revision 3's gate is **documented but not implemented** — the biggest gap
`DECIDABILITY_FIX_20260816.md` §3 specifies `GRID_I = {100000,128000,153500,175000,200000}`,
`S_I = 100000`, and "take the MORE SEVERE of verdict_I and verdict_W". **None of that exists in
the analyzer.** `grep -nE "GRID_I|GRID_W|153500|combine"` over
`code/analyze_b04_wzc1_floor.py` → **zero hits**. The code still hard-codes revision 2:

```python
analyze_b04_wzc1_floor.py:120  G1_READOUT_STEPS = [25000, 50000, 100000, 128000, 200000]
analyze_b04_wzc1_floor.py:121  READOUT_SPAN = max(...) - min(...)   # 175000
analyze_b04_wzc1_floor.py:139  if steps != G1_READOUT_STEPS: sys.exit("PROTOCOL_VIOLATION ...")
```

> **PRECISION FIX by MAIN, 2026-08-16.** The sentence below originally read *"the analyzer never
> reads a read-out dir at all"*. That is **overstated**: the analyzer does read
> `olmo2_downstream_results/` — `:161` and `:176` open `per_example_{task}.jsonl`, `:213` counts
> `shard*of8.json`, `:219` and `:287` load `summary.json` — it uses those for the donor margin and
> the completeness checks. The accurate and still-decisive statement is narrower:
> **nothing that is read from disk ever reaches `phi_budget`.** All three call sites (`:503`,
> `:526`, `:531`) are inside `selftest_phi()` on hand-written vectors, so there is no path from a
> read-out directory to the decision statistic. MAIN verified both halves by enumerating every
> `phi_budget` occurrence with its enclosing `def`, and by grepping the grid constants: `:120` is
> still revision 2's `[25000, 50000, 100000, 128000, 200000]` with `READOUT_SPAN = 175000`, and
> `GRID_I` / `153500` / `S=100000` appear nowhere. The consequence for GPU spend is unchanged.

**The analyzer reads read-out directories, but nothing it reads reaches φ.** The only
`phi_budget()` call sites are
`:503`, `:526`, `:531` — all inside `selftest_phi()`, all fed hand-written y-vectors. There is no
code path that loads `median_margin` at the read-out steps from disk. So **filling the 6 evals
does not by itself make the gate fire**: someone must still implement the revision-3 read-out
path (0 GPU). Per `memory/agent-output-must-be-persisted-to-the-consumers-file.md`, I checked what
the consumer actually reads rather than assuming the doc and the code agree — they do not. The
driver's closing log lines say this explicitly so the next agent is not surprised.

### 5.3 `GRID_I` needs step 153500 and 175000; `GRID_W` needs 25000 and 50000 — the union is 6, but no single grid needs all 6
Consistent with the artifact's §5 table (GRID_I 4 arms / +2 for GRID_W). Worth stating because if
only 4 rungs land, **GRID_I alone is computable** (`STEPS="100000 128000 153500 175000"`) and
that is the *primary* grid. Partial progress is therefore useful, not wasted — but the combined
verdict needs all 6.

### 5.4 Checkpoint-size census confirmed, and `final.pt` is a hard link to `step200000.pt`
Re-verified all 15: 12 × 48724474298 B, 2 × 48724473567 B (step25000/50000), 1 × 48724468275 B
(step200000). Matches §4 Part D exactly. Additionally, `ls -l` shows `final.pt` and `step200000.pt`
both with **link count 2** and identical size/mtime — the same inode. Harmless, and mildly
reassuring: the archived read-out point and `final.pt` cannot disagree.

### 5.5 The artifact says "45 dirs scanned"; I independently recount 45, and confirm exactly 2 seed1234 dirs
Both are step200000: `keep14_s1234_step200000_sv181` (core6, 6 per-example files) and
`..._know` (know5, 5 per-example files). The `_know` dir is a *different task set* and cannot
substitute. `READOUT_ABSENT` for the other 6 steps is confirmed on wzc1 by my own scan.

### 5.6 Not re-verified by me: the zwfy6 side
The artifact's zwfy6 scan (165 dirs, no seed1234 reference) I did **not** re-run — it needs the
other disk and the artifact already flags the checkpoints as wzc1-resident, and the gate must run
on sm_100/wzc1 regardless. Per `memory/two-disk-rule-applies-to-main-too.md` I name this as
inherited, not independently confirmed. It does not affect the fill: the outputs land on wzc1
where the comparator lives.

### 5.7 A `_know` note, for whoever computes φ
`keep14_s1234_step200000_sv181_know` exists at step200000 only. If anyone later wants a know5
read-out they would need 6 more rungs; this driver deliberately does **core6 only**, because
`median_margin`'s `EXPECTED_POOLED = 17195` is the core6 pooled count.

---

## 6. Provenance

Every claim above is either reproduced this session on wzc1 (0 GPU) or quoted with file:line.

- `scripts/_run_paperB_keep14_seedvar_local.sh:48-49,53,57-58,63,112-127` — protocol of record.
- `scripts/eval_olmo2_probe2_downstream.py:412` (`item_id`), `:456-474` (`norm_lens`/`norm_scores`),
  `:476` (samples capped at 6), `:486-566` (merge, `n_scored`), `:605-627` (flags/defaults).
- `scripts/eval_olmo2_probe2_ppl.py:56-123` — `build_pruned_shell` / `load_pruned_model`
  (keep/fresh read from ckpt meta; fp32 master weights).
- `proposal/backlog/B04-eval-fragility-incubator/code/analyze_b04_wzc1_floor.py:84-87`
  (`TASKS`, `EXPECTED_N`, `EXPECTED_POOLED=17195`), `:120-121` (revision-2 grid, still in force in
  code), `:172-201` (margin definition + protocol asserts), `:503/:526/:531` (the only
  `phi_budget` call sites, all selftest).
- `logs/sv181_main.log:2-7,31-36` and `logs/sv181_{ppl,core}_keep14_s{42,1234}_step200000_sv181_shard{0..7}.log`
  — the cold/warm decomposition in §4.
- `status/PAPERB_WITHIN_DISK_FLOOR_V3.md:7,37-38` — same-driver runs byte-identical; the only
  non-zero within-disk comparison crosses a driver boundary.
- `olmo2_downstream_results/keep14_s1234_step200000_sv181/summary.json` + its 6
  `per_example_*.jsonl` — `n_scored`/row counts in §2; `add_bos=False`; `meta.ckpt`/`ckpt_step`.
- `outputs/olmo2_probe2_7B_keep14fresh2_seed1234/` — 15 `step*.pt` sizes re-verified;
  `arch_meta.json` confirms `seed: 1234`, keep 14 / fresh 2, `n_params 4060352512`.
- IO measurements: `dd iflag=direct` single-stream (0.97 GB/s) and 8-way concurrent (2.57 GB/s);
  page-cache eviction behaviour of a 48.7 GB file. CPU/IO only, no GPU.
- Oracle fault-injection (§2) run against a **copy** in a throwaway tmpdir, since deleted.

**GPU used: none.** Nothing was written to `olmo2_downstream_results/`. The only files created are
the driver, this document, and its own `logs/b04_evalfill_*_main.log` dry-run traces.
