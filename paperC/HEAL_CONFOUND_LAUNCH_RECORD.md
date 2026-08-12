# Launch record — healed Qwen3-8B-Base front8+fresh2 (heal-vs-no-heal confound)

Pre-registration: [`HEAL_CONFOUND_PREREGISTRATION.md`](HEAL_CONFOUND_PREREGISTRATION.md),
committed **9920de7** at 2026-08-12 **before** any GPU was allocated.
Launched **after** that commit, so the design is provably pre-hoc.

## Launch config (as it actually ran, copied from the log header)

| field | value |
|---|---|
| node | **`.104`** (`28.83.24.104`), 8×H20, disk **zwfy6** |
| verified idle first | 8×`0 MiB`, `0 %`, `nvidia-smi --query-compute-apps` returned 0 rows |
| launcher | `scripts/run_paperC_qwen3base_heal_k8f2.sh` (md5 `44b1501…`, identical on both disks) |
| trainer | `scripts/train_qwen3_arch_probe2.py` (md5 `50d5111…`, **re-synced**, see §"stale trainer") |
| python | `/opt/conda/envs/torch-base/bin/python`, torch 2.13.0, transformers 5.5.4 |
| base | `models/Qwen3-8B-Base` — guard asserted `eos_token_id == 151643` before launch |
| arch | `keep_front=8` + `n_fresh=2` = **10 layers**, **3.1741B** params |
| eff_bs | `bs2 × accum8 × 8 ranks` = **128** → 262144 tokens/step (matches OLMo-2 keep8) |
| seq_len | 2048 |
| lr | `--lr 1e-4 --min_lr 1e-5 --lr_inherited 2e-5 --min_lr_inherited 2e-6`, warmup 150 |
| cosine horizon | `max_steps 200000`; **pre-registered read-out = step 121000** |
| grad ckpt | on |
| retention | `save_every 500 --milestone_every 5000 --keep_last_n 3 --keep_milestones 8 --keep_steps 121000` |
| data | **`data/slimpajama_chunks_2048_qwen3base_full.npy`** (2705613 rows = **5.541B** tok, Base EOS 151643) — see §"restart" |
| out / log | `outputs/paperC_qwen3base_heal_k8f2/` , `logs/paperC_qwen3base_heal_k8f2.log` |
| pid | **3343471** (`setsid`, survives ssh logout) |
| started | **2026-08-12 14:18:03** (remote wall clock; a first launch at 13:48 was deliberately restarted, §"restart") |

## Verification that it is really training (not merely alive)

```
[healing_front8+fresh2]  world_size=8 bs=2 gaccum=8 eff_bs=128 seq_len=2048 max_steps=200000
[transplant] copied 91 tensors (front 8 layers + embed/norm/lm_head); fresh tail layer-ids [8, 9] left at Qwen3 init
[sanity] unexpected=0 | copied=91==91 | max|model-base|=0.000e+00 (exact) | fresh_ln_all_ones=True fresh_q_std=0.0200 -> ALL 5 CHECKS PASS
model params = 3.1741B (trainable 3.1741B) num_hidden_layers=10
dataset rows=1127824 seq_len=2048 from data/slimpajama_chunks_2048_qwen3.npy
[optim] group inh_decay: 3174.0M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_nodecay: 0.1M params base_lr=2.00e-05 min_lr=2.00e-06
```

Checks, each against a **pre-launch prediction** rather than post-hoc reading:

1. **Arch is the one claimed.** `copied=91 == 3 + 11×8` exactly; `max|model−base| = 0.000e+00`
   (bit-exact transplant); fresh layer-ids `[8, 9]` are the only missing keys;
   fresh RMSNorm all-ones and fresh `q_proj` std `0.0200` ≈ `initializer_range`.
   All 5 asserts are **hard** — a wrong architecture crashes rather than trains.
2. **Param count matches prediction.** Predicted 3.175B from
   `10 × 0.1931 + 1.244` (untied embed + lm_head); logged **3.1741B**.
3. **Param groups match the pre-registered bug-for-bug expectation.** Only
   `inh_decay 3174.0M` + `inh_nodecay 0.1M`, both at **2e-5**, and **no `fresh_*`
   group** — i.e. `--lr 1e-4` is a **no-op**, exactly as
   `olmo2_7B_keep8fresh2.log` / `keep14fresh2.log` behaved. This was predicted in
   preregistration §9.6 from `_classify_param` missing the `module.` strip while
   `build_param_groups` runs after DDP wrap. **Confirmed, and it is the desired
   match** — but the paper must not claim differential LR for either side.
4. **Memory ≈ prediction, and hits the CLAUDE.md target.** Predicted 78-80 GB from
   the `k8f2_frontier` measurement; logged **`maxmem=77.5GB`** (allocator) and
   `nvidia-smi` shows **78775 MiB / 97871 MiB = 80.5%** on all 8 cards with
   **98-100% util**. ≥80% target met.
5. **Loss is descending sanely from a fresh tail.** step20 `13.79` → 40 `10.09`
   → 60 `8.05` → 80 `7.03` → 100 `6.38` → 120 `6.03` → 140 `5.70` → 160 `5.41`
   → 180 `5.24`. Starting near `ln(151936) ≈ 11.9`+ is correct for 2 randomly
   initialised layers; gnorm 9-25 with `grad_clip 1.0`, no spikes.
6. **0 failure syntax.** `grep -aiE 'traceback|runtimeerror|cuda error|out of
   memory|assertionerror'` → **0** matches.
   ⚠️ Method note: a bare `grep -iE 'nan|inf'` returns **every `INFO` line**
   (`inf` ⊂ `INFO`), on top of the known false positive from the passing
   `✓ No NaN/Inf` line. Both must be excluded or the check is meaningless.

## Measured throughput and ETA (not extrapolated from 1-2 points)

Measured from **log timestamps** over the steady-state window, discarding the
step-20 point (it absorbs warmup/compile):

```
window: step 40 -> 180   (140 steps, 800.2 s, 8 log points)
MEASURED s/step = 5.716
per-interval s/step: min 5.713  max 5.722  mean 5.716   n_intervals = 7
```

The 7 independent intervals span **0.009 s** (0.16%), so the rate is genuinely
stable rather than an artefact of a lucky pair. The trainer's own instantaneous
`s/step` field agrees (5.71-5.72).

| milestone | ETA from measured 5.716 s/step |
|---|---|
| step 5000 (first milestone) | 7.9 h |
| **step 121000 (pre-registered read-out)** | **192.1 h = 8.00 days** |
| step 200000 (cosine horizon) | 317.5 h = 13.2 days |

⚠️ These are **pure compute** ETAs. They exclude checkpoint-write stalls (the
historical log shows an occasional 7.7-10.6 s/step at save boundaries, i.e. a few
% overhead) and any preemption. Treat 8 days as a floor, ~8.5 days as realistic.

## The 8-card vs 16-card decision, revisited against the new measurement

The pre-registration rejected 16-card DDP on the historical `armB` numbers
(7.59 s/step @ 8 ranks → 6.91 @ 16 ranks = 1.10×). This arm measures **5.716
s/step**, *faster* than that 8-rank reference, because it is 10 layers / 3.17B
rather than 14 layers / 3.95B. That makes the 16-card case **worse, not better**:
with `eff_bs` pinned at 128 for comparability, 16 ranks forces `accum 4`, halving
local work per rank while the full 3.17B fp32 gradient all-reduce over bond1 TCP
(IB disabled) shrinks only ~20%. The communication:computation ratio therefore
rises. **Decision stands, now on this arm's own measurement.**

Also unchanged: plain DDP (`find_unused_parameters=False`, no param/grad/optim
sharding) means extra ranks would **not** have reduced per-rank memory or enabled
a larger local batch — see `memory/ddp-not-fsdp-per-card-mem`.

## Stale trainer on zwfy6 — caught before launch, would have been silent

The zwfy6 copy of `train_qwen3_arch_probe2.py` was **42 diff-lines behind** wzc1's
and was missing:

- the entire `ckpt_rotation` import + the `rotate_checkpoints()` call and its
  `--keep_last_n / --keep_steps / --keep_milestones / --milestone_every` flags →
  passing those flags would have been an **argparse error**, and running without
  them would have written 242 × 38 GB ≈ **9.2 TB** onto a disk with **3.4 TB** free;
- `--seed` and `DistributedSampler(seed=args.seed)` → any `--seed` would have been
  **silently ignored** (the sampler builds its own generator seeded 0).

`train_qwen3_arch_probe2.py` and its dependency `train_semantic_bottleneck_1b.py`
were `scp -O`'d to zwfy6 and **md5-verified identical** before launch. General
lesson, consistent with `memory/cluster-two-disks-not-shared`: the zwfy6 checkout
lags (`2d98c5a`), so **hash the trainer, not just the data, before a multi-day run**.

## Concurrent CPU-only work on `.82` (0 GPU) — COMPLETED in 189 s

Re-tokenized **all 48** SlimPajama-6B train shards with the **Base** tokenizer and
the correct **Base EOS 151643**, fixing two defects of the corpus the first launch
used (preregistration §6):

- the original npy used a subset (2.31B tok → 13.7 epochs over 121k steps);
- it was built with the **Instruct** EOS **151645** as document separator — verified:
  100444 occurrences of 151645 and **0** of 151643 in a 102M-token sample.

Command: `scripts/preprocess_slimpajama.py --num_train_shards 0 --num_proc 96
--eos_token_id 151643 --tokenizer .../Qwen3-8B-Base` →
`data/slimpajama_chunks_2048_qwen3base_full.npy`,
log `logs/tokenize_slimpajama_qwen3base_full.log`. GPUs on `.82` confirmed `0 MiB`
throughout, before during and after.

**Result:** `5,489,000 docs, 5,535,604,918 content tokens, (2705613, 2048), 22.16 GB,
189 s` (+ a 4435-row val split). Both defects fixed, and the arm was **restarted onto
it** — see the next section.

## Known follow-up before the arm can be scored (not a blocker to training)

`scripts/eval_olmo2_mc_letter_content.py` imports `load_pruned_model` from
`eval_olmo2_probe2_ppl`, whose `build_pruned_shell` hardcodes
`Olmo2Config`/`Olmo2ForCausalLM`. A Qwen3 **pruned checkpoint** therefore cannot
be loaded by the MMLU-Pro harness as-is. The Qwen3 twin already exists
(`scripts/eval_qwen3_probe2_ppl.py`, same builder contract), so this is a small
family dispatch, CPU-testable against the step-500 checkpoint. Listed here so it
is not discovered on day 8.

## Restart onto the better corpus (planned in advance, executed at step 240)

Pre-registration §6 committed to this in writing *before* launch: "If it completes
before the arm reaches 121k, the run is restarted on the larger corpus". The
re-tokenization finished in **189 s** (384-core node, 96 procs) — far sooner than
expected — so the commitment was honoured at **step 240 of 200000 (0.12%)**, a
sunk cost of ~25 min.

| | first launch 13:48 | **relaunch 14:18 (current)** |
|---|---|---|
| corpus | `slimpajama_chunks_2048_qwen3.npy` | `slimpajama_chunks_2048_qwen3base_full.npy` |
| rows | 1127824 | **2705613** |
| tokens | 2.31B | **5.541B** (2.40×) |
| epochs over 121k steps | 13.7 | **5.72** |
| doc separator | **Instruct EOS 151645** (wrong for a Base arm) | **Base EOS 151643** ✅ |
| shards tokenized | subset | **all 48** |

Verified on the new corpus: `151643` count = 100444 and `151645` count = **0** in a
50k-row sample (the exact inverse of the old file), `max id 151643`.

Kill was by **PID** (`kill -9 3334823`), not `pkill -f`, per
`memory/kill-remote-gpu-job-by-pid-not-pkill`; the cmdline was inspected first to
confirm it was the training launcher and **not** an eval process, and GPUs were
confirmed released (8×`0 MiB`, 0 compute apps) before relaunching.

**This does not weaken the corpus caveat, it only shrinks it.** 5.72 epochs of
5.541B SlimPajama tokens is still not 1.0 epoch of 31.7B Dolmino tokens, so
preregistration §6 and §9.2 stand: the corpus asymmetry remains an unremovable
limitation that must be reported with any verdict.

### Re-verification after relaunch (same checks, new run)

```
[sanity] unexpected=0 | copied=91==91 | max|model-base|=0.000e+00 (exact) | fresh_ln_all_ones=True fresh_q_std=0.0200 -> ALL 5 CHECKS PASS
model params = 3.1741B (trainable 3.1741B) num_hidden_layers=10
dataset rows=2705613 seq_len=2048 from data/slimpajama_chunks_2048_qwen3base_full.npy
[optim] group inh_decay: 3174.0M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_nodecay: 0.1M params base_lr=2.00e-05 min_lr=2.00e-06
```

`dataset rows=2705613` is the positive confirmation that the new corpus is the one
in use. Loss 14.70 → 10.28 → 8.20 → 7.09 → 6.49 → 6.01 (steps 20-120);
`maxmem=77.5GB`; 0 failure matches.

**Re-measured throughput on the relaunched run** (timestamps, step-20 excluded):

```
window step 40 -> 120  (80 steps, 457.5 s, 5 log points)
MEASURED s/step = 5.718
per-interval: min 5.716  max 5.724  mean 5.718  n = 4
  step 121000: 192.2 h = 8.01 days
  step 200000: 317.7 h = 13.24 days
```

Statistically indistinguishable from the pre-restart 5.716 (Δ = 0.002 s, inside the
per-interval spread), i.e. the 2.4× larger memory-mapped corpus costs **no**
throughput — expected, since `NpyChunkDataset` mmaps and the read is sequential
per rank. **The 2.4× better corpus was free.**

Disk after relaunch: **4.1 TB free** on zwfy6 (rotation bounds the arm at ~420 GB).

## Status

`RUNNING` since **2026-08-12 14:18:03** on `.104` (pid 3343471), on the 5.541B-token
Base-EOS corpus. Read-out at step 121000, ETA **8.01 days** from measured 5.718
s/step ⇒ ≈ **2026-08-20**, and ~8.5 days allowing for checkpoint-write overhead.

Kill conditions are pre-registered in §13 of the pre-registration; **none are met** —
ppl is descending normally (2410772 → 408 over steps 20-120, from 2 randomly
initialised layers), 8×100% util, `maxmem 77.5GB`, 0 failure syntax.

### Next actions when it reaches milestones

1. **Before day 8**, land the one-line family dispatch so `load_pruned_model` can
   rebuild a **Qwen3** pruned shell (see the follow-up section above). Test it on
   the step-500 checkpoint — CPU-loadable, no GPU needed.
2. **At step 5000** (~8 h) score an early milestone to confirm the whole
   train→score pipeline works end-to-end, rather than discovering a harness
   incompatibility on day 8.
3. **At step 121000**, score with the unchanged MMLU-Pro harness at `MAXLEN=2048`
   and run the pre-registered P1/P2 contrasts from §8. `.73`/`.82` are free for the
   8-GPU sharded scoring.

⚠️ Do not re-choose the read-out step after seeing intermediate numbers. 121000 is
pre-registered because it is `olmo2_7b/keep8`'s own scored step; milestones exist to
characterise the heal *trajectory*, not to shop for a favourable cell.

---

# APPENDED 2026-08-12 by a DIFFERENT agent — follow-up #1 (family dispatch) is CLOSED

> Everything above this line is the launch agent's; I have not altered a word of
> it. This section only closes the item it listed under "Known follow-up before
> the arm can be scored" and "Next actions … 1./2.". Commit **`b2e46fb`**.

## The blocker is gone: a Qwen3 pruned ckpt is now scorable

`scripts/eval_olmo2_mc_letter_content.py` no longer calls the OLMo-2-hardcoded
`load_pruned_model` directly. It calls `_load_pruned_dispatch`, which reads
`model_family` **from the checkpoint** and routes to the family's own builder —
`eval_olmo2_probe2_ppl` for `olmo2`, `eval_qwen3_probe2_ppl` for `qwen3`. The
Qwen3 import is **lazy**, so `Qwen3Config` never enters an OLMo-2 run's import
graph. `--model_family` exists only as an override for pre-`model_family` ckpts
and is a hard error if it contradicts the ckpt.

Dispatching on the artefact rather than a CLI flag matters here: the read-out
will be driven by `scripts/_run_mmlu_pro_letter_content_8gpu.sh`, whose `spec()`
lines carry `output_name|ckpt|keep_front|n_fresh` and **no** family field. A
family flag would have had to be threaded through the driver and could be
forgotten for one arm; reading it from the ckpt cannot be.

## OLMo-2 is provably untouched

Re-scored `olmo2_7b/keep8` @ step121000, shard 0 of 8, **all six** gate-2 tasks,
through the modified file on `.82` (same disk **and** same arch as the `.73/.82/.104`
node that produced the archive on 2026-08-11 21:24), with the driver's exact
arguments. Every per-example JSONL is **bit-identical** to
`olmo2_mc_letter_content_results/7B_keep8_step121000/`:

| task | md5 (archive == re-score) | `diff` lines |
|---|---|---|
| arc_challenge | `ddbe40f0961de783c5beec6d331caf73` | 0 |
| arc_easy | `2e050e331fc70e0cf64af1a8a57567b0` | 0 |
| openbookqa | `0b67d5b31749e2804e1683e18c74be6b` | 0 |
| commonsense_qa | `4617c8b09da7bbf75eeaadfb7c9000c2` | 0 |
| piqa | `e999e510cf492d42a08ee14a959edb70` | 0 |
| winogrande | `722f9395447f8be6721bd9620b48e587` | 0 |

⚠️ **Method warning worth keeping, learnt by getting it wrong first.** My first
attempt used `--batch_size 32` and **one openbookqa item flipped**
(`content_raw` .2381 → .2540 on shard 0, exactly 1/63). The driver's default is
`BS=48`. Nothing about the loader changed between the two runs — **bf16 numerics
in this harness are batch-size dependent**, because batch composition sets the
pad width of each sorted-by-length batch. So a "bit-identical" claim is only
meaningful at the *archive's own* batch size, and any future re-score that
changes `BS` will produce small flips that are **not** evidence of a code bug.

## Qwen3 smoke on the real arm — PASSED on `step500.pt`

`step500.pt` (38 GB) landed at 15:14; verified stable (size unchanged over 2 min)
and that training had moved past it (step 520) before reading it. Scored
`mmlu_pro --limit 24` on **one** GPU of `.82`.

**Architecture asserted against `arch_meta.json`, not against my expectations** —
the task brief guessed "14 layers total"; the arm is **10**:

```
model_family=qwen3  keep_front=8  n_fresh=2  num_hidden_layers=10   (base had 36)
113 tensors, strict load, layer ids present = [0..9]
embed_tokens (151936, 4096)   lm_head (151936, 4096)   untied
has input_layernorm (Qwen3 marker)      = True
has post_feedforward_layernorm (OLMo-2) = False
```

Read-out sanity at the pre-registered `MAXLEN=2048`: **`n_trunc = 0`**,
`n_nan = 0`, log-probs all finite and all ≤ 0 (letter range −8.95 … −1.99),
**0/24 items degenerate**, 0/24 letter ties, median best-letter logprob −3.02
(p ≈ 0.049). Accuracies are meaningless at n=24 and at step 500 and are
deliberately not recorded here. `meta` carries `model_family: "qwen3"` and
`chat_template: false`.

**Pre-registration §8's "one-line family dispatch … listed as a follow-up, not a
blocker" is discharged, and it was NOT one line** — see the scope note below.

## Exact read-out command (the arm is a `MODE=olmo2`-style ckpt arm, not `crossfamily`)

The Qwen3 arm is a **pruned checkpoint**, so it must go through `--ckpt`, i.e. the
driver's `spec()` path — **not** `MODE=crossfamily`, which is for eval-time
truncation of an intact model via `--any_family` and would silently score a
front-8 *un-healed* Qwen3 instead of the healed arm. Add to `spec()`:

```
qwen3heal) echo "qwen3base_heal_k8f2_step121000|outputs/paperC_qwen3base_heal_k8f2/step121000.pt|8|2" ;;
```

and run on `.73`/`.82` with `BASE` pointed at the **Base** model:

```bash
ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PY=/opt/conda/envs/torch-base/bin/python \
BASE=../models/Qwen3-8B-Base MODE=olmo2 ARMS=qwen3heal \
MAXLEN=2048 BS=48 \
MMLU_PRO_PARQUET=/apdcephfs_zwfy6/share_304376610/pighzliu_code/data/hf_datasets/TIGER-Lab___mmlu_pro/data/test-00000-of-00001.parquet \
bash scripts/_run_mmlu_pro_letter_content_8gpu.sh
```

⚠️ `MAXLEN=2048` is **load-bearing for Qwen3 specifically**: the driver's own
table records the Qwen3 tokenizer's max encoded MMLU-Pro prompt as **1660 tok**
versus OLMo-2's 1226, and `n_trunc>0` is now a hard per-shard assert. 1536 would
truncate 2 items.

## A guard was added, and it is aimed at a defect that has already happened once

`models/Qwen3-8B-Base` and `models/Qwen--Qwen3-8b` share `model_type`,
`architectures`, `vocab_size`, `vocab.json` and `merges.txt` — there is **no
signal in any of the obvious fields**. The discriminator is `eos_token_id`
(151643 vs 151645) plus ctx (32768 vs 40960). Handing this harness the Instruct
dir now **raises before any weight is read** and writes no results file
(negative-tested on `.82`); `ALLOW_INSTRUCT_BASE=1` opts in deliberately.

This is not hypothetical: paperB shipped a table row labelled "Full base" that
was Instruct-derived (`status/ISSUES.jsonl ::
paperB-crossfamily-qwen-instruct-mislabelled-as-base`, relabelled in `424f1a8`).
Pre-registration §4 already flagged the two dirs; the guard makes the flag
executable so paperC cannot repeat it.

## Honest scope note

The pre-registration called this "a one-line family dispatch … a known, small,
CPU-testable change". The dispatch itself is small and the OLMo-2 path is
bit-preserving, but the delivered change is **+287/−4 lines**, of which only ~40
are logic: the rest are the Base/Instruct guard, a CPU dispatch selftest (tiny
real Olmo2/Qwen3 shells, exact `state_dict` round-trip, contradiction refused,
untagged-ckpt default preserved, guard fires both ways), and the rationale
comments. Removed lines: the 3-line `load_pruned_model` call and 1 comment.
Calling it "one line" in §8 was an under-estimate.

## What is NOT done

- **No P1/P2 contrast has been computed.** This closes the *plumbing*. The arm is
  at step ~520 of 121000; nothing scientific is claimed.
- The scoring numbers above are n=24 at step 500 and must not be quoted.
- The corpus caveat (§6, §9.2) is untouched by any of this.
