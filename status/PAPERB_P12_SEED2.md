# Paper B P1.2 — second training seed for keep14+fresh2 (7B)

**Status**: RUNNING since 2026-08-07 21:39 +08:00 on LOCAL 8×L20A (wzc1). ETA ~87h → ~2026-08-11 12:4x.
**Scope**: ONE new 7B seed (user decision 2026-08-07: not three — 3 seeds is ~11 days serial; one new seed
answers the "you only ran one seed" reviewer objection, the 3-seed std waits for more machines).

| field | value |
|---|---|
| run name | `olmo2_probe2_7B_keep14fresh2_seed1234` |
| output_dir | `outputs/olmo2_probe2_7B_keep14fresh2_seed1234` |
| log | `logs/olmo2_7B_keep14fresh2_seed1234.log` |
| launcher | `scripts/run_olmo2_7B_keep14_seed2.sh` |
| commit | `5db5d30` |
| node / PID | LOCAL 8×L20A (wzc1) / `4155272` |
| python | `/opt/conda/envs/torch-base/bin/python`, torch 2.13.0 |
| seed | **1234** |
| original run | `outputs/olmo2_probe2_7B_keep14fresh2`, 2026-07-16 21:36 → 07-21 01:56, final loss 2.4138 / ppl 11.18 |

---

## 1. The seed question — settled before launch

### 1.1 The original run has NO seed, and could not have had one

`outputs/olmo2_probe2_7B_keep14fresh2/arch_meta.json` has no `seed` key and the log head prints none.
Reason: **`--seed` did not exist yet.** It was added in commit `c57c4cb` (2026-08-03,
"P1.3 LR-matched init control"). The original launched 2026-07-16 against trainer `afdfa66`, and
`git show afdfa66:scripts/train_olmo2_arch_probe2.py | grep seed` returns **nothing** — that version
never called `random.seed` / `np.random.seed` / `torch.manual_seed` at all.

Consequences, all of which the paper must respect:

* **Seed 1 is an unknown, unrecorded, non-reproducible draw** from torch's default nondeterministic
  seeding (verified: three fresh `torch.initial_seed()` calls give three different 64-bit values).
  It cannot be re-run bit-exactly, ever.
* The `(seed1, seed2)` pair is **still a valid 2-sample look at init spread**, because both are
  independent draws from the same init distribution (`fresh_q_proj_std` = 0.0200 in both).
* Only **this** run is reproducible. Report seed 1 as "unseeded / not recorded", not as "seed 42".
* DDP consistency was never at risk in either run: `DDP.__init__` calls `_sync_module_states`, which
  broadcasts rank-0 parameters at wrap time, so all ranks agreed regardless of seeding.

### 1.2 Trainer default and what the seed controls

Default is `--seed 42` (`train_olmo2_arch_probe2.py:637`). `set_seed()` (line 108) seeds
python / numpy / torch CPU / all CUDA devices on every rank before model construction.

In *this* trainer the seed controls exactly **one** thing that matters:

* ✅ **random init of the 2 fresh tail layers** (layer-ids 14, 15). The 14 inherited layers are
  overwritten bit-exactly by the transplant (`transplant_max_abs_diff = 0.0`), and
  embed_tokens / model.norm / lm_head are inherited too. So the fresh tail is the only weight
  tensor the seed can move.

It does **NOT** control:

* ❌ **data order.** Line 863 is `DistributedSampler(ds, shuffle=True)` with **no `seed=` argument**.
  Torch's `DistributedSampler.__iter__` builds a *private* generator, `g.manual_seed(self.seed + self.epoch)`
  with `self.seed` defaulting to **0** — independent of `--seed` and of the global torch RNG.
  Both runs therefore consume **the same data in the same order**.
* ❌ **dropout.** `OLMo-2-1124-7B/config.json` has `attention_dropout: 0.0` and the trainer adds none
  (`grep -c dropout train_olmo2_arch_probe2.py` = 0).

### 1.3 ★ What P1.2 may and may not claim

> The two runs differ in the **random initialisation of the 2 fresh tail layers only**, under an
> **identical data order**. This is **fresh-block initialisation variance**, NOT full training-seed
> variance (init × data order × dropout).

This is a **weaker claim** than "training seed variance" and the paper must say so explicitly.
A reviewer asking "what about data-order variance?" is **not** answered by this run. Getting that
would require patching `DistributedSampler(..., seed=args.seed)`, which would then differ from the
original in a second way — deliberately not done.

Early evidence consistent with the above: the two trajectories track each other closely
(step20 11.1134 vs 10.9482, step40 8.3723 vs 8.3888, step60 7.6247 vs 7.6295, step80 6.4667 vs 6.6148),
which is what shared-data-order + different-fresh-init should look like.

### 1.4 Trainer patch: not needed

The requested "log the seed at startup and write it into arch_meta" **already exists** at HEAD:
line 690 logs `[seed] set_seed(1234) on all ranks` (confirmed in the new log) and line 845 writes
`"seed": args.seed` into `arch_meta.json` (confirmed: the new arch_meta contains `"seed": 1234`).
No trainer change was made — the ambiguity was a property of the *July* trainer, already fixed.

---

## 2. ★ The LR trap — why the launch command is NOT the original's verbatim

**The original run's differential LR was a silent no-op. It trained at a UNIFORM 2e-5.**

The original was launched with `--lr 1e-4 --lr_inherited 2e-5`, i.e. it *asked* for a differential LR.
It did not get one. At `afdfa66`, `build_param_groups` ran **after** the DDP wrap (line 593, wrap at 570),
and `_classify_param` (line 286) did **not** strip the `module.` prefix that `DDP.named_parameters()`
prepends — so every trainable param fell through to the `inherited` branch. The log proves it:

```
[optim] group inh_decay:   4060.1M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_nodecay:    0.3M params base_lr=2.00e-05 min_lr=2.00e-06
```

Two groups, 4060.1M + 0.3M = the full 4.0604B, all at 2e-5. `--lr 1e-4` moved nothing, and
`arch_meta.json`'s `"lr_fresh": 1e-4` is **aspirational, not what ran**. The `module.` strip landed
later, in `7a330ce` (2026-08-03) — i.e. this is the same bug CLAUDE.md documents for the *distill*
variant, except the fix for the non-distill trainer landed only *after* the original run finished.

**Therefore re-issuing the original command verbatim on today's trainer would have changed the
optimisation** — it would create real `fresh_*` groups at 1e-4, a **second changed variable** on top of
the seed, which is precisely the Retraction-7 failure mode (two variables moved, conclusion withdrawn).

Fix: pass `--lr 2e-5 --min_lr 2e-6`, giving every parameter the same `(base_lr, min_lr, weight_decay,
betas, eps)` it had originally. AdamW is per-parameter, so 4 groups with identical hyper-parameters is
**numerically the same optimiser** as the original's 2 groups. Verified by direct call to
`build_param_groups` on a synthetic `module.`-prefixed model:

* `--lr 2e-5 --min_lr 2e-6` → distinct `(base_lr, min_lr)` = `{(2e-05, 2e-06)}` ✅ single LR
* `--lr 1e-4 --min_lr 1e-5` → distinct = `{(1e-04, 1e-05), (2e-05, 2e-06)}` ❌ two LRs

### Optimizer groups actually observed in the new run: FOUR, all at one LR

```
[optim] group fresh_decay:   815.8M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group fresh_nodecay:   0.0M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_decay:    3244.3M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_nodecay:     0.2M params base_lr=2.00e-05 min_lr=2.00e-06
```

vs the original's TWO groups. **The group *count* differs (2 → 4) but the LR *assignment* is
identical** — the fresh group now exists and is populated (815.8M, i.e. the strip works), but sits at
the same 2e-5 the original gave it by accident. Sum 4060.3M matches the original's 4060.4M.

> **⚠️ PAPER-WRITING CONSTRAINT: do not claim differential LR for the keep14 arm (either seed).**
> Both seeds trained at a uniform 2e-5 → 2e-6 cosine. `arch_meta.json` `lr_fresh` is 1e-4 for seed 1
> (wrong, aspirational) and 2e-5 for seed 2 (correct). Cite the log's `[optim]` lines, not arch_meta.
> Same caveat CLAUDE.md already records for #99 distill — same bug family, so the arms remain
> mutually comparable, but the *claim* must be "uniform 2e-5", not "differential".

---

## 3. Explicit diff of launch command vs original

Everything below was held identical: base ckpt `models/OLMo-2-1124-7B`, `keep_front_layers 14`,
`n_fresh_layers 2`, `batch_size 16`, `grad_accumulation_steps 1`, eff_bs 128, `seq_len 2048`,
`max_steps 200000`, `gradient_checkpointing 1`, warmup 150, wd 0.1, grad_clip 1.0, fp32 master
weights + bf16 autocast, `lr_inherited 2e-5`, torch AdamW betas (0.9, 0.95) eps 1e-8,
`--data_path /dev/shm/dolmino_now15b.npy`, 8×GPU single-node `torchrun --standalone`.

**Intended difference (1):**

```
+ --seed 1234        # original: flag did not exist -> unseeded nondeterministic draw
```

**Equivalence-preserving rewrite (2) — changes the string, not the optimisation** (see §2):

```
- --lr 1e-4                    # was a NO-OP: became 2e-5 for all params
+ --lr 2e-5 --min_lr 2e-6      # makes that same 2e-5 explicit on the fixed trainer
```

**Book-keeping-only differences (3) — cannot touch trained weights:**

```
+ --keep_steps 25000,50000,100000,128000,153500,200000   # pins P1.2's pre-registered eval grid
+ --keep_milestones 8 --keep_last_n 3                    # retention; original ran unbounded (~1.8TB/arm), wzc1 at 90%
  --output_dir ..._seed1234                              # separate dir; original's paper ckpts untouched
```

### Drift audit — everything that changed since July, and whether it matters

| item | original (2026-07) | now (2026-08-07) | affects weights? |
|---|---|---|---|
| dataset | `/dev/shm/dolmino_now15b.npy` | **same file, md5 `e4da8db79c264da70f5b5be5a26f342d`** — verified identical on LOCAL **and** .252, 62,020,903,040 B, 7,570,911 rows | **NO — byte-identical** |
| trainer | `afdfa66` | `5db5d30` (+321/−41 lines) | **NO for this arm** — all additions are new opt-in flags (`--random_trunk`, `--save_step0_and_exit`, `--optimizer bnb`, `--seed`) + `n_fresh_layers==0` guard + rotation refactor. The only behavioural change on this code path is the `module.` strip, neutralised by §2. |
| `get_lr` / dataset class | `train_semantic_bottleneck_1b.py` | **unchanged** (its only diff is added ckpt-rotation plumbing) | **NO** |
| python | `.venv` torch 2.13 | `/opt/conda/envs/torch-base` **torch 2.13.0** — ⚠️ **LOCAL `.venv` no longer has torch at all** (`ModuleNotFoundError`, matches CLAUDE.md's warning) | **same torch minor**; env path differs |
| node | LOCAL 8×L20A | LOCAL 8×L20A (same machine, 183359 MiB/card) | **NO** |
| ckpt retention | unbounded milestones | bounded + `keep_steps` | **NO — disk only** |

**Runtime match confirms no hidden drift**: **1.56 s/step** and **maxmem 122.3 GB** are *identical*
to the original's logged values, and `[sanity] ... ALL 6 CHECKS PASS` with `copied=157==157`,
`max|model-base| = 0.000e+00`, `fresh_q_std = 0.0200`.

---

## 4. Node selection

`.73` / `.82` / `.104` were **ruled out without trying**: maxmem 122.3 GB > H20's 97.8 GB, and this
trainer is plain DDP (no param/grad/optim sharding) so adding cards does not reduce per-card memory.
Both wzc1 nodes (LOCAL, `.252`) were occupied by someone else's `dllm_draft` EvalPlus HumanEval jobs
at first check; I **polled rather than killing anything**, and LOCAL freed at 21:38 (0 MiB × 8, no
orphan processes). `.252` was still busy (~63 GB total) at launch time and was left alone.

Launched with `setsid nohup` so it survives shell exit — required for a ~87h run.

---

## 5. Result table to fill at completion (do NOT edit paperB/TODOList.md — MAIN owns it)

Evaluate at the pre-registered grid (0/25k/50k/100k/128k/153.5k/200k), base protocol
(`chat_template=False`, no BOS), matching the seed-1 numbers already in the ledger.

| Arm | Seed | Final NLL/PPL | Final MMLU | Checkpoint/raw path |
|---|---:|---|---:|---|
| keep14 | *unseeded (not recorded)* | train-loss 2.4138 / ppl 11.18 @200k | 31.91% (clean 31.82%) | `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt` |
| keep14 | 1234 | TBD | TBD | `outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step200000.pt` |

* n=2 → report **range / absolute difference**, not a standard deviation. An SD from two points is
  not meaningful and a reviewer will say so.
* Frame as **fresh-block init variance at fixed data order** (§1.3).
* The load-bearing quantity is the **keep14 − random-init MMLU gap** (+7.11pp, McNemar p=1.64e-46,
  CI [+6.14, +8.09]pp from P1.2/P2.8). Seed 2 shows whether that gap survives a different init draw;
  report the gap under each seed side by side.
