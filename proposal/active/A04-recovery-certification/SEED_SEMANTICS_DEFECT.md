# Defect: `DistributedSampler` seed omission — "seed variance" in this repo is **init variance**

**Status:** FIXED in code (future runs only). All pre-fix runs are affected and their claims
must be re-scoped. **No running job was restarted or disturbed.**

**Date:** 2026-08-09
**Scope of this note:** records the defect + which existing runs are affected + what they may
therefore claim. It deliberately does **not** edit `paperB*`, `paperA*`, `*TODOList*`, or
`status/*.md`; the implication is recorded here for whoever owns those files.

---

## 1. The defect

`scripts/train_olmo2_arch_probe2.py:863` (pre-fix) read:

```python
sampler = DistributedSampler(ds, shuffle=True)     # no seed=
```

`args.seed` was plumbed into `set_seed(args.seed)` at line 688 (which calls `random.seed`,
`np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`) and was recorded into
`arch_meta.json`, so the run **looked** seeded end-to-end. It was not: the data order was
not seeded at all.

## 2. Why `torch.manual_seed` cannot reach it (verified against torch source on disk)

Source read directly:
`/opt/conda/envs/torch-base/lib/python3.14/site-packages/torch/utils/data/distributed.py`
(torch `__version__ = '2.13.0'`, from `torch/version.py`):

| line | content |
|---|---|
| 72 | `seed: int = 0,`  ← constructor default |
| 105 | `self.seed = seed` |
| 107 | `def __iter__(self) -> Iterator[_T_co]:` |
| 109-111 | `g = torch.Generator()` then `g.manual_seed(self.seed + self.epoch)` |
| 112 | `indices = torch.randperm(len(self.dataset), generator=g).tolist()` |

The permutation is drawn from a **freshly constructed `torch.Generator`** explicitly seeded
from `self.seed + self.epoch`. It never consults the global/default RNG. Therefore
`torch.manual_seed(args.seed)` is **causally disconnected** from the shuffle, and every run
used `self.seed == 0`.

`set_epoch()` (line 146-157) only sets `self.epoch`, so the epoch-2+ reshuffles were
`manual_seed(0 + epoch)` — also seed-independent.

## 3. No other data-path stochasticity to rescue it

- **Dropout: none.** `grep -in dropout scripts/train_olmo2_arch_probe2.py` → **exit code 1,
  zero matches.** No dropout anywhere in the trainer.
- **Dataset: deterministic.** `NpyChunkDataset` (defined `scripts/train_semantic_bottleneck_1b.py:51`,
  imported at `train_olmo2_arch_probe2.py:89`) is `np.load(path, mmap_mode="r")` +
  `__getitem__` = a pure slice/cast of row `idx`. No sampling, no augmentation, no RNG.
- `collate_fn` is a pure `torch.stack`. `num_workers=4` with a `DistributedSampler` does not
  reorder indices; workers only fetch the sampler's already-fixed index sequence.

## 4. Consequence (this is the load-bearing statement)

> **Every "seed" arm this trainer family has ever produced differs ONLY in
> fresh-block initialisation. The data order is byte-identical across seeds.**

`args.seed` reached exactly one thing that matters: the RNG state at the moment fresh
(non-transplanted) layers were randomly initialised. Everything else — which documents,
in which order, in which batch, on which rank, at which step — was identical.

### 4.1 Corollary: an arm with `--n_fresh_layers 0` had `sd_run` identically 0

If there are no fresh layers, there is nothing for the seed to initialise. Combined with
§2 (data order seed-independent) and §3 (no dropout, deterministic dataset), a
`--n_fresh_layers 0` arm is **bit-deterministic in `args.seed`**: re-running it with a
different `--seed` reproduces the same weights. Its measured run-to-run standard deviation
`sd_run` is therefore **identically 0 by construction**, not empirically small. Any
variance estimate borrowed from such an arm is a *tautology*, not a measurement.

Runs on disk with `n_fresh_layers == 0` (from their own `arch_meta.json`):

| disk | run |
|---|---|
| wzc1 | `outputs/olmo2_p05_armA_contig16` |
| wzc1 | `outputs/olmo2_probe2_7B_full32_dolmino` |
| wzc1 | `outputs/olmo2_probe2_7B_shortgpt16` |
| wzc1 | `outputs/paperC_depthsweep_keep32_graft_refusal25` |
| wzc1 | `outputs/paperC_pc1_squad_A1_full32ft` |
| zwfy6 | `outputs/olmo2_p05_armA_contig16` |
| zwfy6 | `outputs/paperC_pc1_squad_A1` |

(`paperC_pc1_squad_A2_lora_r160` has `n_fresh_layers: null` — LoRA, different code path,
not assessed here.)

## 5. Which existing runs are affected, and what they may claim

I enumerated `outputs/*/arch_meta.json` on **both** disks (wzc1 root
`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory`, zwfy6 root
`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory`, read over ssh on `.73`).

- wzc1: 81 `arch_meta.json`; distinct `seed` values = `{42, 1234, null}`.
- zwfy6: 43 `arch_meta.json`; distinct `seed` values = `{42, null}` — **no non-42 seed at all.**
- **The only run anywhere with `seed != 42`:** `outputs/olmo2_probe2_7B_keep14fresh2_seed1234`
  (wzc1, LOCAL; `arch_meta.json`: `"seed": 1234`, `keep_front_layers: 14`,
  `n_fresh_layers: 2`, `from_scratch: false`).

### 5.1 The live LOCAL run is mislabelled

`outputs/olmo2_probe2_7B_keep14fresh2_seed1234` is labelled **"seed-variance evidence"**.
Under the pre-fix code it is **init-variance evidence**, plainly: it shares a byte-identical
data order with its `seed=42` sibling `outputs/olmo2_probe2_7B_keep14fresh2`, and the only
difference between them is the random initialisation of the 2 fresh layers (ids 14, 15).

**It may claim:** "sensitivity of the healed model to fresh-block initialisation, holding
data order fixed."
**It may NOT claim:** "seed variance", "run-to-run variance", "training-stochasticity
variance", or anything that a reader would take to include data-order/curriculum
stochasticity. Those components were **never sampled** and their contribution to variance
is **unmeasured** (not "small" — unmeasured).

This matters because init variance is a strict **subset** of run-to-run variance, so any
noise floor / significance threshold derived from this pair is an **underestimate** of the
true run-to-run floor. Claims of the form "effect X exceeds seed noise" become weaker, not
stronger, once this is corrected — the comparison was against a partial noise term.

The run was **not** disturbed. It was live (42 matching processes) while this note was
written and remains so.

### 5.2 Everything else

Every other run used `seed=42` or left it at the default. For those the defect is
**latent, not manifest**: with a single seed there is no cross-seed claim to invalidate.
They are internally consistent and their reported numbers stand. The defect only ever bites
the moment someone varies `--seed` and interprets the spread.

## 6. The fix

`seed=args.seed` now passed at **all 23** `DistributedSampler(..., shuffle=True)` sites in
git-tracked `scripts/*.py` (**20 were missing it and were fixed**; 3 —
`train_olmo2_sft.py:332`, `train_mem_space_babilong.py`, `train_mem_space_distill.py` —
already had it, which is what made the omission elsewhere detectable as an oversight rather
than a design choice). The 9 `shuffle=False` samplers are unaffected and were left alone
(with `shuffle=False`, `__iter__` takes the `list(range(...))` branch at line 114 and never
touches the generator).

13 of the 20 fixed files had **no `--seed` argparse entry at all**; `--seed`
(`type=int, default=42`) was added to each so `args.seed` resolves. Verified by running
`--help` on a sample.

**Not fixed, deliberately:** `scripts/train_rmt_slot.py` and `scripts/train_slot_memory.py`
exist on disk but are **untracked leftovers** of commit `b63b5a1`
("archive dead src/ subsystems ... to legacy/"); the tracked copies are
`legacy/src_dead_subsystems/drivers/`. They were reverted to byte-identical with their
`legacy/` counterparts so no stray diff is introduced. A separate audit of `legacy/**/*.py`
found **13** further instances of the same defect (`legacy/launchers/train_swa_memory.py`,
`legacy/scripts/train_rmt_{original,pg19,v3..v10}.py`,
`legacy/scripts/train_sparse_memory.py`, plus the two drivers above). These are archived
dead code for abandoned directions and are **out of scope**; they are recorded here only so
nobody resurrects one and inherits the defect silently.

Each patched site carries a comment stating the argument is load-bearing, because
`torch.manual_seed` *looks* like it should already cover it and a future reader would
otherwise delete it as redundant. That is precisely how this bug survived.

**The fix takes effect for future runs only.** No checkpoint, log, or number already on disk
changes. No running job was restarted.

## 7. Verification performed

- `ast`-based audit of git-tracked `scripts/*.py`: 32 `DistributedSampler` sites total → 23
  `shuffle=True` (23/23 now have `seed=`, **0 remaining**) + 9 `shuffle=False` (untouched).
- `ast.parse()` clean on all 20 edited files (torch is not importable on LOCAL, so a syntax
  check is the available static verification; the trainers were **not** executed).
- `--help` executed on 5 representative files under `/opt/conda/envs/torch-base/bin/python`
  to confirm `--seed` is a live argparse option.
- torch source quoted in §2 read from disk, not from memory or documentation.

## 8. What a real seed-variance arm now requires

Post-fix, two runs differing only in `--seed` will differ in **both** fresh init **and**
data order — i.e. genuine run-to-run variance. Anyone needing a seed-variance estimate
(e.g. a K2-style gate) must launch **new** runs under the fixed code; the existing
`seed=42` / `seed=1234` pair cannot be retrofitted into one, because their shared data
order is a property of the checkpoints already on disk.
