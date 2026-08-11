---
scope: prep for the token-matched SparseForge re-run (arms 2 and 3) requested by the user and by reviewers
date: 2026-08-11
status: PREP COMPLETE — launchable, NOT launched. Contains one finding that outranks the re-run itself.
node intended: `.21` (8x L20A, wzc1). Nothing was run on GPU for this document.
---

# SparseForge token-matched re-run — preparation

**Target design** (resolved with the user):

| # | arm | steps x gbs x seqlen | tokens | SLoRB | status |
|---|---|---|---|---|---|
| 1 | CAST-repro | 7500 x 256 x 4096 | 7,864,320,000 | no | **DONE** — ppl@4096 6.1372, CAST-7 58.39 plain acc |
| 2 | SparseForge + SLoRB | 7500 x 256 x 4096 | 7,864,320,000 | k=16 | launchable |
| 3 | SparseForge − SLoRB | 7500 x 256 x 4096 | 7,864,320,000 | **retrained without** | launchable |

Arm 3 is a **fresh training run**, not a post-hoc amputation. Deliverables:
`main_llama.py` (data-loader fix), `scripts/_run_sparseforge_tokenmatched.sh` (both arms), this file.

---

## ★★ FINDING THAT OUTRANKS THE RE-RUN — `mask_metric=hessian_obd` never had a Hessian

**In the published 5B checkpoint, `hessian_diag` is exactly 1.0 at every one of 6,476,005,376
positions after 17,900 iterations.** It is warm-started to 1.0 and then never updated, because the
update function returns early on an unrelated guard. Consequence: the OBD score

    scores = (H + eps) * W^2   with H == 1 uniformly   =   (1 + 1e-8) * W^2

is a positive scalar multiple of `W^2`, so it induces **exactly the same ranking as |W|**.
**`hessian_obd` is magnitude pruning.** The "Hessian-aware" mask criterion — the thing the method is
named for — is inert, in the shipped checkpoint, not just in theory.

### The mechanism, at file:line

`sparse_modeling.py:300` — `grad_ema` is only allocated full-size for the *movement* metric:
```python
need_grad_ema = cfg.change_mask and (cfg.mask_metric == "movement")
```
With `mask_metric=hessian_obd` this is False, so `grad_ema` becomes a 1-element placeholder
(`sparse_modeling.py:309-313`, sets `_grad_ema_placeholder = True`).

`sparse_modeling.py:937` — the *only* writer of `hessian_diag` in the non-Hutchinson path tests that
placeholder flag first and returns:
```python
def update_grad_hessian_ema(self, update_hessian_with_grad2: bool = True):
    if self._is_placeholder('grad_ema') or self.weight.grad is None:
        return                                    # <-- always taken for hessian_obd
    ...
    if update_hessian_with_grad2 and not self._is_placeholder('hessian_diag'):
        self.hessian_diag.mul_(b).add_(g * g, alpha=1 - b)   # <-- never reached
```
`hessian_diag` is a full-size Parameter (`sparse_modeling.py:319-327`, since
`need_hessian` is True for `hessian_obd`), and `sparse_modeling.py:449` warm-starts it:
```python
if self.mask_metric == "hessian_obd":
    # Warm start: H≈1 to avoid score collapse from zero Hessian
    self.hessian_diag.fill_(1.0)
```
So it is allocated, filled with 1.0, carried through training and saved — never updated.
The guard is a plain conjunction bug: the `grad_ema` placeholder state has nothing to do with
whether the g² EMA for `hessian_diag` should run.

### Measured, on the published checkpoint itself

```
$ python3 -c "... torch.load('out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/model_best_lm_eval.pt', mmap=True) ..."
iter_num 17900 finalization_done True
  hessian_diag         ->  224 tensors first shape (4096, 4096)
  grad_ema             ->  224 tensors first shape (1,)          <-- placeholder, confirms the branch
  ...
l.layers.0.self_attn.q_proj.hessian_diag  unique=[1.0] n_unique=1 all_ones=True
l.layers.0.self_attn.k_proj.hessian_diag  unique=[1.0] n_unique=1 all_ones=True
l.layers.0.self_attn.v_proj.hessian_diag  unique=[1.0] n_unique=1 all_ones=True
```
and reproduced from a clean module on CPU (5 real backward+update steps):
```
grad_ema placeholder     : True   numel 1
hessian_diag placeholder : False  numel 4096
weight.grad is not None  : True
hessian_diag CHANGED after update_grad_hessian_ema ? False
hessian_diag after 5 steps: sum = 0.0  max = 0.0
ranking identical to |W| ? True
```
(The clean-module run shows 0.0 because `initialize()` was not called; the shipped run shows 1.0
because it was. Either way `H` is a **constant**, and a constant `H` makes OBD ≡ magnitude.)

### Why this outranks the re-run

The re-run's purpose is "does SparseForge's mask machinery beat CAST's at matched data and budget?"
If the mask metric is magnitude, then the honest description of arm 2/3's treatment is
**magnitude-ranked 2:4 with a soft-mask anneal, a mid-penalty, a sparsity penalty and (arm 2) a
dense low-rank branch** — not Hessian-aware pruning. That changes what the paper can claim
regardless of what number comes out.

**Decision I did NOT take:** I have not "fixed" this. Fixing it would make arms 2/3 a *different
method* from the published 5B checkpoint, and the user asked for the published configuration at
matched tokens. Options, in preference order:

1. **Run arms 2/3 as-is** (current script) and report the metric as magnitude-equivalent. Preserves
   comparability to the published checkpoint. Cheapest, most honest.
2. Add a **third arm** with the guard repaired (`hessian_obd` actually accumulating g²) — this is the
   first time the claimed contribution would ever have been tested. +1.28 days.
3. Repair the guard inside arms 2/3. **Do not do this silently** — it would make the re-run
   incomparable to the published result while still being labelled "SparseForge".

I recommend 1 now, 2 as a follow-up, because 2 is the experiment that decides whether the paper's
central claim is real.

---

## TASK 1 — the data-corruption bug: what was actually true, the fix, and the proof

### ⚠️ Three corrections to the brief's premises

**(i) `main.py` is NOT the SparseForge trainer.** `/apdcephfs_wzc1/share_304376610/pighzliu_code/main.py`
(76 flags, mtime 2026-01-28) hardcodes `dataset='c4_dataset'`, `block_size=1024`, and its
`--student_model` only accepts `{gpt2, gpt2-medium, gpt2-large, gpt2-xl}` — it **cannot** have trained
Llama-2-7B. It is AST-lineage (near-identical in structure to `baselines/ast_official_clean/main.py`).

The real trainer is **`main_llama.py`** (108 flags). Proof — matching the published run's own
`args.json` (100 keys) against each candidate's argparse surface:
```
args.json n_keys = 100
main.py         : 76 flags, MISSING 24 incl. ['dataset','out_dir','resume','glu_joint_mask',
                                             'beta_structural_start','hard_mask_type'...]
main_llama.py   : 108 flags, MISSING []          <-- superset, 8 newer flags
main_universal.py: 143 flags, MISSING []         <-- also a superset, but 35 channel-pruning
                                                     flags absent from args.json => not it
```
`main_llama.py` is also what `train_llama.sh` launches. **The dtype bug as described does not exist
in the real trainer** — `main_llama.py:455-479` (pre-patch) already read dtype from `dtype.txt` then
`metadata.json`. Had I patched `main.py` as instructed, I would have shipped a fix to a file the
experiment never executes.

**(ii) `val.bin` DOES exist** for dolmino-llama2:
```
train.bin 310886663436 bytes  mtime=2026-08-09 03:11:21
val.bin       86663436 bytes  mtime=2026-08-09 03:11:53     <-- 21,665,859 uint32 tokens
```
Defect #3 in the brief is wrong. I still made val optional, because it costs nothing and the failure
mode (crash at line 478 on a corpus without one) is real for other datasets.

**(iii) The path problem was real.** `data_dir = os.path.join('data', args.dataset)` is relative to
CWD, and the corpus is at `Mixture-of-Memory/data/dolmino-mix-1124-llama2` while the trainer runs from
`pighzliu_code/`. There is **no** `pighzliu_code/data/dolmino-mix-1124-llama2` and no symlink:
```
--dataset 'dolmino-mix-1124-llama2'    -> pighzliu_code/data/dolmino-mix-1124-llama2   exists=False
--dataset '../Mixture-of-Memory/data/dolmino-mix-1124-llama2'                          exists=True
```
Only the fragile `../` form worked. Fixed with an explicit `--data_root`.

### The bug that IS real and unguarded

dolmino-llama2 is **uint32**; `qa_format_sft_llama` (what the published run used) is **uint16**. The
hazard is not that the trainer lacks detection — it is that **every existing guard is blind to it**:

- `_load_bin`'s size check cannot fire: a uint32 file always has an even byte count
  (`310,886,663,436 % 2 == 0`), so "not a multiple of itemsize" never triggers in the dangerous direction.
- `get_batch`'s `VOCAB_SIZE_CHECK` cannot fire: reading uint32 as uint16 yields `(id, 0)` pairs and
  **every value stays < 32000**. Measured: **0 out-of-vocab in 500 tokens**.
- The old `_load_bin` fallback made it *worse* — on a genuinely truncated file it silently
  reinterpreted as uint16 instead of reporting.

So a wrong dtype produces a run that trains on `<tok> <unk> <tok> <unk> …`, completes, and reports a
plausible loss curve.

### The fix (`main_llama.py`, minimal and backward-compatible)

Four new flags, all defaulting to previous behaviour:

| flag | default | effect |
|---|---|---|
| `--data_root` | `data` | byte-identical to the old hardcoded `os.path.join('data', dataset)`. Absolute `--dataset` bypasses it. |
| `--data_dtype` | `auto` | `auto` = old precedence (dtype.txt → metadata.json → uint16). Explicit `uint16`/`uint32` **hard-fails** if it contradicts on-disk provenance. |
| `--require_val` | `True` | old behaviour (val.bin mandatory). `False` → held-out tail of train.bin. |
| `--val_holdout_tokens` | 20,000,000 | size of that tail. |

Plus three hard failures replacing silent guesses: missing dataset dir; byte count not a whole
number of tokens (was: fall back to uint16); and a **token-count cross-check** against
`metadata.json`'s `total_tokens` that raises when the ratio is 2.0 or 0.5 — i.e. exactly the
uint16/uint32 trap.

**Val choice**: held-out **tail** of `train.bin`, and the log says loudly that it is *not* disjoint
from the training sampler (`get_batch('train')` draws uniformly over the whole memmap), so `val_loss`
is a monitor, not a generalisation estimate. Floor is `block_size + 2`, because `get_batch` does
`randint(len(data) - block_size - 1)` then reads `data[i+1 : i+1+block_size]`. (I got this off by one
on the first attempt; the test below caught it.)

Flag surface: **108 → 112**, and the published `args.json` still parses in full
(`published args.json still fully accepted: True`).

### Proof (a) — correct dtype decodes as natural English

Offset 1e9 tokens, N=500, tokenizer `models/Llama--Llama2-7b`:
```
vocab_size 32000
id min/max: 1 31519 | n>=32000: 0 | n==0: 0
first 40 ids: [997, 29877, 4086, 29889, 3575, 3414, 338, 1423, 565, 278, 997, 29877, 10541, ...]
------------------------------------------------------------------------------
Lao language. Your task is check if the Lao sentence is translation of Japanese. if the translation
is correct than generate label "Yes", otherwise generate label "No".
Teacher: Now, understand the problem? Solve this instance: Japanese: 専門家はこの購入を、イスラエルが
核攻撃を受けた場合に報復することができるという、イランへのはっきりとした信号だと見ている。
 Lao: ຜູ້ຊ່ຽວຊານໄດ້ຕັ້ງຂໍ້ສັງເກດວ່າການຊື້ໃນຄັ້ງນີ້ແມ່ນສັນຍານອັນຊັດເຈນຕໍ່ອິຣານວ່າອິດສະລະເອວສາມາດຕອບໂຕ້ຄືນຖ້າຖືກໂຈມຕີດ້ວຍນິວເຄຍ.
Student: Yes</s><s> You
```
Coherent instruction-following prose with correct `</s><s>` document boundaries — this is dolmino's
FLAN component, consistent with the mix (dclm/flan/math/pes2o/stackexchange/wiki).

### Proof (b) — negative control: the SAME bytes read as uint16

Same byte offset (uint32 index `OFF` == uint16 index `2*OFF`):
```
id min/max: 0 31519
n>=32000 (OUT OF VOCAB): 0/500 = 0.0%      <-- why VOCAB_SIZE_CHECK cannot catch this
n==0     (INJECTED ZEROS): 250/500 = 50.0%
first 40 ids: [997, 0, 29877, 0, 4086, 0, 29889, 0, 3575, 0, 3414, 0, 338, 0, 1423, 0, ...]
------------------------------------------------------------------------------
'La<unk>o<unk> language<unk>.<unk> Your<unk> task<unk> is<unk> check<unk> if<unk> the<unk> La<unk>o
<unk> sentence<unk> is<unk> translation<unk> of<unk> Japanese<unk>. ...'
```
**Exactly 50.0% zeros, 0.0% out-of-vocab.** The corruption is total and completely invisible to every
existing guard. This is the negative control the brief asked for, and it is the reason the fix raises
instead of warning.

### Proof (c) — token count matches metadata

```
train.bin bytes      : 310,886,663,436
metadata dtype       : uint32 -> itemsize 4
bytes/4              : 77,721,665,859
metadata total_tokens: 77,721,665,859
MATCH bytes/4 == total_tokens ? True
bytes%4 ==0 ? True   bytes%2==0 ? True        <-- both, so the size check is useless here
len as uint32: 77,721,665,859
len as uint16: 155,443,331,718  (= 2x, the silent doubling)
```
Also `sum(shard.num_tokens) == 77,721,665,859` over all 778 shards, and `num_shards` 778 == files on
disk. And through the patched loader:
```
[DATA] token dtype = uint32  (source: .../dolmino-mix-1124-llama2/metadata.json)
[DATA] token-count check OK: 77,721,665,859 == metadata total_tokens
```

### Proof (d) — block_size 4096 slicing is in range

Mimicking `main_llama.py:594-596` exactly, `batch_size=8`:
```
x.shape (8, 4096) y.shape (8, 4096) dtype torch.int64
x min 1 x max 31802 | ALL < 32000 ? True
y min 1 y max 31802 | ALL < 32000 ? True
zeros in x: 0 / 32768
shift-by-1 consistency y[:, :-1] == x[:, 1:] ? True
--- val.bin ---
val exists: True bytes 86,663,436 bytes%4 0 -> 21,665,859 tokens
val slice min/max: 1 31752 ALL<32000? True ; val >= block_size+1 ? True
```

### Fix regression tests (all CPU, no GPU)

| case | result |
|---|---|
| 1 dolmino uint32, absolute `--dataset`, `auto` | uint32, 77,721,665,859 tokens, count check OK |
| 2 **backward compat** `qa_format_sft_llama`, default `data_root` | uint16 from `dtype.txt`, **129,752,281 tokens** — matches the confound doc exactly |
| 3 force `--data_dtype uint16` on the uint32 corpus | **refused**: "contradicts metadata.json which says uint32" |
| 4 missing dataset dir | refused, prints cwd + both flags |
| 5 no `val.bin`, `--require_val True` | refused, names the escape hatch |
| 6 no `val.bin`, `--require_val False` | tail holdout, verified it IS the tail, `len-bs-1 >= 1` |
| 7 3-byte truncated `train.bin` | **refused** (old code would have silently used uint16) |

Case 2 is the one that matters for provenance: **the 5B run's data path is unchanged.**

---

## TASK 2 — launch commands

`scripts/_run_sparseforge_tokenmatched.sh`, `ARM=slorb` / `ARM=noslorb`.

```bash
# arm 2
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
ARM=slorb   setsid nohup bash scripts/_run_sparseforge_tokenmatched.sh \
    > logs/sf_tm_slorb_sched.out 2>&1 &

# arm 3 (AFTER arm 2 finishes — see Task 4, they cannot share a node)
ARM=noslorb setsid nohup bash scripts/_run_sparseforge_tokenmatched.sh \
    > logs/sf_tm_noslorb_sched.out 2>&1 &
```

Validation already performed (no GPU):
```
bash -n                        : syntax OK
ARM unset                      : rc=2, refuses
preflight, both arms           : rc=0, PREFLIGHT_ALL_PASSED (asserts corpus dtype==uint32)
35 flags used                  : UNKNOWN to main_llama.py -> none
argparse dry-run, both arms    : PARSE OK, 162 argv tokens
  TOKENS = 7500 x 256 x 4096   = 7,864,320,000  -> MATCH target
  grad_accum total 32, per-rank 4, divisible by 8 -> passes main_llama.py:1616 assert
  hardening ends at 7500       = max_iters      -> LANDS EXACTLY
  only difference slorb/noslorb: SLoRB True/False and out_dir
```

### Judgement calls, each justified

**Budget — `max_iters 7500`, `final_finetune_iters 0`.**
7500 x 256 x 4096 = 7,864,320,000, identical to
`outputs/cast_repro_zero2/run_manifest.json` `"total_tokens": 7864320000`.
`final_finetune_iters 0` is deliberate: the published 3000-iter final stage is **exactly** where the
MC-QA contamination entered (link 3 switched `dataset` to `qa_format_sft_llama` for it). Keeping it
would either re-import the contamination or add 3000 uncounted iters (+3.1 B tokens) CAST never got.
Verified `main_llama.py:3436` — with `extra == 0` the trainer finalizes the mask, saves, and breaks;
the `extra > 0` branch that rebuilds the optimizer is skipped.

**Mask hardening — RESCALED. This is the single most likely way this run silently produces garbage.**
Published `mask_hardening_start 12000` + `duration 5000` = 17000 = max_iters exactly. Copy those onto
7500 and hardening *starts* 4500 iters after training ends:
```
hardening_x with (12000,5000) at steps 0 / 3750 / 7000 / 7500  =  1.0 / 1.0 / 1.0 / 1.0
```
`hardening_x` stays 1.0 forever, and `sparse_modeling.py:787` then takes the
`effective_mask = self.mask` early-out on **every** forward — the model never sees a 2:4 projection.
The run would finish, look healthy, and measure nothing. Rescaled by 7500/17000 = 0.441176,
preserving "hardening completes exactly at max_iters":
```
mask_hardening_start    12000 -> 5294     (soft-only 70.6% of run, as published)
mask_hardening_duration  5000 -> 2206     (anneal    29.4% of run, as published)
5294 + 2206 = 7500  (delta 0);  hardening_x = 1.0@5293, 0.5@6397, 0.0@7500
```
`hardening_period 0` (as published) disables the independent `harden_fraction()` path at
`main_llama.py:3044`, so this schedule is the *only* hardening mechanism. Do not set it > 0.

**LR schedule — proportional rescale, stated as a judgement call.**
Published: lr 1e-4, min_lr 1e-5, warmup 2000, lr_decay_iters 15000 on a 17000 horizon. Note
`lr_decay_iters < max_iters`, so the published run spent its last 2000 iters pinned at `min_lr` —
which is exactly the window where hardening finished. That coupling is plausibly load-bearing (it
stops pruned weights being kicked back to O(lr) after binarisation), so I preserved it:
```
warmup_iters    2000 -> 882      (11.8% of run, as published)
lr_decay_iters 15000 -> 6618     (88.24% of max_iters, as published; last 882 iters at min_lr)
lr / min_lr      unchanged (1e-4 / 1e-5) — magnitudes, not schedule positions
```
Rejected alternative: keep warmup 2000 absolute → 26.7% of a 7500-iter run in warmup vs 11.8%
originally, handicapping the arm we are trying to treat fairly, and decoupling warmup from the
mask-update switch at 2000.

**Defending it to a reviewer:** the honest statement is *"the schedule was rescaled proportionally to
the compressed horizon; no LR search was run at 7500 iters for either arm."* Note the comparison is
**not LR-matched in absolute terms** — CAST-repro used lr 2e-5 / min 2e-6 / warmup 375, a 5x smaller
peak LR — because each method keeps its own published optimiser hyperparameters. That is the standard
choice, and it is precisely why the claim must be about the mask mechanism at matched data+budget,
not "SparseForge is better tuned". A reviewer demanding LR-matching is asking for a third pair of
runs, not a tweak to these.

Also rescaled by the same factor (leaving them absolute would misplace them just as badly, only less
visibly): `mask_update_switch_step` 2000→882, `sparsity_warmup_steps` 500→221,
`beta_structural_start` 2000→882, `beta_structural_end` 17000→7500, `increase_step` 10000→4412.

**Arm 3 — `SLoRB False` removes the branch cleanly.** Verified on CPU, real `SparseLinear`:
```
--- SLoRB=True
   params: ['weight','mask','grad_ema','hessian_diag','frozen_mask_flags','scaler_row','x_proj','SLoRB_Weight']
   forward OK, out (2, 8, 4096) ; SLoRB extra params this layer: 2,097,152
--- SLoRB=False
   params: ['weight','mask','grad_ema','hessian_diag','frozen_mask_flags','scaler_row']
   has x_proj attr: False | has SLoRB_Weight: False
   forward OK, out (2, 8, 4096) ; SLoRB extra params this layer: 0
```
`sparse_modeling.py:408-427` registers `x_proj`/`SLoRB_Weight` only under `if self.SLoRB:`, and
`sparse_modeling.py:819` guards the forward on `self.SLoRB and hasattr(self,"x_proj")`. No dangling
params, no shape errors, no zeroed-but-present branch. This is genuine removal.

**Held fixed across both arms:** student=teacher=`models/Llama--Llama2-7b`, `distill_model True`,
`hardness_task 1.0`, `hardness_kldiv 1.0`, `hardness_squarehead 0.0` (as published — logit KL only),
dataset dolmino-llama2, `block_size 4096`, `global_batch_size 256`, `mask_metric hessian_obd`,
`hard_mask_type nm_2_4`, `srste_decay 0.0`, `enable_hutchinson False`, `seed 1234` (matching
CAST-repro's manifest, not the trainer's 1337 default).

`enable_hutchinson False` is load-bearing twice over: it matches the published run, **and** enabling
it would silently force `gradient_checkpointing` off (double-backward incompatibility) and blow the
memory budget. Do not flip it to "test the contribution" here — that is a separate arm.

**SLoRB param count independently reproduced.** Summing `x_proj (in/k, in) + SLoRB_Weight (out, in/k)`
over the 7 projections x 32 layers at k=16:
```
in-scope tensors      : 224            (matches CAST scope 224: True)
in-scope weight elems : 6,476,005,376  (matches CAST cast_elements: True)
SLoRB extra params    : 848,429,056    (matches SPARSEFORGE_SAME_HARNESS.md: True)
as % of in-scope weights: 13.10%
```
Note: 13.10% of *in-scope weights*, or **12.59% of the full 6.74 B model**. The existing doc's
"+26.2%" is against the 3.238 G *surviving* weights after 50% pruning (848,429,056 / 3,238,002,688 =
26.20%) — a different and smaller denominator. All three are correct; they must not be mixed in one
sentence.

---

## TASK 3 — the FSDP-vs-mask-alignment contradiction: VERDICT

**Both SPEC.md and SparseForge are right, about different code.** SPEC.md §1's "Never FSDP" is
correct **for CAST**, and does **not** transfer to SparseForge, for a reason that has nothing to do
with wrapping granularity.

### FSDP does break shape/element alignment — measured

`use_orig_params=True` does **not** save you when a shard boundary lands mid-tensor. CPU/gloo, 4
ranks, uneven tensor sizes (3072/2048/1024 elems) to force mid-tensor cuts, mask registered
SparseForge-style as a frozen `nn.Parameter`:
```
[rank0] STYLE=param ws=4 -> {'NUMEL_MISMATCH': 2, 'BOTH_EMPTY': 4}
   ('NUMEL_MISMATCH', 'mods.0', 'w3072/m0/full3072')       <-- weight 3072 elems, mask 0
[rank2] ('NUMEL_MISMATCH', 'mods.1', 'w2048/m1024/full2048')  <-- weight 2048, mask 1024
```
With even sizes the boundaries happen to land on tensor edges and everything looks aligned
(`{'ELEMENT_ALIGNED': 2, 'EMPTY_SHARD': 4}`) — which is exactly why this hazard is easy to miss.
So the mechanism SPEC.md describes is **real and reproducible**.

### Why it is harmless for SparseForge: nothing reads `p.mask` in its optimizer

CAST is vulnerable because its selective-L1 decay lives *in the optimizer*, pairing `weight` with
`cast_mask` element-wise at `step()` time — when params are re-sharded. SparseForge consumes the mask
in **two** places, and its published config disables the one that would break:

**1. Optimizer — the mask-reading paths are all switched off.** `adamw.py` touches `p.mask` in
exactly three places, each gated:

| `adamw.py` | guard | published value | active? |
|---|---|---|---|
| `:102-112` SRSTE decay `grad += w*(1-mask)*decay` | `decay = self.decay` from `get_decay(it)` | `srste_decay 0.0` | **no** |
| `:163-197` CAST AdamS | `if cast_mode:` | `cast_mode False` | **no** |
| `:220-223` adaptive L1 | `if self.adaptive_l1_decay > 0.0` | `adaptive_l1_decay 0.0` | **no** |

Verified `main_llama.py:1395` `get_decay()` returns `0.0` for all iters when `srste_decay <= 0.0`,
and the published `args.json` has `srste_decay: 0.0`, `adaptive_l1_decay: 0.0`, `weight_scaling: false`.
So **`scaler.step(optimizer, decay=0.0)` never dereferences a misaligned mask.** The `mask.numel()
!= param.numel() → mask = None` degradation that silently destroyed CAST's mechanism is unreachable,
because the code that would degrade is already inert.

**2. Mask updates — done inside `summon_full_params`, which restores full 2-D shapes.**
`utils.py:209-214` wraps every mask update in
`FSDP.summon_full_params(fsdp_target, writeback=True, recurse=True)`, and `utils.py:152-158` documents
precisely the FlatParameter hazard SPEC.md warns about, as the *reason* for the context. Inside it all
ranks see identical full parameters, so `utils.py:251-273` additionally sets `_in_summon_context` to
suppress the `tau` all-reduce (`sparse_modeling.py:1034-1038`) — otherwise ranks that skipped
`update_mask` on a shape mismatch would desync NCCL.

**3. Forward — `sparse_modeling.py:787` takes `effective_mask = self.mask` while `hardening_x >= 1.0`,
and `:790-795` uses it directly once finalized**, explicitly to avoid `topk` on sharded masks
(`:784-786`). During FSDP forward the module's params are unsharded anyway.

**Verdict: the existing 5B checkpoint is NOT suspect on alignment grounds.** Its
`mask_info`/`finalization_done` and the independent `hard_fold` reproduction to **+0.0078 pp** of the
checkpoint's own anchor (SPARSEFORGE_SAME_HARNESS.md) corroborate that the mask↔weight pairing
survived. SPEC.md §1 should be narrowed from "Never FSDP" to *"never FSDP **when the optimizer pairs
weight with mask element-wise**"* — that is the actual invariant.

**Caveat, stated honestly:** the alignment tests above ran on **CPU/gloo**. `FSDP.summon_full_params`
and FSDP forward both hard-require CUDA (`RuntimeError: An FSDP-managed module unexpectedly has
parameters on cpu`), so the summon-context and forward-time claims rest on **code reading plus the
checkpoint's own reproduction**, not on a live 8-GPU run. Confirming them empirically needs a GPU and
was out of budget. The optimizer argument (the load-bearing one) is config-level and does not depend
on that.

### Memory plan

Reference: CAST at 7B measured **178.33 GiB peak without checkpointing (OOM)** / **174.04 GiB with
(OOM by ~100 MiB)** under plain DDP, and fit only via ZeRO-1 at **145.7 GiB/rank**. Measured from the
successful run's own log: `peak=138.7G` at step 0, **145.7 GiB steady**.

SparseForge is *easier*, because FSDP `HYBRID_SHARD` on a single 8-GPU node degenerates to
`FULL_SHARD` and shards **everything** — including `mask` and `hessian_diag`, which
`sparse_modeling.py:274-288, 316-327` register as frozen `nn.Parameter` **specifically so FSDP will
shard them**. Both student and teacher are FSDP-wrapped (`main_llama.py:1077`, `:1213`).

Per-rank static, ws=8, params 6,738,415,616 + SLoRB 848,429,056 = 7,586,844,672:

| item | GiB/rank |
|---|---:|
| fp32 params (/8) | 3.53 |
| fp32 grads (/8) | 3.53 |
| Adam m (/8) | 3.53 |
| Adam v (/8) | 3.53 |
| mask fp32, weight-shaped (/8) | 3.02 |
| hessian_diag fp32 (/8) | 3.02 |
| grad_ema / frozen_flags | ~0 (1-elem placeholders) |
| bf16 teacher (FSDP-wrapped, /8 + gather) | 12.55 worst case |
| bf16 all-gather buffer (~1 layer) | 0.88 |
| **static total** | **~33.6** |

Against 178.4 GiB that leaves ~145 GiB for activations — comfortable, and consistent with the
published run having succeeded at `batch_size 8` with `gradient_checkpointing True`.

**Micro-batch / grad-accum, `global_batch_size` held at 256:**

| micro | accum total | per-rank | ÷8 ok | activations vs mb=8 |
|---|---|---|---|---|
| 8 (**chosen, as published**) | 32 | 4 | yes | 1.00x |
| 4 | 64 | 8 | yes | 0.50x |
| 2 | 128 | 16 | yes | 0.25x |
| 1 | 256 | 32 | yes | 0.125x |

Both arms use **micro 8**. Arm 3 is strictly smaller than arm 2 (no SLoRB: −1.58 GiB/rank of sharded
fp32 optimizer state, −0.4 GiB params), so if arm 2 fits, arm 3 fits. If arm 2 OOMs, drop to micro 4
first — it halves activation memory at identical `global_batch_size`, so the token budget and the
optimisation trajectory are unchanged. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is exported
**inside** the script (torchrun children do not reliably inherit it from the calling shell).

---

## TASK 4 — cost, from the real run log

Source: `logs/cast_repro_full_20260809_211514.log` — the successful ZeRO-1 CAST run, which carries
cumulative elapsed seconds on every log line. 751 log points.

```
step 0 at 19s  ->  step 7499 at 110842s
7499 steps in 110823 s  =  14.778 s/step   (all-in, includes a ckpt save every 250 steps)

pure compute (720 x 10-step windows containing NO ckpt save):
  median 14.400  mean 14.794  p10 14.400  p90 14.500  s/step
peak mem reported: 145.7 GiB/rank
```
Cross-checked independently against checkpoint directory mtimes (two disjoint 250-step windows):
```
ckpt_step7000 -> ckpt_step7250 : 3696.1 s / 250 = 14.784 s/step
ckpt_step7250 -> loss_trace end: 3592.8 s / 250 = 14.371 s/step
```
Three estimates agree to <3%. **Use 14.78 s/step all-in.**

| | wall clock | GPU-days (8 cards) |
|---|---|---|
| 1 arm, 7500 steps | **30.8 h = 1.28 days** | 10.3 |
| 2 arms serial | **2.57 days** | 20.5 |

**The two arms cannot run concurrently on one node.** At ~33.6 GiB static + activations each they
would nominally fit in 178.4 GiB, but red line #5 forbids two serious 8-GPU experiments on one node,
and halving the SM budget would roughly double each arm's wall clock for zero net gain. **Serial
plan: arm 2, then arm 3, `.21` occupied ~2.6 days** (plus ~1 h model load / finalization overhead per
arm, and inline `lm_eval` at `eval_interval 100` — if that proves expensive, set
`--finalize_lm_eval False`).

Caveat: 14.78 s/step is CAST's harness, not SparseForge's. SparseForge adds SLoRB matmuls (+13.10%
params), a mask update every 10 steps inside `summon_full_params` (an all-gather collective), and
FSDP gather/scatter traffic CAST's ZeRO-1 does not pay. **Arm 2 is likely somewhat slower than 14.78
s/step; I have no measurement.** Treat 2.6 days as a floor. Arm 3 should be slightly faster than arm 2.

**`.21` status: unverified.** Two SSH attempts (`ConnectTimeout` 12 s then 25 s, `-p` omitted per the
global `Port 36000` config) both timed out — consistent with the brief's note that another agent is
scoring a 41 GB checkpoint there. **Confirm `.21` is free with `nvidia-smi` before launching.** LOCAL
was checked and is fully occupied (8x L20A at 100%, 137,578 MiB each = Paper B keep14) — correctly
left alone.

---

## Ways this silently produces garbage

Ordered by (probability x damage). Every one of these completes successfully while measuring the
wrong thing.

1. **Hardening schedule left at 12000/5000.** `hardening_x` never leaves 1.0, the mask never hardens,
   `sparse_modeling.py:787` early-outs to the soft mask on every forward, and you get a **dense**
   model reported as 2:4. Loss curve looks fine. Mitigated (5294/2206, lands exactly on 7500) — but
   this is the first thing to re-check if a resume or a config edit touches `max_iters`.
   **Runtime check:** the log must show `hardening_x` leaving 1.0 by step ~5300; the final checkpoint
   must pass `tools/verify_2of4_hf_export.py` with `bad_tiles = 0`.

2. **`hessian_obd` is magnitude pruning** (the finding above). `hessian_diag ≡ 1.0` for the entire
   run, so the "Hessian-aware" criterion is inert. Not fixed, deliberately — fixing it would change
   the method. **Must be stated in any writeup.**

3. **dtype.** uint32 read as uint16 → 50% injected zeros, 0% out-of-vocab, both existing guards
   blind. Mitigated three ways (metadata precedence, explicit `--data_dtype uint32` asserted in the
   script's preflight, token-count cross-check that raises on a 2.0/0.5 ratio).

4. **LR compression.** Not garbage, but not neutral either. The runs are **not** LR-matched to
   CAST-repro (1e-4/375-scaled vs 2e-5/375). Any claim must be "matched data + budget", never
   "matched optimisation".

5. **Provenance hole: `data/dolmino-mix-1124-raw` no longer contains what links 1-2 trained on.**
   The published chain's links 1-2 used `dataset=dolmino-mix-1124-raw`, but that directory today
   holds only raw `.json.gz` shards — no `train.bin`, no `.npy`, no `dtype.txt`
   (mtime 2026-08-08, i.e. re-downloaded four months *after* the Mar 31/Apr 1 runs). **The exact
   token stream those two links consumed is not reconstructible.** Doesn't affect arms 2/3 (they use
   dolmino-llama2 from scratch, `resume False`), but it does mean the published checkpoint's first
   two-thirds of training is not reproducible, and the `sparseforge_dolmino_link2` row inherits that.

6. **`final_finetune_iters` accidentally non-zero.** Any value > 0 adds uncounted tokens CAST never
   got and, if `dataset` were also switched, re-imports the MC-QA contamination. Set to 0 and
   verified against `main_llama.py:3436`.

7. **`enable_hutchinson True`.** Would silently disable `gradient_checkpointing` (double-backward),
   likely OOM, and confound the comparison with an untested contribution. Pinned False.

8. **Arm 3 misread as "SparseForge without SLoRB" when it is the *amputation* variant.** The existing
   `hard_drop` export (−4.9 pp) is amputation damage from an SLoRB-trained checkpoint; arm 3 is a
   from-scratch SLoRB-free run. **These are different numbers and must never share a row.**

9. **`hardening_period` "helpfully" set > 0.** Would enable the second, independent
   `harden_fraction()` path (`main_llama.py:3044`) on top of the annealed schedule — two hardening
   mechanisms fighting. Published value is 0; keep it.

10. **Seed drift.** `main_llama.py`'s default is 1337; CAST-repro used **1234**. The script pins 1234.
    A comparison at different seeds with n=1 per arm is not a comparison.

11. **`13.10%` vs `26.2%` SLoRB overhead.** Same tensors, different denominators (in-scope weights vs
    surviving weights after 50% prune). Mixing them in one sentence is a factual error.

---

## Files changed

| file | change |
|---|---|
| `main_llama.py` | `--data_root`/`--data_dtype`/`--require_val`/`--val_holdout_tokens`; dtype precedence + contradiction check; token-count cross-check; `_load_bin` raises instead of falling back; optional val via train tail. All defaults preserve prior behaviour (verified: published `args.json` still parses; `qa_format_sft_llama` still reads uint16/129,752,281 tokens). |
| `Mixture-of-Memory/scripts/_run_sparseforge_tokenmatched.sh` | new, `ARM=slorb`/`noslorb`. |
| `Mixture-of-Memory/baselines/cast_repro/SPARSEFORGE_TOKENMATCHED_PREP.md` | this file. |

`main.py` was **not** modified — it is not the trainer for this experiment (see Task 1(i)).
**Nothing was launched.** No GPU work was performed for this document.
