# B01 — d_bottle-width persist path: VERDICT

**Date** 2026-08-17 · **GPU** ~5 min on ONE B200 (LOCAL); everything else 0 GPU
**Commit** `58ff2d2` (content) + `6260eab` (provenance note — see the caveat below)
**Outcome** The blocker is **discharged**. `ready_queue.py` now reports B01 as
**`ready_gpu`** — the queue's only `ready_gpu` item.

---

## 1. What the blocker was

`STATUS.json.blocking_dependency` said, correctly:

> The store still holds the RESTORED full-width hidden, not the d_bottle latent,
> so the storage-saving claim is currently unmeasurable end-to-end.

Mechanically: `BottleneckLayer.forward` computed `up(act(down(h)))`, so the layer's
output was back at `hidden_size`, and `QCMemModel.write_chunk` cached *that*. The
funnel therefore constrained the **rank** of what was stored but not the **width of
the bytes** stored. Consequence: `bytes/token` was **identical** across the
bottleneck and vanilla arms, so the storage axis of the four-arm gate was a
constant and the gate could not measure the thing it exists to measure.

## 2. What changed

`QCMemModel(..., persist_bottleneck_latent=True)` moves the funnel's `up` from the
**write** side to the **read** side:

| | WRITE persists | READ applies |
|---|---|---|
| legacy (flag **off**, default) | `up(act(down(h)))` — `hidden_size` wide | `layers[j:] → norm → lm_head` |
| persist (flag **on**) | `act(down(h))` — **`d_bottle` wide** | `up` per cached piece, then the same read band |

**Why this is exact, not an approximation.** The write band is
`h_j = up(act(down(g(x))))` and the read band is `f(h_j)`. Persisting
`s = act(down(g(x)))` and reading `f(up(s))` is the *same composition* — only the
cut point moves. Nothing is dropped or reordered.

**Which op goes where, and why:**

- `act` (GELU) **stays on the write side.** It is the nonlinearity; it is also
  elementwise, so storing `down(h)` and applying `act` at read time would be
  equally exact and equally `d_bottle`-wide. I chose post-activation so the
  deferred computation is a *single bias-free linear map* (smallest possible, easiest
  to audit), and because GELU's saturation narrows the stored dynamic range — which
  matters if the store is later quantised below bf16.
- `down` **cannot** be deferred: it *is* the width reduction.
- `up` **cannot** be deferred unless the funnel is the **last** layer of the write
  band, or `layers[bottleneck_layer+1 : j]` would consume the full-width hidden
  before the store. Hence `resume_j == bottleneck_layer + 1` is **enforced**, not
  recommended.
- Any future funnel variant with a non-elementwise op *after* `down` (a norm over
  the `d_bottle` axis, a token-mixing op, a residual add) would break the split.
  The three properties this relies on are now written into `BottleneckLayer`'s
  docstring so a redesign has to revisit them.

**Backward compatibility.** The flag defaults **off**, and with it off every path
takes the byte-identical branch it took before (§4 measures this). This is
load-bearing: a large number of already-published QCMem numbers depend on
`write_chunk`'s existing return value.

Files: `src/memory/qcmem/qcmem_model.py` (flag, `_write_band`, `_lift_latent`,
`_resolve_deferred_funnel`, `store_bytes_per_token`),
`scripts/semantic_bottleneck_model.py` (docstring),
`scripts/eval_qcmem_locomo.py` (`--persist_bottleneck_latent` + 5 guards),
plus two new gate scripts.

## 3. Equivalence — MEASURED, on the real 8B endpoint

`outputs/qwenbott_funnel_L12_d512/final.pt` (16,390,019,143 B, `step=2000`),
`resume_j=13`, bf16, 1×B200, sdpa. One model load, two `QCMemModel` views over the
**same parameters** (a second `torch.load` could not prove they are the same weights).

| tensor | max abs diff | differing elements |
|---|---|---|
| `read` | **0.000000e+00** | **0** / 126,562,688 |
| `read_tail` | **0.000000e+00** | **0** / 1,215,488 |
| `decode` (prefill + 3 O(1) steps) | **0.000000e+00** | **0** / 607,744 |
| **total** | **0.000000e+00** | **0 / 128,385,920** |

Greedy tokens identical on both paths: `[782, 2090, 288, 11]`.

**So the answer is BIT-IDENTICAL, and that is a measurement, not an inference from
the algebra.** The task brief anticipated bf16 last-bit drift from the rearrangement;
there is none, and the reason is that `_lift_latent` applies `up` **per cached
piece, before the pack** — the same `[1, T_c, d_bottle]` GEMM shapes the legacy run
used, in the same dtype. Lifting the packed `[1, H, d_bottle]` in one call would
have been just as correct mathematically but would have changed the GEMM shape and
almost certainly cost a few ulps.

Fixture gates agree: **34 PASS / 0 FAIL on CPU and 34 PASS / 0 FAIL on CUDA**,
across `hidden=4096 / d512 / bf16` (the real width, ratio and dtype of the 8B
endpoint; only depth reduced) and a `hidden=256 / d32` fp32+bf16 sweep. Equivalence
max abs diff `0.0` in every cell, including fp32 where any rearrangement error would
show far below bf16's rounding floor.

## 4. bytes/token — MEASURED off the tensors, not computed from the architecture

The gate's mandatory quantity is *"bytes/token of what is written to the store (not
of the restored hidden)"*. Measured as `numel * element_size() / T` on the tensor
`write_chunk` actually returns:

| | shape | dtype | bytes/token |
|---|---|---|---|
| legacy | `[1, 64, 4096]` | bf16 | **8192** |
| persist | `[1, 64, 512]` | bf16 | **1024** |

**Measured ratio 8.0000×**, expected `hidden_size / d_bottle = 4096/512 = 8`.
A 3×256-token context store went from **6,291,456 B → 786,432 B**.
`store_bytes_per_token()` returns `[8192, 1024]`, cross-checked against the measured
tensors rather than trusted.

**The storage axis is no longer constant across arms**, which is exactly what the
four-arm gate needed.

## 5. Vanilla regression — the check that protects published numbers

The baseline is the file **as committed** (`git show HEAD:...qcmem_model.py`,
materialised into a separate module), not a hand-written reimplementation of what I
believe the old code did.

- 9 observable tensors (`write_sink`, `write_q`, `write_ctx_cat`,
  `write_chunks_cat`, `write_prefill_hj`, `read`, `read_tail`,
  `resume_forward_ids`, `decode`)
- × {**bottleneck arm**, **`bottleneck_dim=0` vanilla arm**} × {CPU, CUDA} × {fp32, bf16}

**Result: max abs diff `0.0`, 0 differing elements in every cell.** The
`bottleneck_dim=0` arm is checked explicitly rather than assumed covered, because it
is the configuration every published QCMem number was produced under.

**The committed evidence is the PRE-commit run, and that matters.** Once the change
landed, `HEAD` *is* the new file, so the comparison becomes self-vs-self and proves
nothing. The gate detects this and stamps the condition into its JSON: the committed
`persist_selftest_{cpu,cuda}_20260817.json` carry `baseline_warning: ""` (measured
against a HEAD *without* the flag → valid), while a post-commit re-run carries
`"HEAD ALREADY CONTAINS the persist flag → this comparison is self-vs-self and proves
nothing"`. So the regression claim is auditable rather than resting on my ordering
being remembered correctly. To re-establish it from scratch, run the gate with
`src/memory/qcmem/qcmem_model.py` reverted, or point the baseline at `58ff2d2^`.

The repo's own `scripts/qcmem_resume_primitive_check.py` also still passes at all
`j` (max diff `0.000e+00`).

## 6. Negative controls — all 10 verified to fire

A silent fallback here would report an 8× saving it did not get, so every illegal
configuration raises instead of degrading:

`QCMemModel` refuses: no funnel at `layers[resume_j-1]` · `resume_j == bottleneck_layer` ·
`resume_j > bottleneck_layer+1` · `resume_j == 0` · a full-width cached piece handed
to a persist read.

`eval_qcmem_locomo.py` errors (rc=2, each verified by running it): no
`--bottleneck_ckpt` · `--baseline hcache` · `--top_prepay_b 4` · the
**stock-continued** ckpt `outputs/qwenbott_baseline_L12` (`arch_meta bottleneck_dim=0`)
· `--resume_j 12 != bottleneck_layer+1`. The last two were checked against the real
`arch_meta.json` files, not synthetic ones.

## 7. A pre-existing defect found while verifying (NOT introduced here)

`write_chunks`' docstring claimed bit-identity with per-chunk `write_chunk`. That is
**too strong whenever a funnel sits in the write band.** Two independent sources,
kept separate here because they are different measurements:

**(a) The pre-existence proof** — git HEAD code only, both `qcmem_model.py` and
`semantic_bottleneck_model.py` loaded via `git show`, so this change's files take no
part (`preexisting_writechunks_gemm_*_HEADonly_20260817.txt`):

| configuration | differing elements | max abs diff |
|---|---|---|
| CUDA bf16, `hidden 4096 / d512`, funnel in band | 3764 / 393,216 | 1.5625e-02 |
| CUDA bf16, **identical shapes, funnel removed** | **0** / 393,216 | 0 |
| CPU fp32, `hidden 256 / d32`, funnel in band | 11271 / 16,384 | 6.9849e-09 |
| CPU fp32, **identical shapes, funnel removed** | **0** / 16,384 | 0 |

The funnel-removed rows are the control that makes this attributable: identical
shapes, identical harness, difference vanishes.

**(b) The same phenomenon as the gate measures it** (different seeds/chunk mix, so
the counts differ slightly — this is the *legacy* arm of check 5, not the persist arm):
CPU fp32 `11218 / 16,384 @ 5.588e-09`; CUDA bf16 `real/bfloat16`
`6422 / 393,216 @ 3.125e-02`. CPU `real/bfloat16` and both `tiny/bfloat16` cells are
`0` differing.

Isolated to a bare op: `nn.Linear(4096, 512, bias=False)` in CUDA bf16 at `B=2`
differs from two `B=1` calls in 82/49,152 elements (max abs 3.9062e-03); the CPU fp32
`256→32` analogue differs in 1376/1536 (max abs 6.1095e-07) and is `0` in bf16. The
funnel's `down`/`up` are the only plain `nn.Linear`s in the band, which is why the
divergence appears only when the funnel is present — a batched-vs-unbatched GEMM
blocking difference, not a QCMem defect.

The docstring's **semantic** claim (no cross-chunk information flow) is intact and is
what its argument actually establishes; only the **bitwise** wording was wrong. It now
records the counterexample with its numbers, and the gate asserts *"persist path no
worse than legacy"* rather than an absolute bit-identity it cannot have. My first
draft of the gate did assert absolute bit-identity and reported a FAIL — which is how
this surfaced.

## 8. `ready_queue.py` — actual output, before and after

`STATUS.json` was updated **append-only**: 2 keys added
(`blocking_dependency_discharged_20260817`, `defect_found_while_verifying_20260817`),
**0 removed, 0 changed** — verified byte-for-byte against `git show HEAD:STATUS.json`.
The new key carries `supersedes_key`, `supersedes_paths_verbatim`, and a `discharges`
pointer list naming the four exact dotted blocker paths, plus `evidence` paths that
were asserted to exist on disk before writing.

**Before** (`needs_prior_gate`):

```
  B01-semantic-bottleneck-memory-ready-models
     why: gate + kill gate + novelty all OK, but 4 un-discharged blocker(s) ...
     ! blocker STILL LIVE [blocking_dependency.statement]: ...
     ! blocker STILL LIVE [blocking_dependency.mechanism_verified_in_code]: ...
     ! blocker STILL LIVE [blocking_dependency.consequence]: ...
     ! blocker STILL LIVE [blocking_dependency.from_proposal]: ...

SUMMARY: 0 ready_gpu, 8 ready_cpu (0 GPU, dispatchable NOW), 3 needs_prior_gate
```

**After:**

```
=== ready_gpu  (1) ===
  B01-semantic-bottleneck-memory-ready-models
     why: gate + kill gate + adjudicated novelty all present (novelty_checked=true), no un-discharged blocker
     gate[next_gate_executable_20260814]: Four-arm comparison at ONE scale, with the bottleneck latent ACTUALLY PERSISTED: ...
     cost: ~25-40 GPU-h for the four arms at 8B if the two existing 8B j=12/d512 CPT endpoints are reused; ...

SUMMARY: 1 ready_gpu, 8 ready_cpu (0 GPU, dispatchable NOW), 2 needs_prior_gate
```

`--json` confirms: `lifecycle=ready_gpu`, `live_blockers=0`,
`discharged_blockers` = all four paths, `problems=NONE` (no dangling pointers).

## 9. Honest caveats

1. **The commit message is wrong, and I did not rewrite history.** The code was
   staged for its own commit and was swept into `58ff2d2` — subject *"chore(status):
   topology correction — our 5 nodes ARE taiji pods"* — by a **concurrent
   `git commit` from another agent** that picked up my index (my own commit attempt
   died on `.git/index.lock`). Content at `58ff2d2` is correct and complete
   (`git diff HEAD` over these files is empty). I left history alone because other
   agents were actively committing to `main`; an empty commit `6260eab` documents the
   mapping so a bisect does not chase a subject line that never mentions B01.
2. **Only the storage MECHANISM is discharged.** The four-arm **quality** comparison
   has not been run; the ~25–40 GPU-h estimate stands unchanged.
3. **Still open**, untouched by this work: no full quality/storage/latency frontier;
   Read-LoRA + Write-LoRA never combined; the strong-model/A13B leg; and the
   **residual-FREE vs residual-PRESERVING** ablation that `related_work_status`
   flagged as the arm separating B01 from Variable-Width Transformers.
4. **What was NOT tested:** no quality metric, no LoCoMo/RULER/LongEval run, and no
   multi-GPU/sharded path (the Hy3 subclass overrides `_run_layers` with a pre-cache
   signature; `_write_band` forwards cache kwargs only when non-default so that path
   is unbroken, but `persist_bottleneck_latent` on Hy3 is refused by design rather
   than supported — Hy3 has no `BottleneckLayer`).
5. Environment note: this node lacks `pandas`, `datasets`, `peft` and `pytest`, so
   `eval_qcmem_locomo.py`'s argparse guards were exercised with import-satisfying
   stubs for `pandas`/`datasets` (every stub entry point raises). The missing modules
   are pre-existing — HEAD's own copy of the file fails to import identically.

## 10. Reproduce

```bash
# CPU fixture gate — 34 PASS / 0 FAIL
/opt/conda/envs/torch-base/bin/python scripts/qcmem_bottleneck_persist_selftest.py \
    --json_out /tmp/cpu.json > /tmp/cpu.log; echo $?

# CUDA fixture gate — 34 PASS / 0 FAIL
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/torch-base/bin/python \
    scripts/qcmem_bottleneck_persist_selftest.py --device cuda \
    --json_out /tmp/cuda.json > /tmp/cuda.log; echo $?

# 8B real endpoint — 6 PASS / 0 FAIL  (~40 s load, <2 s of forwards, 1 card)
CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/torch-base/bin/python \
    scripts/qcmem_bottleneck_persist_8b_smoke.py --json_out /tmp/8b.json > /tmp/8b.log; echo $?

# the queue verdict
/opt/conda/envs/torch-base/bin/python proposal/ready_queue.py
```

Note the `cmd > file; echo $?` form: piping into `tail` would report the **pipe's**
rc, not the gate's.

**Evidence on disk** (`evidence/`, all committed):
`persist_8b_smoke_20260817.{json,txt}` ·
`persist_selftest_{cpu,cuda}_20260817.{json,txt}` ·
`preexisting_writechunks_gemm_{HEADonly,cpu_HEADonly}_20260817.txt`
