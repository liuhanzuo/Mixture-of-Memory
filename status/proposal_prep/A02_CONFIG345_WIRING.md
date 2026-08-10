# A02 Config 3/4/5 — Code Gap & Wiring Spec
# Prepared: 2026-08-09 (read-only scout; no GPU job launched, no file edited)

---

## 0. Summary

| Config | Name | Status | Lines needed | Note |
|--------|------|--------|--------------|------|
| C3 | overlap-write w32 | BLOCKED | ~80-120 per script | New param + wrapper around write_chunks |
| C4 | Write-LoRA only (no Read) | BLOCKED | ~60-90 per script | `set_adapter(["write"])` is PEFT-feasible |
| C5 | Write-LoRA + Read-LoRA | BLOCKED | ~60-90 per script | `_load_with_write_lora` port + context manager |
| Grand total | 3 configs × 4 scripts | — | **~600-800 lines** | ~1-2 days coder work |

**Cost ranking** (cheapest to most expensive to implement):
1. **C4/C5** (similar complexity) — port `_load_with_write_lora` once + `_write_lora_enabled`
   context manager once, then wire into 4 eval scripts. C4 and C5 share 90% of the code;
   the only difference is one `set_adapter` call.
2. **C3 (overlap-write)** — more invasive: requires porting `_e2_write_chunk` into
   `qcmem_generate` (which is used by ALL 4 natural-task scripts), plus plumbing the
   left-context prefix list through the selector loop. Touches more core logic.

---

## 1. Synthetic harness implementation (what already exists)

### Config 3: `scripts/eval_p017_e2_overlap_write.py`

| Function | Lines | What it does |
|----------|-------|--------------|
| `_e2_write_chunk(qc, chunk_ids, left_ctx_ids)` | 185-207 | Prepend `w` left-context tokens, run layers 0..resume_j, DISCARD prefix, return chunk's h_j |
| `_e2_write_residual(qc, chunk_ids)` | 210-221 | w=0 identity gate: asserts _e2_write_chunk(w=0) == write_chunk bit-for-bit |
| `_run_e2(qc, bos_id, selected_chunks, selected_left_ctx, ...)` | 228-270+ | Outer loop: call _e2_write_chunk per selected chunk; Read + decode identical to Arm B |

### Config 4/5: `scripts/eval_p018_e4_2x2_writecontrol.py`

| Function | Lines | What it does |
|----------|-------|--------------|
| `_load_with_write_lora(model_path, dtype, attn_impl, device, lora_adapter, write_lora_ckpt, resume_j)` | 190-261 | Load base + Read-LoRA as "default" + Write-LoRA as "write" into a PeftModel; fail-closed: Write must be on layers 0..resume_j-1, disjoint from Read |
| `_write_lora_enabled(peft_model)` [context manager] | 265-278 | Calls `tuner.set_adapter(["default","write"])` on enter, restores `"default"` on exit |
| Usage in write phase | ~735, ~773 | `with _write_lora_enabled(peft_model): h = qc.write_chunk(...)` |

---

## 2. Config 3 wiring: overlap-write in natural-task scripts

### 2.1 What needs to change (per eval script)

**Central change: `eval_qcmem_babilong.py::qcmem_generate` (lines 583-800+)**

This function is imported by `eval_qcmem_longeval.py`, `eval_qcmem_longbench.py`, and
`eval_qcmem_locomo.py` as:
```python
from eval_qcmem_babilong import qcmem_generate as qcb_generate
```
So adding overlap-write support to `qcmem_generate` propagates to ALL four scripts
if done via an optional parameter (backward-compatible default=no overlap).

**Option A (recommended): add `overlap_write_w: int = 0` to `qcmem_generate`.**

Changes inside `qcmem_generate`:
1. Import `_e2_write_chunk` at the top of `eval_qcmem_babilong.py` (copy the
   12-line function from `eval_p017_e2_overlap_write.py`, or refactor into a shared
   utility in `src/memory/qcmem/`).
2. After selector returns `sel_idx`, build a `left_ctx_list` (list of left-context
   token tensors for each selected chunk): `left_ctx_list[i] = context_chunks[sel_idx[i]-1][-w:]`
   (last `w` tokens of the preceding chunk, or None if chunk is first). ~10 lines.
3. Replace the `qc.write_chunks(...)` call (line 671) with a loop:
   ```python
   if overlap_write_w > 0:
       selected_hj = [_e2_write_chunk(qc, context_chunks[i], left_ctx_list[j])
                      for j, i in enumerate(sel_idx)]
   else:
       selected_hj = qc.write_chunks([context_chunks[i] for i in sel_idx])
   ```
   ~5 lines.
4. Disable the `reader_attn` / `iter_reader_attn` path for overlap-write (those
   pre-compute all context h_j with the STANDARD write, then the selected ones would
   be re-written anyway with E2; simplest fix: force recompute when overlap_write_w>0).
   ~5 lines.

**Subtotal for `qcmem_generate`: ~30-40 lines.**

**Per eval script (eval_qcmem_babilong.py / longeval.py / longbench.py / locomo.py):**
- Add `--overlap_write_w` argparse argument (int, default=0): ~5 lines
- Pass `overlap_write_w=args.overlap_write_w` to `qcmem_generate` call: ~1 line
- Total per script: **~6 lines**

**For eval_ruler_qcmem.py:**
RULER uses its own internal write loop (not `qcmem_generate`). Need to add
`_e2_write_chunk` call to the RULER write path separately. The existing harness
`eval_p017_e2_overlap_write.py` already handles RULER/niah_multikey_1 natively.
For A02 phase-1, RULER Config 3 can use the EXISTING `eval_p017` harness directly
(no code change needed for RULER).

**Total new code for Config 3: ~70-90 lines across all scripts (concentrated in
`qcmem_generate` modification, ~40 lines core + ~6 lines × 4 eval scripts).**

### 2.2 RULER harness (no change needed)

The existing `scripts/eval_p017_e2_overlap_write.py` already handles RULER
niah_multikey_1 for w={0,32,64,128}. A02 can reuse it for the RULER column.
The A02_PHASE1_LAUNCH.md §5 "Config 3 on RULER" can therefore be:

```bash
# RULER Config 3 (w=32): use the existing P0.17 harness
for g in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_p017_e2_overlap_write.py \
    --model_path $BASE --lora_adapter $READ_LORA \
    --resume_j 12 --topk 12 --sink_tokens bos \
    --tasks niah_multikey_1 --lengths 4k 8k 16k 32k \
    --overlap_widths 32 --num_samples 100 \
    --chunk_size 512 --add_bos 0 \
    --output_name a02_ruler_c3_overlapw32 \
    --num_shards 8 --shard_index $g \
    > logs/a02_ruler_c3_shard${g}.log 2>&1 &
done
```
(Verify exact argnames from `eval_p017_e2_overlap_write.py --help` before running.)

---

## 3. Config 4 & 5 wiring: Write-LoRA in natural-task scripts

### 3.1 Config 4 vs Config 5 clarification

From `eval_p018_e4_2x2_writecontrol.py`:
- `_load_with_write_lora` ALWAYS loads Read-LoRA as "default" AND Write-LoRA as "write".
- The **default active state** after loading is `set_adapter("default")` (Read only).
- The `_write_lora_enabled` context manager temporarily activates `["default","write"]`
  (both LoRAs simultaneously) for the duration of each write call.

Therefore:
- **Config 5 (Write+Read LoRA)** = existing `_write_lora_enabled` path → both LoRAs
  active during write, Read-only during read. This is the current implementation.
- **Config 4 (Write-only LoRA, no Read)** = requires activating only `["write"]` during
  write AND during read. Read path would have zero LoRA on layers 12..35 (base only).

### 3.2 Config 4 PEFT feasibility (confirmed — read code, not guessing)

PEFT version on .73/.82/.104: `0.19.1`

`LoraLayer.set_adapter` (verified source):
```python
def set_adapter(self, adapter_names: str | list[str], inference_mode=False):
    ...
    self._active_adapter = adapter_names  # stores the list
```

`Linear.forward` (PEFT tuner):
```python
for active_adapter in self.active_adapters:
    if active_adapter not in lora_A_keys:
        continue  # ← SKIP silently
    ...
    result = result + lora_B(lora_A(dropout(x))) * scaling
```

**Conclusion**: `set_adapter(["write"])` on a layer that only has "write" (layers 0..11)
activates it correctly. On layers 12..35 (only "default" / Read-LoRA), "write" is not
in `lora_A_keys` → the `continue` fires → **no LoRA delta** on those layers → base
model passthrough. This is **exactly "Write-LoRA without Read-LoRA"**.

**Config 4 is PEFT-feasible with zero additional library changes.** The implementation
just needs to call `set_adapter(["write"])` instead of `set_adapter(["default","write"])`.

### 3.3 New loading function needed

To support Config 4 ("write only, no read"), we need a **second context manager**
alongside `_write_lora_enabled`:

```python
@contextlib.contextmanager
def _write_lora_only_enabled(peft_model):
    """Activate ONLY Write-LoRA (layers 0..11) for the write phase.
    Read-LoRA ("default", layers 12..35) is INACTIVE throughout.
    Use for Config 4: Write-LoRA-only inference."""
    tuner = peft_model.base_model
    tuner.set_adapter(["write"])   # ← only difference vs _write_lora_enabled
    try:
        yield
    finally:
        tuner.set_adapter(["write"])  # keep write-only; no "default" to restore
```

But for Config 4, the READ phase should also be "no LoRA" (since we're testing Write-LoRA
in isolation). The simplest implementation: for Config 4, use `peft_model.base_model.model`
(the unwrapped base) during the read phase, and `peft_model` with `_write_lora_only_enabled`
during the write phase. This requires:
- Storing both `peft_model` (for write) and `model` (for read) as separate refs
- A `QCMemModel` variant where write uses peft_model and read uses base model

This is about 30-40 extra lines beyond Config 5, but it is NOT the natural architecture:
normally Config 4 = same model for write and read, just with different adapters.
The simpler and more correct path: use `peft_model` for everything; call
`set_adapter(["write"])` before write operations, `set_adapter([])` (or `disable_adapter`)
before read operations. PEFT 0.19.1 supports `disable_adapter()` context manager.

### 3.4 Per-script wiring summary

Each natural-task eval script needs:

**1. New argparse args** (~8-10 lines):
```python
parser.add_argument("--write_lora_ckpt", type=str, default="",
                    help="path to trained Write-LoRA adapter dir (layers 0..resume_j-1)")
parser.add_argument("--write_lora_mode", type=str, default="none",
                    choices=["none","write_only","write_and_read"],
                    help="none=no write lora; write_only=Config4; write_and_read=Config5")
```

**2. Model loading** (replace the single-LoRA load block, ~30-40 lines):
```python
if args.write_lora_ckpt and args.lora_adapter:
    # Config 4 or 5: two-adapter load
    peft_model, write_sha, write_layers = _load_with_write_lora_minimal(
        model, args.lora_adapter, args.write_lora_ckpt, args.resume_j)
    model = peft_model.base_model.model
elif args.lora_adapter:
    # Config 2: single Read-LoRA (existing path)
    peft_model = PeftModel.from_pretrained(model, args.lora_adapter).eval()
    model = peft_model.base_model.model
```

**3. Write call wrapping** (in `qcmem_generate` or at call site, ~20-25 lines):
```python
# In qcmem_generate, add optional peft_model + write_lora_mode params
# wrap write_chunks with context manager:
if write_lora_mode == "write_and_read":  # Config 5
    with _write_lora_enabled(peft_model):
        selected_hj = qc.write_chunks(...)
elif write_lora_mode == "write_only":  # Config 4
    with _write_only_adapter_enabled(peft_model):
        selected_hj = qc.write_chunks(...)
else:
    selected_hj = qc.write_chunks(...)  # Config 2 (unchanged)
```

**Per script total**: ~60-90 lines (10 args + 35 load + 25 write-wrap + 10 doc update)

**Grand total for Configs 4+5**: ~280-380 lines across 4 natural-task scripts plus
one shared utility function (`_load_with_write_lora_minimal`, ~50 lines) that can live
in `eval_qcmem_babilong.py` or a new `scripts/qcmem_write_lora_utils.py`.

---

## 4. Where to copy from (exact source locations)

| Item | Source | Lines |
|------|--------|-------|
| `_e2_write_chunk` | `scripts/eval_p017_e2_overlap_write.py:185-207` | 23 lines |
| `_load_with_write_lora` | `scripts/eval_p018_e4_2x2_writecontrol.py:190-261` | 72 lines |
| `_write_lora_enabled` (context manager) | `scripts/eval_p018_e4_2x2_writecontrol.py:265-278` | 14 lines |
| New `_write_lora_only_enabled` | NEW (3-line variant, see §3.3) | ~10 lines |

---

## 5. Recommended coder scope

**Minimal viable PR for Config 4+5** (skips C3, lower risk):
1. Create `scripts/qcmem_write_lora_utils.py` with:
   - `_load_with_write_lora_minimal` (port of p018 version, minus RULER-specific checks)
   - `_write_lora_enabled` (identical copy)
   - `_write_lora_only_enabled` (new, ~10 lines)
2. Modify `eval_qcmem_babilong.py::qcmem_generate` — add `write_lora_mode` + `peft_model`
   optional params and the 3-branch write wrap (~25 lines)
3. Add `--write_lora_ckpt` + `--write_lora_mode` to each of 4 eval scripts, update load
   block (~40 lines each)

**Total: ~400-450 lines, ~1 coder-day**

**Config 3 add-on** (after C4/C5):
1. Copy `_e2_write_chunk` into `qcmem_write_lora_utils.py`
2. Add `overlap_write_w: int = 0` to `qcmem_generate`, add the left-ctx construction +
   conditional write loop (~45 lines total in qcmem_generate)
3. Add `--overlap_write_w` to 4 eval scripts (4 × 6 lines)

**Total C3 add-on: ~70 lines, ~0.5 coder-day**

---

## 6. What RULER already has (no new code needed)

`scripts/eval_p017_e2_overlap_write.py` already supports RULER niah_multikey_1 +
variable_tracking for overlap-write w={0,32,64,128}.

`scripts/eval_p018_e4_2x2_writecontrol.py` already supports RULER niah_multikey_1
for Write-LoRA (Config 4+5 equivalent under the A/BB/E0/X/Y arm taxonomy).

Therefore, **A02 phase-1 can run Configs 3/4/5 on RULER immediately with zero new code**,
using these existing synthetic harnesses. The gap is ONLY the 4 natural-task scripts
(BABILong, LongEval, LongBench, LoCoMo).

---

## 7. File list / disk

All source files referenced are on **wzc1** (LOCAL + .21) at:
`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/scripts/`

The zwfy6 checkout may be behind; coder should check `git status` on .73 before editing
(or MAIN scp's the modified scripts from wzc1 to zwfy6 after coding).

Eval assets (Write-LoRA ckpt, Read-LoRA, base model, datasets) are on **zwfy6** only.
Modified eval scripts must be deployed to zwfy6 before running.
