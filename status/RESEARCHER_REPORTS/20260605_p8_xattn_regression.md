# P8 `use_memory_xattn` BABILong Regression — Root Cause

**Status:** diagnosis only. No code changed, no training launched. Fix proposed below for review.

## 1. Confirmed root cause
The dedicated xattn read injects a **full-magnitude, non-suppressible** memory contribution
into the frozen backbone at all 32 layers. Two facts combine:

- `MemoryCrossAttentionRead.read` (`selector.py:1392-1414`) runs its **own softmax over
  only the N slots** (`F.softmax(attn_logits, dim=-1)`, `:1396`). A softmax with no null
  option **always sums to 1**, so it always returns a full-magnitude blend of `V`, no matter
  how irrelevant/cold the slots are. There is no "attend to nothing" escape valve.
- In `layer.py:1323-1330` the read output is added **directly** to `next_hidden`, NOT
  multiplied by the shared `inject_gate g` (per the comment at `:1295-1299`). The only
  suppressor is the per-head gate (init 0.4) + small-random `out_proj` (std 0.02) — both
  nonzero by design, so the read is always live. The lone cold guard at `layer.py:1328`
  is gated by `cfg.zero_alpha_on_cold_start`, which **defaults to `False`**
  (`config.py:256`) and is **absent from `adapter_config.json`** → it never fires.

Net effect: a gated, unscaled vector is added to the residual at every layer; small
per-layer perturbations compound across depth and wreck the pretrained backbone's
zero-shot QA. Training reached lm≈2.22 only because the model co-adapted with memory
always-on.

## 2. Why 0k is the tell
At 0k a sample fits one chunk → `chunks==1` → no streaming, straight to generation
(`run_babilong_mem_space.py:339-353`). Slots are just strided-token init of the prompt
(`slot_init: strided_token`), carrying no useful retrieval — yet the always-1 softmax
still injects full-magnitude V at 32 layers. Memory should be irrelevant here, so the
catastrophic 0k drop (qa1 55→11, qa2 28→0, qa5 53→3) isolates the corruption to the read
path itself, not to retrieval quality.

## 3. Why P7P9 (KV-prepend) does NOT corrupt
Prepend puts slots into the **same live-token softmax**. Cold/irrelevant slots get ~0
attention mass and self-normalize away — the live tokens absorb the probability. The
dedicated xattn removed exactly that escape valve.

## 4. Recommended minimal fix
**(a) learnable null/sink slot.** In `MemoryCrossAttentionRead`, concat one learnable
key/value (a `nn.Parameter`, value init 0) to `K`/`V` before the softmax
(`selector.py:~1389`). The softmax can then route mass to the null slot ("read nothing"),
collapsing the contribution when no slot is relevant — exactly what prepend gets for free.
Keeps gradient flowing (unlike P2's dead zero-init out_proj), stays behind
`use_memory_xattn`, ~5 lines. Cheaper stopgap: set `zero_alpha_on_cold_start: true` (only
masks cold step-0, decode steps 1+ still inject — partial only).

## 5. Confidence
**High** on root cause (mechanism + 0k tell + P7P9 contrast are mutually consistent).
**Medium-high** on fix (a); recommend a short retrain or a no-train logit probe to confirm
the null slot absorbs mass at 0k before committing.
