# v_route_aux_supervision — Routing-supervision aux loss (route_aux)

**Date**: 2026-06-04
**Files**: `scripts/train_mem_space_dolmino_cpt.py` (only)
**CLI**: `--route_aux_weight <float>` (default 0.0 = disabled, fully backward-compatible)
**Toy reference**: `scripts/toy_memory_bootstrap.py:358-464` (E2 arm), validated effective.

## Motivation

In real 8B Dolmino CPT, the mem_space selector (Q_sel/K_sel routing to top-k
slots) collapses to **uniform**: top1_sim ≈ 0.013 ≈ 1/128, retrieval_exact_acc = 0,
BABILong ≥2k all 0.0%.

The single-GPU toy diagnostic matrix gave a decisive conclusion:
- **E1 (decoupled-read starves routing grad)**: `mask_h_to_l1` decoupled-read
  attenuates the selector's LM gradient ~10–50× (lm_grad_contrib to Q_sel drops
  from 8–15 → 0.3–4). The selector then sees almost only aux gradient.
- **E2 (the fix lever)**: pure LM loss **cannot** bootstrap content addressing
  (aux_off: retrieval_exact_acc stays 0). Adding a **routing-supervision aux
  loss** lifts retrieval_exact_acc to 0.25 and still rising (aux_on).
- **E4**: the frozen inject gate is NOT the primary cause.

This version ports the toy's E2 `route_aux` into the real 8B training path.

## Architecture (pseudo-code)

The mem_space selector returns `(idx, scores, ste_weights)` where `scores`
[B, N] is the grad-bearing softmax over slots. `MemorySpaceLayer.forward`
**detaches** `last_scores`/`last_idx` after the forward, so the only way to get
a grad-bearing score copy is a forward hook on the selector module (identical to
the toy).

```
# Cross-chunk write -> read routing supervision.
# (dolmino_train_step_tbptt: multi-chunk TBPTT; route_aux uses adjacent chunks.)

prev_idx1 = None
for i, chunk in enumerate(all_chunks):                # ctx chunks ... + target
    # install fwd hook on mem_layers[0].selector to capture grad-bearing scores
    captured, handle = install_score_hook(mem_layers)
    out = model(chunk, labels=chunk)                  # writes + reads memory
    handle.remove()

    chunk_lm  = out.loss / scale
    chunk_aux = collect_aux_loss() / scale
    step_loss = chunk_lm + chunk_aux

    # route_aux: supervise THIS chunk's routing scores to land on the slots
    # the PREVIOUS chunk wrote into (detached idx as a CE target).
    if route_aux_weight > 0 and prev_idx1 is not None:
        scores2 = captured["scores"]                  # [B, N], softmax, grad
        sel_p   = scores2.gather(1, prev_idx1)        # [B, k]
        route_aux = -(sel_p.clamp(min=1e-9).log().mean())   # CE, uniform target
        if isfinite(route_aux):
            step_loss += route_aux_weight * (route_aux / scale)

    window_loss += step_loss
    prev_idx1 = mem_layers[0].last_idx.detach()       # target for NEXT chunk

    if at_window_edge or is_last:
        window_loss.backward(); detach_banks()
```

`route_aux = -mean_b mean_{j in idx1} log scores2[b, j]` — a cross-entropy that
pushes the current chunk's routing distribution to put probability mass on the
slots a prior chunk wrote into. Identical formula to toy E2
(`scripts/toy_memory_bootstrap.py:431-432`).

### Real-path vs toy difference (important)
- Toy: fixed 2-chunk (chunk-1 writes fact, chunk-2 reads), `idx1` from chunk-1.
- Real `dolmino_train_step_tbptt`: **multi-chunk TBPTT**. We generalize to the
  **adjacent-chunk** scheme: chunk `i`'s `last_idx` supervises chunk `i+1`'s
  scores (write→read across every chunk boundary), which subsumes the toy's
  "chunk i supervises chunk i+1" minimal requirement.
- `babilong_train_step`: streams context (no_grad) then grads the last chunk.
  route_aux supervises the grad-bearing last chunk's scores against the slots
  the **last context chunk** wrote (single write→read pair). Single-chunk
  samples contribute route_aux = 0 (no prior write to supervise).

### Numerical safety
- `scores.clamp(min=1e-9).log()` (avoids log(0)).
- `route_aux` skipped (treated as 0 for that chunk) when non-finite, or when
  captured scores are missing / shape-mismatched vs `idx1`.
- The route_aux term is divided by the same `scale = n_chunks * grad_accum` as
  lm/aux so its magnitude is comparable across curriculum context lengths.

### Backward compatibility
`route_aux_weight == 0.0` → `use_route_aux = False` → **no hook installed, no
extra forward overhead, route_aux returned as a zero scalar**. The dolmino/
babilong/main-loop paths are byte-for-byte equivalent to before. Verified.

## Initialization / recommended weight

- **`--route_aux_weight 1.0`** (the value the toy E2 used and validated). Start
  here for the E5 8B verification run.
- The toy applied route_aux at full weight (1.0) alongside lm + aux and saw
  retrieval_exact_acc climb 0 → 0.25+. If route_aux dominates / destabilizes lm
  on the 8B path, sweep `{0.3, 1.0, 3.0}`; if too weak (top1_sim stays ≈1/N),
  raise toward 3.0.

## Relationship to prior work

- **Toy E2 (this repo)**: direct port; same CE formula, same hook mechanism.
- **MemoryLLM / Block-Recurrent / Infini-attention**: those learn memory
  read/write end-to-end purely from LM loss. Our diagnosis (E2 aux_off) shows
  that under the decoupled-read path (v15) LM loss alone cannot bootstrap
  content addressing; route_aux is an explicit teacher signal for *where* to
  route, complementary to the LM objective rather than a replacement.
- Conceptually similar to **load-balance / router auxiliary losses in MoE**
  (Switch Transformer), but here the target is not balanced utilization — it is
  **content-addressed retrieval consistency**: read where you just wrote.

## Known issues / caveats

1. **Hook on `mem_layers[0]` only.** Like the toy, route_aux supervises just
   the layer-0 selector. If routing collapse is layer-specific, may need to
   extend to all mem layers (sum the CE). Layer-0 was sufficient in the toy.
2. **Adjacent-chunk target is a heuristic.** It assumes the slots written by
   chunk i are the *right* ones for chunk i+1 to attend to. For Dolmino plain
   text (no explicit fact→query structure) this is a soft consistency prior,
   not ground-truth retrieval supervision (unlike toy's synthetic fact/query).
   It should still break the uniform-collapse symmetry; monitor `train/route_aux`
   trending down and `memory/top1_sim` rising above 1/num_slots.
3. **`last_idx` is the hard top-k of the prior chunk's own (possibly collapsed)
   routing.** Early in training when routing is uniform, the target is noisy;
   the signal is self-reinforcing once any structure emerges. Toy showed this
   bootstraps fine from cold start.
4. Single-chunk curriculum stages (n_ctx with only target chunk) get no
   cross-chunk pair → route_aux = 0 for those steps (expected; harmless).
5. babilong route_aux only covers the multi-chunk case; single-chunk babilong
   samples contribute 0 (no prior write).
