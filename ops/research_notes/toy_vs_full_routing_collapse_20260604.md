# Toy vs Full 8B routing collapse — root-cause analysis
date: 2026-06-04 | author: general-purpose-1 (researcher) | mode: research-only

Compares `scripts/toy_memory_bootstrap.py` (uses `train_mem_space_dolmino_cpt.T.build_model`)
vs `scripts/train_mem_space_dolmino_cpt.py` to explain why the SAME
`use_decoupled_read` routing learns in toy (top1_sim→0.998) but collapses to
uniform (top1_sim≈0.013) on the full 8B Dolmino run.

Logs read: `logs/p2_decoupled_on.log`, `logs/p2_decoupled_off.log`,
`logs/dolmino_p2_decoupled_local.log`. Code read: both train scripts,
`src/memory/mem_space/layer.py`, `selector.py`.

---

## TL;DR — the premise is partly an illusion, and there is a real gradient bug

1. **`top1_sim` is a red-herring metric. The toy did NOT actually learn the
   task.** The toy's GOLD metric `retrieval_exact_acc` is **0.000 for the entire
   run in BOTH arms** (on and off), and `chunk1to2_overlap` *drops* 0.54→0.26.
   The "0.998" is single-slot collapse (every query maps to ONE slot regardless
   of content = the opposite of content addressing), not learned routing.
   So the framing "toy learns, full collapses" is false: **toy collapses to a
   single slot, full collapses to uniform — both are failures, just different
   degenerate fixed points.** (very_high)

2. **The decisive mechanical bug for the full run: `use_decoupled_read` masks
   off the H→L1 prepend attention (`mask_h_to_l1=True`, layer.py:980/991), which
   is the ONLY direct LM-loss→`scores` gradient path. The decoupled read module
   reads ALL slots through its own softmax and never touches `idx`/`scores`
   (layer.py:1116-1126). Result: the selector (`Q_sel`/`K_sel`) is trained
   almost entirely by `load_balance` + `entropy` aux losses, which both
   *explicitly push toward uniform* → `top1_sim`→1/128.** (high)

3. **Even with the routing path intact (toy off-arm, prepend active,
   top1_sim→0.86) the gold acc is still 0.** So on top of (2), the LM/NTP
   objective on generic text — and even the toy's retrieval-rewarding loss —
   does not produce a usable content-addressing gradient. The loop cannot
   bootstrap content addressing from LM loss alone. (high)

---

## Evidence

### Toy gold metric never moves (both arms)
```
on-arm  step=0   exact_acc=0.000 tok_acc=0.417 top1_sim=0.332 overlap=0.543
on-arm  step=780 exact_acc=0.000 tok_acc=0.333 top1_sim=0.230 overlap=0.258
off-arm step=0   exact_acc=0.000 ... top1_sim=0.346 overlap=0.504
off-arm step=799 exact_acc=0.000 ... top1_sim=0.350 overlap=0.184
max exact_acc (on)=0.000  max exact_acc (off)=0.000  max tok_acc≈0.44 (≈chance)
```
`retrieval_exact_acc=0.000` throughout, `overlap` decreasing → the chunk-2
reader is NOT routing back to chunk-1's written slots. The QUERY_DIAG
`top1_sim_mean→0.998` (on-arm) is measured on the *training* chunk and reflects
softmax peakedness, not retrieval success. The two metrics disagree because
high `top1_sim` here = "one champion slot wins for every input" (content-blind),
exactly the degeneracy the slot_query mode was meant to escape.

### Full run: actively collapses from non-uniform → uniform
```
step=4    top1_sim=0.154
step=644  top1_sim=0.984   (transient spike)
step=756  top1_sim=1.000   (transient spike — also single-slot collapse)
step=865  top1_sim=0.031   (collapses back)
step=1996 top1_sim=0.013 ≈ 1/128 (uniform)
```
The run oscillates between single-slot spikes and uniform, with no stable
content-addressing regime, finally settling at uniform. `slot_attn_entropy`
drifts 3.5→1.7 (slots smear attention over the 1024 tokens). This is the
signature of a selector driven by aux losses + noise, not by task signal.

### inject_gate frozen ⇒ memory barely reaches the logits (compounding)
```
WRITEBACK_DIAG  alpha(inject_gate_mean)=0.1213 (start) → 0.1217 (end)
                inject_gate_std=0.0071 → 0.0076   (essentially constant)
```
`g=sigmoid(inject_gate(h))` is stuck ≈0.12 and content-*independent* (std≈0.007)
for all 2000 steps. With `decoupled_read.out_proj` zero-init (selector.py:1038),
the memory read contribution `g * decoupled_read_out` starts at 0 and the only
lever to grow it (alpha) never opens. So `d(LM loss)/d(memory)` ≈ 0 → no signal
to fix routing OR reads. (This matches the prior 2026-06-03 report's "injection
dilution"; P2 removed the prepend dilution but did NOT make alpha learnable in
practice — it stays flat.)

### Hyperparameter deltas toy vs full (from launch scripts)
| knob | toy | full | note |
|---|---|---|---|
| selector_temperature | 40 | 40 | same |
| num_slots / top_k | 128 / 16 | 128 / 16 | same |
| lr | 1e-4 | 1e-4 | same |
| routing_pool_mode | slot_query | slot_query | same |
| use_decoupled_read | on / off | on | same mechanism |
| **key_repulsion_weight** | **0.05** (default) | **1.0** | **20× stronger in full** |
| chunk T (tokens) | ~8 | 1024 | 128× longer |
| batch_size | 8 | 1 (×8 ranks) | — |
| curriculum n_ctx | n/a (fixed 2-chunk) | 1→8 | full streams many chunks, detached |
The structural knobs match; the routing mechanism is identical. The two big
differences are **T (8 vs 1024)** and **key_repulsion (0.05 vs 1.0)**, plus the
data objective. Long T + strong key-repulsion + uniform-pushing aux all favor
the uniform fixed point in the full run; tiny repetitive T favors the
single-slot fixed point in toy.

---

## Root causes, ranked

1. **`top1_sim` is the wrong success metric; the real signal (retrieval
   acc / chunk-overlap) shows the loop NEVER bootstraps content addressing in
   either setting.** Stop tuning toward `top1_sim`. — confidence: **very_high**

2. **`use_decoupled_read` severs the selector's LM-loss gradient.**
   `mask_h_to_l1=True` removes H→L1 attention (layer.py:980/991), so the
   gradient-bearing soft-proxy `M_sel_hidden_soft` (layer.py:890-900, the only
   differentiable LM path into `scores`) no longer influences the output. The
   decoupled read (layer.py:1116-1126) uses ALL slots via its own softmax and
   never consumes `idx`/`scores`. Net: selector is trained by `load_balance`
   (w=0.01) + `entropy` (w=0.001) + `key_repulsion` (w=1.0), the first two of
   which *minimize* routing concentration → drives `top1_sim`→uniform. This is
   the mechanical reason the full run collapses to exactly 1/128. — **high**

3. **The NTP/LM objective provides ~no retrieval reward, even in the toy where
   loss is masked to the answer tokens.** Toy off-arm keeps the routing
   gradient path alive (top1_sim→0.86) yet gold acc stays 0 — the model finds a
   lower-loss solution that does not require content-correct retrieval. On
   generic Dolmino web text the reward is even weaker. → A direct routing/recall
   supervision aux loss is needed; LM loss alone will not get there. — **high**

4. **inject_gate is frozen at α≈0.12, content-independent**, and decoupled
   `out_proj` is zero-init. Memory's influence on logits is ≈0–0.2%, so even a
   perfect router would get a near-zero LM gradient. Compounds (2)/(3). The v13
   fix added `inject_gate` to the optimizer (train script:371-380) but it still
   does not move — likely because there is no gradient telling it to open (chicken
   /egg). — **medium-high**

5. **temp=40 over 128 slots + key_repulsion=1.0 (20× toy).** Sharp softmax can
   lock onto a champion; strong key-repulsion smears keys. Secondary: the full
   run's endpoint is *uniform* (not over-sharp), so temp is not the primary
   driver, but temp=40 amplifies the single-slot spikes seen mid-run. — **medium**

6. `summary_q_max_cos=0.0` / `S_max_cos=0.0` in both logs is a **diagnostic
   artifact**, not a real zero: those fields are only written in `multi_query`
   mode (selector.py:297-315); `slot_query` leaves them at the 0.0 default. Not
   a bug, ignore. — **high**

---

## Minimal experiments (cheapest → most expensive)

- **E1 — grad-probe (single GPU, ~5 min, decisive for cause #2).** In the toy,
  after `total.backward()`, print `selector.Q_sel.weight.grad.norm()` and
  `K_sel.weight.grad.norm()` for the on-arm (decoupled) vs off-arm (prepend).
  Prediction: on-arm routing grad ≈ pure-aux magnitude and uniform-pushing;
  off-arm has a real task component. Confirms decoupled-read starves routing.

- **E2 — direct routing aux in toy (single GPU, ~15 min, decisive for #3).**
  Add an auxiliary loss that supervises chunk-2 to route to chunk-1's written
  slot indices (`idx1`), e.g. cross-entropy of `scores` toward `idx1`. If
  `retrieval_exact_acc` finally climbs, it proves the loop CAN content-address
  but the LM loss alone never rewards it → adopt a recall-supervision aux for
  the full run. Run with decoupled OFF and ON to see if the aux also rescues the
  decoupled path.

- **E3 — toy with decoupled OFF + key_repulsion 0.05→0 + entropy/​load_balance →
  0 (single GPU).** Removes the uniform-pushing aux entirely; check whether gold
  acc moves. Isolates aux-induced collapse from objective weakness.

- **E4 — `--force_gate_alpha 0.5 --force_gate_steps 400` toy probe (already
  implemented, not yet run in these logs).** Forces α=0.5 so memory strongly
  influences logits; tests cause #4. Cheap, run it.

- **E5 (expensive, 8B) — only after E1/E2 decide.** Full run with a recall aux
  loss OR with decoupled-read OFF (revert to prepend) + temp 40→5. Do NOT spend
  an 8-GPU 12h run before E1/E2 narrow it down.

---

## confidence: high fix recommendation (for /coder, after E1/E2 confirm)

The single most likely fix, targeting cause #2+#3 with minimal change:

- **Add a routing-supervision aux loss** so `scores` receive a direct gradient
  that survives `mask_h_to_l1`. Concretely, in `src/memory/mem_space/layer.py`
  around the selector call (after layer.py:766-769), when a "write target" is
  known, add `aux['route'] = CE(scores, write_idx)` and collect it in
  `_collect_aux_loss` (train script:516). This bypasses the severed LM path.
- **OR**, if keeping decoupled-read pure, route the decoupled read's K/V through
  the top-k `idx` (selected slots only) instead of ALL slots, so `scores`
  re-enter the read output and regain an LM gradient (edit layer.py:1118-1125 to
  gather `read_slots` by `idx` and weight by `ste_weights`).

I do NOT recommend pure hyperparameter changes (temp, key_repulsion) as the
primary fix — they address the secondary cause #5 only. Run E1/E2 first; both
are single-GPU and decisive.
