# Late-Chunk Memory Collapse — Root-Cause Analysis (P11/F2 delta-rule + null-sink)

**Date:** 2026-06-10
**Author:** general-purpose-44 (research, no code changes)
**Scope:** P11/F2 series (chunk512/1024, 128 slots, top_k16, delta-rule writeback + MemoryCrossAttentionRead + null-sink). Pure log/code audit. No training, no source edits.
**Update 2026-06-10:** added §5 "The L3 summary channel's late-sequence behaviour" (L3-vs-L1 collapse, same-root question, L3 value/ablation).

---

## TL;DR

The reported "two-stage progressive collapse" (late `fwd≥250`: retrieved_norm→0.44, top1_sim→0.99, slot_delta frozen) is **largely a misreading of the diagnostic and a rare transient artifact**, NOT a universal late-chunk degradation:

1. **`fwd` is a GLOBAL cumulative forward counter across the whole eval run, not a within-sample chunk index.** (`layer.py:833` `_fwd_count += 1`; emitted every 50 calls `layer.py:929`). A single 32k sample produces ~63 chunks; `fwd` runs to 8000–9000 across the 30-sample eval. So "fwd≥250" is NOT "late within one sample" — it spans dozens of independent samples. The original probe's "early vs late" framing does not hold.
2. **The collapse signature is RARE and TRANSIENT, not progressive.** Across **5,947** QUERY_DIAG emissions in all available F2/P11 eval logs: only **38 (0.6%)** show `retrieved_norm<1`, and **35 (0.6%)** show `top1_sim>0.5`. Every single collapsed point is an **isolated spike that recovers at the very next emission** (e.g. f2_c512_8k: fwd1250 ret=0.44/top1=1.0 → fwd1300 ret=4.98/top1=0.18). There is no monotone freeze.
3. **It is confined to the F2 delta-rule runs.** The P8 chunk1024 (non-delta) eval and all P11 SWA/temp20 evals show **0% collapse** (0/3000+ points). So whatever it is, it is delta-rule/F2-config specific, and even there it is sub-1%.

The chronic, *real* issue these runs share is **routing concentration**, not collapse: usage_cov≈0.29, chunk_idx_jaccard≈0.91 → ~91/128 slots are effectively dead and the same ~37-slot set is re-picked every chunk. This is the load-balance problem, and it is **independent of sequence length / chunk depth** (flat across 0k→32k).

⚠️ **Confidence on phenomenon refutation: high** (5.9k-point statistics across many runs). **Confidence on the transient-spike mechanism: medium** (consistent with sample-boundary re-init, but no per-sample instrumentation to prove it directly — see "What to instrument").

---

## 1. Phenomenon Confirmation (data, not single point)

### 1.1 Diagnostic semantics (code-verified)
- `top1_sim_mean` = `scores.max(dim=-1)` — the **max softmax routing weight**, i.e. how peaked the selection softmax is (`layer.py:940-941`). With `selector_temperature=40` (F2 config, `launch_f2_longdoc_chunk512_diskA.sh:65`), softmax is sharp, so a momentary key-degeneracy makes top1→1.
- `retrieved_norm_mean` = mean L2 norm of the *currently-selected slot vectors* (`layer.py:948-950`). Right after a fresh sample's `init_from_hidden` (mean-pool + tiny noise, `memory_bank.py:145`), slots are near-identical and near-pooled-mean → small norm + top1≈1; once the first chunk writes content, norm jumps back to ~4.9.
- `slot_delta_abs_mean` = mean |slot_new − slot_old| over the selected slots that chunk.
- `fwd` = global cumulative counter (`layer.py:833`); QUERY_DIAG every 50, WRITEBACK_DIAG every 50.

### 1.2 Incidence (all F2/P11 eval logs with QUERY_DIAG)
```
TOTAL QUERY_DIAG = 5947
  retrieved_norm < 1.0 : 38   (0.6%)
  top1_sim    > 0.5    : 35   (0.6%)
```
| run (32k unless noted) | n | ret_norm<1 | top1>0.5 |
|---|---|---|---|
| f2_c512_step5000_4k/8k/16k/32k | 133/187/189/172 | 6/5/6/2 | 6/5/4/2 |
| f2_c512_step500_4k/8k/16k/32k  | 52/98/187/195  | 7/3/6/0 | 3/3/4/5 |
| chunk1024_temp20_full (P8) ALL lengths | ~1660 | **0** | **0** |
| eval_chunk1024_rh20 ALL lengths | ~760 | **0** | **0** |
| eval_p11_* (final/step4500/temp20/ddp/500step) ALL | ~2200 | **0** | **0** |

### 1.3 Transient, not progressive (f2_c512_step5000_8k, every collapse + neighbours)
```
fwd=1200 ret=4.97 top1=0.11      fwd=2500 ret=5.00 top1=0.11      fwd=7600 ret=4.85 top1=0.10
fwd=1250 ret=0.44 top1=1.00 <<<  fwd=2550 ret=0.41 top1=1.00 <<<  fwd=7650 ret=0.42 top1=0.99 <<<
fwd=1300 ret=4.98 top1=0.18      fwd=2600 ret=4.86 top1=0.15      fwd=7700 ret=4.86 top1=0.16
```
Each collapse is one emission wide and recovers immediately. Spacing (~1250, 1700, 2550, 3450, 7650) is irregular — consistent with hitting the diag stride (`fwd%50==0`) exactly on a fresh-sample re-init chunk, not with depth-into-sample.

### 1.4 Within-sample depth is fine
On the single longest contiguous 32k sample window (f2_c512_step5000_32k, fwd 50→1800), retrieved_norm stays 4.6–5.0, top1 0.09–0.23, slot_delta 0.006–0.010, g_in 0.502, g_forget 0.881 — **flat, no decay with depth**.

---

## 2. Mechanism Attribution (a) read-collapse / (b) write-saturation / (c) routing-concentration

### (a) Read collapse via null-sink — **REFUTED**
`XATTN_DIAG sink_mass` in the actual P11-chunk512-deltarule **training** log: **flat at 0.0077–0.0079 for all 1000 logged steps** (step5→5000), `gate_mean=0.40`, `attn_entropy=4.86`. The null-sink never absorbs more than ~0.8% of softmax mass. `retrieved_norm→0` at collapse points is NOT the read dumping mass to the sink (sink lives in the *read* attention `selector.py:1495-1536`; retrieved_norm is measured on the *selected slot vectors before projection*, `layer.py:948-950`). The two are unrelated. **Null-sink is not the driver.** (This also predicts D6 `disable_null_sink` will move the long-range cell little — see §4.)

### (b) Write saturation — **PARTIAL / not gate-closure**
At the 5 collapse points: `slot_delta_abs_mean≈4.2e-5` (vs run mean 0.0072) BUT `g_in_mean=0.50` (gate **wide open**, run range 0.501–0.504) and `g_forget_mean=0.88` (flat). So the writeback freeze at those points is **NOT g_in→0 gating-off**; it is the delta-rule fixed point `slot_new = current + g_in·(new_content − current)` with `new_content ≈ current` → delta→0 even with g_in=0.5 (`memory_bank.py:304`). This happens precisely when slots were just re-initialised from the pooled hidden and the incoming chunk content matches → momentary new≈current. It self-clears next chunk. Over the bulk of the run slot_delta is healthy (~0.007, max 0.012), so **there is no chronic EMA freeze**. The EMA/leaky-integrator form is, however, a *latent* saturation risk if g_in ever trends down — worth a guard, low priority.

### (c) Routing concentration — **CONFIRMED, the real chronic issue**
- `usage_cov ≈ 0.289` (min 0.20, max 0.375) over a 50-chunk window → only ~37/128 slots are ever selected; **~71% dead per window, ≥~91/128 dead globally** given the high jaccard.
- `chunk_idx_jaccard ≈ 0.906–0.915` → every chunk routes to almost the **same** ~16-slot set. Content addressing is not happening; routing is near-static.
- `usage_ent ≈ 0.64` (1.0=uniform) and `uniq_sel_slots` pinned at 16 (=top_k, the floor — B=1 so it cannot exceed top_k per chunk).
- This is **flat across 0k→32k** (not length-dependent) and present in P8 too (top1≈0.014, uniform-ish there but with its own collapse mode). It is the load-balance failure that has dogged this project (cf. PENDING_TASKS "all P1 routing-pool variants collapse to 1-2% noise floor"; P2 decoupled top1≈0.013).

**Verdict:** the "late-chunk collapse" is a **diagnostic artifact (cumulative fwd) + rare sample-boundary transient (delta-rule fixed point at re-init)**. The genuine, actionable pathology is **chronic routing concentration → effective capacity ≈ top_k·(#live sets) ≪ 128**, mechanism (c).

---

## 3. Add num_slots vs. fix loss/mechanism — **fix routing, do NOT add slots**

**Conclusion: adding num_slots (128→256/512) is the WRONG lever and will mostly add dead slots. confidence: high.**

Evidence: usage_cov is only 0.29 and jaccard 0.91 at 128 slots — the model already cannot use the slots it has. Effective utilised capacity is ~37 slots and the *same* set is reused every chunk. Doubling N to 256 leaves the live-set logic unchanged (top_k still 16, same content-router, same near-static routing) → live fraction halves to ~0.14, dead slots ~85%. You pay 2× slot memory/compute for ~0 extra *used* capacity. The bottleneck is **routing diversity / load balance**, not slot count.

Capacity should be added only **after** routing spreads load (usage_cov ≳ 0.6 and jaccard ≲ 0.4). The right knobs now are on the **loss/mechanism side** (and the top_k capacity knob, which is cheaper than N).

Caveat: D1 `slot_dim 16384` (vector *dimension*, not count) is a different axis (per-slot expressivity) and is orthogonal — but note that run CRASHED (RUN_REGISTRY: wbmode_lowrank rank3 exit1, no ckpt), so it currently provides no signal.

---

## 4. Ranked Fix Experiments

All validated on the **same-protocol BABILong long-range cell** (n=100, qa1/qa2/qa5 × 0k–32k, babilong.metrics) + the QUERY_DIAG telemetry (target: usage_cov↑, jaccard↓, top1 mid-range, retrieved_norm stable). Tie each to RUN_REGISTRY.

### FIX-1 (PRIMARY) — Lower selector_temperature 40→~10 + verify loss-free-balance is actually live
- **What:** `--selector_temperature 10` (currently 40, `launch_f2_longdoc_chunk512_diskA.sh:65`). Keep `--use_loss_free_balance` (already on) but **audit it is updating**: P7 bias only updates in `self.training` (`selector.py:471`); during *eval* the bias is frozen, and at train time with B=1 per rank the load signal is extremely sparse (`load = sel_one_hot.mean(dim=0)`, B=1 → 0/1 per slot), so `routing_bias` may be doing almost nothing. Raise `--loss_free_update_rate` 0.001→0.01 and/or accumulate load over the grad-accum window.
- **Mechanism:** temp=40 over-sharpens softmax → winner-take-all → same slots always win → jaccard→1, dead slots. Softer temp + a load bias that actually fires spreads selection.
- **Confidence:** medium-high. Temperature is the most direct cause of the 0.91 jaccard; the loss-free-balance audit is high-value because it may currently be a no-op.
- **Validate:** usage_cov 0.29→target >0.5, jaccard 0.91→<0.6; long-range qa2/qa5 16k–32k vs current F2.

### FIX-2 — Turn ON entropy_aux (currently `--entropy_aux_weight 0.0`)
- **What:** `--entropy_aux_weight 0.01–0.05` (and `--load_balance_weight` >0; both are 0.0 now, line 65). Logic already implemented (`selector.py:574-658`).
- **Mechanism:** explicit per-batch load-entropy / load-balance penalty pushes selection mass off the dominant set → directly attacks usage_cov/jaccard. This is the Switch-Transformer (Fedus 2021) lever the code cites but leaves disabled.
- **Confidence:** medium-high (well-understood; risk = over-regularising routing into uniform, which hurts content addressing — sweep the weight).
- **Validate:** same cell; watch that top1_sim doesn't crash to 1/N (=0.0078) uniform.

### FIX-3 — top_k length-ladder / raise top_k (capacity via selection width, not N)
- **What:** raise `--top_k` 16→32 (or schedule with seq length). **Ladder ckpts already exist** (PENDING "ladder top_k 阶梯" / progressive_chunk v2 on .249 stage chain c128→c1024).
- **Mechanism:** more slots read+written per chunk → more of the bank stays warm, reduces dead-slot fraction without touching N. Cheaper and better-targeted than doubling N.
- **Confidence:** medium. This is the capacity knob that *is* on-mechanism (it widens the live set), unlike N. The in-flight ladder run is the natural validator — **compare its usage_cov/jaccard at matched steps vs F2 top_k16**.
- **Validate:** the ladder eval already in the pipeline; add usage_cov/jaccard read-off.

### FIX-4 — ST-Gumbel top-k ON during training (exploration)
- **What:** `--use_st_gumbel_topk` with `st_gumbel_temperature≈1.0` (implemented `selector.py:451-461`, default OFF). Train-only, gradient-free index perturbation.
- **Mechanism:** stochastic top-k breaks the static winner set early in training so under-used slots get gradient → learns diverse keys → lower jaccard at convergence.
- **Confidence:** medium (exploration helps cold-start routing; must be OFF at eval, which it already is).
- **Validate:** usage_cov/jaccard trajectory during training; final long-range cell.

### FIX-5 (DIAGNOSTIC, LOW) — D6 disable_null_sink as a control, not a fix
- **What:** the D6 `nullsink_off` arm already training on .249.
- **Expected:** given sink_mass≈0.008 flat, removing the sink should change the long-range cell **little**. If it changes a lot, my refutation of mechanism (a) is wrong → re-open.
- **Confidence (of the prediction):** medium-high. Use it to *falsify*, not to fix.

**Recommended order:** FIX-1 (temp + loss-free audit) → FIX-2 (entropy_aux) → FIX-3 (top_k, leverage in-flight ladder). FIX-4 as a combinable add-on. FIX-5 as the control. **Do NOT add num_slots until usage_cov>0.6.**

---

## 5. The L3 summary channel's late-sequence behaviour

**TL;DR for L3:** the `summary_q_*_cos=0.0000` fields are **inert defaults, not L3 collapse** — they only ever fire in `routing_pool_mode="multi_query"`, and every P8/P11/F2 run uses `slot_query`, so L3 telemetry is structurally absent from those logs. With the diagnostics we *do* have, **L3 cannot be shown to collapse late** the way L1's transient does; and on the architecture, **if a late read-side failure exists it would hit L3 and L1 jointly** (shared joint-attention softmax), making the read path the more likely common locus than either channel's write side. But L3 is **under-instrumented to the point of being unfalsifiable** and has **never been cleanly ablated** — so its current contribution is unproven in both directions. confidence: medium (code-grounded), low on the empirical "is L3 helping" question (no clean ablation exists).

### 5.1 Why `summary_q_max_cos` / `summary_q_mean_cos` / `S_max_cos` are all 0.0000 (it is NOT L3 degradation)
These three fields are written **only inside the `multi_query` branch** of the selector (`selector.py:337-355`: `_last_summary_query_max_cos`, `_last_summary_query_mean_cos`, `_last_S_max_cos`). They default to 0.0 (`selector.py:173-181`) and the layer reads them via `getattr(..., 0.0)` (`layer.py:962-968`). **Every P8/P11/F2 launch script sets `--routing_pool_mode slot_query`** (verified across `launch_mem_space_p8*`, `launch_f2_longdoc_chunk512_diskA.sh:74`, `launch_mem_space_p11_chunk512_remote196.sh:39`, etc.). In `slot_query` mode the selector inverts query/key roles — slots ARE the queries, the chunk's H tokens are keys (`selector.py:362-384`) — and the summary-query diagnostics are simply never assigned. So across **32,390** diag emissions, `summary_q_max_cos=0.0000` and `S_max_cos=0.0000` in **100%** of cases. This is a "feature-off → field-stays-default" artifact, the L3-channel analogue of the `fwd`-is-global misreading in §1. **It says nothing about whether L3 is healthy.** (The `slot_attn_entropy` field, by contrast, *is* live under slot_query — it ranges 2.7–3.4, see §1-data — but that measures the slot→token routing attention, i.e. the L1 router, not L3.)

### 5.2 L3 is NOT "recomputed fresh / non-accumulating" — it is **recursive** (corrects the briefing premise)
The premise "L3 is recomputed every chunk, therefore can't saturate like L1" is **only half true**. `L3SummaryPool.forward` takes `prev_summary` and, when present, **uses last chunk's output as this chunk's initial queries instead of the learnable bank** (`l3_summary.py:124-148`: `if prev_summary is not None: S = prev_summary`). The patch hook feeds this chain: after each chunk it stashes `pool._prev_summary = cached_summary.detach()` and `pool._prev_chunk_h = h.detach()` (`patch.py:249-256`), and the next chunk's layer-0 forward computes `l3_summaries = self.l3_pool(prev_h, prev_summary=prev_summary, ...)` (`layer.py:812-823`). So L3 **does** carry cross-chunk state (a recurrent Q-Former), and is therefore *not* immune-by-construction to a late drift/saturation — it has its own recurrence that could degenerate. However: (a) the recurrence is `.detach()`-broken every chunk (no BPTT through the L3 chain), and (b) the LN_out at the pool output (`l3_summary.py:153`) re-normalises each chunk, which bounds output magnitude and makes an L1-style `retrieved_norm→0` magnitude-collapse unlikely for L3. The plausible L3 failure mode is **query/output redundancy** (the 64 summary tokens collapsing to near-identical directions — exactly what `summary_q_*_cos` was built to catch and what the v9 orthogonal-init + `query_diversity_loss` were added to fight, `l3_summary.py:96-105,156-194`), **not** norm collapse. And we have **zero telemetry on it** in slot_query runs.

### 5.3 How H actually reads L3 in P8/F2/P11 — shared softmax with H-tokens, no separate L3 read module
Architecture (`layer.py` extended-sequence `[L3(k_l3) | L1(k) | H(T)]`, mask builder `:108-161`):
- With `use_memory_xattn=True` (P8/F2/P11 all set `--use_memory_xattn --memory_xattn_gate_init 0.4`), `mask_h_to_l1=True` (`layer.py:1195`) blocks the H→**L1** column only: `mask[prefix:, k_l3:k_l3+k] = neg_inf` (`layer.py:160-161`). **It does NOT block the H→L3 column** (`cols 0..k_l3-1` stay "always allowed", `layer.py:122`). L1's read is rerouted to the dedicated `MemoryCrossAttentionRead` module (its own softmax + null-sink, `layer.py:1354-1361`).
- **Consequence:** L1 gets its OWN softmax (no live-token dilution); **L3 does NOT** — H tokens read the 64 L3 summary tokens inside the *same joint-attention softmax* that also covers the T causal H-keys. So L3's read mass competes directly with the (growing) live-token context. As the within-chunk H grows and especially across the long generation chunk, the L3 segment is a fixed 64 keys diluted against an increasingly peaked local-token distribution.
- L3 outputs from the joint attention are **discarded** (`layer.py:1281-1283`: "L3 outputs are spurious — discard"); only the externally-computed `l3_summaries` re-enter as KV. So L3 contributes to H purely through `slot_delta = ext_h[L1+k_slots:] - bypass_h` (`layer.py:1311`) — i.e. **the only path L3 reaches the hidden state is the joint-attention residual**, the same `slot_delta` the §2(b) analysis covers.

### 5.4 Same root or independent? — read-side dilution would be the shared locus, write-side is L1-only
- **L1's transient (§1-2)** is a **write-side** delta-rule fixed point at sample re-init (`slot_delta≈4e-5` while `g_in=0.5`). L3 has **no delta-rule write** — its "memory" is the detached recurrent query carry, refreshed and LN-normalised each chunk. So L1's specific transient mechanism **cannot occur in L3**; on the write side the two channels are **independent**, and there is no evidence L3 shows L1's spike (we have no L3 magnitude telemetry at all, but LN_out makes it structurally unlikely).
- **IF** a genuine late read failure existed (H tokens stop attending to memory at depth), it would manifest in the **joint-attention softmax** that L3 rides in — and since L3 shares that softmax with H-tokens (§5.3), a read-side dilution would degrade L3 and the (joint-attention component of) L1 **together**. This is the one place the two channels are **coupled**. But §2(a) already measured the closest available read-side signal — `MemoryCrossAttentionRead.sink_mass` flat at 0.0077 across all 1000 training steps, `attn_entropy=4.86` — and found **no read collapse on the L1 xattn path**. The joint-attention L3 mass is not separately measured, so I cannot positively confirm L3 read mass is healthy; I can only say the one measured read path is fine and there is no *evidence* of the late read collapse that would be needed to drag L3 down. **Net: read-vs-write — the write-side L1 transient is real-but-rare; a read-side collapse (the only thing that could be "shared" with L3) is unsupported by the available sink/entropy telemetry.** This *strengthens* §2's verdict that the actionable pathology is routing-concentration on the write/route side, not a read collapse — L3 gives no reason to re-weight toward "read collapse".

### 5.5 Is L3 actually contributing, or a lazy free-rider? — UNPROVEN, and not cleanly ablated
- **No clean L3 ablation exists.** The only `use_l3_summary=False` run, `l3iso_noL3_local` (`launch_l3iso_noL3_local.sh`), **also turned route_aux OFF** (it was the E5 route_aux-OFF arm) and was **KILLED** by researcher as an expected-collapse confound (PENDING_TASKS:165). Its training loss is erratic (lm bouncing 1.4–5.1, `logs/l3iso_noL3_local.log` step620-715) and it produced **no BABILong numbers**. So "does turning L3 off hurt BABILong" has **never been answered** with a controlled A/B.
- The L3-related experiments that *did* complete are about an **aux loss**, not the channel: the `l3_recon_token_weight` sweep (w0.3 + w1.0) is **REJECTED** (RUN_REGISTRY §3, PENDING_TASKS:66-83) — token-level reconstruction conflicts with retrieval. That tells us the *recon objective* is harmful; it says **nothing** about whether the L3 summary KV itself helps. `l3_diversity_weight` (v9) and `l_recon_weight` (v12) are **both 0.0 in every canonical run** (verified in all P8/P11/F2 scripts) → the v9/v10/v12 L3-diversity machinery is **dormant**; L3 currently runs as a **plain recurrent Q-Former with no diversity/recon regularisation**.
- **No diagnostic measures H→L3 attention mass.** There is no field anywhere (grep of `layer.py`/`selector.py`) that reports the fraction of joint-attention softmax mass H-tokens place on the `k_l3` columns. So "are H tokens actually using L3" is **not currently observable**. Given (a) L3 rides the diluted joint softmax against growing local context, (b) all L3 regularisers are off, and (c) its outputs are discarded except via `slot_delta`, the **prior** that L3 is a low-contribution / partially free-riding component is plausible — but **unproven**. confidence: low (no measurement either way).

### 5.6 L3 conclusions
1. **L1 and L3 need different fixes.** L1's issue is **routing concentration** (write/route side, §2c) → fixes FIX-1..4. L3 has no routing and no delta-write; its candidate failure is **summary-token redundancy + read dilution in the shared joint softmax**, which the FIX-1..4 routing levers do **not** touch. They are independent problems.
2. **L3 is not the driver of, and does not share a root with, L1's transient** — write mechanisms are disjoint. The only coupling is the shared joint-attention read, and the available read telemetry shows no collapse there.
3. **L3's value is currently unproven and should be settled with ONE clean ablation:** rerun the *canonical* P11/F2 config (route_aux ON, memory_xattn ON, everything identical) with only `--no_l3_summary` flipped, train to a matched step, and eval same-protocol BABILong (n=100, qa1/qa2/qa5 × 0k–32k). If BABILong is flat → L3 is a free-rider (drop it, save the 64-token × 32-layer joint-attention cost + the Q-Former params). If it drops → L3 earns its place and the next step is to (a) add an H→L3 attention-mass diagnostic and (b) turn on `l3_diversity_weight` to fight the redundancy the dormant v9/v10 code was built for. **This is the single highest-value L3 experiment and it does not yet exist.** confidence on the recommendation: high; confidence on the outcome: unknown by design.

---

## What to instrument (if a real per-sample collapse is suspected)

Current QUERY_DIAG cannot distinguish "depth into sample" because `fwd` is global. To *prove/refute* a true within-sample late-chunk freeze, add a **per-sample chunk index** (reset on `memory_bank.reset()`) to the diag line, and log `retrieved_norm`/`top1`/`slot_delta` against THAT, plus per-sample dead-slot count. Until then, "late-chunk collapse" is not measurable and the 0.6% transient spikes are fully explained by sample-boundary re-init. **Do not gate a fix on the collapse hypothesis; gate it on usage_cov/jaccard, which are unambiguous and bad.**
