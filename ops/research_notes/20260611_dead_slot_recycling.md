# L1 Dead-Slot Recycling — Breaking the Cold-Start "Rich-Get-Richer" Deadlock

**Date:** 2026-06-11
**Author:** general-purpose-59 (research, no code changes, no training started)
**Scope:** evaluate the user's "cold-start富者愈富 deadlock" hypothesis for the L1 memory bank and the three candidate recycling mechanisms (dead-slot reset / forced co-update / targeted usage aux). Code-grounded audit of `src/memory/mem_space/{memory_bank.py,selector.py,layer.py,config.py}` + canonical launch scripts. Builds on gp-44 (`20260610_late_chunk_memory_collapse_rootcause.md`) and gp-53 (`20260611_post_routeA_redirect.md`).

---

## TL;DR

1. **The deadlock hypothesis is mechanistically CORRECT and sharper than the briefing assumed.** The canonical run uses `--slot_init strided_token --slot_init_noise 0.0` (`launch_mem_space_p11_chunk512_remote196.sh:35`), so a dead (never-selected) slot does **not** sit at "pooled-mean+noise" — it holds a **frozen chunk-0 token snapshot forever**, because delta-rule writeback only touches the `idx` returned by top-k (`memory_bank.py:295-304`, write only on selected positions). Over a 32k=63-chunk sample, ~91/128 slots keep stale chunk-0 content while the same ~32-37 live slots are continually refreshed → their content becomes "real", their keys `K_sel(content)+slot_key_bias` (`selector.py:263-266`) become more selectable, so they keep winning. **The few slots that got content early monopolize all future content. That is the deadlock, and it is a write-side / content-staleness loop, not a routing-noise problem.**

2. **The user's key discriminator (write-side recycling ≫ arm4 read-side forced exploration) is SOUND.** arm4 (ST-Gumbel, `usage_cov→1.0`, eval top1_sim→0.11, qa5 25/17/11 bottom of pack) added noise to the **selection logits** → forced the query to read random slots → diluted the "few precise hits" pattern. Dead-slot recycling leaves the selector **greedy and noise-free**; it only changes the *content* of slots that were already not being selected. Live slots' keys/content/selection are untouched, so the softmax peak on the live set is preserved → **top1_sim should NOT collapse the way arm4 did**, *provided the recycled content is diverse (strided) not a single pooled-mean broadcast* (see §3). This is "先充实再竞争 (refill then compete by natural content match)" vs arm4's "强行命中 (force the hit)".

3. **Strongest literature anchor = VQ-VAE codebook "random restart".** Jukebox (Dhariwal 2020) and Łańcucki 2020 ("Robust Training of VQ Bottleneck") solve the *identical* dead-code problem by **resetting unused codebook vectors to random encoder outputs from the current batch, without touching the encoder or the active codes** — exactly "capacity recycling without disturbing active slots". This validates candidate-1's design and predicts it is safe.

4. **CRITICAL strategic caveat (do not skip):** gp-53 §3.1 showed L1-only (`--no_l3_summary`) gives qa5 1k=4 vs 89 → **L1's long-range ceiling is ~0; L3 carries essentially all long-range retrieval.** So even a perfect L1 usage_cov fix may move BABILong long-range *little*. **BUT** the user's reframing makes this a clean falsifiable question: *is L1's low ceiling INHERENT, or is it CAUSED by the deadlock (only ~32 slots ever hold content → cannot store 63-chunks-worth of facts)?* Dead-slot recycling is the experiment that answers it. If usage_cov↑ AND L1-sensitive long-range (qa1 single-fact 8k-32k) ↑ → ceiling was deadlock-induced (big win, re-opens L1). If usage_cov↑ but long-range flat → ceiling is inherent (L1 confirmed dead-end, stop). Either way it resolves an open question gp-53 left as an assumption.

**Verdict:** candidate-1 (dead-slot reset to fresh strided content) = PRIMARY, safest, VQ-VAE-precedented, needs coder. candidate-2 (forced co-update) = SECONDARY, homogenization risk. candidate-3 (targeted usage aux) = DEPRIORITIZE — it is the rejected arm1/arm4 family (the gradient-free version is *already implemented and already failed* as arm1).

---

## 1. Mechanism audit of the three candidates (against actual code)

### Candidate 1 — Periodic dead-slot reset
- **Where:** new logic in `layer.py` after the selector call (~`layer.py:874-908`, where `idx` and the layer-0 usage histogram already exist) + a per-`MemoryBank` cumulative usage counter that resets on `memory_bank.reset()` (`memory_bank.py:87`, called at sample boundary). Reset writes into `self.memory_bank.slots` directly (mirror `init_from_hidden`'s `strided_token` branch, `memory_bank.py:153-164`).
- **Implementation:** maintain `_sample_usage[N]` (long), `scatter_add_` the selected `idx[0]` every chunk (the layer-0 hist at `layer.py:904` is *window-reset* — need a separate sample-scoped counter). Every `R` chunks, find `dead = (_sample_usage==0)`, overwrite those slot rows with **strided tokens from the *current* chunk's hidden** (diverse content, on-manifold, writable), and leave them in the bank to compete. **Only when `not self.memory_bank.frozen`** (skip during frozen greedy generation, `memory_bank.py:81,241`).
- **How it stays gentle / preserves retrieval:** touches ONLY never-selected slots → live slots' content, keys, and top-k selection are byte-identical → the score softmax peak on the live set is unchanged → top1_sim preserved. The selector itself is **never modified** (no noise, no uniformity loss on the gradient/score path).
- **arm4-repeat risk:** LOW *if* recycled content is strided-diverse. The one way it could flatten top1_sim: resetting many slots to the **single pooled mean** of the current chunk (≈ the current query) would create many near-equal high-score slots → softmax flattens → top1↓. Using **distinct strided tokens** (as `slot_init=strided_token` already does) avoids minting N identical high-scorers. **Design rule: recycle with strided current-chunk tokens, never pooled-mean.** confidence the rule prevents collapse: medium-high.

### Candidate 2 — Forced co-update (write a few dead slots each chunk, small gate)
- **Where:** `layer.py` write path (~`layer.py:1531-1572`). Append `G` dead-slot indices to the write set with a scaled-down gate (`g_in * co_scale`, e.g. 0.1) — analogous to how `num_global_slots` already appends always-on indices (`layer.py:881-888`).
- **Content source problem (the hard part):** `O_mem_slot` is computed *per selected slot* from the read/attention; dead slots have no natural content vector. You must broadcast either the chunk summary or strided current content into them. **Broadcasting the same vector into G dead slots makes them near-identical → key collapse** — the exact failure `slot_key_bias` (`selector.py:148-150`) was added to fight. Mild `co_scale` slows but does not remove the homogenization.
- **Gentleness/risk:** also write-side (query untouched → top1 safe like candidate-1), BUT the homogenization risk is structural, and it adds a content-routing design choice. confidence: medium-low. Run only if candidate-1 shows usage_cov gains without top1 collapse.

### Candidate 3 — Usage counter + targeted aux router loss
- **Where:** `selector.py` already has the loss-free-balance **gradient-free bias** (`routing_bias`, `selector.py:431-480`) and the entropy/load-balance aux losses (`selector.py:574-658`).
- **Verdict: this is the rejected family.** The gradient-free DeepSeek bias version *is* arm1 (`loss_free_update_rate 0.001→0.01`, REJECTED, gp-53 §1) — and it is structurally global-balancing: `load = sel_one_hot.mean(dim=0)` with B=1/rank is a 0/1 signal that pushes *all* under-used slots up uniformly (`selector.py:477-480`), i.e. arm4-like dispersion. A gradient aux (entropy_aux) was arm2, which *crushed even 0k* (12 vs 74). "Targeted at dead slots not global-uniform" is a real distinction in principle, but every implementable form still applies dispersive pressure on the **selection** distribution = the side arm1-4 proved is the wrong lever. **DEPRIORITIZE.**

---

## 2. Why write-side recycling can avoid the arm4 top1_sim collapse (the core argument)

arm4's failure chain: noise on selection logits (train) → query reads many slots → model learns diffuse routing → at eval the softmax is no longer peaked (top1_sim 0.99→0.11 on the *selected-slot* read) and the "少数槽精确命中" pattern that actually carried the relevant fact is destroyed → long-range qa5 bottoms out.

Recycling breaks the chain at the source: **the selection rule is never perturbed.** Selection is, at every chunk, still "argmax-k of content-similarity scores" (`selector.py:464`), with no Gumbel noise (`use_st_gumbel_topk` stays off) and no uniformity loss on the score path. Therefore:
- The score softmax stays as peaked as the temperature (40) dictates → top1_sim stays in its current healthy 0.1-0.2 band, **structurally cannot be flattened by recycling** unless recycling itself manufactures many equal-scoring slots (avoided by strided-diverse content, §1 design rule).
- A recycled dead slot only wins selection if its *refreshed content genuinely matches a future query better* than a live slot — i.e. by the same content-similarity rule the model already trusts. That is "natural competition", not "forced命中".
- The two are coupled only through "which slots exist with real content", not "how the query chooses" — exactly the user's intuition, and it holds against the code.

**Honest residual risk:** recycling a slot that *would have* been correctly stale (e.g. a slot deliberately parked) could evict a useful live slot if `R` is too aggressive or `dead` is defined too loosely (window-zero vs sample-zero). Mitigated by sample-scoped zero-usage detection + conservative `R` (≥1/4 of the per-sample chunk count). confidence top1 stays intact: **medium-high**.

---

## 3. Literature anchors

- **VQ-VAE codebook "random restart" (Dhariwal Jukebox 2020; Łańcucki 2020 "Robust Training of VQ Bottleneck"):** the canonical "dead-code revival" — unused codebook entries are **reset to random encoder outputs from the current batch**, encoder + active codes untouched. This is *precisely* candidate-1 ("capacity recycling without disturbing active slots") and is the strongest precedent that it is both safe and effective. **Adopt its exact recipe (reset to current activations, not zero, not random-gaussian).**
- **DeepSeek aux-loss-free load balancing (arXiv:2408.15664):** the gradient-free per-expert bias — already implemented (`routing_bias`) and already tried as arm1 (REJECTED). Confirms the *bias* lever alone does not help here.
- **Switch Transformer (Fedus 2021) expert dropout / load-balance aux:** the entropy/load aux family = arm2/arm3 (REJECTED). Global dispersion is the wrong frame for this bank.
- **DNC / NTM usage-based allocation (Graves 2016):** writes to the *least-used* location via a free-list/usage gate — same spirit as candidate-1 but on the **write/allocation** side (not query), reinforcing that "allocate fresh capacity on write, address by content on read" is the established safe split. Candidate-1 = a periodic, coarse version of DNC allocation.

---

## 4. Ranked, immediately-runnable experiments

Validation口径 (ALL): same-protocol BABILong n=100, qa1/qa2/qa5 × 0k-32k, chunk512 bf16 sdpa (`scripts/run_babilong_mem_space.py` + `scripts/score_nested_babilong.py`), **with eval-time SWA W2** (`--swa_eval_chunks 2`, gp-53's biggest proven lever). Train `--total_steps 1000 --save_interval 500`; 续训只在 step1000>step500 时. Base = P11 chunk512 (`launch_mem_space_p11_chunk512_remote196.sh`). Telemetry gate: **usage_cov↑ (target >0.5) AND top1_sim NOT collapsed (must stay ≈0.1-0.2, must NOT spike→~1.0 nor flatten→1/N≈0.008) AND qa1/qa5 8k-32k.**

### EXP-R1 (PRIMARY) — per-sample dead-slot reset to fresh strided content. confidence: medium. **NEEDS CODER.**
- **What:** new flags `--dead_slot_reset_interval R` (chunks; 0=off) + `--dead_slot_reset_mode strided_current` (default; vs `zero`). New per-`MemoryBank` sample-scoped usage counter (resets in `memory_bank.reset()`). Logic in `layer.py` post-selector: every `R` chunks, overwrite zero-usage slots with strided current-chunk tokens (mirror `memory_bank.py:153-164`), skip when `frozen`. Sweep `R∈{8,16}` (32k≈63 chunks → ~4 / ~2 reset events per sample). Single variable vs base. Node: **.196**.
- **Mechanism + why gentle:** §1/§2 — touches only never-selected slots, selector unmodified, strided-diverse content protects top1_sim; VQ-VAE-restart precedent (§3).
- **Why first:** safest of the three, directly tests the user's deadlock hypothesis AND falsifies gp-53's "L1 ceiling is inherent" assumption in one shot.
- **Judge:** usage_cov 0.29→>0.5; top1_sim band held; **qa1 single-fact 8k-32k (most L1-capacity-sensitive)** + qa5 vs base. If qa1 long-range rises → deadlock was the L1 ceiling (re-open L1). If usage_cov rises but long-range flat → L1 ceiling inherent (close L1 line, pivot fully to L3).

### EXP-D2 (DIAGNOSTIC, parallel, low-cost) — true dead-slot telemetry. **NEEDS CODER.**
- **What:** add to QUERY_DIAG two **sample-scoped** scalars (current usage_cov resets every emission window → cannot see across-sample never-selected): `dead_slot_frac` = (#slots with zero selections over the whole sample)/N, and `max_slot_select_count` (the "rich" tail). No-grad, layer-0, fully guarded (mirror the EXP-D L3 block at `layer.py:1019-1095`).
- **Why:** the deadlock and R1's effect are *defined* on cumulative-zero-usage, which is not currently observable (window resets at `layer.py:1008-1010`). Makes R1 interpretable instead of blind. Ship alongside R1 (same coder, same commit area).

### EXP-R2 (SECONDARY, only if R1 promising) — gentle forced co-update. confidence: medium-low. **NEEDS CODER.**
- **What:** `--co_update_dead_slots G` + `--co_update_gate_scale 0.1`; append G zero-usage indices to the write set with scaled gate, content = strided current tokens (NOT a single broadcast vector — see homogenization risk §1). Node: **.249** (parallel) or after R1.
- **Judge:** same cell; **watch key_max_cos / l3-style dead-slot pairwise cosine for homogenization**; usage_cov + top1 + qa1/qa5.

### DEPRIORITIZE
- **Candidate-3 / targeted usage aux / any selection-side dispersion knob:** = rejected arm1-4 family (gp-53 §1). The gradient-free bias is *already implemented and already failed* (arm1). Do not re-run.
- **num_slots↑:** 證伪 (gp-44 §3) — adds dead slots, the opposite of what recycling needs.

---

## 5. Confidence & falsifiers
- Deadlock mechanism (stale-content rich-get-richer, write-side): **confidence high** (code-verified: strided init + selected-only delta-write + content-derived keys).
- Write-side recycling avoids arm4 top1 collapse: **confidence medium-high** (selector unperturbed; contingent on strided-diverse recycle content + conservative R).
- R1 improves BABILong long-range: **confidence medium-low** (capped by gp-53's L1 ≈0 long-range ceiling) — but R1 is explicitly the experiment that tells us whether that ceiling is deadlock-induced or inherent, which is itself high-value.
- **Falsifiers:** if R1 raises usage_cov but qa1/qa5 8k-32k stay flat → L1 ceiling inherent, abandon L1 recycling, all-in on L3 (gp-53 EXP-1/2/3). If R1 top1_sim collapses despite strided content → write-side argument wrong, re-open. **All three candidates need coder; none is pure-hyperparameter** (the only pure-HP variant, the loss-free bias, is arm1 = rejected).
