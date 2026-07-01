# Native Write-Side Mechanism Redesign — Making the Delta-Rule Write DENSE (not top-k sparse)

**Date:** 2026-06-11
**Author:** general-purpose-64 (research, no code changes, no training started)
**Scope:** design "more native" write-side mechanisms that prevent the cold-start dead-slot deadlock at its source, as a parallel/complement to EXP-R1 (gp-59's patch-style periodic reset). Code-grounded audit of `src/memory/mem_space/{memory_bank.py,selector.py,layer.py,config.py}` + `launch_mem_space_p11_chunk512_remote196.sh`. Builds on gp-59 (`20260611_dead_slot_recycling.md`) and gp-53 (`20260611_post_routeA_redirect.md`).

---

## TL;DR

1. **The decisive architectural fact that reframes everything: under the canonical run (`--use_memory_xattn`), the READ is ALREADY over ALL N slots through its own softmax (`MemoryCrossAttentionRead.read`, `selector.py:1457-1540`; wired `layer.py:1671-1700`). The top-k selector does NOT gate the read — it only chooses which slots get WRITTEN (and prepended, but that prepend is masked under xattn).** So the user's "宽写窄读 (wide write / narrow read)" framing is half-already-true: **the read is the wide side; the WRITE is the narrow side (top_k=16+4 global).** The lever to pull is purely the write distribution, and the read being all-N means a write fix needs NO read-side change at all.

2. **Re-reading arm1 and arm4 through this lens sharpens the design rule.** Because the read is all-N, **arm1 (loss_free bias, REJECTED, "harmless no gain") was ALREADY a write-side load-balancer** — it biased which top-k win, i.e. which slots get written — and it didn't help. **arm4 (ST-Gumbel, usage_cov→1.0, top1_sim 0.99→0.11) is the key cautionary tale: it forced the WRITE set to be ~random, so over a sample many slots got *undifferentiated* content → at eval the all-N read softmax had nothing distinctive to peak on → top1_sim collapsed.** The lesson is NOT "don't write broadly" — it is **"broad write is only safe if each slot receives DISTINCT content; broadcasting/randomizing undifferentiated content homogenizes the bank and kills the read peak."** This is exactly why gp-59's R1 uses *strided per-slot* content.

3. **The literature thesis (strongest anchor): the canonical continuous-memory write rule is DENSE, not top-k sparse. The project's top-k write is the anomaly that creates the deadlock.** DeltaNet (Yang et al. 2024, "Parallelizing Linear Transformers with the Delta Rule"), Gated DeltaNet (Yang et al. 2024), Titans (Behrouz et al. 2024), and the fast-weight-programmer line (Schmidhuber 1992; Schlag et al. 2021 "Linear Transformers Are Secretly Fast Weight Programmers") all apply their delta/outer-product update to the **entire** memory matrix every step — every memory cell is touched, addressed by content, weighted by a key-similarity kernel. **None of them gate the write to a hard top-k subset.** Titans even writes the whole neural-memory module per token with a surprise(grad)+momentum+forget rule. **So the "native" fix is simply: make the delta-rule write dense (all slots, similarity-weighted) instead of sparse (top-k). The sparsity is the deviation; removing it returns to the established safe design.** And critically, a dense similarity-weighted write is *automatically differentiated per slot* (each slot's update is weighted by *its own* key match), so it sidesteps the arm4 homogenization trap by construction.

**Verdict:** the user's intuition ("改写入机制，写对了检索自然好") is mechanistically the right lever, AND it coincides with returning to the textbook fast-weight write. PRIMARY native candidate = **dense (all-slot) soft delta-write with per-slot-distinct content** (W2). Cheapest falsifier first = **widen the write top_k 16→32/48** (W1, near-pure-HP). Candidate-3 (write-side load-aware bias) = DEPRIORITIZE — it is arm1, which the all-N-read insight shows was *already* a write-side balancer and *already* failed.

---

## 1. The architecture: where "write" actually lives (code-grounded)

Canonical P11 (`launch_mem_space_p11_chunk512_remote196.sh:30-43`):
`--num_slots 128 --top_k 16 --num_global_slots 4 --selector_temperature 40 --use_delta_rule_writeback --use_memory_xattn --writeback_mode dual_gate`.

Two facts the design must respect:

- **READ = all N slots, own softmax, greedy, content-addressed.** `memory_xattn.read(hidden, slots, slots)` (`layer.py:1671-1678`) attends Q=live tokens over **all N** slot K/V (`selector.py:1503-1509`), gated per-head (`:1514-1521`), added directly to the residual (`layer.py:1693-1700`). The top-k `idx` never enters this path. The H→L1 prepend (the old read) is masked under xattn. **⇒ the read already "sees" every slot; a frozen dead slot is read every chunk — it just contributes nothing useful because its content is a stale chunk-0 snapshot.**
- **WRITE = top_k slots only, delta-rule, per-slot-distinct content.** `idx` (top-16+4) → `memory_bank.write(idx, O_mem_slot, …, delta_rule=True)` (`layer.py:1773-1881`); delta form `slot ← slot + g_in·(new − slot)` (`memory_bank.py:388-397`). `O_mem_slot = hidden_to_slot(O_mem_hidden)` (`layer.py:1739`), where `O_mem_hidden` is the per-slot processed output at the prepended L1 positions (`layer.py:1627`) — **so each selected slot already gets DISTINCT content.** Unselected slots: byte-frozen.

**Conclusion:** the only thing that determines whether a slot ever accumulates content is **whether it enters `idx`.** With strided init + selected-only write, ~91/128 slots never enter `idx` (gp-59 §1, confirmed). The deadlock is 100% a write-coverage problem; the read needs no change.

---

## 2. Candidate evaluation (against the all-N-read architecture)

### Candidate 1 — Soft write / dense all-slot delta (the "native" fix). **PRIMARY (as W2).**
- **Mechanism:** every chunk, in addition to the hard top-k delta write, apply a *weak* delta update to **all N** slots, where each slot's target content is computed **per-slot and distinct** (slots-as-query attention over the chunk tokens — exactly `CrossAttentionMemory.writeback`'s slots-Q→tokens-KV, `selector.py:820-879`, which already produces `[B, N, slot_dim]` distinct content). Update: `slot_n ← slot_n + λ·g_n·(content_n − slot_n)`, λ small (e.g. 0.05).
- **Implementation location:** new block in the write section of `layer.py` (~after `:1881`), gated by a new `--soft_write_weight λ` (0=off → byte-identical to P11). Reuse or add a slots-as-query write attention (a `CrossAttentionMemory`-style module on the layer, or fold into `memory_bank`). **Does NOT touch R1's recycle block (`:1000-1117`) nor `memory_bank.write`'s idx path** — fully isolated, independent on/off switch.
- **Why gentle / preserves the read peak:** λ is small, so live slots' precise content drifts negligibly (`λ·(content−slot)` ≈ 0.05× a residual). Dead slots get a slow trickle of **their own** content (each addresses the chunk through its own key) → they slowly become distinct and writable. Because content is per-slot distinct (NOT a broadcast/pooled vector), the bank does **not** homogenize → the all-N read softmax can still peak → top1_sim protected. This is the precise repair of arm4's failure (arm4 = broad write of *undifferentiated* content).
- **Why "native":** this IS the DeltaNet/Titans/fast-weight write (dense, similarity-weighted, all cells). We are removing an anomalous sparsity, not adding machinery.
- **Failure risk (homogenization /全槽趋同):** the real risk. If λ is too large, or the per-slot write attention collapses (all slots attend the same tokens → near-identical content — the very `summary_q_max_cos→1` collapse v10 fought), all slots converge → read can't discriminate → top1_sim flattens toward 1/N≈0.008. **Mitigations:** (i) keep λ ≤ 0.05; (ii) the slot keys already carry `slot_key_bias` (`selector.py:148`) + `key_repulsion_loss` enforcing key diversity — reuse it to keep the write-attention queries diverse; (iii) telemetry: watch **slot pairwise cosine** (add to QUERY_DIAG) + top1_sim every eval. **How small is safe?** Heuristic: λ·(#chunks a dead slot survives before being selected) should stay ≪ 1 so a dead slot accumulates a *direction*, not a *saturated copy* — at 63 chunks, λ=0.02-0.05 gives cumulative 1.3-3.2× a single residual, enough to differentiate without saturating. Sweep λ∈{0.02,0.05,0.1}; expect 0.1 to start homogenizing.

### Candidate 2 — Wider top-k write (read still narrow-by-selection is moot; read is all-N). **PRIMARY cheap falsifier (as W1).**
- **Mechanism:** raise `--top_k 16 → 32` (and 48). Each chunk, the top-32 by score get a distinct delta write (content already per-slot distinct via `O_mem_hidden`). Read unchanged (all-N greedy). This is "write wider than read" in the only sense that matters here, and it is **near-pure-HP** (`top_k` is already a CLI arg; no new code).
- **Why it might work:** strictly more slots receive real content each chunk → usage_cov rises mechanically; content stays differentiated (per-slot) → low homogenization risk.
- **Honest limitation (vs arm1):** raising top_k only *lowers the selection bar*; it does **not** break "rich-get-richer ordering" — the same high-scoring slots still win the top of the list; we just admit more runners-up. usage_cov 0.29 (~37 slots) → top_k=32 admits ~the top ~50-70 over a sample, still leaving a long tail frozen. So W1 is a partial, cheap probe, not a full fix. **But it is the cheapest test of "does widening the write help at all" — if even W1 moves usage_cov AND long-range, the native W2 is clearly worth the coder cost; if W1 does nothing, it tempers expectations for W2.** Cost: prepend length 16→32 (×512 chunk ×32 layers) → measure wall-clock (~+small).
- **Distinct from arm1:** arm1 kept k=16 and *biased which 16 win* (rebalance) → no new slots written with fresh distinct content beyond the 16. W1 writes to *more* slots with genuinely distinct per-slot content. Distinct from arm4: no noise, ordering preserved, content differentiated.

### Candidate 3 — Write-side load-aware bias (decoupled from read). **DEPRIORITIZE.**
- The premise was "bias the write distribution toward cold slots without touching the read." **But because the read is already all-N, arm1's loss-free bias WAS exactly that** — it biased the selection (=write set) only, never the read (the read doesn't use `idx`). arm1 was REJECTED (harmless, no gain, gp-53 §1). So "read-write-decoupled write bias" is not a *new* lever here — it is the one already tried and failed. The reason it failed is now clear: rebiasing *which* 16 slots win still writes only 16 distinct contents/chunk and keeps the bank's effective capacity low; it doesn't add coverage the way a dense (W2) or wider (W1) write does. **Do not re-run.**

### Candidate 4 (own addition) — surprise/forget-gated dense write (Titans-style), as a W2 refinement.
- If plain dense soft-write (W2) homogenizes, the Titans fix is a **per-slot forget gate** so each slot decays its stale content while accepting new — `slot ← (1−α_n)·slot + λ·g_n·(content_n−slot)`. The dual-gate machinery already exists (`memory_bank.write` forget_gate path, `:369-418`) and could be reused. Keep in reserve; only if W2 shows homogenization.

---

## 3. Relationship to EXP-R1 — orthogonal, combinable, complementary

| axis | EXP-R1 (gp-59) | W2 (this note) |
|---|---|---|
| trigger | **event-driven**: every R chunks, detect zero-usage slots | **continuous**: every chunk, all slots |
| operation | **hard reset** dead rows to strided current content + grace forced-write | **soft delta** trickle (λ small) to all rows, per-slot distinct content |
| failure mode | too-aggressive R evicts a deliberately-parked slot | too-large λ homogenizes the bank |
| code | `recycle_reset`/`force_write` + recycle block `layer.py:1000-1117` | new block in write section `layer.py:~1881` + new flag |
| analogy | VQ-VAE dead-code random restart (Łańcucki 2020) | DeltaNet/Titans dense fast-weight write |

- **Which is more likely to work?** W2 is the more *principled* fix (it removes the root anomaly — write sparsity — and matches the textbook continuous-memory write), but it carries a real homogenization risk that R1 does not. R1 is *safer* (touches only never-selected rows) but is a patch that doesn't address why writes are sparse in the first place, and its discrete reset can churn. **My ranking: try the cheap W1 first to confirm direction; run W2 and R1 in parallel on separate nodes as the two real contenders; expect W2 to give a smoother usage_cov curve and R1 a steppier one.**
- **Combinable?** Yes, and cleanly — different flags, non-overlapping code. R1 rescues fully-dead slots via periodic reset; W2 prevents death via continuous trickle. A combined run (`--dead_slot_reset_interval 16 --soft_write_weight 0.03`) is a sensible *third* arm if both individually show usage_cov↑ without top1 collapse. **Combined risk: double content perturbation → watch top1_sim hardest here.** Isolation guarantee: W2's block must be gated by `soft_write_weight>0` and live *outside* the `_do_recycle` block, so the two switches are independent (R1 on / W2 off / both reproduce as designed).

---

## 4. Ranked, immediately-runnable experiments

Validation口径 (ALL, matching gp-53/gp-59): same-protocol BABILong n=100, qa1/qa2/qa5 × 0k-32k, chunk512 bf16 sdpa (`run_babilong_mem_space.py` + `score_nested_babilong.py`), **eval-time SWA W2** (`--swa_eval_chunks 2`). Train `--total_steps 1000 --save_interval 500` (new convention: 先 1000 步);续训只在 step1000>step500. Base = P11 chunk512. **Telemetry gate (all): usage_cov↑ (>0.5 target) AND top1_sim held ≈0.1-0.2 (must NOT spike→1.0 nor flatten→1/N≈0.008) AND slot pairwise cosine NOT rising AND qa1(single-fact)/qa5 8k-32k vs base.**

### EXP-W1 (PRIMARY-cheap, confidence medium-low, **NO CODER — pure HP**)
- **What:** P11 chunk512 base + `--top_k 32` (arm A) and `--top_k 48` (arm B). Single variable. Nodes: **.196 (k=32) / .249 (k=48)**, or B200 .188.
- **New params:** none (`top_k` exists). **No coder needed.**
- **Mechanism:** widen the write set → more slots get distinct per-slot content/chunk → usage_cov↑; read stays all-N greedy. Cheapest possible test of the "wide write" hypothesis.
- **Judge:** usage_cov 0.29→? ; top1_sim band held; qa1/qa5 8k-32k W2 vs base; measure wall-clock (prepend 16→32/48). If even this lifts long-range → strong green light for W2. If flat → tempers W2 (widening alone insufficient; density+continuity needed).

### EXP-W2 (PRIMARY-native, confidence medium, **NEEDS CODER**)
- **What:** dense all-slot soft delta-write. New flags `--soft_write_weight λ` (0=off) + `--soft_write_content slot_query` (per-slot-distinct content via slots-as-query write-attention over chunk tokens; reuse `CrossAttentionMemory.writeback` content path or a new small module). Apply `slot_n ← slot_n + λ·g_n·(content_n − slot_n)` to ALL N slots each chunk, IN ADDITION to the existing top-k hard write. Sweep λ∈{0.02, 0.05}. Single variable vs base. Node: **.196** (after/with W1).
- **Coder isolation note (vs R1):** implement in the write section of `layer.py` (~after `:1881`), gated by `soft_write_weight>0`; add the content module to the layer. **Must NOT modify the `_do_recycle` block (`:1000-1117`), `memory_bank.recycle_reset`/`force_write`, nor the top-k `memory_bank.write` calls.** Add a new `memory_bank.soft_write(content_all_N, weight, gate)` method (mirrors `force_write` but mask=all, weight=λ) so it composes with R1 without touching R1's methods.
- **Mechanism + why gentle:** §2 candidate-1 — dense similarity-weighted delta = textbook fast-weight write; per-slot-distinct content avoids arm4 homogenization; small λ protects live slots' precise content → read peak preserved.
- **Judge:** usage_cov↑ (expect smoother/higher than W1); **slot pairwise cosine (NEW telemetry, must add) flat** = no homogenization; top1_sim band held; qa1/qa5 8k-32k W2. Falsifier: if top1_sim flattens or slot cosine→1 → λ too large or write-attention collapsed → drop λ / add key-repulsion on write queries; if even λ=0.02 homogenizes → density itself is unsafe here (would re-validate the top-k sparsity, surprising).

### EXP-D3 (DIAGNOSTIC, ship with W2, **needs coder, tiny**)
- **What:** add to QUERY_DIAG: **mean pairwise cosine of slot CONTENT** (`slots[0]`, off-diagonal), the direct homogenization signal for any broad-write method (W1/W2/R1). Mirror the guarded EXP-D2 scalars (`layer.py:1046-1052`). No-grad, layer-0.
- **Why:** broad-write methods all risk全槽趋同; currently only key cosine (`key_repulsion`) is observable, not content cosine. Makes W1/W2/R1 interpretable. Same coder/commit as W2.

### DEPRIORITIZE
- **Candidate-3 / any write-side selection bias (loss_free / aux):** = arm1, which the all-N-read insight shows was *already* a write-side balancer and *already* REJECTED. Do not re-run.
- **num_slots↑:** 證伪 (gp-44 §3) — adds dead slots, the opposite of what a write fix needs.

---

## 5. Confidence & falsifiers
- "Read is all-N; top-k gates write only; deadlock is pure write-coverage": **confidence high** (code-verified, `selector.py:1457-1540` + `layer.py:1671-1700` + write path `:1773-1881`).
- "Dense write = textbook fast-weight/DeltaNet/Titans default; top-k sparsity is the anomaly": **confidence high** (literature).
- "W2 (dense soft write, per-slot distinct) raises usage_cov without collapsing top1": **confidence medium** (mechanism-sound + arm4-lesson-corrected, but homogenization is a real, untested risk — gated by D3 telemetry + λ sweep).
- "W1/W2 improve BABILong long-range": **confidence medium-low** — still capped by gp-53's finding that L1's long-range ceiling may be ~0 (L3 is the主力). **Like R1, W1/W2 are precisely the experiments that test whether that L1 ceiling is deadlock-induced or inherent.** If usage_cov↑ but qa1/qa5 8k-32k flat → L1 ceiling inherent → pivot fully to L3 (gp-53 EXP-1/2/3). If long-range rises → deadlock WAS the ceiling (re-open L1, big win).
- **No code changed, no training started.** W1 is HP-only and launchable now; W2/D3 need one coder pass (isolated from EXP-R1).
