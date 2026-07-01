# L1 Memory-Acquisition Loss — design + root-cause (2026-06-04)

author: general-purpose-9
run analysed: `l3iso_noL3_local` (commit 35ea240), logs/l3iso_noL3_local.log
core question (team-lead/user): we lack a loss that *forces* L1 (slot memory) to
actually acquire and use memory. Design it. Preferred form = restrict the local
window so chunk j+1 can only see chunk j + the slots chunk j wrote, then NTP loss.

FINAL design decision (team-lead, confirmed): KEEP the global lm_loss (preserves
language ability) and ADD a restricted-window NTP aux loss as a *second* term —
NOT a replacement. Plus two new questions: (A) chunk-length curriculum (small→large);
(B) **CRITICAL feasibility check — how long is each dolmino sample?** Answered in
the new sections below.

---

## ★ CRITICAL UPDATE — dolmino data IS long-contiguous; only the loader shuffles it

I read the actual data + preprocessing (not impressions). Decisive facts:

- **Each stored row = exactly 1024 tokens, 100% of rows** (probed 2000 random rows:
  min=median=mean=p90=max=1024). Schema: single `input_ids` list, int32. 463,866
  rows ⇒ ~475M tokens total (the "0.5B").
- **Rows are NOT independent documents — they are contiguous slices of one long
  token stream.** Preprocessing `MemLong/scripts/process_dolmino.py:30-37`:
  `data = np.fromfile(shard).reshape(n_chunks, 1024)` — it flattens each shard into
  one stream and reshapes into 1024-token rows **in order**. I confirmed by decoding
  the tail of row i and head of row i+1: they form continuous readable English
  (e.g. row0 ends "...to capture the Golden" / row1 starts "Fleece, this ambitious
  program..."). Verified at rows 0/1, 100/101, 5000/5001 — all seamless.
- Document boundaries DO occur mid-stream (65.8% of rows contain a BOS/EOT inside,
  only 0.1% start with BOS) — i.e. docs are concatenated end-to-end, so a given
  1024-row may span a doc boundary, but **consecutive rows are physically adjacent
  text** within a shard.

**Implication (this REVERSES the cost estimate of STEP1):** the long-contiguous
data the plan needs ALREADY EXISTS. The ONLY thing destroying cross-chunk
dependency is `DolminoCurriculumDataset` calling `rng.shuffle(indices)` and then
grabbing *random* rows per group (dolmino_dataset.py:104-141). **The fix is to
stop shuffling at the row level and instead take CONSECUTIVE rows as the
(context…→target) group.** That is a ~10-line change, not a data re-pack:

- dolmino_dataset.py:103-111: instead of `indices=range(N); rng.shuffle(indices)`
  then strided sharding, pick a random *start row* per group and take the next
  `n_ctx+1` **consecutive** rows (`s, s+1, …, s+n_ctx`). Shuffle only the group
  *start offsets*, not the within-group order. Keep DDP sharding by assigning
  disjoint start-offset ranges per rank.
- This yields up to (n_ctx+1)×1024 tokens of genuinely continuous text per sample,
  so the target chunk's antecedents really do live in earlier chunks ⇒ memory
  becomes load-bearing.

**Sub-chunking for smaller chunk_size:** since rows are 1024 and contiguous, to get
chunk_size=256 you can either (a) keep loading 1024-rows and `.split(256)` inside
the step to make 4 sub-chunks (already how the step splits — train uses
`chunk_size` arg), or (b) concatenate `m` consecutive rows then split into
`(m*1024)//chunk_size` pieces. Both give contiguous sub-chunks. No re-tokenisation
needed.

**Cross-document contamination caveat:** because docs are concatenated without
per-doc attention isolation, a contiguous span may cross a doc boundary. Within
THIS architecture that is harmless: each chunk is an independent forward (no KV
crosses chunks) and the only cross-chunk channel is memory. A doc boundary inside
the span just means some target tokens have no useful antecedent in memory — which
is fine (the loss simply gets no memory benefit there, same as a hard example). It
does NOT corrupt other tokens. If we later want clean dependency we could filter
spans that contain a BOS, but it is not required for STEP1.

---

## TL;DR (read this first)

1. **The user's "restrict local attention to the previous chunk" is ALREADY TRUE
   in the code — and even stricter.** Each chunk is an *independent* forward
   (`model(input_ids=chunk, use_cache=False)`, **no `past_key_values`** ever
   threaded: train script:978/985/1072/1160/1248). Within a chunk the H×H block
   is plain causal over T=1024 (layer.py:137-141). **No KV crosses a chunk
   boundary at all.** The ONLY cross-chunk information channel is the slot memory
   bank (shared bank, `_reset_banks`/`_detach_banks`). So structurally, memory is
   already the sole long-range path. We do **not** need to build a masked second
   forward or shrink an attention window — that mechanism already exists.

2. **So why isn't memory load-bearing? Because the DATA has no cross-chunk
   dependency.** `DolminoCurriculumDataset` groups **random shuffled, unrelated
   documents** (dolmino_dataset.py:100-141): the N context chunks and the target
   chunk are independently sampled rows (each a different 1024-tok document).
   Predicting the target's tokens gets **zero** benefit from remembering the
   previous (unrelated) chunks. The LM loss therefore has no reason to route/write
   anything useful → routing collapses to uniform, writeback ≈ 0. This is a deeper
   root than "objective lets memory be optional": with this data, **no objective
   change can help, because the answer is simply not in the previous chunks.**

3. **A second, independent severing exists (from general-purpose-1's prior report,
   confirmed):** this run sets `--use_decoupled_read`, which masks H→L1 attention
   (`mask_h_to_l1=cfg.use_decoupled_read`, layer.py:1057/1068; mask built
   layer.py:160-161) and routes the read through `CrossAttentionMemoryV2.read`
   using **all N slots' own softmax** (layer.py:1193-1203) — that path never
   touches `idx`/`scores`. So even the *intra-window* LM gradient to the selector
   is cut. Net: selector is trained only by `load_balance*0.01 + entropy*0.001`,
   both of which **push toward uniform** (layer.py:1426-1429).

**Conclusion: the missing ingredient is (a) data with genuine cross-chunk
dependency, and (b) a gradient path from "got the answer right" back to routing +
writeback. The user's NTP-window design is correct in spirit but is already the
structural setup; what must change is the DATA and the grad path, not the mask.**

---

## Evidence

### Chunks are independent forwards; no KV threading
- `dolmino_train_step_tbptt` (train:1061-1114): loops chunks, each
  `out = model(input_ids=chunk_input, labels=chunk_input, use_cache=False)`
  (train:1072). No `past_key_values=` anywhere in the file (grep: only
  `use_cache=False`). The within-chunk mask is plain causal (layer.py:137-141)
  or SWA if `swa_window>0` (not set here).
- Cross-chunk state = the shared `MemoryBank.slots` only. Within a `bptt_window`
  (=2) the bank stays attached so chunk j's writeback is gradient-connected to
  chunk j+1's read; `window_loss.backward()` then `_detach_banks` at the boundary
  (train:1107-1114). **So the write→read gradient IS connected for window=2** —
  the plumbing the user worried about in Q2 is fine.

### Data has no cross-chunk dependency (the real root)
- `dolmino_dataset.py:103-111`: `indices = range(N); rng.shuffle(indices)`, then
  strided sharding. `group_indices = my_indices[ptr:ptr+n_ctx+1]` (line 125) —
  these are **random, non-contiguous** dataset rows.
- `dolmino_dataset.py:129-141`: each context chunk and the target are read from
  *different shuffled rows* → unrelated documents. Target is NOT the continuation
  of the context. Memory of chunk j is useless for predicting chunk j+1.
- Implication: top1_sim_mean sliding to ≈1/128 (uniform) is the *correct* solution
  for this data + this loss. lm loss stays healthy (~2.3, log) because the model
  predicts each chunk from its own local tokens, never needing memory.

### Read-path severs selector gradient (second root, confirms prior report)
- launch flag `--use_decoupled_read` (scripts/launch_l3iso_noL3_local.sh:32).
- layer.py:1057/1068 pass `mask_h_to_l1=cfg.use_decoupled_read`; mask sets
  H→L1 columns to -inf (layer.py:160-161). Read produced by
  `self.decoupled_read.read(hidden, read_slots, read_slots)` (layer.py:1200-1203)
  over ALL slots, content-blind to `idx`/`scores`. Selector aux only:
  `aux["load_balance"]=lb*0.01`, `aux["entropy"]=ent*0.001` (layer.py:1426-1429).

### Writeback magnitude (answers original hypothesis-1)
- WRITEBACK_DIAG: `alpha(inject_gate_mean)=0.118` constant (std≈0.007), driven by
  `inject_gate_bias_init=-2.0` → sigmoid(-2.0)=0.119 (layer.py:425, 1159).
  `g_forget_mean≈0.87` (forget_bias_init=2.0 → sigmoid(2.0)=0.88), `g_in≈0.5`.
  `slot_delta_abs_mean≈1.6e-5`. The tiny delta is a *symptom* of uniform routing
  (each slot gets 16/128 of diffuse, unspecialised content), not the cause. Even
  with bigger alpha, writing unrelated-document content into slots can't help an
  unrelated target. So tuning the write gate is NOT the fix.

---

## Answers to the four specific questions

### Q1 — Can the code support "only see the previous chunk"?
It already does something stricter: **each chunk sees only itself** (causal over
its own 1024 tokens) **plus the slot bank**. There is no KV from chunk j fed to
chunk j+1 (no `past_key_values`). To *literally* also feed chunk j's local KV
into chunk j+1 you'd have to add KV threading, which does not exist and is **not
needed** — it would actually weaken the memory pressure. **Do not change the
mask.** The lever is the data + the grad path.

### Q2 — Slot injection point + gradient connectivity
- Inject (read): two paths. (i) prepend path: `parts.append(M_sel_hidden)`
  (layer.py:1030), consumed via the extended-seq mask (layer.py:1057). (ii) when
  `use_decoupled_read`: prepend is masked off and read comes from
  `decoupled_read.read(...)` added as `next_hidden += g*decoupled_read_out`
  (layer.py:1216-1217).
- Writeback: layer.py:1215 (`next_hidden = bypass_h + g*slot_delta + ...`) and the
  dual-gate `memory_bank.write(idx, O_mem_slot, gate=g_in, forget_gate=g_forget)`
  (layer.py:1290-1323).
- **Gradient write→read IS connected within a bptt_window** (train:1092-1114; bank
  not detached until the window edge). **bptt_window=2 is enough** for chunk j
  write → chunk j+1 read. The problem is NOT connectivity — it's that with
  decoupled_read the read doesn't depend on `scores`/`idx`, and with random data
  the read content is useless. Fix those two and window=2 already carries the
  signal (raise to 3-4 only if you want chunk j→j+2 credit).

### Q3 — How to compute the NTP loss
There is no separate "global LM" path to preserve, because attention is already
local-per-chunk. The existing target-chunk `lm_loss` (train:1075) **is exactly the
restricted-window NTP the user wants** — but it only becomes memory-forcing once
the target genuinely depends on earlier chunks. So:
- **Do NOT add a second masked forward.** Keep the single forward.
- The "force memory" effect comes from: (1) data dependency + (2) restoring the
  selector gradient. Optionally add an explicit routing-supervision aux as a
  booster (cheap, see Plan step 2).

### Q4 — chunk_size and segment length
With *random* chunks, no size helps. With **contiguous-document chunking** (the
required data change), the dependency length must exceed one chunk:
- Set `chunk_size` to **256-512** (smaller chunk ⇒ a single chunk holds less of
  the document ⇒ predicting later tokens *must* recall earlier chunks).
- Segment length **8-16 chunks** of one long document (≥4K-8K tokens of contiguous
  text), so genuine long-range dependencies span the memory.
- Keep `bptt_window=2` (memory of any prior chunk is reachable via the bank's
  recurrent state; window only controls how many chunks share one backward).

---

## Recommended plan (priority ordered: essentiality × cost)

### STEP 1 (do first) — give the data real cross-chunk dependency  [confidence: very_high]
**This is the load-bearing change. Without it everything else is noise.**
- File: `src/memory/mem_space/dolmino_dataset.py:103-141`.
- Change grouping from "n_ctx+1 random shuffled rows" to "one long contiguous span
  split into n_ctx+1 consecutive chunks." Either (a) concatenate consecutive rows
  of the same source document, or (b) repackage data as long docs and slice into
  `chunk_size` pieces so context_chunks[0..n-1] + target are one continuous text.
- Add `--chunk_size 256` (or 512) and a curriculum that grows n_ctx (already
  supported via `set_n_context`). Effective dependency length = (n_ctx+1)*chunk.
- Expected: target tokens whose antecedents fall in chunk 0..j-1 are now only
  predictable via memory → LM gradient flows to routing + writeback → top1_sim
  rises above uniform, slot_delta grows. **This alone may fix the collapse.**
- Risk: low. Pure data-pipeline change; no architecture risk. Verify with a
  contiguous-vs-shuffled A/B on the same step.

### STEP 2 (do with Step 1) — restore selector gradient  [confidence: high]
Two mutually-exclusive options:
- **(2a) Turn OFF decoupled read** for this experiment: drop `--use_decoupled_read`
  (launch:32). Then H→L1 prepend attention is live and the read directly depends
  on `scores`/`idx` → selector gets LM gradient. Cheapest, lowest-risk. Confidence
  high; this is the single flag most directly responsible for the severed path.
- **(2b) Keep decoupled read but make it top-k/score-weighted**: gather K/V by the
  selected `idx` and weight by the STE routing weights inside
  `decoupled_read.read` (layer.py:1118-1203). More faithful to the intended
  decoupled design but needs a code change in `CrossAttentionMemoryV2.read`.
  Confidence med (more surface area).
- Recommendation: ship **2a** with Step 1 first (fastest signal); revisit 2b only
  if you specifically want decoupled read.

### STEP 3 (booster, optional) — explicit routing-supervision aux  [confidence: med]
- Re-enable `--route_aux_weight` (currently 0). Mechanism already implemented:
  `_compute_route_aux` CE pushing chunk j+1's `scores` onto the slots chunk j
  wrote (`prev_idx1`), folded into the windowed loss (train:558-583, 1085-1090).
- This directly supervises "read where you last wrote," giving routing a gradient
  *independent* of the LM path. Start small (`0.05-0.1`) to avoid fighting the LM
  objective.
- Risk: med. With random data it supervises a meaningless target, so **only enable
  AFTER Step 1** (contiguous data) where "where I wrote last chunk" is genuinely
  where the next chunk's antecedent lives.

### Original hypothesis-2/3 verdicts (for completeness)
- aux losses pushing uniform: **partly true but secondary.** load_balance(0.01)+
  entropy(0.001) do reward uniformity and, once the LM path is severed (decoupled
  read) they are the *only* selector gradient → they win. But lowering them does
  not create a pull toward content addressing; you need Steps 1-2 for that. Lower
  load_balance to ≤0.001 only as a minor assist.
- slot_query softmax-weighted pooling "万金油" collapse: real but tertiary. With a
  genuine objective (Steps 1-2) it is supervised away; not worth special-casing.
- temperature=40 can't save it because the underlying cosine signal is
  uninformative (no objective driving keys apart); raising temperature on noise
  just sharpens noise.

---

## Plan B (retrieval supervision via provenance)  [confidence: med, defer]
- babi needle data DOES carry provenance (we know which sentence holds the
  answer), and babi IS already mixed in (`--babilong_mix_fraction 0.15`,
  `babilong_train_step` train:1125). But the current babi pipeline tokenises a
  formatted prompt and chunks by size (train:1155); the sentence→chunk mapping is
  not currently surfaced. Wiring provenance would require the babi dataset to emit
  the target slot/chunk index. Higher plumbing cost than Step 1-3; recommend only
  if contiguous-LM dependency proves too weak a signal. The route_aux mechanism
  (Step 3) is the natural place to inject a provenance target once available.

## Plan C (just re-open l_recon / route_aux / diversity)  [confidence: low as a fix]
- `l_recon` (layer.py:1476-1479) trains "slots can be DECODED back into the L3
  summary" = *storage fidelity only*. It does not force *retrieval* or *use*, and
  with no-L3 (`--no_l3_summary`) the L3 summaries don't even exist → no-op here.
- `l3_diversity`/`q_multi_diversity` (layer.py:1462-1469) push keys/queries apart
  = *anti-collapse regulariser*, not a "use memory" objective; they treat the
  symptom (uniformity) not the cause (no reason to use memory).
- route_aux alone (without Step 1 data) supervises a meaningless target.
- Verdict: these are regularisers/proxies; none makes memory load-bearing on their
  own. Useful only as boosters on top of Step 1.

---

## Kill decision for l3iso_noL3_local
**Recommend KILL.** It is the no-L3 control of an L3-isolation ablation; its
collapse to uniform is expected and (given the random-grouping data + decoupled
read) it cannot inform the real question. Continuing wastes 8 H20. Free the node
for a Step-1+Step-2 run (contiguous-document data, chunk_size=256, decoupled read
OFF, curriculum on n_ctx) which directly tests whether memory becomes load-bearing.

---

## ★ Dual-loss design (FINAL, per team-lead): global lm_loss + restricted-window NTP aux

Objective = `total = lm_loss_global + aux_weight * lm_loss_restricted`.
- `lm_loss_global`: the EXISTING loss, unchanged — full per-chunk causal NTP, memory
  available, weight 1.0. Preserves language ability.
- `lm_loss_restricted`: same NTP CE on chunk j+1's tokens, but chunk j+1 may ONLY
  use (its own causal tokens) + (chunk j's slots). This is the memory-forcing term.

### Q1 — second forward, or single forward with two losses?

**Subtle but important:** in the CURRENT architecture, the "restricted window" and
the "global" forward are ALMOST identical, because chunks are ALREADY independent
forwards with no KV crossing boundaries. The only thing the global path adds over
"see only chunk j + chunk-j slots" is the *accumulated* memory from chunks 0..j-1.
So the two losses differ ONLY in **what the memory bank contains** when chunk j+1
is forwarded:
- global: bank = writes from chunks 0..j (full history)
- restricted: bank = writes from chunk j ONLY (one-step memory)

That means you **cannot** get both from one forward by just changing an attention
mask — they need different bank states. Options, cheapest first:

- **(Option 1, RECOMMENDED) Reuse the existing chunk loop as the GLOBAL loss, and
  add ONE extra forward per chunk for the restricted loss.** In
  `dolmino_train_step_tbptt` (train:1061-1114), you already forward every chunk and
  accumulate `window_loss` = global lm. Before advancing, do a second forward of
  chunk j+1 against a bank holding only chunk j's write. Concretely: snapshot the
  slot bank after chunk j's write (`slots_after_j`), run a no-history forward of
  chunk j+1 with the bank reset to `slots_after_j`, take its `out.loss` as
  `lm_restricted`, add `aux_weight * lm_restricted` to `window_loss`. Cost: +1
  forward per chunk ⇒ ~2× fwd FLOPs (bwd shared through the connected graph). This
  is the clean, low-risk path. confidence: high.

- **(Option 2, cheaper, approximate) "one-step memory" curriculum instead of a
  second loss.** Replace global with: for EACH chunk, reset bank to only the
  immediately-previous chunk's write before forwarding (sliding 1-chunk memory).
  Single forward, no extra cost. BUT this DROPS the global multi-chunk lm_loss the
  user explicitly wants to keep ⇒ violates the confirmed decision. Reject.

- **(Option 3, single forward, mask trick) NOT possible cleanly** because the
  distinction is bank *contents*, not an attention mask. The H→L1 mask only gates
  whether tokens read slots at all, not *which* writes are in the bank. So a mask
  cannot produce the "only chunk-j memory" state from the full-history forward.

**Recommendation: Option 1.** It is the only one that honours "keep global lm_loss
AND add restricted aux," and it is a localized change in `dolmino_train_step_tbptt`
(train:1061-1114) — snapshot `mem_layers` bank slots after each chunk's write,
re-inject for the restricted forward, fold `aux_weight*lm_restricted` into
`window_loss`. Gradient of the restricted forward flows into chunk j's writeback
(routing+write) because we re-use the SAME `slots_after_j` tensor (grad-connected,
not detached) — this is exactly the targeted "force routing/writeback" signal.

Memory note: the restricted forward should run inside the same window graph (do not
detach `slots_after_j`) so writeback gets gradient. VRAM ≈ +1 chunk activation;
with gradient_checkpointing on H20 this is fine at chunk_size≤512.

### Q2 — aux_weight initial value + warmup

- The restricted lm CE will START HIGHER than global lm (less context ⇒ higher
  perplexity), roughly global+0.5…1.5 nats early on. So at weight 1.0 it would
  dominate. Start **aux_weight = 0.3**, which keeps the aux gradient comparable-to-
  but-below the global term while still being load-bearing.
- **Warmup:** linear ramp 0 → 0.3 over the first ~200 steps. Rationale: at init the
  memory is empty/random; forcing the restricted loss too hard before slots carry
  signal just injects noise into routing. Let the global loss + early writes warm
  the bank, then phase in the pressure. confidence: med (tune by watching
  top1_sim_mean: if it does not rise above ~3×uniform by step ~400, raise to 0.5).
- Keep `load_balance_weight` LOW (≤0.001) while aux is on, so the uniformity
  pressure doesn't fight the new content signal.

---

## ★ Q (curriculum) — chunk-length schedule: small → large

Premise (confirmed sound): small chunks ⇒ each chunk holds less of the document ⇒
predicting later tokens MUST recall earlier chunks ⇒ dense, strong memory signal.
Large chunks ⇒ more is answerable locally ⇒ weaker memory signal but longer-range
test. Anneal small→large so the model first LEARNS to use memory (easy, strong
gradient) then GENERALISES to longer range.

**Recommended schedule (step-triggered, simple & reproducible):**
- start `chunk_size = 128`, n_ctx grows via existing curriculum
- `chunk_size` schedule: `0:128, 600:256, 1500:512, 3000:1024`
- keep `bptt_window = 2` throughout (recurrent bank reaches all prior chunks; window
  only sets how many chunks share one backward).
- segment length: aim effective span (n_ctx+1)*chunk_size ≥ 4×chunk_size early
  (so ≥3 context chunks of real dependency). With 1024-token contiguous rows
  available, at chunk_size=128 a single row already yields 8 sub-chunks — plenty.

**Step- vs loss-triggered:** recommend **step-triggered** (deterministic,
reproducible, matches the existing `--curriculum` mechanism which is step-based:
train:107-138). Loss-triggered (grow when restricted-loss plateaus) is more
adaptive but adds nondeterminism and a plateau-detector; only worth it if step
schedule proves too rigid. confidence: med on exact step boundaries (data-dependent;
watch top1_sim and restricted-lm gap), high on the small→large direction itself.

Implementation: `chunk_size` is currently a fixed arg (launch:22). To make it a
curriculum it must become step-dependent in the dolmino step + dataset split logic
(the step already `.split(chunk_size)`). Small change but touches both
dolmino_dataset (how many rows to concat) and the train step. confidence: med.

---

## Revised priority (supersedes earlier STEP1 cost estimate)

0. **(now trivial) Make grouping contiguous** — dolmino_dataset.py:104-141, ~10 lines
   (consecutive rows instead of shuffled). very_high. The long-contiguous data
   ALREADY EXISTS in the arrow store; we were just shuffling it away.
1. **Restore selector gradient** — drop `--use_decoupled_read` (launch:32). high.
2. **Add dual-loss restricted-window aux** (Option 1 above) — train:1061-1114,
   aux_weight=0.3 + warmup. high (this is the explicit "force memory" objective).
3. **chunk_size curriculum 128→1024** — med, after 0-2 land.
4. route_aux booster 0.05-0.1 — med, optional after contiguous data is in.

Do 0 + 1 first (cheapest, may already lift top1_sim off uniform); then layer in 2.
