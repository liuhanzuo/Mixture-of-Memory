# HIDDEN (FIFO snapshot) vs SWA (live token) — deep analysis (2026-06-26)

> Author: researcher subagent. Read-only, no GPU. All citations `file:line`.
> Question set (user, 4 sharp questions): why does feeding the *correct* hidden
> still fail to retrieve, vs SWA which "should be equivalent"; is hidden trained;
> do we lack MemoryLLM's dim-reduction; is slot-viz worth it. Verdict on whether
> the hidden direction is still alive.

---

## TL;DR

| # | Question | Verdict | conf |
|---|---|---|---|
| Q1 | FIFO-hidden ≠ SWA-token: why? | **They are provably NON-equivalent on the computation graph.** FIFO reads a *per-layer frozen snapshot* of each kept chunk, contextualized *at write time in isolation from the query*. SWA re-forwards the raw tokens so every layer's keys/values are *jointly, query-conditionally re-contextualized*. FIFO = query-side multi-hop accumulation only; SWA = full bidirectional needle↔query coupling. The values FIFO retrieves are query-agnostic. **This is the headline mechanism gap, deeper than pos-0.** | HIGH (mechanism) / LOW-MED (clean magnitude) |
| Q2 | Is hidden trained? | **Yes — same `_forward_fifo` runs at train time, readout path is byte-identical to eval (no mechanism mismatch).** BUT training is ineffective for long-range: (a) `fifo_detach=True` → the *write* side gets ZERO gradient; (b) curriculum `0:3` trains a 3-deep buffer while eval needs 25-64-deep → the dilution/pos regime is never trained; (c) `bptt_window=1` → no cross-chunk credit. So hidden-read is *lightly activated in a regime that does not transfer*. | MED-HIGH |
| Q3 | Missing MemoryLLM dim-reduction? | **No dim-reduction missing — MemoryLLM keeps d=4096; it does not project to lower dimension.** What we lack is a *trained count-compressor* (learnable memory-token attention-pool → fixed 256 slots/chunk + dedicated inject/read LoRA). But the oracle proves raw 4096 hidden IS addressable, so dimensionality is NOT the bottleneck; and MemoryLLM is itself flat-mediocre (~35) on BABILong needle. Low odds this is "the fix". | MED-HIGH |
| Q4 | Slot-usage viz? | **Low value for the hidden/FIFO line.** `dump_slot_usage_dist.py` reads `_cum_usage` routing counts that the FIFO path never populates (FIFO bypasses all routing). It diagnoses the *paused slot architecture*, orthogonal to current work. Skip unless reviving slots. | HIGH |
| Q5 | Is hidden alive? Next exp? | **Conditionally alive but capped; do the zero-train diagnostics first.** Highest-ROI mix=0 experiment = the commissioned **logit-lens hidden-recall probe** (settles stored-vs-unreadable) + a **clean SWA-token W6 eval** (gives the missing clean hidden-vs-token magnitude). Gate any further hidden *training* on the probe. | MED |

---

## Q1 — FIFO-hidden vs SWA-token: the precise computational/information-theoretic difference

### The two paths, exactly

**SWA (W6 open-book), `scripts/run_babilong_mem_space.py:577-583`:**
```python
start  = max(0, len(chunks) - (swa_eval_chunks + 1))
window = torch.cat(list(chunks[start:]), dim=0)   # RAW TOKEN IDS
cur    = window.unsqueeze(0)
... model(input_ids=cur)                           # full layered forward
```
`window` is raw token ids. The model runs a **complete forward** over them. At layer L, the key/value for any window token *t* is `W_{k,v}·h_t^(L)`, where `h_t^(L)` is *t*'s layer-L representation **computed in THIS pass**, i.e. *t* has already attended every other window token (including the query/question tokens) at layers `0..L-1`.

**FIFO (`_forward_fifo`, `src/memory/mem_space/layer.py:1338-1342`):**
```python
kept_chunks      = [valid[i] for i in kept_local_idx]   # entries of self._fifo_buf
prefix           = torch.cat(kept_chunks, dim=1)        # FROZEN per-layer hiddens
extended_hidden  = torch.cat([prefix, hidden_states], dim=1)
... wrapped_layer(extended_hidden, ...)
```
Each `_fifo_buf` entry is written at the **end** of a *prior* forward as `h_stored = hidden_states.detach()` (`layer.py:1554`) — i.e. the **input hidden to layer L** of the chunk *at the time that chunk was the current chunk*. The buffer is **per-layer** (each `MemorySpaceLayer` has its own `_fifo_buf`; reset at `train_mem_space_dolmino_cpt.py:662-663`), and **all 32 layers are FIFO-wrapped** (`patch.py:81-82`, `layer_indices=None`). So at read-time layer L, the prefix key/value for kept chunk *c* is `W_{k,v}·h_c^(L)`, where `h_c^(L)` was produced by running chunk *c* through layers `0..L-1` **attending only c's own then-current buffer — never the query**.

### The four exact differences (ordered by depth)

**(1) Frozen isolated-contextualization snapshot vs live joint re-contextualization — THE core difference.**
- Both inject at *every* layer (FIFO is genuinely all-32-layer, confirmed by the oracle analysis). So "injected at one layer only" is NOT the issue.
- The difference is the **content of the keys/values**. SWA's layer-L prefix keys carry *L layers of cross-window, query-aware* contextualization. FIFO's layer-L prefix keys carry *L layers of within-chunk-only, write-time* contextualization. They are different functions of different inputs; equality would require the query to have been present at write time, which it never is.

**(2) Broken multi-hop needle↔query coupling (the information-theoretic crux).**
- SWA is a fully coupled dynamical system over the window: at layer m the query reads the needle; at layer m+1 the *needle's* representation reflects that it was queried (it attended the query at layer m), and the query's representation reflects what the needle said. Information ping-pongs needle↔query across all 32 layers — deep, multi-hop, query-conditional retrieval.
- FIFO supports **only query-side accumulation**: the current chunk (query) integrates needle info across all 32 layers, but the needle's prefix snapshot at layer m+1 is **frozen** and does NOT reflect that the query just read it at layer m. The back-reaction needle→(sees query)→(refines) is absent. Formally: FIFO preserves `I(chunk_content ; snapshot)` but destroys the conditional structure `I(answer ; query, chunk)` that needs query-conditional, cross-layer coupling.
- Consequence: for **single-fact lookup (qa1)** one query-side hop can suffice → the penalty *can* be modest (consistent with leaked-ckpt oracle≈W6). For **multi-hop / disambiguation (qa2/qa3/qa5 temporal)** the penalty is structurally larger.

**(3) Query-agnostic values (staleness).** Even within one hop, SWA retrieves `v_t = W_v·h_t^(L)` where `h_t` saw the query; FIFO retrieves `v_c = W_v·h_c^(L)` where `h_c` did NOT. FIFO returns "what chunk c found salient at write time," not "what chunk c says about the question."

**(4) Position collapse (secondary, `layer.py:1392-1401`).** Default `pos_mode=None` puts all prefix tokens at RoPE pos-0; SWA has correct relative positions. The oracle analysis already showed this is **secondary** (oracle hit ceiling at pos-0). It compounds (3) but is not the headline.

### Is the mechanism penalty large? Honest bound (this is where the heartbeat's open question lives)

- **Leaked b50 ckpt**: oracle (isolated needle) qa1 8k/16k/32k = 51/63/50 ≈ leaked W6 50/48/34. On the *same* ckpt, isolated-hidden ≈ token-window → for qa1 the practical mechanism penalty looks **small** there.
- **Clean NOLEAK ckpt** (recomputed here from `babilong_results/noleak_oracle/`): oracle qa1 4k/8k/16k/32k = **25/23/27/26**, qa2 4k/8k/16k = 20/9/20. vs clean full-buffer W0 (`H_V2_PLAN.md:61`) qa1 8k/16k/32k = 12/8/2. So isolation **roughly 2-3× and FLATTENS the length decay** (removes dilution) — but tops out near **~25**.
- **We do NOT have a clean SWA-token (W6) number** (`HEARTBEAT_LATEST.md:26`: clean W6 OOM'd; leaked W6 is contaminated by memorized QA). So the clean magnitude of the Q1 penalty (oracle ~25 vs clean-W6 ?) is **unmeasured**. This is exactly the heartbeat's unresolved fork: "oracle gives the right chunk and still reads ~20 — is the fact *not stored* or *stored-but-unreadable*?" The mechanism above predicts a read-side penalty; whether facts are even stored is settled by the logit-lens probe (Q5).

**Q1 conclusion.** The non-equivalence is *real and provable*: FIFO is 32 query-blind single-hop reads from frozen, isolated-contextualization snapshots; SWA is a fully-coupled query-conditional multi-hop read over live tokens. This is the deepest "hidden-can't-be-read" candidate, more fundamental than pos-0. Its practical magnitude on clean ckpts is bounded above by oracle≈25 but its split between "unstored" vs "unreadable" needs the logit-lens probe.

---

## Q2 — Is hidden activated at training, and is that good?

**Path identity (no mechanism mismatch).** `_forward_fifo` is dispatched unconditionally whenever `use_fifo_memory` (`layer.py:1876`) and is **train/eval agnostic** — it only reads `self._fifo_*` attrs. NOLEAK b25 / T2 launch scripts use neither `--last_chunk_loss_only` nor `--sliding_target_loss`, so training runs `dolmino_train_step_tbptt` (`train_mem_space_dolmino_cpt.py:2573`). That step streams `context_chunks + target` (`:2636-2637`) each through `model(...)`, so during training the FIFO buffer fills and the target reads it exactly as at eval. **There is NO train/eval readout-mechanism mismatch** (answers the explicit sub-question).

**But three concrete reasons training does not teach long-range hidden retrieval:**

1. **Detached writes (`layer.py:1554`, `fifo_detach=True` default `:578`).** The buffer entries are `.detach()`-ed. When the target reads chunk-0's snapshot and produces loss, gradient flows into the *reading* machinery (unfrozen layers ≥24 + `inject_gate`, the only adapter param `:1540-1541`) but **never back into chunk-0's write**. The store side is an untrained identity. The model cannot learn to *encode query-retrievable* hidden — it can only learn to read whatever the frozen base LM happens to emit. This is the single biggest training weakness.

2. **Curriculum depth ≪ eval depth.** `--curriculum 0:3` → n_ctx=3 (~2k tokens, 3-4 chunk buffer) for the entire run. Eval qa1 16k = 32 chunks, 32k = 64 chunks. The model is trained to read a **3-deep, nearly-undiluted** buffer and tested on a **25-64-deep, heavily-diluted** one. Pos-0 collapse is also mild at depth 3 (3 chunks stacked at pos-0) but catastrophic at depth 64. The long-range dilution/position regime is **never in the training distribution**.

3. **`bptt_window=1`** → the window graph is one chunk; combined with (1), there is essentially no cross-chunk write→read credit assignment.

**Connection to slots' "SWA only at eval is best."** Same shape: the regime the model is *trained* to read (shallow, undiluted hidden) is not what produces long-range help at eval; the long-range help comes from eval-only mechanisms (SWA tokens, oracle isolation) that **bypass the trained hidden read**. This supports the view that hidden-reading-as-currently-trained contributes little to long-range.

**Q2 judgment.** Hidden IS activated at training (not harmful, the read path must exist), but training is **insufficient, not mismatched**: detached writes + depth-3 curriculum + bptt-1 mean the model never learns to store-and-retrieve at the depths eval needs. Current training does not credibly "bake in" long-range hidden retrieval.

---

## Q3 — Do we lack MemoryLLM's "dimension reduction"?

Read directly from the local source `MemoryLLM-source/modeling_memoryllm.py` (+ `models/memoryllm-8b/config.json`: `num_tokens=256, num_blocks=50, hidden=4096, L=32, drop_memory_per_layer=True, add_bos_embedding=True, add_decoder_lora=True`).

**MemoryLLM does NOT reduce dimension.** The memory pool is `self.memory = nn.Parameter([L=32, num_blocks*num_tokens = 12800, d=4096])` (`:1525`) — full 4096-dim vectors. The user's "maps hidden to lower-dim hidden" hypothesis is **not what MemoryLLM does**.

**What MemoryLLM actually does = learned COUNT-compression + trained write/read machinery:**
- To inject a chunk, it appends `num_tokens=256` learnable memory-token positions to the context, runs the injection forward, and reads the **output hidden at those 256 positions** as `delta_memory` (`:1928-1929` `all_delta_memory.append(hidden_states[:, -self.num_tokens:])`). So an arbitrary-length chunk is attention-pooled into **256 learned vectors per layer** — a Gist/ICAE-style learned compressor, NOT a projection.
- Consumption: `cat_memory_and_hiddens` (`:1673`) prepends `self.memory[idx]` (+ a learnable `bos_embedding`) as a prefix; attention uses `prefix_token_length` so the prefix is keys/values only (`:426-428`). Dedicated injection vs reading is separated by **two LoRA adapters** (`add_decoder_lora`, `:1855-1865`).
- Update: `update_memory_with_delta_memory` (`:1603`) **randomly drops** 1/num_blocks of memory (`drop_memory`, `randperm` `:1578`) and concatenates the new 256. Random eviction; N↑→better *in their eval*.
- The whole 1.67B memory pool + adapters are **trained**.

**Vs our FIFO:** we `torch.cat` **raw, untrained, detached** per-layer hidden (512 tokens × up to 64 chunks), no compressor, no dedicated inject/read adapter, FIFO append/pop instead of learned eviction.

**Is the missing piece the key to "hidden收获不大"?**
- **Dimension reduction: NO.** The oracle result (isolated raw 4096 hidden → qa1 flat ~25, up from 8-12) proves raw hidden is *addressable*; cutting dimension would *lose* information, not help. Reject the dim-reduction framing.
- **Trained write/compressor: a real architectural gap, but low odds it's "the fix" for needles.** (i) MemoryLLM is **flat ~31-37%** on BABILong (`reproducibility_survey.md:214`) — a knowledge-retention mechanism, weak/imprecise on needles; copying it likely yields MemoryLLM-like flat mediocrity, not precise retrieval. (ii) Our own lit note (`fifo_dilution_eviction_litreview_20260625.md:4`) found FIFO **refutes** MemoryLLM's N↑→better: for needles, smaller buffer wins (dilution), so MemoryLLM's random-eviction/large-N design is arguably *wrong* for our goal. (iii) Their trained MLP retriever (M+) is the same family as our already-dead gist selector (`:9`).
- **Where a learned write COULD help:** not readability (oracle says raw hidden is readable) but **selectability** — a trained write that emits well-separated, query-anticipating keys could raise chunk-selection precision, which the oracle analysis names as the *actual* bottleneck. That ties to the selection direction, not to "decompress raw hidden."

**Q3 conclusion.** We are not missing a dim-reduction step (MemoryLLM has none). We lack a *trained count-compressor + dedicated inject/read adapters*. That is a genuine gap but unlikely to be the headline fix (MemoryLLM is mediocre on needles, and our data shows its scaling claim fails for needles). If we ever add it: K learnable memory-token queries per chunk that cross-attend the chunk hidden → K trained, **non-detached** write-vectors stored instead of raw 512, plus a small trainable inject adapter — rated MEDIUM-LOW priority.

---

## Q4 — Is slot-usage visualization worth doing?

`scripts/dump_slot_usage_dist.py` reads the layer-0 `_cum_usage [B, num_slots]` tensor — the **top-k routing selection counts** of the *slot* architecture (`:64-103`). It answers a real slot-line question: routing collapse / dead slots / top-16 concentration vs the uniform `top_k/num_slots` baseline.

**Relevance to the current FIFO/hidden line: essentially none.**
- The FIFO path **bypasses ALL routing** (`layer.py:1872-1885`); it never populates `_cum_usage`. The script **cannot run on FIFO ckpts** and would diagnose only the *paused slot architecture*.
- It tells us nothing about the hidden-snapshot read mechanism, dilution, or selection precision that the current work hinges on.

**Q4 judgment.** **Skip for the hidden direction.** Only worth it if we decide to revive slots. The deeper diagnostic the heartbeat already commissioned — `probe_fifo_hidden_recall.py` (logit-lens on the needle hidden) — is the one that actually moves the hidden question, and `viz_slot_semantics.py` (vector PCA/effective-rank) is a strictly better slot diagnostic than usage-histograms anyway.

---

## Q5 — Is the hidden direction still alive? Highest-ROI next experiment (mix=0)

### Where the evidence actually lands
- **For:** isolation (oracle) lifts clean qa1 long from 8-12 → flat ~25 and removes the length-decay → dilution is real and partly fixable; raw hidden carries addressable signal; clean oracle qa1 (25/23/27/26) is *above* the current clean full-buffer profile.
- **Against / capping:** even with *perfect* selection the clean read tops at ~25; the read mechanism has a structural ceiling (Q1); training does not teach long-range storage (Q2); the leaked-ckpt oracle≈W6 suggests the *clean* low number may be a capability/storage problem, not pure dilution.
- **The decisive unknown:** is ~25 a *storage* ceiling (fact not in the hidden) or a *read* ceiling (fact present, frozen-snapshot read can't extract it)? The fix diverges completely.

### Highest-ROI, zero-training, mix=0-safe experiments (in order)

1. **★ Logit-lens hidden-recall probe** (`probe_fifo_hidden_recall.py`, already commissioned to coder a8a3799) on the clean NOLEAK ckpt. For each needle chunk, project the needle-token hidden at each layer through the unembedding and check if the answer token is decodable.
   - Decodable but oracle still fails → **read-side** ceiling confirmed (Q1 mechanism is the wall) → fix = give the prefix query-conditional re-contextualization (a small cross-attn that lets the query reshape the prefix before reading), or accept the proven SWA-token path.
   - NOT decodable → **write-side**: raw detached hidden doesn't preserve the fact at the needed layers → only then is a trained compressor (Q3) or a pivot justified.
   - This single probe **decides the hidden direction's fate** and costs no training.

2. **Clean SWA-token W6 eval on NOLEAK** (rerun the OOM'd one at batch_size 1 / fewer new tokens). Gives the missing clean **oracle(~25) vs clean-W6(?)** gap = the true magnitude of the Q1 hidden-vs-token penalty. Eval-only, mix=0-safe.

3. **Gate, do not pre-launch, train-time selection.** The train-time reader-native top-k isolation (coder a37179) is only worth running **if probe #1 shows facts are readable** — otherwise you'd learn to select the right chunk and still be unable to read it.

### Bottom line
The hidden direction is **conditionally alive but capped**: oracle proves there is real, currently-unrealized headroom (flat ~25 on qa1 long if selection were solved), but the frozen-snapshot read mechanism (Q1) is a genuine structural ceiling and current training (Q2) does not approach it. **Do not spend more GPU on hidden *training* until the logit-lens probe resolves stored-vs-unreadable.** If the probe says "stored but unreadable," the honest move is either a query-conditional re-contextualization read (a real architectural change, not just "select better") or conceding that the precise-needle long-range win lives on the SWA-token / selection side, not the raw-hidden side. MemoryLLM-style compression and slot-usage viz are both low-odds for this goal.

**Confidence:** Q1 mechanism HIGH, Q1 clean magnitude LOW-MED (probe pending); Q2 MED-HIGH; Q3 MED-HIGH; Q4 HIGH; Q5 MED (lean: limited headroom, gate on probe, be ready to pivot).

---

### Appendix — key citations
- SWA window (raw tokens, full forward): `scripts/run_babilong_mem_space.py:577-583`, generation `:584-597`.
- FIFO read (frozen per-layer snapshot prefix): `src/memory/mem_space/layer.py:1338-1342`; write (detached layer-input) `:1505,:1554`; per-layer buffer reset `train_mem_space_dolmino_cpt.py:662-663`; pos-0 collapse `layer.py:1392-1401`; dispatch (train/eval agnostic) `:1876-1885`; only trainable FIFO param `:1540-1541`.
- Training step (tbptt, default): `train_mem_space_dolmino_cpt.py:2573-2772` (stream `:2636-2637`, backward/detach `:2762-2769`); curriculum `0:3`/`bptt_window 1` from `scripts/launch_mem_space_fifo_b25_chunk512_NOLEAK_diskB.sh` + `scripts/_launch_t2_posfix.sh`; partial unfreeze (≥24) `:551-569`.
- MemoryLLM: pool `MemoryLLM-source/modeling_memoryllm.py:1525`; delta read `:1928-1929`; consume `:1673-1745`; random eviction `:1573-1601`; LoRA adapters `:1855-1865`; config `models/memoryllm-8b/config.json`. Flat BABILong: `ops/research_notes/2026-05-11_memory_papers_reproducibility_survey.md:214`. N↑ refutation: `ops/research_notes/fifo_dilution_eviction_litreview_20260625.md:4,9`.
- Slot-usage viz (routing-only, FIFO-incompatible): `scripts/dump_slot_usage_dist.py:64-103`.
- Clean numbers: oracle recomputed from `babilong_results/noleak_oracle/` (qa1 25/23/27/26); clean full-buffer W0 `status/H_V2_PLAN.md:61`; clean W6 unavailable `status/HEARTBEAT_LATEST.md:26`; oracle analysis `status/FIFO_ORACLE_ANALYSIS_20260626.md`.
