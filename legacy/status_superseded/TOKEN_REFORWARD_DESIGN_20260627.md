# Token-Reforward Memory — deployable design (2026-06-27)

> Author: researcher subagent. Read-only, no GPU. All citations `file:line`.
> Brief: oracle-TOKEN reforward broke qa1 (8k=50/16k=28 vs hidden-oracle 20/24,
> baseline 12/8). The win is "re-forward the SELECTED chunk's *raw tokens*"
> (live, query-conditional, all-layer) instead of reading a frozen hidden
> snapshot. Oracle uses the gold answer to locate the chunk → not deployable.
> Design a deployable (selector-driven) token-reforward readout, train+eval
> consistent, `mix=0`-trainable. Decide MVP: probe vs train.

---

## TL;DR

| # | Answer | conf |
|---|---|---|
| **Arch** | Store **raw token ids per chunk** (+ one selection-layer key cache). Select chunks by **reader-native q·k** over the already-streamed FIFO hidden (no trained selector). Re-forward **selected chunks' raw tokens ∥ last chunk** through the full backbone — this is *byte-identical to the existing `oracle_token_chunks` window path* (`run_babilong_mem_space.py:570-581`), only the chunk source changes. Positions are **packed for free** (concat → contiguous RoPE), exactly as the breakthrough ran. | HIGH |
| **Selection** | The end-to-end cap. Best near-term lever = **top-k=4-6 + recency floor** (token-reforward tolerates distractors far better than frozen hidden, so trading a little window length for recall is cheap). Train-time reader-attn-select + reforward (STE-style, indices non-diff, selected tokens carry grad) is phase-2, *only if* the probe shows selection is the ceiling. Two-stage coarse→fine is over-engineering for k≤6; defer. | MED-HIGH |
| **MVP** | **PROBE FIRST, zero training.** Add `--swa_readerattn_token`: swap `_locate_needle_chunks` for a reader-attn top-k selection computed inside `generate_with_mem_space`, feed the chosen indices into the *existing* token-window builder. Measures the deployable upper bound (= selection precision × token-reforward readout). ~80-120 LOC, no new training, decisive. Only train if probe lands meaningfully above the hidden baseline (>20s). | HIGH |
| **Risk** | 32k: reader-attn top-2 recall is only ~40% (`RUN_REGISTRY.md:1290`) → probe will likely show strong 8k, decaying 16k, weak 32k. Speed: window re-forwarded every step with `use_cache=False` (existing pattern) → k=4 @32k = 5×512=2560 tok × 20 steps, slow-but-ran for oracle. Train compat: FIFO hidden buffer stays (now only for *selection keys*); readout reuses the SWA-window path already in the trainer. | MED |

---

## 1. Deployable token-reforward architecture (concrete enough to implement)

### Mechanism recap (why this is the right target)
- **Hidden FIFO read** (`_forward_fifo`, `layer.py:1262-1564`): each kept chunk is a
  *detached, per-layer, query-blind* snapshot frozen at write time
  (`layer.py:1554`). The reader attends it as a prefix but the snapshot never
  re-attends the query → query-side accumulation only, no needle↔query
  coupling. Caps at ~20 even with perfect (oracle) selection
  (`HIDDEN_VS_SWA_ANALYSIS_20260626.md:15`).
- **Token reforward** (`oracle_token_chunks` branch, `run_babilong_mem_space.py:570-581`):
  the selected chunk's **raw token ids** are concatenated with the last chunk and
  run through a **full forward**. Every layer recomputes the needle's K/V *after*
  it has attended the query (and vice-versa) → full bidirectional, all-32-layer,
  query-conditional retrieval. qa1 8k 12→50.

The deployable system keeps the FIFO buffer ONLY as a **selection index** and
moves the **readout** to token reforward.

### (a) What to store
- **Primary: raw token ids per chunk.** Trivially cheap (~512 int32/chunk →
  2 KB/chunk; 64 chunks @32k = 128 KB/sample). This is the re-forward payload.
  We already split the doc into `chunks = tokens.split(chunk_size)`
  (`run_babilong_mem_space.py:556`), so the ids exist; just retain them
  (the eval path already keeps the full `chunks` list in memory).
- **Selection keys.** Selection needs a per-chunk key to score against the query.
  Two options:
  - **(probe) reuse the existing per-layer FIFO hidden buffer** `_fifo_buf`
    (`layer.py:1300-1304`). Streaming already fills it; the selection layer's
    stored hidden = its keys. Zero new storage.
  - **(production) store only ONE selection-layer key block per chunk**
    (e.g. layer 16: `[chunk_len, d]` pre-`k_proj` hidden, or the post-`k_proj`
    key `[n_kv, chunk_len, hd]`). Keeping 32-layer hidden is wasteful; one layer
    is enough for a single model-level select. ~4 MB/chunk for raw hidden @512×4096×bf16,
    so for long docs pool the chunk to a few summary keys (attn-pool to 8-16 tokens)
    if memory matters; for ≤64 chunks raw is tolerable.
  - **Do NOT store all-32-layer hidden** (current FIFO) once readout is token-based —
    it is dead weight after the switch.

### (b) How to select chunks — reader-native q·k, NOT a trained selector
The project has a wall of dead trained selectors (`RUN_REGISTRY.md:1250-1259`;
gist scorer needle-precision 0-10% = at/below random). The **only** working
selector is the reader's *own* native attention: `RUN_REGISTRY.md:1256-1258`
reader-attn needle precision **55% = 8.8× random**; top-2 chunk-hit at chunk512
scale = **72/78/58/40%** @4k/8k/16k/32k (`RUN_REGISTRY.md:1290`). Reuse it.

The existing per-layer scorer `_fifo_select_keep_set_reader_attn`
(`layer.py:1612-1683`) computes exactly the needed score:
`score(c) = mean_q max_t (q_q · k_t)/√hd`, heads pooled by amax, top-K + recency
floor. **One change of scope:** the per-layer keepset picks a different set per
layer (fine for a hidden prefix); the token reforward needs **one model-level
chunk set** to build the window. So pick a single representative selection layer
(suggest L≈16, the mid/injection band — `RUN_REGISTRY.md:1264` found L16-24
isolation = 97.5%) — or vote across a few layers (sum salience over {12,16,20,24}).
Start with one layer; add voting only if precision is the cap.

### (c) How the selected chunks enter attention — TRUE reforward, not hidden prefix
**Re-forward the raw tokens through the backbone**, i.e. the existing
`oracle_token_chunks` window:
```python
sel    = sorted(c for c in selected_chunks if 0 <= c < last_idx)
pieces = [chunks[c] for c in sel] + [chunks[-1]]   # raw token ids
window = torch.cat(pieces, dim=0).unsqueeze(0)     # [1, (|sel|+1)*chunk_size]
... model(input_ids=window) for each decode step   # full forward
```
This is the breakthrough path verbatim (`run_babilong_mem_space.py:577-581`).
NOT a hidden prefix (that is the thing we are escaping). The earlier
(non-selected) chunks still stream into the bank exactly as W0 — the window is
purely additive direct attention, no double counting.

### (d) Positions — packed for free
Concatenating raw chunks and running one forward gives the window contiguous
RoPE positions `0..L-1` automatically (no special handling). This is exactly how
the oracle-token path scored the breakthrough — packed, in-distribution within
the trained window. Window length for k selected chunks = `(k+1)*512`; for k=2
(3 chunks) = 1536 = the training curriculum `0:3` n_ctx, i.e. **in-distribution**;
k=4-6 (2560-3584) is mild RoPE extrapolation (Llama-3 theta=500000 handles it,
same as the SWA-W6 path which already attends 3072 tokens and scores ~50).

### Train/eval consistency
- **Eval** = §1(a-d) above.
- **Train** = the trainer's existing SWA-window machinery
  (`_make_swa_window`, `train_mem_space_dolmino_cpt.py:2411-2426` /
  `dolmino_train_step_tbptt:2665-2702`) already concatenates *last-W context
  chunks ∥ target* and masks the prefix labels to -100. The ONLY change is to
  pick the W slots by **reader-attn selection** instead of "last W". The selected
  chunks' tokens carry gradient through the forward (the read side learns); the
  index choice is non-diff (STE: identical to `fifo_train_keep_set_mode
  flat_readerattn`'s contract, `train_mem_space_dolmino_cpt.py:1606-1610`).
  This is `mix=0`-trainable on generic dolmino text — the recall pressure is the
  streaming objective (`dolmino_train_step`: context no-grad, loss only on
  target → target can only reach context through the window), no QA/babilong
  needed.

---

## 2. Selection precision — the bottleneck, and how to attack it

**Framing fix:** the heartbeat's "reader-attn keepset qa1 = 9 vs oracle 50"
(`FIFO_ORACLE_ANALYSIS_20260626.md:74,113`) is **end-to-end on the HIDDEN path**
— it conflates (i) approximate selection with (ii) the weak frozen-hidden
readout. Token reforward fixes (ii). So the deployable number is **selection
precision × token-reforward readout**, and the open question is purely: how
much does swapping oracle-location → reader-attn cost, given a *good* readout?
The probe (§3) measures this directly. Levers, evaluated:

**(a) Train reader-attn-select + reforward (STE/gumbel).** Make the backbone's
own q/k learn to be selective AND to read what it selected.
- Pro: it is the *same weights* that select and read — unlike the death-listed
  bolt-on gist selectors, there is a coherent gradient ("encode chunks so my own
  attention finds them, then read them"). STE is already the trainer's contract.
- Con: more expensive; only worth it if the probe says selection is the cap.
- **Verdict: phase-2, gated on the probe.** Gumbel adds noise/instability over
  hard-top-k STE with little expected gain here; prefer STE.

**(b) Larger top-k (recall vs dilution).** *This is the cheap winner.* Key
asymmetry: the frozen-hidden path collapses under added chunks (full-haystack =
0%, `RUN_REGISTRY.md:1265`) because dilution kills the softmax; **token
reforward does NOT** — the SWA-W6 path attends 6×512=3072 distractor tokens and
still scores ~50. So adding a few wrong chunks costs little, while raising k from
2→4-6 lifts recall (top-2 recall 40% @32k → top-4/6 materially higher). Cost is
only window length (OOD/speed), bounded. **Recommend k=4-6 + recency floor 1-2.**
- Verdict: do this *in the probe itself* (sweep k ∈ {2,4,6}).

**(c) Two-stage coarse→fine.** Coarse-select many chunks, then a second finer
read. Justified only if a single k-of-N flat select plateaus AND k must stay
small for cost. Since (b) shows token reforward tolerates k≤6 cheaply, two-stage
buys little now. **Defer** (it is the HNST direction, `vN_HNST_tree...`, already
ranked low-med pending #1 signal, `CLEAN_SOTA_SURVEY_20260625.md:115`).

**Priority: (b) now (free, inside the probe) → (a) if probe shows the cap is
selection → (c) only if (a) plateaus.**

---

## 3. MVP — the single highest-ROI next step: the reader-attn-token PROBE

**Do the probe, not training.** Rationale: it is zero-train, reuses the proven
token-reforward path, and *decides whether to train at all*. It answers "where
between 20 (hidden) and 50 (oracle-token) does the deployable (reader-attn)
version land?" — i.e. how much of the breakthrough survives losing the oracle.
If it lands well above 20, train; if it collapses to ~hidden, selection is the
wall and we attack §2(a) before any expensive readout training.

### Probe spec (for coder)
**Goal:** swap the oracle needle-locator for reader-attn selection; keep the
existing token-reforward window + decode loop unchanged.

1. **New CLI flag** in `run_babilong_mem_space.py` (parallel to `--swa_oracle_token`,
   `:877`): `--swa_readerattn_token` + `--swa_select_layer` (default 16),
   `--swa_select_topk` (default 4), `--swa_select_recency` (default 1).
   Require `--batch_size 1` (same guard as oracle-token, `:1030`).

2. **Selection inside `generate_with_mem_space`** (`:476`). After the streaming
   loop fills `_fifo_buf` (`:562-565`) and banks are frozen (`:568`), and *before*
   building the window (`:570`), when the new flag is set:
   - One extra forward of the last chunk with `output_hidden_states=True` to get
     the query hidden at the selection layer `L`:
     `q_hidden = out.hidden_states[L]` (input to layer L = output of L-1; this
     matches the FIFO write convention `h_stored` = layer input, `layer.py:1554`).
     NOTE side effect: this forward FIFO-writes the last chunk into `_fifo_buf`
     (write is not frozen-gated, `layer.py:1554`); harmless because selection
     excludes `last_idx` and the buffer resets next sample. (Optionally add a
     `_skip_fifo_write_this_call` guard for cleanliness.)
   - Resolve cos/sin for the last-chunk positions via `model.model.rotary_emb`
     (the layer already has `_fifo_resolve_rotary_emb`, `layer.py:1571`).
   - Call the selection-layer wrapper's existing scorer:
     `kept = sel_wrapper._fifo_select_keep_set_reader_attn(hidden_states=q_hidden,
       valid_chunks=[_fifo_buf entries for chunks 0..last_idx-1],
       position_embeddings=(cos,sin), topk=K, recency=R)`
     → it returns sorted local indices; map to document-absolute chunk indices.
   - Set `oracle_token_chunks = <those abs indices>` and fall through to the
     **unchanged** window builder (`:577-581`) + decode loop (`:609-622`).
   (Alternative if reusing the per-layer method is awkward: lift its ~30-line
   q·k scoring to a module-level `_select_chunks_reader_attn(...)` helper; same math.)

3. **Wiring in the bsz=1 loop** (`:1284-1320`): when `--swa_readerattn_token`,
   pass the flag/params into `generate_with_mem_space` instead of computing
   `_oracle_tok`. (Selection now lives inside generate because it needs the
   streamed hidden, so the call site just forwards the params.)

**Run config (clean NOLEAK ckpt only, mix=0):**
```
--swa_readerattn_token --swa_select_layer 16 --swa_select_topk {2,4,6}
--swa_select_recency 1 --batch_size 1
tasks qa1 (then qa5); lengths 8k,16k,32k; n=100; score_nested
```
Compare against the three anchors already in hand: hidden baseline (qa1 8k=12),
hidden-oracle (~20), oracle-token (50). Decision gate:
- **reader-attn-token ≥ ~35 @8k** → selection survives; **train** the §1
  token-reforward readout (reader-attn-select, mix=0). Expect to approach oracle.
- **reader-attn-token ≈ 20-30** → selection is the cap → do §2(a) (train-time
  reader-attn-select) *first*, before betting on readout training.
- **≈ hidden baseline** → reader-attn selection too weak at scale → §2(b) larger
  k didn't save it → escalate to multi-layer voting / HNST.

**Code change estimate:** ~80-120 LOC, 1 file (`run_babilong_mem_space.py`)
+ optional 1 tiny guard in `layer.py`. Reuses: token-window builder, decode
loop, `_fifo_select_keep_set_reader_attn`, `_fifo_resolve_rotary_emb`. No
training, no new dataset, runs on existing clean ckpt. **Low risk, high info.**

---

## 4. Risks & compatibility

- **Selection precision decays with depth (the headline risk).** Reader-attn
  top-2 chunk recall = 72/78/58/**40%** @4k/8k/16k/32k (`RUN_REGISTRY.md:1290`).
  So the probe will most likely show a strong 8k, softer 16k, weak 32k — the
  long-doc win is selection-bound. Mitigation already in the probe: sweep k=4-6
  (cheap recall gain, token reforward tolerates the extra chunks). Expectation:
  qa1 8k lands 30-45 (between hidden 20 and oracle 50); 32k more uncertain.
- **Speed.** Window is re-forwarded every decode step with `use_cache=False`
  (existing oracle-token behaviour, `:611`). k=4 @32k = 5×512=2560 tok × 20 steps.
  Oracle-token already ran this; tolerable for a probe. **Production must add a
  KV cache for the window** (re-forward once, then incremental decode) — out of
  scope for the MVP but flag it.
- **How many chunks survive top-k at 32k?** 64 chunks → select k=4 → window =
  2560 tok. No memory problem (≪ the 50-chunk hidden buffer's footprint). The
  constraint is *recall* (above), not compute/memory.
- **FIFO training-framework compatibility.** The FIFO hidden buffer
  (`_fifo_buf`) is retained but **demoted to a selection index** (keys only);
  readout moves to the SWA-window path the trainer already has
  (`_make_swa_window`, tbptt SWA, `:2665-2702`). Train change = pick window
  slots by reader-attn instead of last-W; STE contract identical to existing
  `fifo_train_keep_set_mode flat_readerattn` (`:1606-1610`). `mix=0`-trainable:
  yes (generic dolmino streaming objective, no QA). Curriculum `0:3`/`bptt 1`
  (`launch_...NOLEAK...:37`) trains a 3-chunk window = k=2 in-distribution; to
  train larger k, widen curriculum (a real but bounded change — and the same
  dilution-regime-mismatch the analysis flagged for hidden, `HIDDEN_VS_SWA...:78`).
- **qa5 (multi-hop) caveat.** Oracle-token qa5 was a *locating* artifact
  (`_locate_needle_chunks` finds only the answer-token chunk, not the reasoning
  chain — `HEARTBEAT_LATEST.md:18`). The reader-attn probe does NOT have this
  bug (it scores ALL chunks by query salience, can pick multiple supporting
  facts via top-k), so the probe is actually a *fairer* qa5 test than
  oracle-token. Run qa5 in the probe too — it may look better than oracle-token's
  10.

---

## Confidence

- Architecture (§1): **HIGH** — it is the breakthrough path with the chunk
  source swapped; every piece already exists and is cited.
- Selection levers (§2): **MED-HIGH** — (b) larger-k rests on the
  token-reforward-tolerates-distractors asymmetry (well-evidenced: SWA-W6 attends
  3072 tok and scores ~50); (a) train-time is sound but unproven for this exact
  readout.
- MVP probe (§3): **HIGH** that it is the right next step (cheap, decisive,
  gates training); **MED** on the magnitude it will return (selection-precision
  bound, esp. 32k).
- Overall recommendation: **run the probe before any training.** It is the
  cheapest experiment that can either justify or kill the token-reforward
  training spend.
