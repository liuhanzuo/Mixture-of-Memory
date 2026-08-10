# FIFO-oracle probe — deep analysis & purity adjudication (2026-06-26)

> Author: researcher subagent. Read-only (no GPU jobs run). All code citations are `file:line`.
> Question: does the new FIFO-path oracle probe genuinely support *"raw hidden is addressable once isolated, dilution is the killer"* — or is it contaminated by the keep-all fallback the run log warns about?

---

## TL;DR verdict

| # | Question | Verdict |
|---|---|---|
| (a) | Fallback purity | **Signal is CREDIBLE (trustworthy).** The fallback is a *streaming-order artifact*, NOT a bug, NOT keep_all_buffer failing, NOT needle-location failure, NOT an abs_idx bug. It fires only during the buffer build-up (before the needle chunk has been appended); at **readout (generation) time the needle is in the buffer and isolation is 100% active**. ~0% of samples have a corrupted readout at 8k/16k. |
| (b) | "raw hidden addressable" | **Holds.** Oracle isolation lifts qa1 8k/16k/32k from full-buffer-W0 27/36/24 (and clean NOLEAK 12/8/2) to **51/63/50 ≈ the W6 open-book ceiling 50/48/34**. Dilution is the dominant gap driver; position-collapse is secondary (oracle ran at legacy pos-0 and still hit the ceiling). |
| (c) | vs rawkv-oracle | **Different mechanism.** rawkv-oracle = single-layer EV-*injection* interface the frozen reader never learned to consume (+2.5pt). FIFO-oracle = the needle chunk sits in the **native, trained, all-32-layer FIFO attention prefix** → the reader uses it. This is the same effect as the Method-A Level-1 isolation (97.5%). |
| (d) | Next step | **Selection precision is now the bottleneck, NOT readout.** Oracle (perfect selection) = 51 vs reader-attn keep-set (approx selection) = 9 on qa1 8k. Direction: improve chunk-selection precision (better selector / hierarchical navigation / train-time isolation), since readout is proven capable. |

---

## (a) Fallback purity adjudication — CREDIBLE

### What the counter actually counts

The oracle keep-set is `_fifo_select_keep_set_oracle` (`src/memory/mem_space/layer.py:1685-1734`). It has **two** fallback branches:

1. **needle unknown** (`layer.py:1715-1719`): `needle_abs` is None/empty → `_fifo_oracle_fallback_count += 1` only.
2. **needle not in buffer** (`layer.py:1725-1729`): no kept buffer entry carries a needle abs-idx → `_fifo_oracle_fallback_count += 1` **and** `_fifo_oracle_evicted_count += 1`.

The run report is `fb (of which needle-evicted: ev)` (`scripts/run_babilong_mem_space.py:1373-1378`). **In every shard log `fb == ev` exactly** (e.g. qa1 8k shard0 `6784 (of which needle-evicted: 6784)`; qa2 8k shard0 `16928 / 16928`). That equality is the key tell:

- Branch 1 (needle unknown) fired **zero** times across all shards → `_locate_needle_chunks` (`run_babilong_mem_space.py:411-440`) **located the needle in 100% of samples**. So this is **NOT** a needle-location failure.

### The "needle-evicted" label is a misnomer — it's "needle-not-yet-streamed"

The decisive arithmetic. b50 buffer = 50 chunks, chunk_size 512, **all 32 layers** are FIFO-wrapped (`patch.py:81-82` + `run_babilong_mem_space.py:237` `layer_indices=None`), bsz=1, max_new_tokens=20.

Generation streams `chunks[:-1]` into the buffer one forward at a time (`run_babilong_mem_space.py:549-552`), and the write happens at the **end** of each `_forward_fifo` (`layer.py:1553-1555`). So when forward *i* reads, the buffer holds chunks `0..i-1` — **the needle chunk at absolute depth k is not readable until forward i = k+1**. For forwards `i = 1..k` the oracle correctly finds no needle and keeps-all → that is the entire fallback.

Per-shard fallback ÷ (25 samples × 32 layers) = fallback-firing forwards per sample, which should equal the needle depth k:

| length | fb/shard (qa1) | fwds-with-fallback/sample | chunks C | implied needle depth k/C |
|---|---|---|---|---|
| 4k | ~3500 | 4.3–4.6 | ~8 | ~60% |
| 8k | 6784–7808 | 8.5–9.8 | ~16 | ~57% |
| 16k | 14240–15744 | 17.8–19.7 | ~32 | ~59% |

The fallback count scales **linearly with document length** and lands at a constant **~57-60% of the stream** — exactly bAbI's roughly-middle needle placement. This is the signature of "needle is a future chunk during early streaming," not of a bug.

**Three facts that independently kill the "eviction / keep_all_buffer broken" reading:**

1. At **8k/16k, C (≈16/32) < buffer (50)** → eviction is *physically impossible* regardless of the flag, yet the "evicted" counter is large. So the counter must be counting not-yet-arrived, not real eviction. (The log printed `fifo_keep_all_buffer=True (eviction suppressed)` — `run_babilong_mem_space.py:1130-1132` — so the flag *was* wired; it's just irrelevant at these lengths.)
2. `_reset_banks` does not touch `_fifo_buf` (`run_babilong_mem_space.py:98-129`); the per-sample reset of `_fifo_buf`/`_fifo_buf_abs_idx`/`_fifo_write_seq=0` comes from `_set_fifo_oracle_needle` (`run_babilong_mem_space.py:443-460`), called per sample at `:1256-1260`. abs-idx bookkeeping is maintained correctly (`layer.py:1248-1260`).
3. If the abs_idx mapping were buggy, isolation could never lift scores. It does (27→51). So the mapping works at readout.

### Why the fallback does NOT pollute the readout

The **answer is produced in the generation phase** (`run_babilong_mem_space.py:584-597`), processing the last chunk. By then **all** chunks — including the needle — are in the buffer, so the generation forwards run **with isolation active** (needle chunk + last-`recency` floor only) and contribute **zero** to the fallback count. Per 8k sample: ~272 fallback layer-calls (all streaming, pre-needle) vs ~848 isolation-active layer-calls (post-needle streaming + ~20 generation steps × 32 layers) — the readout is fully isolated.

Moreover, the needle chunk's *stored* hidden was computed at streaming forward k under keep-all over chunks `0..k-1` — i.e. the **standard FIFO write** (chunks attend their own causal history; `h_stored` is the layer *input*, `layer.py:1554`). So the needle representation that reaches the readout is the normal one; isolation only changes *which* chunks the reader attends at answer time.

**Sample-proportion estimate (the unit that matters):** 100% of samples log *some* streaming fallback (every needle starts as a future chunk — unavoidable), but **≈0% of samples have a corrupted readout** at 8k/16k (C<buffer, needle always present at generation; keep_all_buffer additionally guarantees it at 32k where C≈64>50). The "6880-8160 per shard" figure = 25 samples × 32 layers × ~8.6 pre-needle streaming forwards — a per-forward build-up count, not a per-sample readout-contamination count.

**Verdict: signal CREDIBLE.** The high oracle scores reflect genuine readout-time isolation. The log warning is real-but-benign: it should read "needle-not-yet-in-buffer-during-streaming," and the counter at `layer.py:1727-1728` over-labels build-up forwards as "evicted."

---

## (b) "raw hidden is addressable once isolated" — HOLDS

All on the **same leaked b50 ckpt** (`outputs/mem_space_fifo_b50_chunk512/full_model.pt`), W0, so the relevant comparison is the **delta vs the same-ckpt baseline**, not absolute scores.

| qa1 (W0) | 4k | 8k | 16k | 32k | source |
|---|---|---|---|---|---|
| FIFO-oracle (isolate needle, pos-0) | 98 | **51** | **63** | **50** | `babilong_results/probe_b50_oracle` |
| full-buffer baseline (attend all) | 96 | 27 | 36 | 24 | `babilong_results/fifo_b50_c512_final_W0` |
| reader-attn keep-set top25 (P2) | 69 | 9 | 9 | 2 | `babilong_results/probe_b50_P2_keepset25` |
| W6 open-book (last-6 raw SWA) | 68 | 50 | 48 | 34 | `babilong_results/fifo_b50_c512_final_W6` |
| clean NOLEAK b25 W0 (no leak) | — | 12 | 8 | 2 | `status/HEARTBEAT_LATEST.md:23` |

qa2 oracle 4k/8k = 100/47 vs full-buffer 95/26. (qa5 oracle not yet produced — all `--` in the score table.)

**Interpretation:**
- Isolation lifts qa1 8k **27 → 51** and **meets/exceeds the W6 open-book ceiling** (50/48/34 at 8k/16k/32k). This is the FIFO reproduction of the Method-A Level-1 result (isolated needle chunk = 97.5%, full-haystack = 0%, `RUN_REGISTRY.md:1261-1267`): **the wall is dilution, not a lossy/unusable representation.** The raw hidden is addressable.
- **Position-collapse is secondary.** The oracle probe ran with `fifo_pos_mode=none` (no `FIFO probe: fifo_pos_mode=...` line in any oracle log → legacy all-prefix-at-pos-0, `layer.py:1392-1401`). Even with the needle chunk's tokens collapsed to RoPE pos-0, isolation alone recovers the open-book ceiling. So once distractors are removed, the residual position degeneracy (one needle chunk + 2 recency chunks at pos-0) costs little. This is direct evidence for **H_DIL ≫ H_POS** as the gap driver.

This is the first *positive*, mechanistically-clean result in the FIFO line: closing dilution closes the W0/W6 gap.

⚠️ Caveat (unchanged from the survey): absolute numbers are on a **leaked** ckpt, so 4k=98 is partly memorized. But 8k-32k is the OOD-relatively-clean band, and the *delta* (isolation vs same-ckpt baseline) is the load-bearing quantity. The clean re-run on NOLEAK b25 is still the final judge.

---

## (c) Mechanism difference vs historical rawkv-oracle

| | historical rawkv-oracle (REJECTED, +2.5) | FIFO-oracle (now, +24 over baseline) |
|---|---|---|
| path | EV-prefix **injection** of pre-captured needle hidden (`layer.py:2777-2812`, `_oracle_hidden_by_layer` → `_ev_parts`) | needle chunk sits in the **native FIFO prefix** `extended_hidden = [kept_chunks | current_H]` (`layer.py:1338-1342`), read by the wrapped decoder's **own** attention (`layer.py:1509-1518`) |
| layers | single layer (L16) / a few injection layers | **all 32 layers** (every FIFO-wrapped layer isolates + reads) |
| trained-for? | a foreign interface the frozen reader never learned to consume | the **exact path the model was trained on** (FIFO buffer prefix) |
| result | oracle_pos0 +2.5 (within ±3 noise), oracle_realpos −2.5 (`RUN_REGISTRY.md:1056-1059`) | qa1 8k 27→51 ≈ W6 |

The historical conclusion ("frozen reader can't *use* injected KV even when it's the gold evidence," `RUN_REGISTRY.md:1058-1059`, `:1070`, `:1086`) was a property of the **injection interface + single-layer readout**, not of raw hidden per se. Two project priors converge on why FIFO-oracle escapes it:

1. **All-layer distributed readout.** Single-layer injection = 0% long-range was already a death-listed result (`HANDOFF_NEW_AGENT_20260625.md:228`); Landmark's cross-doc retrieval is explicitly "32-layer distributed readout, single-layer = dead" (`RUN_REGISTRY.md:1159`). FIFO is inherently all-layer.
2. **Native vs bolt-on.** The FIFO reader was trained to attend its buffer prefix; the rawkv EV-prefix was a post-hoc graft the (frozen / lightly-tuned) reader had no objective to consume (`TRAINER_ACTIVITY.jsonl` v2-SFT note: "model has no reason to learn to attend injected KV").

So the FIFO-oracle gain is consistent with — and explained by — the existing evidence; it is not in tension with the rawkv negative.

---

## (d) Implication for next direction

**The bottleneck has moved from readout to selection.** Two clean facts on qa1 8k:

- Oracle (perfect chunk selection) = **51** ≈ W6 ceiling → **readout is NOT the problem** (reverses the old rawkv "readout is broken" conclusion *for the FIFO native path*).
- reader-attn keep-set top-25 (approximate selection) = **9**, even below the full-buffer baseline 27 → approximate selection on qa1's single-fact natural-language needle is failing.

This matches the prior reader-attn precision data at 512-chunk scale (top-2 hit 72/78/58/40% at 4k/8k/16k/32k, `RUN_REGISTRY.md:1290`) and the survey's ranking: dilution + selection is direction #1.

**Recommended priority order:**

1. **(first, free) Run the oracle on the CLEAN NOLEAK b25 ckpt** the moment it/eval lands — confirm the 27→51 lift survives without leakage. This is the real verdict and costs no training. Use `--fifo_keep_set_mode oracle --fifo_keep_all_buffer --batch_size 1`, and **also add `--fifo_keep_recency 2`** (the question chunk).
2. **Chunk-selection precision is the live lever** — pursue in this order:
   - **(a) better reader-native selection** (the only working selector, 55% = 8.8× random, `RUN_REGISTRY.md:1256-1258`): top-k tuning, per-layer 32×heads voting, two-stage grouped-softmax (within-block) — all H2-safe, no trained selector.
   - **(b) train-time isolation** (highest expected payoff): bake "keep-all-store, attend-few (reader-native top-k)" into the **training** forward with `--babilong_mix_fraction 0`, so the reader *learns* to read an isolated prefix instead of a diluted one. Oracle proves the readout ceiling is high (~W6); training the selection-conditioned read should approach it.
   - **(c) hierarchical navigation (HNST)** only if flat reader-attn plateaus — it directly attacks the flat-1-of-64 precision collapse.
   - **Do NOT** revisit evidence-injection / single-layer readout / trained flat gist selectors (death-listed, `HANDOFF_NEW_AGENT_20260625.md:223-234`).
3. **Position-fix (packed RoPE) is demoted to secondary.** The oracle hit the ceiling at pos-0, so position is not the dominant gap driver. Worth keeping as a stacked add-on for qa5 (temporal ordering may still want preserved order), but it is not the headline fix.

**Confidence:** the purity adjudication (a) and the mechanism contrast (c) are **high** (direct from logs + code). The "selection is now the bottleneck" conclusion (d) is **medium-high** — solid on the leaked ckpt and consistent with priors, pending the clean NOLEAK oracle re-run and qa5 oracle numbers.

---

### Appendix — key citations
- Oracle keep-set + fallback counters: `src/memory/mem_space/layer.py:1685-1734` (`fb==ev` branch at :1725-1729; needle-unknown branch at :1715-1719).
- Per-sample needle wiring + buffer reset: `scripts/run_babilong_mem_space.py:411-460` (locate), `:1256-1260` (per-sample), `:1373-1378` (report).
- FIFO buffer write (end-of-forward, layer input) + abs-idx bookkeeping: `layer.py:1248-1260`, `:1553-1555`.
- Native prefix read (all-layer, trained path): `layer.py:1338-1342`, `:1509-1518`; pos-0 legacy at `:1392-1401`; dispatch `:1876-1885`.
- rawkv EV-injection oracle (single-layer interface): `layer.py:2777-2812`; result `RUN_REGISTRY.md:1056-1059`, `:1070`, `:1086`.
- Method-A dilution (0% vs 97.5% isolated): `RUN_REGISTRY.md:1261-1267`. Reader-attn precision: `:1256-1258`, `:1290`.
- Scores: `babilong_results/{probe_b50_oracle, probe_b50_P2_keepset25, fifo_b50_c512_final_W0, fifo_b50_c512_final_W6}`; fallback logs `logs/eval_probe_b50_oracle_taskpool/probe_b50_oracle_qa1_{4k,8k,16k}_shard*of4.log`.
