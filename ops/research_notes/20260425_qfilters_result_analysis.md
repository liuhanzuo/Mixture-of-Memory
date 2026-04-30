# Q-Filters Beating Vanilla Llama-2-7B on pg19 — Analysis (2026-04-25)

**Author**: researcher (autonomous chain)
**Target result**: Q-Filters PPL 3624.64 vs Vanilla 5102.22 on pg19, 200×4096, bf16, sdpa, `kv_budget=512, filter_rank=2, recent_window=64, calib=64`
**Smoke confirmed direction**: 10-chunk run PPL 3142.72

---

## 1. Is this plausible?

Direct verification of the paper tables was blocked (arxiv.org/github.com WebFetch both refused in this sandbox). From secondary summaries (emergentmind, itinai, aibtz, x-mol, all surfaced via WebSearch), the paper's own headline figures are:

- ~99% NIH at **32× compression** on Llama-3.1-70B
- "up to 65% reduction in PPL-drop vs Streaming-LLM"
- Paper tests **Llama-3.1-8B/70B, DeepSeek-R1-Distill** — **not Llama-2-7B**

Nothing I could reach quotes a setup where Q-Filters strictly beats full attention. The paper's framing is "smallest PPL **drop**", implying Q-Filters ≥ vanilla. So our result (Q-Filters < vanilla) is **not claimed by the authors** — it is a pg19/Llama-2 quirk, not a reproduction of a paper headline. That said, the phenomenon "sparse attention beats dense on very long literary text" has precedent (Longformer's PG-19-ish settings, Streaming-LLM's >4k regime).

**Plausibility verdict**: plausible as an out-of-paper observation, but it should be read as a property of our eval harness, not a method claim.

---

## 2. Mechanism hypothesis

**Key datum**: vanilla pg19 PPL of 5102 is absurdly high — typical in-domain Llama-2-7B PPL is 5–12. Something about our harness is inflating the vanilla baseline, and Q-Filters happens to dodge whatever that is.

Evaluating the candidates:

- **(a) pg19 proper-noun denoising** — real but second-order. Trimming attention to 512/4096 tokens cannot make PPL drop by 1500 nats on clean text. Retain as a contributor, not the cause.
- **(b) sliding-window anchor effect — MOST LIKELY.** `budget=512 + recent=64` on 4096-token chunks means >87% of tokens are dropped and only "important" keys + last 64 are kept. This is effectively learned-anchor sliding-window. On boundary-cold chunk-eval where vanilla attention is being polluted by early low-context tokens (chunks are read cold with no carry-over), restricting attention to high-quality keys systematically helps. Longformer / Streaming-LLM results on pg19-like data show exactly this shape.
- **(c) test-time adaptation via calibration** — real but bounded. Filters are rank-2 projections of Q activations from pg19 chunks 0..63; chunks 200+ share the same author-distribution. This would explain a few-percent gap, not a 29% PPL reduction.
- **(d) RoPE extrapolation** — **ruled out.** Llama-2-7B was pretrained at 4096 tokens; eval at 4096 is not extrapolation. RoPE explodes above 4096, not at it.

**Most likely**: (b), strongly reinforced by (a). The filter-score selection is doing the job of a smart sliding window; on pg19 cold chunks, full attention is being hurt by uninformative early-position logits that Q-Filters prunes. Calibration (c) is a secondary tilt.

A corollary: if (b) dominates, naive random-512 or last-512 masking should also beat vanilla — just not by as much.

---

## 3. Recommended next experiment

Given mechanism (b) is the leading hypothesis, the single most decisive experiment is the **sliding-window sanity check**: re-run pg19 200×4096 with plain last-512 window attention (no filter scoring, same recent_window semantics). Three outcomes:

| Result | Interpretation |
|--------|---------------|
| SW PPL ≈ 3600, matches Q-Filters | Win is sliding-window, filters add nothing. Kills the method's novelty on this benchmark. |
| SW PPL ≈ 4200, Q-Filters clearly better | Filter scoring adds real signal. Publishable mechanism claim. |
| SW PPL ≈ 5000 (matches vanilla) | Something specific about Q-Filters' selection — revisit (c). |

This single run costs ~30 min on 8×L20A and answers the "is the win real or is it sliding window" question before we invest in a kv_budget sweep or Llama-3.1 port.

---

## 4. Risks / caveats

1. **Calibration leakage via `make_qfilters_cache` fresh-per-chunk** — if the cache builder re-runs any step of calibration per chunk, or touches current-chunk keys before scoring, we'd be leaking information from the eval chunk into the filter. Coder should confirm: filters are loaded once from `outputs/qfilters_baseline/filters.pt` and are **read-only** during eval. Grep `calibration.py` / `compression.py` for any `.update(` or `.fit(` paths on live keys.
2. **`skip_chunks=200` disjointness** — calibration uses pg19 head chunks 0..63; eval should start at chunk ≥200 (or at a non-overlapping file offset). If chunk 200 is in the *same novel* as chunks 0..63, filters are still somewhat in-distribution. Check `data/pg19_chunks.npy` stride / author boundaries. Acceptable for a first result but worth flagging.
3. **Baseline sanity** — vanilla 5102 is suspicious. Worth re-running the vanilla harness with: (a) BOS token prepended per chunk, (b) PPL computed over tokens 512:4096 only (warm context). Either of these could compress the vanilla/Q-Filters gap dramatically.
4. **Cross-family transfer not validated** — paper is Llama-3.1; our win is on Llama-2. Porting to Llama-3.1-8B remains the paper-claimed model; if the sliding-window sanity check is equivocal, that becomes the next priority.
5. **Chunk boundaries** — cold-start chunks penalize full attention more than masked attention. This is less a "win" than a known artifact of chunked eval.

---

**Recommended action for main loop: Dispatch /coder to add a plain-sliding-window mode to `scripts/eval_qfilters.py` (`--mode sliding_window` that skips filter scoring, keeps last 512 tokens), then /trainer runs it on 200×4096 pg19 — this isolates whether the Q-Filters win is mechanism (b) alone.**
