# Q-Filters Pivot — Research Brief (2026-04-25)

**Author**: researcher (autonomous chain `req_20260424_163200_attention_matching_pivot`)
**Supersedes**: `ops/research_notes/20260424_attention_matching.md` (built on unverifiable arXiv:2602.16284)
**Status**: P0, ready for `/coder` dispatch

---

## 1. Verified citation

- **Paper**: *Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression*
- **arXiv**: **2503.02812** (Godey et al., Mar 2025) — verified via WebSearch 2026-04-25
- **Code**: https://github.com/NathanGodey/qfilters
- **Evaluated models (in paper)**: Llama-3.1-8B/70B, DeepSeek-R1-Distill
- **Claim**: 99% NIH accuracy at 32× KV compression; 65% PPL-drop reduction vs Streaming-LLM

### Why this replaces the earlier target
- Previous brief cited **arXiv:2602.16284** (non-existent — future date)
- Fallback candidate **arXiv:2502.16284** is *MolSpectra* (molecular spectroscopy, unrelated)
- Q-Filters is the nearest real paper matching the description "training-free latent-space KV compression" in the approved pivot

---

## 2. Method summary

Q-Filters replaces the full KV cache with a **fixed-size, per-head projection** of the query-key geometry:

1. **Offline calibration** — run the target LLM on a small calibration corpus (a few hundred sequences). For each attention head, compute the top-`k` right-singular vectors of the stacked Q matrix; these are the head's "filters".
2. **Online inference** — at each decoding step, project both the new query and each cached key onto the head's filters. Cosine-similarity in this low-rank space approximates full attention scores; retain only the top-`budget` keys (and their values).
3. **FlashAttention-compatible** — filters are dense matmuls, no custom kernels.
4. **Training-free** — calibration is a single pass over a few hundred examples; no gradients.

Key hyperparameters:
- `filter_rank` (paper default: 1–4 per head)
- `kv_budget` (cache size kept per head; paper tests 128–2048)
- `recent_window` (always-kept most-recent tokens; paper default 32–64)

---

## 3. Expected numbers

**Baseline (measured on this machine, UPDATELOG 2026-04-23 08:30)**:
- Llama-2-7B vanilla, pg19, 200 chunks × 4096 tokens, bf16: **PPL 5102.22**
- This is the correct reference. The old 41.24 was from a fine-tuned variant that is **not reproducible on disk** (confirmed during 2026-04-24 17:15 cluster audit).

**Expected Q-Filters result**:
- Llama-2-7B + Q-Filters at `kv_budget=512, filter_rank=2`: **PPL ≤ 1.5× baseline** (i.e. ≤ 7653) would be a success
- At `kv_budget=2048` (mild compression): should match vanilla within ~5%
- **Risk**: Q-Filters was tested on Llama-3.1, not Llama-2. QK geometry can differ across model families; calibration may need more examples.

---

## 4. Risks / open questions

| Risk | Mitigation |
|------|------------|
| QK geometry assumption may not hold on Llama-2 | Test first on a single chunk; if filters look degenerate (singular values flat), abort and try Llama-3.1 |
| `trust_remote_code` issues on remote nodes | Pin `transformers==5.6.2` (we just installed) on b200-1 |
| Calibration data mismatch (paper used FineWeb; we have pg19) | pg19 is in-distribution for the eval anyway — acceptable |
| FlashAttention kernel fallback under bf16 | Use `attn_implementation="sdpa"` as fallback |

---

## 5. File layout for coder

```
src/memory/qfilters/
  __init__.py             — exports QFiltersConfig, QFiltersCache, patch_model
  layer.py                — QFiltersAttention (wraps LlamaAttention forward)
  compression.py          — compress_kv(q_proj, filters, keys, values, budget)
  calibration.py          — compute_filters(model, calib_loader, rank)

scripts/
  eval_qfilters.py        — mirror scripts/eval_baseline_ppl.py structure,
                            adds --kv_budget, --filter_rank, --recent_window,
                            --calibration_chunks flags
```

**Reference implementation**: https://github.com/NathanGodey/qfilters — coder should read `qfilters/src/` before implementing. Preserve API names where sensible.

**Integration contract**:
- `QFiltersConfig(kv_budget: int, filter_rank: int, recent_window: int, calibration_chunks: int)`
- `patch_model(model: LlamaForCausalLM, filters: dict[int, Tensor], config: QFiltersConfig)` — monkeypatches each `LlamaAttention.forward`
- Pre-calibration filters are saved to `outputs/qfilters_baseline/filters.pt` and loaded before eval

---

## 6. Smoke contract (must pass before full run)

**Target**: b200-1 (28.89.17.143), 1 GPU, <5 min wall clock

```bash
python scripts/eval_qfilters.py \
  --model models/Llama--Llama2-7b \
  --data data/pg19_chunks.npy \
  --max_chunks 10 \
  --seq_length 4096 \
  --kv_budget 512 \
  --filter_rank 2 \
  --recent_window 64 \
  --calibration_chunks 8 \
  --output_dir outputs/qfilters_smoke \
  --single_gpu
```

**Pass criteria**:
- No NaN in intermediate logits
- Final PPL is finite and ≤ 20000 (sanity upper bound — vanilla is 5102)
- Calibration filters have no NaN/Inf, at least `filter_rank` non-zero singular values per head
- `outputs/qfilters_smoke/eval_results.json` written with `ppl`, `kv_budget`, `filter_rank`, `num_chunks`

**Fail action**: record failure in `status/ISSUES.jsonl`, do not dispatch full run, hand back to /coder.

---

## 7. Full eval contract

**Target**: b200-1, 8 GPU, ~30 min wall clock estimated

```bash
torchrun --nproc_per_node=8 scripts/eval_qfilters.py \
  --model models/Llama--Llama2-7b \
  --data data/pg19_chunks.npy \
  --max_chunks 200 \
  --seq_length 4096 \
  --kv_budget 512 \
  --filter_rank 2 \
  --recent_window 64 \
  --calibration_chunks 64 \
  --output_dir outputs/qfilters_baseline \
  --bf16
```

**Report target**: `outputs/qfilters_baseline/eval_results.json` + appendix in `UPDATELOG.md`.
