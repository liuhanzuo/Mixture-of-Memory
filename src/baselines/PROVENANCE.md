# Vendored KV-compression baselines — provenance (Paper A P1.6)

Standard prefill-then-compress KV-cache baselines. The retained KV/token budget
is fixed to the CoMem read budget (BOS 1 + top-12 x 512 + query <=512 = **6657**)
so the quality row is an equal-retained-token diagnostic against CoMem.

## Upstream repositories (cloned via hy-proxy, 2026-08-02)

| method | upstream repo | commit hash | vendored file |
|---|---|---|---|
| SnapKV (canonical) | `github.com/FasterDecoding/SnapKV` | `e216ddc84c5bd210378cbdbbba12ba02102aa640` | `snapkv/snapkv_utils_fasterdecoding.py` (verbatim), `snapkv/llama_hijack_4_37_reference.py` (reference only) |
| PyramidKV (+ GQA SnapKV) | `github.com/Zefan-Cai/PyramidKV` | `94255b6fe5127117f2e7f3b6d7ca7bd155ba9ab0` | `pyramidkv/pyramidkv_utils.py` (verbatim), `pyramidkv/monkeypatch_reference.py` (reference only) |

Upstream licenses vendored alongside: `snapkv/LICENSE`, `pyramidkv/LICENSE`.

## Why the upstream monkeypatch is NOT used as-is (and what IS used)

Both official repos monkeypatch **transformers 4.37 + Llama/Mistral** attention
(`*_hijack_4_37.py` / `monkeypatch.py`, patching `LlamaFlashAttention2.forward`,
the pre-`Cache`-refactor `past_key_value.update` contract, and `rotary_emb(...,
seq_len=)`). This project's env is **transformers 5.14.1 + Qwen3** (`.venv`,
torch 2.13, L20A sm_100), whose attention API was fully rewritten:
`Qwen3Attention.forward(position_embeddings=..., past_key_values: Cache)`,
`ALL_ATTENTION_FUNCTIONS`, `create_causal_mask`, per-layer `DynamicLayer`
KV tensors, and `q_norm`/`k_norm` on the head dim. The upstream files therefore
`ImportError` / break at runtime on Qwen3.

Faithful port (see `qwen3_kvcompress.py`):
- The **compression algorithm** — the class that decides *which* KV positions to
  keep — is the vendored, unmodified `SnapKVCluster` / `PyramidKVCluster` from
  Zefan-Cai `pyramidkv_utils.py`. These are GQA-aware (Qwen3 = 32 q-heads /
  8 kv-heads), which the older FasterDecoding `SnapKVCluster` is not, so the
  Zefan-Cai clusters are the ones actually driven. Their `update_kv` observation-
  window scoring (last `window_size` queries -> softmax -> pool -> top-k over the
  past, keep the recent window) and PyramidKV's per-layer pyramidal budget
  schedule are byte-identical to upstream.
- Only the **attention forward wrapper** is re-implemented for the Qwen3-5.14
  signature: it reproduces exactly the upstream call site — after RoPE + cache
  update on the *full prefill*, call `kv_cluster.update_kv(key, query, value, ...)`
  and write the compressed K/V back into the layer's `DynamicLayer`. Numerically,
  on the full prefill (no compression triggered when `q_len <
  max_capacity_prompt`) the wrapper is a no-op and matches stock Qwen3 SDPA
  bit-for-bit — this is the faithfulness self-check (`--self_test`).

## Faithfulness self-check

`scripts/eval_p16_kvcompress.py --mode selftest` runs a short (<=2k) input where
`q_len < max_capacity_prompt`, so no compression fires; it asserts the hijacked
logits match stock full-KV Qwen3 SDPA (max|Δlogit| < 1e-3, argmax identical).
This proves the monkeypatch does not perturb the model when it is not
compressing. A second check runs a >budget input and confirms the retained KV
length equals the configured budget on every layer (SnapKV: uniform; PyramidKV:
pyramidal, summing to ~budget*L).

## Retained-budget mapping

The methods parametrize retention by `max_capacity_prompt` (total kept tokens
per layer, INCLUDING the recent observation window). To match the CoMem read
budget of 6657 retained tokens per layer we set `max_capacity_prompt = 6657`
(with `window_size = 32`, i.e. 6625 attention-selected + 32 recent). SnapKV keeps
this uniformly on all 36 layers; PyramidKV keeps 6657 as the *average* per-layer
budget under its pyramidal schedule (lower layers more, upper layers fewer, total
= 6657 * 36). Both report the ACTUAL retained length per layer in the per-cell
JSON so the equal-retained-token claim is auditable.
