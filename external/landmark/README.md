# Landmark Attention Reproduction (Phase 1 S0)

Reproduce the official Landmark Attention passkey-retrieval long-range result
(arXiv 2305.16300, repo `epfml/landmark-attention`) on our H20 infra, **zero
training** — recover the released tuned ckpt from the official weight-diff and
run passkey eval. This is the trusted anchor for the later diff-based migration
toward `mem_space` (see `status/LANDMARK_REPRODUCE_PLAN.md`).

## Layout

```
external/
  landmark-attention/      # official repo clone (DO NOT edit core code)
  landmark_venv/           # isolated venv (NOT the main .venv) — see below
  landmark_ckpts/          # downloaded weights (gitignored, large)
    wdiff/                 # epfml/landmark-attention-llama7b-wdiff
    llama1_7b_base/        # original LLaMA-1-7B base (huggyllama/llama-7b)
    landmark_tuned/        # recovered tuned ckpt (output of recover step)
  landmark/                # OUR harness (committed)
    run_passkey.py         # parametrized passkey eval (env-driven, writes CSV)
    recover_weights.sh     # wrapper around official weight_diff.py recover
    run_eval.sh            # wrapper to launch run_passkey.py with our paths
    README.md              # this file
```

## Why a separate venv

The main project `.venv` is transformers 5.5.4 + torch 2.10, whose internal
APIs are incompatible with Landmark's `llama_mem.py` (written for transformers
4.28). Importing it under the main venv fails. So we built:

```
python3 -m venv external/landmark_venv
external/landmark_venv/bin/pip install \
  --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0
external/landmark_venv/bin/pip install \
  transformers==4.28.1 'tokenizers>=0.13.3' sentencepiece accelerate \
  fire 'numpy<2' rouge_score 'protobuf<3.21'
external/landmark_venv/bin/pip install \
  'huggingface-hub==0.14.1' 'datasets==2.14.0' 'fsspec<=2023.6.0'
```

- torch 2.1.0+cu121 runs on H20 (sm_90); driver 535/CUDA12.8 is backward
  compatible with cu121 wheels. `torch.cuda.is_available()` == True.
- `numpy<2` / `huggingface-hub==0.14.1` / `datasets==2.14.0` are pinned because
  transformers 4.28.1 breaks against the newer majors pip pulls by default.
- non-triton path (`use_flash=False`) is used for inference, avoiding the
  triton/flash-attn version pin (`install_deps.sh`) entirely.

## ★ Base model: LLaMA-1-7B (not LLaMA-2)

The official wdiff is the diff vs the **original LLaMA-1-7B**. Evidence:
- HF card: "weight diff between LLaMA 7B ... and the original model".
- `weight_diff.py recover` backward-compat checksum default `49798.7656` is the
  LLaMA-1-7B param-sum (Alpaca lineage); the wdiff repo ships no separate
  `checksum_psum.txt`.
Using LLaMA-2-7B as base would fail the integrity check. We use
`huggyllama/llama-7b` (standard HF-converted LLaMA-1-7B) as base.

## Steps

All weights downloaded via the woa proxy:
`export http_proxy=http://hy-proxy.woa.com:3128 https_proxy=... no_proxy=...,.woa.com`

1. **Recover tuned ckpt** (adds wdiff onto base, runs checksum check):
   ```bash
   cd external/landmark
   bash recover_weights.sh
   # -> external/landmark_ckpts/landmark_tuned/
   ```

2. **Run passkey eval** (reproduce paper Fig — base collapses >2k, landmark stays
   high to 32k):
   ```bash
   cd external/landmark
   bash run_eval.sh                 # both base + mem arms, full length sweep
   # results -> passkey_results.csv  +  log
   ```

`run_passkey.py` is env-driven (see its header):
`LM_BASE`, `LM_TUNED`, `LM_MODELS` (subset of `base,mem`), `LM_TOPK` (default 5),
`LM_NTESTS` (default 50), `LM_NVALUES`, `LM_OUT`, `LM_{BASE,MEM}_DEVICE`.

## Gate (Phase 1 success)

Base LLaMA collapses to ~0% beyond ~2k tokens; landmark-mem stays near 100% (or
far above base) at 8k/16k/32k. `n_garbage` chars 0→38000 ≈ up to ~32k tokens.

## ★ RESULTS (2026-06-19) — WALL REPRODUCED

Passkey retrieval, 50 tests/length, top_k=5, base=LLaMA-1-7B, mem=recovered
landmark-tuned ckpt. n_garbage chars mapped to token lengths (garbage text ≈
3.7 chars/tok). `mem` long lengths sharded ×3 across GPUs then merged.

| ~tokens | base LLaMA-1-7B | landmark-mem |
|--------:|:---------------:|:------------:|
|     ~70 |          100%   |     100%     |
|    1.1k |          100%   |     100%     |
|    2.2k |           98%   |      94%     |
|     4k  |       **0%**    |   **100%**   |
|     8k  |       **0%**    |      96%     |
|    16k  |    OOM*         |      96%     |
|    32k  |    OOM*         |      96%     |

\* base does full O(n²) attention; >2048 ctx (its `max_position_embeddings`)
the passkey is already past the trained window → 0% accuracy, and ≥16k OOMs a
single 96GB H20 (expected; base genuinely cannot do long ctx).

**Verdict: SUCCESSFUL reproduction of the paper's long-range result.** base
LLaMA collapses to 0% exactly at its 2048-token limit (2.2k→4k: 98%→0%), while
landmark-mem stays 94–100% all the way to **~31k tokens (96%)** — i.e. the
landmark memory mechanism breaks the long-range wall, matching the paper's
passkey figure. This is our trusted anchor for the diff-based migration
(Phase 2/3).

Raw per-cell CSV: `results/passkey_full.csv`. Per-shard logs/CSVs in `results/`.
