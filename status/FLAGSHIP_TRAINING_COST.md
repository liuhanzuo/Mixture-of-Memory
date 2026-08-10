# CoMem Flagship LoRA Training Cost

**Target artifact**: `outputs/qcmem_distill_qwen_j12_r32_4k/final`
**Key sources**: `adapter_config.json`, `distill_args.json`, `logs/qcmem_distill_qwen_j12_r32_4k.log`, `scripts/train_qcmem_distill.py`, `models/Qwen--Qwen3-8b/config.json`

---

## 1. LoRA Trainable Parameters

| Item | Value |
|------|-------|
| Backbone | Qwen3-8B (frozen, ~8.19B params) |
| LoRA rank `r` | 32 |
| LoRA alpha | 64 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (7 modules) |
| LoRA layer range | layers 12–35 (24 of 36 layers; `resume_j=12`) |
| **Trainable params** | **58.20M** |
| As % of backbone | **0.71%** |

**Layer coverage detail**: LoRA is applied **only to layers[12:36]** (the upper 24 layers of the 36-layer model). The bottom 12 layers (`layers[0:12]`) are entirely frozen and carry no LoRA weights. This matches the `resume_j=12` setting: context chunks are cached at depth 12 (query-blind, frozen lower layers only), and only the upper layers participate in the read-out pass where LoRA operates.

**Derivation** (cross-checked against log):
- Log line: `[qcmem-distill] LoRA on layers[12:36] targets=[...] -> trainable 58.20M params`
- Manual calc per layer (r=32, hidden=4096, intermediate=12288, n_kv_heads=8):
  - `q_proj`: (4096+4096)×32 = 262,144
  - `k_proj`: (4096+1024)×32 = 163,840
  - `v_proj`: (4096+1024)×32 = 163,840
  - `o_proj`: (4096+4096)×32 = 262,144
  - `gate_proj`: (4096+12288)×32 = 524,288
  - `up_proj`: (4096+12288)×32 = 524,288
  - `down_proj`: (12288+4096)×32 = 524,288
  - **Per-layer total: 2,424,832 × 24 layers = 58,195,968 ≈ 58.20M** ✓

**Backbone total** (~8.19B):
- Architecture: hidden=4096, intermediate=12288, 36 layers, vocab=151,936, GQA (32 heads / 8 KV heads), `tie_word_embeddings=false`
- 36 decoder layers: ~6.95B; embed_tokens + lm_head: ~1.24B; final norm: negligible
- Total: ~8.191B (source: `models/Qwen--Qwen3-8b/config.json`)

**Adapter file**: `final/adapter_model.safetensors` (232.8 MB, stored as float32 = 58.20M × 4 bytes; training was in bfloat16)

---

## 2. Training GPU Count and Wall-Clock Time

| Item | Value |
|------|-------|
| GPUs | **8× NVIDIA L20A** (183 GiB/card) |
| Node disk | wzc1 (`/apdcephfs_wzc1/share_304376610/`) → B200/L20A node |
| Distributed | `torchrun --nproc_per_node=8` (DDP, all 8 GPUs on one node) |
| Total steps | 4,000 |
| Steady-state throughput | ~24.5 samp/s (global, 8 GPUs; from log `seen*world_size/dt`) |
| **Estimated wall-clock** | **~22 minutes** |

**Wall-clock derivation**: The log reports throughput as `seen * world_size / dt` where `seen` resets every `log_interval=10` steps and `world_size=8`. At steady state ~24.5 samp/s: `dt_per_10_steps = (10×8)/24.5 ≈ 3.27 s`. Total: `(4000/10) × 3.27 s = 1,306 s ≈ 21.8 min`.

**Caveat**: No wall-clock timestamps are present in the log file (`logs/qcmem_distill_qwen_j12_r32_4k.log`). The ~22 min estimate is derived from the reported throughput. Actual elapsed time may differ by ~10% due to checkpoint saves and NCCL sync overhead.

---

## 3. PG19 Distillation Token Count

| Item | Value |
|------|-------|
| Training data | PG19 natural text (`data/pg19_train.jsonl`), streamed on-the-fly |
| Context window | `(n_ctx + 1) × chunk_size = (3+1) × 512 = 2,048 tokens` per sample |
| Global batch size | 8 (1 sample per GPU × 8 GPUs, `grad_accum=1`) |
| Total steps | 4,000 |
| **Total tokens seen** | **4,000 × 8 × 2,048 = 65,536,000 ≈ 65.5M tokens** |
| Loss-bearing tokens (query segment only) | 4,000 × 8 × 512 = 16,384,000 ≈ 16.4M tokens |

**Context structure per sample** (from `PG19Packer`): `[BOS sink (1 tok) ; ctx_0 ; ctx_1 ; ctx_2 (3×512 = 1,536 tokens) ; query (512 tokens)]`. The distillation loss is computed only on the 512-token query segment (`query_loss_tokens=0` means all query tokens are included).

**Data source confirmation**: `distill_args.json` → `"pg19_path": ".../data/pg19_train.jsonl"`. Script header explicitly states: *"NO babilong / NO needles / NO eval data — red line"*. Corpus is PG19 books (natural English text).

---

## 4. Unsupervised Training Declaration

**No benchmark labels, no long-context supervision, no QA annotations of any kind.**

This is a **pure self-distillation** on PG19:

| Component | Description |
|-----------|-------------|
| **Teacher** | Qwen3-8B itself at `j=0` — adapters disabled (`peft.disable_adapter()`), full-depth forward (`layers[0:36]`), run under `no_grad`. This is the RAG upper bound: the selected context chunks are re-processed through the entire frozen backbone in the presence of the query. |
| **Student** | Same Qwen3-8B at `j=12` — context cached at depth 12 (frozen lower layers, query-blind), LoRA on `layers[12:36]`, grad-bearing. |
| **Loss** | Bidirectional top-64 KL divergence between student and teacher logits, computed on the query segment only. `ce_weight=0.0` (no hard-label cross-entropy). |
| **Data** | PG19 books (natural text). No BABILong tasks, no needle-in-a-haystack, no benchmark data. |

**Key consequence for reviewers**: The student learns to predict the same token distribution as the full-context model, entirely from natural PG19 text. There is no second external teacher model and no labeled long-context supervision. The LoRA parameters (0.71% of backbone) are the only weights that change; the Qwen3-8B backbone is frozen throughout.

Source: `scripts/train_qcmem_distill.py` (teacher/student setup at lines 530–563); `distill_args.json` (`ce_weight: 0.0`, `pg19_path`, no eval flags).

---

## Summary

> **CoMem trains a single LoRA adapter (rank-32, 58.2M params, 0.71% of Qwen3-8B) on layers 12–35 only. Training: 8× L20A GPUs, ~22 minutes wall-clock, 4,000 steps on ~65.5M tokens of PG19 natural text. Loss = top-64 KL self-distillation (teacher = same frozen Qwen3-8B at j=0, no external model, no benchmark labels).**

| # | Metric | Value | Source |
|---|--------|-------|--------|
| 1 | LoRA trainable params | 58.20M (0.71% of 8.19B backbone) | `log` line 14, `adapter_config.json`, `config.json` |
| 2 | GPUs / wall-clock | 8× L20A, ~22 min | `log` throughput (no timestamp) |
| 3 | PG19 tokens seen | 65.5M (4000 × 8 × 2048) | `distill_args.json` + code |
| 4 | Supervision | None — pure PG19 KL self-distillation | `distill_args.json` + `train_qcmem_distill.py` |
