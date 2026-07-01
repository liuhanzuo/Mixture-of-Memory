# Mixture-of-Memory: L1 + L2 + L3 Architecture Reference

**Date**: 2026-05-16
**Phase**: 11 (L1+L2+L3 cold-start training, in progress with `--gradient_checkpointing`)
**Champion baseline**: Phase 8 (L1+L3 only, overall mean 59.14 on BABILong qa1/qa2/qa5 × 21 cells)

This is a single reference for drawing the architecture diagram. All numbers and shapes are from actual source code (commits `0f6e8a1` for L2 + `62a26db` for gradient_checkpointing).

---

## 1. 系统总览 (System Overview)

```
                          Llama-3-8B-Instruct (FROZEN backbone)
                          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   input_ids ──► [Embed] ──► H₀ ──► Layer 0 ──► H₁ ──► Layer 1 ──► ... ──► Layer 31 ──► LM Head ──► logits
                                       │                                        │
                                       │ (every layer wrapped)                  │
                                       ▼                                        ▼
                              MemorySpaceLayer 0                       MemorySpaceLayer 31
                              (内含 L1+L2+L3)                          (内含 L1+L2+L3)
                                                                                │
                                                                                ▼
                                                                  [Post-forward hooks]
                                                              ──► L3SummaryPool stash detached H
                                                              ──► L2Compressor compress detached H
                                                                                │
                                                                                ▼
                                                                       For NEXT chunk
```

**关键设计原则**:
1. Backbone (Llama-3-8B-Instruct, ~8B params) **frozen**.
2. 每个 `LlamaDecoderLayer` 被包裹成 `MemorySpaceLayer`, 在内部加入 L1+L2+L3 prefix.
3. Chunked streaming: 输入序列按 `chunk_size=1024` 切, 跨 chunk 通过 L1/L2/L3 的 cross-chunk state 传递信息.
4. Cold-start: L1/L2/L3 全部从零开始训练 (no NIAH pretrain).

---

## 2. 单层 `MemorySpaceLayer` 内部 (P11 完整版)

```
                          Chunk i 进入第 ℓ 层
                                  │
                       hidden_states H ∈ [B, T=1024, d=4096]
                                  │
        ┌─────────────────────────┼─────────────────────────────────┐
        ▼                         ▼                                 ▼
    ┌─ L3 READ ──┐            ┌─ L2 READ ──┐                  ┌─ L1 SELECT ─────────────┐
    │            │            │            │                  │                         │
    │ prev_H 🔒  │            │ prev_latents 🔒                │ slot bank M (per-layer) │
    │ (chunk i-1)│            │ (chunk i-1)│                  │ [B, N=512, slot_dim=2048]│
    │   │        │            │   │        │                  │   │                     │
    │   ▼        │            │   ▼        │                  │   ▼                     │
    │ Q-Former   │            │ kv_b       │                  │ selector q = Q_sel(H)   │
    │ Pool 🌐    │            │ Linear:    │                  │ scores = softmax(qK)    │
    │ 64 queries │            │ d_c=512    │                  │ idx = top_k(scores, 64) │
    │ × 2 cross- │            │ → 2·n_kv·  │                  │ M_sel = gather(M, idx)  │
    │   attn blk │            │   d_head   │                  │ M_sel_hidden =          │
    │   │        │            │ = 1024     │                  │   slot_to_hidden(M_sel) │
    │   ▼        │            │   ▼        │                  │   │                     │
    │ S∈[B,64,d] │            │ K_recon,   │                  │ [B, 64, d=4096]         │
    │            │            │ V_recon    │                  │   │                     │
    │            │            │   │        │                  │   │                     │
    │            │            │   ▼        │                  │   │                     │
    │            │            │ 0.5·(K+V)  │                  │   │                     │
    │            │            │ = l2_tokens│                  │   │                     │
    │            │            │ [B,64,d]   │                  │   │                     │
    └────┬───────┘            └────┬───────┘                  └───┼─────────────────────┘
         │                         │                              │
         └─────────────────────────┴──────────────────────────────┘
                                   │
                                   ▼
                       extended_hidden = cat([
                         L3_tokens (64),   ◄── shared, same across all layers
                         L2_tokens (64),   ◄── shared input, projected per layer
                         L1_tokens (64),   ◄── per-layer slot bank
                         H        (1024),
                       ], dim=1) → [B, 1216, d=4096]
                                   │
                          ┌────────┴────────┐
                          │                 │
                          ▼                 ▼
                  bypass: wrapped_layer(H, mask=causal)
                       output H_bypass [B, T, d]
                                            │
                  extended: wrapped_layer(extended_hidden,
                       mask=[L3/L2/L1 全可见; H 内 causal],
                       RoPE: L3/L2/L1 all pos=0, H pos=0..T-1)
                       output [..., -T:] = H_ext [B, T, d]
                                                │
                                                ▼
                          Flamingo gate (cold-start parity)
                          α = tanh(α_param), init α_param=0 → α=0
                          next_hidden = H_bypass + α·(H_ext − H_bypass)
                                                │
                  ┌─────────────────────────────┘
                  ▼
                                       L1 WRITEBACK (dual-gate)
                  O_mem = H_ext[:, k_l3+k_l2 : k_l3+k_l2+k_slots]
                  O_mem_slot = hidden_to_slot(O_mem)   # [B, k, slot_dim]

                  new      = tanh(gate_proj_new(O_mem_slot))
                  β_in     = σ(gate_proj_input(O_mem_slot) + b_in=0)
                  β_forget = σ(gate_proj_forget(O_mem_slot) + b_forget=2.0)

                  M_sel ← β_forget · M_sel + β_in · new
                  M.scatter_(idx, M_sel)              # in-place update bank

                  (slot value norm cap = 5.0)
                                                │
                                                ▼
                                            next_hidden  → 进入第 ℓ+1 层

  ──────── (在 第 31 层 forward 完后, post-forward hooks 触发, 为下一 chunk 准备) ─────────
                                                │
                          ┌─────────────────────┴───────────────────┐
                          ▼                                          ▼
                  L3 hook:                                  L2 hook:
                  pool._prev_chunk_h = H.detach()           comp.prev_latents = comp.compress(H.detach())
                  (under no_grad)                           (under no_grad)
                                                            ── softmax(w_gate(H) + APE)
                                                            ── pool g=16 tokens → 1 latent
                                                            ── 输出 [B, T/g=64, d_c+d_R=576]
```

---

## 3. 三层"角色"对照

| 层 | 全称 | 容量 | 共享性 | 写法 | 读法 | 学到的是 |
|---|---|---|---|---|---|---|
| **L1** | `mem_space slots` | 每层独立 512 slot × 2048 dim, top-k=64 每次读 | **per-layer** | dual-gate: `M ← β_f·M + β_in·tanh(new)` | selector 投影 H → softmax 选 top-64 slot → 拼到 ext seq 作 prefix | **离散关键事实** (NIAH-style, 长上下文 fact tracking) |
| **L2** | token-compressed KV | 1 个共享 module | **shared 跨 32 层** | post-forward hook 从 detached H 用 NSA/V4-CSA 风格 pool g=16 tokens → 1 latent (d_c=512) | 每层用 `kv_b: d_c→2·n_kv·d_head` 重建 K, V, 平均成 pseudo-token 拼到 ext seq | **连续压缩 KV** (chunk-level "remember everything cheaply") |
| **L3** | Q-Former summary pool | 1 个共享 pool, 64 learnable queries × 2 cross-attn blocks | **shared 跨 32 层** | post-forward hook stash `prev_H.detach()`, 下 chunk 调 pool(prev_H) 用 64 query cross-attend | 每层共用同一 S 拼到 ext seq 作 prefix | **稠密语义摘要** ("what happened last chunk in semantic space") |

---

## 4. Extended sequence 在 attention 里的样子

```
attention_mask 大小 [B, 1, L=1216, L=1216]:

         L3(64)  L2(64)  L1(64)  H(1024)
        ┌─────┬──────┬──────┬───────────┐
L3 (64) │  ✓  │  ✓   │  ✓   │     ✓     │  ← memory 行: 全可见
L2 (64) │  ✓  │  ✓   │  ✓   │     ✓     │
L1 (64) │  ✓  │  ✓   │  ✓   │     ✓     │
        ├─────┼──────┼──────┼───────────┤
H(1024) │  ✓  │  ✓   │  ✓   │  causal   │  ← H 行: 可读 memory + 内部因果
        └─────┴──────┴──────┴───────────┘

RoPE positions:
  L3/L2/L1 → 全部 position 0 (memory tokens 是 position-less)
  H        → 0..1023 (chunk 内正常 RoPE)
```

为什么 L3/L2/L1 用 position 0?
- L3/L2 是上一 chunk 的全局摘要, 不绑定具体 token 位置
- L1 是离散事实存储, 跨多 chunk 共用, 也不绑定位置
- 这与 Flamingo / RETRO 等架构的 memory-prefix 设计一致

---

## 5. 跨 chunk 状态流

```
              chunk i 前向                              chunk i+1 前向
              ─────────────                            ───────────────

L1 (per-layer):  M (slot bank) ──── in-place writeback ────► M (updated)
                                                              (continues evolving)

L3 (shared):     pool ── post-fwd hook ──► _prev_chunk_h ──► layer 0 调用 pool(prev_h)
                                            (detached)         → S 缓存在 _chunk_summary_cache
                                                               → layer 1..31 用 cache
                                                               (避免 32× 重算)

L2 (shared):     comp ── post-fwd hook ──► prev_latents ───► layer 0..31 共读 prev_latents
                       compress(H.detach())                   → kv_b 上投到 K/V 重建
                       under no_grad()                        → 0.5·(K+V) → l2_tokens

                       (post-fwd 触发位置: 最后一个 mem layer 跑完后)
```

**为什么用 detached H + post-forward hook**: chunk-local BPTT.
- chunk i+1 的 `loss.backward()` 只能反传到当前 chunk 内的计算
- L1/L2/L3 的 cross-chunk state 都从 detached H 派生, 不形成跨 chunk autograd 图
- L3 / L2 module 的参数梯度通过当前 chunk 的 forward (`pool(prev_h)` 在 layer.forward 里) 拿到

---

## 6. 训练 / 推理双模

### 训练 (BPTT chunk-local)
- `chunk i+1` 的 loss 只反传到当前 chunk 内的 L2/L3 op
- L1 slot bank 通过 in-place 更新, 跨 chunk 流动 (slot 状态隐式 BPTT, 不形成完整 autograd 图)
- gradient_checkpointing (commit `62a26db`) 包装 `wrapped_layer.forward` 节省激活内存

### 推理 (Chunked streaming)
- 同样 chunked, 但无 backward
- `_reset_banks(model)` + `_reset_l2(model)` 在 document 边界把 L1 slots / L2 prev_latents / L3 prev_h 全部清零
- KV cache 不跨 chunk 累积; 信息只通过 L1/L2/L3 跨 chunk 传递

---

## 7. 参数预算 (Phase 11 实际, Llama-3-8B-Instruct backbone)

| 模块 | 每层 | 总计 (32 层) | shared? |
|---|---|---|---|
| **Frozen Llama-3-8B backbone** | — | **8030M (frozen)** | — |
| L1 mem_space (slots, selector, gates, hidden_to_slot, slot_to_hidden) | ~33M | **~1100M** | per-layer |
| L3 Q-Former pool (2 cross-attn blocks, 64 queries × 2048 dim) | — | **~270M** | **shared 1×** |
| L2 Compressor (w_kv, w_gate, ape, kv_b, w_kR) | — | **8.66M** | **shared 1×** |
| Flamingo α gate (per layer scalar) | trivial | <1M | per-layer |
| **总 trainable** | | **~1387M** | |

**显存占用 (per H20 rank, world_size=8 DDP)**:
- Backbone bf16: 16 GB
- Trainable bf16: 2.8 GB
- AdamW state fp32: 16.6 GB
- Grad fp32: 5.5 GB
- DDP buckets: 1.5 GB
- 静态合计 **~42 GB**
- 激活 (2 chunks BPTT, 2k 训练长度): ~20 GB
- **峰值 ~62 GB / rank** (但 fragmentation 可能让 PyTorch caching allocator 把 GPU 提前撑爆)

---

## 8. L1 / L2 / L3 互补性

```
                时间维度
       chunk 1   chunk 2   chunk 3   ...   chunk K
       ─────────────────────────────────────────►
                
L1:    [离散 fact 1, fact 2, ...] ── 持续累积, 关键事实存储, in-place update
       512 slot bank, top-k=64
       通过 dual-gate (β_in, β_forget) 选择性更新
       
L2:    [chunk 1 KV]   [chunk 2 KV]   [chunk 3 KV]   ...
       每 chunk 64 个 token-compressed latent (d_c=512)
       只保留前一 chunk 的, 不累积
       "上一 chunk 我看到的所有 token 的 KV 压缩版"
       
L3:    [chunk 1 vibe] [chunk 2 vibe] [chunk 3 vibe] ...
       每 chunk 64 个稠密 summary token (d=4096)
       Q-Former cross-attend 提取
       "上一 chunk 的语义摘要"
```

**为什么需要三层**:
- L1 单层: 只存 token-level 离散事实, 缺乏 chunk-level 上下文 (Phase 5b 的 qa5/4k = 46 vs P5a 76 显示 NIAH bias)
- L1+L3: 加入语义摘要, 修复 NIAH bias (Phase 8 mean 59.14, +6.9 vs P5a)
- L1+L2+L3 (P11 目标): 加入 token-level KV 压缩, 期望补足 L3 的 lossy summary 不能捕捉的 fine-grained 信息

---

## 9. Phase 历史对照 (供绘图标注)

| Phase | L1 | L2 | L3 | lr | steps | 结果 | 备注 |
|---|---|---|---|---|---|---|---|
| P4 (dual-gate) | NIAH-init | ❌ | ❌ | 2e-5 | 1000 | ~55 | NIAH 偏置 |
| P5a (pure L3) | disabled | ❌ | cold | 2e-5 | 1000 | 52.24 | 无 L1 baseline |
| P5b (L1+L3) | NIAH-init | ❌ | cold | 2e-5 | 1000 | 50.10 | NIAH bias 在 qa5/4k 拖 -30pp |
| P6 (L1 cold) | cold | ❌ | ❌ | **1e-4** | 2000 | **0 (diverged)** | lr 太高 |
| P7 (L1+L3 cold) | cold | ❌ | cold | **1e-4** | 2000 | **0 (diverged)** | lr 太高 |
| **P8 ★** | **cold** | ❌ | **cold** | **2e-5** | **500** | **59.14** | **当前 champion** |
| P10/s1000 (long train) | cold | ❌ | cold | 2e-5 | 1000 | 60.96 (19 cells) | 长训轻微过拟合长 ctx |
| P11 (L1+L2+L3) | cold | cold | cold | 2e-5 | 500 | ⏳ in progress | OOM 调试中 |

---

## 10. 文件位置 (源码 reference)

| 模块 | 文件 |
|---|---|
| `MemorySpaceLayer.forward` | `src/memory/mem_space/layer.py` (line 419+) |
| Extended seq cat + mask | `src/memory/mem_space/layer.py:694-770` |
| `_maybe_ckpt_wrapped_layer` (gradient_checkpointing) | `src/memory/mem_space/layer.py:455-481` |
| `apply_mem_space_to_model` (patch + hooks) | `src/memory/mem_space/patch.py` |
| L3 Q-Former pool | `src/memory/mem_space/l3_summary.py` |
| L2 NSA-style compressor | `src/memory/mem_space/l2_compressor.py` (commit `0f6e8a1`) |
| L1 memory bank | `src/memory/mem_space/bank.py` |
| L1 selector (Q_sel, top-k) | `src/memory/mem_space/selector.py` |
| Config (use_l2/l3, dual_gate, etc.) | `src/memory/mem_space/config.py` |
| Training script | `scripts/train_mem_space_babilong.py` |
| Eval script | `scripts/run_babilong_mem_space.py` |

---

## 11. 相关设计文档

| 主题 | 文档 |
|---|---|
| L2 spec (NSA/V4-CSA-style) | `docs/L2_DEEPSEEK_MLA_RESEARCH.md` (551 行) |
| L2 实现 plan | `docs/L2_IMPLEMENTATION_PLAN_20260516.md` |
| Paper baselines (BABILong + LM2) | `docs/PAPER_BASELINES_20260516.md` |
| Phase 8 final results | `status/FINAL_RESULTS_20260516.md` |
| FSDP migration plan | `docs/FSDP_MIGRATION_PLAN_20260516.md` (476 行, Phase 12 备选) |

---

*Generated 2026-05-16 14:55 CST.*
*Last live training: Phase 11 (8B+L1+L2+L3 cold-start) attempting on remote H20 8 GPU + Phase 1B-v2 (1B+L1+L3, 10k steps) on local H20 8 GPU.*
