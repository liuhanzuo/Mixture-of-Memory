# v15 — Decoupled Read Path (`use_decoupled_read`)

**状态**: 代码已落地（config.py:280 / layer.py:437-1081 / toy_memory_bootstrap.py:138 / train_mem_space_dolmino_cpt.py:446,585 / _toy_arm.sh DECOUPLED_READ）。
对应 plan [P2]。Toy 验证 inconclusive（toy 是短上下文，无法触发 ≥4k dilution cliff）；真正判据 = Dolmino 8-GPU arm 的离线 BABILong ≥4k 是否越过 1-2% noise floor。

## 动机（researcher 根因 2026-06-03，confidence medium）

P1 verdict（norecon 8.74% vs recon 6.89%，二者 ≥4k 均塌到 1-2% noise floor）证明 memory read-path
在长上下文下完全不可用。根因：

1. **注入稀释（主因）**：legacy prepend 路径把 k=16 个 slot KV token 与最多 1024 个 live
   token 放进**同一个 softmax**。长上下文下 memory 只拿到 ~1.5% 的 attention mass，再被
   slot_delta clip + inject_gate(0.12, std 0.007 flat) 压到 ~0.2% → 等效无注入。
2. **routing collapse（次因）**：top1_sim 0.01-0.03。

## 改了什么

`use_decoupled_read=True` 时：
- **关闭** slot KV-prepend（`M_sel_hidden` 不再拼进扩展序列）→ slot 不再和 live token 抢同一个
  softmax。wrapped layer 跑在 `[L3 | L2 | H]`（或纯 bypass）。
- **新增** 独立 cross-attn read：`Q=hidden, K/V=slots`，在**自己的 softmax** 上算 memory
  贡献，再经同一个 content-conditioned `inject_gate g` 加到 `next_hidden`。
  - `out_proj` 零初始化（LoRA-B 风格）→ step-0 输出 = 0，init 行为与"无注入"完全一致，向后兼容。
- **不变**：top-k routing + writeback（仍决定哪些 slot 被写）。只解耦了 READ→hidden 路径。

`use_decoupled_read=False`（默认）= legacy prepend 路径，完全向后兼容。

## Relationship to prior work

- 与 **MemoryLLM** 的区别：MemoryLLM 把 memory token 直接拼进序列（即我们要废弃的 prepend
  路径）；我们改为独立 softmax 的 cross-attn read，避免长上下文稀释。
- 与 **Infini-attention** 的区别：Infini 用 linear-attention 压缩历史并门控融合；我们保留离散
  top-k slot routing + writeback，只把 read-to-hidden 改成独立 softmax cross-attn。
- 复用项目自身的 `CrossAttentionMemoryV2.read`（selector.py:1081）。

## Known issues

- Toy retrieval_exact_acc 无法验证此改动（短上下文，dilution cliff 在 ≥4k）。toy ON/OFF 同为
  exact_acc=0 / tok_acc=0.375，ON top1_sim 略低于 OFF（0.247 vs 0.350）——非决定性。
- 真正判据需 Dolmino 8-GPU arm + 离线 BABILong ≥4k。
- ON arm 显存比 OFF 高 ~12GB（53GB vs 41GB，独立 cross-attn read 模块）。
- routing collapse（次因）未在本版处理，留待 P2 后续 / P3。
