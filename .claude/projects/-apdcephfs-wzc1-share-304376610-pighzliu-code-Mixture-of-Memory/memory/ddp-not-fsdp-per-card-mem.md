---
name: ddp-not-fsdp-per-card-mem
description: ★DDP (标准 all-reduce grads only) 不 shard params/grads/optim, 加卡不减每卡内存; FSDP/ZeRO 才行. train_olmo2_arch_probe2.py 是 plain DDP, keep28 加卡 = 白加
metadata:
  type: reference
---

**Fact**: PyTorch **plain DDP**（`DistributedDataParallel`）**不分片** params / grads / optim state；
它只做 gradient all-reduce（每 rank 收到完整 grad 后同步平均）。所以每 rank 的**静态内存占用与卡数无关**：
`params + grads + optim_state (AdamW m,v) + activation` 都是 full copy per rank。

**Why this matters**: 2026-08-06 keep28 (30L ~6.9B OLMo-2, fp32-AdamW) 单节点 8×H20 (95GiB) OOM。
我和 subagent 都想到"加到 16 卡分片砍半"。**错了**：`scripts/train_olmo2_arch_probe2.py` 是 plain DDP，
加卡 = 完全没减少每卡内存。subagent 起了 zwfy6 2-node 16 卡后二次 OOM 才反查出真因：
- 每 rank 静态: 27.6 (params fp32) + 27.6 (grads fp32) + 55.2 (AdamW m,v fp32) = **~110 GB > 95 GB H20**
- 与卡数**完全无关**

要"加卡减内存"的效果，需 **FSDP** 或 **ZeRO stage-2/3** —— 它们把 param/grad/opt 分片到各 rank。
`train_olmo2_arch_probe2.py` 没引用 FSDP/ZeRO（`grep 'FSDP\|ZeRO' scripts/` 空）。

**How to apply**:
1. 用 plain DDP 训练大模型 OOM 时，**先算 per-rank 静态占用**（`params + grads + optim_state`），
   与卡数无关。fp32-AdamW ≈ `4 × params_GB` (params + grads + m + v，各 4B/param)。
2. 别指望"加节点"救 plain DDP 的 OOM。选项：
   - 换更大内存的卡（本项目 H20 95GB → B200/L20A 183GB 差不多倍数，OLMo-2 30L 从 OOM 变为 168 GB/卡跑通）
   - 换 bnb 8bit AdamW（m,v 从 55.2GB → 6.9GB，per-rank 静态降到 ~62 GB → H20 装得下）；但**不同 optimizer 与 A3 fp32-AdamW 不 byte-identical，需在报告里明写口径差异**
   - 改脚本引入 FSDP —— 但对已有对照实验会破坏 byte-identical 比较（本项目铁律）
3. 项目里凡是要"加卡就好"的直觉，先 `grep 'FSDP\|fsdp\|zero_stage\|deepspeed'` 确认脚本支持分片。

**Cross-refs**: subagent commit `744bd09` 的 message 有此错误前提，`(placeholder-for-followup-commit)` 是更正条；
[[cluster-two-disks-not-shared]]（同盘 16 卡合并的另一约束是文件系统，本条约束是**算力语义**）。
