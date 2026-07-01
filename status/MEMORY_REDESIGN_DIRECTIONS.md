# Memory 重设计方向（2026-06-13，主会话探讨后）

## 框架修正
之前"写侧/读侧全证伪 → mem-space 不优于长上下文"的结论**下早了**。实际只穷尽了 **L1 写规则的一个维度**（sparse top-k ↔ dense top_k=N）+ L1 读衰减（v20）。真正没碰的：
- **L1 的核心病（槽绑定到少数活跃槽）从未真正解决**——dense write 是把所有槽写崩（生成坍缩成 "ion ion ion"），不是让槽各司其职。R1/W2/v20 都是 stub 或对症补丁，不是寻址机制重设计。
- **L2 完全没开过**（`use_l2=False` 贯穿所有 run，compressor 代码从未被执行）。
- **L3 设计未完**：recon loss 是 stub（config 定义了但训练代码从不实例化）；recursive L3 没压测；query collapse 只用 LayerNorm 打补丁。

## 三层现状（代码实证）
| 层 | 是什么 | 写 | 读 | 状态 |
|---|---|---|---|---|
| **L1** | N 个离散 slot bank | top-k(16) 选中槽 dual-gate/delta 写 | all-N cross-attn（按 key 相似度）| 成熟但**dead-slot 绑定未解**|
| **L2** | NSA 式 token 窗口压缩（g=16→latent）| 全 token 软池化压缩，detach 输入 | 全 latent broadcast 拼接 | **从未启用，纯 stub**|
| **L3** | Q-Former 式 64 summary token | 整 chunk cross-attn 池化 | 全 summary broadcast | **部分**：recon stub/recursive 没测 |

## L1 dead-slot 绑定问题的真正根因
选择 = `Q_sel(hidden)·(K_sel(slots)+slot_key_bias)` → **寻址本质是"内容相似度"**。
- 死锁：槽没被选→没新内容→key 不更新→更不会被选（Catch-22）。
- `slot_key_bias`（正交 init）给了 key 多样性，但**没有真正的"可寻址 key"**（不像 VQ-VAE 有 codebook 索引）。
- 不能靠加噪声破（v8 试过，top1_sim 0.99→0.11 崩）。

## 候选方向（待主会话定优先级）

### 方向 1 — L1 寻址重设计：真正解耦 key 和 content
**核心**：让"哪个槽被选"由**独立可学 key 参数**决定，而非槽内容。
- 1a. **完全独立的可学 slot key**：`key[n]` 是 nn.Parameter（不是 K_sel(slots)），槽 content 只做 value。选择 = Q·key_param。这样空槽也有稳定 key，可被选中获得内容 → 破 Catch-22。
- 1b. **写入 = 擦除+写入（DeltaNet 式但在槽上）**：选中槽时先按 key 相似度擦除旧关联再写，让槽内容真正"关联记忆化"而非 EMA 累加。
- 风险：改寻址可能伤 top1_sim 精确命中（arm4 教训）。但 arm4 动的是分布均摊，这里是换 key 来源，性质不同。

### 方向 2 — L2 首次启用（全新未探索轴）
**核心**：L2 是 token 级压缩 KV，比 L1 的离散槽更接近 DeltaNet/Titans 的 dense 思路。
- 2a. 先裸开 `use_l2=True`（P11 base + L2），看是否补长程。
- 2b. 给 L2 加**可学 readout gate**（当前是 always-on broadcast，无门控）+ 修双投影浪费。
- 2c. L2 压缩输入当前 detach → 可试不 detach，让梯度流回"哪些 token 该压缩"。
- 价值：这是唯一一条**架构上全新**的路，且天然 dense（无 dead-slot 问题）。

### 方向 3 — L3 补完（⚠️ 核查后：两条主补法已撞墙）
**已 REJECTED（别重跑）：**
- 3a. ❌ L3 token-recon loss（ICAE 式）——2026-06-07 已 sweep w0.3/w1.0 均 REJECTED（RUN_REGISTRY L130）。w1.0 qa1=4/0/0...（recon 破坏检索），w0.3 也全面劣于 base。**根因：token 重建目标 = 逼 summary 存所有 token，与"压缩检索关键信息"冲突，权重越大越烈。** 代码其实完整接线（patch.py:160 实例化，trainer:1614 算 loss），不是 stub——是试过被否。
- 3b. ❌ L3 diversity 正则——2026-06-11 已 sweep（RUN_REGISTRY L473）。治了 token 坍缩(cos 0.99→0.18)却伤长程，剂量反向。

**仍开放的 L3 角度（没撞过墙）：**
- 3c. recursive L3 压测 + 修 _prev_chunk_h 内存泄漏（工程，非科学杠杆）。
- 3d. **L3 summary 的"检索式监督"**：不重建 token（已否），而是用对比/检索目标——让 summary 学会"指向后面 query 要找的内容"，而非"重建当前 chunk"。这是 token-recon 失败的反向思路，未试。
- 3e. L3 n_summary/n_layers/n_heads 容量 sweep——base64 最优已知，但没在新 backbone 上多 seed 确认。

**结论**：L3 的"让 summary 可解码"主补法（recon/diversity）都撞过墙，根因是辅助目标与检索目标冲突。L3 不是"未完成"，是"已知难补"。**3d（检索式监督）是唯一没试的新角度，但设计成本高。**

### 方向 B（并行，用户已批）— 容量收益多种子坐实
N192/N384 中长档之前疑似单种子运气，跑齐 3 seed 算 mean±std 确认容量曲线真伪。

## 推荐执行
- **B** 在跑（多种子容量）。
- 架构主攻建议：**方向 2（L2 首次启用）优先** —— 投入产出比最高：全新轴、天然 dense、代码已存只是没接线。其次方向 1a（独立 slot key）破 dead-slot。
- 判据统一：长程 qa5 是否超 P11 step500 baseline（48/45/44）。
