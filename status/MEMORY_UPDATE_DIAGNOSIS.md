# 记忆更新机制诊断 + 重设计调研（2026-06-12）

## 0. 一句话病根
我们的"记忆"在**写**和**读**之间断了链：**写是稀疏的（每 chunk 只 top-k≈16 个槽），读是稠密的（all-N 按 key 相似度），而 key 是槽内容的投影 → 从未被写的槽永远保留 init 的 chunk-0 token 快照，且这些快照 key 多样（strided_token init），于是读 softmax 95% 的注意力落在它们身上。** 结果：模型读到的"记忆"≈ 上下文头部 token 的静态快照池化，训练学到的压缩写入被 ~20:1 稀释，对长程贡献≈0。

---

## 1. 精确机制（代码实证，file:line）

### 写路径（稀疏）
- **选择**：`selector.py` TopKSelector，每 chunk 用 query·key 选 **top-k≈16** 个槽（idx [B,k]），硬 top-k + 直通梯度。
- **写规则**：`memory_bank.py:475-525` dual-gate / delta-rule，**只对选中的 k 个槽** scatter 更新：
  - delta-rule：`updated = current + g_in·(new_content − current)`
  - dual-gate：`updated = g_in·new + g_forget·current`
- **未选中的槽完全冻结**——无衰减、无 trickle、永不更新。
- 一个 chunk 一次写（chunk-level，已确认）。

### 读路径（稠密）
- `selector.py:1469+` MemoryCrossAttentionRead：`use_memory_xattn=True` 时
  - Q = live tokens，K/V = **所有 N 个槽**（不是 top-k！）
  - 独立 softmax over N，与写的 top-k 选择**完全解耦**。
- 即"宽读窄写"：读能看到所有槽，写只动 16 个。

### 槽初始化（病根之源）
- `slot_init=strided_token`：槽 i = 第 (i·stride)%T 个 token 的 hidden snapshot。
- **N=384 个槽里，一个 32k 样本全程只写到 ~24 个 → ~94% 永远是 chunk-0 token 快照。**

### 诊断证据（dead_slot_read_mass，selector.py:1563-1589）
- dead_slot_read_mass ≈ **0.95**，live ≈ 0.05
- per-slot 归一化后 dead:live 读占比 ≈ **1.00**（读对"是否写过"零偏好，纯按 key 相似度）
- 随容量：live 读占比 N128=29% → N384=5% → N896=5%（池子越大，写入越被稀释）

---

## 2. 为什么所有补丁都失败（已证伪方向回顾）
| 方向 | 做法 | 结果 |
|---|---|---|
| ROUTE-A arm4 | 强行均摊 usage_cov→1.0 | 长程崩（打散少数槽精确命中）|
| 写规则（DRoff）| delta-rule→dual-gate | N384 不掉但 N128 掉一半（写规则非长程关键）|
| v20 ArmA | soft-read-decay 软衰减未写槽读权重 | 无提升（40/37/36 ≈ base）|
| v20 ArmB | hard-eviction 按读 mass 硬淘汰 | 近崩、有害 |

**共性教训：在读/选择分布上做任何"强行干预"都伤长程。** 因为读分布落在未写槽不是 bug，是这个架构在当前 init+稀疏写下的稳定工作点——未写槽的 token 快照本身就是有用的检索信号（相当于一个免费的 chunk-0 KV cache），强行抹掉它反而丢信息。

---

## 3. 文献对照：别人怎么更新记忆

### DeltaNet（2406.06484，Yang et al. ICLR'25）
- **token-level**：每个 token 都更新状态矩阵 S。
- **dense write**：`S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ` —— 更新作用在**整个 S 矩阵**（外积写），不是选 top-k 行。
- delta rule 本质 = 先擦除旧的 k 关联（`−β k kᵀ S`）再写入新值 → **写入即检索式的关联记忆**，每步都重写"与当前 key 相似的那部分记忆"。

### Titans（2501.00663，Behrouz et al.）
- **token-level，surprise-gated**：写入量 ∝ 梯度（surprise）`∇ℓ(M; k,v)`，加 momentum + weight decay（遗忘）。
- **dense**：更新整个 memory MLP 参数（test-time learning），非稀疏槽。
- 关键：**memory 是一个被持续梯度更新的网络，不是一组冻结的快照槽**。NIAH 能扩到 2M。

### 共同点（与我们的差异）
| | 我们 | DeltaNet/Titans |
|---|---|---|
| 更新粒度 | chunk-level | token-level |
| 写入范围 | **top-k 稀疏（16/384）** | **dense（全状态）** |
| 未写部分 | 冻结在 init 快照 | 不存在"未写"——全状态每步演化 |
| 读 vs 写 | 解耦（宽读窄写）| 一体（写即改变全部可读状态）|
| key 来源 | 槽内容投影 | 输入 token 投影 |

**核心差异：他们没有"槽"的概念，记忆是一个稠密状态，每个 token 都重写它的相关部分。我们把记忆切成离散槽 + 稀疏 top-k 写，制造了"大量永不更新的槽"这个我们独有的病。**

---

## 4. 重设计候选（按推荐度排序）

### ★方案 A：dense soft-write（去掉 top-k 写，最小改动，最对症）
- **改动**：写入时不再 hard top-k，而是用 selection softmax 权重 `w[b,n]∈[0,1]` 对**所有 N 个槽**做加权写入：
  `slots[n] = slots[n] + w[n]·g_in·(new − slots[n])`
- **预期**：消灭"永不更新的槽"——每个槽都按相关度持续演化，读到的不再是 chunk-0 静态快照。
- **风险**：等于在写侧均摊，可能重蹈 arm4 覆辙（均摊伤精确命中）。但与 arm4 不同——arm4 动的是**读/选择**分布，这里动**写**，且保留 g_in 内容门控不强制均匀。**值得首试。**
- 代价：写从 O(k·d) 变 O(N·d)，N384 下 ~24× 写计算，但仍远小于 attention。

### 方案 B：init 改为零/可学习 + 读 mask 未写槽
- **改动**：`slot_init=zero`（或小随机），且读路径对"从未写过"的槽 mask 掉（不是软衰减，是训练时就不让读）。
- **预期**：强迫读只能用写入的内容 → 逼写路径学到有用压缩。
- **风险**：v20 ArmA/B 已部分试过软/硬干预读分布都失败；但那是在 strided init（未写槽含真信息）上抹信息。配合 zero-init（未写槽是垃圾）可能不同——**这是 A 失败后的次选**。

### 方案 C：token-level 增量写（最大改动，最贴文献）
- **改动**：放弃 chunk-level 单次写，改 chunk 内逐 token 的 DeltaNet 式 dense 外积更新。
- **预期**：直接对齐 DeltaNet/Titans 的成功配方。
- **风险**：架构大改，训练成本高，与现有 L1/L2/L3 stack 兼容性未知。**作为 A/B 都失败后的范式切换选项。**

### 方案 D（保守）：承认结论，收敛文档
- 若 A/B 都不超 P11 step500 baseline → mem-space 在此规模/数据下不优于纯长上下文，写成结论性报告。

---

## 5. 建议的执行顺序
1. **先把正在跑的 DRoff crossover 曲线（N192/256/512）收完**——确认"写规则价值随容量递减"，给"写侧本质无效"补最后一块证据。
2. **起方案 A（dense soft-write）**——最小改动、最对症、可直接在现有 trainer 上加 flag。判据：长程 qa5 是否首次超 P11 step500（48/45/44）。
3. A 出分后：升则深挖（多种子+扫 N），平/降则转方案 B。
4. B 也失败 → 方案 C（token-level）或 D（收敛）由用户拍板。

---

## 附：关键代码锚点
- 写：`src/memory/mem_space/memory_bank.py:384-586`（write，含 delta_rule/dual_gate 分支）
- 选择：`src/memory/mem_space/selector.py:200-517`（TopKSelector）
- 读：`src/memory/mem_space/selector.py:1469-1602`（MemoryCrossAttentionRead）
- 死槽诊断：`selector.py:1563-1589`（dead_slot_read_mass）
- init：`memory_bank.py:108-169`（init_from_hidden，strided_token）
