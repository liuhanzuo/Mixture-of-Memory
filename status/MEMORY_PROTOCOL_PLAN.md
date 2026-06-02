# Memory Protocol Redesign Plan (2026-06-01)

> 目标:把 memory 从"指望 LM loss 自发涌现语义分工"改成"显式设计 write/read 学习协议"。
> 依据:5 轮 routing 实验 + toy task 三臂诊断(commit `e5bb181`)。
> 实施方式:**一次只动一个变量,先在 toy task 验证,再上 Dolmino**。每步做完更新本文件勾选。

---

## ★★★ KEYSTONE 发现（2026-06-02，commit f73fd97）★★★

**真正的根因是一个 bug：`memory_bank.py` 的 `slot_value_norm_cap` 在 `torch.no_grad()` 里 rebind `self.slots`，切断了跨 chunk 写入梯度。** 修复后（grad-preserving norm-cap + inject_gate 加入可训练参数）：

- toy 2-chunk passcode **被解出**：`toy_r2_w0_s42_long`（1500 步，**weight=0 无 recon**）exact_acc 0→0.188，lm 3.0→0.78。
- **grokking 模式**：lm 在 ~3.0 平台期持续到 step~1045，然后相变骤降（1045=2.79→1195=1.40→1270=0.88→0.78）。
- **结论**：整个 5 轮 routing-collapse 调查都是这一个 severed-gradient bug 的下游表象。**cross-chunk grad 修复本身（不需要 recon）就解决了 toy**。recon/key-value 分离等是次要加速器，不是必需。
- ⚠️ 需 ~1000+ 步才 grok（所有 800 步 arm 仍停在相变前 lm~3.0）；当前 1 seed，正在 Round 3 多 seed 复现 + 测 recon 是否加速相变。

→ 这把 PLAN 重排为：**Bug1 修复 = 基石（已完成）**；P1 recon / P3 key-value / P4 双熵 = 加速/增强项，按"是否缩短 grok 相变 / 提升最终 acc"来评估，而非"是否解锁"。

---

## 证据基线(toy task 2-chunk passcode,500 步,commit e5bb181)

| 臂 | top1_sim | chunk1→2 overlap | retrieval_exact_acc |
|---|---|---|---|
| multi_query baseline | 0.018(≈均匀) | 0.125 | **0.0** |
| multi_query + force_gate(α=0.5) | 0.016 | 0.105 | **0.0** |
| **slot_query t40 + force_gate** | **0.32** | **0.29** | **0.0** |

**决定性结论**:
1. **routing 可修好** —— slot_query+temp40 拿到 top1_sim=0.32 / overlap=0.29(寻址成功)。
2. **但 exact_acc 全 0** —— 即使寻址对,写入的 slot 内容**读不回成答案**。
3. **force_gate 无效** —— gate 不是主因。
4. 真正缺口:**write protocol + slot 内容可重建性**,不是 routing,不是 gate。

---

## 改动清单(按优先级,逐个实现)

### [P1] summary reconstruction auxiliary loss ⭐最高优先
**状态: RUNNING（代码 v12/v13 已实现 commit d55b98c；Dolmino 对照实验进行中）**
- **toy 阶段结果（2026-06-02 toy_r4，已 deprioritized）**：反直觉——recon 臂 exact_acc=0 vs base 0.125（即 toy 上 recon **未**帮助、甚至更差）。但 toy 已被判 inconclusive（loop 功能正常但加速器/指标信号弱），团队 03:33 pivot 到真数据。
- **Dolmino 对照（2026-06-02 13:00 heartbeat 启动）**：
  - 本机 H20: `dolmino_bugfix_slotq_t2h`（slot_query temp40, **l_recon_weight=0**, seed42, 2000步）= no-recon control。
  - 远程盘A H20 28.59.80.196: `dolmino_recon_diskA`（**完全相同配置 + l_recon_weight=0.1**）= recon 臂。
  - **2026-06-02 23:20 重启（关键 infra 修复后）**：上述两臂均在 **step~490-493 NCCL hang → 2h watchdog SIGABRT**（与之前 3 个 run 同一确定性死点）。**根因被推翻**：不是 step500 save barrier（无 [save] log，hang 早于 save），而是 `train_mem_space_dolmino_cpt.py:1166` 手动 grad allreduce 被 per-rank 条件 (`step_valid_micros>0` + `p.grad is not None`) 守卫 → 某 rank 因 slot 未选中(uniq_sel_slots=0)跳过该 param 的 all_reduce，其它 rank 仍发 → collective size 不一致(262144 vs 16777216) → 挂死。fix commit **6b6b134**：每 rank 遍历完整 trainable，缺失 grad 补零，all_reduce 序列跨 rank 严格一致。
  - 重启后：本机 `dolmino_norecon_local_v2`(l_recon=0) + 盘A `dolmino_recon_diskA`(l_recon=0.1)，均 commit 6b6b134。决定性测试 = 能否越过 step~490-493 + step500 save。
  - **判据**：对比两臂 lm + 离线 BABILong + top1_sim 轨迹。recon 臂若 retrieval/eval 明显更好 → P1 在真数据上成立（toy 是 artifact）；若无差异或更差 → 转 P2 或查 read-path。
- **改什么**:写入后的 slot value → 用小 cross-attn decoder 重建 chunk 的 L3 summary tokens;`L_recon = MSE(S_hat, stopgrad(S_L3))`,λ_recon 从 0.05~0.1 起。
- **为什么**:toy 证明这是唯一确认缺失的环节——writer 没有近距离目标教它"存可读回的内容",只靠 LM loss 太绕。
- **先验证**:在 toy task 上,给已寻址成功的 **slot_query+t40** 配置加这个 loss,看 retrieval_exact_acc 能否从 0 起来。
- **判据**:exact_acc > 0(哪怕 0.3)就证明 reconstruction 是关键拼图。若仍 0 → 转去查 read-path(slot_to_hidden / α 融合把 slot value 注入 LM 的通路本身)。
- **文件**:selector/layer(取 write slot value)、新增小 decoder 模块、config 加 `l_recon_weight`、train + toy 脚本接 aux key。

### [P2] read selector / write allocator 接口分离
**状态: PENDING(依赖 P1 结论)**
- **改什么**:`read_idx = ReadSelector(current chunk query, slot_keys)`(选旧 slot 读);`write_idx = WriteAllocator(post-chunk summary, slot_keys, usage/age)`(选 slot 写新信息,含 allocation 行为而非纯 retrieval)。先用简单版:write_idx 暂时也用 slot_score topk,但**接口先拆开**。
- **为什么**:读(需要过去什么)和写(产生了什么值得存、放哪)是两个不同问题。新 fact 写入时相关 slot 尚不存在,纯 content-based read router 找不到 → 需要 allocation。
- **判据**:write_unique_slots_per_chunk 上升、新 fact 有稳定落点。

### [P3] slot 拆 key / value
**状态: PENDING**
- **改什么**:`slot = {key_i(routing用), value_i(prepend/readout用)}`;writeback 分别更新:`key ← LN(EMA(key, new_key))`(normalize,防漂)、`value ← RMSNorm(EMA(value, new_value))`(可有 norm)。routing 用 key,prepend 用 `slot_to_hidden(value)`。
- **为什么**:当前单向量同时承担 routing/写入/投影/存储,目标互相打架;且 value norm 长大(实测→5)会破坏 routing 空间。
- **判据**:slot norm 增长不再拖塌 top1_sim;key_max_cos 稳定。

### [P4] 双熵 routing 正则(per-sample sharp + global balanced)
**状态: PENDING**
- **改什么**:区分两个熵——单 chunk 的 `p(slot|chunk)` 要**尖锐**(low entropy,甚至加 L_sharp 最小化);整个 batch/moving-average 的 slot usage 要**均衡**(high global usage entropy,防 dead slot)。检查并修正现有 entropy/load_balance aux——它们可能在奖励 uniform routing(帮倒忙)。
- **为什么**:理想 routing 不是"每 chunk 对 128 slot 均匀",而是"单 chunk sharp + 全局 balanced"。当前 aux 可能混淆了两者。
- **判据**:per-sample top1_sim 升、无 dead slot。

### [P5] read-back 一致性 weak supervision(synthetic only)
**状态: PENDING**
- **改什么**:synthetic KV task 里记录 fact 写入的 slot set W,未来 query 检查读取 slot set R,加 `L_reread = -log Σ_{i∈W} p_read(i|query)`。
- **为什么**:直接教 routing protocol——"写进去的地方将来要找得回"。比 top1_sim 更本质的可寻址性指标。

### [P6 降级] gate warmup
**状态: DEPRIORITIZED**
- toy 证明 force_gate(α=0.5)对 exact_acc 零改善 → gate 不是主因。**暂不做**,等 P1-P3 后若 α 仍钉死再回看。

### [不做] persistent slot identity 多字段 / future-summary prediction / anti-generic loss
- age/usage/confidence 多字段、方案C future prediction、anti-generic cos penalty:**先不做**,避免一次引入太多自由度难归因。等 P1-P3 验证后再评估。

---

## 新增诊断(随实现逐步加)
- 写入:write_slot_entropy_per_chunk、write_unique_slots、slot_key_delta_norm、slot_value_delta_norm
- 读回:overlap = |W∩R|/|W|(已有 chunk1to2_overlap 雏形)
- specialization:intra_slot vs inter_slot summary similarity(intra 应 > inter)
- anti-generic:cos(slot_i, global_mean_slot)

---

## 实施顺序（并行可归因路径 —— 16×H20 可用，2 节点）：盘A 本机 8×H20 + 盘A 远程 28.59.80.196 8×H20 + 盘B 28.49.196.161 8×H20 + 盘B 29.162.241.149 8×H20 = **32 卡（4 节点）**。toy arm = 1 卡 / 4 分钟（500 步）。
→ 可一次并行 ~32 个 toy ablation。盘B 用前需 rsync 代码（见 CODEBUDDY.md）。策略：**每个 arm 只相对同一 baseline 改一个变量**（保持可归因），但多个变量同时铺开测，不串行等。

**Round 1(P1 验证 + 超参扫,toy,~16 arms 并行)**:
- 底座固定 slot_query + temp40 + force_gate off。
- 变量铺开:`l_recon_weight ∈ {0(control), 0.05, 0.1, 0.3, 1.0}` × `stopgrad target {on/off}` × `seed {42,43}` 等。
- 金标准:retrieval_exact_acc。找出 recon loss 是否让 exact_acc>0 + 最优权重。
- 同时可捎带:temp {30,40,60} 复扫、top_k {8,16,32} 在 recon 开启下的表现。

**Round 2(P3/P2/P4 叠加,toy,并行)**:
- 在 Round1 最优 recon 配置上,**并行**测三个独立改动各自的增量:
  - arm-A: + key/value 分离(P3)
  - arm-B: + read/write 接口分离(P2)
  - arm-C: + 双熵正则(P4)
  - arm-D: A+B+C 全叠加
- 每个配 1-2 seed,一次铺开 ~8-10 arms。

**Round 3(Dolmino 真训练,2 节点并行)**:
- toy 闭环通过(写入→保持→读回)后,上 Dolmino。
- 本机跑最优全配置,远程跑一个消融(如去掉某个 aux)做对照。8-GPU DDP each。

**核心原则**:先证明 slot 能学会稳定 write-read protocol(toy exact_acc 起来),再追 LongBench/BABILong。利用多卡把每个 Round 的等待从"串行天"压成"并行小时"。

---

## 当前 routing 模式现状(已实现,供复用)
- `routing_pool_mode`: max_pool / chunk_query / multi_query(均塌缩) / **slot_query(寻址最好,temp 越高越锐,temp40 最佳)**
- slot_query + temp40 是目前 P1 验证的推荐底座。
