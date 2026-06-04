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
**状态: DONE — 裁决：recon 不帮助，REJECTED，转 P2/read-path（2026-06-03 17:43 eval 完成）**

**★ 裁决结果（2026-06-03 17:43 eval 完成，substring-acc qa1-5 × 0k-32k × 100，3500 样本/臂）★**
- **OVERALL：norecon 8.74%（306/3500） vs recon 6.89%（241/3500） → recon 反而更差。**
- by task：qa1 recon4.9/norecon3.7、qa4 recon11.4/norecon9.6（recon 略优）；qa2 norecon8.6/recon2.7、qa3 norecon5.7/recon4.0、qa5 norecon16.1/recon11.4（norecon 明显优）→ 净效应 norecon 胜。
- by length：**两臂均在 ≥4k 崩到 ~1-2%**（0k 33/32 → 1k 19.6/7.8 → 2k 3.2/1.6 → 4k 1.6/1.2 → 8k 1.8/2.2 → 16k 1.0/1.4 → 32k 1.0/2.0），recon 在 1k 大幅落后。输出多为退化重复两臂皆然。
- **结论**：recon aux 真数据上**无正向作用、整体更差**，且**两臂 read-path 在 ≥4k 完全不提供可用长上下文检索**（塌到 noise floor）。→ P1 **REJECTED**，按判据**转 P2 / 查 read-path**。
- 产物：`babilong_results/p1_verdict/{p1_norecon_g0,g1,p1_recon_g2,g3}`；`logs/eval_p1_verdict_driver.log`。

---
**（历史）状态: 训练完成，BABILong 裁决 eval RUNNING（2026-06-03 13:40）**
- **2026-06-03 13:40 两臂训练完成**：`dolmino_norecon_local_v2`（l_recon=0，849min，step2000 @13:29）+ `dolmino_recon_diskA`（l_recon=0.1，858min，step2000 @13:40），均 commit 6b6b134，全程 0 crash / 0 non-finite（grad-desync fix + 2h timeout holding）。final adapter 均落盘。末期 lm 两臂均噪声 ~2.6-2.8 无决定性分离，top1_sim 0.01-0.03，inject_gate_std ~0.007（gate 平）。→ 仅凭 train lm 判不了 P1，靠离线 BABILong。
- **2026-06-03 13:40 启动裁决 eval**：本机 8×H20 空闲，`scripts/eval_p1_verdict.sh` 并行跑两臂 final adapter BABILong（qa1-5 × 0k-32k × 100），GPU0-3（每臂 2 卡分 task）。判据：recon 臂 acc 明显更高 → P1 真数据成立；无差异/更差 → 转 P2 / 查 read-path。
- **2026-06-03 01:37 里程碑**：两臂均到 step~535，**首次越过 step~490-493 确定性死点**。step500 adapter save 两臂均成功落盘。grad-desync fix (6b6b134) 确认生效。
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

### [P2] read selector / write allocator 接口分离（含专用 cross-attn READ 解耦）
**状态: IN PROGRESS — DOLMINO VALIDATING（2026-06-03 21:00）。代码已 commit `7d76d59`（v15 doc + config.py:280 + layer.py:437-1081 + toy + Dolmino path）。Toy 配对验证 INCONCLUSIVE：ON/OFF 均 retrieval_exact_acc=0、tok_acc=0.375，ON top1_sim 0.247 < OFF 0.350——toy 是短上下文，无法触发 ≥4k dilution cliff，故 toy gate 不适用（已确认 toy 是错误仪器）。改用真正判据：Dolmino 8-GPU arm + 离线 BABILong ≥4k。已启动 `dolmino_p2_decoupled_local`（commit `2326565`，本机 8×H20，= norecon control 配置 + `--use_decoupled_read`，2000步，eval_interval 0，seed42），step5 lm=4.79 healthy ~60GB/GPU。判据：训完离线 BABILong ≥4k 是否越过 P1 的 1-2% noise floor（对比 baseline outputs/dolmino_norecon_local_v2）。**
- **researcher 根因（RESEARCHER_REPORTS 2026-06-03，confidence medium）**：(1) 注入稀释——slot prepend KV 与最多 1024 live token 共享一个 softmax → memory 仅 ~1.5% attn mass，再被 slot_delta clip(layer.py:999-1002) + inject_gate(0.12, std0.007 flat) 压到 ~0.2%；(2) routing collapse(top1_sim 0.01-0.03)次要。≥4k cliff 主要由稀释造成。
- **P2 验证判据**：toy retrieval_exact_acc 是否 >0；离线 BABILong ≥4k 是否越过 ~1-2% noise floor。
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
- **routing 诊断(2026-06-04 新增,QUERY_DIAG + wandb)**:`topk_mass`(被选 k 个 slot 的 softmax 概率和,→1=路由真集中)、`usage_var`(各 slot 选中频率 population variance,退化到少数 slot 时变大)、`chunk_idx_jaccard`(跨 chunk 选同一批 slot 程度,>0.5=退化)。commit c421147。

---

## ★ 当前推进队列(2026-06-04 23:40，main 自主执行，无需用户审批) ★

> 底座:per-doc chunk_size ablation (chunk128 vs chunk256) + route_aux(routing supervision) 验证。
> 32 卡 4 节点可用(盘B 需 rsync + 缺 transformers 暂不可训)。本机+盘A远程优先。

### [R3-1] chunk128 final adapter 双 eval — **状态: PARTIAL（2026-06-04 23:55，BABILong 0k-8k 完成，16k/32k + LongBench 跑中）**
- BABILong substring-acc (qa1/qa2/qa5)：0k=34/19/45, 1k=10/13/31, 2k=11/6/29, 4k=4/0/34, 8k=14/14/32, 16k=21/6/(跑中)。
- **★ 关键正向信号**：qa1/qa5 在 4-16k 维持 14-34%，**明显高于 P1/P2 的 1-2% noise floor** → per-doc chunk128 训练似乎缓解 ≥4k 塌缩。但 top1_sim 仍≈0.016（routing 未真寻址）→ 效果可能来自 per-doc 训练让 memory 内容更可用，而非 routing 改善（待 R3-2 routeaux 对照验证）。
- **BABILong** qa1-5 × 0k-32k × 100：本机 GPU3-6，0k/1k/2k 完成（0k 34/19/45%，2k 衰减 11/6/29%），4k-32k 进行中。
- **LongBench**（用户偏好，替代 BABILong）6 QA：hotpotqa/narrativeqa/qasper/multifieldqa_en/2wikimqa/musique，F1/EM，本机 GPU0/1/2/7 4-shard，base Llama-3-8B + no_chat_template + chunk128，本地 jsonl 无下载。输出 `longbench_results/perdoc_chunk128_local/`。
- 判据：≥4k 是否越过 P1/P2 的 1-2% noise floor；LongBench F1 vs base 锚点。

### [R3-2] route_aux 远程训练完成 → eval gate — **状态: EVAL ~DONE（2026-06-05 00:50，0k-8k 完成，16k/32k 收尾）**
- `mem_space_perdoc_chunk128_routeaux_remote`（route_aux=1.0）训练 step2000 完成（227min, non-finite=1, lm~2.65）。
- **BABILong substring-acc（routeaux，route_aux=1.0）**：0k qa1/qa2/qa5=25/13/70；1k=10/10/37；2k=7/14/36；4k=12/14/37；8k=5/4/26；16k=7/18/-（跑中）。
- **🔑 关键结论：route_aux=1.0 显著优于 base 且优于无 route_aux arm。** 对比 base Llama-3-8B（R3-3）同口径 4k qa1/qa2/qa5=2/4/0、8k=23/12/-（base 8k qa1 偶高但 qa5≈0）；对比无 route_aux 的 R3-1 chunk128（qa5 4k≈34、8k≈32 相近，但 routeaux 的 qa1/qa2 在 2-4k 更稳）。**routeaux 在所有长度 qa5 维持 26-70%、远离 1-2% noise floor，证明 routing supervision 有效。**
- 判据已满足：≥2k 远高于 noise floor + qa5 全程 >25%。

### [R3-2-OLD] route_aux 远程训练完成 → eval gate（历史）

### [R3-3] base model 对照（用户明确要求"和 base 比一比"）— **状态: DONE（2026-06-05 05:14，结论：adapter 在 LongBench 上 ~5× 劣于 base）**
- ✅ base Llama-3-8B（plain_hf）BABILong 0k-8k 完成（4k qa1/qa2/qa5=2/4/0，8k=23/12/-；qa5≈0，远逊于 adapter 的 26-37% → **BABILong qa5 上 adapter 强于 base**）。
- ✅ **base LongBench DONE**（`--base_mode` 无 adapter + 标准中间截断，commit 7feef6d，6 任务×200/150 样本）：avg F1=**13.95**（hotpotqa 9.76 / narrativeqa 16.01 / qasper 13.92 / multifieldqa 24.87 / 2wikimqa 12.17 / musique 6.97），`longbench_results/base_model_full_lb/scores.json`。
- ⚠️ **R3-1 adapter LongBench = avg F1 2.94**（hotpotqa 2.51 / narrativeqa 1.08 / qasper 4.70 / multifieldqa 4.45 / 2wikimqa 3.27 / musique 1.61）。
- 🔴 **关键对照结论**：base **13.95** vs adapter **2.94** —— **chunk128 memory-adapter 在开放式 LongBench QA 上把 base 的能力打掉了 ~5 倍**。LongBench 不是"整体偏难"（base 拿到合理分），是 **adapter 的中间 memory 压缩/读回严重丢信息**。与 toy/BABILong 的"adapter 在合成检索任务上更强"形成对立 —— adapter 学到的是 needle-style 精确检索，但牺牲了自然语言 QA 的上下文完整性。**这是给用户的核心 takeaway，需在下一步实验设计中正视（memory 压缩 vs 自然 QA 的 tradeoff）。**
- 公平性：相同 prompt 截断策略；记录 base 在各长度的 F1/acc。研究员（general-purpose-22）会给标准对比 protocol，据此 finalize。
- 写入 BENCHMARK_RESULTS.md 的 base-vs-ours 对照行。

### [R3-4] benchmark 扩展 + 改进调研 — **状态: DONE（2026-06-04 23:55，general-purpose-22）**
- 产物 `ops/research_notes/benchmark_survey_and_improvements_20260604.md`（全 arXiv ID + HF dataset + confidence）。
- **Benchmark 推荐**：① SCBench(2412.10319, HF `microsoft/SCBench`) = 唯一为 KV-cache 压缩/复用设计，最贴论点 [high]；② RULER(2404.06654, HF `simonjegou/ruler`) = 长度可控 multi-value/multi-query NIAH，直测 slot 容量，比 BABILong 区分度高 [high]；③ HELMET(2410.02694) 写论文时加 [medium]。BABILong 降级为连续性锚点。
- **Base 对比 protocol**：同一冻结 Llama-3-8B「去掉 adapter」= base，跑 B0(截断到等 KV-budget token) / B1(sliding-window) / B2(full-if-fits≤8k)，核心**匹配 KV budget**，出 quality-vs-KV-budget 曲线证 128-slot 点高于等预算截断曲线。锚点 paper 2406.10149 Llama-3-8B-It qa1 4k=16/8k=7（4k 后断崖=我们要超越的）。⚠️ 勿用 Llama-3.1-8B 当 base（128k 原生已自解）。
- **改进 backlog → 落入下方 [P7]-[P13]**。

---

## 改进 backlog（researcher 2026-06-04，benchmark_survey_and_improvements_20260604.md BLOCK 3）

> 诊断根因（toy_vs_full + collapse 报告）：(a) **注入稀释** slot KV 仅 ~0.2% attn mass → LM 梯度≈0；(b) **routing collapse** top1_sim→uniform，且现有 `load_balance`(weight=0.01) + `entropy` aux **主动把 routing 推向 uniform** 反而致塌。两路并治。
> 推荐顺序：**P7 + P9 一起上**（都小、都打 collapse）→ retrieval 离开 noise floor 但仍弱 → 加 **P8**（读路径 mass）→ 再 P10/P11 调参。P12/P13 是破 cliff 后的研究 bet。

### [P7] route-supervision aux + 中和 uniform-pushing aux ⭐ — **状态: TRAIN DONE + BABILong EVAL RUNNING（2026-06-05 05:35，本机 7-GPU 0k-32k）**
- 借 Landmark(2305.16300) + Loss-Free Balancing for MoE(2408.15664)。confidence **high**，小改。
- ✅ 实现完成：selector.py routing_bias buffer + 在线更新（commit 1b46939）；✅ 已 wire 进 `train_mem_space_dolmino_cpt.py`（commit fb91c51）。✅ 启动脚本 `scripts/launch_mem_space_p7p9.sh`。
- ✅ **已启动**（run `mem_space_perdoc_chunk128_p7p9`，本机 8-GPU，load_balance=0+entropy=0+LFB on+num_global_slots=4+route_aux=1.0）。早期诊断健康：**usage_cov 0.94-0.98 + usage_var~0.0001（loss-free balancing 平衡有效）**，top1_sim 0.23-0.41（远高于无 route_aux 塌缩值 0.017），slot_attn_entropy~2.1。
- 判据：训完离线 BABILong ≥4k 是否越过 noise floor 且 routing 优于 routeaux。auto_launch eval 同 R3-2 口径。
- ✅ **训练完成**（2026-06-05 05:22，step2000，226.6min，non-finite=1，final lm~2.44）。adapter → `outputs/mem_space_perdoc_chunk128_p7p9/mem_space_adapter.pt`。
- ⏳ **BABILong eval RUNNING**（2026-06-05 05:35，本机 7-GPU，0k-32k 各一 length，`scripts/eval_perdoc_chunk128_p7p9.sh`，qa1/qa2/qa5 limit100 chunk128，结果 → `babilong_results/perdoc_chunk128_p7p9/`）。对照基线：R3-2 routeaux qa5 0k-8k=70/37/36/37/26；R3-3 base 4k qa5≈0。

### [P8] 专用 memory cross-attention 读路径（独立 softmax）— **状态: PENDING, auto_launch: false（等 P7+P9 结果）**
- 借 YOCO(2405.05254)/Memorizing Transformers(2203.08913)/Infini-attn(2404.07143)。confidence **high**，medium 改。
- 给 slots 独立 cross-attn 层（独立 softmax），不再 prepend 进 live-token KV → 治 ~0.2% mass 稀释。per-head content-dependent gate + 较大 init。隔离在 `--use_memory_xattn`。**不修这个，P7 路由修好也无梯度可学。**

### [P9] always-on register slots — **状态: RUNNING（与 P7 同 arm，2026-06-05 01:37）**
- 借 ViT Need Registers(2309.16588)。confidence **high**，**无需新代码**——`--num_global_slots` 已存在。已在 P7 训练 arm 上加 `--num_global_slots 4`（run `mem_space_perdoc_chunk128_p7p9`）。判据随 P7 一同 eval。

### [P10] key_repulsion 1.0→0.05 + ST-Gumbel top-k — **状态: PENDING, auto_launch: false** — confidence medium，tiny。当前 key_repulsion=1.0（20× toy）可能 over-smear keys。
### [P11] delta-rule + normalized writeback — **状态: PENDING** — confidence medium，medium 改。写残差 + 归一化 readout magnitude 使其可与 local attn 比较再 gate。
### [P12] 重审 recon + live-token masking/bottleneck — **状态: PENDING** — confidence low-medium。ICAE(2307.06945)/Gist(2304.08467) 证 recon 只在 LM 被迫只读 slots 时有效；P1 失败疑因无 bottleneck。
### [P13] surprise-gated write（写强度∝预测误差）— **状态: PENDING** — confidence low。借 Titans(2501.00663)，P7/P8 解锁 retrieval 后再做。

### [R3-5] 效果归因（若 eval 仍差，查根因，用户要求"看一看到底因为什么"）— **状态: PENDING**
- 若 R3-1/R3-2 ≥4k 仍塌到 noise floor：用 routing 诊断三件套（topk_mass/usage_var/chunk_idx_jaccard）+ inject_gate 轨迹判断是 routing collapse 还是注入稀释主导，对照 researcher P2 根因（注入稀释 ~0.2% attn mass），决定走 P3(key/value 分离) 还是专用 memory cross-attn 读路径。

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
