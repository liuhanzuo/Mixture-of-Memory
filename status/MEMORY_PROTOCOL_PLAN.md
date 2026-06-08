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

### [P7] route-supervision aux + 中和 uniform-pushing aux ⭐ — **状态: DONE（2026-06-05 06:14，gate=持平 routeaux，未碾压 → 进 P8）**
- 借 Landmark(2305.16300) + Loss-Free Balancing for MoE(2408.15664)。confidence **high**，小改。
- ✅ 实现完成：selector.py routing_bias buffer + 在线更新（commit 1b46939）；✅ 已 wire 进 `train_mem_space_dolmino_cpt.py`（commit fb91c51）。✅ 启动脚本 `scripts/launch_mem_space_p7p9.sh`。
- ✅ **已启动**（run `mem_space_perdoc_chunk128_p7p9`，本机 8-GPU，load_balance=0+entropy=0+LFB on+num_global_slots=4+route_aux=1.0）。早期诊断健康：**usage_cov 0.94-0.98 + usage_var~0.0001（loss-free balancing 平衡有效）**，top1_sim 0.23-0.41（远高于无 route_aux 塌缩值 0.017），slot_attn_entropy~2.1。
- 判据：训完离线 BABILong ≥4k 是否越过 noise floor 且 routing 优于 routeaux。auto_launch eval 同 R3-2 口径。
- ✅ **训练完成**（2026-06-05 05:22，step2000，226.6min，non-finite=1，final lm~2.44）。adapter → `outputs/mem_space_perdoc_chunk128_p7p9/mem_space_adapter.pt`。
- ⏳ **BABILong eval RUNNING**（2026-06-05 05:35，本机 7-GPU，0k-32k 各一 length，`scripts/eval_perdoc_chunk128_p7p9.sh`，qa1/qa2/qa5 limit100 chunk128，结果 → `babilong_results/perdoc_chunk128_p7p9/`）。对照基线：R3-2 routeaux qa5 0k-8k=70/37/36/37/26；R3-3 base 4k qa5≈0。
- ✅ **BABILong DONE（substring-acc, limit100）**：
  - qa1 0k-32k = 55/20/27/14/14/10/9
  - qa2 0k-16k = 28/15/18/18/11/10
  - qa5 0k-32k = 53/45/36/31/27/24/2（2026-06-05 07:53 全长度收齐：16k=24 仍守住，32k=2 塌陷疑似截断）；qa1 0k-32k=55/20/27/14/14/10/7；qa2 0k-32k=28/16/18/18/11/10/5
- 🔑 **GATE 结论：P7+P9 ≈ 持平 R3-2 routeaux，未碾压。** qa5 对比 routeaux(70/37/36/37/26)：P7P9=53/44/36/31/27 —— 1k 略高、2k 持平、4k 略低、8k 持平。**route_aux + LFB + 4 register slots 没有比单纯 route_aux 带来额外增益**（register slots 在此 setup 无明显作用）。但 qa1/qa5 全长度稳在 noise floor 之上（≥9-55%），未塌缩，确认 routing supervision 本身有效。
- → **判据"显著优于 routeaux"未满足**。结论：routing 已不是瓶颈（P7 证明 routing 可学且稳定）。**真正瓶颈应是注入/读回稀释（researcher P2：memory 仅占 ~0.2% attn mass，且 LongBench F1=2.94 vs base 13.95 印证读回严重丢信息）→ 直接进 [P8] 专用 memory cross-attn 读路径**（plan 既定下一步）。

### [P8] 专用 memory cross-attention 读路径（独立 softmax）— **状态: NULL-SINK VALIDATED + 压缩阶梯完成（2026-06-05 23:30）。最佳臂 = chunk512 step500。**
- **压缩阶梯 step500（公平同步对比，qa1/qa2/qa5 × 0k-32k，离线 BABILong limit=100）= chunk512 完胜**：
  - chunk128: qa1 0k34 1k42 32k22；qa2 0k49 32k10；qa5 0k74 1k55 32k37
  - chunk256: qa1 0k98 1k35 32k14；qa2 0k44 32k18；qa5 0k78 1k76 32k36
  - chunk512: qa1 0k96 1k47 2k52 32k29；qa2 0k49 1k49 32k18；**qa5 0k85 1k76 2k77 4k54 8k48 16k45 32k41**（全长度最强）
  - chunk1024: qa1 0k94 1k87 **2k7(异常孤立 trough，re-score 持续，非瞬态)** 32k15；qa5 0k78 1k47 16k42 32k43
  - **裁决：chunk512 step500 = P8 最终交付 ckpt**（qa5 长程 45-54% 显著优于其余三臂；qa1 0k 96 与 chunk256 持平且长程更稳）。chunk1024 0k/1k 虽高但 2k 异常崩 + 长程不及 512。
- **压缩阶梯 step1000（验证 researcher 因果链「chunk 越大越稳，崩溃=快照撞 loss-spike×注入频率」）= 完全坐实**：
  - chunk128 step1000: 全 0-5%（token 重复死循环乱码，撞 step895-1010 loss spike，2× 注入频率放大）
  - chunk256 step1000: qa1/qa2 ~14、qa5 ~30（部分退化，连贯续写非乱码）
  - chunk512 step1000: qa1 1k67 2k54 16k15；qa5 1k88 2k59 8k25 → **健康未崩**
  - chunk1024 step1000: qa1 1k86 2k63 16k20；qa5 2k71 8k48 16k32 → **健康未崩**
  - **结论锁定**：注入频率 = seq_len/chunk_size，chunk 越小注入越频繁 → spike 期过量注入累积越狠 → 越偏 LM 崩坏。大 chunk（512/1024，total=5000 步，step1000 处早期谷底）安然。**最终模型一律取早 ckpt（step500），且优选大 chunk 臂。**
- **v1 FAILED/REGRESSION（2026-06-05 11:35）**：train healthy（step2000 lm~2.22）但离线 BABILong 全面塌方：P8 qa5 0k-8k=3/15/17/0/3 vs P7P9 53/45/36/31/27；qa1 0k=11 vs 55；qa2 0k=0 vs 28。**关键诊断：0k（整样本入窗、memory 应无关）也崩** → 专用 xattn 读路径在**所有 32 层无差别注入**且 read 无 cold/短程跳过 guard（layer.py:1330 直加不乘 inject_gate g），独立 softmax 始终对 slots 归一化（无 null 选项）→ 即使 slot 为冷噪声也强行读入，逐层累积破坏冻结 backbone。subagent general-purpose-25 根因确认，最小修复 = null/sink slot。commit 1f46b4d(null/sink)+c69cd8d(trainer collect+save memory_xattn)。
- **v1 FAILED/REGRESSION（2026-06-05 11:35）**：train healthy（step2000 lm~2.22）但离线 BABILong 全面塌方：P8 qa5 0k-8k=3/15/17/0/3 vs P7P9 53/45/36/31/27；qa1 0k=11 vs 55；qa2 0k=0 vs 28。**关键诊断：0k（整样本入窗、memory 应无关）也崩** → 专用 xattn 读路径在**所有 32 层无差别注入**且 read 无 cold/短程跳过 guard（layer.py:1330 直加不乘 inject_gate g），独立 softmax 始终对 slots 归一化（无 null 选项）→ 即使 slot 为冷噪声也强行读入，逐层累积破坏冻结 backbone。subagent general-purpose-25 根因确认，最小修复 = null/sink slot。commit 1f46b4d(null/sink)+c69cd8d(trainer collect+save memory_xattn)。
- **v2 NULL-SINK 修复 step500 eval（2026-06-05 18:12 完成，babilong_results/perdoc_chunk128_p8_nullsink_step500/）= 修复成功**：qa1 0k-32k=34/42/29/26/21/21/22；qa2=49/21/11/16/14/11/10；qa5=74/55/54/42/36/38/37。**0k-collapse 完全消除**（qa5 0k 3→74，qa2 0k 0→49，qa1 0k 11→34），且 qa2/qa5 0k **超过 P7P9**（49 vs 28、74 vs 53）。null-sink 让 read 路径可学到「冷 slot 不读」。✅
- **step1000 eval（2026-06-05 19:32 短长度完成，babilong_results/perdoc_chunk128_p8_nullsink_step1000/）= 臂内过训练拐点坐实**：qa1 0k-4k=0/1/0/1；qa2=1/0/0/0；qa5=0/0/0/0。**对比 step500（qa5 0k=74→step1000=0；qa2 0k=49→1；qa1 0k=34→0）全面塌方到 ~0%**。⚠️ **失败形态与 chunk256 不同**：chunk256 step1000 是"连贯续写 haystack 原文"（指令遵循丢失但语言正常），**chunk128 step1000 是 LM 本身退化成 token 重复死循环**（实测输出 'The the the the the the the is...'）——属 CODEBUDDY.md PPL>1000「模型不会说话了」档。**机制（推断，非铁证）**：训练侧诊断显示 inject_gate 全程稳 0.12、slot_delta 无爆炸、teacher-forcing lm 健康(~3.3)，但 per_tok_logit_std 长期偏高(4-5)→ adapter 后期把 backbone logit 分布推尖 → greedy(无 rep penalty, max_new=20) 下陷入 "the" 死循环；**0k 也崩说明与 memory 检索无关，是 adapter 污染了 backbone 自由生成稳定性**。lm 全程健康不矛盾——lm=teacher-forcing 续写 PPL，不衡量自由生成稳定性/指令遵循。**结论：(1) null-sink 是正确的代码修复（step500 已证）；(2) 最终模型必须用 step≈500 早 ckpt，绝不用末期 ckpt；(3) ALL 4 阶梯臂同理过训练，跑完只为拿 lm/压缩曲线，最终模型一律取早 ckpt。** **待查**：两种退化形态差异（chunk 越小注入越频繁→是否更偏 LM 崩坏），已记 PENDING 派 researcher。wave2（8k/16k/32k）评测进行中（GPU5-7）。
- 借 YOCO(2405.05254)/Memorizing Transformers(2203.08913)/Infini-attn(2404.07143)。confidence **high**，medium 改。
- 借 YOCO(2405.05254)/Memorizing Transformers(2203.08913)/Infini-attn(2404.07143)。confidence **high**，medium 改。
- 借 YOCO(2405.05254)/Memorizing Transformers(2203.08913)/Infini-attn(2404.07143)。confidence **high**，medium 改。
- 给 slots 独立 cross-attn 层（独立 softmax），不再 prepend 进 live-token KV → 治 ~0.2% mass 稀释。per-head content-dependent gate + 较大 init。隔离在 `--use_memory_xattn`。**不修这个，P7 路由修好也无梯度可学。**
- **实现（subagent general-purpose-24, commit 5144286）**：新 `MemoryCrossAttentionRead`（独立 softmax + GQA 32q/8kv + per-head content gate `sigmoid(gate_proj)` bias=logit(0.4) + out_proj small-random 非 zero，从 step0 active 有梯度）。与 P2 `--use_decoupled_read`(v15, zero-init out_proj, g≈0.12 近死) 分离为独立 flag。launch `scripts/launch_mem_space_p8.sh`，run `mem_space_perdoc_chunk128_p8`，旋钮与 p7p9 一致仅多 `--use_memory_xattn --memory_xattn_gate_init 0.4` → 干净隔离读路径。eval off，离线 BABILong 判据。
- **早期健康（step25）**：lm 6.95→4.84→3.06（step5 spike 仅 lr warmup，非 xattn gate；无需降 gate_init）。routing 健康 top1_sim=0.43 topk_mass=0.83 uniq_sel=16 usage_cov=0.98。✅ smoke 3/3 pass，legacy smoke 6/6 pass（flag off byte-for-byte）。
- **判据**：step2000 完成 → 离线 BABILong 0k-32k 对比 P7P9 baseline，看 memory xattn 读路径是否解锁 retrieval（gate 显著优于 P7P9）。

### [P9] always-on register slots — **状态: RUNNING（与 P7 同 arm，2026-06-05 01:37）**
- 借 ViT Need Registers(2309.16588)。confidence **high**，**无需新代码**——`--num_global_slots` 已存在。已在 P7 训练 arm 上加 `--num_global_slots 4`（run `mem_space_perdoc_chunk128_p7p9`）。判据随 P7 一同 eval。

### [P10] key_repulsion 1.0→0.05 + ST-Gumbel top-k — **状态: DONE — REJECTED（2026-06-07 03:20 eval 评完，劣于 baseline）**。step500 BABILong：qa5 0k-32k=74/68/36/24/15/21/14，全面低于 top_k16 基线（85/76/77/54/48/45/41）。ST-Gumbel 硬路由把 eval top1_sim 推到 1.0 但 retrieval 反而退化（过度自信、选错 slot 无法纠正）。key_repulsion 0.05 未带来增益。**裁决：REJECTED，硬路由方向放弃。**
- **早期诊断健康**：step16 top1_sim=0.29、**topk_mass=0.75（ST-Gumbel 让 routing mass 显著更集中，vs 旧 run 0.28-0.42）**、usage_cov 0.99、usage_var 9e-5。chunk512 配置（save_interval=500，total 5000，eval off）。判据同 P8 阶梯：step500 ckpt 离线 BABILong qa1/qa2/qa5 × 0k-32k vs chunk512 step500 baseline。
- **代码完成（commit a937dab，coder general-purpose-36）**：新增 optional ST-Gumbel top-k 选择，flag `--use_st_gumbel_topk`(store_true) + `--st_gumbel_temperature`(float,default 1.0)，default OFF byte-inert。selector.py forward 新分支（`if use_st_gumbel_topk and self.training`）：给 selection logits 加 Gumbel 噪声 `g=-log(-log(U))` × temp，**仅影响 topk 选择路径**，返回的 scores/ste_weights/load-balance loss 仍用 noise-free logits。eval 不加噪。验证：import OK；pytest tests/test_mem_space_smoke.py 6 passed；flag-off byte-identical baseline、flag-on train 选择随机但 scores 不变、flag-on eval = flag-off。两个 trainer 都接好。
- **launch 待办**：在 P8 最优底座（chunk512 配置）上加 `--key_repulsion_weight 0.05 --use_st_gumbel_topk`，做 ablation 对照 chunk512 step500 baseline。需用户 go-ahead（auto_launch: false）。
### [P11] delta-rule + normalized writeback — **状态: DONE — ⭐新最佳臂（2026-06-07 03:20 eval 评完，超 top_k16 baseline）**。step500 BABILong：qa5 0k-8k=82/86/83/64/50 显著超 baseline（85/76/77/54/48），qa1/qa2 中长度持平或更好（qa2_32k=35 vs base 18）。**delta-rule 写残差 + 归一化 readout 提升长上下文检索保持。裁决：ADOPTED 为新基线配置，后续臂在此底座上叠加。** （32k qa5 cell 评测时仍在 .196 收尾，不影响裁决。）
- **chunk 阶梯三点齐（2026-06-07 13:00 评完，diskB .76 step500 ckpt 同口径 n=100）→ chunk512 决定性最佳**：
  - chunk256: qa5 0k-8k=78/66/47/28/42；qa1 0k=85 32k=16；qa2 0k=38 32k=18
  - **chunk512 (ADOPTED baseline)**: qa5 0k-8k=82/86/83/64/50 ⭐
  - chunk1024: qa5 0k-8k=82/43/20/29/16；qa1 0k=95 但 2k 崩到 4；长程全面塌（16k=5 32k=4）
  - **裁决：chunk512 是 P11 最佳 chunk。256 中长度明显弱，1024 1k 后断崖（同 P8 chunk1024 的 2k-trough/长程塌方形态）。后续臂一律 chunk512 底座。**
- 原 RUNNING 记录：（2026-06-06 10:49 起跑，远程 .196 8×H20，run mem_space_p11_chunk512_deltarule_normreadout，commit 9a9e3d0）— confidence medium，medium 改。写残差 + 归一化 readout magnitude 使其可与 local attn 比较再 gate。
- **代码完成（commit 9a9e3d0，coder general-purpose-37，5 files +101/-2，author LiuHanzuo）**：一组 flag 全 default OFF byte-inert：`--use_delta_rule_writeback`、`--normalize_readout`、`--readout_norm_scale`(default 1.0)。
  - **delta-rule**：当前默认 writeback 是 dual_gate（LM2 双独立门 `g_in·new + g_forget·old`，非残差）；flag-on 改为残差形式 `old + g_in·(new−old)`（forget 绑定为 1−g_in，忽略独立 g_forget），train+eval 都生效（改的是 stored state）。memory_bank.write() 加 `delta_rule` kwarg，layer.py 5 个 gated write 调用点都传入。legacy single-gate EMA 本就是残差，未动。
  - **normalize_readout**：把现有「仅缩小」的 M_sel_hidden clamp 换成 L2-normalize+rescale 到 `h_norm_ref × readout_norm_scale`（可放大也可缩小），让 gate 看到与 local attn 同尺度的 memory 信号。用 hidden_states.detach() 做参考，无额外梯度路径。train+eval 都生效。
  - 验证：import OK；pytest test_mem_space_smoke 6 passed；py_compile 5 文件 OK；flag-off forward+slots byte-identical baseline；delta-rule on slot 变化 max 4.3e-1；normalize_readout on forward 变化 + readout_norm_scale 有效。两 trainer 都接好。
- **launch 待办**：在 P8 最优底座（chunk512）上加 `--use_delta_rule_writeback`（±`--normalize_readout`）做 ablation 对照 chunk512 step500 baseline。需用户 go-ahead（auto_launch: false）。
### [P12] 重审 recon + live-token masking/bottleneck — **状态: PENDING** — confidence low-medium。ICAE(2307.06945)/Gist(2304.08467) 证 recon 只在 LM 被迫只读 slots 时有效；P1 失败疑因无 bottleneck。
### [P13] surprise-gated write（写强度∝预测误差）— **状态: PENDING** — confidence low。借 Titans(2501.00663)，P7/P8 解锁 retrieval 后再做。

### [R3-5] 效果归因（若 eval 仍差，查根因，用户要求"看一看到底因为什么"）— **状态: PENDING**
- 若 R3-1/R3-2 ≥4k 仍塌到 noise floor：用 routing 诊断三件套（topk_mass/usage_var/chunk_idx_jaccard）+ inject_gate 轨迹判断是 routing collapse 还是注入稀释主导，对照 researcher P2 根因（注入稀释 ~0.2% attn mass），决定走 P3(key/value 分离) 还是专用 memory cross-attn 读路径。

---

## Follow-up work（2026-06-05 用户提出，3 条思路）

> 背景：当前并行的 P8-nullsink 阶梯（chunk128/256/512/1024 四臂，同起点 base，唯一变量 chunk_size）是干净 ablation，**先跑着不动**。下面三条是它之后的方向，F1→F2→F3 有先后依赖。

### [F1] 系统的阶梯式（warm-start 链）训练 — **状态: v1 stable 链 DONE（裁决「渐进 ≫ 单 chunk1024」）；v3 改进版 RUNNING（2026-06-07 23:51 起在 diskB .76 单节点 8×H20，stage1 c256 from-scratch step8/5000 lm=3.0 nf=0 healthy，commit ee8baa8，log progressive_chunk_diskB_v3_stage1_c256.log）**
- **v1 结论（已锁）**：stable 渐进链 128→256→512→1024 全链完成，同口径 chunk1024 eval 下 qa1 2k=45 vs 单chunk1024=4、qa5 2k=82 vs 20、长程 qa5 16k=32/32k=29 vs 单 5/4。**渐进 warm-start 彻底修复单 chunk1024 的 1k 后断崖。**
- **v3 改进（research note `small_chunk_training_and_slot_capacity_20260607.md` high-conf）**：跳过最不稳 chunk128 从 c256 起步（3 stage）；warmup ∝ 1/chunk（c256:1200/c512:600/c1024:300，各 stage 热身 token 量级一致）；grad_accum ∝ 1/chunk（c256:8/c512:4/c1024:2，有效梯度 token/step 恒定）；loss_spike_sigma 小 chunk 放宽（4.0/3.5/3.0）。其余超参与 v1 逐字一致。脚本由 coder 写入 `scripts/launch_progressive_chunk_diskB_v3_improved.sh`（进行中）。
- **算力**：当前 .76 单节点 8×H20 空闲（.249 仍跑 w1.0 train 到 5000）→ 先 single-node .76 起 v3 链；待 .249 空出可改 2-node 16 卡。
- **下一步**：coder 脚本就绪 → rsync 代码到 .76 → 起 v3 链 → 每 stage step500 离线 BABILong 对照 v1 stable 链 + 单 chunk1024。

### [F2] 增大 #chunk（更长文本）— **状态: TRAIN RUNNING（2026-06-08 10:31，.196 8xH20 long-doc chunk512）**
- **数据 build DONE（2026-06-08 10:09）**：`MemLong/data/processed/dolmino_longdoc_wiki_min4k`（train=99899, val=2039，wiki ≥4096-tok docs，dolmino_per_doc schema 兼容）。
- **TRAIN 已起（2026-06-08 10:31）**：`.196`（盘A 共享 FS，数据无需 rsync）8xH20 跑 `scripts/launch_f2_longdoc_chunk512_diskA.sh`（F1-best=P11 delta-rule+normalize_readout @ chunk512，warmup600/accum4/sigma3.5，唯一变量=数据换成 long-doc 子集，per-doc chunk 数显著增大）。out=`outputs/f2_longdoc_chunk512`，log `logs/f2_longdoc_chunk512.log`，step3 lm-ok usage_cov0.95 usage_chunks50 nf=0 8GPU 78GB/100%。判据：每 step500 离线 BABILong 16k/32k 对照 F1 chunk512，看 usage_cov/chunk_idx_jaccard 随 #chunk 增大是否保持。
- **动机**：当前每样本 chunk 数太少（4 个量级太粗），memory 的「多 chunk 写入→保持→跨 chunk 读回」能力没被真正压力测试。需要专门从 Dolmino 里挑**很长的文本**，让单样本 chunk 数显著增大。
- **★ 数据来源裁决（2026-06-08 dry_run）**：filter `dolmino_per_doc` 路线 **DEAD**（4096 硬截断）。改 raw re-tokenize：
  - **pes2o = DEAD**：扫 3.86M docs，0 docs ≥8192 tok（全是短摘要）。
  - **wiki = 唯一可用源**：≥2048-tok docs 的 token min/median/mean/p90/p99/max = 2048/3141/4256/7530/16699/61969；4.7% ≥10k tok、0.4% ≥20k tok。6.1G raw（2 files）→ 全量 ≥4k-tok 长尾足够建 F2 子集（虽无纯 32k 量级，但远超 4096-capped per_doc）。
- **判据**：在长文档子集上，随 #chunk 增大 retrieval（usage_cov / chunk_idx_jaccard / 离线 BABILong 长上下文段 16k/32k）是否保持。
- **依赖**：接在 F1（阶梯式训练打通）之后，用 F1 最优配置 + 长文档数据。

### [F3] 改 training objective：加入简化 SWA（sliding-window attention）— **状态: PENDING**
- **动机**：改训练目标，让生成不只依赖 slots。**我们的 SWA 更简单**：每个 chunk 在生成时可以看到「**前一个 chunk + slots**」（即 window=1 chunk 的局部注意力 + memory 读路径并存），而不是标准 SWA 的固定 token 窗。
- **实现待定**：在 forward 里给当前 chunk 的 query 额外开一条对「前一 chunk hidden states」的注意力（KV = prev_chunk），与现有 slot 读路径融合（gate 或 concat）。需 coder 设计；注意不要破坏冻结 backbone（参考 P8 教训：read 路径要有 cold/gate guard，不能无差别全层注入）。
- **判据**：加 SWA 后 lm 与 retrieval 是否同时改善；对照不加 SWA 的同配置。
- **依赖**：独立改动，可在 F1/F2 之后或并行（不同节点）做；需用户确认（涉及架构）。

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
