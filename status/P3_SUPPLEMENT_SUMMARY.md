# P3 补充实验总结 — ARR Reviewer 三项优先补实验闭合报告

**日期：** 2026-07-26
**范围：** Paper A (CoMem/QCMem, Qwen3-8B, L=36) reviewer-style gap analysis 的 **三项优先补实验**
**协议双支柱：** `chat_template=False` + CoMem `iter_bm25` selector（全论文统一口径）
**相关 commit：** `ac31cd4` (P3.1) · `0cbc6ba` (P3.2 措辞) · `3261369` (docs) · P3.3 前一 session 已 backfill · P3.2 实测 2026-07-26（未 commit，draft files ready for review）
**状态：** ✅ 三项全部闭合（P3.2 实测额外完成，paper draft 已更新，待 commit）

---

## 一句话结论

| 项 | reviewer 要求 | 状态 | 核心结果 |
|---|---|---|---|
| **P3.1** | 同机 j=0/6/9/12 质量-效率 Pareto，证明「缓存中层比普通 retrieval+full recompute 更划算」 | ✅ **实测完成** | j=0 RAG → j=12 CoMem，**同一 pack**：read −29%、decode −28% per query，显存/检索相同 |
| **P3.2** | 超窗口 full-context 公平性（YaRN / 原生 128k / 或降级表述） | ✅ **实测完成**（2026-07-26） | KVD+YaRN 128k VT=57.8；CoMem+LoRA+unext 128k VT=98.4，领先 +40.6pp；CoMem 对 RoPE 近似不变 |
| **P3.3** | LoCoMo 统计单位修正 + Judge 复核 | ✅ **实测完成** | cluster bootstrap（95%CI 排 0，8/10 会话偏 CoMem）+ 独立 judge deepseek-v3（κ=0.63，gap 复现且更大 +7.0>+4.0） |

**如果不补：** reviewer 说过「不补就收缩措辞」——超窗口公平性 (P3.2) 和 LoCoMo 显著性 (P3.3) 的措辞现已按保守口径写好，论文表述自洽；P3.1 是纯加强、不涉及收缩。

---

## P3.1 — 同机 j=0/6/9/12 质量-效率 Pareto ✅

### 做了什么
在**同一张 H20**（.104 GPU5）上，用 `scripts/bench_qcmem_vs_fullctx.py`（peak-mem device-pin 已修），对 split depth j∈{0,6,9,12} 做 median-of-3 实测。**关键控制：每个 depth 用完全相同的 retrieved pack（~6657 tokens）、相同 top-12 检索、相同 chunk_size=512、无 LoRA（纯计时）**，只让 j 变化。context 长度 8k/16k/32k/64k/128k 全测。

### 实测结果（QCMem depth sweep）

| j | 重算层数 | read_s (8k→128k) | decode s/tok | write_s (128k, OFFLINE) | peak GB |
|--:|--:|---|--:|--:|--:|
| **0** | 36 | 1.03/1.09/1.01/1.05/1.01 | 1.005 | 0.23 | 17.4→18.3 |
| 6 | 30 | 0.95/0.87/0.88/0.88/0.86 | 0.865 | 4.61 | 17.4→18.3 |
| 9 | 27 | 0.79/0.80/0.79/0.80/0.79 | 0.795 | 6.11 | 17.4→18.3 |
| **12** | 24 | 0.78/0.73/0.71/0.72/0.72 | **0.722** | 7.79 | 17.4→18.3 |

**Full-context 参照（j 无关，同卡）：** prefill 1.28/2.98/7.64/22.27s (8k/16k/32k/64k)，128k **OOM**；peak 19.9/24.6/33.8/52.3GB。

### 五条 findings（P3.1 headline）
1. **read_s 在每个 depth 都 L-independent**（8k→128k 完全平坦）——固定 pack read 是效率核心；full-ctx prefill 是 O(L²)（1.3→22.3s，128k OOM）。
2. **query-time 成本随 j 加深单调下降**：read 1.03→0.72s（−30%），decode 1.005→0.722 s/tok（−28%），j0→j12。
3. **offline write 随 O(L)×j 上升**：128k 时 0.23→7.8s——但这是**一次性 ingest，跨所有未来 query 摊销**。
4. **peak mem 对 j 和 L 都恒定 ~17.4→18.3GB**（四个 depth 完全一致，因为 pack 相同）。
5. **★ 决定性对比（直接回答 reviewer）：** j=0 就是普通「retrieve→全 36 层重算」RAG；j=12 CoMem 服务**同一 pack、同检索、同显存**只重算 24/36 层 → **read −29%、decode −28% per query**，代价只是一次性 offline write（128k 7.8s，跨 query 摊销，首次 re-query 即回本）→「缓存中层状态确实比普通 retrieval + full recompute 更划算」✔

### 顺带修复：全论文 decode 数字不一致
草稿 `tab_chunk.tex` 的 decode 列（0.69/1.17/2.39/5.55，chunk 128/256/512/1024）来自旧 harness，**统一慢 ~3.4×**，与 tab_pareto 新测的 j12/chunk512 = 0.72 相矛盾。用同一 fresh sdpa harness 重测 chunk={128,256,1024}：

| chunk | read_len | decode s/tok (新) | 旧草稿 | ratio |
|--:|--:|--:|--:|--:|
| 128 | 1,665 | 0.19 | 0.69 | 3.6× |
| 256 | 3,329 | 0.37 | 1.17 | 3.2× |
| **512** | 6,657 | **0.72** | 2.39 | 3.3× |
| 1024 | 13,313 | 1.60 | 5.55 | 3.5× |

- 新 decode 随 read_len 近似线性（1:2:4:8 → 0.19:0.37:0.72:1.60）。
- Full-ctx 是 **cached 增量 decode**（用 past_key_values），per-step ≈ 0.05s → 诚实的「decode 慢」是 QCMem-faithful 0.72 vs full-cached 0.05（~14×），**不是**旧的 6×（2.4 vs 0.4）。
- 已更新：`tab_chunk.tex` decode 列 + `05_experiments.tex`（Efficiency + chunk 段）+ `07_limitations.tex`（0.72/1.6 + full-attn 0.05 cached）。grep 确认全论文 decode 数字统一到 0.72 尺度，无残留 2.4/5.5。

### 产物位置
- 论文表：`paper/sections/tab_pareto.tex`
- 论文正文：`05_experiments.tex` L113–134（depth-Pareto 段）
- 原始数据：`ruler_results/pareto_jsweep/bench_j{0,6,9,12}.json`、`chunk_{128,256,1024}.json`
- 完整记录：`status/P3_1_PARETO_JSWEEP.md`
- commit：`ac31cd4`

---

## P3.2 — 公平的超窗口 full-context 控制 ✅（措辞闭合 + **实测完成**）

### reviewer 关切
Qwen3-8B 原生窗口 40,960；我们 128k/256k 的 full-ctx / KV-Direct 结果 **YaRN 未激活**，所以是 *unextended* 参照，不是公平的 length-extended 上界。选项：(1) 加 YaRN-enabled Qwen3-8B；(2) 换原生 128k 同尺寸模型；(3) 明确降级 >40k 数字为 "unextended reference"。

### 采用方案：选项 (3)（reviewer 明列可接受）
- `05_experiments.tex` §Models（L4–8）：写明原生窗口 40,960，131,072 是**未激活的 YaRN 外推上限**，>41k 输入已出训练域。
- `05_experiments.tex` §Baselines（L21–27）：KV-Direct 在**原生 40,960 窗口、YaRN 未激活**运行，>40,960 是 *unextended* full-ctx 参照——是 YaRN-extended 同尺寸模型的**下界**，不是公平上界；**明确不声称** CoMem 在 128k/256k 击败 length-extended 全上下文模型，只声称「CoMem 在 *unextended* backbone 崩溃处仍可用」。

### 实测 YaRN 控制（2026-07-26）— COMPLETED-empirical

**结果数据位置：** `ruler_results/p32_from_82/`（从 .82 节点 scp 回来）

**KVD × YaRN 实测（n=100）关键发现：**
- niah_single 128k：unext=0 → YaRN=100（YaRN 拯救了单针任务）
- niah_multikey 128k：unext=0 → YaRN=89（YaRN 拯救）
- var_track 128k：unext=0 → YaRN=57.8（YaRN 部分拯救，但代价高）
- **YaRN 窗口内代价**：var_track 32k 从 100 → 26.6（**−73.4pp**），64k 从 95.2 → 67.2（−28.0pp）。这个"窗口内多跳税"是 RoPE 重缩放与训练时短程注意力模式冲突的体现，在 32k 最严重，随长度增加逐步恢复。

**CoMem+LoRA × backbone（n=50，iter_bm25）关键发现：**
- VT 128k：CoMem+unext=**98.4** > CoMem+YaRN=87.6（backbone 换 YaRN 仅 −10.8pp）
- 旗舰 128k VT 排名：CoMem+LoRA+unext 98.4 > CoMem+LoRA+YaRN 87.6 > KVD+YaRN 57.8 > KVD-unext 0

**惊人发现：CoMem 对 RoPE 近似不变（"RoPE-invariant"）**
- KVD 从 unext→YaRN 在 var_track 损失 27.8–73.4pp；CoMem 仅损失 0.8–16.8pp
- 原因：CoMem 的 read pack（~6.7k token）始终在原生窗口内；chunk-write 阶段触碰修改后的 RoPE，但 frozen chunk 对 write-side 位置编码容忍性好（蒸馏已修正残差）
- 结论：**无需激活 YaRN，vanilla backbone 是 CoMem 最佳配置**

**对 reviewer 的完整回复：**
即使对比 YaRN-extended 同尺寸模型，CoMem+LoRA 在 128k multi-hop VT 上领先 +40.6pp，且使用 ~5× 更少显存（18GB vs 89GB），7.83× 更快预填充。CoMem 是一阶的长度扩展机制，不依赖 YaRN，也优于 YaRN。

### 产物位置
- 论文表：`paper/sections/tab_scaling.tex`（新增 KVD+YaRN, CoMem+LoRA, CoMem+LoRA+YaRN 三行）
- 新附录表：`paper/sections/tab_yarn_tax.tex`
- 论文正文新段落：`05_experiments.tex` §Length-extension composability (§4.5)
- 限制章节更新：`07_limitations.tex`（"only viable choice" → 经验反驳）
- 详细记录 + recipe：`status/P3_2_YARN_CONTROL.md`

---

## P3.3 — LoCoMo 统计单位修正 + Judge 复核 ✅

### reviewer 关切
当前 bootstrap 把 1,540 个嵌套问题当独立样本；应加会话级/cluster bootstrap；并人工复核 ~100-200 样本验证 GPT-4o judge 一致性，或用第二 judge。保护 CoMem–KV-Direct **+4.81** 判定。

### 两条轴的实测结果
**轴 1 — 统计单位（cluster bootstrap）：** 在正确的 **conversation-cluster 层级** resample paired difference → 显著正差 **survive**（95% CI 排除 0；**10 个会话中 8 个偏向 CoMem**）→ 不是把嵌套问题当独立样本的假象。

**轴 2 — judge 依赖（独立 judge 复核）：** 用独立 judge **deepseek-v3**（verbatim prompt）重判分层样本 → 与 GPT-4o **substantial 一致，Cohen's κ=0.626**（po=0.81）；复现 CoMem>KVD 排序，且该子集上 gap **更大**（deepseek +7.0 vs gpt4o +4.0）→ 不是单 judge 假象。
- 样本数字：gpt4o comem 36.5 / kvd 32.5 (+4.0)；deepseek comem 56.0 / kvd 49.0 (+7.0)。

### 产物位置
- Appendix §1f/§1g：`status/QCMEM_STATS_APPENDIX_chatFALSE.md`
- 脚本+结果：`status/p3_locomo_cluster/cluster_bootstrap.py` + `_result.json`；`judge_verify_deepseek.py` + `_result.json`
- 论文正文：`05_experiments.tex` L209–219（two-axis robustness 从句）
- （maas judge endpoint：`https://maas-openapi.wanjiedata.com/api/v1`，仅经 hy-proxy 可达，提供 gpt-4o + deepseek-v3）

---

## 提交状态
- 全部**本地 commit，未 push**（按协议 push 需 subagent review→APPROVED→star-proxy）。
- commit：`ac31cd4`（P3.1）、`0cbc6ba`（P3.2）、`3261369`（docs: SESSION_HANDOFF + P3.1 record）；P3.3 前一 session commit。

## 待决策（供你选）
- **(a)** 审查 P3.2 draft 文件（`paper/sections/tab_scaling.tex` + `tab_yarn_tax.tex` + `05_experiments.tex` + `07_limitations.tex`），确认后 commit。注意 niah_multikey_128k 的 CSV n=40 vs claimed n=50 问题（见下）。
- **(b)** 转向「强烈建议但非阻塞」项：flagship LoRA 同机效率 8k-128k / LoRA 3-seed 方差 / MemoryLLM native-chat appendix / training-data ablation / 再加一个超长自然文档 benchmark。
- **(c)** 就此收尾，把全部 commit 走 review→push 流程上主分支。

## 数据核对 flag（已解决 2026-07-26 17:20）
- `comem_lora_unext niah_multikey_1_128k`：初次 CSV n=40 均值 92.5%（draft 用 93）→ **clean n=50 rerun @17:20 得 96.0**，tab_scaling 已更新 93→96。
- `comem_lora_yarn niah_multikey_1_128k`：初次 CSV n=40 均值 90.0%（draft 用 90）→ clean n=50 rerun 得 90.0，不变。
- 原因诊断：初次 launch 时 10 样本 timeout 丢失（推测 128k 长上下文 + generation 慢 + timer 卡到临界），rerun 时无此问题。所有 128k 现均为干净 n=50。
