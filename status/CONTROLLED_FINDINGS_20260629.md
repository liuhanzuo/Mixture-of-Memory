# 受控实验结论 — Selection × Reforward × Slot (2026-06-29)

> **性质**: 进展快照, 非论文定稿。基于本轮(06-29)受控实验。所有分数 mix=0 干净, 排OOM净分。
> 旧文档 SLOT_SWA_MECHANISM / SLOT_REFORWARD_TWO_ROUTES 的"slot有正贡献"叙事**已被本文档的受控ablation推翻**。

## 实验装置
- 模型: 主用 **A 模型** = `mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt`(经 supervised-select 训练, mix=0); 对照 **distill 模型** = `distill_pg19_chunk512_nctx63`(仅 PG19 hidden 蒸馏, 无选择训练)。
- 任务: BABILong qa5(多跳传递推理), 8k/16k/32k。max_new_tokens=20, chunk512, topk=4(除 sweep)。
- token-reforward: 选中 chunk 的**原始 token** + 问题拼接重 forward 全模型(query 在场)。三种 chunk 来源: oracle(定位 needle)/ bm25(词重叠)/ reader-attn(layer16 q·k salience)。

## 四条结论

### 结论1: slot 记忆净贡献为负(三档全确认, n100)
同一 distill 模型, slot ON(slotSWA) vs slot OFF(--memory_disabled 纯SWA), 同滑窗 W 同批:

| W | 8k 纯/slot(净) | 16k 纯/slot(净) | 32k 纯/slot(净, B200满100) |
|---|---|---|---|
| 0 | 19/14 (−5) | 14/12 (−2) | 8/7 (−1) |
| 2 | 54/49 (−5) | 40/34 (−6) | 21/17 (−4) |
| 4 | 74/63 (−11) | 55/42 (−13) | 45/27 (−18) |
| 6 | 85/72 (−13) | 64/50 (−14) | 52/36 (−16) |

→ slot 一致拖累, 差距随 W 单调拉大。**slot+SWA 的能力来自滑窗读出窗口, 不是 slot 记忆。** slot 的有损注入(5000×压缩)是噪声。

### 结论2: bm25 选择 > reader-attn 选择(选得准)
A 模型 16k, topk=1 控制信息量(都只给1 chunk):
- **bm25 tk1 = 43-51% vs reader-attn tk1 = 3-6%**
- topk=4: bm25=74 vs reader-attn=46
→ BM25(免费词重叠)选 needle chunk 的能力 >> 学习的 reader-attn salience。reader-attn 选择质量是当前短板。

### 结论3: "bm25 > oracle" 是 oracle 定义缺陷, 非真上界
A 模型 16k bm25 topk4=74 > oracle=59; 但 8k bm25 < oracle(正常)。机制(读代码坐实):
- `_locate_needle_chunks`(run_babilong_mem_space.py:L413) 只定位含 target **字面**的 chunk。
- qa5 是多跳推理, 支持事实散布在多个 chunk; oracle 只取含答案字面的 chunk, **漏了推理链中间事实**。
- 长档(16k/32k)推理链更分散, oracle 漏得更多 → oracle 偏低; bm25 top4 多给 chunk 反而间接覆盖推理链 → 显得超 oracle。
→ **当前 oracle 不是真读出上界**。需 full-chain oracle(含全部 supporting facts)才是正确上界。"bm25>oracle" 不能解读为模型优异。

### 结论4: 选择+reforward 路线强依赖训练(核心)
distill 模型 16k 满100: **bm25=12 / reader-attn=13 / oracle=49** — 非 oracle 选择全崩≈随机。
A 模型(supervised-select)同设置: bm25/reader-attn 都能 work。
机制(读代码坐实):
- A 模型训练含 `t2_select_loss`(监督 layer16 把 needle 排 top) + `t2_recall_mix=0.5` + 训练时就走 "select topk → cat raw token → freeze banks → forward → CE on answer" 路径。
- backbone **in-distribution** 学会"给我任意 k 个选中 chunk 的 raw token + 问题, 我来 decode"。
- distill 只蒸 PG19 hidden, 从没见过"少量 selected chunk 前缀"分布 → OOD → 全崩。
→ **选择性 reforward 不是即插即用, 必须有 in-distribution 的 select+reforward 训练。**

## 对项目叙事的修正
| 旧叙事 | 修正 |
|---|---|
| slot 压缩长程记忆有贡献 | slot 当前净负, 移出主线; 主线 = 纯SWA + 选择性 token-reforward |
| reader-attn 可部署选择器 | BM25 更强 baseline, reader-attn 待改进 |
| oracle 验证读出上界 | 当前 oracle 定义不完整(qa5多跳), 需 full-chain oracle |
| 选择性 reforward 通用机制 | 需 supervised-select 训练, distill 完全失效 |

## 待办/后续方向
1. **full-chain oracle**: 从原始 babi 对齐全部 supporting facts 位置(babilong cache 不直接暴露, 工程量中等)。这是辨清"选择质量 vs 读出质量"真实分解的关键。
2. FAIR100 满100 收尾(8k A三方 / distill 8k+32k)巩固上表。
3. reader-attn 选择器改进(它 << BM25, 是学习选择路线的瓶颈)。

## 红线遵守
mix=0; **泄漏ckpt(b50/b100/P2/c1024/旧b25/P11/l3recontoken)分数一律不引用**(本轮 background agent 曾误引 b50 oracle 70.5, 已丢弃); 真SOTA锚点 pg19 nctx7 qa5 16k=16/32k=9。
