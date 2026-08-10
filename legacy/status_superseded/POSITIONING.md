# 项目定位 — 文献坐标与唯一可立卖点 (2026-07-01)

> opus联网查文献(走代理star-proxy:3128, 直读arxiv abstract核验)给出的残酷坐标。用于后续所有决策对齐。

## 系统三段拆解 vs 文献
我们的系统 = (A) per-layer 128 hidden memory slots + FIFO存近期chunk raw hidden → (B) selector在hidden空间top-k路由 → (C) 只对选中赢家做"原始token reforward读出"。

| 段 | 新? | 谁占了(arxiv) |
|---|---|---|
| (A) hidden memory slots/pool | **红海** | MemoryLLM 2402.04624 / M+ 2502.00592 / RMT 2207.06881 / ARMT 2407.04841(BABILong SOTA) / Activation Beacon 2401.03462 |
| (B) hidden空间top-k路由替代外部检索器 | **红海** | Landmark Attention 2305.16300 / InfLLM 2402.04617 / Quest 2406.10774 / Focused Transformer 2307.03170 / M+ retriever |
| (C) 选中赢家reforward原始token精确读出 | **基本空白** | 几乎没人正面做 |

## 最接近的3篇(警惕)
- **Landmark Attention 2305.16300**: 用注意力本身选block=我们(B)的思想, 2023已做。差别仅: 它载入缓存KV, 不reforward。
- **M+/MemoryLLM 2502.00592**: latent memory pool + co-trained retriever = 我们(A)+(B)成品版, 验证到160k。
- **LongMem 2306.07174**: 冻结骨干+学习reader去hidden检索 = 我们的范式。

## 两个扎心结论
1. **"省检索计算"是死海, 且我们在此轴结构性劣势**: 主流(InfLLM/Quest/RetrievalAttention/CacheBlend 2405.16444)选中block后**复用缓存KV零重算**=真省。我们reforward=把选中chunk原文重过一遍≈重新prefill=成本项非节省项。隔离实验73→40正是这代价:既没省钱读出还垮。**"降低RAG检索成本"当一级卖点会被数据打脸。**
2. **selector输BM25不是偶然**: Focused Transformer/RetrievalAttention都指hidden/key空间检索有OOD/distraction硬伤难学。撞的是公认的墙。

## 唯一可立的卖点: fidelity-routing 解耦
放弃"省计算", 把唯一空白(C)包装成:
> **廉价hidden路由负责"找", 无损reforward负责"读" —— 把recall的便宜和precision的保真解耦。** 别人为省都用有损KV/压缩hidden去读, 我们用reforward换"不丢信息的读出"。

对应用户原话"RAG要读入整篇, 我们希望检索被减少/在hidden level做": hidden路由只挑少数赢家(省掉读全部候选), 但对赢家用无损reforward(不像别人用有损压缩hidden读)。

## ★ 充要条件(方向c的真意义)
此卖点成立 **当且仅当 reforward读出能恢复到接近全文注意力**。现状 **40 < 73(qa5 4k)**, 窟窿在此。
- **方向c MVP正在赌这个**: 把mem-chain读出从40拉回73+。
- 拉回→fidelity-routing解耦有第一块基石, 贡献从"诊断"升级为"方法"。
- 拉不回→我们手里是高质量负结果 + "记忆机制不如直接RAG"的结论。

## 死海轴(别立卖点)
- "hidden state当memory" / "hidden检索替代外部RAG检索器" / "frozen backbone+learned retriever": 任一当主claim会被"how diff from MemoryLLM/M+/Landmark/InfLLM?"打死。
- "学hidden selector检索"当精度卖点: 已输BM25, OOD是公认硬伤。

## 关键arxiv清单
2203.08913 Memorizing / 2305.16300 Landmark / 2307.03170 Focused(LongLLaMA) / 2305.01625 Unlimiformer / 2402.04617 InfLLM / 2406.10774 Quest / 2306.14048 H2O / 2409.10516 RetrievalAttention / 2401.03462 ActivationBeacon / 2402.04624 MemoryLLM / 2502.00592 M+ / 2306.07174 LongMem / 2407.09450 EM-LLM / 2407.04841 ARMT / 2207.06881 RMT / 2405.16444 CacheBlend / 2412.15605 CAG
