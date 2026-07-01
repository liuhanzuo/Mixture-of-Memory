# 新方向调研：training objective（语义摘要 vs token级hidden）& 加强 L3 能否补 LongEval

> 调研者：general-purpose-7（2026-06-07）。纯调研，不改代码/不起训练/不占 GPU。
> 目标：为 .76/.249（盘B，各 8 卡）上的新方向实验定方案。
> 所有结论附 file:line 代码证据或已有 run 数字。

---

## 0. 关键前置结论（最重要，先说）

**项目里没有任何 LongEval（LMSYS LongEval / topic-retrieval / lines 那套）评测脚本。** grep `longeval/LongEval/long_eval` 全仓只命中 `RESEARCH_ROADMAP.md`（D5 假说文字本身）、`launch_experiment_h3_memlong_eval.sh`（MemLong NIAH，不是 LongEval）、以及 babilong 包。**这是 Q2 的前置 blocker：要真正验证"全局总结/聚合"短板，要么接 LongEval 数据，要么用现有最接近的 `scripts/eval_longbench_mem_space.py`（6 个 QA：hotpotqa/narrativeqa/qasper/multifieldqa_en/2wikimqa/musique，含 multi-hop 聚合）当代理。**

**而 LongBench 代理其实已经给过决定性信号**：R3-3（ROADMAP / MEMORY_PROTOCOL_PLAN.md:131-135）实测 **adapter LongBench avg F1=2.94 vs base Llama-3-8B 13.95（~5× 退化）**，而同期 adapter 在 BABILong(NIAH) qa5 反而强于 base。**这是 D5「slot 记 token 级 hidden、擅 NIAH 检索、毁全局聚合」假说目前最硬的证据**，已经成立到 medium-high confidence，不需再"系统验证一遍"才动手——可直接进改 objective。

---

## Q1：有没有更好的 objective 让 slot 学"语义摘要"而非"token 级 hidden"？

### 诊断（为什么当前 LM-loss + routing-aux 倾向 token 级）
- 当前 aux 只有 routing 类：`_collect_aux_loss` 累加 `load_balance/entropy/key_repulsion/weight_ortho/l3_diversity/q_multi_diversity/recon`（train_mem_space_dolmino_cpt.py:634）。除 `recon` 外**全是 routing 几何正则，没有任何"内容语义"目标**。
- slot value 唯一的内容信号是**间接 LM loss**——MEMORY_PROTOCOL_PLAN.md 的 toy 实验（commit e5bb181）已证：寻址可修好（top1_sim 0.32）但 `retrieval_exact_acc=0`，即"写进去读不回"。LM loss 太绕，writer 没有近距离目标教它存"可读回的语义"，于是退化成存表层 token 级 hidden（最省力地拟合 teacher-forcing 续写）。

### 候选 objective（按可落地性排序）

**① recon / autoencoding aux —— ⚠️ 已试过且 REJECTED，不要重跑。** confidence: **high（负结论）**。
- 代码已存在：`--l_recon_weight`（默认 **0.0**，train:791）+ `MemoryReconDecoder`（recon_decoder.py，1-block cross-attn，从 slot value `M_write` 重建 L3 summary，`MSE(S_hat, stopgrad(S_L3))`，layer.py:1625-1628）。
- **但 P1（MEMORY_PROTOCOL_PLAN.md:40-48）已在真 Dolmino 上跑过 l_recon=0.1 vs 0：overall norecon 8.74% > recon 6.89%，recon 反而更差。REJECTED。** → 单纯调 `l_recon_weight` 这条便宜路已被堵死。P12 怀疑 recon 失败因"无 bottleneck（LM 不被迫只读 slots）"，但那要写新代码。

**② L3 summary 直接监督（next-chunk 摘要预测 / 蒸馏）—— 最值得做的新 objective。** confidence: **medium**，需写新 loss（中等改动）。
- 机理：L3 是项目里唯一显式"摘要"部件（l3_summary.py，Q-Former pool 把 chunk → K 个 summary token），但它**目前只靠 LM loss 间接学**，唯一直接信号是 `l3_diversity`（防 query 塌缩，l3_summary.py:157）——没有任何东西逼 L3 输出"语义对、可聚合"。
- 候选：用 chunk_n 的 L3 summary **预测 chunk_{n+1} 的 L3 summary**（或预测下一 chunk 的 bag-of-content / 关键实体），逼 L3 学全局语义流而非局部 token。改动量：新 loss + 在 train loop 缓存 prev/next summary（L3 已有 `_prev_summary` 递归状态，l3_summary.py:127/145），中等。

**③ 对比学习 aux（同文档 chunk 拉近、跨文档推远）—— 便宜且独立。** confidence: **medium**，小-中改动。
- 机理：逼 slot/L3 表示编码"这是哪篇文档的语义"而非表层 token，是经典 把表示从 surface 推向 semantic 的手段。可复用 `key_repulsion_loss`/`query_diversity_loss` 的成对 cosine 框架（l3_summary.py:157-194 已有同款实现），加一个 batch 内 same-doc/diff-doc 标签即可。改动量小，且与 routing 解耦、可单变量归因。

**便宜（调 flag）vs 贵（写代码）划分：**
- 便宜但已死路：①recon（REJECTED）。
- 便宜可立刻试：调 `--l3_diversity_weight`（默认 0.1）/ `--l3_n_summary` —— 但这只调"摘要多样性/容量"，不改"语义 vs token"本质，预期增益有限。
- 真正能改本质的（②③）**都要写新 loss**，属"贵"，需 coder。

---

## Q2：加强 L3 能否提升 LongEval（全局聚合）能力？

### L3 现状（grep 确认）
- 参数（train:820-825）：`--use_l3_summary`（**默认 ON**）、`--l3_n_summary 64`、`--l3_n_layers 2`、`--l3_n_heads 8`、`--shared_memory_bank`（默认 True）、`--l3_diversity_weight 0.1` / `--l3_diversity_threshold 0.5`。
- L3 输出**确实进 LM forward**：extended 序列布局 `[L3(k_l3) | L1(k) | H(T)]`，H token 可 attend L3 keys（layer.py:113-123）。所以 L3 是当前唯一"全局摘要 prefix KV"。L3 还作为 multi-query 子查询喂 selector（layer.py:848）。
- **L3 是项目里最接近"全局摘要"的部件，但从未被专门 scale-up 或专门监督测过其对聚合任务的贡献**（RUN_REGISTRY/ROADMAP 全是围绕 L1 slot + chunk_size + 读路径的 ablation）。

### 最小对照实验设计（8 卡单节点可跑）
- **baseline**：现有最佳底座 = **P11 delta-rule + normalize_readout，chunk512**（ROADMAP/RUN_REGISTRY §3.4 裁决新最佳臂），L3 默认 64/2layers。
- **arm L3-scaled**：verbatim 同配置，仅 `--l3_n_summary 64→128`（或 256）+ `--l3_n_layers 2→4`（可拆两臂单变量）。
- ⚠️ **改 L3 容量会改 ckpt 形状**：L3 queries=`[num_summary, d_model]`、blocks 随 n_layers（l3_summary.py:104/107），warm-start 从现有 adapter 只能 strict=False 部分加载。**故 L3-scaled 必须从头训（base Llama-3-8B 起，不 warm-start），即可绕开形状约束**——L1 slot 部分不受影响（L3 与 L1 bank 解耦）。
- **eval**：跑 `eval_longbench_mem_space.py`（6 QA 聚合任务，对比 base F1=13.95 锚点）+ BABILong qa1/qa2/qa5 对照（确认 NIAH 不退化）。判据：LongBench F1 是否从 ~3 往 base 13.95 方向抬。
- confidence：**low-medium**。L3 已在 forward 里、加大容量是合理 bet，但"间接 LM loss 学不出语义"的诊断意味着**光加容量、不加 L3 专门 objective，可能只是更多没学好的 summary token**。建议 Q2 与 Q1-② 绑定：scale L3 **同时**给 L3 加直接监督，比单纯 scale 更可能见效。

---

## 推荐：第一个该在 .76/.249 上跑的实验
见发回 main 的 350 字结论。
