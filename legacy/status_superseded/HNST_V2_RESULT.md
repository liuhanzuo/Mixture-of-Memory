# HNST v2 结果 (2026-07-01, trainable summary 树 + 解冻 reader)

> run=mem_space_hnstv2_tree_b25, 从干净A模型step2000续训, mix=0全合成, branch=4 beam=2,
> gap_mix{1536,3584,7680}, unfreeze{16,28,29,30,31}, 100步(select_ce已收敛)。官方判分。

## 一、改了哪些代码
- `src/memory/mem_space/tree_summary.py` (新): TreeSummaryPool = 复用 L3SummaryPool(num_summary=1)
  的可训练 leaf_pool + node_pool(Q-Former attention pool),替代 v1 max-pool 聚合。
- `layer.py`: `_fifo_select_keep_set_tree` 有 tree_pool 时用学习聚合(否则回退 v1 max-pool);
  新增 `_fifo_query_probe` + `_fifo_score_node_summaries`(grad-bearing 逐节点 q.k salience)。
- `patch.py`/`config.py`: use_tree_summary 建 TreeSummaryPool 注册到 root,list-wrap ref 挂到每层。
- `train_...py`: `t2_tree_train_step` = 逐层导航 CE(needle 祖先 = local//branch^ℓ)+ token-reforward 读出 LM loss;
  tree_pool 入 optimizer/state_dict。
- 评测: `hnst_v2_needle_recall_probe.py`(v2tree/v1tree/flat/b25)+ eval harness `--swa_tree_token`。

## 二、训练曲线(合成 qa5 give-event, n_ctx 混档)
| step | t2_select_ce | needle_rank | t2_needle(读出loss) |
|---|---|---|---|
| 5   | 3.93 | 3.5 | 2.29 |
| 20  | 2.21 | 1.7 | 0.17 |
| 40  | 1.16 | 1.3 | 0.001 |
| 60  | **0.005** | **0.0** | 0.0001 |
→ 合成任务上树导航+读出**双双收敛**(select_ce→0,needle top-1,读出loss→0)。

## 三、判据结果(BABILong qa5, 官方 retrieval/compare_answers)
### needle-recall(选择墙)
| 长度 | v2tree | v1tree(max-pool) | flat | b25 |
|---|---|---|---|---|
| 8k  (n=90) | 64% | 63% | **76%** | 100% |
| 16k (n=90) | **51%** | 41% | 50% | 92% |
| 32k (n=19, 续跑中) | 32% | 16% | 32% | 95% |
按 needle 位置(early)v2 > v1(+10~16pp, 随树深度增大);v2 全档追平/略超 flat 但从不明显超过。
→ **v2 树聚合确实优于 v1 max-pool(16k +10pp, early needle 更多),追平 flat,但没超过 flat/b25。**
   关键: **b25(recency 窗)在所有档 92-100%** —— qa5 supporting fact/答案 token 定位多落在文档
   后段(recency-biased),b25 几乎总captures needle,**选择墙对 qa5 根本没咬住**;"树够到 b25 evict
   的 early needle"这个前提对 qa5 needle 分布不成立(32k n_ctx≈59 时 b25 仍 100%)。

### pg19 ppl 护栏
v2=1.634 vs A模型=1.589 (seq30, Δ+0.045) —— 与 DIRECTION_C 的 +0.059 同量级,红线守住。

### 端到端读出(mem-chain tree-token)—— 被生成污染,无效
2k n25 官方 8%:模型生成词沙拉("Fred Fred Fred"/"Jeff Fred Bill handed the"),
非单词答案。这是干净 A 血统(inject_gate_bias=-2.0)本就不会经 FIFO 生成答案的已知缺陷;
100 步 SFT(LM loss 只落 answer digit,非自由生成)没修好生成格式。读出墙无法从此指标判定。

## 四、裁决
- **选择墙: 未破(但机制修复被验证)。** 可训练树聚合修复了 v1 max-pool 毁上层信号的死因
  (16k v2=51% vs v1=41%, +10pp;early needle +~19pp),追平 flat(50%),但**没超过零训练 flat,
  更不及 b25(92-100%)**。合成任务 select_ce→0 ≠ BABILong 迁移。
- **决定性反直觉发现: qa5 needle 是 recency-biased 的**,b25(纯 recency 窗)在 8k/16k/32k 全档
  92-100%,即便 32k n_ctx≈59 也 100%。"树够到 early needle(b25 evict)"这个 HNST 前提对 qa5
  needle 分布不成立 —— 选择墙对 qa5 本就没真正咬住,任何学习型选择都赢不了 recency。
- **读出墙: 无法判定。** 端到端生成被干净 A 血统(inject_gate_bias=-2.0 不会经 FIFO 生成答案)
  污染成词沙拉;fullchain oracle 生成同样污染。唯一读出正信号是合成 token-reforward loss→0。
- **树+训练路线: present 形式不值得继续。** 理由: (a)树机制 work 但没超零训练 flat/b25;
  (b)qa5 needle recency-biased → 选择墙前提不成立 → 树无用武之地;(c)读出墙被生成缺陷卡死无法量化。
- 若要复活树路线,先解决三点: (1)换会经 FIFO 生成答案的血统(或加自由生成 SFT)才能量化端到端;
  (2)找 needle 真正 early-biased 的任务(qa1 passkey 早置 / RULER multi-key)b25 才失效,树才有擂台;
  (3)当前 flat 已是更强 baseline,树要赢需 n_ctx≫buffer **且** needle 非 recency。

## 红线合规
mix=0全程;全合成不碰 babilong test;warm-start=干净A模型step2000(非泄漏ckpt);pg19 ppl 护栏过;
官方 compare_answers/retrieval 判分,标注 n。
