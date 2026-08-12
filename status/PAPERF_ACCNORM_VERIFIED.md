# acc_norm 口径重算：核心结论成立，但交付物有两处数字错误（一处 100×、一处 56×）

**日期**: 2026-08-08 ~14:1x CST。**Agent**: `a6f4896a`（task #201）判决 SUPPORTED。
**MAIN 独立复核**：核心统计量**全部复现**，但发现交付物有两个数量级错误，其中一个使它的
batch-size 段落完全无效。

## ✅ 复现成立的部分（我逐 rung 重算，与 agent 完全一致）

用 `norm_scores`（= `option_scores[k] / max(norm_lens[k],1)`，即 **acc_norm 决策变量**）
算 top1−top2 margin，6 个 rung 各 n=17,195：

| rung | core6 | median margin | frac<0.005 | frac<0.010 |
|---|---:|---:|---:|---:|
| base full32 | .70365 | **0.124594** | 2.012% | 4.245% |
| ShortGPT-16 | .62247 | 0.103760 | 3.286% | 6.432% |
| keep14 | .59532 | 0.093614 | 3.280% | 6.723% |
| keep12 | .56888 | 0.084801 | 3.769% | 7.380% |
| keep10 | .52999 | 0.077903 | 4.234% | 8.369% |
| keep8 | .52328 | **0.075801** | 4.461% | 8.607% |

- `Spearman(core6, median margin)` = **+1.0000**, exact two-sided **p = 0.0028**
- `Spearman(core6, frac<0.005)` = **−0.9429**, exact **p = 0.0167**
- `Spearman(core6, frac<0.010)` = **−1.0000**, exact **p = 0.0028**

**这是本方向第一次在正确 metric 上得到的结果，且比旧 raw-logprob 口径更强**
（旧口径的 frac<0.1 在 pruned-only 上 p=0.083 不显著）。median margin 从 base 的
0.1246 单调降到 keep8 的 0.0758，**6 点完美单调**。

> ⚠️ 但注意：这是**同盘同架构的 within-rung 代理量**（near-tie 密度），
> **不是跨协议 flip 的直接测量**。agent 自己也如实标注了这点。

## ❌ 交付物错误 1：`frac<0.005` 整列小数点错位 100×

交付物表里写 base `0.020%` / SG16 `0.033%` / keep8 `0.045%`，
**实测是 `2.012%` / `3.286%` / `4.461%`**。

agent 自己在 batch-size 段落写的 "near-tie items (margin<0.005): 3.3%" 才是对的
—— 同一份数据在同一份报告里出现了两个相差 100 倍的值。

**好消息**：Spearman 与 p 值算的是**正确的底层数据**（我重算完全一致），
所以这只是**表格渲染错误**，不影响结论。但写论文前必须修，否则 reviewer 一算就崩。

## ❌ 交付物错误 2：batch-size 段的 34.7% 翻转率是假的（真值 0.62%，56× 夸大）

交付物写：
> "Batch-size perturbation (ShortGPT-16, bs8 vs bs16, acc_norm):
> **5,969 / 17,195 items flip (34.7%)**"

**实测**：
```
correct flag 翻转 = 107  (0.622%)
pred_letter 翻转  = 122  (0.710%)
```

**根因已定位**：`7B_shortgpt16_step200000_bs8` 目录**没有 `norm_scores` 字段**
（enrichment 只跑了 `_bs16` 一侧），实测：
```
7B_shortgpt16_step200000_bs8:  hellaswag rows=10042  有norm_scores=False
7B_shortgpt16_step200000_bs16: hellaswag rows=10042  有norm_scores=True
```
所以 agent 在算 acc_norm flip 时**一侧有字段一侧没有**，比较退化，产出 5,969 这个
虚假数字。**34.7% 这个量级本身就该触发怀疑** —— 如果 batch size 真能翻掉三分之一的
item，core6 不可能只动 +0.0778 pp。

**连带失效**：交付物基于该数字的整段分析，包括
`P(flip|margin<0.001)=66%` 与 `P(flip|margin>1.0)=4.7%` 的 7-bucket 单调曲线
—— 那条曲线是用污染的 flip 集合拟合的，**必须在两侧都 enrich 后重算**。

## 当前方向 A 的真实状态

| 组件 | 状态 |
|---|---|
| **near-tie 密度随 damage 单调**（acc_norm 口径） | ✅ **成立**，ρ=+1.00 p=0.0028（median margin）；ρ=−0.94 p=0.0167（frac<0.005） |
| **margin → P(flip) 曲线** | ❌ **待重算**（bs8 侧缺 `norm_scores`，现有曲线基于污染数据） |
| **flip rate 随 damage 单调**（主终点） | ❌ **仍未在 acc_norm 口径上测过**（跨架构对需要 wzc1 侧，`.252` refused + LOCAL 在训练） |
| **LOO 中介检验** | ❌ 未做（只有 1/6 rung 有 bs16 数据） |
| **确定性分母 0 flip** | ✅ 成立（旧口径已验，acc_norm 口径同理，因为同代码同数据） |

所以：**"受损模型 near-tie 更密" 现在是**（在正确 metric 上、n=6、p=0.0028）**扎实的**；
但 **"所以它们的 benchmark 数字更容易被实现变量翻转" 这个因果链的后半段仍未验证**。
前者是模型内在性质，后者才是论文的卖点。**两者之间的桥（margin→flip 曲线）恰好是被
污染数据毁掉的那一块。**

## 修法（便宜，纯 CPU + 少量 GPU）

1. **enrich `_bs8` 侧**（纯 CPU，`scripts/enrich_per_example_normscores.py` 已存在）
   → 重算 SG16 的 bs8-vs-bs16 acc_norm flip 与 margin→P(flip) 曲线
2. **补其余 5 个 rung 的 bs16**（6 rung × ~15 min ≈ 1.5 GPU-h）
   → 才能做 LOO 中介检验（需 ≥3 rung）与"flip rate 随 damage"主终点
3. 跨架构那条等 `.252` 复活，或 LOCAL 训练结束后补

## Agent 表现评价（供以后派单参考）

- **该赞的**：harness 改动是**真的纯增量**（我确认 6 个 rung 的 core6 全部与 clean 值
  逐位一致，summary 未变）；14,090 行的 `norm_scores` argmax 与存储的
  `acc_norm_score` **0 mismatch**；发现 item_id=1 那个 `acc==correct` 但
  `acc_norm_score=0.0` 的案例并主动指出，这正是 acc/acc_norm 分歧的实证。
  还额外写了 `enrich_per_example_normscores.py` 让存量数据免重跑（省了 ~11×20min GPU）。
- **该批的**：两个数量级错误都属于"**自己报告内部就矛盾**"（0.033% vs 3.3% 同文出现；
  34.7% 与 +0.0778pp core6 变化在物理上不相容），本该自查出来。
  以及**没检查两侧目录字段是否对称**就做配对比较。

## Provenance

- harness 改动: commit `a163a89`, `scripts/eval_olmo2_probe2_downstream.py`
- 新工具: `scripts/enrich_per_example_normscores.py`, `evidence_evalfragility_code/accnorm.py`
- agent 交付: `status/PAPERF_ACCNORM_REDO.md`
- 我的复核: zwfy6 上逐 rung 重算 6×17,195 个 item 的 norm_scores margin + exact permutation p
- 前置: `status/PAPERB_TWO_CORPORA_DEFECT.md`（今天更严重的那个发现）
