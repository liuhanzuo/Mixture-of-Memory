# SparseForge 主表 headline 有一个 20pp 的 cell 错误 —— margin 从 +2.75pp 塌到 +0.53pp

**日期**: 2026-08-11 12:50 GMT+8
**来源**: workflow `w677cpnqo`(#241, 20 agents) 报了 33 条 finding（17 P0 / 16 P1）。本文档只记我**自己逐条独立复核过**的三条。
**状态**: ⛔ 论文当前 headline 不可提交。**我在 12:23 那轮汇报给用户的 "+2.75pp" 是错的**，据此纠正。

---

## D1 (P0) — RTE cell 69.82 在两个盘上都没有 provenance，真值 49.82

`SparseForge_NIPS_2026/sections/experiments.tex:33` 的 RTE cell 写 **69.82**。

**盘上每一个 artifact 都说 49.82**：

| artifact | rte |
|---|---:|
| `SparseForge_Data/results/rebuttal_artifacts_2026-07-27/historical_sparseforge/ast7_eval.json` | `49.81949458483754` |
| `SparseForge_Data/tables/cast9_dense_ast_current_harness.csv`（`SparseForge-5B` 行） | `49.81949458483754` |
| `SparseForge_Data/results/rebuttal_artifacts_2026-07-27/tables/*.csv` | 同上 |

我用转义点 `grep -rlE '69\.8[0-9]|0\.698'` 全盘搜 —— 命中的 6 个文件全是**其他方法**的 log（elsa / alps / proxsparse），且抽出来的值是 `69.84` 等，不是 SparseForge 的 RTE。**69.82 在盘上不存在。**

### 后果：AVG-9 被单个 cell 抬高 2.22pp

| | AVG-9 | vs AST official (57.94) |
|---|---:|---:|
| 论文印的（rte=69.82） | 60.6889 → 印 60.69 ✅ 自洽 | **+2.75** |
| 盘上真值（rte=49.82） | **58.4663** | **+0.53** |

一个 cell 差 20pp ÷ 9 = 2.22pp 的 AVG 膨胀。**AST 那行的 AVG-9 从它自己的 json 重算是 57.938，与印的 57.94 吻合 —— 所以 baseline 没问题，只有我们这行是虚高的。**

⚠️ 同一行还有第二处不一致：**openbookqa** 论文 35.20 / csv 35.2 / `ast7_eval.json` **35.4**。论文脚注自己承认了（"including OBQA 35.20 rather than 35.40"），但那个脚注同时暴露了更大的问题 —— 它说 "BoolQ/RTE come from a separate unified invocation on the same checkpoint"，即**这 9 个 cell 不是一次 invocation 出来的**。跨 invocation 拼表正是我们刚在 CAST union-9 上踩过的坑。

## D2 (P0) — token budget "5B" 与归档 args 差 3.8x

`args.json`（同一个 `historical_sparseforge` 目录）实测：

```
max_iters            = 17000
final_finetune_iters = 3000
global_batch_size    = 256
block_size           = 4096   (来自 checkpoint inventory)
saved iter           = 17900  (best_lm_eval.json)
```

17900 × 256 × 4096 = **18.77B nominal token**，论文写 **5B**（`experiments.tex:33` 行标签、`:84`、`:74` 的 50 GPU-h 估算、`appendix.tex:325/110`）。**差 3.8x。**

> ⚠️ **我没能核实的部分**：workflow 还说语料只有 ~130M unique token（即 5B≈38 epoch、18.77B≈145 epoch）。它引的 `data/qa_format_sft_llama/metadata.json` **在本盘不存在**（`find SparseForge_Data -name metadata.json` 空，`grep -rl train_tokens` 空）。按两盘规则这条**尚未成立**，需要在 zwfy6 上再找一遍才能下结论。token 算术那半（18.77B vs 5B）我已独立确认。

## D3 (P0) — 复现性超参表与真实 run 至少 9 项不符

`appendix.tex:206-220` 的 hyperparameter 表 vs 同一个 `args.json`：

| 项 | 论文 | 盘上实际 |
|---|---|---|
| base-weight LR | 2e-5 | **1e-4** |
| T0 / decay | 1.0 / γ=0.99 | **2.0 / 0.98** |
| mask-update interval | 100 steps | **10** |
| η_pen | 0.10 | **0.0（penalty 根本没开）** |
| λ_mid | linear 0→0.1 | **0.3** |
| hardening window | final 10% | **0.2** |
| post-projection recovery | 1,000 steps | **3000** |
| global batch / seqlen | 32 / 2048 | **256 / 4096** |
| Hutchinson probes | 1 per update | **`enable_hutchinson=False`（从未跑过）** |

最后两项不只是笔误：
- `methodology.tex:159` 写 "We linearly ramp lambda_mid from 0 to 0.1"，盘上是 0.3 —— **方法描述本身错了**。
- **Hutchinson estimator 是论文列的 contribution (2)，而 headline run 里它是关闭的。** 这是"声称的贡献未在主结果中启用"，比数字错更严重。

---

## 直接回答用户的问题："SparseForge 相比 AST/CAST 提升了么"

**提升了，但幅度是 +0.53pp，不是论文印的 +2.75pp。**

| method | AVG-9 | Wiki PPL |
|---|---:|---:|
| AST official ckpt | 57.94 | 6.3430 |
| **SparseForge（真值）** | **58.47** | **6.2179** |

PPL 的优势（6.2179 vs 6.3430）**不受这个 bug 影响**，是独立的、真实的。

而 token 效率表（`SparseForge 1.25B → 59.22` vs `AST 7.5B → 57.68/58.62`）**用的是另一个 run（1.25B）**，与主表的 5B/18.77B 行不是同一个 ckpt。这张表我**还没核**，它现在是最有力的卖点（6x 更少 token 还更高分），所以必须优先核实 —— 如果它也有类似问题，SparseForge 的核心论证就只剩 PPL。

### ★ 这也印证了用户的原始判断，但方向要改

用户说「表现相近是因为受模型 bound」。**+0.53pp 确实是 near-tie** —— 但它 near-tie 的原因**不是** LLaMA-2 的能力上限，而是**论文的 headline 本来就虚高了 2.22pp，真实 margin 一直很小**。换 Qwen3.5 之前必须先把这个修掉，否则会拿一个错的 baseline margin 去和新模型比。

---

## 下一步（未执行，等决策）

1. **P0 纯文本，0 GPU**：把 RTE 改回 49.82、AVG-9 改 58.47，并连带修 `abstract.tex:4`、`experiments.tex:105`、`motivation.tex:14/18` 四处传播点。
2. **P0，0 GPU**：超参表按 `args.json` 逐项改正；`methodology.tex:159` 的 λ_mid 改 0.3；**决定如何处理 Hutchinson**（要么在主 run 开启重跑，要么把它降级为"未在主结果启用的可选组件"）。
3. **P0，~0.3 GPU-h**：核实 token 效率表那行（1.25B run）的 provenance，它现在是 SparseForge 最强的论证。
4. token budget：要么改成 18.77B 并重算 GPU-h，要么找到 5B 的真实依据。**不能两个数并存。**

其余 30 条 finding（含 SF-02 contamination 80 GPU-h、SF-06 ablation 无 artifact 32 GPU-h、CAST-01..09、RO-01..05、S8-*、PN-*）在 `w677cpnqo` 的完整输出里，**我尚未逐条复核，不得直接引用**。
