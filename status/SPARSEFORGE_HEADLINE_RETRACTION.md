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

### ⚠️ 更正（12:55，同日）：我说「token 效率表是最有力的卖点」也讲错了

核实后两点都不成立：

1. **那张表已被关闭。** `experiments.tex:42` 是 `\iffalse`，`:60` 是 `\fi`，`label` 本身写着
   `tab:llama2_compare_deprecated_ast7`。它不在编译产物的论证链里。

2. **活着的 scaling curve 反而削弱这个论证。** `appendix.tex:336`「(b) Historical
   SparseForge token scaling」是活的：

   | tokens | 0.625B | 1.25B | 2.5B | 5B | 7.5B |
   |---|---:|---:|---:|---:|---:|
   | CAST-7 | 55.70 | 55.96 | 56.65 | 57.27 | 57.40 |

   1.25B → 7.5B 只涨 **1.44pp**，5B → 7.5B 只涨 **0.13pp**（已饱和）。
   如果 1.25B 就已接近饱和，那 AST 用 7.5B 拿 57.68 **不是 token 效率的差距**，而是
   AST 方法本身在任何 budget 下都该拿这么多。真要支撑 token 效率，需要
   **同一条 curve 上低 budget 处显著高于 AST** —— 但 AST 只有 7.5B 一个点，
   **做不出 budget-matched 对比**。

   （注：两表口径不同，A 表是 AST-7、curve 是 CAST-7，数值不可直接对比；但趋势可比，
   而趋势的方向对这个论证不利。）

**所以 SparseForge 目前真正站得住的只有两条**：AVG-9 **+0.53pp**（修正后）和
**PPL 6.2179 vs 6.3430**。token 效率这条论证需要新证据（AST 的 budget ladder）才能立起来，
而我们跑不了 AST 的 ladder（没有它的训练代码）。

### ★ 这也印证了用户的原始判断，但方向要改

用户说「表现相近是因为受模型 bound」。**+0.53pp 确实是 near-tie** —— 但它 near-tie 的原因**不是** LLaMA-2 的能力上限，而是**论文的 headline 本来就虚高了 2.22pp，真实 margin 一直很小**。换 Qwen3.5 之前必须先把这个修掉，否则会拿一个错的 baseline margin 去和新模型比。

---

## 下一步（未执行，等决策）

1. **P0 纯文本，0 GPU**：把 RTE 改回 49.82、AVG-9 改 58.47，并连带修 `abstract.tex:4`、`experiments.tex:105`、`motivation.tex:14/18` 四处传播点。
2. **P0，0 GPU**：超参表按 `args.json` 逐项改正；`methodology.tex:159` 的 λ_mid 改 0.3；**决定如何处理 Hutchinson**（要么在主 run 开启重跑，要么把它降级为"未在主结果启用的可选组件"）。
3. **P0，~0.3 GPU-h**：核实 token 效率表那行（1.25B run）的 provenance，它现在是 SparseForge 最强的论证。
4. token budget：要么改成 18.77B 并重算 GPU-h，要么找到 5B 的真实依据。**不能两个数并存。**

其余 30 条 finding（含 SF-02 contamination 80 GPU-h、SF-06 ablation 无 artifact 32 GPU-h、CAST-01..09、RO-01..05、S8-*、PN-*）在 `w677cpnqo` 的完整输出里，**我尚未逐条复核，不得直接引用**。

---

# 追加：workflow `wsxo6dv4n`(#242) 的两条新缺陷（12:59 独立复核完毕）

## D4 (P0) — 两套结果集不是同一个 harness，同模型差最多 1.60pp

同一个 **dense LLaMA-2-7B**、同为 **plain acc**，两处结果不同：

| task | `SparseForge_Data/tables/*.csv` | `outputs/cast_eval_spec/` | delta |
|---|---:|---:|---:|
| openbookqa | 33.2000 | 31.6000 | **−1.6000** |
| arc_easy | 75.5471 | 76.3468 | +0.7997 |
| arc_challenge | 42.7474 | 43.1741 | +0.4267 |
| winogrande | 69.6133 | 69.2976 | −0.3157 |
| race | 39.5215 | 39.7129 | +0.1914 |
| hellaswag | 57.1002 | 57.1301 | +0.0299 |
| piqa | 77.8564 | 77.8564 | 0.0000 |

mean |Δ| = 0.4805，max |Δ| = 1.60（obqa）。**这是第三个口径陷阱**（前两个：acc_norm 混用、PPL seqlen）。

**对 D1 的影响 —— 一个好消息**：SparseForge-5B 行和 AST official 行**都来自同一个
`cast9_dense_ast_current_harness.csv`**，即同一 harness。所以修正后的 **+0.53pp margin
内部是自洽的**，不受这个 drift 污染。但**任何把该 csv 的数字与 `cast_eval_spec/` 的数字
放进同一张表的做法都是错的**（例如想把我们新跑的 CAST-repro union-9 直接拼到 SparseForge 主表里）。

## D5 (P1) — csv 自己的 cell 列与 mean 列不一致，已定位到 obqa

`cast9_dense_ast_current_harness.csv` 的 `SparseForge-5B` 行：

* 7 个 cell 重算 → **58.3195**
* csv 印的 `ast7_mean` → **58.3481**
* 差 **0.0286**

追因：`0.2 / 7 = 0.0286`。**csv 的 mean 列是用 `obqa = 35.4` 算的，而它自己的 cell 列写 35.2**
（`ast7_eval.json` 里是 35.4）。同一个文件里 cell 与 aggregate 用了不同的值。

AST official 那行**没有**这个问题（重算 57.9436 = csv 57.9436，delta 0.0000），
再次说明只有我们这行有账目问题。

> 这条不改变 +0.53pp 的量级（0.03pp 级），但它说明该 csv 不能当权威源直接引用 ——
> 必须回到 `ast7_eval.json` 重算。

## 其他 workflow 报告但我尚未复核的项（不得引用）

* AST ckpt 的 2:4 已验证 PASS（zero_frac 0.500000000，bad_tiles 0，1,619,001,344 tiles）—— 这条与我 union-9 时的验证一致，可信度高但我这轮没重跑。
* `SPEC.md` 里 `ast_official_clean` 的路径写错了（那是 AST **源码仓**，不是 ckpt；ckpt 在 `models/AST-official-LLaMA2-7B-2of4`）。
* ⚠️ **环境被改过**：`lm_eval` 的 piqa yaml 被就地改成 `dataset_path: ybisk/piqa`，**没有 `.bak`/`.orig`** —— 这是对 site-packages 的未版本化修改，应记进 SPEC 的环境说明，否则换机器复现不出来。
* RTE 精度地板：n=277 → worst-case stderr **3.00pp**（boolq n=3270 → 0.87pp）。**RTE 会主导 union-9 的噪声**，需要脚注 —— 这与 union-9 的 leave-one-out 发现（+0.92pp 全靠 RTE）互相印证。
