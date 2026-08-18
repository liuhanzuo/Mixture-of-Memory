# Scaffold-Coder 双实现对照与 Runtime 分析报告

**报告更新日期：** 2026-08-05

## 1. 重要更正

此前我们暂时将队友报告的：

```text
HumanEval+ = 5.5%
MBPP+      = 23.8%
```

理解为：

```text
Scaffold checkpoint + vanilla Dream decoding
```

但队友提供的完整报告表明，这一理解是错误的。

队友实际运行的是：

```text
structural-dLLM/scaffold/train/sft.py
```

训练入口为：

```text
scaffold.train.sft
```

评测时也使用了专门的 constrained structural decoding，而不是普通
Dream-Coder vanilla sampler。

因此：

> `5.5% / 23.8% → 18.29% / 32.01%` 不能解释为同 checkpoint 下
> Scaffold runtime 相比 vanilla decoding 的因果增益。

这两组结果来自两个不同的 Scaffold-Coder 实现，训练 schedule 和
runtime 容量均不同。它们之间的差值只能称为：

```text
跨实现系统差异
```

不能称为：

```text
纯 runtime 消融
```

---

## 2. 两个系统分别是什么

### 2.1 我们的 Stage-1 Scaffold 实现

训练入口：

```bash
python -m scaffold_coder.training.scaffold_sft_trainer
```

训练脚本：

```text
scripts/run_scaffold_sft_stage1_8gpu.sh
```

最终 checkpoint：

```text
outputs/scaffold_sft_stage1/global_step_4465
```

主要配置：

```text
初始化模型              Dream-Coder-v0-Base-7B
训练样本数              114,363
训练步数                4,465
训练轮数                5
学习率                  1e-5
全局 batch              128
每卡 micro-batch        8
训练并行                8-way FSDP
模式                    hierarchical
gating                  strict / local-body
desync                   0
plain mix               0
```

主要 runtime 参数：

```text
initial root slots       1
initial body slots       2
initial statement masks  4
function header masks    4
loop header masks        4
condition masks          3
max tree depth           16
max lines per body       128
max tokens per hole      512
max expansions           512
max model calls          512
module expand            enabled when legal
```

其特点是：

- 允许递归结构展开；
- body 可以增长到较多行；
- token hole 可以扩展到较长内容；
- module-level 和 body-level 都允许受预算控制的 expand；
- 使用重复状态检测和 expand/delete cycle suppression；
- 最终由可变树确定性渲染为 Python。

### 2.2 队友的 structural-dLLM v0 实现

训练入口：

```text
structural-dLLM/scaffold/train/sft.py
```

最终 checkpoint：

```text
outputs/v0/epoch_4
```

训练配置：

```yaml
data:
  dataset: educational_instruct
  max_length: 1024

train:
  micro_batch_size_per_gpu: 8
  global_batch_size: 128
  lr: 1.0e-5
  epochs: 5
  warmup_ratio: 0.1
  gradient_checkpointing: true
  seed: 1

schedule:
  layout: layout_s_v0
  content_region: [0.0, 0.45]
  structure_region: [0.45, 0.95]
  depth_cap: 4
  desync: 0.15
  plain_ratio: 0.10
```

训练统计：

```text
training steps           4,570
epochs                   5
final train loss         0.2194
base eval loss, 8 rows   0.9484
final eval loss, 8 rows  0.2323
final eval loss, 64 rows 0.2080
```

评测 runtime：

```text
temperature              0.0
init_slots               2
n_hdr                    1
n_stmt                   1
n_doc                    1
max_line_slots           2
max_token_slots          2
max_structural_depth     1
max_total_lines          16
max_steps                512
module-level expand      disabled
```

这一 runtime 的约束明显更紧：

- 每个 header/statement 初始只有一个 slot；
- 每个 line/token region 最多只有两个 slot；
- 最大结构深度只有 1；
- 整个程序最多 16 行；
- module-level expand 被关闭。

---

## 3. 结果对照

### 3.1 功能正确率

| 实现 | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---:|---:|---:|---:|
| 我们的 Stage-1 Scaffold | 19.51% | **18.29%** | 38.36% | **32.01%** |
| 队友 structural-dLLM v0 | 6.10% | **5.49%** | 27.51% | **23.81%** |
| 跨实现差值 | +13.41 pp | **+12.80 pp** | +10.85 pp | **+8.20 pp** |

这里的差值是整个系统的差异，包含：

- 训练 schedule；
- desynchronization；
- plain-data mix；
- 数据过滤和步数差异；
- runtime 容量；
- expand 策略；
- 终止策略；
- sampler 实现。

因此不能将差值归因于单一组件。

### 3.2 Generation failure

| 实现 | HumanEval failure | MBPP failure |
|---|---:|---:|
| 我们的 Stage-1 Scaffold | 15 / 164 = **9.15%** | 35 / 378 = **9.26%** |
| 队友 structural-dLLM v0 | 77 / 164 = **46.95%** | 161 / 378 = **42.59%** |

这是两个系统之间最明显的差异。

队友系统接近一半的任务到达了 sampler step limit。我们的系统也有
budget failure，但比例约为 9%。

### 3.3 NFE

队友报告：

| Benchmark | 全任务平均 NFE | 中位数 | P90 | 成功样本平均 NFE |
|---|---:|---:|---:|---:|
| HumanEval+ | 249.2 | 35 | 511 | 17.4 |
| MBPP+ | 246.3 | 68 | 510 | 50.7 |

我们的报告：

| Benchmark | 成功样本平均 NFE | 中位数 | P90 |
|---|---:|---:|---:|
| HumanEval+ | 58.79 | 53 | 96.0 |
| MBPP+ | 47.77 | 41 | 83.6 |

两边的平均口径不同：

- 队友的全任务均值包含达到 step limit 的失败任务；
- 我们最初报告的均值只覆盖完成生成的任务。

若将我们失败的任务按 512 NFE 计入，则近似得到：

```text
HumanEval 全任务平均 NFE ≈ 100.2
MBPP 全任务平均 NFE      ≈ 90.8
```

因此：

| 实现 | HumanEval 全任务 NFE | MBPP 全任务 NFE |
|---|---:|---:|
| 我们的 Stage-1 Scaffold | 约 100 | 约 91 |
| 队友 structural-dLLM v0 | 249 | 246 |

队友系统在成功任务上的 HumanEval NFE 很低，但高失败率使总体成本显著
上升。

---

## 4. 这组对照真正说明了什么

### 4.1 Runtime 容量是一阶设计变量

队友 runtime 使用：

```text
max_structural_depth = 1
max_total_lines      = 16
max_line_slots       = 2
max_token_slots      = 2
module expand        = disabled
```

这套限制可以让简单成功样本很快结束：

```text
HumanEval 成功样本平均 NFE = 17.4
```

但是它也可能导致：

- 程序长度不够；
- header/statement 表达空间不足；
- module 无法增加顶层结构；
- 无法生成需要多层嵌套的程序；
- 模型不断尝试 expand，但 runtime 不允许；
- 解码状态无法满足任务，于是反复运行到 step limit。

最终表现为：

```text
成功样本非常快
失败样本非常慢
总体失败率接近一半
```

这是一个很有论文价值的观察：

> 对结构化 dLLM 而言，runtime 的结构容量与终止策略不是普通超参数，
> 而是直接决定质量—成本 Pareto frontier 的核心机制。

### 4.2 过度约束对 HumanEval 的影响大于 MBPP

两个系统的 Plus 分数差异：

```text
HumanEval+：12.80 个百分点
MBPP+：      8.20 个百分点
```

队友系统：

```text
HumanEval+ = 5.49%
MBPP+      = 23.81%
```

说明它对较短、较局部的 MBPP 任务仍有一定能力，但对更依赖完整函数
组织和语义规划的 HumanEval 损伤更大。

`max_structural_depth=1`、`max_total_lines=16` 和关闭 module expand 是
最可疑的原因。

### 4.3 训练 loss 不能预测完整生成能力

队友模型的训练和 eval loss 都明显下降：

```text
base eval loss          0.9484
final eval loss, 64     0.2080
```

但 HumanEval+ 只有 5.49%，并有 46.95% generation failure。

这再次证明：

> 部分 mask 条件下的 token reconstruction loss 不能充分反映从 prompt
> 和结构 mask 开始的完整程序生成能力。

训练 loss 应只作为优化稳定性指标，不能作为功能质量替代指标。

### 4.4 Desync 和 plain mix 没有自动解决问题

队友训练使用：

```text
desync       = 0.15
plain_ratio  = 0.10
```

我们的主训练使用：

```text
desync       = 0
plain mix    = 0
```

尽管队友加入了原设计中推荐的 soft/desynchronized 状态和 10% plain
样本，最终分数仍然更低。

但由于 runtime 同时发生了巨大变化，不能据此单独判断 desync 或
plain mix 有害。要判断它们，需要固定 runtime 后做训练消融。

---

## 5. 为什么我们之前的 18.29% / 32.01% 仍然有用

它依然是一个有价值的系统结果，因为它证明：

1. 分层结构 SFT 可以端到端训练；
2. meta-token 可以动态展开为递归程序树；
3. runtime 可以将结构状态可靠地转化为 Python；
4. 在约 48–59 次平均模型调用下，能够获得：
   - HumanEval+ 18.29%
   - MBPP+ 32.01%
5. 相比更严格的 structural-dLLM v0，实现更低的总体失败率和更好的
   功能分数。

但是，准确 claim 应当是：

> 我们当前的 runtime/configuration 比一个更严格、容量更小的
> Scaffold v0 系统效果更好。

而不是：

> Runtime 相比同 checkpoint 的 vanilla decoding 提升 12.8/8.2 点。

后一个 claim 尚未被这两组实验支持。

---

## 6. 目前可以提出的更有趣研究问题

这两个独立实现共同表明，结构化 dLLM 的关键可能不是：

```text
是否使用 AST / meta-token
```

而是：

```text
结构 runtime 应该提供多大的表达容量，以及如何可靠终止。
```

可以将下一阶段问题定义为：

> **Capacity-Calibrated Executable Scaffolds for Diffusion Code Models**

即研究：

- 结构深度预算；
- 行数预算；
- token slot 预算；
- module/body expand；
- 成功任务快速结束；
- 困难任务动态扩容；
- failure-aware NFE；
- verifier 触发扩容或回退。

专业 reviewer 可能会对以下观点感兴趣：

> 静态的小 canvas 能让简单任务极快完成，但会在困难任务上形成
> failure-heavy 长尾；自适应结构容量可以同时保留低成功样本 NFE 和
> 降低 step-limit failure。

---

## 7. 下一步最关键的实验

### 7.1 固定队友 checkpoint，放宽 runtime

在 `outputs/v0/epoch_4` 上逐项修改：

| 消融 | 原配置 | 放宽配置 |
|---|---:|---:|
| max structural depth | 1 | 2 / 4 / 8 |
| max total lines | 16 | 32 / 64 |
| max line slots | 2 | 4 / 8 / 16 |
| max token slots | 2 | 4 / 8 / 32 |
| module expand | disabled | enabled |
| initial header slots | 1 | 2 / 4 |
| initial statement slots | 1 | 2 / 4 |

该实验能回答：

> 5.5% / 23.8% 到底有多少是 runtime 容量不足造成的？

### 7.2 固定我们的 checkpoint，收紧 runtime

在 `global_step_4465` 上复制队友的限制：

```text
depth 1
16 lines
2 line slots
2 token slots
module expand off
```

如果我们的分数也接近 5.5% / 23.8%，就能强力证明 runtime capacity 是
主要原因。

### 7.3 二乘二 checkpoint/runtime 实验

如果两个实现的 token schema 能兼容，最理想的设计是：

| 训练 checkpoint | 小容量 runtime | 大容量 runtime |
|---|---:|---:|
| 队友 v0 checkpoint | A | B |
| 我们 Stage-1 checkpoint | C | D |

这能分解：

- checkpoint/training effect；
- runtime effect；
- checkpoint × runtime interaction。

如果 schema 不兼容，则至少在两个实现内部进行容量 sweep。

### 7.4 统一成本口径

每个配置必须同时报告：

- 成功任务平均 NFE；
- 全任务平均 NFE；
- step-limit failure rate；
- rule-only step 数；
- cumulative model tokens；
- wall-clock；
- parseability；
- pass@1。

只报告成功样本 NFE 会使小容量 runtime 显得异常高效，却忽略大量
失败任务的 510–511 NFE。

### 7.5 按复杂度分析 failure

建议将任务按以下维度切分：

- canonical compound depth；
- canonical solution 行数；
- response token 长度；
- 函数数量；
- 是否需要 loop/if 嵌套；
- 是否需要 imports/helper；
- 生成的最大树深度；
- 最终 line/token slot 使用量。

预期会看到：

> 队友小容量 runtime 在短、浅任务上具有很低 NFE，但随着 canonical
> 长度和深度增加，failure rate 快速上升。

---

## 8. 当前支持和不支持的结论

### 8.1 可以支持

- 两个独立 Scaffold-Coder 实现都完成了 5 epoch 训练和完整 EvalPlus
  评测。
- 我们的系统获得 18.29% HumanEval+ 和 32.01% MBPP+。
- 队友 v0 获得 5.49% HumanEval+ 和 23.81% MBPP+。
- 我们的系统 generation failure rate 约 9%，队友系统约 43–47%。
- 队友系统成功任务可以很快完成，但具有非常重的 step-limit failure
  长尾。
- Runtime 容量和终止策略很可能是结构化 diffusion code generation
  的核心变量。

### 8.2 不能支持

- 12.80/8.20 点差异完全来自 runtime；
- 两个系统使用同一个 checkpoint；
- 队友结果是 vanilla decoding；
- desync 或 plain mix 单独造成了下降；
- 我们 runtime 的某一个单独组件贡献了全部增益；
- 当前任一 Scaffold 系统超过 Dream-Coder。

---

## 9. 推荐的论文定位

基于这两个实现，最有潜力的论文方向可以从：

```text
结构化 SFT 比普通 dLLM 更强
```

转为：

```text
结构化 dLLM 的性能由 checkpoint 与 executable runtime 共同决定；
静态结构容量会产生“简单任务快速完成、困难任务长尾失败”的现象；
自适应容量和结构化修订能够改善该 Pareto frontier。
```

可以使用的标题方向：

> **Executable Scaffolds for Diffusion Code Models: Capacity, Termination,
> and Structural Revision**

中文：

> **面向扩散代码模型的可执行脚手架：结构容量、终止与结构化修订**

---

## 10. 最终判断

队友的 report 并没有否定我们之前的结果，但它改变了对比的含义。

正确结论是：

> 我们和队友都训练并评测了 Scaffold SFT 系统；我们的实现取得了更高
> 的功能分数和更低的总体 failure-adjusted NFE。两者差异同时来自训练
> schedule 和 runtime 容量，当前不能作为纯 runtime 消融。

这组对照最有价值的发现是：

> 在结构化 dLLM 中，过小且静态的结构 canvas 会让成功样本极快完成，
> 但造成大量困难任务达到 step limit；runtime capacity/termination
> 是与训练方法同等重要的一阶设计变量。

