# Semantic-Preservation Gate

日期：2026-08-05

## 训练

语义保真 LoRA 从 Dream-Coder Instruct 初始化，采用：

```text
root/body/leaf = 0.10 / 0.20 / 0.70
LoRA rank      = 16
epochs         = 1
steps          = 893
```

训练完成：

```text
final validation loss = 0.581
last-100 train loss    ≈ 0.573
```

## 第一版 HumanEval 门控

| Decode | HE+ | Parse | Generation failure |
|---|---:|---:|---:|
| Vanilla 512 NFE | 25.61% | 59.15% | 0 |
| Scaffold Medium | 1.22% | 25.00% | 0.61% |

该结果不能作为最终语义结论。输出检查显示：

```text
vanilla literal edit-word pollution:  76 / 164
Scaffold literal edit-word pollution: 154 / 164
```

### 根因

新增 token 行被冻结，而 `[expand]` / `[delete]` 的 input/output rows 从普通
`expand`, `mask`, `delete`, `remove` token 初始化。LoRA 可以改变 hidden
states，却不能拉开相同或高度相似的 output rows。若普通 source rows 仍在
候选词表中，模型会把结构动作输出成字面 Python 文本：

```text
expandexpand
deletedelete
maskmask
```

### 协议修正

1. Vanilla decode 屏蔽所有 Scaffold tokens 和四个 edit-source rows。
2. Scaffold token holes 屏蔽四个普通 edit-source rows，但保留真正的
   `[expand]` / `[delete]` 规则动作。

16-task smoke：

| Decode | Parseable | Pollution | Error |
|---|---:|---:|---:|
| Vanilla suppressed | 13/16 | 1/16 | 0 |
| Scaffold Medium suppressed | 12/16 | 0/16 | 0 |

因此 suppression protocol 通过 smoke gate，现重跑 full HumanEval。仅当
full gate 满足：

```text
vanilla HE+ ≥ 45%
Scaffold HE+ ≥ 30%
Scaffold generation failure ≤ 5%
```

才进入 MBPP+。

## Suppressed full vanilla result

抑制后 full vanilla：

```text
HumanEval+ = 35.98%
HumanEval  = 38.41%
parse rate = 73.78%
pollution  = 5 / 164
```

相比 protocol-invalid 的 25.61% 恢复 10.37 个百分点，但仍低于预注册
`40%` 止损线，故停止当前 scale=1.0 的 Scaffold/MBPP 评测。

下一步不立即重训，而是在固定 64-task HumanEval screening set 上校准
LoRA merge scale：

```text
0.25 / 0.50 / 0.75
```

目标是在保留 Instruct 语义能力的同时寻找最小有效结构适配强度。

首次 scale calibration 的 `scale=0.25` generation 已完整覆盖 64/64，但
EvalPlus 仍加载 164-task problem set，触发 `Missing problems in samples`。
该问题只影响评测路由，不影响生成；已将
`HUMANEVAL_OVERRIDE_PATH` 改为同一个固定 64-task 文件并恢复运行。

## LoRA scale calibration 结果

固定 64-task stratified HumanEval screening set、512 NFE、suppressed
vanilla decode：

| LoRA scale | HumanEval | HumanEval+ | Parse | Generation error |
|---:|---:|---:|---:|---:|
| **0.25** | **64.06%** | **59.38%** | **100.00%** | 0 |
| 0.50 | 53.12% | 46.88% | 92.19% | 0 |
| 0.75 | 48.44% | 43.75% | 87.50% | 0 |

结果随 LoRA scale 增大单调退化，因此按预注册的“最大 HE+，再比较
parseability”规则选择：

```text
scale = 0.25
checkpoint = outputs/semantic_lora_scale_calibration/checkpoint_scale_025
```

该 64-task 结果仅用于选择 scale，不作为最终 benchmark 结论。当前已启动
同一 checkpoint 的完整 164-task HumanEval+ vanilla preservation run；
只有 full HE+ ≥45% 且 generation error=0，才运行 full Scaffold Medium。

## Selected scale full gate

`scale=0.25` 的完整 164-task 结果：

| Decode | HumanEval | HumanEval+ | Parse | Generation failure |
|---|---:|---:|---:|---:|
| Vanilla suppressed | **60.37%** | **56.10%** | **95.73%** | 0 |
| Scaffold Medium | 0.00% | 0.00% | 86.59% | 142 / 164 |

因此 vanilla preservation gate 已通过，并且比 Dream-Coder Instruct 的同
harness 50.00% HE+ 高 6.10 个百分点。不过当前 Scaffold runtime gate
失败，不能进入 MBPP。

Scaffold failure attribution：

```text
depth_capacity_exhausted      138
total_line_capacity_exhausted   4
model-call/expand-cycle failure  0
```

失败任务的 median partial NFE 为 58.5，expansion 次数几乎为零。这说明
根因不是生成长尾，而是 line-level 结构分布在 `scale=0.25` 时过度预测
compound meta-token，递归扩展到 depth cap。

当前在固定 64-task set 上校准仅作用于 `LINE_BODY` 的 compound-construct
logit penalty：

```text
0 / 1 / 2 / 4
```

该 penalty 不改变 module-level `[FUNC]`、lexical token 或 vanilla decode。
只有 failure ≤5%、parse ≥90% 的 arm 才有资格进入 full Scaffold 复测。

### Nested-construct penalty 结果

固定 64-task set、`scale=0.25`：

| Penalty | HE+ | Parse | Generation failure |
|---:|---:|---:|---:|
| 0 | 0.00% | 89.06% | 57 / 64 |
| 1 | 0.00% | 67.19% | 42 / 64 |
| 2 | 0.00% | 29.69% | 18 / 64 |
| 4 | 0.00% | 6.25% | 2 / 64 |

Penalty 能单调压低递归 depth failure，但不能恢复代码质量。进一步检查显示
所有 arm 的 `maximum_tokens_per_hole` 都固定为 4，模型几乎不预测 token
level `[expand]`；因此成功终止的程序仍普遍存在 header/statement 截断。

下一步固定 `penalty=4`，使用 HumanEval 已知函数签名，并校准初始 statement
长度：

```text
4 / 8 / 16 masks
```

这一步用于区分“签名预测错误”和“叶子长度不足”，仍只在 64-task set 上
筛选。

## Structure-token-row adaptation smoke

Inference-only calibration 全部未达到门槛后，尝试只更新 19 个结构 token
的 input/output rows，保持 Transformer 和普通词表不变。

原生 PEFT TrainableTokens smoke：

```text
effective trainable parameters = 136,192
adapter size                   ≈ 545 KB
ordinary vocabulary rows       bit-exact unchanged
train loss                     3.823 → 2.103 → 1.430 → 1.700
validation loss                1.574
```

但是 16-task decode 全部在 1 NFE 预测 module-level `[delete]`，生成空程序：

```text
nonempty       0 / 16
with function  0 / 16
```

因此该 checkpoint 判为失败，不进入完整训练。下一步必须避免训练
module-level delete 行，并在小 LR/少步数下重新校准 root target balance；
仅看 `ast.parse("")` 会造成假阳性，后续 smoke 必须同时要求 nonempty 和
required function。

参数级 gradient hook 在 FSDP 下未能保持普通词表 bit-exact，因此弃用。
最终实现改用 PEFT 0.19 原生 compact `TrainableTokens`：

```text
input embedding delta = 19 × 3584
lm_head delta         = 19 × 3584
total trainable       = 136,192
adapter size          ≈ 545 KB
ordinary rows stored  = 0
```

合并后的 standalone checkpoint 验证：

```text
ordinary input/output rows bit-exact unchanged
selected rows max delta ≈ 0.004
```

训练状态审计显示 `[delete]` 占监督目标：

```text
root-plan 28.6%
body-plan 20.9%
leaf      13.2%
```

为彻底移除 delete/expand 混杂，下一轮只训练真实 topology 标签：

```text
FUNC / FOR / WHILE / IF / ELIF / ELSE / STMT
```

固定训练配方：

```text
expand/delete supervision = 0
root/body/leaf = 0.5 / 0.5 / 0
LR = 1e-4
steps = 4
```

Topology-only smoke 已完成：

```text
trainable parameters = 50,176
train loss = 4.096 → 2.689 → 2.157 → 1.635
validation loss = 1.716
selected row max delta ≈ 0.0004
ordinary rows stored = 0
```

Standalone merge 验证普通 input/output rows bit-exact，selected rows 的
merge 后最大变化为 `4.88e-4`。16-task nonempty/function gate 结果：

```text
nonempty          10 / 16
with function     10 / 16
parseable          6 / 16
generation errors  6 / 16
depth failures     6 / 16
```

因此该 arm 同时未达到 `nonempty=16`、`function=16` 和 `parse≥8` 三个
冻结门槛，不进入 full epoch。这个结果把问题进一步定位为两部分：

1. 排除 `[delete]` 后，root 全删问题确实消失；
2. 单独强化 topology rows 仍会加剧 compound token 过预测，并且不会修复
   由 frozen edit/lexical rows 引起的 header 与 statement 长度不足。

下一项转向 teacher-anchored joint objective：结构位置继续做 topology CE，
普通 masked/lexical states 对 frozen Dream-Coder Instruct 做 KL/replay
锚定。任何完整训练之前先通过 4-step vanilla + Scaffold 双解码 gate。

## Teacher-KL smoke

已实现第一版 joint objective：

```text
student LoRA rank = 8
teacher = frozen Dream-Coder Instruct
root/body/leaf = 0.15 / 0.25 / 0.60
lexical KL roles = TOKEN_STMT / TOKEN_HDR / TOKEN_DOC
KL weight = 1.0
temperature = 1.0
teacher support = top-256
teacher placement = replicated BF16 per GPU
LR = 2e-5
steps = 4
```

CE 仍覆盖采样状态的全部合法目标；KL 只锚定 lexical roles，避免 teacher
对从未预训练过的 topology rows 提供无意义监督。teacher/student 共享
输入 canvas。双独立 FSDP 版本连续约 15 分钟无法完成首步，profile 显示
两棵 FSDP 树交替卡在 unshard rate limiter，因此停止。修正版将冻结的
BF16 teacher 完整常驻每卡，仅 student 使用 FSDP。继续门槛：

- 8 卡无 OOM/NaN；
- 4 steps loss finite 且 checkpoint 完整；
- merge 后 16-task vanilla parse ≥8/16、无生成异常；
- Scaffold nonempty/function ≥12/16、parse ≥8/16、failure ≤4/16。

修正版 4-step smoke 已通过：

```text
train loss: 7.043 → 6.458 → 5.455 → 4.823
validation loss: 6.051
steady step time: 4.68–4.86 s
peak allocated/reserved: 21.95 / 25.39 GiB/GPU
checkpoint: outputs/semantic_teacher_kl_replicated_smoke_v2/global_step_4
```

当前进入 standalone merge 与 16-task 双解码 gate。

双解码 gate 已完成：

```text
vanilla:
  parseable = 15 / 16
  nonempty/function = 16 / 16
  errors = 0

Scaffold:
  depth_capacity_exhausted = 16 / 16
  nonempty/function = 0 / 16
```

结论：lexical teacher KL 达到了语义保真目标，但结构 CE 即使只训练
4 steps，full merge 仍把 compound topology token 推得过强。当前校准
adapter merge scale `0.125 / 0.25 / 0.5`，先确认是否存在兼顾 vanilla
能力和 Scaffold 终止性的最小结构适配强度。

三个 scale 均为 16/16 `depth_capacity_exhausted`，所以递归与 LoRA merge
强度无关。5,000 个实际训练状态的 line target 审计中 `[STMT]` 明显多于
所有 compound labels，但语义词初始化产生的 compound output-row 范数
通常高于 `[STMT]`。当前固定最弱的 `scale=0.125`，校准仅作用于
`LINE_BODY` 的 `[STMT]` logit bonus `1/2/4/8`；module root 和 lexical
预测保持不变。

`[STMT]` prior 将 failure 从 16/16 单调降至 0/16；bonus 4/8 均产生
16/16 非空且含函数的输出，但仍为 0/16 parse。输出主要是错误或截断的
函数 header，以及缺括号/冒号的简单 statement。下一步固定 bonus 4，
注入 benchmark 已知函数签名，并校准 statement masks `4/8/16`。

Seeded length gate 中 `stmt=4` 达到 8/16 parse、16/16 非空函数且零失败；
`stmt=8/16` 均降到 1/16 parse。说明全局增加 slot 数不是修复方式，会让
nested statements 产生更长的非法拼接；同时所有 runs 的 token-level
`[expand]` 使用次数仍为零。当前只增加 depth≤1 的 shallow statement
长度 `6/8/12`，nested statements 保持 4。

Shallow `6/8/12` 分别只有 `5/2/4` 个 parse，均低于 global stmt=4 的
8/16，因此固定长度路线停止。发现一个训练目标 bug：teacher KL 按 lexical
role 选位置，却没有排除同 role 下的 `[expand]/[delete]` labels，可能把
edit 行强行锚回 pretrained teacher。已改为只对 ordinary lexical target
做 KL。当前启动 64-step leaf-only elastic pilot，保持 topology checkpoint
不变，只训练 token merge/delete 与 lexical infilling。

Leaf-only pilot 已完成，final train loss 2.870、validation loss 3.808，
checkpoint 完整。当前 merge 并运行 16-task matched gate，显式报告
`tasks_with_expansion`、总 expansions、parseability 和 vanilla 保真。

Leaf-elastic gate 中 vanilla 仍为 15/16 parse、零错误，但 Scaffold 仅
7/16 parse，且 16 个任务的 expansions 仍全部为零。训练状态包含大量
expand targets，因此当前只校准 token-level `[expand]` logit bonus
`1/2/4`；若仍无法同时激活 expand 并维持 parse≥8，则结束该 edit-token
长度控制路线。
