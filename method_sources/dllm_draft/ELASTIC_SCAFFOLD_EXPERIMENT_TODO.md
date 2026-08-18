# Elastic Scaffold-Coder 后续实验 TODO

本文档只包含下一阶段论文实验。历史已完成实验继续保留在 `TODO.md`。

状态标记：

```text
[ ] 未开始
[~] 进行中
[x] 完成
[B] 等待外部依赖
```

优先级：

```text
P0：论文主线成立所必需
P1：提高说服力
P2：扩展或后续工作
```

---

# P0：统一协议与 Runtime 可配置化

## [ ] CAP-000 — 冻结实验协议

### 任务

- 固定 checkpoint：
  `outputs/scaffold_sft_stage1/global_step_4465`；
- 固定 EvalPlus：
  HumanEval+ v0.1.10、MBPP+ v0.2.0；
- 固定 prompt 和 extraction；
- 固定 temperature 0；
- 固定随机种子；
- 所有运行保存：
  - `solutions.jsonl`
  - `metrics.jsonl`
  - `eval_results.json`
  - config manifest
  - Git commit
  - checkpoint inventory。

### 验收

- 同配置重复运行 16-task smoke，输出逐任务完全一致；
- manifest 能重建全部参数；
- failure 和 NFE 口径有单元测试。

### 产物

```text
configs/runtime_capacity/protocol_v1.yaml
docs/RUNTIME_CAPACITY_PROTOCOL.md
```

---

## [x] CAP-001 — 将 DecoderConfig 全部暴露到 Eval CLI

### 需要新增的参数

```text
--initial-root-slots
--initial-body-slots
--initial-statement-masks
--initial-function-header-masks
--initial-loop-header-masks
--initial-condition-masks
--max-tree-depth
--max-lines-per-body
--max-total-lines
--max-tokens-per-hole
--max-expansions
--allow-module-expand / --no-module-expand
--max-canvas-tokens
```

### 验收

- 默认参数复现当前 Stage-1 sampler；
- Tiny 参数确实限制深度、行数和 token slots；
- 非法组合在启动时失败；
- config 写入每个 metrics sidecar。

---

## [x] CAP-002 — 实现全局行数和 module-expand 开关

当前 runtime 有 `max_lines_per_body`，但缺少：

- `max_total_lines`；
- 独立的 module-level expand 控制。

### 验收

- 任何树状态的总行数不超过 hard cap；
- module expand off 时 root mask 不提供 `[expand]`；
- body expand 仍可独立启用；
- 增加 unit tests。

---

## [x] CAP-003 — Capacity pressure instrumentation

### 每任务新增指标

```text
line_capacity_hits
token_capacity_hits
depth_capacity_hits
total_line_capacity_hits
module_expand_suppressed
expand_budget_hits
repeated_canvas_count
no_progress_calls
maximum_tree_depth
maximum_total_lines
maximum_body_lines
maximum_tokens_per_hole
capacity_expansions
termination_reason
```

### 验收

- scripted runtime 能分别触发每个计数器；
- 真实 16-task smoke 中指标非空；
- failure 必须有唯一 termination reason。

---

## [x] CAP-004 — 统一 failure-adjusted 成本统计

### 报告

- successful-only mean NFE；
- all-task mean NFE；
- median/P90/P99 NFE；
- rule-only steps；
- cumulative model tokens；
- wall-clock；
- max canvas；
- step-limit failure rate。

### 验收

- 所有 Scaffold generation run 都能转换到同一成本 schema；
- 失败任务不能被从 NFE 均值中静默排除。

---

## [ ] CAP-005 — 修复失败任务的 partial-cost instrumentation

### 任务

Sampler 在异常退出时保存已经实际发生的：

- model forward 次数；
- 每次 model canvas 长度；
- cumulative model tokens；
- 最大 model canvas；
- capacity pressure counters；
- 唯一 termination reason。

禁止再将所有失败任务统一按 `max_model_calls` 计费。

### 验收

- model-call budget failure 的 NFE 等于配置上限；
- depth/line/token early failure 保留真实 partial NFE；
- Tiny/Small 64-task screening 使用新口径重跑；
- summary 同时报告 success-only 与 all-task 成本。

---

# P0：固定容量曲线

## [x] CAP-010 — 构造四个固定容量配置

### Tiny

```text
depth=1
lines/body=2
total lines=16
tokens/hole=2
module expand=off
```

### Small

```text
depth=2
lines/body=4
total lines=32
tokens/hole=8
module expand=on
```

### Medium

```text
depth=4
lines/body=16
total lines=64
tokens/hole=32
module expand=on
```

### Large

```text
当前 Stage-1 默认
```

### 验收

- 四个配置均通过 scripted generation；
- Tiny 是明确、可复现的低容量端点；
- Large 数值与当前正式结果一致。

---

## [x] CAP-011 — HumanEval 64-task 分层筛选集

### 任务

构建固定 64-task screening set：

- 按 canonical depth 分层；
- 按 canonical length 分层；
- 包含短浅和长深任务；
- 不根据模型结果选择。

### 验收

- 固定 task ID；
- oracle 全通过；
- 复杂度分布写入 manifest。

---

## [x] CAP-012 — HumanEval screening capacity sweep

运行：

```text
Tiny / Small / Medium / Large
```

### 主要判断

- failure rate 是否随容量单调下降；
- successful-only NFE 是否随容量上升；
- all-task NFE 是否呈 U 型；
- pass@1 是否存在容量拐点。

### 继续门槛

必须观察到至少一项：

- Tiny vs Large failure 差 ≥ 15 pp；
- pass 差 ≥ 5 pp；
- all-task NFE 差 ≥ 20%；
- capacity hit 与 failure 有显著关联。

若全部没有，暂停 adaptive 主线。

### 实际结果

- Tiny: HE+ 0%, failure 34.38%, all-task NFE 186.23；
- Small: HE+ 3.13%, failure 37.50%, all-task NFE 219.52；
- Medium: HE+ 10.94%, failure 0%, all-task NFE 69.55；
- Large: HE+ 10.94%, failure 10.94%, all-task NFE 111.77。

继续门槛已满足；Medium 同功能分数下支配 Large，并显著优于
Tiny/Small。

---

## [ ] CAP-013 — Full HumanEval+ capacity sweep

只对 screening 后有代表性的 3 个配置运行全 164 题。

### 产物

```text
outputs/runtime_capacity/humaneval/{tiny,selected,large}/
ops/artifacts/runtime_capacity_humaneval.json
```

---

## [ ] CAP-014 — Full MBPP+ capacity sweep

对同样 3 个配置运行 378 题。

### 验收

- 使用完全相同的 runtime config；
- 不能为 MBPP 单独调参；
- 报告复杂度 slice。

---

# P0：Adaptive Capacity

## [ ] ADAPT-001 — 定义 no-progress 状态

候选定义：

```text
连续 p 次模型调用：
unresolved masks 不下降
且
tree/canvas fingerprint 未产生有效新结构
```

### 验收

- cycle 和高 entropy 但仍有进展的状态可区分；
- 不因 rule-only fixed point 误触发。

---

## [ ] ADAPT-002 — 实现局部扩容动作

扩容优先级：

1. 当前 token hole cap ×2；
2. 当前 body line cap ×2；
3. total lines +8/+16；
4. depth +1；
5. on-demand module expand。

### 验收

- 只扩大触发压力的 subtree；
- 不重置已提交的其他 subtree；
- 每次扩容可审计；
- hard cap 始终生效。

---

## [ ] ADAPT-003 — 三种 adaptive policy

### A：Expand-on-block

模型预测 expand 但被 cap 阻止时扩容。

### B：Expand-on-stall

no progress 达到 patience 时扩容。

### C：Expand-on-pressure

综合：

- blocked expand；
- capacity saturation；
- entropy；
- repeated state；
- missing static requirement。

### 筛选

先在 CAP-011 的 64-task set 上运行。

---

## [ ] ADAPT-004 — Adaptive vs Fixed 关键实验

比较：

```text
Fixed-Tiny
Fixed-Large
Adaptive-best
```

### 成功门槛

Adaptive vs Tiny：

- failure 降 ≥ 15 pp，或 pass 提升 ≥ 5 pp。

Adaptive vs Large：

- pass 下降 ≤ 1 pp；
- cumulative tokens 或 wall-clock 下降 ≥ 20%。

满足两个 endpoint 中至少一个 dominance 条件才进入 full benchmark。

---

# P0：终止策略

## [ ] TERM-001 — 统一 termination reason

```text
resolved
model_call_budget
capacity_exhausted
cycle_detected
no_progress
invalid_render
context_limit
```

所有失败任务必须归类。

---

## [ ] TERM-002 — Capacity-before-fail

在 step-limit 前：

```text
检测 stall
→ 尝试一次局部扩容
→ 若仍无进展才终止
```

### 验收

- 不增加超过预设的额外 NFE；
- 能救回部分原 step-limit failure；
- 不显著增加简单任务成本。

---

## [ ] TERM-003 — Graceful forced resolution

规则：

- optional slot 强制 delete；
- 空 required body → `pass`；
- unresolved required header 不能伪造；
- final output 必须记录 forced action 数。

### 风险

可能提高 parseability 但降低功能分数，需独立报告。

---

## [ ] TERM-004 — Static verifier 早停/扩容

检查：

- 入口函数存在；
- 函数名和签名；
- `ast.parse`；
- required return；
- undefined names；
- empty body。

根据结果：

- 满足条件允许结束；
- 缺失结构时扩容；
- 不直接修改模型 token。

---

# P1：复杂度与 failure 分析

## [ ] ANALYSIS-001 — Benchmark complexity metadata

每题记录：

- canonical lines；
- canonical tokens；
- compound depth；
- number of functions；
- loop/if/try/with 数；
- imports；
- helper definitions；
- expected return count。

---

## [ ] ANALYSIS-002 — Capacity hit 预测 failure

训练简单分类/统计模型：

```text
failure ~ capacity hits + depth + length + cycle + entropy
```

报告：

- AUROC；
- odds ratio；
- calibration；
- 分层 failure rate。

---

## [ ] ANALYSIS-003 — NFE 双峰图

分别绘制：

- successful-only；
- failed；
- all tasks。

标出 step-limit 峰，并比较 Tiny/Large/Adaptive。

---

## [ ] ANALYSIS-004 — Pareto 图

至少包含：

- Plus pass@1；
- all-task cumulative tokens；
- wall-clock；
- failure rate；
- parseability。

---

# P0：语义保真训练与超越 Dream-Coder

当前 512-NFE 目标线：

```text
Dream-Coder Instruct
HumanEval+ = 50.00%
MBPP+      = 65.08%
```

旧 Scaffold 为 `18.29% / 32.01%`，因此不能靠微调 runtime 单独跨越。
训练主线改为：**从 Instruct 初始化、参数高效结构适配、保留普通 token
能力，再用执行反馈提升语义。**

## [x] SEMTRAIN-001 — Rung-mixture + LoRA smoke

配置：

```text
init: Dream-Coder Instruct
objective: root 10% / body 20% / leaf 70%
schedule: 无 depth bands
adapter: LoRA rank 16
structural rows: 由 def/for/if 等语义行初始化后冻结
steps: 4
```

### 验收

- 8 卡 FSDP/LoRA 前向、反向和 checkpoint 正常；
- checkpoint 可被 Scaffold runtime 加载；
- Transformer backbone、embedding 和 lm-head 保持冻结；
- LoRA 通过 hidden-state adaptation 学习选择固定的结构 token 行；
- 没有 NaN/OOM。

实际结果（2026-08-05）：

```text
8×H20，4 optimizer steps
train loss: 6.733 → 4.957 → 4.198 → 3.727
val loss: 4.991
post-warmup throughput: 9.3k–10.8k non-padding tokens/s
peak reserved memory: 20.67 GiB/GPU
```

冻结 Transformer backbone、embedding 和 lm-head 的 adapter checkpoint
已成功保存；standalone merge 在加载 base 后重新执行确定性的结构 token
初始化，再合并 LoRA。

## [x] SEMTRAIN-002 — 一轮语义保真 Scaffold SFT

```text
init: Dream-Coder Instruct
epochs: 1
LoRA rank: 16
LR: 5e-5（只更新 LoRA adapter）
root/body/leaf: 0.10 / 0.20 / 0.70
```

与旧方案的关键差异：

- 不再从 Base 训练五轮；
- 不再使用已被实验证伪的 depth-banded schedule；
- 70% 训练质量用于 lexical/header/statement infilling；
- backbone 冻结，减少 catastrophic forgetting。

实际训练结果（2026-08-05）：

```text
steps: 893 / 893
final validation loss: 0.581
last-100 train loss mean: 约 0.573
stable profiled step time: 约 1.50 s
checkpoint: outputs/semantic_scaffold_lora_1ep/global_step_893
merged: outputs/semantic_scaffold_1ep_merged
```

## [~] SEMTRAIN-003 — 双解码门控

同一 checkpoint 同时跑：

1. vanilla Dream-Coder decode；
2. Scaffold Medium/Adaptive decode。

HumanEval 先作训练选择，MBPP 只作最终确认，避免反复调 benchmark。

### 第一阶段继续门槛

- vanilla HE+ ≥ 45%，证明语义能力基本保留；
- Scaffold HE+ ≥ 30%，且 generation failure ≤ 5%；
- 若 vanilla HE+ < 40%，立即停止该配方并降低结构比例/adapter rank。

### 协议修正

第一版门控得到：

```text
vanilla HE+          25.61%
Scaffold Medium HE+  1.22%
```

但输出归因发现：

```text
vanilla:  76 / 164 含 literal expand/delete/mask
Scaffold: 154 / 164 含 literal expand/delete/mask
```

原因是冻结的 `[expand]`/`[delete]` 输出行由普通
`expand/mask/delete/remove` 行初始化；解码时普通 source rows 与规则 token
发生概率竞争。因此该版结果标记为 **protocol-invalid**，不能直接归因为
semantic forgetting。

修正：

- vanilla 同时屏蔽全部 Scaffold token 和 edit-source ordinary rows；
- Scaffold lexical holes 禁止 edit-source ordinary rows，只允许真正的
  `[expand]`/`[delete]` 规则动作。

16-task 复核：

```text
                 parseable   polluted   generation errors
vanilla          13 / 16     1 / 16     0
Scaffold Medium  12 / 16     0 / 16     0
```

复核通过，现正按原冻结阈值重跑 full HumanEval+；MBPP 仍等待该结果。

Suppressed full vanilla 最终结果：

```text
HumanEval+ = 35.98%
parse      = 73.78%
pollution  = 5 / 164
```

低于 40% 止损线，因此：

- 阻断 scale=1.0 的 Scaffold Medium full run；
- 不运行 MBPP；
- 先在固定 64-task screening set 上做 LoRA merge scale
  `0.25 / 0.50 / 0.75` 校准；
- 只有 vanilla preservation 恢复后再重开结构 full gate。

Scale calibration 已完成：

```text
scale 0.25: HE+ 59.38%, parse 100.00%, errors 0
scale 0.50: HE+ 46.88%, parse  92.19%, errors 0
scale 0.75: HE+ 43.75%, parse  87.50%, errors 0
```

按冻结规则选择 `scale=0.25`。当前执行依赖链：

```text
selected scale=0.25
→ full vanilla HumanEval+（运行中）
→ vanilla HE+ ≥45% gate
→ full Scaffold Medium
→ combined gate
```

任何门槛失败都会阻断后续；MBPP 仍未运行。

Selected scale full gate 已完成：

```text
vanilla full HE+   = 56.10%
vanilla parse      = 95.73%
vanilla errors     = 0

Scaffold full HE+  = 0.00%
Scaffold failures = 142 / 164
```

Scaffold failure 中 138 个为 `depth_capacity_exhausted`，4 个为
`total_line_capacity_exhausted`；不存在 model-call 或 expand-cycle 长尾。
因此不进入 MBPP，先运行 `LINE_BODY` compound-construct logit penalty
`0/1/2/4` 的 64-task 校准。只有 failure ≤5%、parse ≥90% 的 arm 才继续。

Penalty calibration 结果：

```text
penalty 0: HE+ 0%, parse 89.06%, failures 57/64
penalty 1: HE+ 0%, parse 67.19%, failures 42/64
penalty 2: HE+ 0%, parse 29.69%, failures 18/64
penalty 4: HE+ 0%, parse  6.25%, failures  2/64
```

结论：penalty 只能压制递归，无法修复 lexical/header 截断。所有 arm 的
token hole 最大长度均为 4，说明 checkpoint 几乎不使用 token-level
`[expand]`。当前固定 `penalty=4` 和正确函数签名，校准 statement masks
`4/8/16`；仍不运行 MBPP。

Seeded length calibration 未达到继续门槛：

```text
stmt 4:  HE+ 1.56%, parse 39.06%
stmt 8:  HE+ 3.13%, parse 31.25%
stmt 16: HE+ 0.00%, parse  9.38%
```

随后进行 19 个结构 token row-only smoke。普通词表 bit-exact 保持不变，
但 4-step、LR 1e-3 的模型在 module root 上 16/16 预测 `[delete]`，全部
输出空程序。该 arm 已停止；下一步需要：

- 从 trainable token 集中排除 `[delete]`，或对 root delete 单独冻结；
- 降低 LR/steps；
- smoke gate 强制 nonempty + required function，禁止空 AST 假阳性。

最终 row-only 实现改用 PEFT 0.19 compact `TrainableTokens`，参数量
136,192，adapter 约 545 KB，合并后普通词表 bit-exact 不变。训练分布中
`[delete]` 占 root targets 28.6%，解释了高 LR 4-step 后的全删行为。

下一项冻结配置改为只训练 topology labels：

```text
trainable rows: FUNC/FOR/WHILE/IF/ELIF/ELSE/STMT
expand/delete targets: disabled
root/body/leaf: 0.5 / 0.5 / 0
LR: 1e-4
steps: 4
```

继续门槛：

- ordinary rows bit-exact；
- nonempty 16/16；
- required function 16/16；
- parse ≥8/16；
- generation failure ≤8/16。

Topology-only 4-step training 已完成：

```text
effective parameters = 50,176
train loss = 4.096 → 2.689 → 2.157 → 1.635
val loss = 1.716
ordinary rows stored = 0
```

CPU merge 后普通 input/output rows 保持 bit-exact，7 个 topology rows 的
最大变化约为 `4.88e-4`。但 16-task decode gate 未通过：

```text
nonempty          = 10 / 16
required function = 10 / 16
parseable         =  6 / 16
generation errors =  6 / 16
error type        = depth_capacity_exhausted × 6
```

与未排除 edit labels 的 19-row adapter 相比，删除塌缩已消失，但 topology
过预测仍会造成递归；成功终止样本也主要失败在 header/statement lexical
截断。该 4-step 配方不进入 full epoch。下一步不再单独增加 topology-row
CE，而进入 `SEMTRAIN-004`：用 frozen teacher 对普通/lexical logits 做 KL
锚定，同时只对结构位置做 CE；先做 4-step smoke 与双解码 gate。

## [~] SEMTRAIN-004 — Replay/KL 语义锚定

若 SEMTRAIN-002 的 vanilla decode 仍明显下降，则增加：

- teacher = frozen Dream-Coder Instruct；
- 在普通 masked states 上做 teacher KL/logit distillation；
- structural states 继续做 ground-truth CE；
- 或混入 teacher-generated、执行通过的 replay solutions。

目标是把普通 token 分布锚定在原 Instruct checkpoint 附近，而不是继续
增加 educational_instruct 的重复 epoch。

首个最小实现采用 frozen Dream-Coder Instruct teacher：

```text
student: LoRA rank 8
root/body/leaf: 0.15 / 0.25 / 0.60
objective: structural/lexical CE + lexical-role KL
KL roles: TOKEN_STMT / TOKEN_HDR / TOKEN_DOC
KL weight / temperature: 1.0 / 1.0
teacher-support top-k: 256
teacher placement: replicated BF16 per GPU
LR: 2e-5
steps: 4
```

Teacher 与 student 使用相同 corrupted canvas 和确定性结构 token
初始化；teacher 全冻结。第一版将 student 与 teacher 包成两棵独立 FSDP
树，结果 8 卡连续约 15 分钟停留在两棵树交替 unshard 的 rate limiter，
第一个 optimizer step 未结束，判定该拓扑不可用。修正版利用 H20 的显存
余量，把 BF16 teacher 完整常驻每卡，仅 student 使用 FSDP，消除 teacher
all-gather。KL 只在 teacher 的 top-256 support 上计算，避免对
`[B,S,152k]` 全词表概率张量保留额外 autograd 图。修正版 4-step smoke
已经通过：

```text
train loss: 7.043 → 6.458 → 5.455 → 4.823
val loss: 6.051
steady step: 4.68–4.86 s
peak reserved: 25.39 GiB/GPU
```

现在 merge 该 adapter，并运行 16-task vanilla + Scaffold 双解码 gate。

双解码 gate 结果：

```text
vanilla: 15/16 parseable, 16/16 nonempty+function, 0 errors
Scaffold: 0/16 nonempty, 16/16 depth_capacity_exhausted
```

因此 teacher-KL 成功保住普通代码生成，但 4-step、full-scale adapter 对
topology logits 仍然过强。下一步不进入完整训练；固定该 checkpoint，在
同一 16-task set 上校准 merge scale `0.125 / 0.25 / 0.5`。只有
failure≤4、nonempty/function≥12、parse≥8 的 scale 才允许继续；若全部
失败，则转为降低 root/body CE 权重或对 `[STMT]` 使用类别平衡。

Scale calibration 已完成，`0.125 / 0.25 / 0.5` 全部为：

```text
depth_capacity_exhausted = 16 / 16
nonempty/function = 0 / 16
```

因此递归不是 LoRA delta 强度造成的。实际 rung 分布审计（5,000 states）
显示 line-level 目标：

```text
[STMT] 1715
[FUNC]  681
[IF]    259
[FOR]   191
[WHILE]  48
```

数据中 `[STMT]` 本应占主体，但冻结的结构输出行来自语义词平均初始化；
`[IF]/[FOR]/[WHILE]` 行的范数普遍高于 `[STMT]`，容易在 restricted
line vocabulary 中系统性胜出。下一步固定 `scale=0.125`，只在
`LINE_BODY` 增加 `[STMT]` logit prior `+1/+2/+4/+8`，用同一 16-task
gate 找到最小可终止强度；该校准不影响 module-level `[FUNC]`。

`[STMT]` prior 结果：

```text
bonus 1: errors 13, nonempty/function 3,  parse 13
bonus 2: errors  3, nonempty/function 13, parse  3
bonus 4: errors  0, nonempty/function 16, parse  0
bonus 8: errors  0, nonempty/function 16, parse  0
```

这证明 `[STMT]` prior 已解决 topology 递归，但成功终止输出仍因函数
header 和简单 statement 截断而不可解析。当前固定 `bonus=4`，使用正确
HumanEval 函数签名并校准 statement 初始长度 `4/8/16`，区分 header 错误
与 leaf length/expand 学习不足。

Seeded length 结果：

```text
stmt 4:  errors 0, nonempty/function 16, parse 8
stmt 8:  errors 0, nonempty/function 16, parse 1
stmt 16: errors 0, nonempty/function 16, parse 1
```

全局增加 slot 长度会把 nested placeholder 拼成更长的非法单行代码，反而
降低 parse。`stmt=4` 已达到最低 gate，但 8 个失败样本主要是顶层函数体
statement 被截断；所有 runs 的 model `expansions=0`。下一步仅增加
depth≤1 的 shallow statement masks `6/8/12`，nested statements 仍固定 4，
测试“浅层长度不足”而不破坏嵌套结构。

Shallow-only 结果仍未超过 baseline：

```text
global stmt 4: parse 8 / 16
shallow 6:     parse 5 / 16
shallow 8:     parse 2 / 16
shallow 12:    parse 4 / 16
```

固定 slot 长度路线到此止损。进一步审计发现 teacher KL 原先按
`TOKEN_STMT/HDR/DOC` role 选择位置，但 token-level `[expand]/[delete]`
目标也使用这些 roles；这会要求 pretrained teacher 在新 edit targets 上
匹配分布，可能直接压制弹性动作，和所有运行 `expansions=0` 一致。

下一步修正 KL mask：只锚定 ordinary lexical targets，排除全部 Scaffold
special IDs。随后从保语义的 `scale=0.125` checkpoint 启动 64-step
leaf-only elastic pilot：

```text
root/body/leaf = 0 / 0 / 1
token merge probability = 0.5
max token delete = 1
LR = 1e-5
teacher KL = lexical-only
```

该 pilot 只学习 statement/header 长度控制，不再更新 topology 选择。

64-step pilot 已完成：

```text
final train loss = 2.870
validation loss = 3.808
steady step = 4.0–4.5 s
peak reserved = 29.74 GiB/GPU
```

当前将 leaf adapter 合并回 `scale=0.125` Scaffold checkpoint，并运行
matched 16-task gate：

- vanilla parse/error 保真；
- seeded Scaffold `stmt=4 + STMT bonus=4`；
- parse 是否超过当前 8/16；
- token-level `[expand]` 是否从 0 激活。

Leaf-elastic gate 结果：

```text
vanilla: parse 15/16, errors 0
Scaffold: parse 7/16, errors 0, expansions 0
```

语义保真继续成立，但 64-step leaf-only 训练仍未让 greedy decode 选择
`[expand]`。训练分布审计确认 5,000 states 中含 17,883 个 expand targets，
因此不是缺少监督，而是 expand logit 仍低于普通 lexical token。

下一步做最后一个 inference calibration：仅在 token-level masks 上增加
`[expand]` logit bonus `1/2/4`，line-level expand 不受影响。只有
`tasks_with_expansion>0`、parse≥8 且 errors≤4 的 arm 才保留；若失败，
停止当前 expand-token 设计并转向显式 length-head/length-token。

## [ ] SEMTRAIN-005 — Diffusion-native execution RL

仅在 SFT 达到上述门槛后进入：

- reward：EvalPlus-style hidden unit tests、语法、终止；
- 优先采用 coupled-GRPO / VRPO 类低方差 diffusion policy optimization；
- 训练数据使用独立可执行题库，严禁用 HumanEval+/MBPP+ 测试任务训练；
- 同时保留 KL 到 SEMTRAIN-002，防止结构 token 或代码风格漂移。

### 最终成功标准

必须在一次冻结协议的 full run 中同时满足：

```text
HumanEval+ > 50.00%
MBPP+      > 65.08%
```

并至少满足一项效率优势：

- all-task mean NFE 下降 ≥ 25%；或
- cumulative model tokens 下降 ≥ 25%；或
- parseability/failure 显著优于 Dream-Coder。

---

# P1：Runtime 单组件消融

## [ ] ABLATE-001 — Module expand

```text
off vs on
```

## [ ] ABLATE-002 — Depth cap

```text
1 / 2 / 4 / 8 / 16
```

## [ ] ABLATE-003 — Line capacity

```text
2 / 4 / 8 / 16 / 128
```

## [ ] ABLATE-004 — Token capacity

```text
2 / 4 / 8 / 32 / 512
```

## [ ] ABLATE-005 — Cycle suppression

```text
off vs repeated-state suppression
```

## [ ] ABLATE-006 — Vocabulary legality

```text
unconstrained vs typed legal support
```

每个单组件消融先使用 64-task screening set，只有差异显著才跑 full。

---

# P2：结构化修订

## [ ] REV-001 — Confidence distribution instrumentation

当前固定阈值没有触发 action。

记录：

- token confidence；
- line/meta confidence；
- subtree mean/min confidence；
- correct vs incorrect 分布；
- AUROC。

---

## [ ] REV-002 — Quantile-triggered C1/C2/C3

不再使用绝对阈值，改用：

```text
每样本最低 5% / 10% subtree
```

必须确保 action rate 非零。

---

## [ ] REV-003 — Verifier-guided subtree localization

从：

- traceback；
- failing assertion；
- undefined name；
- missing return；
- parse failure；

定位最小 subtree。

---

## [ ] REV-004 — Revision baseline

比较：

- full restart；
- random span remask；
- lowest-confidence token remask；
- subtree collapse；
- verifier-localized subtree collapse。

### 继续门槛

- 新增 pass ≥ 5/64；
- regression ≤ 2/64；
- extra tokens ≤ 35%。

---

# P2：扩展 Benchmark

## [~] BASEPROTO-001 — 对齐 Dream-Coder Base 官方评测协议

旧 Base 结果 `HE+=7.9% / MBPP+=15.9%` 不能作为复现结论，因为 harness
仍加入 instruction wrapper，并只评分生成 suffix。官方配置为：

```text
raw benchmark function-prefix prompt
add_bos_token = true
solution = prompt + generated continuation
HE: temperature 0.2, top_p 0.9
MBPP: temperature 0.1, top_p 0.9
steps/max_new_tokens = 512
```

先在空闲的 .104 运行 HumanEval 前 16 题 smoke；验收：

- generation errors = 0；
- parseable ≥8/16；
- raw prefix + BOS manifest 正确。

通过后再注册 full HE+/MBPP+，用于清除报告中唯一尚未对齐的模型基线。

## [ ] BENCH-001 — BigCodeBench 结构复杂度 slice

先构建：

- 深层；
- 长程序；
- 多 helper；
- try/with/class。

## [ ] BENCH-002 — LiveCodeBench post-cutoff slice

需明确窗口和 prompt 长度。

## [ ] BENCH-003 — Code repair benchmark

优先选择：

- 有 failing tests；
- 可定位错误行；
- 可评估局部 repair。

---

# 报告与论文产物

## [ ] PAPER-001 — Capacity protocol

```text
docs/RUNTIME_CAPACITY_PROTOCOL.md
```

## [ ] PAPER-002 — 主结果表

```text
ops/artifacts/runtime_capacity_main_table.json
paper/tables/runtime_capacity.tex
```

## [ ] PAPER-003 — Pareto 图

```text
paper/figures/capacity_pareto.pdf
```

## [ ] PAPER-004 — Failure-tail 图

```text
paper/figures/nfe_failure_tail.pdf
```

## [ ] PAPER-005 — Complexity 分析图

```text
paper/figures/capacity_by_complexity.pdf
```

## [ ] PAPER-006 — Claim audit

逐条核对：

- 是否同 checkpoint；
- 是否同任务；
- 是否 matched compute；
- 是否包含 failure；
- 是否有 paired statistics；
- 是否能从 artifact 重建。

---

# 建议执行顺序

```text
CAP-000
→ CAP-001/002/003/004/005
→ CAP-010/011/012
→ CAP-013/014
→ ADAPT-001/002/003
→ ADAPT-004
→ TERM-001/002/003/004
→ ANALYSIS-001/002/003/004
→ CKPT-001
→ full paper matrix
→ revision（若主线通过）
```

# 当前基础设施依赖

```text
服务器 GPU：执行前需确认没有未知外部任务占用
```

# 第一阶段总止损点

完成 `CAP-012` 后决策：

- 若容量与 failure/NFE/quality 无显著关系：停止该主线；
- 若只有固定 Large 最好且 Adaptive 无法节省 ≥20% 成本：降级为分析型
  论文；
- 若 Adaptive 改善 failure 或 Pareto frontier：进入 full benchmark。
