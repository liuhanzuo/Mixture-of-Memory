# A01 — What Survives Null Calibration?

## 状态

**ACTIVE，当前最成熟的独立 proposal。**

## 一句话主张

一个评测量在被解释为“能力”之前，必须与该 construct 自己的
**input-blind null** 比较，而不是与泛化的 chance line 或一个过弱 baseline 比较。
我们统一报告：

```text
reported value / construct-appropriate null / calibrated residual /
residual fraction
```

并展示这个协议不仅改变误差条，还会推翻我们自己的 headline。

## 已完成的四类 construct

| Construct | Reported | Null | Residual fraction |
|---|---:|---:|---:|
| MC content scoring | 0.3598 | longest-option 0.2845 | 20.9% |
| Generative majority prior | 0.6590 | constant refusal 0.4985 | 24.4% |
| Representation similarity | 0.4907 | layer-order shuffle 0.4529 | 7.69% |
| Probe readout depth | 0.6610 | native readout 0.1505 | 77.2% |

稳妥表述是：残余比例约 **8%–77%**；不要把“恰好超过 10×”作为 headline，
因为 C4 aggregation 的合理变体会得到约 7–10×。

## 最强案例：自我证伪

MMLU 两个 scoring interface 确实产生了两个显著、且通过 BH 的 ranking flip；
但参与 flip 的三个 arm 全部处于或低于 letter interface 的 best-constant floor
`always-D = 0.2689`。限制到两个 interface 都有效的四个 arm 后：

- 6 个 pair；
- 0 个 sign flip；
- 0 个显著 flip。

因此原 headline 被撤回。关键是 `.2689` 的 construct-appropriate floor，
而不是 `.25` chance line；用 `.25` 会错误地认为 keep10 有效。

## Representation leg

- observed mean midband z-CKA：`0.490672`
- 2000-permutation layer-order null：`0.452936`
- calibrated residual：`0.037737`
- residual / reported：`7.69%`
- BH `q=.05`：约 `50–52/91` pair across seeds

random-init `0.0912` 是错误 null；使用它会把可用 correspondence signal
夸大约 10.6×。

## 新颖性边界

不能主张：

- 首创 permutation null calibration；
- 首创 BH；
- 表征相似性文献没有 null。

可以主张：

1. 跨多个无关 construct 的统一 null-calibrated reporting；
2. 针对 layer correspondence 问题的 **layer-order null**；
3. 把该协议先用于撤回自己的结果；
4. 给出 calibrated residual fraction，而非只给显著性。

## 下一步 gate

### 必做

1. 第三个模型家族的 MC interface case。
2. 非 MMLU 的一个 MC benchmark。
3. OLMo full-fp32 forward：检验 bf16 exact tie 是否为因果机制。
4. C4 aggregation 预注册，不再选择性报告 10×。

### 成功条件

- 至少三个 construct 的 null calibration 改变科学结论，而非仅缩小数字；
- 第三模型/第二 benchmark 保持“instrument validity before comparison”结论；
- 与已有 similarity-null prior art 的边界经正式 venue/全文核实。

### Kill 条件

- 除 representation 外，其他 construct 的结论在严格 null 下都不改变；
- 第三家族和第二 benchmark 均不复现 interface failure；
- 论文只能退化为已有 similarity-null 方法的案例集合。

## 不得复活的旧数字

- `4.8×`：应为相对 `.25` 的 `4.69×`；相对 content 自己 floor 为 `3.22×`。
- longest-option `.2822`：canonical 为 split-tie `.2845`。
- `58/91 significant`：未做 BH；canonical 为约 `50–52/91`。

