# A01 — What Survives Null Calibration?

## 状态

**ACTIVE，但处于 MAJOR REVISION。** 所有 gate 已跑完、无一 gate 杀死 A01；
第三条 kill clause（novelty）已核查、未触发。**但这不等于"claims 已验证"** ——
2026-08-10 一份外部怀疑论审计
（`../A03-parametric-vs-external-memory/evidence/TCODEX_AUDIT_20260810.md` §2.1 + §7）
对 A01 给出 **Major revision**，其中三条被点名的 claim 经本仓复核后
**两条撤回、一条降级**。逐条回应 = **`TCODEX_AUDIT_RESPONSE.md`**（含实测数字），
复现脚本 = `code/a01_audit_response_recompute.py`，输出 =
`evidence/a01_audit_response_recompute.json`。
`STATUS.json:status` 已从 `active_all_gates_passed_novelty_clear` 改为
`active_MAJOR_REVISION_IN_PROGRESS_after_external_audit`。

当前范围 = `STATUS.json:claim_scope_after_gates` 的 `confirmed_general`
**减去 2026-08-10 的撤回项**（见 `claim_scope_after_gates.RETRACTED_20260810_*`
与 `NARROWED_20260810_*`）。
本文件的 claims 部分已于 2026-08-09 重写以对齐该范围
（此前它仍停留在一个**已被撤回的收窄范围**上），并于 2026-08-10 按审计回应再次修订。


## 一句话主张

一个评测量在被解释为“能力”之前，必须与该 construct 自己的
**input-blind null** 比较，而不是与泛化的 chance line 或一个过弱 baseline 比较。
我们统一报告：

```text
reported value / construct-appropriate null / null 的 convention /
calibrated residual / residual fraction
```

并展示这个协议不仅改变误差条，还会**推翻我们自己的 headline**——包括推翻我们
自己对这个 headline 的第一次撤回。

> ⚠️ **`null 的 convention` 是 2026-08-09 新增的必报字段。**
> 见下方 §「协议本身也需要被校准」。这不是格式要求，是一个实测结论。
> **⚠️ 2026-08-10 降级**：原文写"同一个 null 的 tie convention 单独就能把 6 个 arm
> 里 5 个的判定从 above 翻成 significantly BELOW"——**那 5/6 翻转来自
> `credit`（oracle 上界），不是任何可执行的 tie policy**。三个可执行 convention
> （split/first/last）下 null 只摆动 **0.3365 pp**、**0/6 判定改变**。
> 必报字段本身保留（tie policy **和** tokenizer 都要打印），但理由改成：
> policy 未申明时 null 是**区间**；且 **tokenizer** 差异（0.9003 pp，2.68× 于
> 可执行 convention 跨度）会**真的翻掉一个臂的判定**。见
> `TCODEX_AUDIT_RESPONSE.md` §3/§6。

---

## Claims — 当前范围（2026-08-09 重写）

### 撤回史（必须保留，不得静默删除）

A01 经历了**一次撤回 + 一次对该撤回的反撤回 + 一轮外部审计触发的两撤一降**。
读者必须能看到claim过什么、撤回了什么、为什么。这不是难堪，这是本文的方法论论点本身。

| # | 时间 | 事件 | 依据 |
|---|---|---|---|
| 1 | 2026-08-06 | **原 headline（Obs4）：MC scoring interface 会翻转 model ranking。** 45 arm-pair 里 7 个 sign-opposite、2 个两口径都显著且过 BH。 | `evidence/C5_self_falsification.md` §1 |
| 2 | 2026-08-06 | **自我撤回 #1（成立，保留）**：3/3 参与 flip 的 arm 在 letter 口径上都**处于或低于**自己的 best-constant floor。限制到两口径都显著高于 floor 的 4 个 arm → 6 pair、**0 sign flip、0 显著 flip**。flip 真实存在，但完全发生在仪器已失效的区域。 | `evidence/C5_self_falsification.md` §4 |
| 3 | 2026-08-09 | **撤回 #2（❌ 本身已被撤回）**：`GATE1_VERDICT.md` §1 曾判 `KILL_CONDITION_CLAUSE_2_TRIGGERED`，结论是“把 A01 收窄为 *letter interface 只在结构损伤的 OLMo-2 上退化*”，并要求 drop “letter MC interface 一般而言是不可靠仪器”这一 claim。 | `GATE1_VERDICT.md` §1（原文保留，标注 RETRACTED） |
| 4 | 2026-08-09 | **反撤回（当前有效）**：撤回 #2 测的是**错误的条件**。A01 的 kill clause 讲的是**受损**模型；`GATE1_VERDICT.md` 测的是**完好** base，完好 base 从未被预期出现该病理，因此既不能触发也不能解除该 clause。受损臂实验（**6/6** 非 OLMo 受损臂在自己的 floor 之下）**确认**了一般性 claim。 | `GATE1_DAMAGED_VERDICT.md`、`STATUS.json:claim_scope_after_gates.RETRACTED_must_narrow` |
| **5** | **2026-08-10** | **撤回 #3（外部审计触发）："letter 是 family-general 的 step function / sharp phase transition"。** Llama-2 的 gap-fill 早已跑完却被记成 "in flight"，补齐后全 15 深度网格上 letter 有 **6 个 BH 显著下降**、**5 次 BH 显著方向反转**、floor 判定**穿越 4 次**（k22 从连续三个 above-floor 掉回 `0.230238`，p=9.1e−26）。**per-family 单层大跳保留**（+26.7/+30.3/+48.0 pp），"step function"/"family-general" 撤回。同时更正 `GATE1_DEPTHCURVE_VERDICT.md` 的 "Llama-2 content strictly monotone"（实际两次下降，均在噪声内）。 | `TCODEX_AUDIT_RESPONSE.md` §1–§2、`evidence/a01_audit_response_recompute.json`、`GATE1_DEPTHCURVE_VERDICT.md`（已加 banner） |
| **6** | **2026-08-10** | **降级 #1（外部审计触发）："五种同等 defensible 的 tie convention 翻 5/6 臂"。** `credit` 是 **oracle 上界**（要求知道 gold），`wrong` 是**悲观下界**，两者都不是可执行的 input-blind policy。三个可执行 policy（split/first/last）下 null 只摆 **0.3365 pp**、**0/6 判定改变**、per-arm residual fraction 比值 **1.018×–1.058×**。改写为"三个可执行 convention + 两个 bound"；**tokenizer 那一支（审计未攻击）升为更强的一支**（0.9003 pp = 2.68×，且 63 臂里 robust 翻 1 个）。 | `TCODEX_AUDIT_RESPONSE.md` §3/§6、`GATE3_CONVENTIONS_VERDICT.md`（已加 banner）、`evidence/gate3_content_null_conventions.csv` |

**从 #5/#6 学到的、要写进论文的第二条教训**：一条 claim 若靠"反证它的 run 还没落地"
站着，就必须在那个 run 落地的当天重新判定 —— 而不是让台账替它续命一天。
以及：**在把一个自由度称为"N 种同等合理的选择"之前，先逐个问"这个选择我们的 baseline
真的执行得出来吗"**；执行不出来的那些是 bound，不是 convention。

**从 #3→#4 学到的、要写进论文的教训**：判定一条 kill clause 之前，先确认
自己测的是不是该 clause 说的那个条件。撤回 #2 是在 intact base 上做的，
而 clause 说的是 damaged——这是 A01 自己的协议（“先确认仪器在该条件下有效”）
在 meta 层面的又一次应验。

### ✅ CONFIRMED — 可以作为一般性结论主张

来源：`STATUS.json:claim_scope_after_gates.confirmed_general`（六条），逐条附证据。

1. **结构损伤下，letter MC interface 退化到自己 best-constant floor 之上够不到、
   甚至更低。** 四个家族：OLMo-2-7B（healed keep8，letter `0.2550`）、
   Llama-2-7B（截断，`0.2295`–`0.2415`）、Llama-3-8B（截断，`0.2329`–`0.2527`）、
   Qwen3-8B-Base（截断，`0.2286`–`0.2301`）。floor = always-D `0.2689`，n=14042/臂。
   → `GATE1_DAMAGED_VERDICT.md`

2. ~~**这个塌陷是深度上的 SHARP PHASE TRANSITION，不是渐进衰减。**~~
   **⚠️ 2026-08-10 撤回 —— 见 `TCODEX_AUDIT_RESPONSE.md` §1。**
   **现在只能主张（per-family 描述性）**：四个家族里有**三个**在**单一层**上出现
   letter 跳变，其幅度远大于同一层 content 的任何变化 ——
   Qwen3-8B（36L）在 **k24→k25 一层之内跳 `+48.02` pp**
   （`0.229739` → `0.709942`），而同一批 forward pass 上 content_norm 只动
   **`+1.35` pp**（`0.304515` → `0.317975`）；Llama-3-8B 在 k17→k18 跳
   `+30.34` pp；OLMo-2-7B 在 k18→k19 跳 `+26.68` pp。
   Qwen3 在 k=4…24 上 letter 钉在 `0.229454`–`0.230309`（跨度 `0.0855` pp）。
   **不得**再写成 "step function" / "sharp phase transition" / "family-general"：

   > **❌ 撤回的部分（2026-08-10，外部审计 §2.1 触发）**
   > `STATUS.json:gate1_depth_curve.llama2_anomaly` 曾写 Llama-2 的 gap-fill
   > "in flight on .21" —— **它早已跑完**，五个臂
   > `gate1_dmg_llama2_7b_depth_gap2_k{8,12,18,22,26}` 就在 wzc1 上未被报告，
   > 而被撤回的 claim 却靠 "反证的 run 还没落地" 站着。
   > 补齐后的 Llama-2 全 15 深度网格（n=14042/臂，0 nan，8/8 shard 已断言）：
   > **14 个相邻步里有 6 个是下降，且 6 个全部 BH 显著**（α=0.05）；
   > **BH 显著的方向反转 5 次**（raw 7 次）；floor 判定**穿越 floor 4 次**：
   > BELOW×6(k4..k14) → above×3(k16,k18,k20) → **BELOW**(k22 `0.230238`，
   > −3.867pp，p=9.1e−26) → **AT**(k24 `0.272255`，+0.335pp，p=0.371) →
   > above×4(k26..k31)。k22 在连续三个 above-floor 深度之后**整个掉回 floor 之下**,
   > 这就是杀死 "family-general step" 的事实。
   > 三个"干净"家族本身也不单调：raw letter 反转数 Qwen3 **13**、OLMo-2 **11**、
   > Llama-3 **9**、Llama-2 **7**；最大单步 letter 下降 Qwen3 **−7.49** pp、
   > Llama-2 **−7.52** pp。跳变远大于抖动，但抖动不是零。
   > **仍然成立的**：本 claim 的*用途*不需要 step —— 低于 transition 的臂给出的是
   > floor 值而不是测量值，所以混合 sub-/supra-transition rung 的 damage-scaling
   > 回归不是在估一个量。Llama-2 反而**加强**这一点：它的 floor 判定在深度上非单调，
   > 所以连"按可测性给 rung 排序"都做不到。

   → `GATE1_DEPTHCURVE_VERDICT.md`（已加撤回 banner）、
     `TCODEX_AUDIT_RESPONSE.md` §1–§2、
     `evidence/a01_audit_response_recompute.json`；数值复核自
     `olmo2_mmlu_content_results/gate1_dmg_llama2_7b*_k*/per_example_mmlu_shard*of8.jsonl`（wzc1）
     与 `gate1_dmg_{qwen3_8b,llama3_8b,olmo2_7b}_depth*_k*/summary.json`（zwfy6）

   > **⚠️ 已修正的内部矛盾（2026-08-09）**：`STATUS.json` 曾在两处给出不同的
   > Qwen3 transition：`confirmed_general[1]` 写 “keep30 跳 +45.74pp”，
   > `gate_results.gate1_depth_curve` 写 “k24→k25 跳 +48.0pp”。
   > **正确的是 k24→k25 / +48.02pp。** `+45.74pp` 是 **k30 相对 floor 0.2689 的
   > 残差**（`0.726321 − 0.268908 = +45.7413` pp），不是任何一层的跳变，也不是
   > transition 点——k25 已经在 plateau 上（`0.709942`，相对 floor `+44.10` pp）。
   > `STATUS.json` 已改。

3. **换 interface 救不了受损臂。** 每个受损臂的 content_norm 都落在自己 letter
   accuracy 的 ±3pp 内，且两者都贴着各自的 floor。
   → `GATE1_DAMAGED_VERDICT.md` §4

4. **任何 construct 在跨 arm 比较之前，必须先对齐一个 construct-appropriate null
   （best-constant，而不是 chance）。** 三个可验证的演示：
   * keep8 letter `0.2550` 相对 `.25` chance 读起来“高于随机”，相对 floor
     `0.2689` 则是**低于 floor**（fp32 下 `−1.538` pp，boot p `0.0062`）；
   * BoolQ raw 口径上 keep12 `0.6101` / keep10 `0.6086` / keep8 `0.5948` 相对
     `.50` 读起来“及格”，相对 always-B floor `0.6217` 则在其**之下**；keep14
     `0.6382` 虽 `+0.0165` 但 McNemar `p = 0.20` **不显著** → **6 臂里 4 臂
     无法与一个常数预测器区分**；
   * OBQA 的 residual fraction 在 chance 线下被夸大 **2.15×**（`45.9%` → `21.3%`），
     且 keep10 的 `acc_norm 0.3560` 低于 longest-option floor `0.3635`。
   → `status/scout_21/lane2_a01_gate2.md`（仅 wzc1；zwfy6 无此文件）、
     `STATUS.json:gate2_second_mc_benchmark`、`GATE3_VERDICT.md`

5. **wrong-null 问题不是 MMLU 的产物。** gate-2 在非 MMLU 的 MC benchmark 上复现。
   其中 winogrande 结构性退化（两个选项共享 continuation → `acc == acc_norm`
   恒等、tie rate 100%），**作为 negative control 报告，不作为证据**。
   → `STATUS.json:gate2_second_mc_benchmark`

6. **self-falsification narrative**：用自己的协议撤回自己的 headline——
   现在还包括撤回自己对该 headline 的第一次撤回（见上表 #3/#4），
   以及 `GATE1_HEALEDARMS_VERDICT.md` §7 那次“我在没测 intact base 的
   content_norm 的情况下写了一条关于它的 claim，测完发现符号相反”的更正。

### ⚠️ FAMILY-SPECIFIC — 不得作为一般性结论主张

来源：`STATUS.json:claim_scope_after_gates.family_specific_not_general`。

1. **塌陷的 MECHANISM（机制）是 family-specific 的，OUTCOME 才是一般的。**
   OLMo-2 走的是 **bf16 exact ties**（keep8 上 `30.64%`）被 argmax 的 index bias
   打破这条路；Llama / Qwen 走的是**直接 modal collapse、ties 近零**这条路
   （Llama-2 k12 是 **100.0% modal / 0.00% ties**；Qwen3 k12 是
   **99.1% modal / 0.03% ties**）。**同一个终点，不同的路。**
   论文里必须写成“damage 把 letter interface 压向一个常数预测；在 bf16 下这有时
   表现为 exact ties（OLMo），有时表现为 sharp modal collapse（非 OLMo）”，
   **不得**写成一个统一机制。
   → `GATE1_DAMAGED_VERDICT.md` §3、`GATE3_VERDICT.md` §3

2. **“content 才是公平的 interface” 不是关于 MC interface 的一般陈述，
   它只是关于受损模型的陈述。** 在**健康且 letter 有竞争力**的模型上 letter
   反而比 label-free content **好 13–23 pp**（OLMo-2-7B `−13.59` pp、
   Llama-3-8B `−15.96` pp、Qwen3-8B `−22.91` pp，符号为 `cn − l`）。
   在中等能力档（letter `0.37`–`0.41`）则是 family-dependent：OLMo-2-1B 与
   Llama-2-7B 是 TIE，Llama-3.2-1B 显著但只 `2.4` pp。
   因此**“总是报 content_norm”是过强的建议**。
   → `GATE1_VERDICT.md` §4（verdict 已撤回但数字有效）、
     `GATE1_HEALEDARMS_VERDICT.md` §7.1.3

### ❌ MUST DROP — 已被证伪，不得以任何形式出现

**“Exact ties 是 interface failure 的机制”作为 family-general 的 CAUSAL claim。
四重证伪：**

| # | 证据 | 结果 |
|---|---|---|
| a | gate-3 base：ties `18 → 0`（fp32） | letter acc 完全不动，McNemar **p = 1.000** |
| b | gate-3 keep8：ties `4303 → 0`，**18.03%** 的 argmax 翻转 | letter acc 不动（p = 0.570），**仍在 floor 之下** |
| c | 非 OLMo 受损臂 ties `0.00%`–`1.35%` | 却塌陷到同一个地方 |
| d | **（2026-08-09 新增第四条）** 六个 OLMo-2 臂全部有 fp32-vs-bf16 结果：fp32 把 exact-tie rate 打到 `0.00%`（shortgpt16 `0.01%`），而 **6 个臂里 5 个的 floor 判定一动不动**；keep8 **仍在 floor 之下且更显著**（bf16 `−1.389` pp p `0.0192` → fp32 `−1.538` pp p `0.0062`） | ties 是 bf16 artifact，移除它不恢复任何可测精度 |

正确的（更弱、更干净的）表述：受损模型的四个 letter logit 在 bf16 精度下已无法
分辨（gap median 从 base 的 `1.1185` 压到 keep8 的 `0.2500`，压缩 4.5×）；
tie count 是这种压缩的**读数**，不是它的**原因**。**interface 丢的是信息，
不是打破平局的能力。**

唯一的 dtype 边界情形，**不得过度解读**：keep12 的判定是 dtype-fragile 的
（bf16 `+0.385` pp p `0.3736` = AT floor；fp32 `+0.819` pp p `0.0272` = above floor）。
这是 6 个臂里的 1 个，fp32 的 CI 只以 `0.078` pp 越过零，`p = 0.0272` **未做多重
比较校正**、过不了六臂 BH。且 keep12 在 fp32 下 modal 集中度**升高**
（D `49.3%` → `61.3%`），即精度提高让它**更像**一个常数预测器。
写成“六臂中有一臂的判定在边界上对 dtype 敏感，故每个判定都应连同 dtype 一起报”
（这正是 A01 自己的主张），**不得**写成“fp32 救了 keep12”。

→ `GATE3_VERDICT.md`、`STATUS.json:gate3_fp32_causal_tie_test`

---

## 协议本身也需要被校准（2026-08-09 新增；**2026-08-10 降级重写**）

> **⚠️ 本节的 headline 已于 2026-08-10 降级。** 原文写"五种同等合理的 convention
> 把 6 臂里 5 臂翻转 / null 摆动 25.76 pp"，被外部审计
> （`../A03-parametric-vs-external-memory/evidence/TCODEX_AUDIT_20260810.md` §2.1）
> 判为应撤回，**审计在实质上是对的**。逐条回应见 `TCODEX_AUDIT_RESPONSE.md` §3。
> 本节已按"**三个可执行 convention + 两个 bound**"重写。这条**不再是 A01 最强的
> claim**；最强的是下面的 tokenizer 那半条。

A01 让别人用 construct-appropriate null 取代 chance line。**A01 自己的 MC content
null 确实存在同一类未申明的自由度**："null 取最长选项"这句话没有规定多个选项在
token 数上打平时怎么办，而 MMLU 上 **34.22%** 的 item 存在这种平局（其中
**13.37%（1877 item）四个选项全平**）。但必须区分**三个可执行的 tie policy** 与
**两个 bound**：

| | convention | null | 六臂判定（bf16 content_norm） |
|---|---|---:|---|
| **可执行** | `split`（预注册 canonical，均分） | **0.284450** | **6/6 above** |
| **可执行** | `first`（最低索引 = argmax 的实际行为） | 0.281085 | **6/6 above** |
| **可执行** | `last`（最高索引） | 0.282154 | **6/6 above** |
| ❌ **不是 policy，是 oracle 上界** | `credit`（gold ∈ W 即得 1） | 0.453710 | 1/6 above，5/6 显著 BELOW |
| ❌ **不是 policy，是悲观下界** | `wrong`（任何平局判 0） | 0.196126 | 6/6 above |

**为什么 `credit` / `wrong` 不是 convention**：`credit` 要求把平局朝 gold 的方向打破，
即**必须知道 gold 是哪个字母** —— input-blind baseline 定义上做不到；`wrong` 要求
平局永远判错，而一个必须给出答案的 baseline 期望得 `1/|W|`，也无法达到。两者**夹住
的是"tie policy 未申明时 null 的 identified set"**，不是读者可选的五种读法。

**实测（executable-only）**：null 只摆动 **`0.3365` pp**（`0.281085`…`0.284450`），
**6 臂判定 0 个改变**；per-arm residual fraction 比值 base **1.018×**、
shortgpt16 **1.029×**、keep14 **1.034×**、keep12 **1.043×**、keep10 **1.056×**、
keep8 **1.058×** —— **每臂 ≤6%**。原先 headline 的 `16.26×`（base `0.0359` vs
`0.5832`）是**bound 宽度**，不是 convention 敏感度；而且五个受损/heal 臂的
`credit` residual fraction 是**负数**，比值无定义（residual **变号**）—— 这本身就
说明 `credit` 是 bound 而不是 convention。**因此不能像原文那样把它与 gate-4 的
`6.86×`–`10.04×` aggregation span 直接对比**：后者是"可辩护选项之间"的敏感度。

**降级后的正确表述**：协议要求"报 construct-appropriate null **并且打印该 null 的
tie policy**"。这个要求是**真的但对点估计影响不大**（0.34 pp，0/6 翻转）；它变得
load-bearing 只在 tie policy **未申明**时 —— 那时诚实的 null 是**区间**
`[0.196126, 0.453710]`（25.76 pp），且在 oracle 端 5/6 臂会读成 BELOW。

**并且这个 null 还依赖 tokenizer —— 这一条审计没有攻击，现在是更强的那半条。**
"最长选项"是按 **continuation token 数**量的，各家族分词不同，所以同一个 `split`
convention 在同一批 14042 item 上给出
Llama-2-7B `0.275661` / Qwen3-8B `0.283346` / OLMo-2 `0.284450` / Llama-3-8B `0.284664`
（`evidence/a01_gate1_third_family.json:longest_option_split_tie_null`），
**跨度 `0.9003` pp = 可执行 tie-convention 跨度（`0.3365` pp）的 `2.68×`**。
**而且它真的会翻判定**：本次把全部 **63** 个非 OLMo 受损深度臂用"自家 tokenizer 的
null" vs "共用的 OLMo-2 `0.284450`"重测（exact two-sided binomial，α=0.05），
**2/63 翻转，其中 1 个 robust** —— `gate1_dmg_llama2_7b_depth_k20`
（content_norm `0.287708`）对自家 null `0.275661` 是 **above，p=`0.00146`**，
对共用 null `0.284450` 却是 **AT / 不显著，p=`0.395`**。另一个
（llama3 `depth_k17`，p=`0.0507` vs `0.0443`）跨在 α 两侧，作为**边界 artifact**
报告、不作证据。因此**任何跨家族的 content 比较都必须用 per-family、per-policy
的 null**，不能共用一个 `0.2845`。这一条与 tie policy 相互独立，两者叠加。

> ⚠️ **阈值披露（A01 自己的协议要求）**：α=0.05 与 above/AT/BELOW 三分法沿用
> `code/a01_gate3_fp32_vs_bf16.py` / `code/a01_gate1_verdict.py`，是**既有的**；
> 但 `robust = min(p) < 0.005` 这个标记是**看到两个翻转之后才定义的**，属**事后**，
> 已在 `TCODEX_AUDIT_RESPONSE.md` §6 明确披露。它的作用是阻止 A01 把
> `0.0507`-vs-`0.0443` 当成发现，所以它让 claim **更弱**而非更强。63 个 per-arm
> 检验**未做多重比较校正**；若对 63 个做 BH，那个 robust 翻转（p=`0.00146`）在
> α=0.05 下仍然存活，边界那个不存活。

→ `GATE3_CONVENTIONS_VERDICT.md`（已加降级 banner）、`TCODEX_AUDIT_RESPONSE.md` §3/§6、
  `evidence/gate3_content_null_conventions.{json,csv}`、
  `evidence/a01_audit_response_recompute.json`、
  `code/a01_gate3_content_conventions.py`、`code/a01_audit_response_recompute.py`。
  所有已发表的 A01 数字都用的是 `split` 且已标注，**as published 全部正确**
  （对档案 summary 复核到 `<1e-12`）。


---

## 已完成的四类 construct

| Construct | Reported | Null | convention | Residual fraction |
|---|---:|---:|---|---:|
| C1 MC content scoring | 0.3598 | longest-option 0.2845 | **`split`（tie 平分）** | 20.9% |
| C2 Generative majority prior | 0.6590 | constant refusal 0.4985 | — | 24.4% |
| C3 Representation similarity | 0.4907 | layer-order shuffle 0.4529 | 2000-perm | 7.69% |
| C4 Probe readout depth | 0.6610 | native readout 0.1505 | V1 预注册 | 77.2% |

稳妥表述是：残余比例约 **8%–77%**；**不要**把“恰好超过 10×”作为 headline，
因为 C4 aggregation 的合理变体给出约 **6.86×–10.04×**（gate-4 五个变体里只有
预注册的 V1 达到 10.04×，最近的替代变体给 9.98×）。可写的句子是
“residual fractions span 约 6.9–10.0×，取决于 C4 aggregation convention；在预注册
primary 下为 10.0×”。

→ `GATE4_VERDICT.md`、`evidence/gate4_c4_prereg.json`

## Representation leg（C3）

- observed mean midband z-CKA：`0.490672`
- 2000-permutation layer-order null：`0.452936`
- calibrated residual：`0.037737`
- residual / reported：`7.69%`
- BH `q=.05`：`52/91` pair（raw `p<.05` 为 `57/91`）

random-init `0.0912` 是错误 null；使用它会把可用 correspondence signal
夸大约 10.6×。

---

## 新颖性边界（2026-08-09 经正式 venue 核实后更新）

完整核查见 **`NOVELTY_CHECK.md`**（含每篇的 verified venue + 验证来源）。
**第三条 kill clause 未触发。**

### 不能主张（必须引用，不得据为己有）

- 首创 permutation null calibration / 首创 BH / “表征相似性文献没有 null”；
- “MCQA 应该用比 chance 更强的 baseline” — **Balepur et al., ACL 2024 main**
  (`10.18653/v1/2024.acl-long.555`) 已明确提出；
- “null model / 常数输出能在 benchmark 上拿高分” — **Zheng et al., ICLR 2025 Oral**
  (arXiv:2410.07137) 已提出；
- **`acc` vs `acc_norm` 的 length 敏感性 —— 这一条必须 DROP。**
  **Oostermeijer, ICML 2026**（arXiv:2607.12767，OpenReview `venueid=ICML.cc/2026/Conference`
  + `Camera_Ready_Revision`）已证明 length normalisation 会 over-correct。
  A01 的 OBQA sign-flip 只能改写成“在受损模型这一新设定下的复现”；
- letter/cloze 两种 MC interface 的存在及其改变结果 — **OLMES, Findings of NAACL 2025**
  (`10.18653/v1/2025.findings-naacl.282`) 已标准化；
- “construct validity 是 LLM benchmark 的问题” — **Bean et al., NeurIPS 2025 D&B**
  (arXiv:2511.04703) 已系统综述 445 个 benchmark；
- “MC scoring interface 的有效性令人担忧” — **Cho et al., ICLR 2026**
  (arXiv:2502.18798) 已从 choice sensitivity 角度提出。

### 可以主张

1. 跨多个无关 construct 的统一 null-calibrated reporting（含 convention 字段）；
2. ~~**null 自身的 convention 自由度会反转判定**（`credit` 翻 5/6 臂，
   单臂 residual fraction 16.26×）~~ **2026-08-10 降级重写**：
   **null 自身有两个未申明的自由度，且 tokenizer 那个会真的反转判定** ——
   (a) tie policy：三个可执行 policy 下 null 只摆 `0.3365` pp、`0/6` 判定改变；
   policy 未申明时 null 是区间 `[0.196126, 0.453710]`（`credit` 是 oracle 上界、
   `wrong` 是悲观下界，两者都不是可执行 policy）；
   (b) **tokenizer**：同一 `split` policy 下四家族 null 跨度 `0.9003` pp
   （= 可执行 tie 跨度的 `2.68×`），且用别家的 null 会**抹掉一个真实的 above-null
   判定**（Llama-2 k20：自家 null p=`0.00146` above，共用 null p=`0.395` AT）。
   未在任何已核实的 prior art 中出现；
3. 针对 layer correspondence 问题的 **layer-order null**
   （措辞降级为“我们未发现更早的 layer-order null”，而非“首创”）；
4. 把该协议先用于**撤回自己的结果**，并且用它撤回自己的第一次撤回；
5. 给出 calibrated residual fraction，而非只给显著性；
6. **对 OLMES 的一处 defect 做 follow-up 修正**：按**模型规模**选 interface 的规则
   在**读出能力已受损**的模型上失效（受损 7B 会被规则派到 letter 口径，而它在那里
   位于自己的 best-constant floor 之下）；正确的键是该臂自己的 floor test。

### 必须补的反向 caveat

**Feng, Wallace, Boyd-Graber, “Misleading Failures of Partial-input Baselines”**
（arXiv:1905.05778，venue 待按 Anthology 核实）证明 partial-input baseline
**失败**并不能证明数据集没有 artifact。因此 A01 必须明写：**越过 floor 是必要
条件，不是充分条件。** A01 现有 claim 的方向本来就是安全的（只用 floor-failure
去**取消资格**，从不用 floor-success 去**认证有效**），但论文里要把这句说出来。

---

## Gate 状态

| gate | 状态 | 结论 |
|---|---|---|
| 1. 第三个模型家族的 MC interface case | ✅ DONE | intact 腿测错了条件（已撤回）；damaged 腿 **6/6 确认一般性 claim** |
| 1b. depth curve（四家族，单层分辨率） | ✅ DONE，**结论 2026-08-10 部分撤回** | ~~letter 是 step function~~ → **三家族有单层大跳（+26.7/+30.3/+48.0 pp），Llama-2 是反例（6 个 BH 显著下降、穿越 floor 4 次）；"step function / family-general" 已撤回** |
| 1c. healed arms | ✅ DONE | heal 软化 step 但不闭合 interface gap；shortgpt16 保住 intact 的排序 |
| 2. 非 MMLU 的 MC benchmark | ✅ DONE | **复现**（所以 AND-gate 不可能触发） |
| 3. OLMo full-fp32 forward | ✅ DONE（六臂） | ties 是 bf16 artifact；移除后判定 5/6 不变 → 机制被证伪 |
| 3b. longest-option convention 表 | ✅ DONE（2026-08-09），**2026-08-10 降级** | ~~null 自己有 convention 自由度，翻 5/6 臂~~ → **可执行 policy 只摆 0.3365 pp / 0-6 翻转；5/6 来自 oracle 上界。tokenizer 才是更强的那半条（0.9003 pp，翻 1/63 robust）** |
| 4. C4 aggregation 预注册 | ✅ DONE | headline 必须是区间 `6.86×–10.04×` |
| 5. novelty / prior-art 边界 | ✅ DONE（2026-08-09） | **kill clause 3 未触发**；6 条引用义务 + 1 条 sub-claim 撤回 |

### 成功条件（对照）

- ✅ 至少三个 construct 的 null calibration 改变科学结论：C1（ranking flip 全部
  发生在 floor 之下）、C2（BoolQ 4/6 臂从“及格”变成不可与常数预测器区分）、
  C3（residual fraction 从错误 null 的 10.6× 夸大回落到 7.69%）。
- ✅ 第三模型 / 第二 benchmark 保持 “instrument validity before comparison”：
  damaged 腿 6/6 + gate-2 复现。
- ✅ 与已有 similarity-null prior art 的边界经正式 venue 核实：见 `NOVELTY_CHECK.md`
  （ACL 系走 Anthology，OpenReview 系走 `venueid` + `Camera_Ready_Revision`）。

### Kill 条件（逐条对照，全部未触发）

1. ❌ 未触发 — “除 representation 外其他 construct 在严格 null 下结论都不变”：
   假。C1/C2 的结论都被 null 改变（见上）。
2. ❌ 未触发 — “第三家族和第二 benchmark 均不复现 interface failure”：
   假。damaged 第三/第四家族 **6/6 复现**，gate-2 也复现。
   （2026-08-09 曾被误判为触发，因为测的是 intact base 而 clause 说的是 damaged。）
3. ❌ 未触发 — “论文只能退化为已有 similarity-null 方法的案例集合”：
   假。存活的核心发现（四家族 floor 塌陷、letter-as-step-function、fp32 机制证伪、
   null convention 敏感性）都在 MC-accuracy construct 上，不是 similarity-null 的
   实例。见 `NOVELTY_CHECK.md` §1/§7。

---

## 不得复活的旧数字

来源 `STATUS.json:must_not_resurrect`，逐条给出更正后的正确值。

| 旧数字 | 为什么错 | 正确值 |
|---|---|---|
| `4.8×` | 口径不明 | 相对 `.25` 为 **`4.69×`**；相对 content 自己的 floor 为 **`3.22×`** |
| longest-option `.2822` | **不是算错**——它是 `last`-of-maximal convention（精确 `0.282154`）。它是一个**不同且较难辩护的 convention**，被预注册的 `split` 取代。 | canonical 为 split-tie **`0.2845`**（`0.284450`）。⚠️ 现在应把它放进 convention 表里公开讨论，而不是单纯禁用 |
| `58/91 significant` | 未做 BH | canonical **`52/91`**（BH q=.05）；raw `p<.05` 为 `57/91` |
| 把 `.25` 当 MMLU 的 null | chance 不是 construct-appropriate null | letter null = always-D **`0.2689`**；content null = **`0.2845`**（`split`） |
| 把 `.50` 当 BoolQ 的 null | 同上 | always-B **`0.6217`**（gold 2033 B / 1237 A） |
| “Qwen3 在 keep30 跳 +45.74pp” | 混淆了**跳变**与**相对 floor 的残差** | transition 是 **k24→k25，+48.02 pp**（`0.229739`→`0.709942`）；`+45.74 pp` 是 **k30 相对 floor 的残差**（`0.726321−0.268908`）|
