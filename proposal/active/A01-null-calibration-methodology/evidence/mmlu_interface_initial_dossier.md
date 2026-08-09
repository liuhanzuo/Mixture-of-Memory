# MAIN 亲自复算：MMLU 双 interface 的仪器效度（Paper E 候选）

**日期**：2026-08-06 GMT+8
**性质**：全部数字由 MAIN 从 per-example 原始 jsonl 复算，**非 subagent 转述**。
**原始数据**：`olmo2_mmlu_content_results/<10 arm>/per_example_mmlu.jsonl`（OLMo-2-7B）、
`qwen3_mmlu_content_results/<4 arm>/per_example_mmlu.jsonl`（Qwen3-8B，本轮新跑，8 卡 × 4 arm，
n_nan=0，8/8 shard 全 merge）。零新训练。

---

## 0. 一句话

**在 MMLU 上，"score 字母 A/B/C/D" 这个标准口径在受损模型上会退化成常量预测器**——
5/13 个 arm 的 letter 准确率**显著低于"永远答 D"这个无知识常量基线（.2689）**，
而 content 口径在同样的 item 上仍显著高于各自的无知识地板。
两种退化机制、两个模型家族、n=14042、全部有配对显著性。

---

## 1. 四个观察（含一个被我自己推翻的）

### Obs 1 ✓ 跨家族成立：interface gap 随模型健康度翻转符号

Δ = content_norm − letter（pp），n=14042，全部 McNemar p<1e-9：

| arm | letter | content_norm | Δ |
|---|---|---|---|
| **OLMo-2-7B intact base** | .6054 | .4706 | **−13.48** |
| OLMo-2 full32 @25k | .5877 | .4662 | −12.14 |
| OLMo-2 ShortGPT-16 @200k | .4742 | .4012 | −7.31 |
| OLMo-2 keep14 @200k | .3184 | .3832 | +6.48 |
| OLMo-2 keep12 @124k | .2728 | .3629 | +9.02 |
| OLMo-2 keep10 @83.5k | .2720 | .3445 | +7.24 |
| OLMo-2 keep8 @121k | .2550 | .3423 | +8.72 |
| OLMo-2 freeze-front @200k | .2624 | .3604 | +9.81 |
| OLMo-2 scratch-16L @200k | .2470 | .3598 | +11.28 |
| OLMo-2 keep14-reheal @67.5k | .2492 | .3658 | +11.66 |
| **Qwen3-8B intact base** | .7294 | .5053 | **−22.40** |
| Qwen3 f12k2 inherit @2k | .2297 | .2715 | +4.17 |
| Qwen3 f12k2 scratch @2k | .2305 | .2636 | +3.32 |
| Qwen3 f12k2 inherit @200k | .2515 | .2951 | +4.37 |

**健康模型 letter 更高，受损模型 content 更高。两个家族同向，Qwen 幅度更大（−22.40）。**

### Obs 2 ✓ subject 级 rho 崩塌超出 binomial 噪声

Spearman rho(letter, content) 跨 57 subject，对照"两口径测同一 latent 能力、只差 binomial
采样噪声"的参数 null（2000 抽样）：

| arm | 实测 rho | null 中位 | null 95% 区间 | 判定 |
|---|---|---|---|---|
| intact base | 0.676 | 0.915 | [0.868, 0.949] | 低于 null |
| full32 @25k | 0.691 | 0.916 | [0.870, 0.949] | 低于 null |
| ShortGPT-16 | 0.802 | 0.890 | [0.833, 0.933] | 低于 null |
| freeze-front | 0.057 | 0.617 | [0.444, 0.753] | 低于 null |
| scratch-16L | 0.044 | 0.662 | [0.518, 0.786] | 低于 null |
| keep14 @200k | 0.726 | 0.791 | [0.680, 0.870] | **在 null 内** |

5/6 低于噪声 null 下界。⚠️ 但注意 rho 崩最狠的两个（.057/.044）正是近 chance 的 arm
——**这条不能单独当证据，必须配 split-half reliability 作分母**（workflow attack 4 在做）。

### Obs 3 ✓ Simpson's paradox：等长子集上符号反转

按 option 长度 spread 分层（intact base）：

| 分层 | n | letter | content | Δ |
|---|---|---|---|---|
| **全等长（长度不可能混淆）** | 1877 | .4459 | .5184 | **+7.25**（p=6.7e-7）|
| spread 1–2 | 4841 | .6100 | .5272 | −8.28 |
| spread 3–6 | 3911 | .6622 | .4582 | −20.40 |
| spread >6 | 3413 | .6214 | .3783 | −24.32 |

pooled 是 −13.48，等长子集是 **+7.25**，符号相反。10/10 arm 的等长 Δ 全为正
（+2.56 … +13.05）。⇒ **"length norm 不完美"能解释 pooled 的幅度，但解释不掉效应本身。**

⚠️ **已知缺陷**：等长子集 subject 组成有偏（elementary_math 富集 5.30×、
high_school_math 5.26×、abstract_algebra 5.98×；top-10 subject 占等长子集 55.9% vs 全集 21.4%）。
**必须做 within-subject 分层或重加权**才能用（workflow attack 1 在做）。

### Obs 4 ✗ **被我自己推翻**：排序翻转只发生在仪器已失效处

45 对里 7 对符号相反；paired bootstrap 两侧都显著的 2 对：
- keep10 vs scratch-16L：letter **+2.51** [+1.45,+3.56] vs content **−1.53** [−2.24,−0.83]
- keep10 vs keep14-reheal：letter **+2.29** [+1.15,+3.42] vs content **−2.13** [−2.71,−1.54]

第一对本是"继承 vs 随机初始化"对照，看着是完美的 headline。**但严格检验后作废**：

> 把范围限制到"两个口径都显著优于各自无知识地板"的 4 个 arm（base / full32 / keep14@200k /
> ShortGPT-16），**它们之间的显著翻转 = 0 个**。
> 两个"显著翻转"对里的 arm，**letter 口径全都不优于常量预测器**。

⇒ **不能写"interface 翻转模型排序"**。翻转发生在 letter 已经退化的地方，
那不是"两个有效仪器给出相反结论"，而是"一个坏仪器的噪声符号"。**这条必须降级。**

**但幅度压缩是真的**（在双口径有效的 arm 之间）：

| 对比 | letter | content | 压缩 |
|---|---|---|---|
| keep14 vs ShortGPT-16 | −15.58pp | −1.79pp | **0.12×** |
| base vs full32 | +1.77pp | +0.43pp | 0.24× |
| base vs keep14 | +28.70pp | +8.74pp | 0.30× |
| full32 vs keep14 | +26.93pp | +8.30pp | 0.31× |

**同一组对比在两口径下差 3–8 倍**，方向一致但效应量完全不同 → 任何"恢复了 X%"的说法
都是口径相对的。

---

## 2. ★★ 真正的 headline：letter 口径的两种退化，且可归因到机制

### 无知识常量基线
MMLU gold 字母边际 A=22.9% / B=24.7% / C=25.5% / **D=26.9%** ⇒
**最优常量预测器（永远答 D）acc = .2689**。

配对检验（每 arm 的 letter 正确性 vs 常量预测器在同一 item 上的正确性，4000 bootstrap）：

| arm | letter | vs 常量地板 | 判定 |
|---|---|---|---|
| OLMo-2 base | .6054 | +33.65pp | 优于常量 |
| OLMo-2 full32 | .5877 | +31.88pp | 优于常量 |
| OLMo-2 ShortGPT-16 | .4742 | +20.53pp | 优于常量 |
| OLMo-2 keep14@200k | .3184 | +4.95pp | 优于常量 |
| OLMo-2 keep12 | .2728 | +0.38pp | **不可区分于常量** |
| OLMo-2 keep10 | .2720 | +0.31pp | **不可区分于常量** |
| OLMo-2 freeze-front | .2624 | −0.66pp | **不可区分于常量** |
| OLMo-2 keep8 | .2550 | **−1.39pp** | **★ 显著差于常量** |
| OLMo-2 keep14-reheal | .2492 | **−1.97pp** | **★ 显著差于常量** |
| OLMo-2 scratch-16L | .2470 | **−2.19pp** | **★ 显著差于常量** |
| Qwen3 base | .7294 | +46.05pp | 优于常量 |
| Qwen3 inherit@2k | .2297 | **−3.92pp** | **★ 显著差于常量** |
| Qwen3 scratch@2k | .2305 | **−3.85pp** | **★ 显著差于常量** |
| Qwen3 inherit@200k | .2515 | −1.74pp | 差于常量 |

**5/13 显著差于常量，3/13 不可区分。**

### 公平性：content 也做同样检验（不能只查对手）
content 的对应无知识基线 = 永远选最长 option（.2822 OLMo / .2807 Qwen）。结果：

- OLMo-2：**10/10 arm 的 content 都显著优于最长-option 基线**。
  6 个 arm 是"**仅 content 有效**"，4 个"两者都有效"，**0 个"仅 letter 有效"**。
- Qwen3：base 与 inherit@200k 的 content 有效；**inherit@2k / scratch@2k 两个口径都无效**
  （诚实记账：Qwen 的 2k-step arm 根本没有可测能力，不该用来支持任何结论）。

⇒ **不是"content 更好"，是"letter 在受损模型上先失效"**，且 content 并非万能——
它自己在最差的 arm 上也会撞地板。

### 机制一：bf16 精度导致的精确并列（OLMo-2）
letter margin 大量取值恰为 1/16 的倍数（0.0625/0.125/0.25…），top1==top2 **精确并列**率：

| arm | 并列率 | pred 集中度 | 剔除并列后 letter acc |
|---|---|---|---|
| intact base | **0.1%** | 22%(A) | .6054 → .6059 |
| full32 | 6.4% | | .5877 → .6090 |
| ShortGPT-16 | 10.4% | | .4742 → .4980 |
| keep10 | 13.8% | 60%(D) | .2720 → .2744 |
| freeze-front | 19.4% | | .2624 → .2690 |
| keep14@200k | **24.4%** | 39%(A) | .3184 → **.3446** |
| keep12 | 24.6% | | .2728 → .2787 |
| keep14-reheal | 25.5% | 55%(B) | .2492 → .2543 |
| **keep8** | **30.6%** | 49%(C) | .2550 → .2578 |

**代码级证据**（`scripts/eval_olmo2_mmlu_content.py`）：line 200 `autocast(dtype=torch.bfloat16)`
前向 → line 204 `log_softmax(out.logits.float())`。**fp32 cast 发生在 bf16 logits 之后，
无法恢复已丢失的精度**。受损模型 logit spread 塌缩（median 2.479 → 0.500~1.000）后，
4 个字母的 logit 差落到 bf16 可表示精度以下 → 精确并列 → `argmax` 按索引 tie-break。

### 机制二：字母先验坍缩（Qwen3）
Qwen 受损 arm 没有并列问题（0.1–0.2%），但 **pred 100% 是 "A"**（inherit@2k，
预测分布熵 0.025/1.0）、**96% 是 "A"**（scratch@2k，熵 0.131）。对照 intact base 22/27/27/23、
熵 0.997。⇒ 不同家族退化成不同的常量，但都是常量。

**两种机制都指向同一结论：letter 口径失效的方式取决于实现细节（数值精度 / tokenizer 先验），
不取决于模型知识。**

---

## 3. ★ 对我们自己 Paper B 的影响（必须处理，论文在审）

`paperB/sections/tab_interface_audit.tex` 现报 `Random, 200k` letter = **.2470**，
`Frozen, 200k` = .2624 —— 这两个数字**分别是"显著差于常量"和"不可区分于常量"**，
但表里当作普通测量值呈现，caption 只说 "The random floor shows that an interface gain cannot
by itself certify inherited target recovery"，**没有说 letter 侧本身已失效**。

`paperB/sections/app_tab_recovery.tex` 的 keep14 MMLU recovery = **19.4**：
- MAIN 复算（全 item）= **19.25%** ✓ 与论文一致
- 复算（24.4% 的 bf16 并列按弃权处理）= **26.57%**
- ⇒ **headline 恢复率对"如何处理并列"敏感，差 7.3pp**

其它 arm 的同类偏移：ShortGPT +6.59pp、full32 +5.87pp（后者 recovery 从 95.01% → 100.89%，
**跨过了 100%**）。

**这不是造假，是一个未被声明的口径依赖。** 但既然论文在审，**必须主动加一个 tie-rate 列
+ 一句 limitation**，否则 reviewer 自己发现会更糟。

---

## 4. 现在可写 vs 不可写

| 声明 | 状态 |
|---|---|
| letter 口径在受损模型上退化为常量预测器，5/13 arm 显著差于"永远答 D" | ★★ **最强，双家族，配对显著** |
| 两种可归因机制：bf16 精确并列（OLMo，30.6%）/ 字母先验坍缩（Qwen，100% "A"） | ★★ **有代码级证据** |
| interface gap 随模型健康度翻转符号（−22.40 … +11.66） | ★ 双家族成立 |
| 同一对比的效应量在两口径下差 3–8 倍（0.12×–0.57×） | ★ 在"双口径有效"arm 之间成立 |
| 等长子集上 Δ 符号反转（+7.25 vs pooled −13.48） | ⚠️ **待 subject 重加权**才能用 |
| ~~interface 翻转模型排序~~ | ✗ **MAIN 自己推翻**：0 个翻转发生在双口径有效的 arm 之间 |
| subject-level rho 崩塌 | ⚠️ 需 split-half reliability 作分母，否则可能是 floor effect |

---

## 5. 待办

1. **workflow wrolfidns**（7 路 prior-art × 各自 adversarial verify + 5 路 hostile recompute）
   出结论后决定 framing / venue / GO-NO-GO。最大 kill risk：Holtzman surface-form competition、
   Alzahrani leaderboard sensitivity、以及 lm-eval-harness 的 acc vs acc_norm 是否已覆盖。
2. **Paper B 补 tie-rate 列 + limitation 句**（论文在审，优先级最高）。
3. Obs 3 的 within-subject 分层 / 重加权。
4. Obs 2 换成 split-half reliability 分母。
5. 若要投，需要第三个家族（Llama-3-8B ckpt 在 .73）+ 一个非 MMLU 的 MC benchmark。

## 6. 元教训

**我自己的 headline（Obs 4 排序翻转）被自己的下一步检验推翻了**，而推翻它的过程产出了
更强的 headline（常量地板 + 两种机制）。
教训与 [[two-disk-rule-applies-to-main-too]] 同型：**"看着像 headline 的数字要先问它是否在仪器
有效区内"**。若我当时直接把"interface 翻转排序"写进提案，reviewer 一问
"那两个 arm 的 letter 是否优于常量" 就崩。
