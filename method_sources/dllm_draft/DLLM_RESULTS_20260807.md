# dLLM 实验结果汇总（task #173）

**日期**：2026-08-07 GMT+8
**硬件**：LOCAL wzc1 8× L20A（sm_100 Blackwell，183GB/卡）+ .104 8× H20（AR baseline）
**环境**：`/opt/conda/envs/dllm-env`（torch 2.11+cu128 + transformers 4.51.3）
**评测**：EvalPlus HumanEval+ v0.1.10（n=164）/ MBPP+ v0.2.0（n=378），`evalplus.evaluate` 官方 grader

---

## ⛔⛔ 2026-08-12 CORRECTION BLOCK（A05 K1 canvas sweep + closeout；四条，勿再引用旧值）

本节由 `Mixture-of-Memory/proposal/archive/A05-structural-dllm-cost-frontier/` 的 K1 gate 及其
closeout 加入。**本文件下方所有 DreamOn 相关数字受此约束**；旧值不删（保留 provenance），
但引用前必须先读本节。

**C1 — 归档的 `nfe` 不是 forward-pass 计数。**
`scripts/generate_evalplus_dreamon.py:152-155` 记的是 `len(output.history)`。
DreamOn 的 `models/DreamOn-v0-7B/generation_utils.py` 在 **三处** 各 append 一次
（`histories.append` @ line **445** transfer step / **476** delete batch / **495** expand batch），
所以 `len(history)` 最多可达真实 model call 数的 ~3×；且 `output_history=False` 时它是 `None`
—— 这正是 r2 每个 item 都 `nfe: null` 的原因（实测 r2 non-null = **0/164** 与 **0/378**）。
- 作废：由 `runs/dreamon_*.r1/metrics.rank*.jsonl` 聚合出的 **265.88 (HE+) / 135.65 (MBPP+)**
  （实测确认这两个数就是 r1 的 `mean(len(history))`，即错误量本身）。
- 真值（A05 K1 用 `model.forward` wrapper 直接计数，同 sampler、同 canvas=8 归档设置）：
  **HE+ 172.3 / MBPP+ 153.4**。注意 MBPP+ 方向**相反**（135.65 → 153.4 是变大），
  所以这不是一个可以整体缩放的偏差。
- 任何由旧值推出的 "DreamOn 比 scaffold 贵/便宜 N×" 的说法都必须重算。

**C2 — `mask_expansion` / `delete_eos_token` 的失效已由【执行】确认（此前只是读代码）。**
2026-08-12 实测 `DreamGenerationConfig().update(mask_expansion=True, delete_eos_token=True)`
**原样返回这两个 key 作为 unused**，且 `hasattr(cfg,'mask_expansion')` 前后都是 `False`；
对照组 `update(temperature=0.2)` 返回 `{}` 并正确赋值。
- 因此 **`scripts/_run_baselines_r2_wzc1.sh` 头部注释宣称的 r2「fixes applied:
  mask_expansion=True + delete_eos_token=True」描述了一个从未生效的修复**，该注释已就地更正。
- 本文件 §1 表格里 `DreamOn-v0-7B（+mask_expansion）` 的行名是**错的**，已就地更正。

**C3 — 新 bug：HE+ 的 `combine_humaneval_prompt()` 双重缩进，导致 HE+ 被【低估】。**
`scripts/generate_evalplus_dreamon.py:29-34` 先调 `extract_python()`，而后者以 `.strip()` 结尾
—— 它只把**第一行**的缩进去掉，第 2..n 行仍保持原有深度。DreamOn 输出的本来就是
**已缩进 4 空格的函数体**，于是 `textwrap.indent(body,"    ")` 把第 1 行推到 4、第 2 行推到 8
→ `IndentationError: unexpected indent`。
⚠️ **在 `extract_python()` 之后再 dedent 是 no-op**（第一行已被 strip，公共前缀已是 0）；
必须在**抽取之前** dedent。修正见 `scripts/generate_evalplus_dreamon.py::combine_humaneval_prompt`。
- 影响：canvas 越大、模型吐得越多，这个 bug 越致命。A05 closeout 重新评分（**生成不变**，
  同一批 `raw_output`，0 GPU，evalplus 自检 canonical PASS / stub FAIL）：

| HE+ cell | parseability 修前→修后 | pass@1 plus 修前→修后 |
|---|---|---|
| canvas=8（= 归档设置） | .988 → **1.000** | .1280 → **.1341** (+0.61 pp) |
| canvas=32 | .860 → **.982** | .2134 → **.2561** (+4.27 pp) |
| canvas=128 | .287 → **.963** | .1707 → **.4817** (**+31.10 pp**) |

- **归档的 HE+ .122 也受同一 bug 影响**，量级约 +0.6 pp（canvas=8 处最小）。
- **一个曲线形状的结论被改写**：修正前 HE+ 随 canvas 是**非单调**的（.128 → .213 → .171），
  据此曾推测 canvas=32 是峰值；修正后是**单调上升**（.134 → .256 → **.482**），
  即 c128 才是三档里最好的，"c128 退化"整条叙述是 harness artifact。
- MBPP+ 不受影响（走 `extract_python`，无 stitch），实测 parseability .984 @c32。

> **★ C3 补充（2026-08-12 二次审计，0 GPU）：这个 bug 的【影响范围为零】，且它的严重性【由 canvas 制造】。**
> 预注册falsification条件见 `Mixture-of-Memory/proposal/archive/A05-structural-dllm-cost-frontier/A05_BLAST_RADIUS_PREREGISTRATION.md`（commit `bed7e43`，**先注册后 grep**）；
> 判定 = `.../A05_BLAST_RADIUS_AND_SCAFFOLD_VERDICT.md`；证据 = `.../evidence/a05_blast_radius.json`。
>
> **(1) 影响范围 = 只有 DreamOn HE+ 两个 run，其余全部结构性免疫。** 17 个 arm 用同一批已存输出
> 重新评分（as-run vs 修正后，evalplus 每次调用自检 canonical PASS / stub FAIL），**只有 2 个 arm
> 变化**（`dreamon_heplus.r1` / `_r2`，各 +0.61 pp = 1 题 `HumanEval/13`，0 题倒退）。
> 其余每个 arm 命中 buggy 分支的题数**恰好为 0**：
> `combine_humaneval_prompt` 全仓（三个 checkout：wzc1 `5735d72` / zwfy6 `9651406` / `_104` `d214d37c`）
> **只在 `scripts/generate_evalplus_dreamon.py` 一个 driver、且只在 `args.dataset=="humaneval"` 一个分支**
> 里被调用。AR 与 Dream 系走 `combine_base_continuation` = `extract_python(prompt+continuation)`，
> **先拼接再抽取、从不 indent**；scaffold 系是 `solution = result.text`，**完全无后处理**；
> kspan / refine 是另一套逻辑、另一个 benchmark surface。
> **所以本文件 §1 主表、AR baseline 表（`.707`/`.680`）、scaffold tier 表都不受影响，无需重算。**
>
> **(2) 顺带得到一次跨盘复现**：上述 17 个 arm 里 15 个有已发表数字，用**另一个盘（zwfy6 `.73`）上
> 独立写的 grader** 重算，全部落在 **±0.28 pp** 内（最差 1/378 题）：
> Dream-Coder-Instruct `.707`→`.7073` / `.680`→`.6772`；Dream-Coder-Base `.079`→`.0793` / `.159`→`.1587`；
> DreamOn `.122`→`.1220` / `.085`→`.0847`；scaffold Medium `.177`→`.1768` / `.354`→`.3545`；
> scaffold Large `.177`→`.1768` / `.325`→`.3254`。
>
> **(3) ⚠️ 上面那句「归档的 HE+ .122 也受同一 bug 影响，量级约 +0.6 pp」现已由实测确认为
> `.1220 → .1280`（+0.61 pp，恰好 1 题）——但要点是：这不是"第二个独立的 31 pp 缺陷"。**
> 在 `initial_masks=8`（= 已发表的设置）下，164 条 raw output 里**只有 1 条是多行的**
> （128 条为空，抽取后仅 36 条非空），而双重缩进**只能破坏多行函数体**。
> 所以 +0.61 / +4.27 / +31.10 pp 这条递增序列说明：**canvas 是一个独立 artifact，
> 而 stitch bug 的严重性是"修好 canvas 之后才被暴露出来"的——两者是交互，不是两个同量级的独立缺陷。**
> C4（MBPP+ `.0899 → .3545`）不受此修正影响：它走的路径**根本没有 stitch**。

**C4 — 「DreamOn 在 full-program generation 上弱」是 canvas 预算 artifact。**
只把 `initial_masks` 从 8 改成 32（其余 sampler 旋钮全部冻结在归档 r2 值），
DreamOn 的 **MBPP+ 从 .085 → .3545**、**HE+ 从 .122 → .2134（修 C3 后 .2561，c128 处 .4817）**。
scaffold Medium 是 .177 / .354。**即 DreamOn 在两个 benchmark 上都追平或反超 scaffold。**
所以本文件里任何"DreamOn 在 from-scratch / full-program 上弱"的措辞，
只成立于 `initial_masks=8` 这一个特定调用方式，**不是模型属性**。
Provenance：`Mixture-of-Memory/proposal/archive/A05-structural-dllm-cost-frontier/`
（`A05_K1_CANVAS_SWEEP_VERDICT.md`、`A05_CLOSEOUT_VERDICT.md`、`evidence/`）。

---

## ⛔ TL;DR（2026-08-07 23:40 更新，第 9 次 retraction）

**R8 / R9（2026-08-07 23:40，均为发表前抓到，见下）**：R8 撤回 AR 跨架构 claim —— 评分轴用错导致符号写反，正确值是 **AR −2.44 pt vs dLLM +2.44 pt：量级相等、符号相反**，即对称硬件噪声；**更正后结论更强**，且「dLLM 对硬件异常敏感」在 pass@1 上死、只在 bit level（75/164 vs 40/164）存活。R9 撤回 MBPP+ rank-transfer 的裸 rho=0.9379 —— 它靠 3 个被 headline 自己排除的坏 cell 撑起来，真值是 **n=5 distinct plausible 上 rho=0.60，不显著**。两条都不影响 sampler audit 的核心结论。

**k-span 线状态见下方 Retraction 7。** Retraction 6 里唯一存活的那条（「AR 退化快 2×」）也死了，死于我派去**防御它**的实验：去掉 diffusion 的 oracle 长度施舍后，**交互符号翻转**（+0.153 → −0.068；稳健估计 p=.0038 → p=.943），diffusion 比 AR 退化**更快**。

**k-span infilling 在 HumanEval 上产不出任何 family-level 可发表结论。此线终止。** 存活的只有方法论副产品：(1) 嵌套难度阶梯会制造它想测的结果；(2) null 若该失败却通过，说明指标什么都没测；(3) NFE 跨家族不可比；(4) 成本均值不能条件于成功终止。**这些不是关于 diffusion 能力的结论。**

**以下为 15:00 的 Retraction 4 记录（仍有效）**：加了缺失的 AR baseline 之后，此前"存活"的 finding 也没了。这次不是方法学 bug（前 3 次都是），是**真结果**：一个和 Dream-Coder-v0-Base **架构一模一样**的 AR 模型（Qwen2.5-Coder-7B），在 **`tokens_fed` 口径上比 scaffold Medium 便宜 ~70×**，同时**质量高 0.30**。两个成本口径上都严格支配。

我此前写的"scaffold 独占低成本前沿"是**因为从来没测过 AR**——领域的自然基准根本没进入 Pareto 计算。这是最根本的失误：**方法有意义的前提，是相对合适的 baseline 有意义**。

## ✅ Sampler protocol audit（2026-08-07 20:20 完成）—— 今天唯一活下来的正向结果

**6.7 小时 / 169 tool calls / 25 个 graded cell，`.82`+`.104`。MAIN 已从 `runs/sampler_audit/summary.json` 逐位复核，25 cells 各 164 任务、零 generation error。**

**固定步数预算（NFE=512）下，仅改 decoding 超参就能让 Dream-Coder-v0-Instruct-7B 在 HumanEval+ 上移动：**

| 范围 | HE spread | HE+ spread |
|---|---|---|
| 全部 25 配置 | **59.8 pt**（.207 → .805） | **56.1 pt**（.183 → .744） |
| 排除明显坏的（`alg=origin`、`alg_temp=0.5`），n=21 | **28.7 pt** | **26.8 pt** |

**这个 spread 超过了所有已核实的方法增益**：DiffuCoder coupled-GRPO **+4.4**、Order-Token Search **+6.8**、Edit-Based Refinement **+11.6**、SCOPE+D3IM **+15.3**、MRP **+22.6**（其最佳情形）。**连保守的 26.8 都大于除一个之外的全部。**

### ★ 噪声地板是实测的，不是假设的（这是这条结果可解释的前提）

| 重复条件 | HE spread | 判定 |
|---|---|---|
| T=0.1 参考 × 4（3 显式 seed + 无 seed） | **pass@1 上 0.00 pt**（.762/.689 四次四位小数相同） | pass@1 确定；**文本非 bit-identical** |
| T=0.0 重复 × 2 | **精确 0.0**（.518/.476 两次，逐位相同） | 确定性（真 bit-identical） |
| T=0.7 × 4 seed | **2.4 pt**（sd 0.0117） | 高温区**非**无噪，已如实报告 |

每个 axis 的 spread 都远超此地板。

> ⚠️ **地板口径更正（R8，2026-08-07 23:40）**：T=0.1 那行此前写「精确 0.0」，**只在 pass@1 口径成立**（0 flips、四位小数相同），**不是 bit-identity** —— 同节点同配置下 164 条里仍有 2-3 条 raw text 不同，因为 T=0.1 不是 greedy 且未设 seed（见 `CROSSNODE_REPRODUCIBILITY.md` §2.6）。「精确 0.0 / provably zero / token-identical」只适用于 **T=0.0**（该处 commit 的 token 是 `probs.max()`，dup run 确认逐位相同）。机制隔离那节说的「噪声地板已证为精确 0」指的正是 T=0.0 区，**该论证不受影响**。

### 三条 MAIN 从表里额外读出的

1. **checkpoint 自己推荐的 `top_p=0.95` 不是最优**：0.80/0.85 给 **.805/.744**，比 0.95 的 .762/.689 高 **+4.3/+5.5 pt** —— 发布的 recipe 白扔 4-5 分。而 `top_p=1.00` 是灾难性的 .591/.573（**−17.1 pt**）。
2. **三个 confidence-based `alg` 完全打平**（entropy / maskgit_plus / topk_margin 全是 .762/.689，byte-identical，spread **0.0**）。掉到 `origin`（随机序）才崩到 .226。所以 "token-selection strategy 的选择" 值 **0 分**——真正的 53.6 pt 差异是「有无 confidence 排序」，不是策略之间的排名。
3. **syntax error 数与质量单调同向**：origin 56 → alg_temp0.5 37 → T=0.0 10 → T=0.2 3 → ref 0。低质量配置主要在**产出语法不合法的代码**。

### ★ 机制隔离（不只是测量）——纯 unmasking-order 效应

读 checkpoint 的 `generation_utils.py:sample_tokens()`：`temperature=0` 时 commit 的 token 是 `probs.max()`，而 `top_p` 只把尾部设成 `-inf`。**掩掉尾部不可能改变 argmax。** 所以 T=0 时 top_p **只能**改变「哪些位置被 unmask」的 entropy-confidence 排序。

实测验证（2000×500 logits）：top_p ∈ {0.9, 0.95, 0.99} 下 argmax **完全相同**，而 entropy-confidence 在 **100% 的位置**都变了、top-128 被选集合不同。

再测端到端：**T=0.0，top_p {0.90, 0.95, 1.00} → 3.0 HE / 4.9 HE+ pt spread**，而该区噪声地板**已证为精确 0**（dup run 确认）。**同样的 token、不同的揭示顺序，值约 5 个 HE+ 点。**

### 前作（agent 自己找的，不要当成首个 audit 来 frame）

- **CaRE**（arXiv:2607.24763v1）：发现「temperature 解释 MAUVE 方差的大部分」+ compute-matched 比较会翻转已发表排名。**仅 arXiv preprint**（按标题与作者查 OpenReview 无记录）。**它留的缺口**：MAUVE / OpenWebText+LM1B / base model / 7 种 *remasking* 策略——**无 execution-based pass@1、无代码 benchmark、无 instruct model**。我们这条是它的 execution-graded 补集。
- **"Re-evaluating Confidence Remasking in MDLMs"**：**ICML 2026 workshop poster**（venueid `ICML.cc/2026/Workshop/SPIGM`，OpenReview 已核）。范围是 remasking 方法，不是固定 NFE 下的 sampler 超参。
- **"Diffusion Language Models: An Experimental Analysis"**（arXiv:2606.19475v2）：变 denoising steps / context / block size / parallel unmasking，跨 8 个 DLM。**仅 arXiv。** 研究 inference 因素但不把 temperature×top_p×selection 敏感度当作 evaluation-validity 威胁。
- **DiffuCoder**（arXiv:2506.20639）已注明「提高采样温度不只多样化 token 选择、也多样化其生成顺序」——**直接支持我们的 order-channel 机制**，但它用来动机化 RL rollout，不是当 audit。

**诚实定位**：新的是 (a) execution-based pass@1 而非 MAUVE、在 instruct code model 上；(b) **order-only 隔离**且噪声地板证为零；(c) 发布 recipe 非最优这一观察。

### Caveat（agent 自己列的，必须随数字走）

- **只有 HumanEval+**。MBPP+ 未跑（每配置约 20min×8GPU，cross-plus 控制已吃掉两节点 ~6.5h）。单 benchmark、单 checkpoint —— **未证明可推广到 LLaDA 或 MBPP+**。
- **单模型家族**。跨家族（LLaDA-8B / DiffuCoder）未做。
- 随机区只有 T=0.7 做了 seed 重复；T=0.2/0.4 是单次，其约 2pt 差异需谨慎。
- `alg_temp=0.5` 与 `alg=origin` 应读作「坏区」而非调参选项——故另给 26.8 pt 的 plausible-only 数字，**论文里应以该数为 headline**。
- **过程诚实**：三个 cell 曾因两个 driver 写同一 shard 路径而作废。merge coverage 断言**每次都抓到**（`JSONDecodeError` → `MERGE FAILED`，未 grade 任何东西），逐个单节点重跑。**没有被污染的数字进表**——这正是用 `merge_evalplus_shards.py --expected 164` 而非 repo 里无覆盖断言的 `_merge_rank_solutions.py` 的原因。

**Commits**（`dllm_draft_104` on zwfy6）：`bfbaef58` → `439d0b7c` → `af41bf04`。

---

## ⛔ Retraction 7（2026-08-07 18:20）—— 最后一条存活 claim 也死了；k-span 方向终结

**被撤回的是 Retraction 6 里唯一存活的那条**：「两个范式都随 region 数退化，**AR 退化快约 2×**」。它死于我自己派出去**防御它**的那个实验。

Retraction 6 结尾我记下了一个 scope 缺陷：diffusion 臂拿 oracle per-hole 长度，AR 臂用自己的停止规则。我派 agent 补一个非 oracle 臂（DreamOn-v0-7B 原生 `infilling_with_expansion`，模型自己定长），消费**字节相同**的 frozen spec。结果符号翻转。

**MAIN 独立从 `score.json` 复算（`runs/kspan_diffusion_nonoracle` 415 cells）**：

平衡集（59 个全 k 都在的任务）：

| k | dreamon_nonoracle | diffusion (oracle) | AR-FIM |
|---|---|---|---|
| 1 | .915 | .898 | .949 |
| 2 | .864 | .847 | .898 |
| 3 | .780 | .831 | .780 |
| 4 | **.542** | .746 | .644 |

k=1→k=4 同任务掉幅：**non-oracle .373 > AR .305 > oracle .153**
平衡交互 vs AR：oracle **+0.153** → non-oracle **−0.068**（**符号翻转**）
agent 的任务聚类稳健估计量：`−0.435 (p=.0038)` → **`−0.009 (p=.943)`**

**结论：去掉 oracle 长度施舍后，diffusion 比 AR 退化得更快，不是更慢。** 「AR 退化快 2×」整条是 oracle artifact。**k-span infilling 在 HumanEval 上产不出任何 family-level 的可发表结论。**

**顺带三条实测**：
1. **没有 crash-out。** 415 个任务、每个 k 上 aborted holes 全为 0 —— 我预期的 DreamOn 多 hole `pad_delete_to_right` 崩溃**没有发生**。但 parseable 从 .963 掉到 .814（k=1→k=4），span 定长错误会累积。
2. **k=1 上非 oracle 反而更好**（.726 > .671 全量；.915 > .898 平衡）—— 与 pilot 早先那条「oracle 长度把生成推离分布」一致。给模型正确答案的长度是**负**收益。
3. **效率论点也输**：非 oracle DreamOn 的 `tokens_fed` 是 oracle Dream-Coder 的 1.5–2.4×、AR-FIM 的 **20–36×**。

**这个方向的账**：7 次撤回，其中 R4/R5/R6/R7 都是「补上正确对照后结论反转」，且 R6/R7 撤的是 MAIN 自己当天提出的论点。**dllm k-span 线终止。** 存活到最后的只有方法论副产品（见下方"能写的东西"），不是任何关于 diffusion 能力的实质结论。

详见 `KSPAN_NONORACLE_ARM.md`；commit `ff015b2`。

### 🔬 R6 根因更正（2026-08-07 21:10）—— 不是子集效应，是 grader 轴不匹配

**MAIN 在 R6 里把 k=1 的 .7638 vs .950 分歧归因于「n=60 子集对 AR 更友好」。那个归因是错的。** tournament workflow（107 agents）的 CROSSING-POINT audit 找到真因，MAIN 独立复算逐位确认：

**pilot 用 BASE-only 评分，R5 用 BASE+PLUS。** 用 R5 自己的 `qwen_fim/solutions.jsonl` 在同一批 k=1 hole 上两种口径重评（`evalplus 0.3.1`、`untrusted_check`、CPU）：

| 子集 | BASE-only | PLUS | 差 |
|---|---|---|---|
| 全部 164 eligible | **0.9024** | 0.7622 | +0.1402 |
| **前 40（AR pilot cap）** | **0.9500** | 0.8500 | +0.1000 |
| 前 60（diffusion pilot cap） | 0.9333 | 0.8167 | +0.1167 |

**`0.9500` 四位小数精确复现 pilot 报的 .950。** 所以那 0.19 的分歧里，**+0.14 是 grader 轴**，只有剩余部分是子集。R6 记的「选择效应就是全部」**过度归因了**——选择效应真实存在（k4-capable .898 vs dropped .543 那组数字仍成立、MAIN 已复核），但它不是 k=1 分歧的主因。

**为什么这条重要且必须留在文档里**：R6/R7 的结论（嵌套阶梯制造结果、diffusion 单调下降、oracle artifact）**全部不受影响**——它们都在 frozen spec 单一 `base` 轴内计算（六个臂 `which=base` 已核）。受影响的只有**「R5 与 pilot 为何不一致」这一条诊断**。但它是一个**跨口径比较**的教科书案例：两个都用官方 grader、都正确、却因为轴不同而不可比。这条比原来的子集解释更适合喂给 null-calibration 论文（task #170）——**它是「同一批程序、同一个官方 grader、两个轴、差 14 个点」**。

**tournament 对 CROSSING-POINT 的总评**：mean **2.83 / 26 个候选里排 17**，min 2.5。三个 judge 独立判它低。**MAIN 的先验没有被奉承**，这是我在 workflow prompt 里明确要求的。

**crossing-point workflow 对 R7 提出了一条正确的方法学批评**：R7 同时改了**两个变量**（模型 Dream-Coder→DreamOn **和** 长度供给 oracle→自选），所以「diffusion 退化更快」可被读成「DreamOn 只是更弱」。补了**同模型**对照——Dream-Coder-Instruct-7B + 固定 canvas（8/12 token），spec 与 frozen spec 的差异经 MAIN 逐行核实为**恰好 `hole_token_lengths` + 其派生量 `total_masked_tokens`，236 行无其他差异**。

平衡集（59 任务）MAIN 独立复算，**六个臂全部 `which=base`**：

| 臂 | 长度供给 | k=1 | k=4 | 掉幅 | k:arm vs AR |
|---|---|---|---|---|---|
| oracle | oracle | .898 | .746 | +0.153 | **+0.153** |
| **fix8**（同模型） | 固定 8 tok | .729 | .271 | **+0.458** | **−0.153** |
| **fix12**（同模型） | 固定 12 tok | .831 | .237 | **+0.593** | **−0.288** |
| DreamOn 非 oracle | 模型自选 | .915 | .542 | +0.373 | −0.068 |
| AR-FIM | 自身停止 | .949 | .644 | +0.305 | — |

**结论：R7 被加强，模型互换的逃逸口堵上。** 去掉 oracle 后**同一个模型**掉得比 AR 更狠（.458 / .593 vs .305），k:arm 从 +0.153 (p=.004) 翻到 ≈0 乃至负值。**是 oracle 长度施舍、而非 DreamOn 更弱，制造了全部表面优势。** 证据基础从一个跨模型臂扩到三个臂（其中两个同模型只改长度）。

⚠️ **MAIN 自己在派任务时的尺度错误**：我把 workflow 报的 `−0.148 / −0.017` 当成 per-arm 斜率，实为 `k:arm` **交互系数**。**把 −0.148 摆在 oracle 的 −0.346 旁边会把 fix8 的退化低估约 4×。** 两者不同标度，不可并列。

⚠️ **scope 限制（agent 主动标出，必须随数字一起报）**：固定 canvas 是**长度强制**的——98.8% 的 hole 恰好吐满预算，`masks_left=0` 处处成立，canvas 在 68%/88% 的 hole 上超过 gold。后果：parseability .966→.678 (fix8) / .915→.661 (fix12)，9.0%/13.4% 的 fill 含脚手架（`# Example usage`、裸 `print(`），**86%/96% 的行上 EM-to-gold 结构性不可能**。但 **0 个 chat special token、0 abort、0 grader error、0 空 fill** —— 不是完全的 chat-mode 崩溃。
**决定性子对照**：按 AR 自己的 `first_line()` 规则截断每个 fill → β_k = −0.769 / −0.895，k:arm = −0.012 (p=.95) / +0.114 (p=.50) → **退化不是溢出 artifact**。故这些臂对**退化速率**问题有效，对**非 oracle pass@1 绝对水平**无效。

**做这件事的理由不是复活 k-span**（该线仍终止），而是**那些方法学教训正在喂给 task #170 的 null-calibration 论文** —— 一个带双变量 confound 的撤回不是可用的教训。

详见 `KSPAN_NONORACLE_ARM.md` §9；commit `350c59f`。持久化：`runs/kspan_diffusion_fix{8,12}/` + `data/kspan/kspan_spec_v1_fix{8,12}.jsonl`（此前唯一干净的 de-oracle 对照只活在 `/tmp/kem/`，重启即死）。

## ⛔ Retraction 8（2026-08-07 23:40）—— AR 跨架构 claim 符号写反 + 评分轴用错

**被撤回的是 `SAMPLER_VARIANCE_DECOMPOSITION.md` §3 的三条**（都是同一个 bug 的下游）：

1. AR HE+ 数值 `.5000 / .5305`
2. 「AR 跨架构 gap 是 3.05 pt，**比** dLLM 的 2.44 pt **更大**」
3. 「两个模型都在 L20A 上掉 HE+」/「三个符号里有两个一致」

**三条全错。根因是评分轴**：`scripts/analyze_ar_crossarch.py` 里那句注释
`# In evalplus 0.3.1: 'plus_status' = 'pass' iff both base+plus tests pass`
**断言了一个不成立的约定**。EvalPlus 报的 plus pass@1 是**合取的**
（`base_status=='pass' AND plus_status=='pass'`）；单读 `plus_status` 会把
base 挂掉但 plus 过了的任务算进去（.252 上 1 个、.104 上 2 个），从而**虚高**。

**MAIN 与本 agent 各自独立从 `eval_results.json` 复算，逐位一致**：

| arm | 来源 | base | plus（合取，正确） | plus（单读 `plus_status`，错） |
|---|---|---|---|---|
| AR L20A (.252) | `outputs/ar_qwen25coder7b_base_252/humaneval/eval_results.json` | .5427 (89/164) | **.4939 (81/164)** | .5000 (82/164) |
| AR H20 (.104) | `runs/xnode/ar_control_h20_104/eval_results.json` | .5610 (92/164) | **.5183 (85/164)** | .5305 (87/164) |

**决定性证据**：.104 那个文件**自己记录的** `pass_at_k` 就是
`{'base': 0.5609756097560976, 'plus': 0.5182926829268293}` —— 与合取计数 85/164
= .5183 精确吻合，不是 .5305。**官方 grader 自己的输出一直在文件里，我们读错了轴。**

| 对比 | plus delta | flips | exact p |
|---|---|---|---|
| **AR**（L20A − H20） | **−2.44 pt** | 10 (n01=7, n10=3) | 0.3438 |
| **dLLM**（LOCAL − .73） | **+2.44 pt** | 12 | 0.3877 |

**符号相反、量级相等。**

**⚠️ 更正后的结论比原来更强，不是更弱** —— 这是本次撤回与 R4-R7 的关键区别：
- 「量级相等 + 符号相反」= **对称硬件噪声**，不是某个架构系统性更好。若两边同向，反倒要解释系统性偏置；对称正好指向非确定性 bf16 reduction order（机制见 crossnode §3），它没有理由偏爱任何一张卡。
- 因为 AR gap **恰好等于** dLLM gap（都 2.44 pt），「dLLM 对硬件不确定性异常敏感」这个 framing **在 pass@1 上彻底死了**。它**只在 bit level 存活**：dLLM solution-text 差 **75/164**（raw_output 128/164），AR 只差 **40/164** —— 约 2× 的 per-task 分歧率，与机制预测一致（512 步里每步 unmasking order 都是活的自由度）。**要直说：dLLM 跨架构确实明显更 bit-level 不稳定，而这份额外不稳定在 pass@1 上完全不显现。**

**同时更正的**：任何说 within-node 地板「精确为 0.0」的写法。准确版是
**pass@1 上 0.00 pt**（0 flips、四位小数相同）**但文本非 bit-identical** —— 即使同节点同配置，164 条里也有 2-3 条不同，因为 T=0.1 不是 greedy 且没设 seed。「精确 0.0 / provably zero」只适用于 **T=0.0** 那些 cell。

**修复**：`analyze_ar_crossarch.py` 改为合取轴、删掉那条错注释，并**新增 `check_axis()` 断言**——凡文件带 `pass_at_k` 就与我们的计数逐位比对，不符即 `AssertionError`。这个守卫本来就能抓到这个 bug（.5305 vs 记录的 .5183）。已跑通：`[axis check] ... match evalplus pass_at_k exactly`。

**发表前抓到**（数字从未离开本仓库，未进任何 `.tex`）。这是它与 R1-R3 的区别：R1-R3 是已写进结论后才发现。

---

## ⛔ Retraction 9（2026-08-07 23:40）—— MBPP+ rank-transfer 相关性由「坏 cell」抬起来的

**被撤回的是** `SAMPLER_VARIANCE_DECOMPOSITION.md` §4 把 **rho = 0.9379**（MBPP+ vs HE+ Spearman）当作「rank agreement 高」的裸引用，以及 §5 里「所以这条不在危险中」的判断。

**rho=0.9379 本身精确复现**（10 个共同 cell，MAIN 与本 agent 各自算）。问题是**它靠什么撑起来**：那 10 个里有 **3 个正是 `alg=origin` / `alg_temp=0.5`** —— 而 HE+ headline **自己把这两类当「坏区」排除**（§1 就是靠排除它们才得到 26.8 pt 的 plausible-only spread）。这 3 个 cell 在两个 benchmark 上都远低于其余，是高杠杆锚点，**制造了一个从业者永远享受不到的 rank agreement**。

限制到一个从业者真正会搜的区域：

| cell 集 | n | rho | exact permutation p |
|---|---|---|---|
| 全部共同 | 10 | 0.9379 | — |
| 仅 plausible | 7 | 0.8462 | — |
| **distinct plausible** | **5** | **0.6000** | **0.175 单侧 / 0.350 双侧** |

**7 → 5 的塌缩不是我挑的**：`entropy` / `maskgit_plus` / `topk_margin` 在参考 cell 上**逐字节相同**（HE+ 全是 `.7622/.6890`、MBPP+ 全是 `.6905`；即 §1 里「三个 confidence-based alg 完全打平」那条）。**它们是一个点，不是三个**；按三个算会同时虚高 n 和 rho。

**新 headline：rho = 0.60，n=5，不显著。** 唯一的好消息是 top-1 cell 两个 benchmark 一致。但**「sampler ranking 跨 benchmark 可迁移」这条目前在常规显著性水平上没有证据支撑**。

**不受影响的**：「sampler choice dominates method choice」**不依赖这个 rho** —— 它靠的是 HE+ 内部 spread（plausible-only 26.8 pt）对比已发表方法增益。**受影响的**只是那条*推广性*附注：我们现在不能用 rank correlation 证据声称该结论会带到第二个 benchmark。

**修复**：`analyze_sampler_variance.py` 现在同时打印三档（all / plausible / **distinct plausible** + exact permutation p），并在裸 rho 后面直接标 `<-- DO NOT QUOTE BARE`，把 byte-identical 塌掉的 cell 显式列出。

**⚠️ 一处口径更正（本 agent 复核时发现，MAIN 的 brief 未标明单双侧）**：`0.1750` 只在**单侧**下成立；**双侧是 0.3500**（`scipy.stats.spearmanr` 的渐近近似给 0.285 双侧）。文档两侧都写明，避免下次被当成双侧引用。

**发表前抓到**，且不影响任何存活结论——只影响一条推广性 rider。

---



**被撤回的是 MAIN 16:00 左右提出的"crossing point"框架**：「diffusion vs AR 谁赢不是标量事实，而是沿 hole 数 k 的一个交叉点；k=1 AR 赢，k=4 diffusion 赢；所有前作只采样了一侧」。它建立在 pilot 的 +0.525 交互（n=60/n=40，`/tmp` 探针）之上。全量复现后**撤回**，按 `4051556` 里 16:32:34 预注册的规则执行（该规则在 16:49:04 的去污染结果产出**之前**提交，时序已核实）。

**死因：k 各档的任务集是嵌套的（k=4 的 59 个任务是 k=1 的 164 个的严格子集，逐层皆然）。** 只有长函数容得下 4 个非相邻 hole，所以 k 越大 ladder 就越是在**丢掉自己的难题**，而丢掉的恰恰是 diffusion 最差的那批。MAIN 独立从 `score.json` 重算，确认：

| 任务集（k=1，同协议同任务） | diffusion | AR |
|---|---|---|
| k=4 能做到的（n=59） | .898 | .949 |
| 中途掉队的（n=105） | **.543** | **.819** |

**在全 4 层都存在的 59 个任务上，diffusion 是下降的，不是上升**：

| | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| diffusion | .898 | .847 | .831 | .746 |
| AR-FIM | .949 | .898 | .780 | .644 |
| AR-FIM-fair | .949 | .915 | .831 | .644 |

交互量级三级衰减：pilot **+0.525** → 全量 naive **+0.297**（SE .096, z=3.10, p=.0019）→ **平衡后 +0.153**（MAIN 复算；agent 的任务聚类稳健估计量给 −0.435/−0.208，方向一致）。

**存活的（弱于原论点，但是真的）**：**两个范式都随 region 数退化，AR 退化快约 2×**。经 fairness repair 后仍成立。
**撤回的**：「diffusion 有主场」/「存在交叉点」/「k=4 时 diffusion 赢」。

**三条 MAIN 没预料到的**：
1. **MAIN 传下去的 recipe 本身有缺陷**。`prompt + canonical_solution + suffix` 对 **123/164** 个任务重构出**两个不同文件**（`L0` 行 prompt 多一个尾部换行）。"164/164 可 parse"为真但不蕴含唯一，故 MAIN 指定的 `row['prompt'].count('\n')` 只在该行自身变体下正确。已改为字节级精确准入（910/1033 admitted，零猜测索引）。**k=4 上限是 59 不是 60** —— 无任何单一一致 spec 能复现 `[164,108,84,60]`，MAIN 传的数字是两种构造混出来的。
2. **去污染本身是发现**：标识符重命名令 diffusion .671→.544、AR .866→.606 —— **HumanEval infilling 数字有很大一块是表面记忆**。且交互同时缩水（−0.435→−0.208），说明原始 family gap 里也含记忆成分。
3. **MAIN 预测的 confound 方向反了**。MAIN 警告"diffusion EM 随 k 上升"是致命 confound；实测 EM **下降**（.579→.424）且与 AR 每档几乎相同。基于 pilot 数字的那个判断是错的。

**另一个复用的坑（pilot 与本次都中过）**：raw mutated-gold null 被**不可变异的 gold 行**抬高，且抬高幅度本身随 k 变（43%→90%）。真 null 是 `.043/.025/.014/.000`，不是 pilot 报的 `.800/.433/.200`。

**遗留 scope 缺陷（若成文，审稿人必问）**：diffusion 用 oracle per-hole 长度、AR 用自己的停止规则 —— 这个不对称从 pilot recipe 继承而来，利于 diffusion，与交互正交但需补非 oracle diffusion 臂。

详见 `KSPAN_INFILLING_RESULTS.md`；commits `4051556` + `d16bf51`。

## ⚠️ 撤回明细

### Retraction 4（2026-08-07，AR baseline）—— 主 finding 被杀

**证据**（`.104` 的 `da24eb6` / `ad748b1`）：
- AR 模型：**Qwen2.5-Coder-7B base**（15GB HF 下载）。`config.json` 与 Dream-Coder-v0-Base-7B **架构完全一致**（同 hidden 3584 / 28 layers / 28 heads / 4 KV heads / intermediate 18944 / vocab 152064 / rope_theta 1e6），仅多一个 mask_token_id 151666 —— 这是真正的 matched lineage 对照。
- Grader：`evalplus.evaluate` 官方，8/378-shard 完整跑通，零 generation error。空程序 negative control → pass@1 = **0.000**（Retraction 1 那种"空程序满分"的失效模式已排除）。
- 两个成本口径**都测量**（不是估计）：`tokens_fed`（每次 forward 的 input_ids 宽度累加）和 `attended_context_sum`（每次 forward 的总序列长度累加）。`forward_pre_hook` 实测 vs 闭式预测**在 1084 个 task-run 上 100% 一致**。

| config | HE+ | MBPP+ | `tokens_fed` (mean) |
|---|---|---|---|
| **AR greedy** | **.524** | **.648** | 204 / 101 |
| **AR T=0.1 p0.95** | **.518** | **.651** | 204 / 101 |
| Dream-Coder-Instruct nfe512 | .707 | .680 | ~350,832 / 308,995 |
| **scaffold Medium** | .177 | .354 | 13,980 / 7,080 |

**AR vs scaffold Medium（两个口径都算）**：
- `tokens_fed` 口径：AR 便宜 **70×**（HE+）/ **70×**（MBPP+），且质量高 **+.341** / **+.294**
- `attended_context_sum` 口径：AR HE+ 稍贵 1.44×、MBPP+ 便宜 0.87×，但**仍然质量高 ~+0.30**
- **两个口径上 AR 都严格支配 scaffold Medium**（Pareto dominance）

**AR vs Dream-Coder-Instruct nfe512**：
- `tokens_fed` 口径：AR 便宜 **1719× / 3062×**，质量差 −.189 / −.029
- `attended_context_sum` 口径：AR 便宜 **17.5× / 50.3×**，质量差同上
- **MBPP+ 上：AR 用 50× 更少的 attended cost 只差 2.9 个点** —— 这是 diffusion 效率论点最弱的地方，恰恰是它 published 数字最亮的地方

**Retraction 4 独有的**：这不是我 harness 里的 bug（前三次都是）。这是真的当有合适 baseline 时结果就变了。我之前的 finding 反映的**只是 diffusion-vs-diffusion 的相对位置**，不是任何绝对贡献。

### Retraction 1（verifier 从不比对返回值）

`scripts/refine_verifier_guided.py` 的 `sandbox_exec` 执行 `_ = entry_point(*args)` 后**丢弃返回值**，把"没有抛异常"当作 pass。空 solution 得 7/7 满分。修：改用 `evalplus.eval.untrusted_check`。

### Retraction 2（restart policy 丢弃 scaffold draft）

`restart` 从 prompt 重新生成，完全不看 scaffold draft，所以 grid 测的是 Dream-Coder-Instruct，不是 scaffold。"smaller+refine dominates" 和 "capacity-invariant rescue rate" 均撤回。

### Retraction 3（`process` vs `failure_process` 成本低估）

runtime 把正常任务写 `process`，触顶截断写 `failure_process`。只读前者 → 漏掉最贵的任务。scaffold Large 成本被低估 **2.85× / 3.69×**。修：统一 `proc()` accessor。

---

## 现在能写的东西（诚实版，2026-08-07 15:00）

**能写的（可能）**：
1. **"NFE 不是跨家族公平的成本单位"**（685 vs 181 tok / step，差 3.8×）—— 方法学观察，独立于我们的具体方案
2. **"Diffusion-LM 论文常缺 matched AR 控制；补上后效率论点大幅缩水"** —— 审计型贡献。AR 用 1719×/3062× 更少 `tokens_fed` 换 −0.19/−0.03 质量差；MBPP+ 上 attended 口径 50× 便宜换 2.9 点差。这是这个领域一个待补的空白。
3. **"评测协议敏感度可能大于方法增益"**：`top_p` 1.0→0.95 换来 +14.6 点 HumanEval，大于多数 diffusion 论文声称的方法增益（audit 正在 .82 上跑，会有更强证据）

**不能写的**：
- ❌ "结构化 runtime 独占低成本前沿"（Retraction 4，AR 便宜 70× 且质量高 0.30）
- ❌ "两种范式占据 Pareto 曲线不同段"（同上，AR 支配 scaffold Medium）
- ❌ 任何以 scaffold vs Dream 为独立 finding 的写法（都只是 diffusion 家族内的相对位置）

**下一步的关键问题**：
- 领域里有多少论文的核心 finding **也**是"只和 diffusion baseline 比"？如果这个模式普遍，"缺 AR 对照"本身就是一个可写的审计论文。这条 .82 的 sampler audit + .104 的 AR baseline **合起来**可能是我们最有把握的方向。
- Diffusion 到底在什么 setting 下能真正打过 matched AR？目前实测过的都不是。infilling（.73 在跑）是最有希望的候选，因为 AR 结构上做不了双向。

---

## Retraction 5（2026-08-07，infilling 主场也被 AR 支配）

**目的**：既然 from-scratch 是 AR 主场，去测 diffusion 主场（infilling/FIM）。`.73` agent 用**同一个 Qwen2.5-Coder-7B**（有原生 FIM 预训练）在 HumanEval-SingleLineInfilling（n=1033）上比 3 个 diffusion 模型 + 2 个单向对照。

| arm | ctx | span | pass@1 | tokens-fed | 备注 |
|---|---|---|---|---|---|
| **qwen_fim (AR native FIM)** | prefix+suffix | 自己 | **.7638** | **239** | 官方 FIM sentinel |
| dreamon_oracle | prefix+suffix | oracle 长度 | .7590 | 5827 | 给了长度上界 |
| dream_fim | prefix+suffix | oracle | .7115 | 2035 | |
| dreamon_fim | prefix+suffix | self | .7018 | 4923 | 真实设定 |
| qwen_prefix (AR L→R) | 只 prefix | 自己 | .5324 | 249 | 单向对照 |
| dream_prefix | 只 prefix | oracle | .4124 | 1739 | |

**结论**：
1. **双向上下文的收益 diffusion 和 AR 幅度相当**（同模型加 suffix：diffusion +.299 vs AR +.231）—— 这是**任务框架**的性质，不是 masked diffusion 独有。我此前把双向上下文当 diffusion 独占优势是错的。
2. **DreamOn 在自己主场 .122 → .702**：主场存在，但**仍打不过 AR**（qwen_fim .764，用 8.5× 更少 tokens）。
3. **AR 在两个原本被认为是 diffusion 主场的 setting 里都赢**（from-scratch + infilling）。

**agent 顺便发现的两个真 bug**：
- `mask_expansion=True` / `delete_eos_token=True` 在 DreamOn 代码里被 `**kwargs` 静默吞掉，**根本不是参数**。此前所有 DreamOn 数字（HE+ .122 / MBPP+ .085）**不是"开了 mask_expansion 的"**——虽然结论上仍是"DreamOn 在 from-scratch 上弱"，但 doc 里任何提到"开了 mask_expansion 才这个分"的措辞都要撤。
  > **2026-08-12 更新（C2）**：这条已由**执行**证实，不再只是读代码 —— `update()` 实测原样返回
  > 两个 key 作为 unused，属性前后都不存在。**并且这条 bullet 自己的结论也要撤一半**：
  > "结论上仍是 DreamOn 在 from-scratch 上弱" 已被 C4 推翻 —— 那个"弱"是 `initial_masks=8`
  > 的 canvas artifact，改成 32 之后 DreamOn 追平/反超 scaffold。
- Span-recovery bug（我 .73 agent 自己修）：DreamOn 从 .608 → .702，pre-fix 数字作废。

**Commit**：`eedd0075` in `dllm_draft_104` (zwfy6 checkout)。

---



本文档 2026-08-07 02:00 的初版有三条 headline finding，**其中两条已撤回，一条改写**。撤回原因是两个独立的方法学错误，都由我自己在后续诊断中发现。

### Retraction 1 — verifier 从不比对返回值

`scripts/refine_verifier_guided.py` 的 `sandbox_exec` 执行 `_ = entry_point(*args)` 后**丢弃返回值**，把"没有抛异常"当作 pass。后果：

- HumanEval/0 的**空 solution** 被判 7/7 pass（docstring-only stub 返回 None 不抛异常）
- 旧 verifier 虚报的 visible-pass 率：`scaffold_medium_heplus` 77%（真值 20.7%）、`scaffold_large_mbppplus` 72%（真值 38.6%）、`scaffold_tiny_heplus` 41%（真值 0.0%）
- 于是**真正会 pass 的题和垃圾输出被一起路由到 `keep`（不 refine）**，而最终 pass@1 由 evalplus 正确评分 → 报出的 Δ 完全由"多少题被路由"驱动

**修复**：改用 evalplus 官方 grader `evalplus.eval.untrusted_check`（带 `expected` 比对）。
**验证**：修复后 verifier 的 visible-pass **精确等于** evalplus base pass@1（HE+ tiny 0.000 / HE+ medium 0.207 / MBPP+ large 0.386，三点全中）。
**Commit**：`dllm_draft@67a7a01`。16 个坏 run 归档为 `runs/refine_*.BROKEN_VERIFIER`。

### Retraction 2 — `restart` policy 丢弃 scaffold 输出，grid 测的是 Dream 不是 scaffold

`restart` policy 从 **prompt 重新生成**，完全不看 scaffold draft。所以在 routed 任务上，输出是**纯 Dream-Coder-Instruct**，与 tier 无关。判定性诊断（修复后数据）：

| tier | routed | 该 run 在 routed 上 hidden-pass | **tiny run 在同一批任务上** | diff |
|---|---|---|---|---|
| HE+ tiny | 164 | 0.524 | 0.524 | +0.000 |
| HE+ small | 157 | 0.490 | 0.510 | −0.019 |
| HE+ medium | 130 | 0.485 | 0.492 | −0.008 |
| HE+ large | 132 | 0.470 | 0.492 | −0.023 |
| MBPP+ tiny | 378 | 0.577 | 0.577 | +0.000 |
| MBPP+ small | 309 | 0.528 | 0.537 | −0.010 |
| MBPP+ medium | 224 | 0.455 | 0.469 | −0.013 |
| MBPP+ large | 232 | 0.470 | 0.487 | −0.017 |

差异全部在 −0.008 ~ −0.023，即采样噪声。**scaffold tier 在这个 grid 里唯一的作用是决定哪些题交给 Dream**（tier 越强 → kept 越多 → 交给 Dream 的越少）。

### 三条 finding 的最终状态

| 旧 finding | 状态 |
|---|---|
| ① "低 NFE Pareto 属于 structural runtime" | ✅ **存活**（baseline 数据全程由 evalplus 评分，未受任何 bug 影响） |
| ② "Rescue rate 78-80% capacity-invariant" | ⚠️ **改写**：真值 52-66%（HE+ mean 54.5% sd 2.7%，MBPP+ mean 58.1% sd 6.4%）。但 Retraction 2 说明这个"invariance"本身是 tautology —— restart 后大家都变成同一个 Dream，pass 率当然接近 |
| ③ "Smaller-capacity + refine Pareto-dominates larger + refine" | ❌ **撤回**。修复后单调性反转（HE+ .524/.512/.561/.555，MBPP+ .577/.587/.624/.614，越大越好），且 Retraction 2 说明整个比较无效 |

**"Verifier-guided refinement 把 tier variance 压平 0.28×/0.13×" 也不能作为 finding** —— 那是 Retraction 2 的算术必然，不是 refinement 的性质。

---

## 1. ✅ 存活结果：Baseline 复现

全部由 `evalplus.evaluate` 直接评分，**未经过任何自写 verifier**。

| Model | HE | HE+ | MBPP | MBPP+ | vs paper |
|---|---|---|---|---|---|
| **Dream-Coder-Instruct-7B** | **.762** | **.707** | **.807** | **.680** | HE −6.7；**MBPP +1.1 超过论文** ✓ |
| Dream-Coder-Base-7B | .085 | .079 | .193 | .159 | ⚠️ 未复现（`lm_eval` 路径未走，见 §5） |
| DreamOn-v0-7B（`initial_masks=8`；**注：`mask_expansion` 从未生效，见 C2**） | .140 | .122 | .093 | .085 | 新测量（论文未在此 benchmark 发布） |

> ⚠️ **2026-08-12（C1/C3/C4）**：上面这一行是 **`initial_masks=8`** 下的数字，**不是 DreamOn 的能力上限**。
> 把 canvas 改成 32（其余全冻结）→ HE+ **.2134**、MBPP+ **.3545**；再修 HE+ stitch bug（C3）→
> HE+ **.2561** @c32、**.4817** @c128。原行名写的 "+mask_expansion" 是错的（C2：该 kwarg 被静默吞掉）。
> 本表的 HE+ .122 本身也被 stitch bug 低估约 0.6 pp。

**sampler 敏感性强**：Instruct top_p 1.0 → 0.95 换来 HE .616 → .762（**+14.6 pt**）。round-1 数字保留在 `runs/*.r1/` 作 ablation。

---

## 2. ✅ 存活结果：NFE sweep，两个 benchmark 都在 512 饱和

### HumanEval+（n=164）

| NFE | HE | HE+ | ΔHE+ | syntax_err |
|---|---|---|---|---|
| 64  | .018 | .018 | — | 154/164 (93.9%) |
| 128 | .152 | .140 | +12.2 | 112/164 (68.3%) |
| 256 | .579 | .555 | +41.5 | 33/164 (20.1%) |
| 512 | .762 | .707 | +15.2 | 3/164 (1.8%) |
| **1024** | **.762** | **.707** | **0.0** | 2/164 (1.2%) |

### MBPP+（n=378，2026-08-07 补齐）

| NFE | MBPP | MBPP+ | ΔMBPP+ | syntax_err |
|---|---|---|---|---|
| 64  | .032 | .029 | — | 337/378 (89.2%) |
| 128 | .251 | .233 | +20.4 | 200/378 (52.9%) |
| 256 | .643 | .545 | +31.2 | 53/378 (14.0%) |
| 512 | .807 | .680 | +13.5 | 6/378 (1.6%) |
| **1024** | **.807** | **.680** | **0.0** | 6/378 (1.6%) |

**两 benchmark 完全同型（这是曲线形状的独立复现）**：
- 低 NFE 区收益 = **结构合法性**（syntax-error 从 ~90% 降到 ~15%）
- 中 NFE 区收益 = **语义正确性**
- **512→1024 两侧都是精确 0.0 收益**（HE+ .707→.707，MBPP+ .680→.680），syntax_err 也不再降 → 饱和点一致

---

## 3. ✅ 存活结果：scaffold capacity ladder（两个 benchmark，n=164 / n=378）

scaffold ckpt = `scaffold_sft_stage1 global_step_4465`，从远端 .82 跨盘 `scp -O` 29GB 到 LOCAL。

> **★ 2026-08-12 — 本节 scaffold 数字已被【独立重算】确认（0 GPU）。**
> 此前 A05 把 `.177`/`.354` 记为 "READ from summary table，single round，29GB ckpt wzc1-only"，
> 并称之为**整个方向里 provenance 最弱的一环**。现已在 `.73`（zwfy6，另一个物理盘）用
> **该 run 自己存下的逐题程序**重新评分（evalplus，每次调用自检 canonical PASS / stub FAIL）：
>
> | Tier | HE+ 已发表 | HE+ 重算 | MBPP+ 已发表 | MBPP+ 重算 |
> |---|---:|---:|---:|---:|
> | Tiny | .000 | **.0000** | .000 | **.0000** |
> | Small | .043 | **.0427** | .156 | **.1561** |
> | **Medium** | **.177** | **.1768** ✓ | **.354** | **.3545** ✓ |
> | Large | .177 | **.1768** ✓ | .325 | **.3254** ✓ |
>
> **不需要搬那 29GB checkpoint**：`scripts/generate_evalplus_scaffold.py` 写的是
> `solution = result.text`、**无任何后处理**，所以 `runs/scaffold_*/solutions.jsonl`
> **就是被评分的那个 artifact**（不是它的近似），重新评分 = 重跑评分那一步。共搬 1.6 MB，md5 已核。
> 附带结论：scaffold **完全不经过** C3 那个 stitch（0/8 个 cell 命中 buggy 分支；
> scaffold Medium 的 HE+ 程序 162/164 本身就含 top-level `def`，即使路由过去也会短路），
> 所以 `.177`/`.354` **没有被低估**。
> 仍未解决：**只有单轮**，seed variance 未测（重新评分修的是 provenance，不是方差）。
> 判定见 `Mixture-of-Memory/proposal/archive/A05-structural-dllm-cost-frontier/A05_BLAST_RADIUS_AND_SCAFFOLD_VERDICT.md`。

### HumanEval+（n=164）

| Tier | 参数（depth / lines / tokens） | HE | HE+ | NFE mean | depth_hit | gen_err |
|---|---|---|---|---|---|---|
| Tiny | d=1, lines=2, tok=2 | .000 | .000 | 15.9 | 52/164 (32%) | 52 |
| Small | d=2, lines=4, tok=8 | .043 | .043 | 42.6 | 51/164 (31%) | 51 |
| **Medium** | d=4, lines=16, tok=32 | **.207** | **.177** | 63.8 | 0 | 2 |
| Large | d=16, lines=128, tok=512 | .195 | .177 | 57.3 | 0 | 17 (model_call_budget) |

### MBPP+（n=378）

| Tier | MBPP | MBPP+ | NFE mean | depth_hit | gen_err |
|---|---|---|---|---|---|
| Tiny | .000 | .000 | 33.9 | 69/378 (18%) | 69 |
| Small | .183 | .156 | 37.4 | 58/378 (15%) | 58 |
| **Medium** | **.407** | **.354** | 56.7 | 0 | 0 |
| Large | .386 | .325 | 47.4 | 0 | 35 |

**termination 检查（确认 Medium 不是被预算掐断的）**：HE+ medium `resolved 162 / depth_exhausted 2`；MBPP+ medium `resolved 378/378`。Large 则有 17（HE+）/ 35（MBPP+）个 `model_call_budget` 截断。

### ★ Large budget falsifiability（2026-08-07，@512 / @1024 / @4096 三档）

**⚠️ 本节 15:20 修订**：我 15:00 提交的 "预算翻倍买到字面意义的零收益、这些任务结构性无法收敛" 措辞**被 .252 上的 8× 预算实验部分推翻**。修正如下。

把 `--max-model-calls` 从 512 加倍到 1024（LOCAL，两 benchmark 全量），再把 MBPP+ 的 35 个被截断任务单独跑 4096（.252）：

| | pass@1 (base/plus) | budget_hit | mean cost (全部) | mean cost (被截断) |
|---|---|---|---|---|
| HE+ @512 | .195 / **.177** | **17** | 34,659 | 229,250 |
| HE+ @1024 | .195 / **.177** | **17** | 64,650 | 518,572 |
| MBPP+ @512 | .386 / **.325** | **35** | 21,588 | 175,835 |
| MBPP+ @1024 | .386 / **.325** | **35** | 43,649 | 414,094 |
| **MBPP+ @4096**（35 格重跑 + 确定性 343 合并成 378） | .402 / **.339** | **0** | 44,471 | 422,973 |
| MBPP+ Medium（参照） | .407 / **.354** | 0 | 7,080 | — |

**512→1024 确实零收益，两 benchmark 都是**：pass@1 四位小数不变，**失败任务集合逐位相同**（集合相等性检验 `True`），唯一变化是被截断任务烧掉 2.26× token。

**但 8× 预算揭示了原因，我原来的解释是错的**：@4096 下 **35/35 全部收敛**，且收敛点密集落在 **NFE 1035–1080**（实测 min 1035 / max 1080 / median 1049）——**刚刚越过 1024**。所以"1024 没用"不是因为这些任务无法收敛，而是因为**翻倍差了那么一点点**。

**然而预算仍然不是解法**：
- 救回 35 个任务只换来 **+1.4pp**（.325 → .339），**仍然低于 Medium 的 .354**
- 代价是这些任务每个 ~423k token（约为平均任务的 20×）
- 即 Large 的赤字**部分是预算受限的**，但买下这些任务要 ~8× 算力，**买完还是赢不了 Medium**

**最终措辞（这次不外推）**：
> Large 在 token-cost Pareto 前沿之外。它相对 Medium 的赤字**部分**来自 call-budget 截断（MBPP+ 上 8× 预算能让全部 35 个截断任务收敛），但提高预算不是修复手段——收敛这些任务需要约 8× 算力，且收敛后 .339 仍低于 Medium 的 .354（成本还是 Medium 的 6×）。

**benchmark-specific，不可外推**：HE+ 的 17 个任务在 @1024 下**仍然全部截断**，而 MBPP+ 的 35 个在 @4096 下全部逃出。**HE+@4096 未跑**，所以不知道 HE+ 是否也会逃出。"逃逸行为"是 benchmark 相关的，不能从单一 benchmark 推广。

**关于成本口径（Retraction 3 的坑在这里咬得最狠）**：只读 `process` 会把 MBPP+ @512 低估 **3.69×**、@1024 低估 **7.46×**（5,848 vs 43,649）。@4096 那一档因为零截断，两种读法比值恰好 1.00×。

**统计强度**：+1.4pp = 378 题里的 5 题，**未算置信区间**，只能当 suggestive。

`figures/fig_termination_tail.png` 是这一段的图。

---

## 3b. ★ Token-cost-matched Pareto（存活 finding 的正确成本口径）

**NFE 不是公平的成本单位**：vanilla 每个 diffusion step 对整个 `(prompt + 512 canvas)` 做一次 forward（HE+ 685 tok/step，MBPP+ 604 tok/step）；scaffold 每次 model call 只看当前 tree serialization，序列短得多（HE+ tiny 实测 2875/15.9 ≈ 181 tok/call ≈ prompt 173 + 小 canvas）。

**⚠️ Retraction 3（2026-08-07 12:00，我自己在做图时发现）**：runtime 把**正常结束**的任务 telemetry 写进 `process`，把**触顶截断**的任务写进 `failure_process`。之前只读 `process` 求均值 → **静默漏掉被截断的任务，而那些恰好是最贵的**（1024 NFE、单任务 ~533k tok）。修正幅度：

| run | 只读 process | 全部任务 | 比值 | 漏掉 |
|---|---|---|---|---|
| HE+ Large | 12,156 | **34,659** | **2.85×** | 17 |
| MBPP+ Large | 5,848 | **21,588** | **3.69×** | 35 |
| 其余 6 个 tier | — | — | 0.93–1.15× | ≤69 |

**这与「NFE 当成本单位」是同一类错误**。已修 `scripts/make_pareto_figures.py`（统一 `proc()` accessor + `field_coverage()` 报告每个均值由多少任务贡献），commit `8451e93`。

### HumanEval+（成本 = 累计喂给模型的 token；已修正）

| config | cost (tok) | HE+ | Pareto frontier |
|---|---|---|---|
| scaffold Tiny | 2,734 | .000 | ✓ |
| scaffold Small | 9,729 | .043 | ✓ |
| vanilla nfe16 | 10,964 | **.000** | |
| **scaffold Medium** | **13,980** | **.177** | **✓** |
| vanilla nfe32 | 21,927 | **.000** | |
| scaffold Large | **34,659** | .177 | ✗（同质量、2.5× 成本） |
| vanilla nfe64 | 43,854 | .018 | |
| **plain-SFT nfe64** | 43,854 | **.000** | |
| vanilla nfe128 | 87,708 | .140 | |
| plain-SFT nfe128 | 87,708 | .049 | |
| vanilla nfe256 | 175,416 | .555 | ✓ |
| vanilla nfe512 | 350,832 | **.707** | ✓ |
| plain-SFT nfe512 | 350,832 | .220 | |
| vanilla nfe1024 | 701,665 | .707 | |

### MBPP+（已修正）

| config | cost (tok) | MBPP+ | Pareto frontier |
|---|---|---|---|
| scaffold Tiny | 3,590 | .000 | ✓ |
| scaffold Small | 4,685 | .156 | ✓ |
| **scaffold Medium** | **7,080** | **.354** | **✓** |
| vanilla nfe16 | 9,656 | **.000** | |
| vanilla nfe32 | 19,312 | **.008** | |
| scaffold Large | **21,588** | .325 | ✗（更差质量、3.0× 成本） |
| vanilla nfe64 | 38,624 | .029 | |
| vanilla nfe128 | 77,249 | .233 | |
| vanilla nfe256 | 154,497 | .545 | ✓ |
| vanilla nfe512 | 308,995 | **.680** | ✓ |
| plain-SFT nfe512 | 308,995 | .243 | |
| vanilla nfe1024 | 617,989 | .680 | |

**★ 关键新增：matched Plain-SFT 对照（这是「结构有用」vs「这个模型练得更多」的分水岭）**
`BASELINE_STATS.md` 记录的 plain-SFT 用**同一个 Dream-Coder Base ckpt、同一 train/eval split、同一 corruption 加权、同一 trainer**，唯一区别是**去掉 scaffold 机制**（无 meta token / typed hole / 模板 / expand-delete）。它是 flat canvas，所以按 vanilla 计价。

**结果：plain-SFT 在两个 benchmark 上一个 frontier 点都占不到。** 最锐利的一格 —— plain-SFT nfe64 花 43,854 tok 拿 **.000**，而 scaffold Medium 花 13,980 tok（**1/3 成本**）拿 **.177**。所以低成本段的优势**不能**用"scaffold 只是被多训了"解释。

**Falsifiability tests（全部通过）**：
1. **低成本 vanilla 点**（nfe16/nfe32 专门加来反驳）：HE+ .000/.000，MBPP+ .000/.008 —— vanilla 在 ≤22k tok 一题都做不对
2. **matched trained control**（plain-SFT）：frontier 上 0 个点
3. **成本口径**：从 NFE 改为 token 后结论仍成立（且 Retraction 3 修正后 Large 掉出前沿，方向对我更不利但结论不变）

**结论（诚实版）**：
- **低成本段属于 structural runtime**：scaffold Medium 独占 HE+ ≤14k / MBPP+ ≤7.1k 区间；vanilla 和 matched plain-SFT 在该区间都 ≈ 0
- **高质量段完全属于 flat diffusion**：.707 / .680 是 scaffold 达不到的（峰值 .177 / .354）
- **两段之间没有交叉点**
- **capacity 在 depth≈4 达到最优后是净损失**（不是 plateau）：Large 在两个 benchmark 上都**严格劣于** Medium

---


## 4. ⚠️ 需要重新设计才有意义：verifier-guided refinement

修复后的完整 4×2 grid 数字（`runs/refine_*`，evalplus-graded，`newly_lost = 0` 恒等）：

| bench | tier | base+ | refine+ | Δ | kept | routed | rescue/routed |
|---|---|---|---|---|---|---|---|
| HE+ | tiny | .000 | .524 | +.524 | 0 | 164 | 57.9% |
| HE+ | small | .043 | .512 | +.469 | 7 | 157 | 54.8% |
| HE+ | medium | .177 | .561 | +.384 | 34 | 130 | 53.8% |
| HE+ | large | .177 | .555 | +.378 | 32 | 132 | 51.5% |
| MBPP+ | tiny | .000 | .577 | +.577 | 0 | 378 | 65.6% |
| MBPP+ | small | .156 | .587 | +.431 | 69 | 309 | 61.2% |
| MBPP+ | medium | .354 | .624 | +.270 | 154 | 224 | 52.2% |
| MBPP+ | large | .325 | .614 | +.289 | 146 | 232 | 53.4% |

**这些数字为什么不能当 finding**：`restart` 丢弃 scaffold draft（Retraction 2），所以 refine+ 列本质是"Dream-Coder-Instruct 在 routed 子集上的成绩 + kept 子集的原成绩"的加权和。Δ 越大只说明 base 越差、routed 越多。

**要让这条线成立，必须换成真正 condition 在 draft 上的 policy**：
- `remask` policy 已实现但**同样是 proxy**（把 prev solution 截断后拼进 prompt 文本，不是真的在 canvas 上 remask）—— 需要改成真正的 canvas-level remask
- **Arm C（typed subtree collapse）** 才是原本设计里唯一真正利用 scaffold 结构的臂：用 scaffold runtime 的 typed tree state 只坍缩**失败的子树**回 mask，再局部 re-diffuse。**未实现**
- 正确的对照必须包含 **"Dream-Coder-Instruct 直接 nfe512 单跑"（HE+ .707 / MBPP+ .680）作为上界** —— 现在 grid 里最好的 .624 远低于它，说明这套 pipeline 目前**没有超过直接把题交给 Dream**

**修复后 kept-set 里仍有少量 false-pass**（visible pass 但 hidden fail）：HE+ medium 5、large 3；MBPP+ small 10、medium 20、large 23。这是 visible-vs-hidden test gap 的真实体现，不是 bug——但报告任何 rescue 数字时必须同时报这一列。

---

## 5. 未解决 / 待查

1. **Dream-Coder-Base-7B 未复现**（旧 `--no-chat` 路径 HE .085 vs 论文
   .665）。现已确认旧 harness 仍使用 instruction wrapper，并只评分生成
   suffix；官方 lm-eval 实际使用 raw function prefix、`add_bos_token=true`
   并评分 `prompt + continuation`。先运行 16-task protocol smoke，门槛通过
   后再做 full HE+/MBPP+，因此当前仍定性为**"协议尚未对齐"**。
2. **DreamOn 低分是预期**（论文只在 HumanEval-Infilling / SantaCoder-FIM 上评，不做 from-scratch 生成）。
3. **单模型家族**（Dream-Coder + scaffold_sft_stage1）。DiffuCoder-7B 未下载，跨家族复现待做。
4. **Arm C 未实现**（见 §4）。这是 SotA-scan subagent 判定"无直接前作、threat 1.5/5"的那一臂，也是唯一能救活 refinement 方向的设计。
5. **iter2 数字全部作废**（旧 verifier 产出）。修复后未重跑。

---

## 6. 已修的 5 个 bug（全部自主发现）

| bug | 症状 | 修法 |
|---|---|---|
| **verifier 从不比对返回值** | 空 solution 得满分；两条 finding 撤回 | 改用 `evalplus.eval.untrusted_check`（`67a7a01`） |
| `CUDA_VISIBLE_DEVICES=$g + LOCAL_RANK=$g` | shard 1-7 全挂 `invalid device ordinal` | `LOCAL_RANK=0`（CUDA_VISIBLE_DEVICES 后 torch 只见 1 卡） |
| `refine_verifier_guided.py` 硬编码 HumanEval+ | MBPP+ refinement 跑 0 rescue | 加 `--dataset` |
| `sandbox_exec` json.dumps 撞 complex | shard5 死于 `TypeError: complex not serializable` | 改 `repr()`（该函数后被整体替换） |
| bash heredoc 嵌套引号 | `merge_evalplus_shards` 内联 python 崩 | 抽 `scripts/_merge_rank_solutions.py` |

---

## 7. 与调研（两个 subagent）对接

- **Length-elastic 被 DreamOn (ICLR 2026 Poster) 吃掉** ✓（本轮 DreamOn 复现证实其擅长 infilling 不是 from-scratch gen）
- **SpecRef (preprint) 写了"syntactic scaffold 0→>20%"动机** —— 我们原想用 Tiny→Tiny+refine 量化，但 Retraction 2 后这条不成立
- **Verifier-guided typed AST subtree collapse 无直接前作**（threat 1.5/5）——Arm A (restart) 已证明是错的臂，Arm B (remask) 是 proxy，**Arm C 待写**

---

## 8. 产物清单

**Scripts**（新写/改）：
- `scripts/refine_verifier_guided.py`（核心 harness，**已修 verifier**）
- `scripts/_merge_rank_solutions.py`
- `scripts/_run_baselines_{wzc1_8gpu,r2_wzc1}.sh`
- `scripts/_run_nfe_sweep_wzc1.sh`
- `scripts/_run_refine_two_arms.sh` / `_run_refine_extension.sh` / `_run_mbpp_refine_chain.sh`
- `scripts/_run_scaffold_medium_wzc1.sh` / `_run_capacity_ladder_wzc1.sh` / `_run_capacity_refine_chain.sh`
- `scripts/_run_capacity_ladder_mbpp_wzc1.sh` / `_run_capacity_refine_mbpp_chain.sh` / `_run_mbpp_full_grid.sh`
- `scripts/_run_refine_grid_fixed.sh`（修复后重跑全 8 格）
- `scripts/generate_evalplus_dream.py`（加 `--top-p` / `--no-chat`）
- `scripts/generate_evalplus_dreamon.py`（加 `mask_expansion=True` + `delete_eos_token=True` + T/top_p → 论文值）

**数据**：`data/evalplus/{humaneval,mbpp}_plus.jsonl`（evalplus API 导出）
**模型**：`models/Scaffold-v0-stage1-7B/`（29GB，7 shards，从 .82 跨盘 scp -O）
**运行**：`runs/` 下 34 个 run dir（其中 16 个 `.BROKEN_VERIFIER` 归档，勿引用）

**Commits**：`97595c8` → `f9067ba` → `16bd339` → **`67a7a01`（verifier fix）** → `a6d7e85`

---

## 9. 可写的东西（诚实版）

**一条能写，且经过 3 道反驳测试**：
> **"Structural runtime 与 flat masked diffusion 占据 quality–compute Pareto 曲线的不同段，且两段之间无交叉点。"**
>
> 证据（token-cost-matched，见 §3b，两 benchmark 独立复现）：
> - 低成本段：**vanilla 在 ≤22k tok 全部 ≈ 0**（nfe16 .000/.000，nfe32 .000/.008），scaffold Medium 在 13.98k/7.08k tok 拿 .177/.354
> - **matched trained control**：plain-SFT（同 base ckpt / 同 split / 同加权 / 同 trainer，仅去掉 scaffold 机制）在两 benchmark 的 frontier 上**一个点都占不到**；nfe64 花 43.8k tok 拿 **.000**，而 scaffold Medium 花 1/3 成本拿 .177 → 低成本优势**不是**"练得更多"
> - 高质量段：vanilla nfe512 的 .707 / .680 scaffold 达不到（峰值 .177 / .354）
> - capacity 在 depth≈4 后是**净损失**：Large 严格劣于 Medium（同/更差质量 + 2.5×/3.0× 成本 + ~10% 任务预算翻倍仍跑不完）
> - NFE 曲线两 benchmark 同型，512→1024 精确 0.0 收益

**图表（`figures/`，全部由 `scripts/make_pareto_figures.py` 从 runs/ 直读生成）**：
`fig_pareto_heplus` / `fig_pareto_mbppplus` / `fig_nfe_saturation` / `fig_termination_tail` + `pareto_points.json`。

**三条不能写**（见 §0 与 §3b）：verifier bug、restart policy 丢弃 draft、`process`-only 成本低估。

**要救 refinement 方向，下一步必须做的**：
1. 实现 **Arm C**（typed subtree collapse，真正 condition 在 scaffold tree state 上）
2. 加 **"Dream nfe512 直跑" 上界对照**（HE+ .707 / MBPP+ .680）—— 任何 pipeline 不超过它就没有价值
3. `remask` 改成真正的 canvas-level remask，而非 prompt 文本拼接
4. **所有效率论断都要用 token-cost 口径，且必须包含被截断的任务**（`proc()` accessor，见 Retraction 3）
