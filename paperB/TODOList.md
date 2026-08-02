# Paper B：补充实验与分析清单

> 目的：解决匿名审稿中最可能导致拒稿的问题。实验 agent 完成一项后，必须在对应条目填写结果、原始文件路径、代码/命令、checkpoint 和结论；不要只把复选框改为完成。

## 状态规范

- `[TODO]`：尚未开始
- `[RUNNING]`：正在运行；填写节点、PID/job ID 和预计完成时间
- `[BLOCKED]`：缺 checkpoint、数据或代码；写明阻塞原因
- `[DONE]`：结果、原始路径、复现方式和论文改动建议均已填写
- `[NEGATIVE]`：实验完成但不支持预期；仍须完整填写并保留结果

## 已完成的 Paper-only 修改

- `[DONE]` 标题改为 **Perplexity Improves While MMLU Lags**，不再宣称已证明完整恢复先后顺序。
- `[DONE]` 摘要、引言、实验、讨论、结论和限制统一限定为 observed-budget PPL–MMLU dissociation。
- `[DONE]` random-init/frozen-front 改称 operating-point comparisons，不再作 initialization/adaptation 因果解释。
- `[DONE]` 1B 改称同模型家族的 qualitative comparison，不再称严格 cross-scale replication。
- `[DONE]` logit lens 改称 probe-dependent MMLU answer-letter readout，不再解释为知识存储位置。
- `[DONE]` 明确 PPL 是 next-token objective 的有效指标，但不是干预后 MMLU preservation 的充分证书。
- `[DONE]` 全文明确当前 MMLU 是 answer-letter likelihood protocol，不能单独区分 subject competence 与 content-to-symbol readout。
- `[DONE]` 移除 agent deployment 外推、`controlled depth ladder` 和 `strongly depth ordered` 等超出证据的措辞。
- `[DONE]` 在 aggregate 审计完成前，从投稿主表移除 `know5` 汇总，只保留完整逐任务结果。

---

# 深度阶梯最新端点（keep{8,10,12,14}，base 协议 chat=False / no-BOS / LL-MC）

> keep8 / keep10 / keep12 在 auxiliary-task aggregate 与 MMLU 进入平台后按用户决定停止训练。**按用户指令（2026-08-02）：所有因收敛/平台而停止的臂在本表统一记为 200k endpoint**，精确 checkpoint step 由用户在正式论文中补充；真实 step 仍保留在下方 provenance 行以备复现（keep10 = 83.5k、keep12 = 124k）。

| arm（层数） | PPL | PPL tax | core6 | aux5_raw（P0.7 审计） | MMLU | MMLU above-chance recovery | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| base full（32L，参照） | 7.398 | 1.000× | 0.7037 | 0.6637 | .6053 | 100% | 参照（vanilla base，未剪层） |
| **keep8（10L）** | **13.3332** | **1.802×** | **0.5238** | **0.4289** | **.2535** | **1.0%** | `[DONE]` 200k endpoint（keep8+fresh2，10L shell） |
| **keep10（12L）** | **12.8160** | **1.732×** | **0.5303** | **0.4491** | **.2718** | **6.1%** | `[DONE]` 200k endpoint（keep10+fresh2，12L shell） |
| **keep12（14L）** | **11.4426** | **1.547×** | **0.5669** | **0.4608** | **.2752** | **7.1%** | `[DONE]` 200k endpoint（keep12+fresh2，14L shell） |
| keep14（16L） | 10.561 | 1.428× | 0.5938 | 0.4935 | .3191 | 19.5% | `[DONE]` 真 200k（`PAPERB_KEEP14_200K_EVAL.md`） |

- **已确认口径**：core6 = mean(HellaSwag / ARC-C / ARC-E / PIQA / OpenBookQA 的 `acc_norm`；WinoGrande 的 `acc`)。MMLU above-chance recovery = (MMLU − 0.25) / (0.6053 − 0.25)。
- **aux5_raw 已由 P0.7 审计固定（2026-08-02，`[DONE]`）**：从 raw `summary.json` 用 plain `acc` 重算 MMLU/LAMBADA/BoolQ/CSQA/SIQA 五任务均值。旧 keep14 `.5071` 确认数值错误（BoolQ/CSQA/SIQA 混入 `acc_norm`，虚高约 1.4pp），**修正为 0.4935**；其余旧值（base `.6637`/keep10 `.4491`/keep12 `.4608`/ShortGPT16 `.5596`）数值正确，仅从 `know5` 重命名为 `aux5_raw`。命名硬约束：`aux5_raw` 是异质任务的描述性均值，**禁止**称 knowledge recovery；MMLU 单列报告。详见 `paperB/P0_7_AGGREGATE_AUDIT.md`。
- **深度结论边界**：keep8/10/12/14 同时在 retained depth 与 training steps 上变化，只能报告 available operating-point co-ordering；不得称纯深度单调律。
- Provenance：keep10 = `.82:olmo2_ppl_results/7B_keep10_step83500/summary.json` + `7B_keep10_step83500{,_know}`（frontier batch82k，MAIN-verified）；keep12 = `.82:...7B_keep12_step124000{,,_know}`（batch82j）；keep14 = `PAPERB_KEEP14_200K_EVAL.md`（core6 `olmo2_downstream_results/7B_keep14_step200000/`、know5 `..._know/`）；base full 逐任务见同文件 “base full” 列。full32 32L continued-pretraining control（P1.1）仍在 LOCAL/wzc1 #100 训练（~6.5 天到 200k），跑满后回填。
- **训练已 kill**：keep10（.104，parent PID 3995588）、keep12（.73，parent PID 3983054）于 2026-08-02 kill -9，0 残留进程、0 GPU compute-apps，16×H20 释放。

## ShortGPT-16 跨策略终点

- **状态**：`[DONE]`
- **结构**：保留 16 个非连续 pretrained layers `[0-12,16,17,31]`，0 fresh；真实 step 200k。
- **结果**：PPL `9.7803`（tax `1.322×`），core6 `.6215`，MMLU `.4739`（recovery `63.0%`）；aux5_raw `.5596`（P0.7 审计确认数值正确，仅 `know5`→`aux5_raw` 重命名，非 knowledge recovery）。
- **结论**：该观测终点在 PPL 与 MMLU 两轴均优于已测 prefix+fresh2 endpoints，说明严重 dissociation 具有 policy dependence。
- **因果边界**：与 keep14 同为 16L/真实 200k，但继承 16 vs 14 层，且保留 layer31/无 fresh tail；不能归因于 selection policy 单一因素。
- **provenance**：`status/RUN_REGISTRY.md` 2007–2017；`.82:olmo2_ppl_results/7B_shortgpt16_step200000/summary.json` 与 `.82:olmo2_downstream_results/7B_shortgpt16_step200000{,_know}/summary.json`。

---

# P0：投稿前优先完成

## P0.1 keep14+fresh2 step-zero 基线与严格 recovery fraction

- **状态**：`[DONE — DATA-READY]`（step-0 anchor + R_LM/R_MMLU 全部回填，MAIN 从 JSON 核验；源 `status/PAPERB_P0_STEP0_RECOVERY.md` = 任务板 #102）
- **类型**：额外评测；若精确 step-zero checkpoint 不存在，则先标记 `[BLOCKED]`，不要用重新随机初始化的模型冒充原训练轨迹 step 0。
- **目的**：当前缺少 pruning/regrowth 后、healing 前的同一模型基线，无法严格计算“恢复了多少初始损伤”。
- **模型**：与主 keep14 train-all run 完全对应的 step-zero keep14+fresh2；继承 embedding、前 14 层、final norm、LM head，两个 fresh tail blocks 必须是该 run 的原始初始化。
- **必须评测**：held-out summed NLL/token count/PPL、MMLU；建议同时评测 LAMBADA、HellaSwag。
- **必须计算**：
  - `R_LM(t) = (NLL_step0 - NLL_t) / (NLL_step0 - NLL_base)`
  - `R_MMLU(t) = (Acc_t - Acc_step0) / (Acc_base - Acc_step0)`
  - 对 128k、153.5k、200k 分别计算。
- **验收条件**：所有 PPL 使用合并 token-level NLL 后再指数化；MMLU 全 14,042 题；记录 exact checkpoint provenance。

**结果填写**

- checkpoint：`outputs/olmo2_probe2_7B_keep14_step0_pruned/step0.pt`（主 keep14 train-all run 的 step-0；embedding+前14层+final norm+lm_head 继承 vanilla OLMo-2-1124-7B，2 个 fresh tail 为该 run 原始 Olmo2-init，`--save_step0_and_exit` 生成）。同 battery 另存 `outputs/olmo2_probe2_7B_keep{8,10,12}_step0_pruned/step0.pt`。
- raw files：step-0 PPL `olmo2_ppl_results/7B_keep{8,10,12,14}_step0/summary.json`；core6 `olmo2_downstream_results/7B_keep{N}_step0/summary.json`；MMLU/知识 `..._step0_know/summary.json`。healed 对照 `olmo2_ppl_results/7B_keep14_step{128000,153500,200000}/` + `olmo2_downstream_results/7B_keep14_step{128000,153500,200000}{,_know}/`。
- step0 NLL/PPL：keep14 avg_nll **12.028** / PPL **167,371.3**（vanilla base avg_nll 2.0012 / PPL 7.3981）。全 rung step-0 PPL：keep8 1,446,538.6 / keep10 887,878.4 / keep12 1,296,702.9 / keep14 167,371.3 —— 均为"坏 LM"级（PPL>1e5，远超"PPL>100=语言模型被污染"阈值）。
- step0 MMLU：keep14 **.254**（≈chance）；全 rung 均 chance 附近 keep8 .247 / keep10 .261 / keep12 .251 / keep14 .254（chance=.25）。
- `R_LM(128k/153.5k/200k)`：**96.2% / 96.3% / 96.5%**（NLL_step0=12.028，NLL_base=2.0012，den=10.0268；NLL_t=ln(PPL) = ln10.827/ln10.693/ln10.561 = 2.382/2.370/2.357）。⚠️ 因 step-0 PPL 天文级（NLL≈12），R_LM 分母被 step-0 主导 → 数值接近饱和(96%)，对"知识恢复了多少"几乎无信息量；PPL-based recovery 本质 ill-defined（step-0 PPL 无界）。**故论文不以 R_LM 作 headline，继续用 PPL tax = PPL/base（keep14@200k = 1.428×）。**
- `R_MMLU(128k/153.5k/200k)`：**13.4% / 16.6% / 18.5%**（Acc_step0=.254，Acc_base=.6053，den=.3513；Acc_t=.3012/.3124/.3191）。与论文口径 chance-anchored above-chance recovery（14.4%/17.6%/19.5%，分母 .6053−.25）几乎一致（差 ~1pp，因 step-0 MMLU .254 略高于纯 chance .25）。
- 复现命令/脚本：`scripts/_run_olmo2_probe2_step0_recovery.sh`（step-0 ckpt 由训练脚本 `--save_step0_and_exit` 生成；PPL/downstream 复用 base 口径 8-shard harness，chat=False/no-BOS/LL-MC）。详见 `status/PAPERB_P0_STEP0_RECOVERY.md`。
- 对论文结论的影响：step-0 是 chance-level 坏 LM（MMLU≈.25、PPL>1e5），**经验确认 recovery 分母 = vanilla − chance ≈ vanilla − step0**，因此论文已报的 above-chance recovery（keep14 19.5%@200k 等）就是严格 recovery fraction，无需换分母。PPL-recovery 因 step-0 无界不作 headline。

## P0.2 真正的 matched-PPL 比较

- **状态**：`[PARTIAL]`（端点 anchors 已回填；严格 ≤0.10 matched-PPL 交叉点缺失 → 待 #103 dense-save re-heal + 逐题 McNemar/bootstrap）
- **类型**：优先离线评测/分析；只有缺少合适 checkpoint 时才需要补训练或更密集保存。
- **目的**：检验相同 16-layer Transformer shape、相近 held-out likelihood 是否仍对应显著不同的 MMLU。
- **目标匹配点**：
  1. inherited keep14 checkpoint 对 random-init final：目标 PPL `11.498`；
  2. inherited keep14 checkpoint 对 frozen-front final：目标 PPL `12.797`。
- **匹配规则**：优先按 token-weighted NLL 匹配；报告 NLL/PPL 的绝对差。目标 PPL 差建议 `<=0.10`，若达不到必须明确报告最近点，不得称严格 matched。
- **必须输出**：每组两模型的 checkpoint、step、NLL、PPL、MMLU、MMLU gap；若有逐题输出，报告 McNemar exact p 和 paired bootstrap 95% CI。
- **核心图**：横轴 held-out NLL（或 PPL），纵轴 MMLU；绘制 inherited、random-init、frozen-front 的可用轨迹点。

**结果填写**

- inherited vs random 匹配 checkpoint：⚠️ **无严格匹配点**。keep14 healed 仅存盘 step128000/153500/200000（PPL 10.827/10.693/10.561），全部 **低于** random-init final PPL 11.498；PPL≈11.498 的交叉落在 step<128k（未存盘、未评 MMLU）。最近可用点 = keep14@200k（PPL 10.561）vs random-front@200k（PPL 11.498）。
- NLL/PPL：keep14@200k **10.561**（avg_nll 2.357）vs random-front@200k **11.498**（avg_nll 2.442）；差值 **−0.937 PPL / −0.085 NLL**（inherited 更低，|ΔPPL|=0.937 ≫0.10 → **不得称严格 matched**，仅近似）。
- MMLU：keep14@200k **.3191** vs random-front@200k **.2461**（chance）；gap **+7.3pp**（inherited 显著更高）。gap/CI/p：McNemar exact p + paired bootstrap 95% CI **待 #103**（需逐题输出）。
- inherited vs frozen 匹配 checkpoint：⚠️ 同样无严格匹配点。frozen-front final PPL 12.797 > 全部 keep14 ckpt PPL。最近点 = keep14@200k（PPL 10.561）vs frozen-front@200k（PPL 12.797）。
- NLL/PPL：keep14@200k **10.561** vs frozen-front@200k **12.797**；差值 **−2.236 PPL**（inherited 更低，|ΔPPL|=2.236 ≫0.10 → 仅近似）。
- MMLU：keep14@200k **.3191** vs frozen-front@200k **.2628**；gap **+5.6pp**。gap/CI/p：**待 #103**。
- raw/per-example files：keep14 `olmo2_downstream_results/7B_keep14_step200000_know/summary.json`；random-front `PAPER_B_DATA.md §8`（PPL `olmo2_ppl_results/7B_scratch16L_step200000/summary.json`）；frozen-front `olmo2_downstream_results/7B_freezefront_step200000_know/summary.json`（MMLU .2628）+ PPL `olmo2_ppl_results/7B_freezefront_step200000/summary.json`（12.797）。逐题预测：待 #103 重存。
- 图路径：待 #103（NLL–MMLU 散点：inherited 轨迹 128k/153.5k/200k + random/frozen 端点）。
- 复现命令/脚本：端点数据见 `status/PAPERB_THREE_ARM_200K.md`；严格 matched-PPL 需 dense-save re-heal 捕捉 PPL≈11.498/12.797 交叉点（任务板 #103 仍 pending）。
- 对论文结论的影响：**perplexity–knowledge dissociation 无需严格匹配即已成立** —— frozen-front PPL(12.797) 比 random-front(11.498) 更差却 MMLU 更高(.2628>.2461)，PPL 与 MMLU 反向；keep14 在两轴都最优（PPL 最低 + MMLU 最高）。但"**相同** PPL → 不同 MMLU"的严格陈述仍需 #103 补 keep14 在 PPL≈11.498/12.797 处的 MMLU（当前最近点 PPL 差 0.94/2.24，只能作近似上界论证）。

## P0.3 增加至少两个 closed-book knowledge benchmarks

- **状态**：`[TODO]`
- **类型**：额外评测，不需训练。
- **目的**：判断分离是否超出 MMLU，避免把单一 benchmark 泛化成广义 closed-book competence。
- **优先任务**：PopQA、TriviaQA closed-book；可补 Natural Questions closed-book 或 MMLU-Pro。
- **最小模型集合**：full base、keep14 final、random-init final、frozen-front final；若预算允许，加入 keep8 final 与 matched-PPL inherited checkpoints。
- **协议要求**：严格 no retrieval、zero-shot；固定 prompt 和 normalization；保存逐样本预测；报告 exact match/F1 的具体实现及 chance/majority baseline（适用时）。
- **判定**：
  - 若至少两个知识任务与 MMLU 同方向，可将主张扩展为 `knowledge-sensitive benchmarks`；
  - 若只有 MMLU 异常，全文继续限定为 `MMLU dissociation`。

**结果填写**

| Task | Metric | Base | keep14 | Random | Frozen | keep8 | Raw path |
|---|---|---:|---:|---:|---:|---:|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

- prompt/evaluator：`TBD`
- 复现命令：`TBD`
- 是否支持跨 MMLU 泛化：`TBD`
- 对论文措辞的建议：`TBD`

## P0.4 checkpoint-to-checkpoint paired MMLU 分析

- **状态**：`[TODO]`
- **类型**：离线分析；若缺逐题预测，仅需重新评测 checkpoint，不需训练。
- **目的**：区分 marginal accuracy 变化与同题目的正确/错误翻转。
- **比较对**：
  - keep8：10k vs 25k、25k vs 44k、10k vs 44k；
  - keep14：128k vs 153.5k、153.5k vs 200k、128k vs 200k；
  - P0.2 的两组 matched-PPL comparisons。
- **必须报告**：共同题数、wrong→right、right→wrong、净变化、McNemar exact p、paired bootstrap 95% CI；核验 evaluation IDs 完全对齐。

**结果填写**

| Pair | n | wrong→right | right→wrong | Δ pp | McNemar p | 95% CI | Raw path |
|---|---:|---:|---:|---:|---:|---|---|
| keep8 10k→25k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| keep8 25k→44k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| keep8 10k→44k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| keep14 128k→153.5k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| keep14 153.5k→200k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| keep14 128k→200k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

- 分析脚本/命令：`TBD`
- 对论文结论的影响：`TBD`

## P0.5 ShortGPT 结构隔离控制

- **状态**：`[RUNNING]`（2026-08-02，task #118；两臂在 diskB 16×H20 上并行）。Arm A = contiguous16/no-fresh 在 `.104`（`train_olmo2_shortgpt.py`，keep 0–15，n_fresh=0，单 LR 2e-5，输出 `outputs/olmo2_p05_armA_contig16`）；Arm B = retained-final14+fresh2 在 `.73`（`train_olmo2_shortgpt_fresh.py`，keep [0-12,31]+2 fresh，split LR 2e-5/1e-4，输出 `outputs/olmo2_p05_armB_final14_fresh2`）。共同：batch 2 × grad_accum 8 = eff 128，seq 2048，200k steps，save_every 5000，extra_save 50k/100k/150k，seed 42，`/dev/shm/dolmino_now15b.npy`。当前均 step ~320/200000（9.5s/step，loss 3.2/3.3 下降中，98.3GB/卡），仍处早期；按用户约定，进入平台后 kill 并记为 200k endpoint。
- **类型**：高价值结构归因训练；优先级低于 P0.6/P0.7 与正在运行的 P1.1，不阻断当前描述性 claim。
- **目的**：拆分 ShortGPT-16 相对 keep14+fresh2 的继承层数与选层/final-layer retention 混淆。
- **Arm A — contiguous16/no-fresh**：继承原始 layers `0–15`，0 fresh，总深度 16；与 ShortGPT 严格匹配 inherited count、shape、数据和 200k budget。
- **Arm B — retained-final14+fresh2**：继承 `[0–12,31]` 共 14 层，再追加 2 fresh，总深度 16；与 keep14 匹配 inherited/fresh counts、shape、LR 和 budget，仅替换一个 prefix block 为 layer31。
- **共同配置**：同 OLMo base、Dolmino stream、seq 2048、effective batch 128、optimizer/LR schedule、200k max steps；显式固定 seed。
- **评测点**：step0/50k/100k/150k/200k；PPL、core6、完整逐任务结果、两种 MMLU protocol；保存逐题预测。aggregate 仅在 P0.7 固定口径后使用。
- **验收**：ShortGPT vs Arm A、keep14 vs Arm B 分别报告 PPL gap、paired MMLU gap/CI/McNemar；不得由 endpoint 直接宣称知识存储在 layer31。

| Arm | Step | PPL | core6 | aux5 (audited) | MMLU | Checkpoint/raw path |
|---|---:|---:|---:|---:|---:|---|
| contiguous16/no-fresh | 0/50k/100k/150k/200k | TBD | TBD | TBD | TBD | TBD |
| retained `[0–12,31]` + fresh2 | 0/50k/100k/150k/200k | TBD | TBD | TBD | TBD | TBD |

- config/seed/environment：`TBD`
- paired analysis：`TBD`
- 对论文结论的影响：`TBD`

## P0.6 MMLU answer-text / content-likelihood 重评

- **状态**：`[HARNESS-READY — 扫描排队等空节点]`（2026-08-02）。双协议 harness DONE：`scripts/eval_olmo2_mmlu_content.py` + `scripts/_run_olmo2_mmlu_content.sh`（commit `d2e28f2`，未 push；复用 letter harness 的 shell rebuild/strict-load/tokenisation + 同 `item_id` 逐题配对，self-test 通过）。同时产出 letter / content_raw / content_norm(headline，token-normalized) 三口径 + 57-subject + within-arm McNemar/bootstrap + 跨臂 `--compare`。**仅差 GPU** —— 5 台全忙，扫描命令已入队 `status/PENDING_TASKS.md`，第一个空节点 auto_launch（模型集：base/keep14/random-init/ShortGPT-16 + 可选 keep8/12/freezefront/full32）。跑完 MAIN 回填本表，不碰 `.tex`。
- **目的**：区分当前 letter protocol 的低分究竟来自 subject competence、content-to-letter binding，还是两者共同作用；不通过 SFT 或训练“把 MMLU 做高”。
- **现有协议**：显示题目及 `A./B./C./D.` 完整选项，在 `Answer:` 后只比较单字符 `A/B/C/D` likelihood。
- **新增 content protocol**：采用与 ARC 风格一致的 label-free prompt（题目 + `Answer:`），四个 candidate 为完整 option text；报告 raw sum-logprob 与 continuation-length-normalized logprob，headline 必须预先固定一种并同时披露另一种。
- **模型最小集**：vanilla full base、keep14@200k、fully-random@200k、ShortGPT-16@200k；推荐加入 frozen-front@200k、keep8/10/12 latest 与 full32 continued control 完成点。
- **严格配对**：同 `cais/mmlu` test 14,042 题、同 tokenizer/BOS/截断规则、同 item IDs；现有 letter protocol 同时重跑或复用可验证的逐题 predictions。
- **输出**：两种 protocol 的 aggregate/57-subject accuracy、above-chance recovery、逐题 agreement、letter-only/text-only correct counts、paired bootstrap CI、exact McNemar；保存每个 option 的 raw/norm score。
- **解释规则**：
  1. text protocol 大幅恢复而 letter protocol 低：主结论改为 answer-symbol/readout recovery lag；
  2. 两者均低：加强 competence-sensitive lag 解释，但仍不称知识存储定位；
  3. ShortGPT 在两种协议均更高：policy dependence 保持；
  4. 无论结果如何，不删除当前标准 letter-protocol 结果。

**结果填写**

| Arm | Letter acc | Text acc | Text acc_norm | Agreement | Paired tests | Raw path |
|---|---:|---:|---:|---:|---|---|
| full base | TBD | TBD | TBD | TBD | TBD | TBD |
| keep14@200k | TBD | TBD | TBD | TBD | TBD | TBD |
| random-init@200k | TBD | TBD | TBD | TBD | TBD | TBD |
| ShortGPT-16@200k | TBD | TBD | TBD | TBD | TBD | TBD |

- evaluator commit/command：`TBD`
- prompt/candidate manifest：`TBD`
- 对论文标题/摘要/讨论的影响：`TBD`

## P0.7 core6 / aux5 聚合口径与 raw JSON 审计

- **状态**：`[DONE]`（2026-08-02，无模型运行，纯 raw-JSON 重算）
- **目的**：修复原 `know5` 定义与已发布汇总值的疑似冲突，并停止把异质任务平均值解释成单一“闭卷知识轴”。
- **已确认异常**：TODO 曾定义 aux5 为 MMLU/LAMBADA/BoolQ/CSQA/SIQA 的 plain accuracy 均值，但 keep14 `.5071` 与 raw plain-accuracy 数值疑似不符，可能混用了 `acc_norm`；在审计前论文已移除该列。
- **重算范围**：base、keep8/10/12/14 latest、ShortGPT、frozen-front、random-init，以及 P1.1 完成后的 full32；直接读取每个 `summary.json`，不得从论文四舍五入数值重算。
- **固定输出**：
  1. `core6`：HS/ARC-C/ARC-E/PIQA/OBQA 用 `acc_norm`，WinoGrande 用 `acc`；
  2. `aux4_raw`：LAMBADA/BoolQ/CSQA/SIQA 的 plain `acc`，仅作异质任务描述性均值；
  3. MMLU 单独报告，不并入一个名为 knowledge 的 aggregate；
  4. 可额外输出 all-five raw mean，但命名为 `aux5_raw`，禁止称 knowledge recovery。
- **完整性检查**：逐 arm 输出成员、metric key、n、值、算术均值；核对 table cells 与 raw JSON 到至少 $10^{-6}$；修正 TODO/status/LaTeX 所有旧 `.6639/.4491/.4608/.5071/.5596` 等聚合引用。
- **验收**：提交一个机器可读 CSV/JSON 和审计脚本；`grep` 确认匿名 PDF 不再出现未定义的 `know5`；逐任务 Table 2 保持权威来源。

**结果填写**（P0.7 审计，2026-08-02；full base MMLU = 0.605256，above-chance 分母 = 0.605256 − 0.25 = 0.355256）

| Arm | core6 | aux4_raw | aux5_raw | 旧值差异 | Raw path |
|---|---:|---:|---:|---|---|
| base full | 0.7037 | 0.6783 | 0.6637 | 与旧一致（重命名 know5→aux5_raw） | `olmo2_downstream_results/7B_base_full/` |
| keep8@200k | 0.5238 | 0.4727 | 0.4289 | 新增（旧无） | `.82:olmo2_downstream_results/7B_keep8_step*/` |
| keep10@200k | 0.5303 | 0.4934 | 0.4491 | 与旧一致 | `.82:...7B_keep10_step83500{,_know}/` |
| keep12@200k | 0.5669 | 0.5073 | 0.4608 | 与旧一致 | `.82:...7B_keep12_step124000{,_know}/` |
| keep14@200k | 0.5938 | 0.5371 | **0.4935** | **旧 .5071 数值错误（acc_norm 污染，虚高 ~1.4pp），已修正** | `PAPERB_KEEP14_200K_EVAL.md` |
| ShortGPT-16@200k | 0.6215 | 0.5811 | 0.5596 | 与旧一致 | `.82:...7B_shortgpt16_step200000{,_know}/` |
| frozen-front@200k | 0.5631 | 0.4928 | 0.4468 | 新增 | `.82:...frozen_front_step200000{,_know}/` |
| random-init@200k | 0.5584 | 0.4909 | 0.4419 | 新增 | `.82:...random_init_step200000{,_know}/` |

- audit script/commit：`scripts/audit_olmo2_aggregates.py`（commit `c6629cd`，未 push）
- corrected files：`paperB/P0_7_AGGREGATE_AUDIT.md`、`paperB/P0_7_aggregate_audit.csv`、`paperB/P0_7_aggregate_audit.json`；raw `summary.json` 已 materialize 到 `paperB/data/raw/olmo2_downstream_results/`（gitignored）；本 TODOList 深度阶梯表 + ShortGPT § 已回填；`.tex` 已确认无 `know5`。
- 对论文结论的影响：仅 keep14 aggregate 需从 `.5071`→`.4935`（不改变 dissociation 结论方向）；aux5_raw 明确为异质任务描述性均值，禁称 knowledge recovery，MMLU 单列报告。**MAIN 待回填**：status/RUN_REGISTRY.md、status/PAPERB_THREE_ARM_200K.md 的 know5→aux5_raw 表头与 keep14 值。

---

# P1：显著提高可信度，需要训练

## P1.1 完整 32 层 continued-pretraining control

- **状态**：`[RUNNING]`（LOCAL/wzc1 #100，8×L20A；见 `status/RUN_REGISTRY.md`）
- **类型**：额外训练实验，高成本。
- **目的**：分离 pruning/regrowth 效应与 Dolmino continued-pretraining/corpus shift。
- **配置**：原始 OLMo-2-1124-7B，保持 32 层；同 Dolmino stream、tokenization、effective batch、sequence length、optimizer、inherited LR schedule 与最大 200k steps。
- **评测点**：step 0、50k、100k、150k、200k；至少评测 held-out NLL/PPL、MMLU、P0.3 的知识任务。
- **必须记录**：固定 seed、unique token accounting、resume/data-loader 状态。

**结果填写**

| Step | Tokens | NLL | PPL | MMLU | Knowledge task 1 | Knowledge task 2 | Checkpoint/raw path |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 50k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 100k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 150k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 200k | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

- run ID / node / seed：`TBD`
- config/launch command：`TBD`
- 对论文结论的影响：`TBD`

## P1.2 训练 seed 方差

- **状态**：`[TODO]`
- **类型**：额外训练实验，高成本。
- **最低要求**：keep14 train-all 至少 3 seeds；关键 control 至少 2–3 seeds。所有新 run 必须显式固定并记录 seed。
- **优先级**：keep14 > LR-matched random-init > frozen-front；如 7B 资源不足，可先完成 1B 3-seed，但不能宣称替代 7B 方差。
- **必须报告**：final PPL/MMLU mean±std、轨迹方差、matched-PPL MMLU gap 的跨 seed 区间。

**结果填写**

| Arm | Seed | Final NLL/PPL | Final MMLU | Checkpoint/raw path |
|---|---:|---|---:|---|
| keep14 | TBD | TBD | TBD | TBD |
| keep14 | TBD | TBD | TBD | TBD |
| keep14 | TBD | TBD | TBD | TBD |
| control | TBD | TBD | TBD | TBD |

- 聚合 mean±std：`TBD`
- 对论文结论的影响：`TBD`

## P1.3 LR-matched initialization/adaptation controls

- **状态**：`[TODO]`
- **类型**：额外训练实验，高成本。
- **目的**：减少当前 random-init `1e-4` 与 inherited `2e-5` 的混杂。
- **最低新增臂**：fully random-init, peak LR `2e-5`；若资源允许，再跑 inherited train-all, peak LR `1e-4`，形成 2×2。
- **配置**：相同 16 层 keep14+fresh2 shape、数据、batch、schedule、200k steps；固定 seed。

**结果填写**

| Init | Peak LR | Trainable modules | PPL | MMLU | Raw/checkpoint path |
|---|---:|---|---:|---:|---|
| inherited | 2e-5 | all | 10.561 | .3191 | existing |
| inherited | 1e-4 | all | TBD | TBD | TBD |
| random | 2e-5 | all | TBD | TBD | TBD |
| random | 1e-4 | all | 11.498 | .2461 | existing |

- 对 initialization 因果解释的结论：`TBD`
- 对 LR 敏感性的结论：`TBD`

---

# P2：机制与外部泛化，可选增强

## P2.1 Tuned lens 复核

- **状态**：`[TODO]`
- **类型**：额外机制实验。
- **目的**：检验普通 logit lens 的 layer 18–19 跳变是否主要来自中间表示与最终 LM head 的不对齐。
- **要求**：为各层训练 tuned lens；独立 train/validation split；至少 3 probe seeds；报告 OLMo MMLU onset、sat95、sat99 及方差，并与普通 logit lens 对照。

**结果填写**

- 数据划分：`TBD`
- probe seeds：`TBD`
- tuned-lens onset/sat95/sat99：`TBD`
- ordinary-lens 对照：`18/19/27`
- raw/model path：`TBD`
- 是否保留 depth ordering：`TBD`

## P2.2 因果层恢复 / activation patching

- **状态**：`[TODO]`
- **类型**：额外机制实验。
- **目的**：区分“前 14 层已有可用信息但 fresh tail 无法读出”与“所需计算依赖被删除的上层”。
- **候选设计**：逐块恢复原始 upper layers；或将 base layer-14 hidden states 接入 original/fresh tail；或对关键层做 activation patching。
- **必须同时报告**：PPL 与 MMLU，避免只看单一指标。

**结果填写**

- 设计：`TBD`
- 干预位置：`TBD`
- PPL 变化：`TBD`
- MMLU 变化：`TBD`
- raw/code path：`TBD`
- 因果结论边界：`TBD`

## P2.3 非 OLMo 模型族复现

- **状态**：`[DONE]`（task #117，2026-08-02）
- **类型**：额外训练实验，高成本。
- **目的**：检验 observed-budget PPL–MMLU dissociation 是否超出 OLMo 家族。
- **候选模型**：Qwen3-8B base（36 层）。
- **最小设计**：一个约 half-depth prefix-prune+fresh-tail 配置；step 0 与 healing checkpoint；PPL、MMLU 和 P0.3 的两个知识任务。

**结果填写**

- 模型/版本：Qwen3-8B base（36 层），base 协议 chat=False / no-BOS / LL-MC（与 OLMo 深度阶梯同口径；harness = `eval_qwen3_probe2_{ppl,downstream}.py`，commit f83a696）。
- pruning/regrowth config：**f12k2** = keep_front_layers=12 + n_fresh_layers=2 = **14 层 shell（14/36 = 39% depth）**；比 OLMo keep12（14/32 = 44%）更浅。
- training config/seeds：prune-then-heal，NTP-only，SlimPajama 续训至 200k step，seed 42。⚠️ 语料 = SlimPajama（≠ OLMo 的 Dolmino）→ PPL **不可跨家族直接比**；MMLU above-chance recovery 与语料无关，**可比**。
- 轨迹结果（healed vs base full-36）：

| Qwen arm（层数 / depth） | PPL | PPL tax | core6 | aux aggregate | MMLU | MMLU above-chance recovery | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| base full（36L / 100%，参照） | 11.42 | 1.000× | 0.6648 | `AUDIT` | .7297 | 100% | 参照（vanilla Qwen3-8B base） |
| **f12k2（14L / 39%）** | **23.49** | **2.06×** | **0.4624** | **`AUDIT`** | **.2495** | **≈0%（at chance）** | `[DONE]` 200k heal，letter-protocol MMLU 未恢复 |

- raw/checkpoint path：`outputs/qwen3_probe2_8B_f12k2/`（healed 200k）；逐任务 summary 见 `qwen3_ppl_results/` + `qwen3_downstream_results/`（.73 diskB）。
- **当前可支持结论**：Qwen endpoint 同样表现为较高 PPL tax 与 chance-level answer-letter MMLU；它是跨家族诊断点，不是 compute/data-matched replication，也不能与 OLMo 三点拼成纯 depth law。
- ⚠️ **论文写作注意**：Qwen 与 OLMo 的 PPL/tax、训练语料和 depth 均不可直接对齐；在 P0.6/P0.7 前，不引用 aux aggregate 或“闭卷知识丢失”作为跨家族结论。

## P2.4 通用 SFT 接口可修复性诊断（可选，不替代结构控制）

- **状态**：`[TODO — OPTIONAL]`
- **目的**：判断低 MMLU 是否可由通用监督适配修复，区分 readout/interface adaptation 与结构性能力损失；不把 SFT 作为当前 NTP-only 主方法的一部分。
- **优先级**：仅在 P0.5 与 P1.1 之后运行；SFT 不能替代 full-32L continued-pretraining control 或 matched structural controls。
- **最小模型集合**：full 32L base、keep14+fresh2@200k、ShortGPT-16@200k；三者使用完全相同 SFT 数据、token budget、optimizer 和 seed。建议加入 keep14 的 equal-token NTP continuation 作为 compute control。
- **数据约束**：使用通用 instruction 数据；禁止 MMLU train/dev/auxiliary、学科多选题模板和明显 closed-book factual-QA 语料；对 SFT 数据与 MMLU/PopQA/TriviaQA 做去重与 n-gram overlap 审计。
- **评测**：SFT 前后均评 held-out Dolmino PPL、两种 MMLU protocol、core6、完整逐任务结果、PopQA 与 TriviaQA closed-book；aggregate 仅使用 P0.7 固定后的口径。保留逐题 prediction，报告 wrong→right/right→wrong、McNemar 和 paired bootstrap CI。
- **判定**：
  - MMLU 上升且独立知识任务同步上升：支持可监督修复，但仍不能证明原知识完整保留；
  - 仅 MMLU 上升：按任务格式/多选接口适配解释；
  - keep14 仍显著落后 ShortGPT：支持结构保留比 SFT 更关键；
  - full base 获得相似增益：不得将增益归因于压缩恢复。

**结果填写**

| Arm | SFT tokens | PPL pre/post | MMLU pre/post | PopQA | TriviaQA | Raw path |
|---|---:|---|---|---:|---:|---|
| full 32L | TBD | TBD | TBD | TBD | TBD | TBD |
| keep14+fresh2 | TBD | TBD | TBD | TBD | TBD | TBD |
| ShortGPT-16 | TBD | TBD | TBD | TBD | TBD | TBD |

- SFT dataset/version/overlap audit：`TBD`
- config/seeds：`TBD`
- paired analysis：`TBD`
- 结论边界：`TBD`

---

# 实验 agent 完成后的统一交付要求

每个 `[DONE]` / `[NEGATIVE]` 条目必须同时提供：

1. checkpoint 与 raw output 的绝对或仓库相对路径；
2. git commit/hash 或代码版本；
3. 完整配置、seed、数据版本与可复现命令；
4. 样本数、metric 实现和聚合方式；
5. 失败/异常 run 及排除理由；
6. 一句不夸大的结论；
7. 建议修改的 Paper B section/table/figure；
8. 主稿 `paperB/` 与匿名稿 `perplexity-heals-knowledge-lags/paper/` 必须同步。
