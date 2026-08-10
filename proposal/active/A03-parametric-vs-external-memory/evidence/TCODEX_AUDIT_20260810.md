# TCODEX 怀疑论审计：A01–A04 的结论、测试充分性与统计口径

**日期：2026-08-10**  
**审计范围：** `proposal/active/A01-*` 至 `A04-*`；重点为 A03。  
**操作纪律：** 全程只读。没有修改 repo、没有启动 GPU 作业、没有 kill/干预进程、没有 git commit。唯一写入为本报告及 `/tmp/tcodex_a03_audit/` 下的临时统计脚本/输出。

---

## 0. Executive summary

### 最重要的裁决

1. **A03 的 proposal-level thesis 目前没有被实证完成。** 原 proposal 要比较 6 个 arm（intact / pruned / +CPT / +raw-text RAG / +residual memory / +CPT+memory）；盘上只完成了 baseline/floor gate 和若干 **CPT schedule trajectory**。raw-text RAG、residual memory、joint 三类核心 arm 尚未实现/训练。因此“参数化知识应写回参数还是外部 memory”的主问题仍未被回答。
2. **A03 当前最应撤回的结论是：**“20k Dolmino CPT 恢复了参数化知识”“3 个 metric 一致排除了噪声”“step210k 是共享 trajectory feature”“low-LR/peak-LR 的终点符号翻转证明了 LR regime”。更严重的是，Arm4 首次 step220 checkpoint 被 watcher 截断后从 step215 重跑，而 trainer 没恢复 dataloader offset；因此 Arm4 最后 5k 重复了该 epoch 开头的数据，**Arm3 +0.48 vs Arm4 −0.93 不是干净的 matched-20k endpoint**。现有数据最多支持：在一个 OLMo-2-1B checkpoint和固定/重放的数据前缀上，QA snapshots 对 LR schedule 与数据暴露路径高度敏感。
3. **多重比较不是 A03 headline 的主要致命点，但确实淘汰若干边缘 SIG。** 当前 canonical trajectory JSON 实际有 **90 个成功检验**（Arm3 36 + Arm4 36 + Arm6 当前18），不是 108；若把计划中的完整 3×4×9 family 固定为 108，也可作更保守校正。按与项目相同的 5,000-bootstrap 两侧 p 值做 BH q=.05：
   - 消失：Arm3 step210 `nq_open.contains`；Arm4 step205 `nq_open.f1`；Arm4 step210 `popqa.em`；Arm6 step210 `nq_open.contains`。
   - **不消失：**用户点名的 Arm3 step210 `triviaqa.em +0.373 pp`，以及 Arm3 step220 TriviaQA `em/contains/f1` 三项。
   - Arm4 step210 `popqa.em +0.203 pp CI[+0.02,+0.38]` 是典型应撤回的边缘发现；BH-adjusted p 约 `0.083`（90-test family）或 `0.100`（108-test denominator）。
4. **“em/contains/f1 三项独立一致”是错误论证。** 同一 17,944 条预测上，Arm3 step220 的 item-level delta correlations 为：`corr(em,f1)=0.781`、`corr(contains,f1)=0.497`、`corr(em,contains)=0.380`；原始 metric correlations 更高（em-f1 约 `0.86`）。相关矩阵的有效维度约 **1.8–1.9 个独立 tests，不是 3 个**。三项联合为正提供一些额外信息，但不能当三次独立 replication，更不能补上 training-run replication。
5. **paired item bootstrap 完全没有覆盖 training-run variance。** Arm3/Arm4 各为1个run；Arm6虽在本地 `ce5c298` commit之后启动，但只读SSH实查 `.73` 的实际 trainer 仍是pre-fix代码 `DistributedSampler(ds, shuffle=True)`，三臂都使用默认 sampler seed 0、epoch 1、相同minibatch序列。日志loss相关性为 Arm3–Arm6 `0.99982`、Arm3–Arm4（至step215）`0.99187`。这不是三次独立训练。盘上没有同schedule的post-fix replicate，所以run variance的非零上下界无法识别。
6. **A02 作为“CoMem storage/quality thesis”已经实质失效。** 相对 raw-token RAG，h12 store 为 `8192 vs 4 bytes/token = 2048×`；自然任务 quality-win 不成立；原 proposal 的 write-repair arms 又未测试。可保留的只是极窄的 matched-pack prefill micro-optimization（当前 Qwen3-8B/H20/bf16/single-stream 下 read 约 `1.38×`，总 per-query 约 `1.03–1.37×`）。
7. **A04 尚未认证任何恢复。** 它仍是 protocol proposal。Pilot Zero 的 K1 前置条件未满足、统计单位实现不一致、plateau rule 量纲在粗 grid 上不成立，且 K2/run variance 完全未测。`K1 does not fire / survived cheapest gate` 应撤回，改为 `K1 INDETERMINATE`。
8. **A01 没有整体失效，但 `active_all_gates_passed` 过强。** 最稳结论是“能力解释前先过明确的 input-blind baseline；过 floor 是必要而非充分条件”。family-general sharp phase transition、五种同等 defensible null convention、跨四 construct `7–10×` 等应撤回或大幅收窄。

### 哪个 proposal 已实质失效？

**A02。** 更精确地说：A02 原本的 **CoMem storage/quality/write-repair paper thesis** 已实质失效；只剩一个不足以单独支撑原 proposal 的窄 systems observation。  
A03 的主 thesis 是**未完成、当前 CPT 子结论不成立**，但由于关键 memory/RAG arms 根本未测，我不把“整个研究问题”判为被反证；它是 **not established / needs redesign**。  
A01 仍有较窄方法学贡献；A04 仍是未验证设计，不是完成的 empirical proposal。

---

## 1. A03 重点审查

## 1.1 先区分三种完全不同的 claim

A03 文档把以下三件事混在一起：

1. **instrument detectability：** pruned+healed arm 是否高于 input-blind floor；
2. **absolute recovery：** pruned+healed 相对 intact 恢复了多少；
3. **incremental CPT effect：** step200k 后再训练 5k/10k/15k/20k 是否增加知识。

现有证据只较好支持第 1 点；第 2 点显示损失仍很大；第 3 点不稳定且没有 run replication。

### 原始绝对分数（直接读 summary JSON）

| arm | MMLU content_norm | PopQA EM | TriviaQA EM | NQ-open EM |
|---|---:|---:|---:|---:|
| intact 1B | 38.684% | 15.497% | 40.688% | 10.249% |
| keep7+fresh2 @200k | 32.438% | 3.939% | 9.585% | 2.853% |
| barely healed @500 | 26.321% | 0.000% | 0.022% | 0.166% |
| relevant input-blind null | 28.445% | 2.292% | 0.256% | 0.554% |

来源：

- `proposal/active/A03-parametric-vs-external-memory/evidence/olmo2_mmlu_content_results/*/summary.json`
- `proposal/active/A03-parametric-vs-external-memory/evidence/olmo2_closedbook_results/*/summary.json`
- NQ-open：只读 SSH 实读 `zwfy6:olmo2_closedbook_results/A03_1B_*_nq/summary.json`

**解释：**

- keep7@200k 确实高于 floor，这只证明该轴有可测信号。
- 但它距 intact 很远：PopQA 只保留约四分之一的 raw EM，TriviaQA/NQ-open 也约四分之一；不能把“above floor”写成“知识已恢复”。
- PopQA EM=3.94% 虽高于 floor 2.29%，但绝对动态范围很低；+0.5 pp 只是约 71 个 item。TriviaQA 的 9.59% 不是 floor-level，动态范围较好，但仍远低于 intact 40.69%。

### 关于 floor JSON 的一个公式/文字不一致

`GATE_FOURAXES_VERDICT.md` 写：

> Residual fraction = `(reported − null) / (1 − null)`

但 `a03_1b_floor_nulls_4axes.json` 中实际字段 `residual_fraction` 是：

> `(reported − null) / reported`

例如 pruned PopQA：`(0.03939−0.02292)/0.03939 = 41.8%`，而不是相对 available headroom 的约 1.69%。因此文档中“41.8%、97.3%、80.6%”等不能解释为“恢复了 intact/headroom 的比例”；它们只是**当前 reported score 中有多少高于 null**。这会显著美化低绝对分数，应修正口径。

## 1.2 多重比较审计

### 检验族究竟多大？

用户按计划矩阵计算 `3 arms × 4 steps × 9 = 108`。但当前 canonical JSON：

`proposal/active/A03-parametric-vs-external-memory/evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json`

实际只有：

- Arm3：4×9 = 36
- Arm4：4×9 = 36
- Arm6：当前 2×9 = 18；step215/220 为 0/8 shard error
- 合计 **90 个成功检验**

此外该 JSON **没有 MMLU cells**，虽然 STATUS/MD 还讨论 MMLU trajectory；因此项目的 machine-readable trajectory family 与 prose family 并不完全一致。

### 我采用的校正

从只读 SSH 的 per-example JSON 重建每个 delta vector，用与项目一致的 multinomial paired bootstrap（5,000，seed=42）计算两侧 p，再做 BH q=.05。结果：

| nominal SIG 但 BH 后消失 | delta | bootstrap p | BH adj（90） | BH adj（108 denominator） |
|---|---:|---:|---:|---:|
| Arm3 step210 NQ-open contains | −0.471 pp | .0364 | .091 | .109 |
| Arm4 step205 NQ-open F1 | −0.486 pp | .0276 | .073 | .088 |
| Arm4 step210 PopQA EM | +0.203 pp | .0324 | .083 | .100 |
| Arm6 step210 NQ-open contains | −0.499 pp | .0400 | .097 | .117 |

**仍保留：**

- Arm3 step210 TriviaQA EM `+0.373 pp`：bootstrap p floor `.0002`，BH adj 约 `.00069`（90）/ `.00083`（108）。所以它不是用户举例中会被 FDR 淘汰的 cell。
- Arm3 step220 TriviaQA：
  - EM `+0.479 pp`, BH adj约 `.00069/.00083`
  - contains `+0.474 pp`, BH adj约 `.00869/.01043`
  - F1 `+0.468 pp`, BH adj约 `.00069/.00083`
- Arm4 step220 的 EM/F1/contains 三项也都很强，BH 后保留。

**裁决：**“所有 SIG 都不可信”不正确；但凡 CI 下界只贴着 0 的边缘 cell，确有多个在合理 FDR 下消失。trajectory JSON 不应继续用裸 `SIG=CI excludes 0`。

### 为什么 5,000 bootstrap 不适合非常小的 p

项目把最小两侧 p 截在 `1/5000=.0002`；大量强 cell 都同为 `.0002`，无法排序。这对 BH 结论通常仍足够，但不能把 adjusted p 当高精度值。建议 binary EM/contains 同时报告 exact McNemar；连续 F1 用更多 resamples或 asymptotic paired SE。

## 1.3 “3 metric 一致，所以不是噪声”是否成立？

### 它们不是三个独立实验

Arm3 step220 TriviaQA 的 item-level delta correlation：

| | EM | contains | F1 |
|---|---:|---:|---:|
| EM | 1 | 0.380 | **0.781** |
| contains | 0.380 | 1 | 0.497 |
| F1 | **0.781** | 0.497 | 1 |

原始 metric correctness 在 baseline/arm 内的 correlations 约：EM–F1 `0.86`、contains–F1 `0.69–0.70`、EM–contains `0.60`。相关矩阵的 effective number of tests 约 **1.8–1.9**。

Arm3 step220 的翻转计数进一步说明它们共享同一预测变化：

- EM：233 个 wrong→right，147 个 right→wrong，净 +86；
- contains：466 个 gain，381 个 loss，净 +85；
- F1 与 EM delta correlation 0.781。

### 它们仍有多少增量信息？

有，但有限：contains 不完全由 EM 决定；三项一起为正比只有 EM 正更让人放心。保持 item correlation 的 sign-flip/randomization global test 给出很强的联合 item-level signal。但该结果只能说：

> 对这一个训练 realization、这一个固定 eval set，TriviaQA predictions 在三个相关 scoring functions 上共同改善。

它不能说：

> 三个独立 replication 排除了 chance；或排除了训练随机性。

更致命的是 Arm4 step220 出现：EM `−0.931`、F1 `−0.824`，但 contains `+0.881`。同一模型更新同时被一个度量判坏、另一个判好，说明“三 metric coherence”不是稳定属性；metric choice 本身就是 estimand。

## 1.4 training-run variance：目前能说什么、不能说什么

### 已确认的随机性结构

`git show ce5c298` 证明旧 trainer 的 `DistributedSampler` 未传 `seed=`：global seed 不会影响 shuffle。Arm3 与 Arm4 分别在该修复前启动；Arm6 虽在该 commit 后启动，但只读 SSH 实查 `.73` 的实际 checkout（HEAD `2d98c5a`）仍为：

```python
sampler = DistributedSampler(ds, shuffle=True)
```

所以三者实际都走 pre-fix 默认 `sampler.seed=0`。三者：

- 同一个 step200000 checkpoint；
- seed=42；
- dropout=0；
- 同一个 `data/dolmino_now15b.npy`；
- 都恢复 `epoch=1`，然后 `sampler.set_epoch(1)`；
- 都重新从该 epoch 的 dataloader **开头**建立 iterator。

日志来源（只读 SSH）：

- `zwfy6:logs/a03_arm3_cpt20k.log`
- `zwfy6:logs/a03_arm4_peaklr20k.log`
- `zwfy6:logs/a03_arm4_peaklr20k_redo220k.log`
- `zwfy6:logs/a03_arm6_lowerband20k.log`

因此三臂在共同覆盖的区间使用**相同的数据顺序**；差别主要是 LR shock/schedule。训练日志也提供独立印证：Arm3–Arm6 loss 序列相关 `0.99982`，Arm3–Arm4 至 step215 相关 `0.99187`。这恰恰说明 step205/210 的跨臂同向变化不能当独立 replication。

### 新发现：resume 没有恢复 dataloader offset

checkpoint 只保存 `epoch`，没有保存 epoch 内 batch offset。原 16-GPU run 每 epoch 约 121,028 optimizer steps；step200k 在 epoch 1 内已经走到约 optimizer offset 78,972。resume 后却从 epoch1 permutation 的第一批重新开始。

我按 `DistributedSampler(seed=0, epoch=1)` 重建 rank0：

- 应继续的第一 batch 与实际 resume 第一 batch 完全不同；
- continuation 的 20k-window与“本应继续”的 20k-window在该无放回 permutation 上 **0 overlap**。

所以 Arm3/4/6 不是“继续读取接下来的 Dolmino 数据”，而是**重放 epoch1 的早段 corpus**。这对 A03 非常重要：任何所谓5k/10k/15k/20k dose-response同时也是对某一固定 corpus slice 的 exposure response；对新事实/旧事实的覆盖可能随slice而变。文档没有讨论这一confound。

### Arm4 step220 还有第二层、更严重的 data-exposure confound

Arm4 首次 step220 checkpoint 被 watcher 在写盘中截断。随后项目从 `step215000.pt` 重跑最后5k。由于 resume 再次只恢复 `epoch=1` 而不恢复 epoch内offset，redo的 step215→220 又从 epoch1开头取数。因此当前有效 Arm4 step220 的训练路径是：

> 原共同数据前缀的前15k steps + **最前5k数据再重复一次**

而不是原计划的连续20k数据。原Arm4与redo在相同global steps 215020–220000的loss相关性仅 `−0.0667`，直接证明最后5k数据路径已改变。故当前 Arm3 step220 vs Arm4 step220 的 `+0.48/−0.93` 对比不仅缺少run seeds，连20k data exposure也不匹配；“只因LR不同而sign-flip”应暂停。

### 是否能从现有数据给 run variance 上/下界？

**不能给非平凡的统计界。**

- 下界只能是 0；上界在 accuracy 上可到很大，没有 replicate 无法识别。
- checkpoint-to-checkpoint 波动不是 run variance，因为模型权重、累计数据和 LR 都在变。
- Arm3 vs Arm4 差异不是 seed variance，因为 schedule 不同；而当前 Arm4 final 还多了 redo data-path confound。
- 旧所谓 seed pair 只是 init variance，且与 A03 schedule/topology不匹配。

可以给的只是**决策阈值/敏感性条件**：

- Arm3 final +0.479 pp 若要在单 run 95%意义下超过 run noise，run-level SE 必须远低于约 `0.24 pp`；S=3 时还需 sd_run足够小。
- Arm3 vs Arm4 final EM direct paired-item gap为 `1.410 pp CI[1.137,1.683]`，说明这两个**已产生的 checkpoints**在固定items上差很大；它不是干净的schedule causal effect，因为 Arm4最后5k数据不同。盘上也没有数据判断真实run SD是0.1还是1pp。

因此 STATUS 中“gap 1.4pp 大于 item CI，所以 unlikely seed noise”没有统计依据：item CI 与 run variance是不同层次，不能互相替代。

## 1.5 benchmark 与 metric 是否适合“参数化知识恢复”

### (a) PopQA 动态范围很差；TriviaQA较好；NQ-open对小效应没 power

- PopQA keep7 EM=3.94%，floor=2.29%；它可以做“是否脱离 floor”的 kill gate，但不适合作为 ±0.2–0.5pp recipe effect 的主轴。
- TriviaQA keep7 EM=9.59%，intact=40.69%，floor=0.26%，是三者中最有动态范围的 axis。
- NQ-open n=3610，trajectory item-only MDE（双侧 α=.05、80% power，按观测 discordance/CI）大约：
  - EM `~0.40 pp`
  - contains `~0.69 pp`
  - F1 `~0.49 pp`
  在 multiplicity 和 run variance加入后更差。它对项目关注的 ~0.5pp效应只是边界 power，对 contains基本无力；作为“第四独立轴”会制造很多 TIE，不能解释为无恢复。

### (b) contains 不适合作为与 EM 同等的知识恢复主度量

scorer 定义是：normalized gold alias 是否为 normalized prediction 的任意 substring。它奖励 verbose answer、列表、重复和猜多个候选。原始预测长度：

- baseline TriviaQA mean 72.38 chars
- Arm3 step220 70.30 chars
- Arm4 step220 78.64 chars

Arm4 step220 的反向结果可以由 per-item flips直接看到：

- EM：212 gain、379 loss，净 −167；
- contains：676 gain、518 loss，净 +158。

包含命中中存在大量长、重复、多实体输出，例如“Canterbury Cathedral, York Minster, …”或反复陈述答案；这类输出可 contains=1、EM=0、F1很低。因而 Arm4 的结果不是“两种都 valid 但发现复杂”，而是：**contains 测的是宽松 answer-string coverage/verbosity，不是精确 closed-book recall。**

最低限度应将 EM设 primary，token F1 secondary，contains只作 diagnostic，并使用 per-arm、per-checkpoint length-matched null；trajectory 当前只比较 raw contains delta，没有重新校准预测长度变化。

### (c) train/test contamination 与 answer-string artifact

盘上没有 Dolmino 文档 ID/原始文本到 PopQA/TriviaQA/NQ-open 的 overlap audit。不能判断 benchmark question、answer alias或相关网页文本是否出现在 CPT corpus。尤其 TriviaQA 常见事实与 Dolmino web mixture overlap 的先验风险不低。

这会导致：

- CPT 可能只是重复看到了 benchmark answer/string，而不是一般参数化知识恢复；
- contains 比 EM 更容易受 answer-string exposure影响；
- Arm3 的 TriviaQA-only gain也可能反映 corpus composition，而非 benchmark-general knowledge。

现有盘上数据**无法判断污染程度**，必须列为 unknown，不能猜“无污染”。

### (d) MMLU-content 不是纯“参数化事实知识”轴，四个 axes 也不独立

MMLU full aggregate 混合事实知识、概念理解、数学/逻辑和题型推理；A03 没有预注册 knowledge-heavy subject subset，也没有 subject-level trajectory analysis。把 full-MMLU content_norm 的不动直接解释为“参数化知识不动”存在 construct overreach。

同时 PopQA、TriviaQA、NQ-open 都是开放式、entity-heavy closed-book QA，可能共享实体、网页来源、alias/string artifacts和训练污染。它们是三个 benchmark，不等于三个独立 causal axes。最低补救是：

- MMLU 报 57-subject hierarchical/macro结果，并在看trajectory前冻结 factual subset；
- 三个QA benchmark按一个“closed-book entity QA family”处理，报告benchmark-level heterogeneity，而不是用“4 axes中3个同意”作独立复制计数；
- 增加一个不依赖短答案alias matching的生成式事实评测或人工抽样验证。

## 1.6 “step210000 瞬变”解释是否可信？

### 观察本身是真的，但解释过度

TriviaQA EM trajectory：

- Arm3：`−0.485 → +0.373 → −0.017 → +0.479`
- Arm4：`−1.404 → +1.259 → −0.117 → −0.931`
- Arm6 当前：`−0.730 → +0.775 → ? → ?`

在 step205 与 step210，effect magnitude 还随 LR 大体排序（Arm3 low、Arm6 mid、Arm4 high），这提示 shared data-order × LR interaction，不像独立 eval noise。

但项目说“这是 trajectory 的共享 non-monotone feature”还不够具体，更不能推出它是模型的一般机制。最强替代解释是：

1. **共享固定 data order/corpus slice。** 三臂从同一 epoch 开头读；step205/210对应完全相同的数据边界。某一5k block可能富含/缺少TriviaQA-like facts。Arm3–Arm6 loss相关0.99982，说明共享输入轨迹不是猜测。
2. **resume offset bug。** 它们重放 epoch1早段，而非继续 step200k后的数据；所谓 dose time 与特定重放 slice绑定。
3. **LR × shared slice interaction。** step205早期 damage、step210正跳的幅度随 LR增大，可能是对同一 corpus block的适应/遗忘，不是独立“trajectory phase”。
4. **Arm4 LR shock / schedule confound。** 日志显示102/102 optimizer states成功恢复，所以它不是“Adam warm-restart”或丢失moments；但它把已适应min-LR的moments突然置于约10×更高LR，并加入500-step rising warmup。这测的是完整LR shock+schedule，不是只改一个恒定LR标量的干净control。
5. **checkpoint/eval harness effect较不可能，但checkpoint生产链确有实质问题。** eval是greedy、固定items，batch/order不应造成随机波动；然而watcher race不仅产生过corrupt checkpoint，redo又因loader-offset缺陷改变了最后5k数据。final文件可load不等于它仍是原设计的matched endpoint。

### 20k window是否支持 dose-response？

**不支持稳定 dose-response。** 四个 checkpoint不是独立 doses；它们是同一训练路径上的相关 snapshots。没有重复路径、没有 randomized corpus order、没有 monotonic model，也没有 pre-registered shape。Arm4 endpoint在 EM/F1负、contains正，Arm3 endpoint小正，说明“更多 steps→更多 knowledge”不成立。

可保留的最窄表述：

> 在固定且重放的数据前缀下，closed-book QA metrics随5k checkpoint高度非单调；至step215前，不同LR schedule产生不同trajectory。当前Arm4 step220因redo data-path不匹配，不能作为干净终点比较。

应撤回：

> step220是“useful CPT budget被命中”、step210是data-order-independent共享trajectory feature、低LR恢复而peakLR伤害、Arm3/Arm4干净matched-20k sign flip，或任何单调dose-response。

## 1.7 A03 的逐 claim 裁决

### (a) 站得住

- keep7@200k 在 MMLU-content、PopQA EM、TriviaQA EM、NQ-open EM 上高于各自的 input-blind floor；barely-healed control可落到/低于 floor。**只说明 instrument有动态范围。**
- MMLU-letter在该 1B damaged arm上低于 always-D，不适合作主轴。
- Arm3与当前Arm4 checkpoint在TriviaQA EM/F1上方向不同；direct item-level gap很强。但Arm4 final最后5k重复了错误数据前缀，因此这只是checkpoint差异，不是干净的recipe sensitivity endpoint。
- NQ-open对小 trajectory effects power不足，应降级为 descriptive。

### (b) 需要收窄

- “四个 floor-certified axes”→“四个 axes能区分 intact/healed/barely-healed的大差异”；不能等同于能够检测 0.5pp CPT effect。
- “20k CPT moves TriviaQA by 5% relative”→“这一固定 run 的 TriviaQA EM从9.585%到10.065%，绝对 +0.479pp”；不得推广到配置期望。
- “LR schedule sensitivity”→至多写成“至step215前，同一数据前缀上的两个schedule轨迹不同”；final endpoint被redo data confound污染。
- “contains supports knowledge gain”→只作 verbosity-sensitive diagnostic。

### (c) 应撤回

- “3-metric coherence rules out noise”作为 training-effect结论。
- “Arm3 step220是一项 parametric knowledge recovery discovery”。
- “Arm4证明 peak-LR actively harms而 low-LR recovers”，以及“Arm3/Arm4 matched-20k sign flip”。
- “step210 positive是 shared trajectory feature”作为机制解释。
- “CPT trajectory claim survives/low-LR regime is publishable”在没有同-schedule seeds之前。
- 任何对原 6-arm parametric-vs-external-memory thesis的结论；三类 external-memory arms未测。

## 1.8 A03 的 md / JSON / implementation 不一致

1. `GATE_FOURAXES_VERDICT.md` 把 residual fraction 定义为 `(reported-null)/(1-null)`，JSON 实际计算 `(reported-null)/reported`。
2. `STATUS.json:second_axis_nq_open_gate.analyzer_TODO` 和 `next_gate` 仍要求扩展 NQ analyzer，但 `GATE_FOURAXES_VERDICT.md` 已称该 TODO closed，`a03_1b_floor_nulls_4axes.json` 也已生成。
3. `arm3_arm4_arm6_cpt_trajectory_paired_full.json` 的 machine-readable trajectory 没有 MMLU cells，prose/STATUS 却把 MMLU称为同一“four-axis trajectory”的一部分。
4. `ARM3_CPT_TRAJECTORY_INTERIM_VERDICT.md` 称 em/contains/f1 为“three independent metrics”；per-item correlation直接否定“independent”。
5. `ARM4_PEAKLR_VERDICT.md` 将step220当matched 20k endpoint，但实际redo从step215无offset resume，最后5k data exposure不匹配；JSON只能证明checkpoint分数，不能证明原设计对比。
6. 历史 `GATE_NQOPEN_VERDICT.md` 使用0.0053，canonical analyzer JSON为0.005540；后续文档已承认修正，引用必须以JSON为准。

---

## 2. 逐 proposal 裁决

## 2.1 A01 — null-calibration-methodology

### 站得住

1. **先报告明确的 input-blind baseline，而不是 generic chance。** MMLU always-D=`0.26891`而非0.25；BoolQ always-B=`0.6217`而非0.5。floor failure可以取消一个readout做能力解释的资格。
2. **原 ranking-flip headline应撤回。** `null_calibration_obs4_nperm2000.json`显示两个BH后仍显著的interface ranking flips，但涉及的三个arm letter均未过floor；原 claim不是“两种有效测量给出相反排序”。
3. **exact bf16 ties不是 family-general必要/充分机制。** keep8 ties从30.64%到0、18.03% argmax改变，accuracy不改善；intact Llama-2有15.79% ties却明显过floor。
4. **六个极端 non-OLMo front-truncation arm在固定MMLU protocol下均低于global always-D；**该结果很强，不靠边缘p值。

### 需要收窄

- “general structural damage response”→只限极端front-8/front-12无heal截断，加一个不同regime的OLMo healed keep8；不是所有结构损伤。
- “content swap救不了”→对六个pure-truncation non-OLMo成立；OLMo healed keep8的content_norm=0.3423，明显高于split null 0.2845，因此不能generalize到所有damaged/healed arms。
- “construct-appropriate null”还需说明允许null使用哪些metadata。当前global always-D为0.26891；从per-item按subject选择best constant可得约0.30943。它不会挽救damaged arms，但说明global constant不是唯一合理的input-blind baseline。
- healthy strong models letter高13–23pp是描述性事实，不证明letter更valid，也不存在已验证的0.60 competence threshold。
- gate2只是OLMo BoolQ/OpenBookQA case study，且OBQA n=500。

### 应撤回

- “letter is a family-general step function / sharp phase transition”。Qwen/Llama3/OLMo在盘上确有陡jump，但Llama2明确多次升降；没有change-point inference、answer-order replication或subject replication。
- “damage generally turns letter into a constant predictor”。若modal share只有45–47%，低于constant floor不等于模型本身逐item恒定。
- “五种同等defensible tie conventions使5/6 verdict翻转”。`credit`是oracle upper bound，`wrong`是pessimistic bound；真实可执行的split/first/last三者仅差约0.34pp且6/6均above。
- 跨C1–C4 residual fractions“真实跨约7–10×”。accuracy、EM、CKA、depth比例没有共同estimand；该span只能作所列aggregation的描述性敏感性。
- `active_all_gates_passed`若被理解为一般定律已证成，应撤回。

### 关键 md/JSON 不一致

- `GATE1_DEPTHCURVE_VERDICT.md`称Llama2 content strictly monotone，但summary在k8→10→12小幅下降。
- `7B_base_dtype_summary.json:letter_acc_diff_boot_p=1.042`，是非法p值；应使用CI/McNemar。
- `GATE3_VERDICT.md`“removing ties changes nothing”过宽：keep14和ShortGPT accuracy有显著dtype change，只是floor verdict未变。
- gate2没有proposal-local canonical JSON，主要证据仍在`status/scout_21/lane2_a01_gate2.md`。

读过路径包括：`A01/STATUS.json`、`PROPOSAL.md`、全部`*VERDICT*.md`、`evidence/a01_gate1_third_family.json`、`gate3_content_null_conventions.json`、六个dtype summary、`gate4_c4_prereg.json`、`null_calibration_{p1,obs4}_nperm2000.json`及相关summary/per-item。

## 2.2 A02 — comem-write-read-repair

### 站得住

- 相对raw-token text-RAG，h12 store为`8192/4=2048×` bytes/token；storage-win叙事失败。相对full-depth KV它仍约小18×，比较对象必须写清。
- 固定top-12、read_len=6177、Qwen3-8B/H20/bf16/single-stream microbenchmark中，跳过bottom-12 replay使read约`1.38×`快。
- phase1中相对pack-all C1的巨大速度优势主要来自top-k retrieval，不是CoMem representation。
- BABILong qa1/qa2的巨大quality loss混入重大retrieval/pack narrowing effect；RULER若pack固定，部分cells仍有约10–12pp deployed-read tax。
- 无Read-LoRA的j12 interface在所测RULER subset几乎不可用；结论限当前interface/task/model。

### 需要收窄

- “read-compute survives”只能限当前single-stream resident-store microbenchmark；G=128时总优势约3%，没有continuous batching/concurrency/NVMe/network。
- `N*=8–226`是component-median point estimate，无CI；process-level上界可接近~297。
- “retrieval dominates 54.9–78.6%”是代数point share；只有部分cell能确认retrieval effect严格大于read effect。
- 1M selection latency来自random-token padding corpus，且timed selector pack与actual read pack在128k/1M不完全一致，不能当真实end-to-end retrieval workload。

### 应撤回

- CoMem在natural/semantic QA上不劣或更好；LongBench、LoCoMo canonical F1、open-weight judge均是tie，且绝对分数低。
- GPT-4o LoCoMo `+3.22pp`；instrument替换不合协议。
- LoCoMo F1 `+0.68pp significant`；重实现scorer有bug，canonical是`+0.20pp CI[-0.36,+0.75]`。
- pooled BABILong `−17.89pp`及pooled LongEval `+2pp`作为benchmark headline；它们混合相反方向cells。
- “write repair已被证伪”；原proposal的overlap/Write-LoRA/Write+Read arms没有在自然任务中完成。
- `a02_storage_readcompute_verdict.json` top-level `SURVIVES`作为正确gate判决；其代码没有执行自身rule里的bounded storage premium条件，与STATUS“storage DEAD”冲突。

### 实质状态

**A02原proposal已实质失效。** 剩余read-prefill micro-optimization值得作为附属systems observation，而不是当前CoMem storage/quality/write-repair thesis。

读过路径：`A02/PROPOSAL.md`、`STATUS.json`、三个verdict、所有proposal-local evidence JSON（phase1、storage、depth-vs-retrieval及18个serve records）。

## 2.3 A03 — parametric-vs-external-memory

见第1节。最简裁决：

- **站得住：**instrument floor gate；固定checkpoint之间确有分数差异；NQ-open低power。
- **需收窄：**所有CPT effect只写成单run descriptive difference。
- **应撤回：**parametric knowledge recovery headline、3-metric independence、Arm3/Arm4 matched-20k endpoint、LR regime、shared step210 mechanism、任何对6-arm主问题的答案。

## 2.4 A04 — recovery-certification

### 站得住

- 历史keep7@200k远未接近intact；三个decision axes的arm−intact gap分别约`−31.10/−11.56/−6.25pp`，远超10%-residual margins。
- ratio rule `rho=.85`在已分析8 cells全部拒绝，与NI一致；因此“两种incumbent rules都过早接受”已被自身数据否定。
- NQ-open在当前n/margin下不适合认证；credit convention下MMLU margin为负，退休该cell合理。
- repo没有真实run-to-run variance，K2未测。

### 需要收窄

- “PPL-capability disagreement”只能说：一个只看局部斜率、T=2% heuristic在一个历史arm上可能过早停；PPL本身仍比intact高44.8%。
- “K3 does not fire”只能指intact dynamic-range branch；planned j12 damaged arms不存在。
- `S=3 minimum`只是若sd_run足够小的条件，不是完成的power design。
- “certification”只能称通过一个人为设定的90%-of-residual retention criterion；10% margin缺少external/operational justification。
- 当前是one-sided non-inferiority，不是双侧TOST/equivalence。

### 应撤回

- `K1 does not fire / A04 survived cheapest gate`。K1要求至少24个可比较arm×checkpoint cells；Pilot Zero只有1个arm×checkpoint拥有plateau accept和capability，K1应为`INDETERMINATE`。
- “required disagreement exists”作为confirmatory finding；粗47–53k grid无法检验“2% per 5k”rule。
- `Pilot One=A1 S=3 5k即可回答K2`；K2定义在apex且pooled across arms，一个arm/5k不能外推。
- machine-readable `CERTIFIABLE`命名；当前最多是`MARGIN_ADMISSIBLE`。

### 设计实现问题

- K1 prose以arm×checkpoint为cell，JSON却把三个axis当3个disagreement cells，直接影响`≤1`条件。
- scaled plateau reading接受100k/147k/200k，但只有200k有capability comparison，未检验“每个接受点”。
- item bootstrap与run-level t bound没有组成一个明确的两层estimand。
- planned 96 NI cells没有实际NI p-values/BH执行；BH-FDR也不足以控制“首次通过checkpoint”的false certification。
- `STATUS.json`仍有awaiting Pilot Zero等过时字段；old JSON仍对guard已退休的credit cell运行NI。

**裁决：A04可保留为待验证protocol proposal，但当前没有任何恢复被认证。**

读过路径：`A04/STATUS.json`、`PROPOSAL.md`、`A04_GATE_DESIGN.md`、`A04_MARGIN_GUARD_PREREG.md`、`PILOT_ZERO_VERDICT.md`、`SEED_SEMANTICS_DEFECT.md`及全部evidence/code。

---

## 3. 测试不充分 / 测试集不合理：问题 → 错误结论 → 最小补救

| # | 问题 | 会导致的错误结论 | 最小具体补救 |
|---|---|---|---|
| 1 | **A03每个schedule只有1个training run**；item bootstrap不含run variance | 把+0.48pp或+1.4pp arm差异当配置期望 | 对**同一keep7@200k起点、同一LR schedule**做post-fix seeds；Arm3和Arm4各至少S=3，最好S=5，保存205/210/215/220k；对TriviaQA n=17,944、PopQA n=14,267、MMLU n=14,042做run×item hierarchical bootstrap |
| 2 | A03 resume只恢复epoch，不恢复epoch内offset，重放epoch1早段；Arm4 redo又重复前5k | 把特定corpus slice exposure解释为CPT dose/knowledge recovery，并把不匹配endpoint归因给LR | 修复sampler offset或使用显式precomputed row schedule；从step200k完整复跑Arm3/Arm4，禁止从中间checkpoint用无offset resume拼接final；另设intentional-replay control，各S≥3 |
| 3 | Arm3/4/6共享固定data boundaries；step210同步正跳 | 把data-slice×LR interaction称为共享trajectory机制 | 固定LR、只改变data-order seed（至少3）；另固定data order、改变LR。若step210只随slice复现，则不是一般trajectory feature |
| 4 | A03 trajectory裸看90/计划108 tests，无FDR | 边缘SIG被误报为发现 | 预先声明family；对全部arm×step×task×metric做BH/Holm。立即撤销Arm4 step210 PopQA EM等4个BH失败cells |
| 5 | EM/contains/F1高度相关 | “3项独立一致”夸大证据 | TriviaQA EM设唯一primary；F1 gatekept secondary；contains diagnostic。联合报告delta covariance或multivariate permutation，禁止称3 independent metrics |
| 6 | contains奖励verbosity/multi-answer string | EM下降但contains上升仍被写成知识gain | 每checkpoint计算prediction length、length-matched contains null；人工审计至少200个EM/contains discordant items；primary不使用contains |
| 7 | PopQA keep7 EM只有3.94%、floor 2.29% | 将几十个hit变化解释成稳定参数知识恢复 | PopQA只作floor/detectability secondary；若要检测0.5pp，扩大到独立长尾QA集合并预设aggregate，或用更高base-score task |
| 8 | NQ-open n=3610，contains/F1 MDE约0.69/0.49pp（item-only） | TIE被解释为无效应/独立轴支持null | 将NQ-open降为descriptive；若要检验0.3pp，需约扩大到≥1万级或合并预注册的独立NQ-like set；仍需run seeds |
| 9 | 无Dolmino↔QA overlap/decontamination audit | corpus重复benchmark答案被称为一般知识恢复 | 对20k continuation实际row IDs反tokenize/检索，做question/answer n-gram、BM25和semantic overlap；报告clean subset。至少对TriviaQA全部17,944 item给contamination flag并重算EM |
| 10 | A03原6-arm中RAG/memory/joint未测 | 用CPT子实验回答parametric-vs-external-memory主问题 | 先完成matched evidence的4个最关键arm：keep7 baseline、+CPT、+raw-text RAG、+residual memory；相同QA items、相同evidence、成本同时报告。未完成前不得proposal-level verdict |
| 11 | A01只用canonical answer order/global constant | label-position artifact被称为structural failure | 对Qwen k24/k25、Llama3 k17/k18、OLMo k18/k19及6个k8/k12 arms跑8–24个预固定answer permutations；报permutation-averaged accuracy、subject-conditioned floor、subject-macro结果 |
| 12 | A01 `p>=.05`被写成“at floor/constant predictor” | absence of evidence被当equivalence | 预设δ（如1pp）；用NI/equivalence三区间：valid / inconclusive / floor-equivalent；不要二元化裸p值 |
| 13 | A01 change-point事后从dense sweep读出 | 最大相邻jump被叫phase transition | 对每family候选k−1/k/k+1独立复评；answer permutations；bootstrap change-point location；57 subjects分层验证 |
| 14 | A01 credit/wrong不是可执行null policies | oracle bound制造“5/6 verdict flip” | primary仅split/first/last；credit/wrong改名oracle upper/pessimistic lower bounds，不参与validity verdict |
| 15 | A02核心write repair arms未测 | read-only C2失败被泛化为write repair失败 | 至少比较j12 Read、+overlap-w32、+Write-LoRA、+Write+Read；BABILong/RULER各n≥500/cell，固定selector/top-k/sample IDs；再加一个full natural QA family |
| 16 | A02 cost只有一个task/example、single-stream resident store | microbenchmark被写成high-reuse serving win | 至少100真实queries×3 corpora，continuous batching/concurrency 1/8/32，GPU/CPU/NVMe tiers；每query直接测end-to-end而非component medians |
| 17 | A02旧RULER/BABILong n=100，项目policy要求n=500 | 2–5pp差异power不足、task-class claim不稳 | 重跑deployed read关键10 cells到n=500；BH family预注册；qa5/VT的小差异不得用旧n=100定论 |
| 18 | A04 K1 cell单位和plateau grid错误 | 缺数据被判“gate survived” | cell固定为24个arm×checkpoint；精确每5k PPL；至少连续两个interval满足plateau；每个accepted checkpoint都做capability eval |
| 19 | A04没有两层不确定性/真实variance | item CI很窄导致虚假certification | planned primary arm j12至少S=5直跑20k，保留2.5k checkpoints；run-level或hierarchical CI；margin和sequential FWER预注册 |
| 20 | A04 10% margin无外部意义 | 数学通过被叫“恢复认证” | 先用deployment tolerance、expert elicitation或downstream utility定义margin；否则命名为“90%-residual retention test”，不叫恢复等价 |

---

## 4. 如果只能再做一个实验

### 选择：A03 同 schedule 的真正多-seed replication，而不是再加第四个LR

**目标：避免发表“20k CPT恢复了参数化知识 / low-LR有效、peak-LR伤害”这一最可能错误的结论。**

### 具体设计

- 起点：同一个 `A03_1B_keep7_step200k` checkpoint。
- 两个schedule：
  1. Arm3 cosine-tail：`warmup=150,max_steps=300000`；
  2. Arm4 peak-LR：`warmup=200500,max_steps=240000`。
- **每schedule S=5个post-`ce5c298` seeds**（至少S=3；S=5更可靠），seed同时进入sampler。
- 修复resume：保存并恢复epoch内sample offset；若无法修，预生成并冻结每seed的row-index schedule，保证“continuation”语义明确。任何corrupt final必须从step200k完整重跑，不能从step215用错误offset补5k。
- 每run保存205/210/215/220k。
- Primary endpoint：TriviaQA EM @220k，n=17,944。
- Secondary gatekept：TriviaQA F1、PopQA EM、MMLU-content；NQ-open/contains descriptive。
- 统计：每seed先算arm−baseline paired item delta；最终以run为replication unit比较schedule means，另做run×item hierarchical bootstrap。只检验预注册的：
  - Arm3 final mean >0；
  - Arm3−Arm4 final mean >0；
  - step210 transient是否跨seeds复现。
- 同时记录每run实际row IDs，并做TriviaQA contamination/overlap flag的clean-subset分析。

### 为什么这是最高价值

- 若Arm3 mean不稳定或CI跨0，当前+0.48 headline直接死亡。
- 若Arm3稳定正、Arm4稳定负，才有资格讨论schedule interaction。
- 若两者run variance大于~0.5pp，A03所有现有小效应均应降为noise-level。
- 它同时检验step210是否data-order artifact；再跑一个单seed中间LR不能解决这个根本问题。

---

## 5. 无法从盘上数据判断的事项

1. A03真实run-to-run variance、其随schedule/checkpoint的变化。
2. 若在修复loader offset并完整重跑后，Arm3/Arm4终点符号差异有多少来自LR、有多少来自训练随机性；当前final还混有已知data-path差异。
3. Dolmino与PopQA/TriviaQA/NQ-open的训练污染/answer exposure程度；没有corpus-text overlap artifact。
4. Arm3的TriviaQA gain是否在不同pretraining seed、不同base checkpoint、不同model family复现。
5. step210同步跳变是否由某个具体5k corpus block、optimizer dynamics、或其他训练状态造成。
6. A03 external-memory/RAG/joint arms的质量、成本和Pareto关系；它们尚未完成。
7. A01陡峭depth jump在answer-order permutations、subject-macro及另一MC benchmark上是否复现。
8. A02 microbenchmark在continuous batching、并发、NVMe/networked store和真实1M文本上的方向/幅度。
9. A02 write-repair interventions是否有效；未测试不能判死。
10. A04 j12 recovery level、20k/80k是否足够、任一planned arm的run variance、以及10% margin是否有实际部署意义。
11. NQ-open/PopQA item-level独立性在entity/topic clustering后CI会扩大多少；当前bootstrap以item为i.i.d.。
12. 当前正在运行的Arm6 step215/220最终结果；本审计冻结在canonical JSON仍为0/8 shard error的时间点，且严格没有触碰live job。

---

## 6. 主要证据路径

### A03

- `proposal/active/A03-parametric-vs-external-memory/STATUS.json`
- `PROPOSAL.md`
- `ARM3_CPT_TRAJECTORY_INTERIM_VERDICT.md`
- `ARM4_PEAKLR_VERDICT.md`
- `ARM4_DESIGN.md`
- `ARM6_LOWERBAND_INTERIM.md`
- `GATE_FOURAXES_VERDICT.md`
- `GATE_NQOPEN_VERDICT.md`
- `evidence/a03_1b_floor_nulls.json`
- `evidence/a03_1b_floor_nulls_4axes.json`
- `evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json`
- `evidence/olmo2_{closedbook,mmlu_content}_results/*/summary.json`
- `code/recompute_cpt_trajectory_paired.py`
- `code/analyze_1b_knowledge_floor.py`
- `scripts/eval_olmo2_closedbook_qa.py`
- `scripts/train_olmo2_arch_probe2.py`
- `scripts/_run_a03_arm{3_cpt,4_peaklr,6_lowerband}.sh`
- `git show ce5c298`
- 只读SSH：`zwfy6:olmo2_closedbook_results/A03_1B_*/{summary.json,per_example_*.jsonl}`、`olmo2_mmlu_content_results/A03_1B_*/summary.json`、`logs/a03_arm*.log`

### A01

- `proposal/active/A01-null-calibration-methodology/STATUS.json`
- `PROPOSAL.md`、全部`*VERDICT*.md`
- `evidence/a01_gate1_third_family.json`
- `evidence/gate3_content_null_conventions.json`
- `evidence/gate3_dtype_runs/*.json`
- `evidence/gate4_c4_prereg.json`
- `evidence/null_calibration_{p1,obs4}_nperm2000.json`
- 相关`olmo2_mmlu_content_results/*/summary.json`与per-item；只读SSH核验Qwen/Llama3/OLMo depth curves。

### A02

- `proposal/active/A02-comem-write-read-repair/{PROPOSAL.md,STATUS.json}`
- 三个verdict markdown
- `evidence/phase1_full_summary.json`
- `evidence/{babilong,ruler,longbench,longeval,locomo}_paired_ci.json`
- `evidence/locomo_judge_openweight.json`
- `evidence/a02_storage_readcompute_{aggregate,verdict}.json`
- `evidence/a02_storage_readcompute_serve/*.json`
- `evidence/a02_depth_vs_retrieval_{ci,conditional,per_item}.json`

### A04

- `proposal/active/A04-recovery-certification/{STATUS.json,PROPOSAL.md}`
- `A04_GATE_DESIGN.md`
- `A04_MARGIN_GUARD_PREREG.md`
- `PILOT_ZERO_VERDICT.md`
- `SEED_SEMANTICS_DEFECT.md`
- `evidence/pilot_zero_rule_disagreement.json`
- `evidence/a04_{intact_residual_ci_1b_mmlu,margin_guard_classification,pdisc_mmlu_1b_full}.json`
- `code/pilot_zero_rule_disagreement.py`及margin/pdisc scripts

---

## 7. 最终审稿建议

- **A01：Major revision。** 保留“floor是必要条件”和tie机制反证；撤回phase-transition一般定律、oracle convention flip和跨construct一数量级span。
- **A02：Reject / archive原thesis。** 若继续，仅作为窄systems microbenchmark observation重新定位；不能再称storage method或natural-task quality win。
- **A03：Reject当前empirical claim，允许redesign。** floor gate可保留；CPT recovery/LR-regime结论在真正多seed前不应发表；6-arm主实验未完成。
- **A04：Design paper / preregistration only。** Pilot Zero不是gate pass；没有任何模型已被“certified recovered”。
