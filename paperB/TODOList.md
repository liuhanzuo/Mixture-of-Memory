# Paper B：补充实验与分析清单

> 目的：解决匿名审稿中最可能导致拒稿的问题。实验 agent 完成一项后，必须在对应条目填写结果、原始文件路径、代码/命令、checkpoint 和结论；不要只把复选框改为完成。

## 2026-08-04 v6/v7 六评后剩余实验（当前权威优先级）

> **六评结论**：v6 Overall=`2.92`；v7 artifact audit=`3.5/5`，已确认匿名
> score-level artifact 可复算六臂 MMLU 与 keep14 paired headline。Paper B
> 已稳定在 Findings 档。若目标是 ACL 主会，剩余阻断来自新训练/新评测，而不是
> 继续扩写。下面优先级覆盖后文较早版本中与之冲突的 DONE/当前优先级解释。
>
> **特别说明**：旧 P1.1 的 full32@25k 数据仍有效，但不能称
> “200k-equivalent endpoint”，也不能替代真正 full32@200k。

## 2026-08-04 投稿后 reviewer-risk 驱动实验计划

> 已对冻结提交稿 `final.pdf` 做一次严格 PDF-only ARR 模拟审稿；不修改已提交 PDF。最可能的问题依次是：单条历史轨迹/无训练 seed 方差、full32@25k 被当作 200k-equivalent 的可比性、ShortGPT 选层与保留末层等结构混淆、不同策略的训练 token/FLOPs 是否真正匹配、MMLU answer-letter/readout 敏感性、PPL crossing/停止规则是否事后选择、artifact/provenance 可复算性，以及跨模型/跨语料外推。

### Phase R：两周内形成 rebuttal-ready 证据包

1. **完成 B-P1.3 matched-PPL crossing 后立即停止 #103，不跑到 200k。** 当前 clean re-heal 已过 step21k；依次捕获 frozen-match PPL≈12.797 与 random-match PPL≈11.498 的双侧 checkpoint。每点报告实际 $|\Delta\mathrm{PPL}|$、MMLU、McNemar、paired bootstrap；只有 ≤0.10 才称 matched-PPL。random-match 明示 LR confound，主证据优先使用 LR-matched frozen 腿。
2. **先评测 B-P0.4 两个 80k 结构隔离 checkpoint，再决定是否续训。** contiguous16/no-fresh 与 retained-final14+fresh2 先跑 held-out PPL 和 letter/content MMLU；只有出现可解释的结构信号，才追加 closed-book/core6 并续至 200k。不得在无 held-out gate 时直接购买剩余 120k。
3. **冻结 prospective certification rule。** 在查看后续 seed 结果前写定 absolute PPL tax、gap closure、局部 slope/plateau、目标容差及触发规则；calibration trajectory 与 evaluation seeds 分开，避免 post-hoc crossing 指控。
4. **整理无需 GPU 的 rebuttal artifact。** 统一 keep14 MMLU `.3191/.3184` 和 TriviaQA `34.1/34.2` provenance；补 ShortGPT/full32/keep14 closed-book 逐题索引、OOD PPL、contamination clean-subset 和可 clean-extract 复算说明。提交稿中的事实错误只记录为 rebuttal clarification，不静默改写冻结 PDF。

### Phase M：1--2 个月主会增强实验

1. **full32 从头严格训练到 200k。** 使用与结构臂完全相同的数据顺序、token accounting、optimizer/schedule 和评测里程碑；不要从旧 step25k resume 作为严格 counterfactual，因为现有 resume 不恢复 epoch 内 dataloader cursor。旧 25k 仅作探索性轨迹点。
2. **keep14 至少 3 个 clean training seeds。** 预注册 0/25k/50k/100k/128k/153.5k/200k，并报告 seed-level 曲线、SD/层级 CI、certification false-positive/miss；这是把 observed historical path 提升为可复现结论的核心实验。
3. **根据 80k gate 决定是否把 B-P0.4 两臂续到 200k。** 若 80k 已显示 contiguous selection/final-block retention 的可辨信号，完成四臂 factorial；否则先停止并记录无信息结果，不盲目烧卡。
4. **低优先级增强**：补 capability-versus-total-compute 的 token/FLOPs/GPU-hour/推理成本表；之后才考虑 Qwen protocol-complete replication。LR 2×2 仅在未来要提出 initialization 因果 claim 时恢复，否则保持 operating-point wording，不继续高成本 random-init@2e-5。

### 明确停止/启动决策

- `#103`：两个 crossing 与 paired eval 完成即停；不复制已有 200k endpoint。
- `B-P0.4`：A/B@80k 先评测，评测前不续训。
- `full32`：主会增强阶段从头严格重跑 200k；不把 25k resume 称为严格控制。
- `random-init@2e-5`：维持早停；除非升级为 initialization 因果主张，否则不恢复。

### B-AUDIT ARR 审计指控分诊（2026-08-04，用户指令「有结果么，没有的话加 todolist 然后跑」）

> 分诊原则：**有结果 → 改写/修复（无需跑 GPU）**；**无结果 → 加任务 + 立即跑**。
> PaperB 共 3+1 条，其中 3 条属「有结果」，仅 Finding-3 LR 混淆需要跑。

- **B-AUDIT-1（有结果，P0 编译级硬伤）`tab:downstream = Table ??`** — `[✅ DONE 2026-08-04（MAIN 内联修）]`
  根因：`sections/tab_downstream.tex`（2662B，`\label{tab:downstream}`）存在但当前 build 无任何 `\input`（历史 v7/v8 在 `08_appendix.tex` 有，被误删）；3 处 `\ref`（`05_analysis.tex:31`、`08_appendix.tex:58`、`tab_policy_endpoint.tex:22`）全指 `??`。
  修法：在 `04_experiments.tex:6`（`tab_trajectory_audit` 之后）加 `\input{sections/tab_downstream}`。表内容已有，无需跑。task #147。
- **B-AUDIT-2（有结果，P2）keep14 MMLU 两套数字 .3191 vs .3184** — `[TODO 改写，task #147]`
  `tab_downstream/tab_control/tab_trajectory_audit` 用 `.3191`；`app_tab_protocol_controls/tab_interface_audit` 用 `.3184`（差 0.0007，rounding/不同 run）。选定 canonical run 后正文统一。无需跑。
- **B-AUDIT-3（有结果，P2）TriviaQA 34.1 vs 34.2** — `[TODO 改写，task #147]`
  同文件 `04_experiments.tex` L27=34.1、L48=34.2，typo。统一即可。无需跑。
- **B-AUDIT-4（用户 cost-benefit 决策：不跑到底，Finding-3 保留 hedged wording）Finding 3 fresh-block LR=2e-5 混淆** — `[STOPPED @step25260/200000 — 2026-08-04；task #149 已停，.73 8×H20 腾空（0 MiB）。用户 2026-08-04 cost-benefit 表判定 B-P1.1 类 LR-matched control「改 LR 比加 seeds 对贡献提升小」不值 ~200 GPU·h（全 200k ETA ~18.6 天）→ 不付全程。25k 步 <50k 早读里程碑，命中 fallback 分支：Finding-3 正文维持既有 hedged wording（"operating-point comparison…neither comparison alone identifies a causal factor"），不新增 init-vs-LR 因果 claim；step5000…25000.pt 保留可日后 resume]`
  **确认的混淆**：Finding 3 的 same-shape operating-points 子对比（`04_experiments.tex:73-80`, `tab_paired_operating_points.tex`）中，headline **keep14 − Random = +7.14pp** 差在 **≥2 变量**：init（inherited vs random）AND peak LR（keep14=2e-5 vs Random=1e-4）。根因=DDP `module.`-prefix bug（`build_param_groups`）把 keep14 fresh-tail 误分进 2e-5 桶，而 `from_scratch` 走 `"fresh"` 分支免疫 → Random 训在 1e-4。（注：ShortGPT-vs-keep14 子对比**不混淆**，两者同 peak LR；且正文已 hedge Random 对比为 "operating-point comparison…neither comparison alone identifies a causal factor"。）
  **单变量对照 = random-init @ uniform 2e-5**（vs keep14@2e-5，唯一变量=init；shape/corpus/eff_bs128/cosine-200k/trainable-all/seed42 全 matched）。task #127/#149 已在约 step25k 早停，保留 step5000…25000 checkpoint；output `outputs/paperB_finding3_lr_control_randinit2e5/`（zwfy6 .73），launcher `scripts/run_olmo2_p13_resume_1node_73.sh`。全程成本约 18.6 天/~3580 GPU·h，当前不恢复。
  **最终决策**：fallback 已触发——正文维持 operating-point/非因果措辞，不以不足 50k 的早读结果新增 initialization claim。仅当未来将 initialization 升级为因果主张时，才预注册 2×2 后恢复或重跑；否则该控制保持停止。

### B-P0.0 ShortGPT closed-book 与逐题 QA artifact

- **状态**：`[DONE — 2026-08-04，agent aadf4c60，B200 LOCAL 8 卡]`（task #145）。sanity gate 过（strict-load 16 层 shell，held-out PPL=9.7800 复现已知 9.7803），模型加载正确未污染。
- **目的**：最强 16-layer comparator ShortGPT 目前只有 PPL/MMLU；缺少用于排除
  answer-letter artifact 的 PopQA/TriviaQA/NQ-open。
- **模型**：base、full32@25k、keep14@200k、ShortGPT-16@200k；frozen/random
  可复用已有结果。
- **协议**：严格复用 no-retrieval、zero-shot、chat=False、no-BOS、greedy、
  max_new_tokens=32、同 normalization/aliases 的 closed-book harness。
- **必须保存**：
  - stable item ID、normalized prediction、correctness、score；
  - 匿名 artifact 不包含 benchmark question/text；
  - 每任务 paired difference、bootstrap CI；
  - 适用时报告 EM/containment/token-F1 sensitivity。
- **验收**：
  - ShortGPT 全量 PopQA 14,267、TriviaQA 17,944、NQ-open 3,610；
  - 与 base/keep14 同 item IDs；
  - 更新匿名 artifact，使 closed-book 主结论不再只有 aggregate summaries。

**结果（chat=False, no-BOS, zero-shot, no-retrieval, greedy, max_new_tokens=32, bs=32, 全量）**

| 指标 | ShortGPT-16@200k | base_full | keep14@200k | full32@25k |
|---|---|---|---|---|
| PopQA contains-acc (n=14267) | **0.1585** | .2571 | .1415 | .2280 |
| TriviaQA em (n=17944) | **0.3301** | .6355 | .2940 | .5715 |
| NQ-open em (n=3610) | **0.0668** | .2050 | .0598 | .1582 |

- 一句话解读：ShortGPT-16 三项均**略高于 keep14@200k**（16 层剪枝臂里最强），但**远低于 base_full 与 full32@25k**——尽管其 PPL(9.78) 最好、MMLU 在剪枝臂里最高，closed-book 参数化事实召回仍严重退化 → 支撑 "PPL/MMLU 恢复 ≠ closed-book 知识恢复" 的 knowledge-lags 论点。
- 匿名 artifact（仅 summary.json，剥离本地 ckpt 路径 / base_model 改回 hub id / `initialization: inherited_shortgpt` / 不含 question 文本）：`perplexity-heals-knowledge-lags/data/closedbook/shortgpt16_step200k{,_nqopen}/summary.json`（commit 627efe2 于嵌套匿名仓，用匿名身份 pighzliu 提交以保论文匿名性，未 push）。完整工作产物（含 per_example jsonl）：`olmo2_closedbook_results/shortgpt16_step200k{,_nqopen}/`（含 question 文本，故意未 commit）。PPL sanity：`olmo2_ppl_results/shortgpt16_step200k_ppl_sanity/summary.json`。harness `scripts/eval_olmo2_closedbook_qa.py` + `_run_closedbook_8shard.sh` 原样复用（主仓无 code commit）。

### B-P0.1 keep14 主路径的真正 training-run replication

- **状态**：`[TODO — ACL MAIN GATE；6/6 reviewer 共识]`
- **目的**：估计 principal PPL/target trajectory 是否稳定，而非只描述一个历史 run。
- **最低设计**：
  - keep14+fresh2 至少 3 个独立 seeds；
  - 完全相同 OLMo revision、Dolmino array、effective batch=128、seq=2048、
    optimizer/LR schedule、200k steps；
  - 每个 seed 从 step0 开始，保存 RNG states、data permutation、dataloader
    offset、optimizer/scheduler state；
  - checkpoints 预先固定为 0/25k/50k/100k/128k/153.5k/200k，不能看 target
    metric 后选择。
- **评测**：in-domain PPL、letter/content MMLU、PopQA/TriviaQA/NQ-open、
  论文保留的 core6/downstream aggregate。
- **统计**：seed-level curves、mean/SD/range；seed 为最高层级的 hierarchical CI；
  item-level CI 不得代替 training uncertainty。
- **推荐 comparator**：资源允许时 ShortGPT 补 2--3 seeds；最低限度 keep14 三 seeds。

**结果填写**

| Arm | Seed | PPL@200k | MMLU-L/C | PopQA | TriviaQA | NQ | Raw |
|---|---:|---:|---:|---:|---:|---:|---|
| keep14 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| keep14 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| keep14 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### B-P0.2 真正 full32@200k 的同时域 intact counterfactual

- **状态**：`[REOPEN — ACL MAIN GATE；旧 P1.1 的 25k plateau 不满足六评要求]`
- **目的**：判断 keep14@200k deficit 有多少是 pruning/regrowth 特有，而不是
  200k 同语料继续训练或 schedule 的长时域效应。
- **配置**：
  - intact 32-layer OLMo-2；
  - 与 keep14 完全相同 Dolmino stream、token order、effective batch、seq length、
    optimizer、inherited LR schedule；
  - 训练至 200k，不得因 25k PPL plateau 提前停止；
  - 固定保存 25k/50k/100k/128k/153.5k/200k。
- **评测**：与 B-P0.1 完全相同的 PPL、双 MMLU、三项 closed-book、core6。
- **报告**：
  - 同 step/token 的 full32−keep14 差；
  - realized tokens、训练 FLOPs/GPU-hours；
  - full32 是否出现长期 target drift；
  - 不把等 optimizer steps 当等 FLOPs。
- **验收**：200k checkpoint、完整 optimizer/data-order provenance 与 per-item
  outputs 齐全。

### B-P0.3 预注册 PPL monitoring/stopping rule

- **状态**：`[BLOCKED BY B-P0.1/B-P0.2]`
- **目的**：把当前 “improvement alone 不足” 的案例提升为实用问题：哪些预定义
  likelihood rule 会错误认证 target recovery。
- **必须在新 trajectories 出结果前冻结**：
  - absolute PPL-tax threshold；
  - relative closed-gap threshold；
  - slope/plateau rule；
  - target recovery tolerance。
- **分析**：
  - 每 seed/construction 记录 rule 首次触发 step；
  - 计算 false certification、miss、触发时 target deficit；
  - calibration 与 evaluation seeds 分开；
  - 同时加入 OOD PPL rule（B-P1.2）。
- **验收**：看过 target 后挑的 threshold 只能作 post-hoc sensitivity。

### B-P0.4 最小结构 factorial：解释 ShortGPT 优势

- **状态**：`[STOPPED EARLY/PARTIAL @80k — 两个 checkpoint 已保留；先做 held-out PPL + 双 MMLU gate，评测前不续训；对应旧 P0.5]`
- **目的**：拆分 inherited count、contiguous/non-contiguous selection、
  final-block retention、fresh tail 四个耦合因素。
- **最低四臂**：
  1. keep14+fresh2；
  2. contiguous16/no-fresh；
  3. selected16/no-fresh（ShortGPT）；
  4. retained-final14+fresh2。
- **配置**：同 data、effective batch、schedule、200k、显式 seed；固定
  0/50k/100k/150k/200k。
- **评测**：PPL、双 MMLU、三项 closed-book、core6；保存 per-item。
- **验收**：只归因于四臂中被单独改变的因素，不由 endpoint 声称 layer31
  “存储知识”。

### B-P1.1 LR/trainable-set matched controls

- **状态**：`[DEFERRED — random-init@2e-5 已于约25k早停；仅在提出 initialization 因果主张时恢复完整 2×2]`
- **最低设计**：
  - random-init, peak LR=`2e-5`；
  - inherited keep14, peak LR=`1e-4`；
  - frozen-front 与 train-all 使用相同 LR，并加入“同 trainable modules、
    不同 initialization”的 matched pair；
  - 固定同 seed、data、schedule、200k。
- **验收**：形成至少一个干净 2×2；否则继续只称 operating points。

### B-P1.2 OOD PPL 与 contamination audit

- **状态**：`[DONE — 2026-08-04，agent abaeab6f，OOD PPL 池在 .252/LOCAL、contamination 单遍扫 15.505B-token Dolmino 流]`（task #146）。两部分结论均支撑论文主张。
- **OOD token-weighted PPL**（全部 `scripts/eval_olmo2_probe2_ppl.py`，chat=False、no BOS、OLMo-2 packing → 与 in-domain 直接可比）：

  | model | in-domain Dolmino | OOD WikiText-103 | OOD PG-19 |
  |---|---|---|---|
  | base (OLMo-2-7B) | 7.40 | 6.00 | 10.18 |
  | full32 @25k | 7.67 | 6.77 | 10.76 |
  | shortgpt16 @200k | 9.78 | 10.58 | 14.45 |
  | keep14 @200k | 10.56 | 11.56 | 15.43 |
  | random-init @200k | 11.50 | 13.45 | 17.54 |
  | frozen-front @200k | 12.80 | 13.66 | 17.35 |

  **结论**：best→worst 排序（base≈full32 ≪ shortgpt < keep14 < random≈frozen）在三语料全部保持；相对 base 的 PPL 比值 OOD 反而更大（keep14：in-domain 1.43× → wikitext 1.93× → pg19 1.52×）→ **in-domain PPL proxy 未掩盖 OOD 崩塌，prune-then-heal LM 税 OOD 真实复现**。
- **Dolmino train↔eval contamination**（token 13/8-gram containment vs 精确 tokenized 15.505B-token 续训流，反向单遍无下采样，CONTAMINATED≥0.80）：

  | benchmark | n | n13 污染 | n8 污染 |
  |---|---|---|---|
  | MMLU | 14042 | 267 (1.90%) | 484 (3.45%) |
  | TriviaQA | 17944 | 142 (0.79%) | 247 (1.38%) |
  | NQ-open | 3610 | 1 (0.03%) | 24 (0.66%) |
  | PopQA | 14267 | 0 (0%) | 10 (0.07%) |

  **污染可忽略**。MMLU clean-subset 重算（仅两臂有 per-example）：keep14 31.91%→31.82% clean、random 24.61%→24.62% clean；**keep14−random gap 7.30pp → 7.20pp(n13)/7.13pp(n8)，变动 ≤0.17pp** → 主结论去污染后成立。（PopQA/NQ 问句多 <13 token → n13 SHORT/undecidable；n8 序给覆盖仍 <1%。）
- **⚠️ 运维注记**：earlier LOCAL PG-19 数值（full32=1.996/keep14=6.92 等）被并发的 #103 keep14 re-heal 训练 GPU 争用**污染**，已在 .252 干净重跑（`base_pg19=10.175` 两节点复现、base-WikiText=5.9969 与 LOCAL 一致做版本交叉校验）。wzc1 盘现存干净数值。
- **closed-book clean-subset 重算 DONE**（2026-08-04，B-P1.2-sub task #153，agent a0f9c7b6，纯 CPU；`scripts/recompute_closedbook_clean_subset.py` commit **1624d55**，复用 `eval_olmo2_closedbook_qa.score_prediction` 原始 scorer；污染 id 取 `clean_subset_ids.json` 的 `keep_for_recompute_ids`@n13）：TriviaQA 污染 142/17944→clean n=17802、NQ-open 1/3610→n=3609、PopQA 0/14267（clean=full）。**full→clean（TriviaQA/NQ=EM，PopQA=contains，%）**：base_full TriviaQA 63.55→63.46 / NQ 20.50→20.50 / PopQA 25.71；keep14_step200k TriviaQA 29.40→29.27 / NQ 5.98→5.99 / PopQA 14.15。7 臂全算齐，排序 base>full32>shortgpt16>keep14>freezefront≈keep8>fromscratch 完全保持，去污染最大变动 ≤0.15pp（TriviaQA）→ **closed-book 知识表对 Dolmino 污染稳健，与 MMLU clean-subset 结论一致**。PopQA join 坑：audit qid=native PopQA `ex["id"]` vs 我们记的 positional `item_id` id-空间不相交 → clean=full fallback（标 `"popqa_join":"unmatched_report_full"`；因 PopQA 污染=0 数值精确无损）；TriviaQA/NQ `item_id==qid` join 成功。full 数字与各 arm `summary.json` 完全一致（scorer 复用验证通过）。产物 `bench_results/olmo2_dolmino_contamination/closedbook_clean_subset_recomputed.json`（wzc1，21 行 arm×dataset，非 git-tracked）。⚠️ 多数 arm 的 per-example 只在 zwfy6 .73（shortgpt16 在 wzc1），故 scp -O 合并到 .73 一次跑全 7 arm，output json scp 回 wzc1。
- **产出路径**（wzc1 绝对路径）：consolidated `bench_results/olmo2_bp12_ood_contam/deliverable.json`；OOD PPL `ood_ppl_results/{base,full32_step25000,keep14_step200000,shortgpt_step200000,random_step200000,frozen_step200000}_{wikitext103,pg19}/summary.json`；contamination `bench_results/olmo2_dolmino_contamination/{contamination_summary,thresholds,clean_subset_ids,examples}.json` + `per_record_{mmlu,triviaqa,popqa,nq_open}_n13.jsonl`；OOD npy `data/ood_ppl/{wikitext103_test,pg19_test}.npy`。脚本 commit `ec155d7`（build_ood_ppl_npy.py / audit_olmo2_dolmino_contamination.py / _run_ood_ppl_pool.sh / _pg19_rerun_252.sh，未 push）。
- **验收**：达成——in-domain proxy failure、OOD likelihood、benchmark overlap 三者已区分，主 gap 去污染后稳健。

### B-P1.3 严格 matched-PPL checkpoint 轨迹

- **状态**：`[RUNNING — 2026-08-04 22:09 复检：re-heal @LOCAL 8×B200 step ~21.1k/200k（1.56 s/step，healthy）；crossing-eval 已评到 step20k，held-out PPL 从 step15k 13.7987 降至 step17.5k 13.5276、step20k 13.2849。下一目标先 bracket frozen-match≈12.797，再 bracket random-match≈11.498；两 crossing 的 MMLU/McNemar/paired bootstrap 完成后立即停止，不跑到 200k。random-match 有 LR confound，frozen-match 是主干净证据。]`（属 #103 matched-PPL 腿）。
- **目的**：检验“相近 likelihood、不同 target performance”。
- **re-heal run（LOCAL B200，健康，逐项对齐原 keep14 轨迹）**：launcher `scripts/run_olmo2_keep14_densesave_reheal.sh`（commit `d4b0084`，未 push），output_dir `outputs/olmo2_keep14_densesave_reheal`（新目录不覆盖原 ckpt），**save_every=2500 + milestone_every=2500**（新增 `--milestone_every` arg 防 rolling-retention 剪掉非 5000 倍数），inherited init（transplant OLMo-2 front-14 + 2 fresh，4.0604B all-trainable，157 tensors 6-check pass），eff_bs=128（bs16 ga1 ×8）、seq2048、warmup150→cosine over max_steps=200000、AdamW fp32 betas(.9,.95) wd0.1/grad_clip1.0/gc=1、data `/dev/shm/dolmino_now15b.npy`。实测 1.56–1.61s/step、maxmem 122.3GB；step20 loss10.5871 vs 原 11.1134、step40 8.3296 vs 原 8.3723（LR schedule/s-step/maxmem 与原一致，loss step40 收敛）。ETA 到 80k≈34.7h、200k≈3.6d。首个 dense ckpt step2500≈65min。
- **⚠️★ 关键发现（coder 实测，影响本腿解读 + 论文 recipe 文档）——原始 keepN ladder 全臂是 UNIFORM lr=2e-5，非差分 LR**：`build_param_groups` 在 DDP-wrapped 名字（`module.…`）上未 strip 前缀 → fresh tail + lm_head 落入 "inherited" 组，拿 2e-5 而非本意的 1e-4（`logs/olmo2_7B_keep14fresh2.log:15` 单 `inh_decay` 组 base_lr=2e-5、4060.1M params，无 fresh 组）。**keep14 与 frozen-front(12.797) 端点均 @2e-5；但 from_scratch（random-front, 11.498 端点）因 `_classify_param` 对 from_scratch 早返回 "fresh" 不受 bug 影响 → fresh @1e-4**。后续 `module.`-strip 修复使当前代码 fresh→1e-4。
  - **对 matched-PPL 对照的影响**：两个 matched 端点用了**不同 LR**（random-front @1e-4 vs frozen-front @2e-5）→ **random-front 腿的 keep14-vs-random 比较含 LR 混淆**（LR 与结构干预不可分）；**frozen-front 腿（keep14 与 frozen 均 @2e-5）是干净对照**。论文写作须如实披露：优先以 frozen-front matched-PPL 腿作干净"相同 PPL→不同 MMLU"证据；random-front 腿标 LR-confound caveat 或降级。
  - re-heal 用 **uniform 2e-5**（`--lr 2e-5 --min_lr 2e-6`）复现——这是**唯一**能让交叉点 ckpt 落在产出现有 keep14 ckpt（10.827/10.693/10.561）的同一轨迹上的配方。若改差分 fresh 1e-4，交叉点将不对齐现有 keep14。
- **crossing-eval 流水线（.252，共享 wzc1 → 直读 re-heal ckpt）**：轮询新 ckpt → held-out Dolmino PPL（`eval_olmo2_probe2_ppl.py` 同口径）→ 锁定 PPL≈12.797 / ≈11.498 交叉 ckpt（报实际 |ΔPPL|，严格 ≤0.10 否则称 nearest）→ 在两点跑 MMLU per-item（P2.8 口径 + McNemar + paired bootstrap）。若 2500-step 粒度太粗（目标落两 ckpt 间且两侧 >0.10）→ 回报 MAIN 决定加密 save_every。
- **验收**：达不到 0.10 必须称 nearest-PPL，不称 matched-PPL。
- **⚠️ disk**：每 ckpt ~48.7GB（含 optimizer state），dense 到 80k=32 ckpt≈1.56TB（12T free 可容）；定位交叉步后 MAIN 需 prune 非交叉 ckpt 或 kill re-heal 回收盘。

### B-P2.1 跨家族 protocol-complete replication

- **状态**：`[OPTIONAL；仅在 P0/P1 后]`
- **内容**：完成 Qwen base/f12k2 的 content-MMLU 与三项 closed-book，并至少
  新增 seeds 或同家族 intact long-horizon control。
- **边界**：只比较各自 base-normalized recovery，不比较跨语料绝对 PPL。

### B-P2.2 Efficiency/compute accounting

- **状态**：`[OPTIONAL]`
- **内容**：对 keep14、ShortGPT、full32 报 inference FLOPs、latency、memory、
  训练 token/FLOPs/GPU-hours 与 capability-versus-total-compute frontier。
- **目的**：若继续以 compression 为动机，给出实际收益；不能替代主 measurement controls。

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

## 2026-08-03 运行任务审计与今晚队列

> 审计范围：当前 Todo、主稿/附录中的未完成证据声明，以及本地 raw/checkpoint。不要重复运行 P0.1/P0.3/P0.4/P0.6/P0.7/P1.1/P2.3；这些项目已有结果，下一步是回填论文。今晚按以下顺序占用空闲节点，长程任务使用唯一输出目录和显式 seed。

1. **继续，不要重启 P0.5 两个结构隔离臂**：Arm A 本机进程健康，审计时已过 step 10.1k，并有 `step0/5000/10000.pt`；Arm B 按远端现有 run 继续。到 50k/100k/150k/最终平台点后运行 PPL、core6、`aux5_raw`、letter/content MMLU 和逐题 paired eval。
2. **[✅ DONE 2026-08-03] 新增 P0.8（纯评测）**：full32@25k PopQA/TriviaQA/NQ-open 已完成，full32 ≈ base（~89%/~90%/~77% of base，远超 keep14@200k），闭合 P1.1 原验收条件。详见 P0.8 章节。
3. **[⏹ STOPPED EARLY 2026-08-04] P1.3 random-init LR=2e-5 已 kill @~26.9k/200k**：.73+.82 16 卡 IB DDP（scratch16L uniform lr2e-5 seed42 eff_bs128，output `outputs/olmo2_p13_scratch16_lr2e5_uniform`，commit c57c4cb）。用户决策与 P0.5 armA/armB 同样处理（B-P1.1 类不值得跑完），记录 train-loss 曲线（step26860 ppl~25.4）后主动 kill，腾 .73+.82 跑 Paper A。详见 §P1.3。
4. **若有独立空闲训练节点，优先 P0.2 strict matched-PPL dense re-heal**：仅当论文要保留"相同 PPL、不同 MMLU"命题时启动；否则维持最近点措辞即可。随后才是 P1.2 keep14 多 seed。
5. **今晚不启动 P2.1/P2.2/P2.4/P2.5**：tuned lens、activation patching、SFT 和 Qwen protocol-complete eval 都不是当前投稿阻断项；P2.5 还缺专用 harness。

---

# 深度阶梯最新端点（keep{8,10,12,14}，base 协议 chat=False / no-BOS / LL-MC）

> ⚠️ **2026-08-16 更正（#192 A+）**：上一版这里写「所有深度阶梯 arm 均已完整运行至 200k step，论文 Table 4 的 Budget 列统一为 200k」——**那是错的**。三个浅层 arm 从未跑到 200k：轮转策略保留的最深 ckpt 是 keep8@**121000**、keep10@**83500**、keep12@**124000**（分别是 200k 的 60.5% / 41.8% / 62.0%）。Table 4 的 step 列现在如实写 121k / 83.5k / 124k，附录同步。**Table 4 是「已保留端点清单」，不是 step / token / FLOP 对齐的比较。**

| arm（层数） | PPL | PPL tax | core6 | aux5_raw（P0.7 审计） | MMLU（dual-iface letter） | MMLU above-chance recovery | 真实 step |
|---|---:|---:|---:|---:|---:|---:|---|
| base full（32L，参照） | 7.398 | 1.000× | 0.7037 | 0.6637 | .6054 | 100% | 参照（vanilla base，未剪层） |
| **keep8（10L）** | **13.3332** | **1.802×** | **0.5233** | **0.4289** | **.2550** | **1.4%** | `[DONE]` **121000**（60.5% of 200k） |
| **keep10（12L）** | **12.8160** | **1.732×** | **0.5300** | **0.4491** | **.2720** | **6.2%** | `[DONE]` **83500**（41.8% of 200k） |
| **keep12（14L）** | **11.4426** | **1.547×** | **0.5689** | **0.4608** | **.2728** | **6.4%** | `[DONE]` **124000**（62.0% of 200k） |
| keep14（16L） | 10.561 | 1.428× | 0.5938 | 0.4935 | .3184 | 19.2% | `[DONE]` 200000（`PAPERB_KEEP14_200K_EVAL.md`） |

- **2026-08-16 数值更正**：keep8/10/12 的 core6 改用干净单协议 `_v2` 目录（torch 2.13 / BS=8 / 8-of-8 shard / 满额 `n_scored`），故 0.5238→0.5233、0.5303→0.5300、0.5669→0.5689。其中 **keep12 的变化含一处真实缺陷修复**：旧值的 `arc_easy` 只 merge 了 6/8 shard（`n_scored`=1782 而非 2376，shard0/shard5 因 HF 429 被 skip），`.689`→`.694`。
- **MMLU 列统一到 dual-interface letter 快照**（`olmo2_mmlu_content_results/`，两盘 byte-identical，全 9 行同源），故 keep8/10/12 的 MMLU 由 `.2535`/`.2718`/`.2752`（downstream `_know` 口径）改为 `.2550`/`.2720`/`.2728`。**above-chance recovery 分母改用 .6054**。

- **已确认口径**：core6 = mean(HellaSwag / ARC-C / ARC-E / PIQA / OpenBookQA 的 `acc_norm`；WinoGrande 的 `acc`)。MMLU above-chance recovery = (MMLU − 0.25) / (0.6053 − 0.25)。
- **aux5_raw 已由 P0.7 审计固定（2026-08-02，`[DONE]`）**：从 raw `summary.json` 用 plain `acc` 重算 MMLU/LAMBADA/BoolQ/CSQA/SIQA 五任务均值。旧 keep14 `.5071` 确认数值错误（BoolQ/CSQA/SIQA 混入 `acc_norm`，虚高约 1.4pp），**修正为 0.4935**；其余旧值（base `.6637`/keep10 `.4491`/keep12 `.4608`/ShortGPT16 `.5596`）数值正确，仅从 `know5` 重命名为 `aux5_raw`。命名硬约束：`aux5_raw` 是异质任务的描述性均值，**禁止**称 knowledge recovery；MMLU 单列报告。详见 `paperB/P0_7_AGGREGATE_AUDIT.md`。
- **深度结论边界（2026-08-16 更正）**：**不是**「所有臂统一完成 200k」。keep8/10/12 停在 121k/83.5k/124k，只有 keep14/ShortGPT/frozen/random 到 200k，所以**不能声称受预算控制的单调深度效应**。6 组配对里 4 组「更深」与「预算更多」同向（混淆），只有 keep8@121k vs keep12@124k 是事实上等预算（差 2.5%，Δcore6 +4.56pp），且 keep8 比 keep10 多 45% step 却低 0.67pp core6 —— 后者已落在仪器噪声范围内（同权重换 stack 可动 0.5pp、换架构 0.9pp），**不作为排序使用**。
- Provenance：keep8/10/12 用 zwfy6 `_v2` 单协议目录；keep14/ShortGPT/frozen/random/base/full32 沿用旧 stack（torch 2.7，同 BS=8、同满额 `n_scored`），Table 4 caption 已如实披露该 stack 不同质。

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
- **目的**：当前缺少 pruning/regrowth 后、healing 前的同一模型基线，无法严格计算"恢复了多少初始损伤"。
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
- MMLU：keep14@200k **.3191** vs random-front@200k **.2461**（chance）；gap **+7.3pp**（inherited 显著更高）。**gap/CI/p [DONE #103 P2.8]**：per-item 配对（n=14042，同 MMLU test 全集）Δ=**+7.11pp**（keep14 acc .31797 vs scratch .24683，重算口径复现 ledger ±0.005），McNemar exact **p=1.64e-46**，paired bootstrap 95% CI **[+6.14, +8.09]pp**（seed=1234, n_boot=10000）；b(keep14对/scratch错)=2945, c=1946，CI 排除 0 → 高度显著。
- inherited vs frozen 匹配 checkpoint：⚠️ 同样无严格匹配点。frozen-front final PPL 12.797 > 全部 keep14 ckpt PPL。最近点 = keep14@200k（PPL 10.561）vs frozen-front@200k（PPL 12.797）。
- NLL/PPL：keep14@200k **10.561** vs frozen-front@200k **12.797**；差值 **−2.236 PPL**（inherited 更低，|ΔPPL|=2.236 ≫0.10 → 仅近似）。
- MMLU：keep14@200k **.3191** vs frozen-front@200k **.2628**；gap **+5.6pp**。**gap/CI/p [DONE #103 P2.8]**：per-item 配对（n=14042，同 MMLU test 全集）Δ=**+5.50pp**（keep14 .31797 vs frozen .26293，重算复现 ledger ±0.005），McNemar exact **p=6.99e-27**，paired bootstrap 95% CI **[+4.51, +6.49]pp**（seed=1234, n_boot=10000）；b(keep14对/frozen错)=2982, c=2209，CI 排除 0 → 高度显著。McNemar exact 与 normal-approx 交叉核对一致。（附 frozen vs scratch：Δ+1.61pp, p=2.59e-03, CI[+0.58,+2.64]pp — 三臂 knowledge 排序 keep14>frozen>scratch 每对均显著。）
- raw/per-example files：keep14 `olmo2_downstream_results/7B_keep14_step200000_know/summary.json`；random-front `PAPER_B_DATA.md §8`（PPL `olmo2_ppl_results/7B_scratch16L_step200000/summary.json`）；frozen-front `olmo2_downstream_results/7B_freezefront_step200000_know/summary.json`（MMLU .2628）+ PPL `olmo2_ppl_results/7B_freezefront_step200000/summary.json`（12.797）。**逐题预测 [DONE #103 P2.8]**：三臂 per-item MMLU jsonl（各 14042 行，`per_example_mmlu.jsonl`，含 `option_scores` per-letter sum-logprob）——keep14 `.73:olmo2_downstream_results/7B_keep14_step200k_mmlu_peritem/`；frozen `.104:olmo2_downstream_results/7B_frozenfront_step200k_mmlu_peritem/`；scratch `.82(zwfy6 root):/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/olmo2_downstream_results/7B_fromscratch_step200k_mmlu_peritem/`（⚠️ 三处独立物理盘不跨节点）。本地汇总+分析：`.t27_tmp/p28_peritem/`（`{keep14,frozen,scratch}_mmlu.jsonl` + `paired_stats_p28.py` + `paired_stats_result.json`）。eval commit `051c9c3`，launch `scripts/_run_olmo2_mmlu_peritem_8gpu.sh`（commit 8dd4694，未 push）。
- 图路径：NLL–MMLU 散点仍 **待**（strict matched-PPL 腿 #103 BLOCKED——盘无 dense-save 交叉 ckpt）。⚠️ 但 P2.8 逐题 jsonl 里 `option_scores[gold_letter]` = gold-continuation sum-logprob，可直接作 gold-NLL–MMLU 散点的近似替代（若论文需要，无需重跑）。
- 复现命令/脚本：端点数据见 `status/PAPERB_THREE_ARM_200K.md`；**P2.8 逐题 McNemar 腿已 DONE**（见上，2026-08-02，coder a70d2dbc，脚本 `scripts/_run_olmo2_mmlu_peritem_8gpu.sh`）。仅剩**严格 matched-PPL 腿仍 BLOCKED**：需 dense-save re-heal 捕捉 PPL≈11.498/12.797 交叉点 ckpt（盘上无，须新训练；当前用最近点近似上界论证，见下）。
- 对论文结论的影响：**perplexity–knowledge dissociation 无需严格匹配即已成立** —— frozen-front PPL(12.797) 比 random-front(11.498) 更差却 MMLU 更高(.2628>.2461)，PPL 与 MMLU 反向；keep14 在两轴都最优（PPL 最低 + MMLU 最高）。但"**相同** PPL → 不同 MMLU"的严格陈述仍需 #103 补 keep14 在 PPL≈11.498/12.797 处的 MMLU（当前最近点 PPL 差 0.94/2.24，只能作近似上界论证）。

## P0.3 增加至少两个 closed-book knowledge benchmarks

- **状态**：`[DONE — 2026-08-02，PopQA+TriviaQA+NQ-open 全 5 模型跑完（3 个 free-form 知识 benchmark），.73/.104/.82 三节点分片；harness eval_olmo2_closedbook_qa.py commit d05ef59 (+nq_open branch 9fabb88) + launcher 051c9c3；agent af1e5f79 (PopQA/TriviaQA) + a56c04b3 (NQ-open)]`
- **类型**：额外评测，不需训练。
- **目的**：判断分离是否超出 MMLU，避免把单一 benchmark 泛化成广义 closed-book competence。
- **优先任务**：PopQA、TriviaQA closed-book；可补 Natural Questions closed-book 或 MMLU-Pro。
- **最小模型集合**：full base、keep14 final、random-init final、frozen-front final；若预算允许，加入 keep8 final 与 matched-PPL inherited checkpoints。
- **协议要求**：严格 no retrieval、zero-shot；固定 prompt 和 normalization；保存逐样本预测；报告 exact match/F1 的具体实现及 chance/majority baseline（适用时）。
- **判定**：
  - 若至少两个知识任务与 MMLU 同方向，可将主张扩展为 `knowledge-sensitive benchmarks`；
  - 若只有 MMLU 异常，全文继续限定为 `MMLU dissociation`。

**结果填写**

协议（5 模型完全一致）：OLMo-2 BASE 协议 `chat_template=False`、`--add_bos 0`（`add_special_tokens=False`，无 BOS）、zero-shot、**无检索**、greedy（`do_sample=False`, `max_new_tokens=32`），prompt = `Question: {q}\nAnswer:`，预测 = completion 首行，SQuAD-style normalization，8-GPU shard+merge。全量集：**PopQA n=14267，TriviaQA rc.nocontext validation n=17944**（逐样本 jsonl 行数核验 = full n）。所有 ckpt `load_state_dict(strict=True)` 无 missing/unexpected。

| Task | Metric（headline 加粗） | Base | keep14@200k | Random@200k | Frozen@200k | keep8@200k | Raw path |
|---|---|---:|---:|---:|---:|---:|---|
| PopQA (n=14267) | **contains** | **0.2571** | **0.1415** | **0.1112** | **0.1283** | 0.1246 | 见下 |
| PopQA | em / f1 | 0.2475 / 0.2902 | 0.0798 / 0.1296 | 0.0489 / 0.1016 | 0.0974 / 0.1381 | 0.0425 / 0.0873 | 见下 |
| TriviaQA (n=17944) | **em** | **0.6355** | **0.2940** | **0.2086** | **0.2477** | 0.1577 | 见下 |
| TriviaQA | contains / f1 | 0.6742 / 0.6843 | 0.4126 / 0.3718 | 0.3352 / 0.2982 | 0.3240 / 0.3152 | 0.3047 / 0.2426 | 见下 |
| NQ-open (n=3610) | **em** | **0.2050** | **0.0598** | **0.0632** | **0.0496** | 0.0368 | 见下 |
| NQ-open | contains / f1 | 0.2490 / 0.2841 | 0.1343 / 0.1206 | 0.1066 / 0.1208 | 0.0931 / 0.0995 | 0.0925 / 0.0810 | 见下 |

- floors（三任务、跨模型一致）：`empty_em=0.0`；`majority_em` PopQA=**0.0229**、TriviaQA=**0.00256**、NQ-open=**0.0053**。5 模型三任务均远高于 majority floor → 真实信号非偶然。
- † keep8 是**不同架构**（10 层 vs 16 层）且**不同 step**（121k vs 200k），不属严格 keep14-triad，仅作额外深度参考。
- prompt/evaluator：`scripts/eval_olmo2_closedbook_qa.py`（commit d05ef59，CLOSEDBOOK_TASKS=[popqa,triviaqa]；NQ-open 由 `nq_open` task branch commit **9fabb88** 加入，加载 `google-research-datasets/nq_open` validation n=3610，answer=alias list、em=max-over-aliases，与 TriviaQA 同口径；SQuAD normalization，contains=归一化包含，f1=token-F1，PopQA/TriviaQA 行为向后兼容不变）+ `load_pruned_model/load_base_model` from `eval_olmo2_probe2_ppl.py`（无架构漂移，fp32 master + bf16 autocast）。
- 复现命令：`scripts/_run_closedbook_8shard.sh`（8 分片→merge，`--add_bos 0`，bs=32）；数据 stage：`eval_olmo2_closedbook_qa.py --prepare_data --tasks popqa,triviaqa`（NQ-open 用 `--tasks nq_open`，hy-proxy，之后 `HF_HUB_OFFLINE=1`）。Python `/opt/conda/envs/torch-base/bin/python`（torch2.13/transformers5.5.4/datasets5.0.0）三节点一致。
- 是否支持跨 MMLU 泛化：**是（3 个 free-form closed-book knowledge benchmark 一致确认主分离）**。
  - **主分离（base_full vs 全部 matched/剪层-heal 变体）在 3 个任务上一致且大幅**：base_full 远高于任何剪层-heal 变体——PopQA contains 0.257 vs ≤0.142、TriviaQA em 0.636 vs ≤0.294、NQ-open em 0.205 vs ≤0.063。→ 续训不恢复参数化知识，这是知识性而非格式适配（裸 `Answer:` continuation）。
  - **matched keep14-triad 的细粒度序（keep14 > frozen > random）在 PopQA/TriviaQA 上干净成立，在 NQ-open headline(em) 上不完全成立**：NQ em 序为 random(0.0632) ≳ keep14(0.0598) > frozen(0.0496)——random 以 ~0.33pp（3610 中约 12 例，噪声级）微超 keep14，翻转了 keep14>random 这一对；但 NQ **contains** 上 keep14(0.1343) > random(0.1066) > frozen(0.0931)、**f1** 上 keep14≈random≫frozen，keep14 重回榜首。frozen 在三任务上稳定最低。
  - 净判定：**分离方向泛化到 3 个知识 benchmark**（主分离 + frozen 稳定最低一致）；但 keep14 vs random 的更细排序在 NQ em headline 上落在噪声内、非稳健，论文措辞不应把 keep14>random 说成跨全部 benchmark 成立。
- 对论文措辞的建议：**"MMLU dissociation" 可安全扩展为 "knowledge-sensitive benchmarks (MMLU, PopQA, TriviaQA, NQ-open)"**，主张限定在「base_full ≫ 剪层-heal 全变体」这一主分离 + 「frozen-front 最差」；keep14 vs random 的细序仅在 PopQA/TriviaQA 稳健，NQ em 上为噪声级平局，措辞需相应弱化。注：本项比较 QA *知识排序* 与 MMLU 排序；perplexity–knowledge dissociation 的 perplexity 腿在 MAIN 手上的 ppl eval（本 agent 未触碰）。**（措辞落 .tex 由 MAIN 后续处理，不在本回填内。）**
- Raw JSON 绝对路径（各 model dir 含 `summary.json` + `per_example_{popqa,triviaqa}.jsonl` + `shard{0..7}of8.json`；⚠️ 分处 3 个物理盘、均不与 LOCAL 共享，如需汇总到 LOCAL 供论文需 rsync 各节点 `olmo2_closedbook_results/<model>/`）：
  - **.73**（`28.85.35.73:36000`，wzc1 root）：`olmo2_closedbook_results/base_full/`、`.../keep14_step200k/`
  - **.104**（`28.83.24.104:36000`，同路径串但独立 diskB alias）：`olmo2_closedbook_results/fromscratch_step200k/`、`.../freezefront_step200k/`
  - **.82**（`28.82.250.82`，zwfy6 root）：`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/olmo2_closedbook_results/keep8_step200k/`
  - **NQ-open raw**（各 model dir 含 `summary.json` + `per_example_nq_open.jsonl`(3610 行) + `shard{0..7}of8.json`）：.73 `olmo2_closedbook_results/{base_full_nqopen,keep14_step200k_nqopen}/`；.104 `olmo2_closedbook_results/{fromscratch_step200k_nqopen,freezefront_step200k_nqopen}/`；.82 `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/olmo2_closedbook_results/keep8_step200k_nqopen/`。
  - ⚠️ infra 记录（agent 实测）：.73/.104/.82 三节点实为**同一物理卷、不同 alias 挂载**（.73/.104 见 `/apdcephfs_wzc1/share_304376610/...`，.82 见 `/apdcephfs_zwfy6/share_304376610/...`）——GPU 各节点独立故 3 路并行 eval 有效，但 launch helper 需按节点唯一命名避免互相覆盖，且 .82 必须传 `PROJECT_ROOT=<zwfy6 path>`。5 run 全 shard_fail=0。

## P0.4 checkpoint-to-checkpoint paired MMLU 分析

- **状态**：`[DONE（可行切片）— 2026-08-02，keep14 128k→200k + keep8 45k→121k 两配对全完，coder a685a4bb；task #124]`。⚠️ 计划中其余配对**盘上无 ckpt 不可做**（见下"未做"）。
- **类型**：离线分析；若缺逐题预测，仅需重新评测 checkpoint，不需训练。
- **目的**：区分 marginal accuracy 变化与同题目的正确/错误翻转。
- **比较对（实际可行）**：keep14 128k→200k（apex→endpoint，★）、keep8 45k→121k（mid→endpoint）。
- **必须报告**：共同题数、wrong→right、right→wrong、净变化、McNemar exact p、paired bootstrap 95% CI；核验 evaluation IDs 完全对齐。

**结果填写**

口径：n=14042（同 MMLU test 全集，item_id 完全对齐，n_nan=0），add_bos=False、mode=mc、BS=8、max_len=1024；row=**后期 ckpt**，Δpp=later−earlier；wrong→right = earlier 错/later 对（=下表 b），right→wrong = earlier 对/later 错（=c）；McNemar exact two-sided binomial（log-space lgamma，已与 chi-square/normal-approx 交叉核对一致）；paired bootstrap 95% CI（seed=1234, n_boot=10000）。

| Pair | n | wrong→right | right→wrong | Δ pp | McNemar p | 95% CI (pp) | Raw path |
|---|---:|---:|---:|---:|---:|---|---|
| **keep14 128k→200k**（★apex→endpoint）| 14042 | 1038 | 802 | **+1.681** | **4.12e-08**（显著）| **[+1.075, +2.286]**（排除0）| keep14@128k `.73:olmo2_downstream_results/7B_keep14_step128000_mmlu_peritem/`；@200k `.73:.../7B_keep14_step200k_mmlu_peritem/`（复用 P2.8）|
| **keep8 45k→200k**（mid→endpoint）| 14042 | 1470 | 1437 | **+0.235** | **0.553**（n.s.）| **[−0.513, +0.983]**（含0）| keep8@45k `.104:olmo2_downstream_results/7B_keep8_step45000_mmlu_peritem/`；@200k `.104:.../7B_keep8_step200000_mmlu_peritem/` |

- per-ckpt 聚合 MMLU acc + ledger 核验：keep14@128k **0.301168**（== `_know` ledger exact ✓）、keep14@200k **0.317975**（复用 P2.8 ✓）、keep8@45k **0.252101**（介于 ledger keep8@44k .24633 与 @48k .25004，一致 ✓）、keep8@200k **0.254451**（200k endpoint）。gold-NLL Δ(later−earlier)：keep14 −0.0210、keep8 −0.0388。
- **未做的计划配对（盘上无 ckpt）**：keep14 153.5k（step153500.pt 已清）；keep8 10k/25k/44k（盘上只有 45k/47.5k/48k/200k）。P0.2 的两组 matched-PPL 配对属 #103 matched-PPL 腿（BLOCKED，需 dense-save re-heal）。
- 分析脚本/命令：dump 用 `scripts/_run_olmo2_mmlu_peritem_kf_8gpu.sh`（KF 参数化，默认 14/2 与原 keep14 launcher 字节一致）；分析 `python scripts/analyze_traj_paired_mmlu.py --pair NAME earlier.jsonl later.jsonl labelA labelB --json_out out.json`（numpy only，McNemar lgamma，bootstrap RandomState(1234)）。本地汇总+分析产物：`analysis/traj_p04/`（`keep14_pair.json`、`keep8_pair.json` + 4 个 per-item jsonl 各 14042 行）。commit `918b6ed`（2 脚本，committer LiuHanzuo，未 push）。48GB keep14@128k.pt 只读 `ck["model_state"]`（optimizer state 忽略，CPU map_location，无 OOM）；strict load 全通过（keep14=179 tensors/16 层，keep8=113 tensors/10 层）。
- 对论文结论的影响：**深臂（keep14）沿 heal 轨迹 apex→endpoint 有统计显著的知识增益**（+1.68pp，p=4e-8，CI 排除 0）——即 apex 之后继续训练仍在获取参数化知识，不只是拟合 PPL。**浅臂（keep8，全程 ≈chance .25）MMLU 无显著变化**（+0.24pp，p=0.55，CI 含 0），但 gold-NLL 仍略降 → 校准/置信度改善而非 accuracy 增益。→ 支持"知识恢复依赖足够的继承深度"的叙事。（措辞落 .tex 由 MAIN 后续，不在本回填内。）

## P0.5 ShortGPT 结构隔离控制

- **状态**：`[STOPPED EARLY @~81k/200k — 2026-08-04，用户决策：B-P0.4 4-arm factorial 不值得跑（~270 GPU·h），空出 B200 补 B-P0.0/B-P1.2 quick-win eval]`（task #118）。**两臂在 step ~80.7k/81k（≈40% budget）主动 kill，非崩溃**：Arm A（LOCAL 8×B200）+ Arm B（.252 8×B200）的当前训练曲线已记录在下表，checkpoint 保留至 `step80000.pt`（每 5000 一存）。曲线单调下降、健康，但按 Paper B 重定位为 **"A Measurement Protocol for Post-Intervention Recovery Assessment"**，此 4-arm 结构归因实验优先级降级，不再续训到 200k。历史配置（供复现）：Arm A = contiguous16/no-fresh（`train_olmo2_shortgpt.py`，keep 0–15，n_fresh=0，单 LR 2e-5，输出 `outputs/olmo2_p05_armA_contig16`，log `logs/olmo2_p05_armA_b200.log`）；Arm B = retained-final14+fresh2（`train_olmo2_shortgpt_fresh.py`，keep [0-12,31]+2 fresh，split LR 2e-5/1e-4，输出 `outputs/olmo2_p05_armB_final14_fresh2`，log `logs/olmo2_p05_armB_b200_252.log`）。两臂均用 `.venv`（torch2.13）严格同栈。共同：batch 2 × grad_accum 8 = eff 128，seq 2048，(计划 200k) steps，save_every 5000，extra_save 50k/100k/150k，seed 42，`/dev/shm/dolmino_now15b.npy`。实测 1.77s/step（vs H20 9.5s = 5.4× 加速）。⚠️ 下表 PPL 列为 **train-loss 派生 ppl**（非 held-out eval PPL）；early-stop 未跑 core6/MMLU/held-out 下游 eval。若 reviewer 强烈要求此对照，可从 `step80000.pt` resume 续训。
- **类型**：高价值结构归因训练；优先级低于 P0.6/P0.7 与正在运行的 P1.1，不阻断当前描述性 claim。
- **目的**：拆分 ShortGPT-16 相对 keep14+fresh2 的继承层数与选层/final-layer retention 混淆。
- **Arm A — contiguous16/no-fresh**：继承原始 layers `0–15`，0 fresh，总深度 16；与 ShortGPT 严格匹配 inherited count、shape、数据和 200k budget。
- **Arm B — retained-final14+fresh2**：继承 `[0–12,31]` 共 14 层，再追加 2 fresh，总深度 16；与 keep14 匹配 inherited/fresh counts、shape、LR 和 budget，仅替换一个 prefix block 为 layer31。
- **共同配置**：同 OLMo base、Dolmino stream、seq 2048、effective batch 128、optimizer/LR schedule、200k max steps；显式固定 seed。
- **评测点**：step0/50k/100k/150k/200k；PPL、core6、完整逐任务结果、两种 MMLU protocol；保存逐题预测。aggregate 仅在 P0.7 固定口径后使用。
- **验收**：ShortGPT vs Arm A、keep14 vs Arm B 分别报告 PPL gap、paired MMLU gap/CI/McNemar；不得由 endpoint 直接宣称知识存储在 layer31。

**train-loss ppl 曲线（early-stop @ ~81k，2026-08-04；PPL=exp(train_loss)，非 held-out eval）**

| Arm | step20k | step40k | step60k | step80k | 停点 | 保存 ckpt | 备注 |
|---|---:|---:|---:|---:|---:|---|---|
| contiguous16/no-fresh (Arm A) | 12.49 | 11.89 | 11.21 | 10.40 | 10.64 @80.68k | `outputs/olmo2_p05_armA_contig16/step{0..80000}.pt`（每 5k） | 单 LR 2e-5，gnorm~0.45，1.78s/step，maxmem 98.3GB |
| retained `[0–12,31]`+fresh2 (Arm B) | 12.75 | 12.13 | 11.48 | 10.65 | 11.42 @81k | `outputs/olmo2_p05_armB_final14_fresh2/step{0..80000}.pt`（每 5k） | split LR 2e-5/1e-4，gnorm~0.50，1.77s/step，maxmem 98.3GB |

- config/seed/environment：OLMo-2-1124-7B base；Dolmino `/dev/shm/dolmino_now15b.npy`；seq 2048；eff batch 128（bs2×ga8×8gpu）；warmup 150；wd 0.1；grad_clip 1.0；gradient_checkpointing；seed 42；`.venv` torch2.13；Arm A=LOCAL 8×B200，Arm B=.252 8×B200。
- paired analysis：`N/A — 两臂 early-stop @~40% budget，未跑 held-out PPL / core6 / MMLU / paired McNemar；仅有上表 train-loss 曲线。`
- 对论文结论的影响：`此 4-arm 结构隔离 factorial（B-P0.4）按 2026-08-04 用户决策不值得完整跑（~270 GPU·h），Paper B 重定位为 measurement-protocol 论文后降级。ShortGPT-16 vs keep14 的结构混淆讨论改为在正文以 caveat/limitation 陈述，不以此对照 endpoint 支撑。曲线仅记录当前健康趋势（两臂均单调下降，Arm A 略低于 Arm B），供后续如 reviewer 强烈要求时从 step80000.pt resume。`

## P0.6 MMLU answer-text / content-likelihood 重评

- **状态**：`[DONE — 2026-08-02，9/9 arm，.73+.104 8×H20 diskB]`。全 9 arm 双协议（letter + content_raw + content_norm）跑完，每 arm `n_valid=14042 / n_nan=0`，full 14,042-item `cais/mmlu`，base 协议 chat_template=False / add_bos=0 / LL-MC。harness 已 rsync 到 diskB（.73/.104 用 torch-base py 因 .venv 坏）。双协议 harness：`scripts/eval_olmo2_mmlu_content.py` + `scripts/_run_olmo2_mmlu_content.sh`（commit `d2e28f2`，未 push）。**★harness bug 修复（load-bearing）**：`mcnemar_exact_p` 在 full-set merge 时 `math.comb(n,i)*(0.5**n)` 于 n=b+c≈数千时 `OverflowError`（per-shard n 小未暴露，merge 才炸）→ 重写为 log-space（lgamma + log-sum-exp），小 n 与旧公式一致、n≈9000 不再溢出，commit `324a44f`（committer LiuHanzuo，无 AI 署名，未 push，已 md5 校验同步两远端）。产出 letter / content_raw / content_norm(headline，token-normalized) 三口径 + 57-subject + within-arm McNemar/bootstrap + 跨臂 `--compare`（24 个 compare JSON）。
- **目的**：区分当前 letter protocol 的低分究竟来自 subject competence、content-to-letter binding，还是两者共同作用；不通过 SFT 或训练"把 MMLU 做高"。
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

**结果填写（2026-08-02，全 9 arm；recovery = (acc−0.25)/(base_acc−0.25)，分母=对应协议 full-base above-chance）**

| Arm | letter acc | content_raw | content_norm(headline) | letter_rec | cnorm_rec |
|---|---:|---:|---:|---:|---:|
| full base (32L) | .6054 | .4400 | .4706 | 1.000 | 1.000 |
| full32 continued-pretrain @25k | .5877 | .4344 | .4662 | 0.950 | 0.980 |
| keep8@200k | .2550 | .3219 | .3423 | 0.014 | 0.418 |
| keep10@200k | .2720 | .3230 | .3445 | 0.062 | 0.428 |
| keep12@200k | .2728 | .3419 | .3629 | 0.064 | 0.512 |
| keep14@200k | .3184 | .3548 | .3832 | 0.192 | 0.604 |
| frozen-front@200k | .2624 | .3349 | .3604 | 0.035 | 0.501 |
| random-init@200k | .2470 | .3439 | .3598 | **−0.009** | 0.498 |
| ShortGPT-16@200k | .4742 | .3764 | .4012 | 0.631 | 0.685 |

- 全 arm n_valid=14042 / n_nan=0；within-arm McNemar p ≤ 1e-32；24 个跨臂 vs-base compare JSON 已存。letter_acc 与已发布数字逐项复现（base .6054 vs P0.7 .6053；full32 .5877 vs P1.1 .5867；各 keep arm 与 P0.7 对齐）。
- evaluator commit/command：`scripts/eval_olmo2_mmlu_content.py`（harness `d2e28f2`；mcnemar log-space 修复 `324a44f`，均未 push）；driver `scripts/_run_olmo2_mmlu_content.sh`（env: TAG/CKDIR/STEPS/KEEP_FRONT/N_FRESH/KEEP_INDICES，PY=torch-base）；4-arm/节点 8-GPU shard，dual-protocol，full 14042-item。
- prompt/candidate manifest：letter=题目+`A./B./C./D.`四选项，`Answer:`后比 4 个单字符 likelihood；content=label-free（题目+`Answer:`），4 candidate 为完整 option text，报 raw sum-logprob + continuation-length-normalized logprob（headline=content_norm）。同 `cais/mmlu` test 14042 题 / 同 tokenizer/BOS/截断 / 同 item IDs。
- **对论文标题/摘要/讨论的影响**：content 协议在每个 depth arm 都比 letter 恢复更多 above-chance（keep14: 19.2%→60.4%）→ 支持 **answer-symbol/readout-binding lag** 解释（解释规则①）。**但 random-init 控制是决定性的**：其 letter recovery ≈0（−0.9%，纯 chance，零学科能力）而 content_norm=.3598（49.8% "recovery"）→ content_norm 有一个与知识无关的大 fluency floor。∴ content≫letter gap 支持 readout-lag，但 content recovery 必须相对此 floor 解读，**不等于 "knowledge recovery"**；competence lag 仍在（content 也远低于 base）。full32 intact-continued-pretrain 在三协议几乎复现 base（content_norm 98%，McNemar p=0.062）= 干净无损上界，验证 harness。ShortGPT-16 两协议均更高（policy dependence 保持，解释规则③）。
- Deliverables（LOCAL wzc1 绝对路径）：consolidated `olmo2_mmlu_content_results/P0_6_content_mmlu_summary.json`；per-arm `olmo2_mmlu_content_results/<TAG>/summary.json`+`per_example_mmlu.jsonl`（9 arm，各含 57-subject）；cross-arm `olmo2_mmlu_content_results/*_vs_base_*_compare.json`（24 个）。远端 raw 亦在 `.73`/`.104:/apdcephfs_wzc1/.../olmo2_mmlu_content_results/`。

## P0.7 core6 / aux5 聚合口径与 raw JSON 审计

- **状态**：`[DONE]`（2026-08-02，无模型运行，纯 raw-JSON 重算）
- **目的**：修复原 `know5` 定义与已发布汇总值的疑似冲突，并停止把异质任务平均值解释成单一"闭卷知识轴"。
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
| keep10@200k | 0.5303 | 0.4934 | 0.4491 | 与旧一致 | `.82:...7B_keep10_step200000{,_know}/` |
| keep12@200k | 0.5669 | 0.5073 | 0.4608 | 与旧一致 | `.82:...7B_keep12_step200000{,_know}/` |
| keep14@200k | 0.5938 | 0.5371 | **0.4935** | **旧 .5071 数值错误（acc_norm 污染，虚高 ~1.4pp），已修正** | `PAPERB_KEEP14_200K_EVAL.md` |
| ShortGPT-16@200k | 0.6215 | 0.5811 | 0.5596 | 与旧一致 | `.82:...7B_shortgpt16_step200000{,_know}/` |
| frozen-front@200k | 0.5631 | 0.4928 | 0.4468 | 新增 | `.82:...frozen_front_step200000{,_know}/` |
| random-init@200k | 0.5584 | 0.4909 | 0.4419 | 新增 | `.82:...random_init_step200000{,_know}/` |

- audit script/commit：`scripts/audit_olmo2_aggregates.py`（commit `c6629cd`，未 push）
- corrected files：`paperB/P0_7_AGGREGATE_AUDIT.md`、`paperB/P0_7_aggregate_audit.csv`、`paperB/P0_7_aggregate_audit.json`；raw `summary.json` 已 materialize 到 `paperB/data/raw/olmo2_downstream_results/`（gitignored）；本 TODOList 深度阶梯表 + ShortGPT § 已回填；`.tex` 已确认无 `know5`。
- 对论文结论的影响：仅 keep14 aggregate 需从 `.5071`→`.4935`（不改变 dissociation 结论方向）；aux5_raw 明确为异质任务描述性均值，禁称 knowledge recovery，MMLU 单列报告。**MAIN 待回填**：status/RUN_REGISTRY.md、status/PAPERB_THREE_ARM_200K.md 的 know5→aux5_raw 表头与 keep14 值。

## P0.8 full32 continued control 的 closed-book 补评测

- **状态**：`[DONE — 2026-08-03；full32@25k ≈ base，闭合 P1.1 knowledge-task 腿]`（@.104 8×H20，agent a06d44436931e2db3）。
- **目的**：P1.1 已证明 full32@25k 在 PPL/core6/MMLU 上近乎无损，但原验收条件还要求 P0.3 的独立知识任务；当前 `olmo2_closedbook_results/` 没有 full32 raw。
- **checkpoint**：`outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt`（32 inherited layers、0 fresh，实测 87.6GB / 87,583,881,932 bytes）。
- **协议**：严格复用 P0.3 的 no-retrieval、zero-shot、chat=False、no-BOS（add_bos=0）、greedy（max_new_tokens=32）与 normalization；跑 PopQA、TriviaQA、NQ-open 全量并保存逐题 JSONL。
- **结果**（full32@25k，keep_front=32/n_fresh=0，base 协议）：

  | Task | n | em | contains | f1 |
  |------|-----|--------|----------|--------|
  | PopQA    | 14267 | 0.1842 | **0.2280** | 0.2348 |
  | TriviaQA | 17944 | **0.5715** | 0.6838 | 0.6389 |
  | NQ-open  | 3610  | **0.1582** | 0.2443 | 0.2369 |

- **与 P0.3 三臂对照**（full32 ≈ base，远超 pruned keep14@200k）：

  | Model | PopQA contains | TriviaQA em | NQ em |
  |-------|---------------|-------------|-------|
  | base_full | .2571 | .6355 | .2050 |
  | **full32@25k (本项)** | **.2280** | **.5715** | **.1582** |
  | keep14@200k | .1415 | .2940 | .0598 |

- **结论**：full32@25k 保留 base 绝大部分 closed-book 知识（PopQA contains ~89% of base、TriviaQA em ~90%、NQ em ~77%），三任务均**远高于**重剪的 keep14@200k（TriviaQA em 近 2×、NQ em 2.6×）。与 base 的小残差与温和的 continued-pretraining corpus shift 一致，非知识破坏。→ **强化控制结论：剪层臂的大幅 closed-book 知识损失来自 pruning/policy，而非 Dolmino continued-pretraining corpus 本身。**
- **raw**：`olmo2_closedbook_results/full32_step25000/summary.json`（PopQA+TriviaQA，per_example_popqa.jsonl / per_example_triviaqa.jsonl）、`olmo2_closedbook_results/full32_step25000_nqopen/summary.json`（NQ-open，per_example_nq_open.jsonl）。
- **核验（全绿）**：strict `load_state_dict(strict=True)` 355 tensors 无 missing/unexpected（两 batch 均 log `[pruned] ... num_hidden_layers=32 (355 tensors, strict)`）；per-example jsonl 行数 = 全 n（14267/17944/3610）；8-of-8 shards，`shard_fail=0`，无 Traceback/OOM/mismatch；meta 确认 mode=pruned/keep_front=32/n_fresh=0/add_bos=false/max_new_tokens=32。env `/opt/conda/envs/torch-base/bin/python`（torch2.13/transformers5.5.4/datasets5.0.0），HEAD ac01684 无改动，无代码改动。
- **复现入口**：
  - `OUTPUT_NAME=full32_step25000 MODEL_ARGS="--ckpt outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt --keep_front_layers 32 --n_fresh_layers 0" TASKS=popqa,triviaqa bash scripts/_run_closedbook_8shard.sh`
  - `OUTPUT_NAME=full32_step25000_nqopen MODEL_ARGS="--ckpt outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt --keep_front_layers 32 --n_fresh_layers 0" TASKS=nq_open bash scripts/_run_closedbook_8shard.sh`

---

# P1：显著提高可信度，需要训练

## P1.1 完整 32 层 continued-pretraining control

- **状态**：`[PARTIAL / REOPEN]`（25k intact control 与 closed-book 已完成，
  数据仍有效；但 2026-08-04 六评一致认为 25k plateau 不能视作
  200k-equivalent，无法控制 keep14@200k。真正 full32@200k 由顶部 B-P0.2 接管。）
- **目的**：分离 pruning/regrowth 效应与 Dolmino continued-pretraining/corpus shift。
- **配置**：原始 OLMo-2-1124-7B，32 inherited layers、0 fresh；同 Dolmino stream、tokenization、effective batch、sequence length、optimizer 和 inherited LR schedule。
- **实际 endpoint**：avg NLL **2.03730**、PPL **7.66986**（8,384,512 held-out tokens / 4,096 windows）、MMLU **.58674**（14,042/14,042，0 NaN）、content_norm **.4662**；checkpoint `outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt`。
- **raw**：`olmo2_ppl_results/7B_full32_step25000/summary.json`、`olmo2_downstream_results/7B_full32_step25000{,_know}/summary.json`、`olmo2_mmlu_content_results/P0_6_content_mmlu_summary.json`。
- **复现入口**：训练 `scripts/_run_olmo2_full32_dolmino_heal.sh`；endpoint battery `scripts/_run_olmo2_eval_full32_plateau.sh`。保留的 `step5000/10000/15000/20000/25000.pt` 可作轨迹补评测，但不是当前投稿阻断项。
- **剩余缺口**：需要真正训练/保存 full32 至 200k，并在相同 checkpoints 完成
  全 battery；25k 结果只能支持 short-horizon closer-to-base。
- **对论文结论的影响**：intact-32L continued pretraining 近乎复现 vanilla base，是深度阶梯与 ShortGPT 的无损上锚；论文应删除"缺 full-32L control"的旧限制。

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

- **状态**：`[STOPPED EARLY @~26.9k/200k — 2026-08-04，用户决策：与 P0.5 armA/armB 同样处理（B-P1.1 LR-matched controls 类 ~200 GPU·h 不值得跑完），记录当前曲线后主动 kill，腾 .73+.82 16 卡跑 Paper A]`（task #127）。random-init peak LR 2e-5 臂在 step ~26860（≈13.4% budget）主动 kill，**非崩溃**：16 卡 IB DDP（.73 node_rank0 + .82 node_rank1），`train_olmo2_arch_probe2.py --from_scratch --keep_front_layers 14 --n_fresh_layers 2 --lr 2e-5 --min_lr 2e-6 --batch_size 2 --grad_accumulation_steps 4 --seq_len 2048 --max_steps 200000 --seed 42`（eff_bs 128，uniform init），output `outputs/olmo2_p13_scratch16_lr2e5_uniform`，commit c57c4cb，data `/dev/shm/dolmino_now15b.npy`，4.68s/step maxmem 98.3GB。**train-loss 派生曲线**：step20 loss11.5742/ppl106323 → step12000 loss3.5706/ppl35.54 → step24080 loss3.2579/ppl26.00 → **step26860(last) loss3.2356/ppl~25.4**。⚠️ 未跑 held-out PPL/MMLU/closed-book（未到平台点，无收敛 endpoint）。checkpoint 每 5000 存（step0/5000/.../25000.pt 在 H20 FS `.73:outputs/olmo2_p13_scratch16_lr2e5_uniform/`），若 reviewer 强烈要求可 resume 续训到 200k。
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

- **状态**：`[HARNESS-READY — GPU run 待 .104 空出]`（harness 已建 + CPU 静态验证，commit `f2f524b`，未 push；无 GPU run。ckpt 在 LOCAL wzc1，需 rsync 到运行节点后先做 32-item sanity：restore k=0 / graft_layer=-1 必须逐题复现 published plain keep14）
- **类型**：额外机制实验。
- **目的**：区分"前 14 层已有可用信息但 fresh tail 无法读出"与"所需计算依赖被删除的上层"。
- **候选设计**：逐块恢复原始 upper layers；或将 base layer-14 hidden states 接入 original/fresh tail；或对关键层做 activation patching。
- **必须同时报告**：PPL 与 MMLU，避免只看单一指标。
- **已建 harness**（2026-08-03，task #128）：`scripts/eval_olmo2_activation_patching.py` + `scripts/_run_olmo2_actpatch_8gpu.sh` + `paperB/P2_2_actpatch_NOTES.md`。扫 `GRAFT_LAYERS="13 16 20 24 28 31"` × `RESTORE_KS="0 2 4 6 9 12 18"`，每点跑 PPL + MMLU(n=14042, LL-MC)，8-shard+merge，base 协议(add_bos=0/chat=False/greedy)硬编码。ckpt：keep14 `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt`(48.7GB) + base OLMo-2-1124-7B，均在 LOCAL wzc1，需 rsync 到运行节点。`base_head` 变体去 tail-can't-digest 混淆；`L=13` 控制点校准 OOD。

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

- raw/checkpoint path：`outputs/qwen3_minarch_armB_f12k2_200k/final.pt`（healed 200k，47GB，含 optimizer；loader 只读 model_state；⚠️ 旧 TODOList 写的 `outputs/qwen3_probe2_8B_f12k2/` 不存在，已更正）；逐任务 summary 见 `qwen3_ppl_results/` + `qwen3_downstream_results/`（.73 diskB）。base full-36 = `models/Qwen3-8b-local`（symlink→`models/Qwen--Qwen3-8b`，即 f12k2 carve 自的同一权重）。
- **当前可支持结论**：Qwen endpoint 同样表现为较高 PPL tax 与 chance-level answer-letter MMLU；它是跨家族诊断点，不是 compute/data-matched replication，也不能与 OLMo 三点拼成纯 depth law。
- ⚠️ **论文写作注意**：Qwen 与 OLMo 的 PPL/tax、训练语料和 depth 均不可直接对齐；P0.6/P0.7 已完成，但 Qwen 尚未运行对应的 content-MMLU 与 closed-book 协议，因此当前跨家族结论仍只限于 answer-letter MMLU endpoint，完整补评测见 P2.5。

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

## P2.5 Qwen 跨家族点的 protocol-complete 补评测

- **状态**：`[HARNESS-READY — GPU run 待 .104 空出]`（harness 已建 + CPU 静态验证，commit `d5f2b9e`，未 push；无 GPU run）
- **目的**：P2.3 当前只有 letter-MMLU 与 PPL/core6，尚未满足其原计划中的 P0.3 closed-book 知识任务，也未检查 P0.6 揭示的 answer-symbol/readout 敏感性。
- **最小集合**：Qwen3-8B base 与 f12k2@200k；两臂运行 letter/content_raw/content_norm MMLU、PopQA、TriviaQA、NQ-open，协议与 OLMo 对齐并保存逐题 option scores/predictions。
- **前置工作**：从现有 `eval_qwen3_probe2_{ppl,downstream}.py` 的 strict loader 抽取统一 Qwen loader，移植而非复制 P0.6/P0.3 的 evaluator；先做 32-item base sanity，与已有 letter-MMLU 聚合对齐后再全量。
- **已建 harness**（2026-08-03，task #129）：`scripts/eval_qwen3_mmlu_content.py` + `scripts/eval_qwen3_closedbook_qa.py` + `scripts/_run_qwen3_p25_8gpu.sh` + `paperB/P2_5_qwen_protocol_NOTES.md`。移植（非复制）Qwen strict loader；letter-MMLU 分支与 P2.3 byte-identical。BOS 对齐：Qwen `bos_token=None`，`add_special_tokens` T==F → `--add_bos 0` = OLMo-equivalent no-BOS。ckpt = `outputs/qwen3_minarch_armB_f12k2_200k/final.pt`（`--keep_front_layers 12 --n_fresh_layers 2`），base = `models/Qwen3-8b-local`，均在 LOCAL wzc1。**验证门（信任 content 数前必过）**：full base letter_acc 须 ≈ .7297、full f12k2 letter_acc 须 ≈ .2495（复现 P2.3）；否则 harness/base 未对齐即停。
- **验收**：MMLU n=14,042，closed-book 三任务样本数与 P0.3 一致，无 NaN/缺 shard；base 与 f12k2 使用同 tokenizer、BOS、prompt、截断和 metric。跨家族只比较各自 base-normalized recovery/方向，不比较绝对 PPL。

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
