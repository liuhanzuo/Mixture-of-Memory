# Proposal Repository

本目录是 Mixture-of-Memory 项目的**唯一提案索引**。目标不是复制所有实验状态，
而是让每个方向只有一个最新、可执行、可证伪的入口。

最后整理：2026-08-11。

## ★★ 晋升规则（2026-08-11 用户指令，覆盖此前所有「必须全部满足才能晋升」的写法）

**用户原话：「我感觉没必要晋升条件卡这么死(所有的proposal都是) 只要有发现就可以继续做」**

**默认继续做，不是默认卡住。** 一个方向只要**有经核实的发现**，就可以继续投入 ——
不需要先凑满一张预设的条件清单。

### 为什么改（这不是放松标准，是修一个真实的 bug）

A01 是活证据。它的「独立成篇 gate」第 1 条要求
*「full-fp32 forward 消除 ties，**并恢复 letter validity**」*。
而 `GATE3_VERDICT.md` 实测结论是 **`MECHANISM_FALSIFIED`**：
fp32 移除了 **100%** 的 ties、改变了受损臂 **18.03%** 的 argmax 决策，
但 letter accuracy **完全没动**（Δ = −0.0015，McNemar **p = 0.570**），
且受损臂在 fp32 下**更显著**地低于自己的 floor（−1.54 pp, p = 0.0062）。

**按字面读，这条门永远不可能通过** —— 它把「这个缺陷是个可修的数值 bug」
当成了晋升前提。可实际结果**更强**：一个能被 fp32 修掉的缺陷不值得写论文，
修不掉的才值得。旧规则会把最好的结果判为不达标。

**教训**：预设的 gate 是在**知道答案之前**写的。当实测结果落在
「既不是 H1 也不是 H2 而是第三种情况」时，**要改的是 gate，不是把发现丢掉**。

### 现在的规则

| 情形 | 处置 |
|---|---|
| 有经核实的发现（数字可复算、provenance 完整） | **继续做。** 写进 `claims/`，标明范围，不必等其他条件 |
| gate 结果落在预设之外的第三种情况 | **改写 gate**，并在 proposal 里记下为什么改（旧文本保留、标 SUPERSEDED） |
| 发现被证伪 | 记 retraction（**不得静默删除**），方向可继续用剩下的证据 |
| 主张与已发表工作**完全相同** | 才收窄或归档；「有重叠」不是理由（见 `memory/prior-work-differentiate-dont-abandon`） |

`active/` 与 `backlog/` 的区别现在**只是资源排序**，不是资格判定 ——
backlog 不代表「不许做」，只代表「当前没排到卡」。

**仍然强制的**（这些是诚实性要求，不是门槛）：
1. 每个数字有 provenance，能从盘上原始文件复算；
2. 撤回史保留，不得静默改写；
3. 跨 arm 比较前确认**同口径**（同 task set、同 seqlen、同 scorer）；
4. Related Work / 新颖性边界章节仍要写 —— 但它的作用是**划清主张边界**，
   不是当作准入考试。

## 状态定义

- `active/`：证据已较完整，当前值得优先补 gate 或写作。
- `backlog/`：科学问题仍成立，但需要前置实验、资源或进一步新颖性核验。
  ⚠️ 按上方晋升规则，这**不表示不许推进**，只表示资源上排在后面。
- `archive/`：已死亡、被合并或被新 framing 取代的方向；保留必要 provenance，
  防止旧 claim 被误复活。
- `shared/`：多个提案共用的原始证据、代码和文献审计工具。

每个活跃/候选提案使用：

```text
PROPOSAL.md      最新主张、实验和 kill gate
STATUS.json      机器可读状态
SOURCES.md       证据源路径
```

## Related Work 强制门槛

从 2026-08-08 起，每个 active/backlog proposal 在启动新 GPU 实验前必须：

1. `PROPOSAL.md` 中有独立的 **Related Work / 新颖性边界**章节；
2. 列出最接近工作的具名 collision、重叠点、剩余空缺和“不得主张”；
3. `SOURCES.md` 中保存外部一手来源，而不只列内部结果；
4. benchmark、baseline、系统组件和评价协议都纳入 related-work 审计；
5. 若文献已覆盖核心主张，先收窄或归档 proposal，不能靠换应用包装。

当前逐提案缺口与补齐优先级见：

```text
shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md
```

其中 A03、A02、A04、B01 是最高优先级补洞项。

> ⚠️ **2026-08-11 更正**：**A03 已不再是补洞项**——它已 `ARCHIVE` decided
> （`active/A03-parametric-vs-external-memory/ARM_SET_DECISION.md`），
> 死于 effect-size-vs-apparatus-spread 而不是死于 related-work 覆盖，所以给它补
> Related Work 不会改变任何结论。当前补洞优先级为 **A02 → A04 → B01**。

> ⚠️ **2026-08-10 更正**：本行原写「B09 当前最完整」。那说的是**文献/设计完整度**，
> 但会被读成「最接近可跑」，而 B09 恰恰相反——它的候选池（~10K agent trajectories
> / ~100K SFT rows）**在两个盘上都不存在**，Phase 0 数据审计无法执行，状态已改为
> `backlog_blocked_data_does_not_exist`。见
> `backlog/B09-trajectory-aware-sft-data-selection/DATA_AUDIT_VERDICT_20260810.md`。

## 当前排序

### Active

1. `active/A01-null-calibration-methodology/` — **✅ 已晋升 → `paperG/`（2026-08-11）**
   - 跨 construct 的 input-blind null calibration。
   - ⚠️ **晋升 ≠ claim 被验证。** 它是一个**资源决定**（给这个方向一个 paper 目录 +
     算力优先级）。2026-08-10 外部审计的 **Major revision** 判定仍然有效，六条
     retraction/narrowing 全部随之带走，见 `paperG/README.md` 的「Scope discipline」。
   - **本目录不删**：它仍是该方向证据与决策历史的**唯一权威入口**，`paperG/` 里每个
     数字都必须能追回这里。
   - 命名：用 `paperG` 而非 `paperE` —— `paperE` 是**烧掉的名字**（task #171 把
     "Paper E: eval-interface construct validity" 判为 NO-GO 并把资产并入 Paper B
     #172），复用会撞 provenance；`paperC`/`paperD` 同理已归档。`paperG` 在全仓 0 引用。
   - MMLU interface failure、SQuAD majority prior、CKA layer-order null 和
     probe/native readout 是同一方法学框架的案例。
   - ⚠️ **2026-08-10：MAJOR REVISION**（外部审计
     `active/A03-parametric-vs-external-memory/evidence/TCODEX_AUDIT_20260810.md`
     §2.1+§7）。两条 claim 撤回（family-general step function；Llama-2 content
     strictly monotone）+ 一条降级（tie convention 翻 5/6 → 可执行 convention 翻
     0/6）。**读任何 2026-08-10 之前的 A01 verdict 文件前先读
     `active/A01-null-calibration-methodology/TCODEX_AUDIT_RESPONSE.md`。**
     `STATUS.json:status` 已不再声称 all gates passed。
2. `active/A02-comem-write-read-repair/`
   - 先验证已有 Write-LoRA/overlap repair 是否迁移到自然任务，再重做
     equal-latency frontier。
3. `active/A04-recovery-certification/`
   - 用干净、多 seed、同语料同 token 的实验研究 recovery certification，
     而非把现有混杂 depth ladder 当 scaling law。
   - ⚠️ **2026-08-11：A03 归档削弱了下一笔算力的理由**（**不是**否证 Pilot Zero ——
     Pilot Zero 是单 checkpoint 的 level 对比，结构上不继承杀死 A-2 的 sampler-seed
     方差，`DATAORDER_VERDICT.md`「Does NOT mean」#1 已明写）。真正的依赖是：
     A03 在**共用同一套装置**上把 CPT recovery increment 测成
     **+0.0818 pp，CI95 [−0.945, +1.108]**（= 31.10 pp 缺口的 **0.26%**，CI 含 0），
     而 certification rule 只能对**会动**的 recovery trajectory 做出有信息量的裁决。
   - ⚠️ `STAGE_B_DECISION.md` 问的那 **135 GPU-h 已经花掉了**（Path A 于 08-11 05:53
     启动；seeds 101/102/103 全部训完并在四轴评完，
     `evidence/stageB_S3_verdict.json` = `STAGE_A_DOES_NOT_FIRE`）。真正待批的是
     **Pilot Two（1077-4309 GPU-h，需用户显式批准）**。**批之前必须先在 prereg 里
     pre-data 写明「要裁决多大的 recovery，并证明它大于所选 S 对应的 MDE」**
     （S=3 → 1.10 pp @ σ̂=0.362；σ 的 χ² 上界处 3.16 pp）。
   - 另一条已被证伪的设计假设：**spread 不随 damage 单调**——keep12 在
     popqa/mmlu_content/nq_open 上的 σ 比 keep7 **更大**（mmlu 高 3.1×）。
     按「损伤越小方差越小」排 seed 预算是错的。
     完整表述见 `active/A03-.../STATUS.json:consequence_for_A04_135gpuh` 与
     `active/A03-.../ARM_SET_DECISION.md` §4。

### 已决定归档，但物理目录暂留 `active/`

- `active/A03-parametric-vs-external-memory/` — **`ARCHIVE` decided 2026-08-11**
  （执行了它自己的 `next_gate[0]`，**零 GPU**）。判定：`ARM_SET_DECISION.md`；
  死因复盘：`POSTMORTEM.md`。
  - **不是 kill clause 触发**，而是 **effect size vs 装置自身 spread**：
    pooled σ_run = **0.3620 pp（df=4，χ² 95% CI [0.217, 1.040]）** →
    两臂 S=3 的 MDE = **1.10 pp**（σ 上界处 **3.16 pp**）。A03 剩下的**训练配方类臂**
    目标效应全部 ≤1 pp，**任何可负担的 S 都测不出来**（S=8 要 1451 GPU-h，
    在 σ 上界处也只到 1.57 pp）。
  - 三条腿的结局：**参数腿（CPT）测出来是 0**（+0.0818 pp，CI 含 0，= 缺口的 0.26%；
    而 200k heal 的 **52.43 B token**、906.7 GPU-h 也没补上 31.10 pp 的缺口）；
    **RAG 腿两个盘都没有资产**（实测：PopQA 缓存 arrow schema 17 列**无 passage 字段**、
    TriviaQA 只缓存 `rc.nocontext`、`data/` 无 Wikipedia passage index、
    closed-book harness 按设计不吃 context）；**residual-memory 腿**要的 1B adapter
    从未存在（canonical 那个是 Qwen3-8B layers 12..35/36、hidden 4096），
    且该 thesis **A02 已在 8B 上测成负**（storage 2048×、read-compute 仅 1.03-1.37×、
    86-93% 的收益来自 retrieval）。
  - **A-1 存活但不属于 A03**：它是 level-vs-floor 的 null-calibration case study，
    即 **A01 的 thesis**（`GATE_FOURAXES_VERDICT.md` §2 自己承认与 A01 同定义、同数字），
    因此 **migrate 到 A01**，而不是把 A03 narrow 成 A-1。
  - ⚠️ **物理 `git mv` 被阻塞，现在不要搬**：`A04/code/pilot_one_stage_a_sd_run.py:81`
    用硬编码路径 import A03 的 canonical loaders（A04 全部 Stage-A/B 数字依赖其
    8/8-shard、item-count、dup-id、NaN 断言）；
    `A01/code/a01_audit_response_recompute.py:5,310` 引用 A03 的 audit。
    正确顺序：先把 loaders 提到 `shared/` 并 repoint，然后才搬目录。
    见 `STATUS.json:arm_set_decision.archive_move_blocked_by`。
  - 读该目录前先读 `claims/A03_SURVIVING_CLAIMS.md`（claim 的唯一权威；§B 有 9 条
    已死 claim，不得复活）。**seed 45 仍在跑，但两个可达分支都撤回 A-2**
    （`SEED45_PREDECLARATION.md` §3），它落地不改变本判定。

### Backlog

- `backlog/B01-semantic-bottleneck-memory-ready-models/`
- `backlog/B02-adaptive-depth-and-read-budget/`
- `backlog/B03-cyclic-layer-reset-boundary/`
- `backlog/B04-eval-fragility-incubator/`
  - `NARROWED_TO_OLMO_2_ONLY`。2026-08-10：Qwen cross-family「kill」降级为
    `NON_MATCHED_INCONCLUSIVE`（该 ladder 把 damage 与 training budget 混在一起）
    → 跨家族**未被检验**，而非被证伪。这**不是**晋升理由。见
    `DIRECTION_A_QWEN_LADDER_CONFOUND_ADDENDUM.md`。
- `backlog/B05-semantic-handoff-phase-diagram/`
- `backlog/B06-portable-decompression-adapter/`
- `backlog/B07-mutable-comem-serving/`
- `backlog/B08-memory-applications/`
- `backlog/B09-trajectory-aware-sft-data-selection/`
  - **`backlog_blocked_data_does_not_exist`（2026-08-10）**：候选池两盘皆无，
    先要做数据获取项目，Phase 0 审计才有对象。见 `DATA_AUDIT_VERDICT_20260810.md`。
  - 从 10K agent trajectories 展开的约 100K SFT rows 中选择 5K；
    以 trajectory/decision credit、target relevance 和集合覆盖替代 flat Top-K。
- `backlog/B10-dllm-infilling-ar-dominance/`
  - **`backlog_headline_not_significant_motivation_false`（2026-08-11）**。
    从 `archive/revival-slate/SLATE.md#3` 提升（live 方向不能挂在 archive 下）。
  - 数字**全部精确复现**（六个 arm 的 pass@1 与四个 cost 值，从
    `score.json:per_task[]` + `metrics.jsonl` 重算，10 位小数吻合），但
    **headline 与 motivation 都不成立**：
    (1) AR vs 最强 diffusion arm 只差 5/1033 题，McNemar `p=0.635`，CI 跨 0；
    (2) 该 split 自身 plus 轴 gold ceiling 只有 **0.8025**（204 题按构造不可能通过），
    只取 829 个 gold-feasible 题后**排序反转**（diffusion .9337 vs AR .9324）；
    (3) SLATE 的动机「DreamOn 没有 matched AR 对照」**是错的**——DreamOn
    camera-ready Table 1 就有 Qwen2.5-Coder-7B **92.6**；
    (4) claim (b)「length-elastic toggle 是 inert」**被推翻**：expansion 由 token id
    驱动、默认就开，实测 84.3% 的题输出长度被改过；且 README 从未 advertise 那两个 kwarg。
  - **Gate 1 花 0 GPU**（把已存在的 solutions 用 `--which base` 重打分），
    且大概率直接杀掉方向 → 先跑它再谈任何 GPU。详见 `NUMBER_AUDIT.md`。

### Archive

- `archive/paperC-v1-frozen-cap/`：SQuAD prune-and-graft capability 与
  forward-probe depth predictor 已死亡。
- `archive/paperD-cross-family-stitching/`：跨家族 layer stitching 方法已死亡；
  存活的 CKA/null 与 affine readout pilot 已迁到 A01/shared。
- `archive/revival-slate/`：旧 proposal slate，包含后来被修正的数字，仅供 provenance。
  - ⚠️ **其 `#3 dllm-infilling-ar-dominance` 已被 `backlog/B10-.../` 取代（2026-08-11）**。
    SLATE 的数字是对的，但 **claim 与路径都不能再引用**：headline 不显著、
    gold ceiling 会让排序反转、motivation（「DreamOn 无 AR 对照」）是错的、
    claim (b) 被推翻，且 SLATE 写的资产路径是 stale 的
    （真实路径 = `pighzliu_code/dllm_draft{,_104}/`，且 DreamOn 权重**在盘上**，
    SLATE 说「两盘都没有」是错的）。**一律读 B10，不要读 SLATE#3。**
- `archive/superseded/`：明确错误或已被更正的 proposal 文档。

## 使用规则

1. 新方向先写 `PROPOSAL.md` 和 kill gate，再启动 GPU。
2. 不得用旧 `status/*.md` 或 `versions/*.md` 中的历史 proposal 作为当前决策入口。
3. 原始结果尽量放在 `shared/evidence/`；大模型 checkpoint 和大数据仅在
   `SOURCES.md` 中引用，不复制。
4. dead proposal 的证据可以复用，但其旧 claim 不自动复活。
