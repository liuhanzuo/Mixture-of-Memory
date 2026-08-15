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
> （`archive/A03-parametric-vs-external-memory/ARM_SET_DECISION.md`），
> 死于 effect-size-vs-apparatus-spread 而不是死于 related-work 覆盖，所以给它补
> Related Work 不会改变任何结论。当前补洞优先级为 **A02 → A04 → B01**。

> ⚠️ **2026-08-12 更正（覆盖上面两条里关于 A02 的排序）**：**A02 已 CLOSED 并移入
> `backlog/A02-comem-write-read-repair/`**，`status:
> CLOSED_NO_THESIS_DIAGNOSTIC_ASSETS_RETAINED`，不再是补洞项，**不再消耗算力**。
> 理由：A0（完全不加 adapter）在所有 retrieval-closed cell 上都是最优臂，A02 产出的
> 每个数字都是 **tax**；storage form 死（h12 = 2048× 原文），read-compute 只有
> 1.03–1.37×。2026-08-12 的 gate 让它**更差而非更好**：在更难的 retrieval-closed cell
> (`niah_single_3`×16k) 上 j=6 已有 **−8.00 pp（显著）**，所以连「knob 便宜」也被收窄为
> 「只在两个饱和任务上成立」。**不 archive** 是因为它的证据仍在承重（depth-tax ladder、
> 精确 capacity-matched pair、**直接测出的 VT recall = 100%**、新的 de-saturation cell），
> 且 **B11 建立在它的 generation 之上**。
> 详见 `backlog/A02-comem-write-read-repair/A02_BABILONG_MISORDER_VERDICT.md`。
> **当前补洞优先级为 A04 → B01。**
>
> 同日新建 **`backlog/B11-generative-scorer-format-fragility/`**（A02 的方法论副产品：
> 生成式 scorer 的文本预处理可以把输出格式编码进分数，强到**破坏一个真实 +70pp 效应的排序**；
> 但**反向不显著**，机制是 metric 预处理 + floor，**不是** retrieval）。
> 它**不归 A02、也不归 B04**（B04 是 damage 下的 per-item `acc_norm` margin compression，
> likelihood ranking，不同 construct / 不同机制）。**novelty 未核查，通过前不得花任何 GPU。**

> ⚠️ **2026-08-10 更正**：本行原写「B09 当前最完整」。那说的是**文献/设计完整度**，
> 但会被读成「最接近可跑」，而 B09 恰恰相反——它的候选池（~10K agent trajectories
> / ~100K SFT rows）**在两个盘上都不存在**，Phase 0 数据审计无法执行，状态已改为
> `backlog_blocked_data_does_not_exist`。见
> `backlog/B09-trajectory-aware-sft-data-selection/DATA_AUDIT_VERDICT_20260810.md`。

## 当前排序

### Active

1. `active/A01-null-calibration-methodology/` — **✅ 已晋升 → `paperC/`（2026-08-11）**
   - 跨 construct 的 input-blind null calibration。
   - ⚠️ **晋升 ≠ claim 被验证。** 它是一个**资源决定**（给这个方向一个 paper 目录 +
     算力优先级）。2026-08-10 外部审计的 **Major revision** 判定仍然有效，六条
     retraction/narrowing 全部随之带走，见 `paperC/README.md` 的「Scope discipline」。
   - **本目录不删**：它仍是该方向证据与决策历史的**唯一权威入口**，`paperC/` 里每个
     数字都必须能追回这里。
   - 命名：用 `paperC` 而非 `paperE` —— `paperE` 是**烧掉的名字**（task #171 把
     "Paper E: eval-interface construct validity" 判为 NO-GO 并把资产并入 Paper B
     #172），复用会撞 provenance；`paperC`/`paperD` 同理已归档。`paperC` 在全仓 0 引用。
   - MMLU interface failure、SQuAD majority prior、CKA layer-order null 和
     probe/native readout 是同一方法学框架的案例。
   - ⚠️ **2026-08-10：MAJOR REVISION**（外部审计
     `archive/A03-parametric-vs-external-memory/evidence/TCODEX_AUDIT_20260810.md`
     §2.1+§7）。两条 claim 撤回（family-general step function；Llama-2 content
     strictly monotone）+ 一条降级（tie convention 翻 5/6 → 可执行 convention 翻
     0/6）。**读任何 2026-08-10 之前的 A01 verdict 文件前先读
     `active/A01-null-calibration-methodology/TCODEX_AUDIT_RESPONSE.md`。**
     `STATUS.json:status` 已不再声称 all gates passed。
2. ~~`active/A02-comem-write-read-repair/`~~ → **已 CLOSED，移入
   `backlog/A02-comem-write-read-repair/`（2026-08-12）**
   - 原计划「先验证已有 Write-LoRA/overlap repair 是否迁移到自然任务，再重做
     equal-latency frontier」**已作废**：phase-1 的自然任务信号被证明主要是
     top-12 `iter_bm25` 的 recall/pack-narrowing artifact，且 A0（不加 adapter）
     在所有 retrieval-closed cell 上是最优臂 —— 没有可迁移的正信号。
   - `status: CLOSED_NO_THESIS_DIAGNOSTIC_ASSETS_RETAINED`，**不再消耗算力**；
     复活需要**新机制**，而不是把同一个 ladder 再读一遍。
3. `active/A04-recovery-certification/`
   - 用干净、多 seed、同语料同 token 的实验研究 recovery certification，
     而非把现有混杂 depth ladder 当 scaling law。
   - ⚠️ **2026-08-11：A03 归档削弱了下一笔算力的理由**（**不是**否证 Pilot Zero ——
     Pilot Zero 是单 checkpoint 的 level 对比，结构上不继承杀死 A-2 的 sampler-seed
     方差，`DATAORDER_VERDICT.md`「Does NOT mean」#1 已明写）。真正的依赖是：
     A03 在**共用同一套装置**上把 CPT recovery increment 测成
     **−0.0293 pp，CI95 [−0.672, +0.613]**（4 个 sampler-seed draw；= 31.10 pp 缺口的
     **−0.09%**，CI 含 0；n=3 时读作 +0.0818 pp / +0.26%），
     而 certification rule 只能对**会动**的 recovery trajectory 做出有信息量的裁决。
     **2026-08-12 seed 45 落地后这条理由更强了**：点估计跨过零，从 +0.08 pp 变成 −0.03 pp。
   - ⚠️ `STAGE_B_DECISION.md` 问的那 **135 GPU-h 已经花掉了**（Path A 于 08-11 05:53
     启动；seeds 101/102/103 全部训完并在四轴评完，
     `evidence/stageB_S3_verdict.json` = `STAGE_A_DOES_NOT_FIRE`）。真正待批的是
     **Pilot Two（1077-4309 GPU-h，需用户显式批准）**。**批之前必须先在 prereg 里
     pre-data 写明「要裁决多大的 recovery，并证明它大于所选 S 对应的 MDE」**
     （S=3 → **1.11 pp @ σ̂=0.3666（df=5）**；σ 的 χ² 上界 0.899 处 **2.73 pp**；
     S=8 → 0.55 / 1.35 pp。~~df=4 时是 1.10 / 3.16~~，2026-08-12 已按 df=5 重述。
     悲观端门槛**降低**了 14%，但**该不该花这笔算力的理由反而更弱**——见上一条）。
     该门槛现已写进 `A04/STATUS.json:next_gate[4]` 本身（此前只存在于 prose 里，
     gate 自己不带 bar）。
   - ★ **Path B 不要买**：它要的「keep7 两个额外 seed 跑到 n=4/df=3」在 seed 45
     落地后**已经交付**，边际成本 **0 GPU-h**（keep7-20k 现为 S=4、df=3、
     s=0.4039 pp、χ² [0.229, 1.506]，另三轴也首次有了 df=3 的真 σ）。
     ⚠️ 但这**不能**关掉 K2 的悲观端：K2 的 prereg estimator 用的是 **keep12** 组
     自己 df=2 的 σ，seed 45 是 keep7 draw，**K2 算术分毫未动**——popqa 在 keep12
     σ 的 χ² 上界处仍会触发（3.526 vs Δ=1.321）。要关掉它只能加 **keep12** 的 seed。
   - 另一条已被证伪的设计假设：**spread 不随 damage 单调**——keep12 在
     popqa/mmlu_content/nq_open 上的 σ 比 keep7 **更大**（2026-08-12 更新：keep7 四轴
     现均为 df=3 的真 σ，所以这条不再是「σ 对比 df=1 range」；倍数为 popqa 1.7×、
     nq_open 2.8×、mmlu_content **1.4×**——旧文写的 3.1× 是 df=1 range 造成的高估）。
     按「损伤越小方差越小」排 seed 预算是错的。
     完整表述见 `archive/A03-.../STATUS.json:consequence_for_A04_135gpuh` 与
     `archive/A03-.../ARM_SET_DECISION.md` §4、`archive/A03-.../SEED45_VERDICT.md` §4。

### 已归档（物理目录 2026-08-11 已移入 `archive/`）

- `archive/A05-structural-dllm-cost-frontier/` — **`ARCHIVE` decided 2026-08-12**
  （**K1 实验 gate 触发**，~21 GPU-h；closeout 零 GPU）。
  判定：`A05_K1_CANVAS_SWEEP_VERDICT.md` + `A05_CLOSEOUT_VERDICT.md`；死因复盘：`POSTMORTEM.md`。
  - **是 kill clause 触发**（与 A03 不同）：预注册的 K1 说「DreamOn 在最佳 non-oracle canvas 上
    若逼近 scaffold Medium 5.0 pp 以内则方向死」。只把 `initial_masks` 从 8 改成 32
    （其余 sampler 旋钮全冻结），DreamOn 的 **MBPP+ .085 → .3545**、**HE+ .122 → .2134**
    （再修一个 HE+ stitch bug 后 **.4817**）→ 两个 benchmark 都**追平/反超** scaffold
    （.177/.354）。**A05 赖以立项的 +26.9 pp margin 是 canvas 预算 artifact。**
  - **K1 之后唯一存活的 cost claim（"matched quality 下 scaffold 便宜 6.2×/8.2×"）也已判死**：
    5 条预注册 falsification 条件中 **4 条触发** —— 均值比在**中位数上方向翻转**
    （DreamOn 中位 item 反而更便宜，0.56-0.96×）；12-13% 打到**未对齐的迭代上限**的 item
    扛了 57-61% 的 NFE；HE+ 上质量根本没 matched（scaffold 差 30.5 pp）；
    **AR 对照（Qwen2.5-Coder-7B）在 `tokens_fed` 上比 scaffold 便宜 ~70× 且质量高 +.29/+.35**
    = 严格 Pareto 支配 —— 正是当年 Pareto claim 被 RETRACTED 的那个「家族内部点」描述。
    **故未晋升为独立 proposal，且不得再花 GPU（K2/K3 明确不跑）。**
  - **存活并外溢的**：三个 harness 缺陷已在源仓修好（wzc1 `58bbb20`、zwfy6 `9651406`、
    `_104` `d214d37c`）—— 归档 `nfe` 其实是 `len(output.history)`（真值 172.3/153.4 而非
    265.88/135.65）；`mask_expansion`/`delete_eos_token` 一直是 inert（已由执行证实）；
    HE+ stitch 双重缩进**低估**了本仓所有 HE+ 数字（c128 处 `.1707 → .4817`）。
    另外顺手撤掉 roadmap 里那条**假的** "`generate_infilling.py` is missing" blocker（它在另一个盘）。
  - **可发表的那条不属于 A05**：「一个被广泛引用的 diffusion baseline 在 full-program 上的弱
    是 canvas 预算 artifact，一个配置整数就把 MBPP+ 从 .085 抬到 .3545」是**评测实践**结论，
    建议归 **A01**（同类方法论、换 surface），已记在
    `A05/STATUS.json:finding_that_outlives_a05`，**故意不自动晋升**。

- `archive/A03-parametric-vs-external-memory/` — **`ARCHIVE` decided 2026-08-11**
  （执行了它自己的 `next_gate[0]`，**零 GPU**）。判定：`ARM_SET_DECISION.md`；
  死因复盘：`POSTMORTEM.md`。
  - **不是 kill clause 触发**，而是 **effect size vs 装置自身 spread**：
    pooled σ_run = **0.3666 pp（df=5，χ² 95% CI [0.229, 0.899]）** →
    两臂 S=3 的 MDE = **1.11 pp**（σ 上界处 **2.73 pp**）。A03 剩下的**训练配方类臂**
    目标效应全部 ≤1 pp，**任何可负担的 S 都测不出来**（S=8 要 1451 GPU-h，
    在 σ 上界处也只到 1.35 pp）。
    *（~~df=4 时：0.3620 pp、χ² [0.217, 1.040]、MDE 1.10 / 3.16 / S=8 上界 1.57~~ ——
    2026-08-12 seed 45 落地后已重算为 df=5；旧值保留划线，不静默删除。
    点估计只动了 +1.3%，χ² 上界收紧 22%，**判定不翻转、反而两头都更硬**：
    见 `SEED45_VERDICT.md` §4.1。）*
  - 三条腿的结局：**参数腿（CPT）测出来是 0**（**−0.0293 pp**，CI95 [−0.672, +0.613]，
    = 缺口的 **−0.09%**，4 个 sampler-seed draw；~~n=3 时 +0.0818 pp / 0.26%~~；
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
  - ✅ **物理 `git mv` 已于 2026-08-11 完成（commit `3949f14` 重构 + 本次搬移）。
    曾被阻塞的原因保留在此，不得静默删除**：`A04/code/pilot_one_stage_a_sd_run.py`
    当时用**硬编码路径 + 读源码文本 + `exec`** 的方式取 A03 的 canonical loaders
    （A04 全部 Stage-A/B 数字依赖其 8/8-shard、item-count、dup-id、NaN 断言），
    另有 **5 个** A04 脚本用硬编码相对路径 `sys.path.insert` 后
    `from analyze_1b_knowledge_floor import ...`；
    `A01/code/a01_audit_response_recompute.py:5,310` 引用 A03 的 audit（仅文档路径）。
    解除办法（已执行，顺序不可颠倒）：
    1. loaders **逐字节**提到 `shared/code/canonical_eval_loaders.py`（断言、
       `n_boot=5000/seed=42/CI95` 协议一字未改），A03 自己改为 import 而非留副本；
    2. A03 目录位置收敛到唯一 resolver `shared/code/proposal_paths.py`
       （先找 `archive/`，回退 `active/`，找不到就**大声报错**，不返回猜测路径）；
    3. **搬移前**在 zwfy6 用重构后代码重跑并逐字段比对：Stage-A(43/44)、
       Stage-B(101/102)、`stageB_pair_101_103` 各 **68/68 字段完全一致**，
       `stageB_S3_verdict` 的 means/`sd_run_pp`/`bound_S3_pp` **逐位一致**，
       A03 trajectory recompute **664 个重叠字段 0 差异**；只有指向 loader 模块的
       provenance 字符串变了。
    ⚠️ **zwfy6 的 `proposal/` 是手抄目录、不是 git checkout**
    （`git ls-files proposal/` 返回 0），所以 `git mv` **不会**传播到那边；
    A04 脚本里那个 positive `_FIX_MARKER` stale-copy 断言因此**必须保留**。
  - 读该目录前先读 `claims/A03_SURVIVING_CLAIMS.md`（claim 的唯一权威；§B 有 9 条
    已死 claim，不得复活）。
  - ✅ **seed 45 已收尾（2026-08-12 00:10，task #243）—— 判定不变，A-2 仍撤回。**
    它是本 prereg 下的**最后一个 run**（**没有 seed 46**）。结果：
    primary 轴 triviaqa em **θ = −0.3622 pp，CI95 [−0.5517, −0.1838]，SIG 负**
    → **NOT-CONFIRM**（同时不满足 θ>0 与 [+0.20,+0.80] band）→ 聚合
    **0/3 CONFIRM → ARTIFACT**（维持现状）。两个可达分支**都**撤回 A-2，且这一点在
    run 之前就写死在 `SEED45_PREDECLARATION.md` §3 里，数字落地后不得重新论证。
    - ckpt 完好：`step220000.pt` 与 step205/210/215000 **字节数完全相同**
      （12,181,311,650 B），v1 停止竞态（曾把 Arm 4 截到 49%）**这次没触发**；
      trainer `rc=1` 是 `kill -TERM` 的预期返回码，不是崩溃。
    - shard 完整性：四轴 × (arm + baseline) 共 8 个 cell 全部 **8/8 shard、
      n_scored == expected**（popqa 14267 / triviaqa 17944 / nq_open 3610 / mmlu 14042）、
      **0 重复 item_id、0 nan**。
    - σ_run 交付：keep7-20k → **S=4、df=3、s=0.4039 pp、χ² [0.229, 1.506]**
      （另三轴首次有真 σ：popqa 0.1959 / nq_open 0.0750 / mmlu_content 0.0555 pp，
      取代此前不可引用的 df=1 pairwise range 0.2726 / 0.0000 / 0.0252）；
      pooled → **df=5、0.3666 pp、χ² [0.229, 0.899]**。
    - 判定文档 `archive/A03-.../SEED45_VERDICT.md`；证据
      `evidence/a03_cpt_trajectory_paired_full_with_seed45.json`、
      `evidence/a03_sigma_run_n3.json`、`evidence/a03_seed45_integrity.json`。
    - ⚠️ 历史坑（保留）：seed 45 的 eval **不会自动触发**——
      `_run_a03_dataorder_repl.sh` 只有 trainer-stop watcher，没有 eval watcher
      （2026-08-11 21:30 实测 `pgrep -af watcher` 为空）。这次是手动投的。

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
  - ⛔ **`lifecycle: dead`（2026-08-15）——Gate 1 已跑，`FIRED = KILL`。**
    base 轴重打分（0 GPU，solutions 原样读盘）：`qwen_fim` vs `dreamon_oracle`
    exact McNemar **p=1.0000**（b=39, c=38, discordant=77, n=1033）、
    **Δ=+0.00096805**（= 1033 题里差 **1 题**）、95% CI [−0.0164, +0.0183]。
    两个 kill 条件同时成立，阈值 α=0.05 / |Δ|<0.02 **逐字未改**。三个轴现在一致
    （plus +0.0048 p=0.635；plus-feasible −0.0012 p=1.000；base +0.00097 p=1.0000），
    且**方向都不稳定**（wzc1 1022 题 feasible 子集上符号翻转 −0.00098）。
    → **Gate 2（2-4 GPU-h）与 Gate 3（1-2 GPU-h）不再被授权**，B10 无任何耗卡下一步。
    MAIN 已独立复算 McNemar 与 evalplus 源码差异（`GATE1_BASE_AXIS_VERDICT.md` §9）。
  - 残值按 `gate_1.if_killed` 的**第一个分支**封存为 **`PROTOCOL_NOTE.md`**（非 archive）：
    4 条存活结论（最强的是 suffix visibility +0.2314 AR / +0.2991 diffusion，both
    **p<1e-56**，且解释是「bidirectional context 是 FIM task **framing** 的 affordance，
    AR 也能用，**不是 model class 的属性**」）+ 本 gate 新产出的**跨主机 gold ceiling
    不可复现**（base 0.9894 wzc1 vs 1.0000 zwfy6 = PyPI evalplus 0.3.1 的 `find_zero`
    分支裸 `continue` 漏了 `details[i]=True`；plus 0.8025→0.8122 = sandbox 4 GiB
    `RLIMIT_AS` 把 `MemoryError` 记成 wrong answer）+ 一条可复用教训：
    **「vendored 文件两盘 byte-identical」不能证明解释器加载了它**。
    ⛔ 禁止从同 6 臂再捞 ranking（nested-ladder，已 retract 两次 = Retractions 6/7）；
    pairwise matrix 只是 provenance 不是菜单；gate_4 常驻规则仍生效
    （无 decontaminated companion 不得把 absolute pass@1 当 capability）。
  - 目录**留在 `backlog/` 不移 archive**：`PROTOCOL_NOTE.md` 与 `evidence/gate1_base/`
    的路径被 SOURCES/NUMBER_AUDIT/VERDICT 三处引用，`lifecycle: dead` 已承载关闭事实。

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
