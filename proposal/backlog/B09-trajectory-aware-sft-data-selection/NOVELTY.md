# Novelty Audit — B09 Trajectory-Aware SFT Data Selection

核验日期：2026-08-08。

## 结论

**存在强相关工作，但尚未发现单篇工作完整覆盖 B09 的问题设定。**

不能把 novelty 放在以下任何单独组件上：

- agent trajectory selection；
- multi-turn dialogue-level selection；
- critical-step supervision；
- recovery/efficiency-aware curation；
- target-aware instruction selection；
- facility-location/submodular coreset；
- benchmark/task budget allocation；
- token-efficient agent tuning。

B09 当前最安全的定位是：

> 现有工作通常选择完整 agent/dialogue trajectories，或在给定 trajectory
> 内选择关键步骤；B09 研究一条 parent agent trajectory 被展开成多个相关
> SFT child rows 后的**两层联合选择问题**，并显式控制 parent multiplicity、
> decision credit、target relevance、candidate/skill coverage 和 assistant-target
> token cost。

这是“问题形式化 + 严格受控实证”的 novelty，而不是任一基础 score、
facility-location 或 greedy algorithm 的首创。

---

## 1. 最接近的 agent / multi-turn 工作

### 1.1 TopoCurate

**TopoCurate: Modeling Interaction Topology for Tool-Use Agent Training**  
arXiv:2603.01714，2026。

已做：

- 将同一任务的 multi-trial rollouts 投影为统一的 semantic quotient topology；
- 显式建模 action-observation state、成功/失败分支和 interaction dynamics；
- 为 SFT 选择完整 trajectories；
- 使用 Reflective Recovery、Semantic Efficiency 和 Strategic Diversity；
- 同时为 RL 选择具有高 error-branch ratio 和 strategic heterogeneity 的任务。

未覆盖：

- 一条 parent trajectory 派生多个 SFT child rows 后的 parent cap；
- 选择完整 trajectory 后，再选择其中的 decision rows；
- 独立 target-query relevance；
- target/pool facility coverage；
- assistant-target-token knapsack；
- benchmark × trajectory × decision-type 联合约束。

**威胁等级：Critical。**

不能主张：

- 首次 agent trajectory-aware data selection；
- 首次 recovery/efficiency/diversity-aware tool-use trajectory filtering；
- 首次超越 outcome-only trajectory selection。

### 1.2 MDS

**Data Selection for Multi-turn Dialogue Instruction Tuning**  
Findings of ACL 2026，arXiv:2604.07892。

已做：

- 明确以完整 multi-turn dialogue 为单位，而非 isolated turns；
- global stage 在 user-query trajectory space 做 semantic binning、
  coverage 和 redundancy control；
- local stage 使用 entity grounding、information progress 和
  query-answer form consistency；
- 在固定 dialogue budget 下构建 compact multi-turn dataset。

未覆盖：

- agent action、observation、state transition 和 recovery branches；
- dialogue 内 critical assistant-turn selection；
- parent trajectory expansion 后的 child-row cap；
- 独立 target-test query relevance；
- assistant-token knapsack。

**威胁等级：Critical。**

不能主张：

- 首次发现 multi-turn data 不应按独立 turns 评分；
- 首次做 global coverage + local structural-quality selection；
- 首次以完整长对话为 selection unit。

### 1.3 ATLAS

**ATLAS: Agent Tuning via Learning Critical Steps**  
Findings of ACL 2025，arXiv:2503.02197。

已做：

- 在 expert agent trajectories 内选择 critical steps；
- 类别包括 plan creation、critical observation、critical action 和
  self-correction；
- 保留完整 causal prefix，只在选中 steps 上计算 loss；
- 以约 30% steps 训练，并与 full steps、random steps、PPL/value selection
  比较。

未覆盖：

- 选择哪些 trajectories 进入训练；
- parent-row cap；
- 跨 trajectory coverage/diversity；
- target-query relevance；
- benchmark quota 和 exact token budget。

**威胁等级：Critical。**

不能主张：

- 首次选择 agent trajectory 中的关键步骤；
- 首次证明 critical-step training 优于 full trajectory imitation；
- 首次提出 ATLAS 的四类 critical steps。

ATLAS 必须作为直接 baseline，而不是仅作为设计动机。

### 1.4 Verified Critical Step Optimization

**Verified Critical Step Optimization for LLM Agents**  
arXiv:2602.03412，2026。

已做：

1. 从失败 policy trajectory 出发；
2. 使用 PRM 找 candidate critical steps；
3. expert 提出替代 action；
4. 从替代 action 继续 rollout；
5. 仅保留能够把 failure 翻转为 success 的 verified decisions；
6. 在这些步骤上构造 preference pairs 并做 targeted DPO。

未覆盖：

- training-free SFT subset selection；
- fixed-5K child-row selection；
- target/pool coverage；
- parent cap 和 token-knapsack coreset。

**威胁等级：High。**

B09 中普通 LLM criticality 或 NLL 信号只能称：

- critical-step candidacy；
- heuristic decision importance；
- proxy decision credit。

若要主张 verified/causal credit，必须加入 sibling branch 或 outcome-flip
验证。

### 1.5 SWE-TRACE

**SWE-TRACE: Optimizing Long-Horizon SWE Agents through Rubric Process
Reward Models and Heuristic Test-Time Scaling**  
arXiv:2604.14820，2026。

已做：

- step-wise oracle verification；
- 构造偏向 token-efficient、shortest-path trajectories 的 SFT corpus；
- process reward model 和 critical-step memory；
- 面向 long-horizon software agents 优化 trajectory quality 和 token cost。

未覆盖：

- parent-expanded child-row selection；
- trajectory-first + decision-row coreset；
- target-query facility coverage；
- benchmark/skill submodular coverage；
- 显式 fixed-5K assistant-token knapsack。

**威胁等级：High。**

不能主张首次 token-efficient agent trajectory curation 或首次结合
step verification 与 agent SFT 精简。

### 1.6 Code-agent trajectory curation

**A Systematic Evaluation of Trajectory Data Curation for LoRA Fine-Tuning
of Code Agents**  
arXiv:2607.17205，2026。

已做：

- 在大规模 SWE trajectories 上做 trajectory-level filtering；
- 使用 Efficiency 和 Style 质量维度；
- 系统比较 selection strategy、数据规模和质量-数量 trade-off；
- 报告 error-retry rate 等子指标。

未覆盖：

- trajectory 内 decision-step selection；
- parent-child dependency；
- target relevance 和 set-level coverage；
- 多 benchmark agent success 的统一约束实验。

**威胁等级：Medium。**

它否定“agent trajectory filtering 尚未被研究”的表述。

---

## 2. 最接近的通用 instruction-selection 工作

### 2.1 RDS+ / large-scale selection

**Large-Scale Data Selection for Instruction Tuning**  
arXiv:2503.01807，2025。

已做：

- 使用预训练 LM hidden states 的 position-weighted pooling；
- 计算 candidate 与 target queries 的 representation similarity；
- 以 query/task round-robin 进行 target-aware selection；
- 显示多种复杂 selector 在大池中可能低于 random，而 RDS+ 相对稳定。

B09 不能主张 target-query embedding relevance 或 query coverage 是新概念。
必须证明 parent grouping 和 decision semantics 在 RDS+ 之上有独立收益。

**威胁等级：High。**

### 2.2 LESS

**LESS: Selecting Influential Data for Targeted Instruction Tuning**  
ICML 2024，arXiv:2402.04333。

已做：

- 短 LoRA warmup；
- optimizer-aware projected gradient datastore；
- 使用 few-shot target gradients 做 targeted selection。

B09 的 LESS 分支只是 low-training comparator/extension，不是 novelty。

**威胁等级：High。**

### 2.3 DELIFT

**DELIFT: Data Efficient Language Model Instruction Fine-Tuning**  
ICLR 2025，arXiv:2411.04425。

已做：

- 以 ICL-based pairwise utility 衡量样本的信息价值；
- 使用 facility location、facility-location mutual information 等
  submodular objectives；
- 对 instruction tuning、task-specific adaptation 和 continual tuning
  采用不同 selection objective；
- task-specific setting 已联合 target relevance 与 diversity/coverage。

B09 不能主张 target-aware submodular/facility-location selection 首创。
剩余区别必须来自 parent-expanded agent rows、decision events 和
group/token constraints。

**威胁等级：High。**

### 2.4 SMART

**SMART: Submodular Data Mixture Strategy for Instruction Tuning**  
Findings of ACL 2024，arXiv:2403.08370。

已做：

- 用 submodular objective 选择/加权 tasks；
- 在总预算下分配 task budgets；
- 在 task 内选择 representative/non-redundant instances；
- 使用 facility location 和 log determinant。

B09 不能把 benchmark quota、task budget allocation 或 submodular
within-task selection 作为单独 novelty。

**威胁等级：High。**

### 2.5 DEITA / MIG / IFD / Superfiltering

- **DEITA**, ICLR 2024，arXiv:2312.15685：
  complexity × quality，再做 representation diversity filtering。
- **MIG**, arXiv:2504.13835：
  quality-weighted label graph、信息增益和递减边际 coverage。
- **From Quantity to Quality / IFD**, NAACL 2024，arXiv:2308.12032：
  model-aware instruction-following difficulty。
- **Superfiltering**, ACL 2024，arXiv:2402.00530：
  以弱 proxy 模型低成本近似 difficulty ranking。

这些工作覆盖 quality、difficulty、标签/语义 coverage 和 weak-to-strong
filtering。B09 的 novelty 不能是“quality + diversity”或“小模型打分”。

### 2.6 Agent-FLAN 与 token/boilerplate selection

- **Agent-FLAN: Designing Data and Methods of Effective Agent Tuning for
  Large Language Models**, Findings of ACL 2024，arXiv:2403.12881：
  区分 reasoning、tool retrieval、parameter understanding、
  instruction/format following，并重配能力数据比例。
- **Disentangling Reasoning Tokens and Boilerplate Tokens for Language
  Model Fine-tuning**, Findings of ACL 2025，arXiv:2412.14780：
  使用 Shuffle-Aware Discriminator 区分 reasoning 与 boilerplate tokens，
  对 reasoning tokens 提高训练权重。

因此 skill/capability balancing、no-tool data 和“format-only 信息价值较低”
都不能作为 B09 的单独首创点。

---

## 3. 组件覆盖矩阵

符号：`✓` 明确处理；`△` 部分相似；`—` 未处理。

| 工作 | Parent/task 分组 | 整轨迹选择 | 轨迹内 step | Target-aware | Set coverage / submodular | 显式 assistant-token knapsack |
|---|---:|---:|---:|---:|---:|---:|
| **B09 计划** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TopoCurate | ✓ | ✓ | — | — | △ strategic diversity | — |
| MDS | △ dialogue unit | ✓ | — | — | ✓ semantic coverage | — |
| ATLAS | △ | — | ✓ | — | — | △ step fraction |
| CSO | △ | — | ✓ verified | — | — | △ sparse supervision |
| SWE-TRACE | △ | ✓ | △ verified synthesis | — | — | △ token-efficient |
| Code-agent curation | △ | ✓ | — | — | — | — |
| RDS+ | — | — | — | ✓ | △ round-robin | — |
| LESS | — | — | — | ✓ gradient | — | — |
| DELIFT | — | — | — | ✓ | ✓ FL/FLMI | — |
| SMART | △ task groups | △ | — | — | ✓ | △ count budget |
| Agent-FLAN | △ skill groups | — | — | — | △ mixture balance | — |

检索结论：

- whole-trajectory/dialogue selection 已被 TopoCurate、MDS 等覆盖；
- within-trajectory critical-step supervision 已被 ATLAS、CSO 覆盖；
- target-aware selection 已被 RDS+、LESS、DELIFT 覆盖；
- submodular budget/coverage 已被 DELIFT、SMART、MIG 等覆盖；
- **本次未发现单篇工作在 parent-expanded agent-row setting 中统一处理这几层。**

这不是对全部文献的数学“不存在证明”，论文中必须写成：

> To the best of our knowledge, among the audited literature...

---

## 4. 可安全主张

### 4.1 问题形式化

最安全：

> Existing work either selects whole agent/dialogue trajectories or identifies
> critical steps within given trajectories. We study the dependent-child
> setting where each parent agent trajectory yields multiple candidate SFT
> rows, and jointly select across parents and within-trajectory decisions under
> target-token and coverage constraints.

可强调：

- row-expanded trajectory dependency；
- parent trajectory partition constraint
  \[
  |S\cap U_g|\le m_g;
  \]
- assistant-target-token knapsack；
- trajectory 与 decision 两层 marginal gain；
- target-query、candidate-pool 和 skill coverage 的联合目标；
- constraint-matched random，用于剥离 cap、配额和复杂 score 的收益。

### 4.2 受控经验贡献

如果实验支持，可主张：

1. flat row selectors 会不会被 sibling multiplicity 欺骗；
2. parent cap 是否比 quality score 更重要；
3. critical decision signal 在 constraint-matched random 之上的边际价值；
4. target relevance 是否主要帮助 ID，而 broad coverage 帮助 OOD；
5. fixed-row 与 fixed-token 结论是否不同；
6. selector 是否跨 student family 转移；
7. whole-trajectory、step-only 和 joint hierarchical selection 的系统比较。

---

## 5. 不能主张

禁止使用：

- “首次 trajectory-aware agent SFT data selection”；
- “首次进行 tool-use trajectory filtering”；
- “首次选择 agent trajectories 中的关键步骤”；
- “首次证明 critical steps 优于完整 trajectory”；
- “首次 recovery-aware / efficiency-aware trajectory selection”；
- “首次使用 target queries 选择 instruction data”；
- “首次 target-aware facility-location/submodular selection”；
- “首次用 submodular optimization 分配 instruction-tuning budget”；
- “首次平衡 reasoning/tool/argument/format 数据”；
- “首次降低 boilerplate token 的训练权重”；
- “首次 token-efficient agent trajectory curation”；
- “首次 hierarchical instruction data selection”；
- “LLM criticality 等价于 causal credit”；
- “找到全局最优 5K”。

---

## 6. 如何避免退化为简单组合

如果最终方法只是：

```text
ATLAS criticality
+ RDS+ similarity
+ MIG/DELIFT facility-location
+ parent cap
```

审稿人很可能将其视为工程组合。

要形成更强方法 novelty，至少加入一个实质核心：

1. **联合两层目标，而非串行筛选**
   - 选择一个 parent 会改变其 child decisions 的后续边际收益；
   - 同一 parent 的 siblings 具有显式 conditional redundancy。
2. **Branch-verified decision credit**
   - 对同 task 的成功/失败 sibling trajectories 做 alignment；
   - 用 divergence point 与 outcome difference 估计 decision advantage。
3. **Group-aware target coverage**
   - target query gain 先分配到 parent trajectory，再分配到 decision events；
   - 避免多个 sibling rows 重复覆盖同一 target mode。
4. **针对 laminar/partition + token knapsack 的算法或分析**
   - 不只是普通 scalar score 后 greedy。
5. **专门的 row-multiplicity stress test**
   - 人工改变每条 trajectory 可派生 rows 的数量；
   - 测量 flat selector 是否因 multiplicity 而改变 subset 和下游结果。

其中第 2 和第 5 项最容易把 B09 从组合 recipe 变成有清晰机制问题的工作。

---

## 7. 推荐标题与 framing

比宽泛的 `Trajectory-Aware Data Selection for Agent SFT` 更安全：

- **Group-Constrained Coreset Selection for Row-Expanded Agent Trajectories**
- **From Trajectories to Decisions: Hierarchical Data Selection for Agent SFT**
- **When One Trajectory Becomes Many Rows: Dependency-Aware Data Selection
  for Agent Fine-Tuning**

推荐核心叙事：

> 普通 instruction selector 假设候选样本近似独立，但 agent trajectory
> expansion 产生高度相关的 child rows。现有 agent 方法分别停留在完整
> trajectory selection 或 trajectory 内 critical-step learning。B09 研究两层
> selection 之间的 dependency，并在严格匹配 parent constraints 和 training
> tokens 后，测量 quality、credit、relevance 与 coverage 的真实边际价值。

---

## 8. Novelty gate

在升级 active 或开始写论文前，必须完成：

1. TopoCurate、MDS、ATLAS、CSO 的正式 baseline/差异表；
2. RDS+、DELIFT、SMART、LESS 的实现和成本比较；
3. 检索是否已有 parent-child/group-constrained agent-row selector；
4. 实现 row-multiplicity stress test；
5. 至少实现一个不是简单 score addition 的核心：
   - joint hierarchical marginal gain；或
   - branch-verified decision credit。

若第 5 项没有实现，proposal 应定位为：

> controlled empirical study + robust selection recipe

而不是新的 selection algorithm。
