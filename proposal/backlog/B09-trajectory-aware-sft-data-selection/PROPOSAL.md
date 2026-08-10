# B09 — Trajectory-Aware Data Selection for Agent SFT

## 状态

> ## ⚠️ 2026-08-10 — BLOCKED：本提案的候选池在盘上不存在
>
> **`BACKLOG / READY FOR DATA AUDIT` → `BACKLOG / BLOCKED — DATA DOES NOT EXIST`。**
>
> 下面 §1（第 47-58 行）把 `|G| ≈ 10000` 条 agent trajectories 与
> `|U| ≈ 100000` 条派生 SFT rows 当作**已有资产**（第 168 行明确写
> `Candidate U | 100K 候选池 | 可见`）。**两个物理盘都搜过，这个池子不存在。**
> 因此 `next_gate[0]`（Phase 0 数据审计）**无法执行**——没有东西可审。
>
> 本仓库里 "trajectory" 一律指 **checkpoint-over-training-steps**，从来不是
> agent rollout；所有 `*trajector*` 命中都是 Paper B 图、A03 CPT 轨迹、或本仓库
> 自己的 agent 编排代码。
>
> 最接近的资产是 `data/olmo2_sft/tulu3_general_*`（两盘都有；zwfy6 侧 234,483
> conversations → 122,070 条 2048-packed sequences；wzc1 侧 107,740×2048）。
> **它不能替代**：无 parent trajectory（`source` 是数据集名，9 个源 × ~26K，
> 不是 ~10K 组 × ~10）、无 step/decision type、无 success/reward/tool family、
> 无 branch 结构、且已按 Paper B 的 `deny_sources` 去污过滤打包。用它等于把
> 自变量删掉。
>
> **真正的第一步是数据获取，不是审计**，且不是 Phase 0 声称的「无 GPU」小事：
> 要么外网下载现成 agent-trajectory 语料（需代理 + licence 核查，仍零 GPU，
> 最便宜），要么自己 rollout ~10K 条（大额 GPU + 本仓库没有的 agent task
> harness）。
>
> 完整证据、搜索范围与审计局限见 **`DATA_AUDIT_VERDICT_20260810.md`**；
> 机器可读状态见 `STATUS.json` 的 `blocker_2026_08_10`。下面原文保留不改。

**BACKLOG / READY FOR DATA AUDIT。**（← 已被上面 banner 取代）

科学问题、方法和 falsification protocol 已明确，但本项目尚无下游 SFT 结果。
在完成候选池审计和无训练 selector pilot 之前，不进入 active。

## 一句话主张

当约 10K 条 agent trajectories 被展开为约 100K 条 SFT rows 时，最佳 5K
不应由独立 row score 的 Top-K 得到，而应通过：

> **trajectory-aware validity filtering + critical decision-point selection +
> target/pool coverage + benchmark/skill coverage**

构造一个受 parent-trajectory、benchmark 和 token-budget 约束的 coreset。

核心待证假设是：**层次化分组与集合覆盖，比单点 quality/difficulty 排名更重要；**
在此基础上，target relevance 才可能进一步改善 in-domain performance。

## 新颖性边界

新颖性审计见 `NOVELTY.md`。当前结论：

- TopoCurate 已做 tool-use agent 的 whole-trajectory selection；
- MDS 已做 multi-turn whole-dialogue coverage + local structural quality；
- ATLAS/CSO 已做 within-trajectory critical-step supervision/verification；
- RDS+/LESS 已做 target-aware selection；
- DELIFT/SMART/MIG 已做 submodular coverage、task budget 或 facility-location。

因此不能主张任一单独组件的首创。最安全的空缺是：

> **parent trajectory 被展开为多个相关 SFT child rows 时，联合选择 parent
> 和 decision rows，并同时施加 target-token、coverage 与 group constraints。**

若最终方法只是现有 score 的线性组合，本方向只能定位为 controlled empirical
study / robust recipe。要主张新的 selection method，至少需要 joint hierarchical
marginal gain、branch-verified decision credit 或 row-multiplicity stress-test
中的实质技术核心。

---

## 1. 问题设定

设原始 trajectory 集合为：

\[
\mathcal G=\{g_1,\ldots,g_{10000}\},
\]

每条 trajectory \(g\) 产生若干 SFT rows \(U_g\)，总候选池：

\[
U=\bigcup_{g\in\mathcal G}U_g,\qquad |U|\approx100000.
\]

目标是选择：

\[
S\subset U,\qquad |S|=5000,
\]

用于 SFT 一个固定 student model，并在约 1K 条独立测试任务上评价。

必须同时记录：

- parent trajectory \(g(i)\)；
- task family；
- benchmark \(b(i)\)；
- trajectory step/decision type \(t(i)\)；
- assistant target-token cost \(c_i\)；
- success/reward、tool family 和 trajectory position。

## 2. 研究边界

### 2.1 三种 selector 成本等级

1. **TF-0：严格 zero-update**
   - 规则、metadata、冻结 embedding；
   - 不对当前候选池训练 scorer。
2. **TF-F：forward-only**
   - 冻结 proxy/student 的 NLL、IFD/IRA、hidden state 或冻结 judge；
   - 无 parameter update，但必须报告推理 FLOPs/GPU-hours。
3. **LT：low-training**
   - 少量 LoRA warmup 和低维 gradient feature；
   - 以 LESS 类方法作为性能上界 comparator。

“Training-free”不等于“cost-free”。统一报告：

\[
C_{\mathrm{total}}
=C_{\mathrm{selection}}+C_{\mathrm{SFT}}.
\]

### 2.2 不主张

- 不声称找到组合意义上的全局最优 5K；
- 不声称一个 scalar score 可普适衡量 agent SFT utility；
- 不把使用最终 test prompts 的选择称为严格 OOD；
- 不把使用已训练 DEITA/MIG scorer 称为方法级别的严格 training-free；
- 不把 Data Shapley 作为 100K 逐样本选择的可行方案。
- 不声称首次 trajectory-aware agent data selection、critical-step learning、
  target-aware selection、facility-location 或 token-efficient trajectory curation；
- 不把 LLM 判断的 criticality 称为 verified/causal credit。

---

## 3. 为什么 flat Top-K 可能失败

同一 trajectory 的派生 rows 往往共享：

- 相同任务、初始状态和工具定义；
- 高重叠 prefix 与 observation；
- 相邻 action；
- 相同 final answer；
- 相同接口和格式模板。

因此 row-level Top-K 可能：

1. 让少量长 trajectory 垄断 5K；
2. 重复学习 JSON/ReAct/interface surface form；
3. 过度选择 final-answer 或最高-loss turns；
4. 把 sibling redundancy 误当成质量；
5. 在 row 层看似无重复，实际发生 task-family leakage；
6. 牺牲规划、状态 grounding、recovery 和 no-tool 边界行为。

本 proposal 的基本选择单位为：

\[
\text{task family}\rightarrow
\text{trajectory}\rightarrow
\text{decision event}\rightarrow
\text{SFT row}.
\]

---

## 4. 数据协议：先分组，再展开

### 4.1 Group key

在 trajectory expansion 前构造：

```text
family_key =
    benchmark
  + normalized task goal/template
  + environment instance/seed
  + initial-state signature
  + canonicalized tool semantics
  + target-state or answer type
```

以下对象必须整体进入同一个 partition：

- 同一原始 trajectory 的所有 rows；
- 同一 task instance 的多次 rollout；
- 仅替换实体、随机种子或格式的同模板任务；
- 同一 prompt family 的成功/失败轨迹；
- embedding/MinHash 判定的近重复 connected component。

### 4.2 四个互斥集合

| Split | 用途 | Selector 可见性 |
|---|---|---:|
| Candidate \(U\) | 100K 候选池 | 可见 |
| Selection-query \(Q\) | target relevance，仅 prompts | 可见 |
| Validation \(V\) | 选权重、阈值、checkpoint | 只评价 |
| Test \(T\) | 最终检验 | 不可见 |

若当前总共只有 1K 可评价 prompts，则必须：

- 将其中一部分改为 validation，例如 200/800；或
- 额外构造 validation。

否则不能同时声称使用 target-aware selection 且拥有 untouched 1K test。

### 4.3 三种 evaluation setting

1. **In-domain inductive**
   - benchmark 相同；
   - task family/trajectory 完全不重叠。
2. **Strict benchmark-OOD**
   - 整个 benchmark/family 不参与 selector tuning 和 query construction；
   - 若只有 20 个 benchmark，使用 5-fold benchmark holdout。
3. **Target-aware OOD**
   - 可提供 32/128/512 条独立、无标签 target prompts；
   - 明确称为 unlabeled target-domain adaptation，而非 zero-shot OOD。

---

## 5. 方法：TA-Coreset

### Stage 0 — Hard validity filtering

直接删除，而不是软降权：

- trajectory/SFT 格式无法解析或被截断；
- tool 不存在或 arguments 不符合 schema；
- action/observation 时序错误；
- final answer 与环境结果冲突；
- assistant input 含未来 observation、reward 或答案；
- 未纠正错误 action 被作为正向 SFT target；
- 仅 context 多一轮但 target 相同的 sibling row；
- 大段固定 boilerplate 或工具文档复制；
- 与 test family 冲突的 exact/near duplicate。

### Stage 1 — Critical decision-point extraction

每条 trajectory 只进行一次 step-level 分析，将 turns 标记为：

1. `PLAN`
2. `CRITICAL_OBSERVATION`
3. `TOOL_SELECTION`
4. `ARGUMENT_GROUNDING`
5. `PIVOTAL_ACTION`
6. `RECOVERY`
7. `STOP_OR_FINAL`
8. `FORMAT_ONLY`

优先保留前七类；`FORMAT_ONLY` 仅保留最低覆盖。

每条 trajectory 的候选 critical turns 上限：

\[
\min(3,\lceil0.3T\rceil).
\]

若训练框架支持 loss masking：

- 保留完整 causal prefix；
- 只在 selected assistant turns 上计算 loss。

否则物化：

\[
(\text{goal}+\tau_{1:t-1})\rightarrow(h_t,a_t),
\]

禁止把未来 trajectory 放进输入。

### Stage 2 — Per-row quality and learnability

每个候选 \(i\) 计算：

- \(v_i\)：verifier correctness / action validity；
- \(p_i\)：state progress 或经验 advantage；
- \(k_i\)：criticality；
- \(o_i\)：observation grounding；
- \(r_i\)：recovery value；
- \(a_i\)：instruction-response alignment；
- \(\ell_i\)：student/proxy learnability；
- \(u_i\)：token utility。

所有分数先在 `benchmark × decision_type` 内做 percentile 或 robust
z-normalization，避免不同 benchmark 的 reward、长度和 NLL 不可比。

质量项：

\[
q_i =
w_vv_i+w_pp_i+w_kk_i+w_oo_i+w_rr_i+w_aa_i+w_\ell\ell_i+w_uu_i.
\]

初始 pilot 不预注册精确最终权重，只预注册：

- validity/verifier 为最高权重；
- PPL/IFD 不能成为唯一质量信号；
- 所有权重必须在 validation 上冻结后才打开 test。

#### Difficulty 使用 band-pass，而非 Top-PPL

令 \(r_i^{\mathrm{NLL}}\) 为组内 NLL percentile：

\[
\ell_i=
\exp\left(
-\frac{(r_i^{\mathrm{NLL}}-\mu)^2}{2\sigma^2}
\right).
\]

Pilot 取 \(\mu=0.7,\sigma=0.25\)，并与 monotonic IFD、无 difficulty
做消融。极高 loss 可能是错误、格式异常或不可学习样本。

### Stage 3 — Target coverage

对独立 selection-query prompts \(Q\)，计算多视图 embedding：

- task/goal view；
- state/decision view；
- 可选 response/label view。

目标覆盖：

\[
F_{\mathrm{target}}(S)
=
\sum_{v\in Q}
\alpha_v\max_{i\in S}\operatorname{sim}(v,i).
\]

这不是按平均 target similarity 排序，而是保证每个 target mode 有代表点。

### Stage 4 — Candidate representativeness

从候选池构建分层 landmarks \(\mathcal L\) 或 cluster centroids：

\[
F_{\mathrm{pool}}(S)
=
\sum_{u\in\mathcal L}
\rho_u\max_{i\in S}\operatorname{sim}(u,i).
\]

禁止建立完整 \(100K^2\) 相似度矩阵。实现采用：

- ANN/top-k neighbor graph；
- 2K–10K stratified landmarks；
- lazy 或 stochastic greedy。

### Stage 5 — Explicit metadata coverage

覆盖：

- benchmark；
- skill/task family；
- tool family/topology；
- decision type；
- early/middle/late；
- success/recovery/no-tool；
- length 和 difficulty bin。

\[
F_{\mathrm{meta}}(S)
=
\sum_h\omega_h
\log\left(1+\frac{n_h(S)}{\tau_h}\right).
\]

递减边际收益防止单个 benchmark、工具或格式占满预算。

### Stage 6 — Constrained set optimization

联合目标：

\[
\begin{aligned}
F(S)=&
\lambda_q\sum_{i\in S}q_i\\
&+\lambda_tF_{\mathrm{target}}(S)\\
&+\lambda_pF_{\mathrm{pool}}(S)\\
&+\lambda_mF_{\mathrm{meta}}(S).
\end{aligned}
\]

约束：

\[
|S|=5000,
\]

\[
\sum_{i\in S}c_i\le B,
\]

\[
|S\cap U_g|\le m_g,
\]

\[
l_b\le |S\cap U_b|\le u_b.
\]

主实验预注册：

- \(m_g=1\)；
- \(m_g=2\) 和无 cap 为消融；
- exact 5K 和 fixed assistant-target-token 两条评测轨道。

Greedy rule：

\[
i^\star=
\arg\max_i
\frac{\Delta F(i\mid S)}{c_i^\gamma}.
\]

- count-controlled：\(\gamma=0\)；
- token-controlled：\(\gamma=1\)；
- \(\gamma=0.5\) 仅作 sensitivity。

---

## 6. In-domain 与 OOD 版本

### In-domain / target-aware

初始 weight family：

```text
quality          35–50%
target coverage  20–35%
pool coverage    15–25%
metadata         10–20%
```

### Strict OOD / generalist

不使用 held-out test query relevance：

```text
quality          35–50%
target coverage   0%
pool coverage    25–40%
metadata         20–35%
```

OOD 重点增加：

- state/argument grounding；
- recovery；
- no-tool/stop；
- tool/interface semantic invariance；
- benchmark 和 skill breadth。

---

## 7. Low-training extension

TA-Coreset 先将 100K 缩至约 15K–20K，再运行 LESS-style rerank：

1. 用 500–1K 条高覆盖数据做短 LoRA warmup；
2. 保存 2–4 个 early checkpoints；
3. 为 shortlist 计算 projected Adam/LoRA gradient features；
4. 用独立 labelled validation examples 计算 target gradients；
5. 将 gradient alignment 作为额外 relevance 项；
6. 最后仍执行 trajectory cap、token budget 和 coverage optimization。

不得：

- 使用 final test labels/rewards 计算 target gradient；
- 在只有 prompts 时把 teacher pseudo-label selection 描述为 ground-truth influence；
- 忽略 selection backward cost。

---

## 8. Baselines

### 必做

1. `Zero-shot student`
2. `Full-100K`
3. `Row-Random-5K`
4. `Benchmark+Trajectory-Stratified-Random-5K`
5. `Constraint-Matched-Random-5K`
6. `Longest-Response-5K`
7. `RDS+-5K`
8. `IFD/IRA + trajectory cap`
9. `Quality + diversity / DEITA-style`
10. `TA-Coreset-5K`
11. `TA-Coreset + LESS`

主 null baseline 不是 row-random，而是：

- 匹配 benchmark distribution；
- 在 benchmark 内均匀抽 trajectory；
- 匹配 decision type、length bin 和 target-token budget；
- 每 trajectory 使用同样 cap。

否则 proposed method 的收益可能仅来自去 sibling redundancy。

---

## 9. 核心假设与可证伪预测

### H1 — Grouping dominates flat scoring

\[
\text{trajectory-stratified random}
>
\text{row-random}.
\]

若差异已经很大，说明主要矛盾是 sibling redundancy，而非复杂 quality score。

### H2 — Critical decisions beat indiscriminate imitation

在相同 loss-bearing tokens 下：

\[
\text{critical steps}
>
\text{random steps / final-only / top-PPL / all steps}.
\]

### H3 — Target relevance mainly helps ID

\[
\Delta_{\mathrm{relevance}}^{ID}
>
\Delta_{\mathrm{relevance}}^{OOD}.
\]

严格 OOD 下 pool/skill coverage 应比 target surface similarity 更稳健。

### H4 — Quality and coverage are complementary

完整 \(Q+R+D+C\) 应优于：

- quality-only；
- relevance-only；
- diversity-only；
- quality + dedup。

### H5 — LESS is an upper bound, not automatically cost-effective

LESS 可能提高 target performance，但只有在：

\[
\text{performance gain}/C_{\mathrm{total}}
\]

仍优于 forward-only selector 时，才算实用收益。

---

## 10. 最小可行实验

### Phase 0 — Data audit，无 GPU 训练

输出：

- rows per trajectory 分布和 Gini；
- exact/near duplicate rate；
- benchmark/skill/tool/decision-type 分布；
- target/context-token 分布；
- success/recovery/no-tool 比例；
- test-family contamination audit；
- `cap=1/2/∞` 下可用候选数。

**Go gate：**

- 100K 中存在可观 sibling redundancy 或 benchmark imbalance；
- 至少能构造 5K 个 validity 通过、来自广泛 trajectories 的候选。

若 5K 本身几乎等于全部有效独立 decision points，则复杂 selection 问题不成立。

### Phase 1 — Zero-update selector pilot

比较：

1. stratified random；
2. longest；
3. RDS+；
4. IFD/IRA + cap；
5. TA-Coreset。

每种 3 个 paired SFT seeds，统一：

- student checkpoint；
- optimizer/schedule；
- processed target tokens；
- evaluation task/environment seeds。

只使用 validation。

### Phase 2 — Low-training upper bound

仅当 TA-Coreset 在 Phase 1 至少满足：

- 3 seeds 平均优于 stratified random；
- 至少 2/3 seeds 获胜；
- OOD/retention 无明显退化；

才增加：

- LESS；
- TA-Coreset + LESS。

### Phase 3 — Frozen final test

冻结：

- selector weights；
- trajectory cap；
- query count；
- SFT hyperparameters；
- checkpoint rule；
- evaluator。

然后一次性打开 final test。

---

## 11. 评价与统计

### 主指标

优先使用 benchmark-native success/reward：

\[
M_{\mathrm{macro}}
=
\frac1{B}\sum_b
\frac1{|T_b|}
\sum_{x\in T_b}\operatorname{score}(x).
\]

ID 与 OOD 分开报告。

### 次要指标

- worst benchmark 和 bottom-20% CVaR；
- tool-selection/argument validity；
- no-tool hallucination；
- recovery rate；
- final-answer groundedness；
- interface/tool rename robustness；
- episode length 和 inference tokens；
- general instruction capability retention。

### Dataset diagnostics

- unique trajectories；
- per-trajectory count/Gini；
- benchmark entropy；
- skill/tool/decision coverage；
- near-duplicate rate；
- subset pairwise similarity；
- selector Jaccard stability；
- total target/context tokens；
- selection + SFT GPU-hours。

### 统计单位

不能把 100K 派生 rows 当成独立样本。使用：

- paired end-to-end SFT seeds；
- benchmark/task-family hierarchical bootstrap；
- 预注册 primary contrasts：
  1. TA-Coreset vs stratified random；
  2. TA-Coreset vs RDS+；
  3. TA-Coreset vs TA-Coreset+LESS。

初筛 3 seeds；正式结论至少 5，优先 10 个 paired seeds，或先用 pilot
variance 做 power analysis。

---

## 12. 成功、降级与 kill 条件

### 升级为 active

同时满足：

1. Phase 0 证明 flat rows 存在明显 sibling redundancy/imbalance；
2. TA-Coreset 在 fixed-token 轨道优于 trajectory-stratified random；
3. 增益不能仅由 trajectory cap 或 benchmark balancing 解释；
4. 至少一个 OOD/held-out benchmark setting 不退化；
5. 相比 RDS+ 形成性能或总成本 Pareto 改善。

### 降级为工程 recipe

- grouping/cap 有效，但 quality/relevance/coverage 的复杂组合不优于
  constraint-matched random；
- 方法只适用于一个 student 或一个 benchmark；
- 主要收益来自更长 target tokens。

此时保留“先按 trajectory 去重/分层抽样”的工程结论，不主张新 selector。

### Kill

- constraint-matched random 与 TA-Coreset 在足够 seeds 上无实际差异；
- RDS+ 稳定支配 TA-Coreset 的性能和成本；
- critical-step selection 不优于随机 step；
- target-aware 增益在 untouched test 上消失；
- selector 对 proxy/judge choice 极不稳定；
- 无法避免 scorer/evaluator circularity 或 task-family leakage。

---

## 13. 首轮推荐配置

如果直接构造第一版 5K：

1. trajectory/task-family split 与 decontamination；
2. rule verifier 硬过滤；
3. 每 trajectory 选 1–3 个 critical candidates；
4. 小 proxy 计算 alignment + band-pass difficulty；
5. 使用 task/decision 双视图 embedding；
6. 以 target coverage + candidate facility-location + metadata coverage 选集合；
7. 主结果 `trajectory cap=1`；
8. exact-5K 与 fixed-target-token 两条轨道；
9. success/recovery/boundary 大致以 `70/20/10` 作为 pilot，而不是硬性结论；
10. 未纠正错误 action 永不作为正向 SFT label。

该 proposal 的最小独立贡献应是：

> 把 agent SFT selection 从 iid instruction-row ranking，重构为具有
> trajectory dependency、decision credit、target coverage 和 token/partition
> constraints 的层次化 coreset 问题，并用强 constraint-matched random
> 自证其增益不是简单去重或配额效应。
