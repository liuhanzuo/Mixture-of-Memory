# Related-Work Gap Audit for the Proposal Repository

日期：2026-08-08。

## 目的

本审计不是每篇 proposal 的最终 bibliography，而是：

1. 判断当前入口是否已经有 reviewer 可检查的新颖性边界；
2. 指定必须补的 prior-work 类别；
3. 防止“内部证据充分”被误当作“方法新颖性已核实”。

状态：

- **充分**：有具名 closest collisions、不得主张和安全空缺；
- **部分充分**：有边界意识，但缺一手来源或逐项 collision；
- **不足**：主要是内部动机/实验设计，没有 Related Work；
- **严重不足**：方向太宽，容易落入已有问题设定。

---

## Active proposals

| ID | 状态 | 必补 Related Work | 下一步 |
|---|---|---|---|
| A01 null calibration | 部分充分 | construct validity 与 input-blind/label-only baseline；MC/QA option priors；CKA/RSA correspondence null；probe controls 与 native readout | 在 `PROPOSAL.md` 加 closest-collision 表；`SOURCES.md` 增加外部一手文献。 |
| A02 CoMem write/read repair | 不足 | reusable prefix/KV/hidden-state memory；activation/KV compression 与 selective decompression；chunk-local vs contextual encoding；split inference/readout repair；write/read joint distillation | 先明确贡献是 context-aware write、portable read repair，还是 fixed-cost joint repair。 |
| A03 parametric vs external memory | **严重不足** | parametric vs non-parametric knowledge；CPT/continual update 与 forgetting；knowledge editing；RAG/search routing；temporal conflict；latent/residual memory；cost-aware hybrid | 必须加 capability × update regime × cost collision matrix；不能只写“CPT vs RAG vs memory”。 |
| A04 recovery certification | 不足 | layer pruning/drop；post-pruning CPT/distillation；activation alignment；generative-reasoning recovery limits；PPL vs downstream capability；multi-seed/non-inferiority certification | 新颖性必须是“认证规则/停止条件”，不是普通 pruning recovery study。 |

### A02 已核到的近期直接碰撞

- **SeDeM: Selective Decompression of Hidden-State Memories for Long-Context
  Question Answering**, arXiv `2608.00311`
  - intermediate hidden-state storage；
  - query-conditioned selection；
  - selected block decompression 到 decoder-compatible hidden states。
- **Persistent Memory for Continuous Latent Reasoning**, arXiv `2606.07720`
  - learned write/read/forget gates over persistent residual memory。
- Q-Filters、PromptDistill 及大量 KV/context compression 工作
  - 说明“压缩/选择中间表示”本身不是空白。

A02 必须明确：

> CoMem 的候选新意是 **跨 query 可复用 object 的独立 Write interface tax 与
> context-aware Write/portable Read repair**，而不是一般 hidden-state
> compression。

### A03 已核到的近期直接碰撞

- **Knowledgeless Language Models**, arXiv `2607.12831`
  - 通过 pretraining 抑制 parametric recall、增强 evidence grounding。
- **When Should LLMs Search?**, arXiv `2607.05752`
  - instance-level no-search/search/unsolved routing。
- **Navigating Unreliable Parametric and Contextual Knowledge**, arXiv
  `2606.20245`
  - 参数知识与外部 context conflict resolution。
- **Retrievable Gradients**, arXiv `2606.15734`
  - 可检索、临时参数适配，直接位于 CPT 与 RAG 之间。
- 近期 memory-update/conflict benchmark 和 agent memory survey
  - 说明 update regime 与 conflict 已高度拥挤。

A03 必须把安全空缺收窄为：

> 在同一结构受损模型、同 evidence 与统一总成本下，比较参数恢复、raw RAG、
> reusable residual memory 及 joint system 对旧知识、新知识、更新冲突和
> 多证据组合的分工。

### A04 已核到的近期直接碰撞

- **Ghosted Layers**, arXiv `2605.15491`
  - training-free activation alignment 恢复 layer-pruned LLM。
- **On the Limits of Layer Pruning for Generative Reasoning**, arXiv
  `2602.01997`
  - classification 可恢复，但 GSM8K/HumanEval+ 等 generative reasoning
    恢复受限。
- **Layer as Puzzle Pieces**, arXiv `2510.15304`
  - layer merging + hierarchical distillation recovery。

A04 不能主张首次研究 pruning recovery。安全主张是：

> 预注册、多 seed、matched token/FLOPs 的 **recovery certification**：
> 什么联合指标和停止规则足以宣布结构损伤已恢复？

---

## Backlog proposals

| ID | 状态 | 必补 Related Work | 安全边界 |
|---|---|---|---|
| B01 semantic bottleneck | 不足 | bottleneck Transformer；activation/KV codec；split-inference compression；recurrent/compressive memory；semantic/latent codec | 区分 post-hoc compression 与 pretraining-time formation of a persistable latent。 |
| B02 adaptive depth/read budget | 不足 | early exit/adaptive depth；RAG/search routing；dynamic top-k/multi-hop；SLA-constrained routing；adaptive cache budget | 必须证明联合 controller 不只是独立 router 的线性组合。 |
| B03 cyclic reset boundary | 部分充分 | LLF/layer reinitialization；plasticity loss；prune-regrow；optimizer-state reset；single-pass vs repeated-data | 已正确降级为 regime-boundary gate，不得包装成新 reset 方法。 |
| B04 eval fragility | 不足 | numerical/hardware/batch nondeterminism；benchmark ranking instability；margin/calibration；damage robustness；mediation | 新意只能是 model damage 是否系统性放大 nuisance sensitivity。 |
| B05 semantic handoff | 不足但自限 | logit/tuned lens；layerwise emergence；causal tracing；early exit；split computing；model stitching/readout adapter | `j_content/j_native/j_adapt` phase diagram，而非 universal semantic cut。 |
| B06 portable decompression | 不足 | activation decompression adapter；split-compute reconstruction；adapter transfer；intermediate self-distillation；cross-codec portability | “portable”至少要求多 task、多 compressor、多 model 或明确 layer/module transfer。 |
| B07 mutable serving | 不足 | prefix/KV caching；paged/disaggregated KV；versioning/invalidation；memory tiering；reuse-aware admission；incremental recompute | 需要逐 feature systems collision table。 |
| B08 memory applications | **严重不足** | query-focused context compression；grounded notes/provenance；personal/conversational memory；temporal KG/event sourcing；hierarchical memory | 三个子方向最好拆开，各写 Related Work。 |
| B09 trajectory-aware SFT selection | **较充分** | offline RL/imitation coreset；laminar/group constraints；verified decision credit；完整组合 collision | 主要继续补 `SOURCES.md`；发现直接 collision 时再收窄。 |

### B01 已核到的近期碰撞

- **RAC: Reference-Aware Activation Compression**, arXiv `2608.04991`
  - split LLM boundary activation codec。
- **SeDeM**, arXiv `2608.00311`
  - hidden memory compression + selective decompression。
- **PromptDistill**, arXiv `2503.23274`
  - early-layer token selection 与 hidden-state retention。
- **Q-Filters**, arXiv `2503.02812`
  - QK geometry based KV compression。

B01 的 Related Work 必须明确：

```text
这些工作压缩已有 representation；
B01 声称通过 pretraining 让 representation 本身成为低维、可持久化 latent。
```

若最终仍需恢复为 full-width hidden 才能读，方法差异明显削弱。

### B02 已核到的直接碰撞

- **AdaPonderLM**, arXiv `2603.01914`
  - token-wise adaptive depth。
- **RAGRouter-Bench**, arXiv `2602.00296`
  - adaptive RAG routing。
- **When Should LLMs Search?**, arXiv `2607.05752`
  - search/no-search oracle 与 learned routing。

因此 B02 必须先做 per-example joint oracle headroom；没有超越最佳独立
depth/router policy 的 headroom，就不训练联合 controller。

### B07/B08 的文献压力

近期工作已覆盖：

- distributed prefix/KV store；
- cross-model KV transfer；
- HBM/CPU/CXL memory serving；
- adaptive user memory；
- hierarchical/graph memory；
- memory-update gap；
- version control/rollback；
- long-term memory benchmarks 与 surveys。

所以 B07/B08 不能依靠“可版本化/可更新/分层 memory”这一功能清单主张
新颖性。必须给出：

- 与现有系统逐 feature 差异；
- 端到端质量/latency/bytes；
- stale/conflict failure；
- 明确 workload（尤其跨 query reuse）。

---

## 优先补齐顺序

1. **A03**：问题最宽，最容易被已有 CPT/RAG/editing/memory 工作覆盖。
2. **A02**：近期 SeDeM 与 hidden-state memory 已形成直接 collision。
3. **A04**：Ghosted Layers 与 pruning recovery limits 必须正面讨论。
4. **B01**：activation compression/latent memory 已高度活跃。
5. A01、B03：已有边界意识，补具名来源。
6. B02/B05/B06/B07/B08：在 promotion 前完成。
7. B09：保持当前审计质量，继续查完整组合 collision。

---

## 执行规则

每个 proposal promotion PR/commit 应同时修改：

```text
PROPOSAL.md
SOURCES.md
STATUS.json
```

`STATUS.json` 建议新增：

```json
{
  "related_work_status": "audited|partial|missing",
  "closest_prior_art": [],
  "novelty_status": "..."
}
```

若 `related_work_status != audited`，不得从 backlog 升到 active，也不得启动
用于 headline 的大规模 GPU sweep。

