# A01 — Execution-Grounded Typed Subtree Repair with Diffusion Code Models

## 状态

**ACTIVE / GATE-ONLY。**

这是 `dllm_draft` 当前最高优先级的方法 proposal，但尚无可引用的神经修复
结果。第一阶段只验证 repair operator 的 oracle upper bound；oracle gate
未通过前，不训练 fault localizer，不扩大 benchmark，也不启动新的
Scaffold full-generation SFT。

---

## 1. 工作标题

主标题：

> **Edit the Failing Subtree, Not the Whole Program: Execution-Grounded Typed
> Repair with Diffusion Code Models**

备选：

> **Typed Subtree Rediffusion for Preservation-Aware Program Repair**

项目简称：

```text
TypedRepair-DLM
```

---

## 2. 一句话主张

> 当程序错误可以定位到一个局部 typed subtree 时，masked diffusion code
> model 可以只把该子树退回原生 mask 状态并双向重生成，在严格保持其余程序
> 不变的前提下，以低于完整重生成的代价修复一部分执行错误。

本 proposal **不**预设该主张成立。首先测试的是：

```text
即使给出 oracle 错误子树，局部 rediffusion 是否真的能修？
```

如果答案是否定的，则 localization、RL 和新训练全部没有必要。

---

## 3. 为什么从完整生成转向局部修复

### 3.1 当前 full-generation 证据不支持继续

现有 Scaffold Stage-1 的 Plus pass@1：

| 方法 | HumanEval+ | MBPP+ |
|---|---:|---:|
| Scaffold Medium | 0.177 | 0.354 |
| Dream-Coder Instruct, 512 NFE | 0.707 | 0.680 |
| Qwen2.5-Coder-7B AR control | 约 0.494–0.518 | 约 0.648–0.651 |

Scaffold 的结构约束改善了低预算 parseability，但主要失败已经转为语义：
HumanEval 上约 62.2% 的任务属于成功生成且可解析、却无法通过基础测试。

语义保持训练也没有救活 Scaffold 接口：

- `scale=0.25` 的 vanilla HE+ 为 56.10%；
- 同 checkpoint 的 Scaffold HE+ 为 0；
- 142/164 任务发生结构容量失败；
- teacher-KL 可保住 vanilla parseability，但仍触发结构递归；
- leaf-only elastic 训练后 token-level expansion 仍为零。

因此下一步不应继续让弱结构 checkpoint 独立写完整程序。

### 3.2 仓库已经具备真正的局部修订 primitive

当前 runtime 已实现：

- stable typed anchors；
- committed lexical cell provenance；
- `remask_leaf(cell_id)`；
- `completed_structural_subtrees()`；
- `backtrack_structural_subtree(anchor_id)`；
- deepest-first C2 collapse；
- correction rounds、backtracks 和失败成本遥测。

对应代码：

```text
scaffold_coder/decoder_runtime.py:474-560
scaffold_coder/model_sampler.py:517-557
tests/test_model_sampler.py:201-284
```

重要澄清：

> C2 reverse primitive 已实现；尚未实现的是“执行反馈/静态诊断
> → typed anchor → 强 checkpoint 局部 rediffusion”的完整神经 repair
> pipeline。

### 3.3 旧 refinement 结果不能使用

旧实验有两个独立问题：

1. 早期 verifier 只检查是否抛异常，不比较函数返回值；
2. 修复后的 `restart` 从 prompt 重新生成，完全不读取原 draft。

当前 `remask` arm 也只是把 draft 尾部拼回 prompt，不是真正的 canvas
remasking。仓库中 16 个 `*.BROKEN_VERIFIER` run 永久排除。

---

## 4. 核心研究问题

### RQ1：Oracle-localized repair operator 是否有正 upper bound？

给定 gold 错误区域，比较：

- flat token-span remask；
- C1 lexical remask；
- typed C2 subtree collapse；
- full regeneration；
- AR FIM / iterative AR repair。

如果 oracle typed repair 不优于 matched random，则论文终止。

### RQ2：Typed region 是否比同尺寸 flat span 更适合作为修订单元？

typed subtree 的潜在优势是：

- 对齐 Python 语义边界；
- 保留兄弟节点和外围上下文；
- 允许 statement/construct 级插入删除；
- 将结构合法性留给 deterministic runtime。

但 typed region 也可能过大，退化为 disguised regeneration。因此必须同时
报告 region size、outside-region preservation 和成本。

### RQ3：执行反馈能否定位到可修的最小 region？

只有 RQ1 通过后才研究：

- parser/static diagnostic；
- undefined name/type/operator rule；
- SBFL/Ochiai；
- failing-test delta；
- model regeneration probability；
- denoising trajectory stability。

### RQ4：局部修复相对完整重生成的价值是什么？

可能的价值不只体现在最终 pass@1，还包括：

- 不破坏已经正确的代码；
- 更小 patch；
- 更低 cumulative model tokens；
- 更低 wall time；
- 更容易审计和回滚。

如果完整重生成在相同成本下质量不差，则 typed repair 不成立。

### RQ5：哪些错误类型适合 dLLM repair？

预期分层：

- replacement-only operator/value/variable misuse；
- expression/statement 级缺失逻辑；
- excess logic / deletion；
- length-changing subtree；
- 多点依赖修改。

第一篇只要求在局部、单区域修复上建立结果；多区域一致性编辑是后续扩展。

---

## 5. 方法

## 5.1 输入与状态

输入：

```text
task specification
buggy or failed draft P
visible tests / parser / static-analysis evidence E
candidate typed region R
```

输出：

```text
repaired program P'
```

约束：

```text
P'[outside(R)] == P[outside(R)]
```

除非实验臂显式允许 region expansion，否则外围 token 必须 bit-exact。

## 5.2 Oracle region construction

第一阶段从 HumanEvalPack-Python 的：

```text
buggy_solution
canonical_solution
```

构造 gold edit。

流程：

1. `ast.parse` buggy 和 canonical；
2. 使用 line/column offsets 建立 source spans；
3. 计算 text diff 与 AST node diff 的最小覆盖；
4. 选择同时覆盖所有 changed tokens 的最深共同 AST ancestor；
5. 将 region 映射到 tokenizer IDs；
6. 记录：
   - node type；
   - char/token/line span；
   - replacement-only 或 length-changing；
   - gold token-length delta；
   - region / whole-program token ratio。

若 AST 映射不唯一，任务在 manifest 中显式标记，不允许人工选择有利 region。

## 5.3 Flat span rediffusion

强 Dream-Coder checkpoint 接收：

```text
prefix + [MASK]^m + suffix
```

其中 `m` 的供给必须对所有相应 arm 对称：

- oracle-length diagnostic；
- non-oracle fixed bucket；
- learned length bucket（仅后续）。

第一阶段可以报告 oracle-length upper bound，但不能用其结果直接宣称实际系统
优于 AR；必须同时给 non-oracle arm。

## 5.4 Typed subtree rediffusion

typed arm 将程序转为可逆 typed state：

```text
program
→ typed AST/runtime tree
→ select anchor R
→ collapse R to one typed mask
→ allocate lexical slots
→ strong dLLM predicts subtree content
→ deterministic render
```

这里有两个实现层次：

### T1：AST wrapper + strong flat dLLM

- typed tree 仅负责 region boundary、render 和 preservation；
- region 内仍使用 Dream flat infilling；
- 最容易验证“typed unit 是否有价值”。

### T2：Scaffold C2 runtime + strong semantic proposal model

- 复用 `backtrack_structural_subtree`；
- 不使用弱 Stage-1 checkpoint承担完整语义；
- 需要 adapter 将强 Dream 的 token predictions 接入 typed runtime；
- topology/length 若无法由强模型可靠提供，则使用独立小 head，而不是共享
  lexical meta-token softmax。

第一轮先完成 T1；T1 没有 upper-bound 信号时，不投入 T2。

## 5.5 Repair acceptance

系统只能基于 visible evidence 接受 patch：

- parser/static checks；
- HumanEvalPack 提供的可见测试；
- CanItEdit 自带 tests。

最终评价使用完整 held-out tests。不得使用 hidden tests 进行 region selection、
candidate ranking 或回滚。

接受策略：

```text
若 visible tests 改善且无已通过测试回归 → 接受
否则保留原程序
```

同时报告：

- generated candidate；
- accepted candidate；
- visible gain；
- hidden/generalization gain；
- regression。

---

## 6. 第一阶段实验：64-task Oracle Gate

## 6.1 数据

主数据：

```text
data/edit/humanevalpack_python.parquet
```

共 164 条 Python repair pairs：

| Bug type | 数量 |
|---|---:|
| value misuse | 44 |
| missing logic | 33 |
| excess logic | 31 |
| operator misuse | 25 |
| variable misuse | 23 |
| function misuse | 8 |

简单 line-diff proxy 下，138/164 是单行变化。

构造固定 64-task manifest：

- 按 bug type 分层；
- 按 changed-line 数分层；
- replacement-only / length-changing 分层；
- 按 region token ratio 分层；
- 不根据任何模型结果选题；
- 全部 buggy 必须 fail、canonical 必须 pass。

其余 100 条留作后续 learned-localizer/full evaluation，不参与 threshold 选择。

第二阶段外部验证：

```text
data/edit/canitedit.parquet
```

105 条任务，adaptive/perfective/corrective 各 35 条。只有 HumanEvalPack gate
通过后才进入。

## 6.2 模型

### dLLM

```text
models/Dream-Coder-v0-Instruct-7B
```

现有同 harness 结果：

- HumanEval+：116/164 = 70.73%；
- MBPP+：257/378 = 67.99%。

### AR/FIM

```text
models/Qwen2.5-Coder-7B
```

它与 Dream 具有相同的 7B 级宽度/深度配置，且有原生 FIM 路径。它是必须的
产品级 matched-capacity baseline，但不是严格的 AR-vs-diffusion 因果匹配，
因为 post-training 不同。

### Scaffold checkpoint

```text
models/Scaffold-v0-stage1-7B
```

只用于：

- 验证 runtime serialization；
- 复用 C1/C2 primitive；
- 做弱 proposal-model 负对照。

不把它作为主修复模型。

## 6.3 实验臂

| Arm | Region | Generator | Purpose |
|---|---|---|---|
| A0 | none | none | buggy baseline |
| A1 | gold token span | Dream dLLM | flat oracle remask |
| A2 | random same-size token span | Dream dLLM | localization null |
| A3 | gold AST subtree | Dream dLLM | typed-boundary upper bound |
| A4 | random matched AST subtree | Dream dLLM | typed localization null |
| A5 | gold lexical cell, where valid | Dream dLLM | C1 scope |
| A6 | gold typed C2 anchor | strong proposal + runtime | target method |
| A7 | full program | Dream dLLM | diffusion restart |
| A8 | gold FIM span | Qwen AR-FIM | strong edit baseline |
| A9 | full program / iterative patch | Qwen AR | strong repair baseline |
| A10 | gold subtree | Scaffold Stage-1 | weak-checkpoint negative control |

所有随机 arm 使用固定 seeds，并匹配：

- region token count；
- node depth（AST arm）；
- replacement/length-change type；
- 可用上下文。

## 6.4 两个长度协议

### Diagnostic oracle-length

所有局部 arm均获得 gold output length。用途仅是测 repair operator ceiling。

### Deployable non-oracle

统一使用长度 bucket：

```text
1 / 2 / 4 / 8 / 16 / 32 / 64+
```

bucket 只能由 buggy region、AST type、visible diagnostics 预测，不能读取
canonical patch。

任何 headline 必须基于 non-oracle arm；oracle 只作为机制 upper bound。

## 6.5 主要指标

质量：

- strict test pass；
- gained / lost / unchanged；
- parseability；
- visible-pass / hidden-fail；
- bug-type slice；
- replacement/length-change slice。

preservation：

- outside-region exact token match；
- unchanged-line exact match；
- normalized patch size；
- over-edit / under-edit；
- deletion precision/recall。

成本：

- NFE（仅作 dLLM 内部描述）；
- `tokens_fed`；
- attended-context integral；
- wall-clock；
- GPU time；
- peak memory；
- failure-inclusive mean/P50/P90。

统计：

- paired bootstrap interval；
- exact McNemar；
- 多 arm 的预注册 primary comparison；
- 不以 aggregate pass 掩盖 newly-lost。

---

## 7. Gate 与止损

## 7.1 Operator gate

Oracle typed arm必须同时满足：

```text
gained >= 5 / 64
lost <= 2 / 64
outside-region exact preservation = 100%
extra cumulative model tokens <= 35%
```

并且：

- 优于 matched random subtree；
- 不被 Qwen FIM 在质量和成本上共同支配；
- 相对 full Dream regeneration 至少提供明确的 preservation 或成本收益。

失败则：

```text
停止 typed repair 主线
不训练 localizer
不做 execution RL
不扩大 CanItEdit
```

## 7.2 Typed-unit gate

若 gold AST subtree 与同长度 flat span 的差异：

- 小于 3pp；
- 或 preservation/成本不更好；

则删除“typed subtree 是关键修复单元”的方法主张，最多保留 flat local
rediffusion 工程结果。

## 7.3 Non-oracle length gate

non-oracle arm必须保留 oracle arm至少 90% 的新增修复，并且 generation
failure 不超过 5%。否则停止当前长度机制。

## 7.4 Learned-localizer gate

只有 operator gate 通过后才执行。

要求：

- eligible-task coverage ≥60%；
- hit@3 或 subtree IoU 显著高于 matched random；
- end-to-end gained ≥5/64；
- regression ≤2/64；
- 不依赖 hidden tests。

---

## 8. 第二阶段：Execution-Grounded Localization

候选信号：

1. syntax/parse error span；
2. undefined name、wrong arity、type/operator diagnostic；
3. failing-test input/output delta；
4. SBFL/Ochiai top-k；
5. AST/data-flow neighborhood；
6. Dream regeneration probability；
7. denoising trajectory persistence/entropy；
8. 多信号 rank fusion。

必须比较：

- random；
- pure static；
- pure execution；
- pure dLLM confidence；
- static + execution；
- static + execution + dLLM。

不将 traceback 行号直接视为 fault location；测试框架中的 assertion traceback
可能只指向测试代码。

---

## 9. 第三阶段：多点一致性修复

仅当前两阶段通过后扩展。

任务类型：

- function signature + call sites；
- variable definition + uses；
- type annotation + conversions；
- duplicated guards/invariants；
- paired delimiter；
- sibling bugs。

所有 region 必须位于同一 canvas，且 `k+1` 严格包含 k 的原 region。
对照：

- simultaneous dLLM；
- serial dLLM；
- iterative AR FIM；
- full regeneration；
- SiblingRepair/MultiFixer 类 workflow baseline。

成功线：

- 相对最佳 AR repair 至少 +5pp，CI > 0；
- parseability 不低 3pp 以上；
- `tokens_fed <= 5x` AR；
- wall time `<= 2x` AR；
- 去标识符/新鲜变换后复现。

---

## 10. Related Work 与新颖性边界

本节是 proposal 的组成部分，不是投稿前再补的装饰。当前日期为
2026-08-08；具体来源与核查状态见 `SOURCES.md`。

## 10.1 dLLM code generation 与 infilling

Dream-Coder 和 DiffuCoder 证明了 code-specialized dLLM 可以进行完整代码
生成，并强调 any-order decoding、generation order 与 diffusion-native RL。
DreamOn进一步研究 variable-length code infilling。

与本 proposal 的区别：

- 不主张新的 code dLLM backbone；
- 不主张 any-order infilling 本身；
- 不以完整生成 pass@1 为核心；
- 研究对象是**已有失败程序上的局部、可保留、可验证修复**。

DreamOn 已覆盖 expand/delete 式长度控制，因此长度弹性不能单独成为本文新意。

## 10.2 dLLM remasking、editing 与 draft-refine

最接近的工作包括：

- Targeted Remasking：token-to-mask 纠错；
- SCOPE+D3IM：可见 token 的自纠正 sampler 与 sampler-matched training；
- MRP：残差预测与 aggressive decoding 后的 remasking；
- Edit-Based Refinement / ME-DLM：插入、删除、替换式后编辑；
- Multi-Block Editing：跨 block 重开上下文修改；
- Speculative Correction：完整 draft 后的全局/局部 diffusion refinement；
- Detect-Remask-Repair：对 evolving summary 进行检测、局部 remask 和保真修复；
- OSCAR、DiSE、TACG：分别使用 cross-chain、regeneration 或 trajectory
  信号进行定位、自评或 commit gating。

这些工作已经覆盖：

```text
remask
local/global refinement
draft-then-refine
trajectory confidence
preservation trade-off
```

因此本文不能主张：

- 首次 remask；
- 首次 dLLM self-correction；
- 首次 local refinement；
- 首次使用 confidence/trajectory 定位错误。

剩余空缺限定为：

> **execution-grounded、typed AST/subtree-localized 的程序修复，并把未编辑
> 区域 preservation、paired regression、AR/FIM 和 full-regeneration
> controls 作为主评价。**

## 10.3 Confidence remasking 的负结果

`Re-evaluating Confidence Remasking in Masked Diffusion Language Models`
显示 confidence-based post-hoc remasking 的收益依赖 setting，在标准短 block
下可能几乎没有增益。

这直接决定本文实验顺序：

```text
先 oracle operator
后 localizer
```

如果 oracle region 都不能修，继续调 confidence threshold 没有科学价值。

## 10.4 自动程序修复与 fault localization

传统和 neural APR 已研究：

- AST/graph diff；
- fine-grained fix localization；
- SBFL；
- tree/subtree transformation；
- execution-guided search；
- repository/file/function localization；
- multi-hunk 与 sibling repair。

代表性近邻：

- Graph2Diff：在代码图上预测 AST diff；
- Beep：预测必须修改的 AST/token element 与 repair action；
- DEAR：结合 SBFL、data flow 与 tree context 处理 multi-statement/multi-hunk；
- AutoCodeRover：AST code search + SBFL + LLM patching；
- CodePilot：execution-guided MCTS repair；
- SHERLOC、Loc2Repair：将 localization 与 repair 解耦并测量 gold/predicted
  localization headroom；
- SiblingRepair、MultiFixer：多点相关修复。

本文不主张 typed/AST repair 是全新概念。区别在于：

```text
传统/AR APR：定位后生成 patch 或 diff
本文：定位后把 typed subtree 返回 masked-diffusion 原生状态，
      双向重生成，并严格冻结外围程序
```

## 10.5 Code editing benchmark 与 preservation

HumanEvalPack 提供跨语言的 synthesis/explanation/repair task；CanItEdit
专门测试 instruction-driven editing。近期 benchmark audit 表明现有 editing
数据在语言、领域和测试覆盖上仍较窄，因此本文第一阶段不能外推到真实仓库
维护。

Copy-as-Decode 强调大量编辑任务主要复制原输入，并把 copy preservation 与
编辑效率作为 decoding 问题；PAIR-Bench显式评价 progressive feedback、
targeted repair 和 already-correct behavior preservation；删除偏差工作进一步
说明“测试通过”不等于 patch 与目标 edit 一致。

因此本文将 preservation 设为 primary metric，而不是附录分析。

## 10.6 Grammar-constrained decoding

CFG-constrained diffusion decoding 和 EPIC 已能显著提高结构合法性并保留并行
commit。它们解决：

```text
partial sequence 是否仍能完成为语法合法程序
```

本文解决：

```text
哪个执行相关 subtree 应被重开，以及局部重开能否修复语义错误
```

语法合法不等于功能修复；当前仓库也已经证明 parseability 提高后语义错误仍是
主瓶颈。

## 10.7 dLLM for software engineering

`Exploring the Power of Diffusion Large Language Models for Software
Engineering` 报告了 generation、defect detection 和 repair 的广泛比较，
但使用的模型匹配与引用质量存在明显争议。本文不会用它作为“dLLM repair
已优于 AR”的证据，只将其视作必须讨论的直接 topic collision。

## 10.8 新颖性声明

如果实验通过，最安全的贡献表述是：

1. **repair-operator decomposition**：先用 oracle region 分离“能否修”与
   “能否定位”；
2. **typed mask-native repair operator**：将执行相关 AST subtree 退回
   dLLM 原生 mask 状态，而不是生成自由文本 diff；
3. **preservation-first protocol**：outside-region exactness、newly-lost 和
   failure-inclusive cost 为 primary metrics；
4. **matched repair controls**：同区域 Dream flat remask、随机 region、
   Qwen FIM/AR、full regeneration；
5. 若 learned localization 通过，再加入 execution/static/diffusion trajectory
   的组合定位。

不能主张：

- 首次 dLLM repair；
- 首次 remasking；
- 首次 AST repair；
- 首次 execution-guided repair；
- 首次 preservation-aware editing；
- dLLM 普遍优于 AR。

---

## 11. Reviewer 会攻击什么

### 攻击 1：Gold localization 给了答案

回应：

- oracle arm只用于 operator upper bound；
- headline 必须来自 non-oracle learned/static localization；
- random matched region 与 AR-FIM 同样获得相同 region/length 信息。

### 攻击 2：HumanEvalPack 太小且可能污染

回应：

- 64-task 只作 gate；
- 剩余 100 条为独立验证；
- 再到 CanItEdit；
- 加 identifier renaming / literal perturbation；
- 不以单个 benchmark 形成 family claim。

### 攻击 3：Typed subtree 只是换一种 span

回应：

- 直接比较同 token 数 flat span；
- 报告 region size、preservation、长度变化和成本；
- 若差异 <3pp，删除 typed contribution。

### 攻击 4：强 Dream 本身就能重生成

回应：

- full Dream regeneration 是必要 baseline；
- 只有局部修复在 preservation 或成本上有明确收益时，方法才成立。

### 攻击 5：AR FIM 更便宜

回应：

- Qwen FIM/iterative repair 是 primary baseline；
- 不用 NFE 做跨范式比较；
- 若 AR 在相同或更低成本下不差，proposal 被 kill。

### 攻击 6：Visible tests 导致 overfitting

回应：

- 最终用 held-out tests；
- 报 visible gain / hidden loss；
- 不允许 hidden tests 参与 selection；
- 使用 paired newly-gained/newly-lost。

---

## 12. 实施顺序

### CPU / 无模型阶段

1. 固定 64-task manifest；
2. 验证 buggy fail / canonical pass；
3. 构建 AST/token spans；
4. 测试 round-trip 与 outside-region preservation；
5. 建立 random matched-region sampler；
6. 写 paired scorer 和 cost schema；
7. 修复/替换当前 proxy `remask` harness。

### 16-task neural smoke

要求：

- 每个 arm 完整覆盖；
- mask 外 token 100% 不变；
- 至少一个 oracle arm触发实际修改；
- 无 prompt-splice proxy；
- EvalPlus/HumanEvalPack grader self-test 通过；
- cost telemetry 非空。

### 64-task gate

启动前检查 GPU，无 orphan；每个 8-GPU run 由后台 worker/subagent 管理，并
使用同一 frozen manifest。

### Full evaluation

只在 gate 通过后：

- HumanEvalPack 剩余 100；
- CanItEdit 105；
- 可选 fresh mutation/renaming controls；
- learned localizer。

---

## 13. 预期论文结构

1. Introduction
2. Related Work
3. Problem Definition and Preservation Protocol
4. Typed Subtree Remasking
5. Oracle Repair-Operator Gate
6. Execution-Grounded Localization
7. Experiments
8. Error-Type and Cost Analysis
9. Limitations
10. Conclusion

---

## 14. 最小可发表版本

最小版本必须同时包含：

1. HumanEvalPack-Python oracle/non-oracle split；
2. flat span、typed subtree、random、Dream restart、Qwen FIM/AR；
3. outside-region exactness；
4. gained/lost paired statistics；
5. 至少一个 learned/static localizer；
6. CanItEdit 或独立 second dataset；
7. 完整 Related Work 与 novelty collision 表；
8. 失败样本计入的成本。

只有 oracle 结果没有 end-to-end localization，不足以构成完整方法论文，但可以
作为明确的 go/no-go technical report。

