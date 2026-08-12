---
name: read-the-trainer-docstring-before-designing-a-control
description: 派实验前先读 trainer 的 docstring/正文——退化性经常已经写在源码里；2026-08-12 我委托的 A02 j=0 control 在 train_qcmem_distill.py:37 明文写着 teacher==student by construction
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**设计一个对照实验之前，先读那个 trainer 自己的 docstring 和关键分支。** 实验是否**退化（degenerate）**这件事，经常已经被原作者写在源码里了。

## 2026-08-12 的实例

A02 的 `STATUS.json:next_gate[4]` 说：「**NOT tested anywhere yet**: a matched-quality depth
control (a LoRA distilled for j=0) — without it, depth cannot be isolated from adapter quality.
This is the only remaining way to make a clean depth claim, and it requires TRAINING, not eval.」

我照此派了一个 8 卡训练。agent 回来说：**这个对照按构造就是退化的**，而证据是——

`scripts/train_qcmem_distill.py` 第 37-39 行，原作者自己写的：
> "Correctness: at `--resume_j 0` teacher==student by construction (both are the full forward,
> adapters are zero-init so make no difference at step 0)"

TEACHER = QCMem 在 j=0 的读（adapters disabled, no_grad），STUDENT = 在 j=`--resume_j` 的读。
`resume_j=0` 时两者是**同一条计算路径**，只差一个零初始化的 LoRA delta → top-k KL 恒为 0。

实测：loss@step1 = **0.0000**，step10 = **0.0021**，跑到 step 880 仍是 **0.0020**；
而 flagship j=12 在 step10 是 **0.2991**。

**科学后果不是「修好了这个缺陷」，而是「缺陷消失了」**：j=0 的最优 adapter 就是 identity =
base model，而它**已经在盘上**（dvr arm `j0_top12`）。所以原本那个 `read_deployed` 的
−3 到 −12 pp **本身就已经是** matched-quality 的深度对比。§7.1 的「没有任何臂在 matched-quality
adapter 下隔离了深度」是**过度悲观**，next_gate[4] 应当**由演示退休**。

## Why

`next_gate` / `PROPOSAL.md` 里的「NOT tested anywhere yet」是**某个时点某个人的判断**，不是
对代码的断言。它可能因为没读实现而漏掉「这个实验在数学上是空的」。我把它当成了事实，
直接换算成 8 卡训练预算。

## How to apply

1. 派任何**新对照臂**之前，先 grep 那个 trainer 的 docstring + 定义 teacher/student（或
   arm A/arm B）的那几行。**问一句：在我要设的这个参数取值上，两个臂会不会退化成同一个东西？**
2. 若怀疑退化 → 先跑一个 **1 GPU / 20 step 的 GATE 0 探针**（成本几分钟），看 loss 是不是
   ~0。不要直接上 8 卡多小时。
3. **退化的对照要照样跑完并落账，但结论要写「defect dissolves」而不是「defect repaired」**
   —— 并说明「最优参数下该臂 == 某个已在盘上的臂」。这本身是可发表的澄清。
4. 派 agent 时明确允许它**否定任务前提**并给出证据；本次 agent 正是这样做的，还预注册了
   自己的证伪条件（「若 A1 结果好于 base，我的 vacuity 论证就错了，必须撤回」）。

配套：[[one-sample-is-not-a-trend-or-state]]（先补测量再下结论）、
[[prior-work-differentiate-dont-abandon]]（审计的产出是定位，不是判死）。
