---
name: paperc-pc1-scooped-eval-invalid
description: "★2026-08-05三独立reviewer收敛+MAIN核实:PaperC P-C1构造被arXiv:2411.15558整条占掉、P-C2的forward-only probe hook被2210.10041(EMNLP'22)占掉、squad_val有49.85%恒定中文拒答标签→常量函数EM高于我们所有臂;P-C1降级不作主线,#134取消"
metadata: 
  node_type: memory
  type: project
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**Paper C（"shallow regrown cap on a frozen trunk"）的 P-C1 不能作为主线命题。**
2026-08-05 三名互相独立的 tcodex/gpt-5.6-sol reviewer 收敛到同一结论，MAIN 逐条回原始
文献核实为真（不是模型幻觉）。完整报告：`paperC_research/reviewers_20260805/R{1,2,3}_*.md`，
决策与证据沉淀在 `versions/paperC_scoping.md` 末节。

**三条硬结论**：
1. **构造被 scoop**：`arXiv:2411.15558`（Lu et al., "Reassessing Layer Pruning in LLMs"）
   摘要原文即"prune the final 25% of layers followed by fine-tuning the lm_head and the
   remaining last three layer"，模型是 Vicuna-7B/Qwen1.5-7B/Llama-3.1-8B。
   ⇒ 我们自称的三条区分点（真变浅 / trunk 真冻结 / decoder 7B）**一篇全占**。
   随机初始化替换那一侧另有 `2403.19135`(LLM-Streamline, ICLR'25)、`2407.16286`、
   `2410.02330`、`2401.02415`(LLaMA Pro)。
2. **P-C2 的 hook 也被 scoop**：`arXiv:2210.10041`（Xie et al., EMNLP'22 Findings）已用
   **forward-only、免训练**的 hidden-state variability 做 task-specific 选层 **+ 决定 head
   放哪层**（=丢弃上层）。**这是 P-C2 必须打败的 baseline**，不是可选引用。
3. **eval set validity gate 失败**：`data/squad_val.jsonl` 有 **997/2000=49.85%** 是同一句
   中文拒答 `根据提供的信息无法回答这个问题`（train 仅 17.56%，差 32.29pp）。
   ⇒ 一个**不看输入的常量函数 EM=49.85%**，而 A4=29.30 / A3=26.05 / BASE=33.85 **全部低于它**
   （只有 A2_lora=65.90 高）。**A4>A3 的 +3.25pp 不能支撑任何 init 优劣结论**；#133 的
   keep20/24/28 曲线(0.3440/0.3560/0.4190)同样全在常量线下。

**另外，我们自己 brief 里"唯一差异=继承+冻结 vs 随机初始化"是事实错误**：
`scripts/run_paperC_pc1.sh:61-66` → A4 lr=1e-4/2e-5、A3 lr=**3e-4**、A1 lr=1e-5。
单 lr mismatch 即可解释全部差异。每臂只 1 seed。A4 可训参数里 fresh 层只占 33%，
另 67% 是继承的 embedding+lm_head+norm（且 embedding 可训 → 名义冻结的前段也会被间接改）。
⇒ 行文只能叫 "frozen decoder prefix with trainable inherited lexical/readout modules + fresh cap"。
（**澄清**：`_classify_param` 的 `module.` 剥离与 #92 runner 同属 commit 7a330ce，所以
#92 A4 的差分 LR **确实生效**；我此前"差分 LR 是 no-op"的说法只适用于该修复之前的 run。）

**How to apply**：
- **#134 (A1 full-FT-32L ceiling) 已取消**，别再排。它在此 eval 上无意义，且当时挤占 .21
  害得 Paper B #103 crossing 掉 shard。
- 任何 Paper C 后续 GPU 工作**开跑前**必须先：(a) 重建/受控 eval set 并把
  **常量-拒答基线作为每张表的强制报告行**（低于它的数字一律不得称 capability）；
  (b) 补 lr/optimizer/seed 对齐的控制臂。
- 值得写的方向是重新框定（R3）：**"Preserve Here, Adapt There"** —— 保留知识所需的深度
  与最利于 adaptation 的深度不必相同，用 base 模型 depth 诊断**先验**分配深度预算；
  把崩坏的模型当**受控 lesion** 而非失败的压缩模型。probe 那个量要叫
  **task linearization depth**（readout 兼容性），不是 "adaptation onset"/"storage depth"
  —— 因为我们本地 full-FT CKA 曲线在 OLMo L18 **没有膝点**，而 knowledge logit-lens 有陡跳。

相关：[[tcodex-exec-no-dash-c-flag]]、[[kill-remote-gpu-job-by-pid-not-pkill]]、[[paperb-olmo2-base-not-chat]]
