# ★★ 最严重: Paper B depth ladder 横跨两个不同大小的训练语料，且 appendix 对此陈述错误

**日期**: 2026-08-08 ~13:4x CST。**发现者**: workflow adversarial agent（FATAL 21 与 FATAL 2，
两个独立 lens 各自发现）；**MAIN 已实测复核确认**。
**这条比 budget defect（#192）和今晚所有其它缺陷都严重，因为它使 depth ladder 的跨行比较失去
共同基础，且论文对自己的训练数据做了错误陈述。**

## 实测（从训练 log 的 `dataset rows=` 直读，这是训练时真实加载的行数）

| arm | 训练语料 rows | 物理盘 | log |
|---|---:|---|---|
| keep14 + fresh2 | **7,570,911** | wzc1 | `logs/olmo2_7B_keep14fresh2.log` |
| ShortGPT-16 | **7,570,911** | wzc1 | `logs/olmo2_7B_shortgpt16.log` |
| freeze_front | **7,570,911** | wzc1 | `logs/olmo2_7B_keep14fresh2_freezefront.log` |
| keep8 + fresh2 | **15,491,607** | zwfy6 | `zwfy6:logs/olmo2_7B_keep8fresh2.log` |
| keep10 + fresh2 | **15,491,607** | zwfy6 | `zwfy6:logs/olmo2_7B_keep10fresh2.log` |
| keep12 + fresh2 | **15,491,607** | zwfy6 | `zwfy6:logs/olmo2_7B_keep12fresh2.log` |

比值 **15,491,607 / 7,570,911 = 2.046×**。

两个 `.npy` 同名（`dolmino_now15b.npy`）但**不是同一文件** —— 前一个 verify agent 已实测
wzc1 版是 zwfy6 版的**严格字节前缀**（三段 md5 相同，wzc1 只是被截短）。所以：

- **同样跑 200k 步**，wzc1 上的 arm 走了 **3.38 epoch**（重复见过数据 3 次以上），
  zwfy6 上的 arm 走了 **1.65 epoch**。
- `DistributedSampler` 的 shuffle 排列依赖 `len(dataset)`，两盘 `len` 不同
  → **同 seed 同 epoch 的样本顺序完全不同**，steps/epoch 也不同（59,147 vs 121,028）。

## 论文对此的陈述是错的

`paperB/sections/08_appendix.tex:114`:

> "contains **7,570,911 windows** and the fixed in-domain validation array contains ..."

**这句话对 keep8 / keep10 / keep12 三个 arm 是假的** —— 它们训练在 15,491,607 windows 上。
appendix 把它写成单一训练数组，实际是两个。

## 为什么这比 #192 严重

#192（budget defect）说的是**同一语料上步数不齐**（keep8=121k / keep10=83.5k / keep12=124k
vs keep14/SG16=200k）。可以靠如实披露 steps 化解，且我此前论证过"keep8 多 45% 预算却 core6
更低"反而是更强证据。

**但语料不同意味着**：
1. "多 45% 预算"这个说法本身**不成立** —— keep8 的 121k 步在 15.5M 语料上 = 1.0 epoch，
   keep10 的 83.5k 步在同语料 = 0.69 epoch，而 keep14 的 200k 步在 7.57M 语料上 = 3.38 epoch。
   **token 预算、epoch 数、数据多样性三者全不可比。**
2. 我昨天写进 `PAPERB_RESUME_WARM_RESTART_DEFECT.md` 的那个"宝石论证"
   （keep8 比 keep10 多 45% 预算却 core6 更低 ⇒ ladder 单调性在更不利条件下成立）
   **失效** —— 两个 arm 虽同盘同语料（都是 zwfy6 15.5M），这一对**内部**仍可比，
   但把它们与 keep14/SG16 放同一张表做 depth ladder 就不可比了。
   > adversarial agent 的 FATAL 25/26 另外指出：即使 keep8-vs-keep12 这一对，
   > 0.67pp 的差异也**落在同一 arm 内相邻 checkpoint 的抖动范围内**
   > （keep12 在 111.5k-124k 有 18 个 core6 点，peak-to-peak 1.185pp、sd 0.286pp；
   > 仅 123500→124000 这 500 步就摆 0.668pp）。**MAIN 未独立复核这一条，但若成立，
   > 那个"宝石"是双重失效。**
3. Table 4 的 depth ladder 是 Paper B 的**主表**。行与行之间现在有三个混淆同时存在：
   **depth × 训练步数 × 训练语料大小**。

## 我今晚的判断错误（第五次）

我在 `PAPERB_RESUME_WARM_RESTART_DEFECT.md` 与 heartbeat 里两次论证"选项 A（如实披露
真实 steps）比选项 B 更强，因为 keep8 多 45% 预算却更低"。**这个论证建立在"同一语料"
的隐含假设上，而该假设为假。** 我从未查过各 arm 的 `dataset rows=`。

顺序记录（今晚第 5 次归因/判断错误）:
1. flip boundary = runtime jitter（自己错）
2. flip boundary = driver drift（自己错）
3. flip boundary = batch size（自己错，真凶是 torch 版本，subagent 找到）
4. P2.4 第五 confound = Dolmino step（接受 subagent 假设，自己用 PPL 反驳）
5. **"keep8 多 45% 预算"论证（自己错，未查语料大小）** ← 本条

## 必须做的（按优先级）

- [ ] **P0：修 `08_appendix.tex:114`** —— 不能声称单一 7,570,911 windows。必须如实写
      两个语料及各 arm 的归属，或改成 per-arm 表格。
- [ ] **P0：Table 4 caption / 正文**必须交代 depth ladder 的行跨两个语料，
      因此不是 controlled depth comparison。
- [ ] **P0：撤回我自己的"宝石"论证** —— 已在本文件记录，需同步到
      `PAPERB_RESUME_WARM_RESTART_DEFECT.md`。
- [ ] P1：核实 adversarial agent 的 FATAL 25/26（相邻 checkpoint 抖动 1.185pp）——
      若成立，则任何 <1.2pp 的 ladder 行间差异都需要 checkpoint-抖动 error bar。
      **数据已在盘上**（keep12 有 18 点、keep10 有 23 点），纯 CPU 可算。
- [ ] P1：这也解释了为什么 keep8/10/12 与 keep14/SG16 的"budget"看起来不齐 ——
      部分是因为**同一步数在不同语料上意义不同**。

## 我尚未核实的 adversarial 结论（不当作已确认）

28 个 fatal 里我只复核了 FATAL 21 + FATAL 2（本条）。其余高优先级但**未复核**的:
- FATAL 9/12/13: **margin 是从错的 metric 算的** —— core6 5/6 task 用 acc_norm，
  但 `per_example` 的 `option_scores` 是 raw sum-logprob，所以所有 near-tie / margin
  分析描述的是 acc 决策不是 acc_norm 决策。若成立，我今晚"near-tie 密度"整条 mechanism
  叙述的定量部分都要重做（定性方向可能仍对）。
- FATAL 3/14: 中介检验 observed/predicted ≈ 1.0 是**代数恒等式**不是证据。
- FATAL 22: "GPU architecture" 不是单变量 —— 同时改了 arch 和 NVIDIA driver 大版本。
- FATAL 5/6: dLLM 侧 P-B 的 churn/|net| 命题**不可证伪**；且"McNemar 暴露更高"方向反了。
- FATAL 7/8/28: P2.4 "intact 是格式漂移 / pruned 是真丢知识" 被 item-level 分解推翻，
  且可能是 **输出长度** artifact（`contains` 是子串测试，回答越长越容易命中）。

## Provenance

- 训练 log（wzc1）: `logs/olmo2_7B_{keep14fresh2,shortgpt16,keep14fresh2_freezefront}.log`
  → `dataset rows=7570911`
- 训练 log（zwfy6）: `logs/olmo2_7B_{keep8,keep10,keep12}fresh2.log`
  → `dataset rows=15491607`
- 错误陈述: `paperB/sections/08_appendix.tex:114`
- 前置发现: workflow `wf_fef88742-1c9` adversarial 阶段 FATAL 2 / FATAL 21；
  数据集前缀关系由 verify agent `adfaca37dbdc` 实测（三段 md5 + 文件大小）
- 被本条推翻的我的文件: `status/PAPERB_RESUME_WARM_RESTART_DEFECT.md` 的
  "选项 A 更强" 论证段
