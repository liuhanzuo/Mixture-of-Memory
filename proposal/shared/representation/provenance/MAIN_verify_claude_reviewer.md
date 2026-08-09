# MAIN 独立核实：claude reviewer 的 D_direction 报告

**核实日期**：2026-08-06 16:0x GMT+8
**被核实对象**：`paperD_research/direction_20260806/D_direction_claude_reviewer.md`
**核实方法**：本机经 `hy-proxy.woa.com:3128` 直取 `arxiv.org/abs/<id>` 的 `citation_title` +
Semantic Scholar Graph API `paper/arXiv:<id>?fields=venue,publicationVenue,externalIds`
（**带 4 次重试**——Paper C 教训：429 是限流不是"无 venue"，必须重试到 http200 才能下结论）。

⚠️ **`export.arxiv.org/api/query` 在本机经代理返回空**（对已知真实的 `2405.07987` 也返回空），
所以 arXiv API 不可用作核实通道，**必须用 abs 页 `citation_title`**。reviewer 声称用 arXiv API
核实过，那条通道在我这里是坏的；但它给的标题经 abs 页复核**全部正确**，所以标题不是幻觉。

---

## 1. 标题核实：13/13 全部真实，无幻觉

| arXiv | abs 页 `citation_title`（我实取） | reviewer 标题 | 一致 |
|---|---|---|---|
| 2403.17887 | The Unreasonable Ineffectiveness of the Deeper Layers | 同 | ✓ |
| 2403.03853 | ShortGPT: Layers in Large Language Models are More Redundant Than You Expect | 同 | ✓ |
| 2402.02834 | Shortened LLaMA: Depth Pruning for Large Language Models with Comparison… | 同 | ✓ |
| 2502.05795 | The Curse of Depth in Large Language Models | 同 | ✓ |
| 2503.21676 | How do language models learn facts? Dynamics, curricula and hallucinations | 同 | ✓ |
| 2402.04177 | Scaling Laws for Downstream Task Performance of Large Language Models | 同 | ✓ |
| 2304.01373 | Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling | 同 | ✓ |
| 2312.12141 | Neuron-Level Knowledge Attribution in Large Language Models | 同 | ✓ |
| 2405.07987 | The Platonic Representation Hypothesis | 同 | ✓ |
| 2210.16156 | （S2 venue=ICLR，见下）Reliability of CKA as a Similarity Measure… | 同 | ✓ |
| **2602.14486** | **Revisiting the Platonic Representation Hypothesis: An Aristotelian View** | 同 | ✓ |
| **2606.16897** | **Contrastive-Difference CKA Reveals Concept-Specific Structural Alignment Across Language Model Architectures** | 同 | ✓ |

**两篇 2026 preprint 是真的，不是幻觉。** 这一点最关键，因为它们是 reviewer 判 D2/D5 受阻的依据。

---

## 2. ★ Venue 核实：4 处 reviewer 标错，2 处方向相反

`arxiv.org/abs` 的 `citation_journal_title` 对**全部 13 篇都是空的** → reviewer 报告里的
venue 标注（ICML 2024 / EMNLP 2024 / AAAI 2024 / ICML 2023 / NeurIPS 2022）**不可能来自它自称的
arXiv API 通道**。我用 S2 独立核：

| arXiv | reviewer 标的 venue | **S2 实测 venue** | 判定 |
|---|---|---|---|
| 2403.17887 | "preprint, later ICML 2024 spotlight" | **ICLR**, DBLP `conf/iclr/GromovTSGR25` | ❌ **是 ICLR'25，不是 ICML'24** |
| 2502.05795 | "2025 preprint" | **NeurIPS**, DBLP `conf/nips/SunSLYZL25` | ❌ **是 NeurIPS'25 peer-reviewed，不是 preprint** |
| 2210.16156 | "NeurIPS 2022" | **ICLR** (type=conference) | ❌ **是 ICLR，不是 NeurIPS** |
| 2405.07987 | "NeurIPS 2024" | **ICML** (type=conference) | ❌ **是 ICML，不是 NeurIPS** |
| 2503.21676 | "2025 preprint" | `arXiv.org` | ✓ preprint 正确 |
| 2602.14486 | "2026 preprint" | （S2 未收录，abs 页无 journal_ref） | ✓ preprint 合理 |
| 2606.16897 | "2026 preprint" | （S2 未收录） | ✓ preprint 合理 |
| 2403.03853 / 2402.02834 / 2304.01373 / 2312.12141 | ShortGPT/Shortened-LLaMA/Pythia/Neuron-Level 分别标 preprint / EMNLP'24 / ICML'23 / AAAI'24 | S2 按 ID 查 **全部 None**；title 搜也只有 Shortened LLaMA 命中且 venue=`arXiv.org` | ⚠️ **未核实，进 .bib 前必须查 ACL Anthology / PMLR / AAAI proceedings** |

### 两处「方向相反」的错误为什么重要

1. **2403.17887 = ICLR'25**（不是 ICML'24）。这篇是 D1/D3/D4 三个方向的**共同头号竞争者**
   （prune 连续层段 + QLoRA heal）。把它记成 ICML'24 spotlight 会低估它的时间新近度：
   它是 **ICLR'25**，即比我们判断的更晚、更"当前"。
2. **2502.05795 "Curse of Depth" = NeurIPS'25 peer-reviewed**（不是 preprint）。reviewer 把它当
   preprint 轻描淡写，实际它是**已过审的 NeurIPS 论文**，讨论「现代 LLM 深层系统性低效」，
   与 D1/D3「深层可裁」的框架直接相邻。**必须在 related work 正面处理，不能当 preprint 略过。**

---

## 3. 内容核实：两篇 2026 preprint 的摘要是否真的占掉 D2/D5

我实取了两篇摘要原文。

### 2602.14486（Gröger, Wen, Brbić，2026-02-16）摘要要点（原文）
> "existing metrics used to measure representational similarity are **confounded by network scale**:
> increasing model depth or width can systematically inflate representational similarity scores.
> To correct these effects, we introduce a **permutation-based null-calibration framework** that
> transforms any representational similarity metric into a calibrated score with statistical
> guarantees. … the apparent convergence reported by global spectral measures **largely disappears
> after calibration**, while **local neighborhood similarity** … retains significant agreement"

**MAIN 判定**：reviewer 说「与我们 R4 的 shuffle-null 在精神上几乎相同」——**这一半成立，一半不成立**：
- **成立**：它确实先于我们提出 permutation-based null calibration，且核心批判（未校准的相似度被
  inflate）与我们 shuffle-null 得到的 +0.038 结论方向一致。**它是 R4 H3 那条 null-calibration
  贡献的直接 prior，必须引，且不能声称我们首创 null calibration。**
- **不成立**：它的 permutation 是 **跨 network scale 的 confound 校正**（depth/width inflation），
  我们的 shuffle 是 **打乱 B 的层序**——问的是"层 i 是否真是层 j 的正确配偶"，
  是 **layer-correspondence** 而非 scale confound。两者数学上不是同一个 null。
  ⇒ **D5 的边际贡献仍存在**（depth-diagonal 上的层序 null），但**必须显式区分**，
  并承认 scale-confound 校正的优先权归它。

### 2606.16897（Gao Xueping，2026-06-15）摘要要点（原文）
> "moderate **geometric convergence coexists with near-perfect functional transfer**. Using
> contrastive-difference CKA (CKA_Delta) … **architectural outlier detector (Gemma: d=1.08, AUC=0.79)**"

**MAIN 判定**：reviewer 说它「blocked D2」——**基本成立，且比 reviewer 说得更硬**：
- 它已经做了 **cross-architecture CKA + 架构 outlier detection**，而 D2 的整个 pitch 就是
  「OLMo-2 是 14 模型里最难对齐的 outlier，用 post-norm 解释它」。**"架构 outlier detector"
  这个功能位已被占**（它甚至给了具体 outlier：Gemma d=1.08）。
- 但它的 outlier 归因是**经验性的**（"practical regime classifier"），**没有把 outlier 归因到
  pre-norm/post-norm 这个具体机制**。所以 D2 若坚持做，唯一活口是**因果干预**
  （训 pre-norm vs post-norm 双生子），而这要花 GPU 预算在一个已被占掉功能位的方向上。
- ⚠️ 一个 reviewer 没说的**反向机会**：它报的是「geometric convergence 中等，但 **functional
  transfer 近乎完美**」。这与我们 R3 实测的 **oracle affine ppl 596 vs 原 18.8（差 32×）**
  **直接矛盾**。我们有一个 peer-review 级的反例数据点。这本身可能比 D2/D5 都更有价值
  ——见下文「MAIN 新增候选」。

---

## 4. MAIN 对 reviewer 结论的修正

| reviewer 结论 | MAIN 核实后 |
|---|---|
| D1 top-1（3-way dissociation） | **保留为 top-1 候选**，但风险被低估：头号竞争者是 **ICLR'25**（不是 ICML'24），且 **NeurIPS'25 的 Curse of Depth** 也在同一框架内。这两篇都是 peer-reviewed 且都比 reviewer 标注的更"当前"。 |
| D2 effectively blocked | **同意 blocked，且理由更硬**（2606.16897 已占"架构 outlier detector"功能位）。 |
| D3 top-2（logit-lens onset 作剪枝准则） | **保留**，但 4 篇竞争者里 2 篇（ShortGPT、Neuron-Level）的 venue **我没核到**，进 .bib 前必查。 |
| D5 = workshop/findings 级 | **同意**，但边际贡献必须重写为「**depth-diagonal 上的 layer-order null**」，并**显式让出** scale-confound null calibration 的优先权给 2602.14486。 |
| Q3「没有论文报告过 CKA U-shape」 | ⚠️ **这是弱证据**。reviewer 只查了 4 篇 CKA/Platonic 系；我没能力在这轮穷举。**"无人报告过"是重指控，门槛要高**（[[two-disk-rule-applies-to-main-too]] 同一教训）。进论文前需要一轮专门的 U-shape 检索。 |

## 5. ★ MAIN 新增候选（reviewer 没提，来自核实过程）

**D6：「几何对齐 vs 功能可迁移」的直接对撞测量。**
- 2606.16897（2026-06）声称 cross-architecture 上「moderate geometric convergence coexists with
  **near-perfect functional transfer**」。
- 我们 R3 实测的 oracle affine readout（1 层 stitch 的**能力下界**，已经是最优线性映射）在
  OLMo-2-1B→Llama-3.2-1B 上给出 **ppl 596 vs 原 18.8（32×）**，即**功能迁移是灾难性的**。
- 我们还有 R4 的 91 pair 几何量 + shuffle null。
- ⇒ 一个直接命题：**「功能可迁移性」的结论强依赖于探针是分类头还是生成头**。concept-probe
  级的 functional transfer 可以近乎完美，同时 next-token generation 级的 transfer 差 32×。
  这不是反驳它，是**给它的结论划定适用边界**——而且我们两边的数据都已经在盘上（零新 GPU）。
- **kill risk**：如果它的 "functional transfer" 定义本来就只声称 concept-probe 级
  （需读全文确认，摘要里 "near-perfect functional transfer" 未限定），那我们只是在
  复述它的 scope，不构成贡献。**必须先读 2606.16897 全文再决定。**

---

## 6. 核实成本记录

- 标题核实 13/13 ✓（abs 页 `citation_title`，代理直取）
- venue 核实 9 篇（S2 带重试到 http200），**发现 4 处错**，4 篇未核到
- 摘要实取 2 篇 2026 preprint 原文
- **arXiv API 通道在本机不可用**（对照组也返回空）→ 后续核实一律走 abs 页 + S2

**教训**：subagent 自称「VERIFIED via arxiv API」时，那条通道在本机可能根本不通。
标题它没错（说明它确实拿到了什么），但 **venue 4/9 错** —— 说明 venue 是它推断/记忆的，
不是核到的。**venue 必须 MAIN 自己核**，这与 Paper C 的教训完全一致
（当时一个 subagent 把 S2 的 429 当成 "preprint only"）。
