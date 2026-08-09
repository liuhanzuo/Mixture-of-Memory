# AUDIT0 — Venue 核实（纯核实，不做方向判断）

**日期**：2026-08-06
**任务**：对 49 个 arXiv ID 做 venue 三路交叉核实（S2 Graph API + arXiv abs `citation_journal_title`/`journal_ref` + 论文 COMMENT 自述），
并优先裁决 `2411.15558` 的 MAIN-vs-tcodex 冲突。
**原始数据**：`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/paperC_v2_research/_venue_raw/`
（`s2_results.json`、`s2_match.json`、`dblp.json`、`dblp_author.json`、`openreview.json`、`arxiv_abs.json`、`abs_html/*.html`）

---

## 0. 方法与口径（先说清，因为它决定了下面每一行的可信度）

实际用了 **5 条通路**（比任务书要求的 3 路多 2 路，原因见 §0.2）：

| # | 通路 | 端点 | 状态 |
|---|------|------|------|
| C1 | S2 Graph（按 arXiv ID） | `api.semanticscholar.org/graph/v1/paper/arXiv:<id>?fields=title,venue,publicationVenue,externalIds,year,publicationTypes` | 严重限流，多轮 backoff 后 34/49 拿到 200 |
| C2 | S2 Graph（按标题 match） | `.../paper/search/match?query=<title>` | **独立配额**，把剩下 15 个补到 200（见 §0.2） |
| C3 | arXiv abs meta | `arxiv.org/abs/<id>` 的 `citation_journal_title` / `<td class="tablecell jref">` / `comments` | 49/49 拿到 200 |
| C4 | DBLP | `dblp.org/search/publ/api`（标题检索 + **作者集检索**）+ `dblp.org/rec/<key>.xml` 取准确 booktitle | 49/49（一次 500 后重试成功） |
| C5 | OpenReview API2 | `api2.openreview.net/notes/search?query=<title>` 取 `venue`/`venueid`/`invitations` | 49/49 |

**最终 S2 覆盖率 = 49/49 全部 HTTP 200**，因此**本报告中没有任何一行是因为 429 而下的 preprint 结论**，
也没有任何一行需要标 UNRESOLVED。

### 0.1 判定规则（严格按任务书）
- `"<会议> <年份> (peer-reviewed)"` ← 满足任一：S2 `publicationVenue.type ∈ {conference, journal}`；
  或 DBLP `conf/` / 非-corr `journals/` key；或 OpenReview `venueid` 为正式录用（**不是** `*/Submission`、
  `*_Withdrawn_Submission`、`*_Rejected_Submission`、`TMLR/Rejected`、`dblp.org/*`、`OpenReview.net/Public_Article`）。
- `"preprint"` ← 三路（这里是五路）都无正式记录。
- `"UNRESOLVED"` ← S2 持续非 200 或各路冲突无法裁决。**本次 0 行。**

### 0.2 两个方法学坑（会影响别人复现，必须写下来）
1. **`paper/arXiv:<id>` 与 `search/match` 是两套配额。** 前者被打到 429 打了 90+ 分钟只爬到 34/49；
   换 `search/match?query=<title>` 后 15 分钟内把剩下 15 个全部拿到 200。
   下次做 venue 核实**直接优先用 `search/match`**，别在 `paper/arXiv:` 上死等。
2. **标题会在 camera-ready 时被改，纯标题检索会漏。** 这次实测到 2 例：
   - `2310.04680`：arXiv 标题 *"The Cost of **Down-Scaling** Language Models: **Fact Recall** Deteriorates before In-Context Learning"*，
     ICLR 2024 正式标题是 *"The Cost of **Scaling Down** Large Language Models: Reducing Model Size Affects **Memory** before In-context Learning"*。
     标题检索 → 只见 CoRR；**作者集检索**（7 作者完全一致）→ `conf/iclr/JinCDNCRD24`。
   - `2402.02834`：arXiv-v2 标题 *"Shortened LLaMA: Depth Pruning ... with Comparison of Retraining Methods"*，
     S2 `search/match` 对 v2 标题返回 **404 "Title match not found"**；用 v1 标题
     *"Shortened LLaMA: A Simple Depth Pruning for Large Language Models"* → 200（venue=arXiv.org）。
   → 所以我对**所有**无 peer 证据的行都补跑了一遍作者集 DBLP 检索（`dblp_author.json`），
     除 `2310.04680` 外没有再发现新的改名录用。

---

## 1. ★ 优先冲突裁决：`2411.15558` "Reassessing Layer Pruning in LLMs: New Insights and Methods"

### 1.1 venue：**tcodex 对，MAIN 错。这篇是 ICLR 2026 Poster。**

OpenReview API2 对该标题返回 4 条同题 note，`venueid` 原文如下（`api2.openreview.net/notes/search?query=Reassessing+Layer+Pruning+in+LLMs`）:

| note id | `venue` 原文 | `venueid` 原文 | invitations（关键项） |
|---|---|---|---|
| `04Tfwy3LLC` | **`ICLR 2026 Poster`** | **`ICLR.cc/2026/Conference`** | `ICLR.cc/2026/Conference/Submission2804/-/Camera_Ready_Revision` ← **有 camera-ready** |
| `EjHtQlKEzV` | `ICLR 2025 Conference Withdrawn Submission` | `ICLR.cc/2025/Conference/Withdrawn_Submission` | `.../-/Withdrawn_Submission` |
| `BPRfeyGYbq` | `CoRR 2024` | `dblp.org/journals/CORR/2024` | `DBLP.org/-/Record` |
| `8xD1W1pNeJ` | (review 正文，非 paper note) | — | `ICML.cc/2024/Workshop/WANT/Submission8/-/Official_Review` |

**冲突为什么会发生**：这篇投过三次。ICLR 2025 **撤稿**（`Withdrawn_Submission`），
然后 ICLR 2026 **录用为 Poster**（Submission2804，已有 `Camera_Ready_Revision`）。
- MAIN 在 2026-08-05 查 S2 得 `venue=arXiv.org` → 这**不是查错**，是 **S2 尚未回填 ICLR 2026**
  （我今天复查 S2 仍是 `venue='arXiv.org'`, `publicationVenue=None`）。DBLP 同样只有 `journals/corr/abs-2411-15558`。
- 教训：**S2/DBLP 对最新一届会议（2026）有滞后**，"S2 说 arXiv.org" 不能直接推出 "preprint"。
  对 2026 年的会议，**OpenReview `venueid` + `Camera_Ready_Revision` 是比 S2/DBLP 更快、更权威的信号**。

**最终判定：ICLR 2026 Poster (peer-reviewed)。**（旧 Paper C 若曾按 "preprint" 弱化它，需改口。）

### 1.2 全文事实核实：它训练的是**剩余模型中最后 1–3 个预训练层**，**不是随机初始化的新层**。**tcodex 说的是对的。**

抓取：`https://arxiv.org/html/2411.15558`（200，1.17 MB，转纯文本 120,247 chars，落盘 `/tmp/full_2411.15558.txt`）。

证据 1 — **§4.2 "Is the LoRA family the best choice for post-pruning fine-tuning?" 中 Partial-layer Fine-tuning 的定义（原文逐字）**：

> "Partial-layer Fine-tuning. Compared to LoRA and QLoRA, which inject trainable low-rank factorization matrices into each layer, **partial-layer fine-tuning simply freezes the weights of some layers while updating only the specified layers** to save computing resources and time (Shen et al., 2021; Ngesthi et al., 2021; Peng & Wang, 2020). Following by the common practice of previous studies (Khan & Fang, 2023), **we choose to fine-tune only the later layers that are closer to the output, while keeping the earlier layers, which capture more general features, frozen.** Specifically, we use two different fine-tuning strategies: one is to finetune only the model head (lm_head only), and the other is to finetune the lm_head plus the last layer (lm_head + last layer), the last two layers (lm_head + last two layers), and the last three layers (lm_head + last three layers)."

→ "freezes ... / updating only the specified layers" 是对**已有预训练权重**的 freeze/update 二分；
全篇没有任何"新增层"的动作。

证据 2 — **Abstract（原文逐字）**：

> "Our results demonstrate that a simple approach, i.e., **pruning the final 25% of layers followed by fine-tuning the lm_head and the remaining last three layer**, yields remarkably strong performance."

→ 关键词是 "**the remaining** last three layer"，即**剪完之后剩下来的**最后三层，是存活的预训练层。

证据 3 — **反向检索（negative evidence）**：在 120k 字符全文中 grep
`regrow` / `re-grow` / `grow back` / `add(ing) new layers` / `newly initial` / `random(ly) initializ` /
`reinitializ` / `expand depth` / `up-scal` → **0 命中**。
（相对地，`Figure 1` 的三条 insight 是："1) Prune from the tail. 2) Fine-tune the last few layers (instead of using LoRA). 3) Iterative pruning benefits rarely."，全是剪+调，没有增。）

**结论**：2411.15558 = 「剪掉尾部 25% → 只训 lm_head + 剩余最后 1/2/3 个**预训练**层」。
它**没有**做「补随机初始化新层」。因此**它并不占掉 fresh-cap（随机初始化新层）这个构造**。
MAIN 的独立确认与 tcodex 一致。
（另外它的 §4.3 有 "iterative pruning"，但那是 score-prune-finetune-**merge** 循环，**只减不增**，
终点模型比原模型浅——与"剪完再长回原深度"的 cyclic prune-regrow 不是一回事。这一点是 §1.2 之外的补充观察，
供 MAIN 划界限时参考；我在此不做方向判断。）

---

## 2. 主表（49 行）

约定：
- **S2 venue / S2 type**：`venue` 与 `publicationVenue.type` 原文；`(match)` = 走 `search/match` 端点取得；空串写 `''`。
- **DBLP key**：只列**与本文同题**的非-CoRR key（`journals/corr/*` 一律记 `corr-only`）；booktitle 由 `dblp.org/rec/<key>.xml` 单独核准（所以能区分 main / Findings / Industry Track）。
- **arXiv journal_ref**：`<td class="tablecell jref">` 原文。
- **COMMENT 自述**：`<td class="tablecell comments">` 原文节选。
- 判定依据列的 `C1..C5` 对应 §0 的通路编号。

| arXiv | 标题(citation_title 实取) | S2 venue | S2 type | DBLP key | arXiv journal_ref | 论文 COMMENT 自述 | 最终判定 | 判定依据 |
|---|---|---|---|---|---|---|---|---|
| 2411.15558 | Reassessing Layer Pruning in LLMs: New Insights and Methods | arXiv.org | None | corr-only | — | — | **ICLR 2026 Poster (peer-reviewed)** | C5 `venueid=ICLR.cc/2026/Conference` + `Camera_Ready_Revision`；S2/DBLP 滞后未回填（详见 §1.1） |
| 2606.07978 | MechLens: Late Crystallization of Factual Knowledge Explains Intervention Effectiveness in Language Models | `''` | None | corr-only | — | — | **preprint** | C1 200 且 `publicationVenue=None`；C4 corr-only；C5 无 note |
| 2606.16897 | Contrastive-Difference CKA Reveals Concept-Specific Structural Alignment Across Language Model Architectures | `''` (match) | None | corr-only | — | — | **preprint** | C2 200 `publicationVenue=null`；C4 corr-only；C5 无 note |
| 2607.25663 | Localized Adaptation Reveals Distinct Learning Signatures in Transformers | `''` | None | corr-only | — | "Main text: 8 pages, 2 figures; appendix: 13 tables, 10 figures; code and data available at …"（**未自述任何 venue**） | **preprint** | C1 200 `publicationVenue=None`；C4 corr-only；C5 无 note |
| 2510.18871 | How Do LLMs Use Their Depth? | arXiv.org (match) | None | corr-only | — | — | **preprint**（注：有 workshop 录用 + ARR 在审） | C5 仅 `aclweb.org/ACL/ARR/2026/August/**Submission**`（在审）、`ICLR.cc/2026/Conference/**Rejected_Submission**`（被拒）、`aclweb.org/ACL/2026/Workshop/KnowFM`（**workshop** 录用）；主会无录用 |
| 2605.11416 | Freeze Deep, Train Shallow: Interpretable Layer Allocation for Continued Pre-Training | arXiv.org | None | corr-only | — | — | **preprint** | C5 仅 `ACL/ARR/2026/May/**Submission**` + `August/**Submission**`（均在审，非录用）；C1/C4 无正式记录 |
| 2605.02105 | Sharpness-Aware Pretraining Mitigates Catastrophic Forgetting | arXiv.org | None | corr-only | — | "43 pages, 64 figures, 9 tables, **accepted to ICML2026**" | **ICML 2026 (peer-reviewed)** | C5 `venueid=ICML.cc/2026/Conference`（`venue='ICML 2026 regular'`）+ `Submission19173/-/Camera_Ready_Revision`；C3 COMMENT 自述一致；S2/DBLP 2026 滞后 |
| 2602.11137 | Weight Decay Improves Language Model Plasticity | arXiv.org (match) | None | corr-only | — | — | **ICML 2026 (peer-reviewed)** | C5 `venueid=ICML.cc/2026/Conference` + `Submission26903/-/Camera_Ready_Revision`，`_bibtex` 为 `@inproceedings{...booktitle={Forty-third International…}}`；S2/DBLP 2026 滞后 |
| 2606.09932 | When RL Fails after SFT: Rejuvenating Model Plasticity for Robust SFT-to-RL Handoff | `''` | None | corr-only | — | — | **preprint** | C1 200 `publicationVenue=None`；C4 corr-only；C5 无 note |
| 2602.14486 | Revisiting the Platonic Representation Hypothesis: An Aristotelian View | arXiv.org | None | corr-only | — | "**ICML 2026 camera-ready**" | **ICML 2026 (peer-reviewed)** | C5 `venueid=ICML.cc/2026/Conference` + `Submission15852/-/Camera_Ready_Revision`；C3 COMMENT 自述一致 |
| 2410.06981 | Quantifying Feature Space Universality Across Large Language Models via Sparse Autoencoders | `''` | None | corr-only | — | — | **preprint** | C1 200 `publicationVenue=None`；C4 标题+作者双检索皆 corr-only；C5 无 note |
| 2503.04429 | Activation Space Interventions Can Be Transferred Between Large Language Models | International Conference on Machine Learning (match) | **conference** | `conf/icml/OozeerNPLHA25`（booktitle=ICML, 2025） | — | "75 pages. **Accepted to ICML 2025**" | **ICML 2025 (peer-reviewed)** | C2+C4+C5(`ICML.cc/2025/Conference`)+C3 四路一致 |
| 2312.02730 | Towards Measuring Representational Similarity of Large Language Models | arXiv.org | None | corr-only | — | "Extended abstract in **UniReps Workshop @ NeurIPS 2023**" | **preprint**（workshop extended abstract，非主会） | C1 `publicationVenue=None`；C4 corr-only；C5 仅 `NeurIPS.cc/2023/**Workshop**/UniReps`；C3 自述亦为 workshop extended abstract |
| 2109.08406 | Fine-Tuned Transformers Show Clusters of Similar Representations Across Layers | BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP (match) | **conference** | `conf/blackboxnlp/PhangLB21`（booktitle=BlackboxNLP@EMNLP, 2021） | — | "BlackboxNLP 2021" | **BlackboxNLP@EMNLP 2021 (peer-reviewed)**（注：是 workshop，但有 DBLP `conf/` key 且经同行评审） | C2 `type=conference` + C4 `conf/` key + C3 自述一致 |
| 2502.05795 | The Curse of Depth in Large Language Models | Neural Information Processing Systems | **conference** | `conf/nips/SunSLYZL25`（booktitle=NeurIPS, 2025） | — | "**Accepted by NeurIPS 2025**" | **NeurIPS 2025 (peer-reviewed)** | C1+C4+C5(`NeurIPS.cc/2025/Conference`)+C3 四路一致 |
| 2310.04680 | The Cost of Down-Scaling Language Models: Fact Recall Deteriorates before In-Context Learning | arXiv.org | None | **`conf/iclr/JinCDNCRD24`**（booktitle=ICLR, 2024；**标题已改**） | **"The Twelfth International Conference on Learning Representations (ICLR), 2024"** | — | **ICLR 2024 (peer-reviewed)** | ⚠️ 曾冲突：C1/C5/标题-C4 均只见 CoRR，但 C3 journal_ref 自述 ICLR 2024。**作者集 C4** 命中 `conf/iclr/JinCDNCRD24`，7 位作者(Jin/Clement/Dong/Nagarajan/Carbin/Ragan-Kelley/Dziugaite)完全一致，正式标题改为 "The Cost of **Scaling Down** … Affects **Memory** before In-context Learning"，`dblp.org/rec` 内 openreview forum id=`ldJXXxPE0L` → 裁定录用（详见 §0.2） |
| 2506.00288 | Emergent Abilities of Large Language Models under Continued Pretraining for Language Adaptation | Annual Meeting of the Association for Computational Linguistics | **conference** | `conf/acl/ElhadyAA25`（booktitle=**ACL (1)**, 2025 → main） | — | "Published as a Conference Paper at the **main track of ACL 2025**" | **ACL 2025 main (peer-reviewed)** | C1 `type=conference` + C4 `ACL (1)`=main + C3 自述一致（C5 只有 ARR Submission，不作为依据） |
| 2407.17467 | CMR Scaling Law: Predicting Critical Mixture Ratios for Continual Pre-training of Language Models | Conference on Empirical Methods in Natural Language Processing | **conference** | `conf/emnlp/GuYDZT24`（booktitle=**EMNLP**, 2024 → main） | — | "**EMNLP 2024 main conference**" | **EMNLP 2024 main (peer-reviewed)** | C1+C4+C3 三路一致 |
| 2403.17887 | The Unreasonable Ineffectiveness of the Deeper Layers | International Conference on Learning Representations (match) | **conference** | `conf/iclr/GromovTSGR25`（booktitle=ICLR, **2025**） | `MIT-CTP/5694`（是 preprint 编号，非 venue） | "v2: **ICLR camera-ready** version" | **ICLR 2025 (peer-reviewed)** | C2+C4+C5(`ICLR.cc/2025/Conference` Poster)+C3 一致。注：S2 `year=2024` 是 arXiv 年份，**正式年份取 DBLP/OpenReview 的 2025** |
| 2403.03853 | ShortGPT: Layers in Large Language Models are More Redundant Than You Expect | Annual Meeting of the Association for Computational Linguistics | **conference** | `conf/acl/MenXZYWL0HC25`（booktitle=**ACL (Findings)**, 2025） | — | — | **ACL 2025 Findings (peer-reviewed)** | C1 `type=conference` + C4 booktitle 明确为 **Findings（不是 main）**；C5 另有 `ICLR.cc/2025/Conference/Rejected_Submission`（ICLR 被拒→改投 ACL）。⚠️ 引用时勿写成 "ACL 2025 main" |
| 2402.02834 | Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods | arXiv.org (match, **须用 v1 标题**) | None | corr-only | — | "Update (arXiv-v2): continued pretraining for severe pruning ratios … **Preliminary work (arXiv-v1) accepted at ICLR 2024 Workshop on ME-FoMo**" | **preprint**（仅 v1 的 preliminary 版进过 ICLR 2024 **workshop**；当前 v2 无正式录用） | C2 200 `publicationVenue=arXiv.org`（v2 标题 404，v1 标题命中）；C4 corr-only；C5 无录用 note；C3 自述仅 workshop 且限于 v1 |
| 2304.01373 | Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling | International Conference on Machine Learning | **conference** | `conf/icml/BidermanSABOHKP23`（booktitle=ICML, 2023） | — | "Code at …" | **ICML 2023 (peer-reviewed)** | C1+C4+C5(`ICML.cc/2023/Conference` OralPoster) 三路一致 |
| 2312.12141 | Neuron-Level Knowledge Attribution in Large Language Models | Conference on Empirical Methods in Natural Language Processing | **conference** | `conf/emnlp/YuA24`（booktitle=**EMNLP**, 2024 → main） | — | "**Accepted by EMNLP 2024 main.**" | **EMNLP 2024 main (peer-reviewed)** | C1+C4+C3 三路一致。注：S2 `year=2023` 为 arXiv 年份，正式年份 2024 |
| 2601.13580 | Neural Organ Transplantation (NOT): Checkpoint-Based Modular Adaptation for Transformer Models | arXiv.org | None | corr-only | — | "27 pages, 8 figures, 16 tables. Decoder-only transformers (124M-20B parameters)…"（**未自述 venue**） | **preprint** | C1 200 `publicationVenue=arXiv.org`；C4 corr-only；C5 该题无 note（OR 对全标题查询返 400，改用 "Neural Organ Transplantation Checkpoint" 重查，返回的全是无关的器官移植医学文献） |
| 2506.11389 | Curriculum-Guided Layer Scaling for Language Model Pretraining | arXiv.org | None | corr-only | — | "**Accepted to ICML 2026.** Code available at …" | **ICML 2026 (peer-reviewed)** | C5 `venueid=ICML.cc/2026/Conference` + `Submission6733/-/Camera_Ready_Revision`；C3 自述一致；C5 另有 `ICLR.cc/2026/Conference/Rejected_Submission`（先 ICLR 被拒后 ICML 录用）；S2/DBLP 2026 滞后 |
| 2509.06518 | Crown, Frame, Reverse: Layer-Wise Scaling Variants for LLM Pre-Training | arXiv.org | None | corr-only | — | ⚠️ "**The reported results are skewed due to a data type mismatch.** The dataset was saved with int32, but the data loader interpreted it as uint16 … every other token is zero" | **preprint** | C1 200 `publicationVenue=arXiv.org`；C4 corr-only；C5 无 note。**额外提醒 MAIN：作者自己在 COMMENT 里声明结果有 bug 不可用，引用其数字须谨慎** |
| 2502.13794 | LESA: Learnable LLM Layer Scaling-Up | Annual Meeting of the Association for Computational Linguistics | **conference** | `conf/acl/YangCMYC0Z25`（booktitle=**ACL (1)**, 2025 → main） | — | — | **ACL 2025 main (peer-reviewed)** | C1 `type=conference` + C4 `ACL (1)`=main（C5 只有 ARR Submission，不作依据） |
| 2508.08011 | Progressive Depth Up-scaling via Optimal Transport | arXiv.org (match) | None | corr-only | — | — | **preprint** | C2 200 `publicationVenue=arXiv.org`；C4 corr-only；C5 仅 `TMLR/**Rejected**`（被拒，非录用） |
| 2402.05913 | Efficient Stagewise Pretraining via Progressive Subnetworks | International Conference on Learning Representations (match) | **conference** | `conf/iclr/PanigrahiSLMRKK25`（booktitle=ICLR, **2025**） | — | — | **ICLR 2025 (peer-reviewed)** | C2+C4+C5(`ICLR.cc/2025/Conference` Poster) 一致；C5 另有 `ICLR.cc/2024/Conference/Rejected_Submission`（2024 被拒 2025 录用）。S2 `year=2024`=arXiv 年份，正式年份 2025 |
| 2511.03270 | SCALE: Upscaled Continual Learning of Large Language Models | Annual Meeting of the Association for Computational Linguistics (match) | **conference** | `conf/acl/LeeCHCKYLJPPJ26`（booktitle=**ACL (Findings)**, **2026**） | — | — | **ACL 2026 Findings (peer-reviewed)** | C2 `type=conference` + C4 booktitle=**Findings**。⚠️ 勿写成 ACL main。S2 `year=2025`=arXiv 年份，正式年份 2026 |
| 2509.01213 | Mitigating Catastrophic Forgetting in Continual Learning through Model Growth | arXiv.org | None | corr-only | — | — | **preprint** | C1 200 `publicationVenue=arXiv.org`；C4 corr-only；C5 无 note。（Crossref 曾返回 `10.62441/nano-ntp.v20i6.34` "…for Natural Language Processing Tasks" in *Nanotechnology Perceptions* → **经比对是不同论文，false positive，已排除**） |
| 2505.20155 | Pangu Light: Weight Re-Initialization for Pruning and Accelerating LLMs | arXiv.org (match) | None | corr-only | — | — | **preprint** | C2 200 `publicationVenue=arXiv.org`；C4 corr-only；C5 无 note |
| 2307.01163 | Improving Language Plasticity via Pretraining with Active Forgetting | Neural Information Processing Systems (match) | **conference** | `conf/nips/ChenMRAS0A23`（booktitle=NeurIPS, 2023） | — | "**NeurIPS 2023 Final Version**" | **NeurIPS 2023 (peer-reviewed)** | C2+C4+C5(`NeurIPS.cc/2023/Conference` poster)+C3 四路一致 |
| 2006.05987 | Revisiting Few-sample BERT Fine-tuning | International Conference on Learning Representations | **conference** | `conf/iclr/0007WKWA21`（booktitle=ICLR, **2021**） | — | "Code available at …" | **ICLR 2021 (peer-reviewed)** | C1 `type=conference` + C4 `conf/iclr` key。S2 `year=2020`=arXiv 年份，正式年份 2021 |
| 2004.14975 | Investigating Transferability in Pretrained Language Models | **Findings** | **journal** | `conf/emnlp/TamkinSGG20`（booktitle=**EMNLP (Findings)**, 2020） | — | "**Findings of EMNLP 2020**" | **EMNLP 2020 Findings (peer-reviewed)** | C1 `venue='Findings'`/`type=journal`（S2 把 Findings 归为 journal，属其分类习惯）+ C4 booktitle=**Findings** + C3 自述一致 |
| 2410.06225 | A Timeline and Analysis for Representation Plasticity in Large Language Models | arXiv.org | None | corr-only | — | — | **preprint** | C1 200 `publicationVenue=arXiv.org`；C4 corr-only；C5 无 note |
| 2410.11654 | Transformer Layer Injection: A Novel Approach for Efficient Upscaling of Large Language Models | arXiv.org | None | corr-only | — | — | **preprint** | C1 200 `publicationVenue=arXiv.org`；C4 corr-only；C5 无 note |
| 2312.15166 | SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling | North American Chapter of the ACL | **conference** | `conf/naacl/KimKPLSKKKLKAYLPGCLK24`（booktitle=**NAACL (Industry Track)**, 2024） | — | "accepted to **NAACL 2024 Industry Track**" | **NAACL 2024 Industry Track (peer-reviewed)** | C1 `type=conference` + C4 booktitle=**Industry Track** + C3 自述一致。⚠️ 勿写成 NAACL main |
| 2401.02415 | LLaMA Pro: Progressive LLaMA with Block Expansion | Annual Meeting of the ACL | **conference** | `conf/acl/WuGGLWFSL24`（booktitle=**ACL (1)**, 2024 → main） | — | "**Accepted by ACL 2024, Main Conference**" | **ACL 2024 main (peer-reviewed)** | C1+C4+C3 三路一致 |
| 2410.02330 | Llama SLayer 8B: Shallow Layers Hold the Key to Knowledge Injection | Conference on EMNLP (match) | **conference** | `conf/emnlp/ChenTGW00YY24`（booktitle=**EMNLP (Findings)**, 2024） | — | — | **EMNLP 2024 Findings (peer-reviewed)** | C2 `type=conference` + C4 booktitle=**Findings**。⚠️ 勿写成 EMNLP main |
| 2403.19135 | Streamlining Redundant Layers to Compress Large Language Models | `''` | None | `conf/iclr/ChenHZWL025`（booktitle=ICLR, 2025） | — | — | **ICLR 2025 Spotlight (peer-reviewed)** | C4 `conf/iclr` key + C5 `venueid=ICLR.cc/2025/Conference`（`venue='ICLR 2025 Spotlight'`）。注：C1 `venue=''` 且 `publicationVenue=None` —— **S2 单路会误判为 preprint 的典型案例** |
| 2407.16286 | A deeper look at depth pruning of LLMs | arXiv.org (match) | None | corr-only | — | — | **preprint**（有 ICML 2024 workshop poster） | C2 200 `publicationVenue=arXiv.org`；C4 corr-only；C5 仅 `ICML.cc/2024/**Workshop**/TF2M` Poster，主会无录用 |
| 2210.10041 | Hidden State Variability of Pretrained Language Models Can Guide Computation Reduction for Transfer Learning | Conference on EMNLP | **conference** | `conf/emnlp/XieQPDQM22`（booktitle=**EMNLP (Findings)**, 2022） | — | "**EMNLP 2022 camera-ready**" | **EMNLP 2022 Findings (peer-reviewed)** | C1 `type=conference` + C4 booktitle=**Findings** + C3 自述 camera-ready。⚠️ 勿写成 EMNLP main |
| 2403.17919 | LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning | Neural Information Processing Systems | **conference** | `conf/nips/PanLDPZH024`（booktitle=NeurIPS, 2024） | — | "**NeurIPS 2024**" | **NeurIPS 2024 (peer-reviewed)** | C1+C4+C5(`NeurIPS.cc/2024/Conference` poster)+C3 四路一致 |
| 2406.11753 | A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models | Annual Meeting of the ACL | **conference** | `conf/acl/0001A0Z25`（booktitle=**ACL (Findings)**, 2025） | — | "**accepted by ACL 2025**, the camera-ready version" | **ACL 2025 Findings (peer-reviewed)** | C1 `type=conference` + C4 booktitle=**Findings**（作者 COMMENT 只说 "ACL 2025" 未提 Findings）。⚠️ 以 DBLP 为准，勿写 main |
| 2505.23811 | LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions | Neural Information Processing Systems (match) | **conference** | `conf/nips/AskariGWCC25`（booktitle=NeurIPS, 2025） | — | "Neurips 2025" | **NeurIPS 2025 (peer-reviewed)** | C2+C4+C5(`NeurIPS.cc/2025/Conference` poster)+C3 四路一致 |
| 2510.10071 | ADEPT: Continual Pretraining via Adaptive Expansion and Dynamic Decoupled Tuning | arXiv.org | None | corr-only | — | — | **ICLR 2026 Poster (peer-reviewed)** | C5 `venueid=ICLR.cc/2026/Conference` + `Submission2077/-/Camera_Ready_Revision`；S2/DBLP 2026 滞后（**又一个"S2 说 arXiv.org 但其实已录用"的案例**） |
| 2601.20009 | LinguaMap: Which Layers of LLMs Speak Your Language and How to Tune Them? | arXiv.org | None | corr-only | — | — | **ICLR 2026 Poster (peer-reviewed)** | C5 `venueid=ICLR.cc/2026/Conference` + `Submission10410/-/Camera_Ready_Revision`；S2/DBLP 2026 滞后 |
| 2404.07066 | Exploring Concept Depth: How Large Language Models Acquire Knowledge and Concept at Different Layers? | `''` | None | `conf/coling/JinYHZWH0MMDYDZ25`（booktitle=COLING, 2025） | — | "**COLING 2025**" | **COLING 2025 (peer-reviewed)** | C4 `conf/coling` key + C3 自述一致。C1 `venue=''` → 又一个 S2 单路会漏判的案例 |

---

## 3. 汇总统计

计数方式：脚本解析 §2 主表的 49 行、取第 8 列"最终判定"自动归类（不是手数），核对 **32 + 17 + 0 = 49 ✓**。

| 判定 | 数量 | ID |
|---|---|---|
| peer-reviewed | **32** | 2411.15558, 2605.02105, 2602.11137, 2602.14486, 2503.04429, 2109.08406, 2502.05795, 2310.04680, 2506.00288, 2407.17467, 2403.17887, 2403.03853, 2304.01373, 2312.12141, 2506.11389, 2502.13794, 2402.05913, 2511.03270, 2307.01163, 2006.05987, 2004.14975, 2312.15166, 2401.02415, 2410.02330, 2403.19135, 2210.10041, 2403.17919, 2406.11753, 2505.23811, 2510.10071, 2601.20009, 2404.07066 |
| preprint | **17** | 2606.07978, 2606.16897, 2607.25663, 2510.18871, 2605.11416, 2606.09932, 2410.06981, 2312.02730, 2402.02834, 2601.13580, 2509.06518, 2508.08011, 2509.01213, 2505.20155, 2410.06225, 2410.11654, 2407.16286 |
| UNRESOLVED | **0** | — |

其中 peer-reviewed 32 篇按强度再分层（便于写 related work 时不高估）。核对 **24 + 6 + 1 + 1 = 32 ✓**：

| 层级 | 数量 | ID |
|---|---|---|
| 主会 / 主 track（ICLR / ICML / NeurIPS / ACL(1) / EMNLP main / COLING） | **24** | 2411.15558, 2605.02105, 2602.11137, 2602.14486, 2503.04429, 2502.05795, 2310.04680, 2506.00288, 2407.17467, 2403.17887, 2304.01373, 2312.12141, 2506.11389, 2502.13794, 2402.05913, 2307.01163, 2006.05987, 2401.02415, 2403.19135, 2403.17919, 2505.23811, 2510.10071, 2601.20009, 2404.07066 |
| Findings | **6** | 2403.03853, 2511.03270, 2406.11753, 2410.02330, 2210.10041, 2004.14975 |
| Industry Track | **1** | 2312.15166 |
| Workshop 但有 DBLP `conf/` key | **1** | 2109.08406 |

---

## 4. 给 MAIN 的落账要点（只列事实，不做方向判断）

1. **`2411.15558` 必须改口为 ICLR 2026 Poster (peer-reviewed)**；MAIN 2026-08-05 的 "S2 → arXiv.org → preprint" 结论作废。
   同类需改口的还有 **`2510.10071`、`2601.20009`（ICLR 2026 Poster）** 与
   **`2605.02105`、`2602.11137`、`2602.14486`、`2506.11389`（ICML 2026）** —— 这 7 篇 S2/DBLP 全部还写着 `arXiv.org`/corr-only。
2. **`2411.15558` 训的是"剩余模型的最后 1–3 个预训练层 + lm_head"，不是随机初始化新层**（§1.2 三重证据：
   §4.2 定义原文、Abstract "the remaining last three layer"、全文 0 命中 regrow/random-init 词表）。
3. **Findings / Industry Track / Workshop 必须写清**，否则会高估强度：
   - ACL **Findings**：`2403.03853`(ShortGPT)、`2511.03270`(SCALE)、`2406.11753`
   - EMNLP **Findings**：`2410.02330`、`2210.10041`、`2004.14975`
   - NAACL **Industry Track**：`2312.15166`(SOLAR)
   - 仅 **workshop**（判为 preprint）：`2312.02730`(UniReps@NeurIPS'23)、`2407.16286`(TF2M@ICML'24)、
     `2402.02834`(ME-FoMo@ICLR'24，且只是 v1 preliminary)、`2510.18871`(KnowFM@ACL'26)
   - `2109.08406` 是 BlackboxNLP workshop，但有 DBLP `conf/` key → 本报告计为 peer-reviewed（若 MAIN 想统一"只算主会"，此行需降级，请显式决定）
4. **正式年份 ≠ arXiv 年份**，S2 的 `year` 常是 arXiv 年：`2403.17887`(→ICLR 2025)、`2402.05913`(→ICLR 2025)、
   `2312.12141`(→EMNLP 2024)、`2006.05987`(→ICLR 2021)、`2511.03270`(→ACL 2026)、`2312.15166`(→NAACL 2024)。
5. **`2509.06518` 作者自述结果有 data-type bug（int32/uint16 误读，每隔一个 token 为 0）**，其数字不宜引用。
6. **方法学入库建议**（避免第三次被 venue 坑）：
   (a) venue 核实优先 `S2 search/match`（配额独立于 `paper/arXiv:`）；
   (b) **2026 年会议一律查 OpenReview `venueid` + `Camera_Ready_Revision`**，S2/DBLP 有滞后；
   (c) `venueid` 里含 `/Submission`、`Withdrawn_`、`Rejected_`、`TMLR/Rejected` 的**不算录用**；
   (d) 标题检索无果时补**作者集 DBLP 检索**（camera-ready 改名会漏，实测 2 例）；
   (e) 精确 main/Findings 只能靠 `dblp.org/rec/<key>.xml` 的 `booktitle`，S2 的 `venue` 字段不区分。

---

## 5. 未能做到 / 边界声明

- **Crossref** 我只对 8 个无证据行做了抽查（作为第 6 路），未全量跑；其中唯一命中经比对是 false positive（见 `2509.01213` 行）。
  因此本报告的 preprint 结论建立在 C1/C2(S2)+C4(DBLP)+C5(OpenReview)+C3(arXiv 自述) 之上，**未穷尽 Crossref/Google Scholar**。
- `2601.13580` 的 OpenReview 查询对完整标题返 HTTP 400，退化用关键词查询，返回结果全部无关；
  故该行的"C5 无 note"是**弱证据**（不排除 OR 上存在但检索没命中）。
- 我**没有**修改任何 `.tex` / `status/*.md` / `versions/*.md` / `*TODOList*`，也没有跑 GPU。
- 本报告**不含**任何"该不该做这个方向"的判断，也**不含**"无人做过 X"这类指控 —— 那是 AUDIT1+ 的任务。
