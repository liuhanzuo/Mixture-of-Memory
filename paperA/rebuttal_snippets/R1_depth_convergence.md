# R1 — 「三方法论独立收敛」rebuttal 弹药（depth structure 非 probe artifact）

**任务**：为 paperA `tab_depth.tex` 自认的软处（caption 原文 "These correlational probes do not
localize understanding or establish the same knees on long-memory tasks."；`03_motivation.tex` 原文
"Figure~\ref{fig:depth} supplies motivation, while the matched experiment ... supplies the actual
evidence."）准备答复弹药。

**抓取状态**：三篇全文均抓到（`arxiv.org/html/<id>`，51.8k / 72.0k / 90.0k chars 纯文本），
含 appendix 与 table 数值。引用句均自全文读到，标 section。

**venue 核实（Semantic Scholar Graph API，全部重试到 http200）**：

| id | S2 `venue` / `publicationVenue` | 判定 |
|---|---|---|
| 2606.07978 | `venue:""`, `publicationVenue:null` | **arXiv preprint**（arXiv 无 `Journal ref` / 无 `Comments`；正文脚注自述 code repo 名含 `MechLens-EMNLP2026`，属**投稿意向而非 acceptance**，不得当 peer-reviewed） |
| 2510.18871 | `venue:"arXiv.org"`, `publicationVenue.name:"arXiv.org"`, DBLP `journals/corr/abs-2510-18871` | **arXiv preprint**（HTML 里 "Machine Learning, ICML" 是 ICML LaTeX 模板的 keywords 行，**不是** accepted 标记；S2 venue 明确是 arXiv.org） |
| 2607.25663 | `venue:""`, `publicationVenue:null` | **arXiv preprint**（arXiv `Comments` 只写 "Main text: 8 pages, 2 figures; appendix: 13 tables, 10 figures; code and data available at..."，无 venue 自述） |

→ **三篇全部是 preprint。** rebuttal 里必须写 "concurrent preprints"，不能暗示 peer-reviewed。

---

## 1. 三篇方法 + 与我们 tab_depth 数字的对应关系

### 我们的数字（复核过磁盘 source，不是从 tex 抄的）

- `paperA/sections/tab_depth.tex`：linear knee（frozen linear probe 达到 held-out peak 98% 的首层）
  Qwen3-8B **0.393** [0.319,0.466] / Llama-3-8B **0.275** / OLMo-2-7B **0.285**；
  native knee（自身 final norm + LM head 逐层）**0.824 / 1.000 / 0.875**；gap **0.43 / 0.725 / 0.59**。
  磁盘 source = `results/p1_2/p1_2_summary.json`（`content_j_frac_mean` 0.3926 / 0.2688 / 0.2854，
  per-task `native_knee_frac`：Qwen RTE 0.9444 + SST2 0.6389 + WiC 0.8889 → 均值 0.824 ✓；
  OLMo 1.0 + 0.75 + 0.875 → 0.875 ✓）。**注意 native knee 是 SST2/WiC/RTE 三个非事实任务的均值。**
- `results/knowledge_logit_lens_{OLMo-2-1124-7B,Qwen3-8b-local}.json`（MMLU logit-lens，n=1000，
  hidden@last-prompt-pos → final_norm → lm_head，argmax over {A,B,C,D}）：
  OLMo-2-7B `onset_layer 18` (0.562L)、L18→L19 由 **0.326 → 0.544**、`sat99_frac_depth 0.844`、peak 0.551@L32；
  Qwen3-8B `onset_layer 25` (0.694L)、L24→L25 由 **0.236 → 0.621**、`sat99_frac_depth 0.778`、peak 0.638@L34。
- `paperA/sections/tab_distilled_depth_curve.tex`：RULER 98.29 (j=6) → 96.07 (j=12) → **55.41 (j=18)**；
  Read speedup 1.166× → 1.403× → 1.807×（Qwen3-8B，L=36）。

### 对照表

| 维度 | 我们（Paper A） | MechLens `2606.07978`（preprint） | Gupta et al. `2510.18871`（preprint） | Ramnauth & Scassellati `2607.25663`（preprint） |
|---|---|---|---|---|
| **测量手段** | frozen linear probe（SST2/WiC/RTE，5 splits）+ native LM-head readout | logit lens FEP（answer 首次进 top-10 的层）+ **tuned lens 交叉验证** + LayerNorm ablation + per-head ablation | **TunedLens** rank-tracking + **activation patching**（因果）+ early-exit | **localized LoRA**（early/middle/late 各 quarter-depth 窗口）+ **两种 parameter-matched control** |
| **是否 probe-free 的因果腿** | 无（这正是被打的点）；因果腿在 matched j=0 vs j=12 | 半：tuned lens 只是"更好的 probe"；LN ablation 与 head ablation 是 intervention | **有**：SST→MMLU 的 activation patch，原文 §4.1.1 明确 "this activation replacement experiment is done independent of the TunedLens probe" | **有**：完全不用 readout probe，靠"在哪训得动"来定位；且 loss 侧不是解码侧 |
| **"浅层已可线性访问"对应物** | linear knee 0.275–0.393L | 无直接对应（它只测 top-10 native rank） | 功能词/PUNCT rank-1 平均 **~layer 5**（Pythia-6.9B & Llama3-8B，§4.3）；MCQ 的 option-collection "usually happens within the first half of the model"（§4.1） | **lexical binding 偏 early-quarter**：early 的 acquisition 高于 late **20.9 pp** [17.2,23.6]、高于 middle **10.8 pp** [7.6,13.2]；boundedness 高于 full-stack **28.3 pp**（§Lexical Binding） |
| **"native/知识 readout 很深"对应物** | native knee 0.824 / 0.875 / 1.000；MMLU logit-lens sat99 0.844L (OLMo) / 0.778L (Qwen) | **FEP Depth 全部 >80% depth**：Qwen2.5-7B 97.5%、Qwen2.5-14B 95.8%、Llama-3.1-8B 91.9%、Mistral-7B 82.3%、Pythia-6.9B 96.2%、Gemma-7B 97.7%（Table 6；正文写 range "82.2%–97.7%"）；MMLU 上 **98.2%** 的正确答案在任何中间层都没进 top-10（§7.3 / Table 17） | 单 token fact 首现 **layer 15/32（Pythia-6.9B）、layer 20/32（Llama3-8B）**；multi-token fact 的**首 token 两个模型都 layer 25**；3-token fact 首 token Pythia **~layer 27**，第 2/3 token 反而 ~20 / ~12（§4.2） | **factual association 偏 late**：late 比 early 的 transfer 高 **42.7 pp** [35.3,47.3]、acquisition 高 **19.7 pp** [18.0,21.6]；early 绝对值很差（acq 32.4% / transfer 20.4%），full-stack 最好（89.6 / 82.0）（§Factual Association） |
| **probe-artifact 排除做法** | 有 lexical-only / position-only / random-label / majority 四条地板（`p1_2_summary.json` 里都在） | tuned lens：2,000 WikiText-2 样本训 per-layer affine probe，817 TruthfulQA 上得 **85.7% vs logit lens 85.9%（Δ=0.2pp）**，74.9% 样本 FEP 完全相同（§6.1 Table 4）。另三条：LN ablation FEP 分布不变（27.30 / 85.9% 完全相同，§7.5 Table 8）；跨架构梯度与 attention 机制相关；crystallization 度能预测最优 intervention | §5：比较 TunedLens 各层 high-freq token 概率质量 vs 末层；重训 custom TunedLens 把最高频 token 更新频率**降 1000 倍**，早层 top-1 仍被它主导 → 结论 "not a consequence of probe bias but rather a reflection of the information content in the early layer representations" | 两个 total-parameter-matched control：**localized-expanded-rank**（localized 升 rank 到 32×8=256 对齐 full 8×32=256）与 **full-reduced-rank**（full 降到 2×32=64 对齐 localized 8×8=64）。Table 11 结论 "the main adaptation geometries are not explained by trainable parameter count alone" |
| **模型广度** | 3 家族（Qwen3-8B / Llama-3-8B / OLMo-2-7B），base 口径 | 5 家族 6 checkpoint：Pythia-6.9B, Gemma-7B, Qwen2.5-7B, Qwen2.5-14B, Llama-3.1-8B, Mistral-7B（base；另有 Qwen2.5-7B-Instruct pilot） | 4 模型：GPT2-XL, Pythia-6.9B, Llama2-7B, Llama3-8B（用原作者发布的 TunedLens probe） | 5 家族：Llama-3.1-8B（主）+ Mistral-7B, Gemma-2-9B, OLMo-2-7B, Qwen2.5-14B；**原文自述均为 instruction-tuned** |
| **任务** | SST2/WiC/RTE（linear+native）、MMLU（logit-lens）、RULER/LoCoMo（真实证据腿） | TruthfulQA (817)、MMLU (1200/24 subj)、SST-2 control (200) | Wikipedia NTP、MMLU/SST/NLI/MRPC (4-shot)、MQuAKE fact recall、POS | 自建合成 benchmark，5 objective × 25 latent spec |

### 三点真正的收敛（可直接引用）

1. **"native / 知识型 readout 出现在 >0.8L"** —— 我们 native knee 0.824/0.875/1.000 与 MMLU
   logit-lens sat99 0.844L/0.778L；MechLens FEP Depth 82.3–97.7%（6 checkpoint 全部 >80%）、
   MMLU 98.2%；Gupta multi-token fact 首 token layer 25/32 = 0.78L。**三者区间重叠。**
2. **"浅层已足够支撑一部分任务"** —— 我们 linear knee 0.275–0.393L；Gupta 的 option-collection
   在前半、功能词 ~layer 5；Ramnauth 的 lexical binding 在 early quarter 显著优于 late（20.9 pp）。
3. **"probe artifact 不是解释"** —— 我们四条常量/退化地板；MechLens tuned lens Δ=0.2pp + LN ablation
   零变化；Gupta 的 activation patching 明确 probe-independent；Ramnauth 完全不用 readout probe。

---

## 2. 诚实的不一致 / 我们过度解读处（必须自己先说，别等 reviewer 说）

**(a) MechLens 的 factual-specificity control 与我们 tab_depth 的任务选择相冲突。**
MechLens §7.7 Table 10：Qwen2.5-7B 在 SST-2 上 late crystallization 只有 **0.5%**（factual 84.9%，
170× gap），Mistral **2.0% vs 26.8%**，结论是 "Late Crystallization is specific to factual knowledge
retrieval rather than a generic property of transformer predictions"。
**但我们 tab_depth 的 native knee 恰恰是 SST2/WiC/RTE 三个非事实任务的均值（Qwen 0.824 = RTE 0.944
+ SST2 0.639 + WiC 0.889）。** 如果按 MechLens 的口径，非事实任务本该早就 native 可读。
→ 二者口径不同（MechLens FEP = 正确答案进 top-10 的**首层**；我们 native knee = native argmax 达
peak 98% 的**首层**，是"饱和"而非"首现"；模型也不同 Qwen2.5 vs Qwen3），**不能直接横比**。
→ 结论：**MechLens 只应用来支持我们的 MMLU knowledge-logit-lens 腿（0.778–0.844L），不要拿它去
背书 SST2/WiC/RTE 的 native knee。** rebuttal 措辞里我只用 "brackets our native knees" 这种区间语言，
并且主证据换成 MMLU 那条。若 reviewer 抓这点，正面承认：我们的 native knee 是 saturation 定义，
且是三任务均值，MechLens 的 first-appearance 定义在非事实任务上更早。

**(b) Gupta et al. 有一句可以被反向引用来打我们。**
§5 原文：早层 top-1 被高频 token 主导 "is not a consequence of probe bias but rather a reflection of
the **information content in the early layer representations**"；§3.1 更直白 "In early layers, the
model has incomplete contextual information ... the model is also unable to access the factual
knowledge stored in its parameters"；§3.2 "approximately 60–80% of early top-ranked guesses are
eventually replaced"。
→ 字面读，这是在说**早层信息不足**，与"复用前 j 层做 prepaid depth"张力明显。
→ 我们的反驳（诚实版）：他们测的是**vocabulary-space 的 top-1**，我们测的是**trained readout 能否
用上**；两者不矛盾（正是我们 linear-vs-native gap 的本体）。**但必须承认这不构成"低层信息已足够"的
正面证据**——它只说明 native 解码晚，不说明 trained 解码早。真正支持"trained 解码早"的是我们自己的
linear probe + `tab_h12_oracle` 的 continuous-prefix oracle，以及 Ramnauth 的 lexical binding。

**(c) Gupta 的 content-word 深度（~layer 20/32 = 0.625L）比我们的 linear knee（0.275–0.393L）深得多。**
不能说"数字一致"。差异来自 probe 类型（TunedLens native-ish vs 我们 trained linear on task label）
与目标（rank-1 of predicted token vs task-label accuracy）。措辞上只能说"同向"，不能说"同值"。

**(d) Gupta 的 transition-layer 定义在 Llama3-8B 上失效（自述）。**
Appendix G 原文："it does not prediction the transition in Llama3-8B as shown in Figure 19(c)"，原因是
中层 top choice 出现不在选项集里的 'neutral'、以及大小写变体。→ 他们的 probe-based transition 与因果
patching 只在 GPT2-XL / Pythia-6.9B / Llama2-7B 三个模型上对齐，**Llama3-8B 不对齐**。所以"因果与
probe 完全一致"是过度解读，只能说"在 3/4 模型上一致"。

**(e) Ramnauth 的三处限制正好卡我们最在意的两条铁律。**
- 他们明确用 **instruction-tuned** 模型（§Cross-Model Robustness Analysis 原文 "open-weight,
  instruction-tuned transformers"）。我们全论文是 **chat_template=False / base 口径**（项目铁律）。
  → 跨口径引用必须标注，不能当作 base-model 证据。
- benchmark 是**合成**的（Limitations 第 1 条自述 "the benchmark is synthetic"），不是自然长文。
- **model identity 解释的 profile 方差（34.2%）大于 objective identity（25.0%）**，residual 40.8%；
  他们自己写 "These results support directional replication ... but not a strong
  architecture-invariant account of localization geometry"。
  → 这直接削弱"深度结构跨家族不变"的强版本。我们 tab_depth 三家族 linear knee 0.275/0.285/0.393、
  native knee 0.824/0.875/1.000 本身也是家族间散得开的，**所以我们也不该主张 architecture-invariant**。
- 只有 3 个 seed，作者自述 p 值 "descriptive"（Table 8 caption "Given only three seeds, we treat these
  p-values as descriptive"）。→ 引用 42.7 pp / 20.9 pp 时应带 CI，不引 p。

**(f) 无一篇与我们结论相反。** 三篇里没有任何一篇给出"深度是均匀的 / 浅层与深层功能无差别"的相反结论。
最接近"相反"的是 (b) Gupta 的早层信息不足表述，但那是**关于 native 解码**而非关于 trained readout，
属"不支持"而非"反证"。**同时也没有任何一篇覆盖 long-memory / long-context 任务** ——
MechLens 是 TruthfulQA/MMLU/SST-2，Gupta 是 Wikipedia NTP + 4-shot MCQ + MQuAKE + POS，
Ramnauth 是合成 5-objective。→ **tab_depth caption 里 "do not ... establish the same knees on
long-memory tasks" 这句免责声明必须保留，不能因为这三篇就删掉。**

**(g) MechLens 自身数字内部不一致，引用时避开。**
Mistral 的 late-crystallization 在 abstract/§7.1/§8 写 **26.8%**，但 Table 6 与 Conclusion 写 **27.1%**；
Llama 在 §1/§7.4 写 **70.4%**，Table 6 与 §6.1/§7.1 写 **71.0%**；跨架构 FEP-Depth 下界正文写
**82.2%** 而 Table 6 是 **82.3%**。
→ rebuttal 只引它**稳定**的三个数：tuned-lens Δ=**0.2 pp**（85.7 vs 85.9）、FEP Depth 一律 **>80% depth**、
MMLU **98.2%**。不要引 26.8/27.1 这类会被 reviewer 拿去核的摇摆数字。

---

## 3. LaTeX rebuttal 段落（可直接贴 response letter；228 词）

```latex
\paragraph{R1: the depth probes are correlational.}
We agree, and we keep that disclaimer. We add that three concurrent preprints reach
the same depth structure using methods that do not share our probe's failure modes.
(i) \citet{mechlens2026} replace the logit lens with a tuned lens trained on 2{,}000
WikiText-2 samples and reproduce their late-readout rate to within $0.2$ points
($85.7\%$ vs.\ $85.9\%$); across six checkpoints from five families, mean factual
emergence lies above $80\%$ of depth ($98.2\%$ of MMLU answers never enter top-10
at any intermediate layer), which brackets our MMLU logit-lens saturation at
$0.78$--$0.84L$ and our native knees at $0.82$--$1.00L$.
(ii) \citet{gupta2026depth} validate a lens-defined phase transition with activation
patching, which they state is measured independently of the probe: patching before
the transition still lets the model recover the target task's options; patching after
it does not. Their option-collection phase ends within the first half of depth, which
is where our frozen interface degrades ($96.07$ RULER at $j{=}12$ vs.\ $55.41$ at
$j{=}18$ of $36$ layers).
(iii) \citet{ramnauth2026localized} use no readout probe at all: localized LoRA with
two parameter-matched controls shows lexical binding favouring early layers and
factual association favouring late ones (late-over-early transfer $+42.7$\,pp, 95\%
CI $[35.3,47.3]$, replicating in $5/5$ families).
None of the three evaluates long-memory tasks, so our disclaimer stands and the
matched $j{=}0$ versus $j{=}12$ experiment remains our evidence; what changes is
that the motivating structure is now corroborated by tuned-lens, causal-intervention,
and training-based methods rather than by frozen probes alone.
```

**用法提示**
- 若审稿人追问 "为什么用 preprint 支撑"：加一句 "All three are concurrent preprints; we cite them
  as methodological triangulation, not as established results."
- 若审稿人抓 §2(a)（MechLens 的 SST-2 control）：按上面 (a) 正面承认口径差异，并把重心挪到
  MMLU logit-lens 那条（0.778/0.844L），**不要**用 MechLens 去背书 SST2/WiC/RTE 的 native knee。
- 若审稿人抓 "cross-family invariance"：引 Ramnauth 自己的 34.2% vs 25.0% 方差分解，说明我们同样
  **不主张** architecture-invariant knee，只主张 "linear precedes native within each family"。
- 段落里刻意没写 "understanding"、"localize"、"causally establishes our knees" 这类会被反打的词。

---

## 4. .bib 条目（venue 已核实；三条全部 preprint，已显式标注）

```bibtex
% arXiv preprint (Semantic Scholar: venue="", publicationVenue=null; no arXiv journal-ref).
% The repo name in the paper's footnote mentions EMNLP2026 but that is a submission
% intent, NOT an acceptance. Cite as preprint.
@article{mechlens2026,
  title={{MechLens}: Late Crystallization of Factual Knowledge Explains Intervention Effectiveness in Language Models},
  author={Gao, Xueping},
  journal={arXiv preprint arXiv:2606.07978},
  year={2026},
  eprint={2606.07978},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  note={Preprint}
}

% arXiv preprint (Semantic Scholar venue="arXiv.org"; DBLP journals/corr/abs-2510-18871).
% The HTML shows "Machine Learning, ICML" -- that is the ICML LaTeX keywords line,
% not a venue. v1 Oct 2025, v2 Mar 2026. Cite as preprint.
@article{gupta2026depth,
  title={How Do {LLMs} Use Their Depth?},
  author={Gupta, Akshat and Yeung, Jay and Anumanchipalli, Gopala and Ivanova, Anna},
  journal={arXiv preprint arXiv:2510.18871},
  year={2025},
  eprint={2510.18871},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2510.18871},
  note={Preprint}
}

% arXiv preprint (Semantic Scholar: venue="", publicationVenue=null; arXiv Comments
% field states page/table counts and a code link only). Cite as preprint.
@article{ramnauth2026localized,
  title={Localized Adaptation Reveals Distinct Learning Signatures in Transformers},
  author={Ramnauth, Rebecca and Scassellati, Brian},
  journal={arXiv preprint arXiv:2607.25663},
  year={2026},
  eprint={2607.25663},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  note={Preprint}
}
```

---

## 5. 检索透明度（关于"是否还有第四篇同向工作"）

我**没有**主张 "无人做过 X"，因此不需要 negative-claim 门槛。为完整性记录：本次只精读了 MAIN 指定的
三篇，未做额外 web 搜索（任务未要求）。三篇内部各自引用的相关先行工作里，与本收敛论证同向、
但**我未抓取全文因此不引用**的有：Belrose et al. 2023 (tuned lens，2303.08112)、
Csordás et al. 2025 "Do language models use their depth efficiently?" (2505.13898)、
Din et al. 2023 "Jump to conclusions" (2303.09435)、Geva et al. 2023 (fact-recall 三步 circuit)、
Tenney/Das/Pavlick 2019 (BERT rediscovers the classical NLP pipeline)。
若需要把 rebuttal 从 "三篇" 扩到 "四—五篇"，**Csordás et al. 2505.13898 是最值得下一步抓的**
（题目与我们的命题几乎正对，且被 Gupta et al. 列在 Related Work）——但我未读其全文，
**现在不得引用其结论**。
