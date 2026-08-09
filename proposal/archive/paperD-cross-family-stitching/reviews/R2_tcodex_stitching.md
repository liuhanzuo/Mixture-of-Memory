# Paper D: literature and feasibility review

**Search cutoff: August 5, 2026.**

## Executive verdict

**The broad idea is already occupied.** You cannot credibly claim to introduce:

- LLM layer stitching;
- frozen lower-model + frozen upper-model composition through a learned connector;
- architecture-level recombination of pretrained Transformer blocks;
- or lightweight learned integration of frozen specialists.

Three papers are especially damaging to a broad novelty claim:

1. **“StitchLLM: Serving LLMs, One Block at a Time”** — DOI **10.18653/v1/2025.acl-long.1305** — already connects frozen decoder blocks from differently sized LLMs using trainable stitching layers. ([aclanthology.org](https://aclanthology.org/2025.acl-long.1305/))
2. **“BTS: Harmonizing Specialized Experts into a Generalist LLM”** — arXiv:**2502.00075**, DOI **10.18653/v1/2025.emnlp-main.347** — already trains lightweight stitches between frozen domain experts and a frozen hub to retain code, math, and multilingual capabilities. ([aclanthology.org](https://aclanthology.org/2025.emnlp-main.347/))
3. **“Building LLMs Like LEGO: Two-dimensional Architecture Reassembly of Large Language Models”** — DOI **10.18653/v1/2026.acl-long.2081** — already presents vertical depth-wise recombination of frozen Transformer blocks from multiple pretrained models, plus lightweight glue layers trained by knowledge distillation. ([aclanthology.org](https://aclanthology.org/2026.acl-long.2081/))

The exact conjunction you propose is **not clearly occupied**:

> independently pretrained, genuinely cross-family donors; one serial A-front → tiny bridge → B-back boundary; possibly different widths and tokenizers; only the bridge trained; and demonstrated retention of complementary end-task capabilities from both donors.

That is still open. But it is open largely because **the evidence for feasibility is weak**, not because nobody thought of stitching LLMs.

My prior is:

- **Same family/shared tokenizer:** likely technically workable, but not novel.
- **Independent families/shared tokenizer:** plausible for recovering perplexity or coarse semantics; doubtful for preserving the stronger donor’s reasoning or knowledge.
- **Different tokenizers:** a conventional one- or two-layer fixed-length bridge is structurally inadequate.
- **“Best of both”:** unlikely without substantially more machinery than a single splice, because capabilities are not known to localize cleanly above or below one depth boundary.

---

# 1. Has this been done?

## 1.1 Classical stitching lineage

### Lenc & Vedaldi

**Exact title:** *Understanding Image Representations by Measuring Their Equivariance and Equivalence*  
**arXiv:** **1411.5908**  
**CVPR 2015 DOI:** **10.1109/CVPR.2015.7298701**

This is the conceptual ancestor. It asks whether one representation can be reconstructed from another and whether the downstream part of one network can consume the mapped representation. It is a representation-equivalence study, not a “combine complementary capabilities” study. ([arxiv.org](https://arxiv.org/abs/1411.5908))

### Bansal, Nakkiran & Barak

**Exact title:** *Revisiting Model Stitching to Compare Neural Representations*  
**arXiv:** **2106.07682**  
**Venue:** NeurIPS 2021

The paper’s exact construction is the bottom of frozen model A, a simple trainable stitch, and the top of frozen model B. It shows that same-architecture vision networks trained from different initializations or objectives can often be stitched successfully, and argues that this functional test reveals information not captured by CKA. It uses stitching mainly as a **measurement tool**, not as a capability-fusion method. ([arxiv.org](https://arxiv.org/abs/2106.07682))

### Csiszárik et al.

**Exact title:** *Similarity and Matching of Neural Network Representations*  
**arXiv:** **2110.14633**  
**Venue:** NeurIPS 2021, pp. 5656–5668

The “Dr. Frankenstein” framework joins two trained vision networks at selected layers through linear/affine transformations. Its useful result for Paper D is that independently initialized networks of the same architecture can sometimes be made functionally compatible with a low-capacity affine map. It does not test LLMs, different tokenizers, positional systems, or complementary capabilities. ([arxiv.org](https://arxiv.org/abs/2110.14633))

### Stitchable Neural Networks

**Exact title:** *Stitchable Neural Networks*  
**arXiv:** **2302.06586**  
**DOI:** **10.1109/CVPR52729.2023.01545**

SN-Net stitches pretrained members of compatible vision-model families to generate an accuracy–compute frontier. The paper shows that a few epochs of stitch training can interpolate between anchors of different scale. Its purpose is deployment elasticity, not specialist-capability union. ([arxiv.org](https://arxiv.org/abs/2302.06586))

### Conclusion from the classical lineage

The classical literature supports:

- low-capacity recovery of functional compatibility in favorable settings;
- different random seeds and sometimes different training objectives;
- width-changing linear maps;
- and the inadequacy of CKA alone as a functional criterion.

It does **not** support the assumption that arbitrary frozen computational prefixes and suffixes are composable, or that a capability belonging to one donor survives transplantation into another.

---

## 1.2 LLM depth concatenation and community “FrankenMerging”

### SOLAR 10.7B

**Exact title:** *SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling*  
**arXiv:** **2312.15166**  
**DOI:** **10.18653/v1/2024.naacl-industry.3**

SOLAR’s depth up-scaling duplicates and rearranges layers from one pretrained model and then performs continued pretraining of the enlarged model. In Figure 1, a 32-layer source is expanded to 48 layers. Table 2 reports approximately **66.04 H6 for SOLAR-10.7B base** and **74.20 for its instruction-tuned version**, but those numbers follow continued pretraining and, for the latter, instruction/alignment tuning. They do not isolate raw layer concatenation. ([arxiv.org](https://arxiv.org/abs/2312.15166))

**What it shows:** rough depth manipulation can be repaired by substantial continued training.

**What it does not show:** independently pretrained donors, complementary capabilities, or repair by only one or two trainable interface layers.

### MergeKit passthrough / FrankenMerging

**Exact title:** *Arcee’s MergeKit: A Toolkit for Merging Large Language Models*  
**arXiv:** **2403.13257**

MergeKit documents passthrough or “FrankenMerging”: piecewise assembly of a model from selected layer ranges. It identifies this as a practical merge mode and points to community checkpoints such as Goliath-120B. The paper is a toolkit and practice report, not a controlled scientific study of depth splicing. ([arxiv.org](https://arxiv.org/abs/2403.13257))

I found **no controlled primary study** that systematically varies:

- same checkpoint versus different donors;
- shared versus independent ancestry;
- splice direction;
- splice layer;
- donor order;
- tokenizer compatibility;
- with/without stitch training;
- and retention of each donor’s specialist capability.

So community FrankenModels establish that people do this, not that cross-base passthrough merging reliably works.

**Goliath-120B:** verified as a canonical community example through the MergeKit paper; a matched, controlled donor comparison was not found.

**MegaDolphin:** verified as a community model name/model card, but I found no archival paper or controlled evaluation demonstrating complementary inheritance. Under your rule, do not cite it as scientific evidence.

---

## 1.3 Direct LLM stitching: the actual novelty collisions

### StitchLLM

**Exact title:** *StitchLLM: Serving LLMs, One Block at a Time*  
**DOI:** **10.18653/v1/2025.acl-long.1305**

This is a direct method collision. It uses frozen blocks from differently sized LLMs, a trainable stitching layer, and dynamic routing through hybrid block paths. Its experiments cover Llama 2 pairs—13B/7B/1.1B—and Llama 3 pairs—8B/3B/1B. The paper reports training **112 candidate stitch layers for Llama 2** and **92 for Llama 3** under its stitch-search procedure. ([aclanthology.org](https://aclanthology.org/2025.acl-long.1305/))

**Abstract claim:** improved serving throughput with limited performance degradation.

**What it actually shows:** hybrid decoder paths can work within closely related Llama ecosystems and across widths.

**What it does not show:**

- independent architectural families;
- tokenizer changes;
- specialist-capability inheritance;
- or a hybrid that beats both donors on different axes.

Paper D therefore cannot claim to be the first learned depth-wise LLM stitch.

### BTS

**Exact title:** *BTS: Harmonizing Specialized Experts into a Generalist LLM*  
**arXiv:** **2502.00075**  
**DOI:** **10.18653/v1/2025.emnlp-main.347**

BTS is the closest collision at the **problem-definition level**. It creates domain experts from a common seed, freezes the experts and seed/hub, inserts lightweight stitch layers, and trains those stitches on a small domain mixture. It explicitly claims retention of complementary code, math, and multilingual capabilities and reports better generalist performance than the merging/upcycling alternatives it evaluates. ([aclanthology.org](https://aclanthology.org/2025.emnlp-main.347/))

Important limitation: despite the abstract’s “independently trained experts” wording, the experts are branches from a **common seed**. Architecture, width, tokenizer, positional conventions, and initial representational basis are therefore controlled. BTS also inserts alternating expert-to-hub and hub-to-expert interfaces throughout the network, rather than using one serial boundary.

Paper D’s surviving distinctions are:

1. independent ancestry;
2. one sequential cut rather than parallel streams;
3. substantially greater architecture/tokenizer heterogeneity.

### LEGO-LLM

**Exact title:** *Building LLMs Like LEGO: Two-dimensional Architecture Reassembly of Large Language Models*  
**DOI:** **10.18653/v1/2026.acl-long.2081**

This is the strongest broad occupancy result. It treats pretrained Transformer blocks as reusable components, supports vertical recombination across depth and horizontal composition at a depth, freezes inherited blocks, and trains lightweight glue layers using data-free knowledge distillation. Table 5 is the glue-layer ablation; the retrieved paper text states that glue layers substantially improve over direct projection, though I could not reliably recover the numerical cells and therefore do not quote them. ([aclanthology.org](https://aclanthology.org/2026.acl-long.2081/))

**What is occupied:** architecture-level LLM block reassembly through learned glue.

**What remains unverified in that paper:** a rigorous Llama↔Qwen↔Gemma↔Mistral-style experiment crossing widths and tokenizers while preserving donor-specific benchmark strengths.

### Bottom line on occupancy

| Claim | Status |
|---|---|
| First LLM layer stitching | **Dead** — StitchLLM |
| First frozen-block LLM architecture reassembly | **Dead** — LEGO-LLM |
| First learned integration of frozen LLM specialists | **Dead** — BTS and CALM |
| First affine mapping between LLM residual streams | **Dead** — Chen et al. |
| First heterogeneous/cross-tokenizer knowledge fusion | **Dead** — FuseLLM/FuseChat |
| First single-boundary cross-family capability-preserving transplant | **Not verified as done** |
| First true cross-tokenizer serial depth splice | **Not verified as done** |

---

## 1.4 Knowledge fusion and composition rather than serial stitching

### FuseLLM

**Exact title:** *Knowledge Fusion of Large Language Models*  
**arXiv:** **2401.10491**  
**Venue:** ICLR 2024

FuseLLM aligns and combines output distributions from structurally diverse source models and continually trains one target model. Its sources include Llama-2-7B, OpenLLaMA-7B, and MPT-7B, so architecture and tokenizer heterogeneity are real. ([arxiv.org](https://arxiv.org/abs/2401.10491))

The tables are more modest than the broad “fused strengths” framing:

- **Table 1:** average **5.16% relative BBH improvement** over original Llama-2, versus **1.86%** for ordinary continued LM training.
- **Table 2:** **1.25% relative commonsense improvement**, versus **0.16%** for continued LM training.
- **Table 3:** **6.36% relative MultiPL-E gain**, versus **1.37%** for continued LM training.
- **Figure 2:** about **2.5 absolute BBH points over the continued-training control**, reaching that control’s best level after roughly **0.52B tokens**. ([arxiv.org](https://arxiv.org/pdf/2401.10491))

Crucially, FuseLLM improves the target but does **not** reliably reach the best code donor. That is negative evidence against assuming cheap fusion will preserve every specialist’s ceiling.

### FuseChat

Two records exist and should not be conflated:

- *FuseChat: Knowledge Fusion of Chat Models*, arXiv:**2402.16107**.
- Expanded *FuseChat: Knowledge Fusion of Chat Models*, arXiv:**2408.07990**, DOI **10.18653/v1/2025.emnlp-main.1096**.

The expanded method performs pairwise distribution fusion from structurally different chat models into instances of a common target architecture, followed by same-architecture parameter merging. It is heterogeneous-teacher fusion, not depth-wise execution of donor blocks. ([arxiv.org](https://arxiv.org/abs/2408.07990))

### CALM

**Exact title:** *LLM Augmented LLMs: Expanding Capabilities through Composition*  
**arXiv:** **2401.02412**

CALM freezes an anchor and an augmenting specialist and learns cross-attention modules between them. It is probably the most relevant conceptual baseline because it combines internal representations from frozen specialists while avoiding the assumption of a position-by-position residual handoff. Both models continue running, so it is more expensive than a serial hybrid but much less brittle. ([arxiv.org](https://arxiv.org/abs/2401.02412))

### LLM-Blender

**Exact title:** *LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion*  
**arXiv:** **2306.02561**  
**DOI:** **10.18653/v1/2023.acl-long.792**

This is the mandatory output-level baseline. It uses PairRanker to select candidate responses and GenFuser to combine them, exploiting the observation that the best donor varies by example. It does not create a single internally fused model, but it tests whether the hypothesized complementarity exists at all. ([arxiv.org](https://arxiv.org/abs/2306.02561))

---

# 2. What does the evidence say about feasibility?

## 2.1 Evidence by difficulty regime

| Regime | Evidence | Judgment |
|---|---|---|
| Same architecture, different seed | Classical stitching is often successful with affine maps | Positive but vision-heavy |
| Same family, different size/width | StitchLLM and affine residual maps work for restricted objectives | Plausible and occupied |
| Different family, same/similar width | Concept vectors and final states can sometimes be mapped | Weak evidence for interior stitching |
| Different family and width | No verified broad interior A-front→B-back result | Open and unsupported |
| Different tokenizer | Tokenizer transfer/distillation exists, but not serial residual handoff | Conventional bridge is structurally inadequate |

## 2.2 Direct residual-stream evidence

### Chen et al.

**Exact title:** *Transferring Linear Features Across Language Models With Model Stitching*  
**arXiv:** **2506.06609**  
**Venue:** NeurIPS 2025 spotlight

This paper learns affine maps between residual streams of differently sized LMs and transfers:

- sparse autoencoders;
- probes;
- steering vectors;
- and related linear features.

Its most concrete headline number is that initializing a large-model SAE from a transferred small-model SAE can reduce subsequent **SAE-training compute by approximately 50%**. That is not a 50% saving in LLM training and not evidence of benchmark-capability fusion. ([arxiv.org](https://arxiv.org/abs/2506.06609))

Figure 3 covers probe transfer over eight binary datasets; retraining transferred probes on target activations nearly recovers directly trained target probes on most tasks. Figure 9 reports weaker zero-shot transfer for code- and language-related probes. Table 1 evaluates next-token cross-entropy after residual mapping, but I could not recover the cells reliably and do not quote them.

This supports:

> related models share transferable linear feature subspaces.

It does not establish:

> the complete source computation is a valid input to many frozen nonlinear target blocks.

### Cross-family concepts

**Exact title:** *Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts*  
**arXiv:** **2501.02009**  
**DOI:** **10.18653/v1/2025.acl-long.185**

This paper learns linear transformations between model concept spaces and transfers steering vectors, including weak-to-strong transfers. That is real cross-family functional transfer, but the transferred object is a low-dimensional direction, not an entire tokenwise residual state. ([aclanthology.org](https://aclanthology.org/2025.acl-long.185/))

### Cross-family final states

**Exact title:** *Characterizing Linear Alignment Across Language Models*  
**arXiv:** **2603.18908**

This March 2026 preprint maps final hidden states between independently trained models and tests embedding tasks and generation across **34 model pairs**. It reports that text generation is sometimes possible after mapping a model’s final representation into another model’s output head. ([arxiv.org](https://arxiv.org/abs/2603.18908))

This is much easier than Paper D. A final-state map only has to preserve information needed by a mostly linear vocabulary decoder. An interior stitch must remain usable by every downstream attention block, MLP, normalization layer, and autoregressive update.

### KV-cache translation

**Exact title:** *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse*  
**arXiv:** **2608.03893**

This is extremely recent—submitted August 4, 2026—and preliminary. Across six **within-family** pairs, a linear mapper retains **73–98%** of standalone-prefill benchmark accuracy on four pairs; two pairs degrade sharply. An MLP recovers up to **37 percentage points of HellaSwag retention** on failed pairs, and the method runs **2.7–25× faster** than recomputing prefill. ([arxiv.org](https://arxiv.org/abs/2608.03893))

The unfavorable detail is that useful target caches often require multiple source layers and explicit positional handling. Even within a family, a simple linear relation is not universal.

## 2.3 Direct negative result

**Exact title:** *A Negative Result on Cross-Model Activation Transfer in a Pythia Multi-Hop Setting*  
**arXiv:** **2606.03280**

A linear Pythia-160M→410M map achieves normalized cosine similarity near **0.97 across seeds**, yet injecting the mapped activation does not improve downstream answering. Low-strength additive injection is indistinguishable from no injection, replacement is destructive, and rescaling to the receiver’s native norm does not rescue it. ([arxiv.org](https://arxiv.org/html/2606.03280v1))

This is not a definitive refutation of a task-trained two-layer bridge—it uses post-hoc linear translation and activation injection—but it directly falsifies the easy inference:

> high hidden-state alignment ⇒ receiver can use the state causally.

That is probably the most relevant negative result.

## 2.4 CKA/SVCCA: the requested quantitative answer

I did **not** find a citation-safe table giving exact CKA or SVCCA values for intermediate residual streams across independent modern families such as Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B, and Mistral-7B under a controlled common corpus.

The available evidence is weaker:

- *The Platonic Representation Hypothesis*, arXiv:**2405.07987**, compares pooled example geometry and argues that larger models become more representationally aligned. It does not perform tokenwise interior stitching. ([arxiv.org](https://arxiv.org/abs/2405.07987))
- *Revisiting the Platonic Representation Hypothesis: An Aristotelian View*, arXiv:**2602.14486**, shows that model width and depth can inflate raw similarity metrics and that much global convergence weakens under null calibration. ([arxiv.org](https://arxiv.org/abs/2602.14486))
- *Relative representations enable zero-shot latent space communication*, arXiv:**2209.15430**, and *Latent Space Translation via Semantic Alignment*, arXiv:**2311.00664**, demonstrate zero-shot latent communication and simple closed-form translations across encoders/decoders, architectures, and even modalities. Neither is an interior decoder-LLM computation result. ([arxiv.org](https://arxiv.org/abs/2209.15430))

Therefore, CKA/SVCCA should be treated as **screening diagnostics**, not feasibility evidence. Similar sample geometry does not imply that B’s frozen nonlinear dynamics are approximately conjugate to A’s lower computation.

---

# 3. Cross-tokenizer transfer

Relevant verified work includes:

- *Zero-Shot Tokenizer Transfer*, arXiv:**2405.07883**;
- *Franken-Adapter: Cross-Lingual Adaptation of LLMs by Embedding Surgery*, arXiv:**2502.08037**;
- *Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching*, arXiv:**2503.20083**;
- *Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit*, arXiv:**2506.06607**. ([arxiv.org](https://arxiv.org/abs/2502.08037))

These papers show that tokenizer interfaces can be replaced or that knowledge can be distilled across different tokenizations. ZeTT comes close to native performance but reports that the remaining gap can be closed by continued training on **less than one billion tokens**—efficient relative to pretraining, but not “near-zero.” ([arxiv.org](https://arxiv.org/pdf/2405.07883))

Franken-Adapter reports, at the abstract level, gains of up to **20% across 96 languages**, English regression below **1%**, and approximately **14% improvement over a math-specialized model across 20 languages**. But it recombines tokenizer/embedding modules with compatible model bodies; it does not feed one model family’s mid-layer states into another’s upper stack. ([arxiv.org](https://arxiv.org/abs/2502.08037))

## Structural problem

If A tokenizes a string into \(n_A\) positions and B into \(n_B\), normally \(n_A \neq n_B\). A normal Transformer bridge preserves sequence length:

\[
\mathbb{R}^{n_A \times d_A}
\rightarrow
\mathbb{R}^{n_A \times d_B},
\]

but B’s upper layers were trained on:

\[
\mathbb{R}^{n_B \times d_B}.
\]

A width projection does not solve:

- sequence-length conversion;
- different subword boundaries;
- B-position semantics and KV-cache indexing;
- which tokenizer controls each autoregressive generation step;
- or how B-generated tokens are embedded by A on the next step.

Consequently, “cross-tokenizer stitching” is not just representation alignment. It requires a sequence transducer, byte/span interface, latent resampler, or prior tokenizer transplantation. The literature repeatedly avoids this serial mismatch through output-distribution alignment, byte/chunk likelihoods, embedding replacement, or cross-attention composition. ([arxiv.org](https://arxiv.org/abs/2503.20083))

**Bluntly:** true cross-tokenizer serial stitching should be separated from the initial paper. Otherwise a negative result will be uninterpretable.

---

# 4. Benchmarks

## 4.1 First problem: finding genuinely complementary donors

I did not find a convincing modern base-model pair with strong bidirectional complementarity established under one reproducible protocol. Published model cards and reports use different prompts, shot counts, likelihood normalization, majority voting, answer parsers, and chat templates.

The best provisional base-model lead is **Mistral-7B Base versus Llama-3.1-8B Base**, from Mistral’s official evaluation table:

| Benchmark | Mistral-7B | Llama-3.1-8B |
|---|---:|---:|
| MMLU | 62.5 | **64.7** |
| AGIEval | 42.5 | **44.4** |
| ARC-Challenge | **67.9** | 46.0 |
| TriviaQA | **62.5** | 60.2 |
| HumanEval pass@1 | 26.8 | **37.8** |
| GSM8K maj@8 | 32.0 | **42.2** |
| German MMLU | 49.6 | **52.8** |
| Spanish MMLU | 51.4 | **54.6** |

([huggingface.co](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410))

This looks complementary, but the **21.9-point ARC gap is suspiciously large** and should be assumed to reflect protocol sensitivity until reproduced. The first project milestone should therefore be a standardized donor screen, not stitch training.

Aya-23-8B versus Mistral-7B-Instruct is a possible multilingual stress pair: the Aya report’s Figure 1 reports an average **65.2% win rate against Mistral-7B-Instruct-v0.2** over its multilingual evaluation, while Mistral remains preferred in English. But these are instruction-tuned models and the result entangles language ability, alignment data, response style, verbosity, and preference judging. ([arxiv.org](https://arxiv.org/abs/2405.15032))

A same-family general/code pair such as Qwen2.5-7B Base and Qwen2.5-Coder-7B Base is a useful positive control, but not a strong novelty pair: shared architecture, tokenizer, and ancestry make weight merging and ordinary continued training applicable.

## 4.2 Required donor-screening criterion

Before any stitching, calculate:

1. **Task oracle:** choose the better donor per benchmark family.
2. **Example oracle:** count an item correct if either donor gets it correct.
3. **Disagreement rate:** how often one donor is right and the other wrong.
4. **Bidirectional specialty gaps:** A must materially beat B somewhere and B materially beat A elsewhere.
5. Confidence intervals under one frozen harness.

If the example oracle is only marginally better than the stronger donor, there is no meaningful “best of both” target.

## 4.3 Recommended final suite

### Knowledge and difficult science

- **MMLU-Pro** — *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark*, arXiv:**2406.01574**. It expands questions from four to ten choices and removes trivial/noisy MMLU items. ([arxiv.org](https://arxiv.org/abs/2406.01574))
- **GPQA**, including Diamond and full-set results — *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*, arXiv:**2311.12022**. It contains only **448 questions**, so report paired uncertainty. ([arxiv.org](https://arxiv.org/abs/2311.12022))

### Math and reasoning

- A fixed-date release of **LiveBench** math and reasoning — *LiveBench: A Challenging, Contamination-Limited LLM Benchmark*, arXiv:**2406.19314**. ([arxiv.org](https://arxiv.org/abs/2406.19314))
- MMLU-Pro subject-level scores.
- GSM8K and MATH only as legacy diagnostics, with exact shot count, CoT format, `maj@1` versus `maj@8`, and decoding budget.

### Code

- **LiveCodeBench** with a problem-date window after the donors’ documented training cutoffs — *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code*, arXiv:**2403.07974**. It evaluates generation, self-repair, execution, and test-output prediction. ([arxiv.org](https://arxiv.org/abs/2403.07974))
- HumanEval and MBPP only for compatibility with old reports.

### Multilinguality

- **Global MMLU** — *Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation*, arXiv:**2412.03304**, DOI **10.18653/v1/2025.acl-long.919**. Report all 42 languages and separate culturally sensitive/agnostic subsets. ([ar5iv.labs.arxiv.org](https://ar5iv.labs.arxiv.org/html/2412.03304))
- **MGSM** — *Language Models are Multilingual Chain-of-Thought Reasoners*, arXiv:**2210.03057**. It contains 250 GSM8K problems translated into ten languages; it is useful but insufficient as the sole multilingual measure. ([arxiv.org](https://arxiv.org/abs/2210.03057))

### Long context

- **RULER** across 4K–128K, not merely needle-in-a-haystack — *RULER: What’s the Real Context Size of Your Long-Context Language Models?*, arXiv:**2404.06654**. ([arxiv.org](https://arxiv.org/abs/2404.06654))
- **LongBench v2** — arXiv:**2412.15204**, DOI **10.18653/v1/2025.acl-long.183** — 503 problems with 8K-to-2M-word contexts across six realistic categories. ([arxiv.org](https://arxiv.org/abs/2412.15204))

For cross-tokenizer comparisons, report context in both native tokens and tokenizer-independent units such as UTF-8 bytes or words.

## 4.4 Benchmark traps

1. **Base versus chat scores:** do not mix them.
2. **Likelihood MC versus generated answers:** report both separately.
3. **`maj@8` versus pass@1:** these are different compute budgets.
4. **Chat templates:** evaluate each donor with its official template; the hybrid has no natural template.
5. **Contamination:** HumanEval, MBPP, GSM8K, MATH, and MMLU are diagnostics, not decisive evidence.
6. **Training leakage:** if bridge training includes benchmark-like math/code prompts, gains may be learned by the bridge rather than inherited.
7. **Single average:** an aggregate can hide that the hybrid is mediocre on every axis.
8. **Long-context passkey tests:** RULER explicitly shows that simple retrieval is only a superficial test. ([proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/file/94074dd5a072d28ff75a76dabed43767-Paper-Conference.pdf))

## 4.5 Mandatory baselines

### Architecture controls

- intact donor A;
- intact donor B;
- A and B truncated to matched hybrid FLOPs/depth;
- A-front → bridge → A-back;
- B-front → bridge → B-back;
- direct projection;
- affine map;
- MLP;
- one Transformer block;
- two Transformer blocks;
- randomly initialized replacement blocks;
- LoRA/adapters on the stronger donor with equal trainable parameters and tokens.

### Complementarity controls

- task oracle;
- example oracle;
- cost-adjusted oracle.

### Practical combination baselines

- LLM-Blender / candidate ranking and fusion;
- task-level router;
- learned input router such as **RouteLLM**, *RouteLLM: Learning to Route LLMs with Preference Data*, arXiv:**2406.18665**;
- frozen parallel branches with a learned gate;
- CALM-style cross-attention composition;
- FuseLLM-style output-distribution distillation. ([arxiv.org](https://arxiv.org/abs/2406.18665))

Whenever donors share architecture or ancestry, also include weight averaging, Task Arithmetic, TIES, DARE, and relevant MergeKit recipes. Otherwise reviewers can reasonably argue that the paper selected an unnecessarily difficult mechanism.

---

# 5. Ranked kill risks

## 1. Cross-tokenizer sequence mismatch — **critical if tokenizers differ**

This is a structural mismatch, not merely a basis mismatch. A conventional bridge cannot generally turn \(n_A\) token positions into \(n_B\) positions or define consistent autoregressive updates across two vocabulary factorizations.

**Likely consequence:** either restrict the first paper to a shared tokenizer or admit a sequence-transducing bridge that is much larger than the proposed interface.

## 2. Causal co-adaptation / off-manifold receiver states — **critical**

B’s upper layers were optimized only on states produced by B’s lower layers. Matching mean squared error, cosine, CKA, or even next-token loss at the immediate boundary does not guarantee that the state has the correct rare features, correlations, residual-update ratios, local derivatives, or positional circuits.

*How Not to Stitch Representations to Measure Similarity: Task Loss Matching versus Direct Matching*, arXiv:**2412.11299**, DOI **10.1609/aaai.v39i15.33698**, shows that task-loss-trained stitches can manufacture out-of-distribution states that solve the supervised task and can produce nonsensical representation-similarity rankings. ([arxiv.org](https://arxiv.org/abs/2412.11299))

**Interpretation risk:** a bridge that scores well on a finite suite may have learned benchmark-specific control codes, not inherited A’s capability.

## 3. Capabilities are not localized to contiguous depth — **high**

The premise “A’s lower layers encode language; B’s upper layers reason” is unsupported. B’s strength may require:

- early token binding and retrieval;
- middle-layer state construction;
- repeated refinement over many blocks;
- and late decoding.

*Investigating Layer Importance in Large Language Models*, arXiv:**2409.14381**, reports “cornerstone” early layers whose removal can collapse performance toward random guessing, while other layers are relatively redundant. This is inconsistent with a simple homogeneous-depth story. ([arxiv.org](https://arxiv.org/html/2409.14381v1))

If B’s specialist advantage begins below the cut, the bridge would have to reconstruct missing B computation. That is distillation, not cheap alignment.

## 4. Representational similarity is not causal usability — **high**

The Pythia negative result gives the cleanest warning: approximately **0.97 normalized cosine**, no useful transfer. ([arxiv.org](https://arxiv.org/html/2606.03280v1))

Raw CKA is additionally confounded by width and depth. ([arxiv.org](https://arxiv.org/abs/2602.14486))

A paper whose main evidence is “CKA is high” should be rejected.

## 5. Positional and attention convention mismatch — **high for long context**

Different RoPE bases, scaling methods, sliding-window patterns, attention-head structures, and learned positional circuits can make identical token semantics computationally different. The recent KV-transfer paper explicitly removes RoPE before fitting its maps, illustrating that positional structure cannot simply be ignored. ([arxiv.org](https://arxiv.org/abs/2608.03893))

Short-context same-RoPE experiments may survive. Long-context cross-family transfer is much less likely.

## 6. LayerNorm/RMSNorm and residual-scale mismatch — **high but repairable**

Normalizing the bridge output does not align covariance, rare directions, per-channel gains, or the relative scale of residual state and block update. The Pythia experiment’s norm-matching control did not rescue transfer. ([arxiv.org](https://arxiv.org/html/2606.03280v1))

Trainable norms, whitening/coloring, residual gates, and target-statistic penalties should be mandatory ablations.

## 7. “One or two Transformer layers” are not very small — **medium-high**

For width \(d=4096\), a conventional dense block is on the order of:

\[
12d^2 \approx 201.3\text{ million parameters}.
\]

Two blocks are approximately **402.7M parameters**, before width-conversion projections. This is small relative to two full 7–8B donors, but not small enough to dismiss the possibility that the bridge itself learns a significant portion of the benchmark behavior.

Parameter count is also the wrong cost metric. Report:

- training tokens;
- training FLOPs;
- wall-clock GPU hours;
- activation memory;
- and inference latency.

## 8. No real donor complementarity — **high for interest**

If the standardized donor screen shows one model broadly dominates the other, a successful stitch merely approximates the stronger donor badly.

This is a serious risk: modern 7–9B reports often show broad dominance rather than balanced specialization. The apparently strongest candidate crossover—Mistral versus Llama—contains a suspiciously protocol-sensitive ARC result.

## 9. Easier alternatives will win — **high for practical relevance**

Routing and output ensembling do not require internal-state compatibility. CALM avoids serial token alignment. FuseLLM performs explicit knowledge transfer. Weight merging is simpler when applicable.

If Paper D does not beat these in either quality or a clear compute/memory Pareto tradeoff, the method is uninteresting even if it technically works.

---

# 6. Defensible novelty, if the project survives

## Framing 1: **Limits of single-boundary cross-family transplantation**

> Can independently pretrained LLM families be made functionally composable at one interior boundary, and where does composability break?

### Decisive experiment

Use a precommitted ladder:

1. self-stitch;
2. different seed, same architecture;
3. same family, different width;
4. different family with a shared tokenizer;
5. different family and tokenizer.

Evaluate every layer-pair direction with:

- held-out next-token cross-entropy;
- downstream benchmark retention;
- OOD residual detection;
- downstream-state matching;
- and broad tasks absent from bridge training.

A strong negative result could be publishable if it precisely maps the failure boundary and reproduces the distinction between geometric alignment and causal usability.

**This is the safest framing.** It remains valuable even if “best of both” fails.

## Framing 2: **Causal usability, not representational similarity**

> Which alignment objectives predict whether a frozen receiver can actually continue computation?

### Decisive experiment

Compare:

- hidden-state MSE;
- whitened affine matching;
- CKA/SVCCA-selected layer pairs;
- next-block output matching;
- multi-layer trajectory matching;
- KL matching against intact B;
- Jacobian-vector matching;
- task loss;
- and combinations thereof.

The contribution would be a demonstrated predictor of full-stack stitch success—not another report of high CKA or probe transfer.

A compelling result would show that conventional similarity metrics fail, while a causal/trajectory metric predicts recovery of held-out perplexity and generation across unseen tasks.

## Framing 3: **True cross-tokenizer sequence-transducing transplantation**

> Can a byte/span-aligned latent resampler convert A-token residual sequences into valid B-token residual sequences?

### Decisive experiment

Compare:

1. native-token serial bridge;
2. tokenizer transplantation first, then ordinary stitching;
3. byte/span-aligned variable-length resampler;
4. CALM-style cross-attention;
5. cross-tokenizer distillation;
6. output-level ensemble.

Success requires stable autoregressive generation and preservation of donor specialties, not just input classification.

This would be genuinely novel, but it is probably **not** compatible with the original “one or two ordinary Transformer layers” simplicity claim.

---

# 7. Explicitly unverified items

The following were not verified as archival papers matching the proposed mechanism:

- **“Text-To-Text alignment”** as a cross-LLM hidden-state alignment paper — **unverified; do not cite**.
- A systematic paper evaluating MergeKit passthrough across independently pretrained base models — **not found**.
- A controlled Goliath-120B donor-versus-merge study under matched protocols — **not found**.
- A controlled MegaDolphin capability-inheritance study — **not found**.
- **“LLM chimera”** as a canonical peer-reviewed name for serial depth stitching — **unverified; do not cite without a specific paper**.
- **“Cross-model layer transplant”** as an exact archival paper establishing independent-family serial stitching — **not found**.
- **“Transplant then Align”** as a matching LLM paper — **unverified; do not cite**.
- *Transplanting Knowledge: A Study on Layer-Specific Grafting in LLMs* was found as an OpenReview/workshop document, but I could not verify an arXiv ID or DOI. Under your rule: **unverified; do not cite as formal prior art**.
- Community claims that FrankenMerging systematically combines independent model strengths — **unsupported by the retrieved MergeKit paper**.

One correction: *Latent Space Translation via Semantic Alignment* is real and verified—arXiv:**2311.00664**, NeurIPS 2023. It is not “Text-To-Text alignment,” and it concerns encoder/decoder latent translation across tasks, architectures, and modalities rather than interior autoregressive LLM blocks. ([arxiv.org](https://arxiv.org/abs/2311.00664))

# Final recommendation

**Do not pitch Paper D as a new model-stitching or modular-LLM method. That claim is untenable after StitchLLM, BTS, and LEGO-LLM.**

A defensible paper title would be closer to:

> **Limits and Possibilities of Single-Boundary Cross-Family LLM Transplantation**

The project should begin with two kill tests:

1. **Standardized donor complementarity screen.** Stop if there is no substantial bidirectional crossover and example-oracle headroom.
2. **Same-family/different-width causal stitch.** Stop if the bridge achieves high geometric alignment but cannot recover receiver perplexity and generation on held-out text.

If those pass, proceed to independent families with a shared tokenizer. Do **not** begin with different tokenizers.

My blunt forecast is that:

- self-stitch and same-family cases will work;
- independent-family mappings will recover some language-model loss and coarse semantics;
- performance will be strongly directional;
- “best of both” will usually degrade toward an average or the tail donor;
- and genuine cross-tokenizer serial stitching will require enough resampling/distillation machinery that the original near-zero-cost premise no longer holds.