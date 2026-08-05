# Novelty / prior-art teardown

**Internet access:** Yes. I had web/internet access and checked primary papers, arXiv metadata, conference versions where verifiable, and the full local brief. This review covers work available through **August 5, 2026**. Where I could not verify an archival venue, I explicitly label the work as an arXiv preprint rather than guessing.

---

## Executive verdict

### Has the paper been scooped?

**The broad construction claim has been materially scooped. The exact full tuple has not, as far as I could verify.**

The most damaging overlooked paper is:

- **Yao Lu et al., “Reassessing Layer Pruning in LLMs: New Insights and Methods,” arXiv:2411.15558, 2024.**

It already does all three of the differentiators currently emphasized in the brief:

1. **Genuine shallowing:** removes the final 25% of layers.
2. **Frozen trunk:** freezes earlier surviving layers and trains only the final one to three surviving layers plus `lm_head`.
3. **Decoder-only LLMs at 7–8B scale:** Vicuna-7B, Qwen1.5-7B, and Llama-3.1-8B-Instruct.

Therefore the claim

> “Unlike prior work, we genuinely shorten a 7B decoder LLM and keep the trunk frozen”

is **false as a novelty claim**.

What Lu et al. do **not** do is replace a larger deleted suffix with fewer **randomly initialized full Transformer blocks**. They train the last few **surviving pretrained** blocks. That is the remaining architectural distinction.

However, even the replacement side is crowded:

- **Xiaodong Chen et al., “Streamlining Redundant Layers to Compress Large Language Models,” arXiv:2403.19135, ICLR 2025** removes a consecutive block span, inserts a smaller FFN/SwiGLU/Transformer replacement, freezes the original model, and trains only the replacement. It explicitly tests a randomly initialized Transformer replacement.
- **Shoaib Ahmed Siddiqui et al., “A Deeper Look at Depth Pruning of LLMs,” arXiv:2407.16286, 2024 preprint** inserts trainable low-rank linear modules in place of missing 7B LLM blocks, including recovery by supervised fine-tuning.
- **Tianxiang Chen et al., “Llama SLayer 8B: Shallow Layers Hold the Key to Knowledge Injection,” arXiv:2410.02330, Findings of EMNLP 2024** deletes final layers, adds/reinitializes blocks, freezes inherited blocks, and trains only the new/reinitialized blocks—although its resulting model is usually deeper, its blocks are inherited/averaged rather than random, and it uses continued pretraining.
- **Chengyue Wu et al., “LLaMA Pro: Progressive LLaMA with Block Expansion,” arXiv:2401.02415, 2024 preprint** freezes a 7B decoder backbone and trains only newly inserted full Transformer blocks.
- **Dyah Adila et al., “Grow, Don’t Overwrite: Fine-tuning Without Forgetting,” arXiv:2603.08647, 2026 preprint** freezes all original model parameters and performs downstream adaptation using only newly added capacity, though it grows MLP width rather than replacing deleted depth.

### My strict area-chair assessment

- **P-C1 as a broad principle is not novel enough.**
- **The exact random full-cap construction is narrowly unscooped but is presently a combination novelty, not a strong conceptual novelty.**
- **P-C2 is the strongest remaining direction, but its broad form is also partially scooped.**
- **The current keep14 evidence does not support a top-venue claim of useful adaptation or capability retention.**
- In the present form, I would expect a **reject on both novelty and empirical significance**.
- A top-venue submission remains possible only after a substantial reframe and decisive new comparisons.

---

# A. Related-work survey, 2021–2026

## A.1 Direct construction-level neighbors

| Work | Exact overlap with Paper C | Important difference | Scoop risk |
|---|---|---|---|
| **Lu et al., “Reassessing Layer Pruning in LLMs: New Insights and Methods,” arXiv:2411.15558, 2024** | Tail-prunes 25%; produces a genuinely shallower decoder-only 7–8B model; freezes the earlier surviving trunk; SFTs only `lm_head` and the final 1–3 surviving blocks | Uses pretrained surviving blocks, not newly random blocks | **Very high. Scoops all three stated differentiators.** |
| **Chen et al., “Streamlining Redundant Layers to Compress Large Language Models,” arXiv:2403.19135, ICLR 2025** | Deletes multiple full decoder blocks; inserts one smaller replacement network; freezes inherited model; trains only replacement; experiments at Llama-2-7B/13B; includes random Transformer replacement | Usually removes an internal consecutive span; replacement is one bridge; main objective is hidden-state reconstruction/LM recovery rather than downstream task adaptation | **Very high architectural overlap.** |
| **Siddiqui et al., “A Deeper Look at Depth Pruning of LLMs,” arXiv:2407.16286, 2024 preprint** | Removes full blocks in Llama-2-7B/Mistral-7B and puts trainable modules in missing-block positions; considers SFT, hidden-state MSE, and logit distillation | Replacement is a low-rank linear map, not a full Transformer cap; not specifically a terminal suffix | **High.** |
| **Chen et al., “Llama SLayer 8B: Shallow Layers Hold the Key to Knowledge Injection,” arXiv:2410.02330, Findings of EMNLP 2024** | Deletes final layers, adds/reinitializes blocks, freezes inherited blocks, trains only expanded/reinitialized blocks | Expands mainly in the shallow half; average/identity initialization; net model is normally deeper; 30B-token continued pretraining | **High component-level overlap.** |
| **Wu et al., “LLaMA Pro: Progressive LLaMA with Block Expansion,” arXiv:2401.02415, 2024 preprint** | Decoder-only 7B; freezes original backbone; trains only newly added full blocks | Does not prune; grows from 32 to 40 layers; identity/copy initialization; continued pretraining | **High for frozen-backbone block grafting.** |
| **Yuan et al., “Why Lift so Heavy? Slimming Large Language Models by Cutting Off the Layers,” arXiv:2402.11700, 2024/2025 preprint** | Removes exactly the top \(k\) decoder layers, retains the lower prefix, downstream-fine-tunes the shallower model | Small GPT-2-XL/OPT scale; remaining prefix is fine-tuned rather than frozen; no fresh cap | **High for top-tail truncation plus downstream FT.** |
| **Adila et al., “Grow, Don’t Overwrite: Fine-tuning Without Forgetting,” arXiv:2603.08647, 2026 preprint** | Downstream adaptation by freezing all old parameters and training only added capacity; studies task-dependent number/location of expanded layers | Grows MLP width rather than depth; no inference reduction; function-preserving copied initialization; mainly Gemma 1B/4B | **High for the general “adapt only new capacity” thesis.** |
| **Bochkov, “Growing Transformers: Modular Composition and Layer-wise Expansion on a Frozen Substrate,” arXiv:2507.07129, 2025/2026 preprint** | Stacks new top decoder blocks above a frozen lower stack; trains only newest blocks and LM head | Starts from a shallow model and grows; small scale; continued pretraining; does not delete a large pretrained suffix | **Medium–high.** |

### Critical point

No verified paper I found contains the exact conjunction:

\[
\text{retain prefix }1{:}j,\quad
\text{delete suffix }j{+}1{:}L,\quad
\text{append random full blocks }K<L-j,\quad
\text{freeze prefix},\quad
\text{downstream-SFT only cap+norm+head}
\]

at 7B scale.

But almost every constituent operation has already appeared, often in combinations of three or four. Thus, the novelty is a **specific intersection**, not a new research paradigm.

---

## A.2 Requested depth-pruning and recovery literature

### ShortGPT

- **Xin Men et al., “ShortGPT: Layers in Large Language Models are More Redundant Than You Expect,” arXiv:2403.03853; Findings of ACL 2025.**
- Uses **Block Influence**, based on input/output hidden-state cosine similarity, to remove full Transformer blocks.
- Main method is training-free.
- A later section replaces removed blocks with lightweight gated MLPs and performs substantial post-training.
- Tested on Llama-2 and Baichuan2 at 7B/13B.

**Overlap:** genuine decoder depth reduction and use of small replacements after pruning.

**Difference:** no frozen-prefix downstream cap in the main method; post-training replacement is an MLP and uses large-scale LM training rather than task SFT.

**Implication:** You cannot claim novelty for “prune full LLM layers, add a smaller module, and retrain.”

---

### LaCo

- **Yifei Yang, Zouying Cao, Hai Zhao, “LaCo: Large Language Model Pruning via Layer Collapse,” arXiv:2402.11187; Findings of EMNLP 2024.**
- Collapses adjacent decoder layers through representation similarity and parameter merging.
- Applies post-training to the resulting shallower Llama/Baichuan models.

**Overlap:** consecutive depth reduction and recovery.

**Difference:** no random blocks and no frozen trunk; it merges removed-layer information into surviving blocks and post-trains the compressed model.

---

### SliceGPT

- **Saleh Ashkboos et al., “SliceGPT: Compress Large Language Models by Deleting Rows and Columns,” arXiv:2401.15024; ICLR 2024.**
- Width/channel slicing through rotational invariance and PCA.
- Optional LoRA recovery.

**Overlap:** compressed decoder LLM plus restricted recovery.

**Difference:** block count is unchanged. This is not depth pruning and does not create a terminal cap.

**Risk:** low as construction prior art, but necessary as a compression baseline.

---

### Shortened LLaMA

- **Bo-Kyeong Kim et al., “Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods,” arXiv:2402.02834, 2024 preprint.**
- Removes full blocks using Taylor-based or one-at-a-time perplexity criteria.
- Protects the first four and final two layers.
- Compares no retraining, LoRA, continued pretraining, and continued pretraining followed by LoRA.
- Reports that aggressively pruned inherited models learn substantially faster and better than identical random-initialized architectures.

**Overlap:** inherited shallow network versus same-shape random initialization; depth pruning plus recovery.

**Difference:** no fresh cap and no frozen trunk; aggressive recovery updates the entire pruned model with continued pretraining.

**P-C1 risk:** The logical core of your A4-vs-A3 finding—“inherited pruned initialization beats random same-depth initialization”—already exists in a stronger language-model recovery setting.

---

### LLM-Pruner

- **Xinyin Ma, Gongfan Fang, Xinchao Wang, “LLM-Pruner: On the Structural Pruning of Large Language Models,” arXiv:2305.11627; NeurIPS 2023.**
- Uses gradient/Taylor information over coupled structural groups.
- Mostly prunes channels, attention heads, and MLP structures rather than deleting whole depth.
- Uses LoRA for rapid recovery.

**Overlap:** task-agnostic pruning followed by restricted-parameter adaptation.

**Difference:** generally not a shallower network and no newly grown full blocks.

---

### Sheared-LLaMA

- **Mengzhou Xia et al., “Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning,” arXiv:2310.06694; ICLR 2024.**
- Learns masks to prune a 7B LLaMA2 into specified 1.3B/2.7B shapes, including fewer layers.
- Then continued-pretrains the entire student for roughly 50B tokens, with dynamic data reweighting.

**Overlap:** inherited shallower model rather than training the small model from scratch.

**Difference:** all remaining weights are trainable; no fresh cap; compression/pretraining objective rather than downstream-only adaptation.

---

### Minitron

Two related papers:

1. **Saurav Muralidharan et al., “Compact Language Models via Pruning and Knowledge Distillation,” arXiv:2407.14679, 2024 preprint.**
2. **Sharath Turuvekere Sreenivas et al., “LLM Pruning and Distillation in Practice: The Minitron Approach,” arXiv:2408.11796, 2024 preprint.**

They prune depth, width, heads, and MLP dimensions and then retrain the whole student using knowledge distillation. The “in practice” paper constructs a **32-to-16-layer Llama-3.1-8B** depth model.

**Overlap:** inherited depth reduction and comparison with random initialization; actual decoder LLM compute reduction.

**Difference:** recovery is full-student distillation on hundreds of billions of tokens, not a frozen trunk and task cap.

**Empirical warning:** even this vastly more expensive recovery regime does not guarantee recovery of all generative/reasoning capabilities.

---

### LLM-Streamline

- **Xiaodong Chen et al., “Streamlining Redundant Layers to Compress Large Language Models,” arXiv:2403.19135; ICLR 2025.**

This is the most important replacement precedent.

It:

1. Chooses a consecutive span using endpoint hidden-state cosine similarity.
2. Deletes the span.
3. Replaces it with an FFN, SwiGLU network, or Transformer block.
4. Freezes the original LLM.
5. Trains only the replacement.
6. Tests random, first-pruned-layer, last-pruned-layer, and averaged initialization for the Transformer replacement.
7. Includes an LM-loss post-training comparison against parameter-matched LoRA.

**Overlap:** almost the whole abstract construction: multiple blocks removed, smaller replacement inserted, retained model frozen, only replacement trained, Llama-2-7B/13B, real inference compression.

**Remaining differences:**

- replacement is normally one internal bridge rather than a terminal task cap;
- its main training target reconstructs the removed hidden transition on generic data;
- your proposal makes the replacement task-specific and discards the entire suffix;
- your \(K\) may be greater than one but is still less than the removed suffix.

This paper should be named in the first paragraph of related work, not buried among pruning baselines.

---

### FinerCut

- **Yang Zhang et al., “FinerCut: Finer-grained Interpretable Layer Pruning for Large Language Models,” arXiv:2405.18218, 2024 preprint.**
- Treats attention and FFN residual sublayers as separate candidates.
- Greedily minimizes output-logit change.
- No recovery fine-tuning in its main experiments.

**Overlap:** real executed-depth/compute reduction at decoder 7B–70B scale.

**Difference:** no graft, no freezing problem, and no downstream adaptation.

---

### SLEB and Gromov et al.

Additional central depth-pruning papers:

- **Jiwon Song et al., “SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks,” arXiv:2402.09025, 2024 preprint.**
- **Andrey Gromov et al., “The Unreasonable Ineffectiveness of the Deeper Layers,” arXiv:2403.17887; ICLR 2025.**

SLEB iteratively removes the blocks whose deletion least increases calibration perplexity. Gromov et al. remove a contiguous span selected by angular distance and heal with QLoRA; they also show that simply removing the deepest non-final layers can work after healing.

**Overlap:** large-scale decoder depth removal and healing, often focusing on deeper blocks.

**Difference:** no new full cap and no permanently frozen prefix.

---

## A.3 Reinitializing upper layers

### Revisiting Few-sample BERT Fine-tuning

- **Tianyi Zhang et al., “Revisiting Few-sample BERT Fine-tuning,” arXiv:2006.05987; ICLR 2021.**

The arXiv identifier is from 2020 even though the venue is ICLR 2021.

It reinitializes BERT-Large’s pooler and top \(L\in\{1,\dots,6\}\) Transformer blocks, then performs ordinary full-model fine-tuning.

**Overlap:** top pretrained layers may be a poor downstream initialization; newly initialized upper Transformer blocks can adapt well.

**Your differentiators relative to it do hold:**

1. Their network is not shortened.
2. Their lower trunk is not permanently frozen.
3. Their primary setting is BERT-Large, not a 7B decoder.

But the paper makes “fresh upper layers can help adaptation” old. Your novelty cannot be merely reinitialization.

---

## A.4 Surgical Fine-Tuning

- **Yoonho Lee et al., “Surgical Fine-Tuning Improves Adaptation to Distribution Shifts,” arXiv:2210.11466; ICLR 2023.**

It freezes all but a selected contiguous block of **existing** parameters. Selection methods include cross-validation, relative gradient norm, and gradient signal-to-noise ratio.

**Overlap:** selective full-block tuning while the rest of the network is frozen.

**Actual distinction:** not that “Surgical FT still updates its selected blocks”—Paper C also updates its cap. The distinction is:

- Surgical FT retains the original architecture and trains selected **existing** layers.
- Paper C deletes layers, changes inference depth, and trains **new** layers.

Surgical FT is therefore a conceptual predecessor for the freezing argument but not an exact construction scoop.

---

## A.5 Block and partial freezing

Representative relevant work includes:

- **Yuhan Liu et al., “AutoFreeze: Automatically Freezing Model Blocks to Accelerate Fine-tuning,” arXiv:2102.01386, 2021 preprint.**
- **Howard and Ruder, “Universal Language Model Fine-tuning for Text Classification,” arXiv:1801.06146; ACL 2018**, for gradual unfreezing. This falls outside 2021–2026 but is historical background.
- **Lu et al., “Reassessing Layer Pruning in LLMs,” arXiv:2411.15558, 2024**, the direct decoder-LLM partial-layer recovery precedent.
- **Jian Gu et al., “A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models,” arXiv:2406.11753; Findings of ACL 2025.**
- **Guangyuan Shi et al., “Understanding Layer Significance in LLM Alignment,” arXiv:2410.17875, 2024/2025 preprint.**

The general idea that lower layers can remain frozen while a suffix adapts is thoroughly occupied. Paper C must obtain novelty from changing the architecture, not from freezing alone.

---

## A.6 Layer-selective PEFT

### LISA

- **Rui Pan et al., “LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning,” arXiv:2403.17919; NeurIPS 2024.**

LISA always trains embedding/head parameters and periodically unfreezes a randomly sampled subset of existing intermediate layers. It covers TinyLlama, Mistral-7B, Llama-2-7B, and Llama-2-70B.

**Overlap:** trains a small number of full-width LLM layers while freezing most of the model; compares favorably with LoRA.

**Difference:** all original layers remain at inference, and the active layers are periodically resampled rather than forming a new terminal cap.

Any statement that “training a few dense full layers rather than LoRA is novel” is untenable.

### Importance-aware and significance-aware PEFT

- **Kai Yao et al., “Layer-wise Importance Matters: Less Memory for Better Performance in Parameter-efficient Fine-tuning of Large Language Models,” arXiv:2410.11772; Findings of EMNLP 2024.**
- **Guangyuan Shi et al., “Understanding Layer Significance in LLM Alignment,” arXiv:2410.17875, preprint.**

These estimate important layers during or after adaptation and focus PEFT updates there.

**Overlap:** layer-selective parameter allocation and claims that only a subset of layers matters.

**Difference:** no depth reduction and no fresh cap.

### LayerNorm tuning

Two relevant papers:

- **Wang Qi et al., “Parameter-Efficient Tuning on Layer Normalization for Pre-trained Language Models,” arXiv:2211.08682, 2022 preprint.**
- **Taha ValizadehAslani and Hualou Liang, “LayerNorm: A Key Component in Parameter-Efficient Fine-Tuning,” arXiv:2403.20284, 2024 preprint.**

They freeze most or all pretrained weights and tune only LayerNorm parameters, with the latter also using Fisher information to choose subsets.

**Overlap:** Paper C trains final norm, and the papers reinforce that norm/head updates can account for a surprising fraction of adaptation.

**Required control:** final norm + `lm_head` only. Otherwise, part of the fresh-cap gain may simply be output calibration.

---

# B. Do the three claimed differentiators survive?

## Differentiator 1: “We discard \(L-j\) upper layers and regrow \(K<L-j\), so the model genuinely becomes shallower.”

### Broad form: **No**

Genuine decoder depth reduction is heavily occupied by ShortGPT, LaCo, SLEB, Gromov et al., Shortened LLaMA, Sheared-LLaMA, Minitron, Lu et al., Why Lift so Heavy?, and FinerCut.

Replacement with fewer modules is also occupied by LLM-Streamline and Siddiqui et al.

### Narrow form: **Possibly**

I found no exact prior that combines:

- complete suffix deletion,
- fewer newly random-initialized **full Transformer** blocks,
- permanent frozen prefix,
- downstream-only SFT.

That exact topology remains defensible, but only as a narrowly worded “to our knowledge” claim.

---

## Differentiator 2: “The trunk is genuinely frozen.”

### Broad form: **No**

This is directly occupied by:

- Lu et al. 2024 after tail pruning;
- LLM-Streamline during replacement training;
- LLaMA Pro;
- Llama SLayer;
- Growing Transformers;
- Grow, Don’t Overwrite;
- numerous adapters, LoRA, LayerNorm tuning, and surgical/partial-freezing methods.

### Important correction

The contrast with Surgical FT is currently misstated. “Surgical FT updates the selected blocks” is not a distinction. The actual contrast is:

> Surgical FT updates selected **pretrained** blocks in an intact network; Paper C updates **newly initialized** blocks in a shallower network.

---

## Differentiator 3: “Decoder-only LLM at 7B scale, not BERT.”

### Relative to Zhang et al. and Surgical FT: **Yes**

This distinguishes Paper C from the early BERT and vision precedents.

### Relative to the full 2021–2026 literature: **No**

Lu et al., LLM-Streamline, LLaMA Pro, Llama SLayer, LISA, SALF, ShortGPT, Gromov et al., Shortened LLaMA, Minitron, and Siddiqui et al. all operate on decoder LLMs around or above 7B.

---

## Combined verdict on the construction

The three advertised differentiators **do not hold collectively**. Lu et al. already has all three.

The remaining exact distinction is:

> a net-shallower **terminal** cap composed of fewer **new random full Transformer blocks**, trained for the downstream task while the retained prefix stays frozen.

That distinction is real but thin. It needs an empirical result demonstrating that the **freshness and cap topology themselves** matter.

The single mandatory baseline is:

\[
\begin{array}{ll}
\textbf{Fresh-cap:}& \text{keep }j + \text{random }K,\;\text{train random }K\\
\textbf{Surviving-cap:}& \text{keep }j+K,\;\text{train final pretrained }K\\
\textbf{Head-only:}& \text{keep }j+K,\;\text{train only norm/head}
\end{array}
\]

All must have the same final depth and almost the same trainable parameter count.

Without this comparison, the paper cannot isolate what is new relative to Lu et al.

---

# C. P-C2: a-priori prediction of \(j\) and \(K\)

## Verdict

**Broad P-C2 is already partially scooped. Exact joint architecture prediction remains plausibly open.**

The paper cannot claim:

- first forward-only layer selector;
- first training-free adaptation-depth selector;
- first use of base-model hidden states to decide where to fine-tune;
- first intrinsic probe for useful intermediate depth.

Several precedents contradict these formulations.

---

## C.1 The single closest precedent: Xie et al. 2022

- **Shuo Xie et al., “Hidden State Variability of Pretrained Language Models Can Guide Computation Reduction for Transfer Learning,” arXiv:2210.10041; Findings of EMNLP 2022.**

This paper explicitly asks whether, given an untouched pretrained LM and downstream task, one can determine which layers to adapt and where to place the classifier **without task-specific model tuning**.

It computes a forward-derived, label-conditioned hidden-state variability ratio:

- low within-class variability;
- high between-class variability.

The chosen layer guides:

- which layers are fine-tuned;
- where adapters are installed;
- where the classifier is attached;
- whether upper layers are removed from training and inference.

One reported configuration keeps only the bottom 14 RoBERTa-Large layers and matches full fine-tuning.

### Why this matters

This is already:

- pre-fine-tuning;
- task-specific;
- based on base-model representations;
- used to select adaptation depth;
- sometimes used to discard upper layers.

### Remaining difference

It does not:

- use a decoder-only generative LLM;
- add fresh Transformer blocks;
- predict how many blocks to regrow;
- jointly predict \((j,K)\);
- optimize a performance/compute objective over a regrown architecture.

This is the strongest defensible boundary for P-C2.

---

## C.2 SALF

- **Jian Gu et al., “A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models,” arXiv:2406.11753; Findings of ACL 2025.**

SALF obtains every intermediate representation in one forward pass, compares representations to label-derived semantic bases, computes the gain of each layer, and selects an “end-of-freezing” layer. Earlier layers are frozen; the existing suffix is trained.

Models include Qwen2-7B, Gemma2-9B, and Llama-3-8B.

### Overlap

- one forward pass;
- target-task examples and labels;
- decoder-only 7–9B LLMs;
- predicts a contiguous frozen-prefix boundary.

### Difference

- leaves the original suffix in place;
- does not select an inference architecture;
- does not predict \(K\);
- applies its selection per datum/budget during fine-tuning rather than necessarily committing once to a global \((j,K)\).

P-C2 therefore cannot broadly claim the first forward-only prediction of a freezing boundary.

---

## C.3 Data-oblivious CKA critical layers

- **Xuyuan Liu et al., “Spectral Insights into Data-Oblivious Critical Layers in Large Language Models,” arXiv:2506.00382; Findings of ACL 2025.**

This paper uses CKA on the pre-fine-tuned model to identify intrinsic change-point layers. It then shows that these layers are the ones most affected by fine-tuning and trains only five critical layers during Llama-2-7B-Chat adaptation on Dolly and OpenBookQA.

### Overlap

- no completed fine-tuning is needed to locate the layers;
- uses base-model representation geometry;
- decoder-only 7B;
- applies the signal to selective adaptation.

### Difference

- finds a sparse set of existing critical layers, not a contiguous cutoff;
- does not prune the network;
- fixes the number of layers;
- does not predict regrowth capacity.

This is a direct threat to the phrase “adaptation onset is visible in the base model.”

---

## C.4 Other relevant precedents

- **Zining Zhu et al., “Predicting Fine-Tuning Performance with Probing,” arXiv:2210.07352; EMNLP 2022.** Pretrained-model probing features predict later fine-tuning performance, although not architecture.
- **Mingyu Jin et al., “Exploring Concept Depth: How Large Language Models Acquire Knowledge and Concepts at Different Layers?,” arXiv:2404.07066; COLING 2025.** Simple concepts emerge at shallower layers and complex concepts at deeper layers; this anticipates the “computational depth” narrative.
- **Keith Ando Ogawa et al., “Layer-wise LoRA Fine-tuning: A Similarity Metric Approach,” arXiv:2602.05988, 2026 preprint.** Uses adjacent-layer CKA dissimilarity before fine-tuning to select existing layers for LoRA in decoder 7B models; the number of selected layers remains fixed externally.
- **Aldrin Kabya Biswas et al., “PRiSM: Partial Ranking via Inter-layer Semantic Measurement for Efficient Fine-tuning of Language Models,” LREC 2026; DOI 10.63317/3eyz8rr5qun6.** Describes a training-free single-forward-pass selector for layers to fine-tune. I could not find an arXiv identifier, so I cite the venue rather than inventing one.
- **Qinghua Zhao et al., “A Layer-wise Analysis of Supervised Fine-Tuning,” arXiv:2604.11838, 2026 preprint.** Analyzes layer-wise emergence and proposes selective tuning, but leaves automatic training-free boundary detection as future work.
- **Yichen Xu et al., “Understanding and Guiding Layer Placement in Parameter-Efficient Fine-Tuning of Large Language Models,” arXiv:2602.04019, 2026 preprint.** Uses gradients, activation conditioning, and inter-layer coupling to guide adapter placement. It is not forward-only because its layer cards include target-loss gradients and validation performance.

---

## C.5 What pruning literature normally optimizes

The bulk of pruning work estimates **damage to the current pretrained model after deletion**, not future downstream adaptability.

### Gradient, Fisher, Taylor, and curvature

- LLM-Pruner: gradient/Taylor saliency.
- Surgical FT: relative gradient norm and gradient SNR.
- LayerNorm tuning: Fisher information.
- **Yannic van der Ouderaa et al., “The LLM Surgeon,” arXiv:2312.17244; ICLR 2024:** curvature/Fisher-style compression.
- ILA and IST: importance learned during adaptation.

These require backward passes or an adaptation trajectory.

### Perplexity or loss after ablation

- Shortened LLaMA;
- SLEB;
- Minitron;
- BlockPruner;
- Gromov et al. for validation loss analyses.

These simulate removal and measure post-ablation LM damage.

### Representation similarity and block influence

- ShortGPT;
- LLM-Streamline;
- LaCo;
- Gromov et al.;
- FinerCut;
- “What Matters in Transformers? Not All Attention is Needed,” arXiv:2406.15786.

These generally identify blocks that currently make small representational changes.

### Task performance after ablation

- Minitron’s Winogrande sensitivity;
- Llama SLayer’s layer-removal/insertion tests;
- Shapley-style task-specific pruning.

These directly evaluate candidate damage and therefore are not cheap a-priori predictors in the proposed sense.

---

## C.6 What remains genuinely open

The strongest still-defensible P-C2 claim is:

> From forward probes of the intact pretrained decoder, before any parameter update or candidate-architecture fine-tuning, predict the compute-optimal architecture of a **different, net-shallower adapted model**, jointly choosing frozen-prefix depth \(j\) and newly grown capacity \(K\).

I found no verified precedent that predicts:

- the number of newly grown full blocks;
- joint \((j,K)\);
- the best regrown architecture rather than the best existing layers;
- selection regret over an empirically measured \(j\times K\) surface.

That is plausibly novel.

### Evidence required

Correlation alone is not enough. Report:

\[
R=
S(j^\*,K^\*)-
S(\hat j,\hat K),
\]

where \(S\) is a clearly specified performance/compute utility and \((j^\*,K^\*)\) is the oracle architecture.

Required baselines:

1. fixed depth fraction;
2. random cut;
3. ShortGPT block influence;
4. Gromov/LLM-Streamline endpoint cosine;
5. adjacent-layer CKA;
6. Xie hidden-state variability;
7. SALF semantic gain;
8. calibration perplexity;
9. gradient/Fisher selection as an expensive oracle;
10. standard-FT \(\Delta W\)/CKA drift as a post-hoc upper bound.

The predictor must be evaluated out of task, and preferably out of model family. Fitting and evaluating a mapping on the same handful of tasks will not establish predictive novelty.

---

# D. Consequences of finding #132

## The result must be treated as central, not as a caveat

The decisive facts are:

- both 16-layer arms are at or near chance on capability benchmarks;
- A4-over-A3 shrinks from +3.25 pp SQuAD EM to **+0.39 pp pooled accuracy**;
- 9 of 14 benchmark cells are null;
- A4 is worse on BoolQ;
- closed-book QA collapses to approximately zero EM;
- A4 repeatedly emits the same Chinese refusal on more than half of PopQA and NQ-open;
- A4 MMLU is only marginally above random-choice accuracy;
- the intact pretrained model substantially outperforms the task-adapted models on general capabilities.

This is not “some capability degradation.” It means the keep14+fresh2 system is **not a functioning general-purpose language model**.

## What the SQuAD result actually establishes

The statistically clean A4>A3 comparison shows:

> Given an aggressively truncated, severely capacity-limited 16-layer architecture and an overtrained SQuAD recovery protocol, an inherited frozen prefix provides a better starting representation than an entirely random same-depth network.

That is valid. It does **not** show:

- capability retention;
- knowledge preservation;
- reasoning recovery;
- useful general adaptation;
- parity with full fine-tuning;
- superiority over LoRA;
- that fine-tuning “mainly occurs at the top” in a functionally sufficient sense.

The LoRA arm beats A4 by roughly 36.6 EM points on SQuAD. The intact unfine-tuned base also beats keep14 A4, despite the confound. There is no current evidence of a competitive method.

## Is P-C1 still publishable at a top venue?

### In its current positive form: **No**

The proposed P-C1 says that because standard fine-tuning drift is concentrated near the top, replacing the upper suffix with a small trainable cap should match full fine-tuning and outperform parameter-matched LoRA.

Your current results contradict this:

- A4 does not match full fine-tuning.
- It badly loses to LoRA.
- It does not retain knowledge or reasoning.
- The apparent inherited-prefix benefit is largely confined to two broken systems.
- The literature already reports selective-layer adaptation, tail pruning plus partial-layer SFT, and frozen replacement modules.

This is insufficient for ACL/NeurIPS as a positive method paper.

### In a narrow diagnostic form: **Possibly, but not yet top-tier**

A paper about “initialization advantages under extreme truncation” is scientifically legitimate, but a +0.39 pp pooled capability difference between near-chance models is too weak by itself. It would need:

- multiple model families;
- a controlled phase-transition characterization;
- mechanistic diagnostics;
- explicit comparison against surviving pretrained layers and function-preserving cap initialization;
- a general finding beyond SQuAD.

### Stronger interpretation

The negative result suggests that:

\[
\text{small standard-FT drift}
\;\not\Rightarrow\;
\text{layer dispensability}.
\]

A layer can change little during downstream fine-tuning yet remain indispensable because it performs pretrained computations that fine-tuning reuses. This is a much more interesting scientific result than the original P-C1.

The current premise conflates:

1. **adaptation locality**—where parameters change under fine-tuning; and
2. **functional necessity**—which pretrained computations must remain present.

Finding #132 is evidence that these are not equivalent.

This is the best conceptual lesson in the data.

## Interaction with recent prior art

The 2026 preprint

- **Safal Shrestha et al., “On the Limits of Layer Pruning for Generative Reasoning in Large Language Models,” arXiv:2602.01997**

shows that pruning can preserve multiple-choice classification much better than generative reasoning, while arithmetic, syntax, and algorithmic capabilities remain damaged even after recovery and large-scale distillation.

That paper substantially occupies a generic “depth pruning hurts generative reasoning” reframe. You therefore need a more specific angle—e.g. the failure of **drift-based dispensability**, or phase transitions unique to frozen-cap recovery.

---

# E. Ranked reframings and new angles

Ranking considers both **novelty and feasibility with the infrastructure in the brief**.

---

## 1. Predict the compute-optimal regrown architecture before training

### Proposed paper

**“Predicting the Architecture of Downstream-Adapted LLMs from the Unmodified Base Model”**

### Core question

Can forward-only base-model probes predict the best \((j,K)\) under a performance/latency budget, avoiding an expensive architecture sweep?

### Why it remains novel

Xie, SALF, PRiSM, CKA selection, and layer-selective LoRA choose which **existing** layers to use or tune. They do not predict the architecture of a new shallower model with newly grown full blocks, nor jointly predict \(j\) and \(K\).

### Experiments

Use the existing probe and training infrastructure:

- models: OLMo-2-7B, Qwen3-8B, OLMo-2-1B, Qwen3-4B;
- probes: linguistic edge probes, logit lens, knowledge logit lens, CKA, block influence;
- grid: \(j\in\{14,18,20,22,24,26,28\}\), \(K\in\{0,1,2,4\}\), subject to compute;
- tasks covering shallow classification, extractive QA, generation, knowledge, and reasoning;
- outcome: selection regret and Pareto-front recovery, not only correlation.

### Necessary discipline

- Predictor must commit before candidate training.
- Hold out entire tasks and at least one model family.
- \(K\) must vary; fixing \(K=2\) destroys the strongest novelty.
- Include Lu-style surviving-cap architectures in the target search space, not only fresh caps.

### Rating

**Novelty: 8.5/10. Feasibility: 7/10. Overall rank: #1.**

---

## 2. Adaptation locality is not architectural dispensability

### Proposed paper

**“Fine-Tuning Drift Does Not Identify Dispensable Layers in Large Language Models”**

### Thesis

Layers with small \(\|\Delta W\|\) or CKA drift during standard fine-tuning may still be essential pretrained computational operators. Removing them and asking a small fresh cap to relearn their function causes catastrophic language-model and generative failure.

This directly converts finding #132 from embarrassment into the scientific result.

### Why it is interesting

Many papers implicitly move from “these layers change little” to “these layers can be frozen, skipped, or pruned.” Your data suggest the first implication—freezing—can hold while the second—removal—fails.

That distinction is not the same as the generic observation that pruning hurts reasoning.

### Decisive experiment matrix

For the same final depth and trainable budget:

1. intact 32L, frozen prefix + tune existing suffix;
2. 24/28L tail-pruned, tune last existing \(K\);
3. same depth, replace last \(K\) with random blocks;
4. same depth, copy/average/function-preserving cap;
5. freeze low-drift layers in place without deleting them;
6. delete low-drift layers;
7. delete high-drift layers;
8. full FT and LoRA controls.

Then compare:

- \(\Delta W\);
- CKA drift;
- deletion sensitivity;
- single-layer ablation loss;
- PPL;
- closed-book generation;
- MC capability;
- task performance.

### Central test

Does standard-FT drift predict:

- best layers to freeze?
- best layers to remove?
- best cut depth?
- capacity required for recovery?

A result that it predicts freezing but not deletion would be conceptually strong.

### Rating

**Novelty: 8/10. Feasibility: 8/10. Overall rank: #2.**

---

## 3. The survival boundary of frozen-cap recovery

### Proposed paper

**“How Shallow Can a Frozen-Trunk LLM Go? Phase Transitions in Task Recovery and Generative Collapse”**

### Core question

Where is the boundary between:

1. usable compressed adaptation;
2. task-only pattern fitting;
3. globally corrupted generation?

Your current data already hint at this:

- keep14+2: broken;
- keep20+2: SQuAD improves;
- keep24+2: improves further;
- keep28+2: reaches 0.419 EM.

### Why potentially novel

Generic depth-pruning phase transitions are known. What is less occupied is a controlled phase diagram for **frozen-prefix + trainable terminal cap**, separating:

- local supervised-task recovery;
- intact LM fluency/PPL;
- parametric knowledge;
- multiple-choice capability;
- free generation;
- refusal/repetition pathologies.

### Experiments

Across \(j\), \(K\), model family, and training budget, measure:

- validation SFT score;
- generic-text PPL;
- capability average;
- closed-book QA;
- repetition/refusal rate;
- KL from the base model;
- next-token entropy;
- lm-head norm and logit scale;
- CKA at the trunk-cap interface;
- cap attention entropy;
- speed and memory.

Identify whether there is a sharp critical \(j/L\), and whether the existing base probes predict it.

### Literature boundary

Differentiate explicitly from **“On the Limits of Layer Pruning for Generative Reasoning,” arXiv:2602.01997**: your object is constrained frozen-cap adaptation and the interface/cap failure boundary, not generic pruning plus QLoRA recovery.

### Rating

**Novelty: 7/10. Feasibility: 9/10. Overall rank: #3.**

---

## 4. Freshness versus preservation: what should replace a deleted suffix?

### Proposed paper

**“What Should Replace a Pruned LLM Suffix? Random Regrowth, Surviving Blocks, or Function-Preserving Compression”**

### Core comparison

At the same final depth and training budget:

1. fresh random full blocks;
2. final \(K\) surviving pretrained blocks, Lu-style;
3. copy of the first discarded block;
4. copy of the final retained block;
5. average of discarded blocks;
6. identity/zero-output initialization, LLaMA-Pro style;
7. LLM-Streamline-style FFN/SwiGLU bridge;
8. low-rank stitching;
9. no cap, norm/head only.

### Why it matters

This is the experiment required to establish whether the exact remaining architectural novelty has any scientific value.

Possible interesting outcomes:

- random blocks outperform inherited blocks under large distribution shift;
- function-preserving initializations dominate random blocks in capability retention;
- fresh blocks improve target fitting but cause greater forgetting;
- surviving blocks dominate everywhere, falsifying the proposed construction.

Even a negative result could be useful if the comparison is comprehensive.

### Risk

LLM-Streamline already covers several replacement architectures and initializations under generic LM recovery. Therefore the novelty must be explicitly about **downstream task adaptation, terminal suffix replacement, and the adaptation-versus-retention trade-off**.

### Rating

**Novelty: 6.5/10. Feasibility: 8.5/10. Overall rank: #4.**

---

## 5. Task difficulty predicts required adaptation capacity—but not necessarily depth alone

### Proposed paper

**“From Linguistic Onset to Adaptation Capacity: Which Tasks Need How Much Trainable Decoder?”**

### Core question

Do linguistic, knowledge, and reasoning probes predict:

- minimum viable retained depth;
- minimum cap size;
- recovery data requirement;
- whether the task is learnable through a frozen trunk at all?

### Why it could work

The brief already has:

- POS, dependency, CoLA, WiC, SST-2, RTE probes;
- knowledge logit lens;
- multiple model families;
- a working training script with variable \(j,K\).

### Needed distinction

Concept depth itself is occupied by:

- **Jin et al., “Exploring Concept Depth…,” arXiv:2404.07066; COLING 2025.**

The novel object must be **predictive capacity allocation**, not another layerwise probing plot.

### Strong version

Construct a taxonomy:

- readout tasks need only head/norm;
- shallow cap tasks need \(K=1\);
- compositional tasks need larger \(K\);
- knowledge tasks require deeper retained trunks rather than larger caps;
- generative reasoning cannot be repaired under a frozen-trunk constraint.

### Rating

**Novelty: 6.5/10. Feasibility: 7.5/10. Overall rank: #5.**

---

# Mandatory empirical additions regardless of reframe

## Construction baselines

1. Full fine-tuning.
2. Parameter-matched LoRA.
3. `lm_head` only.
4. final norm + `lm_head`.
5. Keep \(j+K\) pretrained blocks, freeze first \(j\), train last \(K\).
6. Keep \(j+K\), freeze every Transformer block, train norm/head.
7. Keep \(j\) + random \(K\).
8. Keep \(j\) + copied/averaged/identity \(K\).
9. Keep \(j\) + one FFN/SwiGLU bridge.
10. Same final depth, all-parameter recovery where memory permits.
11. Same-depth from-scratch control.

## P-C2 baselines

1. fixed 50%, 75%, and 87.5% depth;
2. random depth;
3. reverse-order rule;
4. block influence;
5. endpoint cosine/angular distance;
6. adjacent CKA;
7. Xie variability ratio;
8. SALF semantic gain;
9. calibration PPL;
10. gradient/Fisher and observed \(\Delta W\) as costly oracle baselines.

## Evaluation

Do not permit multiple-choice accuracy to stand in for retained language-model capability. Every configuration needs:

- generic-text PPL;
- MC capability;
- free generation;
- closed-book QA;
- repetition/refusal diagnostics;
- target-task score;
- latency/throughput;
- trainable parameters and peak memory.

The current finding that near-chance multiple-choice models can still show statistically significant differences is exactly why absolute competence floors must be reported.

---

# Claims that should be deleted or rewritten

## Unsafe

- “Fine-tuning primarily happens in the top layers.”
- “Our construction is the first genuinely shallower frozen-trunk decoder LLM.”
- “Unlike prior pruning methods, we train only a small replacement.”
- “Unlike Surgical FT, our trunk is frozen.”
- “This is the first forward-only predictor of adaptation depth.”
- “A4 recovers knowledge or reasoning.”
- “A4 approaches full fine-tuning.”
- “A4 outperforms parameter-matched LoRA.”
- “The SQuAD A4>A3 result demonstrates general capability retention.”

## Safer

> We study task-conditioned terminal regrowth: a pretrained decoder suffix is replaced with fewer newly initialized full Transformer blocks, while the inherited prefix remains frozen.

> Existing work separately studies tail pruning followed by tuning surviving layers, compression-oriented replacement of internal pruned spans, and block expansion over frozen backbones. We isolate whether a newly learned terminal cap is a useful downstream adaptation substrate.

> We distinguish adaptation locality from architectural dispensability: layers that change little during fine-tuning may nevertheless remain essential to the pretrained computation.

> We investigate whether base-model probes can predict the compute-optimal architecture of a shallower adapted model, rather than merely selecting existing layers to update.

---

# Final area-chair recommendation

## Scoop decision

### Broad claim

**Scooped.**

Primary scooping work:

- **Yao Lu et al., “Reassessing Layer Pruning in LLMs: New Insights and Methods,” arXiv:2411.15558, 2024.**

It directly invalidates the proposed three-way distinction:

- real shallowing;
- frozen trunk;
- decoder-only 7B/8B.

### Replacement mechanism

**Heavily anticipated.**

Closest works:

- **Xiaodong Chen et al., “Streamlining Redundant Layers to Compress Large Language Models,” arXiv:2403.19135; ICLR 2025.**
- **Shoaib Ahmed Siddiqui et al., “A Deeper Look at Depth Pruning of LLMs,” arXiv:2407.16286, 2024 preprint.**
- **Tianxiang Chen et al., “Llama SLayer 8B: Shallow Layers Hold the Key to Knowledge Injection,” arXiv:2410.02330; Findings of EMNLP 2024.**
- **Chengyue Wu et al., “LLaMA Pro: Progressive LLaMA with Block Expansion,” arXiv:2401.02415, 2024 preprint.**

### Exact construction

**Not fully scooped in the literature I could verify.**

The unscooped residue is:

- terminal suffix deletion;
- fewer random full Transformer blocks;
- permanent frozen prefix;
- task-only SFT;
- net depth reduction.

This is too narrow to carry the paper unless random regrowth demonstrates a clear advantage over surviving pretrained layers and lightweight/function-preserving replacements.

### P-C2

**Partially scooped, exact form plausibly open.**

Closest works:

- **Shuo Xie et al., “Hidden State Variability of Pretrained Language Models Can Guide Computation Reduction for Transfer Learning,” arXiv:2210.10041; Findings of EMNLP 2022.**
- **Jian Gu et al., “A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models,” arXiv:2406.11753; Findings of ACL 2025.**
- **Xuyuan Liu et al., “Spectral Insights into Data-Oblivious Critical Layers in Large Language Models,” arXiv:2506.00382; Findings of ACL 2025.**
- **Keith Ando Ogawa et al., “Layer-wise LoRA Fine-tuning: A Similarity Metric Approach,” arXiv:2602.05988, 2026 preprint.**
- **Aldrin Kabya Biswas et al., “PRiSM: Partial Ranking via Inter-layer Semantic Measurement for Efficient Fine-tuning of Language Models,” LREC 2026; DOI 10.63317/3eyz8rr5qun6.**

What remains open is predicting the architecture of a **new regrown shallow model**, particularly joint \((j,K)\), rather than selecting locations in the intact model.

## Publishability of current P-C1

**Not publishable at ACL/NeurIPS as presently framed.**

The construction performs poorly in absolute terms, loses badly to LoRA, destroys closed-book knowledge and generation, and shows only a small inherited-initialization advantage between two broken 16-layer systems.

The paper must be reframed.

## Best path

My recommended order is:

1. **P-C2 as pre-training-free architecture selection**, with real joint \((j,K)\) prediction and held-out selection-regret evaluation.
2. **Adaptation locality is not architectural dispensability**, using finding #132 as the central discovery.
3. **Phase transitions and failure boundaries of frozen-cap recovery.**
4. Replacement-initialization study as the supporting mechanistic section.

If the deeper keep24/28 configurations enter a genuinely functional regime and the probe predicts that boundary across OLMo and Qwen, there is a potentially strong paper. If not, the honest paper is a negative one about why fine-tuning drift does not imply that pretrained layers can be deleted and relearned by a tiny cap.