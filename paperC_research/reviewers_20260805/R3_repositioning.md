# Research report

**Web/internet access: YES.** I read `paperC_BRIEF_for_research.md` in full, inspected the relevant local Paper B/C results and scripts, and searched primary literature available through **August 5, 2026**. Citations below use verified paper titles plus an arXiv ID or venue; where I did not verify a venue, I cite only the arXiv ID.

## Executive verdict

The paper worth writing is **not**:

> “A shallow randomly regrown cap on a frozen trunk can match conventional fine-tuning.”

Your own #132 result has already killed the broad version of that claim. The controlled SQuAD difference is real, but it is largely a comparison between two severely impaired models, and the architectural neighborhood is crowded by LLM-Streamline and related pruning/recovery work.

The strongest pivot is:

> # **Preserve Here, Adapt There: Causal Depth Profiles for Knowledge-Preserving LLM Adaptation**
>
> The pretrained computations that must be preserved for factual knowledge and the layers that are most useful for downstream adaptation are not necessarily the same. Base-model depth diagnostics can prospectively allocate a compact model’s limited depth between preservation and adaptation.

This uses P-C2 as a **prospective architecture-selection result**, while turning your catastrophic models into controlled lesions rather than failed compressed models.

A standalone P-C2 paper remains viable only if it becomes a **preregistered quantitative depth law**, not a post-hoc correlation:

\[
K^* \approx \left\lceil \frac{[d_t-j+\delta]_+}{\gamma}\right\rceil.
\]

The present “format SFT succeeded while knowledge disappeared” story is **not established by the current evidence**. In particular, \(F1-EM\approx0\) is not a format-compliance metric.

---

## Three corrections that should govern any pivot

### 1. Call the probe quantity **task linearization depth**, not adaptation onset or storage depth

The OLMo knowledge logit lens has a sharp jump from MMLU accuracy .326 at hidden depth 18 to .544 at depth 19; Qwen3-8B jumps from approximately .236 at depth 24 to .621 at depth 25. These are striking transitions in **readout compatibility**.

They do not establish where knowledge is stored. They show where a particular final norm/LM-head interface can decode it.

That distinction is especially important because the current full-FT CKA curve in your local results does **not** show a knee at OLMo L18. It declines smoothly and accelerates only near the final layers. Thus:

- knowledge-readout onset;
- representation drift under SFT;
- causal knowledge necessity;
- optimal adaptation location

are empirically different quantities until proven otherwise.

### 2. The current A4 treatment is not literally a wholly frozen trunk

The front decoder blocks are frozen, but the inherited embedding, final normalization, and LM head are trainable. The trainable embedding also causes small changes to the outputs of nominally frozen blocks.

For accurate writing, call it:

> **a frozen decoder prefix with trainable inherited lexical/readout modules and a fresh cap**

unless you rerun a strict variant with the embedding frozen as well.

### 3. The SQuAD data have a major refusal-prior confound

A direct local audit shows:

- training set: **1,756 / 10,000 = 17.56%** targets are the same Chinese refusal;
- validation set: **997 / 2,000 = 49.85%** targets are that refusal;
- the string is exactly the one that dominates A4’s PopQA/NQ-open outputs.

Therefore, the 52% identical closed-book refusal rate is plausibly a learned response prior interacting with decoder damage. It is not evidence, by itself, that the model successfully learned a general answer format.

---

# (a) Steelman P-C2

## The strongest possible claim

The strongest scientifically meaningful P-C2 is:

> **Computational-depth conservation:** a task has an intrinsic base-model linearization depth. Cutting the model before that depth creates a measurable computational deficit, and the minimum fresh-cap depth needed for adaptation is approximately the missing effective depth. This architecture can be predicted before any downstream backbone training.

Let \(q_{m,t}(\ell)\) be the chance-normalized frozen-base probe score for model \(m\), task \(t\), and depth \(\ell\). Rather than relying on an arbitrary onset threshold, fit a transition curve:

\[
q_{m,t}(\ell)
\approx
\sigma\left(\frac{\ell-\mu_{m,t}}{w_{m,t}}\right),
\]

where:

- \(\mu_{m,t}\): task linearization midpoint;
- \(w_{m,t}\): transition width;
- a small \(w\) implies a sharp depth cliff;
- a large \(w\) implies a broad range of nearly interchangeable depths.

For a frozen cut \(j\), define the observed minimum useful cap:

\[
K^*_{m,t,j}(\eta)
=
\min\left\{
K:
R_{m,t}(j,K)\geq \eta
\right\},
\]

where \(R\) is floor-to-best normalized target performance, \(\eta\) is fixed in advance—e.g. \(0.95\)—and configurations must also pass language-model health gates.

The strongest law is:

\[
j+\gamma K^*_{m,t,j}
\approx
\mu_{m,t}+\delta.
\]

Equivalently:

\[
K^*_{m,t,j}
\approx
\left\lceil
\frac{[\mu_{m,t}-j+\delta]_+}{\gamma}
\right\rceil.
\]

Interpretation:

- \(\gamma=1\): one fresh block replaces approximately one missing base block;
- \(\gamma<1\): fresh task-trained blocks are less depth-efficient;
- \(\gamma>1\): task specialization lets one cap block replace several generic pretrained blocks;
- \(\delta\): global readout/safety margin.

A still stronger curve-collapse claim is:

\[
R_{m,t}(j,K)
\approx
F\left(
\frac{j+\gamma K-\mu_{m,t}}{w_{m,t}}
\right).
\]

If configurations from different tasks and model families collapse onto one response curve \(F\), this is a scaling-law-style result rather than curve fitting per task.

## What the headline figure should look like

### Main panel: universal depth-deficit curve

- **x-axis:** normalized effective-depth surplus

  \[
  x =
  \frac{j+\hat\gamma K-\mu_{m,t}}{w_{m,t}}.
  \]

- **y-axis:** normalized held-out target performance

  \[
  R =
  \frac{S-S_{\rm floor}}
       {S_{\rm sweep\ best}-S_{\rm floor}}.
  \]

- **Each point:** one `(model, task, cut j, cap K, training seed)` configuration.
- **Color:** model family.
- **Marker shape:** task family.
- **Filled markers:** completely held-out model/task families.
- **Curve:** one globally fitted sigmoid, fitted only on development tasks/models.
- **Vertical line at zero:** predicted viability boundary.

A good result would show that tasks with very different raw performance collapse when plotted relative to their base-model linearization deficit.

### Second panel: prospective prediction

- **x-axis:** predicted minimum cap depth \(\widehat K^*\).
- **y-axis:** observed minimum cap depth \(K^*\) from the later exhaustive sweep.
- **Each point:** one held-out `(model, task, cut j)` tuple.
- Diagonal: exact prediction.
- Error bars: seed-level uncertainty in the observed boundary.

Report:

- mean/median absolute layer error;
- percentage within \(\pm1\) layer;
- performance regret;
- GPU-hours saved relative to exhaustive search.

### Mechanism panel

Use either a matrix or arrows:

- **x-axis:** omitted original base layers \(j+1,\ldots,L\);
- **y-axis:** cap layers \(1,\ldots,K\);
- **cell value:** CKA, linear-map predictability, or causal substitution from activation patching.

A near-diagonal mapping—cap layer \(k\) resembling or substituting for original layer \(j+k\)—would support actual computation reconstruction. If this panel is absent, the result is still useful architecture prediction, but the “depth conservation” mechanism remains speculative.

## Mechanistic story

The correct story is **progressive linearization**, not “information first appears at layer \(d\).”

1. Early representations may contain task-relevant information in a nonlinear or distributed form.
2. Successive layers progressively transform that information into a geometry accessible to a simple readout.
3. Cutting at \(j<d_t\) removes part of this transformation.
4. Fresh cap layers learn the missing transformations from the frozen residual stream.
5. If \(j\geq d_t\), most of the necessary task representation is already linearly accessible and only a small cap/readout is needed.

This mechanism makes causal predictions beyond correlation:

- **Translation prediction:** increasing \(j\) by \(\Delta\) should reduce \(K^*\) by approximately \(\Delta/\gamma\), holding task and data fixed.
- **Activation-graft prediction:** injecting a base activation from \(j+r\) should reduce the required cap depth by approximately \(r/\gamma\).
- **Transition-width prediction:** tasks with narrow probe transitions should have sharp viability cliffs; broad transitions should have several near-optimal \(K\)'s.
- **Probe-capacity prediction:** a nonlinear probe may decode information earlier, but the linear-probe depth should better predict the depth needed by a shallow linearizing cap.
- **Format invariance:** semantically equivalent prompt formats should give similar \(\mu_t\); a probe that mainly detects dataset format should not predict held-out cap depth.

## Related precedent and novelty boundary

| Literature | Closest papers | What is already occupied | What remains for you |
|---|---|---|---|
| Layerwise decodability | **BERT Rediscovers the Classical NLP Pipeline** (ACL 2019; arXiv:1905.05950); **Linguistic Knowledge and Transferability of Contextual Representations** (NAACL 2019; arXiv:1903.08855) | Linguistic/task information has different layerwise profiles | Predicting a structural cut-and-regrow architecture from those profiles |
| Logit/tuned lenses | **Eliciting Latent Predictions from Transformers with the Tuned Lens** (arXiv:2303.08112); **The Remarkable Robustness of LLMs: Stages of Inference?** (arXiv:2406.19384); **How Do LLMs Use Their Depth?** (arXiv:2510.18871) | Intermediate prediction refinement and effective inference stages | Linking base inference depth quantitatively to required adaptation depth |
| Intrinsic dimension | **Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning** (arXiv:2012.13255); **GeLoRA** (arXiv:2412.09250); **Less Is More: Local Intrinsic Dimensions of Contextual Language Models** (NeurIPS 2025; arXiv:2506.01034) | How many adaptation degrees of freedom are needed | Where those degrees of freedom must occur in depth |
| Early exit/effective depth | **DeeBERT** (ACL 2020; arXiv:2004.12993); **BERT Loses Patience** (NeurIPS 2020; arXiv:2006.04152); **LayerSkip** (ACL 2024; arXiv:2404.16710) | Predicting when an intact model can exit | Predicting how many new layers a cut model must learn |
| Fine-tuning prediction | **Predicting Fine-Tuning Performance with Probing** (EMNLP 2022; arXiv:2210.07352); **Selecting Large Language Model to Fine-tune via Rectified Scaling Law** (arXiv:2402.02314); **Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?** (arXiv:2504.12491); **TuneAhead** (ICML 2026; arXiv:2606.17660) | Predicting model/run performance | Predicting a discrete architectural intervention before training |
| Representation-guided placement | **Hidden State Variability of Pretrained Language Models Can Guide Computation Reduction for Transfer Learning** (EMNLP 2022; arXiv:2210.10041); **Understanding and Guiding Layer Placement in PEFT** (arXiv:2602.04019); **RSRA** (arXiv:2607.09757); **Dominant-Layer ZO** (arXiv:2606.05516); **RDP LoRA** (arXiv:2604.19321) | Forward-only or cheap signals already choose layers/ranks to adapt | A held-out quantitative law for **cut depth plus fresh cap depth**, with low architecture-search regret |

The last row is the crucial novelty warning. By August 2026, “a forward-only base-model signal tells us where to adapt” is no longer novel by itself.

Your defensible novelty is:

> A preregistered law predicting the minimum viable **shallower architecture**, validated on held-out task and model families, with a causal depth-substitution test.

Also, the construction itself is substantially occupied:

- **Streamlining Redundant Layers to Compress Large Language Models / LLM-Streamline** (arXiv:2403.19135; ICLR 2025 Spotlight) prunes consecutive blocks, inserts/train a lightweight replacement, and freezes the remaining model.
- **Reassessing Layer Pruning in LLMs: New Insights and Methods** (arXiv:2411.15558; venue not verified) reports pruning the final 25% and fine-tuning the LM head plus the remaining final three layers.

So P-C1 cannot carry novelty.

---

# (b) The sharper and hardest-to-dismiss P-C2

## Recommended confirmatory hypothesis

Use all current OLMo/SQuAD results as **pilot data only**.

Before inspecting any new held-out adaptation sweep, preregister:

1. exact probe statistic and probe capacity;
2. prompt templates;
3. transition-fitting method;
4. \(j\) and \(K\) grids;
5. score normalization;
6. language-model health gates;
7. the formula for \(\widehat K^*\);
8. development versus held-out task/model families;
9. stopping rules and seeds;
10. success and falsification thresholds.

Commit a machine-readable prediction file with its hash before launching held-out runs.

## Strong primary claim

> On a held-out model family and held-out task family, the preregistered predictor selects a cap reaching at least 95% of the sweep-best valid performance, with median regret at most 1 absolute point and cap-depth error at most one layer on at least 80% of model–task–cut tuples.

These should be declared as proposed acceptance criteria, not retrofitted after results.

## Double holdout

A strong split would be:

- **Development models:** OLMo-2-1B and OLMo-2-7B.
- **Model holdout:** Qwen3-4B, which you already possess but have not used to fit the law.
- **Development task families:** linguistic classification and semantic classification.
- **Task-family holdout:** contextual QA and one reasoning/code family.

Qwen3-8B can serve as a secondary test, but because it already appears in your probe and pruning work, Qwen3-4B is a cleaner confirmatory holdout.

Target approximately 12–18 genuinely held-out `(task, cut)` tuples, not four points and a correlation coefficient.

## Strongest within-task test

For each task, use at least three cuts and test:

\[
K^*(j+\Delta)-K^*(j)
\approx
-\frac{\Delta}{\gamma}.
\]

This is much harder to dismiss than a cross-task correlation because it holds task difficulty, data size, label entropy, and evaluation protocol constant.

Under the strongest one-for-one conservation claim:

\[
K^*(j+1)-K^*(j)\approx -1.
\]

If the relationship is flat, positive, or highly task-specific, the mechanistic depth law is false even if a pooled correlation happens to be positive.

## Baselines the probe must beat

- fixed \(K=2\);
- fixed total depth \(j+K\);
- random \(K\);
- dataset size or number of training examples;
- base zero-shot task score;
- final-layer probe accuracy;
- probe AUC rather than probe depth;
- intrinsic/effective rank;
- ShortGPT Block Influence;
- layer activation norm or activation-outlier depth;
- Layer Card / representation-sensitivity predictors where implementable;
- a one-step gradient/Fisher oracle as a more expensive upper baseline.

## Probe-validity controls

1. Ordinary logit lens versus tuned lens.
2. Linear probe versus shallow MLP probe.
3. Random-label control.
4. Multiple semantically equivalent prompt formats.
5. A dataset/source-format classifier.
6. Calibration examples disjoint from SFT and evaluation.
7. Report that fitting a probe head is **not** fully training-free; call it a frozen-backbone, pre-adaptation diagnostic.
8. Standardize all depths as “number of completed transformer blocks,” avoiding hidden-state-array off-by-one ambiguity.

A highly relevant warning is **Linear Probes Detect Task Format, Not Reasoning Mode in Language Model Hidden States** (ACL TrustNLP Workshop 2026; arXiv:2606.02907).

## Statistical unit and seeds

- Training seed—not benchmark item—must be the top-level unit.
- Use at least three seeds for predicted and boundary configurations.
- Use item-paired inference within each seed, then hierarchical or seed-level intervals.
- Do not let huge benchmark \(n\) convert a practically useless \(0.4\)-point difference into the headline.

## Health gates

A configuration that produces polluted language-model behavior is a failed architecture, not missing data. Predeclare gates such as:

- held-out PPL not in the catastrophic regime;
- parseable output rate;
- unique-output ratio and top-response frequency;
- refusal concentration;
- no single output dominating an ordinary non-refusal task;
- target-task performance above a substantive, not merely statistical, floor.

In particular, a configuration with PPL \(>100\) or severe single-string mode concentration should not be allowed to “win” because of one narrow task metric.

## Decisive falsifiers

P-C2 should be considered falsified if:

- fixed \(K=2\) matches its held-out regret;
- \(K^*\) does not decrease as \(j\) increases;
- the law needs task-specific coefficients to work;
- the held-out model requires refitting \(\gamma\) or \(\delta\);
- final base accuracy predicts \(K^*\) as well as depth;
- prompt-format depth predicts \(K^*\) as well as semantic depth;
- tuned-lens and ordinary-lens depths differ so radically that the predictor is unstable;
- cap states neither align with nor causally substitute for omitted computation;
- predicted configurations repeatedly fail the language-model health gates.

---

# (c) The paper I would write with the full asset set

# **Preserve Here, Adapt There: Causal Depth Profiles for Knowledge-Preserving LLM Adaptation**

## Central claim

> The pretrained computations required to preserve factual knowledge and the depth regions most useful for downstream adaptation form different causal profiles. Treating them as a single “important layer” score causes compact models either to retain knowledge but adapt poorly or adapt superficially while losing knowledge. A base-model diagnostic can prospectively allocate depth between the two requirements.

This claim is conditional, but it is both important and falsifiable.

## Why your assets are unusually well suited

Your checkpoint family spans four otherwise hard-to-obtain intervention axes:

| Asset/contrast | Question it can answer |
|---|---|
| Base versus intact full32 CPT | Does the healing corpus or optimizer itself erase knowledge? |
| Prefix keep8/10/12/14/16 ladder | How much original-prefix computation survives? |
| Freeze-front versus train-all | Is inherited information present but inaccessible without topological re-coordination? |
| Fully random model | What can the healing/SFT corpus learn without inherited knowledge? |
| ShortGPT-16 | Does non-contiguous selection or retention of the native final readout path matter more than nominal depth? |
| Multiple checkpoints | Does fluency, readout, or factual knowledge recover first? |
| OLMo/Qwen logit-lens curves | Where does knowledge become directly decodable in the intact base? |
| Catastrophic pruned checkpoints | Controlled lesions in which contextual behavior and output habits can survive after factual recall collapses |

Most groups have one compressed endpoint. You have a lesion family with inherited, frozen, trained, random, topology-altered, and time-resolved controls.

## The headline figure

### Panel A: two causal depth profiles

- **x-axis:** original layer index or two-layer window.
- **blue y-axis:** **knowledge-preservation necessity**, measured as closed-book/content-MMLU loss after removing a window, or rescue after restoring/patching that window.
- **orange y-axis:** **adaptation utility**, measured as downstream gain when only that matched window is adaptable.
- **Each point:** one original two-layer window, with seed-level intervals.
- Show OLMo and Qwen in separate subpanels.

The headline result would be low overlap between the windows most necessary for knowledge and the windows most useful for adaptation.

### Panel B: storage/computation/readout decomposition

A graft-rescue matrix:

- **x-axis:** source layer/window grafted from base or train-all.
- **y-axis:** receiving checkpoint—freeze-front, keep14, random, ShortGPT-like shell.
- **cell:** fraction of the target gap rescued.

Use separate rows for:

- tuned-lens decodability;
- content-MMLU;
- PopQA/TriviaQA/NQ-open;
- contextual QA;
- PPL;
- output mode concentration.

This separates:

1. information being decodable;
2. a component being causally necessary;
3. a readout being able to express it.

### Panel C: practical Pareto frontier

- **x-axis:** retained inference FLOPs or decoder depth.
- **y-axis:** target-task adaptation score.
- **color:** closed-book knowledge retention.
- **Each point:** one layer allocation policy and seed.

Compare:

- prefix keep-\(N\);
- ShortGPT;
- random retained layers;
- knowledge-only selector;
- adaptation-only selector;
- proposed preserve-plus-adapt selector.

The proposed policy should Pareto-dominate simple policies at a fixed depth.

## Mechanistic story

The most interesting possible result is:

> Pruning does not simply delete isolated “knowledge layers.” It interrupts a depth-dependent transformation that makes latent factual features usable by the native readout. Downstream adaptation is concentrated elsewhere, so task SFT can recover output habits or contextual mappings without rebuilding factual retrieval.

This naturally explains your observations:

- PPL or contextual behavior can improve while closed-book knowledge remains poor.
- Frozen inherited features show only a small edge unless the topology is re-coordinated.
- ShortGPT can outperform prefix truncation because it preserves native later-stage computation/readout structure.
- A narrow SFT can teach response patterns even when the parametric factual path is broken.

## Decisive experiments

### 1. Causal knowledge-necessity profile

For two-layer windows:

- remove from intact base;
- restore into prefix-pruned checkpoints;
- patch base activations into the damaged model;
- report content-MMLU and closed-book generation, not only answer letters.

Your existing activation-restoration harness is exactly the correct starting instrument. Include both:

- native/fresh-tail readout;
- original base head/readout.

That is necessary because a failed patch may otherwise mean only that the receiving tail cannot digest an out-of-distribution hidden state.

### 2. Matched adaptation-utility profile

Across at least three task families, make only one window adaptable at a time under:

- identical parameter count;
- identical data order and token budget;
- matched optimizer;
- matched training FLOPs where practical;
- several seeds.

An equal-rank per-window LoRA or equal-size trainable block gives a cleaner adaptation map than raw \(\|\Delta W\|\), because the latter is affected by scale and optimization.

### 3. Latent-versus-relearned knowledge control

Use either:

- a synthetic fact bank introduced before surgery; or
- facts verified absent from the healing stream.

Then ask whether patching/restoration recovers those facts without re-exposure. This is the cleanest distinction between preserved latent information and corpus relearning.

### 4. Prospective joint allocation on held-out Qwen

Before training:

1. derive a knowledge-preservation profile;
2. derive an adaptation-depth prediction;
3. commit the selected layer set/cut/cap;
4. compare it against prefix, ShortGPT, random, and fixed-top policies at the same depth.

## Closest competing work

- **Dynamic Weight Grafting: Localizing Finetuned Factual Knowledge in Transformers** (arXiv:2506.20746) already uses weight grafting to separate factual enrichment and recall pathways.
- **Surgical Fine-Tuning Improves Adaptation to Distribution Shifts** (ICLR 2023; arXiv:2210.11466) studies which layers to adapt.
- **On Surgical Fine-tuning for Language Encoders** (EMNLP 2023; arXiv:2310.17041) uses layerwise Fisher selection.
- **Layer by Layer: Uncovering Where Multi-Task Learning Happens in Instruction-Tuned Large Language Models** (EMNLP 2024; arXiv:2410.20008) studies where task-specific representations emerge.
- **Understanding and Guiding Layer Placement in PEFT** (arXiv:2602.04019) directly targets adaptation placement.
- **Locating and Editing Factual Associations in GPT** (NeurIPS 2022; arXiv:2202.05262), **Transformer Feed-Forward Layers Are Key-Value Memories** (EMNLP 2021; arXiv:2012.14913), and **Knowledge Neurons in Pretrained Transformers** (ACL 2022; arXiv:2104.08696) occupy broad factual localization.
- **Understanding Performance Collapse in Layer-Pruned LLMs via Decision Representation Transitions** (arXiv:2605.07271) is particularly close to any generic “there is a depth transition” claim.

The novelty must therefore be the **joint causal comparison of knowledge preservation and adaptation utility, followed by prospective architecture allocation**. Pure observational curves are not enough.

## What would kill this paper

- Frozen inherited checkpoints contain no more latent factual signal than matched random controls.
- Knowledge-necessity and adaptation-utility profiles coincide almost perfectly.
- Restoring a small number of windows cannot rescue knowledge.
- Recovery disappears for facts absent from the healing corpus.
- A simple “retain more original blocks” or “always retain the final layer” baseline matches the joint policy.
- The OLMo profile does not transfer at all to Qwen.

---

# (d) Is “format succeeds while capability is gone” publishable?

## Verdict on the present evidence: **No**

The current evidence does not establish that format SFT succeeded.

### Why \(F1-EM\approx0\) is not a format metric

An approximately zero gap can result from:

- short wrong answers with no token overlap;
- a single repeated refusal;
- one-token outputs;
- exact-or-completely-wrong behavior;
- genuine answer-only format compliance.

Those possibilities are observationally indistinguishable from aggregate EM/F1.

The problem is sharper here because:

- A4 has \(F1-EM\approx0.004\);
- A3 has an even smaller gap, approximately \(0.0007\);
- both are capability-collapsed;
- deeper A4 keep28 recovers a normal gap of roughly six points;
- the exact repeated refusal occurs massively in the SQuAD targets.

Thus the current pattern is better described as:

> **degenerate whole-string output behavior with occasional exact matches**

not successful format learning.

### SQuAD and closed-book QA are not a matched semantic comparison

SQuAD supplies evidence in context. PopQA and NQ-open do not. A model can succeed at contextual extraction while having almost no parametric factual recall.

Therefore:

> contextual answer extraction surviving while closed-book knowledge collapses

would be meaningful—but it is not the same as format alone satisfying the benchmark.

### The 52% refusal frequency is evidence of mode concentration

It should trigger reporting of:

- top-response frequency;
- unique-output ratio;
- output entropy;
- language distribution;
- average output length;
- refusal frequency;
- answerability-conditioned accuracy;
- performance after removing refusal-target examples.

Until those are measured, knowledge and format interpretations are secondary.

## The broad intellectual territory is already occupied

### Style/format versus capability

- **LIMA: Less Is More for Alignment** (NeurIPS 2023; arXiv:2305.11206) is the canonical superficial-alignment argument.
- **The False Promise of Imitating Proprietary LLMs** (arXiv:2305.15717) shows models can imitate style and appear instruction-following while closing little of the underlying capability gap.
- **A Closer Look at the Limitations of Instruction Tuning** (ICML 2024; arXiv:2402.05119) directly argues that LoRA instruction tuning often learns response initiation/style while full tuning can degrade knowledge.
- **Revealing the Inherent Instructability of Pre-Trained Language Models** / Response Tuning (Findings of EMNLP 2025; arXiv:2410.02465) shows that training only response distributions can elicit broad instruction-following behavior.
- **Revisiting the Superficial Alignment Hypothesis** (arXiv:2410.03717) shows that the strongest “post-training only changes style” claim is an oversimplification; capability can scale substantially with more post-training data.

### Explicit format measurement

- **FOFO: A Benchmark to Evaluate LLMs’ Format-Following Capability** (arXiv:2402.18667) explicitly finds that format-following can be independent of content quality.
- **ReFF: Reinforcing Format Faithfulness in Language Models across Varied Tasks** (AAAI 2025; arXiv:2412.09173) separately measures format faithfulness and task F1.
- **The Price of Format: Diversity Collapse in LLMs** (arXiv:2505.18949) studies format-induced output concentration.
- **Style over Substance: Distilled Language Models Reason Via Stylistic Replication** (COLM 2025; arXiv:2504.01738) finds that even incorrect synthetic reasoning traces can transfer useful stylistic patterns.

### Compression-specific proximity

- **Fragile Knowledge, Robust Instruction-Following: The Width Pruning Dichotomy in Llama-3.2** (arXiv:2512.22671) is very close: parametric-knowledge tasks degrade while instruction-following remains robust or improves.

### Evaluation-interface and shortcut work

- **“My Answer is C”: First-Token Probabilities Do Not Match Text Answers** (Findings of ACL 2024; arXiv:2402.14499).
- **When Benchmarks Are Targets: Revealing the Sensitivity of Large Language Model Leaderboards** (ACL 2024).
- **Shortcut Learning in Deep Neural Networks** (Nature Machine Intelligence 2020; arXiv:2004.07780).
- **What do Models Learn from Question Answering Datasets?** (EMNLP 2020; arXiv:2004.03490).
- **Look at the First Sentence: Position Bias in Question Answering** (EMNLP 2020; arXiv:2004.14602).

## Where room remains

There is room for a paper if the contribution is a **causal factorization of benchmark behavior**, not an anecdote about F1 and EM.

Measure four distinct axes:

1. **Schema compliance**
   - answer-only;
   - parseability;
   - correct output type;
   - no preamble/refusal;
   - length and syntax constraints.

2. **Context grounding**
   - does the answer change under counterfactual evidence?
   - does irrelevant context alter the answer?
   - is the answer supported by the passage?

3. **Answer-interface binding**
   - letter versus complete option;
   - randomized answer labels;
   - answer-order perturbation;
   - open generation versus likelihood scoring.

4. **Parametric knowledge**
   - same question with no evidence;
   - closed-book generation;
   - content likelihood;
   - latent probe/tuned-lens readout.

A publishable factorial would train or evaluate:

| | Correct semantics | Randomized/wrong semantics |
|---|---|---|
| Normal format | healthy SFT | format-preserving semantic corruption |
| Scrambled format | content without normal schema | neither |

Use the same underlying questions under:

- gold context;
- no context;
- irrelevant context;
- counterfactual context.

If all four format/capability quadrants are populated across multiple tasks and model families, that is a real evaluation-methodology result.

**Bottom line:** potentially publishable after a substantial redesign; not publishable from the current \(F1-EM\) and refusal aggregates.

---

# (e) Ranked pivotable paper propositions

## Ranking summary

| Rank | Proposition | Scientific upside | Feasibility with current assets | Principal risk |
|---:|---|---:|---:|---|
| **1** | Preserve Here, Adapt There | Very high | High | Requires causal separation, not just probes |
| **2** | Depth-Deficit Law for Fresh Caps | Very high if prospective | Medium | Recent forward-only PEFT work crowds novelty |
| **3** | Probe-Certified Knowledge-Survival Frontier | High | Very high | Probe onset may not predict causal preservation |
| **4** | Recovery Phase Diagram after Depth Surgery | Medium-high | Extremely high | Strong overlap with Paper B |
| **5** | Capability–Interface Factorization | Medium after redesign | Medium | Current evidence does not establish format success |

---

## 1. Preserve Here, Adapt There

**Claim:** The causal layer profile required to preserve pretrained factual knowledge differs from the profile that provides maximal downstream adaptation utility; jointly allocating depth to both profiles gives a better compact model.

**Key figure:** original layer on the x-axis; normalized knowledge-necessity and adaptation-utility curves on the y-axis, followed by a fixed-depth Pareto frontier colored by knowledge retention.

**Experiments:**

1. Window removal/restoration and activation patching for content-MMLU plus closed-book QA.
2. Matched one-window-at-a-time adaptation on three task families.
3. Synthetic-fact or fact-excluded healing control.
4. Prospective preserve-plus-adapt policy on Qwen.

**Falsified if:** the profiles coincide, inherited frozen states contain no latent signal, small-window restoration cannot rescue knowledge, or a simple depth/count baseline matches the proposed policy.

**Closest work:**

- **Dynamic Weight Grafting** (arXiv:2506.20746);
- **Surgical Fine-Tuning** (ICLR 2023; arXiv:2210.11466);
- **Layer by Layer** (EMNLP 2024; arXiv:2410.20008);
- **Understanding and Guiding Layer Placement in PEFT** (arXiv:2602.04019);
- **Decision Representation Transitions** (arXiv:2605.07271).

**Recommendation:** **best paper if it works.**

---

## 2. The Depth-Deficit Law for Fresh Caps

**Claim:** For a cut at \(j\), the minimum useful fresh-cap depth is determined by the base model’s task-linearization deficit \(\mu_t-j\).

**Key figure:** universal curve with normalized depth surplus on x and normalized target performance on y; secondary predicted-versus-observed \(K^*\) scatter showing held-out points.

**Experiments:**

1. Fit \(\gamma,\delta,F\) on OLMo-1B/7B development tasks.
2. Freeze predictions for Qwen3-4B and held-out task families.
3. Run predicted \(K\) and adjacent controls before exposing the exhaustive sweep.
4. Test cap/base alignment and activation-graft substitution.

**Falsified if:** \(K^*\) does not shift inversely with \(j\), fixed \(K=2\) has equivalent regret, held-out tasks require refitting, or cap layers do not causally replace omitted computation.

**Closest work:**

- **Predicting Fine-Tuning Performance with Probing** (EMNLP 2022; arXiv:2210.07352);
- **Hidden State Variability…Can Guide Computation Reduction** (EMNLP 2022; arXiv:2210.10041);
- **RSRA** (arXiv:2607.09757);
- **Dominant-Layer ZO** (arXiv:2606.05516);
- **RDP LoRA** (arXiv:2604.19321);
- **TuneAhead** (ICML 2026; arXiv:2606.17660).

**Recommendation:** conditional greenlight. A post-hoc correlation is incremental; a preregistered low-regret law could be excellent.

---

## 3. Probe-Certified Knowledge-Survival Frontier

**Claim:** A base-model knowledge-transition profile prospectively predicts the minimum original-layer coverage needed for parametric-knowledge survival after pruning and healing, whereas PPL and nominal resulting depth do not.

**Key figure:**

- **x-axis:** distance from the probe-predicted survival boundary or probe-weighted layer coverage.
- **y-axis:** normalized closed-book knowledge retention.
- **Each point:** one `(model, pruning policy, retained-depth, checkpoint)` configuration.
- **Color:** PPL.
- **Shape:** prefix, ShortGPT, final-layer-retaining, frozen, random.

OLMo fits the predictor; Qwen points remain hidden until predictions are frozen.

**Experiments:**

1. Evaluate Paper C keep20/24/28 checkpoints on closed-book and general-capability tasks, not only SQuAD.
2. Evaluate or finish the 16-layer structural factorial: prefix14+fresh2, contiguous16/no-fresh, final-layer-retaining14+fresh2, ShortGPT-16.
3. Run a Qwen ladder concentrated around its measured \(0.694L\) transition.
4. Replace ordinary logit lens with tuned-lens and prompt-format controls.

**Falsified if:** retained-block count or PPL predicts equally well, Qwen violates the frontier, final-layer retention dominates in a way the probe cannot represent, or the boundary moves substantially with prompt/probe choices.

**Closest work:**

- **The Unreasonable Ineffectiveness of the Deeper Layers** (ICLR 2025; arXiv:2403.17887);
- **ShortGPT** (Findings of ACL 2025; arXiv:2403.03853);
- **LLM-Streamline** (arXiv:2403.19135);
- **Decision Representation Transitions** (arXiv:2605.07271);
- **The Cost of Compression** (Findings of EMNLP 2023; arXiv:2312.00960);
- **MechLens: Late Crystallization of Factual Knowledge** (arXiv:2606.07978).

**Recommendation:** best feasibility-to-novelty fallback, but it must be prospective and cross-model.

---

## 4. Recovery Phase Diagram after Depth Surgery

**Claim:** After structural damage, language-model fit, contextual behavior, answer-interface binding, and parametric factual recall recover on separable timescales, while architecture primarily changes the recovery asymptote rather than merely convergence speed.

**Key figure:**

- **x-axis:** cumulative post-surgery training FLOPs, log-scaled.
- **y-axis:** floor-to-intact normalized recovery.
- Curves: PPL, ordinary language tasks, contextual QA, direct schema compliance, content-MMLU, closed-book QA, output entropy.
- One panel per construction; every point is an actual checkpoint.

**Experiments:**

1. Evaluate a common checkpoint grid for keep8/10/12/14, ShortGPT, frozen, random, and full32.
2. Replicate keep14 and ShortGPT over at least three seeds.
3. Add a small Qwen or OLMo-1B replication.
4. Fit preregistered monotone/asymptotic curves and estimate half-lives and asymptotes.

**Falsified if:** phase ordering disappears under matched FLOPs, knowledge catches up with extended training, seed variance dominates metric separation, or the ordering changes arbitrarily across model families.

**Closest work:**

- **IteRABRe** (arXiv:2503.06291);
- **ShortOPD** (arXiv:2607.13124);
- **On the Limits of Layer Pruning for Generative Reasoning** (arXiv:2602.01997);
- **The Unreasonable Ineffectiveness of the Deeper Layers** (arXiv:2403.17887);
- your own Paper B.

**Recommendation:** excellent analysis inside Proposition 1 or 3. As a standalone paper, it risks self-overlap with Paper B.

---

## 5. Capability–Interface Factorization

**Claim:** Benchmark behavior after pruning/SFT is non-identifying because schema compliance, context grounding, answer-symbol binding, and parametric knowledge can change independently.

**Key figure:**

- **x-axis:** semantic correctness or closed-book knowledge.
- **y-axis:** direct schema-compliance rate.
- **color:** counterfactual grounding score.
- **Each arrow:** one model before and after SFT.
- Shapes: intact, prefix-pruned, ShortGPT, frozen, random.

**Experiments:**

1. Correct/random content × normal/scrambled format SFT factorial.
2. Gold/no/irrelevant/counterfactual context evaluation on the same items.
3. Letter/content/open-generation and answer-order/label randomization.
4. Replicate across two task and two model families.

**Falsified if:** A4 is not directly format-compliant, format and semantics remain tightly coupled, counterfactual contexts do not separate grounding, or the decomposition predicts no external failure.

**Closest work:**

- **LIMA** (NeurIPS 2023; arXiv:2305.11206);
- **The False Promise of Imitating Proprietary LLMs** (arXiv:2305.15717);
- **A Closer Look at the Limitations of Instruction Tuning** (ICML 2024; arXiv:2402.05119);
- **FOFO** (arXiv:2402.18667);
- **ReFF** (AAAI 2025; arXiv:2412.09173);
- **Fragile Knowledge, Robust Instruction-Following** (arXiv:2512.22671);
- **“My Answer is C”** (Findings of ACL 2024; arXiv:2402.14499).

**Recommendation:** do not write this from the current evidence. It becomes publishable only after direct, same-item, causal factorization.

---

# Final recommendation and go/no-go sequence

## Greenlight

Combine Propositions **1 and 2**:

> Build causal knowledge-preservation and adaptation-utility depth profiles, then test whether a base-model depth diagnostic prospectively predicts a preserve-plus-adapt architecture on held-out Qwen tasks.

This gives one coherent paper:

- **scientific question:** are the computations needed for knowledge and adaptation co-located?
- **mechanism:** latent features, depth-dependent transformation, and readout repair;
- **practical outcome:** a predicted compact layer allocation;
- **unique assets:** lesion family, frozen/train-all/random controls, ShortGPT, trajectories, two-model probes;
- **hard falsifier:** profile coincidence or failed held-out prediction.

## Immediate experimental order

1. **Cheap validity cleanup**
   - direct per-example SQuAD refusal/answerable decomposition;
   - output concentration and schema metrics;
   - ordinary versus tuned lens;
   - strict-frozen naming or rerun.

2. **Causal forward-only experiments**
   - complete restoration/patching with both fresh-tail and base-head readouts;
   - derive a knowledge-necessity profile;
   - evaluate keep20/24/28 on closed-book tasks.

3. **Matched adaptation map**
   - three clean task families;
   - equal parameter and training budget per candidate window;
   - several seeds.

4. **Freeze the predictor**
   - fit only on OLMo development data;
   - commit Qwen3-4B predictions and acceptance criteria.

5. **Run the held-out test**
   - predicted cap and neighbors first;
   - exhaustive sweep only afterward;
   - report regret, not just correlation.

## Kill criteria

Abandon the combined paper if:

- causal knowledge and adaptation profiles coincide;
- tuned-lens depth is unstable across prompts;
- activation/weight interventions cannot rescue knowledge;
- the depth law fails the within-task translation test;
- fixed \(K=2\) or nominal total depth matches held-out regret;
- Qwen requires refitting.

If that happens, the honest destination is to fold the recovery and interface diagnostics into Paper B rather than force a separate Paper C.

The one valuable paper inside these results is:

> **Depth surgery separates preservation from adaptation, and base-model structure can predict how compact depth should be allocated.**

It is not “fresh caps work,” and it is not yet “format can fake capability.”