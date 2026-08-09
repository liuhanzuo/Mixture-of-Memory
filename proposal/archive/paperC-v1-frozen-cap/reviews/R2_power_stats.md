# Methodology review

**Internet access: yes.** I used the internet to verify the existence and bibliographic metadata of cited work. The experimental audit itself is based on the full contents of `paperC_BRIEF_for_research.md` plus a read-only inspection of the local training, evaluation, data, and status files. I did not modify any files.

## Overall verdict

The current A4-versus-A3 result supports only this narrow descriptive statement:

> For one seed, after approximately 166 passes over a tiny and severely imbalanced custom QA corpus, the final A4 checkpoint obtained 586/2,000 exact matches and the final A3 checkpoint obtained 521/2,000.

It does **not** presently establish:

> Useful inherited trunk features caused a 3.25 percentage-point advantage at equal depth.

The phrase in the brief that A4 versus A3 differs only in “front-block inherit+freeze versus random initialization” is factually too strong. The arms also differ in:

- number of trainable parameters;
- which components are pretrained;
- optimizer implementation and precision;
- learning rates;
- regularization induced by freezing;
- optimization dimensionality and training FLOPs;
- and likely sensitivity to the extremely distorted evaluation label distribution.

The most damaging additional findings from the local audit are:

1. **The actual headline A3 run used bitsandbytes AdamW8bit after fp32 AdamW OOMed; A4 used torch AdamW with fp32 state.**
2. **A4 inherits and trains the token embedding, final norm, and LM head—not only the trunk.**
3. Of A4’s 1.2269B trainable parameters, only about **404.8M, or 33%, are fresh transformer layers**. The remaining **822.1M, or 67%, are inherited embedding/head/norm parameters.**
4. The custom `squad_val.jsonl` has **997/2,000 = 49.85%** of examples with the same Chinese refusal target, `根据提供的信息无法回答这个问题`. A constant-output refusal system would score **49.85% EM**, substantially above A4’s 29.30% and A3’s 26.05%.
5. That refusal label is only **17.56%** of the training set, so train and validation differ by 32.29 percentage points on the dominant label.
6. Training uses full-language-model loss over packed sequences. By direct tokenization audit, answer tokens constitute only about **3.68%** of the example tokens; approximately 96% of the objective is prompt/context language modeling.
7. There is only one training seed per arm. The existing McNemar and bootstrap analyses describe uncertainty conditional on these exact two checkpoints, not uncertainty of the training methods.

These points materially weaken the current headline, but they also make the corrective experiment unusually clear.

---

# (a) Attack on the existing A4-versus-A3 evidence

## What is actually controlled

The following are legitimately matched:

- deployed depth: 16 transformer layers;
- hidden size and block architecture;
- total deployed parameter count: approximately 4.060B;
- nominal training corpus;
- 1,000 optimizer steps;
- effective batch size 128;
- sequence length 2,048;
- nominal seed 42;
- evaluation prompts, decoding, and test examples.

That is useful. But it does not make trunk inheritance the only treatment.

## What the two arms actually are

| Component | A4 freeze-graft | A3 from scratch |
|---|---:|---:|
| Total parameters | 4.06035B | 4.06035B |
| Trainable parameters | **1.22687B** | **4.06035B** |
| Frozen decoder-block parameters | 2.83348B | 0 |
| Two fresh cap blocks | 404.78M | Included among all-random blocks |
| Token embedding | **Pretrained, trainable** | Random, trainable |
| Final norm | Pretrained, trainable | Random, trainable |
| LM head | **Pretrained, trainable** | Random, trainable |
| Peak LR | 1e-4 cap/head; 2e-5 embedding/norm | 3e-4 uniformly |
| Actual optimizer | torch AdamW, fp32 state | **bitsandbytes AdamW8bit** |
| Unique packed token positions | 1.58M | 1.58M |
| Approximate corpus passes | 166 | 166 |
| Training seeds | 1 | 1 |

Thus A4 is not simply “a frozen pretrained trunk plus a fresh cap.” It is:

> pretrained frozen decoder blocks + pretrained trainable embedding + pretrained trainable final norm + pretrained trainable LM head + fresh cap.

## Confound-by-confound judgment

Rating meanings:

- **Yes—strongly plausible:** this factor alone is large enough that it could reasonably account for the full 3.25pp.
- **Yes—plausible:** it could account for the full gap, but an ablation is needed.
- **Possible:** not the leading explanation, but cannot be excluded.
- **Inference only:** does not create the point estimate, but invalidates the claimed certainty or generality.

### 1. Treatment-composition confounds

| Confound | Why it matters | Could it alone explain +3.25pp without useful inherited trunk features? |
|---|---|---|
| **1.23B versus 4.06B trainable parameters** | A3 must optimize 3.31× more parameters from random initialization. In this tiny-data regime, more trainable capacity means a much harder optimization problem and much weaker regularization. | **Yes—strongly plausible.** Freezing could improve held-out EM through regularization alone. |
| **Frozen versus fully updated trunk** | A4 cannot destroy its retained blocks; A3 can memorize the training stream throughout all layers. | **Yes—plausible.** A freezing/regularization effect requires no useful task-specific inherited representation story. |
| **Inherited token embedding** | The embedding has approximately 411M parameters. A3 must learn token geometry from the custom QA corpus; A4 starts from a language-trained embedding. | **Yes for defeating the trunk claim.** This still invokes pretrained information, but not useful inherited *trunk* features. |
| **Inherited LM head** | The untied LM head is another approximately 411M parameters. A3 must learn vocabulary readout geometry from scratch. | **Yes for defeating the trunk claim.** The output head alone could account for several EM points. |
| **Inherited final norm** | A4 begins with a readout-compatible normalization layer; A3 does not. | **Possible**, especially in a decoder that is already near collapse. |
| **Only 33% of A4’s trainable parameters are fresh blocks** | The claimed “regrown cap” is not the majority of A4’s trainable model. | Not an independent numerical explanation, but it shows that the current treatment is mischaracterized. |
| **No inheritance × freezing factorial** | Initialization and updating are perfectly entangled. There is no inherited-updated arm and no random-frozen arm. | It does not itself create the gap, but makes causal attribution **impossible**. |
| **No trainable-parameter-matched random arm** | Equal deployed depth is not equal optimization dimensionality. | **Yes—plausible.** A random model restricted to the same trainable subset could close the gap. |
| **No I/O-matched random arm** | There is no arm with pretrained embedding/head but randomized trunk blocks. | **Yes—strongly plausible.** Such an arm could reveal that the current advantage comes almost entirely from I/O initialization. |
| **No pretrained-statistics negative control** | A fully random frozen trunk differs from a pretrained trunk in activation scale and geometry as well as learned content. | **Possible.** A tensor-shuffled or weight-scrambled pretrained trunk would be a useful secondary negative control. |

### 2. Optimization confounds

| Confound | Why it matters | Could it alone explain +3.25pp? |
|---|---|---|
| **Actual optimizer mismatch** | Operational records show A3 fp32 AdamW OOMed and was relaunched with bitsandbytes AdamW8bit. A4 used torch AdamW. | **Yes—plausible.** A 3.25pp effect between unstable models is easily within the range an optimizer/state-precision change could produce. |
| **Peak LR mismatch** | A3 uses 3e-4 on every parameter; A4 uses 1e-4 on cap/head and 2e-5 on embedding/norm. | **Yes—strongly plausible.** A3 receives 3× the high-group LR and 15× the low-group LR. |
| **Different integrated LR exposure** | The whole cosine trajectory differs, not just the peak. | **Yes—plausible.** The experiment compares different optimization regimes. |
| **Same 150-step warmup despite different problems** | Warmup is 15% of the complete run, but the two arms differ 3.31× in trainable dimension and use different optimizers. | **Possible.** |
| **Global gradient clipping at 1.0** | The norm is computed over 1.23B versus 4.06B trainable parameters. A3 may be clipped more often or more severely. No clipping-frequency statistics are reported. | **Possible, potentially sufficient.** |
| **Weight decay acts on very different sets** | A3 decays all random matrices; A4 does not update or decay the frozen trunk. | **Possible.** |
| **Unequal hyperparameter-tuning effort** | A4 is the “hero” with a differential-LR recipe; A3 receives one uniform LR. | **Yes—strongly plausible.** A tuned recipe versus a single baseline setting can readily create a 3.25pp gap. |
| **Training compute is not matched** | Equal tokens and steps do not imply equal backward/optimizer FLOPs. | **Yes as a resource-comparison confound.** It does not identify which recipe is more sample-efficient versus compute-efficient. |
| **Hardware/numerical path is not fully common** | Even when nominal precision matches, H20/B200 or different optimizer kernels can change trajectories. | **Possible**, mainly relevant for future cross-node comparisons. |

### 3. Data and objective confounds

| Confound | Why it matters | Could it alone explain +3.25pp? |
|---|---|---|
| **Only 1.58M unique token positions** | A3 is a 4B-parameter random model exposed repeatedly to a tiny unique corpus. It is not a credible language-learning baseline. | **Yes—strongly plausible.** The contrast is close to “pretrained language model versus random network on an extremely small corpus.” |
| **Approximately 166 epochs** | There are 770 packed chunks and effective batch 128, or about six optimizer steps per corpus pass. | **Yes—strongly plausible.** The arms can overfit at different rates and cross during training. |
| **Training PPL approaches 1.0** | This demonstrates memorization, not useful capacity saturation. | **Yes—plausible.** Differential memorization/generalization can account for the gap. |
| **Full-LM instead of answer-only loss** | Direct audit gives mean prompt length about 150.95 tokens and answer length about 5.77 tokens. Only **3.68%** of tokens are answer tokens. | **Yes—strongly plausible.** A4’s pretrained LM is advantaged on the approximately 96% of training loss that is not answer correctness. |
| **Packed examples attend across boundaries** | EOS separates examples, but attention is not reset. The model can learn cross-example/order transitions unrelated to QA. | **Possible.** Shared across arms, but pretrained and random models may exploit the artifact differently. |
| **Validation is 49.85% one refusal label** | A constant Chinese-refusal output achieves 49.85% EM. | **Yes—very strongly plausible.** The 65-net-example A4 advantage could be refusal-rate calibration rather than QA. |
| **Train refusal prevalence is only 17.56%** | The dominant label shifts from 17.56% in training to 49.85% in validation. | **Yes—very strongly plausible.** Different extrapolation of the learned refusal prior could account for all 3.25pp. |
| **No answerable/refusal-stratified score** | The 65-example gap could occur entirely in the 997-refusal subset: 65/997 = 6.52pp; or entirely in the 1,003 non-refusal subset: 6.48pp. | **Yes.** Either subgroup alone could produce the entire aggregate gap. |
| **Custom transformed split, not ordinary SQuAD reporting** | The evaluation contains synthetic refusal targets and a highly nonstandard label distribution. | **Yes—plausible.** Another balanced split could materially change or reverse the ordering. |
| **Possible pretraining contamination/familiarity** | A4 may have encountered SQuAD passages or associations during pretraining; A3 cannot have. | **Yes—plausible**, though this would still be a pretrained-information effect rather than evidence for the proposed architecture. |
| **Question/passage clustering** | The 2,000 examples collapse to roughly 235 context/passage groups. | **Inference only** for the observed gap, but ordinary item bootstrap is anticonservative. |

The tiny-data issue is not subtle. Repeatedly presenting 1.58M unique token positions does not turn them into a diverse pretraining corpus. Compute-optimal pretraining work underscores how far a multi-billion-parameter random model is from a credible language learner under such data diversity (“Training Compute-Optimal Large Language Models,” arXiv:2203.15556). ([arxiv.org](https://arxiv.org/abs/2203.15556))

### 4. Checkpoint and reproducibility confounds

| Confound | Why it matters | Could it alone explain +3.25pp? |
|---|---|---|
| **One training seed** | A4 randomizes only the cap; A3 randomizes the entire network. Their seed variances need not be equal. | **Yes—strongly plausible.** The whole 3.25pp can be a favorable seed pair. |
| **One data order** | Order sensitivity is unmeasured. A3 was also relaunched after OOM. | **Possible.** |
| **Final step-1,000 checkpoint only** | There is no preregistered validation-selection rule or common early-stopping criterion. | **Yes—plausible.** Curves may peak or cross at different times. |
| **The run endpoint changed from the launcher default** | The launcher default is 2,000 steps; the orchestrator used 1,000. | **Possible** if the choice was informed by preliminary behavior. |
| **Code-version provenance** | The A4 path initially had a DDP `module.` classification bug that assigned the cap the wrong LR; it was reportedly fixed and restarted. | **Unlikely if the final checkpoint manifest proves it came from the fixed run; potentially fatal if not.** |
| **OOM/restart provenance** | A3’s failed fp32 run and bnb8bit restart require immutable manifests to exclude stale or mixed artifacts. | Mostly covered by the optimizer mismatch, but auditability is weakened. |

Fine-tuning instability with respect to initialization, data order, optimizer, and early stopping is well documented in “Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping” (arXiv:2002.06305), “On the Stability of Fine-tuning BERT” (arXiv:2006.04884), and “Accounting for Variance in Machine Learning Benchmarks” (arXiv:2103.03098). ([arxiv.org](https://arxiv.org/abs/2002.06305))

### 5. Evaluation and statistical confounds

| Confound | Why it matters | Could it alone explain +3.25pp? |
|---|---|---|
| **Greedy first-line exact match** | A correct answer with extra text, an answer prefix, punctuation, alias, or newline can become an error. | **Yes—plausible.** Sixty-five formatting differences are sufficient. |
| **One gold string per item** | Semantically equivalent variants can be scored wrong. | **Possible.** |
| **F1 nearly equals EM for both keep14 arms** | A4 F1−EM is 0.40pp; A3 is only 0.07pp. This is consistent with all-or-nothing emission of a small set of memorized complete strings. | Diagnostic rather than an independent confound, but it strongly undermines a graded QA interpretation. |
| **No refusal-output diagnostics in the headline** | Repeated-output rate, output entropy, answer length, and false-refusal rate are absent. | **Yes—plausible.** |
| **Ordinary McNemar/item bootstrap conditions on fixed checkpoints** | It treats evaluation items as the source of uncertainty, not the training process. | **Inference only.** It cannot show that the method reliably wins across training runs. |
| **Item independence is false** | Questions share passages; MMLU items share subjects; benchmark items are heterogeneous. | **Inference only**, but nominal p-values can be too optimistic. |
| **Multiple endpoints and researcher degrees of freedom** | EM, F1, multiple arms, depths, checkpoints, 14 benchmarks, pooled tests, and per-task tests were examined. | **Selection can plausibly explain apparent significance** unless the primary contrast was preregistered. |
| **No multiplicity adjustment for per-task tests** | Several reported p-values around .01–.015 would not survive Holm correction over 14 tasks. | **Inference only.** |
| **Item-count-weighted benchmark pooling** | MMLU and PopQA receive many times the influence of smaller tasks solely because they contain more examples. | It can explain much of the pooled +0.39pp summary. |

Exact match and token F1 are known to miss semantic answer equivalence; see “Tomayto, Tomahto. Beyond Token-level Answer Equivalence for Question Answering Evaluation” (EMNLP 2022, arXiv:2202.07654). ([arxiv.org](https://arxiv.org/abs/2202.07654))

## Judgment on the headline result

At least six single factors could plausibly account for the complete 3.25pp without requiring useful inherited trunk representations:

1. freezing as regularization;
2. learning-rate mismatch;
3. optimizer mismatch;
4. final-checkpoint selection after severe overtraining;
5. one favorable seed pair;
6. refusal-label calibration/format behavior.

Inherited embedding and LM-head initialization provide an additional explanation that uses pretrained weights but does **not** establish that the inherited 14-layer trunk is the operative component.

The McNemar and bootstrap results demonstrate that the two fixed checkpoints behave differently on the fixed evaluation set. They do not establish why, and they do not establish method-level reproducibility.

---

# (b) Are comparisons between two near-chance models meaningful?

## Direct answer

**They are meaningful only as failure-mode or threshold-crossing studies.**

A comparison between two near-chance models can answer questions such as:

- Which model collapses first?
- Which initialization reaches usable behavior sooner?
- Does one arm cross a preregistered viability threshold?
- Do the arms have different calibration or refusal modes?

It cannot support a graded **capability-retention** claim when neither arm has meaningful capability to grade.

For Finding #132, the scientifically primary result is:

> Both 16-layer arms fail to retain useful general capability.

The A4>A3 result is secondary:

> Conditional on these exact checkpoints and an item-weighted benchmark pool, A4 is marginally less failed.

That is not a capability win.

## Why the current binomial z is insufficient

For MMLU:

- A4 = 0.2596, only +0.96pp over 0.25;
- A3 = 0.2474, −0.26pp below 0.25;
- intact base = 0.6056.

Define recovered above-chance signal:

\[
R=\frac{s-c}{b-c},
\]

where \(s\) is the arm score, \(c\) the trivial/chance baseline, and \(b\) the intact-base score.

For A4 MMLU:

\[
R=\frac{0.2596-0.25}{0.6056-0.25}\approx0.027.
\]

A4 therefore retains only about **2.7% of the intact model’s available above-chance MMLU signal**.

A \(z=2.6\) result says that 0.2596 is detectably different from exactly 0.25 given 14,042 nominal items. It does not say that 0.2596 is scientifically useful.

Using the rounded 14-task scores, the equal-task macro A4−A3 difference is only about **+0.13pp**, versus the reported item-weighted +0.39pp. The latter gives enormous weight to whichever benchmarks happen to contain the most rows.

## Recommended minimum capability floor

There should not be one universal raw-accuracy cutoff. Use a preregistered multi-gate floor.

### Gate 1: evaluation validity

For each task, define the strongest trivial baseline \(c_t\), not automatically \(1/K\).

For MC tasks, include:

- random guessing;
- majority/label-prior baseline;
- question-ablated baseline;
- option-position baseline;
- option-order permutation sensitivity.

For generated QA, include:

- empty output;
- most-common training answer;
- fixed refusal;
- answer-frequency baseline.

For the current custom SQuAD split, the trivial baseline is at least the **49.85% constant-refusal baseline**. Because both A4 and A3 score below it, aggregate EM does not pass the evaluation-validity gate.

### Gate 2: behavioral nondegeneracy

Recommended preregistered minimums for answerable-only generation:

- nonempty output on at least 95% of examples;
- unrelated fixed-refusal rate below 5%;
- no single identical completion on more than 10% of examples;
- report output entropy, answer length, and number of distinct normalized outputs.

For MC tasks:

- valid finite score on at least 99.9% of items;
- no pathological single-option collapse;
- prediction entropy reasonably close to gold-label entropy;
- robustness to answer-option permutation.

Both existing keep14 arms fail this gate for closed-book QA.

### Gate 3: practical capability

For each task, require:

\[
\operatorname{LCB}_{95\%}(s-c)>
\max\left(0.02,\ 0.10(b-c)\right).
\]

That is:

- the lower confidence bound must be at least 2 absolute points above the strongest trivial baseline; and
- it must retain at least 10% of the intact model’s available signal.

This is a deliberately lenient floor for deciding whether a graded comparison is interpretable.

For MMLU:

\[
c=0.25,\qquad b=0.6056,
\]

so 10% recovery requires:

\[
s>0.28556.
\]

Ignoring seed and subject clustering, a point estimate of roughly 0.292 would be needed for the one-sided 95% lower bound to exceed 0.2856. Actual seed-aware requirements are stricter. A4 at 0.2596 and A3 at 0.2474 fail decisively.

### Gate 4: breadth

For a phrase such as **general capability retention**, require:

- macro recovered signal \(R\ge0.20\);
- lower CI above zero;
- floor clearance in at least two preregistered families, e.g. knowledge and reasoning/comprehension;
- no LM-integrity or output-degeneration failure.

The distinction is:

- **Both arms pass:** graded comparison is interpretable.
- **Only A4 passes:** report threshold crossing—A4 is viable, A3 is not.
- **Neither passes:** report failure behavior only.

Finding #132 is in the third category.

## How the floor should be measured and reported

For every task, arm, and training seed, report:

- raw accuracy or NLL;
- strongest trivial baseline;
- absolute margin over that baseline;
- intact-base score;
- recovered-signal fraction \(R\);
- behavioral-validity diagnostics;
- paired A4−A3 effect;
- 95% seed- and cluster-aware interval;
- practical-floor pass/fail;
- equivalence-to-chance result.

For SQuAD-like data, additionally report separately:

- answerable-item EM/F1;
- refusal-item sensitivity;
- false-refusal rate on answerable items;
- balanced accuracy over answerable versus unanswerable;
- constant-refusal baseline;
- passage-clustered CI.

## Better statistics

1. **Separate exact-chance testing from practical equivalence.**  
   Test superiority to the practical floor and conduct TOST or a ROPE analysis around chance, e.g. ±2pp. A4 can be statistically above exact chance while practically chance-equivalent.

2. **Use training seed as the method-level unit.**  
   The 78,656 benchmark items are not 78,656 independent replications of the training method.

3. **Use paired cluster resampling.**  
   Resample:
   - training seeds;
   - tasks or capability families;
   - MMLU subjects, SQuAD passages, or equivalent natural clusters;
   - items within clusters.

4. **Use equal-task or equal-family macro aggregation.**  
   Do not concatenate all benchmark rows and run one binomial test.

5. **Report heterogeneity.**  
   Show task-specific effects, task-level median, positive/null/negative counts, and arm×task interaction.

6. **Correct multiplicity.**  
   Preregister one primary composite and apply Holm correction to secondary task tests.

7. **Keep McNemar secondary.**  
   McNemar is appropriate for paired binary outcomes on a fixed test set, but it does not include training-seed variance.

Inference across heterogeneous datasets should follow multi-dataset replication logic rather than treating every item as exchangeable; see “Replicability Analysis for Natural Language Processing: Testing Significance with Multiple Datasets” (TACL 2017, arXiv:1709.09500) and “With Little Power Comes Great Responsibility” (EMNLP 2020, arXiv:2010.06595). ([arxiv.org](https://arxiv.org/abs/1709.09500))

---

# (c) Experiment that would actually settle the question

## First define the question narrowly

There are two distinct questions:

1. **Causal trunk question:**  
   Holding I/O initialization, cap, trainable subset, optimizer, and data fixed, do pretrained front blocks outperform random front blocks?

2. **Policy question:**  
   Is the complete freeze-graft recipe better than building and training the whole shallow model from scratch under a fixed resource budget?

The current A4−A3 comparison conflates them. The decisive experiment should answer both, with the first as primary.

## Select a viable depth before confirmation

Do not make 16 layers the primary depth.

Use a development-only PPL/capability evaluation to choose:

> the shallowest of 26 or 30 deployed layers that clears the preregistered capability floor.

Then lock that depth \(D^*\) before evaluating the final test suite.

- If 26 layers passes, use `keep24+fresh2`.
- If only 30 layers passes, use `keep28+fresh2` and honestly frame the result as modest depth reduction.
- If neither passes, stop: the construction is not presently a usable capability-retention method.

Use a disjoint validation suite for this choice, not the confirmatory test tasks.

## Core 2×2 factorial

Let \(j=D^*-2\) and \(K=2\).

To isolate the decoder-block trunk:

- use the same **pretrained token embedding** in every factorial arm;
- freeze that embedding in every arm;
- use the same pretrained final norm and LM head initialization;
- train final norm, LM head, and fresh cap identically;
- use bit-identical fresh-cap initialization within each paired seed.

| Arm | Front-block initialization | Front blocks updated? | Purpose |
|---|---|---:|---|
| **IF: inherited-frozen** | Pretrained | No | Proposed method |
| **RF: random-frozen** | Correct OLMo/Qwen random init | No | Clean inheritance control |
| **IU: inherited-updated** | Pretrained | Yes | “Inherit but do not freeze” |
| **RU: random-updated** | Random | Yes | Random-init control with matching trainability |

The primary causal contrast is:

\[
\tau_{\text{inherit}\mid\text{frozen}}
=Y(\mathrm{IF})-Y(\mathrm{RF}).
\]

Other contrasts are:

\[
\tau_{\text{inherit}\mid\text{updated}}
=Y(\mathrm{IU})-Y(\mathrm{RU}),
\]

\[
\tau_{\text{freeze}\mid\text{inherit}}
=Y(\mathrm{IF})-Y(\mathrm{IU}),
\]

and the factorial interaction:

\[
\psi=
[Y(\mathrm{IF})-Y(\mathrm{RF})]
-
[Y(\mathrm{IU})-Y(\mathrm{RU})].
\]

### Fully random bridge arm

Retain a fifth descriptive arm:

- random embedding;
- random trunk;
- random cap;
- random norm/head;
- all trainable.

Call this **FS**, the fully from-scratch policy baseline.

Then:

- IF−RF identifies the pretrained trunk;
- RU−FS identifies pretrained I/O initialization;
- IF−FS is the total package effect.

If RU≫FS while IF≈RF, then the original A4>A3 result came mainly from embedding/head initialization, not inherited trunk computation.

## What must be held constant

For compared arms:

- deployed depth, width, and total deployed parameters;
- tokenizer and prompt construction;
- fresh-cap tensor values within paired seeds;
- exact unique training examples and order;
- optimizer implementation and state precision;
- autocast/master-weight precision;
- global token batch;
- warmup fraction and LR schedule;
- weight decay, betas, epsilon, and clipping;
- checkpoint schedule;
- test prompts and scoring;
- hyperparameter-search budget.

For IU versus RU, the front-block LR must be identical. Do not give the pretrained and random trunks different manually selected LRs in the confirmatory comparison.

## Training data and objective

Do not repeat 166-epoch full-LM SQuAD training.

Recommended confirmatory protocol:

- **100M unique SFT tokens**, approximately one pass;
- approximately 5M held-out validation tokens;
- multiple task/format sources;
- exact and near-duplicate filtering against evaluation sets;
- assistant/answer-only loss;
- attention reset or separate examples rather than cross-example packed attention;
- fixed mixture weights;
- no arm-specific task sampling.

With global batch \(128\times2048=262{,}144\) tokens, 100M tokens is approximately **382 optimizer steps**. Use 400 steps for a simple fixed budget.

If a broad instruction mixture cannot be obtained, train for 3–10 SQuAD epochs and sharply restrict the paper to a sample-efficiency study. Another 100+ epoch SQuAD run is not scientifically useful.

## Matching axis

There is no single universal axis.

### Primary causal estimand: equal architecture and equal unique tokens

Match:

- depth;
- deployed parameters;
- data;
- unique tokens;
- steps;
- batch;
- checkpoint locations.

Under IF versus RF, this also matches:

- trainable parameters;
- updated tensors;
- optimizer state;
- approximately the training computation.

The same is true for IU versus RU.

### Secondary: equal cumulative training FLOPs

Record actual:

- forward FLOPs;
- activation backward FLOPs;
- parameter-gradient FLOPs;
- optimizer-update FLOPs.

Plot capability against cumulative measured FLOPs.

At equal FLOPs, a cheaper frozen arm may process more **unique** data—not additional repeats of the same tiny SQuAD corpus.

### Wall clock

Wall-clock is a systems endpoint, not the causal matching axis.

Report time to threshold on:

- the same GPU type;
- same GPU count;
- same software and precision;
- same batch policy.

For the factorial interaction, run at least two complete IF/RF/IU/RU seed blocks on B200 so hardware does not become perfectly confounded with freezing. Extra IF/RF seeds can run on H20 as paired blocks.

### Trainable parameters

Report trainable parameter counts, but do not force trainable-parameter equality across IF and IU; doing so changes the treatment.

The correct claim is:

> IF is non-inferior to IU while updating fewer parameters.

## Evaluation that is not format dominated

### Co-primary endpoint 1: LM integrity

Evaluate token NLL/PPL on at least two corpora:

- WikiText-103;
- PG19 or held-out Dolmino.

Use at least:

- 1M scored tokens per corpus;
- approximately 1,000 document/segment clusters.

Report:

- mean NLL;
- PPL;
- PPL ratio to intact base;
- paired document-level difference.

A lenient integrity gate is PPL no more than 2× intact base on either corpus.

### Co-primary endpoint 2: likelihood capability composite

Use likelihood-based MC, with full per-example scores:

- MMLU;
- HellaSwag;
- ARC-Challenge and ARC-Easy;
- PIQA;
- WinoGrande;
- BoolQ;
- CommonsenseQA or SocialIQA.

For each task:

\[
R_{a,t}
=
\frac{s_{a,t}-c_t}
{s_{\text{base},t}-c_t}.
\]

Average tasks equally, or average equally within prespecified families and then across families.

Report both:

- chance-normalized retention \(R\);
- candidate log loss or gold-choice NLL.

Log loss is more informative than accuracy when models are near chance.

### Target-task endpoint

Keep SQuAD EM as secondary. Primary target-task metrics should be:

- conditional gold-answer NLL;
- length-normalized gold-versus-hard-distractor margin;
- constrained span likelihood;
- answerable/unanswerable balanced accuracy.

For example:

\[
m_i=
\frac{\log p(y_i\mid x_i)}{|y_i|}
-
\max_k
\frac{\log p(\tilde y_{ik}\mid x_i)}{|\tilde y_{ik}|}.
\]

This measures whether the model prefers the correct content without requiring an exact generation format.

### Generation diagnostics

Report:

- repeated-response rate;
- largest identical-completion frequency;
- refusal rate;
- false-refusal rate;
- nonempty output rate;
- output entropy;
- answer length;
- EM/F1/contains as secondary results.

## Sample sizes and tests

### Training seeds

- **Six paired IF/RF seeds** are preferable for the primary contrast.
- Six is the minimum at which an exact two-sided sign-flip test can attain \(p<.05\) when all paired effects have the same direction.
- Five paired seeds are useful for effect-size estimation but the minimum exact two-sided sign-flip p-value is .0625.
- IU/RU should have at least two paired seeds under the strict budget and should be described as secondary unless expanded.

### Evaluation items

Use full test sets. As a rough unpaired binomial calculation at a baseline near 0.25:

- detecting 3pp with 80% power at two-sided \(\alpha=.05\) requires roughly 3,400 items per arm;
- detecting 2pp requires roughly 7,500.

Paired tests may require fewer depending on discordance, but many individual benchmarks remain underpowered for 2–3pp. This is another reason to use seed-aware hierarchical aggregation, not insist that every task be significant.

### Statistical tests

Primary:

- paired seed-level difference in macro \(R\);
- exact paired sign-flip/randomization test;
- seed bootstrap CI;
- hierarchical bootstrap over seeds→tasks→natural item clusters.

Secondary:

- difference-in-differences for the factorial interaction;
- paired document bootstrap for LM NLL;
- McNemar per task and seed;
- TOST for IF versus IU non-inferiority;
- Holm correction for secondary per-task tests.

Recommended non-inferiority margins:

- capability: \(\delta_R=0.05\);
- LM integrity: \(\delta_{\mathrm{NLL}}=\log(1.05)\), corresponding to at most 5% PPL degradation.

A nonsignificant IF−IU difference is not evidence that freezing is equivalent to updating.

## Explicit falsification

The claim:

> At equal depth, inherited frozen trunk blocks outperform random initialization.

is **falsified for practical purposes** if any of the following occurs:

1. The upper 95% CI for IF−RF is below the prespecified meaningful effect, e.g. \(+0.05\) macro-retention units.
2. IF−RF is zero or negative at the seed level.
3. The effect disappears when embedding, norm, and LM-head initialization are held common.
4. Both IF and RF fail the capability floor; then no capability-retention claim is established even if item-level \(p<.05\).
5. The result exists only on free-form SQuAD EM and not on LM NLL, candidate log loss, or likelihood-based capability.
6. The result reverses on Qwen3-8B, precluding a model-general claim.

The stronger freeze-graft claim is falsified if:

- IF is inferior to IU beyond the non-inferiority margin;
- an inherited-cap or native-top control matches or beats IF;
- deletion-only step zero already explains nearly all IF performance;
- or the supposed depth benefit disappears after matched controls are plotted.

---

# (d) Controls for the monotone depth curve

The existing A4 curve is not evidence for a method effect:

\[
0.2930,\ 0.3440,\ 0.3560,\ 0.4190.
\]

Increasing keep depth simultaneously:

- retains more pretrained computation;
- deletes fewer blocks;
- increases deployed parameters;
- increases inference FLOPs;
- moves the cut closer to the native readout stack.

Monotonicity is therefore expected.

## What should replace the raw curve

At each tested depth \(d\), report contrasts:

\[
\Delta_{\text{inherit,frozen}}(d)
=Y_{\mathrm{IF}}(d)-Y_{\mathrm{RF}}(d),
\]

\[
\Delta_{\text{inherit,updated}}(d)
=Y_{\mathrm{IU}}(d)-Y_{\mathrm{RU}}(d),
\]

\[
\Delta_{\text{freeze}}(d)
=Y_{\mathrm{IF}}(d)-Y_{\mathrm{IU}}(d),
\]

\[
\Delta_{\text{fresh cap}}(d)
=Y_{\mathrm{IF}}(d)-Y_{\mathrm{inherited\ cap}}(d).
\]

The method’s contribution is the **vertical gap to matched controls**, not the slope of the hero curve.

## Required controls

### 1. Random-frozen trunk

Same:

- depth;
- trainable subset;
- pretrained I/O components;
- fresh cap;
- optimizer;
- training data.

Only trunk-block initialization changes.

This is the cleanest inheritance control.

### 2. Inherit but do not freeze

IU identifies whether freezing is beneficial or merely a resource-saving restriction.

If IU materially outperforms IF, inheritance helps but freezing does not.

### 3. Random and updated

RU is required for the full factorial and tells whether random trunks can recover when allowed to adapt.

### 4. Contiguous inherited-cap control

Construct:

\[
P_0,\ldots,P_{j-1},P_j,\ldots,P_{j+K-1},
\]

freeze the first \(j\), and train the inherited next \(K\) blocks plus common head/norm components.

This matches:

- deployed depth;
- parameters;
- trainable cap size;
- approximate training compute.

If it matches or beats the fresh cap, random regrowth contributes nothing.

### 5. Native-top/readout control

Construct:

\[
P_0,\ldots,P_{j-1},P_{L-K},\ldots,P_{L-1}.
\]

This tests whether retaining the original final readout blocks is better than growing random ones.

If this arm wins, the important idea is transplanted pretrained readout—not fresh regrowth.

### 6. Deletion-only step-zero control

Evaluate the first \(d\) pretrained blocks without adaptation.

This is evaluation-only and costs no training run.

It establishes how much is retained simply by deleting fewer layers.

### 7. Full-depth top-only tuning

Keep all 32 layers, freeze the first 30, and train the top two plus common I/O components.

This is approximately trainable-parameter matched but not inference-compute matched. It gives a critical capability-versus-deployment-compute Pareto anchor.

### 8. Matched FLOP curves

Plot all arms against:

- downstream tokens;
- measured training FLOPs;
- wall-clock;
- inference FLOPs;
- deployed parameters.

No single curve answers every efficiency question.

## Interpretation patterns

- **All controls rise in parallel with depth:** result is “deleting fewer layers is better.”
- **IF−RF positive across viable depths:** useful inherited trunk computation.
- **IU≫IF:** inheritance helps, freezing hurts.
- **Inherited cap≈IF:** fresh cap unnecessary.
- **Native-top≫IF:** native readout blocks matter more than regrowth.
- **IF works only at 30 layers:** method supports modest compression, not aggressive shallow recovery.
- **Both IF and RF below the capability floor:** only failure-mode conclusions are allowed.

Closely related work makes these controls necessary rather than optional: top-layer reinitialization appears in “Revisiting Few-sample BERT Fine-tuning” (arXiv:2006.05987); gradual unfreezing in ULMFiT (ACL 2018, arXiv:1801.06146); selective block tuning in “Surgical Fine-Tuning Improves Adaptation to Distribution Shifts” (ICLR 2023, arXiv:2210.11466); and modern prune/replacement or block-expansion constructions in “Streamlining Redundant Layers to Compress Large Language Models” (arXiv:2403.19135), “LLaMA Pro: Progressive LLaMA with Block Expansion” (ACL 2024, arXiv:2401.02415), and “Reassessing Layer Pruning in LLMs” (arXiv:2411.15558). ([arxiv.org](https://arxiv.org/abs/2006.05987))

---

# (e) Minimum additional runs for a defensible paper

## Ruthless prioritization

Do **not** spend the next runs on:

- more 1000-step SQuAD depth points;
- more LoRA ranks;
- A1 full-FT merely as another ceiling;
- a large \(j\times K\) sweep;
- more single-seed capability evaluations;
- extra tasks evaluated on the broken keep14 checkpoints.

A2 and BASE already bracket the absolute capability regime. The missing evidence is causal identification and seed replication.

## Before any training: zero-run gates

These are evaluation/code tasks, not training runs:

1. Recover or regenerate per-example A4/A3 SQuAD predictions.
2. Report refusal versus answerable strata.
3. Verify A4 and A3 optimizer/code manifests.
4. Evaluate existing keep20/24/28 A4 checkpoints on PPL and likelihood capability.
5. Select \(D^*\) using the preregistered validation-floor rule.
6. Implement common pretrained I/O initialization and truly freeze the embedding.
7. Add answer-only masks and example-boundary attention isolation.
8. Unit-test that paired caps are bit-identical across arms.
9. Unit-test RF really randomizes only the intended trunk blocks.
10. Lock the test suite before confirmatory runs.

If no candidate depth clears the floor, stop the paper rather than launching 20 more failed models.

## If only 10 runs are available

A complete factorial is not possible with adequate replication. Narrow the paper to the one clean causal question.

| Runs | Model/depth | Arm | Seeds | Hardware |
|---:|---|---|---|---|
| 1–5 | OLMo-2-7B at \(D^*\) | IF | 11, 22, 33, 44, 55 | H20 or B200, paired by hardware |
| 6–10 | OLMo-2-7B at \(D^*\) | RF | 11, 22, 33, 44, 55 | Same hardware as paired IF |

This gives five paired seed differences.

What it can support:

> At fixed depth, fixed trainable subset, fixed I/O initialization, and equal tokens, pretrained frozen decoder blocks outperform—or do not outperform—random frozen decoder blocks.

What it cannot support:

- that freezing is preferable to updating;
- that fresh regrowth is useful;
- that the result generalizes to Qwen;
- a formal exact two-sided seed-level \(p<.05\), because five paired sign flips have minimum exact two-sided \(p=.0625\).

Thus 10 runs produce a strong estimate but a deliberately narrow paper.

## Minimum I would call defensible for the central OLMo mechanism: 16 runs

Add:

| Additional runs | Arm | Seeds | Hardware |
|---:|---|---|---|
| 11–12 | IF/RF sixth paired seed | 66 | Matched hardware |
| 13–14 | IU/RU | 11 | B200 |
| 15–16 | IU/RU | 22 | B200 |

Now there are:

- six paired IF/RF seeds;
- two complete factorial seed blocks;
- a seed-level exact test for the primary contrast;
- initial evidence on whether freezing helps.

This is the minimum defensible **single-model causal paper**.

## Recommended 20-run package

Use the following exact allocation.

### OLMo-2 primary inference: 16 runs

| Arm | Seeds | Count | Preferred node |
|---|---|---:|---|
| IF | 11, 22 | 2 | B200 |
| RF | 11, 22 | 2 | B200 |
| IU | 11, 22 | 2 | B200 |
| RU | 11, 22 | 2 | B200 |
| IF | 33, 44, 55, 66 | 4 | H20 |
| RF | 33, 44, 55, 66 | 4 | H20 |
| **Subtotal** |  | **16** |  |

This provides:

- six paired IF/RF seeds;
- two complete factorial seeds on one hardware class;
- extra primary precision using H20;
- hardware blocked within every paired contrast.

### Cap mechanism controls: 2 runs

| Arm | Seed | Count | Node |
|---|---:|---:|---|
| Contiguous inherited cap | 11 | 1 | B200 |
| Native-top/readout cap | 11 | 1 | B200 |
| **Subtotal** |  | **2** |  |

These are mechanism screens rather than definitive equivalence tests. If either is competitive with IF, the next allocation should replicate that control rather than add benchmarks.

### Qwen3 external-validity smoke pair: 2 runs

| Model | Arm | Seed | Count | Node |
|---|---|---:|---:|---|
| Qwen3-8B | IF | 11 | 1 | H20 |
| Qwen3-8B | RF | 11 | 1 | H20 |
| **Subtotal** |  |  | **2** |  |

Choose the Qwen cut to match the OLMo retained-depth fraction, with \(K=2\).

This is not a confirmatory Qwen replication. It is a preregistered external-validity check:

- agreement strengthens plausibility;
- reversal forbids a model-general claim;
- a null result remains inconclusive because \(n_{\text{seed}}=1\).

### Total

\[
16+2+2=\boxed{20\text{ training runs}}.
\]

## Training budget per run

Recommended:

- 100M unique SFT tokens;
- global batch 128×2,048 tokens;
- approximately 400 optimizer steps;
- 5% warmup;
- checkpoints at step 0, 25, 50, 100, 200, and 400;
- primary endpoint at fixed 100M tokens;
- validation AUC as secondary;
- no test-based checkpoint selection.

Based on the observed local throughput anchors:

- the existing 16-layer frozen H20 run was about 7.34 seconds/step;
- the B200 22-layer all-trainable run was about 2.57 seconds/step.

For 400 steps, a conservative planning range is:

- frozen H20 run: roughly 1–2 training hours;
- full-trainable B200 run: roughly 0.4–1 hour;
- add evaluation and checkpoint overhead separately.

The 20-run package should therefore be feasible in roughly **one to two wall-clock days** with all five nodes used efficiently, with an approximate total of **150–250 GPU-hours including evaluation**, subject to actual depth and data-loader throughput.

## Scheduling

### B200 nodes

Run complete factorial blocks so hardware does not confound the interaction:

- B200-1: seed 11 IF→RF→IU→RU;
- B200-2: seed 22 IF→RF→IU→RU.

The order within each block should be randomized or counterbalanced.

Then run the two cap controls on B200.

### H20 nodes

Use the three H20 nodes for:

- IF/RF seeds 33–66, always pairing each seed within the same node type;
- Qwen IF/RF smoke pair;
- evaluation once training completes.

Do not run two serious jobs on one node.

## Stop/go rules

After the first two complete factorial seeds:

- **Stop for futility** if IF fails the capability floor and IF−RF is nonpositive on both seeds.
- **Continue primary replication** if IF clears the floor or IF−RF is consistently positive.
- **Redirect remaining runs to IU** if IF is degenerate but IU is healthy.
- **Redirect remaining runs to cap-control replication** if either inherited-cap control matches IF.
- **Do not add more benchmarks** when the unresolved uncertainty is training-seed variance.

---

# Final recommendation

The present paper should not headline the current +3.25pp as a clean causal result. It is a one-seed, optimizer-confounded, LR-confounded, severely overtrained comparison on an evaluation set where a constant Chinese refusal obtains 49.85% EM.

The next paper-quality experiment is not another depth point. It is:

> a seed-replicated, fixed-depth inheritance × freezing factorial with common pretrained I/O components, common optimizer and role-specific LR policy, non-format-dominated likelihood/PPL endpoints, and explicit capability and nondegeneracy gates.

The claim is defensible only if all of the following hold:

1. IF clears the capability and LM-integrity floors.
2. IF−RF is positive and practically meaningful across paired seeds.
3. IF is non-inferior to IU, or the paper drops the “freezing is beneficial” claim.
4. Fresh cap beats the inherited-cap/native-top alternatives, or the paper drops the “regrowth” claim.
5. The result is present on likelihood/PPL endpoints, not only SQuAD EM.
6. The Qwen smoke result does not reverse the main effect.

Otherwise, the scientifically correct conclusion is narrower:

> inherited initialization delays collapse in aggressively truncated models, but the current freeze-graft construction has not been shown to retain useful general capability.