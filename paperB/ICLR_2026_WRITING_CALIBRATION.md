# ICLR 2026 Writing and Review Calibration — Paper B

**Collected:** 2026-08-03. Official ICLR 2026 titles/abstracts and acceptance were checked in proceedings. Several OpenReview pages presented challenge verification; no unverified reviewer quotations or paper-specific scores are asserted here.

## Core positioning

Paper B is strongest as a **measurement study of proxy validity after structural intervention**, not as a new compression algorithm or a general theory of knowledge recovery:

> In-domain perplexity can be a useful training signal, but it is not by itself a certificate that the evaluated capabilities have recovered after depth pruning and regrowth.

The bounded claim supported by the paper is:

> In the measured OLMo-2 constructions, training interval, and evaluation interfaces, likelihood and knowledge-sensitive behavior follow different observed paths; short-horizon corpus shift, one scoring interface, and nominal depth alone do not fully explain the observations.

Avoid claims that knowledge was removed, stored in particular layers, or permanently failed to recover.

## Relevant ICLR 2026 writing patterns

### Reassessing Layer Pruning in LLMs: New Insights and Methods

- Proceedings hash: `40624bef9a7d22c0c780dfe9291f5843`
- OpenReview: `04Tfwy3LLC`
- Pattern: deployment need → unresolved practical questions → systematic benchmark → simple reproducible rule → theory as explanation rather than sole causal proof.
- Transfer: ask three bounded questions: whether likelihood and task behavior recover synchronously; how much scoring interface explains; whether construction changes same-depth endpoints.

### Rethinking Layer Relevance in Large Language Models Beyond Cosine Similarity

- Proceedings hash: `4aa5e0b33e6f774c0d34ab2e1bc3bced`
- Pattern: popular proxy → theoretical counterexample → cross-model correlation audit → intervention-grounded criterion → bounded methodological conclusion.
- Transfer: “Perplexity is not poor in general; it is insufficient as a proxy for capability recovery in this intervention.”

### Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers

- Proceedings hash: `7ab63a5314680e2f083cb288abeaeb8e`
- OpenReview: `USyGD0eUod`
- Pattern: widely used metric → strong null condition → aggregate metric cannot distinguish meaningful from null → useful but insufficient → report null baseline plus targeted measure.
- Transfer: make the same-shape random control a headline null baseline. Its content-score floor while answer-letter accuracy remains at chance questions complete-option accuracy as a recovery certificate.

### Training Dynamics Impact Post-Training Quantization Robustness

- Proceedings hash: `6937a7c60361d05f5b6cfa04d2c27a5b`
- OpenReview: `ZXr3Xx7Z1O`
- Pattern: trajectory evidence first, schedule events marked explicitly, observational divergence separated from controlled intervention.
- Transfer: x-axis is actual step/token; show LR schedule and actual checkpoint positions; say “observed stabilization over available checkpoints,” not convergence or a uniform plateau unless stopping was pre-specified.

### Revisiting the Scaling Properties of Downstream Metrics in Large Language Model Training

- Proceedings hash: `c89876c031d1686a5777a87b5624e14c`
- OpenReview: `YnJ2s4WeNF`
- Pattern: constrain the setting, distinguish fit from extrapolation, report out-of-sample error, release intermediate checkpoints.
- Transfer: characterize observed paths over the measured budget; do not infer a universal recovery law from keep14.

### Identifying and Evaluating Inactive Heads in Pretrained LLMs

- Proceedings hash: `e491ad4c74b24b6d3fd92adcd0923fad`
- Pattern: multiple proxy definitions → direct intervention validates construct → report removable fraction under explicit tolerance.
- Transfer: logit-lens/linear-probe evidence is only a descriptive, readout-dependent correlate without intervention; do not claim a knowledge-storage layer.

## Recommended claim ladder

1. **Direct observation:** keep14 in-domain PPL and MMLU change along different measured paths.
2. **Cross-interface recurrence:** deficits occur for answer-letter MMLU and closed-book QA; content scoring only partially changes the gap.
3. **Bound alternatives:** full32 short-horizon control does not make corpus shift a complete explanation; random initialization reveals a content-score fluency floor; ShortGPT shows nominal depth alone does not determine the endpoint.
4. **Explicit non-claims:** no localization of knowledge, no proof that recovery can never occur, no universal dynamics claim, no claim that perplexity is generally useless.

Preferred language:

| Avoid | Prefer |
|---|---|
| knowledge recovery | performance on knowledge-sensitive evaluations |
| structural damage | post-intervention deficit |
| capability was restored | performance improved/reached X |
| controls separate interpretations | controls bound several alternative interpretations |
| robust across interfaces | recurs across the tested interfaces and datasets |
| knowledge lags perplexity | evaluated capabilities change at a different observed rate |
| layers store knowledge | layer-wise readouts are descriptive correlates |
| converged/plateaued | stabilized over the available checkpoints |

## Abstract recommendation

Use a proxy-validity/null-baseline structure:

1. Depth-pruned models are often trained until PPL improves, implicitly using likelihood recovery as evidence of capability recovery.
2. Test that assumption with OLMo-2-7B trajectories, two MMLU interfaces, and three closed-book QA datasets.
3. Report keep14 endpoint: PPL 10.561 at 200k, MMLU .319 versus .605 base, with closed-book deficits.
4. State that full32 remains near base over its observed 25k likelihood interval; this only argues against short-horizon corpus shift as a complete explanation.
5. Content scoring narrows the gap, but a same-shape random-init model reaches a similar content-score floor while answer-letter performance remains at chance.
6. A different 16-layer ShortGPT construction reaches .474, showing construction dependence rather than a clean depth effect.
7. Conclude only that in-domain PPL is an insufficient certificate for the evaluated capabilities and that likelihood, task, interface, construction, and recovery compute should be reported separately.

## Contributions

Replace “controls that separate interpretations” with:

> **Controls that bound alternative explanations.** Continued-full-model, same-shape initialization, scoring-interface, closed-book, and alternative-construction comparisons test corpus shift, fluency floors, answer interfaces, and construction dependence, while leaving unmatched-budget and coupled-structure effects explicit.

“Separate” would imply clean causal isolation, which the random LR, frozen trainable set, and coupled ShortGPT construction do not provide.

## Figure 1

Figure 1 should answer one question: **Does likelihood recovery certify capability recovery?**

Recommended panels:

- **Trajectory:** x = CPT steps/tokens; left y = in-domain PPL; right y = MMLU letter; base and chance references; caption states one training run per construction.
- **Endpoint falsification:** keep14, random-init, frozen-front, and ShortGPT, showing PPL and MMLU together; highlight random content floor and construction dependence.

Move layer-wise probes out of Figure 1. If reporting normalized above-chance recovery, define the formula in the caption.

## Main table

Expose protocol differences in columns, not footnotes:

| Construction | Inherited/Fresh | CPT steps | PPL domain | MMLU letter | MMLU content | QA |
|---|---:|---:|---|---:|---:|---:|

Separate reference/diagnostic controls (base, full32@25k) from half-depth operating points (keep14, ShortGPT, frozen, random). Explicitly mark inherited/fresh counts, 25k versus 200k, and em dash as not evaluated. Use consistent precision and remove unexplained markers.

At first use, not only in Limitations, state: single run per construction; full32 only 25k; ShortGPT changes multiple factors; PPL is in-domain on a same-source held-out shard.

## Review calibration

Operational ICLR-style score semantics for future reviewer prompts:

- **8 / Accept:** important claim with evidence likely robust to model family, seeds, budget, and metric variation.
- **6 / Weak Accept:** careful, reproducible, useful bounded measurement study; no new algorithm required.
- **5 / Weak Reject:** reasonable idea but one key novelty/control/evidence gap.
- **3 / Reject:** main conclusion rests on invalid comparisons, unsupported causal interpretation, or missing key validation.
- 10 and 1 are exceptional extremes.

Strict reviewers will focus on prior knowledge that PPL and downstream metrics differ, 25k-vs-200k mismatch, single-run dynamics, coupled ShortGPT differences, whether MMLU/QA operationalizes “knowledge,” item uncertainty versus seed uncertainty, and causal misuse of probes. A strict score should be capped at 5 if unmatched operating points are used causally, behavioral deficits are called knowledge deletion, readouts are called storage locations, or content-score improvement is treated as recovery without the random floor.

Normal reviewers can reasonably score a bounded measurement study at 6 if the question is clear, controls are more systematic than prior work, results and protocols are reproducible, conclusions are candid, and the reporting recommendation is useful. Packaging it as a knowledge mechanism or novel pruning method invites a 5.

Likely current role calibration:

| Role | ICLR-like score | Main issue |
|---|---:|---|
| Strict compression | 5 | unmatched budgets; coupled ShortGPT; no new SOTA method |
| Strict measurement/statistics | 5 | single runs; non-uniform stopping; item CI is not seed variance |
| Normal compression | 6 | useful trajectory and controls |
| Normal evaluation | 6 | dual interfaces, closed-book checks, random floor |
| Interpretability | 5–6 | depends on eliminating causal/storage readings of probes |
| Meta-reviewer | 5–6 | bounded usefulness versus narrow scope |

Highest-risk critiques:

1. “PPL differs from downstream performance” is already known.
2. Comparisons use 25k versus 200k, different LRs, and different trainable sets.
3. MMLU/QA are overclaimed as knowledge.
4. Probe depth is overclaimed as storage/damage location.
5. One family/scale/mixture limits generality.
6. Single runs give no run-level uncertainty.
7. Main tables hide protocol differences.

The best defense is not to claim the controls are complete, but to define the contribution as a rich-control demonstration of proxy insufficiency in a clearly bounded setting plus a reporting protocol for future work.
