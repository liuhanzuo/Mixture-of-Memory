---
summary: 3
soundness: 3
excitement: 3
reproducibility: 3
confidence: 4
---

# Independent ARR-style review — Paper B v8

## Summary and overall assessment

This is a substantially clearer and more defensible version of the paper. Its best contribution is no longer the familiar observation that post-pruning perplexity and task performance can diverge. Instead, it is a **measurement argument**: a recovery claim should be checked across trajectory, evaluation interface plus null floor, construction, budget, and uncertainty. The main OLMo-2-7B evidence is coherent: `keep14+fresh2` improves in held-out in-domain PPL from 10.826 (128k) to 10.561 (200k) and gains 1.68 MMLU-letter points, yet remains 28.62 points below the intact base; complete-option MMLU raises both keep14 and the random 16-layer model, with the latter reaching nearly the same content score as Frozen; and a ShortGPT-derived 16-layer construction reaches 0.474 letter-MMLU versus 0.319 for keep14 at the same nominal 200k-step budget.

The revised organization works. The new “multi-axis recovery audit” gives the paper a specific, appropriately scoped thesis, and moving the three core tables into Results makes the argument much easier to evaluate. The manuscript is candid about its central weaknesses: one training run per headline construction, a 25k-only full32 control, confounded ShortGPT comparison, in-domain PPL, missing ShortGPT closed-book scores, and incomplete historical training provenance.

However, the paper still overreaches at two important points. First, **“to our knowledge, the first multi-interface recovery audit” is not yet defensible as a literal priority claim from the provided literature audit**. It is defensible only as a claim about the paper’s exact *combination* of ingredients. Second, ShortGPT substantially improves the paper’s **practical importance**, but not yet its causal or mechanistic insight: it is a single, heavily confounded comparison and its large effect is strikingly interface-dependent (15.5 pp on letter MMLU, only 1.8 pp on content MMLU). The manuscript mostly acknowledges this, but the abstract/conclusion rhetoric should match that evidential limit more exactly.

**Recommendation: Weak Reject / borderline.** I would be positive after targeted claim calibration and a small set of presentation/reproducibility corrections. The paper has a useful and timely measurement contribution, but its evidence is currently a strong single-case audit rather than a broadly validated recovery methodology.

## What is effective in v8

### 1. The Measurement/Evaluation repositioning is effective

The paper now has a crisp construct-validity question rather than an underpowered claim about a universally valid pruning phenomenon. The three axes are well chosen:

- **Trajectory:** PPL decreases while the large intact-base target gap remains.
- **Interface:** the same 14,042 MMLU items are scored in two protocols, and a random same-shape floor prevents a naïve reading of the content-interface gain.
- **Construction:** two nominally 16-layer models with the same displayed step budget reach materially different endpoints.

This reframing makes the controls meaningful. In particular, the random floor is not merely another baseline: it supports a concrete interpretation that a higher complete-option score alone cannot certify inherited target recovery. The paper also correctly avoids treating its item-level bootstrap/McNemar results as run-level uncertainty.

### 2. The more confident innovation narrative is partly earned

The abstract, opening figure, and contributions now tell one consistent story. The paper is right to say that its novelty is the **joint audit**, not the generic fact that perplexity can fail to track downstream outcomes. The Results section delivers the three advertised diagnostics rather than deferring them to an appendix. This is a real improvement in clarity and reviewer confidence.

The confidence is strongest when it is tied to the exact design: same-source PPL, within-path checkpoints, paired MMLU interfaces, three closed-book datasets, an intact CPT point, null operating points, and a same-depth construction contrast. It becomes too strong when phrased as an unqualified “first.”

### 3. The ShortGPT comparison raises excitement, but only as a design signal

The ShortGPT result is the most compelling new empirical hook. It replaces a purely negative “PPL is insufficient” message with a constructive one: **the chosen pruned construction can matter as much as subsequent recovery training**, so “16 layers” is an inadequate experimental descriptor. The broader endpoint table helps: ShortGPT exceeds keep14 on every reported multiple-choice task, not only MMLU.

Still, this increases excitement more than it establishes explanation. The comparison simultaneously changes (i) inherited block count (16 vs. 14), (ii) contiguous prefix vs. BI-selected non-contiguity, (iii) retention of original block 31, and (iv) fresh-tail insertion. Therefore it does not show that ShortGPT’s selection rule is responsible, nor that final-layer retention is responsible. The paper states this limitation, but the headline claim should remain “construction-sensitive endpoint” rather than “ShortGPT reveals why recovery fails.” The unusually large letter/content discrepancy itself is informative, but it also narrows what can be inferred about knowledge recovery.

## Required final revisions

### Major 1 — Replace the literal priority claim with a bounded, auditable claim

The phrase in the abstract, “**to our knowledge, the first multi-interface recovery audit**,” is not adequately established by the supplied related-work review. Gromov, Shortened LLaMA, Minitron, and IteRABRe already cover important subsets of post-pruning recovery trajectories and loss/task evaluation; Cost of Compression and Beyond Perplexity establish broader multidimensional compression evaluation. The paper’s Table 1 is helpful, but it is a self-curated binary coding of only a small nearest-work set, and it cannot prove global priority.

Use one of the following safer formulations everywhere (abstract, introduction, conclusion, and caption):

> “To our knowledge, this is the first **post-depth-pruning study we found to combine** same-source PPL trajectories, paired MMLU scoring interfaces with a random null floor, closed-book QA, an intact CPT reference, and an alternative same-depth construction in one audit.”

or, more concise:

> “We present a **multi-interface recovery audit** that combines …”

This retains the innovation while making the claim commensurate with the evidence. Also change “Our literature audit found no prior …” to “Among the post-depth-pruning studies we identified, none combines …” and make the scope of the search explicit (databases/search date/query or a short appendix protocol). Given that the bibliography includes 2026 concurrent work, this needs particular care.

### Major 2 — Calibrate the ShortGPT claim and show the evidential asymmetry prominently

The paper correctly says the ShortGPT contrast is confounded, but the abstract and conclusion still risk being read as evidence for a same-depth *causal construction effect*. Make the main statement:

> “At the same nominal 16-layer/200k-step operating point, the observed ShortGPT construction is substantially stronger; because it changes four factors jointly, this is a design diagnostic rather than an attribution to the selection method.”

Two concrete edits would make this persuasive rather than defensive:

1. In Table 5 and its surrounding prose, replace “same depth / 200k” with “same **nominal** depth / displayed step budget; inherited-block count and construction differ.” This is already present in prose, but needs to be visually impossible to miss.
2. In the abstract and conclusion, pair the 15.5 pp letter-MMLU difference with the 1.8 pp content-MMLU difference in the same sentence. That prevents the result from being mistaken for an unqualified 15.5-point knowledge-recovery advantage.

The current ShortGPT analysis does increase excitement, but it should be framed as an unusually strong **factorial-experiment target**, not a resolution of the mechanism.

### Major 3 — Repair source/artifact numerical inconsistencies before submission

I checked the released v8 summaries and per-item MMLU files. Several main-text numbers are internally consistent at displayed precision, but some exact paired-control values do not match the released per-item MMLU artifacts:

- Table 6 reports keep14–Random = **+7.11** pp, keep14–Frozen = **+5.50** pp, and Frozen–Random = **+1.61** pp. The v8 per-item files supplied with the paper yield **+7.14**, **+5.60**, and **+1.54** pp, respectively, using the Table 4 endpoint scores. The associated discordant counts also differ from the JSON file `keep14_vs_random.json`.
- The per-item summary records give keep14 letter/content = **.318402/.383208**, Random = **.246973/.359778**, and Frozen = **.262356/.360419**. These agree with Table 4 to four decimals, but imply the above differences rather than Table 6’s values.
- Conversely, the 128k→200k trajectory is well supported: the paired artifact gives **+1.6807 pp**, CI **[1.075, 2.286]**, and the stated exact McNemar p-value.

This may reflect multiple evaluation reruns or a stale paired-analysis output, but a reviewer cannot reconcile it from the v8 bundle. Regenerate all headline tables and paired analyses from a single declared snapshot, include the exact checkpoint/file identifiers used for each, and make the artifact names match table terminology (`Random` versus `scratch16L`; `Frozen` versus `freezefront`). This is essential because measurement integrity is the paper’s main contribution.

### Major 4 — Distinguish “same nominal budget” from a matched recovery comparison

The full32 25k point usefully rules out the simplest immediate same-corpus-shift story, but it does not support an endpoint comparison with keep14 at 200k. The paper acknowledges this well. Tighten the abstract and conclusion from “full32 remains much closer” to “the **available 25k** full32 point is closer to base,” and avoid any wording that sounds like full32 is a matched intact recovery control.

Likewise, “same nominal depth and 200k-step budget” should not be allowed to imply token-, compute-, data-order-, or optimization-matched training. The table does expose LR and trainable set, which is good; add “single-run operating points” directly in the short caption/title of the main endpoint table.

### Minor 1 — Strengthen the literature-positioning table or reduce what it bears

Table 1 is useful but too compressed to substantiate the firstness claim. A few cells depend on nuanced definitions (“trajectory,” “intact CPT,” “MMLU interface,” “closed-book,” “construction”) and do not indicate whether the cited work has a closely related non-MMLU alternative. Add an appendix table with short evidence pointers (section/table/figure in each cited work) or make Table 1 explicitly illustrative rather than exhaustive.

The treatment of the requested nearest work is otherwise directionally appropriate:

- **Gromov:** correctly recognized as the closest precedent for loss/task dissociation after depth removal.
- **Shortened LLaMA and Minitron:** correctly prevent the paper from claiming trajectories, recovery controls, or initialization comparisons as new in isolation.
- **IteRABRe:** correctly prevents a claim that iterative recovery or weak MMLU recovery is novel.
- **Cost of Compression / Beyond Perplexity:** correctly motivate broader evaluation, but they make it especially important not to imply the paper discovered the general inadequacy of perplexity.

### Minor 2 — Improve table readability without moving them back to the appendix

The three relocated diagnostic tables are in the right place and are readable. Table 2 is dense but legible in the rendered PDF. The main readability issue is **Table 8**, whose 13 narrow columns and abbreviations make it difficult to extract the ShortGPT message at normal zoom. Consider moving the complete 11-task matrix to the appendix and retaining a compact main-text table with MMLU, core6, and 3–4 representative tasks, or visually group columns and bold the ShortGPT–keep14 deltas. This does not weaken the argument; it makes the new construction finding more reviewable.

For Tables 2–6, define `MMLU-L` and `MMLU-C` immediately in the column header/caption on first occurrence, not only in Table 2’s caption. Keep the explicit “same 14,042 items” language in Table 4—it is excellent.

### Minor 3 — Clean appendix migration artifacts

The appendix correctly retains complementary details rather than duplicating the three core tables verbatim. That migration is mostly successful. However, the paragraph before Appendix Table 16 contains a visible copy-edit error: “**The late-healing The late keep14 audit** has moved …”. Fix it.

Also revise the nearby sentence that says “the full 11-task checkpoint trajectory remains available in Table~\ref{tab:downstream}”: Table 8 is now in the main paper, not the appendix, and it is an endpoint/profile table rather than a complete within-arm trajectory. This is a small but revealing inconsistency after the table move.

### Minor 4 — Preserve the good structural cleanup

The requested placement checks pass:

- **Conclusion begins on page 8** and is complete there.
- **Limitations** and **Ethical Considerations** are unnumbered, start on a separate **page 9**, and do not spill into the references.
- **References begin on page 10**.

Do not alter the float barriers/clear-page setup in a way that regresses this. The conclusion’s one-column start on page 8 is slightly visually abrupt but acceptable.

## Score rationale

- **Summary: 3/5.** Clear thesis, meaningful audit design, and useful empirical finding; however, it is a narrowly scoped single-case study with limited causal identification.
- **Soundness: 3/5.** The core descriptive conclusions follow from the measurements, and limitations are unusually transparent. Scores are capped by single-run evidence, unmatched full32 horizon, confounded ShortGPT comparison, in-domain PPL, and unresolved v8 artifact/table discrepancies.
- **Excitement: 3/5.** The null-calibrated interface argument and the ShortGPT construction gap make this more interesting than a standard “PPL is not enough” paper. Excitement would rise with even one targeted factorial control or a matched intact long-horizon run.
- **Reproducibility: 3/5.** The source bundle includes evaluator scripts, manifests, summaries, and MMLU per-item records. But missing seeds, loader offset, training provenance, closed-book prediction files, and the paired-number mismatch prevent a higher rating.
- **Confidence: 4/5.** The paper is sufficiently clear to assess, and I directly checked the submitted PDF plus its corresponding v8 source/artifact bundle. The remaining uncertainty concerns experimental provenance and the external literature-priority claim, not the manuscript’s basic intent.

## Final recommendation

This version has a credible and potentially publishable measurement contribution. I would encourage revision rather than a broad rewrite: **bound the novelty claim, describe ShortGPT strictly as a confounded but valuable construction diagnostic, reconcile every headline statistic to one artifact snapshot, and clean the appendix/table migration.** Those changes would make the paper’s confident new narrative match the actual strength of its evidence.
