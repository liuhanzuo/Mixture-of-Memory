# Responsible NLP Checklist Draft — Paper B

This file is for the ARR submission form and is not included in the anonymous
paper PDF.

## A. Scope, limitations, and risks

- **A1 — Yes.** See the dedicated **Limitations** section after the conclusion.
- **A2 — Yes.** See **Ethical Considerations** in the appendix. It discusses the
  risk of treating low perplexity as a deployment certificate, inherited model
  harms, unequal capability retention, and energy use.

## B. Scientific artifacts

- **B1 — Yes.** The paper cites OLMo-2, Dolma/Dolmino, every benchmark, and the
  pruning/retraining methods used for comparison.
- **B2 — Partially yes.** Artifact licenses and usage restrictions should be
  enumerated in the anonymous README before submission. Any release must retain
  the upstream model, corpus, and benchmark terms.
- **B3 — Yes.** Released artifacts are used for research compression and
  evaluation, consistent with their documented purposes. The release should not
  redistribute model weights or benchmark data.
- **B4 — No new data were collected.** We do not redistribute examples. The
  study uses public benchmark data under upstream handling and licensing.
- **B5 — Yes.** Appendix B describes English-language evaluation domains,
  prompts, metrics, and task scope.
- **B6 — Yes.** Appendix B reports train/validation/test choices, actual sample
  counts, truncations, and zero-NaN integrity checks.

## C. Computational experiments

- **C1 — Partially yes.** The paper reports model scale, GPU setup, training
  steps, and checkpoint provenance. The anonymous README should additionally
  provide a conservative total GPU-hour estimate over all completed runs.
- **C2 — Yes.** Appendix B reports data preparation, optimizer, learning-rate
  schedules, batch size, sequence length, reconstruction, stopping decisions,
  and evaluation protocols.
- **C3 — Yes, with limitations.** The paper reports paired bootstrap intervals,
  exact McNemar tests, and marginal confidence intervals. Most training arms are
  single runs; this is explicitly stated in Limitations.
- **C4 — Yes.** Appendix B and the anonymous README should list Python,
  PyTorch, Transformers, Datasets, and evaluator versions and explain all
  architecture modifications.

## D. Human participants

- **D1–D5 — Not applicable.** No human participants or new annotators were
  recruited.

## E. AI assistance

- **E1 — Yes.** Generative AI assistants were used for literature search,
  low-novelty drafting and language revision, coding assistance, table/figure
  preparation, and critical review. GPT Image 2 generated an initial Figure 1
  concept from an author-written prompt; AutoFigure-Edit was used to import and
  reconstruct the figure, and the final editable vector was manually rebuilt
  and scientifically checked. Human authors determined the ideas, methods,
  experimental design, interpretation, and citations; they verified all text,
  code, results, and references and take full responsibility. No AI system is
  an author.

## Acknowledgement text (for a non-anonymous final version)

> Generative AI assistants were used for literature discovery, language
> revision, coding support, figure preparation, and critical review. GPT Image
> 2 and AutoFigure-Edit supported the initial and editable Figure 1 workflow.
> The authors manually verified all methods, code, figures, numerical results,
> and citations and remain fully responsible for the paper.
