# Post-pruning recovery study — anonymous paper artifacts

This directory contains the review manuscript, figure sources, and a compact
`anonymous_artifact/` snapshot for a study of likelihood and target recovery
after depth pruning and continued pretraining. Model weights and benchmark text
are not redistributed.

Figure 1 began from an author-written prompt rendered with GPT Image 2, was
imported through AutoFigure-Edit, and was manually rebuilt and verified as an
editable SVG. Full AI-assistance disclosure is prepared for the Responsible NLP
Checklist.

The anonymous artifact snapshot contains publishable evaluator scripts,
sanitized configs/manifests, aggregate closed-book summaries, prompt-free
per-item MMLU score records for the six headline arms, and the available keep14
paired trajectory records. It also includes prompt-free ShortGPT closed-book
scores, endpoint OOD-PPL summaries, and aggregate continuation-stream overlap
statistics. Local-only evaluator commits are represented by a source snapshot
plus abbreviated provenance IDs; no public commit ancestry, historical seed,
or lost loader offset is claimed.
