# Anonymous evaluation artifact snapshot

This directory is a compact snapshot of currently publishable Paper B artifacts.
It contains the evaluator source files used for PPL, zero-shot likelihood tasks,
dual-interface MMLU, closed-book QA, and paired item analysis; sanitized protocol
and provenance manifests; aggregate closed-book summaries; prompt-free MMLU
per-item score records for the six headline arms; and the prompt-free keep14 128k/200k
paired trajectory records. It now also includes the three full-split
ShortGPT closed-book summaries and prompt-free per-item score records, the
six-arm WikiText-103/PG-19 PPL table, and aggregate continuation-stream
contamination statistics. The file
`scores/paired/same_snapshot_letter_endpoint_comparisons.json` recomputes the
keep14--Random, keep14--Frozen, and Frozen--Random letter-MMLU comparisons
directly from the six-arm `scores/mmlu_content/` snapshot, using 10,000 paired
bootstrap resamples with seed 0 and exact two-sided McNemar tests.

The per-item JSONL files contain stable item IDs, subject labels where applicable,
gold/predicted option indices or letters, option scores, continuation token counts,
and correctness. ShortGPT closed-book records contain only stable item ID and
EM/containment/F1 scores. They do **not** contain benchmark questions, aliases,
predictions, option text, prompts, model weights, credentials, private URLs, or
private filesystem paths.

The original task-specific evaluator revisions were local-only commits. Their
currently available file contents are snapshotted here and the abbreviated commit
IDs are recorded in `manifests/local_commit_snapshot.json`. This does not claim
that the historical commit graph is anonymously recoverable, nor that the missing
historical training seed or the lost keep14 within-epoch loader offset can be
reconstructed. The released scripts pin public dataset revisions where available;
closed-book dataset names/splits and exact scoring rules are recorded in
`configs/evaluation_protocol.json`.

The contamination files report only aggregate counts/rates and protocol
thresholds; raw benchmark text and matched continuation-stream examples are
excluded.

Not included: benchmark text, closed-book generations or answer strings,
checkpoints/model weights, training arrays, credentials, and any inferred
seed/offset history. Aligned prompt-free closed-book records for every non-
ShortGPT arm were not all consolidated, so paired closed-book intervals are not
claimed.
