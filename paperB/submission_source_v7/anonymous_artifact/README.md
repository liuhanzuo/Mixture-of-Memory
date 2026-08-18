# Anonymous evaluation artifact snapshot

This directory is a compact snapshot of currently publishable Paper B artifacts.
It contains the evaluator source files used for PPL, zero-shot likelihood tasks,
dual-interface MMLU, closed-book QA, and paired item analysis; sanitized protocol
and provenance manifests; aggregate closed-book summaries; prompt-free MMLU
per-item score records for the six headline arms; and the prompt-free 128k/200k
keep14 paired trajectory records.

The per-item JSONL files contain stable item IDs, subject labels where applicable,
gold/predicted option indices or letters, option scores, continuation token counts,
and correctness. They do **not** contain benchmark questions, option text, prompts,
model weights, credentials, private URLs, or private filesystem paths.

The original task-specific evaluator revisions were local-only commits. Their
currently available file contents are snapshotted here and the abbreviated commit
IDs are recorded in `manifests/local_commit_snapshot.json`. This does not claim
that the historical commit graph is anonymously recoverable, nor that the missing
historical training seed or the lost keep14 within-epoch loader offset can be
reconstructed. The released scripts pin public dataset revisions where available;
closed-book dataset names/splits and exact scoring rules are recorded in
`configs/evaluation_protocol.json`.

Not included: benchmark text, closed-book generations/per-item predictions (not
consolidated), ShortGPT closed-book results (not run), checkpoints/model weights,
training arrays, credentials, and any inferred seed/offset history.
