"""Standard KV-cache compression baselines (Paper A P1.6).

Vendored official implementations of prefill-then-compress KV baselines
(SnapKV, PyramidKV) integrated onto Qwen3-8B / transformers 5.14. See
``src/baselines/PROVENANCE.md`` for upstream commit hashes and the porting
notes (the upstream monkeypatches target transformers 4.37 + Llama/Mistral and
cannot run as-is on this env; the *compression clusters* are vendored verbatim
and driven by a faithful Qwen3-5.14 attention hijack).
"""
