# CacheBlend-style baseline artifact snapshot

This directory snapshots the aggregate and representative protocol evidence for the same-backbone CacheBlend-style baseline integrated into Paper A on 2026-08-04.

- Raw evaluation root at collection time: `<PROJECT_ROOT>/bench_results/cacheblend/` on an anonymous H20 evaluation cluster.
- `aggregate.json`: all required 48 RULER, 24 BABILong, and 4 full-LoCoMo ratio cells; `all_tasks_reported=true`, `missing_required_cells=[]`.
- `selftest.out`: real Qwen3-8B fp32 gate for RoPE reindexing and the `r=1` full-prefill ceiling.
- `ruler_sample.json` and `babilong_sample.json`: representative per-shard configs including 147,456 B/token, recompute ratio, latency, peak memory, selector, chunking, and runtime.
- `REMOTE_PROVENANCE.txt`: remote aggregate hash, script hashes, raw-file count, and checkout HEAD.
- `aggregate_cacheblend_143.py`: exact CPU aggregation logic.

The baseline is a minimal faithful CacheBlend-style implementation: full-depth per-chunk KV, global RoPE reindexing, and selective token recomputation. It is training-free and therefore has no CoMem LoRA. RULER and LoCoMo records show `resume_j=12` only because they reuse the host evaluator configuration; CacheBlend itself executes/caches all 36 layers. BABILong records show the driver's default `resume_j=6`, likewise not a method parameter.

The full 1,733-file remote raw tree is not duplicated into the paper directory; this snapshot preserves the aggregate, correctness gate, representative raw configs, hashes, and exact remote provenance.
