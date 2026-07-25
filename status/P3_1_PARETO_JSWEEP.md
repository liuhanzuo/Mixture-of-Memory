# P3.1 — Same-hardware depth-sweep quality/efficiency Pareto (j=0/6/9/12)

**Reviewer ask (P3.1):** on the SAME hardware, report prefill / peak mem / retrieval
time / total latency across split depths, especially j=0 full-recompute RAG vs
j=12 CoMem under the *same* top-12 pack — to prove "caching mid-layer state is
more cost-effective than ordinary retrieval + full recompute."

**Run:** `scripts/bench_qcmem_vs_fullctx.py` (peak-mem device-pin fixed), driver
`scripts/run_pareto_jsweep.sh`. Model `Qwen--Qwen3-8b` (36 layers), chunk_size=512,
topk=12, n_repeat=3, warmup=1, n_decode=20, **no LoRA** (pure timing), single H20
(.104 GPU5, `/usr/bin/python3.11`+py3.11 site-packages). Median-of-3. Same
retrieved pack (~6657 tokens) at every depth; only the split j varies.
Raw: `ruler_results/pareto_jsweep/bench_j{0,6,9,12}.json` (2026-07-26).

## Depth sweep (QCMem), per length 8k/16k/32k/64k/128k

Read recomputes layers[j:36]; write does embed+layers[0:j] over ALL chunks (offline).

| j | read layers | read_s (8k→128k) | decode_s (20 steps) | write_s (8k→128k, OFFLINE) | peak GB |
|--:|--:|---|---|---|---|
| **0** | 36 | 1.03/1.09/1.01/1.05/1.01 | ~20.1 | 0.03/0.05/0.06/0.06/0.23 | 17.4→18.3 |
| 6 | 30 | 0.95/0.87/0.88/0.88/0.86 | ~17.4 | 0.27/0.44/1.21/2.13/4.61 | 17.4→18.3 |
| 9 | 27 | 0.79/0.80/0.79/0.80/0.79 | ~15.9 | 0.32/0.73/1.43/2.96/6.11 | 17.4→18.3 |
| **12** | 24 | 0.78/0.73/0.71/0.72/0.72 | ~14.4 | 0.49/0.92/2.06/3.98/7.79 | 17.4→18.3 |

**decode_s/token** = decode_s/20: j0=1.005, j6=0.865, j9=0.795, **j12=0.722** — tracks read_s
(each decode step ≈ one read of the pack).

## Full-context reference (j-independent, same GPU)
| len | prefill_s | decode_s | peak GB |
|--:|--:|--:|--:|
| 8k | 1.28 | 1.70 | 19.9 |
| 16k | 2.98 | 1.77 | 24.6 |
| 32k | 7.64 | 1.07 | 33.8 |
| 64k | 22.27 | 1.19 | 52.3 |
| 128k | **OOM** | — | — |
(matches tab_eff full-mem 20/25/34/52 GB; 128k OOM on shared GPU5 vs tab_eff's clean-GPU 89 GB — full-context is the point either way.)

## Findings (the P3.1 headline)
1. **read_s is L-INDEPENDENT at every depth** (flat across 8k→128k) — the fixed-pack
   read is the core efficiency property; full-context prefill is O(L²) (1.3→22.3s, OOM@128k).
2. **Query-time cost falls monotonically with deeper j** (fewer recompute layers [j:36]):
   read 1.03→0.72s (−30%), decode 20.1→14.4s (−28%) from j0→j12.
3. **Offline write rises O(L)×j** (embed+[0:j] over all N chunks): at 128k, 0.23→7.8s
   from j0→j12 — but this is a ONE-TIME ingest, amortized over all future queries.
4. **Peak mem is flat ~17.4→18.3 GB regardless of j AND L** (context-independent);
   identical across all four depths (the pack is the same).
5. **The decisive comparison — j=0 (ordinary full-recompute RAG) vs j=12 (CoMem),
   same retrieval, same pack, same peak mem:** j=12 serves each query with 24 of 36
   layers recomputed → **read −29%, decode −28% per query**, at the cost of a
   one-time offline write (7.8s @128k for all chunks). **Caching the mid-layer
   state is strictly cheaper at query time than full-recompute RAG**, and the extra
   ingest amortizes across queries → "缓存中层状态确实比普通 retrieval + full recompute 更划算" ✔.

Quality at equal retrieval: the depth-sweep accuracy (tasks #71 j=0 iter_bm25 control,
#73 j={6,12} frozen depth sweep) shows j=12+distilled-LoRA is comparable to j=0 →
the query-time efficiency is effectively free after amortization.

## Chunk-size decode refresh (fresh sdpa harness, 2026-07-26) — reconciles tab_chunk

**Why:** the draft `tab_chunk.tex` decode column (0.69/1.17/2.39/5.55 s for chunk
128/256/512/1024) came from an older, uniformly ~3.4× slower draft harness. It
CONTRADICTED tab_pareto's fresh j=12/chunk512 decode (0.72 s/tok) for the *identical*
config. Re-measured chunk={128,256,1024} at j=12 with the SAME fresh sdpa harness
(`bench_qcmem_vs_fullctx.py`, .104 GPU5, median-of-3, 64k+128k, n_decode=20).
Raw: `ruler_results/pareto_jsweep/chunk_{128,256,1024}.json`.

| chunk | read_len | decode s/tok (fresh) | old draft | ratio | peak GB (64k/128k) |
|--:|--:|--:|--:|--:|--:|
| 128  | 1{,}665  | 0.19 (3.75s/20) | 0.69 | 3.6× | 16.3/16.8 |
| 256  | 3{,}329  | 0.37 (7.37s/20) | 1.17 | 3.2× | 16.8/17.3 |
| **512**  | 6{,}657  | **0.72** (14.5s/20) | 2.39 | 3.3× | 17.4/18.3 |
| 1024 | 13{,}313 | 1.60 (32.0s/20) | 5.55 | 3.5× | 19.8/20.3 |

- Fresh decode scales ~linearly with read_len (1665:3329:6657:13313 ≈ 1:2:4:8 →
  0.19:0.37:0.72:1.60 ≈ 1:1.9:3.8:8.4), as expected (each faithful step recomputes
  layers[12:36] over the whole pack, no KV-cache reuse).
- **Full-context is CACHED incremental decode** (uses `past_key_values`): per-step
  ≈ 0.05 s (8k 0.061 / 16k 0.050 / 64k 0.045) — so the honest "decode is slow"
  limitation is QCMem-faithful 0.72 vs full-cached 0.05 (~14×), NOT the old 6× (2.4 vs 0.4).
- **Prefill-speedup + multikey columns of tab_chunk LEFT UNCHANGED** — they are a
  separate clean-GPU measurement (128k full-ctx OOMs on shared GPU5), cross-consistent
  with tab_eff (7.83× @128k). Only the harness-sensitive decode column was refreshed.
- Paper updated: `tab_chunk.tex` decode col → 0.19/0.37/0.72/1.60; `05_experiments.tex`
  Efficiency+chunk prose 2.4→0.72; `07_limitations.tex` 2.4/5.5→0.72/1.6 and full-attn
  0.3–0.5→0.05 (cached). Now all decode numbers match tab_pareto's 0.72 scale.
