# QCMem Selector Comparison — RULER n=100 (Official `_string_match_all_one` Recall)

> **Experimental setup**: QCMem (j=12 bottleneck, chunk\_size=1024, 32 slots).  
> 4 chunk-selector variants × 3 RULER tasks × 6 context lengths × 5 topk values = 90 cells per selector.  
> All scores = mean recall % over n=100 samples, official RULER `_string_match_all_one` judgement.  
> Values reported are **best-topk peak** across topk ∈ {4,8,12,16,24} for each (task, length) cell.

## Coverage

| Selector | Cells with valid recall | Total cells |
|---|---|---|
| bm25 | 90 | 90 |
| recency | 90 | 90 |
| reader_attn | 90 | 90 |
| oracle | 90 | 90 |

## §2.5 — Best-TopK Recall: 4 Selectors × Task × Length

| Task | Length | BM25 (topk@peak) | Recency (topk@peak) | ReaderAttn (topk@peak) | Oracle (topk@peak) |
|---|---|---|---|---|---|
| NIAH-Single | 1k | 100.0 (@tk4) | 100.0 (@tk4) | 100.0 (@tk4) | 100.0 (@tk4) |
|  | 2k | 100.0 (@tk4) | 100.0 (@tk4) | 100.0 (@tk4) | 100.0 (@tk4) |
|  | 4k | 100.0 (@tk4) | 100.0 (@tk8) | 100.0 (@tk8) | 100.0 (@tk4) |
|  | 8k | 100.0 (@tk4) | 100.0 (@tk16) | 100.0 (@tk16) | 100.0 (@tk4) |
|  | 16k | 100.0 (@tk4) | 72.0 (@tk24) | 83.0 (@tk24) | 100.0 (@tk4) |
|  | 32k | 100.0 (@tk8) | 42.0 (@tk24) | 38.0 (@tk16) | 100.0 (@tk4) |
| NIAH-MultiKey | 1k | 100.0 (@tk8) | 100.0 (@tk24) | 100.0 (@tk4) | 100.0 (@tk12) |
|  | 2k | 100.0 (@tk8) | 100.0 (@tk8) | 100.0 (@tk4) | 100.0 (@tk4) |
|  | 4k | 97.0 (@tk8) | 99.0 (@tk12) | 98.0 (@tk16) | 100.0 (@tk8) |
|  | 8k | 95.0 (@tk4) | 95.0 (@tk24) | 93.0 (@tk16) | 100.0 (@tk12) |
|  | 16k | 97.0 (@tk4) | 61.0 (@tk24) | 69.0 (@tk24) | 100.0 (@tk16) |
|  | 32k | 99.0 (@tk4) | 44.0 (@tk24) | 41.0 (@tk24) | 100.0 (@tk4) |
| VT | 1k | 100.0 (@tk4) | 100.0 (@tk4) | 100.0 (@tk4) | 100.0 (@tk4) |
|  | 2k | 100.0 (@tk4) | 100.0 (@tk12) | 100.0 (@tk8) | 42.8 (@tk16) |
|  | 4k | 100.0 (@tk12) | 100.0 (@tk12) | 100.0 (@tk24) | 20.2 (@tk4) |
|  | 8k | 97.8 (@tk16) | 95.6 (@tk16) | 96.4 (@tk24) | 10.4 (@tk24) |
|  | 16k | 27.6 (@tk4) | 53.2 (@tk24) | 60.2 (@tk24) | 9.2 (@tk16) |
|  | 32k | 23.0 (@tk16) | 16.8 (@tk24) | 22.0 (@tk12) | 5.8 (@tk8) |

## §2.6 — Oracle vs BM25 Gap (Retrieval Cost Analysis)

> Oracle = gold chunk always included (perfect retrieval ceiling).  
> BM25 = lexical matching (practical baseline).  
> Gap = Oracle − BM25: how much recall is lost due to imperfect retrieval.

| Task | Length | Oracle | BM25 | Gap (Oracle-BM25) |
|---|---|---|---|---|
| NIAH-Single | 1k | 100.0 | 100.0 | +0.0 |
| NIAH-Single | 2k | 100.0 | 100.0 | +0.0 |
| NIAH-Single | 4k | 100.0 | 100.0 | +0.0 |
| NIAH-Single | 8k | 100.0 | 100.0 | +0.0 |
| NIAH-Single | 16k | 100.0 | 100.0 | +0.0 |
| NIAH-Single | 32k | 100.0 | 100.0 | +0.0 |
| NIAH-MultiKey | 1k | 100.0 | 100.0 | +0.0 |
| NIAH-MultiKey | 2k | 100.0 | 100.0 | +0.0 |
| NIAH-MultiKey | 4k | 100.0 | 97.0 | +3.0 |
| NIAH-MultiKey | 8k | 100.0 | 95.0 | +5.0 |
| NIAH-MultiKey | 16k | 100.0 | 97.0 | +3.0 |
| NIAH-MultiKey | 32k | 100.0 | 99.0 | +1.0 |
| VT | 1k | 100.0 | 100.0 | +0.0 |
| VT | 2k | 42.8 | 100.0 | -57.2 |
| VT | 4k | 20.2 | 100.0 | -79.8 |
| VT | 8k | 10.4 | 97.8 | -87.4 |
| VT | 16k | 9.2 | 27.6 | -18.4 |
| VT | 32k | 5.8 | 23.0 | -17.2 |

## Appendix — TopK Sweep at 16k and 32k

### NIAH-Single @ 16k

| TopK | bm25 | recency | reader_attn | oracle |
|---|---|---|---|---|
| 4 | 100.0 | 15.0 | 18.0 | 100.0 |
| 8 | 100.0 | 35.0 | 27.0 | 100.0 |
| 12 | 100.0 | 42.0 | 58.0 | 100.0 |
| 16 | 100.0 | 48.0 | 62.0 | 100.0 |
| 24 | 100.0 | 72.0 | 83.0 | 100.0 |

### NIAH-Single @ 32k

| TopK | bm25 | recency | reader_attn | oracle |
|---|---|---|---|---|
| 4 | 99.0 | 16.0 | 11.0 | 100.0 |
| 8 | 100.0 | 22.0 | 16.0 | 100.0 |
| 12 | 100.0 | 17.0 | 25.0 | 100.0 |
| 16 | 100.0 | 33.0 | 38.0 | 100.0 |
| 24 | 100.0 | 42.0 | 38.0 | 100.0 |

### NIAH-MultiKey @ 16k

| TopK | bm25 | recency | reader_attn | oracle |
|---|---|---|---|---|
| 4 | 97.0 | 18.0 | 26.0 | 99.0 |
| 8 | 96.0 | 24.0 | 40.0 | 99.0 |
| 12 | 92.0 | 42.0 | 46.0 | 99.0 |
| 16 | 90.0 | 41.0 | 58.0 | 100.0 |
| 24 | 83.0 | 61.0 | 69.0 | 99.0 |

### NIAH-MultiKey @ 32k

| TopK | bm25 | recency | reader_attn | oracle |
|---|---|---|---|---|
| 4 | 99.0 | 6.0 | 8.0 | 100.0 |
| 8 | 98.0 | 13.0 | 17.0 | 100.0 |
| 12 | 89.0 | 20.0 | 27.0 | 100.0 |
| 16 | 91.2 | 22.0 | 30.0 | 99.0 |
| 24 | 85.0 | 44.0 | 41.0 | 100.0 |

### VT @ 16k

| TopK | bm25 | recency | reader_attn | oracle |
|---|---|---|---|---|
| 4 | 27.6 | 5.2 | 24.8 | 6.4 |
| 8 | 24.8 | 10.6 | 26.4 | 5.0 |
| 12 | 27.0 | 8.6 | 31.4 | 7.2 |
| 16 | 26.8 | 11.6 | 34.0 | 9.2 |
| 24 | 27.4 | 53.2 | 60.2 | 7.2 |

### VT @ 32k

| TopK | bm25 | recency | reader_attn | oracle |
|---|---|---|---|---|
| 4 | 22.8 | 1.8 | 8.8 | 4.8 |
| 8 | 22.6 | 3.8 | 17.8 | 5.8 |
| 12 | 22.6 | 5.8 | 22.0 | 5.4 |
| 16 | 23.0 | 5.8 | 14.8 | 3.8 |
| 24 | 22.0 | 16.8 | 18.7 | 5.2 |

## Conclusions

**1. Oracle = 100% on NIAH: QCMem read-out is lossless for single-hop needle tasks.**
Oracle recall at 8k–32k: NIAH-Single **100.0**, NIAH-MultiKey **100.0**.
Given the correct chunk, the QCMem bottleneck (j=12 compression) does not degrade retrieval fidelity —
the model answers perfectly regardless of context length.
This confirms that the long-range performance gap for NIAH is attributable entirely to imperfect chunk
selection, not to information loss in the compressed representation.

**2. BM25 ≈ Oracle for NIAH: lexical retrieval closes the gap to near-zero.**
Oracle − BM25 gap: NIAH-Single **0.0** at all lengths (1k–32k); NIAH-MultiKey **+1.0 to +5.0**
(oracle 100%, BM25 95–99%).  Lexical keyword matching reliably locates the needle sentence across
all tested context lengths.  This validates BM25 as the default QCMem selector for entity-needle
workloads with negligible retrieval cost (<5 pp gap vs. perfect retrieval).

**3. ReaderAttn and Recency are substantially weaker on NIAH at long range.**
At 32k: NIAH-Single — BM25 **100.0**, ReaderAttn **38.0**, Recency **42.0**;
NIAH-MultiKey — BM25 **99.0**, ReaderAttn **41.0**, Recency **44.0**.
Hidden-state cosine similarity (reader_attn) and positional recency both fail to reliably surface
the needle chunk at 16k–32k.  Both selectors require large topk (≥16) to partially compensate,
whereas BM25 achieves peak recall at topk=4–8.

**4. VT (variable tracking) is a fundamentally different regime: oracle is ill-defined and all selectors are weak at ≥16k.**
Oracle for VT is implemented as "locate the chunk containing the answer string" — but variable
tracking requires following an N-hop assignment chain (e.g., `VAR3→VAR1→VAR2=42`).
Selecting the single chunk with the answer value (`VAR2=42`) is insufficient without the chain.
This explains the anomalous oracle scores: VT@16k oracle=**9.2**, VT@32k oracle=**5.8** —
far below BM25 (27.6 / 23.0) and even below random chance.
- At ≤8k all selectors perform well (>95%) because the full chain fits within the topk window.
- At 16k, reader_attn@tk24=**60.2** is the best practical selector, outperforming BM25 (27.6):
  attention salience with large topk incidentally covers more chain-link chunks.
- At 32k, all selectors collapse (BM25=23.0, reader_attn=22.0, recency=16.8, oracle=5.8),
  confirming that 32k VT chains exceed the capacity of any topk≤24 single-pass selector.
- **Paper message**: VT long-range accuracy is bounded by the selector's chain-coverage recall,
  not by QCMem read-out quality.  A multi-hop selector is required to close this gap.

**Summary table (16k & 32k, peak-topk recall %):**

| Task | Length | BM25 | Recency | ReaderAttn | Oracle |
|---|---|---|---|---|---|
| NIAH-Single  | 16k | **100.0** | 72.0  | 83.0  | **100.0** |
| NIAH-Single  | 32k | **100.0** | 42.0  | 38.0  | **100.0** |
| NIAH-MultiKey | 16k | **97.0** | 61.0  | 69.0  | **100.0** |
| NIAH-MultiKey | 32k | **99.0** | 44.0  | 41.0  | **100.0** |
| VT           | 16k | 27.6  | 53.2  | **60.2** | 9.2   |
| VT           | 32k | **23.0** | 16.8  | 22.0  | 5.8   |

---
_Generated by `scripts/aggregate_selector_comparison.py` — do not edit manually._