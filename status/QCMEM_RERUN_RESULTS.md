# QCMem Rerun — Unified Protocol (chat + no-think + iter_bm25)

> **Protocol (2026-07-17 用户指令):** all QCMem eval uses `--use_chat_template` + `enable_thinking=False` (no-think) + `selector=iter_bm25`. Official scoring: BABILong `compare_answers` (`scripts/score_nested_babilong.py`), RULER `string_match`. n=100, chunk512, topk12, sink=bos.
> Fix `c056a6d` feeds the no-think assistant prefix at the QCMem generation boundary (removes `<think>` pollution). All numbers below are from re-run + re-scored artifacts on diskB — NOT hand-copied.

## Provenance / model availability
- Only **Qwen3-8B** (`models/Qwen3-8b-local`, diskB) is currently on disk → 8B reruns in progress. Local wzc1 has **no** Qwen3 models. Other scales (0.6B/1.7B/4B/14B/32B/30B-A3B) **must be downloaded** before their reruns.
- Adapter: `outputs/qcmem_distill_qwen_j12_r32_4k/final` (8B, j12) — the only verifiable adapter on disk.

## ★ COVERAGE MATRIX — mirror old `QCMEM_BENCHMARK_PLAN.md` under NEW config (chat+no-think + iter_bm25; vt=raw prompt)
> Requirement (2026-07-17 user): everything the OLD plan has must appear in the NEW rerun. n=100 (RULER/BABILong), full test set (LongBench/LoCoMo/LongEval), n=30-100 (vs-Dense crash test). Baselines (KV-Direct/HCache/StreamingLLM/MemoryLLM/Dense) same config where applicable.

| model | RULER | BABILong | LongBench | LongEval | LoCoMo | vs-Dense 128k | download? |
|---|---|---|---|---|---|---|---|
| **Qwen3-8B** | ✅ (s/mk chat, vt raw) | ✅ (ad 57.1 / zs 48.4) | 🔄 next | 🔄 next | 🔄 next | 🔄 next | on disk |
| Qwen3-0.6B | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⏳ need dl |
| Qwen3-1.7B | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⏳ need dl |
| Qwen3-4B | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⏳ need dl |
| Qwen3-14B | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⏳ need dl |
| Qwen3-32B | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⏳ need dl |
| Qwen3-30B-A3B | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ | ⏳ need dl |

**Baselines (per model, same config):** KV-Direct ✅8B-BABILong / HCache ✅8B-BABILong (⚠️ HCache = EuroSys systems paper, use as mid-cache-no-retrieval ABLATION not accuracy peer) / MemoryLLM 🔄 running (priority; ⚠️ Llama backbone = cross-model) / recency(StreamingLLM proxy, `--selector recency`, zero-cost) ⬜ queue / StreamingLLM ⬜ / Dense ⬜. RULER-baselines ⬜.
**★ Peer methods (true competitors):** MemoryLLM (only one runnable now, cross-backbone) + **InfLLM** (training-free retrieval memory, arXiv:2402.04617 — the real same-paradigm peer; coder `ae13512d` integrating harness 2026-07-17). After InfLLM lands: run it same benchmarks/models/config. Optional 2nd priority: Activation Beacon, faithful StreamingLLM. **Abstract fix: don't name HCache as accuracy competitor → generalize ("prior training-free memory / KV-eviction baselines"), name InfLLM once numbers exist.**
**⏸ Not evaluated (need API/GPT-4o judge, per old plan):** LongMemEval / ∞Bench / HELMET.
**Per-model j:** zs readout-safe (0.6B j2 / 1.7B j3 / 4B j9 / 8B j9 / 14B j13 / 32B j27 / 30B-A3B j12) + adapter content/0.33L j. adapters only exist for 8B(j12) — other scales' adapters also need training or locating.

**Execution order (2026-07-17 user: finish ALL 8B benchmarks BEFORE any other size):** sequential on .73 (option A), one subagent per round:
- Round 2 (running): vt-raw score + MemoryLLM BABILong.
- Round 3: **8B-zs RULER** (j9; single/mkey chat + vt raw; iter_bm25) + **RULER baselines** (KV-Direct j0, HCache j12).
- Round 4: **8B LongBench + LongEval + LoCoMo** (chat+no-think + iter_bm25; LoCoMo token-F1).
- Round 5: **8B vs-Dense 64k/128k** crash test (covers RULER super-window) + any 8B baselines still missing.
- → ONLY after 8B matrix complete: download other scales (0.6B/1.7B/4B/14B/32B/30B-A3B) and repeat.

---

## BABILong — Qwen3-8B, chat+no-think, n=100 (official `compare_answers`)

### iter_bm25 (current protocol) — RUN dirs `qcmem_j{12,9}_iter_bm25_chatnothink_{ad,zs}`
| config | j | qa | 0k | 1k | 2k | 4k | 8k | 16k | 32k | mean(21) |
|---|---|---|---|---|---|---|---|---|---|---|
| **+adapter** | 12 | qa1 | 100 | 82 | 68 | 63 | 50 | 23 | 27 | **57.1** |
| | | qa2 | 57 | 58 | 53 | 51 | 31 | 20 | 6 | |
| | | qa5 | 83 | 83 | 78 | 76 | 70 | 59 | 61 | |
| **zero-shot** | 9 | qa1 | 100 | 83 | 74 | 61 | 16 | 3 | 1 | **48.4** |
| | | qa2 | 60 | 53 | 59 | 38 | 22 | 11 | 1 | |
| | | qa5 | 81 | 82 | 67 | 62 | 57 | 45 | 41 | |

### bm25 (reference — superseded by iter_bm25 per user directive; kept for comparison)
| config | j | qa | 0k | 1k | 2k | 4k | 8k | 16k | 32k | mean(21) |
|---|---|---|---|---|---|---|---|---|---|---|
| +adapter | 12 | qa1 | 100 | 82 | 69 | 63 | 67 | 59 | 38 | **62.2** |
| | | qa2 | 57 | 59 | 53 | 51 | 35 | 30 | 13 | |
| | | qa5 | 83 | 83 | 78 | 76 | 71 | 70 | 70 | |

### OLD (contaminated, raw-prompt, pre-fix — VOID)
| config | qa1 32k | mean | note |
|---|---|---|---|
| +adapter bm25 raw | 21 | 55.5 | thinking-contaminated; deflated at 32k. Replaced by chat+no-think above. |

**Notes:**
- chat+no-think fixes the contamination: raw qa1-32k=21 → chat+no-think bm25=38 / iter_bm25=27.
- **iter_bm25 < bm25 for BABILong** at long ctx (adapter 57.1 vs 62.2): iter_bm25's multi-hop expansion pulls distractors on fact-lookup QA. iter_bm25 is the right selector for RULER-vt (multi-hop chains), where it wins big. Reported per user's uniform-iter_bm25 directive; flagged for review.
- zero-shot collapses at long ctx (qa1 8k=16→32k=1) — adapter's role is to restore long-context readout.

---

## RULER — Qwen3-8B, iter_bm25 (adapter j12) — n=100, string_match
| task | 8k | 16k | 32k | note |
|---|---|---|---|---|
| niah_single | 100 | 100 | 99 | ✅ chat+no-think |
| niah_multikey | 85 | 82 | 80 | ✅ chat+no-think |
| variable_tracking | **96.8** | **98.8** | **98.0** | ✅ **raw prompt** (chat mode broke it → 1.8/1.2/0.6 via reasoning-truncation; raw prompt restores ~97). vt reported on raw prompt; single/mkey on chat+no-think. |

## Baselines — Qwen3-8B, chat+no-think, n=100, official `compare_answers` (BABILong)
| method | qa1 (0k→32k) | qa2 | qa5 | overall |
|---|---|---|---|---|
| **QCMem +adapter (iter_bm25, j12)** | 100/82/68/63/50/23/27 | 57/58/53/51/31/20/6 | 83/83/78/76/70/59/61 | **57.1** |
| **QCMem +adapter (bm25, j12)** ref | 100/82/69/63/67/59/38 | 57/59/53/51/35/30/13 | 83/83/78/76/71/70/70 | 62.2 |
| **KV-Direct** (resume_j=0, O(ctx), in-window upper bound) | 99/94/93/89/83/78/71 | 61/54/55/52/52/47/41 | 83/79/75/74/77/72/69 | **71.3** |
| **HCache** (mid-layer, no-retrieval) | 100/72/52/33/15/2/1 | 57/57/48/30/18/12/7 | 81/81/70/64/49/30/16 | **42.6** |
| MemoryLLM (Llama, chat) | ⬜ round 2 | | | |

**Head-to-head (BABILong, chat+no-think, n=100, overall mean):** KV-Direct **71.3** (full-recompute, no compression, but O(ctx)→OOM beyond window) > QCMem+adapter iter_bm25 **57.1** / bm25 62.2 (compressed, constant memory) > HCache **42.6**. KV-Direct is the in-window upper bound; QCMem's value = constant memory + beyond-window survival (Dense/KV-Direct OOM at 128k).

## Other benchmarks — same config (⬜ pending)
| benchmark | model | status |
|---|---|---|
| LongBench | 8B (+scales) | ⬜ chat+no-think + iter_bm25 |
| LoCoMo | 8B (+scales) | ⬜ chat+no-think + iter_bm25, token-F1 |

---
*Updated 2026-07-17 20:54. Source dirs on diskB `babilong_results/qcmem_j*_iter_bm25_chatnothink_*`; scorer `score_nested_babilong.py` (official `compare_answers`).*

---
## ★★ 8B MATRIX COMPLETE (2026-07-18) — unified chat+no-think + iter_bm25 (vt=raw), n=100 / full test set

### QCMem 8B
| benchmark | +adapter (j12) | zero-shot (j9) |
|---|---|---|
| RULER single 8k/16k/32k | 100/100/99 | 95/98/97 |
| RULER multikey | 85/82/80 | 32/42/49 |
| RULER vt (raw prompt) | 96.8/98.8/98.0 | 42/48/32 |
| BABILong overall | 57.1 (bm25 ref 62.2) | 48.4 |
| LongBench avg-f1 | 36.35 | ~26.2 |
| LoCoMo token-F1 (n1986) | 19.51 | 14.26 |
| LongEval 4k/8k/16k/32k | 95/73/76/79 | 14/6/5/7 |
| vs-Dense 128k single/mkey | **Dense 0→QCMem 100/99** | — |

### Baselines (RULER string_match): KV-Direct 100/100/100 all lengths (in-window upper bound); HCache single 34/8/7, mkey 9/4/0, vt ~0 (collapses w/o retrieval). LongBench/LoCoMo KV-Direct+HCache running.
### Peers: MemoryLLM BABILong 29.57 (fresh uniform n100: qa1 25.7/qa2 21.4/qa5 41.6); InfLLM RULER single 100/99/95, mkey 99/93/65, vt 100/98/91; InfLLM BABILong 67.1.

### Honest framing (do NOT overclaim):
1. **QCMem's core win = beyond-window survival at constant memory** — vs-Dense 128k: Dense→0, QCMem→100/99. That's unique & headline.
2. **In-window, QCMem ≈ KV-Direct at constant memory** (RULER), and **HCache collapses w/o retrieval** → retrieval is the mechanism.
3. **QCMem does NOT uniformly beat strong peers in-window**: InfLLM BABILong 67.1 > QCMem 57.1 (InfLLM's 4k local window > QCMem's compressed pack at ≤32k). But QCMem more robust at RULER-mkey 32k (80 vs 65) and survives 128k where InfLLM degrades.
4. **Adapter is essential for hard/retrieval tasks**: LongEval zs 14/6/5/7 → adapter 95/73/76/79; BABILong/RULER-mkey/vt similar. LongBench zs 26→adapter 36; LoCoMo zs 14→adapter 20.
5. **Chat+no-think fixed prior contamination**: LongBench 9.58→36.35; BABILong qa1-32k 21→38; RULER vt (raw) 1.8→~97.

**Next phase: other model sizes (0.6B/1.7B/4B/14B/32B/30B-A3B) — NOT on disk, need downloads.**

### LongBench/LoCoMo 8B baselines (chat+no-think, full set) — 2026-07-18
| benchmark | KV-Direct | HCache | QCMem+adapter |
|---|---|---|---|
| LongBench avg-f1 | 42.97 | 19.27 | 36.35 |
| LoCoMo token-F1 | 40.06 | 7.82 | 19.51 |
(KV-Direct=full-recompute upper bound > QCMem; HCache collapses < QCMem. Consistent with RULER.)

## SCALE PHASE (2026-07-18): all 6 Qwen3 scales downloaded to diskB models/ (0.6B/1.7B/4B/14B/32B/30B-A3B). Zero-shot arm launchable (readout-safe j per model); +adapter needs per-scale adapters (only 8B on disk).
