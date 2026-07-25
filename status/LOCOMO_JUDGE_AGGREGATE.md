# LoCoMo GPT-4o Judge — Aggregate (single source of truth for Paper A LoCoMo table)

**Compiled 2026-07-24.** All numbers are official `run_scoring` output read directly from
`locomo_results/*/scores.json` on **.73 (28.85.35.73, wzc1 mount)**. Protocol = **chat=False**
(mandated paper protocol — no chat template, models have no SFT/RL so chat template is unfair),
**GPT-4o judge** via funded `maas-openapi` JWT key, **n=1986** each. CoMem selector = `iter_bm25`.
cat1–4 scored by GPT-4o over the API; cat5 (adversarial "no answer") scored locally as abstention
(so its judge == f1 == em == acc by construction).

## 1. Main comparison — LoCoMo GPT-4o judge, chat=False, n=1986

Ranked by overall judge (the format-robust headline metric the paper reports for LoCoMo):

| # | Method (chat=False) | **overall judge** | overall F1 | overall acc | overall EM | source dir | scored |
|---|---------------------|:---:|:---:|:---:|:---:|---|---|
| 1 | **CoMem flagship (+distilled LoRA, j12, iter_bm25)** | **38.27** | 9.15 | 23.36 | 0.55 | `qcmem_8b_iter_chatFALSE` | 07-22 |
| 2 | KV-Direct (full-context recompute) | 34.59 | 9.02 | 22.36 | 0.60 | `kvdirect_8b_chatFALSE` | 07-22 |
| 3 | StreamingLLM | 25.63 | 7.67 | 13.75 | 1.56 | `streamingllm_8b_chatFALSE` | 07-24 (#64) |
| 4 | InfLLM | 22.21 | 7.39 | 13.34 | 1.71 | `infllm_8b_chatFALSE` | 07-24 (#64) |
| 5 | MemoryLLM | 16.11 | 5.91 | 9.52 | 0.10 | `memoryllm_chatFALSE` | 07-24 (#64) |
| 6 | HCache | 8.11 | 4.67 | 6.29 | 0.25 | `hcache_8b_chatFALSE` | 07-22 |
| — | **CoMem adapter-free (frozen, j9, iter_bm25)** | *PENDING (#65)* | — | — | — | `qcmem_8b_zeroshot_j9_chatFALSE` | running on .252 |

**Headline:** under the mandated chat=False + GPT-4o judge protocol the LoCoMo ranking is
**CoMem 38.27 > KV-Direct 34.59 > StreamingLLM 25.63 > InfLLM 22.21 > MemoryLLM 16.11 > HCache 8.11**.
CoMem (constant read cost) edges out KV-Direct (full recompute, no compression) while every fixed-budget
streaming/compression baseline trails by ≥9 judge points.

## 2. Per-category GPT-4o judge (chat=False, n=1986)

Categories: 1 = single-session, 2 = multi-hop temporal, 3 = open-domain, 4 = single-hop (largest, n=841),
5 = adversarial / "no answer" (local abstention).

| Method | cat1 (282) | cat2 (321) | cat3 (96) | cat4 (841) | cat5 adv (446) |
|--------|:---:|:---:|:---:|:---:|:---:|
| CoMem flagship | 26.95 | 19.00 | 30.21 | **69.32** | 2.47 |
| KV-Direct      | 24.11 | 18.69 | 25.00 | 62.19 | 2.69 |
| StreamingLLM   | 22.70 | 11.21 | 23.96 | 42.21 | 6.95 |
| InfLLM         | 18.44 | 14.33 | 25.00 | 33.89 | 7.62 |
| MemoryLLM      | 15.60 |  8.72 | 15.63 | 27.47 | 0.45 |
| HCache         |  6.74 |  2.49 |  9.38 | 14.27 | 1.12 |

Notes:
- **cat4 (single-hop, largest bucket) is where CoMem's lead is decisive** (69.32, +7 over KVD, +27 over StreamingLLM).
- **cat2 (multi-hop temporal) is the hardest for everyone.**
- **cat5 (adversarial abstention):** the aggressive-compression baselines abstain *more often*
  (StreamingLLM 6.95, InfLLM 7.62) than the full-information methods (CoMem 2.47, KVD 2.69) —
  dropping context makes them likelier to (correctly) say "no answer". MemoryLLM 0.45 = almost never abstains correctly.

## 3. ⚠️ KEY CORRECTION — "40.06" is F1, not judge (fixes earlier errata note)

An earlier note (PAPER_LOCOMO_ERRATA §12, since corrected) mis-stated "KV-Direct judge 40.06 >> CoMem 19.51".
**Those 40.06 / 19.51 figures are token-F1, NOT the judge metric.** On the actual GPT-4o judge the two
methods are near-tied under chat=True and CoMem *leads* under chat=False:

| Method | chat=True judge | chat=True F1 | chat=False judge | chat=False F1 |
|--------|:---:|:---:|:---:|:---:|
| CoMem flagship | 37.76 | 19.51 | **38.27** | 9.15 |
| KV-Direct      | 38.22 | **40.06** | 34.59 | 9.02 |

Sources: `qcmem_8b_iter_chatnothink` / `kvdirect_8b_chatnothink` (chat=True), `*_chatFALSE` (chat=False).

**Interpretation (the §9 formatting thesis, quantified):** KV-Direct reads the *full raw context*, so its
verbose extractive output inflates token-F1 (40.06) even though correctness matches CoMem — the GPT-4o
judge, which scores meaning not surface form, puts them within 0.5 pt at chat=True (37.76 vs 38.22).
Removing the chat template collapses *both* F1 scores to ~9 (the formatting artifact disappears) and, on
the format-robust judge, **CoMem chat=False (38.27) exceeds KV-Direct (34.59)**. → The paper must report
**GPT-4o judge as the headline LoCoMo metric**, not token-F1; where F1 is shown, pair it with chat state
and the formatting caveat.

## 4. Reference — chat=True (no-think) judge numbers on disk (superseded by chat=False)

For completeness; NOT for the main table (chat=True is the non-mandated protocol):

| Method | chat=True judge | dir |
|--------|:---:|---|
| CoMem flagship (iter_bm25) | 37.76 | `qcmem_8b_iter_chatnothink` |
| KV-Direct | 38.22 | `kvdirect_8b_chatnothink` |
| CoMem 8B iter (older chat=True, per-cat in errata §1: F1 19.51/acc 28.65) | 37.76 | same as above |

Older chat=True judge dirs also on wzc1-local `locomo_results/`: `qcmem_j12` 39.48, `qcmem_tk12` 36.46,
`qcmem_tk4b` 40.54 (n=1490, partial), `kvdirect_full` 33.48 (n=920), `kvdirect` 31.54 (n=780), `hcache` 12.29.
These are earlier config/partial-n runs, kept only for provenance.

## 5. Provenance / reproducibility

- Node: **.73 = 28.85.35.73**, port 36000, wzc1 path `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/locomo_results/`.
- Judge: GPT-4o via `maas-openapi.wanjiedata.com/api/v1` (funded JWT key in `.env`, reachable only via hy-proxy). `judge_model=gpt-4o` in every scores.json.
- The 3 baselines (StreamingLLM/InfLLM/MemoryLLM) judge scores were backfilled 2026-07-24 (task #64) with the new funded key; CoMem/KVD/HCache chat=False judged 2026-07-22.
- Re-score any dir: `python scripts/eval_qcmem_locomo.py --score_only --use_llm_judge --output_dir locomo_results/<dir>` (needs `.env` key + hy-proxy).
- **Pending:** CoMem adapter-free (j9, frozen, no LoRA) row — generations running on .252 (#65); its judge pass runs after `SCHED_DONE`, filling row 1b of the main table.
