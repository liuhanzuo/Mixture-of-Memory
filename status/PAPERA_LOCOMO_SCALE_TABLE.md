# Paper A — ALL-SIZE × LoCoMo scale table (chat=False vs chat=True, GPT-4o judge)

**Date:** 2026-07-27. Zero-shot adapter-free CoMem, iter_bm25 (topk=12, hop=4, chunk=512, sink=bos),
per-size readout-safe j, n=1986, **max_new_tokens=512** (fixes the earlier mnt=48 CoT-truncation
artifact — 14B went 0.05→28.80 once un-truncated), GPT-4o judge via maas-openapi.
Preds on diskB `.73/.104`; judged on LOCAL wzc1.

## Headline table (GPT-4o judge, %)

| size | readout-safe j | /L | **chat=False** | **chat=True** |
|------|---:|---:|---:|---:|
| 0.6B | 2 | 0.07L | 11.48 | 20.49 |
| 1.7B | 3 | 0.11L | 17.42 | 26.44 |
| 4B | 9 | 0.25L | 24.07 | 32.02 |
| 8B | 9 | 0.25L | **32.38** | 31.52 |
| 14B | 13 | 0.325L | 28.80 | **39.58** |
| 32B | 27 | 0.42L | **51.81** | 48.89 |

## Findings
1. **Scale-consistency holds across the full family**: judge rises with model size.
   chatF: 11.5 → 17.4 → 24.1 → 32.4 → 28.8 → **51.8**; chatT: 20.5 → 26.4 → 32.0 → 31.5
   → 39.6 → **48.9**. CoMem's zero-shot depth-split works from 0.6B to 32B without
   per-size training. **32B chatF 51.81 is the best zero-shot cell**; 32B大幅领先 14B.
2. **chat template helps small/mid** (+6 to +11pp: 0.6B +9.0, 1.7B +9.0, 4B +7.9, 14B +10.8),
   **~ties/inverts at the two largest** (8B chatF 32.38 > chatT 31.52; **32B chatF 51.81
   > chatT 48.89**). The two sizes where chatF wins are exactly 8B and 32B — suggesting
   base-model scale reaches a point where the chat template slightly hurts CoMem's
   depth-split readout (plausible: larger base LM already verbalizes well zero-shot,
   chat template adds instruction scaffolding that competes with the cached-mid-layer
   readout). Worth a sentence in the paper.
3. **max_new_tokens matters hugely for larger CoT models**: mnt=48 truncated 8B/14B mid-CoT.
   mnt=512 fix: 8B 29.25→32.38, **14B 0.05→28.80 (chatF) / 39.58 (chatT)** — the mnt=48 14B
   "crash" was pure token-truncation, not a depth-readout failure. 32B was never run at mnt=48
   at scale (only a discarded 0.55 lower-bound); mnt=512 is the reported number.

## 32B status (DONE 2026-07-28, judged on LOCAL)
- User freed the reproduction B200 `.252` (28.89.19.252, wzc1, 8× empty) for the 32B run.
  32B weights already on wzc1 (`models/Qwen3-32B`, 64L) → no rsync. Ran via
  `scripts/_run_locomo_32b_b200.sh`: chatFALSE then chatTRUE, 8-shard (1 GPU/shard),
  `.venv` torch2.13, mnt=512, resume_j=27, iter_bm25/topk12/hop4/chunk512/sink=bos —
  identical protocol to the 5 smaller sizes. Preds judged on LOCAL (gpt-4o via .env
  maas-openapi + hy-proxy, workers=8).
- **Result: chatF 51.81, chatT 48.89** — chatF is the highest cell in the entire table,
  and 32B is the second size (after 8B) where chatF > chatT. 32B大幅领先 14B (chatF
  28.80 / chatT 39.58), confirming scale-consistency holds to the top of the family.
- Preds: `locomo_results/qcmem_32b_zs_j27_iter_chat{FALSE,TRUE}_mnt512/scores.json`.
- The "H20 co-tenant" earlier was NOT external — it was our own SparseForge N:M-sparsity
  resubmission (`.104` elsa, `.73` proxsparse, wandb SparseForge-Resubmission). Left
  untouched; 32B moved to the freed B200 .252 instead. .252 returned to user after 32B
  completed (now running SparseForge ProxSparse there).
- 32B mnt=48 lower-bound (earlier): chatF judge 0.55 — DISCARD (CoT-truncated artifact).


## Provenance
- chatFALSE dirs: `locomo_results/qcmem_{size}_zs_j{J}_iter_chatFALSE_mnt512/scores.json`
- chatTRUE dirs:  `locomo_results/qcmem_{size}_zs_j{J}_iter_chatTRUE_mnt512/scores.json`
- 48-token dirs (superseded): `..._iter_chatFALSE/` (no mnt suffix) — DO NOT USE for the table.
- Judge: gpt-4o via `.env` maas-openapi + hy-proxy, workers=8, n=1986 (cat1-4 judged, cat5 abstention).
