# P0.19 — Retrieval-Recall vs In-Pack-Readout Decomposition (Paper A)

**Thesis being tested.** The CoMem (`j=12`, mid-depth resume) vs `j=0` (RAG /
full-depth recompute upper bound) accuracy gap has two possible causes:

- **(a) retrieval miss** — the flagship `iter_bm25` selector failed to place the
  gold support span into the top-12 pack; or
- **(b) cached-state readout failure** — the gold span WAS in the pack, but the
  mid-depth (`j=12`) resume could not read it out of the compressed cached state.

P0.19 separates these by labelling each sample HIT/MISS **independently of answer
accuracy**, then measuring `j=0` and `j=12` accuracy on the SAME recall-HIT
subset (and separately on the MISS subset). All arms use `selector=iter_bm25`,
`topk=12`, `chunk_size=512`, `chat_template=False`, Qwen3-8B (`models/Qwen3-8b-local`).

**Deliverable script:** `scripts/analyze_p019_recall_readout.py` — pure-CPU, zero
GPU, zero training. It only *imports* (never modifies) the unmodified flagship
selection / sample primitives and recomputes recall + joins with the already-run
per-sample correctness.

**Headline result (cross-task, computed here).** On the faithful single-support
cells (LongEval, BABILong qa1), retrieval is NOT the bottleneck — recall@12 is
77–100% and `j=0` (full recompute of the very same retrieved pack) answers
65–100% — yet `j=12` with the adapter **frozen** collapses to 0–12% on that SAME
recall-HIT subset. The CoMem vs RAG gap is therefore an **in-pack readout failure
(b)**, not a retrieval miss (a). LongEval further shows the readout is *learnable*:
the flagship `j=12`+LoRA recovers to 63–75% while frozen `j=12` is a hard 0%.

---

## Recall判定口径 (how a HIT is decided — NEVER from answer accuracy)

1. Regenerate the sample deterministically (same seed as the eval), tokenize
   with the local tokenizer (`add_special_tokens=True`, exactly as the eval).
2. `chunks = tokens.split(512)`; `context_chunks = chunks[:-1]`,
   `query_chunk = chunks[-1]` — bit-identical to `qcmem_generate`.
3. Run the flagship selector on the SAME inputs the eval used:
   `_iter_bm25_indices(context_chunks, bare_question_ids, topk=12, iter_rounds=0,
   iter_hop_topk=4)` → `sel_set`.
4. Locate the **gold support span's** document-absolute chunk index/indices via
   `harness._locate_needle_chunks` (BABILong / RULER-NIAH) or
   `eval_qcmem_longeval._oracle_needle_chunks` (LongEval).
5. **HIT** iff every gold-support chunk ∈ `sel_set` (single-support tasks reduce
   to "the gold chunk was selected"). We also report `mean_recall_frac` (fraction
   of gold chunks in-pack) so any-in-pack vs all-in-pack is visible.

Recall is thus decided purely by *"is the gold-support chunk in the selected
pack"* — completely orthogonal to whether the model answered correctly.

**Stats.** recall@12 + Wilson 95% CI; subset accuracies + Wilson CI; the paired
`j12 − j0` gap on the HIT subset via paired bootstrap (10k resamples) 95% CI.
Raw sample IDs of the HIT and MISS subsets are emitted per cell in the JSON
(`cells[...].raw_ids.{hit,miss}` and full `per_sample`).

---

## Located asset absolute paths (audit)

### BABILong (paired j0 / j12 frozen, LOCAL — pure-CPU-computable ✅)
Root: `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`
- j0  : `babilong_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE/`
- j12 : `babilong_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/`
- (j6 also present: `babilong_results/qcmem_8b_zeroshot_j6_frozen_iterbm25_chatFALSE/`)
- Layout: `<root>/<run>_<length>/qaN_<length>_..._shard{0..3}of4.csv` (+ `.json`
  with recorded `score`/`correct`). CSV columns `[target, output, question]`.
- Samples: fixed HF Arrow cache
  `.hf_cache/datasets/RMT-team___babilong/<length>/*/*/babilong-qaN.arrow`
  (columns `question,input,target`; **no supporting-fact annotation**).
- Config (both arms, verified): `selector=iter_bm25`, `topk=12`, `chunk_size=512`,
  `resume_j={0,12}`, `lora_adapter=null`, `chat_template=false`,
  `zero_training_no_adapter=true`, model `models/Qwen3-8b-local`.
- Shard→dataset-index map: CSV row `r` of `shardSofN` = dataset index `S + r*N`
  (eval walks `range(n)[shard::N]` in order). **Validated:** recomputing
  `babilong.metrics.compare_answers` per row reproduces the recorded per-shard
  `score`/`correct` bit-identically (qa1/8k: j0 60/100, j12 4/100).

### LongEval (paired j0 / j12-frozen / j12-LoRA-flagship — pure-CPU-computable ✅)
Node **.104** (`28.83.24.104:36000`, diskB `share_304376610`, **different FS from
local** — pulled read-only via rsync to `/tmp/p019_longeval/{j0,j12,flag}`):
- j0            : `longeval_results/p0_2_c2_j0_iterbm25_chatFALSE/longeval_8b/` (lengths 4k–128k)
- j12 frozen    : `longeval_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/` (lengths 8k–128k)
- j12 + LoRA    : `longeval_results/qcmem_8b_iter_chatFALSE/longeval_8b/`
  (flagship, LoRA `outputs/qcmem_distill_qwen_j12_r32_4k/final`, lengths 4k–128k)
- Per-sample records: `longeval_<length>_shard{0..7}of8.json` → `records[]` each
  with `{sample_index, label, expected, output, pred, correct, ...}`. Config in
  sibling `eval_config_shard*of8.json`.
- Config (all three arms, verified): `seed=1234`, `selector=iter_bm25`, `topk=12`,
  `chunk_size=512`, `iter_rounds=0`, `iter_hop_topk=4`, `use_chat_template=False`,
  model `models/Qwen3-8b-local`.
- Reproducibility: `length_seed = 1234 + zlib.crc32(length)%100000`; per-sample
  `rng = random.Random(length_seed*1000 + i)`. crc32 is **stable** across
  processes → all three arms share ONE sample set → trivially paired by
  `sample_index`. Single unambiguous gold needle line.

### RULER (NOT paired — needs a fresh PYTHONHASHSEED-pinned re-run ⚠️)
Node **.104**:
- j0  (niah) : `ruler_results/p0_2_c2_j0_iterbm25_niah_chatFALSE/`
- j12 frozen : `ruler_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE/`
- RULER JSONs store **no per-sample records** (CSV only, columns
  `[target, output, question, recall]` — where `recall` = `_string_match_all_one`
  = ANSWER match, **not** retrieval recall, so it must NOT be reused for P0.19).

---

## Per-task CPU-vs-GPU status

| Task | Paired? | Gold-support locator | Status |
|------|---------|----------------------|--------|
| BABILong qa1 | ✅ j0/j12 local | answer-location string → chunk (mostly 1 chunk; 88/100 @8k) | **pure-CPU, DONE** |
| BABILong qa2 | ✅ j0/j12 local | answer-location string (2-fact task; locator = final answer only) | **pure-CPU, DONE — but see caveat** |
| BABILong qa5 | ✅ j0/j12 local | answer-location string (3-arg give task) | **pure-CPU, DONE — locator UNFAITHFUL, see caveat** |
| LongEval | ✅ j0/j12/flagship (.104) | single gold needle line (`_oracle_needle_chunks`) | **pure-CPU, DONE** |
| RULER niah_multikey / VT | ❌ existing runs unpaired | needle sentence (`_oracle_needle_chunks`) | **needs GPU paired re-run** |

**qa5 (and to a lesser extent qa2) caveat.** `_locate_needle_chunks` locates the
gold **answer** string, which for qa1/qa3 coincides with the single supporting
fact's location. For qa5 (3-arg "who gave what to whom") and qa2 (2 supporting
facts) the gold-answer string is NOT a faithful support-span locator: the qa5
MISS-subset accuracy is ≈ the HIT-subset accuracy (e.g. 16k: HIT 54.7 vs MISS
46.8; 4k: HIT 55.2 vs MISS 54.8) — i.e. our HIT/MISS label barely correlates with
whether the model can answer, which means the locator is not actually finding the
*support* chunk for qa5. **qa1 is the clean, faithful cell** (single-fact, single
support chunk); qa2 is directional but the two-fact support is only partially
captured. If a faithful qa2/qa5 support-span annotation is needed, it must come
from the babiTasks generator's supporting-fact indices (not in the HF Arrow
cache) — that is a data-provenance task, flagged for MAIN, not fixable here.

---

## Results

### BABILong (n=100 per cell; selector=iter_bm25, topk=12, chunk=512, chat=False)
`j0|HIT` / `j12|HIT` = accuracy on the SAME recall-HIT subset; `gap` = paired
bootstrap `j12−j0` on HIT subset with 95% CI. Full JSON:
`paperA/p0_19_babilong_decomp.json`.

| cell | recall@12 | HIT n | j0\|HIT | j12\|HIT | paired j12−j0 (HIT) | MISS n | j0\|MISS | j12\|MISS |
|------|-----------|-------|---------|----------|---------------------|--------|----------|-----------|
| qa1 4k  | 77.0% | 77 | 72.7 | 11.7 | −61.0pp [−72.7,−49.4] | 23 | 73.9 | 26.1 |
| qa1 8k  | 83.0% | 83 | 65.1 |  4.8 | −60.2pp [−71.1,−49.4] | 17 | 35.3 |  0.0 |
| qa1 16k | 58.0% | 58 | 39.7 |  0.0 | −39.7pp [−51.7,−27.6] | 42 | 21.4 |  4.8 |
| qa1 32k | 55.0% | 55 | 38.2 |  1.8 | −36.4pp [−50.9,−21.8] | 45 | 31.1 |  2.2 |
| qa2 4k  | 75.0% | 75 | 45.3 | 13.3 | −32.0pp [−44.0,−20.0] | 25 | 40.0 | 28.0 |
| qa2 8k  | 86.0% | 86 | 38.4 |  1.2 | −37.2pp [−47.7,−26.7] | 14 | 21.4 |  0.0 |
| qa2 16k | 47.0% | 47 | 27.7 |  4.3 | −23.4pp [−38.3,−10.6] | 53 |  5.7 |  3.8 |
| qa2 32k | 21.0% | 21 | 38.1 |  0.0 | −38.1pp [−57.1,−19.0] | 79 |  5.1 |  1.3 |
| qa5 4k  | 58.0% | 58 | 55.2 | 70.7 | +15.5pp [−1.7,+32.8]  | 42 | 54.8 | 59.5 |
| qa5 8k  | 70.0% | 70 | 75.7 | 50.0 | −25.7pp [−40.0,−11.4] | 30 | 60.0 | 66.7 |
| qa5 16k | 53.0% | 53 | 54.7 | 54.7 |  +0.0pp [−17.0,+17.0] | 47 | 46.8 | 34.0 |
| qa5 32k | 48.0% | 48 | 68.8 | 47.9 | −20.8pp [−39.6,−2.1]  | 52 | 53.9 | 30.8 |

**Reading (qa1/qa2 — the faithful cells).** recall@12 is high at short lengths
(77–86% @4k/8k) and the gold chunk IS in the pack, yet `j=12` accuracy on that
recall-HIT subset collapses to 0–13% while `j=0` on the SAME samples is 38–73%.
The paired gap is large and its 95% CI is far from 0. **Conclusion: at 4k–8k the
CoMem vs RAG gap is overwhelmingly a READOUT failure (b), not a retrieval miss
(a)** — the support was retrieved into the pack, and the mid-depth resume still
could not surface it. recall@12 does fall at 16k–32k (58→21%), so a growing share
of the gap at long context is genuine retrieval miss (a), but even there the
in-pack readout on HITs stays near-zero. (qa5 numbers are shown for completeness
but the locator is unfaithful there — see caveat; do not draw readout conclusions
from qa5.)

### LongEval (n=100 per cell; selector=iter_bm25, topk=12, chunk=512, chat=False)
Arms: `j0` (RAG upper bound) / `j12` (CoMem, LoRA-**frozen**) / `flag` (CoMem
`j12` + LoRA flagship `outputs/qcmem_distill_qwen_j12_r32_4k/final`), lengths
8k–128k paired. Gold needle is single + unambiguous → locatable 100/100 every
cell. Full JSON: `paperA/p0_19_longeval_decomp.json`.

| cell | recall@12 | HIT n | j0\|HIT | j12-frozen\|HIT | j12+LoRA\|HIT | MISS n |
|------|-----------|-------|---------|-----------------|---------------|--------|
| 8k   | 100.0% | 100 | 100.0 | 0.0 | 69.0 | 0 |
| 16k  |  98.0% |  98 |  98.0 | 0.0 | 75.5 | 2 |
| 32k  |  99.0% |  99 |  99.0 | 0.0 | 63.6 | 1 |
| 64k  |  97.0% |  97 |  96.9 | 0.0 | 69.1 | 3 |
| 128k |  98.0% |  98 |  98.0 | 0.0 | 70.4 | 2 |

**Reading (the cleanest P0.19 cell).** Retrieval is essentially perfect at every
length (recall@12 = 97–100%, gold always locatable), and `j0` (full recompute of
the retrieved pack) answers ≈ 100% — so retrieval is emphatically NOT the
bottleneck. Yet **`j12` with the adapter frozen scores exactly 0%** on that same
recall-HIT subset: the support is sitting in the pack and the mid-depth resume
reads out nothing. Adding the flagship LoRA (`j12+LoRA`) recovers readout to
63–75%. This isolates the CoMem gap to cause **(b) in-pack readout**, and shows
that the readout is learnable — LoRA training, not better retrieval, is what
closes it. (The 0% frozen number also confirms the effect is a property of the
*resume* path, since the identical pack recomputed from depth 0 answers ~100%.)

### RULER
Recall side computed by the script (e.g. niah_multikey_1 @4k, 20-sample smoke:
recall@12 = 90%). The **accuracy join cannot be done from the existing runs** —
see the reproducibility bug below.

---

## ⚠️ Bug / reproducibility finding for MAIN (RULER cannot be paired as-is)

`scripts/eval_ruler_qcmem.py:517`:
```python
base_seed = args.seed + (hash((task, length)) % 100000)
```
uses Python's built-in `hash()`, which is **salted per process** unless
`PYTHONHASHSEED` is pinned. The existing j0 and j12 RULER runs on .104 both
recorded `runtime.pythonhashseed = null` (seed=42), so their `base_seed` — hence
the entire regenerated sample set (needle keys, positions, VT chains) — differs
run-to-run. Consequences:
1. The existing j0 and j12 RULER predictions are **NOT the same samples**, so they
   cannot be paired for P0.19 (verified earlier: same-slot needles differ).
2. More broadly, any RULER run whose 8 shards were launched as separate processes
   with `PYTHONHASHSEED` unset would have had **misaligned sample sets across its
   own shards**, silently corrupting that run's aggregate.

Per task scope I am **reporting, not fixing** this (no edits to shared modules /
`.tex` / TODOList / status ledgers). To close the RULER leg, MAIN should launch a
fresh **paired** re-run with `PYTHONHASHSEED` pinned identically for BOTH arms and
all shards, then re-run the pure-CPU join. Prepared command (no GPU run performed
by this task):

```bash
# ---- RULER paired re-run, PYTHONHASHSEED pinned (run on a GPU node, e.g. .104) ----
# Same two arms (j0 upper bound, j12 CoMem), identical selector/topk/chat, ONE seed set.
PROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
for J in 0 12; do
  NAME=p0_19_ruler_j${J}_iterbm25_niahmk_chatFALSE_HSEED0
  for S in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$S PYTHONHASHSEED=0 nohup $PROOT/.venv/bin/python \
      scripts/eval_ruler_qcmem.py \
      --model_path models/Qwen3-8b-local \
      --resume_j $J --selector iter_bm25 --topk 12 --chunk_size 512 \
      --iter_rounds 0 --iter_hop_topk 4 \
      --ruler_tasks niah_multikey_1 variable_tracking \
      --lengths 4k 8k 16k 32k 64k 128k \
      --limit 100 --num_shards 8 --shard_index $S --seed 42 \
      --output_name $NAME \
      > logs/${NAME}_shard${S}.out 2>&1 &
  done; wait
done
# Then the pure-CPU join (recall independently recomputed; accuracy from the new CSVs):
PYTHONHASHSEED=0 .venv/bin/python scripts/analyze_p019_recall_readout.py --task ruler \
  --model_path models/Qwen3-8b-local \
  --ruler_tasks niah_multikey_1 variable_tracking \
  --lengths 4k 8k 16k 32k 64k 128k --topk 12 --chunk_size 512 --limit 100 --seed 42 \
  --out paperA/p0_19_ruler_decomp.json
# NOTE: analyze_p019 currently emits the RULER RECALL side only (j0/j12 columns
# blank) because it does not yet ingest the new RULER CSVs. Once the paired CSVs
# exist, extend analyze_ruler() to join them by (task,length,i) exactly like the
# BABILong/LongEval paths (the sample regeneration + i indexing already match the
# eval, so it is a pure CSV-row join). Kept minimal here per task scope.
```
Both arms must use the **same** `PYTHONHASHSEED` value so `base_seed`, and thus the
regenerated needles, match sample-for-sample; the CPU analyzer must also be run
with that same value.

---

## How to reproduce the computed tables

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory

# BABILong (local, ~few min)
.venv/bin/python scripts/analyze_p019_recall_readout.py --task babilong \
  --model_path models/Qwen3-8b-local \
  --j0_dir  babilong_results/qcmem_8b_zeroshot_j0_iterbm25_chatFALSE \
  --j12_dir babilong_results/qcmem_8b_zeroshot_j12_frozen_iterbm25_chatFALSE \
  --babilong_tasks qa1 qa2 qa5 --lengths 4k 8k 16k 32k --topk 12 --chunk_size 512 \
  --out paperA/p0_19_babilong_decomp.json

# LongEval (needs the record JSONs from .104; pulled to /tmp/p019_longeval/*)
.venv/bin/python scripts/analyze_p019_recall_readout.py --task longeval \
  --model_path models/Qwen3-8b-local \
  --j0_dir /tmp/p019_longeval/j0 --j12_dir /tmp/p019_longeval/j12 \
  --flagship_dir /tmp/p019_longeval/flag \
  --lengths 8k 16k 32k 64k 128k --topk 12 --chunk_size 512 --seed 1234 \
  --out paperA/p0_19_longeval_decomp.json
```

## Open questions about the P0.19 spec (for MAIN)
1. **HIT criterion for multi-support tasks** — I used all-gold-chunks-in-pack
   (single-support tasks unaffected). `mean_recall_frac` is also emitted so
   any-in-pack can be recomputed. Confirm which the paper wants as the headline.
2. **qa2/qa5 support-span provenance** — the HF Arrow cache has no supporting-fact
   annotation, so the current gold-support locator = gold-answer string (faithful
   only for qa1/single-support). A faithful qa2/qa5 support annotation needs the
   babiTasks generator's supporting-fact line indices. Should P0.19 restrict to
   qa1 (+ LongEval + RULER-NIAH, all single-support) for the readout claim, and
   treat qa2/qa5 as descriptive only?
3. **RULER** — confirm the paired PYTHONHASHSEED re-run above is authorized (GPU),
   and whether both niah_multikey_1 and variable_tracking are in scope.
