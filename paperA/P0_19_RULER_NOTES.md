# P0.19 — RULER leg (retrieval-recall × in-pack-readout decomposition), PAIRED

Task #135 (Paper A P0.19). Closes the RULER leg that was deferred in
`P0_19_decomp_NOTES.md` (§"Bug / reproducibility finding for MAIN"). EVAL-ONLY,
no training. Author: automated P0.19 RULER pass (2026-08-03).

MAIN owns `.tex` / TODOList / `versions/` / `status/` ledgers; this file is a
scratch design + results record only.

## 1. The seed-pairing bug — status: ALREADY FIXED (commit `d1e1389`, in HEAD)

The deferral reason in `P0_19_decomp_NOTES.md` was:
`scripts/eval_ruler_qcmem.py` seeded the per-`(task,length)` sample RNG with
`args.seed + hash((task, length)) % 100000`. Python's built-in `hash()` is
per-process **salted** unless `PYTHONHASHSEED` is pinned, so the existing on-disk
j0/j12 RULER runs (which recorded `pythonhashseed=null`) drew a *different*
`base_seed` each process → the regenerated needles/positions differed sample-for-
sample between arms → **unpaired** (and even a single run's 8 shards were mutually
misaligned).

That line was already repaired before this task in commit `d1e1389`:
```python
# scripts/eval_ruler_qcmem.py:523  (and eval_ruler_mem_space.py:820)
base_seed = args.seed + (zlib.crc32(f"{task}\x00{length}".encode()) % 100000)
```
`zlib.crc32` is deterministic across processes (PYTHONHASHSEED-independent), so
shards and arms now share one sample set. This matches P1.9's `iter_ruler`
(`scripts/eval_p1_9_dense_rag.py:362`), which uses the identical crc32 convention
→ all RULER harnesses are now on one seeding scheme.

So the **root cause was already closed**. What was still missing to *close the
P0.19 RULER leg* was: (a) the eval never emitted a per-example fingerprint, so
there was no fail-closed proof the arms were byte-identically paired; and (b) the
CPU analyzer's `analyze_ruler()` computed the recall side only, leaving the
j0/j12/flag accuracy columns blank. This pass adds both.

## 2. Byte-identity is achievable (verified) — so we do NOT stop at a blocker

`_make_niah` / `_make_vt` draw exclusively from the passed `random.Random(rng)`
(keys, values, needle positions, haystack sampling) — no `hash()`, no
set/dict-content ordering that leaks into the output text. The essay haystack is
a fixed local read (`data/pg19_train.jsonl`, first 8 MB, cached). Local CPU check
(regenerate `niah_multikey_1 @ {8k,16k}`, i∈{0,1,2}, twice each) → identical
`input_ids_sha256` and token counts every time. With `PYTHONHASHSEED=0` pinned in
every arm + the crc32 seed, the three arms produce byte-identical prompts per
index. The runtime sha gate (below) proves it per-example; on mismatch it aborts.

## 3. What changed (this pass)

* **`scripts/eval_ruler_qcmem.py`** — additive per-sample provenance:
  * `_sha256_ids(input_ids)` = `sha256(",".join(map(str, ids)))` over the FULL
    prompt token ids (same convention as P0.20 Phase B / P1.9 `_sha256_str`).
  * each processed sample appends `{sample_index, input_ids_sha256, target,
    output, recall, correct}` (`correct` = RULER string_match_all recall == 1.0);
  * per cell writes `<task>_<length>_shard{S}of{N}.records.json` alongside the
    CSV (config: selector/resume_j/topk/lora/seed/pythonhashseed + records).
  * NO existing gate loosened; the CSV writer / self-test / scoring untouched.

* **`scripts/analyze_p019_recall_readout.py`** — `analyze_ruler()` rewritten to
  JOIN accuracy (was recall-only):
  * loads the paired records.json from `--j0_dir/--j12_dir/--flagship_dir`;
  * regenerates each sample with the identical crc32 seed → recomputes recall
    (gold-support chunk ∈ iter_bm25 top-k, answer-independent) with the SAME
    selector the run used (`--ruler_selector iter_bm25`; verified against the
    selector recorded in each arm's records.json — fail-close on disagreement);
  * **FAIL-CLOSED seed-pairing gate**: for every paired index, asserts every
    present arm's `input_ids_sha256` == each other AND == the regenerated prompt's
    sha. Any mismatch (or a missing sha) → `RuntimeError` (exit 1), NO numbers
    emitted. (Aligns with Phase B's per-example provenance gate.)
  * joins binary correctness → recall@k + j0|HIT / j12-frozen|HIT / j12+LoRA|HIT
    + MISS-subset acc + paired-bootstrap j12−j0 gap, exactly like the LongEval
    cell.
  * self-tested on CPU with synthetic records: PASS path emits full 3-arm table;
    a single corrupted arm sha → abort exit 1.

* **`scripts/_run_p0_19_ruler_paired.sh`** — NEW 8-GPU flock task-pool launcher
  (same pattern as `_run_p1_9_dense_rag_8gpu.sh` / `_run_p0_20_phaseB_dense.sh`).
  DRY-by-default (`RUN=1` to execute). Pins `PYTHONHASHSEED=0`, offline env,
  empty proxies, `WANDB_MODE=offline`. STEP0 = QCMem j=0 self-test gate (read ==
  full forward, fp32 <1e-4; abort on fail). STEP2 = 3 arms × niah_multikey_1 ×
  {8k,16k} × 8 shards = 48 shard-jobs. STEP3 = analyzer join (aborts nonzero if
  the sha gate trips).

## 4. Flagship config (unified with all Paper A)

Qwen3-8B `models/Qwen3-8b-local`; selector `iter_bm25`, topk=12, iter_hop_topk=4,
iter_rounds=0 (=auto multi-hop); chunk_size=512; sink=bos; **chat_template=False**
(base LM); bf16; sdpa; seed=42. Three arms:
* **j0** — resume_j=0, NO LoRA → RAG full-depth recompute upper bound.
* **j12-frozen** — resume_j=12, NO LoRA → cached-state readout gap (adapter off).
* **j12+LoRA** — resume_j=12, flagship LoRA `outputs/qcmem_distill_qwen_j12_r32_4k/final`
  (r=32/α=64, layers 12..35) → the CoMem RULER leg.

Cohort = `niah_multikey_1 × {8k,16k}`, n=100 (== Phase B's RULER cell;
single-support NIAH, so the gold-support locator is faithful, matching the clean
qa1 / LongEval cells; VT is deliberately out of the headline — multi-support
locator is unfaithful per §caveat in P0_19_decomp_NOTES.md).

## 5. Reproduce / launch (.104, diskB — different FS, scripts rsync'd)

```bash
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
RUN=1 setsid nohup bash scripts/_run_p0_19_ruler_paired.sh \
  >/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/logs/p0_19_ruler/sched.out 2>&1 &
```
Output decomposition JSON: `paperA/p0_19_ruler_decomp.json`.

## 6. RESULTS

(filled after the .104 run — see final report / paperA/p0_19_ruler_decomp.json)
