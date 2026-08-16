# Disk survey — lane: `data/` and `models/` (both disks)

Surveyor lane, 2026-08-16. **READ-ONLY. Nothing was deleted, moved or truncated. No GPU touched
(no nvidia-smi, no launch, no kill).** Only file written is this one.

Repo: `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory` (wzc1)
Second disk: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory` (zwfy6)
Measured `df`: wzc1 120T/110T used/10T avail (92%); zwfy6 689T/667T used/22T avail (97%).

---

## 0. Method — and a flag bug worth remembering

`du -sh --max-depth=1` is an **invalid flag combination** (`-s` conflicts with `--max-depth`). It
returned `du: warning: summarizing conflicts with --max-depth=1`, rc=1, and **zero measurements** on
my first two attempts. Correct form: `du -b --max-depth=1 <dir>`.

On this CephFS (`dop-fuse`) the **size field of a directory in `ls -la` is the recursive byte total**
of that subtree. Validated against `du -b`:

| dir | `ls -la` field | `du -b` | match |
|---|---|---|---|
| wzc1 `data/` | 983,724,538,293 | 983,724,538,293 | exact |
| wzc1 `data/dolmino_olmo2_shards` | 126,940,820,276 | 126,940,820,276 | exact |
| wzc1 `models/` | 17,411,334,693 | 17,411,334,693 | exact |
| wzc1 `models/Meta-Llama-3-8B` | 16,069,769,797 | 16,069,769,797 | exact |

4/4 exact, so all byte counts below are **measured, not estimated**. Download sizes in §3 are the
only estimates and are labelled as such. rc captured on its own line (`cmd > f 2>&1; rc=$?`), never
through a pipe.

---

## 1. Headline: the earlier claim is wrong on both halves

Earlier claim: *"data 2.12 TiB, models 1.08 TiB, models is re-downloadable."*

| claim | measured | verdict |
|---|---|---|
| data 2.12 TiB | **1.41 TiB** (wzc1 916.2 GiB + zwfy6 528.9 GiB) | **overstated ~1.5x** |
| models 1.08 TiB | repo `models/` = **16.2 GiB** wzc1, **258.4 GiB** zwfy6 | **wrong directory** |

The "1.08 TiB" figure is real but belongs to the **parent** dir `pighzliu_code/models/`
(measured 1,174,911,445,854 B = **1.068 TiB** on wzc1) — one level ABOVE the repo. Acting on
"delete models/, 1.08 TiB, re-downloadable" would `rm` the repo's `models/` and free **16 GiB, not
1.1 TiB** — while the 1.068 TiB parent dir holds the weights **all five live runs are reading now**.

Also: `models/Qwen3-8b-local` is a **symlink** on both disks. Deleting it frees 0 bytes.

---

## 2. Per-subdirectory measurements

### 2a. wzc1 repo `data/` = 983,724,538,293 B = **916.2 GiB**

| entry | bytes | GiB | kind |
|---|---|---|---|
| dolmino-mix-1124-llama2 | 621,860,067,617 | 579.2 | raw HF corpus (Llama-2 tok, 778 shards) |
| dolmino_olmo2_shards | 126,940,820,276 | 118.2 | 84 tokenised shards — **rebuild source, see §4** |
| armt_pg19_real_tokenized_full | 67,346,218,640 | 62.7 | tokenised |
| dolmino_now15b.npy | 62,020,903,040 | 57.8 | tokenised — **7.57M-row PARTIAL PREFIX, see §5** |
| slimpajama_chunks_2048_qwen3base_full.npy | 22,164,381,824 | 20.6 | tokenised |
| slimpajama_chunks_4096_llama3.npy | 21,529,870,464 | 20.1 | tokenised |
| slimpajama-6b | 14,048,983,210 | 13.1 | raw HF corpus (tokeniser input) |
| slimpajama_chunks_4096.npy.BROKEN_llama2tok_uint16overflow | 12,830,695,552 | 11.9 | **known-broken, 0 refs** |
| pg19_train.jsonl | 11,450,766,349 | 10.7 | extracted corpus |
| slimpajama_chunks_2048_qwen3.npy | 9,239,134,336 | 8.6 | tokenised |
| slimpajama_chunks_2048_hunyuan.npy | 8,329,412,736 | 7.8 | tokenised |
| olmo2_sft | 1,765,212,416 | 1.64 | SFT data |
| mag_train_generated_causal.jsonl | 494,503,443 | 0.46 | generated |
| slimpajama_chunks_2048_hymt2.npy | 453,992,576 | 0.42 | tokenised |
| rmt_train_mag_concat_session.jsonl | 447,930,477 | 0.42 | generated |
| hf_datasets_cache | 398,107,725 | 0.37 | cache |
| mag_train_recall.jsonl | 381,442,669 | 0.36 | generated |
| longbench_raw | 366,705,323 | 0.34 | raw benchmark |
| longmemeval | 278,025,796 | 0.26 | raw benchmark |
| knowledge_axes | 206,520,985 | 0.19 | derived eval |
| rmt_train_mixed / wiki_zh_10k / wikitext | 355,113,260 | 0.33 | generated |
| pg19_chunks*.npy (3) | 240,066,944 | 0.22 | tokenised |
| mag_train_generated.jsonl | 121,423,145 | 0.11 | generated |
| slimpajama_val_* (3) | 107,979,136 | 0.10 | tokenised val |
| wikitext_chunks_llama3_4096.npy | 83,345,536 | 0.08 | tokenised |
| squad_raw | 81,636,712 | 0.08 | raw |
| paperC_squad_v2 | 76,346,909 | 0.07 | derived eval set |
| dolmino_now_val.npy | 33,554,560 | 0.031 | tokenised val (4096 rows) |
| raw/ | 23,256,410 | 0.02 | raw |
| wikitext_tokenized_1024.npy | 17,793,152 | 0.017 | tokenised |
| ood_ppl | 10,986,267 | 0.010 | eval |
| squad_train/val.jsonl, squad_sft_olmo2, mag_eval | ~17.4 M | 0.016 | small |
| cache/, processed/ (empty), dolmino_stage_now (5,289 B manifest) | ~0 | 0 | — |

### 2b. wzc1 repo `models/` = 17,411,334,693 B = **16.2 GiB** — only 3 entries

| entry | bytes | GiB | kind |
|---|---|---|---|
| Meta-Llama-3-8B | 16,069,769,797 | 14.97 | HF snapshot |
| bge-large-en-v1.5 | 1,341,564,829 | 1.25 | HF snapshot |
| Qwen3-8b-local | symlink → parent `models/Qwen--Qwen3-8b` | **0** | frees nothing |

### 2c. zwfy6 repo `data/` = 567,761,437,981 B = **528.9 GiB**

| entry | bytes | GiB | kind |
|---|---|---|---|
| dolmino_olmo2_shards | 126,940,820,401 | 118.2 | tokenised shards |
| dolmino_chunks_2048_olmo2.npy | 126,907,244,672 | 118.2 | tokenised (full 15.49M rows) |
| dolmino_now15b.npy | 126,907,244,672 | 118.2 | tokenised — same size as above |
| armt_pg19_real_tokenized_full | 87,044,060,416 | 81.1 | tokenised |
| slimpajama_chunks_2048_qwen3base_full.npy | 22,164,381,824 | 20.6 | **LIVE-READ by .104, see §3** |
| slimpajama_chunks_4096_llama3.npy | 21,529,870,464 | 20.1 | tokenised |
| slimpajama-6b | 14,048,983,210 | 13.1 | raw corpus (50 shards, tokeniser input) |
| slimpajama_chunks_4096.npy | 12,830,695,552 | 11.9 | tokenised |
| pg19_train.jsonl | 11,450,766,349 | 10.7 | corpus |
| slimpajama_chunks_2048_qwen3.npy | 9,239,134,336 | 8.6 | tokenised |
| olmo2_sft | 4,545,057,239 | 4.23 | SFT data |
| infinitebench | 484,202,257 | 0.45 | raw benchmark |
| longbench_raw | 480,637,978 | 0.45 | raw benchmark |
| mag_*/rmt_* jsonl (7) | 1,710,383,381 | 1.59 | generated |
| dialogmem | 296,219,799 | 0.28 | derived |
| remainder (val npys, squad, pg19_chunks, wikitext, ood_ppl, paperC_squad_v2, hf cache, armt smoke×2, codeparrot, proofpile, p1_qwen3_distractor_pool) | ~0.90 G | 0.84 | small |

⚠️ zwfy6 holds **three ~118 GiB near-duplicate dolmino tokenisations** (`dolmino_chunks_2048_olmo2.npy`,
`dolmino_now15b.npy`, and the `dolmino_olmo2_shards/` tree they were built from) = **354.6 GiB**,
i.e. **67% of zwfy6 `data/`**. This is the single largest consolidation opportunity in the lane —
but one of the three is the live-run rebuild source and one is byte-verified against a running job
(§5), so **which** copy is redundant needs an md5 comparison I did not run (it would read 354 GiB).

### 2d. zwfy6 repo `models/` = 277,445,714,614 B = **258.4 GiB**

| entry | bytes | GiB |
|---|---|---|
| Qwen3-32B | 65,540,308,519 | 61.0 |
| Qwen3-30B-A3B | 61,084,196,992 | 56.9 |
| Qwen3-14B | 29,552,619,815 | 27.5 |
| Llama--Llama2-7b | 26,953,781,249 | 25.1 |
| Beacon-Qwen2-7B | 16,168,476,719 | 15.06 |
| Meta-Llama-3-8B | 16,069,769,797 | 14.97 |
| Meta-Llama-3-8B-Instruct | 16,069,772,700 | 14.97 |
| Qwen2-7B-Instruct | 15,242,799,646 | 14.20 |
| Qwen3-4B | 8,060,930,437 | 7.51 |
| facebook--opt-2.7b | 5,303,359,381 | 4.94 |
| Qwen3-1.7B | 4,079,453,480 | 3.80 |
| openai-community--gpt2-large | 3,247,202,234 | 3.02 |
| Llama-3.2-1B-Instruct | 2,488,924,072 | 2.32 |
| bge-m3 | 2,315,472,329 | 2.16 |
| Qwen3-0.6B | 1,519,211,940 | 1.41 |
| openai-community--gpt2-medium | 1,520,013,706 | 1.42 |
| bge-large-en-v1.5 | 1,341,564,829 | 1.25 |
| openai-community--gpt2 | 548,118,077 | 0.51 |
| openai-community--gpt2-xl | 339,738,624 | 0.32 |
| Qwen3-8b-local | symlink | 0 |

### 2e. Parent `pighzliu_code/models/` — what "1.08 TiB" actually referred to

wzc1 parent = **1,174,911,445,854 B = 1.068 TiB**; zwfy6 parent = **1,192,782,207,705 B = 1.085 TiB**.

wzc1 parent, largest first:

| entry | bytes | GiB | note |
|---|---|---|---|
| Hy3 | 597,598,388,122 | 556.6 | Tencent Hunyuan-3 — **internal, NOT on HF** |
| Hunyuan-A13B-Pretrain | 160,806,848,934 | 149.8 | internal |
| Qwen3-30B-A3B | 76,051,453,135 | 70.8 | HF |
| Qwen3-32B | 65,540,278,672 | 61.0 | HF |
| Hy-MT2-30B-A3B | 60,147,303,166 | 56.0 | internal |
| Llama--Llama2-7b | 40,434,334,229 | 37.7 | HF, gated |
| **OLMo-2-1124-7B** | 29,204,228,800 | 27.2 | HF — **LIVE `--model_path` of 4 runs** |
| Qwen1.5-MoE-A2.7B | 28,644,042,322 | 26.7 | HF |
| Qwen3.5-9B | 19,329,398,399 | 18.0 | HF |
| Qwen--Qwen3-8b (Instruct) | 16,397,462,922 | 15.27 | HF; target of both `Qwen3-8b-local` symlinks |
| Qwen3-8B-Base | 16,393,044,618 | 15.27 | HF — **not interchangeable** (eos 151643 vs 151645) |
| Llama--Llama3-8b | 16,069,769,606 | 14.97 | HF |
| AST-official-LLaMA2-7B-2of4 | 13,479,303,457 | 12.55 | published sparse ckpt, SparseForge baseline |
| Qwen3-4B + Qwen3-4B-Base | 16,117,412,389 | 15.01 | HF |
| OLMo-2-0425-1B | 5,949,390,091 | 5.54 | HF |
| Qwen3-1.7B-Base, Qwen--Qwen3-1.7b, Qwen3-0.6B-Base/-Instruct, Llama-3.2-1B | 12,744,595,692 | 11.87 | HF |
| models--*/ stubs, .locks, modules, xet | ~4.9 M | 0.005 | HF cache scaffolding, ~empty |

zwfy6 parent extras: **GLM-5.1-FP8 = 756,209,332,932 B = 704.3 GiB** (largest single object anywhere
in this lane, 63% of that dir), deepseek-moe-16b-base 30.5 GiB, plus gpt2-xl/large/medium duplicates
totalling 66.6 GiB that are far larger than the same models under the repo `models/` (HF blob+snapshot
double-storage).

---

## 3. LIVE-RUN DEPENDENCY CHECK — this is the part that matters

I read actual command lines (`ps -eo cmd` locally, `pgrep -af` on remotes) and then verified open
file handles in `/proc/<pid>/{fd,maps}`. **This overturns the assumption that `data/` is cold.**

| node | run | `--data_path` actually in use | reads repo `data/`? |
|---|---|---|---|
| LOCAL | keep10fresh2 (step ~178.6k) | `/dev/shm/dolmino_now15b_wzc1.npy` | no (tmpfs) |
| .212 | keep14fresh2_distill (step ~38,980, log-confirmed) | `/dev/shm/dolmino_now15b_wzc1.npy` | no (tmpfs) |
| .73 | keep12fresh2 (step ~189k) | `/dev/shm/dolmino_now15b.npy` | no (tmpfs) |
| .82 | keep8fresh2 (step ~162k) | `/dev/shm/dolmino_now15b.npy` | no (tmpfs) |
| **.104** | **paperC_qwen3base_heal_k8f2** | **`data/slimpajama_chunks_2048_qwen3base_full.npy`** | **YES — RELATIVE PATH INTO `data/`** |

### 3a. `.104` is holding a live mmap on a file in `data/` — hard evidence

`.104`'s trainer was launched with a **relative** `--data_path data/slimpajama_chunks_2048_qwen3base_full.npy`
and `cwd = /apdcephfs_zwfy6/.../Mixture-of-Memory`. All **8 rank workers (PID 3343484–3343491)** show
`maps_hits=1 fd_hits=1`; the other ~40 python PIDs on the box show 0. Resolved handle:

```
/proc/3343484/fd/121 -> /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/data/slimpajama_chunks_2048_qwen3base_full.npy
7fe3d6e65000-7fe900000000 r--s 00000000 00:4b 14467180857113886515   <same path>
```

`r--s` = shared read-only mmap of a 20.6 GiB file. **Deleting `data/slimpajama_chunks_2048_qwen3base_full.npy`
on zwfy6 would corrupt or kill the .104 run at step ~62,760/200,000.** The earlier note's premise
that `data/` is safe to clean is false for at least this one file.

Rebuildable in principle: `scripts/preprocess_slimpajama.py` from `data/slimpajama-6b/data`
(50 shards, verified present on zwfy6) — but not while the run holds it.

### 3b. The `/dev/shm` runs are only *conditionally* independent of `data/`

The four dolmino runs mmap tmpfs, so an `rm` in `data/` would not kill the *current* processes. But
`/dev/shm` is tmpfs: it does not survive reboot or an shm purge. Both launchers say so explicitly —
`scripts/build_dolmino_corpus_wzc1.py:19` *"`/dev/shm` is tmpfs and does NOT survive a reboot — rerun
this script"*, and `launch_keep14_distill_resume_212_0815.sh:86` hard-fails
`FATAL: $DATA_PATH missing (tmpfs is wiped by reboot)`.

The rebuild reads `data/dolmino_olmo2_shards/` (all 84 shards). So **`dolmino_olmo2_shards/` is the
only re-staging source** for four runs at steps 162k–189k/200k. Deleting it converts a recoverable
restart into an unrecoverable one. **Treat it as live infrastructure, not cold data.**

---

## 4. `data/` — what a script can rebuild vs raw downloads

Tokeniser identified where I could find it (`grep -rln` over `scripts/`):

| artefact | GiB (wzc1) | rebuild script | needs input | verdict |
|---|---|---|---|---|
| dolmino_olmo2_shards | 118.2 | `scripts/tokenize_dolmino_olmo2.py`, `download_and_tokenize_dolmino.py` | dolmino-mix raw / HF | **NO — live rebuild source (§3b)** |
| dolmino_now15b.npy (wzc1) | 57.8 | `scripts/build_dolmino_corpus_wzc1.py` | the 84 shards | prefix, superseded (§5) |
| slimpajama_chunks_* (all toks) | 20.6+20.1+8.6+7.8+0.42 | `scripts/preprocess_slimpajama.py` | `data/slimpajama-6b` | rebuildable **except .104's live one** |
| armt_pg19_real_tokenized_full | 62.7 | `scripts/setup_data.sh` / `tokenize_pg19_fast.py` | pg19_train.jsonl | rebuildable, 11 refs |
| olmo2_sft | 1.64 | `scripts/prepare_olmo2_sft_data.py` | HF | rebuildable |
| paperC_squad_v2 + squad_sft_olmo2 | 0.08 | `scripts/make_paperC_squad_v2_npy.sh`, `tokenize_squad_olmo2_sft.py` | squad_raw | rebuildable, tiny |
| ood_ppl | 0.010 | `scripts/build_ood_ppl_npy.py` | HF | rebuildable, tiny |
| **dolmino-mix-1124-llama2** | **579.2** | raw HF download (`allenai/dolmino-mix-1124`), Llama-2 tok, 778 shards, 77.7B tokens per its `metadata.json` | proxy | re-downloadable but **5 refs incl. SparseForge union-9 watcher** |
| slimpajama-6b | 13.1 | raw HF download | proxy | re-downloadable; **is the tokeniser input above** |
| longbench_raw / longmemeval / squad_raw / infinitebench | 0.34/0.26/0.08/(z6 0.45) | raw benchmark downloads | proxy | re-downloadable |
| **knowledge_axes** | **0.19** | **no builder found (0 refs in scripts/)** | — | **see §6 IRREPLACEABLE** |
| mag_*/rmt_* generated jsonl | ~1.7 | `prepare_causal_data.py` / `prepare_mixed_dataset.py` (generation, likely non-deterministic) | — | **see §6** |
| slimpajama_chunks_4096.npy.BROKEN_llama2tok_uint16overflow | 11.9 | n/a | — | **0 refs, self-labelled broken → safest single delete** |

Largest genuinely re-downloadable item is **dolmino-mix-1124-llama2, 579.2 GiB** — that is 63% of
wzc1 `data/` and by itself bigger than the whole "models" claim. Re-download estimate: at a
*hypothetical* sustained 100 MB/s through `hy-proxy.woa.com:3128`, 579 GiB ≈ **1.7 h minimum**, plus
re-tokenisation; **I did not benchmark the proxy, so treat the time as an estimate**. Note the
in-repo measured cross-disk rate was only 17.7 MB/s single-stream / ~92 MB/s with 6 streams
(`build_dolmino_corpus_wzc1.py:17-19`), so 100 MB/s is optimistic, not conservative.

---

## 5. A trap in `data/`: wzc1's `dolmino_now15b.npy` is a PARTIAL PREFIX

Measured with numpy (`mmap_mode='r'`, no data read):

| file | rows | expected full |
|---|---|---|
| wzc1 `data/dolmino_now15b.npy` | **7,570,911** × 2048 uint32 | 15,491,607 |
| zwfy6 `data/dolmino_now15b.npy` | (126,907,244,672 B ⇒ 15,491,607 rows) | 15,491,607 |
| wzc1 84 shards, summed | **15,495,703** = 15,491,607 + 4,096 val | exact match |

`scripts/build_dolmino_corpus_wzc1.py:21-23` warns in its own docstring: *"Do NOT substitute wzc1's
`data/dolmino_now15b.npy`: it is a 7,570,911-row PARTIAL PREFIX of this corpus (same leading bytes,
less than half the rows), so it looks plausible and silently trains on the wrong data."*

Two consequences for a cleaner:
1. It is **not** a duplicate of the zwfy6 file despite the identical name — 57.8 GiB vs 118.2 GiB.
2. **7 scripts still pass it as `DATA=data/dolmino_now15b.npy`** (`_run_a03_arm3_cpt.sh`,
   `_run_a03_arm4_peaklr.sh`, `_run_a03_arm6_lowerband.sh`, `_run_a03_dataorder_repl.sh`,
   `_run_a04_stageB.sh`, `_run_a04_shallow_ladder.sh`, plus `shortgpt_select_layers.py` default and
   `audit_olmo2_dolmino_contamination.py`). So it is load-bearing for A03/A04 provenance even though
   it is the wrong-length file — deleting it silently breaks re-runs of already-published arms.

---

## 6. IRREPLACEABLE under `data/` — produced once, no script found

| artefact | GiB | why irreplaceable |
|---|---|---|
| `data/knowledge_axes` | 0.19 | **0 references in `scripts/`**; no builder found on either disk. Derived eval axes; if these back any published table, regenerating them is not defined. |
| `data/mag_train_generated{,_causal}.jsonl`, `mag_train_recall.jsonl`, `mag_eval_generated.jsonl` | 1.00 | model-**generated** synthetic data; generators (`prepare_causal_data.py`, `prepare_mixed_dataset.py`) are unseeded/likely non-deterministic → a rebuild yields *different* data, not the same data. Only 2 refs, but the bytes are the record. |
| `data/rmt_train_{mag_concat_session,mixed,wikitext,wiki_zh_10k}.jsonl` | 0.75 | same argument; 19 refs across scripts. |
| `data/dolmino_stage_now/` (5,289 B) + zwfy6 (10,920 B) | ~0 | manifest describing which shards formed the heal corpus — tiny, high provenance value, keep unconditionally. |
| `data/dolmino_now_val.npy` (4,096 rows) | 0.031 | md5-identical to the zwfy6 copy and asserted equal to `concat(84 shards)[0:4096]`; it is the **val split for the whole heal ladder**. Cheap to keep, expensive to get wrong. |

Under `models/`: nothing in either repo `models/` is a locally trained artefact — all 22 entries are
HF snapshots or symlinks. **In the parent dir, `Hy3` (556.6 GiB), `Hunyuan-A13B-Pretrain` (149.8 GiB)
and `Hy-MT2-30B-A3B` (56.0 GiB) are Tencent-internal weights, NOT on HuggingFace → not re-downloadable
by the documented proxy recipe**, and `AST-official-LLaMA2-7B-2of4` (12.55 GiB) is a published sparse
checkpoint that is a SparseForge baseline. Treat those four as not-ours / not-re-downloadable.

---

## 7. Ranked, with the constraint that nothing live may be touched

Safe now (0 references, self-labelled broken):
1. `data/slimpajama_chunks_4096.npy.BROKEN_llama2tok_uint16overflow` — **11.9 GiB**, 0 refs.

Large and re-downloadable, but check the 5 SparseForge refs first:
2. `data/dolmino-mix-1124-llama2` — **579.2 GiB** (wzc1 only; absent on zwfy6). Referenced by
   `_run_sparseforge_tokenmatched{,_resume}.sh`, `_run_alps_slorb_gate0.sh`, `_run_cast_direct.sh`,
   `_run_sparseforge_tokenmatched_union9_watcher.sh`. Task #245 (ALPS+SLoRB reproduction) is still
   `pending` and is exactly the kind of job that would want it. I could not verify `.21` liveness
   (password rejected on `configs/password_b200_19021.txt`), so **I cannot certify this idle.**

Biggest opportunity, needs one md5 job first:
3. zwfy6's triple dolmino tokenisation — **354.6 GiB**, of which ~118 GiB is plausibly redundant
   (`dolmino_chunks_2048_olmo2.npy` vs `dolmino_now15b.npy`, identical byte size). Requires comparing
   md5 and confirming which one four live runs' `/dev/shm` copies were built from.

Do **not** touch: `data/slimpajama_chunks_2048_qwen3base_full.npy` (live mmap, 8 PIDs),
`data/dolmino_olmo2_shards/` (sole rebuild source for 4 runs), `data/dolmino_now_val.npy`,
`data/dolmino_stage_now/`, repo `models/` (only 16 GiB, and the symlink frees 0),
parent `models/OLMo-2-1124-7B` and `Qwen3-8B-Base` (live `--model_path`).

## 8. Caveats

- `.212` not reachable at `28.89.182.12:36000` (timed out); its data path came from
  `logs/olmo2_7B_keep14_distill_212_0815.log:22` (`rows=15491607 from /dev/shm/dolmino_now15b_wzc1.npy`)
  plus its launcher, not from `/proc`. Its step ~38,980 is log-confirmed and advancing.
- `.21` SSH rejected the password file, so **SparseForge/CAST liveness on `.21` is unverified** — the
  579 GiB `dolmino-mix` recommendation is therefore conditional.
- zwfy6 numbers come from `ls -la` over one SSH hop to `.73`; validated identical to `du -b` on wzc1
  but not separately re-validated with `du -b` on zwfy6.
- I did not md5 anything ≥118 GiB; the duplicate-detection in §7.3 is inference from equal byte
  sizes, not proof.
- Row counts for zwfy6 `.npy` files are derived from byte size ÷ (2048 × 4 B), not read from headers.
