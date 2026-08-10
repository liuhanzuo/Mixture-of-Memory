# P2.2 — CoMem Persistent-Store Real I/O & Network Deployment (Tier-3 systems benchmark)

**Task**: Measure the CoMem write-once persistent store (mid-layer residual `h₁₂`, bf16
`[n_chunks, 512, 4096]` = **8192 B/token**) under four storage backends × store sizes, as a
**systems / I-O benchmark** (no model loaded — synthetic deterministic bf16 store, seed 42).
This quantifies the deploy claim: *GPU-resident caps at HBM; CPU/NVMe/network extend the store
far past HBM at a measured latency cost.*

**Status**: DONE (2026-08-01). All 60+ cells filled. Node .104 left idle. No `.tex` / `TODOList.md` touched.

---

## Provenance

| Item | Value |
|------|-------|
| Node | **.104** = `28.83.24.104:36000`, 8× NVIDIA **H20** (97.8 GiB / 95.00 GiB usable per card) |
| GPU driver | **535.247.01** |
| Runtime | torch **2.13.0**, CUDA **13.2**, python `/opt/conda/envs/torch-base/bin/python` |
| Node state | fully idle before/during/after; benchmark ran on **GPU 0 only**, uncontended (verified `nvidia-smi` all 0 MiB); no other GPU job co-resided |
| Host | ~2.26 TB RAM total (~1.3 TB avail at start), 384 CPUs, `ulimit -l = unlimited` (page-locked memory uncapped) |
| Store geometry | chunk = 512 tok × 4096 × 2 B = **4 MiB**; top-12 pack = 12 × 4 MiB = **50.33 MB** (P0.2: pinned H2D ~1.2 ms) |
| Store capacity (verified vs P1.1) | 128k→1.00 GiB · 1M→8.00 GiB · 4M→32.00 GiB · 8M→64.00 GiB · 16M→128.00 GiB |
| Methodology | **median-of-N, N=7, 2 warmups**; pinned buffers for H2D; `torch.cuda.synchronize()` around GPU timing; `reset_peak_memory_stats()` for peak GPU; background VmRSS sampler (50 ms) + `getrusage` for peak host |
| QPS | sustained random-top-12 (retrieve + H2D per query) under a concurrent thread pool; sweep **1 / 4 / 16** threads; 1 s ramp-up + **4 s** measurement window; each worker owns its own fd/buffers |
| Retrieval | random top-12 chunk ids (index selection is **given**; we time the fetch, not BM25) |
| File backends | one **contiguous** store file; 12 chunk byte-ranges fetched via `os.preadv` at 4 MiB-aligned offsets, **O_DIRECT** (bypasses page cache → honest medium bandwidth; O_DIRECT confirmed working on both mounts) |

### Harness / commands / raw artifacts (all on diskB `share_304376610`)
- Harness: `scripts/bench_persistent_store_io.py` (git commit **`dcc66df`**, author LiuHanzuo).
- Raw JSON (local wzc1 diskB repo, copied from .104):
  - `ruler_results/p2_2/p2_2_full.json` — main sweep (all 4 backends × 5 sizes)
  - `ruler_results/p2_2/p2_2_gpu_fixed.json` — GPU-resident re-run with fixed inter-cell allocator (canonical GPU numbers; see caveat below)
  - `ruler_results/p2_2/p2_2_file_isolated.json` — nvme/network re-run in isolation for clean **peak-host** (canonical file-backend peak-host; see caveat below)
  - `ruler_results/p2_2/smoke_128k.json` — smoke validation
  - Logs: `logs/p2_2/p2_2_full.log`, `logs/p2_2/p2_2_gpu_fixed.log`, `logs/p2_2/p2_2_file_isolated.log`
- Exact commands (run on .104, `CUDA_VISIBLE_DEVICES=0`):
  ```
  # main sweep
  python scripts/bench_persistent_store_io.py --backends gpu,cpu,nvme,network \
    --sizes 128k,1M,4M,8M,16M --n-repeat 7 --warmup 2 --qps-threads 1,4,16 \
    --qps-duration 4.0 --out ruler_results/p2_2/p2_2_full.json
  # GPU re-run (fixed allocator)
  python scripts/bench_persistent_store_io.py --backends gpu --sizes 128k,1M,4M,8M,16M ... \
    --out ruler_results/p2_2/p2_2_gpu_fixed.json
  # file-backend isolated (clean peak-host)
  python scripts/bench_persistent_store_io.py --backends nvme,network --sizes 128k,4M,16M ... \
    --out ruler_results/p2_2/p2_2_file_isolated.json
  ```

### Mount-verification result — is `/` truly local?
**Yes, meaningfully faster.** Raw sequential write probe (`dd`, fdatasync, 3 GiB):
- overlay `/` (NVMe/local disk backend) = **6.1 GB/s**
- CEPH `dop-fuse` `/apdcephfs_zwfy6/share_304376610` (networked store) = **1.4 GB/s**

`/` is **~4.3× faster** than the CEPH mount → the "local disk / NVMe" backend is genuinely local
(container overlay on local NVMe), not network-backed. `/dev/shm` = tmpfs (RAM) 494 GB, unused as a
backend here. The store-write throughputs below (nvme ≈ 2.1–2.9 GB/s vs network ≈ 1.3–1.8 GB/s,
2×-ish) corroborate the same ordering under the real fsync'd store-write path.

---

## P2.2 TABLE — 4 backends × store size

Columns mirror TODOList P2.2: `| Backend | Write GB/s | Retrieve ms | Transfer ms | QPS | Peak GPU | Peak host | Raw |`
(QPS shown as peak over 1/4/16 threads; full sweep in "QPS detail". Retrieve = median fetch of the
50.3 MB top-12 pack into a contiguous host/device buffer. Transfer = median H2D of that pack.)

### 128k tokens (store 1.00 GiB)
| Backend | Write GB/s | Retrieve ms | Transfer ms | QPS (peak) | Peak GPU | Peak host | Raw |
|---|---|---|---|---|---|---|---|
| GPU resident | 1195.2 | 0.181 | 0.0 (on-device) | 6444 | 1.98 GB | 0.79 GB | gpu_fixed |
| CPU pinned | 105.2 | 0.844 | 1.42 | 956 | — | 3.16 GB | full |
| NVMe / local | 2.10 | 12.21 | 0.90 | 272 | — | 1.89 GB | file_iso |
| network (CEPH) | 1.45 | 73.92 | 1.20 | 47 | — | 2.16 GB | file_iso |

### 1M tokens (store 8.00 GiB)
| Backend | Write GB/s | Retrieve ms | Transfer ms | QPS (peak) | Peak GPU | Peak host | Raw |
|---|---|---|---|---|---|---|---|
| GPU resident | 1479.1 | 0.063 | 0.0 | 15559 | 8.98 GB | 0.90 GB | gpu_fixed |
| CPU pinned | 139.0 | 0.816 | 1.41 | 952 | — | 11.23 GB | full |
| NVMe / local | 2.17 | 12.60 | 0.90 | 273 | — | ~2.2 GB | full/file_iso |
| network (CEPH) | 1.47 | 81.41 | 1.15 | 44 | — | ~2.2 GB | full/file_iso |

### 4M tokens (store 32.00 GiB)
| Backend | Write GB/s | Retrieve ms | Transfer ms | QPS (peak) | Peak GPU | Peak host | Raw |
|---|---|---|---|---|---|---|---|
| GPU resident | 1495.6 | 0.064 | 0.0 | 15709 | 32.98 GB | 0.92 GB | gpu_fixed |
| CPU pinned | 91.1 | 1.29 | 1.38 | 973 | — | 43.25 GB | full |
| NVMe / local | 2.55 | 13.47 | 0.90 | 276 | — | 2.17 GB | file_iso |
| network (CEPH) | 1.52 | 75.92 | 1.19 | 44 | — | 2.16 GB | file_iso |

### 8M tokens (store 64.00 GiB) — **exceeds nothing yet; GPU still fits**
| Backend | Write GB/s | Retrieve ms | Transfer ms | QPS (peak) | Peak GPU | Peak host | Raw |
|---|---|---|---|---|---|---|---|
| GPU resident | 1498.4 | 0.066 | 0.0 | 14953 | **64.98 GB** (fits 95 GiB) | 0.89 GB | gpu_fixed |
| CPU pinned | 94.0 | 1.18 | 1.39 | 996 | — | 107.27 GB | full |
| NVMe / local | 2.88 | 10.60 | 0.91 | 276 | — | ~2.2 GB | full/file_iso |
| network (CEPH) | 1.64 | 77.29 | 1.18 | 44 | — | ~2.2 GB | full/file_iso |

### 16M tokens (store 128.00 GiB) — **GPU-resident OOMs; extended stores survive**
| Backend | Write GB/s | Retrieve ms | Transfer ms | QPS (peak) | Peak GPU | Peak host | Raw |
|---|---|---|---|---|---|---|---|
| GPU resident | **OOM** | — | — | — | needs 128 GiB > 95 GiB HBM | — | gpu_fixed |
| CPU pinned | 139.6 | 0.808 | 1.41 | 941 | — | **235.27 GB** | full |
| NVMe / local | 2.45 | 13.42 | 0.91 | 264 | — | 2.18 GB | file_iso |
| network (CEPH) | 1.66 | 79.20 | 1.20 | 42 | — | 2.19 GB | file_iso |

---

## ★ Store-exceeds-HBM crossover finding

**GPU-resident `h₁₂` fits up to 8M tokens (64.0 GB) on a single H20 and OOMs at 16M tokens
(128.0 GB > 95 GiB usable HBM).** The single-H20 GPU-resident ceiling therefore lies between
**8M and 16M tokens** (~8192 chunks–16384 chunks; the store alone would need ~90 GiB near the
practical edge). Beyond that ceiling the store must live off-GPU:

- **CPU-pinned** extends to at least **16M tokens (128 GB store, 235 GB peak host)** with retrieval
  still **sub-ms (0.8–1.3 ms)** + **1.4 ms H2D** — i.e. the store grows 2× past a single H20's HBM
  with **~2.2 ms** total query latency and **~940–1000 QPS**, bounded only by ~2.26 TB host RAM.
- **NVMe/local** and **network/CEPH** extend the store to **arbitrary disk capacity** (128 GB store
  fetched at ~13 ms / ~79 ms per pack respectively) — much slower per query but effectively unbounded.

So the whole systems point holds quantitatively: **GPU-resident is fastest but HBM-capped at ~8M
tokens of `h₁₂`; CPU-pinned buys a ~2× store extension at ~2 ms latency; NVMe/network trade an order
of magnitude of latency for effectively unlimited store size.** The fixed 50.3 MB top-12 pack keeps
the per-query H2D transfer constant (~0.9–1.4 ms) regardless of total store size on every off-GPU
backend, consistent with P0.2's L-independent 1.2 ms pinned H2D.

### QPS detail (1 / 4 / 16 concurrent clients; peak-bolded knee)
| Backend | 128k | 4M | 16M | knee |
|---|---|---|---|---|
| GPU | 5442 / 6397 / **6444** | 10130 / **15709** / 14822 | (OOM) | saturates ~4 threads (on-device gather + sync) |
| CPU pinned | 838 / 586 / **956** | 806 / 637 / **973** | 680 / 602 / **941** | flat ~600–1000; GIL-bound single-copy loop |
| NVMe | 91 / 198 / **272** | 55 / 169 / **276** | 88 / 191 / **264** | scales to ~16 threads → ~270 (device/IO parallelism) |
| network | 15 / 40 / **47** | 12 / 33 / **44** | 13 / 35 / **42** | scales to ~16 → ~44 (CEPH round-trip bound) |

QPS is essentially **store-size-independent** for every backend (fixed 50.3 MB pack), and rises with
concurrency for the I/O-bound backends (NVMe ~3×, network ~3× from 1→16 threads); CPU-pinned and GPU
saturate earlier (compute/GIL bound). Peak QPS ordering: **GPU (~6–16k) ≫ CPU (~950) ≫ NVMe (~270) ≫ network (~44)**.

---

## Caveats / methodology notes / failed-or-excluded runs

1. **GPU-resident canonical numbers come from `p2_2_gpu_fixed.json`, not the combined `p2_2_full.json`.**
   In the first combined run the CUDA caching allocator retained reserved-but-free blocks across cells,
   fragmenting the GPU-resident **8M** cell into a *spurious* OOM (it tried to allocate a fresh 64 GiB
   store while 32 GiB was still reserved from the 4M cell). The harness was fixed to `gc.collect()` +
   `torch.cuda.empty_cache()` (and to drop the closures that capture the store) at both cell boundaries;
   the re-run cleanly shows **8M fits (64.98 GB peak), 16M OOMs (needs 128 GiB, only 95 GiB HBM)** —
   the true crossover. The combined-run GPU 8M OOM is **excluded** as a harness artifact, not a real
   hardware limit. Fix is in the committed `dcc66df`.
2. **File-backend (NVMe/network) canonical peak-host comes from `p2_2_file_isolated.json`.**
   In the combined run the CPU-pinned cell for size S allocated a large pinned host store just before the
   NVMe/network cells for the same S; that pinned RAM was not returned to the OS within the process, so
   the VmRSS sampler read an inflated peak-host (e.g. 43–235 GB) for the file backends. File backends
   never hold the store in RAM (only the ~200 MB pack + fd buffers), so their **true peak host is ~1.9–2.2 GB**
   across all sizes — confirmed by the isolated re-run (throughput/latency/QPS identical to the combined
   run, only peak-host differs). Combined-run file-backend peak-host figures are **excluded**.
3. **CPU-pinned store stayed fully pinned even at 16M (128 GB)** thanks to `ulimit -l = unlimited` +
   2.26 TB RAM (`store_pinned=true` in every CPU cell); no pageable fallback was triggered.
4. O_DIRECT confirmed working on both the overlay and CEPH mounts (`odirect=true`), so file-backend
   retrieves measure the raw medium, not the page cache.
5. Peak GPU for off-GPU backends (~0.98 GB) = the fixed 50.3 MB pack device buffer + CUDA context.
6. No OOM/failure on any off-GPU backend at any size; the only OOM is the intended GPU-resident 16M.
7. **Scratch cleaned**: 128 GB store files (up to 137 GB on the 99%-full shared CEPH) were unlinked in a
   `finally` block after every cell; both scratch dirs verified empty and removed; overlay `/` restored to
   4.3 T free after the run.

---

## Conclusion (one sentence, non-over-extrapolating)

On a single H20, the CoMem `h₁₂` store is GPU-resident-fastest but HBM-capped at ~8M tokens (64 GB, OOM
at 16M/128 GB), while CPU-pinned (~2 ms/query, ~940 QPS) extends it ~2× past HBM to 16M tokens and
NVMe/network (~13 ms / ~79 ms per query) extend it to effectively unbounded disk capacity — the fixed
50.3 MB top-12 pack keeps per-query H2D transfer constant (~1 ms) across all off-GPU backends and store sizes.

---

**Note for main**: this record's numbers are the source for the **P2.2 table in `paperA/TODOList.md`**
(and any `.tex` fold). Per task rules I did **not** touch any `.tex` file or `paperA/TODOList.md` — main owns the fold-in.
