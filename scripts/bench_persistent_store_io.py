#!/usr/bin/env python
"""P2.2 — CoMem persistent-store real I/O & network-deployment benchmark.

This is a SYSTEMS / I-O benchmark, NOT a model-quality eval. No model is loaded.
We synthesise the CoMem persistent store (the write-once mid-layer residual h12)
as a deterministic bf16 tensor of shape [n_chunks, 512, 4096] = 8192 B/token and
measure it under four storage backends:

  (1) gpu      — store resident in HBM;    retrieval = on-device index_select gather
                 (Transfer ms == 0, pack already on device).
  (2) cpu      — store in pinned host RAM;  retrieval = gather top-12 rows on host
                 -> pinned pack -> H2D copy.
  (3) nvme     — store as one contiguous file on the LOCAL overlay mount;
                 retrieval = read the 12 chunk byte-ranges (O_DIRECT / pread at
                 known 4 MiB-aligned offsets) -> pinned pack -> H2D.
  (4) network  — same contiguous file layout on the CEPH dop-fuse mount;
                 retrieval = network read -> pinned pack -> H2D.

Per backend x store-size we report:
  Write GB/s  : ingest throughput writing the FULL store to that backend
                (total bytes / wall-time; disk/net writes are fsync'd).
  Retrieve ms : median time to fetch a random top-12 pack into a contiguous host
                (or device, for gpu) buffer. Index selection is GIVEN (random 12),
                we time the fetch, not BM25.
  Transfer ms : H2D time for the fixed 50.3 MB pack (0 for gpu-resident).
  QPS         : sustained random-top-12 queries/sec under a concurrent client pool
                (sweep 1/4/16 threads); each query = retrieve + H2D into device.
  Peak GPU / Peak host memory.

Store geometry (verified vs P1.1):
  chunk = 512 tok x 4096 dim x 2 B(bf16) = 4 MiB.  8192 B / token.
  128k tok -> 256 chunks -> 1.07 GB ; 1M -> 2048 -> 8.59 GB ; 4M -> 8192 -> 34.36 GB ;
  8M -> 16384 -> 68.7 GB ; 16M -> 32768 -> 137.4 GB.
  top-12 pack = 12 x 512 x 4096 x 2 B = 50.33 MB (P0.2 pinned H2D ~1.2 ms).

Methodology: median-of-N (default N=7) with >=2 warmups, pinned buffers for H2D,
torch.cuda.synchronize() around GPU timing, cuda.reset_peak_memory_stats() for peak
GPU, a VmRSS sampler thread + resource.getrusage for peak host. Content is
irrelevant to I/O timing; store is filled from a deterministic seed-42 bf16 staging
block (we time the write/transfer, not the RNG).

Author: LiuHanzuo (P2.2, Paper A / CoMem-QCMem).
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# store geometry
# --------------------------------------------------------------------------- #
CHUNK_TOK = 512
DIM = 4096
BF16_B = 2
CHUNK_BYTES = CHUNK_TOK * DIM * BF16_B          # 4 MiB
TOPK = 12
PACK_BYTES = TOPK * CHUNK_BYTES                 # 50.33 MB

SIZE_TOKENS = {
    "128k": 131072,
    "1M": 1048576,
    "4M": 4194304,
    "8M": 8388608,
    "16M": 16777216,
}


def n_chunks_for(size_key: str) -> int:
    return SIZE_TOKENS[size_key] // CHUNK_TOK


def store_bytes_for(size_key: str) -> int:
    return n_chunks_for(size_key) * CHUNK_BYTES


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def _rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / (1024 ** 2)  # kB -> GiB
    return 0.0


class RSSSampler:
    """Background thread tracking peak VmRSS (GiB)."""

    def __init__(self, period=0.05):
        self.period = period
        self.peak = _rss_gb()
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                self.peak = max(self.peak, _rss_gb())
                time.sleep(self.period)
        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1.0)
        self.peak = max(self.peak, _rss_gb())


def _staging_block(block_chunks: int, device="cpu", pinned=False):
    """Deterministic seed-42 bf16 staging block [block_chunks,512,4096]."""
    g = torch.Generator(device="cpu").manual_seed(42)
    t = torch.randn(block_chunks, CHUNK_TOK, DIM, generator=g).to(torch.bfloat16)
    if pinned:
        p = torch.empty(block_chunks, CHUNK_TOK, DIM, dtype=torch.bfloat16,
                        pin_memory=True)
        p.copy_(t)
        return p
    if device != "cpu":
        return t.to(device)
    return t


def _rand_idx(n_chunks: int, k: int = TOPK):
    # random top-k chunk ids (given selection; we time the fetch)
    return torch.randperm(n_chunks)[:k].tolist()


# --------------------------------------------------------------------------- #
# O_DIRECT capability probe
# --------------------------------------------------------------------------- #
def probe_odirect(path_dir: str) -> bool:
    """Return True if O_DIRECT aligned read works on this mount."""
    O_DIRECT = getattr(os, "O_DIRECT", 0)
    if not O_DIRECT:
        return False
    tmp = os.path.join(path_dir, ".odirect_probe.bin")
    try:
        # write 8 MiB
        with open(tmp, "wb") as f:
            f.write(b"\0" * (8 * 1024 * 1024))
            f.flush()
            os.fsync(f.fileno())
        fd = os.open(tmp, os.O_RDONLY | O_DIRECT)
        try:
            buf = _aligned_buffer(CHUNK_BYTES)
            n = os.preadv(fd, [buf], 0)
            ok = (n == CHUNK_BYTES)
        finally:
            os.close(fd)
        return ok
    except OSError:
        return False
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _aligned_buffer(nbytes: int, align: int = 4096) -> memoryview:
    """Page-aligned writable buffer for O_DIRECT preadv."""
    raw = ctypes.create_string_buffer(nbytes + align)
    addr = ctypes.addressof(raw)
    off = (align - (addr % align)) % align
    return memoryview((ctypes.c_char * nbytes).from_address(addr + off))


# --------------------------------------------------------------------------- #
# backend: write full store (ingest throughput)
# --------------------------------------------------------------------------- #
def write_store(backend: str, size_key: str, path: str, block_chunks: int):
    """Write full store, return dict {write_gb_s, store_bytes, handle...}.

    handle is what retrieve() needs: for gpu -> device tensor; cpu -> pinned tensor;
    file backends -> file path (+ odirect flag)."""
    N = n_chunks_for(size_key)
    total = store_bytes_for(size_key)
    total_gb = total / (1024 ** 3)

    if backend == "gpu":
        stage = _staging_block(block_chunks, device="cuda")
        dev = torch.empty(N, CHUNK_TOK, DIM, dtype=torch.bfloat16, device="cuda")
        _sync()
        t0 = time.perf_counter()
        i = 0
        while i < N:
            j = min(i + block_chunks, N)
            dev[i:j].copy_(stage[: j - i])
            i = j
        _sync()
        dt = time.perf_counter() - t0
        del stage
        torch.cuda.empty_cache()
        return {"write_gb_s": total_gb / dt, "handle": ("gpu", dev), "N": N}

    if backend == "cpu":
        stage = _staging_block(block_chunks, pinned=True)
        # Prefer a pinned store (spec); very large stores may exceed the OS
        # page-locked limit -> fall back to pageable host RAM (pack stays pinned
        # for H2D). Record which so the result is honest.
        store_pinned = True
        try:
            pinned = torch.empty(N, CHUNK_TOK, DIM, dtype=torch.bfloat16,
                                 pin_memory=True)
        except RuntimeError as e:
            if "memory" not in str(e).lower() and "pin" not in str(e).lower():
                raise
            store_pinned = False
            pinned = torch.empty(N, CHUNK_TOK, DIM, dtype=torch.bfloat16)
        t0 = time.perf_counter()
        i = 0
        while i < N:
            j = min(i + block_chunks, N)
            pinned[i:j].copy_(stage[: j - i])
            i = j
        dt = time.perf_counter() - t0
        del stage
        return {"write_gb_s": total_gb / dt, "handle": ("cpu", pinned), "N": N,
                "store_pinned": store_pinned}

    # file backends (nvme / network)
    stage = _staging_block(block_chunks)
    stage_bytes = stage.contiguous().view(torch.uint8).numpy().tobytes()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    written = 0
    t0 = time.perf_counter()
    while written < total:
        n = min(len(stage_bytes), total - written)
        os.write(fd, stage_bytes[:n] if n < len(stage_bytes) else stage_bytes)
        written += n
    os.fsync(fd)
    os.close(fd)
    dt = time.perf_counter() - t0
    del stage
    odirect = probe_odirect(os.path.dirname(path))
    return {"write_gb_s": total_gb / dt, "handle": (backend, path, odirect),
            "N": N, "odirect": odirect}


# --------------------------------------------------------------------------- #
# backend: retrieve one random top-12 pack
# --------------------------------------------------------------------------- #
def make_retriever(handle, N: int):
    """Return (retrieve_fn, transfer_fn, cleanup_fn).

    retrieve_fn(idx) -> puts the 50.3MB pack into a per-call/pinned host (or device)
        buffer, returns a token the transfer_fn understands.
    transfer_fn(tok) -> H2D copy of the pack (0-cost / no-op for gpu).
    """
    kind = handle[0]

    if kind == "gpu":
        dev = handle[1]
        dev_out = torch.empty(TOPK, CHUNK_TOK, DIM, dtype=torch.bfloat16,
                              device="cuda")

        def retrieve(idx):
            idx_t = torch.as_tensor(idx, device="cuda")
            torch.index_select(dev, 0, idx_t, out=dev_out)
            return dev_out

        def transfer(_tok):
            return  # already on device

        return retrieve, transfer, (lambda: None)

    if kind == "cpu":
        pinned = handle[1]
        pack = torch.empty(TOPK, CHUNK_TOK, DIM, dtype=torch.bfloat16,
                           pin_memory=True)
        dev_out = torch.empty(TOPK, CHUNK_TOK, DIM, dtype=torch.bfloat16,
                              device="cuda")

        def retrieve(idx):
            for j, i in enumerate(idx):
                pack[j].copy_(pinned[i])
            return pack

        def transfer(_tok):
            dev_out.copy_(pack, non_blocking=True)
            return dev_out

        return retrieve, transfer, (lambda: None)

    # file backends
    _, path, odirect = handle
    O_DIRECT = getattr(os, "O_DIRECT", 0)
    flags = os.O_RDONLY | (O_DIRECT if odirect else 0)
    fd = os.open(path, flags)
    pack = torch.empty(TOPK, CHUNK_TOK, DIM, dtype=torch.bfloat16, pin_memory=True)
    pack_np = pack.view(torch.uint8).numpy().reshape(TOPK, CHUNK_BYTES)
    dev_out = torch.empty(TOPK, CHUNK_TOK, DIM, dtype=torch.bfloat16, device="cuda")

    def retrieve(idx):
        if not odirect:
            # evict from page cache to measure the real medium (best-effort on fuse)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (OSError, AttributeError):
                pass
        for j, i in enumerate(idx):
            mv = memoryview(pack_np[j])
            got = os.preadv(fd, [mv], i * CHUNK_BYTES)
            if got != CHUNK_BYTES:
                raise IOError(f"short read {got} != {CHUNK_BYTES}")
        return pack

    def transfer(_tok):
        dev_out.copy_(pack, non_blocking=True)
        return dev_out

    def cleanup():
        os.close(fd)

    return retrieve, transfer, cleanup


# --------------------------------------------------------------------------- #
# timed measurements
# --------------------------------------------------------------------------- #
def measure_retrieve(retrieve, N, n_repeat, warmup, on_device):
    times = []
    for it in range(warmup + n_repeat):
        idx = _rand_idx(N)
        if on_device:
            _sync()
            t0 = time.perf_counter()
            retrieve(idx)
            _sync()
        else:
            t0 = time.perf_counter()
            retrieve(idx)
        dt = time.perf_counter() - t0
        if it >= warmup:
            times.append(dt)
    return statistics.median(times) * 1e3  # ms


def measure_transfer(retrieve, transfer, N, n_repeat, warmup, is_gpu):
    if is_gpu:
        return 0.0
    times = []
    for it in range(warmup + n_repeat):
        idx = _rand_idx(N)
        retrieve(idx)  # fill pinned pack (not timed here)
        _sync()
        t0 = time.perf_counter()
        transfer(None)
        _sync()
        dt = time.perf_counter() - t0
        if it >= warmup:
            times.append(dt)
    return statistics.median(times) * 1e3  # ms


def measure_qps(handle, N, threads_list, duration, is_gpu):
    """Sustained random-top-12 QPS: each query = retrieve + H2D into device.

    Each worker keeps a LIVE counter in a shared list (int store is GIL-atomic);
    the main thread warms up 1s, snapshots counts, sleeps `duration`, snapshots
    again, and reports (delta / window). This excludes the ramp-up cleanly."""
    out = {}
    for K in threads_list:
        # each thread gets its own retriever (own fd / own buffers)
        rets = [make_retriever(handle, N) for _ in range(K)]
        counts = [0] * K
        stop = threading.Event()

        def worker(w):
            retrieve, transfer, _ = rets[w]
            while not stop.is_set():
                idx = _rand_idx(N)
                retrieve(idx)
                transfer(None)
                if is_gpu:
                    _sync()
                counts[w] += 1  # live, GIL-atomic

        ts = [threading.Thread(target=worker, args=(w,)) for w in range(K)]
        for t in ts:
            t.start()
        time.sleep(1.0)                    # warmup / ramp-up
        snap0 = sum(counts)
        w0 = time.perf_counter()
        time.sleep(duration)               # measurement window
        snap1 = sum(counts)
        window = time.perf_counter() - w0
        stop.set()
        for t in ts:
            t.join()
        qps = (snap1 - snap0) / window
        out[K] = round(qps, 1)
        for _, _, cl in rets:
            cl()
    return out


# --------------------------------------------------------------------------- #
# driver for one (backend, size)
# --------------------------------------------------------------------------- #
def run_cell(backend, size_key, args):
    path = None
    if backend == "nvme":
        path = os.path.join(args.scratch_local, f"store_{size_key}.bin")
    elif backend == "network":
        path = os.path.join(args.scratch_net, f"store_{size_key}.bin")

    # Fresh allocator state so a prior cell's reserved-but-free blocks cannot
    # fragment this cell (was causing a spurious GPU OOM at 8M).
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec = {"backend": backend, "size": size_key,
           "tokens": SIZE_TOKENS[size_key],
           "n_chunks": n_chunks_for(size_key),
           "store_bytes": store_bytes_for(size_key),
           "store_gb": round(store_bytes_for(size_key) / (1024 ** 3), 3),
           "pack_bytes": PACK_BYTES,
           "n_repeat": args.n_repeat, "warmup": args.warmup,
           "qps_threads": args.qps_threads, "qps_duration_s": args.qps_duration}
    try:
        with RSSSampler() as rss:
            w = write_store(backend, size_key, path, args.block_chunks)
            handle, N = w["handle"], w["N"]
            rec["write_gb_s"] = round(w["write_gb_s"], 2)
            if "odirect" in w:
                rec["odirect"] = w["odirect"]
            if "store_pinned" in w:
                rec["store_pinned"] = w["store_pinned"]

            retrieve, transfer, cleanup = make_retriever(handle, N)
            is_gpu = (backend == "gpu")
            rec["retrieve_ms"] = round(
                measure_retrieve(retrieve, N, args.n_repeat, args.warmup, is_gpu), 4)
            rec["transfer_ms"] = round(
                measure_transfer(retrieve, transfer, N, args.n_repeat,
                                 args.warmup, is_gpu), 4)
            cleanup()

            rec["qps"] = measure_qps(handle, N, args.qps_threads,
                                     args.qps_duration, is_gpu)
            rec["peak_gpu_gb"] = round(_gpu_peak_gb(), 3)
            # Drop the store AND the closures that capture it, then GC +
            # empty_cache so the next cell starts from a clean allocator.
            del retrieve, transfer, cleanup, handle, w
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rec["peak_host_gb"] = round(rss.peak, 3)
        rec["peak_host_maxrss_gb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 3)
        rec["status"] = "ok"
    except (RuntimeError, MemoryError, OSError) as e:
        msg = str(e)
        rec["status"] = "OOM" if "out of memory" in msg.lower() else "error"
        rec["error"] = msg[:300]
        torch.cuda.empty_cache()
    finally:
        # clean scratch file (esp. shared CEPH which is 99% full)
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="gpu,cpu,nvme,network")
    ap.add_argument("--sizes", default="128k,1M,4M,8M,16M")
    ap.add_argument("--n-repeat", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--block-chunks", type=int, default=32,
                    help="staging block size in chunks (128 MiB @32)")
    ap.add_argument("--qps-threads", default="1,4,16")
    ap.add_argument("--qps-duration", type=float, default=4.0)
    ap.add_argument("--scratch-local", default="/root/p2_2_scratch")
    ap.add_argument("--scratch-net",
                    default="/apdcephfs_zwfy6/share_304376610/pighzliu_code/"
                            "Mixture-of-Memory/logs/p2_2/scratch")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.qps_threads = [int(x) for x in args.qps_threads.split(",")]

    backends = args.backends.split(",")
    sizes = args.sizes.split(",")

    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    meta = {
        "device": dev_name,
        "driver": "535.247.01",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "chunk_bytes": CHUNK_BYTES, "pack_bytes": PACK_BYTES,
        "topk": TOPK,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    results = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for size_key in sizes:
        for backend in backends:
            print(f"[run] backend={backend} size={size_key} "
                  f"({store_bytes_for(size_key)/(1024**3):.2f} GB) ...",
                  flush=True)
            rec = run_cell(backend, size_key, args)
            results.append(rec)
            print("       ->", json.dumps({k: rec.get(k) for k in (
                "status", "write_gb_s", "retrieve_ms", "transfer_ms",
                "qps", "peak_gpu_gb", "peak_host_gb", "error")}), flush=True)
            # incremental dump so partial failures keep prior cells
            with open(args.out, "w") as f:
                json.dump({"meta": meta, "results": results}, f, indent=2)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
