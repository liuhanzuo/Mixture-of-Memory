"""Rebuild the 15,491,607-row dolmino corpus on wzc1 from the 84 local shards.

VERIFIED RECIPE (2026-08-15, byte-level):
  concat(sorted 84 shards) = 15,495,703 rows
  concat[0:4096]  == data/dolmino_now_val.npy   (md5 identical to zwfy6 copy)
  concat[4096:]   == the 15,491,607-row training corpus that keep8/keep10/keep12
                     resume from on zwfy6 (/dev/shm/dolmino_now15b.npy)
  32 spot-checks (incl. shard boundaries) against the local 7.57M-row PREFIX
  data/dolmino_now15b.npy all matched.
Writes to /dev/shm (tmpfs, 944G free) to match the zwfy6 recipe.

OUTPUT md5 = 7df19b217e5b0670d58bf6e01e6559d0, byte-identical to
.82:/dev/shm/dolmino_now15b.npy -- that equality is what makes a wzc1 arm
comparable to the keep8/keep12 arms training on zwfy6, so the consuming launcher
asserts this md5 (not just rows/size) before it will start.

WHY REBUILD INSTEAD OF TRANSFER: 153 s locally vs ~3.2 h for 118 GiB over the
cross-disk link (measured 17.7 MB/s single-stream, ~92 MB/s with 6 parallel
streams). /dev/shm is tmpfs and does NOT survive a reboot -- rerun this script.

⚠️ Do NOT substitute wzc1's data/dolmino_now15b.npy: it is a 7,570,911-row
PARTIAL PREFIX of this corpus (same leading bytes, less than half the rows), so
it looks plausible and silently trains on the wrong data.

Usage: /opt/conda/envs/torch-base/bin/python scripts/build_dolmino_corpus_wzc1.py
"""
import glob, numpy as np, os, sys, time

OUT = "/dev/shm/dolmino_now15b_wzc1.npy"
ROWS, COLS = 15491607, 2048
SKIP = 4096  # first 4096 rows are the val split

fs = sorted(glob.glob('/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/'
                      'data/dolmino_olmo2_shards/dolmino_chunks_2048_olmo2_shard*.npy'))
assert len(fs) == 84, len(fs)
arrs = [np.load(f, mmap_mode='r') for f in fs]
total = sum(a.shape[0] for a in arrs)
assert total == ROWS + SKIP, (total, ROWS + SKIP)
for a in arrs:
    assert a.dtype == np.uint32 and a.shape[1] == COLS, (a.dtype, a.shape)

out = np.lib.format.open_memmap(OUT, mode='w+', dtype=np.uint32, shape=(ROWS, COLS))
w = 0
skipped = 0
t0 = time.time()
for i, a in enumerate(arrs):
    n = a.shape[0]
    src_start = 0
    if skipped < SKIP:
        take = min(SKIP - skipped, n)
        src_start = take
        skipped += take
    if src_start >= n:
        continue
    m = n - src_start
    # copy in 20k-row blocks to bound RAM
    for s in range(src_start, n, 20000):
        e = min(s + 20000, n)
        out[w:w + (e - s)] = a[s:e]
        w += e - s
    if i % 10 == 0:
        print(f"shard {i}/{len(arrs)} rows_written={w} {time.time()-t0:.0f}s", flush=True)
out.flush()
del out
assert w == ROWS, (w, ROWS)
print("WROTE", OUT, "rows", w, "bytes", os.path.getsize(OUT), f"{time.time()-t0:.0f}s", flush=True)
