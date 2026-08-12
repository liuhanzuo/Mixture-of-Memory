#!/usr/bin/env python3
"""A04 Job 1 -- EXECUTION proof that `seed=` on DistributedSampler is load-bearing.

Run 2026-08-12. CPU only: no GPU, no model, no data file is read (the dataset is a
FakeDS of the right LENGTH -- DistributedSampler only ever looks at len(ds)).

Why this exists. `STATUS.json:next_gate` listed a "~1 line, BLOCKING for K2" fix at
train_olmo2_arch_probe2.py:863 and asked whether it had landed. Reading the file can
only show the argument is PRESENT; it cannot show it CHANGES anything. This probe
instantiates the real torch sampler both ways and prints the batch-index sequences.

Result (see STATUS.json:sampler_fix_and_pilot_one_disposition_20260812):
  PRE-FIX  -- 6 different --seed values -> sampler.seed==0 for all, 1 distinct order.
  POST-FIX -- the same 6 values         -> sampler.seed==seed,   6 distinct orders.
  Slice overlap on the 16.53%-of-epoch window a 20k-step run consumes:
  post-fix seed43-vs-44 Jaccard 0.0102 (near-disjoint); pre-fix 1.0000 (identical).

Reproduce (either disk, any node, no GPU):
  /opt/conda/envs/torch-base/bin/python \
    proposal/active/A04-recovery-certification/code/a04_sampler_seed_probe.py
"""
import torch
from torch.utils.data.distributed import DistributedSampler

class FakeDS(torch.utils.data.Dataset):
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, i): return i

ds = FakeDS(15491607)   # dolmino_now15b.npy row count, per A03 DATAORDER_VERDICT
WORLD, RANK = 8, 0

def order(seed, use_fix, k=12):
    if use_fix:
        s = DistributedSampler(ds, num_replicas=WORLD, rank=RANK, shuffle=True, seed=seed)
    else:
        s = DistributedSampler(ds, num_replicas=WORLD, rank=RANK, shuffle=True)  # PRE-FIX: no seed=
    s.set_epoch(0)
    return list(s)[:k], s.seed

print("=== PRE-FIX  DistributedSampler(ds, shuffle=True)   [no seed=] ===")
for sd in (42, 43, 44, 101, 102, 103):
    o, internal = order(sd, use_fix=False)
    print(f"  --seed {sd:>3}  sampler.seed={internal}  first12={o}")

print()
print("=== POST-FIX DistributedSampler(ds, shuffle=True, seed=args.seed) ===")
for sd in (42, 43, 44, 101, 102, 103):
    o, internal = order(sd, use_fix=True)
    print(f"  --seed {sd:>3}  sampler.seed={internal}  first12={o}")

print()
pre  = {sd: order(sd, False)[0] for sd in (42,43,44,101,102,103)}
post = {sd: order(sd, True )[0] for sd in (42,43,44,101,102,103)}
print("PRE-FIX  distinct orders across 6 seeds:", len({tuple(v) for v in pre.values()}))
print("POST-FIX distinct orders across 6 seeds:", len({tuple(v) for v in post.values()}))

# overlap of the 16.53% epoch slice actually consumed (20000 steps x eff_bs 128)
def slice_set(seed, use_fix, nseq=2_560_000//WORLD):
    if use_fix:
        s = DistributedSampler(ds, num_replicas=WORLD, rank=RANK, shuffle=True, seed=seed)
    else:
        s = DistributedSampler(ds, num_replicas=WORLD, rank=RANK, shuffle=True)
    s.set_epoch(0)
    it = iter(s); return {next(it) for _ in range(nseq)}
a = slice_set(43, True); b = slice_set(44, True); c = slice_set(43, False); d = slice_set(44, False)
print(f"POST-FIX seed43 vs seed44 rank0 slice Jaccard = {len(a&b)/len(a|b):.4f}  (|A|={len(a)})")
print(f"PRE-FIX  seed43 vs seed44 rank0 slice Jaccard = {len(c&d)/len(c|d):.4f}  (identical => 1.0)")
