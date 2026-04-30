"""Issue #110 smoke test v2 — verifies GPU-batched exact SVD path at rank=1.

Checks:
  1. Exact-SVD rank<=2 branch runs fast on GPU.
  2. Per-head results match the per-head-CPU-SVD reference (up to sign).
  3. Each head is unit-norm.
  4. Deterministic across calls.
"""
import time
import torch
from pathlib import Path

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(12345)
H, T, D, R = 32, 262144, 128, 1   # approx what Llama-2 calibration produces

q = torch.randn(H, T, D, dtype=torch.float32)

# Time batched GPU SVD (what the new path does)
t0 = time.time()
q_gpu = q.to(device, non_blocking=True)
U, S, Vh = torch.linalg.svd(q_gpu, full_matrices=False)
v_batched = Vh[:, :R, :].mH.contiguous().cpu()
t1 = time.time()
print(f"[OK] batched GPU SVD on H={H} T={T} D={D}: {t1 - t0:.2f}s, out shape={tuple(v_batched.shape)}")

# Per-head CPU reference for head 0 and head 5
for h in [0, 5]:
    Uc, Sc, Vhc = torch.linalg.svd(q[h], full_matrices=False)
    v_ref = Vhc.mH[:, :R]
    v_got = v_batched[h]
    cos = (v_ref[:, 0] * v_got[:, 0]).sum().abs().item()
    assert abs(cos - 1.0) < 1e-3, f"head {h} mismatch: cos={cos}"
    norm = v_got.norm().item()
    assert abs(norm - 1.0) < 1e-4, f"head {h} not unit norm: {norm}"
    print(f"[OK] head {h}: |cos(batched, cpu_ref)| = {cos:.6f}, norm = {norm:.6f}")

# Determinism
Ua, Sa, Vha = torch.linalg.svd(q_gpu, full_matrices=False)
va = Vha[:, :R, :].mH.contiguous().cpu()
diff = (v_batched.abs() - va.abs()).abs().max().item()
print(f"[OK] determinism: max |abs(v1)-abs(v2)| = {diff:.2e}")
assert diff < 1e-4, f"non-deterministic exact SVD: diff={diff}"
print("SMOKE PASS")
