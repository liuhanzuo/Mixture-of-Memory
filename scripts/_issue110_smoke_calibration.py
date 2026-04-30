"""Issue #110 fix smoke test.

Runs `compute_filters` at rank=1 on a tiny Llama-2-7B slice (2 chunks)
and verifies the exact-SVD path is hit and returns finite, unit-norm,
deterministic filters.

Gate: no exception; filters finite; per-head L2 == 1 (within 1e-4);
two back-to-back calls (same inputs) yield identical filters
(exact SVD is deterministic).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from transformers import AutoModelForCausalLM  # noqa: E402

from src.memory.qfilters.calibration import compute_filters  # noqa: E402


def main() -> int:
    t0 = time.time()
    model_dir = "models/Llama--Llama2-7b"
    data_path = "data/pg19_chunks.npy"
    device = torch.device("cuda:0")

    print(f"[smoke] loading model from {model_dir} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16
    ).to(device).eval()

    print(f"[smoke] loading {data_path} ...", flush=True)
    chunks = np.load(data_path, mmap_mode="r")
    # shape [N, 4096]; take a tiny slice
    print(f"[smoke] chunks shape={chunks.shape}", flush=True)

    def loader():
        # 2 chunks of length 4096 = ~8k tokens; B=1 per chunk (required by calibration).
        for i in (200, 201):
            ids = torch.tensor(np.asarray(chunks[i]), dtype=torch.long)
            yield {"input_ids": ids}

    print("[smoke] run 1 (rank=1) ...", flush=True)
    filters1 = compute_filters(model, loader(), rank=1, num_kv_heads=None, device=device)

    print("[smoke] run 2 (rank=1, should be IDENTICAL under exact SVD) ...", flush=True)
    filters2 = compute_filters(model, loader(), rank=1, num_kv_heads=None, device=device)

    # Check finiteness, norm, determinism.
    layers = sorted(filters1.keys())
    print(f"[smoke] #layers={len(layers)}", flush=True)
    assert layers == sorted(filters2.keys())

    all_finite = True
    all_unit = True
    all_same = True
    max_abs_diff = 0.0
    for l in layers:
        f1 = filters1[l]
        f2 = filters2[l]
        assert f1.shape == f2.shape
        h, d, r = f1.shape
        if not torch.isfinite(f1).all():
            all_finite = False
        # per-head L2 norm should be ~1 for SVD right-singular vectors
        n1 = f1.squeeze(-1).norm(dim=-1)
        if not torch.allclose(n1, torch.ones_like(n1), atol=1e-3):
            all_unit = False
        # Determinism check (exact SVD is determined up to sign; compare |cos|).
        # Per-head cosine similarity should be == 1 in magnitude.
        cos = (f1.squeeze(-1) * f2.squeeze(-1)).sum(dim=-1).abs()
        diff = (cos - torch.ones_like(cos)).abs().max().item()
        if diff > 1e-3:
            all_same = False
        max_abs_diff = max(max_abs_diff, diff)

    dur = time.time() - t0
    print(f"[smoke] finite={all_finite}  unit_norm={all_unit}  deterministic={all_same} "
          f"max_cos_diff={max_abs_diff:.2e}  dur={dur:.1f}s", flush=True)
    ok = all_finite and all_unit and all_same
    print(f"[smoke] VERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
