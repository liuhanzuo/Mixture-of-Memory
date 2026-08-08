#!/usr/bin/env python3
"""End-to-end integration test of the CAST training loop with a TINY model.

Exercises the real `train_cast_llama.main()` code path -- mask refresh timing,
grad accumulation, DDP no_sync, cast_loss, AdamS assertions, finalization -- by
stubbing `transformers.LlamaForCausalLM` with a tiny LLaMA-shaped model.  This
avoids installing transformers just to prove the loop runs.

    python tools/integration_tiny.py
    torchrun --nproc_per_node 2 tools/integration_tiny.py --ddp
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

H, I, L, V = 128, 256, 2, 512  # tiny LLaMA-ish dims (H % 4 == 0, I % 4 == 0)


class TinyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, H, bias=False)
        self.v_proj = nn.Linear(H, H, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        a = torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), is_causal=True
        ).squeeze(1)
        return self.o_proj(a)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttn()
        self.mlp = TinyMLP()
        self.n1 = nn.LayerNorm(H)
        self.n2 = nn.LayerNorm(H)

    def forward(self, x):
        x = x + self.self_attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class TinyOut:
    def __init__(self, logits):
        self.logits = logits


class TinyLlama(nn.Module):
    """Quacks like LlamaForCausalLM for the parts the trainer touches."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(V, H)
        self.layers = nn.ModuleList([TinyLayer() for _ in range(L)])
        self.norm = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, V, bias=False)
        self.config = types.SimpleNamespace(use_cache=False)

    def forward(self, input_ids=None, **kw):
        h = self.embed_tokens(input_ids)
        for lyr in self.layers:
            h = lyr(h)
        return TinyOut(self.lm_head(self.norm(h)))

    def gradient_checkpointing_enable(self, **kw):
        pass

    @classmethod
    def from_pretrained(cls, path, torch_dtype=None, **kw):
        m = cls()
        if torch_dtype is not None:
            m = m.to(torch_dtype)
        return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddp", action="store_true")
    args, rest = ap.parse_known_args()

    # stub transformers BEFORE importing the trainer
    stub = types.ModuleType("transformers")
    stub.LlamaForCausalLM = TinyLlama
    sys.modules["transformers"] = stub

    tmp = Path(tempfile.mkdtemp())
    (tmp / "data").mkdir()
    # tiny token stream, ids < V
    np.random.randint(0, V, size=200_000, dtype=np.uint16).tofile(tmp / "data" / "train.bin")

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tcl", str(HERE / "cast" / "train_cast_llama.py")
    )
    tcl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcl)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    sys.argv = [
        "train_cast_llama.py",
        "--project-root", str(tmp),
        "--model", ".",
        "--data", "data",
        "--out", "out",
        "--max-steps", "12",
        "--mask-period", "5",
        "--global-batch", str(2 * world),
        "--micro-batch", "1",
        "--seq-len", "64",
        "--warmup", "2",
        "--lr", "1e-3",
        "--min-lr", "1e-5",
        "--l1-decay", "1e-3",
        "--log-every", "4",
        "--diag-every", "6",
        "--save-every", "0",
        "--smoke",
        *rest,
    ]
    print(f"running tiny integration ({world} rank(s))", flush=True)
    tcl.main()

    if int(os.environ.get("RANK", "0")) == 0:
        final = tmp / "out" / "final_sparse.pt"
        assert final.exists(), "final_sparse.pt was not written"
        blob = torch.load(final, map_location="cpu", weights_only=False)
        sd = blob["model"]
        # verify exact 2:4 in the saved in-block projections
        checked = 0
        for k, v in sd.items():
            if any(k.endswith(f"{p}.weight") for p in
                   ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
                assert v.dim() == 2 and v.shape[1] % 4 == 0
                nz = (v != 0).reshape(v.shape[0], -1, 4).sum(-1)
                assert (nz == 2).all(), f"{k} is not exact 2:4 (nnz per group: {nz.unique().tolist()})"
                checked += 1
        assert checked == L * 7, f"checked {checked} projections, expected {L*7}"
        print(f"\nPASS integration: loop ran, {checked} saved projections are exact 2:4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
