"""No-train probe: confirm the P8 null/sink slot is reachable and can absorb
softmax mass.

This loads the EXISTING P8 adapter (``outputs/mem_space_perdoc_chunk128_p8``)
with the NEW selector code (which adds a learnable ``null_key`` / ``null_value``
sink column to ``MemoryCrossAttentionRead.read``). The trained adapter has NO
sink params, so they load with their fresh init (null_key small-random,
null_value zero) via ``strict=False``.

We run ONE short 0k-ish forward (a few hundred tokens of plain text) and print,
per MemoryCrossAttentionRead instance, the mean softmax mass landing on the sink
column vs the mean mass over the N real slots.

>>> NOTE <<<
This probes an adapter trained WITHOUT the sink; we are only confirming the
mechanism RUNS and the sink is REACHABLE — real benefit (cold slots routing
their mass into the sink so the read injects ~nothing on irrelevant context)
requires a RETRAIN. At init null_value is zero, so sink mass contributes ~0 to
the read regardless.

Usage:
    /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/.venv/bin/python \
        scripts/probe_p8_null_sink.py [--device cuda:0]

If full-model loading is unavailable, pass ``--synthetic`` to instead exercise a
standalone MemoryCrossAttentionRead on random slots (still confirms the sink can
absorb mass).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.memory.mem_space.selector import MemoryCrossAttentionRead  # noqa: E402

DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "Meta-Llama-3-8B")
DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "outputs", "mem_space_perdoc_chunk128_p8")


def _collect_xattn_modules(model) -> list:
    """Return every MemoryCrossAttentionRead instance patched onto the model."""
    return [m for m in model.modules() if isinstance(m, MemoryCrossAttentionRead)]


def run_synthetic(device: torch.device) -> None:
    print("[probe] SYNTHETIC mode — exercising a standalone "
          "MemoryCrossAttentionRead on random slots (no model load).")
    torch.manual_seed(0)
    m = MemoryCrossAttentionRead(
        d_model=4096, n_heads=32, n_kv_heads=8, gate_init=0.4
    ).to(device)
    N = 128
    hidden = torch.randn(1, 256, 4096, device=device)
    slot_k = torch.randn(1, N, 4096, device=device)
    slot_v = torch.randn(1, N, 4096, device=device)
    with torch.no_grad():
        out = m.read(hidden, slot_k, slot_v)
    sink = m._last_sink_mass
    per_real = (1.0 - sink) / N
    print(f"[probe] out finite={torch.isfinite(out).all().item()} shape={tuple(out.shape)}")
    print(f"[probe] N real slots = {N}")
    print(f"[probe] mean sink-column mass        = {sink:.6f}")
    print(f"[probe] mean per-real-slot mass      = {per_real:.6f}")
    print(f"[probe] sink mass / per-real-slot    = {sink / max(per_real, 1e-12):.3f}x")
    print("[probe] NOTE: synthetic random slots; confirms the sink absorbs mass "
          "and the read runs. Real benefit requires a RETRAIN.")


def run_real_model(args, device: torch.device) -> None:
    # Reuse the babilong loader's model-build path verbatim.
    from scripts.run_babilong_mem_space import (  # noqa: E402
        build_mem_space_config,
        load_mem_space_model,
    )
    from transformers import AutoTokenizer  # noqa: E402

    ckpt = os.path.join(args.ckpt_dir, "mem_space_adapter.pt")
    cfg_path = os.path.join(args.ckpt_dir, "adapter_config.json")

    with open(cfg_path, "r") as f:
        adapter_cfg = json.load(f)
    mem_config = build_mem_space_config(adapter_cfg)
    print(f"[probe] use_memory_xattn={mem_config.use_memory_xattn} "
          f"num_slots={mem_config.num_slots} top_k={mem_config.top_k}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=ckpt,
        mem_config=mem_config,
        device=device,
        dtype=torch.bfloat16,
        attn_impl="sdpa",
    )

    xattn_mods = _collect_xattn_modules(model)
    print(f"[probe] found {len(xattn_mods)} MemoryCrossAttentionRead module(s)")
    if not xattn_mods:
        print("[probe] ERROR: no xattn modules found — is use_memory_xattn on?")
        return
    # Confirm fresh sink params loaded (untrained): null_value must be zero.
    nv0 = xattn_mods[0].null_value
    print(f"[probe] null_value[0] zero-init at load? "
          f"{torch.count_nonzero(nv0).item() == 0} "
          f"(abs-max={nv0.abs().max().item():.3e})")

    # Reset banks (cold start) then run a short 0k-ish forward.
    from scripts.run_babilong_mem_space import _reset_banks, _reset_l2  # noqa: E402
    _reset_banks(model)
    _reset_l2(model)

    text = (
        "The quick brown fox jumps over the lazy dog. " * 30
        + "Memory is a fixed-size buffer that compresses long context. " * 10
    )
    input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").to(device)
    print(f"[probe] forward on {input_ids.shape[1]} tokens (0k-ish, memory irrelevant)")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        _ = model(input_ids=input_ids, use_cache=False)

    sinks = [m._last_sink_mass for m in xattn_mods]
    N = mem_config.num_slots
    mean_sink = sum(sinks) / len(sinks)
    per_real = (1.0 - mean_sink) / N
    print("\n[probe] ===== SINK-MASS RESULTS (across "
          f"{len(xattn_mods)} layers) =====")
    print(f"[probe] N real slots                 = {N}")
    print(f"[probe] mean sink-column mass         = {mean_sink:.6f}")
    print(f"[probe] mean per-real-slot mass       = {per_real:.6f}")
    print(f"[probe] sink mass / per-real-slot     = {mean_sink / max(per_real, 1e-12):.3f}x")
    print(f"[probe] per-layer sink mass (first 8) = "
          f"{[round(s, 5) for s in sinks[:8]]}")
    print("\n[probe] NOTE: this probes an adapter trained WITHOUT the sink; we "
          "are only confirming the mechanism runs and the sink is reachable — "
          "real benefit requires retrain.")


def main() -> None:
    p = argparse.ArgumentParser(description="P8 null/sink reachability probe")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--synthetic", action="store_true",
                   help="Skip model load; probe a standalone module on random slots.")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device}")

    if args.synthetic:
        run_synthetic(device)
        return

    try:
        run_real_model(args, device)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\n[probe] real-model path failed ({type(e).__name__}: {e}); "
              "falling back to SYNTHETIC probe.")
        run_synthetic(device)


if __name__ == "__main__":
    main()
