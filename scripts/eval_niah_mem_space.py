#!/usr/bin/env python3
"""Needle-in-Haystack (NIAH) evaluation for the mem_space streaming memory architecture.

Tests whether memory slots can retain and retrieve a "needle" fact embedded at
varying depths in a long context, streamed as seq_len=4096 chunks.

The model is LlamaForCausalLM patched with MemorySpaceLayer.  The full haystack
is drawn from pg19_chunks_llama3.npy and fed to the model in non-overlapping
seq_len chunks — memory persists across chunks (NO reset between them) so the
whole context accumulates in the memory bank before generation.

Architecture notes
------------------
* ``apply_mem_space_to_model(model, cfg)`` wraps every decoder layer.
* Memory bank persists across forward calls UNLESS ``_reset_banks(model)`` is
  called.  Call it once per sample before streaming, never between chunks.
* ``--bypass_memory`` monkey-patches all layers to ``forward_no_memory`` for an
  ablation run that exercises identical code paths but with memory disabled.
* The model's forward() signature is identical to vanilla Llama (MemorySpaceLayer
  is transparent to LlamaForCausalLM).
* ``use_cache=False`` in ALL forward calls — the custom attention layer does not
  support KV caching.
* H7 fix v2 is applied after ``.to(device, dtype)`` to restore rotary buffers to
  float32 (see inline comment).

Usage examples
--------------
# With memory (champion config):
python scripts/eval_niah_mem_space.py \\
    --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \\
    --data  /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \\
    --output_dir outputs/niah_mem_space_champion \\
    --slot_init random --slot_init_noise 0.05 \\
    --writeback_warmup_steps 1000 \\
    --shared_memory_bank \\
    --unfreeze_hidden_to_slot

# Without memory (bypass control):
python scripts/eval_niah_mem_space.py \\
    --model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \\
    --data  /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \\
    --output_dir outputs/niah_bypass \\
    --bypass_memory
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import traceback
from typing import List

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM

from src.memory.mem_space import (
    MemorySpaceConfig,
    apply_mem_space_to_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Memory helpers  (copied verbatim from train_mem_space_pg19.py)
# --------------------------------------------------------------------------- #


def _reset_banks(model: torch.nn.Module) -> None:
    """Wipe per-sample slot state between documents (chunks).

    Branch-3 (2026-04-26): under ``config.shared_memory_bank=True`` patch.py
    exposes ``_mem_space_shared_bank``; resetting it once is equivalent to
    resetting every wrapper's bank (they all point to the same object) but
    avoids 32x wasted work and, more importantly, guarantees only one
    inter-chunk graph break even if the wrappers were mid-iteration.
    """
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.reset()
        return
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return
    for w in mem_layers:
        w.memory_bank.reset()


def _freeze_banks(model: torch.nn.Module) -> None:
    """Freeze all memory banks so greedy generation does not overwrite needle info."""
    shared_bank = getattr(model, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.frozen = True
    else:
        for w in getattr(model, "_mem_space_layers", []):
            w.memory_bank.frozen = True


def _unfreeze_banks(model: torch.nn.Module) -> None:
    """Unfreeze memory banks (call after generation is done)."""
    shared_bank = getattr(model, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.frozen = False
    else:
        for w in getattr(model, "_mem_space_layers", []):
            w.memory_bank.frozen = False


# --------------------------------------------------------------------------- #
# Needle / haystack construction
# --------------------------------------------------------------------------- #


def make_needle(rng: random.Random) -> tuple[str, str, str]:
    """Return (needle_sentence, question_suffix, code).

    needle_sentence  — injected verbatim into the haystack token stream at the
                       chosen depth position.  Uses a highly unnatural format
                       (MEMORIZE/END_MEMORIZE) that cannot appear in pg19 prose,
                       preventing false-positive matches during generation.
    question_suffix  — appended AFTER the full haystack is streamed; it is NOT
                       part of the memory input.  Uses the exact same key format
                       as the needle so the model can complete the cloze.
    code             — 5-digit string the model must reproduce (expected answer)

    Fix F (2026-04-27): Changed from the old "The secret passphrase for
    experiment {name} is: {code}" format to a structured key=value format.
    The old format overlapped ambiguously with pg19 continuation text — the
    model would generate pg19 prose instead of recalling the needle, because
    strings like "1234567890" genuinely appear in the haystack.  The new
    MEMORIZE/END_MEMORIZE sentinel is vanishingly unlikely to appear in pg19.
    """
    name: str = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=6))
    code: str = "".join(rng.choices("0123456789", k=5))
    # Fix F: Use a format that is extremely unlikely to appear in pg19 prose.
    # The structured key=value layout also makes the expected completion
    # unambiguous: the model must output exactly the 5-digit code.
    needle_sentence = f"The secret number for agent {name} is {code}."
    question_suffix = f"\n\nThe secret number for agent {name} is "
    return needle_sentence, question_suffix, code


def build_haystack_ids(
    pg19_data: np.ndarray,
    target_len: int,
    skip_chunks: int,
) -> list[int]:
    """Flatten consecutive pg19 rows into a 1-D token list of exactly target_len tokens.

    Rows are taken starting at skip_chunks and wrap around to the beginning of
    the array if the array is exhausted before target_len tokens are collected.

    Args:
        pg19_data:    Memory-mapped numpy array of shape [N, chunk_size] (int32/int64).
        target_len:   Number of tokens required for the haystack.
        skip_chunks:  First row index to use (skip training rows to avoid contamination).

    Returns:
        List of target_len integer token IDs.
    """
    n_rows = len(pg19_data)
    flat: list[int] = []
    chunk_idx = skip_chunks
    while len(flat) < target_len:
        flat.extend(int(x) for x in pg19_data[chunk_idx % n_rows])
        chunk_idx += 1
    return flat[:target_len]


def insert_needle(
    haystack_ids: list[int],
    needle_ids: list[int],
    depth: float,
) -> list[int]:
    """Insert needle_ids at token position ``int(depth * len(haystack_ids))``.

    depth=0.0 → beginning of haystack, depth=1.0 → end of haystack.

    Returns a list of length len(haystack_ids) + len(needle_ids).
    """
    insert_pos = int(depth * len(haystack_ids))
    insert_pos = max(0, min(insert_pos, len(haystack_ids)))
    return haystack_ids[:insert_pos] + needle_ids + haystack_ids[insert_pos:]


# --------------------------------------------------------------------------- #
# Streaming memory processing
# --------------------------------------------------------------------------- #


def stream_haystack(
    model: torch.nn.Module,
    tokenizer,
    haystack_ids: list[int],
    seq_len: int,
    device: torch.device,
) -> None:
    """Process haystack in seq_len chunks, accumulating memory.

    NO reset between chunks — memory carries state across the full haystack so
    the model can integrate information from the entire long context.

    Call ``_reset_banks(model)`` BEFORE this function to get a clean slate for
    the current sample.  This function intentionally does NOT call _reset_banks,
    allowing memory to compound across chunks.
    """
    pad_id: int = tokenizer.pad_token_id
    model.eval()
    for i in range(0, len(haystack_ids), seq_len):
        chunk = haystack_ids[i : i + seq_len]
        if len(chunk) < 4:
            break
        # Pad the last (short) chunk to seq_len so attention shapes are uniform.
        if len(chunk) < seq_len:
            chunk = chunk + [pad_id] * (seq_len - len(chunk))
        chunk_tensor = torch.tensor([chunk], device=device, dtype=torch.long)
        with torch.no_grad():
            _ = model(input_ids=chunk_tensor, use_cache=False)


# --------------------------------------------------------------------------- #
# Greedy generation (no KV cache)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def greedy_generate(
    model: torch.nn.Module,
    question_ids: list[int],
    device: torch.device,
    max_new_tokens: int = 32,
    eos_id: int | None = None,
) -> list[int]:
    """Generate after memory is already populated from haystack streaming.

    DO NOT call _reset_banks here — memory must reflect the full haystack that
    was streamed before this call.

    Re-feeds the full growing sequence at every step (no KV cache) because the
    custom MemorySpaceLayer attention does not support use_cache=True.
    """
    input_ids = torch.tensor([question_ids], device=device, dtype=torch.long)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids, use_cache=False)
        next_id: int = out.logits[0, -1].argmax().item()
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device, dtype=torch.long)],
            dim=1,
        )
    return generated


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def exact_match(generated_text: str, code: str) -> bool:
    """Return True iff the expected code appears anywhere in generated_text.

    Works for both the old "secret passphrase" format and the new
    MEMORIZE/END_MEMORIZE format (Fix F, 2026-04-27).  The expected code is
    always a 5-digit string; substring search is sufficient.
    """
    return code in generated_text


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIAH evaluation for mem_space streaming memory (Llama-3-8B)"
    )

    # --- required --- #
    p.add_argument("--model", type=str, required=True,
                   help="Path to Llama-3-8B (or compatible) base model directory")
    p.add_argument("--data", type=str, required=True,
                   help="Path to pg19_chunks_llama3.npy (pre-tokenized pg19 chunks)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write niah_results.json")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to saved adapter state dict (.pt file from "
                        "train_mem_space_pg19.py). If omitted the adapter "
                        "uses random init weights (only valid for bypass "
                        "ablation runs).")

    # --- streaming / generation --- #
    p.add_argument("--seq_len", type=int, default=4096,
                   help="Chunk size for haystack streaming (default: 4096)")
    p.add_argument("--max_new_tokens", type=int, default=32,
                   help="Max greedy-decoding steps for the answer (default: 32)")

    # --- MemorySpace hypers (mirrors train_mem_space_pg19.py defaults) --- #
    p.add_argument("--num_slots", type=int, default=512,
                   help="N — per-layer slot count, kv_budget-equivalent (default: 512)")
    p.add_argument("--top_k", type=int, default=64,
                   help="k — slots prepended to the extended sequence (default: 64)")
    p.add_argument("--selector_dim", type=int, default=128,
                   help="Selector projection dimensionality (default: 128)")
    p.add_argument("--writeback_gate_max", type=float, default=0.3,
                   help="Max writeback gate value (default: 0.3)")
    p.add_argument("--writeback_warmup_steps", type=int, default=0,
                   help="beta warm-up steps; 0 = beta kicks in immediately "
                        "(eval-only default)")
    p.add_argument("--load_balance_weight", type=float, default=0.01,
                   help="Load-balance auxiliary loss coefficient (default: 0.01)")
    p.add_argument("--slot_init", type=str, default="random",
                   choices=["zero", "random", "hidden_pool"],
                   help="Slot initialisation strategy (default: random)")
    p.add_argument("--slot_init_noise", type=float, default=0.05,
                   help="Std of slot init noise (default: 0.05)")
    p.add_argument("--unfreeze_hidden_to_slot", action="store_true",
                   help="Allow hidden_to_slot projection to be used with gradients "
                        "(no training effect in eval, but matches the training config)")

    # --- shared memory bank (Branch-3, mutually exclusive, default ON) --- #
    mbg = p.add_mutually_exclusive_group()
    mbg.add_argument("--shared_memory_bank", dest="shared_memory_bank",
                     action="store_true", default=True,
                     help="Share one MemoryBank across all patched decoder layers "
                          "(Branch 3 Option A.2; default)")
    mbg.add_argument("--no_shared_memory_bank", dest="shared_memory_bank",
                     action="store_false",
                     help="Ablation: per-layer banks (no cross-layer writeback BPTT)")

    # --- ablation --- #
    p.add_argument("--bypass_memory", action="store_true",
                   help="Monkey-patch all MemorySpaceLayer.forward to "
                        "forward_no_memory (no slot prepend, no writeback). "
                        "Used to verify that the wrapping itself is clean.")

    # --- model / device --- #
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])

    # --- eval grid --- #
    p.add_argument("--context_lengths", type=str, default="8192,16384,32768",
                   help="Comma-separated context lengths in tokens "
                        "(default: 8192,16384,32768)")
    p.add_argument("--depths", type=str, default="0.1,0.3,0.5,0.75",
                   help="Comma-separated fractional needle insertion depths 0-1 "
                        "(default: 0.1,0.3,0.5,0.75)")
    p.add_argument("--num_samples", type=int, default=5,
                   help="Samples per (context_len, depth) cell (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for needle generation (default: 42)")
    p.add_argument("--skip_pg19_chunks", type=int, default=0,
                   help="Skip first N pg19 rows to avoid train contamination "
                        "(default: 0 for eval mode)")

    # --- N_list mode (chunk-gap grid, 2026-04-27) --- #
    p.add_argument("--N_list", type=str, default=None,
                   help="Comma-separated list of N_gap values (number of chunks between needle "
                        "and query). E.g. '1,2,4,8,16,32'. When set, overrides --context_lengths "
                        "and --depths: eval grid becomes N_gap x samples_per_cell instead.")
    p.add_argument("--samples_per_cell", type=int, default=5,
                   help="Samples per cell when using --N_list mode (default: 5)")
    p.add_argument("--chunk_size", type=int, default=None,
                   help="Chunk size for N_list mode (default: same as --seq_len)")
    p.add_argument("--output_csv", type=str, default=None,
                   help="If set, write per-N_gap accuracy to this CSV file in addition to JSON.")

    # --- SWA window for eval (2026-04-27) --- #
    p.add_argument("--swa_window", type=int, default=0,
                   help="SWA sliding window size (0=full causal). Passed to MemorySpaceConfig.")

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    # ------------------------------------------------------------------ #
    # Fix E (2026-04-27): Multi-process torchrun guard.
    #
    # This script is single-GPU only — do NOT launch with torchrun.
    # Preferred launch: plain `python scripts/eval_niah_mem_space.py ...`
    #
    # If accidentally launched via `torchrun --nproc_per_node 8`, all 8
    # worker processes would try to use cuda:0 simultaneously (8 × 22 GB =
    # OOM at ctx=32768) AND each would independently run the full eval grid,
    # producing 8× interleaved duplicate results in the log.
    #
    # Guard: when LOCAL_RANK is set (torchrun environment), ranks 1-7 exit
    # immediately and rank-0 continues as the sole worker on its assigned GPU.
    # ------------------------------------------------------------------ #
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank != 0:
            # Non-rank-0 processes have nothing to do — exit immediately.
            sys.exit(0)
        # Rank-0 continues; it will use the GPU assigned by LOCAL_RANK below.

    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    # Single-GPU eval — no DDP / torchrun needed.
    # When launched via torchrun, LOCAL_RANK is set to 0 here (ranks 1-7 have
    # already exited above), so we honour the GPU assignment from torchrun.
    # When launched via plain `python`, LOCAL_RANK is absent → default to 0.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }[args.dtype]

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse evaluation grid axes
    context_lengths: list[int]  = [int(x.strip()) for x in args.context_lengths.split(",")]
    depths:          list[float] = [float(x.strip()) for x in args.depths.split(",")]

    # N_list eval mode (chunk-distance based, overrides context_lengths × depths grid)
    n_list_mode: bool = bool(args.N_list and args.N_list.strip())
    N_list: list[int] = []
    if n_list_mode:
        N_list = [int(x.strip()) for x in args.N_list.split(",") if x.strip()]
        logger.info("N_list eval mode: N_gap values=%s  samples_per_cell=%d", N_list, args.samples_per_cell)

    logger.info(
        "NIAH eval | model=%s | bypass_memory=%s | dtype=%s | attn=%s",
        args.model, args.bypass_memory, args.dtype, args.attn_impl,
    )
    logger.info(
        "Grid: context_lengths=%s  depths=%s  num_samples=%d  seed=%d",
        context_lengths, depths, args.num_samples, args.seed,
    )

    # ------------------------------------------------------------------ #
    # Tokenizer
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id: int = tokenizer.eos_token_id

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    logger.info("Loading Llama model in %s with attn=%s ...", args.dtype, args.attn_impl)
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
    ).to(device)

    # ------------------------------------------------------------------ #
    # Patch every decoder layer with MemorySpaceLayer
    # ------------------------------------------------------------------ #
    ms_cfg = MemorySpaceConfig(
        num_slots=args.num_slots,
        top_k=args.top_k,
        selector_dim=args.selector_dim,
        writeback_gate_warmup_steps=args.writeback_warmup_steps,
        writeback_gate_max=args.writeback_gate_max,
        load_balance_weight=args.load_balance_weight,
        slot_init=args.slot_init,
        slot_init_noise=args.slot_init_noise,
        enable_writeback=True,
        return_aux_losses=True,
        hidden_to_slot_frozen=not args.unfreeze_hidden_to_slot,
        shared_memory_bank=args.shared_memory_bank,
        swa_window=args.swa_window,
    )
    # Patch underlying LlamaModel.layers (HF's LlamaForCausalLM wraps it under .model)
    apply_mem_space_to_model(model, ms_cfg, layer_indices=None)

    # ------------------------------------------------------------------ #
    # H7 fix v2 (2026-04-26 23:30): snapshot rotary inv_freq in fp32 BEFORE
    # the lossy `.to(dtype=bf16)` cast. The v1 approach (upcast after cast)
    # did NOT work: bf16 rounding of inv_freq is destructive, upcasting a
    # rounded tensor cannot recover mantissa bits. Direct evidence:
    #   inv_freq[1] = 0.81640625  (bf16-rounded)
    #   true fp32   = 0.81225...
    # At pos=1023, angle error ≈ 1023 × (0.81640625 - 0.81225) ≈ 4.25 rad
    # → cos drift up to ±2, matching the observed 1.578 absmax.
    # HF deliberately keeps inv_freq / original_inv_freq in fp32 (see
    # modeling_llama.LlamaRotaryEmbedding.__init__ and
    # modeling_rope_utils.dynamic_rope_update); blanket `.to(dtype=...)`
    # recurses into buffers and destroys that invariant.
    # Evidence: tests/test_wrapper_internal_parity.py H7 probe,
    #           scripts/probe_branch3_bypass_parity.py §5.4 decision tree,
    #           ops/research_notes/20260426_branch3_A2_pollution_debug.md §5.4.
    # ------------------------------------------------------------------ #
    _rope_snapshot: dict[str, torch.Tensor] = {}
    try:
        _rot = model.model.rotary_emb
        for _name in ("inv_freq", "original_inv_freq"):
            if hasattr(_rot, _name):
                _rope_snapshot[_name] = getattr(_rot, _name).detach().to(torch.float32).clone()
    except AttributeError:
        pass

    # Mem-space modules were freshly constructed on CPU in fp32 — align them
    # with the backbone (device + dtype) so everything runs on one device.
    model.to(device=device, dtype=dtype)

    # Restore the rotary buffers to fp32 on the current device.
    try:
        _rot = model.model.rotary_emb
        for _name, _buf in _rope_snapshot.items():
            _buf = _buf.to(device=device, dtype=torch.float32)
            # Re-register so PyTorch's buffer bookkeeping stays consistent.
            _rot._buffers[_name] = _buf
            setattr(_rot, _name, _buf)
        if _rope_snapshot:
            logger.info(
                "H7 fix v2 applied: restored rotary_emb buffers %s to float32",
                sorted(_rope_snapshot.keys()),
            )
    except AttributeError:
        logger.warning("H7 fix v2: rotary_emb not accessible on model.model — skipping")

    # ------------------------------------------------------------------ #
    # Load trained adapter checkpoint (if provided)
    # ------------------------------------------------------------------ #
    if args.checkpoint is not None:
        logger.info("Loading adapter checkpoint from %s ...", args.checkpoint)
        ckpt_state = torch.load(args.checkpoint, map_location=device)
        # NOTE (2026-04-27): checkpoint keys model.layers.* are already the
        # correct namespace for the patched model (MemorySpaceLayer wraps
        # Llama decoder layers at model.layers.*).  No remapping needed.
        missing_keys, unexpected_keys = model.load_state_dict(ckpt_state, strict=False)
        logger.info(
            "Checkpoint loaded: %d keys  |  missing=%d  unexpected=%d",
            len(ckpt_state), len(missing_keys), len(unexpected_keys),
        )
        if unexpected_keys:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected_keys[:10])
        if missing_keys:
            adapter_missing = [k for k in missing_keys if any(
                s in k for s in ("slot_output_gate", "gate_param", "Q_sel", "K_sel",
                                  "slot_to_hidden", "hidden_to_slot")
            )]
            if adapter_missing:
                logger.warning(
                    "Adapter keys NOT loaded (%d missing). First 5: %s",
                    len(adapter_missing), adapter_missing[:5]
                )
            else:
                logger.info(
                    "All adapter keys loaded OK; %d non-adapter (base-model) keys missing "
                    "(expected with strict=False)", len(missing_keys)
                )
    else:
        logger.warning(
            "No --checkpoint provided — using RANDOM adapter weights. "
            "Results are not meaningful for with-memory runs."
        )

    # ------------------------------------------------------------------ #
    # Optional bypass patching (ablation: no memory)
    # ------------------------------------------------------------------ #
    if args.bypass_memory:
        from src.memory.mem_space.layer import MemorySpaceLayer
        mem_layers = getattr(model, "_mem_space_layers", [])
        for w in mem_layers:
            w.forward = w.forward_no_memory.__get__(w, MemorySpaceLayer)
        logger.info("BYPASS MODE: patched %d layers → forward_no_memory", len(mem_layers))

    n_patched = len(getattr(model, "_mem_space_layers", []))
    logger.info(
        "Patched %d decoder layers | num_slots=%d top_k=%d "
        "shared_bank=%s bypass=%s",
        n_patched, args.num_slots, args.top_k,
        args.shared_memory_bank, args.bypass_memory,
    )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Fix J (2026-04-27): step_counter is not saved in the checkpoint.
    # At eval we need warmup_frac=1.0 (fully trained beta).
    # Set step_counter = writeback_gate_warmup_steps on every
    # MemorySpaceLayer so warmup_frac = min(steps/steps, 1.0) = 1.0.
    from src.memory.mem_space.layer import MemorySpaceLayer as _MSL
    _mem_layers = getattr(model, "_mem_space_layers", [])
    _warmup_target = args.writeback_warmup_steps if args.writeback_warmup_steps > 0 else 1
    for _w in _mem_layers:
        if isinstance(_w, _MSL):
            _w.step_counter = _warmup_target
    if _mem_layers:
        logger.info(
            "Fix J: set step_counter=%d on %d MemorySpaceLayer(s) → warmup_frac=1.0",
            _warmup_target, len(_mem_layers),
        )

    # ------------------------------------------------------------------ #
    # pg19 haystack data — memory-mapped, never fully loaded into RAM
    # ------------------------------------------------------------------ #
    logger.info("Loading pg19 chunks from %s (mmap) ...", args.data)
    pg19_data: np.ndarray = np.load(args.data, mmap_mode="r")  # [N, seq_len]
    logger.info(
        "pg19 shape: %s  dtype: %s  skip_pg19_chunks=%d",
        pg19_data.shape, pg19_data.dtype, args.skip_pg19_chunks,
    )

    # ------------------------------------------------------------------ #
    # Evaluation loop
    # ------------------------------------------------------------------ #

    # grid[str(context_len)][str(depth)] → {"correct": int, "total": int, "accuracy": float}
    grid: dict[str, dict[str, dict]] = {
        str(cl): {str(d): {"correct": 0, "total": 0} for d in depths}
        for cl in context_lengths
    }
    per_sample: List[dict] = []
    total_correct = 0
    total_done    = 0

    # ------------------------------------------------------------------ #
    # N_list mode (2026-04-27): chunk-gap grid instead of context_len×depth
    # ------------------------------------------------------------------ #
    grid_N_list: dict[str, dict] = {}

    if args.N_list is not None:
        # N_list mode: evaluate over a grid of N_gap values (number of
        # background chunks between needle insertion and query chunk).
        chunk_size_nl = args.chunk_size if args.chunk_size else args.seq_len
        N_values = [int(x.strip()) for x in args.N_list.split(",")]
        n_samples_nl = args.samples_per_cell

        logger.info("=" * 70)
        logger.info(
            "N_list mode: N_values=%s  samples_per_cell=%d  chunk_size=%d",
            N_values, n_samples_nl, chunk_size_nl,
        )
        logger.info("=" * 70)

        for N_gap in N_values:
            n_correct_N = 0
            logger.info("N_gap = %d", N_gap)

            for sample_idx in range(n_samples_nl):
                # 1. Generate needle ----------------------------------------
                needle_sentence, question_suffix, code = make_needle(rng)

                # 2. Build N_gap background chunks from pg19 ----------------
                needle_chunk_pos = N_gap // 2
                haystack_chunk_ids: list[list[int]] = []
                for ci in range(N_gap):
                    bg_tokens = build_haystack_ids(
                        pg19_data=pg19_data,
                        target_len=chunk_size_nl,
                        skip_chunks=args.skip_pg19_chunks + ci,
                    )
                    if ci == needle_chunk_pos:
                        needle_tokens_nl: list[int] = tokenizer.encode(
                            " " + needle_sentence, add_special_tokens=False
                        )
                        insert_at = max(0, chunk_size_nl // 2 - len(needle_tokens_nl) // 2)
                        chunk_w_needle = (
                            bg_tokens[:insert_at]
                            + needle_tokens_nl
                            + bg_tokens[insert_at + len(needle_tokens_nl):]
                        )[:chunk_size_nl]
                        haystack_chunk_ids.append(chunk_w_needle)
                    else:
                        haystack_chunk_ids.append(bg_tokens[:chunk_size_nl])

                # Build flat stream_ids (all N_gap chunks concatenated)
                stream_ids_nl: list[int] = []
                for ch in haystack_chunk_ids:
                    stream_ids_nl.extend(ch)

                # 3. Tokenize question ---------------------------------------
                question_ids_nl: list[int] = tokenizer.encode(
                    question_suffix, add_special_tokens=False
                )

                logger.info(
                    "  N_gap=%d sample=%d/%d  stream=%d question=%d  code=%s",
                    N_gap, sample_idx + 1, n_samples_nl,
                    len(stream_ids_nl), len(question_ids_nl), code,
                )

                try:
                    # 4a. Stream haystack (reset banks first) ----------------
                    _reset_banks(model)
                    stream_haystack(
                        model=model,
                        tokenizer=tokenizer,
                        haystack_ids=stream_ids_nl,
                        seq_len=chunk_size_nl,
                        device=device,
                    )

                    # 4b. F2 fix: replay last chunk before generation --------
                    if args.bypass_memory:
                        gen_input_ids_nl = stream_ids_nl + question_ids_nl
                    else:
                        last_chunk_replay_nl = (
                            stream_ids_nl[-chunk_size_nl:]
                            if len(stream_ids_nl) >= chunk_size_nl
                            else stream_ids_nl
                        )
                        gen_input_ids_nl = last_chunk_replay_nl + question_ids_nl

                    # 4c. Generate (banks frozen) ----------------------------
                    _freeze_banks(model)
                    try:
                        gen_ids_nl: list[int] = greedy_generate(
                            model=model,
                            question_ids=gen_input_ids_nl,
                            device=device,
                            max_new_tokens=args.max_new_tokens,
                            eos_id=eos_id,
                        )
                    finally:
                        _unfreeze_banks(model)
                    gen_text_nl: str = tokenizer.decode(gen_ids_nl, skip_special_tokens=True)

                    # 4d. Score ----------------------------------------------
                    hit_nl: bool = exact_match(gen_text_nl, code)
                    if hit_nl:
                        n_correct_N += 1
                        total_correct += 1
                    total_done += 1

                    status_nl = "\u2713" if hit_nl else "\u2717"
                    logger.info("    %s  expected=%s  generated=%r",
                                status_nl, code, gen_text_nl.strip())

                    per_sample.append({
                        "N_gap":       N_gap,
                        "sample_idx":  sample_idx,
                        "code":        code,
                        "generated":   gen_text_nl.strip(),
                        "correct":     hit_nl,
                    })

                except Exception as exc:
                    logger.error("    ERROR N_gap=%d sample=%d: %s", N_gap, sample_idx, exc)
                    traceback.print_exc()
                    total_done += 1
                    per_sample.append({
                        "N_gap":      N_gap,
                        "sample_idx": sample_idx,
                        "code":       code,
                        "generated":  f"ERROR: {exc}",
                        "correct":    False,
                        "error":      str(exc),
                    })

                torch.cuda.empty_cache()

            n_acc_N = n_correct_N / max(n_samples_nl, 1)
            grid_N_list[str(N_gap)] = {
                "correct":  n_correct_N,
                "total":    n_samples_nl,
                "accuracy": n_acc_N,
            }
            logger.info("  N_gap=%d  %d/%d = %.1f%%",
                        N_gap, n_correct_N, n_samples_nl, n_acc_N * 100)

        # Print N_list table
        print(f"\nNIAH N_list Results ({'bypass' if args.bypass_memory else 'with_memory'}):")
        print(f"{'N_gap':<10} {'correct':<10} {'total':<10} {'accuracy':<10}")
        print("-" * 42)
        for N_gap in N_values:
            cell = grid_N_list[str(N_gap)]
            print(f"{N_gap:<10} {cell['correct']:<10} {cell['total']:<10} {cell['accuracy']:.1%}")

        # Write CSV if requested
        if args.output_csv is not None:
            import csv
            with open(args.output_csv, "w", newline="") as csv_f:
                writer = csv.DictWriter(csv_f,
                                        fieldnames=["N_gap", "correct", "total", "accuracy"])
                writer.writeheader()
                for N_gap in N_values:
                    row = dict(grid_N_list[str(N_gap)])
                    row["N_gap"] = N_gap
                    writer.writerow(row)
            logger.info("Wrote CSV: %s", args.output_csv)

    else:
        # ------------------------------------------------------------------ #
        # Standard context_len × depth grid eval (original mode)
        # ------------------------------------------------------------------ #
        for context_len in context_lengths:
            logger.info("=" * 70)
            logger.info("context_len = %d tokens", context_len)
            logger.info("=" * 70)

            for depth in depths:
                depth_key = str(depth)
                cell_correct = 0

                for sample_idx in range(args.num_samples):

                    # 1. Generate random needle fact ----------------------------
                    needle_sentence, question_suffix, code = make_needle(rng)

                    # 2. Build haystack token IDs of context_len tokens --------
                    haystack_ids: list[int] = build_haystack_ids(
                        pg19_data=pg19_data,
                        target_len=context_len,
                        skip_chunks=args.skip_pg19_chunks,
                    )

                    # 3. Encode needle and insert at depth position ------------
                    #    Leading space improves tokenization at insertion boundary.
                    needle_tokens: list[int] = tokenizer.encode(
                        " " + needle_sentence, add_special_tokens=False
                    )
                    full_ids: list[int] = insert_needle(haystack_ids, needle_tokens, depth)
                    # Trim back to context_len + needle length (keep the needle intact,
                    # trim any surplus haystack tokens that would exceed our budget).
                    stream_ids: list[int] = full_ids[: context_len + len(needle_tokens)]

                    # 4. Tokenize question (NOT part of haystack / memory) -----
                    question_ids: list[int] = tokenizer.encode(
                        question_suffix, add_special_tokens=False
                    )

                    logger.info(
                        "  ctx=%d depth=%.2f sample=%d/%d  "
                        "stream=%d needle=%d question=%d  code=%s",
                        context_len, depth, sample_idx + 1, args.num_samples,
                        len(stream_ids), len(needle_tokens), len(question_ids), code,
                    )

                    try:
                        # 5a. Stream haystack through model (accumulate memory) -
                        _reset_banks(model)  # fresh start per sample
                        stream_haystack(
                            model=model,
                            tokenizer=tokenizer,
                            haystack_ids=stream_ids,
                            seq_len=args.seq_len,
                            device=device,
                        )

                        # 5b. Greedy generation (memory frozen — no writeback contaminates slots)
                        # Fix K (2026-04-27): In bypass mode there is no memory bank to carry
                        # haystack context across chunks, so we must feed the full context at
                        # generation time to make the bypass baseline semantically valid.
                        # Memory-enabled runs keep question_ids only (bank holds the context).
                        if args.bypass_memory:
                            gen_input_ids = stream_ids + question_ids
                        else:
                            # F2 fix (2026-04-27): replay last chunk before generation.
                            # Root cause of v8 0/60: at inference with only 12 question tokens,
                            # k=64 slots dominate (84%) vs 1.5% at training (T=4096).
                            # Fix: prepend the last seq_len tokens of the haystack so T≈seq_len
                            # during generation, restoring the training-time k/(k+T) ratio.
                            # Banks are frozen (no writes) so this doesn't corrupt accumulated state.
                            last_chunk_replay = stream_ids[-args.seq_len:] if len(stream_ids) >= args.seq_len else stream_ids
                            gen_input_ids = last_chunk_replay + question_ids
                        if args.bypass_memory:
                            logger.debug(
                                "  [bypass] gen_input_ids = stream_ids(%d) + question_ids(%d) = %d tokens",
                                len(stream_ids), len(question_ids), len(gen_input_ids),
                            )
                        _freeze_banks(model)
                        try:
                            gen_ids: list[int] = greedy_generate(
                                model=model,
                                question_ids=gen_input_ids,
                                device=device,
                                max_new_tokens=args.max_new_tokens,
                                eos_id=eos_id,
                            )
                        finally:
                            _unfreeze_banks(model)
                        gen_text: str = tokenizer.decode(gen_ids, skip_special_tokens=True)

                        # 6. Score ---------------------------------------------
                        hit: bool = exact_match(gen_text, code)
                        if hit:
                            cell_correct  += 1
                            total_correct += 1
                        total_done += 1

                        status = "\u2713" if hit else "\u2717"
                        logger.info(
                            "    %s  expected=%s  generated=%r",
                            status, code, gen_text.strip(),
                        )

                        per_sample.append({
                            "context_len":     context_len,
                            "depth":           depth,
                            "sample_idx":      sample_idx,
                            "needle_sentence": needle_sentence,
                            "code":            code,
                            "generated":       gen_text.strip(),
                            "correct":         hit,
                            "stream_tokens":   len(stream_ids),
                            "needle_tokens":   len(needle_tokens),
                            "question_tokens": len(question_ids),
                        })

                    except Exception as exc:
                        logger.error("    ERROR ctx=%d depth=%.2f sample=%d: %s",
                                     context_len, depth, sample_idx, exc)
                        traceback.print_exc()
                        per_sample.append({
                            "context_len": context_len,
                            "depth":       depth,
                            "sample_idx":  sample_idx,
                            "code":        code,
                            "generated":   f"ERROR: {exc}",
                            "correct":     False,
                            "error":       str(exc),
                        })
                        total_done += 1

                    torch.cuda.empty_cache()

                # --- record cell accuracy ----------------------------------- #
                cell_acc = cell_correct / max(args.num_samples, 1)
                grid[str(context_len)][depth_key] = {
                    "correct":  cell_correct,
                    "total":    args.num_samples,
                    "accuracy": cell_acc,
                }
                logger.info(
                    "  depth=%.2f  %d/%d = %.1f%%",
                    depth, cell_correct, args.num_samples, cell_acc * 100,
                )

    # ------------------------------------------------------------------ #
    # Overall accuracy
    # ------------------------------------------------------------------ #
    overall_accuracy: float = total_correct / max(total_done, 1)

    # Per-context-length accuracy (only meaningful in standard mode)
    per_ctx_accuracy: dict[str, float] = {}
    if args.N_list is None:
        for cl in context_lengths:
            ctx_correct = sum(grid[str(cl)][str(d)]["correct"] for d in depths)
            ctx_total   = sum(grid[str(cl)][str(d)]["total"]   for d in depths)
            per_ctx_accuracy[str(cl)] = ctx_correct / max(ctx_total, 1)

    # ------------------------------------------------------------------ #
    # Print results grid (standard mode only)
    # ------------------------------------------------------------------ #
    if args.N_list is None:
        mode_str = "bypass (no memory)" if args.bypass_memory else "with_memory=True"
        print(f"\nNIAH Results ({mode_str}):")

        col_w     = 12
        depth_hdr = "  ".join(f"depth={d:.2f}".center(col_w) for d in depths)
        print(f"{'context_len':<13}| {depth_hdr}  |   avg")
        print("-" * (15 + (col_w + 2) * len(depths) + 8))

        for cl in context_lengths:
            cells     = [grid[str(cl)][str(d)] for d in depths]
            cell_strs = "  ".join(f"{c['correct']}/{c['total']}".center(col_w) for c in cells)
            row_avg   = sum(c["accuracy"] for c in cells) / max(len(cells), 1)
            print(f"{cl:<13}| {cell_strs}  |  {row_avg:.1%}")

    print(f"\nOverall accuracy: {total_correct}/{total_done} = {overall_accuracy:.1%}")

    # ------------------------------------------------------------------ #
    # Save JSON
    # ------------------------------------------------------------------ #
    results = {
        "config": {
            "model":                   args.model,
            "data":                    args.data,
            "seq_len":                 args.seq_len,
            "swa_window":              args.swa_window,
            "num_slots":               args.num_slots,
            "top_k":                   args.top_k,
            "selector_dim":            args.selector_dim,
            "writeback_gate_max":      args.writeback_gate_max,
            "writeback_warmup_steps":  args.writeback_warmup_steps,
            "load_balance_weight":     args.load_balance_weight,
            "slot_init":               args.slot_init,
            "slot_init_noise":         args.slot_init_noise,
            "unfreeze_hidden_to_slot": args.unfreeze_hidden_to_slot,
            "shared_memory_bank":      args.shared_memory_bank,
            "dtype":                   args.dtype,
            "attn_impl":               args.attn_impl,
            "context_lengths":         context_lengths,
            "depths":                  depths,
            "n_list_mode":             n_list_mode,
            "N_list":                  N_list,
            "samples_per_cell":        args.samples_per_cell,
            "num_samples":             args.num_samples,
            "max_new_tokens":          args.max_new_tokens,
            "seed":                    args.seed,
            "skip_pg19_chunks":        args.skip_pg19_chunks,
        },
        "bypass_memory":      args.bypass_memory,
        "grid":               grid,
        "grid_N_list":        grid_N_list,
        "per_ctx_accuracy":   per_ctx_accuracy,
        "overall_accuracy":   overall_accuracy,
        "total_correct":      total_correct,
        "total_samples":      total_done,
        "per_sample":         per_sample,
    }

    out_json = os.path.join(args.output_dir, "niah_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", out_json)


if __name__ == "__main__":
    main()
