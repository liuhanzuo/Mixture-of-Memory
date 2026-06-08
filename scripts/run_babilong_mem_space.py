"""BABILong evaluation wrapper for the mem_space streaming memory architecture.

This is the mem_space counterpart to ``scripts/run_babilong_h6.py``.  It evaluates
a Llama-3-8B model patched with ``MemorySpaceLayer`` (the H7+/champion family) on
BABILong qa1-qa5 tasks across multiple context lengths (0k-32k).

mem_space is stateful:
    1.  Input is chunked into ``chunk_size`` segments.
    2.  Each chunk is run through ``model(input_ids=...)`` — the patched decoder
        layers prepend slot tokens and EMA-write the memory bank in-place.  The
        memory bank persists across chunks (no reset between them).
    3.  Each new BABILong sample resets the bank via ``_reset_banks(model)``.

Differences vs ``run_babilong_h6.py``:
    * mem_space uses the **HF LlamaForCausalLM forward signature** (the patched
      MemorySpaceLayer is transparent to the wrapping HF model).  We call
      ``model(input_ids=...)`` directly, not ``model.forward_chunk(...)``.
    * Reset is done via ``_reset_banks(model)`` (copied from
      ``eval_niah_mem_space.py``), which prefers the shared memory bank.
    * ``use_cache=False`` is mandatory because MemorySpaceLayer attention does
      not support HF's KV cache code path.
    * The adapter_config.json field names use the abbreviated form
      (``writeback_warmup_steps``, ``unfreeze_hidden_to_slot``); we translate
      them to the MemorySpaceConfig dataclass field names.

Usage:
    python scripts/run_babilong_mem_space.py \
        --model_path /path/to/Llama-3-8B \
        --checkpoint outputs/champion_ckpt/mem_space_adapter.pt \
        --adapter_config outputs/champion_ckpt/adapter_config.json \
        --output_name mem_space_champion \
        [--tasks qa1 qa2 ...] [--lengths 0k 1k ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add babilong to path — relative to repo root
_BABILONG_PKG = os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")
if os.path.isdir(_BABILONG_PKG) and _BABILONG_PKG not in sys.path:
    sys.path.insert(0, _BABILONG_PKG)

import datasets  # noqa: E402
from transformers import AutoTokenizer, LlamaForCausalLM  # noqa: E402

from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402

from src.memory.mem_space import MemorySpaceConfig, apply_mem_space_to_model  # noqa: E402


# --------------------------------------------------------------------------- #
# Memory helpers (copied verbatim from eval_niah_mem_space.py:82-101)
# --------------------------------------------------------------------------- #


def _reset_banks(model: torch.nn.Module) -> None:
    """Wipe per-sample slot state between BABILong samples.

    Under ``config.shared_memory_bank=True`` the patch exposes
    ``_mem_space_shared_bank`` on the root model; resetting that one object is
    equivalent to resetting every wrapper's bank (they all reference the same
    object).  Falls back to per-layer bank reset if no shared bank is present.

    Also clears L3 summary state (prev_chunk_h, chunk cache, legacy
    _current_summary) so each new sample starts cold.
    """
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.reset()
    else:
        mem_layers = getattr(root, "_mem_space_layers", None)
        if mem_layers:
            for w in mem_layers:
                w.memory_bank.reset()
    # Reset L3 summary state (cold start for new sample)
    l3_pool = getattr(root, "_l3_pool", None)
    if l3_pool is not None:
        l3_pool._current_summary = None
        l3_pool._prev_chunk_h = None
        if hasattr(l3_pool, "_chunk_summary_cache"):
            l3_pool._chunk_summary_cache = None
        # Batched-eval padding (2026-06-09): clear the previous-chunk token mask
        # consumed by the L3 pool. No-op for bsz=1 (mask never set).
        if hasattr(l3_pool, "_prev_chunk_token_mask"):
            l3_pool._prev_chunk_token_mask = None
        if hasattr(l3_pool, "_prev_summary"):
            l3_pool._prev_summary = None
    # Batched-eval padding: clear the per-chunk token mask stashed on the bank.
    if shared_bank is not None and hasattr(shared_bank, "_active_token_mask"):
        shared_bank._active_token_mask = None
    else:
        for w in getattr(root, "_mem_space_layers", []) or []:
            _b = getattr(w, "memory_bank", None)
            if _b is not None and hasattr(_b, "_active_token_mask"):
                _b._active_token_mask = None


def _reset_l2(model: torch.nn.Module) -> None:
    """Zero the L2 compressor's cross-chunk state (prev_latents).

    Called at every document boundary alongside ``_reset_banks``. No-op if the
    model was patched without ``use_l2``.
    """
    root = getattr(model, "module", model)
    comp = getattr(root, "_l2_compressor", None)
    if comp is not None:
        comp.reset()


def _freeze_banks(model: torch.nn.Module) -> None:
    """Freeze memory banks during greedy generation so writeback doesn't
    overwrite slots accumulated from the context."""
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.frozen = True
        return
    for w in getattr(root, "_mem_space_layers", []):
        w.memory_bank.frozen = True


def _unfreeze_banks(model: torch.nn.Module) -> None:
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.frozen = False
        return
    for w in getattr(root, "_mem_space_layers", []):
        w.memory_bank.frozen = False


# --------------------------------------------------------------------------- #
# Adapter config → MemorySpaceConfig translation
# --------------------------------------------------------------------------- #


# Map abbreviated field names found in adapter_config.json → MemorySpaceConfig
# fields. Anything not in this map and not a MemorySpaceConfig field is ignored.
_ADAPTER_CONFIG_FIELD_MAP = {
    "writeback_warmup_steps": "writeback_gate_warmup_steps",
}


def build_mem_space_config(adapter_cfg: dict) -> MemorySpaceConfig:
    """Construct a MemorySpaceConfig from an adapter_config.json dict.

    Handles two pieces of impedance mismatch:
      * `writeback_warmup_steps` (json) → `writeback_gate_warmup_steps` (dataclass)
      * `unfreeze_hidden_to_slot=True` (json) → `hidden_to_slot_frozen=False` (dataclass)

    Unknown keys (e.g. `max_train_steps`, `lr`) are silently dropped.
    """
    valid_fields = set(MemorySpaceConfig.__dataclass_fields__.keys())
    kwargs: dict = {}
    for k, v in adapter_cfg.items():
        # Rename if needed
        target = _ADAPTER_CONFIG_FIELD_MAP.get(k, k)
        if target == "unfreeze_hidden_to_slot":
            # Flip semantics: unfreeze=True means hidden_to_slot_frozen=False.
            kwargs["hidden_to_slot_frozen"] = not bool(v)
            continue
        if target in valid_fields:
            kwargs[target] = v
        # else: silently ignore (training-only keys like lr, max_train_steps)
    return MemorySpaceConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def load_mem_space_model(
    model_path: str,
    checkpoint_path: str,
    mem_config: MemorySpaceConfig,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    attn_impl: str = "sdpa",
):
    """Build base Llama + mem_space patch + load adapter ckpt.

    Mirrors eval_niah_mem_space.py:472-620:
      1. Load base LlamaForCausalLM in bfloat16.
      2. Snapshot rotary inv_freq in fp32 (H7 fix v2 pre-step).
      3. Apply mem_space patch to all decoder layers.
      4. .to(device, dtype) — moves everything (including freshly-built CPU/fp32
         mem_space modules) to the right place.
      5. Restore rotary buffers to fp32 (H7 fix v2 post-step).
      6. Load adapter checkpoint (strict=False; handle ddp `module.` prefix and
         common state-dict-wrapper layouts).
      7. Force step_counter = writeback_warmup_steps so warmup_frac=1.0 at eval
         (Fix J from eval_niah_mem_space.py).
    """
    print(f"[mem_space-BABILong] Loading base model from: {model_path}")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)

    # H7 fix v2 pre-step: snapshot rotary inv_freq in fp32 BEFORE any
    # `.to(dtype=bf16)` corrupts them. See eval_niah_mem_space.py:502-525.
    _rope_snapshot: dict = {}
    try:
        _rot = model.model.rotary_emb
        for _name in ("inv_freq", "original_inv_freq"):
            if hasattr(_rot, _name):
                _rope_snapshot[_name] = getattr(_rot, _name).detach().to(torch.float32).clone()
    except AttributeError:
        pass

    # Patch all decoder layers with MemorySpaceLayer
    print(f"[mem_space-BABILong] Applying mem_space patch (num_slots={mem_config.num_slots}, "
          f"top_k={mem_config.top_k}, shared_bank={mem_config.shared_memory_bank})")
    apply_mem_space_to_model(model, mem_config, layer_indices=None)

    # Move freshly-created mem_space modules to device/dtype
    model.to(device=device, dtype=dtype)

    # H7 fix v2 post-step: restore rotary buffers to fp32 on the right device.
    try:
        _rot = model.model.rotary_emb
        for _name, _buf in _rope_snapshot.items():
            _buf = _buf.to(device=device, dtype=torch.float32)
            _rot._buffers[_name] = _buf
            setattr(_rot, _name, _buf)
        if _rope_snapshot:
            print(f"[mem_space-BABILong] H7 fix v2 applied: restored rotary buffers "
                  f"{sorted(_rope_snapshot.keys())} to float32")
    except AttributeError:
        print("[mem_space-BABILong] WARNING: rotary_emb not accessible — skipping H7 fix")

    # Load checkpoint
    print(f"[mem_space-BABILong] Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Common state-dict layouts: raw OrderedDict / {model_state_dict: ...} / {state_dict: ...}.
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Assume the dict itself is the state_dict (this is what
            # eval_niah_mem_space.py:552 expects for the champion ckpt).
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Strip DDP "module." prefix if present.
    cleaned: dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[mem_space-BABILong] Loaded {len(cleaned)} keys | "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print(f"[mem_space-BABILong] WARNING: first 5 unexpected keys: {list(unexpected)[:5]}")
    # Adapter-specific missing keys are real failures; base-model missing keys
    # are expected with strict=False (the base weights came from from_pretrained).
    adapter_missing = [
        k for k in missing
        if any(s in k for s in (
            "slot_output_gate", "gate_param", "Q_sel", "K_sel",
            "slot_to_hidden", "hidden_to_slot",
        ))
    ]
    if adapter_missing:
        print(f"[mem_space-BABILong] WARNING: {len(adapter_missing)} adapter keys NOT "
              f"loaded — first 5: {adapter_missing[:5]}")

    # Fix J: force step_counter = warmup_steps so β/warmup_frac is fully ramped.
    from src.memory.mem_space.layer import MemorySpaceLayer as _MSL  # local import to avoid cycles
    _mem_layers = getattr(model, "_mem_space_layers", [])
    _warmup_target = mem_config.writeback_gate_warmup_steps if mem_config.writeback_gate_warmup_steps > 0 else 1
    for _w in _mem_layers:
        if isinstance(_w, _MSL):
            _w.step_counter = _warmup_target
    print(f"[mem_space-BABILong] Fix J: set step_counter={_warmup_target} on "
          f"{len(_mem_layers)} MemorySpaceLayer(s)")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# --------------------------------------------------------------------------- #
# Chunked generation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def generate_with_mem_space(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    chunk_size: int,
    max_new_tokens: int,
    device: torch.device,
    swa_eval_chunks: int = 0,
) -> str:
    """Streaming generation for a single BABILong sample.

    Strategy (mirrors stream_haystack + F2 "last-chunk replay" trick from
    eval_niah_mem_space.py:858-901):

      1. Reset memory banks (fresh state for this sample).
      2. Stream all-but-last chunks through ``model(input_ids=...)`` so the
         memory bank accumulates context (no return value needed; mem_space
         writes the bank in-place during forward).
      3. Freeze the bank, then autoregressively generate from the last chunk.
         The last chunk is consumed in the FIRST forward call (we read logits
         at its last position); subsequent steps append one token at a time.
      4. Unfreeze the bank (for cleanliness; doesn't matter for inference but
         keeps the contract).

    We do NOT do the F2 last-chunk replay: BABILong's question_suffix is already
    embedded at the END of the formatted input (after the haystack); the last
    chunk already contains the question text + the right context, so logit
    quality at its tail is what we want to read.

    eval-only cross-chunk SWA (``swa_eval_chunks`` = W, D2a, 2026-06-09):
        Default W=0 reproduces the original behaviour bit-for-bit — the
        generation window is exactly the last chunk, and only the memory bank
        carries information about the earlier chunks. When W>0, the generation
        window becomes the concatenation of the last (W+1) chunks, so the final
        forward's self-attention can attend DIRECTLY to the previous W chunks'
        raw KV (sliding window), *in addition to* the memory readback. The bank
        streaming loop is unchanged (still ``chunks[:-1]``), i.e. those W chunks
        remain in the bank too — SWA is purely additive direct attention. This
        tests whether the no-cross-chunk-SWA eval systematically under-estimates
        the model's true long-context ability. Note that the combined window
        gives those tokens correct *relative* RoPE positions (within the window)
        instead of each chunk restarting at position 0.

    Args:
        input_ids: [1, total_len] tensor on `device`.
        swa_eval_chunks: W >= 0. 0 = original (no cross-chunk SWA, default).

    Returns:
        Decoded text of `max_new_tokens` generated tokens (skip_special_tokens=True).
    """
    if device is None:
        device = next(model.parameters()).device

    _reset_banks(model)
    _reset_l2(model)

    tokens = input_ids[0]  # [total_len]
    chunks = list(tokens.split(chunk_size))

    # Stream all-but-last chunks (memory accumulation only — no logit reads).
    # NOTE: unchanged by SWA — the bank always accumulates chunks[:-1] exactly
    # as before, so W>0 only ADDS direct attention, it never removes context
    # from the bank.
    if len(chunks) > 1:
        for chunk in chunks[:-1]:
            chunk_tensor = chunk.unsqueeze(0).to(device)  # [1, <=chunk_size]
            _ = model(input_ids=chunk_tensor, use_cache=False)

    # Freeze the bank — generation should not pollute the slots that hold the context.
    _freeze_banks(model)
    try:
        if swa_eval_chunks > 0 and len(chunks) > 1:
            # Cross-chunk SWA window: last (W+1) chunks concatenated.
            start = max(0, len(chunks) - (swa_eval_chunks + 1))
            window = torch.cat(list(chunks[start:]), dim=0)  # [<= (W+1)*chunk_size]
            cur = window.unsqueeze(0).to(device)
        else:
            # W=0 (or single chunk): byte-identical to the original path.
            cur = chunks[-1].unsqueeze(0).to(device)  # [1, last_chunk_len]
        generated_ids: list[int] = []
        for step in range(max_new_tokens):
            outputs = model(input_ids=cur, use_cache=False)
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            if step == 0 and tokenizer.eos_token_id is not None:
                # Match H6 behaviour: suppress EOS as the very first generated
                # token so we don't return an empty answer.
                logits[:, tokenizer.eos_token_id] = float("-inf")
            next_tok = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
            tok_id = int(next_tok.item())
            if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id and step > 0:
                break
            generated_ids.append(tok_id)
            cur = torch.cat([cur, next_tok], dim=-1)
    finally:
        _unfreeze_banks(model)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Batched (cell-internal) generation — opt-in via --batch_size > 1
# --------------------------------------------------------------------------- #


def _set_active_token_mask(model, mask) -> None:
    """Stash the current chunk's [B, T] token mask (1=real, 0=pad) on the
    memory bank(s) so MemorySpaceLayer's selector pooling can exclude pads.
    ``mask=None`` clears it (full/streaming chunks)."""
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank._active_token_mask = mask
        return
    for w in getattr(root, "_mem_space_layers", []) or []:
        _b = getattr(w, "memory_bank", None)
        if _b is not None:
            _b._active_token_mask = mask


def _set_prev_chunk_token_mask(model, mask) -> None:
    """Stash the PREVIOUS chunk's [B, T] token mask on the L3 pool so the
    recursive L3 summary reduces over real tokens only. ``mask=None`` clears."""
    root = getattr(model, "module", model)
    l3_pool = getattr(root, "_l3_pool", None)
    if l3_pool is not None:
        l3_pool._prev_chunk_token_mask = mask


@torch.no_grad()
def generate_batch_with_mem_space(
    model,
    token_list,
    tokenizer,
    chunk_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[str]:
    """Batched streaming generation for several BABILong samples at once.

    All samples in ``token_list`` MUST share the same number of chunks
    (``ceil(len/chunk_size)``) and that number MUST be >= 2 — the caller
    (``main``) buckets samples by chunk-count and routes single-chunk samples
    to the bsz=1 path. Under that contract every "streaming" chunk
    (``chunks[:-1]``) is EXACTLY ``chunk_size`` long for every sample (because
    ``Tensor.split`` only shortens the final chunk), so the streaming forwards
    are unpadded and byte-identical to the bsz=1 path. ONLY the final
    generation chunk varies in length and is RIGHT-padded; right-padding is
    free under causal self-attention (real tokens never attend to trailing
    pads), so the wrapped decoder needs no mask change — only the two
    non-causal pooling reductions (selector routing + the recursive L3 summary)
    receive an explicit token mask so they ignore pad positions.

    Args:
        token_list: list of 1-D LongTensors (variable length), same chunk-count.

    Returns:
        list[str]: decoded answer for each input, in the same order.
    """
    B = len(token_list)
    if B == 0:
        return []
    if device is None:
        device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    eos_id = tokenizer.eos_token_id

    _reset_banks(model)
    _reset_l2(model)

    # Split each sample into chunks; verify the shared-chunk-count contract.
    per_sample_chunks = [list(t.split(chunk_size)) for t in token_list]
    n_chunks = len(per_sample_chunks[0])
    assert n_chunks >= 2, "batched path requires >=2 chunks; caller must bucket"
    for c in per_sample_chunks:
        assert len(c) == n_chunks, "batched samples must share chunk count"

    # ---- Stream all-but-last chunks (full chunk_size, unpadded) ----
    _set_active_token_mask(model, None)
    _set_prev_chunk_token_mask(model, None)
    for j in range(n_chunks - 1):
        stacked = torch.stack([per_sample_chunks[b][j] for b in range(B)], dim=0)
        assert stacked.shape[1] == chunk_size  # streaming chunks are always full
        stacked = stacked.to(device)
        _ = model(input_ids=stacked, use_cache=False)

    # ---- Build the (right-padded) generation chunk ----
    last_chunks = [per_sample_chunks[b][-1] for b in range(B)]
    last_lens = [int(c.shape[0]) for c in last_chunks]
    width = max(last_lens)
    cur = torch.full((B, width), pad_id, dtype=torch.long, device=device)
    for b in range(B):
        cur[b, : last_lens[b]] = last_chunks[b].to(device)
    cur_len = list(last_lens)                      # per-sample real length
    rows = torch.arange(B, device=device)

    def _mask_for(W: int) -> torch.Tensor:
        # [B, W] bool: True for positions < cur_len[b].
        ar = torch.arange(W, device=device).unsqueeze(0)       # [1, W]
        lens = torch.tensor(cur_len, device=device).unsqueeze(1)  # [B, 1]
        return ar < lens

    _freeze_banks(model)
    generated: list[list[int]] = [[] for _ in range(B)]
    finished = [False] * B
    prev_mask = None  # mask of the chunk currently held in l3_pool._prev_chunk_h
    try:
        for step in range(max_new_tokens):
            cur_mask = _mask_for(cur.shape[1])                 # [B, W]
            _set_active_token_mask(model, cur_mask)
            _set_prev_chunk_token_mask(model, prev_mask)

            outputs = model(input_ids=cur, use_cache=False)
            logits_all = outputs.logits                        # [B, W, V]
            # Read each row's logits at its OWN last real position.
            read_pos = torch.tensor(
                [cur_len[b] - 1 for b in range(B)], device=device
            )
            logits = logits_all[rows, read_pos, :]             # [B, V]
            if step == 0 and eos_id is not None:
                logits[:, eos_id] = float("-inf")
            next_tok = logits.argmax(dim=-1)                   # [B]

            # After this forward, l3_pool._prev_chunk_h == this cur's hidden;
            # remember the mask that matches it for the next step's L3 reduce.
            prev_mask = cur_mask

            # Place tokens / update lengths for unfinished samples.
            need_grow = False
            for b in range(B):
                if finished[b]:
                    continue
                tok = int(next_tok[b].item())
                if eos_id is not None and tok == eos_id and step > 0:
                    finished[b] = True
                    continue
                generated[b].append(tok)
                if cur_len[b] >= cur.shape[1]:
                    need_grow = True
            if all(finished):
                break
            # Grow the buffer by one pad column if any sample needs the slot.
            if need_grow:
                pad_col = torch.full((B, 1), pad_id, dtype=torch.long, device=device)
                cur = torch.cat([cur, pad_col], dim=1)
            for b in range(B):
                if finished[b]:
                    continue
                # Write the just-generated token at this sample's next position.
                cur[b, cur_len[b]] = generated[b][-1]
                cur_len[b] += 1
    finally:
        _unfreeze_banks(model)
        _set_active_token_mask(model, None)
        _set_prev_chunk_token_mask(model, None)

    return [
        tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated
    ]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="BABILong evaluation for mem_space architecture")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base Llama-3-8B model directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to mem_space adapter .pt checkpoint")
    parser.add_argument("--adapter_config", type=str, required=True,
                        help="Path to adapter_config.json describing the MemorySpaceConfig")
    parser.add_argument("--results_folder", type=str, default="./babilong_results",
                        help="Folder to store BABILong eval results")
    parser.add_argument("--output_name", type=str, required=True,
                        help="Subfolder name for this evaluation run")
    parser.add_argument("--dataset_name", type=str, default="RMT-team/babilong",
                        help="HuggingFace dataset name")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["qa1", "qa2", "qa5"],
                        help="BABILong tasks to evaluate")
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["0k", "1k", "2k", "4k", "8k", "16k"],
                        help="BABILong context lengths to evaluate")
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="Chunk size for memory accumulation (matches mem_space training seq_len)")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Maximum new tokens to generate per sample")
    parser.add_argument("--limit", type=int, default=100,
                        help="Maximum samples per task/length cell (default 100; -1 = all)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Cell-internal sample batch size. 1 (default) = the "
                             "original byte-for-byte per-sample path. >1 batches "
                             "same-chunk-count samples through a single forward "
                             "(~1.4x/cell); single-chunk samples always use the "
                             "bsz=1 path. NOTE: batching is numerically correct "
                             "(B=1 batched == bsz=1 exactly) but >1 under bf16 + "
                             "hard top-k routing + greedy decode does NOT preserve "
                             "the exact BABILong score (qa2/2k drifted 27->21 over "
                             "n=100). Use >1 only for fast triage, not final numbers.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")
    parser.add_argument("--swa_eval_chunks", type=int, default=0,
                        help="Eval-only cross-chunk sliding-window attention "
                             "(D2a). W=0 (default) = original behaviour, "
                             "bit-identical: the generation window is the last "
                             "chunk only and earlier chunks reach the final "
                             "forward solely via the memory bank. W>0 makes the "
                             "generation window the last (W+1) chunks "
                             "concatenated, so the final forward attends "
                             "DIRECTLY to the previous W chunks' raw KV (in "
                             "addition to memory readback). Bank streaming is "
                             "unchanged. Only supported on the bsz=1 path.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Wrap the formatted input in the tokenizer's chat template")
    parser.add_argument("--use_instruction", action="store_true", default=True,
                        help="Include BABILong DEFAULT_PROMPTS[task]['instruction']")
    parser.add_argument("--use_examples", action="store_true", default=True,
                        help="Include BABILong DEFAULT_PROMPTS[task]['examples']")
    parser.add_argument("--use_post_prompt", action="store_true", default=True,
                        help="Include BABILong DEFAULT_PROMPTS[task]['post_prompt']")
    args = parser.parse_args()

    if args.swa_eval_chunks < 0:
        parser.error("--swa_eval_chunks must be >= 0")
    if args.swa_eval_chunks > 0 and args.batch_size > 1:
        parser.error(
            "--swa_eval_chunks > 0 is only supported on the bsz=1 path "
            "(use --batch_size 1). The batched generation path does not "
            "implement the cross-chunk SWA window."
        )

    print(f"[mem_space-BABILong] Configuration:")
    print(f"  Base model:      {args.model_path}")
    print(f"  Checkpoint:      {args.checkpoint}")
    print(f"  Adapter config:  {args.adapter_config}")
    print(f"  Tasks:           {args.tasks}")
    print(f"  Lengths:         {args.lengths}")
    print(f"  Chunk size:      {args.chunk_size}")
    print(f"  Max new tokens:  {args.max_new_tokens}")
    print(f"  Limit/cell:      {args.limit}")
    print(f"  SWA eval chunks: {args.swa_eval_chunks}")
    print(f"  Device:          {args.device}")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[mem_space-BABILong] Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Load + parse adapter config
    with open(args.adapter_config, "r") as f:
        adapter_cfg = json.load(f)
    mem_config = build_mem_space_config(adapter_cfg)
    # L3 token-recon head builds pos_queries of shape [l3_recon_max_positions, d].
    # At train time this is set to chunk_size (train_mem_space_dolmino_cpt.py:1088),
    # but adapter_config.json carries no chunk_size, so the dataclass default (1024)
    # would mismatch a ckpt trained with a different chunk_size. Mirror training here.
    mem_config.l3_recon_max_positions = args.chunk_size
    print(f"[mem_space-BABILong] MemorySpaceConfig: num_slots={mem_config.num_slots}, "
          f"top_k={mem_config.top_k}, selector_dim={mem_config.selector_dim}, "
          f"warmup_steps={mem_config.writeback_gate_warmup_steps}, "
          f"slot_init={mem_config.slot_init}, "
          f"shared_bank={mem_config.shared_memory_bank}, "
          f"hidden_to_slot_frozen={mem_config.hidden_to_slot_frozen}")

    # Build + load model
    model = load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        mem_config=mem_config,
        device=device,
        dtype=dtype,
        attn_impl=args.attn_impl,
    )

    # ------------------------------------------------------------------ #
    # BABILong eval loop (mirrors run_babilong_h6.py:406-512)
    # ------------------------------------------------------------------ #
    for task in tqdm(args.tasks, desc="tasks"):
        if task not in DEFAULT_PROMPTS:
            print(f"[WARNING] Task {task} not in DEFAULT_PROMPTS, skipping.")
            continue

        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"] if args.use_instruction else "",
            "examples":    DEFAULT_PROMPTS[task]["examples"]    if args.use_examples    else "",
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"] if args.use_post_prompt else "",
            "template":    DEFAULT_TEMPLATE,
            "chat_template": args.use_chat_template,
            "system_prompt": "",
        }
        prompt_name = "_".join(
            [f"{k}_yes" if prompt_cfg[k] else f"{k}_no"
             for k in prompt_cfg if k != "template"]
        )

        for split_name in tqdm(args.lengths, desc="lengths", leave=False):
            print(f"\n[mem_space-BABILong] task={task}, length={split_name}")

            try:
                data = datasets.load_dataset(args.dataset_name, split_name)
                task_data = data[task]
            except Exception as e:
                print(f"[ERROR] Failed to load dataset {args.dataset_name}/{split_name}/{task}: {e}")
                continue

            outdir = Path(args.results_folder) / args.output_name
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"{task}_{split_name}_{prompt_name}.csv"
            cfg_file = outdir / f"{task}_{split_name}_{prompt_name}.json"

            json.dump(
                {
                    "prompt": prompt_cfg,
                    "generate_kwargs": {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False,
                        "num_beams": 1,
                    },
                    "model": {
                        "model_path":      args.model_path,
                        "checkpoint":      args.checkpoint,
                        "adapter_config":  args.adapter_config,
                        "chunk_size":      args.chunk_size,
                        "swa_eval_chunks": args.swa_eval_chunks,
                        "num_slots":       mem_config.num_slots,
                        "top_k":           mem_config.top_k,
                        "shared_memory_bank": mem_config.shared_memory_bank,
                    },
                },
                open(cfg_file, "w"),
                indent=4,
            )

            df = pd.DataFrame({"target": [], "output": [], "question": []})

            num_samples = len(task_data)
            if args.limit > 0:
                num_samples = min(num_samples, args.limit)

            def _encode_sample(idx):
                sample = task_data[idx]
                input_text = get_formatted_input(
                    sample["input"],
                    sample["question"],
                    prompt_cfg["examples"],
                    prompt_cfg["instruction"],
                    prompt_cfg["post_prompt"],
                    template=prompt_cfg["template"],
                )
                if args.use_chat_template:
                    messages = [{"role": "user", "content": input_text}]
                    input_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
                if isinstance(ids, list):
                    ids = torch.tensor([ids], dtype=torch.long)
                return sample["target"], sample["question"], ids

            if args.batch_size <= 1:
                # ---- bsz=1 path: byte-for-byte the original per-sample loop ----
                for idx in tqdm(range(num_samples), desc=f"{task}/{split_name}", leave=False):
                    target, question, input_ids = _encode_sample(idx)
                    input_ids = input_ids.to(device)
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        output = generate_with_mem_space(
                            model=model,
                            input_ids=input_ids,
                            tokenizer=tokenizer,
                            chunk_size=args.chunk_size,
                            max_new_tokens=args.max_new_tokens,
                            device=device,
                            swa_eval_chunks=args.swa_eval_chunks,
                        )
                    df.loc[len(df)] = [target, output, question]
                    if (idx + 1) % 10 == 0 or idx == num_samples - 1:
                        df.to_csv(outfile, index=False)
            else:
                # ---- batched path: bucket by chunk-count, then batch ----
                # Encode everything first so we can group by chunk count. Each
                # row keeps its original index so the CSV order is preserved.
                import math as _math
                rows = []  # (orig_idx, target, question, tokens_1d, n_chunks)
                for idx in range(num_samples):
                    target, question, input_ids = _encode_sample(idx)
                    toks = input_ids[0]
                    n_chunks = max(1, _math.ceil(toks.shape[0] / args.chunk_size))
                    rows.append((idx, target, question, toks, n_chunks))

                results: dict = {}  # orig_idx -> output text

                # Single-chunk samples: must use the bsz=1 cold-start path.
                singles = [r for r in rows if r[4] <= 1]
                multis = [r for r in rows if r[4] > 1]
                for (idx, target, question, toks, _nc) in tqdm(
                    singles, desc=f"{task}/{split_name}/single", leave=False
                ):
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        out = generate_with_mem_space(
                            model=model,
                            input_ids=toks.unsqueeze(0).to(device),
                            tokenizer=tokenizer,
                            chunk_size=args.chunk_size,
                            max_new_tokens=args.max_new_tokens,
                            device=device,
                            swa_eval_chunks=args.swa_eval_chunks,
                        )
                    results[idx] = out

                # Multi-chunk samples: group by exact chunk count, then split
                # into batches of <= batch_size.
                from collections import defaultdict as _dd
                by_nc = _dd(list)
                for r in multis:
                    by_nc[r[4]].append(r)
                for nc, group in by_nc.items():
                    # Sort the group by total token length so each <=batch_size
                    # slice has similar last-chunk lengths → minimal right-pad
                    # (less wasted compute, and padded rows stay numerically
                    # closer to the unpadded last chunk).
                    group = sorted(group, key=lambda r: int(r[3].shape[0]))
                    for s in tqdm(
                        range(0, len(group), args.batch_size),
                        desc=f"{task}/{split_name}/nc{nc}", leave=False,
                    ):
                        batch = group[s:s + args.batch_size]
                        tok_list = [b[3] for b in batch]
                        with torch.amp.autocast(device_type="cuda", dtype=dtype):
                            outs = generate_batch_with_mem_space(
                                model=model,
                                token_list=tok_list,
                                tokenizer=tokenizer,
                                chunk_size=args.chunk_size,
                                max_new_tokens=args.max_new_tokens,
                                device=device,
                            )
                        for b, o in zip(batch, outs):
                            results[b[0]] = o

                # Reassemble in original order.
                for (idx, target, question, _toks, _nc) in rows:
                    df.loc[len(df)] = [target, results[idx], question]
                df.to_csv(outfile, index=False)

            df.to_csv(outfile, index=False)
            print(f"[mem_space-BABILong] Saved {len(df)} results to {outfile}")

    print("\n[mem_space-BABILong] Evaluation complete!")


if __name__ == "__main__":
    main()
