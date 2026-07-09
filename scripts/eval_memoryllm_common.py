#!/usr/bin/env python
"""Shared MemoryLLM plumbing for the RULER / LongEval eval drivers.

MemoryLLM (YuWangX/memoryllm-8b-chat) is a *stateful* long-context memory model:
a long context is streamed chunk-by-chunk into a fixed-size memory pool
(``model.memory`` — [L, num_blocks*num_tokens, d]) via ``inject_memory``, and
generation is then done from the (short) question prompt alone, with every
decoder layer attending over the injected memory pool. On overflow the pool
drops old blocks (FIFO-style), so unlike QCMem's retrieval-over-chunks it has a
*fixed* capacity and can lose early needles at long lengths — precisely the
contrast we want to publish against QCMem's fixed-read retrieval.

This module factors out the model side that the RULER and LongEval drivers share
(loading, per-sample memory reset, context injection, and greedy generation), so
each driver only has to add its own task construction + scoring (reused verbatim
from the existing RULER / LongEval task frameworks).

The inject/reset/generate recipe is copied faithfully from
``scripts/run_babilong_memoryllm.py`` (the existing, verified MemoryLLM BABILong
driver). TWO deliberate adaptations vs. that file, both node-forced and
documented here:

  1. MODEL SOURCE. ``run_babilong_memoryllm.py`` imports ``modeling_memoryllm``
     from ``../MemoryLLM-source`` and runs under the pinned transformers-4.43
     ``external/memoryllm_venv``. Neither exists on this wzc1 L20A/B200 node
     (``../MemoryLLM-source`` holds only the safetensors + config, no modeling
     file; there is no memoryllm_venv). The node-native, verified path is the
     PORTED package ``src/memory/memoryllm_ported`` (adapted to transformers
     5.5.4 / torch 2.10 / sm_100), which ``scripts/smoke_memoryllm_port.py``
     already exercises end-to-end (load + inject + generate) under ``.venv``.
     We therefore load ``MemoryLLM`` from the port. The weights + tokenizer come
     from the same on-disk snapshot (``MemoryLLM-source``); the port's
     ``from_pretrained`` force-reloads the checkpoint tensors and recomputes the
     RoPE inv_freq buffers (see the ``# PORT:`` notes in the port).

  2. GENERATION LOOP. Under transformers 5.5.4 the ported model's
     ``model.generate()`` hits a prefill-plumbing gap
     (``prepare_inputs_for_generation`` is called with ``cache_position=None``),
     unrelated to the memory mechanism. As in the smoke test we therefore use a
     manual no-cache greedy loop of full forwards: each step re-reads the
     injected memory pool and appends the argmax token. O(n^2) in the (short)
     generation length, correct, and identical in output口径 to greedy decoding.
"""
from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer  # noqa: E402

# Node-native, verified MemoryLLM path (transformers 5.5.4 / torch 2.10 / sm_100).
from src.memory.memoryllm_ported import MemoryLLM  # noqa: E402

# On-disk snapshot of YuWangX/memoryllm-8b-chat (config + tokenizer + safetensors).
# This is the weights directory referenced by scripts/smoke_memoryllm_port.py.
DEFAULT_MEMORYLLM_PATH = (
    "/apdcephfs_wzc1/share_304376610/pighzliu_code/MemoryLLM-source"
)


# --------------------------------------------------------------------------- #
# Load + memory snapshot / reset (copied from run_babilong_memoryllm.py)
# --------------------------------------------------------------------------- #
def load_memoryllm(model_path: str, device, dtype, attn_impl: str = "sdpa"):
    """Load the ported MemoryLLM + tokenizer and snapshot the CLEAN memory pool.

    Returns (model, tokenizer, initial_state) where ``initial_state`` holds a
    detached clone of ``model.memory`` and ``model.initialized`` — the exact
    clean checkpoint state that must be restored before every sample so sample
    N+1 never leaks the injected context of sample N.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = MemoryLLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            local_files_only=True,
        )
    except (ValueError, ImportError) as e:
        print(f"[MemoryLLM] attn_impl={attn_impl} failed ({e}); falling back to sdpa")
        model = MemoryLLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        )
    model = model.to(device).eval()
    model.config.use_cache = False

    if not hasattr(model, "memory"):
        raise RuntimeError(
            "MemoryLLM model has no .memory attribute; cannot guarantee "
            "per-sample isolation")

    initial_state = {
        "memory": model.memory.detach().clone(),
        "initialized": (model.initialized.detach().clone()
                        if hasattr(model, "initialized") else None),
    }
    L = int(model.config.num_hidden_layers)
    print(f"[MemoryLLM] loaded from {model_path}: "
          f"memory_pool={tuple(model.memory.shape)} "
          f"(L={L}, num_blocks={getattr(model, 'num_blocks', '?')}, "
          f"num_tokens={getattr(model, 'num_tokens', '?')}), "
          f"initialized={int(initial_state['initialized'].item()) if initial_state['initialized'] is not None else 'NA'}")
    return model, tokenizer, initial_state


@torch.no_grad()
def reset_memory(model, initial_state: dict, verify: bool = False) -> None:
    """Restore the exact clean checkpoint memory before a sample."""
    model.memory.copy_(initial_state["memory"])
    if initial_state.get("initialized") is not None and hasattr(model, "initialized"):
        model.initialized.copy_(initial_state["initialized"])
    if verify:
        if not torch.equal(model.memory, initial_state["memory"]):
            raise RuntimeError("Memory reset verification failed: model.memory differs from clean snapshot")
        if initial_state.get("initialized") is not None and hasattr(model, "initialized"):
            if int(model.initialized.item()) != int(initial_state["initialized"].item()):
                raise RuntimeError("Memory reset verification failed: model.initialized differs from clean snapshot")


# --------------------------------------------------------------------------- #
# Context injection (copied from run_babilong_memoryllm.inject_long_context)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def inject_context(model, tokenizer, context: str, device, chunk_size: int = 1024) -> int:
    """Stream ``context`` into the memory pool ``chunk_size`` tokens at a time.

    Returns the number of chunks actually injected. Chunks shorter than 16
    tokens are skipped (MemoryLLM's injection hard minimum, per its README and
    the BABILong driver). Empty / whitespace-only / <16-token contexts inject
    nothing (the clean pool is used as-is)."""
    if not context or not context.strip():
        return 0
    ids = tokenizer(context, return_tensors="pt",
                    add_special_tokens=False).input_ids.to(device)
    total_len = ids.shape[1]
    if total_len < 16:
        return 0
    n_chunks = 0
    for pos in range(0, total_len, chunk_size):
        chunk = ids[:, pos:pos + chunk_size]
        if chunk.shape[1] >= 16:
            model.inject_memory(chunk, update_memory=True)
            n_chunks += 1
    return n_chunks


# --------------------------------------------------------------------------- #
# Greedy generation from the injected memory (manual no-cache loop, tf5-safe)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_answer(model, tokenizer, question_prompt: str, device,
                    max_new_tokens: int = 48, use_chat_template: bool = True) -> str:
    """Greedy-decode the answer from ``question_prompt`` with the memory pool
    injected. Uses a manual no-cache loop (see module docstring for why
    ``model.generate`` is avoided under transformers 5.5.4).

    ``use_chat_template=True`` mirrors run_babilong_memoryllm.generate_answer:
    the question is wrapped in the Llama-3 chat template and the tokenizer's
    leading BOS is dropped (the model adds its own learned bos_embedding at every
    layer, so a second textual BOS would double-anchor position 0)."""
    if use_chat_template and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": question_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, return_tensors="pt",
            add_generation_prompt=True)
        if isinstance(inputs, list):
            inputs = torch.tensor([inputs], dtype=torch.long)
        if hasattr(inputs, "input_ids"):
            inputs = inputs.input_ids
        inputs = inputs[:, 1:].to(device)  # drop tokenizer BOS
    else:
        inputs = tokenizer(question_prompt, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(device)

    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    terminators = {t for t in (tokenizer.eos_token_id, eot)
                   if t is not None and t >= 0}

    cur = inputs
    gen_ids: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(input_ids=cur, return_dict=True).logits[0, -1]
        nxt = int(logits.float().argmax().item())
        if nxt in terminators:
            break
        gen_ids.append(nxt)
        cur = torch.cat(
            [cur, torch.tensor([[nxt]], device=device, dtype=cur.dtype)], dim=1)
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
