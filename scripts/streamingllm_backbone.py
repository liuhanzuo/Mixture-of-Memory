#!/usr/bin/env python
"""Shared StreamingLLM (equal-budget, truncation-approx) backbone + forward for
the QA-benchmark StreamingLLM drivers (LoCoMo / LongBench / BABILong / LongEval).

This is the SAME truncation-approx mechanism proven in
``scripts/eval_ruler_streamingllm.py`` (RULER), factored out so every QA driver
shares one byte-identical forward:

    fed = concat( input_ids[:, :sink_size] , input_ids[:, -window_size:] )

then run the UNMODIFIED full-attention backbone greedily on ``fed``. StreamingLLM
(Xiao et al. 2023) keeps only the first ``sink_size`` tokens (the attention sink)
plus the most-recent ``window_size`` tokens and DROPS the middle, re-assigning
RoPE positions so the kept tokens occupy contiguous positions
``0 .. sink+window-1`` (no extrapolation). transformers 5.5.4 no longer ships
``SinkCache``, so instead of re-implementing the rotary/attention forward we use
the functionally-equivalent truncation: ``fed`` has length
``sink+window`` (~6657) < the model window, so the default contiguous
``position_ids`` place the sink at ``0..sink-1`` and the recent window at
``sink..sink+window-1`` — exactly StreamingLLM's position rolling. This
reproduces StreamingLLM's two defining properties: (i) constant peak
memory/compute regardless of nominal length (the model only ever sees ``budget``
tokens); (ii) the middle of the context is invisible (a needle in the dropped
middle is structurally unrecoverable). NOT bit-faithful (no token-by-token
streaming re-rotation) but functionally equivalent for a fixed-budget retrieval
eval, and it never touches attention.

Equal budget = sink 4 + window 6653 = 6657 tokens ~= CoMem constant read
(~16.9 GB), matching ``eval_ruler_streamingllm.py`` so the QA rows are
apples-to-apples with the RULER StreamingLLM row and the CoMem/InfLLM cohort.

Only the model forward differs from the QCMem/InfLLM QA drivers — the data,
prompt templates, chat-template protocol (chat ON + no-think), scoring, sharding
and on-disk layout are all reused verbatim by each driver so
StreamingLLM/CoMem/InfLLM are graded identically.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Equal-budget defaults (identical to scripts/eval_ruler_streamingllm.py).
DEFAULT_SINK = 4
DEFAULT_WINDOW = 6653  # sink + window = 6657 ~= CoMem constant read


def load_backbone(model_path, device, dtype, attn_impl="sdpa"):
    """Load a plain backbone (Qwen3-8B / Llama-3-8B) + tokenizer. No adapter, no
    memory patch — StreamingLLM is training-free and only truncates the input.

    ``local_files_only=True``: offline nodes otherwise treat a local dir path as
    an HF repo_id and error ("Repo id must be in the form ...")."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    return model, tokenizer


def im_end_ids(tokenizer):
    """Chat end tokens so decode stops at ``<|im_end|>`` (Qwen3 chat). Mirrors the
    ``_im_end_ids`` helper in every InfLLM QA driver."""
    ids = []
    try:
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(tid, int) and tid >= 0:
            ids.append(tid)
    except Exception:
        pass
    return ids


@torch.no_grad()
def streaming_generate(model, tokenizer, input_ids, sink_size, window_size,
                       max_new_tokens, device, extra_end_token_ids=None):
    """Keep the first ``sink_size`` tokens (attention sink) + the last
    ``window_size`` tokens (recent window), drop the middle, then greedily decode
    with the unmodified full-attention model. Returns (decoded_text, kept_len,
    orig_len).

    The chat prompt is fed already-templated (add_generation_prompt=True), so the
    recent window keeps the trailing question + assistant generation prefix while
    the sink keeps the leading system/user tokens — exactly the two anchors
    StreamingLLM relies on. min_new_tokens=1 masks EOS on the first step (matching
    the step-0 EOS suppression in the QCMem/InfLLM greedy loops) so a well-formed
    chat prompt never decodes to the empty string.
    """
    ids = input_ids
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    orig_len = int(ids.shape[1])
    budget = sink_size + window_size
    if orig_len > budget:
        sink = ids[:, :sink_size]
        recent = ids[:, -window_size:]
        ids = torch.cat([sink, recent], dim=1)
    ids = ids.to(device)
    kept_len = int(ids.shape[1])

    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    for t in (extra_end_token_ids or []):
        if int(t) not in eos_ids:
            eos_ids.append(int(t))

    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        min_new_tokens=1,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=(eos_ids or None),
    )
    gen = out[0, ids.shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text, kept_len, orig_len
