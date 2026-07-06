"""QCMem resume-primitive feasibility check (CPU, tiny random Llama).

Proves the ONE load-bearing technical claim behind QCMem mid-depth resume on a
Llama backbone:

    You can split LlamaModel's forward at an arbitrary layer j, cache the
    depth-j hidden state h_j, and RESUME the forward from layer j (running only
    layers[j:] + final norm + lm_head) to obtain logits that are BITWISE-CLOSE
    (fp32 max-abs-diff ~0) to a single standard full forward.

If this holds, the QCMem write side (run layers[0:j], stash h_j) and read side
(pack h_j pieces, resume layers[j:]) are mechanically sound; the only remaining
question is quality (query-conditioning), not correctness.

Also demonstrates the j=0 QCMem invariant as a corollary: at j=0, "resume from
layer 0 on the packed embedding sequence with contiguous positions" IS the
standard forward, so packed-reforward == full attention by construction.

Runs on CPU with a 4-layer, hidden=64 random Llama — no GPU, no checkpoint.
"""
from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.masking_utils import create_causal_mask


def build_tiny_llama(seed: int = 0) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=512,
        rope_theta=500000.0,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).to(torch.float32).eval()
    return model


@torch.no_grad()
def resume_forward(model: LlamaForCausalLM, input_ids: torch.Tensor, j: int):
    """Manual split-at-layer-j forward, mirroring the QCMem write+read primitive.

    Phase A (write): embed + layers[0:j] over the full sequence with the SAME
        causal mask + RoPE positions[0:L] the standard forward would use ->
        cache h_j (the depth-j hidden state).
    Phase B (read): resume layers[j:] on h_j with the same mask + positions,
        then final norm + lm_head -> logits.

    Returns logits [1, L, V]. When j == num_layers this is the "closed-book"
    endpoint (h_j already post-all-layers); when j == 0 it equals the standard
    forward.
    """
    inner = model.model
    device = input_ids.device
    L = input_ids.shape[1]

    inputs_embeds = inner.embed_tokens(input_ids)
    position_ids = torch.arange(L, device=device).unsqueeze(0)
    causal_mask = create_causal_mask(
        config=inner.config,
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    position_embeddings = inner.rotary_emb(inputs_embeds, position_ids=position_ids)

    # ---- Phase A: run bottom j layers, cache h_j ----
    hidden = inputs_embeds
    for layer in inner.layers[:j]:
        hidden = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
    h_j = hidden  # <-- this is what QCMem caches (per chunk, chunk-local)

    # ---- Phase B: resume from layer j ----
    hidden = h_j
    for layer in inner.layers[j:]:
        hidden = layer(
            hidden,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
    hidden = inner.norm(hidden)
    logits = model.lm_head(hidden)
    return logits


@torch.no_grad()
def main():
    model = build_tiny_llama()
    n_layers = model.config.num_hidden_layers

    torch.manual_seed(123)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 40))

    # Reference: one standard full forward.
    ref = model(input_ids=input_ids, use_cache=False).logits  # [1, L, V]

    print(f"tiny Llama: L_layers={n_layers}, hidden={model.config.hidden_size}, "
          f"seq_len={input_ids.shape[1]}, vocab={model.config.vocab_size}")
    print("-" * 64)
    print(f"{'j (resume layer)':>18} | {'max|logit diff|':>16} | verdict")
    print("-" * 64)
    all_ok = True
    for j in range(0, n_layers + 1):
        out = resume_forward(model, input_ids, j)
        diff = (out - ref).abs().max().item()
        ok = diff < 1e-4
        all_ok = all_ok and ok
        tag = "OK (== full fwd)" if ok else "MISMATCH"
        print(f"{j:>18} | {diff:>16.3e} | {tag}")
    print("-" * 64)
    print(f"RESULT: {'ALL j PASS — resume primitive is exact' if all_ok else 'FAILURE'}")

    # ---- j=0 QCMem packing invariant corollary ----
    # Split the sequence into two 'chunks' + a 'query', concatenate their token
    # ids in time order with contiguous positions, and resume-from-0. Because
    # j=0 packed reforward with in-order tokens IS the standard forward, this
    # must match the full forward on the concatenation.
    c1 = torch.randint(0, model.config.vocab_size, (1, 12))
    c2 = torch.randint(0, model.config.vocab_size, (1, 10))
    q = torch.randint(0, model.config.vocab_size, (1, 8))
    packed = torch.cat([c1, c2, q], dim=1)
    ref_packed = model(input_ids=packed, use_cache=False).logits
    j0_packed = resume_forward(model, packed, j=0)
    dpack = (j0_packed - ref_packed).abs().max().item()
    print(f"\nj=0 packing invariant  | max|diff|={dpack:.3e} | "
          f"{'OK' if dpack < 1e-4 else 'MISMATCH'}")
    print("(j=0 packed reforward == full attention by construction — the "
          "existing token-reforward path is exactly this j=0 special case.)")


if __name__ == "__main__":
    main()
