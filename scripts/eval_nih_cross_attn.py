#!/usr/bin/env python3
"""Needle-In-Haystack evaluation for middle-layer memory (joint self-attention).

Tests cross-chunk information retrieval: a needle (random 7-digit code) is
placed in chunk A, and the model must recall it from a query in the final
chunk. Memory slots persist across chunks (no reset between chunks within
a document), so the model must store needle information in memory slots
during the needle chunk and retrieve it during the query chunk.

Architecture matches train_cross_attn_memory.py's _forward_middle_layer_memory:
  - Layers < memory_write_layer: vanilla forward (no slots)
  - Layer == memory_write_layer: concat slots + hidden_states -> self-attention -> update slots
  - Layers in memory_read_layers: concat slots + hidden_states -> self-attention (read-only)
  - All other layers: vanilla forward (no slots)

No separate cross-attention modules -- uses the base model's own decoder layers.

Compares two modes:
  --use_memory  : middle-layer memory active (write/read across chunks)
  --no_memory   : Vanilla forward pass (no memory, baseline)

Usage (single GPU):
  python scripts/eval_nih_cross_attn.py \
      --model meta-llama/Llama-3-8B \
      --checkpoint outputs/experiment_h_middle_layer/step_5000.pt \
      --use_memory \
      --middle_layer_memory \
      --num_chunks 4,8,16 \
      --needle_depths 0,-1 \
      --num_samples 20

Usage (8 GPU torchrun):
  torchrun --nproc_per_node 8 scripts/eval_nih_cross_attn.py ...
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CrossAttentionMemoryModel -- middle-layer joint self-attention memory
# Matches train_cross_attn_memory.py's _forward_middle_layer_memory exactly.
# --------------------------------------------------------------------------- #

class CrossAttentionMemoryModel(nn.Module):
    """Full model with middle-layer memory for eval.

    Uses the base model's own decoder layers (no separate cross-attention modules).
    At the write layer, slots are concatenated with hidden_states and processed
    together through self-attention. At read layers, the same pattern is used
    but slots are read-only (detached).
    """

    def __init__(
        self,
        base_model: LlamaForCausalLM,
        num_slots: int = 64,
        use_memory: bool = True,
        gradient_checkpointing: bool = False,
        middle_layer_memory: bool = True,
        memory_write_layer: int = 16,
        memory_read_layers: str = "18,22,26,30",
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.use_memory = use_memory
        self.middle_layer_memory = middle_layer_memory

        config = base_model.config
        self.num_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.d_model // self.num_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)

        self.base_model = base_model

        if gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()

        # Freeze base model params (eval only)
        for p in self.base_model.parameters():
            p.requires_grad = False

        self._decoder_layers: list[nn.Module] = list(self.base_model.model.layers)

        # Memory configuration -- matches training's middle_layer_memory mode
        self.memory_write_layer = memory_write_layer
        if isinstance(memory_read_layers, str):
            self.memory_read_layers = set(
                int(x.strip()) for x in memory_read_layers.split(",") if x.strip()
            )
        else:
            self.memory_read_layers = set(memory_read_layers)

        # Slot state: per-layer, runtime state (not nn.Parameters)
        # Only slot_values at memory_write_layer is actually used.
        self.slot_keys: list[torch.Tensor | None] = [None] * self.num_layers
        self.slot_values: list[torch.Tensor | None] = [None] * self.num_layers

        # Cache for extended attention mask (rebuilt if batch size or seq len changes)
        self._ext_attn_mask_cache = None

        # No cross_attn_modules -- middle-layer memory uses the base model's
        # own decoder layers directly. The checkpoint only has base model keys.

    def _get_embed_tokens(self):
        return self.base_model.model.embed_tokens

    def _get_lm_head(self):
        return self.base_model.lm_head

    def _get_norm(self):
        return self.base_model.model.norm

    def _get_rotary_emb(self):
        return self.base_model.model.rotary_emb

    def reset_slots(self) -> None:
        """Reset all memory slots for a new document."""
        for i in range(self.num_layers):
            self.slot_keys[i] = None
            self.slot_values[i] = None
        self._ext_attn_mask_cache = None

    def _init_slots(self, layer_idx: int, hidden_states: torch.Tensor) -> None:
        """Initialize slot values for a layer from hidden states (strided sampling).

        Preserves existing slots across chunks -- only initializes on first chunk.
        Matches train_cross_attn_memory.py's strided init for slot_forward mode.
        """
        B, T, D = hidden_states.shape

        if self.slot_values[layer_idx] is not None and self.slot_values[layer_idx].shape[0] == B:
            # Slots already initialized, preserve across chunks
            self.slot_values[layer_idx] = self.slot_values[layer_idx].detach()
            return

        # Strided sampling from current chunk (first chunk only)
        stride = max(1, T // self.num_slots)
        indices = torch.arange(0, T, stride)[: self.num_slots]
        if len(indices) < self.num_slots:
            pad_indices = indices[-1:].expand(self.num_slots - len(indices))
            indices = torch.cat([indices, pad_indices])
        sampled = hidden_states[:, indices, :].detach()
        noise = torch.randn_like(sampled) * 0.02
        self.slot_values[layer_idx] = (sampled + noise).clone()

    def _build_extended_attn_mask(self, S, T, dtype, device, batch_size):
        """Build [B, 1, S+T, S+T] additive attention mask.

        Matches train_cross_attn_memory.py exactly:
        - Slots (rows 0..S-1): attend to everything (all zeros)
        - Tokens (rows S..S+T-1): attend to all slots + causal mask on tokens
        """
        if (self._ext_attn_mask_cache is not None
                and self._ext_attn_mask_cache.shape[-1] == S + T
                and self._ext_attn_mask_cache.shape[0] == batch_size):
            return self._ext_attn_mask_cache

        L = S + T
        mask = torch.zeros(L, L, dtype=dtype, device=device)
        neg_inf = torch.finfo(dtype).min
        causal = torch.triu(
            torch.full((T, T), neg_inf, dtype=dtype, device=device),
            diagonal=1,
        )
        mask[S:, S:] = causal
        result = mask.view(1, 1, L, L).expand(batch_size, 1, L, L).contiguous()
        self._ext_attn_mask_cache = result
        return result

    def _extend_position_embeddings(self, position_embeddings, S):
        """Prepend S position-0 entries to RoPE cos/sin tables.

        Matches train_cross_attn_memory.py:
        Position 0: cos=1, sin=0 -> no rotation, slots are position-agnostic.
        """
        cos, sin = position_embeddings  # each [B, T, head_dim]
        cos0 = cos[:, :1, :]
        sin0 = sin[:, :1, :]
        cos_ext = torch.cat(
            [cos0.expand(cos.shape[0], S, cos.shape[-1]), cos], dim=1,
        )
        sin_ext = torch.cat(
            [sin0.expand(sin.shape[0], S, sin.shape[-1]), sin], dim=1,
        )
        return cos_ext, sin_ext

    @torch.no_grad()
    def forward_chunk(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward one chunk, return logits. Memory persists across calls.

        Matches train_cross_attn_memory.py's _forward_middle_layer_memory exactly:
        - Layers < memory_write_layer: vanilla forward
        - Layer == memory_write_layer: joint attention with slots, UPDATE slots
        - Layers in memory_read_layers: joint attention with slots (READ-ONLY)
        - All other layers: vanilla forward
        """
        B, T = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        if not self.use_memory:
            out = self.base_model(input_ids=input_ids)
            return out.logits

        S = self.num_slots
        embed_tokens = self._get_embed_tokens()
        hidden_states = embed_tokens(input_ids).to(dtype)

        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        rotary_emb = self._get_rotary_emb()
        position_embeddings = rotary_emb(hidden_states, position_ids)

        # Pre-compute extended position embeddings and attention mask for memory layers
        ext_pos_emb = self._extend_position_embeddings(position_embeddings, S)
        ext_attn_mask = self._build_extended_attn_mask(S, T, dtype, device, B)

        write_layer = self.memory_write_layer
        read_layers = self.memory_read_layers

        for layer_idx, layer in enumerate(self._decoder_layers):
            if self.middle_layer_memory and layer_idx == write_layer:
                # --- WRITE LAYER: init slots, joint attention, update slots ---
                self._init_slots(write_layer, hidden_states)
                slots = self.slot_values[write_layer]  # [B, S, d_model]

                extended = torch.cat([slots, hidden_states], dim=1)  # [B, S+T, d_model]

                layer_out = layer(
                    extended,
                    attention_mask=ext_attn_mask,
                    position_ids=None,
                    past_key_value=None,
                    use_cache=False,
                    position_embeddings=ext_pos_emb,
                )
                output = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                new_slots = output[:, :S, :]
                hidden_states = output[:, S:, :]

                # Update slot_values at write layer (detach for eval)
                self.slot_values[write_layer] = new_slots.detach()

            elif self.middle_layer_memory and layer_idx in read_layers:
                # --- READ LAYER: use write_layer's slots, joint attention, NO update ---
                slots = self.slot_values[write_layer]
                if slots is None:
                    # Fallback: vanilla forward if slots not yet initialized
                    layer_out = layer(
                        hidden_states,
                        attention_mask=None,
                        position_ids=None,
                        past_key_value=None,
                        use_cache=False,
                        position_embeddings=position_embeddings,
                    )
                    hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                else:
                    read_slots = slots.detach()
                    extended = torch.cat([read_slots, hidden_states], dim=1)  # [B, S+T, d_model]

                    layer_out = layer(
                        extended,
                        attention_mask=ext_attn_mask,
                        position_ids=None,
                        past_key_value=None,
                        use_cache=False,
                        position_embeddings=ext_pos_emb,
                    )
                    output = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                    # Discard slot outputs (read-only), keep only hidden_states
                    hidden_states = output[:, S:, :]

            else:
                # --- VANILLA LAYER: no memory interaction ---
                layer_out = layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        norm = self._get_norm()
        lm_head = self._get_lm_head()
        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states)
        return logits


# --------------------------------------------------------------------------- #
# Needle construction
# --------------------------------------------------------------------------- #

def make_needle(rng: random.Random, use_training_format: bool = False) -> tuple[str, str, str]:
    """Return (needle_sentence, query_prompt, code).

    needle_sentence: injectable into haystack at chosen depth.
    query_prompt:    appended after haystack context for the model to complete.
    code:            expected answer.
    """
    if use_training_format:
        name = "".join(rng.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=6))
        code = "".join(rng.choices("0123456789", k=5))
        needle_sentence = f"MEMORIZE: The secret code for agent {name} is {code}. END_MEMORIZE"
        query_prompt = f"The secret code for agent {name} is"
    else:
        code = "".join(rng.choices("0123456789", k=7))
        needle_sentence = f"The secret passcode for the vault is {code}. Do not forget this number."
        query_prompt = f"The secret passcode for the vault is"
    return needle_sentence, query_prompt, code


def build_haystack_tokens(
    pg19_data: np.ndarray,
    num_chunks: int,
    seq_len: int,
    skip_chunks: int,
) -> list[list[int]]:
    """Return list of num_chunks chunks, each a list of seq_len token IDs."""
    n_rows = len(pg19_data)
    chunks = []
    for i in range(num_chunks):
        row_idx = (skip_chunks + i) % n_rows
        row = pg19_data[row_idx]
        chunks.append([int(x) for x in row[:seq_len]])
    return chunks


def inject_needle_into_chunk(
    chunk_tokens: list[int],
    needle_tokens: list[int],
    position: str = "middle",
) -> list[int]:
    """Replace tokens at the chosen position within a chunk.

    Keeps chunk length exactly the same (seq_len).
    position: 'start', 'middle', 'end'.
    """
    n = len(needle_tokens)
    if position == "start":
        offset = 0
    elif position == "end":
        offset = len(chunk_tokens) - n
    else:
        offset = len(chunk_tokens) // 2 - n // 2
    offset = max(0, min(offset, len(chunk_tokens) - n))
    new_chunk = chunk_tokens[:offset] + needle_tokens + chunk_tokens[offset + n:]
    return new_chunk[:len(chunk_tokens)]


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def greedy_generate(
    model: CrossAttentionMemoryModel,
    prompt_ids: list[int],
    device: torch.device,
    max_new_tokens: int = 32,
    eos_id: int | None = None,
) -> list[int]:
    """Autoregressive greedy generation (no KV cache)."""
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    generated = []
    for _ in range(max_new_tokens):
        logits = model.forward_chunk(input_ids)
        next_id = logits[0, -1].argmax().item()
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device, dtype=torch.long)],
            dim=1,
        )
    return generated


# --------------------------------------------------------------------------- #
# Single NIAH evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_single(
    model: CrossAttentionMemoryModel,
    tokenizer,
    chunks: list[list[int]],
    needle_chunk_idx: int,
    needle_sentence: str,
    question_text: str,
    code: str,
    device: torch.device,
    seq_len: int,
    max_new_tokens: int = 32,
) -> dict:
    """Run one NIAH sample: inject needle, stream chunks, generate answer."""
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    needle_tokens = tokenizer.encode(" " + needle_sentence, add_special_tokens=False)
    question_tokens = tokenizer.encode(" " + question_text, add_special_tokens=False)

    chunks_copy = [list(c) for c in chunks]
    chunks_copy[needle_chunk_idx] = inject_needle_into_chunk(
        chunks_copy[needle_chunk_idx], needle_tokens, position="middle",
    )

    all_chunks = chunks_copy

    model.reset_slots()
    model.eval()

    for chunk_tokens in all_chunks:
        chunk_tensor = torch.tensor([chunk_tokens], device=device, dtype=torch.long)
        _ = model.forward_chunk(chunk_tensor)

    if not model.use_memory:
        all_stream = []
        for c in all_chunks:
            all_stream.extend(c)
        gen_prompt = all_stream[-seq_len:] if len(all_stream) > seq_len else all_stream
        gen_prompt = gen_prompt + question_tokens
        gen_prompt = gen_prompt[-seq_len:]
    else:
        last_chunk = chunks_copy[-1]
        tail_len = min(seq_len - len(question_tokens), len(last_chunk))
        gen_prompt = last_chunk[-tail_len:] + question_tokens

    gen_ids = greedy_generate(
        model, gen_prompt, device,
        max_new_tokens=max_new_tokens, eos_id=eos_id,
    )
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    hit = code in gen_text

    return {
        "generated": gen_text.strip(),
        "correct": hit,
        "code": code,
        "needle_chunk_idx": needle_chunk_idx,
        "num_chunks": len(chunks),
    }


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NIAH evaluation for middle-layer memory (joint self-attention)"
    )

    p.add_argument("--model", type=str, required=True,
                   help="Path to Llama-3-8B base model")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to cross-attn checkpoint (.pt)")
    p.add_argument("--data", type=str,
                   default="data/pg19_chunks_llama3.npy",
                   help="Path to pg19 tokenized chunks .npy")

    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--num_chunks", type=str, default="4,8,16",
                   help="Comma-separated list of total chunks (haystack+query)")
    p.add_argument("--needle_depths", type=str, default="0,-1",
                   help="Comma-separated needle chunk indices (0=first, -1=last before query)")
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_pg19_chunks", type=int, default=0)

    p.add_argument("--use_memory", action="store_true", default=True)
    p.add_argument("--no_memory", action="store_true", default=False)
    p.add_argument("--num_slots", type=int, default=64)

    # Middle-layer memory configuration (matches train_cross_attn_memory.py)
    p.add_argument("--middle_layer_memory", action="store_true", default=True,
                   help="Use middle-layer memory architecture (joint self-attention)")
    p.add_argument("--memory_write_layer", type=int, default=16,
                   help="Layer index where memory write occurs (default: 16)")
    p.add_argument("--memory_read_layers", type=str, default="18,22,26,30",
                   help="Comma-separated layer indices for memory read (default: 18,22,26,30)")

    p.add_argument("--output_dir", type=str,
                   default="outputs/nih_cross_attn")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--use_training_format", action="store_true", default=False,
                   help="Use training-format needles (MEMORIZE/agent/5-digit)")

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank != 0:
            sys.exit(0)
    else:
        local_rank = 0

    args = parse_args()
    if args.no_memory:
        args.use_memory = False

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    os.makedirs(args.output_dir, exist_ok=True)

    num_chunks_list = [int(x.strip()) for x in args.num_chunks.split(",")]
    needle_depths_raw = [int(x.strip()) for x in args.needle_depths.split(",")]

    logger.info(
        "NIAH MiddleLayerMemory eval | model=%s | use_memory=%s | num_slots=%d",
        args.model, args.use_memory, args.num_slots,
    )
    logger.info(
        "Memory config: middle_layer_memory=%s write_layer=%d read_layers=%s",
        args.middle_layer_memory, args.memory_write_layer, args.memory_read_layers,
    )
    logger.info(
        "Grid: num_chunks=%s  needle_depths=%s  num_samples=%d  seq_len=%d",
        num_chunks_list, needle_depths_raw, args.num_samples, args.seq_len,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model from %s ...", args.model)
    base_model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": device},
    )

    logger.info(
        "Building CrossAttentionMemoryModel (middle_layer_memory=%s, use_memory=%s) ...",
        args.middle_layer_memory, args.use_memory,
    )
    cm_model = CrossAttentionMemoryModel(
        base_model=base_model,
        num_slots=args.num_slots,
        use_memory=args.use_memory,
        gradient_checkpointing=False,
        middle_layer_memory=args.middle_layer_memory,
        memory_write_layer=args.memory_write_layer,
        memory_read_layers=args.memory_read_layers,
    ).to(device).to(dtype)

    # Verify model has no extra learnable parameters beyond the base model
    total_params = sum(p.numel() for p in cm_model.parameters())
    base_params = sum(p.numel() for p in base_model.parameters())
    logger.info("Model params: total=%d  base=%d  extra=%d",
                total_params, base_params, total_params - base_params)
    assert total_params == base_params, (
        f"Model has {total_params - base_params} extra parameters beyond base model. "
        "This should not happen with middle-layer memory (no separate cross-attention modules)."
    )

    if args.checkpoint:
        logger.info("Loading checkpoint from %s ...", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        missing, unexpected = cm_model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Checkpoint loaded: %d keys | missing=%d unexpected=%d",
            len(state_dict), len(missing), len(unexpected),
        )
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected[:10])
        if missing:
            # missing keys are fine -- they're the non-existent cross_attn_modules
            # that the old eval model had. With middle-layer memory, there are none.
            logger.info(
                "Missing keys (%d) are expected (no separate memory modules in middle-layer mode): %s",
                len(missing), missing[:5],
            )
        # Verify ALL checkpoint keys loaded
        matched_keys = len(state_dict) - len(unexpected)
        logger.info("Checkpoint key match: %d/%d loaded successfully", matched_keys, len(state_dict))
    else:
        if args.use_memory:
            logger.warning(
                "No --checkpoint provided with --use_memory. "
                "Base model weights are pretrained but no memory-specific training has been applied. "
                "This tests the architecture correctness with pretrained weights only."
            )

    cm_model.eval()

    logger.info("Loading haystack data from %s ...", args.data)
    pg19_data = np.load(args.data, mmap_mode="r")
    logger.info("Haystack data: shape=%s dtype=%s", pg19_data.shape, pg19_data.dtype)

    results = []
    t0 = time.time()

    for num_chunks in num_chunks_list:
        actual_depths = []
        for d in needle_depths_raw:
            if d < 0:
                actual_d = max(0, num_chunks + d - 1)
            else:
                actual_d = min(d, num_chunks - 1)
            actual_depths.append(actual_d)

        for needle_chunk_idx in actual_depths:
            n_correct = 0

            for sample_idx in range(args.num_samples):
                needle_sentence, question_text, code = make_needle(rng, args.use_training_format)

                skip = args.skip_pg19_chunks + sample_idx * num_chunks
                chunks = build_haystack_tokens(
                    pg19_data, num_chunks, args.seq_len, skip,
                )

                try:
                    result = evaluate_single(
                        model=cm_model,
                        tokenizer=tokenizer,
                        chunks=chunks,
                        needle_chunk_idx=needle_chunk_idx,
                        needle_sentence=needle_sentence,
                        question_text=question_text,
                        code=code,
                        device=device,
                        seq_len=args.seq_len,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as exc:
                    logger.error(
                        "ERROR chunks=%d depth=%d sample=%d: %s",
                        num_chunks, needle_chunk_idx, sample_idx, exc,
                    )
                    traceback.print_exc()
                    result = {
                        "generated": f"ERROR: {exc}",
                        "correct": False,
                        "code": code,
                        "needle_chunk_idx": needle_chunk_idx,
                        "num_chunks": num_chunks,
                    }

                if result["correct"]:
                    n_correct += 1

                status = "OK" if result["correct"] else "MISS"
                logger.info(
                    "[%s] chunks=%d depth=%d sample=%d/%d code=%s gen=%r",
                    status, num_chunks, needle_chunk_idx,
                    sample_idx + 1, args.num_samples, code,
                    result["generated"][:80],
                )

                results.append(result)
                torch.cuda.empty_cache()

            acc = n_correct / max(args.num_samples, 1)
            logger.info(
                "RESULT chunks=%d depth=%d accuracy=%.4f (%d/%d)",
                num_chunks, needle_chunk_idx, acc, n_correct, args.num_samples,
            )

    # ------------------------------------------------------------------ #
    # Aggregate and save
    # ------------------------------------------------------------------ #
    grid = {}
    for r in results:
        key = (r["num_chunks"], r["needle_chunk_idx"])
        if key not in grid:
            grid[key] = {"correct": 0, "total": 0}
        grid[key]["total"] += 1
        if r["correct"]:
            grid[key]["correct"] += 1

    summary_rows = []
    for (nc, depth), cell in sorted(grid.items()):
        acc = cell["correct"] / max(cell["total"], 1)
        summary_rows.append({
            "num_chunks": nc,
            "needle_depth": depth,
            "memory_acc" if args.use_memory else "vanilla_acc": acc,
            "correct": cell["correct"],
            "total": cell["total"],
        })

    overall_correct = sum(cell["correct"] for cell in grid.values())
    overall_total = sum(cell["total"] for cell in grid.values())
    overall_acc = overall_correct / max(overall_total, 1)

    print(f"\nNIAH Results ({'with_memory' if args.use_memory else 'vanilla'}):")
    print(f"{'chunks':<10} {'depth':<10} {'acc':<10} {'correct':<10}")
    print("-" * 40)
    for row in summary_rows:
        acc_key = "memory_acc" if args.use_memory else "vanilla_acc"
        print(f"{row['num_chunks']:<10} {row['needle_depth']:<10} "
              f"{row[acc_key]:.4f}    {row['correct']}/{row['total']}")
    print(f"\nOverall: {overall_correct}/{overall_total} = {overall_acc:.4f}")

    output = {
        "config": {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "use_memory": args.use_memory,
            "num_slots": args.num_slots,
            "middle_layer_memory": args.middle_layer_memory,
            "memory_write_layer": args.memory_write_layer,
            "memory_read_layers": args.memory_read_layers,
            "seq_len": args.seq_len,
            "num_chunks_list": num_chunks_list,
            "needle_depths": needle_depths_raw,
            "num_samples": args.num_samples,
            "dtype": args.dtype,
            "seed": args.seed,
            "use_training_format": args.use_training_format,
        },
        "results": summary_rows,
        "per_sample": results,
        "overall_accuracy": overall_acc,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "summary": (
            f"{'Memory' if args.use_memory else 'Vanilla'} mode: "
            f"overall accuracy {overall_acc:.4f} ({overall_correct}/{overall_total})"
        ),
        "elapsed_s": time.time() - t0,
    }

    out_path = os.path.join(args.output_dir, "niah_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
