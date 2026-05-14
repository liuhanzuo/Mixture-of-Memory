#!/usr/bin/env python3
"""H-series v2 Phase 1: LM Pretrain on PG19 with frozen backbone + trainable memory.

Supports 3 memory architecture variants:
  A: Cross-attention slots (write at layer 8, read at 10/12/14) — adapted for 1B (16 layers)
  B: Joint attention (slots prepended, joint self-attention at every layer)
  D: Variant A + LoRA(r=8, alpha=32) on q_proj/v_proj

Variant C (ARMT) uses the ARMT run_finetuning_lm_rmt_hf.py directly.

Based on train_cross_attn_memory.py CrossAttentionMemoryModel architecture.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM

# Import our cross-attention memory module
from src.memory.mem_space.selector import CrossAttentionMemoryV2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [R%(rank)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# PG19 Dataset (from armt_pg19_real_tokenized_full)
# --------------------------------------------------------------------------- #

class PG19SegmentDataset(Dataset):
    """Loads the PG19 tokenized dataset and serves multi-segment samples.

    Each sample is (segment_size * max_n_segments) tokens from a book.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        segment_size: int = 512,
        max_n_segments: int = 2,
        seed: int = 42,
    ):
        import datasets as hf_datasets
        ds = hf_datasets.load_from_disk(dataset_path)
        self.data = ds[split]
        self.segment_size = segment_size
        self.max_n_segments = max_n_segments
        self.total_len = segment_size * max_n_segments
        self.rng = np.random.RandomState(seed)
        self.n_books = len(self.data)
        # Use a fixed virtual size (large enough for training)
        self._virtual_size = max(50000, self.n_books * 10)
        logger.info(
            "PG19SegmentDataset[%s]: %d books, virtual_size=%d (seg=%d, n_seg=%d)",
            split, self.n_books, self._virtual_size, segment_size, max_n_segments,
        )

    def __len__(self):
        return self._virtual_size

    def __getitem__(self, idx):
        # Random book, random offset (lazy — no pre-scanning)
        rng = np.random.RandomState(idx + 12345)
        for _ in range(10):  # retry if book too short
            book_idx = rng.randint(0, self.n_books)
            tokens = self.data[book_idx]["tokens"]
            book_len = len(tokens)
            if book_len >= self.total_len:
                max_offset = book_len - self.total_len
                offset = rng.randint(0, max_offset + 1)
                chunk = tokens[offset:offset + self.total_len]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                return {"input_ids": input_ids, "labels": input_ids.clone()}
        # Fallback: just take first total_len tokens from any book
        tokens = self.data[0]["tokens"][:self.total_len]
        if len(tokens) < self.total_len:
            tokens = tokens + [0] * (self.total_len - len(tokens))
        input_ids = torch.tensor(tokens, dtype=torch.long)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


# --------------------------------------------------------------------------- #
# V2 Memory Model
# --------------------------------------------------------------------------- #

class HSeriesV2Model(nn.Module):
    """Frozen backbone + trainable memory for H-series v2.

    Variant A: Cross-attention (write at write_layer, read at read_layers)
    Variant B: Joint attention (slots prepended at every layer)
    Variant D: Same as A + LoRA on q_proj/v_proj
    """

    def __init__(
        self,
        base_model: LlamaForCausalLM,
        memory_variant: str = "A",  # A, B, D
        num_slots: int = 64,
        segment_size: int = 512,
        max_n_segments: int = 2,
        freeze_backbone: bool = True,
        no_loss_from_first_segment: bool = True,
        # Cross-attn specific (A, D)
        memory_write_layer: int = 8,
        memory_read_layers: str = "10,12,14",
        write_lr: float = 0.1,
        residual_scale: float = 0.01,
        # LoRA specific (D)
        lora_r: int = 8,
        lora_alpha: int = 32,
        # Dual gate
        use_dual_gate: bool = True,
        forget_bias_init: float = 1.0,
        input_bias_init: float = 0.0,
    ):
        super().__init__()
        self.memory_variant = memory_variant
        self.num_slots = num_slots
        self.segment_size = segment_size
        self.max_n_segments = max_n_segments
        self.freeze_backbone = freeze_backbone
        self.no_loss_from_first_segment = no_loss_from_first_segment
        self.memory_write_layer = memory_write_layer
        self.residual_scale = residual_scale

        config = base_model.config
        self.num_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.head_dim = self.d_model // self.num_heads

        # Parse read layers
        if isinstance(memory_read_layers, str):
            self.memory_read_layers = set(int(x.strip()) for x in memory_read_layers.split(",") if x.strip())
        else:
            self.memory_read_layers = set(memory_read_layers)

        self.base_model = base_model
        self.base_model.gradient_checkpointing_enable()

        # Freeze backbone
        if freeze_backbone:
            for p in self.base_model.parameters():
                p.requires_grad = False

        # Apply LoRA for variant D
        if memory_variant == "D":
            self._apply_lora(lora_r, lora_alpha)

        self._decoder_layers = self._get_decoder_layers()

        # Memory slot state (runtime, not parameters)
        self.slot_values: list[Optional[torch.Tensor]] = [None] * self.num_layers

        # Build memory modules based on variant
        if memory_variant in ("A", "D"):
            # Cross-attention memory: write at one layer, read at select upper layers
            n_read_layers = len(self.memory_read_layers)
            self.cross_attn_modules = nn.ModuleList([
                CrossAttentionMemoryV2(
                    d_model=self.d_model,
                    n_heads=self.num_heads,
                    n_kv_heads=self.num_kv_heads,
                    num_slots=num_slots,
                    dropout=0.0,
                    write_lr=write_lr,
                    use_dual_gate=use_dual_gate,
                    forget_bias_init=forget_bias_init,
                    input_bias_init=input_bias_init,
                )
                for _ in range(n_read_layers)
            ])
            self._read_layer_to_ca_idx = {
                layer_idx: i for i, layer_idx in enumerate(sorted(self.memory_read_layers))
            }
            # Dual-gate writeback for write layer
            if use_dual_gate:
                d = self.d_model
                self.dual_gate_proj_new = nn.Linear(d, 2 * d, bias=False)
                self.dual_gate_proj_mem = nn.Linear(d, 2 * d, bias=False)
                bias_init = torch.cat([
                    torch.full((d,), float(input_bias_init)),
                    torch.full((d,), float(forget_bias_init)),
                ])
                self.dual_gate_bias = nn.Parameter(bias_init)
                nn.init.xavier_uniform_(self.dual_gate_proj_new.weight, gain=0.5)
                nn.init.xavier_uniform_(self.dual_gate_proj_mem.weight, gain=0.5)
            else:
                self.dual_gate_proj_new = None
                self.dual_gate_proj_mem = None
                self.dual_gate_bias = None

            # Learnable slot init
            self.slot_init_embed = nn.Parameter(
                torch.randn(num_slots, self.d_model) * 0.02
            )

        elif memory_variant == "B":
            # Joint attention: slots prepended at every layer
            self.slot_init_embed = nn.Parameter(
                torch.randn(num_slots, self.d_model) * 0.02
            )
            # No cross_attn_modules needed for joint attention
            self.cross_attn_modules = nn.ModuleList()
            self._read_layer_to_ca_idx = {}
            self.dual_gate_proj_new = None
            self.dual_gate_proj_mem = None
            self.dual_gate_bias = None

    def _apply_lora(self, r: int, alpha: int):
        """Apply LoRA to q_proj and v_proj of all attention layers."""
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        # Temporarily unfreeze for peft wrapping
        for p in self.base_model.parameters():
            p.requires_grad = True
        self.base_model = get_peft_model(self.base_model, lora_config)
        # Re-freeze non-LoRA params
        for name, p in self.base_model.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False

    def _get_inner_model(self):
        """Navigate through peft/HF wrapper to get the actual LlamaModel."""
        m = self.base_model
        # PeftModelForCausalLM: .model -> LlamaForCausalLM (with LoRA injected)
        if hasattr(m, 'model') and hasattr(m.model, 'model') and hasattr(m.model.model, 'layers'):
            return m.model.model
        # LlamaForCausalLM: .model -> LlamaModel
        if hasattr(m, 'model') and hasattr(m.model, 'layers'):
            return m.model
        # Already LlamaModel
        if hasattr(m, 'layers'):
            return m
        raise RuntimeError(f"Cannot find inner LlamaModel in {type(self.base_model)}")

    def _get_embed_tokens(self):
        return self._get_inner_model().embed_tokens

    def _get_lm_head(self):
        m = self.base_model
        # PeftModelForCausalLM: .model -> LlamaForCausalLM
        if hasattr(m, 'model') and hasattr(m.model, 'lm_head'):
            return m.model.lm_head
        if hasattr(m, 'lm_head'):
            return m.lm_head
        raise RuntimeError(f"Cannot find lm_head in {type(self.base_model)}")

    def _get_norm(self):
        return self._get_inner_model().norm

    def _get_rotary_emb(self):
        return self._get_inner_model().rotary_emb

    def _get_decoder_layers(self):
        return list(self._get_inner_model().layers)

    def reset_slots(self):
        for i in range(self.num_layers):
            self.slot_values[i] = None

    def _build_extended_attn_mask(self, S, T, dtype, device, batch_size):
        L = S + T
        mask = torch.zeros(L, L, dtype=dtype, device=device)
        neg_inf = torch.finfo(dtype).min
        causal = torch.triu(
            torch.full((T, T), neg_inf, dtype=dtype, device=device),
            diagonal=1,
        )
        mask[S:, S:] = causal
        return mask.view(1, 1, L, L).expand(batch_size, 1, L, L).contiguous()

    def _extend_position_embeddings(self, position_embeddings, S):
        cos, sin = position_embeddings
        cos0 = cos[:, :1, :]
        sin0 = sin[:, :1, :]
        cos_ext = torch.cat([cos0.expand(cos.shape[0], S, cos.shape[-1]), cos], dim=1)
        sin_ext = torch.cat([sin0.expand(sin.shape[0], S, sin.shape[-1]), sin], dim=1)
        return cos_ext, sin_ext

    def forward_segment(self, input_ids: torch.Tensor, segment_idx: int) -> dict:
        """Forward one segment through the model with memory.

        Args:
            input_ids: [B, T] token ids for this segment
            segment_idx: which segment (0-indexed)

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        B, T = input_ids.shape
        device = input_ids.device
        # Use base model dtype (bf16), not memory param dtype (may be float32)
        embed_tokens = self._get_embed_tokens()
        dtype = embed_tokens.weight.dtype
        S = self.num_slots

        decoder_layers = self._get_decoder_layers()
        hidden_states = embed_tokens(input_ids).to(dtype)

        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        rotary_emb = self._get_rotary_emb()
        position_embeddings = rotary_emb(hidden_states, position_ids)

        if self.memory_variant == "B":
            # Joint attention: prepend slots at every layer
            ext_pos_emb = self._extend_position_embeddings(position_embeddings, S)
            ext_attn_mask = self._build_extended_attn_mask(S, T, dtype, device, B)

            for layer_idx, layer in enumerate(decoder_layers):
                # Init or persist slots
                if self.slot_values[layer_idx] is None:
                    slots = self.slot_init_embed.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype, device=device)
                else:
                    slots = self.slot_values[layer_idx]

                extended = torch.cat([slots, hidden_states], dim=1)
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
                self.slot_values[layer_idx] = new_slots

        elif self.memory_variant in ("A", "D"):
            # Cross-attention: write at write_layer, read at read_layers
            write_layer = self.memory_write_layer
            read_layers = self.memory_read_layers
            ext_pos_emb = self._extend_position_embeddings(position_embeddings, S)
            ext_attn_mask = self._build_extended_attn_mask(S, T, dtype, device, B)

            for layer_idx, layer in enumerate(decoder_layers):
                if layer_idx == write_layer:
                    # Init or persist slots
                    if self.slot_values[write_layer] is None:
                        slots = self.slot_init_embed.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype, device=device)
                    else:
                        slots = self.slot_values[write_layer]

                    # Joint attention to update slots
                    extended = torch.cat([slots, hidden_states], dim=1)
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

                    # Dual-gate writeback
                    if self.dual_gate_proj_new is not None and self.slot_values[write_layer] is not None:
                        old_slots = self.slot_values[write_layer]
                        gate_logits = (
                            self.dual_gate_proj_new(new_slots)
                            + self.dual_gate_proj_mem(old_slots)
                            + self.dual_gate_bias
                        )
                        g_in_logit, g_forget_logit = gate_logits.chunk(2, dim=-1)
                        g_in = torch.sigmoid(g_in_logit)
                        g_forget = torch.sigmoid(g_forget_logit)
                        new_slots = g_in * torch.tanh(new_slots) + g_forget * old_slots

                    self.slot_values[write_layer] = new_slots

                elif layer_idx in read_layers:
                    slots = self.slot_values[write_layer]
                    if slots is None:
                        layer_out = layer(
                            hidden_states,
                            attention_mask=None, position_ids=None,
                            past_key_value=None, use_cache=False,
                            position_embeddings=position_embeddings,
                        )
                        hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                    else:
                        # Vanilla self-attn
                        layer_out = layer(
                            hidden_states,
                            attention_mask=None, position_ids=None,
                            past_key_value=None, use_cache=False,
                            position_embeddings=position_embeddings,
                        )
                        hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                        # Cross-attention read
                        ca_idx = self._read_layer_to_ca_idx[layer_idx]
                        cross_attn = self.cross_attn_modules[ca_idx]
                        memory_output, _attn_weights = cross_attn.read(
                            hidden_states, slots, slots,
                        )
                        hidden_states = hidden_states + self.residual_scale * memory_output
                else:
                    layer_out = layer(
                        hidden_states,
                        attention_mask=None, position_ids=None,
                        past_key_value=None, use_cache=False,
                        position_embeddings=position_embeddings,
                    )
                    hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        norm = self._get_norm()
        lm_head = self._get_lm_head()
        hidden_states = norm(hidden_states)
        logits = lm_head(hidden_states)
        return {"logits": logits}

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        """Process full sequence as segments, accumulating loss.

        Args:
            input_ids: [B, total_len] where total_len = segment_size * max_n_segments
            labels: [B, total_len]
        """
        B, L = input_ids.shape
        num_segments = L // self.segment_size

        self.reset_slots()

        total_loss = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)
        n_loss_segments = 0

        for seg_idx in range(num_segments):
            start = seg_idx * self.segment_size
            end = start + self.segment_size
            seg_ids = input_ids[:, start:end]

            result = self.forward_segment(seg_ids, seg_idx)
            logits = result["logits"]

            # Compute loss for this segment
            if labels is not None:
                seg_labels = labels[:, start:end]
                # Skip loss from first segment if requested
                if self.no_loss_from_first_segment and seg_idx == 0:
                    continue

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = seg_labels[..., 1:].contiguous()
                if not torch.any(shift_labels != -100):
                    continue
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                total_loss = total_loss + loss
                n_loss_segments += 1

        if n_loss_segments > 0:
            avg_loss = total_loss / n_loss_segments
        else:
            avg_loss = total_loss

        return {"loss": avg_loss, "num_segments": num_segments}


# --------------------------------------------------------------------------- #
# Distributed init
# --------------------------------------------------------------------------- #

def init_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group("nccl")

    return rank, world_size, local_rank


# --------------------------------------------------------------------------- #
# Parse args
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="H-series v2 Phase 1 training")

    # Model
    parser.add_argument("--base_model", type=str,
                        default="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B")
    parser.add_argument("--memory_variant", type=str, choices=["A", "B", "D"], default="A")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", action="store_true", default=False)

    # Memory
    parser.add_argument("--num_slots", type=int, default=64)
    parser.add_argument("--memory_write_layer", type=int, default=8)
    parser.add_argument("--memory_read_layers", type=str, default="10,12,14")
    parser.add_argument("--write_lr", type=float, default=0.1)
    parser.add_argument("--residual_scale", type=float, default=0.01)
    parser.add_argument("--use_dual_gate", action="store_true", default=True)
    parser.add_argument("--no_dual_gate", action="store_true", default=False)
    parser.add_argument("--forget_bias_init", type=float, default=1.0)
    parser.add_argument("--input_bias_init", type=float, default=0.0)

    # LoRA (variant D)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Data
    parser.add_argument("--dataset_path", type=str,
                        default="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/armt_pg19_real_tokenized_full")
    parser.add_argument("--segment_size", type=int, default=512)
    parser.add_argument("--max_n_segments", type=int, default=2)
    parser.add_argument("--no_loss_from_first_segment", action="store_true", default=True)

    # Training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # Logging / output
    parser.add_argument("--output_dir", type=str, default="outputs/h_v2_phase1_A")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    # Resume
    parser.add_argument("--resume_checkpoint", type=str, default=None)

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# LR scheduler (linear warmup + linear decay)
# --------------------------------------------------------------------------- #

def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    # Linear decay to 0
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return base_lr * max(0.0, 1.0 - progress)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    if args.no_freeze_backbone:
        args.freeze_backbone = False
    if args.no_dual_gate:
        args.use_dual_gate = False

    rank, world_size, local_rank = init_distributed()
    is_main = rank == 0

    for handler in logging.root.handlers:
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s [R{rank}] %(levelname)s %(message)s")
        )

    device = torch.device(f"cuda:{local_rank}")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Seed
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    if is_main:
        logger.info("=" * 60)
        logger.info("H-series v2 Phase 1: variant=%s, freeze=%s", args.memory_variant, args.freeze_backbone)
        logger.info("=" * 60)
        logger.info("Args: %s", vars(args))

    # Load base model
    if is_main:
        logger.info("Loading base model: %s", args.base_model)
    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map={"": device},
    )

    # Build v2 model
    model = HSeriesV2Model(
        base_model=base_model,
        memory_variant=args.memory_variant,
        num_slots=args.num_slots,
        segment_size=args.segment_size,
        max_n_segments=args.max_n_segments,
        freeze_backbone=args.freeze_backbone,
        no_loss_from_first_segment=args.no_loss_from_first_segment,
        memory_write_layer=args.memory_write_layer,
        memory_read_layers=args.memory_read_layers,
        write_lr=args.write_lr,
        residual_scale=args.residual_scale,
        use_dual_gate=args.use_dual_gate,
        forget_bias_init=args.forget_bias_init,
        input_bias_init=args.input_bias_init,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device).to(dtype)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info("Trainable: %d / %d (%.4f%%)", trainable, total, 100.0 * trainable / total)

    # Resume / initialize from checkpoint if requested.
    if args.resume_checkpoint is not None:
        if is_main:
            logger.info("Loading checkpoint: %s", args.resume_checkpoint)
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        cleaned = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if is_main and missing:
            logger.info("Resume missing keys (%d): %s", len(missing), missing[:5])
        if is_main and unexpected:
            logger.info("Resume unexpected keys (%d): %s", len(unexpected), unexpected[:5])

    # DDP
    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        ddp_model = model

    # Dataset
    train_dataset = PG19SegmentDataset(
        dataset_path=args.dataset_path,
        split="train",
        segment_size=args.segment_size,
        max_n_segments=args.max_n_segments,
        seed=args.seed,
    )
    val_dataset = PG19SegmentDataset(
        dataset_path=args.dataset_path,
        split="validation",
        segment_size=args.segment_size,
        max_n_segments=args.max_n_segments,
        seed=args.seed,
    )

    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    accum_loss = 0.0
    accum_count = 0

    if is_main:
        logger.info("Starting training: max_steps=%d, grad_accum=%d, effective_batch=%d",
                    args.max_steps, args.gradient_accumulation_steps,
                    args.batch_size * args.gradient_accumulation_steps * world_size)

    train_iter = iter(train_loader)
    start_time = time.time()

    while global_step < args.max_steps:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            if train_sampler is not None:
                train_sampler.set_epoch(global_step)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward
        result = ddp_model(input_ids, labels)
        loss = result["loss"] if isinstance(result, dict) else result[0]
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        accum_loss += loss.item() * args.gradient_accumulation_steps
        accum_count += 1

        if accum_count >= args.gradient_accumulation_steps:
            # Grad clip
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip,
                )

            # LR schedule
            lr = get_lr(global_step, args.warmup_steps, args.max_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            avg_loss = accum_loss / accum_count

            # Logging
            if is_main and global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                logger.info(
                    "step=%d loss=%.4f lr=%.2e elapsed=%.1fs",
                    global_step, avg_loss, lr, elapsed,
                )

            # Save checkpoint
            if is_main and global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_{global_step}.pt")
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, ckpt_path)
                logger.info("Saved checkpoint: %s", ckpt_path)

            accum_loss = 0.0
            accum_count = 0

    # Final save
    if is_main:
        ckpt_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        torch.save({
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
        }, ckpt_path)
        logger.info("Training complete. Final checkpoint: %s", ckpt_path)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
