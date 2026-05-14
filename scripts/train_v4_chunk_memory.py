#!/usr/bin/env python3
"""V4 Per-Layer Chunk Memory Bank -- Phase 1+2 training.

Phase 1 (append-only): slots fill up one by one, model sees all filled slots.
Phase 2 (top-k selection): once bank is full, select top-k slots by cosine
similarity + epsilon-greedy exploration, EMA-update selected slots.

Trains LoRA adapters on a frozen Llama-3-8B so the model learns to
attend to memory-bank slot prefixes.  The memory bank itself is pure
runtime state (no gradient).

Design reference:  versions/v4_chunk_last_hidden_memory.md
Feasibility:       status/V4_FEASIBILITY_ANALYSIS.md
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

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

from src.memory.mem_space.chunk_memory_bank import ChunkMemoryBank

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [R%(rank)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Prefix causal mask (from v4 design doc Section 2.3)
# --------------------------------------------------------------------------- #

def make_prefix_causal_mask(
    n_slots: int,
    n_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor:
    """Build [B, 1, n_slots+n_tokens, n_slots+n_tokens] additive mask.

    Pattern:
        slot -> slot  : allowed (0)
        slot -> token : masked  (-inf)  -- slots do NOT see future tokens
        token -> slot : allowed (0)     -- tokens see ALL slots
        token -> token: causal
    """
    N = n_slots + n_tokens
    neg_inf = torch.finfo(dtype).min
    mask = torch.zeros(N, N, dtype=dtype, device=device)

    # slot -> token: masked
    mask[:n_slots, n_slots:] = neg_inf

    # token -> token: causal upper triangle
    token_causal = torch.triu(
        torch.full((n_tokens, n_tokens), neg_inf, dtype=dtype, device=device),
        diagonal=1,
    )
    mask[n_slots:, n_slots:] = token_causal

    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, N, N).contiguous()


# --------------------------------------------------------------------------- #
# Extend position embeddings (same logic as layer.py:_extend_position_embeddings)
# --------------------------------------------------------------------------- #

def extend_position_embeddings(
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepend k position-0 entries to (cos, sin) tables.

    position_id=0 -> cos(0)=1, sin(0)=0 -> no RoPE rotation on slots.
    """
    cos, sin = position_embeddings
    cos0 = cos[:, :1, :]
    sin0 = sin[:, :1, :]
    cos_ext = torch.cat([cos0.expand(cos.shape[0], k, cos.shape[-1]), cos], dim=1)
    sin_ext = torch.cat([sin0.expand(sin.shape[0], k, sin.shape[-1]), sin], dim=1)
    return cos_ext, sin_ext


# --------------------------------------------------------------------------- #
# ChunkMemoryModel
# --------------------------------------------------------------------------- #

class ChunkMemoryModel(nn.Module):
    """Wraps a PeftModel (LoRA on frozen Llama-3-8B) with per-layer memory banks.

    Forward processes one chunk at a time.  The caller is responsible for
    iterating chunks within a document and calling reset_banks() between docs.
    """

    def __init__(
        self,
        base_model: LlamaForCausalLM,
        num_slots: int = 64,
        lora_rank: int = 16,
        top_k: int = 8,
        epsilon: float = 0.05,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.top_k = top_k
        self.epsilon = epsilon

        # Apply LoRA to Q/V projections on the frozen backbone.
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        )
        # Freeze all backbone params first.
        for p in base_model.parameters():
            p.requires_grad = False
        self.peft_model = get_peft_model(base_model, lora_config)

        # Derive model metadata.
        config = base_model.config
        self.num_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.d_model // self.num_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)

        # Per-layer memory banks (plain Python objects, NOT nn.Module).
        self.banks: list[ChunkMemoryBank] = [
            ChunkMemoryBank(num_slots, self.d_model) for _ in range(self.num_layers)
        ]

        # Direct references to the internal decoder layers for hooking.
        self._decoder_layers: list[nn.Module] = self._get_decoder_layers()

    def _get_decoder_layers(self) -> list[nn.Module]:
        """Walk the peft model to find model.model.layers."""
        # peft wraps: PeftModel -> base_model (PeftModelForCausalLM) ->
        #   model (LlamaForCausalLM) -> model (LlamaModel) -> layers
        base = self.peft_model.base_model.model.model
        return list(base.layers)

    def reset_banks(self) -> None:
        for bank in self.banks:
            bank.reset()

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward one chunk through all decoder layers with memory-bank prefix.

        Args:
            input_ids: [B, T]
            labels:    [B, T] or None (eval-only mode)

        Returns:
            dict with "loss" (if labels provided) and "logits".
        """
        B, T = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # Get the internal LlamaModel to compute embeddings + position embeddings.
        llama_model = self.peft_model.base_model.model.model
        embed_tokens = llama_model.embed_tokens
        hidden_states = embed_tokens(input_ids)  # [B, T, d]
        hidden_states = hidden_states.to(dtype)

        # Position ids: [1, T]
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        # Compute rotary embeddings from the model's rotary_emb.
        rotary_emb = llama_model.rotary_emb
        position_embeddings = rotary_emb(hidden_states, position_ids)  # (cos, sin)

        # Build the base causal mask for the tokens-only portion.
        # We will extend it layer-by-layer as needed.
        neg_inf = torch.finfo(dtype).min
        base_causal = torch.triu(
            torch.full((T, T), neg_inf, dtype=dtype, device=device), diagonal=1
        )  # [T, T]
        base_causal_4d = base_causal.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T).contiguous()

        # ------------------------------------------------------------------
        # Pass through each decoder layer, injecting bank slots at each layer.
        # ------------------------------------------------------------------
        for layer_idx, layer in enumerate(self._decoder_layers):
            bank = self.banks[layer_idx]
            n_filled = bank.num_filled

            if n_filled == 0:
                # No slots yet -- normal forward.
                layer_out = layer(
                    hidden_states,
                    attention_mask=base_causal_4d,
                    position_ids=position_ids,
                    past_key_value=None,
                    use_cache=False,
                    position_embeddings=position_embeddings,
                )
                if isinstance(layer_out, tuple):
                    hidden_out = layer_out[0]
                else:
                    hidden_out = layer_out

                # Update bank with last token hidden (detached).
                last_h = hidden_out[:, -1, :].detach()  # [B, d]
                bank.append(last_h)
                hidden_states = hidden_out
            else:
                # Phase 1 (bank not full) or Phase 2 (bank full).
                selected_idx = None  # track for Phase 2 EMA update

                if not bank.is_full:
                    # Phase 1: use all filled slots.
                    slots = bank.get_all()  # [B, n_filled, d]
                    n_slots = slots.shape[1]
                else:
                    # Phase 2: top-k selection + EMA update.
                    query = hidden_states.detach().mean(dim=1)  # [B, d]

                    # epsilon-greedy exploration.
                    if random.random() < self.epsilon:
                        # Random selection for exploration.
                        k = min(self.top_k, bank.num_slots)
                        idx = torch.randperm(bank.num_slots, device=device)[:k]
                        idx = idx.unsqueeze(0).expand(B, -1)  # [B, k]
                        slots = bank.slots.gather(
                            1, idx.unsqueeze(-1).expand(-1, -1, bank.d_model)
                        ).detach()
                        selected_idx = idx
                        n_slots = k
                    else:
                        slots, selected_idx = bank.top_k(query, self.top_k)
                        n_slots = slots.shape[1]  # = top_k

                # Build extended sequence [slots | tokens].
                extended = torch.cat([slots, hidden_states], dim=1)  # [B, n_slots+T, d]

                # Build prefix causal mask.
                ext_mask = make_prefix_causal_mask(n_slots, T, dtype, device, B)

                # Extend position embeddings: slots get pos=0.
                ext_pos_emb = extend_position_embeddings(position_embeddings, n_slots)

                # Forward through the decoder layer.
                layer_out = layer(
                    extended,
                    attention_mask=ext_mask,
                    position_ids=None,  # RoPE driven by ext_pos_emb
                    past_key_value=None,
                    use_cache=False,
                    position_embeddings=ext_pos_emb,
                )
                if isinstance(layer_out, tuple):
                    ext_output = layer_out[0]
                else:
                    ext_output = layer_out

                # Take only token portion.
                hidden_out = ext_output[:, n_slots:, :]  # [B, T, d]

                # Update bank with last token hidden (detached).
                last_h = hidden_out[:, -1, :].detach()
                if not bank.is_full:
                    # Phase 1: append.
                    bank.append(last_h)
                else:
                    # Phase 2: EMA update selected slots.
                    bank.update_selected(selected_idx, last_h)

                hidden_states = hidden_out

        # ------------------------------------------------------------------
        # Final layernorm + LM head (from the peft model).
        # ------------------------------------------------------------------
        llama_model_out = llama_model.norm(hidden_states)
        lm_head = self.peft_model.base_model.model.lm_head
        logits = lm_head(llama_model_out)  # [B, T, vocab]

        result = {"logits": logits}
        if labels is not None:
            # Shift for NTP loss.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fn = nn.CrossEntropyLoss(reduction="mean")
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            result["loss"] = loss
        return result

    def forward(self, input_ids, labels=None, **kwargs):
        return self.forward_chunk(input_ids, labels=labels)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class DocumentChunkDataset(Dataset):
    """Groups pg19 chunks into documents of `chunks_per_doc` sequential chunks.

    Each sample is a list of chunk arrays (each chunk = seq_len tokens).
    During training, the model processes chunks sequentially within a document,
    with memory banks persisting across chunks.
    """

    def __init__(
        self,
        npy_path: str,
        seq_length: int,
        skip_chunks: int,
        max_chunks: int,
        chunks_per_doc: int = 8,
    ) -> None:
        data = np.load(npy_path, mmap_mode="r")
        self.data = data[skip_chunks: skip_chunks + max_chunks].astype(np.int32)
        self.seq_length = seq_length
        self.chunks_per_doc = chunks_per_doc

        # Number of complete documents we can form.
        n_chunks = len(self.data)
        self.n_docs = max(1, n_chunks // chunks_per_doc)
        self.n_docs = min(self.n_docs, n_chunks)

        logger.info(
            "Loaded %d chunks -> %d documents (%d chunks/doc) from %s",
            n_chunks, self.n_docs, chunks_per_doc, npy_path,
        )

    def __len__(self) -> int:
        return self.n_docs

    def __getitem__(self, idx: int):
        start = idx * self.chunks_per_doc
        end = start + self.chunks_per_doc
        chunks = []
        for i in range(start, min(end, len(self.data))):
            tokens = torch.tensor(self.data[i], dtype=torch.long)[: self.seq_length]
            chunks.append({"input_ids": tokens, "labels": tokens.clone()})
        # Pad if we ran out of chunks.
        while len(chunks) < self.chunks_per_doc:
            tokens = torch.zeros(self.seq_length, dtype=torch.long)
            chunks.append({"input_ids": tokens, "labels": torch.full_like(tokens, -100)})
        return {"chunks": chunks}


def doc_collate_fn(batch):
    """Collate a batch of documents.  Each doc has chunks_per_doc chunks."""
    # batch: list of dicts with "chunks" key
    # chunks: list of dicts with "input_ids" and "labels"
    return batch[0]["chunks"]  # batch_size=1, return the list of chunks


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #

def init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


# --------------------------------------------------------------------------- #
# Eval helper
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_vanilla_ppl(model, loader, device, pad_token_id, world_size):
    """Compute PPL WITHOUT memory banks (vanilla baseline)."""
    model.eval()
    # Temporarily disable banks by resetting before each chunk.
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        # Reset banks so no slots are used.
        root = model.module if hasattr(model, "module") else model
        root.reset_banks()

        out = root.forward_chunk(input_ids, labels=labels)
        loss = out["loss"].detach()
        if not torch.isfinite(loss):
            continue
        n_tok = (labels != -100).sum()
        total_loss += loss.double() * n_tok.double()
        total_tokens += n_tok.double()

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    tot = int(total_tokens.item())
    if tot == 0:
        return float("inf"), 0
    avg_loss = (total_loss / total_tokens).item()
    return math.exp(avg_loss), tot


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V4 Chunk Memory Bank -- Phase 1 training")
    p.add_argument("--model", type=str, required=True, help="Path to Llama-3-8B weights")
    p.add_argument("--data", type=str, required=True, help="Path to pg19_chunks_llama3.npy")
    p.add_argument("--num_slots", type=int, default=64)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_steps", type=int, default=1000, help="Max optimizer steps")
    p.add_argument("--max_chunks", type=int, default=500, help="Max chunks to load")
    p.add_argument("--skip_chunks", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--chunks_per_doc", type=int, default=8,
                   help="Number of chunks per document (banks persist across these)")
    p.add_argument("--top_k", type=int, default=8,
                   help="Number of slots to select in Phase 2 (top-k)")
    p.add_argument("--epsilon", type=float, default=0.05,
                   help="Epsilon-greedy exploration probability for Phase 2")
    p.add_argument("--resume_checkpoint", type=str, default=None,
                   help="Path to Phase 1 checkpoint to resume from")
    p.add_argument("--output_dir", type=str, default="outputs/v4_chunk_memory")
    p.add_argument("--eval_interval", type=int, default=50, help="Eval every N steps")
    p.add_argument("--eval_chunks", type=int, default=100,
                   help="Number of chunks for eval PPL computation")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    is_main = rank == 0

    # Patch logger with rank info.
    for handler in logging.root.handlers:
        handler.setFormatter(
            logging.Formatter(f"%(asctime)s [R{rank}] %(levelname)s %(message)s")
        )

    device = torch.device(f"cuda:{local_rank}")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    if is_main:
        logger.info("=" * 60)
        logger.info("V4 Chunk Memory Bank -- Phase 1+2 (append + top-k selection)")
        logger.info("=" * 60)
        logger.info("Args: %s", vars(args))

    # ------------------------------------------------------------------
    # Load model.
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Loading base model from %s ...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map={"": device},
    )
    if is_main:
        logger.info("Base model loaded.  Building ChunkMemoryModel with LoRA ...")

    cm_model = ChunkMemoryModel(
        base_model=base_model,
        num_slots=args.num_slots,
        lora_rank=args.lora_rank,
        top_k=args.top_k,
        epsilon=args.epsilon,
    ).to(device)

    # Print trainable params.
    trainable = sum(p.numel() for p in cm_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in cm_model.parameters())
    if is_main:
        logger.info("Trainable params: %d / %d (%.4f%%)",
                    trainable, total, 100.0 * trainable / total)

    # Wrap in DDP.
    ddp_model = DDP(cm_model, device_ids=[local_rank])

    # Resume from Phase 1 checkpoint if provided.
    if args.resume_checkpoint:
        if is_main:
            logger.info("Resuming from checkpoint: %s", args.resume_checkpoint)
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        cm_model.peft_model.load_state_dict(ckpt['lora_state_dict'])
        if is_main:
            logger.info("Resumed from %s (step %d)",
                        args.resume_checkpoint, ckpt.get('global_step', 0))

    # ------------------------------------------------------------------
    # Optimizer (LoRA params only).
    # ------------------------------------------------------------------
    lora_params = [p for p in ddp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

    # ------------------------------------------------------------------
    # Data loaders.
    # ------------------------------------------------------------------
    train_ds = DocumentChunkDataset(
        npy_path=args.data,
        seq_length=args.seq_len,
        skip_chunks=args.skip_chunks,
        max_chunks=args.max_chunks,
        chunks_per_doc=args.chunks_per_doc,
    )
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=doc_collate_fn,
        num_workers=0,
    )

    # Eval dataset: flat chunks (1 chunk = 1 sample), no document grouping.
    class FlatChunkDataset(Dataset):
        def __init__(self, npy_path, seq_len, skip, max_c):
            d = np.load(npy_path, mmap_mode="r")
            self.data = d[skip: skip + max_c].astype(np.int32)
            self.seq_len = seq_len
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            t = torch.tensor(self.data[idx], dtype=torch.long)[:self.seq_len]
            return {"input_ids": t, "labels": t.clone()}

    eval_skip = args.skip_chunks + args.max_chunks
    eval_ds = FlatChunkDataset(args.data, args.seq_len, eval_skip, args.eval_chunks)
    eval_sampler = DistributedSampler(eval_ds, num_replicas=world_size, rank=rank, shuffle=False)
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, sampler=eval_sampler,
        num_workers=0, collate_fn=lambda b: {
            "input_ids": torch.stack([x["input_ids"] for x in b]),
            "labels": torch.stack([x["labels"] for x in b]),
        },
    )

    # ------------------------------------------------------------------
    # Training loop.
    # ------------------------------------------------------------------
    pad_token_id = tokenizer.pad_token_id or 0
    root_model = ddp_model.module

    if is_main:
        logger.info("Starting training for %d steps ...", args.max_steps)
        os.makedirs(args.output_dir, exist_ok=True)

    step = 0
    global_step = 0
    epoch = 0
    best_ppl = float("inf")
    t0 = time.time()

    while global_step < args.max_steps:
        train_sampler.set_epoch(epoch)
        ddp_model.train()

        for doc_idx, chunks in enumerate(train_loader):
            if global_step >= args.max_steps:
                break

            # Reset banks at document boundary.
            root_model.reset_banks()

            doc_loss = 0.0
            doc_tokens = 0
            chunk_ppls = []

            for chunk_i, chunk in enumerate(chunks):
                input_ids = chunk["input_ids"].unsqueeze(0).to(device)
                labels = chunk["labels"].unsqueeze(0).to(device)

                # Skip padding-only chunks.
                if (labels != -100).sum() == 0:
                    continue

                result = ddp_model(input_ids=input_ids, labels=labels)
                loss = result["loss"]

                if not torch.isfinite(loss):
                    if is_main:
                        logger.warning(
                            "[step %d doc %d chunk %d] Non-finite loss!",
                            global_step, doc_idx, chunk_i,
                        )
                    continue

                n_tok = (labels != -100).sum().item()
                chunk_ppl = math.exp(min(loss.item(), 20))  # cap for display
                chunk_ppls.append(chunk_ppl)
                doc_loss += loss.item() * n_tok
                doc_tokens += n_tok

                # Backward on every chunk (LoRA learns from each chunk).
                if ddp_model.training:
                    loss.backward()

            if doc_tokens > 0 and ddp_model.training:
                # Gradient accumulation is 1:1 with docs here.
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                doc_ppl = math.exp(min(doc_loss / doc_tokens, 20))

                if is_main and (global_step % 10 == 0 or global_step <= 5):
                    elapsed = time.time() - t0
                    bank_fill = root_model.banks[0].num_filled
                    phase_label = "P2" if bank_fill >= args.num_slots else "P1"
                    logger.info(
                        "[step %d] %s doc_ppl=%.4f chunk_ppls=[%s] bank_fill=%d/%d "
                        "time=%.1fs",
                        global_step, phase_label, doc_ppl,
                        ", ".join(f"c{i}={p:.2f}" for i, p in enumerate(chunk_ppls)),
                        bank_fill, args.num_slots,
                        elapsed,
                    )

                # Periodic evaluation.
                if global_step % args.eval_interval == 0:
                    ddp_model.eval()
                    vanilla_ppl, _ = evaluate_vanilla_ppl(
                        ddp_model, eval_loader, device, pad_token_id, world_size,
                    )
                    if is_main:
                        logger.info(
                            "[EVAL step=%d] vanilla_ppl=%.4f",
                            global_step, vanilla_ppl,
                        )

                    # Memory-augmented eval: run chunks sequentially with bank.
                    mem_total_loss = torch.zeros((), device=device, dtype=torch.float64)
                    mem_total_tokens = torch.zeros((), device=device, dtype=torch.float64)
                    root_model.reset_banks()
                    for ei, ebatch in enumerate(eval_loader):
                        e_ids = ebatch["input_ids"].to(device)
                        e_labels = ebatch["labels"].to(device)
                        e_result = root_model.forward_chunk(e_ids, labels=e_labels)
                        e_loss = e_result["loss"].detach()
                        if torch.isfinite(e_loss):
                            n_tok = (e_labels != -100).sum()
                            mem_total_loss += e_loss.double() * n_tok.double()
                            mem_total_tokens += n_tok.double()

                    if world_size > 1:
                        dist.all_reduce(mem_total_loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(mem_total_tokens, op=dist.ReduceOp.SUM)

                    if mem_total_tokens.item() > 0:
                        mem_avg_loss = (mem_total_loss / mem_total_tokens).item()
                        mem_ppl = math.exp(mem_avg_loss)
                    else:
                        mem_ppl = float("inf")

                    if is_main:
                        logger.info(
                            "[EVAL step=%d] memory_ppl=%.4f  vanilla_ppl=%.4f  ratio=%.4f",
                            global_step, mem_ppl, vanilla_ppl,
                            mem_ppl / max(vanilla_ppl, 1e-8),
                        )
                        # Go/No-Go check.
                        if mem_ppl > vanilla_ppl * 1.5:
                            logger.warning(
                                "GO/NO-GO: memory_ppl=%.4f > vanilla*1.5=%.4f -- KILL AND DIAGNOSE",
                                mem_ppl, vanilla_ppl * 1.5,
                            )

                    ddp_model.train()

                # Save checkpoint.
                if is_main and global_step % 200 == 0:
                    ckpt_path = os.path.join(args.output_dir, f"step_{global_step}.pt")
                    torch.save(
                        {"global_step": global_step, "lora_state_dict": cm_model.peft_model.state_dict()},
                        ckpt_path,
                    )
                    logger.info("Saved checkpoint: %s", ckpt_path)

        epoch += 1

    # ------------------------------------------------------------------
    # Final evaluation.
    # ------------------------------------------------------------------
    if is_main:
        logger.info("Training complete.  Running final evaluation ...")

    ddp_model.eval()
    vanilla_ppl, _ = evaluate_vanilla_ppl(
        ddp_model, eval_loader, device, pad_token_id, world_size,
    )
    if is_main:
        logger.info("[FINAL] vanilla_ppl=%.4f", vanilla_ppl)

    # Save final checkpoint.
    if is_main:
        final_path = os.path.join(args.output_dir, "final.pt")
        torch.save(
            {"global_step": global_step, "lora_state_dict": cm_model.peft_model.state_dict()},
            final_path,
        )
        logger.info("Saved final checkpoint: %s", final_path)

        # Write summary.
        summary = {
            "global_step": global_step,
            "vanilla_ppl": vanilla_ppl,
            "num_slots": args.num_slots,
            "top_k": args.top_k,
            "epsilon": args.epsilon,
            "lora_rank": args.lora_rank,
            "lr": args.lr,
            "chunks_per_doc": args.chunks_per_doc,
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
