#!/usr/bin/env python3
"""Training script for ConcatFusionAttention sparse memory (Phase 2: concat fusion + bypass gate).

Phase 2 Goal: Fix chicken-and-egg problem via concatenation fusion.
    - bypass_gate_bias_init=-2.0 → sigmoid(-2)≈0.12 (strong local bias)
    - Memory output added via bypass gate, not gated replacement
    - Bypass gate LR = 10x base LR for faster adaptation
    - Gradients flow through both paths via addition

Usage:
    torchrun --nproc_per_node=8 scripts/train_gated_sparse_memory.py \
        --data_path /path/to/data.jsonl \
        --output_dir outputs/phase2_concat_fusion \
        --bypass_bias_init -2.0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.sparse import SparseMemoryModel


class JsonlDataset(Dataset):
    """Simple dataset for causal language modeling. Handles JSONL and plain text."""

    def __init__(self, data_path: str, tokenizer, max_seq_length: int = 2048):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        import json
        self.examples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # Detect format: if first non-empty line parses as JSON dict, treat as JSONL
            is_jsonl = False
            if first_line:
                try:
                    obj = json.loads(first_line)
                    if isinstance(obj, dict) and ('text' in obj or 'content' in obj):
                        is_jsonl = True
                        self.examples.append(obj.get('text', obj.get('content', '')))
                except (json.JSONDecodeError, ValueError):
                    pass

            if not is_jsonl:
                # Plain text: first line is already read, collect non-empty lines as chunks
                chunks = []
                current_chunk = [first_line] if first_line else []
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        current_chunk.append(stripped)
                    else:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                            current_chunk = []
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                self.examples = chunks
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if not isinstance(data, dict):
                            continue
                        if 'text' in data:
                            self.examples.append(data['text'])
                        elif 'content' in data:
                            self.examples.append(data['content'])
                    except json.JSONDecodeError:
                        continue

        print(f"Loaded {len(self.examples)} examples from {data_path} (format: {'jsonl' if is_jsonl else 'plain_text'})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids = encodings['input_ids'].squeeze(0)
        labels = input_ids.clone()
        attention_mask = encodings['attention_mask'].squeeze(0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class GateMetricsCallback(TrainerCallback):
    """Callback to track gate statistics during training."""

    def __init__(self):
        super().__init__()
        self.global_step = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log gate statistics."""
        if logs is None:
            return

        # Gate stats are logged by the model
        # This is just a placeholder if needed for custom logging
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConcatFusionAttention sparse memory")

    # Model args
    parser.add_argument("--model_name_or_path", type=str,
                        default="models/Llama--Llama2-7b",
                        help="Path to base model or HF model name")
    parser.add_argument("--config_name", type=str, default=None,
                        help="Pretrained config name or path if not the same as model_name")

    # Data args
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data (JSONL format)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4,
                        help="Number of preprocessing workers")

    # Sparse memory config
    parser.add_argument("--num_slots", type=int, default=128,
                        help="Number of memory slots per layer")
    parser.add_argument("--window_size", type=int, default=256,
                        help="Sliding window size for local attention")
    parser.add_argument("--top_k", type=int, default=8,
                        help="Top-k memory retrieval per token per head")
    parser.add_argument("--ema_alpha", type=float, default=0.1,
                        help="EMA decay rate for memory bank updates")
    parser.add_argument("--write_top_k", type=int, default=0,
                        help="Number of top-important tokens to write per chunk. 0 = write all (legacy)")
    parser.add_argument("--importance_mode", type=str, default="combined",
                        choices=["magnitude", "attention_surprise", "combined"],
                        help="Token importance scoring method for selective writing")

    # Phase 2: Bypass gate initialization
    parser.add_argument("--bypass_bias_init", type=float, default=-2.0,
                        help="Bypass gate bias. -2.0 → sigmoid(-2)≈0.12 (strong local bias)")
    parser.add_argument("--gate_bias_init", type=float, default=None,
                        help="Deprecated: use --bypass_bias_init instead")
    parser.add_argument("--bypass_gate_lr_multiplier", type=float, default=10.0,
                        help="LR multiplier for bypass gate params (default 10x)")

    # Training args
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite output directory if it exists")
    parser.add_argument("--do_train", action="store_true", default=True,
                        help="Whether to run training")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per GPU/CPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=50,
                        help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X updates steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Limit the total amount of checkpoints")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Use gradient checkpointing to save memory")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", default=True,
                        help="Find unused parameters in DDP")

    # Distributed training
    # local_rank is provided by torchrun automatically; do not define it here.

    args = parser.parse_args()

    # Handle deprecated gate_bias_init
    if args.gate_bias_init is not None and args.bypass_bias_init == -2.0:
        args.bypass_bias_init = args.gate_bias_init
        print(f"Warning: --gate_bias_init is deprecated, using {args.bypass_bias_init} as bypass_bias_init")

    return args


def main():
    args = parse_args()

    # Setup distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device(local_rank)
        dist.init_process_group(backend='nccl')

    # Load config
    config_name = args.config_name if args.config_name else args.model_name_or_path
    config = AutoConfig.from_pretrained(config_name)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"Loading base model: {args.model_name_or_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        device_map={"": 0} if local_rank == -1 else None,
    )

    if local_rank != -1:
        base_model = base_model.to(local_rank)

    # Wrap with ConcatFusionAttention + SparseMemoryBank
    print(f"\n=== Phase 2: Concatenation Fusion + Bypass Gate ===")
    print(f"Bypass gate bias init: {args.bypass_bias_init}")
    print(f"Expected bypass at init: {torch.sigmoid(torch.tensor(args.bypass_bias_init)):.4f}")
    print(f"Bypass gate LR multiplier: {args.bypass_gate_lr_multiplier}")
    print(f"Num slots: {args.num_slots}, Window: {args.window_size}, Top-k: {args.top_k}")
    print(f"Write top-k: {args.write_top_k}, Importance mode: {args.importance_mode}")

    model = base_model  # Keep PreTrainedModel for Trainer compatibility

    # Patch attention layers and attach memory bank
    from src.memory.sparse.attention import ConcatFusionAttention
    from src.memory.sparse.memory_bank import SparseMemoryBank

    config = model.config
    num_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_dim // num_heads

    # Create per-layer independent memory banks to avoid shared-tensor checkpoint crash
    memory_banks = [
        SparseMemoryBank(
            num_layers=1,
            num_slots=args.num_slots,
            hidden_dim=hidden_dim,
            head_dim=head_dim,
            ema_alpha=args.ema_alpha,
            gate_bias_init=args.bypass_bias_init,
            write_top_k=args.write_top_k,
            importance_mode=args.importance_mode,
        )
        for _ in range(num_layers)
    ]

    # Patch attention modules
    for layer_idx in range(num_layers):
        attn = model.model.layers[layer_idx].self_attn
        gated_attn = ConcatFusionAttention(
            original_attn=attn,
            layer_idx=0,  # each bank is per-layer, index into 1-layer bank
            memory_bank=memory_banks[layer_idx],
            window_size=args.window_size,
            top_k=args.top_k,
            head_dim=head_dim,
            bypass_bias_init=args.bypass_bias_init,
        )
        model.model.layers[layer_idx].self_attn = gated_attn

    # Load dataset
    print(f"\nLoading dataset from: {args.data_path}")
    train_dataset = JsonlDataset(
        args.data_path,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # overwrite_output_dir removed in transformers 5.x
        do_train=args.do_train,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        report_to="none",
    )

    # Create custom parameter groups for bypass gate LR multiplier
    bypass_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'bypass_gate_proj' in name:
            bypass_params.append(param)
        else:
            other_params.append(param)

    print(f"\nParameter groups: bypass_gate={len(bypass_params)}, other={len(other_params)}")
    if bypass_params:
        param_groups = [
            {"params": bypass_params, "lr": args.learning_rate * args.bypass_gate_lr_multiplier},
            {"params": other_params},
        ]
    else:
        param_groups = model.parameters()

    # Initialize trainer with custom param groups
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        callbacks=[],
    )

    # Override optimizer with custom param groups.
    # FIX: Override create_optimizer so param groups reference the
    # DDP-wrapped model's parameters (created inside train()),
    # not the pre-DDP parameters.
    if bypass_params:
        import types
        from torch.optim import AdamW
        _blr = args.learning_rate * args.bypass_gate_lr_multiplier
        _base_lr = args.learning_rate
        _wd = args.weight_decay

        def _custom_create_optimizer(self_trainer):
            model_ref = getattr(self_trainer, 'model_wrapped', None) or self_trainer.model
            bp, op = [], []
            for name, param in model_ref.named_parameters():
                if 'bypass_gate_proj' in name:
                    bp.append(param)
                else:
                    op.append(param)
            self_trainer.optimizer = AdamW(
                [{"params": bp, "lr": _blr}, {"params": op}],
                lr=_base_lr,
                weight_decay=_wd,
                betas=(0.9, 0.999),
            )
            print(f"[Optimizer] bypass_gate({len(bp)}) LR={_blr:.2e}, other({len(op)}) LR={_base_lr:.2e}")
        trainer.create_optimizer = types.MethodType(_custom_create_optimizer, trainer)
        print(f"Optimizer will use custom param groups (bypass_gate LR={_blr:.2e}, base LR={_base_lr:.2e})")

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )

    # Train
    print("\n=== Starting Training ===")
    print(f"[Debug] Dataset: {len(train_dataset)} examples, batch_size={args.per_device_train_batch_size}")
    print(f"[Debug] Grad accum={args.gradient_accumulation_steps}, max_steps={args.max_steps}, log_every={args.logging_steps}")
    print(f"[Debug] BF16={args.bf16}, FP16={args.fp16}, GradCkpt={args.gradient_checkpointing}, DDP_find_unused={args.ddp_find_unused_parameters}")
    checkpoint = None
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
        print(f"Resuming training from {checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Save final model
    trainer.save_model()
    trainer.save_state()

    print("\n=== Training Complete ===")
    print(f"Final loss: {train_result.training_loss:.4f}")
    print(f"Total steps: {train_result.global_step}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
