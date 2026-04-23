#!/usr/bin/env python3
"""DMS Retrofitting Training Script.

Retrofits a pre-trained model (Qwen3-8B, Llama-2-7B, etc.) with Dynamic Memory Sparsification (DMS)
for KV cache compression. Uses logit distillation from the original model.

Usage:
    python scripts/train_dms.py \
        --model_name_or_path Qwen/Qwen3-8B \
        --compression_ratio 8 \
        --output_dir outputs/dms_8x \
        --num_train_steps 1000 \
        --per_device_train_batch_size 4 \
        --learning_rate 1e-4 \
        --sliding_window 256

Based on: Łańcucki et al., "Inference-Time Hyper-Scaling with KV Cache Compression"
(arXiv:2506.05345).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.memory.dms import apply_dms_to_model, DMSLoss, CompressionScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="DMS Retrofitting Training")

    # Model
    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="Pre-trained model path (e.g., meta-llama/Llama-2-7b-hf)")
    p.add_argument("--teacher_model_name_or_path", type=str, default=None,
                   help="Teacher model for distillation (default: same as student)")

    # DMS config
    p.add_argument("--compression_ratio", type=float, default=8.0,
                   help="Target compression ratio (4 or 8)")
    p.add_argument("--sliding_window", type=int, default=256,
                   help="Sliding window size for delayed eviction")
    p.add_argument("--tau", type=float, default=0.1,
                   help="Gumbel-Sigmoid temperature")
    p.add_argument("--lambda_aux", type=float, default=1.0,
                   help="Weight for auxiliary compression loss")
    p.add_argument("--warmup_steps", type=int, default=100,
                   help="Steps before compression starts")

    # Training
    p.add_argument("--num_train_steps", type=int, default=1000,
                   help="Total training steps (DMS paper: CR * 100)")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler_type", type=str, default="cosine")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--bf16", action="store_true", default=False)
    p.add_argument("--fp16", action="store_true", default=False)

    # Data
    p.add_argument("--dataset_name", type=str, default="allenai/dolma",
                   help="Training data for distillation")
    p.add_argument("--dataset_subset", type=str, default=None)
    p.add_argument("--dataset_split", type=str, default="train")
    p.add_argument("--num_train_samples", type=int, default=5000,
                   help="Number of training samples")

    # Output
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    # Multi-GPU
    p.add_argument("--num_gpus", type=int, default=None,
                   help="Number of GPUs (default: all available)")

    return p.parse_args()


def load_tokenizer(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_and_preprocess_data(args, tokenizer):
    """Load and tokenize training data."""
    logger.info(f"Loading dataset: {args.dataset_name}")

    try:
        if args.dataset_subset:
            ds = load_dataset(args.dataset_name, args.dataset_subset, split=args.dataset_split)
        else:
            ds = load_dataset(args.dataset_name, split=args.dataset_split)
    except Exception as e:
        logger.warning(f"Failed to load {args.dataset_name}: {e}")
        logger.info("Falling back to wikitext...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Limit samples
    ds = ds.select(range(min(args.num_train_samples, len(ds))))

    def tokenize_fn(example):
        text = example.get("text", "")
        if not text or not text.strip():
            text = example.get("content", "")
        if not text or not text.strip():
            return {"input_ids": [], "attention_mask": [], "labels": []}

        result = tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    ds = ds.map(tokenize_fn, remove_columns=ds.column_names, desc="Tokenizing")

    # Filter empty examples
    ds = ds.filter(lambda x: len(x["input_ids"]) > 16, desc="Filtering short")

    # Pad to max length for collation
    def pad_fn(x):
        pl = args.max_seq_length - len(x["input_ids"])
        x["input_ids"] = x["input_ids"] + [tokenizer.pad_token_id] * pl
        x["attention_mask"] = x["attention_mask"] + [0] * pl
        x["labels"] = x["labels"] + [-100] * pl
        return x
    ds = ds.map(pad_fn, desc="Padding")

    logger.info(f"Dataset size: {len(ds)} samples")
    return ds


class DMSTrainer(Trainer):
    """Custom trainer that incorporates DMS auxiliary loss."""

    def __init__(self, dms_loss: DMSLoss, scheduler: CompressionScheduler,
                 teacher_model=None, **kwargs):
        super().__init__(**kwargs)
        self.dms_loss = dms_loss
        self.scheduler = scheduler
        self.teacher_model = teacher_model
        self._step = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)

        # Student forward
        outputs = model(**inputs)
        student_logits = outputs.logits

        # Teacher forward (no gradient)
        teacher_logits = None
        if self.teacher_model is not None:
            with torch.no_grad():
                # Ensure teacher is on the same device as the student
                teacher_device = next(model.parameters()).device
                if next(self.teacher_model.parameters()).device != teacher_device:
                    self.teacher_model = self.teacher_model.to(teacher_device)
                teacher_outputs = self.teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits

        # Get current alpha_star
        alpha_star = self.scheduler.get_alpha_star(self._step)

        # Compute DMS loss
        loss_dict = self.dms_loss(
            student_logits=student_logits,
            labels=labels,
            teacher_logits=teacher_logits,
            alpha_star=alpha_star,
            model=model,
        )

        self._step += 1

        # Log compression info
        if self._step % self.args.logging_steps == 0:
            current_cr = self.scheduler.get_current_cr(self._step)
            logger.info(
                f"Step {self._step}: CR={current_cr:.2f}, "
                f"α*={alpha_star:.4f}, "
                f"loss={loss_dict['total']:.4f}, "
                f"aux={loss_dict['aux']:.4f}"
            )

        return (loss_dict["total"], outputs) if return_outputs else loss_dict["total"]


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # GPU config
    num_gpus = args.num_gpus or torch.cuda.device_count()
    logger.info(f"Using {num_gpus} GPUs")

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_name_or_path)

    # Load teacher model (original, frozen)
    teacher_path = args.teacher_model_name_or_path or args.model_name_or_path
    logger.info(f"Loading teacher model from: {teacher_path}")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto" if num_gpus == 1 else None,
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # Load student model (copy of teacher + DMS heads)
    logger.info(f"Loading student model from: {args.model_name_or_path}")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    student_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
    )

    # Apply DMS
    logger.info(f"Applying DMS with CR={args.compression_ratio}, window={args.sliding_window}")
    student_model = apply_dms_to_model(
        student_model,
        sliding_window=args.sliding_window,
        tau=args.tau,
    )

    # Count parameters
    total_params = sum(p.numel() for p in student_model.parameters())
    dms_params = sum(p.numel() for n, p in student_model.named_parameters() if "decision_head" in n)
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    logger.info(f"Total: {total_params:,}, DMS: {dms_params:,}, Trainable: {trainable_params:,}")

    # Freeze base model, only train DMS heads (DMS paper: only decision heads)
    # Training all params requires ~65GB optimizer states which OOMs on H20
    for name, param in student_model.named_parameters():
        if "decision_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Load data
    train_dataset = load_and_preprocess_data(args, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # DMS loss and scheduler
    dms_loss = DMSLoss(lambda_aux=args.lambda_aux)
    num_steps = args.num_train_steps
    scheduler = CompressionScheduler(
        target_cr=args.compression_ratio,
        total_steps=num_steps,
        warmup_steps=args.warmup_steps,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=num_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=True,
        report_to="none",
        seed=args.seed,
    )

    # Trainer
    trainer = DMSTrainer(
        dms_loss=dms_loss,
        scheduler=scheduler,
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting DMS retrofitting...")
    trainer.train()

    # Save
    output_path = os.path.join(args.output_dir, "final")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # Save DMS config
    dms_config = {
        "compression_ratio": args.compression_ratio,
        "sliding_window": args.sliding_window,
        "tau": args.tau,
        "lambda_aux": args.lambda_aux,
        "num_train_steps": num_steps,
        "warmup_steps": args.warmup_steps,
        "dms_parameters": dms_params,
    }
    with open(os.path.join(output_path, "dms_config.json"), "w") as f:
        json.dump(dms_config, f, indent=2)

    logger.info(f"DMS model saved to: {output_path}")
    logger.info(f"DMS config: {dms_config}")


if __name__ == "__main__":
    main()
