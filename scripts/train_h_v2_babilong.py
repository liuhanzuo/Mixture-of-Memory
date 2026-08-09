#!/usr/bin/env python3
"""H-series v2 Phase 2: BABILong qa1 fine-tuning on top of Phase 1 checkpoints.

This keeps the same H-v2 A/B/D architectures from scripts/train_h_v2.py and swaps the
Phase 1 PG19 LM objective for answer-only supervised fine-tuning on noisy bAbI qa1.

Data recipe is adapted from the ARMT/RMT BABILong training pipeline, but uses the local
Llama-3.2-1B-tokenized PG19 corpus for noise so the path is fully local / shared-FS.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM


def _find_babilong() -> str:
    candidates = [
        "/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong",
        "/apdcephfs_wzc1/share_304376610/pighzliu_code/babilong",
        "/apdcephfs_zwfy6/share_304376610/pighzliu_code/babilong",
    ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "babilong")):
            return candidate
    raise RuntimeError("Could not locate babilong package on known cluster mounts")


sys.path.insert(0, _find_babilong())

from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [R%(rank)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _import_phase1_module():
    train_path = os.path.join(PROJECT_ROOT, "scripts", "train_h_v2.py")
    spec = importlib.util.spec_from_file_location("train_h_v2", train_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_h_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


PHASE1 = _import_phase1_module()
HSeriesV2Model = PHASE1.HSeriesV2Model
init_distributed = PHASE1.init_distributed
get_lr = PHASE1.get_lr

class TaskDataset(Dataset):
    """Minimal bAbI task parser without the full ARMT dependency stack."""

    def __init__(self, dataset_path: str, max_n_facts: int | None = None):
        self.samples = []
        story_rows = []

        with open(dataset_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                phrase_num_str, text = line.split(" ", 1)
                phrase_num = int(phrase_num_str)
                if phrase_num == 1:
                    story_rows = []

                if "\t" in text:
                    question, answer, refs = text.split("\t")
                    ref_nums = [int(n) for n in refs.split() if n]
                    row = {
                        "phrase_num": phrase_num,
                        "text": question,
                        "answer": answer,
                        "reference_nums": ref_nums,
                        "is_question": True,
                    }
                    candidate_rows = [r for r in story_rows] + [row]
                    if max_n_facts is None or len(candidate_rows) <= max_n_facts:
                        self.samples.append(
                            {
                                "facts": [r["text"] for r in story_rows],
                                "question": question,
                                "answer": answer,
                                "references": [
                                    r["text"]
                                    for r in story_rows
                                    if r["phrase_num"] in ref_nums
                                ],
                            }
                        )
                else:
                    story_rows.append(
                        {
                            "phrase_num": phrase_num,
                            "text": text,
                            "answer": None,
                            "reference_nums": [],
                            "is_question": False,
                        }
                    )

    def __getitem__(self, ind: int):
        return self.samples[ind]

    def __len__(self):
        return len(self.samples)


class TokenizedPg19Sampler:
    """Sample token noise directly from the local tokenized PG19 dataset."""

    def __init__(self, dataset_path: str, split: str = "train", chunk_len: int = 64, seed: int = 42):
        import datasets as hf_datasets

        ds = hf_datasets.load_from_disk(dataset_path)
        self.data = ds[split]
        self.chunk_len = chunk_len
        self.rng = np.random.RandomState(seed)
        self.n_books = len(self.data)

    def get_sample(self, sample_size: int) -> list[list[int]]:
        if sample_size <= 0:
            return []

        chunks: list[list[int]] = []
        total = 0
        while total < sample_size:
            book_idx = self.rng.randint(0, self.n_books)
            tokens = self.data[book_idx]["tokens"]
            if len(tokens) < self.chunk_len + 1:
                continue

            span_target = min(sample_size - total + self.chunk_len, self.chunk_len * self.rng.randint(2, 5))
            span_target = min(span_target, len(tokens))
            max_offset = max(0, len(tokens) - span_target)
            offset = self.rng.randint(0, max_offset + 1) if max_offset > 0 else 0
            span = tokens[offset:offset + span_target]

            for i in range(0, len(span), self.chunk_len):
                piece = span[i:i + self.chunk_len]
                if not piece:
                    continue
                take = min(len(piece), sample_size - total)
                piece = piece[:take]
                chunks.append(piece)
                total += len(piece)
                if total >= sample_size:
                    break

        return chunks


class BabilongQADataset(Dataset):
    """Create noisy qa1 sequences with exact total length = segment_size * max_n_segments."""

    def __init__(
        self,
        babi_path: str,
        task_dataset: str,
        noise_dataset_path: str,
        tokenizer,
        segment_size: int,
        max_n_segments: int,
        split: str = "train",
        seed: int = 42,
        qa_margin: int = 64,
        max_n_facts: int | None = None,
    ):
        self.segment_size = segment_size
        self.max_n_segments = max_n_segments
        self.total_len = segment_size * max_n_segments
        self.qa_margin = qa_margin
        self.tokenizer = tokenizer
        self.rng = np.random.RandomState(seed)
        self.task_dataset = TaskDataset(
            os.path.join(babi_path, f"{task_dataset}_{split}.txt"),
            max_n_facts=max_n_facts,
        )
        self.noise_sampler = TokenizedPg19Sampler(noise_dataset_path, split="train" if split == "train" else "validation", seed=seed)
        self.prompt_task = task_dataset.split("_", 1)[0]
        if self.prompt_task not in DEFAULT_PROMPTS:
            raise ValueError(f"Unsupported bAbI prompt task: {self.prompt_task}")
        self.prompt_cfg = DEFAULT_PROMPTS[self.prompt_task]
        self.context_marker = "__CTX__"
        self.eos_token = tokenizer.eos_token_id

    def __len__(self):
        return len(self.task_dataset)

    @staticmethod
    def _sum_lengths(seqs: list[list[int]]) -> int:
        return sum(len(x) for x in seqs)

    def __getitem__(self, idx: int):
        sample = self.task_dataset[idx % len(self.task_dataset)]
        facts_tok = self.tokenizer(list(sample["facts"]), add_special_tokens=False)["input_ids"]
        answer_tok = self.tokenizer(sample["answer"], add_special_tokens=False)["input_ids"]

        prompt_text = get_formatted_input(
            self.context_marker,
            sample["question"],
            self.prompt_cfg["examples"],
            self.prompt_cfg["instruction"],
            self.prompt_cfg["post_prompt"],
            template=DEFAULT_TEMPLATE,
        )
        prompt_prefix_text, prompt_suffix_text = prompt_text.split(self.context_marker, 1)
        prompt_prefix_tok = self.tokenizer(prompt_prefix_text, add_special_tokens=False)["input_ids"]
        prompt_suffix_tok = self.tokenizer(prompt_suffix_text, add_special_tokens=False)["input_ids"]

        facts_len = self._sum_lengths(facts_tok)
        overhead = len(prompt_prefix_tok) + len(prompt_suffix_tok) + len(answer_tok) + 1  # EOS
        noise_budget = max(self.total_len - facts_len - overhead, 0)
        noise_chunks = self.noise_sampler.get_sample(noise_budget)

        possible_positions = list(range(len(noise_chunks) + 1))
        fact_positions = self.rng.choice(possible_positions, len(facts_tok), replace=True)
        fact_positions.sort()

        interleaved: list[list[list[int]]] = [[] for _ in range(len(noise_chunks) + 1)]
        for fact, pos in zip(facts_tok, fact_positions):
            interleaved[pos].append(fact)
        for i, chunk in enumerate(noise_chunks):
            interleaved[i].append(chunk)

        context_tokens = [tok for group in interleaved for seq in group for tok in seq]
        max_context_len = max(self.total_len - overhead, 0)
        if len(context_tokens) > max_context_len:
            context_tokens = context_tokens[-max_context_len:]

        input_ids = prompt_prefix_tok + context_tokens + prompt_suffix_tok + answer_tok + [self.eos_token]
        labels = [-100] * (len(prompt_prefix_tok) + len(context_tokens) + len(prompt_suffix_tok)) + answer_tok + [self.eos_token]

        if len(input_ids) > self.total_len:
            overflow = len(input_ids) - self.total_len
            keep_context_from = min(overflow, len(context_tokens))
            context_tokens = context_tokens[keep_context_from:]
            input_ids = prompt_prefix_tok + context_tokens + prompt_suffix_tok + answer_tok + [self.eos_token]
            labels = [-100] * (len(prompt_prefix_tok) + len(context_tokens) + len(prompt_suffix_tok)) + answer_tok + [self.eos_token]
        elif len(input_ids) < self.total_len:
            pad = self.total_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad
            labels = labels + [-100] * pad

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="H-series v2 Phase 2 BABILong training")

    parser.add_argument("--base_model", type=str, default="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B")
    parser.add_argument("--memory_variant", type=str, choices=["A", "B", "D"], default="A")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", action="store_true", default=False)

    parser.add_argument("--num_slots", type=int, default=64)
    parser.add_argument("--memory_write_layer", type=int, default=8)
    parser.add_argument("--memory_read_layers", type=str, default="10,12,14")
    parser.add_argument("--write_lr", type=float, default=0.1)
    parser.add_argument("--residual_scale", type=float, default=0.01)
    parser.add_argument("--use_dual_gate", action="store_true", default=True)
    parser.add_argument("--no_dual_gate", action="store_true", default=False)
    parser.add_argument("--forget_bias_init", type=float, default=1.0)
    parser.add_argument("--input_bias_init", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)

    parser.add_argument("--noise_dataset_path", type=str, default="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/armt_pg19_real_tokenized_full")
    parser.add_argument("--babi_path", type=str, default="/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong/data/tasks_1-20_v1-2/en-10k")
    parser.add_argument("--task_dataset", type=str, default="qa1_single-supporting-fact")
    parser.add_argument("--segment_size", type=int, default=512)
    parser.add_argument("--max_n_segments", type=int, default=2)
    parser.add_argument("--max_n_facts", type=int, default=None)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    parser.add_argument("--output_dir", type=str, default="outputs/h_v2_phase2_A")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_checkpoint", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.no_freeze_backbone:
        args.freeze_backbone = False
    if args.no_dual_gate:
        args.use_dual_gate = False

    if not os.path.exists(args.babi_path):
        raise FileNotFoundError(f"Missing bAbI data dir: {args.babi_path}")

    rank, world_size, local_rank = init_distributed()
    is_main = rank == 0

    for handler in logging.root.handlers:
        handler.setFormatter(logging.Formatter(f"%(asctime)s [R{rank}] %(levelname)s %(message)s"))

    device = torch.device(f"cuda:{local_rank}")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main:
        logger.info("=" * 60)
        logger.info("H-series v2 Phase 2: variant=%s, task=%s, segments=%d", args.memory_variant, args.task_dataset, args.max_n_segments)
        logger.info("=" * 60)
        logger.info("Args: %s", vars(args))
        logger.info("Loading base model: %s", args.base_model)

    base_model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map={"": device},
    )

    model = HSeriesV2Model(
        base_model=base_model,
        memory_variant=args.memory_variant,
        num_slots=args.num_slots,
        segment_size=args.segment_size,
        max_n_segments=args.max_n_segments,
        freeze_backbone=args.freeze_backbone,
        no_loss_from_first_segment=False,
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

    ckpt = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if is_main and missing:
        logger.info("Resume missing keys (%d): %s", len(missing), missing[:5])
    if is_main and unexpected:
        logger.info("Resume unexpected keys (%d): %s", len(unexpected), unexpected[:5])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main:
        logger.info("Trainable: %d / %d (%.4f%%)", trainable, total, 100.0 * trainable / total)

    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        ddp_model = model

    train_dataset = BabilongQADataset(
        babi_path=args.babi_path,
        task_dataset=args.task_dataset,
        noise_dataset_path=args.noise_dataset_path,
        tokenizer=tokenizer,
        segment_size=args.segment_size,
        max_n_segments=args.max_n_segments,
        split="train",
        seed=args.seed,
        max_n_facts=args.max_n_facts,
    )

    if world_size > 1:
        # seed=args.seed is LOAD-BEARING: DistributedSampler.__iter__ builds its OWN
        # generator (g.manual_seed(self.seed + self.epoch)) and self.seed defaults to 0,
        # so torch.manual_seed()/set_seed() CANNOT reach it. Without this argument every
        # --seed value gives a BYTE-IDENTICAL data order. Do not delete as redundant.
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True,
                                           seed=args.seed)
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

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    accum_loss = 0.0
    accum_count = 0

    if is_main:
        logger.info(
            "Starting Phase 2 training: max_steps=%d, grad_accum=%d, effective_batch=%d",
            args.max_steps,
            args.gradient_accumulation_steps,
            args.batch_size * args.gradient_accumulation_steps * world_size,
        )

    train_iter = iter(train_loader)
    start_time = time.time()

    while global_step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            if train_sampler is not None:
                train_sampler.set_epoch(global_step)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        result = ddp_model(input_ids, labels)
        loss = result["loss"] if isinstance(result, dict) else result[0]
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        accum_loss += loss.item() * args.gradient_accumulation_steps
        accum_count += 1

        if accum_count >= args.gradient_accumulation_steps:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)

            lr = get_lr(global_step, args.warmup_steps, args.max_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            avg_loss = accum_loss / accum_count

            if is_main and global_step % args.log_every == 0:
                elapsed = time.time() - start_time
                logger.info("step=%d loss=%.4f lr=%.2e elapsed=%.1fs", global_step, avg_loss, lr, elapsed)

            if is_main and global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_{global_step}.pt")
                torch.save(
                    {
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    ckpt_path,
                )
                logger.info("Saved checkpoint: %s", ckpt_path)

            accum_loss = 0.0
            accum_count = 0

    if is_main:
        ckpt_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        torch.save(
            {
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            ckpt_path,
        )
        logger.info("Training complete. Final checkpoint: %s", ckpt_path)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
