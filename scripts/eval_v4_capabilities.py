"""Evaluate trained v4 ChunkMemory model capabilities.

Three tests:
1. Base model PPL on WikiText (pure Llama-3-8B, no LoRA)
2. LoRA-only PPL (LoRA loaded, no memory banks)
3. LoRA + Memory PPL (LoRA + active memory banks)

Plus text generation from all three configs.
"""

import argparse
import math
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LlamaForCausalLM
from peft import LoraConfig, TaskType, get_peft_model

# Reuse ChunkMemoryModel and helpers from the training script.
sys.path.insert(0, os.path.dirname(__file__))
from train_v4_chunk_memory import ChunkMemoryModel, make_prefix_causal_mask, extend_position_embeddings


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


class DocDataset(Dataset):
    """Groups chunks into documents for memory-bank evaluation."""
    def __init__(self, npy_path, seq_len, skip, max_chunks, chunks_per_doc):
        d = np.load(npy_path, mmap_mode="r")
        self.data = d[skip: skip + max_chunks].astype(np.int32)
        self.seq_len = seq_len
        self.chunks_per_doc = chunks_per_doc
        self.n_docs = max(1, len(self.data) // chunks_per_doc)

    def __len__(self):
        return self.n_docs

    def __getitem__(self, idx):
        start = idx * self.chunks_per_doc
        end = start + self.chunks_per_doc
        chunks = []
        for i in range(start, min(end, len(self.data))):
            tokens = torch.tensor(self.data[i], dtype=torch.long)[:self.seq_len]
            chunks.append({"input_ids": tokens, "labels": tokens.clone()})
        while len(chunks) < self.chunks_per_doc:
            tokens = torch.zeros(self.seq_len, dtype=torch.long)
            chunks.append({"input_ids": tokens, "labels": torch.full_like(tokens, -100)})
        return {"chunks": chunks}


@torch.no_grad()
def eval_base_ppl(model, loader, device, max_chunks=200):
    """Compute PPL for base model (no LoRA, no memory)."""
    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    for batch in loader:
        if count >= max_chunks:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs.loss.detach()
        if not torch.isfinite(loss):
            continue
        n_tok = (labels != -100).sum()
        total_loss += loss.double() * n_tok.double()
        total_tokens += n_tok.double()
        count += 1
    avg_loss = (total_loss / total_tokens).item()
    ppl = math.exp(avg_loss)
    return ppl, int(total_tokens.item())


@torch.no_grad()
def eval_lora_only_ppl(cm_model, loader, device, max_chunks=200):
    """Compute PPL with LoRA but no memory (banks reset before each chunk)."""
    cm_model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)
    count = 0
    for batch in loader:
        if count >= max_chunks:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        cm_model.reset_banks()
        out = cm_model.forward_chunk(input_ids, labels=labels)
        loss = out["loss"].detach()
        if not torch.isfinite(loss):
            continue
        n_tok = (labels != -100).sum()
        total_loss += loss.double() * n_tok.double()
        total_tokens += n_tok.double()
        count += 1
    avg_loss = (total_loss / total_tokens).item()
    ppl = math.exp(avg_loss)
    return ppl, int(total_tokens.item())


@torch.no_grad()
def eval_lora_memory_ppl(cm_model, loader, device, chunks_per_doc):
    """Compute PPL with LoRA + memory banks active across chunks_per_doc."""
    cm_model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    for doc_idx, batch in enumerate(loader):
        chunks = batch["chunks"]
        cm_model.reset_banks()
        for chunk in chunks:
            input_ids = chunk["input_ids"].unsqueeze(0).to(device)
            labels = chunk["labels"].unsqueeze(0).to(device)
            if (labels != -100).sum() == 0:
                continue
            out = cm_model.forward_chunk(input_ids, labels=labels)
            loss = out["loss"].detach()
            if not torch.isfinite(loss):
                continue
            n_tok = (labels != -100).sum()
            total_loss += loss.double() * n_tok.double()
            total_tokens += n_tok.double()

    avg_loss = (total_loss / total_tokens).item()
    ppl = math.exp(avg_loss)
    return ppl, int(total_tokens.item())


@torch.no_grad()
def generate_text(model, tokenizer, prompts, device, max_new_tokens=100, use_memory=False, cm_model=None):
    """Generate text from prompts."""
    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        if use_memory and cm_model is not None:
            cm_model.reset_banks()
            # Use model.generate won't work with ChunkMemoryModel directly.
            # Use forward_chunk autoregressively.
            generated_ids = inputs["input_ids"].tolist()[0]
            current_ids = inputs["input_ids"]
            for _ in range(max_new_tokens):
                out = cm_model.forward_chunk(current_ids)
                logits = out["logits"][:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated_ids.append(next_token.item())
                current_ids = next_token
                if next_token.item() == tokenizer.eos_token_id:
                    break
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        results.append({"prompt": prompt, "generated": text})
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate v4 ChunkMemory model capabilities")
    parser.add_argument("--model", type=str, default="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b")
    parser.add_argument("--data", type=str, default="data/wikitext_chunks_llama3_4096.npy")
    parser.add_argument("--checkpoint", type=str, default="outputs/v4_ablation_slots4_lr3e5/final.pt",
                        help="Path to trained checkpoint")
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--chunks_per_doc", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--max_eval_chunks", type=int, default=200,
                        help="Number of chunks for flat PPL evaluation")
    parser.add_argument("--max_eval_docs", type=int, default=10,
                        help="Number of documents for memory PPL evaluation")
    parser.add_argument("--skip_chunks", type=int, default=600,
                        help="Skip first N chunks (avoid train/eval overlap)")
    parser.add_argument("--output_dir", type=str, default="outputs/v4_capability_eval")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("V4 ChunkMemory Capability Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: slots={args.num_slots}, top_k={args.top_k}, lora_rank={args.lora_rank}")
    print(f"Data: {args.data}, seq_len={args.seq_len}")
    print(f"Eval: {args.max_eval_chunks} flat chunks, {args.max_eval_docs} docs x {args.chunks_per_doc} chunks/doc")
    print()

    # Load tokenizer.
    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ==================================================================
    # TEST 1: Base model PPL (no LoRA, no memory)
    # ==================================================================
    print("[2/5] Loading base model (no LoRA)...")
    base_model = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map={"": device}
    )
    base_model.eval()

    flat_ds = FlatChunkDataset(args.data, args.seq_len, args.skip_chunks, args.max_eval_chunks)
    flat_loader = DataLoader(flat_ds, batch_size=1, shuffle=False, num_workers=0)

    print("  Computing base model WikiText PPL...")
    t0 = time.time()
    base_ppl, base_tokens = eval_base_ppl(base_model, flat_loader, device)
    print(f"  Base model PPL: {base_ppl:.4f} ({base_tokens} tokens, {time.time()-t0:.1f}s)")
    print()

    # ==================================================================
    # TEST 2: LoRA-only PPL (LoRA loaded, no memory)
    # ==================================================================
    print("[3/5] Loading LoRA model (no memory)...")
    # Delete base_model to free memory, load fresh for LoRA.
    del base_model
    torch.cuda.empty_cache()

    fresh_base = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map={"": device}
    )
    cm_model = ChunkMemoryModel(
        base_model=fresh_base,
        num_slots=args.num_slots,
        lora_rank=args.lora_rank,
        top_k=args.top_k,
        epsilon=args.epsilon,
    ).to(device)

    # Load checkpoint.
    ckpt = torch.load(args.checkpoint, map_location=device)
    cm_model.peft_model.load_state_dict(ckpt['lora_state_dict'])
    print(f"  Loaded checkpoint (step {ckpt.get('global_step', '?')})")
    cm_model.eval()

    print("  Computing LoRA-only WikiText PPL (banks reset)...")
    t0 = time.time()
    lora_ppl, lora_tokens = eval_lora_only_ppl(cm_model, flat_loader, device, args.max_eval_chunks)
    print(f"  LoRA-only PPL: {lora_ppl:.4f} ({lora_tokens} tokens, {time.time()-t0:.1f}s)")
    print()

    # ==================================================================
    # TEST 3: LoRA + Memory PPL
    # ==================================================================
    print("[4/5] Computing LoRA+Memory WikiText PPL (banks active across chunks)...")
    doc_ds = DocDataset(
        args.data, args.seq_len, args.skip_chunks,
        args.max_eval_docs * args.chunks_per_doc, args.chunks_per_doc
    )
    doc_loader = DataLoader(doc_ds, batch_size=1, shuffle=False, num_workers=0,
                            collate_fn=lambda b: b[0])

    t0 = time.time()
    mem_ppl, mem_tokens = eval_lora_memory_ppl(cm_model, doc_loader, device, args.chunks_per_doc)
    print(f"  LoRA+Memory PPL: {mem_ppl:.4f} ({mem_tokens} tokens, {time.time()-t0:.1f}s)")
    print()

    # ==================================================================
    # TEST 4: Text generation comparison
    # ==================================================================
    print("[5/5] Text generation comparison...")

    prompts = [
        "The history of artificial intelligence began in the",
        "In a surprising finding, researchers discovered that",
        "The most important thing to remember about machine learning is",
    ]

    # Generation: Base model (need to reload).
    # Reuse cm_model's peft base for LoRA generation.
    # For base generation, we need the raw model. Use cm_model's internal base.
    print("  Generating from LoRA model (no memory)...")
    lora_results = generate_text(
        cm_model.peft_model, tokenizer, prompts, device,
        max_new_tokens=80, use_memory=False
    )

    print("  Generating from LoRA+Memory model...")
    mem_results = generate_text(
        None, tokenizer, prompts, device,
        max_new_tokens=80, use_memory=True, cm_model=cm_model
    )

    # Also need base model generation for comparison. Reload.
    print("  Loading base model for generation comparison...")
    del cm_model
    torch.cuda.empty_cache()
    base_gen_model = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map={"": device}
    )
    base_gen_model.eval()
    print("  Generating from base model...")
    base_results = generate_text(
        base_gen_model, tokenizer, prompts, device,
        max_new_tokens=80, use_memory=False
    )
    del base_gen_model
    torch.cuda.empty_cache()

    # ==================================================================
    # Summary
    # ==================================================================
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Base model PPL:       {base_ppl:.4f}")
    print(f"  LoRA-only PPL:        {lora_ppl:.4f}  (ratio vs base: {lora_ppl/base_ppl:.4f})")
    print(f"  LoRA+Memory PPL:      {mem_ppl:.4f}  (ratio vs base: {mem_ppl/base_ppl:.4f})")
    print()

    # Generation comparison.
    print("-" * 70)
    print("GENERATION COMPARISON")
    print("-" * 70)
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt: \"{prompt}\"")
        print(f"  Base:       {base_results[i]['generated'][len(prompt):].strip()[:200]}")
        print(f"  LoRA:       {lora_results[i]['generated'][len(prompt):].strip()[:200]}")
        print(f"  LoRA+Mem:   {mem_results[i]['generated'][len(prompt):].strip()[:200]}")

    # Save results.
    results = {
        "checkpoint": args.checkpoint,
        "config": {
            "num_slots": args.num_slots,
            "top_k": args.top_k,
            "lora_rank": args.lora_rank,
            "chunks_per_doc": args.chunks_per_doc,
        },
        "base_ppl": base_ppl,
        "lora_only_ppl": lora_ppl,
        "lora_memory_ppl": mem_ppl,
        "lora_vs_base_ratio": lora_ppl / base_ppl,
        "memory_vs_base_ratio": mem_ppl / base_ppl,
        "memory_vs_lora_ratio": mem_ppl / lora_ppl,
        "base_tokens": base_tokens,
        "lora_tokens": lora_tokens,
        "mem_tokens": mem_tokens,
        "generation": {
            "base": base_results,
            "lora": lora_results,
            "memory": mem_results,
        },
    }
    out_path = os.path.join(args.output_dir, "capability_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
