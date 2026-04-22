#!/usr/bin/env python3
"""Quick PPL eval for base Llama2-7B (no SparseMemory wrapper)."""
import os, sys, math, argparse, logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NumpyEvalDataset(Dataset):
    def __init__(self, data_path, seq_length=4096, skip_chunks=0, max_chunks=None):
        logger.info(f"Loading numpy data from {data_path}...")
        self.data = np.load(data_path)
        assert self.data.shape[1] == seq_length
        end = len(self.data) if max_chunks is None else min(skip_chunks + max_chunks, len(self.data))
        self.data = self.data[skip_chunks:end]
        logger.info(f"Loaded {len(self.data)} eval chunks (skip={skip_chunks}, max={max_chunks})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.data[idx], dtype=torch.long)
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--skip_chunks", type=int, default=40000)
    parser.add_argument("--max_chunks", type=int, default=100)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    logger.info(f"Loading base Llama from {args.model_path}...")
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    dataset = NumpyEvalDataset(args.data_path, args.seq_length, args.skip_chunks, args.max_chunks)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    total_loss = 0.0
    total_tokens = 0
    seq_len = args.seq_length - 1

    logger.info(f"Evaluating {len(dataset)} chunks on GPU {args.gpu}...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.item()
            total_loss += loss * seq_len
            total_tokens += seq_len
            cumul_ppl = math.exp(total_loss / total_tokens)
            if (i + 1) % 10 == 0:
                logger.info(f"  chunk {i+1}/{len(dataset)}: loss={loss:.4f}, cumul_ppl={cumul_ppl:.4f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    logger.info(f"RESULT: PPL={ppl:.4f}, avg_loss={avg_loss:.6f}, tokens={total_tokens}")
    print(f"\nBase Llama PPL: {ppl:.4f}")

if __name__ == "__main__":
    main()
