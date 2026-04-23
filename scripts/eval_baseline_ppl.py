"""Quick vanilla Llama2-7B PPL eval (no memory wrapper)."""
import os, sys, math, argparse, logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PreTokenizedEvalDataset(torch.utils.data.Dataset):
    def __init__(self, npy_path, seq_length=4096, skip_chunks=40000, max_chunks=200):
        data = np.load(npy_path, mmap_mode="r")
        self.data = data[skip_chunks:skip_chunks + max_chunks].astype(np.int32)
        self.seq_length = seq_length
        logger.info(f"Loaded {len(self.data)} chunks of {self.seq_length} tokens")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self.data[idx], dtype=torch.long)
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}

def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/pg19_chunks.npy")
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--skip_chunks", type=int, default=40000)
    parser.add_argument("--max_chunks", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}")
    logger.info(f"Loading model from {args.model_path}...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params/1e9:.2f}B params")
    
    dataset = PreTokenizedEvalDataset(args.data_path, args.seq_length, args.skip_chunks, args.max_chunks)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    total_loss = 0.0
    total_tokens = 0
    
    logger.info(f"Evaluating PPL on {len(dataset)} chunks...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            n_tokens = (labels != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            
            if (i + 1) % 50 == 0:
                logger.info(f"  chunk {i+1}/{len(dataset)}: loss={loss.item():.4f}, cumul_ppl={math.exp(total_loss/total_tokens):.4f}")
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    logger.info(f"RESULT: PPL={ppl:.4f}, avg_loss={avg_loss:.6f}, tokens={total_tokens}, docs={len(dataset)}")
    print(f"\nVanilla Llama2-7B PPL: {ppl:.4f}")

if __name__ == "__main__":
    main()
