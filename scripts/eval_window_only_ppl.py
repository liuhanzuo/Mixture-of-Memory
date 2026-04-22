#!/usr/bin/env python3
"""Evaluate sliding-window-only baseline PPL on PG-19.

Patches LlamaModel.forward to inject sliding window mask via and_mask_function
(same as training). Uses default SDPA attention.
"""
import os, sys, math, argparse, logging, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WINDOW_SIZE = 256
_original_llama_forward = None


def patch_llama_model_for_window(window: int):
    """Patch LlamaModel.forward to inject sliding window mask directly.

    Uses eager attention with a precomputed causal+window mask tensor.
    """
    from transformers.models.llama.modeling_llama import LlamaModel as _LM
    global _original_llama_forward
    _original_llama_forward = _LM.forward

    _window_mask_cache = {}

    def _make_mask(seq_len, win, device):
        q = torch.arange(seq_len, device=device).unsqueeze(1)
        k = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = ((k <= q) & (k >= (q - win))).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
        return torch.where(
            mask > 0,
            torch.tensor(0.0, device=device, dtype=torch.bfloat16),
            torch.tensor(torch.finfo(torch.bfloat16).min, device=device, dtype=torch.bfloat16)
        )

    def _patched_forward(self, *args, **kwargs):
        dev = next(self.parameters()).device
        if dev not in _window_mask_cache:
            _window_mask_cache[dev] = _make_mask(window, window, dev)
        kwargs['attention_mask'] = _window_mask_cache[dev]
        return _original_llama_forward(self, *args, **kwargs)

    _LM.forward = _patched_forward
    logger.info(f"Patched LlamaModel.forward with sliding_window={window} (eager attention)")


# ── Dataset ──────────────────────────────────────────────────────────────────
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


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate sliding-window baseline PPL")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", default="data/pg19_chunks.npy")
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--max_chunks", type=int, default=200)
    parser.add_argument("--skip_chunks", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sliding_window", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Patch BEFORE loading model
    patch_llama_model_for_window(args.sliding_window)

    device = torch.device(f"cuda:{args.gpu}")
    logger.info(f"Loading model from {args.model_path} (eager attention)...")
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    dataset = NumpyEvalDataset(args.data_path, args.seq_length, args.skip_chunks, args.max_chunks)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0
    seq_len = args.seq_length - 1

    logger.info(f"Evaluating {len(dataset)} chunks on GPU {args.gpu}...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.item()
            total_loss += loss * seq_len
            total_tokens += seq_len
            cumul_ppl = math.exp(total_loss / total_tokens)
            if (i + 1) % 10 == 0:
                logger.info(f"  chunk {i+1}/{len(dataset)}: loss={loss:.4f}, cumul_ppl={cumul_ppl:.4f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    logger.info(f"RESULT: PPL={ppl:.4f}, avg_loss={avg_loss:.6f}, tokens={total_tokens}, docs={len(dataset)}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        result = {"ppl": ppl, "avg_loss": avg_loss, "tokens": total_tokens, "chunks": len(dataset),
                  "model": args.model_path, "window": args.sliding_window}
        with open(os.path.join(args.output_dir, "window_only_ppl_results.json"), "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved to {args.output_dir}/window_only_ppl_results.json")

    print(f"\nSliding-Window-Only Baseline PPL: {ppl:.4f}")


if __name__ == "__main__":
    main()
