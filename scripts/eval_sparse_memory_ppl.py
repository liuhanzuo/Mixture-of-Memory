"""Evaluate SparseMemory model PPL on PG-19."""
import os, sys, json, math, argparse, logging
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.memory.sparse_memory.model import SparseMemoryLlamaForCausalLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PreTokenizedEvalDataset(Dataset):
    """Load pre-tokenized chunks from numpy file.  Fast, no tokenizer needed.

    Parameters
    ----------
    data_path : str
        Path to ``*.npy`` file with shape ``(N, seq_length)`` (uint16 or int).
    skip_chunks : int
        Skip the first *skip_chunks* chunks (to avoid training-data overlap).
    max_chunks : int
        Only use at most *max_chunks* chunks for evaluation.
    """
    def __init__(self, data_path, seq_length=4096, skip_chunks=0, max_chunks=None):
        import numpy as np
        logger.info(f"Loading numpy data from {data_path}...")
        self.data = np.load(data_path)  # shape [N, seq_length]
        assert self.data.shape[1] == seq_length, f"Expected seq_length={seq_length}, got {self.data.shape[1]}"
        end = len(self.data) if max_chunks is None else min(skip_chunks + max_chunks, len(self.data))
        self.data = self.data[skip_chunks:end]
        logger.info(f"Loaded {len(self.data)} eval chunks (skip={skip_chunks}, max={max_chunks}, total_in_file={end-skip_chunks})")

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


def load_sparse_model(checkpoint_path, device, memory_slots=128, top_k=8, sliding_window=256, ema_alpha=0.1):
    """Load base Llama from pretrained, wrap with SparseMemory, load checkpoint weights.

    The checkpoint saves the inner model via save_pretrained() after SparseMemory
    wrapper replaces self_attn. So keys are like:
      model.layers.0.self_attn.original_attn.q_proj.weight
      model.layers.0.self_attn.gate_proj.weight
    This matches SparseMemoryLlamaForCausalLM's structure, so we load pretrained
    first, wrap, then load the checkpoint directly.
    """
    # Find original pretrained model (sibling of checkpoint or default)
    pretrained_path = os.path.join(os.path.dirname(checkpoint_path), "..", "models", "Llama--Llama2-7b")
    if not os.path.exists(pretrained_path):
        pretrained_path = os.path.join(os.path.dirname(__file__), "..", "models", "Llama--Llama2-7b")
    if not os.path.exists(pretrained_path):
        pretrained_path = "/root/Mixture-of-Memory/models/Llama--Llama2-7b"
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Cannot find pretrained model. Tried: {pretrained_path}")
    logger.info(f"Loading pretrained Llama from {pretrained_path}...")
    model = SparseMemoryLlamaForCausalLM(
        base_model=pretrained_path,
        memory_slots=memory_slots,
        top_k=top_k,
        sliding_window=sliding_window,
        ema_alpha=ema_alpha,
        torch_dtype=torch.bfloat16,
    )

    # Load fine-tuned weights from checkpoint (keys match SparseMemory structure)
    logger.info(f"Loading fine-tuned weights from {checkpoint_path}...")
    from safetensors.torch import load_file as st_load
    import glob
    st_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
    if st_files:
        ckpt_sd = {}
        for f in st_files:
            ckpt_sd.update(st_load(f, device="cpu"))
        # Remap keys: checkpoint has "model.layers.X..." prefix
        # Our model's state_dict has same prefix via self.model
        model_sd = model.state_dict()
        remapped = {}
        for k, v in ckpt_sd.items():
            # Keys like model.layers.0.self_attn.original_attn.q_proj.weight
            # should map to model.model.layers.0.self_attn.original_attn.q_proj.weight
            if k.startswith("model.") and not k.startswith("model.model."):
                new_key = "model.model." + k[len("model."):]
            else:
                logger.warning(f"  Unexpected key format: {k}")
                continue
            # Skip memory_bank dynamic state (batch-size dependent, re-initialized at forward)
            if "memory_bank.memory" in new_key or "memory_bank.write_index" in new_key:
                continue
            if new_key in model_sd:
                remapped[new_key] = v
            else:
                logger.warning(f"  Key {k} -> {new_key} not found in model")
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing:
            logger.warning(f"  Missing keys (using pretrained): {missing[:5]}...")
        if unexpected:
            logger.warning(f"  Unexpected keys (ignored): {unexpected[:5]}...")
        logger.info(f"Loaded {len(remapped)} parameters from checkpoint.")
    else:
        logger.warning(f"No safetensors found in {checkpoint_path} — using pretrained weights only!")

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate SparseMemory PPL")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--data_path", type=str, default="data/pg19_chunks.npy")
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--max_chunks", type=int, default=200, help="Max eval chunks")
    parser.add_argument("--skip_chunks", type=int, default=40000, help="Skip first N chunks (avoid training overlap)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--memory_slots", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--sliding_window", type=int, default=256)
    parser.add_argument("--ema_alpha", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    model = load_sparse_model(args.checkpoint, device, args.memory_slots, args.top_k, args.sliding_window, args.ema_alpha)
    model = model.to(device)
    model.eval()

    dataset = PreTokenizedEvalDataset(
        args.data_path,
        seq_length=args.seq_length,
        skip_chunks=args.skip_chunks,
        max_chunks=args.max_chunks,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    total_loss = 0.0
    total_tokens = 0
    seq_len = args.seq_length - 1  # after shift

    logger.info(f"Evaluating PPL on {len(dataset)} chunks (seq_length={args.seq_length}, gpu={args.gpu})...")
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
            logger.info(f"  chunk {i+1}/{len(dataset)}: loss={loss:.4f}, cumul_ppl={cumul_ppl:.4f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    logger.info(f"RESULT: PPL={ppl:.4f}, avg_loss={avg_loss:.6f}, tokens={total_tokens}, docs={len(dataset)}")

    result = {
        "checkpoint": args.checkpoint,
        "perplexity": ppl,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_docs": len(dataset),
        "seq_length": args.seq_length,
        "skip_chunks": args.skip_chunks,
        "timestamp": datetime.now().isoformat(),
    }

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "sparse_memory_ppl_results.json")
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved to {out_path}")
    else:
        out_path = os.path.join(args.checkpoint, "eval_ppl_results.json")
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved to {out_path}")

    print(f"\nSparseMemory PPL: {ppl:.4f}")


if __name__ == "__main__":
    main()
