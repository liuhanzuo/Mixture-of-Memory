#!/usr/bin/env python3
"""PPL eval for OLMo-2 arch-probe#2 (prune-heal) models.

OLMo-2 port of scripts/eval_qwen_ppl.py. Two modes:
  * C0 / full base : --model_path <OLMo-2-1124-7B> (no --ckpt) -> load the full
    32-layer OLMo-2-7B with Olmo2ForCausalLM.from_pretrained. Upper reference.
  * arm ckpt       : --ckpt <arm>.pt (+ --keep_front / --n_fresh, or read from
    the ckpt meta) -> rebuild the (keep_front + n_fresh)-layer model and
    load_state_dict(strict=True). Evaluates a trained (healed) arm.

The ckpt meta (written by scripts/train_olmo2_arch_probe2.py) carries
model_family="olmo2", base_model_path, keep_front_layers, n_fresh_layers,
num_hidden_layers, hidden_size, vocab_size, tie_word_embeddings=False, and the
raw model_state -- enough to rebuild an identical model.

PPL measurement matches scripts/eval_qwen_ppl.py EXACTLY:
  * NO pre-shift: labels = input_ids.clone(); Olmo2ForCausalLM.forward(labels=)
    does its own internal shift. Pre-shifting is a double-shift bug.
  * per-chunk contribution = loss * (seq_len - 1); ppl = exp(sum / total_tokens).

OLMo-2 vs Qwen3 config difference: OLMo-2 is POST-norm and has NO `layer_types`
field, so we DO NOT set cfg.layer_types (Olmo2Config has no such attribute); we
only reset it defensively if a future transformers version adds one. Everything
else (fp32 config, use_cache=False, bf16 autocast forward) is analogous.

Defaults: seq_len 2048, max_chunks 200, val = /dev/shm/dolmino_now_val.npy
(held-out, shape (4096, 2048) uint32).
"""
import os
import sys
import math
import argparse
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Olmo2Config, Olmo2ForCausalLM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NumpyEvalDataset(Dataset):
    def __init__(self, data_path, seq_length=2048, skip_chunks=0, max_chunks=None):
        logger.info(f"Loading numpy data from {data_path}...")
        self.data = np.load(data_path, mmap_mode="r")
        assert self.data.shape[1] >= seq_length, (self.data.shape, seq_length)
        self.seq_length = seq_length
        end = len(self.data) if max_chunks is None else min(skip_chunks + max_chunks, len(self.data))
        self.data = self.data[skip_chunks:end]
        logger.info(f"Loaded {len(self.data)} eval chunks (skip={skip_chunks}, max={max_chunks}, "
                    f"seq_len={seq_length})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # NO pre-shift (see module docstring): Olmo2ForCausalLM shifts internally.
        # val npy is uint32 -> cast to int64 for token ids.
        row = np.asarray(self.data[idx, : self.seq_length]).astype(np.int64)
        tokens = torch.from_numpy(row)
        return {"input_ids": tokens, "labels": tokens.clone()}


def _build_from_ckpt(ckpt_path, model_path, keep_front, n_fresh, device):
    """Rebuild a (keep_front + n_fresh)-layer OLMo-2 and load the arm state_dict.

    keep_front / n_fresh default to the values stored in the ckpt meta; CLI
    overrides win if provided (>0)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    kf = keep_front if keep_front and keep_front > 0 else ck.get("keep_front_layers")
    nf = n_fresh if n_fresh and n_fresh > 0 else ck.get("n_fresh_layers")
    assert kf is not None and nf is not None, (
        "keep_front / n_fresh not in ckpt meta; pass --keep_front and --n_fresh"
    )
    total = kf + nf
    cfg = Olmo2Config.from_pretrained(model_path, local_files_only=True)
    cfg.num_hidden_layers = total
    # OLMo-2 config has NO layer_types (POST-norm); reset defensively only if a
    # future transformers version adds it so a stale length cannot leak through.
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = ["full_attention"] * total
        assert len(cfg.layer_types) == total
    cfg.use_cache = False
    model = Olmo2ForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=True)
    logger.info(f"[ckpt] rebuilt keep_front={kf} n_fresh={nf} total={total} layers; "
                f"loaded state_dict (missing={len(missing)} unexpected={len(unexpected)})")
    model = model.to(device=device, dtype=torch.bfloat16)
    return model, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B",
                        help="OLMo-2-7B path: base model (C0) and/or config source for ckpt rebuild")
    parser.add_argument("--ckpt", type=str, default="",
                        help="arm checkpoint .pt; empty -> eval full base model (C0)")
    parser.add_argument("--keep_front", type=int, default=0, help="override ckpt meta (0=use meta)")
    parser.add_argument("--n_fresh", type=int, default=0, help="override ckpt meta (0=use meta)")
    parser.add_argument("--data_path", type=str, default="/dev/shm/dolmino_now_val.npy")
    parser.add_argument("--skip_chunks", type=int, default=0)
    parser.add_argument("--max_chunks", type=int, default=200)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.ckpt:
        logger.info(f"Loading arm ckpt from {args.ckpt} (config from {args.model_path})...")
        model, total_layers = _build_from_ckpt(
            args.ckpt, args.model_path, args.keep_front, args.n_fresh, device)
        tag = f"ckpt({total_layers}L):{os.path.basename(args.ckpt)}"
    else:
        logger.info(f"Loading FULL base OLMo-2 from {args.model_path} (C0)...")
        model = Olmo2ForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            local_files_only=True, device_map={"": device})
        tag = f"C0-full({model.config.num_hidden_layers}L)"
    model.eval()
    model.config.use_cache = False

    dataset = NumpyEvalDataset(args.data_path, args.seq_length, args.skip_chunks, args.max_chunks)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    total_loss = 0.0
    total_tokens = 0
    seq_len = args.seq_length - 1

    logger.info(f"Evaluating {len(dataset)} chunks [{tag}] on {device}...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.item()
            total_loss += loss * seq_len
            total_tokens += seq_len
            if (i + 1) % 20 == 0:
                cumul_ppl = math.exp(total_loss / total_tokens)
                logger.info(f"  chunk {i+1}/{len(dataset)}: loss={loss:.4f}, cumul_ppl={cumul_ppl:.4f}")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    logger.info(f"RESULT [{tag}]: PPL={ppl:.4f}, avg_loss={avg_loss:.6f}, tokens={total_tokens}")
    print(f"\nOLMo-2 arch-probe#2 PPL [{tag}]: {ppl:.4f}")


if __name__ == "__main__":
    main()
