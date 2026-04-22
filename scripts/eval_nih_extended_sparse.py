#!/usr/bin/env python3
"""Run NIH-Extended eval with SparseMemoryModel by monkey-patching load_model."""
import sys, os, argparse
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch
from src.memory.sparse import SparseMemoryModel

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_with_sparse(model_path, device, dtype_str="bfloat16",
                           num_slots=128, window_size=256, top_k=8,
                           layers_to_patch=None):
    dtype = getattr(torch, dtype_str, torch.bfloat16)
    model_path = os.path.realpath(model_path)
    print(f"[sparse] Loading base model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype, device_map={"": device}
    )
    model.eval()

    print(f"[sparse] Wrapping with SparseMemoryModel (slots={num_slots}, window={window_size}, top_k={top_k})")
    sparse_model = SparseMemoryModel(
        model,
        layers_to_patch=layers_to_patch,
        num_slots=num_slots,
        window_size=window_size,
        top_k=top_k,
    )
    sparse_model.to(device)
    # Cast all sparse memory params to model dtype to match bfloat16
    sparse_model = sparse_model.to(dtype)
    sparse_model.eval()

    print(f"[sparse] Model ready: {model.config.model_type}, hidden={model.config.hidden_size}, "
          f"layers={model.config.num_hidden_layers}, sparse_memory=True")
    return sparse_model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config_filter", type=str, default="")
    parser.add_argument("--num_slots", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--layers_to_patch", type=str, default=None)
    args = parser.parse_args()

    layers = None
    if args.layers_to_patch:
        layers = [int(x) for x in args.layers_to_patch.split(",")]

    # Pre-load model with sparse memory
    model, tokenizer = load_model_with_sparse(
        args.model_path, args.device, args.dtype,
        num_slots=args.num_slots, window_size=args.window_size, top_k=args.top_k,
        layers_to_patch=layers,
    )

    # Monkey-patch the eval module's load_model
    import importlib
    nih_mod = importlib.import_module("scripts.eval_nih_extended")
    nih_mod.load_model = lambda mp, dev, dt="bfloat16", max_seq_len=None: (model, tokenizer)

    # Override sys.argv and exec the __main__ block
    sys.argv = [
        "eval_nih_extended.py",
        "--model_path", args.model_path,
        "--output_dir", args.output_dir,
        "--device", args.device,
        "--dtype", args.dtype,
        "--max_new_tokens", str(args.max_new_tokens),
        "--num_trials", str(args.num_trials),
        "--seed", str(args.seed),
    ]
    if args.smoke:
        sys.argv.append("--smoke")
    if args.config_filter:
        sys.argv += ["--config_filter", args.config_filter]

    # Read and exec the __main__ block
    import ast
    src = Path(nih_mod.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            if (isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__"
                and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in node.test.comparators)):
                code = compile(ast.Module(body=node.body, type_ignores=[]), nih_mod.__file__, "exec")
                exec(code, nih_mod.__dict__)
                break
