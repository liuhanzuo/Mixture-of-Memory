"""Single-checkpoint, single-length BABILong evaluator for the H-series
CrossAttentionMemoryModel checkpoints.

Runs N samples on M tasks at a single context length, returns one JSONL line
of the form expected by the watchdog:

    {"ts":..., "exp":..., "step":..., "length": 4096,
     "tasks": {"qa1":..., "qa2":..., "qa5":...},
     "avg":..., "num_samples": 30}

Designed to be invoked by `babilong_ckpt_watchdog.py` — one process per
(ckpt, length) pair on a single GPU. All H-series runs share the same
architecture (slot_forward + middle_layer_memory, write@16 read@18,22,26,30,
64 slots, dual-gate), so we don't need per-experiment branches.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_cross_attn_babilong.py \
        --ckpt_path /path/to/step_500.pt \
        --exp_name H11_v2 \
        --step 500 \
        --length 4k \
        --num_samples 30 \
        --tasks qa1 qa2 qa5 \
        --output_jsonl /path/to/babilong_realtime.jsonl
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, LlamaForCausalLM


def _find_project_root() -> str:
    """Locate Mixture-of-Memory root (works on b200 wzc1 path or h20 zwfy6 mirror)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


PROJECT_ROOT = _find_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _find_babilong() -> str:
    """Find the babilong package (different mount on b200 vs h20)."""
    candidates = [
        "/apdcephfs_zwfy6/share_304376610/pighzliu_code/babilong",
        "/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong",
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "babilong")):
            return c
    raise RuntimeError("Could not locate babilong package on either cluster mount")


sys.path.insert(0, _find_babilong())

import datasets  # noqa: E402
from babilong.metrics import TASK_LABELS, compare_answers  # noqa: E402
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402


def _import_train_module():
    """Import scripts/train_cross_attn_memory.py without triggering its argparse."""
    train_path = os.path.join(PROJECT_ROOT, "scripts", "train_cross_attn_memory.py")
    spec = importlib.util.spec_from_file_location("train_cross_attn_memory", train_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_cross_attn_memory"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_h_series_model(model_path: str, num_slots: int, memory_write_layer: int,
                         memory_read_layers: str, memory_init: str,
                         use_dual_gate: bool, forget_bias_init: float,
                         input_bias_init: float, device: torch.device):
    train_module = _import_train_module()
    CrossAttentionMemoryModel = train_module.CrossAttentionMemoryModel

    print(f"[eval] Loading base model: {model_path}", flush=True)
    base_model = LlamaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )

    model = CrossAttentionMemoryModel(
        base_model=base_model,
        num_slots=num_slots,
        top_k=8,
        full_finetune=True,
        use_memory=True,
        use_cross_attn_memory=True,
        gradient_checkpointing=False,
        cross_attn_dropout=0.0,
        residual_scale=0.01,
        swa_window=0,
        write_lr=0.1,
        slot_forward=True,
        slot_isolated=False,
        memory_init=memory_init,
        recon_loss_weight=0.0,
        cross_chunk_propagation=False,
        middle_layer_memory=True,
        memory_write_layer=memory_write_layer,
        memory_read_layers=memory_read_layers,
        use_dual_gate=use_dual_gate,
        forget_bias_init=forget_bias_init,
        input_bias_init=input_bias_init,
        dual_gate_tanh_new=True,
    )
    return model


def load_ckpt_into_model(model, ckpt_path: str, device: torch.device):
    print(f"[eval] Loading checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[eval] missing keys ({len(missing)}): {missing[:3]}...", flush=True)
    if unexpected:
        print(f"[eval] unexpected keys ({len(unexpected)}): {unexpected[:3]}...", flush=True)
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    return model


@torch.no_grad()
def generate_with_memory(model, input_ids: torch.Tensor, tokenizer,
                         chunk_size: int = 4096, max_new_tokens: int = 20,
                         device=None) -> str:
    """Process input in chunks (slot state accumulates), then autoregress."""
    if device is None:
        device = next(model.parameters()).device
    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    # Process all but the last chunk to accumulate memory
    for chunk in chunks[:-1]:
        ct = chunk.unsqueeze(0).to(device)
        model.forward_chunk(ct, enable_write_grad=False)
    last = chunks[-1].unsqueeze(0).to(device)
    out = model.forward_chunk(last, enable_write_grad=False)
    logits = out["logits"]
    nl = logits[:, -1, :].clone()
    nl[:, tokenizer.eos_token_id] = float("-inf")
    nxt = nl.argmax(dim=-1, keepdim=True)
    generated = [nxt.item()]
    for _ in range(max_new_tokens - 1):
        if generated[-1] == tokenizer.eos_token_id:
            break
        gi = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
        out = model.forward_chunk(gi, enable_write_grad=False)
        nl = out["logits"][:, -1, :]
        generated.append(nl.argmax(dim=-1).item())
    if tokenizer.eos_token_id in generated:
        generated = generated[:generated.index(tokenizer.eos_token_id)]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate_single(ckpt_path: str, exp_name: str, step: int, length: str,
                    tasks, num_samples: int, model_path: str,
                    output_jsonl: str, dataset_name: str = "RMT-team/babilong",
                    chunk_size: int = 4096, max_new_tokens: int = 20,
                    device_str: str = "cuda:0"):
    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = build_h_series_model(
        model_path=model_path, num_slots=64,
        memory_write_layer=16, memory_read_layers="18,22,26,30",
        memory_init="strided", use_dual_gate=True,
        forget_bias_init=1.0, input_bias_init=0.0, device=device,
    )
    model = load_ckpt_into_model(model, ckpt_path, device)

    # Map "4k" -> 4096
    LEN_MAP = {"0k": 0, "1k": 1024, "2k": 2048, "4k": 4096, "8k": 8192,
               "16k": 16384, "32k": 32768}
    length_int = LEN_MAP.get(length, int(length))

    use_instruction = use_examples = use_post_prompt = True
    use_chat_template = False

    per_task_acc = {}
    t0 = time.time()
    for task in tasks:
        if task not in DEFAULT_PROMPTS:
            print(f"[eval] WARNING task {task} not in prompts, skipping", flush=True)
            continue
        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"] if use_instruction else "",
            "examples": DEFAULT_PROMPTS[task]["examples"] if use_examples else "",
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"] if use_post_prompt else "",
            "template": DEFAULT_TEMPLATE,
            "chat_template": use_chat_template,
        }
        try:
            data = datasets.load_dataset(dataset_name, length)
            task_data = data[task]
        except Exception as e:
            print(f"[eval] Failed loading {dataset_name}/{length}/{task}: {e}", flush=True)
            continue

        n = min(num_samples, len(task_data))
        correct = 0
        for idx in range(n):
            sample = task_data[idx]
            target = sample["target"]
            ctx = sample["input"]
            question = sample["question"]
            input_text = get_formatted_input(
                ctx, question, prompt_cfg["examples"], prompt_cfg["instruction"],
                prompt_cfg["post_prompt"], template=prompt_cfg["template"],
            )
            input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt").to(device)
            model.reset_slots()
            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = generate_with_memory(
                        model, input_ids, tokenizer,
                        chunk_size=chunk_size, max_new_tokens=max_new_tokens, device=device,
                    )
            except torch.cuda.OutOfMemoryError:
                print(f"[eval] OOM at {task}/{length} idx={idx}, skipping rest of task", flush=True)
                torch.cuda.empty_cache()
                break
            if compare_answers(target, output, question, TASK_LABELS[task]):
                correct += 1
        acc = correct / max(n, 1)
        per_task_acc[task] = acc
        print(f"[eval] {exp_name} step={step} len={length} {task}: {correct}/{n} = {acc:.3f}", flush=True)

    elapsed = time.time() - t0
    avg_acc = sum(per_task_acc.values()) / max(len(per_task_acc), 1)
    record = {
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
        "exp": exp_name,
        "step": int(step),
        "length": length_int,
        "length_label": length,
        "tasks": per_task_acc,
        "avg": avg_acc,
        "num_samples": num_samples,
        "elapsed_s": round(elapsed, 1),
        "ckpt_path": ckpt_path,
    }
    Path(os.path.dirname(output_jsonl) or ".").mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[eval] DONE {exp_name} step={step} len={length} avg={avg_acc:.3f} "
          f"elapsed={elapsed:.0f}s -> {output_jsonl}", flush=True)
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--exp_name", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--length", required=True, help="One of 1k/2k/4k/8k/...")
    p.add_argument("--tasks", nargs="+", default=["qa1", "qa2", "qa5"])
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--model_path", default=None)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--dataset_name", default="RMT-team/babilong")
    p.add_argument("--chunk_size", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    if args.model_path is None:
        # auto-detect llama path on either cluster
        for c in [
            "/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Llama--Llama3-8b",
            "/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b",
            "/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Llama--Llama3-8b",
        ]:
            if os.path.isdir(c):
                args.model_path = c
                break
        if args.model_path is None:
            raise RuntimeError("Could not find Llama3-8b on either mount")

    evaluate_single(
        ckpt_path=args.ckpt_path,
        exp_name=args.exp_name,
        step=args.step,
        length=args.length,
        tasks=args.tasks,
        num_samples=args.num_samples,
        model_path=args.model_path,
        output_jsonl=args.output_jsonl,
        dataset_name=args.dataset_name,
        chunk_size=args.chunk_size,
        max_new_tokens=args.max_new_tokens,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
