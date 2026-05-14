"""BABILong evaluation wrapper for the mem_space streaming memory architecture.

This is the mem_space counterpart to ``scripts/run_babilong_h6.py``.  It evaluates
a Llama-3-8B model patched with ``MemorySpaceLayer`` (the H7+/champion family) on
BABILong qa1-qa5 tasks across multiple context lengths (0k-32k).

mem_space is stateful:
    1.  Input is chunked into ``chunk_size`` segments.
    2.  Each chunk is run through ``model(input_ids=...)`` — the patched decoder
        layers prepend slot tokens and EMA-write the memory bank in-place.  The
        memory bank persists across chunks (no reset between them).
    3.  Each new BABILong sample resets the bank via ``_reset_banks(model)``.

Differences vs ``run_babilong_h6.py``:
    * mem_space uses the **HF LlamaForCausalLM forward signature** (the patched
      MemorySpaceLayer is transparent to the wrapping HF model).  We call
      ``model(input_ids=...)`` directly, not ``model.forward_chunk(...)``.
    * Reset is done via ``_reset_banks(model)`` (copied from
      ``eval_niah_mem_space.py``), which prefers the shared memory bank.
    * ``use_cache=False`` is mandatory because MemorySpaceLayer attention does
      not support HF's KV cache code path.
    * The adapter_config.json field names use the abbreviated form
      (``writeback_warmup_steps``, ``unfreeze_hidden_to_slot``); we translate
      them to the MemorySpaceConfig dataclass field names.

Usage:
    python scripts/run_babilong_mem_space.py \
        --model_path /path/to/Llama-3-8B \
        --checkpoint outputs/champion_ckpt/mem_space_adapter.pt \
        --adapter_config outputs/champion_ckpt/adapter_config.json \
        --output_name mem_space_champion \
        [--tasks qa1 qa2 ...] [--lengths 0k 1k ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add babilong to path — same locations as run_babilong_h6.py
BABILONG_ROOTS = [
    "/apdcephfs_zwfy6/share_303098609/pighzliu_code/babilong",
    "/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong",
]
for _root in BABILONG_ROOTS:
    if os.path.isdir(_root) and _root not in sys.path:
        sys.path.insert(0, _root)

import datasets  # noqa: E402
from transformers import AutoTokenizer, LlamaForCausalLM  # noqa: E402

from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402

from src.memory.mem_space import MemorySpaceConfig, apply_mem_space_to_model  # noqa: E402


# --------------------------------------------------------------------------- #
# Memory helpers (copied verbatim from eval_niah_mem_space.py:82-101)
# --------------------------------------------------------------------------- #


def _reset_banks(model: torch.nn.Module) -> None:
    """Wipe per-sample slot state between BABILong samples.

    Under ``config.shared_memory_bank=True`` the patch exposes
    ``_mem_space_shared_bank`` on the root model; resetting that one object is
    equivalent to resetting every wrapper's bank (they all reference the same
    object).  Falls back to per-layer bank reset if no shared bank is present.
    """
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.reset()
        return
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return
    for w in mem_layers:
        w.memory_bank.reset()


def _freeze_banks(model: torch.nn.Module) -> None:
    """Freeze memory banks during greedy generation so writeback doesn't
    overwrite slots accumulated from the context."""
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.frozen = True
        return
    for w in getattr(root, "_mem_space_layers", []):
        w.memory_bank.frozen = True


def _unfreeze_banks(model: torch.nn.Module) -> None:
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.frozen = False
        return
    for w in getattr(root, "_mem_space_layers", []):
        w.memory_bank.frozen = False


# --------------------------------------------------------------------------- #
# Adapter config → MemorySpaceConfig translation
# --------------------------------------------------------------------------- #


# Map abbreviated field names found in adapter_config.json → MemorySpaceConfig
# fields. Anything not in this map and not a MemorySpaceConfig field is ignored.
_ADAPTER_CONFIG_FIELD_MAP = {
    "writeback_warmup_steps": "writeback_gate_warmup_steps",
}


def build_mem_space_config(adapter_cfg: dict) -> MemorySpaceConfig:
    """Construct a MemorySpaceConfig from an adapter_config.json dict.

    Handles two pieces of impedance mismatch:
      * `writeback_warmup_steps` (json) → `writeback_gate_warmup_steps` (dataclass)
      * `unfreeze_hidden_to_slot=True` (json) → `hidden_to_slot_frozen=False` (dataclass)

    Unknown keys (e.g. `max_train_steps`, `lr`) are silently dropped.
    """
    valid_fields = set(MemorySpaceConfig.__dataclass_fields__.keys())
    kwargs: dict = {}
    for k, v in adapter_cfg.items():
        # Rename if needed
        target = _ADAPTER_CONFIG_FIELD_MAP.get(k, k)
        if target == "unfreeze_hidden_to_slot":
            # Flip semantics: unfreeze=True means hidden_to_slot_frozen=False.
            kwargs["hidden_to_slot_frozen"] = not bool(v)
            continue
        if target in valid_fields:
            kwargs[target] = v
        # else: silently ignore (training-only keys like lr, max_train_steps)
    return MemorySpaceConfig(**kwargs)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #


def load_mem_space_model(
    model_path: str,
    checkpoint_path: str,
    mem_config: MemorySpaceConfig,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    attn_impl: str = "sdpa",
):
    """Build base Llama + mem_space patch + load adapter ckpt.

    Mirrors eval_niah_mem_space.py:472-620:
      1. Load base LlamaForCausalLM in bfloat16.
      2. Snapshot rotary inv_freq in fp32 (H7 fix v2 pre-step).
      3. Apply mem_space patch to all decoder layers.
      4. .to(device, dtype) — moves everything (including freshly-built CPU/fp32
         mem_space modules) to the right place.
      5. Restore rotary buffers to fp32 (H7 fix v2 post-step).
      6. Load adapter checkpoint (strict=False; handle ddp `module.` prefix and
         common state-dict-wrapper layouts).
      7. Force step_counter = writeback_warmup_steps so warmup_frac=1.0 at eval
         (Fix J from eval_niah_mem_space.py).
    """
    print(f"[mem_space-BABILong] Loading base model from: {model_path}")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)

    # H7 fix v2 pre-step: snapshot rotary inv_freq in fp32 BEFORE any
    # `.to(dtype=bf16)` corrupts them. See eval_niah_mem_space.py:502-525.
    _rope_snapshot: dict = {}
    try:
        _rot = model.model.rotary_emb
        for _name in ("inv_freq", "original_inv_freq"):
            if hasattr(_rot, _name):
                _rope_snapshot[_name] = getattr(_rot, _name).detach().to(torch.float32).clone()
    except AttributeError:
        pass

    # Patch all decoder layers with MemorySpaceLayer
    print(f"[mem_space-BABILong] Applying mem_space patch (num_slots={mem_config.num_slots}, "
          f"top_k={mem_config.top_k}, shared_bank={mem_config.shared_memory_bank})")
    apply_mem_space_to_model(model, mem_config, layer_indices=None)

    # Move freshly-created mem_space modules to device/dtype
    model.to(device=device, dtype=dtype)

    # H7 fix v2 post-step: restore rotary buffers to fp32 on the right device.
    try:
        _rot = model.model.rotary_emb
        for _name, _buf in _rope_snapshot.items():
            _buf = _buf.to(device=device, dtype=torch.float32)
            _rot._buffers[_name] = _buf
            setattr(_rot, _name, _buf)
        if _rope_snapshot:
            print(f"[mem_space-BABILong] H7 fix v2 applied: restored rotary buffers "
                  f"{sorted(_rope_snapshot.keys())} to float32")
    except AttributeError:
        print("[mem_space-BABILong] WARNING: rotary_emb not accessible — skipping H7 fix")

    # Load checkpoint
    print(f"[mem_space-BABILong] Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Common state-dict layouts: raw OrderedDict / {model_state_dict: ...} / {state_dict: ...}.
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Assume the dict itself is the state_dict (this is what
            # eval_niah_mem_space.py:552 expects for the champion ckpt).
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Strip DDP "module." prefix if present.
    cleaned: dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[mem_space-BABILong] Loaded {len(cleaned)} keys | "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print(f"[mem_space-BABILong] WARNING: first 5 unexpected keys: {list(unexpected)[:5]}")
    # Adapter-specific missing keys are real failures; base-model missing keys
    # are expected with strict=False (the base weights came from from_pretrained).
    adapter_missing = [
        k for k in missing
        if any(s in k for s in (
            "slot_output_gate", "gate_param", "Q_sel", "K_sel",
            "slot_to_hidden", "hidden_to_slot",
        ))
    ]
    if adapter_missing:
        print(f"[mem_space-BABILong] WARNING: {len(adapter_missing)} adapter keys NOT "
              f"loaded — first 5: {adapter_missing[:5]}")

    # Fix J: force step_counter = warmup_steps so β/warmup_frac is fully ramped.
    from src.memory.mem_space.layer import MemorySpaceLayer as _MSL  # local import to avoid cycles
    _mem_layers = getattr(model, "_mem_space_layers", [])
    _warmup_target = mem_config.writeback_gate_warmup_steps if mem_config.writeback_gate_warmup_steps > 0 else 1
    for _w in _mem_layers:
        if isinstance(_w, _MSL):
            _w.step_counter = _warmup_target
    print(f"[mem_space-BABILong] Fix J: set step_counter={_warmup_target} on "
          f"{len(_mem_layers)} MemorySpaceLayer(s)")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# --------------------------------------------------------------------------- #
# Chunked generation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def generate_with_mem_space(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    chunk_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Streaming generation for a single BABILong sample.

    Strategy (mirrors stream_haystack + F2 "last-chunk replay" trick from
    eval_niah_mem_space.py:858-901):

      1. Reset memory banks (fresh state for this sample).
      2. Stream all-but-last chunks through ``model(input_ids=...)`` so the
         memory bank accumulates context (no return value needed; mem_space
         writes the bank in-place during forward).
      3. Freeze the bank, then autoregressively generate from the last chunk.
         The last chunk is consumed in the FIRST forward call (we read logits
         at its last position); subsequent steps append one token at a time.
      4. Unfreeze the bank (for cleanliness; doesn't matter for inference but
         keeps the contract).

    We do NOT do the F2 last-chunk replay: BABILong's question_suffix is already
    embedded at the END of the formatted input (after the haystack); the last
    chunk already contains the question text + the right context, so logit
    quality at its tail is what we want to read.

    Args:
        input_ids: [1, total_len] tensor on `device`.

    Returns:
        Decoded text of `max_new_tokens` generated tokens (skip_special_tokens=True).
    """
    if device is None:
        device = next(model.parameters()).device

    _reset_banks(model)

    tokens = input_ids[0]  # [total_len]
    chunks = list(tokens.split(chunk_size))

    # Stream all-but-last chunks (memory accumulation only — no logit reads)
    if len(chunks) > 1:
        for chunk in chunks[:-1]:
            chunk_tensor = chunk.unsqueeze(0).to(device)  # [1, <=chunk_size]
            _ = model(input_ids=chunk_tensor, use_cache=False)

    # Freeze the bank — generation should not pollute the slots that hold the context.
    _freeze_banks(model)
    try:
        cur = chunks[-1].unsqueeze(0).to(device)  # [1, last_chunk_len]
        generated_ids: list[int] = []
        for step in range(max_new_tokens):
            outputs = model(input_ids=cur, use_cache=False)
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            if step == 0 and tokenizer.eos_token_id is not None:
                # Match H6 behaviour: suppress EOS as the very first generated
                # token so we don't return an empty answer.
                logits[:, tokenizer.eos_token_id] = float("-inf")
            next_tok = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
            tok_id = int(next_tok.item())
            if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id and step > 0:
                break
            generated_ids.append(tok_id)
            cur = torch.cat([cur, next_tok], dim=-1)
    finally:
        _unfreeze_banks(model)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="BABILong evaluation for mem_space architecture")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base Llama-3-8B model directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to mem_space adapter .pt checkpoint")
    parser.add_argument("--adapter_config", type=str, required=True,
                        help="Path to adapter_config.json describing the MemorySpaceConfig")
    parser.add_argument("--results_folder", type=str, default="./babilong_results",
                        help="Folder to store BABILong eval results")
    parser.add_argument("--output_name", type=str, required=True,
                        help="Subfolder name for this evaluation run")
    parser.add_argument("--dataset_name", type=str, default="RMT-team/babilong",
                        help="HuggingFace dataset name")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=["qa1", "qa2", "qa5"],
                        help="BABILong tasks to evaluate")
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["0k", "1k", "2k", "4k", "8k", "16k"],
                        help="BABILong context lengths to evaluate")
    parser.add_argument("--chunk_size", type=int, default=4096,
                        help="Chunk size for memory accumulation (matches mem_space training seq_len)")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Maximum new tokens to generate per sample")
    parser.add_argument("--limit", type=int, default=100,
                        help="Maximum samples per task/length cell (default 100; -1 = all)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run on")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Wrap the formatted input in the tokenizer's chat template")
    parser.add_argument("--use_instruction", action="store_true", default=True,
                        help="Include BABILong DEFAULT_PROMPTS[task]['instruction']")
    parser.add_argument("--use_examples", action="store_true", default=True,
                        help="Include BABILong DEFAULT_PROMPTS[task]['examples']")
    parser.add_argument("--use_post_prompt", action="store_true", default=True,
                        help="Include BABILong DEFAULT_PROMPTS[task]['post_prompt']")
    args = parser.parse_args()

    print(f"[mem_space-BABILong] Configuration:")
    print(f"  Base model:      {args.model_path}")
    print(f"  Checkpoint:      {args.checkpoint}")
    print(f"  Adapter config:  {args.adapter_config}")
    print(f"  Tasks:           {args.tasks}")
    print(f"  Lengths:         {args.lengths}")
    print(f"  Chunk size:      {args.chunk_size}")
    print(f"  Max new tokens:  {args.max_new_tokens}")
    print(f"  Limit/cell:      {args.limit}")
    print(f"  Device:          {args.device}")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[mem_space-BABILong] Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Load + parse adapter config
    with open(args.adapter_config, "r") as f:
        adapter_cfg = json.load(f)
    mem_config = build_mem_space_config(adapter_cfg)
    print(f"[mem_space-BABILong] MemorySpaceConfig: num_slots={mem_config.num_slots}, "
          f"top_k={mem_config.top_k}, selector_dim={mem_config.selector_dim}, "
          f"warmup_steps={mem_config.writeback_gate_warmup_steps}, "
          f"slot_init={mem_config.slot_init}, "
          f"shared_bank={mem_config.shared_memory_bank}, "
          f"hidden_to_slot_frozen={mem_config.hidden_to_slot_frozen}")

    # Build + load model
    model = load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        mem_config=mem_config,
        device=device,
        dtype=dtype,
        attn_impl=args.attn_impl,
    )

    # ------------------------------------------------------------------ #
    # BABILong eval loop (mirrors run_babilong_h6.py:406-512)
    # ------------------------------------------------------------------ #
    for task in tqdm(args.tasks, desc="tasks"):
        if task not in DEFAULT_PROMPTS:
            print(f"[WARNING] Task {task} not in DEFAULT_PROMPTS, skipping.")
            continue

        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"] if args.use_instruction else "",
            "examples":    DEFAULT_PROMPTS[task]["examples"]    if args.use_examples    else "",
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"] if args.use_post_prompt else "",
            "template":    DEFAULT_TEMPLATE,
            "chat_template": args.use_chat_template,
            "system_prompt": "",
        }
        prompt_name = "_".join(
            [f"{k}_yes" if prompt_cfg[k] else f"{k}_no"
             for k in prompt_cfg if k != "template"]
        )

        for split_name in tqdm(args.lengths, desc="lengths", leave=False):
            print(f"\n[mem_space-BABILong] task={task}, length={split_name}")

            try:
                data = datasets.load_dataset(args.dataset_name, split_name)
                task_data = data[task]
            except Exception as e:
                print(f"[ERROR] Failed to load dataset {args.dataset_name}/{split_name}/{task}: {e}")
                continue

            outdir = Path(args.results_folder) / args.output_name
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"{task}_{split_name}_{prompt_name}.csv"
            cfg_file = outdir / f"{task}_{split_name}_{prompt_name}.json"

            json.dump(
                {
                    "prompt": prompt_cfg,
                    "generate_kwargs": {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False,
                        "num_beams": 1,
                    },
                    "model": {
                        "model_path":      args.model_path,
                        "checkpoint":      args.checkpoint,
                        "adapter_config":  args.adapter_config,
                        "chunk_size":      args.chunk_size,
                        "num_slots":       mem_config.num_slots,
                        "top_k":           mem_config.top_k,
                        "shared_memory_bank": mem_config.shared_memory_bank,
                    },
                },
                open(cfg_file, "w"),
                indent=4,
            )

            df = pd.DataFrame({"target": [], "output": [], "question": []})

            num_samples = len(task_data)
            if args.limit > 0:
                num_samples = min(num_samples, args.limit)

            for idx in tqdm(range(num_samples), desc=f"{task}/{split_name}", leave=False):
                sample = task_data[idx]
                target = sample["target"]
                context = sample["input"]
                question = sample["question"]

                # Build formatted text
                input_text = get_formatted_input(
                    context,
                    question,
                    prompt_cfg["examples"],
                    prompt_cfg["instruction"],
                    prompt_cfg["post_prompt"],
                    template=prompt_cfg["template"],
                )

                if args.use_chat_template:
                    messages = [{"role": "user", "content": input_text}]
                    input_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
                if isinstance(input_ids, list):
                    input_ids = torch.tensor([input_ids], dtype=torch.long)
                input_ids = input_ids.to(device)

                # Generate (handles _reset_banks + _freeze_banks internally)
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    output = generate_with_mem_space(
                        model=model,
                        input_ids=input_ids,
                        tokenizer=tokenizer,
                        chunk_size=args.chunk_size,
                        max_new_tokens=args.max_new_tokens,
                        device=device,
                    )

                df.loc[len(df)] = [target, output, question]

                if (idx + 1) % 10 == 0 or idx == num_samples - 1:
                    df.to_csv(outfile, index=False)

            df.to_csv(outfile, index=False)
            print(f"[mem_space-BABILong] Saved {len(df)} results to {outfile}")

    print("\n[mem_space-BABILong] Evaluation complete!")


if __name__ == "__main__":
    main()
