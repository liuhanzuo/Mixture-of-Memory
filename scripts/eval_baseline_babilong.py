"""Unified BABILong-100 evaluator for 3 baselines.

Baselines:
    --baseline llama32_1b_instruct  : Llama-3.2-1B-Instruct (vanilla HF)
    --baseline memoryllm            : MemoryLLM-8B-chat (inject_memory + chunked)
    --baseline beacon               : Activation Beacon Qwen2-7B-Instruct (trust_remote_code)

Outputs CSVs in <results_folder>/<output_name>/<task>_<length>_<suffix>.csv
matching the format of babilong/scripts/run_model_on_babilong.py so we can
re-use score_babilong_results.py.

Designed for transformers 4.46.3 on H20.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm

# ---- path setup ----
H20_BABILONG = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/babilong-pkg"
H20_MEMORYLLM_SRC = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/MemoryLLM-source"
LOCAL_BABILONG = "/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong"
LOCAL_MEMORYLLM_SRC = "/apdcephfs_wzc1/share_303098609/pighzliu_code/MemoryLLM-source"

for p in (H20_BABILONG, LOCAL_BABILONG):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import datasets  # noqa: E402
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402


def _set_proxy() -> None:
    os.environ.setdefault("http_proxy", "http://star-proxy.oa.com:3128")
    os.environ.setdefault("https_proxy", "http://star-proxy.oa.com:3128")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


# ---------------------------------------------------------------------------
# Baseline: Llama-3.2-1B-Instruct  (and any plain HF causal LM)
# ---------------------------------------------------------------------------

def run_plain_hf(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[plain_hf] Loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = None
    for impl in ("flash_attention_2", "sdpa"):
        try:
            print(f"[plain_hf] Trying attn_implementation={impl}")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=impl,
                device_map="cuda:0",
            )
            print(f"[plain_hf] Loaded with {impl}")
            break
        except (ValueError, ImportError, RuntimeError) as e:
            print(f"[plain_hf] {impl} failed: {e}")
    if model is None:
        raise RuntimeError("Could not load model with FA2 or SDPA")
    model.eval()

    _eval_loop(
        args,
        prepare_input=lambda ctx, q, examples, instr, post, use_chat: _hf_prepare_input(
            tokenizer, ctx, q, examples, instr, post, use_chat
        ),
        generate=lambda inputs: _hf_generate(model, tokenizer, inputs, args.max_new_tokens),
        per_sample_setup=None,
    )


def _hf_prepare_input(tokenizer, ctx, q, examples, instr, post, use_chat):
    text = get_formatted_input(ctx, q, examples, instr, post, template=DEFAULT_TEMPLATE)
    if use_chat:
        msgs = [{"role": "user", "content": text}]
        ids = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
    else:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    return ids.to("cuda:0")


def _hf_generate(model, tokenizer, input_ids, max_new):
    eos_ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )
    new_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Baseline: MemoryLLM-8B-chat (inject_memory)
# ---------------------------------------------------------------------------

def run_memoryllm(args: argparse.Namespace) -> None:
    """MemoryLLM uses inject_memory(chunk) to load context, then generate(question)."""
    if H20_MEMORYLLM_SRC not in sys.path:
        sys.path.insert(0, H20_MEMORYLLM_SRC)
    if LOCAL_MEMORYLLM_SRC not in sys.path:
        sys.path.insert(0, LOCAL_MEMORYLLM_SRC)

    from transformers import AutoConfig, AutoTokenizer
    try:
        from modeling_memoryllm import MemoryLLM
    except ImportError as e:
        raise RuntimeError(
            f"Cannot import MemoryLLM source from {H20_MEMORYLLM_SRC}: {e}"
        )

    print(f"[memoryllm] Loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Patch rope_theta if missing (transformers 4.46 may strip it from attached config)
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    cfg_path = os.path.join(args.model_path, "config.json")
    raw = json.load(open(cfg_path))
    rope_theta = raw.get("rope_theta", None)
    if rope_theta is None and isinstance(raw.get("rope_scaling"), dict):
        rope_theta = raw["rope_scaling"].get("rope_theta", 500000.0)
    if rope_theta is None:
        rope_theta = 500000.0
    cfg.rope_theta = rope_theta
    print(f"[memoryllm] config.rope_theta = {rope_theta}")

    model = None
    for impl in ("flash_attention_2", "sdpa"):
        try:
            model = MemoryLLM.from_pretrained(
                args.model_path,
                config=cfg,
                attn_implementation=impl,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            print(f"[memoryllm] Loaded with {impl}")
            break
        except (ValueError, ImportError, AttributeError, RuntimeError) as e:
            print(f"[memoryllm] {impl} failed: {e}")
    if model is None:
        raise RuntimeError("MemoryLLM: cannot load")

    model = model.to("cuda:0")
    model.eval()

    # Fix: the underlying LlamaModel inside MemoryLLM may be missing
    # `prepare_inputs_for_generation` if the override happened only on the wrapper.
    # Add a no-op shim if missing on whichever module .generate() walks.
    _patch_for_generate(model)

    initial_memory = None
    if hasattr(model, "memory") and isinstance(model.memory, torch.Tensor):
        initial_memory = model.memory.detach().clone()
        print(f"[memoryllm] initial_memory: {tuple(initial_memory.shape)} {initial_memory.dtype}")
    else:
        print("[memoryllm] WARN: no .memory attribute found; reset will be a no-op")

    chunk_size = getattr(model.config, "num_tokens", 1024)
    print(f"[memoryllm] chunk_size={chunk_size}")

    def _per_sample(_sample):
        if initial_memory is not None:
            model.memory.copy_(initial_memory)

    def _prepare(ctx, q, examples, instr, post, use_chat):
        # Inject the long context into memory first
        if ctx and ctx.strip():
            ids = tokenizer(ctx, return_tensors="pt", add_special_tokens=False).input_ids
            ids = ids.to("cuda:0")
            total = ids.shape[1]
            if total >= 16:
                pos = 0
                while pos < total:
                    chk = ids[:, pos:pos + chunk_size]
                    if chk.shape[1] >= 16:
                        with torch.no_grad():
                            try:
                                model.inject_memory(chk, update_memory=True)
                            except TypeError:
                                model.inject_memory(chk)
                    pos += chunk_size
        # Build the question prompt without context (it's in memory now)
        text = get_formatted_input("", q, examples, instr, post, template=DEFAULT_TEMPLATE)
        if use_chat:
            msgs = [{"role": "user", "content": text}]
            ids = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
            # MemoryLLM trained without leading BOS — drop it if present
            if ids[0, 0].item() == tokenizer.bos_token_id and ids.shape[1] > 1:
                ids = ids[:, 1:]
        else:
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        return ids.to("cuda:0")

    def _generate(input_ids):
        return _hf_generate(model, tokenizer, input_ids, args.max_new_tokens)

    _eval_loop(
        args,
        prepare_input=_prepare,
        generate=_generate,
        per_sample_setup=_per_sample,
    )


def _patch_for_generate(model: Any) -> None:
    """Add stubs that older transformers versions (<5) may expect on a custom model."""
    # On transformers 4.46, MemoryLLM should already have prepare_inputs_for_generation
    # via PreTrainedModel; this is a safety net in case the override is broken.
    if not hasattr(model, "prepare_inputs_for_generation"):
        def _stub(input_ids, past_key_values=None, attention_mask=None, **kwargs):
            return {"input_ids": input_ids, "past_key_values": past_key_values, "attention_mask": attention_mask}
        model.prepare_inputs_for_generation = _stub
    # _supports_cache_class came in transformers 4.40+; ensure it exists
    if not hasattr(model, "_supports_cache_class"):
        model._supports_cache_class = False


# ---------------------------------------------------------------------------
# Baseline: Activation Beacon (Qwen2-7B-Instruct)
# ---------------------------------------------------------------------------

def run_beacon(args: argparse.Namespace) -> None:
    """Beacon model uses trust_remote_code; reset memory between samples."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[beacon] Loading {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = None
    for impl in ("flash_attention_2", "sdpa"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=impl,
                device_map="cuda:0",
            )
            print(f"[beacon] Loaded with {impl}")
            break
        except (ValueError, ImportError, RuntimeError) as e:
            print(f"[beacon] {impl} failed: {e}")
    if model is None:
        raise RuntimeError("Beacon: cannot load")
    model.eval()

    has_memory_reset = hasattr(model, "memory") and hasattr(model.memory, "reset")
    print(f"[beacon] has memory.reset = {has_memory_reset}")

    def _per_sample(_sample):
        if has_memory_reset:
            model.memory.reset()

    def _prepare(ctx, q, examples, instr, post, use_chat):
        return _hf_prepare_input(tokenizer, ctx, q, examples, instr, post, use_chat)

    def _generate(input_ids):
        out = _hf_generate(model, tokenizer, input_ids, args.max_new_tokens)
        if has_memory_reset:
            try:
                model.memory.reset()
            except Exception:
                pass
        return out

    _eval_loop(
        args,
        prepare_input=_prepare,
        generate=_generate,
        per_sample_setup=_per_sample,
    )


# ---------------------------------------------------------------------------
# Common eval loop
# ---------------------------------------------------------------------------

def _eval_loop(args, prepare_input, generate, per_sample_setup):
    """Iterate over (task, length) -> samples; write a CSV per cell."""
    use_chat = args.use_chat_template
    suffix_parts = [
        "instruction_yes" if args.use_instruction else "instruction_no",
        "examples_yes" if args.use_examples else "examples_no",
        "post_prompt_yes" if args.use_post_prompt else "post_prompt_no",
        "chat_template_yes" if use_chat else "chat_template_no",
        "system_prompt_no",
    ]
    suffix = "_".join(suffix_parts)

    out_dir = Path(args.results_folder) / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in tqdm(args.tasks, desc="tasks"):
        instr = DEFAULT_PROMPTS[task].get("instruction", "") if args.use_instruction else ""
        examples = DEFAULT_PROMPTS[task].get("examples", "") if args.use_examples else ""
        post = DEFAULT_PROMPTS[task].get("post_prompt", "") if args.use_post_prompt else ""

        for length in tqdm(args.lengths, desc=f"{task}", leave=False):
            outfile = out_dir / f"{task}_{length}_{suffix}.csv"
            if outfile.exists() and not args.overwrite:
                # resume: count existing rows
                try:
                    existing = pd.read_csv(outfile)
                    if len(existing) >= (args.limit or 100):
                        print(f"[eval_loop] {outfile.name} already done ({len(existing)} rows) — skip")
                        continue
                except Exception:
                    pass

            t0 = time.time()
            data = datasets.load_dataset(args.dataset_name, length, split=task)
            samples = list(data)
            if args.limit:
                samples = samples[: args.limit]

            rows = []
            for sample in tqdm(samples, desc=f"{task}/{length}", leave=False):
                if per_sample_setup is not None:
                    per_sample_setup(sample)
                target = sample["target"]
                ctx = sample["input"]
                q = sample["question"]
                try:
                    input_ids = prepare_input(ctx, q, examples, instr, post, use_chat)
                    output = generate(input_ids)
                except Exception as e:
                    output = f"<<ERROR: {type(e).__name__}: {str(e)[:200]}>>"
                    print(f"[eval_loop] sample failed: {output}")
                rows.append({"target": target, "output": output, "question": q})

                # Periodic checkpoint write
                if len(rows) % 25 == 0:
                    pd.DataFrame(rows, columns=["target", "output", "question"]).to_csv(outfile, index=False)

            pd.DataFrame(rows, columns=["target", "output", "question"]).to_csv(outfile, index=False)
            dt = time.time() - t0
            print(f"[eval_loop] saved {len(rows)} -> {outfile} ({dt:.1f}s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", choices=["plain_hf", "memoryllm", "beacon"], required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_name", type=str, required=True)
    p.add_argument(
        "--results_folder",
        type=str,
        default="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/babilong_results",
    )
    p.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    p.add_argument(
        "--tasks", type=str, nargs="+",
        default=["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"],
    )
    p.add_argument(
        "--lengths", type=str, nargs="+",
        default=["1k", "2k", "4k", "8k", "16k", "32k"],
    )
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument("--use_instruction", action="store_true")
    p.add_argument("--use_examples", action="store_true")
    p.add_argument("--use_post_prompt", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--overwrite", action="store_true")

    args = p.parse_args()
    _set_proxy()

    print("=" * 60)
    print(f"BABILong eval baseline={args.baseline}")
    print(f"  model_path  = {args.model_path}")
    print(f"  output_name = {args.output_name}")
    print(f"  tasks       = {args.tasks}")
    print(f"  lengths     = {args.lengths}")
    print(f"  results_dir = {args.results_folder}/{args.output_name}")
    print(f"  limit       = {args.limit}")
    print("=" * 60)

    if args.baseline == "plain_hf":
        run_plain_hf(args)
    elif args.baseline == "memoryllm":
        run_memoryllm(args)
    elif args.baseline == "beacon":
        run_beacon(args)
    else:
        raise ValueError(args.baseline)

    print(f"[main] DONE: {args.output_name}")


if __name__ == "__main__":
    main()
