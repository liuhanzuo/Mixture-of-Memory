"""BABILong evaluation wrapper for MemoryLLM-8B-chat.

MemoryLLM is stateful: inject each BABILong context into its memory pool,
generate from the question prompt, then reset to the checkpoint memory before the
next sample. The reset restores both ``model.memory`` and ``model.initialized``;
otherwise sample N+1 can leak state from sample N.

ENV REQUIREMENT (verified 2026-06-23): run this with ``external/memoryllm_venv``
(transformers==4.43.4, peft==0.10.0, torch==2.6.0+cu124). Under transformers 5.x the
MemoryLLM custom forward produces degenerate token-0 output ("!!!!"/"MarcusMarcus...");
under the pinned torch 2.5.1+cu121 the PEFT LoRA GEMM SIGFPEs on H20 (sm90). Use the
launch script ``scripts/run_babilong_memoryllm.sh`` which points at that venv.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORYLLM_SRC = PROJECT_ROOT.parent / "MemoryLLM-source"
BABILONG_ROOT = PROJECT_ROOT / "third_party" / "babilong-pkg"
for _path in (str(MEMORYLLM_SRC), str(BABILONG_ROOT), str(PROJECT_ROOT)):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import datasets  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from peft import PeftModelForCausalLM  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.generation import GenerationMixin  # noqa: E402

from modeling_memoryllm import LlamaForCausalLM, LlamaModel, MemoryLLM  # noqa: E402
from babilong.metrics import TASK_LABELS, compare_answers  # noqa: E402
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402

DEFAULT_CACHE_DIR = str(PROJECT_ROOT / ".hf_cache")
DEFAULT_MODEL_ID = "YuWangX/memoryllm-8b-chat"


# transformers>=4.50 and current PEFT expect GenerationMixin on model classes.
# The MemoryLLM source is pinned to transformers 4.43-era semantics, so patch the
# local classes once instead of editing ../MemoryLLM-source.
for _cls in (LlamaForCausalLM, MemoryLLM):
    if not issubclass(_cls, GenerationMixin):
        _cls.__bases__ = tuple(dict.fromkeys(_cls.__bases__ + (GenerationMixin,)))
if not hasattr(LlamaModel, "prepare_inputs_for_generation"):
    LlamaModel.prepare_inputs_for_generation = LlamaForCausalLM.prepare_inputs_for_generation
if not hasattr(PeftModelForCausalLM, "prepare_inputs_for_generation"):
    PeftModelForCausalLM.prepare_inputs_for_generation = LlamaForCausalLM.prepare_inputs_for_generation
if not hasattr(DynamicCache, "from_legacy_cache"):
    def _from_legacy_cache(cls, legacy_cache=None):
        cache = cls()
        if legacy_cache is None:
            return cache
        for layer_idx, layer_cache in enumerate(legacy_cache):
            cache.update(layer_cache[0], layer_cache[1], layer_idx)
        return cache
    DynamicCache.from_legacy_cache = classmethod(_from_legacy_cache)
if not hasattr(DynamicCache, "to_legacy_cache"):
    def _to_legacy_cache(self):
        return tuple((layer.keys, layer.values) for layer in self.layers)
    DynamicCache.to_legacy_cache = _to_legacy_cache


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


def _candidate_babilong_cache_dirs(user_cache_dir: str | None) -> list[Path]:
    roots: list[Path] = []
    if user_cache_dir:
        roots.append(Path(user_cache_dir).expanduser())
    for env in ("HF_DATASETS_CACHE", "HF_HOME"):
        if os.environ.get(env):
            root = Path(os.environ[env]).expanduser()
            roots.append(root if env == "HF_DATASETS_CACHE" else root / "datasets")
    roots += [PROJECT_ROOT / ".cache/huggingface/datasets", Path.home() / ".cache/huggingface/datasets"]
    seen, out = set(), []
    for root in roots:
        key = str(root.absolute())
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def _load_babilong_from_arrow_cache(dataset_name: str, split_name: str, cache_dir: Path):
    root = cache_dir / dataset_name.replace("/", "___") / split_name
    arrow_roots = [p for p in root.glob("*/*") if p.is_dir() and any(p.glob("babilong-*.arrow"))]
    if not arrow_roots:
        return None
    arrow_root = max(arrow_roots, key=lambda p: p.stat().st_mtime)
    data = {
        p.stem.removeprefix("babilong-"): datasets.Dataset.from_file(str(p))
        for p in sorted(arrow_root.glob("babilong-*.arrow"))
    }
    if data:
        print(f"[MemoryLLM-BABILong] Loaded {dataset_name}/{split_name} from Arrow cache: {arrow_root}")
    return data or None


def load_babilong_dataset(dataset_name: str, split_name: str, cache_dir: str | None = None):
    last_error = None
    for candidate in _candidate_babilong_cache_dirs(cache_dir):
        try:
            data = datasets.load_dataset(
                dataset_name,
                split_name,
                cache_dir=str(candidate),
                download_mode="reuse_dataset_if_exists",
            )
            print(f"[MemoryLLM-BABILong] Loaded {dataset_name}/{split_name} with cache_dir={candidate}")
            return data
        except Exception as e:
            last_error = e
            data = _load_babilong_from_arrow_cache(dataset_name, split_name, candidate)
            if data is not None:
                return data
    try:
        return datasets.load_dataset(dataset_name, split_name, download_mode="reuse_dataset_if_exists")
    except Exception:
        if last_error is not None:
            raise last_error
        raise


# --------------------------------------------------------------------------- #
# Model loading and memory reset
# --------------------------------------------------------------------------- #


def resolve_model_path(model_path: str, cache_dir: str, local_files_only: bool) -> str:
    """Return a local directory for a path or cached HF repo id."""
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            repo_id=model_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            max_workers=8,
        )
    except Exception as e:
        mode = "local cache" if local_files_only else "HuggingFace"
        raise RuntimeError(
            f"Could not resolve {model_path!r} from {mode}. "
            f"Set HF_HOME/cache_dir correctly or pass --allow_download. Original error: {e}"
        ) from e


def load_model(model_path: str, cache_dir: str, device: str, local_files_only: bool, direct_device_map: bool = False):
    resolved_model_path = resolve_model_path(model_path, cache_dir, local_files_only)
    print(f"[MemoryLLM-BABILong] Loading from: {resolved_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(resolved_model_path, local_files_only=True)
    cfg_path = os.path.join(resolved_model_path, "config.json")
    with open(cfg_path) as f:
        raw = json.load(f)
    if "rope_theta" not in raw and isinstance(raw.get("rope_scaling"), dict):
        raw_rope_theta = raw["rope_scaling"].get("rope_theta", 500000.0)
    else:
        raw_rope_theta = raw.get("rope_theta", 500000.0)
    config.rope_theta = raw_rope_theta
    if isinstance(getattr(config, "rope_scaling", None), dict) and "rope_type" in config.rope_scaling and "type" not in config.rope_scaling:
        config.rope_scaling["type"] = config.rope_scaling["rope_type"]
    print(f"[MemoryLLM-BABILong] Patched config.rope_theta = {raw_rope_theta}")

    load_kwargs = {
        "config": config,
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if direct_device_map:
        load_kwargs["device_map"] = {"": device}
        load_kwargs["low_cpu_mem_usage"] = True

    try:
        model = MemoryLLM.from_pretrained(
            resolved_model_path,
            attn_implementation="flash_attention_2",
            **load_kwargs,
        )
        print("[MemoryLLM-BABILong] Loaded with flash_attention_2")
    except (ValueError, ImportError) as e:
        print(f"[MemoryLLM-BABILong] FA2 failed: {e}; falling back to sdpa")
        model = MemoryLLM.from_pretrained(
            resolved_model_path,
            attn_implementation="sdpa",
            **load_kwargs,
        )

    if not direct_device_map:
        model = model.to(device)
    model.eval()
    model.config.use_cache = False

    if not hasattr(model, "memory"):
        raise RuntimeError("MemoryLLM model has no .memory attribute; cannot guarantee per-sample isolation")

    initial_state = {
        "memory": model.memory.detach().clone(),
        "initialized": model.initialized.detach().clone() if hasattr(model, "initialized") else None,
    }
    print(
        "[MemoryLLM-BABILong] Snapshotted clean memory state: "
        f"memory_shape={tuple(initial_state['memory'].shape)}, "
        f"initialized={int(initial_state['initialized'].item()) if initial_state['initialized'] is not None else 'NA'}"
    )
    return model, tokenizer, initial_state, resolved_model_path


def reset_memory(model, initial_state: dict, verify: bool = False) -> None:
    """Restore the exact clean checkpoint memory before a BABILong sample."""
    with torch.no_grad():
        model.memory.copy_(initial_state["memory"])
        if initial_state.get("initialized") is not None and hasattr(model, "initialized"):
            model.initialized.copy_(initial_state["initialized"])
    if verify:
        if not torch.equal(model.memory, initial_state["memory"]):
            raise RuntimeError("Memory reset verification failed: model.memory differs from clean snapshot")
        if initial_state.get("initialized") is not None and hasattr(model, "initialized"):
            if int(model.initialized.item()) != int(initial_state["initialized"].item()):
                raise RuntimeError("Memory reset verification failed: model.initialized differs from clean snapshot")


def inject_long_context(model, tokenizer, context: str, device: str, max_chunk: int = 1024):
    if not context or not context.strip():
        return
    ids = tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    total_len = ids.shape[1]
    if total_len < 16:
        return
    for pos in range(0, total_len, max_chunk):
        chunk = ids[:, pos:pos + max_chunk]
        if chunk.shape[1] >= 16:
            with torch.no_grad():
                model.inject_memory(chunk, update_memory=True)


def generate_answer(model, tokenizer, question_prompt: str, device: str, max_new_tokens: int = 20) -> str:
    messages = [{"role": "user", "content": question_prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    if isinstance(inputs, list):
        inputs = torch.tensor([inputs], dtype=torch.long)
    if hasattr(inputs, "input_ids"):
        inputs = inputs.input_ids
    inputs = inputs[:, 1:].to(device)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=False,
        )
    output_ids = outputs[0][inputs.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


def _sanitize_output(text) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\r", " ").replace("\n", " ")


def _write_results_csv(rows: list[dict], outfile: Path) -> None:
    df = pd.DataFrame(rows, columns=["target", "output", "question"])
    if "output" in df.columns:
        df["output"] = df["output"].map(_sanitize_output)
    df.to_csv(outfile, index=False, quoting=csv.QUOTE_ALL)


def main():
    parser = argparse.ArgumentParser(description="BABILong eval wrapper for MemoryLLM-8B-chat")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--allow_download", action="store_true", help="Allow downloading model files if not in cache")
    parser.add_argument("--results_folder", type=str, default=str(PROJECT_ROOT / "babilong_results"))
    parser.add_argument("--output_name", type=str, default="MemoryLLM-8B-chat")
    parser.add_argument("--tasks", type=str, nargs="+", default=["qa1", "qa2", "qa5"])
    parser.add_argument("--lengths", type=str, nargs="+", default=["0k", "1k", "2k", "4k", "8k", "16k", "32k"])
    parser.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=1024, help="Token chunk size for MemoryLLM context injection")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--direct_device_map", action="store_true", help="Load weights directly onto --device instead of CPU then .to(device)")
    parser.add_argument("--limit", type=int, default=100, help="Samples per task/length cell; -1 = all")
    parser.add_argument("--max_samples", type=int, default=None, help="Alias for --limit")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify_memory_reset", action="store_true", help="Exact-check the first few clean memory resets")
    parser.add_argument("--verify_resets", type=int, default=3)
    parser.add_argument("--print_examples", type=int, default=0)
    args = parser.parse_args()

    if args.max_samples is not None:
        args.limit = args.max_samples
    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard_index must be in [0, {args.num_shards}); got {args.shard_index}")

    print("[MemoryLLM-BABILong] Configuration:")
    print(f"  model_path:     {args.model_path}")
    print(f"  cache_dir:      {args.cache_dir}")
    print(f"  tasks:          {args.tasks}")
    print(f"  lengths:        {args.lengths}")
    print(f"  limit/cell:     {args.limit}")
    print(f"  shard:          {args.shard_index}/{args.num_shards}")
    print(f"  device:         {args.device}")

    model, tokenizer, initial_state, resolved_model_path = load_model(
        args.model_path,
        args.cache_dir,
        args.device,
        local_files_only=not args.allow_download,
        direct_device_map=args.direct_device_map,
    )

    use_instruction = True
    use_examples = True
    use_post_prompt = True
    use_chat_template = True
    suffix_parts = [
        "instruction_yes" if use_instruction else "instruction_no",
        "examples_yes" if use_examples else "examples_no",
        "post_prompt_yes" if use_post_prompt else "post_prompt_no",
        "chat_template_yes" if use_chat_template else "chat_template_no",
        "system_prompt_no",
    ]
    prompt_name = "_".join(suffix_parts)
    results_dir = Path(args.results_folder) / args.output_name
    results_dir.mkdir(parents=True, exist_ok=True)

    reset_checks_done = 0
    for task in tqdm(args.tasks, desc="tasks"):
        if task not in DEFAULT_PROMPTS:
            print(f"[MemoryLLM-BABILong] WARNING: unknown task {task}, skipping")
            continue
        instruction = DEFAULT_PROMPTS[task].get("instruction", "") if use_instruction else ""
        examples = DEFAULT_PROMPTS[task].get("examples", "") if use_examples else ""
        post_prompt = DEFAULT_PROMPTS[task].get("post_prompt", "") if use_post_prompt else ""

        for length in tqdm(args.lengths, desc=f"{task} lengths", leave=False):
            data = load_babilong_dataset(args.dataset_name, length)
            task_data = data[task]
            num_samples = len(task_data)
            if args.limit > 0:
                num_samples = min(num_samples, args.limit)
            sample_indices = list(range(num_samples))[args.shard_index::args.num_shards]

            shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
            outfile = results_dir / f"{task}_{length}_{prompt_name}{shard_tag}.csv"
            cfg_file = results_dir / f"{task}_{length}_{prompt_name}{shard_tag}.json"
            if outfile.exists() and not args.overwrite:
                print(f"[MemoryLLM-BABILong] Skip existing {outfile}")
                continue

            json.dump(
                {
                    "model": "MemoryLLM-8B-chat",
                    "model_path": args.model_path,
                    "resolved_model_path": resolved_model_path,
                    "dataset_name": args.dataset_name,
                    "task": task,
                    "length": length,
                    "limit": args.limit,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                    "sample_indices": sample_indices,
                    "prompt": {
                        "instruction": instruction,
                        "examples": examples,
                        "post_prompt": post_prompt,
                        "template": DEFAULT_TEMPLATE,
                        "chat_template": use_chat_template,
                        "system_prompt": "",
                    },
                    "memory_isolation": "model.memory and model.initialized restored from clean snapshot before every sample",
                },
                open(cfg_file, "w"),
                indent=2,
            )

            rows: list[dict] = []
            correct = 0
            for out_pos, idx in enumerate(tqdm(sample_indices, desc=f"{task}/{length}", leave=False)):
                sample = task_data[idx]
                target = str(sample["target"])
                context = sample["input"]
                question = sample["question"]

                verify = args.verify_memory_reset and reset_checks_done < args.verify_resets
                reset_memory(model, initial_state, verify=verify)
                if verify:
                    reset_checks_done += 1
                    print(f"[MemoryLLM-BABILong] Verified clean memory reset before sample_idx={idx}")

                inject_long_context(model, tokenizer, context, args.device, max_chunk=args.chunk_size)
                question_prompt = get_formatted_input(
                    "",
                    question,
                    examples,
                    instruction,
                    post_prompt,
                    template=DEFAULT_TEMPLATE,
                )
                output = generate_answer(model, tokenizer, question_prompt, args.device, args.max_new_tokens)
                is_correct = compare_answers(target, output, question, TASK_LABELS[task])
                correct += int(is_correct)
                rows.append({"target": target, "output": output, "question": question})

                if out_pos < args.print_examples:
                    print(
                        f"[MemoryLLM-BABILong][example] idx={idx} target={target!r} "
                        f"correct={is_correct} output={output!r}"
                    )
                if len(rows) % 10 == 0 or out_pos == len(sample_indices) - 1:
                    _write_results_csv(rows, outfile)

            _write_results_csv(rows, outfile)
            acc = correct / max(len(rows), 1)
            print(f"[MemoryLLM-BABILong] Saved {len(rows)} -> {outfile}; acc={correct}/{len(rows)}={acc:.4f}")

    print("[MemoryLLM-BABILong] Evaluation complete!")


if __name__ == "__main__":
    main()
