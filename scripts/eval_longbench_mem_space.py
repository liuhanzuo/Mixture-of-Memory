"""LongBench evaluation for the mem_space streaming memory architecture.

Evaluates Llama-3-8B + mem_space adapter on the LongBench benchmark (same tasks
as reported in the MemoryLLM paper: hotpotqa, narrativeqa, qasper,
multifieldqa_en, 2wikimqa, musique).

Our approach: chunk the full context through the model (memory accumulates across
chunks), then generate the answer from the final chunk with frozen memory bank.
No truncation needed — the memory system handles arbitrary length input.

Usage:
    # Single GPU evaluation (one shard):
    python scripts/eval_longbench_mem_space.py \
        --model_path models/Meta-Llama-3-8B-Instruct \
        --checkpoint outputs/babilong_sft_phase8b_scale_chunk1024_20260519_1416/mem_space_adapter.pt \
        --adapter_config outputs/babilong_sft_phase8b_scale_chunk1024_20260519_1416/adapter_config.json \
        --output_dir longbench_results/mem_space_p8_chunk1024 \
        --gpu_id 0 --num_gpus 8

    # Score only (aggregate all shards):
    python scripts/eval_longbench_mem_space.py --score_only \
        --output_dir longbench_results/mem_space_p8_chunk1024
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from pathlib import Path

# Proxy for HuggingFace downloads
os.environ.setdefault("http_proxy", "http://hy-proxy.woa.com:3128")
os.environ.setdefault("https_proxy", "http://hy-proxy.woa.com:3128")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM  # noqa: E402

from src.memory.mem_space import MemorySpaceConfig, apply_mem_space_to_model  # noqa: E402

# --------------------------------------------------------------------------- #
# Dataset prompt templates (from MemoryLLM longbench_config/dataset2prompt.json)
# --------------------------------------------------------------------------- #

DATASET2PROMPT = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question asconcisely as you can, using a single "
        "phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\n"
        "Now, answer the question based on the story asconcisely as you can, using "
        "a single phrase if possible. Do not provide any explanation.\n\n"
        "Question: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question as "
        "concisely as you can, using a single phrase or sentence if possible. If the "
        'question cannot be answered based on the information in the article, write '
        '"unanswerable". If the question is a yes/no question, answer "yes", "no", '
        'or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n'
        " Answer the question based on the above article as concisely as you can, "
        "using a single phrase or sentence if possible. If the question cannot be "
        'answered based on the information in the article, write "unanswerable". '
        'If the question is a yes/no question, answer "yes", "no", or "unanswerable". '
        "Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow, answer "
        "the following question based on the above text, only give me the answer "
        "and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give "
        "me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
}

# Max generation tokens per dataset (follows MemoryLLM convention)
DATASET2MAXGEN = {
    "hotpotqa": 32,
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "2wikimqa": 32,
    "musique": 32,
}

DEFAULT_DATASETS = list(DATASET2MAXGEN.keys())


# --------------------------------------------------------------------------- #
# Memory helpers (from run_babilong_mem_space.py)
# --------------------------------------------------------------------------- #


def _reset_banks(model: torch.nn.Module) -> None:
    """Wipe per-sample slot state between samples."""
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.reset()
    else:
        mem_layers = getattr(root, "_mem_space_layers", None)
        if mem_layers:
            for w in mem_layers:
                w.memory_bank.reset()
    # Reset L3 summary state
    l3_pool = getattr(root, "_l3_pool", None)
    if l3_pool is not None:
        l3_pool._current_summary = None
        l3_pool._prev_chunk_h = None
        if hasattr(l3_pool, "_chunk_summary_cache"):
            l3_pool._chunk_summary_cache = None


def _reset_l2(model: torch.nn.Module) -> None:
    """Zero L2 compressor cross-chunk state."""
    root = getattr(model, "module", model)
    comp = getattr(root, "_l2_compressor", None)
    if comp is not None:
        comp.reset()


def _freeze_banks(model: torch.nn.Module) -> None:
    """Freeze memory banks during generation."""
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

_ADAPTER_CONFIG_FIELD_MAP = {
    "writeback_warmup_steps": "writeback_gate_warmup_steps",
}


def build_mem_space_config(adapter_cfg: dict) -> MemorySpaceConfig:
    """Construct MemorySpaceConfig from adapter_config.json dict."""
    valid_fields = set(MemorySpaceConfig.__dataclass_fields__.keys())
    kwargs: dict = {}
    for k, v in adapter_cfg.items():
        target = _ADAPTER_CONFIG_FIELD_MAP.get(k, k)
        if target == "unfreeze_hidden_to_slot":
            kwargs["hidden_to_slot_frozen"] = not bool(v)
            continue
        if target in valid_fields:
            kwargs[target] = v
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
    """Build base Llama + mem_space patch + load adapter checkpoint."""
    print(f"[LongBench] Loading base model from: {model_path}")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    ).to(device)

    # H7 fix v2: snapshot rotary inv_freq in fp32
    _rope_snapshot: dict = {}
    try:
        _rot = model.model.rotary_emb
        for _name in ("inv_freq", "original_inv_freq"):
            if hasattr(_rot, _name):
                _rope_snapshot[_name] = getattr(_rot, _name).detach().to(torch.float32).clone()
    except AttributeError:
        pass

    # Patch with MemorySpaceLayer
    print(f"[LongBench] Applying mem_space patch (num_slots={mem_config.num_slots}, "
          f"top_k={mem_config.top_k}, shared_bank={mem_config.shared_memory_bank})")
    apply_mem_space_to_model(model, mem_config, layer_indices=None)
    model.to(device=device, dtype=dtype)

    # Restore rotary buffers to fp32
    try:
        _rot = model.model.rotary_emb
        for _name, _buf in _rope_snapshot.items():
            _buf = _buf.to(device=device, dtype=torch.float32)
            _rot._buffers[_name] = _buf
            setattr(_rot, _name, _buf)
        if _rope_snapshot:
            print(f"[LongBench] H7 fix v2: restored rotary buffers to float32")
    except AttributeError:
        print("[LongBench] WARNING: rotary_emb not accessible — skipping H7 fix")

    # Load checkpoint
    print(f"[LongBench] Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # Strip DDP "module." prefix
    cleaned: dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[LongBench] Loaded {len(cleaned)} keys | "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print(f"[LongBench] WARNING: first 5 unexpected: {list(unexpected)[:5]}")

    adapter_missing = [
        k for k in missing
        if any(s in k for s in (
            "slot_output_gate", "gate_param", "Q_sel", "K_sel",
            "slot_to_hidden", "hidden_to_slot",
        ))
    ]
    if adapter_missing:
        print(f"[LongBench] WARNING: {len(adapter_missing)} adapter keys NOT loaded!")

    # Fix J: force step_counter = warmup_steps
    from src.memory.mem_space.layer import MemorySpaceLayer as _MSL
    _mem_layers = getattr(model, "_mem_space_layers", [])
    _warmup_target = mem_config.writeback_gate_warmup_steps if mem_config.writeback_gate_warmup_steps > 0 else 1
    for _w in _mem_layers:
        if isinstance(_w, _MSL):
            _w.step_counter = _warmup_target
    print(f"[LongBench] Fix J: set step_counter={_warmup_target} on "
          f"{len(_mem_layers)} MemorySpaceLayer(s)")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# --------------------------------------------------------------------------- #
# Chunked generation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def generate_base_truncated(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    max_new_tokens: int,
    device: torch.device,
    max_ctx: int = 7900,
) -> str:
    """Base Llama (no adapter) baseline. Uses the standard LongBench
    middle-truncation: keep the head and tail of the context so the prompt
    fits the model window, then greedily generate. This is the fair
    quality-vs-fixed-window anchor for the memory adapter (R3-3)."""
    ids = input_ids[0]
    budget = max_ctx - max_new_tokens
    if ids.shape[0] > budget:
        half = budget // 2
        ids = torch.cat([ids[:half], ids[-(budget - half):]], dim=0)
    cur = ids.unsqueeze(0).to(device)
    generated_ids: list[int] = []
    for step in range(max_new_tokens):
        outputs = model(input_ids=cur, use_cache=False)
        logits = outputs.logits[:, -1, :]
        if step == 0 and tokenizer.eos_token_id is not None:
            logits[:, tokenizer.eos_token_id] = float("-inf")
        next_tok = logits.argmax(dim=-1, keepdim=True)
        tok_id = int(next_tok.item())
        if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id and step > 0:
            break
        generated_ids.append(tok_id)
        cur = torch.cat([cur, next_tok], dim=-1)
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_with_mem_space(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    chunk_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Stream context through memory, then generate answer.

    1. Reset memory banks (fresh state for this sample).
    2. Stream all-but-last chunks through model (memory accumulates).
    3. Freeze bank, generate from last chunk autoregressively.
    4. Unfreeze bank.
    """
    _reset_banks(model)
    _reset_l2(model)

    tokens = input_ids[0]  # [total_len]
    chunks = list(tokens.split(chunk_size))

    # Stream all-but-last chunks (memory accumulation)
    if len(chunks) > 1:
        for chunk in chunks[:-1]:
            chunk_tensor = chunk.unsqueeze(0).to(device)
            _ = model(input_ids=chunk_tensor, use_cache=False)

    # Freeze bank for generation
    _freeze_banks(model)
    try:
        cur = chunks[-1].unsqueeze(0).to(device)
        generated_ids: list[int] = []
        for step in range(max_new_tokens):
            outputs = model(input_ids=cur, use_cache=False)
            logits = outputs.logits[:, -1, :]
            if step == 0 and tokenizer.eos_token_id is not None:
                logits[:, tokenizer.eos_token_id] = float("-inf")
            next_tok = logits.argmax(dim=-1, keepdim=True)
            tok_id = int(next_tok.item())
            if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id and step > 0:
                break
            generated_ids.append(tok_id)
            cur = torch.cat([cur, next_tok], dim=-1)
    finally:
        _unfreeze_banks(model)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# SQuAD-style F1 scoring
# --------------------------------------------------------------------------- #


def normalize_answer(s: str) -> str:
    """Lower text, remove articles, punctuation, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and a single ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0

    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_f1_multi(prediction: str, answers: list[str]) -> float:
    """Max F1 across multiple reference answers."""
    if not answers:
        return 0.0
    return max(compute_f1(prediction, ans) for ans in answers)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_em_multi(prediction: str, answers: list[str]) -> float:
    """Max EM across multiple reference answers."""
    if not answers:
        return 0.0
    return max(compute_exact_match(prediction, ans) for ans in answers)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_longbench_dataset(dataset_name: str, datasets_list: list[str], data_dir: str = None):
    """Load LongBench datasets from local JSONL files or HuggingFace.

    Prefers local JSONL files at `data_dir/{ds_name}.jsonl` (pre-downloaded).
    Falls back to HuggingFace if local files not found.

    Returns dict mapping dataset_name -> list of samples.
    Each sample has: {"input": ..., "context": ..., "answers": [...], "length": ...}
    """
    # Default data directory: project_root/data/longbench_raw/data/
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "longbench_raw", "data")

    all_data = {}
    for ds_name in datasets_list:
        local_path = os.path.join(data_dir, f"{ds_name}.jsonl")

        if os.path.exists(local_path):
            # Load from local JSONL file
            print(f"[LongBench] Loading from local: {local_path}")
            samples = []
            with open(local_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    sample = {
                        "context": item.get("context", ""),
                        "input": item.get("input", ""),
                        "answers": item.get("answers", []),
                        "length": item.get("length", 0),
                        "dataset": ds_name,
                    }
                    # Handle answers format
                    if isinstance(sample["answers"], str):
                        try:
                            sample["answers"] = json.loads(sample["answers"])
                        except json.JSONDecodeError:
                            sample["answers"] = [sample["answers"]]
                    if not isinstance(sample["answers"], list):
                        sample["answers"] = [sample["answers"]]
                    samples.append(sample)
            all_data[ds_name] = samples
            print(f"[LongBench]   Loaded {len(samples)} samples for {ds_name}")
        else:
            # Fallback: try HuggingFace
            print(f"[LongBench] Local file not found, trying HuggingFace: {dataset_name}/{ds_name}")
            try:
                import datasets as hf_datasets
                data = hf_datasets.load_dataset(dataset_name, ds_name, split="test",
                                                trust_remote_code=True)
                samples = []
                for item in data:
                    sample = {
                        "context": item.get("context", ""),
                        "input": item.get("input", ""),
                        "answers": item.get("answers", []),
                        "length": item.get("length", 0),
                        "dataset": ds_name,
                    }
                    if isinstance(sample["answers"], str):
                        try:
                            sample["answers"] = json.loads(sample["answers"])
                        except json.JSONDecodeError:
                            sample["answers"] = [sample["answers"]]
                    if not isinstance(sample["answers"], list):
                        sample["answers"] = [sample["answers"]]
                    samples.append(sample)
                all_data[ds_name] = samples
                print(f"[LongBench]   Loaded {len(samples)} samples for {ds_name}")
            except Exception as e:
                print(f"[LongBench] ERROR loading {ds_name}: {e}")
                all_data[ds_name] = []

    return all_data


def format_prompt(sample: dict, dataset_name: str, tokenizer, use_chat_template: bool = True) -> str:
    """Format a LongBench sample into the prompt string.

    Uses the MemoryLLM-style prompt templates.
    """
    template = DATASET2PROMPT.get(dataset_name, DATASET2PROMPT["hotpotqa"])
    prompt = template.format(context=sample["context"], input=sample["input"])

    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    return prompt


# --------------------------------------------------------------------------- #
# Scoring mode
# --------------------------------------------------------------------------- #


def run_scoring(output_dir: str, datasets_list: list[str]):
    """Aggregate prediction JSONL files and compute metrics."""
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"[LongBench] ERROR: output_dir does not exist: {output_dir}")
        return

    results = {}
    for ds_name in datasets_list:
        # Find all shard files for this dataset
        shard_files = sorted(output_path.glob(f"{ds_name}_*.jsonl"))
        if not shard_files:
            print(f"[LongBench] No prediction files found for {ds_name}")
            continue

        # Collect all predictions (deduplicate by index if present)
        predictions = []
        seen_indices = set()
        for sf in shard_files:
            with open(sf, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    idx = item.get("index", len(predictions))
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        predictions.append(item)

        if not predictions:
            print(f"[LongBench] No predictions for {ds_name}")
            continue

        # Compute metrics
        f1_scores = []
        em_scores = []
        for pred_item in predictions:
            pred = pred_item.get("pred", "")
            answers = pred_item.get("answers", [])
            if isinstance(answers, str):
                try:
                    answers = json.loads(answers)
                except json.JSONDecodeError:
                    answers = [answers]
            if not isinstance(answers, list):
                answers = [answers]

            f1 = compute_f1_multi(pred, answers)
            em = compute_em_multi(pred, answers)
            f1_scores.append(f1)
            em_scores.append(em)

        avg_f1 = sum(f1_scores) / len(f1_scores) * 100
        avg_em = sum(em_scores) / len(em_scores) * 100
        results[ds_name] = {
            "f1": avg_f1,
            "em": avg_em,
            "n_samples": len(predictions),
        }
        print(f"[LongBench] {ds_name:20s}: F1={avg_f1:.2f}  EM={avg_em:.2f}  (n={len(predictions)})")

    # Compute average across all datasets
    if results:
        all_f1 = [v["f1"] for v in results.values()]
        all_em = [v["em"] for v in results.values()]
        avg_all_f1 = sum(all_f1) / len(all_f1)
        avg_all_em = sum(all_em) / len(all_em)
        print(f"\n[LongBench] {'AVERAGE':20s}: F1={avg_all_f1:.2f}  EM={avg_all_em:.2f}  "
              f"(across {len(results)} datasets)")
        results["AVERAGE"] = {"f1": avg_all_f1, "em": avg_all_em, "n_datasets": len(results)}

    # Save results JSON
    results_file = output_path / "scores.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[LongBench] Results saved to: {results_file}")

    return results


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #


def main():
    parser = argparse.ArgumentParser(description="LongBench evaluation for mem_space")
    parser.add_argument("--model_path", type=str, default="models/Meta-Llama-3-8B-Instruct",
                        help="Path to base Llama-3-8B model")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/babilong_sft_phase8b_scale_chunk1024_20260519_1416/mem_space_adapter.pt",
                        help="Path to mem_space adapter checkpoint")
    parser.add_argument("--adapter_config", type=str,
                        default="outputs/babilong_sft_phase8b_scale_chunk1024_20260519_1416/adapter_config.json",
                        help="Path to adapter_config.json")
    parser.add_argument("--output_dir", type=str, default="longbench_results/mem_space_p8_chunk1024",
                        help="Directory for output predictions and scores")
    parser.add_argument("--hf_dataset", type=str, default="THUDM/LongBench",
                        help="HuggingFace dataset name")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Datasets to evaluate (default: all 6 QA tasks)")
    parser.add_argument("--base_mode", action="store_true",
                        help="Evaluate plain base Llama (no mem_space adapter) with "
                             "standard LongBench middle-truncation. R3-3 anchor.")
    parser.add_argument("--chunk_size", type=int, default=1024,
                        help="Chunk size for memory accumulation")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU index for this process (for data parallelism)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Total number of GPU processes (for sharding)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--use_chat_template", action="store_true", default=True,
                        help="Wrap prompts in chat template")
    parser.add_argument("--no_chat_template", action="store_true",
                        help="Disable chat template wrapping")
    parser.add_argument("--score_only", action="store_true",
                        help="Only compute scores from existing prediction files")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples per dataset (-1 = all)")
    args = parser.parse_args()

    if args.no_chat_template:
        args.use_chat_template = False
    if args.base_mode:
        # Plain non-instruct Llama has no chat_template; always use raw prompts
        # (also matches the R3-1 mem_space protocol which used no chat template).
        args.use_chat_template = False

    datasets_list = args.datasets if args.datasets else DEFAULT_DATASETS

    # Score-only mode
    if args.score_only:
        run_scoring(args.output_dir, datasets_list)
        return

    # Resolve paths relative to PROJECT_ROOT
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)
    adapter_config_path = args.adapter_config
    if not os.path.isabs(adapter_config_path):
        adapter_config_path = os.path.join(PROJECT_ROOT, adapter_config_path)

    # When launched with CUDA_VISIBLE_DEVICES=N, only cuda:0 is visible.
    # gpu_id is used only for data sharding, not for device selection.
    device = torch.device("cuda:0")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[LongBench] Configuration:")
    print(f"  Model:         {model_path}")
    print(f"  Checkpoint:    {checkpoint_path}")
    print(f"  Adapter cfg:   {adapter_config_path}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Datasets:      {datasets_list}")
    print(f"  Chunk size:    {args.chunk_size}")
    print(f"  GPU:           {args.gpu_id}/{args.num_gpus}")
    print(f"  Chat template: {args.use_chat_template}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load adapter config
    with open(adapter_config_path, "r") as f:
        adapter_cfg = json.load(f)
    mem_config = build_mem_space_config(adapter_cfg)
    print(f"[LongBench] MemorySpaceConfig: num_slots={mem_config.num_slots}, "
          f"top_k={mem_config.top_k}, shared_bank={mem_config.shared_memory_bank}")

    # Load model
    if args.base_mode:
        print("[LongBench] BASE MODE: plain Llama (no adapter), middle-truncation")
        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
        ).to(device)
        model.eval()
    else:
        model = load_mem_space_model(
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            mem_config=mem_config,
            device=device,
            dtype=dtype,
            attn_impl=args.attn_impl,
        )

    # Load LongBench data
    all_data = load_longbench_dataset(args.hf_dataset, datasets_list)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save eval config
    config_file = output_path / f"eval_config_gpu{args.gpu_id}.json"
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=2)

    # Evaluate each dataset
    for ds_name in datasets_list:
        samples = all_data.get(ds_name, [])
        if not samples:
            print(f"[LongBench] Skipping {ds_name} (no data)")
            continue

        # Apply max_samples limit
        if args.max_samples > 0:
            samples = samples[:args.max_samples]

        # Shard samples across GPUs
        total_samples = len(samples)
        shard_size = (total_samples + args.num_gpus - 1) // args.num_gpus
        start_idx = args.gpu_id * shard_size
        end_idx = min(start_idx + shard_size, total_samples)
        shard_samples = samples[start_idx:end_idx]

        if not shard_samples:
            print(f"[LongBench] GPU {args.gpu_id}: no samples for {ds_name} "
                  f"(total={total_samples}, shard_start={start_idx})")
            continue

        max_gen = DATASET2MAXGEN.get(ds_name, 64)
        outfile = output_path / f"{ds_name}_{args.gpu_id}.jsonl"

        print(f"\n[LongBench] GPU {args.gpu_id}: Evaluating {ds_name} "
              f"({len(shard_samples)} samples, indices {start_idx}-{end_idx-1}, "
              f"max_gen={max_gen})")

        results_buffer = []
        t0 = time.time()

        for local_idx, sample in enumerate(tqdm(
            shard_samples, desc=f"{ds_name}[gpu{args.gpu_id}]", leave=True
        )):
            global_idx = start_idx + local_idx

            # Format prompt
            prompt = format_prompt(sample, ds_name, tokenizer,
                                   use_chat_template=args.use_chat_template)

            # Tokenize
            input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
            if isinstance(input_ids, list):
                input_ids = torch.tensor([input_ids], dtype=torch.long)
            input_ids = input_ids.to(device)

            n_tokens = input_ids.shape[1]
            n_chunks = (n_tokens + args.chunk_size - 1) // args.chunk_size

            # Generate
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                if args.base_mode:
                    pred = generate_base_truncated(
                        model=model,
                        input_ids=input_ids,
                        tokenizer=tokenizer,
                        max_new_tokens=max_gen,
                        device=device,
                    )
                else:
                    pred = generate_with_mem_space(
                        model=model,
                        input_ids=input_ids,
                        tokenizer=tokenizer,
                        chunk_size=args.chunk_size,
                        max_new_tokens=max_gen,
                        device=device,
                    )

            # Record result
            result = {
                "index": global_idx,
                "pred": pred,
                "answers": sample["answers"],
                "dataset": ds_name,
                "n_tokens": n_tokens,
                "n_chunks": n_chunks,
            }
            results_buffer.append(result)

            # Periodic save (every 10 samples)
            if (local_idx + 1) % 10 == 0 or local_idx == len(shard_samples) - 1:
                with open(outfile, "w") as f:
                    for r in results_buffer:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")

            # Progress logging
            if (local_idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                speed = (local_idx + 1) / elapsed
                # Quick F1 on results so far
                cur_f1s = [compute_f1_multi(r["pred"], r["answers"]) for r in results_buffer]
                avg_f1 = sum(cur_f1s) / len(cur_f1s) * 100
                print(f"  [{ds_name}] {local_idx+1}/{len(shard_samples)} done | "
                      f"{speed:.2f} samples/s | running F1={avg_f1:.1f}% | "
                      f"last_pred='{pred[:60]}...'")

        # Final save
        with open(outfile, "w") as f:
            for r in results_buffer:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Per-dataset summary
        elapsed = time.time() - t0
        f1_scores = [compute_f1_multi(r["pred"], r["answers"]) for r in results_buffer]
        avg_f1 = sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0
        print(f"[LongBench] GPU {args.gpu_id} finished {ds_name}: "
              f"F1={avg_f1:.2f}% ({len(results_buffer)} samples in {elapsed:.1f}s)")

    print(f"\n[LongBench] GPU {args.gpu_id}: All datasets complete!")
    print(f"[LongBench] Predictions saved to: {args.output_dir}")

    # If single GPU (num_gpus=1), auto-run scoring
    if args.num_gpus == 1:
        print("\n[LongBench] Running scoring (single-GPU mode)...")
        run_scoring(args.output_dir, datasets_list)


if __name__ == "__main__":
    main()
