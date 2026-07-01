"""Full-chain oracle probe for BABILong qa5.

Direction C probe: verifies whether the current oracle (which locates only the
gold-answer token's last occurrence) UNDER-ESTIMATES the true read-out upper
bound because it misses the supporting fact chunk.

For BABILong qa5 (three-arg-relations), each question has EXACTLY ONE supporting
fact of the form:
    "X gave/handed/passed the Y to Z."
The answer is always contained in this sentence. However, the current oracle
uses ``_locate_needle_chunks`` which finds the LAST occurrence of the answer
token in the token stream — but common names ("Jeff", "Mary", etc.) and common
words ("apple", "milk") appear many times in the pg19 noise, so the last
occurrence is frequently NOT the supporting fact.

This probe replaces _locate_needle_chunks with _locate_qa5_supporting_fact,
which reconstructs the expected fact sentence from (question, answer) and
searches for it directly in the input text.

Both oracle variants are run on the SAME 20-30 samples (--limit) so the scores
are directly comparable.

Usage (run on B200 .53, GPU 0):
    CUDA_VISIBLE_DEVICES=0 \\
    HF_HOME=.hf_cache HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    PYTHONPATH=third_party/babilong-pkg \\
    .venv/bin/python scripts/probe_fullchain_oracle_qa5.py \\
        --model_path models/Meta-Llama-3-8B \\
        --checkpoint outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt \\
        --adapter_config outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json \\
        --tasks qa5 \\
        --lengths 16k \\
        --limit 25 \\
        --oracle_mode both \\
        --results_folder babilong_results/probe_fullchain \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

# ---- project path setup ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_BABILONG_PKG = os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")
if os.path.isdir(_BABILONG_PKG) and _BABILONG_PKG not in sys.path:
    sys.path.insert(0, _BABILONG_PKG)

import datasets  # noqa: E402
from transformers import AutoTokenizer, LlamaForCausalLM  # noqa: E402
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402
from src.memory.mem_space import MemorySpaceConfig, apply_mem_space_to_model, _reset_fifo_memory  # noqa: E402


# ============================================================
# Inline the minimal helpers we need from run_babilong_mem_space
# (to avoid import side-effects from the full script)
# ============================================================

def _reset_banks(model):
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.reset()
    else:
        for w in getattr(root, "_mem_space_layers", None) or []:
            w.memory_bank.reset()
    l3_pool = getattr(root, "_l3_pool", None)
    if l3_pool is not None:
        l3_pool._current_summary = None
        l3_pool._prev_chunk_h = None
        if hasattr(l3_pool, "_chunk_summary_cache"):
            l3_pool._chunk_summary_cache = None
        if hasattr(l3_pool, "_prev_chunk_token_mask"):
            l3_pool._prev_chunk_token_mask = None
        if hasattr(l3_pool, "_prev_summary"):
            l3_pool._prev_summary = None


def _reset_l2(model):
    root = getattr(model, "module", model)
    comp = getattr(root, "_l2_compressor", None)
    if comp is not None:
        comp.reset()


def _freeze_banks(model):
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.frozen = True
        return
    for w in getattr(root, "_mem_space_layers", []):
        w.memory_bank.frozen = True


def _unfreeze_banks(model):
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.frozen = False
        return
    for w in getattr(root, "_mem_space_layers", []):
        w.memory_bank.frozen = False


def _set_fifo_oracle_needle(model, needle_chunks):
    root = getattr(model, "module", model)
    nset = set(int(c) for c in needle_chunks) if needle_chunks else None
    for w in getattr(root, "_mem_space_layers", []) or []:
        w._fifo_oracle_needle_chunks = nset
        w._fifo_buf = []
        w._fifo_buf_abs_idx = []
        w._fifo_write_seq = 0


# ============================================================
# Original oracle: find LAST occurrence of answer token
# ============================================================

def _locate_needle_chunks_original(input_ids, target, tokenizer, chunk_size):
    """Current oracle: last occurrence of answer token in token stream."""
    ids = input_ids[0].tolist()
    L = len(ids)
    tgt = (target or "").strip()
    if not tgt:
        return None
    cands = []
    for variant in (tgt, " " + tgt):
        enc = tokenizer.encode(variant, add_special_tokens=False)
        if enc:
            cands.append(enc)

    def _scan_last(needle_ids):
        ns = len(needle_ids)
        if ns == 0 or ns > L:
            return None
        for s in range(L - ns, -1, -1):
            if ids[s:s + ns] == needle_ids:
                return s
        return None

    chunks = set()
    for needle_ids in cands:
        start = _scan_last(needle_ids)
        if start is None:
            # fallback: try dropping leading token
            for drop in range(1, min(4, len(needle_ids))):
                start = _scan_last(needle_ids[drop:])
                if start is not None:
                    break
        if start is None:
            continue
        end = min(L - 1, start + len(needle_ids) - 1)
        for p in range(start, end + 1):
            chunks.add(p // chunk_size)
    return chunks or None


# ============================================================
# Full-chain oracle: find supporting fact sentence for qa5
# ============================================================

# All bAbI agent names in qa5
_BABI_AGENTS = ['Fred', 'Bill', 'Jeff', 'Mary', 'Sandra', 'Daniel', 'John',
                'Bob', 'Susan', 'Julie']

def _parse_qa5_question(question: str):
    """Parse a qa5 question and return structured elements.

    Returns dict with keys:
        type: 'who_receiver' | 'what_obj' | 'who_giver' | 'who_receiver2' | 'unknown'
        and optionally: giver, receiver, obj (strings, title-cased for agents)
    """
    q = question.strip().rstrip("?").lower()
    # 'Who did X give the Y to?' -> receiver = answer
    m = re.match(r'who did (\w+) give (?:the |a )?(\w+) to', q)
    if m:
        return {'type': 'who_receiver', 'giver': m.group(1).title(), 'obj': m.group(2).lower()}
    # 'What did X give to Y?' -> obj = answer
    m = re.match(r'what did (\w+) give to (\w+)', q)
    if m:
        return {'type': 'what_obj', 'giver': m.group(1).title(), 'receiver': m.group(2).title()}
    # 'Who gave the Y?' -> giver = answer
    m = re.match(r'who gave (?:the |a )?(\w+)', q)
    if m:
        return {'type': 'who_giver', 'obj': m.group(1).lower()}
    # 'Who received the Y?' -> receiver = answer
    m = re.match(r'who received (?:the |a )?(\w+)', q)
    if m:
        return {'type': 'who_receiver2', 'obj': m.group(1).lower()}
    # 'Who gave the Y to X?' -> giver = answer
    m = re.match(r'who gave (?:the |a )?(\w+) to (\w+)', q)
    if m:
        return {'type': 'who_giver_to', 'obj': m.group(1).lower(), 'receiver': m.group(2).title()}
    # 'Who did X give the Y?' (without 'to') -> receiver = answer
    m = re.match(r'who did (\w+) give (?:the |a )?(\w+)', q)
    if m:
        return {'type': 'who_receiver', 'giver': m.group(1).title(), 'obj': m.group(2).lower()}
    return {'type': 'unknown'}


_TRANSFER_VERB_PAT = r'(?:gave|handed|passed|gave back)'


def _locate_qa5_supporting_fact(input_text: str, question: str, answer: str,
                                 ids: list, tokenizer, chunk_size: int):
    """Locate the supporting fact chunk for a qa5 question.

    Reconstructs the expected fact sentence from (question, answer),
    then searches for it in the raw input text (pre-tokenized).
    Returns a set of chunk indices, or None on failure.

    The fact takes the form:
        'X gave/handed/passed the Y to Z.'
    All variants of verbs are tried.
    """
    q_elem = _parse_qa5_question(question)
    answer = answer.strip()
    if not answer:
        return None

    # Build regex patterns for the expected supporting fact
    patterns = []

    if q_elem['type'] == 'who_receiver':
        giver = re.escape(q_elem['giver'])
        obj = re.escape(q_elem['obj'])
        receiver = re.escape(answer)
        patterns.append(rf'{giver} {_TRANSFER_VERB_PAT} (?:the |a )?{obj} to {receiver}')

    elif q_elem['type'] == 'what_obj':
        giver = re.escape(q_elem['giver'])
        receiver = re.escape(q_elem['receiver'])
        obj = re.escape(answer)
        patterns.append(rf'{giver} {_TRANSFER_VERB_PAT} (?:the |a )?{obj} to {receiver}')

    elif q_elem['type'] == 'who_giver':
        obj = re.escape(q_elem['obj'])
        giver = re.escape(answer)
        patterns.append(rf'{giver} {_TRANSFER_VERB_PAT} (?:the |a )?{obj} to \w+')

    elif q_elem['type'] == 'who_giver_to':
        obj = re.escape(q_elem['obj'])
        receiver = re.escape(q_elem['receiver'])
        giver = re.escape(answer)
        patterns.append(rf'{giver} {_TRANSFER_VERB_PAT} (?:the |a )?{obj} to {receiver}')

    elif q_elem['type'] == 'who_receiver2':
        obj = re.escape(q_elem['obj'])
        receiver = re.escape(answer)
        patterns.append(rf'\w+ {_TRANSFER_VERB_PAT} (?:the |a )?{obj} to {receiver}')

    else:
        # unknown question type — fall back to answer token search
        return None

    chunks = set()
    L = len(ids)
    for pat in patterns:
        for m in re.finditer(pat, input_text, re.IGNORECASE):
            matched_text = m.group()
            # Try to locate this text span in the tokenized stream
            # Try a few prefix variants (space, no space) for tokenizer boundary
            for prefix in (' ' + matched_text, matched_text):
                enc = tokenizer.encode(prefix, add_special_tokens=False)
                if not enc:
                    continue
                # Search forward (FIRST occurrence = the actual embedded fact,
                # because pg19 noise rarely has this exact bAbI sentence)
                for j in range(L - len(enc) + 1):
                    if ids[j:j + len(enc)] == enc:
                        for p in range(j, min(L, j + len(enc))):
                            chunks.add(p // chunk_size)
                        break
    return chunks or None


def _locate_needle_chunks_fullchain(input_ids, input_text, target, question,
                                    tokenizer, chunk_size, task='qa5'):
    """Full-chain oracle: for qa5 use supporting-fact locator; else fall back."""
    if task == 'qa5':
        ids = input_ids[0].tolist()
        result = _locate_qa5_supporting_fact(
            input_text=input_text,
            question=question,
            answer=target,
            ids=ids,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
        )
        if result:
            return result
        # Fall back to original oracle if SF not found
    return _locate_needle_chunks_original(input_ids, target, tokenizer, chunk_size)


# ============================================================
# Dataset loading (minimal, from run_babilong_mem_space.py)
# ============================================================

def _load_babilong_from_arrow_cache(dataset_name, split_name, cache_dir):
    root = cache_dir / dataset_name.replace("/", "___") / split_name
    arrow_roots = [p for p in root.glob("*/*") if p.is_dir() and any(p.glob("babilong-*.arrow"))]
    if not arrow_roots:
        return None
    arrow_root = max(arrow_roots, key=lambda p: p.stat().st_mtime)
    data = {
        p.stem.removeprefix("babilong-"): datasets.Dataset.from_file(str(p))
        for p in sorted(arrow_root.glob("babilong-*.arrow"))
    }
    return data or None


def load_babilong_dataset(dataset_name, split_name, cache_dir=None):
    roots = []
    if cache_dir:
        roots.append(Path(cache_dir).expanduser())
    for env in ("HF_DATASETS_CACHE", "HF_HOME"):
        if os.environ.get(env):
            root = Path(os.environ[env]).expanduser()
            roots.append(root if env == "HF_DATASETS_CACHE" else root / "datasets")
    roots += [Path(PROJECT_ROOT) / ".cache/huggingface/datasets",
               Path.home() / ".cache/huggingface/datasets"]
    seen, unique = set(), []
    for r in roots:
        k = str(r.absolute())
        if k not in seen:
            seen.add(k)
            unique.append(r)
    last_error = None
    for candidate in unique:
        try:
            data = datasets.load_dataset(dataset_name, split_name,
                                         cache_dir=str(candidate),
                                         download_mode="reuse_dataset_if_exists")
            return data
        except Exception as e:
            last_error = e
            data = _load_babilong_from_arrow_cache(dataset_name, split_name, candidate)
            if data is not None:
                return data
    try:
        return datasets.load_dataset(dataset_name, split_name,
                                     download_mode="reuse_dataset_if_exists")
    except Exception:
        raise last_error


# ============================================================
# Adapter config loader
# ============================================================

_ADAPTER_CONFIG_FIELD_MAP = {"writeback_warmup_steps": "writeback_gate_warmup_steps"}


def _load_mem_config_from_adapter(adapter_config_path):
    if not adapter_config_path:
        return MemorySpaceConfig()
    with open(adapter_config_path) as f:
        ac = json.load(f)
    mc_fields = {f.name for f in MemorySpaceConfig.__dataclass_fields__.values()} \
        if hasattr(MemorySpaceConfig, "__dataclass_fields__") else {}
    kwargs = {}
    for k, v in ac.items():
        mapped = _ADAPTER_CONFIG_FIELD_MAP.get(k, k)
        if mapped in mc_fields:
            kwargs[mapped] = v
    mc = MemorySpaceConfig(**kwargs)
    # Dynamic (non-dataclass) HNST v2 tree attrs — set so apply_mem_space_to_model
    # reconstructs the tree pool and the full_model ckpt loads without noise.
    for _dyn, _default in (("use_tree_summary", False), ("tree_summary_heads", 8),
                           ("tree_summary_layers", 1), ("tree_summary_ffn_mult", 2)):
        setattr(mc, _dyn, ac.get(_dyn, _default))
    return mc


# ============================================================
# Model loading
# ============================================================

def load_mem_space_model(model_path, checkpoint_path, mem_config, device, dtype, attn_impl="sdpa"):
    print(f"[probe] Loading base model from {model_path} …")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    apply_mem_space_to_model(model, mem_config)
    model.to(device)
    if checkpoint_path:
        print(f"[probe] Loading checkpoint {checkpoint_path} …")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate_sample(model, input_ids, tokenizer, chunk_size, max_new_tokens, device,
                     oracle_token_chunks=None):
    """Streaming generation with optional oracle-token window.

    Mirrors generate_with_mem_space in run_babilong_mem_space.py exactly:
    - use_cache=False throughout (required for FIFO memory layers)
    - append next_tok to cur tensor each step (NOT past_key_values)
    - suppress EOS on first step (matches H6 behaviour)
    """
    _reset_banks(model)
    _reset_l2(model)

    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))

    if len(chunks) > 1:
        for chunk in chunks[:-1]:
            _ = model(input_ids=chunk.unsqueeze(0).to(device), use_cache=False)

    _freeze_banks(model)

    if oracle_token_chunks:
        n_chunks = len(chunks)
        last_idx = n_chunks - 1
        sel = sorted(c for c in oracle_token_chunks if 0 <= c < last_idx)
        pieces = [chunks[c] for c in sel] + [chunks[-1]]
        cur = torch.cat(pieces, dim=0).unsqueeze(0).to(device)
    else:
        cur = chunks[-1].unsqueeze(0).to(device)

    generated_ids = []
    try:
        for step in range(max_new_tokens):
            out = model(input_ids=cur, use_cache=False)
            logits = out.logits[:, -1, :]  # [1, vocab_size]
            if step == 0 and tokenizer.eos_token_id is not None:
                # Suppress EOS on first token (matches H6 behaviour)
                logits[:, tokenizer.eos_token_id] = float("-inf")
            next_tok = logits.argmax(dim=-1, keepdim=True)  # [1, 1]
            tok_id = int(next_tok.item())
            if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id and step > 0:
                break
            generated_ids.append(tok_id)
            cur = torch.cat([cur, next_tok], dim=-1)  # append, NOT use_cache
    finally:
        _unfreeze_banks(model)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ============================================================
# Scoring
# ============================================================

def compare_answers(target, output):
    from babilong.babilong_utils import compare_answers as _ca
    return _ca(target, output)


def score_df(df):
    correct = sum(compare_answers(r["target"], r["output"]) for _, r in df.iterrows())
    total = len(df)
    return correct, total


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Full-chain oracle probe for qa5")
    parser.add_argument("--model_path", type=str, default="models/Meta-Llama-3-8B")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt")
    parser.add_argument("--adapter_config", type=str,
                        default="outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json")
    parser.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    parser.add_argument("--tasks", nargs="+", default=["qa5"])
    parser.add_argument("--lengths", nargs="+", default=["16k"])
    parser.add_argument("--limit", type=int, default=25,
                        help="Max samples per cell (for quick probe)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the first --limit samples into this many shards (round-robin)")
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="Which shard (0-based) this process handles")
    parser.add_argument("--oracle_mode", type=str, default="both",
                        choices=["original", "fullchain", "both"],
                        help="'original' = current answer-token oracle; "
                             "'fullchain' = supporting-fact oracle; "
                             "'both' = run both on identical samples (default)")
    parser.add_argument("--results_folder", type=str,
                        default="babilong_results/probe_fullchain")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--use_instruction", action="store_true", default=True)
    parser.add_argument("--use_examples", action="store_true", default=False)
    parser.add_argument("--use_post_prompt", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    print(f"[probe] oracle_mode={args.oracle_mode}, limit={args.limit}")
    print(f"[probe] tasks={args.tasks}, lengths={args.lengths}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    mem_config = _load_mem_config_from_adapter(args.adapter_config)
    print(f"[probe] MemorySpaceConfig: num_slots={mem_config.num_slots}, "
          f"top_k={mem_config.top_k}")

    model = load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        mem_config=mem_config,
        device=device,
        dtype=dtype,
        attn_impl=args.attn_impl,
    )

    results_summary = {}

    for task in args.tasks:
        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"] if args.use_instruction else "",
            "examples":    DEFAULT_PROMPTS[task]["examples"]    if args.use_examples    else "",
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"] if args.use_post_prompt else "",
            "template":    DEFAULT_TEMPLATE,
        }

        for split_name in args.lengths:
            print(f"\n[probe] task={task}, length={split_name}")
            try:
                data = load_babilong_dataset(args.dataset_name, split_name)
                task_data = data[task]
            except Exception as e:
                print(f"[ERROR] Failed to load {task}/{split_name}: {e}")
                continue

            n = min(len(task_data), args.limit) if args.limit > 0 else len(task_data)
            sample_indices = list(range(n))
            if args.num_shards > 1:
                sample_indices = sample_indices[args.shard_idx::args.num_shards]
                print(f"[probe] shard {args.shard_idx}/{args.num_shards}: {len(sample_indices)} samples")

            outdir = Path(args.results_folder)
            outdir.mkdir(parents=True, exist_ok=True)
            shard_tag = f"_shard{args.shard_idx}of{args.num_shards}" if args.num_shards > 1 else ""

            # Collect results for both oracle modes
            rows_original = []
            rows_fullchain = []

            # Stats
            orig_sf_match = 0  # how many times original oracle found SF chunk
            full_sf_found = 0  # how many times fullchain found SF chunk
            disagree = 0       # how many times they disagree

            for idx in tqdm(sample_indices, desc=f"{task}/{split_name}"):
                s = task_data[idx]
                target = s["target"]
                question = s["question"]
                input_text = s["input"]

                full_input = get_formatted_input(
                    input_text,
                    question,
                    prompt_cfg["examples"],
                    prompt_cfg["instruction"],
                    prompt_cfg["post_prompt"],
                    template=prompt_cfg["template"],
                )
                ids = tokenizer.encode(full_input, add_special_tokens=True,
                                       return_tensors="pt")
                if isinstance(ids, list):
                    ids = torch.tensor([ids], dtype=torch.long)
                input_ids = ids.to(device)

                # ---- Locate needle chunks for both oracles ----
                orig_chunks = _locate_needle_chunks_original(
                    input_ids, target, tokenizer, args.chunk_size
                )
                full_chunks = _locate_needle_chunks_fullchain(
                    input_ids, input_text, target, question,
                    tokenizer, args.chunk_size, task=task
                )

                # Track agreement
                if full_chunks:
                    full_sf_found += 1
                if orig_chunks and full_chunks and orig_chunks == full_chunks:
                    orig_sf_match += 1
                elif orig_chunks != full_chunks:
                    disagree += 1

                # ---- Run generation with original oracle ----
                if args.oracle_mode in ("original", "both"):
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        out_orig = generate_sample(
                            model, input_ids, tokenizer, args.chunk_size,
                            args.max_new_tokens, device,
                            oracle_token_chunks=orig_chunks,
                        )
                    rows_original.append({"target": target, "output": out_orig, "question": question,
                                          "orig_chunk": str(orig_chunks), "full_chunk": str(full_chunks)})

                # ---- Run generation with full-chain oracle ----
                if args.oracle_mode in ("fullchain", "both"):
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        out_full = generate_sample(
                            model, input_ids, tokenizer, args.chunk_size,
                            args.max_new_tokens, device,
                            oracle_token_chunks=full_chunks,
                        )
                    rows_fullchain.append({"target": target, "output": out_full, "question": question,
                                           "orig_chunk": str(orig_chunks), "full_chunk": str(full_chunks)})

            # ---- Score ----
            print(f"\n[probe] Chunk-locator stats for {task}/{split_name} (n={n}):")
            print(f"  full-chain found SF chunk: {full_sf_found}/{n} ({100*full_sf_found/max(1,n):.1f}%)")
            print(f"  oracle == fullchain: {n-disagree}/{n} ({100*(n-disagree)/max(1,n):.1f}%)")
            print(f"  oracle DISAGREES with fullchain: {disagree}/{n} ({100*disagree/max(1,n):.1f}%)")

            key = f"{task}/{split_name}"
            results_summary[key] = {}

            if rows_original:
                df_orig = pd.DataFrame(rows_original)
                correct_orig, total_orig = score_df(df_orig)
                score_orig = 100.0 * correct_orig / max(1, total_orig)
                print(f"  ORIGINAL oracle: {correct_orig}/{total_orig} = {score_orig:.1f}%")
                results_summary[key]["original_oracle"] = score_orig
                out_csv = outdir / f"{task}_{split_name}_original_oracle_n{n}{shard_tag}.csv"
                df_orig.to_csv(out_csv, index=False, quoting=csv.QUOTE_ALL)
                print(f"  -> {out_csv}")

            if rows_fullchain:
                df_full = pd.DataFrame(rows_fullchain)
                correct_full, total_full = score_df(df_full)
                score_full = 100.0 * correct_full / max(1, total_full)
                print(f"  FULL-CHAIN oracle: {correct_full}/{total_full} = {score_full:.1f}%")
                results_summary[key]["fullchain_oracle"] = score_full
                out_csv = outdir / f"{task}_{split_name}_fullchain_oracle_n{n}{shard_tag}.csv"
                df_full.to_csv(out_csv, index=False, quoting=csv.QUOTE_ALL)
                print(f"  -> {out_csv}")

            # Also write a chunk-disagreement diagnostic
            if args.oracle_mode == "both" and rows_original and rows_fullchain:
                delta = results_summary[key].get("fullchain_oracle", 0) - \
                        results_summary[key].get("original_oracle", 0)
                verdict = ("ORACLE UNDERESTIMATED (fullchain >> original)"
                           if delta > 5 else
                           "Oracle adequate (fullchain ≈ original)" if abs(delta) <= 5
                           else "fullchain lower than original (unexpected)")
                print(f"\n  VERDICT: delta = {delta:+.1f}pp → {verdict}")
                results_summary[key]["delta_fullchain_minus_original"] = delta
                results_summary[key]["verdict"] = verdict

    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, v in results_summary.items():
        print(f"{key}:")
        for metric, val in v.items():
            print(f"  {metric}: {val}")

    summary_path = Path(args.results_folder) / f"fullchain_oracle_summary{('_shard%dof%d'%(args.shard_idx,args.num_shards)) if args.num_shards>1 else ''}.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n[probe] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
