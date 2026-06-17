"""Dialog-memory evaluation (LongMemEval + LOCOMO) for the mem_space architecture.

Both benchmarks probe *conversational memory* (as opposed to document/synthetic
retrieval): a long multi-session chat history + a QA that requires extracting /
reasoning over facts mentioned in earlier sessions.

  * LongMemEval (xiaowu0162/longmemeval): 500 questions over multi-session chat
    haystacks. Question types: single-session-{user,assistant,preference},
    multi-session, knowledge-update, temporal-reasoning, plus 30 abstention
    (_abs) questions whose correct answer is "I don't know / not mentioned".
    Officially scored by a GPT judge; here we use literal/F1 + a type-aware
    substring-accuracy proxy (abstention via refusal-phrase detection).
  * LOCOMO (snap-research/locomo, locomo10.json): 10 long two-speaker
    conversations (up to ~19 sessions each) with ~199 QA per conversation.
    Categories: 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop,
    5=adversarial(unanswerable). Officially scored by F1; we use F1 + EM.

For mem_space (W0 closed-book readout):
    The whole dialog history is rendered to text, chunked, and streamed through
    the model so the 128-slot memory bank accumulates it; then the bank is
    frozen and the answer is generated from the final (question) chunk. This is
    the same inference path as BABILong W0 / LongBench.

For base Llama-3-8B (open-book anchor):
    The dialog history + question is middle-truncated to the model window and
    answered with full attention.

Usage (mem_space, single GPU shard):
    python scripts/eval_dialogmem_mem_space.py \
        --benchmark longmemeval \
        --data data/dialogmem/longmemeval/longmemeval_oracle \
        --model_path models/Meta-Llama-3-8B \
        --checkpoint outputs/mem_space_p11_chunk1024_deltarule_normreadout/mem_space_adapter.pt \
        --adapter_config outputs/mem_space_p11_chunk1024_deltarule_normreadout/adapter_config.json \
        --output_dir dialogmem_results/lme_oracle_p11 \
        --chunk_size 1024 --max_samples 100 \
        --num_shards 4 --shard_index 0

    # base anchor:
    python scripts/eval_dialogmem_mem_space.py --base_mode \
        --benchmark longmemeval --data ... --model_path models/Meta-Llama-3-8B \
        --output_dir dialogmem_results/lme_oracle_base ...

    # score-only (aggregate shards):
    python scripts/eval_dialogmem_mem_space.py --score_only \
        --benchmark longmemeval --output_dir dialogmem_results/lme_oracle_p11
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

os.environ.setdefault("http_proxy", "http://hy-proxy.woa.com:3128")
os.environ.setdefault("https_proxy", "http://hy-proxy.woa.com:3128")

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM  # noqa: E402

# Reuse the exact W0 model-loading + streaming-generation path from BABILong /
# LongBench so the inference is identical to our other evals.
from scripts.eval_longbench_mem_space import (  # noqa: E402
    build_mem_space_config,
    load_mem_space_model,
    generate_with_mem_space,
    generate_base_truncated,
)

# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

_LME_INSTRUCTION = (
    "You are a helpful assistant with memory of a previous conversation between "
    "the user and the assistant, organized into dated sessions. Read the "
    "conversation history, then answer the user's question using only the "
    "information in the history. If the answer is a date or number, give it "
    "directly. Answer as concisely as possible. If the information needed to "
    "answer is not in the conversation history, reply exactly \"I don't know\"."
)

_LOCOMO_INSTRUCTION = (
    "You are a helpful assistant with memory of a long conversation between "
    "{spa} and {spb}, organized into dated sessions. Read the conversation "
    "history, then answer the question using only the information in the "
    "history. Answer as concisely as possible with a short phrase, date, or "
    "number. Do not explain."
)


def render_longmemeval_history(sample: dict) -> str:
    """Flatten LongMemEval multi-session haystack into a dated transcript."""
    parts = []
    dates = sample.get("haystack_dates", [])
    for si, session in enumerate(sample["haystack_sessions"]):
        date = dates[si] if si < len(dates) else ""
        parts.append(f"\n=== Session {si + 1}{(' (' + date + ')') if date else ''} ===")
        for turn in session:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            speaker = "User" if role == "user" else "Assistant"
            parts.append(f"{speaker}: {content}")
    return "\n".join(parts)


def render_locomo_history(conv: dict) -> str:
    """Flatten a LOCOMO conversation (session_1..N + dates) into a transcript."""
    parts = []
    i = 1
    while f"session_{i}" in conv:
        date = conv.get(f"session_{i}_date_time", "")
        parts.append(f"\n=== Session {i}{(' (' + date + ')') if date else ''} ===")
        for turn in conv[f"session_{i}"]:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            parts.append(f"{speaker}: {text}")
        i += 1
    return "\n".join(parts)


def build_longmemeval_samples(data_path: str) -> list[dict]:
    data = json.load(open(data_path))
    samples = []
    for d in data:
        history = render_longmemeval_history(d)
        is_abs = d["question_id"].endswith("_abs")
        prompt = (
            f"{_LME_INSTRUCTION}\n\n# Conversation history\n{history}\n\n"
            f"# Question\n{d['question']}\n\n# Answer\n"
        )
        samples.append({
            "id": d["question_id"],
            "prompt": prompt,
            "answers": [str(d["answer"])],
            "question_type": d["question_type"],
            "is_abstention": is_abs,
        })
    return samples


def build_locomo_samples(data_path: str) -> list[dict]:
    data = json.load(open(data_path))
    samples = []
    for conv_idx, d in enumerate(data):
        conv = d["conversation"]
        spa = conv.get("speaker_a", "Speaker A")
        spb = conv.get("speaker_b", "Speaker B")
        instr = _LOCOMO_INSTRUCTION.format(spa=spa, spb=spb)
        history = render_locomo_history(conv)
        for qi, qa in enumerate(d["qa"]):
            ans = qa.get("answer", qa.get("adversarial_answer", ""))
            prompt = (
                f"{instr}\n\n# Conversation history\n{history}\n\n"
                f"# Question\n{qa['question']}\n\n# Answer\n"
            )
            samples.append({
                "id": f"conv{conv_idx}_qa{qi}",
                "prompt": prompt,
                "answers": [str(ans)],
                "category": qa.get("category", -1),
                "is_abstention": qa.get("category") == 5,
            })
    return samples


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

_REFUSAL_RE = re.compile(
    r"\b(i don'?t know|not (mentioned|sure|provided|available|specified)|"
    r"no (information|mention|record)|cannot (find|determine|answer)|"
    r"unanswerable|isn'?t (mentioned|provided)|wasn'?t mentioned)\b",
    re.IGNORECASE,
)


def normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        return "".join(ch for ch in t if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(pred: str, gt: str) -> float:
    pt = normalize_answer(pred).split()
    gt_t = normalize_answer(gt).split()
    if len(pt) == 0 and len(gt_t) == 0:
        return 1.0
    if len(pt) == 0 or len(gt_t) == 0:
        return 0.0
    common = collections.Counter(pt) & collections.Counter(gt_t)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pt)
    r = num_same / len(gt_t)
    return 2 * p * r / (p + r)


def compute_f1_multi(pred: str, answers: list[str]) -> float:
    return max((compute_f1(pred, a) for a in answers), default=0.0)


def substring_acc(pred: str, answers: list[str]) -> float:
    """Type-agnostic accuracy proxy: gold answer (normalized) appears as a
    substring of the normalized prediction, OR high token overlap. This better
    matches the LongMemEval/LOCOMO 'GPT-judge says correct' notion than strict
    EM for short factual answers."""
    np = normalize_answer(pred)
    for a in answers:
        na = normalize_answer(a)
        if na and (na in np or np in na):
            return 1.0
    return 0.0


def score_sample(item: dict) -> dict:
    pred = item.get("pred", "")
    answers = item.get("answers", [])
    is_abs = item.get("is_abstention", False)
    refused = bool(_REFUSAL_RE.search(pred)) or pred.strip() == ""
    if is_abs:
        # correct iff the model refuses / says it doesn't know
        acc = 1.0 if refused else 0.0
        f1 = acc
    else:
        f1 = compute_f1_multi(pred, answers)
        acc = max(substring_acc(pred, answers), 1.0 if f1 >= 0.5 else 0.0)
    return {"f1": f1, "acc": acc, "refused": refused}


# --------------------------------------------------------------------------- #
# Scoring mode (aggregate shards)
# --------------------------------------------------------------------------- #


def run_scoring(output_dir: str, benchmark: str):
    output_path = Path(output_dir)
    shard_files = sorted(output_path.glob("preds_*.jsonl"))
    if not shard_files:
        print(f"[dialogmem] No prediction files in {output_dir}")
        return
    preds = []
    seen = set()
    for sf in shard_files:
        with open(sf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item["id"] not in seen:
                    seen.add(item["id"])
                    preds.append(item)
    if not preds:
        print("[dialogmem] No predictions")
        return

    overall_f1, overall_acc = [], []
    by_type = collections.defaultdict(lambda: {"f1": [], "acc": []})
    type_key = "question_type" if benchmark == "longmemeval" else "category"
    for item in preds:
        sc = score_sample(item)
        overall_f1.append(sc["f1"])
        overall_acc.append(sc["acc"])
        t = str(item.get(type_key, "?"))
        by_type[t]["f1"].append(sc["f1"])
        by_type[t]["acc"].append(sc["acc"])

    n = len(preds)
    results = {
        "benchmark": benchmark,
        "n_samples": n,
        "overall_f1": sum(overall_f1) / n * 100,
        "overall_acc": sum(overall_acc) / n * 100,
        "by_type": {},
    }
    print(f"\n[dialogmem] {benchmark}  n={n}")
    print(f"  OVERALL  acc={results['overall_acc']:.2f}  F1={results['overall_f1']:.2f}")
    for t in sorted(by_type):
        v = by_type[t]
        m = len(v["acc"])
        a = sum(v["acc"]) / m * 100
        f = sum(v["f1"]) / m * 100
        results["by_type"][t] = {"acc": a, "f1": f, "n": m}
        print(f"  {t:28s} acc={a:6.2f}  F1={f:6.2f}  (n={m})")

    with open(output_path / "scores.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[dialogmem] saved {output_path / 'scores.json'}")
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description="Dialog-memory eval for mem_space")
    p.add_argument("--benchmark", choices=["longmemeval", "locomo"], required=True)
    p.add_argument("--data", type=str, help="Path to dataset json file")
    p.add_argument("--model_path", type=str, default="models/Meta-Llama-3-8B")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--adapter_config", type=str, default="")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--base_mode", action="store_true",
                   help="Plain base Llama (no adapter), middle-truncation anchor")
    p.add_argument("--chunk_size", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--swa_eval_chunks", type=int, default=0)
    p.add_argument("--base_max_ctx", type=int, default=7900)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--score_only", action="store_true")
    args = p.parse_args()

    if args.score_only:
        run_scoring(args.output_dir, args.benchmark)
        return

    # Resolve paths
    def _abs(pth):
        return pth if (not pth or os.path.isabs(pth)) else os.path.join(PROJECT_ROOT, pth)
    model_path = _abs(args.model_path)
    data_path = _abs(args.data)

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[dialogmem] benchmark={args.benchmark} base_mode={args.base_mode}")
    print(f"  data={data_path}  chunk_size={args.chunk_size}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build samples
    if args.benchmark == "longmemeval":
        samples = build_longmemeval_samples(data_path)
    else:
        samples = build_locomo_samples(data_path)
    print(f"[dialogmem] total samples: {len(samples)}")

    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    # Shard
    shard = [s for i, s in enumerate(samples) if i % args.num_shards == args.shard_index]
    print(f"[dialogmem] shard {args.shard_index}/{args.num_shards}: {len(shard)} samples")

    # Load model
    if args.base_mode:
        print("[dialogmem] BASE MODE: plain Llama (no adapter), middle-truncation")
        model = LlamaForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
        ).to(device)
        model.eval()
    else:
        adapter_config_path = _abs(args.adapter_config)
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        mem_config = build_mem_space_config(adapter_cfg)
        mem_config.l3_recon_max_positions = args.chunk_size
        model = load_mem_space_model(
            model_path=model_path,
            checkpoint_path=_abs(args.checkpoint),
            mem_config=mem_config,
            device=device,
            dtype=dtype,
            attn_impl=args.attn_impl,
        )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outfile = output_path / f"preds_{args.shard_index}of{args.num_shards}.jsonl"

    buffer = []
    t0 = time.time()
    for li, sample in enumerate(tqdm(shard, desc=f"{args.benchmark}[s{args.shard_index}]")):
        prompt = sample["prompt"]
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
        n_tokens = input_ids.shape[1]
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            if args.base_mode:
                pred = generate_base_truncated(
                    model=model, input_ids=input_ids, tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens, device=device,
                    max_ctx=args.base_max_ctx,
                )
            else:
                pred = generate_with_mem_space(
                    model=model, input_ids=input_ids, tokenizer=tokenizer,
                    chunk_size=args.chunk_size, max_new_tokens=args.max_new_tokens,
                    device=device, swa_eval_chunks=args.swa_eval_chunks,
                )
        rec = {k: sample[k] for k in sample if k != "prompt"}
        rec["pred"] = pred
        rec["n_tokens"] = n_tokens
        buffer.append(rec)

        if (li + 1) % 10 == 0 or li == len(shard) - 1:
            with open(outfile, "w") as f:
                for r in buffer:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if (li + 1) % 20 == 0:
            accs = [score_sample(r)["acc"] for r in buffer]
            speed = (li + 1) / (time.time() - t0)
            print(f"  [{li+1}/{len(shard)}] {speed:.2f} s/s | running acc="
                  f"{sum(accs)/len(accs)*100:.1f}% | last='{pred[:50]}'")

    with open(outfile, "w") as f:
        for r in buffer:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    accs = [score_sample(r)["acc"] for r in buffer]
    print(f"[dialogmem] shard {args.shard_index} done: acc="
          f"{sum(accs)/len(accs)*100:.2f}% ({len(buffer)} samples, "
          f"{time.time()-t0:.1f}s)")

    if args.num_shards == 1:
        run_scoring(args.output_dir, args.benchmark)


if __name__ == "__main__":
    main()
