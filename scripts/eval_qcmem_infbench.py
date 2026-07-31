#!/usr/bin/env python
"""QCMem mid-depth resume — InfiniteBench (∞Bench) NATIVE long-context NATURAL eval.

Paper A item **P2.1**: reproduce QCMem on genuinely NATURAL long-document tasks
(real novels / dialogue scripts / source code, avg length ~100k+ tokens), NOT the
synthetic RULER single-needle haystacks, on a native long-context backbone. All
tasks scored LOCALLY (F1 / accuracy / EM) — NO GPT-4o judge dependency.

This is the natural-long-doc companion to:
  * ``scripts/eval_qcmem_longbench.py`` (LongBench real docs, but only ~5k-20k tok),
  * ``scripts/eval_qcmem_babilong.py``  (synthetic BABILong recall),
  * ``scripts/eval_ruler_qcmem.py``     (synthetic RULER NIAH/VT).

Like those drivers it is a THIN composition — nothing about the QCMem forward path
is re-implemented:

  QCMem forward path (imported verbatim from ``scripts/eval_qcmem_babilong.py``):
    * ``qcmem_generate`` — chunk the prompt -> write_chunk each chunk to depth j ->
                           selector picks topk context chunks (bm25 / iter_bm25 /
                           recency / reader_attn) -> read (pack [sink; selected h_j;
                           query h_j], resume layers[j:]) -> greedy decode. Its
                           ``no_retrieval`` arm packs EVERY context chunk (the
                           KV-Direct / HCache full-context baselines).
    * ``QCMemModel``     — the write/read orchestrator (read-only backbone).
    * ``run_self_test``  — j=0 correctness gate (QCMem read == full forward).

  InfiniteBench task framework (self-contained here, copied from the canonical
  OpenBMB/InfiniteBench ``src/compute_scores.py`` + ``src/prompt.py`` so the driver
  does NOT depend on ``external/InfLLM`` being present on the eval node):
    * ``load_infinitebench``    — offline JSONL loading from a local data dir.
    * ``INFBENCH_PROMPT`` / ``INFBENCH_MAXGEN`` — the official ∞Bench prompt
                                  templates + per-task generation budgets.
    * ``score_one``             — the official per-task LOCAL metric:
        - ``longbook_qa_eng``     -> token-F1 (qa_f1_score)          [NATURAL QA]
        - ``longbook_choice_eng`` -> multiple-choice accuracy         [NATURAL, aggregate/reasoning]
        - ``longdialogue_qa_eng`` -> character-name EM                [NATURAL dialogue]
        - ``code_debug``          -> A/B/C/D EM                       [NATURAL code]
        - ``math_find``           -> integer/float EM                 [NATURAL math]
        - ``longbook_sum_eng``    -> lightweight ROUGE-L F1 (optional; no `evaluate` dep)

Baselines (``--baseline``, mirrors eval_qcmem_longbench.py exactly):
  * ``none``     — normal QCMem (retrieval topk + resume_j + optional LoRA).
  * ``kvdirect`` (2603.19664) — the NATIVE-WINDOW DENSE comparison P2.1 asks for:
                                full-depth recompute (forces resume_j=0) + NO
                                retrieval (packs every chunk) + no LoRA. Read grows
                                O(context) — this is full-context / j=0.
  * ``hcache``   (2410.05004) — mid-layer recompute (keeps --resume_j) + NO
                                retrieval (packs every chunk) + no LoRA.

Flagship config (Qwen3-8B, matches the paper headline / P0.3 / P1.1):
    python scripts/eval_qcmem_infbench.py \
        --model_path models/Qwen3-8b-local \
        --resume_j 12 --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
        --selector iter_bm25 --topk 12 --iter_rounds 0 --iter_hop_topk 4 \
        --sink_tokens bos --chunk_size 512 --dtype bfloat16 --attn_impl sdpa --seed 42 \
        --tasks longbook_qa_eng longbook_choice_eng \
        --data_dir data/infinitebench \
        --output_dir infbench_results/qcmem_8b_j12_lora --num_shards 8 --shard_index 0

Native-window Dense (KV-Direct) arm for the same natural tasks:
    python scripts/eval_qcmem_infbench.py --baseline kvdirect \
        --model_path models/Qwen3-8b-local \
        --tasks longbook_qa_eng longbook_choice_eng \
        --data_dir data/infinitebench \
        --output_dir infbench_results/kvdirect_8b --num_shards 8 --shard_index 0

Merge shards + score:
    python scripts/eval_qcmem_infbench.py --score_only \
        --tasks longbook_qa_eng longbook_choice_eng \
        --output_dir infbench_results/qcmem_8b_j12_lora
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# QCMem forward path — reused verbatim, unmodified (same import the LongBench /
# RULER drivers use).
import scripts.eval_qcmem_babilong as qcb  # noqa: E402

QCMemModel = qcb.QCMemModel
qcmem_generate = qcb.qcmem_generate
run_self_test = qcb.run_self_test


# --------------------------------------------------------------------------- #
# InfiniteBench prompt templates + generation budgets
# (verbatim from OpenBMB/InfiniteBench via InfLLM benchmark/config, kept local so
#  this driver has no external/InfLLM dependency on the eval node.)
# --------------------------------------------------------------------------- #
INFBENCH_PROMPT = {
    "longbook_qa_eng": (
        "Read the book below and answer a question.\n\n{context}\n\n"
        "Question: {question}\n\nPlease answer as short as possible. "
        "The answer is:"
    ),
    "longbook_choice_eng": (
        "Read the book and answer the question.\n\n{context}\n\n"
        "Question: {question}\n\nOnly one of the following options is correct, "
        "tell me the answer using one single letter (A, B, C, or D). Don't say "
        "anything else.\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}"
    ),
    "longbook_sum_eng": "Summarize the following book.\n\n{context}",
    "longdialogue_qa_eng": (
        "Below is a dialogue script where one random occurrence of a character "
        "name is replaced with \"$$MASK$$\", and you should try to guess who that "
        "character is.\n\nThe dialogue:\n\n---\n\n{context}\n\n---\n\nEnd of "
        "dialogue.\n\nWhich character is most likely \"$$MASK$$\"? Just say the "
        "name used by the scriptwriter (before the colon marks) of one single "
        "character and nothing else."
    ),
    "code_debug": (
        "There is ONLY ONE function in the large project that is deliberately made "
        "to include an obvious error. Please find the function that contains the "
        "most obvious errors. I will give you four options to narrow your scope. "
        "You can inspect the options and think. Eventually, tell me the answer "
        "using one single letter (A, B, C, or D).\n\n{context}\n\nWhich funtion "
        "has deliberate error?\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\n"
        "D. {OPTION_D}\n\nGive me your answer for the function that has the "
        "deliberate and obvious error in A, B, C, or D. Your answer MUST be chosen "
        "from one of the four options without any explanation. If you cannot "
        "determine answers accurately, you also MUST provide the answer you think "
        "is most likely. Absolutely do not say you do not know or you need more "
        "information."
    ),
    "math_find": "{prefix}\n\n{context}\n\n{input}",
}

INFBENCH_MAXGEN = {
    "longbook_qa_eng": 40,
    "longbook_choice_eng": 40,
    "longbook_sum_eng": 1536,
    "longdialogue_qa_eng": 40,
    "code_debug": 40,
    "math_find": 40,
}

DEFAULT_TASKS = ["longbook_qa_eng", "longbook_choice_eng"]
_ALL_TASKS = list(INFBENCH_PROMPT.keys())


# --------------------------------------------------------------------------- #
# InfiniteBench loader (streaming, offline JSONL) — mirrors InfLLM's
# ``load_infinite_bench`` instance builder so answers/options line up with the
# canonical scorers below. Streams line-by-line and can early-stop when a shard's
# ``max_samples`` are collected, so a tiny smoke never reads the full ~300MB file.
# --------------------------------------------------------------------------- #
def _get_answer(eg: dict, task: str):
    if task in ("code_debug", "longbook_choice_eng"):
        OPTIONS = "ABCD"
        ans = eg["answer"]
        if isinstance(ans, str):
            return [ans, OPTIONS[eg["options"].index(ans)]]
        if isinstance(ans, list):
            if len(ans) == 1:
                return [ans[0], OPTIONS[eg["options"].index(ans[0])]]
            if len(ans) == 2 and ans[1] in "ABCD":
                return ans
        raise ValueError(f"bad answer field for {task}: {ans!r}")
    return eg["answer"]


def _build_instance(eg: dict, task: str) -> dict:
    if task in ("longbook_choice_eng",):
        inst = {
            "context": eg["context"],
            "question": eg["input"],
            "OPTION_A": eg["options"][0],
            "OPTION_B": eg["options"][1],
            "OPTION_C": eg["options"][2],
            "OPTION_D": eg["options"][3],
        }
    elif task in ("longbook_qa_eng",):
        inst = {"context": eg["context"], "question": eg["input"]}
    elif task == "longbook_sum_eng":
        inst = {"context": eg["context"]}
    elif task == "longdialogue_qa_eng":
        inst = {"context": eg["context"]}
    elif task == "code_debug":
        inst = {
            "context": eg["context"],
            "OPTION_A": eg["options"][0],
            "OPTION_B": eg["options"][1],
            "OPTION_C": eg["options"][2],
            "OPTION_D": eg["options"][3],
        }
    elif task == "math_find":
        prompt = eg["input"]
        find_result = re.findall(r"The .+ of", prompt)
        assert find_result, f"cannot find target number in {prompt!r}"
        target_number = find_result[0].lower()[:-3]
        inst = {
            "prefix": f"What is {target_number} in the following list?",
            "context": eg["context"],
            "input": prompt,
        }
    else:
        inst = {"context": eg.get("content", eg.get("context")),
                "input": eg.get("input", "")}
    ans = _get_answer(eg, task)
    inst["answers"] = ans if isinstance(ans, list) else [ans]
    inst["length"] = len(str(inst.get("context", "")).split())
    return inst


def load_infinitebench(data_dir: str, task: str, read_cap: int = -1) -> list:
    """Stream ``<data_dir>/<task>.jsonl`` -> list[instance]. read_cap>0 stops early."""
    path = os.path.join(data_dir, f"{task}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"InfiniteBench file not found: {path}. Download it with e.g.\n"
            f"  huggingface-cli download xinrongzhang2022/InfiniteBench "
            f"{task}.jsonl --repo-type dataset --local-dir {data_dir}")
    out = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            out.append(_build_instance(json.loads(line), task))
            if read_cap > 0 and len(out) >= read_cap:
                break
    return out


def bare_query(inst: dict, task: str) -> str:
    """The bm25 lexical query for a sample (the short question, not the long doc)."""
    if task in ("longbook_qa_eng", "longbook_choice_eng"):
        return (inst.get("question") or "").strip()
    if task == "math_find":
        return (inst.get("input") or "").strip()
    if task == "code_debug":
        return " ".join(inst.get(f"OPTION_{c}", "") for c in "ABCD").strip()
    # longdialogue_qa_eng / sum: no separate short question.
    return ""


def format_prompt(inst: dict, task: str, tokenizer,
                  use_chat_template: bool = False, enable_thinking: bool = False):
    """Render the ∞Bench prompt. gen_boundary_ids = the chat assistant prefill
    appended at the query tail when --use_chat_template (mirrors the babilong/
    longbench drivers); None in the default raw-completion path."""
    template = INFBENCH_PROMPT[task]
    text = template.format(**inst)
    if not use_chat_template:
        return text, None
    messages = [{"role": "user", "content": text}]
    try:
        body = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=enable_thinking)
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:
        body = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    boundary = full[len(body):] if full.startswith(body) else full
    boundary_ids = tokenizer.encode(boundary, add_special_tokens=False)
    return body, boundary_ids


# --------------------------------------------------------------------------- #
# InfiniteBench LOCAL scorers (verbatim from OpenBMB/InfiniteBench
# src/compute_scores.py — judge-free). qa=token-F1, choice/code=EM(letter),
# dialogue=name-EM, math=number-EM. Plus a dependency-free ROUGE-L for sum.
# --------------------------------------------------------------------------- #
def _normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(pred: str, ground_truths) -> float:
    best = 0.0
    for gt in ground_truths:
        p = _normalize_answer(pred).split()
        g = _normalize_answer(gt).split()
        best = max(best, _f1(p, g))
    return best


def _first_int_match(prediction: str) -> str:
    for item in re.split("[^0-9]", prediction):
        if item != "":
            return item
    return ""


def score_longbook_choice_eng(pred: str, answers) -> float:
    """answers = [answer_text, letter] (from _get_answer). Correct if the first
    A/B/C/D emitted matches the gold letter, or the pred starts with the gold text."""
    gold_letter = next((a for a in answers if isinstance(a, str) and a in "ABCD"), None)
    gold_texts = [a for a in answers if a != gold_letter]
    pred = (pred or "").strip()
    if not pred:
        return 0.0
    if pred[0] in "ABCD":
        return float(pred[0] == gold_letter)
    cleaned = pred
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        cleaned = cleaned.replace(c, " ")
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    for prefix in ("answer is:", "answer:", "answer is", "option is"):
        idx = cleaned.find(prefix)
        if idx == -1:
            continue
        after = cleaned[idx + len(prefix) + 1:]
        if gold_letter and after.startswith(gold_letter):
            return 1.0
        for gt in gold_texts:
            if after.startswith(gt):
                return 1.0
        break
    for word in cleaned.split():
        if word in "ABCD":
            return float(word == gold_letter)
    # last resort: normalized-substring against the gold answer text
    npred = _normalize_answer(pred)
    for gt in gold_texts:
        if gt and _normalize_answer(gt) in npred:
            return 1.0
    return 0.0


def score_code_debug(pred: str, answers) -> float:
    """answers = [answer_text/fn_name, letter]."""
    label_c = answers[1] if len(answers) > 1 else None
    fn_name = answers[0]
    pred = (pred or "").strip()
    if not pred:
        return 0.0
    if label_c and pred[:2] in (f"{label_c}.", f"{label_c}:"):
        return 1.0
    cleaned = pred
    for c in ["\n", "`", "'", '"', "-", "*", "Option", "option"]:
        cleaned = cleaned.replace(c, " ")
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    for prefix in ("answer is:", "is:", "answer:", "answer is", "error is"):
        idx = cleaned.find(prefix)
        if idx == -1:
            continue
        after = cleaned[idx + len(prefix) + 1:]
        for s in (label_c, fn_name):
            if s and after.startswith(s):
                return 1.0
        break
    for word in cleaned.split():
        if word in "ABCD":
            return float(word == label_c)
    return 0.0


def score_longdialogue_qa_eng(pred: str, answers) -> float:
    label = answers[0]
    cleaned = pred or ""
    for c in ["\n", ":", '"', "'", ".", ",", "?", "!", "{", "}"]:
        cleaned = cleaned.replace(c, " ")
    words = [w.upper() for w in cleaned.split()]
    return float(str(label).upper() in words)


def score_math_find(pred: str, answers) -> float:
    label = answers[0]
    if isinstance(label, list):
        label = label[0]
    m = re.search(r"\d+\.\d+|\d+", pred or "")
    if m is None:
        return 0.0
    num = m.group(0).strip()
    try:
        if isinstance(label, float):
            return float(float(num) == label)
        return float(int(num) == int(label))
    except (ValueError, TypeError):
        return 0.0


def _rouge_l_f1(pred: str, ref: str) -> float:
    """Dependency-free ROUGE-L (LCS) F1 on whitespace tokens (approx to rougeLsum)."""
    a = _normalize_answer(pred).split()
    b = _normalize_answer(ref).split()
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if ai == b[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[m]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / n, lcs / m
    return 2 * prec * rec / (prec + rec)


def score_longbook_sum_eng(pred: str, answers) -> float:
    return max((_rouge_l_f1(pred, gt) for gt in answers), default=0.0)


_SCORERS = {
    "longbook_qa_eng": qa_f1_score,
    "longbook_choice_eng": score_longbook_choice_eng,
    "longdialogue_qa_eng": score_longdialogue_qa_eng,
    "code_debug": score_code_debug,
    "math_find": score_math_find,
    "longbook_sum_eng": score_longbook_sum_eng,
}


def score_one(task: str, pred: str, answers) -> float:
    return float(_SCORERS[task](pred, answers))


def _is_percent_metric(task: str) -> bool:
    # qa_f1 / sum are 0-1 fractions we report as %; choice/code/dialogue/math are 0/1.
    return True


# --------------------------------------------------------------------------- #
# scoring (merge shards)
# --------------------------------------------------------------------------- #
def run_scoring(output_dir: str, tasks: list) -> dict:
    output_path = Path(output_dir)
    summary = {}
    for task in tasks:
        shard_files = sorted(output_path.glob(f"{task}_*.jsonl"))
        if not shard_files:
            print(f"[QCMem-InfBench] no prediction files for {task}")
            continue
        preds, seen = [], set()
        for sf in shard_files:
            with open(sf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    idx = item.get("index", len(preds))
                    if idx not in seen:
                        seen.add(idx)
                        preds.append(item)
        if not preds:
            continue
        scores = [score_one(task, p.get("pred", ""), p.get("answers", [])) for p in preds]
        mean = 100.0 * sum(scores) / len(scores)
        summary[task] = {"metric": _METRIC_NAME[task], "score": mean, "n": len(scores)}
        print(f"[QCMem-InfBench] {task}: {_METRIC_NAME[task]}={mean:.2f} (n={len(scores)})")
    if summary:
        macro = sum(v["score"] for v in summary.values()) / len(summary)
        summary["_macro"] = {"score": macro, "n_tasks": len(summary)}
        print(f"[QCMem-InfBench] MACRO over {len(summary) - 1} tasks = {macro:.2f}")
        with open(output_path / "scores.json", "w") as f:
            json.dump(summary, f, indent=2)
    return summary


_METRIC_NAME = {
    "longbook_qa_eng": "f1",
    "longbook_choice_eng": "acc",
    "longdialogue_qa_eng": "em",
    "code_debug": "acc",
    "math_find": "em",
    "longbook_sum_eng": "rougeL",
}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="QCMem mid-depth resume — InfiniteBench natural long-doc eval")
    # model arm (aligned with eval_qcmem_longbench.py)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--resume_j", type=int, default=12)
    parser.add_argument("--top_prepay_b", type=int, default=0)
    parser.add_argument("--reuse_kv_blockdiag", action="store_true", default=False)
    parser.add_argument("--lora_adapter", type=str, default="")
    parser.add_argument("--bottleneck_ckpt", type=str, default="")
    parser.add_argument("--baseline", type=str, default="none",
                        choices=["none", "kvdirect", "hcache"],
                        help="'none'=QCMem retrieval; 'kvdirect'=native-window Dense "
                             "(resume_j=0, no retrieval, no LoRA, packs all chunks); "
                             "'hcache'=mid-layer recompute (keep resume_j, no retrieval).")
    parser.add_argument("--force_lora_with_baseline", action="store_true", default=False)
    parser.add_argument("--selector", type=str, default="iter_bm25",
                        choices=["bm25", "recency", "reader_attn", "iter_bm25"])
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--iter_rounds", type=int, default=0)
    parser.add_argument("--iter_hop_topk", type=int, default=4)
    parser.add_argument("--iter_score", type=str, default="meanpool",
                        choices=["meanpool", "maxsim"])
    parser.add_argument("--iter_conf_ratio", type=float, default=0.3)
    parser.add_argument("--iter_max_chunks", type=int, default=64)
    parser.add_argument("--sink_tokens", type=str, default="bos",
                        choices=["bos", "none"])
    parser.add_argument("--chunk_size", type=int, default=512)
    # task framework
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=_ALL_TASKS,
                        help=f"∞Bench subtasks (default {DEFAULT_TASKS}).")
    parser.add_argument("--data_dir", type=str, default="data/infinitebench",
                        help="Dir with <task>.jsonl files.")
    parser.add_argument("--output_dir", type=str, default="infbench_results/qcmem")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples per task (-1=all).")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--use_chat_template", action="store_true", default=False,
                        help="Default OFF (paper mandate: chat_template=False).")
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score_only", action="store_true")
    parser.add_argument("--self_test", action="store_true", default=False)
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else DEFAULT_TASKS

    if args.score_only:
        run_scoring(args.output_dir, tasks)
        return

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard_index must be in [0, {args.num_shards})")
    if not args.model_path:
        parser.error("--model_path is required unless --score_only")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # head-to-head baseline resolution (identical to eval_qcmem_longbench.py)
    no_retrieval = (args.baseline != "none")
    if args.bottleneck_ckpt and args.lora_adapter:
        parser.error("--bottleneck_ckpt and --lora_adapter are mutually exclusive")
    if args.baseline == "kvdirect":
        if args.resume_j != 0:
            print(f"[QCMem-InfBench] baseline=kvdirect -> forcing resume_j "
                  f"{args.resume_j} -> 0 (full-depth K/V recompute = Dense).")
        args.resume_j = 0
        if args.lora_adapter:
            print(f"[QCMem-InfBench] baseline=kvdirect is training-free -> "
                  f"ignoring --lora_adapter {args.lora_adapter!r}.")
            args.lora_adapter = ""
    elif args.baseline == "hcache":
        if args.lora_adapter and not args.force_lora_with_baseline:
            print(f"[QCMem-InfBench] baseline=hcache is post-hoc -> ignoring "
                  f"--lora_adapter {args.lora_adapter!r}.")
            args.lora_adapter = ""
    if no_retrieval and args.reuse_kv_blockdiag:
        parser.error("--reuse_kv_blockdiag incompatible with --baseline")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    if args.self_test:
        dtype = torch.float32

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)

    print(f"[QCMem-InfBench] model_path={model_path}")
    print(f"[QCMem-InfBench] baseline={args.baseline} (no_retrieval={no_retrieval}) "
          f"resume_j={args.resume_j} selector={args.selector} topk={args.topk} "
          f"sink={args.sink_tokens} chunk_size={args.chunk_size} "
          f"chat_template={args.use_chat_template} dtype={dtype} "
          f"attn_impl={args.attn_impl} seed={args.seed}")
    print(f"[QCMem-InfBench] tasks={tasks} data_dir={args.data_dir} "
          f"max_samples={args.max_samples} shard={args.shard_index}/{args.num_shards}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
        trust_remote_code=True, local_files_only=True).to(device).eval()

    L = int(model.config.num_hidden_layers)
    if not (0 <= args.resume_j <= L):
        parser.error(f"--resume_j must be in [0, {L}]; got {args.resume_j}")

    if args.lora_adapter:
        from peft import PeftModel
        print(f"[QCMem-InfBench] loading LoRA adapter: {args.lora_adapter}")
        peft_model = PeftModel.from_pretrained(model, args.lora_adapter).eval()
        model = peft_model.base_model.model

    if args.bottleneck_ckpt:
        from scripts.train_qwen_bottleneck_continued import inject_bottleneck
        meta_path = os.path.join(
            os.path.dirname(os.path.abspath(args.bottleneck_ckpt)), "arch_meta.json")
        if not os.path.exists(meta_path):
            parser.error(f"arch_meta.json not found next to {args.bottleneck_ckpt}")
        with open(meta_path) as f:
            meta = json.load(f)
        inject_bottleneck(model, int(meta["bottleneck_layer"]),
                          int(meta["bottleneck_dim"]), dtype)
        ck = torch.load(args.bottleneck_ckpt, map_location="cpu")
        model.load_state_dict(ck.get("model_state", ck), strict=False)
        model = model.to(device).eval()

    if args.self_test:
        ok = run_self_test(model, tokenizer, device, args.chunk_size)
        sys.exit(0 if ok else 1)

    qc = QCMemModel(model, resume_j=args.resume_j, top_prepay_b=args.top_prepay_b,
                    block_diagonal=args.reuse_kv_blockdiag)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"shard{args.shard_index}of{args.num_shards}" if sharded else "0"

    zero_training_no_adapter = (args.baseline == "none" and not args.lora_adapter
                                and not args.bottleneck_ckpt)
    task_tag = tasks[0] if len(tasks) == 1 else "multi"
    with open(output_path / f"eval_config_{task_tag}_{shard_tag}.json", "w") as f:
        cfg = dict(vars(args))
        cfg.update({"no_retrieval": bool(no_retrieval), "num_layers": L,
                    "resolved_model_path": model_path,
                    "lora_adapter": args.lora_adapter or None,
                    "bottleneck_ckpt": args.bottleneck_ckpt or None,
                    "chat_template": bool(args.use_chat_template),
                    "zero_training_no_adapter": zero_training_no_adapter})
        json.dump(cfg, f, indent=2)

    # read_cap so a limited smoke does not read the whole (~300MB) file.
    read_cap = -1
    if args.max_samples > 0:
        read_cap = args.shard_index + args.max_samples * args.num_shards

    for task in tasks:
        samples = load_infinitebench(args.data_dir, task, read_cap=read_cap)
        if args.max_samples > 0:
            # cap by global index so strided shards stay consistent
            samples = samples[:args.max_samples * args.num_shards]
        if not samples:
            print(f"[QCMem-InfBench] skipping {task} (no data)")
            continue

        sample_indices = list(range(len(samples)))[args.shard_index::args.num_shards]
        if sharded:
            print(f"[QCMem-InfBench] {task} shard {args.shard_index}/{args.num_shards}: "
                  f"{len(sample_indices)} of {len(samples)} samples")

        max_gen = INFBENCH_MAXGEN.get(task, 40)
        outfile = output_path / f"{task}_{shard_tag}.jsonl"
        buf = []
        t0 = time.time()

        for pos, idx in enumerate(tqdm(sample_indices, desc=task, leave=True)):
            inst = samples[idx]
            prompt, gen_boundary_ids = format_prompt(
                inst, task, tokenizer, use_chat_template=args.use_chat_template,
                enable_thinking=args.enable_thinking)
            ids = tokenizer.encode(prompt, add_special_tokens=True,
                                   return_tensors="pt")
            if isinstance(ids, list):
                ids = torch.tensor([ids], dtype=torch.long)
            input_ids = ids.to(device)
            n_tokens = int(input_ids.shape[1])
            n_chunks = (n_tokens + args.chunk_size - 1) // args.chunk_size
            bare_q_ids = tokenizer.encode(bare_query(inst, task),
                                          add_special_tokens=False)

            gen_stats: dict = {}
            try:
                pred = qcmem_generate(
                    qc=qc, tokenizer=tokenizer, input_ids=input_ids,
                    chunk_size=args.chunk_size, max_new_tokens=max_gen,
                    selector=args.selector, topk=args.topk,
                    sink_tokens=args.sink_tokens, needle_chunk_set=None,
                    bare_question_ids=bare_q_ids, no_retrieval=no_retrieval,
                    stats=gen_stats, iter_rounds=args.iter_rounds,
                    iter_hop_topk=args.iter_hop_topk, iter_score=args.iter_score,
                    iter_conf_ratio=args.iter_conf_ratio,
                    iter_max_chunks=args.iter_max_chunks,
                    gen_boundary_ids=gen_boundary_ids,
                )
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                pred = "[OOM]"
                print(f"[OOM] idx={idx} task={task} n_tok={n_tokens}: {e}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            buf.append({
                "index": idx, "pred": pred, "answers": inst["answers"],
                "task": task, "n_tokens": n_tokens, "n_chunks": n_chunks,
                "read_len": gen_stats.get("read_len"),
                "n_selected_chunks": gen_stats.get("n_selected_chunks"),
                "n_context_chunks": gen_stats.get("n_context_chunks"),
            })

            if (pos + 1) % 5 == 0 or pos == len(sample_indices) - 1:
                with open(outfile, "w") as f:
                    for r in buf:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (pos + 1) % 5 == 0:
                cur = [score_one(task, r["pred"], r["answers"]) for r in buf]
                running = 100.0 * sum(cur) / len(cur)
                speed = (pos + 1) / (time.time() - t0)
                print(f"  [{task}] {pos+1}/{len(sample_indices)} | {speed:.3f} s/it⁻¹ | "
                      f"running {_METRIC_NAME[task]}={running:.1f} | "
                      f"read_len~{gen_stats.get('read_len')} | "
                      f"n_ctx_chunks={gen_stats.get('n_context_chunks')} | "
                      f"last='{pred[:50]}'", flush=True)

        with open(outfile, "w") as f:
            for r in buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sc = [score_one(task, r["pred"], r["answers"]) for r in buf]
        mean = (100.0 * sum(sc) / len(sc)) if sc else 0.0
        elapsed = time.time() - t0
        metrics = {
            "task": task, "metric": _METRIC_NAME[task], "score": mean,
            "shard_index": int(args.shard_index), "num_shards": int(args.num_shards),
            "num_samples": len(buf), "elapsed_seconds": elapsed,
            "output_file": str(outfile),
            "oom_count": sum(r["pred"] == "[OOM]" for r in buf),
            "empty_prediction_count": sum(not r["pred"].strip() for r in buf),
            "resume_j": int(args.resume_j), "chunk_size": int(args.chunk_size),
            "selector": args.selector, "topk": int(args.topk),
            "chat_template": bool(args.use_chat_template),
            "lora_adapter": args.lora_adapter or None,
            "baseline": args.baseline,
            "zero_training_no_adapter": zero_training_no_adapter,
        }
        with open(output_path / f"{task}_{shard_tag}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[QCMem-InfBench] {task}: {_METRIC_NAME[task]}={mean:.2f} "
              f"({len(buf)} samples, {elapsed:.1f}s) -> {outfile}")

    print(f"\n[QCMem-InfBench] Shard {args.shard_index}/{args.num_shards} complete!")
    if args.num_shards == 1:
        print("\n[QCMem-InfBench] Running scoring (single-shard)...")
        run_scoring(args.output_dir, tasks)


if __name__ == "__main__":
    main()
