#!/usr/bin/env python3
"""Closed-book open-ended QA eval for the Qwen3 prune-then-heal probe
(Paper B P2.5, protocol-complete cross-family port of the OLMo-2 P0.3 harness).

Ported from scripts/eval_olmo2_closedbook_qa.py (2026-08-03). The ONLY change is
the import source of the model-construction helpers: load_base_model /
load_pruned_model / _log come from eval_qwen3_probe2_ppl (Qwen3 family) instead
of the OLMo-2 module. Every protocol detail -- the fixed `Question: ...\nAnswer:`
prompt, add_special_tokens=False (base protocol), greedy do_sample=False
num_beams=1 max_new_tokens=32 with left padding, first-line-of-completion
prediction, SQuAD-style normalisation, em / contains / f1 max-over-aliases,
majority_em floor, the strided 8-GPU shard + count-summed merge, and the stable
item_ids -- is family-agnostic and UNCHANGED, so this harness is byte-for-byte
protocol-aligned with the OLMo-2 P0.3 harness. Cross-family: only compare
base-normalised direction/recovery vs OLMo, never absolute numbers.

What this measures
------------------
Strict zero-shot, NO retrieval, free-form greedy generation on three closed-book
factual-QA benchmarks (same three, same splits, same sample counts as OLMo P0.3):
  * PopQA    (akariasai/PopQA, test, 14267 q)              -- long-tail entity facts.
  * TriviaQA (mandarjoshi/trivia_qa, rc.nocontext, validation, 17944 q).
  * NQ-open  (google-research-datasets/nq_open, validation, 3610 q).
The model answers from parametric memory alone (no passages/context). Alongside
MMLU this is the "independent knowledge task" the P2.4/P2.5 diagnostic uses: if
MMLU (esp. the content protocol) recovers but PopQA/TriviaQA/NQ do not, the MMLU
gain is interface/format adaptation rather than restored knowledge.

Prompt (fixed, base protocol -- matches the Qwen3 letter-MMLU / downstream framing)
    Question: {question}\nAnswer:
No system prompt, no chat template, no few-shot exemplars, add_special_tokens=
False. add_bos=0 default; Qwen3 tokenizer has bos_token=None / add_bos_token=
False so --add_bos 1 is a no-op (no BOS token to prepend) -> naturally
equivalent to OLMo's no-BOS base protocol. Greedy decode (do_sample=False,
num_beams=1) for --max_new_tokens tokens, keep the first line of the completion.

Metrics (per benchmark, reported together; nothing hidden)
    em          exact match after SQuAD-style normalisation, max over gold aliases
    contains    a normalised gold alias is a substring of the normalised prediction
                (PopQA's own "accuracy" convention; headline for PopQA)
    f1          token-level F1 after normalisation, max over gold aliases
Baselines (open-ended QA has no fixed chance level; we report honest floors)
    empty_em                always-"" prediction (0.0 by construction, sanity)
    majority_em             always predict the single most-frequent gold answer
                            string in the eval set (strongest label-only baseline).

Sharding: examples strided examples[shard_index::num_shards] PER benchmark
(same scheme as the PPL / downstream / MMLU evals). Each shard writes
shard{i}of{N}.json (sum counts) and, always, per-example predictions
per_example_{task}_shard{i}of{N}.jsonl (item_id = shard_index + local_idx *
num_shards). --merge sums the shard counts (em = sum(em_hits)/sum(n)) and
concatenates the per-example jsonl sorted by item_id -- enabling wrong->right /
right->wrong and paired analysis exactly like the MC harnesses.

Model construction: NO drift. Imports load_pruned_model / load_base_model from
eval_qwen3_probe2_ppl.py (Qwen3 shell rebuild with cfg.layer_types reset +
strict load, fp32 master weights, bf16-autocast forward), so the shell is built
identically to every other Paper B Qwen3 eval. The f12k2 healed probe loads with
--keep_front_layers 12 --n_fresh_layers 2 (keep/fresh are also read from the ckpt
meta; a CLI mismatch is a hard error inside the loader).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import string
import sys
import time
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# NO arch drift: reuse the Qwen3 PPL eval's model construction verbatim.
from eval_qwen3_probe2_ppl import (  # noqa: E402
    _log,
    load_base_model,
    load_pruned_model,
)

CLOSEDBOOK_TASKS = ["popqa", "triviaqa"]


# ---------------------------------------------------------------------------
# SQuAD / TriviaQA answer normalisation (standard: lower, strip punct+articles,
# collapse whitespace). Used for every metric so em/f1/contains are consistent.
# ---------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    return s


def _f1(pred: str, gold: str) -> float:
    p_toks = normalize_answer(pred).split()
    g_toks = normalize_answer(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)  # both empty -> 1.0, one empty -> 0.0
    common = Counter(p_toks) & Counter(g_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def score_prediction(pred: str, golds: list[str]) -> dict:
    """Max over gold aliases: em (exact), contains (gold substring of pred), f1."""
    np_pred = normalize_answer(pred)
    em = 0
    contains = 0
    f1 = 0.0
    for g in golds:
        ng = normalize_answer(g)
        if not ng:
            continue
        if np_pred == ng:
            em = 1
        if ng in np_pred:
            contains = 1
        f1 = max(f1, _f1(pred, g))
    return {"em": em, "contains": contains, "f1": f1}


# ---------------------------------------------------------------------------
# dataset -> list of {"question": str, "answers": [gold aliases ...]}
# ---------------------------------------------------------------------------
def load_task_examples(task: str):
    from datasets import load_dataset

    if task == "popqa":
        d = load_dataset("akariasai/PopQA", split="test")
        out = []
        for ex in d:
            raw = ex["possible_answers"]
            try:
                answers = json.loads(raw) if isinstance(raw, str) else list(raw)
            except Exception:
                answers = [raw]
            answers = [a for a in answers if isinstance(a, str) and a.strip()]
            if not answers:
                continue
            out.append({"question": ex["question"].strip(), "answers": answers})
        return out

    if task == "triviaqa":
        d = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        out = []
        for ex in d:
            a = ex["answer"]
            aliases = list(a.get("aliases") or [])
            aliases += list(a.get("normalized_aliases") or [])
            if a.get("value"):
                aliases.append(a["value"])
            aliases = [x for x in aliases if isinstance(x, str) and x.strip()]
            # dedup while preserving order
            seen = set()
            uniq = []
            for x in aliases:
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            if not uniq:
                continue
            out.append({"question": ex["question"].strip(), "answers": uniq})
        return out

    if task == "nq_open":
        # Natural Questions open (closed-book). Short free-form answers; `answer`
        # is an alias list (multiple gold), scored max-over-aliases like TriviaQA.
        # Try the canonical name first, then known aliases; pick whichever loads
        # a validation split carrying question + answer(list).
        from datasets import load_dataset

        last_err = None
        d = None
        for name in ("google-research-datasets/nq_open", "nq_open",
                     "natural_questions_open"):
            try:
                d = load_dataset(name, split="validation")
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                d = None
        if d is None:
            raise RuntimeError(f"could not load nq_open (tried gr-datasets/nq_open,"
                               f" nq_open, natural_questions_open): {last_err}")
        out = []
        for ex in d:
            raw = ex.get("answer")
            if isinstance(raw, str):
                answers = [raw]
            elif raw is None:
                answers = []
            else:
                answers = list(raw)
            answers = [a for a in answers if isinstance(a, str) and a.strip()]
            if not answers:
                continue
            out.append({"question": ex["question"].strip(), "answers": answers})
        return out

    raise ValueError(f"unknown closed-book task {task}")


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_answers(model, tok, examples, device, batch_size, max_new_tokens,
                     add_bos, bos_id, pad_id, max_ctx_len):
    """Greedy free-form generation. Left-padded batches; returns list[str] preds
    (first line of the decoded completion) aligned with `examples`."""
    tok.padding_side = "left"
    preds = ["" for _ in examples]
    # sort by prompt length for tighter batches, remember original index
    order = sorted(range(len(examples)),
                   key=lambda i: len(examples[i]["question"]))
    for b in range(0, len(order), batch_size):
        bidx = order[b:b + batch_size]
        prompts = [build_prompt(examples[i]["question"]) for i in bidx]
        enc_ids = [tok.encode(p, add_special_tokens=False) for p in prompts]
        if add_bos and bos_id is not None:
            enc_ids = [[bos_id] + ids for ids in enc_ids]
        enc_ids = [ids[-max_ctx_len:] for ids in enc_ids]  # left-truncate long
        maxl = max(len(ids) for ids in enc_ids)
        B = len(bidx)
        input_ids = torch.full((B, maxl), pad_id, dtype=torch.long)
        attn = torch.zeros((B, maxl), dtype=torch.long)
        for r, ids in enumerate(enc_ids):
            input_ids[r, maxl - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attn[r, maxl - len(ids):] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_id,
            )
        new_tokens = gen[:, maxl:]
        for r, i in enumerate(bidx):
            txt = tok.decode(new_tokens[r], skip_special_tokens=True)
            preds[i] = txt.strip().split("\n")[0].strip()
    return preds


# ---------------------------------------------------------------------------
def merge(results_dir):
    shard_files = sorted(glob.glob(os.path.join(results_dir, "shard*of*.json")))
    if not shard_files:
        raise FileNotFoundError(f"no shard*of*.json in {results_dir}")
    agg = {}  # task -> sums
    meta = None
    add_bos = None
    for sf in shard_files:
        with open(sf) as f:
            d = json.load(f)
        meta = d.get("meta", meta)
        add_bos = d.get("add_bos", add_bos)
        for task, t in d["tasks"].items():
            a = agg.setdefault(task, {"n": 0, "em": 0, "contains": 0, "f1": 0.0,
                                      "majority_em": 0})
            if t.get("skipped"):
                a["skipped"] = True
                a["error"] = t.get("error", "")
                continue
            a["n"] += t["n"]
            a["em"] += t["em_hits"]
            a["contains"] += t["contains_hits"]
            a["f1"] += t["f1_sum"]
            a["majority_em"] += t.get("majority_em_hits", 0)
    tasks = {}
    for task, a in agg.items():
        if a.get("skipped") and a["n"] == 0:
            tasks[task] = {"skipped": True, "error": a.get("error", ""), "n": 0}
            continue
        n = max(a["n"], 1)
        tasks[task] = {
            "n": a["n"],
            "em": a["em"] / n,
            "contains": a["contains"] / n,
            "f1": a["f1"] / n,
            "em_hits": a["em"],
            "contains_hits": a["contains"],
            "f1_sum": a["f1"],
            "majority_em": a["majority_em"] / n,
            "empty_em": 0.0,
        }
    summary = {
        "output_name": os.path.basename(results_dir.rstrip("/")),
        "n_shards": len(shard_files),
        "add_bos": add_bos,
        "meta": meta,
        "tasks": tasks,
    }
    out = os.path.join(results_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"[merge] {os.path.basename(results_dir.rstrip('/'))}: " + " | ".join(
        (f"{t}: SKIPPED ({v.get('error','')[:40]})" if v.get("skipped")
         else f"{t}: em={v['em']:.4f} contains={v['contains']:.4f} "
              f"f1={v['f1']:.4f} (n={v['n']}, maj_em={v['majority_em']:.4f})")
        for t, v in tasks.items()))
    _merge_per_example(results_dir)
    return summary


def _merge_per_example(results_dir):
    pe_files = sorted(glob.glob(
        os.path.join(results_dir, "per_example_*_shard*of*.jsonl")))
    if not pe_files:
        return
    by_task = {}
    pat = re.compile(r"per_example_(.+)_shard\d+of\d+\.jsonl$")
    for pf in pe_files:
        m = pat.search(os.path.basename(pf))
        if not m:
            continue
        recs = by_task.setdefault(m.group(1), [])
        with open(pf) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    for task, recs in by_task.items():
        recs.sort(key=lambda r: r.get("item_id", 0))
        outp = os.path.join(results_dir, f"per_example_{task}.jsonl")
        with open(outp, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        _log(f"[merge] per-example {task}: {len(recs)} rows -> "
             f"{os.path.basename(outp)}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=False,
                   help="pretrained Qwen3-8B path (cfg source for pruned mode; the "
                        "full model itself in base mode)")
    p.add_argument("--ckpt", type=str, default="",
                   help="prune-then-heal .pt (omit -> full-base mode)")
    p.add_argument("--keep_front_layers", type=int, default=None,
                   help="pruned mode; default read from ckpt meta")
    p.add_argument("--n_fresh_layers", type=int, default=None,
                   help="pruned mode; default read from ckpt meta (else 2)")
    p.add_argument("--tasks", type=str, default=",".join(CLOSEDBOOK_TASKS))
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--max_ctx_len", type=int, default=512,
                   help="left-truncate the prompt to this many tokens (QA prompts "
                        "are short; guards against a pathological long question)")
    p.add_argument("--add_bos", type=int, default=0,
                   help="0 (default, base protocol) or 1 to prepend BOS. Qwen3 "
                        "tokenizer has bos_token=None so 1 is a no-op (no BOS to "
                        "add) -> equivalent to OLMo's no-BOS add_bos=0.")
    p.add_argument("--limit", type=int, default=0,
                   help=">0 caps examples per task (post-strided); sanity only")
    p.add_argument("--output_name", type=str, required=False)
    p.add_argument("--results_root", type=str, default="qwen3_closedbook_results")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--prepare_data", action="store_true",
                   help="load all --tasks datasets (populate cache) then exit; "
                        "run ONCE before fanning out shards to avoid a download race")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.prepare_data:
        for t in tasks:
            try:
                ex = load_task_examples(t)
                _log(f"[prepare] {t}: {len(ex)} examples cached")
            except Exception as e:
                _log(f"[prepare] {t}: SKIPPED (load failed: {e})")
        _log("[prepare] all datasets attempted")
        return

    if args.merge:
        if not args.output_name:
            raise ValueError("--merge requires --output_name")
        merge(os.path.join(args.results_root, args.output_name))
        return

    if not args.output_name:
        raise ValueError("--output_name required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    bos_id = tok.bos_token_id
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    if args.ckpt:
        model, meta = load_pruned_model(
            args.ckpt, args.base_model, args.keep_front_layers,
            args.n_fresh_layers, device)
    else:
        if not args.base_model:
            raise ValueError("base mode requires --base_model")
        model, meta = load_base_model(args.base_model, device)
    meta["base_model"] = args.base_model
    meta["add_bos"] = bool(args.add_bos)
    meta["max_new_tokens"] = args.max_new_tokens

    results_dir = os.path.join(args.results_root, args.output_name)
    os.makedirs(results_dir, exist_ok=True)

    task_results = {}
    for task in tasks:
        try:
            examples_all = load_task_examples(task)
        except Exception as e:
            _log(f"[shard {args.shard_index}/{args.num_shards}] {task}: SKIPPED "
                 f"(load failed: {e})")
            task_results[task] = {"skipped": True, "error": str(e)[:400], "n": 0}
            continue

        # majority-answer baseline (computed over the FULL set for stability, then
        # scored only on this shard's items): the single most-frequent normalised
        # first-alias gold string.
        gold_counter = Counter(
            normalize_answer(ex["answers"][0]) for ex in examples_all
            if ex["answers"])
        majority_norm = gold_counter.most_common(1)[0][0] if gold_counter else ""

        shard = examples_all[args.shard_index::args.num_shards]
        if args.limit and args.limit > 0:
            shard = shard[: args.limit]

        t0 = time.time()
        preds = generate_answers(
            model, tok, shard, device, args.batch_size, args.max_new_tokens,
            bool(args.add_bos), bos_id, pad_id, args.max_ctx_len)
        dt = time.time() - t0

        em_hits = contains_hits = 0
        f1_sum = 0.0
        majority_em_hits = 0
        pe_out = os.path.join(
            results_dir,
            f"per_example_{task}_shard{args.shard_index}of{args.num_shards}.jsonl")
        with open(pe_out, "w") as pef:
            for li, (ex, pred) in enumerate(zip(shard, preds)):
                sc = score_prediction(pred, ex["answers"])
                em_hits += sc["em"]
                contains_hits += sc["contains"]
                f1_sum += sc["f1"]
                # majority baseline scored per item (contains-style against golds)
                maj_hit = 0
                if majority_norm:
                    for g in ex["answers"]:
                        if normalize_answer(g) == majority_norm:
                            maj_hit = 1
                            break
                majority_em_hits += maj_hit
                item_id = args.shard_index + li * args.num_shards
                pef.write(json.dumps({
                    "item_id": item_id,
                    "question": ex["question"],
                    "gold": ex["answers"],
                    "pred": pred,
                    "em": sc["em"],
                    "contains": sc["contains"],
                    "f1": round(sc["f1"], 4),
                }) + "\n")
        n = len(shard)
        task_results[task] = {
            "n": n,
            "em_hits": em_hits,
            "contains_hits": contains_hits,
            "f1_sum": f1_sum,
            "majority_em_hits": majority_em_hits,
            "majority_answer": majority_norm,
            "em_shard": em_hits / max(n, 1),
            "contains_shard": contains_hits / max(n, 1),
            "f1_shard": f1_sum / max(n, 1),
            "seconds": round(dt, 1),
        }
        _log(f"[shard {args.shard_index}/{args.num_shards}] {task}: n={n} "
             f"em={em_hits/max(n,1):.4f} contains={contains_hits/max(n,1):.4f} "
             f"f1={f1_sum/max(n,1):.4f} maj_em={majority_em_hits/max(n,1):.4f} "
             f"({dt:.1f}s) -> {pe_out}")

    out = os.path.join(results_dir, f"shard{args.shard_index}of{args.num_shards}.json")
    with open(out, "w") as f:
        json.dump({
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "add_bos": bool(args.add_bos),
            "meta": meta,
            "tasks": task_results,
        }, f, indent=2)
    _log(f"[shard {args.shard_index}/{args.num_shards}] wrote {out}")


if __name__ == "__main__":
    main()
