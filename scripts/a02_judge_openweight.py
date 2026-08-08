#!/usr/bin/env python3
"""A02 open-weight transformers-based LoCoMo judge (Qwen3-8B, non-thinking, deterministic).

Replaces the vLLM-served judge for the A02 phase-1 C1 vs C2 gate.
vLLM 0.26.0 on .82 cannot load Qwen3ForCausalLM (too-old version), so this
script uses transformers directly to run the same judge protocol as in commits
7aa4e14 + 15f7325.

Protocol preserved verbatim:
  - model: Qwen3-8B
  - non-thinking: enable_thinking=False (via apply_chat_template kwarg)
  - deterministic: temperature=0, top_p=1, max_new_tokens=8
  - prompt template: same _JUDGE_TEMPLATE as eval_qcmem_locomo.py::_judge_one
  - refusal scoring: same _REFUSAL_RE as eval_qcmem_locomo.py
  - judge_meta.json written next to judge_cache.jsonl

Usage (run directly on .82 zwfy6 node):
  python scripts/a02_judge_openweight.py \
    --result_dirs locomo_results/a02_locomo_c1_kvdirect locomo_results/a02_locomo_c2_j12_readlora \
    --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
    --output_json proposal/active/A02-comem-write-read-repair/evidence/locomo_judge_openweight.json \
    --n_bootstrap 2000 --bootstrap_seed 42 \
    [--gpus 0,1,2,3] [--batch_size 16] [--force_rejudge]
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

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# Judge prompt template — verbatim from eval_qcmem_locomo.py::_JUDGE_TEMPLATE
# --------------------------------------------------------------------------- #
_JUDGE_TEMPLATE = (
    "You are grading a model's answer against the gold answer for a question "
    "about a long, multi-session dialogue (the LoCoMo benchmark).\n\n"
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Model answer: {pred}\n\n"
    "Grade whether the model answer is CORRECT. It is CORRECT if it conveys the "
    "same key information as the gold answer (a semantic match), even if phrased "
    "differently, more verbosely, or with extra correct context. It is WRONG if "
    "it contradicts the gold answer, omits the key information, or is empty / "
    "refuses when an answer exists. For date/time answers, accept any unambiguous "
    "equivalent phrasing.\n\n"
    "Respond with ONLY one word: CORRECT or WRONG."
)

# Refusal regex — verbatim from eval_qcmem_locomo.py::_REFUSAL_RE
_REFUSAL_RE = re.compile(
    r"\b(i don'?t know|not (mentioned|sure|provided|available|specified)|"
    r"no (information|mention|record)|cannot (find|determine|answer)|"
    r"unanswerable|isn'?t (mentioned|provided)|wasn'?t mentioned)\b",
    re.IGNORECASE,
)

# Token ids for "CORRECT" and "WRONG" (first subword).
# Resolved lazily once we have the tokenizer.
_CORRECT_IDS: list[int] = []
_WRONG_IDS: list[int] = []


def _resolve_verdict_tokens(tokenizer):
    global _CORRECT_IDS, _WRONG_IDS
    # Use the first token of each word (no-BOS encoding).
    _CORRECT_IDS = tokenizer.encode("CORRECT", add_special_tokens=False)[:1]
    _WRONG_IDS = tokenizer.encode("WRONG", add_special_tokens=False)[:1]


# --------------------------------------------------------------------------- #
# Build judge prompt using the model's chat template (non-thinking).
# --------------------------------------------------------------------------- #
def _build_prompt(tokenizer, question: str, golds: list, pred: str) -> str:
    gold = " OR ".join(str(g) for g in golds if str(g).strip()) or "(none)"
    # Add /no_think suffix (belt-and-suspenders for Qwen3 template variants).
    content = _JUDGE_TEMPLATE.format(
        question=question, gold=gold, pred=pred or ""
    ) + "\n/no_think"
    try:
        # enable_thinking=False is the canonical Qwen3 vLLM switch; transformers
        # apply_chat_template supports it in recent transformers versions.
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older transformers: fall back without the kwarg (still no <think> because
        # the /no_think suffix triggers the same behaviour at the model level).
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


# --------------------------------------------------------------------------- #
# Batch judge: load model once, judge all items in batches.
# --------------------------------------------------------------------------- #
def judge_items_batch(items: list[dict], tokenizer, model, device_list: list,
                      batch_size: int = 8) -> list[tuple[float, str]]:
    """Return (verdict_float, raw_text) for each item.
    verdict: 1.0 = CORRECT, 0.0 = WRONG.
    Uses simple DataParallel-style batching on a single GPU (fastest for this
    short-output judge task).
    """
    device = device_list[0]
    results: list[tuple[float, str]] = [None] * len(items)

    prompts = []
    for it in items:
        p = _build_prompt(
            tokenizer,
            question=it.get("question", ""),
            golds=it.get("answers", []),
            pred=it.get("pred", ""),
        )
        prompts.append(p)

    # Batch encode left-padded so the generation cursor is at the same position.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for batch_start in tqdm(range(0, len(prompts), batch_size),
                            desc="judging", leave=False):
        batch_prompts = prompts[batch_start: batch_start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,  # judge prompts are short
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,          # greedy = deterministic
                temperature=1.0,          # ignored when do_sample=False
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (after the prompt).
        prompt_len = input_ids.shape[1]
        for bi, gen_ids in enumerate(out):
            new_ids = gen_ids[prompt_len:].tolist()
            txt = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            up = txt.upper()
            if up.startswith("CORRECT"):
                v = 1.0
            elif up.startswith("WRONG"):
                v = 0.0
            elif "CORRECT" in up and "WRONG" not in up:
                v = 1.0
            elif "WRONG" in up and "CORRECT" not in up:
                v = 0.0
            else:
                v = 0.0  # unparseable -> conservative WRONG
            results[batch_start + bi] = (v, txt[:80])

    return results


# --------------------------------------------------------------------------- #
# Load / merge prediction shards from a result directory.
# --------------------------------------------------------------------------- #
def load_preds(result_dir: str) -> list[dict]:
    p = Path(result_dir)
    shards = sorted(p.glob("preds*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No preds*.jsonl in {result_dir}")
    preds = {}
    for sf in shards:
        with open(sf) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                preds[item["id"]] = item
    return list(preds.values())


# --------------------------------------------------------------------------- #
# Bootstrap CI (paired difference, C2 - C1).
# --------------------------------------------------------------------------- #
def bootstrap_paired_ci(scores_c1: list[float], scores_c2: list[float],
                         n: int = 2000, seed: int = 42,
                         alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    arr_c1 = np.array(scores_c1)
    arr_c2 = np.array(scores_c2)
    diffs = arr_c2 - arr_c1
    pt_est = float(diffs.mean())
    boot = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n)
    ])
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return {
        "n_pairs": len(diffs),
        "c1_mean": float(arr_c1.mean()),
        "c2_mean": float(arr_c2.mean()),
        "diff_pt_pp": round(pt_est * 100, 4),
        "ci_lo_pp": round(lo * 100, 4),
        "ci_hi_pp": round(hi * 100, 4),
        "n_bootstrap": n,
        "seed": seed,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="A02 open-weight LoCoMo judge (Qwen3-8B transformers fallback)"
    )
    parser.add_argument("--result_dirs", nargs="+", required=True,
                        help="One or more locomo result dirs to judge. First = C1, second = C2.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Qwen3-8B model weights.")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output path for locomo_judge_openweight.json.")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU indices to use (e.g. '0,1,2,3'). "
                             "Only the first GPU is used for inference; others listed "
                             "for info only.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for judge inference.")
    parser.add_argument("--force_rejudge", action="store_true", default=False,
                        help="Ignore existing open-weight judge_cache_openweight.jsonl "
                             "and re-judge everything.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    device = torch.device(f"cuda:{gpu_ids[0]}")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[A02-judge] Loading Qwen3-8B from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    _resolve_verdict_tokens(tokenizer)
    print(f"[A02-judge] Model loaded on {device}.")

    # Write judge_meta.json next to each result dir and save method info.
    judge_meta = {
        "judge_model": "Qwen3-8B",
        "judge_model_path": args.model_path,
        "method": "transformers_generate",
        "non_thinking": True,
        "sampling": {
            "do_sample": False,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 8,
        },
        "chat_template_kwargs": {"enable_thinking": False},
        "prompt_suffix": "/no_think",
        "prompt_template": _JUDGE_TEMPLATE,
        "refusal_regex": _REFUSAL_RE.pattern,
        "note": ("transformers fallback: vLLM 0.26.0 on .82 cannot load "
                 "Qwen3ForCausalLM (too old). Protocol semantics identical: "
                 "same prompt, same non-thinking mode, same greedy sampling."),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Collect per-dir results.
    dir_results = {}  # result_dir -> {"preds": [...], "scores": [...]}

    for result_dir in args.result_dirs:
        result_dir = str(result_dir)
        print(f"\n[A02-judge] Processing {result_dir} ...")
        preds = load_preds(result_dir)
        print(f"[A02-judge]   loaded {len(preds)} predictions.")

        # Write judge_meta.json (overwrite; this is intentionally the open-weight version).
        cache_dir = Path(result_dir)
        with open(cache_dir / "judge_meta.json", "w") as fh:
            json.dump(judge_meta, fh, indent=2, ensure_ascii=False)

        # Load existing open-weight cache (separate file from GPT-4o cache).
        ow_cache_path = cache_dir / "judge_cache_openweight.jsonl"
        ow_cache = {}
        if not args.force_rejudge and ow_cache_path.exists():
            with open(ow_cache_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        ow_cache[rec["id"]] = rec
                    except Exception:
                        pass
            print(f"[A02-judge]   loaded {len(ow_cache)} cached open-weight verdicts.")

        # Separate abstention (local grading) vs non-abstention (need judge).
        to_judge = []
        for item in preds:
            if item.get("is_abstention", False):
                refused = (bool(_REFUSAL_RE.search(item.get("pred", "")))
                           or item.get("pred", "").strip() == "")
                item["judge_ow"] = 1.0 if refused else 0.0
            elif item["id"] in ow_cache:
                item["judge_ow"] = float(ow_cache[item["id"]]["judge"])
            else:
                to_judge.append(item)

        if to_judge:
            print(f"[A02-judge]   {len(to_judge)} items need open-weight judging ...")
            results_list = judge_items_batch(
                to_judge, tokenizer, model, [device], batch_size=args.batch_size)
            cache_fh = open(ow_cache_path, "a")
            for item, (v, raw) in zip(to_judge, results_list):
                item["judge_ow"] = v
                rec = {
                    "id": item["id"],
                    "judge": v,
                    "category": item.get("category"),
                    "question": item.get("question", ""),
                    "gold": item.get("answers", []),
                    "pred": (item.get("pred", "") or "")[:200],
                    "raw": raw,
                    "model": "Qwen3-8B",
                }
                cache_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache_fh.flush()
            cache_fh.close()
            print(f"[A02-judge]   wrote {len(to_judge)} verdicts to {ow_cache_path}")
        else:
            print("[A02-judge]   all items cached, no inference needed.")

        # Compute per-item scores.
        scores = [item.get("judge_ow", 0.0) for item in preds]
        dir_results[result_dir] = {"preds": preds, "scores": scores}

        # Print summary.
        n = len(preds)
        mean_judge = sum(scores) / n * 100
        by_cat = collections.defaultdict(list)
        for item in preds:
            by_cat[item.get("category", "?")].append(item.get("judge_ow", 0.0))
        print(f"[A02-judge]   overall judge acc = {mean_judge:.2f}% (n={n})")
        for cat in sorted(by_cat):
            c_scores = by_cat[cat]
            print(f"[A02-judge]   cat{cat}: {sum(c_scores)/len(c_scores)*100:.2f}% (n={len(c_scores)})")

        # Update the existing scores.json with judge metrics.
        scores_path = cache_dir / "scores.json"
        if scores_path.exists():
            with open(scores_path) as fh:
                sc = json.load(fh)
        else:
            sc = {}
        sc["overall_judge_openweight"] = mean_judge
        sc["judge_model_openweight"] = "Qwen3-8B (transformers, non-thinking, greedy)"
        by_cat_out = sc.get("by_category", {})
        for cat, c_scores in by_cat.items():
            k = str(cat)
            if k not in by_cat_out:
                by_cat_out[k] = {}
            by_cat_out[k]["judge_ow"] = sum(c_scores) / len(c_scores) * 100
        sc["by_category"] = by_cat_out
        with open(scores_path, "w") as fh:
            json.dump(sc, fh, indent=2)
        print(f"[A02-judge]   updated {scores_path}")

    # ---------- paired CI (C2 - C1) ----------
    if len(args.result_dirs) >= 2:
        dir_c1 = args.result_dirs[0]
        dir_c2 = args.result_dirs[1]

        # Build aligned score lists (same sample ids, same order).
        preds_c1 = {item["id"]: item for item in dir_results[dir_c1]["preds"]}
        preds_c2 = {item["id"]: item for item in dir_results[dir_c2]["preds"]}
        common_ids = sorted(set(preds_c1) & set(preds_c2))
        print(f"\n[A02-judge] Paired CI: {len(common_ids)} common ids "
              f"(C1={len(preds_c1)} C2={len(preds_c2)})")

        scores_c1_paired = [preds_c1[i].get("judge_ow", 0.0) for i in common_ids]
        scores_c2_paired = [preds_c2[i].get("judge_ow", 0.0) for i in common_ids]

        ci = bootstrap_paired_ci(
            scores_c1_paired, scores_c2_paired,
            n=args.n_bootstrap, seed=args.bootstrap_seed)

        # Also compute F1 CI for reference.
        def _compute_f1(pred_str: str, answers: list) -> float:
            def _norm(s):
                s = s.lower()
                s = re.sub(r"\b(a|an|the)\b", " ", s)
                s = "".join(ch for ch in s if ch not in string.punctuation)
                return " ".join(s.split())
            pt = _norm(pred_str).split()
            gts = [_norm(a).split() for a in answers]
            best = 0.0
            for gt_t in gts:
                if not pt and not gt_t:
                    return 1.0
                if not pt or not gt_t:
                    continue
                common = collections.Counter(pt) & collections.Counter(gt_t)
                ns = sum(common.values())
                if ns == 0:
                    continue
                p = ns / len(pt)
                r = ns / len(gt_t)
                f = 2 * p * r / (p + r)
                best = max(best, f)
            return best

        f1_c1 = [_compute_f1(preds_c1[i].get("pred", ""),
                              preds_c1[i].get("answers", [])) for i in common_ids]
        f1_c2 = [_compute_f1(preds_c2[i].get("pred", ""),
                              preds_c2[i].get("answers", [])) for i in common_ids]
        ci_f1 = bootstrap_paired_ci(f1_c1, f1_c2,
                                     n=args.n_bootstrap, seed=args.bootstrap_seed)

        # Per-item score list for provenance.
        per_item = []
        for sid in common_ids:
            per_item.append({
                "id": sid,
                "category": preds_c1[sid].get("category"),
                "is_abstention": preds_c1[sid].get("is_abstention", False),
                "judge_c1": preds_c1[sid].get("judge_ow"),
                "judge_c2": preds_c2[sid].get("judge_ow"),
                "f1_c1": preds_c1[sid].get("_f1_tmp_"),  # computed below
                "f1_c2": preds_c2[sid].get("_f1_tmp_"),
            })
        for i, sid in enumerate(common_ids):
            per_item[i]["f1_c1"] = f1_c1[i]
            per_item[i]["f1_c2"] = f1_c2[i]

        output = {
            "description": "A02 phase-1 C1 vs C2 LoCoMo judge re-run with open-weight Qwen3-8B judge",
            "judge_method": "transformers_generate (vLLM fallback; same prompt/sampling protocol)",
            "judge_model": "Qwen3-8B",
            "judge_model_path": args.model_path,
            "non_thinking": True,
            "sampling": {"do_sample": False, "max_new_tokens": 8},
            "dirs": {"c1": dir_c1, "c2": dir_c2},
            "judge_ci": ci,
            "f1_ci": ci_f1,
            "per_item": per_item,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)
        print(f"\n[A02-judge] Wrote {out_path}")

        print(f"\n[A02-judge] === PAIRED CI SUMMARY ===")
        print(f"  Judge (C2-C1): point={ci['diff_pt_pp']:+.2f}pp  "
              f"95%CI=[{ci['ci_lo_pp']:+.2f}, {ci['ci_hi_pp']:+.2f}]  "
              f"n={ci['n_pairs']}")
        print(f"  F1   (C2-C1): point={ci_f1['diff_pt_pp']:+.2f}pp  "
              f"95%CI=[{ci_f1['ci_lo_pp']:+.2f}, {ci_f1['ci_hi_pp']:+.2f}]  "
              f"n={ci_f1['n_pairs']}")
        print(f"  C1 judge acc: {ci['c1_mean']*100:.2f}%")
        print(f"  C2 judge acc: {ci['c2_mean']*100:.2f}%")
    else:
        print("[A02-judge] Only one result dir provided — no paired CI computed.")
    print("\n[A02-judge] Done.")


if __name__ == "__main__":
    main()
