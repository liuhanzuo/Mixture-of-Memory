#!/usr/bin/env python3
"""Likelihood-based multiple-choice DOWNSTREAM eval for the OLMo-2 prune-then-heal
probe (Paper B, direction #4). Companion to scripts/eval_olmo2_probe2_ppl.py.

Question this answers
---------------------
The held-out PPL ledger (status/OLMO2_PRUNEHEAL_PPL.md) showed the pruned+healed
OLMo-2 recovers language-modelling to ~1.47x (1B) / ~2.33x (7B early) the full
base. But PPL is not capability. Here we measure zero-shot **downstream MC
accuracy** (NO generation -- for each candidate answer we score the teacher-forced
sum log-prob of the continuation tokens and take argmax) to test whether the
dropped middle layers were redundant (capability recovers with PPL) or carried
progressive refinement (capability lags PPL).

Arch construction: NO drift. We import load_pruned_model / load_base_model from
eval_olmo2_probe2_ppl.py (which copied the trainer's build verbatim), so the shell
is rebuilt + strict-loaded identically to the PPL eval.

Scoring (lm-eval-harness convention, no drift from the PPL eval's dtype policy)
------------------------------------------------------------------------------
* fp32 weights, bf16-autocast forward (matches training + the PPL eval).
* For each (context, continuation) we encode context+continuation, split off the
  continuation tokens as whole_enc[len(context_enc):], run the model, and sum the
  fp32 log-softmax log-prob of each continuation token (teacher forced).
* target_delimiter = " " (a leading space is prepended to every candidate string
  before tokenising), matching lm-eval defaults.
* OLMo-2 tokenizer does NOT auto-add BOS (add_special_tokens is a no-op); we keep
  add_special_tokens=False, matching how published OLMo-2 lm-eval numbers are made.
* acc      = argmax over sum-logprob  hits gold.
* acc_norm = argmax over sum-logprob / len(candidate_string_in_chars) hits gold
  (char-length normalisation, the lm-eval `completion_len` convention using the raw
  choice text WITHOUT the leading delimiter space).

Tasks (zero-shot, standard splits): hellaswag, arc_challenge, arc_easy, piqa,
winogrande, openbookqa. winogrande uses the standard partial/double-cloze scoring
(the continuation is the shared suffix after the blank; the two option-filled
prefixes are the two contexts) -> acc_norm == acc there by construction.

Sharding: examples are strided windows[shard_index::num_shards] PER TASK (same
scheme as the PPL eval). Each shard writes shard{i}of{N}.json holding, per task,
n / n_correct_acc / n_correct_accnorm / n_nan (+ a few sample log-prob vectors for
the degenerate/NaN check). --merge sums the shard counts: acc = sum(correct)/sum(n).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# NO arch drift: reuse the PPL eval's model construction verbatim.
from eval_ppl import (  # noqa: E402
    _log,
    load_base_model,
    load_pruned_model,
)

ALL_TASKS = ["hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande", "openbookqa"]
# knowledge / comprehension extension (Paper B direction #4, 2026-07-19):
#   mmlu           57-subject 4-choice knowledge (letter continuations; acc==acc_norm),
#                  aggregate acc + per-subject breakdown.
#   lambada_openai last-word prediction; metric = is_greedy (exact match of the whole
#                  last-word continuation under greedy argmax), NOT MC argmax -> acc only.
#   boolq          yes/no reading comprehension (2-choice likelihood) -> acc only.
#   commonsense_qa 5-choice; social_iqa 3-choice (choice-text continuations).
KNOWLEDGE_TASKS = ["mmlu", "lambada_openai", "boolq", "commonsense_qa", "social_iqa"]
# Immutable HF revisions used for the released evaluation protocol.
DATASET_REVISIONS = {
    "hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "arc": "210d026faf9955653af8916fad021475a3f00453",
    "openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
    "piqa": "142c51238b3ca2bc61e9a075913871b8b600e8e1",
    "winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    "mmlu": "c30699e8356da336a370243923dbaf21066bb9fe",
    "lambada_openai": "900124bf3b8235c6daf21033af9948b3f07346c4",
    "boolq": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
    "super_glue": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
    "commonsense_qa": "94630fe30dad47192a8546eb75f094926d47e155",
    "social_iqa": "537a2ec8ec565adc0b70b70752893e59e024df26",
}
# tasks scored by greedy last-word match instead of MC argmax over candidates.
GREEDY_TASKS = {"lambada_openai"}


# ---------------------------------------------------------------------------
# dataset -> list of examples, each: {"gold": int, "cands": [(ctx, cont, norm_chars), ...]}
# (ctx, cont) are raw strings; cont already includes the leading " " delimiter.
# norm_chars = len of the raw candidate text WITHOUT the leading space (lm-eval).
# ---------------------------------------------------------------------------
def _hs_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def load_task_examples(task: str):
    from datasets import load_dataset

    if task == "hellaswag":
        d = load_dataset(
            "Rowan/hellaswag", split="validation",
            revision=DATASET_REVISIONS["hellaswag"],
        )
        out = []
        for ex in d:
            ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
            query = _hs_preprocess(ex["activity_label"] + ": " + ctx)
            choices = [_hs_preprocess(e) for e in ex["endings"]]
            out.append({
                "gold": int(ex["label"]),
                "cands": [(query, " " + c, len(c)) for c in choices],
            })
        return out

    if task in ("arc_challenge", "arc_easy"):
        cfg = "ARC-Challenge" if task == "arc_challenge" else "ARC-Easy"
        d = load_dataset(
            "allenai/ai2_arc", cfg, split="test",
            revision=DATASET_REVISIONS["arc"],
        )
        out = []
        for ex in d:
            q = "Question: " + ex["question"] + "\nAnswer:"
            texts = ex["choices"]["text"]
            labels = ex["choices"]["label"]
            ans = ex["answerKey"]
            if ans not in labels:
                continue  # malformed; skip
            out.append({
                "gold": labels.index(ans),
                "cands": [(q, " " + t, len(t)) for t in texts],
            })
        return out

    if task == "openbookqa":
        d = load_dataset("allenai/openbookqa", "main", split="test", revision=DATASET_REVISIONS["openbookqa"])
        out = []
        for ex in d:
            q = ex["question_stem"]
            texts = ex["choices"]["text"]
            labels = ex["choices"]["label"]
            ans = ex["answerKey"]
            if ans not in labels:
                continue
            out.append({
                "gold": labels.index(ans),
                "cands": [(q, " " + t, len(t)) for t in texts],
            })
        return out

    if task == "piqa":
        # ybisk/piqa still ships a loading script (unsupported in datasets>=3);
        # the HF auto-converted parquet branch is the stable source.
        d = load_dataset(
            "ybisk/piqa", revision=DATASET_REVISIONS["piqa"],
            split="validation",
        )
        out = []
        for ex in d:
            q = "Question: " + ex["goal"] + "\nAnswer:"
            sols = [ex["sol1"], ex["sol2"]]
            out.append({
                "gold": int(ex["label"]),
                "cands": [(q, " " + s, len(s)) for s in sols],
            })
        return out

    if task == "winogrande":
        d = load_dataset(
            "allenai/winogrande", "winogrande_xl", split="validation",
            revision=DATASET_REVISIONS["winogrande"],
        )
        answer_to_idx = {"1": 0, "2": 1}
        out = []
        for ex in d:
            s = ex["sentence"]
            idx = s.index("_")
            target = s[idx + 1:].strip()
            prefix = s[:idx]
            opts = [ex["option1"], ex["option2"]]
            # partial scoring: shared suffix `target` is the continuation; the two
            # option-filled prefixes are the two contexts. norm len identical ->
            # acc_norm == acc.
            out.append({
                "gold": answer_to_idx[ex["answer"]],
                "cands": [(prefix + o, " " + target, len(target)) for o in opts],
            })
        return out

    if task == "mmlu":
        # cais/mmlu "all" = the 57-subject union. lm-eval flan-style zero-shot:
        # per-subject description + question + A./B./C./D. lettered choices +
        # "Answer:", candidates are the single letters -> acc == acc_norm.
        d = load_dataset("cais/mmlu", "all", split="test", revision=DATASET_REVISIONS["mmlu"])
        letters = ["A", "B", "C", "D"]
        out = []
        for ex in d:
            subject_h = ex["subject"].replace("_", " ")
            desc = ("The following are multiple choice questions (with answers) "
                    f"about {subject_h}.\n\n")
            ch = ex["choices"]
            body = "\n".join(f"{letters[i]}. {ch[i]}" for i in range(len(ch)))
            q = desc + ex["question"].strip() + "\n" + body + "\nAnswer:"
            out.append({
                "gold": int(ex["answer"]),
                "subject": ex["subject"],
                "cands": [(q, " " + letters[i], len(letters[i]))
                          for i in range(len(ch))],
            })
        return out

    if task == "lambada_openai":
        # last-word prediction; single candidate scored by greedy exact match.
        d = load_dataset(
            "EleutherAI/lambada_openai", split="test",
            revision=DATASET_REVISIONS["lambada_openai"],
        )
        out = []
        for ex in d:
            text = ex["text"].strip()
            ctx, last = text.rsplit(" ", 1)
            out.append({
                "gold": 0,
                "cands": [(ctx, " " + last, len(last))],
            })
        return out

    if task == "boolq":
        # google/boolq (fallback super_glue/boolq). lm-eval: passage + Question:?
        # + Answer:, doc_to_choice = ["no","yes"], gold = int(answer). acc only.
        try:
            d = load_dataset(
                "google/boolq", split="validation",
                revision=DATASET_REVISIONS["boolq"],
            )
        except Exception:
            d = load_dataset(
                "aps/super_glue", "boolq", split="validation",
                revision=DATASET_REVISIONS["super_glue"],
            )
        out = []
        for ex in d:
            q = ex["passage"] + "\nQuestion: " + ex["question"] + "?\nAnswer:"
            out.append({
                "gold": int(ex["answer"]),
                "cands": [(q, " no", len("no")), (q, " yes", len("yes"))],
            })
        return out

    if task == "commonsense_qa":
        d = load_dataset(
            "tau/commonsense_qa", split="validation",
            revision=DATASET_REVISIONS["commonsense_qa"],
        )
        out = []
        for ex in d:
            q = "Question: " + ex["question"] + "\nAnswer:"
            texts = ex["choices"]["text"]
            labels = ex["choices"]["label"]
            ans = ex["answerKey"]
            if ans not in labels:
                continue  # test split has no gold; skip if answerKey blank
            out.append({
                "gold": labels.index(ans),
                "cands": [(q, " " + t, len(t)) for t in texts],
            })
        return out

    if task == "social_iqa":
        # allenai/social_i_qa still ships a loading script (unsupported in
        # datasets>=3); use the HF auto-converted parquet branch (like piqa).
        d = load_dataset(
            "allenai/social_i_qa", revision=DATASET_REVISIONS["social_iqa"],
            split="validation",
        )
        out = []
        for ex in d:
            q = ex["context"] + "\nQuestion: " + ex["question"] + "\nAnswer:"
            answers = [ex["answerA"], ex["answerB"], ex["answerC"]]
            gold = int(str(ex["label"]).strip()) - 1  # label is "1"/"2"/"3"
            out.append({
                "gold": gold,
                "cands": [(q, " " + a, len(a)) for a in answers],
            })
        return out

    raise ValueError(f"unknown task {task}")


# ---------------------------------------------------------------------------
# tokenisation of a (context, continuation) pair -> (input_ids, cont_start, cont_len)
# lm-eval _encode_pair space-fixup + whole/context split.
# ---------------------------------------------------------------------------
def encode_pair(tok, context, continuation, add_bos, bos_id):
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tok.encode(context + continuation, add_special_tokens=False)
    ctx = tok.encode(context, add_special_tokens=False)
    cont_start = len(ctx)
    cont_len = len(whole) - cont_start
    if add_bos and bos_id is not None:
        whole = [bos_id] + whole
        cont_start += 1
    if cont_len <= 0:
        # degenerate (empty continuation after tokenisation); score last token.
        cont_start = max(cont_start - 1, 1)
        cont_len = 1
    return whole, cont_start, cont_len


@torch.no_grad()
def score_task(model, tok, examples, device, batch_size, add_bos, bos_id,
               pad_id, max_len, mode="mc"):
    """Return (n, n_correct_acc, n_correct_accnorm, n_nan, sample_lls, n_trunc,
    subjects).
    mode="mc": argmax over candidate sum-logprob (acc) / length-normed (acc_norm).
    mode="greedy": single candidate scored by is_greedy (every continuation token
      is the argmax of the model's next-token distribution) -> acc==acc_norm; used
      for lambada last-word prediction.
    subjects: {subject -> {n, n_correct_acc}} accumulated when examples carry a
      "subject" key (mmlu per-subject breakdown); empty dict otherwise.
    sample_lls = list of {gold, lls, norm_lls, pred_acc, pred_accnorm} for the
    first few examples (degenerate/NaN inspection)."""
    # flatten candidates into items, remember (ex_idx, cand_idx)
    items = []  # (ex_idx, cand_idx, ids, cont_start, cont_len)
    n_trunc = 0
    for ei, ex in enumerate(examples):
        for ci, (ctx, cont, _norm) in enumerate(ex["cands"]):
            ids, cs, cl = encode_pair(tok, ctx, cont, add_bos, bos_id)
            if len(ids) > max_len:
                drop = len(ids) - max_len
                ids = ids[drop:]
                cs = max(cs - drop, 1)
                cl = min(cl, len(ids) - cs)
                if cl <= 0:
                    cs, cl = max(len(ids) - 1, 1), 1
                n_trunc += 1
            items.append((ei, ci, ids, cs, cl))

    lls = [[0.0] * len(ex["cands"]) for ex in examples]
    nan = [[False] * len(ex["cands"]) for ex in examples]
    greedy_ok = [[False] * len(ex["cands"]) for ex in examples]

    order = sorted(range(len(items)), key=lambda i: len(items[i][2]))
    for b in range(0, len(order), batch_size):
        bidx = order[b:b + batch_size]
        maxl = max(len(items[i][2]) for i in bidx)
        B = len(bidx)
        input_ids = torch.full((B, maxl), pad_id, dtype=torch.long)
        attn = torch.zeros((B, maxl), dtype=torch.long)
        for r, i in enumerate(bidx):
            ids = items[i][2]
            input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[r, :len(ids)] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attn)
        logprobs = torch.log_softmax(out.logits.float(), dim=-1)  # [B, L, V]
        for r, i in enumerate(bidx):
            ei, ci, ids, cs, cl = items[i]
            end = cs + cl
            # positions cs-1 .. end-2 predict tokens cs .. end-1
            pos = torch.arange(cs - 1, end - 1, device=device)
            tgt = torch.tensor(ids[cs:end], dtype=torch.long, device=device)
            lp = logprobs[r, pos, tgt].sum().item()
            lls[ei][ci] = lp
            if not math.isfinite(lp):
                nan[ei][ci] = True
            if mode == "greedy":
                pred = logprobs[r, pos].argmax(dim=-1)
                greedy_ok[ei][ci] = bool((pred == tgt).all().item())

    n = len(examples)
    n_correct_acc = 0
    n_correct_norm = 0
    n_nan = 0
    samples = []
    subjects = {}  # subject -> {n, n_correct_acc}
    for ei, ex in enumerate(examples):
        cand_lls = lls[ei]
        norm_lens = [c[2] for c in ex["cands"]]
        if any(nan[ei]):
            n_nan += 1
            continue
        if mode == "greedy":
            correct = greedy_ok[ei][0]
            pred_acc = 0 if correct else -1
            norm_lls = [cand_lls[0]]
            pred_norm = pred_acc
            if correct:
                n_correct_acc += 1
                n_correct_norm += 1
        else:
            pred_acc = max(range(len(cand_lls)), key=lambda k: cand_lls[k])
            norm_lls = [cand_lls[k] / max(norm_lens[k], 1) for k in range(len(cand_lls))]
            pred_norm = max(range(len(norm_lls)), key=lambda k: norm_lls[k])
            correct = (pred_acc == ex["gold"])
            if correct:
                n_correct_acc += 1
            if pred_norm == ex["gold"]:
                n_correct_norm += 1
        subj = ex.get("subject")
        if subj is not None:
            sb = subjects.setdefault(subj, {"n": 0, "n_correct_acc": 0})
            sb["n"] += 1
            if correct:
                sb["n_correct_acc"] += 1
        if len(samples) < 6:
            samples.append({
                "gold": ex["gold"],
                "lls": [round(x, 4) for x in cand_lls],
                "norm_lls": [round(x, 5) for x in norm_lls],
                "pred_acc": pred_acc,
                "pred_accnorm": pred_norm,
            })
    return n, n_correct_acc, n_correct_norm, n_nan, samples, n_trunc, subjects


# ---------------------------------------------------------------------------
def merge(results_dir):
    shard_files = sorted(glob.glob(os.path.join(results_dir, "shard*of*.json")))
    if not shard_files:
        raise FileNotFoundError(f"no shard*of*.json in {results_dir}")
    parsed = []
    for sf in shard_files:
        match = re.fullmatch(r"shard(\d+)of(\d+)\.json", os.path.basename(sf))
        if match:
            parsed.append((int(match.group(1)), int(match.group(2)), sf))
    totals = {total for _, total, _ in parsed}
    if len(parsed) != len(shard_files) or len(totals) != 1:
        raise ValueError(f"invalid or inconsistent shard files in {results_dir}")
    expected_total = totals.pop()
    indices = sorted(index for index, _, _ in parsed)
    if indices != list(range(expected_total)):
        raise ValueError(
            f"incomplete shards in {results_dir}: got {indices}, expected 0..{expected_total - 1}"
        )

    agg = {}  # task -> dict of sums
    meta = None
    add_bos = None
    expected_tasks = None
    for _, _, sf in parsed:
        with open(sf) as f:
            d = json.load(f)
        if meta is None:
            meta = d.get("meta")
            add_bos = d.get("add_bos")
            expected_tasks = set(d["tasks"])
        elif d.get("meta") != meta or d.get("add_bos") != add_bos:
            raise ValueError(f"metadata mismatch across shards in {results_dir}")
        elif set(d["tasks"]) != expected_tasks:
            raise ValueError(f"task-set mismatch across shards in {results_dir}")
        for task, t in d["tasks"].items():
            a = agg.setdefault(task, {"n": 0, "n_correct_acc": 0,
                                      "n_correct_accnorm": 0, "n_nan": 0,
                                      "n_trunc": 0, "n_skipped_shards": 0,
                                      "subjects": {}})
            if t.get("skipped"):
                a["n_skipped_shards"] += 1
                a.setdefault("skip_error", t.get("error", ""))
                continue
            a["n"] += t["n"]
            a["n_correct_acc"] += t["n_correct_acc"]
            a["n_correct_accnorm"] += t["n_correct_accnorm"]
            a["n_nan"] += t.get("n_nan", 0)
            a["n_trunc"] += t.get("n_trunc", 0)
            for subj, sv in t.get("subjects", {}).items():
                sb = a["subjects"].setdefault(subj, {"n": 0, "n_correct_acc": 0})
                sb["n"] += sv["n"]
                sb["n_correct_acc"] += sv["n_correct_acc"]
    tasks = {}
    for task, a in agg.items():
        if a["n_skipped_shards"]:
            raise RuntimeError(
                f"task {task} failed on {a['n_skipped_shards']}/{expected_total} shards: "
                f"{a.get('skip_error', '')}"
            )
        if a["n"] == 0:
            raise RuntimeError(f"task {task} produced zero examples")
        if a["n_nan"]:
            raise RuntimeError(
                f"task {task} produced {a['n_nan']} non-finite examples; refusing partial score"
            )
        denom = a["n"]
        entry = {
            "n": a["n"],
            "n_scored": a["n"] - a["n_nan"],
            "n_nan": a["n_nan"],
            "n_trunc": a["n_trunc"],
            "acc": a["n_correct_acc"] / denom,
            "acc_norm": a["n_correct_accnorm"] / denom,
            "n_correct_acc": a["n_correct_acc"],
            "n_correct_accnorm": a["n_correct_accnorm"],
        }
        if a["subjects"]:
            subj_out = {}
            for subj, sv in sorted(a["subjects"].items()):
                sd = max(sv["n"], 1)
                subj_out[subj] = {
                    "n": sv["n"],
                    "n_correct_acc": sv["n_correct_acc"],
                    "acc": sv["n_correct_acc"] / sd,
                }
            entry["subjects"] = subj_out
        tasks[task] = entry
    summary = {
        "output_name": os.path.basename(results_dir.rstrip("/")),
        "n_shards": expected_total,
        "add_bos": add_bos,
        "meta": meta,
        "tasks": tasks,
    }
    out = os.path.join(results_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"[merge] {os.path.basename(results_dir.rstrip('/'))}: " +
         " | ".join(
             (f"{t}: SKIPPED ({v.get('error', '')[:40]})" if v.get("skipped")
              else f"{t}: acc={v['acc']:.4f} accn={v['acc_norm']:.4f} "
                   f"(n={v['n']},nan={v['n_nan']})")
             for t, v in tasks.items()))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=False)
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--keep_front_layers", type=int, default=None)
    p.add_argument("--n_fresh_layers", type=int, default=None)
    p.add_argument("--tasks", type=str, default=",".join(ALL_TASKS))
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--add_bos", type=int, default=0,
                   help="0 (default, matches OLMo-2 lm-eval) or 1 to prepend BOS")
    p.add_argument("--limit", type=int, default=0,
                   help=">0 caps examples per task (post-strided); sanity only")
    p.add_argument("--output_name", type=str, required=False)
    p.add_argument("--results_root", type=str, default="olmo2_downstream_results")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--prepare_data", action="store_true",
                   help="load all --tasks datasets (populate cache) then exit; "
                        "run ONCE before fanning out 8 shards to avoid a download race")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.prepare_data:
        failures = []
        for t in tasks:
            try:
                ex = load_task_examples(t)
                if not ex:
                    raise RuntimeError("zero examples")
                _log(f"[prepare] {t}: {len(ex)} examples cached")
            except Exception as e:
                failures.append((t, str(e)))
                _log(f"[prepare] {t}: FAILED (load failed: {e})")
        if failures:
            raise RuntimeError(f"dataset preparation failed: {failures}")
        _log("[prepare] all datasets cached successfully")
        return

    if args.merge:
        if not args.output_name:
            raise ValueError("--merge requires --output_name")
        merge(os.path.join(args.results_root, args.output_name))
        return

    if not args.output_name:
        raise ValueError("--output_name required")
    unknown = sorted(set(tasks) - set(ALL_TASKS) - set(KNOWLEDGE_TASKS))
    if unknown:
        raise ValueError(f"unknown tasks: {unknown}")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"invalid shard_index={args.shard_index} for num_shards={args.num_shards}"
        )
    if args.batch_size <= 0 or args.max_len <= 1:
        raise ValueError("batch_size must be positive and max_len must exceed 1")
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

    task_results = {}
    for task in tasks:
        try:
            examples_all = load_task_examples(task)
        except Exception as e:  # dataset unreachable / format change: skip, don't crash
            _log(f"[shard {args.shard_index}/{args.num_shards}] {task}: SKIPPED "
                 f"(load failed: {e})")
            task_results[task] = {
                "skipped": True, "error": str(e)[:400], "n": 0,
                "n_correct_acc": 0, "n_correct_accnorm": 0, "n_nan": 0,
                "n_trunc": 0, "subjects": {},
            }
            continue
        shard = examples_all[args.shard_index::args.num_shards]
        if args.limit and args.limit > 0:
            shard = shard[: args.limit]
        mode = "greedy" if task in GREEDY_TASKS else "mc"
        t0 = time.time()
        n, nca, ncn, nnan, samples, ntr, subjects = score_task(
            model, tok, shard, device, args.batch_size, bool(args.add_bos),
            bos_id, pad_id, args.max_len, mode)
        dt = time.time() - t0
        acc = nca / max(n - nnan, 1)
        accn = ncn / max(n - nnan, 1)
        task_results[task] = {
            "n": n, "n_correct_acc": nca, "n_correct_accnorm": ncn,
            "n_nan": nnan, "n_trunc": ntr, "acc_shard": acc,
            "acc_norm_shard": accn, "mode": mode, "seconds": round(dt, 1),
            "subjects": subjects, "samples": samples,
        }
        _log(f"[shard {args.shard_index}/{args.num_shards}] {task}: n={n} "
             f"acc={acc:.4f} acc_norm={accn:.4f} nan={nnan} trunc={ntr} "
             f"mode={mode} ({dt:.1f}s)")

    results_dir = os.path.join(args.results_root, args.output_name)
    os.makedirs(results_dir, exist_ok=True)
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
