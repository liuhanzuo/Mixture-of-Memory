#!/usr/bin/env python3
"""Dual-protocol MMLU re-eval for the OLMo-2 prune-then-heal probe (Paper B P0.6).

Question this answers
---------------------
The standard *letter* protocol (show the question + `A./B./C./D.` lettered
options, then compare the likelihood of the single letter token after
`Answer:`) gives the healed / pruned OLMo-2 arms low MMLU accuracy. That low
score conflates two very different failure modes:
  (a) the model lacks the subject competence, or
  (b) the model *has* the competence but cannot bind answer content to a bare
      letter symbol (a readout / answer-symbol binding lag).
To separate them we add a *content* protocol (label-free, ARC-style): the
prompt is just the question + `Answer:`, and each candidate is the full option
*text* scored as a teacher-forced continuation. If content recovers while
letter stays low, the low letter score is a readout lag, not missing knowledge.

Strict pairing
--------------
Both protocols are scored in ONE run over the SAME `cais/mmlu` "all" test set
(14,042 items), the SAME tokenizer, the SAME base-protocol tokenisation
(add_special_tokens=False -> add_bos=0), the SAME truncation rule, and the SAME
stable item_ids (item_id = shard_index + local_idx * num_shards, identical to
scripts/eval_olmo2_probe2_downstream.py) so every downstream paired test
(McNemar, paired bootstrap) lines up letter-vs-content item by item.

Arch construction: NO drift. We import load_base_model / load_pruned_model from
scripts/eval_olmo2_probe2_ppl.py (which copied the trainer's build verbatim) and
encode_pair from scripts/eval_olmo2_probe2_downstream.py, so the shell rebuild +
strict-load AND the tokenisation are byte-identical to the letter-protocol
harness. The letter prompt below is reproduced EXACTLY from that harness's
`mmlu` branch, so this harness's `letter_acc` matches the published letter
numbers item-for-item.

Scoring (lm-eval convention, no dtype drift)
-------------------------------------------
* fp32 weights, bf16-autocast forward (matches training + the PPL/MC evals).
* For each (context, continuation) we encode context+continuation, split off the
  continuation tokens, run the model, and SUM the fp32 log-softmax log-prob of
  every continuation token (teacher forced).
* letter protocol: candidate = single letter " A".." D"; cont is 1 token so
  raw == length-normalised. pred = argmax of the 4 raw sum-logprobs.
* content protocol: candidate = " " + full option text.
    - content_raw  = argmax over the RAW sum-logprob.
    - content_norm = argmax over sum-logprob / (#continuation TOKENS).
      HEADLINE = content_norm (length-normalised), raw disclosed alongside.
* An item is *valid* iff no candidate in EITHER protocol produced a non-finite
  score. Every accuracy is over the valid set, so letter and content are always
  scored on the identical item set -> clean pairing.

Modes
-----
* (default) score a shard: writes shard{i}of{N}.json (counts) +
  per_example_mmlu_shard{i}of{N}.jsonl (full per-item record for both protocols).
* --merge --output_name X : concatenate per-example rows across shards, recompute
  all accuracies + within-arm paired analysis (letter vs content_norm:
  agreement, letter-only / content-only correct, exact McNemar p, paired
  bootstrap 95% CI on content_norm_acc - letter_acc), 57-subject breakdown,
  above-chance for each protocol; writes summary.json + per_example_mmlu.jsonl.
* --compare --file_a ARM.jsonl --file_b BASE.jsonl --protocol P : cross-arm
  paired comparison for one protocol (McNemar, paired bootstrap CI on
  acc_a - acc_b, and above-chance recovery (acc_a-0.25)/(acc_b-0.25) relative to
  a full-base reference). This is how "above-chance recovery relative to full
  base" is produced.
* --selftest : build a TINY random OLMo-2 on CPU + synthetic items, run the full
  pipeline end-to-end and unit-check the raw/norm/letter log-prob maths and the
  McNemar / bootstrap estimators. No GPU, no 7B weights needed.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# NO arch drift: reuse the PPL eval's model construction + the MC eval's exact
# (context, continuation) tokeniser verbatim.
from eval_olmo2_probe2_ppl import (  # noqa: E402
    _log,
    load_base_model,
    load_base_model_any_family,
    load_pruned_model,
    load_truncated_any_family,
)
from eval_olmo2_probe2_downstream import (  # noqa: E402
    _safe_lp,
    encode_pair,
)

_LETTERS = "ABCD"
CHANCE = 0.25  # 4-choice MMLU


# ---------------------------------------------------------------------------
# dataset: cais/mmlu "all" test -> list of examples with BOTH candidate sets.
# The letter prompt is reproduced EXACTLY from eval_olmo2_probe2_downstream.py's
# `mmlu` branch (subject description + lettered body + "\nAnswer:", candidate =
# single letter). The content prompt drops ONLY the lettered option body (the
# part that leaks content<->letter) and keeps the subject description + question
# (so the "question" seen by both protocols is identical), candidate = full text.
# ---------------------------------------------------------------------------
def load_mmlu_examples(content_desc: str = "full"):
    from datasets import load_dataset

    d = load_dataset("cais/mmlu", "all", split="test")
    letters = ["A", "B", "C", "D"]
    out = []
    for ex in d:
        subject_h = ex["subject"].replace("_", " ")
        desc = ("The following are multiple choice questions (with answers) "
                f"about {subject_h}.\n\n")
        ch = ex["choices"]
        n_opt = len(ch)
        question = ex["question"].strip()

        # --- letter protocol prompt (EXACT copy of the letter harness) ---
        body = "\n".join(f"{letters[i]}. {ch[i]}" for i in range(n_opt))
        q_letter = desc + question + "\n" + body + "\nAnswer:"
        letter_cands = [(q_letter, " " + letters[i], len(letters[i]))
                        for i in range(n_opt)]

        # --- content protocol prompt (label-free, ARC style) ---
        if content_desc == "none":
            q_content = "Question: " + question + "\nAnswer:"
        else:  # "full" (default): keep the subject description + question
            q_content = desc + question + "\nAnswer:"
        content_cands = [(q_content, " " + str(ch[i]), len(str(ch[i])))
                         for i in range(n_opt)]

        out.append({
            "gold": int(ex["answer"]),
            "subject": ex["subject"],
            "n_opt": n_opt,
            "letter_cands": letter_cands,
            "content_cands": content_cands,
        })
    return out


# ---------------------------------------------------------------------------
# scoring: one batched teacher-forced pass over ALL candidates (both protocols).
# Returns per-example records with letter / content_raw / content_norm results.
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_examples(model, tok, examples, device, batch_size, add_bos, bos_id,
                   pad_id, max_len, shard_index=0, num_shards=1):
    """For each example score both protocols. Returns (records, n_trunc).
    Each record:
      {item_id, subject, gold, n_opt, nan(bool),
       letter:  {pred, correct, scores[raw per opt]},
       content_raw:  {pred, correct, scores[raw per opt]},
       content_norm: {pred, correct, scores[norm per opt], cont_tokens[per opt]}}
    NaN in ANY candidate -> nan=True and preds/correct set to None/False; such
    items are dropped from every accuracy at merge/aggregate time (clean pairing).
    """
    # flatten: (ex_idx, proto, cand_idx, ids, cont_start, cont_len)
    items = []
    n_trunc = 0
    for ei, ex in enumerate(examples):
        for proto, key in (("L", "letter_cands"), ("C", "content_cands")):
            for ci, (ctx, cont, _norm_chars) in enumerate(ex[key]):
                ids, cs, cl = encode_pair(tok, ctx, cont, add_bos, bos_id)
                if len(ids) > max_len:
                    drop = len(ids) - max_len
                    ids = ids[drop:]
                    cs = max(cs - drop, 1)
                    cl = min(cl, len(ids) - cs)
                    if cl <= 0:
                        cs, cl = max(len(ids) - 1, 1), 1
                    n_trunc += 1
                items.append((ei, proto, ci, ids, cs, cl))

    # raw sum-logprob and continuation token count, per (example, proto, cand)
    raw = {}   # (ei, proto, ci) -> float sum-logprob
    ntok = {}  # (ei, proto, ci) -> int continuation token count
    nanflag = {}

    order = sorted(range(len(items)), key=lambda i: len(items[i][3]))
    for b in range(0, len(order), batch_size):
        bidx = order[b:b + batch_size]
        maxl = max(len(items[i][3]) for i in bidx)
        B = len(bidx)
        input_ids = torch.full((B, maxl), pad_id, dtype=torch.long)
        attn = torch.zeros((B, maxl), dtype=torch.long)
        for r, i in enumerate(bidx):
            ids = items[i][3]
            input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[r, :len(ids)] = 1
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attn)
        else:
            out = model(input_ids=input_ids, attention_mask=attn)
        logprobs = torch.log_softmax(out.logits.float(), dim=-1)  # [B, L, V]
        for r, i in enumerate(bidx):
            ei, proto, ci, ids, cs, cl = items[i]
            end = cs + cl
            pos = torch.arange(cs - 1, end - 1, device=device)
            tgt = torch.tensor(ids[cs:end], dtype=torch.long, device=device)
            lp = logprobs[r, pos, tgt].sum().item()
            raw[(ei, proto, ci)] = lp
            ntok[(ei, proto, ci)] = cl
            if not math.isfinite(lp):
                nanflag[(ei, proto, ci)] = True

    records = []
    for ei, ex in enumerate(examples):
        n_opt = ex["n_opt"]
        gold = ex["gold"]
        item_id = shard_index + ei * num_shards
        L = [raw[(ei, "L", ci)] for ci in range(n_opt)]
        C = [raw[(ei, "C", ci)] for ci in range(n_opt)]
        Ct = [ntok[(ei, "C", ci)] for ci in range(n_opt)]
        Cn = [C[ci] / max(Ct[ci], 1) for ci in range(n_opt)]
        is_nan = any(not math.isfinite(x) for x in L + C)
        rec = {
            "item_id": item_id,
            "subject": ex["subject"],
            "gold": gold,
            "gold_letter": _LETTERS[gold] if 0 <= gold < len(_LETTERS) else str(gold),
            "n_opt": n_opt,
            "nan": bool(is_nan),
        }
        if is_nan:
            rec["letter"] = {"pred": None, "correct": False,
                             "scores": {_LETTERS[k]: _safe_lp(L[k]) for k in range(n_opt)}}
            rec["content_raw"] = {"pred": None, "correct": False,
                                  "scores": {_LETTERS[k]: _safe_lp(C[k]) for k in range(n_opt)}}
            rec["content_norm"] = {"pred": None, "correct": False,
                                   "scores": {_LETTERS[k]: _safe_lp(Cn[k]) for k in range(n_opt)},
                                   "cont_tokens": {_LETTERS[k]: Ct[k] for k in range(n_opt)}}
            records.append(rec)
            continue
        p_letter = max(range(n_opt), key=lambda k: L[k])
        p_craw = max(range(n_opt), key=lambda k: C[k])
        p_cnorm = max(range(n_opt), key=lambda k: Cn[k])
        rec["letter"] = {
            "pred": p_letter, "pred_letter": _LETTERS[p_letter],
            "correct": bool(p_letter == gold),
            "scores": {_LETTERS[k]: _safe_lp(L[k]) for k in range(n_opt)}}
        rec["content_raw"] = {
            "pred": p_craw, "pred_letter": _LETTERS[p_craw],
            "correct": bool(p_craw == gold),
            "scores": {_LETTERS[k]: _safe_lp(C[k]) for k in range(n_opt)}}
        rec["content_norm"] = {
            "pred": p_cnorm, "pred_letter": _LETTERS[p_cnorm],
            "correct": bool(p_cnorm == gold),
            "scores": {_LETTERS[k]: _safe_lp(Cn[k]) for k in range(n_opt)},
            "cont_tokens": {_LETTERS[k]: Ct[k] for k in range(n_opt)}}
        records.append(rec)
    return records, n_trunc


# ---------------------------------------------------------------------------
# paired statistics (pure numeric; no scipy dep)
# ---------------------------------------------------------------------------
def mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value. b = #(A correct, B wrong),
    c = #(A wrong, B correct). Under H0 each discordant pair is 50/50, so the
    smaller count ~ Binomial(n=b+c, 0.5). p = min(1, 2 * P[X <= min(b,c)]).

    Computed in log-space (lgamma) with a log-sum-exp accumulation so that large
    n (thousands of discordant pairs over the full 14,042-item MMLU set) does not
    overflow: math.comb(n, i) alone is astronomically large and multiplying it by
    the underflowed float 0.5**n raises OverflowError."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    log_half_n = n * math.log(0.5)
    # log P[X = i] = log C(n, i) + n*log(0.5), i = 0..k
    log_terms = [
        (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
         + log_half_n)
        for i in range(0, k + 1)
    ]
    m = max(log_terms)
    log_tail = m + math.log(sum(math.exp(t - m) for t in log_terms))
    tail = math.exp(log_tail)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_diff(correct_a, correct_b, n_boot=10000, seed=0):
    """Paired bootstrap 95% CI for acc_a - acc_b over the same item set.
    correct_a / correct_b are 0/1 int arrays of equal length (paired by item)."""
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    assert a.shape == b.shape and a.size > 0
    n = a.size
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    d = a - b
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = d[idx].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


# ---------------------------------------------------------------------------
# aggregate a list of per-example records -> summary block
# ---------------------------------------------------------------------------
def aggregate(records, do_subjects=True, n_boot=10000, seed=0):
    valid = [r for r in records if not r["nan"]]
    n_all = len(records)
    n_valid = len(valid)
    n_nan = n_all - n_valid

    L = np.array([1 if r["letter"]["correct"] else 0 for r in valid], dtype=np.int64)
    CR = np.array([1 if r["content_raw"]["correct"] else 0 for r in valid], dtype=np.int64)
    CN = np.array([1 if r["content_norm"]["correct"] else 0 for r in valid], dtype=np.int64)

    def acc(x):
        return float(x.mean()) if x.size else None

    letter_acc = acc(L)
    craw_acc = acc(CR)
    cnorm_acc = acc(CN)

    # within-arm paired analysis: letter vs content_norm (the headline pair)
    both = int(np.sum((L == 1) & (CN == 1)))
    letter_only = int(np.sum((L == 1) & (CN == 0)))
    content_only = int(np.sum((L == 0) & (CN == 1)))
    neither = int(np.sum((L == 0) & (CN == 0)))
    agreement = float(np.mean(L == CN)) if n_valid else None
    mcnemar_p = mcnemar_exact_p(letter_only, content_only)  # b=letter-only, c=content-only
    if n_valid:
        diff, lo, hi = paired_bootstrap_diff(CN, L, n_boot=n_boot, seed=seed)
    else:
        diff = lo = hi = None

    summary = {
        "n": n_all,
        "n_valid": n_valid,
        "n_nan": n_nan,
        "chance": CHANCE,
        "letter_acc": letter_acc,
        "content_raw_acc": craw_acc,
        "content_norm_acc": cnorm_acc,
        "above_chance": {
            "letter": (letter_acc - CHANCE) if letter_acc is not None else None,
            "content_raw": (craw_acc - CHANCE) if craw_acc is not None else None,
            "content_norm": (cnorm_acc - CHANCE) if cnorm_acc is not None else None,
        },
        # headline within-arm pairing: content_norm vs letter
        "letter_vs_content_norm": {
            "both_correct": both,
            "letter_only_correct": letter_only,
            "content_only_correct": content_only,
            "neither_correct": neither,
            "agreement": agreement,
            "mcnemar_exact_p": mcnemar_p,
            "bootstrap_diff_content_norm_minus_letter": diff,
            "bootstrap_ci95": [lo, hi],
            "n_boot": n_boot,
        },
    }

    if do_subjects:
        subj = {}
        for r in valid:
            s = r["subject"]
            d = subj.setdefault(s, {"n": 0, "letter_c": 0, "craw_c": 0, "cnorm_c": 0})
            d["n"] += 1
            d["letter_c"] += 1 if r["letter"]["correct"] else 0
            d["craw_c"] += 1 if r["content_raw"]["correct"] else 0
            d["cnorm_c"] += 1 if r["content_norm"]["correct"] else 0
        subj_out = {}
        for s, d in sorted(subj.items()):
            nn = max(d["n"], 1)
            subj_out[s] = {
                "n": d["n"],
                "letter_acc": d["letter_c"] / nn,
                "content_raw_acc": d["craw_c"] / nn,
                "content_norm_acc": d["cnorm_c"] / nn,
            }
        summary["subjects"] = subj_out
    return summary


# ---------------------------------------------------------------------------
# merge shards
# ---------------------------------------------------------------------------
def read_per_example(results_dir):
    pe_files = sorted(glob.glob(
        os.path.join(results_dir, "per_example_mmlu_shard*of*.jsonl")))
    if not pe_files:
        raise FileNotFoundError(
            f"no per_example_mmlu_shard*of*.jsonl in {results_dir}")
    recs = []
    for pf in pe_files:
        with open(pf) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    recs.sort(key=lambda r: r.get("item_id", 0))
    return recs, pe_files


def merge(results_dir, n_boot=10000, seed=0):
    recs, pe_files = read_per_example(results_dir)
    # carry meta from any shard json
    meta = None
    for sf in sorted(glob.glob(os.path.join(results_dir, "shard*of*.json"))):
        with open(sf) as f:
            meta = json.load(f).get("meta", meta)
    agg = aggregate(recs, do_subjects=True, n_boot=n_boot, seed=seed)
    summary = {
        "output_name": os.path.basename(results_dir.rstrip("/")),
        "task": "mmlu",
        "protocol": "letter+content(dual)",
        "n_shards": len(pe_files),
        "meta": meta,
        **agg,
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # merged per-example file (sorted)
    with open(os.path.join(results_dir, "per_example_mmlu.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    lv = agg["letter_vs_content_norm"]
    _log(f"[merge] {summary['output_name']}: n_valid={agg['n_valid']} nan={agg['n_nan']} "
         f"| letter={agg['letter_acc']:.4f} content_raw={agg['content_raw_acc']:.4f} "
         f"content_norm={agg['content_norm_acc']:.4f} "
         f"| content_only={lv['content_only_correct']} letter_only={lv['letter_only_correct']} "
         f"McNemar_p={lv['mcnemar_exact_p']:.3e} "
         f"CI95(cn-l)=[{lv['bootstrap_ci95'][0]:.4f},{lv['bootstrap_ci95'][1]:.4f}]")
    return summary


# ---------------------------------------------------------------------------
# cross-arm paired compare (above-chance recovery relative to a base reference)
# ---------------------------------------------------------------------------
def _load_pe(path):
    if os.path.isdir(path):
        p = os.path.join(path, "per_example_mmlu.jsonl")
    else:
        p = path
    recs = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["item_id"]] = r
    return recs


def compare(file_a, file_b, protocol, n_boot=10000, seed=0):
    """Paired arm-vs-base comparison for one protocol
    (protocol in {letter, content_raw, content_norm}). file_a = arm, file_b =
    base reference. Only items VALID in both are used."""
    A = _load_pe(file_a)
    B = _load_pe(file_b)
    common = sorted(set(A) & set(B))
    ca, cb = [], []
    for iid in common:
        ra, rb = A[iid], B[iid]
        if ra["nan"] or rb["nan"]:
            continue
        ca.append(1 if ra[protocol]["correct"] else 0)
        cb.append(1 if rb[protocol]["correct"] else 0)
    ca = np.array(ca, dtype=np.int64)
    cb = np.array(cb, dtype=np.int64)
    n = ca.size
    acc_a = float(ca.mean()) if n else None
    acc_b = float(cb.mean()) if n else None
    b = int(np.sum((ca == 1) & (cb == 0)))  # arm right, base wrong
    c = int(np.sum((ca == 0) & (cb == 1)))  # arm wrong, base right
    mcp = mcnemar_exact_p(b, c)
    diff, lo, hi = paired_bootstrap_diff(ca, cb, n_boot=n_boot, seed=seed)
    denom = (acc_b - CHANCE) if (acc_b is not None and acc_b != CHANCE) else None
    recovery = ((acc_a - CHANCE) / denom) if denom else None
    out = {
        "protocol": protocol,
        "file_a": file_a, "file_b": file_b,
        "n_paired_valid": int(n),
        "acc_a": acc_a, "acc_b": acc_b,
        "above_chance_a": (acc_a - CHANCE) if acc_a is not None else None,
        "above_chance_b": (acc_b - CHANCE) if acc_b is not None else None,
        "above_chance_recovery_a_over_b": recovery,
        "arm_right_base_wrong": b,
        "arm_wrong_base_right": c,
        "mcnemar_exact_p": mcp,
        "bootstrap_diff_a_minus_b": diff,
        "bootstrap_ci95": [lo, hi],
        "n_boot": n_boot,
    }
    _log(f"[compare/{protocol}] acc_a={acc_a:.4f} acc_b={acc_b:.4f} "
         f"recovery={recovery if recovery is None else round(recovery,4)} "
         f"McNemar_p={mcp:.3e} diff={diff:.4f} CI95=[{lo:.4f},{hi:.4f}] n={n}")
    return out


# ---------------------------------------------------------------------------
# self-test: tiny CPU OLMo-2 + synthetic items; validate maths end-to-end
# ---------------------------------------------------------------------------
def _selftest():
    from transformers import Olmo2Config, Olmo2ForCausalLM, AutoTokenizer
    import torch.nn.functional as F

    _log("[selftest] building tiny CPU OLMo-2 + synthetic MMLU items ...")
    base = os.environ.get("SELFTEST_BASE", "../models/OLMo-2-1124-7B")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    cfg = Olmo2Config.from_pretrained(base, local_files_only=True)
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_hidden_layers = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 4
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    torch.manual_seed(0)
    model = Olmo2ForCausalLM(cfg).to(torch.float32).eval()
    device = torch.device("cpu")
    bos_id = tok.bos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (
        tok.eos_token_id if tok.eos_token_id is not None else 0)

    # 4 tiny synthetic 4-choice items (letter + content candidate sets)
    examples = []
    seeds = [
        ("What is the capital of France?", ["Paris", "London city", "Berlin", "Rome"], 0),
        ("Two plus two equals?", ["three", "four", "five", "six"], 1),
        ("Water is made of hydrogen and?", ["oxygen", "carbon", "nitrogen", "helium"], 0),
        ("The sun rises in the?", ["west", "north", "east", "south"], 2),
    ]
    letters = ["A", "B", "C", "D"]
    for q, ch, gold in seeds:
        desc = "The following are multiple choice questions (with answers) about test.\n\n"
        body = "\n".join(f"{letters[i]}. {ch[i]}" for i in range(4))
        q_letter = desc + q + "\n" + body + "\nAnswer:"
        q_content = desc + q + "\nAnswer:"
        examples.append({
            "gold": gold, "subject": "test", "n_opt": 4,
            "letter_cands": [(q_letter, " " + letters[i], 1) for i in range(4)],
            "content_cands": [(q_content, " " + ch[i], len(ch[i])) for i in range(4)],
        })

    recs, n_trunc = score_examples(
        model, tok, examples, device, batch_size=8, add_bos=False, bos_id=bos_id,
        pad_id=pad_id, max_len=512, shard_index=0, num_shards=1)

    # --- structural / sanity checks ---
    assert len(recs) == 4
    for r in recs:
        assert set(r.keys()) >= {"item_id", "subject", "gold", "nan",
                                 "letter", "content_raw", "content_norm"}
        for proto in ("letter", "content_raw", "content_norm"):
            assert r[proto]["pred"] in (0, 1, 2, 3), (proto, r[proto]["pred"])
            for v in r[proto]["scores"].values():
                assert v is None or math.isfinite(v)
        # length-normalised == raw / cont_tokens for content
        for k in ("A", "B", "C", "D"):
            raw_v = r["content_raw"]["scores"][k]
            nt = r["content_norm"]["cont_tokens"][k]
            norm_v = r["content_norm"]["scores"][k]
            assert abs(norm_v - raw_v / max(nt, 1)) < 1e-6, (k, raw_v, nt, norm_v)
    _log("[selftest] OK: schema + norm==raw/cont_tokens for all items/options")

    # --- independent recompute of ONE candidate's raw sum-logprob ---
    ex = examples[0]
    ctx, cont, _ = ex["content_cands"][0]
    ids, cs, cl = encode_pair(tok, ctx, cont, False, bos_id)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]))
    lp = torch.log_softmax(out.logits.float(), dim=-1)
    pos = torch.arange(cs - 1, cs + cl - 1)
    tgt = torch.tensor(ids[cs:cs + cl])
    manual = float(lp[0, pos, tgt].sum().item())
    stored = recs[0]["content_raw"]["scores"]["A"]
    assert abs(manual - stored) < 1e-4, (manual, stored)
    _log(f"[selftest] OK: independent raw sum-logprob recompute matches "
         f"({manual:.5f} == {stored:.5f})")

    # --- McNemar exact known cases ---
    assert abs(mcnemar_exact_p(0, 0) - 1.0) < 1e-12
    # b=10,c=0 -> p = 2 * 0.5^10 = 1/512
    assert abs(mcnemar_exact_p(10, 0) - 2 * (0.5 ** 10)) < 1e-12
    # symmetric in (b,c)
    assert abs(mcnemar_exact_p(3, 7) - mcnemar_exact_p(7, 3)) < 1e-12
    # b=c is maximally non-significant -> p == 1
    assert abs(mcnemar_exact_p(5, 5) - 1.0) < 1e-9
    _log("[selftest] OK: mcnemar_exact_p known cases")

    # --- paired bootstrap: A strictly dominates B -> diff>0, CI lo>=0 ---
    a = [1] * 80 + [0] * 20
    b = [0] * 80 + [0] * 20  # A right on first 80, B never right there
    diff, lo, hi = paired_bootstrap_diff(a, b, n_boot=2000, seed=1)
    assert diff > 0 and lo <= diff <= hi and lo >= 0, (diff, lo, hi)
    _log(f"[selftest] OK: paired bootstrap diff={diff:.3f} CI=[{lo:.3f},{hi:.3f}]")

    # --- aggregate + compare run end-to-end ---
    agg = aggregate(recs, do_subjects=True, n_boot=500, seed=0)
    assert 0.0 <= agg["letter_acc"] <= 1.0
    assert 0.0 <= agg["content_norm_acc"] <= 1.0
    lv = agg["letter_vs_content_norm"]
    assert (lv["both_correct"] + lv["letter_only_correct"] +
            lv["content_only_correct"] + lv["neither_correct"]) == agg["n_valid"]
    assert 0.0 <= lv["mcnemar_exact_p"] <= 1.0
    assert lv["bootstrap_ci95"][0] <= lv["bootstrap_ci95"][1]
    _log(f"[selftest] OK: aggregate letter_acc={agg['letter_acc']:.3f} "
         f"content_norm_acc={agg['content_norm_acc']:.3f} "
         f"(random tiny model -> accs not meaningful, structure valid)")
    _log("[selftest] ALL CHECKS PASSED")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=False,
                   help="pretrained OLMo-2 path (cfg source for pruned mode; the "
                        "full model itself in base mode)")
    p.add_argument("--ckpt", type=str, default="",
                   help="prune-then-heal .pt (omit -> full-base mode). ShortGPT is "
                        "loaded with --keep_front_layers 16 --n_fresh_layers 0.")
    p.add_argument("--any_family", action="store_true",
                   help="base mode only: load --base_model with AutoModelForCausalLM "
                        "instead of Olmo2ForCausalLM, so a non-OLMo family (Llama, "
                        "Qwen, ...) can be scored through the identical MC interface. "
                        "A01 gate-1. Incompatible with --ckpt (layer surgery is "
                        "OLMo-specific).")
    p.add_argument("--keep_front_layers", type=int, default=None,
                   help="pruned mode; default read from ckpt meta")
    p.add_argument("--n_fresh_layers", type=int, default=None,
                   help="pruned mode; default read from ckpt meta (else 2)")
    p.add_argument("--keep_indices", type=str, default="",
                   help="informational only (ShortGPT selected-layer indices). The "
                        "healed ShortGPT ckpt stores a plain N-layer state_dict, so "
                        "loading uses keep_front_layers=N/n_fresh_layers=0; this arg "
                        "is recorded in meta for provenance and does not affect load.")
    p.add_argument("--content_desc", type=str, default="full",
                   choices=["full", "none"],
                   help="content-protocol prompt: 'full' keeps the MMLU subject "
                        "description + question (default, matches letter framing); "
                        "'none' uses a bare 'Question: ...\\nAnswer:'.")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--add_bos", type=int, default=0,
                   help="0 (default, base protocol / OLMo-2 lm-eval) or 1 to prepend BOS")
    p.add_argument("--limit", type=int, default=0,
                   help=">0 caps examples for THIS shard (post-strided); sanity only")
    p.add_argument("--output_name", type=str, required=False)
    p.add_argument("--results_root", type=str, default="olmo2_mmlu_content_results")
    p.add_argument("--n_boot", type=int, default=10000,
                   help="paired-bootstrap resamples (merge/compare)")
    p.add_argument("--boot_seed", type=int, default=0)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--prepare_data", action="store_true",
                   help="load cais/mmlu (populate cache) then exit; run ONCE before "
                        "fanning out shards to avoid a download race")
    p.add_argument("--compare", action="store_true",
                   help="cross-arm paired compare mode (needs --file_a --file_b --protocol)")
    p.add_argument("--file_a", type=str, default="",
                   help="compare: arm per_example dir or jsonl")
    p.add_argument("--file_b", type=str, default="",
                   help="compare: base reference per_example dir or jsonl")
    p.add_argument("--protocol", type=str, default="content_norm",
                   choices=["letter", "content_raw", "content_norm"],
                   help="compare: which protocol to pair on")
    p.add_argument("--selftest", action="store_true",
                   help="CPU self-test (tiny model + synthetic items); no GPU/7B needed")
    args = p.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.compare:
        if not (args.file_a and args.file_b):
            raise ValueError("--compare requires --file_a and --file_b")
        res = compare(args.file_a, args.file_b, args.protocol,
                      n_boot=args.n_boot, seed=args.boot_seed)
        if args.output_name:
            os.makedirs(args.results_root, exist_ok=True)
            outp = os.path.join(args.results_root, args.output_name + "_compare.json")
            with open(outp, "w") as f:
                json.dump(res, f, indent=2)
            _log(f"[compare] wrote {outp}")
        return

    if args.prepare_data:
        ex = load_mmlu_examples(args.content_desc)
        _log(f"[prepare] cais/mmlu: {len(ex)} examples cached")
        return

    if args.merge:
        if not args.output_name:
            raise ValueError("--merge requires --output_name")
        merge(os.path.join(args.results_root, args.output_name),
              n_boot=args.n_boot, seed=args.boot_seed)
        return

    if not args.output_name:
        raise ValueError("--output_name required")
    if args.any_family and args.ckpt:
        raise ValueError(
            "--any_family with --ckpt is not supported; use --any_family with "
            "--keep_front_layers (base mode, truncation, no heal) or without "
            "--keep_front_layers (untouched base)"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (use --selftest for a CPU dry run)")
    device = torch.device("cuda")

    from transformers import AutoTokenizer
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
        if args.any_family:
            # A01 gate-1: the same letter-vs-content MC interface on a NON-OLMo
            # family. Only the model class changes; scoring/tokenising/nulls are
            # the identical code path, which is what makes the comparison valid.
            if args.keep_front_layers:
                # Damaged variant: truncate the base to its first keep_front layers,
                # no fresh block, no heal. Answers whether letter-interface failure
                # transfers to *damaged* non-OLMo transformers.
                model, meta = load_truncated_any_family(
                    args.base_model, args.keep_front_layers, device)
            else:
                model, meta = load_base_model_any_family(args.base_model, device)
        else:
            model, meta = load_base_model(args.base_model, device)
    meta["base_model"] = args.base_model
    meta["add_bos"] = bool(args.add_bos)
    meta["content_desc"] = args.content_desc
    if args.keep_indices:
        meta["keep_indices"] = args.keep_indices

    examples = load_mmlu_examples(args.content_desc)
    examples = examples[args.shard_index::args.num_shards]
    if args.limit and args.limit > 0:
        examples = examples[: args.limit]

    results_dir = os.path.join(args.results_root, args.output_name)
    os.makedirs(results_dir, exist_ok=True)

    t0 = time.time()
    records, n_trunc = score_examples(
        model, tok, examples, device, args.batch_size, bool(args.add_bos),
        bos_id, pad_id, args.max_len,
        shard_index=args.shard_index, num_shards=args.num_shards)
    dt = time.time() - t0

    # shard-local aggregate (no per-subject to keep the shard file small; the
    # merge recomputes everything from per-example rows)
    shard_agg = aggregate(records, do_subjects=False, n_boot=1000,
                          seed=args.boot_seed)

    pe_out = os.path.join(
        results_dir,
        f"per_example_mmlu_shard{args.shard_index}of{args.num_shards}.jsonl")
    with open(pe_out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    shard_json = os.path.join(
        results_dir, f"shard{args.shard_index}of{args.num_shards}.json")
    with open(shard_json, "w") as f:
        json.dump({
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "n_trunc": n_trunc,
            "seconds": round(dt, 1),
            "meta": meta,
            "shard_aggregate": shard_agg,
        }, f, indent=2)
    _log(f"[shard {args.shard_index}/{args.num_shards}] n={len(records)} "
         f"valid={shard_agg['n_valid']} nan={shard_agg['n_nan']} "
         f"letter={shard_agg['letter_acc']:.4f} "
         f"content_raw={shard_agg['content_raw_acc']:.4f} "
         f"content_norm={shard_agg['content_norm_acc']:.4f} "
         f"trunc={n_trunc} ({dt:.1f}s) -> {pe_out}")


if __name__ == "__main__":
    main()
