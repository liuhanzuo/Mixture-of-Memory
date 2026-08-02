#!/usr/bin/env python3
"""Paper C P-C1 headline eval: SQuAD dev EM / F1 for the finetuned arms.

Loads ANY P-C1 arm with ZERO arch drift by reusing the Paper B eval loaders
(eval_olmo2_probe2_ppl.load_pruned_model / load_base_model):
  * A1 / A3 / A4 (raw state_dict .pt from train_olmo2_arch_probe2)  -> --ckpt
    (keep_front/n_fresh read from ckpt meta).
  * A2 LoRA merged HF dir (train_olmo2_lora_sft --> <out>/merged)   -> --base_model <merged>
  * base OLMo-2 (untuned reference)                                 -> --base_model <base>

Prompt matches the tokeniser (scripts/tokenize_squad_olmo2_sft.py) EXACTLY but
STOPS before the answer, so the model must produce it:
    Context: {context}\n\nQuestion: {question}\n\nAnswer:
context = " ".join(memory_texts); question = English question (CN prefix stripped
on the fullwidth colon); gold = target_text (single SQuAD short answer).

Greedy free-form generation (do_sample=False), keep the first line of the
completion. Scoring = SQuAD-style normalised EM + token-F1 (verbatim from
eval_olmo2_closedbook_qa: normalize_answer / _f1 / score_prediction). Single-GPU
by default (the val set is 2000 q, short); optional shard/merge like the sibling
evals. add_special_tokens=False / no BOS (OLMo-2 base protocol, --add_bos 1 to
prepend BOS). This is the P-C1 headline; downstream MC (capability) is a separate
run via eval_olmo2_probe2_downstream.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eval_olmo2_probe2_ppl import _log, load_base_model, load_pruned_model  # noqa: E402
from eval_olmo2_closedbook_qa import (  # noqa: E402
    normalize_answer,
    score_prediction,
)

CN_PREFIX_SEP = "："


def _clean_question(input_text: str) -> str:
    if CN_PREFIX_SEP in input_text:
        return input_text.split(CN_PREFIX_SEP, 1)[1].strip()
    return input_text.strip()


def load_squad(path: str):
    """-> list of {"prompt","question","context","gold":[answer]}"""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = _clean_question(rec.get("input_text", ""))
            gold = (rec.get("target_text") or "").strip()
            mem = rec.get("memory_texts") or []
            context = " ".join(m.strip() for m in mem if isinstance(m, str) and m.strip())
            prompt = (f"Context: {context}\n\nQuestion: {q}\n\nAnswer:"
                      if context else f"Question: {q}\n\nAnswer:")
            out.append({"prompt": prompt, "question": q, "context": context,
                        "gold": [gold]})
    return out


@torch.no_grad()
def generate(model, tok, examples, device, batch_size, max_new_tokens,
             add_bos, bos_id, pad_id, max_ctx_len):
    tok.padding_side = "left"
    preds = ["" for _ in examples]
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["prompt"]))
    for b in range(0, len(order), batch_size):
        bidx = order[b:b + batch_size]
        enc_ids = [tok.encode(examples[i]["prompt"], add_special_tokens=False) for i in bidx]
        if add_bos and bos_id is not None:
            enc_ids = [[bos_id] + ids for ids in enc_ids]
        enc_ids = [ids[-max_ctx_len:] for ids in enc_ids]
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
            gen = model.generate(input_ids=input_ids, attention_mask=attn,
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 num_beams=1, pad_token_id=pad_id)
        new_tokens = gen[:, maxl:]
        for r, i in enumerate(bidx):
            txt = tok.decode(new_tokens[r], skip_special_tokens=True)
            preds[i] = txt.strip().split("\n")[0].strip()
    return preds


def merge(results_dir):
    shard_files = sorted(glob.glob(os.path.join(results_dir, "shard*of*.json")))
    if not shard_files:
        raise FileNotFoundError(f"no shard*of*.json in {results_dir}")
    n = em = 0
    f1 = 0.0
    meta = None
    for sf in shard_files:
        with open(sf) as fh:
            d = json.load(fh)
        meta = d.get("meta", meta)
        n += d["n"]
        em += d["em_hits"]
        f1 += d["f1_sum"]
    n = max(n, 1)
    summary = {"output_name": os.path.basename(results_dir.rstrip("/")),
               "n_shards": len(shard_files), "n": n,
               "em": em / n, "f1": f1 / n, "em_hits": em, "f1_sum": f1,
               "meta": meta}
    with open(os.path.join(results_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    _log(f"[merge] {summary['output_name']}: EM={summary['em']:.4f} "
         f"F1={summary['f1']:.4f} (n={n}, {len(shard_files)} shards)")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=False,
                   help="base/merged HF dir (base or A2-merged mode); also the "
                        "cfg source + tokenizer for --ckpt mode")
    p.add_argument("--ckpt", type=str, default="",
                   help="A1/A3/A4 raw .pt (keep/fresh from ckpt meta)")
    p.add_argument("--keep_front_layers", type=int, default=None)
    p.add_argument("--n_fresh_layers", type=int, default=None)
    p.add_argument("--val_path", type=str, default="data/squad_val.jsonl")
    p.add_argument("--tokenizer", type=str, default="",
                   help="tokenizer dir; defaults to --base_model")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--max_ctx_len", type=int, default=1024)
    p.add_argument("--add_bos", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--output_name", type=str, required=False)
    p.add_argument("--results_root", type=str, default="paperC_squad_results")
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()

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

    tok_path = args.tokenizer or args.base_model
    if not tok_path:
        raise ValueError("need --tokenizer or --base_model for the tokenizer")
    tok = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
    bos_id = tok.bos_token_id
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    if args.ckpt:
        model, meta = load_pruned_model(args.ckpt, args.base_model,
                                        args.keep_front_layers, args.n_fresh_layers,
                                        device)
    else:
        if not args.base_model:
            raise ValueError("base/merged mode requires --base_model")
        model, meta = load_base_model(args.base_model, device)
    meta["val_path"] = args.val_path
    meta["add_bos"] = bool(args.add_bos)

    examples_all = load_squad(args.val_path)
    shard = examples_all[args.shard_index::args.num_shards]
    if args.limit and args.limit > 0:
        shard = shard[: args.limit]

    results_dir = os.path.join(args.results_root, args.output_name)
    os.makedirs(results_dir, exist_ok=True)

    t0 = time.time()
    preds = generate(model, tok, shard, device, args.batch_size,
                     args.max_new_tokens, bool(args.add_bos), bos_id, pad_id,
                     args.max_ctx_len)
    dt = time.time() - t0

    em_hits = 0
    f1_sum = 0.0
    pe_out = os.path.join(results_dir,
                          f"per_example_shard{args.shard_index}of{args.num_shards}.jsonl")
    with open(pe_out, "w") as pef:
        for li, (ex, pred) in enumerate(zip(shard, preds)):
            sc = score_prediction(pred, ex["gold"])
            em_hits += sc["em"]
            f1_sum += sc["f1"]
            pef.write(json.dumps({
                "item_id": args.shard_index + li * args.num_shards,
                "question": ex["question"], "gold": ex["gold"], "pred": pred,
                "em": sc["em"], "f1": round(sc["f1"], 4),
            }) + "\n")
    n = len(shard)
    out = os.path.join(results_dir, f"shard{args.shard_index}of{args.num_shards}.json")
    with open(out, "w") as fh:
        json.dump({"shard_index": args.shard_index, "num_shards": args.num_shards,
                   "n": n, "em_hits": em_hits, "f1_sum": f1_sum,
                   "em_shard": em_hits / max(n, 1), "f1_shard": f1_sum / max(n, 1),
                   "seconds": round(dt, 1), "add_bos": bool(args.add_bos),
                   "meta": meta}, fh, indent=2)
    _log(f"[shard {args.shard_index}/{args.num_shards}] n={n} "
         f"EM={em_hits/max(n,1):.4f} F1={f1_sum/max(n,1):.4f} ({dt:.1f}s) -> {out}")


if __name__ == "__main__":
    main()
