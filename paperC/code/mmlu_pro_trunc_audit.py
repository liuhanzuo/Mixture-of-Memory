#!/usr/bin/env python
"""paperC task #251 follow-up — quantify the MMLU-Pro truncation defect.

WHY
---
#251's cross-family driver ran at MAXLEN=1536, a cap that was measured with the
**OLMo-2** tokenizer (max letter prompt 1226 tok). Two of the three non-OLMo
families overflow it:

    llama2_7b        n_trunc = 40   (identical on base/k8/k10/k12/k14)
    qwen3_8b_base    n_trunc = 20   (identical on every rung)
    llama3_8b        n_trunc = 0
    OLMo-2           n_trunc = 0

`n_trunc` counts *candidate encodings*, not items. It is constant across rungs
within a family and independent of damage, so this is a (tokenizer x prompt
length) property, not a model effect -- but `score_examples` LEFT-truncates, so
for those candidates the labelled option body that the letter interface is
supposed to read is partly gone. The thing being scored changed.

WHAT THIS SCRIPT DOES (CPU only, no GPU, no model weights)
----------------------------------------------------------
For each tokenizer, re-encodes every (item, proto, candidate) exactly as
`encode_pair` does and reports:
  * per-family n_trunc at a given cap (must reproduce 40 / 20 / 0 / 0 at 1536)
  * the max encoded length, hence the cap that guarantees n_trunc == 0
  * the exact set of affected item_ids, so option (b) (union-exclusion) can be
    costed against option (a) (raise the cap and re-run)

Usage:
  python paperC/code/mmlu_pro_trunc_audit.py --max_len 1536 [--out audit.json]
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from eval_olmo2_mc_letter_content import load_mc_examples  # noqa: E402
from eval_olmo2_mmlu_content import encode_pair  # noqa: E402

MODELS_DIRS = [
    "/apdcephfs_zwfy6/share_304376610/pighzliu_code/models",
    "/apdcephfs_wzc1/share_304376610/pighzliu_code/models",
]
FAMILIES = {
    "llama2_7b": "Llama--Llama2-7b",
    "llama3_8b": "Llama--Llama3-8b",
    "qwen3_8b_base": "Qwen3-8B-Base",
    "olmo2_7b": "OLMo-2-1124-7B",
}


def resolve(stem):
    for d in MODELS_DIRS:
        p = os.path.join(d, stem)
        if os.path.isdir(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--task", type=str, default="mmlu_pro")
    ap.add_argument("--num_shards", type=int, default=8)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    examples = load_mc_examples(args.task, "none")
    print(f"[audit] {args.task}: {len(examples)} items", flush=True)

    report = {"task": args.task, "n_items": len(examples),
              "max_len_probed": args.max_len, "num_shards": args.num_shards,
              "families": {}}

    for fam, stem in FAMILIES.items():
        path = resolve(stem)
        if path is None:
            print(f"[audit] SKIP {fam}: dir absent ({stem})")
            continue
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        bos_id = tok.bos_token_id
        n_trunc = 0
        maxlen_seen = 0
        max_letter = 0
        max_content = 0
        by_proto = {"L": 0, "C": 0}
        hit_items = {}
        for ei, ex in enumerate(examples):
            # item_id in the sharded harness: shard_index + local_i * num_shards,
            # with shard_index = ei % num_shards and local_i = ei // num_shards,
            # i.e. item_id == ei. Verified by construction below.
            shard_index = ei % args.num_shards
            local_i = ei // args.num_shards
            item_id = shard_index + local_i * args.num_shards
            assert item_id == ei
            for proto, key in (("L", "letter_cands"), ("C", "content_cands")):
                for ci, (ctx, cont, _nc) in enumerate(ex[key]):
                    ids, _cs, _cl = encode_pair(tok, ctx, cont, False, bos_id)
                    L = len(ids)
                    maxlen_seen = max(maxlen_seen, L)
                    if proto == "L":
                        max_letter = max(max_letter, L)
                    else:
                        max_content = max(max_content, L)
                    if L > args.max_len:
                        n_trunc += 1
                        by_proto[proto] += 1
                        d = hit_items.setdefault(item_id, {"L": 0, "C": 0,
                                                           "max_tok": 0,
                                                           "n_opt": ex["n_opt"],
                                                           "shard": shard_index})
                        d[proto] += 1
                        d["max_tok"] = max(d["max_tok"], L)
        report["families"][fam] = {
            "model_path": path,
            "vocab_size": len(tok),
            "n_trunc_at_probe": n_trunc,
            "n_trunc_letter": by_proto["L"],
            "n_trunc_content": by_proto["C"],
            "max_encoded_tokens": maxlen_seen,
            "max_letter_prompt_tokens": max_letter,
            "max_content_pair_tokens": max_content,
            "affected_item_ids": sorted(hit_items),
            "affected_detail": {str(k): v for k, v in sorted(hit_items.items())},
        }
        print(f"[audit] {fam:16s} vocab={len(tok):6d} n_trunc@{args.max_len}="
              f"{n_trunc:4d} (L={by_proto['L']} C={by_proto['C']}) "
              f"max_tok={maxlen_seen} (letter {max_letter} / content {max_content}) "
              f"items={sorted(hit_items)}", flush=True)

    fams = report["families"]
    union = sorted({i for f in fams.values() for i in f["affected_item_ids"]})
    report["union_affected_item_ids"] = union
    report["n_union_affected"] = len(union)
    report["global_max_encoded_tokens"] = max(
        (f["max_encoded_tokens"] for f in fams.values()), default=0)
    print(f"[audit] union affected items = {len(union)} -> {union}")
    print(f"[audit] global max encoded tokens = "
          f"{report['global_max_encoded_tokens']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[audit] wrote {args.out}")


if __name__ == "__main__":
    main()
