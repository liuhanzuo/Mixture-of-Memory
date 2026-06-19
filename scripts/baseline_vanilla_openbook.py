#!/usr/bin/env python3
"""Vanilla Llama-3-8B open-book BABILong baseline — eval-harness credibility check.

NO mem_space patch, NO raw-KV, NO chunking: the WHOLE formatted babilong input
(haystack + question) is fed into the plain HF LlamaForCausalLM in ONE forward,
then greedy-generated. Uses the EXACT same prompt formatting (get_formatted_input
+ DEFAULT_PROMPTS / DEFAULT_TEMPLATE, instruction+examples+post_prompt all yes,
no chat template) and the EXACT same scoring (babilong.metrics.compare_answers)
as scripts/run_babilong_mem_space.py, so the only variable removed is the
memory/chunking architecture.

If this scores ~70-90% on qa1 @2k/4k -> the harness is fine and the 10% openbook
seen on the mem_space ckpt was a model/ckpt artifact (double-injection), so the
W0 verdict stands on a good ruler. If this ALSO scores ~10% -> the harness
prompt/scoring is broken and absolute W0 numbers must be re-examined.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402
from babilong.metrics import compare_answers, TASK_LABELS  # noqa: E402
from transformers import AutoTokenizer, LlamaForCausalLM  # noqa: E402
import datasets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Meta-Llama-3-8B")
    ap.add_argument("--dataset_name", default="RMT-team/babilong")
    ap.add_argument("--tasks", nargs="+", default=["qa1"])
    ap.add_argument("--lengths", nargs="+", default=["2k", "4k"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()

    device = torch.device(cli.device)
    tok = AutoTokenizer.from_pretrained(cli.model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[baseline] loading vanilla {cli.model_path} (NO patch)")
    model = LlamaForCausalLM.from_pretrained(
        cli.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()

    for task in cli.tasks:
        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"],
            "examples":    DEFAULT_PROMPTS[task]["examples"],
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"],
        }
        for length in cli.lengths:
            data = datasets.load_dataset(cli.dataset_name, length)
            split = data[task]
            n = min(cli.limit, len(split))
            correct = 0
            for i in range(n):
                s = split[i]
                input_text = get_formatted_input(
                    s["input"], s["question"],
                    prompt_cfg["examples"], prompt_cfg["instruction"],
                    prompt_cfg["post_prompt"], template=DEFAULT_TEMPLATE,
                )
                ids = tok.encode(input_text, add_special_tokens=True,
                                 return_tensors="pt").to(device)
                # Mirror run_babilong_mem_space.generate_with_mem_space greedy
                # decode EXACTLY: manual argmax loop that does NOT stop on EOS at
                # step 0 (the base model often emits EOS first; model.generate
                # would return empty — a harness artifact, not a capability gap).
                with torch.no_grad(), torch.amp.autocast(device_type="cuda",
                                                          dtype=torch.bfloat16):
                    cur = ids
                    gen_ids = []
                    past = None
                    for step in range(cli.max_new_tokens):
                        out_m = model(input_ids=cur if past is None else cur[:, -1:],
                                      past_key_values=past, use_cache=True)
                        past = out_m.past_key_values
                        nxt = out_m.logits[0, -1].argmax()
                        tid = int(nxt.item())
                        if (tok.eos_token_id is not None
                                and tid == tok.eos_token_id and step > 0):
                            break
                        gen_ids.append(tid)
                        cur = torch.cat([cur, nxt.view(1, 1)], dim=-1)
                out = tok.decode(gen_ids, skip_special_tokens=True).strip()
                if compare_answers(str(s["target"]), out, s["question"],
                                   TASK_LABELS[task]):
                    correct += 1
                if i < 8:
                    print(f"  [{task}/{length}] tgt={s['target']!r} "
                          f"out={out[:60]!r} ntok={ids.shape[1]}")
            acc = 100.0 * correct / n
            print(f"==> VANILLA {task} @ {length}: {acc:.1f}%  ({correct}/{n})  "
                  f"ctx_tokens≈{ids.shape[1]}")


if __name__ == "__main__":
    main()
