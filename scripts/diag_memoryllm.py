"""Diagnostic for MemoryLLM-8B-chat repetitive-output bug (MarcusMarcus...).

Runs a sequence of isolating checks on the SAME loaded checkpoint:
  [README]   verbatim README example (inject David ctx -> ask fruits)
  [NOINJECT] generate with model.initialized=0 (pure base-Llama path)
  [PEFT]     inspect active adapter switching (peft 0.10 vs 0.19 drift)
  [QA1]      5 qa1/0k babilong samples, inject + generate, print raw prompt/tokens
  [ABL]      prompt/gen ablations on sample 0 (chat on/off, examples on/off,
             repetition_penalty / no_repeat_ngram diagnostically)

Goal: distinguish checkpoint/load/generation failure vs inject-memory path vs
prompt/template vs peft-adapter drift.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORYLLM_SRC = PROJECT_ROOT.parent / "MemoryLLM-source"
BABILONG_ROOT = PROJECT_ROOT / "third_party" / "babilong-pkg"
for _path in (str(MEMORYLLM_SRC), str(BABILONG_ROOT), str(PROJECT_ROOT)):
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

import datasets  # noqa: E402
from peft import PeftModelForCausalLM  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.generation import GenerationMixin  # noqa: E402

from modeling_memoryllm import LlamaForCausalLM, LlamaModel, MemoryLLM  # noqa: E402
from babilong.metrics import TASK_LABELS, compare_answers  # noqa: E402
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402

# --- same monkeypatches as the real runner ------------------------------- #
for _cls in (LlamaForCausalLM, MemoryLLM):
    if not issubclass(_cls, GenerationMixin):
        _cls.__bases__ = tuple(dict.fromkeys(_cls.__bases__ + (GenerationMixin,)))
if not hasattr(LlamaModel, "prepare_inputs_for_generation"):
    LlamaModel.prepare_inputs_for_generation = LlamaForCausalLM.prepare_inputs_for_generation
if not hasattr(PeftModelForCausalLM, "prepare_inputs_for_generation"):
    PeftModelForCausalLM.prepare_inputs_for_generation = LlamaForCausalLM.prepare_inputs_for_generation
if not hasattr(DynamicCache, "from_legacy_cache"):
    def _from_legacy_cache(cls, legacy_cache=None):
        cache = cls()
        if legacy_cache is None:
            return cache
        for layer_idx, layer_cache in enumerate(legacy_cache):
            cache.update(layer_cache[0], layer_cache[1], layer_idx)
        return cache
    DynamicCache.from_legacy_cache = classmethod(_from_legacy_cache)
if not hasattr(DynamicCache, "to_legacy_cache"):
    def _to_legacy_cache(self):
        return tuple((layer.keys, layer.values) for layer in self.layers)
    DynamicCache.to_legacy_cache = _to_legacy_cache

DEVICE = "cuda:0"
SNAP = os.environ["MLLM_SNAP"]


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def load():
    tok = AutoTokenizer.from_pretrained(SNAP, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    config = AutoConfig.from_pretrained(SNAP, local_files_only=True)
    raw = json.load(open(os.path.join(SNAP, "config.json")))
    if "rope_theta" not in raw and isinstance(raw.get("rope_scaling"), dict):
        config.rope_theta = raw["rope_scaling"].get("rope_theta", 500000.0)
    else:
        config.rope_theta = raw.get("rope_theta", 500000.0)
    if isinstance(getattr(config, "rope_scaling", None), dict) and "rope_type" in config.rope_scaling and "type" not in config.rope_scaling:
        config.rope_scaling["type"] = config.rope_scaling["rope_type"]
    try:
        model = MemoryLLM.from_pretrained(SNAP, attn_implementation="flash_attention_2",
                                          config=config, torch_dtype=torch.bfloat16, local_files_only=True)
        print("[load] flash_attention_2")
    except Exception as e:
        print(f"[load] FA2 failed: {e}; sdpa")
        model = MemoryLLM.from_pretrained(SNAP, attn_implementation="sdpa",
                                          config=config, torch_dtype=torch.bfloat16, local_files_only=True)
    model = model.to(DEVICE)
    model.eval()
    model.config.use_cache = False
    return model, tok


def gen(model, tok, input_ids, max_new=20, **kw):
    terms = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, max_new_tokens=max_new, eos_token_id=terms,
                             do_sample=False, num_beams=1,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id, use_cache=False, **kw)
    return tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


def chat_ids(tok, content):
    inp = tok.apply_chat_template([{"role": "user", "content": content}],
                                  tokenize=True, return_tensors="pt", add_generation_prompt=True)
    if isinstance(inp, list):
        inp = torch.tensor([inp], dtype=torch.long)
    if hasattr(inp, "input_ids"):
        inp = inp.input_ids
    return inp[:, 1:].to(DEVICE)  # drop bos per README


def snapshot(model):
    return {"memory": model.memory.detach().clone(),
            "initialized": model.initialized.detach().clone()}


def reset(model, snap):
    with torch.no_grad():
        model.memory.copy_(snap["memory"])
        model.initialized.copy_(snap["initialized"])


def inject(model, tok, ctx, max_chunk=1024):
    ids = tok(ctx, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    n = ids.shape[1]
    if n < 16:
        return
    for pos in range(0, n, max_chunk):
        ch = ids[:, pos:pos + max_chunk]
        if ch.shape[1] >= 16:
            with torch.no_grad():
                model.inject_memory(ch, update_memory=True)


def main():
    model, tok = load()
    clean = snapshot(model)
    print(f"[info] memory shape {tuple(model.memory.shape)}, initialized={int(model.initialized.item())}, "
          f"memory mean={model.memory.float().mean():.5f} std={model.memory.float().std():.5f}")

    # ---- PEFT adapter introspection ---- #
    banner("[PEFT] adapter introspection")
    n_default = n_decoder = n_active_attr = 0
    sample_mods = []
    for name, mod in model.named_modules():
        if hasattr(mod, "_active_adapter"):
            n_active_attr += 1
            if len(sample_mods) < 5:
                sample_mods.append((name, type(mod).__name__, getattr(mod, "_active_adapter", None),
                                    getattr(mod, "active_adapter", None),
                                    getattr(mod, "active_adapters", None)))
        if hasattr(mod, "lora_A"):
            keys = list(mod.lora_A.keys()) if hasattr(mod.lora_A, "keys") else []
            if "default" in keys:
                n_default += 1
            if "decoder_adapter" in keys:
                n_decoder += 1
    print(f"modules with _active_adapter attr: {n_active_attr}")
    print(f"lora modules with 'default': {n_default}, with 'decoder_adapter': {n_decoder}")
    for nm, ty, aa, single, multi in sample_mods:
        print(f"  {nm} [{ty}] _active_adapter={aa} active_adapter={single} active_adapters={multi}")

    # ---- README verbatim example ---- #
    banner("[README] inject David context -> 'What fruits does David like?'")
    reset(model, clean)
    ctx = ("Last week, John had a wonderful picnic with David. During their conversation, "
           "David mentioned multiple times that he likes eating apples. Though he didn't mention "
           "any other fruits, John says he can infer that David also like bananas.")
    inject(model, tok, ctx)
    print(f"after inject: initialized={int(model.initialized.item())}")
    ids = chat_ids(tok, "What fruits does David like?")
    print("README answer:", repr(gen(model, tok, ids, max_new=30)))

    # ---- No-inject (pure base path, initialized=0) ---- #
    banner("[NOINJECT] initialized=0, ask a trivial question (base Llama path)")
    reset(model, clean)
    model.initialized.fill_(0)
    ids = chat_ids(tok, "What is the capital of France? Answer in one word.")
    print("no-inject answer:", repr(gen(model, tok, ids, max_new=20)))
    reset(model, clean)

    # ---- 5 qa1/0k samples ---- #
    banner("[QA1] 5 qa1/0k samples: inject context + generate")
    data = datasets.load_dataset("RMT-team/babilong", "0k", download_mode="reuse_dataset_if_exists")
    td = data["qa1"]
    instr = DEFAULT_PROMPTS["qa1"].get("instruction", "")
    exes = DEFAULT_PROMPTS["qa1"].get("examples", "")
    post = DEFAULT_PROMPTS["qa1"].get("post_prompt", "")
    correct = 0
    for i in range(5):
        s = td[i]
        reset(model, clean)
        inject(model, tok, s["input"])
        qprompt = get_formatted_input("", s["question"], exes, instr, post, template=DEFAULT_TEMPLATE)
        ids = chat_ids(tok, qprompt)
        out = gen(model, tok, ids, max_new=20)
        ok = compare_answers(str(s["target"]), out, s["question"], TASK_LABELS["qa1"])
        correct += int(ok)
        if i == 0:
            print(f"--- sample0 raw question_prompt (first 300 chars) ---\n{qprompt[:300]}")
            print(f"--- sample0 chat input_ids[:20]: {ids[0][:20].tolist()}")
            print(f"--- sample0 first decoded out tokens: {repr(out)}")
        print(f"  qa1/0k[{i}] target={s['target']!r} ok={ok} out={out!r}")
    print(f"[QA1] {correct}/5")

    # ---- ablations on sample 0 ---- #
    banner("[ABL] sample0 ablations")
    s = td[0]
    qprompt = get_formatted_input("", s["question"], exes, instr, post, template=DEFAULT_TEMPLATE)

    # ABL-1: chat template ON + repetition_penalty
    reset(model, clean); inject(model, tok, s["input"])
    ids = chat_ids(tok, qprompt)
    print("chat+rep_penalty1.3:", repr(gen(model, tok, ids, max_new=20, repetition_penalty=1.3)))

    # ABL-2: chat template ON + no_repeat_ngram
    reset(model, clean); inject(model, tok, s["input"])
    ids = chat_ids(tok, qprompt)
    print("chat+no_repeat_ngram3:", repr(gen(model, tok, ids, max_new=20, no_repeat_ngram_size=3)))

    # ABL-3: NO chat template (raw tokenization, drop bos)
    reset(model, clean); inject(model, tok, s["input"])
    raw_ids = tok(qprompt, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    print("no-chat-template:", repr(gen(model, tok, raw_ids, max_new=20)))

    # ABL-4: no examples
    reset(model, clean); inject(model, tok, s["input"])
    qp2 = get_formatted_input("", s["question"], "", instr, post, template=DEFAULT_TEMPLATE)
    ids = chat_ids(tok, qp2)
    print("no-examples:", repr(gen(model, tok, ids, max_new=20)))

    # ABL-5: pretrained-style template (README non-chat), no inject reliance on chat
    reset(model, clean); inject(model, tok, s["input"])
    pre = f"Question: {s['question']} Answer:"
    pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
    print("pretrained-style:", repr(gen(model, tok, pre_ids, max_new=20)))

    print("\n[diag] done", flush=True)


if __name__ == "__main__":
    main()
