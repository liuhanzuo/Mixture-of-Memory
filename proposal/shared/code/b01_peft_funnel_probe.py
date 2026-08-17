#!/usr/bin/env python
"""Does PEFT still target the READ band after inject_bottleneck() wraps layer 12?

WHY THIS MATTERS
----------------
Arms 3/4 of B01's four-arm gate need "funnel checkpoint + Read-LoRA". Every eval driver
currently refuses that combination via an argparse mutex. The tempting conclusion is
"just lift the mutex". This script tests whether that would be SUFFICIENT or whether it
would silently produce a WRONG model.

inject_bottleneck() replaces model.model.layers[12] with
BottleneckLayer(inner=<Qwen3DecoderLayer>). So the submodule formerly named
    model.layers.12.self_attn.q_proj
is now named
    model.layers.12.inner.self_attn.q_proj

The Read-LoRA adapter_config.json pins layers_to_transform=[12..35] with
layers_pattern="layers". PEFT resolves a target's layer index by regex over the module
NAME. If the extra ".inner" segment changes what PEFT matches, then loading the adapter
onto a funnel model would attach the wrong set of modules -- or none at layer 12 --
WITHOUT raising. That is a silent-wrong-model failure, which is worse than the mutex.

Tiny Qwen3 (36 layers x hidden 64) so this is seconds on CPU. No GPU. No writes.
"""
from __future__ import annotations

import json
import sys

import torch
import torch.nn as nn


def main():
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    cfg = Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=36, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512,
    )
    model = AutoModelForCausalLM.from_config(cfg)
    L = cfg.num_hidden_layers
    print(f"[peft-probe] tiny Qwen3: L={L} hidden={cfg.hidden_size}", flush=True)

    def adapted_layer_indices(m):
        """Which layer indices actually received a lora_A?"""
        import re
        idx = set()
        for name, _ in m.named_modules():
            if name.endswith("lora_A"):
                mm = re.search(r"layers\.(\d+)\.", name)
                if mm:
                    idx.add(int(mm.group(1)))
        return sorted(idx)

    from peft import LoraConfig, get_peft_model

    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    read_band = list(range(12, L))

    def build_lc():
        return LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=targets, layers_to_transform=list(read_band),
            layers_pattern="layers",
        )

    # ---- control: STOCK model (no funnel) -----------------------------------
    import copy
    stock = copy.deepcopy(model)
    p_stock = get_peft_model(stock, build_lc())
    got_stock = adapted_layer_indices(p_stock)
    print(f"[peft-probe] STOCK adapted layers: {got_stock[:5]}...{got_stock[-3:]} "
          f"n={len(got_stock)}", flush=True)

    # ---- treatment: funnel-injected model ----------------------------------
    sys.path.insert(0, "/apdcephfs_wzz/share_303419932/pighzliu_code/Mixture-of-Memory")
    from scripts.train_qwen_bottleneck_continued import inject_bottleneck

    funnel = copy.deepcopy(model)
    inject_bottleneck(funnel, 12, 16, torch.float32)
    wrapped = funnel.model.layers[12]
    print(f"[peft-probe] layers[12] is now {type(wrapped).__name__}", flush=True)
    names12 = [n for n, _ in funnel.named_modules() if ".layers.12." in n][:6]
    print(f"[peft-probe] sample layer-12 module names: {names12}", flush=True)

    p_funnel = get_peft_model(funnel, build_lc())
    got_funnel = adapted_layer_indices(p_funnel)
    print(f"[peft-probe] FUNNEL adapted layers: {got_funnel[:5]}...{got_funnel[-3:]} "
          f"n={len(got_funnel)}", flush=True)

    # count adapted modules inside layer 12 specifically, on each arm
    def n_lora_in_layer(m, li):
        return sum(1 for n, _ in m.named_modules()
                   if n.endswith("lora_A") and f"layers.{li}." in n)

    s12, f12 = n_lora_in_layer(p_stock, 12), n_lora_in_layer(p_funnel, 12)
    s13, f13 = n_lora_in_layer(p_stock, 13), n_lora_in_layer(p_funnel, 13)

    out = {
        "stock_adapted_layers": got_stock,
        "funnel_adapted_layers": got_funnel,
        "stock_n_adapted_layers": len(got_stock),
        "funnel_n_adapted_layers": len(got_funnel),
        "expected_read_band": read_band,
        "stock_matches_expected": got_stock == read_band,
        "funnel_matches_expected": got_funnel == read_band,
        "n_lora_modules_in_layer12": {"stock": s12, "funnel": f12},
        "n_lora_modules_in_layer13": {"stock": s13, "funnel": f13},
        "layer12_module_type_after_inject": type(wrapped).__name__,
        "raised_an_error": False,
    }
    print(json.dumps(out, indent=2), flush=True)

    verdict = []
    if got_stock == read_band:
        verdict.append("CONTROL OK: on the stock model PEFT adapts exactly layers 12..35.")
    else:
        verdict.append("CONTROL BROKEN: even the stock arm did not match; probe is invalid.")
    if got_funnel != read_band:
        verdict.append(
            f"FUNNEL MISMATCH: adapted {len(got_funnel)} layers, expected {len(read_band)}. "
            "Loading the Read-LoRA onto a funnel model attaches a DIFFERENT module set.")
    else:
        verdict.append("FUNNEL matches the stock module set at the layer-index level.")
    if s12 != f12:
        verdict.append(
            f"LAYER-12 DIVERGENCE: stock got {s12} lora modules at layer 12, funnel got {f12}. "
            "This happened WITHOUT raising -- a silent wrong-model failure.")
    else:
        verdict.append(f"layer 12 got {s12} lora modules on BOTH arms.")
    for v in verdict:
        print("[peft-probe] " + v, flush=True)

    with open("/root/b01_peft_funnel_probe.json", "w") as f:
        json.dump({"result": out, "verdict": verdict}, f, indent=2)
    print("[peft-probe] wrote /root/b01_peft_funnel_probe.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
