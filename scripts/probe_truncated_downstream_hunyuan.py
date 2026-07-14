#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Truncated-depth downstream evaluation of **Hunyuan-A13B (MoE)**: the CAUSAL test
of the "division of labour" hypothesis, ported from scripts/probe_truncated_downstream.py
(Qwen3-8B). Same design and honesty caveats -- read that file's header for the full
(i) truncate-and-head == layer-j probing / (ii) faithful-recompute == full-model-top
discussion. Only the model-specific plumbing changes here; the probe logic
(Part A truncated linear probe, Part C native verbalizer readout, the summaries)
is REUSED verbatim from the Qwen downstream module (imported, NOT modified).

What is Hunyuan-specific and re-implemented here (照搬 probe_minimal_arch_hunyuan.py
+ train_hunyuan_a13b_probe2.py):
  1. load_model_hunyuan: native transformers class (model_type="hunyuan_v1_moe",
     not trust_remote_code), the head_dim=None from_pretrained gotcha, device_map
     for sharding the 80B across GPUs, and output_hidden_states=True.
  2. verify_recompute_identity_hunyuan: Part-B numeric identity
     norm(layers[j:L](hidden[j])) == hidden[L], using the exact
     HunYuanMoEV1Model.forward plumbing (create_causal_mask with input_embeds= and
     cache_position=, the native decoder-layer call signature, device-aware moves).
     Reuses the same helpers proven bit-identical in probe_minimal_arch_hunyuan.py.

L = num_hidden_layers = 32, so hidden_states has 33 entries (embed + 32 layers);
hs[0]==embeds, hs[-1]==last_hidden_state (post final-norm), verified on the native
class. The default --depths are set for a 32-layer model; the top layer index
(=n_layers-1) is always appended as the full-model baseline.

Tokenizer: Hunyuan ships its own tiktoken-based HYTokenizer -> AutoTokenizer needs
--trust_remote_code. The GLUE/SuperGLUE text tasks are tokenized with it directly
(they are plain English; token-id range is irrelevant for probing since we read
hidden states, not embeddings-by-id sanity).
"""
import argparse
import json
import os
import sys
import time
from collections import Counter


# ---------------------------------------------------------------------------
# transformers import guard (see probe_minimal_arch_hunyuan.py for rationale:
# dev box transformers 4.57.6 spuriously pins huggingface-hub<1.0).
# ---------------------------------------------------------------------------
def _import_transformers():
    try:
        import transformers  # noqa: F401
        return
    except ImportError as e:
        if "huggingface-hub" not in str(e).lower():
            raise
    import importlib.metadata as _im
    _orig = _im.version

    def _clamped(name, _o=_orig):
        if name.replace("_", "-").lower() == "huggingface-hub":
            return "0.34.1"
        return _o(name)

    _im.version = _clamped
    for _m in [k for k in list(sys.modules) if k == "transformers" or k.startswith("transformers.")]:
        del sys.modules[_m]
    import transformers  # noqa: F401


_import_transformers()

import numpy as np  # noqa: E402
import torch  # noqa: E402

# --- reuse the Qwen downstream + probing scripts (imported, NOT modified) -----
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import probe_linguistic_layerwise as PL          # noqa: E402  (extractors, load_hf)
import probe_truncated_downstream as PT          # noqa: E402  (probe/summary/verbalizer)
# Hunyuan-specific plumbing helpers proven bit-identical in the minimal-arch probe.
from probe_minimal_arch_hunyuan import (          # noqa: E402
    build_full_config,
    _module_device,
    _to_dev,
)

from transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe import (  # noqa: E402
    HunYuanMoEV1ForCausalLM,
)


# ---------------------------------------------------------------------------
# Hunyuan model / tokenizer loading (replaces PL.load_model)
# ---------------------------------------------------------------------------
def load_model_hunyuan(model_path, device, dtype, device_map="", trust_remote_code=True):
    """Load Hunyuan-A13B via the native class with output_hidden_states. Returns
    (model, tok, n_layers) where n_layers = num_hidden_layers + 1 (== number of
    entries in output_hidden_states, matching the Qwen script's convention)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    cfg = build_full_config(model_path)         # native class + head_dim fix
    cfg.output_hidden_states = True
    kw = dict(config=cfg, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
              local_files_only=True, attn_implementation="eager")
    if device_map:
        kw["device_map"] = device_map
    model = HunYuanMoEV1ForCausalLM.from_pretrained(model_path, **kw)
    if not device_map:
        model.to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1     # + embedding layer
    return model, tok, n_layers


# ---------------------------------------------------------------------------
# Part B: verify faithful-recompute identity for Hunyuan (native plumbing)
# ---------------------------------------------------------------------------
@torch.no_grad()
def verify_recompute_identity_hunyuan(model, tok, sentences, device, max_len, depths):
    """For a small batch, recompute the top (post-norm) hidden state from each
    layer-j hidden state via the model's own upper layers + final norm, and
    compare to the directly-produced top state (output_hidden_states[-1]).

    Uses the exact HunYuanMoEV1Model.forward plumbing (input_embeds=, cache_position=,
    native layer signature). ~0 diff => faithful recompute of the upper layers ==
    full model top for every j (j-invariant, == probe@top). Returns {j: max_abs_diff}.
    """
    from transformers.masking_utils import create_causal_mask

    base = getattr(model, "model", model)
    n_layers = model.config.num_hidden_layers      # 32
    enc = tok(sentences, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len, add_special_tokens=False)
    enc = {k: v.to(_module_device(base.embed_tokens)) for k, v in enc.items()}
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states                          # len n_layers+1 ; hs[-1] post-norm
    h_top = hs[-1]
    attn = enc["attention_mask"]                    # (B,T)
    valid = attn.bool().unsqueeze(-1)               # (B,T,1) -- ignore right-pad
    T = enc["input_ids"].shape[1]
    dev0 = hs[0].device
    cache_position = torch.arange(T, device=dev0)
    pos = cache_position.unsqueeze(0)
    pe = base.rotary_emb(hs[0], pos)
    cm = create_causal_mask(
        config=model.config, input_embeds=hs[0], attention_mask=attn,
        cache_position=cache_position, past_key_values=None, position_ids=pos,
    )
    diffs = {}
    for j in depths:
        if j >= n_layers:                           # j == top: nothing to recompute
            diffs[int(j)] = 0.0
            continue
        h = hs[j].clone()
        for i in range(j, n_layers):
            layer = base.layers[i]
            ldev = _module_device(layer)
            h = h.to(ldev)
            o = layer(h, attention_mask=_to_dev(cm, ldev),
                      position_ids=_to_dev(pos, ldev), past_key_values=None,
                      use_cache=False, cache_position=_to_dev(cache_position, ldev),
                      position_embeddings=_to_dev(pe, ldev))
            h = o[0] if isinstance(o, tuple) else o
        h = base.norm(h.to(_module_device(base.norm)))
        d = (h.float() - h_top.float().to(h.device)).abs()
        vmask = valid.to(h.device).expand_as(d)
        diffs[int(j)] = float(d[vmask].max().item())
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Hunyuan-A13B-Pretrain")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device_map", default="",
                    help="'auto' shards the 80B across visible GPUs (one H200 cannot "
                         "hold 160GB bf16). '' -> single --device.")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--trust_remote_code", type=int, default=1,
                    help="needed for Hunyuan's tiktoken-based HYTokenizer")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_train_sent", type=int, default=2000)
    ap.add_argument("--n_dev_sent", type=int, default=1000)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--tasks", default="SST2,WiC,RTE")
    ap.add_argument("--depths", default="4,8,10,12,13,16,20,24,28,32",
                    help="truncation depths j (index into hidden_states; 32=top for A13B).")
    ap.add_argument("--skip_native", action="store_true", help="skip Part C native readout")
    ap.add_argument("--skip_verify", action="store_true", help="skip Part B identity verification")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading Hunyuan-A13B {args.model_path} "
          f"(device_map={args.device_map or 'single:'+args.device})", flush=True)
    model, tok, n_layers = load_model_hunyuan(
        args.model_path, args.device, args.dtype, args.device_map,
        bool(args.trust_remote_code))
    print(f"model loaded: {n_layers} hidden states (embed + {n_layers-1} layers), "
          f"hidden={model.config.hidden_size}", flush=True)
    depths = [j for j in depths if 0 <= j <= n_layers - 1]
    if (n_layers - 1) not in depths:
        depths.append(n_layers - 1)
    depths = sorted(set(depths))

    results = {
        "model": args.model_path,
        "model_family": "hunyuan_v1_moe",
        "n_hidden_states": n_layers,
        "n_transformer_layers": n_layers - 1,
        "depths": depths,
        "design_note": (
            "Part A (truncated probe on hidden[j]) is mathematically identical to "
            "layer-j linear probing. Faithful truncate-and-recompute (option ii) "
            "reproduces the full model top exactly (verified in part_B_recompute_identity), "
            "so it is j-invariant and equals probe@top; the 'increment from the upper "
            "layers' is probe@top - probe@j. Part C (native verbalizer) is the "
            "genuinely new generation-side reference. Ported to Hunyuan-A13B MoE."
        ),
        "tasks": {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # ---- Part B: verify faithful-recompute identity (Hunyuan plumbing) --------
    if not args.skip_verify:
        print(f"\n[{time.strftime('%H:%M:%S')}] === PART B: verify recompute identity (SST2 batch) ===", flush=True)
        try:
            sents = [r["sentence"] for r in PL.load_hf("glue", "sst2", split="validation").select(range(8))]
            diffs = verify_recompute_identity_hunyuan(model, tok, sents, args.device, args.max_len, depths)
            max_over_all = max(diffs.values())
            results["part_B_recompute_identity"] = {
                "max_abs_diff_by_j": diffs,
                "max_abs_diff_over_all_j": max_over_all,
                "is_identity": bool(max_over_all < 1e-1),
                "note": ("norm(layers[j:L](hidden[j])) vs hidden[L], at NON-PAD positions. "
                         "~0 => faithful recompute of upper layers == full model top for "
                         "every j, so option-(ii) downstream acc is j-invariant and == "
                         "probe@top. The only non-trivial recompute would MODIFY hidden[j] "
                         "(compression / QCMem), which needs training and is out of scope."),
            }
            for j, d in diffs.items():
                print(f"    j={j:2d}: max|recompute-top|={d:.4g}", flush=True)
            print(f"  -> identity holds: {results['part_B_recompute_identity']['is_identity']} "
                  f"(max diff {max_over_all:.4g})", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results["part_B_recompute_identity"] = {"error": repr(e)[:300]}
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    # ---- Part A + C per task (reuse the Qwen downstream implementation) --------
    for name in task_names:
        if name not in PT.VERBALIZERS:
            print(f"skip unknown task {name}", flush=True)
            continue
        print(f"\n[{time.strftime('%H:%M:%S')}] === TASK {name} ===", flush=True)
        try:
            te = time.time()
            ftr, ytr, fdv, ydv, dev_rows = PT.build_task_features(
                name, model, tok, args.device, args, n_layers)
            maj = Counter(ydv).most_common(1)[0][1] / len(ydv)
            print(f"  features: n_train={len(ytr)} n_dev={len(ydv)} "
                  f"feat_dim={ftr[depths[0]].shape[1]} extract={time.time()-te:.0f}s "
                  f"majority={maj:.4f}", flush=True)
            acc_by_j = PT.truncated_probe(ftr, ytr, fdv, ydv, depths, C=args.C)
            summ = PT.summarize_truncated(acc_by_j, depths, n_layers)
            summ["majority_baseline"] = round(maj, 4)
            summ["n_train"] = len(ytr)
            summ["n_dev"] = len(ydv)
            summ["equivalent_to_layerwise_probing"] = True
            if not args.skip_native:
                try:
                    nv = PT.native_verbalizer_acc(model, tok, dev_rows, name, args.device,
                                                  args.max_len, args.batch_size)
                    summ["native_generative_readout"] = nv
                    full = summ["full_model_acc"]
                    summ["native_vs_probe"] = {
                        "native_acc": nv["native_acc"],
                        "probe_full_layer_acc": full,
                        "probe_best_mid_acc": summ["best_mid_acc"],
                        "native_minus_probefull": (round(nv["native_acc"] - full, 4)
                                                   if nv["native_acc"] is not None else None),
                    }
                    print(f"  native verbalizer acc={nv['native_acc']} "
                          f"(maj {nv['majority_baseline']}, collision={nv['verbalizer_collision']})",
                          flush=True)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    summ["native_generative_readout"] = {"error": repr(e)[:300]}
            results["tasks"][name] = summ
            print(f"  -> full-model(L{summ['full_model_layer']}) acc={summ['full_model_acc']:.4f}; "
                  f"j*(95%)={summ['j_star_95pct_of_full']}; "
                  f"best_mid=L{summ['best_mid_layer']}({summ['best_mid_acc']}); "
                  f"mid_exceeds_full={summ['mid_exceeds_full_model']}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            results["tasks"][name] = {"error": repr(e)[:300]}
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    results["elapsed_sec"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # ---- console summary ------------------------------------------------------
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {results['elapsed_sec']}s -> {args.out}")
    print("\n=== TRUNCATED DOWNSTREAM (acc-vs-j; full model = top layer) ===")
    hdr = "task     full  " + " ".join(f"j{j:<4d}" for j in depths)
    print(hdr)
    for name, s in results["tasks"].items():
        if "error" in s:
            print(f"{name:8s} ERROR {s['error'][:80]}"); continue
        row = f"{name:8s} {s['full_model_acc']:.3f} "
        row += " ".join(f"{s['acc_by_j'][j]:.3f}" for j in depths)
        print(row)
    print("\n=== j* (earliest depth matching 95% of full model) + mid-vs-top ===")
    for name, s in results["tasks"].items():
        if "error" in s:
            continue
        line = (f"{name:8s} j*={s['j_star_95pct_of_full']:>2d}  "
                f"full=L{s['full_model_layer']}:{s['full_model_acc']:.3f}  "
                f"best_mid=L{s['best_mid_layer']}:{s['best_mid_acc']:.3f}  "
                f"mid_exceeds_full={s['mid_exceeds_full_model']}  "
                f"(full-best_mid={s['full_minus_best_mid']})")
        if "native_generative_readout" in s and "native_acc" in s.get("native_generative_readout", {}):
            line += f"  native={s['native_generative_readout']['native_acc']}"
        print(line)
    if "part_B_recompute_identity" in results and "is_identity" in results["part_B_recompute_identity"]:
        b = results["part_B_recompute_identity"]
        print(f"\n=== (ii) faithful recompute: identity={b['is_identity']} "
              f"(max|recompute-top| over j = {b['max_abs_diff_over_all_j']:.3g}) "
              f"=> faithful recompute == full model top == probe@top, j-invariant ===")


if __name__ == "__main__":
    main()
