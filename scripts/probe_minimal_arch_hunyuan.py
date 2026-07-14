#!/usr/bin/env python3
"""Layer-truncation probe for the "front-j + NTP head" minimal-architecture
hypothesis, ported to **Hunyuan-A13B (MoE)**.

This is the Hunyuan sibling of scripts/probe_minimal_arch.py (Llama). Same idea
(Direction 4 / QCMEM_AUTONOMOUS_AGENDA.md §1 "understand-then-generate division
of labour"): our layer-wise probing shows the semantic state saturates by a mid
layer j. The MINIMAL-ARCH hypothesis asks whether the middle layers are therefore
REDUNDANT -- i.e. can a smaller transformer of "front j layers (understand) + a
few (k=1..2) tail layers that only do next-token prediction (NTP)" match the full
model?  This CHEAP, TRAINING-FREE probe takes the *pretrained* Hunyuan-A13B and
does "layer-skip" forward passes, measuring next-token perplexity / accuracy on
held-out text for three arms per truncation point j (L = num_hidden_layers = 32):

  (a) FULL          : all L layers  ->  norm -> lm_head            (j-invariant baseline)
  (b) FRONT-j+TAILk : layers[0:j] then the last k layers           (the minimal-arch
                      layers[L-k:L] (SKIP the middle [j:L-k])        surrogate; k in --ntp_ks)
                      -> norm -> lm_head
  (c) FRONT-j direct: layers[0:j] -> norm -> lm_head               (logit-lens @ j;
                                                                     no NTP layer at all)

Interpretation
--------------
  * (b) ppl ~= (a) ppl  for small j  ->  the middle layers ARE largely redundant;
    "front-j + k NTP layers" preserves LM quality  ->  hypothesis SUPPORTED.
  * (b) ppl  >>  (a) ppl              ->  the skipped layers do real, non-removable
    work (progressive refinement)  ->  hypothesis FALSIFIED for the truncation route.
  * (c) tells you how much even a single dedicated NTP layer buys over reading the
    layer-j state raw (logit-lens NLL).

HONESTY / LIMITATION (project red-line #2)
------------------------------------------
Truncating an ALREADY-TRAINED model is NOT the same as training a from-scratch
minimal architecture. The kept layers were trained to consume the *outputs of the
layers we skip*; feeding layer-j's state straight into the tail layers is off the
training manifold, so a large ppl jump is an UPPER bound on the damage. A SMALL
(b)-vs-(a) gap is strong positive evidence; a LARGE gap is only SUGGESTIVE against
the hypothesis. Fast feasibility SIGNAL, not the final architectural claim.

Hunyuan-A13B specifics (照搬 scripts/train_hunyuan_a13b_probe2.py):
  * native class transformers.models.hunyuan_v1_moe (model_type set to
    "hunyuan_v1_moe" so the native class -- not trust_remote_code -- is used).
  * HunYuanMoEV1Config.from_pretrained leaves head_dim=None -> we set it (attn does
    head_dim**-0.5 and crashes otherwise).
  * decoder layer's mlp is a MoE; we NEVER hand-build a layer. The manual layer-skip
    forward calls the *native* HunYuanMoEV1DecoderLayer with the exact plumbing
    HunYuanMoEV1Model.forward uses (create_causal_mask with input_embeds= &
    cache_position=, rotary_emb(position_embeddings), position_ids). A built-in
    identity check (`sanity_identity`) proves the manual FULL forward reproduces
    model(input_ids).logits, so any measured gap is real, not a plumbing bug.
  * tie_word_embeddings=True (lm_head.weight IS embed_tokens.weight); readout uses
    model.lm_head(model.model.norm(h)) exactly like ForCausalLM.

80B won't fit one GPU: pass --device_map auto to shard the (bf16) model across all
visible GPUs (accelerate). The manual layer loop is device-aware -- it moves the
running hidden state + mask/rope to each layer's own device -- so it stays correct
under a sharded placement. --smoke_random builds a tiny random HunYuanMoEV1 on CPU
(fp32) and needs NO real weights: it validates the layer-skip plumbing + the FULL
arm bit-identity, the correctness gate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys


# ---------------------------------------------------------------------------
# transformers import guard (dev box: transformers 4.57.6 hard-pins
# huggingface-hub<1.0 but 1.23.0 is installed -> spurious ImportError. On lhz the
# .venv_hy3 has transformers 5.13.1 which accepts hub 1.x, so the first import
# succeeds and this shim never fires.)
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
        # Report a hub version inside old-transformers' pinned (>=0.34.0,<1.0)
        # range so the spurious dependency check passes. The installed hub API we
        # actually touch (config/model build + local forward) is compatible.
        if name.replace("_", "-").lower() == "huggingface-hub":
            return "0.34.1"
        return _o(name)

    _im.version = _clamped
    for _m in [k for k in list(sys.modules) if k == "transformers" or k.startswith("transformers.")]:
        del sys.modules[_m]
    import transformers  # noqa: F401  (retry)


_import_transformers()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe import (  # noqa: E402
    HunYuanMoEV1Config,
    HunYuanMoEV1ForCausalLM,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# config / model construction
# ---------------------------------------------------------------------------
def build_full_config(model_path):
    """Load the native Hunyuan config and patch the from_pretrained gotchas
    (照搬 train_hunyuan_a13b_probe2.build_pruned_config, but keep all 32 layers)."""
    cfg = HunYuanMoEV1Config.from_pretrained(model_path, local_files_only=True)
    cfg.model_type = "hunyuan_v1_moe"  # native class, not trust_remote_code
    if getattr(cfg, "head_dim", None) is None:
        cfg.head_dim = getattr(cfg, "attention_head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
    cfg.use_cache = False
    return cfg


def load_hunyuan(model_path, device, dtype, device_map=""):
    """Load the pretrained Hunyuan-A13B (native fused-expert layout via
    from_pretrained). device_map='auto' shards across visible GPUs (needed: 80B
    bf16 ~160GB > one H200); '' -> single device `device`."""
    cfg = build_full_config(model_path)
    kw = dict(config=cfg, torch_dtype=dtype, low_cpu_mem_usage=True,
              local_files_only=True)
    if device_map:
        kw["device_map"] = device_map
    # transformers 5.13.x dispatches MoE experts through the `@use_experts_implementation`
    # interface; the default (`grouped_mm`) calls torch 2.8-nv's grouped-GEMM CUDA kernel,
    # which asserts `delta % 16 == 0` on the dynamic per-expert token dim and hard-crashes
    # (GroupMMCommon.cuh:51) on our unpadded token counts. Force the officially-supported
    # `eager` path (the native per-expert index_add_ loop, HunYuanMoEV1Experts.forward) —
    # no kernel alignment constraint, numerically equivalent. Passing `experts_implementation`
    # to from_pretrained sets config._experts_implementation before the layers are built.
    # Guarded getattr keeps the tiny-random CPU smoke (older API / direct construction) working.
    try:
        model = HunYuanMoEV1ForCausalLM.from_pretrained(
            model_path, experts_implementation="eager", **kw)
    except TypeError:
        # transformers without the experts_implementation kwarg (e.g. dev-box 4.57): default path.
        model = HunYuanMoEV1ForCausalLM.from_pretrained(model_path, **kw)
    if not device_map:
        model.to(device)
    model.eval()
    return model, cfg


def build_random_tiny(dtype=torch.float32):
    """Tiny random HunYuanMoEV1 for CPU smoke (validates layer-skip logic +
    FULL-arm identity). MoE with a handful of tiny experts."""
    L = 6
    cfg = HunYuanMoEV1Config(
        vocab_size=512, hidden_size=64, num_hidden_layers=L,
        num_attention_heads=8, num_key_value_heads=2, head_dim=8,
        attention_head_dim=8, intermediate_size=64,
        num_experts=4, num_local_experts=4,
        moe_topk=[2] * L, num_shared_expert=[1] * L, moe_intermediate_size=[32] * L,
        max_position_embeddings=256, rms_norm_eps=1e-5,
        use_qk_norm=True, tie_word_embeddings=True, use_cache=False,
    )
    cfg.head_dim = 8
    model = HunYuanMoEV1ForCausalLM(cfg).to(dtype).eval()
    return model, cfg


# ---------------------------------------------------------------------------
# device helpers (support --device_map auto: layers may live on different GPUs)
# ---------------------------------------------------------------------------
def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        for b in module.buffers():
            return b.device
        return torch.device("cpu")


def _to_dev(x, dev):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    if isinstance(x, (tuple, list)):
        return type(x)(_to_dev(t, dev) for t in x)
    return x


# ---------------------------------------------------------------------------
# manual layer-skip forward (bit-identical to HunYuanMoEV1Model.forward plumbing)
# ---------------------------------------------------------------------------
def _prep(model, ids):
    """Return the exact (embed, position_embeddings, causal_mask, position_ids,
    cache_position) that HunYuanMoEV1Model.forward builds, so a manual layer loop
    reproduces model() bit-for-bit."""
    from transformers.masking_utils import create_causal_mask
    base = model.model
    ids = ids.to(_module_device(base.embed_tokens))
    embeds = base.embed_tokens(ids)
    T = ids.shape[1]
    cache_position = torch.arange(T, device=embeds.device)
    pos = cache_position.unsqueeze(0)
    pe = base.rotary_emb(embeds, pos)
    # transformers 5.13.x: kwarg is `inputs_embeds` and there is no `cache_position`
    # param (4.57.x used `input_embeds` + `cache_position`). Try new sig, fall back.
    try:
        cm = create_causal_mask(
            config=model.config, inputs_embeds=embeds, attention_mask=None,
            past_key_values=None, position_ids=pos,
        )
    except TypeError:
        cm = create_causal_mask(
            config=model.config, input_embeds=embeds, attention_mask=None,
            cache_position=cache_position, past_key_values=None, position_ids=pos,
        )
    return embeds, pe, cm, pos, cache_position


def _call_layer(layer, h, pe, cm, pos, cache_position):
    """Call one native HunYuanMoEV1DecoderLayer with the Model.forward plumbing,
    moving the running context onto the layer's own device first (device_map)."""
    dev = _module_device(layer)
    if h.device != dev:
        h = h.to(dev)
    pe_d = _to_dev(pe, dev)
    cm_d = _to_dev(cm, dev)
    pos_d = _to_dev(pos, dev)
    cp_d = _to_dev(cache_position, dev)
    out = layer(h, attention_mask=cm_d, position_ids=pos_d, past_key_values=None,
                use_cache=False, cache_position=cp_d, position_embeddings=pe_d)
    # native decoder layer returns a plain tensor (not a tuple)
    return out[0] if isinstance(out, tuple) else out


def _run_layers(model, h, layer_indices, pe, cm, pos, cache_position):
    base = model.model
    for i in layer_indices:
        h = _call_layer(base.layers[i], h, pe, cm, pos, cache_position)
    return h


@torch.no_grad()
def _readout_metrics(model, h, ids):
    """norm -> lm_head -> next-token NLL(sum) + top1-correct(sum) + n_tok."""
    norm = model.model.norm
    lm_head = model.lm_head
    h = h.to(_module_device(norm))
    # fp32 readout: mid-layer hidden can carry massive activations; a bf16 lm_head
    # matmul (4096->128167 tied embedding) overflows to inf on those -> exploding
    # nll (84/235). Compute norm+projection in fp32 so the logit-lens is numerically
    # honest. (FULL arm is unaffected: hidden[L] is in-distribution for final norm.)
    hn = norm(h).to(_module_device(lm_head)).float()
    logits = torch.nn.functional.linear(hn, lm_head.weight.float())
    tgt = ids[:, 1:].to(logits.device)
    logits = logits[:, :-1]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
    correct = (logits.argmax(-1) == tgt).sum()
    return float(nll), int(correct), int(tgt.numel())


@torch.no_grad()
def probe_batch(model, ids, js, ntp_ks, L):
    """One batch. Full forward ONCE, snapshotting the hidden state after each
    front-j block, then replays only the needed tail-k layers for each (j,k)."""
    embeds, pe, cm, pos, cp = _prep(model, ids)
    js_set = set(js)
    snap = {}
    h = embeds
    if 0 in js_set:
        snap[0] = h
    base = model.model
    for i in range(L):
        h = _call_layer(base.layers[i], h, pe, cm, pos, cp)
        if (i + 1) in js_set:
            snap[i + 1] = h
    h_full = h  # output of all L layers (pre-final-norm)

    acc = {"full": _readout_metrics(model, h_full, ids),
           "front_direct": {}, "front_tail": {}}
    for j in js:
        hj = snap[j]
        # (c) front-j direct
        acc["front_direct"][j] = _readout_metrics(model, hj, ids)
        # (b) front-j + last-k layers (skip the middle [j : L-k])
        for k in ntp_ks:
            tail_start = L - k
            if tail_start < j:
                acc["front_tail"].setdefault(j, {})[k] = None  # tail overlaps front
                continue
            ht = _run_layers(model, hj, range(tail_start, L), pe, cm, pos, cp)
            acc["front_tail"].setdefault(j, {})[k] = _readout_metrics(model, ht, ids)
    return acc


@torch.no_grad()
def sanity_identity(model, ids, L):
    """Manual full forward must reproduce model(input_ids).logits (identity)."""
    embeds, pe, cm, pos, cp = _prep(model, ids)
    h = _run_layers(model, embeds, range(L), pe, cm, pos, cp)
    manual = model.lm_head(
        model.model.norm(h.to(_module_device(model.model.norm)))
        .to(_module_device(model.lm_head))
    ).float()
    ref = model(input_ids=ids, use_cache=False).logits.float()
    manual = manual.to(ref.device)
    return float((manual - ref).abs().max().item())


# ---------------------------------------------------------------------------
# accumulation / finalize
# ---------------------------------------------------------------------------
def _accumulate(dst, src):
    def add(a, b):
        return None if b is None else (a[0] + b[0], a[1] + b[1], a[2] + b[2])
    if dst is None:
        return {
            "full": src["full"],
            "front_direct": dict(src["front_direct"]),
            "front_tail": {j: dict(kd) for j, kd in src["front_tail"].items()},
        }
    dst["full"] = add(dst["full"], src["full"])
    for j, v in src["front_direct"].items():
        dst["front_direct"][j] = add(dst["front_direct"][j], v)
    for j, kd in src["front_tail"].items():
        for k, v in kd.items():
            dst["front_tail"][j][k] = add(dst["front_tail"][j][k], v)
    return dst


def _finalize(t):
    if t is None:
        return None
    nll, correct, n = t
    mean_nll = nll / n
    return {"ppl": round(math.exp(min(mean_nll, 20)), 3),
            "nll": round(mean_nll, 4),
            "acc": round(correct / n, 4)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",
                    default="models/Hunyuan-A13B-Pretrain",
                    help="HF directory of the pretrained Hunyuan-A13B (local_files_only)")
    ap.add_argument("--truncate_js", default="8 10 12 14 16 20 24",
                    help="front-layer counts j (space/comma sep)")
    ap.add_argument("--ntp_ks", default="1 2", help="tail NTP layer counts k")
    ap.add_argument("--val_path", default="data/slimpajama_val_hunyuan.npy",
                    help="(N, seq_len) Hunyuan-tokenized npy held-out text")
    ap.add_argument("--n_examples", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device_map", default="",
                    help="'auto' -> shard the 80B model across visible GPUs "
                         "(required; single H200 cannot hold 160GB bf16). '' -> "
                         "single device --device.")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out_json", default="outputs/minimal_arch_probe_hunyuan.json")
    ap.add_argument("--smoke_random", action="store_true",
                    help="tiny random HunYuanMoEV1 on CPU fp32 (no real weights): "
                         "validates layer-skip logic + FULL-arm identity")
    args = ap.parse_args()

    def _ints(s):
        return [int(x) for x in s.replace(",", " ").split() if x.strip()]
    # smoke: tiny L=6, so the A13B default js (8..24) would all be filtered out.
    # Use a set that exercises real front-j+tail-k skipping on the tiny model.
    truncate_js = "1 2 3 4" if args.smoke_random else args.truncate_js
    js = sorted(set(_ints(truncate_js)))
    ntp_ks = sorted(set(_ints(args.ntp_ks)))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.smoke_random:
        # CPU-thread deadlock guard: the native HunYuanMoEV1Moe expert loop
        # (per-expert index_add_ over a python loop) hangs under multi-threaded
        # CPU BLAS on some torch builds; single-thread runs it in ~0.02s. Only the
        # CPU smoke is affected (GPU path unaffected).
        torch.set_num_threads(1)
        print("[smoke_random] tiny random HunYuanMoEV1 on CPU fp32 (threads=1)", flush=True)
        model, cfg = build_random_tiny(torch.float32)
        device = "cpu"
    else:
        assert args.model_path, "provide --model_path (or --smoke_random)"
        print(f"loading Hunyuan-A13B from {args.model_path} "
              f"(dtype={args.dtype} device_map={args.device_map or 'single:'+args.device})",
              flush=True)
        model, cfg = load_hunyuan(args.model_path, args.device, dtype, args.device_map)
        device = args.device

    L = model.config.num_hidden_layers
    js = [j for j in js if 0 <= j <= L]
    print(f"L={L} hidden={cfg.hidden_size} | js={js} ntp_ks={ntp_ks} "
          f"seq_len={args.seq_len} n={args.n_examples}", flush=True)

    # data
    arr = None
    if not args.smoke_random and os.path.exists(args.val_path):
        arr = np.load(args.val_path, mmap_mode="r")
    if arr is not None:
        n = min(args.n_examples, arr.shape[0])
        sl = min(args.seq_len, arr.shape[1])
        tokens = torch.from_numpy(np.asarray(arr[:n, :sl]).astype(np.int64))
        vmax = int(tokens.max())
        assert vmax < cfg.vocab_size, (
            f"val token id {vmax} >= vocab {cfg.vocab_size} -> WRONG tokenizer "
            f"(embedding out-of-range). Re-tokenize with the Hunyuan tokenizer.")
    else:
        if not args.smoke_random:
            print(f"[warn] val_path {args.val_path} missing -> random ids (smoke only, "
                  f"ppl numbers meaningless)", flush=True)
        n_ex = args.n_examples if not args.smoke_random else min(args.n_examples, 8)
        sl = args.seq_len if not args.smoke_random else 16
        tokens = torch.randint(0, cfg.vocab_size, (n_ex, sl))
    print(f"data: {tokens.shape[0]} x {tokens.shape[1]}", flush=True)

    # sanity: manual forward == model()
    ids0 = tokens[: min(args.batch_size, tokens.shape[0])].to(device)
    id_diff = sanity_identity(model, ids0, L)
    ok = id_diff < 1e-1
    print(f"[sanity] max|manual_full - model()| = {id_diff:.4g} "
          f"({'OK identity' if ok else 'WARN plumbing mismatch'})", flush=True)

    # main loop
    accum = None
    for b0 in range(0, tokens.shape[0], args.batch_size):
        ids = tokens[b0:b0 + args.batch_size].to(device)
        accum = _accumulate(accum, probe_batch(model, ids, js, ntp_ks, L))
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    full = _finalize(accum["full"])
    front_direct = {j: _finalize(accum["front_direct"][j]) for j in js}
    front_tail = {j: {k: _finalize(accum["front_tail"][j][k]) for k in ntp_ks} for j in js}

    result = {
        "model": args.model_path if not args.smoke_random else "smoke_random",
        "model_family": "hunyuan_v1_moe",
        "n_layers": L, "hidden_size": cfg.hidden_size, "vocab_size": cfg.vocab_size,
        "seq_len": int(tokens.shape[1]), "n_examples": int(tokens.shape[0]),
        "truncate_js": js, "ntp_ks": ntp_ks,
        "sanity_identity_max_abs_diff": round(id_diff, 6),
        "full_model": full,
        "front_j_direct": front_direct,       # (c) logit-lens @ j
        "front_j_plus_tail_k": front_tail,    # (b) minimal-arch surrogate
        "limitation": ("truncating a trained model != training a from-scratch minimal "
                       "arch; a large gap is an UPPER bound on damage (off-manifold), a "
                       "small gap is strong positive evidence. Fast feasibility signal only."),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    # ---- console report ----
    def _cell(m):
        return f"{m['ppl']:.2f} ({m['acc']:.3f})" if m else "= full/overlap"

    print("\n=== MINIMAL-ARCH LAYER-TRUNCATION PROBE (Hunyuan-A13B) ===")
    print(f"FULL (all {L} layers): ppl={full['ppl']} nll={full['nll']} acc={full['acc']}")
    col_titles = ["front-j direct"] + [f"front-j+tail{k}" for k in ntp_ks]
    hdr = f"{'j':>4} | " + " | ".join(t.rjust(20) for t in col_titles)
    print("\n" + hdr)
    print("-" * len(hdr))
    for j in js:
        cells = [_cell(front_direct[j])]
        cells += [_cell(front_tail[j][k]) for k in ntp_ks]
        print(f"{j:>4} | " + " | ".join(c.rjust(20) for c in cells))

    # verdict (smallest j whose front-j+tail-k1 ppl within 10% of full)
    k1 = ntp_ks[0]
    print(f"\n=== VERDICT (front-j + tail-{k1}; ppl_gap = arm_b/arm_a) ===")
    base = full["ppl"]
    verdict_j = None
    for j in js:
        ft = front_tail[j].get(k1)
        if ft is None:
            continue
        ratio = ft["ppl"] / base if base > 0 else float("inf")
        tag = ""
        if ratio <= 1.10:
            tag = "  <= within 10% of full (REDUNDANT-supportive)"
            if verdict_j is None:
                verdict_j = j
        elif ratio >= 2.0:
            tag = "  >> 2x full (skipped layers do real work)"
        print(f"  j={j:>3}: front{j}+tail{k1} ppl={ft['ppl']:.2f}  ppl_gap={ratio:.2f}x{tag}")
    if verdict_j is not None:
        print(f"  => smallest j with near-full quality (front{verdict_j}+tail{k1}, <=10%): j={verdict_j}\n"
              f"     hypothesis SUPPORTIVE at this scale -> consider from-scratch probe #2.")
    else:
        print("  => NO j reaches within 10% of full with a single tail layer.\n"
              "     hypothesis NOT supported by truncation (skipped mid layers matter).\n"
              "     NOTE: off-manifold upper bound; see 'limitation' in json.")
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
