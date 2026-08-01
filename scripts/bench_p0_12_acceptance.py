#!/usr/bin/env python
"""P0.12 ACCEPTANCE — strict-provenance superset of the depth-replay latency bench.

This is the "加强版" companion to ``scripts/bench_p0_12_depth_replay.py``. It runs
the SAME single-variable isolation (resume_j=0 vs resume_j=12, flagship LoRA held
fixed, identical top-12 retrieved pack) and reproduces the byte-identical packed
token ids (``packed_ids_sha256`` must equal the existing 16k set's
``f7fc76177dd60a664f3b37a3934c1efb83c5a9d6bcaf3804697c9f684aafdc13``), but ALSO
emits the acceptance fields the lumped harness never collected:

  (1) backbone ``name_or_path`` + sha256 of a handful of key weight tensors
      (embed / layer0.q_proj / layer{L-1}.mlp.down_proj / norm / lm_head).
  (2) the actual mounted LoRA modules enumerated (module names carrying a
      non-empty ``lora_A`` / ``lora_B`` ModuleDict).
  (4) a full version string: driver / torch / cuda / cuda-driver / gpu-driver /
      transformers / peft / git-commit / python.
  (6) the read path split into TWO separately-timed sub-kernels — the upper-layer
      transformer forward (``layers[j:L]`` over the pack) vs the final RMSNorm +
      LM-head projection. This is the prerequisite for upgrading the "read-path"
      wording to a "kernel" claim.
  (7) OUTPUT-CONSISTENCY between the two arms on the IDENTICAL pack: does feeding a
      DIFFERENT layer-12 input (j=0 global-causal h_0 vs j=12 chunk-local h_12)
      actually change the final output? Reports cosine similarity, top-1 agreement
      rate, and KL divergence over the query-tail logits + a step-by-step greedy
      decode. This directly answers the core P0.12 caveat.

Two modes (one process each; parallelise the 6 timing procs + 1 consistency proc
across the 8 free GPUs of .82):

  * ``--mode timing --resume_j {0,12} --rep_id R --output ...`` — one arm, one
    process-repeat: 3 warmup + 20 timed reads (single-sync ``qc.read`` series,
    directly comparable to the existing set) + a split-timed series for (6),
    plus provenance (1)(2)(4). Emits one JSON per (arm, rep).
  * ``--mode consistency --output ...`` — loads the model + LoRA ONCE, builds the
    identical pack, runs BOTH arms' read on it and compares the logits (7). Run
    exactly once.

Nothing here mutates ``bench_p0_12_depth_replay.py``; it imports the pure helpers
from it (``_summ`` / ``_pctile`` / ``_sha256_file``) and the VERBATIM bm25 selector
+ length parser from ``bench_qcmem_vs_fullctx.py`` so the retrieval / packing is
bit-identical to both harnesses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJECT_ROOT,
          os.path.join(PROJECT_ROOT, "scripts"),
          os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if p not in sys.path:
        sys.path.insert(0, p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.memory.qcmem import QCMemModel  # noqa: E402
# Reuse the EXACT bm25 selector + length parser so retrieval/packing is
# bit-identical to bench_qcmem_vs_fullctx.py AND bench_p0_12_depth_replay.py.
from bench_qcmem_vs_fullctx import _bm25_scores, parse_length  # noqa: E402
# Reuse the pure timing/percentile/file-hash helpers from the base driver so the
# summary schema (median/p10/p90/min/max/mean/raw) is identical.
from bench_p0_12_depth_replay import _summ, _sha256_file  # noqa: E402

EXPECTED_PACK_SHA = "f7fc76177dd60a664f3b37a3934c1efb83c5a9d6bcaf3804697c9f684aafdc13"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


# --------------------------------------------------------------------------- #
# (1) key-tensor provenance
# --------------------------------------------------------------------------- #
def _tensor_sha256(t: torch.Tensor) -> str:
    """Deterministic sha256 of a weight tensor.

    bf16 -> fp32 upcast is lossless and deterministic, so identical bf16 weights
    hash identically (numpy has no native bfloat16 dtype, hence the upcast)."""
    arr = t.detach().to(torch.float32).cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _base_weight(module):
    """Return a LoRA-wrapped or plain Linear's underlying backbone weight."""
    base = getattr(module, "base_layer", module)
    return getattr(base, "weight", None)


def _backbone_provenance(qc: QCMemModel, model_path: str) -> dict:
    L = qc.num_layers
    key = {}
    try:
        key["embed_tokens.weight"] = _tensor_sha256(qc.embed_tokens.weight)
    except Exception as e:  # pragma: no cover
        key["embed_tokens.weight"] = f"ERR:{e}"
    try:
        key["norm.weight"] = _tensor_sha256(qc.norm.weight)
    except Exception as e:  # pragma: no cover
        key["norm.weight"] = f"ERR:{e}"
    try:
        key["lm_head.weight"] = _tensor_sha256(qc.lm_head.weight)
    except Exception as e:  # pragma: no cover
        key["lm_head.weight"] = f"ERR:{e}"
    try:
        w0 = _base_weight(qc.layers[0].self_attn.q_proj)
        key["layers.0.self_attn.q_proj.weight"] = _tensor_sha256(w0)
    except Exception as e:  # pragma: no cover
        key["layers.0.self_attn.q_proj.weight"] = f"ERR:{e}"
    try:
        wl = _base_weight(qc.layers[L - 1].mlp.down_proj)
        key[f"layers.{L - 1}.mlp.down_proj.weight"] = _tensor_sha256(wl)
    except Exception as e:  # pragma: no cover
        key[f"layers.{L - 1}.mlp.down_proj.weight"] = f"ERR:{e}"
    return {
        "cli_model_path": model_path,
        "config_name_or_path": getattr(qc.config, "_name_or_path", None),
        "config_class": type(qc.config).__name__,
        "model_type": getattr(qc.config, "model_type", None),
        "num_hidden_layers": qc.num_layers,
        "hidden_size": qc.hidden_size,
        "vocab_size": int(qc.config.vocab_size),
        "torch_dtype": str(qc.dtype),
        "tie_word_embeddings": bool(getattr(qc.config, "tie_word_embeddings", False)),
        "key_tensor_sha256": key,
    }


# --------------------------------------------------------------------------- #
# (2) mounted LoRA module enumeration
# --------------------------------------------------------------------------- #
def _lora_modules(model) -> dict:
    names = []
    active_adapters = set()
    for name, mod in model.named_modules():
        la = getattr(mod, "lora_A", None)
        lb = getattr(mod, "lora_B", None)
        has_a = isinstance(la, torch.nn.ModuleDict) and len(la) > 0
        has_b = isinstance(lb, torch.nn.ModuleDict) and len(lb) > 0
        if has_a or has_b:
            names.append(name)
            if isinstance(la, torch.nn.ModuleDict):
                active_adapters.update(la.keys())
    return {
        "count": len(names),
        "adapter_names": sorted(active_adapters),
        "modules": names,
    }


# --------------------------------------------------------------------------- #
# (4) full version string
# --------------------------------------------------------------------------- #
def _versions(device) -> dict:
    def _shell(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=15).stdout.strip() or None
        except Exception:  # pragma: no cover
            return None

    try:
        import transformers
        tv = transformers.__version__
    except Exception:  # pragma: no cover
        tv = None
    try:
        import peft
        pv = peft.__version__
    except Exception:  # pragma: no cover
        pv = None
    cuda_driver = None
    try:
        cuda_driver = torch._C._cuda_getDriverVersion()
    except Exception:  # pragma: no cover
        cuda_driver = None
    return {
        "driver_script": os.path.basename(__file__),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_driver_version": cuda_driver,
        "gpu_driver_version": _shell(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"]),
        "transformers": tv,
        "peft": pv,
        "git_commit": _shell(["git", "-C", PROJECT_ROOT, "rev-parse", "HEAD"]),
        "git_commit_short": _shell(
            ["git", "-C", PROJECT_ROOT, "rev-parse", "--short", "HEAD"]),
        "python": platform.python_version(),
        "gpu": (torch.cuda.get_device_name(device)
                if device.type == "cuda" else None),
    }


# --------------------------------------------------------------------------- #
# model + LoRA load (mirrors bench_p0_12_depth_replay ordering EXACTLY so the
# torch RNG state at the randint() pack draw is identical -> same pack sha)
# --------------------------------------------------------------------------- #
def _load(model_path, dtype, attn_impl, device, lora_adapter):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    lora_sha256 = None
    lora_layers = None
    if lora_adapter:
        from peft import PeftModel
        print(f"[acc] loading LoRA adapter: {lora_adapter}", flush=True)
        peft_model = PeftModel.from_pretrained(model, lora_adapter).eval()
        model = peft_model.base_model.model
        adapter_file = os.path.join(lora_adapter, "adapter_model.safetensors")
        if os.path.exists(adapter_file):
            lora_sha256 = _sha256_file(adapter_file)
        cfg_file = os.path.join(lora_adapter, "adapter_config.json")
        if os.path.exists(cfg_file):
            with open(cfg_file) as f:
                lora_layers = json.load(f).get("layers_to_transform")
    return tokenizer, model, lora_sha256, lora_layers


def _build_pack(tokenizer, vocab, args, device):
    """Reproduce the byte-identical top-k pack. MUST be called with the same torch
    RNG state as bench_p0_12_depth_replay (seed set, nothing else drawing CUDA
    RNG in between). Returns everything needed to write/read + provenance."""
    L = parse_length(args.length)
    input_ids = torch.randint(0, vocab, (1, L), device=device)
    tokens = input_ids[0]
    chunks = list(tokens.split(args.chunk_size))
    if len(chunks) < args.topk + 1:
        raise SystemExit(
            f"length {args.length} gives {len(chunks)} chunks; need >= topk+1")
    context_chunks = chunks[:-1]
    query_chunk = chunks[-1]
    n_ctx = len(context_chunks)
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = int(tokens[0].item())
    query_tok_list = query_chunk.tolist()
    docs = [c.tolist() for c in context_chunks]
    scores = _bm25_scores(docs, query_tok_list)
    k = max(0, int(args.topk))
    order = sorted(range(n_ctx), key=lambda i: scores[i], reverse=True)
    sel_idx = sorted(order[:k])
    packed_ids = [int(bos_id)]
    for i in sel_idx:
        packed_ids.extend(context_chunks[i].tolist())
    packed_ids.extend(query_tok_list)
    packed_ids_sha256 = hashlib.sha256(
        b",".join(str(t).encode() for t in packed_ids)).hexdigest()
    return {
        "L": L, "context_chunks": context_chunks, "query_tok_list": query_tok_list,
        "n_ctx": n_ctx, "bos_id": int(bos_id), "sel_idx": sel_idx,
        "packed_ids_sha256": packed_ids_sha256,
        "pack_token_count": len(packed_ids),
    }


def _check_pack_sha(pack, args):
    got = pack["packed_ids_sha256"]
    match = (got == args.expected_pack_sha)
    print(f"[acc] packed_ids_sha256 = {got}")
    print(f"[acc] expected          = {args.expected_pack_sha}")
    print(f"[acc] pack_sha_match    = {match}  "
          f"pack_token_count={pack['pack_token_count']} sel_idx={pack['sel_idx']}",
          flush=True)
    if not match and not args.allow_sha_mismatch:
        print("[acc] FATAL: pack sha256 mismatch -> refusing to emit results "
              "(re-run with --allow_sha_mismatch only for debugging).", flush=True)
        sys.exit(2)
    return match


# --------------------------------------------------------------------------- #
# (6) split-timed read: upper-layer forward vs final-norm + lm-head
# --------------------------------------------------------------------------- #
def _read_split_timed(qc: QCMemModel, sink_hj, selected_hj_list, query_hj):
    """Faithful replica of read_core (top_prepay_b==0, block_diagonal==False) with
    the pack-prep / upper-layer-forward / (norm+lm_head) sub-kernels timed
    separately. Returns (logits, t_prep, t_upper, t_head)."""
    _sync(); tp0 = time.perf_counter()
    pieces = []
    if sink_hj is not None:
        pieces.append(sink_hj)
    for h in selected_hj_list:
        if h is not None and h.shape[1] > 0:
            pieces.append(h)
    pieces.append(query_hj)
    packed = torch.cat(pieces, dim=1)
    H = packed.shape[1]
    positions = torch.arange(H, device=qc.device).unsqueeze(0)
    causal_mask, position_embeddings = qc._make_mask_and_rope(packed, positions)
    _sync(); tp1 = time.perf_counter()
    hidden = qc._run_layers(
        packed, slice(qc.resume_j, qc.num_layers),
        causal_mask, positions, position_embeddings)
    _sync(); tp2 = time.perf_counter()
    normed = qc.norm(hidden)
    logits = qc.lm_head(normed)
    _sync(); tp3 = time.perf_counter()
    return logits, (tp1 - tp0), (tp2 - tp1), (tp3 - tp2)


# --------------------------------------------------------------------------- #
# TIMING mode: one arm, one process-repeat
# --------------------------------------------------------------------------- #
def run_timing(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L_layers = int(model.config.num_hidden_layers)
    vocab = int(model.config.vocab_size)
    if not (0 <= args.resume_j <= L_layers):
        raise SystemExit(f"--resume_j must be in [0,{L_layers}]; got {args.resume_j}")
    qc = QCMemModel(model, resume_j=args.resume_j)
    print(f"[acc] backbone: {L_layers} layers hidden={qc.hidden_size} vocab={vocab}; "
          f"replay layers[{args.resume_j}:{L_layers}] = {L_layers - args.resume_j}",
          flush=True)

    # ---- pack draw MUST follow the load with no intervening CUDA RNG ----
    pack = _build_pack(tokenizer, vocab, args, device)
    _check_pack_sha(pack, args)

    # ---- provenance (no RNG; safe to collect after the pack draw) ----
    prov_backbone = _backbone_provenance(qc, args.model_path)      # (1)
    prov_lora = _lora_modules(model)                               # (2)
    prov_versions = _versions(device)                             # (4)
    print(f"[acc] LoRA modules mounted: {prov_lora['count']} "
          f"(adapters={prov_lora['adapter_names']})", flush=True)

    context_chunks = pack["context_chunks"]
    query_tok_list = pack["query_tok_list"]
    sel_idx = pack["sel_idx"]
    bos_id = pack["bos_id"]

    # ---- offline ingest: write sink + all ctx, keep selected ----
    torch.cuda.reset_peak_memory_stats()
    _sync(); tw0 = time.perf_counter()
    with torch.inference_mode():
        sink_hj = qc.write_chunk([bos_id])
        all_hj = [qc.write_chunk(c) for c in context_chunks]
    _sync()
    write_all_s = time.perf_counter() - tw0
    selected_hj = [all_hj[i] for i in sel_idx]

    # ---- (A) primary read_s loop: single-sync qc.read (comparable series) ----
    qwrite_times, read_times = [], []
    peak = 0.0
    read_len = None
    for it in range(args.warmup + args.n_repeat):
        torch.cuda.reset_peak_memory_stats()
        _sync(); t0 = time.perf_counter()
        with torch.inference_mode():
            q_hj = qc.write_chunk(query_tok_list)
        _sync(); t1 = time.perf_counter()
        with torch.inference_mode():
            logits = qc.read(sink_hj, selected_hj, q_hj)
        _ = int(logits[0, -1].float().argmax().item())
        _sync(); t2 = time.perf_counter()
        read_len = (1 + sum(int(h.shape[1]) for h in selected_hj)
                    + int(q_hj.shape[1]))
        if it >= args.warmup:
            qwrite_times.append(t1 - t0)
            read_times.append(t2 - t1)
            peak = max(peak, _peak_gb())
        del q_hj, logits
        torch.cuda.empty_cache()

    # ---- (B) split-timed read loop (6): upper-forward vs norm+lm_head ----
    prep_times, upper_times, head_times, split_total = [], [], [], []
    with torch.inference_mode():
        q_hj_fixed = qc.write_chunk(query_tok_list)
    for it in range(args.warmup + args.n_repeat):
        with torch.inference_mode():
            logits, tprep, tup, thead = _read_split_timed(
                qc, sink_hj, selected_hj, q_hj_fixed)
        del logits
        if it >= args.warmup:
            prep_times.append(tprep)
            upper_times.append(tup)
            head_times.append(thead)
            split_total.append(tprep + tup + thead)
        torch.cuda.empty_cache()
    del q_hj_fixed

    read_summ = _summ(read_times)
    upper_summ = _summ(upper_times)
    head_summ = _summ(head_times)
    print(f"[acc] read_s med={read_summ['median']*1e3:.2f}ms | "
          f"(6) upper_forward med={upper_summ['median']*1e3:.2f}ms "
          f"norm+lm_head med={head_summ['median']*1e3:.2f}ms | "
          f"peak={peak:.2f}GB read_len={read_len}", flush=True)

    result = {
        "mode": "timing",
        "arm_name": args.arm_name,
        "rep_id": args.rep_id,
        "config": {
            "model_path": args.model_path,
            "resume_j": args.resume_j,
            "layers_replayed_at_read": L_layers - args.resume_j,
            "chunk_size": args.chunk_size,
            "topk": args.topk,
            "length": args.length,
            "L_tokens": pack["L"],
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
            "seed": args.seed,
            "n_repeat": args.n_repeat,
            "warmup": args.warmup,
            "num_layers": L_layers,
            "vocab_size": vocab,
            "lora_adapter": args.lora_adapter or None,
            "lora_sha256": lora_sha256,
            "lora_layers_to_transform": lora_layers,
        },
        "pack": {
            "read_len": read_len,
            "pack_token_count": pack["pack_token_count"],
            "packed_ids_sha256": pack["packed_ids_sha256"],
            "expected_packed_ids_sha256": args.expected_pack_sha,
            "pack_sha_match": pack["packed_ids_sha256"] == args.expected_pack_sha,
            "selected_idx": sel_idx,
            "n_ctx_chunks": pack["n_ctx"],
        },
        "provenance": {
            "backbone": prov_backbone,      # (1)
            "lora": prov_lora,              # (2)
            "versions": prov_versions,      # (4)
        },
        "timing": {
            "read_s": read_summ,            # PRIMARY single-sync read (comparable)
            "qwrite_s": _summ(qwrite_times),
            "write_all_s_once": write_all_s,
            "read_split_s": {               # (6)
                "prep_s": _summ(prep_times),
                "upper_forward_s": upper_summ,
                "norm_lm_head_s": head_summ,
                "split_total_s": _summ(split_total),
            },
        },
        "peak_gb": peak,
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": prov_versions.get("gpu"),
            "python": platform.python_version(),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[acc] wrote timing JSON -> {args.output}", flush=True)


# --------------------------------------------------------------------------- #
# CONSISTENCY mode (7): j=0 vs j=12 on the IDENTICAL pack
# --------------------------------------------------------------------------- #
def _kl(logits_p, logits_q):
    """mean KL(P||Q) over positions; logits_[*] are [N, V]. P=softmax(p)."""
    logp = torch.log_softmax(logits_p, dim=-1)
    logq = torch.log_softmax(logits_q, dim=-1)
    p = logp.exp()
    return float((p * (logp - logq)).sum(dim=-1).mean().item())


def _cosine_rows(a, b):
    """mean & min row-wise cosine similarity between [N, V] logit tensors."""
    cs = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return float(cs.mean().item()), float(cs.min().item())


def run_consistency(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L_layers = int(model.config.num_hidden_layers)
    vocab = int(model.config.vocab_size)
    # Two wrappers over the SAME model (no weight mutation, no RNG).
    qc0 = QCMemModel(model, resume_j=0)
    qc12 = QCMemModel(model, resume_j=12)

    pack = _build_pack(tokenizer, vocab, args, device)
    _check_pack_sha(pack, args)

    prov_backbone = _backbone_provenance(qc0, args.model_path)
    prov_lora = _lora_modules(model)
    prov_versions = _versions(device)

    context_chunks = pack["context_chunks"]
    query_tok_list = pack["query_tok_list"]
    sel_idx = pack["sel_idx"]
    bos_id = pack["bos_id"]
    T_q = len(query_tok_list)

    def _read_full(qc):
        with torch.inference_mode():
            sink = qc.write_chunk([bos_id])
            sel = [qc.write_chunk(context_chunks[i]) for i in sel_idx]
            q = qc.write_chunk(query_tok_list)
            logits = qc.read(sink, sel, q)  # [1, H, V]
        return logits, sink, sel

    logits0, sink0, sel0 = _read_full(qc0)
    logits12, sink12, sel12 = _read_full(qc12)
    H0, H12 = int(logits0.shape[1]), int(logits12.shape[1])
    assert H0 == H12, f"pack length mismatch {H0} vs {H12}"

    # ---- query-tail (teacher-forced) comparison over the last T_q positions ----
    tail0 = logits0[0, -T_q:, :].float()   # [T_q, V]
    tail12 = logits12[0, -T_q:, :].float()
    cos_mean, cos_min = _cosine_rows(tail0, tail12)
    argmax0 = tail0.argmax(dim=-1)
    argmax12 = tail12.argmax(dim=-1)
    top1_agree = float((argmax0 == argmax12).float().mean().item())
    kl_0_12 = _kl(tail0, tail12)
    kl_12_0 = _kl(tail12, tail0)

    # ---- last-position (next-token) single comparison ----
    lp0 = logits0[0, -1, :].float()
    lp12 = logits12[0, -1, :].float()
    lp_cos = float(torch.nn.functional.cosine_similarity(
        lp0.unsqueeze(0), lp12.unsqueeze(0), dim=-1).item())
    lp_match = bool(int(lp0.argmax().item()) == int(lp12.argmax().item()))
    lp_kl = _kl(lp0.unsqueeze(0), lp12.unsqueeze(0))

    query_tail_summary = {
        "n_positions": T_q,
        "cosine_mean": cos_mean,
        "cosine_min": cos_min,
        "top1_agreement_rate": top1_agree,
        "kl_mean_0to12": kl_0_12,
        "kl_mean_12to0": kl_12_0,
    }
    last_pos_summary = {
        "cosine": lp_cos,
        "top1_match": lp_match,
        "kl_0to12": lp_kl,
    }
    print(f"[acc][7] query-tail (T_q={T_q}): cos_mean={cos_mean:.4f} "
          f"cos_min={cos_min:.4f} top1_agree={top1_agree:.4f} "
          f"KL(0||12)={kl_0_12:.4f} KL(12||0)={kl_12_0:.4f}", flush=True)
    print(f"[acc][7] last-pos: cos={lp_cos:.4f} top1_match={lp_match} "
          f"KL={lp_kl:.4f}", flush=True)

    del logits0, logits12, tail0, tail12
    torch.cuda.empty_cache()

    # ---- step-by-step greedy decode agreement (advance by arm-0 argmax) ----
    per_step = []
    matches = 0
    cos_acc = 0.0
    kl_acc = 0.0
    query_ids = list(query_tok_list)
    with torch.inference_mode():
        for step in range(args.n_decode):
            q0 = qc0.write_chunk(query_ids)
            l0 = qc0.read(sink0, sel0, q0)[0, -1, :].float()
            q12 = qc12.write_chunk(query_ids)
            l12 = qc12.read(sink12, sel12, q12)[0, -1, :].float()
            t0 = int(l0.argmax().item())
            t12 = int(l12.argmax().item())
            cos = float(torch.nn.functional.cosine_similarity(
                l0.unsqueeze(0), l12.unsqueeze(0), dim=-1).item())
            kl = _kl(l0.unsqueeze(0), l12.unsqueeze(0))
            match = (t0 == t12)
            matches += int(match)
            cos_acc += cos
            kl_acc += kl
            per_step.append({"step": step, "tok_arm0": t0, "tok_arm12": t12,
                             "match": match, "cosine": cos, "kl_0to12": kl})
            query_ids.append(t0)  # advance both arms on the SAME (arm-0) token
            del q0, l0, q12, l12
    nd = max(1, args.n_decode)
    decode_summary = {
        "n_decode": args.n_decode,
        "top1_agreement_rate": matches / nd,
        "cosine_mean": cos_acc / nd,
        "kl_mean_0to12": kl_acc / nd,
        "per_step": per_step,
    }
    print(f"[acc][7] decode ({args.n_decode} steps): "
          f"top1_agree={decode_summary['top1_agreement_rate']:.4f} "
          f"cos_mean={decode_summary['cosine_mean']:.4f} "
          f"kl_mean={decode_summary['kl_mean_0to12']:.4f}", flush=True)

    result = {
        "mode": "consistency",
        "config": {
            "model_path": args.model_path,
            "arms": {"A": {"resume_j": 0}, "B": {"resume_j": 12}},
            "chunk_size": args.chunk_size,
            "topk": args.topk,
            "length": args.length,
            "L_tokens": pack["L"],
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
            "seed": args.seed,
            "n_decode": args.n_decode,
            "num_layers": L_layers,
            "vocab_size": vocab,
            "lora_adapter": args.lora_adapter or None,
            "lora_sha256": lora_sha256,
            "lora_layers_to_transform": lora_layers,
        },
        "pack": {
            "pack_token_count": pack["pack_token_count"],
            "packed_ids_sha256": pack["packed_ids_sha256"],
            "expected_packed_ids_sha256": args.expected_pack_sha,
            "pack_sha_match": pack["packed_ids_sha256"] == args.expected_pack_sha,
            "selected_idx": sel_idx,
            "n_ctx_chunks": pack["n_ctx"],
            "query_len": T_q,
        },
        "provenance": {
            "backbone": prov_backbone,
            "lora": prov_lora,
            "versions": prov_versions,
        },
        "consistency": {                    # (7)
            "note": ("j=0 feeds global-causal h_0 into layer 12; j=12 feeds "
                     "chunk-local h_12. This measures whether that DIFFERENT "
                     "layer-12 input changes the final output on the SAME pack."),
            "query_tail": query_tail_summary,
            "last_position": last_pos_summary,
            "decode": decode_summary,
        },
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": prov_versions.get("gpu"),
            "python": platform.python_version(),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[acc] wrote consistency JSON -> {args.output}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="P0.12 acceptance bench (superset)")
    ap.add_argument("--mode", choices=["timing", "consistency"], required=True)
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--resume_j", type=int, default=0,
                    help="timing-mode only: read resume-start layer j (0 or 12).")
    ap.add_argument("--lora_adapter", type=str, default="")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--length", type=str, default="16k")
    ap.add_argument("--n_repeat", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--n_decode", type=int, default=16,
                    help="consistency-mode greedy-decode agreement steps.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm_name", type=str, default="")
    ap.add_argument("--rep_id", type=int, default=0)
    ap.add_argument("--expected_pack_sha", type=str, default=EXPECTED_PACK_SHA)
    ap.add_argument("--allow_sha_mismatch", action="store_true",
                    help="DEBUG ONLY: emit results even if the pack sha mismatches.")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print("=" * 78)
    print(f"P0.12 ACCEPTANCE :: mode={args.mode} arm={args.arm_name} "
          f"rep={args.rep_id} resume_j={args.resume_j}")
    print(f"  model_path={args.model_path} lora={args.lora_adapter or None}")
    print(f"  length={args.length} chunk_size={args.chunk_size} topk={args.topk} "
          f"dtype={dtype} attn={args.attn_impl} device={device} seed={args.seed}")
    print("=" * 78, flush=True)

    if args.mode == "timing":
        run_timing(args, device, dtype)
    else:
        run_consistency(args, device, dtype)


if __name__ == "__main__":
    main()
