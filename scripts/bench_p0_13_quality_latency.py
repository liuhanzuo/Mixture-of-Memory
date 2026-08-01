#!/usr/bin/env python
"""P0.13 — same-pack / same-LoRA / same-examples benchmark quality<->latency loop.

FINAL required model run before submission. Answers: on the SAME retrieved pack,
the SAME flagship rank-32 LoRA and the EXACT SAME RULER examples, how much real
task-quality difference corresponds to the P0.12 model-side read-path speedup
(``resume_j=0`` full 36-layer replay  vs  ``resume_j=12`` upper-24-layer replay
from cached residual ``h12``)?

Two arms differ ONLY in ``resume_j``:
  * Arm A: resume_j=0  + flagship LoRA, replay layers[0:36] over the pack.
  * Arm B: resume_j=12 + SAME LoRA,     replay layers[12:36] from cached h12.

Because the FLAGSHIP RULER selector (``iter_bm25``) is forward-free (pure lexical
BM25 over the raw token ids), the selected context-chunk indices, their order and
the packed token ids are ``resume_j``-INDEPENDENT. So for EACH example we build the
pack ONCE, verify the sha, then run BOTH arms on that identical pack — strict 1:1
pairing by construction. The LoRA (layers 12..35) is active in the read of BOTH
arms (j=0 replays [0:36] ⊇ [12:36]; j=12 replays [12:36]), so ``j=0 + same LoRA``
is a semantically valid RAG-upper-bound arm, not an invalid config.

This is a THIN composition of the two existing, unmodified drivers (nothing here
mutates them):
  * ``scripts/eval_ruler_qcmem.py`` / ``scripts/eval_ruler_mem_space.py`` — the
    RULER sample construction (``_build_sample`` + per-(task,length,i) seed + shard
    filter), the FLAGSHIP selector routing, and the ``_string_match_all_one``
    scorer, all reused VERBATIM so the quality口径 == the headline QCMem RULER口径.
  * ``scripts/eval_qcmem_babilong.py`` — the ``_select_context_chunk_indices``
    (iter_bm25) selection + the QCMem write/read/decode primitives. The per-arm
    generate replica below reproduces the ``can_kv`` branch of ``qcmem_generate``
    VERBATIM (chat_template=False => gen_boundary=None, use_kv_cache=True,
    top_prepay_b=0, block_diagonal=False), only adding phase timers, so the
    prediction/score is bit-identical to the flagship path.
  * ``scripts/bench_p0_12_acceptance.py`` — the strict provenance (backbone key-
    tensor sha, mounted-LoRA enumeration, version string) and the pack-sha idea.

Modes (one process each; the launcher fans them across the 8 GPUs of .82):
  * ``--mode manifest``   — load model+LoRA once, assert backbone/LoRA hashes match
                            the P0.12 acceptance record, dump manifest.json. Aborts
                            (exit 3) on any strict-fix mismatch. Run ONCE first.
  * ``--mode quality``    — one (task,length[,shard]): for each example build the
                            pack once, run BOTH arms, write per-example JSONL + a
                            per-cell summary json (paired 1:1).
  * ``--mode latency``    — one process-repeat of the controlled latency protocol
                            on a FIXED representative pack: 3 warmups + N timed reads
                            per arm (Write / qc.read / decode split), median/p10/p90.
                            Run with >=3 distinct --proc_id for the 3-process test.
  * ``--mode aggregate``  — pure CPU: merge quality shards + latency procs ->
                            summary.json (per-cell + macro), latency.json (per-arm
                            median/p10/p90, direction-consistency), stats
                            (paired bootstrap 95% CI on macro diff, McNemar exact
                            paired test, agreement + failure breakdown).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# RULER task framework (sample construction + scorer) — reused verbatim.
import scripts.eval_ruler_mem_space as ruler  # noqa: E402
# QCMem forward path (selector + write/read/decode primitives) — reused verbatim.
import scripts.eval_qcmem_babilong as qcb  # noqa: E402
# RULER QCMem driver helpers (bare-question extractor + task alias resolver).
from scripts.eval_ruler_qcmem import _bare_question, _resolve_task  # noqa: E402
# Strict provenance helpers (backbone key-tensor sha / LoRA enum / versions).
from bench_p0_12_acceptance import (  # noqa: E402
    _backbone_provenance, _lora_modules, _versions,
)
# Pure timing / hashing helpers (identical summary schema to P0.12).
from bench_p0_12_depth_replay import _summ, _sha256_file  # noqa: E402

QCMemModel = qcb.QCMemModel

# ---- P0.12 acceptance record: the hashes P0.13 must match (strict fixes) ---- #
EXPECTED_LORA_SHA = \
    "dd09cd17457c63578c0f38dab79b287ab5da6e3f14c119aedafec1c34400536f"
# key backbone tensor shas as recorded in bench_results/p0_12_acceptance/*.json
EXPECTED_BACKBONE_KEY_SHA = {
    "layers.0.self_attn.q_proj.weight":
        "7a47839076cfd599146e7d2f1e9fece9dc4797b76ca2f213e055b4eb4f8ef381",
}
EXPECTED_LORA_MODULE_COUNT = 168  # 24 layers (12..35) x 7 target modules


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


# --------------------------------------------------------------------------- #
# model + LoRA load (mirrors eval_ruler_qcmem ordering: base -> PeftModel).
# --------------------------------------------------------------------------- #
def _load(model_path, dtype, attn_impl, device, lora_adapter):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation=attn_impl,
        trust_remote_code=True, local_files_only=True,
    ).to(device).eval()
    lora_sha256 = None
    lora_layers = None
    if lora_adapter:
        from peft import PeftModel
        print(f"[p0.13] loading LoRA adapter: {lora_adapter}", flush=True)
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


def _eos_ids(qc, tokenizer):
    """Replicate qcmem_generate's EOS contract (int/list generation_config EOS +
    tokenizer EOS fallback)."""
    generation_config = getattr(qc.model, "generation_config", None)
    configured_eos = getattr(generation_config, "eos_token_id", None)
    if configured_eos is None:
        configured_eos = []
    elif isinstance(configured_eos, int):
        configured_eos = [configured_eos]
    else:
        configured_eos = list(configured_eos)
    eos = {int(e) for e in configured_eos if e is not None}
    if tokenizer.eos_token_id is not None:
        eos.add(int(tokenizer.eos_token_id))
    return sorted(eos)


# --------------------------------------------------------------------------- #
# per-arm generate replica of qcmem_generate's ``can_kv`` branch (chat=False),
# with Write / qc.read / decode phase timers. Numerics bit-identical to flagship.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _run_arm(qc, tokenizer, bos_id, selected_chunk_tensors, query_ids,
             max_new_tokens, eos_ids, capture_first=False):
    """Run ONE arm on an ALREADY-selected pack. ``selected_chunk_tensors`` are the
    doc-order context chunks to pack (already picked, resume_j-independent);
    ``query_ids`` is the raw query chunk token id list (chat=False => no gen
    boundary). Returns (generated_ids, timings, read_len, peak_gb, finite,
    first_logits_or_None)."""
    torch.cuda.reset_peak_memory_stats()

    # ---- WRITE phase: encode sink + selected chunks + query prefill to depth j.
    _sync(); tw0 = time.perf_counter()
    sink_hj = qc.write_chunk([bos_id])
    selected_hj = qc.write_chunks(list(selected_chunk_tensors)) \
        if selected_chunk_tensors else []
    q_hj, bottom_cache, q_local_pos = qc.write_prefill(query_ids)
    _sync(); t_write = time.perf_counter() - tw0

    # ---- qc.read phase: prefill the top band over the pack -> first-step logits.
    _sync(); tr0 = time.perf_counter()
    logits1, top_cache, pack_pos = qc.read_prefill(sink_hj, selected_hj, q_hj)
    _sync(); t_read = time.perf_counter() - tr0

    read_len = (int(sink_hj.shape[1]) if sink_hj is not None else 0) \
        + int(sum(h.shape[1] for h in selected_hj)) + len(query_ids)

    first_logits = logits1[0, -1].float()
    finite = bool(torch.isfinite(first_logits).all().item())
    next_logits = first_logits.clone()
    if eos_ids:
        next_logits[eos_ids] = float("-inf")  # step 0 never emits EOS
    first_capture = first_logits.detach().cpu().clone() if capture_first else None

    # ---- decode phase: O(1)/step KV-cache decode (identical to qcmem_generate).
    _sync(); td0 = time.perf_counter()
    generated = []
    next_tok = int(next_logits.argmax().item())
    generated.append(next_tok)
    for _step in range(1, max_new_tokens):
        logits = qc.decode_step(
            next_tok, bottom_cache, top_cache, q_local_pos, pack_pos)
        q_local_pos += 1
        pack_pos += 1
        nl = logits[0, -1].float()
        if not bool(torch.isfinite(nl).all().item()):
            finite = False
        next_tok = int(nl.argmax().item())
        if next_tok in eos_ids:
            break
        generated.append(next_tok)
    _sync(); t_decode = time.perf_counter() - td0

    peak = _peak_gb()
    timings = {"write_s": t_write, "read_s": t_read, "decode_s": t_decode,
               "total_s": t_write + t_read + t_decode}
    return generated, timings, read_len, peak, finite, first_capture


# --------------------------------------------------------------------------- #
# pack construction (forward-free, resume_j-independent) -> the 1:1 pairing key.
# --------------------------------------------------------------------------- #
def _build_pack(input_ids, chunk_size, selector, topk, iter_hop_topk,
                bare_q_ids, tokenizer):
    """Chunk the sample, select context chunks with the FLAGSHIP forward-free
    selector, and return the paired-pack descriptor. Identical for both arms."""
    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    context_chunks = chunks[:-1]
    query_chunk = chunks[-1]
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = int(tokens[0].item())
    sel_idx = qcb._select_context_chunk_indices(
        selector, context_chunks, list(bare_q_ids or []), topk,
        None,  # needle_chunk_set (oracle only)
        context_hj=None, query_hj=None,
        iter_rounds=0, iter_hop_topk=iter_hop_topk, iter_score="meanpool",
        iter_conf_ratio=0.3, iter_max_chunks=64,
    )
    query_ids = query_chunk.tolist()
    packed_ids = [int(bos_id)]
    for i in sel_idx:
        packed_ids.extend(context_chunks[i].tolist())
    packed_ids.extend(query_ids)
    pack_sha = hashlib.sha256(
        b",".join(str(t).encode() for t in packed_ids)).hexdigest()
    return {
        "bos_id": int(bos_id),
        "sel_idx": sel_idx,
        "selected_chunk_tensors": [context_chunks[i] for i in sel_idx],
        "query_ids": query_ids,
        "n_ctx_chunks": len(context_chunks),
        "pack_token_count": len(packed_ids),
        "pack_read_len": 1 + sum(int(context_chunks[i].shape[0]) for i in sel_idx)
                         + len(query_ids),
        "packed_ids_sha256": pack_sha,
    }


# --------------------------------------------------------------------------- #
# QUALITY mode
# --------------------------------------------------------------------------- #
def run_quality(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L = int(model.config.num_hidden_layers)
    # STRICT FIX: LoRA sha must match the P0.12 acceptance record.
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p0.13][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    qc0 = QCMemModel(model, resume_j=args.resume_j_a)    # arm A (0)
    qc12 = QCMemModel(model, resume_j=args.resume_j_b)   # arm B (12)
    eos0 = _eos_ids(qc0, tokenizer)
    eos12 = _eos_ids(qc12, tokenizer)

    task = _resolve_task(args.task)
    length = args.length
    if length not in ruler._LENGTH_TOKENS:
        raise SystemExit(f"[p0.13] unknown length {length}")
    target_tokens = ruler._LENGTH_TOKENS[length]
    sel_name = "iter_bm25"  # FLAGSHIP RULER QCMem selector (mandated)

    base_seed = args.seed + (hash((task, length)) % 100000)
    vt_icl = None
    if task == "variable_tracking":
        vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)
    mnt = args.max_new_tokens if task != "variable_tracking" \
        else max(args.max_new_tokens, 60)

    shard_tag = (f"_shard{args.shard_index}of{args.num_shards}"
                 if args.num_shards > 1 else "")
    sample_indices = set(range(args.limit)[args.shard_index::args.num_shards])

    outdir = Path(args.output_dir) / "quality"
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / f"{task}_{length}{shard_tag}.jsonl"
    fout = open(jsonl_path, "w")

    print(f"[p0.13][quality] {task}/{length}{shard_tag}: selector={sel_name} "
          f"topk={args.topk} hop={args.iter_hop_topk} n={len(sample_indices)}/"
          f"{args.limit} armA=j{args.resume_j_a} armB=j{args.resume_j_b} mnt={mnt}",
          flush=True)

    records = []
    n_done = 0
    for i in range(args.limit):
        rng = random.Random(base_seed * 1000 + i)
        prompt, answers, gold_needle = ruler._build_sample(
            task, target_tokens, tokenizer, rng, vt_icl)
        if i not in sample_indices:
            continue
        bare_q = _bare_question(prompt)
        bare_q_ids = tokenizer.encode(bare_q, add_special_tokens=False)
        ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        if isinstance(ids, list):
            ids = torch.tensor([ids], dtype=torch.long)
        input_ids = ids.to(device)
        approx_tokens = int(input_ids.shape[1])

        # ---- build the pack ONCE (forward-free, resume_j-independent) ----
        t_ret0 = time.perf_counter()
        pack = _build_pack(input_ids, args.chunk_size, sel_name, args.topk,
                           args.iter_hop_topk, bare_q_ids, tokenizer)
        retrieval_s = time.perf_counter() - t_ret0

        # ---- run BOTH arms on that identical pack ----
        oom = False
        try:
            genA, tA, rlA, pkA, finA, lA = _run_arm(
                qc0, tokenizer, pack["bos_id"], pack["selected_chunk_tensors"],
                pack["query_ids"], mnt, eos0, capture_first=True)
            genB, tB, rlB, pkB, finB, lB = _run_arm(
                qc12, tokenizer, pack["bos_id"], pack["selected_chunk_tensors"],
                pack["query_ids"], mnt, eos12, capture_first=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            oom = True
            torch.cuda.empty_cache()
            print(f"[p0.13][OOM] i={i} {task}/{length}: {e}", flush=True)

        if oom:
            rec = {"example_id": i, "task": task, "length": length,
                   "oom": True, "gold": " | ".join(answers)}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            records.append(rec); n_done += 1
            continue

        predA = tokenizer.decode(genA, skip_special_tokens=True).strip()
        predB = tokenizer.decode(genB, skip_special_tokens=True).strip()
        recA = ruler._string_match_all_one(predA, answers)
        recB = ruler._string_match_all_one(predB, answers)

        # 1:1 pairing guard: both arms MUST have consumed the identical pack.
        assert rlA == rlB == pack["pack_read_len"], \
            f"read_len mismatch i={i}: A={rlA} B={rlB} pack={pack['pack_read_len']}"

        # per-example agreement (predictions + first-token next-token logits).
        pred_agree = (predA == predB)
        m = min(len(genA), len(genB))
        decode_top1_agree = (sum(1 for k in range(m) if genA[k] == genB[k]) / m
                             if m else 1.0)
        first_tok_match = bool(int(lA.argmax()) == int(lB.argmax())) \
            if (lA is not None and lB is not None) else None
        first_cos = None; first_kl = None
        if lA is not None and lB is not None:
            first_cos = float(torch.nn.functional.cosine_similarity(
                lA.unsqueeze(0), lB.unsqueeze(0), dim=-1).item())
            logp = torch.log_softmax(lA, -1); logq = torch.log_softmax(lB, -1)
            first_kl = float((logp.exp() * (logp - logq)).sum().item())

        rec = {
            "example_id": i, "task": task, "length": length,
            "approx_tokens": approx_tokens,
            "gold": " | ".join(answers), "n_refs": len(answers),
            "retrieved_chunk_ids": pack["sel_idx"],
            "n_ctx_chunks": pack["n_ctx_chunks"],
            "pack_token_count": pack["pack_token_count"],
            "pack_read_len": pack["pack_read_len"],
            "packed_ids_sha256": pack["packed_ids_sha256"],
            "lora_sha256": lora_sha256,
            "retrieval_s": retrieval_s,
            "armA": {"resume_j": args.resume_j_a, "prediction": predA,
                     "score": recA, "correct": bool(recA >= 1.0),
                     "gen_len": len(genA), "read_len": rlA,
                     "latency_s": tA, "peak_gb": pkA, "finite": finA},
            "armB": {"resume_j": args.resume_j_b, "prediction": predB,
                     "score": recB, "correct": bool(recB >= 1.0),
                     "gen_len": len(genB), "read_len": rlB,
                     "latency_s": tB, "peak_gb": pkB, "finite": finB},
            "agreement": {"prediction_exact": pred_agree,
                          "decode_top1_rate": decode_top1_agree,
                          "first_token_match": first_tok_match,
                          "first_token_cosine": first_cos,
                          "first_token_kl_AtoB": first_kl},
            "diff_score": recA - recB,
        }
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        records.append(rec); n_done += 1
        torch.cuda.empty_cache()
        if n_done % 5 == 0:
            print(f"[p0.13][quality] {task}/{length}{shard_tag} {n_done} done "
                  f"(A={recA:.2f} B={recB:.2f} readlen={rlA})", flush=True)
    fout.close()

    valid = [r for r in records if not r.get("oom")]
    def _mean(key_arm):
        xs = [r[key_arm]["score"] for r in valid]
        return round(sum(xs) / len(xs) * 100.0, 3) if xs else 0.0
    cell = {
        "task": task, "length": length, "shard": shard_tag,
        "n": len(records), "n_valid": len(valid),
        "oom_count": sum(1 for r in records if r.get("oom")),
        "armA_score": _mean("armA"), "armB_score": _mean("armB"),
        "diff_A_minus_B": round(_mean("armA") - _mean("armB"), 3),
        "selector": sel_name, "topk": args.topk, "iter_hop_topk": args.iter_hop_topk,
        "chunk_size": args.chunk_size, "max_new_tokens": mnt,
        "resume_j_a": args.resume_j_a, "resume_j_b": args.resume_j_b,
        "lora_sha256": lora_sha256, "num_layers": L,
        "runtime": {"node": socket.gethostname(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "device": args.device,
                    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                    "seed": args.seed, "dtype": args.dtype,
                    "attn_implementation": args.attn_impl},
        "jsonl": str(jsonl_path),
    }
    with open(outdir / f"{task}_{length}{shard_tag}_cell.json", "w") as f:
        json.dump(cell, f, indent=2)
    print(f"[p0.13][quality] DONE {task}/{length}{shard_tag}: "
          f"A={cell['armA_score']} B={cell['armB_score']} "
          f"diff={cell['diff_A_minus_B']} (n_valid={len(valid)})", flush=True)


# --------------------------------------------------------------------------- #
# LATENCY mode (controlled protocol on a FIXED representative pack)
# --------------------------------------------------------------------------- #
def run_latency(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L = int(model.config.num_hidden_layers)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p0.13][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    qc0 = QCMemModel(model, resume_j=args.resume_j_a)
    qc12 = QCMemModel(model, resume_j=args.resume_j_b)
    eos0 = _eos_ids(qc0, tokenizer)
    eos12 = _eos_ids(qc12, tokenizer)

    task = _resolve_task(args.task)
    length = args.length
    target_tokens = ruler._LENGTH_TOKENS[length]
    base_seed = args.seed + (hash((task, length)) % 100000)
    vt_icl = None
    if task == "variable_tracking":
        vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)
    mnt = args.max_new_tokens if task != "variable_tracking" \
        else max(args.max_new_tokens, 60)

    # fixed representative example
    i = args.example_index
    rng = random.Random(base_seed * 1000 + i)
    prompt, answers, gold_needle = ruler._build_sample(
        task, target_tokens, tokenizer, rng, vt_icl)
    bare_q = _bare_question(prompt)
    bare_q_ids = tokenizer.encode(bare_q, add_special_tokens=False)
    ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    if isinstance(ids, list):
        ids = torch.tensor([ids], dtype=torch.long)
    input_ids = ids.to(device)
    pack = _build_pack(input_ids, args.chunk_size, "iter_bm25", args.topk,
                       args.iter_hop_topk, bare_q_ids, tokenizer)

    def _time_arm(qc, eos):
        writes, reads, decodes, totals = [], [], [], []
        for it in range(args.warmup + args.n_repeat):
            gen, t, rl, pk, fin, _ = _run_arm(
                qc, tokenizer, pack["bos_id"], pack["selected_chunk_tensors"],
                pack["query_ids"], mnt, eos, capture_first=False)
            if it >= args.warmup:
                writes.append(t["write_s"]); reads.append(t["read_s"])
                decodes.append(t["decode_s"]); totals.append(t["total_s"])
            torch.cuda.empty_cache()
        return {"write_s": _summ(writes), "read_s": _summ(reads),
                "decode_s": _summ(decodes), "total_s": _summ(totals),
                "read_len": rl, "peak_gb": pk}

    resA = _time_arm(qc0, eos0)
    resB = _time_arm(qc12, eos12)
    print(f"[p0.13][latency] proc={args.proc_id} {task}/{length} "
          f"A.read med={resA['read_s']['median']*1e3:.1f}ms "
          f"B.read med={resB['read_s']['median']*1e3:.1f}ms "
          f"A.total med={resA['total_s']['median']*1e3:.1f}ms "
          f"B.total med={resB['total_s']['median']*1e3:.1f}ms", flush=True)

    outdir = Path(args.output_dir) / "latency"
    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": "latency", "proc_id": args.proc_id,
        "task": task, "length": length, "example_index": i,
        "config": {"resume_j_a": args.resume_j_a, "resume_j_b": args.resume_j_b,
                   "selector": "iter_bm25", "topk": args.topk,
                   "iter_hop_topk": args.iter_hop_topk,
                   "chunk_size": args.chunk_size, "max_new_tokens": mnt,
                   "warmup": args.warmup, "n_repeat": args.n_repeat,
                   "dtype": args.dtype, "attn_impl": args.attn_impl,
                   "lora_sha256": lora_sha256, "num_layers": L},
        "pack": {"pack_read_len": pack["pack_read_len"],
                 "packed_ids_sha256": pack["packed_ids_sha256"],
                 "sel_idx": pack["sel_idx"]},
        "armA": resA, "armB": resB,
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda" else None,
                "python": platform.python_version(),
                "node": socket.gethostname()},
    }
    with open(outdir / f"latency_proc{args.proc_id}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[p0.13][latency] wrote latency_proc{args.proc_id}.json", flush=True)


# --------------------------------------------------------------------------- #
# MANIFEST mode (strict-fix verification + provenance dump)
# --------------------------------------------------------------------------- #
def run_manifest(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    qc0 = QCMemModel(model, resume_j=args.resume_j_a)
    prov_backbone = _backbone_provenance(qc0, args.model_path)
    prov_lora = _lora_modules(model)
    prov_versions = _versions(device)

    abort = []
    if lora_sha256 != EXPECTED_LORA_SHA:
        abort.append(f"LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    for k, v in EXPECTED_BACKBONE_KEY_SHA.items():
        got = prov_backbone["key_tensor_sha256"].get(k)
        if got != v:
            abort.append(f"backbone {k} sha {got} != expected {v}")
    if prov_lora["count"] != EXPECTED_LORA_MODULE_COUNT:
        abort.append(f"LoRA module count {prov_lora['count']} != "
                     f"{EXPECTED_LORA_MODULE_COUNT}")
    if sorted(lora_layers or []) != list(range(12, 36)):
        abort.append(f"LoRA layers_to_transform {lora_layers} != [12..35]")

    manifest = {
        "run": "P0.13_quality_latency",
        "arms": {"A": {"resume_j": args.resume_j_a, "note": "flagship LoRA, full 36-layer replay"},
                 "B": {"resume_j": args.resume_j_b, "note": "same LoRA, upper-24-layer replay from cached h12"}},
        "strict_fixes": {
            "model_path": args.model_path,
            "lora_adapter": args.lora_adapter,
            "lora_sha256": lora_sha256,
            "expected_lora_sha256": EXPECTED_LORA_SHA,
            "lora_sha_match": lora_sha256 == EXPECTED_LORA_SHA,
            "lora_layers_to_transform": lora_layers,
            "lora_module_count": prov_lora["count"],
            "selector": "iter_bm25", "topk": args.topk,
            "iter_hop_topk": args.iter_hop_topk,
            "sink_tokens": "bos", "chunk_size": args.chunk_size,
            "chat_template": False, "enable_thinking": False,
            "dtype": args.dtype, "attn_impl": args.attn_impl,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "provenance": {"backbone": prov_backbone, "lora": prov_lora,
                       "versions": prov_versions},
        "command": " ".join(sys.argv),
        "abort_reasons": abort,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    if abort:
        print("[p0.13][manifest][ABORT] strict-fix mismatch:", flush=True)
        for a in abort:
            print("   - " + a, flush=True)
        sys.exit(3)
    print(f"[p0.13][manifest] OK — LoRA sha {lora_sha256[:12]}… "
          f"{prov_lora['count']} modules, layers [12..35]; "
          f"torch {prov_versions['torch']} tf {prov_versions['transformers']} "
          f"peft {prov_versions['peft']} git {prov_versions['git_commit_short']}",
          flush=True)


# --------------------------------------------------------------------------- #
# AGGREGATE mode (pure CPU: stats over the per-example JSONL + latency procs)
# --------------------------------------------------------------------------- #
def _paired_bootstrap_ci(diffs, n_boot=10000, seed=0, alpha=0.05):
    """95% CI on the mean paired diff via bootstrap resampling of the example
    pairs (percentile method)."""
    if not diffs:
        return (None, None, None)
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return (sum(diffs) / n, lo, hi)


def _mcnemar_exact(b, c):
    """Exact (binomial) McNemar test. b = #(A correct, B wrong), c = #(A wrong,
    B correct). Two-sided exact p-value under p=0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial: 2 * sum_{i=0}^{k} C(n,i) 0.5^n, capped at 1.
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def run_aggregate(args):
    outdir = Path(args.output_dir)
    qdir = outdir / "quality"
    # collect per-example records from all jsonl shards
    all_recs = []
    for jf in sorted(glob.glob(str(qdir / "*.jsonl"))):
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_recs.append(json.loads(line))
    valid = [r for r in all_recs if not r.get("oom")]
    # de-dup on (task,length,example_id) in case a shard was re-run
    seen = {}
    for r in valid:
        seen[(r["task"], r["length"], r["example_id"])] = r
    valid = list(seen.values())

    # per-cell scores
    cells = {}
    for r in valid:
        key = (r["task"], r["length"])
        cells.setdefault(key, []).append(r)
    per_cell = {}
    macro_A, macro_B = [], []
    for (task, length), recs in sorted(cells.items()):
        a = sum(x["armA"]["score"] for x in recs) / len(recs) * 100.0
        b = sum(x["armB"]["score"] for x in recs) / len(recs) * 100.0
        per_cell[f"{task}/{length}"] = {
            "n": len(recs), "armA": round(a, 2), "armB": round(b, 2),
            "diff_A_minus_B": round(a - b, 2),
        }
        macro_A.append(a); macro_B.append(b)
    n_cells = len(per_cell)
    macroA = sum(macro_A) / n_cells if n_cells else 0.0
    macroB = sum(macro_B) / n_cells if n_cells else 0.0

    # paired bootstrap on the per-EXAMPLE macro diff: weight every example equally
    # within its cell, then macro-average. We build the per-example diff vector
    # (armA.score - armB.score)*100 and bootstrap over examples grouped by cell.
    # For a clean paired CI on the macro, resample examples WITHIN cells and
    # recompute the cell-macro each iteration.
    cell_list = sorted(cells.keys())
    diff_vec_by_cell = {
        c: [(x["armA"]["score"] - x["armB"]["score"]) * 100.0 for x in cells[c]]
        for c in cell_list
    }
    rng = random.Random(0)
    boot_macros = []
    for _ in range(args.n_boot):
        cell_means = []
        for c in cell_list:
            dv = diff_vec_by_cell[c]
            n = len(dv)
            s = sum(dv[rng.randrange(n)] for _ in range(n))
            cell_means.append(s / n)
        boot_macros.append(sum(cell_means) / len(cell_means))
    boot_macros.sort()
    macro_diff = macroA - macroB
    ci_lo = boot_macros[int(0.025 * args.n_boot)]
    ci_hi = boot_macros[int(0.975 * args.n_boot)]

    # McNemar on discrete correctness over ALL paired examples
    b = sum(1 for r in valid if r["armA"]["correct"] and not r["armB"]["correct"])
    c = sum(1 for r in valid if not r["armA"]["correct"] and r["armB"]["correct"])
    both = sum(1 for r in valid if r["armA"]["correct"] and r["armB"]["correct"])
    neither = sum(1 for r in valid
                  if not r["armA"]["correct"] and not r["armB"]["correct"])
    mcnemar_p = _mcnemar_exact(b, c)

    # per-example agreement + failure breakdown
    n = len(valid)
    pred_exact = sum(1 for r in valid if r["agreement"]["prediction_exact"]) / n \
        if n else 0.0
    first_match = sum(1 for r in valid
                      if r["agreement"]["first_token_match"]) / n if n else 0.0
    decode_top1 = sum(r["agreement"]["decode_top1_rate"] for r in valid) / n \
        if n else 0.0
    first_cos = sum(r["agreement"]["first_token_cosine"] for r in valid
                    if r["agreement"]["first_token_cosine"] is not None)
    first_cos = first_cos / n if n else 0.0
    # failure categories on the paired correctness
    fail = {"both_correct": both, "both_wrong": neither,
            "A_only_correct": b, "B_only_correct": c}
    any_oom = sum(1 for r in all_recs if r.get("oom"))
    any_nonfinite = sum(1 for r in valid
                        if not r["armA"]["finite"] or not r["armB"]["finite"])
    pack_paired = all(r["armA"]["read_len"] == r["armB"]["read_len"]
                      == r["pack_read_len"] for r in valid)

    summary = {
        "n_examples_paired": n, "n_cells": n_cells,
        "per_cell": per_cell,
        "macro": {"armA": round(macroA, 3), "armB": round(macroB, 3),
                  "diff_A_minus_B": round(macro_diff, 3)},
        "oom_examples": any_oom, "nonfinite_examples": any_nonfinite,
        "all_packs_paired_1to1": pack_paired,
    }
    stats = {
        "macro_diff_A_minus_B": round(macro_diff, 4),
        "paired_bootstrap_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "bootstrap_n": args.n_boot,
        "mcnemar": {"A_only_correct_b": b, "B_only_correct_c": c,
                    "both_correct": both, "neither_correct": neither,
                    "exact_two_sided_p": mcnemar_p},
        "agreement": {"prediction_exact_rate": round(pred_exact, 4),
                      "first_token_match_rate": round(first_match, 4),
                      "decode_top1_rate_mean": round(decode_top1, 4),
                      "first_token_cosine_mean": round(first_cos, 4)},
        "failure_breakdown": fail,
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(outdir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # latency aggregation across the >=3 procs
    lat_procs = []
    for lf in sorted(glob.glob(str(outdir / "latency" / "latency_proc*.json"))):
        with open(lf) as f:
            lat_procs.append(json.load(f))
    latency = {"n_procs": len(lat_procs), "procs": []}
    if lat_procs:
        for lp in lat_procs:
            latency["procs"].append({
                "proc_id": lp["proc_id"], "task": lp["task"],
                "length": lp["length"],
                "armA_read_ms": round(lp["armA"]["read_s"]["median"] * 1e3, 3),
                "armB_read_ms": round(lp["armB"]["read_s"]["median"] * 1e3, 3),
                "armA_total_ms": round(lp["armA"]["total_s"]["median"] * 1e3, 3),
                "armB_total_ms": round(lp["armB"]["total_s"]["median"] * 1e3, 3),
            })
        # pooled per-arm medians over all procs' raw timed reads
        def _pool(arm, phase):
            raw = []
            for lp in lat_procs:
                raw.extend(lp[arm][phase]["raw"])
            return _summ(raw)
        latency["pooled"] = {
            "armA": {"read_s": _pool("armA", "read_s"),
                     "total_s": _pool("armA", "total_s"),
                     "write_s": _pool("armA", "write_s"),
                     "decode_s": _pool("armA", "decode_s")},
            "armB": {"read_s": _pool("armB", "read_s"),
                     "total_s": _pool("armB", "total_s"),
                     "write_s": _pool("armB", "write_s"),
                     "decode_s": _pool("armB", "decode_s")},
        }
        aR = latency["pooled"]["armA"]["read_s"]["median"]
        bR = latency["pooled"]["armB"]["read_s"]["median"]
        aT = latency["pooled"]["armA"]["total_s"]["median"]
        bT = latency["pooled"]["armB"]["total_s"]["median"]
        latency["read_speedup_A_over_B"] = round(aR / bR, 4) if bR else None
        latency["total_speedup_A_over_B"] = round(aT / bT, 4) if bT else None
        # direction-consistency: B faster than A in EVERY proc?
        latency["direction_consistent_B_faster_read"] = all(
            lp["armB"]["read_s"]["median"] < lp["armA"]["read_s"]["median"]
            for lp in lat_procs)
    with open(outdir / "latency.json", "w") as f:
        json.dump(latency, f, indent=2)

    print("=" * 70)
    print(f"[p0.13][aggregate] n_paired={n} n_cells={n_cells}")
    print(f"  macro  A(j=0)={macroA:.2f}  B(j=12)={macroB:.2f}  "
          f"diff(A-B)={macro_diff:.2f}  95%CI=[{ci_lo:.2f},{ci_hi:.2f}]")
    print(f"  McNemar exact p={mcnemar_p:.4g} (b={b} c={c} both={both} "
          f"neither={neither})")
    print(f"  agreement: pred_exact={pred_exact:.3f} first_tok={first_match:.3f} "
          f"decode_top1={decode_top1:.3f}")
    print(f"  packs_paired_1to1={pack_paired} oom={any_oom} "
          f"nonfinite={any_nonfinite}")
    if lat_procs:
        print(f"  latency pooled: A.read={latency['pooled']['armA']['read_s']['median']*1e3:.1f}ms "
              f"B.read={latency['pooled']['armB']['read_s']['median']*1e3:.1f}ms "
              f"read_speedup(A/B)={latency.get('read_speedup_A_over_B')}")
    print("=" * 70, flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="P0.13 quality<->latency paired bench")
    ap.add_argument("--mode", required=True,
                    choices=["manifest", "quality", "latency", "aggregate"])
    ap.add_argument("--model_path", type=str,
                    default="models/Qwen3-8b-local")
    ap.add_argument("--lora_adapter", type=str,
                    default="outputs/qcmem_distill_qwen_j12_r32_4k/final")
    ap.add_argument("--resume_j_a", type=int, default=0)
    ap.add_argument("--resume_j_b", type=int, default=12)
    ap.add_argument("--task", type=str, default="niah_single_3")
    ap.add_argument("--length", type=str, default="16k")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--iter_hop_topk", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str,
                    default="bench_results/p0_13_quality_latency")
    # latency mode
    ap.add_argument("--proc_id", type=int, default=0)
    ap.add_argument("--example_index", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--n_repeat", type=int, default=20)
    # aggregate mode
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    if args.mode == "aggregate":
        run_aggregate(args)
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print("=" * 78)
    print(f"P0.13 :: mode={args.mode} task={args.task} length={args.length} "
          f"shard={args.shard_index}/{args.num_shards}")
    print(f"  model={args.model_path} lora={args.lora_adapter}")
    print(f"  armA=j{args.resume_j_a} armB=j{args.resume_j_b} topk={args.topk} "
          f"hop={args.iter_hop_topk} chunk={args.chunk_size} dtype={dtype} "
          f"attn={args.attn_impl} device={device}")
    print("=" * 78, flush=True)

    if args.mode == "manifest":
        run_manifest(args, device, dtype)
    elif args.mode == "quality":
        run_quality(args, device, dtype)
    elif args.mode == "latency":
        run_latency(args, device, dtype)


if __name__ == "__main__":
    main()
