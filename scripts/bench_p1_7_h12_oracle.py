#!/usr/bin/env python
"""P1.7 — continuous-prefix h12 attribution oracle (INFERENCE-ONLY).

Decomposes the P0.13 3.12-pp macro A-B gap into (i) "continuous lower-layer
context computation" vs (ii) the "deployable chunk-local Write / repositioning"
interface, by inserting a THIRD, non-deployable ORACLE arm between the two P0.13
arms. All three arms share the SAME example, the SAME retrieved pack
(same context-chunk ids + order + packed token ids + pack sha), the SAME flagship
rank-32 LoRA (sha ``dd09cd17…``, layers 12..35), and the SAME decode / scorer.

Three arms (differ ONLY in HOW h12 is produced before the shared upper-24 replay):
  * Arm A — ``resume_j=0`` full replay + flagship LoRA (== P0.13 Arm A). The pack
            [sink ; ctx… ; query] is run through layers[0:36] as ONE continuous,
            fully-causal forward with fresh contiguous RoPE positions. (RAG upper
            bound.) Reused VERBATIM from ``bench_p0_13_quality_latency._run_arm``.
  * Arm B — ``resume_j=12`` chunk-local cached h12 + SAME LoRA (== P0.13 Arm B).
            Each selected chunk (and the sink, and the query) is encoded to depth
            12 CHUNK-LOCALLY (isolated, RoPE 0:T each), the per-chunk h12 are then
            REPOSITIONED into a fresh contiguous pack and layers[12:36] resume from
            there. Deployable (cache once, read many). Reused VERBATIM from
            ``bench_p0_13_quality_latency._run_arm``.
  * Arm C — ORACLE (new). On the EXACT SAME packed token ids as A/B, layers[0:12]
            are run as ONE CONTINUOUS, fully-causal forward over the WHOLE pack
            (contiguous positions 0:H, every position attends causally to all
            preceding pack positions — i.e. cross-chunk context is present), the
            pack-level h12 is captured ONCE, and layers[12:36] resume from it with
            the SAME LoRA. This is the "resume at 12 from a *continuous* (not
            chunk-local) h12" operating point.

Attribution logic (per paperA/TODOList.md §P1.7):
  * Because Arm C computes h12 by a continuous split-at-12 forward over the same
    packed ids that Arm A runs continuously through all 36 layers, Arm C ≈ Arm A to
    floating-point tolerance BY CONSTRUCTION (a continuous forward is split-point
    invariant). Arm C therefore holds the resume-at-12 INTERFACE fixed (same as B)
    while replacing chunk-local h12 with continuous h12.
      - C ≈ A (and C ≫ B): the resume interface is lossless; essentially ALL of the
        A-B gap is attributable to chunk-local Write / repositioning.
      - C ≈ B (and C ≪ A): even a perfect continuous h12 cannot be recovered once
        the read skips the lower-12 recompute; the gap is the skip / interface.
    Any result is admissible; the empirical answer (with paired CI + McNemar) is the
    deliverable. No claim about knowledge "residing" in any layer is made.

!! ORACLE IS NOT A DEPLOYABLE METHOD. It must re-run the lower-12 layers over the
   full selected pack FOR EVERY QUERY (the continuous h12 depends on the query's
   position in the pack), so it cannot be cached across queries. It exists ONLY for
   this attribution and must never be reported as a deployable configuration.

KEY INVARIANT (violating it invalidates the oracle):
   Arm C's layer-12 state MUST equal the SAME-PACK stock lower-12 forward
   (continuous positions, full causal mask) — it is NOT stitched together from
   independent chunk caches. The ``--verify`` flag / ``--mode h12_sanity`` run a
   numerical assertion: compute the oracle continuous h12 and compare it against the
   stock model's ``output_hidden_states[resume_j]`` on the SAME packed token ids
   (the LoRA lives on layers 12..35, so hidden_states[12] is the pure lower-12
   forward, unaffected by the adapter). The check reports the actual bf16 max-abs
   residual and asserts it is below ``--h12_tol``.

Pack-pairing (resume_j-independent, forward-free):
   The flagship RULER selector ``iter_bm25`` is pure lexical BM25 over raw token
   ids — no model forward — so the selected chunk indices / order / packed token ids
   are ``resume_j``-INDEPENDENT. This harness builds the pack ONCE per example with
   the UNMODIFIED ``bench_p0_13_quality_latency._build_pack`` (same seed, task,
   length, chunk_size, topk, iter_hop_topk, tokenizer as P0.13), so the pack sha is
   bit-identical to P0.13 and all THREE arms read the identical pack. The oracle
   reconstructs the packed token id list from the same descriptor and re-verifies the
   ``packed_ids_sha256`` before running. Optionally (``--p013_manifest_dir``) it
   cross-checks each example's pack sha against the P0.13 per-example JSONL.

Modes (one process each; the launcher fans them across GPUs):
  * ``manifest``   — load model+LoRA once, assert backbone/LoRA hashes match the
                     P0.12/P0.13 acceptance record, dump manifest.json. Exit 3 on
                     any strict-fix mismatch. Run ONCE first.
  * ``h12_sanity`` — build ONE pack, run the oracle continuous h12 + the stock
                     lower-12 reference, print the max-abs residual, assert < tol.
  * ``quality``    — one (task,length[,shard]): per example build the pack once, run
                     ALL THREE arms on that identical pack, write per-example JSONL +
                     a per-cell summary json. ``--verify`` additionally runs the h12
                     sanity on the first processed example before scoring.
  * ``latency``    — controlled per-arm read/write/decode timing on a fixed pack
                     (3 arms; median/p10/p90 over N timed reads).
  * ``aggregate``  — pure CPU: merge quality shards -> per-cell A/B/C means, macro,
                     pairwise (A-B, A-C, B-C) paired bootstrap 95% CI + exact
                     McNemar + first-token/decode agreement + failure breakdown.

This file NEVER mutates ``bench_p0_13_quality_latency.py`` or ``qcmem_model.py`` —
it imports the P0.13 pack/arm helpers and the QCMemModel primitives read-only and
adds only the oracle forward (which uses QCMemModel's public low-level accessors,
never patching the backbone).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import random
import socket
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

# Import the UNMODIFIED P0.13 harness and pull every shared primitive from it so
# arms A and B are BIT-IDENTICAL to the P0.13 headline path (same pack builder,
# same per-arm generate replica, same loader, same strict-fix hashes, same stats).
import bench_p0_13_quality_latency as p013  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

ruler = p013.ruler
qcb = p013.qcb
QCMemModel = p013.QCMemModel
_bare_question = p013._bare_question
_resolve_task = p013._resolve_task
_build_pack = p013._build_pack
_run_arm = p013._run_arm
_load = p013._load
_eos_ids = p013._eos_ids
_sync = p013._sync
_peak_gb = p013._peak_gb
_paired_bootstrap_ci = p013._paired_bootstrap_ci
_mcnemar_exact = p013._mcnemar_exact
_summ = p013._summ
_backbone_provenance = p013._backbone_provenance
_lora_modules = p013._lora_modules
_versions = p013._versions
EXPECTED_LORA_SHA = p013.EXPECTED_LORA_SHA
EXPECTED_BACKBONE_KEY_SHA = p013.EXPECTED_BACKBONE_KEY_SHA
EXPECTED_LORA_MODULE_COUNT = p013.EXPECTED_LORA_MODULE_COUNT


# --------------------------------------------------------------------------- #
# pack -> packed token id list (the pairing key shared with A/B and P0.13).
# --------------------------------------------------------------------------- #
def _packed_ids_from_pack(pack) -> list:
    """Reconstruct the EXACT packed token id list [bos ; ctx… ; query] that
    ``_build_pack`` hashed, so the oracle consumes the identical pack. Verifies the
    sha before returning (any mismatch aborts — the pairing guarantee is broken)."""
    ids = [int(pack["bos_id"])]
    for ch in pack["selected_chunk_tensors"]:
        ids.extend(ch.tolist())
    ids.extend(list(pack["query_ids"]))
    sha = hashlib.sha256(b",".join(str(t).encode() for t in ids)).hexdigest()
    if sha != pack["packed_ids_sha256"]:
        raise AssertionError(
            f"[p1.7] packed_ids sha mismatch: reconstructed {sha[:12]}… != "
            f"pack {pack['packed_ids_sha256'][:12]}… — pairing broken")
    return ids


# --------------------------------------------------------------------------- #
# ORACLE continuous lower-12 forward (the h12 the sanity asserts against).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _oracle_lower12(qc: QCMemModel, packed_ids):
    """Run layers[0:resume_j] as ONE continuous fully-causal forward over the whole
    packed sequence (contiguous positions 0:H). Returns the pack-level h12
    ``[1, H, d]`` — this MUST match the stock lower-12 forward of the SAME ids."""
    ids = qc._as_ids(packed_ids)                 # [1, H]
    H = ids.shape[1]
    emb = qc.embed_tokens(ids)
    positions = torch.arange(H, device=qc.device).unsqueeze(0)
    mask, pe = qc._make_mask_and_rope(emb, positions)
    h12 = qc._run_layers(emb, slice(0, qc.resume_j), mask, positions, pe)
    return h12


@torch.no_grad()
def _stock_lower12_ref(qc: QCMemModel, packed_ids):
    """Stock reference: the model's own ``output_hidden_states[resume_j]`` on the
    SAME packed ids. hidden_states[j] == the residual stream AFTER layers[0:j], i.e.
    the input to layer j. The flagship LoRA lives on layers 12..35, so at j=12 this
    reference is the PURE lower-12 forward (adapter-independent) — an INDEPENDENT
    code path from ``_oracle_lower12`` (HF's own masking/RoPE), which is exactly what
    makes it a meaningful numerical cross-check."""
    ids = qc._as_ids(packed_ids)
    out = qc.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return out.hidden_states[qc.resume_j]        # [1, H, d]


@torch.no_grad()
def _h12_residual(qc: QCMemModel, packed_ids):
    """Max-abs residual between the oracle continuous h12 and the stock lower-12
    reference on the same ids (reported in bf16; the invariant gate)."""
    h12 = _oracle_lower12(qc, packed_ids).float()
    ref = _stock_lower12_ref(qc, packed_ids).float()
    diff = (h12 - ref).abs()
    return {"max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "ref_abs_max": float(ref.abs().max().item()),
            "H": int(h12.shape[1])}


# --------------------------------------------------------------------------- #
# ORACLE arm generate (continuous h12 -> upper-24 replay + O(1) KV-cache decode).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _run_oracle(qc: QCMemModel, packed_ids, max_new_tokens, eos_ids,
                capture_first=False):
    """Run the ORACLE arm on an already-selected pack (identical token ids to arms
    A/B). Continuous lower-``resume_j`` over the WHOLE pack -> pack-level h12 ->
    layers[resume_j:L] resume, then O(1)/step decode.

    Decode reuses ``QCMemModel.decode_step`` VERBATIM with ``q_local_pos == pack_pos``
    (the oracle has a SINGLE continuous coordinate system — the bottom band's KV
    cache spans the WHOLE pack, so a generated token attends to the whole continuous
    lower-12 context, unlike Arm B whose bottom cache is the isolated query only).

    Returns ``(generated_ids, timings, read_len, peak_gb, finite, first_logits)`` —
    the SAME 6-tuple shape ``_run_arm`` returns, so the caller treats all three arms
    uniformly."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ids = qc._as_ids(packed_ids)                 # [1, H]
    H = ids.shape[1]

    # ---- WRITE phase (oracle): CONTINUOUS lower-band over the WHOLE pack ----
    _sync(); tw0 = time.perf_counter()
    emb = qc.embed_tokens(ids)
    positions = torch.arange(H, device=qc.device).unsqueeze(0)
    mask, pe = qc._make_mask_and_rope(emb, positions)
    bottom_cache = DynamicCache(config=qc.config)
    h12 = qc._run_layers(emb, slice(0, qc.resume_j), mask, positions, pe,
                         past_key_values=bottom_cache, use_cache=True)  # [1,H,d]
    _sync(); t_write = time.perf_counter() - tw0

    # ---- READ phase (oracle): upper-band resume over the continuous h12 ----
    _sync(); tr0 = time.perf_counter()
    top_cache = DynamicCache(config=qc.config)
    hidden = qc._run_layers(h12, slice(qc.resume_j, qc.num_layers),
                            mask, positions, pe,
                            past_key_values=top_cache, use_cache=True)  # [1,H,d]
    last = qc.norm(hidden[:, -1:, :])
    logits1 = qc.lm_head(last)                   # [1,1,V]
    _sync(); t_read = time.perf_counter() - tr0

    read_len = H
    first_logits = logits1[0, -1].float()
    finite = bool(torch.isfinite(first_logits).all().item())
    next_logits = first_logits.clone()
    if eos_ids:
        next_logits[eos_ids] = float("-inf")     # step 0 never emits EOS
    first_capture = first_logits.detach().cpu().clone() if capture_first else None

    # ---- DECODE phase: O(1)/step, continuous positions (q_local_pos == pack_pos) --
    _sync(); td0 = time.perf_counter()
    generated = []
    next_tok = int(next_logits.argmax().item())
    generated.append(next_tok)
    pack_pos = H
    for _step in range(1, max_new_tokens):
        logits = qc.decode_step(next_tok, bottom_cache, top_cache,
                                pack_pos, pack_pos)   # continuous: same position
        pack_pos += 1
        nl = logits[0, -1].float()
        if not bool(torch.isfinite(nl).all().item()):
            finite = False
        next_tok = int(nl.argmax().item())
        if next_tok in eos_ids:
            break
        generated.append(next_tok)
    _sync(); t_decode = time.perf_counter() - td0

    peak = _peak_gb() if torch.cuda.is_available() else 0.0
    timings = {"write_s": t_write, "read_s": t_read, "decode_s": t_decode,
               "total_s": t_write + t_read + t_decode}
    return generated, timings, read_len, peak, finite, first_capture


# --------------------------------------------------------------------------- #
# per-example pairwise agreement helpers.
# --------------------------------------------------------------------------- #
def _pair_agree(genX, lX, genY, lY):
    """First-token match / cosine + decode top-1 rate between two arms' outputs."""
    m = min(len(genX), len(genY))
    decode_top1 = (sum(1 for k in range(m) if genX[k] == genY[k]) / m) if m else 1.0
    ft_match = None; ft_cos = None
    if lX is not None and lY is not None:
        ft_match = bool(int(lX.argmax()) == int(lY.argmax()))
        ft_cos = float(torch.nn.functional.cosine_similarity(
            lX.unsqueeze(0), lY.unsqueeze(0), dim=-1).item())
    return {"decode_top1_rate": decode_top1,
            "first_token_match": ft_match, "first_token_cosine": ft_cos}


# --------------------------------------------------------------------------- #
# QUALITY mode (3 arms on the identical pack)
# --------------------------------------------------------------------------- #
def run_quality(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L = int(model.config.num_hidden_layers)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p1.7][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    # Arm A (full replay), Arm B (chunk-local h12), Arm C (oracle: continuous h12).
    qcA = QCMemModel(model, resume_j=args.resume_j_a)     # 0
    qcB = QCMemModel(model, resume_j=args.resume_j_b)     # 12
    qcC = QCMemModel(model, resume_j=args.resume_j_c)     # 12 (oracle)
    eosA = _eos_ids(qcA, tokenizer)
    eosB = _eos_ids(qcB, tokenizer)
    eosC = _eos_ids(qcC, tokenizer)

    task = _resolve_task(args.task)
    length = args.length
    if length not in ruler._LENGTH_TOKENS:
        raise SystemExit(f"[p1.7] unknown length {length}")
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

    # Optional cross-check against the P0.13 pack manifest.
    p013_shas = {}
    if args.p013_manifest_dir:
        for jf in glob.glob(os.path.join(args.p013_manifest_dir, "quality",
                                         f"{task}_{length}*.jsonl")):
            with open(jf) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if not r.get("oom") and "packed_ids_sha256" in r:
                        p013_shas[int(r["example_id"])] = r["packed_ids_sha256"]

    print(f"[p1.7][quality] {task}/{length}{shard_tag}: selector={sel_name} "
          f"topk={args.topk} hop={args.iter_hop_topk} n={len(sample_indices)}/"
          f"{args.limit} A=j{args.resume_j_a} B=j{args.resume_j_b} "
          f"C(oracle)=j{args.resume_j_c} mnt={mnt} verify={args.verify}",
          flush=True)

    records = []
    n_done = 0
    verified_once = False
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
        packed_ids = _packed_ids_from_pack(pack)  # verifies sha internally

        # cross-check pack sha vs P0.13 (if a manifest dir was given)
        p013_sha_match = None
        if i in p013_shas:
            p013_sha_match = (p013_shas[i] == pack["packed_ids_sha256"])
            if not p013_sha_match:
                raise AssertionError(
                    f"[p1.7] example {i}: pack sha != P0.13 pack sha "
                    "— pairing with P0.13 arms broken")

        # optional h12 invariant sanity on the FIRST processed example
        h12_check = None
        if args.verify and not verified_once:
            h12_check = _h12_residual(qcC, packed_ids)
            print(f"[p1.7][verify] example {i}: h12 continuous-vs-stock "
                  f"max_abs={h12_check['max_abs']:.3e} "
                  f"mean_abs={h12_check['mean_abs']:.3e} "
                  f"(ref_abs_max={h12_check['ref_abs_max']:.3e} H={h12_check['H']}) "
                  f"tol={args.h12_tol:.3e}", flush=True)
            assert h12_check["max_abs"] < args.h12_tol, (
                f"[p1.7][ABORT] oracle h12 residual {h12_check['max_abs']:.3e} "
                f">= tol {args.h12_tol:.3e} — continuous-prefix invariant violated")
            verified_once = True

        # ---- run ALL THREE arms on that identical pack ----
        oom = False
        try:
            genA, tA, rlA, pkA, finA, lA = _run_arm(
                qcA, tokenizer, pack["bos_id"], pack["selected_chunk_tensors"],
                pack["query_ids"], mnt, eosA, capture_first=True)
            genB, tB, rlB, pkB, finB, lB = _run_arm(
                qcB, tokenizer, pack["bos_id"], pack["selected_chunk_tensors"],
                pack["query_ids"], mnt, eosB, capture_first=True)
            genC, tC, rlC, pkC, finC, lC = _run_oracle(
                qcC, packed_ids, mnt, eosC, capture_first=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            oom = True
            torch.cuda.empty_cache()
            print(f"[p1.7][OOM] i={i} {task}/{length}: {e}", flush=True)

        if oom:
            rec = {"example_id": i, "task": task, "length": length,
                   "oom": True, "gold": " | ".join(answers)}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            records.append(rec); n_done += 1
            continue

        predA = tokenizer.decode(genA, skip_special_tokens=True).strip()
        predB = tokenizer.decode(genB, skip_special_tokens=True).strip()
        predC = tokenizer.decode(genC, skip_special_tokens=True).strip()
        recA = ruler._string_match_all_one(predA, answers)
        recB = ruler._string_match_all_one(predB, answers)
        recC = ruler._string_match_all_one(predC, answers)

        # 1:1 pairing guard: ALL arms MUST have consumed the identical pack length.
        assert rlA == rlB == rlC == pack["pack_read_len"], \
            (f"read_len mismatch i={i}: A={rlA} B={rlB} C={rlC} "
             f"pack={pack['pack_read_len']}")

        rec = {
            "example_id": i, "task": task, "length": length,
            "approx_tokens": approx_tokens,
            "gold": " | ".join(answers), "n_refs": len(answers),
            "retrieved_chunk_ids": pack["sel_idx"],
            "n_ctx_chunks": pack["n_ctx_chunks"],
            "pack_token_count": pack["pack_token_count"],
            "pack_read_len": pack["pack_read_len"],
            "packed_ids_sha256": pack["packed_ids_sha256"],
            "p013_pack_sha_match": p013_sha_match,
            "lora_sha256": lora_sha256,
            "retrieval_s": retrieval_s,
            "h12_sanity": h12_check,
            "armA": {"resume_j": args.resume_j_a, "kind": "full_replay",
                     "prediction": predA, "score": recA,
                     "correct": bool(recA >= 1.0), "gen_len": len(genA),
                     "read_len": rlA, "latency_s": tA, "peak_gb": pkA,
                     "finite": finA},
            "armB": {"resume_j": args.resume_j_b, "kind": "chunk_local_h12",
                     "prediction": predB, "score": recB,
                     "correct": bool(recB >= 1.0), "gen_len": len(genB),
                     "read_len": rlB, "latency_s": tB, "peak_gb": pkB,
                     "finite": finB},
            "armC": {"resume_j": args.resume_j_c, "kind": "continuous_h12_oracle",
                     "prediction": predC, "score": recC,
                     "correct": bool(recC >= 1.0), "gen_len": len(genC),
                     "read_len": rlC, "latency_s": tC, "peak_gb": pkC,
                     "finite": finC},
            "agreement": {
                "A_vs_B": _pair_agree(genA, lA, genB, lB),
                "A_vs_C": _pair_agree(genA, lA, genC, lC),
                "B_vs_C": _pair_agree(genB, lB, genC, lC),
            },
            "diff_A_minus_B": recA - recB,
            "diff_A_minus_C": recA - recC,
            "diff_C_minus_B": recC - recB,
        }
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        records.append(rec); n_done += 1
        torch.cuda.empty_cache()
        if n_done % 5 == 0:
            print(f"[p1.7][quality] {task}/{length}{shard_tag} {n_done} done "
                  f"(A={recA:.2f} B={recB:.2f} C={recC:.2f} readlen={rlA})",
                  flush=True)
    fout.close()

    valid = [r for r in records if not r.get("oom")]
    def _mean(arm):
        xs = [r[arm]["score"] for r in valid]
        return round(sum(xs) / len(xs) * 100.0, 3) if xs else 0.0
    cell = {
        "task": task, "length": length, "shard": shard_tag,
        "n": len(records), "n_valid": len(valid),
        "oom_count": sum(1 for r in records if r.get("oom")),
        "armA_score": _mean("armA"), "armB_score": _mean("armB"),
        "armC_score": _mean("armC"),
        "diff_A_minus_B": round(_mean("armA") - _mean("armB"), 3),
        "diff_A_minus_C": round(_mean("armA") - _mean("armC"), 3),
        "diff_C_minus_B": round(_mean("armC") - _mean("armB"), 3),
        "selector": sel_name, "topk": args.topk, "iter_hop_topk": args.iter_hop_topk,
        "chunk_size": args.chunk_size, "max_new_tokens": mnt,
        "resume_j_a": args.resume_j_a, "resume_j_b": args.resume_j_b,
        "resume_j_c": args.resume_j_c, "lora_sha256": lora_sha256, "num_layers": L,
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
    print(f"[p1.7][quality] DONE {task}/{length}{shard_tag}: "
          f"A={cell['armA_score']} B={cell['armB_score']} C={cell['armC_score']} "
          f"(A-B={cell['diff_A_minus_B']} A-C={cell['diff_A_minus_C']} "
          f"C-B={cell['diff_C_minus_B']} n_valid={len(valid)})", flush=True)


# --------------------------------------------------------------------------- #
# LATENCY mode (controlled per-arm read/write/decode timing on a fixed pack)
# --------------------------------------------------------------------------- #
def run_latency(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L = int(model.config.num_hidden_layers)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p1.7][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    qcA = QCMemModel(model, resume_j=args.resume_j_a)
    qcB = QCMemModel(model, resume_j=args.resume_j_b)
    qcC = QCMemModel(model, resume_j=args.resume_j_c)
    eosA = _eos_ids(qcA, tokenizer); eosB = _eos_ids(qcB, tokenizer)
    eosC = _eos_ids(qcC, tokenizer)

    task = _resolve_task(args.task)
    length = args.length
    target_tokens = ruler._LENGTH_TOKENS[length]
    base_seed = args.seed + (hash((task, length)) % 100000)
    vt_icl = None
    if task == "variable_tracking":
        vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)
    mnt = args.max_new_tokens if task != "variable_tracking" \
        else max(args.max_new_tokens, 60)

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
    packed_ids = _packed_ids_from_pack(pack)

    def _time_ab(qc, eos):
        writes, reads, decodes, totals = [], [], [], []
        rl = pk = None
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

    def _time_c():
        writes, reads, decodes, totals = [], [], [], []
        rl = pk = None
        for it in range(args.warmup + args.n_repeat):
            gen, t, rl, pk, fin, _ = _run_oracle(
                qcC, packed_ids, mnt, eosC, capture_first=False)
            if it >= args.warmup:
                writes.append(t["write_s"]); reads.append(t["read_s"])
                decodes.append(t["decode_s"]); totals.append(t["total_s"])
            torch.cuda.empty_cache()
        return {"write_s": _summ(writes), "read_s": _summ(reads),
                "decode_s": _summ(decodes), "total_s": _summ(totals),
                "read_len": rl, "peak_gb": pk}

    resA = _time_ab(qcA, eosA)
    resB = _time_ab(qcB, eosB)
    resC = _time_c()
    print(f"[p1.7][latency] proc={args.proc_id} {task}/{length} "
          f"A.read={resA['read_s']['median']*1e3:.1f}ms "
          f"B.read={resB['read_s']['median']*1e3:.1f}ms "
          f"C.read={resC['read_s']['median']*1e3:.1f}ms "
          f"C.write={resC['write_s']['median']*1e3:.1f}ms "
          f"(oracle recomputes lower-12 over the whole pack per query)", flush=True)

    outdir = Path(args.output_dir) / "latency"
    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": "latency", "proc_id": args.proc_id,
        "task": task, "length": length, "example_index": i,
        "config": {"resume_j_a": args.resume_j_a, "resume_j_b": args.resume_j_b,
                   "resume_j_c": args.resume_j_c, "selector": "iter_bm25",
                   "topk": args.topk, "iter_hop_topk": args.iter_hop_topk,
                   "chunk_size": args.chunk_size, "max_new_tokens": mnt,
                   "warmup": args.warmup, "n_repeat": args.n_repeat,
                   "dtype": args.dtype, "attn_impl": args.attn_impl,
                   "lora_sha256": lora_sha256, "num_layers": L},
        "pack": {"pack_read_len": pack["pack_read_len"],
                 "packed_ids_sha256": pack["packed_ids_sha256"],
                 "sel_idx": pack["sel_idx"]},
        "armA": resA, "armB": resB, "armC": resC,
        "note": "Arm C (oracle) is NOT deployable: it re-runs the lower-12 layers "
                "over the whole selected pack per query (see write_s); it cannot be "
                "cached across queries. Reported only for attribution.",
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda" else None,
                "python": platform.python_version(),
                "node": socket.gethostname()},
    }
    with open(outdir / f"latency_proc{args.proc_id}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[p1.7][latency] wrote latency_proc{args.proc_id}.json", flush=True)


# --------------------------------------------------------------------------- #
# H12_SANITY mode (standalone continuous-prefix invariant check on ONE pack)
# --------------------------------------------------------------------------- #
def run_h12_sanity(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p1.7][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    qcC = QCMemModel(model, resume_j=args.resume_j_c)

    task = _resolve_task(args.task)
    length = args.length
    target_tokens = ruler._LENGTH_TOKENS[length]
    base_seed = args.seed + (hash((task, length)) % 100000)
    vt_icl = None
    if task == "variable_tracking":
        vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)
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
    packed_ids = _packed_ids_from_pack(pack)

    res = _h12_residual(qcC, packed_ids)
    print("=" * 70)
    print(f"[p1.7][h12_sanity] {task}/{length} example {i}: "
          f"pack H={res['H']} (sha {pack['packed_ids_sha256'][:12]}…)")
    print(f"  continuous-oracle-h12 vs stock lower-{args.resume_j_c} forward:")
    print(f"    max_abs={res['max_abs']:.3e}  mean_abs={res['mean_abs']:.3e}  "
          f"(ref_abs_max={res['ref_abs_max']:.3e})  tol={args.h12_tol:.3e}")
    print("=" * 70, flush=True)
    assert res["max_abs"] < args.h12_tol, (
        f"[p1.7][ABORT] oracle h12 residual {res['max_abs']:.3e} >= "
        f"tol {args.h12_tol:.3e} — continuous-prefix invariant violated")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "h12_sanity.json", "w") as f:
        json.dump({"task": task, "length": length, "example_index": i,
                   "packed_ids_sha256": pack["packed_ids_sha256"],
                   "resume_j_c": args.resume_j_c, "dtype": args.dtype,
                   "h12_residual": res, "h12_tol": args.h12_tol,
                   "passed": True}, f, indent=2)
    print(f"[p1.7][h12_sanity] PASS — residual within tol; wrote h12_sanity.json",
          flush=True)


# --------------------------------------------------------------------------- #
# MANIFEST mode (strict-fix verification + provenance dump)
# --------------------------------------------------------------------------- #
def run_manifest(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    qcA = QCMemModel(model, resume_j=args.resume_j_a)
    prov_backbone = _backbone_provenance(qcA, args.model_path)
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
        "run": "P1.7_h12_oracle",
        "arms": {
            "A": {"resume_j": args.resume_j_a,
                  "note": "flagship LoRA, full 36-layer continuous replay "
                          "(== P0.13 Arm A)"},
            "B": {"resume_j": args.resume_j_b,
                  "note": "same LoRA, upper-24 replay from CHUNK-LOCAL cached h12 "
                          "(== P0.13 Arm B, deployable)"},
            "C": {"resume_j": args.resume_j_c,
                  "note": "ORACLE (NOT deployable): upper-24 replay from CONTINUOUS "
                          "pack-level h12 (layers[0:12] run continuously/full-causal "
                          "over the whole selected pack; layer-12 state == stock "
                          "lower-12 forward, verified by --mode h12_sanity)."},
        },
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
        print("[p1.7][manifest][ABORT] strict-fix mismatch:", flush=True)
        for a in abort:
            print("   - " + a, flush=True)
        sys.exit(3)
    print(f"[p1.7][manifest] OK — LoRA sha {lora_sha256[:12]}… "
          f"{prov_lora['count']} modules, layers [12..35]; "
          f"torch {prov_versions['torch']} tf {prov_versions['transformers']} "
          f"peft {prov_versions['peft']} git {prov_versions['git_commit_short']}",
          flush=True)


# --------------------------------------------------------------------------- #
# AGGREGATE mode (pure CPU: 3-arm per-cell + pairwise paired stats)
# --------------------------------------------------------------------------- #
def _macro_and_cells(valid, arm):
    cells = {}
    for r in valid:
        cells.setdefault((r["task"], r["length"]), []).append(r)
    per_cell = {}
    macro = []
    for key, recs in sorted(cells.items()):
        m = sum(x[arm]["score"] for x in recs) / len(recs) * 100.0
        per_cell[f"{key[0]}/{key[1]}"] = round(m, 2)
        macro.append(m)
    return (sum(macro) / len(macro) if macro else 0.0), per_cell, cells


def _pairwise(valid, cells_keys, cells, arm_x, arm_y, n_boot, seed=0):
    """Paired macro diff (X-Y) bootstrap 95% CI (resample examples within cells,
    recompute cell-macro) + exact McNemar on discrete correctness."""
    diff_by_cell = {
        c: [(r[arm_x]["score"] - r[arm_y]["score"]) * 100.0 for r in cells[c]]
        for c in cells_keys
    }
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        cm = []
        for c in cells_keys:
            dv = diff_by_cell[c]; nn = len(dv)
            s = sum(dv[rng.randrange(nn)] for _ in range(nn))
            cm.append(s / nn)
        boot.append(sum(cm) / len(cm))
    boot.sort()
    macro_diff = sum(sum(diff_by_cell[c]) / len(diff_by_cell[c])
                     for c in cells_keys) / len(cells_keys)
    ci_lo = boot[int(0.025 * n_boot)]; ci_hi = boot[int(0.975 * n_boot)]
    b = sum(1 for r in valid if r[arm_x]["correct"] and not r[arm_y]["correct"])
    c_ = sum(1 for r in valid if not r[arm_x]["correct"] and r[arm_y]["correct"])
    both = sum(1 for r in valid if r[arm_x]["correct"] and r[arm_y]["correct"])
    neither = sum(1 for r in valid
                  if not r[arm_x]["correct"] and not r[arm_y]["correct"])
    return {
        "macro_diff": round(macro_diff, 4),
        "paired_bootstrap_95ci": [round(ci_lo, 4), round(ci_hi, 4)],
        "mcnemar": {"X_only_correct_b": b, "Y_only_correct_c": c_,
                    "both_correct": both, "neither_correct": neither,
                    "exact_two_sided_p": _mcnemar_exact(b, c_)},
    }


def _agree_means(valid, pair):
    n = len(valid)
    if not n:
        return {}
    ft = sum(1 for r in valid if r["agreement"][pair]["first_token_match"]) / n
    d1 = sum(r["agreement"][pair]["decode_top1_rate"] for r in valid) / n
    cos = [r["agreement"][pair]["first_token_cosine"] for r in valid
           if r["agreement"][pair]["first_token_cosine"] is not None]
    return {"first_token_match_rate": round(ft, 4),
            "decode_top1_rate_mean": round(d1, 4),
            "first_token_cosine_mean": round(sum(cos) / len(cos), 4) if cos else None}


def run_aggregate(args):
    outdir = Path(args.output_dir)
    qdir = outdir / "quality"
    all_recs = []
    for jf in sorted(glob.glob(str(qdir / "*.jsonl"))):
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_recs.append(json.loads(line))
    valid = [r for r in all_recs if not r.get("oom")]
    seen = {}
    for r in valid:
        seen[(r["task"], r["length"], r["example_id"])] = r
    valid = list(seen.values())

    macroA, cellA, cells = _macro_and_cells(valid, "armA")
    macroB, cellB, _ = _macro_and_cells(valid, "armB")
    macroC, cellC, _ = _macro_and_cells(valid, "armC")
    cells_keys = sorted(cells.keys())

    per_cell = {}
    for key in cells_keys:
        tag = f"{key[0]}/{key[1]}"
        per_cell[tag] = {
            "n": len(cells[key]),
            "armA": cellA[tag], "armB": cellB[tag], "armC": cellC[tag],
            "diff_A_minus_B": round(cellA[tag] - cellB[tag], 2),
            "diff_A_minus_C": round(cellA[tag] - cellC[tag], 2),
            "diff_C_minus_B": round(cellC[tag] - cellB[tag], 2),
        }

    pairwise = {
        "A_vs_B": _pairwise(valid, cells_keys, cells, "armA", "armB", args.n_boot),
        "A_vs_C": _pairwise(valid, cells_keys, cells, "armA", "armC", args.n_boot),
        "C_vs_B": _pairwise(valid, cells_keys, cells, "armC", "armB", args.n_boot),
    }
    agreement = {
        "A_vs_B": _agree_means(valid, "A_vs_B"),
        "A_vs_C": _agree_means(valid, "A_vs_C"),
        "B_vs_C": _agree_means(valid, "B_vs_C"),
    }

    n = len(valid)
    any_oom = sum(1 for r in all_recs if r.get("oom"))
    any_nonfinite = sum(1 for r in valid if not (r["armA"]["finite"]
                        and r["armB"]["finite"] and r["armC"]["finite"]))
    pack_paired = all(r["armA"]["read_len"] == r["armB"]["read_len"]
                      == r["armC"]["read_len"] == r["pack_read_len"] for r in valid)
    p013_checked = [r for r in valid if r.get("p013_pack_sha_match") is not None]
    p013_all_match = all(r["p013_pack_sha_match"] for r in p013_checked) \
        if p013_checked else None

    summary = {
        "n_examples_paired": n, "n_cells": len(cells_keys),
        "per_cell": per_cell,
        "macro": {"armA_full_replay": round(macroA, 3),
                  "armB_chunk_local_h12": round(macroB, 3),
                  "armC_continuous_h12_oracle": round(macroC, 3),
                  "diff_A_minus_B": round(macroA - macroB, 3),
                  "diff_A_minus_C": round(macroA - macroC, 3),
                  "diff_C_minus_B": round(macroC - macroB, 3)},
        "oom_examples": any_oom, "nonfinite_examples": any_nonfinite,
        "all_packs_paired_1to1": pack_paired,
        "p013_pack_sha_checked": len(p013_checked),
        "p013_pack_sha_all_match": p013_all_match,
        "attribution_hint": (
            "C≈A (and C>>B) => A-B gap is chunk-local Write/repositioning; "
            "C≈B (and C<<A) => gap is the lower-12 skip / resume interface. "
            "Oracle is NOT deployable (per-query lower-12 recompute)."),
    }
    stats = {"macro": summary["macro"], "pairwise": pairwise,
             "agreement": agreement, "bootstrap_n": args.n_boot}
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(outdir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # latency aggregation (3 arms) across procs
    lat_procs = []
    for lf in sorted(glob.glob(str(outdir / "latency" / "latency_proc*.json"))):
        with open(lf) as f:
            lat_procs.append(json.load(f))
    latency = {"n_procs": len(lat_procs), "procs": []}
    if lat_procs:
        for lp in lat_procs:
            latency["procs"].append({
                "proc_id": lp["proc_id"], "task": lp["task"], "length": lp["length"],
                "armA_read_ms": round(lp["armA"]["read_s"]["median"] * 1e3, 3),
                "armB_read_ms": round(lp["armB"]["read_s"]["median"] * 1e3, 3),
                "armC_read_ms": round(lp["armC"]["read_s"]["median"] * 1e3, 3),
                "armC_write_ms": round(lp["armC"]["write_s"]["median"] * 1e3, 3),
            })

        def _pool(arm, phase):
            raw = []
            for lp in lat_procs:
                raw.extend(lp[arm][phase]["raw"])
            return _summ(raw)
        latency["pooled"] = {a: {p: _pool(a, p) for p in
                                 ("read_s", "write_s", "decode_s", "total_s")}
                             for a in ("armA", "armB", "armC")}
    with open(outdir / "latency.json", "w") as f:
        json.dump(latency, f, indent=2)

    print("=" * 74)
    print(f"[p1.7][aggregate] n_paired={n} n_cells={len(cells_keys)}")
    print(f"  macro  A(full)={macroA:.2f}  B(chunk-local)={macroB:.2f}  "
          f"C(oracle-continuous)={macroC:.2f}")
    for name, pk in (("A-B", "A_vs_B"), ("A-C", "A_vs_C"), ("C-B", "C_vs_B")):
        pw = pairwise[pk]
        print(f"  {name}: diff={pw['macro_diff']:+.2f} "
              f"CI={pw['paired_bootstrap_95ci']} "
              f"McNemar p={pw['mcnemar']['exact_two_sided_p']:.3g}")
    print(f"  packs_paired_1to1={pack_paired} p013_sha_match={p013_all_match} "
          f"oom={any_oom} nonfinite={any_nonfinite}")
    print("=" * 74, flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="P1.7 continuous-prefix h12 oracle")
    ap.add_argument("--mode", required=True,
                    choices=["manifest", "quality", "latency", "aggregate",
                             "h12_sanity"])
    ap.add_argument("--model_path", type=str, default="models/Qwen3-8b-local")
    ap.add_argument("--lora_adapter", type=str,
                    default="outputs/qcmem_distill_qwen_j12_r32_4k/final")
    ap.add_argument("--resume_j_a", type=int, default=0)    # full replay
    ap.add_argument("--resume_j_b", type=int, default=12)   # chunk-local h12
    ap.add_argument("--resume_j_c", type=int, default=12)   # oracle continuous h12
    ap.add_argument("--task", type=str, default="niah_multikey_1")
    ap.add_argument("--length", type=str, default="8k")
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
                    default="bench_results/p1_7_h12_oracle")
    ap.add_argument("--p013_manifest_dir", type=str, default="",
                    help="optional P0.13 output dir to cross-check pack shas against")
    # h12 sanity / verify
    ap.add_argument("--verify", action="store_true",
                    help="in quality mode, run the h12 invariant assert on the "
                         "first processed example before scoring")
    ap.add_argument("--h12_tol", type=float, default=5e-2,
                    help="max-abs bf16 tolerance for the continuous-prefix h12 "
                         "invariant (report actual residual regardless)")
    ap.add_argument("--example_index", type=int, default=0)
    # latency mode
    ap.add_argument("--proc_id", type=int, default=0)
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
    print(f"P1.7 :: mode={args.mode} task={args.task} length={args.length} "
          f"shard={args.shard_index}/{args.num_shards}")
    print(f"  model={args.model_path} lora={args.lora_adapter}")
    print(f"  A=j{args.resume_j_a}(full) B=j{args.resume_j_b}(chunk-local) "
          f"C=j{args.resume_j_c}(oracle-continuous) topk={args.topk} "
          f"hop={args.iter_hop_topk} chunk={args.chunk_size} dtype={dtype} "
          f"attn={args.attn_impl} device={device}")
    print("=" * 78, flush=True)

    if args.mode == "manifest":
        run_manifest(args, device, dtype)
    elif args.mode == "quality":
        run_quality(args, device, dtype)
    elif args.mode == "latency":
        run_latency(args, device, dtype)
    elif args.mode == "h12_sanity":
        run_h12_sanity(args, device, dtype)


if __name__ == "__main__":
    main()
