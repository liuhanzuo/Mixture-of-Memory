#!/usr/bin/env python
"""P0.17 — E2 overlapping-chunk Write (INFERENCE-ONLY, paired; zero training).

Conditional follow-up to P0.16. P0.16 established that the deployable chunk-local
Write (Arm B = 92.5 pooled) loses +7.5pp vs the document-contextual control E0
(== full replay A == continuous oracle C, 100/100/100), attributed ENTIRELY to
chunk-local Write LACKING document context (the Read interface / RoPE repositioning
is near-lossless, McNemar b=15/c=0 for E0>B). ⇒ inject a small amount of document
context into the *persistent* Write while keeping the deployable Read UNCHANGED.

E2 = overlapping-chunk Write. When encoding a 512-token context chunk to depth 12,
PREPEND the ``w`` tokens immediately preceding it in the ORIGINAL DOCUMENT
(``w ∈ {32,64,128}``), run layers[0:12) over the (w+512) span CHUNK-LOCALLY
(isolated causal, RoPE 0:(w+512)), then DISCARD the first ``w`` prefix hidden states
and store ONLY the original 512-token chunk's ``h12``. Everything else — the sink
write, the query write, the store pack layout, the persistent bytes/token, the Read
(fresh contiguous pack positions from layer 12), and the O(1) two-coordinate decode
— is BIT-IDENTICAL to the deployable Arm B. The ONLY change vs B is a one-time,
per-chunk, longer lower-12 Write forward (the extra cost is the ``w`` prefix tokens,
which never enter the store, the pack, the Read, or the decode).

Six strictly-paired arms (all consume the IDENTICAL ``_build_pack`` pack per
example; A / B / E0 are BYTE-FOR-BYTE the P0.16 / P1.7 / P0.13 headline paths,
imported verbatim from the unmodified harnesses):

  * A       — ``resume_j=0`` full 36-layer continuous replay + flagship LoRA
              (RAG upper anchor; == P0.16/P1.7/P0.13 Arm A). ``p017._run_arm``.
  * B (w0)  — ``resume_j=12`` DEPLOYABLE chunk-local h12 Write + SAME LoRA
              (== P0.16/P1.7/P0.13 Arm B). This IS the E2 ``w=0`` baseline: E2 with
              zero left-context degenerates to Arm B by construction (the numeric
              gate below asserts ``_e2_write_chunk(w=0) == write_chunk`` max_abs 0).
              ``p017._run_arm``.
  * E2_w32  — overlapping Write, left-context width 32. ``_run_e2``.
  * E2_w64  — overlapping Write, left-context width 64. ``_run_e2``.
  * E2_w128 — overlapping Write, left-context width 128. ``_run_e2``.
  * E0      — DOCUMENT-CONTEXTUAL Write control (== P0.16 Arm E0; O(L),
              cross-query-reusable control, NOT a shipping config nor a strict upper
              bound). ``p016._run_e0``.

  ALL FOUR widths are reported (w0 baseline + w32/w64/w128); reporting only the best
  ``w`` is FORBIDDEN by the pre-registration. Each E2_w is compared paired against
  BOTH the ``w0`` deployable baseline (B) AND the E0 control, with a paired bootstrap
  95% CI + exact McNemar, PLUS Write latency / peak GPU mem and the extra
  Write-span tokens (≈ extra lower-12 FLOPs) vs w0. The pre-registered primary
  target is ``multikey pooled 92.5 → ≥97.0`` at UNCHANGED store/Read cost; the
  result is reported in full regardless of sign (a small or negative effect is a
  boundary / negative result, per the spec).

Pack pairing (forward-free, resume_j-independent, == P0.16/P1.7/P0.13):
   the flagship RULER selector ``iter_bm25`` is pure lexical BM25 over raw token ids
   (no model forward) ⇒ selected chunk ids / order / packed ids / pack sha are
   bit-identical to P0.13/P1.7. The pack is built ONCE per example with the
   UNMODIFIED ``p017._build_pack`` and every arm reads that identical pack.
   ``--p013_manifest_dir bench_results/p1_7_h12_oracle`` fail-closed cross-checks each
   pack sha against the SAME 200 paired examples used by P1.7 / P0.16.

FAIL-CLOSED invariants (any violation hard-aborts):
   * ``_e2_write_chunk`` slice provenance: the doc-origin chunk span
     ``doc_ids[ci*cs : ci*cs+cs]`` MUST equal the pack's selected-chunk tensor
     bit-for-bit; the left-context span MUST sit strictly before the chunk.
   * E2 stored chunk h12 shape == ``[1, chunk_size, hidden_size]`` (prefix discarded).
   * read_len parity: every arm's read_len == ``pack_read_len`` (1:1 pairing).
   * E2 ``w=0`` identity: ``_e2_write_chunk(chunk, no-prefix)`` == ``write_chunk(chunk)``
     max_abs 0 (so the w0 baseline B is EXACTLY the deployable endpoint).
   * E0 h12 residual: E0 document-contextual lower-12 == stock lower-12 on a doc
     prefix (reused verbatim from P0.16; bf16 max_abs < ``--h12_tol``).
   * ``packed_ids_sha256`` == P1.7 manifest sha (when ``--p013_manifest_dir`` given).

This file NEVER mutates ``eval_p016_e0_write_control.py``, ``bench_p1_7_h12_oracle``,
``bench_p0_13_quality_latency`` or ``qcmem_model.py`` — it imports the P0.16 / P1.7
primitives READ-ONLY (so A/B/C/E0 stay bit-identical) and adds ONLY the E2 overlap
Write forward (which uses QCMemModel's public low-level accessors ``embed_tokens``,
``_make_mask_and_rope``, ``_run_layers``, ``write_chunk``, ``write_prefill``,
``read_prefill``, ``decode_step`` — no backbone patching, no shared-module edits).
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

# Import the UNMODIFIED P0.16 harness (which imports P1.7 -> P0.13 verbatim) and the
# P1.7 harness, and pull every shared primitive from them — so arms A / B / E0 are
# BIT-IDENTICAL to the P0.16 / P1.7 / P0.13 headline paths (same pack builder, same
# per-arm generate replica, same E0 doc-contextual Write, same loader, same
# strict-fix hashes, same stats). The ONLY new forward here is E2's overlap Write.
import bench_p1_7_h12_oracle as p017            # noqa: E402
import eval_p016_e0_write_control as p016        # noqa: E402

ruler = p017.ruler
qcb = p017.qcb
QCMemModel = p017.QCMemModel
_bare_question = p017._bare_question
_resolve_task = p017._resolve_task
_build_pack = p017._build_pack
_packed_ids_from_pack = p017._packed_ids_from_pack
_run_arm = p017._run_arm                        # arms A (j0) + B (chunk-local j12 = w0)
_run_e0 = p016._run_e0                           # E0 document-contextual control
_e0_doc_spans = p016._e0_doc_spans               # E0 slicing map (reused verbatim)
_e0_h12_residual = p016._e0_h12_residual         # E0 numeric invariant (reused verbatim)
_load = p017._load
_eos_ids = p017._eos_ids
_sync = p017._sync
_peak_gb = p017._peak_gb
_pair_agree = p017._pair_agree
_macro_and_cells = p017._macro_and_cells
_pairwise = p017._pairwise
_agree_means = p017._agree_means
_mcnemar_exact = p017._mcnemar_exact
_backbone_provenance = p017._backbone_provenance
_lora_modules = p017._lora_modules
_versions = p017._versions
EXPECTED_LORA_SHA = p017.EXPECTED_LORA_SHA
EXPECTED_BACKBONE_KEY_SHA = p017.EXPECTED_BACKBONE_KEY_SHA
EXPECTED_LORA_MODULE_COUNT = p017.EXPECTED_LORA_MODULE_COUNT

DEFAULT_WIDTHS = (32, 64, 128)


def _arm_key(w: int) -> str:
    return f"armE2_w{w}"


# --------------------------------------------------------------------------- #
# E2 left-context provenance: map each selected chunk back onto ORIGINAL DOCUMENT
# token coordinates and slice the ``w`` preceding tokens. Fail-closed on any
# mismatch (the doc-origin chunk MUST equal the pack's selected-chunk tensor).
# --------------------------------------------------------------------------- #
def _e2_left_ctx(doc_ids, sel_idx, selected_chunk_tensors, chunk_size, w):
    """Return the ordered list of left-context tensors (each ``[w']`` or ``None``)
    for the selected context chunks, verifying that the doc-origin chunk span equals
    the pack's stored chunk tensor bit-for-bit.

    For selected chunk ``ci`` (document span ``[ci*cs, ci*cs+cs)``), the left context
    is ``doc_ids[max(0, ci*cs - w) : ci*cs]`` (the up-to-``w`` tokens immediately
    preceding the chunk in the ORIGINAL document; empty ⇒ ``None``, e.g. chunk 0)."""
    if len(sel_idx) != len(selected_chunk_tensors):
        raise AssertionError(
            f"[p0.17][E2] sel_idx/tensor mismatch {len(sel_idx)} != "
            f"{len(selected_chunk_tensors)}")
    N = int(doc_ids.shape[0])
    out = []
    for k, ci in enumerate(sel_idx):
        start = int(ci) * chunk_size
        end = start + chunk_size
        if start < 0 or end > N:
            raise AssertionError(
                f"[p0.17][E2] chunk {ci} span [{start},{end}) escapes doc [0,{N})")
        # doc-origin chunk MUST match the pack's stored chunk tensor bit-for-bit.
        doc_chunk = doc_ids[start:end]
        pack_chunk = selected_chunk_tensors[k].to(doc_chunk.device)
        if doc_chunk.shape[0] != pack_chunk.shape[0] or \
                not bool(torch.equal(doc_chunk, pack_chunk)):
            raise AssertionError(
                f"[p0.17][E2] chunk {ci} doc slice != pack selected tensor "
                f"(len {doc_chunk.shape[0]} vs {pack_chunk.shape[0]}) — slicing broke")
        lc_start = max(0, start - int(w))
        lc = doc_ids[lc_start:start]
        out.append(lc if int(lc.shape[0]) > 0 else None)
    return out


# --------------------------------------------------------------------------- #
# E2 overlap Write for ONE chunk: chunk-local lower-12 forward with a w-token left
# prefix; DISCARD the prefix hidden states, keep only the chunk's h12.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _e2_write_chunk(qc, chunk_ids, left_ctx_ids):
    """Encode ONE context chunk to depth ``resume_j`` with a ``w``-token LEFT-CONTEXT
    prefix, then discard the prefix hidden states and return the chunk's ``h12``.

    Runs ``embed_tokens`` + ``layers[0:resume_j]`` over the concatenated
    ``[left_ctx ; chunk]`` span with a chunk-local causal mask and RoPE positions
    ``0:(w'+T)`` (the span is contextualised in ISOLATION, exactly like
    ``write_chunk`` but over the longer span), then returns ``h12[:, w':, :]`` — the
    original chunk's depth-``resume_j`` hidden ``[1, T, d]``. With ``left_ctx_ids is
    None`` (w'=0) this is BIT-IDENTICAL to ``qc.write_chunk(chunk_ids)`` (the w0
    identity gate asserts max_abs 0)."""
    if left_ctx_ids is None or int(left_ctx_ids.shape[-1]) == 0:
        return qc.write_chunk(chunk_ids)          # w'=0 ⇒ exactly Arm B's write
    lc = qc._as_ids(left_ctx_ids)                 # [1, w']
    ch = qc._as_ids(chunk_ids)                    # [1, T]
    w_pref = int(lc.shape[1])
    span = torch.cat([lc, ch], dim=1)             # [1, w'+T]
    T_total = span.shape[1]
    emb = qc.embed_tokens(span)
    positions = torch.arange(T_total, device=qc.device).unsqueeze(0)
    mask, pe = qc._make_mask_and_rope(emb, positions)
    h = qc._run_layers(emb, slice(0, qc.resume_j), mask, positions, pe)  # [1,w'+T,d]
    return h[:, w_pref:, :]                        # [1, T, d] — prefix discarded


@torch.no_grad()
def _e2_write_residual(qc, chunk_ids):
    """w=0 identity gate: ``_e2_write_chunk(chunk, no-prefix)`` MUST equal
    ``qc.write_chunk(chunk)`` bit-for-bit (both are the isolated chunk-local lower-12
    forward). This proves the E2 ``w=0`` operating point is EXACTLY the deployable
    Arm B, so the reported ``w0`` baseline is not a re-implementation drift."""
    a = _e2_write_chunk(qc, chunk_ids, None).float()
    b = qc.write_chunk(chunk_ids).float()
    diff = (a - b).abs()
    return {"max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "T": int(a.shape[1])}


# --------------------------------------------------------------------------- #
# E2 arm generate: overlap-Write chunk h12 -> SAME pack/Read/decode as Arm B.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _run_e2(qc, bos_id, selected_chunk_tensors, selected_left_ctx, query_ids,
           chunk_size, max_new_tokens, eos_ids, capture_first=False):
    """Run the E2 overlapping-chunk Write arm.

    WRITE: sink = chunk-local ``write_chunk([bos])`` (== Arm B); each selected
    context chunk = ``_e2_write_chunk`` (chunk-local lower-12 over the w-token-prefixed
    span, prefix discarded); query = chunk-local ``write_prefill`` (== Arm B, keeps a
    bottom KV cache for O(1) decode). READ + DECODE: IDENTICAL to Arm B (fresh
    contiguous pack positions from layer 12, two-coordinate O(1) decode). The store
    pack, persistent bytes/token and Read compute are therefore bit-identical to B;
    only the one-time Write forward is longer (by the ``w`` prefix tokens).

    Returns the SAME 6-tuple shape as ``p017._run_arm`` so the caller treats all arms
    uniformly, PLUS the write-span token accounting is computed by the caller."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- WRITE phase (E2): sink + overlap-Write context chunks + query prefill ----
    _sync(); tw0 = time.perf_counter()
    sink_hj = qc.write_chunk([bos_id])
    selected_hj = []
    for ch, lc in zip(selected_chunk_tensors, selected_left_ctx):
        h = _e2_write_chunk(qc, ch, lc)
        # fail-closed: prefix discarded ⇒ stored h12 shape == [1, chunk_len, d].
        exp_t = int(ch.shape[0]) if torch.is_tensor(ch) else len(ch)
        if h.dim() != 3 or int(h.shape[0]) != 1 or int(h.shape[1]) != exp_t \
                or int(h.shape[2]) != qc.hidden_size:
            raise AssertionError(
                f"[p0.17][E2] stored chunk h12 shape {tuple(h.shape)} != "
                f"[1,{exp_t},{qc.hidden_size}] — prefix not discarded / bad slice")
        selected_hj.append(h)
    q_hj, bottom_cache, q_local_pos = qc.write_prefill(query_ids)
    _sync(); t_write = time.perf_counter() - tw0

    # ---- READ phase (E2): resume the top band over the repositioned pack (== B) ----
    _sync(); tr0 = time.perf_counter()
    logits1, top_cache, pack_pos = qc.read_prefill(sink_hj, selected_hj, q_hj)
    _sync(); t_read = time.perf_counter() - tr0

    read_len = (int(sink_hj.shape[1]) if sink_hj is not None else 0) \
        + int(sum(h.shape[1] for h in selected_hj)) + len(query_ids)

    first_logits = logits1[0, -1].float()
    finite = bool(torch.isfinite(first_logits).all().item())
    next_logits = first_logits.clone()
    if eos_ids:
        next_logits[eos_ids] = float("-inf")     # step 0 never emits EOS
    first_capture = first_logits.detach().cpu().clone() if capture_first else None

    # ---- DECODE phase: O(1)/step, IDENTICAL to Arm B (two-coordinate) ----
    _sync(); td0 = time.perf_counter()
    generated = []
    next_tok = int(next_logits.argmax().item())
    generated.append(next_tok)
    for _step in range(1, max_new_tokens):
        logits = qc.decode_step(next_tok, bottom_cache, top_cache,
                                q_local_pos, pack_pos)
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

    peak = _peak_gb() if torch.cuda.is_available() else 0.0
    timings = {"write_s": t_write, "read_s": t_read, "decode_s": t_decode,
               "total_s": t_write + t_read + t_decode}
    return generated, timings, read_len, peak, finite, first_capture


def _write_span_tokens(sel_idx, selected_chunk_tensors, chunk_size, w):
    """Total lower-12 Write-span tokens for the CONTEXT chunks at left-width ``w``
    (sink + query are identical across arms, so excluded from the delta). w=0 ⇒
    sum of chunk lengths (== Arm B). Extra vs w0 = sum of clamped prefix widths."""
    total = 0
    prefix = 0
    for k, ci in enumerate(sel_idx):
        start = int(ci) * chunk_size
        t = int(selected_chunk_tensors[k].shape[0])
        wp = min(int(w), start)               # clamp at document start
        total += wp + t
        prefix += wp
    return total, prefix


# --------------------------------------------------------------------------- #
# QUALITY mode (6 arms on the identical pack + identical document)
# --------------------------------------------------------------------------- #
def run_quality(args, device, dtype):
    torch.manual_seed(args.seed)
    widths = [int(w) for w in args.widths]
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L = int(model.config.num_hidden_layers)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p0.17][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    qcA = QCMemModel(model, resume_j=args.resume_j_a)      # 0  full replay (anchor)
    qcB = QCMemModel(model, resume_j=args.resume_j_b)      # 12 chunk-local (w0 baseline)
    qcE2 = QCMemModel(model, resume_j=args.resume_j_b)     # 12 overlap Write (all widths)
    qcE0 = QCMemModel(model, resume_j=args.resume_j_e0)    # 12 document-contextual control
    eosA = _eos_ids(qcA, tokenizer)
    eosB = _eos_ids(qcB, tokenizer)
    eosE2 = _eos_ids(qcE2, tokenizer)
    eosE0 = _eos_ids(qcE0, tokenizer)

    task = _resolve_task(args.task)
    length = args.length
    if length not in ruler._LENGTH_TOKENS:
        raise SystemExit(f"[p0.17] unknown length {length}")
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

    # Optional cross-check against the P0.13 / P1.7 pack manifest (same examples).
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

    print(f"[p0.17][quality] {task}/{length}{shard_tag}: selector={sel_name} "
          f"topk={args.topk} hop={args.iter_hop_topk} n={len(sample_indices)}/"
          f"{args.limit} A=j{args.resume_j_a} B(w0)=j{args.resume_j_b} "
          f"E2 widths={widths} E0=j{args.resume_j_e0} mnt={mnt} verify={args.verify}",
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
        doc_ids = input_ids[0]                       # full ORIGINAL document ids
        doc_len = int(doc_ids.shape[0])
        doc_sha = hashlib.sha256(
            b",".join(str(int(t)).encode() for t in doc_ids.tolist())).hexdigest()

        # ---- build the pack ONCE (forward-free, resume_j-independent) ----
        t_ret0 = time.perf_counter()
        pack = _build_pack(input_ids, args.chunk_size, sel_name, args.topk,
                           args.iter_hop_topk, bare_q_ids, tokenizer)
        retrieval_s = time.perf_counter() - t_ret0
        packed_ids = _packed_ids_from_pack(pack)     # verifies pack sha internally

        sel_idx = pack["sel_idx"]
        sel_tensors = pack["selected_chunk_tensors"]

        # ---- E0 document-origin slicing map (fail-closed; reused from P0.16) ----
        sink_span, chunk_spans, query_span = _e0_doc_spans(
            pack, args.chunk_size, doc_len)

        # ---- E2 left-context tensors per width (fail-closed slice provenance) ----
        left_ctx_by_w = {
            w: _e2_left_ctx(doc_ids, sel_idx, sel_tensors, args.chunk_size, w)
            for w in widths
        }
        write_span_by_w = {
            "w0": _write_span_tokens(sel_idx, sel_tensors, args.chunk_size, 0),
        }
        for w in widths:
            write_span_by_w[f"w{w}"] = _write_span_tokens(
                sel_idx, sel_tensors, args.chunk_size, w)

        # cross-check pack sha vs P0.13 / P1.7 (if a manifest dir was given)
        p013_sha_match = None
        if i in p013_shas:
            p013_sha_match = (p013_shas[i] == pack["packed_ids_sha256"])
            if not p013_sha_match:
                raise AssertionError(
                    f"[p0.17] example {i}: pack sha != P0.13/P1.7 pack sha "
                    "— pairing with the existing arms broken")

        # ---- numeric gates on the FIRST processed example (both cheap, exact) ----
        h12_check = None
        e2_w0_check = None
        if args.verify and not verified_once:
            # (a) E0 document-contextual invariant (reused verbatim from P0.16).
            pfx = min(int(args.h12_check_prefix), doc_len)
            h12_check = _e0_h12_residual(qcE0, doc_ids[:pfx])
            h12_check["prefix_tokens"] = pfx
            print(f"[p0.17][verify] example {i}: E0 doc-lower12 vs stock (prefix="
                  f"{pfx}) max_abs={h12_check['max_abs']:.3e} "
                  f"mean_abs={h12_check['mean_abs']:.3e} "
                  f"(ref_abs_max={h12_check['ref_abs_max']:.3e}) "
                  f"tol={args.h12_tol:.3e}", flush=True)
            assert h12_check["max_abs"] < args.h12_tol, (
                f"[p0.17][ABORT] E0 h12 residual {h12_check['max_abs']:.3e} >= "
                f"tol {args.h12_tol:.3e} — document-contextual invariant violated")
            # (b) E2 w=0 identity: overlap-Write with no prefix == chunk-local write.
            if sel_tensors:
                e2_w0_check = _e2_write_residual(qcE2, sel_tensors[0])
                print(f"[p0.17][verify] example {i}: E2 w=0 write vs write_chunk "
                      f"max_abs={e2_w0_check['max_abs']:.3e} "
                      f"mean_abs={e2_w0_check['mean_abs']:.3e} "
                      f"(T={e2_w0_check['T']}) tol={args.e2_w0_tol:.3e}", flush=True)
                assert e2_w0_check["max_abs"] < args.e2_w0_tol, (
                    f"[p0.17][ABORT] E2 w=0 write residual {e2_w0_check['max_abs']:.3e}"
                    f" >= tol {args.e2_w0_tol:.3e} — w0 baseline != deployable Arm B")
            verified_once = True

        # ---- run the arms on the identical pack / identical document ----
        oom = False
        try:
            genA, tA, rlA, pkA, finA, lA = _run_arm(
                qcA, tokenizer, pack["bos_id"], sel_tensors,
                pack["query_ids"], mnt, eosA, capture_first=True)
            genB, tB, rlB, pkB, finB, lB = _run_arm(
                qcB, tokenizer, pack["bos_id"], sel_tensors,
                pack["query_ids"], mnt, eosB, capture_first=True)
            e2_out = {}
            for w in widths:
                e2_out[w] = _run_e2(
                    qcE2, pack["bos_id"], sel_tensors, left_ctx_by_w[w],
                    pack["query_ids"], args.chunk_size, mnt, eosE2,
                    capture_first=True)
            genE0, tE0, rlE0, pkE0, finE0, lE0 = _run_e0(
                qcE0, doc_ids, sink_span, chunk_spans, query_span,
                mnt, eosE0, capture_first=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            oom = True
            torch.cuda.empty_cache()
            print(f"[p0.17][OOM] i={i} {task}/{length}: {e}", flush=True)

        if oom:
            rec = {"example_id": i, "task": task, "length": length,
                   "oom": True, "gold": " | ".join(answers)}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            records.append(rec); n_done += 1
            continue

        predA = tokenizer.decode(genA, skip_special_tokens=True).strip()
        predB = tokenizer.decode(genB, skip_special_tokens=True).strip()
        predE0 = tokenizer.decode(genE0, skip_special_tokens=True).strip()
        recA = ruler._string_match_all_one(predA, answers)
        recB = ruler._string_match_all_one(predB, answers)
        recE0 = ruler._string_match_all_one(predE0, answers)

        # 1:1 pairing guard: ALL arms MUST have consumed the identical pack length.
        pack_rl = pack["pack_read_len"]
        assert rlA == rlB == rlE0 == pack_rl, \
            (f"read_len mismatch i={i}: A={rlA} B={rlB} E0={rlE0} pack={pack_rl}")
        for w in widths:
            rlE2 = e2_out[w][2]
            assert rlE2 == pack_rl, \
                f"read_len mismatch i={i} E2_w{w}: {rlE2} != pack {pack_rl}"

        rec = {
            "example_id": i, "task": task, "length": length,
            "approx_tokens": approx_tokens,
            "gold": " | ".join(answers), "n_refs": len(answers),
            "doc_len": doc_len, "doc_ids_sha256": doc_sha,
            "retrieved_chunk_ids": sel_idx,
            "n_ctx_chunks": pack["n_ctx_chunks"],
            "chunk_size": args.chunk_size,
            "widths": widths,
            "e0_doc_slices": {
                "sink_span": list(sink_span),
                "chunk_spans": [list(s) for s in chunk_spans],
                "query_span": list(query_span),
            },
            # Write-span token accounting: (total_ctx_tokens, extra_prefix_tokens).
            # E2 adds ONLY the extra_prefix_tokens to the one-time lower-12 Write; the
            # store pack / persistent bytes / Read / decode are unchanged vs w0.
            "write_span_tokens": {
                k: {"total_ctx_tokens": v[0], "prefix_tokens": v[1]}
                for k, v in write_span_by_w.items()
            },
            "pack_token_count": pack["pack_token_count"],
            "pack_read_len": pack_rl,
            "packed_ids_sha256": pack["packed_ids_sha256"],
            "p013_pack_sha_match": p013_sha_match,
            "lora_sha256": lora_sha256,
            "retrieval_s": retrieval_s,
            "h12_sanity": h12_check,
            "e2_w0_identity": e2_w0_check,
            "rope_positions": {
                "e2_read_pack_positions": [0, pack_rl],
                "e0_write_doc_positions": [0, doc_len],
                "e0_read_pack_positions": [0, pack_rl],
            },
            "armA": {"resume_j": args.resume_j_a, "kind": "full_replay",
                     "prediction": predA, "score": recA,
                     "correct": bool(recA >= 1.0), "gen_len": len(genA),
                     "read_len": rlA, "latency_s": tA, "peak_gb": pkA,
                     "finite": finA},
            "armB": {"resume_j": args.resume_j_b, "kind": "chunk_local_h12_w0",
                     "prediction": predB, "score": recB,
                     "correct": bool(recB >= 1.0), "gen_len": len(genB),
                     "read_len": rlB, "latency_s": tB, "peak_gb": pkB,
                     "finite": finB},
            "armE0": {"resume_j": args.resume_j_e0, "kind": "document_contextual_write",
                      "prediction": predE0, "score": recE0,
                      "correct": bool(recE0 >= 1.0), "gen_len": len(genE0),
                      "read_len": rlE0, "latency_s": tE0, "peak_gb": pkE0,
                      "finite": finE0},
        }
        agree = {"A_vs_B": _pair_agree(genA, lA, genB, lB),
                 "E0_vs_B": _pair_agree(genE0, lE0, genB, lB)}
        for w in widths:
            genE, tE, rlE, pkE, finE, lE = e2_out[w]
            predE = tokenizer.decode(genE, skip_special_tokens=True).strip()
            recE = ruler._string_match_all_one(predE, answers)
            rec[_arm_key(w)] = {
                "resume_j": args.resume_j_b, "kind": "overlap_write", "w": w,
                "prediction": predE, "score": recE,
                "correct": bool(recE >= 1.0), "gen_len": len(genE),
                "read_len": rlE, "latency_s": tE, "peak_gb": pkE, "finite": finE,
                "extra_prefix_tokens": write_span_by_w[f"w{w}"][1],
            }
            rec[f"diff_E2_w{w}_minus_B"] = recE - recB
            rec[f"diff_E0_minus_E2_w{w}"] = recE0 - recE
            agree[f"E2_w{w}_vs_B"] = _pair_agree(genE, lE, genB, lB)
            agree[f"E2_w{w}_vs_E0"] = _pair_agree(genE, lE, genE0, lE0)
        rec["diff_A_minus_B"] = recA - recB
        rec["diff_E0_minus_B"] = recE0 - recB
        rec["agreement"] = agree

        fout.write(json.dumps(rec) + "\n"); fout.flush()
        records.append(rec); n_done += 1
        torch.cuda.empty_cache()
        if n_done % 5 == 0:
            e2str = " ".join(f"w{w}={rec[_arm_key(w)]['score']:.2f}" for w in widths)
            print(f"[p0.17][quality] {task}/{length}{shard_tag} {n_done} done "
                  f"(A={recA:.2f} B={recB:.2f} {e2str} E0={recE0:.2f} "
                  f"readlen={rlA})", flush=True)
    fout.close()

    valid = [r for r in records if not r.get("oom")]

    def _mean(arm):
        xs = [r[arm]["score"] for r in valid if arm in r]
        return round(sum(xs) / len(xs) * 100.0, 3) if xs else 0.0
    cell = {
        "task": task, "length": length, "shard": shard_tag,
        "n": len(records), "n_valid": len(valid),
        "oom_count": sum(1 for r in records if r.get("oom")),
        "widths": widths,
        "armA_score": _mean("armA"), "armB_score": _mean("armB"),
        "armE0_score": _mean("armE0"),
        "selector": sel_name, "topk": args.topk, "iter_hop_topk": args.iter_hop_topk,
        "chunk_size": args.chunk_size, "max_new_tokens": mnt,
        "resume_j_a": args.resume_j_a, "resume_j_b": args.resume_j_b,
        "resume_j_e0": args.resume_j_e0,
        "lora_sha256": lora_sha256, "num_layers": L,
        "runtime": {"node": socket.gethostname(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "device": args.device,
                    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                    "seed": args.seed, "dtype": args.dtype,
                    "attn_implementation": args.attn_impl},
        "jsonl": str(jsonl_path),
    }
    for w in widths:
        cell[f"armE2_w{w}_score"] = _mean(_arm_key(w))
        cell[f"diff_E2_w{w}_minus_B"] = round(
            _mean(_arm_key(w)) - _mean("armB"), 3)
    with open(outdir / f"{task}_{length}{shard_tag}_cell.json", "w") as f:
        json.dump(cell, f, indent=2)
    e2str = " ".join(f"w{w}={cell[f'armE2_w{w}_score']}" for w in widths)
    print(f"[p0.17][quality] DONE {task}/{length}{shard_tag}: "
          f"A={cell['armA_score']} B={cell['armB_score']} {e2str} "
          f"E0={cell['armE0_score']} (n_valid={len(valid)})", flush=True)


# --------------------------------------------------------------------------- #
# E2_SANITY mode (E2 w=0 identity + E0 doc-contextual invariant on one example)
# --------------------------------------------------------------------------- #
def run_e2_sanity(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p0.17][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    qcE2 = QCMemModel(model, resume_j=args.resume_j_b)
    qcE0 = QCMemModel(model, resume_j=args.resume_j_e0)

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
    doc_ids = input_ids[0]
    doc_len = int(doc_ids.shape[0])
    pack = _build_pack(input_ids, args.chunk_size, "iter_bm25", args.topk,
                       args.iter_hop_topk, bare_q_ids, tokenizer)
    sel_tensors = pack["selected_chunk_tensors"]

    # (a) E0 document-contextual invariant (reused verbatim from P0.16).
    pfx = min(int(args.h12_check_prefix), doc_len)
    e0_res = _e0_h12_residual(qcE0, doc_ids[:pfx])
    # (b) E2 w=0 identity: overlap-Write with no prefix == chunk-local write.
    if not sel_tensors:
        raise SystemExit(f"[p0.17][e2_sanity] example {i} selected 0 context chunks")
    e2_res = _e2_write_residual(qcE2, sel_tensors[0])

    print("=" * 72)
    print(f"[p0.17][e2_sanity] {task}/{length} example {i}: doc_len={doc_len} "
          f"n_ctx={pack['n_ctx_chunks']} n_sel={len(sel_tensors)}")
    print(f"  E0 document-contextual lower-{args.resume_j_e0} vs stock (prefix={pfx}):"
          f" max_abs={e0_res['max_abs']:.3e} mean_abs={e0_res['mean_abs']:.3e} "
          f"(ref_abs_max={e0_res['ref_abs_max']:.3e}) tol={args.h12_tol:.3e}")
    print(f"  E2 w=0 write vs write_chunk (T={e2_res['T']}): "
          f"max_abs={e2_res['max_abs']:.3e} mean_abs={e2_res['mean_abs']:.3e} "
          f"tol={args.e2_w0_tol:.3e}")
    print("=" * 72, flush=True)
    assert e0_res["max_abs"] < args.h12_tol, (
        f"[p0.17][ABORT] E0 h12 residual {e0_res['max_abs']:.3e} >= tol "
        f"{args.h12_tol:.3e} — document-contextual invariant violated")
    assert e2_res["max_abs"] < args.e2_w0_tol, (
        f"[p0.17][ABORT] E2 w=0 write residual {e2_res['max_abs']:.3e} >= tol "
        f"{args.e2_w0_tol:.3e} — w0 baseline != deployable Arm B")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "e2_sanity.json", "w") as f:
        json.dump({"task": task, "length": length, "example_index": i,
                   "resume_j_b": args.resume_j_b, "resume_j_e0": args.resume_j_e0,
                   "dtype": args.dtype, "e0_h12_residual": e0_res,
                   "e2_w0_residual": e2_res, "h12_tol": args.h12_tol,
                   "e2_w0_tol": args.e2_w0_tol, "passed": True}, f, indent=2)
    print("[p0.17][e2_sanity] PASS — E0 invariant + E2 w=0 identity within tol; "
          "wrote e2_sanity.json", flush=True)


# --------------------------------------------------------------------------- #
# LATENCY mode (per-arm write/read/decode timing on a fixed pack/document)
# --------------------------------------------------------------------------- #
def run_latency(args, device, dtype):
    torch.manual_seed(args.seed)
    widths = [int(w) for w in args.widths]
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L = int(model.config.num_hidden_layers)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p0.17][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    qcA = QCMemModel(model, resume_j=args.resume_j_a)
    qcB = QCMemModel(model, resume_j=args.resume_j_b)
    qcE2 = QCMemModel(model, resume_j=args.resume_j_b)
    qcE0 = QCMemModel(model, resume_j=args.resume_j_e0)
    eosA = _eos_ids(qcA, tokenizer); eosB = _eos_ids(qcB, tokenizer)
    eosE2 = _eos_ids(qcE2, tokenizer); eosE0 = _eos_ids(qcE0, tokenizer)

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
    doc_ids = input_ids[0]
    doc_len = int(doc_ids.shape[0])
    pack = _build_pack(input_ids, args.chunk_size, "iter_bm25", args.topk,
                       args.iter_hop_topk, bare_q_ids, tokenizer)
    packed_ids = _packed_ids_from_pack(pack)
    sel_idx = pack["sel_idx"]; sel_tensors = pack["selected_chunk_tensors"]
    sink_span, chunk_spans, query_span = _e0_doc_spans(pack, args.chunk_size, doc_len)
    left_ctx_by_w = {w: _e2_left_ctx(doc_ids, sel_idx, sel_tensors,
                                     args.chunk_size, w) for w in widths}
    span_by_w = {"w0": _write_span_tokens(sel_idx, sel_tensors, args.chunk_size, 0)}
    for w in widths:
        span_by_w[f"w{w}"] = _write_span_tokens(sel_idx, sel_tensors,
                                                args.chunk_size, w)

    from bench_p0_13_quality_latency import _summ  # identical summary schema

    def _time(fn):
        writes, reads, decodes, totals = [], [], [], []
        rl = pk = None
        for it in range(args.warmup + args.n_repeat):
            gen, t, rl, pk, fin, _ = fn()
            if it >= args.warmup:
                writes.append(t["write_s"]); reads.append(t["read_s"])
                decodes.append(t["decode_s"]); totals.append(t["total_s"])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {"write_s": _summ(writes), "read_s": _summ(reads),
                "decode_s": _summ(decodes), "total_s": _summ(totals),
                "read_len": rl, "peak_gb": pk}

    resA = _time(lambda: _run_arm(qcA, tokenizer, pack["bos_id"], sel_tensors,
                                  pack["query_ids"], mnt, eosA, capture_first=False))
    resB = _time(lambda: _run_arm(qcB, tokenizer, pack["bos_id"], sel_tensors,
                                  pack["query_ids"], mnt, eosB, capture_first=False))
    resE2 = {}
    for w in widths:
        resE2[f"w{w}"] = _time(
            lambda w=w: _run_e2(qcE2, pack["bos_id"], sel_tensors, left_ctx_by_w[w],
                                pack["query_ids"], args.chunk_size, mnt, eosE2,
                                capture_first=False))
    resE0 = _time(lambda: _run_e0(qcE0, doc_ids, sink_span, chunk_spans, query_span,
                                  mnt, eosE0, capture_first=False))

    e2_read_str = " ".join(f"E2_w{w}.read={resE2[f'w{w}']['read_s']['median']*1e3:.1f}ms"
                           for w in widths)
    e2_write_str = " ".join(
        f"E2_w{w}.write={resE2[f'w{w}']['write_s']['median']*1e3:.1f}ms" for w in widths)
    print(f"[p0.17][latency] proc={args.proc_id} {task}/{length} "
          f"A.read={resA['read_s']['median']*1e3:.1f}ms "
          f"B.read={resB['read_s']['median']*1e3:.1f}ms {e2_read_str} "
          f"| B.write={resB['write_s']['median']*1e3:.1f}ms {e2_write_str} "
          f"(E2 adds ONLY a one-time longer lower-12 Write; Read == B)", flush=True)

    outdir = Path(args.output_dir) / "latency"
    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": "latency", "proc_id": args.proc_id,
        "task": task, "length": length, "example_index": i, "doc_len": doc_len,
        "widths": widths,
        "config": {"resume_j_a": args.resume_j_a, "resume_j_b": args.resume_j_b,
                   "resume_j_e0": args.resume_j_e0,
                   "selector": "iter_bm25", "topk": args.topk,
                   "iter_hop_topk": args.iter_hop_topk,
                   "chunk_size": args.chunk_size, "max_new_tokens": mnt,
                   "warmup": args.warmup, "n_repeat": args.n_repeat,
                   "dtype": args.dtype, "attn_impl": args.attn_impl,
                   "lora_sha256": lora_sha256, "num_layers": L},
        "pack": {"pack_read_len": pack["pack_read_len"],
                 "packed_ids_sha256": pack["packed_ids_sha256"],
                 "sel_idx": pack["sel_idx"]},
        "write_span_tokens": {k: {"total_ctx_tokens": v[0], "prefix_tokens": v[1]}
                              for k, v in span_by_w.items()},
        "armA": resA, "armB": resB, "armE0": resE0,
        "armE2": resE2,
        "note": "E2 overlap Write adds ONLY a one-time, per-chunk longer lower-12 "
                "forward (by the w left-context prefix tokens, which are discarded and "
                "never enter the store / pack / Read / decode). The store pack, "
                "persistent bytes/token, Read compute and decode are BIT-IDENTICAL to "
                "Arm B (w0). Arm E0 (control) is O(L) per document, NOT deployable.",
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda" else None,
                "python": platform.python_version(),
                "node": socket.gethostname()},
    }
    with open(outdir / f"latency_proc{args.proc_id}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[p0.17][latency] wrote latency_proc{args.proc_id}.json", flush=True)


# --------------------------------------------------------------------------- #
# MANIFEST mode (strict-fix verification + provenance dump)
# --------------------------------------------------------------------------- #
def run_manifest(args, device, dtype):
    torch.manual_seed(args.seed)
    widths = [int(w) for w in args.widths]
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    qcA = QCMemModel(model, resume_j=args.resume_j_a)
    _ = QCMemModel(model, resume_j=args.resume_j_b)      # E2 / B model
    _ = QCMemModel(model, resume_j=args.resume_j_e0)     # E0 model
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
        "run": "P0.17_E2_overlap_write",
        "widths": widths,
        "arms": {
            "A": {"resume_j": args.resume_j_a,
                  "note": "flagship LoRA, full 36-layer continuous replay "
                          "(== P0.16/P1.7/P0.13 Arm A; RAG upper anchor)"},
            "B_w0": {"resume_j": args.resume_j_b,
                     "note": "same LoRA, upper-24 replay from DEPLOYABLE chunk-local "
                             "cached h12 (== P0.16/P1.7/P0.13 Arm B); this IS the E2 "
                             "w=0 baseline (numeric identity gate: _e2_write_chunk "
                             "with no prefix == write_chunk, max_abs 0)"},
            "E2": {"resume_j": args.resume_j_b, "widths": widths,
                   "note": "overlapping-chunk Write: prepend the w preceding DOCUMENT "
                           "tokens to each 512-token context chunk, run lower-12 over "
                           "the (w+512) span chunk-locally, DISCARD the prefix h12, "
                           "store ONLY the 512-token chunk h12. Sink/query write, "
                           "store pack, persistent bytes/token, Read (fresh contiguous "
                           "pack positions) and O(1) decode are BIT-IDENTICAL to B; "
                           "only the one-time Write forward is longer."},
            "E0": {"resume_j": args.resume_j_e0,
                   "note": "DOCUMENT-CONTEXTUAL Write control (== P0.16 Arm E0; O(L), "
                           "cross-query-reusable control, NOT a shipping config nor a "
                           "strict upper bound)."},
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
            "add_bos": 0, "dtype": args.dtype, "attn_impl": args.attn_impl,
            "max_new_tokens": args.max_new_tokens, "seed": args.seed,
            "widths": widths,
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
        print("[p0.17][manifest][ABORT] strict-fix mismatch:", flush=True)
        for a in abort:
            print("   - " + a, flush=True)
        sys.exit(3)
    print(f"[p0.17][manifest] OK — LoRA sha {lora_sha256[:12]}… "
          f"{prov_lora['count']} modules, layers [12..35]; widths={widths}; "
          f"torch {prov_versions['torch']} tf {prov_versions['transformers']} "
          f"peft {prov_versions['peft']} git {prov_versions['git_commit_short']}",
          flush=True)


# --------------------------------------------------------------------------- #
# AGGREGATE mode (pure CPU: 6-arm per-cell + pairwise paired stats)
# --------------------------------------------------------------------------- #
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
    if not valid:
        raise SystemExit("[p0.17][aggregate] no valid records found")

    # discover widths present across ALL valid records (fail-closed on ragged sets).
    width_sets = {tuple(r.get("widths", [])) for r in valid}
    if len(width_sets) != 1:
        raise AssertionError(
            f"[p0.17][aggregate] inconsistent widths across records: {width_sets}")
    widths = list(width_sets.pop())
    e2_arms = [_arm_key(w) for w in widths]
    all_arms = ["armA", "armB"] + e2_arms + ["armE0"]
    for r in valid:
        for a in all_arms:
            if a not in r:
                raise AssertionError(
                    f"[p0.17][aggregate] record {r.get('example_id')} missing {a}")

    macros = {}
    cells_by_arm = {}
    cells = None
    for a in all_arms:
        m, c, cc = _macro_and_cells(valid, a)
        macros[a] = m
        cells_by_arm[a] = c
        cells = cc
    cells_keys = sorted(cells.keys())

    per_cell = {}
    for key in cells_keys:
        tag = f"{key[0]}/{key[1]}"
        entry = {"n": len(cells[key])}
        for a in all_arms:
            entry[a] = cells_by_arm[a][tag]
        entry["diff_A_minus_B"] = round(
            cells_by_arm["armA"][tag] - cells_by_arm["armB"][tag], 2)
        entry["diff_E0_minus_B"] = round(
            cells_by_arm["armE0"][tag] - cells_by_arm["armB"][tag], 2)
        for w in widths:
            ak = _arm_key(w)
            entry[f"diff_E2_w{w}_minus_B"] = round(
                cells_by_arm[ak][tag] - cells_by_arm["armB"][tag], 2)
            entry[f"diff_E0_minus_E2_w{w}"] = round(
                cells_by_arm["armE0"][tag] - cells_by_arm[ak][tag], 2)
        per_cell[tag] = entry

    # pairwise: each E2_w vs w0 baseline (B) AND vs E0 control; + A-B, E0-B context.
    pairwise = {
        "A_vs_B": _pairwise(valid, cells_keys, cells, "armA", "armB", args.n_boot),
        "E0_vs_B": _pairwise(valid, cells_keys, cells, "armE0", "armB", args.n_boot),
    }
    agreement = {"A_vs_B": _agree_means(valid, "A_vs_B"),
                 "E0_vs_B": _agree_means(valid, "E0_vs_B")}
    for w in widths:
        ak = _arm_key(w)
        pairwise[f"E2_w{w}_vs_B"] = _pairwise(
            valid, cells_keys, cells, ak, "armB", args.n_boot)
        pairwise[f"E2_w{w}_vs_E0"] = _pairwise(
            valid, cells_keys, cells, ak, "armE0", args.n_boot)
        agreement[f"E2_w{w}_vs_B"] = _agree_means(valid, f"E2_w{w}_vs_B")
        agreement[f"E2_w{w}_vs_E0"] = _agree_means(valid, f"E2_w{w}_vs_E0")

    n = len(valid)
    any_oom = sum(1 for r in all_recs if r.get("oom"))
    any_nonfinite = sum(1 for r in valid
                        if not all(r[a]["finite"] for a in all_arms))
    pack_paired = all(
        all(r[a]["read_len"] == r["pack_read_len"] for a in all_arms) for r in valid)
    p013_checked = [r for r in valid if r.get("p013_pack_sha_match") is not None]
    p013_all_match = all(r["p013_pack_sha_match"] for r in p013_checked) \
        if p013_checked else None

    # pooled extra Write-span tokens (mean over examples) per width, vs w0.
    write_cost = {}
    w0_totals = [r["write_span_tokens"]["w0"]["total_ctx_tokens"] for r in valid]
    mean_w0 = sum(w0_totals) / len(w0_totals)
    write_cost["w0_mean_ctx_write_tokens"] = round(mean_w0, 2)
    for w in widths:
        pre = [r["write_span_tokens"][f"w{w}"]["prefix_tokens"] for r in valid]
        tot = [r["write_span_tokens"][f"w{w}"]["total_ctx_tokens"] for r in valid]
        mean_pre = sum(pre) / len(pre)
        mean_tot = sum(tot) / len(tot)
        write_cost[f"w{w}"] = {
            "mean_extra_prefix_tokens": round(mean_pre, 2),
            "mean_total_ctx_write_tokens": round(mean_tot, 2),
            "extra_write_flops_ratio_vs_w0": round(mean_tot / mean_w0, 4)
            if mean_w0 else None,
            "note": "extra lower-12 Write cost only; store/Read/decode identical to w0",
        }

    macro = {"armA_full_replay": round(macros["armA"], 3),
             "armB_chunk_local_w0_deployable": round(macros["armB"], 3),
             "armE0_document_contextual_control": round(macros["armE0"], 3),
             "diff_A_minus_B": round(macros["armA"] - macros["armB"], 3),
             "diff_E0_minus_B": round(macros["armE0"] - macros["armB"], 3)}
    for w in widths:
        ak = _arm_key(w)
        macro[f"armE2_w{w}_overlap_write"] = round(macros[ak], 3)
        macro[f"diff_E2_w{w}_minus_B"] = round(macros[ak] - macros["armB"], 3)
        macro[f"diff_E0_minus_E2_w{w}"] = round(macros["armE0"] - macros[ak], 3)

    # pre-registered primary target: pooled B(92.5) -> >=97.0 at unchanged store/Read.
    best_w = max(widths, key=lambda w: macros[_arm_key(w)]) if widths else None
    prereg = {
        "target_metric": "multikey pooled (macro over cells) armE2_w accuracy",
        "baseline_w0_pooled": round(macros["armB"], 3),
        "target_threshold": 97.0,
        "best_width": best_w,
        "best_width_pooled": round(macros[_arm_key(best_w)], 3) if best_w else None,
        "target_met": (macros[_arm_key(best_w)] >= 97.0) if best_w else None,
        "note": "ALL widths reported below regardless; reporting only the best w is "
                "forbidden. store/Read cost is identical across all widths (== w0).",
    }

    summary = {
        "n_examples_paired": n, "n_cells": len(cells_keys), "widths": widths,
        "per_cell": per_cell, "macro": macro,
        "write_cost": write_cost,
        "prereg_target": prereg,
        "oom_examples": any_oom, "nonfinite_examples": any_nonfinite,
        "all_packs_paired_1to1": pack_paired,
        "p013_pack_sha_checked": len(p013_checked),
        "p013_pack_sha_all_match": p013_all_match,
        "attribution_hint": (
            "E2_w recovers the E0-vs-B document-context gap in proportion to the "
            "left-context width w, at unchanged store/Read cost. If the best w reaches "
            "the pre-registered >=97.0 pooled target it becomes a new deployable Write "
            "variant (extend Cohort-B); if only a small/negative effect, report as a "
            "boundary/negative result and pivot to P0.18. E0 is the O(L) "
            "document-contextual control (not deployable), the recovery ceiling."),
    }
    stats = {"macro": macro, "pairwise": pairwise, "agreement": agreement,
             "write_cost": write_cost, "prereg_target": prereg,
             "bootstrap_n": args.n_boot}
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(outdir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # latency aggregation (all arms) across procs
    lat_procs = []
    for lf in sorted(glob.glob(str(outdir / "latency" / "latency_proc*.json"))):
        with open(lf) as f:
            lat_procs.append(json.load(f))
    latency = {"n_procs": len(lat_procs), "procs": []}
    if lat_procs:
        from bench_p0_13_quality_latency import _summ
        for lp in lat_procs:
            entry = {
                "proc_id": lp["proc_id"], "task": lp["task"], "length": lp["length"],
                "armA_read_ms": round(lp["armA"]["read_s"]["median"] * 1e3, 3),
                "armB_read_ms": round(lp["armB"]["read_s"]["median"] * 1e3, 3),
                "armB_write_ms": round(lp["armB"]["write_s"]["median"] * 1e3, 3),
                "armE0_read_ms": round(lp["armE0"]["read_s"]["median"] * 1e3, 3),
                "armE0_write_ms": round(lp["armE0"]["write_s"]["median"] * 1e3, 3),
            }
            for w in lp.get("widths", []):
                e2 = lp["armE2"][f"w{w}"]
                entry[f"armE2_w{w}_read_ms"] = round(e2["read_s"]["median"] * 1e3, 3)
                entry[f"armE2_w{w}_write_ms"] = round(e2["write_s"]["median"] * 1e3, 3)
            latency["procs"].append(entry)

        def _pool_simple(arm, phase):
            raw = []
            for lp in lat_procs:
                raw.extend(lp[arm][phase]["raw"])
            return _summ(raw)

        def _pool_e2(w, phase):
            raw = []
            for lp in lat_procs:
                if f"w{w}" in lp.get("armE2", {}):
                    raw.extend(lp["armE2"][f"w{w}"][phase]["raw"])
            return _summ(raw)
        pooled = {a: {p: _pool_simple(a, p) for p in
                      ("read_s", "write_s", "decode_s", "total_s")}
                  for a in ("armA", "armB", "armE0")}
        for w in widths:
            pooled[f"armE2_w{w}"] = {p: _pool_e2(w, p) for p in
                                     ("read_s", "write_s", "decode_s", "total_s")}
        latency["pooled"] = pooled
    with open(outdir / "latency.json", "w") as f:
        json.dump(latency, f, indent=2)

    print("=" * 82)
    print(f"[p0.17][aggregate] n_paired={n} n_cells={len(cells_keys)} widths={widths}")
    print(f"  macro  A(full)={macros['armA']:.2f}  B(w0 deployable)="
          f"{macros['armB']:.2f}  E0(doc-ctx ctrl)={macros['armE0']:.2f}")
    for w in widths:
        print(f"         E2_w{w}(overlap)={macros[_arm_key(w)]:.2f}")
    order = [("A-B", "A_vs_B"), ("E0-B", "E0_vs_B")]
    for w in widths:
        order += [(f"E2_w{w}-B", f"E2_w{w}_vs_B"),
                  (f"E2_w{w}-E0", f"E2_w{w}_vs_E0")]
    for name, pk in order:
        pw = pairwise[pk]
        print(f"  {name}: diff={pw['macro_diff']:+.2f} "
              f"CI={pw['paired_bootstrap_95ci']} "
              f"McNemar p={pw['mcnemar']['exact_two_sided_p']:.3g} "
              f"(b={pw['mcnemar']['X_only_correct_b']} "
              f"c={pw['mcnemar']['Y_only_correct_c']})")
    print(f"  prereg target(pooled>=97.0): best_w={prereg['best_width']} "
          f"pooled={prereg['best_width_pooled']} met={prereg['target_met']}")
    print(f"  packs_paired_1to1={pack_paired} p013_sha_match={p013_all_match} "
          f"oom={any_oom} nonfinite={any_nonfinite}")
    print("=" * 82, flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="P0.17 E2 overlapping-chunk Write (paired, 6-arm)")
    ap.add_argument("--mode", required=True,
                    choices=["manifest", "quality", "latency", "aggregate",
                             "e2_sanity"])
    ap.add_argument("--model_path", type=str, default="models/Qwen3-8b-local")
    ap.add_argument("--lora_adapter", type=str,
                    default="outputs/qcmem_distill_qwen_j12_r32_4k/final")
    ap.add_argument("--resume_j_a", type=int, default=0)     # full replay (anchor)
    ap.add_argument("--resume_j_b", type=int, default=12)    # chunk-local / E2 / w0
    ap.add_argument("--resume_j_e0", type=int, default=12)   # document-contextual ctrl
    ap.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128],
                    help="space-separated E2 left-context widths (all reported)")
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
                    default="bench_results/p0_17_e2_overlap")
    ap.add_argument("--p013_manifest_dir", type=str, default="",
                    help="optional P0.13/P1.7 output dir to cross-check pack shas")
    # numeric gates / verify
    ap.add_argument("--verify", action="store_true",
                    help="in quality mode, run the E0 h12 invariant + E2 w=0 identity "
                         "assert on the first processed example")
    ap.add_argument("--h12_tol", type=float, default=5e-2,
                    help="max-abs bf16 tolerance for the E0 document-contextual h12 "
                         "invariant (reused from P0.16)")
    ap.add_argument("--e2_w0_tol", type=float, default=1e-3,
                    help="max-abs tolerance for the E2 w=0 identity gate "
                         "(_e2_write_chunk(no prefix) == write_chunk; expected ~0)")
    ap.add_argument("--h12_check_prefix", type=int, default=1024,
                    help="document-prefix length (tokens) for the E0 h12 numeric check")
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

    print("=" * 80)
    print(f"P0.17 E2 :: mode={args.mode} task={args.task} length={args.length} "
          f"shard={args.shard_index}/{args.num_shards}")
    print(f"  model={args.model_path} lora={args.lora_adapter}")
    print(f"  A=j{args.resume_j_a}(full) B/E2=j{args.resume_j_b}(chunk-local/overlap) "
          f"E0=j{args.resume_j_e0}(doc-ctx) widths={args.widths} "
          f"topk={args.topk} hop={args.iter_hop_topk} chunk={args.chunk_size} "
          f"dtype={dtype} attn={args.attn_impl} device={device}")
    print("=" * 80, flush=True)

    if args.mode == "manifest":
        run_manifest(args, device, dtype)
    elif args.mode == "quality":
        run_quality(args, device, dtype)
    elif args.mode == "latency":
        run_latency(args, device, dtype)
    elif args.mode == "e2_sanity":
        run_e2_sanity(args, device, dtype)


if __name__ == "__main__":
    main()
