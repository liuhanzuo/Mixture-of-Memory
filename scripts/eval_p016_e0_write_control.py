#!/usr/bin/env python
"""P0.16 — E0 document-contextual Write control (INFERENCE-ONLY, paired).

Adds a FOURTH operating point — the ``E0`` document-contextual Write — to the
P1.7 continuous-prefix attribution, so that on the SAME paired examples / SAME
selected pack / SAME flagship LoRA we can place E0 between the two P0.16 endpoints
(``j=0`` full replay and the deployable chunk-local ``j=12`` Write) and against the
continuous-pack oracle. This resolves whether the P0.13 deployable gap is caused by
(i) independent chunk Write LACKING cross-document context, or (ii) the Write→Read
coordinate remapping (RoPE repositioning). No training, no model surgery.

Four strictly paired arms (differ ONLY in HOW h12 is produced before the SHARED
upper-24 replay + O(1) decode). Arms A / B / C are imported VERBATIM from the
unmodified P1.7 harness (which itself imports P0.13 verbatim), so they are
bit-identical to the P0.13 / P1.7 headline rows:

  * Arm A  — ``resume_j=0`` full 36-layer continuous replay + flagship LoRA
             (RAG upper bound; == P0.13 / P1.7 Arm A). ``p017._run_arm(qc0, …)``.
  * Arm B  — ``resume_j=12`` DEPLOYABLE chunk-local cached h12 + SAME LoRA
             (== P0.13 / P1.7 Arm B). Each selected chunk / sink / query is
             encoded to depth 12 CHUNK-LOCALLY (isolated, RoPE 0:T each), the
             per-chunk h12 are REPOSITIONED into a fresh contiguous pack, and
             layers[12:36] resume. This is the deployable endpoint the P0.16
             interpretation rule compares E0 against. ``p017._run_arm(qc12, …)``.
             (Included so the deployable comparison is IN THE SAME paired run — no
             cross-run re-pairing; toggle with ``--no_armB``.)
  * Arm C  — CONTINUOUS-PACK ORACLE (== P1.7 Arm C, NOT deployable). layers[0:12]
             run continuously / full-causal over the SELECTED PACK (contiguous pack
             positions 0:H, no repositioning), pack-level h12 captured once, resume.
             ``p017._run_oracle(qcC, packed_ids, …)``.
  * Arm E0 — DOCUMENT-CONTEXTUAL WRITE (new; query-INDEPENDENT, O(L) Write).
             layers[0:12] run ONCE, CONTINUOUSLY and FULL-CAUSALLY, over the WHOLE
             ORIGINAL DOCUMENT in its original causal order (RoPE positions 0:N =
             document-origin positions), producing a per-token h12 that sees the
             full preceding document context and is INDEPENDENT of which chunks the
             query later selects (cacheable once per document, reused across
             queries). The per-token h12 is then SLICED at the BM25-selected chunk
             boundaries and assembled into the SAME store-pack layout as Arm B
             ([sink ; selected chunk h12 (doc order) ; query h12]); the Read runs
             layers[12:36] over that pack with FRESH CONTIGUOUS pack positions 0:H
             (the deployable Read interface — so the store→read RoPE REPOSITIONING
             is IDENTICAL to Arm B). Decode is O(1)/step: the bottom band continues
             in DOCUMENT coordinates (a generated token attends to the whole
             document-contextual lower-12 KV) while the top band continues in PACK
             coordinates — mirroring Arm B's two-coordinate decode.

  E0 vs Arm B  (same repositioning, different lower-12 attention scope) isolates
               the value of DOCUMENT CONTEXT at fixed Write→Read repositioning.
  E0 vs Arm C  (both context-aware; C has no repositioning, E0 does) isolates the
               cost of the Write→Read REPOSITIONING at fixed context availability.

Attribution logic (per paperA/TODOList.md §P0.16):
  * E0 ≈ A / C  (and E0 ≫ B): most loss is chunk-independent Write lacking document
    context; the deployable Read interface / repositioning is largely lossless ⇒
    prioritise P0.17 (overlap Write to inject context).
  * E0 ≈ B      (and E0 ≪ A / C): even a perfect document-contextual h12 cannot be
    recovered once it is repositioned into the selected-pack coordinates; the loss
    is the Write→Read remapping ⇒ prioritise P0.18 (two-factor decomposition).
  * E0 in between: both factors present ⇒ run P0.17 + P0.18.
  Any result is admissible; the paired CI + McNemar (positive OR negative) is the
  deliverable and flows into the mechanism table.

!! E0 IS NOT PRESENTED AS A STRICT UPPER BOUND. Per the TODOList wording rule it is
   a "cross-query-reusable document-contextual control": its Write is O(L) over the
   whole document (write latency + peak reported), and long-document position
   extension + document-update cost must be reported alongside it.

KEY INVARIANT (violating it invalidates E0):
   E0's document-contextual layer-12 state MUST equal the SAME-DOCUMENT stock
   lower-12 forward — it is NOT stitched from per-chunk caches. Because a causal
   lower-12 forward at position p depends only on tokens[0:p+1], E0's h12 over the
   full document, restricted to the first P positions, equals the stock lower-12
   forward over the P-token document PREFIX. ``--mode e0_h12_sanity`` / ``--verify``
   therefore run the numerical assertion on a short document prefix (cheap, exact
   same code path): they REUSE ``p017._h12_residual`` (oracle continuous lower-12 vs
   the stock model's ``output_hidden_states[12]``; the LoRA lives on layers 12..35,
   so hidden_states[12] is the pure lower-12 forward, adapter-independent) and
   assert the actual bf16 max-abs residual is below ``--h12_tol``.

Pack pairing (forward-free, resume_j-independent):
   The flagship RULER selector ``iter_bm25`` is pure lexical BM25 over raw token ids
   (no model forward), so the selected chunk indices / order / packed token ids /
   pack sha are IDENTICAL to P0.13 / P1.7. This harness builds the pack ONCE per
   example with the UNMODIFIED ``p017._build_pack`` (== P0.13 ``_build_pack``) and
   ALL FOUR arms read that identical pack. Optionally cross-checks each pack sha
   against the P0.13 / P1.7 per-example JSONL (``--p013_manifest_dir``).

This file NEVER mutates ``bench_p1_7_h12_oracle.py``, ``bench_p0_13_quality_latency``
or ``qcmem_model.py`` — it imports the P1.7 primitives read-only and adds only the
E0 document-contextual Write forward (which uses QCMemModel's public low-level
accessors: ``embed_tokens``, ``_make_mask_and_rope``, ``_run_layers``,
``read_prefill``, ``decode_step`` — never patching the backbone).
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

# Import the UNMODIFIED P1.7 harness (which imports P0.13 verbatim) and pull every
# shared primitive from it — so arms A / B / C are BIT-IDENTICAL to the P0.13/P1.7
# headline paths (same pack builder, same per-arm generate replica, same oracle,
# same loader, same strict-fix hashes, same stats).
import bench_p1_7_h12_oracle as p017  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

ruler = p017.ruler
qcb = p017.qcb
QCMemModel = p017.QCMemModel
_bare_question = p017._bare_question
_resolve_task = p017._resolve_task
_build_pack = p017._build_pack
_packed_ids_from_pack = p017._packed_ids_from_pack
_run_arm = p017._run_arm                # arms A (j0) + B (chunk-local j12)
_run_oracle = p017._run_oracle          # arm C (continuous-pack oracle)
_h12_residual = p017._h12_residual      # numeric invariant (reused for E0 prefix)
_stock_lower12_ref = p017._stock_lower12_ref  # stock lower-j reference (numeric gate)
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


# --------------------------------------------------------------------------- #
# document-origin chunk spans (the E0 slicing map). Fail-closed on any mismatch.
# --------------------------------------------------------------------------- #
def _e0_doc_spans(pack, chunk_size, doc_len):
    """Map the selected pack back onto ORIGINAL DOCUMENT token coordinates.

    Returns ``(sink_span, chunk_spans, query_span)`` where each span is a
    ``[start, end)`` half-open interval into the full document token sequence, in
    the EXACT packed order ``[sink ; selected chunks (doc order) ; query]``:

      * sink_span   = ``[0, 1)``           — the BOS/sink is document token 0
                       (== ``_build_pack``'s ``bos_id == tokens[0]`` when the
                       tokenizer has no BOS, e.g. Qwen).
      * chunk_spans = ``[ [i*chunk_size, i*chunk_size + len_i) for i in sel_idx ]``
                       — selected context chunk ``i`` occupies that document span
                       (``context_chunks = tokens.split(chunk_size)[:-1]``, all of
                       length ``chunk_size``).
      * query_span  = ``[n_ctx_chunks*chunk_size, doc_len)`` — the last chunk.

    The lengths are cross-checked against the pack's own tensors so the E0 read pack
    is length-identical to arms A/B/C (the 1:1 pairing guarantee)."""
    sel_idx = pack["sel_idx"]
    sel_tensors = pack["selected_chunk_tensors"]
    n_ctx = pack["n_ctx_chunks"]
    if len(sel_idx) != len(sel_tensors):
        raise AssertionError(
            f"[p0.16][E0] sel_idx/tensor mismatch {len(sel_idx)} != {len(sel_tensors)}")
    q_start = n_ctx * chunk_size
    if not (0 <= q_start < doc_len):
        raise AssertionError(
            f"[p0.16][E0] query start {q_start} out of doc range [0,{doc_len})")
    query_span = (q_start, doc_len)
    chunk_spans = []
    for k, ci in enumerate(sel_idx):
        start = int(ci) * chunk_size
        length = int(sel_tensors[k].shape[0])
        end = start + length
        # every selected chunk is a CONTEXT chunk => must sit strictly before query.
        if start < 0 or end > q_start:
            raise AssertionError(
                f"[p0.16][E0] chunk {ci} span [{start},{end}) escapes context "
                f"region [0,{q_start}) (doc_len={doc_len})")
        chunk_spans.append((start, end))
    sink_span = (0, 1)
    # pack_read_len parity: 1 (sink) + sum(chunk lens) + query len.
    e0_read_len = 1 + sum(e - s for s, e in chunk_spans) + (doc_len - q_start)
    if e0_read_len != pack["pack_read_len"]:
        raise AssertionError(
            f"[p0.16][E0] E0 read_len {e0_read_len} != pack_read_len "
            f"{pack['pack_read_len']} — E0 slicing broke pairing")
    return sink_span, chunk_spans, query_span


# --------------------------------------------------------------------------- #
# E0 document-contextual lower-12 forward (query-INDEPENDENT; the invariant target
# is the SAME continuous lower-12 as the oracle, only fed the WHOLE DOCUMENT).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _e0_doc_lower12(qc, doc_ids, keep_cache=False):
    """Run layers[0:resume_j] as ONE continuous, fully-causal forward over the WHOLE
    document (contiguous document-origin positions 0:N). Returns ``(h12_doc [1,N,d],
    bottom_cache_or_None, N)``. With ``keep_cache=True`` the bottom-band KV is kept
    (spans the whole document) so decode can continue in document coordinates — this
    is the O(L) Write cost of the document-contextual control."""
    ids = qc._as_ids(doc_ids)                    # [1, N]
    N = ids.shape[1]
    emb = qc.embed_tokens(ids)
    positions = torch.arange(N, device=qc.device).unsqueeze(0)
    mask, pe = qc._make_mask_and_rope(emb, positions)
    cache = DynamicCache(config=qc.config) if keep_cache else None
    h12 = qc._run_layers(emb, slice(0, qc.resume_j), mask, positions, pe,
                         past_key_values=cache, use_cache=keep_cache)  # [1,N,d]
    return h12, cache, N


@torch.no_grad()
def _e0_h12_residual(qc, doc_ids):
    """Max-abs residual between E0's OWN document-contextual lower-``resume_j`` forward
    (``_e0_doc_lower12`` — the exact code path / cache mode E0 uses in ``_run_e0``) and
    the stock model's ``output_hidden_states[resume_j]`` on the SAME ids (an
    INDEPENDENT HF forward). This is the numerical acceptance gate for E0's Write:
    the sliced-into-pack h12 is taken verbatim from this continuous forward, so if the
    continuous forward matches stock, every slice does too. (``resume_j==12`` puts the
    LoRA on layers 12..35, so hidden_states[12] is the pure lower-12 forward,
    adapter-independent — same argument as the P1.7 oracle invariant.)"""
    h12 = _e0_doc_lower12(qc, doc_ids, keep_cache=True)[0].float()
    ref = _stock_lower12_ref(qc, doc_ids).float()
    diff = (h12 - ref).abs()
    return {"max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "ref_abs_max": float(ref.abs().max().item()),
            "H": int(h12.shape[1])}


# --------------------------------------------------------------------------- #
# E0 arm generate (document-contextual h12 -> repositioned pack read + O(1) decode)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _run_e0(qc, doc_ids, sink_span, chunk_spans, query_span,
            max_new_tokens, eos_ids, capture_first=False):
    """Run the E0 document-contextual Write arm.

    WRITE (query-independent, O(L)): continuous lower-``resume_j`` over the WHOLE
    document, keep the bottom-band KV cache (document coordinates). SLICE the
    per-token h12 at the sink / selected-chunk / query document spans → store pack.
    READ: ``qc.read_prefill`` over ``[sink ; selected chunks ; query]`` with FRESH
    CONTIGUOUS pack positions 0:H (the deployable Read interface; == Arm B's
    repositioning). DECODE: O(1)/step — bottom band continues at document position N,
    top band at pack position H (two-coordinate decode, like Arm B).

    Returns the SAME 6-tuple shape as ``p017._run_arm`` / ``p017._run_oracle``
    (generated_ids, timings, read_len, peak_gb, finite, first_logits) so the caller
    treats all four arms uniformly."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- WRITE phase (E0): CONTINUOUS lower-band over the WHOLE document ----
    _sync(); tw0 = time.perf_counter()
    h12_doc, bottom_cache, N = _e0_doc_lower12(qc, doc_ids, keep_cache=True)  # [1,N,d]
    # slice the document-contextual h12 into the store pack (doc-origin -> pack).
    s0, s1 = sink_span
    sink_hj = h12_doc[:, s0:s1, :]
    selected_hj = [h12_doc[:, cs:ce, :] for (cs, ce) in chunk_spans]
    q0, q1 = query_span
    query_hj = h12_doc[:, q0:q1, :]
    _sync(); t_write = time.perf_counter() - tw0

    # ---- READ phase (E0): resume the top band over the repositioned pack ----
    _sync(); tr0 = time.perf_counter()
    # read_prefill packs [sink ; selected ; query], assigns fresh contiguous pack
    # positions 0:H, resumes layers[resume_j:L] with a top KV cache, and returns the
    # last-position logits (identical Read interface to the deployable Arm B).
    logits1, top_cache, H = qc.read_prefill(sink_hj, selected_hj, query_hj)
    _sync(); t_read = time.perf_counter() - tr0

    read_len = int(sink_hj.shape[1]) + int(sum(h.shape[1] for h in selected_hj)) \
        + int(query_hj.shape[1])

    first_logits = logits1[0, -1].float()
    finite = bool(torch.isfinite(first_logits).all().item())
    next_logits = first_logits.clone()
    if eos_ids:
        next_logits[eos_ids] = float("-inf")     # step 0 never emits EOS
    first_capture = first_logits.detach().cpu().clone() if capture_first else None

    # ---- DECODE phase: O(1)/step. Bottom band continues in DOCUMENT coordinates
    #      (q_local_pos starts at N, attends the whole document-contextual lower-12
    #      KV); top band continues in PACK coordinates (pack_pos starts at H). ----
    _sync(); td0 = time.perf_counter()
    generated = []
    next_tok = int(next_logits.argmax().item())
    generated.append(next_tok)
    q_local_pos = N          # document coordinate of the first generated token
    pack_pos = H             # pack coordinate of the first generated token
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


# --------------------------------------------------------------------------- #
# QUALITY mode (4 arms on the identical pack + identical document)
# --------------------------------------------------------------------------- #
def run_quality(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L = int(model.config.num_hidden_layers)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p0.16][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    qcA = QCMemModel(model, resume_j=args.resume_j_a)      # 0  full replay
    qcB = QCMemModel(model, resume_j=args.resume_j_b)      # 12 chunk-local (deployable)
    qcC = QCMemModel(model, resume_j=args.resume_j_c)      # 12 continuous oracle
    qcE0 = QCMemModel(model, resume_j=args.resume_j_e0)    # 12 document-contextual
    eosA = _eos_ids(qcA, tokenizer)
    eosB = _eos_ids(qcB, tokenizer)
    eosC = _eos_ids(qcC, tokenizer)
    eosE0 = _eos_ids(qcE0, tokenizer)

    task = _resolve_task(args.task)
    length = args.length
    if length not in ruler._LENGTH_TOKENS:
        raise SystemExit(f"[p0.16] unknown length {length}")
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

    # Optional cross-check against a P0.13 / P1.7 pack manifest (same examples).
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

    print(f"[p0.16][quality] {task}/{length}{shard_tag}: selector={sel_name} "
          f"topk={args.topk} hop={args.iter_hop_topk} n={len(sample_indices)}/"
          f"{args.limit} A=j{args.resume_j_a} B=j{args.resume_j_b} "
          f"C(oracle)=j{args.resume_j_c} E0(doc-ctx)=j{args.resume_j_e0} "
          f"armB={'on' if not args.no_armB else 'off'} mnt={mnt} verify={args.verify}",
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

        # ---- E0 document-origin slicing map (fail-closed on any parity break) ----
        sink_span, chunk_spans, query_span = _e0_doc_spans(
            pack, args.chunk_size, doc_len)

        # cross-check pack sha vs P0.13 / P1.7 (if a manifest dir was given)
        p013_sha_match = None
        if i in p013_shas:
            p013_sha_match = (p013_shas[i] == pack["packed_ids_sha256"])
            if not p013_sha_match:
                raise AssertionError(
                    f"[p0.16] example {i}: pack sha != P0.13/P1.7 pack sha "
                    "— pairing with the existing arms broken")

        # optional E0 h12 invariant sanity on the FIRST processed example (a short
        # document PREFIX: causal lower-12 restricted to the prefix == stock lower-12
        # over that prefix — cheap, exact same forward E0 uses over the full doc).
        h12_check = None
        if args.verify and not verified_once:
            pfx = min(int(args.h12_check_prefix), doc_len)
            prefix_ids = doc_ids[:pfx]
            h12_check = _e0_h12_residual(qcE0, prefix_ids)
            h12_check["prefix_tokens"] = pfx
            print(f"[p0.16][verify] example {i}: E0 doc-lower12 vs stock (prefix="
                  f"{pfx}) max_abs={h12_check['max_abs']:.3e} "
                  f"mean_abs={h12_check['mean_abs']:.3e} "
                  f"(ref_abs_max={h12_check['ref_abs_max']:.3e}) "
                  f"tol={args.h12_tol:.3e}", flush=True)
            assert h12_check["max_abs"] < args.h12_tol, (
                f"[p0.16][ABORT] E0 h12 residual {h12_check['max_abs']:.3e} >= "
                f"tol {args.h12_tol:.3e} — document-contextual invariant violated")
            verified_once = True

        # ---- run the arms on the identical pack / identical document ----
        oom = False
        armB = None
        try:
            genA, tA, rlA, pkA, finA, lA = _run_arm(
                qcA, tokenizer, pack["bos_id"], pack["selected_chunk_tensors"],
                pack["query_ids"], mnt, eosA, capture_first=True)
            if not args.no_armB:
                genB, tB, rlB, pkB, finB, lB = _run_arm(
                    qcB, tokenizer, pack["bos_id"], pack["selected_chunk_tensors"],
                    pack["query_ids"], mnt, eosB, capture_first=True)
            genC, tC, rlC, pkC, finC, lC = _run_oracle(
                qcC, packed_ids, mnt, eosC, capture_first=True)
            genE, tE, rlE, pkE, finE, lE = _run_e0(
                qcE0, doc_ids, sink_span, chunk_spans, query_span,
                mnt, eosE0, capture_first=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            oom = True
            torch.cuda.empty_cache()
            print(f"[p0.16][OOM] i={i} {task}/{length}: {e}", flush=True)

        if oom:
            rec = {"example_id": i, "task": task, "length": length,
                   "oom": True, "gold": " | ".join(answers)}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            records.append(rec); n_done += 1
            continue

        predA = tokenizer.decode(genA, skip_special_tokens=True).strip()
        predC = tokenizer.decode(genC, skip_special_tokens=True).strip()
        predE = tokenizer.decode(genE, skip_special_tokens=True).strip()
        recA = ruler._string_match_all_one(predA, answers)
        recC = ruler._string_match_all_one(predC, answers)
        recE = ruler._string_match_all_one(predE, answers)
        if not args.no_armB:
            predB = tokenizer.decode(genB, skip_special_tokens=True).strip()
            recB = ruler._string_match_all_one(predB, answers)

        # 1:1 pairing guard: ALL arms MUST have consumed the identical pack length.
        pack_rl = pack["pack_read_len"]
        assert rlA == rlC == rlE == pack_rl, \
            (f"read_len mismatch i={i}: A={rlA} C={rlC} E0={rlE} pack={pack_rl}")
        if not args.no_armB:
            assert rlB == pack_rl, f"read_len mismatch i={i}: B={rlB} pack={pack_rl}"

        rec = {
            "example_id": i, "task": task, "length": length,
            "approx_tokens": approx_tokens,
            "gold": " | ".join(answers), "n_refs": len(answers),
            # provenance / coordinates required by the P0.16 acceptance spec.
            "doc_len": doc_len, "doc_ids_sha256": doc_sha,
            "retrieved_chunk_ids": pack["sel_idx"],
            "n_ctx_chunks": pack["n_ctx_chunks"],
            "chunk_size": args.chunk_size,
            "e0_doc_slices": {
                "sink_span": list(sink_span),
                "chunk_spans": [list(s) for s in chunk_spans],
                "query_span": list(query_span),
            },
            "pack_token_count": pack["pack_token_count"],
            "pack_read_len": pack_rl,
            "packed_ids_sha256": pack["packed_ids_sha256"],
            "p013_pack_sha_match": p013_sha_match,
            "lora_sha256": lora_sha256,
            "retrieval_s": retrieval_s,
            "h12_sanity": h12_check,
            # Write/Read RoPE position id ranges (the repositioning is E0-vs-A/B/C's
            # defining coordinate): E0 Write uses document-origin positions [0,N);
            # E0 Read uses fresh contiguous pack positions [0,H); the query is
            # written at doc positions [q0,q1) but read at pack positions
            # [H-T_q, H) — that offset IS the store->read repositioning.
            "rope_positions": {
                "e0_write_doc_positions": [0, doc_len],
                "e0_read_pack_positions": [0, pack_rl],
                "e0_query_write_doc_positions": list(query_span),
                "e0_query_read_pack_positions":
                    [pack_rl - (query_span[1] - query_span[0]), pack_rl],
                "oracle_write_read_pack_positions": [0, pack_rl],
            },
            "armA": {"resume_j": args.resume_j_a, "kind": "full_replay",
                     "prediction": predA, "score": recA,
                     "correct": bool(recA >= 1.0), "gen_len": len(genA),
                     "read_len": rlA, "latency_s": tA, "peak_gb": pkA,
                     "finite": finA},
            "armC": {"resume_j": args.resume_j_c, "kind": "continuous_h12_oracle",
                     "prediction": predC, "score": recC,
                     "correct": bool(recC >= 1.0), "gen_len": len(genC),
                     "read_len": rlC, "latency_s": tC, "peak_gb": pkC,
                     "finite": finC},
            "armE0": {"resume_j": args.resume_j_e0, "kind": "document_contextual_write",
                      "prediction": predE, "score": recE,
                      "correct": bool(recE >= 1.0), "gen_len": len(genE),
                      "read_len": rlE, "latency_s": tE, "peak_gb": pkE,
                      "finite": finE},
            "diff_A_minus_E0": recA - recE,
            "diff_C_minus_E0": recC - recE,
        }
        agree = {"A_vs_E0": _pair_agree(genA, lA, genE, lE),
                 "C_vs_E0": _pair_agree(genC, lC, genE, lE)}
        if not args.no_armB:
            rec["armB"] = {"resume_j": args.resume_j_b, "kind": "chunk_local_h12",
                           "prediction": predB, "score": recB,
                           "correct": bool(recB >= 1.0), "gen_len": len(genB),
                           "read_len": rlB, "latency_s": tB, "peak_gb": pkB,
                           "finite": finB}
            rec["diff_E0_minus_B"] = recE - recB
            rec["diff_A_minus_B"] = recA - recB
            rec["diff_C_minus_B"] = recC - recB
            agree["A_vs_B"] = _pair_agree(genA, lA, genB, lB)
            agree["B_vs_E0"] = _pair_agree(genB, lB, genE, lE)
        rec["agreement"] = agree

        fout.write(json.dumps(rec) + "\n"); fout.flush()
        records.append(rec); n_done += 1
        torch.cuda.empty_cache()
        if n_done % 5 == 0:
            bstr = f" B={recB:.2f}" if not args.no_armB else ""
            print(f"[p0.16][quality] {task}/{length}{shard_tag} {n_done} done "
                  f"(A={recA:.2f}{bstr} C={recC:.2f} E0={recE:.2f} readlen={rlA})",
                  flush=True)
    fout.close()

    valid = [r for r in records if not r.get("oom")]

    def _mean(arm):
        xs = [r[arm]["score"] for r in valid if arm in r]
        return round(sum(xs) / len(xs) * 100.0, 3) if xs else 0.0
    cell = {
        "task": task, "length": length, "shard": shard_tag,
        "n": len(records), "n_valid": len(valid),
        "oom_count": sum(1 for r in records if r.get("oom")),
        "armA_score": _mean("armA"), "armC_score": _mean("armC"),
        "armE0_score": _mean("armE0"),
        "diff_A_minus_E0": round(_mean("armA") - _mean("armE0"), 3),
        "diff_C_minus_E0": round(_mean("armC") - _mean("armE0"), 3),
        "selector": sel_name, "topk": args.topk, "iter_hop_topk": args.iter_hop_topk,
        "chunk_size": args.chunk_size, "max_new_tokens": mnt,
        "resume_j_a": args.resume_j_a, "resume_j_b": args.resume_j_b,
        "resume_j_c": args.resume_j_c, "resume_j_e0": args.resume_j_e0,
        "armB_included": (not args.no_armB),
        "lora_sha256": lora_sha256, "num_layers": L,
        "runtime": {"node": socket.gethostname(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "device": args.device,
                    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                    "seed": args.seed, "dtype": args.dtype,
                    "attn_implementation": args.attn_impl},
        "jsonl": str(jsonl_path),
    }
    if not args.no_armB:
        cell["armB_score"] = _mean("armB")
        cell["diff_E0_minus_B"] = round(_mean("armE0") - _mean("armB"), 3)
    with open(outdir / f"{task}_{length}{shard_tag}_cell.json", "w") as f:
        json.dump(cell, f, indent=2)
    bstr = f" B={cell['armB_score']}" if not args.no_armB else ""
    print(f"[p0.16][quality] DONE {task}/{length}{shard_tag}: "
          f"A={cell['armA_score']}{bstr} C={cell['armC_score']} "
          f"E0={cell['armE0_score']} (A-E0={cell['diff_A_minus_E0']} "
          f"C-E0={cell['diff_C_minus_E0']} n_valid={len(valid)})", flush=True)


# --------------------------------------------------------------------------- #
# E0_H12_SANITY mode (document-contextual invariant on a short document prefix)
# --------------------------------------------------------------------------- #
def run_e0_h12_sanity(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p0.16][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
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
    ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    if isinstance(ids, list):
        ids = torch.tensor([ids], dtype=torch.long)
    doc_ids = ids.to(device)[0]
    pfx = min(int(args.h12_check_prefix), int(doc_ids.shape[0]))
    prefix_ids = doc_ids[:pfx]

    # Causal lower-12 over the prefix == stock lower-12 over the prefix; and by
    # causality that equals E0's full-document h12 restricted to the first `pfx`
    # positions — so this validates the EXACT forward E0 runs over the whole doc.
    res = _e0_h12_residual(qcE0, prefix_ids)
    print("=" * 72)
    print(f"[p0.16][e0_h12_sanity] {task}/{length} example {i}: "
          f"doc prefix P={pfx}")
    print(f"  E0 document-contextual lower-{args.resume_j_e0} vs stock lower-"
          f"{args.resume_j_e0} forward:")
    print(f"    max_abs={res['max_abs']:.3e}  mean_abs={res['mean_abs']:.3e}  "
          f"(ref_abs_max={res['ref_abs_max']:.3e})  tol={args.h12_tol:.3e}")
    print("=" * 72, flush=True)
    assert res["max_abs"] < args.h12_tol, (
        f"[p0.16][ABORT] E0 h12 residual {res['max_abs']:.3e} >= "
        f"tol {args.h12_tol:.3e} — document-contextual invariant violated")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "e0_h12_sanity.json", "w") as f:
        json.dump({"task": task, "length": length, "example_index": i,
                   "prefix_tokens": pfx, "resume_j_e0": args.resume_j_e0,
                   "dtype": args.dtype, "h12_residual": res,
                   "h12_tol": args.h12_tol, "passed": True}, f, indent=2)
    print("[p0.16][e0_h12_sanity] PASS — residual within tol; wrote "
          "e0_h12_sanity.json", flush=True)


# --------------------------------------------------------------------------- #
# LATENCY mode (per-arm write/read/decode timing on a fixed pack/document)
# --------------------------------------------------------------------------- #
def run_latency(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L = int(model.config.num_hidden_layers)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p0.16][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    qcA = QCMemModel(model, resume_j=args.resume_j_a)
    qcB = QCMemModel(model, resume_j=args.resume_j_b)
    qcC = QCMemModel(model, resume_j=args.resume_j_c)
    qcE0 = QCMemModel(model, resume_j=args.resume_j_e0)
    eosA = _eos_ids(qcA, tokenizer); eosB = _eos_ids(qcB, tokenizer)
    eosC = _eos_ids(qcC, tokenizer); eosE0 = _eos_ids(qcE0, tokenizer)

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
    sink_span, chunk_spans, query_span = _e0_doc_spans(pack, args.chunk_size, doc_len)

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

    resA = _time(lambda: _run_arm(qcA, tokenizer, pack["bos_id"],
                                  pack["selected_chunk_tensors"], pack["query_ids"],
                                  mnt, eosA, capture_first=False))
    resB = _time(lambda: _run_arm(qcB, tokenizer, pack["bos_id"],
                                  pack["selected_chunk_tensors"], pack["query_ids"],
                                  mnt, eosB, capture_first=False))
    resC = _time(lambda: _run_oracle(qcC, packed_ids, mnt, eosC, capture_first=False))
    resE = _time(lambda: _run_e0(qcE0, doc_ids, sink_span, chunk_spans, query_span,
                                 mnt, eosE0, capture_first=False))
    print(f"[p0.16][latency] proc={args.proc_id} {task}/{length} "
          f"A.read={resA['read_s']['median']*1e3:.1f}ms "
          f"B.read={resB['read_s']['median']*1e3:.1f}ms "
          f"C.read={resC['read_s']['median']*1e3:.1f}ms "
          f"E0.read={resE['read_s']['median']*1e3:.1f}ms "
          f"E0.write={resE['write_s']['median']*1e3:.1f}ms "
          f"(E0 write is O(L) over the whole document, per-doc cacheable)",
          flush=True)

    outdir = Path(args.output_dir) / "latency"
    outdir.mkdir(parents=True, exist_ok=True)
    result = {
        "mode": "latency", "proc_id": args.proc_id,
        "task": task, "length": length, "example_index": i, "doc_len": doc_len,
        "config": {"resume_j_a": args.resume_j_a, "resume_j_b": args.resume_j_b,
                   "resume_j_c": args.resume_j_c, "resume_j_e0": args.resume_j_e0,
                   "selector": "iter_bm25", "topk": args.topk,
                   "iter_hop_topk": args.iter_hop_topk,
                   "chunk_size": args.chunk_size, "max_new_tokens": mnt,
                   "warmup": args.warmup, "n_repeat": args.n_repeat,
                   "dtype": args.dtype, "attn_impl": args.attn_impl,
                   "lora_sha256": lora_sha256, "num_layers": L},
        "pack": {"pack_read_len": pack["pack_read_len"],
                 "packed_ids_sha256": pack["packed_ids_sha256"],
                 "sel_idx": pack["sel_idx"]},
        "armA": resA, "armB": resB, "armC": resC, "armE0": resE,
        "note": "Arm E0 Write is O(L): it runs lower-12 over the WHOLE document once "
                "(query-independent, cacheable per document); the reported write_s is "
                "the one-time per-document Write cost. Arm C (oracle) re-runs lower-12 "
                "over the selected pack PER QUERY. Neither replaces the deployable "
                "chunk-local Write (Arm B) as the shipping configuration.",
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device)
                if device.type == "cuda" else None,
                "python": platform.python_version(),
                "node": socket.gethostname()},
    }
    with open(outdir / f"latency_proc{args.proc_id}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[p0.16][latency] wrote latency_proc{args.proc_id}.json", flush=True)


# --------------------------------------------------------------------------- #
# MANIFEST mode (strict-fix verification + provenance dump)
# --------------------------------------------------------------------------- #
def run_manifest(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    qcA = QCMemModel(model, resume_j=args.resume_j_a)
    # construct the E0 model too, to fail fast on any resume_j misconfig.
    _ = QCMemModel(model, resume_j=args.resume_j_e0)
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
        "run": "P0.16_E0_document_contextual_write",
        "arms": {
            "A": {"resume_j": args.resume_j_a,
                  "note": "flagship LoRA, full 36-layer continuous replay "
                          "(== P0.13/P1.7 Arm A)"},
            "B": {"resume_j": args.resume_j_b, "included": (not args.no_armB),
                  "note": "same LoRA, upper-24 replay from DEPLOYABLE chunk-local "
                          "cached h12 (== P0.13/P1.7 Arm B)"},
            "C": {"resume_j": args.resume_j_c,
                  "note": "ORACLE (NOT deployable): upper-24 replay from CONTINUOUS "
                          "pack-level h12 over the selected pack (== P1.7 Arm C)"},
            "E0": {"resume_j": args.resume_j_e0,
                   "note": "DOCUMENT-CONTEXTUAL Write (new; query-INDEPENDENT, O(L)): "
                           "lower-12 run ONCE continuously/full-causal over the WHOLE "
                           "document (RoPE 0:N), per-token h12 sliced at BM25-selected "
                           "chunk boundaries into the SAME pack layout as Arm B, read "
                           "with fresh contiguous pack positions (== Arm B's "
                           "repositioning). NOT a strict upper bound; a cross-query-"
                           "reusable control. h12 verified == stock lower-12 over a "
                           "document prefix by --mode e0_h12_sanity."},
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
        print("[p0.16][manifest][ABORT] strict-fix mismatch:", flush=True)
        for a in abort:
            print("   - " + a, flush=True)
        sys.exit(3)
    print(f"[p0.16][manifest] OK — LoRA sha {lora_sha256[:12]}… "
          f"{prov_lora['count']} modules, layers [12..35]; "
          f"torch {prov_versions['torch']} tf {prov_versions['transformers']} "
          f"peft {prov_versions['peft']} git {prov_versions['git_commit_short']}",
          flush=True)


# --------------------------------------------------------------------------- #
# AGGREGATE mode (pure CPU: 4-arm per-cell + pairwise paired stats)
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
    have_B = bool(valid) and all("armB" in r for r in valid)

    macroA, cellA, cells = _macro_and_cells(valid, "armA")
    macroC, cellC, _ = _macro_and_cells(valid, "armC")
    macroE, cellE, _ = _macro_and_cells(valid, "armE0")
    if have_B:
        macroB, cellB, _ = _macro_and_cells(valid, "armB")
    cells_keys = sorted(cells.keys())

    per_cell = {}
    for key in cells_keys:
        tag = f"{key[0]}/{key[1]}"
        entry = {
            "n": len(cells[key]),
            "armA": cellA[tag], "armC": cellC[tag], "armE0": cellE[tag],
            "diff_A_minus_E0": round(cellA[tag] - cellE[tag], 2),
            "diff_C_minus_E0": round(cellC[tag] - cellE[tag], 2),
        }
        if have_B:
            entry["armB"] = cellB[tag]
            entry["diff_E0_minus_B"] = round(cellE[tag] - cellB[tag], 2)
            entry["diff_A_minus_B"] = round(cellA[tag] - cellB[tag], 2)
            entry["diff_C_minus_B"] = round(cellC[tag] - cellB[tag], 2)
        per_cell[tag] = entry

    pairwise = {
        "A_vs_E0": _pairwise(valid, cells_keys, cells, "armA", "armE0", args.n_boot),
        "C_vs_E0": _pairwise(valid, cells_keys, cells, "armC", "armE0", args.n_boot),
    }
    agreement = {"A_vs_E0": _agree_means(valid, "A_vs_E0"),
                 "C_vs_E0": _agree_means(valid, "C_vs_E0")}
    if have_B:
        pairwise["E0_vs_B"] = _pairwise(valid, cells_keys, cells, "armE0", "armB",
                                        args.n_boot)
        pairwise["A_vs_B"] = _pairwise(valid, cells_keys, cells, "armA", "armB",
                                       args.n_boot)
        pairwise["C_vs_B"] = _pairwise(valid, cells_keys, cells, "armC", "armB",
                                       args.n_boot)
        agreement["A_vs_B"] = _agree_means(valid, "A_vs_B")
        agreement["B_vs_E0"] = _agree_means(valid, "B_vs_E0")

    n = len(valid)
    any_oom = sum(1 for r in all_recs if r.get("oom"))

    def _fin(r):
        keys = ["armA", "armC", "armE0"] + (["armB"] if have_B else [])
        return all(r[k]["finite"] for k in keys)
    any_nonfinite = sum(1 for r in valid if not _fin(r))

    def _paired_rl(r):
        keys = ["armA", "armC", "armE0"] + (["armB"] if have_B else [])
        return all(r[k]["read_len"] == r["pack_read_len"] for k in keys)
    pack_paired = all(_paired_rl(r) for r in valid)
    p013_checked = [r for r in valid if r.get("p013_pack_sha_match") is not None]
    p013_all_match = all(r["p013_pack_sha_match"] for r in p013_checked) \
        if p013_checked else None

    macro = {"armA_full_replay": round(macroA, 3),
             "armC_continuous_h12_oracle": round(macroC, 3),
             "armE0_document_contextual": round(macroE, 3),
             "diff_A_minus_E0": round(macroA - macroE, 3),
             "diff_C_minus_E0": round(macroC - macroE, 3)}
    if have_B:
        macro["armB_chunk_local_h12_deployable"] = round(macroB, 3)
        macro["diff_E0_minus_B"] = round(macroE - macroB, 3)
        macro["diff_A_minus_B"] = round(macroA - macroB, 3)
        macro["diff_C_minus_B"] = round(macroC - macroB, 3)

    summary = {
        "n_examples_paired": n, "n_cells": len(cells_keys),
        "armB_included": have_B,
        "per_cell": per_cell, "macro": macro,
        "oom_examples": any_oom, "nonfinite_examples": any_nonfinite,
        "all_packs_paired_1to1": pack_paired,
        "p013_pack_sha_checked": len(p013_checked),
        "p013_pack_sha_all_match": p013_all_match,
        "attribution_hint": (
            "E0 ≈ A/C (and E0 >> B) => the A-B gap is chunk-independent Write "
            "lacking document context; the deployable Read interface / repositioning "
            "is largely lossless => prioritise P0.17. E0 ≈ B (and E0 << A/C) => even a "
            "perfect document-contextual h12 cannot survive repositioning into the "
            "selected-pack coordinates => prioritise P0.18. E0 in between => both "
            "factors present. E0 is NOT a strict upper bound (O(L) query-independent "
            "control; deployable Write remains Arm B)."),
    }
    stats = {"macro": macro, "pairwise": pairwise, "agreement": agreement,
             "bootstrap_n": args.n_boot}
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(outdir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # latency aggregation (4 arms) across procs
    lat_procs = []
    for lf in sorted(glob.glob(str(outdir / "latency" / "latency_proc*.json"))):
        with open(lf) as f:
            lat_procs.append(json.load(f))
    latency = {"n_procs": len(lat_procs), "procs": []}
    if lat_procs:
        from bench_p0_13_quality_latency import _summ
        for lp in lat_procs:
            latency["procs"].append({
                "proc_id": lp["proc_id"], "task": lp["task"], "length": lp["length"],
                "armA_read_ms": round(lp["armA"]["read_s"]["median"] * 1e3, 3),
                "armB_read_ms": round(lp["armB"]["read_s"]["median"] * 1e3, 3),
                "armC_read_ms": round(lp["armC"]["read_s"]["median"] * 1e3, 3),
                "armE0_read_ms": round(lp["armE0"]["read_s"]["median"] * 1e3, 3),
                "armE0_write_ms": round(lp["armE0"]["write_s"]["median"] * 1e3, 3),
            })

        def _pool(arm, phase):
            raw = []
            for lp in lat_procs:
                raw.extend(lp[arm][phase]["raw"])
            return _summ(raw)
        latency["pooled"] = {a: {p: _pool(a, p) for p in
                                 ("read_s", "write_s", "decode_s", "total_s")}
                             for a in ("armA", "armB", "armC", "armE0")}
    with open(outdir / "latency.json", "w") as f:
        json.dump(latency, f, indent=2)

    print("=" * 78)
    print(f"[p0.16][aggregate] n_paired={n} n_cells={len(cells_keys)} "
          f"armB={'on' if have_B else 'off'}")
    bstr = f"  B(deployable)={macroB:.2f}" if have_B else ""
    print(f"  macro  A(full)={macroA:.2f}{bstr}  C(oracle)={macroC:.2f}  "
          f"E0(doc-ctx)={macroE:.2f}")
    order = [("A-E0", "A_vs_E0"), ("C-E0", "C_vs_E0")]
    if have_B:
        order += [("E0-B", "E0_vs_B"), ("A-B", "A_vs_B"), ("C-B", "C_vs_B")]
    for name, pk in order:
        pw = pairwise[pk]
        print(f"  {name}: diff={pw['macro_diff']:+.2f} "
              f"CI={pw['paired_bootstrap_95ci']} "
              f"McNemar p={pw['mcnemar']['exact_two_sided_p']:.3g}")
    print(f"  packs_paired_1to1={pack_paired} p013_sha_match={p013_all_match} "
          f"oom={any_oom} nonfinite={any_nonfinite}")
    print("=" * 78, flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="P0.16 E0 document-contextual Write control (paired, 4-arm)")
    ap.add_argument("--mode", required=True,
                    choices=["manifest", "quality", "latency", "aggregate",
                             "e0_h12_sanity"])
    ap.add_argument("--model_path", type=str, default="models/Qwen3-8b-local")
    ap.add_argument("--lora_adapter", type=str,
                    default="outputs/qcmem_distill_qwen_j12_r32_4k/final")
    ap.add_argument("--resume_j_a", type=int, default=0)     # full replay
    ap.add_argument("--resume_j_b", type=int, default=12)    # chunk-local (deployable)
    ap.add_argument("--resume_j_c", type=int, default=12)    # continuous oracle
    ap.add_argument("--resume_j_e0", type=int, default=12)   # document-contextual
    ap.add_argument("--no_armB", action="store_true",
                    help="skip the deployable chunk-local Arm B (headline is the "
                         "3 mandated arms A/C/E0; Arm B is included by default so the "
                         "E0-vs-deployable comparison is in the SAME paired run)")
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
                    default="bench_results/p0_16_e0_write_control")
    ap.add_argument("--p013_manifest_dir", type=str, default="",
                    help="optional P0.13/P1.7 output dir to cross-check pack shas")
    # e0 h12 sanity / verify
    ap.add_argument("--verify", action="store_true",
                    help="in quality mode, run the E0 h12 invariant assert on a "
                         "document prefix of the first processed example")
    ap.add_argument("--h12_tol", type=float, default=5e-2,
                    help="max-abs bf16 tolerance for the E0 document-contextual h12 "
                         "invariant (report actual residual regardless)")
    ap.add_argument("--h12_check_prefix", type=int, default=1024,
                    help="document-prefix length (tokens) used for the E0 h12 "
                         "numeric check (short => cheap; validates the same forward)")
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
    print(f"P0.16 E0 :: mode={args.mode} task={args.task} length={args.length} "
          f"shard={args.shard_index}/{args.num_shards}")
    print(f"  model={args.model_path} lora={args.lora_adapter}")
    print(f"  A=j{args.resume_j_a}(full) B=j{args.resume_j_b}(chunk-local) "
          f"C=j{args.resume_j_c}(oracle) E0=j{args.resume_j_e0}(doc-ctx) "
          f"topk={args.topk} hop={args.iter_hop_topk} chunk={args.chunk_size} "
          f"dtype={dtype} attn={args.attn_impl} device={device}")
    print("=" * 80, flush=True)

    if args.mode == "manifest":
        run_manifest(args, device, dtype)
    elif args.mode == "quality":
        run_quality(args, device, dtype)
    elif args.mode == "latency":
        run_latency(args, device, dtype)
    elif args.mode == "e0_h12_sanity":
        run_e0_h12_sanity(args, device, dtype)


if __name__ == "__main__":
    main()
