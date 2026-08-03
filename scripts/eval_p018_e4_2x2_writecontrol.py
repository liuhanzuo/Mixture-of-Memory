#!/usr/bin/env python
"""P0.18 — E4 two-factor (2x2) decomposition of the deployable Write gap (INFERENCE-ONLY, paired).

Splits the single number that P0.16 called "the deployable gap" (Arm A full-replay
minus Arm B chunk-local Write) into the TWO physically distinct factors that P0.16 /
Limitations lumped together, as a strictly-paired ``2x2`` diagnostic on the SAME 200
paired examples / SAME selected pack / SAME flagship LoRA as P0.13 / P1.7 / P0.16:

  factor 1  (lower-layer attention scope — how the cached ``h12`` is PRODUCED)
    * chunk-local        : ``layers[0:12]`` see ONLY within-chunk tokens (each chunk
                           encoded in isolation, RoPE 0:T per chunk). == Arm B write.
    * document-contextual: ``layers[0:12]`` run ONCE, fully-causal, over the WHOLE
                           document (RoPE = document positions); the per-token ``h12``
                           is then SLICED at the selected-chunk boundaries. == E0 write.

  factor 2  (READ position IDs — the RoPE coordinate each cached ``h12`` token is
             assigned when ``layers[12:36]`` resume over the pack)
    * local / reset      : FRESH CONTIGUOUS pack positions ``0:H`` (the deployable
                           Read interface; == Arm B / E0 read).
    * document-origin    : each token KEEPS its ORIGINAL document RoPE coordinate
                           (sink -> 0 ; selected chunk ``i`` token ``t`` -> doc pos
                           ``i*chunk_size + t`` ; query token ``t`` -> doc pos
                           ``q0 + t``). The pack is gapped/non-contiguous in RoPE space
                           but the ATTENTION CONNECTIVITY is UNCHANGED (still the same
                           full-causal pack read) — only the RoPE coordinate moves.

Crossing the two factors gives the 4 arms (+ an A anchor):

  * A   = ``resume_j=0`` full 36-layer continuous replay + flagship LoRA (RAG upper
          bound; == P0.13 / P1.7 / P0.16 Arm A). The paired reference for logit KL.
  * BB  = (chunk-local , local-pos)          = Arm B baseline (deployable endpoint;
          run VERBATIM through ``p017._run_arm`` -> bit-identical to the headline row).
  * E0  = (document-contextual , local-pos)  = P0.16 E0 (run VERBATIM through
          ``p016._run_e0`` -> bit-identical to the P0.16 E0 row).
  * X   = (chunk-local , document-origin-pos) NEW. Arm B's per-chunk h12, but at READ
          each token gets its document-origin RoPE coordinate.
  * Y   = (document-contextual , document-origin-pos) NEW. E0's document-contextual
          h12, read at document-origin RoPE coordinates (the "no repositioning at all"
          corner: both context AND coordinates are the document's own).

The four SINGLE-FACTOR controls (each changes EXACTLY one declared factor):
    BB -> E0 : flip factor 1 (lower-layer scope) at fixed factor 2 (local-pos).
    BB -> X  : flip factor 2 (read positions) at fixed factor 1 (chunk-local).
    E0 -> Y  : flip factor 2 (read positions) at fixed factor 1 (doc-contextual).
    X  -> Y  : flip factor 1 (lower-layer scope) at fixed factor 2 (doc-origin-pos).
The 2x2 lets us attribute the A-B gap: if BB->E0 explains it, the gap is LACK OF
DOCUMENT CONTEXT in the Write (=> P0.17 overlap Write). If BB->X explains it, the gap
is the store->read RoPE REPOSITIONING (=> learn a position interface at P1.10). If the
two single-factor effects do NOT add up to the joint effect (BB->Y), there is an
INTERACTION (both must be fixed jointly). Any sign is admissible; the paired CI +
McNemar is the deliverable and flows into the mechanism table.

--------------------------------------------------------------------------------------
HOW document-origin READ positions are made SAFELY (the enabling insight / the reason
none of the four arms has to be dropped):

  A NAIVE implementation would build the read attention mask by passing the gapped
  document-origin ``position_ids`` to ``transformers.masking_utils.create_causal_mask``.
  In transformers >=5 that path calls ``find_packed_sequence_indices(position_ids)``:
  ANY pair of consecutive positions differing by >1 is treated as a NEW packed
  sub-sequence, so a gapped position vector silently yields a BLOCK-DIAGONAL mask
  (each chunk isolated, query re-attends context) instead of the intended full-causal
  read. That would conflate factor 2 (RoPE coordinate) with an UNDECLARED change of
  attention connectivity (factor "1.5"), invalidating the single-factor claim.

  We AVOID that: RoPE and attention connectivity are decoupled on this backbone.
  Qwen3 attention applies RoPE from the ``position_embeddings=(cos,sin)`` tuple it is
  HANDED (``cos, sin = position_embeddings ; apply_rotary_pos_emb(q, k, cos, sin)``),
  and the attention MASK is a separate, explicitly-passed argument. So we:
    (i)  build the mask from CONTIGUOUS positions ``arange(H)`` (== the exact
         full-causal read Arm B / E0 use; ``find_packed_sequence_indices`` returns
         ``None`` for contiguous positions -> pure causal), and
    (ii) build ``(cos,sin)`` from the DOCUMENT-ORIGIN positions via
         ``rotary_emb(hidden, position_ids=doc_origin_positions)``.
  The layer then attends full-causally over the pack while every token's RoPE phase is
  its document coordinate — EXACTLY the single declared factor-2 change, nothing else.
  This is validated numerically by ``--mode pos_sanity``: feeding CONTIGUOUS positions
  through this SAME custom read/decode path must reproduce ``qc.read_prefill`` /
  ``decode_step`` (Arm B) to fp tolerance, proving the plumbing changes nothing but the
  RoPE coordinate.

  The KV-cache O(1) decode ``qcmem_model.read_prefill`` / ``decode_step`` HARD-CODE
  contiguous pack RoPE, so X / Y cannot call them; we implement an equivalent
  custom-position prefill + O(1) decode here using ONLY QCMemModel's public low-level
  accessors (``embed_tokens``, ``rotary_emb``, ``_run_layers``, ``norm``, ``lm_head``,
  ``_decode_attn_mask``, ``_make_mask_and_rope``) — never patching the backbone.

This file NEVER mutates ``eval_p016_e0_write_control.py``, ``bench_p1_7_h12_oracle.py``,
``bench_p0_13_quality_latency.py`` or ``qcmem_model.py``: it IMPORTS every shared
primitive read-only (so A / BB / E0 are byte-identical to their headline rows) and adds
ONLY the document-origin-position read/decode for the two new arms X / Y.
"""
from __future__ import annotations

import argparse
import contextlib
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
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the UNMODIFIED P0.16 harness (which imports P1.7 -> P0.13 verbatim) and pull
# every shared primitive from it. A / BB / E0 are therefore BIT-IDENTICAL to their
# P0.13 / P1.7 / P0.16 headline paths (same pack builder, same per-arm generate, same
# oracle, same loader, same strict-fix hashes, same stats, same E0 doc-slicing).
import eval_p016_e0_write_control as p016  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

p017 = p016.p017
ruler = p016.ruler
qcb = p016.qcb
QCMemModel = p016.QCMemModel
_bare_question = p016._bare_question
_resolve_task = p016._resolve_task
_build_pack = p016._build_pack
_packed_ids_from_pack = p016._packed_ids_from_pack
_run_arm = p016._run_arm                    # arms A (j0) + BB (chunk-local j12)
_run_e0 = p016._run_e0                       # arm E0 (doc-contextual, local-pos)
_e0_doc_spans = p016._e0_doc_spans           # doc-origin span map (fail-closed)
_e0_doc_lower12 = p016._e0_doc_lower12       # doc-contextual lower-12 forward
_e0_h12_residual = p016._e0_h12_residual     # E0 h12 numeric invariant (reused)
_stock_lower12_ref = p016._stock_lower12_ref
_load = p016._load
_eos_ids = p016._eos_ids
_sync = p016._sync
_peak_gb = p016._peak_gb
_pair_agree = p016._pair_agree
_macro_and_cells = p016._macro_and_cells
_pairwise = p016._pairwise
_agree_means = p016._agree_means
_mcnemar_exact = p016._mcnemar_exact
_backbone_provenance = p016._backbone_provenance
_lora_modules = p016._lora_modules
_versions = p016._versions
EXPECTED_LORA_SHA = p016.EXPECTED_LORA_SHA
EXPECTED_BACKBONE_KEY_SHA = p016.EXPECTED_BACKBONE_KEY_SHA
EXPECTED_LORA_MODULE_COUNT = p016.EXPECTED_LORA_MODULE_COUNT


# --------------------------------------------------------------------------- #
# BBWL arm (2026-08-04): the deployable chunk-local Write path (== Arm BB) but with
# the P1.10-trained WRITE LoRA loaded on layers [0..resume_j-1] and ENABLED only
# during the write phase. It quantifies how much of the Arm B (chunk-local, 92.5)
# -> E0 (document-contextual, 100) gap the trained Write recovers WITHOUT giving the
# lower band the whole document.
#
# Two-adapter design (why A/BB/E0/X/Y stay bit-identical):
#   * the flagship READ LoRA lives on layers 12..35 and is loaded as the LIVE peft
#     adapter "default" by ``_load`` (never merged);
#   * the trained WRITE LoRA lives on layers 0..11 (DISJOINT from READ) and is loaded
#     as a SECOND adapter "write". Because the two adapters target disjoint layers,
#     every layer holds exactly ONE of them in its lora ModuleDict:
#       - layers 12..35 -> only "default";  - layers 0..11 -> only "write".
#     With the active adapter set to "default" (the post-load default state), each
#     layer-0..11 ``lora.Linear`` falls through to ``base_layer(x)`` (no "default"
#     key in its ModuleDict) == the ORIGINAL ``nn.Linear`` -> BIT-IDENTICAL to a load
#     that never saw the write adapter. So A/BB/E0/X/Y (run with active=="default")
#     are numerically unchanged. The WRITE LoRA fires ONLY inside
#     ``_write_lora_enabled`` (active==["default","write"]), which wraps the BBWL arm:
#     then layers 0..11 apply ONLY "write" and layers 12..35 apply ONLY "default" —
#     i.e. Arm BB's exact write/read/decode pipeline plus the trained write on the
#     lower band. This reproduces the P1.10 trainer's student forward, where the READ
#     LoRA was merged into the base and the WRITE LoRA sat on layers 0..11 (layers
#     0..11 never carried READ either way, so the lower-band write is identical here).
# --------------------------------------------------------------------------- #
def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_with_write_lora(model_path, dtype, attn_impl, device, lora_adapter,
                          write_lora_ckpt, resume_j):
    """Reproduce p013 ``_load`` EXACTLY (base + flagship READ LoRA as the live
    "default" adapter on layers 12..35) but KEEP the ``PeftModel`` handle so the
    trained WRITE LoRA (layers 0..resume_j-1) can be added as a SECOND, disjoint
    adapter named "write". Returns ``(tokenizer, model, lora_sha256, lora_layers,
    peft_model, write_sha, write_layers)`` — the first four fields are IDENTICAL to
    ``_load``'s return so the caller path for A/BB/E0/X/Y is unchanged.

    The base + READ load mirrors ``bench_p0_13_quality_latency._load`` line for line
    (same ``AutoModelForCausalLM.from_pretrained`` args, same ``PeftModel.from_pretrained``
    ordering), so the resulting base+READ weights are bit-identical to the default
    harness. Loading the disjoint WRITE adapter and restoring ``set_adapter("default")``
    leaves every A/BB/E0/X/Y forward untouched (inactive lora.Linear == base passthrough).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if not lora_adapter:
        raise SystemExit("[p0.18][BBWL][ABORT] --lora_adapter (flagship READ) is "
                         "required to build the BBWL arm")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, attn_implementation=attn_impl,
        trust_remote_code=True, local_files_only=True,
    ).to(device).eval()

    print(f"[p0.18] loading flagship READ LoRA (adapter='default'): {lora_adapter}",
          flush=True)
    peft_model = PeftModel.from_pretrained(model, lora_adapter).eval()
    read_file = os.path.join(lora_adapter, "adapter_model.safetensors")
    lora_sha256 = _sha256_file(read_file) if os.path.exists(read_file) else None
    lora_layers = None
    read_cfg = os.path.join(lora_adapter, "adapter_config.json")
    if os.path.exists(read_cfg):
        with open(read_cfg) as f:
            lora_layers = json.load(f).get("layers_to_transform")

    print(f"[p0.18] loading trained WRITE LoRA (adapter='write'): {write_lora_ckpt}",
          flush=True)
    peft_model.load_adapter(write_lora_ckpt, adapter_name="write")
    # Restore the READ-only default state; the write adapter stays DORMANT until a
    # ``_write_lora_enabled`` block activates it. This is what guarantees A/BB/E0/X/Y
    # are numerically identical to a load without the write adapter.
    peft_model.base_model.set_adapter("default")
    model = peft_model.base_model.model

    write_file = os.path.join(write_lora_ckpt, "adapter_model.safetensors")
    write_sha = _sha256_file(write_file) if os.path.exists(write_file) else None
    write_layers = None
    write_cfg = os.path.join(write_lora_ckpt, "adapter_config.json")
    if os.path.exists(write_cfg):
        with open(write_cfg) as f:
            write_layers = json.load(f).get("layers_to_transform")

    # fail-closed: the trained WRITE LoRA MUST live on layers [0..resume_j-1] and be
    # DISJOINT from the READ LoRA (12..35), else the two-adapter bit-identity argument
    # (each layer holds exactly one adapter) breaks.
    exp_write_layers = list(range(0, int(resume_j)))
    if sorted(write_layers or []) != exp_write_layers:
        raise SystemExit(
            f"[p0.18][BBWL][ABORT] WRITE LoRA layers_to_transform {write_layers} != "
            f"expected {exp_write_layers} (must be the lower band 0..{resume_j - 1}, "
            f"disjoint from READ 12..35)")
    if set(lora_layers or []) & set(write_layers or []):
        raise SystemExit(
            f"[p0.18][BBWL][ABORT] READ {sorted(lora_layers or [])} and WRITE "
            f"{sorted(write_layers or [])} LoRA share layers — not disjoint")
    return (tokenizer, model, lora_sha256, lora_layers,
            peft_model, write_sha, write_layers)


@contextlib.contextmanager
def _write_lora_enabled(peft_model):
    """Activate BOTH the READ ("default", layers 12..35) and the trained WRITE
    ("write", layers 0..11) adapters for the duration of the block, then restore the
    READ-only "default" state. Because the two adapters live on DISJOINT layer sets,
    activating both makes layers 0..11 apply ONLY "write" and layers 12..35 apply ONLY
    "default" — i.e. Arm BB's exact pipeline plus the trained write on the lower band.
    Everything outside this block runs with active=="default" (READ only), so the
    other arms are unaffected."""
    tuner = peft_model.base_model            # LoraModel (BaseTuner.set_adapter)
    tuner.set_adapter(["default", "write"])
    try:
        yield
    finally:
        tuner.set_adapter("default")


# --------------------------------------------------------------------------- #
# document-origin READ position ids (the factor-2 coordinate vector).
# --------------------------------------------------------------------------- #
def _doc_origin_read_positions(sink_span, chunk_spans, query_span, pack_read_len,
                               doc_len, device):
    """Build the ``[1, H]`` RoPE position vector that assigns every packed h12 token
    its ORIGINAL DOCUMENT coordinate, in the EXACT packed order
    ``[sink ; selected chunks (doc order) ; query]``.

    Fail-closed: length must equal ``pack_read_len`` (the 1:1 pairing guarantee), sink
    must be document position 0, and every coordinate must lie in ``[0, doc_len)``."""
    pos_list = list(range(int(sink_span[0]), int(sink_span[1])))          # [0]
    for (cs, ce) in chunk_spans:
        pos_list.extend(range(int(cs), int(ce)))
    pos_list.extend(range(int(query_span[0]), int(query_span[1])))
    if len(pos_list) != int(pack_read_len):
        raise AssertionError(
            f"[p0.18][E4] doc-origin position vector len {len(pos_list)} != "
            f"pack_read_len {pack_read_len} — factor-2 mapping broke pairing")
    if not pos_list or pos_list[0] != 0:
        raise AssertionError(
            f"[p0.18][E4] doc-origin positions must start at sink==0; got "
            f"{pos_list[:3]}")
    mn, mx = min(pos_list), max(pos_list)
    if mn < 0 or mx >= int(doc_len):
        raise AssertionError(
            f"[p0.18][E4] doc-origin position range [{mn},{mx}] escapes document "
            f"[0,{doc_len})")
    pos = torch.tensor([pos_list], dtype=torch.long, device=device)
    return pos


def _assert_contiguous_causal_no_packsplit(qc, packed):
    """Fail-closed: the CONTIGUOUS-position mask this file uses for the doc-origin read
    must be the ordinary full-causal mask (``find_packed_sequence_indices`` returns
    ``None`` for contiguous positions), NOT a block-diagonal one. If a future
    transformers silently changed this, the single-factor claim would be void — so we
    assert it here rather than trust it."""
    from transformers.masking_utils import find_packed_sequence_indices
    H = int(packed.shape[1])
    contig = torch.arange(H, device=packed.device).unsqueeze(0)
    split = find_packed_sequence_indices(contig)
    if split is not None:
        raise AssertionError(
            "[p0.18][E4] contiguous positions unexpectedly split into packed "
            f"sub-sequences ({split.tolist()[:8]}…) — the doc-origin read would no "
            "longer be a clean single-factor change; aborting.")


# --------------------------------------------------------------------------- #
# document-origin-position READ prefill + O(1) decode (the factor-2 machinery).
# Uses ONLY QCMemModel public accessors; never patches the backbone.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _read_prefill_docpos(qc, sink_hj, selected_hj_list, query_hj, doc_read_positions):
    """Top-band prefill WITH a KV cache and DOCUMENT-ORIGIN RoPE positions.

    Identical to ``qc.read_prefill`` EXCEPT the RoPE ``(cos,sin)`` are built from
    ``doc_read_positions`` (each token's document coordinate) instead of the contiguous
    ``arange(H)``. The attention MASK is built from CONTIGUOUS positions so the read is
    the SAME full-causal pack read (no block-diagonal artifact) — the ONLY change vs.
    Arm B / E0 is the RoPE coordinate (factor 2). Returns
    ``(logits_last [1,1,V], top_cache, H)``."""
    pieces = []
    if sink_hj is not None:
        pieces.append(sink_hj)
    for h in selected_hj_list:
        if h is not None and h.shape[1] > 0:
            pieces.append(h)
    pieces.append(query_hj)
    packed = torch.cat(pieces, dim=1)                       # [1, H, d]
    H = packed.shape[1]
    if int(doc_read_positions.shape[1]) != H:
        raise AssertionError(
            f"[p0.18][E4] doc_read_positions len {int(doc_read_positions.shape[1])} "
            f"!= packed len {H}")
    _assert_contiguous_causal_no_packsplit(qc, packed)
    contig = torch.arange(H, device=qc.device).unsqueeze(0)
    causal_mask, _ = qc._make_mask_and_rope(packed, contig)   # mask from CONTIGUOUS pos
    position_embeddings = qc.rotary_emb(packed, position_ids=doc_read_positions)  # RoPE
    cache = DynamicCache(config=qc.config)
    hidden = qc._run_layers(
        packed, slice(qc.resume_j, qc.num_layers),
        causal_mask, doc_read_positions, position_embeddings,
        past_key_values=cache, use_cache=True,
    )
    last = qc.norm(hidden[:, -1:, :])
    logits_last = qc.lm_head(last)                            # [1,1,V]
    return logits_last, cache, H


@torch.no_grad()
def _decode_step_docpos(qc, token_id, bottom_cache, top_cache,
                        q_bottom_pos, top_doc_pos, bottom_kv_len, top_kv_len):
    """One O(1) decode step with a DOCUMENT-ORIGIN top-band RoPE coordinate.

    Mirrors ``qc.decode_step`` EXACTLY for the bottom band (``layers[0:a]`` at
    ``q_bottom_pos`` — the factor-1 write coordinate: query-local for chunk-local,
    document for doc-contextual), and for the top band (``layers[a:L]``) uses
    ``top_doc_pos`` (the document coordinate of the freshly-generated token =
    ``query_span[1] + step``) instead of the contiguous pack coordinate. The attention
    masks size to the true cache lengths (``bottom_kv_len`` / ``top_kv_len``); for
    SDPA/FlashAttention ``_decode_attn_mask`` returns ``None`` (attend-all, correct for
    a single causal-last query), so the RoPE coordinate is the sole change. Returns
    ``logits_last [1,1,V]``."""
    ids = torch.tensor([[int(token_id)]], device=qc.device, dtype=torch.long)
    emb = qc.embed_tokens(ids)                                # [1,1,d]
    if qc.resume_j > 0:
        b_pos = torch.tensor([[int(q_bottom_pos)]], device=qc.device)
        b_pe = qc.rotary_emb(emb, position_ids=b_pos)
        b_mask = qc._decode_attn_mask(int(bottom_kv_len))
        new_hj = qc._run_layers(
            emb, slice(0, qc.resume_j), b_mask, b_pos, b_pe,
            past_key_values=bottom_cache, use_cache=True,
        )
    else:
        new_hj = emb
    t_pos = torch.tensor([[int(top_doc_pos)]], device=qc.device)
    t_pe = qc.rotary_emb(new_hj, position_ids=t_pos)
    t_mask = qc._decode_attn_mask(int(top_kv_len))
    hidden = qc._run_layers(
        new_hj, slice(qc.resume_j, qc.num_layers), t_mask, t_pos, t_pe,
        past_key_values=top_cache, use_cache=True,
    )
    hidden = qc.norm(hidden)
    return qc.lm_head(hidden)                                 # [1,1,V]


@torch.no_grad()
def _run_docpos_arm(qc, tokenizer, pack, doc_ids, sink_span, chunk_spans, query_span,
                    factor1, max_new_tokens, eos_ids, capture_first=False):
    """Run a DOCUMENT-ORIGIN-POSITION read arm (X or Y).

    ``factor1`` selects how the cached ``h12`` is produced:
      * ``"chunk_local"`` (arm X) — sink/selected via ``write_chunk``/``write_chunks``
        (chunk-local, bit-identical to Arm B's write), query via ``write_prefill``
        (chunk-local, KV-cached; bottom decode continues in QUERY-LOCAL coords).
      * ``"doc_ctx"`` (arm Y) — continuous lower-12 over the WHOLE document
        (bit-identical to E0's write), sliced at the doc spans; bottom decode continues
        in DOCUMENT coords (the whole-document bottom KV is kept).

    Both then READ + O(1)-DECODE the top band over the pack with DOCUMENT-ORIGIN RoPE
    (factor 2). Returns the SAME 6-tuple as ``_run_arm`` / ``_run_e0`` plus a 7th item:
    the packed h12 states ``[1,H,d]`` (for the state-vs-doc-ctx metric)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    bos_id = pack["bos_id"]
    selected_chunk_tensors = pack["selected_chunk_tensors"]
    query_ids = pack["query_ids"]

    _sync(); tw0 = time.perf_counter()
    if factor1 == "chunk_local":
        sink_hj = qc.write_chunk([bos_id])
        selected_hj = qc.write_chunks(list(selected_chunk_tensors)) \
            if selected_chunk_tensors else []
        query_hj, bottom_cache, q_bottom_start = qc.write_prefill(query_ids)
        bottom_kv0 = int(q_bottom_start)                     # query-local cache length
    elif factor1 == "doc_ctx":
        h12_doc, bottom_cache, N = _e0_doc_lower12(qc, doc_ids, keep_cache=True)
        s0, s1 = sink_span
        sink_hj = h12_doc[:, s0:s1, :]
        selected_hj = [h12_doc[:, cs:ce, :] for (cs, ce) in chunk_spans]
        q0, q1 = query_span
        query_hj = h12_doc[:, q0:q1, :]
        q_bottom_start = N                                   # document coordinate
        bottom_kv0 = int(N)                                  # whole-document cache len
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unknown factor1 {factor1!r}")
    _sync(); t_write = time.perf_counter() - tw0

    # state-slicing fail-closed: piece lengths MUST equal the document span lengths.
    exp = [("sink", sink_hj.shape[1], sink_span[1] - sink_span[0])]
    for k, (cs, ce) in enumerate(chunk_spans):
        exp.append((f"chunk{k}", selected_hj[k].shape[1], ce - cs))
    exp.append(("query", query_hj.shape[1], query_span[1] - query_span[0]))
    for name, got, want in exp:
        if int(got) != int(want):
            raise AssertionError(
                f"[p0.18][E4][{factor1}] {name} state slice len {got} != span len "
                f"{want} — state slicing / pairing broken")

    packed_states = torch.cat([sink_hj] + list(selected_hj) + [query_hj], dim=1)
    H = int(packed_states.shape[1])
    doc_read_positions = _doc_origin_read_positions(
        sink_span, chunk_spans, query_span, pack["pack_read_len"],
        int(doc_ids.shape[0]), qc.device)

    _sync(); tr0 = time.perf_counter()
    logits1, top_cache, H2 = _read_prefill_docpos(
        qc, sink_hj, selected_hj, query_hj, doc_read_positions)
    _sync(); t_read = time.perf_counter() - tr0
    if H2 != H:
        raise AssertionError(f"[p0.18][E4] read pack len {H2} != state pack len {H}")

    read_len = (int(sink_hj.shape[1])
                + int(sum(h.shape[1] for h in selected_hj))
                + int(query_hj.shape[1]))

    first_logits = logits1[0, -1].float()
    finite = bool(torch.isfinite(first_logits).all().item())
    next_logits = first_logits.clone()
    if eos_ids:
        next_logits[eos_ids] = float("-inf")
    first_capture = first_logits.detach().cpu().clone() if capture_first else None

    _sync(); td0 = time.perf_counter()
    generated = []
    next_tok = int(next_logits.argmax().item())
    generated.append(next_tok)
    q_bottom_pos = int(q_bottom_start)          # bottom coordinate (factor-1 continue)
    top_doc_pos = int(query_span[1])            # top doc coordinate: continues at q1
    bottom_kv = bottom_kv0
    top_kv = H
    for _step in range(1, max_new_tokens):
        logits = _decode_step_docpos(
            qc, next_tok, bottom_cache, top_cache,
            q_bottom_pos, top_doc_pos, bottom_kv + 1, top_kv + 1)
        q_bottom_pos += 1
        top_doc_pos += 1
        bottom_kv += 1
        top_kv += 1
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
    return generated, timings, read_len, peak, finite, first_capture, packed_states


# --------------------------------------------------------------------------- #
# state / logit diagnostics.
# --------------------------------------------------------------------------- #
def _h12_state_metrics(states, ref_states):
    """Per-token layer-12 state agreement between an arm's packed h12 and the
    DOCUMENT-CONTEXTUAL stock lower-12 reference (== E0 / Y states). ``cosine==1`` and
    ``rel_l2==0`` iff the arm's h12 IS the document-contextual h12 (E0 / Y)."""
    a = states.float(); b = ref_states.float()
    if a.shape != b.shape:
        raise AssertionError(
            f"[p0.18][E4] h12 state shape {tuple(a.shape)} != ref {tuple(b.shape)}")
    cos = F.cosine_similarity(a, b, dim=-1)                  # [1,H]
    l2 = (a - b).norm(dim=-1)                                # [1,H]
    rel = l2 / (b.norm(dim=-1) + 1e-6)
    return {"h12_cosine_vs_docctx_mean": round(float(cos.mean().item()), 6),
            "h12_cosine_vs_docctx_min": round(float(cos.min().item()), 6),
            "h12_rel_l2_vs_docctx_mean": round(float(rel.mean().item()), 6),
            "h12_abs_l2_vs_docctx_mean": round(float(l2.mean().item()), 6),
            "H": int(a.shape[1])}


def _logit_kl_top1(ref_logits, arm_logits):
    """KL(softmax(A) || softmax(arm)) at the first read position + top-1 agreement vs
    the A (full-replay) anchor. ``ref_logits`` / ``arm_logits`` are ``[V]`` tensors."""
    if ref_logits is None or arm_logits is None:
        return {"kl_A_to_arm": None, "top1_match_vs_A": None}
    lp = F.log_softmax(ref_logits.float(), dim=-1)
    lq = F.log_softmax(arm_logits.float(), dim=-1)
    kl = float((lp.exp() * (lp - lq)).sum().item())
    top1 = bool(int(ref_logits.argmax()) == int(arm_logits.argmax()))
    return {"kl_A_to_arm": round(kl, 6), "top1_match_vs_A": top1}


# --------------------------------------------------------------------------- #
# QUALITY mode (5 arms on the identical pack + state/logit diagnostics)
# --------------------------------------------------------------------------- #
def run_quality(args, device, dtype):
    torch.manual_seed(args.seed)
    include_bbwl = bool(args.write_lora_ckpt)
    peft_model = None
    write_lora_sha256 = None
    write_lora_layers = None
    if include_bbwl:
        (tokenizer, model, lora_sha256, lora_layers,
         peft_model, write_lora_sha256, write_lora_layers) = _load_with_write_lora(
            args.model_path, dtype, args.attn_impl, device, args.lora_adapter,
            args.write_lora_ckpt, args.resume_j)
    else:
        tokenizer, model, lora_sha256, lora_layers = _load(
            args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L = int(model.config.num_hidden_layers)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p0.18][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    qcA = QCMemModel(model, resume_j=args.resume_j_a)     # 0  full replay (anchor)
    qcBB = QCMemModel(model, resume_j=args.resume_j)      # 12 chunk-local  local-pos
    qcE0 = QCMemModel(model, resume_j=args.resume_j)      # 12 doc-ctx      local-pos
    qcX = QCMemModel(model, resume_j=args.resume_j)       # 12 chunk-local  doc-origin
    qcY = QCMemModel(model, resume_j=args.resume_j)       # 12 doc-ctx      doc-origin
    # BBWL: chunk-local (== BB) but with the trained WRITE LoRA enabled during write.
    qcBBWL = QCMemModel(model, resume_j=args.resume_j) if include_bbwl else None
    eosA = _eos_ids(qcA, tokenizer)
    eosBB = _eos_ids(qcBB, tokenizer)
    eosE0 = _eos_ids(qcE0, tokenizer)
    eosX = _eos_ids(qcX, tokenizer)
    eosY = _eos_ids(qcY, tokenizer)
    eosBBWL = _eos_ids(qcBBWL, tokenizer) if include_bbwl else None
    include_e0 = not args.no_e0

    task = _resolve_task(args.task)
    length = args.length
    if length not in ruler._LENGTH_TOKENS:
        raise SystemExit(f"[p0.18] unknown length {length}")
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

    # Optional cross-check against a P0.13 / P1.7 / P0.16 pack manifest (same examples).
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

    print(f"[p0.18][quality] {task}/{length}{shard_tag}: selector={sel_name} "
          f"topk={args.topk} hop={args.iter_hop_topk} n={len(sample_indices)}/"
          f"{args.limit} A=j{args.resume_j_a} BB/E0/X/Y=j{args.resume_j} "
          f"e0={'on' if include_e0 else 'off'} "
          f"bbwl={'on' if include_bbwl else 'off'} mnt={mnt} verify={args.verify}",
          flush=True)
    if include_bbwl:
        print(f"[p0.18][quality] BBWL WRITE LoRA: {args.write_lora_ckpt} "
              f"sha={str(write_lora_sha256)[:12]}… layers={write_lora_layers}",
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
        doc_ids = input_ids[0]
        doc_len = int(doc_ids.shape[0])
        doc_sha = hashlib.sha256(
            b",".join(str(int(t)).encode() for t in doc_ids.tolist())).hexdigest()

        # ---- build the pack ONCE (forward-free, resume_j-independent) ----
        t_ret0 = time.perf_counter()
        pack = _build_pack(input_ids, args.chunk_size, sel_name, args.topk,
                           args.iter_hop_topk, bare_q_ids, tokenizer)
        retrieval_s = time.perf_counter() - t_ret0
        packed_ids = _packed_ids_from_pack(pack)     # verifies pack sha internally

        # ---- document-origin span map (fail-closed on any parity break) ----
        sink_span, chunk_spans, query_span = _e0_doc_spans(
            pack, args.chunk_size, doc_len)

        p013_sha_match = None
        if i in p013_shas:
            p013_sha_match = (p013_shas[i] == pack["packed_ids_sha256"])
            if not p013_sha_match:
                raise AssertionError(
                    f"[p0.18] example {i}: pack sha != P0.13/P1.7/P0.16 pack sha "
                    "— pairing with the existing arms broken")

        # ---- gates on the FIRST processed example ----
        h12_check = None
        pos_check = None
        if args.verify and not verified_once:
            # (a) E0/Y doc-contextual h12 invariant (reuses P0.16's numeric gate).
            pfx = min(int(args.h12_check_prefix), doc_len)
            h12_check = _e0_h12_residual(qcY, doc_ids[:pfx])
            h12_check["prefix_tokens"] = pfx
            print(f"[p0.18][verify] ex{i}: doc-ctx lower12 vs stock (prefix={pfx}) "
                  f"max_abs={h12_check['max_abs']:.3e} tol={args.h12_tol:.3e}",
                  flush=True)
            assert h12_check["max_abs"] < args.h12_tol, (
                f"[p0.18][ABORT] doc-ctx h12 residual {h12_check['max_abs']:.3e} >= "
                f"tol {args.h12_tol:.3e} — factor-1 invariant violated")
            # (b) plumbing gate: feeding CONTIGUOUS positions through the custom
            #     doc-pos read path must reproduce Arm B's read_prefill first logits
            #     (proves the custom path changes ONLY the RoPE coordinate).
            pos_check = _pos_sanity_one(qcX, pack, args.pos_tol)
            print(f"[p0.18][verify] ex{i}: pos-plumbing custom(contiguous) vs "
                  f"read_prefill max_abs={pos_check['max_abs']:.3e} "
                  f"cos={pos_check['cosine']:.6f} tol={args.pos_tol:.3e}", flush=True)
            assert pos_check["max_abs"] < args.pos_tol, (
                f"[p0.18][ABORT] custom-pos read plumbing residual "
                f"{pos_check['max_abs']:.3e} >= tol {args.pos_tol:.3e}")
            verified_once = True

        # ---- reference document-contextual packed h12 (for the state metric) ----
        h12_doc_ref = _e0_doc_lower12(qcY, doc_ids, keep_cache=False)[0]  # [1,N,d]
        ref_states = torch.cat(
            [h12_doc_ref[:, sink_span[0]:sink_span[1], :]]
            + [h12_doc_ref[:, cs:ce, :] for (cs, ce) in chunk_spans]
            + [h12_doc_ref[:, query_span[0]:query_span[1], :]], dim=1)  # [1,H,d]

        # ---- run the arms on the identical pack / identical document ----
        oom = False
        try:
            genA, tA, rlA, pkA, finA, lA = _run_arm(
                qcA, tokenizer, pack["bos_id"], pack["selected_chunk_tensors"],
                pack["query_ids"], mnt, eosA, capture_first=True)
            genBB, tBB, rlBB, pkBB, finBB, lBB = _run_arm(
                qcBB, tokenizer, pack["bos_id"], pack["selected_chunk_tensors"],
                pack["query_ids"], mnt, eosBB, capture_first=True)
            if include_e0:
                genE, tE, rlE, pkE, finE, lE = _run_e0(
                    qcE0, doc_ids, sink_span, chunk_spans, query_span,
                    mnt, eosE0, capture_first=True)
            genX, tX, rlX, pkX, finX, lX, statesX = _run_docpos_arm(
                qcX, tokenizer, pack, doc_ids, sink_span, chunk_spans, query_span,
                "chunk_local", mnt, eosX, capture_first=True)
            genY, tY, rlY, pkY, finY, lY, statesY = _run_docpos_arm(
                qcY, tokenizer, pack, doc_ids, sink_span, chunk_spans, query_span,
                "doc_ctx", mnt, eosY, capture_first=True)
            if include_bbwl:
                # BBWL == Arm BB (chunk-local, local-pos) run VERBATIM through the same
                # p017._run_arm machinery, but with the trained WRITE LoRA ENABLED on
                # layers 0..11 for the whole write/read/decode (the write phase +
                # decode bottom band go through the trained lower band; the read stays
                # the flagship READ LoRA, bit-identical to BB's read). Wrapping the
                # entire call activates ["default","write"]; outside it every other arm
                # runs with active=="default" (bit-identical to the E0 harness).
                with _write_lora_enabled(peft_model):
                    genBW, tBW, rlBW, pkBW, finBW, lBW = _run_arm(
                        qcBBWL, tokenizer, pack["bos_id"],
                        pack["selected_chunk_tensors"], pack["query_ids"],
                        mnt, eosBBWL, capture_first=True)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            oom = True
            torch.cuda.empty_cache()
            print(f"[p0.18][OOM] i={i} {task}/{length}: {e}", flush=True)

        if oom:
            rec = {"example_id": i, "task": task, "length": length,
                   "oom": True, "gold": " | ".join(answers)}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            records.append(rec); n_done += 1
            continue

        # chunk-local packed h12 (shared by BB & X) for the state metric.
        cl_sink = qcBB.write_chunk([pack["bos_id"]])
        cl_sel = qcBB.write_chunks(list(pack["selected_chunk_tensors"])) \
            if pack["selected_chunk_tensors"] else []
        cl_query = qcBB.write_chunk(pack["query_ids"])
        cl_states = torch.cat([cl_sink] + list(cl_sel) + [cl_query], dim=1)

        m_chunklocal = _h12_state_metrics(cl_states, ref_states)      # BB & X
        m_docctx = _h12_state_metrics(statesY, ref_states)            # Y (== E0); ~id
        # sanity: Y states MUST equal the doc-ctx reference (both are the doc-ctx h12).
        assert m_docctx["h12_cosine_vs_docctx_mean"] > 0.999, (
            f"[p0.18][ABORT] Y h12 not == doc-ctx reference "
            f"(cos={m_docctx['h12_cosine_vs_docctx_mean']}) — factor-1 mislabelled")

        # BBWL trained-write packed h12 (WRITE LoRA on layers 0..11): quantifies how
        # far the trained chunk-local write moves h12 TOWARD the document-contextual
        # reference (cos->1 / rel_l2->0 == it recovered document context).
        m_bbwl = None
        if include_bbwl:
            with _write_lora_enabled(peft_model):
                bw_sink = qcBBWL.write_chunk([pack["bos_id"]])
                bw_sel = qcBBWL.write_chunks(list(pack["selected_chunk_tensors"])) \
                    if pack["selected_chunk_tensors"] else []
                bw_query = qcBBWL.write_chunk(pack["query_ids"])
            bw_states = torch.cat([bw_sink] + list(bw_sel) + [bw_query], dim=1)
            m_bbwl = _h12_state_metrics(bw_states, ref_states)

        predA = tokenizer.decode(genA, skip_special_tokens=True).strip()
        predBB = tokenizer.decode(genBB, skip_special_tokens=True).strip()
        predX = tokenizer.decode(genX, skip_special_tokens=True).strip()
        predY = tokenizer.decode(genY, skip_special_tokens=True).strip()
        recA = ruler._string_match_all_one(predA, answers)
        recBB = ruler._string_match_all_one(predBB, answers)
        recX = ruler._string_match_all_one(predX, answers)
        recY = ruler._string_match_all_one(predY, answers)
        if include_e0:
            predE = tokenizer.decode(genE, skip_special_tokens=True).strip()
            recE = ruler._string_match_all_one(predE, answers)
        if include_bbwl:
            predBW = tokenizer.decode(genBW, skip_special_tokens=True).strip()
            recBW = ruler._string_match_all_one(predBW, answers)

        # 1:1 pairing guard: ALL arms MUST consume the identical pack length.
        pack_rl = pack["pack_read_len"]
        assert rlA == rlBB == rlX == rlY == pack_rl, (
            f"read_len mismatch i={i}: A={rlA} BB={rlBB} X={rlX} Y={rlY} "
            f"pack={pack_rl}")
        if include_e0:
            assert rlE == pack_rl, f"read_len mismatch i={i}: E0={rlE} pack={pack_rl}"
        if include_bbwl:
            assert rlBW == pack_rl, \
                f"read_len mismatch i={i}: BBWL={rlBW} pack={pack_rl}"

        def _arm_rec(resume_j, f1, f2, pred, rec_score, gen, rl, tt, pk, fin, first):
            d = {"resume_j": resume_j, "factor1": f1, "factor2": f2,
                 "prediction": pred, "score": rec_score,
                 "correct": bool(rec_score >= 1.0), "gen_len": len(gen),
                 "read_len": rl, "latency_s": tt, "peak_gb": pk, "finite": fin}
            d.update(_logit_kl_top1(lA, first))
            return d

        rec = {
            "example_id": i, "task": task, "length": length,
            "approx_tokens": approx_tokens,
            "gold": " | ".join(answers), "n_refs": len(answers),
            "doc_len": doc_len, "doc_ids_sha256": doc_sha,
            "retrieved_chunk_ids": pack["sel_idx"],
            "n_ctx_chunks": pack["n_ctx_chunks"], "chunk_size": args.chunk_size,
            "e0_doc_slices": {"sink_span": list(sink_span),
                              "chunk_spans": [list(s) for s in chunk_spans],
                              "query_span": list(query_span)},
            "pack_token_count": pack["pack_token_count"], "pack_read_len": pack_rl,
            "packed_ids_sha256": pack["packed_ids_sha256"],
            "p013_pack_sha_match": p013_sha_match,
            "lora_sha256": lora_sha256, "retrieval_s": retrieval_s,
            "h12_sanity": h12_check, "pos_sanity": pos_check,
            # per-factor h12 state metrics (chunk-local shared by BB&X; doc-ctx by E0&Y)
            "h12_state_chunklocal_vs_docctx": m_chunklocal,
            "h12_state_docctx_vs_docctx": m_docctx,
            # RoPE coordinate ranges (the factor-2 defining quantity)
            "rope_positions": {
                "local_read_pack_positions": [0, pack_rl],
                "docorigin_read_positions_sink": 0,
                "docorigin_read_positions_query": list(query_span),
                "docorigin_decode_top_start": query_span[1],
            },
            "armA": _arm_rec(args.resume_j_a, "full_replay", "n/a",
                             predA, recA, genA, rlA, tA, pkA, finA, lA),
            "armBB": _arm_rec(args.resume_j, "chunk_local", "local_pos",
                              predBB, recBB, genBB, rlBB, tBB, pkBB, finBB, lBB),
            "armX": _arm_rec(args.resume_j, "chunk_local", "doc_origin_pos",
                             predX, recX, genX, rlX, tX, pkX, finX, lX),
            "armY": _arm_rec(args.resume_j, "doc_contextual", "doc_origin_pos",
                             predY, recY, genY, rlY, tY, pkY, finY, lY),
            # single-factor diffs (accuracy; the 2x2 cell contrasts)
            "diff_A_minus_BB": recA - recBB,
            "diff_f1_BB_to_E0_localpos": None,        # filled below if E0 on
            "diff_f2_BB_to_X_chunklocal": recX - recBB,
            "diff_f2_E0_to_Y_docctx": None,           # filled below if E0 on
            "diff_f1_X_to_Y_docorigin": recY - recX,
            "diff_joint_BB_to_Y": recY - recBB,
        }
        agree = {"A_vs_BB": _pair_agree(genA, lA, genBB, lBB),
                 "A_vs_X": _pair_agree(genA, lA, genX, lX),
                 "A_vs_Y": _pair_agree(genA, lA, genY, lY),
                 "BB_vs_X": _pair_agree(genBB, lBB, genX, lX),
                 "X_vs_Y": _pair_agree(genX, lX, genY, lY)}
        if include_e0:
            rec["armE0"] = _arm_rec(args.resume_j, "doc_contextual", "local_pos",
                                    predE, recE, genE, rlE, tE, pkE, finE, lE)
            rec["diff_f1_BB_to_E0_localpos"] = recE - recBB
            rec["diff_f2_E0_to_Y_docctx"] = recY - recE
            agree["A_vs_E0"] = _pair_agree(genA, lA, genE, lE)
            agree["BB_vs_E0"] = _pair_agree(genBB, lBB, genE, lE)
            agree["E0_vs_Y"] = _pair_agree(genE, lE, genY, lY)
        if include_bbwl:
            # BBWL = chunk-local Write with the trained WRITE LoRA (== BB + trained
            # lower band). factor2 is still local_pos (BB's deployable Read interface).
            rec["armBBWL"] = _arm_rec(
                args.resume_j, "chunk_local_trained_write", "local_pos",
                predBW, recBW, genBW, rlBW, tBW, pkBW, finBW, lBW)
            rec["h12_state_bbwl_vs_docctx"] = m_bbwl
            # accuracy of the trained Write path: how much of BB->E0 it recovers.
            rec["diff_BBWL_minus_BB"] = recBW - recBB          # trained-write gain
            rec["diff_A_minus_BBWL"] = recA - recBW            # residual to full replay
            rec["diff_E0_minus_BBWL"] = (recE - recBW) if include_e0 else None
            agree["A_vs_BBWL"] = _pair_agree(genA, lA, genBW, lBW)
            agree["BB_vs_BBWL"] = _pair_agree(genBB, lBB, genBW, lBW)
            if include_e0:
                agree["E0_vs_BBWL"] = _pair_agree(genE, lE, genBW, lBW)
        rec["agreement"] = agree

        fout.write(json.dumps(rec) + "\n"); fout.flush()
        records.append(rec); n_done += 1
        torch.cuda.empty_cache()
        if n_done % 5 == 0:
            estr = f" E0={recE:.2f}" if include_e0 else ""
            bwstr = f" BBWL={recBW:.2f}" if include_bbwl else ""
            print(f"[p0.18][quality] {task}/{length}{shard_tag} {n_done} done "
                  f"(A={recA:.2f} BB={recBB:.2f}{estr}{bwstr} X={recX:.2f} "
                  f"Y={recY:.2f} rl={rlA})", flush=True)
    fout.close()

    valid = [r for r in records if not r.get("oom")]

    def _mean(arm):
        xs = [r[arm]["score"] for r in valid if arm in r]
        return round(sum(xs) / len(xs) * 100.0, 3) if xs else 0.0
    cell = {
        "task": task, "length": length, "shard": shard_tag,
        "n": len(records), "n_valid": len(valid),
        "oom_count": sum(1 for r in records if r.get("oom")),
        "armA_score": _mean("armA"), "armBB_score": _mean("armBB"),
        "armX_score": _mean("armX"), "armY_score": _mean("armY"),
        "e0_included": include_e0,
        "selector": sel_name, "topk": args.topk, "iter_hop_topk": args.iter_hop_topk,
        "chunk_size": args.chunk_size, "max_new_tokens": mnt,
        "resume_j_a": args.resume_j_a, "resume_j": args.resume_j,
        "lora_sha256": lora_sha256, "num_layers": L,
        "runtime": {"node": socket.gethostname(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "device": args.device,
                    "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                    "seed": args.seed, "dtype": args.dtype,
                    "attn_implementation": args.attn_impl},
        "jsonl": str(jsonl_path),
    }
    if include_e0:
        cell["armE0_score"] = _mean("armE0")
    if include_bbwl:
        cell["armBBWL_score"] = _mean("armBBWL")
        cell["diff_BBWL_minus_BB"] = round(_mean("armBBWL") - _mean("armBB"), 3)
        cell["write_lora_ckpt"] = args.write_lora_ckpt
        cell["write_lora_sha256"] = write_lora_sha256
        cell["write_lora_layers"] = write_lora_layers
        if include_e0:
            cell["diff_E0_minus_BBWL"] = round(_mean("armE0") - _mean("armBBWL"), 3)
    with open(outdir / f"{task}_{length}{shard_tag}_cell.json", "w") as f:
        json.dump(cell, f, indent=2)
    estr = f" E0={cell.get('armE0_score')}" if include_e0 else ""
    bwstr = f" BBWL={cell.get('armBBWL_score')}" if include_bbwl else ""
    print(f"[p0.18][quality] DONE {task}/{length}{shard_tag}: "
          f"A={cell['armA_score']} BB={cell['armBB_score']}{estr}{bwstr} "
          f"X={cell['armX_score']} Y={cell['armY_score']} n_valid={len(valid)}",
          flush=True)


# --------------------------------------------------------------------------- #
# POS_SANITY: custom doc-pos read with CONTIGUOUS positions must == read_prefill.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _pos_sanity_one(qc, pack, tol):
    """Build the chunk-local pack, run BOTH (a) ``qc.read_prefill`` (contiguous, the
    Arm B path) and (b) ``_read_prefill_docpos`` with CONTIGUOUS positions, and compare
    the first-step logits. They must match to ``tol`` — proving the custom doc-pos
    read/decode plumbing is a no-op when positions are contiguous, so the ONLY thing
    the X/Y arms change vs. Arm B is the RoPE coordinate."""
    sink_hj = qc.write_chunk([pack["bos_id"]])
    selected_hj = qc.write_chunks(list(pack["selected_chunk_tensors"])) \
        if pack["selected_chunk_tensors"] else []
    query_hj = qc.write_chunk(pack["query_ids"])
    ref_logits, _, H = qc.read_prefill(sink_hj, selected_hj, query_hj)
    contig = torch.arange(H, device=qc.device).unsqueeze(0)
    got_logits, _, H2 = _read_prefill_docpos(qc, sink_hj, selected_hj, query_hj, contig)
    assert H2 == H, f"[p0.18][pos_sanity] pack len mismatch {H2} != {H}"
    a = ref_logits[0, -1].float(); b = got_logits[0, -1].float()
    diff = (a - b).abs()
    cos = float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item())
    return {"max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "cosine": cos, "top1_match": bool(int(a.argmax()) == int(b.argmax())),
            "H": int(H)}


def run_pos_sanity(args, device, dtype):
    torch.manual_seed(args.seed)
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    if lora_sha256 != EXPECTED_LORA_SHA:
        raise SystemExit(
            f"[p0.18][ABORT] LoRA sha {lora_sha256} != expected {EXPECTED_LORA_SHA}")
    qc = QCMemModel(model, resume_j=args.resume_j)
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
    res = _pos_sanity_one(qc, pack, args.pos_tol)
    print("=" * 72)
    print(f"[p0.18][pos_sanity] {task}/{length} ex{i} H={res['H']}: "
          f"custom-docpos(contiguous) vs read_prefill max_abs={res['max_abs']:.3e} "
          f"mean_abs={res['mean_abs']:.3e} cos={res['cosine']:.6f} "
          f"top1={res['top1_match']} tol={args.pos_tol:.3e}")
    print("=" * 72, flush=True)
    assert res["max_abs"] < args.pos_tol, (
        f"[p0.18][ABORT] custom-pos read residual {res['max_abs']:.3e} >= tol "
        f"{args.pos_tol:.3e} — doc-origin read plumbing is not a clean single factor")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "pos_sanity.json", "w") as f:
        json.dump({"task": task, "length": length, "example_index": i,
                   "resume_j": args.resume_j, "pos_tol": args.pos_tol,
                   "result": res, "passed": True}, f, indent=2)
    print("[p0.18][pos_sanity] PASS — wrote pos_sanity.json", flush=True)


# --------------------------------------------------------------------------- #
# MANIFEST mode (strict-fix verification + provenance dump)
# --------------------------------------------------------------------------- #
def run_manifest(args, device, dtype):
    torch.manual_seed(args.seed)
    include_bbwl = bool(args.write_lora_ckpt)
    write_lora_sha256 = None
    write_lora_layers = None
    if include_bbwl:
        (tokenizer, model, lora_sha256, lora_layers, _peft,
         write_lora_sha256, write_lora_layers) = _load_with_write_lora(
            args.model_path, dtype, args.attn_impl, device, args.lora_adapter,
            args.write_lora_ckpt, args.resume_j)
    else:
        tokenizer, model, lora_sha256, lora_layers = _load(
            args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    qcA = QCMemModel(model, resume_j=args.resume_j_a)
    _ = QCMemModel(model, resume_j=args.resume_j)
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
    # The WRITE adapter adds resume_j x 7 LoRA modules on the DISJOINT lower band
    # (layers 0..resume_j-1), so the expected count grows by exactly that when BBWL is
    # loaded; the READ (12..35) count is unchanged.
    exp_lora_modules = EXPECTED_LORA_MODULE_COUNT + (int(args.resume_j) * 7
                                                     if include_bbwl else 0)
    if prov_lora["count"] != exp_lora_modules:
        abort.append(f"LoRA module count {prov_lora['count']} != {exp_lora_modules}")
    if sorted(lora_layers or []) != list(range(12, 36)):
        abort.append(f"READ LoRA layers_to_transform {lora_layers} != [12..35]")
    if include_bbwl:
        if sorted(write_lora_layers or []) != list(range(0, int(args.resume_j))):
            abort.append(f"WRITE LoRA layers_to_transform {write_lora_layers} != "
                         f"[0..{int(args.resume_j) - 1}]")
        if set(lora_layers or []) & set(write_lora_layers or []):
            abort.append("READ and WRITE LoRA layers overlap (must be disjoint)")

    manifest = {
        "run": "P0.18_E4_two_factor_2x2_write_control",
        "factors": {
            "factor1_lower_layer_scope": ["chunk_local", "document_contextual"],
            "factor2_read_position_ids": ["local_reset", "document_origin"],
        },
        "arms": {
            "A": {"resume_j": args.resume_j_a, "factor1": "full_replay",
                  "factor2": "n/a",
                  "note": "flagship LoRA full 36-layer continuous replay "
                          "(== P0.13/P1.7/P0.16 Arm A); logit-KL / top1 anchor"},
            "BB": {"resume_j": args.resume_j, "factor1": "chunk_local",
                   "factor2": "local_pos",
                   "note": "== P0.16/P0.13 Arm B (deployable). Run VERBATIM through "
                           "p017._run_arm -> bit-identical to the headline row."},
            "E0": {"resume_j": args.resume_j, "factor1": "document_contextual",
                   "factor2": "local_pos", "included": (not args.no_e0),
                   "note": "== P0.16 E0. Run VERBATIM through p016._run_e0."},
            "X": {"resume_j": args.resume_j, "factor1": "chunk_local",
                  "factor2": "document_origin_pos",
                  "note": "NEW. Arm B's per-chunk h12, READ with document-origin RoPE "
                          "coordinates (mask still full-causal from contiguous "
                          "positions; only RoPE (cos,sin) moves). Isolates factor 2 "
                          "at fixed factor 1."},
            "Y": {"resume_j": args.resume_j, "factor1": "document_contextual",
                  "factor2": "document_origin_pos",
                  "note": "NEW. E0's document-contextual h12 READ at document-origin "
                          "RoPE coordinates (no repositioning at all)."},
        },
        "single_factor_controls": {
            "BB->E0 (flip factor1 @ local_pos)": "lower-layer document context value",
            "BB->X  (flip factor2 @ chunk_local)": "read RoPE repositioning cost",
            "E0->Y  (flip factor2 @ doc_ctx)": "read RoPE repositioning cost (ctx-aware)",
            "X->Y   (flip factor1 @ doc_origin)": "context value at doc-origin coords",
            "interaction": "diff_joint_BB_to_Y vs (BB->E0)+(BB->X): non-additive => "
                           "factors interact (must be fixed jointly).",
        },
        "docpos_read_safety": (
            "Document-origin RoPE is applied via rotary_emb(position_ids=doc_origin) "
            "while the attention mask is built from CONTIGUOUS positions, so "
            "find_packed_sequence_indices does NOT split the pack into a block-diagonal "
            "mask. Validated by --mode pos_sanity (custom read with contiguous "
            "positions == qc.read_prefill to fp tolerance)."),
        "strict_fixes": {
            "model_path": args.model_path, "lora_adapter": args.lora_adapter,
            "lora_sha256": lora_sha256, "expected_lora_sha256": EXPECTED_LORA_SHA,
            "lora_sha_match": lora_sha256 == EXPECTED_LORA_SHA,
            "lora_layers_to_transform": lora_layers,
            "lora_module_count": prov_lora["count"],
            "selector": "iter_bm25", "topk": args.topk,
            "iter_hop_topk": args.iter_hop_topk, "sink_tokens": "bos",
            "chunk_size": args.chunk_size, "chat_template": False,
            "enable_thinking": False, "add_bos": 0, "dtype": args.dtype,
            "attn_impl": args.attn_impl, "max_new_tokens": args.max_new_tokens,
            "seed": args.seed, "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "provenance": {"backbone": prov_backbone, "lora": prov_lora,
                       "versions": prov_versions},
        "command": " ".join(sys.argv), "abort_reasons": abort,
    }
    if include_bbwl:
        # Pure increment: BBWL == BB but with the P1.10-trained WRITE LoRA enabled
        # (peft adapter "write", layers 0..resume_j-1) ONLY during the write phase.
        # The five existing arms (A/BB/E0/X/Y) keep the WRITE adapter DISABLED
        # (active set = ["default"]), so their forward is bit-identical to a load
        # without the write ckpt.
        manifest["arms"]["BBWL"] = {
            "resume_j": args.resume_j, "factor1": "chunk_local_trained_write",
            "factor2": "local_pos", "included": True,
            "note": "NEW (P1.10). == BB (chunk-local, local-pos) but the trained "
                    "WRITE LoRA (layers 0..%d) is enabled during the write phase via "
                    "peft set_adapter([\"default\",\"write\"]); disabled elsewhere. "
                    "Quantifies how much a trained deployable Write recovers of the "
                    "BB(92.5)->E0(100) document-context gap." % (int(args.resume_j) - 1),
        }
        manifest["strict_fixes"]["write_lora_ckpt"] = args.write_lora_ckpt
        manifest["strict_fixes"]["write_lora_sha256"] = write_lora_sha256
        manifest["strict_fixes"]["write_lora_layers"] = write_lora_layers
        manifest["strict_fixes"]["lora_adapter_names"] = prov_lora.get("adapter_names")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    if abort:
        print("[p0.18][manifest][ABORT] strict-fix mismatch:", flush=True)
        for a in abort:
            print("   - " + a, flush=True)
        sys.exit(3)
    print(f"[p0.18][manifest] OK — LoRA sha {lora_sha256[:12]}… "
          f"{prov_lora['count']} modules, layers [12..35]; "
          f"torch {prov_versions['torch']} tf {prov_versions['transformers']} "
          f"peft {prov_versions['peft']} git {prov_versions['git_commit_short']}",
          flush=True)


# --------------------------------------------------------------------------- #
# AGGREGATE mode (pure CPU: 5-arm per-cell + single-factor paired stats + 2x2)
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
        print(f"[p0.18][aggregate] no valid records under {qdir} — nothing to "
              "aggregate (run --mode quality first).", flush=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(outdir / "summary.json", "w") as f:
            json.dump({"n_examples_paired": 0, "note": "no quality records found"},
                      f, indent=2)
        return
    have_e0 = bool(valid) and all("armE0" in r for r in valid)
    have_bbwl = bool(valid) and all("armBBWL" in r for r in valid)

    arms = (["armA", "armBB", "armX", "armY"]
            + (["armE0"] if have_e0 else [])
            + (["armBBWL"] if have_bbwl else []))
    macros = {}
    cell_maps = {}
    cells = None
    for a in arms:
        m, cm, cc = _macro_and_cells(valid, a)
        macros[a] = m; cell_maps[a] = cm; cells = cc
    cells_keys = sorted(cells.keys())

    # single-factor pairwise contrasts (each flips exactly one factor)
    pairs = [
        ("A_vs_BB", "armA", "armBB"),
        ("f2_BB_to_X_chunklocal", "armX", "armBB"),
        ("f1_X_to_Y_docorigin", "armY", "armX"),
        ("joint_BB_to_Y", "armY", "armBB"),
    ]
    if have_e0:
        pairs += [("f1_BB_to_E0_localpos", "armE0", "armBB"),
                  ("f2_E0_to_Y_docctx", "armY", "armE0")]
    if have_bbwl:
        # BBWL vs BB isolates the trained-Write accuracy gain; E0 vs BBWL is the
        # residual gap to the (non-deployable) document-contextual Write.
        pairs += [("BBWL_vs_BB", "armBBWL", "armBB")]
        if have_e0:
            pairs += [("E0_vs_BBWL", "armE0", "armBBWL")]
    pairwise = {name: _pairwise(valid, cells_keys, cells, ax, ay, args.n_boot)
                for (name, ax, ay) in pairs}

    agree_pairs = ["A_vs_BB", "A_vs_X", "A_vs_Y", "BB_vs_X", "X_vs_Y"]
    if have_e0:
        agree_pairs += ["A_vs_E0", "BB_vs_E0", "E0_vs_Y"]
    if have_bbwl:
        agree_pairs += ["A_vs_BBWL", "BB_vs_BBWL"] + (["E0_vs_BBWL"] if have_e0 else [])
    agreement = {p: _agree_means(valid, p) for p in agree_pairs
                 if valid and p in valid[0].get("agreement", {})}

    # logit-KL-to-A and top1-vs-A per arm (mean over paired examples)
    def _kl_top1_means(arm):
        kls = [r[arm]["kl_A_to_arm"] for r in valid
               if arm in r and r[arm].get("kl_A_to_arm") is not None]
        t1 = [r[arm]["top1_match_vs_A"] for r in valid
              if arm in r and r[arm].get("top1_match_vs_A") is not None]
        return {"kl_A_to_arm_mean": round(sum(kls) / len(kls), 6) if kls else None,
                "top1_match_vs_A_rate": round(sum(1 for x in t1 if x) / len(t1), 4)
                if t1 else None}
    logit_vs_A = {a: _kl_top1_means(a) for a in arms if a != "armA"}

    # h12 state metrics (factor-1 property): chunk-local vs doc-ctx reference.
    def _state_means(key):
        cos = [r[key]["h12_cosine_vs_docctx_mean"] for r in valid if key in r]
        rl2 = [r[key]["h12_rel_l2_vs_docctx_mean"] for r in valid if key in r]
        return {"cosine_mean": round(sum(cos) / len(cos), 6) if cos else None,
                "rel_l2_mean": round(sum(rl2) / len(rl2), 6) if rl2 else None,
                "n": len(cos)}
    h12_states = {
        "chunk_local_vs_docctx": _state_means("h12_state_chunklocal_vs_docctx"),
        "docctx_vs_docctx": _state_means("h12_state_docctx_vs_docctx"),
    }
    if have_bbwl:
        h12_states["bbwl_trained_vs_docctx"] = _state_means(
            "h12_state_bbwl_vs_docctx")

    per_cell = {}
    for key in cells_keys:
        tag = f"{key[0]}/{key[1]}"
        entry = {"n": len(cells[key])}
        for a in arms:
            entry[a] = cell_maps[a][tag]
        entry["diff_f2_BB_to_X"] = round(cell_maps["armX"][tag]
                                         - cell_maps["armBB"][tag], 2)
        entry["diff_f1_X_to_Y"] = round(cell_maps["armY"][tag]
                                        - cell_maps["armX"][tag], 2)
        entry["diff_joint_BB_to_Y"] = round(cell_maps["armY"][tag]
                                            - cell_maps["armBB"][tag], 2)
        if have_e0:
            entry["diff_f1_BB_to_E0"] = round(cell_maps["armE0"][tag]
                                              - cell_maps["armBB"][tag], 2)
            entry["diff_f2_E0_to_Y"] = round(cell_maps["armY"][tag]
                                             - cell_maps["armE0"][tag], 2)
        if have_bbwl:
            entry["diff_BBWL_minus_BB"] = round(cell_maps["armBBWL"][tag]
                                                - cell_maps["armBB"][tag], 2)
            if have_e0:
                entry["diff_E0_minus_BBWL"] = round(cell_maps["armE0"][tag]
                                                    - cell_maps["armBBWL"][tag], 2)
        per_cell[tag] = entry

    macro = {a: round(macros[a], 3) for a in arms}
    # additivity check: does (BB->X)+(X->Y) == (BB->Y)?  (always true by telescoping);
    # the INTERACTION test is (BB->E0)+(BB->X) vs (BB->Y): if E0 present.
    interaction = None
    if have_e0:
        e_f1 = macros["armE0"] - macros["armBB"]      # factor1 alone (@ local pos)
        e_f2 = macros["armX"] - macros["armBB"]       # factor2 alone (@ chunk-local)
        e_joint = macros["armY"] - macros["armBB"]    # both flipped
        interaction = {
            "factor1_effect_localpos_BB_to_E0": round(e_f1, 3),
            "factor2_effect_chunklocal_BB_to_X": round(e_f2, 3),
            "joint_effect_BB_to_Y": round(e_joint, 3),
            "sum_of_single_effects": round(e_f1 + e_f2, 3),
            "interaction_residual": round(e_joint - (e_f1 + e_f2), 3),
            "note": "interaction_residual ~ 0 => factors are additive/separable; "
                    "large |residual| => they interact (must be fixed jointly).",
        }

    n = len(valid)
    any_oom = sum(1 for r in all_recs if r.get("oom"))

    def _fin(r):
        return all(r[a]["finite"] for a in arms if a in r)
    any_nonfinite = sum(1 for r in valid if not _fin(r))

    def _paired_rl(r):
        return all(r[a]["read_len"] == r["pack_read_len"] for a in arms if a in r)
    pack_paired = all(_paired_rl(r) for r in valid)
    p013_checked = [r for r in valid if r.get("p013_pack_sha_match") is not None]
    p013_all_match = all(r["p013_pack_sha_match"] for r in p013_checked) \
        if p013_checked else None

    summary = {
        "n_examples_paired": n, "n_cells": len(cells_keys), "e0_included": have_e0,
        "bbwl_included": have_bbwl,
        "per_cell": per_cell, "macro": macro, "interaction_2x2": interaction,
        "logit_vs_A": logit_vs_A, "h12_state_metrics": h12_states,
        "oom_examples": any_oom, "nonfinite_examples": any_nonfinite,
        "all_packs_paired_1to1": pack_paired,
        "p013_pack_sha_checked": len(p013_checked),
        "p013_pack_sha_all_match": p013_all_match,
        "attribution_hint": (
            "Read the 2x2: A-BB is the full deployable gap. factor2 (BB->X) isolates "
            "the store->read RoPE repositioning cost at chunk-local h12; factor1 "
            "(BB->E0, if on) isolates the missing document-context. If BB->X ~ A-BB, "
            "the gap is the position interface (=> learn positions at P1.10); if "
            "BB->E0 ~ A-BB, it is Write context (=> P0.17 overlap Write). "
            "interaction_residual != 0 => both must be fixed jointly. Any sign is "
            "admissible; paired CI + McNemar is the deliverable."),
    }
    stats = {"macro": macro, "pairwise": pairwise, "agreement": agreement,
             "interaction_2x2": interaction, "logit_vs_A": logit_vs_A,
             "h12_state_metrics": h12_states, "bootstrap_n": args.n_boot}
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(outdir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("=" * 78)
    print(f"[p0.18][aggregate] n_paired={n} n_cells={len(cells_keys)} "
          f"e0={'on' if have_e0 else 'off'} bbwl={'on' if have_bbwl else 'off'}")
    estr = f"  E0={macros['armE0']:.2f}" if have_e0 else ""
    bwstr = f"  BBWL={macros['armBBWL']:.2f}" if have_bbwl else ""
    print(f"  macro  A={macros['armA']:.2f}  BB={macros['armBB']:.2f}{estr}{bwstr}  "
          f"X={macros['armX']:.2f}  Y={macros['armY']:.2f}")
    for name in [p[0] for p in pairs]:
        pw = pairwise[name]
        print(f"  {name}: diff={pw['macro_diff']:+.2f} "
              f"CI={pw['paired_bootstrap_95ci']} "
              f"McNemar p={pw['mcnemar']['exact_two_sided_p']:.3g}")
    if interaction:
        print(f"  2x2 interaction_residual="
              f"{interaction['interaction_residual']:+.2f} "
              f"(f1={interaction['factor1_effect_localpos_BB_to_E0']:+.2f} "
              f"f2={interaction['factor2_effect_chunklocal_BB_to_X']:+.2f} "
              f"joint={interaction['joint_effect_BB_to_Y']:+.2f})")
    print(f"  h12 chunk-local vs doc-ctx: cos="
          f"{h12_states['chunk_local_vs_docctx']['cosine_mean']} "
          f"rel_l2={h12_states['chunk_local_vs_docctx']['rel_l2_mean']}")
    print(f"  packs_paired_1to1={pack_paired} p013_sha_match={p013_all_match} "
          f"oom={any_oom} nonfinite={any_nonfinite}")
    print("=" * 78, flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="P0.18 E4 two-factor 2x2 Write-control decomposition (paired)")
    ap.add_argument("--mode", required=True,
                    choices=["manifest", "quality", "aggregate", "pos_sanity"])
    ap.add_argument("--model_path", type=str, default="models/Qwen3-8b-local")
    ap.add_argument("--lora_adapter", type=str,
                    default="outputs/qcmem_distill_qwen_j12_r32_4k/final")
    ap.add_argument("--write_lora_ckpt", type=str, default="",
                    help="path to a P1.10-trained WRITE LoRA step dir (adapter_config"
                         ".json + adapter_model.safetensors; layers 0..resume_j-1, "
                         "r32/α64). When set, adds the BBWL arm = Arm BB (chunk-local, "
                         "local-pos) with the trained WRITE LoRA ENABLED during the "
                         "write phase. Empty (default) => A/BB/E0/X/Y only, "
                         "bit-identical to the existing P0.18 harness.")
    ap.add_argument("--resume_j_a", type=int, default=0)   # full replay (anchor A)
    ap.add_argument("--resume_j", type=int, default=12)    # BB/E0/X/Y split depth
    ap.add_argument("--no_e0", action="store_true",
                    help="drop the E0 (doc-ctx, local-pos) arm; keep A/BB/X/Y only "
                         "(then the E0-based single-factor controls are unavailable)")
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
                    default="bench_results/p0_18_e4_2x2")
    ap.add_argument("--p013_manifest_dir", type=str, default="",
                    help="optional P0.13/P1.7/P0.16 output dir to cross-check pack shas")
    ap.add_argument("--verify", action="store_true",
                    help="in quality mode, run the doc-ctx h12 invariant AND the "
                         "doc-pos read-plumbing sanity on the first processed example")
    ap.add_argument("--h12_tol", type=float, default=5e-2,
                    help="bf16 max-abs tolerance for the doc-ctx h12 invariant")
    ap.add_argument("--h12_check_prefix", type=int, default=1024)
    ap.add_argument("--pos_tol", type=float, default=5e-2,
                    help="bf16 max-abs tolerance for the doc-pos read-plumbing sanity "
                         "(custom read with contiguous positions == read_prefill)")
    ap.add_argument("--example_index", type=int, default=0)
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
    print(f"P0.18 E4 :: mode={args.mode} task={args.task} length={args.length} "
          f"shard={args.shard_index}/{args.num_shards}")
    print(f"  model={args.model_path} lora={args.lora_adapter}")
    print(f"  A=j{args.resume_j_a}(full) BB/E0/X/Y=j{args.resume_j} "
          f"e0={'off' if args.no_e0 else 'on'} topk={args.topk} "
          f"hop={args.iter_hop_topk} chunk={args.chunk_size} dtype={dtype} "
          f"attn={args.attn_impl} device={device}")
    print("=" * 80, flush=True)

    if args.mode == "manifest":
        run_manifest(args, device, dtype)
    elif args.mode == "quality":
        run_quality(args, device, dtype)
    elif args.mode == "pos_sanity":
        run_pos_sanity(args, device, dtype)


if __name__ == "__main__":
    main()
