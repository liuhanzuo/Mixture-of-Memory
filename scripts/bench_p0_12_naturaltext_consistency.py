#!/usr/bin/env python
"""P0.12 NATURAL-TEXT consistency check — companion to the acceptance bench.

Motivation
----------
``scripts/bench_p0_12_acceptance.py`` measured the j=0 vs j=12 output consistency
of the SAME single-variable isolation (flagship LoRA held fixed, identical top-12
retrieved pack, only the read resume-start layer varies), but it did so on a
SYNTHETIC random-token pack — the pack was drawn with ``torch.randint`` so its
``packed_ids_sha256`` could be forced byte-identical to the existing 16k
authoritative set (``f7fc7617…``). Before the P0.12 wording can be upgraded from
"read-path 对照" toward "near-same-quality", the user/main session requires ONE
more check: does the near-equivalence still hold on *natural text* (a different,
new pack sha) rather than only on random ids?

This script answers exactly that and NOTHING else:

  * pack source = a REAL document tokenised from an on-disk natural-text corpus
    (default ``data/rmt_train_wikitext.jsonl``), split into ``chunk_size`` chunks,
    bm25-selecting the top-``topk`` context chunks — the SAME retrieval/packing
    machinery as the acceptance bench, only the token source changes.
  * both arms (j=0 and j=12) consume the IDENTICAL natural-text pack (same
    ``sel_idx``, same token sequence). The per-arm reconstructed
    ``packed_ids_sha256`` is asserted equal across arms (a NEW sha — it does NOT
    need to equal the synthetic ``f7fc7617…``; it only needs to be consistent
    between the two arms). Mismatch => hard abort.
  * model / LoRA / dtype / attn / chat_template=False / single-card serial are
    IDENTICAL to the acceptance bench: it imports ``_load`` +
    ``_backbone_provenance`` + ``_lora_modules`` + ``_versions`` + ``_kl`` +
    ``_cosine_rows`` VERBATIM from ``bench_p0_12_acceptance.py`` so there is no
    config drift, and the bm25 selector + length parser from
    ``bench_qcmem_vs_fullctx.py`` + the timing/hash helpers from
    ``bench_p0_12_depth_replay.py``. NOTHING in those files is modified.
  * only OUTPUT-CONSISTENCY is measured (timing / peak-mem already live in the
    acceptance bench): per natural-text document,
      - last-position (next-token): logit cosine, top-1 match, KL(0||12);
      - query-tail teacher-forced (last ``T_q`` positions): cos_mean/min, top-1
        agreement rate, KL(0||12) + KL(12||0);
      - greedy decode ``n_decode`` steps: per-step top-1 agreement, cos_mean,
        kl_mean (both arms advanced on the SAME arm-0 argmax token).
    ``--n_docs`` different documents are run (default 3) to de-risk single-doc
    luck; per-doc JSONs + an aggregated summary JSON are written.

Run (single card, serial):
    CUDA_VISIBLE_DEVICES=0 python scripts/bench_p0_12_naturaltext_consistency.py \
        --model_path models/Qwen3-8b-local \
        --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
        --corpus data/rmt_train_wikitext.jsonl \
        --output_dir bench_results/p0_12_naturaltext
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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

from src.memory.qcmem import QCMemModel  # noqa: E402
# Reuse the EXACT bm25 selector + length parser so retrieval/packing is
# bit-identical to bench_qcmem_vs_fullctx.py AND bench_p0_12_acceptance.py.
from bench_qcmem_vs_fullctx import _bm25_scores, parse_length  # noqa: E402
# Reuse the model/LoRA loader + provenance + consistency helpers VERBATIM from
# the acceptance bench so config (model, LoRA, dtype, attn) cannot drift.
from bench_p0_12_acceptance import (  # noqa: E402
    _load, _backbone_provenance, _lora_modules, _versions, _kl, _cosine_rows,
)


# --------------------------------------------------------------------------- #
# natural-text corpus -> a real token sequence of exactly ``target_length``
# --------------------------------------------------------------------------- #
def _index_corpus(corpus_path):
    """Return an ordered list of ``(line_index, byte_offset, text)`` for every
    non-empty JSON line carrying a natural-text field (``text`` or ``input_text``).

    ``byte_offset`` is the file byte position of that line's first byte (for
    provenance). The corpus is small (tens of MB / ~thousands of lines) so a
    single pass building an in-memory index is cheap."""
    entries = []
    with open(corpus_path, "rb") as f:
        idx = 0
        while True:
            byte_offset = f.tell()
            raw = f.readline()
            if not raw:
                break
            this_idx = idx
            idx += 1
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text") or obj.get("input_text") or ""
            if not isinstance(text, str) or not text:
                continue
            entries.append((this_idx, byte_offset, text))
    if not entries:
        raise SystemExit(f"no natural-text lines found in corpus {corpus_path}")
    return entries


def _natural_tokens_from(entries, start_pos, target_length, tokenizer):
    """Accumulate token ids from ``entries[start_pos:]`` until >= ``target_length``
    real tokens are collected, then truncate to EXACTLY ``target_length``.

    Returns ``(tokens[list[int]], provenance dict)``. Tokenisation uses
    ``add_special_tokens=False`` so the sequence is pure document text (BOS is
    added by the packer as the sink, exactly like the acceptance bench)."""
    tokens: list[int] = []
    src_lines = []
    start_line = None
    start_byte = None
    char_count = 0
    pos = start_pos
    n = len(entries)
    while len(tokens) < target_length and pos < n:
        line_idx, byte_off, text = entries[pos]
        if start_line is None:
            start_line = line_idx
            start_byte = byte_off
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        tokens.extend(int(t) for t in ids)
        src_lines.append(line_idx)
        char_count += len(text)
        pos += 1
    if len(tokens) < target_length:
        raise SystemExit(
            f"corpus exhausted at line {pos}: collected {len(tokens)} tokens "
            f"< target {target_length} (need a longer corpus or smaller length)")
    tokens = tokens[:target_length]
    prov = {
        "corpus_path": None,           # filled by caller
        "start_line_index": start_line,
        "start_byte_offset": start_byte,
        "source_line_indices": src_lines,
        "n_source_lines": len(src_lines),
        "source_char_count": char_count,
        "target_length_tokens": target_length,
        "next_line_pos": pos,          # where the NEXT doc should start scanning
    }
    return tokens, prov


def _build_natural_pack(tokens, tokenizer, args):
    """Reproduce the acceptance-bench packing on a NATURAL-TEXT token sequence.

    Mirrors ``bench_p0_12_acceptance._build_pack`` exactly (same chunking, same
    bm25 top-k selection, same ``[bos ; selected ctx ; query]`` layout, same
    sha256 recipe) — the ONLY change is ``tokens`` come from real text instead of
    ``torch.randint``. Returns the pack dict (a NEW sha, not the synthetic one)."""
    L = len(tokens)
    tok = torch.tensor(tokens, dtype=torch.long)
    chunks = list(tok.split(args.chunk_size))
    if len(chunks) < args.topk + 1:
        raise SystemExit(
            f"length {L} gives {len(chunks)} chunks; need >= topk+1={args.topk + 1}")
    context_chunks = chunks[:-1]
    query_chunk = chunks[-1]
    n_ctx = len(context_chunks)
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = int(tok[0].item())
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
        "L": L,
        "context_chunks": context_chunks,
        "query_tok_list": query_tok_list,
        "n_ctx": n_ctx,
        "bos_id": int(bos_id),
        "sel_idx": sel_idx,
        "packed_ids": packed_ids,
        "packed_ids_sha256": packed_ids_sha256,
        "pack_token_count": len(packed_ids),
    }


def _arm_pack_sha(pack, arm_label):
    """Recompute the packed-id sha256 exactly as an arm feeds it into ``read``
    (BOS sink + selected ctx chunks in doc order + query), independently, so we
    can assert BOTH arms consume byte-identical tokens. Because both arms use the
    SAME ``pack`` object (same ``sel_idx`` + same ``context_chunks`` + same
    query), this recomputation is deterministic and must agree; if it ever does
    not, the caller hard-aborts."""
    ids = [int(pack["bos_id"])]
    for i in pack["sel_idx"]:
        ids.extend(pack["context_chunks"][i].tolist())
    ids.extend(pack["query_tok_list"])
    sha = hashlib.sha256(b",".join(str(t).encode() for t in ids)).hexdigest()
    return sha, len(ids)


# --------------------------------------------------------------------------- #
# per-document consistency (7) — j=0 vs j=12 on the IDENTICAL natural pack
# --------------------------------------------------------------------------- #
def _consistency_for_pack(qc0, qc12, pack, n_decode):
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

    # ---- query-tail (teacher-forced) over the last T_q positions ----
    tail0 = logits0[0, -T_q:, :].float()
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
    print(f"[nat][7] query-tail (T_q={T_q}): cos_mean={cos_mean:.4f} "
          f"cos_min={cos_min:.4f} top1_agree={top1_agree:.4f} "
          f"KL(0||12)={kl_0_12:.4f} KL(12||0)={kl_12_0:.4f}", flush=True)
    print(f"[nat][7] last-pos: cos={lp_cos:.4f} top1_match={lp_match} "
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
        for step in range(n_decode):
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
    nd = max(1, n_decode)
    decode_summary = {
        "n_decode": n_decode,
        "top1_agreement_rate": matches / nd,
        "cosine_mean": cos_acc / nd,
        "kl_mean_0to12": kl_acc / nd,
        "per_step": per_step,
    }
    print(f"[nat][7] decode ({n_decode} steps): "
          f"top1_agree={decode_summary['top1_agreement_rate']:.4f} "
          f"cos_mean={decode_summary['cosine_mean']:.4f} "
          f"kl_mean={decode_summary['kl_mean_0to12']:.4f}", flush=True)

    del sink0, sel0, sink12, sel12
    torch.cuda.empty_cache()
    return {
        "query_tail": query_tail_summary,
        "last_position": last_pos_summary,
        "decode": decode_summary,
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser(
        description="P0.12 natural-text output-consistency check (j=0 vs j=12)")
    ap.add_argument("--model_path", type=str, default="models/Qwen3-8b-local")
    ap.add_argument("--lora_adapter", type=str,
                    default="outputs/qcmem_distill_qwen_j12_r32_4k/final")
    ap.add_argument("--corpus", type=str, default="data/rmt_train_wikitext.jsonl",
                    help="on-disk natural-text jsonl (text/input_text field).")
    ap.add_argument("--n_docs", type=int, default=3,
                    help="number of distinct natural-text documents to test.")
    ap.add_argument("--start_lines", type=str, default="",
                    help="comma-separated corpus line indices to start each doc "
                         "at; default auto-spaces n_docs docs across the corpus.")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--length", type=str, default="16k",
                    help="target natural-text token length per doc (>= topk+1 "
                         "chunks). Matches the acceptance bench geometry at 16k.")
    ap.add_argument("--n_decode", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_dir", type=str,
                    default="bench_results/p0_12_naturaltext")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    target_length = parse_length(args.length)

    print("=" * 78)
    print(f"P0.12 NATURAL-TEXT consistency :: model={args.model_path}")
    print(f"  lora={args.lora_adapter or None} corpus={args.corpus}")
    print(f"  n_docs={args.n_docs} length={args.length} chunk_size={args.chunk_size} "
          f"topk={args.topk} dtype={dtype} attn={args.attn_impl} device={device}")
    print("=" * 78, flush=True)

    # ---- load model + LoRA ONCE (identical loader to the acceptance bench) ----
    tokenizer, model, lora_sha256, lora_layers = _load(
        args.model_path, dtype, args.attn_impl, device, args.lora_adapter)
    L_layers = int(model.config.num_hidden_layers)
    vocab = int(model.config.vocab_size)
    qc0 = QCMemModel(model, resume_j=0)
    qc12 = QCMemModel(model, resume_j=12)
    prov_backbone = _backbone_provenance(qc0, args.model_path)
    prov_lora = _lora_modules(model)
    prov_versions = _versions(device)
    print(f"[nat] backbone: {L_layers} layers hidden={qc0.hidden_size} vocab={vocab}; "
          f"LoRA modules mounted={prov_lora['count']} "
          f"adapters={prov_lora['adapter_names']}", flush=True)

    # ---- build the corpus index + choose per-doc start lines ----
    entries = _index_corpus(args.corpus)
    print(f"[nat] corpus indexed: {len(entries)} natural-text lines", flush=True)
    if args.start_lines.strip():
        start_positions = [int(x) for x in args.start_lines.split(",") if x.strip()]
    else:
        # auto-space n_docs docs across the corpus (positions into `entries`).
        step = max(1, len(entries) // max(1, args.n_docs))
        start_positions = [min(i * step, len(entries) - 1)
                           for i in range(args.n_docs)]
    print(f"[nat] doc start positions (index into corpus lines): {start_positions}",
          flush=True)

    out_dir = Path(PROJECT_ROOT) / args.output_dir \
        if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_doc_results = []
    shas = []
    doc_files = []
    t_start = time.perf_counter()
    for di, sp in enumerate(start_positions):
        print(f"\n[nat] ===== doc {di} (corpus start pos {sp}) =====", flush=True)
        tokens, doc_prov = _natural_tokens_from(entries, sp, target_length, tokenizer)
        doc_prov["corpus_path"] = args.corpus
        pack = _build_natural_pack(tokens, tokenizer, args)

        # ---- assert BOTH arms consume byte-identical natural-text tokens ----
        sha_a, len_a = _arm_pack_sha(pack, "A(j=0)")
        sha_b, len_b = _arm_pack_sha(pack, "B(j=12)")
        print(f"[nat] packed_ids_sha256 (NEW, natural text) = {pack['packed_ids_sha256']}")
        print(f"[nat] arm-A sha={sha_a}  arm-B sha={sha_b}  "
              f"pack_tok=A:{len_a}/B:{len_b}", flush=True)
        if not (sha_a == sha_b == pack["packed_ids_sha256"]):
            print("[nat] FATAL: the two arms do NOT consume identical packed "
                  "token ids -> hard abort (P0.12 requires an identical pack "
                  "across arms).", flush=True)
            sys.exit(3)

        cons = _consistency_for_pack(qc0, qc12, pack, args.n_decode)

        doc_result = {
            "mode": "naturaltext_consistency",
            "doc_index": di,
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
                "source": "natural_text",
                "corpus_path": args.corpus,
                "doc_provenance": doc_prov,
                "pack_token_count": pack["pack_token_count"],
                "packed_ids_sha256": pack["packed_ids_sha256"],
                "arm_A_packed_ids_sha256": sha_a,
                "arm_B_packed_ids_sha256": sha_b,
                "arms_pack_sha_consistent": bool(sha_a == sha_b),
                "note_new_sha": ("NEW natural-text pack sha; NOT expected to "
                                 "equal the synthetic acceptance sha "
                                 "f7fc76177dd60a664f3b37a3934c1efb83c5a9d6bcaf"
                                 "3804697c9f684aafdc13 — only required to be "
                                 "consistent between the two arms."),
                "selected_idx": pack["sel_idx"],
                "n_ctx_chunks": pack["n_ctx"],
                "query_len": len(pack["query_tok_list"]),
            },
            "provenance": {
                "backbone": prov_backbone,
                "lora": prov_lora,
                "versions": prov_versions,
            },
            "consistency": {
                "note": ("j=0 feeds global-causal h_0 into layer 12; j=12 feeds "
                         "chunk-local h_12. On this NATURAL-TEXT pack this measures "
                         "whether the DIFFERENT layer-12 input changes the final "
                         "output."),
                "query_tail": cons["query_tail"],
                "last_position": cons["last_position"],
                "decode": cons["decode"],
            },
            "env": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": prov_versions.get("gpu"),
                "python": platform.python_version(),
            },
        }
        fname = out_dir / f"doc{di}_startline{doc_prov['start_line_index']}.json"
        with open(fname, "w") as f:
            json.dump(doc_result, f, indent=2)
        print(f"[nat] wrote per-doc JSON -> {fname}", flush=True)
        per_doc_results.append(doc_result)
        shas.append(pack["packed_ids_sha256"])
        doc_files.append(str(fname))

    # ---- aggregate summary across docs ----
    agg = {
        "last_position_cosine_mean": _mean(
            [d["consistency"]["last_position"]["cosine"] for d in per_doc_results]),
        "last_position_top1_match_rate": _mean(
            [1.0 if d["consistency"]["last_position"]["top1_match"] else 0.0
             for d in per_doc_results]),
        "last_position_kl_0to12_mean": _mean(
            [d["consistency"]["last_position"]["kl_0to12"] for d in per_doc_results]),
        "query_tail_cosine_mean": _mean(
            [d["consistency"]["query_tail"]["cosine_mean"] for d in per_doc_results]),
        "query_tail_cosine_min": min(
            [d["consistency"]["query_tail"]["cosine_min"] for d in per_doc_results]),
        "query_tail_top1_agreement_mean": _mean(
            [d["consistency"]["query_tail"]["top1_agreement_rate"]
             for d in per_doc_results]),
        "query_tail_kl_0to12_mean": _mean(
            [d["consistency"]["query_tail"]["kl_mean_0to12"]
             for d in per_doc_results]),
        "query_tail_kl_12to0_mean": _mean(
            [d["consistency"]["query_tail"]["kl_mean_12to0"]
             for d in per_doc_results]),
        "decode_top1_agreement_mean": _mean(
            [d["consistency"]["decode"]["top1_agreement_rate"]
             for d in per_doc_results]),
        "decode_cosine_mean": _mean(
            [d["consistency"]["decode"]["cosine_mean"] for d in per_doc_results]),
        "decode_kl_0to12_mean": _mean(
            [d["consistency"]["decode"]["kl_mean_0to12"] for d in per_doc_results]),
    }
    summary = {
        "mode": "naturaltext_consistency_summary",
        "n_docs": len(per_doc_results),
        "elapsed_s": time.perf_counter() - t_start,
        "config": {
            "model_path": args.model_path,
            "lora_adapter": args.lora_adapter or None,
            "lora_sha256": lora_sha256,
            "corpus_path": args.corpus,
            "chunk_size": args.chunk_size,
            "topk": args.topk,
            "length": args.length,
            "n_decode": args.n_decode,
            "dtype": args.dtype,
            "attn_impl": args.attn_impl,
            "seed": args.seed,
            "num_layers": L_layers,
            "vocab_size": vocab,
        },
        "per_doc_packed_ids_sha256": shas,
        "all_arm_shas_consistent": all(
            d["pack"]["arms_pack_sha_consistent"] for d in per_doc_results),
        "aggregate_consistency": agg,
        "per_doc": [
            {
                "doc_index": d["doc_index"],
                "start_line_index": d["pack"]["doc_provenance"]["start_line_index"],
                "start_byte_offset": d["pack"]["doc_provenance"]["start_byte_offset"],
                "packed_ids_sha256": d["pack"]["packed_ids_sha256"],
                "last_position": d["consistency"]["last_position"],
                "query_tail": d["consistency"]["query_tail"],
                "decode": {k: v for k, v in d["consistency"]["decode"].items()
                           if k != "per_step"},
            }
            for d in per_doc_results
        ],
        "provenance": {
            "backbone": prov_backbone,
            "lora": prov_lora,
            "versions": prov_versions,
        },
        "reference_synthetic_acceptance": {
            "source": "bench_results/p0_12_acceptance/consistency.json",
            "synthetic_pack_sha256": (
                "f7fc76177dd60a664f3b37a3934c1efb83c5a9d6bcaf3804697c9f684aafdc13"),
            "note": ("acceptance bench numbers were measured on a SYNTHETIC "
                     "random-token pack; this natural-text run uses NEW shas and "
                     "is the requested different-sha check."),
        },
    }
    summary_file = out_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[nat] wrote summary JSON -> {summary_file}", flush=True)
    print("[nat] AGGREGATE across %d docs:" % len(per_doc_results))
    print(f"  last-pos cos_mean={agg['last_position_cosine_mean']:.4f} "
          f"top1_match_rate={agg['last_position_top1_match_rate']:.4f} "
          f"KL={agg['last_position_kl_0to12_mean']:.4f}")
    print(f"  query-tail cos_mean={agg['query_tail_cosine_mean']:.4f} "
          f"top1_agree={agg['query_tail_top1_agreement_mean']:.4f} "
          f"KL(0||12)={agg['query_tail_kl_0to12_mean']:.4f}")
    print(f"  decode top1_agree={agg['decode_top1_agreement_mean']:.4f} "
          f"cos_mean={agg['decode_cosine_mean']:.4f} "
          f"KL={agg['decode_kl_0to12_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
