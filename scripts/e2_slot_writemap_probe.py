"""E2-slot — write-map needle-localization probe for the SLOT (routing) model.

ZERO-TRAINING, pure-eval probe. NO checkpoint mutation, NO change to the main
eval/train path. Standalone script: it imports the loaders/locators it reuses
from ``run_babilong_mem_space.py`` and never invokes that module's ``main()``.

================================================================================
★★★  RED-LINE CKPT  ★★★
This probe is ONLY valid on a CLEAN, slot (non-FIFO) checkpoint with
``babilong_mix_fraction == 0.0`` (no BABILong leakage). The intended ckpt is:

    outputs/distill_pg19_chunk512_nctx63/mem_space_adapter.pt
        num_slots=128, top_k=16, num_global_slots=4, slot_evict_mode="off",
        routing_pool_mode=max_pool (default), use_fifo_memory=False,
        babilong_mix_fraction=0.0  → clean slot SOTA.

DO NOT point this at mem_space_p11_* / *l3recontoken* / b50/b100/c1024 etc.
(those are leaked / babilong_mix>0 or FIFO ckpts). The script aborts loudly if
``use_fifo_memory`` is True or ``babilong_mix_fraction != 0`` unless you pass
``--allow_dirty`` (you almost never should).
================================================================================

What this probe measures
------------------------
In a SLOT model the reader cannot attend to a past chunk directly: query→chunk
relevance must be mediated by the slot bank. A context chunk's content is routed
(top-k) into a handful of slots; at question time the question chunk routes into
some slots too. If routing is content-addressable, the needle chunk should write
to the SAME slots the question later routes to. We test this with a zero-training
write-map probe:

  1. Stream context chunks 0..n_chunks-2 one at a time. After each chunk's full
     forward, read the chosen selection layer's ``selector.last_idx`` ([1, top_k],
     the HARD top-k routed slot indices for THAT chunk) and record a binary
     write-map row  W[c, n] = 1 iff chunk c routed (top-k) to slot n.
  2. Freeze the bank (mirrors eval generation) and forward the LAST (question)
     chunk. Read that layer's ``selector._last_routing_q`` ([1, T, S]) and
     ``_last_routing_k`` ([1, N, S]) and recompute the question's slot routing
     distribution  s_q[n] = softmax_n( max_t( q_q[t]·k[n] ) * temperature )
     — byte-identical to the selector's own max_pool path (selector.py:451-462).
  3. chunk_score[c] = Σ_n W[c, n] · s_q[n]  (sum of question routing mass over
     the slots chunk c wrote to). Rank every candidate chunk by chunk_score; the
     needle chunk's competition rank + recall@{1,2,4,8,16} are reported.

★ GLOBAL SLOTS are EXCLUDED. The last ``num_global_slots`` slots
  (idx >= num_slots - num_global_slots) are written on EVERY chunk unconditionally
  (layer.py:1995-2003) → they carry no localization signal and would add a constant
  to every chunk_score. They are dropped from BOTH W and s_q.

★ NO EVICTION. ``slot_evict_mode="off"`` on this ckpt → every streamed context
  chunk is a candidate. n_candidates = n_chunks - 1 (the last/question chunk is
  never a write target). This DIFFERS from the reader-attn E2, whose candidate
  pool is the capped FIFO buffer (eval default 25).

================================================================================
★★★  CROSS-MODEL CAVEAT — READ BEFORE COMPARING TO READER-ATTN E2  ★★★
This probe runs on the SLOT ckpt (distill_pg19_chunk512_nctx63). The reader-attn
E2 (scripts/e2_needle_recall_probe.py) runs on a DIFFERENT, FIFO ckpt. The two are
DIFFERENT MODELS trained differently:
  * The recall@K numbers are NOT directly subtractable across the two probes.
  * The candidate pools differ: slot probe = ALL context chunks (no eviction);
    reader-attn probe = capped FIFO buffer (~25). A bigger candidate pool makes a
    fixed recall@K harder, so we additionally report:
        - rank percentile  = needle_rank / n_candidates   (scale-free, 0=top)
        - chance recall@K   = K / n_candidates             (random-guess baseline)
        - recall-vs-chance  = recall@K / chance@K          (>1 = better than random)
  Compare LIFT-OVER-CHANCE and rank-percentile across models, not raw recall@K.
================================================================================
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from statistics import median

# --------------------------------------------------------------------------- #
# Environment: offline HF + local arrow cache. MUST be set before importing the
# run module (which imports `datasets` at module top).
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_BABILONG_PKG = os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")
if os.path.isdir(_BABILONG_PKG) and _BABILONG_PKG not in sys.path:
    sys.path.insert(0, _BABILONG_PKG)

import json  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from babilong.prompts import (  # noqa: E402
    DEFAULT_PROMPTS,
    DEFAULT_TEMPLATE,
    get_formatted_input,
)

from src.memory.mem_space import MemorySpaceConfig, _reset_fifo_memory  # noqa: E402

# Reuse — do NOT re-implement — the loaders/locators from the main eval script.
from scripts.run_babilong_mem_space import (  # noqa: E402
    build_mem_space_config,
    load_babilong_dataset,
    load_mem_space_model,
    _reset_banks,
    _reset_l2,
    _freeze_banks,
    _unfreeze_banks,
    _locate_needle_chunks,
)

RECALL_KS = (1, 2, 4, 8, 16)


# --------------------------------------------------------------------------- #
# Slot-routing access helpers
# --------------------------------------------------------------------------- #


def _get_select_layer(model, select_layer):
    """Return the MemorySpaceLayer wrapper at index ``select_layer`` (or None)."""
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return None
    L = int(select_layer)
    if L < 0 or L >= len(mem_layers):
        return None
    return mem_layers[L]


def _writemap_row(sel_layer, num_slots, num_global):
    """Binary [num_slots] write-map row from ``selector.last_idx`` of THIS chunk.

    ``selector.last_idx`` is the HARD top-k routed indices ([B, top_k]) recorded
    at selector.py:540 — set BEFORE the global slots are appended in layer.py
    (the global append at layer.py:1995-2003 mutates a LOCAL ``idx``, not
    ``selector.last_idx``). So last_idx already excludes the always-on globals,
    but we additionally mask out any routed index that falls in the global range
    [num_slots - num_global, num_slots) defensively.

    Returns a float CPU tensor [num_slots] (1.0 on routed non-global slots) or
    None if the selector did not run.
    """
    sel = getattr(sel_layer, "selector", None)
    if sel is None:
        return None
    idx = getattr(sel, "last_idx", None)
    if idx is None:
        return None
    idx = idx.detach().reshape(-1).cpu()  # [top_k] (B=1)
    row = torch.zeros(num_slots, dtype=torch.float32)
    glob_lo = num_slots - num_global
    for n in idx.tolist():
        n = int(n)
        if 0 <= n < glob_lo:  # drop global slots (idx >= glob_lo)
            row[n] = 1.0
    return row


@torch.no_grad()
def _compute_s_q(sel_layer, num_slots, num_global):
    """Recompute the question chunk's slot routing distribution s_q[num_slots].

    Reproduces the selector's max_pool path EXACTLY (selector.py:451-462):
        per_token_logits = einsum("ts,ns->tn", q, k) * temperature
        logits           = max over t
        s_q              = softmax(logits)   over N slots
    using the detached, already-F.normalize'd ``_last_routing_q`` / ``_last_routing_k``
    stashed by the selector during the question chunk forward.

    Global slots (idx >= num_slots - num_global) are zeroed AFTER the softmax so
    they contribute nothing to chunk_score. Ranking is invariant to whether s_q is
    renormalized after zeroing (a positive global scale), so we leave it un-renorm.

    Returns a float CPU tensor [num_slots] or None on failure.
    """
    sel = getattr(sel_layer, "selector", None)
    if sel is None:
        return None
    q = getattr(sel, "_last_routing_q", None)   # [B, T, S]
    k = getattr(sel, "_last_routing_k", None)   # [B, N, S]
    if q is None or k is None:
        return None
    if q.dim() != 3 or k.dim() != 3 or q.shape[0] != 1 or k.shape[0] != 1:
        return None
    temperature = float(getattr(sel, "temperature", 1.0))
    qf = q[0].float()                            # [T, S]
    kf = k[0].float()                            # [N, S]
    if kf.shape[0] != num_slots:
        return None
    per_token_logits = torch.einsum("ts,ns->tn", qf, kf) * temperature  # [T, N]
    logits = per_token_logits.max(dim=0).values  # [N]
    s_q = F.softmax(logits, dim=-1)              # [N]
    glob_lo = num_slots - num_global
    if num_global > 0:
        s_q = s_q.clone()
        s_q[glob_lo:] = 0.0                       # drop global slots
    return s_q.detach().cpu()


# --------------------------------------------------------------------------- #
# Per-sample probe
# --------------------------------------------------------------------------- #


@torch.no_grad()
def probe_sample(model, input_ids, target, tokenizer, chunk_size, device,
                 select_layer, num_slots, num_global):
    """Run the slot write-map probe on one encoded sample. Returns a CSV-row dict."""
    _reset_banks(model)
    _reset_l2(model)
    _reset_fifo_memory(model)  # harmless on a slot ckpt; keeps state hygiene

    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    n_chunks = len(chunks)
    ingested = n_chunks - 1  # context chunks streamed; last chunk = question

    needle = _locate_needle_chunks(input_ids, target, tokenizer, chunk_size)

    sel_layer = _get_select_layer(model, select_layer)

    row = {
        "task": None, "length": None, "idx": None,
        "n_chunks": n_chunks,
        "needle_chunks": "" if needle is None else ";".join(str(c) for c in sorted(needle)),
        "needle_located": needle is not None,
        "n_candidates": max(ingested, 0),
        "needle_in_candidates": False,
        "needle_rank": "",
        "rank_pctile": "",
        "wmap_density": "",
    }
    for k in RECALL_KS:
        row[f"recall@{k}"] = ""

    if sel_layer is None or ingested <= 0:
        # No selector / single-chunk doc → nothing to rank.
        return row

    # ---- stream context chunks one at a time, recording the write-map ----
    W = torch.zeros(ingested, num_slots, dtype=torch.float32)
    for c in range(ingested):
        _ = model(input_ids=chunks[c].unsqueeze(0).to(device), use_cache=False)
        r = _writemap_row(sel_layer, num_slots, num_global)
        if r is not None:
            W[c] = r
    row["wmap_density"] = round(float(W.sum(dim=1).mean().item()), 3)

    # ---- question chunk: freeze (mirror eval) then forward, read s_q ----
    _freeze_banks(model)
    try:
        _ = model(input_ids=chunks[-1].unsqueeze(0).to(device), use_cache=False)
        s_q = _compute_s_q(sel_layer, num_slots, num_global)
    finally:
        _unfreeze_banks(model)

    if s_q is None:
        return row

    # chunk_score[c] = Σ_n W[c, n] · s_q[n]
    chunk_score = (W @ s_q)  # [ingested]

    if needle is None:
        return row  # answer not locatable → measurement gap, rank/recall blank

    needle_cand = sorted(c for c in needle if 0 <= c < ingested)
    if not needle_cand:
        # Needle only in the (un-streamed) question chunk → never a write target.
        row["needle_in_candidates"] = False
        for k in RECALL_KS:
            row[f"recall@{k}"] = 0
        return row

    row["needle_in_candidates"] = True
    # Competition rank: # chunks scoring STRICTLY higher than the needle + 1.
    # Best (lowest) rank over all needle candidate chunks (multi-mention).
    best_rank = min(int((chunk_score > chunk_score[c]).sum().item()) + 1 for c in needle_cand)
    row["needle_rank"] = best_rank
    row["rank_pctile"] = round(best_rank / float(ingested), 4)
    for k in RECALL_KS:
        row[f"recall@{k}"] = 1 if best_rank <= k else 0
    return row


# --------------------------------------------------------------------------- #
# Encoding (mirrors run_babilong_mem_space._encode_sample / e2 probe)
# --------------------------------------------------------------------------- #


def _build_prompt_cfg(task):
    return {
        "instruction": DEFAULT_PROMPTS[task]["instruction"],
        "examples": DEFAULT_PROMPTS[task]["examples"],
        "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"],
        "template": DEFAULT_TEMPLATE,
    }


def _encode_sample(sample, prompt_cfg, tokenizer):
    input_text = get_formatted_input(
        sample["input"],
        sample["question"],
        prompt_cfg["examples"],
        prompt_cfg["instruction"],
        prompt_cfg["post_prompt"],
        template=prompt_cfg["template"],
    )
    ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
    if isinstance(ids, list):
        ids = torch.tensor([ids], dtype=torch.long)
    return sample["target"], ids


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description="E2-slot write-map needle-localization probe (slot ckpt)")
    ap.add_argument("--model_path", default=os.path.join(PROJECT_ROOT, "models", "Meta-Llama-3-8B"),
                    help="Base Llama-3-8B dir")
    ap.add_argument("--checkpoint", required=True,
                    help="CLEAN slot adapter .pt (e.g. distill_pg19_chunk512_nctx63/mem_space_adapter.pt)")
    ap.add_argument("--adapter_config", required=True, help="adapter_config.json (slot ckpt)")
    ap.add_argument("--dataset_name", default="RMT-team/babilong")
    ap.add_argument("--tasks", nargs="+", default=["qa1", "qa5"])
    ap.add_argument("--lengths", nargs="+", default=["8k", "16k", "32k"])
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=100, help="max samples per task/length (-1 = all)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--select_layer", type=int, default=16,
                    help="which patched decoder layer's selector to read routing from "
                         "(default 16 to mirror the reader-attn E2 select layer; the slot "
                         "model has a per-layer selector, this picks ONE).")
    ap.add_argument("--allow_dirty", action="store_true",
                    help="override the clean-ckpt guard (FIFO / babilong_mix>0). Do NOT use "
                         "for the headline result — it breaks the no-leak red-line.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--output_csv", default=None,
                    help="output CSV path (default: e2slot_writemap_<tag>.csv in cwd)")
    args = ap.parse_args()

    if args.num_shards < 1:
        ap.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        ap.error(f"--shard_index must be in [0, {args.num_shards})")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print("[E2-slot] Configuration:")
    print(f"  model_path     {args.model_path}")
    print(f"  checkpoint     {args.checkpoint}")
    print(f"  adapter_config {args.adapter_config}")
    print(f"  tasks          {args.tasks}")
    print(f"  lengths        {args.lengths}")
    print(f"  chunk_size     {args.chunk_size}")
    print(f"  select_layer   {args.select_layer}")
    print(f"  limit          {args.limit}")
    if args.num_shards > 1:
        print(f"  shard          {args.shard_index}/{args.num_shards}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)
    mem_config: MemorySpaceConfig = build_mem_space_config(adapter_cfg)
    mem_config.l3_recon_max_positions = args.chunk_size  # mirror training (run script:1362)

    # ---- ★ CLEAN-CKPT RED-LINE GUARD ★ ----
    is_fifo = bool(getattr(mem_config, "use_fifo_memory", False))
    mix = float(adapter_cfg.get("babilong_mix_fraction",
                                getattr(mem_config, "babilong_mix_fraction", 0.0) or 0.0))
    num_slots = int(getattr(mem_config, "num_slots", 0))
    num_global = int(getattr(mem_config, "num_global_slots", 0) or 0)
    top_k = int(getattr(mem_config, "top_k", 0))
    evict = getattr(mem_config, "slot_evict_mode", "off")
    routing_mode = getattr(mem_config, "routing_pool_mode", "max_pool")
    sel_temp = float(getattr(mem_config, "selector_temperature", 1.0))
    print(f"[E2-slot] use_fifo_memory={is_fifo} babilong_mix_fraction={mix} "
          f"num_slots={num_slots} top_k={top_k} num_global_slots={num_global} "
          f"slot_evict_mode={evict} routing_pool_mode={routing_mode} "
          f"selector_temperature={sel_temp}")
    _bad = []
    if is_fifo:
        _bad.append("use_fifo_memory=True (this is a FIFO ckpt, NOT a slot model)")
    if mix != 0.0:
        _bad.append(f"babilong_mix_fraction={mix} != 0 (LEAKED ckpt)")
    if _bad:
        msg = ("[E2-slot] ★ CLEAN-CKPT GUARD TRIPPED ★ — " + "; ".join(_bad) +
               ". This probe is only valid on the clean slot ckpt "
               "(distill_pg19_chunk512_nctx63, mix=0).")
        if not args.allow_dirty:
            raise SystemExit(msg + " Refusing to run. Pass --allow_dirty ONLY for debugging.")
        print(msg + " --allow_dirty set: continuing anyway (results are NOT publishable).")
    if num_slots <= 0:
        raise SystemExit("[E2-slot] num_slots not set in config — cannot build write-map.")
    if evict not in (None, "off"):
        print(f"[E2-slot] WARNING: slot_evict_mode={evict} != 'off' — this probe assumes NO "
              f"eviction (all context chunks are candidates). Eviction is NOT modelled here.")
    if routing_mode != "max_pool":
        print(f"[E2-slot] WARNING: routing_pool_mode={routing_mode} != 'max_pool'. s_q is "
              f"recomputed with the max_pool formula; it will NOT match a non-max_pool selector.")

    model = load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        mem_config=mem_config,
        device=device,
        dtype=dtype,
        attn_impl=args.attn_impl,
    )

    # Confirm we are on the slot forward path (selector exists, not FIFO).
    sel_layer = _get_select_layer(model, args.select_layer)
    if sel_layer is None or getattr(sel_layer, "selector", None) is None:
        raise SystemExit(f"[E2-slot] No selector on layer {args.select_layer} — not a slot model?")
    if getattr(sel_layer, "_use_fifo_memory", False):
        raise SystemExit("[E2-slot] Layer is on the FIFO forward path (_use_fifo_memory=True). Abort.")
    print(f"[E2-slot] Confirmed slot path: layer {args.select_layer} has a TopKSelector, "
          f"_use_fifo_memory={getattr(sel_layer, '_use_fifo_memory', False)}.")

    out_path = args.output_csv
    if out_path is None:
        tag = f"L{args.select_layer}_cs{args.chunk_size}"
        if args.num_shards > 1:
            tag += f"_shard{args.shard_index}of{args.num_shards}"
        out_path = os.path.join(os.getcwd(), f"e2slot_writemap_{tag}.csv")
    out_path = os.path.abspath(out_path)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    fieldnames = (
        ["idx", "task", "length", "n_chunks", "needle_chunks", "needle_located",
         "n_candidates", "needle_in_candidates", "needle_rank", "rank_pctile", "wmap_density"]
        + [f"recall@{k}" for k in RECALL_KS]
    )
    all_rows = []
    fcsv = open(out_path, "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    writer.writeheader()

    for task in args.tasks:
        if task not in DEFAULT_PROMPTS:
            print(f"[E2-slot] WARNING task {task} not in DEFAULT_PROMPTS; skipping")
            continue
        prompt_cfg = _build_prompt_cfg(task)
        for length in args.lengths:
            try:
                data = load_babilong_dataset(args.dataset_name, length)
                task_data = data[task]
            except Exception as e:
                print(f"[E2-slot] ERROR loading {args.dataset_name}/{length}/{task}: {e}")
                continue
            n = len(task_data)
            if args.limit > 0:
                n = min(n, args.limit)
            sample_indices = list(range(n))[args.shard_index::args.num_shards]
            print(f"\n[E2-slot] task={task} length={length}: {len(sample_indices)} samples")

            for idx in tqdm(sample_indices, desc=f"{task}/{length}", leave=False):
                target, input_ids = _encode_sample(task_data[idx], prompt_cfg, tokenizer)
                input_ids = input_ids.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    row = probe_sample(
                        model, input_ids, target, tokenizer,
                        args.chunk_size, device, args.select_layer,
                        num_slots, num_global,
                    )
                row["idx"] = idx
                row["task"] = task
                row["length"] = length
                writer.writerow(row)
                fcsv.flush()
                all_rows.append(row)

    fcsv.close()
    print(f"\n[E2-slot] Wrote {len(all_rows)} rows to {out_path}")

    _print_summary(all_rows, args, mem_config)


def _print_summary(rows, args, mem_config):
    print("\n" + "=" * 92)
    print("[E2-slot] SUMMARY — SLOT write-map needle localization")
    print(f"  CKPT: {args.checkpoint}")
    print("  ★ SLOT ckpt (pg19 nctx63, mix=0). This is a DIFFERENT MODEL from the reader-attn")
    print("    E2 (FIFO ckpt) — recall@K is a CROSS-MODEL comparison, NOT directly subtractable.")
    print("  ★ NO eviction (slot_evict_mode=off): candidates = ALL context chunks "
          "(n_chunks-1),")
    print("    vs reader-attn E2's capped FIFO buffer (~25). Bigger pool ⇒ recall@K is harder;")
    print("    read rank-percentile and recall-vs-chance, NOT just raw recall@K.")
    print(f"  select_layer={args.select_layer}  num_slots={getattr(mem_config,'num_slots',0)}  "
          f"top_k={getattr(mem_config,'top_k',0)}  "
          f"num_global_slots={getattr(mem_config,'num_global_slots',0)} (excluded from W & s_q)")
    print("=" * 92)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["length"])].append(r)

    header = (f"{'task':<6}{'len':<6}{'N':>5}{'loc':>5}{'inC':>5}"
              + "".join(f"{'r@'+str(k):>7}" for k in RECALL_KS)
              + f"{'mRank':>7}{'mPct':>7}{'nCand':>7}{'r@4/ch':>8}{'r@8/ch':>8}")
    print(header)
    print("-" * len(header))

    def _fmt_group(label_task, label_len, grp):
        N = len(grp)
        located = [r for r in grp if r["needle_located"]]
        nloc = len(located)
        in_c = [r for r in located if r["needle_in_candidates"]]
        n_in = len(in_c)
        line = f"{label_task:<6}{label_len:<6}{N:>5}{nloc:>5}{n_in:>5}"
        recall_means = {}
        for k in RECALL_KS:
            vals = [r[f"recall@{k}"] for r in located if isinstance(r[f"recall@{k}"], int)]
            if vals:
                m = sum(vals) / len(vals)
                recall_means[k] = m
                line += f"{m:>7.2f}"
            else:
                line += f"{'-':>7}"
        ranks = [r["needle_rank"] for r in located if isinstance(r["needle_rank"], int)]
        line += f"{median(ranks):>7.1f}" if ranks else f"{'-':>7}"
        pcts = [r["rank_pctile"] for r in located if isinstance(r["rank_pctile"], (int, float)) and r["rank_pctile"] != ""]
        line += f"{median(pcts):>7.3f}" if pcts else f"{'-':>7}"
        ncands = [r["n_candidates"] for r in grp if isinstance(r["n_candidates"], int) and r["n_candidates"] > 0]
        mean_nc = (sum(ncands) / len(ncands)) if ncands else 0.0
        line += f"{mean_nc:>7.1f}" if ncands else f"{'-':>7}"
        # recall-vs-chance for K=4 and K=8 (chance@K = K / mean_n_candidates)
        for k in (4, 8):
            if k in recall_means and mean_nc > 0:
                chance = k / mean_nc
                ratio = recall_means[k] / chance if chance > 0 else float("nan")
                line += f"{ratio:>8.2f}"
            else:
                line += f"{'-':>8}"
        print(line)

    for (task, length), grp in sorted(groups.items()):
        _fmt_group(task, length, grp)

    print("-" * len(header))
    by_len = defaultdict(list)
    for r in rows:
        by_len[r["length"]].append(r)
    for length in sorted(by_len):
        _fmt_group("ALL", length, by_len[length])

    print("=" * 92)
    print("  Legend: inC=needle landed in candidate set; mRank=median needle rank (1=best);")
    print("          mPct=median rank percentile (rank/n_cand, 0=top, 0.5=chance-median);")
    print("          r@4/ch, r@8/ch = recall@K divided by chance K/n_cand (>1 beats random).")
    nloc = sum(1 for r in rows if not r["needle_located"])
    if nloc:
        print(f"  needle NOT located (answer string not found in tokens): {nloc}/{len(rows)} "
              f"samples — excluded from recall/rank stats.")
    print("  ★ Reader-attn E2 reference (DIFFERENT FIFO model, capped-25 pool): "
          "recall@4≈0.15-0.17, @8≈0.34-0.44, @16≈0.62-0.73.")
    print("  ★ Cross-model: compare rank-percentile / recall-vs-chance, do not subtract raw recall@K.")


if __name__ == "__main__":
    main()
