#!/usr/bin/env python
"""QCMem mid-depth resume — **Hy3** (Hunyuan hy_v3, 80-layer MoE) long-context
RULER eval driver (2026-07-12).

This is the multi-GPU / long-context companion to ``scripts/eval_ruler_qcmem.py``
(the Qwen/Llama single-device RULER-QCMem driver). It runs the SAME real
long-document RULER tasks (NIAH single / multikey / variable_tracking) with the
SAME RULER-faithful sample generation + official ``string_match_all`` scoring,
but on the 597 GB Hy3 backbone sharded across the local 8 L20A via
``device_map="auto"`` and wrapped in :class:`QCMemHy3Model`.

Why a separate Hy3 driver (vs. adding a branch to eval_ruler_qcmem.py)
---------------------------------------------------------------------
Two Hy3-specific pieces cannot be expressed with eval_ruler_qcmem.py's
single-device ``.to(device)`` load + ``PeftModel.from_pretrained`` load:

  1. **Sharded load.** Hy3 only fits when ``device_map="auto"`` splits its 80
     ``HYV3DecoderLayer`` blocks across the GPUs; the write/read loop must hop
     the residual stream between shards, which is exactly what
     :class:`QCMemHy3Model` (device-aware ``_run_layers`` + norm/lm_head) does.
  2. **LoRA load.** ``PeftModel.from_pretrained(model, adapter)`` CRASHES on a
     device_map-sharded Hy3 in this stack (peft 0.19 + tf 5.13.1):
       TypeError: WeightConverter.__init__() got an unexpected keyword
                  argument 'distributed_operation'
     So we reuse the PROVEN loader from ``scripts/qcmem_hy3_jsweep.py``: rebuild
     the identical empty LoRA structure with ``get_peft_model`` (the exact call
     the distill trainer used to CREATE the weights), then blit the trained
     tensors in with a manual key remap + plain ``load_state_dict(strict=False)``
     — pure torch, no peft/HF WeightConverter, so the bug never fires. A hard
     ``sum|lora_B| >> 0`` sanity aborts rather than report a fake (no-op) result.

Everything else is imported UNCHANGED from the two shipped drivers:

  * RULER task framework  (``scripts/eval_ruler_mem_space``):
      ``_build_sample`` / ``_make_vt_icl`` / ``_string_match_all_one`` (official
      RULER string_match_all recall — NOT a hand-rolled re.search) /
      ``_LENGTH_TOKENS``.
  * QCMem forward path     (``scripts/eval_qcmem_babilong``):
      ``qcmem_generate`` — chunk prompt -> write each SELECTED chunk to depth j
      -> bm25 topk selector picks a FIXED number of context chunks -> read
      (pack [sink ; selected h_j ; query h_j], resume layers[j:]) -> greedy
      decode. Because the selector packs a fixed ``topk`` chunks, the read length
      (``stats['read_len']`` = sink + topk*chunk + query) is CONSTANT regardless
      of context length — this is what makes 256k tractable (WRITE is cheap,
      chunk-local depth-j hidden; READ never grows). We report ``avg_read_len``
      per cell to make the constant-read property auditable.
  * RULER-QCMem glue       (``scripts/eval_ruler_qcmem``):
      ``_resolve_task`` (task aliases) / ``_bare_question`` (bm25 query) /
      ``_oracle_needle_chunks``.

The sample RNG + shard filter are replicated bit-for-bit from
``eval_ruler_qcmem.main`` (identical to ``eval_ruler_mem_space.main``), so shards
share one sample set and the needle format / scoring口径 are identical — only the
model forward differs (sharded Hy3 QCMem instead of Qwen/Llama QCMem).

Usage (local 8x L20A, distilled j32 adapter):
    .venv_hy3/bin/python scripts/eval_ruler_qcmem_hy3.py \
        --model_path /apdcephfs_wzc1/share_304376610/pighzliu_code/models/Hy3 \
        --resume_j 32 \
        --lora_adapter outputs/qcmem_distill_hy3_j32_r32/final \
        --selector bm25 --topk 8 --sink_tokens bos \
        --ruler_tasks niah_single niah_multikey --lengths 16k 32k 64k 128k \
        --limit 50 --output_name hy3_qcmem_j32 \
        --results_folder ruler_results/hy3_qcmem_j32
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# RULER task framework (generation + scoring) — reused verbatim, unmodified.
import scripts.eval_ruler_mem_space as ruler  # noqa: E402
# QCMem forward path (chunk / write / select / read / decode) — reused verbatim.
import scripts.eval_qcmem_babilong as qcb  # noqa: E402
# RULER-QCMem glue (task aliases, bm25 query, oracle needle locator) — reused.
import scripts.eval_ruler_qcmem as ruler_qcmem  # noqa: E402

from src.memory.qcmem.qcmem_hy3 import QCMemHy3Model  # noqa: E402

qcmem_generate = qcb.qcmem_generate
_resolve_task = ruler_qcmem._resolve_task
_bare_question = ruler_qcmem._bare_question
_oracle_needle_chunks = ruler_qcmem._oracle_needle_chunks


# --------------------------------------------------------------------------- #
# LoRA loader (proven path, copied from scripts/qcmem_hy3_jsweep.py)
# --------------------------------------------------------------------------- #
def _load_hy3_lora(model, lora_adapter: str):
    """Load a QCMem-distill LoRA adapter onto a device_map-sharded Hy3.

    Rebuild the identical empty LoRA structure with ``get_peft_model`` (proven to
    work under device_map — it is what the distill trainer used to CREATE these
    weights), then blit the trained tensors in with a MANUAL key remap + plain
    ``nn.Module.load_state_dict(strict=False)``. Pure torch, no peft/HF
    WeightConverter (the ``distributed_operation`` bug never fires). Returns the
    UNDERLYING HYV3ForCausalLM (LoRA modules swapped in-place, adapters enabled),
    so every downstream access (``model.config`` / ``model.model.*`` /
    ``QCMemHy3Model(model, ...)``) is identical to the no-adapter path.

    On-disk keys ``'...q_proj.lora_A.weight'`` differ from live param names
    ``'...q_proj.lora_A.default.weight'`` only by the missing adapter-name
    segment, which we insert before the trailing ``.weight``. Aborts on any
    unmapped / unexpected key or ``sum|lora_B| == 0`` (silent no-op).
    """
    import json as _json
    from peft import LoraConfig, get_peft_model, TaskType
    from safetensors.torch import load_file

    cfg_path = os.path.join(lora_adapter, "adapter_config.json")
    wt_path = os.path.join(lora_adapter, "adapter_model.safetensors")
    acfg = _json.load(open(cfg_path))
    print(f"[hy3-ruler] rebuilding LoRA from {cfg_path} "
          f"(r={acfg['r']} a={acfg['lora_alpha']} "
          f"layers[{acfg['layers_to_transform'][0]}:"
          f"{acfg['layers_to_transform'][-1]}] "
          f"targets={acfg['target_modules']})", flush=True)
    for prm in model.parameters():
        prm.requires_grad = False
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(acfg["r"]), lora_alpha=int(acfg["lora_alpha"]),
        lora_dropout=float(acfg.get("lora_dropout", 0.0)),
        target_modules=list(acfg["target_modules"]),
        layers_to_transform=list(acfg["layers_to_transform"]),
        layers_pattern=acfg.get("layers_pattern", "layers"),
        bias=acfg.get("bias", "none"),
        use_rslora=bool(acfg.get("use_rslora", False)),
    )
    peft_model = get_peft_model(model, lora_cfg)

    sd = load_file(wt_path)  # CPU bf16 tensors, keys = '...lora_{A,B}.weight'
    model_keys = set(peft_model.state_dict().keys())
    remapped = {}
    no_home = []
    for k, v in sd.items():
        if not k.endswith(".weight"):
            no_home.append(k)
            continue
        tgt = k[: -len(".weight")] + ".default.weight"
        if tgt in model_keys:
            remapped[tgt] = v
        else:
            no_home.append(k)
    load_res = peft_model.load_state_dict(remapped, strict=False)
    unexpected = list(getattr(load_res, "unexpected_keys", []))
    b_abs = sum(float(v.abs().sum()) for k, v in sd.items() if "lora_B" in k)
    print(f"[hy3-ruler] adapter loaded: {len(sd)} on-disk tensors -> "
          f"{len(remapped)} mapped | no_home={len(no_home)} "
          f"unexpected={len(unexpected)} | sum|lora_B|={b_abs:.3e} (>>0 req)",
          flush=True)
    if b_abs == 0.0 or no_home or unexpected or len(remapped) != len(sd):
        raise SystemExit(
            f"[hy3-ruler] ADAPTER LOAD FAILED (silent no-op risk): "
            f"no_home={no_home[:4]} unexpected={unexpected[:4]} "
            f"mapped={len(remapped)}/{len(sd)} sum|B|={b_abs}. Aborting.")
    peft_model.eval()
    print("[hy3-ruler] distilled LoRA active on underlying HYV3ForCausalLM",
          flush=True)
    return peft_model.base_model.model


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="QCMem mid-depth resume — Hy3 (sharded) RULER long-context eval"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Hy3 (hy_v3) backbone dir.")
    parser.add_argument("--resume_j", type=int, default=32,
                        help="Layer split index j (0=RAG upper bound, L=closed-book). "
                             "Distilled adapter was trained at j=32.")
    parser.add_argument("--lora_adapter", type=str, default="",
                        help="Path to the trained QCMem-distill LoRA adapter dir "
                             "(loaded via the manual sharded-safe loader).")
    parser.add_argument("--baseline", type=str, default="none",
                        choices=["none", "hcache", "kvdirect"],
                        help="Mechanism-level baseline. 'none'=QCMem (bm25 topk, "
                             "CONSTANT read). 'hcache'/'kvdirect'=NO retrieval "
                             "(pack every context chunk -> read grows O(context), "
                             "OOMs at long lengths — the comparison that shows why "
                             "constant-read retrieval is required past the window).")
    parser.add_argument("--selector", type=str, default="bm25",
                        choices=["bm25", "recency", "oracle", "reader_attn"],
                        help="Chunk selector. oracle is NIAH-only (degrades to "
                             "recency on variable_tracking).")
    parser.add_argument("--topk", type=int, default=8,
                        help="Number of context chunks packed into the read "
                             "(fixed => constant read length).")
    parser.add_argument("--sink_tokens", type=str, default="bos",
                        choices=["bos", "none"])
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed (matches eval_ruler_qcmem sample set).")
    parser.add_argument("--ruler_tasks", type=str, nargs="+",
                        default=["niah_single", "niah_multikey"],
                        help="RULER tasks/aliases: niah_single(_1/_2), "
                             "niah_multi(key_1), vt(variable_tracking).")
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["16k", "32k", "64k", "128k"])
    parser.add_argument("--results_folder", type=str, default="./ruler_results")
    parser.add_argument("--output_name", type=str, required=True)
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard_index must be in [0, {args.num_shards})")

    no_retrieval = (args.baseline != "none")
    if args.baseline == "kvdirect":
        if args.resume_j != 0:
            print(f"[hy3-ruler] baseline=kvdirect -> forcing resume_j "
                  f"{args.resume_j} -> 0 (full-depth recompute).")
        args.resume_j = 0
        if args.lora_adapter:
            print("[hy3-ruler] baseline=kvdirect is training-free -> ignoring "
                  f"--lora_adapter {args.lora_adapter!r}.")
            args.lora_adapter = ""
    elif args.baseline == "hcache":
        if args.lora_adapter:
            print("[hy3-ruler] baseline=hcache is post-hoc -> ignoring "
                  f"--lora_adapter {args.lora_adapter!r}.")
            args.lora_adapter = ""

    tasks = [_resolve_task(t) for t in args.ruler_tasks]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[hy3-ruler] model_path={args.model_path}")
    print(f"[hy3-ruler] baseline={args.baseline} (no_retrieval={no_retrieval}) "
          f"resume_j={args.resume_j} selector={args.selector} topk={args.topk} "
          f"sink={args.sink_tokens} chunk_size={args.chunk_size} dtype={dtype} "
          f"attn_impl={args.attn_impl}")
    print(f"[hy3-ruler] tasks={tasks} lengths={args.lengths} limit={args.limit}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    t0 = time.time()
    print(f"[hy3-ruler] loading Hy3 device_map=auto dtype={dtype} "
          f"attn={args.attn_impl} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto",
        attn_implementation=args.attn_impl, low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()

    if args.lora_adapter:
        model = _load_hy3_lora(model, args.lora_adapter)

    dm = getattr(model, "hf_device_map", None)
    if dm is not None:
        devs = sorted({str(v) for v in dm.values()})
        print(f"[hy3-ruler] hf_device_map spans {len(devs)} device(s): {devs}",
              flush=True)

    L = int(model.config.num_hidden_layers)
    if not (0 <= args.resume_j <= L):
        parser.error(f"--resume_j must be in [0, {L}]; got {args.resume_j}")
    print(f"[hy3-ruler] Hy3 loaded in {time.time()-t0:.0f}s | L={L}", flush=True)

    qc = QCMemHy3Model(model, resume_j=args.resume_j)
    device = qc.device  # embed device — input_ids live here; write_chunk hops.

    outdir = Path(args.results_folder) / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    summary: dict = {}
    for task in tqdm(tasks, desc="tasks"):
        summary[task] = {}
        for length in tqdm(args.lengths, desc="lengths", leave=False):
            if length not in ruler._LENGTH_TOKENS:
                print(f"[WARN] unknown length {length}, skipping")
                continue
            target_tokens = ruler._LENGTH_TOKENS[length]
            base_seed = args.seed + (hash((task, length)) % 100000)

            vt_icl = None
            if task == "variable_tracking":
                vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)

            sample_indices = set(
                list(range(args.limit))[args.shard_index::args.num_shards]
            )
            if sharded:
                print(f"[hy3-ruler] {task}/{length} shard "
                      f"{args.shard_index}/{args.num_shards}: "
                      f"{len(sample_indices)} of {args.limit} samples")

            df = pd.DataFrame({"target": [], "output": [], "question": [],
                               "recall": []})
            recall_sum = 0.0
            total = 0
            n_tok_seen = 0
            read_len_sum = 0
            read_len_last = 0
            mnt = args.max_new_tokens if task != "variable_tracking" \
                else max(args.max_new_tokens, 60)

            cell_t0 = time.time()
            for i in tqdm(range(args.limit), desc=f"{task}/{length}", leave=False):
                # Build EVERY sample (fixed per-i seed) so shard sample sets align.
                rng = random.Random(base_seed * 1000 + i)
                prompt, answers, gold_needle = ruler._build_sample(
                    task, target_tokens, tokenizer, rng, vt_icl)
                if i not in sample_indices:
                    continue

                ids = tokenizer.encode(prompt, add_special_tokens=True,
                                       return_tensors="pt")
                if isinstance(ids, list):
                    ids = torch.tensor([ids], dtype=torch.long)
                input_ids = ids.to(device)
                n_tok_seen = int(input_ids.shape[1])

                bare_q = _bare_question(prompt)
                bare_q_ids = tokenizer.encode(bare_q, add_special_tokens=False)

                needle_set = None
                if args.selector == "oracle":
                    needle_set = _oracle_needle_chunks(
                        input_ids, gold_needle, answers,
                        tokenizer, args.chunk_size)

                try:
                    gen_stats: dict = {}
                    output = qcmem_generate(
                        qc=qc, tokenizer=tokenizer, input_ids=input_ids,
                        chunk_size=args.chunk_size, max_new_tokens=mnt,
                        selector=args.selector, topk=args.topk,
                        sink_tokens=args.sink_tokens,
                        needle_chunk_set=needle_set, bare_question_ids=bare_q_ids,
                        no_retrieval=no_retrieval, stats=gen_stats,
                    )
                    if "read_len" in gen_stats:
                        read_len_last = int(gen_stats["read_len"])
                        read_len_sum += read_len_last
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    output = "[OOM]"
                    print(f"[OOM] i={i} task={task} length={length}: {e}",
                          flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                rec = ruler._string_match_all_one(output, answers)
                recall_sum += rec
                total += 1
                df.loc[len(df)] = [" | ".join(answers), output, bare_q, rec]
                if len(df) % 10 == 0:
                    qcb.harness._write_results_csv(
                        df, outdir / f"{task}_{length}{shard_tag}.csv")

            score = (recall_sum / total * 100.0) if total else 0.0
            avg_read_len = round(read_len_sum / total, 1) if total else 0
            summary[task][length] = {
                "score": round(score, 2), "n": total,
                "approx_tokens": n_tok_seen,
                "avg_read_len": avg_read_len,      # CONSTANT for QCMem (fixed topk)
                "last_read_len": read_len_last,
                "secs": round(time.time() - cell_t0, 1),
            }
            outfile = outdir / f"{task}_{length}{shard_tag}.csv"
            qcb.harness._write_results_csv(df, outfile)
            cfg_file = outdir / f"{task}_{length}{shard_tag}.json"
            json.dump(
                {
                    "task": task, "length": length,
                    "summary": summary[task][length],
                    "baseline": args.baseline,
                    "no_retrieval": bool(no_retrieval),
                    "qcmem": {
                        "resume_j": args.resume_j,
                        "selector": (None if no_retrieval else args.selector),
                        "topk": (None if no_retrieval else args.topk),
                        "sink_tokens": args.sink_tokens, "num_layers": L,
                        "lora_adapter": args.lora_adapter or None,
                        "chunk_size": args.chunk_size,
                    },
                    "model": {"model_path": args.model_path, "backbone": "Hy3"},
                },
                open(cfg_file, "w"), indent=2,
            )
            print(f"[hy3-ruler] {task}/{length}: recall={score:.2f} "
                  f"({total} samples, ~{n_tok_seen} tok, "
                  f"read_len~{avg_read_len}, {summary[task][length]['secs']:.0f}s) "
                  f"-> {outfile}", flush=True)

    print("\n[hy3-ruler] SUMMARY (recall | read_len)")
    for task in summary:
        row = "  ".join(
            f"{ln}={summary[task][ln]['score']:.1f}"
            f"(rl{summary[task][ln]['avg_read_len']})" for ln in summary[task])
        print(f"  {task:>18}: {row}")
    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[hy3-ruler] Evaluation complete!")
    print("HY3_RULER_DONE", flush=True)


if __name__ == "__main__":
    main()
