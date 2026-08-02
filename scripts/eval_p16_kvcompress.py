#!/usr/bin/env python
"""Paper A P1.6 — standard SnapKV / PyramidKV KV-cache-compression baselines on
Qwen3-8B, at a FIXED retained KV/token budget (default 6657 = CoMem read budget:
BOS 1 + top-12 x 512 + query <=512), for an equal-retained-token diagnostic.

These are PREFILL-THEN-COMPRESS methods: they run the FULL exact prefill over the
whole prompt, THEN compress the STORED KV so decode is O(retained). Unlike CoMem's
persistent bounded store, they MUST see the entire prompt and pay the full-prefill
compute + peak-memory cost. This harness therefore measures and records, per cell:
  * quality (RULER string_match_all recall / LoCoMo F1-EM-acc)
  * full-prefill latency (s)               [prefill_latency_s]
  * peak GPU memory during prefill (GiB)   [prefill_peak_gb]
  * compressed retained KV bytes + per-layer retained length  [compressed_kv_*]
  * decode latency / tok (ms)              [decode_latency_per_tok_ms]
  * whether the full prompt must be seen   [full_prompt_seen = True, always]
  * OOM / fallback                         [oom_count / status]

Protocol (matched to CoMem / the other P1 baselines): same Qwen3-8B revision,
chat=False, enable_thinking=False, bf16, SDPA, greedy, same RULER/LoCoMo sample
sets + seeds + scorers as scripts/eval_ruler_mem_space & scripts/eval_qcmem_locomo.
Per-example predictions are saved (RULER per-cell CSV, LoCoMo preds JSONL).

Modes:
  --mode ruler      RULER Cohort A cells (task x length), per-cell CSV + JSON.
  --mode locomo     LoCoMo full set (or --categories), preds JSONL + scores.
  --mode selftest   faithfulness gate: (1) short input (< budget) -> hijack is a
                    bit-for-bit no-op vs stock SDPA (max|Δlogit| < 1e-3, argmax
                    identical); (2) long input (> budget) -> per-layer retained
                    length == budget (snapkv uniform / pyramidkv pyramidal sum).
  --mode aggregate  merge sharded RULER cells (sum shards) OR score LoCoMo shards.

Native window vs YaRN (64k/128k > Qwen3-8B native 40960):
  --long_ctx native : cap prompt to the native window (left-truncate) — reported
                      as "<method>-native". This is the default (Qwen3-8B ships
                      NO rope_scaling).
  --long_ctx yarn   : apply YaRN rope_scaling (factor from --yarn_factor) so the
                      full >40960 prompt is prefilled — reported as
                      "<method>-yarn". Bound to the method name in outputs.

Example (single RULER cell, one shard):
  PYTHONHASHSEED=0 .venv/bin/python scripts/eval_p16_kvcompress.py \
    --mode ruler --method snapkv \
    --model_path models/Qwen3-8b-local \
    --tasks niah_single_2 --lengths 8k --num_samples 100 \
    --num_shards 4 --shard_index 0 \
    --results_folder ruler_results/p16_snapkv --output_name snapkv_b6657
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
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

# RULER sample framework (generation + scoring) — reused verbatim, unmodified.
import scripts.eval_ruler_mem_space as ruler  # noqa: E402
# QCMem babilong module — only for the shared QUOTE_ALL CSV writer.
import scripts.eval_qcmem_babilong as qcb  # noqa: E402
# LoCoMo sample framework + scoring — reused verbatim, unmodified.
import scripts.eval_qcmem_locomo as locomo  # noqa: E402
# The faithful Qwen3-5.14 KV-compression hijack (drives the vendored clusters).
from src.baselines import qwen3_kvcompress as kvc  # noqa: E402

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# RULER Cohort A (Paper A tab_h2h): 3 tasks x 5 lengths = 15 cells.
_COHORT_A_TASKS = ["niah_single_2", "niah_multikey_1", "variable_tracking"]
_COHORT_A_LENGTHS = ["8k", "16k", "32k", "64k", "128k"]

_QWEN3_NATIVE_WINDOW = 40960


# --------------------------------------------------------------------------- #
# model load (mirror bench_p0_13 / eval_ruler_qcmem: AutoTokenizer + AutoModel,
# local_files_only, bf16, SDPA). No LoRA (these are training-free baselines).
# --------------------------------------------------------------------------- #
def _load_model(model_path, dtype, attn_impl, device, yarn_factor=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(torch_dtype=dtype, attn_implementation=attn_impl,
                  trust_remote_code=True, local_files_only=True)
    if yarn_factor is not None:
        # YaRN long-context: bound to the "-yarn" method label in outputs.
        kwargs["rope_scaling"] = {
            "rope_type": "yarn", "factor": float(yarn_factor),
            "original_max_position_embeddings": _QWEN3_NATIVE_WINDOW,
        }
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).to(device).eval()
    return tokenizer, model


def _eos_ids(model, tokenizer):
    gc = getattr(model, "generation_config", None)
    eos = getattr(gc, "eos_token_id", None) if gc is not None else None
    if eos is None:
        eos = []
    elif isinstance(eos, int):
        eos = [eos]
    out = {int(e) for e in eos if e is not None}
    if tokenizer.eos_token_id is not None:
        out.add(int(tokenizer.eos_token_id))
    return sorted(out)


def _im_end_ids(tokenizer, use_chat_template):
    if not use_chat_template:
        return []
    try:
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        return [tid] if isinstance(tid, int) and tid >= 0 else []
    except Exception:
        return []


def _maybe_yarn_factor(args):
    """YaRN scaling factor needed to cover the longest requested length, or None
    (native) when --long_ctx native or all lengths fit the native window."""
    if args.long_ctx != "yarn":
        return None
    if args.yarn_factor and args.yarn_factor > 0:
        return args.yarn_factor
    max_tok = max((ruler._LENGTH_TOKENS.get(l, 0) for l in args.lengths), default=0)
    if max_tok <= _QWEN3_NATIVE_WINDOW:
        return None
    # round up to a sensible factor (RULER 128k -> 131072/40960 ~= 3.2 -> 4).
    import math
    return float(max(2, math.ceil(max_tok / _QWEN3_NATIVE_WINDOW)))


def _method_label(args, yarn_factor):
    if yarn_factor is not None:
        return f"{args.method}-yarn{int(yarn_factor)}"
    # native: only tag when a length actually exceeds the native window.
    lengths_over = any(ruler._LENGTH_TOKENS.get(l, 0) > _QWEN3_NATIVE_WINDOW
                       for l in getattr(args, "lengths", []))
    return f"{args.method}-native" if lengths_over else args.method


# --------------------------------------------------------------------------- #
# tokenize a prompt (chat=False by default per Paper A mandate)
# --------------------------------------------------------------------------- #
def _encode(prompt, tokenizer, use_chat_template, enable_thinking, device):
    text = prompt
    if use_chat_template:
        msgs = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking)
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    if isinstance(ids, list):
        ids = torch.tensor([ids], dtype=torch.long)
    return ids.to(device)


# --------------------------------------------------------------------------- #
# RULER mode
# --------------------------------------------------------------------------- #
def run_ruler(args, tokenizer, model, device, yarn_factor):
    end_ids = _im_end_ids(tokenizer, args.use_chat_template)
    eos_ids = _eos_ids(model, tokenizer)
    method_label = _method_label(args, yarn_factor)
    native_cap = _QWEN3_NATIVE_WINDOW if yarn_factor is None else None

    outdir = Path(args.results_folder) / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    L = int(model.config.num_hidden_layers)
    summary: dict = {}
    for task in tqdm(args.tasks, desc="tasks"):
        task = _resolve_ruler_task(task)
        summary[task] = {}
        for length in tqdm(args.lengths, desc="lengths", leave=False):
            if length not in ruler._LENGTH_TOKENS:
                print(f"[WARN] unknown length {length}, skipping")
                continue
            cell_started = time.time()
            target_tokens = ruler._LENGTH_TOKENS[length]
            base_seed = args.seed + (hash((task, length)) % 100000)
            vt_icl = None
            if task == "variable_tracking":
                vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)

            sample_indices = set(
                list(range(args.num_samples))[args.shard_index::args.num_shards])
            mnt = args.max_new_tokens if task != "variable_tracking" \
                else max(args.max_new_tokens, 60)

            df = pd.DataFrame({"target": [], "output": [], "question": [], "recall": []})
            recall_sum, total, oom_count = 0.0, 0, 0
            n_tok_seen = 0
            truncated_any = False
            metric_bucket = {k: [] for k in (
                "prefill_latency_s", "prefill_peak_gb", "decode_latency_per_tok_ms",
                "compressed_kv_MB", "mean_retained_len", "min_retained_len",
                "max_retained_len")}

            for i in tqdm(range(args.num_samples), desc=f"{task}/{length}", leave=False):
                rng = random.Random(base_seed * 1000 + i)
                prompt, answers, _gold = ruler._build_sample(
                    task, target_tokens, tokenizer, rng, vt_icl)
                if i not in sample_indices:
                    continue
                bare_q = prompt[prompt.rfind("\n") + 1:].strip()
                input_ids = _encode(prompt, tokenizer, args.use_chat_template,
                                    args.enable_thinking, device)
                if native_cap is not None and input_ids.shape[1] > native_cap:
                    input_ids = input_ids[:, -native_cap:]  # left-truncate to native window
                    truncated_any = True
                n_tok_seen = int(input_ids.shape[1])

                gstats: dict = {}
                try:
                    gen = kvc.generate_kvcompress(
                        model, input_ids, max_new_tokens=mnt,
                        eos_token_ids=eos_ids, extra_end_token_ids=end_ids,
                        stats=gstats)
                    output = tokenizer.decode(gen, skip_special_tokens=True).strip()
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    output = "[OOM]"
                    oom_count += 1
                    print(f"[OOM] i={i} {task}/{length}: {e}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                rec = ruler._string_match_all_one(output, answers)
                recall_sum += rec
                total += 1
                for k in metric_bucket:
                    if gstats.get(k) is not None:
                        metric_bucket[k].append(gstats[k])
                df.loc[len(df)] = [" | ".join(answers), output, bare_q, rec]
                if len(df) % 10 == 0:
                    qcb.harness._write_results_csv(df, outdir / f"{task}_{length}{shard_tag}.csv")

            score = (recall_sum / total * 100.0) if total else 0.0
            perf = {k: (round(sum(v) / len(v), 4) if v else None)
                    for k, v in metric_bucket.items()}
            summary[task][length] = {"score": round(score, 2), "n": total,
                                     "approx_tokens": n_tok_seen, "perf": perf}
            qcb.harness._write_results_csv(df, outdir / f"{task}_{length}{shard_tag}.csv")
            json.dump(
                {
                    "status": "completed" if oom_count == 0 else "failed",
                    "task": task, "length": length,
                    "n_requested": args.num_samples,
                    "sharding": {"num_shards": args.num_shards,
                                 "shard_index": args.shard_index},
                    "summary": summary[task][length],
                    "score": summary[task][length]["score"],
                    "oom_count": oom_count,
                    "elapsed_seconds": round(time.time() - cell_started, 3),
                    "runtime": {
                        "node": socket.gethostname(),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "device": args.device, "seed": args.seed,
                        "dtype": args.dtype, "attn_implementation": args.attn_impl,
                    },
                    "chat_template": bool(args.use_chat_template),
                    "enable_thinking": bool(args.enable_thinking),
                    "scoring": "scripts.eval_ruler_mem_space._string_match_all_one",
                    "baseline": method_label,
                    "kvcompress": {
                        "method": args.method,
                        "max_capacity_prompt": args.max_capacity_prompt,
                        "window_size": args.window_size,
                        "kernel_size": args.kernel_size, "pooling": args.pooling,
                        "gqa_score_agg": args.gqa_score_agg,
                        "long_ctx": args.long_ctx,
                        "yarn_factor": yarn_factor,
                        "native_window": _QWEN3_NATIVE_WINDOW,
                        "prompt_truncated_to_native": truncated_any,
                        "full_prompt_seen": (not truncated_any),
                        "num_layers": L,
                    },
                    "model": {"model_path": args.model_path, "num_hidden_layers": L},
                },
                open(outdir / f"{task}_{length}{shard_tag}.json", "w"), indent=2,
            )
            pf = summary[task][length]["perf"]
            print(f"[P16-RULER:{method_label}] {task}/{length}: recall={score:.2f} "
                  f"({total} n, ~{n_tok_seen} tok) | prefill={pf['prefill_latency_s']}s "
                  f"peak={pf['prefill_peak_gb']}GB decode/tok={pf['decode_latency_per_tok_ms']}ms "
                  f"kv={pf['compressed_kv_MB']}MB retained~{pf['mean_retained_len']}")

    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[P16-RULER:{method_label}] done -> {outdir}")


def _resolve_ruler_task(name):
    alias = {"niah_single": "niah_single_2", "niah_multi": "niah_multikey_1",
             "niah_multikey": "niah_multikey_1", "vt": "variable_tracking"}
    if name in ruler._NIAH_TASK_CFG or name == "variable_tracking":
        return name
    if name in alias:
        return alias[name]
    raise ValueError(f"unknown ruler task {name!r}")


# --------------------------------------------------------------------------- #
# LoCoMo mode
# --------------------------------------------------------------------------- #
def run_locomo(args, tokenizer, model, device, yarn_factor):
    end_ids = _im_end_ids(tokenizer, args.use_chat_template)
    eos_ids = _eos_ids(model, tokenizer)
    method_label = _method_label(args, yarn_factor)
    native_cap = _QWEN3_NATIVE_WINDOW if yarn_factor is None else None
    L = int(model.config.num_hidden_layers)

    data_path = args.locomo_data or os.path.join(PROJECT_ROOT, "locomo", "data", "locomo10.json")
    samples = locomo.build_locomo_samples(data_path)
    categories = None
    if args.categories:
        categories = set(int(c) for c in args.categories.split(","))
        samples = [s for s in samples if s["category"] in categories]
    if args.max_samples and args.max_samples > 0:
        samples = samples[:args.max_samples]
    shard = samples[args.shard_index::args.num_shards]
    print(f"[P16-LoCoMo:{method_label}] {len(samples)} samples; "
          f"shard {args.shard_index}/{args.num_shards} -> {len(shard)}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    with open(outdir / f"eval_config{shard_tag}.json", "w") as f:
        cfg = dict(vars(args))
        cfg.update({"baseline": method_label, "num_layers": L,
                    "resolved_data_path": data_path, "yarn_factor": yarn_factor,
                    "native_window": _QWEN3_NATIVE_WINDOW})
        json.dump(cfg, f, indent=2)

    outfile = outdir / f"preds{shard_tag}.jsonl"
    buf = []
    oom_count = 0
    t0 = time.time()
    for pos, sample in enumerate(tqdm(shard, desc="locomo", leave=True)):
        input_ids = _encode(sample["prompt"], tokenizer, args.use_chat_template,
                            args.enable_thinking, device)
        truncated = False
        if native_cap is not None and input_ids.shape[1] > native_cap:
            input_ids = input_ids[:, -native_cap:]
            truncated = True
        n_tokens = int(input_ids.shape[1])
        gstats: dict = {}
        try:
            gen = kvc.generate_kvcompress(
                model, input_ids, max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_ids, extra_end_token_ids=end_ids, stats=gstats)
            pred = tokenizer.decode(gen, skip_special_tokens=True).strip()
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            pred = "[OOM]"
            oom_count += 1
            print(f"[OOM] id={sample['id']} n_tok={n_tokens}: {e}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        buf.append({
            "id": sample["id"], "pred": pred, "answers": sample["answers"],
            "category": sample["category"], "is_abstention": sample["is_abstention"],
            "question": sample["question"], "n_tokens": n_tokens,
            "prompt_truncated_to_native": truncated,
            "prefill_latency_s": gstats.get("prefill_latency_s"),
            "prefill_peak_gb": gstats.get("prefill_peak_gb"),
            "decode_latency_per_tok_ms": gstats.get("decode_latency_per_tok_ms"),
            "compressed_kv_MB": gstats.get("compressed_kv_MB"),
            "mean_retained_len": gstats.get("mean_retained_len"),
            "full_prompt_seen": gstats.get("full_prompt_seen") and not truncated,
        })
        if (pos + 1) % 10 == 0 or pos == len(shard) - 1:
            with open(outfile, "w") as f:
                for r in buf:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(outfile, "w") as f:
        for r in buf:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[P16-LoCoMo:{method_label}] shard done: {len(buf)} preds "
          f"(oom={oom_count}, {time.time()-t0:.1f}s) -> {outfile}")

    if args.num_shards == 1:
        print(f"\n[P16-LoCoMo:{method_label}] scoring (single-shard)...")
        locomo.run_scoring(args.output_dir, use_bertscore=args.use_bertscore,
                           use_llm_judge=args.use_llm_judge,
                           judge_model=args.judge_model,
                           judge_base_url=args.judge_base_url,
                           judge_api_key=args.judge_api_key,
                           judge_workers=args.judge_workers)


# --------------------------------------------------------------------------- #
# selftest mode — faithfulness gate
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_selftest(args, tokenizer, model, device):
    print(f"[P16-selftest] method={args.method} budget={args.max_capacity_prompt} "
          f"window={args.window_size}")

    # (0) capture stock full-KV logits BEFORE installing the hijack.
    short_ids = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. " * 20,
        add_special_tokens=True, return_tensors="pt").to(device)
    short_len = int(short_ids.shape[1])
    assert short_len < args.max_capacity_prompt, \
        f"selftest short input ({short_len}) must be < budget ({args.max_capacity_prompt})"
    stock_logits = model(input_ids=short_ids, use_cache=False).logits[:, -1, :].float().cpu()

    # (1) install hijack; short input (< budget) must be a bit-for-bit no-op.
    cfg = kvc.install_kv_compression(
        model, args.method, max_capacity_prompt=args.max_capacity_prompt,
        window_size=args.window_size, kernel_size=args.kernel_size,
        pooling=args.pooling, gqa_score_agg=args.gqa_score_agg, beta=args.beta)
    print(f"[P16-selftest] installed: {cfg}")

    from transformers.cache_utils import DynamicCache
    cache = DynamicCache(config=model.config)
    hijack_logits = model(input_ids=short_ids, past_key_values=cache,
                          use_cache=True).logits[:, -1, :].float().cpu()
    max_dlogit = float((hijack_logits - stock_logits).abs().max().item())
    argmax_same = bool(int(hijack_logits.argmax(-1)) == int(stock_logits.argmax(-1)))
    stats0 = kvc.compressed_kv_bytes(cache)
    no_compress_ok = (max_dlogit < 1e-3) and argmax_same
    print(f"[P16-selftest] (1) short<budget no-op: max|Δlogit|={max_dlogit:.3e} "
          f"argmax_same={argmax_same} retained_len(layer0)={stats0[1][0]} "
          f"(should == input_len {short_len}) -> {'PASS' if no_compress_ok else 'FAIL'}")

    # (2) long input (> budget): per-layer retained length == budget.
    long_text = "".join("token%d " % i for i in range(args.max_capacity_prompt + 4000))
    long_ids = tokenizer.encode(long_text, add_special_tokens=True,
                                return_tensors="pt").to(device)
    long_len = int(long_ids.shape[1])
    assert long_len > args.max_capacity_prompt, \
        f"selftest long input ({long_len}) must exceed budget ({args.max_capacity_prompt})"
    cache2 = DynamicCache(config=model.config)
    _ = model(input_ids=long_ids, past_key_values=cache2, use_cache=True)
    _, per_layer = kvc.compressed_kv_bytes(cache2)
    budget = args.max_capacity_prompt
    if args.method == "snapkv":
        # uniform: every layer holds exactly the budget.
        ok_budget = all(pl == budget for pl in per_layer)
        detail = f"all==budget({budget})? {ok_budget}; observed set={sorted(set(per_layer))}"
    else:
        # pyramidkv: per-layer varies pyramidally — LOWER layers retain MORE than
        # the uniform budget, HIGHER layers retain LESS (down to ~window_size), and
        # the per-layer mean tracks the budget. This is the method's defining
        # behaviour, so the floor is the recent window (not budget-window), and we
        # additionally assert the pyramid is non-increasing with layer index.
        lo, hi = min(per_layer), max(per_layer)
        mean = sum(per_layer) / len(per_layer)
        bounded = all(args.window_size <= pl <= long_len for pl in per_layer)
        varied = (lo != hi)
        mean_ok = abs(mean - budget) <= max(2, 0.05 * budget)
        monotonic = all(per_layer[i] >= per_layer[i + 1] for i in range(len(per_layer) - 1))
        ok_budget = bounded and varied and mean_ok and monotonic
        detail = (f"pyramidal min={lo} max={hi} mean={mean:.1f} budget={budget} "
                  f"bounded={bounded} varied={varied} mean_ok={mean_ok} monotonic={monotonic}")
    print(f"[P16-selftest] (2) long>budget retained: {detail} "
          f"-> {'PASS' if ok_budget else 'FAIL'}")

    kvc.uninstall_kv_compression(model)
    overall = "PASS" if (no_compress_ok and ok_budget) else "FAIL"
    print(f"[P16-selftest] OVERALL: {overall}")
    result = {
        "method": args.method, "budget": budget, "window_size": args.window_size,
        "no_compress_noop": {"max_abs_dlogit": max_dlogit, "argmax_same": argmax_same,
                             "retained_len_layer0": stats0[1][0], "input_len": short_len,
                             "pass": no_compress_ok},
        "over_budget_retained": {"per_layer_retained_len": per_layer,
                                 "input_len": long_len, "pass": bool(ok_budget)},
        "overall_pass": (no_compress_ok and bool(ok_budget)),
    }
    if args.selftest_out:
        Path(args.selftest_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.selftest_out, "w"), indent=2)
        print(f"[P16-selftest] wrote {args.selftest_out}")
    return result


# --------------------------------------------------------------------------- #
# aggregate mode — merge sharded RULER cells (sum shards) / score LoCoMo shards
# --------------------------------------------------------------------------- #
def run_aggregate(args):
    if args.agg_kind == "locomo":
        print(f"[P16-agg] scoring LoCoMo shards in {args.output_dir}")
        locomo.run_scoring(args.output_dir, use_bertscore=args.use_bertscore,
                           use_llm_judge=args.use_llm_judge,
                           judge_model=args.judge_model,
                           judge_base_url=args.judge_base_url,
                           judge_api_key=args.judge_api_key,
                           judge_workers=args.judge_workers)
        return
    # RULER: delegate to the shared shard-merge scorer if present.
    print(f"[P16-agg] RULER shard merge: use scripts/score_nested_babilong.py "
          f"over {args.results_folder}/{args.output_name} "
          f"(sums _shard{{i}}of{{N}} CSVs per cell). See launcher for the exact command.")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Paper A P1.6 SnapKV/PyramidKV KV-compression baselines")
    p.add_argument("--mode", choices=["ruler", "locomo", "selftest", "aggregate"], required=True)
    p.add_argument("--method", choices=["snapkv", "pyramidkv"], default="snapkv")
    p.add_argument("--model_path", type=str, default="models/Qwen3-8b-local")

    # KV-compression budget (default = CoMem read budget 6657).
    p.add_argument("--max_capacity_prompt", type=int, default=6657,
                   help="total retained tokens per layer INCLUDING the recent window")
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--pooling", choices=["avgpool", "maxpool"], default="avgpool")
    p.add_argument("--gqa_score_agg", choices=["mean", "max", "sum"], default="mean")
    p.add_argument("--beta", type=int, default=20, help="PyramidKV pyramidal slope")

    # protocol
    p.add_argument("--dtype", choices=list(_DTYPES), default="bfloat16")
    p.add_argument("--attn_impl", choices=["sdpa", "eager"], default="sdpa")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_chat_template", action="store_true", default=False,
                   help="Paper A mandate: default OFF (chat=False).")
    p.add_argument("--enable_thinking", action="store_true", default=False)
    p.add_argument("--max_new_tokens", type=int, default=48)

    # long-context handling (64k/128k > native 40960)
    p.add_argument("--long_ctx", choices=["native", "yarn"], default="native")
    p.add_argument("--yarn_factor", type=float, default=0.0,
                   help="explicit YaRN factor (0 -> auto from longest length)")

    # RULER
    p.add_argument("--tasks", type=str, nargs="+", default=_COHORT_A_TASKS)
    p.add_argument("--lengths", type=str, nargs="+", default=_COHORT_A_LENGTHS)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--results_folder", type=str, default="ruler_results/p16")
    p.add_argument("--output_name", type=str, default="p16_kvcompress")

    # LoCoMo
    p.add_argument("--locomo_data", type=str, default="")
    p.add_argument("--categories", type=str, default=None, help="comma list, e.g. 1,2,3,4,5")
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--output_dir", type=str, default="locomo_results/p16_kvcompress")
    p.add_argument("--use_bertscore", action="store_true", default=False)
    p.add_argument("--use_llm_judge", action="store_true", default=False)
    p.add_argument("--judge_model", type=str, default="gpt-4o")
    p.add_argument("--judge_base_url", type=str, default=None)
    p.add_argument("--judge_api_key", type=str, default=None)
    p.add_argument("--judge_workers", type=int, default=8)

    # sharding (task-pool: [shard_index::num_shards])
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)

    # selftest / aggregate
    p.add_argument("--selftest_out", type=str, default="")
    p.add_argument("--agg_kind", choices=["ruler", "locomo"], default="ruler")
    args = p.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        p.error("bad shard config")

    if args.mode == "aggregate":
        run_aggregate(args)
        return

    if not args.locomo_data:
        args.locomo_data = os.path.join(PROJECT_ROOT, "locomo", "data", "locomo10.json")

    # normalize model_path relative to PROJECT_ROOT if not absolute.
    if not os.path.isabs(args.model_path) and not os.path.exists(args.model_path):
        cand = os.path.join(PROJECT_ROOT, args.model_path)
        if os.path.exists(cand):
            args.model_path = cand

    device = torch.device(args.device)
    dtype = _DTYPES[args.dtype]

    print(f"[P16] mode={args.mode} method={args.method} model={args.model_path}")
    print(f"[P16] budget(max_capacity_prompt)={args.max_capacity_prompt} "
          f"window={args.window_size} chat={args.use_chat_template} "
          f"think={args.enable_thinking} dtype={args.dtype} attn={args.attn_impl}")

    if args.mode == "selftest":
        tokenizer, model = _load_model(args.model_path, dtype, args.attn_impl, device, None)
        run_selftest(args, tokenizer, model, device)
        return

    yarn_factor = _maybe_yarn_factor(args)
    tokenizer, model = _load_model(args.model_path, dtype, args.attn_impl, device, yarn_factor)
    cfg = kvc.install_kv_compression(
        model, args.method, max_capacity_prompt=args.max_capacity_prompt,
        window_size=args.window_size, kernel_size=args.kernel_size,
        pooling=args.pooling, gqa_score_agg=args.gqa_score_agg, beta=args.beta)
    print(f"[P16] installed KV-compression: {cfg}  long_ctx={args.long_ctx} "
          f"yarn_factor={yarn_factor}")

    if args.mode == "ruler":
        run_ruler(args, tokenizer, model, device, yarn_factor)
    elif args.mode == "locomo":
        run_locomo(args, tokenizer, model, device, yarn_factor)


if __name__ == "__main__":
    main()
