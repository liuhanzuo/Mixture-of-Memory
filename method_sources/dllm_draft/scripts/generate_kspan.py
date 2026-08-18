#!/usr/bin/env python3
"""k-span (multi-region) code-infilling generation: diffusion vs AR-FIM arms.

Consumes a FROZEN spec produced by ``scripts/build_kspan_infilling.py`` so every
arm sees byte-identical holes. Nothing about hole placement is decided here.

ARMS
====
diffusion
    Dream-Coder-v0-Instruct-7B, ONE bidirectional canvas containing all k mask
    regions simultaneously. Oracle per-hole token length (this sampler has a
    fixed canvas and cannot choose its own span length). steps = n_masks, so
    exactly one token is committed per step. T=0, top_p=0.95, alg=entropy.

ar_fim
    Qwen2.5-Coder-7B native FIM sentinels, holes filled sequentially
    left-to-right (InCoder Appendix B.2). At hole j the already-filled holes
    < j are substituted with the model's own text; holes > j are DELETED from
    the suffix. Greedy, 48 new tokens, truncate at first newline.

ar_fim_fair
    Identical, except holes > j are kept as ``pass  # TODO`` at the gold line's
    indentation so the FIM suffix stays syntactically valid. This is a fairness
    repair for ar_fim, not a separate method.

NULL CONTROLS (must be run on the SAME spec; they are what catches a dead pipeline)
    null_gold    -- refill each hole with the gold line. MUST score ~1.0.
    null_delete  -- delete each hole line. MUST score ~0.0.
    null_mutate  -- deterministic semantic mutation of the gold line. MUST
                    DEGRADE WITH k. A flat/high null_mutate is the signature of
                    the docstring-hole bug that produced a prior retraction.

COST
====
``ForwardCostTracker`` is imported from forward_cost.py -- the SAME single
implementation ``generate_evalplus_ar.py`` uses, NOT a reimplementation -- so
diffusion and AR rows are measured by identical instrumentation. The tracker is reset once per task and accumulates across the
k sequential AR calls, so a k-span task is charged for all of its calls.
``tokens_fed`` and ``attended_context_sum`` are the two comparable axes; NFE is
not comparable across families and is reported only as ``forward_passes``.

TERMINATION (assert (e))
========================
``truncated`` (hit the 48-token budget without emitting a newline) and
``abort`` (exception) are recorded per hole and per task, and are reported
SEPARATELY from grading failures. Cost means must never be conditioned on
successful termination -- that was a prior retraction.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

import torch

from forward_cost import ForwardCostTracker

DREAM_MASK_ID = 151666
FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX = 151659, 151660, 151661

ARMS = ("diffusion", "ar_fim", "ar_fim_fair",
        "null_gold", "null_delete", "null_mutate")
AR_ARMS = ("ar_fim", "ar_fim_fair")
NULL_ARMS = ("null_gold", "null_delete", "null_mutate")


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                yield json.loads(line)


def parseable(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except (SyntaxError, ValueError):
        return False


def first_line(text: str) -> tuple[str, bool]:
    """First physical line with newline preserved; flag if no newline was found."""
    idx = text.find("\n")
    if idx == -1:
        return (text + "\n" if text else "\n"), True
    return text[: idx + 1], False


def indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


# --------------------------------------------------------------------------
# null_mutate: deterministic semantic mutation, first applicable rule wins
# --------------------------------------------------------------------------
_MUT_RULES: list[tuple[str, str]] = [
    (r"(?<![<>=!])<=(?!=)", ">="),
    (r"(?<![<>=!])>=(?!=)", "<="),
    (r"(?<![<>=!])==(?!=)", "!="),
    (r"(?<![<>=!])!=(?!=)", "=="),
    (r"(?<![<>=!])<(?![=<])", ">"),
    (r"(?<![<>=!])>(?![=>])", "<"),
    (r"\bTrue\b", "False"),
    (r"\bFalse\b", "True"),
    (r"\band\b", "or"),
    (r"\bor\b", "and"),
    (r"(?<![\w.])\+(?!=)", "-"),
]


def mutate_line(line: str) -> tuple[str, str | None]:
    body = line.rstrip("\n")
    nl = line[len(body):]
    # never mutate inside a string literal / comment-only line
    for pat, rep in _MUT_RULES:
        new, n = re.subn(pat, rep, body, count=1)
        if n:
            return new + nl, pat
    # fall back: bump the first integer literal
    m = re.search(r"(?<![\w.])(\d+)(?![\w.])", body)
    if m:
        v = str(int(m.group(1)) + 1)
        return body[: m.start(1)] + v + body[m.end(1):] + nl, "int+1"
    return line, None  # unmutable; logged so it is never silently counted


# --------------------------------------------------------------------------
# canvas / prompt construction from the frozen segments
# --------------------------------------------------------------------------
def splice(segments, fills: list[str]) -> str:
    out, j = [], 0
    for kind, text in segments:
        if kind == "text":
            out.append(text)
        else:
            out.append(fills[j])
            j += 1
    assert j == len(fills), "fill count does not match hole count"
    return "".join(out)


def run_diffusion(model, tok, row, args, tracker):
    """One canvas, all k mask regions at once. Oracle per-hole lengths."""
    segs = row["segments"]
    lens = row["hole_token_lengths"]
    canvas: list[int] = []
    spans: list[tuple[int, int]] = []
    j = 0
    for kind, text in segs:
        if kind == "text":
            canvas.extend(tok(text, add_special_tokens=False).input_ids)
        else:
            n = lens[j]
            spans.append((len(canvas), len(canvas) + n))
            canvas.extend([DREAM_MASK_ID] * n)
            j += 1
    n_masks = sum(lens)
    steps = n_masks if args.steps <= 0 else min(args.steps, n_masks)

    ids = torch.tensor([canvas], device=model.device, dtype=torch.long)
    tracker.reset()
    tracker.enabled = True
    with torch.inference_mode():
        out = model.diffusion_generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=1,          # masks already in canvas; keeps validator happy
            steps=max(1, steps),
            temperature=args.temperature,
            top_p=args.top_p if args.temperature > 0 else args.top_p,
            alg=args.alg,
            alg_temp=0.0,
            output_history=False,
            return_dict_in_generate=True,
        )
    tracker.enabled = False

    seq = out.sequences[0].tolist()
    fills, left = [], 0
    for a, b in spans:
        ids_hole = [t for t in seq[a:b] if t != DREAM_MASK_ID]
        left += sum(1 for t in seq[a:b] if t == DREAM_MASK_ID)
        fills.append(tok.decode(ids_hole, skip_special_tokens=True))
    return fills, {
        "canvas_len": len(canvas),
        "n_masks": n_masks,
        "steps": max(1, steps),
        "masks_left": left,
        "truncated_holes": 0,
        "aborted_holes": 0,
    }


def run_ar_fim(model, tok, row, args, tracker, *, fair: bool):
    """Sequential left-to-right FIM fill, one generate() call per hole."""
    segs = row["segments"]
    gold = row["gold_lines"]
    k = len(gold)
    fills: list[str] = []
    n_trunc = 0
    eos_ids = [i for i in (tok.convert_tokens_to_ids(t) for t in
               ("<|endoftext|>", "<|fim_pad|>", "<|file_sep|>", "<|repo_name|>"))
               if isinstance(i, int) and i >= 0]

    tracker.reset()
    per_hole = []
    for j in range(k):
        # ---- left context: text + already-generated fills
        pre_parts, h = [], 0
        for kind, text in segs:
            if kind == "text":
                if h <= j:
                    pre_parts.append(text)
            else:
                if h < j:
                    pre_parts.append(fills[h])
                h += 1
                if h > j:
                    break
        prefix_text = "".join(pre_parts)

        # ---- right context: remaining text; later holes deleted or stubbed
        suf_parts, h, seen = [], 0, False
        for kind, text in segs:
            if kind == "hole":
                if h == j:
                    seen = True
                elif seen:
                    if fair:
                        suf_parts.append(indent_of(gold[h]) + "pass  # TODO\n")
                    # else: deleted
                h += 1
            elif seen:
                suf_parts.append(text)
        suffix_text = "".join(suf_parts)

        ids = ([FIM_PREFIX] + tok(prefix_text, add_special_tokens=False).input_ids
               + [FIM_SUFFIX] + tok(suffix_text, add_special_tokens=False).input_ids
               + [FIM_MIDDLE])
        t = torch.tensor([ids], device=model.device, dtype=torch.long)
        plen = t.shape[-1]

        tracker.enabled = True
        with torch.inference_mode():
            out = model.generate(
                t,
                attention_mask=torch.ones_like(t),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=eos_ids or tok.eos_token_id,
                pad_token_id=eos_ids[0] if eos_ids else tok.eos_token_id,
                use_cache=True,
            )
        tracker.enabled = False

        gen = out[0, plen:].tolist()
        for s in eos_ids:
            if s in gen:
                gen = gen[: gen.index(s)]
        raw = tok.decode(gen, skip_special_tokens=True)
        line, no_nl = first_line(raw)
        hit_budget = len(gen) >= args.max_new_tokens and no_nl
        if hit_budget:
            n_trunc += 1
        fills.append(line)
        per_hole.append({"prompt_tokens": plen, "gen_tokens": len(gen),
                         "truncated": bool(hit_budget)})

    return fills, {
        "n_holes": k,
        "truncated_holes": n_trunc,
        "aborted_holes": 0,
        "per_hole": per_hole,
        "fair": fair,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ks", default="", help="comma list; empty = all in spec")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--steps", type=int, default=-1, help="-1 => steps = n_masks")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--alg", default="entropy")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    rows = list(read_jsonl(Path(args.spec)))
    if args.ks:
        keep = {int(x) for x in args.ks.split(",")}
        rows = [r for r in rows if r["k"] in keep]
    rows.sort(key=lambda r: r["spec_id"])
    if args.limit:
        rows = rows[: args.limit]
    assigned = [r for i, r in enumerate(rows) if i % world == rank]

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sol_p = outdir / f"solutions.rank{rank:02d}.jsonl"
    met_p = outdir / f"metrics.rank{rank:02d}.jsonl"
    done: set[str] = set()
    if args.resume and met_p.exists():
        done = {r["spec_id"] for r in read_jsonl(met_p)}
        assigned = [r for r in assigned if r["spec_id"] not in done]
    mode = "a" if args.resume else "w"

    model = tok = None
    tracker = ForwardCostTracker()
    if args.arm not in NULL_ARMS:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        ckpt = str(Path(args.checkpoint).resolve())
        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True,
                                            local_files_only=True)
        dev = torch.device("cuda", local_rank)
        torch.cuda.set_device(dev)
        loader = AutoModelForCausalLM if args.arm in AR_ARMS else AutoModel
        model = loader.from_pretrained(
            ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=True, low_cpu_mem_usage=True,
        ).to(dev).eval()
        model.register_forward_pre_hook(tracker.hook, with_kwargs=True)
        if args.arm in AR_ARMS:
            for name, want in (("<|fim_prefix|>", FIM_PREFIX),
                               ("<|fim_middle|>", FIM_MIDDLE),
                               ("<|fim_suffix|>", FIM_SUFFIX)):
                got = tok.convert_tokens_to_ids(name)
                assert got == want, f"FIM sentinel {name}: expected {want}, got {got}"

    with sol_p.open(mode, encoding="utf-8") as sf, met_p.open(mode, encoding="utf-8") as mf:
        for row in assigned:
            t0 = time.perf_counter()
            fills, info, err = None, None, None
            try:
                if args.arm == "null_gold":
                    fills = list(row["gold_lines"])
                    info = {"truncated_holes": 0, "aborted_holes": 0}
                elif args.arm == "null_delete":
                    fills = ["" for _ in row["gold_lines"]]
                    info = {"truncated_holes": 0, "aborted_holes": 0}
                elif args.arm == "null_mutate":
                    out = [mutate_line(g) for g in row["gold_lines"]]
                    fills = [a for a, _ in out]
                    info = {"truncated_holes": 0, "aborted_holes": 0,
                            "rules": [b for _, b in out],
                            "unmutable_holes": sum(1 for _, b in out if b is None)}
                elif args.arm == "diffusion":
                    fills, info = run_diffusion(model, tok, row, args, tracker)
                else:
                    fills, info = run_ar_fim(model, tok, row, args, tracker,
                                             fair=(args.arm == "ar_fim_fair"))
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                fills = ["" for _ in row["gold_lines"]]
                info = {"truncated_holes": 0,
                        "aborted_holes": len(row["gold_lines"])}

            program = splice(row["segments"], fills)
            cost = tracker.summary() if args.arm not in NULL_ARMS else {
                "forward_passes": 0, "tokens_fed": 0, "attended_context_sum": 0,
                "per_pass_new_tokens": [], "per_pass_attended": []}
            gold = row["gold_lines"]
            em_hole = [f == g for f, g in zip(fills, gold)]
            em_hole_s = [f.strip() == g.strip() for f, g in zip(fills, gold)]
            el = time.perf_counter() - t0

            sf.write(json.dumps({
                "spec_id": row["spec_id"], "task_id": row["task_id"],
                "entry_point": row["entry_point"], "k": row["k"],
                "fills": fills, "solution": program,
            }) + "\n")
            cost_slim = {k2: v for k2, v in cost.items() if not k2.startswith("per_pass")}
            mf.write(json.dumps({
                "spec_id": row["spec_id"], "task_id": row["task_id"], "k": row["k"],
                "arm": args.arm, "rank": rank, "elapsed_seconds": el,
                "parseable": parseable(program),
                "em_all": all(em_hole), "em_all_stripped": all(em_hole_s),
                "em_holes": int(sum(em_hole)), "em_holes_stripped": int(sum(em_hole_s)),
                "n_holes": len(gold),
                "truncated_holes": int(info.get("truncated_holes", 0)),
                "aborted_holes": int(info.get("aborted_holes", 0)),
                "cost": cost_slim, "info": info, "error": err,
                "raw_fills": fills,
            }) + "\n")
            sf.flush(); mf.flush()
            print({"rank": rank, "spec_id": row["spec_id"], "sec": round(el, 2),
                   "err": err}, flush=True)


if __name__ == "__main__":
    main()
