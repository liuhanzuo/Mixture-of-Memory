#!/usr/bin/env python3
"""k-span (multi-region) infilling: NON-ORACLE diffusion arm via DreamOn.

This closes the scope hole in Retraction 6: the surviving `kspan_diffusion` arm
gives Dream-Coder-v0-Instruct-7B the oracle per-hole token length, while
AR-FIM has to stop on its own. Reviewer's fair question: does the surviving
"AR degrades ~2x faster than diffusion in k" claim survive when diffusion also
has to pick its own span length?

This script re-scores the frozen `data/kspan/kspan_spec_v1.jsonl` with:

  DreamOn-v0-7B, `infilling_with_expansion` (native variable-length API).
  Sequential left-to-right, one call per hole (matches `ar_fim`: holes < j
  filled with the model's own text; holes > j deleted from the suffix).

DreamOn's native infilling only accepts ONE contiguous mask block between
prefix and suffix, so multi-hole must be sequentialised. This is the same
approach as `ar_fim`, applied to the diffusion side. The output for each hole
is the model's decoded mask region up to the first newline (matches ar_fim's
first_line() rule so hole-count == region-count).

DreamOn config, aligned to `runs/kspan_diffusion` sampling policy:
  temperature=0, top_p=0.95, alg=entropy, alg_temp=0.0,
  min_gen_len=4, max_gen_len=64, steps=128,
  delete_eos_token=True, pad_eos_to_right=True, batch_size=1,
  pad_to_max_len=False, max_prompt_len=1024, max_tokens=2048.

Records per-hole `generated_tokens`, `expand_events`, and whether the
DreamOn sampler CRASHED or produced empty output at that hole (an anticipated
failure mode: DreamOn's variable-length sampler is designed for single-region
infilling and may misbehave when the target span is short/multi-line).

Cost is instrumented with the SAME `ForwardCostTracker` used by
`generate_kspan.py` so the cost axes are directly comparable.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

# Bring in vendor/DreamOn as a source of MDMGenerator
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "vendor" / "DreamOn"))

from forward_cost import ForwardCostTracker  # noqa: E402
from eval.generator import MDMGenerator, MDMGeneratorArgs  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402


class HFTokenizerWrapper:
    """Minimal duplicate of vendor/DreamOn/eval/evaluate.HFTokenizerWrapper.

    Inlined here so we do not have to import `eval.evaluate`, which pulls in
    `datasets`, `human_eval_infilling`, etc. We only need the four methods
    (mask_id / expand_id / eos_id / bos_id attrs + encode / decode).
    """

    def __init__(self, hf_tokenizer) -> None:
        self.tokenizer = hf_tokenizer
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.mask_id = self.tokenizer.mask_token_id
        # DreamOn added `<|expand|>` at id 151667 (see added_tokens.json)
        self.expand_id = 151667

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        ids = list(self.tokenizer.encode(s))
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, tokens, **kwargs):
        return self.tokenizer.decode(tokens, **kwargs)


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


def build_generator(model, tok, args, tracker):
    """DreamOn wants a config-like object with attribute access.

    `MDMGeneratorArgs` (@dataclass) does NOT declare `delete_eos_token`, but
    `MDMGenerator.__init__` reads `cfg.delete_eos_token`. We synthesize the
    missing attributes by subclassing to add them, so the object has every
    attribute the generator reads.
    """
    from dataclasses import dataclass, field

    @dataclass
    class _Cfg(MDMGeneratorArgs):
        delete_eos_token: bool = True

    cfg = _Cfg(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=None,
        show_progress=False,
        dtype="bf16",
        device="cuda",
        max_tokens=args.max_tokens,
        min_gen_len=args.min_gen_len,
        max_prompt_len=args.max_prompt_len,
        max_gen_len=args.max_gen_len,
        pad_to_max_len=False,
        pad_eos_to_right=True,
        batch_size=1,
        eps=1e-3,
        steps=args.steps,
        alg="entropy",
        alg_temp=0.0,
        delete_eos_token=True,
    )
    wrapped_tok = HFTokenizerWrapper(tok)
    # Sanity: expand_id and mask_id must be valid
    assert wrapped_tok.mask_id is not None, "tokenizer must have mask_token_id"
    assert wrapped_tok.expand_id == 151667, f"expand_id expected 151667, got {wrapped_tok.expand_id}"
    gen = MDMGenerator(cfg, model, wrapped_tok)
    return gen, wrapped_tok


def run_dreamon_nonoracle(gen, wrapped_tok, model, row, args, tracker):
    """Sequential left-to-right non-oracle fill via DreamOn native infilling.

    At hole j: prefix = text_before_j + already-generated fills (h<j);
               suffix = text_after_j; later holes (h>j) DELETED (matches ar_fim).
    """
    segs = row["segments"]
    gold = row["gold_lines"]
    k = len(gold)
    fills: list[str] = []
    n_trunc = 0
    n_abort = 0
    per_hole = []

    tracker.reset()

    for j in range(k):
        # left context: text + previously filled holes
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

        # right context: remaining text; later holes DELETED
        suf_parts, h, seen = [], 0, False
        for kind, text in segs:
            if kind == "hole":
                if h == j:
                    seen = True
                # else: deleted (both h<j never happens since text between them is added; h>j deleted)
                h += 1
            elif seen:
                suf_parts.append(text)
        suffix_text = "".join(suf_parts)

        # Run DreamOn native infilling_with_expansion on one hole
        gen_tokens = None
        expand_events = 0
        error_str = None
        raw = ""
        tracker.enabled = True
        t0 = time.perf_counter()
        try:
            outs = gen.infilling_with_expansion([prefix_text], [suffix_text])
            raw = outs[0] if outs else ""
        except Exception as exc:  # noqa: BLE001
            error_str = f"{type(exc).__name__}: {exc}"
            n_abort += 1
        finally:
            tracker.enabled = False
        dt = time.perf_counter() - t0

        # Post-process: keep only up to the first newline, matching ar_fim.
        # If DreamOn produced no newline within max_gen_len, count as truncated
        # but still take what we have.
        if error_str is not None or not raw:
            fills.append("")
            per_hole.append({
                "hole_index": j,
                "raw": raw,
                "line": "",
                "gen_tokens": gen_tokens,
                "expand_events": expand_events,
                "elapsed_seconds": dt,
                "truncated": False,
                "error": error_str,
                "empty_output": True,
            })
            continue

        line, no_nl = first_line(raw)
        # DreamOn's max_gen_len is `args.max_gen_len`; treat "no newline" as truncated
        if no_nl:
            n_trunc += 1
        fills.append(line)
        per_hole.append({
            "hole_index": j,
            "raw": raw,
            "line": line,
            "gen_tokens": None,  # DreamOn doesn't expose token count from decoded str
            "expand_events": expand_events,
            "elapsed_seconds": dt,
            "truncated": bool(no_nl),
            "error": None,
            "empty_output": False,
        })

    info = {
        "n_holes": k,
        "truncated_holes": n_trunc,
        "aborted_holes": n_abort,
        "per_hole": per_hole,
        "fair": False,
    }
    return fills, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ks", default="", help="comma list; empty = all in spec")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-gen-len", type=int, default=4)
    ap.add_argument("--max-gen-len", type=int, default=64)
    ap.add_argument("--max-prompt-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=128)
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

    ckpt = str(Path(args.checkpoint).resolve())
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, local_files_only=True)
    dev = torch.device("cuda", local_rank)
    torch.cuda.set_device(dev)
    model = AutoModel.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True,
    ).to(dev).eval()

    tracker = ForwardCostTracker()
    model.register_forward_pre_hook(tracker.hook, with_kwargs=True)

    gen, wrapped_tok = build_generator(model, tok, args, tracker)

    print(f"[rank={rank}] assigned {len(assigned)} spec rows; "
          f"mask_id={wrapped_tok.mask_id} expand_id={wrapped_tok.expand_id}",
          flush=True)

    with sol_p.open(mode, encoding="utf-8") as sf, met_p.open(mode, encoding="utf-8") as mf:
        for row in assigned:
            t0 = time.perf_counter()
            fills, info, err = None, None, None
            try:
                fills, info = run_dreamon_nonoracle(gen, wrapped_tok, model, row, args, tracker)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                fills = ["" for _ in row["gold_lines"]]
                info = {"truncated_holes": 0,
                        "aborted_holes": len(row["gold_lines"]),
                        "traceback": traceback.format_exc()}

            program = splice(row["segments"], fills)
            cost = tracker.summary()
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
                "arm": "dreamon_nonoracle", "rank": rank, "elapsed_seconds": el,
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
            print({"rank": rank, "spec_id": row["spec_id"], "k": row["k"],
                   "sec": round(el, 2), "trunc": info.get("truncated_holes", 0),
                   "abort": info.get("aborted_holes", 0), "err": err}, flush=True)


if __name__ == "__main__":
    main()
