#!/usr/bin/env python3
"""Build a DECONTAMINATED variant of the frozen k-span spec.

WHY
===
Both model families may have memorised HumanEval. If so, a pass@1-vs-k slope
could be recall rather than reasoning. This builds a spec whose surface form the
models cannot have seen verbatim, while keeping the task exactly as well-posed:

  1. IDENTIFIER RENAMING -- every local name (function params, locals,
     comprehension vars) is renamed to a semantically-neutral token. Applied via
     Python's own tokenizer over the WHOLE file, so prefix, suffix and gold lines
     stay mutually consistent by construction. Renaming only the gold line would
     make the task unanswerable; renaming inconsistently would make it
     ill-posed. Neither happens here.
  2. DOCSTRING / COMMENT STRIPPING -- the docstring is where the natural-language
     statement of the problem lives, and it is the single strongest memorisation
     cue. Optional (`--keep-docstring`) so the effect can be attributed.

WHAT IS DELIBERATELY *NOT* RENAMED
==================================
  * the entry point (EvalPlus calls it by name -- renaming breaks grading);
  * builtins, imported names, attribute names, and anything in `KEEP`;
  * string/number literals (they carry the semantics the tests check).

WELL-POSEDNESS GATE
===================
Every emitted row must satisfy: the gold-refilled renamed program still PASSES
the official EvalPlus tests. Rows failing that gate are DROPPED and counted, so
a rename that silently breaks a task can never enter the measurement. This is
the analogue of the null_gold control for the decontaminated set.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import io
import json
import keyword
import tokenize
from pathlib import Path

BUILTINS = set(dir(builtins))
KEEP = {"self", "cls", "List", "Dict", "Tuple", "Set", "Optional", "Any",
        "Union", "Callable", "Iterable", "Sequence", "math", "re", "collections",
        "itertools", "functools", "heapq", "string", "np", "numpy", "typing"}

# Neutral, plausible-looking but non-canonical identifiers.
POOL = [
    "qz_a", "qz_b", "qz_c", "qz_d", "qz_e", "qz_f", "qz_g", "qz_h",
    "qz_i", "qz_j", "qz_k", "qz_l", "qz_m", "qz_n", "qz_o", "qz_p",
    "qz_q", "qz_r", "qz_s", "qz_t", "qz_u", "qz_v", "qz_w", "qz_x",
]


def read_jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as h:
        for line in h:
            if line.strip():
                yield json.loads(line)


def collect_renamable(src: str, entry_point: str) -> dict[str, str]:
    """Local identifiers eligible for renaming, in deterministic order."""
    tree = ast.parse(src)
    names: list[str] = []
    seen: set[str] = set()

    def add(n: str) -> None:
        if (n and n not in seen and n != entry_point and n not in BUILTINS
                and n not in KEEP and not keyword.iskeyword(n)
                and not n.startswith("__")):
            seen.add(n)
            names.append(n)

    imported: set[str] = set()
    funcs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                imported.add((a.asname or a.name).split(".")[0])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.add(node.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Load)):
            add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # nested helper functions are fair game, the entry point is not
            if node.name != entry_point:
                add(node.name)

    names = [n for n in names if n not in imported]
    # attribute names must never be renamed; drop anything used as an attribute
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = [n for n in names if n not in attrs]

    if len(names) > len(POOL):
        names = names[: len(POOL)]
    return {n: POOL[i] for i, n in enumerate(sorted(names))}


def rename_lines(src: str, mapping: dict[str, str]) -> list[str]:
    """Rename the WHOLE file and return its lines, plus a per-source-line map.

    Renaming must be done on the complete file: a single gold line is often not
    independently tokenizable (`for x in y:` alone is an EOF-in-multi-line-
    statement), and re-tokenizing fragments is exactly how a "consistent
    renaming" quietly stops being consistent. So we rename once, globally, and
    then locate holes by LINE POSITION, which `untokenize` preserves because we
    only ever substitute NAME tokens (never insert or delete tokens).
    """
    if not mapping:
        return src.splitlines(keepends=True)
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    out, prev_dot = [], False
    for t in toks:
        if t.type == tokenize.NAME and not prev_dot and t.string in mapping:
            out.append(t._replace(string=mapping[t.string]))
        else:
            out.append(t)
        prev_dot = (t.type == tokenize.OP and t.string == ".")
    return tokenize.untokenize(out).splitlines(keepends=True)


def apply_rename(src: str, mapping: dict[str, str]) -> str:
    """Token-level rename: only NAME tokens, never strings/comments/attributes."""
    if not mapping:
        return src
    out: list[tokenize.TokenInfo] = []
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    prev_was_dot = False
    for t in toks:
        if t.type == tokenize.NAME and not prev_was_dot and t.string in mapping:
            out.append(t._replace(string=mapping[t.string]))
        else:
            out.append(t)
        prev_was_dot = (t.type == tokenize.OP and t.string == ".")
    return tokenize.untokenize(out)


def strip_docstrings(src: str) -> tuple[str, dict[int, int]]:
    """Replace every docstring body with one neutral sentence and drop the rest.

    Returns the new source AND an old-line -> new-line map, so hole positions can
    be carried across the edit instead of being re-found by text matching.
    """
    tree = ast.parse(src)
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.Module)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                targets.append(body[0])
    lines = src.splitlines(keepends=True)
    kill: set[int] = set()
    for t in targets:
        for ln in range(t.lineno - 1, (t.end_lineno or t.lineno)):
            kill.add(ln)
        first = lines[t.lineno - 1]
        ind = first[: len(first) - len(first.lstrip())]
        lines[t.lineno - 1] = f'{ind}"""Solve the task."""\n'
        kill.discard(t.lineno - 1)
    new_lines, lmap, j = [], {}, 0
    for i, l in enumerate(lines):
        if i in kill:
            continue
        lmap[i] = j
        new_lines.append(l)
        j += 1
    return "".join(new_lines), lmap


def segments_from(F: str, holes: list[int]):
    FL = F.splitlines(keepends=True)
    segs, cur = [], 0
    for h in holes:
        if h > cur:
            segs.append(("text", "".join(FL[cur:h])))
        segs.append(("hole", FL[h]))
        cur = h + 1
    if cur < len(FL):
        segs.append(("text", "".join(FL[cur:])))
    assert "".join(s for _, s in segs) == F
    return segs


def splice(segments, fills):
    out, j = [], 0
    for kind, text in segments:
        if kind == "text":
            out.append(text)
        else:
            out.append(fills[j]); j += 1
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="models/Dream-Coder-v0-Instruct-7B")
    ap.add_argument("--keep-docstring", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.eval import PASS, untrusted_check
    from evalplus.evaluate import get_groundtruth

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    he = get_human_eval_plus()
    gt = get_groundtruth(he, get_human_eval_plus_hash(), [])

    def gold_passes(task_id, program):
        t, ref = he[task_id], gt[task_id]
        try:
            st, det = untrusted_check(
                "humaneval", program, t["base_input"], t["entry_point"],
                expected=ref["base"], atol=t["atol"], ref_time=ref["base_time"],
                fast_check=False, min_time_limit=1.0, gt_time_limit_factor=4.0)
        except Exception:
            return False
        det = list(det) if det is not None else []
        return st == PASS and det and all(bool(d) for d in det)

    rows = list(read_jsonl(Path(args.spec)))
    kept, dropped = [], {"rename_error": 0, "gate_fail": 0, "line_shift": 0,
                         "no_rename": 0, "adjacency": 0, "docstring": 0}
    for r in rows:
        F = r["reference_file"]
        ep = r["entry_point"]
        try:
            mapping = collect_renamable(F, ep)
            if not mapping:
                dropped["no_rename"] += 1
                continue
            # 1) rename the whole file; line count is preserved because we only
            #    substitute NAME tokens.
            ren = rename_lines(F, mapping)
            if len(ren) != len(F.splitlines(keepends=True)):
                dropped["line_shift"] += 1
                continue
            holes = list(r["hole_line_numbers"])
            G = "".join(ren)
            # 2) optionally strip docstrings, carrying hole positions along
            if not args.keep_docstring:
                G, lmap = strip_docstrings(G)
                if any(h not in lmap for h in holes):
                    dropped["docstring"] += 1   # a hole was inside a docstring
                    continue
                holes = [lmap[h] for h in holes]
            ast.parse(G)
        except Exception:
            dropped["rename_error"] += 1
            continue

        GL = G.splitlines(keepends=True)
        if any(h >= len(GL) for h in holes):
            dropped["line_shift"] += 1
            continue
        if any(b - a < 2 for a, b in zip(holes, holes[1:])):
            dropped["adjacency"] += 1
            continue
        gold_actual = [GL[h] for h in holes]
        if any(not g.strip() for g in gold_actual):
            dropped["line_shift"] += 1
            continue

        segs = segments_from(G, holes)

        # --- well-posedness gate: gold refill of the RENAMED file must pass ---
        if not gold_passes(r["task_id"], splice(segs, gold_actual)):
            dropped["gate_fail"] += 1
            continue

        hole_tok = [len(tok(g, add_special_tokens=False).input_ids)
                    for g in gold_actual]
        kept.append({
            "spec_id": r["spec_id"], "task_id": r["task_id"],
            "entry_point": ep, "k": r["k"],
            "hole_line_numbers": holes, "gold_lines": gold_actual,
            "hole_token_lengths": hole_tok,
            "total_masked_tokens": int(sum(hole_tok)),
            "total_masked_lines": len(holes),
            "reference_file": G, "segments": segs,
            "rename_map": mapping,
            "decontaminated": True,
            "docstring_stripped": not args.keep_docstring,
        })

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as h:
        for r in kept:
            h.write(json.dumps(r) + "\n")

    import collections
    per = collections.Counter(r["k"] for r in kept)
    print(f"input rows {len(rows)}  kept {len(kept)}  dropped {dict(dropped)}")
    print("per-cell n (decontaminated):")
    for k in sorted(per):
        print(f"  k={k}  n={per[k]}")
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
