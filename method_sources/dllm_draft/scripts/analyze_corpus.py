#!/usr/bin/env python3
"""AST and v0-grammar coverage statistics for a parquet code corpus."""

from __future__ import annotations

import argparse
import ast
import collections
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from scaffold_coder.errors import UnsupportedSyntaxError
from scaffold_coder.ir import iter_lines
from scaffold_coder.parser import parse_source
from scaffold_coder.renderer import render_module


COMPOUND_TYPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.Try,
    ast.With,
    ast.AsyncWith,
)
if hasattr(ast, "TryStar"):
    COMPOUND_TYPES = (*COMPOUND_TYPES, ast.TryStar)
if hasattr(ast, "Match"):
    COMPOUND_TYPES = (*COMPOUND_TYPES, ast.Match)


def is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    if not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
        return False
    left, right = test.left, test.comparators[0]

    def is_name(value: ast.AST) -> bool:
        return isinstance(value, ast.Name) and value.id == "__name__"

    def is_main(value: ast.AST) -> bool:
        return (
            isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and value.value == "__main__"
        )

    return (is_name(left) and is_main(right)) or (
        is_main(left) and is_name(right)
    )


def compound_depth(node: ast.AST, depth: int = 0) -> int:
    maximum = depth
    for child in ast.iter_child_nodes(node):
        child_depth = depth + 1 if isinstance(child, COMPOUND_TYPES) else depth
        maximum = max(maximum, compound_depth(child, child_depth))
    return maximum


def unsupported_category(message: str) -> str:
    lowered = message.lower()
    categories = [
        ("decorator", "decorator"),
        ("async function", "async_function"),
        ("async for", "async_for"),
        ("asyncwith", "async_with"),
        ("classdef", "class"),
        ("trystar", "try_star"),
        ("try", "try"),
        ("with", "with"),
        ("match", "match"),
        ("single physical line", "multiline_leaf_or_header"),
        ("unregistered compound", "unregistered_compound"),
    ]
    for needle, category in categories:
        if needle in lowered:
            return category
    return message[:160]


def analyze_one(item: tuple[int, str]) -> dict[str, Any]:
    seq_id, code = item
    result: dict[str, Any] = {
        "seq_id": seq_id,
        "chars": len(code),
        "physical_lines": len(code.splitlines()),
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except Exception as exc:
        result["ast_ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["ast_ok"] = True
    counts = collections.Counter(type(node).__name__ for node in ast.walk(tree))
    result["constructs"] = {
        name: counts.get(name, 0)
        for name in (
            "FunctionDef",
            "AsyncFunctionDef",
            "ClassDef",
            "For",
            "AsyncFor",
            "While",
            "If",
            "Try",
            "TryStar",
            "With",
            "AsyncWith",
            "Match",
        )
    }
    result["decorated_defs"] = sum(
        bool(node.decorator_list)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    result["docstrings"] = sum(
        ast.get_docstring(node, clean=False) is not None
        for node in ast.walk(tree)
        if isinstance(
            node,
            (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        )
    )
    result["main_guards"] = sum(is_main_guard(node) for node in ast.walk(tree))
    result["has_decorated_def"] = result["decorated_defs"] > 0
    result["has_docstring"] = result["docstrings"] > 0
    result["has_main_guard"] = result["main_guards"] > 0
    result["compound_depth"] = compound_depth(tree)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            module = parse_source(code)
            normalized = render_module(module)
            ast.parse(normalized)
        result["v0_ok"] = True
        result["normalized_chars"] = len(normalized)
        result["normalized_lines"] = len(normalized.splitlines())
        result["ir_lines"] = sum(1 for _ in iter_lines(module.body))
    except UnsupportedSyntaxError as exc:
        result["v0_ok"] = False
        result["v0_error"] = unsupported_category(str(exc))
    except Exception as exc:
        result["v0_ok"] = False
        result["v0_error"] = f"internal:{type(exc).__name__}:{exc}"[:240]
    return result


def iter_parquet(path: str, code_key: str, id_key: str) -> Iterable[tuple[int, str]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=[id_key, code_key], batch_size=2048):
        ids = batch.column(id_key).to_pylist()
        codes = batch.column(code_key).to_pylist()
        yield from zip(ids, codes, strict=True)


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(results: Iterable[dict[str, Any]], total_hint: int) -> dict[str, Any]:
    started = time.time()
    total = ast_ok = v0_ok = 0
    constructs: collections.Counter[str] = collections.Counter()
    errors: collections.Counter[str] = collections.Counter()
    depths: collections.Counter[int] = collections.Counter()
    physical_lines: list[int] = []
    normalized_lines: list[int] = []
    chars: list[int] = []
    normalized_chars: list[int] = []
    decorated_defs = docstrings = main_guards = 0
    samples_with_decorated_defs = samples_with_docstrings = samples_with_main_guards = 0
    parse_examples: list[dict[str, Any]] = []
    v0_examples: dict[str, list[int]] = collections.defaultdict(list)

    for result in results:
        total += 1
        chars.append(result["chars"])
        physical_lines.append(result["physical_lines"])
        if result.get("ast_ok"):
            ast_ok += 1
            constructs.update(result["constructs"])
            decorated_defs += result["decorated_defs"]
            docstrings += result["docstrings"]
            main_guards += result["main_guards"]
            samples_with_decorated_defs += int(result["has_decorated_def"])
            samples_with_docstrings += int(result["has_docstring"])
            samples_with_main_guards += int(result["has_main_guard"])
            depths[result["compound_depth"]] += 1
        elif len(parse_examples) < 20:
            parse_examples.append(
                {"seq_id": result["seq_id"], "error": result["error"]}
            )

        if result.get("v0_ok"):
            v0_ok += 1
            normalized_lines.append(result["normalized_lines"])
            normalized_chars.append(result["normalized_chars"])
        elif result.get("ast_ok"):
            category = result.get("v0_error", "unknown")
            errors[category] += 1
            if len(v0_examples[category]) < 5:
                v0_examples[category].append(result["seq_id"])

        if total % 10_000 == 0:
            elapsed = time.time() - started
            rate = total / elapsed if elapsed else 0
            print(
                f"processed={total}/{total_hint} rate={rate:.1f}/s "
                f"ast_ok={ast_ok/total:.3%} v0_ok={v0_ok/total:.3%}",
                file=sys.stderr,
                flush=True,
            )

    def distribution(values: list[int]) -> dict[str, Any]:
        return {
            "count": len(values),
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values) if values else None,
        }

    return {
        "total": total,
        "ast_ok": ast_ok,
        "ast_ok_fraction": ast_ok / total if total else None,
        "v0_ok": v0_ok,
        "v0_ok_fraction_total": v0_ok / total if total else None,
        "v0_ok_fraction_ast_valid": v0_ok / ast_ok if ast_ok else None,
        "construct_counts": dict(constructs.most_common()),
        "unsupported_counts": dict(errors.most_common()),
        "compound_depth_histogram": {
            str(key): value for key, value in sorted(depths.items())
        },
        "decorated_defs": decorated_defs,
        "docstrings": docstrings,
        "main_guards": main_guards,
        "samples_with_decorated_defs": samples_with_decorated_defs,
        "samples_with_docstrings": samples_with_docstrings,
        "samples_with_main_guards": samples_with_main_guards,
        "physical_lines": distribution(physical_lines),
        "normalized_lines": distribution(normalized_lines),
        "characters": distribution(chars),
        "normalized_characters": distribution(normalized_chars),
        "parse_error_examples": parse_examples,
        "unsupported_example_seq_ids": dict(v0_examples),
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--code-key", default="code")
    parser.add_argument("--id-key", default="seq_id")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=128)
    args = parser.parse_args()

    metadata = pq.ParquetFile(args.parquet).metadata
    total_hint = metadata.num_rows
    items = iter_parquet(args.parquet, args.code_key, args.id_key)
    if args.workers == 1:
        analyzed = map(analyze_one, items)
        summary = summarize(analyzed, total_hint)
    else:
        context = mp.get_context("fork")
        with context.Pool(args.workers) as pool:
            analyzed = pool.imap_unordered(
                analyze_one, items, chunksize=args.chunksize
            )
            summary = summarize(analyzed, total_hint)

    summary["parquet"] = str(Path(args.parquet).resolve())
    summary["workers"] = args.workers
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
