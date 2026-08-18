#!/usr/bin/env python3
"""Measure all-masked typed-region collapse versus token length and clock."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scaffold_coder.canvas import TokenRegistry
from scaffold_coder.corruption import (
    GlobalBandSampler,
    HierarchicalBandConfig,
    Rung,
)
from scaffold_coder.ir import (
    Body,
    ForStatement,
    FunctionDefinition,
    IfStatement,
    SimpleStatement,
    WhileStatement,
)
from scaffold_coder.roles import HDR, STMT, MaskRole
from scaffold_coder.serialization import module_from_dict


def length_bin(length: int) -> str:
    if length == 1:
        return "1"
    if length == 2:
        return "2"
    if length <= 4:
        return "3-4"
    if length <= 8:
        return "5-8"
    if length <= 16:
        return "9-16"
    return "17+"


def clock_bin(value: float) -> str:
    index = min(9, int(value * 10))
    return f"{index / 10:.1f}-{(index + 1) / 10:.1f}"


def regions(body: Body, tokenizer):
    for line in body.lines:
        if isinstance(line, SimpleStatement):
            yield (
                line.node_id,
                STMT,
                len(tokenizer.encode(line.text, add_special_tokens=False)),
                line.depth,
            )
        elif isinstance(line, FunctionDefinition):
            yield (
                line.node_id,
                HDR,
                len(tokenizer.encode(line.header, add_special_tokens=False)),
                line.depth,
            )
            yield from regions(line.body, tokenizer)
        elif isinstance(line, IfStatement):
            yield (
                line.node_id,
                HDR,
                len(tokenizer.encode(line.condition, add_special_tokens=False)),
                line.depth,
            )
            yield from regions(line.body, tokenizer)
            for clause in line.elif_clauses:
                yield (
                    clause.node_id,
                    HDR,
                    len(
                        tokenizer.encode(
                            clause.condition, add_special_tokens=False
                        )
                    ),
                    clause.depth,
                )
                yield from regions(clause.body, tokenizer)
            if line.else_body:
                yield from regions(line.else_body, tokenizer)
        elif isinstance(line, ForStatement):
            header = f"{line.target} in {line.iterator}"
            yield (
                line.node_id,
                HDR,
                len(tokenizer.encode(header, add_special_tokens=False)),
                line.depth,
            )
            yield from regions(line.body, tokenizer)
            if line.else_body:
                yield from regions(line.else_body, tokenizer)
        elif isinstance(line, WhileStatement):
            yield (
                line.node_id,
                HDR,
                len(tokenizer.encode(line.condition, add_special_tokens=False)),
                line.depth,
            )
            yield from regions(line.body, tokenizer)
            if line.else_body:
                yield from regions(line.else_body, tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-row", type=int, default=10)
    parser.add_argument(
        "--collapse-mode", choices=("all_mask", "coupled"), default="all_mask"
    )
    parser.add_argument("--collapse-exponent", type=float, default=1.0)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    registry = TokenRegistry.build(tokenizer)
    sampler = GlobalBandSampler(
        registry,
        HierarchicalBandConfig(
            region_collapse_mode=args.collapse_mode,
            region_collapse_exponent=args.collapse_exponent,
        ),
    )
    hdr_id = registry.special_id(HDR)
    stmt_id = registry.special_id(STMT)
    opportunities = collections.Counter()
    collapsed = collections.Counter()
    joint_opportunities = collections.Counter()
    joint_collapsed = collections.Counter()
    leaf_samples = 0

    parquet = pq.ParquetFile(args.parquet)
    columns = ["seq_id", "prompt", "ir_json"]
    for batch in parquet.iter_batches(columns=columns, batch_size=128):
        for row in batch.to_pylist():
            module = module_from_dict(json.loads(row["ir_json"]))
            clean_regions = list(regions(module.body, tokenizer))
            for draw in range(args.samples_per_row):
                seed = int(row["seq_id"]) * 1_000_003 + draw
                t = random.Random(seed ^ 0xC011A95E).random()
                sampled = sampler.sample(
                    module, row["prompt"], seed=seed, t=t
                )
                if sampled.rung is not Rung.LEAF_INFILL:
                    continue
                leaf_samples += 1
                collapsed_nodes = set()
                for input_id, label, role, node_id in zip(
                    sampled.state.input_ids,
                    sampled.state.labels,
                    sampled.state.roles,
                    sampled.state.node_ids,
                    strict=True,
                ):
                    if input_id in {hdr_id, stmt_id}:
                        collapsed_nodes.add(node_id)
                    elif (
                        label == stmt_id
                        and role
                        in {MaskRole.LINE_MODULE, MaskRole.LINE_BODY}
                    ):
                        collapsed_nodes.add(node_id)

                clocks = {
                    int(depth): float(value)
                    for depth, value in sampled.metadata["clocks"].items()
                }
                for node_id, notation, length, depth in clean_regions:
                    lbin = length_bin(length)
                    cbin = clock_bin(clocks.get(depth, 0.0))
                    key = (notation, lbin)
                    joint_key = (notation, lbin, cbin)
                    opportunities[key] += 1
                    joint_opportunities[joint_key] += 1
                    if node_id in collapsed_nodes:
                        collapsed[key] += 1
                        joint_collapsed[joint_key] += 1

    rows = []
    for key, count in sorted(opportunities.items()):
        value = collapsed[key]
        rows.append(
            {
                "notation": key[0],
                "length_bin": key[1],
                "opportunities": count,
                "collapsed": value,
                "rate": value / count,
            }
        )
    joint_rows = []
    for key, count in sorted(joint_opportunities.items()):
        value = joint_collapsed[key]
        joint_rows.append(
            {
                "notation": key[0],
                "length_bin": key[1],
                "clock_bin": key[2],
                "opportunities": count,
                "collapsed": value,
                "rate": value / count,
            }
        )
    report = {
        "collapse_mode": args.collapse_mode,
        "collapse_exponent": args.collapse_exponent,
        "leaf_samples": leaf_samples,
        "by_length": rows,
        "by_length_and_clock": joint_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
