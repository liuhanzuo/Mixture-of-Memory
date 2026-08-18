#!/usr/bin/env python3
"""Cross-node reproducibility analysis for a fixed decoding protocol.

Every arm is graded CENTRALLY with a single evalplus version so the grader is
never an independent variable (the original observation was confounded by
evalplus 0.3.1 on wzc1 vs 0.1.0.dev1 on zwfy6 -- see --crossgrade output).

Reports, per arm pair:
  * pass@1 on base and base+plus, with the axis stated explicitly
  * McNemar discordant cells n01/n10 and an EXACT two-sided binomial p
  * generated-text disagreement count
  * first divergent token index (tokenised with the model tokenizer)
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def load_eval(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text())
    out = {}
    for tid, entries in payload["eval"].items():
        e = entries[0]
        out[tid] = {
            "base": e["base_status"] == "pass",
            "plus": e["plus_status"] == "pass",
            "solution": e["solution"],
        }
    return out, payload["hash"]


def load_raw(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["task_id"]] = row
    return out


def exact_mcnemar(n01: int, n10: int) -> float:
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / 2**n)


def pass1(d: dict, axis: str) -> tuple[int, float]:
    k = sum(1 for v in d.values() if v[axis])
    return k, k / len(d)


def compare(name_a: str, a: dict, name_b: str, b: dict) -> dict:
    assert set(a) == set(b), "task id sets differ"
    res = {"arm_a": name_a, "arm_b": name_b, "n": len(a)}
    for axis in ("base", "plus"):
        ka, pa = pass1(a, axis)
        kb, pb = pass1(b, axis)
        n10 = sum(1 for t in a if a[t][axis] and not b[t][axis])
        n01 = sum(1 for t in a if b[t][axis] and not a[t][axis])
        res[axis] = {
            "pass1_a": round(pa, 4), "k_a": ka,
            "pass1_b": round(pb, 4), "k_b": kb,
            "delta_pt": round((pa - pb) * 100, 2),
            "n10_a_only": n10, "n01_b_only": n01,
            "flips": n01 + n10,
            "exact_mcnemar_p": round(exact_mcnemar(n01, n10), 4),
        }
    res["solution_text_differs"] = sum(
        1 for t in a if a[t]["solution"] != b[t]["solution"]
    )
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="name=path/to/eval_results.json")
    ap.add_argument("--pair", action="append", default=[],
                    help="nameA:nameB")
    ap.add_argument("--raw", action="append", default=[],
                    help="name=path/to/metrics.jsonl for token-level divergence")
    ap.add_argument("--tokenizer")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms, hashes = {}, {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        arms[name], hashes[name] = load_eval(Path(path))
    if len(set(hashes.values())) != 1:
        raise SystemExit(f"evalplus ground-truth hash differs across arms: {hashes}")

    raws = {}
    for spec in args.raw:
        name, path = spec.split("=", 1)
        raws[name] = load_raw(Path(path))

    tok = None
    if args.tokenizer and raws:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    report = {"gt_hash": next(iter(hashes.values())), "pairs": []}
    for spec in args.pair:
        na, nb = spec.split(":", 1)
        entry = compare(na, arms[na], nb, arms[nb])
        if na in raws and nb in raws:
            ra, rb = raws[na], raws[nb]
            shared = sorted(set(ra) & set(rb))
            diff = [t for t in shared if ra[t]["raw_output"] != rb[t]["raw_output"]]
            entry["raw_output_differs"] = len(diff)
            entry["raw_output_n_compared"] = len(shared)
            if tok is not None:
                first_tok = {}
                for t in diff:
                    ta = tok(ra[t]["raw_output"], add_special_tokens=False)["input_ids"]
                    tb = tok(rb[t]["raw_output"], add_special_tokens=False)["input_ids"]
                    i = 0
                    while i < min(len(ta), len(tb)) and ta[i] == tb[i]:
                        i += 1
                    first_tok[t] = i
                if first_tok:
                    vals = sorted(first_tok.values())
                    entry["first_divergent_token_index"] = {
                        "min": vals[0], "median": vals[len(vals) // 2],
                        "max": vals[-1],
                        "n_diverge_at_token_0": sum(1 for v in vals if v == 0),
                    }
                    entry["per_task_first_divergent_token"] = first_tok
        report["pairs"].append(entry)

    Path(args.out).write_text(json.dumps(report, indent=1))
    for p in report["pairs"]:
        print(f"\n=== {p['arm_a']}  vs  {p['arm_b']}  (n={p['n']}) ===")
        for axis in ("base", "plus"):
            d = p[axis]
            print(f"  {axis:5s}: {d['pass1_a']:.4f} ({d['k_a']}) vs "
                  f"{d['pass1_b']:.4f} ({d['k_b']})  delta={d['delta_pt']:+.2f}pt  "
                  f"flips={d['flips']} (n10={d['n10_a_only']}, n01={d['n01_b_only']})  "
                  f"exact_p={d['exact_mcnemar_p']}")
        print(f"  solution text differs: {p['solution_text_differs']}/{p['n']}")
        if "raw_output_differs" in p:
            print(f"  raw_output differs: {p['raw_output_differs']}/"
                  f"{p['raw_output_n_compared']}")
        if "first_divergent_token_index" in p:
            print(f"  first divergent token idx: {p['first_divergent_token_index']}")


if __name__ == "__main__":
    main()
