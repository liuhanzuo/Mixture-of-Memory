#!/usr/bin/env python3
"""Freeze the k-span (multi-region) code-infilling hole spec.

WHY THIS FILE EXISTS
====================
An earlier pilot of this ladder treated the ``L{k}`` suffix in
``SingleLineInfilling/HumanEval/N/L{k}`` as a *file* line index. It is not: it
indexes lines WITHIN THE SOLUTION BODY. That mistake put 379/379 holes inside
docstrings and produced a beautiful, entirely meaningless flat ladder. It was
caught only because a mutated-gold null control *passed* instead of degrading.

So this builder does not infer the hole line from ``L{k}`` at all. It recovers
it by BYTE-EXACT reconstruction: a row's hole line ``i`` in reference file ``F``
is admitted only if

    F_lines[i] == row['canonical_solution']
    and ''.join(F_lines[:i]) == row['prompt']
    and ''.join(F_lines[i+1:]) == row['suffix']

i.e. the row is *provably* the "blank out line i of F" task. Anything that does
not satisfy this is dropped, not guessed.

Five hard asserts (all fatal, per the spec contract):
  (a) every hole line lies OUTSIDE any docstring / bare string expression
  (b) holes are pairwise non-adjacent (min gap 2 in line numbers)
  (c) the reconstructed reference file ``ast.parse``-es
  (d) per-cell n is logged
  (e) truncation/abort counts are logged SEPARATELY from failures
      -- (e) is enforced in the runners; this file emits the schema field
      ``total_masked_tokens`` they need and asserts it is > 0.

Emits, per row: {task_id, k, hole_line_numbers, gold_lines, total_masked_tokens}
plus the reference file and its sha256 so every arm consumes byte-identical
holes on both physical disks.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import sys
from pathlib import Path

POOL_REL = "data/infilling/HumanEval-SingleLineInfilling.jsonl"


# --------------------------------------------------------------------------
# reference-file recovery
# --------------------------------------------------------------------------
def load_pool(path: Path) -> dict[str, dict[int, dict]]:
    by: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    with path.open(encoding="utf-8") as h:
        for line in h:
            if not line.strip():
                continue
            r = json.loads(line)
            parts = r["task_id"].split("/")
            base = "/".join(parts[1:3])          # HumanEval/N
            body_idx = int(parts[3][1:])         # L{k} -> k  (BODY index, not file)
            by[base][body_idx] = r
    return by


def reference_file(rows: dict[int, dict]) -> tuple[str, dict]:
    """Majority reconstruction ``prompt + canonical_solution + suffix``.

    123/164 base tasks disagree across their own rows by exactly one blank line
    (the ``L0`` row carries a trailing ``'\\n'`` the others lack). Majority vote
    with a deterministic tie-break (shortest, then lexicographic) makes the
    choice reproducible instead of order-dependent.
    """
    c: collections.Counter = collections.Counter()
    for r in rows.values():
        c[r["prompt"] + r["canonical_solution"] + r["suffix"]] += 1
    ranked = sorted(c.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))
    F = ranked[0][0]
    return F, {"n_variants": len(c), "majority_votes": ranked[0][1], "n_rows": len(rows)}


def admit(F: str, row: dict) -> int | None:
    """Byte-exact hole-line recovery. Returns the file line index, or None."""
    FL = F.splitlines(keepends=True)
    hits = [
        i
        for i in range(len(FL))
        if FL[i] == row["canonical_solution"]
        and "".join(FL[:i]) == row["prompt"]
        and "".join(FL[i + 1:]) == row["suffix"]
    ]
    if len(hits) == 1:
        return hits[0]
    return None  # 0 hits => this row belongs to the other blank-line variant
                 # >1 hits => genuinely ambiguous (duplicate line); refuse to guess


# --------------------------------------------------------------------------
# assert (a): docstring lines
# --------------------------------------------------------------------------
def docstring_lines(F: str) -> set[int]:
    """0-based file line numbers covered by any docstring / bare string expr."""
    bad: set[int] = set()
    for node in ast.walk(ast.parse(F)):
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            for ln in range(node.lineno - 1, (node.end_lineno or node.lineno)):
                bad.add(ln)
    return bad


# --------------------------------------------------------------------------
# hole selection
# --------------------------------------------------------------------------
def greedy_nonadjacent(avail: list[int]) -> list[int]:
    """Max independent set on a path == leftmost-first greedy. Optimal here."""
    keep: list[int] = []
    last = -(10**9)
    for x in avail:
        if x - last >= 2:
            keep.append(x)
            last = x
    return keep


def spread(pool: list[int], k: int) -> list[int]:
    """Deterministically take k of the pool, spread as evenly as possible."""
    m = len(pool)
    if k > m:
        raise ValueError(f"cannot take {k} from {m}")
    if k == 1:
        return [pool[m // 2]]
    idx = sorted({round(j * (m - 1) / (k - 1)) for j in range(k)})
    while len(idx) < k:  # rounding collision guard
        for cand in range(m):
            if cand not in idx:
                idx = sorted(idx + [cand])
                break
    return [pool[i] for i in idx[:k]]


def contiguous_runs(avail: list[int], run_len: int, n_runs: int) -> list[list[int]] | None:
    """For the topology control: n_runs disjoint runs of run_len consecutive
    admitted lines, with >=1 clean line between consecutive runs."""
    runs: list[list[int]] = []
    s = set(avail)
    i = 0
    ordered = sorted(avail)
    while i < len(ordered):
        start = ordered[i]
        run = [start + j for j in range(run_len)]
        if all(x in s for x in run) and (not runs or start - runs[-1][-1] >= 2):
            runs.append(run)
            if len(runs) == n_runs:
                return runs
            i = ordered.index(run[-1]) + 1 if run[-1] in ordered else i + 1
            while i < len(ordered) and ordered[i] <= run[-1] + 1:
                i += 1
            continue
        i += 1
    return None


# --------------------------------------------------------------------------
# tokenisation (piecewise, so canvas construction is exactly reproducible)
# --------------------------------------------------------------------------
def segment(F: str, holes: list[int]) -> list[tuple[str, str]]:
    """Split F into alternating ('text', s) / ('hole', gold_line) segments."""
    FL = F.splitlines(keepends=True)
    segs: list[tuple[str, str]] = []
    cur = 0
    for h in holes:
        if h > cur:
            segs.append(("text", "".join(FL[cur:h])))
        segs.append(("hole", FL[h]))
        cur = h + 1
    if cur < len(FL):
        segs.append(("text", "".join(FL[cur:])))
    assert "".join(s for _, s in segs) == F, "segmentation is not lossless"
    return segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="data/kspan/kspan_spec_v1.jsonl")
    ap.add_argument("--topology-out", default="data/kspan/topology_spec_v1.jsonl")
    ap.add_argument("--ks", default="1,2,3,4")
    ap.add_argument(
        "--tokenizer",
        default="models/Dream-Coder-v0-Instruct-7B",
        help="oracle hole lengths are defined by the DIFFUSION tokenizer",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    pool_path = root / POOL_REL
    if not pool_path.exists():
        print(f"FATAL: pool not found: {pool_path}", file=sys.stderr)
        return 2

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(root / args.tokenizer), trust_remote_code=True)

    by = load_pool(pool_path)
    ks = [int(x) for x in args.ks.split(",")]

    admission = collections.Counter()
    per_task: dict[str, dict] = {}
    n_parse_fail = 0

    for base, rows in sorted(by.items(), key=lambda kv: int(kv[0].split("/")[1])):
        F, meta = reference_file(rows)

        # --- assert (c): reference file parses ---
        try:
            ast.parse(F)
        except SyntaxError as exc:
            n_parse_fail += 1
            print(f"ASSERT-C FAIL {base}: reference file does not parse: {exc}",
                  file=sys.stderr)
            continue

        doc = docstring_lines(F)
        FL = F.splitlines(keepends=True)

        avail: list[int] = []
        for body_idx, row in sorted(rows.items()):
            i = admit(F, row)
            if i is None:
                admission["dropped_not_byte_exact"] += 1
                continue
            # --- assert (a): hole outside docstring ---
            if i in doc:
                admission["dropped_docstring"] += 1
                continue
            if not FL[i].strip():
                admission["dropped_blank_line"] += 1
                continue
            admission["admitted"] += 1
            avail.append(i)

        avail = sorted(set(avail))
        if not avail:
            continue
        per_task[base] = {
            "reference_file": F,
            "sha256": hashlib.sha256(F.encode()).hexdigest(),
            "avail": avail,
            "nonadj_pool": greedy_nonadjacent(avail),
            "entry_point": next(iter(rows.values()))["entry_point"],
            "variant_meta": meta,
        }

    if n_parse_fail:
        print(f"FATAL assert (c): {n_parse_fail} reference files did not parse",
              file=sys.stderr)
        return 3

    # ------------------------------------------------------------------ main ladder
    out_rows: list[dict] = []
    per_cell: collections.Counter = collections.Counter()
    for base, t in per_task.items():
        pool = t["nonadj_pool"]
        for k in ks:
            if len(pool) < k:
                continue
            holes = sorted(spread(pool, k))

            # --- assert (b): pairwise non-adjacent ---
            for a, b in zip(holes, holes[1:]):
                assert b - a >= 2, f"assert (b) FAIL {base} k={k}: adjacent holes {holes}"
            # --- assert (a) again, post-selection ---
            doc = docstring_lines(t["reference_file"])
            for h in holes:
                assert h not in doc, f"assert (a) FAIL {base} k={k}: hole {h} in docstring"

            FL = t["reference_file"].splitlines(keepends=True)
            gold = [FL[h] for h in holes]
            hole_tok = [len(tok(g, add_special_tokens=False).input_ids) for g in gold]
            total = int(sum(hole_tok))
            assert total > 0, f"assert (e-schema) FAIL {base} k={k}: 0 masked tokens"

            out_rows.append({
                "spec_id": f"{base}/K{k}",
                "task_id": base,
                "entry_point": t["entry_point"],
                "k": k,
                "hole_line_numbers": holes,
                "gold_lines": gold,
                "hole_token_lengths": hole_tok,
                "total_masked_tokens": total,
                "total_masked_lines": len(holes),
                "reference_file": t["reference_file"],
                "reference_sha256": t["sha256"],
                "segments": segment(t["reference_file"], holes),
            })
            per_cell[k] += 1

    # ------------------------------------------------- topology-vs-length control
    # total masked LINES held at 4; k in {1,2,4}
    topo_rows: list[dict] = []
    topo_cell: collections.Counter = collections.Counter()
    TOPO = {1: (4, 1), 2: (2, 2), 4: (1, 4)}
    topo_eligible = []
    for base, t in per_task.items():
        ok = {}
        for k, (run_len, n_runs) in TOPO.items():
            runs = contiguous_runs(t["avail"], run_len, n_runs)
            if runs is not None:
                ok[k] = runs
        if len(ok) == len(TOPO):          # only tasks that support ALL THREE
            topo_eligible.append(base)
            for k, runs in ok.items():
                holes = sorted(x for r in runs for x in r)
                assert len(holes) == 4, f"topology FAIL {base} k={k}: {holes}"
                doc = docstring_lines(t["reference_file"])
                for h in holes:
                    assert h not in doc, f"topology assert (a) FAIL {base} k={k} h={h}"
                # non-adjacency here applies BETWEEN runs, not within
                for r1, r2 in zip(runs, runs[1:]):
                    assert r2[0] - r1[-1] >= 2, f"topology run gap FAIL {base} k={k}"
                FL = t["reference_file"].splitlines(keepends=True)
                gold = [FL[h] for h in holes]
                hole_tok = [len(tok(g, add_special_tokens=False).input_ids) for g in gold]
                topo_rows.append({
                    "spec_id": f"{base}/T{k}",
                    "task_id": base,
                    "entry_point": t["entry_point"],
                    "k": k,
                    "runs": runs,
                    "hole_line_numbers": holes,
                    "gold_lines": gold,
                    "hole_token_lengths": hole_tok,
                    "total_masked_tokens": int(sum(hole_tok)),
                    "total_masked_lines": 4,
                    "reference_file": t["reference_file"],
                    "reference_sha256": t["sha256"],
                    "segments": segment(t["reference_file"], holes),
                })
                topo_cell[k] += 1

    outp = root / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as h:
        for r in out_rows:
            h.write(json.dumps(r) + "\n")
    topop = root / args.topology_out
    with topop.open("w", encoding="utf-8") as h:
        for r in topo_rows:
            h.write(json.dumps(r) + "\n")

    # --- assert (d): per-cell n logged ---
    print("=" * 70)
    print("ADMISSION (row level, pool n=%d)" % sum(len(v) for v in by.values()))
    for kk, vv in sorted(admission.items()):
        print(f"  {kk:28s} {vv}")
    print()
    print("MAIN LADDER per-cell n  [assert (d)]")
    for k in ks:
        print(f"  k={k}  n={per_cell[k]}")
    print()
    print("TOPOLOGY control per-cell n (total masked LINES == 4)  [assert (d)]")
    for k in sorted(TOPO):
        print(f"  k={k}  n={topo_cell[k]}  ({TOPO[k][1]} run(s) x {TOPO[k][0]} line(s))")
    print(f"  tasks supporting all three topologies: {len(topo_eligible)}")
    print()
    tokens_by_k = collections.defaultdict(list)
    for r in out_rows:
        tokens_by_k[r["k"]].append(r["total_masked_tokens"])
    print("masked-token budget per cell (mean/min/max)")
    for k in ks:
        v = tokens_by_k[k]
        if v:
            print(f"  k={k}  mean={sum(v)/len(v):.1f}  min={min(v)}  max={max(v)}")
    print()
    print(f"wrote {len(out_rows)} rows -> {outp}")
    print(f"wrote {len(topo_rows)} rows -> {topop}")
    print("spec sha256:", hashlib.sha256(outp.read_bytes()).hexdigest()[:16])
    print("topo sha256:", hashlib.sha256(topop.read_bytes()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
