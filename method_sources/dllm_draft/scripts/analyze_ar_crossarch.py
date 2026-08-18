#!/usr/bin/env python
"""Paired AR-control analysis: L20A (.252) vs H20 (.104).

Uses evalplus solutions_eval_results.json from both arms (Qwen2.5-Coder-7B,
same base-continuation protocol, T=0.1 top_p=0.95). Reports paired pass@1,
McNemar flips, and solution-text differences — the same statistics used for
the dLLM cross-node comparison in CROSSNODE_REPRODUCIBILITY.md, so magnitudes
are directly comparable.
"""
from __future__ import annotations
import json
from pathlib import Path
from math import comb

WZC1_L20A = Path("/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/"
                 "outputs/ar_qwen25coder7b_base_252/humaneval/eval_results.json")
H20_MIRROR = Path("/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/"
                  "runs/xnode/ar_control_h20_104/eval_results.json")


def extract(path: Path) -> dict[str, dict]:
    """Return {task_id: {'base': int, 'plus': int, 'solution': str}}."""
    d = json.loads(path.read_text())
    out = {}
    for task_id, entries in d["eval"].items():
        e = entries[0]  # only one sample per task
        base_ok = 1 if (e.get("base_status") == "pass") else 0
        # EvalPlus's reported "plus" pass@1 is CONJUNCTIVE:
        #   plus_ok  <=>  base_status == 'pass' AND plus_status == 'pass'
        # 'plus_status' alone only reports the extra plus-test verdict and can be
        # 'pass' on a task whose base tests failed (1 such task on .252, 2 on .104).
        # Reading it alone inflates plus pass@1 and is NOT the published axis.
        # Cross-check: .104's own recorded pass_at_k['plus'] = 0.5182926829268293
        # == 85/164 (conjunctive), not 87/164 (plus_status alone).
        plus_ok = 1 if (e.get("base_status") == "pass"
                        and e.get("plus_status") == "pass") else 0
        out[task_id] = {
            "base": base_ok,
            "plus": plus_ok,
            "solution": e.get("solution", ""),
        }
    return out


def check_axis(path: Path, rows: dict[str, dict]) -> None:
    """If the file carries evalplus's own pass_at_k, assert our axis matches it.

    This is the guard that would have caught the original bug: reading
    'plus_status' alone gave .5305 while evalplus itself recorded .5183.
    """
    d = json.loads(path.read_text())
    pak = d.get("pass_at_k")
    if not pak:
        print(f"  [axis check] {path.name}: no pass_at_k recorded, skipped")
        return
    n = len(rows)
    for axis in ("base", "plus"):
        ours = sum(r[axis] for r in rows.values()) / n
        theirs = pak[axis]["pass@1"]
        assert abs(ours - theirs) < 1e-9, (
            f"AXIS MISMATCH in {path.name} [{axis}]: ours={ours!r} "
            f"evalplus={theirs!r} -- grading axis is wrong"
        )
    print(f"  [axis check] {path.name}: base+plus match evalplus pass_at_k exactly")


def mcnemar_exact(n01: int, n10: int) -> float:
    """Exact binomial McNemar p-value (two-sided)."""
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = 0.0
    for i in range(0, k + 1):
        p += comb(n, i) * (0.5 ** n)
    p *= 2
    return min(p, 1.0)


def main():
    A = extract(WZC1_L20A)   # .252, L20A
    B = extract(H20_MIRROR)  # .104, H20

    print("# AR control: Qwen2.5-Coder-7B base, T=0.1 top_p=0.95, HumanEval+")
    print("# plus axis = CONJUNCTIVE (base_status AND plus_status), evalplus convention")
    check_axis(WZC1_L20A, A)
    check_axis(H20_MIRROR, B)
    print(f"L20A (.252, wzc1)  n={len(A)}  base_pass={sum(v['base'] for v in A.values())}"
          f"  plus_pass={sum(v['plus'] for v in A.values())}")
    print(f"H20  (.104, zwfy6) n={len(B)}  base_pass={sum(v['base'] for v in B.values())}"
          f"  plus_pass={sum(v['plus'] for v in B.values())}")

    common = sorted(set(A) & set(B))
    assert len(common) == 164, f"expected 164 common tasks, got {len(common)}"

    # pass@1
    a_base = sum(A[t]["base"] for t in common)
    b_base = sum(B[t]["base"] for t in common)
    a_plus = sum(A[t]["plus"] for t in common)
    b_plus = sum(B[t]["plus"] for t in common)
    print()
    print(f"Paired n=164")
    print(f"  HE base pass@1: L20A={a_base}/164={a_base/164:.4f}  "
          f"H20={b_base}/164={b_base/164:.4f}  "
          f"delta={100*(a_base-b_base)/164:+.2f} pt")
    print(f"  HE+   pass@1: L20A={a_plus}/164={a_plus/164:.4f}  "
          f"H20={b_plus}/164={b_plus/164:.4f}  "
          f"delta={100*(a_plus-b_plus)/164:+.2f} pt")

    # McNemar on base
    n01 = sum(1 for t in common if A[t]["base"] == 0 and B[t]["base"] == 1)
    n10 = sum(1 for t in common if A[t]["base"] == 1 and B[t]["base"] == 0)
    p = mcnemar_exact(n01, n10)
    print()
    print(f"  base flips: n01(L20A=0,H20=1)={n01}  n10(L20A=1,H20=0)={n10}  "
          f"total={n01+n10}  exact p={p:.4f}")

    n01p = sum(1 for t in common if A[t]["plus"] == 0 and B[t]["plus"] == 1)
    n10p = sum(1 for t in common if A[t]["plus"] == 1 and B[t]["plus"] == 0)
    pp = mcnemar_exact(n01p, n10p)
    print(f"  plus flips: n01(L20A=0,H20=1)={n01p}  n10(L20A=1,H20=0)={n10p}  "
          f"total={n01p+n10p}  exact p={pp:.4f}")

    # solution text differences
    diff = sum(1 for t in common if A[t]["solution"] != B[t]["solution"])
    print(f"  solution text differs: {diff}/164")


if __name__ == "__main__":
    main()
