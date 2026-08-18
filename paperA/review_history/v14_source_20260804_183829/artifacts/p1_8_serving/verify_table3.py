#!/usr/bin/env python3
"""Verify and print Paper A Table 3 from the P1.8 aggregate.

This performs no model execution.  It checks that the released aggregate has
the complete 3-length x 2-placement x 4-generation grid and prints the rounded
break-even values used by ``sections/tab_serving_crossover.tex``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AGGREGATE = ROOT / "p1_8_serving_aggregate.json"
LENGTHS = ("32k", "128k", "1M")
TIERS = ("cpu", "gpu")
GENERATIONS = ("1", "32", "128", "512")


def format_qstar(value: float) -> str:
    return r"\infty" if math.isinf(value) else f"{value:.1f}"


def main() -> None:
    data = json.loads(AGGREGATE.read_text())
    assert data["n_files"] == 18
    assert set(data["cells"]) == {
        f"{length}|{tier}" for length in LENGTHS for tier in TIERS
    }

    print("Store & G=1 & G=32 & G=128 & G=512")
    n_cells = 0
    for length in LENGTHS:
        for tier in TIERS:
            cell = data["cells"][f"{length}|{tier}"]
            assert cell["n_procs"] == 3
            assert set(cell["crossover"]) == set(GENERATIONS)
            values = [
                format_qstar(cell["crossover"][generation]["break_even_Q"])
                for generation in GENERATIONS
            ]
            n_cells += len(values)
            print(f"{length}, {tier}: " + " & ".join(values))

    assert n_cells == 24
    print("PASS: complete 24-cell Table 3 grid.")


if __name__ == "__main__":
    main()
