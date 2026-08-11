#!/usr/bin/env python3
"""Where the proposals live -- one source of truth for cross-proposal paths.

A03 was decided ARCHIVE on 2026-08-11 and physically moved from
`proposal/active/` to `proposal/archive/`. Several A04 analyses import A03's
canonical scorers/nulls from `analyze_1b_knowledge_floor.py`, and A03's own
seed-45 recompute still runs. Rather than hard-code either location in five
places, resolve it here and fail LOUDLY if the directory is in neither.

`canonical_eval_loaders` (load_cb / load_mmlu / paired) was lifted OUT of A03 for
the same reason and needs no lookup -- import it directly.
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
PROPOSAL_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

# archive first: that is where A03 now lives. `active` is retained so a
# not-yet-moved checkout (e.g. the zwfy6 hand-copied proposal tree, which is not
# a git checkout and so does not receive the `git mv`) keeps working.
_A03_CANDIDATES = ("archive", "active")
_A03_DIRNAME = "A03-parametric-vs-external-memory"


def a03_code_dir() -> str:
    """Absolute path to A03's `code/`. Raises if absent -- never returns a guess."""
    tried = []
    for stage in _A03_CANDIDATES:
        p = os.path.join(PROPOSAL_ROOT, stage, _A03_DIRNAME, "code")
        tried.append(p)
        if os.path.isdir(p):
            return p
    raise SystemExit(
        "FATAL: A03's code/ directory is in neither archive/ nor active/. "
        f"Tried: {tried}. A03 holds the canonical scorers/nulls "
        "(analyze_1b_knowledge_floor.py) that this analysis imports rather than "
        "reimplements -- two subagents have already produced spurious "
        "significance by reimplementing a metric. Refusing to continue.")
