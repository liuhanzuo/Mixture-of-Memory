#!/usr/bin/env python3
"""Carve a small B08 leg-1 test fixture out of the REAL LongMemEval-S file.

WHY THIS EXISTS
---------------
``memory/selftest-over-invented-inputs-proves-nothing-about-the-pipeline.md``:
B04's ``--selftest`` passed every day while **no code path ever fed on-disk data
into the metric**. The fix is a fixture with the *real* data shapes, carved from
the real file, so a test exercises the same field names, the same
``question_id`` conventions (including the ``_abs`` suffix), the same
``haystack_sessions`` turn dicts and the same ``answer_session_ids`` linkage
the gate will see at n=134.

``data/longmemeval/longmemeval_s.json`` is 278,025,796 B, so this script
**streams** the top-level JSON array with ``JSONDecoder.raw_decode`` and keeps
only the records it selects. It never holds the whole file.

WHAT IT SELECTS (and why each one)
----------------------------------
From the retrieval-closed stratum only (``knowledge-update`` +
``single-session-assistant`` — the gate's cell):

  * 2 x ``knowledge-update``            non-abstention
  * 1 x ``knowledge-update``  ``_abs``  -> the abstention rule must be exercised;
                                          per prereg 5.2 mis-scoring these
                                          corrupts BOTH ACC and U
  * 2 x ``single-session-assistant``    non-abstention (0 ``_abs`` exist in SSA)

and it TRIMS each record's haystack to the gold sessions + a few distractors so
the fixture is small enough to commit, while keeping ``answer_session_ids``
resolvable (i.e. retrieval can still hit). The trim is recorded per record.

Usage (0 GPU)::

    python3 proposal/backlog/B08-memory-applications/code/build_b08_fixture.py \
        --out tests/fixtures/longmemeval_b08_stratum_fixture.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from json import JSONDecoder

DEFAULT_SRC = "data/longmemeval/longmemeval_s.json"
DEFAULT_OUT = "tests/fixtures/longmemeval_b08_stratum_fixture.json"

STRATUM_TYPES = ("knowledge-update", "single-session-assistant")

# (question_type, want_abstention, how_many)
WANT = [
    ("knowledge-update", False, 2),
    ("knowledge-update", True, 1),
    ("single-session-assistant", False, 2),
]

MAX_DISTRACTOR_SESSIONS = 3
MAX_TURNS_PER_SESSION = 8


def _stream_records(path):
    """Yield (index, record) over a top-level JSON array without loading it all."""
    dec = JSONDecoder()
    chunk = 1 << 22
    with open(path, "r", encoding="utf-8") as f:
        buf = f.read(chunk)
        if not buf:
            raise ValueError(f"empty file: {path}")
        pos = buf.index("[") + 1
        idx = 0
        while True:
            while pos < len(buf) and buf[pos] in " \n\r\t,":
                pos += 1
            if pos >= len(buf):
                more = f.read(chunk)
                if not more:
                    return
                buf += more
                continue
            if buf[pos] == "]":
                return
            while True:
                try:
                    obj, end = dec.raw_decode(buf, pos)
                    break
                except ValueError:
                    more = f.read(chunk)
                    if not more:
                        raise
                    buf += more
            yield idx, obj
            idx += 1
            pos = end
            if pos > (1 << 23):
                buf = buf[pos:]
                pos = 0


def _trim(rec):
    """Keep gold sessions + a few distractors; cap turns. Record what was dropped."""
    gold = set(rec.get("answer_session_ids") or [])
    sessions = rec["haystack_sessions"]
    dates = rec["haystack_dates"]
    sids = rec["haystack_session_ids"]
    assert len(sessions) == len(dates) == len(sids), "aligned triple expected"

    keep_idx = [i for i, s in enumerate(sids) if s in gold]
    if not keep_idx:
        raise ValueError(f"{rec['question_id']}: no gold session found in haystack")
    n_dist = 0
    for i in range(len(sids)):
        if i in keep_idx:
            continue
        if n_dist >= MAX_DISTRACTOR_SESSIONS:
            break
        keep_idx.append(i)
        n_dist += 1
    keep_idx.sort()

    out = dict(rec)
    out["haystack_sessions"] = [sessions[i][:MAX_TURNS_PER_SESSION] for i in keep_idx]
    out["haystack_dates"] = [dates[i] for i in keep_idx]
    out["haystack_session_ids"] = [sids[i] for i in keep_idx]
    out["_fixture_provenance"] = {
        "source": DEFAULT_SRC,
        "original_n_sessions": len(sessions),
        "kept_n_sessions": len(keep_idx),
        "kept_session_indices": keep_idx,
        "gold_session_ids": sorted(gold),
        "max_turns_per_session": MAX_TURNS_PER_SESSION,
        "note": "gold sessions ALWAYS kept so answer_session_ids stays resolvable",
    }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not os.path.exists(args.src):
        print(f"ERROR: source not found: {args.src}", file=sys.stderr)
        return 2

    need = {(t, a): n for t, a, n in WANT}
    got = {k: [] for k in need}
    n_seen = 0
    for idx, rec in _stream_records(args.src):
        n_seen += 1
        qtype = rec.get("question_type", "")
        if qtype not in STRATUM_TYPES:
            continue
        is_abs = str(rec.get("question_id", "")).endswith("_abs")
        key = (qtype, is_abs)
        if key in need and len(got[key]) < need[key]:
            rec["_fixture_source_index"] = idx
            got[key].append(_trim(rec))
        if all(len(got[k]) >= need[k] for k in need):
            break

    missing = {k: need[k] - len(got[k]) for k in need if len(got[k]) < need[k]}
    if missing:
        print(f"ERROR: could not fill fixture slots: {missing}", file=sys.stderr)
        return 3

    records = [r for k in WANT for r in got[(k[0], k[1])]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    blob = json.dumps(records, indent=1, ensure_ascii=False)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(blob)

    print(json.dumps({
        "src": args.src,
        "src_bytes": os.path.getsize(args.src),
        "records_scanned": n_seen,
        "out": args.out,
        "out_bytes": len(blob.encode("utf-8")),
        "out_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "n_records": len(records),
        "by_type": {t: sum(1 for r in records if r["question_type"] == t)
                    for t in STRATUM_TYPES},
        "n_abstention": sum(1 for r in records
                            if str(r["question_id"]).endswith("_abs")),
        "question_ids": [r["question_id"] for r in records],
        "source_indices": [r["_fixture_source_index"] for r in records],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
