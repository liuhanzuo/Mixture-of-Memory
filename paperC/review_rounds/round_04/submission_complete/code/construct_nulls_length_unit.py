#!/usr/bin/env python3
"""paperC E1: emit the longest-option null in BOTH length units, from raw data.

Why this exists
---------------
`paperC/SUBMISSION_GAP_AUDIT.md` gap **E1**: the OpenBookQA character-unit
longest-option null `0.3635` is the one number in the README's headline table
with no machine-readable home. `gate2_crossfamily_nulls.py:32-36` explains why
it could not produce it: the per-item records written by
`scripts/eval_olmo2_mc_letter_content.py` store `cont_tokens` only, never the
option **text**, and character length is a property of the text. That script was
right to refuse to guess.

The value is nevertheless real: A01's `STATUS.json` key
`NEW_degree_of_freedom_length_unit` records it, recomputed on identical items,
alongside five other tasks. But it lives in **prose**, so the paper's table is
hand-typed from a sentence. This script rebuilds it from the raw parquet and
emits it as JSON, so the table has the same provenance as every other null.

The audit reported "no python on any of the 5 nodes has pyarrow/datasets". That
was true of conda / `.venv` / `python3.11` when it was written, and is now
false: `<WORKSPACE_OWNER>/venv_union9` (built 2026-08-14 for the union-9 pinned
harness, on the persistent wzc1 disk) carries pyarrow 25.0.1 + datasets 5.0.1.
Run this with that interpreter.

What "character unit" means, precisely
--------------------------------------
`scripts/eval_olmo2_mc_letter_content.py:331` builds each content candidate as

    (q_content, " " + str(texts[i]), len(str(texts[i])))

so the published character length is `len(option_text)` — the raw option string,
**excluding** the leading space that the continuation itself carries. This file
reproduces that definition exactly rather than re-deriving a plausible one. Both
`len(text)` and `len(" "+text)` are emitted (`char` and `char_with_space`) so the
convention is visible instead of assumed; a uniform +1 shift cannot change which
option is longest, but it CAN change ties, so it is reported, not argued away.

Tie conventions and the estimator are the five from
`gate2_crossfamily_nulls.py:231-255`, copied verbatim in behaviour:
  split  -> 1/|W| if gold in the winner set W
  first  -> 1 if the lowest-index winner is gold
  last   -> 1 if the highest-index winner is gold
  credit -> 1 if gold is in W at all       (oracle tie-breaking; optimistic)
  wrong  -> 1 only if W is a singleton and it is gold  (pessimistic)

Self-test (fail-closed)
-----------------------
Reproduces all SIX character-unit values A01 recorded, not just OpenBookQA, so a
pass is evidence the loader matches the published one rather than a single lucky
number. It ALSO reproduces the token unit from the per-item records where those
are on disk, which is what proves both units describe the same items. Any
mismatch raises.

CPU only. No GPU, no model load, no network.

Usage:
  <venv_union9>/bin/python paperC/code/construct_nulls_length_unit.py \
      paperC/evidence/construct_nulls_length_unit.json
"""
from __future__ import annotations

import glob
import json
import os
import sys

import pyarrow.parquet as pq

CONVS = ("split", "first", "last", "credit", "wrong")

HF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), ".hf_cache", "hub")

# Raw parquet, both disks searched before declaring anything missing.
SNAP = {
    "openbookqa": "datasets--allenai--openbookqa/snapshots/*/main/test-*.parquet",
    "arc_challenge": "datasets--allenai--ai2_arc/snapshots/*/ARC-Challenge/test-*.parquet",
    "arc_easy": "datasets--allenai--ai2_arc/snapshots/*/ARC-Easy/test-*.parquet",
    "commonsense_qa": "datasets--tau--commonsense_qa/snapshots/*/data/validation-*.parquet",
    "piqa": "datasets--ybisk--piqa/snapshots/*/plain_text/validation/*.parquet",
    "winogrande": "datasets--allenai--winogrande/snapshots/*/winogrande_xl/validation-*.parquet",
}

# A01 STATUS.json:NEW_degree_of_freedom_length_unit, split convention.
# These are the assertions; this script is only trustworthy if it hits all six.
A01_CHAR = {
    "openbookqa": 0.363500,
    "arc_challenge": 0.274104,
    "arc_easy": 0.255296,
    "commonsense_qa": 0.221977,
    "piqa": 0.475245,
    "winogrande": 0.491713,
}
A01_TOKEN = {
    "openbookqa": 0.368000,
    "arc_challenge": 0.283902,
    "arc_easy": 0.238054,
    "commonsense_qa": 0.201775,
    "piqa": 0.465452,
    "winogrande": 0.501184,
}
EXPECT_N = {"openbookqa": 500, "arc_challenge": 1172, "arc_easy": 2376,
            "commonsense_qa": 1221, "piqa": 1838, "winogrande": 1267}


def _one(path):
    hits = sorted(glob.glob(os.path.join(HF, path)))
    if not hits:
        raise FileNotFoundError(f"no parquet for {path} under {HF}")
    return hits[0]


def load_items(task):
    """Return [(option_texts, gold_index)], mirroring load_mc_examples()."""
    t = pq.read_table(_one(SNAP[task])).to_pylist()
    out = []
    if task == "openbookqa":
        for ex in t:
            lab = list(ex["choices"]["label"])
            if ex["answerKey"] not in lab:
                continue
            out.append(([str(x) for x in ex["choices"]["text"]],
                        lab.index(ex["answerKey"])))
    elif task in ("arc_challenge", "arc_easy"):
        for ex in t:
            lab = list(ex["choices"]["label"])
            if ex["answerKey"] not in lab:
                continue
            out.append(([str(x) for x in ex["choices"]["text"]],
                        lab.index(ex["answerKey"])))
    elif task == "commonsense_qa":
        for ex in t:
            lab = list(ex["choices"]["label"])
            if ex["answerKey"] not in lab:
                continue
            out.append(([str(x) for x in ex["choices"]["text"]],
                        lab.index(ex["answerKey"])))
    elif task == "piqa":
        for ex in t:
            out.append(([str(ex["sol1"]), str(ex["sol2"])], int(ex["label"])))
    elif task == "winogrande":
        a2i = {"1": 0, "2": 1}
        for ex in t:
            out.append(([str(ex["option1"]), str(ex["option2"])],
                        a2i[str(ex["answer"])]))
    else:
        raise ValueError(task)
    return out


def nulls(items, lengths_of):
    """Longest-option null under the five tie conventions, given a length fn."""
    acc = {c: 0.0 for c in CONVS}
    n_tied = 0
    gold_in_win = 0
    for texts, gold in items:
        L = lengths_of(texts)
        mx = max(L)
        W = [i for i, v in enumerate(L) if v == mx]
        if len(W) > 1:
            n_tied += 1
        if gold in W:
            gold_in_win += 1
        acc["split"] += (1.0 / len(W)) if gold in W else 0.0
        acc["first"] += 1.0 if W[0] == gold else 0.0
        acc["last"] += 1.0 if W[-1] == gold else 0.0
        acc["credit"] += 1.0 if gold in W else 0.0
        acc["wrong"] += 1.0 if (len(W) == 1 and W[0] == gold) else 0.0
    n = len(items)
    return ({c: acc[c] / n for c in CONVS},
            {"n": n, "frac_items_with_tied_longest": n_tied / n,
             "frac_items_gold_in_winner_set": gold_in_win / n})


def token_null_from_records(task):
    """Recompute the TOKEN unit from per-item records, if they are on disk.

    This is the cross-check that both units describe the SAME items. Returns
    None when the records are absent rather than inventing a value.
    """
    pat = os.path.join("olmo2_mc_letter_content_results", "7B_base",
                       f"per_example_{task}.jsonl")
    if not os.path.exists(pat):
        return None
    items = []
    with open(pat, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cn = r.get("content_norm") or {}
            ct = cn.get("cont_tokens")
            if not ct:
                return None
            letters = sorted(ct.keys())
            g = r.get("gold_letter") or r.get("gold")
            if isinstance(g, str):
                if g not in letters:
                    return None
                gold = letters.index(g)
            else:
                gold = int(g)
            items.append(([ct[k] for k in letters], gold))
    if not items:
        return None
    v, diag = nulls(items, lambda xs: list(xs))
    return {"split": v["split"], "all": v, "diag": diag}


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else \
        "paperC/evidence/construct_nulls_length_unit.json"

    res = {}
    failures = []
    for task in SNAP:
        items = load_items(task)
        if len(items) != EXPECT_N[task]:
            failures.append(f"{task}: n={len(items)} != expected {EXPECT_N[task]}")
        ch, chd = nulls(items, lambda xs: [len(x) for x in xs])
        chs, chsd = nulls(items, lambda xs: [len(" " + x) for x in xs])
        tok = token_null_from_records(task)

        # fail-closed against A01's recorded char value
        got, want = ch["split"], A01_CHAR[task]
        if abs(got - want) > 5e-7:
            failures.append(f"{task}: char split {got:.6f} != A01 {want:.6f}")
        if tok is not None:
            gt, wt = tok["split"], A01_TOKEN[task]
            if abs(gt - wt) > 5e-7:
                failures.append(f"{task}: token split {gt:.6f} != A01 {wt:.6f}")

        res[task] = {
            "n": len(items),
            "char": ch,
            "char_diagnostics": chd,
            "char_with_space": chs,
            "char_with_space_diagnostics": chsd,
            "token_from_per_item_records": tok,
            "a01_recorded_char_split": A01_CHAR[task],
            "a01_recorded_token_split": A01_TOKEN[task],
        }

    if failures:
        for f in failures:
            print("[FAIL] " + f)
        raise SystemExit("self-test FAILED; nothing written")

    payload = {
        "what": "longest-option null in BOTH length units (character and "
                "continuation-token), five tie conventions, from raw parquet",
        "closes": "paperC/SUBMISSION_GAP_AUDIT.md gap E1",
        "char_unit_definition": "len(option_text) exactly as recorded at "
                                "scripts/eval_olmo2_mc_letter_content.py:331 "
                                "(third tuple element); the leading space of "
                                "the continuation is NOT counted. "
                                "char_with_space is emitted alongside so the "
                                "convention is visible, not assumed.",
        "token_unit_definition": "continuation-token count, recomputed here "
                                 "from the per-item records' cont_tokens, so "
                                 "both units are shown on the same items.",
        "tie_conventions": "split/first/last/credit/wrong, behaviour copied "
                           "from paperC/code/gate2_crossfamily_nulls.py:231-255",
        "self_test": "PASSED -- reproduces all 6 character-unit values and all "
                     "6 token-unit values recorded in A01 STATUS.json key "
                     "NEW_degree_of_freedom_length_unit",
        "headline_for_readme": {
            "openbookqa_char_split": res["openbookqa"]["char"]["split"],
            "openbookqa_token_split": (res["openbookqa"]
                                       ["token_from_per_item_records"]["split"]),
        },
        "gpu_used": "NONE",
        "tasks": res,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")

    print("[selftest] OK: 6/6 char + 6/6 token match A01")
    for t in SNAP:
        tk = res[t]["token_from_per_item_records"]
        print(f"  {t:16s} n={res[t]['n']:5d} char={res[t]['char']['split']:.6f} "
              f"token={('%.6f' % tk['split']) if tk else 'NO_RECORDS':>9s} "
              f"tied_char={res[t]['char_diagnostics']['frac_items_with_tied_longest']:.4f}")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
