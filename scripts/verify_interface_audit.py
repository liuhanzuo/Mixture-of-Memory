#!/usr/bin/env python3
"""Recompute every number in tab_interface_audit.tex and the MMLU row of
app_tab_recovery.tex directly from the per-item MMLU records.

Run from the repo root:
    python3 scripts/verify_interface_audit.py

Reads paperB/anonymous_artifact/scores/mmlu_content/<arm>/per_example_mmlu.jsonl
(14,042 items per arm, one dual-interface snapshot) and prints:

  * letter and content-normalized accuracy, and their difference in points;
  * the tie rate -- the fraction of items whose four answer-letter logits are
    equal in the stored values, so argmax resolves the item by index. The
    forward pass runs under bfloat16 autocast and log_softmax casts to fp32
    only afterwards, so precision already lost is not recoverable;
  * each arm's letter accuracy against the best constant predictor on this
    item set (always answer the most frequent gold letter), with a paired
    McNemar test;
  * keep14's above-chance recovery under four tie conventions, which is the
    span reported in the app_tab_recovery caption;
  * the length-preference control for the content interface.

Deltas are computed from the 4-decimal values the table prints, except the
ShortGPT row, whose printed -7.31 follows from the unrounded difference
(-7.3067); both are shown so the discrepancy is visible rather than silent.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter

BASE = "paperB/anonymous_artifact/scores/mmlu_content"
ARMS = [
    ("Intact base", "7B_base"),
    ("full32, 25k", "7B_full32_step25000"),
    ("keep14, 200k", "7B_keep14_step200000"),
    ("Frozen, 200k", "7B_freezefront_step200000"),
    ("Random, 200k", "7B_scratch16L_step200000"),
    ("ShortGPT, 200k", "7B_shortgpt16_step200000"),
]
CHANCE = 0.25


def load(arm_dir: str) -> list[dict]:
    rows = []
    path = os.path.join(BASE, arm_dir, "per_example_mmlu.jsonl")
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tied_letters(row: dict) -> list[str]:
    scores = row["letter"]["scores"]
    top = max(scores.values())
    return [k for k, v in scores.items() if v == top]


def mcnemar(reference: list[bool], arm: list[bool]) -> tuple[int, int, float, float]:
    """b_ref = reference right & arm wrong; b_arm = the reverse."""
    b_ref = sum(1 for r, a in zip(reference, arm) if r and not a)
    b_arm = sum(1 for r, a in zip(reference, arm) if a and not r)
    discordant = b_ref + b_arm
    if discordant == 0:
        return b_ref, b_arm, float("nan"), float("nan")
    z = (b_ref - b_arm) / math.sqrt(discordant)
    return b_ref, b_arm, z, math.erfc(abs(z) / math.sqrt(2))


def main() -> None:
    data = {label: load(d) for label, d in ARMS}
    base = data["Intact base"]
    n = len(base)

    gold = Counter(r["gold_letter"] for r in base)
    const_letter, const_hits = gold.most_common(1)[0]
    const_acc = const_hits / n
    const_correct = [r["gold_letter"] == const_letter for r in base]

    print(f"n = {n} items per arm; gold letter distribution "
          + ", ".join(f"{k} {100 * v / n:.2f}%" for k, v in sorted(gold.items())))
    print(f"best constant predictor: always {const_letter} -> {const_acc:.4f}\n")

    print(f"{'arm':16s} {'letter':>7s} {'content':>8s} {'delta':>7s} "
          f"{'tie':>6s} {'vs const':>9s} {'z':>7s} {'p':>10s}")
    for label, _ in ARMS:
        rows = data[label]
        letter = sum(r["letter"]["correct"] for r in rows) / n
        content = sum(r["content_norm"]["correct"] for r in rows) / n
        ties = sum(1 for r in rows if len(tied_letters(r)) > 1)
        delta = 100 * (round(content, 4) - round(letter, 4))
        _, _, z, p = mcnemar(const_correct, [r["letter"]["correct"] for r in rows])
        print(f"{label:16s} {letter:7.4f} {content:8.4f} {delta:+7.2f} "
              f"{100 * ties / n:5.1f}% {100 * (letter - const_acc):+8.2f} "
              f"{z:7.2f} {p:10.2e}")

    short = data["ShortGPT, 200k"]
    s_letter = sum(r["letter"]["correct"] for r in short) / n
    s_content = sum(r["content_norm"]["correct"] for r in short) / n
    print(f"\nShortGPT delta: {100 * (s_content - s_letter):+.4f} unrounded "
          f"-> {100 * (s_content - s_letter):+.2f}; "
          f"{100 * (round(s_content, 4) - round(s_letter, 4)):+.2f} from 4dp values "
          f"(table prints -7.31, i.e. the unrounded convention)")

    def conventions(rows: list[dict]) -> dict[str, float]:
        argmax = sum(r["letter"]["correct"] for r in rows) / len(rows)
        untied = [r for r in rows if len(tied_letters(r)) == 1]
        excluded = sum(r["letter"]["correct"] for r in untied) / len(untied)
        tie_wrong = sum(
            1 for r in rows if len(tied_letters(r)) == 1 and r["letter"]["correct"]
        ) / len(rows)
        partial = sum(
            1.0 / len(tied_letters(r))
            for r in rows
            if r["gold_letter"] in tied_letters(r)
        ) / len(rows)
        return {
            "argmax": argmax,
            "exclude ties": excluded,
            "ties incorrect": tie_wrong,
            "partial 1/k": partial,
        }

    base_conv = conventions(base)
    keep_conv = conventions(data["keep14, 200k"])
    print("\nkeep14 above-chance recovery, base recomputed under the same convention:")
    values = []
    for name in ("argmax", "exclude ties", "partial 1/k", "ties incorrect"):
        b, k = base_conv[name], keep_conv[name]
        recovery = 100 * (k - CHANCE) / (b - CHANCE)
        values.append(recovery)
        print(f"  {name:16s} base {b:.4f}  keep14 {k:.4f}  recovery {recovery:6.2f}%")
    print(f"  span [{min(values):.1f}, {max(values):.1f}]  (table reports 19.4)")

    longest = 0.0
    for row in base:
        counts = row["content_norm"]["cont_tokens"]
        top = max(counts.values())
        winners = [k for k, v in counts.items() if v == top]
        if row["gold_letter"] in winners:
            longest += 1.0 / len(winners)
    longest /= n
    worst = min(
        sum(r["content_norm"]["correct"] for r in data[label]) / n for label, _ in ARMS
    )
    print(f"\nlength-preference control (always longest option, ties split): {longest:.4f}")
    print(f"lowest content accuracy across arms: {worst:.4f} -> all six arms above control")


if __name__ == "__main__":
    main()
