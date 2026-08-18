#!/usr/bin/env python3
"""Paper B: same-model OLMo semantic, MMLU, and next-token readout depths.

MMLU logit-lens accuracy comes from
``results/knowledge_logit_lens_OLMo-2-1124-7B.json``. Semantic and next-token
sat95 depths come from ``results/probe_linguistic_olmo2_7b.json``.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "fig_two_depths.pdf")


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


knowledge = load_json(os.path.join(
    REPO, "results", "knowledge_logit_lens_OLMo-2-1124-7B.json"
))
linguistic = load_json(os.path.join(
    REPO, "results", "probe_linguistic_olmo2_7b.json"
))

points = knowledge["per_layer"]
frac_depth = [point["frac_depth"] for point in points]
mmlu_acc = [point["mmlu_acc"] for point in points]
summary = knowledge["summary"]
chance = knowledge["meta"].get("chance", 0.25)
semantic_sat = linguistic["division_of_labour"]["semantic_sat95_frac_depth"]
nexttoken_sat = linguistic["division_of_labour"]["nexttoken_sat95_frac_depth"]

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(
    frac_depth,
    mmlu_acc,
    "o-",
    color="#1f77b4",
    ms=3,
    label="OLMo-2-7B MMLU logit lens",
)
ax.axhline(chance, color="gray", ls="--", lw=0.8)
ax.text(0.02, chance + 0.01, "chance", fontsize=7, color="gray")

ax.axvline(semantic_sat, color="#2ca02c", ls="-.", lw=1.2)
ax.text(
    semantic_sat + 0.012,
    0.57,
    f"semantic sat$_{{95}}$\n{semantic_sat:.3f}L",
    fontsize=7,
    color="#2ca02c",
)

onset = summary["onset_frac_depth"]
sat95 = summary["sat95_frac_depth"]
ax.axvline(sat95, color="#1f77b4", ls=":", lw=1.1)
ax.annotate(
    f"MMLU onset/sat$_{{95}}$\n{onset:.3f}L / {sat95:.3f}L",
    (onset, 0.326),
    fontsize=7,
    color="#1f77b4",
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8),
    xytext=(onset - 0.24, 0.44),
)

ax.axvline(nexttoken_sat, color="#9467bd", ls="-.", lw=1.2)
ax.text(
    nexttoken_sat - 0.015,
    0.64,
    f"next-token sat$_{{95}}$\n{nexttoken_sat:.3f}L",
    fontsize=7,
    color="#9467bd",
    ha="right",
)

ax.axvspan(8 / 32, 12 / 32, color="#d62728", alpha=0.10)
ax.text(
    10 / 32,
    0.675,
    "keep8--keep12\nfrontier",
    fontsize=6.7,
    color="#a61c1c",
    ha="center",
    va="top",
)
ax.axvline(14 / 32, color="#d62728", ls=":", lw=1.1)
ax.text(
    14 / 32,
    0.67,
    "keep14 cut",
    rotation=90,
    fontsize=6.7,
    color="#a61c1c",
    ha="right",
    va="top",
)

ax.set_xlabel("fractional depth")
ax.set_ylabel("MMLU logit-lens accuracy")
ax.set_title("OLMo readout depths separate across the stack")
ax.set_xlim(0, 1.02)
ax.set_ylim(0.15, 0.72)
ax.legend(fontsize=7, loc="lower right")
ax.grid(True, ls=":", alpha=0.4)

fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
