#!/usr/bin/env python3
"""Paper B: above-chance recovery across train-all, frozen-front, and random-init controls.

Data come from PAPER_B_DATA.md sec. 3(b).

Above-chance recovery uses the SAME formula the materials use for MMLU:
    recovery = (acc - chance) / (base - chance)
The MMLU chance floor .25 and the recovery formula are from the materials; the
per-task standard chance floors below are the conventional MC baselines, and this
derivation reproduces the final-checkpoint recovery numbers
(e.g. MMLU healed 19.5% / scratch ~0%; WinoGrande healed 51.6% / scratch 18.4%).
No invented raw numbers.

Run:  python fig_capability_cliff.py  -> writes fig_capability_cliff.pdf.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_capability_cliff.pdf")

# Canonical mixed metrics from the final 200k evaluations:
# acc_norm for HS/ARC/PIQA/OBQA, raw accuracy for all remaining tasks.
DATA = {
    # task            base   train-all  frozen   random-init  chance  family
    "ARC-E":        (0.829, 0.705, 0.666, 0.697, 0.25, "surface"),
    "PIQA":         (0.811, 0.745, 0.724, 0.733, 0.50, "surface"),
    "HellaSwag":    (0.805, 0.645, 0.595, 0.578, 0.25, "reasoning"),
    "ARC-C":        (0.571, 0.438, 0.381, 0.414, 0.25, "reasoning"),
    "WinoGrande":   (0.744, 0.626, 0.646, 0.545, 0.50, "reasoning"),
    "OBQA":         (0.462, 0.404, 0.366, 0.384, 0.25, "reasoning"),
    "LAMBADA":      (0.732, 0.577, 0.513, 0.484, 0.00, "reasoning"),
    "SIQA":         (0.502, 0.434, 0.414, 0.416, 0.333, "reasoning"),
    "CSQA":         (0.665, 0.499, 0.453, 0.45045045045045046, 0.20, "reasoning"),
    "BoolQ":        (0.815, 0.638, 0.592, 0.614, 0.50, "comprehension"),
    "MMLU":         (0.6053, 0.3191, 0.2628, 0.2461, 0.25, "knowledge-sensitive"),
}

FAMILY_ORDER = ["surface", "reasoning", "comprehension", "knowledge-sensitive"]
FAMILY_COLOR = {"surface": "#7f7f7f", "reasoning": "#1f77b4",
                "comprehension": "#2ca02c", "knowledge-sensitive": "#d62728"}


def recovery(acc, base, chance):
    return max(0.0, (acc - chance) / (base - chance)) * 100.0


# order tasks by family
tasks = sorted(DATA.keys(), key=lambda t: FAMILY_ORDER.index(DATA[t][5]))
healed_rec = [recovery(DATA[t][1], DATA[t][0], DATA[t][4]) for t in tasks]
frozen_rec = [recovery(DATA[t][2], DATA[t][0], DATA[t][4]) for t in tasks]
scratch_rec = [recovery(DATA[t][3], DATA[t][0], DATA[t][4]) for t in tasks]

x = np.arange(len(tasks))
w = 0.26
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(x - w, healed_rec, w, label="inherited, train all", color="#1f77b4")
ax.bar(x, frozen_rec, w, label="inherited front frozen", color="#ff7f00", alpha=0.85)
ax.bar(x + w, scratch_rec, w, label="random init (5x peak LR)", color="#d62728", alpha=0.8)

ax.set_ylabel("above-chance recovery (%)")
ax.set_xticks(x)
ax.set_xticklabels(tasks, rotation=40, ha="right", fontsize=8)
ax.set_title("Capability recovery at the keep14 shape")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, axis="y", ls=":", alpha=0.4)
ax.set_ylim(0, 100)

# color the x tick labels by family
for lbl, t in zip(ax.get_xticklabels(), tasks):
    lbl.set_color(FAMILY_COLOR[DATA[t][5]])

# Highlight MMLU as the clearest dissociation.
mi = tasks.index("MMLU")
ax.annotate(
    "MMLU recovery\nlags",
    xy=(mi, healed_rec[mi]),
    xytext=(mi - 0.35, 46),
    textcoords="data",
    fontsize=8,
    color="#d62728",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
)

fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
