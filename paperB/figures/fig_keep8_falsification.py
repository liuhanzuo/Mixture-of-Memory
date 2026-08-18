#!/usr/bin/env python3
"""Paper B: keep8 within-arm falsification trajectory.

Top-panel PPL values are the five held-out checkpoints in PAPER_B_DATA.md
sec. 3(a).  Bottom-panel task values are the 10k/25k/44k downstream checkpoints
in sec. 3(b).  The panels intentionally retain their native checkpoint grids;
the 15k PPL value is not mislabeled as a 10k result.
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_keep8_falsification.pdf")

PPL_STEPS = [5, 15, 25, 35, 44]
PPL = [22.331, 17.868, 16.426, 15.612, 15.131]

TASK_STEPS = [10, 25, 44]
RAW = {
    "HellaSwag": [0.3915, 0.4390, 0.4694],
    "ARC-C": [0.2654, 0.3114, 0.3140],
    "LAMBADA": [0.3429, 0.3827, 0.4333],
    "BoolQ": [0.5713, 0.6080, 0.5881],
    "MMLU": [0.2542, 0.2502, 0.2463],
    "WinoGrande": [0.5209, 0.5083, 0.5185],
}
COLORS = {
    "HellaSwag": "#1f77b4",
    "ARC-C": "#17becf",
    "LAMBADA": "#2ca02c",
    "BoolQ": "#7f7f7f",
    "MMLU": "#d62728",
    "WinoGrande": "#ff7f0e",
}
MARKERS = {
    "HellaSwag": "o",
    "ARC-C": "^",
    "LAMBADA": "s",
    "BoolQ": "D",
    "MMLU": "X",
    "WinoGrande": "P",
}

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(4.35, 4.15), gridspec_kw={"height_ratios": [1.0, 1.25]}
)

ax1.plot(PPL_STEPS, PPL, "o-", color="#1f77b4", lw=1.7, ms=5)
for step, value in zip(PPL_STEPS, PPL):
    ax1.annotate(f"{value:.2f}", (step, value), xytext=(0, 5),
                 textcoords="offset points", ha="center", fontsize=6.5)
ax1.set_ylabel("held-out PPL")
ax1.set_xlim(3, 46)
ax1.set_ylim(14.2, 23.3)
ax1.set_xticks(PPL_STEPS)
ax1.set_title("keep8: language modelling improves with healing", fontsize=9)
ax1.grid(True, ls=":", alpha=0.35)

for task, values in RAW.items():
    gains = [(v - values[0]) * 100 for v in values]
    emphasis = task in {"MMLU", "WinoGrande"}
    ax2.plot(
        TASK_STEPS, gains,
        marker=MARKERS[task],
        color=COLORS[task],
        lw=1.8 if emphasis else 1.2,
        ms=5 if emphasis else 4,
        label=task,
    )

ax2.axhline(0, color="black", lw=0.7)
ax2.set_xlabel("healing step (thousands)")
ax2.set_ylabel(r"$\Delta$ accuracy from 10k (points)")
ax2.set_xlim(8, 46)
ax2.set_xticks(TASK_STEPS)
ax2.set_ylim(-2.2, 10.2)
ax2.grid(True, ls=":", alpha=0.35)
ax2.legend(fontsize=6.1, ncol=3, loc="upper left", frameon=False,
           handlelength=1.7, columnspacing=0.9)
ax2.text(
    10.2, -1.85,
    "MMLU raw: 24.6--25.4%   WinoGrande raw: 50.8--52.1%",
    fontsize=6.2, color="#8c1d1d",
)

fig.tight_layout(h_pad=0.8)
fig.savefig(OUT)
print("wrote", OUT)
