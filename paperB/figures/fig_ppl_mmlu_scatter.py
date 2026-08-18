#!/usr/bin/env python3
"""Paper B endpoint scatter: held-out PPL versus answer-letter MMLU."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

OUT = Path(__file__).with_name("fig_ppl_mmlu_scatter.pdf")

points = [
    ("Base", 7.3981, 0.6053, "anchor"),
    ("Full32", 7.6700, 0.5877, "control"),
    ("keep8", 13.3332, 0.2535, "platform"),
    ("keep10", 12.8160, 0.2718, "platform"),
    ("keep12", 11.4426, 0.2752, "platform"),
    ("keep14", 10.5613, 0.3191, "trained"),
    ("ShortGPT", 9.7803, 0.4739, "trained"),
    ("Frozen", 12.7970, 0.2628, "trained"),
    ("Random", 11.4980, 0.2470, "trained"),
]

styles = {
    "anchor": dict(marker="*", s=85, color="#222222"),
    "control": dict(marker="D", s=43, color="#2ca02c"),
    "platform": dict(marker="o", s=43, facecolors="white", edgecolors="#1f77b4", linewidths=1.3),
    "trained": dict(marker="o", s=43, color="#d62728"),
}
offsets = {
    "Base": (5, -2), "Full32": (5, -12), "keep8": (-33, 8),
    "keep10": (-32, 10), "keep12": (-18, 7), "keep14": (-7, 7),
    "ShortGPT": (5, -1), "Frozen": (7, -17), "Random": (7, -11),
}

fig, ax = plt.subplots(figsize=(4.7, 3.25))
for name, ppl, mmlu, group in points:
    ax.scatter([ppl], [mmlu], zorder=3, **styles[group])
    dx, dy = offsets[name]
    ax.annotate(name, (ppl, mmlu), xytext=(dx, dy), textcoords="offset points", fontsize=7)

# Connect the inherited-prefix endpoint ladder without implying equal compute.
ladder = [p for p in points if p[0] in {"keep8", "keep10", "keep12", "keep14"}]
ladder.sort(key=lambda x: x[1], reverse=True)
ax.plot([p[1] for p in ladder], [p[2] for p in ladder], color="#1f77b4", lw=0.8, ls="--", alpha=0.7, zorder=1)
ax.axhline(0.25, color="gray", ls=":", lw=0.8)
ax.text(7.08, 0.256, "chance", color="gray", fontsize=7, ha="left")

ax.set_xlabel("held-out perplexity (lower is better)")
ax.set_ylabel("MMLU answer-letter accuracy")
ax.set_xlim(7.0, 13.8)
ax.set_ylim(0.225, 0.63)
ax.grid(alpha=0.18, lw=0.5)

legend_handles = [
    ax.scatter([], [], **styles["anchor"], label="base"),
    ax.scatter([], [], **styles["control"], label="full32 control"),
    ax.scatter([], [], **styles["trained"], label="200k/control endpoint"),
    ax.scatter([], [], **styles["platform"], label="platform-stopped depth arm"),
]
ax.legend(handles=legend_handles, fontsize=6.5, loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
