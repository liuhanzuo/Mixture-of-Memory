#!/usr/bin/env python3
"""Paper B: available-checkpoint depth/PPL frontier.

All values come from PAPER_B_DATA.md secs. 3(a) and 8.1.  The x-axis is the
resulting model depth (keep + 2 fresh blocks) / 32, not the anatomical cut
depth keep / 32.  Checkpoint steps are shown because cross-arm points are not
compute matched.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_depth_ppl.pdf")
BASE = 7.398

# label, resulting model depth, PPL, step (k), plotting style
POINTS = [
    ("keep8", 10 / 32, 15.131, 44.0, "partial"),
    ("keep10", 12 / 32, 17.239, 10.0, "early"),
    ("keep12", 14 / 32, 11.566, 111.5, "partial"),
    ("keep14", 16 / 32, 10.561, 200.0, "final"),
]

fig, ax = plt.subplots(figsize=(4.35, 3.15))

# A light guide through the informative (non-early) checkpoints.  It is
# intentionally dashed to avoid implying a compute-matched scaling law.
guide = [POINTS[i] for i in [0, 2, 3]]
ax.plot([p[1] for p in guide], [p[2] for p in guide],
        "--", color="#7aa6d8", lw=1.0, alpha=0.8)

for label, x, y, step, style in POINTS:
    if style == "early":
        ax.plot(x, y, "o", mfc="white", mec="#1f77b4", mew=1.4, ms=7.5,
                label="very early checkpoint")
        offset = (5, 2)
        suffix = " (early)"
    else:
        ax.plot(x, y, "o", color="#1f77b4", ms=7.5,
                label="inherited + healed" if label == "keep8" else None)
        offset = {
            "keep8": (5, 5),
            "keep12": (-38, 7),
            "keep14": (8, -24),
        }[label]
        suffix = ""
    ax.annotate(
        f"{label} @ {step:g}k{suffix}\n{y:.2f} ({y / BASE:.2f}x)",
        (x, y), xytext=offset, textcoords="offset points", fontsize=6.8,
        color="#555555" if style == "early" else "#1f4e79",
    )

# keep14 apex at the same architecture.
ax.plot(16 / 32, 10.826, "o", mfc="white", mec="#1f77b4", ms=5.5)
ax.annotate("128k: 10.83", (16 / 32, 10.826), xytext=(8, 3),
            textcoords="offset points", fontsize=6.5, color="#1f4e79")

# Cross-policy endpoint, fully random-initialized operating point, and full base.
ax.plot(16 / 32, 9.7803, "D", color="#2ca02c", ms=7,
        label="ShortGPT-16 @ 200k")
ax.annotate("ShortGPT: 9.78 (1.32x)", (16 / 32, 9.7803),
            xytext=(17, -4), textcoords="offset points", fontsize=6.8,
            color="#1b7f1b")

ax.plot(16 / 32, 11.498, "s", color="#d62728", ms=7,
        label="random init (5x peak LR)")
ax.annotate("random init @ 200k\n11.50 (1.55x)",
            (16 / 32, 11.498), xytext=(18, 13), textcoords="offset points",
            fontsize=6.8, color="#a61c1c")

ax.plot(1.0, BASE, "^", color="black", ms=7.5, label="full base")
ax.annotate("full 32L\n7.398", (1.0, BASE), xytext=(-37, 4),
            textcoords="offset points", fontsize=7)

ax.set_xlabel("resulting decoder depth  $(\\mathrm{keep}+2)/32$")
ax.set_ylabel("held-out next-token PPL")
ax.set_xlim(0.27, 1.04)
ax.set_ylim(6.7, 18.3)
ax.grid(True, ls=":", alpha=0.35)
ax.legend(fontsize=6.7, loc="upper right", frameon=False)

fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
