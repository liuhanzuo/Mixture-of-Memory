#!/usr/bin/env python3
"""paper/figures/pack.pdf --- the read pack is constant in context length.

Top: context of arbitrary length as many small chunk cells (grows with L).
A BM25 arrow selects top-k.
Bottom: the fixed read pack [sink | h_j(sel_1..k) | query], with a bracket
"sink + k*c + query  ~= 6.7k tokens (fixed, independent of context length)".
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})

# muted, low-saturation tints with thin dark edges (paper style, not infographic)
C_CTX   = "#eef2f6"   # context chunks fill
C_CTXE  = "#9fb3c8"   # context chunks edge
C_SEL   = "#dcebe8"   # selected hiddens fill (light teal)
C_SELE  = "#2e8b7f"   # selected edge
C_SINK  = "#ededed"   # sink fill
C_SINKE = "#8a8a8a"
C_QUERY = "#fbe6d8"   # query fill (light coral)
C_QUERYE = "#d98a5c"
C_ARROW = "#3a3a3a"

fig, ax = plt.subplots(figsize=(3.35, 2.05))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")


def cell(x, y, w, h, fc, ec="#5a5a5a", lw=0.6):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw,
                 zorder=2))


# ---- top: long context ----
ax.text(0.1, 9.35, "context (up to 128k tokens)", fontsize=8, ha="left")
n = 18
x0, w, gap = 0.3, 0.46, 0.075
y = 8.2
for i in range(n):
    cell(x0 + i * (w + gap), y, w, 0.72, C_CTX, ec=C_CTXE)
# ellipsis to imply "grows"
ax.text(x0 + n * (w + gap) + 0.15, y + 0.36, r"$\cdots$", fontsize=11, va="center")

# ---- BM25 arrow ----
ax.add_patch(FancyArrowPatch((5.0, 7.9), (5.0, 6.15), arrowstyle="-|>",
             mutation_scale=11, lw=1.2, color=C_ARROW, zorder=3))
ax.text(5.22, 7.0, "BM25 top-$k$", fontsize=8.2, color="#222222", ha="left", va="center")

# ---- bottom: the fixed read pack ----
py = 4.2
ph = 0.95
labels = [("sink", C_SINK, C_SINKE),
          (r"$h_j(\mathrm{sel}_1)$", C_SEL, C_SELE),
          (r"$h_j(\mathrm{sel}_2)$", C_SEL, C_SELE),
          (r"$\cdots$", None, None),
          (r"$h_j(\mathrm{sel}_k)$", C_SEL, C_SELE),
          ("query", C_QUERY, C_QUERYE)]
pw = 1.5
pgap = 0.12
total = len(labels) * pw + (len(labels) - 1) * pgap
px0 = (10 - total) / 2
for i, (lab, fc, ec) in enumerate(labels):
    x = px0 + i * (pw + pgap)
    if fc is None:
        ax.text(x + pw / 2, py + ph / 2, lab, fontsize=11, ha="center", va="center")
        continue
    cell(x, py, pw, ph, fc, ec=ec, lw=0.9)
    ax.text(x + pw / 2, py + ph / 2, lab, fontsize=8.2, ha="center", va="center",
            color="black")

# bracket under the pack
by = py - 0.35
ax.add_line(Line2D([px0, px0 + total], [by, by], color="black", lw=0.9))
ax.add_line(Line2D([px0, px0], [by, by + 0.12], color="black", lw=0.9))
ax.add_line(Line2D([px0 + total, px0 + total], [by, by + 0.12], color="black", lw=0.9))
ax.text(5.0, by - 0.75,
        r"$\mathrm{sink}+k\!\cdot\!c+\mathrm{query}\approx 6.7$k tokens",
        fontsize=8.6, ha="center", va="center")
ax.text(5.0, by - 1.5, "fixed, independent of context length",
        fontsize=8.0, ha="center", va="center", style="italic")

fig.savefig("pack.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("pack.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote pack.pdf / pack.png")
