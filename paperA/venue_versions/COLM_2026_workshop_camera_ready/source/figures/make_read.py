#!/usr/bin/env python3
"""paper/figures/read_motiv.pdf --- selection, not fidelity, is the bottleneck.

RULER niah_single recall vs context length for four selectors:
  oracle (gold chunk always in), BM25, recency, and residual-state reader
  attention. Data are the chat-template-free selector sweep in tab_selector.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "mathtext.fontset": "cm", "axes.linewidth": 0.8})

C_ORACLE = "#8a8a8a"
C_BM25   = "#2e8b7f"
C_RECENCY = "#e0873a"
C_READER = "#7a5aa6"

x = np.array([8, 16, 32])
oracle   = [100, 100, 100]
bm25     = [100, 100, 100]
recency  = [100, 100, 82]
reader   = [100, 100, 73]

fig, ax = plt.subplots(figsize=(3.2, 2.0))
ax.set_xscale("log", base=2)
ax.plot(x, oracle, "--", color=C_ORACLE, lw=1.4, label="oracle")
ax.plot(x, bm25, "^-", color=C_BM25, lw=1.9, ms=4.5, label="BM25")
ax.plot(x, recency, "o-", color=C_RECENCY, lw=1.5, ms=3.6, label="recency")
ax.plot(x, reader, "s-", color=C_READER, lw=1.5, ms=3.6, label="reader attention")

ax.annotate("BM25 $\\approx$ oracle", (16, 100), (8.6, 84), fontsize=6.6,
            color=C_BM25, ha="left")
ax.annotate("selection gap", (32, 73), (17, 50), fontsize=6.6, color=C_READER,
            ha="left", arrowprops=dict(arrowstyle="-|>", color=C_READER, lw=0.8))

ax.set_ylim(-6, 112)
ax.set_xticks([8, 16, 32])
ax.set_xticklabels(["8k", "16k", "32k"], fontsize=7)
ax.tick_params(labelsize=7)
ax.set_xlabel("context length", fontsize=8)
ax.set_ylabel("niah\\_single recall", fontsize=8)
ax.legend(fontsize=6.4, loc="lower left", frameon=False, handlelength=1.5,
          borderaxespad=0.2)

fig.savefig("read_motiv.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("read_motiv.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote read_motiv.pdf / .png")
