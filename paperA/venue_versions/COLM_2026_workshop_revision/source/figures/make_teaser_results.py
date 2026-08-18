#!/usr/bin/env python3
"""Paper A teaser using audited Qwen3-8B result cohorts.

(a) RULER variable tracking with YaRN enabled for both methods.
(b) Same-platform LoRA-on peak memory.
(c) Same-platform LoRA-on prefill speedup.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "mathtext.fontset": "cm", "axes.linewidth": 0.8})

C_DENSE = "#d1495b"
C_COMEM = "#2e8b7f"
C_DEC   = "#8ec3b9"
C_WIN   = "#8a8a8a"

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.1, 2.5))
fig.subplots_adjust(wspace=0.38, bottom=0.24, top=0.84, left=0.06, right=0.98)

# ---- (a) accuracy ----
x = np.array([8, 16, 32, 64, 128])
comem = [96.2, 98.0, 98.2, 98.6, 99.0]      # native, n=100/cell
full_yarn = [99.2, 99.4, 26.6, 67.2, 57.8]  # YaRN, n=100/cell
axA.set_xscale("log", base=2)
axA.plot(x, comem, "^-", color=C_COMEM, lw=1.8, ms=4.5,
         label="CoMem + LoRA")
axA.plot(x, full_yarn, "o-", color=C_DENSE, lw=1.6, ms=4,
         label="KV-Direct + YaRN")
axA.axvline(40.96, color=C_WIN, ls=":", lw=1.1)
axA.text(52, 10, "native\nwindow", color=C_WIN, fontsize=6.2, ha="center",
         bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
axA.set_ylim(0, 108)
axA.set_xticks([8, 32, 128])
axA.set_xticklabels(["8k", "32k", "128k"], fontsize=7)
axA.tick_params(labelsize=7)
axA.set_ylabel("RULER var-track", fontsize=8)
axA.set_title("(a) length extension", fontsize=8.5)
axA.legend(fontsize=5.5, loc="center right", frameon=False, handlelength=1.1)

# ---- (b) memory ----
xm = np.array([8, 32, 128])
dense_mem = [19.92, 33.82, 89.39]
axB.set_xscale("log", base=2)
axB.plot(xm, dense_mem, "o-", color=C_DENSE, lw=1.6, ms=4, label="Full-context")
axB.plot([8, 32, 128], [17.60, 17.79, 18.54], "^--", color=C_COMEM, lw=1.8, ms=4.5,
         label="CoMem + LoRA")
axB.axvline(40.96, color=C_WIN, ls=":", lw=1.1)
axB.annotate("bounded $\\approx$18 GB", xy=(128, 18.54),
             xytext=(40, 6), color=C_COMEM, fontsize=6.5, ha="center",
             arrowprops=dict(arrowstyle="->", color=C_COMEM, lw=0.8))
axB.set_ylim(0, 100)
axB.set_xticks([8, 32, 128])
axB.set_xticklabels(["8k", "32k", "128k"], fontsize=7)
axB.tick_params(labelsize=7)
axB.set_ylabel("peak mem (GB)", fontsize=8)
axB.set_title("(b) memory", fontsize=8.5)
axB.legend(fontsize=6.2, loc="upper left", frameon=False, handlelength=1.3)

# ---- (c) speedup ----
lengths = ["8k", "32k", "128k"]
prefill = [0.53, 1.21, 2.74]
xi = np.arange(3)
bars = axC.bar(xi, prefill, 0.55, color=C_COMEM, label="prefill")
axC.axhline(1.0, color=C_WIN, ls=":", lw=1.0)
axC.text(2.35, 1.05, "parity", color=C_WIN, fontsize=6, va="bottom", ha="right")
for bar, v in zip(bars, prefill):
    axC.text(bar.get_x() + bar.get_width() / 2, v + 0.07, f"{v:g}$\\times$",
             ha="center", fontsize=6.5)
axC.set_ylim(0, 3.25)
axC.set_xticks(xi)
axC.set_xticklabels(lengths, fontsize=7)
axC.tick_params(labelsize=7)
axC.set_ylabel("prefill speedup", fontsize=8)
axC.set_title("(c) same-platform speed", fontsize=8.5)
axC.legend(fontsize=6.2, loc="upper left", frameon=False, handlelength=1.1)

fig.savefig("teaser_results.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("teaser_results.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote teaser_results.pdf / .png")
