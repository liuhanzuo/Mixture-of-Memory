#!/usr/bin/env python3
"""paper/figures/depth_motiv.pdf --- the depth division of labor (trend figure).

(a) RULER niah_single (16k, Qwen3-8B) vs split depth j: zero-shot recall is flat
    while j is shallow then falls off a cliff; a light self-distilled adapter
    pushes the readable depth deeper.
(b) Across three model families: semantic content is linearly decodable in the
    mid-network, while each model's own LM-head pathway surfaces it much later.
    The shaded gap is the readout work that adaptation must repair. [tab_depth]
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "mathtext.fontset": "cm", "axes.linewidth": 0.8})

C_ZS   = "#d1495b"   # zero-shot
C_AD   = "#2e8b7f"   # +adapter / CoMem
C_CONT = "#1f6f8b"   # content depth
C_READ = "#d1495b"   # readable depth
C_GAP  = "#bcd4cf"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(3.38, 1.85))
fig.subplots_adjust(wspace=0.5, bottom=0.26, top=0.86, left=0.13, right=0.98)

# ---- (a) knee vs j (Qwen3-8B, niah_single 16k) ----
j_zs = [0, 3, 6, 9, 12, 18]
r_zs = [100, 100, 100, 100, 12, 0]
axA.plot(j_zs, r_zs, "o-", color=C_ZS, lw=1.5, ms=3.2, label="zero-shot")
axA.plot(12, 100, "*", color=C_AD, ms=9, label="+adapter", zorder=5)
axA.annotate("adapter\npushes deeper", (12, 100), (5.2, 66), fontsize=5.6,
             color=C_AD, ha="left",
             arrowprops=dict(arrowstyle="-|>", color=C_AD, lw=0.8))
axA.set_xlim(-1, 19)
axA.set_ylim(-6, 112)
axA.set_xlabel("split depth $j$", fontsize=7.5)
axA.set_ylabel("niah\\_single", fontsize=7.5)
axA.set_title("(a) readable depth", fontsize=7.8)
axA.tick_params(labelsize=6.6)
axA.legend(fontsize=5.8, loc="lower left", frameon=False, handlelength=1.2)

# ---- (b) linear content vs native readout across families ----
families = ["Qwen", "Llama", "OLMo"]
x_family = np.arange(len(families))
content = np.array([0.393, 0.269, 0.285])
native = np.array([0.824, 0.985, 0.875])
axB.fill_between(x_family, content, native, color=C_GAP, alpha=0.9, zorder=1,
                 label="readout gap")
axB.plot(x_family, content, "s-", color=C_CONT, lw=1.5, ms=3,
         label="linear knee")
axB.plot(x_family, native, "o--", color=C_READ, lw=1.5, ms=3,
         label="native knee")
axB.set_ylim(0, 1.08)
axB.set_xticks(x_family)
axB.set_xticklabels(families, fontsize=6.6)
axB.tick_params(labelsize=6.6)
axB.set_xlabel("model family", fontsize=7.5)
axB.set_ylabel("depth $/L$", fontsize=7.5)
axB.set_title("(b) linear vs. native readout", fontsize=7.8)
axB.legend(fontsize=5.4, loc="center left", frameon=False, handlelength=1.2,
           borderaxespad=0.2)

fig.savefig("depth_motiv.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig("depth_motiv.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
print("wrote depth_motiv.pdf / .png")
