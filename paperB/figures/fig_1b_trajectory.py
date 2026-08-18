#!/usr/bin/env python3
"""Appendix figure: OLMo-2-1B keep7 healing trajectory.

Values are transcribed from status/OLMO2_PRUNEHEAL_{PPL,DOWNSTREAM}.md.
PPL checkpoints and downstream checkpoints differ slightly at the final point,
so each curve uses its native x coordinates.
"""
import os

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_1b_trajectory.pdf")

ppl_steps = [50, 100, 147]
ppl = [17.619, 16.161, 15.628]

core_steps = [50, 100, 147, 148.5]
hs = [0.411, 0.438, 0.450, 0.448]
arc_e = [0.568, 0.593, 0.595, 0.606]
piqa = [0.671, 0.687, 0.692, 0.695]

know_steps = [50, 100, 147, 150]
mmlu = [0.2495, 0.2558, 0.2529, 0.2480]
lambada = [0.3536, 0.3740, 0.4021, 0.3969]

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(5.0, 4.25), gridspec_kw={"height_ratios": [0.9, 1.25]}
)

ax1.plot(ppl_steps, ppl, "o-", color="#1f77b4", lw=1.7, ms=5)
for x, y in zip(ppl_steps, ppl):
    ax1.annotate(f"{y:.2f}", (x, y), xytext=(0, 5),
                 textcoords="offset points", ha="center", fontsize=7)
ax1.axhline(10.642, color="black", ls=":", lw=0.9, label="full 1B base")
ax1.set_ylabel("held-out PPL")
ax1.set_xlim(43, 156)
ax1.set_ylim(10.0, 18.4)
ax1.grid(True, ls=":", alpha=0.35)
ax1.legend(fontsize=7, frameon=False, loc="upper right")
ax1.set_title("OLMo-2-1B keep7: PPL improves while MMLU stays at chance",
              fontsize=9)

ax2.plot(core_steps, hs, "o-", label="HellaSwag", color="#1f77b4")
ax2.plot(core_steps, arc_e, "^-", label="ARC-E", color="#17becf")
ax2.plot(core_steps, piqa, "s-", label="PIQA", color="#2ca02c")
ax2.plot(know_steps, lambada, "D-", label="LAMBADA", color="#9467bd")
ax2.plot(know_steps, mmlu, "X-", label="MMLU", color="#d62728", lw=2.0)
ax2.axhline(0.25, color="#d62728", ls=":", lw=0.9)
ax2.text(44, 0.258, "MMLU chance", fontsize=6.7, color="#a61c1c")
ax2.set_xlabel("healing step (thousands)")
ax2.set_ylabel("accuracy / acc_norm")
ax2.set_xlim(43, 156)
ax2.set_ylim(0.20, 0.73)
ax2.grid(True, ls=":", alpha=0.35)
ax2.legend(fontsize=6.7, ncol=3, loc="upper left", frameon=False)

fig.tight_layout(h_pad=0.8)
fig.savefig(OUT)
print("wrote", OUT)
