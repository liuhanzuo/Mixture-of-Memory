#!/usr/bin/env python3
"""Paper B keep14 trajectory: PPL and MMLU during late healing.

Evaluated checkpoints are 128k, 153.5k, and 200k. Values come from the copied
merged PPL and downstream summaries documented in PAPER_B_DATA.md.
"""
import os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_trajectory.pdf")

steps_eval = [128000, 153500, 200000]
heldout_ppl = [10.826, 10.693, 10.561]
mmlu = [0.3012, 0.3124, 0.3191]

fig, ax1 = plt.subplots(figsize=(4.6, 3.2))

l1 = ax1.plot(steps_eval, heldout_ppl, "o-", color="#1f77b4", label="held-out PPL")
ax1.set_xlabel("healing step")
ax1.set_ylabel("perplexity", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.set_ylim(9.5, 11.5)
for s, p in zip(steps_eval, heldout_ppl):
    ax1.annotate(f"{p:.3f}", (s, p), textcoords="offset points", xytext=(0, 6), fontsize=7, color="#1f77b4")

ax2 = ax1.twinx()
l2 = ax2.plot(steps_eval, mmlu, "s--", color="#d62728", label="MMLU")
ax2.set_ylabel("MMLU accuracy", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax2.axhline(0.25, color="gray", ls=":", lw=0.8)
ax2.set_ylim(0.24, 0.33)
for s, m in zip(steps_eval, mmlu):
    ax2.annotate(f"{m:.4f}", (s, m), textcoords="offset points", xytext=(0, -12), fontsize=7, color="#d62728")

ax1.axvline(128000, color="gray", ls="-.", lw=0.7)
ax1.text(128500, 11.3, "128k checkpoint", fontsize=7, color="gray")

ax1.set_title("Late keep14 healing: PPL and MMLU")
lines = l1 + l2
ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=7, loc="center left")

fig.tight_layout()
fig.savefig(OUT)
print("wrote", OUT)
