#!/usr/bin/env python3
"""Broad-domain MMLU accuracy and chance-adjusted recovery for Paper B."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER_DIR = HERE.parent
REPO_DIR = PAPER_DIR.parent
DATA_DIR = PAPER_DIR / "data" if (PAPER_DIR / "data").exists() else REPO_DIR / "data"
RAW = DATA_DIR / "raw" / "olmo2_downstream_results"
OUT = HERE / "fig_domain_recovery.pdf"

FILES = {
    "base": RAW / "7B_base_full_know" / "summary.json",
    "full32": RAW / "7B_full32_step25000_know" / "summary.json",
    "keep14": RAW / "7B_keep14_step200000_know" / "summary.json",
    "ShortGPT": RAW / "7B_shortgpt16_step200000_know" / "summary.json",
    "random": RAW / "7B_scratch16L_step200000_know" / "summary.json",
}

GROUPS = {
    "STEM": ["abstract_algebra", "anatomy", "astronomy", "college_biology", "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics", "electrical_engineering", "elementary_mathematics", "high_school_biology", "high_school_chemistry", "high_school_computer_science", "high_school_mathematics", "high_school_physics", "high_school_statistics", "machine_learning", "medical_genetics", "professional_medicine", "virology"],
    "Humanities": ["formal_logic", "high_school_european_history", "high_school_us_history", "high_school_world_history", "international_law", "jurisprudence", "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy", "prehistory", "professional_law", "world_religions"],
    "Social sci.": ["econometrics", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology", "human_sexuality", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy"],
    "Other": ["business_ethics", "clinical_knowledge", "global_facts", "human_aging", "management", "marketing", "miscellaneous", "nutrition", "professional_accounting"],
}

def load(path):
    return json.loads(path.read_text())["tasks"]["mmlu"]["subjects"]

def weighted(subjects, names):
    correct = sum(subjects[s]["n_correct_acc"] for s in names)
    total = sum(subjects[s]["n"] for s in names)
    return correct / total

data = {name: load(path) for name, path in FILES.items()}
models = list(FILES)
groups = list(GROUPS)
accuracy = {m: [weighted(data[m], GROUPS[g]) for g in groups] for m in models}
recovery = {
    m: [(accuracy[m][i] - 0.25) / (accuracy["base"][i] - 0.25) for i in range(len(groups))]
    for m in models if m != "base"
}

colors = {"base": "#222222", "full32": "#2ca02c", "keep14": "#d62728", "ShortGPT": "#1f77b4", "random": "#999999"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.65), gridspec_kw={"width_ratios": [1.18, 1.0]})
x = np.arange(len(groups))
w = 0.16
for i, m in enumerate(models):
    ax1.bar(x + (i - 2) * w, accuracy[m], width=w, color=colors[m], label=m)
ax1.axhline(0.25, color="gray", ls=":", lw=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(groups, rotation=18, ha="right", fontsize=7)
ax1.set_ylim(0.18, 0.76)
ax1.set_ylabel("MMLU accuracy")
ax1.set_title("(a) Sample-weighted broad groups", fontsize=8.5)
ax1.grid(axis="y", alpha=0.18, lw=0.5)

for m in ["full32", "keep14", "ShortGPT", "random"]:
    ax2.plot(groups, recovery[m], marker="o", ms=3.5, lw=1.1, color=colors[m], label=m)
ax2.axhline(0, color="gray", ls=":", lw=0.8)
ax2.axhline(1, color="gray", ls="--", lw=0.7)
ax2.set_ylim(-0.08, 1.08)
ax2.set_ylabel("chance-adjusted recovery")
ax2.set_xticks(x)
ax2.set_xticklabels(groups, rotation=18, ha="right", fontsize=7)
ax2.set_title("(b) Recovery relative to base", fontsize=8.5)
ax2.grid(axis="y", alpha=0.18, lw=0.5)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
