#!/usr/bin/env python3
"""Appendix MMLU subject analysis from copied raw summary JSONs."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "data" / "raw" / "olmo2_downstream_results"
OUT = HERE / "fig_mmlu_subjects.pdf"

MODELS = {
    "base": "7B_base_full_know",
    "keep14": "7B_keep14_step200000_know",
    "frozen front": "7B_freezefront_step200000_know",
    "random init": "7B_scratch16L_step200000_know",
}


def load_subjects(name):
    p = RAW / name / "summary.json"
    return json.load(open(p))["tasks"]["mmlu"]["subjects"]


data = {k: load_subjects(v) for k, v in MODELS.items()}
subjects = list(data["base"])

# Canonical broad MMLU groupings.
categories = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_medicine", "college_physics", "computer_security",
        "conceptual_physics", "electrical_engineering", "elementary_mathematics",
        "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics",
        "high_school_physics", "high_school_statistics", "machine_learning",
        "medical_genetics", "professional_medicine", "virology",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social sci.": [
        "econometrics", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "global_facts", "human_aging",
        "management", "marketing", "miscellaneous", "nutrition",
        "professional_accounting",
    ],
}


def weighted_acc(model, subs):
    correct = sum(data[model][s]["n_correct_acc"] for s in subs)
    n = sum(data[model][s]["n"] for s in subs)
    return correct / n


colors = {
    "base": "#222222",
    "keep14": "#33a02c",
    "frozen front": "#ff7f00",
    "random init": "#e31a1c",
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                               gridspec_kw={"width_ratios": [1.05, 1.35]})

# Panel A: broad-category weighted accuracy.
cats = list(categories)
x = np.arange(len(cats))
models_bar = ["base", "keep14", "frozen front", "random init"]
w = 0.19
for i, m in enumerate(models_bar):
    ys = [weighted_acc(m, categories[c]) for c in cats]
    ax1.bar(x + (i - 1.5) * w, ys, width=w, label=m, color=colors[m])
ax1.axhline(0.25, color="gray", ls=":", lw=0.9)
ax1.set_xticks(x)
ax1.set_xticklabels(cats, rotation=25, ha="right", fontsize=7)
ax1.set_ylabel("MMLU accuracy")
ax1.set_ylim(0.15, 0.75)
ax1.grid(True, axis="y", ls=":", alpha=0.3)
ax1.set_title("(a) Broad subject groups", fontsize=9)

# Panel B: all 57 subjects, sorted by inherited-vs-random gap.
ordered = sorted(subjects,
                 key=lambda s: data["keep14"][s]["acc"] -
                 data["random init"][s]["acc"],
                 reverse=True)
xx = np.arange(len(ordered))
ax2.plot(xx, [data["keep14"][s]["acc"] for s in ordered],
         color=colors["keep14"], lw=1.2, label="keep14, train all")
ax2.plot(xx, [data["frozen front"][s]["acc"] for s in ordered],
         color=colors["frozen front"], lw=1.1, label="frozen front")
ax2.plot(xx, [data["random init"][s]["acc"] for s in ordered],
         color=colors["random init"], lw=1.1, label="random init")
ax2.axhline(0.25, color="gray", ls=":", lw=0.9)
ax2.set_xlabel("57 subjects sorted by inherited $-$ random-init gap")
ax2.set_ylabel("subject accuracy")
ax2.set_ylim(0.10, 0.60)
ax2.set_xlim(0, len(ordered) - 1)
ax2.set_xticks([])
ax2.grid(True, axis="y", ls=":", alpha=0.3)
ax2.set_title("(b) Subject-level heterogeneity", fontsize=9)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7,
           frameon=False, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.91))
fig.savefig(OUT)
print("wrote", OUT)
