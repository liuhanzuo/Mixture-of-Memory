#!/usr/bin/env python3
"""Generate detailed Paper B appendix tables from merged evaluation summaries."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw"
RAW = RAW_ROOT / "olmo2_downstream_results"
PPL_RAW = RAW_ROOT / "olmo2_ppl_results"
RESULTS = ROOT.parent / "results"
PAIRED = ROOT / "data" / "paired_analysis.json"
SECTIONS = ROOT / "sections"

RUNS = {
    "base": "7B_base_full",
    "keep8": "7B_keep8_step44000",
    "keep10": "7B_keep10_step10000",
    "keep12": "7B_keep12_step111500",
    "keep14a": "7B_keep14_step128000",
    "keep14p": "7B_keep14_step153500",
    "keep14f": "7B_keep14_step200000",
    "frozen": "7B_freezefront_step200000",
    "random": "7B_scratch16L_step200000",
}

CATEGORIES = {
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
    "Social science": [
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


def load_summary(name: str, know: bool = False) -> dict:
    suffix = "_know" if know else ""
    with open(RAW / f"{name}{suffix}" / "summary.json") as f:
        return json.load(f)


def tex(s: str) -> str:
    return s.replace("_", r"\_")


def score(x: float) -> str:
    return f"{x:.3f}".lstrip("0")


def weighted(subjects: dict, names: list[str]) -> tuple[int, float]:
    n = sum(subjects[name]["n"] for name in names)
    correct = sum(subjects[name]["n_correct_acc"] for name in names)
    return n, correct / n


def write_mmlu_tables(know: dict[str, dict]) -> None:
    subjects = {key: value["tasks"]["mmlu"]["subjects"] for key, value in know.items()}
    names = sorted(subjects["base"])
    if len(names) != 57 or any(set(v) != set(names) for v in subjects.values()):
        raise AssertionError("MMLU subject maps are incomplete or inconsistent")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Group} & $\boldsymbol{n}$ & \textbf{base} & \textbf{k8} & \textbf{k10} & \textbf{k12} & \textbf{k14-final} & \textbf{frozen} & \textbf{random} \\",
        r"\midrule",
    ]
    for category, members in CATEGORIES.items():
        n, _ = weighted(subjects["base"], members)
        vals = [weighted(subjects[key], members)[1] for key in ["base", "keep8", "keep10", "keep12", "keep14f", "frozen", "random"]]
        lines.append(f"{category} & {n:,} & " + " & ".join(score(x) for x in vals) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Sample-weighted MMLU broad-group accuracy.} k8, k10, k12, and k14-final denote keep8@44k, keep10@10k, keep12@111.5k, and keep14@200k. Frozen-front and random init are the two other 200k controls. Group definitions follow the standard four-way MMLU taxonomy.}",
        r"\label{tab:app-mmlu-groups}",
        r"\end{table}",
        "",
    ]
    (SECTIONS / "app_tab_mmlu_groups.tex").write_text("\n".join(lines))

    header = (
        r"\textbf{Subject} & $\boldsymbol{n}$ & \textbf{base} & \textbf{k8} & "
        r"\textbf{k12} & \textbf{k14-p} & \textbf{k14-f} & \textbf{frozen} & \textbf{rand} \\"
    )
    halves = [names[:29], names[29:]]
    lines = [r"\begin{table*}[t]", r"\centering", r"\scriptsize"]
    for idx, half in enumerate(halves):
        if idx:
            lines.append(r"\hfill")
        lines += [
            r"\begin{minipage}[t]{.495\textwidth}",
            r"\centering",
            r"\setlength{\tabcolsep}{1.5pt}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{@{}lrrrrrrrr@{}}",
            r"\toprule",
            header,
            r"\midrule",
        ]
        for name in half:
            n = subjects["base"][name]["n"]
            vals = [subjects[key][name]["acc"] for key in ["base", "keep8", "keep12", "keep14p", "keep14f", "frozen", "random"]]
            lines.append(f"{tex(name)} & {n} & " + " & ".join(score(x) for x in vals) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{minipage}"]
    lines += [
        r"\caption{\textbf{Complete 57-subject MMLU results.} All values are accuracy. k14-p and k14-f are inherited keep14 at 153.5k and 200k; frozen-front and random init are the two other 200k controls. Subjects are split alphabetically across the two panels.}",
        r"\label{tab:app-mmlu-full}",
        r"\end{table*}",
        "",
    ]
    (SECTIONS / "app_tab_mmlu_full.tex").write_text("\n".join(lines))


def write_metric_sensitivity(know: dict[str, dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Task} & \textbf{metric} & \textbf{base} & \textbf{k8} & \textbf{k12} & \textbf{k14-p} & \textbf{k14-f} & \textbf{frozen} & \textbf{random} \\",
        r"\midrule",
    ]
    for task, label in [("boolq", "BoolQ"), ("commonsense_qa", "CSQA"), ("social_iqa", "SIQA")]:
        for metric, metric_label in [("acc", "raw"), ("acc_norm", "norm.")]:
            vals = [know[key]["tasks"][task][metric] for key in ["base", "keep8", "keep12", "keep14p", "keep14f", "frozen", "random"]]
            lines.append(f"{label} & {metric_label} & " + " & ".join(score(x) for x in vals) + r" \\")
        if task != "social_iqa":
            lines.append(r"\addlinespace[1pt]")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Raw versus character-normalized accuracy.} The normalized score divides candidate log-likelihood by the raw continuation's character count before selecting an answer. This is distinct from the token-normalized complete-option MMLU protocol. The main results pre-specify raw accuracy for BoolQ, CommonsenseQA, and SocialIQA. Normalization can materially change BoolQ and SIQA, so both views are reported here. k14-p/k14-f denote keep14 at 153.5k/200k; frozen-front and random init are the other 200k controls.}",
        r"\label{tab:app-metric-sensitivity}",
        r"\end{table}",
        "",
    ]
    (SECTIONS / "app_tab_metric_sensitivity.tex").write_text("\n".join(lines))


def write_integrity(core: dict[str, dict], know: dict[str, dict]) -> None:
    display = {
        "base": ("full base", "---", 32, "pretrained"),
        "keep8": ("keep8", "44k", 10, "inherited"),
        "keep10": ("keep10", "10k", 12, "inherited"),
        "keep12": ("keep12", "111.5k", 14, "inherited"),
        "keep14a": ("keep14-a", "128k", 16, "inherited"),
        "keep14p": ("keep14-p", "153.5k", 16, "inherited"),
        "keep14f": ("keep14-f", "200k", 16, "inherited"),
        "frozen": ("frozen-front", "200k", 16, "inherited/frozen"),
        "random": ("random init", "200k", 16, "random"),
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lrrlrr@{}}",
        r"\toprule",
        r"\textbf{Run} & \textbf{step} & \textbf{layers} & \textbf{init.} & \textbf{NaNs} & \textbf{BoolQ trunc.} \\",
        r"\midrule",
    ]
    for key in display:
        name, step, layers, init = display[key]
        n_nan = sum(task["n_nan"] for task in core[key]["tasks"].values()) + sum(task["n_nan"] for task in know[key]["tasks"].values())
        trunc = know[key]["tasks"]["boolq"]["n_trunc"]
        lines.append(f"{name} & {step} & {layers} & {init} & {n_nan} & {trunc} " + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Evaluation-integrity manifest for the 7B results.} NaN denotes a not-a-number output; every task cell retains its full sample count and has zero NaNs. keep14-a/p/f denote the 128k/153.5k/200k checkpoints. The same two overlength BoolQ passages are truncated in every run containing BoolQ; all samples are still scored, and all other tasks have zero truncations. The scratch summaries retain the evaluator's generic \texttt{mode=pruned} reconstruction field, so initialization provenance is taken from the training checkpoint metadata rather than that field.}",
        r"\label{tab:app-integrity}",
        r"\end{table}",
        "",
    ]
    (SECTIONS / "app_tab_integrity.tex").write_text("\n".join(lines))


def paired_rows(points: list[dict]) -> list[str]:
    half = (len(points) + 1) // 2
    left, right = points[:half], points[half:]
    rows = []
    for i, a in enumerate(left):
        b = right[i] if i < len(right) else None
        first = f"{a['layer_idx']} & {a['frac_depth']:.4f} & {a['mmlu_acc']:.3f} & {a['mmlu_correct_ll']:.3f}"
        second = " & & &" if b is None else f"{b['layer_idx']} & {b['frac_depth']:.4f} & {b['mmlu_acc']:.3f} & {b['mmlu_correct_ll']:.3f}"
        rows.append(first + " & " + second + r" \\")
    return rows


def write_logitlens_tables() -> None:
    with open(RESULTS / "knowledge_logit_lens_OLMo-2-1124-7B.json") as f:
        olmo = json.load(f)
    # NOTE: "Qwen3-8b-local" is a symlink to models/Qwen--Qwen3-8b, which is
    # Qwen3-8B-*Instruct* (eos_token_id 151645 <|im_end|>, ctx 40960) and NOT
    # Qwen3-8B-Base (eos 151643, ctx 32768). The label below must say so; the
    # numbers are correct for the Instruct checkpoint that was actually run.
    with open(RESULTS / "knowledge_logit_lens_Qwen3-8b-local.json") as f:
        qwen = json.load(f)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{depths} & \textbf{onset} & \textbf{sat95} & \textbf{sat99} & \textbf{peak} & \textbf{top acc.} \\",
        r"\midrule",
    ]
    for name, data in [("OLMo-2-7B", olmo), ("Qwen3-8B (Instruct)", qwen)]:
        s = data["summary"]
        peak = f"L{s['peak_layer']} ({s['peak_acc']:.3f})"
        lines.append(f"{name} & {len(data['per_layer'])} & L{s['onset_layer']} & L{s['sat95_top_layer']} & L{s['sat99_top_layer']} & {peak} & {s['top_acc']:.3f} " + r"\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Knowledge logit-lens summary diagnostics.} The logit lens applies the model's final normalization and output head to each intermediate hidden state. Onset is the first depth above chance by $.05$; sat95/sat99 are the first depths reaching $95\%$/$99\%$ of the top-layer above-chance signal. Peak and top can differ, as for Qwen3.}",
        r"\label{tab:app-logitlens-summary}",
        r"\end{table}",
        "",
    ]
    lines += [r"\begin{table*}[t]", r"\centering", r"\scriptsize"]
    for idx, (name, data) in enumerate([("OLMo-2-7B", olmo), ("Qwen3-8B (Instruct)", qwen)]):
        if idx:
            lines.append(r"\hfill")
        lines += [
            r"\begin{minipage}[t]{.495\textwidth}",
            r"\centering",
            rf"\textbf{{({chr(97 + idx)}) {name}}}\\[2pt]",
            r"\setlength{\tabcolsep}{1.5pt}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{@{}rrrr@{\hspace{5pt}}rrrr@{}}",
            r"\toprule",
            r"\textbf{L} & \textbf{frac.} & \textbf{acc.} & \textbf{gold LL} & \textbf{L} & \textbf{frac.} & \textbf{acc.} & \textbf{gold LL} \\",
            r"\midrule",
            *paired_rows(data["per_layer"]),
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{minipage}",
        ]
    lines += [
        r"\caption{\textbf{Complete knowledge logit-lens trajectories.} Layer 0 is the embedding/input residual state and layer $L$ follows decoder block $L$. Accuracy is four-letter MMLU accuracy; gold LL is full-vocabulary gold-letter log-likelihood on the same 1,000 questions.}",
        r"\label{tab:app-logitlens-full}",
        r"\end{table*}",
        "",
    ]
    (SECTIONS / "app_tab_logitlens_full.tex").write_text("\n".join(lines))


def write_paired_table() -> None:
    data = json.loads(PAIRED.read_text())
    labels = {
        "mmlu": "MMLU",
        "lambada_openai": "LAMBADA",
        "commonsense_qa": "CSQA",
        "boolq": "BoolQ",
        "social_iqa": "SIQA",
    }
    order = ["mmlu", "lambada_openai", "commonsense_qa", "boolq", "social_iqa"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"\textbf{Task} & $\boldsymbol{n}$ & \textbf{keep14} & \textbf{random} & $\boldsymbol{\Delta}$ \textbf{(pp)} & \textbf{McNemar $p$} & \textbf{paired bootstrap 95\% CI (pp)} \\",
        r"\midrule",
    ]
    for key in order:
        row = data[key]
        lo, hi = row["bootstrap_ci95"]
        p = row["mcnemar_p"]
        if p < 0.001:
            exponent = int(f"{p:.2e}".split("e")[1])
            mantissa = p / (10 ** exponent)
            p_text = rf"{mantissa:.2f}\times10^{{{exponent}}}"
        else:
            p_text = f"{p:.3f}"
        lines.append(
            f"{labels[key]} & {row['n_paired']:,} & {row['keep14_acc']:.4f} & "
            f"{row['random_acc']:.4f} & {100 * row['diff']:+.2f} & "
            f"${p_text}$ & $[{100 * lo:.2f}, {100 * hi:.2f}]$ " + r"\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{\textbf{Paired keep14 versus fully random-init analysis at 200k.} Items are matched by evaluation ID. McNemar uses an exact two-sided binomial test on discordant correctness pairs; confidence intervals use 10,000 paired bootstrap resamples with seed 0. All five intervals exclude zero and all $p<.05$. Frozen-front is excluded because per-example predictions were not retained.}",
        r"\label{tab:app-paired}",
        r"\end{table*}",
        "",
    ]
    (SECTIONS / "app_tab_paired.tex").write_text("\n".join(lines))


def write_shortgpt_table() -> None:
    ppl = json.loads((PPL_RAW / "7B_shortgpt_step0" / "summary.json").read_text())
    core = json.loads((RAW / "7B_shortgpt_step0" / "summary.json").read_text())
    know = json.loads((RAW / "7B_shortgpt_step0_know" / "summary.json").read_text())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"\textbf{Policy / point} & \textbf{PPL} & \textbf{MMLU} & \textbf{LAMBADA} & \textbf{HellaSwag} \\",
        r"\midrule",
        f"ShortGPT, step 0 & {ppl['ppl']:.3f} & {know['tasks']['mmlu']['acc']:.4f} & "
        f"{know['tasks']['lambada_openai']['acc']:.5f} & {core['tasks']['hellaswag']['acc_norm']:.3f} " + r"\\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\caption{\textbf{Immediate damage under a non-contiguous ShortGPT policy.} Cosine block influence selects 16 of the original 32 OLMo layers using 128 Dolmino windows. \texttt{core6} is the unweighted macro-average of HellaSwag, ARC-Challenge, ARC-Easy, PIQA, and OpenBookQA character-normalized accuracy plus WinoGrande raw accuracy. This is the unhealed step-zero model, so it quantifies immediate pruning damage rather than cross-policy recovery after healing.}",
        r"\label{tab:app-shortgpt}",
        r"\end{table}",
        "",
    ]
    (SECTIONS / "app_tab_shortgpt.tex").write_text("\n".join(lines))


def main() -> None:
    core = {key: load_summary(name) for key, name in RUNS.items()}
    know = {key: load_summary(name, know=True) for key, name in RUNS.items()}
    write_mmlu_tables(know)
    write_metric_sensitivity(know)
    write_integrity(core, know)
    write_logitlens_tables()
    write_paired_table()
    write_shortgpt_table()
    print("Generated detailed appendix tables in", SECTIONS)


if __name__ == "__main__":
    main()
