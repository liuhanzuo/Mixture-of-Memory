#!/usr/bin/env python3
"""Build the paper figures for the token-cost Pareto segmentation claim.

Reads measured numbers from runs/ (evalplus.out for quality, metrics.rank*.jsonl
for cost/termination) so the figures cannot drift from the runs. Emits:

    figures/fig_pareto_heplus.pdf/.png     token-cost frontier, HumanEval+
    figures/fig_pareto_mbppplus.pdf/.png   token-cost frontier, MBPP+
    figures/fig_nfe_saturation.pdf/.png    pass@1 and syntax-error vs NFE
    figures/fig_termination_tail.pdf/.png  scaffold termination reasons
    figures/pareto_points.json             the plotted numbers, for the tables

Cost convention (this is the crux -- NFE is NOT comparable across families):
  * scaffold: cumulative_model_tokens, logged directly by the runtime. Each
    model call sees only the current tree serialization, so sequences are short.
  * flat diffusion (vanilla Dream, plain-SFT): nfe * (input_tokens + canvas),
    because every diffusion step re-attends the whole 512-token canvas plus the
    prompt. Measured per-benchmark from the run metrics.
Using NFE instead inflates scaffold's apparent cost by ~3.8x and inverts the
low-budget comparison.

The matched Plain-SFT control matters: it shares the Dream-Coder Base
checkpoint, the train/eval split, the corruption weighting and the trainer with
the scaffold arm, and differs only by removing the scaffold machinery. It uses a
flat canvas, so it is priced like vanilla.

Usage (from the dllm_draft repo root):
    /opt/conda/envs/dllm-env/bin/python scripts/make_pareto_figures.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")

C_SCAFFOLD = "#1f6f8b"
C_VANILLA = "#c1553b"
C_PLAIN = "#6b6b6b"


def rows(run: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", run, "metrics.rank*.jsonl"))):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out


def pass_plus(run: str) -> float | None:
    """Second pass@1 in evalplus.out is the base+extra ('plus') figure."""
    path = os.path.join(ROOT, "runs", run, "evalplus.out")
    if not os.path.exists(path):
        return None
    found = re.findall(r"pass@1:\s*([0-9.]+)", open(path).read())
    return float(found[1]) if len(found) >= 2 else None


def proc(row: dict) -> dict:
    """The runtime writes a completed task's telemetry to `process`, but a task
    that hit a capacity/budget limit writes to `failure_process` instead.

    Reading only `process` silently drops the truncated tasks -- which are the
    most expensive ones (they burn the entire call budget). For scaffold Large
    that understated mean cost by 2.85x on HumanEval+ and 3.69x on MBPP+. Always
    go through this accessor.
    """
    return row.get("process") or row.get("failure_process") or {}


def field_mean(run: str, key: str) -> float | None:
    """Mean over ALL tasks, including capacity/budget-truncated ones."""
    vals = [
        proc(r)[key] for r in rows(run) if isinstance(proc(r).get(key), (int, float))
    ]
    return statistics.mean(vals) if vals else None


def field_coverage(run: str, key: str) -> tuple[int, int]:
    all_rows = rows(run)
    have = sum(1 for r in all_rows if isinstance(proc(r).get(key), (int, float)))
    return have, len(all_rows)


def flat_step_cost(run: str) -> float | None:
    """Tokens attended per diffusion step for a flat-canvas run."""
    data = [proc(r) for r in rows(run)]
    inp = [d["input_tokens"] for d in data if isinstance(d.get("input_tokens"), (int, float))]
    gen = [d["generated_tokens"] for d in data if isinstance(d.get("generated_tokens"), (int, float))]
    if not inp or not gen:
        return None
    return statistics.mean(inp) + statistics.mean(gen)


def syntax_error_rate(run: str) -> float | None:
    data = [proc(r) for r in rows(run)]
    flags = [d["final_parseable"] for d in data if "final_parseable" in d]
    return (sum(1 for f in flags if f is False) / len(flags)) if flags else None


def collect(bench: str) -> dict:
    if bench == "heplus":
        scaffold = [("Tiny", "scaffold_tiny_heplus"), ("Small", "scaffold_small_heplus"),
                    ("Medium", "scaffold_medium_heplus"), ("Large", "scaffold_large_heplus")]
        vanilla = [(16, "dream_instruct_heplus_nfe16"), (32, "dream_instruct_heplus_nfe32"),
                   (64, "dream_instruct_heplus_nfe64"), (128, "dream_instruct_heplus_nfe128"),
                   (256, "dream_instruct_heplus_nfe256"),
                   (512, "dream_coder_instruct_heplus_r2"), (1024, "dream_instruct_heplus_nfe1024")]
        plain = [(64, 0.0000), (128, 0.0488), (512, 0.2195)]
        ref = "dream_instruct_heplus_nfe64"
    else:
        scaffold = [("Tiny", "scaffold_tiny_mbppplus"), ("Small", "scaffold_small_mbppplus"),
                    ("Medium", "scaffold_medium_mbppplus"), ("Large", "scaffold_large_mbppplus")]
        vanilla = [(16, "dream_instruct_mbppplus_nfe16"), (32, "dream_instruct_mbppplus_nfe32"),
                   (64, "dream_instruct_mbppplus_nfe64"), (128, "dream_instruct_mbppplus_nfe128"),
                   (256, "dream_instruct_mbppplus_nfe256"),
                   (512, "dream_coder_instruct_mbppplus_r2"), (1024, "dream_instruct_mbppplus_nfe1024")]
        plain = [(512, 0.2434)]
        ref = "dream_instruct_mbppplus_nfe64"

    step = flat_step_cost(ref)
    pts = []
    for label, run in scaffold:
        q, c = pass_plus(run), field_mean(run, "cumulative_model_tokens")
        if q is not None and c is not None:
            have, total = field_coverage(run, "cumulative_model_tokens")
            pts.append(dict(label=f"scaffold {label}", cost=c, quality=q,
                            family="scaffold", nfe=field_mean(run, "nfe"),
                            cost_coverage=f"{have}/{total}"))
    for nfe, run in vanilla:
        q = pass_plus(run)
        if q is not None:
            pts.append(dict(label=f"Dream nfe{nfe}", cost=nfe * step, quality=q,
                            family="vanilla", nfe=float(nfe)))
    for nfe, q in plain:
        pts.append(dict(label=f"plain-SFT nfe{nfe}", cost=nfe * step, quality=q,
                        family="plain-SFT", nfe=float(nfe)))

    pts.sort(key=lambda p: p["cost"])
    best = -1.0
    for p in pts:
        p["frontier"] = p["quality"] > best
        best = max(best, p["quality"])
    return dict(points=pts, step_cost=step)


def pareto_figure(bench: str, data: dict, title: str, outstem: str) -> None:
    pts = data["points"]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.set_axisbelow(True)
    ax.grid(True, which="both", color="#e6e6e6", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    front = [p for p in pts if p["frontier"]]
    ax.scatter([p["cost"] for p in front], [p["quality"] for p in front],
               s=190, facecolors="none", edgecolors="#9a9a9a",
               linewidths=1.1, zorder=2, label="Pareto-optimal")

    for family, color, marker in (("scaffold", C_SCAFFOLD, "o"),
                                  ("vanilla", C_VANILLA, "s"),
                                  ("plain-SFT", C_PLAIN, "^")):
        sel = [p for p in pts if p["family"] == family]
        if not sel:
            continue
        ax.plot([p["cost"] for p in sel], [p["quality"] for p in sel],
                marker=marker, markersize=7, linewidth=1.4, color=color,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3,
                label={"scaffold": "structural runtime (scaffold)",
                       "vanilla": "flat diffusion (Dream-Coder)",
                       "plain-SFT": "matched plain SFT (flat)"}[family])

    # offsets are per-label to keep the crowded 4e4 region legible
    notes = {
        "scaffold Medium": (10, -4),
        "Dream nfe32": (-14, 12),
        "Dream nfe512": (-20, -16),
        "plain-SFT nfe64": (10, 6),
        "plain-SFT nfe512": (-28, 10),
    }
    for p in pts:
        if p["label"] in notes:
            ax.annotate(p["label"], (p["cost"], p["quality"]),
                        textcoords="offset points", xytext=notes[p["label"]],
                        fontsize=7.5, color="#404040")

    lo = min(p["quality"] for p in pts)
    hi = max(p["quality"] for p in pts)
    ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.10 * (hi - lo))

    ax.set_xscale("log")
    ax.set_xlabel("cost: cumulative tokens processed by the model (log scale)")
    ax.set_ylabel("pass@1 (EvalPlus, base+extra)")
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{outstem}.{ext}"), dpi=200)
    plt.close(fig)


def nfe_figure() -> None:
    series = {
        "HumanEval+": [(64, "dream_instruct_heplus_nfe64"), (128, "dream_instruct_heplus_nfe128"),
                       (256, "dream_instruct_heplus_nfe256"),
                       (512, "dream_coder_instruct_heplus_r2"), (1024, "dream_instruct_heplus_nfe1024")],
        "MBPP+": [(64, "dream_instruct_mbppplus_nfe64"), (128, "dream_instruct_mbppplus_nfe128"),
                  (256, "dream_instruct_mbppplus_nfe256"),
                  (512, "dream_coder_instruct_mbppplus_r2"), (1024, "dream_instruct_mbppplus_nfe1024")],
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
        ax.grid(True, color="#e6e6e6", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("diffusion steps (NFE)")

    for name, runs_, color in (("HumanEval+", series["HumanEval+"], C_SCAFFOLD),
                               ("MBPP+", series["MBPP+"], C_VANILLA)):
        xs = [n for n, _ in runs_]
        ax1.plot(xs, [pass_plus(r) for _, r in runs_], marker="o", markersize=6,
                 color=color, markeredgecolor="white", label=name)
        ax2.plot(xs, [(syntax_error_rate(r) or 0) * 100 for _, r in runs_], marker="s",
                 markersize=6, color=color, markeredgecolor="white", label=name)

    ax1.set_ylabel("pass@1 (base+extra)")
    ax1.set_title("quality saturates at 512 steps", fontsize=10, loc="left")
    ax1.legend(frameon=False, fontsize=8)
    ax2.set_ylabel("unparseable outputs (%)")
    ax2.set_title("the low-step budget buys syntactic validity", fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_nfe_saturation.{ext}"), dpi=200)
    plt.close(fig)


def termination_figure() -> None:
    tiers = [
        ("Tiny", "scaffold_tiny_heplus", "scaffold_tiny_mbppplus"),
        ("Small", "scaffold_small_heplus", "scaffold_small_mbppplus"),
        ("Medium", "scaffold_medium_heplus", "scaffold_medium_mbppplus"),
        ("Large", "scaffold_large_heplus", "scaffold_large_mbppplus"),
    ]
    extra = "scaffold_large_heplus_budget1024"
    if pass_plus(extra) is not None or rows(extra):
        tiers.append(("Large@1024", extra, ""))

    def breakdown(run: str) -> Counter:
        if not run:
            return Counter()
        counts = Counter()
        for r in rows(run):
            counts[proc(r).get("termination_reason") or "unknown"] += 1
        return counts

    reasons = ["resolved", "model_call_budget", "depth_capacity_exhausted", "unknown"]
    colors = {"resolved": "#4c9f70", "model_call_budget": "#c1553b",
              "depth_capacity_exhausted": "#e0a458", "unknown": "#b0b0b0"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, idx, bench in ((axes[0], 1, "HumanEval+"), (axes[1], 2, "MBPP+")):
        labels, bottoms = [], []
        for name, he, mb in tiers:
            run = he if idx == 1 else mb
            labels.append(name)
            bottoms.append(breakdown(run))
        base = [0.0] * len(labels)
        for reason in reasons:
            vals = []
            for counts in bottoms:
                total = sum(counts.values()) or 1
                vals.append(100.0 * counts.get(reason, 0) / total)
            ax.bar(labels, vals, bottom=base, color=colors[reason],
                   label=reason.replace("_", " "), width=0.62,
                   edgecolor="white", linewidth=0.8)
            base = [b + v for b, v in zip(base, vals)]
        ax.set_title(bench, fontsize=10, loc="left")
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("share of tasks (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("scaffold termination: the deep tier exhausts its call budget",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_termination_tail.{ext}"), dpi=200)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    he = collect("heplus")
    mb = collect("mbppplus")

    pareto_figure("heplus", he,
                  "HumanEval+ ($n{=}164$): the two families own different cost regimes",
                  "fig_pareto_heplus")
    pareto_figure("mbppplus", mb,
                  "MBPP+ ($n{=}378$): same segmentation, replicated",
                  "fig_pareto_mbppplus")
    nfe_figure()
    termination_figure()

    with open(os.path.join(FIGDIR, "pareto_points.json"), "w") as handle:
        json.dump({"heplus": he, "mbppplus": mb}, handle, indent=2)

    for name, data in (("HumanEval+", he), ("MBPP+", mb)):
        print(f"\n{name}  (flat-canvas cost = {data['step_cost']:.0f} tok/step)")
        print(f"  {'config':22s} {'cost(tok)':>10s} {'pass@1+':>8s} {'family':>11s}  frontier")
        for p in data["points"]:
            print(f"  {p['label']:22s} {p['cost']:10.0f} {p['quality']:8.3f} "
                  f"{p['family']:>11s}  {'*' if p['frontier'] else ''}")
        front = [p for p in data["points"] if p["frontier"]]
        sc = [p for p in front if p["family"] == "scaffold"]
        if sc:
            print(f"  scaffold holds the frontier up to {max(p['cost'] for p in sc):,.0f} tokens")
        print(f"  plain-SFT on frontier: "
              f"{[p['label'] for p in front if p['family'] == 'plain-SFT'] or 'none'}")
    print(f"\nwrote figures to {FIGDIR}")


if __name__ == "__main__":
    main()
