#!/usr/bin/env python3
"""Build the SparseForge-vs-{dense,CAST-repro,AST-official,Wanda} same-harness table.

Every number here comes from lm-eval 0.4.8 @ git b86c479 on node .21 with a
byte-identical invocation (dtype=bfloat16, parallelize=True, add_bos_token=False,
--batch_size auto, --num_fewshot 0, --seed 0, no chat template); only
`pretrained` differs. PPL comes from baselines/eval_hf_sparse_model.py on the
same 335,872 WikiText-2 target tokens, reported at BOTH 2048 and 4096 because
the two published SparseForge sources disagree on which window they used.

Emits outputs/cast_eval_spec/sparseforge_5b/sparseforge_same_harness_table.json
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/apdcephfs_wzc1/share_304376610/pighzliu_code")
U9 = ROOT / "outputs/cast_eval_spec_union9"
SF = ROOT / "outputs/cast_eval_spec/sparseforge_5b"
PPL2048 = ROOT / "outputs/cast_eval_spec_ppl2048"
PPL4096_SF = ROOT / "outputs/cast_eval_spec_ppl4096_sf"

UNION9 = ("boolq", "rte", "hellaswag", "race", "piqa",
          "winogrande", "arc_easy", "arc_challenge", "openbookqa")
AST7 = ("boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa")
CAST7 = ("hellaswag", "race", "piqa", "winogrande", "arc_easy", "arc_challenge", "openbookqa")

# Pre-existing four arms (task #243) + the three SparseForge variants (task #244).
ARMS = {
    "dense_ref":              U9 / "dense_ref/zeroshot_union9.json",
    "cast_repro_7500":        U9 / "cast_7500/zeroshot_union9.json",
    "ast_official":           U9 / "ast_official/zeroshot_union9.json",
    "wanda":                  U9 / "wanda/zeroshot_union9.json",
    "sparseforge_hard_drop":  SF / "hard_drop/zeroshot_union9.json",
    "sparseforge_soft_fold":  SF / "soft_fold/zeroshot_union9.json",
    "sparseforge_hard_fold":  SF / "hard_fold/zeroshot_union9.json",
}

# PPL @2048: the three pre-existing sparse arms live in cast_eval_spec_ppl2048;
# dense/AST 2048 values are the SPEC.md:202 measured table; SparseForge variants
# were measured by _sparseforge_same_harness_21.sh into SF/<variant>/.
PPL2048_FILES = {
    "cast_repro_7500": PPL2048 / "cast_7500/ppl_metrics.json",
    "wanda": PPL2048 / "wanda/ppl_metrics.json",
    "sparseforge_hard_drop": SF / "hard_drop/ppl_metrics.json",
    "sparseforge_soft_fold": SF / "soft_fold/ppl_metrics.json",
    "sparseforge_hard_fold": SF / "hard_fold/ppl_metrics.json",
    "ast_official": ROOT / "rebuttal_artifacts/2026-07-27/ast_official/ppl_metrics.json",
}
PPL2048_LITERAL = {"dense_ref": 5.563655320068427}  # SPEC.md:204, same harness/tokens

PPL4096_FILES = {
    a: PPL4096_SF / f"{a}/ppl_metrics.json" for a in
    ("sparseforge_hard_drop", "sparseforge_soft_fold", "sparseforge_hard_fold",
     "dense_ref", "ast_official", "cast_7500", "wanda")
}

VARIANT_NOTE = {
    "sparseforge_hard_drop": (
        "EXACT 2:4 (verified zero_frac 0.500000000, 0/1,619,001,344 bad tiles, 224 "
        "tensors). SLoRB low-rank branch REMOVED. This is the only SparseForge "
        "variant that is apples-to-apples with the other four arms."),
    "sparseforge_hard_fold": (
        "2:4 support + SLoRB folded into W. NOT 2:4 (dense after folding). This is "
        "the protocol behind the PUBLISHED SparseForge numbers: "
        "outputs/paper_v2/ast7_eval/sparseforge_5b_table2/eval.log shows "
        "'Set hardening_x=0 ... using hard masks' + 'Keeping sparse_forward mode "
        "(SLoRB enabled)'. Reproduces the checkpoint's own CAST-7 anchor 57.2672 "
        "to +0.0078 pp, which validates this whole export/eval pipeline."),
    "sparseforge_soft_fold": (
        "continuous mask + SLoRB folded in. NOT 2:4. Ablation showing the mask "
        "hardening step itself costs almost nothing once SLoRB is present."),
}


def load(p: Path):
    if not p.exists():
        return None
    return json.load(open(p))


def main() -> int:
    per_task = {}
    slices = {}
    for arm, path in ARMS.items():
        b = load(path)
        if b is None:
            print(f"[table] MISSING {arm}: {path}")
            continue
        got = set(b["per_task"])
        if got != set(UNION9):
            raise SystemExit(f"{arm} does not have exactly the union-9 tasks: {sorted(got)}")
        per_task[arm] = b["per_task"]
        slices[arm] = {k: b[k] for k in ("union9", "cast7", "ast7")}

    ppl2048 = dict(PPL2048_LITERAL)
    for a, p in PPL2048_FILES.items():
        d = load(p)
        if d is None:
            continue
        assert d["seqlen"] == 2048, f"{p} is not seqlen 2048"
        ppl2048[a] = d["wikitext2_ppl"]

    ppl4096 = {}
    for a, p in PPL4096_FILES.items():
        d = load(p)
        if d is None:
            continue
        assert d["seqlen"] == 4096, f"{p} is not seqlen 4096"
        key = "cast_repro_7500" if a == "cast_7500" else a
        ppl4096[key] = d["wikitext2_ppl"]

    # RTE integrality check: acc must be k/277 for integer k.
    rte = {}
    for arm, pt in per_task.items():
        n = pt["rte"]["n_samples"]
        acc = pt["rte"]["acc"]
        k = acc * n
        rte[arm] = {"n": n, "acc": acc, "k_float": k, "k_int": round(k),
                    "is_integer_k": abs(k - round(k)) < 1e-6}
        if not rte[arm]["is_integer_k"]:
            raise SystemExit(f"{arm} RTE acc {acc} is not k/{n} for integer k -- scoring is suspect")

    def plain(arm, tasks):
        return sum(per_task[arm][t]["acc"] for t in tasks) / len(tasks) * 100

    headline = {
        arm: {
            "ast7_plain_acc": plain(arm, AST7),
            "cast7_plain_acc": plain(arm, CAST7),
            "union9_plain_acc": plain(arm, UNION9),
            "ast7_primary": slices[arm]["ast7"]["mean_primary"] * 100,
            "union9_primary": slices[arm]["union9"]["mean_primary"] * 100,
            "wiki_ppl_2048": ppl2048.get(arm),
            "wiki_ppl_4096": ppl4096.get(arm),
        }
        for arm in per_task
    }

    out = {
        "generated": "2026-08-11",
        "task": "#244 -- put SparseForge's 5B headline checkpoint on the CAST-repro harness",
        "node": ".21 (8x L20A, wzc1)",
        "harness": {
            "lm_eval_pip_version": "0.4.8",
            "git_hash": "b86c479",
            "dtype": "bfloat16",
            "add_bos_token": False,
            "chat_template": None,
            "num_fewshot": 0,
            "batch_size": "auto",
            "seed": 0,
            "parallelize": True,
            "note": "byte-identical across all seven rows; only `pretrained` differs",
        },
        "ppl_harness": {
            "script": "baselines/eval_hf_sparse_model.py",
            "corpus": "data/wikitext/wikitext-2-raw-v1/wiki.test.raw",
            "target_tokens": 335872,
            "note": "335,872 = 164x2048 = 82x4096, so 2048 and 4096 score the same token budget",
        },
        "source_checkpoint": {
            "path": str(ROOT / "out_llama/models_Llama--Llama2-7b_mask-unstructured_s0.5_m-hessian_obd_20260413_201320/model_best_lm_eval.pt"),
            "iter_num": 17900,
            "finalization_done": True,
            "saved_weights_are_dense": True,
            "mask_is_continuous": True,
            "mask_range": [7.958078640513122e-11, 1.0],
            "hard_projection_used": "nm_2_4 exact top-2-per-4 (sparse_modeling.py:594-621)",
            "threshold_0p5_would_misproject_tiles": 26726,
            "slorb_branch": {
                "active": True,
                "SLoRB_Weight_elements": 404750336,
                "x_proj_elements": 443678720,
                "extra_live_params_vs_pure_2of4": 848429056,
                "pct_extra_vs_surviving_weights": 26.2,
                "fro_ratio_vs_masked_weight_range": [0.204, 0.469],
                "pct_energy_on_pruned_positions": 50.0,
                "x_proj_trained_away_from_fixed_blocksum": True,
            },
        },
        "variant_notes": VARIANT_NOTE,
        "per_task_plain_acc": {
            arm: {t: per_task[arm][t]["acc"] * 100 for t in UNION9} for arm in per_task
        },
        "per_task_full": per_task,
        "slice_means": slices,
        "headline": headline,
        "rte_integrality": rte,
        "wiki_ppl_2048": ppl2048,
        "wiki_ppl_4096": ppl4096,
    }

    # --- the actual comparison the task asked for -------------------------
    comp = {}
    for sf in ("sparseforge_hard_drop", "sparseforge_hard_fold"):
        if sf not in headline:
            continue
        comp[sf] = {
            base: round(headline[sf]["ast7_plain_acc"] - headline[base]["ast7_plain_acc"], 4)
            for base in ("dense_ref", "cast_repro_7500", "ast_official", "wanda")
            if base in headline
        }
    out["ast7_plain_acc_gap_vs_baselines"] = comp

    outp = SF / "sparseforge_same_harness_table.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[table] wrote {outp}\n")

    order = ["dense_ref", "sparseforge_hard_fold", "sparseforge_soft_fold",
             "ast_official", "cast_repro_7500", "sparseforge_hard_drop", "wanda"]
    print("PLAIN-ACC (acc,none), one harness, 9 tasks")
    hdr = f"{'arm':24s} " + " ".join(f"{t[:9]:>9s}" for t in UNION9)
    print(hdr)
    for a in order:
        if a not in per_task:
            continue
        print(f"{a:24s} " + " ".join(f"{per_task[a][t]['acc']*100:9.4f}" for t in UNION9))
    print()
    print(f"{'arm':24s} {'AST-7':>8s} {'CAST-7':>8s} {'UNION-9':>8s} {'PPL2048':>9s} {'PPL4096':>9s} {'2:4':>5s}")
    is24 = {"dense_ref": "no", "cast_repro_7500": "YES", "ast_official": "YES", "wanda": "YES",
            "sparseforge_hard_drop": "YES", "sparseforge_soft_fold": "no",
            "sparseforge_hard_fold": "no"}
    for a in order:
        if a not in headline:
            continue
        h = headline[a]
        p2 = f"{h['wiki_ppl_2048']:9.4f}" if h["wiki_ppl_2048"] else "       --"
        p4 = f"{h['wiki_ppl_4096']:9.4f}" if h["wiki_ppl_4096"] else "       --"
        print(f"{a:24s} {h['ast7_plain_acc']:8.4f} {h['cast7_plain_acc']:8.4f} "
              f"{h['union9_plain_acc']:8.4f} {p2} {p4} {is24[a]:>5s}")
    print()
    for a in order:
        if a in rte:
            r = rte[a]
            print(f"RTE {a:24s} acc={r['acc']:.10f} = {r['k_int']}/{r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
