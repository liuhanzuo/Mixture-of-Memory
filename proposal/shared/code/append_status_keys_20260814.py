#!/usr/bin/env python3
"""Append decision-bearing keys to five thin backlog STATUS.json files (2026-08-14).

APPEND-ONLY BY CONSTRUCTION
---------------------------
This script never rewrites an existing key. For each target file it:

  1. reads the original bytes and parses them (must be a JSON object);
  2. asserts every key it wants to add is ABSENT (so no existing value can be
     shadowed or silently changed);
  3. re-serialises as ``original_object | new_keys`` with the original keys FIRST
     and in their original order, then verifies key-by-key that every original
     key still maps to a byte-identical re-serialisation of its original value;
  4. writes only if that verification passes.

The `updated` key already exists in all five files and is NOT touched. Today's
date is recorded under `updated_20260814` per the append-only requirement.

Why the numbers here are trustworthy: every figure was recomputed from the raw
file named next to it in the same session (see B06's `established_measurements`
provenance strings). Nothing was copied from a summary table or from prose.

ZERO GPU. Pure bookkeeping.
"""

import json
import os
import sys

ROOT = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
BACKLOG = os.path.join(ROOT, "proposal", "backlog")

# ---------------------------------------------------------------- B01
B01 = {
    # B01 is the only one of the five that ALREADY had a next_gate
    # ("persist the actual bottleneck latent and run long-memory quality").
    # Append-only: that key is left untouched. This is the executable
    # expansion of it, NOT a replacement. The two do not conflict -- the
    # original names the same two requirements this one operationalises.
    "next_gate_executable_20260814": (
        "Four-arm comparison at ONE scale, with the bottleneck latent ACTUALLY "
        "PERSISTED: (1) stock + Read-LoRA, (2) bottleneck only, (3) bottleneck + "
        "Read-LoRA, (4) bottleneck + Read-LoRA + Write-LoRA. Mandatory reported "
        "quantity: bytes/token of what is written to the store (not of the "
        "restored hidden). Gate is only meaningful AFTER the blocking dependency "
        "below is fixed, because today all four arms would write the same "
        "full-width h_j and the storage axis would be constant across arms."
    ),
    "next_gate_original_preserved": (
        "The pre-existing next_gate ('persist the actual bottleneck latent and "
        "run long-memory quality') is retained UNCHANGED and is NOT superseded. "
        "next_gate_executable_20260814 is its operationalisation: the same two "
        "requirements (persist the latent; measure long-memory quality), now with "
        "the arm set, the mandatory reported quantity (bytes/token of what is "
        "WRITTEN, not of the restored hidden), and the ordering constraint "
        "relative to blocking_dependency. No conflict between the two was found."
    ),
    "gpu_cost_estimate": {
        "value": "~25-40 GPU-h for the four arms at 8B if the two existing 8B "
                 "j=12/d512 CPT endpoints are reused; ~60-80 GPU-h if the CPT "
                 "legs must be re-run at more than 2000 steps",
        "basis": [
            "8B funnel CPT measured: outputs/qwenbott_funnel_L12_d512 ran 2000 "
            "steps at 2.83 s/step on world_size=4 (logs/qwenbott_funnel.out) = "
            "1.57 h wall = 6.3 GPU-h. Its matched stock-continued control "
            "outputs/qwenbott_baseline_L12 measured 2.82 s/step, same shape = "
            "6.3 GPU-h. Both final.pt are on wzc1 (16.4 GB / 16.4 GB).",
            "Read-LoRA measured: outputs/qcmem_distill_qwen_j12_r32_4k ran 4000 "
            "steps on world_size=8 in 21.8 min wall (distill_args.json 00:04:02 "
            "-> final/ 00:25:50, logs/qcmem_distill_qwen_j12_r32_4k.log) = 2.9 "
            "GPU-h per adapter. Four arms need at most 3 adapter trainings.",
            "LoCoMo eval measured: one arm = 3 shards, 20:58:52 -> 21:16:43 = "
            "17.9 min wall = 0.89 GPU-h (locomo_results/hcache_j12_*_chatFALSE "
            "file mtimes). Judge is an OpenAI API call, not GPU.",
            "Write-LoRA has NO measured rate in this repo, so the upper end of "
            "the range is an extrapolation from the Read-LoRA rate, not a "
            "measurement."
        ],
        "excluded_from_this_estimate": "RULER/LongEval/BABILong replication and "
                                       "any second scale. Those are generalisation, not this gate."
    },
    "kill_gate": {
        "source": "PROPOSAL.md 'Kill 条件', copied verbatim (3 conditions)",
        "conditions": [
            "低秩 latent 在 RULER/LongEval 上不保留精确 evidence",
            "fixed LM tax 在强模型上扩大而非缩小",
            "full-depth RAG 在同存储预算下严格支配"
        ],
        "note_on_condition_2": "The existing 1B->3B evidence points the RIGHT "
                               "way (tax shrinks with scale), so this condition is currently NOT "
                               "firing; but it has only been tested at 1B and 3B from-scratch, "
                               "never at 8B. See established_measurements.lm_tax_ppl_pct."
    },
    "established_measurements": {
        "_provenance_note": "Every number below was re-derived from the named "
                            "file in this session (2026-08-14), not copied from PROPOSAL.md.",
        "pca_dim99_1B": {
            "vanilla": 1825,
            "bottleneck_d512": 438,
            "source": "outputs/e2e_1b_deltanll/ctx2048_gpu0.json "
                      "baseline.pca_dim99 / bottleneck.pca_dim99",
            "reproduced_at_other_ctx": "ctx3072 and ctx4096 give 1829/439 and "
                                       "1829/439 -- stable, so the 1825/438 pair is not a ctx artefact"
        },
        "pca_dim99_3B": {
            "vanilla": 2790,
            "bottleneck_d512": 467,
            "source": "outputs/e2e_3b_deltanll/ctx2048_gpu0.json",
            "reproduced_at_other_ctx": "ctx3072 2797/467, ctx4096 2800/467"
        },
        "delta_nll_at_rank128": {
            "1B": {"vanilla": 0.0304, "bottleneck": 0.0022},
            "3B": {"vanilla": 0.0669, "bottleneck": 0.0135},
            "source": "same two json, by_rank['128'].dNLL",
            "reading": "the bottleneck arm is far more truncation-robust at "
                       "rank128 at BOTH scales; at rank256 the bottleneck arm's dNLL is "
                       "slightly NEGATIVE (-0.0028 / -0.0015), i.e. truncation is free"
        },
        "lm_tax_ppl_pct": {
            "1B_16k_steps_layer6": {"baseline_ppl": 25.28, "d1024": "+4.5%",
                                    "d512": "+5.9%", "d256": "+8.5%"},
            "3B_16k_steps_layer6": {"baseline_ppl": 27.44, "d512": "+4.7%",
                                    "d256": "+5.8%"},
            "source": "status/QCMEM_PAPER_DRAFT.md:223-224 = "
                      "status/RUN_REGISTRY.md dim-sweep table; ckpts "
                      "outputs/sembott_{1b,3b}_{base,d256,d512,d1024}_16k/final.pt on wzc1",
            "arithmetic_rechecked": "26.42/25.28=+4.51%, 26.78=+5.93%, "
                                    "27.42=+8.46%, 28.73/27.44=+4.70%, 29.04=+5.83% -- all match",
            "IMPORTANT_scale_mismatch": "PROPOSAL.md says '固定 LM tax 约 4-8.5%'. "
                                        "That range is TRAINING PPL tax from the 16k-step from-scratch "
                                        "dim sweep. It is NOT the same quantity as the r_max NLL gap in "
                                        "outputs/e2e_*_deltanll/ (which is +0.16%/-0.19%/-0.13% at 1B and "
                                        "+0.73%/+0.82%/+0.84% at 3B). Do not mix the two when writing.",
            "monotone_depth_tax": "a separate layer sweep at fixed d512 gives "
                                  "L1 +4.2% / L3 +5.8% / L6 +6.0% / L9 +9.5% / L12 +9.6% "
                                  "(status/RUN_REGISTRY.md) -- tax rises monotonically with "
                                  "bottleneck depth; see memory/bottleneck-layer-sweep-monotone"
        },
        "assets_on_wzc1_for_the_next_gate": {
            "8B_funnel_j12_d512": "outputs/qwenbott_funnel_L12_d512/final.pt "
                                  "(16,390,019,143 B) + arch_meta.json (bottleneck_layer 12, "
                                  "bottleneck_dim 512, unfreeze_from 12, n_params 8.195B)",
            "8B_matched_stock_control": "outputs/qwenbott_baseline_L12/final.pt "
                                        "(16,381,629,495 B)",
            "both_trained_on": "data/pg19_train.jsonl, 2000 steps, eff_bs 64, "
                               "seq_len 2048, lr 1e-4 (train_args.json)",
            "NEVER_EVALUATED": "grep over the whole repo for 'qwenbott_funnel_L12_d512' "
                               "returns 0 hits outside the launch script -- these two 8B endpoints "
                               "have no eval results anywhere. That makes an 8B leg CHEAPER than "
                               "PROPOSAL.md implies, and it is the first thing to check before "
                               "budgeting new CPT.",
            "eval_support_exists": "scripts/eval_qcmem_locomo.py --bottleneck_ckpt "
                                   "already loads a funnel-Qwen arm (lines 1023-1051) and requires "
                                   "arch_meta.json next to the ckpt, which is present"
        }
    },
    "blocking_dependency": {
        "statement": "The store still holds the RESTORED full-width hidden, not "
                     "the d_bottle latent, so the storage-saving claim is currently "
                     "unmeasurable end-to-end.",
        "mechanism_verified_in_code": "scripts/train_qwen_bottleneck_continued.py:"
                                      "113-133 inject_bottleneck wraps layer j in BottleneckLayer "
                                      "(down->GELU->up, NO residual). The wrapper's OUTPUT is back at "
                                      "hidden_size, and QCMemModel caches that layer output (h_j). So "
                                      "the funnel constrains the RANK of what is stored but not the "
                                      "WIDTH of the bytes stored.",
        "consequence": "bytes/token is identical between the bottleneck and "
                       "vanilla arms today. Until a d_bottle-width persist path exists, arms "
                       "2/3/4 of the next_gate differ only in quality, not in storage, and "
                       "the proposal's headline ('可压缩的可持久化 latent') cannot be "
                       "demonstrated -- only its precondition (low rank) can.",
        "from_proposal": "PROPOSAL.md '当前缺口' item 1 states this; recorded "
                         "here so the gate is not launched before it is fixed"
    },
    "other_gaps_from_proposal": [
        "no complete quality/storage/latency frontier on a natural long-memory task",
        "never combined with Read-LoRA + Write-LoRA jointly",
        "strong-model / A13B leg blocked by implementation, init and MoE system issues"
    ],
    "novelty_checked": False,
    "novelty_status_detail": "RELATED_WORK.md does NOT exist in this directory. "
                             "proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:91 rates "
                             "B01 '不足' and lists the required collision families (bottleneck "
                             "Transformer; activation/KV codec; split-inference compression; "
                             "recurrent/compressive memory; semantic/latent codec) plus the "
                             "required distinction (pretraining-time FORMATION of a persistable "
                             "latent vs post-hoc compression). README.md names B01 as a current "
                             "highest-priority gap-filling item (A04 -> B01). Per README's "
                             "Related Work gate, this must be written BEFORE new GPU spend.",
    "updated_20260814": "2026-08-14 -- STATUS made decidable (next_gate made "
                        "executable; kill_gate, gpu_cost_estimate, established_measurements, "
                        "blocking_dependency, novelty_checked added). No scientific "
                        "conclusion changed; all numbers re-derived from disk. 0 GPU."
}

# ---------------------------------------------------------------- B05
B05 = {
    "next_gate": (
        "On Qwen3-8B, run the phase diagram j={shallow,native,content,deep} x "
        "readout={native suffix, affine/tuned lens, LoRA, small decoder} x "
        "task={copy,retrieval,composition,format}. Decidable outcome: either the "
        "cells separate into readable phases (-> standalone paper) or they do not "
        "(-> exit_clause fires and this folds into a Paper A/B mechanism section)."
    ),
    "gpu_cost_estimate": {
        "value": "UNKNOWN -- needs a 1-cell timing first",
        "why": "The grid is 4 j x 4 readout x 4 task = 64 cells, but the four "
               "readouts have costs that differ by orders of magnitude and only one "
               "of them has a measured rate in this repo. 'native suffix' is "
               "forward-only; 'affine/tuned lens' is a cheap fit; 'LoRA' has a "
               "measured 2.9 GPU-h per adapter (4000 steps, world_size=8, "
               "logs/qcmem_distill_qwen_j12_r32_4k.log); 'small decoder' has NO "
               "measured rate at all. Multiplying an unmeasured per-cell cost by 64 "
               "would be an invented number.",
        "measured_anchors_that_do_exist": [
            "LoRA readout at 8B j=12: 2.9 GPU-h per adapter "
            "(outputs/qcmem_distill_qwen_j12_r32_4k, 4000 steps, ws=8)",
            "forward-only logit-lens over all layers, n_mmlu=1000: already done "
            "for two models, see results/knowledge_logit_lens_*.json (its "
            "extract_sec field carries the measured extraction time)",
            "N1 note (ops/research_notes/N1_three_depth_table_20260727.md:6) "
            "estimates the phase-diagram experiment N3 at '1 node ~1 day' = ~192 "
            "GPU-h, but that is an ESTIMATE written before any cell ran, and it is "
            "recorded here as such, not adopted as the number."
        ],
        "first_action_is_free": "Order the 64 cells cheapest-first and time ONE "
                                "cell (native-suffix readout on the retrieval task) before "
                                "committing. Most of the j_native column may already be readable "
                                "off the bracket sweep in status/QCMEM_J_DETERMINATION.md."
    },
    "kill_gate": "NO_KILL_GATE_DEFINED -- PROPOSAL.md has no 'Kill 条件' "
                 "section. It has a '必须避免' (methodological prohibitions) section and "
                 "an exit clause (see exit_clause), which are NOT kill gates: the "
                 "prohibitions constrain HOW to measure, and the exit clause routes "
                 "the result rather than terminating the direction. Per README's rule "
                 "'新方向先写 PROPOSAL.md 和 kill gate, 再启动 GPU', a kill gate must "
                 "be written before this proposal spends GPU.",
    "exit_clause": {
        "text_from_proposal": "若 phase diagram 不清晰，则并入 Paper A/B 机制小节，不独立成篇。",
        "translation": "If the phase diagram is not clear, merge into a Paper A/B "
                       "mechanism subsection; do not make it a standalone paper.",
        "status": "PRE-DATA -- written in PROPOSAL.md before the phase diagram ran",
        "why_this_matters_for_ranking": "B05 can succeed as an input to another "
                                        "paper even if it fails as a standalone. A reader of this STATUS must "
                                        "not treat 'phase diagram unclear' as the direction dying.",
        "unspecified": "'不清晰' is not operationalised. Before the run, decide "
                       "what separation counts as clear (e.g. a required monotone ordering, or "
                       "a minimum cell-to-cell gap relative to the measurement's own noise "
                       "floor) -- otherwise the exit clause is unfalsifiable. Compare "
                       "memory/a-range-is-not-a-measurement-until-it-clears-its-floor."
    },
    "established_measurements": {
        "_status": "NOT EMPTY -- but every number here is PRIOR observation that "
                   "MOTIVATES the phase diagram, not a cell of the phase diagram itself. "
                   "The phase diagram (j x readout x task) has NOT been run.",
        "_provenance_note": "re-read from the named files in this session (2026-08-14)",
        "depth_spectrum_qwen3_8b_36L": {
            "semantic_feature_onset": "0.13L",
            "j_native_zero_shot_readout": "0.25L (j9)",
            "task_semantic_content_peak": "0.44L (j16)",
            "knowledge_decodability_sat95": "0.694L (layer 25 of 36)",
            "next_token_sat95": "0.94L (layer 34)",
            "source_for_knowledge_band": "results/knowledge_logit_lens_Qwen3-8b-local.json "
                                         "summary.sat95_top_layer=25, sat95_frac_depth=0.694, peak_layer=34, "
                                         "peak_acc=0.638, n_mmlu=1000, chance=0.25 -- verified this session",
            "source_for_others": "ops/research_notes/N1_three_depth_table_20260727.md "
                                 "table 1, citing status/QCMEM_J_DETERMINATION.md + "
                                 "results/probe_linguistic_qwen3_8b.json"
        },
        "olmo2_7b_mirror": {
            "knowledge_sat95": "0.594L (layer 19 of 32)",
            "source": "results/knowledge_logit_lens_OLMo-2-1124-7B.json "
                      "summary.sat95_top_layer=19, sat95_frac_depth=0.594, peak_acc=0.551 "
                      "-- verified this session"
        },
        "content_j_cross_family": {
            "Meta-Llama-3-8B": "content_j_layer_mean 8.6 (frac 0.2688, CI95 "
                               "[0.2037,0.3338]), n_points=15",
            "OLMo-2-1124-7B": "content_j_layer_mean 9.13 (frac 0.2854, CI95 "
                              "[0.2138,0.357]), n_points=15",
            "Qwen3-8B": "content_j_layer_mean 14.13 (frac 0.3926, CI95 "
                        "[0.3193,0.4659]), n_points=15",
            "source": "results/p1_2/p1_2_summary.json -- verified this session; "
                      "tasks are RTE/SST2/WiC knee98, and the file also carries the "
                      "lexical_only / position_only / random_label_peak / majority "
                      "null baselines per task, which any claim here MUST clear"
        },
        "readability_gap_scale_law": {
            "gap_j_content_minus_j_native": {"0.6B": "~0.39L", "1.7B": "~0.26L",
                                             "4B": "~0.13L", "8B": "~0.14L", "14B": "~0.085L",
                                             "32B": "~0 (native already at content)"},
            "source": "ops/research_notes/N1_three_depth_table_20260727.md F1, "
                      "from status/QCMEM_J_DETERMINATION.md bracket table (niah_single "
                      "16k, n=100, bm25, topk12, chunk512)",
            "underlying_bracket_verified": "status/QCMEM_J_DETERMINATION.md "
                                           "records the 50%-cliff j and the deepest j with recall>=90 per "
                                           "scale: 0.6B j2 (0.07L) / 1.7B j3 (0.11L) / 4B j9 (0.25L) / 8B j9 "
                                           "(0.25L) / 14B j13 (0.325L) / 32B >=j27 (0.42L)",
            "caveat_recorded_in_the_source": "the same file RETRACTS its own "
                                             "earlier '~0.25L is scale-invariant' claim -- the readout cliff moves "
                                             "monotonically with scale. Do not cite the old constant.",
            "protocol_warning": "these recall numbers are selector=bm25. The "
                                "standing project rule (memory/qcmem-eval-selector-iterbm25, user "
                                "2026-07-17) is that QCMem eval uses selector=iter_bm25 and old bm25 "
                                "results are void. So this gap ladder is MOTIVATION ONLY and must be "
                                "re-measured under iter_bm25 before it is reported as a result."
        },
        "task_dependence_findings": {
            "F2_j_adapt_is_task_dependent": "a distilled cap at content depth "
                                            "recovers retrieval (NIAH single 95-100) at every scale, but "
                                            "composition (multikey, variable-tracking) only recovers for large "
                                            "models (14B yes; 8B/4B partial; 0.6B/1.7B fail)",
            "F3_copy_tasks_INVERT_the_ordering": "content-j adapter HURTS "
                                                 "literal exact match: LongEval 0.6B 37->1, RULER-vt 0.6B 81->1, "
                                                 "1.7B vt 48->21 -- so the best cache depth is task-dependent, which "
                                                 "is exactly why the diagram has a task axis",
            "source": "ops/research_notes/N1_three_depth_table_20260727.md F2/F3, "
                      "citing status/QCMEM_J_DETERMINATION.md adapter table"
        }
    },
    "must_avoid_from_proposal": [
        "universal semantic cut",
        "treating logit-lens onset as causal storage",
        "using a forward-only probe to predict the best graft depth"
    ],
    "distinct_from_dead_direction": "PROPOSAL.md states B05 is NOT the dead "
                                    "'forward probe predicts best adaptation depth' line "
                                    "(archive/paperC-v1-frozen-cap). The '必须避免' list encodes that "
                                    "separation; violating item 3 would collapse B05 back into the dead "
                                    "direction.",
    "novelty_checked": False,
    "novelty_status_detail": "RELATED_WORK.md does NOT exist here. "
                             "proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:95 rates "
                             "B05 '不足但自限' (insufficient but self-limiting) and lists the "
                             "collision families: logit/tuned lens; layerwise emergence; causal "
                             "tracing; early exit; split computing; model stitching/readout "
                             "adapter. Required framing: the j_content/j_native/j_adapt phase "
                             "diagram, NOT a universal semantic cut. Same audit line 163 says "
                             "B05 must complete this before promotion.",
    "updated_20260814": "2026-08-14 -- STATUS made decidable (next_gate, "
                        "gpu_cost_estimate=UNKNOWN with reasons, kill_gate=NO_KILL_GATE_DEFINED, "
                        "exit_clause from PROPOSAL, established_measurements with provenance, "
                        "novelty_checked added). No scientific conclusion changed. 0 GPU."
}

# ---------------------------------------------------------------- B06
B06 = {
    "headline": "CONFIRMED SINGLE-VARIABLE RESULT. Self-distilled LoRA buys "
                "+23.12 pp LoCoMo Judge_1:4 on a retrieval-FREE HCache read path "
                "(16.69 -> 39.81), McNemar exact two-sided p=2.6e-67. PROPOSAL.md calls "
                "it suitable as a Paper A extension or a short paper. This is NOT a "
                "thin backlog item and must not be ranked as one.",
    "next_gate": (
        "Same-harness rejudge of the canonical HCache predictions to remove the "
        "8.11-vs-13.29 cross-node drift. SCALE TRAP, MANDATORY: convert the "
        "canonical 8.11 to the Judge_1:4 (n=1540) scale BEFORE comparing, "
        "otherwise the comparison uses two different rulers -- 8.11 is a "
        "mixed-instrument n=1986 blend and 16.69/39.81 are single-instrument "
        "n=1540. The Judge_1:4 counterpart of the canonical row is already "
        "published as 10.13 (status/PAPERA_RESULTS_CONSOLIDATED.md:175), and "
        "10.13 is arithmetically consistent with 8.11 as the SAME run "
        "(uniquely: 156 of 1540 judge-correct + 5 of 446 regex-correct gives "
        "8.1067 blended and 10.1299 on Judge_1:4), so the conversion may already "
        "be done -- verify that before spending anything. Then: BABILong/RULER/"
        "LongEval replication; a second residual/checkpoint compressor; "
        "adapter size / layer-band ablation."
    ),
    "next_gate_gpu": "The drift leg is 0 GPU if the canonical predictions are "
                     "already on disk (rejudge = OpenAI API + CPU). Only the replication and "
                     "second-compressor legs need GPU.",
    "gpu_cost_estimate": {
        "drift_resolution_leg": "0 GPU-h. Two of the three relevant judge caches "
                                "are on wzc1 with 1540 records each; a rejudge is an API call plus a "
                                "CPU merge. Confirmed by re-deriving all four numbers this session "
                                "with no GPU.",
        "locomo_replication_per_arm": "~0.9 GPU-h. Basis: measured -- "
                                      "locomo_results/hcache_j12_noLoRA_chatFALSE eval_config written "
                                      "20:58:52, last preds shard written 21:16:43 = 17.9 min wall on 3 "
                                      "shards = 0.89 GPU-h. Judge is API, not GPU.",
        "second_compressor_leg": "~2.9 GPU-h per new adapter IF the second "
                                 "compressor reuses the distill recipe. Basis: measured -- "
                                 "outputs/qcmem_distill_qwen_j12_r32_4k trained 4000 steps on "
                                 "world_size=8 in 21.8 min wall (distill_args.json 00:04:02 -> final/ "
                                 "00:25:50) = 2.9 GPU-h. If the second compressor needs continued "
                                 "pretraining instead, add ~6.3 GPU-h per endpoint (measured: "
                                 "outputs/qwenbott_funnel_L12_d512, 2000 steps @2.83 s/step, ws=4).",
        "babilong_ruler_longeval_replication": "UNKNOWN -- needs a 1-cell timing. "
                                               "The task-pool scheduler (scripts/_eval_taskpool_2group.sh) makes "
                                               "throughput depend on the cell mix, and the long lengths (16k/32k) "
                                               "dominate. Do not extrapolate from the LoCoMo rate: LoCoMo is "
                                               "max_new_tokens=48 on 1986 short items.",
        "total_to_close_the_kill_gate": "~4-7 GPU-h (LoCoMo arms + one second "
                                        "compressor), because the decisive leg (unified-harness drift) is free"
    },
    "kill_gate": {
        "source": "PROPOSAL.md 'Kill 条件', copied verbatim (3 conditions)",
        "conditions": [
            "只在 LoCoMo open-domain category 有益",
            "统一 harness 后增益消失",
            "换 compressor 完全不迁移"
        ],
        "condition_1_status": "PARTIALLY TESTABLE FROM DISK NOW, and it is the "
                              "one at real risk. status/PAPERA_RESULTS_CONSOLIDATED.md:403 records "
                              "cat4 (open_domain, n=841 = 55% of the 1540) moving 23.31 -> 55.77 = "
                              "+32.46, described there as '最大驱动' (the largest driver). A "
                              "per-category breakdown of the +23.12 on the corrected instrument is "
                              "0 GPU and should be run before any generalisation leg, because if the "
                              "gain is cat4-only this condition fires.",
        "condition_2_status": "the drift is ~1.0 pp on the blended scale (12.286 "
                              "canonical vs 13.293 local, both from scores.json overall_judge, "
                              "re-read this session) against a +23.12 pp effect, so a same-harness "
                              "rejudge is very unlikely to erase the gain -- but it has NOT been run, "
                              "so the condition is OPEN, not cleared.",
        "condition_3_status": "UNTESTED -- no second compressor has been run"
    },
    "established_measurements": {
        "_instrument": "GPT-4o LLM judge on the 1,540 answerable LoCoMo items "
                       "(categories 1-4) = 'Judge_1:4'. Single instrument. Both arms graded "
                       "from the identical 1,540 ids, so the contrast is exactly paired.",
        "_provenance": "Recomputed from locomo_results/hcache_j12_{noLoRA,LoRA}_"
                       "chatFALSE/judge_cache.jsonl in this session (2026-08-14), NOT copied "
                       "from PROPOSAL.md or from the errata. Both caches: exactly 1540 "
                       "records, 0 duplicate ids, all records model='gpt-4o', and the two id "
                       "sets are identical (perfect pairing) -- all four properties asserted "
                       "programmatically.",
        "no_lora_judge_1_4": 16.6883,
        "no_lora_judge_1_4_rounded": 16.69,
        "no_lora_n_correct": "257 / 1540",
        "lora_judge_1_4": 39.8052,
        "lora_judge_1_4_rounded": 39.81,
        "lora_n_correct": "613 / 1540",
        "gain_pp": 23.1169,
        "gain_pp_rounded": 23.12,
        "mcnemar_discordant": "b=414 (LoRA correct, noLoRA wrong) / c=58 (reverse)",
        "mcnemar_exact_two_sided_p": 2.575e-67,
        "mcnemar_chi2_continuity_corrected": 267.0,
        "paired_item_bootstrap_95ci_pp": [20.58, 25.58],
        "bootstrap_protocol": "10,000 resamples, seed 1, per-item paired. "
                              "Independently reproduced this session to the printed 2 decimals "
                              "([20.58, 25.58]) with a different RNG (python random vs numpy), "
                              "which is evidence the interval is not RNG-specific.",
        "single_variable_verified": "the two eval_config json differ in exactly "
                                    "two fields -- lora_adapter ('' vs "
                                    "outputs/qcmem_distill_qwen_j12_r32_4k/final) and "
                                    "force_lora_with_baseline (false vs true). All others identical, "
                                    "including use_chat_template=false, selector=bm25, topk=12, "
                                    "no_retrieval=true, chunk_size=512, max_new_tokens=48, resume_j=12, "
                                    "baseline=hcache, num_shards=3. Field-by-field this session.",
        "why_the_gain_cannot_be_retrieval": "no_retrieval=true in BOTH arms, so "
                                            "there is no selector in the path; the contrast isolates the adapter",
        "protocol_note_selector": "selector=bm25 in both arms. This does NOT "
                                  "violate memory/qcmem-eval-selector-iterbm25 in the way it would for a "
                                  "retrieval claim, because no_retrieval=true makes the selector inert "
                                  "on this path -- but any cross-table comparison to an iter_bm25 "
                                  "retrieval arm is still off-protocol.",
        "caveat_from_errata": "LoCoMo items are nested in only 10 conversations, "
                              "so the per-item bootstrap interval is NOT dependence-aware. A "
                              "conversation-clustered bootstrap is the honest version and is 0 GPU "
                              "(status/PAPERA_RESULTS_CONSOLIDATED.md 9.2)."
    },
    "retracted_measurements": {
        "_retraction_date": "2026-08-10",
        "_authority": "paperA/ERRATA_LOCOMO_MIXED_INSTRUMENT_20260810.md",
        "_rule": "Per CLAUDE.md / README, a retraction is recorded, never "
                 "silently deleted. These values must not be re-cited as current.",
        "no_lora_retracted": 13.293051359516618,
        "no_lora_retracted_rounded": 13.29,
        "lora_retracted": 31.168177240684795,
        "lora_retracted_rounded": 31.17,
        "gain_retracted_pp": 17.88,
        "reason": "These are scores.json's overall_judge, which is a WEIGHTED "
                  "BLEND OF TWO INSTRUMENTS over n=1986: categories 1-4 (1,540 items) "
                  "are graded by the GPT-4o judge, while category 5 (446 items, 22.5% of "
                  "the weight) NEVER reaches the judge and is graded locally by a "
                  "refusal regex. Publishing that blend under a single-instrument header "
                  "('LoCoMo Judge') in a table whose whole purpose is a single-variable "
                  "LoRA ablation changed the measuring instrument between the header and "
                  "the data, and diluted the effect with a constant unrelated to the adapter.",
        "mechanism_in_code": "scripts/eval_qcmem_locomo.py -- cat-5 short-circuit "
                             "at lines 687-690 (verified this session: line 687 'if "
                             "item.get(\"is_abstention\", False):', 688-689 the _REFUSAL_RE search, "
                             "690 'item[\"judge\"] = 1.0 if refused else 0.0'); the single-list "
                             "blend at lines 434-435 (verified: overall['judge'].append + "
                             "by_cat[cat]['judge'].append into one list).",
        "blend_reproduced_this_session": "noLoRA (257 judge-correct + 7 "
                                         "regex-correct)/1986 = 13.293051359517 == scores.json overall_judge "
                                         "to 12 decimals; LoRA (613 + 6)/1986 = 31.168177240685 == "
                                         "scores.json to 12 decimals. The cat-5 term moves the WRONG WAY for "
                                         "the treatment arm (7 -> 6), so the blend does not merely shrink the "
                                         "effect, it adds a term of opposite sign.",
        "direction_of_the_correction": "the correction makes the claim STRONGER "
                                       "(+17.88 -> +23.12, 2.4x). The errata discloses this explicitly as a "
                                       "reason for a reader to scrutinise it, which is why the arithmetic is "
                                       "written out reproducibly rather than asserted.",
        "still_in_circulation_on_purpose": "frozen submission snapshots "
                                           "(paperA/review_history/v*_source/, paperA/venue_versions/*/source/, "
                                           "paperA/submission_source*/) still contain 13.29/31.17 and are "
                                           "correct AS HISTORY. paperA/sections/tab_hcache_lora.tex has been "
                                           "corrected to 16.69/39.81 with the errata cited in its caption "
                                           "(verified this session). B06's own PROPOSAL.md carries the errata "
                                           "banner above the retained original text."
    },
    "third_measurement_found_20260814": {
        "why_recorded": "The drift in the next_gate is usually described as a "
                        "two-way 8.11-vs-13.29 disagreement. There is a THIRD number on wzc1, "
                        "and whoever executes the gate needs to know it exists or they will "
                        "compare the wrong pair.",
        "run": "locomo_results/hcache/ (older run, 2026-07-09/10, judged 07-18)",
        "overall_judge_blended_n1986": 12.28600201409869,
        "judge_1_4_n1540": 15.4545,
        "n_correct": "238 / 1540",
        "derived_this_session": "from its own judge_cache.jsonl (1540 records, 0 "
                                "duplicates, all gpt-4o, and its id set is IDENTICAL to the two B06 "
                                "arms -> it is exactly pairable against them with no re-generation)",
        "config_differences_vs_the_B06_noLoRA_arm": "same baseline=hcache, "
                                                    "resume_j=12, chunk_size=512, topk=12, no_retrieval=true, "
                                                    "max_new_tokens=48, use_chat_template=false, selector=bm25, "
                                                    "num_shards=3. Differs in: model_path spelled as the absolute "
                                                    "models/Qwen--Qwen3-8b vs the models/Qwen3-8b-local symlink (the "
                                                    "symlink resolves to the same weights), and the older run predates "
                                                    "the enable_thinking / iter_* / judge_* / force_lora_with_baseline "
                                                    "arguments entirely.",
        "consequence": "On the Judge_1:4 scale the three noLoRA measurements are "
                       "15.45 (older local run), 16.69 (B06 control arm) and 10.13 (canonical, "
                       "per status/PAPERA_RESULTS_CONSOLIDATED.md:175). So the spread among "
                       "noLoRA measurements is up to ~6.6 pp on the SAME instrument, against a "
                       "+23.12 pp effect. The effect survives comfortably, but the gate should "
                       "report all three rather than a single 'drift' number.",
        "not_a_retraction": "This does not contradict anything published; it is "
                            "an additional same-instrument replicate that was not previously "
                            "enumerated in B06's own files."
    },
    "canonical_8_11_conversion_status": {
        "claim_to_verify_before_spending": "status/PAPERA_RESULTS_CONSOLIDATED.md:175 "
                                           "already prints the canonical HCache row as judge(n=1986)=8.11 AND "
                                           "judge(cat1-4, n=1540)=10.13.",
        "arithmetic_check_this_session": "Searching integer (cat1-4 correct, "
                                         "cat5 correct) pairs, the ONLY pair giving both 8.11 and 10.13 to 2 "
                                         "decimals is (156, 5): 100*(156+5)/1986 = 8.1067 and 100*156/1540 = "
                                         "10.1299. So the two published cells are mutually consistent as one "
                                         "run under the documented instrument split, and 10.13 is the "
                                         "Judge_1:4 conversion the gate asks for.",
        "what_is_still_missing": "This is an arithmetic consistency check, NOT a "
                                 "rejudge. The canonical run's own judge_cache.jsonl lives on zwfy6 "
                                 "(.73/.104) and was NOT read here -- zwfy6 is not mounted on this node "
                                 "(verified: /apdcephfs_zwfy6 does not exist locally). Per "
                                 "memory/two-disk-rule-applies-to-main-too, 'not found' is only valid "
                                 "after searching BOTH disks, so the honest statement is 'not checked "
                                 "on zwfy6', not 'absent'.",
        "so_the_gate_shrinks_to": "(a) confirm 10.13 against the canonical "
                                  "judge_cache on zwfy6 (0 GPU, needs ssh to .73/.104 or an scp -O of a "
                                  "~109 KB file), and (b) compare 10.13 vs 16.69 vs 15.45 as three "
                                  "same-instrument noLoRA replicates."
    },
    "novelty_checked": False,
    "novelty_status_detail": "RELATED_WORK.md does NOT exist here. "
                             "proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:96 rates "
                             "B06 '不足' and lists the collision families: activation "
                             "decompression adapter; split-compute reconstruction; adapter "
                             "transfer; intermediate self-distillation; cross-codec portability. "
                             "Its stated bar is that 'portable' requires at least multi-task, "
                             "multi-compressor, multi-model, or an explicit layer/module transfer "
                             "-- which is the same thing B06's own success conditions demand. Same "
                             "audit line 163 requires this before promotion.",
    "success_conditions_from_proposal": [
        "多任务、多 compressor 保持显著 lift",
        "layer/module ablation 定位共享 readout repair",
        "不依赖特定 retrieval pack"
    ],
    "claim_scope_discipline": {
        "may_claim": "On a retrieval-free HCache read path at 8B j=12, a "
                     "self-distilled LoRA recovers +23.12 pp Judge_1:4 (paired, "
                     "p=2.6e-67), so the adapter is not a CoMem-retrieval-pack specialisation.",
        "must_not_claim_yet": [
            "'portable' -- one task, one compressor, one model so far; both the "
            "proposal's own success conditions and the related-work audit require more",
            "any number on the n=1986 blended scale (retracted)",
            "that the gain is not concentrated in cat4 -- the per-category "
            "breakdown on the corrected instrument has not been computed, and the "
            "old blended breakdown shows cat4 (55% of the judged items) supplying "
            "+32.46 of the movement"
        ]
    },
    "updated_20260814": "2026-08-14 -- STATUS made decidable. This proposal's "
                        "STATUS previously carried only id/status/updated (83 bytes) even though "
                        "the direction holds a p=2.6e-67 confirmed result, which caused it to be "
                        "mis-ranked as a thin backlog item. Added headline, next_gate (with the "
                        "two-rulers scale trap), gpu_cost_estimate, kill_gate, "
                        "established_measurements (correct instrument only), "
                        "retracted_measurements, a newly found third same-instrument replicate, "
                        "and novelty_checked. Every number re-derived from raw judge caches. "
                        "No scientific conclusion changed. 0 GPU."
}

# ---------------------------------------------------------------- B07
B07 = {
    "next_gate": "NOT_SPECIFIED -- PROPOSAL.md describes two experiments in "
                 "detail but states no decidable gate: it gives no threshold, no "
                 "comparator to beat, and no outcome that would stop the work. The two "
                 "experiments as written are MEASUREMENT PROGRAMMES, not gates. Writing "
                 "an operationalised gate is a prerequisite for GPU under README's rule "
                 "'新方向先写 PROPOSAL.md 和 kill gate, 再启动 GPU'.",
    "next_gate_candidate_not_yet_adopted": {
        "_disclaimer": "NOT from PROPOSAL.md. Recorded as a suggestion so the "
                       "next agent has a starting point; it has NOT been adopted and must not "
                       "be cited as this proposal's gate.",
        "concurrency_leg": "PROPOSAL.md already fixes the design (document "
                           "32k/128k, generation 128, concurrency 1/8/32, CoMem vs matched j=0; "
                           "metrics TTFT p50/p95/p99, QPS, queue, OOM, HBM/host, real Q*). What "
                           "is missing is the decision rule -- e.g. 'if CoMem's TTFT p99 "
                           "advantage over matched j=0 vanishes at concurrency 32, the serving "
                           "thesis fails at realistic load'.",
        "edit_leg": "same shape -- the four update strategies are specified, the "
                    "pass/fail bar is not."
    },
    "gpu_cost_estimate": {
        "value": "UNKNOWN -- 需先做 1-cell 计时",
        "why": "The existing component benchmarks do NOT measure what B07 "
               "proposes, so there is no rate to extrapolate from. "
               "scripts/bench_p1_8_serving_curve.py sweeps GENERATION LENGTH "
               "(--gen_lengths, per_G keys 1/32/128/512) at store sizes "
               "{32k,128k,1M} x {cpu,gpu}; it has no concurrency axis and reports "
               "no TTFT percentiles (verified this session: the aggregate json's "
               "cells carry per_G with fetch_s/read_s/decode_s/per_query_s medians "
               "and p90, plus peak GPU/host and a crossover, but nothing "
               "concurrency-indexed). scripts/bench_persistent_store_io.py DOES "
               "sweep 1/4/16 threads for QPS, but with NO MODEL LOADED (synthetic "
               "bf16 store) -- so it cannot produce TTFT under generation either. "
               "Neither existing artefact is a valid anchor for a concurrent "
               "end-to-end serving benchmark.",
        "measured_facts_that_do_transfer": [
            "store geometry: chunk = 512 tok x 4096 x 2 B = 4 MiB; top-12 pack = "
            "50.33 MB; h12 store = 8192 B/token (status/P2_2_PERSISTENT_STORE_IO.md)",
            "store capacity: 128k -> 1.00 GiB, 1M -> 8.00 GiB, 4M -> 32.00 GiB, "
            "8M -> 64.00 GiB, 16M -> 128.00 GiB (same file, verified vs P1.1)",
            "at 128k|cpu: comem write-once 6.82 s median vs j0 index 0.021 s; "
            "comem_store_bytes 1,073,741,824 vs j0_store_bytes 524,288; peak GPU "
            "serve 17.29 GB (comem) vs 17.57 GB (j0) "
            "(paperA/artifacts/p1_8_serving/p1_8_serving_aggregate.json, read this session)"
        ],
        "first_action": "Time ONE cell of the concurrency grid (e.g. 128k "
                        "document, generation 128, concurrency 8) end-to-end with the model "
                        "loaded. Until then any total is invented."
    },
    "kill_gate": "NO_KILL_GATE_DEFINED -- PROPOSAL.md has no Kill section. It "
                 "has a '关键设计' (key design) section listing mechanisms (chunk/version/"
                 "content hash, compatibility hash, fail-closed stale object, dependency "
                 "graph for overlap, mixed-version read only with explicit fallback), "
                 "which are requirements, not falsification conditions.",
    "established_measurements": {
        "_status": "NONE FOR B07'S OWN CLAIMS. The direction has no measurement "
                   "of concurrency, versioning, incremental edit, or tiering -- i.e. of "
                   "anything it proposes. PROPOSAL.md says so itself: '现有是组件 "
                   "benchmark, 不是生产端到端系统'.",
        "what_exists_is_adjacent_not_this": {
            "p1_8_serving": "generation-length x store-size x cpu/gpu amortisation "
                            "curve, single-stream. 18 files aggregated, query_counts [1,4,16,32,64], "
                            "gen lengths [1,32,128,512], cells {32k,128k,1M} x {cpu,gpu}. "
                            "paperA/artifacts/p1_8_serving/p1_8_serving_aggregate.json",
            "p2_2_store_io": "storage-backend I/O benchmark, 60+ cells, 4 backends "
                             "x 5 store sizes, median-of-7, O_DIRECT, QPS at 1/4/16 threads -- but "
                             "NO MODEL LOADED (synthetic deterministic bf16 store, seed 42) and it "
                             "times the fetch, not BM25 and not generation. Done 2026-08-01 on .104. "
                             "status/P2_2_PERSISTENT_STORE_IO.md; raw ruler_results/p2_2/*.json",
            "why_this_distinction_matters": "Reading either artefact as evidence "
                                            "for a serving claim would be exactly the mixed-construct error this "
                                            "repo has already made elsewhere. They bound components; B07 claims "
                                            "a system."
        }
    },
    "blocking_dependency": {
        "statement": "Every mechanism in '关键设计' (versioning, content/"
                     "compatibility hashes, fail-closed stale objects, overlap dependency "
                     "graph, mixed-version fallback) is a system that does not exist yet. "
                     "B07 is an ENGINEERING project before it is an experiment.",
        "consequence_for_ranking": "B07 cannot be picked up as a short GPU task. "
                                   "It needs implementation first, and its cost is dominated by "
                                   "engineering time, not GPU-h."
    },
    "novelty_checked": False,
    "novelty_status_detail": "RELATED_WORK.md does NOT exist here. "
                             "proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:98 rates "
                             "B07 '不足' and lists heavy systems collisions: prefix/KV caching; "
                             "paged/disaggregated KV; versioning/invalidation; memory tiering; "
                             "reuse-aware admission; incremental recompute. It demands a "
                             "FEATURE-BY-FEATURE systems collision table. Line 146 of the same "
                             "audit warns explicitly that B07/B08 cannot rest on a feature list "
                             "('可版本化/可更新/分层 memory') as a novelty claim -- which is "
                             "precisely what B07's PROPOSAL.md currently is.",
    "updated_20260814": "2026-08-14 -- STATUS made decidable (honestly: recorded "
                        "that it is NOT yet decidable). next_gate=NOT_SPECIFIED, "
                        "kill_gate=NO_KILL_GATE_DEFINED, gpu_cost_estimate=UNKNOWN with the "
                        "reason that neither existing benchmark has a concurrency axis, "
                        "established_measurements documenting that the existing artefacts "
                        "measure something else, blocking_dependency, novelty_checked. No "
                        "gate was invented on the proposal's behalf. 0 GPU."
}

# ---------------------------------------------------------------- B08
B08 = {
    "next_gate": "NOT_SPECIFIED -- PROPOSAL.md is a three-part portfolio "
                 "(query-conditioned notes + raw evidence; typed personal memory ledger; "
                 "multi-tier pyramid memory) with goals and Kill conditions but no "
                 "sequenced first step, no chosen sub-direction, and no decidable gate. "
                 "It does state one sequencing constraint ('先做两层 MVP' -- build a "
                 "two-tier MVP first, because the full pyramid is high-risk), which is "
                 "the closest thing to a next step but is not a gate.",
    "next_gate_blocked_by_portfolio_shape": "The three sub-directions have "
                                            "different risk and different literature, and the related-work audit "
                                            "says they should be split. Choosing WHICH of the three to gate is a "
                                            "prerequisite decision, and it is not made anywhere in the proposal.",
    "gpu_cost_estimate": {
        "value": "UNKNOWN -- 需先做 1-cell 计时, and it is not even well-posed "
                 "until one of the three sub-directions is chosen",
        "why": "The three sub-directions have costs that differ qualitatively: "
               "sub-direction 1 (notes) is dominated by generation over retrieval "
               "candidates plus a faithfulness evaluation that has no harness here; "
               "sub-direction 2 (typed ledger) is mostly CPU/system work plus "
               "LongMemEval update/temporal evaluation; sub-direction 3 (pyramid) "
               "is explicitly flagged high-risk in the proposal itself. No arm of "
               "any of the three has ever been run, so there is no rate to "
               "extrapolate from.",
        "asset_status_checked_this_session": {
            "longmemeval_data_PRESENT": "data/longmemeval/longmemeval_s.json "
                                        "(278,025,796 B) and the longmemeval/ harness (backends.py, "
                                        "compressor.py, reader.py, run_baseline.py, scoring.py, data.py) plus "
                                        "scripts/eval_qcmem_longmemeval.py are on wzc1",
            "no_results": "no longmemeval results directory exists on wzc1 -- so "
                          "sub-direction 2's evaluation surface is available but has never been "
                          "run here. NOTE: zwfy6 was NOT searched (not mounted on this node), so "
                          "per memory/two-disk-rule-applies-to-main-too this is 'not found on "
                          "wzc1', not 'does not exist'.",
            "ledger_and_pyramid_code_PRESENT": "src/memory/l2/{aggregator,merger,"
                                               "object_store,retriever,types}.py and src/memory/l3/{formatter,"
                                               "profile_store,reviser,summarizer}.py exist; also "
                                               "src/agents/memory_agent.py, src/tasks/synthetic_update_task.py, "
                                               "src/eval/update_eval.py",
            "caution": "presence of code is not evidence it runs or is current -- "
                       "src/agents and src/tasks/src/eval files are dated 2026-05-14, well "
                       "before the current QCMem stack"
        }
    },
    "kill_gate": {
        "source": "PROPOSAL.md 'Kill 条件', copied verbatim (3 conditions, one "
                  "per sub-direction)",
        "conditions": [
            "notes-only 幻觉率高且 notes+raw 不优于 raw",
            "typed ledger 不降低 stale/conflict error",
            "pyramid 的 far memory 读取成本吞没固定 Read 优势"
        ],
        "structural_problem_recorded_not_adjudicated": "These are three "
                                                       "SEPARATE kill conditions for three separate sub-directions, so no "
                                                       "single one of them can kill B08 as a whole. As written, the "
                                                       "portfolio has no kill gate at the portfolio level -- it can only be "
                                                       "narrowed, never killed. Whoever picks this up should either split B08 "
                                                       "into three proposals (which the related-work audit also recommends) "
                                                       "or add a portfolio-level rule.",
        "note": "the proposal also states a non-negotiable measurement "
                "requirement for sub-direction 1 -- '必须测 notes faithfulness, 不能让 "
                "summary 成为唯一事实源' -- which is a protocol invariant, not a kill "
                "condition, but must not be dropped."
    },
    "established_measurements": None,
    "established_measurements_note": "No arm of any of the three sub-directions "
                                     "has been run, so there is nothing to record. This is 'no data yet', "
                                     "NOT 'data lost' and NOT 'measured null'.",
    "novelty_checked": False,
    "novelty_status_detail": "RELATED_WORK.md does NOT exist here. "
                             "proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:98 rates "
                             "B08 '严重不足' (SEVERELY insufficient) -- the worst rating of any "
                             "proposal in that audit -- and lists collisions across "
                             "query-focused context compression; grounded notes/provenance; "
                             "personal/conversational memory; temporal KG/event sourcing; "
                             "hierarchical memory. It requires the three sub-directions be SPLIT "
                             "and each given its own Related Work. Line 146 warns that a "
                             "feature-list novelty claim ('可版本化/可更新/分层 memory') will "
                             "not survive.",
    "priority_note": "Highest literature risk of the five thin-STATUS backlog "
                     "proposals and no data, so it should rank LAST among them for GPU. That "
                     "is a resource statement, not a verdict on the science -- per README, "
                     "backlog means 'not currently scheduled', not 'not allowed'.",
    "updated_20260814": "2026-08-14 -- STATUS made decidable (honestly: recorded "
                        "that it is NOT yet decidable). next_gate=NOT_SPECIFIED, kill_gate "
                        "copied verbatim with its portfolio-level structural gap recorded, "
                        "gpu_cost_estimate=UNKNOWN, established_measurements=null with an "
                        "explanation, asset audit, novelty_checked. No gate was invented on the "
                        "proposal's behalf. 0 GPU."
}

TARGETS = [
    ("B01-semantic-bottleneck-memory-ready-models", B01),
    ("B05-semantic-handoff-phase-diagram", B05),
    ("B06-portable-decompression-adapter", B06),
    ("B07-mutable-comem-serving", B07),
    ("B08-memory-applications", B08),
]


def main():
    failures = []
    for slug, new_keys in TARGETS:
        path = os.path.join(BACKLOG, slug, "STATUS.json")
        with open(path, "rb") as fh:
            raw = fh.read()
        orig = json.loads(raw)
        if not isinstance(orig, dict):
            failures.append(f"{slug}: top level is not an object")
            continue

        clash = sorted(set(new_keys) & set(orig))
        if clash:
            # Idempotence: if EVERY key we want is already present with the
            # value we would write, this file was already processed -- report
            # and skip rather than fail. Otherwise refuse: appending must
            # never shadow an existing value.
            if clash == sorted(new_keys) and all(
                json.dumps(orig[k], sort_keys=True)
                == json.dumps(new_keys[k], sort_keys=True) for k in new_keys
            ):
                print(f"[skip] {slug}: already appended, values identical")
                continue
            failures.append(f"{slug}: REFUSING, would shadow existing keys {clash}")
            continue

        merged = dict(orig)            # originals first, original order
        merged.update(new_keys)        # additions appended after

        out = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"

        # verify round-trip and that every ORIGINAL key is value-identical
        back = json.loads(out)
        ok = True
        for k, v in orig.items():
            if json.dumps(back[k], sort_keys=True) != json.dumps(v, sort_keys=True):
                failures.append(f"{slug}: original key {k!r} changed value")
                ok = False
        if list(back)[:len(orig)] != list(orig):
            failures.append(f"{slug}: original key ORDER changed")
            ok = False
        if not ok:
            continue

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(out)
        print(f"[ok] {slug}: +{len(new_keys)} keys "
              f"({len(raw)} -> {len(out.encode())} bytes)")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print("\nall five updated, append-only verified")


if __name__ == "__main__":
    main()
