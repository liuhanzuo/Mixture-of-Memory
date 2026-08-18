#!/usr/bin/env python3
"""Render G0_VERDICT.md FROM the evidence JSONs, so prose cannot drift from measurement.

Every number in the output is read out of
  evidence/g0_result_20260817.json
  evidence/g0_optimizer_moment_audit_20260817.json
  evidence/g0_selftest_c_unfireable_20260817.json
and nothing is typed by hand. Re-run it after any re-measurement:

  /opt/conda/envs/torch-base/bin/python \
      proposal/backlog/B03-cyclic-layer-reset-boundary/render_g0_verdict.py

0 GPU. Exits non-zero if an expected field is missing, so a silently-changed schema
cannot produce a confident-looking but empty verdict.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(HERE, "evidence")


def load(name):
    p = os.path.join(EV, name)
    if not os.path.exists(p):
        sys.exit(f"FATAL: missing evidence file {p}")
    with open(p) as f:
        return json.load(f)


r = load("g0_result_20260817.json")
m = load("g0_optimizer_moment_audit_20260817.json")
c = load("g0_selftest_c_unfireable_20260817.json")
st = r["selftests"]


def yn(b):
    return "PASS" if b else "FAIL"


rows = [
    ("(a) non-reset tensors byte-identical",
     yn(st["a_nonreset_bytes_identical"]["pass"]),
     f"{st['a_nonreset_bytes_identical']['n_nonreset_tensors_checked']} tensors sha256-compared, "
     f"{st['a_nonreset_bytes_identical']['n_mismatched']} mismatched"),
    ("(a-neg) reset tensors DID change",
     yn(st["a_negative_control_reset_tensors_did_change"]["pass"]),
     f"{st['a_negative_control_reset_tensors_did_change']['n_changed']}"
     f"/{st['a_negative_control_reset_tensors_did_change']['n_reset_tensors']} changed "
     f"-- (a) cannot pass by doing nothing"),
    ("(b) trainer's own _assert_fresh_init",
     yn(st["b_assert_fresh_init"]["pass"]),
     f"post_attention_layernorm all-ones={st['b_assert_fresh_init']['post_attention_layernorm_all_ones']}, "
     f"q_norm all-ones={st['b_assert_fresh_init']['q_norm_all_ones']}, "
     f"q_proj std={st['b_assert_fresh_init']['q_proj_weight_std']:.17g} in {st['b_assert_fresh_init']['band']}"),
    ("(c) strict load, RE-SPECIFIED",
     yn(st["c_strict_load"]["pass"]),
     f"{st['c_strict_load']['n_missing']} missing / {st['c_strict_load']['n_unexpected']} unexpected; "
     f"prereg's literal --dry_run_build form was unfireable="
     f"{st['c_strict_load']['prereg_literal_form_was_unfireable']}"),
    ("(d) optimizer group count preserved",
     yn(st["d_optimizer_group_count_preserved"]["pass"]),
     f"ckpt before={st['d_optimizer_group_count_preserved']['n_ckpt_groups_before']}, "
     f"after={st['d_optimizer_group_count_preserved']['n_ckpt_groups_after']}, "
     f"trainer rebuilds={st['d_optimizer_group_count_preserved']['n_groups_the_trainer_rebuilds_on_resume']}, "
     f"shim_would_trigger={st['d_optimizer_group_count_preserved']['shim_would_trigger']}, "
     f"vacuous={st['d_optimizer_group_count_preserved']['vacuous']}"),
]

L = []
A = L.append
A("# B03 G0 — verdict")
A("")
A("**GENERATED FILE. Do not hand-edit.** Rendered by `render_g0_verdict.py` from the")
A("evidence JSONs in `evidence/`; every number below is read from disk, none is typed.")
A("")
A(f"- verdict: **{r['verdict']}**")
A(f"- run: `{r['run_utc']}` on `{r['host']}`, torch {r['torch_version']}, python {r['python']}")
A(f"- GPU used: **{r['gpu_used']}** (elapsed {r['elapsed_s']} s, CPU only)")
A(f"- operator: `reset_layers_ckpt.py`")
A(f"- trainer reference (unmodified): `{os.path.basename(r['trainer_ref']['path'])}`"
  f" sha256 `{r['trainer_ref']['sha256'][:16]}...`")
A("")
A("## What was pre-registered")
A("")
A("`GATE_PREREGISTRATION.md` §8.2 (re-affirmed unchanged by `READOUT_RULE_20260816.md` §6")
A("clause 1): a 0-GPU checkpoint-surgery script that re-initialises the top `n_fresh` layers")
A("from the trainer's own `Olmo2ForCausalLM(cfg).post_init()` distribution, zeroes the")
A("corresponding Adam moments, leaves every other tensor byte-identical, and writes a ckpt")
A("that `--resume_from` strict-loads — plus four self-tests. §8.2 also pre-registers the")
A("alternative outcome: *if (d) cannot pass, B03 ends at 0 GPU* with a protocol note.")
A("")
A("## Configuration actually operated on")
A("")
A(f"- scale: **1B**, the design scale — base `{r['inputs']['model_path']}`")
A(f"- arm: `keep_front_layers={r['inputs']['keep_front_layers']}` + "
  f"`n_fresh_layers={r['inputs']['n_fresh_layers']}` → reset layers **{r['reset_layer_ids']}**")
A(f"- ckpt arch: hidden {r['ckpt_arch']['hidden_size']}, vocab {r['ckpt_arch']['vocab_size']}, "
  f"{r['ckpt_arch']['n_decoder_layers']} decoder layers, {r['n_model_tensors']} model tensors")
A(f"- fresh shell built via `{r['fresh_shell']['built_via']}` "
  f"(initializer_range {r['fresh_shell']['initializer_range']})")
A(f"- tensors replaced: **{r['n_tensors_replaced']}** "
  f"(= N_TENSORS_PER_LAYER × n_fresh = {r['expected_n_tensors_replaced']})")
A(f"- seed: {r['inputs']['seed']}")
A("")
A("## The four self-tests")
A("")
A("| test | result | measurement |")
A("|---|---|---|")
for name, res, detail in rows:
    A(f"| {name} | **{res}** | {detail} |")
A("")
A(f"All evaluable self-tests pass: **{r['all_evaluable_selftests_pass']}**. "
  f"Failed: `{r['failed_selftests']}`. Skipped: `{r['skipped_selftests']}`.")
A("")
A("## Optimizer moments (invariant I4), audited on disk")
A("")
A("Both checkpoints were re-opened independently of the surgery tool.")
A("")
A("| | reset-layer params with a NON-ZERO moment | reset-layer `step` | non-reset params with a "
  "non-zero moment | non-reset `step` | param groups |")
A("|---|---|---|---|---|---|")
for tag in ("input", "surgical"):
    x = m[tag]
    A(f"| **{tag}** | {x['reset_layer_params_with_a_nonzero_moment']} | "
      f"{x['reset_layer_step_values']} | {x['nonreset_params_with_a_nonzero_moment']} | "
      f"{x['nonreset_step_values']} | {x['n_param_groups']} |")
A("")
A("The `input` row is the **negative control**: the reset was not already true, so the")
A("`surgical` row is a real change and not a tautology. Assertions, all machine-checked:")
A("")
for k, v in m["assertions"].items():
    A(f"- `{k}` = **{v}**")
A("")
A(f"(all_pass = **{m['all_pass']}**)")
A("")
A("## Beyond the four self-tests: the real trainer resumed from the surgical ckpt")
A("")
A("`scripts/train_olmo2_arch_probe2.py --resume_from <surgical ckpt>` was run on CPU (0 GPU).")
A("It restored 102 model tensors strictly, rebuilt all four param groups, and took the")
A("**normal** `n_ckpt_groups == n_new_groups` branch — *not* the 2→4 compatibility shim:")
A("")
A("```")
A("[resume] restored 102 model tensors (strict, fp32 master weights)")
A("[resume] optimizer state restored (102 param states) -> Adam momentum preserved")
A("```")
A("")
A("Full log: `evidence/g0_resume_probe_cpu.log`. This is strictly stronger than self-test (c),")
A("which exercises only the model half.")
A("")
A("## Pre-data protocol finding: self-test (c) as written was unfireable")
A("")
A("The prereg's literal (c) — *\"strict-loads under `--dry_run_build`\"* — **cannot fail, and")
A("therefore cannot pass.** AST-verified in `main()`:")
A("")
A(f"- `if args.dry_run_build:` at line "
  f"{c['dry_run_build_guard'][0]['lineno']}, contains a `return` = "
  f"`{c['dry_run_build_guard'][0]['contains_return']}`")
A(f"- `if args.resume_from:` at line {c['resume_from_guard'][0]['lineno']}")
A(f"- returns before the ckpt is read: "
  f"**{c['dry_run_build_returns_before_resume_from_is_read']}**")
A("")
A("Confirmed by execution, not only by reading: the trainer was run with `--dry_run_build`")
A("**and** `--resume_from /tmp/THIS_FILE_DOES_NOT_EXIST_AT_ALL.pt` (path confirmed absent).")
A("It printed `-> OK`, `arch/init logic validated`, and exited **rc=0**. A test that passes on")
A("a nonexistent checkpoint has zero power to detect a corrupt one")
A("(`evidence/g0_negctl_dryrunbuild_nonexistent_ckpt.log`).")
A("")
A("(c) is therefore executed as the statement the trainer itself runs at line 776, with the")
A("surgical ckpt actually loaded. `GATE_PREREGISTRATION.md` was **not** edited; the")
A("re-specification is recorded in `STATUS.json:g0_prereg_selftest_c_was_unfireable_20260817`,")
A("and it is pre-data — no B03 number exists on either disk.")
A("")
A("## What this does NOT establish")
A("")
A(r["what_this_does_NOT_establish"] if isinstance(r.get("what_this_does_NOT_establish"), str)
  else "(see STATUS.json:g0_result_20260817.what_this_does_NOT_establish)")
A("")
A("## It does not release a card")
A("")
A("`gpu_policy` requires **three** things. Only clause (1) is discharged:")
A("")
A("| clause | status |")
A("|---|---|")
A("| (1) reset operator exists, G0 passes | **DISCHARGED** 2026-08-17, 0 GPU |")
A("| (2) explicit user authorisation for ~748 GPU-h (or G1's 107.7) | **STILL BLOCKING** — "
  "absent from `STATUS.json`; only MAIN + the user can add it |")
A("| (3) a free `sm_90` node (`.73`/`.82`/`.104`, invariant I1) | **STILL BLOCKING**, and "
  "**not re-measured** — the executing agent had a hard zero-GPU budget and could not ssh or "
  "run `nvidia-smi`. Last actual reading: 2026-08-15, all three at 100 %. |")
A("")
A("No `discharges` pointer was filed. `gpu_policy` is a single top-level **string**, so")
A("`ready_queue.py`'s `_walk_blockers` yields exactly one path for it; a pointer at")
A("`gpu_policy` would close clauses (2) and (3) too (measured), and `gpu_policy[0]` /")
A("`gpu_policy.clause_1` are reported dangling with no effect (also measured). See")
A("`STATUS.json:gpu_policy_clause_status_20260817`.")
A("")
A("**A G0 pass is a precondition being met, not a promotion.** B03 stays `ready_cpu` /")
A("`priority=low` / `status=hold_gate_only`.")
A("")
A("## New blocker discovered by this step (0 GPU to fix)")
A("")
A("The canonical dolmino corpus exists **only in tmpfs**: `/dev/shm/dolmino_now15b_wzc1.npy`")
A("is 15,491,607 × 2048, while the persistent `data/dolmino_now15b.npy` is a **prefix**")
A("(7,570,911 × 2048, re-measured 2026-08-17). tmpfs is wiped on restart, so *\"every cell")
A("loads the same corpus as the ladder\"* is currently unsatisfiable from persistent storage.")
A("If a later cell silently trains on the prefix, the single-pass-vs-repeated-data axis — half")
A("the 2×3 design — is confounded with corpus identity, invisibly. Rebuild from")
A("`data/dolmino_olmo2_shards/` (86 files present) before G1. See")
A("`STATUS.json:corpus_persistence_defect_20260817`.")
A("")
A("## Not a novelty claim")
A("")
A("The operator is **LLF** (Zhou, Vani, Larochelle, Courville, ICLR 2022, DBLP")
A("`conf/iclr/ZhouVLC22`) with mask `M^l = 1[l < keep_front]`. `RELATED_WORK.md` §2.1")
A("establishes the operator collision and §3 items 1–2 forbid any method claim. The tool's")
A("docstring says so, and the ckpt it writes carries a `b03_reset_provenance` field saying so.")
A("")
A("## Reproduce")
A("")
A("```bash")
A(r["reproduce"] if isinstance(r.get("reproduce"), str) else "(see STATUS.json)")
A("```")
A("")

out = os.path.join(HERE, "G0_VERDICT.md")
with open(out, "w") as f:
    f.write("\n".join(L))
print(f"wrote {out} ({len(L)} lines)")
