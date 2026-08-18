# B03 G0 — verdict

**GENERATED FILE. Do not hand-edit.** Rendered by `render_g0_verdict.py` from the
evidence JSONs in `evidence/`; every number below is read from disk, none is typed.

- verdict: **G0_PASS**
- run: `2026-08-17T05:00:05Z` on `TENCENT64.site`, torch 2.13.0, python 3.14.6
- GPU used: **False** (elapsed 69.18 s, CPU only)
- operator: `reset_layers_ckpt.py`
- trainer reference (unmodified): `train_olmo2_arch_probe2.py` sha256 `88d263c3d5f7c5ac...`

## What was pre-registered

`GATE_PREREGISTRATION.md` §8.2 (re-affirmed unchanged by `READOUT_RULE_20260816.md` §6
clause 1): a 0-GPU checkpoint-surgery script that re-initialises the top `n_fresh` layers
from the trainer's own `Olmo2ForCausalLM(cfg).post_init()` distribution, zeroes the
corresponding Adam moments, leaves every other tensor byte-identical, and writes a ckpt
that `--resume_from` strict-loads — plus four self-tests. §8.2 also pre-registers the
alternative outcome: *if (d) cannot pass, B03 ends at 0 GPU* with a protocol note.

## Configuration actually operated on

- scale: **1B**, the design scale — base `/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-0425-1B`
- arm: `keep_front_layers=7` + `n_fresh_layers=2` → reset layers **[7, 8]**
- ckpt arch: hidden 2048, vocab 100352, 9 decoder layers, 102 model tensors
- fresh shell built via `train_olmo2_arch_probe2.build_olmo2_minimal(transplant=False) -> Olmo2ForCausalLM(cfg) post_init` (initializer_range 0.02)
- tensors replaced: **22** (= N_TENSORS_PER_LAYER × n_fresh = 22)
- seed: 42

## The four self-tests

| test | result | measurement |
|---|---|---|
| (a) non-reset tensors byte-identical | **PASS** | 80 tensors sha256-compared, 0 mismatched |
| (a-neg) reset tensors DID change | **PASS** | 22/22 changed -- (a) cannot pass by doing nothing |
| (b) trainer's own _assert_fresh_init | **PASS** | post_attention_layernorm all-ones=True, q_norm all-ones=True, q_proj std=0.019991094246506691 in 0.01 < std < 0.04 |
| (c) strict load, RE-SPECIFIED | **PASS** | 0 missing / 0 unexpected; prereg's literal --dry_run_build form was unfireable=True |
| (d) optimizer group count preserved | **PASS** | ckpt before=4, after=4, trainer rebuilds=4, shim_would_trigger=False, vacuous=False |

All evaluable self-tests pass: **True**. Failed: `[]`. Skipped: `[]`.

## Optimizer moments (invariant I4), audited on disk

Both checkpoints were re-opened independently of the surgery tool.

| | reset-layer params with a NON-ZERO moment | reset-layer `step` | non-reset params with a non-zero moment | non-reset `step` | param groups |
|---|---|---|---|---|---|
| **input** | 22 | [2] | 80 | [2] | 4 |
| **surgical** | 0 | [0] | 80 | [2] | 4 |

The `input` row is the **negative control**: the reset was not already true, so the
`surgical` row is a real change and not a tautology. Assertions, all machine-checked:

- `input_reset_moments_were_NOT_already_zero` = **True**
- `surgical_all_reset_moments_zero` = **True**
- `surgical_nonreset_moments_untouched` = **True**
- `surgical_reset_step_is_zero` = **True**
- `surgical_nonreset_step_preserved` = **True**
- `group_count_preserved_on_disk` = **True**

(all_pass = **True**)

## Beyond the four self-tests: the real trainer resumed from the surgical ckpt

`scripts/train_olmo2_arch_probe2.py --resume_from <surgical ckpt>` was run on CPU (0 GPU).
It restored 102 model tensors strictly, rebuilt all four param groups, and took the
**normal** `n_ckpt_groups == n_new_groups` branch — *not* the 2→4 compatibility shim:

```
[resume] restored 102 model tensors (strict, fp32 master weights)
[resume] optimizer state restored (102 param states) -> Adam momentum preserved
```

Full log: `evidence/g0_resume_probe_cpu.log`. This is strictly stronger than self-test (c),
which exercises only the model half.

## Pre-data protocol finding: self-test (c) as written was unfireable

The prereg's literal (c) — *"strict-loads under `--dry_run_build`"* — **cannot fail, and
therefore cannot pass.** AST-verified in `main()`:

- `if args.dry_run_build:` at line 714, contains a `return` = `True`
- `if args.resume_from:` at line 752
- returns before the ckpt is read: **True**

Confirmed by execution, not only by reading: the trainer was run with `--dry_run_build`
**and** `--resume_from /tmp/THIS_FILE_DOES_NOT_EXIST_AT_ALL.pt` (path confirmed absent).
It printed `-> OK`, `arch/init logic validated`, and exited **rc=0**. A test that passes on
a nonexistent checkpoint has zero power to detect a corrupt one
(`evidence/g0_negctl_dryrunbuild_nonexistent_ckpt.log`).

(c) is therefore executed as the statement the trainer itself runs at line 776, with the
surgical ckpt actually loaded. `GATE_PREREGISTRATION.md` was **not** edited; the
re-specification is recorded in `STATUS.json:g0_prereg_selftest_c_was_unfireable_20260817`,
and it is pre-data — no B03 number exists on either disk.

## What this does NOT establish

(see STATUS.json:g0_result_20260817.what_this_does_NOT_establish)

## It does not release a card

`gpu_policy` requires **three** things. Only clause (1) is discharged:

| clause | status |
|---|---|
| (1) reset operator exists, G0 passes | **DISCHARGED** 2026-08-17, 0 GPU |
| (2) explicit user authorisation for ~748 GPU-h (or G1's 107.7) | **STILL BLOCKING** — absent from `STATUS.json`; only MAIN + the user can add it |
| (3) a free `sm_90` node (`.73`/`.82`/`.104`, invariant I1) | **STILL BLOCKING**, and **not re-measured** — the executing agent had a hard zero-GPU budget and could not ssh or run `nvidia-smi`. Last actual reading: 2026-08-15, all three at 100 %. |

No `discharges` pointer was filed. `gpu_policy` is a single top-level **string**, so
`ready_queue.py`'s `_walk_blockers` yields exactly one path for it; a pointer at
`gpu_policy` would close clauses (2) and (3) too (measured), and `gpu_policy[0]` /
`gpu_policy.clause_1` are reported dangling with no effect (also measured). See
`STATUS.json:gpu_policy_clause_status_20260817`.

**A G0 pass is a precondition being met, not a promotion.** B03 stays `ready_cpu` /
`priority=low` / `status=hold_gate_only`.

## New blocker discovered by this step (0 GPU to fix)

The canonical dolmino corpus exists **only in tmpfs**: `/dev/shm/dolmino_now15b_wzc1.npy`
is 15,491,607 × 2048, while the persistent `data/dolmino_now15b.npy` is a **prefix**
(7,570,911 × 2048, re-measured 2026-08-17). tmpfs is wiped on restart, so *"every cell
loads the same corpus as the ladder"* is currently unsatisfiable from persistent storage.
If a later cell silently trains on the prefix, the single-pass-vs-repeated-data axis — half
the 2×3 design — is confounded with corpus identity, invisibly. Rebuild from
`data/dolmino_olmo2_shards/` (86 files present) before G1. See
`STATUS.json:corpus_persistence_defect_20260817`.

## Not a novelty claim

The operator is **LLF** (Zhou, Vani, Larochelle, Courville, ICLR 2022, DBLP
`conf/iclr/ZhouVLC22`) with mask `M^l = 1[l < keep_front]`. `RELATED_WORK.md` §2.1
establishes the operator collision and §3 items 1–2 forbid any method claim. The tool's
docstring says so, and the ckpt it writes carries a `b03_reset_provenance` field saying so.

## Reproduce

```bash
(see STATUS.json)
```
