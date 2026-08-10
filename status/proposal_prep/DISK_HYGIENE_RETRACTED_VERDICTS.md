# Disk-Hygiene Check — Retracted Verdict Scan
# Prepared: 2026-08-09 (read-only; no edits made)

---

## Search methodology

Four retracted verdicts were searched across all `.md` and `.json` files in the project
(both wzc1 local disk). `.pyc`, `outputs/`, `models/`, `__pycache__/`, `.claude/workflows/`,
`.claude/tasks/`, `status/dllm_reposition_salvage/` were excluded from MAIN's action scope
(they are either auto-generated, large binary contexts, or archived dllm work outside the
A01/A02/A03 proposals). Findings are listed with file path and context.

---

## Retracted verdict 1: "kill clause 2 triggered"

**Background**: `GATE1_VERDICT.md` was produced by an early eval run on INTACT (healthy)
non-OLMo models. The run found no letter-interface failure on intact models and concluded
"kill clause 2 triggered / narrow to OLMo-only". This verdict was RETRACTED the same
session (2026-08-08) when a follow-up run on DAMAGED non-OLMo models showed all 6/6 arms
at/below floor — confirming the GENERAL claim. The correct verdict is in
`STATUS.json:gate_results.gate1_third_model_family_INTACT` ("SUPERSEDED -- tested the wrong
condition") and in `TRAINER_ACTIVITY.jsonl` line timestamp 2026-08-08T22:58.

### Files still containing the stale "kill clause 2 triggered" language:

| File | Line | Stale text | Notes |
|------|------|-----------|-------|
| `proposal/active/A01-null-calibration-methodology/GATE1_VERDICT.md` | line 1 header + line 123 | `"verdict: KILL_CONDITION_CLAUSE_2_TRIGGERED"` (header) and `"gate-1 (third family): **DONE, kill clause 2 triggered on this half.**"` | This entire file's verdict section is RETRACTED per `STATUS.json`. The numeric results in the file remain valid; only the verdict paragraph is wrong. The STATUS.json already labels this file `"SUPERSEDED -- tested the wrong condition"`. |

No other active .md or .json file outside of the source `GATE1_VERDICT.md` and the
`dllm_reposition_salvage/` archive contains the stale phrase.

**MAIN action**: The file `GATE1_VERDICT.md` should have its verdict header updated from
`KILL_CONDITION_CLAUSE_2_TRIGGERED` to `SUPERSEDED_BY_DAMAGED_ARM_TEST` and a prominent
note added pointing to `GATE1_DAMAGED_VERDICT.md`. Do NOT delete the numeric results.

---

## Retracted verdict 2: "narrow A01 to OLMo-2 only"

**Background**: Same session, same reason as verdict 1. A01's `STATUS.json` field
`claim_scope_after_gates.RETRACTED_must_narrow` explicitly documents the retraction.

### Files still containing the stale "narrow A01 to OLMo-only" framing:

| File | Line | Stale text | Notes |
|------|------|-----------|-------|
| `proposal/active/A01-null-calibration-methodology/GATE1_VERDICT.md` | lines 12-16 | `"A01's claim must narrow from *'the letter MC interface is an unreliable instrument'* to *'the letter interface degenerates in structurally damaged OLMo-2 arms.'*"` | Same file as verdict 1 — both stale verdicts live here. The text is definitionally retracted by STATUS.json. |
| `status/scout_21/lane1_a01_gate1.md` | line 58 | `"Verdict: **narrowing gate for A01, kill gate for the spin-out.** More than decorative, not..."` | Scout file from the same session. It is a historical record of the scout's findings before the retraction. As a scout record it is lower priority than GATE1_VERDICT.md; MAIN may choose to leave it as-is (it represents what was known at that moment) or add a retraction header. |

No other active proposal or status file contains this framing.

---

## Retracted verdict 3: "outputs/lora_best_ref/ 512 bytes" (it is an empty directory)

**Background**: `A02/STATUS.json` claimed `lora_best_ref` "512 bytes — NOT adapter weights".
Two problems: (a) 512 bytes is the filesystem metadata for an EMPTY directory, not a small
file; (b) the actual canonical Read-LoRA is `outputs/qcmem_distill_qwen_j12_r32_4k/final/`
(222 MB, confirmed). This was noted in `TRAINER_ACTIVITY.jsonl` (2026-08-09T00:06) as a
doc bug but the A02 STATUS.json itself was not updated.

### Files still containing the stale "512 bytes" claim:

| File | Line/Section | Stale text | Notes |
|------|-------------|-----------|-------|
| `proposal/active/A02-comem-write-read-repair/STATUS.json` | `premise_check_2026_08_08.read_lora.suspect_size` | `"512 bytes -- NOT adapter weights (a real r=32 LoRA on Qwen3-8B is tens-to-hundreds of MB with adapter_config.json + adapter_model.safetensors or a .pt)"` | The "512 bytes" description is for `lora_best_ref/` which is an empty directory, not a 512-byte file. The broader point (it is not valid adapter weights) is correct, but the description misleads future readers into thinking there is a tiny file rather than nothing. |
| `proposal/active/A02-comem-write-read-repair/STATUS.json` | `premise_check_2026_08_08.read_lora.status` | `"NOT IDENTIFIED -- this is the blocker"` | This is stale: the canonical Read-LoRA IS identified as `outputs/qcmem_distill_qwen_j12_r32_4k/final`. The blocker is resolved. |
| `proposal/MINIMAL_VALIDATION_PLAN.md` | line 138 + line 146 | `"outputs/lora_best_ref/ 只有 512 字节"` and `"lora_best_ref 只有 512 B，需要先定位真正的 flagship adapter。"` | Historical planning doc. The adapter has now been identified; these notes are outdated. |
| `status/proposal_prep/NEXT_GATES_READY.md` | line 233 | `"**A02/STATUS.json calls lora_best_ref "512 bytes" — it is actually an empty directory"` | This file already has the CORRECT characterization ("empty directory, not 512B file"). No fix needed here. |

**MAIN action**: Update `proposal/active/A02-comem-write-read-repair/STATUS.json`:
- `read_lora.status` → `"CONFIRMED: outputs/qcmem_distill_qwen_j12_r32_4k/final (222MB, layers 12-35, sha dd09cd17)"`
- `read_lora.suspect_size` → `"lora_best_ref/ is an EMPTY DIRECTORY (0 bytes content); this was the doc bug"`
- `read_lora.status` in `next_gate[0]` → no longer needed (resolved)

---

## Retracted verdict 4: bare "10×" reference to A01's C4 span

**Background**: A01's gate-4 found that the span is 10.04× ONLY under the pre-registered
primary V1 aggregation, and drops to 9.98×/9.20×/8.91×/6.86× under four alternatives.
The consensus position (recorded in `GATE4_VERDICT.md` and `STATUS.json`) is that the
paper must print "~7–10×, primary 10.0×" or a range, not bare "10×" as a headline.

### Files containing bare "10×" as A01 headline (without the range caveat):

The search did NOT find any active paper-facing file (no `.tex` files in `paperA/sections/`,
no current `PROPOSAL.md` or `STATUS.json`) claiming bare "10×" as a finalized headline.
The references found in active documents are:

| File | Context | Type | Action needed? |
|------|---------|------|----------------|
| `proposal/active/A01-null-calibration-methodology/PROPOSAL.md` | Line 29: "稳妥表述是：残余比例约 **8%–77%**；不要把'恰好超过 10×'作为 headline" | Correct — this is the caution AGAINST "10×" | No action |
| `proposal/active/A01-null-calibration-methodology/PROPOSAL.md` | Line 78: "4. C4 aggregation 预注册，不再选择性报告 10×。" | Correct — gate-4 objective | No action |
| `proposal/active/A01-null-calibration-methodology/GATE4_VERDICT.md` | Multiple references to "≥10×" and "cannot print 10×" | Correct — this is the gate-4 analysis document | No action |
| `status/proposal_prep/NEXT_GATES_READY.md` | Line 58-60: "span=10.04×; every alternative drops below 10×. The spread is 6.86–10.04×" | Correct — documents the full range | No action |
| `proposal/active/A01-null-calibration-methodology/evidence/P1_four_constructs.md` | Line 128: "Residual fractions span **0.0769 → 0.7724 = 10.04×**. The gate (≥10×) **PASSES**." and line 142: "ranges from 8% to 77%, a ~7–10× spread" | Mixed — the GATE.PASSES sentence is technically correct for V1 only; the document also shows the range. | The gate-passes sentence should be accompanied by the qualifier "under V1 primary aggregation only." The document partially does this at line 142 but the standalone line 128 could be misread as unconditional. Low priority. |

**No active files were found claiming bare "10×" as an unconditional A01 headline.**
The gate-4 analysis has been correctly internalized into the active documents.

---

## Summary table: files MAIN should edit

| Priority | File | Issue | Action |
|----------|------|-------|--------|
| HIGH | `proposal/active/A01-null-calibration-methodology/GATE1_VERDICT.md` | Header says `KILL_CONDITION_CLAUSE_2_TRIGGERED`; verdict section claims "narrow to OLMo-only" | Add prominent retraction header: "VERDICT RETRACTED — see GATE1_DAMAGED_VERDICT.md. The numeric results below remain valid but the verdict was superseded by damaged-arm testing." Do NOT delete numeric results. |
| MEDIUM | `proposal/active/A02-comem-write-read-repair/STATUS.json` | `read_lora.status="NOT IDENTIFIED"`, `suspect_size="512 bytes"` | Update to reflect confirmed Read-LoRA at `outputs/qcmem_distill_qwen_j12_r32_4k/final/` (222 MB, sha dd09cd17, layers 12-35). Update `lora_best_ref` description to "empty directory". |
| LOW | `status/scout_21/lane1_a01_gate1.md` | States "narrowing gate for A01" (the pre-retraction verdict) | Add a note: "Superseded by GATE1_DAMAGED_VERDICT.md — this lane tested intact models only." Low priority as it is a scout record, not a decision document. |
| LOW | `proposal/MINIMAL_VALIDATION_PLAN.md` | Lines 138 and 146: "lora_best_ref 只有 512 B" | Cosmetic fix; the planning doc is effectively superseded by A02 STATUS.json and this launch plan. |
| LOW | `proposal/active/A01-null-calibration-methodology/evidence/P1_four_constructs.md` | Line 128 unconditionally says "gate PASSES" without V1-only qualifier | Add "under V1 (pooled-model) aggregation only" after the gate-passes verdict. |

---

## Files that do NOT need editing (already correct or properly scoped)

- `proposal/active/A01-null-calibration-methodology/STATUS.json` — correctly marks gate1
  verdict as "SUPERSEDED", documents retraction
- `proposal/active/A01-null-calibration-methodology/GATE1_DAMAGED_VERDICT.md` — correct
  general-claim-confirmed verdict
- `status/proposal_prep/NEXT_GATES_READY.md` — already says "empty directory, not 512B"
- `status/TRAINER_ACTIVITY.jsonl` — append-only record; retraction already appended
- All `GATE4_VERDICT.md`, `a01_gate4_c4_prereg.py` files — correct range reporting

---

## Files excluded from scan (not MAIN's action scope)

- `status/dllm_reposition_salvage/` — archived dllm planning files, unrelated to A01-A03
- `.claude/tasks/`, `.claude/workflows/` — auto-generated agent context
- `outputs/`, `models/` — binary data
- `.claude/worktrees/` — past worktree snapshots
