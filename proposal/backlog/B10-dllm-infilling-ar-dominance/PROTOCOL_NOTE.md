# B10 — PROTOCOL NOTE (terminal document)

**Written**: 2026-08-15 · **GPU used**: 0 (writing only; no measurement was made
for this document) · **Disposition authority**: `STATUS.json`
`disposition_20260815_MAIN` (`choice: "PROTOCOL NOTE, not archive"`), which is the
verbatim first branch of `kill_gate.gate_1.if_killed`.

**This document is the terminal form of B10.** It carries only the residue that
survives the direction's own kill gate. It adds no new claim, and it is not a
re-framing: the allow-list of what may appear here was fixed *before* it was
written, in `disposition_20260815_MAIN.what_the_note_may_carry`, and §5 records
what is foreclosed.

---

## 1. What this is not

**This is not a paper — or a claim — about AR-vs-diffusion infilling ability.**
That thesis was B10's headline, and B10's own pre-registered Gate 1 killed it on
2026-08-15.

Gate 1 re-scored all six existing arms on the **base** axis — the axis
HumanEval-SingleLineInfilling was designed for — from `solutions.jsonl` already on
disk (0 GPU, nothing regenerated). Its pre-registered kill condition, quoted
verbatim from `STATUS.json` `kill_gate.gate_1.kill_if`:

> on the base axis (gold ceiling 0.9894, so ≥98 % of items feasible), the
> `qwen_fim` vs `dreamon_oracle` paired contrast is **not** significant at
> α=0.05 **and** |Δ| < 0.02

Measured, with Δ = `qwen_fim` − `dreamon_oracle` (AR minus best diffusion arm;
positive = AR advantage), n = 1033 paired items:

| quantity | measured |
|---|---|
| exact McNemar p (two-sided) | **p = 1.0000** |
| discordant pairs | **77** (b = 39 qwen-only, c = 38 dreamon-only) |
| **Δ** | **+0.00096805** (+0.097 pp = **one item** of 1033) |
| Δ 95 % CI | **[−0.0164, +0.0183]** — lies entirely inside ±0.02 |

Both kill conditions hold → **KILL**. The thresholds **α = 0.05 and |Δ| < 0.02
were retained verbatim and unmodified**; in particular they were *not* rewritten
when the gold ceiling turned out to move (§3), because a post-hoc threshold edit
is precisely the failure mode this direction's retraction history forbids.

Two further facts make this final rather than merely unlucky:

- **All three axes now agree.** plus axis Δ = +0.0048 (p = 0.635); plus-axis
  gold-feasible subset Δ = −0.0012 (p = 1.000); base axis Δ = +0.00097
  (p = 1.0000).
- **The direction is not even stable.** On the 1022-item wzc1 feasible subset the
  sign flips to Δ = −0.00098 (exact McNemar p = 1.0000), which independently fails
  the gate's PROCEED requirement of a *directionally stable* AR advantage.

Consequently Gate 2 (lineage repair, 2–4 GPU-h) and Gate 3 (matched suffix-gain,
1–2 GPU-h) — both conditioned on Gate 1 saying PROCEED — are **not authorised**,
and there is no GPU-costing next step for B10 at all.

Full adjudication, including MAIN's independent recomputation of the McNemar
arithmetic from the raw per-item files: `GATE1_BASE_AXIS_VERDICT.md` (§4, §5, §9).

---

## 2. The four results that survive the kill

Quoted **verbatim** from `STATUS.json` `robust_findings_that_survive`, p-values
unaltered. None of these depends on the dead AR-dominance ranking claim.

1. > Suffix visibility is worth **+0.2314 (AR)** and **+0.2991 (diffusion)**, both
   > **p < 1e-56**; bidirectional context is an affordance of the FIM task
   > **FRAMING**, available to AR, not a property of the model class. This is the
   > most defensible result in the pilot.

2. > AR beats the two non-oracle-equivalent diffusion arms by **5–6 pp
   > (p ≤ 1.9e-09)**, and this survives ceiling restriction **(p ≤ 3.7e-06)**.

3. > The oracle length handout is worth **+5.7 pp** to DreamOn **(p = 4.1e-14)**,
   > so which diffusion configuration is the comparator decides whether AR "wins".

4. > Zero generation errors and a passing grader self-test in all six arms; merge
   > asserted 8/8 shards and no duplicate ids.

### 2.1 The reading of (1), stated explicitly so it cannot be misquoted

Survivor (1) is the strongest result here, and it is **framing-level, not
model-class-level**. The measured gain from making the suffix visible is large for
*both* families (+0.2314 AR, +0.2991 diffusion, both p < 1e-56). The correct
interpretation is therefore:

> **Bidirectional context is an affordance of the FIM task FRAMING, which an
> autoregressive model can also exploit — it is not a property of the model
> class.**

It must **not** be read as "diffusion benefits more from bidirectionality, hence
diffusion has a structural advantage". The two gains are not measured under
matched conditions (`dream_prefix` retains an oracle length while `qwen_prefix`
does not), and Gate 3 — which existed to equalise that handout — is unauthorised
and will not be run.

---

## 3. The protocol finding this gate *produced*: the gold ceiling is not reproducible across hosts

This is the main new output of Gate 1, and it is a statement about the **harness**,
not about any model or arm. The **gold ceiling** is obtained by splicing the
benchmark's own gold middle back in (`prompt + canonical_solution + suffix`) and
grading it; it is the denominator every arm is normalised against. Measured on the
identical split (1033 rows, md5 `30129634e180d80c19d6ddcd4cf43f9c`, byte-identical
on both disks):

| axis | wzc1 (`NUMBER_AUDIT.md:284`) | zwfy6 (Gate 1) | moved items | which items |
|---|---|---|---|---|
| base | 0.9894 (1022/1033) | **1.0000** (1033/1033) | **11** | every line `L0`–`L10` of `HumanEval/32` (`find_zero`) |
| plus | 0.8025 (829/1033) | **0.8122** (839/1033) | **10** | 9 lines of `HumanEval/130` (`tri`) + `HumanEval/15/L0` (`string_sequence`) |

Closure is asserted, not eyeballed: `1033 − 1022 = 11` and `839 − 829 = 10` match
the per-item diff sets exactly, the `task_id` sets are identical, and **all 21
discordant items move in the same direction** (wzc1 FAIL → zwfy6 PASS). The two
axes have **two different, separately demonstrated mechanisms**, and both are
**silent** — neither surfaces to the caller as an error.

### 3.1 Base axis — an evalplus **version** defect (11 items)

The wzc1 ceiling run resolved `import evalplus` to **PyPI evalplus 0.3.1** in its
venv's `site-packages`. In 0.3.1, `eval/__init__.py:unsafe_execute` handles the
`find_zero` special oracle with a **bare `continue`**:

```
# PyPI evalplus 0.3.1, eval/__init__.py
187:  if "find_zero" == entry_point:
188:      assert abs(_poly(*inp, out)) <= atol
189:      continue                       # <-- no details[i] = True, no progress.value += 1
```

whereas the repo's vendored evalplus (upstream commit `26d6d00`) records the pass
first:

```
# vendored evalplus @ 26d6d00, eval/__init__.py
187:  if "find_zero" == entry_point:
188:      assert abs(_poly(*inp, out)) <= atol
189:      details[i] = True
190:      progress.value += 1
191:      continue
```

Because 0.3.1 `continue`s before recording, `progress.value` stays 0,
`untrusted_check` returns `details = []`, and **its own guard**
`if len(details) != len(inputs): stat = FAIL` **rewrites a genuine PASS into a
FAIL**. Measured on **one host, same data files, same grade wrapper, only
`PYTHONPATH` differing**, probe item `SingleLineInfilling/HumanEval/32/L0`:

| grader | status | `n_details` | `n_inputs` | `n_pass` |
|---|---|---|---|---|
| PyPI evalplus **0.3.1** | `fail` | **0** | 100 | 0 |
| vendored (`26d6d00`) | `pass` | **100** | 100 | 100 |

Deterministic across 3 repeats per version — **not** host load, **not**
wall-clock flakiness. `HumanEval/32` is the **only** `find_zero` task in the split,
which is exactly why the base-axis discrepancy is precisely its 11 rows and
nothing else. (Both line numbers and the source difference were read directly from
the two files by MAIN — `GATE1_BASE_AXIS_VERDICT.md` §9.2 — not taken from a
quoted snippet.)

### 3.2 Plus axis — the sandbox's **4 GiB `RLIMIT_AS`** (10 items), host-dependent

A *different* mechanism, and **not** a version difference: with the **same
vendored evalplus**, LOCAL/wzc1 still fails these items and `.73`/zwfy6 passes
them. `query_maximum_memory_bytes()` defaults to 4 GiB and `reliability_guard()`
applies it as `RLIMIT_AS`/`RLIMIT_DATA` inside the grading subprocess.
`HumanEval/130` and `HumanEval/15` carry `plus_input` entries with n ≈ 10⁶ whose
reference outputs are ~10⁶-element lists; materialising them exceeds 4 GiB of
address space once the interpreter's own footprint counts. The bare `MemoryError`
is swallowed by `unsafe_execute`'s `except BaseException` and **booked as a wrong
answer rather than an error**. Causal test on LOCAL, vendored evalplus,
`SingleLineInfilling/HumanEval/130/L0`:

| `RLIMIT_AS` | result | exception |
|---|---|---|
| 4 GiB (evalplus default) | 7 / 125 inputs fail | `MemoryError` at `n = 999999, 999997, …` |
| unlimited (`EVALPLUS_MAX_MEMORY_BYTES=-1`) | **0 / 125 fail, status `pass`** | — |

Because the trip point depends on the host's baseline footprint, this is a
**cross-host reproducibility defect of the harness**, not a property of the
benchmark or of any arm.

### 3.3 The reusable methodological lesson

> **Byte-identity of a vendored file is not evidence about which file the
> interpreter loaded.**

This is worth stating as a rule because it produced a wrong exculpation *inside
this very direction*. An earlier reading verified that the **vendored**
`evalplus/eval/*.py` are byte-identical across both disks (md5
`bcd21dfd412e10b6825fab093428d579` — true) and inferred from that md5 match that
the grader was excluded as the cause of the ceiling discrepancy. The md5 claim was
correct; the inference was wrong. The wzc1 ceiling run **never imported the
vendored copy at all** — its venv resolved `evalplus` to PyPI 0.3.1 in
`site-packages`. Verified by resolution, not by inference:

```
$ dllm_draft/.venv_b200/bin/python -c "import evalplus.eval as E; print(E.__file__)"
.../dllm_draft/.venv_b200/lib/python3.11/site-packages/evalplus/eval/__init__.py
```

and no `site-packages/evalplus` exists on `.73` at all, so the zwfy6 run could
only have loaded the vendored copy.

Operationally: when two hosts disagree about a graded number, hashing the
vendored source is **not** a sufficient check. Print `module.__file__` and the
package version *from the interpreter that produced the number*.

### 3.4 Which ceiling is authoritative, and why it does not touch the KILL

**Authoritative: zwfy6 / Gate 1 — base `1.0`, plus `0.8122`** (`n_gold_base_pass`
1033, `n_gold_plus_pass` 839). Both wzc1 deviations are harness defects with
identified mechanisms; neither is a property of the benchmark.
`NUMBER_AUDIT.md:284` is **superseded** — per `LIFECYCLE_SCHEMA.md` §0 its
original line is left byte-intact with a dated append-only note.

The KILL does not depend on this. `kill_if` is a function of the paired contrast
**only**; the ceiling enters solely as the parenthetical precondition "so ≥98 % of
items feasible", and a measured base ceiling of **1.0 = 100 % feasible** satisfies
that *a fortiori* — the change is favourable, not adverse. Gate 1 was adjudicated
on **both** ceiling readings (1033 items under zwfy6; 1022 under wzc1) and returns
exact McNemar **p = 1.0000** with **|Δ| < 0.001** under each.

---

## 4. Scope of this note

Everything above, and nothing else. The allow-list was fixed before drafting
(`STATUS.json` `disposition_20260815_MAIN.what_the_note_may_carry`):

1. the four entries of `robust_findings_that_survive`, verbatim, with their
   p-values → §2;
2. the cross-host gold-ceiling irreproducibility with **both** root causes named
   → §3.1, §3.2;
3. the methodological point that byte-identity of a vendored file is not evidence
   about which file the interpreter loaded → §3.3.

---

## 5. What this note forecloses (binding)

Quoted from `STATUS.json` `disposition_20260815_MAIN.forbidden_in_the_note`:

1. **Any re-framing that hunts a different ranking from the same six arms.** That
   is the nested-ladder error this line of work has **already retracted twice**
   (Retractions 6 and 7). A KILL is not an invitation to re-cut the same 1033
   items until some pair clears α.
2. **The full pairwise base-axis matrix is provenance, NOT a menu.** It lives in
   `evidence/gate1_base/gate1_base_stats.json` and is reproduced in
   `GATE1_BASE_AXIS_VERDICT.md` §4.3 so that nobody has to re-run it — *not* so
   that a new headline can be selected from it. `qwen_fim`–`dreamon_oracle` is the
   pre-registered contrast and is the only pair the gate adjudicates.
3. **Gate 4's standing rule still binds**: no absolute pass@1 from this surface may
   be reported as a capability measurement **without a decontaminated companion**.
   `KSPAN_INFILLING_RESULTS.md` §4.5 measured a 26–28 pp drop from identifier
   renaming + docstring replacement on a set whose gold refill still scores 1.000,
   so any absolute number here is substantially surface recall. This applies to
   every pass@1 in this note and in `GATE1_BASE_AXIS_VERDICT.md` §3.

Additionally, and for the same reason B10 exists as an audit rather than a claim:
the `MUST NOT CLAIM` list in `PROPOSAL.md` §4.4 and `STATUS.json`
`novelty_verdict.must_not_claim` stand unchanged, and are now additionally
supported on the base axis.

---

## 6. Provenance

Every number in §1 and §3 is recomputable from files on disk. Root:
`proposal/backlog/B10-dllm-infilling-ar-dominance/`.

### Adjudication documents

| file | role |
|---|---|
| `GATE1_BASE_AXIS_VERDICT.md` | full Gate-1 verdict; §4 contrast, §5 adjudication, §"Ceiling discrepancy vs `NUMBER_AUDIT.md:284`" for §3 of this note, §8 provenance, **§9 MAIN's independent verification** |
| `STATUS.json` → `gate_1_result` | machine-readable KILL record |
| `STATUS.json` → `gold_ceiling_discrepancy_ROOTCAUSED_20260815` | machine-readable root-cause record for both axes |
| `STATUS.json` → `disposition_20260815_MAIN` | the allow-list and forbid-list this note obeys |
| `NUMBER_AUDIT.md` | origin of the survivor p-values in §2; its line 284 is superseded (dated append-only note at end of file) |

### Evidence files — `evidence/gate1_base/`

| file | md5 | what it carries |
|---|---|---|
| `gate1_base_stats.json` | `804056f7f9dbb015c4c05dc483d03fa6` | the §1 contrast, both ceiling readings, the pairwise matrix, `exact_stat_implementation_selfcheck.all_ok = true` |
| `gate1_integrity.json` | `cbd8c0ea70a7975cdf8f8b4d4657342a` | the pre-statistics integrity assertions and the six grader self-tests (survivor 4) |
| `gate1_gold_ceiling_zwfy6.json` | `f770e2893bfb2fefa117e604375a9f55` | zwfy6 ceiling, per item, both axes |
| `gold_ceiling_SingleLine_wzc1_reference.json` | `21b4766e824e9210adf0d6ce08240eea` | the wzc1 ceiling record, byte-identical copy of the original |
| `ceiling_discrepancy_rootcause.json` | `ca0c4b09a9239f75f0593101fefb4a44` | §3.1 version probe + §3.2 `RLIMIT_AS` causal test |
| `ceiling_discrepancy_rootcause.py` | — | the script that produced the row above |
| `score_base/qwen_fim_score_base.json` | `55c11d014c13218d278d4826f8f6688b` | **966/1033** `per_task[].pass` — one of the two arms Δ is computed from |
| `score_base/dreamon_oracle_score_base.json` | `1f653ffb60a053b5fea041a5e9e834ac` | **965/1033** — the other; 966 − 965 = the one item Δ consists of |
| `score_base/dream_fim_score_base.json` | `2a29a03333845836083d68d510f55c08` | `per_task[].pass`, 1033 rows |
| `score_base/dreamon_fim_score_base.json` | `6c03dd1a11a82ac0179decbae936c32f` | `per_task[].pass`, 1033 rows |
| `score_base/qwen_prefix_score_base.json` | `e14b6392e5b8a83634588e021c2bf43e` | `per_task[].pass`, 1033 rows |
| `score_base/dream_prefix_score_base.json` | `d6e69cbcc403448a686092fe26040496` | `per_task[].pass`, 1033 rows |

> The four non-primary arms' base-axis pass@1 values are deliberately **not
> tabulated here**. They exist in `GATE1_BASE_AXIS_VERDICT.md` §3 as provenance;
> reproducing them as a leaderboard in a terminal note is the first step of the
> re-ranking that §5 item 1 forbids, and every one of them is additionally subject
> to the gate-4 rule in §5 item 3.
| `replicate/qwen_fim_score_base_rep2.json` | `55c11d014c13218d278d4826f8f6688b` | = rep1 ⇒ **0 per-item flips** |
| `replicate/dreamon_oracle_score_base_rep2.json` | `1f653ffb60a053b5fea041a5e9e834ac` | = rep1 ⇒ **0 per-item flips** |
| `gate1_integrity_assert.py`, `run_gate1_base_rescore.sh`, `gate1_gold_ceiling.py`, `gate1_stats.py` | — | re-runnable pipeline (also on zwfy6 `dllm_draft_104/scripts/`) |
| `scorer_logs/*.txt` | — | per-arm scorer logs incl. the six grader self-tests (`.txt` because the repo `.gitignore` drops `logs/` and `*.log`) |

### How the numbers were produced

- **Scorer**: `dllm_draft_104/scripts/score_infilling.py`, md5
  `41a5dd1816a7ef8a51e66f43d33ef730`, **unmodified** (`--which base` is an
  existing flag). Grader: `evalplus.eval.untrusted_check`, **vendored**
  `dllm_draft/vendor/evalplus` @ `26d6d00` (`evalplus.__version__ 0.1.0.dev1`).
- **Split**: `data/humaneval_infilling/HumanEval-SingleLineInfilling.jsonl`,
  1033 rows, md5 `30129634e180d80c19d6ddcd4cf43f9c` (identical on both disks).
  EvalPlus dataset `HumanEvalPlus-v0.1.10.jsonl`, md5
  `fe585eb4df8c88d844eeb463ea4d0302` (identical on both disks).
- **Statistics**: exact McNemar (two-sided exact binomial on discordant pairs; no
  χ², no continuity correction) as the pre-registered primary; paired bootstrap
  10 000 item-level resamples, seed `20260815`, as corroboration (p = 0.9498).
  No scipy on either disk, so the hand-rolled exact tests are validated in-run.
- **Where it ran**: `.73` (`28.85.35.73`, 8×H20, zwfy6),
  `.venv_dream` = Python 3.11.6, numpy 1.26.4, `dllm_draft_104` @ `3555dc79`.
  **0 GPU**: every step forced `CUDA_VISIBLE_DEVICES=""`; all 8 H20s stayed
  96421 MiB / 99–100 % occupied by the pre-existing `train_olmo2_arch_probe2.py`
  DDP job before and after (`GATE1_BASE_AXIS_VERDICT.md` §6).
- **Non-destructive**: base-axis output went to `score_base.json`; the
  pre-existing plus-axis `score.json` — provenance for the numbers already in
  `STATUS.json` — was not overwritten.
- **Independently re-verified by MAIN** from the raw `per_task[].pass` arrays,
  ignoring every summary field: all six arms reproduce, and MAIN's own exact
  McNemar gives b = 39, c = 38, p = 1.000000, Δ = +0.00096805 (§9.1). The
  source-level evalplus claim in §3.1 was likewise verified by reading both files
  and resolving the import (§9.2).

---

## 7. Terminal state

`kill_gate.gate_1` **FIRED** (2026-08-15). The disposition chosen was **protocol
note, not archive**, so that the four survivors in §2 and the harness finding in
§3 stay citable rather than being buried — none of them depends on the dead
ranking claim. With this note on disk, B10's residual value is sealed and
`lifecycle` moves to its terminal value (`dead`, per `LIFECYCLE_SCHEMA.md` §1:
*"已证伪 / 已关闭 … claim 不自动复活"*). No GPU work remains; Gates 2 and 3 are
unauthorised.

`STATUS.json` records this at key `protocol_note_20260815`.
