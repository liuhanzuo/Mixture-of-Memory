# B09 GATE -1 — PRE-REGISTRATION of the §4.3 / H3 re-scope

- **Written**: 2026-08-17T12:51:55+0800 (2026-08-17T04:51:55Z)
- **GPU used**: **none** (0 GPU-seconds). This file is 0-network, 0-dependency.
- **Written BEFORE**: any parquet shard was downloaded, and therefore before any row,
  any repo distribution, and any outcome was seen. Evidence anchoring that ordering is in
  §0 below and is independently checkable from file mtimes.
- **Supersedes nothing.** This is a NEW artifact; `ls` on 2026-08-17 confirmed the B09
  directory had no file matching `*PREREG*`.
- **Machine-readable pointer**: `STATUS.json` key `gate_minus1_prereg_20260817`
  (added in the same session) records this path. No tool reads this file automatically.

---

## 0. Why the ordering matters, and the evidence that it was respected

`GATE_MINUS1_FEASIBILITY_20260816.md` §6 item 4 says:

> **Pre-register the benchmark-breadth re-scope of §4.3 / H3 to repo-family holdout
> BEFORE looking at any outcome.** Deciding this after seeing results is exactly the
> leakage §4.3 exists to prevent.

That item is **5th of 6** in the reading order of the next-action list, so the natural
execution order puts it LAST — after the pool has been built and its repo distribution
inspected. At that point the "pre"-registration is retrospective and §4.3's leakage
control is void, while the artifact on disk still looks like a valid pre-registration
with a timestamp. **Nothing in the repository enforces this ordering.** Only file mtimes
would reveal a violation, and only if someone thought to check.

So, the check, recorded here so a future reader can run it:

```bash
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
stat -c '%y %n' proposal/backlog/B09-trajectory-aware-sft-data-selection/B09_GATE_MINUS1_PREREG.md
stat -c '%y %n' data/agent_traj/*.parquet 2>/dev/null || echo "no parquet on disk"
```

**State at the moment of writing (measured, not asserted):**

| fact | command | result |
|---|---|---|
| landing dir absent | `ls -la data/agent_traj` | `No such file or directory` (rc=2) |
| no parquet anywhere in `data/agent_traj` | same | dir does not exist |
| no PREREG existed | `ls B09.../ \| grep -i prereg` | rc=1, no match |
| clock | `date -u` | `2026-08-17T04:51:55Z` |

**If any `data/agent_traj/*.parquet` has an mtime EARLIER than this file's mtime, this
pre-registration is void and must be treated as retrospective.** I am stating the
falsifier explicitly because a pre-registration whose own violation is undetectable is
decoration.

---

## 1. What is being re-scoped, and why it is forced

### 1.1 The commitment in PROPOSAL.md that cannot be honoured

PROPOSAL.md §4.3 defines three evaluation settings. Setting 2 reads (verbatim, lines
216-219):

> 2. **Strict benchmark-OOD**
>    - 整个 benchmark/family 不参与 selector tuning 和 query construction；
>    - 若只有 20 个 benchmark，使用 5-fold benchmark holdout。

and §9 H3 reads (verbatim, lines 526-534):

> ### H3 — Target relevance mainly helps ID
> \[ \Delta_{\mathrm{relevance}}^{ID} > \Delta_{\mathrm{relevance}}^{OOD} \]
> 严格 OOD 下 pool/skill coverage 应比 target surface similarity 更稳健。

Both presuppose **B > 1 benchmarks**. The §4.3 fallback ("若只有 20 个 benchmark") already
contemplates a small B and prescribes 5-fold benchmark holdout — but its floor is 20, not 1.

### 1.2 The measured fact that forces the re-scope

`GATE_MINUS1_FEASIBILITY_20260816.md` §5 established, from exact HF `/filter` counts over
the full split (not samples):

- The only corpus satisfying all four GATE -1 fields natively is
  `SWE-bench/SWE-smith-trajectories`, `tool` split, licence `mit`, `gated:False`.
- It is **single-benchmark**: SWE / Python repo repair, one benchmark, a 3-verb tool
  vocabulary (`bash`, `str_replace_editor`, `submit`).
- The multi-environment alternatives that would give benchmark breadth
  (`AgentGym/AgentTraj-L` 12 envs, `Solaris99/AgentBank` 16+ envs,
  `agent-eto/eto-sft-trajectory` 3 envs) **all lack a per-row success/reward field**, and
  two of three declare **no licence at all**. Breadth and reward are, in the currently
  downloadable public set, mutually exclusive.

Therefore **B = 1**, benchmark holdout is undefined, and no drop-in fix exists. This is a
scientific scoping decision, not a download problem.

### 1.3 The re-scope, registered

**§4.3 setting 2 "Strict benchmark-OOD" is replaced, for all B09 experiments on this
corpus, by "Strict repo-family-OOD".** H3 is correspondingly re-scoped: the OOD arm of
H3's inequality is measured across held-out **repo families**, not across held-out
benchmarks.

This is a **weaker claim** than the one PROPOSAL.md wrote, and B09 must say so in any
write-up. The specific weakening: a repo-family holdout varies the codebase, its
idioms, its test harness and its dependency surface, but it holds constant the
benchmark, the task format (SWE-bench-style issue→patch), the agent scaffold, the
observation format and the 3-verb tool vocabulary. So a positive result under
repo-family holdout does **not** license the sentence "generalises out of distribution";
it licenses "generalises to unseen repositories within SWE-style Python repair".

---

## 2. The holdout axis, defined operationally BEFORE seeing the distribution

### 2.1 Primary axis: `repo_family`

`instance_id` in this corpus is structured as `<repo>.<task-generation-suffix>`. Observed
values in the feasibility audit (n=10, from `/filter` counts, so these are real strings
from the corpus and not invented): `MONAI.pr_7187`, `MONAI.pr_6735`,
`MONAI.lm_rewrite__roex7dhi`, `conan.pr_13390`, `conan.pr_12880`,
`voluptuous.lm_rewrite__9e2nb1af`, `flake8.lm_rewrite__tmxef7uw`, `schedule.6`,
`django-money.func_pm_ctrl_shuffle`, `starlette.lm_rewrite__wq8ip1tf`.

**Registered definition:**

```
repo_family(instance_id) := instance_id.split(".", 1)[0]
```

i.e. everything before the FIRST `.`. On the ten observed strings this yields
`{MONAI, conan, voluptuous, flake8, schedule, django-money, starlette}`.

**Registered contingencies, decided now rather than after inspection:**

| contingency | pre-registered decision |
|---|---|
| an `instance_id` contains no `.` | the whole string is the family; count these and report the count |
| an `instance_id`'s prefix differs from another only by case or by `-`/`_` | do NOT merge automatically; report both spellings and their row counts, and merge only if the two spellings' `repo_family` sets are disjoint in `traj_id` — a merge decision is a REPORTED decision, not a silent one |
| the prefix is a bare integer | treat as its own family; count and report |
| a family has < 5 trajectories | it is still a family; it is assigned to a fold, it is not dropped (dropping small families would silently shrink the population, cf. §4) |

The definition is deliberately **crude and mechanical**. A smarter normalisation (e.g.
mapping `django-money` and `django` to one owner) would be a judgement call made after
seeing the data, which is precisely what this file exists to prevent.

### 2.2 Fold count: K = 5, with a stated fallback

**Registered: K = 5 folds over `repo_family`, families assigned to folds greedily by
descending trajectory count (largest-remainder balancing on trajectory count, not on
family count), seed 0.**

Rationale for 5: it is the number §4.3 itself names ("使用 5-fold benchmark holdout"), so
keeping K=5 preserves the one quantitative commitment the proposal did make while
changing only the unit.

**Fallback, registered now:** if the number of distinct `repo_family` values is < 5, K
becomes that number and this must be reported as a further weakening. If it is < 3,
**repo-family holdout is ALSO undefined and H3 must be reported as UNTESTABLE on this
corpus** rather than tested on some third substitute axis. I am registering this floor so
that a thin repo distribution cannot be quietly rescued by inventing a new axis later.

### 2.3 Secondary axis, registered as a SEPARATE and SUBORDINATE analysis

The suffixes above are not noise: `pr_*`, `lm_rewrite__*`, `func_pm_*` and bare integers
are SWE-smith's **bug-generation strategies**, i.e. how the defect was synthesised. A
holdout on that axis asks a different and interesting question (does the selector
generalise to unseen defect-synthesis processes?).

**Registered: strategy-holdout is a SECONDARY, exploratory analysis. It is not H3.** It
may be reported, must be labelled exploratory, and may not be substituted for the primary
repo-family result if the primary result is unfavourable. Registering it here — before
seeing either result — is what stops it from becoming a post-hoc rescue.

```
task_strategy(instance_id) := (
    "pr"        if suffix startswith "pr_"
    "lm_rewrite" if suffix startswith "lm_rewrite"
    "func_pm"    if suffix startswith "func_pm"
    "numeric"    if suffix is all digits
    "other"      otherwise            # report the raw suffixes falling here
)
```

### 2.4 What "holdout" binds

For a held-out fold F, **none** of the following may touch any row whose
`repo_family ∈ F`:

- selector fitting, scoring, or any threshold/weight choice (PROPOSAL §2.1 TF-0/TF-F/LT
  all included);
- selection-query construction Q (§4.2);
- validation V used to freeze weights (§4.2);
- near-duplicate graph construction, if the graph is used to make selection decisions.

This is the §4.1 partition requirement lifted from `family_key` to `repo_family`. The
§4.1 objects that must stay whole inside one partition still must: all rows of one
`traj_id`, **all rollouts of one `instance_id`**, and any near-duplicate connected
component. `repo_family` is a coarsening of `instance_id`, so grouping by it
automatically keeps sibling rollouts together — that is a reason to prefer it over any
finer key.

---

## 3. The group variable — registered, because the records state it two ways

This is not strictly part of the §4.3 re-scope, but it is order-critical for the same
reason and the existing records are ambiguous, so it is registered here.

`STATUS.json`'s inherited phrasing ("9116 parents is within 9% of |G|≈10000") invites
setting `g(i) = traj_id` and emitting **9116 groups of size 1**. In the feasibility
audit's 44-row sample `traj_id` was **unique in 44/44 rows**; in an independent 50-row
sample at offsets 0/5000/12000/20000/24000 it was unique in **50/50**. If `traj_id` is
1:1 with parquet rows, then:

- the hard parent-multiplicity cap (§5 Stage 1, `min(3, ⌈0.3T⌉)`) is **vacuous**;
- "sibling redundancy" (§3, §8) is **undefined**;
- trajectory-stratified constraint-matched random becomes **identical** to plain random,
  so **kill_criteria #1** ("constraint-matched random matches TA-Coreset") fires for a
  purely structural reason and B09 is killed by an artifact of its own data prep.

**Registered decomposition:**

```
GROUP (the partition/cap unit)  g(i) = traj_id        # one agent rollout
ROW   (the selectable unit)     one DECISION TURN extracted from the `messages` JSON
PARENT TASK (the split unit)    instance_id            # 1-13 rollouts measured
HOLDOUT UNIT                    repo_family            # §2.1
```

i.e. a parquet ROW is a **trajectory**, not an SFT row. `|G|` = number of usable
trajectories; `|U|` = number of decision turns derived from them. The
"1-13 rollouts per instance" figure is at `instance_id` granularity, and the
`|U| ≈ 2.95e5` figure is turns, not rows.

**Registered assertion, to be run on the parquet:** if `traj_id` is NOT unique per row
over the full split, this decomposition is wrong and must be revised in a dated
superseding record — not silently. Report the exact duplicate count either way.

---

## 4. The exclusion that must be reported, not absorbed

24,100 rows are in the `tool` split; ~9,116 carry both `message_type` and `tool_calls`.
A 50-row structural sample split cleanly **30 structured / 20 bare**, where the bare rows
parse to lists of `{content, role}` only — no `message_type`, no `tool_calls`, no
`thought`. So roughly **38% of the split is usable and ~62% is plain chat**.

**Registered: the exclusion rate must be reported as a first-class number wherever the
pool size is reported, and the exclusion must be crosstabbed against `model` before it is
called benign.** `model` is a column (claude-3-7-sonnet 17,715 / claude-3-5-sonnet 5,751 /
gpt-4o 634). If structure correlates with `model`, the "SWE-smith tool split" is actually
"the subset emitted by one scaffold", and that is a hidden confound for every downstream
selector comparison — it must be named in the write-up, not discovered by a reviewer.

**Registered: `resolved` must also be crosstabbed against structure.** The feasibility
audit measured structured ∧ resolved=true = 3,452 and structured ∧ resolved=false = 5,664
(sum 9,116 ✓), i.e. 37.9% resolved inside the structured subset versus 39.1% (9,427/24,100)
over the whole split. Those are close, which is mild evidence the exclusion is not
outcome-selective — but it is one axis, and `model` is the axis that worries me.

---

## 5. What this file does NOT pre-register

Being explicit, so later additions cannot be passed off as having been registered here:

1. **Selector hyperparameters.** §5 Stage 2 already says the pilot does not pre-register
   final weights, only three ordering constraints (validity/verifier highest; PPL/IFD not
   the sole quality signal; weights frozen on validation before test opens). Unchanged.
2. **The eight-way decision taxonomy.** §5 Stage 1's `PLAN … FORMAT_ONLY` labels are
   assigned by the method, not by the corpus. The corpus supplies only
   `message_type ∈ {system_prompt, action, observation}` plus a positional index as the
   substrate. Unchanged.
3. **Any target-token budget, any |S|.** §1 says |S| = 5000 of |U| ≈ 100000; the actual
   |U| must be recomputed on the parquet (§3) and if it differs materially the budget is
   a separate, dated decision.
4. **The in-domain setting.** §4.3 setting 1 (in-domain inductive) and setting 3
   (target-aware OOD) are unaffected by B=1 and are not re-scoped. Only setting 2 is.
5. **Whether `nebius/SWE-agent-trajectories` gets added.** It has a richer reward signal
   (3 channels) but lacks step type and tool family as fields; adding it is a separate
   decision with its own licence conditions (per-repo licences + Llama-3.1 notice) and its
   own pre-registration.

---

## 6. Honest limits of THIS pre-registration

1. **It cannot make B=1 into B>1.** Re-scoping to repo-family holdout is damage control.
   If a reviewer's objection is "your OOD claim is within one benchmark", this file
   concedes that objection in advance rather than answering it.
2. **The `repo_family` definition is registered on n=10 observed `instance_id` strings**
   (from the feasibility audit's `/filter` probes). If the real distribution has forms
   those ten do not exhibit, §2.1's contingency table is what governs — and that table was
   written blind, so it may be a poor fit. It may be revised **only** by a dated
   superseding record that states what was seen and why the blind rule failed.
3. **The 30/20 structural split and the traj_id uniqueness are samples** (n=50 and n=44+50
   respectively), not full-split counts. §3 and §4 register assertions to be run on the
   parquet; they do not report results.
4. **No enforcement.** Nothing in the repo will stop a later agent from ignoring this
   file. Its only teeth are (a) the mtime check in §0 and (b) the `STATUS.json` pointer.
5. **I did not download anything.** The corpus-level numbers quoted here are all carried
   from `GATE_MINUS1_FEASIBILITY_20260816.md` §3.1, which obtained them from HF
   `datasets-server` `/filter num_rows_total` over the full split. They are exact counts
   from the HF index, not from the parquet on disk — and the whole point of next-action
   item 1 is that the index and the parquet must be checked against each other.

---

# APPENDIX A — post-hoc record, appended 2026-08-17 AFTER the parquet was parsed

**Everything above this line was written before any shard existed and is UNCHANGED.**
This appendix exists because §2.1 and §6.2 said a blind rule may be revised only by a dated
record stating what was seen and why the rule failed. Machine-readable twin:
`STATUS.json:prereg_rule_correction_20260817`.

## A.1 §2.3's `task_strategy` rule was WRONG — degenerate, not merely imprecise

`instance_id` is **`<owner>__<repo>.<commit8>.<strategy>`**, not `<repo>.<strategy>`.
Measured over 5,215 distinct ids in the pool: 5,177 have exactly 2 dots, 38 have 3.

The blind rule read segment `[1]`, which is the **commit hash**, so it returned `"other"`
for **9,116 of 9,116** trajectories — a constant. Corrected to segment `[2]`:

| task_strategy | trajectories |
|---|---|
| `pr` | 4311 |
| `lm_rewrite` | 1718 |
| `func_pm` | 1284 |
| `combine_module` | 827 |
| `combine_file` | 698 |
| `numeric` | 197 |
| `other:1a8bd2f7` | 54 |
| `func_basic` | 27 |

`combine_file`, `combine_module` and `func_basic` were **not in §2.3's enumeration at all**.
The 54 `other:1a8bd2f7` are the 3-dot ids whose segment `[2]` is a second hash.

**Why the reference strings misled me.** §2.1 registered the rule on ten `instance_id`
strings quoted in `GATE_MINUS1_FEASIBILITY_20260816.md` §3.1. **None of the ten exists
verbatim** — 0/10 exact matches against the 10,637 distinct ids of the full split. They are
abbreviations with the owner prefix and commit hash stripped. Nine resolve uniquely by
suffix (`...pr_7187` → `Project-MONAI__MONAI.a09c1f08.pr_7187`); `func_pm_ctrl_shuffle`
matches nothing even by suffix. A rule inferred from their *shape* could not have been right.

**This correction makes B09's position better** (a dead axis becomes an 8-level one), which
is the direction that deserves the most scrutiny. It does not touch H3, which is uncomputed.

## A.2 §2.1's `repo_family` rule was RIGHT — but not because I reasoned correctly

Split-on-first-dot yields exactly `owner__repo`: **128 families**, largest
`getmoto__moto` (25,558 rows), `pandas-dev__pandas` (15,336), `iterative__dvc` (14,940).
The *same* rule that killed the secondary axis produced the intended unit for the primary
one. So §2.1 is confirmed, but the confirmation is luck; it is not evidence that the blind
contingency table in §2.1 is well designed. Of that table, only the "bare integer" branch
fired (197 trajectories) and it fired on the *strategy*, not the family. No id lacked a dot
(0), so that branch is untested.

## A.3 §2.2's K=5 stands; neither fallback fires

128 distinct families ≥ 5. Folds (groups / rows): 1823/60236, 1823/52320, 1823/52919,
1823/53105, 1824/59778. **Verified: 0 `traj_id` and 0 `instance_id` straddle a fold** — §2.4
holds by construction because `repo_family` coarsens `instance_id`. The assignment is
materialised as a `prereg_fold` **column in the pool parquet**, so no consumer can silently
re-derive a different partition.

## A.4 §3's group decomposition was forced, and is now confirmed at full scale

`traj_id` is unique in **all 24,100** parquet rows (0 duplicates). Under the registered
decomposition (row = decision turn) the pool has |U| = 278,358 rows over |G| = 9,116 groups:
rows/group min 3, max 151, mean 30.54, **Gini 0.3329**. Had the row been the parquet row,
every group would have had size 1 and `kill_criteria` #1 would have fired for a purely
structural reason. §3's registered assertion ("if traj_id is NOT unique per row this
decomposition is wrong") resolved in favour of the decomposition.

Sibling substrate for §5's branch-verified credit: 1,764 instances have >1 rollout and
**474 instances have rollouts with differing `resolved` outcomes**.

## A.5 §4's exclusion check — one axis clean, one NOT

62.174% (14,984/24,100) excluded. Structure rate by `model`:
gpt-4o **83.28%** (528/634), claude-3-5-sonnet **47.19%** (2714/5751),
claude-3-7-sonnet **33.16%** (5874/17715) — a **2.51× spread**. The structured subset is a
model-reweighted sample (gpt-4o 2.63%→5.79% of the pool; claude-3-7 73.51%→64.44%).
**Not benign.** By `resolved`: structured 37.87% vs bare 39.88% vs split 39.12% — a 2.01pp
difference, benign; the exclusion is not outcome-selective.

Required framing, per §4: this pool is **"the SWE-agent-formatted 37.8% of the SWE-smith
`tool` split, whose model mix differs from the parent split"** — never "the SWE-smith tool
split".

## A.6 A number in the feasibility audit is a SQL artefact (see STATUS.json)

`GATE_MINUS1_FEASIBILITY_20260816.md` §3.1's "rows containing `message_type` = 9,285" and
its 169-row interpretation are artefacts of `_` being a **single-character wildcard** in SQL
`LIKE`. Parsed, `message_type` / `tool_calls` / `"thought"` mark the **same 9,116 rows**;
there is no 169-row subpopulation of "trajectories that emitted no tool call". Details and
the four exactly-reproduced HF counts are in `STATUS.json:sql_like_underscore_wildcard_20260817`.

## A.7 The §0 mtime falsifier, evaluated

| file | mtime |
|---|---|
| `B09_GATE_MINUS1_PREREG.md` (body, §0-§6) | 2026-08-17 **12:53:29** |
| first `data/agent_traj/*.parquet` | 2026-08-17 **12:55** |

The PREREG body predates every shard. **The pre-registration is not retrospective.** This
appendix was written after and is labelled as such.
