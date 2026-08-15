# B09 GATE -1 — feasibility of acquiring a trajectory-STRUCTURED candidate pool

- **Date**: 2026-08-16
- **GPU used**: **none** (0 GPU-seconds, 0 GPU-hours, no node touched for compute; `.73` was
  used read-only over ssh for a filesystem listing on the second disk)
- **Method**: two-disk filename/directory search + HuggingFace control-plane survey
  (`/api/datasets`, `datasets-server.huggingface.co` `info` / `first-rows` / `rows` /
  `filter` / `statistics` / `size`) through `hy-proxy.woa.com:3128`
- **Verdict**: **(a)** — a corpus exists that carries all four required fields natively.
  GATE -1 becomes a **1.0 GiB download + derive task**, not a rollout-generation project.
  **Outcome (c) is refuted.** One material caveat is recorded in §5 and it is not
  a field-availability caveat, it is a benchmark-breadth caveat.

---

## 0. What the gate demands (verbatim, from `STATUS.json:next_gate[0]`)

> GATE -1 (NEW 2026-08-10, BLOCKING, was never scheduled): obtain a trajectory-STRUCTURED
> candidate pool — rows carrying **parent trajectory id**, **step/decision type**,
> **success/reward**, and **tool family**. Either download an existing agent-trajectory corpus
> (zero GPU, needs proxy + licence check) or generate rollouts (large GPU + a task harness we
> lack). Until this lands, every gate below is undefined.

Four fields. That is the whole test. §3 checks each candidate field-by-field against the
*actual schema*, not the dataset name or card prose.

---

## 1. Two-disk search — nothing agentic is on disk (confirms and EXTENDS the 08-10 audit)

Both physical disks searched. `zwfy6` is **not mounted on LOCAL**
(`ls /apdcephfs_zwfy6` → `No such file or directory`; `mount` shows only
`/apdcephfs_wzc1/share_304376610` and `/apdcephfs_wzc1_304376610/share_304376610`), so the
second disk was searched read-only over ssh on `.73`. No `find /` was ever run.

| disk | root searched | depth | patterns | hits |
|---|---|---|---|---|
| wzc1 | `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory` | 6 | 22 | 0 relevant |
| wzc1 | `/apdcephfs_wzc1/share_304376610/pighzliu_code` (outer) | 5 | 22 | 1 irrelevant |
| zwfy6 | `/apdcephfs_zwfy6/share_304376610/pighzliu_code` (via `.73`) | 5 | 22 | **0** |

Patterns (**25** — corrected 2026-08-16 by MAIN, who counted the enumerated list; three places in
an earlier draft said 22 while the list below has 25. Superset of the 08-10 audit's ~15): `*agent*traj*`, `*traj*agent*`,
`*toolbench*`, `*toolllama*`, `*xlam*`, `*swe_gym*`, `*swe-gym*`, `*agentinstruct*`,
`*webarena*`, `*glaive*`, `*nemotron*`, `*rollout*`, `*tool_use*`, `*tooluse*`,
`*func*call*`, `*agentbench*`, `*api_bank*`, `*apibank*`, `*tau_bench*`, `*taubench*`,
`*mind2web*`, `*miniwob*`, `*alfworld*`, `*webshop*`, `*react*`.
**Eight of these (`mind2web`, `miniwob`, `alfworld`, `webshop`, `tau_bench`, `api_bank`,
`rollout`, `func*call`) were NOT in the 08-10 audit's pattern list** — so this is an independent
widening of that search, not a repeat of it. It still finds nothing.

> **CORRECTED 2026-08-16 by MAIN.** The draft said *nine*, including `tool_use`. But
> `grep -c tool_use DATA_AUDIT_VERDICT_20260810.md` returns 1 — `tool_use` **was** already in the
> 08-10 list, so the count of genuinely-new patterns is **eight**, not nine. The
> "independently widens that search" conclusion is unaffected; only the tally was wrong.

The only `*rollout*` hits on wzc1 are:

| hit | what it actually is |
|---|---|
| `.../pighzliu_code/dllm_draft/vendor/verl/tests/rollout` | vendored `verl` library's own unit-test directory |
| `.../Mixture-of-Memory/.codex/sessions/2026/*/rollout-*.jsonl` | tcodex CLI session transcripts (the CLI names its session files `rollout-*`) |

Neither is agent-trajectory training data. Full `data/` enumerated on both disks
(20 entries wzc1 outer, 18 wzc1 inner, 15 zwfy6 outer, 18 zwfy6 inner): all are C4 /
dolmino / fineweb / redpajama / slimpajama / pg19 / wikitext / babilong / longbench /
infinitebench / squad / MC-benchmark HF caches / `olmo2_sft` (Tulu-3) /
`benchmark_sft_llama` + `qa_format_sft_llama` + `qa_dolmino_mix_llama` (packed `.bin`
instruction data, no trajectory fields). **None is an agent corpus.**

**Conclusion: the 2026-08-10 `DATA_AUDIT_VERDICT` stands. Nothing to reuse, nothing to correct.**

---

## 2. Proxy positive controls (run BEFORE any not-found claim)

| control | command | result |
|---|---|---|
| HF control plane | `curl https://huggingface.co/api/datasets/allenai/tulu-3-sft-mixture` | **PASS** — returned `_id`, `sha b14afda6…`, `lastModified 2024-12-02` |
| HF datasets-server | `curl .../info?dataset=allenai%2Ftulu-3-sft-mixture` | **PASS** — returned the `messages`/`source` feature tree |
| HF **data plane** (parquet) | `curl -r 0-2097151 .../nebius/SWE-agent-trajectories/resolve/main/data/train-00000-of-00012.parquet` | **PASS** — `http=206`, 2,097,152 B, magic `PAR1`, 1.02 MB/s |

All three pass, so absence results below are meaningful. Proxy exports were on **separate
lines**. One 404 was observed first (`train-00000-of-00003.parquet`) — that was my wrong
shard name, corrected via the `tree` API to `-of-00012`; it was **not** a proxy failure.

---

## 3. Candidate table — field-by-field against the ACTUAL schema

Every row below is **RETRIEVED** from the HF API unless a cell says INFERRED.
`P` = parent trajectory id, `S` = step/decision type, `R` = success/reward, `T` = tool family.

| repo id | rows | download | licence | P | S | R | T | 4/4? |
|---|---|---|---|---|---|---|---|---|
| **`SWE-bench/SWE-smith-trajectories`** | 76,002 (tool 24,100 / xml 26,076 / ticks 25,826) | **3.93 GiB** all; **1.0 GiB** `tool` only | **MIT** | ✅ `instance_id`+`traj_id` | ⚠️ `message_type` on **9,285/24,100** of `tool` | ✅ `resolved` bool, **100%** | ⚠️ `tool_calls[].function.name` on **9,116/24,100** | **YES on the 9,116-row structured subset** |
| **`nebius/SWE-agent-trajectories`** | 80,036 | **1.04 GiB** | **CC-BY-4.0** (+ per-repo licences; Llama-3.1 licence notice for outputs) | ✅ `instance_id` | ❌ no field (`role`+`mask`+position only) | ✅✅✅ `target` bool + `exit_status` (9 cats) + `eval_logs` | ❌ no field (100% regex-recoverable, §4) | **NO — 2 of 4 native** |
| `Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k` | 65,994 | 1.47 GiB (`usedStorage` 2,492,345,470 B) | **MIT** | ✅ `instance_id` | ❌ | ❌ **no success field at all** | ❌ | NO |
| `israel-adewuyi/kwaiklear-sample-level-agent-trajectories-2.2M` | 1,826,649 + 270,544 | 14.7 GiB | ⛔ **NO LICENCE DECLARED** | ✅ `instance_id` | ✅ `assistant_turn_index`, `target_message_index` | ❌ | ❌ | NO + **unlicenced, flagged** |
| `AgentGym/AgentTraj-L` | 14,485 | 103 MiB | ⛔ **NO LICENCE DECLARED** | ✅ `item_id` | ❌ | ❌ | ⚠️ 12 env files = tool family at *file* level only | NO + **unlicenced, flagged** |
| `Solaris99/AgentBank` | 16+ configs (alfworld 3,321; gsm8k 7,385; apps 4,408; hotpotqa 4,273; alfred 623; …) | 30 MiB | Apache-2.0 | ✅ `id` | ❌ | ❌ | ⚠️ config name only | NO |
| `agent-eto/eto-sft-trajectory` | 3 env splits (webshop/scienceworld/alfworld) | 41 MiB | Apache-2.0 | ✅ `id` | ❌ | ❌ | ⚠️ split name only | NO |
| `Salesforce/APIGen-MT-5k` | 5,000 | 131 MiB | ⛔ **CC-BY-NC-4.0 — non-commercial only, flagged** | ❌ | ❌ | ❌ | ✅ `tools` field | NO |
| `Salesforce/xlam-function-calling-60k` | 60k | 451 MiB | CC-BY-4.0, **`gated: auto`** | ❌ flat pairs | ❌ | ❌ | ✅ | NO |
| `osunlp/Mind2Web` | 1,009 | 14.7 GiB | CC-BY-4.0 | ✅ `annotation_id` | ⚠️ `actions`/`action_reprs` list | ❌ | ⚠️ web-only | NO |
| `R2E-Gym/R2E-Gym-V1` | 8,101 | 2.7 GiB | Apache-2.0 | — | — | — | — | **NO — it is a task ENV set, not trajectories** |
| `SWE-Gym/SWE-Gym` | 2,438 (`1K<n<10K`) | 42 MiB | MIT | — | — | — | — | **NO — task set, not trajectories** |
| `SWE-Gym/SWE-Gym-Trajectories` | — | — | — | — | — | — | — | **UNVERIFIED**: `/api/datasets` → `Invalid username or password`; datasets-server → `does not exist, or is not accessible without authentication`. Gated or nonexistent. |
| `ToolBench/ToolBench` | — | — | — | — | — | — | — | **UNVERIFIED**: `/api/datasets` → `Invalid username or password`. Auth required. |
| `THUDM/AgentInstruct` | — | — | — | — | — | — | — | **UNVERIFIED**: datasets-server → `The dataset has been renamed. Please use the current dataset name.` New name not resolved. |
| `nvidia/Llama-Nemotron-Post-Training-Dataset` | 1M-10M | 353 GiB | CC-BY-4.0 | ❌ | ❌ | ❌ | ❌ | NO — post-training SFT/RL text, configs `{code,math,science,chat,safety,instruction_following}` |

### 3.1 The winner, in exact numbers (all RETRIEVED)

`SWE-bench/SWE-smith-trajectories`, **`tool` split**, `n = 24,100`:

| quantity | value | how obtained |
|---|---|---|
| declared licence | `mit` | `cardData.license` + README front-matter |
| declared splits | `tool` 24,100 / `xml` 26,076 / `ticks` 25,826 | `/info` and `/splits` |
| **undeclared 4th shard family** | `data/train-*` , 8 shards, 0.98 GiB | `/api/.../tree/main/data`; **`train` is NOT in `/splits`** — dead weight, do not download |
| parquet bytes, `tool` | 1,077,271,863 B = **1.0 GiB** | `/size` |
| parquet bytes, all 4 families | 4,223,841,784 B = **3.93 GiB** | sum of `/tree` sizes |
| `resolved` = true / false | **9,427 / 14,673** (= 24,100 ✓) | `/statistics` frequencies |
| rows containing `message_type` | **9,285** (38.53%) | `/filter … "messages" LIKE '%message_type%'` |
| rows containing `tool_calls` | **9,116** (37.83%) | `/filter … LIKE '%tool_calls%'` |
| rows containing `"thought"` | **9,116** (identical — cross-check) | `/filter … LIKE '%"thought"%'` |
| structured ∧ resolved=true | **3,452** | `/filter … LIKE '%tool_calls%' AND "resolved"=true` |
| structured ∧ resolved=false | **5,664** (3,452+5,664 = 9,116 ✓) | same, `=false` |
| generator models | claude-3-7-sonnet 17,715 / claude-3-5-sonnet 5,751 / gpt-4o 634 | `/statistics` |
| structured rows in `xml` / `ticks` | **266** / **266** (≈1%) | `/filter` per split |

The **9,285 vs 9,116 gap (169 rows)** is real, not rounding: 169 rows carry `message_type`
but no `tool_calls` — plausibly trajectories that emitted no tool call. I report both rather
than conflating them. The operative subset for B09 is the **9,116** that carry *both*.

Observed message-level structure on a structured row (RETRIEVED, `offset=0`, `tool` split):
`n_messages=49`, `roles={assistant:24, tool:23, system:1, user:1}`,
`message_type={system_prompt:1, action:24, observation:24}`, per-message keys
`['action','agent','cache_control','content','message_type','role','thought','tool_call_ids','tool_calls']`.
So for a structured row, `1 + 2N = n_messages` and **assistant action turns = (n_messages−1)/2**.

**Is 9,116 trajectories enough for B09?** B09 §1 wants `|G| ≈ 10,000` and `|U| ≈ 100,000`.
Mean `n_messages` over the 9 structured rows in the stratified probe = **65.9** (n=9,
values 49/31/65/65/153/59/57/67/47) → **≈32.4 action turns per trajectory** → **≈2.95×10⁵
derivable decision rows** from 9,116 parents. That is `|G|` within 9% of target and `|U|`
~3× above target. **INFERRED** from n=9; the exact count needs one CPU pass over the
downloaded parquet, which is exactly what Phase 0 is for.

**Sibling/branch structure — the thing Tulu-3 could not provide.** `traj_id` was unique in
44/44 sampled rows, but `instance_id` repeats. Measured rollouts-per-instance in the `tool`
split (10 instances, `/filter` exact counts):
`MONAI.pr_7187`→13, `MONAI.pr_6735`→8, `MONAI.lm_rewrite__roex7dhi`→5,
`conan.pr_13390`→5, `voluptuous.lm_rewrite__9e2nb1af`→4, `flake8.lm_rewrite__tmxef7uw`→3,
`schedule.6`→3, `django-money.func_pm_ctrl_shuffle`→2, `conan.pr_12880`→1,
`starlette.lm_rewrite__wq8ip1tf`→1. **Range 1–13.** For
`apispec.func_pm_remove_assign__kdkrbg6a`, both rollouts (`…8qa84d2e`, `…g1uzn18p`) are
`resolved=True`. Parent task pool = `SWE-bench/SWE-smith` train, **59,136** instances.

This is precisely PROPOSAL §4.1's "同一 task instance 的多次 rollout" partition requirement,
and it is exactly what §5's branch-verified decision credit needs: sibling rollouts of one
instance with **differing** `resolved` outcomes.

### 3.2 The complementary corpus, in exact numbers (all RETRIEVED)

`nebius/SWE-agent-trajectories`, `n = 80,036`, **CC-BY-4.0**, **1.04 GiB** (12 shards):

| quantity | value |
|---|---|
| `target` true / false | **13,389 / 66,647** (= 80,036 ✓) |
| `exit_status`, 9 categories | `submitted` 51,087 · `submitted (exit_context)` 21,026 · `exit_context` 3,568 · `early_exit` 3,176 · `submitted_no_patch` 1,066 · `submitted (exit_format)` 80 · `exit_format` 20 · `submitted (exit_cost)` 10 · `exit_cost` 3 (sums to 80,036 ✓) |
| `eval_logs` NaN / `generated_patch` NaN | 9,397 / 9,294 |
| generator models | `swe-agent-llama-70b` 74,792 · `-8b` 4,053 · `-405b` 1,191 |
| rollouts per instance (measured) | `AnalogJ__lexicon-336`→**11**, `iterative__dvc-8882`→**17**, `postlund__pyatv-978`→**13** |
| parent task pool | `nebius/SWE-bench-extra` train = **6,376** + SWE-bench dev |
| card-reported steps (resolved vs not) | 31.3 vs 58.4 |

Its **reward signal is the richest of any candidate** — three independent channels
(`target` bool, a 9-way `exit_status` taxonomy, and raw `eval_logs` test output). Its
weakness is that neither step type nor tool family exists as a field.

---

## 4. Reconstruction cost for the missing/partial fields (all 0 GPU)

| gap | corpus | measured reconstructability | cost |
|---|---|---|---|
| step/decision type on the 62% of `tool` rows lacking `message_type` | SWE-smith | role alternation is strictly `system → user → (assistant, user)*`; step **index** is free from position; `action` vs `observation` follows from `role` | trivial CPU |
| tool family on the 62% lacking `tool_calls` | SWE-smith | `<function=NAME>` markup regex-recovers names in **21/30** unstructured sampled rows; recovered histogram `str_replace_editor 375, bash 247, submit 39` **plus 21 false hits on the literal `example_function_name` from the system prompt** — a de-boilerplate step is mandatory | ~1 CPU-hour + a validation pass |
| tool family (no field) | nebius | **261/261 = 100%** of `ai` turns carry a parseable fenced command. Recovered: `edit` 77, `open` 49, `python` 23, `dvc` 22, `ls` 12, `rm` 12, `create` 10, `search_dir` 8, `find_file` 7, `goto` 7, `submit` 7, `cat` 7, `search_file` 6, `scroll_down` 3 …; **67%** fall in the fixed 9-verb SWE-agent ACI vocabulary, the rest are bash | ~1 CPU-hour |
| step/decision type (no field) | nebius | `mask` bool marks loss-bearing turns; type is derivable from the recovered verb | trivial once the verb is parsed |

**B09 does not actually need the 8-way taxonomy pre-labelled.** PROPOSAL §5 Stage 1 says the
*method itself* assigns `PLAN / CRITICAL_OBSERVATION / TOOL_SELECTION / ARGUMENT_GROUNDING /
PIVOTAL_ACTION / RECOVERY / STOP_OR_FINAL / FORMAT_ONLY`. The gate only requires the corpus
carry *a* step/decision type so Stage 1 has a substrate. `message_type ∈
{system_prompt, action, observation}` plus a step index is that substrate.

---

## 5. Verdict

### **(a)** — with one honest, named limitation that is NOT about field availability

**`SWE-bench/SWE-smith-trajectories` (MIT) satisfies GATE -1 as written.** On its 9,116-row
structured subset of the `tool` split, all four required fields are present *natively, in the
schema, verified by fetching rows and running exact `/filter` counts*:

1. **parent trajectory id** — `traj_id` (unique per rollout) nested under `instance_id`
   (1–13 rollouts per instance, measured), over a 59,136-instance parent pool;
2. **step/decision type** — `message_type ∈ {system_prompt, action, observation}` on
   9,285 rows, plus a free positional step index;
3. **success/reward** — `resolved` bool on **100%** of rows; within the structured subset
   3,452 resolved / 5,664 unresolved, so both classes are well populated;
4. **tool family** — `tool_calls[].function.name` on 9,116 rows
   (`bash`, `str_replace_editor`, `submit`).

Scale: 9,116 parents ≈ B09's `|G| ≈ 10,000`; ≈2.95×10⁵ derivable decision rows ≥ B09's
`|U| ≈ 100,000`. Licence MIT — permissive, no non-commercial restriction, not gated.
Download 1.0 GiB.

**Therefore outcome (c) is refuted. GATE -1 does not require rollout generation and does not
require GPU.** The blocker recorded on 2026-08-10 is dischargeable by a 1.0 GiB download plus
a CPU derive pass.

### The limitation that survives, and it is a real one

**Both viable corpora are single-benchmark (SWE / Python repo repair) with a 3-verb tool
vocabulary.** That does not violate GATE -1 — the gate asks for four *fields*, and they are
there. But it does undercut two things downstream:

- PROPOSAL §5 **Stage 5** (`F_meta`) asks for coverage over *benchmark*, *skill/task family*,
  *tool family/topology*. With one benchmark and 3 tools, the benchmark term is degenerate
  and the tool term is near-degenerate. Repo/task-family and decision-type coverage remain
  meaningful (59,136 parent instances across many repos).
- §4.3 **Strict benchmark-OOD** and **H3** explicitly contemplate "若只有 20 个 benchmark,
  使用 5-fold benchmark holdout". With `B = 1`, benchmark holdout is undefined. It would have
  to be re-scoped to **repo-family holdout**, which is a weaker but still defensible claim —
  and that re-scope must be pre-registered, not discovered after the fact.

The multi-environment corpora that *would* give benchmark breadth (`AgentGym/AgentTraj-L`,
12 envs; `Solaris99/AgentBank`, 16+ envs; `agent-eto/eto-sft-trajectory`, 3 envs) **all lack a
per-row success/reward field**, and two of the three (`AgentTraj-L`, and
`israel-adewuyi/…-2.2M`) **declare no licence at all**. So there is no drop-in fix: breadth
and reward are, in the currently downloadable public set, mutually exclusive.

**This is the finding to carry forward.** B09 is unblocked on data, but its H3 (ID-vs-OOD
benchmark contrast) and its Stage 5 metadata-coverage term are *not* supported by any single
licenced corpus, and that is a scientific scoping decision, not a download problem.

---

## 6. Exact next action

**GATE -1 → a download + derive task. Recommended landing: `wzc1`.**

```
mkdir -p /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/data/agent_traj
# primary, 1.0 GiB, MIT — the tool split ONLY (8 shards); skip xml/ticks (≈1% structured)
#   and skip data/train-* entirely (undeclared in /splits, 0.98 GiB of dead weight)
#   https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories
#     data/tool-0000{0..7}-of-00008.parquet
# optional complement, 1.04 GiB, CC-BY-4.0 — richest reward signal + 11-17 rollouts/instance
#   https://huggingface.co/datasets/nebius/SWE-agent-trajectories
#     data/train-000{00..11}-of-00012.parquet
```

**Why wzc1, not zwfy6:** (i) B09's own directory and all its records live on wzc1, and this
is CPU-only work — Phase 0 is an audit; (ii) wzc1 has **11 T free** of 120 T (92% used) and
1.0 GiB is negligible; (iii) `zwfy6` has more headroom (**33 T free** of 689 T, 96% used) and
is the disk the standing node policy assigns to `.73/.82/.104`, so **if** a later SFT arm needs
GPU, copy then — at 1.0 GiB a `scp -O` is minutes, not the 42 h that large checkpoints cost.
**`/tmp` is forbidden** — proven wiped on reboot (`memory/persist-artifacts-on-wzc1-or-diskb.md`).

Then, in order:

1. **Assert, do not assume.** Over the downloaded `tool` shards, recompute:
   `n_rows == 24100`; `resolved` true/false `== 9427/14673`; rows with both
   `message_type` and `tool_calls` `== 9116`. If any differs, stop — the HF index and the
   parquet disagree and every number in §3.1 is suspect.
2. Emit the flat candidate pool with B09's §1 required columns:
   `g(i)=traj_id`, `instance_id`, `t(i)=message_type`, step index, `resolved`,
   `tool_name`, `c_i` = assistant target-token count.
3. Only then run `next_gate[1]` (the verbatim-kept audit: rows-per-trajectory Gini,
   near-duplicate rate, tool/decision-type distribution, `cap=1/2/∞` availability).
4. **Pre-register the benchmark-breadth re-scope of §4.3 / H3 to repo-family holdout
   BEFORE looking at any outcome.** Deciding this after seeing results is exactly the
   leakage §4.3 exists to prevent.

**Do not** schedule rollout generation. **Do not** substitute `data/olmo2_sft/tulu3_general_*`
— the 2026-08-10 verdict on that is unchanged and correct.

---

## 7. Honest limits of THIS audit

1. **Schema coverage is sampled, not exhaustive, for the per-message fields.** The row counts
   in §3.1 are exact (`/filter` `num_rows_total` over the full split). But
   "≈32.4 action turns per trajectory" and "21/30 regex-recoverable" rest on **n=9** and
   **n=30** samples respectively, and are marked INFERRED. The stratified `message_type`
   coverage probe (9/21 = 42.9%) has a Wald 95% CI of 22–64% — it is *consistent with* but
   much looser than the exact 38.53%; I report the exact figure and note the probe only as
   corroboration.
2. **`xml` / `ticks` structured counts both returned exactly 266.** Two different splits
   returning an identical count is the kind of coincidence that is usually an artefact. The
   `tool` split cross-check passed cleanly (`tool_calls` 9,116 == `"thought"` 9,116, and
   3,452+5,664 == 9,116), so I trust the `tool` numbers; **treat the two 266s as
   provisional.** It does not affect the verdict, which rests on `tool` alone.
3. **Three candidates are UNVERIFIED, with the exact authority and error recorded**:
   `SWE-Gym/SWE-Gym-Trajectories` (`Invalid username or password` / `not accessible without
   authentication`), `ToolBench/ToolBench` (`Invalid username or password`),
   `THUDM/AgentInstruct` (`The dataset has been renamed`). A token exists at
   `configs/password_hf_token.txt` but I did not use it — authenticating would change what
   "publicly downloadable" means, which is the gate's own criterion.
4. **Licences are read from HF metadata + README front-matter, not from a legal review.**
   `nebius` in particular adds two conditions beyond CC-BY-4.0 in its card: respect each
   source repository's licence (per-instance licences are in `SWE-bench-extra`), and comply
   with the Llama-3.1 licence if using model outputs. Anyone shipping a model trained on it
   must read those, not this file.
5. **The disk search is filenames and directories, not a content scan.** Same limitation the
   08-10 audit disclosed. Mitigated by both disks being searched with 25 patterns and by
   `data/` being small and fully enumerated on both. The burden remains on producing a path.

## 8. Provenance

- `proposal/backlog/B09-trajectory-aware-sft-data-selection/{PROPOSAL.md, STATUS.json,
  NOVELTY.md, RELATED_WORK.md, SOURCES.md, DATA_AUDIT_VERDICT_20260810.md}` — read in full.
  `RELATED_WORK.md` (447 lines) names **no** corpus, only papers; that is why this survey had
  to start from HF search rather than from the proposal's own reading list.
- `https://huggingface.co/api/datasets?search=…` — 4 searches (`agent+trajectories`,
  `SWE-agent+trajectories`, `agent+trajectory`, `tool+use+trajectories`) + `AgentBank`,
  `AgentGym`, sorted by downloads.
- `https://huggingface.co/api/datasets/<id>` — licence / `gated` / `usedStorage` / size tags.
- `https://datasets-server.huggingface.co/{info,splits,first-rows,rows,filter,statistics,size}`
  — feature trees, exact row counts, column statistics, per-split parquet bytes.
- `https://huggingface.co/api/datasets/<id>/tree/main/data` — shard names and byte sizes.
- `https://huggingface.co/datasets/<id>/raw/main/README.md` — card front-matter and licence text.
- Local: `mount`, `df -h`, `find -maxdepth {5,6}` on wzc1;
  `sshpass -f configs/password_h20_853573.txt ssh root@28.85.35.73 'ls / find'` for zwfy6.
