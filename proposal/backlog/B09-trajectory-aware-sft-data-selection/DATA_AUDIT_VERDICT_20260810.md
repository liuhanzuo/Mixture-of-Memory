# B09 — Phase 0 data audit ran, and it cannot run: the candidate pool does not exist

- **Date**: 2026-08-10
- **Executed by**: CPU-only search of both physical disks. Zero GPU.
- **Result**: `next_gate[0]` ("audit trajectory expansion, sibling redundancy, token
  distribution, and benchmark imbalance") is **not executable**. There is no agent
  trajectory corpus on either disk, so there is nothing to audit.
- **Status correction**: `backlog_ready_for_data_audit` →
  `backlog_blocked_data_does_not_exist`.

## 1. What the proposal assumes exists

`PROPOSAL.md` §1 (lines 47-58) sets up the problem over a concrete pool:

```
G = {g_1, ..., g_10000}          ~10K agent trajectories
U = union of U_g,  |U| ~= 100000  ~100K derived SFT rows
S subset of U,     |S| = 5000     the coreset to select
```

Line 168 lists `Candidate U | 100K 候选池 | 可见`, i.e. the 100K pool is treated as
a **visible, already-available** asset. The Phase 0 Go gate (lines 543-548) is
stated entirely in terms of properties *of that pool*:

- "100K 中存在可观 sibling redundancy 或 benchmark imbalance"
- "至少能构造 5K 个 validity 通过、来自广泛 trajectories 的候选"

Both clauses presuppose the pool. Neither can be evaluated against an empty set.

## 2. What is actually on disk

Searched **both** physical disks (the two-disk rule applies here: `zwfy6` is not
mounted on LOCAL, so the second disk was searched over ssh on `.73`, read-only):

- `wzc1`: `/apdcephfs_wzc1/share_304376610/pighzliu_code/`
- `zwfy6`: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/` (via `.73`)

Searches run: `*trajector*`; directories matching `*agent*`, `*swe*`,
`*toolbench*`, `*tool_use*`, `*tulu*`; and specific known agent corpora
`*agentinstruct*`, `*toolllama*`, `*webarena*`, `*react*traj*`, `*agentbench*`,
`*swe_gym*`, `*nemotron*tool*`, `*glaive*`; plus `*traj*.{jsonl,parquet,json}`.

**Findings — nothing agentic.** Every `*trajector*` hit is unrelated:

| hit | what it actually is |
|---|---|
| `Mixture-of-Memory/paperB/figures/fig_trajectory.{py,pdf}`, `fig_1b_trajectory.*` | Paper B training-curve figures |
| `Mixture-of-Memory/paperB/sections/tab_trajectory_audit.tex` | Paper B checkpoint-trajectory table |
| `proposal/active/A03-*/evidence/arm{3,4,6}_cpt_trajectory_*.json` | A03 continued-pretraining **checkpoint** trajectories |
| `Mixture-of-Memory/logs/a03_arm{3,4,6}_trajectory_progress.log` (zwfy6) | A03 watcher logs |
| `Mixture-of-Memory/src/agents`, `mom_agent.egg-info` | this repo's own agent-orchestration code |
| `locomo/generative_agents` | LoCoMo benchmark data, a conversational memory benchmark, not agent rollouts |
| `.openclaw/agents/*` (zwfy6) | the CLI harness's own agent configs |

"Trajectory" in this repo means **checkpoint-over-training-steps**, never
**agent-rollout**. That is the collision that made this proposal look
data-ready when it never was.

## 3. The one asset that superficially resembles the pool, and why it is not

`Mixture-of-Memory/data/olmo2_sft/tulu3_general_*` exists on **both** disks and is
the closest thing to a "100K SFT row" pool. It is not the B09 pool.

Verified numbers (manifest read from zwfy6; array shapes read from wzc1):

```
tulu3_general_manifest.json   (zwfy6 only)
  dataset:                    allenai/tulu-3-sft-mixture
  n_conversations_used:       234,483
  n_conversations_seen:       281,072
  n_sequences (packed 2048):  122,070
  n_tokens_packed:            249,999,360
  n_supervised_tokens_packed: 161,118,687
  seed 42, shuffle_buffer 50000, commit 2d98c5a
  source_histogram: 9 sources, top = evol_codealpaca_heval_decontaminated 46,774

wzc1: data/olmo2_sft/tulu3_general_clean_input_ids.npy  shape (107740, 2048) uint32
      data/olmo2_sft/tulu3_general_clean_labels.npy     shape (107740, 2048) int32
zwfy6: tulu3_general_{input_ids,labels}.npy (122,070 seqs) + tulu3_general_text.jsonl (757 MB)
       + p24smoke_* (1,012 seqs, 1,999 conversations) smoke variant
```

The 107,740-sequence wzc1 array is numerically near the "~100K rows" the proposal
wants, which is exactly the trap. It fails on every structural requirement:

1. **No parent trajectory.** The pool's whole premise is that rows have a parent
   `g(i)` and that siblings from one parent are redundant. Tulu-3 rows are
   independent single- or multi-turn instruction conversations with no parent
   grouping. `tulu3_general_text.jsonl` carries `{source, prompt, response}` —
   `source` is a **dataset name** (e.g. `ai2-adapt-dev/tulu_v3.9_wildchat_100k`),
   not a trajectory id. There are 9 sources for 234K conversations, so grouping by
   `source` gives 9 groups of ~26K, not ~10K groups of ~10.
2. **No steps, no decision points.** The proposal needs `t(i)` (trajectory
   step/decision type) to do critical-step selection. Tulu-3 has no step index and
   no notion of a decision that could have gone another way.
3. **No success/reward, no tool family.** §1 requires both. Tulu-3 has neither.
4. **No branch structure**, so "branch-verified decision credit" — one of the two
   candidate non-compositional cores — has no substrate.
5. **Already packed and tokenized, and provenance-restricted.** These arrays are
   pre-packed to 2048 with an OLMo-2 tokenizer for Paper B's P2.4 general-SFT
   repairability pipeline, under a 19-pattern `deny_sources` filter (flan, mmlu,
   arc, triviaqa, squad, ... ) chosen to protect *that* experiment's eval
   integrity. Row-level selection needs unpacked rows; and the deny filter is a
   Paper-B-specific decontamination choice, not a B09 design decision.

So the correct statement is: a general instruction-tuning pool exists; the
**trajectory-structured** pool B09 is defined over does not. Using Tulu-3 would
not be a substitution, it would delete the independent variable.

## 4. Status correction and what must happen first

`STATUS.json` is corrected to `backlog_blocked_data_does_not_exist`, and
`next_gate` is reordered so the data-construction step precedes the audit. The
audit itself is **kept verbatim** as the second gate — it is a good gate, it is
simply downstream of an acquisition step nobody had scheduled.

Before B09 can be worked at all, one of these must happen:

- **(a) Acquire an existing agent-trajectory corpus.** Requires external download
  (proxy `hy-proxy.woa.com:3128`) and a licence check. Candidates named in
  `SOURCES.md`'s own collision list imply such corpora exist publicly
  (tool-use/SWE-agent trajectory sets). **Cheapest path, and it is the only one
  that does not need GPU.**
- **(b) Generate rollouts ourselves.** ~10K trajectories of agent rollout against
  a task suite. This is a **large GPU spend plus a task harness we do not have**,
  and it would make B09 far more expensive than any other backlog item.

Either way the cost is **not** what the proposal's Phase 0 implies ("无 GPU 训练").
Phase 0 was scoped as a cheap CPU audit; the real first step is a data-acquisition
project.

## 5. Honest note on scope of this audit

This is a **filename and directory search**, not a content scan of every file on
both disks. It is possible in principle that an agent-trajectory corpus exists
under a name none of the ~15 patterns matched. Two things bound that risk: the
search covered both disks to depth 4-6 including all of `data/`, and this repo's
`data/` directory is small and fully enumerable (18 entries on wzc1, 14 on zwfy6,
all listed and all accounted for as C4 / dolmino / fineweb / redpajama / wikitext /
hf_datasets MC benchmarks / tulu3 SFT / instruct-prep scripts). None is an agent
corpus. If someone later finds such a pool, this file should be corrected rather
than silently ignored — but the burden has now shifted to producing the path.

## 6. Provenance

- `proposal/backlog/B09-trajectory-aware-sft-data-selection/PROPOSAL.md` §1 lines
  47-58, the visibility table at line 168, the Phase 0 gate at lines 531-548
- `proposal/backlog/B09-trajectory-aware-sft-data-selection/SOURCES.md` — lists
  only literature plus `status/RESEARCHER_REPORTS.jsonl`; **no internal data path
  is claimed anywhere in it**, which is itself the tell that the pool was never
  located
- `/apdcephfs_zwfy6/.../Mixture-of-Memory/data/olmo2_sft/tulu3_general_manifest.json`
  and `p24smoke_manifest.json` (read via `.73`)
- `/apdcephfs_wzc1/.../Mixture-of-Memory/data/olmo2_sft/tulu3_general_clean_{input_ids,labels}.npy`
  (shapes read with numpy mmap)
- `scripts/prepare_olmo2_sft_data.py` — confirms the pool's construction: streams
  `allenai/tulu-3-sft-mixture`, filters `deny_sources`, emits
  `{source, prompt, response}` rows; no trajectory/step/reward field anywhere
