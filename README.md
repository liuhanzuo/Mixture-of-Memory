# Mixture-of-Memory

> ### 🚀 新协作者？先读 **[`ONBOARDING.md`](ONBOARDING.md)** — 30 分钟上手（环境 / 当前方向 / 怎么训怎么评 / 三条红线）。
> 权威**实时**状态在 **[`status/HEARTBEAT_LATEST.md`](status/HEARTBEAT_LATEST.md)**；集群/SSH 配置在 **[`CODEBUDDY.md`](CODEBUDDY.md)**。
>
> ⚠️ 本 README 以下"Current architecture"一节描述的是**较早的 slot 架构**；当前主攻已转向 **hidden 级路线**（hidden FIFO / HNST 树 / 多尺度 beacon）。以 `ONBOARDING.md` 和 `HEARTBEAT_LATEST.md` 为准。

Research on compressing long context into a **fixed-size memory buffer** so that an
8B LLM (Llama-3-8B) can handle very long sequences under a *bounded* KV budget —
instead of letting the KV cache grow linearly with sequence length.

The current line of work trains a lightweight **`mem_space` adapter** on top of a
frozen-ish Llama-3-8B and evaluates it on **BABILong** (tasks qa1 / qa2 / qa5 across
context lengths 0k–32k, n=100 samples per cell).

---

## Current architecture: the `mem_space` adapter

Code: `src/memory/mem_space/` (`config.py`, `layer.py`, `selector.py`,
`memory_bank.py`, plus `l3_summary.py`, `dolmino_dataset.py`, `babilong_dataset.py`).

A bank of **N memory slots** is shared across all transformer layers (a single
`MemoryBank` per sample). The input stream is split into chunks
(`chunk_size = 512`). For each chunk:

1. **Route.** A lightweight *selector* scores all N slots against the chunk and
   picks the **top-k = 16** slots to participate this step.
2. **Write** = a **gated delta-rule writeback** into *only the top-k selected
   slots*. The default gate is a **dual gate** (input gate + forget gate); under
   `--use_delta_rule_writeback` the write is a residual delta-rule update keyed by
   the input gate. Writes are top-k, so each step touches at most k slots.
3. **Read** = a dedicated `MemoryCrossAttentionRead` (`--use_memory_xattn`) with
   its **own softmax over ALL N slots** plus a null/sink slot. Read is *all-N*,
   write is *top-k* — the two paths are deliberately decoupled.
4. **L3 summary.** A shared L3 cross-attention path
   (`--use_l3_summary`, `l3_n_summary = 64`) produces K summary tokens that carry
   long-range information; this L3 path is the main long-range workhorse.

Architecture iterations are documented in `versions/` (v8 → v20 are current; v2–v7
are archived under `legacy/versions_pre_v8/`). v20 (`read_based_slot_lifecycle`) is
the active research frontier.

---

## Key empirical findings

1. **chunk512 is the sweet spot.** chunk256 and chunk1024 both underperform.
2. **Early-stop wins.** step500–1000 ≫ step5000; continuing to train monotonically
   degrades BABILong accuracy (over-training hurts). Launchers therefore train for
   1000 steps with `save_interval 500`.
3. **Capacity sweep N128 → N896.** Adding slots helps mid/long context in the
   128–384 range, but single-run variance is high — multi-seed means are required
   before drawing conclusions (this is the active `expR1c*` sweep).
4. **Read ≠ Write (the central puzzle).** ~93–95% of read attention lands on slots
   that were *never written* — they still hold the strided-token chunk-0 snapshot
   the bank was initialized with. Per-slot the read is roughly uniform over written
   and unwritten slots, so the trained delta-rule write contributes only ~5% of the
   read-out. This motivates the current read-based slot-lifecycle work (v20).

---

## Repo layout

```
src/memory/mem_space/   active adapter — config.py, layer.py, selector.py,
                        memory_bank.py, l3_summary.py, dataset loaders
scripts/                launch_*.sh        training launchers (live: expR1c* sweep)
                        eval_*.sh          evaluation launchers
                        train_mem_space_dolmino_cpt.py   the trainer
                        run_babilong_mem_space.py        BABILong inference
                        score_nested_babilong.py         aggregate scorer
status/                 live state:
                          RUN_REGISTRY.md      per-run config + BABILong ledger
                          SESSION_HANDOFF.md   READ THIS FIRST on a new session
                          PENDING_TASKS.md     task board
                          BENCHMARK_RESULTS.md results (ours + external papers)
                          *.jsonl              append-only logs
versions/               architecture version docs (v8–v20 current)
legacy/                 archived abandoned/superseded directions (see its README)
CODEBUDDY.md            full operating manual + authoritative cluster/GPU setup
HEARTBEAT.md            monitoring / autonomous-ops playbook
```

The trainer, the two BABILong scripts, and `scripts/launch_expR1c*` / the
`expR1c` eval schedulers are the **live** experiment chain — do not move them.

---

## How to train

The active experiment family is the `mem_space` capacity sweep (`expR1c*`). A
launcher trains the adapter for 1000 steps on the per-doc Dolmino corpus:

```bash
bash scripts/launch_expR1cN192_cum_slots192_local.sh
```

Each launcher wraps `torch.distributed.run --nproc_per_node=8
scripts/train_mem_space_dolmino_cpt.py` with the run's hyperparameters. Salient
defaults: `--chunk_size 512 --num_slots <N> --top_k 16`, `--total_steps 1000
--save_interval 500`, and **`--eval_interval 0`** — inline eval inside the DDP loop
causes NCCL hangs (variable-length generation desyncs ranks), so evaluation is run
offline against saved checkpoints.

Environment notes:
- `WANDB_MODE=offline` on remote nodes (no wandb.ai connectivity).
- Use the project `.venv/bin/python` (torch 2.10 + transformers; H20/L20A-compatible).
- See `CODEBUDDY.md` for the proxy / HF-cache exports needed on offline nodes.

---

## How to evaluate

```bash
# 1. Run BABILong inference for a checkpoint -> per-(task,length) CSVs
python scripts/run_babilong_mem_space.py --help

# 2. Aggregate qa1/qa2/qa5 x 0k-32k (n=100), babilong.metrics scoring
python scripts/score_nested_babilong.py <results_dir>
```

`run_babilong_mem_space.py` produces a nested per-(task, length) result directory;
`score_nested_babilong.py` aggregates it into the qa1/qa2/qa5 × 0k–32k table that
feeds `status/RUN_REGISTRY.md` and `status/BENCHMARK_RESULTS.md`.

---

## Compute / cluster

The project runs across multiple multi-node GPU clusters (H20 + B200/L20A nodes
spanning several CEPH disks). The **authoritative, up-to-date** node table — IPs,
SSH credentials, per-node Python env, and rsync recipes between disks — lives in
**`CODEBUDDY.md`**. It is not duplicated here because it changes often; always read
`CODEBUDDY.md` (and `status/SESSION_HANDOFF.md`) before launching remote jobs.

---

## Where to start (new contributor)

1. Read `status/SESSION_HANDOFF.md` — one-paragraph current state + active runs.
2. Skim `status/RUN_REGISTRY.md` — what configs have been run and their scores.
3. Read `versions/v20_read_based_slot_lifecycle.md` — the current research frontier.
4. See `CODEBUDDY.md` for cluster setup and the team's operating conventions.

Abandoned directions (RMT, sparse-memory, Q-Filters, SWA, h-series dual-gate,
attention-matching, P2/P8/P11 pre-capacity-sweep, routeA) are archived under
`legacy/` for git-history reference and are **not** imported by live code.
