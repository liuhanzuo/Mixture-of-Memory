# lane3_a02_adapters — A02 stage-1 "zero new training" premise

Scout: read-only. Nothing launched, nothing killed, nothing edited outside this dir.

## VERDICT (one line)

**"Zero new training" is TRUE, but "zero new CODE" is FALSE.** All three assets
(Read-LoRA, Write-LoRA, Qwen3-8B base) exist and are loadable, and the whole thing runs on
`.82` with **0 GB of cross-disk transfer**. BUT A02's stage-1 as written (5 configs x 5
*natural* benchmarks) is **not runnable today**: the only harness that can load a Write LoRA
is RULER-synthetic-only, and none of the 5 natural-task harnesses has write-LoRA plumbing.

---

## ★ FINDING 0 (methodological — affects MAIN's own checks)

**`/apdcephfs_zwfy6` is NOT mounted on LOCAL.** The task brief's claim that ".21 checks can be
done with a plain local ls" is true for wzc1, but the converse trap bit me first:

```
$ ls -d /apdcephfs*          # on LOCAL
/apdcephfs_wzc1
/apdcephfs_wzc1_304376610
$ df -h | grep apdcephfs
dop-fuse  120T  109T  12T  91%  /apdcephfs_wzc1/share_304376610
```

A LOCAL `ls /apdcephfs_zwfy6/...` returns "No such file or directory" for **every** zwfy6
path. That is a MOUNT MISS, not file absence. Symmetrically, `.82` cannot see wzc1
(`ls -d /apdcephfs_wzc1` on .82 → No such file or directory). **All zwfy6 evidence below was
obtained by ssh to `.82`.** Any prior report claiming "X is not on zwfy6" based on a LOCAL ls
is void.

---

## FINDING 1 — Read-LoRA is REAL, and on BOTH disks, byte-identical

Real path (named by A02 `SOURCES.md`, and the argparse default of all three write harnesses):
`outputs/qcmem_distill_qwen_j12_r32_4k/final/`

| file | size |
|---|---|
| `adapter_config.json` | 1341 B |
| `adapter_model.safetensors` | 232829168 B (222 MB) |
| `README.md` | 5288 B |

- wzc1: present, mtime Jul 7 00:25, md5 `d0a180a094bdf942b290bf0d6a667eb5`
- zwfy6 (via .82): present, same size, same mtime, **md5 `d0a180a094bdf942b290bf0d6a667eb5` — identical**
- wzc1 also holds step500..step4000 + final (9 ckpts); zwfy6 holds a subset incl. `final`.

`adapter_config.json` confirms it is the genuine flagship: `r=32`, `lora_alpha=64`,
`layers_to_transform=[12..35]` (upper 24 = READ), 7 target modules,
`base_model_name_or_path=/apdcephfs_wzc1/.../models/Qwen--Qwen3-8b`, `peft_version 0.19.0`.

### `outputs/lora_best_ref` — DOES NOT EXIST AT ALL (not even a dangling symlink)
```
$ ls -la .../outputs/lora_best_ref/   -> No such file or directory
$ readlink -f .../outputs/lora_best_ref -> (echoes path unchanged = no such entry)
$ ls .../outputs/ | grep -i lora      -> (empty)
```
The prior "512 bytes" observation does not correspond to anything on wzc1 today. It was **not**
the flagship Read adapter. Treat that lead as closed/red-herring.

---

## FINDING 2 — Write-LoRA is zwfy6-ONLY; the wzc1 copy is an EMPTY STUB

**zwfy6 (real, via .82):** `outputs/qcmem_writepath_distill_qwen_j12_r32/` = **556 MB** (`du -sh`)
```
distill_args.json  993 B
step500/ step1000/ step1500/ step2000/ step2500/
  each: adapter_config.json 1189 B
        adapter_model.safetensors 116414304 B (111 MB)
        README.md 5196 B
```
Per-step md5 of `adapter_model.safetensors` (all distinct → 5 genuinely different ckpts):
`step500 eee37af1…`, `step1000 d2245d0f…`, `step1500 dadad091…`,
`step2000 dd09ad75…`, `step2500 9836c559…`

`step2000/adapter_config.json`: `r=32`, `alpha=64`, `layers_to_transform=[0..11]`,
`base_model_name_or_path=models/Qwen3-8b-local`. **Disjoint from the Read LoRA's [12..35]** —
which is exactly what `eval_p018` asserts at line 1053-1056.

**wzc1 (stub):** `outputs/qcmem_writepath_distill_qwen_j12_r32_b200/` = **1.5K total**,
containing `distill_args.json` ONLY, **zero step dirs**. That b200 run was configured
(mtime Aug 4 15:35, `output_dir: outputs/qcmem_writepath_distill_qwen_j12_r32_b200`) but never
produced a checkpoint. **There is no usable Write adapter on wzc1.**

### Transfer vs. run-on-.82 — run on .82
556 MB at the measured 12-37 MB/s is ~15-45 min, so it is *tolerable*. But it is also
**completely unnecessary**: the read-LoRA, the write-LoRA, the base model, the scripts, and the
`p1_7_h12_oracle` prereq are ALL already on zwfy6, and `.82` has 8 idle H20. **Recommendation:
run on `.82`, transfer 0 GB.** (Going to `.21` would require moving 556 MB write-LoRA — and
`.21` is currently saturating its network with the Dolmino HF download PID 25999, so don't.)

---

## FINDING 3 — Base model: present on BOTH disks, and the repo symlink works on BOTH

| disk | absolute path | size | repo-relative symlink |
|---|---|---|---|
| wzc1 | `/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b` | ~16.4 GB | `models/Qwen3-8b-local -> ...` ✓ |
| zwfy6 | `/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b` | ~16.4 GB | `models/Qwen3-8b-local -> ...` ✓ (mtime Jul 11 17:21) |

**CORRECTION to project lore:** CLAUDE.md says `models/Qwen3-8b-local` is a "wrong path that
does not exist on zwfy6". That is **out of date** — on `.82` the symlink exists and resolves to
the zwfy6 `Qwen--Qwen3-8b`. So the harness default `--model_path models/Qwen3-8b-local` works
as-is on `.82` with no override. (Also present on wzc1 → works on `.21` too.) Note wzc1
additionally has a separate `Qwen3-8B-Base` (16.4 GB) — do NOT substitute it; the adapters were
trained against `Qwen--Qwen3-8b`.

---

## FINDING 4 — the REAL blocker: no natural-task harness can load a Write LoRA

Grep of `--write_lora_ckpt` support across every `scripts/eval_*qcmem*.py`:

| harness | `write_lora_ckpt` | verdict |
|---|---|---|
| `eval_p018_e4_2x2_writecontrol.py` | **yes** (line 1364) | only write-capable harness; **RULER-synthetic only** (`--task` default `niah_multikey_1`, builds samples via `eval_ruler_mem_space`) |
| `eval_qcmem_babilong.py` | 0 | cannot |
| `eval_qcmem_locomo.py` | 0 | cannot |
| `eval_qcmem_longeval.py` | 0 | cannot |
| `eval_qcmem_longbench.py` | 0 | cannot |
| `eval_ruler_qcmem.py` | 0 | cannot |
| `eval_qcmem_infbench.py` | 0 | cannot |

(the `overlap` hits in `eval_qcmem_babilong.py` / `locomo` are all **lexical**-overlap selector
comments, NOT the E2 overlapping-chunk Write; grep-verified line by line.)

Likewise A02 config 3 (`j=12 + overlap w32`): the overlap Write lives in
`eval_p017_e2_overlap_write.py` (`--widths`, default `[32,64,128]`) which is **also**
RULER-synthetic-only (`--task niah_multikey_1`, `ruler._LENGTH_TOKENS`). No natural harness has it.

So of A02's five configs, on natural tasks today:
1. `j=0` full-depth replay — natural harnesses support (`--resume_j 0`) ✓
2. `j=12` Read-LoRA only — supported ✓
3. `j=12 + overlap w32` — **NEEDS CODE** (p017 write path must be ported into 5 harnesses)
4. `j=12 + Write-LoRA` — **NEEDS CODE** (p018 `_load_with_write_lora` must be ported)
5. `j=12 + Write + Read` — **NEEDS CODE** (same; note p018 hard-requires a Read LoRA:
   line 207-208 `if not lora_adapter: raise SystemExit`, so config 4 "Write-only without Read"
   is not expressible in p018 either)

## FINDING 5 — and the RULER-synthetic version of this gate has ALREADY BEEN RUN

On zwfy6 (`.82`) there are four completed p018 result sets:
```
bench_results/p0_18_e4_2x2            (BBWL off)
bench_results/p0_18_e4_bbwl_step1000
bench_results/p0_18_e4_bbwl_step1500
bench_results/p0_18_e4_bbwl_step2000
```
each with `manifest.json pos_sanity.json quality/ stats.json summary.json`, plus logs
`logs/p018_bbwl_step{1000,1500,2000}.out` (Aug 4). This is project task #150, already
`[completed]`, and already distilled into `paperA/artifacts/writepath_distill_150/summary.json`.

`p0_18_e4_bbwl_step2000/summary.json` (n=200 paired, niah_multikey_1 @ 8k+16k, iter_bm25,
topk=12, chunk 512, chat=False):
```
macro  A=100.00  BB=92.50  E0=100.00  BBWL=98.50  X=88.00  Y=100.00
A_vs_BB: diff=+7.50 CI=[4.0,11.5] McNemar p=6.1e-05
```
**So re-running p018 on `.82` would reproduce an existing number, not decide anything.** The
scientific gap is precisely the natural-task transfer, which is the part that needs code.

Runtime evidence for cost estimation (from `logs/p018_bbwl_step2000.out`): 8 quality jobs
(2 cells x 4 shards) on 8 H20 GPUs, all `rc=0`, one wave, wall ≈ 14 min
(log mtimes step2000 17:10 / step1500 17:12 / step1000 17:24 — three full runs inside ~25 min
including staggering). So **one p018 config on 8 GPUs ≈ 15-25 min**, very cheap.

---

## Runtime environment, verified on `.82`

```
/opt/conda/envs/torch-base/bin/python
torch 2.13.0   cuda True   device_count 8
transformers 5.5.4
peft 0.19.1          (adapters were written by peft 0.19.0 → compatible)
nvidia-smi: all 8 GPUs 0 MiB / 0 %      (idle, confirmed)
df /apdcephfs_zwfy6/share_304376610: 86T free
```
Script md5s are **identical across the two disks** for all four relevant files, so the
"zwfy6 checkout is behind" caveat does NOT bite here:
```
eval_p018_e4_2x2_writecontrol.py  5c028119c436baa76e42ec319264961e   (both)
eval_p016_e0_write_control.py     c597edae5742422e6f0459861765f128   (both)
eval_p017_e2_overlap_write.py     951fe518eac8307fd2d604fa30612d9e   (both)
eval_ruler_qcmem.py               5cd7f9e108cf1a5cbba1881f65ee95ad   (both)
_run_p018_e4_8gpu.sh              2e6bb17e2ae59d2ce37f1bf08a4b9496   (both)
```
(.82 HEAD is `2d98c5a`, behind LOCAL, but these files happen to match byte-for-byte.)

Driver prereq also present on zwfy6: `bench_results/p1_7_h12_oracle/` (manifest.json 10550 B,
summary.json, quality/) — this is `P013_MANIFEST_DIR`'s default for the strict-fix pack-sha
cross-check.

RULER length buckets available (`scripts/eval_ruler_mem_space.py:333`):
`1k 2k 4k 8k 16k 32k 64k 128k 256k` — so the p018 cohort **can** be extended to 32k/64k
without code (only 8k+16k were run).

---

## A02 kill gate, VERBATIM (`PROPOSAL.md` lines 62-73)

```
## 决定性系统 gate

用修复后的 Write 重做 equal-latency frontier：

- raw/dense replay latency-matched `k*`
- CoMem-w0
- CoMem-overlap-w32
- CoMem-Write-LoRA
- Joint

若 paired quality CI 仍显著低于 0，则停止"CoMem 优于 RAG"的叙事，
定位为高复用 workload 的 storage/read-compute 方案。
```
Stage-1 success conditions (lines 55-60): ≥3 task families close half the original j0-j12 gap;
LongEval or multikey ≥2pp over Read-only; LoCoMo judge ≥1.5pp; persistent bytes and per-query
Read must not increase; plain-text PPL degradation ≤5%.

**This gate can genuinely KILL** the "CoMem > RAG" framing — but only in its natural-task /
equal-latency form. The synthetic multikey form is already answered (Finding 5).

---

## Recommended launch (if MAIN wants `.82` filled with A02 work today)

Zero-code, zero-transfer, and it *adds* information rather than reproducing: extend the
already-validated p018 BBWL cohort to **longer lengths** (32k, and 64k if time), which is the
one axis where "does the trained Write survive?" is still unmeasured.

```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory && \
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
RUN=1 TASKS="niah_multikey_1" LENGTHS="32k" LIMIT=100 NUM_SHARDS=4 \
GPUS="0 1 2 3 4 5 6 7" \
WRITE_LORA=outputs/qcmem_writepath_distill_qwen_j12_r32/step2000 \
MODEL=models/Qwen3-8b-local \
LORA=outputs/qcmem_distill_qwen_j12_r32_4k/final \
OUTDIR=bench_results/p0_18_e4_bbwl_step2000_32k \
LOGDIR=logs/p0_18_e4_step2000_32k \
setsid nohup bash scripts/_run_p018_e4_8gpu.sh \
  > logs/p018_bbwl_step2000_32k.out 2>&1 &
```
Idempotency: distinct `OUTDIR` **and** distinct `LOGDIR` (the driver keys its flock queue/done
markers off `LOGDIR`, per line 121 comment "per-run override so concurrent adapters don't share
queue/done markers"). Safe to re-run; will not clobber the Aug-4 8k/16k results.

Est. wall: ~20-40 min for 32k (8k+16k took ~14 min); ~1.5-3 h if 64k is added.
This is **decoration, not a kill** — see `what_it_decides`.
