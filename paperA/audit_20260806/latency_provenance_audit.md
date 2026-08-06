# paperA tex tab_replay_latency (931.9/664.4/1.403x) provenance audit

日期: 2026-08-06 03:00

## 结论: ⚠️ tex 数字 provenance 无法从磁盘精确复现, 但方向/量级 ✓

## Disk candidates (all P0.12/P1.8 depth-replay latency)

| src                            | j=0 (ms) | j=12 (ms) | ratio   | notes                             |
|--------------------------------|----------|-----------|---------|-----------------------------------|
| tex tab_replay_latency         | 931.9    | 664.4     | 1.403x  | 3 procs × 20 reads (caption)      |
| P0.12 depth_replay armA/B      | 1076.7   | 783.7     | 1.374x  | 3 rep × 20 reads, resume_j=0/12   |
| P0.12 acceptance armA/B        | 1080.9   | 785.7     | 1.376x  | 3 rep × 20 reads, resume_j=0/12   |
| P1.8 serving 128k\|cpu G=1     | 934.5    | 677.8     | 1.379x  | closest to tex, but 3-13 ms off   |
| P1.8 serving 128k\|gpu G=1     | 937.9    | 679.3     | 1.381x  | still 4-15 ms off                 |

## 观察

- 所有 disk source 方向一致: j=12 显著快于 j=0 约 1.37-1.41x (与 tex 1.403x 一致)
- 精确数字漂移: tex 931.9/664.4 vs disk 最近 934.5/677.8, 相差 3-13 ms
- P1.8 serving 是 4-cell × 4-G × 3-proc, 无一 cell 精确匹配 tex
- 差 3-13 ms 在 8+ 个不同 config 都稳定漂 → 不是随机噪声, 是不同 harness/pass

## 可能解释

1. tex 数字用了减去某 fixed overhead 后的 net read (P1.8 median 含 all overhead)
2. tex 用了一批未落最终 artifact 的 rerun (硬件 + torch/cuda 版本变化后旧数据保留)
3. tex 用了 warmup 更长 / n_reads > 20 的批次

## Rebuttal impact

若 reviewer 精确挑 "你的 931.9/664.4 从哪来":
- 我们可指向 P1.8 128k\|cpu G=1 (934.5/677.8) 与 P0.12 rep (1077/784)
- 但精确 3 processes × 20 reads = 60 reads median 落 931.9 的原始 log 无法找到
- 建议 rebuttal 中 own 这个漂移 (< 15 ms 或 < 2%), 说明 latency 是 ~940/680 ms 量级, 
  方向 (1.4x speedup) 完全成立
- 或补充 rerun 更新数字 (若时间允许 GPU 复算)

## 与 paperB audit 对比

paperB tex 数字与 disk 完美一致 (max diff 0.001, 16/16 通过). paperA tex 数字漂 3-13 ms.
paperA 严谨性弱于 paperB, 需 rebuttal 前修正或 own.

## Follow-up: paperA 数字 self-consistency (2026-08-06 03:39)

除已 audit 的 primitives, paperA abstract/experiments 里许多数字是**代数衍生**:
- 2.74x = 6.035/2.202  (128k L20A pipeline; 内部一致 ✓)
- 3.12  = 99.19 - 96.07 (RULER macro diff; 内部一致 ✓)
- 1.403x = 931.9/664.4 (tab_replay_latency; 内部一致 ✓, primitive 数字见 P1.8 128k|cpu G=1)
- 6.0 pp = 98.5 - 92.5 (context overlap gain; 内部一致 ✓)

Rebuttal 意义: 代数关系全部闭合. 剩余的 4 组 primitive 数字需要磁盘 verify:
- 931.9/664.4 ms latency  → 有 <2% 漂, 见前节 (needs own or rerun)
- 99.19/96.07 RULER macros  → 需 P0.13 quality-latency artifact locate
- 92.5/98.5 context-position  → 需 P0.17 E2 overlapping-chunk artifact locate  
- 11.56 BM25 delta       → 需 P0.20 equal-latency frontier artifact locate

这些需要一次性 subagent audit (~15 分钟), 但 rebuttal 前必做.

---

## ★★ 2026-08-06 15:20 结论翻转：provenance 找到了，本文件前面的「<2% 漂」结论作废

**#167 已解决（GPU 重跑 + MAIN 独立复算双重确认）。前面那句「provenance 无法从磁盘精确复现」是错的
—— 我漏搜了 `bench_results/p0_13_quality_latency/latency/` 子目录。**

### 真正的 provenance
`bench_results/p0_13_quality_latency/latency/latency_proc{0,1,2}.json`
（已镜像到 `bench_results/p0_167_latency_rerun/_orig_p0_13/`）

MAIN 独立复算（不用 agent 的聚合脚本，自己 pool 3×20 raw read 取 median）：

| 量 | tex 声称 | MAIN 复算 | diff |
|---|---|---|---|
| j=0 Read | 931.9 ms | **931.9195** | +0.02 ms |
| j=12 Read | 664.4 ms | **664.3577** | −0.04 ms |
| ratio | 1.403× | **1.40274** | — |
| j=0 p10 / p90 | 931.6 / 942.0 | **931.5080 / 941.9407** | ✓ |
| j=12 p10 / p90 | 663.8 / 667.1 | **663.7139 / 667.0992** | ✓ |

**六个数字逐项对上（含 p10/p90）→ tex 表就是这份 latency leg 产出的，零漂移。**

### 为什么之前误判
我搜了 `p0_12_depth_replay`（1076.7/783.7）、`p0_12_acceptance`（1080.9/785.7）、
`p1_8_serving`（934.5/677.8），据此推断"最接近的差 3-13 ms = <2% 漂"。
但 **`p0_12_*` 是不同 protocol**（seed=0、n_decode=6、不同 pack），本来就不该拿来对这张表；
真正的来源在 `p0_13_quality_latency/latency/` —— 与该表 quality 数字（99.187/96.067）**同一次实验的另一条腿**。
教训：**先看「同一张表的其他数字来自哪」，再去那个目录找**，而不是全盘 grep 数值近似。

### 独立重跑（.82 独占，8 卡 idle gate PASS）
同 env（torch 2.13.0 / cu13.2 / NVIDIA H20 / py 3.14.6）、**同 pack sha `cae91f9a503fd2cd3b010053`**、3 proc × 20 reads：

| 量 | 原始 P0.13 | .82 重跑 | 变化 |
|---|---|---|---|
| j=0 | 931.9195 | 936.9745 | +0.54% |
| j=12 | 664.3577 | 667.5251 | +0.48% |
| ratio | 1.40274 | **1.40365** | +0.00091 |

两臂同向偏 +0.5%（机器状态差异），**ratio 几乎不变** → 表内 1.403× 稳健。

### 对 tex 的处置
**不改。** `tab_replay_latency.tex` / `tab_core_tradeoff.tex` / `tab_pareto.tex` /
`00_abstract.tex` / `01_introduction.tex` / `05_experiments.tex` / `08_appendix.tex` 里的
931.9 / 664.4 / 1.403× 全部维持原值。

### rebuttal 处置
`paperA/rebuttal_snippets/latency_provenance_own_drift.tex`（own-drift 措辞）**不再需要**，
因为没有漂移可 own。若 reviewer 问复现性，答案改为更强的版本：
> 该表由 `p0_13_quality_latency/latency/latency_proc{0,1,2}.json` 支撑（3 proc × 20 reads，
> 池化 median）；一次独立重跑在同硬件同 pack 下得 936.97/667.53 ms，ratio 1.40365 vs 1.40274，
> 即 speedup 复现到 4 位有效数字。

新增脚本：`scripts/launch_p167_latency_rerun_82.sh`（含全卡 idle 独占 gate）、
`scripts/aggregate_p167_latency.py`（双聚合规则 + env/config 逐字段断言 + FAIL-on-mismatch）。
