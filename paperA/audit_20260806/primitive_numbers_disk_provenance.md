# paperA primitive numbers disk provenance audit (2026-08-06 04:40)

MAIN 独立核实版。**取代 subagent 初版结论**（初版只搜 wzc1 盘，误报 2/3 "文件不在磁盘"；
真实原因是这两组实验跑在 **.82 (zwfy6 盘)**，见文末方法学教训）。

## 结论: 3/3 精确匹配 ✓

## 1. RULER macros 99.19 / 96.07 + 3.12 gap + CI [2.36, 3.93]  ✓

- **节点/盘**: `.82` (28.82.250.82:36000) — **zwfy6 盘**，不在 wzc1
- **source**:
  - `bench_results/p0_13_quality_latency/summary.json`
  - `bench_results/p0_13_quality_latency/stats.json`
- **disk exact**:
  - `macro/armA           = 99.187`  → tex `99.19`  ✓（四舍五入）
  - `macro/armB           = 96.067`  → tex `96.07`  ✓
  - `macro/diff_A_minus_B = 3.12`    → tex `3.12`   ✓（精确）
  - `stats/paired_bootstrap_95ci = [2.36, 3.9333]` → tex `[2.36, 3.93]` ✓
  - `stats/mcnemar` 字段存在 → tex "exact McNemar" ✓
  - `summary/all_packs_paired_1to1 = True` → tex "paired" 声明 ✓
- **verdict**: ✓ 精确匹配，rebuttal 可直接引用
- **注**: subagent 提到 `tab_pareto.tex` 写 99.20 vs `tab_replay_latency.tex` 写 99.19。
  磁盘真值 99.187 → 99.19 才是正确的四舍五入；**tab_pareto.tex 的 99.20 是错的**，
  需在 rebuttal/camera-ready 前统一（见新任务）。

## 2. context-position overlap 92.5 → 98.5 (w=32) / 99.0 (w=128)  ✓

- **节点/盘**: `.82` — **zwfy6 盘**
- **source**:
  - `bench_results/p0_17_e2_overlap/summary.json`
  - `bench_results/p0_17_e2_overlap/stats.json`
- **disk exact**:
  - `macro/armB_chunk_local_w0_deployable = 92.5`  → tex `92.5`  ✓
  - `macro/armE2_w32_overlap_write        = 98.5`  → tex `98.5`  ✓
  - `macro/armE2_w64_overlap_write        = 98.5`   (tex 未引用 w=64)
  - `macro/armE2_w128_overlap_write       = 99.0`  → tex `99.0`  ✓
  - `pairwise/E2_w32_vs_B/paired_bootstrap_95ci = [3.0, 9.5]` → tex caption CI ✓
  - `prereg_target/baseline_w0_pooled = 92.5`, `best_width_pooled = 99.0` ✓
- **verdict**: ✓ 精确匹配 + 有 pre-registered target 记录（强化可信度）

## 3. BM25 delta 11.56  ✓

- **节点/盘**: 本机 LOCAL — **wzc1 盘**
- **source**: `paperA/analysis/equal_latency/source/bm25/decision.json`
- **disk exact**:
  - `primary_anchor_diff = -11.56`  → tex "leads by 11.56 points" ✓
  - `primary_anchor_95ci = [-14.444, -8.667]`（符号约定：负 = raw replay 领先）
- **verdict**: ✓ 精确匹配

## 数字 provenance 汇总（全 paperA）

| 数字 | 类型 | 盘/节点 | 状态 |
|------|------|---------|------|
| 99.19 / 96.07 RULER | primitive | .82 / zwfy6 | ✓ 精确 |
| 3.12 + CI [2.36, 3.93] | primitive | .82 / zwfy6 | ✓ 精确 |
| 92.5 / 98.5 / 99.0 overlap | primitive | .82 / zwfy6 | ✓ 精确 |
| 11.56 BM25 | primitive | LOCAL / wzc1 | ✓ 精确 |
| 931.9 / 664.4 ms latency | primitive | 追不到精确源 | ⚠ <2% 漂（见 latency_provenance_audit.md） |
| 2.74× = 6.035/2.202 | derived | — | ✓ 代数闭合 |
| 3.12 = 99.19−96.07 | derived | — | ✓ 代数闭合 |
| 1.403× = 931.9/664.4 | derived | — | ✓ 代数闭合 |
| 6.0pp = 98.5−92.5 | derived | — | ✓ 代数闭合 |

**5 组 primitive 中 4 组精确匹配磁盘，1 组（latency）有 <2% 漂。**

## Rebuttal impact

- **质量数字全部可辩护**：RULER macro / overlap / BM25 delta 都能指向 .82 上的
  `summary.json` + `stats.json`（含 paired bootstrap CI + exact McNemar + pre-reg target）。
- **唯一软点仍是 latency 931.9/664.4 ms**，见 #167。方向 1.4× 在所有 disk source 都成立。
- **发现一处 tex 内部不一致**：`tab_pareto.tex` 的 `99.20` 与磁盘真值 99.187 不符，
  应为 `99.19`（与 `tab_replay_latency.tex` 一致）。
- **可复现性运维要点**：P0.13/P0.17 artifact **只在 zwfy6**。若要打包 anonymous artifact
  或 camera-ready，必须先 `scp -O` 到 wzc1 —— 当前 `paperA/anonymous_artifact/` 下没有它们。

## 方法学教训

subagent 只搜了当前工作目录所在盘（wzc1），对 2/3 数字得出 "no candidate found"。
**paperA 的实验横跨两个物理盘**：`launch_p0_13_82.sh` 显式跑在 .82，输出落 zwfy6。
以后派 provenance audit 必须在 prompt 里写明"跨 4 节点 2 盘搜索，含 .82/.73 的
`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`"。
这与 CLAUDE.md 顶部「两个物理盘」纠正条是同一个坑的第 N 次复现。
