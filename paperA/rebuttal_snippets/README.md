# paperA/rebuttal_snippets/

Drop-in LaTeX / plaintext 片段用于 rebuttal 答复 letter，若 Paper A rebuttal 用到今晚 audit 的结论。

## ⚠️ 2026-08-06 15:20 更新：#167 已解决，无漂移

provenance 找到了（`bench_results/p0_13_quality_latency/latency/latency_proc{0,1,2}.json`），
池化 3×20 raw reads 得 931.9195/664.3577 ms、ratio 1.40274、p10/p90 931.51/941.94 与 663.71/667.10，
**六个数字逐项对上 tex**。.82 独占重跑（同 env 同 pack sha）936.97/667.53，ratio 1.40365 —— speedup
复现到 4 位有效数字。**tex 不改。**

→ **用 `latency_reproducibility.tex`**（正确的强答复）。
→ `latency_provenance_own_drift.tex` 已作废（它 own 一个不存在的漂移），仅留作被撤回推理的记录。

## 内容

- **`latency_provenance_own_drift.tex`** — 若 #167 三选一中选 **(a) own <2% 漂**：
  一段 rebuttal 措辞，坦承 `tab_replay_latency` 里 931.9/664.4 ms 与最近的 disk source 差 ≤ 2%，
  但方向 1.37–1.41× 在所有 candidates 都成立；paired 质量 gap（3.12pt, CI [2.36, 3.93]）从
  `paperA/anonymous_artifact/scores/p0_13_quality_latency/` 独立可复现。适合放 rebuttal
  response letter 或 appendix 补注。

## 若选其他 #167 选项

- **(b) 找回原始 log**：`grep` 全盘 `bench_results/**/*.json` 与 `paperA/artifacts/**/*.json` 已扫过；
  最近命中是 P1.8 128k|cpu G=1（差 3-13 ms）。若要精确 931.9/664.4，需要看 archived rerun log
  或 `.tmp/**/timing.log`——目前搜过的位置见 `paperA/audit_20260806/latency_provenance_audit.md`。
- **(c) rerun**：直接跑 `bench_results/p0_12_acceptance/` 对应 script 60-reads median 更新 tex 到最新 primitive。
  ~15 min GPU × 8 卡。

## 数据来源

`paperA/audit_20260806/latency_provenance_audit.md`（commit `550a81a`）
`paperA/audit_20260806/primitive_numbers_disk_provenance.md`（commit `9883ef9`）
