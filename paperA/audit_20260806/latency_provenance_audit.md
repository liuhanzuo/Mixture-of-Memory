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
