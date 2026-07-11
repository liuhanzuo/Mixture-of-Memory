# QCMem RULER 结果台账 (RULER_RESULT_LEDGER.md)

> **权威去重源**：启动任何 QCMem RULER eval 前先查此表，避免重测。
> 自动扫描：`ruler_results/*/_summary.json` (本机 wzc1) + `logs/qcw_qcmem_n100_*.log` (diskB H20)。
> 最后扫描：2026-07-11 21:00 GMT+8。数字=recall/score (%).

## A. 论文正文用的 QCMem RULER 实验 (本机 wzc1, n=50)

| 实验(folder) | 说明 | 覆盖 |
|---|---|---|
| `h2h_qcmem_*` | head-to-head vs KVD/HCache (§2.1) | 3 folders, lengths: 16k,32k,8k |
| `qcmem_128k/64k/4k` | 超窗口长档曲线 (§2.2 scaling) | 1 folders, lengths: 128k |
| `bs_cs512/256/128/1024_*` | chunk-size sweep (§2 ablations) | 1 folders, lengths: 128k |
| `funnel3col_*` | 方向2 funnel 对照 | 9 folders, lengths: 16k,32k,8k |

### 本机各 QCMem folder 明细 (task×length=score, n=50)

- **h2h_hcache_16k**: niah_multikey 16k=0.0; niah_single 16k=2.0
- **h2h_hcache_32k**: niah_multikey 32k=0.0; niah_single 32k=4.0
- **h2h_hcache_8k**: niah_multikey 8k=4.0; niah_single 8k=34.0
- **h2h_kvdirect_16k**: niah_multikey 16k=100.0; niah_single 16k=100.0
- **h2h_kvdirect_32k**: niah_multikey 32k=98.0; niah_single 32k=100.0
- **h2h_kvdirect_8k**: niah_multikey 8k=100.0; niah_single 8k=100.0
- **h2h_qcmem_16k**: niah_multikey 16k=98.0; niah_single 16k=100.0
- **h2h_qcmem_32k**: niah_multikey 32k=90.0; niah_single 32k=100.0
- **h2h_qcmem_8k**: niah_multikey 8k=96.0; niah_single 8k=100.0
- **h2h_smoke**: niah_single 4k=100.0
- **hcache_64k**: niah_single 64k=2.0
- **kvdirect_128k**: niah_single 128k=0.0
- **kvdirect_64k**: niah_multikey 64k=80.0; niah_single 64k=100.0
- **qcmem_128k**: niah_multikey 128k=96.0; niah_single 128k=100.0
- **qcmem_4k**: niah_multikey 4k=90.0; niah_single 4k=100.0
- **qcmem_64k**: niah_multikey 64k=86.0; niah_single 64k=100.0
- **qcmem_tk24_16k**: niah_multikey 16k=82.0; niah_single 16k=100.0
- **qcmem_tk4_16k**: niah_multikey 16k=96.0; niah_single 16k=100.0
- **qcmem_vt_16_32k**: niah_single 16k=98.0; niah_single 32k=96.0; vt 16k=27.2; vt 32k=23.2

## B. n=100 topk×length 网格 (diskB H20, 进行中)

| task | length | topk | recall(n=100) |
|---|---|---|---|
| niah_multikey | 16k | 2 | 99.0 |
| niah_multikey | 16k | 6 | 97.0 |
| niah_single | 16k | 2 | 100.0 |
| niah_single | 16k | 6 | 100.0 |
| niah_single | 8k | 8 | 100.0 |

## C. 尚未测 / 待补 (n=100 网格目标)

n=100 网格目标：niah_single/niah_multikey/vt × {8k,16k,32k,64k,128k} × topk{2,4,6,8,12,16,24}。
已完成见 B 表；启动新 cell 前对照 B 表 + logs/qcw_qcmem_n100_*.log 去重。