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
## Hy3 QCMem 长档 RULER eval (2026-07-12, 本机 8× L20A, 收官验证)
scripts/eval_ruler_qcmem_hy3.py (commit 52e5bbf) | Hy3 hy_v3 80L MoE device_map=auto 8卡分片 + 蒸馏adapter(qcmem_distill_hy3_j32_r32/final, j32 LoRA r32/α64 layers[32:79]) | bm25 topk=8 恒定read + resume_j=32 + 官方 string_match_all | limit=50 | log logs/hy3_ruler_longctx.log out ruler_results/hy3_qcmem_j32_longctx/
- niah_single_2 16k: recall=98.0 (read_len~4312, 1173s)
- niah_single_2 32k: recall=92.0 (read_len~4581, 1384s)
- niah_single_2 64k: recall=100.0 (intermediate 10/50, read_len~4.6k)  [跑中→128k + niah_multikey_1]
- ★关键: read_len 4.3k→4.6k 基本恒定(context翻倍不涨), per-sample ~27s 恒定, 0 OOM through 64k. 超窗口可用 + read恒定坐实.
- niah_single_2 128k: recall=98.0 (read_len~4548, 1664s); niah_multikey_1 128k: recall=100.0 (read_len~4275, 1595s). 16k-128k 全档 DONE.

### Hy3 QCMem 256k (2026-07-12, 本机 8× L20A, 长档最后一档) — RUNNING
- out ruler_results/hy3_qcmem_j32_256k/, log logs/hy3_ruler_256k.log, output_name hy3_qcmem_j32_256k
- niah_single_2/niah_multikey_1 × 256k, limit=50, bm25 topk=8, resume_j=32, 蒸馏adapter j32
- 起因: 之前"起不来"根因=`import scripts.eval_ruler_mem_space` 需 ~22s(fla triton+tf+sklearn) > 15s timeout, 脚本本身无bug. 256k 在 _LENGTH_TOKENS 中已支持(262144), topk=8 恒定read → 与128k同量级(~4.3k), 可跑不OOM.
- ✅ 已确认启动: Hy3 107s加载分片8卡, LoRA真加载(672 tensors sum|lora_B|=7.24e4>>0), first sample ~52s → 预计 ~90-100min 出完整 256k.
