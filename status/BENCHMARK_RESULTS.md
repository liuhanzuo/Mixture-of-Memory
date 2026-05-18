# BABILong Benchmark Results
# 所有实验结果汇总，方便查阅和对比。
# 格式：10 task × 7 length (0k/1k/2k/4k/8k/16k/32k) × 100 samples per cell。
# 评分：babilong.metrics.compare_answers（与论文口径一致）。
# 更新时间：2026-05-18

---

## 我们的方法（Mixture-of-Memory）

### P8 — L1+L3 dual-gate, Llama-3-8B-Instruct, 500 steps
**Backbone**: Meta-Llama-3-8B-Instruct (8B)
**Config**: L1(512 slots, top_k=64) + L3(64 summary tokens), dual gate, shared bank, selector_temp=1.0
**Training**: 500 steps, tasks=qa1/qa2/qa5, lengths=1k/2k/4k, lr=2e-5
**Eval date**: 2026-05-18
**Result dir**: outputs/eval_p8_full_20260518_144631/

```
task     0k   1k   2k   4k   8k  16k  32k    avg
qa1      89   89   77   74   62   41   32   66.3
qa2      44   48   44   35   48   36   23   39.7
qa3      29   38   29   26   37   37   25   31.6
qa4      51   57   47   50   48   27   23   43.3
qa5      72   73   66   67   88   65   50   68.7
qa6      84   80   74   60   56   48   43   63.6
qa7      36   28   18   17    0    0    0   14.1
qa8      61   54   41   41   34   19    9   37.0
qa9      90   81   70   66   64   58   60   69.9
qa10     74   65   57   50   60   50   40   56.6
AVG    63.0 61.3 52.3 48.6 49.7 38.1 30.5   49.1
```
**Overall mean (10 tasks × 7 lengths)**: **49.1**

Notes:
- Short context (0k/1k) very strong: 63/61 avg
- qa5/qa9 strongest tasks (retrieval/coreference)
- qa7/qa8 collapse at 8k+ (counting tasks — EMA decay issue, motivation for v6/v7)

---

### v2 final — L1+L3 dual-gate, Llama-3.2-1B-Instruct, 10000 steps
**Backbone**: Llama-3.2-1B-Instruct (1B)
**Config**: same L1+L3 recipe as P8
**Training**: 10000 steps, tasks=qa1/qa2/qa5, lengths=1k/2k/4k
**Eval date**: 2026-05-18
**Result dir**: outputs/eval_phase1b_v2_full_20260518/p1bv2_final_fullqa_20260518/

```
task     0k   1k   2k   4k   8k  16k  32k    avg
qa1      66   39   46   41   40   47   36   45.0
qa2      28   20   23   20   30   21   19   23.0
qa3      29   18   25   16   29   17   12   20.9
qa4      24   27   29   28   27   24   18   25.3
qa5      51   46   37   32   76   66   54   51.7
qa6      45   24   35   31   51   43   50   39.9
qa7      18   12   12   27    1    0    0   10.0
qa8      45   14   20   20   28   18   12   22.4
qa9      56   36   28   41   54   58   54   46.7
qa10     35    9   25   24   34   46   45   31.1
AVG    39.7 24.5 28.0 28.0 37.0 34.0 30.0   31.6
```
**Overall mean**: **31.6**

---

### v2-base (step4000) — L1+L3 dual-gate, raw Llama-3.2-1B (no Instruct), 5000 steps
**Backbone**: Meta-Llama-3.2-1B base (1B, NO chat template)
**Config**: same L1+L3 recipe as v2
**Training**: 5000 steps
**Eval date**: 2026-05-18
**Result dir**: outputs/eval_phase1b_v2_llama32_1b_base_step4000/

```
task     0k   1k   2k   4k   8k  16k  32k    avg
qa1      59   47   38   42   44   44   32   43.7
qa2      34   21   20   15   28   26   24   24.0
qa3      21   20   24   21   26   24   17   21.9
qa4      24   25   29   25   29   31   19   26.0
qa5      56   47   48   47   77   64   53   56.0
qa6      37   25   38   36   41   38   40   36.4
qa7      18   15   29   30    0    0    0   13.1
qa8      42   21   24   15   27   20   15   23.4
qa9      39   22   29   36   36   30   21   30.4
qa10     40   17   15   19   49   41   32   30.4
AVG    37.0 26.0 29.4 28.6 35.7 31.8 25.3   30.5
```
**Overall mean**: **30.5**

---

### Plain Meta-Llama-3.2-1B baseline (no memory, vanilla inference)
**Backbone**: Meta-Llama-3.2-1B base (1B)
**Config**: vanilla HF generation, no memory module
**Eval date**: 2026-05-18
**Result dir**: outputs/eval_llama32_1b_base_b2002_20260518_110237/

```
task     0k   1k   2k   4k   8k  16k  32k    avg
qa1       0    1   28   42   48   43   30   27.4
qa2       0    0   26   18   24   17   15   14.3
qa3       0    0    5   12   26   26   22   13.0
qa4       0    2    4    6   10   10   22    7.7
qa5       0    1   10    9   13   24   34   13.0
qa6       0    0    0    1    4    3   20    4.0
qa7       0    0    0    0    2    1    3    0.9
qa8       0    0    0    0    4    3   11    2.6
qa9       0    0    0    0   11   19   42   10.3
qa10      0    0    0    0    8    9   26    6.1
AVG     0.0  0.4  7.3  8.8 15.0 15.5 22.5    9.9
```
**Overall mean**: **9.9**

Notes:
- 0k columns all 0: raw base model cannot answer BABILong without any context retrieval
- Short lengths (0k/1k) near-zero; only starts working at 2k+

---

## 其他论文（参考数字）

### LM2-1.7B (Large Memory Models, arXiv 2502.06049)
**Backbone**: vanilla-Llama-1.7B (**trained from scratch** by LM2 team, NOT Meta's open release)
**Method**: auxiliary memory module with dual input/forget gates
**Source**: Table 1 of the LM2 paper; only qa1/qa2/qa5 reported; ≥8k is aggregate average
**Note**: NOT directly comparable to our work — backbone is a custom 1.7B pretrained model

```
length   qa1   qa2   qa5   10-task-avg
0k       99    89    98    92.5
1k       85    59    91    78.3
2k       58    43    87    65.8
4k       46    37    78    55.9
>=8k avg 23.8  15.0  38.8  39.9
```

### LM2 paper baseline: Meta Llama-3.2-1.2B (vanilla, no memory)
**Note**: This is Meta's released Llama-3.2-1B used as baseline by LM2 paper; directly comparable to our plain baseline above.

```
length   qa1   qa2   qa5   10-task-avg
0k       54    25    59    40.7
1k       48    22    69    39.5
2k       44    18    64    38.6
4k       37    16    56    36.8
>=8k avg 19     8    36.5  28.2
```

### BABILong paper (NeurIPS 2024, arXiv 2406.10149) — Table 4
**Backbone**: Meta-Llama-3-8B-Instruct / Meta-Llama-3.1-8B-Instruct
**Note**: n=1000 samples per cell (vs our n=100)

```
qa   length  Llama-3-8B-It  Llama-3.1-8B-It
qa1  0k      98             99
qa1  1k      93             97
qa1  2k      80             97
qa1  4k      16             95
qa1  8k       7             83
qa1  16k     31            100
qa1  32k     23             87
qa2  0k      47             53
qa2  4k      10             51
qa2  8k       4             44
qa2  32k      2             56
qa5  0k      85             81
qa5  4k      52             85
qa5  8k      43             86
qa5  32k     50             85
```
**P8 vs Llama-3-8B-Instruct vanilla overall**: P8 mean=49.1 vs paper Llama-3-8B mean≈42.6 (+6.5pp)

---

## 正在进行的实验

### v6 — replace writeback (slot ← s_new, no EMA)
**Status**: Training started 2026-05-18 on local H20 8 GPUs
**Config**: same as v2 (1B Instruct) but with --use_replace_writeback
**Expected**: better counting/tracking (qa7/qa8), potentially worse long-range retrieval
**Results**: TBD

### v7 — hybrid EMA + 8 global always-on slots
**Status**: Training started 2026-05-18 on remote H20 28.59.80.196 8 GPUs
**Config**: same as v2 (1B Instruct) + --num_global_slots 8
**Expected**: keep retrieval quality while improving counting via always-on registers
**Results**: TBD

### MemoryLLM-8B eval
**Status**: Running on B200 (28.89.17.144), extremely slow (~100s/sample due to chunked inject_memory)
**Config**: full qa1-10 × 0k-32k, no chat template
**Expected**: ~10-20 hours to complete
**Results**: TBD
