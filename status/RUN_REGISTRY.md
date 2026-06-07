# RUN_REGISTRY.md — 训练 run 配置 + BABILong 结果总账

> **用途**：本文件是 mem_space 系列每个训练 run 的**配置 + 离线 BABILong eval 结果**的横向对照总账。
> 每启动一个新 run / 跑完一次 eval，必须在此追加或更新对应行，方便快速回答"X 配置 vs Y 配置在 BABILong 上差多少"。
> 评测口径统一：`scripts/run_babilong_mem_space.py`，n=100/length，babilong.metrics（`compare_answers`），qa1/qa2/qa5 × 0k-32k。
> 与 `status/BENCHMARK_RESULTS.md` 的分工：BENCHMARK_RESULTS 是含外部论文数字的大杂烩；**本文件只记我们自己的 mem_space run，强调配置可复现 + 同口径对照**。

最后更新：2026-06-05 15:10 GMT+8

---

## 1. 共享默认配置（除非下表标注覆盖）

- backbone：`models/Meta-Llama-3-8B`，bf16，attn=sdpa
- `num_slots=128, top_k=16, selector_dim=128, num_global_slots=4`
- `slot_dim=None`（=backbone hidden 4096），`slot_init=strided_token`，shared_bank=True
- `bptt_window=2`，`curriculum 0:3`（n_ctx=3 → 每样本 3 context chunk + 1 target chunk = **4 chunk**）
- 训练：`total_steps 2000`（chunk256 arm=5000），`lr 1e-4`，`warmup 100`，per-doc Dolmino CPT
- ⚠️ eval 当前**无 cross-chunk SWA**：前 N-1 chunk 只喂进 memory，仅最后一个 chunk 正常 forward（`train_mem_space_dolmino_cpt.py:1326`）。用户已指出 eval 至少应保留 SWA — 见 §4 待办。

---

## 2. Run 配置对照表

| run | 启动脚本 | chunk_size | slot_dim | route_aux | 读路径 / 特殊 | steps | 节点 | 状态 |
|-----|---------|-----------|----------|-----------|--------------|-------|------|------|
| **chunk1024_temp20** | launch_phase8b_chunk1024_temp20.sh | 1024 | 4096 | — | SFT phase8b 谱系，selector_temp=20 | — | — | DONE（基线，不同谱系）|
| **perdoc_chunk128_local** | launch_mem_space_perdoc_chunk128.sh | 128 | 4096 | OFF | LB=0.01 ent=0.001 temp40 | 2000 | local | DONE |
| **perdoc_chunk128_routeaux** | launch_mem_space_perdoc_chunk128_routeaux_remote.sh | 128 | 4096 | 1.0 | LB=0.01 ent=0.001 temp40 | 2000 | .196 | DONE |
| **perdoc_chunk128_p7p9** | launch_mem_space_p7p9.sh | 128 | 4096 | 1.0 | loss_free_balance + register slots(P9) temp40 | 2000 | — | DONE |
| **perdoc_chunk128_p8** | launch_mem_space_p8.sh | 128 | 4096 | 1.0 | P7P9 + 专用 memory_xattn 读路径(gate_init0.4) | 2000 | — | DONE — **FAILED/REGRESSION**（0k 也塌，frozen xattn 注噪声）|
| **perdoc_chunk128_p8_nullsink** | launch_mem_space_p8_nullsink.sh | 128 | 4096 | 1.0 | P8 + null/sink slot + memory_xattn 可训练+存盘（修 P8 regression）| 2000 | local GPU0-3 | **RUNNING**（step560，ckpt500 已 eval，见 §3）|
| **perdoc_chunk256_p8_nullsink_r196** | launch_mem_space_p8_nullsink_chunk256_remote196.sh | 256 | 4096 | 1.0 | 同 nullsink，chunk 128→256 scale-up | 5000 | .196 8-GPU | **RUNNING**（step230/5000）|
| **wbmode_lowrank (slot_dim 16384)** | launch_mem_space_wbmode_lowrank_local.sh | 1024 | **16384** | — | lowrank_gate r=256 | — | — | **CRASHED**（2026-06-04 00:39 rank3 exit1，无 ckpt，无 eval）|

---

## 3. BABILong 结果（accuracy %，n=100，qa1/qa2/qa5 × 0k-32k）

### chunk1024_temp20（slot_dim=4096，chunk1024）— 最强基线，10 task 全测
```
qa   |   0k   1k   2k   4k   8k  16k  32k
qa1  |   89   78   77   58   37   29   24
qa2  |   32   34   51   54   14   21   20
qa5  |   72   68   93   79   49   29   31
qa6  |   83   78   56   57   56   48   52
qa9  |   87   77   45   47   35   40   42
qa10 |   67   68   50   54   46   40   44
```

### perdoc_chunk128 系列（slot_dim=4096，chunk128）
```
                  0k   1k   2k   4k   8k  16k  32k
p7p9      qa1     55   20   27   14   14   10    7
          qa2     28   15   18   18   11   10    5
          qa5     53   42   36   31   27   24    2
routeaux  qa1     20    9    6   12    4    4    0
          qa2      5    7   14   14    4    9    2
          qa5     45   30   36   26   23   11   13
local     qa1     34   10    9    4   14   21    0
          qa2     19   13    6    0   14   11    0
          qa5     45   31   29   28   31   34    2
p8(OLD)   qa1     11    1    0    0    0    0    0   ← regression（0k 也塌）
          qa2      0    0    0    0    0    0    0
          qa5      3   15   14    0    2    7    1
```

### perdoc_chunk256_p8_nullsink_r196 step500（chunk256，5000-step run 的早 ckpt）✅ 健康
```
qa   |   0k   1k   2k   4k   8k  16k  32k
qa1  |   98   35   26   24   20   18   14
qa2  |   44   37   20   20   20   15   --
qa5  |   78   76   55   43   44   39   --
```
→ **成功修好 old-P8 的 0k 塌方**（old qa1_0k=11 → 98）。null-sink + 可训练 memory_xattn 读路径生效。0k 接近 chunk1024 基线（89）。

### perdoc_chunk128_p8_nullsink step1000 ❌ 全崩（乱码）
```
qa   |   0k   1k   2k   4k   8k  16k  32k
qa1  |    0    1    0    1    0    0    5
qa2  |    1    0    0    0    0    0   --
qa5  |    0    0    0    0    0    0   --
```
→ **不是 adapter 坏了**：researcher gp-35 查实 step1000 ckpt 恰好存盘在 TF-loss spike（step895-1010，PPL~3000）顶上；chunk 越小注入越频繁（128 是 256 的 2×）放大崩坏成 'the the the' 死循环。step500 ckpt 健康、step1500 已恢复 lm~4.0。**结论：chunk128 用 step500 交付；按 TF-loss-min 选 ckpt 而非固定步数。**

### perdoc_chunk512 / chunk1024 p8_nullsink_diskB
> chunk512 step500 eval 进行中（GPU5，启动 2026-06-05 20:36）；step1000 rsync 中。chunk1024 step1000 约 1h 后存盘。验证假设：chunk 越大、step1000 在 5000-step run 中越早期 → step1000 越不易撞 spike。

> ⚠️ **eval 评分口径修正（2026-06-05）**：`compare_answers(target, output, question, TASK_LABELS[task])` 第 4 参数必须是 `TASK_LABELS[task]`（候选标签集），传 task 名会让所有分都算成 0。

### 4-arm chunk512 ablation step500（2026-06-07 评完，base = top_k16 nullsink）
全部 chunk512 / slot_dim4096 / 5000步训练 / step500 ckpt 交付 / 同口径 n=100 qa1/qa2/qa5：

```
arm                          qa  |   0k   1k   2k   4k   8k  16k  32k
top_k16 (baseline P8)        qa1 |   96   47   52   37   30   28   29
                             qa2 |   49   49   35   25   22   22   18
                             qa5 |   85   76   77   54   48   45   41
P11 delta-rule+normreadout★  qa1 |   98   68   51   32   21   26   20
                             qa2 |   59   42   32   24   18   21   35
                             qa5 |   82   86   83   64   50   35   --(32k draining)
P10 keyrep0.05+ST-Gumbel     qa1 |   85   61   27   22   19   26   10
                             qa2 |   44   17    8   16   17    7   11
                             qa5 |   74   68   36   24   15   21   14
topk8 (P8b)                  qa1 |   91   34   33   15   16   17   17
                             qa2 |   53   37   21   14    8    7    8
                             qa5 |   16   51   52   25   24   26   --
```
→ **裁决：P11 delta-rule + normalized writeback 是新最佳臂**。qa5 1k-8k（86/83/64/50）显著超 top_k16 基线（76/77/54/48），qa1/qa2 中长度也持平或更好。**delta-rule 写残差 + 归一化 readout 提升了长上下文检索保持。** P10（ST-Gumbel 硬路由 top1_sim=1.0 但 retrieval 反而退化）和 topk8（top_k 减半→检索覆盖不足，qa5_0k=16 异常低）均劣于基线，REJECTED。

---

## 4. 关键观察 & 待办

1. **chunk1024 全面 >> chunk128**（qa1_0k 89 vs 55，qa5_2k 93 vs 36）→ 缩小 chunk 大幅削弱每步局部窗口（SWA），长句掉分是预期。**chunk size 是当前最大杠杆**。
2. **slot_dim 4096→16384 对照缺失**：唯一的 16384 run 启动即崩、无 ckpt 无 eval。需修 wbmode 启动失败才能补这个对照。
3. **eval 无 cross-chunk SWA**：可能系统性低估真实能力（前文只能走 slots）。用户已指出 eval 至少应保留 SWA — 待加 eval 选项重测。
4. **slot 装 token 级 hidden 而非语义摘要** → BABILong（NIAH 式事实定位）相对行、LongEval（需全局总结）弱，符合预期。

---

## 5. 物理 batch_size 上限速查（2026-06-07 实测，commit ac2abe4 起支持 bs>1）

**单卡显存上限**，P11 配置（8B + gradient_checkpointing + bf16 + num_slots128/top_k16/slot_dim4096 + L3 + dual_gate + curriculum 0:3 + bptt_window2），`torch.max_memory_allocated`，在 .249 空闲 H20（95 GiB usable）实测。每档跑 8 步 smoke，全部无 OOM / 无 non-finite / loss 正常。

| chunk | bs | peak alloc (GiB) | free (GiB) | 状态 |
|-------|----|------------------|------------|------|
| 128   | 1  | 61.2 | 33.8 | ✅ |
| 128   | 2  | 62.0 | 33.0 | ✅ |
| 128   | 4  | 63.7 | 31.3 | ✅ |
| 128   | 8  | 67.0 | 28.0 | ✅ |
| 128   | 16 | 75.8 | 19.2 | ✅ |
| 128   | 24 | 88.0 | 7.0  | ✅ 接近顶 |
| 128   | 32 | —    | —    | ❌ OOM |
| 512   | 1  | 63.2 | 31.8 | ✅ |
| 512   | 2  | 65.9 | 29.1 | ✅ |
| 512   | 4  | 71.4 | 23.6 | ✅ |
| 512   | 6  | 80.9 | 14.1 | ✅ |
| 512   | 7  | 85.8 | 9.2  | ✅ |
| 512   | 8  | 90.7 | 4.3  | ⚠️ 能跑但余量极小，activation peak 抖动有风险 |

**推荐物理 bs（留 ≥10 GiB 余量给 activation peak + NCCL buffer + DDP）：**
- **H20 (97 GiB)**：chunk128 → **bs=16**（19GB free，再大边际递减）；chunk512 → **bs=6**（14GB free）。要更激进可 chunk128 bs=24 / chunk512 bs=7，但余量 <10GB。
- **H800 (80 GiB)**：静态占用本身更高（实测 chunk128 bs=1 ≈74GB，比 H20 高，疑 attn/碎片差异）→ **chunk128/512 维持 bs=1，用 --gradient_accumulation_steps 提有效 batch**（grad_accum 不增峰值显存）。H800 不建议物理 bs>1。
- 增量规律：~60GB 是冻结 backbone+adapter+optimizer 静态占用，bs 只加 activation；chunk128 每 +1 bs ≈ +0.5GB，chunk512 每 +1 bs ≈ +2.5GB（线性）。

**落账目的**：未来 launch 直接查表设 `--batch_size`，无需重新 probe。验证脚本 `scripts/test_mem_space_batch_correctness.py`（bs2==2×bs1 loss 等价 rel<1e-4 + per-sample slot 独立）。

### 已 fold 进 launch 脚本（2026-06-07，eff_batch 恒定）

**约束**：bs>1 的目的是「相同优化动态下的 wall-clock 加速」，必须**保持 eff_batch 不变**以与既有 bs=1 P11 ladder 可比。既有脚本 eff_batch = bs1 × grad_accum4 × 8gpu = **32**。要 eff_batch 恒定 + grad_accum 取整 → 物理 bs 上限被 grad_accum(=4) 卡死在 **bs4/ga1**（bs×ga 必须=4）。物理显存上限（chunk512 bs7 / chunk128 bs24）用不上，除非提高 eff_batch（那就不是同口径了）。

| 脚本 | chunk | 旧(bs×ga) | 新(bs×ga) | eff_batch | peak alloc |
|------|-------|-----------|-----------|-----------|------------|
| `launch_mem_space_p11_chunk512_remote196.sh` | 512 | 1×4 | **4×1** | 32 | 71.4 GiB |
| `launch_mem_space_p11_chunk256_remote196.sh` | 256 | 1×4 | **4×1** | 32 | <71 GiB |
| `launch_mem_space_perdoc_chunk128.sh` | 128 | 1×4 | **4×1** | 32 | 63.7 GiB |
| `launch_mem_space_perdoc_chunk128_routeaux_remote.sh` | 128 | 1×4 | **4×1** | 32 | 63.7 GiB |

- 每个脚本头部加了 H20 ceiling + margin 注释，注释在 `bash -c "..."` 引号外（`bash -n` 已校验语法 OK）。
- **chunk1024 / H800 脚本保持 bs=1 不变**：chunk1024 未在本次 probe 覆盖范围；H800 静态占用更高，维持 bs=1 + grad_accum。
- 未触碰任何在跑训练（local chunk256 step4945 / chunk1024 step2560 都是 bs=1 mid-flight），改动只对未来 launch 生效。
