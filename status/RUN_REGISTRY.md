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
| **d2b_swa_train_w2** | launch_d2b_swa_train_w2_remote196.sh | 512 | 4096 | 1.0 | P11 deltarule+normreadout 底座 + **cross-chunk SWA TRAIN window W=2**（target forward 扩成 last-2-ctx+target 拼接1536tok，prefix labels-100，bank frozen 防二次写）；eval-side D2a 的训练侧对称版；bs2 eff16（bs4 OOM）| 5000 | .196 8-GPU | **RUNNING**（commit 9d2417f，2026-06-09 16:08 起，step15+ nf=0 健康）|

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
                             qa2 |   59   42   32   24   18   21   21
                             qa5 |   82   86   83   64   50   46   41
P10 keyrep0.05+ST-Gumbel     qa1 |   85   61   27   22   19   26   10
                             qa2 |   44   17    8   16   17    7   11
                             qa5 |   74   68   36   24   15   21   14
topk8 (P8b)                  qa1 |   91   34   33   15   16   17   17
                             qa2 |   53   37   21   14    8    7    8
                             qa5 |   16   51   52   25   24   26   --
```
→ **裁决：P11 delta-rule + normalized writeback 是新最佳臂**。qa5 1k-8k（86/83/64/50）显著超 top_k16 基线（76/77/54/48），qa1/qa2 中长度也持平或更好。**delta-rule 写残差 + 归一化 readout 提升了长上下文检索保持。** P10（ST-Gumbel 硬路由 top1_sim=1.0 但 retrieval 反而退化）和 topk8（top_k 减半→检索覆盖不足，qa5_0k=16 异常低）均劣于基线，REJECTED。

### chunk-size sweep step500（2026-06-07 评完，P11 delta-rule + normreadout 同架构，仅 chunk_size 不同）
同口径 canonical `compare_answers(target,output,question,TASK_LABELS[task])`，n=100，step500 ckpt：

```
chunk                        qa  |   0k   1k   2k   4k   8k  16k  32k
chunk256 deltarule+normro    qa1 |   91   22   22   17   17   15   15
                             qa2 |   40   17   18   17   18   18   18
                             qa5 |   77   69   53   36   37   10   34
chunk512 deltarule+normro★   qa1 |   98   68   51   32   21   26   20
                             qa2 |   59   42   32   24   18   21   --
                             qa5 |   82   86   83   64   50   35   --
chunk1024 deltarule+normro   qa1 |   97   45    7   21   11   11    6
                             qa2 |   47   31    9    7    0    4    9
                             qa5 |   80   41   24   19   16    5    4
```
→ **裁决：chunk512 是同架构下的甜区**。qa5 中长度（1k-8k）chunk512=86/83/64/50 ≫ chunk256=69/53/36/37 ≫ chunk1024=41/24/19/16。chunk1024 仅 0k 强（97/47/80），>0k 急剧掉分（每步局部窗口被 1024 token 稀释 + 注入太稀疏）。chunk256 在 32k 偶有回升（qa5=34）但整体不及 512。
> ⚠️ 与早期 "chunk1024 >> chunk128"（§4-1）不矛盾：那是 **p8_nullsink** 谱系、且对照的是 chunk128。本次是 **P11 delta-rule+normreadout** 谱系 chunk{256,512,1024} 三点对照，512 居中最优。**step500 早 ckpt，三个 run 都在跑满 5000 步，后续 ckpt 待评。**

### l3_recon_token_weight sweep step500（2026-06-07 ✅ 评完 REJECTED，P11 chunk512 底座 + L3 token-recon aux）
同口径 canonical scorer（`scripts/score_nested_babilong.py`，`compare_answers`），n=100，step500 ckpt：

```
arm                          qa  |   0k   1k   2k   4k   8k  16k  32k
P11 chunk512 baseline (无aux)★qa1 |   98   68   51   32   21   26   20
                             qa2 |   59   42   32   24   18   21   --
                             qa5 |   82   86   83   64   50   35   --
l3recontoken w1.0 ❌          qa1 |   77    4    6    8    3    2    1
                             qa2 |   43    4    5    3    1    2    3
                             qa5 |   67   22   16    8    3    1    0
l3recontoken w0.3 ❌         qa1 |   78   26   42   31   22   21   14
                             qa2 |   33    3   15   14   14    9   11
                             qa5 |   54   61   56   34   25   21   10
```
→ **w1.0 裁决：L3 token-recon aux 权重 1.0 = 灾难性。** 仅 0k 部分存活（qa5=67 vs baseline 82），≥1k 全面塌方（qa5 1k=22 vs 86、2k=16 vs 83、≥8k≈0）。强 token-recon aux 把 P11 baseline 原有的长程寻址彻底破坏。
→ **w0.3 裁决（2026-06-07 23:15 评完）：弱权重 token-recon aux 仍一致劣于无-aux baseline。** qa5 全长度低于 baseline（54<82 / 61<86 / 56<83 / 34<64 / 25<50 / 21<35 / 10<41），qa1/qa2 同样下移。弱 aux 破坏比 w1.0 温和（≥1k 未塌成 0，仍有 20-60 区间），但**没有任何长度优于 baseline**。
→ **★sweep 终裁：L3 token-level reconstruction aux 在 w0.3 与 w1.0 两个权重下均 REJECTED。token-level recon 目标与 routing/检索目标冲突——权重越大破坏越烈，弱权重也只是「破坏更小」而非「有益」。最佳配置仍是 P11 无-aux（delta-rule+normreadout）baseline。** 两 train run（.196 w0.3 / .249 w1.0）继续跑满 5000 仅留 lm/recon 曲线，BABILong 已定论。均为真实结果（CSV 满 n=100，无 silent-fail）。

### l3_recon w0.3 CONVERGED（step5000）eval（2026-06-08 05:xx，diskA .196，确认非翻案）
同口径 scorer，n=100，final adapter（step5000）：
```
arm                          qa  |   0k   1k   2k   4k   8k  16k  32k
l3recontoken w0.3 (step5000) qa1 |   80   27   43   15    3    1    2
                             qa2 |   50    2   25    7   11    2    4
                             qa5 |   50   59   45   20   19    9   13
```
→ 收敛点与 step500 同向（一致劣于无-aux baseline，长程崩塌），**确认 REJECTED 裁决在收敛点成立，不翻案**。（2026-06-08 15:xx 用 third_party/babilong-pkg scorer 补全此前 `--` 的 qa2-32k=4 / qa5-16k=9 / qa5-32k=13，结论不变：≥4k 全面塌方。）
→ **w1.0 converged eval（之前因 .249 silent-fail 0 CSV 缺失）：2026-06-08 15:23 起在空闲 B200 .188（L20A，env OK）补跑，bg agent general-purpose-1 进行中——纯粹补全 sweep 表格，裁决已终裁 REJECTED 不会翻案。**

### P11 chunk1024 deltarule_normreadout FINAL（step5000）eval（2026-06-08 04:37 起，本机 7-GPU，确认断崖在满训仍持续）
同口径 scorer，n=100，final adapter（step5000，1478min nf=0）：
```
arm                          qa  |   0k   1k   2k   4k   8k  16k  32k
P11 chunk1024 FINAL(step5000)qa1 |   56   56   15   15    7    5    0
                             qa2 |   37   31   11    5    5    1    2
                             qa5 |   29   68   29   15    7    4   --
```
→ **★裁决确认：chunk1024 的 1k 后断崖在满 5000 步训练后依然持续。** 对照 chunk512 baseline（qa5=82/86/83/64/50/35）：chunk1024 即便满训，qa5 2k=29 vs 512=83、8k=7 vs 50，长程几乎归零。**满训没有修复单 chunk1024 的稀释/注入太稀疏问题——chunk512 仍是决定性甜区，且渐进 warm-start（F1 v1）才是修单 chunk1024 断崖的正解。**（32k qa5 cell 收尾中，方向已定。）

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

### P11 chunk512 delta-rule normreadout — FINAL step5000 离线 BABILong（2026-06-08 20:50，B200 .188 L20A，n=100，babilong.metrics）

口径：qa1/qa2/qa5 × 0k-32k，acc%。

| len | qa1 | qa2 | qa5 |
|-----|-----|-----|-----|
| 0k  | 93  | 53  | 36  |
| 1k  | 44  | 21  | 47  |
| 2k  | 42  | 18  | 42  |
| 4k  | 24  | 22  | 23  |
| 8k  | 22  | 12  | 18  |
| 16k | 20  | 15  | 19  |
| 32k | 18  | 19  | 20  |

对照 P11 同配置 l3recon 变体（step5000 final，仅 qa1 部分跑全）：
- l3recontoken **w0.3**：qa1 0k-32k = 71/25/40/17/6/5/1（全面劣于 deltarule_normreadout）。
- l3recontoken **w1.0**：qa1 0k=4，1k+=0（**recon 权重 1.0 破坏检索，确认 recon 高权重有害**）。

**结论**：delta-rule + normalized readout（无 recon）= F1 最佳 final 配置，confirms P11 step500 裁决在 step5000 仍成立。recon aux 在此底座上有害（w1.0 灾难、w0.3 劣化），P12 recon 方向 confidence 进一步下调。

### F2 long-doc chunk1024 — FINAL step5000 离线 BABILong（2026-06-09 00:5x，.249 diskB H20，n=100，babilong.metrics）

F2 = 真实长文（wiki re-tokenize min4k，单样本多 chunk 压力测）+ F1 最佳底座（delta-rule normreadout chunk1024）训满 5000。口径 qa1/qa2/qa5 × 0k-32k，acc%。

| len | qa1 | qa2 | qa5 |
|-----|-----|-----|-----|
| 0k  | 90  | 40  | 32  |
| 1k  | 80  | 57  | 56  |
| 2k  | 39  | 24  | 64  |
| 4k  | 17  | 8   | 39  |
| 8k  | 15  | 2   | 29  |
| 16k | 12  | 3   | 16  |
| 32k | 17  | 10  | 12  |

对照 chunk1024 标准 dolmino-perdoc FINAL（qa5=29/68/29/15/7/4/--）：F2 长文训练后 qa5 在 2k-8k（64/39/29）明显优于标准 perdoc（29/15/7），qa1 0-1k（90/80）也更强 → **长文档训练数据对 mid-range（1k-8k）检索保持有正面作用**，但 ≥16k 仍随长度衰减。ckpt `outputs/f2_longdoc_chunk1024/mem_space_adapter.pt`。

### F2 long-doc chunk512 — FINAL step5000 离线 BABILong（2026-06-09 03:5x，.196 diskA H20，n=100，babilong.metrics）

F2 长文（wiki re-tokenize min4k）+ F1 最佳底座（delta-rule normreadout chunk512）训满 5000。口径 qa1/qa2/qa5 × 0k-32k，acc%。

| len | qa1 | qa2 | qa5 |
|-----|-----|-----|-----|
| 0k  | 74  | 67  | 45  |
| 1k  | 30  | 18  | 34  |
| 2k  | 22  | 15  | 32  |
| 4k  | 22  | 15  | 22  |
| 8k  | 16  | 8   | 18  |
| 16k | 13  | 5   | 20  |
| 32k | 15  | 14  | 20  |

⚠️ **OVERTRAINING-DEGRADATION 确认**：FINAL（qa5=45/34/32/22/18/20/20）显著弱于训练早期 step3000 估计（~84/64/61/54/46/35）——与 P11 chunk512 base 同样的「早期 ckpt >> final」满训退化模式。**已起 step3000 ckpt 离线 eval（.196 diskA zero-transfer，7 GPU）以 canonical 标定峰值。** ckpt `outputs/f2_longdoc_chunk512/mem_space_adapter.pt`（final），peak 候选 `..._step003000.pt`。

### P11 chunk512 step500（SOTA 峰值 ckpt）× cross-chunk SWA W0/W1/W2 离线 BABILong（2026-06-09 17:18-18:37，本机盘A 8×H20 全速 jobpool，n=100，babilong.metrics）

P11 deltarule_normreadout chunk512 的 **step500 早期峰值 ckpt**（`outputs/mem_space_p11_chunk512_deltarule_normreadout/mem_space_adapter_step000500.pt`）配 eval-only cross-chunk SWA（D2a fix，含短文档 W0 回退）。21 cell = W{0,1,2}×7 length，单节点同批跑（消除跨进程噪声）。调度脚本 `scripts/eval_p11_step500_swa_local_jobpool.sh`（LPT job-pool，cell-level resumable）。results `babilong_results/p11_step500_local_swa{0,1,2}`。acc%：

**qa5（主指标，长程检索）**

| len | W0 | W1 | W2 |
|-----|----|----|----|
| 0k  | 74 | 70 | 79 |
| 1k  | 89 | 85 | 89 |
| 2k  | 81 | 89 | 86 |
| 4k  | 60 | 75 | 88 |
| 8k  | 48 | 67 | 72 |
| 16k | 45 | 57 | 67 |
| 32k | 44 | 46 | 49 |

**qa1**

| len | W0 | W1 | W2 |
|-----|----|----|----|
| 0k  | 97 | 96 | 97 |
| 1k  | 67 | 61 | 64 |
| 2k  | 53 | 74 | 85 |
| 4k  | 37 | 46 | 59 |
| 8k  | 20 | 35 | 42 |
| 16k | 25 | 33 | 40 |
| 32k | 18 | 23 | 20 |

**qa2**

| len | W0 | W1 | W2 |
|-----|----|----|----|
| 0k  | 51 | 52 | 56 |
| 1k  | 42 | 39 | 40 |
| 2k  | 33 | 49 | 48 |
| 4k  | 24 | 34 | 46 |
| 8k  | 20 | 25 | 25 |
| 16k | 18 | 22 | 22 |
| 32k | 21 | 18 | 21 |

**结论（SOTA step500 配 SWA 的天花板）**：
- **SWA 单调放大长程增益，W2 是 qa5 全程最优**。qa5 4k-16k 上 W2 vs W0 = +28/+24/+22（88 vs 60、72 vs 48、67 vs 45），mid-range（2k-8k）增幅最大；32k 收敛到 49/46/44（仍 W2 略优但增益变小，超长靠少数 chunk 难补）。
- qa1/qa2 同向：W2 在 2k-16k 普遍领先 W0 ~+15~+22（如 qa1 2k 85 vs 53、4k 59 vs 37）。
- **对照 step5000+SWA W2（qa5=58/29/68/62/42/39/39）**：step500（峰值）+W2（79/89/86/88/72/67/49）在几乎所有长度全面碾压 step5000+W2，尤其 1k-16k（89/86/88/72/67 vs 29/68/62/42/39）→ **再次确认「早期峰值 ckpt >> 满训」+ SWA 叠加是当前最强组合**。
- **对照 step500 canonical W0（之前 qa5=82/86/83/64/50/46/41）**：本次本机 W0=74/89/81/60/48/45/44 与之同档（0k/4k/8k 小幅差异属跨批/进程方差）；短长度（0k/1k）有高方差，**仅供参考**。
- 长程（4k-32k）SWA 增益坐实：mid-range 显著，超长（32k）受限于可回看 chunk 数。
