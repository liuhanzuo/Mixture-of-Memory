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

## 3b. LongBench 结果 — P11 chunk512 step500 vs step5000（2026-06-10，EVAL-2，B200 GPU3/4）

**动机（用户 2026-06-09 insight）**：假说 = 单层 L1 slot 训久了从「NIAH 精确检索」转向「预训练式高级语义压缩」。**验证方法**：若 step5000 在 LongBench（需全局语义总结）上**不比 step500 差、甚至更好**，就坐实「L1 没变差、只是能力从检索挪到语义压缩」。

口径：base Meta-Llama-3-8B + P11 adapter，chunk_size=512，bf16 sdpa，--no_chat_template，SQuAD-F1，两 ckpt 完全同口径同 index 采样。hotpotqa n=200，其余 5 任务 n=100（narrativeqa 31k-token 全 200 太慢，capped；同 index → 严格可比）。`scripts/eval_longbench_mem_space.py`。

```
任务              类型           step500 F1   step5000 F1
hotpotqa         多跳针式检索      7.76         6.72
2wikimqa         多跳针式检索     13.65         9.36
musique          多跳针式检索      4.70         3.05
narrativeqa      全局故事理解      5.72         2.07   ← 假说预期 step5000 应更好
qasper           科学文章 QA       4.85         3.89   ← 假说预期 step5000 应更好
multifieldqa_en  阅读理解         16.53        11.25   ← 假说预期 step5000 应更好
AVERAGE                          8.87         6.06
```

**★裁决：假说被反驳（REFUTED）。** step5000 在**全部 6 个任务上一致劣于 step500**，包括三个「全局语义总结」型任务（narrativeqa 2.07 vs 5.72、qasper 3.89 vs 4.85、multifieldqa 11.25 vs 16.53）——这恰恰是假说预期 step5000 应追平/超过的地方，结果反而退化更狠（narrativeqa step5000 掉了 64%）。

**含义**：BABILong 上 step5000 ≪ step500（qa5 step500=82/86/83/64/50/46/41 vs step5000=54/62/51/30/28/22/31）不是「检索能力换成语义压缩能力」的此消彼长——LongBench 证明语义总结能力**也**退化了。**这是单调的过训退化（L1 整体被污染），不是能力迁移。** 早 ckpt（step500）是双口径（NIAH 检索 + 全局语义）下的统一最佳交付点。routing 诊断佐证：step500 topk_mass≈1.2-1.7、step5000≈0.8-1.0（更弥散）；step5000 generation 频繁不出 EOS 跑满 max_tokens（~2× wall time），是 LM 退化特征而非更强压缩。

→ **「过训退化」从单向假说升级为双向证据：检索 + 语义两条线在 step5000 同时劣于 step500。** 早停（step500）是 P11 的正确交付策略。

---

## 3c. 2026-06-10 批量 eval 归档（F2 long-doc / ladder / v3imp / D6 / D1 / D2b / L1-only）

> 口径：`scripts/run_babilong_mem_space.py`，n=100/cell，qa1/qa2/qa5 × 0k-32k，chunk512(各 run 用自己训练 chunk)，bf16 sdpa。**对照基准 = P11 chunk512 step500（当前最佳）**：qa5=74/89/81/60/48/45/44，qa1=97/67/53/37/20/25/18。
> 列顺序：0k/1k/2k/4k/8k/16k/32k。

### F2 long-doc（用 wiki ≥4k-token 长文档训练，底座 P11 chunk512/1024）
```
                  qa5: 0k  1k  2k  4k  8k  16k 32k
F2 c512  step500       43  75  60  50  38  44  33
F2 c512  step5000      86  39  34  27  21  10   8   ← 0k 飙升,长程崩(过训)
F2 c1024 step500       82  53  40  42  25  22  17
F2 c1024 step5000      23  34  48  43  31  17  20
```
裁决：**F2 长文档训练未超过 P11 base**（c512 step500 长程 16k=44/32k=33 与 P11 持平,但 1k-8k 段 75/60/38 全面低于 P11 的 89/60/48）。长文档数据没带来增益。step500≫step5000 铁律再现（F2 c512 step5000 长程崩到 10/8）。

### ladder（渐进 chunk 阶梯训练，top_k 固定）
```
                       qa5: 0k  1k  2k  4k  8k  16k 32k
ladder s1_c256 step500      53  30  19  27  31   8  26
ladder s1_c256 step5000     49  51  51  18  25  18  21   ← 唯一 step5000≥step500 的反例
ladder s2_c512 step500      54  73  27  30  32  19  35
```
注：ladder s1_c256 step5000 反常优于 step500（51/51 vs 30/19），可能 s1 step500 欠训；s2_c512 step500 的 1k=73 较好但整体仍输 P11。

### v3imp（v3 improved 渐进链，盘B）
```
                       qa5: 0k  1k  2k  4k  8k  16k 32k
v3imp s1_c256 step500       20  23   4  27  22  14  10
v3imp s1_c256 step5000      54  51  59  39  44  32  32   ← step5000 显著优(s1欠训同款)
v3imp s2_c512 step500       37  88  67  57  46  37  34   ← 除P11外最强长程
v3imp s2_c512 step5000      11  21  30  33  24  18  23
```
注：v3imp s2_c512 step500（88/67/57/46/37/34）是这批里最接近 P11 的，但 1k 后仍略逊。

### D6 读机制三臂消融（隔离 cross-attn 读 + null-sink，2026-06-11 补齐）
```
臂            机制                         qa5 step500: 0k  1k  2k  4k  8k  16k 32k
A=P11 base   xattn ON + null-sink ON       74  89  81  60  48  45  44
B=xattn_off  完全关 cross-attn             21  42  58  39  36  37  29
C=nullsink_off xattn ON + null-sink OFF    21  68  51  25  14  17  12
（C step5000 generation 崩溃：≥1k 全输出重复串 "the the and..."，0k 也弱 → 三臂用 step500 同口径对照；xattn_off step5000 同样崩 0/12/45/3/8/7/23）
```
**★裁决：排序 A(全开) ≫ B(无xattn) > C(有xattn无null-sink)。两个结论：**
1. **独立 cross-attn 读机制有用**（A≫B）：xattn_off 全面低于 P11（2k 58 vs 81，8k 36 vs 48），验证 P8/P11 引入独立读路径的价值。
2. **★null-sink 是必要稳定器，非可选小部件**（A vs C）：关掉 null-sink 后 qa5 每个长度大幅塌方（0k 74→21、4k 60→25、32k 44→12），且**长档（≥4k）C 甚至差于 B**（4k 25<39，32k 12<29）——「带 cross-attn 但抽掉 null-sink」比「根本不要 cross-attn」还糟。机制：null-sink 提供"本 chunk 无可检索内容时安全退出"出口；抽掉后 cross-attn 被迫每步硬从 slot 读 → 注入噪声污染 residual → 检索崩 + 最终 LM 训崩（C step5000 generation collapse）。读机制各部件耦合，不能拆用。

### D1 slot_dim 16384（+ lowrank_gate，双变量，B200）
```
            qa5: 0k  1k  2k  4k  8k  16k 32k      qa1: 0k  1k  2k  4k  8k  16k 32k
D1 step500       85  37  37  25  23  24  16            87  13  25  21  22  18  15
D1 step5000       1  54  61  22  37  28  25            0  26  40  33  27  24  21
```
注：⚠️ slot_dim16384 + lowrank_gate(r=256) **双变量**，无法单独归因。step5000 长程(1k-32k)明显优于 step500（罕见,可能 lowrank_gate 改变了过训动态），**但 step5000 在 0k 完全崩溃**（输出退化成全 `00000`，100/100 相同）。

### D2b 训练侧 SWA（训练时 --swa_train_chunks 2，双口径 eval）
```
                   qa5: 0k  1k  2k  4k  8k  16k 32k
D2b step500 W0          32  68  62  37  28  29  24
D2b step500 W2          46  55  75  62  53  37  30
D2b step5000 W0         26  84  68  48  25  24  20
D2b step5000 W2         25  85  81  84  59  49  35
对照 P11 base:
P11 step500 W0          74  89  81  60  48  45  44
P11 step500 W2          79  89  86  88  72  67  49
```
**★裁决：训练侧 SWA（D2b）REJECTED。** D2b 两个口径都低于对应 P11（D2b-W0 28@8k vs P11-W0 48；D2b-W2 53@8k vs P11-W2 72）。「训练时也见过 SWA」未带来正向迁移，反而损害基础 memory 学习（训练有 SWA 直连"拐杖"→ memory bank 学得更差）。**最佳仍是「普通训练 P11 + eval 时加 SWA」。** 注：D2b step5000 0k/1k 比 step500 高（同款过训→短上下文强）。

### L1-only（关掉 L3 summary，--no_l3_summary，B200，单变量）
```
              qa5: 0k  1k  2k  4k  8k  16k 32k      qa1: 0k  1k  2k  4k  8k  16k 32k
noL3 step500       75   4  13  10   2   5   2            94   5  10   5   3   6   4
noL3 step5000      70  57  51  28  15  10  10            95  20  28  20  13   9   3
对照 P11 base step500（带 L3）: qa5=74/89/81/60/48/45/44  qa1=97/67/53/37/20/25/18
```
**★裁决：L3 summary pool 是长程检索的顶梁柱，不能删。** 关掉 L3 后 step500 长程几乎全崩（qa5 1k 4 vs P11 89、8k 2 vs 48、2k 13 vs 81），仅 0k 存活（75≈74）。证明 L3 summary 通道承担了≥1k 的几乎全部检索能力，L1 memory slot 单独几乎不 work。**注：noL3 step5000 反而比 step500 好（qa5 1k 57 vs 4）——L3 缺失时 L1 需要更久训练才学会单独承担,但仍远不及有 L3。** 配合 gp-44 调研结论（L3≠搭便车）形成实测+机制双证据。

---

## 3d. ROUTE-A 路由均衡 sweep（2026-06-11，step500 + step1000，3 节点并行）

> 底座 = P11 chunk512 delta-rule+normreadout，total_steps=1000，各只改一个路由旋钮。口径 n=100，qa1/qa2/qa5 × 0k-32k，chunk512，bf16 sdpa。
> **对照基准 = P11 base step500**：qa5=74/89/81/60/48/45/44，qa1=97/67/53/37/20/25/18。
> 列顺序：0k/1k/2k/4k/8k/16k/32k。节点：arm1=本机盘A、arm2=.196、arm3=.76 盘B，全 8×H20。

```
arm                          qa  |   0k   1k   2k   4k   8k  16k  32k
arm1 lossfree0.01 step500    qa1 |   97   29   20   23   18   13   20
                             qa2 |   51   10    8    2   12   11   16
                             qa5 |   54   39   45   35   34   33   31
arm1 lossfree0.01 step1000   qa1 |   95   50   41   28   23   14   18
                             qa2 |   53   31   25   22   17   17   17
                             qa5 |   73   72   64   35   36   42   31
arm2 entropy0.01  step500    qa1 |   95    9    7    2    2    2    1
                             qa2 |   46    8    5    3    2    1    2
                             qa5 |   12   25   16    6    2    1    2
arm2 entropy0.01  step1000   qa1 |   95    9    1    1    0    0    0
                             qa2 |   36    7    9    2    1    2    1
                             qa5 |   43   35   19    6    4    3    1
arm3 temp20       step500    qa1 |   82   26    9    2    2    6    2
                             qa2 |   50   11   10    0    2    0    2
                             qa5 |   46   43   17    1    2    4    1
arm3 temp20       step1000   qa1 |   85   24   15    8    1    6    1
                             qa2 |   40   18    3    7    5    1    3
                             qa5 |   36   40   28   15   10    1    4
arm4 gumbel(t=1)  step500    qa1 |   93   43   35   24   15   24   14
                             qa2 |   42   14   16   19   15   13   11
                             qa5 |   57   58   62   32   25   17   11
arm4 gumbel(t=1)  step1000   qa1 |   71   13   23   21   15    7    8
                             qa2 |   39    7    8   13    9   10    9
                             qa5 |   55   44   51   32   23   17   13
```
(arm4 = use_st_gumbel_topk, st_gumbel_temperature=1.0, 训练 usage_cov→1.0；B200 .188 L20A，2026-06-11 评完)
→ **★arm4 裁决（2026-06-11）：REJECTED，长程垫底。** Gumbel 强制探索把 128 槽全用满（训练 usage_cov→1.0，前三臂均衡机制没做到），但**"槽用满"≠"长程变好"被证伪**：长程 qa5 8k/16k/32k=25/17/13(s500) 远低于 P11 base(48/45/44) 和 arm1 best(36/42/31)，是四臂里长程最差的。短程也被压崩（qa5 0k 55-57 vs base 74，qa2 全程≤42）。step1000<step500（qa1 0k 93→71）→ 续训放大破坏，非欠训。eval QUERY_DIAG 实测 usage_cov 仅~0.54-0.59、top1_sim_mean~0.11——强制均匀探索把检索 query 打散、命中"对的槽"概率下降，覆盖率高是负信号。
→ **★裁决：四个路由旋钮（loss_free/entropy/temp/gumbel）调整全部 REJECTED，均显著劣于 P11 base step500。** 没有任何 arm 在长程 cell（8k/16k/32k）改善：P11 base qa5 8k/16k/32k=48/45/44，最好的 arm1 step1000 仅 36/42/31，arm2/arm3 几乎全崩（≤4）。**关键诊断**：(1) 这批 run 只训了 1000 步（P11 base 是 5000-step run 的 step500 早 ckpt），步数/谱系不完全可比，但同为 1000 步内对照下 **arm1（loss_free 0.01）最稳、arm2（entropy_aux 0.01）最差**——加 entropy 正则反而把短上下文（qa5 0k 12→劣于 base 74）和长程全压崩。(2) **step1000 > step500**：arm1（qa5 0k 54→73、1k 39→72）和 arm3（qa5 2k 17→28、4k 1→15）续训均改善，arm2 0k 改善但长程持平/更崩。按"1k>500 才值得续训"约定，arm1/arm3 续训有正向但绝对值仍远低于 base。**结论：单纯调路由均衡旋钮（loss_free↑/entropy↑/temp↓）不能修复路由集中→长程检索退化，反而损害；arm1 方向（弱化 loss_free 提速）相对无害但无增益。下一步应回到 §4-9 的读侧坍缩 + delta-rule 写饱和根因，而非继续在路由 aux 上调参。**

---

## 4. 关键观察 & 待办

1. **chunk1024 全面 >> chunk128**（qa1_0k 89 vs 55，qa5_2k 93 vs 36）→ 缩小 chunk 大幅削弱每步局部窗口（SWA），长句掉分是预期。**chunk size 是当前最大杠杆**。
2. **slot_dim 4096→16384 对照缺失**：唯一的 16384 run 启动即崩、无 ckpt 无 eval。需修 wbmode 启动失败才能补这个对照。
3. **eval 无 cross-chunk SWA**：可能系统性低估真实能力（前文只能走 slots）。用户已指出 eval 至少应保留 SWA — 待加 eval 选项重测。
4. **slot 装 token 级 hidden 而非语义摘要** → BABILong（NIAH 式事实定位）相对行、LongEval（需全局总结）弱，符合预期。
5. **★过训是单调退化非能力迁移（2026-06-10 EVAL-2 坐实）**：P11 chunk512 step5000 在 LongBench 全 6 任务一致劣于 step500（AVG 6.06 vs 8.87），含三个全局语义任务——反驳「检索→语义压缩」迁移假说。step500 是 NIAH + 全局语义双口径统一最佳，早停是正确交付策略。详见 §3b。
6. **★L3 summary 是长程顶梁柱（2026-06-10 L1-only 坐实，§3c）**：关掉 L3 后 step500 长程几乎全崩（qa5 1k 4 vs 89），L1 memory slot 单独不 work。L3 不可删。
7. **★独立 cross-attn 读机制有用（2026-06-10 D6 §3c）**：xattn_off 全面低于 P11，验证 MemoryCrossAttentionRead 价值。
8. **★训练侧 SWA REJECTED（2026-06-10 D2b §3c）**：D2b 双口径均劣于 P11；最佳是「普通训练 + eval 时加 SWA」。
9. **★根因=路由集中（2026-06-10 gp-44 报告）**：usage_cov~0.25，128 槽只用~32 个，叠加 delta-rule 写饱和 + 读侧坍缩。修复方向 ROUTE-A sweep 进行中（loss_free/entropy/temp/Gumbel 四 arm）。加 num_slots 无用（128 都没用满）。
10. **新训练约定（2026-06-10 用户）**：训练默认先跑 1000 步，step500 vs step1000 评完，1k>500 才续训（鉴于过训退化铁律，避免盲跑 5000 浪费算力）。
5. **★过训是单调退化非能力迁移（2026-06-10 EVAL-2 坐实）**：P11 chunk512 step5000 在 LongBench 全 6 任务一致劣于 step500（AVG 6.06 vs 8.87），含三个全局语义任务——反驳「检索→语义压缩」迁移假说。step500 是 NIAH + 全局语义双口径统一最佳，早停是正确交付策略。详见 §3b。

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

---

## L3 Diversity Sweep (2026-06-11) — 路由证伪后转 L3 方向的核心验证

**动机**：路由证伪后转 L3 summary 方向。诊断发现 L3 summary token 严重坍缩（l3_tok_cos≈0.99）。假设：token 坍缩是长程差的瓶颈，diversity 正则是「治本杠杆」。本 sweep 验证「治坍缩」能否转化为 BABILong 长程改善。

**共享配置**（与 P11 base 一致）：chunk512, num_slots=128, top_k=16, selector_dim=128, num_global_slots=4, delta_rule writeback, normalize_readout, dual_gate, use_memory_xattn(gate_init=0.4), use_l3_summary(l3_n_summary=64, n_layers=2, n_heads=8), shared_memory_bank, route_aux_weight=1.0, lr=1e-4, total_steps=1000, save_interval=500。唯一变量：`--l3_diversity_weight` + `--l3_diversity_threshold=0.5`。

| run | l3_div_weight | l3_tok_cos末值 | 节点 | 状态 |
|-----|---------------|---------------|------|------|
| EXP-1 | 0.1 | ~0.40 | 盘A (.196训, .196评) | done |
| EXP-3 | 0.3 | ~0.18 | 盘A (本机训+评) | done |
| EXP-5 | 0.5 | — | 盘B (.76) | 训练中(step125) |

**BABILong eval（n=100, qa1/qa2/qa5 × 0k-32k, chunk512, babilong.metrics）**

### qa5（最关键，step500 = 各 run 可用峰值点）
```
                  l3_tok_cos   0k  1k  2k  4k  8k 16k 32k
P11 base(无div)      ~0.99     74  89  81  60  48  45  44   ← 基线/SOTA
EXP-1 (div=0.1)      ~0.40     77  81  53  36  31  32  32
EXP-3 (div=0.3)      ~0.18     79  57  50  24  28  40  20*
EXP-4 (减容量,对照)   —         76  28  12  16  10  10   8
```
（* EXP-3 32k 及 EXP-1 32k 个别 cell 评分时 n<100，补跑到 n=100 中；长程整体已显著低于 base，结论不受影响。）

### EXP-1 (div=0.1) 完整网格
step500: qa1=95/38/22/11/19/16/6, qa2=53/26/13/11/8/22/9, qa5=77/81/53/36/31/32/32
step1000(退化): qa1=25/23/15/11/3/9/8, qa2=53/15/6/4/0/6/1, qa5=81/69/45/20/14/18/0

### EXP-3 (div=0.3) 完整网格
step500: qa1=95/20/27/18/17/15/16, qa2=36/16/16/11/7/11/16, qa5=79/57/50/24/28/40/20
step1000(退化): qa1=92/10/11/14/4/3/1, qa2=42/8/7/6/4/5/1, qa5=53/32/32/16/13/18/10

**结论（假设证伪）**：
- **判据②「治坍缩」成立**：diversity 按权重单调把 l3_tok_cos 0.99→0.40(w=0.1)→0.18(w=0.3)。
- **判据①「治坍缩→长程改善」证伪，且反向剂量关系**：两 arm 长程(8k-32k)全面低于 P11 base；**w 越大坍缩压越狠，BABILong 反而越差**（qa5 中段 2k/4k：base 81/60 → w=0.1 53/36 → w=0.3 50/24）。0k 都≈base(74)，越长越崩。
- 两 arm step1000 都过拟合退化，step500 才是可用点。
- **核心 takeaway**：L3 summary token 坍缩(cos≈0.99) **不是** BABILong 长程差的因果瓶颈——治好它(cos→0.18)非但没救长程，反因 diversity 正则与 LM 目标争夺 summary 容量/扰乱表征而轻微伤害长程。"治坍缩是治本杠杆"在 BABILong 上**不成立**。L3-diversity 方向不优于 P11 base，应回到 P11 base 或换思路（让 summary 真正承载可检索 KV，而非仅追求 token 互异）。

### 写入侧臂 BABILong 结果（2026-06-11，n=100，qa5，对比 P11 base=74/89/81/60/48/45/44）
| run | 配置 | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|------|----|----|----|----|----|----|----|------|
| W1b top_k48 step500 | 写入top_k 16→48 | 44 | 56 | 39 | 20 | 16 | 13 | 14 | 证伪(全面崩) |
| W1 top_k32 step500 | 写入top_k 16→32 | 76 | 67 | 62 | 38 | 24 | 19 | 17 | 欠训 |
| **W1 top_k32 step1000** | 写入top_k 16→32 | **79** | 76 | 74 | 50 | 42 | 42 | 41 | 0k超base,长程仍低~6pt,未收敛 |
- W1 routing 破死锁：32k top1_sim 0.076(vs 死锁0.0022) usage_cov 0.80。但加宽 top_k 单旋钮不足补长程。W1(32)≫W1b(48)：适度加宽有益，过宽有害。
| R1 deadslot_r8 step500 | 死槽回收 reset_interval8 | 65 | 40 | 17 | 17 | 14 | 20 | 13 | 长程崩 |
| **R1 deadslot_r8 step1000** | 死槽回收 reset_interval8 | 77 | 45 | 54 | 26 | 14 | 17 | 14 | 破死锁但长程崩塌(<base 3x) |
- **R1 关键反例**：训练 dead_slot_frac0.834 usage_cov0.703(vs P11 0.25)→机制上破死锁成功,但 qa5 长程 8k-32k=14/17/14 vs base 48/45/44 崩塌3倍。eval recycle_resets=7 forward确触发。**结论:覆盖率(usage_cov)不是瓶颈,强制回收死槽反而打乱已学稳定记忆→长程退化。**
- **写入侧横向裁决(R1/W1/W1b三臂已出,W2待)**：所有"强制提高slot参与度"的操作(R1回收/W1b过宽top_k)都让长程退化;唯W1(适度top_k32)逼近base但仍低6pt。**强信号:usage_cov↑≠长程↑,与ROUTE-A arm4教训一致(强制均摊伤长程)。** 待W2(dense soft write,最温和)定生死。
| W2 softwrite λ=0.05 step1000 | dense all-slot soft-write λ0.05 | 60 | 11 | 0 | 0 | 0 | 0 | 0 | 证伪(λ太强,≥2k输出空"-") |
- W2 λ=0.05 slot_content_cos=0.45(非完全坍缩)但≥1k生成崩(output="-"):dense soft-write λ0.05漂移slot过强,破坏长档生成。λ=0.02(更温和)主臂在.76跑,是W2真正判据。
| W2 softwrite λ=0.02 step500 | dense all-slot soft-write λ0.02 | 42 | 19 | 5 | 0 | 1 | 1 | 0 | 证伪 |
| W2 softwrite λ=0.02 step1000 | dense all-slot soft-write λ0.02 | 56 | 29 | 1 | 0 | 0 | 0 | 0 | 证伪(长档≈0,比W1b更差) |
- **写入侧四方案全证伪(R1/W1/W1b/W2)**：横向 W2(λ0.02,≥2k≈0) < W2(λ0.05) < W1b(topk48) < R1(回收) < W1(topk32,唯一逼近base但长程仍-6pt) < P11 base。所有"强制更多slot参与写入/回收死槽"都伤长程,usage_cov↑≠长程↑铁律再次坐实。唯一待验:R1c(采样级死判据,只回收真·从未用过的槽,不动长程静默槽)。

### ★ R1c 采样级死判据（2026-06-11，写入侧首个相对 R1 显著改善，qa5）
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|----|----|----|----|----|----|----|------|
| R1 window step1000 | 77 | 45 | 54 | 26 | 14 | 17 | 14 | 长程崩 |
| **R1c cumulative step1000** | 84 | 46 | 63 | 41 | **32** | **27** | **29** | 长程大幅回升(R1→base缺口的1/2~2/3) |
| P11 base | 74 | 89 | 81 | 60 | 48 | 45 | 44 | SOTA |
- **关键洞察验证（用户提出）**：R1 的窗口级死判据(_recycle_usage==0,近8chunk没选)误把长程记忆静默槽当死槽回收抹除→长程崩。改成采样级(_cum_usage==0,整条样本从未选中)后，qa5 8k/16k/32k=32/27/29 vs R1 的14/17/14，回升+18/+10/+15。证实"回收动作本身不有害,有害的是窗口级判死太激进"。dead诊断:32k ingestion n_recycled76-95(真触发),0k n_recycled=0(不动短档)。
- 仍低于base(8k 32 vs 48):采样级回收缓解但未完全消除长档扰动;qa1单supporting fact长档仍偏低(对回收时机更敏感)。**写入侧方向因R1c复活,值得继续(reset_interval/grace sweep)。**

### top_k 甜区 + R reset_interval sweep（2026-06-12，qa5 step1000，vs base 74/89/81/60/48/45/44）
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|----|----|----|----|----|----|----|------|
| W1d topk20 | (待评) | | | | | | | |
| W1c topk24 | 71 | 57 | 47 | 23 | 13 | 16 | 7 | 劣于topk32(中长程断崖) |
| W1 topk32 | 79 | 76 | 74 | 50 | 42 | 42 | 41 | topk甜区最优,长程仍-6pt |
| W1b topk48 | 78 | 30 | 30 | 19 | 21 | 15 | 12 | 过宽证伪 |
| R1b 窗口级reset16 | 74 | 34 | 37 | 20 | 16 | 14 | 11 | 长程仍崩(放宽interval无效) |
| R1 窗口级reset8 | 77 | 45 | 54 | 26 | 14 | 17 | 14 | 长程崩 |
| **R1c 采样级reset8** | 84 | 46 | 63 | 41 | 32 | 27 | 29 | ★最佳,长程大幅回升 |
- **两条线裁决坐实**:(1)top_k甜区=32(20/24<32<48两侧都差,32唯一逼近base);(2)R线必须改判据—R1b(窗口级+reset16)长程仍崩(2k37/4k20 vs R1c 63/41),"放宽interval不能替代改判据,采样级R1c才是关键"。R1c interval sweep(4/8/16)进行中验证最优回收频率。

### R1c reset_interval sweep（采样级判据 cumulative，qa5 step1000，vs base/R1c-int8）
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|----|----|----|----|----|----|----|------|
| R1c4 interval4 | 54 | 64 | 57 | 26 | 16 | 18 | 9 | 回收太频,长程劣于int8 |
| **R1c interval8** | 84 | 46 | 63 | 41 | 32 | 27 | 29 | ★sweep最优 |
| R1c16 interval16 | (训练中) | | | | | | | |
- R1c4(int4,更频回收)长程8k=16/32k=9 显著劣于 R1c(int8)的32/29 → **回收过频也伤长程,interval8优于int4**。配合R1b(窗口级)结论:回收要"判据对(采样级)+频率适中(int8)"。待R1c16补完曲线看是否int8就是峰值。

### W1 长训轨迹（top_k32 续训到3000，qa5，vs base 74/89/81/60/48/45/44）
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|----|----|----|----|----|----|----|------|
| W1 step1000 | 79 | 76 | 74 | 50 | 42 | 42 | 41 | 长程最佳点 |
| W1cont step2000 | 54 | 82 | 78 | 45 | 33 | 33 | 34 | 长程已退 |
| W1cont step2500 | 41 | 82 | 75 | 44 | 34 | 31 | 32 | 退 |
| W1cont step3000 | 24 | 86 | 70 | 47 | 36 | 28 | 33 | 0k崩塌,8k-32k不升反降 |
- **长训假设证伪**:W1 top_k32 续训到3000步,8k-32k qa5从step1000的42/42/41反降到36/28/33,0k从79崩到24(过拟合)。"更长训练补长程"不成立→**写入侧长程瓶颈不是训练步数,step1000附近就是该配置上限**。结合过训单调退化铁律,后续写入侧实验默认step500-1000即可,不必长训。

### R1c grace/interval/长训 补充 sweep（qa5 step1000除注明，vs R1c int8基准 84/46/63/41/32/27/29）
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|----|----|----|----|----|----|----|------|
| R1c interval8 grace1 (基准) | 84 | 46 | 63 | 41 | 32 | 27 | 29 | ★最佳 |
| R1c16 interval16 | -- | 57 | 47 | 33 | 15 | 17 | 21 | 长程劣于int8(8k15<32) |
| R1c_cont3k step3000(长训) | 34 | 37 | 43 | 29 | 23 | 21 | 18 | 长训退化(8k23<32),同W1长训证伪 |
- R1c16(int16)长程8k=15劣于R1c(int8)32 → 配合R1c4(int4,8k=16)坐实**interval8是回收频率甜区**(4/16两侧都差)。R1c长训(cont3k)8k-32k=23/21/18<step1000的32/27/29 → **采样级回收同样过训退化,step1000即上限**。R1c16的0k列CSV解析--(qa1/qa5 0k cell无有效行,可能错峰/分片残留),不影响长程裁决。

### R1c grace sweep（采样级 cumulative，qa5 step1000，vs R1c grace1基准 84/46/63/41/32/27/29）
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|----|----|----|----|----|----|----|------|
| R1c grace1 (基准) | 84 | 46 | 63 | 41 | 32 | 27 | 29 | ★最佳 |
| R1cg3 grace3 | 49 | 55 | 62 | 44 | 29 | 24 | 27 | 0k退化(84→49),长程略低,未破基线 |
- grace3(更长强制写窗口)0k从84崩到49,8k-32k 29/24/27≤grace1 32/27/29→**grace1最优,加长grace有害**(强制写窗口越长越扰动已学内容)。配合interval8甜区:R1c最优=cumulative+int8+grace1+strided。

### R1cW1 组合（采样级回收 + top_k32，两正向杠杆叠加）qa5 vs R1c grace1基准 84/46/63/41/32/27/29
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|----|----|----|----|----|----|----|------|
| R1cW1 step500 | 74 | 30 | 27 | 21 | 15 | 15 | 14 | |
| R1cW1 step1000 | 65 | 49 | 38 | 22 | 13 | 11 | 8 | 证伪:叠杠杆负向,8k-32k远低于基准32/27/29 |
- **关键负结果**:R1c(采样回收)+W1(top_k32)两个单独正向的杠杆叠加→8k-32k=13/11/8 ≪ R1c grace1基准 32/27/29,反而严重退化。两杠杆互相干扰(top_k32加宽写入 + 采样回收 双重扰动slot内容→长档检索崩)。step500(15/15/14)>step1000(13/11/8)又一次过训退化。
- **写入侧最终结论**:最优=R1c单一(cumulative+int8+grace1+strided+topk16)=84/46/63/41/32/27/29,任何叠加(W1 topk/grace3/interval4or16/长训/组合)都未破,多数负向。距P11 base(48/45/44)长程仍差~13-16pt。写入侧sweep到此收敛。

### R1cz reset_mode=zero（采样级，死槽重置成0而非strided token，qa5 vs strided基准 84/46/63/41/32/27/29）
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|----|----|----|----|----|----|----|------|
| R1cz reset_zero step1000 | 61 | 54 | 49 | 26 | 21 | 31 | 17 | 混合:16k=31略超基准27,但8k/32k=21/17<32/29,0k 61<84,整体≤strided |
- reset成0 vs strided_current_token:整体略劣(0k 61<84,8k 21<32),仅16k 31>27偶发。strided(差异化当前token)仍优于zero。R1c最优配方不变=cum+int8+grace1+strided。

### R1c grace sweep 完整（采样级，qa5 step1000，grace1 基准 84/46/63/41/32/27/29）
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|----|----|----|----|----|----|----|------|
| R1c grace1 (基准★) | 84 | 46 | 63 | 41 | 32 | 27 | 29 | 最佳 |
| R1cg2 grace2 step1000 | 25 | 49 | 20 | 14 | 20 | 13 | 17 | 退化(8k20<32,0k25崩) |
| R1cg3 grace3 | 49 | 55 | 62 | 44 | 29 | 24 | 27 | 略退 |
- **grace sweep 完整确认 grace1 最优**:grace2 step1000 长程20/13/17更差(注:grace2 step500 8k=39异常高于step1000,过训退化明显)。grace 1<2<3 单调:强制写窗口越长越伤,grace1(最短)最优。

### ★★ 容量 sweep（num_slots，采样级R1c cum+int8+grace1，qa5 step1000，vs N128基准84/46/63/41/32/27/29 + P11 base 74/89/81/60/48/45/44）
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|----|----|----|----|----|----|----|------|
| N128 (R1c基准) | 84 | 46 | 63 | 41 | 32 | 27 | 29 | 写入侧旧最佳 |
| N256 | 40 | 30 | 42 | 38 | 30 | 26 | 22 | 异常低(0k40),疑欠训/坏ckpt,待复核 |
| **N384** | 85 | 68 | 74 | 53 | 39 | 42 | 30 | ★★全面超基准!16k 27→42,4k 41→53,向base收敛 |
| N512 | (评中) | | | | | | | |
- **★突破**:N384(扩slot 128→384)首次全面超越R1c基准,长程16k=42(base45!)/4k=53(base60)/2k=74(base81),多档逼近P11 base。**扩memory容量是写入侧第一个有效正向杠杆**——更多slot给采样级回收更大空间,死锁缓解+长程检索改善。N256反常低需复核(可能训练异常)。待N512看是否继续提升或见顶。**这是写入侧重大进展,值得深挖(N384续训/N768/与top_k联调)。**

### 容量 sweep 更新（qa5 step1000）
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|-----|----|----|----|----|----|----|----|
| N128 | 84 | 46 | 63 | 41 | 32 | 27 | 29 |
| N384 ★ | 85 | 68 | 74 | 53 | 39 | 42 | 30 |
| N512 | 59 | 36 | 11 | 30 | 24 | 14 | 14 |
- 容量非单调:N384最佳,N512(24/14/14)反而劣于N128基准。疑N512/N256训练异常(N512用bs2,N256反常低也是)→**N384是当前峰值,但N256/N512的反常需复核(可能bs2或大slot训练不稳)**。N256重训中,N768待评。若N768也低→N384是真甜区;若N256复核后正常→之前是坏run。

### R1c interval sweep 更新 + 变异性警示（qa5 step1000）
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|-----|----|----|----|----|----|----|----|
| R1c int8 (基准) | 84 | 46 | 63 | 41 | 32 | 27 | 29 |
| R1c int12 (新) | 74 | 70 | 60 | 47 | 42 | 43 | 31 |
| R1c int16 | 21 | 55 | 47 | 33 | 15 | 17 | 21 |
| R1c int4 | 54 | 64 | 57 | 26 | 16 | 18 | 9 |
- **⚠️变异性问题浮现**:int12 长程8k/16k=42/43 明显>int8基准32/27,但之前判定int8是峰值(int4/int16都差)。int12>int8 与 int16<int8 矛盾→**这些single-run分数有高run-to-run方差**(同N384>N128但N256/N512<N128的非单调也是同一现象)。**关键反思:之前所有"sweep最优/证伪"的单run裁决可能被噪声污染,N384突破和int12好分都可能部分是方差**。需要:对候选好配置(int12/N384)做2-3次重复跑确认,而非单run定论。

### 容量 sweep 全貌（qa5 step1000）— 非单调坐实方差污染
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|-----|----|----|----|----|----|----|----|
| N128 | 84 | 46 | 63 | 41 | 32 | 27 | 29 |
| N384 | 85 | 68 | 74 | 53 | 39 | 42 | 30 |
| N512 | 59 | 36 | 11 | 30 | 24 | 14 | 14 |
| N768 | 65 | 51 | 52 | 35 | 23 | 18 | 8 |
- **容量曲线非单调(N384峰,N512/N768崩)物理不合理→坐实single-run高方差**。N384"突破"很可能部分是运气。**正在做方差归因:B200 re-eval N384同一ckpt(隔离eval方差) + .196起N384c训练复刻(隔离train方差)**。结论待定:若N384b/N384c回到N128水平→N384是噪声;若稳定高→容量真有效但需多seed确认。**方法论修正:后续候选配置必须2-3 seed重复,单run分数只作筛选不作定论。**

#### ★★多seed方差裁决（2026-06-12 23:xx 评完，qa5 step500，canonical 口径）= 容量 sweep 是噪声，N384"突破"证伪
| run (seed) | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 备注 |
|-----|----|----|----|----|----|----|----|----|
| N384 orig | 85 | 68 | 74 | 53 | 39 | 42 | 30 | 原"突破"（step1000 口径，下方对照取 step500）|
| N384c (seed1234) | 76 | 23 | 37 | 29 | 25 | 13 | 9 | ★长程**崩**(16k=13/32k=9)，完全不复刻 orig |
| N384d (seed2026) | **0** | 81 | 81 | 51 | 40 | 35 | 29 | ★0k**崩到0**，中长档尚可但形态全异 |
| N192 (orig) | 90 | 37 | 43 | 30 | 21 | 21 | 15 | qa1 行；qa5=54/58/73/40/23/21/13 |
| N192b (seed1234) | 87 | 27 | 41 | 25 | 23 | 20 | 12 | qa1 行；qa5=22/65/55/36/25/27/20 |
- **★裁决：N384/N192 容量"突破"= single-run 高方差噪声，不复刻。** 三个 N384 seed 形态彼此完全不同（orig 全档高 / seed1234 长程崩 / seed2026 0k 崩到 0），无任何一个 seed 稳定超过 P11 base step500(74/89/81/60/48/45/44)。N192 两 seed 同样发散（0k 90 vs 22）。**扩 memory 容量(128→384)不是有效杠杆——之前的"N384 全面超基准"是运气，被多 seed 证伪。** 坐实方法论修正：单 run 分数只作筛选不作定论；写入侧（top_k/容量/写规则/dense）全谱系已穷尽证伪。N192c(seed777) eval 跑完即补全三 seed×两容量完整方差表，但结论已定。

### 容量 sweep + N192（qa5 step1000）— 中长档收益 vs 短档回退
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|-----|----|----|----|----|----|----|----|
| N128 | 84 | 46 | 63 | 41 | 32 | 27 | 29 |
| N192 | 72 | 33 | 69 | 51 | 39 | 45 | 27 |
| N384 | 85 | 68 | 74 | 53 | 39 | 42 | 30 |
| N512 | 59 | 36 | 11 | 30 | 24 | 14 | 14 |
| N768 | 65 | 51 | 52 | 35 | 23 | 18 | 8 |
- **N192/N384 中长档(4k-16k)一致超N128**(N192 4k/8k/16k=51/39/45, N384=53/39/42 vs N128 41/32/27)→**128→256区间扩容量对中长程检索有真实单调收益**(两个独立点N192/N384都显示,非孤例)。但短档(0k/1k)非单调回退,且N512/N768崩(疑bs2大slot训练不稳/欠拟合)。**初步信号:容量提升在128-384区间对中长程有效,但≥512训练不稳定+短档代价**。仍需N384b/c复刻确认幅度。N192 16k=45追平base!

### ★ DENSE-write (top_k=N 全局写) 容量矩阵裁决（2026-06-12，qa5）
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|----|----|----|----|----|----|----|------|
| **DENSE N128 topk=128 step500** | 0 | — | 0 | 0(4k) | 0 | 0 | 0 | ★证伪(生成崩) |
| **DENSE N128 topk=128 step1000** | 0 | — | 0 | 0(4k) | 0 | 0 | 0 | ★证伪(生成崩) |
| P11 base | 74 | 89 | 81 | 60 | 48 | 45 | 44 | SOTA |
- **★ top_k=N(全局软写,消灭"永不更新的槽")彻底证伪**：训练健康(lm正常 nf=0 1000步完成,全槽live dead_slot_frac=0.0 usage_cov=1.0),但离线 BABILong qa5 **每个长度(含2k)=0%**,生成坍缩成"ion ion ion ion"重复。同一 B200 harness 对 N128 DRoff 给出真实分(33/40/32...)→排除 harness bug,是**真·记忆污染**(印证 CODEBUDDY「PPL>100=语言模型被污染」洞察:每chunk每槽都写→slot 漂移过强→破坏冻结 backbone 正常输出)。
- **与既有 W2(dense soft-write λ0.02/0.05)裁决一致并更强**:W2 是部分崩(短档还有分),top_k=N 是全崩(连2k=0)。**写入侧最终裁决:从"稀疏 top_k 写"到"全局 dense 写"全谱系——越逼迫所有槽参与写入,长程/生成越退化;唯适度 top_k(W1=32)+采样级回收(R1c)逼近 base。usage_cov↑≠长程↑ 铁律第N次坐实。** N192/N256/N384/N512 dense 同理大概率证伪(在跑只为补曲线,不抱期望)。

### ★★ pg19 真长文蒸馏：攻长程天花板（2026-06-16，qa5 W0）
配置：冻结 Llama-3-8B + 128 slot，self-study 蒸馏（teacher=开卷全上下文，KL λ0.6 + hidden-cos layers12,20,28），训练数据从 dolmino(全≤16k,锁n_ctx=3) 换成 **PG19 真长书(78%≥32k token)**，chunk512 / **n_ctx=7(有效训练窗口=(7+1)*512=4096 token)** / 不加 mass / 500步 / seed42。teacher 缓存 distill_cache/pg19_512_nctx7(26444 npz)。
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 裁决 |
|-----|----|----|----|----|----|----|----|------|
| **pg19 distill final(step500)** | 75 | 73 | 51 | 29 | 19 | **16** | 9 | ★16k破天花板+3 |
| **pg19 distill step250** | 78 | 74 | 50 | 29 | 17 | **16** | 8 | 同上,250步已收敛 |
| 现有方法天花板(mass/蒸馏/课程/叠加) | — | — | — | — | ~15 | **13** | **9** | 所有方法都卡这 |
- **★核心结论**：真长文训练在 **16k 档稳定突破长程天花板**（final+step250 都=16 vs 所有现有方法 13，+3）→ **验证"训练期见过真长文能破长程墙"假设**。这是相对 mass/蒸馏/课程/叠加（全卡 16k≈13）的明确进步。
- **32k 仍硬墙**：final=9/step250=8，未超天花板 9。推断根因：n_ctx=7 有效训练窗口 4096 token，32k 推理超训练分布 8 倍。
- **250 步收敛**：step250≈final，长程无额外增益，可省半训练。
- **下一步**：n_ctx=15（窗口 8192，翻倍）攻 32k，缓存在 .249 构建中（diskB 154T free）。

### ★★ n_ctx=15 攻 32k：加大训练窗口的负结果（2026-06-17，qa5 W0）
动机：pg19 n_ctx=7（窗口 (7+1)*512=4096）在 16k 破天花板（=16）但 32k 仍硬墙（=9）。假设：加大 n_ctx 让训练窗口更接近 32k 推理能破 32k。做法：n_ctx=15（窗口 (15+1)*512=8192，翻倍），其余同 pg19 蒸馏配方（chunk512/全量缓存33064 npz/distill λ0.6 layers12,20,28/500步），seed42。
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|-----|----|----|----|----|----|----|----|
| pg19 n_ctx7 final（对照） | 75 | 73 | 51 | 29 | 19 | **16** | **9** |
| pg19 n_ctx15 step250 | 80 | 76 | 56 | 30 | 19 | 14 | 8 |
| pg19 n_ctx15 final | 77 | 73 | 55 | 31 | 17 | **15** | **8** |
- **★负结果**：加大训练窗口 4096→8192 **未突破 32k**（8 ≤ n_ctx7 的 9），16k 反略降（15 < 16）。step250≈final（负结果稳健，非噪声）。
- **结论**：32k 硬墙不是"训练窗口不够大"能解。更可能是**压缩比根本瓶颈**（32k token → 128 slot ≈ 250:1），或 n_ctx15 每样本上下文更长但有效梯度信号更稀疏致中程略退。**16k 是 pg19 真长文蒸馏的甜点（n_ctx7 =16），再加窗口边际为负。**
- 排除"加窗口"路线。下一步攻 32k 应转向：增大 slot 数（降压缩比）/ 分层记忆 / 而非单纯训练长度。

**n_ctx15 跨 seed 确认（2026-06-17）**：seed1234 final qa5=79/71/59/31/18/15/**10**。两 seed 32k=8(s42)/10(s1234) 均值≈9 = n_ctx7 的 9（噪声内持平，未突破）；16k 两 seed 都=15 ≤ n_ctx7 的 16。**负结果跨 seed 稳健坐实**：加窗口对 32k 无效、16k 边际为负。32k 硬墙确认为压缩比瓶颈（非训练长度）。

### ★★ N256 降压缩比攻 32k：第二个负结果（2026-06-17，qa5 W0）
动机：n_ctx15 加训练窗口未破 32k，定位 32k=压缩比瓶颈（32k→128 slot=250:1）。假设：增大 slot 降压缩比能破 32k。做法：pg19 n_ctx7 配方（16k 破墙那个）+ num_slots 128→256（压缩比 125:1），top_k 仍 16，复用 pg19_512_nctx7 缓存（teacher 与 slot 数无关），500步 seed42。
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|-----|----|----|----|----|----|----|----|
| pg19 n_ctx7 N128（基线）| 75 | 73 | 51 | 29 | 19 | **16** | **9** |
| N256 step250 | 79 | 69 | 54 | 29 | 15 | 16 | 9 |
| N256 final | 77 | 71 | 58 | 30 | 17 | **16** | **9** |
- **★负结果**：降压缩比 250:1→125:1（slot 翻倍）**对 32k 无效**（仍 9），16k=16 持平 N128。step250≈final（稳健）。
- **结合 n_ctx15 负结果 → 32k 硬墙双重证伪**：既非训练窗口（n_ctx15）、也非 2x 压缩比（N256）能解。32k 比预想顽固。
- **推断**：32k 需更深层改动（分层记忆/检索增强），或已超 128-256 slot 在此架构的信息论容量。slot 数/训练长度的线性外推到此为止。
- **16k=16 是 pg19 真长文蒸馏的稳健甜点**（N128/N256、step250/final 都 16）。

**pg19 LongBench 补点（2026-06-17）**：pg19 n_ctx7 final LongBench AVG=**6.5**（hotpotqa6.4/narrativeqa2.5/qasper4.9/multifieldqa12.7/2wikimqa8.5/musique3.9）。对照: mass_coef1=6.56、弱mass+蒸馏(s2026)=10.4。
- **★关键发现：BABILong 长程突破 ≠ 真实长文档能力**。pg19 真长文蒸馏在 BABILong 16k 破墙（+3），但 LongBench 仅 6.5（≈mass_coef1，远低于弱mass+蒸馏的 10.4）。pg19 的 16k 突破是 **BABILong 合成事实链检索任务特定**，未迁移到真实长文档理解。真实长文档上**弱mass+蒸馏仍是最优**。两 benchmark 测不同能力，优化目标需分清。

### ★★ LongEval lines-retrieval（2026-06-17，W0 纯检索探针）
脚本 scripts/eval_longeval_mem_space.py（自写最小生成器，格式忠于 LongChat，口径=BABILong W0：chunk512/swa0 闭卷纯记忆读出）。ckpt=P11 chunk512 deltarule_normreadout step500（128 slot），n=50/档。
| 长度 | 1k | 2k | 4k | 8k | 16k | 32k |
|------|----|----|----|----|----|----|
| accuracy(精确6位数) | 8% | 14% | 10% | **0%** | **0%** | **0%** |
- **★断崖式失效：≥8k 精确单条检索直接归零**（非渐进衰减）。比 BABILong 更干净的纯检索探针，强烈印证 32k 长程墙。
- 结合 slot 分布诊断（写入健康 0 dead 用满 128）→ **32k 硬墙 = 读出端精确检索失效，非写入/容量/路由问题**。
- ★新线索：8k 突变（16 chunk）暗示开关性失效（readout query 在 chunk 数超阈后分辨不出目标 slot，或 L3 summary 聚合糊化），非压缩比连续退化。下一步攻 readout 机制。

### ★★ 对话记忆 benchmark（2026-06-17 实测落地，mem_space P11 SOTA W0 vs Llama-3-8B base 开卷）
脚本 scripts/eval_dialogmem_mem_space.py（chunk=1024，节点 .196 diskA 共享FS）。判分 F1+子串近似+拒答检测（无 LLM-judge，时序类相对日期低估绝对分，但 mem-vs-base 相对差距可信）。**LongMemEval mem 已 n=500 全 6 题型与 base 完全对齐 rerun（2026-06-17，8-shard）；此前 n=100/2-题型 的不可比数字已作废。**
| benchmark | mem acc/F1 | base acc/F1 | 差距 | 关键子任务对比 |
|-----------|-----------|------------|------|-----------|
| LongMemEval oracle (n=500, 全6题型对齐) | 10.4/5.5 | 39.2/10.9 | **base≈3.8×** | multi-session 5.3 vs 18.8；single-user 18.6 vs 74.3；single-assistant 12.5 vs 66.1；knowledge-update 10.3 vs 56.4；temporal 12.8 vs 28.6；preference 0 vs 0 |
| LOCOMO (n=400) | 2.75/2.7 | 19.25/16.9 | **base≈7×** | 单跳(cat4) 2.8 vs 29.9；多跳(cat1) 5.4 vs 21.6；时序(cat2) 0 vs 17.8；对抗(cat5,拒答) 2.8 vs 0 |
- **★对话记忆是 mem_space 与 base 差距最大的场景**：base 全注意力即使中段截断仍 ~3.8×(LME n=500 全对齐)/~7×(LOCOMO) 领先。对话信息碎+需精确时序/说话人绑定，128-slot 记忆压缩丢失精确绑定。
- **mem_space 输出在跑（非崩溃）**：抽样确认生成连贯、主题相关，但事实细节错（如"4年3个月" vs gold"4年9个月"）→ 记忆保留 gist、丢失精确事实，与 RULER/BABILong 32k 墙的"精确读出失效"根因一致。
- LME 中 single-session(用户/助手单段事实)差距最大(single-user 18.6→74.3, single-assistant 12.5→66.1，n=500 全对齐口径)：单段精确事实记忆都丢，印证读出端而非容量问题。LOCOMO 对抗类(cat5)base 反而 0(不拒答硬答错)，mem 偶尔拒答得分。
- 注：LOCOMO base 中段截断到 7900 token（原文 ~14.7k），是受限上限对照而非全开卷。

### ★★ Sliding-window 32k 长上下文 PPL（2026-06-17，commit fe0f28d，节点 .76）
脚本 scripts/eval_sliding_ppl.py + run_sliding_ppl_matrix.sh。每 cell 40 seq × 32768 tok（1.31M scored tok）。base = sliding window=8192/stride=4096（每 token 取最大左上下文打分一次）；mem_space = chunk=1024 流式过 128-slot persistent bank（chunk 内 LM loss，镜像训练 TBPTT 目标）。同一批 seq → 直接可比。ckpt=P11 chunk1024 deltarule normreadout。
| 数据集 | base PPL | mem_space PPL | Δ (mem/base) |
|--------|---------|--------------|-------------|
| codeparrot | 2.160 | 3.122 | +44% |
| proofpile | 4.234 | 5.975 | +41% |
| pg19 | 7.901 | 16.483 | **+109%** |
- **★mem_space 付出一致的长上下文流畅度税**：code/math（低熵、局部结构）退化温和（+40%），叙事散文 pg19（长程依赖最强）退化 2×——记忆瓶颈在长程连贯最关键处伤最重，与对话 eval gap 同向。
- ⚠️ **数据 bug 已修**：旧 data/pg19_chunks_llama3_noeos.npy 头部是 King James Bible（Llama-3 已记忆 → base PPL≈1.1 假信号）。已换 clean held-out Gutenberg 流 data/pg19_real_llama3_noeos.npy（4M tok，跳过 Bible 前缀）。任何用旧文件的历史 pg19 sliding-PPL 数字作废。single-forward sanity 验证（proofpile 6.30/codeparrot 2.35）后跑全量。
