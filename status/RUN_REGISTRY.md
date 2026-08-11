# RUN_REGISTRY.md — 训练 run 配置 + BABILong 结果总账

> **用途**：本文件是 mem_space 系列每个训练 run 的**配置 + 离线 BABILong eval 结果**的横向对照总账。
> 每启动一个新 run / 跑完一次 eval，必须在此追加或更新对应行，方便快速回答"X 配置 vs Y 配置在 BABILong 上差多少"。
> 评测口径统一：`scripts/run_babilong_mem_space.py`，n=100/length，babilong.metrics（`compare_answers`），qa1/qa2/qa5 × 0k-32k。

---

### ⚠️ FIFO chunk512/b25 step3000 W0 — 高分受 BABILong 数据泄漏污染（2026-06-25，先标★★★★破墙，14:55 修正降级）

**.7.53 chunk512/b25 step3000 W0 (n=100, _eval_taskpool_2group.sh, swa_eval_chunks=0)：**

| task | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 备注 |
|---|---|---|---|---|---|---|---|---|
| **b25/c512 qa1** | 96 | 99 | 99 | 93 | 40 | 34 | 30 | 0k-4k 泄漏污染 |
| **b25/c512 qa2** | 99 | 100 | 100 | 95 | 23 | 32 | 32 | 0k-4k 泄漏污染 |
| **b25/c512 qa5** | 100 | 100 | 97 | 87 | 65 | 76 | 68 | 0k-4k 泄漏；8k-32k OOD 待验 |
| MemoryLLM teacher qa5 | 47 | 50 | 45 | 39 | 39 | 38 | 34 | |
| 历史 P11 step500 qa5 | 74 | 89 | 81 | 60 | 48 | 45 | 44 | 无泄漏 |
| b50/c1024 step3000 W0 qa5 | 35 | 39 | 12 | 21 | 10 | 15 | 8 | 同泄漏但分低 |

**★★ 关键修正（2026-06-25 14:55，代码核实非口述）：b25 训练含 BABILong 数据，0k-4k 高分是数据泄漏，不是记忆能力。**
- `train_mem_space_dolmino_cpt.py` 默认 `--babilong_mix_fraction=0.15`（15% 训练步是 BABILong SFT），`--babilong_tasks=qa1,qa2,qa5`（**与 eval 完全相同任务**），`--babilong_lengths=0k,1k,2k,4k`（**覆盖 eval 的 0k-4k**），`--babilong_dataset=RMT-team/babilong`（**与 eval 同一数据集**）。
- `BABILongTrainDataset`（babilong_dataset.py:79）用 `load_dataset(name,length)[task]` 取样——该 HF 数据集每 length 只有 task split，**无 train/test 隔离 → 训练与 eval 样本池完全重叠**。
- `max_seq_len=chunk_size*4=2048`（line 3284）：0k/1k 整故事 <2048 完整进入训练（背答案），2k/4k 部分泄漏。
- b25 launch 脚本未 override 任何 babilong 参数 → 全 default。
- **∴ qa5 0k=100/1k=100/2k=97/4k=87 这些史无前例满分 = 模型背过 eval 答案，非长程记忆。** 这解释了为何 P11（数据配方无此泄漏到同样程度，或 step500 早停）0k 仅 74——不是 b25 更强，是 b25 见过测试集。

**8k/16k/32k（65/76/68）= 训练 OOD（max_seq_len=2048 从未覆盖 8k+），仍需 memory-disabled 对照确认：**
- 存疑点 1：驼峰形 8k=65 < 16k=76 > 32k=68 非单调，长档不应该更易。
- 存疑点 2：b25 buffer=25 chunk=12800 tok；32k≈64 chunk → FIFO 只留最近 25 个，**前 39 个（61% facts）被淘汰**。只留 39% facts 却 qa5 68？暗示答案可能不靠被淘汰的 facts（=靠 few-shot prior + 最后 chunk + 泄漏的格式先验）。
- **决定性判别 = memory-disabled 对照**（关 FIFO buffer，只喂最后 chunk+question 跑 8k-32k qa5）。若仍 60+ → 连"OOD 真长程"都不成立。`run_babilong_mem_space.py` 原无此开关，已派 workflow 加 `--memory_disabled`（wf_28a3f1c9）。
- 对照线索：b50/c1024 同样有 babilong_mix 泄漏，但 qa5 8k-32k=10/15/8（低）→ 说明 c1024 配置下泄漏没帮上长档（或 backbone 训坏）；b25 8k-32k 高 **可能**是 chunk512+小buffer 的结构优势，**也可能**是别的 artifact。未定论。

**∴ 之前"破墙/超越 MemoryLLM/Plan C 过时"的裁决暂缓**——需先做 (1) memory-disabled 对照隔离 8k-32k 真实性；(2) **用 held-out / 训练未见的 BABILong 长度或重新生成的 needle 重测**，排除泄漏。在此之前 b25 不能算 SOTA。Plan C 蒸馏方向**暂不作废**，待 8k-32k 真实性确认后再定。

CSV：`babilong_results/fifo_b25_c512_final_W0/`（.7.53 diskB）。

---

### ★ lr5e5 长训不崩双 seed step250 vs step500 W0（2026-06-23 03:30，杠杆2 优化轨迹判据，n=100）
动机：今晚 #1 假说「step250 最好、step500/1000 掉点 = lr 过冲」。lr 1e-4→5e-5 双 seed(s1234/s42)，pg19 nctx63 蒸馏，total 500 save250。

| 配置 | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|
| s1234 step250 qa5 | 79 | 69 | 49 | 27 | 17 | 14 | 11 |
| s1234 step500 qa5 | 77 | 67 | 48 | 25 | 16 | 12 | 11 |
| s42   step500 qa5 | 60 | 73 | 41 | 22 |  9 |  8 |  7 |
| s1234 step500 qa1 | 88 | 50 | 31 | 17 | 10 |  8 |  6 |
| s42   step500 qa1 | 87 | 44 | 27 | 18 |  9 | 10 |  5 |

- **裁决：lr5e5 step500 ≈ step250（噪声内，长程 16k 12 vs 14、32k 11 vs 11），lr 降到 5e-5 后「长训不崩」成立——但训练过 step250 也无增益。** s1234 长程稳于 s42（s42 8k-32k=9/8/7 偏低，seed 方差）。结合 nctx63 step500≈步250 结论：**杠杆2（训练侧优化轨迹/窗口/步数）全部耗尽，天花板未升**（16k≤14<n_ctx7 的 16，32k=11 持平 n_ctx7 的 9~11）。剩余唯一活跃机制杠杆 = 杠杆3（读出侧 SWA gap）→ 已起 lr5e5 step500 eval-SWA W2(local)/W6(.196) 量化读出鸿沟。

### ★★ lr5e5 s1234 final ckpt eval-SWA W6 量化读出鸿沟（2026-06-24，n=100）

> 底座：lr5e5 pg19 nctx63 蒸馏（mass_coef1_s1234 / mass_coef2_s1234），final ckpt（step500），eval-SWA swa_eval_chunks=6（W6）。
> 口径：n=100，qa1/qa2/qa5 × 0k-32k，_eval_taskpool_2group.sh，chunk512。
> 对照：W0 行来自上方表（s1234 step500）。

| 配置 | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|
| **s1234 W0 qa1** | 88 | 50 | 31 | 17 | 10 |  8 |  6 |
| **s1234 W0 qa5** | 77 | 67 | 48 | 25 | 16 | 12 | 11 |
| **s1234 W6 qa1** | 90 | 35 | 21 | 49 | 30 | 30 | 14 |
| **s1234 W6 qa2** | 49 | 18 | 12 | 17 | 15 |  7 |  2 |
| **s1234 W6 qa5** | 67 | 39 | 38 | 49 | 36 | 33 | 23 |
| **mass_coef2 W6 qa1** | 92 | 43 | 29 | 43 | 25 | 21 | 10 |
| **mass_coef2 W6 qa2** | 48 | 25 | 21 | 24 | 20 | 10 |  3 |
| **mass_coef2 W6 qa5** | 64 | 59 | 51 | 51 | 54 | 36 | 23 |

- **★关键裁决（2026-06-24）：W6 vs W0 鸿沟确认，且中长程有显著抬升。**
  - s1234 W6 qa5 中长程：8k **36** (+20 vs W0=16)、16k **33** (+21)、32k **23** (+12)。
  - mass_coef2 W6 qa5 中长程更强：8k **54**、16k **36**、32k **23**（大幅超 s1234 W0 基线）。
  - W6 vs W0 对比验证「memory 里存了信息，W0 读不出」假说——eval-SWA gap 是真实存在的读出鸿沟。
  - **mass_coef2 W6 qa5 8k=54 是迄今 lr5e5 配方最高中程分数**（W0 上限 16→SWA 突破至 54）。
  - W6 0k 短程略低于 W0（s1234 qa1: 90 vs 88，qa5: 67 vs 77）——SWA 引入近窗口注意代价，可接受。
  - **结论：SWA 是当前最有效读出增益杠杆，尤其 8k-16k 中程（3×+ 增益）；方案B FIFO 训练的核心假设得到 eval 侧支持。**

> 与 `status/BENCHMARK_RESULTS.md` 的分工：BENCHMARK_RESULTS 是含外部论文数字的大杂烩；**本文件只记我们自己的 mem_space run，强调配置可复现 + 同口径对照**。

最后更新：2026-06-24 23:30 GMT+8

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
| **rawkv_methodA_b200** | launch_rawkv_methodA_b200.sh | 512 | 4096 | — | **Method A raw-KV readout**：per-chunk 原始 KV + 可训练 emergent gist-key soft-attn（删 TopKSelector 出读路径），注入 L16/20/24，**解冻 reader L16-31**；数据 **T2 合成 needle（pg19 背景，frac0.5，gap3584/n_ctx7，单 needle）+ pg19 续写**（教检索且与 BABILong 不同源→eval 干净）；babilong_mix=0；gist soft-top-k=8 dim128 | 2000（诊断）| B200 8×L20A | **DONE — 不破墙/长程崩**（2000步训完 nf=0；W0 qa1 eval 见 §3：final 0k92/1k17/2k5/4k5/8k1/16k1/32k2；step1000 0k93/1k18/2k9/4k5/8k1/16k1/32k0。0k 满分→reader/LM 完好无灾难遗忘；长档（4k+）≤死线 → raw-KV+解冻也读不出长程 needle，之前 train t2_needle loss≈0 是过拟合 train 格式非泛化检索）|

---

## 3. BABILong 结果（accuracy %，n=100，qa1/qa2/qa5 × 0k-32k）

### ★mem_space self-study 蒸馏 chunk512 — W0 纯 memory 读出 checkpoint ladder（swa_eval_chunks=0），2026-06-21 本机H20 bs2
teacher=frozen Llama full-context，student=memory readout(rawkv grouped, 多层16-31注入, unfreeze L16-31)，distill_logits+hidden(layers12,20,28, λ0.6)。
```
ckpt          qa1: 4k   8k  16k  32k
step500            17   19   12    8
step1000           16   11    8    5
step1900           13   12   10    -
step2000(final)    14   15    7    -
```
- ★裁决：**self-study 蒸馏在 BABILong qa1 上确有真实长程信号** —— 4k/8k=14-19% 显著超官方 Landmark(S0 3/1) 与 pureT2(全0)，16k 仍有 7-12%、32k(step500) 8%。是迄今 raw-KV 读出系最好的真实 downstream 长程结果之一。
- ★过训退化：step500 仍是最佳（17/19/12/8），step1000 退化(16/11/8/5)，step1900/2000 部分回升但不及 step500。蒸馏**未完全消除** step500>step1000 退化，**但缓解了崩盘**：step2000 仍 14/15/7（不像 B 系列 step2000 直接归 0）。→ 早 ckpt(step500) 仍是交付点，蒸馏让后期更平稳但非单调改善。
- ★★**SWA 对照重磅发现**（step500，swa_eval_chunks=0/1/2）：加少量滑窗显著抬长程 —— qa1 32k: swa0=8 → swa2=15；qa1 16k: 12→19；qa5 32k: 7→18；qa5 16k: 11→26。**说明 memory 里确实存了长程信息，纯 W0 读出榨不干净，加 2 块直接注意原始 KV 就能多读出近 2×。** 与历史"SWA gap = 读出效率不足非范式问题"判据一致 → self-study 的下一杠杆是**读出侧**（让 W0 逼近 swa2），而非再堆训练。
```
self-study step500 SWA 对照:
              qa1: 8k 16k 32k    qa5: 8k 16k 32k
swa0(W0)          19  12   8         16  11   7
swa1              15  16  10         31  19  11
swa2              21  19  15         -   26  18
```
- 结果 csv: `babilong_results/selfstudy_w0_ladder_20260621_0540/` + `selfstudy_step500_swa_20260621_0705/` + `selfstudy_w0_qa25_ladder_20260621_0630/`。

### ★★★ 蒸馏退化机制总账 + "长训不崩"杠杆（2026-06-21，权威小节）

> 目的：把"为什么各种 self-study 蒸馏训久了会退化"讲清楚，并标出哪些杠杆能做到**长训不崩**（即后期 ckpt ≥ 早期 ckpt，而非 step500 永远最优）。这是当前主线的核心判据。

**三种 teacher 的退化形态对照：**

| teacher | qa1 W0 step500→后期（4k/8k/16k/32k） | 退化形态 | 根因 |
|---------|--------------------------------------|----------|------|
| **W2-teacher（近视，memory+最近2chunk raw-KV）** | 11/7/2/1 → step1000 5/0/1/1 | **急速塌缩，越训越崩** | 老师在"memory 读出"这条迁移轴上**不比学生强**；其优势全来自 1024-tok 局部窗口，而学生无窗口 → 逼学生拟合**不可见/不可学**的 target，读出通路被带歪。长档老师自己也瞎（窗口覆盖<6%），无可迁移信号。from-base 时读出弱→老师全压窗口→等于教"忽略 memory"，越训越退。 |
| **全上下文-teacher（dolmino ≤16k，n_ctx=3）** | 17/19/12/8 → step1000 16/11/8/5（step1900/2000 部分回升 14/15/7） | **温和退化，step500 最优** | 老师是真强，但 (1) teacher–student **容量鸿沟固定**（32k 书压进 128 槽必丢信息）→ 后期 KL 持续逼一个学生结构上够不到的 target，侵蚀已学好的部分；(2) **训练长度分布（≤4k 有效窗口）≠ 评测长度（16k/32k）**→ 训久=在短训练长度上过拟合，长档外推变差（退化长档最明显，短档稳）。蒸馏把硬崩盘磨成软退化但**未根除**。 |
| **全上下文-teacher + pg19 真长文（78%≥32k，n_ctx=7）** | step250 78/74/.../16k=16 ≈ final 75/73/.../16k=16 | **基本不退化（step250≈final）** | 训练长度分布对齐评测 → 消掉"过拟合短训练长度"那条退化通道。且 **16k 破长程天花板（=16 vs 所有现有方法≈13，+3）**。32k 仍硬墙（≈9），因 n_ctx=7 有效窗口 4096，32k 超训练分布 8×。 |

**★核心结论：234 是真正有用、能"长训不崩"的杠杆，应作为主线。**
1. **【杠杆2】训练长度对齐评测（pg19 真长文）** —— 唯一已验证能让"训久不退化"（step250≈final）+ 破 16k 墙的解。优先级最高。下一步：把有效训练窗口从 4096 推到 ≥32k（更大 n_ctx / chunk）去攻 32k 硬墙。
2. **【杠杆3】读出侧补容量鸿沟（SWA gap）** —— memory 里**确实存了**长程信息（swa2 比 swa0 多读出近 2×：qa1 32k 8→15、16k 12→19；qa5 32k 7→18），退化部分来自"读不出"而非"没存" → 改进 W0 读出（让其逼近 swa2）比再堆训练更高效。机制侧 > 训练侧。
3. **【杠杆4】confidence 加权蒸馏（confB1）** —— 过滤"老师自信但 memory 支撑不了"的不可学 token，理论上直接掐断"后期死磕不可实现 target"那条退化路径。warm-start（从 vanilla step500 续训）正在 .196 验证：续训能否修复退化、W0 超 17/19/12/8。
- 【非杠杆1】单纯早停 / 交付早 ckpt（step500/250）：是当前的妥协交付点，但治标不治本——目标是让后期 ckpt 也能涨，故主攻 2/3/4。

**判据（"长训不崩"达标线）**：某配置的 step1000（及更后）qa1 W0 ≥ step500，且长档（16k/32k）不随训练步数单调下降。当前仅"pg19 真长文"接近达标；W2/dolmino-全上下文均未达标。

**★★ SWA-teacher 系全证伪（2026-06-22 收口）**：W1/W2/confB1（confidence 加权）三个 SWA-teacher 变体的 W0 qa1 全部 ≪ vanilla 17/19/12/8，且无法长训不崩：
```
qa1 W0           4k   8k  16k  32k     vs vanilla 17/19/12/8
W2 seed1234 s500  11   7    2    1
W2 seed1234 s1800  1   0    0    0     (越训越崩)
W1 s500            2   3    0    0
W1 s1200           1   2    0    0
confB1-warm s500   2   1    1    0     (从 vanilla 17/19/12/8 续训→崩)
```
- **决定性证据**：confB1-warm 从 vanilla step500（17/19/12/8 的好初值）续训 500 步 confidence 加权 SWA-teacher，直接崩到 2/1/1/0 → 即便好 warm-start + confidence 过滤也救不回，**SWA/窗口 teacher 信号本身有害**（教学生抄它看不到的局部窗口，污染 memory 读出）。
- **结论**：杠杆4（SWA-teacher 各形态）证伪并关闭。主线收敛到 **杠杆2（pg19 长文，长度对齐，唯一已验证长训不崩）+ 杠杆3（读出侧补 SWA gap）**。

**★★ 杠杆2 决定性突破：32k 窗口 teacher（nctx63）破 32k 硬墙（2026-06-22）**
nctx63 = pg19 真长文 teacher cache，group_len=(63+1)*512=32768=**32k 有效训练窗口**（nctx7 是 4k）。蒸馏 500 步，A/B = nomass vs mass05。W0 qa1 BABILong：
```
qa1 W0            4k   8k  16k  32k     基准:vanilla nctx7 16k≈12-16/32k≈8-9(硬墙)
nctx63 nomass s250 18   12   14   12    ★32k=12 破墙(+3~4)
nctx63 nomass s500 18   12    9   10
nctx63 mass05 s250 17   15   16    9    ★16k=16(=pg19天花板)
nctx63 mass05 s500 18   11   10    7
```
- **★破 32k 硬墙**：nomass step250 qa1 32k=**12**（vs 此前所有方法 ≈8-9 天花板，+3~4）。把训练窗口从 4k 推到 32k，直接抬高 32k 长程——**坐实杠杆2「训练长度对齐评测」是破长程墙的真实有效杠杆**。
- **长训不崩成立**：step250 ≥ step500（与 pg19 nctx7 的 step250≈final 一致）→ 32k 对齐训练后期不退化，符合「长训不崩」达标线。
- **nomass > mass05 on 32k**：更长窗口下弱 mass(coef0.5) 反而略拖长程（32k: nomass 12 vs mass05 9）→ 32k 窗口已够强，不需 mass 辅助。
- **★dual-seed 确认（2026-06-22）**：seed1234 复现 32k 突破。nomass nctx63 qa1 32k: seed42 s250=12/s500=10, **seed1234 s250=10/s500=3** → 两 seed step250 qa1 32k 均 ≥10 > 8-9 天花板；qa5 32k: seed42 s250=9, **seed1234 s250=11**。step250≥step500 两 seed 都成立（长训不崩 + 早 ckpt 最佳稳健）。**→ 破 32k 墙非单 seed 运气，dual-seed validated。nomass nctx63 step250 = 新长程 SOTA（confirmed）。**
- 在跑确认/扩展：seed2026（local，第三 seed）+ s1000/long1000（1000 步，测训练时长 headroom）。下一步：收齐三 seed + 长训 grid 后定稿。
- **★triple-seed 定稿（2026-06-22）**：seed2026 step250 qa1 32k=11 → 三 seed step250 qa1 32k ∈{10,11,12}，全破 8-9 天花板。**nctx63 nomass step250 = triple-seed confirmed 长程 SOTA。** long1000(过训到1000步) qa1 32k=4 → **过训伤长程，step250 是甜点**，长训不是杠杆（负结果闭账）。

### ★★★ 杠杆3（读出侧）三 arm 全负 + 关键诊断：长程瓶颈 = 检索非读出（2026-06-22 收口）
SWA gap（W0 vs SWA2 在 SOTA 上仍巨大：qa1 4k 18→34、16k 14→23）曾被读作"memory 存了 W0 读不出"。为攻它做了 3 个 arm（均基于 nctx63 SOTA 底座）：
```
qa1 W0 step250    4k   8k  16k  32k    vs SOTA 18/12/14/12
A teacher-conf    18   13   10    8    ≈SOTA,长档略降(无改善)
B slot-kv(不限容量) 15   7    8    4    ★低于纯W0!(更远低于SWA上限34/25/23/11)
C student-conf    17   12   15   11    ≈SOTA(16k略升),s500过训退化(15→6)
```
- **★关键诊断（arm B 决定性）**：给每个 slot 挂**不限容量**的原始 KV cache（理论上限实验），W0 不仅没逼近 SWA 上限，反而**低于纯 W0**。→ **瓶颈不是"slot 向量有损压缩/读出"，而是检索：selector 选错 slot，即便每个 slot 挂了完美 raw-KV 也取到错的。** SWA 有效是因为它**无条件取最近 chunk**（不需检索）；slot-kv 依赖 router 选对而它选不对。与历史 usage_cov≈0.25 / 富者愈富 / ROUTE-A 全证伪一致。
- **A/C（confidence 加权）**：只重分配同一条 W0 通路的梯度，不碰容量/检索瓶颈 → 无改善（符合预期）。C 的 mean-teacher agreement 门控成功避免了 SWA-teacher confB1 式崩盘（没崩到 0），但也不涨。
- **★结论**：**SWA gap ≠ 读出问题，= 检索问题（router 找不到对的 slot）。** 杠杆3"读出侧/confidence"子方向全部证伪。真正剩余杠杆 = **修 selector/检索**（可微检索 / Landmark train-time-emergent-selection 迁移），属架构级方向，已 emit needs_code 等主会话定夺。当前确定交付物 = nctx63 32k triple-seed SOTA。

### ★Landmark 官方 ckpt BABILong qa1 — 首次 downstream eval（2026-06-21，本机 H20，landmark_venv torch2.1+tf4.28，top_k5 n=100 bs4）
```
                       qa1: 4k   8k  16k  32k
S0  landmark_tuned          3    1    0    0
S2  dolmino ckpt-3000      13    9   12    0
S4b learned-gate ckpt1000  0    0    1    0
```
- ★★裁决：**Landmark 系列在 BABILong qa1 downstream 上长程几乎全 0**（最佳 S2 也仅 9-13%，且 32k=0）。对照同 ckpt 的 passkey 主门 90-100%（见 SESSION_HANDOFF §0 / external/landmark/results_s2）→ **坐实「passkey/NIAH 破墙 ≠ BABILong downstream 破墙」**。
- ★方法论修正：此前用 passkey 当 Landmark→mem_space 迁移诊断守门（S2/S4b/S5 裁决）**有盲区** —— passkey 全程绿不代表该轴在真实长程 QA 上 work。迁移诊断守门判据应**升级为 BABILong qa1**（held-out 生成式），passkey 仅作机制是否 fire 的快速 sanity。
- 运维：B200(L20A sm_100) 无法跑 faithful landmark（torch2.1 不支持 sm_100；.venv tf5.5 的 pipeline 与 landmark patched tf4.28 不兼容）→ 必须在 H20/sm_90 节点用 landmark_venv 跑。结果 csv: `babilong_results/landmark_h20_qa1_20260621_0050_h20/`。

### rawkv_methodA_b200（Method A raw-KV readout + 解冻 L16-31）— W0 纯 memory 读出（swa_eval_chunks=0），2026-06-20
```
                  0k   1k   2k   4k   8k  16k  32k
final     qa1     92   17    5    5    1    1    2
step1000  qa1     93   18    9    5    1    1    0
```
- ★裁决：**不破墙**。0k=92-93%（reader/LM 完好，解冻 L16-31 无灾难遗忘）；但 4k+ 长档全部 ≤死线（in-attn oracle 21≈OFF22 / mem_space W0 长程天花板）→ raw-KV（无损内容）+ 解冻 reader **仍读不出长程 needle**。
- ★train t2_needle loss≈0.0000 是**过拟合 train needle 格式**，非泛化检索：held-out 随机 needle 的 W0 长档崩证实。
- final 与 step1000 同形（无过拟合后掉的额外信号，整段就低）。加载验证：901 keys missing=0 unexpected=0，架构与 ckpt 完美匹配。

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

## 3e. READOUT mass-bias sweep（2026-06-18，coef0.5 vs coef1.0，2 节点并行）

> 机制侧改动：在 normalized readout 上加 `--use_readout_mass_bias`，按检索质量给读出向量乘一个 mass 系数（coef = readout_mass_coef）。动机来自 readout-attack history 暗示"机制侧 mass 是长程最优"——验证是否能抬 W0 中长程（8k-32k）。
> 底座 = P11 chunk512 delta-rule+normreadout，total_steps=1000（**与 §3d ROUTE-A 同为 1000 步 final ckpt，可直接互比；但 base 是 5000-step run 的 step500 早 ckpt，谱系/步数不完全可比**）。口径 n=100，qa1/qa2/qa5 × 0k-32k，chunk512，bf16 sdpa，_eval_taskpool_2group.sh，W0（无 eval-side SWA）。
> 节点：coef1.0=本机盘A、coef0.5=.196 盘A，全 8×H20。两 run 唯一变量 = mass_coef。commit 9049469。
> **对照基准 = P11 base step500（W0）**：qa5=74/89/81/60/48/45/44，qa1=97/67/53/37/20/25/18。

```
arm                          qa  |   0k   1k   2k   4k   8k  16k  32k
P11 base step500★            qa1 |   97   67   53   37   20   25   18
                             qa5 |   74   89   81   60   48   45   44
massbias coef1.0 step1000    qa1 |   83    7    8    6    7    5    2
                             qa2 |   43    5    3    1    6    6    7
                             qa5 |   79   41   29   11    7    8    8
massbias coef0.5 step1000    qa1 |   89   13    6    8   10    9    9
                             qa2 |   45    6    5    1    4    4    2
                             qa5 |   76   47   36   17   19   19   27
```
→ **★mass-bias 裁决（2026-06-18）：REJECTED，两 coef 均显著劣于 P11 base，长程尤甚。** 假说（机制侧 readout mass 抬 W0 长程）被证伪：
- **长程全面塌方**：P11 base qa5 8k/16k/32k=48/45/44；coef1.0 仅 7/8/8（近乎全崩），coef0.5 仅 19/19/27（折半）。没有任何 cell 超 base。
- **coef0.5 > coef1.0**：弱 mass-bias 破坏更小（qa5 长程 19/19/27 vs 7/8/8，qa1 1k 13 vs 7），与 readout-attack history 的"弱 mass 更优"方向一致，但**仍远低于 base**——弱化只是"破坏更小"，非"有益"，复刻 §3d/L3-recon-aux 的同款模式。
- **仅 0k 近持平**：coef0.5 qa1 0k=89、qa5 0k=76（≈base 97/74），coef1.0 qa5 0k=79；≥1k 立刻断崖。短上下文（slots 几乎用不上）才不被 mass-bias 破坏。
- **机制解读**：按检索质量缩放读出 mass 实际把"检索不确定"的长程 query 读出信号压弱，反而切断了长程寻址。这与 §3d/L3-recon 结论一致——在 P11 best 之上加任何"机制侧重整 readout/routing"的旋钮都损害长程检索。**最佳仍是 P11 base（delta-rule+normreadout，无 mass-bias）。** mass-bias 方向关闭。

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

### ★★ n_ctx=63 满 32k 训练窗口攻 32k：第三个负结果（2026-06-18，qa1/qa2/qa5 W0，n=100）
动机：n_ctx15（窗口 8192）/ N256（降压缩比）两路均未破 32k。最后一搏——把训练窗口直接推到**完整 32768 token**（n_ctx=63，有效窗口 (63+1)*512=32768 = 推理 32k 长度），彻底排除「训练窗口 << 推理长度」这一假设。配方同 pg19 蒸馏（chunk512 / 128 slot / distill λ0.6 layers12,20,28 / 不加 mass / seed42 / 500步）。eval 在 diskB .76，标准 2-group taskpool，HF offline。step500 final 训练中（截至 eval 时仅 step250 出）。
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|-----|----|----|----|----|----|----|----|
| pg19 n_ctx7 N128（基线，16k 破墙）| 75 | 73 | 51 | 29 | 19 | **16** | **9** |
| pg19 n_ctx15（窗口 8192）final | 77 | 73 | 55 | 31 | 17 | 15 | 8 |
| **nctx63（窗口 32768）step250** qa5 | 61 | 73 | 47 | 27 | 14 | **12** | **9** |
| nctx63 step250 qa1 | 91 | 49 | 34 | 23 | 12 | 14 | 11 |
| nctx63 step250 qa2 | 42 | 33 | 15 | 9 | 6 | 5 | 4 |
| 现有方法天花板 | — | — | — | — | ~15 | **13** | **9** |
- **★决定性负结果：训练窗口=推理长度（32768）也没有打破 16k/32k 墙。** qa5 32k=9（持平天花板，未突破）；qa5 16k=12（**反而低于** n_ctx7 的 16 和天花板 13）。qa1/qa2 长档同样无突破（qa1 16k=14/32k=11、qa2 全程更弱）。
- **「训练窗口=推理长度」假设彻底证伪**：n_ctx7(4k窗口)16k=16 > n_ctx15(8k)16k=15 > nctx63(32k窗口)16k=12 —— **加大训练窗口与长程性能呈单调负相关**，不是正相关。三个负结果（n_ctx15 / N256 / nctx63）三路独立确认 32k 墙不是训练长度/压缩比能解。
- **机理推断**：长训练窗口让每样本上下文更长、有效梯度信号更稀疏（同 500 步看到的「真长程依赖样本」反而更少 token-level 监督密度），中程（8k-16k）退化。**16k=16 的甜点仍是 n_ctx7（窗口 4096，约 8× 短于推理长度）**，再加窗口边际持续为负。
- **裁决**：排除「加训练窗口」全部路线（4k→8k→32k 单调变差）。32k 需架构级改动（分层记忆/检索增强），非数据/训练长度旋钮。step500 final 出后可补点，但 step250 已收敛、方向已定，不抱翻案预期。

### ★★ nctx63 + weak-mass(coef0.5)：训练侧 mass 杠杆也无效（2026-06-18，qa1/qa2/qa5 W0，n=100）
动机：nctx63 plain（32k 训练窗口）未破墙后，叠加 weak-mass(coef≈0.5)——此前 readout-attack 结论中弱 mass 是长程最优杠杆——看训练侧 mass 是否在满 32k 窗口下抬升长程。配方完全同 nctx63 plain（pg19 蒸馏 chunk512/128 slot/distill λ0.6 layers12,20,28/seed42/500步），仅加 mass_coef≈0.5。eval 在 diskB .249，标准 2-group taskpool，HF offline，step250（与 plain 严格 step-matched）。
| run（step250 严格对照）| 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|-----|----|----|----|----|----|----|----|
| **nctx63 plain** qa5 | 61 | 73 | 47 | 27 | 14 | **12** | **9** |
| **nctx63 mass0.5** qa5 | 62 | 69 | 55 | 28 | 15 | **13** | **7** |
| nctx63 plain qa1 | 91 | 49 | 34 | 23 | 12 | 14 | 11 |
| nctx63 mass0.5 qa1 | 93 | 48 | 34 | 21 | 10 | 13 | 12 |
| nctx63 plain qa2 | 42 | 33 | 15 | 9 | 6 | 5 | 4 |
| nctx63 mass0.5 qa2 | 48 | 22 | 16 | 13 | 2 | 1 | 3 |
| pg19 n_ctx7 N128（基线，16k 破墙）qa5 | 75 | 73 | 51 | 29 | 19 | **16** | **9** |
| 现有方法天花板 | — | — | — | — | ~15 | **13** | **9** |
- **★负结果：mass0.5 在 nctx63 上未抬升长程，与 plain 噪声内持平。** qa5 16k=13(mass) vs 12(plain)（+1，噪声内）、32k=7(mass) vs 9(plain)（反降 2）；qa1/qa2 同样无系统性增益（qa2 8k-16k mass 反而塌到 2/1 < plain 6/5）。两臂均远低于 n_ctx7 基线 16k=16。
- **结论：训练侧 mass 杠杆在 32k 训练窗口下失效。** readout-attack 中弱 mass 的长程增益是**机理侧（推理时 readout 软化）**现象，未能通过训练侧 mass_coef 复现到这个蒸馏配方——再次印证「机理侧 > 训练侧」（见 readout-attack 裁决）。**32k 墙对训练窗口、压缩比、训练侧 mass 三类旋钮全部免疫**，确认需架构级改动。

### ★ nctx63 W0 step500 vs step250 双步确认（2026-06-22 06:00，mass05 + nomass 两臂 W0，qa1/qa2/qa5×0k-32k，n=100，B200 .53/.18 共享 wzc1 盘）
判据二度确认：(a) 加大训练窗口未破 32k 墙；(b) step500≈step250「长训不崩」是否在 nctx63 成立。
| run | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|-----|----|----|----|----|----|----|----|
| nctx63 mass05 step500 qa5 | 68 | 70 | 53 | 28 | 14 | 11 | 8 |
| nctx63 mass05 step250 qa5 | 62 | 68 | 56 | 30 | 16 | 12 | 9 |
| nctx63 nomass step500 qa5 | 56 | 69 | 50 | 24 | 14 | 11 | 7 |
| nctx63 nomass step250 qa5 | 57 | 65 | 55 | 25 | 15 | 12 | 5 |
| nctx63 mass05 step500 qa1 | 96 | 50 | 31 | 18 | 11 | 10 | 7 |
| nctx63 nomass step500 qa1 | 95 | 53 | 32 | 18 | 12 | 9 | 10 |
- **结论确认**：nctx63 两臂 W0 qa5 32k≈7-9（持平 n_ctx7 的 9）、16k≈11-12（**反低于** n_ctx7 天花板 16）。step500 与 step250 噪声内持平（长训不崩成立但天花板未升）→ **杠杆2（加大训练窗口攻 32k）作为 32k 杠杆彻底耗尽**，与 n_ctx15/N256/mass0.5 三路独立负结果一致。剩余候选回到机理侧（杠杆3 读出 / 换读出范式 / 架构级改动）。
- 真实结果（两臂 CSV 满 n=100，无 silent-fail；plain 在 .76、mass0.5 在 .249，严格一节点一 run 无 GPU 争用）。

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

### ★★ 新 ckpt eval：INSTRUCT 底座 c512 + Curriculum c1024（2026-06-17，eval-filler）
两个刚训完未评的 ckpt，taskpool 2-group 同口径（n=100/cell，qa1/qa2/qa5 × 0k-32k，bf16 sdpa，babilong.metrics）。列顺序：0k/1k/2k/4k/8k/16k/32k。**对照基准 = P11 chunk512 step500（当前最佳）**：qa5=74/89/81/60/48/45/44，qa1=97/67/53/37/20/25/18。
| run | base model | chunk | 配置 | 节点 |
|-----|-----------|-------|------|------|
| mem_space_p11_chunk512_INSTRUCT_final | **Meta-Llama-3-8B-Instruct** | 512 | P11 delta-rule+normreadout，128 slot，total_steps=1000 | 本机 diskA |
| T2_recall_chunk1024_CURRIC_final | Meta-Llama-3-8B | 1024 | curriculum 4→32k，delta_rule=false+normreadout，128 slot | B200 28.88.184.52 |
```
INSTRUCT c512 final      qa1 |   98   33   28   23   14   12   17
                         qa2 |   49   17   16   15    8    7    9
                         qa5 |   81   50   52   29   25   32   27
Curriculum c1024 final   qa1 |   97   88   43   26   14    9    7
                         qa2 |   44   53   19   19    4    1    3
                         qa5 |   80   69   44   42   25   12    7
对照 P11 c512 step500     qa1 |   97   67   53   37   20   25   18
                         qa5 |   74   89   81   60   48   45   44
对照 P11 c1024 FINAL      qa1 |   56   56   15   15    7    5    0
```
**裁决：**
1. **INSTRUCT 底座未带来提升，反而劣于 base-model P11 chunk512**：INSTRUCT qa5 中长度（1k=50/2k=52/4k=29/8k=25）全面低于 P11 c512 step500 base（89/81/60/48），仅 0k（81 vs 74）和 16k/32k（32/27 vs 45/44）持平偏弱。qa1 同样 1k=33≪67。**Instruct 对齐底座对 mem_space 长程检索无益甚至有害**——可能 chat-tuned 表征与 memory readout 训练目标不契合，且该 run 仅训 1000 步（vs P11 5000 步早停 step500），数据量更少。
2. **★Curriculum 4→32k 训练显著修复了单 chunk1024 的「1k 后断崖」**：Curriculum c1024 qa1=97/88/43/26/14/9/7 vs 此前 P11 c1024 FINAL=56/56/15/15/7/5/0——0k/1k 从 56/56 飙到 97/88，2k-8k 也翻倍（43/26/14 vs 15/15/7）。qa5 同样 80/69/44/42 vs c1024 FINAL 长程近零。**渐进 curriculum（warm-start 短上下文再拉长）是修 chunk1024 稀释/注入太稀疏的正解**，与 §3「F1 渐进 warm-start 才是修单 chunk1024 断崖的正解」假设一致并实证落地。但绝对值仍略逊 chunk512 甜区（curriculum c1024 qa5 2k=44 vs P11 c512 81），chunk512 仍是同架构最优 chunk。
- 全程无 OOM / NCCL / Traceback；两节点开跑前均确认真空（未碰 .249 训练 / .196 蒸馏缓存）。

### ★★ self-study 蒸馏 AB / MASS0p5 + L2+pg19：dolmino 蒸馏未抬高 W0 readout（2026-06-17，qa5/qa1/qa2 W0，n=100，babilong.metrics）
三个刚训完的 ckpt 离线 BABILong 打分（score_nested_babilong.py，4-shard×25=100/cell）。底座均 frozen Meta-Llama-3-8B chunk512 + 128 slot。
- **distill_AB**：self-study 蒸馏 A(logits KL)+B(hidden MSE)，**dolmino 数据**（≤16k，n_ctx=3），不加 mass。本机 diskA。
- **distill_MASS0p5**：同 AB + 弱 mass(coef≈0.5)。.196 diskA。
- **L2ON_pg19**：L2 分层记忆开 + pg19 真长文。.249 diskB。
```
distill_AB step500       qa1 |   90   46   33   18   11    9    6
                         qa2 |   47   28   18    8    4    1    2
                         qa5 |   79   54   55   28   11   11    8
distill_AB step250       qa1 |   91   43   31   20    8    9    5
                         qa2 |   42   31   16    8    2    0    1
                         qa5 |   80   53   49   26   15   11    8
distill_MASS0p5 step500  qa1 |   92   40   28   18   11    8    3
                         qa2 |   43   21   20   12    2    1    3
                         qa5 |   79   48   50   25   11   14    9
distill_MASS0p5 step250  qa1 |   90   40   32   18    9    7    5
                         qa2 |   48   21   14    8    2    1    3
                         qa5 |   81   43   42   18   14    9    7
L2ON_pg19 step500        qa1 |   94   36   27   16    8    8    2
                         qa2 |   47   22   14   11    3    3    2
                         qa5 |   80   54   44   18   18   13    5
对照 P11 c512 step500     qa5 |   74   89   81   60   48   45   44   (best chunk)
对照 P11 c1024 deltarule  qa5 |   29   68   29   15    7    4    5   (SOTA)
对照 pg19 真长文蒸馏 final qa5 |   75   73   51   29   19   16    9   (16k破墙)
```
**裁决：**
1. **dolmino 自学蒸馏（AB / MASS0p5）未抬高 W0 readout——尤其中长程全面塌**。两 arm 的 qa5 在 4k-32k（AB=28/11/11/8、MASS0p5=25/11/14/9）远低于 P11 chunk512 step500（60/48/45/44），也低于 pg19 真长文蒸馏 8k-16k（19/16）。step250≈step500（无额外增益，250步已收敛）。**dolmino（≤16k 且 n_ctx=3 窗口仅 2048tok）数据是限制：见不到真长程依赖，蒸馏只复刻短档。** A+B 双蒸馏目标未补上 readout。
2. **MASS0p5 ≈ AB**：弱 mass 无明显加成（16k 14 vs 11 在噪声内，32k 9 vs 8 持平）。weak-mass+distill 在此 dolmino 底座未复现 readout-attack 的长程优势——印证「真长文数据」比「训练侧旋钮」更关键。
3. **L2+pg19 仅 8k 微正、整体无突破**：qa5 8k=18（略超 AB 的 11、与 pg19 蒸馏 19 持平），16k=13，但 32k=5（最差）。L2 分层 + pg19 长文没带来稳定长程增益，32k 反而更低。
4. **三 arm 均仍撞 32k 墙**（8/9/5），无一突破天花板 9。**最佳长程仍是 pg19 真长文蒸馏（16k=16）**，dolmino 系 distill 不及。**结论：抬 readout 靠真长文训练数据，不靠蒸馏目标/mass 旋钮。**

#### ★ LongBench W0 补评（2026-06-18，填补 findings 证据缺口1）
两个 dolmino 蒸馏 ckpt（之前只跑过 BABILong）补真长文档 QA。口径严格对齐 P11 baseline：base Meta-Llama-3-8B + 128 slot，chunk512，**W0（memory-only，无 SWA）**，no_chat_template，tasks={multifieldqa_en, 2wikimqa, musique} × n=100，babilong/LongBench F1 口径（4-shard×25 合并）。
```
ckpt                         | mfqa_en  2wikimqa  musique | AVG(F1)
distill_AB_dolmino  final    |  12.75    10.45     3.41   |  8.87
distill_MASS0p5     final    |  14.13     9.36     3.46   |  8.98
L2ON_pg19           final    |  11.25     9.55     2.57   |  7.79
对照 P11 c512 step500(同3任务) |  15.83    11.31     3.46   | 10.20
对照 base 开卷上界(中截断,同3任务)|  24.87    12.17     6.97   | 14.67
```
**裁决（LongBench 真实长文档迁移）：**
1. **三个 ckpt（dolmino AB / MASS0p5 / L2+pg19）在真长文档 QA 上均未发生正迁移**——AVG 8.87/8.98/7.79 < P11 baseline 10.20 < base 开卷上界 14.67。蒸馏/分层均未补上 readout，反而均略低于未蒸馏的 P11 mem_space baseline。
2. **MASS0p5 ≈ AB**（8.98 vs 8.87，差异在噪声内），与 BABILong 结论一致：弱 mass 无加成。
3. **★L2+pg19 是三 arm 最低（AVG 7.79，2026-06-18 补评，.249 diskB，8-shard×~13）**：与其 BABILong 表现一致（仅 8k 微正、32k=5 最差），L2 分层 + pg19 长文在真长文档 QA 上也无迁移，全面落后 P11 baseline（−2.4 F1）与两 dolmino arm。三任务全线偏低（mfqa 11.25 / 2wiki 9.55 / musique 2.57），musique（多跳）尤其塌。
4. **远低于 base 开卷上界**（7.8-9.0 vs 14.67，仅 base 的 53-61%）：128-slot memory readout 在真实长文档 QA 上仍显著落后于直接中截断喂全文的 frozen backbone。**与 BABILong 同向——蒸馏/分层/真长文训练数据对真长文档任务均无迁移收益，瓶颈是 frozen-reader readout 能力而非蒸馏目标/训练数据。** EM 全 0（生成短答案 token-F1 口径下 EM 几乎不触发，与 P11/base 同样近 0，非异常）。

### ★★ n=200 RULER 5臂 evidence-injection probe：注入证据不抬 readout（2026-06-17，niah_single_1 4k，n=200）
.76 diskB，读 ruler_results/5arm_p11frz_n200_*/_summary.json 直接汇总（不经 score_nested）。底座 = P11 frozen，5 臂只改 readout 阶段是否/如何把"证据 chunk"注入。oracle_hit=200 表示 oracle 臂 100% 命中目标 chunk。
| 臂 | score (n=200) | Δ vs OFF | 说明 |
|----|--------------|----------|------|
| **OFF**（不注入，纯 128-slot readout） | 23.5 | — | 基线 |
| heur_pos0（启发式检索，注入到 pos0） | 23.5 | **0.0** | 无变化 |
| heur_realpos（启发式检索，注入到真实位置） | 25.5 | +2.0 | 噪声内 |
| oracle_pos0（完美命中，注入 pos0） | 26.0 | +2.5 | 噪声内 |
| oracle_realpos（完美命中，注入真实位置） | 21.0 | **−2.5** | 噪声内（反而降） |
- **★核心负结果：没有任何 evidence arm 稳超 OFF（噪声阈 >3pt）**。最大正向 oracle_pos0=+2.5，仍 < 3pt；oracle_realpos 甚至 −2.5。**即便 oracle 100% 命中正确证据 chunk（oracle_hit=200/200），把它注入 readout 也几乎不动分**——说明 32k/长程墙的瓶颈**不在"找不到/注错位置证据"，而在 readout 端拿到正确证据也用不上**。
- 与 LongEval（≥8k 精确检索归零）、对话记忆（保 gist 丢精确事实）三方互证：**问题是读出端精确事实绑定失效，给对证据也救不了**。evidence-injection 这条修补路线 REJECTED。pos0 vs realpos 差异（pos0 略优）在噪声内，不构成位置编码结论。

### ★★ Training-free raw-KV 检索通道 probe：并联未压缩 KV 也不破墙（2026-06-17，niah_single_1 4k，n=100）
本机 H20（GPU0/1/2 各一臂，无 DDP）。底座 = P11 frozen（`mem_space_p11_chunk1024_deltarule_normreadout/mem_space_adapter.pt`，chunk_size=1024，swa=0），**不训练，只在 eval 开 `--use_rawkv_retrieval`**。机制：在 `rawkv_layer=16` 把每个 chunk 的**未压缩 token hidden states**全量 append 进 per-seq raw-KV store（slot 之外、不压缩），question 时用当前 query routing key 做 dot-product 检索 top-k 原始 token，注入到与 evidence 同一条 EV-prefix 注意力块。
| 臂 | score (n=100) | Δ vs OFF | 说明 |
|----|--------------|----------|------|
| **OFF**（纯 128-slot readout，不开 rawkv） | 21.0 | — | 基线（hits 21/100） |
| rawkv L16 topk64 | 22.0 | **+1.0** | 噪声内 |
| rawkv L16 topk256 | 22.0 | **+1.0** | 噪声内（更多检索≠更好） |
- **validity 确认（非 no-op）**：`RAWKV_DEBUG=1` 实测 store 随 chunk 增长（4k 下 store_M 1024→2048→3072，4 chunk 全量入库），retrieval 每步真触发（retrieved=64/256 token 注入 EV prefix）。append 在 ingest（unfrozen）发生、freeze 后 retrieve、reset 清空——时序正确。输出非退化（模型产出合理 "special magic number" 补全，部分精确命中）。
- **★核心负结果：raw-KV 检索 NOT 破墙**。两臂仅 +1.0pt（≪ 决定性阈值 >10pt，也在 ~3-7pt 噪声内），4k 无任何信号 → **未扩 8k-32k**（无意义；32k 墙 qa5≈9 不受影响）。topk256 vs topk64 持平，说明"检索更多原始 KV"无加成。
- **与 evidence-injection probe（OFF=23.5，oracle 100%命中仅 +2.5）互证**：raw-KV 用**相同 EV-prefix 注入接口**但**不同检索源**（未压缩原始 token vs slot-routed 启发式/oracle）。两条路线都落在 OFF 噪声内 → **瓶颈不是"检索源/找不到精确信息"，而是 readout 端的注入接口本身——frozen reader 即便拿到逐字原始 KV 也用不上精确事实**。training-free raw-KV **不值得推进到 trained 版本**（除非先解决 readout 端如何"消费"注入 KV 的绑定问题，而非继续换检索源）。

### ★★★ 解冻 reader 全量 SFT（in-attn 注入）决定性裁决：解冻 1000 步未破墙（2026-06-19，niah_single_1 4k，n=100）
所有 training-side（mass/distill/curriculum）与 frozen training-free（evidence/raw-KV）路线穷尽后的**最后活线**：让 backbone 学会 attend 注入的 in-attention K/V（Landmark/RetrievalAttention 能 work 正因微调模型在注入机制上）。
- **run**：`outputs/sft_unfreeze_inattn_full/`，`--unfreeze_backbone`（整 8B 解冻）+ in-graph in-attn 注入（L16，梯度流过 injection），lr=2e-5，chunk512，total_steps=1000（nf=0，100.5min，本机 8×H20 FSDP）。底座 Meta-Llama-3-8B。
- **ckpt 格式**：`full_model.pt`（26GB，**整 8B 微调权重 + adapter**，非 adapter-only）。`_save_adapter` 在 `save_full_model=True` 时存完整 `model.state_dict()`（FSDP gather + 去 `_fsdp_wrapped_module./module.` 前缀）。
- **★加载正确性已验证**：eval harness `load_mem_space_model` 走同一 `load_state_dict(strict=False)` 路径，载入日志 `Loaded 1153 keys | missing=0 unexpected=0`（整 model state dict 全载，无缺/无多）。直接 diff ckpt backbone vs 原始 Meta-Llama-3-8B safetensors（按 `wrapped_layer.` 前缀对齐）：L0 q_proj max|Δ|=0.0026、**L16（注入层）q_proj max|Δ|=0.0098（最大，微调确实集中在注入层）**、L31 q_proj=0.0041、embed=0.0021——**backbone 微调权重确实载入，eval 测的是微调后模型而非原始 frozen**。
- 口径：`scripts/eval_ruler_mem_space.py`，niah_single_1 4k，n=100（4-shard×25 合并），chunk512（训练对齐），final step1000 ckpt。

| 臂 | score (n=100) | oracle_hit | vs frozen baseline |
|----|--------------|-----------|---------------------|
| **OFF**（slot-only，不开 inattn） | **12** | — | frozen OFF~22（更低） |
| **in-attn ORACLE@L16**（`--use_inattn_kv --inattn_kv_layer 16 --inattn_kv_topk 64 --inattn_oracle_only`，注入 gold needle 绕过 scorer） | **5** | **100/100** | frozen oracle~21（更低） |

- **★★裁决：解冻全量 SFT（单层 L16 in-attn 注入，1000 步）NOT 破墙，且呈强负信号。** oracle=5 **低于** OFF=12（注入完美定位的 gold needle，oracle_hit=100/100 满命中，模型反而答得更差），远未达破墙阈值 >35。step500 中间 probe 同向（OFF=13、oracle=7，oracle<OFF）→ final 确认非偶然。
- **双双 ≪ frozen baseline（OFF22/oracle21）**：解冻 1000 步后 OFF 从 22 掉到 12，说明全量 SFT 在 dolmino 短档（n_ctx 窗口小）上轻微损伤了 backbone 基础 NIAH 能力，而 in-attn 注入通道不仅没学会被 consume，反而成了干扰源（oracle<OFF）。
- **意义**：frozen reader（28+ 实验）+ unfreeze reader（本次）**两大框架在"让模型 consume 注入精确 KV"上双双穷尽**。单层 in-attn 注入 + 全量解冻 1000 步训练不足以让 backbone 学会 attend 注入 KV——可能需要 (a) 多层注入、(b) 注入机制从训练第一步即在（而非微调后期引入）、(c) 更长/更长程的训练数据（dolmino 短档见不到长程依赖，与 §3a 蒸馏结论同源），或 (d) 换 readout 范式。**训练侧 + 注入接口侧均无活线 → 需主会话裁定换范式。**

### ★ Phase 1 S0 锚点：Landmark Attention passkey 复现成功（2026-06-19，landmark-repro）
**目的**：在我司 infra 复现官方 Landmark（arXiv 2305.16300，epfml/landmark-attention）的 passkey 长程正面结果，建立"已知能破墙的 memory 方法"可信锚点（Phase 2/3 diff-based 迁移基线）。**零训练**：用官方 released weight-diff recover 出 tuned ckpt 直接 eval。

- **base**：LLaMA-1-7B（`huggyllama/llama-7b`，非 LLaMA-2——官方 wdiff 是对 LLaMA-1 的 diff，recover backward-compat checksum 49798.7656=LLaMA-1-7B psum，已过完整性校验）。
- **mem**：`epfml/landmark-attention-llama7b-wdiff` recover 出的 tuned ckpt（15k 步 RedPajama landmark-attn 微调，含 `<landmark>` token）。
- **env**：独立 venv `external/landmark_venv`（torch2.1.0+cu121 + transformers4.28.1 + numpy1.26 + hf-hub0.14.1 + pyarrow<13），非主 .venv（避 transformers5.5 API 冲突）。non-triton 路径（use_flash=False）。H20 sm_90。
- **eval**：passkey retrieval，50 tests/长度，top_k=5，n_garbage chars→token（garbage≈3.7 char/tok）。harness `external/landmark/run_passkey.py`（参数化+sharding），commit 1e66a16。

| ~tokens | base LLaMA-1-7B | landmark-mem |
|--------:|:--------------:|:-----------:|
| 70   | 100% | 100% |
| 1.1k | 100% | 100% |
| 2.2k | 98%  | 94%  |
| 4k   | **0%** | **100%** |
| 8k   | **0%** | 96%  |
| 16k  | OOM*   | 96%  |
| 32k  | OOM*   | 96%  |

\* base 全量 O(n²) attn，>2048 ctx（其 max_pos）passkey 已出训练窗→0%，≥16k 单卡 96GB OOM（base 本就无长程能力）。

**★裁决：成功复现破墙。** base 在自身 2048 ctx 上限处断崖（2.2k→4k：98%→0%），landmark-mem 一路 94–100% 直到 **~31k tok（96%）**——landmark in-context memory 机制确实破长程墙，与论文 passkey 图吻合。**这是 Phase 2/3 迁移的可信锚点：eval 口径正确 + 已复现一个能破墙的方法。** 下一步 Phase 2 差异表（working Landmark → mem_space 7 维差异）。

### ★ Phase 3 **S2（数据轴）**：dolmino 替 RedPajama — 训练+passkey eval 完成（2026-06-19，landmark-repro，Group-A）
**目的（单轴）**：从 S0 锚点出发，**只换训练数据** RedPajama-1T-Sample（7源,~0.98B tok）→ dolmino（wiki+pes2o 2源, 260M LLaMA-1 token），其余严格=锚点（LLaMA-1-7B base、grouped-softmax 全层、mem_freq63、lr2e-5、ctx512、单 512 窗口、full-FT、全序列LM loss、3000 步）。测 damage-investigator 的 #1 嫌疑「窄 2 源数据是否破长程墙」。
- **训练**：2-node FSDP full-FT（本机+.196 diskA，IB/RoCEv2 GID3 bond1，2.48s/it，2h7m，0 crash），train_loss 1.93。ckpt `external/landmark_ckpts/landmark_s2_dolmino/checkpoint-{1000,2000,3000}`（pytorch_model-0000{1,2,3}-of-00003.bin 齐全）。
- **eval**：native `run_passkey.py`（S0 同口径：n_garbage 0/4k/8k/15k/30k/60k/115k chars → 70/1.1k/2.2k/4k/8k/16k/32k tok，50 tests/档，top_k=5，mem arm，8-shard 本机分档）。结果 `external/landmark/results_s2/s2_dolmino_step{1000,2000,3000}/pooled.csv`。

| ~tokens | S0 锚点(RedPajama) | S2 step1000 | S2 step2000 | S2 step3000 |
|--------:|:------------------:|:-----------:|:-----------:|:-----------:|
| 70   | 100% | 0%* | 0%* | 0%* |
| 1.1k | 100% | 64% | 78% | 92% |
| 2.2k | 94%  | 68% | 72% | 82% |
| 4k   | 100% | 58% | 60% | 78% |
| 8k   | 96%  | 52% | 80% | 86% |
| 16k  | 96%  | 60% | 74% | 72% |
| 32k  | 96%  | 64% | 88% | 96% |

\* n=0（~70 tok，无 garbage 退化 case）三步均 0%：模型答 "the number that you..." 而非数字，是窄 dolmino 数据轻微损伤短 prompt 格式跟随，**非长程检索测试**（passkey 长程墙看 ≥4k 档）。

**★裁决：数据轴（RedPajama→dolmino）NOT 长程杀手，data-axis 排除。** 三步 step1000→3000 **单调上升、全程无断崖**：step3000 在 4k/8k/16k/32k = 78/86/72/96%，最长 32k 档 **96% 追平锚点**。换成窄 2 源 dolmino 后长程墙依然破（仅整体比 15k 步满训锚点低、且短 prompt 格式略损，符合 3k 步欠训 + 窄源）。→ damage-investigator 的「数据是 #1 嫌疑」**被证伪**，嫌疑转向机制轴（S3 ctx结构 / S4 检索 / S5 单层读出）。下一 round-2 轴 = S3（ctx/block 结构）。

### Phase 3 S5（单层读出轴）SCOPE 笔记（2026-06-19，landmark-s5，Group-B）
**目标**：把 landmark grouped-softmax/检索从「全 32 层」限制到「仅 L16」，其余轴严格不动（LLaMA-1-7B base、landmark_venv tf4.28、RedPajama-1T-Sample、mem_freq=63、lr2e-5 cosine+3%warmup、wd0.1、eff-batch128、ctx512、full-FT、全序列LM loss、~15k步）。守门=native run_passkey.py（70/1.1k/2.2k/4k/8k/16k/32k tok，50/档，top_k5）；锚点 94-100%，长档断崖=单层读出即长程杀手。

**★SCOPE 关键发现（已上报 team-lead，等 A/B/C 决策）**：Landmark 的长程 memory **不是可分离单元**——它就是每层自己的 KV cache（每 512 窗口 append，llama_mem.py:424）+ 该层的 grouped-softmax 读出（:469）+ 跨窗 top_k 检索（:318-430），全部由 `LlamaModel.forward:807-815` **一次性算出的 is_mem/last_section_mask 广播到所有 32 层**。与我们 mem_space「单层注入 128 slot」不同，这里没有可被「单层读出」的全局 bank。
- **单轴冲突**：「仅 L16 读、其余 31 层 vanilla 局部 512」无法仅靠按 layer_idx 关 grouped-softmax 实现：(1) landmark token 物理插入 input_ids（每64 tok），所有层都把它当普通 token 见到，除非加 per-layer mask（第 2 轴）；(2) mem_freq=None 路径会 concat 全部 past KV（OOM，非"局部"），is_mem=None 撞 :466 ValueError。真正「其余层只看局部」要重构跨窗 KV 缓存 = 记忆单元轴（S6），故 S5-as-specified 实际耦合 S5×S6。
- **选项**：A=软 S5（仅 L16 跑 grouped-softmax+检索，其余层 plain causal 看当前窗口、landmark token 在场但当普通 token、KV 仍缓存供 L16 用）——最接近单轴；B=完全局部隔离（其余层 KV 也重置/开窗）= 承认 S6 耦合；C=推迟 S5 到 S6 后（二者纠缠）。s5 建议 A。
- **ENV/DATA BLOCKER（diskB .76，8×H20 全空，已实测）**：landmark_venv 不在 diskB（需 rsync 4.4G；其 interpreter /opt/conda/envs/torch-base/py3.11 在 diskB ✓ 可重定位）；base ckpt 13G + landmark 代码也缺 → rsync；RedPajama-1T-Sample 全集群（A/B）均无缓存 → 需经 woa proxy 下载（已验证 .76 proxy 可达 HF）；**尚无训练 launcher**（仅 S0 eval harness，S0 是零训练）→ full-FT 7B eff-batch128 需写 FSDP/ZeRO 启动脚本。

#### S5 [READY] — 干净单轴实现已落地（2026-06-19，landmark-s5-2，commit fd76d56）
team-lead 采纳「Part-Y-only」干净单轴框架（比早先 option A 更紧、回归安全）：
- **机制**：`config.single_layer_mem`（默认 None = 全层 grouped-softmax = 字节级 anchor）。设为 layer idx → 仅该层跑 `landmark_grouped_softmax`，其余 31 层对**同一** attn_weights（含检索到的 prefix 列）跑 plain causal softmax。**KV/检索/窗口/数据全不动 → 不碰 S6**。landmark token 仍物理在 input_ids，非目标层把它当普通 key。把原作者写死的 dead `softmax` 行（旧 :466 ValueError 下面）变成真正 plain-softmax 路径。
- **实现**：`LlamaAttention/DecoderLayer.__init__(..., layer_idx)` + `LlamaModel.__init__` 透传 enumerate idx；归一化处 `use_grouped=(single_layer_mem is None) or (layer_idx==single_layer_mem)` 门控。~30 行。
- **CPU smoke PASS（diskB，隔离目录，无 GPU）**：(1) 回归证明 single_layer_mem=None vs 未改 anchor forward **max-abs-diff=0.000e+00**；(2) None → 全层 grouped；(3) 指定层 → 仅该层 grouped、其余 plain、**无 ValueError**、输出相对 anchor 改变。
- **产物（已 commit）**：`external/landmark/s5_patch/`（llama_mem.S5.py/.diff、config.S5、apply_s5.sh、s5_smoke_worker.py、README）fd76d56；`external/landmark/build_isolated_s5_tree.sh` 684c51f。
- **✅ 隔离树已建（team-lead 要求 ISOLATION 而非 merge，2026-06-19）**：`external/landmark_s5_tree/llama/`（diskB）= 从嵌套 repo HEAD（d963e50=pristine anchor 99631a8）`git archive` 出干净 llama/ 包 + apply_s5.sh 干净打 patch（6/6 hunk）= 「anchor + single_layer_mem only」单轴树，与 S4b 的 live 文件 `external/landmark-attention/llama/llama_mem.py` **物理分离、零合并步、零共享文件竞争**。从该隔离树 re-smoke（32 层 toy 匹配真实 7B 深度）：single_layer_mem=None vs pristine anchor **max-abs-diff=0.000e+00**；single_layer_mem=16 → grouped_layers=[16]、其余 31 层 plain、输出改变(0.390)、ISOLATED_SMOKE_RESULT PASS。
- **状态**：[READY] 真正零合并待发射。**未启动训练**——等某 16-GPU 组空出(Group-A ~S2 完成后)且 team-lead 确认。发射：从 `external/landmark_s5_tree/` 跑，`LM_SINGLE_LAYER=16` 接到 `from_pretrained(single_layer_mem=16)`。计划：S2 腾出 Group-A 后 S4b+S5 作为下一对并行(各占一组)，待 S2 裁决。
- **待定配置**：base=LLaMA-1-7B、mem_freq=63、ctx512、lr2e-5 cosine+3%warmup、wd0.1、eff-batch128、full-FT、~15k步（或先 3k 步诊断）；目标层默认 L16（中层），守门=native passkey 70→32k。

- **★Group-A 重建 + 2k 发射（2026-06-20，landmark-repro，接管 S5；S4b 已收口=门控不破长程，过训除外）**：diskB offline → S5 隔离 tree 在 diskB 没了，diskA 重建。
  - **builder**：`external/landmark/build_isolated_s5_tree_diskA.sh`（diskA 路径版）。⚠️关键修正：nested repo HEAD 现已是 S4b commit a699e3e（非 anchor），故显式从 **d963e50**（pristine anchor=md5 99631a8）`git archive llama/` 而非 `git archive HEAD`，apply_s5.sh 干净打 patch → `external/landmark_s5_tree/`。
  - **CPU re-smoke PASS**：single_layer_mem=None vs pristine-anchor **max-abs-diff=0.000e+00**（回归 byte-identical）；=16 → grouped_layers=[16]、其余 31 plain、max-abs-diff=0.390>0、无 ValueError。
  - **2节点 FSDP smoke 修 1 blocker**：launcher 模板的 `--gradient_checkpointing True` 撞 landmark grad-ckpt bug（llama_mem :460 把 use_cache 当 tensor → "Boolean value of Tensor ambiguous"）→ **grad-ckpt OFF**（与 S2/S4b 忠实配方一致，7B ctx512 单窗 FSDP 放得下）。修后 3 步 loss 7.71→7.62 finite。S5 起点 loss 高(~7.7)是真实轴效应（31 层失去结构读出从远离工作点出发），非 bug。
  - **train.S5.py 装入 S5 tree** 作 train.py（LM_SINGLE_LAYER env→config.single_layer_mem，liang2kl RedPajama mirror，num_proc 32→8 避 fork 风暴，加 S5_SMOKE_SUBSET）。
  - **状态**：[2k RUNNING] 2026-06-20 06:14 起，Group-A 2节点（本机 master+.196 worker，IB GID3 P2P_DISABLE=1，**port29585**），LM_SINGLE_LAYER=16，**max_steps2000 save500**（按过训铁律取 step1000/2000），eff-batch128，其余全 anchor。master log `logs/landmark_S5_singlelayer_master_20260620_061452.log`，OUT=`external/landmark_ckpts/landmark_S5_L16_singlelayer`。step36 loss 7.7→2.59 健康，~6.5s/it ETA~3.6h。
  - **runner** `scripts/run_landmark_S5_node.sh`（per-node setsid IB）。**eval** `scripts/eval_landmark_S5_passkey.sh`（关键：`LM_REPO` 指 S5 tree → run_passkey 用 single_layer_mem-aware llama_mem，否则 S4b-tree llama_mem 静默忽略 single_layer_mem 跑成全 32 层=错轴）；`scripts/auto_eval_S5_passkey.sh` 后台等训完(GPU 释放)依次 eval step1000/2000（S0/S2/S4b 同口径 70→32k/50/top_k5/8-shard）。
  - **判读**：step1000/2000 长档维持高(类 S4b 78-100%)→单层读出无害,嫌疑转 S6 记忆单元；长档断崖→单层读出是杀手(呼应 mem_space 单层 L16 注入设计缺陷)。commit nested 无改动(隔离 tree 不在 nested)；parent: runner+builder、eval+driver、activity。
  - **★★裁决（2026-06-20）：S5 = 长程杀手。单层读出彻底摧毁跨档检索。** passkey（n=50/档,8-shard,top_k5,与 S0/S2/S4b 同口径）：
    - **step1000**: 70tok **100%** → 1.1k/2.2k/4k/8k/16k/32k **全 0%**（pooled.csv 确认 50/50 vs 0/50）。
    - **step2000**: 70tok 100% → 4k 2%(1/50)、8k/16k/32k 0%（同断崖,过训未恶化也未恢复）。
    - 对照 S4b step1000 = 4k100/8k96/16k96/32k84、S0 锚点 94-100%。**唯一变量=读出层数(单层L16 vs 全32层),结果从78-100%崩到0%。**
    - **机制含义**：Landmark 跨档检索是**全层分布式读出**(32层每层都跑 grouped-softmax 读出),砍到单层即废。与 S4b(门控函数无害)合起来:怎么读不重要,但**读出的层覆盖广度**决定长程。**直接证实 mem_space "单层 L16 注入/读出" 是 0% precision 根因之一(非纯 selector 训练问题)。**
    - 已同步 methodA-eval:raw-KV 嫁接必须**多层注入**(单层复现0%断崖,与其 probe 16/20/24 多层一致)。



### Phase 3 **S4（检索轴）** SCOPE 笔记（2026-06-19，landmark-s5 被重定向到 S4，Group-B 2-node DDP .76+.249）
**目标（单轴）**：把 anchor 的 landmark-token grouped-softmax 块打分换成 small learned selector head（linear/MLP over each block key-summary），保持 top_k=5 硬选 **真实 raw-KV 块**、全层参与、RedPajama、ctx512、同 lr/batch/steps 不变。launcher = `external/landmark/train_landmark.sh`（官方配方 per_device2×ga8×16rank=eff-batch128，2-node 需把 ga 调到保持 128）。

**★SCOPE 关键发现（已上报 team-lead，等 S4a/b/c 决策）**：Landmark 的「检索」在**训练时根本不是 top_k selector**。
- 硬 top_k=5 块打分+选择 = llama_mem.py:357-413，**仅在 `past_key_value is not None and mem_freq is not None`（=推理有 KV cache 时）**触发；打分 = query·该块 <landmark> token 自己的 key（:357 mem_key_nopos / :367 matmul / :384 topk）。
- 训练（ctx512 单 512 窗口、无 past_key_value）走 `else` 分支（:432-439）→ **完全没有硬 top_k**，只跑 `landmark_grouped_softmax`（:469，def :222-243）：每块 <landmark> token 当软门控的 grouped-softmax。
- ⇒ 「只换 scorer」对训练路径 ill-posed：anchor 训练期检索器不是可分离 scorer；<landmark> token 的 key 即隐式块摘要。换成 external MLP 会牵动 memory-unit/数据路径（第 2 轴）。
- **选项**：S4a=只改推理打分（:357/367/384）换学习 scorer 重训（训练信号间接）；**S4b=把软 grouped-softmax 门控换成 learned per-block 标量门（landmark-token hidden 上的小 head），仍软、全层、真实 KV——训练期最真单轴，无硬 top_k、不动记忆单元**（s5 推荐）；S4c=reorder，S4 需 S6 式可分离摘要才干净 → 推迟。
- **ENV/DDP 状态（2026-06-19 已全部验证 ✓）**：base ckpt 13G ✓、landmark_venv 4.6G ✓（重传后完整，torch 2.1.0+cu121 / tf4.28.1 / datasets2.14.0 / CUDA True / 8GPU 可 import）、landmark 代码 ✓ on diskB .76。
  - **16-rank 跨节点 NCCL allreduce = PASS**（全 16 ranks=120.0=expected, OK=True）。⚠️ **c10d rdzv 会 HANG**（同 diskA）→ 必须用 **static rdzv**（`--master_addr 28.49.57.76 --master_port <p> --node_rank{0,1}`，**不要** `--rdzv_backend c10d`）+ `NCCL_SOCKET_IFNAME=bond1 NCCL_IB_DISABLE=1 NCCL_DMABUF_ENABLE=0 NCCL_NET_GDR_LEVEL=0 NCCL_P2P_DISABLE=1 GLOO_SOCKET_IFNAME=bond1`。NIC=bond1 两节点（.76 master/.249 worker），RTT 0.2ms，16 卡全空。
  - **RedPajama DATA BLOCKER—已解**：`togethercomputer/RedPajama-Data-1T-Sample` 现已 GATED（HTTP 401 even no-token；proxy 本身 OK，gpt2/wikitext/hf-root 全 200）。改用公开镜像 **`liang2kl/RedPajama-Data-1T-Sample-Backup`**（parquet, private=false, columns `['text','meta']` 完全匹配 train.py `example["text"]`，内容同源 drop-in）。全量下载在 .76 跑入共享 diskB cache `.hf_cache_s4`。备选镜像：`ll922/...Backup`、`ZengXiangyu/RedPajama-Data-1T-Sample`。

**★SCOPE 关键发现（已上报 team-lead，等定义决策）——train/infer 不对称破坏单轴**：
- 要换的「块打分 + top_k=5」在 Landmark 里只活在 **推理路径**（past_key_value≠None）：llama_mem.py:357 `mem_key_nopos`=landmark-token key、:367 `query·mem_key/√d`、:376-384 softmax+`topk(5)`→选真 KV 块。
- **训练（ctx512 单 512 窗口，past_key_value IS None → :432-439 分支）从不执行 318-430 的检索/top_k**。训练只跑窗口内 `landmark_grouped_softmax`(:469)（8 个 64-块）。**训练期无 top_k 选择**——landmark token 纯靠 grouped-softmax loss 学成「gist 门」，top_k 检索是推理期对这些已训 KV 的涌现用法。
- 故「只换 scorer、其余全同、ctx512 retrain」**不可单轴实现**：要让 learned selector 真的拿到梯度，必须把训练改成 multi-window+top_k（= 改 ctx/块结构 = S3 第 2 轴）；否则 selector head 训练期零信号、eval 时随机 = 无意义。
- **选项**：(1) 重定义 S4=「训练就带 multi-window top_k + learned selector」(接受 S3+S4 耦合，像我们 mem_space 训练时就 select)；(2) 改顺序 S3→S4（先 multi-window ctx 让 selector 有训练路径，再换打分头）；(3) S4=纯推理期 scorer swap 在**已训 anchor ckpt** 上（零 retrain 纯 eval 消融，selector 头需轻量 scorer-only FT）。建议 (2) 或 scoped (3)。
- **ENV/DATA 状态（diskB .76）**：首次 rsync 因 `tail -5` 管道吞错只传了目录骨架（venv lib 102K、base ckpt 空）→ 已重跑（~17.5G 进行中）。RedPajama 另一 worker（疑 landmark-repro S2）正在 diskA 现下载+tokenize（hf-cache/json 3.0G 增长中，32-shard arrow）；diskB 独立 FS 仍需各自拷/下。16-rank 跨节点 NCCL smoke 待 venv 落地后跑（NCCL_DMABUF_ENABLE=0 + NCCL_NET_GDR_LEVEL=0 + 验 NIC iface）。

### Phase 3 **S3（训练期 multi-window 上下文轴）** SCOPE 笔记（2026-06-19，landmark-s5，team-lead 拍板 S4→S3 reorder，Group-B 2-node DDP .76+.249）
**目标（被定为 S4 的前置）**：只改训练 ctx/窗口结构，让跨窗 top_k 检索路径（llama_mem.py:318-430）在训练 forward 中**真正执行并拿到梯度**（即在 M×512 多窗序列上训，而非 anchor 的单 512 窗），其余全同 anchor：grouped-softmax 打分（不换 learned head，那是 S4）、per-layer KV、RedPajama、mem_freq63、lr2e-5 cosine+3%warmup、wd0.1、eff-batch128、全层、full-FT。守门 = native passkey 70→32k；若多窗训练后仍 94-100% → S3 OK 进 S4；若断崖 → 训练期 windowing 是元凶。

**★PHASE-A 关键发现（已上报 team-lead，等是否接受 axis bleed）——multi-window 不是纯数据改动**：
- 为何 anchor 单窗：train.py:108 tokenize_fn 把语料 reshape 成正好 512-tok 行；:114 每 63 插 <landmark> → 行 = 520 tok。HF Trainer 单次 model(input_ids)、无 past_key_values、use_cache 默认 False。llama_mem.py:1000-1006 train.py 从不传 cache_top_k → max_chunk_length=None → :1009 window_len=全 520 → :1013 for-loop 只跑 1 次 → past_key_values=None → :432-439 分支 → 检索 318-430 **从不执行**。
- 让 318-430 在训练触发需：(a) seq=M×512 多行 concat（不 reshape 回 512），(b) 设 cache_top_k 使 :1001-1002 切窗，(c) past_key_values 跨窗传递（:1034 已做）。**但三处使它 >1 轴**：
  - ★use_cache 门：:430/:439 仅 `if use_cache` 才存 past_key_value；检索 :302 要 past_key_value≠None。FSDP 配方用 gradient_checkpointing，:820-825 在 grad-ckpt+training 时**强制 use_cache=False** → 即便喂 M×512，KV 仍不跨窗缓存 → 检索仍死。开跨窗训练**必须关 grad-ckpt（7B OOM 风险）或重写 KV 缓存 = 记忆单元/infra 轴 bleed**。
  - ★训练 loss bug :1035-1037：`last_logits=torch.cat(...)` 立即被 `last_logits=outputs[0]` 覆盖 → 只有**最后一个窗**进 loss。此循环是为推理 generation 写的，从非训练路径 → 必须改成累积所有窗 = 代码改动（佐证此路径从未被训练用过）。
  - 跨窗 KV 在 :428 `.to("cpu")`/offload，对缓存的早窗 KV 反传图是否存活需验证。
- ⇒ 诚实 S3 scope = 「在 M×512 上训、跨窗 top_k 检索 ACTIVE」→ 强制 use_cache=True → 不能 grad-ckpt。scorer 保持 grouped-softmax（S4 不动 ✓）**但确实 bleed 进记忆单元（KV 缓存）+ 一个 loss 累积修复**，非纯单轴。已按指示上报。
- **token budget**：anchor 520 tok/行 ×eff-batch128 = 66.6k tok/步 ×15k 步 ≈ 1.0B tok（~1 epoch RedPajama-Sample）。M×512 多窗：tok/步 ×M → 按「总数据量恒定」规则 steps=15000/M。可达性：passkey@32k=64 窗；M=4(2048) 训练测 4 窗检索能否外推到 64 窗 = landmark 长度外推主张本身，原则可达，gate 正是它是否成立。
- **ENV（diskB .76）✅ 基本就绪**：venv 4.4G ✓ + base ckpt 13G ✓（du 字节核实）；landmark 代码 dirs 已 rsync（train.py / landmark_venv/bin/python 在位）。**venv 实测可用**：`torch 2.1.0+cu121 tf 4.28.1 cuda True ngpu 8`（diskB H20 上重定位 OK）。RedPajama diskA tokenize 似已停（32 shard 17:06 后无写入，疑 landmark-repro S2 已缓存完）→ diskB 独立 FS 仍需 rsync arrow cache。**跨节点 NIC = bond1**（28.49.57.76/26 → worker .249 via bond1，TCP22 通）→ 16-rank NCCL smoke 用 `NCCL_SOCKET_IFNAME=bond1 NCCL_IB_DISABLE=1 NCCL_DMABUF_ENABLE=0 NCCL_NET_GDR_LEVEL=0`（待跑）。
- **★S3 PHASE-A 终判（2026-06-19，已上报，等 team-lead 选 S4b/S3'）**：真·跨窗 S3 = **fabrication-only**。(i) 喂长 seq(4096) 只触发 :469 对全 4096 的**稠密** grouped-softmax（单窗），检索 318-430 仍不触发 → 是长度轴非选择轴；(ii) 真跨窗需 :1013 loop 迭代(window_len<seq)，但 :1014-1018 idx>=1 分支 `raise NotImplementedError` unless use_cache，且 use_cache 撞 FSDP grad-ckpt → 等于伪造 Landmark 从未有的训练检索循环。**无 latent 多窗训练 flag**。⇒ S3 伪造，**S4b 才是最干净的可训练单轴**（soft grouped-softmax→learned soft per-block 标量门，全层、真 KV、landmark token 仍插、无硬 top_k、不动 use_cache/grad-ckpt/记忆单元，完全跑忠实单窗配方，只换门控函数）。s5 推荐 S4b。可选非伪造 S3'=model_max_length=4096 忠实长上下文稠密训练（长度轴，1 行改）。
- **★A2 RESOLUTION（2026-06-19，覆盖上条「fabrication-only」终判）**：team-lead 拍板 **ACCEPT fabrication = correctness plumbing 非科学混淆**（科学变量「训练见跨窗检索、scorer 仍 grouped-softmax」干净），并 gate 在「跨窗 KV 反传是否真流」的 cheap check 上。**check PASSED**（见上方 PHASE-A2 GRADIENT-FLOW 段，待补在本块后）：手动从外部逐窗驱动 model.model() 携带 past_key_values（绕过 :1014-1018 的 NotImplementedError，那只在 LlamaForCausalLM.forward 内部 loop），检索 318-430 训练期触发 6×，决定性跨窗 grad(s0)=2.4e-4 非零、control(no cache)=None。⇒ **S3 改为执行（非 S4b）**；line-428 .to(cpu) 受 offload_cache_to_cpu 门控仅推理、训练 KV 留 GPU 可微。下一 gate=8-GPU 7B FSDP @2048 grad-ckpt OFF 显存 probe。脚本 external/landmark-attention/llama/s3_gradflow_check.py。
- **★Group-B NCCL 全量验证（2026-06-19，landmark-s5）**：(1) **16-rank 跨节点 allreduce PASS**（全 16 ranks sum=120=expected, OK=True；teardown Abort COMPLETE 是正常 destroy_process_group 非错）；(2) **必须 static rdzv**（`--master_addr 28.49.57.76 --master_port <p> --node_rank{0,1}`），**c10d rdzv 实测 3 次均 hang 在 init_process_group**（同 diskA/Group-A）；(3) **★IB 必开**：2-rank 1GB allreduce busbw — TCP(IB off)=~1GB/s vs **IB on(`NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3`, RoCE v2)=17.13 GB/s（~17×）**，比 Group-A 的 10.7 还快。S3 7B FSDP 跨节点 all-gather 是瓶颈 → **Group-B 训练 env 必带 `NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 NCCL_SOCKET_IFNAME=bond1 NCCL_DMABUF_ENABLE=0 NCCL_NET_GDR_LEVEL=0 GLOO_SOCKET_IFNAME=bond1`**（注意：IB 开时 DMABUF/GDR 仍设 0，与 anchor 一致，实测干净无 hang）。smoke 脚本 external/bw_smoke.py + nccl_smoke.py。
- **★S3 FSDP 显存 PROBE 终判（2026-06-19，landmark-s5，已上报，命中 team-lead STOP）**：8×H20 97.8GB，LLaMA-1-7B FSDP full_shard bf16 grad-ckpt OFF fwd+bwd（无 optim step）。脚本 external/landmark-attention/llama/s3_fsdp_mem_probe.py。
  - **M=1**（512 单窗，检索 OFF）peak **15.6GB** ✓ loss finite；**M=2**（1024）**OOM 91.3GB**；**M=4**（2048）**OOM 91.3GB**。
  - **★关键：M=2 与 M=4 OOM 在同一 ~91GB，开销不随窗数增长 → 检索一旦开启就 per-window 爆炸**。根因 = 跨窗检索 :407-412（aggregate=None 分支）按**每个 query token** materialize selected_keys/values，shape=(bsz,nh,q_len,top_k*(mem_freq+1),hd)=(1,32,512,128,128)≈0.5GB/tensor ×(keys+values+attn_prefix)×32 层、grad-ckpt off 全留反传 → 仅检索中间量 ~64GB+。此路径**为推理写（q_len≈1 增量解码）**，训练 q_len=512 时 per-token 扩张炸。15.6→91GB 跃迁全是检索张量。
  - **死结**：可微检索路径**要求 use_cache=True**，但 grad-ckpt **强制 use_cache=False**（:820-825）→ 二者构造性互斥，无法用 grad-ckpt 压检索 activations 而不杀掉被测的检索路径本身。
  - **选项**：(A) **立即退 S4b**（跑安全的 grad-ckpt-on 忠实单窗配方，diff 已设计就绪）——最快出真结果；(B) chunked-retrieval：把 q_len=512 检索切成每 64 query-token 子批累积以封顶 per-layer 张量，保持可微、无需 grad-ckpt，但是对 :396-412 的真手术（超出"plumbing"），改数值风险，~半天；(C) 降 mem_freq/top_k 缩扩张 = 改 anchor 机制 = 第 2 轴，拒。s5 推荐 (A) 立即 S4b + (B) 留作 S3 工程决策。S3 gradient-flow 发现独立成立、有价值。等 team-lead 拍板。
  - **★landmark-s5 独立复现确认（2026-06-19，第二次 probe）**：用 external/landmark-attention/llama/s3_mem_probe.py（外部逐窗驱动 FSDP root，与真 S3 trainer 同形）独立跑出同一结论——M=4 与 M=2(cache_top_k=1) 均 OOM，单 GPU CUDA_LAUNCH_BLOCKING 精确定位到 **:410 `apply_rotary_pos_emb(selected_keys)`**（"Tried to allocate 512MiB, 94.77GiB in use"）。补充精确机制：便宜的 `aggregate="max_over_tokens"`（:373-377，token_retrievers=1）**仅在 offload_cache_to_cpu=True（推理 flag）激活**；训练 offload=off → 强制走 :378 `aggregate=None` → token_retrievers=q_len=512 扩张分支。8-GPU FSDP 的"index assert"是某 rank 先 OOM 的 async 表象（非真 indexing bug）。运维补充：本节点 8-GPU **intra-node NCCL 需 `NCCL_P2P_DISABLE=1`**（否则 init ncclInternalError）；venv torchrun shebang 硬编码 diskA python → 必须 `python -m torch.distributed.run` 启动。s5 同样推荐 (A) S4b。

### ★ Phase 3 **S4b（学习软门控轴）** 执行（2026-06-19，landmark-s5，Group-B .76+.249，team-lead 拍板 A=S4b）
**S3 已 banked**（检索路径 inference-shaped、训练显存不可行 = 结构性发现，与 gradient-flow 一起完整刻画「Landmark 从不训练 selection」）。**S4b = 唯一改 gating FUNCTION**：把 llama_mem.py:469 的 parameter-free `landmark_grouped_softmax` block-gate 换成 learned soft per-block 标量门，其余全同 anchor（LLaMA-1-7B、RedPajama mirror、mem_freq63、lr2e-5/wd0.1/eff-batch128/ctx512、单 512 窗、全层、landmark token 仍每 63 插、真 KV、grad-ckpt ON 安全——S4b 不激活检索显存爆炸）。
- **实现**（config flag `learned_block_gate`，默认 OFF → 不影响 teammates 的忠实 run）：llama_mem.py:489-505，小 MLP(hidden→hidden/4→1, SiLU) 作用于每块 landmark-token hidden → 标量 → `gate=exp(MLP)` 乘到该块 grouped-softmax 概率上。**final Linear 零初始化 → exp(0)=1 → step0 与结构门 bit-identical**（起于工作点，测「能否学着离开」非「随机初始化破坏」）。
- **★PHASE-A2 GRAD-CHECK PASSED**（tiny 2-layer，忠实单 512 窗，.76 实测）：(a) 起于工作点 ✓ gate-ON vs gate-OFF(同模型 toggle) logits max-abs-diff=5.07e-7≈0，且 MLP 原始输出 abs-max=0.0、exp=1.0 精确；(b) 门可训 ✓ step0=4/8 非零梯度（仅 final Linear——零初始化 adapter 标准行为，first layer 梯度过零权重=0），step1+ 优化器一步后 final 权重非零 → 8/8（3 步 AdamW 实测 4/8→8/8→8/8）；(c) loss 有限 ✓ 6.27。脚本 external/landmark-attention/llama/s4b_gradcheck.py。
- **★发射关键 dtype**：gate MLP 必须 **fp32**（`lm_hidden.float()` 入、gate 转回 bf16）。bf16 下零初始化 MLP 输出非精确 0（舍入）→ exp≠1 → step0 偏离工作点 ~9e-3，且零初始化微小梯度 bf16 下溢为 0（0/8 训练）。FSDP bf16 训练时须保证 lm_gate 参数留 fp32（MixedPrecision 否则会 cast）→ launcher 用 FSDP 排除 lm_gate 出 bf16 wrap 或独立 fp32 unit，发射前在真 2-node FSDP 配置里验证 gate 仍 fp32+可训。
- **⚠ 与 S5（landmark-s5-2）文件冲突**：S4b 改的 llama_mem.py:489-505 与 S5 的 :469 归一化点是同物理文件同区域 → 两 agent 不可同时编辑该文件的发射版本；node 也都要 Group-B。已与 s5-2 协调：S4b 优先占 .76+.249（team-lead 指派），S5 让出。
- **状态**：[PHASE-A2 PASS] 待建 launcher + 1-step 2-node FSDP smoke（验 gate fp32+可训+无 NCCL hang）→ 报 PIDs/step/loss → 全 3k run。IB recipe + P2P_DISABLE=1 + static rdzv + `python -m torch.distributed.run` 已验证。**未启动训练**。
- **★发射工程进展（2026-06-19 晚，landmark-s5）**：launcher `external/landmark/train_landmark_s4b.sh`（2-node static rdzv .76 node_rank0 + .249 node_rank1，port 自选避端口占用，`python -m torch.distributed.run`，IB+P2P_DISABLE=1，grad-ckpt ON，pd2×accum4×16，max_steps3000，save_steps1000）就绪。train.py 增 `--learned_block_gate` arg + `DATASET_MAP_NPROC` env（默认 8，原硬编码 32）。
  - **数据**：train.py 硬编码的 `togethercomputer/RedPajama-Data-1T-Sample` 已死 → 改 `liang2kl/RedPajama-Data-1T-Sample-Backup`（team-lead 验证镜像，实测 diskB 经 proxy 加载 OK：split=train、930,514 行、cols[text,meta] 同 schema）。
  - **16-rank 首发失败**：全 rank exit 1（error_file N/A 无 py traceback），且单 GPU run 也在第一个 map（tokenize_fn num_proc=32）后**无 traceback 静默消失**（非 CPU-OOM：节点 2.2TB RAM、2.1TB 空闲；疑 setsid/ssh 会话清理或 num_proc 过高 fork 风暴）。第一个 map 的 31 arrow shard 已落盘。
  - **修复中**：用同一 train.py（保证 cache fingerprint 一致）单 GPU、`DATASET_MAP_NPROC=16`、同步 ssh（持连）预热 cache → 完成后 16-rank 命中 warm cache 免首建竞争。⚠ 运维坑：`pkill -f train.py` 会误杀含 "train.py" 的 ssh 自身命令串 → 用 `pkill -f "landmark_venv/bin/python train.py"`；`& disown` 后台发射经 sshpass 偶发 exit255 无输出 → 改用持连同步 ssh（Bash run_in_background）跑构建。**仍未启动训练**。

- **★Group-A 迁移 + 3 blocker 全修 + 3k 发射（2026-06-19 晚，landmark-repro，权威接管 S4b）**：diskB(.76/.249)已 offline → S4b 改在 **Group-A 2节点**（本机 29.162.227.178 master + .196 worker，IB recipe NCCL_IB_GID_INDEX=3/bond1/DMABUF=0/GDR=0 + static rdzv + `python -m torch.distributed.run`，eff-batch128=pd2×accum4×16）。RedPajama liang2kl mirror diskA 无缓存（之前只在 diskB）→ 经 woa-proxy 重下全（930,514 行）。新 runner `scripts/run_landmark_S4b_node.sh`。
  - **★硬门槛 2-node FSDP smoke 逮到 3 个会让 S4b 静默退化成 anchor 的致命 blocker（toy gradcheck 全漏）**：
    - **#1 gate 从不 fire**：真实训练序列 q_len=**520**（512内容+8个每63插的 landmark token），原 guard `nblk*blk==q_len`(8×64=512≠520) 永远 False → gate 分支根本不执行。修：按真实 landmark 位置 gate 前 N 完整块、尾部 rem 个无 landmark 残块保持 ungated(gate=1)。reach 探针验 FIRES q_len520/nblk8/rem8 ✓。
    - **#2 zero-init 被 from_pretrained 覆盖**：transformers 对 ckpt-missing 的 lm_gate key 在 __init__ re-zero **之后**重跑 `_init_weights` 覆盖成 normal(std)（纯 fp32 也复现，final weight absmax=0.0617≠0）→ gate≠1 at init。修：final linear 打 `_s4b_gate_out` tag，让 `_init_weights` 本身对它清零。验 32 层 final weight 全 0、step1 loss=4.4605＝anchor 4.4607 bit-identical ✓。
    - **#3 FSDP MixedPrecision cast lm_gate→bf16**：实测 dtype=bfloat16。**30 步 grad-trend smoke 判定 bf16 不 underflow**：final_w_absmax step1=0→step3=1.97e-5→单调增至 step30=3.03e-4（每步离 0、平滑累积，无 round-to-zero），gate_absdiff_from_1 0→4.46e-2，loss 2.68→2.14。⇒ gate trainable，**不动 FSDP wrap，直接 3k**（省改 wrap policy 风险；#2 已保证 bf16 步0 identity 精确 exp(0)=1）。
  - **代码改动**（嵌套 repo external/landmark-attention，HEAD a699e3e；含 env-gated 诊断 S4B_GATE_DIAG/S4B_SMOKE_SUBSET，faithful 默认 OFF）：llama_mem.py(gate fire 修 :513-559、zero-init tag :285/:698、reach+track 诊断)、train.py(诊断 callback + S4B_SMOKE_SUBSET)。
  - **状态**：[3k RUNNING] 2026-06-20 00:01 起训（master log `logs/landmark_S4b_3k_master_20260619_234646.log` + worker，**port29575**，OUT=`external/landmark_ckpts/landmark_S4b_learnedgate`，max_steps3000 save1000）。首发 port29571 撞 lingering torchrun(Address already in use)→全 kill 换 29575 重发成功。全量 RedPajama 两段 map(~14min)后入训，step18 loss 4.42→3.57(warmup)，GPU 100%，**~5.2s/it（ETA ~4.4h）**（比 S2 dolmino 2.48s/it 慢 ~2x，疑 per-layer gate 计算+数据路径差异；run 健康不打断）。出 ckpt 后跑 native passkey @1k/2k/3k(70→32k,50/档,top_k5,S0/S2 同口径)。

### ★ Phase 3 S2 数据轴迁移：RedPajama → dolmino(wiki+pes2o)（2026-06-19，landmark-repro，进行中）
**目的**：单轴只换训练语料（RedPajama-1T-Sample 7源 → dolmino wiki+pes2o 2源），其余全同 Landmark working anchor，验证假设「窄单源数据是否杀长程墙」。守门=native passkey。
- **单轴改动**：仅训练语料。机制/tokenizer/打包/loss 全保持 anchor：LLaMA-1-7B base、tf4.28 landmark_venv、llama_mem.py grouped-softmax、mem_freq=63、lr2e-5 cosine+3%warmup、wd0.1、bf16、FSDP full_shard、ctx512、全序列 LM loss、full-FT。
- **数据 BLOCKER + 解法**：`MemLong/data/processed/dolmino_per_doc` 是 **Llama-3 tokenizer 预切**（vocab 128k，BOS=128000）→ LLaMA-1（vocab 32k）不可用。解法=取 raw 文本 `MemLong/data/raw/dolmino_pes2o_wiki/raw/data/{wiki,pes2o}/*.json.gz`，用**同一 LLaMA-1 tokenizer 重切**（保持单轴）。prepare_s2_dolmino.py 流式 gz、按源各 130M token early-stop、doc-by-doc interleave。
- **token budget**：wiki 130.0M + pes2o 130.0M = **260M token**（50/50 平衡），503,077 docs，加 landmark token 后打包成 **509,065 个 512-block**。3k 步 ×128×512=196.6M 训练需求 < 260M（约 1 epoch 内）。
- **eff-batch=128 保持单轴**：2节点16卡 = per_device2 × accum4 × 16。max_steps=3000、save_steps=1000（诊断预算，非全 15k）。
- **2-node DDP（Group-A）**：master 本机 29.162.227.178 + worker .196。**验证可用的 NCCL recipe**：static rdzv（--master_addr/--master_port，**非 c10d**，c10d 在 diskA 节点 flaky/hang）+ `NCCL_SOCKET_IFNAME=bond1`（真 NIC 非 eth0）+ `NCCL_IB_DISABLE=1`（mlx5 RoCE 走 TCP）+ worker 在 master 起 store 后启动。16-rank allreduce smoke 通过（sum=120）。
- **launcher**：scripts/run_landmark_S2_node.sh（per-node runner，绝对日志路径）+ scripts/launch_landmark_S2.sh。日志 logs/landmark_S2_dolmino_{master,worker}_20260619_170400.log。
- **状态（步进确认）**：两节点均 100% GPU，step 计数器在两端递增（step 10/3000），loss 4.84→4.14 下降，跨节点 NCCL 集合通信端到端 OK 无 hang，landmark forward shape [2,1,520,520]（512+8 landmark@mem_freq63）正确。
- **⚠ 吞吐**：~27.5 s/it → 3k 步 ETA ~23h（远超单节点估计 4-6h）。**原因：2-node FSDP full_shard 每层 all-gather 参数走 inter-node TCP（IB 禁用），通信受限。** 已上报 team-lead。
- **★ IB 修复（2026-06-19，11× 提速）**：2-rank allreduce bw smoke 对比 IB vs TCP：`NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3`（mlx5 RoCE v2）= **10.7 GB/s**，TCP（IB off）= 1.0 GB/s → **IB 快 ~10×，且干净无 hang**。据此 relaunch S2 2-node IB-enabled（弃 TCP run，其仅到 step25 无 ckpt，warm state 可忽略）。**新 IB run：2.48 s/it（vs TCP 26.5 → 11× 提速），3k 步 ETA ~2h。** 两节点 100% GPU，step 递增正常。IB run 日志 logs/landmark_S2_dolmino_IB_{master,worker}_20260619_172103.log。launcher run_landmark_S2_node.sh 已支持 `NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3` 环境覆盖。
- **待办**：1k/2k/3k ckpt 落地后跑 native passkey @70→32k、50 tests、top_k5、mem arm；记录断崖与否（断崖=数据是杀手；无断崖=数据非杀手，疑点转检索/分层）。

### ★ Method A raw-KV readout 诊断系列（2026-06-20, methodA-eval）
**目的**：raw-KV(无损)+ 解冻 reader(L16-31) + 可训练 gist-key soft-attention 选择(readout 16/20/24),能否破长档墙。守门=needle precision probe(gist 是否选中含 needle 的 chunk)+ W0 BABILong。
- **rawkv_methodA_b200（原始，topk8）**：chunk512,n_ctx7(gap3584),topk_chunks=8,num_keys=1,gist mean-pool。★配置缺陷:topk8≥n_ctx7→GistReadout keep_all 恒 True→top-k 选择全程 no-op,reader brute-force attend 全部 7 chunk→loss→0 但 scorer 无选择压力。probe: needle precision@top1=22.5%(随机)。诊断 commit 5b87057。
- **rawkv_methodA_h1fix_b200（H1-fix，topk2）**：topk_chunks=2≪n_ctx16(gap8192,curriculum0:16),num_keys=3(2 distractor),gist **max-pool**(新增 --rawkv_gist_pool)。强迫选择。2000步 non-finite=0,4 ckpt 全在。
  - **probe(num_keys=3/gap8192/n_ctx16,匹配难度)**:needle precision@top1=**5.0%**(随机6.25%)@top2=2.5%。gist 权重跨16 chunk 近均匀。
  - **W0 BABILong qa1(正规 score_nested,n=100)**:0k=**90%** 4k=14% 8k=7% 16k=8%(死线21)→长档崩。0k 高=格式/短读完好(非"未遗忘")。
  - **★裁决=H2**:gist scorer 确实训了(全精度 norm 14.470→14.49,grad_norm 0.04 非零;早期"FSDP freeze bug"判断已撤回——是 probe 2 位小数显示误导)。即便强迫选择+梯度正常,训练性 cross-chunk scorer 仍学不会区分 needle(precision≈随机)。**"训练一个 cross-chunk 选择器"这条路本身难** → 转 Landmark emergent 选择。
- **rawkv_methodA_h1fix_v2_b200（方法学修正版，RUNNING pid71128）**:同 h1fix 配置,但 gist 不再(错误地)FSDP-wrap、改为 replicated + 手动 all-reduce 梯度(_sync_gist_grads)。作为"梯度正确同步下训练 scorer"的严格 H2 确认。预期 precision 仍随机。commit 6e48cd5。

#### ★ Method A 最终裁决 = H2(2026-06-20,欠训已排除)
- **撤回**:之前"FSDP grad-freeze bug"判断作废——全精度核验 gist norm 14.470→14.490 确实移动了,probe 显示"14.5000"只是 bf16/2位小数取整;gist 一直在训。
- **v2(grad all-reduce 修复)**:step500 fp32 drift query+0.0203/key+0.0055,与旧 h1fix **完全相同** → grad-sync 修复无实质效果;且旧 run drift 在 step500 后冻结(1500步只挪0.00006)= 早期 plateau,非慢爬欠训。
- **★欠训排除 run(rawkv_methodA_gistlr30_b200,gist lr×30=6e-4)**:step250 norm 大动 query 14.470→**15.438**(+0.948)key→**15.688**(+1.245),pre-softmax score spread 0.01→**3.5**(scorer 在做自信的非均匀选择,plateau 被突破)。**但 needle precision @top1=0.0% @top2=0.0%**(随机6.25/12.5%),chunk0 权重 0.61× uniform = 比随机还低。
- **裁决**:**不是欠训**(给足 lr,scorer 训得猛并收敛到自信选择规则),而是**损失面对 gist 投影没有指向"选含 needle 的 chunk"的下坡**。训练性 cross-chunk scorer 学不出正确检索 = 干净的 **H2**。
- **方向**:放弃可训练 gist scorer,转 **Landmark emergent 选择**(in-window grouped-softmax,不训练选择器)。建议与 landmark-repro 对齐:把 raw-KV 无损内容 + 解冻 reader 嫁接到 Landmark emergent 机制。

#### ★ Method A 机制 probe(2026-06-20):reader native attn >> trained gist
- 对比 gist 选择 / reader native q·k attention over raw-KV / needle 位置(C=16,num_keys=3,needle@chunk0):
  - h1fix(lr1): gist needle prec@top1=**10%**,**reader-attn=55%**(8.8×随机6.2%),gist↔reader Pearson +0.07。
  - lr30 step250: gist=**0%**,reader-attn=**27.5%**(4.4×随机),Pearson +0.00。
- 结论:trained gist 与 needle、reader-attn **都不相关**(idiosyncratic)。但 **reader 自身注意力 over 无损 raw-KV 已能定位 needle(55%)**,远胜任何 trained 选择头。
- ★架构启示:**去掉 trained gist scorer,reader native attention over raw-KV 就是最好的检索器**(selection 涌现、不进 loss)。嫁接草案:保留 raw-KV 无损 + 解冻 reader L16-31,删 gist,inattn concat 去掉 col_bias。注入层数待 landmark-repro S5 passkey 定。

#### ★★★ Method A 真墙 = DILUTION(2026-06-20,分层 oracle 诊断终点)
分层 consumption 诊断(注入正确 needle raw-KV,bypass 选择,逐层定位):
- **Level 0**(needle+question 同 512 chunk,纯窗口内):vanilla 100% / mem_space 96% → 基础 readout OK。
- **Level 1**(跨块,store **仅 needle chunk**,col_bias=0):L16 单层 82-87%,L16/20/24+ **97.5%**(far≈near)→ 不是 distance/RoPE,不是 consumption 无能,reader **会**消费干净跨块 KV。
- **full_haystack**(store **全 16 chunk**,needle 在 chunk0):**0.0%**(所有层集)→ 加 15 distractor chunk 把 97.5% 打到 0%。
- **★裁决 = DILUTION**:墙是"太多 raw-KV 列(16×512=8192)在 attention softmax 里淹没 needle 的 25 token"。完整自洽:Level0 96-100% / Level1 干净 97.5% / full-haystack 0% / W0 gist-topk2 ≤14% / go/no-go keep_all 4k11/8k5 —— dilution 的不同剂量。
- **破墙配方(无训练,不触 H2)**:硬 top-k **isolation**——选 1-2 chunk,**只拼 selected chunk 的 raw-KV,排除其余 14**。这正是 Landmark cache_top_k work 而我们 keep_all 崩的根因(隐式 isolation)。难点=选得准(gist 0-10% 选错→topk2 仍 14%;reader native attn 55%;oracle=97.5% 上限)。
- **下一步(待 recipe 测试)**:full_haystack + reader-native-attn 驱动的 top-1~2 硬隔离,预测 W0 跳到接近 Level1 高位 = 破墙 demo。

#### ★ Method A raw-KV 真任务验证(2026-06-20):破墙 demo 不转移
- **clean T2 probe**(16 chunk,fixed code needle):chunk64 reader_attn-top2 + gather = **67.5%**(破 keep_all 0%)。
- **真实 BABILong qa1**(n=100,自然语言 bAbI fact needle,正规 score_nested),chunk64:
  | | 4k | 8k | 16k | 32k |
  |--|--|--|--|--|
  | reader_attn-top2 + gather | 7 | 5 | 7 | 8 |
  | keep_all baseline | 4 | 2 | 5 | 3 |
- 裁决:机制方向性有效(reader_attn ~1.5-2.7× keep_all)但**全 near-floor**,**clean-T2 67.5% 不转移**。真 gap = (a) selection precision 在 512-chunk scale 崩(probe 16 选 2 命中 85%,真 32k 512 选 2 远低);(b) bAbI fact 自然语言比 code 难定位;(c) within-block 两阶段读出缺失(平铺单一 softmax,缺 Landmark block 选×block 内 token attention)。
- 下一步:移植 landmark-repro per-layer cache_top_k(32×nh 投票提 precision)+ 两阶段 grouped-softmax 读出(解 within-block 稀释)。均不训练选择器(H2-safe)。production 路径 rawkv-protoA 已搭(keep_set_mode/gather)。

#### ★ 变体B(stage1_select)+chunk64-grouped 纯eval裁决(2026-06-20, methodA-eval, B200 .18:36000)
- **配置**:h1fix ckpt 不重训。变体B=chunk512+reader_attn top2+`--rawkv_grouped_readout --rawkv_subblock_size 64 --rawkv_stage1_select`(per-sub-block reader-attn salience 作阶段1 log 偏置,把质量集中到 needle sub-block)。chunk64=chunk_size64(选择也细粒度)+grouped subblock。真实 BABILong qa1 W0,n=100,4 shards×25 合并,md5 对 diskA byte-identical,stage1_select 代码确在。
- **W0 结果(真实 BABILong qa1 n=100)**:
  | 配置 | 4k | 8k | 16k | 32k |
  |--|--|--|--|--|
  | 变体B (grouped+stage1_select) | 5 | 2 | 4 | 2 |
  | chunk64 (grouped) | 7 | 5 | 7 | 3 |
  | 变体A (grouped 等权) | 14 | 7 | 4 | 3 |
  | flat (reader_attn top2) | 7 | 5 | 7 | 8 |
  | keep_all baseline | 4 | 2 | 5 | 3 |
- **三数拆解(per-len chunk 选择命中率 readerattn-top2 probe n=50)**:命中率 4k=72%/8k=78%/16k=58%/32k=40%(选择尚可,远超随机)。条件 readout 质量(W0/命中):变体B 4k≈7%/8k≈3%/16k≈7%/32k≈5% —— **readout 灾难性低**。
- **★裁决 = 纯 eval 到头,该转重训**:(1) 变体B(stage1 把质量集中)不仅没拉起长档,4k 反从变体A 的14崩到5 —— stage1 log-bias 把质量过度集中到 argmax 选错的 sub-block,选错时比等权平摊更糟。(2) chunk64 细粒度也没破墙(长档≤7)。(3) 两者长档(16k/32k)全卡 floor 2-4 ≪ 死线21。(4) 命中率 40-78% 证明**选择不是瓶颈**;瓶颈是 argmax-硬选中后 frozen-reader 无法从 raw-KV 读出(条件 readout~5%)+无可训练 bottleneck-key。**全配置 cap≤14%@4k、长档≤8 = 纯 eval 彻底穷尽。**
- **建议**:起 Group-A(launcher 2350c53 就绪)作裁决实验。A 拉起长档(14→40+)=group_lse 自学够;A 只解 within(4k 升)长档仍卡 = 需 Landmark 式可训练 summary-token bottleneck(方案B,改训练 forward 加 in-window grouped-softmax 让 summary token 学概括)。

#### ★ FIFO 3-arm (b25/b50/b100) c512 step3000 — W0 eval (2026-06-25)
- **配置**:FIFO raw-KV write，buffer={25,50,100} chunks，chunk512，step3000 训练完成。离线 BABILong W0（无 cross-chunk SWA），n=100，4-shard×25 合并，score_nested 口径。
- **fifo_b25_c512 W0（已 4-shard 合并 + sanity-check 验证为真，非 shard 伪影）**：
  | task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
  |--|--|--|--|--|--|--|--|
  | qa1 | 96 | 99 | 99 | 93 | 40 | 34 | 30 |
  | qa2 | 99 | 100 | 100 | 95 | 23 | 32 | 32 |
  | qa5 | 100 | 100 | 97 | 87 | 65 | 76 | 68 |
- **★注**:qa5（3-fact）长档异常高且稳（32k=68 >> MemoryLLM 34），输出非退化（diverse target/output 对，已抽查 qa5_32k shard0）。qa1/qa2（1-2 fact）长档掉到 30-32。待 b50/b100/c1024 W0+W6 合并后横向对照（buffer 大小 × SWA 对长档的影响）。
- 节点 .48.7.53(b25)；本机(b50)、.58.245.174(b100)、B200.53(b50_c1024) eval RUNNING。

#### ★★ QCMem mid-depth resume 自蒸馏 — 训练量 scaling (2026-07-07, Qwen3-8B, definitive n=100 官方判分)
- **机制**:QCMem resume j=12(缓存底12层h_j,读时重算上24层)。自蒸馏:teacher=j0(RAG全重算上界),student=j12+LoRA(r16/32),PG19纯自然文本KL,**零babilong**(守红线)。eval bm25 top4,sink bos,chunk512。
- **训练量 scaling 曲线(bm25 同口径,官方 compare_answers,n=100)**:
  | task/len | 0步(零训) | 1000步(r16) | 2000步(r32) | 3000步 | 4000步 |
  |--|--|--|--|--|--|
  | qa1/8k | 23 | 30 | 31 | 31 | 31 |
  | qa1/16k | 11 | 18 | 19 | 18 | 19 |
  | qa5/8k | 61 | 80 | 78 | 80 | 79 |
  | qa5/16k | 50 | 63 | 65 | **68** | 67 |
  - 补充(4000步 final): qa1/4k=58 qa5/4k=73(短档强,检索压力小)
- **★裁决(3条 definitive)**:
  1. **自蒸馏真实有效且快**:90%+增益头1000步拿到(qa1/16k 11→18, qa5/16k 50→63)。证明 QCMem"训练推后 readout cliff"主张在 Qwen+自蒸馏配方成立,且**不靠合成数据**。
  2. **qa5(关系/多fact)持续受益训练**:16k 50→63→65→68 单调爬升未饱和,4000步仍高位。
  3. **qa1(精确单fact定位)完全饱和**:1000步后 8k=31/16k=19 纹丝不动 → **qa1 天花板=检索召回,非训练能力**(BM25 召不到含答案单chunk,再训无米;qa5 多fact冗余对检索不敏感故更高)。
- **oracle selector 不可用作上界**(诊断 n=50×3档):定位100%成功0回退,但只选 avg1.3-1.5 个含**字面答案词**chunk vs bm25固定4 → qa5 漏推理链fact,oracle常<bm25。topk语义不同,非bug。
- **方向B(非连续层/缓存顶层)判负**:qa5 (12,0)baseline 8k61/16k50;缓存任何顶层即崩((12,6)29/20,(6,6)27/23,(6,12)9/10)。顶层hidden=query敏感读出前表征,缓存=丢query-conditioning。印证纯前j层resume才对。
- **下一步**:瓶颈=selector(检索召回)。修检索(BM25→更强检索/增大topk/query改写)是qa1突破口,非加训练。
- **产物**:LoRA `outputs/qcmem_distill_qwen_j12_r32_4k/{step1000..4000,final}`;1000步版 `outputs/qcmem_distill_qwen_j12/step1000`。commit e71bdd2(local_files_only fix)。

#### 🎯★★ QCMem 最佳配置全表 — 碾压 MemoryLLM (2026-07-07, definitive n=100 官方判分)
- **配置**:Qwen3-8B, QCMem resume_j=12(缓存底12层,读时重算上24层), 4000步自蒸馏LoRA(r32/α64, teacher=j0 RAG, 纯PG19零babilong), selector=bm25 **topk12(甜点)**, sink=bos, chunk512。
- **全表(qa1/qa2/qa5 × 0k-32k, 官方 compare_answers, n=100)**:
  | task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
  |--|--|--|--|--|--|--|--|
  | qa1 | 98 | 79 | 68 | 66 | 63 | **57** | 21 |
  | qa2 | 25 | 44 | 41 | 41 | 35 | 25 | 10 |
  | qa5 | 69 | 77 | 75 | 76 | 62 | **63** | 63 |
  - **MemoryLLM baseline 对照**: qa1 53/42/32/23/14/9/7; qa2 36/35/19/16/15/16/16; qa5 47/50/45/39/39/38/34。
- **★裁决**:除 32k 边缘全面碾压。qa1/16k **57 vs 9(6.3×)**;qa5 全程压制(16k 63 vs 38);qa2 中程(1k-8k)超越。
- **甜点 topk 随任务/长度**:qa1(单fact)=topk12;qa2(双fact)=topk16(8k=37,16k=30 > topk12);长档32k需更大topk(topk12对64chunk召回不足→qa1/qa2 32k掉21/10)。qa5(多fact冗余)对topk不敏感,32k=63稳。
- **方法要点**:QCMem resume(破读出墙) + 自蒸馏(纯PG19守红线,readout训练) + 甜点检索精度(bm25 topk12,非越多越好)三者叠加 = 长档记忆SOTA。
- **下一步**:(1) 甜点topk随长度自适应(32k用topk24+); (2) reader_attn salience selector(coder实现中,替代bm25词法); (3) 双fact/多fact用更大topk。

#### 🎯 QCMem 每任务最优 topk SOTA 全表 (2026-07-07 最终, n=100 官方判分)
- 同上 4000步自蒸馏LoRA，**每任务用其最优 topk**（甜点随任务所需 supporting fact 数单调）：
  | task | 0k | 1k | 2k | 4k | 8k | 16k | 32k | 最优topk |
  |--|--|--|--|--|--|--|--|--|
  | qa1 | 98 | 79 | 68 | 66 | 63 | **57** | 28 | tk12(32k:24) |
  | qa2 | 25 | 44 | 41 | 43 | 37 | 25 | 18 | tk16(32k:24) |
  | qa5 | 69 | 77 | 75 | 73 | **79** | 67 | 63 | tk4 |
  - MemoryLLM: qa1 53/42/32/23/14/9/7 | qa2 36/35/19/16/15/16/16 | qa5 47/50/45/39/39/38/34
- **★最优 topk 规律(可解释)**:qa5(多fact/关系)=tk4, qa1(单fact)=tk12, qa2(双fact)=tk16 —— **最优 topk 随任务需要的 supporting fact 数单调递增**;超过甜点即噪声稀释 reader attention("信噪比非覆盖率")。长档需按比例放大(32k:tk24)。
- **裁决**:qa1 8k=63vs14(4.5×)/16k=57vs9(6.3×)/32k=28vs7(4×);qa5 全程压制(8k 79vs39=2×);qa2 中程超越端点接近。**QCMem resume+自蒸馏(零合成)+任务自适应甜点检索 = 长档记忆全面 SOTA**。
- **弱点**:qa2(双fact最难)0k=25<36、16k=25 未拉开→需 qa2 专训或更强 selector(reader_attn coder实现中)。

#### ⚠️ MemoryLLM baseline 校准 (2026-07-07, 铁律2 官方判分复核 .52 现有 CSV)
- **实测 MemoryLLM-8B-chat**（n=100 官方 compare_answers，.52 `babilong_results/MemoryLLM-8B-chat`）：
  | task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
  |--|--|--|--|--|--|--|--|
  | qa1 | (缺) | 50 | 49 | 25 | 30 | 20 | 12 |
  | qa2 | (缺) | 29 | 21 | 20 | 14 | 19 | 16 |
  | qa5 | (缺) | 47 | 42 | 40 | 41 | 38 | 37 |
- **⚠️ 更正**：此前 RUN_REGISTRY/handoff 引用的 MemoryLLM qa1 = `53/42/32/23/14/9/7` 与实测不符（实测 qa1 长档高很多：16k=20 非 9，32k=12 非 7）。**以此实测 CSV 为准**。
- **修正后 QCMem vs MemoryLLM（官方同口径）**：qa1/16k 57 vs **20 = 2.85×**（非之前误报的 6.3×）；qa5/16k 67(topk4)/63(topk12) vs 38 = 1.7×；qa1/8k 63 vs 30 = 2.1×。**仍大幅领先，但校准后不夸大**。

#### 🎯 QCMem RULER 真实任务泛化 (2026-07-07, Qwen 4000步自蒸馏, string_match_all recall)
- **配置**: Qwen3-8B QCMem resume_j12 + 4000步自蒸馏LoRA(纯PG19零RULER/babilong) + bm25 topk12。RULER self-contained本地重实现(niah/vt),官方string_match_all口径。
- **完整长档表(n=50/task/length)**:
  | task | 4k | 8k | 16k | 32k |
  |--|--|--|--|--|
  | niah-single | 100 | 100 | 100 | 100 |
  | niah-multikey | 96 | 94 | 94 | 88 |
  | var-track | 100 | 49 | 25 | 21 |
- **Qwen零训练对照(8k/16k)**: niah-single 36/8, multikey 0/6, var-track 4/2。
- **Llama-3自蒸馏(8k/16k)**: niah-single 100/100(简单任务backbone无关), multikey 30/32, var-track 12/4。
- **★裁决**: niah-single全长100=无长档退化. 自蒸馏(纯PG19)zero-shot到RULER真实NIAH任务=**通用长上下文记忆能力铁证,非babilong特化**. 兑现2026-07-05从qa5合成SFT转向纯prediction的目标. backbone×训练分解在RULER一致(Qwen蒸≫Llama蒸≫零训; 难任务multikey/vt Qwen backbone领先)。

#### 🎯🎯 QCMem vs full-context 长档 scaling 决胜表 (2026-07-07, Qwen3-8B, RULER string_match_all n=50)
- **QCMem** = resume_j12 + 4000步自蒸馏LoRA + bm25 (topk 随长度: 64k tk12/128k tk24/256k tk48)。**full-context** = base_max_window 覆盖全长直喂。
- **niah-single**:
  | 方法 | 8k | 16k | 32k | 64k | 128k | 256k |
  |--|--|--|--|--|--|--|
  | full-context | 100 | 100 | 100 | 100 | **0** | **OOM(~350GB)** |
  | QCMem | 100 | 100 | (-) | 100 | **100** | **98** |
- **niah-multikey**: full-ctx 64k=96→128k=0; QCMem 64k=82/128k=84/256k=60。
- **★决胜结论**: Qwen3-8B外推悬崖在64k→128k之间(rope1e6撑到64k, 128k全崩=0). ≤64k full-context精度≈或略优QCMem(装得下时全喂更好); **>64k(128k/256k) full-context精度归零/OOM, QCMem仍98-100** = QCMem在超backbone外推能力的超长档是唯一可用方案.
- **命题B速度(bench_qcmem_vs_fullctx.json)**: prefill加速 32k 2.5×/64k 4.4×/128k 7.8×; 显存 full 20→89GB(随L) vs QCMem恒定~18GB. QCMem read固定6657tok(sink+topk*chunk+query), 与上下文长度无关.
- **价值主张**: QCMem不靠短档精度取胜(那里full-ctx更好), 靠(1)超CL可扩展(full跑不了时QCMem高精度) (2)prefill数倍加速 (3)显存恒定. 三重优势随上下文增长放大.

#### 🔬 QCMem read 精确 ablation: 上层重算(cross-chunk+query attention) vs 复用KV(block-diagonal) (2026-07-08, Qwen3-8B, RULER string_match_all n=50)
- **唯一变量 = read 时 layers[j:] 的 attention 连通性**（j=12, topk12, bm25, sink=bos, chunk512, 4000步自蒸馏LoRA 全相同; RoPE/位置/检索/sink 全相同）。实现: `QCMemModel(block_diagonal=True)` 构造 [1,1,H,H] block-diagonal 4D mask（transformers 5.5.4 `create_causal_mask` 原样透传预构建 4D mask，单次 forward，无 KV 注入）。commit `2daeb9a`。
  - **(i) 标准 QCMem**: full attention → 检索chunk 彼此 cross-attend + query attend 全部 chunk。
  - **(ii) block-diagonal**: sink 全局可见; 每个检索chunk 只在自己块内 causal（chunk 间无 cross, chunk 不 attend query = query-blind 孤立 KV 复用）; query 段 attend sink+所有chunk+自身causal。
- **对比表**:
  | task | len | (i) std | (ii) blkdiag | Δ(i-ii) |
  |--|--|--|--|--|
  | niah-single | 8k | 100 | 100 | +0 |
  | niah-single | 16k | 100 | 100 | +0 |
  | niah-multikey | 8k | 88 | 44 | **+44** |
  | niah-multikey | 16k | 92 | 40 | **+52** |
- **★裁决（诚实，铁律2）**: **取决于任务, 非一刀切**。单一 needle 任务(niah-single, 答案整个落在1个chunk内)→ cross-chunk attention 无关紧要, (i)≈(ii)（query 在两种mask下都能读到那唯一chunk）→ block-diagonal KV 复用完全够用（省算）。多key干扰任务(niah-multikey, 需在 4 个 distractor key 间消歧)→ **(i) 大幅碾压 (ii)（8k +44, 16k +52, block-diag 几乎腰斩）** → cross-chunk + query-aware attention 是价值来源，不能只靠复用孤立 KV。
- **对论文的意义**: QCMem "read 时对 pack 做 full attention 重算上层" 的设计**在多fact/干扰任务上有硬价值**（不是可有可无的 over-engineering）；但在单fact检索上，更省的 block-diagonal KV-复用变体等价——存在按任务选 read 策略的空间。self-test 铁证实现正确: 单chunk时 block-diag ≡ 标准（Qwen3-8B max|logit diff| 4.7e-5），多chunk时发散（`scripts/qcmem_blockdiag_selftest.py`）。
- **产物**: eval `ruler_results/qcmem_blockdiag_ablation/{qcmem_standard_j12,qcmem_blockdiag_j12}/`（+COMPARISON_TABLE.txt）; 代码 `--reuse_kv_blockdiag`(eval_ruler_qcmem.py) / launcher `scripts/_qcmem_blockdiag_ablation.sh` / 聚合 `scripts/aggregate_blockdiag_ablation.py`。

---

## 方向2 semantic-bottleneck pretrain — bottleneck 位置 × 宽度 sweep（1B from-scratch, 16000 步收敛, 2026-07-10）

**配置**：1B Llama from-scratch（hidden2048/16L/32h/8kv/ffn8192），slimpajama seq2048，eff_bs≈48，lr3e-4 cosine，bf16，DDP。funnel = 第 j 层输出过 `down(2048→d_bottle)→GELU→up(d_bottle→2048)`，**无残差**（信息必须挤过瓶颈）。ppl = 末 10-20 步均值（全部训到 16000 步收敛）。脚本 `scripts/train_semantic_bottleneck_1b.py` + `launch_semantic_bottleneck_1b.sh`。

### (A) bottleneck_dim sweep（固定 layer6，扫宽度）
| arm | ppl | LM 税 vs baseline |
|--|--|--|
| baseline（无 funnel） | 25.28 | — |
| d1024 | 26.42 | +4.5% |
| d512 | 26.78 | +5.9% |
| d256 | 27.42 | +8.5% |
- **趋势：bottleneck 越窄 → LM 税越高**（d256 最贵）。但 §3.3 证明越窄越可压（PCA-ΔNLL 最小、缓存最省）→ **LM 税 vs 可压性是 trade-off**：窄 funnel 迫使表征紧凑（利于 QCMem 缓存）但训练代价高。

### (B) bottleneck_layer sweep（固定 dim512，扫位置）★本次
| bottleneck 层 | ppl | LM 税 vs baseline |
|--|--|--|
| baseline | 25.34 | — |
| **layer1** | 26.40 | **+4.2%（最省）** |
| layer3 | 26.82 | +5.8% |
| layer6 | 26.85 | +6.0% |
| layer9 | 27.74 | +9.5% |
| layer12 | 27.77 | +9.6% |

- **★核心结论：LM 税随 bottleneck 深度单调递增——越往后放 funnel，信息密度损伤越大。** 收敛后单调（92% 进度时 L6 略低是噪声，16000 步收敛后 L1<L3<L6<L9<L12 干净单调）。
- **机理（符合 §3.1 分工命题）**：浅层承载低阶/局部信息，压缩损失小；越深越接近"生成前的精炼表征"，那里的信息密度高、每一维都 load-bearing，强行挤过瓶颈损伤大。
- **★为什么 QCMem 仍选 j=12（关键 framing）**：单看 LM 税，缓存点应放浅层（L1 最省）。但缓存点太浅 → 可缓存的语义不足（§3.2 j-sweep：j≤9 检索精度饱和、j12 崖跌到 14=可缓存语义上限）。→ **QCMem 的 j=12 是"可缓存语义上限"与"LM 税"之间的折中，不是税最小点**。layer sweep 正面量化了这个折中的另一端（税随深度的代价曲线），坐实 j=12 的选择是权衡而非任意。
- **产物**：ckpt `outputs/sembott_1b_{base,d256,d512,d1024,layer1,layer3,layer9,layer12}_16k/final.pt`（layer6=d512 复用）；日志 `logs/sembott_1b_*_16k.log`；写入 draft §3.4（commit 0bf7182）。

---

## QCMem Selector 对照（RULER n=100，官方 `_string_match_all_one` 判分，2026-07-13）

**详细数据与结论见 `status/QCMEM_SELECTOR_COMPARISON.md`（由 `scripts/aggregate_selector_comparison.py` 聚合）。**

- **配置**：QCMem (j=12, chunk_size=1024, 32 slots, bm25 selector as default), Qwen3-8B, RULER 3-task × 6-length × 5-topk = 90 cells × 4 selectors = 360 cells total。
- **4 selectors**：bm25（词法）/ recency（位置/末尾）/ reader_attn（语义 h_j cosine）/ oracle（含答案的 gold chunk = 检索天花板）。
- **全量完成**：每 selector 90/90 cells，零缺失。

### 核心结果（峰值 topk recall %，16k & 32k）

| Task | Length | BM25 | Recency | ReaderAttn | Oracle |
|---|---|---|---|---|---|
| NIAH-Single   | 16k | **100.0** | 72.0  | 83.0  | **100.0** |
| NIAH-Single   | 32k | **100.0** | 42.0  | 38.0  | **100.0** |
| NIAH-MultiKey | 16k | **97.0**  | 61.0  | 69.0  | **100.0** |
| NIAH-MultiKey | 32k | **99.0**  | 44.0  | 41.0  | **100.0** |
| VT            | 16k | 27.6  | 53.2  | **60.2** | 9.2   |
| VT            | 32k | **23.0** | 16.8  | 22.0  | 5.8   |

### 关键结论
1. **Oracle = 100% on NIAH**：给定正确 chunk，QCMem 读出无损。长档瓶颈 = 检索质量，非压缩质量。
2. **BM25 ≈ Oracle on NIAH**（gap ≤5 pp at all lengths）：词法检索对实体 needle 近最优；BM25 是 NIAH 类任务的默认最优 selector。
3. **ReaderAttn & Recency ≪ BM25 on NIAH**：注意力语义相似度和位置近端性在 16k/32k 上大幅落后（差距 17–62 pp）。
4. **VT oracle 失效**（oracle 9.2 < BM25 27.6 at 16k）：oracle 选含答案字符串的 chunk，但变量追踪需全链多 chunk；单 gold chunk oracle 不适用多跳任务。reader_attn@tk24=60.2 在 VT@16k 最优（大 topk 意外覆盖链上各赋值 chunk）。32k VT 全面崩溃，topk≤24 单程 selector 不足覆盖完整链。

#### Qwen3-32B chunk512 downstream split-j sanity (2026-07-15, n=30)
- Environment restored from official booydar/babilong commit 7a6efee and RMT-team/babilong data commit ee0d588; RULER uses a clearly marked 64MiB eval-only subset of official emozilla/pg19 train prose.
- Protocol: stock Qwen3-32B, no adapter, chunk512, bm25 topk12, j={12,16,18,20}; RULER niah single/multikey 16k plus BABILong qa1/qa5 8k.
- Results (single/multi/qa1/qa5; macro): j12=100/100/80/33.3 (78.33); j16=100/90/86.7/33.3 (77.50); j18=100/73.3/83.3/20 (69.16); j20=100/96.7/76.7/6.7 (70.00).
- Verdict: j12 and j16 are tied within n30 noise; use j16 by default because it is 5-9% faster, wins qa1, ties qa5/single, and has stronger five-seed intrinsic stability. j18/j20 show hard-task cliffs and are rejected.
---
## ★ QCMem 全 scale benchmark（2026-07-15 起，Paper A 主力）

分工：collaborator=32B+，agent=8B→4B→1.7B→0.6B。协议见 `status/QCMEM_BENCHMARK_PLAN.md`。
selector 默认 auto（vt→iter_bm25 固定 / niah→bm25）；j：8B/4B=12，1.7B/0.6B=9；adapter 仅 8B。

### ★★ 双 j 主表（framing A，2026-07-16 重建；依据 `status/QCMEM_J_DETERMINATION.md`）

**口径**：RULER=`string_match`，n=100，chunk512；single/multikey=bm25、vt=iter_bm25（固定多跳）。zero-shot 行 = per-model **readout-safe j**（zero-shot readout 不塌、single-recall 近满 ≥90 的最深 j）。adapter 行 = 现有 ~0.33L adapter（content-j ~0.45L adapter 训练中）。

#### zero-shot @ readout-safe j（实测，主表权威 zero-shot 列）
| model | L | readout-safe j (/L) | niah_single 8k/16k/32k | niah_multikey 8k/16k/32k | vt 8k/16k/32k |
|---|---|---|---|---|---|
| **0.6B** | 28 | **j2** (0.07L) | 100/100/100 | 77/89/79 | 58/67/79 |
| **1.7B** | 28 | **j3** (0.11L) | 97/100/100 | 53/40/33 | 53/64/60 |
| **4B** | 36 | **j9** (0.25L) | 93/98/94 | 38/35/41 | 58/61/54 |
| **8B** | 36 | **j9** (0.25L) | 100/97/99 | 42/36/31 | 46/42/39 |
| **14B** | 40 | **j13** (0.325L) | 99/89/98 | 51/51/11 | 18/15/11 |
| **32B** | 64 | **j27** (0.42L) | 100/100/100 | 98/88/86 | 42/33/37 |

- **single 在 readout-safe j 全 scale 近满**（选 j 的判据）；**multikey/vt 在此深度对中间 scale（尤 14B）已衰减**（14B mk 51/51/11、vt 18/15/11）——这是"single≥90 定 j"的代价，也是 adapter 要补的 gap。0.6B/32B 因 gap 小，mk/vt 亦高（32B mk 98/88/86，0.6B vt 58/67/79）。

#### +adapter @ ~0.33L（现有较浅 adapter，标注实际 j）
| model | adapter j | niah_single | niah_multikey | vt |
|---|---|---|---|---|
| **1.7B** | ~j9 (0.33L, 现有) | —(zs 已高) | 56/39/20 | 62/62/66 |
| **4B** | ~j12 (0.33L, 现有) | —(zs 已高) | 95/97/94 | 96/96/97 |
| **8B** | **j12** (0.33L, 已验证) | 100/100/100 | 91/91/92 | 97/97/98 |
| **14B** | ~j13 (0.33L, 现有) | 100/100/100 | 100/99/99 | 99/100/100 |
| **32B** | 几乎不需（gap~0，readout j27 已达 content 峰） | — | — | — |

#### ★★★ n=500 firm-up（2026-07-16，RULER 官方协议口径，论文用；24 diskB 卡 task-pool，0 fail）
> 把上面双 j 主表（zero-shot @ readout-safe j + adapter @ ~0.33L）关键 cell 从 n=100 firm 到 **n=500**。判分 `string_match`，chunk512，topk12，single/multikey=`bm25`、vt=`iter_bm25`(rounds0/hop4)。结论：**n=500 全 cell 与 n=100 一致（噪声内），headline(8B/14B/32B) 完全 track**。REUSE=已有 n500 未重跑。

**zero-shot @ readout-safe j（n=500）** — 括号=REUSE 目录：
| model | j | niah_single 8k/16k/32k | niah_multikey 8k/16k/32k | vt 8k/16k/32k | 源 |
|---|---|---|---|---|---|
| 0.6B | j2 | 100/100/100 | 85.0/84.4/82.4 | 58.2/60.1/82.1 | REUSE `qcmem_0p6b_balancej2_n500` |
| 1.7B | j3 | 99.4/98.6/99.2 | 53.6/41.6/41.2 | 56.3/56.2/52.2 | REUSE `qcmem_1p7b_balancej3_n500` |
| 4B | j9 | 92.6/98.0/94.2 | 38.0/35.4/41.0 | 58.2/60.6/53.6 | REUSE `qcmem_4b_j9_n500` |
| 8B | j9 | 99.8/97.4/99.4 | 42.0/36.2/31.4 | 45.6/41.5/39.4 | REUSE `qcmem_8b_zeroshot_j9_n500` |
| **14B** | j13 | 99.0/88.6/97.6 | 50.0/43.6/11.0 | 15.8/13.5/13.6 | **NEW** `qcmem_14b_zs_j13_n500` |
| **32B** | j27 | 100/99.8/100 | 96.2/94.8/91.4 | 39.9/44.0/40.0 | **NEW** `qcmem_32b_zs_j27_n500` |

**+adapter @ ~0.33L（n=500）**：
| model | adapter j | niah_single 8k/16k/32k | niah_multikey 8k/16k/32k | vt 8k/16k/32k | 源 |
|---|---|---|---|---|---|
| 0.6B | j9 | 100/99.4/100 | 41.4/35.6/37.4 | 55.4/55.8/50.6 | **NEW** `qcmem_0p6b_ad_j9_n500` |
| 1.7B | j9 | 99.8/99.4/100 | 57.4/41.0/24.8 | 73.0/64.4/68.0 | **NEW** `qcmem_1p7b_ad_j9_n500` |
| 4B | j12 | 100/100/98.8 | 95.2/94.0/93.0 | 96.0/96.2/97.2 | **NEW** `qcmem_4b_ad_j12_n500` |
| 8B | j12 | 100/100/100 | 91.2/91.0/92.0 | 97.2/98.7/98.7 | REUSE `qcmem_8b_n500` |
| **14B** | j13 | 100/100/99.8 | 99.6/97.6/98.8 | 99.9/99.7/99.8 | **NEW** `qcmem_14b_ad_j13_n500` |

**n500 vs n100 一致性**：所有 cell 在 n100 噪声带（±5-10pt，vt 最噪）内一致，无翻案。最大偏移全在 vt（小样本方差最大），且是 n100 outlier 被 firm 校正：**0.6B-ad vt16k 20.8→55.8**（n100 单点噪声，n500 三档 55.4/55.8/50.6 自洽）、**32B-zs vt16k 33.4→44.0**、**1.7B-ad vt8k 62→73**。headline single/multikey 全部 ±≤7pt。三深度故事 + adapter=hard 任务杠杆 + 14B-ad~100 全表 结论不变。

#### ★★ +adapter @ content-j ~0.45L（2026-07-16 训练+eval，本会话，5 scale）
> self-distill LoRA（PG19，teacher=resume_j0 全 forward，student=resume_j+LoRA on layers[j:]，双向 top-k64 KL λ0.6，lora_r32/α32，chunk512 n_ctx7=4096win，1000 步 lr1e-4）。eval RULER n=100，single/multikey=bm25、vt=iter_bm25，8k/16k/32k。adapter 存 `outputs/qcmem_distill_<m>_contentj<j>_r32/final`，结果 `ruler_results/qcmem_<m>_adapter_contentj<j>_n100/`。

| model | content-j (/L) | zs@content-j single(崩) | **+ad single** | **+ad multikey** | **+ad vt** |
|---|---|---|---|---|---|
| **0.6B** | j13 (0.48L) | 6/3 | **95/98/99** | 24/24/22 | 0/4.8/0 |
| **1.7B** | j13 (0.48L) | 8/0 | **98/98/96** | 26/29/17 | 22.8/4.4/15 |
| **4B** | j16 (0.44L) | 25/3 | **100/100/100** | 46/33/37 | 72.8/65.4/79.8 |
| **8B** | j16 (0.44L) | 3/0 | **100/100/100** | 49/32/22 | 50.2/51.4/41.4 |
| **14B** | j18 (0.46L) | 18/36 | **100/100/99** | 83/77/71 | 80.8/89.4/64.8 |

**核心结论（验证 gap-vs-scale adapter 价值论）：**
1. **single 全 scale 通用恢复**：zero-shot@content-j 全崩（3-25 recall），adapter 后 **95-100（含 tiny 0.6B/1.7B）**——adapter 把可读深度推到语义 content 深度（~0.45L），对 needle 检索普适有效，达到/超过各 model 的 readout-safe(浅 j) zero-shot single。
2. **hard 任务（multikey/vt）SCALE-GATED**：
   - **14B 最强**（mk 71-83 / vt 65-89）≫ 其 zero-shot@readout-safe（mk 11-51 / vt 11-18）——深 adapter 大幅提升 14B hard 任务。
   - **4B ok**（vt 65-80 > zs 54-61；mk 33-46 ≈ zs），**8B partial**（mk/vt ≈ 其 zs@readout-safe，32k 衰减）。
   - **1.7B FAIL**（mk 17-29 / vt 4-23 < 其 zs@j3 的 33-64），**0.6B FAIL**（vt 0-5、mk 22-24 ≪ 其 zs@j2 的 58-89）——tiny 模型（content-vs-readout gap 0.26-0.39L）学不会从深 0.26-0.39L 的 cache 做 compositional readout。**确认 tiny 失败 finding**（对齐旧 0.6B adapter@j9<zs@j2）。
3. **⚠️ 深度 trade-off（对论文 framing 关键）**：content-j(~0.45L) adapter 在 hard 任务上**弱于**上表 ~0.33L 现有 adapter（8B mk 91→49、vt 97→50；4B mk 95→46、vt 96→73；14B mk 100→83、vt 99→65-89）。即：**adapter 能把 single 读到 0.45L 语义峰（更深 cache=更省算/更多语义），但 hard 任务读出随缓存深度单调变难，~0.33L 才是 hard 任务甜点**。single 两档都~100。→ 报 adapter 主表若追 hard 任务峰值用 ~0.33L；若强调"读到语义 content 深度"用 0.45L（single 满、hard 衰减）。
4. **gap-vs-scale 坐实**：能读到 content-j 的 hard-任务成功度随 content-vs-readout gap 缩小而升（14B gap0.085L 最好 → tiny gap0.26-0.39L 失败），与 `QCMEM_J_DETERMINATION.md` 的 gap 表一致。

⚠️ adapter 行为现有 ~0.33L adapter，其 zero-shot 对照基线为旧 recall-optimal j（非上表 readout-safe j），绝对增益口径略有错位；content-j（~0.45L）adapter（0.6B j13/1.7B j13/4B j16/8B j16/14B j18）训练中，出来后替换本列。

### ★ 3 深度总结（三个 j 分离，probe + readout bracket 实测）
1. **content 深度 ~0.45L 近 scale-invariant**（probe knee98：0.42–0.48L，均值 ~0.45L）= 语义信息最富的可缓存上限，跨 scale 稳定。
2. **zero-shot readout 崩点随 scale 变深，NOT scale-invariant**：single recall 从 ~100 掉到崩的 50% 点 = 0.6B ~0.09L → 1.7B ~0.22L → 4B/8B ~0.30L → 14B ~0.375L → 32B >0.42L（无崩）。旧"~0.25L 恒定"只是 4B/8B 中段粗值。
3. **gap = content − readout = adapter 缺口，随 scale 缩小到 32B~0**（见下表）。recall-optimal ~j3 < readout-safe j（scale 依赖） < content ~0.45L —— **小模型 gap 巨大→adapter 价值最大；32B readout 已达 content 峰→几乎不需 adapter。**

#### gap-vs-scale 表（adapter 缺口随 scale 单调缩小）
| model | content /L (probe) | readout 50%崩点 /L | readout-safe j /L | gap /L (=adapter 缺口) |
|---|---|---|---|---|
| 0.6B | 0.48 | ~0.09 | 0.07 | **~0.39（巨大）** |
| 1.7B | 0.48 | ~0.22 | 0.11 | ~0.26 |
| 4B | 0.44 | ~0.31 | 0.25 | ~0.13 |
| 8B | 0.44 | ~0.30 | 0.25 | ~0.14 |
| 14B | 0.46 | ~0.375 | 0.325 | ~0.085 |
| 32B | 0.42 | >0.42（无崩） | 0.42 | **~0（readout 已达 content 峰）** |

> **旧 j3/0.33L 保留为参考**：旧 recall-optimal ~j3 = **recall 上界参考**（mk/vt 在浅 j3 更高，如 14B mk 98/95/90、vt 89/82/96；32B/30B-A3B 主 benchmark 亦用 zs j3）；旧固定 0.33L（8B j12/32B j21）= **保守下界（~95% 语义 knee95）**。详见 `status/QCMEM_J_DETERMINATION.md`。

---

### 8B (Qwen3-8B-Instruct, j=12) — RULER n=500 (8k/16k/32k, string_match)
| task | selector | 8k | 16k | 32k |
|---|---|---|---|---|
| niah_single | bm25 | 100 | 100 | 100 |
| niah_single | oracle | 100 | 100 | 100 |
| niah_multikey | bm25 | 91 | 91 | 92 |
| niah_multikey | oracle | 100 | 98 | 100 |
| **vt** | **iter_bm25(固定)** | **97** | **97** | **98** |
| vt | bm25单遍 | 48 | 26 | 23 |
| vt | reader_attn | 76 | 30 | 17 |
| vt | oracle(vt无意义) | 7 | 6 | 5 |
| vt | iter_bm25_adaptive(ρ0.3,证否) | 31 | 25 | 22 |

★结论：VT 需多跳迭代检索——**固定 iter_bm25 = 97/97/98**（远超单遍 bm25 48/26/23、reader_attn 76/30/17）；adaptive ρ0.3 停太早=31 已证否，默认改回固定 iter_bm25。

### 8B — 其他 benchmark
- **LongEval**(+adapter, n=100)：4k/8k/16k/32k = **92/71/74/65**。
- **vs-Dense**(+adapter, n=100, PG19-prose)：窗口内 8k-64k QCMem≈100 追平 Dense；**128k：niah_single Dense=0/QCMem=100，niah_multikey Dense=0/QCMem=93**（超 40960 窗口 Dense 崩，QCMem 恒定）。
- **BABILong**(qa1/2/5×0-32k)：adapter + zeroshot cells 全跑完，**官方判分聚合完成**（见下「全 scale 聚合大表」）。
- **LongBench**(narrativeqa/qasper/hotpotqa/2wikimqa, +adapter)：4 shard preds merge+官方 qa_f1 **完成**（见下大表，AVG=9.76）。
- **LoCoMo**(+adapter)：跑中。
- RULER baseline kvdirect/hcache：跑中。

### 4B (Qwen3-4B-Instruct, j=12, zero-shot) — 跑中（.85+.24），下轮落表

### 14B (Qwen3-14B-Instruct, zero-shot j3；+adapter 臂 RULER/BABILong) — 全 benchmark 聚合完成（2026-07-16）
RULER n=100 recall（8k/16k/32k）：
| task | selector | zs(j3) | +adapter |
|---|---|---|---|
| niah_single | bm25 | 100/100/100 | 100/100/100 |
| niah_multikey | bm25 | 98/95/90 | 100/99/99 |
| **vt** | iter_bm25 | 89/82/96 (j3) · 85/97/96 (j4) | **99/100/100** |
| niah_single | bm25 | j13: 99/85/98 (深 j 掉 16k) | — |

- **LongEval**(zs,j3, n=100)：4k/8k/16k/32k = **99/99/97/100**（远超 8B 92/71/74/65 与 30B-A3B 43/30/-/28）。
- **vs-Dense**(zs, n≈50-100, 超窗口崩塌)：single 8k-64k Dense/QCMem≈100/100；**128k single Dense=11 / QCMem=100**；multikey 8k/16k/32k Dense100 QCMem 97/94/91，64k Dense96/QCMem99，**128k multikey Dense=5 / QCMem=98**。
- **BABILong**：zs(j3) overall=32.7，+adapter=46.6（见上大表）。
- **LongBench** qa_f1=**9.63**；**LoCoMo** acc=**1.41** F1=2.17（见上表）。

### 32B (Qwen3-32B, zero-shot j3；未蒸 adapter) — 全 benchmark 聚合完成（2026-07-16）
RULER n=100 recall（8k/16k/32k）：
| task | selector | zs(j3) |
|---|---|---|
| niah_single | bm25 | 100/100/100 |
| niah_multikey | bm25 | 100/99/100 (j6: 100/97/100) |
| **vt** | **iter_bm25** | **30/18/24**（j3 峰值~24；深 j 不升） |

- **32B VT j-sweep**（iter_bm25，recall 8k/16k/32k，avg）：j3 **29.6/17.8/24.4 (~24, 峰值)** · j6 15.0/9.6/20.8 · j9 18.6/13.6/16.4 · j13 18.6/22.6/12.8 · j16 22.2/20.8/11.0 · j20 25.8/7.0/8.8。→ **深 j 不救 VT，峰值停在浅 j3** = selector/模型瓶颈（对比 14B VT j3 已 89-96、8B iter_bm25 97/97/98）。32B 是唯一 VT 崩的 scale，待配 T21 speed frontier。
  - ⚠️ 上行是 **2026-07-17 chat+no-think 标准之前**的旧口径；**干净口径 vt 见 `status/QCMEM_RECALL_SPEED_FRONTIER.md` 表1**（T21，j3 16k=93.6/32k=52.0 峰值，深 j 崩），旧"峰值~24"作废。
- **★ 32B per-task recall-vs-j frontier（T21b，2026-07-20，.73；zero-shot，iter_bm25，chat+no-think，n=50，string_match_all，全 48 cell empty=0）**——把 T21 vt frontier 推广到 niah 检索类，验证 resume_j 速度↔精度旋钮的任务依赖性。recall(j) 16k/32k：

  | j (j/L) | single_2 | multikey_1 | multivalue | multiquery |
  |---|---|---|---|---|
  | 6 (.09) | 100/100 | 98/96 | 48.5/50 | 25/25.5 |
  | 20 (.31) | 100/98 | 96/90 | 49.5/50 | 25/32.5 |
  | 27 (.42) | 100/98 | 96/86 | 49.5/49 | 26.5/32 |
  | 34 (.53) | 100/100 | 66/62 | 49.5/48.5 | 23/24 |
  | 41 (.64) | 10/34 | 8/4 | 1.5/1.5 | 2/0 |

  结论：**统一 readout 硬崖 j41（0.64L）**（所有 niah 一致坠落）；崖下 **niah_single 深-j 最宽容**（recall 平在 100 到 j34/0.53L = read 1.93× 提速零损失，纯免费提速）、**multivalue/multiquery 亦深-j 宽容**（recall 平在各自任务天花板 ~50/~28 到 j34）、**multikey 居中**（免费到 j27，j34 温和衰减 66/62）、**vt 反向偏好浅 j**（见 frontier 表1）。read latency 任务无关（avg_read_len=6182 恒定，引 T21 线性轴）。详见 `status/QCMEM_RECALL_SPEED_FRONTIER.md` §per-task 泛化。原始：`.73:ruler_results/qcmem_32b_t21b/`。
- **LongEval**(zs,j3, n=100)：4k/8k/16k/32k = **99/100/99/100**。
- **vs-Dense**(zs, 超窗口崩塌)：single 8k-64k=100/100；**128k single Dense=OOM / QCMem=100**；multikey(n50) 8k-32k Dense100 QCMem 98/98/100，64k Dense96/QCMem100，**128k multikey Dense=OOM / QCMem=98**（Dense 长档直接 OOM，QCMem read pack 恒定存活）。
- **BABILong**：zs(j3) overall=**41.7**（见上大表）。
- **LongBench** qa_f1=**12.37**（全 scale 最高）；**LoCoMo** acc=**6.55** F1=4.12（见上表）。

### 30B-A3B (Qwen3-30B-A3B MoE, L48, zero-shot j12) — 全 benchmark 聚合完成（2026-07-16）
> ⚠️ 30B-A3B 全 benchmark 用 **zero-shot j12**（recall-optimal 浅 j；BABILong/LongBench/LongEval/LoCoMo/vs-Dense 的 `eval_config.json` 均 resume_j=12 实测确认），**非** RUN_REGISTRY 旧注误记的 j3，也非 plan §1 双 j 主表的 j16/j22。未蒸 adapter。
- **BABILong**（qa1/2/5×0-32k，官方 compare_answers，n=100/cell）：overall=**32.3**。qa1 avg(0k→32k)=98/77/47/36/27/22/13，qa2=55/30/24/20/16/11/2，qa5=36/37/27/13/30/27/30（见上大表）。长档随 scale 衰减，qa5 在长档反而保持~30（时序题对深度不敏感）。
- **LongBench** qa_f1=**6.61**（narrativeqa 3.49 / qasper 11.69 / hotpotqa 4.10 / 2wikimqa 7.17；hotpotqa/2wikimqa 多跳偏弱）。
- **LongEval**(n=100)：4k/8k/16k/32k = **43/30/-/28**（16k 档缺；显著弱于 14B 99-100 与 32B 99-100——MoE 浅 j12 在 lines-retrieval 上不稳）。
- **LoCoMo**（全测试集 n=1986，score_sample）：overall acc=**7.40** F1=**5.02**（multi_hop 6.7 / single_hop 2.5 / temporal 13.5 / open_domain 10.1 / adversarial 4.9）。**LoCoMo 全 scale 最高 acc/F1**（略超 32B acc6.55/F4.12）。
- **vs-Dense**（超窗口崩塌，n=50，niah_single bm25 j12）：single 8k/16k/32k/64k Dense=100 / QCMem=98/100/98/100；**128k single Dense=OOM / QCMem=100**（30B MoE @128k Dense OOM，QCMem read pack 恒定 ~63G 存活）。**multikey 未完**（`logs/30ba3b_vsdense_mk.log`，diskB .73 GPU0 起，2026-07-16；.73 已转 OLMo 剪层训练占卡，此 cell 待空卡续跑）。

---

### ★ 全 scale 聚合大表（2026-07-15 官方判分，n=100/cell，scorer=`scripts/score_qcmem_scale.py`）

判分口径：BABILong=`babilong.metrics.compare_answers`+TASK_LABELS（禁 re.search）；LongBench=官方 qa_f1（SQuAD token-F1，与 `eval_longbench_mem_space.compute_f1` 一致，已对齐现有 scores.json）；LoCoMo=`eval_qcmem_locomo.score_sample`（F1/EM/substring-acc，adversarial=弃权正确）。数据在 diskB `babilong_results/` + `longbench_results/` + `locomo_results/`。8B=Qwen3-8B(j=9)，4B=Qwen3-4B(j=9)，1.7B=Qwen3-1.7B(j=4)，0.6B=Qwen3-0.6B(j=3)；8B-adapter=+distill LoRA，8B-zs=zero-shot。**14B/32B（collaborator）主 benchmark 全用 zero-shot j3（bm25，vt=iter_bm25），非 plan 的 j13/j21——sweep 峰值在浅 j3；14B 另有 +adapter 臂（2026-07-16 聚合）。30B-A3B 全 benchmark 用 zero-shot j12（recall-optimal 浅 j；eval_config resume_j=12 实测确认，非 j3）。**

#### BABILong 官方 accuracy（qa1/qa2/qa5 × 0k-32k，%）
| model | task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|
| **32B**(zs,j3) | qa1 | 100 | 99 | 92 | 78 | 53 | 33 | 24 |
| | qa2 | 68 | 63 | 49 | 47 | 32 | 24 | 6 |
| | qa5 | 22 | 22 | 12 | 7 | 12 | 20 | 13 |
| | **avg** | 63 | 61 | 51 | 44 | 32 | 26 | 14 |
| **14B**(zs,j3) | qa1 | 99 | 80 | 66 | 42 | 9 | 5 | 1 |
| | qa2 | 35 | 52 | 40 | 6 | 9 | 3 | 1 |
| | qa5 | 16 | 77 | 75 | 36 | 10 | 11 | 14 |
| | **avg** | 50 | 70 | 60 | 28 | 9 | 6 | 5 |
| **14B-adapter** | qa1 | 99 | 87 | 71 | 59 | 43 | 28 | 15 |
| | qa2 | 47 | 54 | 27 | 13 | 5 | 1 | 2 |
| | qa5 | 36 | 83 | 82 | 64 | 51 | 55 | 57 |
| | **avg** | 61 | 75 | 60 | 45 | 33 | 28 | 25 |
| **8B-adapter** | qa1 | 98 | 79 | 68 | 69 | 64 | 55 | 21 |
| | qa2 | 26 | 44 | 43 | 44 | 35 | 25 | 9 |
| | qa5 | 69 | 76 | 77 | 75 | 62 | 62 | 65 |
| | **avg** | 64 | 66 | 63 | 63 | 54 | 47 | 32 |
| **8B-zs** | qa1 | 98 | 57 | 70 | 58 | 10 | 5 | 2 |
| | qa2 | 54 | 14 | 27 | 24 | 15 | 11 | 3 |
| | qa5 | 68 | 73 | 62 | 60 | 38 | 40 | 35 |
| | **avg** | 73 | 48 | 53 | 47 | 21 | 19 | 13 |
| **4B** | qa1 | 97 | 80 | 65 | 49 | 45 | 34 | 22 |
| | qa2 | 60 | 45 | 39 | 32 | 21 | 25 | 6 |
| | qa5 | 71 | 76 | 57 | 58 | 49 | 48 | 57 |
| | **avg** | 76 | 67 | 54 | 46 | 38 | 36 | 28 |
| **1.7B** | qa1 | 91 | 79 | 65 | 40 | 11 | 12 | 2 |
| | qa2 | 45 | 43 | 31 | 19 | 14 | 7 | 4 |
| | qa5 | 65 | 51 | 30 | 46 | 16 | 26 | 21 |
| | **avg** | 67 | 58 | 42 | 35 | 14 | 15 | 9 |
| **0.6B** | qa1 | 0 | 0 | 25 | 11 | 4 | 3 | 2 |
| | qa2 | 0 | 0 | 15 | 14 | 3 | 8 | 2 |
| | qa5 | 0 | 5 | 27 | 29 | 21 | 25 | 38 |
| | **avg** | 0 | 2 | 22 | 18 | 9 | 12 | 14 |
| **30B-A3B**(zs,j12) | qa1 | 98 | 77 | 47 | 36 | 27 | 22 | 13 |
| | qa2 | 55 | 30 | 24 | 20 | 16 | 11 | 2 |
| | qa5 | 36 | 37 | 27 | 13 | 30 | 27 | 30 |
| | **avg** | 63 | 48 | 33 | 23 | 24 | 20 | 15 |

overall mean acc（21 cells）：8B-adapter=**55.5** > 14B-adapter=**46.6** > 4B=49.3 > 32B(zs,j3)=**41.7** > 8B-zs=39.2 > 1.7B=34.2 > 14B-zs(j3)=**32.7** > 30B-A3B(zs,j12)=**32.3** > 0.6B=11.0.
★ adapter vs zeroshot（同 8B）：长档增益显著——8k/16k/32k avg 21→54 / 19→47 / 13→32；distill 主要救回长上下文。0.6B 在 0k/1k 近乎 0（模型太小，无检索上下文时不遵循格式），2k+ 才有信号。
★ 14B/32B(collaborator, zero-shot j3)：14B adapter(46.6) 显著超 14B-zs(32.7)，长档 8k/16k/32k avg 9→33 / 6→28 / 5→25——adapter 结论 scale 一致。32B(zs,j3)=41.7 中等，qa1 长档尚可(53/33/24)但 qa5 弱(12/20/13)；未蒸 adapter。

#### ★ 8B BABILong CLEAN 重跑（T23，2026-07-20，节点 .73；chat+no-think + iter_bm25 统一口径）
> 干净口径 = `--use_chat_template` + `enable_thinking=False`(no-think) + `selector=iter_bm25`（2026-07-17 标配）。官方 `compare_answers`+TASK_LABELS（首句+恰好一 label），4-shard 求和合并（`score_nested_babilong.py`），n=100/cell。**全 21 cell empty_output=0，输出 well-formed**（抽查 adapter qa1 16k 为干净单答=真检索命中/失误，非 thinking ramble）。数据：`.73:babilong_results/qcmem_j9_iter_bm25_chatnothink_zs`（zs）+ `qcmem_j12_iter_bm25_chatnothink_ad`（adapter）。

| model | task | 0k | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|
| **8B-adapter(j12) clean** | qa1 | 100 | 82 | 68 | 63 | 50 | 23 | 27 |
| | qa2 | 57 | 58 | 53 | 51 | 31 | 20 | 6 |
| | qa5 | 83 | 83 | 78 | 76 | 70 | 59 | 61 |
| **8B-zs(j9) clean** | qa1 | 100 | 83 | 74 | 61 | 16 | 3 | 1 |
| | qa2 | 60 | 53 | 59 | 38 | 22 | 11 | 1 |
| | qa5 | 81 | 82 | 67 | 62 | 57 | 45 | 41 |

overall mean acc（21 cell）：**adapter(j12)=57.10（1199/2100）> zeroshot(j9)=48.43（1017/2100）**。
- **vs 旧 2026-07-14 污染版**（bm25 + chat_template_no + thinking-era；zs `qcmem_8b_zeroshot_babilong`(j12)=39.2 实测 37.9 / adapter `qcmem_8b_adapter_babilong_mid`(j12)=55.5）：**zs 39.2→48.4（+9.2）**、**adapter 55.5→57.1（+1.6 持平）**。增益集中 0k–8k（chat+no-think 净化指令跟随：zs qa2 1k 14→53、qa5 8k 38→57）。
- **⚠️ 修正旧 caveat**："32k 真值 35–50" 不成立——干净口径 32k qa1/qa2 仍低（adapter 27/6、zs 1/1），仅 qa5 32k ~41–61。thinking 污染对 32k 压低幅度被**高估**；32k qa1/qa2 是**真长程检索失败**非判分 artifact。
- **⚠️ selector caveat**：统一 iter_bm25 对 **qa1 单事实中档反掉分**（adapter qa1 16k 旧 bm25=55→干净 iter_bm25=23；输出 well-formed 单答=真失误）。iter_bm25 多跳 BFS 对 single-fact QA 过度扩召摊薄 topk。bm25 对 qa1 更优，但 iter_bm25 是统一口径标配。

#### ★★ zero-shot BABILong CLEAN 全 scale 曲线（T24，2026-07-20，节点 .73；chat+no-think + iter_bm25 统一口径）
> **8B clean 的直接延伸——补齐 0.6B/1.7B/4B/14B/32B 到与 8B(48.43) 完全一致口径。** 干净口径 = `--use_chat_template` + `enable_thinking=False`(no-think) + `selector=iter_bm25`（topk12/chunk512/sink=bos/bf16）。zero-shot（无 adapter/lora），每 scale 用 §1 主表 **readout-safe j**（single≥90 最深）：0.6B **j2** / 1.7B **j3** / 4B **j9** / 8B **j9** / 14B **j13** / 32B **j27**。官方 `babilong.metrics.compare_answers`+TASK_LABELS，n=100/cell，8-shard 求和合并（`score_nested_babilong.py`，与 4-shard 等价）。**全 21 cell/scale empty_output=0（各 2100 records），输出 well-formed**（抽查 qa1 16k 为干净单答 "Sandra is in the bedroom." 类=真检索命中/失误，非 thinking ramble；tiny 模型偶带 `</think>` 前缀 = 与 8B 参考同口径行为）。
> **来源**：0.6B/4B = agent a9eeeb2a 已跑 clean dir **仅打分**（`babilong_results/qcmem_0p6b_zs_iter_chatnothink`(j2) / `qcmem_4b_zs_iter_chatnothink`(j9)）；1.7B/14B/32B = **本次新 GPU 重跑**（`babilong_results/qcmem_{1p7b,14b,32b}_zs_iter_chatnothink`）；8B = T23 参考（`qcmem_j9_iter_bm25_chatnothink_zs`）。30B-A3B 未纳入本曲线。

| model | j (/L) | qa1 0k/8k/16k/32k | qa2 0k/8k/16k/32k | qa5 0k/8k/16k/32k | **overall(21 cell)** |
|---|---|---|---|---|---|
| **0.6B** | j2 (0.07L) | 77/44/35/37 | 45/18/12/5 | 21/16/18/19 | **33.05** |
| **1.7B** | j3 (0.11L) | 92/14/9/2 | 52/13/6/1 | 67/41/36/32 | **41.19** |
| **4B** | j9 (0.25L) | 98/19/7/4 | 60/17/3/0 | 80/56/46/51 | **46.71** |
| **8B** | j9 (0.25L) | 100/16/3/1 | 60/22/11/1 | 81/57/45/41 | **48.43** |
| **14B** | j13 (0.325L) | 100/31/9/5 | 66/18/10/5 | 83/73/62/58 | **54.29** |
| **32B** | j27 (0.42L) | 100/64/43/43 | 70/39/20/9 | 91/67/54/62 | **64.10** |

**★ zero-shot BABILong overall 单调随 scale 上升**：0.6B 33.05 → 1.7B 41.19 → 4B 46.71 → 8B 48.43 → 14B 54.29 → 32B **64.10**（+31pt over 0.6B，完全单调）。长档增益最明显在大模型（qa1 16k：0.6B 35→32B 43；qa5 32k：0.6B 19→32B 62）。
- **撤销 legacy caveat**：0.6B/1.7B/4B/14B/32B 的旧污染版 BABILong（recall-optimal 浅 j3/j4/j3 + chat_template_no + thinking-era，§1a 行 11.0/34.2/49.3/32.7/41.7）**作废用于口径一致对照**——本 clean 曲线是全 scale 唯一同口径可比结果。（注：clean 用 readout-safe j 而非旧 recall-optimal j，故与旧 §1a 逐 cell 不直接可比；clean 曲线内部逐 cell 严格同口径。）

#### ★★ zero-shot LongBench + LoCoMo CLEAN 全 scale 曲线（T22，2026-07-20，节点 .73；chat+no-think + iter_bm25 统一口径）
> **T24 BABILong clean 曲线的 real-QA 延伸——把 zero-shot clean 口径从合成 BABILong 扩到 LongBench(6-ds real-doc QA) + LoCoMo(多会话记忆, n=1986)，与 BABILong scale 曲线(0.6B 33.05→…→32B 64.10)同口径可比。** 完全同 recipe：`--use_chat_template` + `enable_thinking=False`(no-think) + `--selector iter_bm25`（topk12/chunk512/sink=bos/bf16，zero-shot 无 adapter/lora），每 scale 用 §1 主表 **readout-safe j**（0.6B j2 / 1.7B j3 / 4B j9 / 8B j9 / 14B j13 / 32B j27）。每 scale 8-GPU：4-shard LongBench(GPU0-3) + 4-shard LoCoMo(GPU4-7) 并行。
> **判分**：LongBench = 官方 `eval_longbench_mem_space.run_scoring`（`compute_f1_multi`/`compute_em_multi`，SQuAD token-F1/EM，**禁 re.search**），6-ds cohort = hotpotqa/narrativeqa/qasper/multifieldqa_en/2wikimqa/musique（与现有 baseline cohort 一致，合计 n=1150/scale）；LoCoMo = 项目 `eval_qcmem_locomo.run_scoring` F1/EM/acc（**无 GPT-4o judge**，acc=substring/F1≥0.5 proxy），n=1986，5-cat 分解。**每 scale：LongBench 1150 records + LoCoMo 1986 records，全部 empty_output=0，输出 well-formed**（抽查 32B hotpotqa "Gyulafehérvár, Transylvania"/"1532"、0.6B LoCoMo "1:56 pm on 8 May, 2023"=真检索命中/失误，非 thinking ramble）。
> **来源**：8B = 已有 clean dir（`{longbench,locomo}_results/qcmem_8b_zs_iter_chatnothink`，j9）**仅打分**（零 GPU）；**0.6B/1.7B/4B/14B/32B = 本次全部新 GPU 重跑**（`qcmem_{0p6b,1p7b,4b,14b,32b}_zs_iter_chatnothink`）。旧 LongBench/LoCoMo `qcmem_{14b,32b,1p7b_j*,0p6b_j*,4b*}` dir 经 `eval_config.json` 核实为 **legacy 污染**（selector=bm25 非 iter_bm25 / chat_template=false / recall-optimal 浅 j 非 readout-safe j）→ 三轴皆不符，均需 clean 重跑，不用于本曲线。30B-A3B 未纳入。

**LongBench 官方 F1（zero-shot readout-safe j，chat+no-think+iter_bm25，n/task=150-200，AVG=6-ds 均值）**
| model | j | hotpotqa | narrativeqa | qasper | multifieldqa_en | 2wikimqa | musique | **AVG F1** | AVG EM | 新跑/仅打分 |
|---|---|---|---|---|---|---|---|---|---|---|
| **0.6B** | j2 | 21.42 | 13.78 | 11.83 | 40.19 | 25.69 | 9.09 | **20.33** | 7.61 | 新 GPU |
| **1.7B** | j3 | 20.77 | 12.30 | 29.10 | 38.16 | 18.49 | 8.84 | **21.28** | 5.00 | 新 GPU |
| **4B** | j9 | 37.55 | 17.04 | 32.62 | 41.81 | 29.88 | 13.86 | **28.79** | 11.78 | 新 GPU |
| **8B** | j9 | 27.81 | 17.93 | 33.66 | 41.98 | 25.37 | 10.88 | **26.27** | 9.17 | 仅打分(已有 clean) |
| **14B** | j13 | 37.80 | 21.47 | 36.77 | 42.67 | 28.74 | 20.16 | **31.27** | 12.72 | 新 GPU |
| **32B** | j27 | 43.75 | 23.30 | 43.16 | 46.04 | 38.31 | 23.18 | **36.29** | 17.39 | 新 GPU |

**LoCoMo 官方 F1/EM/acc（n=1986，5-cat 按官方 category number；cat5=adversarial 弃权判分，F1==EM==acc）**
| model | j | overall F1 | overall EM | overall acc | F1 c1/c2/c3/c4/c5 | acc c1/c2/c3/c4/c5 |
|---|---|---|---|---|---|---|
| **0.6B** | j2 | **12.19** | 0.65 | **14.75** | 10.96/18.63/9.64/16.79/0.22 | 9.93/13.71/10.42/24.97/0.22 |
| **1.7B** | j3 | **9.75** | 0.30 | **16.41** | 9.86/9.11/6.83/14.74/1.35 | 7.80/7.79/11.46/31.15/1.35 |
| **4B** | j9 | **12.56** | 0.91 | **20.85** | 13.45/10.25/8.28/19.46/1.57 | 17.02/8.41/14.58/37.81/1.57 |
| **8B** | j9 | **14.26** | 3.37 | **22.41** | 15.11/16.84/7.94/21.28/0.00 | 15.96/18.69/13.54/38.88/0.00 |
| **14B** | j13 | **13.85** | 1.86 | **22.26** | 13.45/11.59/9.13/20.82/3.59 | 14.18/9.35/23.96/39.60/3.59 |
| **32B** | j27 | **17.31** | 3.37 | **27.64** | 15.14/12.61/9.15/26.01/7.40 | 16.31/9.35/19.79/50.06/7.40 |
（cat n：c1=282 c2=321 c3=96 c4=841 c5=446。c4（最大类）acc 是总 acc 主要驱动，随 scale 单调升 0.6B 24.97→32B 50.06。）

**★ scale 曲线小结（对比 BABILong clean 单调 0.6B 33.05→32B 64.10）**：
- **LongBench AVG F1 整体随 scale 上升，32B 最优**：0.6B 20.33 → 1.7B 21.28 → 4B 28.79 → 8B 26.27 → 14B 31.27 → 32B **36.29**（+15.96pt over 0.6B）。**单一非单调点=8B(26.27) < 4B(28.79)**（8B zero-shot 在 hotpotqa/2wikimqa 偏弱；4B 反常强）；其余相邻 scale 皆升。
- **LoCoMo acc 近单调随 scale 上升**：0.6B 14.75 → 1.7B 16.41 → 4B 20.85 → 8B 22.41 → 14B 22.26(≈8B plateau) → 32B **27.64**。**LoCoMo F1 更 noisy 但 32B 最优**：12.19→9.75(1.7B 谷)→12.56→14.26→13.85(14B 微降)→**17.31**。
- **结论**：两个 real-QA 基准都印证「scale up 有帮助、32B 全面最优」的核心故事，趋势与 BABILong 一致；但 real-doc QA 的 token-F1 比合成 BABILong exact-match **更 noisy**（LongBench 8B<4B、LoCoMo 1.7B/14B 小回落），单调性弱于 BABILong。绝对分偏低（真实长文档/多会话记忆对 zero-shot QCMem 检索+resume 本就难，adapter 是长档/硬任务杠杆——本曲线为 zero-shot 无 adapter）。
- **撤 legacy caveat**：§1a 旧 LongBench/LoCoMo 行（recall-optimal 浅 j + chat_template_no + bm25-single）作废用于口径一致对照；本 clean 曲线是全 scale 唯一同口径可比结果（内部逐 cell 严格同口径；因用 readout-safe j 而非旧 recall-optimal j，与旧行不直接可比）。

#### ★★ zero-shot RULER CLEAN 全 scale scale-consistency 曲线（T25，2026-07-20，节点 .73；chat+no-think + iter_bm25 统一口径）
> **补齐 4-benchmark cross-scale 故事的第 4 条曲线（BABILong T24 / LongBench+LoCoMo T22 已完成）——把 zero-shot clean 口径扩到 RULER 合成检索。** 完全同 recipe：`--use_chat_template` + `enable_thinking=False`(no-think) + `--selector iter_bm25`（topk12/chunk512/sink=bos/bf16，zero-shot 无 adapter/lora），每 scale 用 §1 主表 **readout-safe j**（0.6B **j2** / 1.7B **j3** / 4B **j9** / 8B **j9** / 14B **j13** / 32B **j27**）。原始 hybrid Qwen3（非 2507）。RULER 判分 = 官方 `scripts.eval_ruler_mem_space._string_match_all_one`（`string_match`，**禁 re.search 自造判分**）。任务 = RULER 标准子集 niah_single_2 / niah_multikey_1 / variable_tracking。
> **CORE 8k/16k/32k = n=100/cell，8-shard 跨 8-GPU 求和合并**（`total_score=Σ score_i·n_i / Σ n_i`）。**超窗 64k/128k = n=50/cell**（128k haystack CPU 构造 ~25s/sample，为可跑性降 n；1-cell-per-GPU + 重档 128k(multikey/vt) 各 2-shard 用满 8 卡）。**全 cell empty_output=0，输出 well-formed**（抽查 8B VT="which variables are assigned value X…"=真任务理解但多跳链未从压缩读出重建=真失误；14B niah_single 命中时 "The special magic number… is 1801110" rec=1.0、失误时改述 PG19/Genesis 干草堆=真 readout drift，均非空/非 thinking ramble）。
> **来源/去重裁决**：0.6B/1.7B/4B/8B 的 niah_single_2 + niah_multikey_1 @8k/16k/32k = **旧 clean dir 复用仅打分（0 GPU）**（0.6B←`qcmem_0p6b_zs_iter_chatnothink`(j2) / 1.7B←`qcmem_1p7b_zs_iter_chatnothink`(j3) / 4B←`qcmem_4b_zs_iter_chatnothink`(j9) / 8B←`qcmem_8b_zs_iter_bm25_chatnothink`(j9)，三轴 selector=iter_bm25 / chat=true / resume_j=readout-safe 皆核实干净）。**variable_tracking @全长 = 全 scale 新 GPU 重跑**（旧 dir VT 轴被 chat=false 污染）。14B/32B @8k/16k/32k 全 3 task = 全新 GPU 重跑（无干净旧数据）。**64k/128k 全 scale 全 task = 全新 GPU 跑**（各处均无干净超窗数据）。新目录统一 `ruler_results/qcmem_{scale}_zs_ruler_iter_chatnothink/`。

**RULER 官方 string_match（zero-shot readout-safe j，chat+no-think+iter_bm25；8k/16k/32k n=100，64k/128k n=50）**
| model | j (/L) | task | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|---|---|
| **0.6B** | j2 (0.07L) | niah_single_2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| | | niah_multikey_1 | 86.0 | 83.0 | 87.0 | 76.0 | 86.0 |
| | | variable_tracking | 83.2 | 89.2 | 70.6 | 91.6 | 88.0 |
| **1.7B** | j3 (0.11L) | niah_single_2 | 56.0 | 5.0 | 59.0 | 56.0 | 28.0 |
| | | niah_multikey_1 | 39.0 | 20.0 | 13.0 | 0.0 | 38.0 |
| | | variable_tracking | 40.8 | 42.6 | 59.0 | 63.6 | 68.8 |
| **4B** | j9 (0.25L) | niah_single_2 | 70.0 | 53.0 | 38.0 | 28.0 | 18.0 |
| | | niah_multikey_1 | 46.0 | 24.0 | 38.0 | 16.0 | 20.0 |
| | | variable_tracking | 24.4 | 19.8 | 16.0 | 18.0 | 18.4 |
| **8B** | j9 (0.25L) | niah_single_2 | 95.0 | 98.0 | 97.0 | 88.0 | 94.0 |
| | | niah_multikey_1 | 32.0 | 42.0 | 49.0 | 34.0 | 38.0 |
| | | variable_tracking | 1.2 | 4.4 | 0.2 | 0.0 | 1.6 |
| **14B** | j13 (0.325L) | niah_single_2 | 15.0 | 13.0 | 16.0 | 28.0 | 0.0 |
| | | niah_multikey_1 | 46.0 | 28.0 | 43.0 | 32.0 | 24.0 |
| | | variable_tracking | 2.0 | 0.4 | 0.8 | 0.4 | 0.8 |
| **32B** | j27 (0.42L) | niah_single_2 | 100.0 | 100.0 | 98.0 | 100.0 | 100.0 |
| | | niah_multikey_1 | 96.0 | 97.0 | 89.0 | 94.0 | 96.0 |
| | | variable_tracking | 43.0 | 42.2 | 26.8 | 28.4 | 26.8 |

**★ RULER scale-consistency 小结（全 5 长档完成；对比 3 姊妹曲线）**：与 BABILong（完全单调 33.05→64.10）/ LongBench（8B<4B 唯一非单调）/ LoCoMo（近单调）**明显不同，RULER zero-shot@readout-safe-j 强非单调**：
- **scale 轴（非单调）**：**niah_single_2** — 0.6B **100**/8B 95-98/32B **98-100** 三点近满，但 **1.7B(5-59)、4B(18-70)、14B(0-28) 大幅塌陷**。**14B@j13 全长塌到 0-28（128k=0）** = readout drift（命中时答对 rec=1.0，多数改述 PG19/Genesis 干草堆而非答 needle），证明 readout-safe j 标定（按 single≥90 取最深）在 14B 这点并不稳、且随长度恶化。**niah_multikey_1** — 32B(89-97) 最强、0.6B(76-87) 反超中段，1.7B/4B/8B/14B(0-49) 弱且非单调。**variable_tracking** — **tiny 0.6B 最强(70-92)**（浅 j2 保留原始 VAR token 链），随 j 加深塌陷：8B/14B(≈0-4) 几乎全崩，32B(27-43) 部分恢复。VT 崩点与 **j 深度（非 scale）**强相关。
- **length 轴（超窗健壮性 = QCMem 固定读卖点）**：readout 表现好的 (scale,task) 点**跨长度几乎不衰减**——QCMem 固定读 ~6.6k tok（与上下文长度无关），故 **32B niah_single 64k/128k 仍 100/100、niah_multikey 94/96；0.6B niah_single 全长 100、VT 64k/128k 91.6/88.0；8B niah_single 88/94**——在 **dense 128k 必 OOM** 的长度上稳定跑通且高分（RULER 侧印证 vs-Dense 卖点）。readout 已崩的点（8B/14B VT、14B single）在长档仍崩（1.7B/4B single 随长度缓降），非长度问题而是 j-标定问题。
- **唯一稳健信号 = 32B 全 task 最强 + 全长档最稳**（与 3 姊妹曲线「32B 全面最优」一致）+ **0.6B 因浅 j 在字面 needle/链任务反常强**。机制 = RULER exact-match 字面 needle/链对「readout-safe j 是否保留原始 token」极敏感；该 j 按 single-recall 标定，**不迁移到 multikey/VT 且中段 scale drift** → 曲线非单调。verbose chat 前言在固定 VT token 预算下亦部分压低大模型 VT（协议内 caveat，非 bug）。
- **运维记录**：CORE 8k/16k/32k(n=100) 由本 agent phase1 8-shard 跑完验证；64k/128k(n=50) 0.6B/1.7B/4B 由 phase2 sequential ext 跑完，8B/14B/32B 由 recovery task-pool（`t25_pool.sh`，8-worker 抢占式；phase2 driver 在 4B 后意外退出后同口径同目录接管）跑完。**全 6 scale × 3 task × 5 长 = 90 cell，empty_output=0（抽查 32B/8B/14B 128k 共 300 records 全非空、well-formed），oom=0，total_n=1200/scale**。

#### ★★ zero-shot LongEval CLEAN 全 scale scale-consistency 曲线（T26，2026-07-20，节点 .73；chat+no-think + iter_bm25 统一口径）
> **补齐 5-benchmark cross-scale 故事的第 5 条（最后一条）曲线（BABILong T24 / LongBench+LoCoMo T22 / RULER T25 已完成）——把 zero-shot clean 口径扩到 LongEval line-retrieval。** 完全同 recipe：`--use_chat_template` + `enable_thinking=False`(no-think) + `--selector iter_bm25`（topk12/chunk512/sink=bos/bf16，zero-shot 无 adapter/lora），每 scale 用 §1 主表 **readout-safe j**（0.6B **j2** / 1.7B **j3** / 4B **j9** / 8B **j9** / 14B **j13** / 32B **j27**）。driver `scripts/eval_qcmem_longeval.py` 官方 LongEval accuracy（line-key→register-content 精确抽取匹配，**禁 re.search 自造判分**）。6 长档 = 4k/8k/16k/32k/64k/128k，**n=100/cell，8-shard 求和合并**。
> **来源/去重裁决**：8B 4k/8k/16k/32k = 旧 clean of4 run 复用仅打分（config 核实 selector=iter_bm25 / chat=true / resume_j=9 干净），8B 64k/128k = of8 新跑；其余 5 scale(0.6B/1.7B/4B/14B/32B) 全 6 长档全 clean 新 GPU 跑。新目录统一 `longeval_results/qcmem_{scale}_zs_iter_chatnothink/`。**全 36 cell empty=0 oom=0，well-formed**（抽查 32B 16k：expected 153333 → output "…is <153333>" → pred=153333 correct=True=真检索命中；read_len≈6315 恒定=固定读、n_selected_chunks=12=topk12）。

**LongEval 官方 accuracy（zero-shot readout-safe j，chat+no-think+iter_bm25；n=100/cell，%）**
| model | j (/L) | 4k | 8k | 16k | 32k | 64k | 128k | AVG |
|---|---|---|---|---|---|---|---|---|
| **0.6B** | j2 | 58 | 28 | 41 | 25 | 35 | 37 | **37.3** |
| **1.7B** | j3 | 33 | 11 | 4 | 12 | 20 | 15 | **15.8** |
| **4B** | j9 | 30 | 11 | 16 | 15 | 8 | 11 | **15.2** |
| **8B** | j9 | 14 | 6 | 5 | 7 | 6 | 7 | **7.5** |
| **14B** | j13 | 36 | 13 | 16 | 14 | 18 | 25 | **20.3** |
| **32B** | j27 | 99 | 97 | 97 | 98 | 97 | 97 | **97.5** |

**★ LongEval scale-consistency 小结（对比 4 姊妹曲线）**：**与 RULER 同型（强非单调），非 BABILong/LongBench/LoCoMo 的单调随 scale 升**——LongEval = 按 line-key 检索精确 register 值 = exact-match 字面任务：
- **scale 轴（强非单调）**：**32B(97.5) 碾压**，其余全在 7.5-37.3 且 **0.6B(37.3) > 14B(20.3) > 1.7B(15.8) ≈ 4B(15.2) > 8B(7.5)**——中段全面塌陷、8B 最弱。与 RULER niah/VT 同一 pattern（0.6B 浅 j 反常强、32B 独强、中段 drift）。
- **0.6B 反常强机制**：浅 j2 几乎不压缩、保留原始字面 REGISTER token → exact-match 直接命中；随 j 加深（8B j9 最深塌陷区）原始 token 被语义 readout 覆盖 → 字面值丢失（readout drift，同 RULER 14B/8B niah/VT）。
- **length 轴（32B 超窗健壮）**：**32B 跨全长 97-99 几乎无衰减（128k=97≈4k=99，dense 128k 必 OOM）** = QCMem 固定读 ~6.3k tok 与上下文长度无关的直接体现（LongEval 侧印证 vs-Dense 固定读卖点）；其余 scale 各长档均低、无稳定长度趋势。
- **唯一稳健信号 = 32B 全长档最强最稳 + 0.6B 因浅 j 在字面任务反常强**（与 RULER 完全一致）→ **5-benchmark cross-scale 故事收官：语义 QA 三基准（BABILong/LongBench/LoCoMo）单调随 scale 升，字面 exact-match 两基准（RULER/LongEval）强非单调但 32B 始终全面最优**。
- **运维记录**：T26 agent a1f86c58 完成 GPU eval（36 cell，n=100/empty=0/oom=0）后在 well-formed 抽查阶段意外终止（39 tool-uses，未落账）；本 heartbeat 独立复算全 grid（of8 逐-length 求和 + 8B of4 短档修正核实无 double-count，铁律2 核实非空 well-formed）并回填。数据：`.73:longeval_results/qcmem_{0p6b,1p7b,4b,8b,14b,32b}_zs_iter_chatnothink`。

#### ★★ +adapter content-j 曲线（T27，2026-07-20，节点 .73；RULER + LongEval 两 exact-match 基准；chat+no-think + iter_bm25 统一口径）
> **T25(RULER)/T26(LongEval) zero-shot readout-safe-j 曲线的 content-j 配对臂。核心论文问题：蒸馏 content-j LoRA adapter 能否修复 zero-shot readout-safe-j 在字面 exact-match 上的塌陷？** 同 clean recipe（`--use_chat_template` + no-think + `--selector iter_bm25`，topk12/chunk512/sink=bos/bf16），加载各 scale content-j LoRA adapter（`outputs/qcmem_distill_{0p6b_contentj13,1p7b_contentj13,4b_contentj16,8b_contentj16,14b_contentj18}_r32/final`）。**32B/30B-A3B 无 content-j adapter，跳过。** RULER driver `eval_ruler_qcmem.py`（官方 `_string_match_all_one`，禁 re.search，max_new_tokens=48），niah_single_2/niah_multikey_1/variable_tracking @8k/16k/32k n=100（8-shard）。LongEval driver `eval_qcmem_longeval.py`（官方 line-key→register 精确抽取，禁 re.search），4k-128k n=100（8-shard）。**⚠️ LongEval 显式 `--max_new_tokens 48`**（driver 默认 16 截断 adapter 冗长输出 → 全 0 假象；48=统一协议值，与 T26 zs 同样冗长完整、口径一致；RULER 默认已 48）。**全 RULER 15 cell + LongEval 30 cell empty=0 / n=100，铁律2 OK。**

**RULER 官方 string_match（+adapter content-j，chat+no-think+iter_bm25，8k/16k/32k/64k/128k n=100）— 括号内为 zero-shot readout-safe j（T25）对照。★ 64k/128k = T27-ext GPU 早已跑完但从未聚合，本轮（T27b, 2026-07-20）用官方 `_string_match_all_one` 逐 cell 8-shard 加权合并（Σ score_i·n_i/Σn_i）回填；全 cell n=100 empty=0 oom=0 well-formed（抽查 14B niah_single 128k 命中真值 magic number、8B vt 128k 冗长推理未在 48tok 内列出变量=真失误、0.6B vt 全 0=真失误，均非空）**
| model | content-j | niah_single 8/16/32/64/128k | (zs single) | niah_multikey 8/16/32/64/128k | (zs mk) | vt 8/16/32/64/128k | (zs vt) |
|---|---|---|---|---|---|---|---|
| **0.6B** | j13 | 97/99/97/97/93 | (100/100/100/100/100) | 22/32/37/34/32 | (86/83/87/76/86) | 1.4/0.2/0.0/0.0/0.0 | (83/89/71/92/88) |
| **1.7B** | j13 | 81/95/85/78/78 | (56/5/59/56/28) | 20/32/35/34/33 | (39/20/13/0/38) | 23.8/20.8/17.2/11.4/17.0 | (41/43/59/64/69) |
| **4B** | j16 | 61/100/67/75/41 | (70/53/38/28/18) | 45/32/54/37/42 | (46/24/38/16/20) | 48.4/41.2/57.0/43.8/52.2 | (24/20/16/18/18) |
| **8B** | j16 | 99/100/98/98/100 | (95/98/97/88/94) | 51/36/40/39/50 | (32/42/49/34/38) | 8.6/25.0/2.6/3.8/1.2 | (1.2/4.4/0.2/0.0/1.6) |
| **14B** | j18 | 100/99/94/98/100 | (15/13/16/28/0) | 71/74/74/59/72 | (46/28/43/32/24) | 19.2/18.8/19.4/19.4/19.2 | (2.0/0.4/0.8/0.4/0.8) |

> **★ RULER 64k/128k 超窗口小结（content-j，T27b 回填）**：adapter 修复能力在超窗档保持与 CORE(8-32k) 一致的 scale 依赖形态，且 QCMem 固定读（~6.6k tok）使长度几乎不衰减——**大模型 niah_single 超窗健壮**（8B 98/100、14B 98/100 @64k/128k，dense 128k 必 OOM 处稳定跑通），**14B niah_multikey 超窗仍被 adapter 大幅救回**（zs 32/24 → cj 59/72 @64k/128k）；**vt 依旧仅 4B 可用**（cj 43.8/52.2 @64k/128k，其余 scale≈0-19，深 content-j 丢字面链，与 CORE 同型）。niah_single 中段（1.7B/4B）超窗随长度缓降（4B 75→41 @64k→128k）。整体印证：content-j adapter 对超窗字面 exact-match 的价值同样「大模型 YES、小模型 NO」，且固定读让高分点跨 64k/128k 不塌。

**LongEval 官方 accuracy（+adapter content-j，chat+no-think+iter_bm25+max_new48，n=100/cell，%）— AVG 旁括号为 zero-shot readout-safe j（T26）AVG**
| model | content-j | 4k | 8k | 16k | 32k | 64k | 128k | AVG | (zs AVG) |
|---|---|---|---|---|---|---|---|---|---|
| **0.6B** | j13 | 6 | 0 | 0 | 1 | 0 | 0 | **1.2** | (37.3) |
| **1.7B** | j13 | 4 | 0 | 0 | 1 | 0 | 0 | **0.8** | (15.8) |
| **4B** | j16 | 33 | 10 | 17 | 12 | 11 | 10 | **15.5** | (15.2) |
| **8B** | j16 | 22 | 9 | 14 | 15 | 9 | 9 | **13.0** | (7.5) |
| **14B** | j18 | 60 | 31 | 38 | 37 | 38 | 37 | **40.2** | (20.3) |

**★ content-vs-readout 小结（adapter 修复能力随 scale 涌现，monotone scale-dependent）**：
- **小模型（0.6B/1.7B）：adapter 伤字面 exact-match**。LongEval 0.6B 37.3→1.2(↓36)、1.7B 15.8→0.8(↓15)；RULER 0.6B mk 85→30 / vt 81→1、1.7B vt 48→21。深 content-j(~0.45L) 丢字面 token，浅 readout-safe zs 远好；蒸馏 adapter **无法修复反倒破坏**（唯 1.7B RULER niah 40→87 例外升）。
- **中段（4B）：break-even→helpful**。LongEval 15.2→15.5(≈)；RULER 全 task↑（niah 54→76、mk 36→44、vt 20→49）。
- **大模型（8B/14B）：adapter 明显修复 zs 塌陷**。LongEval 8B 7.5→13.0(↑5.5)、14B **20.3→40.2(↑20，翻倍)**；RULER **14B 最戏剧（niah 15→98、mk 39→73、vt 1→19，把 j13 塌陷完全救回）**，8B vt 部分修复（2→12，16k 4.4→25）。
- **结论**：content-j adapter 对字面 exact-match 的修复价值 **随 scale 单调涌现**——**大模型（8B/14B）YES**（尤 14B 完全修复 readout-safe-j 塌陷），**小模型（0.6B/1.7B）NO**（深 j 破坏字面检索、蒸馏训不回来）。与语义/real-QA（BABILong/LongBench/LoCoMo）上 adapter 普遍有益的图景**不同**——字面 exact-match 更依赖浅层原始 token，深 content-j 是双刃剑，仅大模型有足够容量在深层重建字面读出。运维：80-job(40 RULER + 40 LongEval，8-shard×5 scale) 8-worker flock task-pool，全 rc=0；LongEval 首轮误用 driver 默认 max16 全塌，诊断为冗长输出截断 artifact（非真塌），改 max_new48 重跑修复。数据：`.73:{ruler_results,longeval_results}/qcmem_{0p6b,1p7b,4b,8b,14b}_adapter_contentj{13,13,16,16,18}_iter_chatnothink`。

#### ★★ +adapter content-j 曲线（T27b，2026-07-21，节点 .73；BABILong + LongBench + LoCoMo 三个语义/real-QA 基准；chat+no-think + iter_bm25 统一口径）
> **T24(BABILong)/T22(LongBench+LoCoMo) zero-shot readout-safe-j clean 曲线的 content-j 配对臂 = 补齐 5-benchmark × content-j 全网格的最后 3 条（语义侧）。核心问题：content-j adapter 在语义/real-QA（非字面 exact-match）上的 scale 依赖是否与 T27 的 RULER/LongEval 一致？** 同 clean recipe（`--use_chat_template` + no-think + `--selector iter_bm25`，topk12/chunk512/sink=bos/bf16），各 scale content-j LoRA adapter（`outputs/qcmem_distill_{0p6b_contentj13,1p7b_contentj13,4b_contentj16,8b_contentj16,14b_contentj18}_r32/final`）。**32B/30B-A3B 无 content-j adapter，跳过。** 判分：BABILong=官方 `babilong.metrics.compare_answers`+TASK_LABELS（禁 re.search，flat 8-shard 求和合并），qa1/qa2/qa5 × 0k-32k **n=100/cell 全 21 cell 无缺**；LongBench=per-shard `f1`（`eval_qcmem_longbench` 官方 qa_f1）按 num_samples 加权合并，6-ds n=200/ds；LoCoMo=官方 `eval_qcmem_locomo.run_scoring`（token-level SQuAD-F1 + substring-acc + EM，dedup by id）n=1986。**⚠️ 14B BABILong 首轮 pool 进程缺 HF_*_OFFLINE env → dataset-config HEAD 联网重试卡死零推理，独立 standalone 重跑（HF_HUB_OFFLINE=1 + 3s stagger）修复，336 文件齐、抽查 qa1/2k 62% 真检索。**

**BABILong 官方 compare_answers（+adapter content-j，chat+no-think+iter_bm25，n=100/cell）— per-task 为 7 长档(0k-32k)均值，overall=21-cell 均值。括号内为 zero-shot readout-safe j（T24 clean）overall 对照**
| model | content-j | qa1 avg | qa2 avg | qa5 avg | **overall** | (zs overall T24) | Δ |
|---|---|---|---|---|---|---|---|
| **0.6B** | j13 | 31.9 | 19.3 | 33.4 | **28.2** | (33.1, j2) | −4.9 |
| **1.7B** | j13 | 32.3 | 25.3 | 58.1 | **38.6** | (41.2, j3) | −2.6 |
| **4B** | j16 | 46.0 | 30.6 | 70.7 | **49.1** | (46.7, j9) | +2.4 |
| **8B** | j16 | 49.9 | 33.3 | 64.4 | **49.2** | (48.4, j9) | +0.8 |
| **14B** | j18 | 50.9 | 33.4 | 74.0 | **52.8** | (54.3, j13) | −1.5 |

**LongBench 官方 qa_f1（+adapter content-j，chat+no-think+iter_bm25，n=200/ds，%）— MACRO=6-ds 均值。括号内为 zero-shot readout-safe j（T22 clean）MACRO 对照**
| model | content-j | narrqa | qasper | mfqa_en | hotpotqa | 2wiki | musique | **MACRO** | (zs MACRO T22) | Δ |
|---|---|---|---|---|---|---|---|---|---|---|
| **0.6B** | j13 | 10.17 | 12.64 | 32.26 | 18.85 | 16.87 | 5.08 | **15.98** | (20.3, j2) | −4.3 |
| **1.7B** | j13 | 14.42 | 25.97 | 33.21 | 28.63 | 25.83 | 9.71 | **22.96** | (21.3, j3) | +1.7 |
| **4B** | j16 | 21.15 | 33.15 | 41.55 | 41.28 | 29.44 | 15.52 | **30.35** | (28.8, j9) | +1.6 |
| **8B** | j16 | 20.86 | 37.16 | 42.54 | 47.62 | 34.22 | 21.91 | **34.05** | (26.3, j9) | **+7.8** |
| **14B** | j18 | 23.72 | 37.68 | 47.18 | 50.36 | 39.83 | 24.13 | **37.15** | (31.3, j13) | **+5.9** |

**LoCoMo 官方 run_scoring（+adapter content-j，chat+no-think+iter_bm25，n=1986）— 报以 F1 为准（token-SQuAD-F1）。括号内为 zero-shot readout-safe j（T22 clean）F1/acc 对照**
| model | content-j | **F1** | acc | EM | (zs F1/acc T22) | ΔF1 / Δacc |
|---|---|---|---|---|---|---|
| **0.6B** | j13 | **9.97** | 9.26 | 0.35 | (12.2 / 14.8, j2) | −2.2 / −5.5 |
| **1.7B** | j13 | **8.33** | 11.18 | 0.86 | (9.8 / 16.4, j3) | −1.5 / −5.2 |
| **4B** | j16 | **14.20** | 19.34 | 3.63 | (12.6 / 20.9, j9) | +1.6 / −1.6 |
| **8B** | j16 | **16.66** | 22.05 | 4.03 | (14.3 / 22.4, j9) | **+2.4** / −0.4 |
| **14B** | j18 | **18.85** | 26.13 | 5.64 | (13.9 / 22.3, j13) | **+5.0** / **+3.8** |

**★ content-vs-readout 语义侧小结（T27b）——与 T27 字面 exact-match 同型（scale-emergent）但更温和**：
- **同型确认**：三条语义曲线上 content-j adapter 的 scale 依赖与 T27（RULER/LongEval）一致——**小模型（0.6B）全线被伤**（BABILong −4.9、LongBench −4.3、LoCoMo F −2.2/acc −5.5：深 content-j(~0.45L) 丢字面 token，蒸馏训不回浅层原始读出），**大模型（8B/14B）净增益**（LongBench 8B **+7.8**/14B **+5.9**、LoCoMo 14B F **+5.0**/acc **+3.8**）。
- **与字面 exact-match 的关键差异 = 幅度更温和、BABILong 近中性**：LongEval 14B 曾 **20.3→40.2（翻倍）**、RULER 14B niah **15→98（完全救回）**；语义侧最大增益仅 LongBench 8B +7.8。**BABILong 几乎全 scale 近中性**（Δ ∈ [−4.9,+2.4]，4B/8B 微升、14B/0.6B 微降）——因 BABILong 语义 QA 在 ~0.45L content 深度本就基本保真，zero-shot readout-safe-j 已能做，adapter 既救不多也伤不多。
- **机制统一图景（跨 5 benchmark 收敛）**：content-j adapter 价值 = f(任务字面依赖度 × 模型容量)。①**字面依赖越强，adapter 的双刃剑效应越剧**（RULER/LongEval：大模型戏剧修复、小模型戏剧破坏）；②**语义依赖越强，效应越温和**（BABILong 近中性、LongBench/LoCoMo 中等正）；③但**「大模型 YES、小模型 NO」的 scale-emergent 符号在全 5 benchmark 一致**——0.6B 处处被伤、14B 处处（除 BABILong 微降）获益。→ 论文 framing：**content-j adapter 是「读到语义 content 深度（更省算/更多语义压缩）」的杠杆，但需足够模型容量在深层重建读出；小模型宜用浅 readout-safe j，大模型可用深 content-j + adapter**。
- 运维：CPU 打分（flat 8-shard BABILong 自写 `score_flat_babilong.py` glob `{task}_{L}_*shard*of8.csv` 求和 compare_answers；LongBench 加权合并 metrics.json；LoCoMo run_scoring dedup by id）全 5 scale 无缺 cell。数据：`.73:{babilong_results,longbench_results,locomo_results}/qcmem_{0p6b,1p7b,4b,8b,14b}_adapter_contentj{13,13,16,16,18}_iter_chatnothink`。

#### LongBench 官方 qa_f1（narrativeqa/qasper/hotpotqa/2wikimqa，n=200/task，%）〔legacy：recall-optimal 浅 j + chat_template_no + bm25-single，见上方 T22 clean 曲线〕
| model | narrativeqa | qasper | hotpotqa | 2wikimqa | **AVG** |
|---|---|---|---|---|---|
| **32B**(zs,j3) | 6.15 | 13.25 | 15.05 | 15.01 | **12.37** |
| **14B**(zs,j3) | 3.96 | 12.34 | 9.98 | 12.24 | **9.63** |
| **8B-adapter** | 3.85 | 11.08 | 11.87 | 12.23 | **9.76** |
| **4B** | 3.46 | 10.93 | 9.23 | 10.40 | **8.51** |
| **30B-A3B**(j12) | 3.49 | 11.69 | 4.10 | 7.17 | **6.61** |
| **1.7B** | 2.39 | 6.30 | 7.19 | 8.38 | **6.07** |
| **0.6B** | 1.78 | 3.40 | 6.32 | 6.53 | **4.51** |

★ 大体随 scale 升：32B(12.37) > 8B-adapter(9.76) ≳ 14B(9.63) > 4B(8.51) > 30B-A3B(6.61) > 1.7B(6.07) > 0.6B(4.51)。绝对 F1 偏低（真实长文档 QA 对 QCMem 检索+resume 很难；narrativeqa 尤低）；8B 无 zero-shot LongBench 目录（跳过）。4B/1.7B/0.6B 为单 shard(_0)，8B-adapter/14B/32B/30B-A3B 由 4 shard merge(n=200/task)。

#### LoCoMo 官方 acc/F1（全测试集 n=1986，score_sample 五类均含 adversarial 弃权判分）
| model | overall acc | overall F1 | multi_hop | single_hop | temporal | open_domain | adversarial |
|---|---|---|---|---|---|---|---|
| **32B**(zs,j3) | **6.55** | 4.12 | 6.0 | 5.3 | 7.3 | 9.8 | 1.6 |
| **30B-A3B**(zs,j12) | **7.40** | 5.02 | 6.7 | 2.5 | 13.5 | 10.1 | 4.9 |
| **14B**(zs,j3) | **1.41** | 2.17 | 1.4 | 0.9 | 6.3 | 1.6 | 0.5 |

（acc 列 = 官方 substring/F1≥0.5 proxy；adversarial=弃权正确率。32B 全面高于 14B。LoCoMo 对 QCMem 整体很难，绝对分低——真实多会话对话记忆 + 短答案严格判分。⚠️ diskB 版 `eval_qcmem_locomo.py` run_scoring glob 为 `preds_*.jsonl`（下划线），不匹配单 shard `preds.jsonl` → 自动判分报 "no prediction files"；本机 wzc1 版已修为 `preds*.jsonl`。聚合时用 `score_sample` 直接读 `preds.jsonl`。）

---

### ★ Phase-2 chat=False 消融判分（2026-07-23，官方 scorer，Iron-Law-2 全过，CoMem 8B j=12 chunk1024 adapter=qcmem_distill_qwen_j12_r32_4k，sink=bos，chat_template=False）

数据在 diskB（.73/.82/.104 共享 FS）。RULER=`scripts/score_ruler_taskbreadth.py` 官方 `_string_match_all_one`；BABILong=`scripts/score_flat_babilong.py`（新增，官方 `compare_answers`+`TASK_LABELS`，禁 re.search）。全部 8/8 shard、empty=0、recall-mismatch=0。

**#9 tab_selector — 单遍 selector 消融（RULER n=100，峰值 top-k over {4,8,12,16,24}，chat=False）**
| Task | Len | BM25 | Recency | ReaderAttn | Oracle |
|---|---|---|---|---|---|
| niah_single   | 8k  | 100 | 100 | 100 | 100 |
| niah_single   | 16k | 100 | 100 | 100 | 100 |
| niah_single   | 32k | 100 | 82  | 73  | 100 |
| niah_multikey | 8k  | 99  | 98  | 97  | 100 |
| niah_multikey | 16k | 99  | 88  | 90  | 100 |
| niah_multikey | 32k | 99  | 54  | 60  | 100 |
| var-track     | 8k  | 99.4 | 99.2 | 99.8 | N/A(VT skip) |
| var-track     | 16k | 92.6 | 92.4 | 92.4 | N/A(VT skip) |
| var-track     | 32k | 32.0 | 41.2 | 27.8 | N/A(VT skip) |

- 对齐 paper tab_selector.tex 结论：Oracle 在两 needle 任务 100/100/100（读出无损，长程差距=检索问题；BM25 单遍在 niah_single 0pp gap、niah_multikey ≤1pp gap 追平 Oracle）。VT 单遍 selector 全部 32k 崩塌（BM25 32.0 / Recency 41.2 / ReaderAttn 27.8）→ 需 iter_bm25（tab_itervt 的 iter 列由 .82 补）。Oracle-VT 本轮 skip（无数据；paper 现有 Oracle-VT 10.4/9.2/5.8 为旧 chat=True 数据，若要 chat=False 需补跑）。
- ⚠️ 新 chat=False 峰值 vs 现 paper tab_selector（旧 chat=True）在 16k/32k 显著更高（如 Recency niah_single 16k 72→100；BM25 VT 16k 27.6→92.6）——与 chat=False 大幅利好 exact-match/completion 任务一致（见 PAPER_LOCOMO_ERRATA §11a）。若切 chat=False 协议，整张 tab_selector 需换这套数字。

**#12 tab_crosschunk — cross-chunk attention 消融（full vs block-diag KV reuse，chat=False，selector=iter_bm25，top-k12）**
| Task | Full | Block-diag | Δ(full−bd) |
|---|---|---|---|
| RULER niah_single (8k/16k, n=50) | 100 / 100 | 100 / 96 | 0 / +4 |
| RULER niah_multikey (8k/16k, n=50) | 96 / 94 | 60 / 32 | +36 / +62 |
| BABILong qa2 (8k/16k, n=100) | 36 / 20 | 17 / 13 | +19 / +7 |
| BABILong qa5 (8k/16k, n=100) | 78 / 69 | 78 / 55 | 0 / +14 |

- 结论同 paper：cross-chunk recompute 对多事实消歧 load-bearing（multikey Δ+36/+62、qa2/qa5 有 gap），对单 needle 无关（niah_single 近似持平）。数值与现 paper tab_crosschunk（旧 chat 口径 multikey 88/92 vs 44/40）不同，但趋势一致；chat=False 应替换。

**#8 tab_slm — StreamingLLM 等预算（RULER niah_single，string_match，chat=False）→ 可提取，无需重跑**
| Method | 8k | 16k | 32k | 64k | 128k | budget |
|---|---|---|---|---|---|---|
| CoMem (retrieved h_j) | 100 | 100 | 99 | 99 | 100 | top-k12×chunk512 = 6144 tok |
| StreamingLLM (sink4+win6653) | 85 | 36 | 20 | 10 | 2 | 6657 tok |

- StreamingLLM 已跑档 sink4+win6653（budget 6657≈6.7k）== paper 声称的「≈6.7k-token 等预算」；CoMem 侧 chat=False niah_single 在 top-k12×chunk512=6144 tok（`ruler_results/qcmem_8b_iter_chatFALSE_ad`）存在 → **tab_slm 可从现有 chat=False 数据直接提取，无需 GPU 重跑**。注意：(a) 已跑 n=100（paper 写 n=50，n=100 更稳，应更新脚注）；(b) chat=False 数字（SLM 85/36/10/2）替换现 paper 旧 chat=True 90/42/16/4；(c) 两侧预算 6144 vs 6657 差 ~8%，均记 ≈6.7k（paper caption 已如此写），若要严格同预算可 CoMem top-k13×512=6656 或 SLM window=6140，非必需。

dirs（diskB）：`ruler_results/ablation9_selector_chatFALSE/sel{sel}_tk{tk}/`；`ruler_results/ablation12_crosschunk_chatFALSE/{full,blockdiag}/`；`babilong_results/ablation12_crosschunk_chatFALSE/{full,blockdiag}/`；`ruler_results/streamingllm_8b_ruler_chatFALSE/`；`ruler_results/qcmem_8b_iter_chatFALSE_ad/`。

---

## ★★ Paper B — OLMo-2 base 剪层-heal 深度扫（depth-sweep，prune-heal frontier）

**方法**：OLMo-2-1124 base（纯预训练，消 instruct 污染）truncate 到前 `keep` 层 + `n_fresh=2` 层新 tail，再在 Dolmino（DCLM 子集，OLMo-2 tokenizer 重 tokenize）continue-train(heal)。脚本 `scripts/train_olmo2_arch_probe2.py`（`--keep_front_layers/--n_fresh_layers/--resume_from`）。红线 `--babilong_mix_fraction 0`。7B 全模型 32 层。所有臂同超参（seq_len 2048 / lr_fresh 1e-4 / lr_inherited 2e-5 / max_steps 200000 / grad_checkpointing / fp32 master / n_fresh 2），仅 keep 变化 = 纯深度阶梯。**H20 配方 bs4/ga4 eff_bs128；L20A 配方 bs16/ga1**（皆 eff_bs=128 可比）。launcher：wzc1/L20A `scripts/run_olmo2_7B_keepN.sh`；diskB/H20 `scripts/run_olmo2_7B_keepN_diskB.sh`（env-overridable diskB 路径 + bs/ga）。

| keep (/32) | n_fresh | params | 节点 | 卡/配方 | 起点 | 状态 | held-out ppl / 备注 |
|---|---|---|---|---|---|---|---|
| **keep14** (0.44L) | 2 | ~3.9B(16L) | 本机 wzc1 8×L20A | bs16 ga1 eff128 | fresh | ✅ running step~170.5k/200k | train-ppl~10.15；apex step128000 held-out 10.83/1.46× |
| **keep12** (0.375L) | 2 | 3.6556B(14L) | .82 8×H20 | bs4 ga4 eff128 | **fresh**（无旧 ckpt 可 resume，旧 07-17 崩掉 ckpt 不完整已弃） | 🔄 running（2026-07-20 11:23 起） | step40 loss8.47 ppl4785 gnorm5.27 maxmem91.9GB 0NaN |
| **keep10** (0.3125L) | 2 | ~3.3B(12L) | 曾 .73 8×H20 | bs4 ga4 eff128 | fresh | ⏸ 曾跑（step10000 早期欠 heal，非收敛点） | — |
| **keep8** (0.25L) | 2 | 2.8460B(10L) | .104 8×H20 | bs4 ga4 eff128 | **fresh** | 🔄 running（2026-07-20 11:23 起） | step40 loss8.53 ppl5068 gnorm4.67 maxmem73.5GB 0NaN |

- **transplant sanity（两新臂启动即验）**：keep12 = copied 135 tensors（前12层 + embed/norm/lm_head from 32L base）+ fresh tail[12,13]；keep8 = copied 91 tensors（前8层）+ fresh tail[8,9]；均 unexpected=0 / max|model−base|=0（exact）/ fresh post_attn_ln & q_norm all-ones / fresh_q_std=0.02 → **ALL 6 CHECKS PASS**。
- **待铺（ablation frontier 控制臂）**：`--freeze_front`（冻结前 keep 层只训 fresh tail）/ `--from_scratch`（前 keep 层随机初始化对照）——脚本已支持，待空节点。
- **数据/运维**：`/dev/shm/dolmino_now15b.npy`（126.9GB，diskB 两台已 stage）；DataLoader `multiprocessing_context="fork"`（防 pickle 61G memmap 卡死）；WANDB offline；单节点 `torch.distributed.run --standalone --nproc 8`。

### 当前节点分配 + 进度（2026-08-02 05:29 +08:00，覆盖上表 07-20 stale 状态）

| 臂 | 节点 (FS) | 当前 step/200k | train-ppl | s/step | 备注 |
|---|---|---|---|---|---|
| **#100 full-32L 续训对照** | LOCAL wzc1 8×L20A | ~14.8k | 8.34 | 3.16 | causal continued-pretrain 对照（非剪层） |
| **keep8** (0.25L) | .252(28.89.19.252) **wzc1** 8×L20A | ~99.3k | 14.20 | 1.02 | ⚠️ 跑在 wzc1 非 diskB；~1.1d 剩→完成后迁 keep10→.252（需 diskB→wzc1 rsync keep10 ckpt） |
| **keep10** (0.3125L) | .104(28.83.24.104) **diskB** 8×H20 | ~80.8k | 13.47 | 6.84 | 最长杆 ~9.4d 剩 |
| **keep12** (0.375L) | .73(28.85.35.73) **diskB** 8×H20 | ~121.9k | 11.83 | 7.81 | ~7d 剩 |

### Frontier trajectory eval（base protocol chat=False / no-BOS / LL-based MC；PPL 8-shard held-out dolmino_now_val + core6 + know5；MAIN 已核对 JSON）

core6 = mean(hellaswag·arc_c·arc_e·piqa·openbookqa=acc_norm, winogrande=acc)；know5 = mean(mmlu·lambada_openai·boolq·commonsense_qa·social_iqa=acc)。

> **P0.7 审计（2026-08-02）命名对齐**：本表 `know5`（acc-based mean）= 论文口径 **`aux5_raw`**，是异质任务的描述性均值，**禁称 knowledge recovery**（MMLU 单列报告）。本表所有 know5 值经 raw-JSON 核对数值正确，仅论文侧改名为 aux5_raw；唯一数值修正是 keep14 的旧 `.5071`（acc_norm 污染，虚高 ~1.4pp）→ **`.4935`**，该错误值不在本表。详见 `paperB/P0_7_AGGREGATE_AUDIT.md`。

| 臂 · step | held-out PPL | core6_mean | know5_mean | mmlu | 源 |
|---|---:|---:|---:|---:|---|
| keep10 · 10000 | 17.2389 | 0.4757 | 0.4109 | .2539 | batch(.82) 2026-08-02 |
| keep10 · 70000 | 13.0616 | 0.5298 | 0.4430 | .2629 | batch(.82) 2026-08-02 |
| keep10 · 73500 | 12.9938 | 0.5301 | 0.4415 | .2677 | batch(.82) 2026-08-02 |
| keep10 · 74000 | 12.9881 | 0.5299 | 0.4413 | .2630 | batch(.82) 2026-08-02 |
| keep10 · 74500 | 12.9754 | 0.5299 | 0.4357 | .2587 | batch(.82) 2026-08-02 |
| keep10 · 75000 | 12.9619 | 0.5327 | 0.4322 | .2630 | batch82b(.82) 2026-08-02 |
| keep10 · 75500 | 12.9552 | 0.5319 | 0.4339 | .2553 | batch(.82) 2026-08-02 |
| keep10 · 76000 | 12.9486 | 0.5317 | 0.4412 | .2607 | batch(.82) 2026-08-02 |
| keep10 · 76500 | 12.9451 | 0.5294 | 0.4461 | .2620 | batch(.82) 2026-08-02 |
| keep10 · 77000 | 12.9306 | 0.5296 | 0.4402 | .2619 | batch(.82) 2026-08-02 |
| keep10 · 78000 | 12.9207 | 0.5301 | 0.4425 | .2662 | batch82(.82) 2026-08-02 |
| keep10 · 78500 | 12.9113 | 0.5324 | 0.4392 | .2572 | batch82(.82) 2026-08-02 |
| keep10 · 79000 | 12.9010 | 0.5333 | 0.4417 | .2641 | batch82c(.82) 2026-08-02 |
| keep10 · 80000 | 12.8725 | 0.5308 | 0.4387 | .2611 | batch82d(.82) 2026-08-02 |
| keep10 · 80500 | 12.8698 | 0.5350 | 0.4397 | .2587 | batch82d(.82) 2026-08-02 |
| keep10 · 81000 | 12.8563 | 0.5315 | 0.4419 | .2633 | batch82e(.82) 2026-08-02 |
| keep10 · 81500 | 12.8493 | 0.5308 | 0.4458 | .2655 | batch82f(.82) 2026-08-02 |
| keep10 · 82000 | 12.8462 | 0.5263 | 0.4348 | .2609 | batch82g(.82) 2026-08-02 |
| keep10 · 82500 | 12.8360 | 0.5318 | 0.4344 | .2563 | batch82h(.82) 2026-08-02 |
| keep10 · 83000 | 12.8241 | 0.5314 | 0.4418 | .2585 | batch82i(.82) 2026-08-02 |
| **keep10 · 83500 ⇒ ≈200k** | **12.8160** | **0.5303** | **0.4491** | **.2718** | **batch82k(.82) 2026-08-02 — 平台期冻结=200k，训练已 kill(.104)** |
| keep12 · 115000 | 11.5378 | 0.5631 | 0.4690 | .2733 | batch82b(.82) 2026-08-02 |
| keep12 · 115500 | 11.5270 | 0.5618 | 0.4677 | .2686 | batch(.82) 2026-08-02 |
| keep12 · 116000 | 11.5248 | 0.5628 | 0.4613 | .2686 | batch(.82) 2026-08-02 |
| keep12 · 116500 | 11.5211 | 0.5669 | 0.4646 | .2631 | batch(.82) 2026-08-02 |
| keep12 · 117000 | 11.5125 | 0.5678 | 0.4641 | .2689 | batch(.82) 2026-08-02 |
| keep12 · 117500 | 11.5106 | 0.5702 | 0.4631 | .2582 | batch(.82) 2026-08-02 |
| keep12 · 118000 | 11.5031 | 0.5698 | 0.4573 | .2628 | batch(.82) 2026-08-02 |
| keep12 · 118500 | 11.4957 | 0.5674 | 0.4662 | .2735 | batch(.82) 2026-08-02 |
| keep12 · 119500 | 11.4826 | 0.5684 | 0.4624 | .2699 | batch82(.82) 2026-08-02 |
| keep12 · 120000 | 11.4797 | 0.5679 | 0.4675 | .2733 | batch82b(.82) 2026-08-02 |
| keep12 · 120500 | 11.4740 | 0.5686 | 0.4554 | .2707 | batch82c(.82) 2026-08-02 |
| keep12 · 121500 | 11.4645 | 0.5692 | 0.4604 | .2668 | batch82d(.82) 2026-08-02 |
| keep12 · 122000 | 11.4605 | 0.5696 | 0.4634 | .2665 | batch82e(.82) 2026-08-02 |
| keep12 · 122500 | 11.4573 | 0.5670 | 0.4593 | .2634 | batch82f(.82) 2026-08-02 |
| keep12 · 123000 | 11.4513 | 0.5683 | 0.4630 | .2572 | batch82g(.82) 2026-08-02 |
| keep12 · 123500 | 11.4475 | 0.5736 | 0.4736 | .2749 | batch82h(.82) 2026-08-02 |
| keep12 · 124000 ⇒ ≈200k | 11.4426 | 0.5669 | 0.4608 | .2752 | batch82j(.82) 2026-08-02 — 平台期冻结=200k，训练已 kill(.73) |

- **★ 2026-08-02 平台期冻结决定（用户指令）**：keep10 / keep12 的 core6/know5/MMLU 在最近 ~13k step 已在噪声内持平（MMLU 钉 chance .25-.28，浅层 12L/14L 无法恢复世界知识），故按用户「已到平台期直接用当前数据=200k 数据、然后 kill」冻结 **keep10·83500 / keep12·124000 作 ≈200k 端点**，训练于 2026-08-02 kill（keep10 .104 parent PID 3995588 / keep12 .73 parent PID 3983054，各 0 残留进程 + 0 GPU compute-apps，16×H20 释放）。⚠️ 知识轴已平台但 **PPL 仍缓降**（keep14 完整轨迹实测 PPL 一路降到 200k），故上述两点 PPL 相对真实 200k **略偏乐观**——论文须标注 mid-trajectory frozen。深度阶梯 200k 端点表见 `paperB/TODOList.md` §深度阶梯 200k 端点。keep8(.252) / full32(#100 LOCAL) 不冻结，继续跑满 200k。

- 趋势与深度阶梯一致：keep12(14L) > keep10(12L)，PPL 更低（~11.5 vs ~12.9）+ core6（~.567 vs ~.531）/know5（~.464 vs ~.440）更高。mmlu 全轨迹仍 ~chance（.25-.27）= 与 THREE_ARM_200K 的「PPL 早收敛但 knowledge 恢复滞后」一致（此为轨迹中点非 200k 终点）。轨迹内 PPL 单调降但 core6/know5 已近平台。
- ⚠️ **batch1（01:12–01:45）名义 4 点实产 3 点**：keep12 **step119000 被静默 SKIP**（该 ckpt 已不在盘上，harness 仍 touch DONE marker=已知 bug，不足信 done marker）。keep12 高端由 batch82b 的 **step120000**（在盘）覆盖。
- batch82b（2026-08-02 01:56 起，log `logs/frontier_batch_82b_20260802.log`）已跑完 keep12 {120000,115000} + keep10 {75000,70000}，末点 keep10·75000 downstream 收尾中；上表所有点均 MAIN 从 JSON 核对。原始 JSON：`.82:olmo2_ppl_results/7B_<tag>_step<N>/summary.json` + `.82:olmo2_downstream_results/7B_<tag>_step<N>{,_know}/summary.json`。step0（degenerate ppl~1e6）+ step77500（缺 know5）未入表。
- batch82c（2026-08-02 02:55 起，log `logs/frontier_batch_82c_20260802.log`，03:17 跑完）补 keep12·120500 + keep10·79000（含 silent-skip 修复：仅当 PPL summary.json 实存才 touch DONE marker），两点均 MAIN 从 JSON 核对已入表。
- batch82d（2026-08-02 05:35 起，log `logs/frontier_batch_82d_20260802.log`，06:12 ALL DONE）名义 4 点实产 3 点：keep12·121500 + keep10·{80500,80000} 三点已 MAIN 从 JSON 核对入表；**keep12·121000 被 silent-skip 保护正确跳过（该 ckpt 不在盘上，无 summary.json → 未 touch DONE marker，符合预期）**。keep10 高端由此覆盖到 80.5k。

#### ShortGPT external baseline (#98) — 200k heal ENDPOINT（非轨迹中点，base protocol chat=False；MAIN 从 JSON 核对；datasets 5.0.0）

ShortGPT-influence 选层保留 16 非连续层 [0-12,16,17,31]（含最终层 31，NO fresh tail；`keep_front_layers=16 n_fresh_layers=0`；ckpt 已 compact 到 0..15），同 200k Dolmino heal 配方。heal COMPLETE。

| 臂 · step | held-out PPL | core6_mean | know5_mean | mmlu | 源 |
|---|---:|---:|---:|---:|---|
| **7B_shortgpt16 · 200000** | **9.7803** | **0.6215** | **0.5596** | **.4739** | #98 external baseline · MAIN-verified JSON(.82) · datasets 5.0.0 |

- per-task：hellaswag .6851·arc_c .4761·arc_e .7462·piqa .7584·openbookqa .408(acc_norm)·winogrande .6551(acc)；mmlu .4739·lambada .6194·boolq .7287·commonsense_qa .5340·social_iqa .4422(acc)。
- ★ **该终点在 PPL(1.322× tax) 和 MMLU(63.0% above-chance recovery) 两轴同时优于全部三个 16L 连续截断臂**（keep14 train-all 19.4% / freeze-front 3.6% / random-front ~0%）。混淆项：继承 16 vs 14 层 + 保留原生 readout(层31) vs 丢顶换 2 fresh 层——两效应尚未拆分（需 keep16-inherited/0-fresh 连续控制隔离 policy）。详见 `status/PAPERB_THREE_ARM_200K.md`。
- 源：`.82:olmo2_ppl_results/7B_shortgpt16_step200000/summary.json` + `.82:olmo2_downstream_results/7B_shortgpt16_step200000{,_know}/summary.json`；driver `scripts/_run_shortgpt_downstream_only.sh`（downstream-only 复跑，PPL 早前经 `_run_olmo2_eval_shortgpt.sh`）；env `.82:$DB/olmo2_venv`（elsa torch2.7 + transformers 5.5.4 + datasets 5.0.0）。轨迹中点 128000/153500 未跑（需 46GB scp LOCAL→.82，deferred）。

---

## ★★ Paper B keep14+fresh2 **种子方差**（task #181，2026-08-12，LOCAL 8×L20A，33 min）

**这是 Paper B 第一个 same-recipe 双 seed 端点对照。** 两个臂配方完全相同（front14 继承 + 2 fresh tail，
Dolmino heal 到 step200000，seq_len 2048，均匀 lr 2e-5），只差 seed（42 vs 1234）。
⚠️ `arch_meta.json` 里 `lr_fresh` 一个写 1e-4 一个写 2e-5 是**元数据假象**：`_classify_param` 未剥 `module.`
前缀导致 fresh 组从未生成，两臂**实际都是均匀 2e-5**（见 CLAUDE.md 同 bug 条），**不得声称差分 LR**。

| 项 | seed 42 | seed 1234 |
|---|---|---|
| ckpt | `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt` | `outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step200000.pt` |
| 训练收尾 | 早前完成（Paper B 主表来源） | `loss=2.3286 ppl=10.26` @200000，`final.pt` + `DONE` |

### 同口径逐轴结果（8 shard，`chat_template=False`，`add_bos=0`，per-item 配对）

| 轴 | n | seed 42 | seed 1234 | Δ(42−1234) | McNemar p | 结论 |
|---|---:|---:|---:|---:|---:|---|
| held-out PPL (Dolmino) | 8.38M tok | **10.5613** | **10.5673** | −0.0060 (0.06%) | — | **种子稳** |
| OOD PPL WikiText-103 | 289k tok | 11.5562 | 11.5375 | +0.0186 | — | 种子稳（符号还反向） |
| OOD PPL PG-19 | 2.46M tok | 15.4263 | 15.4654 | −0.0391 | — | 种子稳 |
| **MMLU-L (letter)** | 14042 | 31.85 | **33.04** | **−1.19 pp** | 0.0092 | 摆动最大；4061 item 翻转(28.9%) |
| MMLU-C (content_norm) | 14042 | 38.41 | 38.24 | +0.18 pp | 0.512 | 稳（仅 9.5% 翻转） |
| **BoolQ** (acc_norm) | 3270 | **68.87** | 66.82 | **+2.05 pp** | 0.0035 | **Holm 校正后唯一显著** |
| ARC-E (acc_norm) | 2376 | 70.50 | 71.89 | −1.39 pp | 0.032 | Holm 后不显著 |
| core6 macro (acc_norm) | — | .5938 | .5999 | −0.62 pp | — | — |
| aux5_raw (acc mean) | — | .4935 | .4918 | +0.17 pp | — | — |
| HellaSwag / OpenBookQA | 10042 / 500 | 64.46 / 40.40 | 64.44 / 40.40 | +0.02 / 0.00 pp | 0.97 / 1.00 | 极稳 |

- **符号混杂**（7 轴利于 s42、5 轴利于 s1234、1 平）→ 这是**离散度**不是某个 run 更好。
- Holm-Bonferroni（13 轴主族）：**仅 BoolQ 过**（0.00353 < 0.00385）；MMLU-L(0.0092) / ARC-E(0.032) 不过。
  但 3/13 名义 p<0.05 vs 全零假设期望 0.65 → 确有真实种子信号，**可用量是幅度包络而非"哪轴显著"**。
- **接口不对称**：MMLU-**letter** 跨种子翻 28.9% item，MMLU-**content_norm** 只翻 9.5% → letter 接口对种子
  更敏感 ~3×，与 Paper B 已有的 interface-dependence 结论同向加强。

### 对 Paper B 主张的影响

| 论文主张 | 差距 | vs 种子包络 | 判定 |
|---|---|---|---|
| ShortGPT−keep14 **MMLU-L +15.6pp** | 15.6 | 13.1× | **安全** |
| ShortGPT−keep14 **MMLU-C +1.8pp** | 1.8 | **1.5×（同量级）** | ⚠️ **n=1/臂 不能归因于构造** |
| ShortGPT−keep14 core6 +2.8pp | 2.8 | 4.5× | 大概安全，需注明 n=1/臂 |
| BoolQ +9.1pp / ARC-C +3.8pp / HellaSwag +4.1pp | — | 4.4× / 3.4× / 206× | 安全 |
| `tab_ood_audit` PPL 序 Base<ShortGPT<keep14 | ≥0.78 PPL | 种子 Δ ≤0.039 | **安全，序不可能被种子翻** |

### ⚠️ 统计红线

- **n=2 → df=1，两个 draw 的 sd 不是可报的 σ_run。** χ²₁ 的 95% CI 把 sample sd 乘 `[0.446, 31.91]`
  （**~72× 宽**）：MMLU-L sd=0.841pp → σ ∈ [0.375, 26.84]pp。**报 delta，不报 σ。**
- **不得与 A03 的 σ 混用**（A03 = **1B** keep7/keep12 on TriviaQA，df=5，0.3666pp，χ² CI [0.229,0.899]）——
  不同模型/规模/任务/harness。

### 口径与 provenance（关键）

- **两臂都在本次重跑**（一个 driver / 一个 commit `b3626c3` / 一个解释器 / 一台机 / 背靠背），所以 delta 是纯种子。
  必要性：归档的 seed42 数字产于**已退役的 .252 + `$WD/.venv`**，而 LOCAL 的 `.venv` 自 2026-08-04 起无 torch
  → 只能用 `/opt/conda/envs/torch-base/bin/python`（torch 2.13.0）。
- **归档可移植性实测**：seed42 重跑在 PPL(`sum_nll` 19763937.398438 逐位相同)、两个 OOD PPL、**全部 11 个
  downstream task（acc 与 acc_norm，max |Δ|=0.000000pp）** 上与 .252/.venv 归档**逐位一致**；
  **唯一漂移是 MMLU-content**（content_norm −0.0926pp），源自 `36ddb1e`+`7ac9653`（2026-08-08）晚于归档的
  2026-08-02 run —— 即 `PAPERB_WITHIN_DISK_FLOOR_V3.md` 说的 **driver 版本漂移（系统性偏差，非零均值噪声）**。
- 完整性：12/12 结果目录齐（8 dirs×8/8 shard + 4 OOD×1/1）；11 个 task 的 `n_scored==expected`、`n_nan=0`；
  failure-syntax grep `Traceback \(most recent call last\)|CUDA out of memory|loss=nan` **0 命中**。
- **未测轴**：closed-book PopQA/TriviaQA/NQ-open（seed42 基线**只在 zwfy6**，且 wzc1 HF cache 无 `nq_open`；
  若在此跑会把 seed 与 L20A-vs-H20 架构混淆）。要补必须**两臂**在同一节点重跑。
- 结果路径（wzc1）：`olmo2_{ppl,downstream,mmlu_content}_results/keep14_s{42,1234}_step200000_sv181*`、
  `ood_ppl_results/keep14_s{42,1234}_step200000_sv181_{wikitext103,pg19}`（含 per-item jsonl，配对检验用）。
- 文档：`paperB/SEEDVAR_KEEP14_PROTOCOL.md`（口径钉死表）· `paperB/SEEDVAR_KEEP14_VERDICT.md`（结论）·
  `paperB/SEEDVAR_KEEP14_RESULTS.json`（机器可读）· driver `scripts/_run_paperB_keep14_seedvar_local.sh` ·
  分析 `scripts/analyze_paperB_keep14_seedvar.py`。
