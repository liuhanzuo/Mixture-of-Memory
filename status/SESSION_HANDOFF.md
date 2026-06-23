# SESSION_HANDOFF.md — compact / 新会话交接文档

> **本文件是 compact 后或新会话启动时的第一手交接。** 读完这份 + `status/RUN_REGISTRY.md` §3/§4 + `status/TRAINER_ACTIVITY.jsonl` 尾部，就能接上当前研究状态。
> 维护规则：main agent 每当方向/结论/在跑实验有重大变化时，**覆盖更新本文件的「当前快照」区**（保持精简，旧结论沉淀到 RUN_REGISTRY）。
> 最后更新：2026-06-19 21:05 GMT+8

---

## 0. 一句话现状

★★★**当前主线（2026-06-21）= self-study 蒸馏「长训不崩」**——核心判据：让后期 ckpt(step1000+) qa1 W0 ≥ step500，长档不随步数单调下降。已搞清三种 teacher 的退化机制（详见 RUN_REGISTRY「蒸馏退化退化机制总账」小节），结论：**杠杆 2/3/4 是真正有用、能长训不崩的方向**：
- **【杠杆2】训练长度对齐评测（pg19 真长文 n_ctx7）**：唯一已验证「训久不退化」（step250≈final）+ 破 16k 墙（=16 vs 现有方法≈13）。优先级最高，下一步推有效训练窗口→≥32k 攻 32k 硬墙。
- **【杠杆3】读出侧补容量鸿沟（SWA gap）**：memory 里确实存了长程信息（swa2 比 swa0 多读出近 2×），退化部分来自「读不出」非「没存」→ 改 W0 读出 > 再堆训练（机制侧 > 训练侧）。
- **【杠杆4】confidence 加权蒸馏（confB1 warm-start）**：过滤「老师自信但 memory 支撑不了」的不可学 token，掐断「后期死磕不可实现 target」退化路径。正在 .196 验证。
- 退化机制速记：**W2-teacher**（近视，memory+2chunk窗口）= 老师在迁移轴上不比学生强，逼学生抄不可见窗口 → 急崩（seed1234 step500 11/7/2/1 → step1000 5/0/1/1）；**全上下文 teacher（dolmino≤16k）** = 容量鸿沟 + 过拟合短训练长度 → 温和退化（17/19/12/8 → 16/11/8/5）；**全上下文 + pg19 真长文** = 长度对齐 → 基本不退化。

<details><summary>（旧）2026-06-19 Landmark diff-based 迁移方向</summary>

★★★**方向已 PIVOT 到「复现 Landmark → 列差异表 → 逐步迁移」的 diff-based 调试**（2026-06-19，用户定方法论）——不再 patch 自己 broken 的 mem_space + 猜，改为从一个已知能破长程墙的方法（Landmark Attention）出发，一次只改一个差异向我们迁移，哪一步 passkey 断崖 = 那个差异就是长程杀手。

**进度**：
- **Phase 1 复现 ✅ 完成**（commit 62c2d68）：在 **LLaMA-1-7B**（非 L2，wdiff checksum 49798 验）上忠实复现 Landmark passkey 破墙——base 在自身 2048 ctx 断崖（2.2k→4k：98%→0%），landmark-mem 全程 94-100% 到 ~31k。独立 venv `external/landmark_venv`（torch2.1+tf4.28.1，non-triton，不污染主 .venv），harness 在 `external/landmark/`，recover 的 tuned ckpt 在 `external/landmark_ckpts/landmark_tuned`（26G 保留）。这是可信迁移锚点。
- **Phase 2 差异表 ✅ 完成**（commits a469a8b/88ed1a5，`docs/LANDMARK_VS_MEMSPACE_DIFF.md`）：7 轴（解冻/base/数据/ctx-block/检索/读出/记忆单元）。守门=passkey 主+BABILong qa1 对照，PPL 排除。
- **Phase 3 迁移进行中**（2-group 并行，用户指令每实验用满 2 节点组 16 卡）：
  - **Group-A（本机+.196 diskA）= S2 数据轴**：✅✅**完成并裁决（2026-06-19 20:52）**：dolmino(wiki+pes2o)替 RedPajama-7源,其余=锚点,2-node FSDP+IB(10.7GB/s,2.48s/it,2h7m,0crash)训到 step3000 train_loss1.93。**passkey 主门(0/4k/8k/15k/30k/60k/115k garbage,top_k5,n=50)全 3 ckpt 评完**:step3000=**0/92/82/78/86/72/96%**,step1000→3000 单调上升,**无任何 length 出现 cliff**(0k=0% 是无 filler 的已知 harness quirk,全 step 一致非回归)。**★裁决:数据轴(RedPajama→dolmino 短/双源)不是长程杀手** —— anchor 机制 + 换数据仍全程保 passkey。data-axis 排除,嫌疑收窄到 S3/S4/S5/S6(ctx-block/检索/读出/记忆单元)。ckpt: `external/landmark_ckpts/landmark_s2_dolmino/checkpoint-{1000,2000,3000}`,pooled csv `external/landmark/results_s2/`。⚠️BABILong qa1 control-gate 未跑(landmark ckpt 接 babilong harness 需开发,待主会话定)。Group-A 现空闲。
  - ★**运维金句(diskA 2-node 必用)**:NCCL_IB_DISABLE=0 + NCCL_IB_GID_INDEX=3(mlx5 RoCEv2,iface bond1) + static-rdzv(c10d rdzv 会 hang)→ 2-node 11× 提速(TCP 1.0GB/s→IB 10.7GB/s)。
  - **Group-B（.76+.249 diskB）= S3 训练期 multi-window ctx**（从 S4 重排而来）：landmark-s5 owner，Phase-A 调研中。

★★**本轮最重磅机制发现（landmark-s5 扫 llama_mem.py，重塑「检索坏」假设）**：**Landmark 从不训练 retriever**。训练时（单 512-window，past_key_value=None）走 else 分支(432-439)只跑 in-window grouped-softmax(469)，**top_k 块选择(318-430)只在 inference 存在**，是 train-free 复用 grouped-softmax 学出的 landmark-token key 的 trick。我们 mem_space 相反——**训练一个显式 cross-window selector**（实测 0% needle precision）。→ 真正的轴 = **train-time-emergent-selection（Landmark,work）vs trained-explicit-selector（我们,坏）**。推论：(1) S4「换 scorer 为学习头」无法单轴实现——训练 forward 不跑检索路径，学习头拿不到梯度；且 landmark token 同时是被打分对象+块 key（scorer 与 memory-unit 是同一物）。(2) S5 readout 与 S6 memory-unit 也 entangled（memory=每层 per-window KV cache 非可分离 bank）。(3) 要让可训练 selector 有梯度须改 multi-window 训练=S3，但 S3 也有坑：官方 FSDP recipe 在 grad-ckpt 下强制 use_cache=False(820-825) → 跨 window KV 不缓存 → 检索仍 dead；且 multi-window loop(1013-1037) 是 inference-only（last_logits 被覆盖 bug 证明从没当训练 loss 用过）。**故 S3 当前 gate = 先做 cheap 梯度流验证**（M=4×512 跑一次 fwd+bwd，确认检索路径 fire + 跨 window KV 梯度非零 + loss 累积全 window），梯度确认流动才启真训练。若梯度不流（KV offload 不可微）=本身是发现，pivot 到 S4b inference 消融。

**运维**：B200 offline。diskB(.76) faithful-env 资产已 rsync（venv 4.4G✓+base 13G✓+tuned 26G✓+code✓），RedPajama 待从 diskA S2 的 arrow cache rsync。2-node NCCL：diskA 需 static-rdzv+bond1+NCCL_IB_DISABLE=1（c10d rdzv 会 hang）；diskB 待 smoke。

</details>

<details><summary>（旧）2026-06-17 22:00 slot-evidence gating 快照</summary>

★★★slot-evidence gating + prefix retrain = **负结果，n=200 决定性 probe 闭账**（2026-06-17 22:00）——后续探索的「slot-evidence mid-layer gating」路线（heur/oracle 把 memory slot 当 evidence 在 L16 注入），用 n=200 niah_single_1 4k 同 seed42 5 臂决定性 probe 裁决：**OFF=23.5 / heur_pos0=23.5 / heur_realpos=25.5 / oracle_pos0=26.0 / oracle_realpos=21.0，全部落 OFF±2.5（n=200 噪声~3pt）内，无 arm 稳超 OFF>3pt**。两个硬结论：(1) 此前 run1 的 heuristic +28 坐实为噪声（n=200 最多 +2）；(2) **oracle 天花板根本没被打破**（oracle 反在 -2.5/+2.5 散在噪声内）→ 即便"完美"注入真实 in-chunk 位置 evidence 也救不动 4k niah。**slot-evidence gating 这条路证伪，finetune 取消。** 决策树触发 **PIVOT** 分支：候选 = Landmark in-attention mid-layer KV injection（非 prefix retrain，架构级改动）。**这是架构方向决策，超 heartbeat 自主权限 → 已 emit needs_code alert 等主会话定夺。** 5 节点 24+ 卡现全空闲（三训练 eval 全收尾 + probe 完成），无现成 auto_launch PENDING 可跑。⚠️ 注：主线「pg19 蒸馏攻 32k」此前已闭账（pg19 n_ctx7 蒸馏 final step500 仍最佳：W0 qa5 16k=16 破天花板，32k=9 持平）；evidence gating 是闭账后的补充探索，今已一并证伪。

<details><summary>（旧）2026-06-17 00:40 n_ctx15 快照</summary>

★★★n_ctx15「加大训练窗口」攻 32k = **负结果，已双 seed 确认**（2026-06-17 00:40）——把训练窗口从 4096(n_ctx7) 加到 8192(n_ctx15) **未破 32k 硬墙**。step250 双 seed qa5(0k-32k)：seed42=80/76/56/30/19/14/8、seed1234=90/--/--/--/10/12/6（seed1234 qa1/qa2 偏崩；qa5 长程 8≤9）。对比 n_ctx7 天花板 16k=13~16/32k=9：**32k 持平未破(8≤9)，16k 反略降(14<16)**。结论：**32k 硬墙不是「训练窗口不够大」能解** → 排除「加大窗口」这条路（科学负结果，有价值）。推断真瓶颈 = 32k→128slot 的 250:1 压缩比本身，或更长 per-sample context 致有效梯度更稀疏。当前在跑：本机 seed1234 final eval(pid 1014377) + .249 seed42 final eval（补齐 2seed×2step 确认网格）。**主线「pg19 蒸馏攻 32k」至此闭账**——pg19 n_ctx7 蒸馏 final(step500) 仍是最佳（W0 qa5 16k=16>所有方法天花板13 ✓，32k=9 持平）。下一方向需主会话决策（加大窗口/容量 sweep 均已证伪，剩余候选 = 改压缩比 250:1 / 换读出范式）。

<details><summary>（旧）2026-06-16 20:40 快照</summary>

★★★pg19 蒸馏破天花板 + n_ctx15 frontier（2026-06-16 20:40）——pg19 蒸馏（n_ctx7→训练窗口 4096）final(step500) W0：qa5=75/73/51/29/19/16/9。**16k=16>13 破天花板 ✓，32k=9 持平 ✗**。frontier=增大 n_ctx7→15（窗口 4096→8192）验证能否破 32k。→ 见 §0 顶部：已证伪。

</details>

</details>

</details>

<details><summary>（旧）2026-06-13 21:20 快照</summary>

★★★判据校准（2026-06-13 21:20，用户指令）——**不要拿"闭合 SWA gap"当成败线**：SWA(W6)效果好是**可预见的**（直接注意上下文原始 KV，等于开卷），纯靠 memory slot 预测**本来就很难**。所以 harder-objective / arch arm 的成败判据 = **W0（纯 memory 读出）相对 baseline 的提升**，而非"是否逼近 SWA 天花板"（后者是过高标尺，注定达不到）。按校准后的判据：harder-objective(`--last_chunk_loss_only`)把 W0 长程 qa5 从 BASE_mix0 的 5/5/3 抬到 ctx3 step500=13/8/9、step1000=13/9/6（≈翻倍）→ **方向是对的**，是温和但真实的纯 memory 长程提升。下一步看 ctx7（深 curriculum）W0 是否比 ctx3 更强。⚠️ **B200(.188) 暂 offline**（sshd 持续挂，用户确认先不管）→ 巡检按 4 节点（本机/.196/.76/.249）算，不再重试 .188；其上 seed1234 ckpt/eval 等节点恢复再说。

★★★3 路架构实验判决（2026-06-13）——**confound 已识破，BABILong 不可裁，转 LongBench**：3 路 arch arm（L1KEY 独立 key / L2ON L2 首启 / L1ERASE delta 擦除写）全部用 `--babilong_mix_fraction 0.0` 训练，而 P11 SOTA(48/45/44) 训练掺了 0.15 BABILong。**BASE_mix0 锚点（全 arch flag OFF 纯 dolmino）step500 qa5 长程 8k/16k/32k=5/5/3，比三路 arch arm 都低**（L1KEY step1000=17/10/9，L2ON=9/8/5，L1ERASE=10/6/6）→ 证实长程崩是 **mix=0 主导（训练不掺 BABILong），非 arch 有害**；三路 arm 都略高于 no-arch base 但被 mix=0 封顶，无法用"超 P11"判据裁定。BABILong 对 P11 比较根本不公平（P11 掺 BABILong=见过考题）。**按用户框架 LongBench(held-out 真实长文，从未进训练)才是真判据 → 4 个 mix=0 ckpt 全起 LongBench W0**（L1KEY/BASE/L1ERASE/L2ON，2026-06-13 17:20 在跑）。判据：LongBench 上 arch arm 是否超 BASE_mix0 锚点 → 超=arch 真有益(此前被 mix=0 BABILong 假象掩盖)，平=arch 无效。

★★框架修正（2026-06-13，用户洞察）：**"memory 全证伪"是错的判读。memory 确实在 work** —— 无 SWA 时（生成窗口仅最后 1 块，针几乎不在窗内）qa5 32k=44 远超随机(~20-30)，这 44 分**完全来自 memory bank**，是 memory 承载长程的硬证据。已证伪的只是**具体旋钮**（路由四臂 / L3 recon+diversity / delta-rule 这一写规则 / 容量 sweep / dense 全局写 / v20 读基生命周期），**不是 memory 范式本身**。
- ★eval-time SWA(W=6) 把 qa5 32k 44→73：BABILong 针是**全上下文均匀随机**(官方 RMT-team/babilong 预生成集)，swa6 在 32k 只直接覆盖最后 7/64 块≈11%，纯靠直接注意最多救 11% 样本(~50 分上限)，到 73 说明**剩余 89% 上下文的针靠 memory 答对**。→ SWA 增益的真正含义：**memory 没把长程信息榨干，读出效率不足**（原文里有 memory 没成功传递的信息），这是**改进空间不是证伪**。
- **当前主攻（用户定，3 路并行）= 让 memory 自己把 SWA 那 29 分也读出来**：L1 独立 key 破 dead-slot / L1 DeltaNet 擦除写 / L2 首次启用。判据：长程 qa5 超 P11 step500(48/45/44) 且逼近 swa6(73)。
- ★★容量 sweep = 噪声（2026-06-12 多 seed 证伪）：N384"全面超基准"不复刻——三 seed 形态全异（orig 全高 / seed1234 长程崩 16k=13 / seed2026 0k 崩到 0），N192 两 seed 也发散（0k 90 vs 22），无 seed 稳超 P11 base step500。**扩 memory 容量(128→384)不是有效杠杆，之前的突破是运气。** 铁律：单 run 分数只作筛选不作定论，候选必须 2-3 seed。
- ★dense 全局写(top_k=N) = 证伪：N128 dense 生成坍缩(qa5 含 2k=0%)，全槽 live 但破坏冻结 backbone。
- 写入侧：DRoff（换 dual-gate 写规则）长程不掉于 N384，**但 N128 DRoff 长程掉一半** → delta-rule 写规则价值随容量递减；正在跑 N192/N256/N512 DRoff/dense 画完整 crossover 曲线（只为补已证伪曲线）。
- ★读出侧（2026-06-12 判决=负）：v20 ArmA(soft-read-decay) step1000 长程 qa5 8k/16k/32k=40/37/36 ≈ N384 base 四种子均值(39/37/31)，无提升；ArmB(hard-eviction) 近崩(0/4/31/7/4/5/1) 有害。**读基槽生命周期改造未解锁长程。**
- ⚠️ 旧结论"mem-space 不优于长上下文/全证伪"已**作废**——见 §0 顶部修正。memory 在 work，是读出效率问题不是范式问题。

</details>

## 1. 核心认知（这两天用 ~20 个实验换来的，别重走）
- **P11 chunk512 step500 仍是 SOTA**（qa5 0k-32k = 74/89/81/60/48/45/44），迄今无配置超过它。
- **过训单调退化铁律**：几乎所有 run step500 ≫ step5000（0k 除外）。**训练默认只跑 1000 步**（save_interval 500），1k>500 才续。
- **路由集中(usage_cov≈0.25)是症状不是病根**：ROUTE-A 四臂（loss_free/entropy/temp/gumbel）全 REJECTED。arm4 把 usage_cov 强行拉到 1.0 反而长程最差（强制探索打散“少数槽精确命中”，top1_sim 0.99→0.11 崩）。**教训：绝不能在读/选择侧粗暴均摊。**
- **L3 summary 是长程主力**（L1-only 关掉 L3 后 qa5 1k=4 vs 89，L1 单独几乎不 work）。但 L3 调参也到头了：diversity 正则治了 token 坍缩(cos 0.99→0.18)却伤长程（剂量反向）；容量 sweep base64 最优，128/32 都退化（l3_tok_cos 恒=1.0，坍缩吃掉容量）。
- **读机制部件耦合**（D6 三臂）：独立 cross-attn 读有用(A≫B)；null-sink 是必要稳定器(A≫C，且 C<B，抽掉 null-sink 比不要 cross-attn 还糟)。
- **真病根 = 写入侧冷启动死锁**（用户洞察 + gp-59/gp-64 调研确认）：delta-rule 只写 top-k 选中的槽 → ~91/128 死槽永久冻结在 chunk-0 token 快照 → 永远不被选/不被写 → 富者愈富。**top-k 稀疏写本身是异常**（DeltaNet/Titans/fast-weight 的写规则都是 dense 全槽相似度加权）。
- **关键架构事实**：P11 用 `use_memory_xattn` 时，**读已是 all-N**（独立 MemoryCrossAttentionRead，自己的 softmax 看所有槽），top-k 只 gate **写入**。所以“宽写窄读”已半实现，写入侧改造**不影响读路径**——这是写入侧方案能避开 arm4 崩塌的结构性保证。
- **★读出侧也已证伪（v20，2026-06-12）**：试图让活槽主导读出的两臂都失败——ArmA(soft-read-decay 软衰减未写槽读权重) step1000 长程 qa5 8k/16k/32k=40/37/36 与 N384 base 四种子均值(39/37/31) 持平无提升；ArmB(hard-eviction 按累计读 mass 硬淘汰) 近崩(qa5 step500=0/4/31/7/4/5/1) 有害。**结论：读路径的"95%注意力落在未写槽"不是可通过生命周期规则修掉的 bug，而是该架构在此规模下的稳定工作点——强行干预读分布（无论软/硬）都伤长程，与 ROUTE-A arm4 教训一致（不能在读/选择侧粗暴均摊）。**
- **★memory 更新粒度 = chunk level（确认）**：序列切 chunk_size(512) 个 chunk，逐 chunk forward，每 chunk 一次写入；写入内容是该 chunk token 经 cross-attn 压缩出的 k 个 slot 向量（layer.py:1889 O_mem_hidden → :2037 O_mem_slot → :2071 write）。非 token-level 递归（区别于 DeltaNet/Titans per-token fast-weight）。用户确认 chunk-level 是预期设计。
- **★★写入侧三连诊断（2026-06-12）**：(1) dead_slot_read_mass：读路径 95% 注意力落在"从未写入"的槽（=上下文 token 快照），活槽只承载 ~5%，且 per-slot ratio=1.00（读对死/活槽零偏好，纯按 key 相似度均匀铺）。(2) read_mass 随 N：活槽读占比 N128=29%→N384=5%→N896=5%（池子越大写入越被未写 snapshot 稀释）。(3) **delta-rule-vs-dual-gate 对照**：把写规则从 delta-rule 残差式换成 dual-gate 式（DRoff，`--use_dual_gate` 仍在=仍在写，只是规则不同），长程 qa5 不掉（42/43/42 ≥ ON 均值）。**结论：delta-rule 这个具体写规则不是长程关键；记忆主要来自未写槽（原始 token 快照）的均匀池化。** ⚠️ 注意 DRoff 不等于"关写"——真正的 no-write 对照（dual-gate 也关）尚未做；但三条证据合起来已足够把优先级从"调写规则"移到读出侧。真杠杆 = 让读出有结构（v20 读基生命周期）或换范式。

## 2. 当前在跑的实验（2026-06-21 22:00）
| 节点 | 实验 | 说明 |
|---|---|---|
| 本机 H20 | **swateacherA W2 seed42**（W2-teacher 训练） | `mem_space_selfstudy_swateacherA_chunk512`，step~1400/2000。W2=memory+最近2chunk raw-KV 老师。from-base seed1234 已证退化，本 seed 训完一并归档。 |
| .196 | **★confB1-warm（杠杆4 验证）** | `confB1_warm`，step~640/1000。confidence 加权 SWA-teacher，**从 vanilla self-study step500 续训**。判据：续训能否修复退化、W0 超 17/19/12/8。 |
| B200 .53 | W0 eval（comprehensive swateacherA_w0eval） | seed1234 step500/1000/1800 W0 BABILong，含 §RUN_REGISTRY 已记的 seed1234 退化结果。 |
| B200 .18 | W1 W0 eval | `swaA_W1_W0_eval` step500/1000。W1=memory+最近1chunk 窗口，验证「窗口越小退化越轻」机制。 |

> 旧（landmark/写入侧三臂）实验已收尾，详见下方折叠历史。

<details><summary>（旧）写入侧三臂 + landmark 迁移在跑实验</summary>

| 节点 | 实验 | 说明 |
|---|---|---|
| .76 | **★T2 curriculum 4K→32K（关键）** | `T2_recall_chunk1024_CURRIC_4to32k_H20bs4ga4_N128`。渐进拉长 needle→query 距离(n_ctx 4→8→16→32=4K→8K→16K→32K,每125步一阶)攻 32k 长程墙。dolmino 锁 n_ctx=3(dadc19f 修 DDP hang)。**2026-06-15 17:00 从 dead B200 迁来**:B200(.52/.188)双双 No route to host,bs8×ga2→H20 OOM,改 bs4×ga4(eff 128 不变)修好。500 步,save_interval 125→每阶段末出 ckpt 看轨迹 |
| 本机/.196/.249 | T2 SWA×chunk×seq 矩阵 eval | swa6/W6 强基线格填充(c1024 swa6 / c1024_seq16k_W6 等),W0 已大体齐 |
| B200 .52/.188 | **DOWN** | No route to host(.52)/denied(.188),用户确认先不管;其上 curriculum 已迁走,seed1234 ckpt 等节点恢复再说 |

诊断：EXP-D（commit de7e499）给 QUERY_DIAG 加了 l3_attn_mass / l3_tok_cos；EXP-D2（f516d0a）加了累计 dead_slot_frac / max_slot_select_count。

</details>

## 3. 待办 / 储备（coder 已备料或调研完）
- **EXP-W2 dense soft write**（gp-64 设计，待派 coder）：所有槽按相似度做极弱软更新（DeltaNet 式），新参 `--soft_write_weight/--soft_write_content`，新增 memory_bank.soft_write()。判据加“slot pairwise cosine 不上升”（防趋同）。**与 R1 代码隔离**（R1 改 recycle 块 + force_write，W2 改写入段，独立开关）。配套 EXP-D3 加 slot 内容 pairwise cosine 诊断。
- **BABILong harness 样本分片**（gp-66 在做）：给 scripts/run_babilong_mem_space.py 加 `--num_shards/--shard_index`，让 32k 等长档样本拆多卡，eval 提速 2-3 倍。
- **判决性意义**：R1/W1/W2 都在检验“L1 长程天花板(≈0) 是死锁造成 vs 固有”。usage_cov↑且长程↑→重开 L1；usage_cov↑但长程平→L1 固有天花板，全力转 L3/换架构。

## 4. 集群 & 运维（关键，否则会踩坑）
- **5 节点**：本机(盘A) + .196(盘A 共享) + .76/.249(盘B share_304376610) + B200 .188(wzc1 share_304376610 独立)。盘间不共享 FS，ckpt 在哪个盘只能在哪个盘的节点评。
- **密码文件**：configs/password_diskA.txt(.196)、password_h20_new2.txt(.76/.249)、password_b200_188.txt(B200,末尾含逗号)。
- **PYBIN**：本机/.76/.249/B200 用各自 `.venv/bin/python`；.196 用 `/opt/conda/envs/torch-base/bin/python`。
- **⚠️ heartbeat 告警链路已废弃**：`status/HEARTBEAT_ALERTS.jsonl` 的 cron 探针因 token 过期(6-09 04:00起)一直 auth 失败，**不要依赖它**。现由 **main 主动巡检**：每轮真查 5 节点 GPU(nvidia-smi/ssh)，空闲立即起任务，落账 TRAINER_ACTIVITY.jsonl。session-only cron 370e2b20 每 20min 兜底。**绝不能只看告警文件就报"无事"——必须真查 GPU。**（曾因此空转 8 小时）
- **worker 爱误报"完成"**：醒来瞥一眼就空返回是常见 bug；务必自己核 GPU 进程数 / csv 数 / DONE_SCORING 标记，别轻信。所有后台任务用 `setsid nohup ... </dev/null &` 脱离会话（否则 agent 退出连带 kill 子进程）。
- **训练红线**：eval_interval=0（内联 eval 会 NCCL 崩）；git commit 不加任何 AI/Claude 署名 trailer，committer=LiuHanzuo。

## 5. 关键文件
- `status/RUN_REGISTRY.md` §3/§3b/§3c/§3d/§4 — 所有实验配置+结果+裁决总账（最权威）
- `ops/research_notes/20260611_*.md` — 路由证伪重判 / 死槽回收 / 原生写入机制 三份调研
- `status/PENDING_TASKS.md` — ROUTE-A/B 立项记录（ROUTE-A 已证伪；ROUTE-B 重定位为死锁破解=EXP-R1）
- `status/TRAINER_ACTIVITY.jsonl` 尾部 — 每轮巡检流水
