# PENDING_TASKS.md — Task Board
## Updated 2026-06-25 00:15 CST

---

## 🌙 [TONIGHT-FIFO-EVAL][PLAN 2026-06-25] 方案B FIFO 消融 4臂 eval 计划（按依赖分类）

### 依赖层级

```
[无需等待，立即可做]
  T1. H20三臂 lm 曲线趋势分析（读远程日志，无GPU）
  T2. 准备所有 eval 启动命令草稿（无GPU）

[依赖 B200 step3000，约1h内]
  T3. rsync B200 ckpt (wzc1→diskA)  ← 需要网络，无GPU
  T4. 本机 H20 跑 B200 chunk1024/b50 W0+W6 eval  ← 需要 本机8×H20

[依赖 H20三臂 step3000，约18.5h后（明天白天）]
  T5. .196 b50/chunk512 W0+W6 eval  ← 训练结束后直接在.196跑（盘A共享）
  T6. .7.53 b25/chunk512 W0+W6 eval  ← 在.7.53本机跑（ckpt在盘B）
  T7. .245.174 b100/chunk512 W0+W6 eval  ← 在.245.174本机跑（ckpt在盘B）
  （T5/T6/T7 三臂并行，各自训练结束后立即在原节点启动）
```

### T1 [PENDING, auto_launch:true, 无GPU] H20三臂 lm曲线趋势分析
- 读 .7.53/.245.174 的训练日志（b25/b100），对比 b25/b50/b100 三臂 lm 收敛曲线
- 判断 buffer_length 对 FIFO lm 的影响趋势
- 无任何依赖，立即可做

### T3 [DONE/SKIP 2026-06-24 23:53] rsync B200 ckpt — 不需要：T4 改在 B200 .53 本机跑（ckpt 已 native 在 wzc1 盘）。step3000 ckpt 也已 scp 回本机 outputs/mem_space_fifo_b50_chunk1024/mem_space_adapter_step003000.pt（7.1G）作备份。
<details><summary>原 rsync 草稿</summary>

```bash
# wzc1→diskA，只同步 step3000 ckpt + adapter_config
sshpass -f configs/password_b200_53.txt scp \
  -o StrictHostKeyChecking=no -o PreferredAuthentications=password -P 36000 \
  root@28.88.184.53:/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/mem_space_fifo_b50_chunk1024/mem_space_adapter_step003000.pt \
  outputs/mem_space_fifo_b50_chunk1024/
# adapter_config 已在本机（训练前同步过），若无：
# sshpass ... scp ... adapter_config.json outputs/mem_space_fifo_b50_chunk1024/
```
</details>

### T4 [RUNNING 2026-06-24 23:53 @B200 .53] B200 chunk1024/b50 W0+W6 eval
- **改在 B200 .53 本机跑**（训练完成后节点全空闲，ckpt 在 wzc1 盘 native 无需 rsync，比等本机 H20 更快；本机 H20 仍占于 c512/b50 训练）。
- ckpt=outputs/mem_space_fifo_b50_chunk1024/mem_space_adapter.pt（step3000 final, lm=3.849）
- W0(swa0)→W6(swa6) 串行两次 scheduler 调用；wrap /tmp/fifo_c1024_eval_w0w6.sh pid 3806156
- log: logs/fifo_b50_c1024_eval_W0.out / _W6.out（B200 .53）；结果 babilong_results/fifo_b50_c1024_step3000_W0|W6
- W0 已起：21 tasks，8GPU 100% healthy。预计 W0+W6 共 ~5h。完成后 scp 结果回本机 score + 填 RUN_REGISTRY。
- 原草稿（本机 H20，留档）：
```bash
# 本机执行（PROJECT_ROOT=diskA）
RUN_PREFIX=fifo_b50_c1024 \
CKPT_FILES="outputs/mem_space_fifo_b50_chunk1024/mem_space_adapter_step003000.pt outputs/mem_space_fifo_b50_chunk1024/mem_space_adapter_step003000.pt" \
CK_NAMES="fifo_b50_c1024_step3000_W0 fifo_b50_c1024_step3000_W6" \
ADAPTER_CONFIG=outputs/mem_space_fifo_b50_chunk1024/adapter_config.json \
CHUNK_SIZE=1024 \
EXTRA_ARGS="--swa_eval_chunks 0 --swa_eval_chunks 6" \
setsid nohup bash scripts/_eval_taskpool_2group.sh >logs/fifo_b50_c1024_eval_sched.out 2>&1 &
```
⚠️ W0/W6 需分开两次调用（EXTRA_ARGS 不能合并），或传两个 ckpt + 两个 swa_eval_chunks 值——需检查脚本支持方式，见草稿。
- 预计时长：~2.5h（42 tasks，2组并行，chunk1024 单task约10min）

### ✅ 三臂训练完成 + eval 已启动（2026-06-25 07:11 heartbeat）
- **b25/b50/b100 chunk512 三臂均 step3000 完成**（07:02-07:07，0 crash/0 non-finite，~622min），`full_model.pt` 落盘。
- T5/T6/T7 **均已在各自原节点启动 W0+W6 BABILong eval**（ckpt=`full_model.pt`，loader strict=False 兼容；注意实际产物是 full_model.pt 而非草稿假设的 mem_space_adapter_final.pt）：
  - **T5 b50 @ 本机 8×H20**（diskA，.venv）：driver /tmp/fifo_b50_c512_eval_w0w6.sh，log logs/fifo_b50_c512_eval_{W0,W6}.out，结果 babilong_results/fifo_b50_c512_final_{W0,W6}。
  - **T6 b25 @ .48.7.53**（diskB，.venv）：driver /tmp/fifo_b25_c512_eval.sh，结果 babilong_results/fifo_b25_c512_final_{W0,W6}。
  - **T7 b100 @ .58.245.174**（diskB，.venv）：driver /tmp/fifo_b100_c512_eval.sh，结果 babilong_results/fifo_b100_c512_final_{W0,W6}。
- **完成后**：score_nested_babilong.py 聚合 → 填 RUN_REGISTRY（b25/b50/b100 × W0/W6）→ 与 B200 c1024/b50 + MemoryLLM baseline 对比 buffer_length×chunk_size 效应。

### T5 [RUNNING 2026-06-25 07:11 @本机8×H20] b50/chunk512 W0+W6 eval
```bash
# .196 节点执行（PROJECT_ROOT=diskA，.venv PYBIN）
sshpass -f configs/password_diskA.txt ssh root@28.59.80.196 \
  "cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory && \
   RUN_PREFIX=fifo_b50_c512 \
   CKPT_FILES='outputs/mem_space_fifo_b50_chunk512/mem_space_adapter_final.pt' \
   CK_NAMES='fifo_b50_c512_final_W0' \
   ADAPTER_CONFIG=outputs/mem_space_fifo_b50_chunk512/adapter_config.json \
   CHUNK_SIZE=512 \
   setsid nohup bash scripts/_eval_taskpool_2group.sh >logs/fifo_b50_c512_eval_sched.out 2>&1 &"
# W6 另一次调用加 EXTRA_ARGS="--swa_eval_chunks 6"
```

### T6 [RUNNING 2026-06-25 07:13 @.48.7.53] b25/chunk512 W0+W6 eval
```bash
# .7.53 节点执行（PROJECT_ROOT=diskB，.venv PYBIN）
# ckpt: /apdcephfs_zwfy6/share_304376610/.../outputs/mem_space_fifo_b25_chunk512/
```

### T7 [RUNNING 2026-06-25 07:15 @.58.245.174] b100/chunk512 W0+W6 eval
```bash
# .245.174 节点执行（PROJECT_ROOT=diskB，.venv PYBIN）
# ckpt: /apdcephfs_zwfy6/share_304376610/.../outputs/mem_space_fifo_b100_chunk512/
```

### 并行策略总结
- **现在**：T1立即做（无需GPU）+ T2 准备命令草稿
- **~1h后**：T3 rsync → T4 本机启动 eval（本机全空闲等这个任务）
- **~明天白天**：T5/T6/T7 三臂各自训练结束后，**在原节点立即启动**（不需要 rsync，ckpt 在本地）
- **eval 完成后**：score_nested_babilong.py 聚合 → 填 RUN_REGISTRY → 与 MemoryLLM baseline (qa5 32k=34) 比较

---

## 🧪 [N16-TOP16][RUNNING 2026-06-23 12:08 @28.58.245.174] 16-slot / top16 对照 — auto_launch:true

- **问题**：标准 nctx63 是 128 slots 中选 top16；用户提出比较“一共只有 16 个 slot”的效果。
- **配置**：`scripts/launch_distill_pg19_nctx63_N16_top16.sh`，`--num_slots 16 --top_k 16`，其余对齐 nctx63 SOTA，PG19 nctx63 cache，total_steps=500/save250。
- **运行**：`28.58.245.174`（盘B H20）已启动，run=`distill_pg19_nctx63_N16_top16`，log=`logs/distill_pg19_nctx63_N16_top16.log`。
- **判据**：step250/500 同口径 BABILong W0；若 N16 接近/超过 N128-top16，说明大 bank 的 selector/稀疏选择不是优势；若明显差，说明 128 bank 容量仍必要。

## 📏 [LONGEVAL-RULER-COMPARE][PLANNING] MoM ckpt vs Landmark baseline — auto_launch:true

- **目标**：测之前关键 ckpt 的 LongEval 与 RULER，并和 Landmark baseline 比较。
- **候选 ckpt**：nctx63 SOTA step250/500、lr5e5 step250/500、train-time router/recency step250/500、必要时 N16 step250/500。
- **已知脚本**：`scripts/eval_longeval_mem_space.py`、`scripts/eval_longeval_landmark.py`、`scripts/eval_ruler_mem_space.py`、`scripts/eval_ruler_landmark.py`、`scripts/_ruler_eval_2group.sh`。
- **下一步**：等当前 router/recency/N16 训练与 eval 节点释放后，优先跑 nctx63 SOTA vs Landmark 的 LongEval/RULER 小网格，再扩到 lr5e5/router/recency。

## 🌙 [TONIGHT-LONGTRAIN][RUNNING/PLANNING] 重点：让训练“长训不掉点” — auto_launch:true

- **核心原则**：每个机制先分清 train-time / eval-time / both。若模型训练时没见过对应路径，eval-only 切换只能算 OOD diagnostic，不能作为机制裁决。
- **已在跑**：
  - local: `distill_pg19_nctx63_lr5e5_s1234`（lr 1e-4→5e-5，seed1234，total_steps=500，save250）
  - .196: `distill_pg19_nctx63_lr5e5`（seed42，同配置）
  - .53: `slot_kv_cache_pg19_chunk512_nctx63_recency`（train-time recency slot-kv）
- **今晚优先假说**：
  1. **优化轨迹过冲**：step250 最好、step500/1000 掉点可能是 lr 过高或后期梯度噪声；lr5e5 双 seed 是第一判据。
  2. **训练/评测路径不匹配**：slot-kv eval-only selector 已证不能裁；必须 train-time recency/all 后再同 mode eval。
  3. **后期不可学 token 死磕**：若 lr5e5 仍掉，下一步考虑 distill loss schedule（后期降低 logits KL/保留 hidden 或只训高置信且同 train path）。
  4. **保长程能力的 regularization**：候选包括 replay/anchor loss（固定少量 step250-like long-context batches）、early-best EMA/SWA 权重平均、step250→500 小 lr continuation 对比。
- **下一步自动动作**：任一 8GPU 节点释放后启动 `scripts/launch_nctx63_slotkv_all.sh`（train-time all-slot arm）；lr5e5 step250/500 ckpt 出后立即同口径 BABILong eval，判断是否真正“不掉点”。

---

### [OVERLAY-COEF05][DONE 2026-06-17 06:00 — seed确认通过] mass coef0.5 + 蒸馏叠加 — ★假说反转→长程最佳（2026-06-16）
- **★假说反转（2026-06-16 08:02 完整W0曲线）**：coef0.5+蒸馏叠加，弱mass+蒸馏长程轻微协同最优（超纯蒸馏11/8、mass coef2 12/7、coef2叠加6/5）。
- **★seed确认通过（2026-06-17 06:00）**：seed1234 final W0 BABILong eval已完成（diskB .249本盘）：
  - qa5 0k-32k = **80/43/48/25/15/11/9**（seed42 = 70/49/44/25/14/13/9）
  - 两seed长程一致：16k=11~13、32k=9 均稳定，远超 coef2(12/7)；qa1=94/35/29/16/7/7/6。
  - **裁决：坐实"弱mass(coef0.5)+蒸馏 = 长程稳定最优组合"，非单seed运气。** 16k 有 11~13 的轻微seed方差（噪声级），32k=9两seed完全一致。
- **下一步**：已入阶段报告候选；结果待落 RUN_REGISTRY（与 32k 双负结果一并写）。

## 🚨 [BASE-MIX0][RUNNING @.249 13:18] mix=0 干净 P11 baseline — 架构实验公平对照前置（2026-06-13）

> **13:18 启动**：`scripts/launch_BASE_mix0_N128.sh` @ H20-.249（diskB，.venv PYBIN），RUN=BASE_mix0_N128，pid=1197243。= expL1ERASE 配置去掉三个架构 flag（无 delta_erase / 无 independent_slot_key / 无 use_l2），唯一区别即 baseline。step5 健康 lm=3.21 babi=0 满载 80GB/79%。total 1000 步，eval_interval=0，ckpt 落 `outputs/BASE_mix0_N128/`（step500+final）。**跑完起同口径 task-pool BABILong eval**。
> **判据用途**：3 架构实验（expL1KEY/expL1ERASE/expL2ON，均 mix=0）长程 qa5 要超的是**这个 mix=0 baseline**，不是旧的 mix=0.15 P11(48/45/44)。
> **下一步**：BASE_mix0 + 3 架构实验都跑到 final → 同口径 eval → 横向对照写 RUN_REGISTRY。

---

## 🎯 ROUTING / CAPACITY 方向（2026-06-10 用户立项）— 根因：usage_cov~0.25，128 槽只用 ~32 个

> **背景（gp-44 根因报告 + gp-39 32k 样本诊断 + 用户总结 2026-06-10）**：长上下文退化的核心瓶颈是 **selector 路由集中**——usage_cov 仅 ~0.25，128 槽实际只活跃 ~32 个，富者愈富（死槽 key 停初值竞争不过）。叠加 delta-rule EMA 写到不动点（new≈current → slot_delta→0，门 g_in≈0.5 没关但写不动）+ 读侧晚期坍缩到单槽（retrieved_norm 4.8→0.44，top1_sim→0.99）。
> **结论**：先修路由均衡（用满 128 槽 = 把"饱和长度"后推 ~4×，零额外参数/显存），再谈遗忘/容量。**加 num_slots 大概率没用（128 都没用满，只多死槽）。**

### [ROUTE-A][~CLOSING — 全旋钮证伪] 路由均衡 sweep — 目标 usage_cov 0.25 → 接近 1.0 — auto_launch:true
> **2026-06-13 06:56 收尾裁决**：四类旋钮全臂跑完且大多评分完毕，**无一在长程(8k-32k)超 P11 base step500(qa5≈48/45/44)**：
> - `loss_free` arm1(0.01): usage_cov 0.25→0.88 达标但 qa5 8k-32k=48/45/44 仅持平 base，未超 → 坐实「usage_cov↑≠长程↑」。
> - `entropy_aux` arm2: 比 arm1 更差。
> - `selector_temperature` {20,40,80}: arm3(temp20) qa5 8k/16k/32k=**10/7/5** 长程崩 REJECTED；arm4(temp80) 38/40/28 REJECTED；temp40=arm1-2 底座。三档全证伪。
> - `load_balance_weight=0.01` 三 seed{42,1234,2026}: seed2026 长程崩(14/14/10)+step500 0k=8 坏，seed42/1234 eval 收尾中（.196/.76），方差噪声，无稳超。
> **裁决：路由均衡旋钮（loss_free/entropy/temp/lbw）全证伪——把 usage_cov 推到 ~0.9 并不能改善长程检索保持。归入写入/读出侧全证伪谱系。下一方向待主会话决策（见 needs_code alert）。** 剩余未扫的 loss_free{0.005,0.02} 不再跑（已证伪方向铺 noise-curve 无意义）。
- **现成抓手（全是 hyperparameter，无需改代码，授权内可自主跑）**：
  - `--loss_free_update_rate`：P11=0.001 可能太弱 → 扫 {0.001, 0.005, 0.01, 0.02}（DeepSeek loss-free-balance，调 router bias 不污染主 loss）。
  - `--entropy_aux_weight`：当前=0 → 扫 {0, 0.001, 0.01} 小权重开。
  - `--load_balance_weight`：当前=0 → 可选小权重 {0, 0.01}（Switch Transformer aux）。
  - `--selector_temperature`：P11=40 → 可对照 {20, 40, 80}（温度影响 softmax 锐度/路由集中度）。
- **底座**：P11 chunk512 delta-rule+normreadout，**只改上述路由旋钮，单变量**，step500 即可读 usage_cov 趋势（不必跑满 5000）。
- **判据**：(1) 训练 diag 的 usage_cov / uniq_sel_slots 是否抬升；(2) step500 同口径 BABILong 长程 cell（8k-32k）是否改善。usage_cov↑ 且长程↑ = 路由修复有效。
- **优先级最高**：ROI 最高、零额外显存。空闲节点优先消化此 sweep。
- 关联在跑：ladder top_k 实验（容量旋钮）、L1-only（L3贡献）、D6三臂（读机制）——这些结果会进一步收窄路由修复的具体形态，可并行。

### [ROUTE-B][PENDING, 需设计讨论 auto_launch:false] 周期性 reset 最不活跃 slots（主动遗忘机制）
- **思路（用户 2026-06-10）**：路由修好后，每隔一段时间直接 reset 最不活跃的一部分 memory slots，给饱和记忆腾位置（比 dual-gate 被动遗忘更直接）。
- **待定设计点（做前讨论）**：(1) "不活跃"定义（累计选中次数 / 最近 N chunk 频率 / EMA 写入幅度）；(2) reset 成什么（清零 / strided_token 重新初始化为可写空槽）；(3) 频率 + reset 数量；(4) **风险**：不活跃 ≠ 无用，可能误删存着早期关键事实的槽 → 伤 NIAH，reset 策略需和"是否还会被读"挂钩。
- **依赖**：必须先做完 ROUTE-A（路由修好、128 槽用满）+ 对"容量-长度饱和拐点"有量化，否则瞎调。与"L1 精确 retrieval 应覆盖多远"这个根本权衡直接相关（reset 旧槽 = 拿"记多远"换"记多清楚"）。

---

## 📋 EVAL QUEUE（2026-06-09 建立）— 空闲卡按序消化，每条跑完落 RUN_REGISTRY + 用 hb_emit_alert 报 train_done

> ★ **eval 调度方式（2026-06-13 用户指定,权威）**：用 `scripts/_eval_taskpool_2group.sh` —— 8 GPU 分 2 组(0-3/4-7),每个 (ckpt,task,length) 任务在一组 4 卡各跑 25 样本(num_shards=4),21+ 任务进共享 pool 哪组空就 flock 原子 pop append。详见 CODEBUDDY.md「标准 eval 方式」。**旧的 per-GPU LPT 静态调度器(`_expR1c*_eval_sched.sh`)已弃用**(长档 shard 致空转)。
> 规则：有空闲节点 → 用 task-pool 调度器跑下一条未完成 eval。
> 统一口径：`scripts/run_babilong_mem_space.py`，qa1/qa2/qa5 × 0k-32k，n=100，chunk512 bf16 sdpa；评分 `scripts/score_nested_babilong.py`。
> ⚠️ 已知坑：远程节点需 woa proxy + HF_HUB_OFFLINE=1 + HF_DATASETS_OFFLINE=1（否则 BABILong dataset HEAD 挂）；短长度(0k/1k)有跨进程 bf16 非确定性，W0/W1/W2 同节点同批跑、重点看 4k-32k；跨节点脚本改对 PROJECT_ROOT（盘A=303098609 / 盘B+wzc1=304376610）+ diskB PYBIN 用 .venv。

### [EVAL-1][RUNNING] P11 step500 (SOTA峰值) × SWA W0/W1/W2 — gp-29 在 B200 收尾中
- ckpt `outputs/mem_space_p11_chunk512_deltarule_normreadout/mem_space_adapter_step000500.pt`，`--swa_eval_chunks {0,1,2}`。
- 目的：SOTA 峰值 ckpt 配 cross-chunk SWA 的天花板，对照 step5000+SWA(W2 qa5=58/29/68/62/42/39/39)。
- 状态：B200 共存运行中（各 ~5-7/21），gp-29 盯出齐评分。

### [EVAL-2][DONE 2026-06-10] ★ step5000 vs step500 的 LongEval 对照 — 假说被反驳
- **结果（B200 GPU3/4，6 LongBench QA 任务，F1，chunk512，no_chat_template，hotpotqa n=200 其余 n=100 同 index 可比）**：step500 AVG=8.87 vs step5000 AVG=6.06。step5000 在**全部 6 任务一致更差**，含三个全局语义任务（narrativeqa 2.07 vs 5.72、qasper 3.89 vs 4.85、multifieldqa 11.25 vs 16.53）——恰是假说预期 step5000 应追平/超过处，结果反而退化最狠。
- **★裁决：假说 REFUTED。过训是单调退化（L1 整体被污染），不是「检索→语义压缩」能力迁移。** LongBench 证明语义总结能力也退化了。step500 是 NIAH 检索 + 全局语义双口径统一最佳，早停是正确交付。详见 RUN_REGISTRY §3b。输出 `longbench_results/p11_step500` / `p11_step5000`（B200 wzc1）。

### [EVAL-2-orig][SUPERSEDED] ★ step5000 vs step500 的 LongEval 对照 — auto_launch:true
- **动机（用户 2026-06-09 insight）**：发现一根因假说 = 单层 L1 slot 训久了转去承担「预训练式高级语义压缩」，把 NIAH 精确检索能力挤掉。**验证**：若 step5000 在 LongEval（需全局语义总结）上**不比 step500 差、甚至更好**，就坐实「L1 没变差、只是能力从检索挪到语义压缩」——发现一从单向假说变双向证据。
- 跑：P11 chunk512 的 step500 ckpt 与 step5000(final) ckpt，各跑 LongEval（用 `scripts/launch_longbench_eval.sh` 或现有 LongEval/LongBench harness，确认口径一致）。
- 同口径对照输出：两个 ckpt 的 LongEval 分数 + 已有的 BABILong qa5（step500=82/86/83/64/50/46/41 vs step5000=54/62/51/30/28/22/31）并排。
- 交付：结论写 RUN_REGISTRY + 回报 main（决定是否进 PPT 发现一作为双向证据）。

### [EVAL-3][PENDING] D1 slot16384 final BABILong — 依赖 D1 训练完(~B200 10h)
- D1 run `outputs/d1_slotdim16384` 跑完 5000 步后，step500/最佳 ckpt 同口径 BABILong，对照 P11 chunk512 baseline（slot4096）。注意 D1 是 (slot_dim16384 + lowrank_gate) 双变量，解读需谨慎。
- 建议补：lowrank_gate@slot4096 对照（gp-20 指出的纯净 slot_dim 消融缺口）——可另起一条 EVAL 或 train。

### [EVAL-4][PENDING] D6 xattn 消融三臂 BABILong — 依赖 D6 训练完(.249)
- 臂A=P11基线(复用)、臂B=xattn OFF(`outputs/d6_xattn_off`)、臂C=xattn ON+sink OFF(`outputs/d6_nullsink_off`)。三臂 step500+final 同口径 BABILong。
- 判据：A vs C 隔离 null-sink 贡献；B vs C 隔离「独立 own-softmax 读机制」整体贡献。

### [EVAL-5][PENDING] D2b 训练侧SWA 双口径 BABILong — 依赖 D2b 训练完(.196)
- D2b run `outputs/d2b_swa_train_w2` ckpt（重点 step1500-3000，避开过训）跑 **双口径**：W0 标准单chunk eval + W2 (`--swa_eval_chunks 2`) eval。
- 判据：对照 D2a 纯-eval-SWA 增益，看「训练时也见过 SWA」是否消除 train/eval mismatch / 进一步提升长程。

---

## [RUNNING 2026-06-08 08:22] F2 prep-3 — real wiki long-doc re-tokenize build（CPU，data-validity gate 已通过）
- **dry_run 裁决（08:22）**：pes2o=DEAD（扫 3.86M docs，0 docs ≥8192 tok，全短摘要）；wiki=唯一可用源（≥2048-tok docs token p90/p99/max=7530/16699/61969；4.7% ≥10k tok、0.4% ≥20k tok）。filter `dolmino_per_doc` 路线 confirmed DEAD（4096 硬截断）。
- **已起真实 build**（pid 615605，CPU 16-proc，log `logs/f2_build_wiki_min4k.log`）：`build_dolmino_longdoc_raw_retokenize.py --raw_glob wiki/*.json.gz --min_tokens 4096 --out_path MemLong/data/processed/dolmino_longdoc_wiki_min4k`。dolmino_per_doc schema 兼容，`--per_doc_data` 可直读。
- **TODO(next HB)**：build 完（检查 out_path DatasetDict + n_docs/长度分布）→ 用 F1 当前最佳（P11 delta-rule + chunk512）+ 此长文子集 → .196/.249 起 F2 long-doc 8-GPU train（eval_interval=0，per_doc_data 指向新子集）→ 落 RUN_REGISTRY。auto_launch: true（F2 既定方向，data 一就绪即起）。
- ⚠️ .196/.249 现 idle 但**合法被 data-build 阻塞**（F2 train 无 confirmed long-doc data 不能起），非「idle-with-runnable-step」。

## [RUNNING 2026-06-08 06:12] F1 v3 top_k-ladder 对照臂（本机 8×H20 空闲 → 自主起，coder general-purpose-2 写脚本）
- **背景**：本机/.196/.249 三节点空闲（FINAL chunk1024 eval + l3recon eval 都收工）；canonical F1 v3（固定 top_k=16）在 .76 跑。研究 note（high-conf）建议测 top_k 随 chunk 阶梯（唯一 warm-start 安全的 slot 容量旋钮）。
- **派 coder（reasoning）写 `scripts/launch_progressive_chunk_local_v3_topk_ladder.sh`**：盘A 本机版 v3 链，单变量改 top_k schedule（c256→16 / c512→24 / c1024→32），num_slots/selector_dim 保持 128 不变（warm-start 形状匹配），其余逐字对齐 canonical v3。独立 output_dir/log/master_port，eval_interval=0。
- **判据**：每 stage step500 离线 BABILong 对照 canonical v3（固定 top_k16）+ v1 stable 链，验证「大 chunk 增 top_k 是否提升长上下文检索」。
- TODO(next HB)：收 coder 报告（脚本 + bash -n + commit + 启动命令）→ 本机 8×H20 仍空闲 → 立即启动 → 落 RUN_REGISTRY。属 adopted F1 v3 base 的 ablation 延伸，可自主起。

## [DONE 2026-06-08 06:08] F2 prep — long-doc 子集 dry_run → ★关键发现：现有 per-doc 数据集 4096 硬截断，filter 路线死路
- coder general-purpose-1 已写 `scripts/build_dolmino_longdoc_subset.py`（commit 待查）。
- **06:05 main dry_run 结果**：train/val 全部文档 max=4096 tok（p99=4096，~6.8% 文档卡在 4096）。chunk512 ≥10 chunks 的文档 = **0%**，chunk1024 ≥10 = 0%。**filter 现有 dolmino_per_doc 永远拿不到「单样本几十~上百 chunk」**——最多 16(chunk256)/8(chunk512)/4(chunk1024) chunk。
- **根因**：`dolmino_per_doc` 是从 packed `dolmino_0.5B_1024`（max_length packing）按 EOS 还原出来的，原始 packing 上限把文档截在 4096。要拿真正长文档必须 **重新 tokenize 原始 raw json.gz**（不截断）。
- **原始长文档源已就位**：`MemLong/data/raw/dolmino_pes2o_wiki/raw/data/{pes2o,wiki}/*.json.gz`（pes2o=学术论文，天然长文）。
- → 转 [RUNNING] 重 tokenize 任务（见下）。

## [DONE 2026-06-08 08:22] F2 prep-2 — raw re-tokenize 脚本 + dry_run 长度裁决（→ 转 prep-3 real build，见顶部）
- 脚本 `build_dolmino_longdoc_raw_retokenize.py` 就位且 dolmino_per_doc schema 兼容。dry_run 裁决：pes2o DEAD、wiki 唯一可用源（详见顶部 prep-3）。→ real wiki build 已起。
- **派发 coder（reasoning）写 `scripts/build_dolmino_longdoc_raw_retokenize.py`**：直读 `MemLong/data/raw/dolmino_pes2o_wiki/raw/data/{pes2o,wiki}/*.json.gz`，用 Llama-3 tokenizer **不截断**整篇 tokenize（加 BOS/EOS），保留 ≥ N tok 的长文档，输出 HF DatasetDict（单列 `input_ids`，schema 与 `dolmino_per_doc` 一致，`--per_doc_data` 可直读）到 `MemLong/data/processed/dolmino_longdoc`。先 `--dry_run --max_files 2` 打印真实长度分布（确认 pes2o 能产出 ≥20k tok 长文）。
- **动机**：plan [F2] 要单样本几十~上百 chunk 压力测 memory 多 chunk 写入→保持→读回；现有数据 4096 截断做不到，必须重 tokenize。纯 CPU prep，与 F1 v3 ladder（.76）无冲突。
- TODO(next HB)：收 coder 报告（脚本 + dry_run 长度分布 + commit）→ 分布合理（pes2o 出 ≥20k tok 文档）→ 实跑生成子集 → 待 F1 v3 ladder 完成用 F1 最优配置起 F2 长文训练。

## [DONE 2026-06-08 05:48] l3_recon CONVERGED (step5000) eval — 确认 REJECTED 不翻案
- **w0.3@.196 converged eval 成功**（diskA 有外网）：qa5 step5000=50/59/45/20/19（16k/32k cell 未全补），qa1=80/27/43/15/3/1/2 → 与 step500 同向，**确认收敛点仍一致劣于无-aux baseline，REJECTED 裁决成立**。已锁进 RUN_REGISTRY §3。
- **w1.0@.249 converged eval silent-fail（0 CSV）**：.249=diskB 无直连外网，BABILong dataset 下载失败 → 0 样本无 CSV（已知 proxy 问题 `reference_h800_babilong_proxy.md`）。**不重跑**——sweep 已终裁 REJECTED，converged 仅确认用，w0.3 收敛点已足够确认，无需补 w1.0。
- **chunk1024 FINAL（step5000）eval 完成**：qa5=29/68/29/15/7/4（32k 收尾），qa1=56/56/15/15/7/5/0。**确认 chunk1024 的 1k 后断崖满训后依然持续**（对照 chunk512 qa5=82/86/83/64/50/35），渐进 warm-start（F1 v1）仍是修断崖正解。已锁进 RUN_REGISTRY §3。

---

## [DONE 2026-06-07 22:05] l3_recon_token_weight sweep — w1.0 step500 BABILong eval 评完 + 裁决 ❌
- 21/21 CSV（qa1/qa2/qa5×7 len，n=100）全完成，已 canonical 评分（`scripts/score_nested_babilong.py`，diskB .76）。
- **结果（灾难）**：qa5 0k-32k = **67/22/16/8/3/1/0**；qa1=77/4/6/8/3/2/1；qa2=43/4/5/3/1/2/3。
- **裁决：L3 token-recon aux weight=1.0 灾难性破坏长程寻址。** 对照无-aux P11 chunk512 baseline（qa5=82/86/83/64/50/35）→ 仅 0k 部分存活，≥1k 全面塌方。真实实验结果（CSV 满 n=100 非 silent-fail）。已锁进 RUN_REGISTRY §3「l3_recon_token_weight sweep」。
- 含义：强 token-level recon aux 与 routing/检索目标冲突；待 w0.3 弱权重确认是否「弱即无害 vs 仍劣于无 aux」。两 train run（.196 w0.3 / .249 w1.0）继续跑满 5000 仅为 lm/recon 曲线，BABILong 已基本判定 token-recon aux 不优于 baseline。

## [DONE 2026-06-07 23:15] l3_recon_token_weight sweep — w0.3 step500 BABILong eval 评完 + sweep 终裁 ❌
- **23:11 .76 eval 节点全空闲（8 GPU 0 MiB）→ w0.3 step500 eval 完成（7/7 长度 × qa1/qa2/qa5 CSV 齐，n=100）。** canonical 评分（`scripts/score_nested_babilong.py`，.76 diskB）。
- **w0.3 结果**：qa5 0k-32k=**54/61/56/34/25/21/10**；qa1=78/26/42/31/22/21/14；qa2=33/3/15/14/14/9/11。
- **裁决：弱权重 token-recon aux 仍一致劣于无-aux P11 baseline（qa5=82/86/83/64/50/35/41）——全长度无一更优。** 破坏比 w1.0（67/22/16/8/3/1/0）温和但方向相同。
- **★sweep 终裁：L3 token-level recon aux 在 w0.3 + w1.0 均 REJECTED。token-recon 与 routing/检索目标冲突，弱权重也只是「破坏更小」非「有益」。最佳仍是 P11 无-aux baseline。** 已锁进 RUN_REGISTRY §3。两 train run（.196 w0.3 / .249 w1.0）继续到 5000 仅留 lm/recon 曲线。

---

## [DEAD 2026-06-07 17:25] H800 16卡 lease 又被回收 — hung-fix subagent 失败（节点消失）
- 16:40 派的 general-purpose-1 修 H800 hung 没能完成：~17:20 两节点 SSH 全拒（port 36000 refused、port 22 password denied），跟之前所有 H800 IP 一样被回收。
- stage1/stage2 ckpt（step600+final）在 jn2 共享 FS 上，现已不可访问；stage3/4 从未存出。
- **所有 H800 IP（.247/.130.90 及历史全部）现已死，别再试**。H800 stable-ladder 工作挂起，等新 lease 重新分配。mem_space ablation 全部转到 4 个 H20 节点继续。

## [RUNNING 2026-06-07 17:22] chunk 阶梯 step500 judge evals（auto_launch 自主起，on diskB .76 free GPUs）
- diskB .76 GPU6/7 在跑旧 eval、GPU0-5 空闲 → 自主起两个 step500 BABILong eval：
  - **chunk256** deltarule_normreadout step500：GPU0-2，driver pid 194650（17:22）。已到 qa1/0k 17%。
  - **chunk1024** deltarule_normreadout step500：GPU3-5，driver pid 195766（17:24）。模型加载中。
- 同口径 qa1/qa2/qa5 × 0k-32k，n=100，babilong.metrics。对照 P11 chunk512 step500 baseline（qa5 0k-8k=82/86/83/64/50）。woa proxy + HF_HOME 已 export，worker log 无 network err。
- 完成判读：补全 P11 deltarule_normreadout 的 chunk 阶梯三点（256/512/1024）横向对照，写入 RUN_REGISTRY.md。

---

## [用户决策 2026-06-07 10:25]
- **D6（null-sink vs xattn 解耦）= 取消**。用户："null sink 和 xattn 的解耦可以暂时先不做，毕竟现在效果很好"。不改 selector.py。从 roadmap 移除（不再 BLOCKED-pending-decision）。
- **下一轮阶梯式训练 = 等远程两个H20(.76/.249)评测跑完后起**。但用户要求先 research：(1) 小 chunk size 训练波动大 → 找"更合适的小-chunk 训练方式"；(2) 阶梯/小chunk 对 slot 容量的要求可能不同 → 谨慎探讨 slot 容量 vs chunk size。调研中（general-purpose-4，写 status/research_notes/small_chunk_training_and_slot_capacity_20260607.md）。调研出方案 + 节点空出 → 起改进版阶梯。

---

## [DONE 2026-06-07 13:04] stable progressive-ladder FINAL ckpt BABILong eval（.76 空闲自主起）
- **背景**：diskB(.76) 的 stable progressive chunk 阶梯 08:41 全链路完成（4 stage: 128→256→512→1024, nf=0, stage4 121.5min）。
- ckpt = `outputs/progressive_chunk_diskB_stable/stage4_c1024/mem_space_adapter.pt`（P11 delta_rule+normreadout 渐进训练）。
- **评完（21/21 CSV，eval@chunk1024）**：qa1 0k-8k=86/69/45/41/25；qa2=39/35/32/16/12；qa5=14/23/82/59/39（qa5 0k/1k 低是 chunk1024 短长度已知抖动，2k 起 82/59 强）。
- **★关键裁决：渐进式 chunk 训练 ≫ 单 chunk1024 训练。** 同在 chunk1024 eval 下：qa1 2k ladder=45 vs 单chunk1024=4；qa5 2k ladder=82 vs 单=20；长程 qa5 16k=32/32k=29 vs 单 16k=5/32k=4。**渐进 warm-start（小→大 chunk）彻底修复了单 chunk1024 的 1k 后断崖塌方。** 这是阶梯训练价值的决定性证据。已锁进 MEMORY_PROTOCOL_PLAN。
- driver 已退（GPU6 仅剩 stage1_c128 step400 32k 收尾 cell，非调度器，~分钟级完成）。

## [DONE 2026-06-07 13:02] chunk-ladder step500 BABILong eval 评完 + 裁决
- 两个 step500 eval dir（21/21 CSV）已 babilong.metrics 评分（diskB .76）。qa5 0k-8k：chunk256=78/66/47/28/42，**chunk512(baseline)=82/86/83/64/50 ⭐**，chunk1024=82/43/20/29/16。
- **裁决：chunk512 决定性最佳。chunk256 中长度弱，chunk1024 1k 后断崖（2k=20、16k=5/32k=4，复现 P8 chunk1024 长程塌方形态）。** 已锁进 MEMORY_PROTOCOL_PLAN P11 段。后续臂一律 chunk512 底座。c256/c1024 训练继续到 5000 仅为 lm/压缩曲线。

## [SUPERSEDED 2026-06-07 08:22] chunk-ladder step500 BABILong eval 补全（chunk256 + chunk1024）— RELAUNCHED w/ proxy（评分已在上面 13:02 完成）
- ⚠️ **07:48 首launch 静默失败**：diskB(.249) 无直连外网，BABILong dataset HEAD 请求报 "Network is unreachable"，0 样本评出、无 CSV，driver 仍打印 "all done"（假完成）。根因同 memory `reference_h800_babilong_proxy.md`（diskB 须挂 woa proxy + HF_HOME）。
- **08:22 重启修复**：export http_proxy/https_proxy=hy-proxy.woa.com:3128 + HF_HOME=.../share_304376610/.../.hf_home 后重跑。chunk256 GPUs0-3 (driver pid201775) + chunk1024 GPUs4-7 (driver pid201776)。已确认 worker 加载 766 keys + 经 proxy 触达 HF Hub（不再 Network unreachable），8 卡各 35GB busy。
- qa1/qa2/qa5 × 0k-32k，n=100，commit 同 P11。脚本 `scripts/eval_p11_chunk{256,1024}_deltarule_normreadout_step500.sh`（diskB）。step500 ckpt 两个均在 diskB（chunk256 5:50、chunk1024 6:25）。
- 对照 P11 chunk512 step500 baseline（qa5 0k-8k=82/86/83/64/50）→ 三点齐定 P11 最佳 chunk。
- ETA ~1.3h。完成后 aggregate 三 chunk → 更新 MEMORY_PROTOCOL_PLAN + RUN_REGISTRY。
- 🔧 **TODO(auto_launch:false)**：eval driver 在 worker 全失败时仍打印 "all eval lengths done" + exit 0，掩盖网络失败。应在 run_on_gpu 后校验 CSV 生成 / worker 退出码，否则 driver 退非零。避免再静默假完成。

---

## [DONE 2026-06-07 03:20] 4-arm chunk512 step500 ablation 评分 + 裁决
- 4 臂全训到 5000、step500 ckpt 同口径 BABILong 评完。**P11 (delta-rule + normalized writeback) = 新最佳臂**，qa5 1k-8k=86/83/64/50 超 top_k16 基线（76/77/54/48）。P10(ST-Gumbel 硬路由) 与 topk8 均劣于基线 → REJECTED。结果锁进 RUN_REGISTRY.md §3 + MEMORY_PROTOCOL_PLAN P10/P11。

## [RUNNING 2026-06-07 03:56] 下一臂 arm-1：P11 + chunk1024（ablation 延伸，auto_launch 自主起跑）
- P11(delta-rule+normreadout) 已确立为新基线。本机 8×H20 空闲 2 个 patrol → 按 heartbeat「adopted 底座的 ablation 延伸可自主起」启动 arm-1。
- run `mem_space_p11_chunk1024_deltarule_normreadout`，本机 8×H20，commit 9a9e3d0 配置，单变量 chunk_size 512→1024（chunk = 最大杠杆，§4 观察1）。script `scripts/launch_mem_space_p11_chunk1024_local.sh`（flags 与 chunk512 逐项一致，仅 chunk_size/run/port 差）。total_steps5000 save500 eval0 seed42 bs1×ga4×8=eff32 lr1e-4。pid 4061522 master_port29794。
- health: step5 lm=4.8064 route_aux=3.37 nf=0，8 卡 79-100% util ~81GB/卡，no error。
- judge: step500 ckpt 同口径 BABILong（qa1/qa2/qa5×0k-32k，n=100）对照 P11 chunk512 step500（qa5 0k-8k=82/86/83/64/50）。
- **剩余备选臂（仍 auto_launch: false，等用户/下个空闲节点）**：(2) P13 surprise-gated write（Titans 2501.00663）；(3) P11 + register slots(P9 num_global_slots) 组合。

## [RUNNING 2026-06-07 04:37] arm-2：P11 + chunk256（chunk 阶梯补全，auto_launch 自主起跑）
- .196 在 P11 step500 eval 全部 drain 完后空闲 → 按「adopted 底座 ablation 延伸可自主起」启动 chunk 阶梯第三点。
- run `mem_space_p11_chunk256_deltarule_normreadout`，远程 .196 8×H20，单变量 chunk_size 512→256（脚本 `scripts/launch_mem_space_p11_chunk256_remote196.sh`，flags 与 chunk512 逐项一致仅 chunk_size/run/port 差，master_port29793）。total_steps5000 save500 eval0 seed42 bs1×ga4×8=eff32 lr1e-4。pid 2687516。
- health: step5 lm=4.5015 route_aux=5.10 nf=0，8 卡 84-100% util ~75GB/卡，no error。
- judge: step500 ckpt 同口径 BABILong（qa1/qa2/qa5×0k-32k，n=100）对照 P11 chunk512 step500（qa5 0k-8k=82/86/83/64/50）+ chunk1024（本机跑中）。
- **chunk 阶梯（P11 base）现况**：256(此/.196)·512(adopted baseline DONE)·1024(本机 RUNNING)。三点齐则可定 P11 最佳 chunk。

---

## [DONE] researcher: chunk128 vs chunk256 step1000 退化形态差异根因 (general-purpose-35, 2026-06-05 20:08)
- **现象**：null-sink P8 两个臂 step500 都好，step1000 都崩到 ~0%，但**失败形态不同**（chunk256=连贯续写 haystack，chunk128=token 重复死循环乱码）。
- **根因（confidence high）**：⚠️ **推翻旧前提"TF lm 全程健康~3.3"**——chunk128 的 TF lm loss 在 **step895-1010 飙到 ~8-9（PPL~3000）**，step1000 ckpt 恰好存在这个 loss spike 中段；step490-510=~2.4（谷底），step1490-1510 已回落~4.0。每 500 步存盘节奏不巧把 chunk128 step1000 存在了 spike 顶上。
- **为何 chunk 越小越偏 LM 崩坏**：注入次数=seq_len/chunk_size，chunk128 是 chunk256 的 2×；spike 期过量注入（topk_mass>1.5）在 2× 注入事件上累积 → backbone 彻底塌成功能词死循环。chunk256 同期注入少，只退化成连贯续写。chunk256 跑 5000 步，step1000 lm=3.35（谷底未崩），其 spike 在 1200-1300 / 1750-1950。
- **不是 adapter 永久损坏，也不是单纯 greedy 假象**：是瞬态训练不稳定的快照。rep_penalty/temp 只能减轻不能完全救回。
- **结论**："早 ckpt=最终交付"对 chunk128 成立（用 step500），但原因从"过训练"改写为"快照撞 loss spike"。
- 诊断脚本 `scripts/diag_chunk128_step1000_repgen.py` 已写好未运行（GPU 全忙）。报告已 append RESEARCHER_REPORTS.jsonl。

## [DONE 2026-06-06 02:54] eval chunk512/1024 step500+step1000 (验证 chunk 越大越稳假设)
- **完成**：chunk512/1024 step500 与 step1000 全部 0k-32k 已评完，数字已锁进 MEMORY_PROTOCOL_PLAN.md P8 阶梯表。结论坐实「chunk 越大越稳，step1000 崩=快照撞 loss-spike×注入频率」。最佳臂=chunk512 step500。无遗留 eval。

## [DONE 2026-06-06 05:25] coder: 加 topk_mass + chunk_idx_jaccard routing 诊断 (agent general-purpose-21)
- **完成**：commit `5656cb6` 已落地，新指标 topk_mass / chunk_idx_jaccard 已在 QUERY_DIAG 中输出（chunk128_routeaux eval log 已可见）。纯诊断 no-grad，不改训练数值。后续 launch 自动带上。
- **动机存档**：top_k=16 等权监督下 top1_sim 有数学天花板 ln(16)=2.7726，top1_sim≈1/16 平是预期非 routing 崩。topk_mass 判 mass 是否集中，chunk_idx_jaccard 区分真寻址 vs 退化捷径。

## [DONE 2026-06-06 05:25] E5 route_aux 8B 验证 run (commit 35ea240) + offline BABILong eval gate
- **完成**：E5 train 出 step500 ckpt（outputs/e5_route_aux_remote/，train 后续停在 ~step830，step500 即交付点）。offline BABILong eval（qa5 × 0k-32k，commit 35ea240）已于 2026-06-05 02:20 跑完，CSV 存 babilong_results/perdoc_chunk128_routeaux/*。
- **结论（已存档于 TRAINER_ACTIVITY + RESEARCHER_REPORTS）**：route_aux 是 routing differentiation 的 driver（key_max_cos 0.47→0.58，top1_sim 0.015→0.10+，lm 1.60 vs l3iso 2.63）。但 eval QUERY_DIAG 显示 ≥2k 仍 top1_sim≈0.02-0.03、topk_mass 仅 0.28-0.42、chunk_idx_jaccard 0.33-0.44（退化捷径迹象）—— route_aux 提升了 key 可区分性但未把 retrieval 真正爬起来。
- l3iso_noL3_local 是 E5 的 route_aux-OFF 对照，researcher 已判 KILL（预期 no-L3 collapse，无法回答真问题），不再续跑。

---

## [DONE] toy 诊断矩阵 E1/E2/E4 (2026-06-04 14:00)
- 5 arm 全完成。**E1**：decoupled-read 饿死 selector LM 梯度（ON lm_grad 0.3–4 vs OFF 8–15，~10–50× 衰减）。**E2**：纯 LM loss 无法 bootstrap content addressing（aux_off exact_acc=0）；routing-supervision aux → exact_acc 0.25↑。**E4**：冻结 inject gate 非主因（force-open top1_sim→0.30 但 exact_acc 仍 0）。
- 决定：自动派 coder 实现 route_aux + E5 8B 验证 run。

---

## [PENDING] 修 FSDP checkpoint-save host OOM — auto_launch: false
- fsdp_smoke_remote @2026-06-04 11:56 在首个 checkpoint save 时 SIGKILL -9（FSDP full state_dict gather 8B 模型 → host mem OOM）
- commit 02561b4 "complete FSDP migration" 的存盘路径需改：用 sharded state_dict / get_state_dict API（日志里有 deprecation 提示），或 rank0 流式存盘避免一次性 gather 全量
- 优先级：仅当需要 FSDP 路径时才修；当前 DDP+gradient_checkpointing 在本机 8B 已能跑通 2000 step
- auto_launch: false（涉及存盘逻辑改动，等确认确实需要 FSDP）

---

## [DONE] P2 decoupled-read offline BABILong eval (2026-06-04 13:25)
- 21/21 cells (qa1/qa2/qa5 × 0k-32k)。**FAILS gate**：0k qa1=72/qa2=27/qa5=53，≥2k 全 0.0%。
- 结果已写入 status/BENCHMARK_RESULTS.md。eval 期 top1_sim≈0.05≈uniform → routing collapse 确认。

## [DONE] researcher toy-vs-full routing collapse 报告 (2026-06-04 12:30)
- ops/research_notes/toy_vs_full_routing_collapse_20260604.md。confidence high/very_high。
- 关键：top1_sim 是 red-herring（toy retrieval_exact_acc=0 全程）；decoupled-read 切断 selector LM 梯度（mask_h_to_l1）；LM loss 单独无法 bootstrap content addressing；inject_gate 冻结 α≈0.12。
- 建议先跑单 GPU E1/E2/E4 再决定 8B 修复 → 已于 13:49 在 H20-1 GPU0-4 启动诊断矩阵。

## [DONE] P2 decoupled-read full 8B run (2026-06-04 12:13)
- dolmino_p2_decoupled_local step2000/2000 完成。Routing 仍塌缩 top1_sim≈0.013≈uniform。
- 关键发现：同机制在 toy arm 能学会(0.998)，full 8B 塌缩 → 已派 researcher 分析 scale/data gap。
- checkpoint: outputs/dolmino_p2_decoupled_local/mem_space_adapter.pt，offline eval 进行中。

## [DONE] P1-v3 routing fix 系列、multi_query、chunk_query（早前）
- 结论汇总见 status/gpu_runs.jsonl 与历史 UPDATELOG。所有 P1 routing-pool 变体均塌缩在 1-2% noise floor。

## [DONE 2026-06-07 12:17] eval P11 chunk512 deltarule CONVERGED ckpt (step5000) — on .249
- P11 chunk512 deltarule+normreadout train FINISHED 02:20 (step5000, lm=2.43, non-finite=0); only its step500 ckpt was BABILong-evaluated. Converged ckpt eval **COMPLETE** (21/21 CSVs, "all eval lengths done" 12:17, 1h32m).
- output: `babilong_results/p11_chunk512_deltarule_normreadout_final/` on diskB (raw CSVs target/output/question — needs babilong.metrics scoring to aggregate).
- TODO(next): score converged CSVs w/ babilong.metrics; compare converged-vs-step500 (step500 qa5 0k-8k=82/86/83/64/50); update RUN_REGISTRY + MEMORY_PROTOCOL_PLAN P11 row.

## [RUNNING 2026-06-07 12:25] v2 progressive chunk ladder (per-stage scaled warmup/grad_accum) — LAUNCHED on .249
- **背景**：用户 10:25 决策门——"下一轮阶梯式训练 = 等远程两个H20(.76/.249)评测跑完后起，先 research 小-chunk 训练方式 + slot 容量"。research note `status/research_notes/small_chunk_training_and_slot_capacity_20260607.md`（11:08 完成）+ v2 脚本（commit 5aa2329, 11:21）均就绪。.249 的 converged-c512 eval 12:17 跑完→8 卡全空闲→门已满足，自主起 v2 ladder。
- **v2 vs v1**：per-stage 反比缩放 warmup + grad_accum（c128:warmup800/accum8, c256:500/4, c512:300/2, c1024:200/1），使 warmup-token 与有效梯度-token/step 跨 stage 恒定，压小 chunk 梯度方差（research note 标 [high,可直接采用] 零风险）。其余配方 = v1 = P11 stable（delta-rule writeback + normalize_readout + loss_spike_skip + ST-Gumbel OFF）逐项一致。
- node **.249** 8×H20（自有卡，非 .76），warm-start 链 stage1 c128(scratch)→s2 c256→s3 c512→s4 c1024，各 stage 从上一 stage step000600 adapter init。driver pid 230717，log `logs/progressive_chunk_diskB_v2.driver.log` + 各 stage `logs/progressive_chunk_diskB_v2_stage*.log`。total_steps800/stage save200 chain_step600 eval0 seed42。
- **health**：stage1 c128 8 ranks 全载入权重（15.7→74GB/卡），util 38-100%，无 error/unreachable/nan。代码已从 diskA rsync 到 diskB（v2 脚本确认存在 + delta_rule flag）。
- judge: 对照 v1 stable ladder（.76 已跑完，FINAL ckpt eval 收尾中）+ P11 单 chunk 各点 → 验证 per-stage 缩放是否改善小-chunk 稳定性 / 最终 retrieval。

---

## [PENDING] ★ b25/c512 中间 ckpt 早评（step500/1000/1500/2000/2500）— auto_launch: true (next-free-node)
- 动机：.7.53 b25/c512 step3000 W0 全档破墙(qa5 32k=68 vs MemoryLLM 34)。过训退化铁律：历史 step500 普遍是甜区，step3000 可能已退化。早评中间 ckpt 找峰值。
- ckpts: `outputs/mem_space_fifo_b25_chunk512/full_model_step00{500,1000,1500,2000,2500}.pt` on .7.53 (diskB)
- eval: `_eval_taskpool_2group.sh`，W0+W6，CHUNK_SIZE=512，n=100，21 cells/ckpt × 5 ckpts × 2 modes = 210 tasks。可分批：先 step500/step1000，足够定形态。
- 节点选择：等任一节点 free 即起。本机/.196 不持 b25 ckpt 需 rsync (~23GB/ckpt)；.7.53 自持 ckpt 但目前 W6 在跑；.245.174 共享 diskB 可直接读 .7.53 路径。
- 优先级 P0(决定破墙结果时序稳定性)。auto_launch: true。

## [PENDING] ★ b25/c512 step3000 真实长文档 benchmark（LongBench / LongMemEval / LongEval）— auto_launch: true (next-free-node-after-b25-ckpt-eval)
- 动机：BABILong 破墙不等于真实长文档破墙(pg19 nctx7 案例：BABILong 16k +3 但 LongBench AVG 6.5；对话记忆 mem vs base 差 3.8-7×)。必须验证 b25 c512 不是 BABILong 过拟合。
- benchmark：LongBench (hotpotqa/2wikimqa/musique/narrativeqa/qasper/multifieldqa_en)、LongMemEval (oracle n=500 全6题型)、LongEval (lines retrieval ≥8k)。
- 脚本已有：`scripts/eval_longbench_mem_space.py`、`scripts/eval_longmemeval_mem_space.py`、`scripts/eval_dialogmem_mem_space.py`、`scripts/eval_longeval_mem_space.py`。
- 优先级 P1(决定结果迁移性)。auto_launch: true，但排在 b25 中间 ckpt 早评之后。

## [PENDING] b50/c512 + b100/c512 中间 ckpt + 跨臂 ckpt-curve 对照 — auto_launch: true (eval-after-final-W0)
- 动机：等本机 b50/c512 W0 + .245.174 b100/c512 W0 出炉后，若长档分数(8k-32k)显示 buffer_length 单调影响 → 确认 dilution 剂量曲线；若 b50/b100 也破墙 → buffer_length 不是 load-bearing → H3/H4 候选；若 b50/b100 不破 → b25 是 load-bearing → 探索更小 buffer (b10/b5)。
- 跨臂中间 ckpt(b50/b100 各 5 个早 ckpt) 用于过训退化对照。
- 优先级 P1(决定 H1/H2 假说裁决)。auto_launch: true。
