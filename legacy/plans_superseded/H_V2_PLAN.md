# H-series v2 执行计划书

**创建时间**: 2026-05-11 (最后更新)  
**目标**: 在 BABILong 上获得非零分数，验证 cross-attention memory 架构  
**维护方式**: 该文件现视为常态化 / 持续维护的 plan 文件；heartbeat/main 在处理完状态变化后可以直接更新

## 2026-05-17 当前执行面（heartbeat refresh）

- 当前主执行面已切到 **Phase-1B / BABILong SFT**，不再是早期 H-v2 多节点推进阶段。
- **Local H20** 上的 `P11 8B FSDP` 已在本轮 heartbeat 确认完成：`logs/p11_fsdp_full_20260516_181417.log` 已到 `step 5000/5000`，`outputs/babilong_sft_phase11_fsdp_full/mem_space_adapter.pt` 与 `adapter_config.json` 均已存在，本地 8×H20 现已回到 `0 MiB / 0% util` 空闲状态。
- `P11` 训练阶段的核心未解现象没有消失：末段 QUERY_DIAG 仍显示 `top1_sim_mean≈0.00203–0.00217`，而这次 **P11 8B final eval 已经给出了明确判定**——这种 flat-routing 不是 benign 指标噪声，而是和真实任务退化强相关。
- **Remote 28.59.80.196** 上的 `P11` 7-length final eval 已在本轮 heartbeat 完整收齐：`qa1/qa2/qa5 × {0k,1k,2k,4k,8k,16k,32k}` 全部达到 `100 parsed rows`，结果目录仍是 `outputs/eval_phase1b_p11_final/`，canonical logs 仍是 `logs/eval_p11_final_{qa1,qa2,qa5}_{short,long}_20260517_015625.log`。此前误拉起的重复 worker（`276532`–`276537`，日志时间戳 `015651`）保持已清理状态；canonical workers `274873, 274938, 275004, 275070, 275136, 275202` 现都已退出，远程 8×H20 重新回到 `0 MiB / 0% util`。
- `P11` 的最终 21-cell mean 只有 **26.33**。逐任务为：`qa1 = 64/51/39/28/10/1/1 → 27.71`，`qa2 = 24/21/16/14/4/0/0 → 11.29`，`qa5 = 69/73/64/59/13/0/2 → 40.00`。聚合后 short average (`0k/1k/2k/4k`) 仍有 **43.50**，但 long average (`8k/16k/32k`) 只剩 **3.44**。
- 这个结果直接否定了“`top1_sim≈1/512` 虽 flat，但 8B 也许仍能在任务上 work”的乐观解释。`P11` 不仅远低于 `8B P8 = 59.14`（**-32.81pp**），也低于 BABILong paper 的 `Meta-Llama-3-8B-Instruct` vanilla mean `42.6`（**-16.27pp**，见 `docs/PAPER_BASELINES_20260516.md` 的表格摘录）。因此 **当前这支 8B recipe 不只是没把 memory 学好，而是已经把 vanilla 长上下文能力也拖下去了**。
- 现在最合理的机制解释是：router 在训练和 eval 中都长期停留在接近均匀的 flat regime，memory slots 没学出 usable specialization。于是短长度下模型还能部分依赖 base LM / 近程上下文 / L3 summary 维持一些分数，但一旦到了真正需要跨长上下文检索的 `8k/16k/32k`，读取到的就是近似随机或均值化的 memory，long-context 能力就几乎完全塌掉。
- **Remote 28.59.80.196** 上的 `1B v3 shortfix final eval` 已在上一轮 heartbeat 完整收齐并打分：`qa1/qa2/qa5 × {0k,1k,2k,4k,8k,16k,32k}` 全部达到 `100 parsed rows`，`v3_shortfix_final` mean = **37.48**。它与 `v2_final=37.43` 基本持平（**+0.05pp**），同时远好于 `v4_final=15.14`（**+22.33pp**）；这再次说明“memory 方向整体无效”不是结论，真正的问题是 **当前 P11 / flat-routing recipe 本身**。
- 与此同时，`v4` 的负面对照仍成立：`1B v4 L1+L2+L3 FSDP` 的 7-length BABILong final eval 最终只有 **15.14**，相对 `v2` 整体 **-22.29pp**；尤其长长度退化严重（`8k/16k/32k` 平均 delta `-52.3/-37.0/-28.7pp`），说明当前这版 **L2-enabled recipe 明显差于 v2 (L1+L3 only)**。
- 这轮 heartbeat 再次确认：**raw line count 不能再作为 eval 完成度信号**。部分 CSV 含嵌入换行，导致物理行数大于真实样本数；后续必须用 `csv.DictReader` 的 parsed row count 判断是否完成。
- 用空闲远程 H20 跑完的 **`selector_temperature=20.0`、500-step、复用 v4 路径** 的短诊断消融，已在约 `23:33 CST` 干净完成。最关键的新信号是：`top1_sim_mean` 曾达到 **`0.034180`**（step 443），尾段也保留在 **`0.007629 / 0.006226`**，显著高于旧的 `~1/512≈0.002` floor。这使“selector softmax 太平”从怀疑变成了更像**真实可控的机制杠杆**。
- 当前关于 `top1_sim_mean≈1/512≈0.002` 的证据链已扩展为：**窄范围 researcher 诊断 + 窄范围 coder audit + L1-only 窄范围分析 + temp20 机制试验 + 这次 P11 8B final eval negative comparison point**。综合解读现在更明确：
  1. 旧配方里的 router 很可能确实长期处于 flat regime；
  2. 这不等于“memory 方向整体无效”，因为 P8、v2、以及 v3 shortfix 都已经提供了正收益 comparison point；
  3. temp20 说明 selector sharpness 至少能显著抬升 top1 分布；
  4. v4 的额外崩塌很可能还叠加了 **L2 实现/训练问题**，而不只是 selector 太平。
- 关于 **L2 回退**，researcher 的高置信结论仍成立：最可疑的两处是 `src/memory/mem_space/patch.py:256` 的 `torch.no_grad()` L2 hook，以及 `scripts/train_mem_space_babilong.py` 训练 sample reset 未清理 L2 状态。最便宜的后续验证仍是：**eval-only 禁用 L2 跑 v4 final ckpt**，看分数是否回升接近 v2。
- `P11` 的 cheapest follow-up `step4500` checkpoint-level eval 已在本轮 heartbeat 完整收齐：`qa1/qa2/qa5 × {0k,1k,2k,4k,8k,16k,32k}` 全部达到 `100 parsed rows`，结果目录为 `outputs/eval_phase1b_p11_step4500_20260517_040951/`，日志为 `logs/eval_p11_step4500_{qa1,qa2,qa5}_{short,long}_20260517_040951.log`。远程 tmux 会话仍在，但 `step004500` worker 已全部退出，远程 8×H20 重新回到 `0 MiB / 0% util`。
- `step4500` 的最终分数为 **27.43**，其中 `qa1 = 63/59/39/29/6/1/0 → 28.14`，`qa2 = 20/24/25/15/4/0/0 → 12.57`，`qa5 = 73/77/63/64/13/0/1 → 41.57`；聚合后 short average (`0k/1k/2k/4k`) = **45.92**，long average (`8k/16k/32k`) = **2.78**。
- 这与 `P11 final=26.33` 的差距只有 **+1.10pp**，而且 long average 反而更差（`2.78 < 3.44`）；同时 `step4500` eval 全程 `QUERY_DIAG top1_sim_mean≈0.00204–0.00213`，与 final 训练末段的 `0.00203–0.00217` 同属一个 flat-routing regime。因此“只是 late-training collapse”的解释已明显变弱，更像 **当前 8B recipe 到 `step4500` 时就已经整体失效**。
- researcher `general-purpose-2` 的 postmortem 已返回，结论是 **whole-recipe failure，不是 late collapse**，且置信度为 **very high (95%+)**。最高信息增益 / 最低成本的下一步是：**8B `selector_temperature=20`、500-step 短消融**。
- 这一步的 detached tmux 版 `temp20` 短消融已干净完成：`logs/p11_temp20_500_20260517_063303.log:282-284` 明确给出 `step 500/500`、final `mem_space_adapter.pt` 与 `Training complete: steps=500 babilong=411 pg19=89 non-finite=0`。
- 更关键的是，temp20 的后半程 QUERY_DIAG 没有塌回 flat floor：`top1_sim_mean` 在 `step 413/443/464/484` 分别为 **`0.008911 / 0.013489 / 0.006653 / 0.012512`**，说明被拉高的 routing sharpness 一直保持到训练结束。
- `temp20 final BABILong eval` 现已全部收齐并打分。`outputs/eval_p11_temp20_final_20260517_073341/` 下 21 个 cell 全部达到 `100 parsed rows`；最终 `qa1=36.86`、`qa2=12.86`、`qa5=56.00`，overall **35.24**，short avg **45.42**，long avg **21.67**。
- 这个结果证明 `selector_temperature=20` 确实是**真实有效的 8B 机制杠杆**：相对 old flat-routing 对照，temp20 final 比 `step500=33.81` 高 **+1.43pp**，比 `step4500=27.43` 高 **+7.81pp**，比 `P11 final=26.33` 高 **+8.91pp**。但它仍明显低于 `P8=59.14`，所以还不能把“只调 temp20”视为终局解。
- 与此同时，`28.59.80.196` 上的 `step000500` checkpoint eval 也已全部完成并打分：overall **33.81**，short avg **45.92**，long avg **17.67**；远程当前无 eval worker，仅剩 stale tmux `p11_step4500_eval_20260517_040951`。
- 这轮 `p11_fsdp_500step_validate` 已在本地 `8×H20` 上干净完成：`logs/p11_fsdp_500step_validate_20260517_0851.log:280-282` 给出 `step 500/500`、final `mem_space_adapter.pt` 与 `Training complete: steps=500 babilong=411 pg19=89 non-finite=0`。
- validate 输出目录 `outputs/babilong_sft_phase11_fsdp_500step_validate/` 现已具备 `adapter_config.json`、`mem_space_adapter_step000250.pt`、`mem_space_adapter.pt`，说明“优化器参数修复 + checkpoint 保存路径”这个验证目标已经通过。
- 但 validate training 末段 QUERY_DIAG 仍基本贴着旧的 flat floor：`step464 top1_sim_mean=0.002106`、`step484=0.002151`；因此这轮 validate 目前更像“训练/保存稳定性通过”，而不是“routing 已重新学起来”。
- 在 final ckpt 落盘后，`outputs/eval_p11_500step_validate/` 的 21-cell BABILong eval 已继续推进到 **`21/21` 个 result CSV 全部 materialize**；其中已有 `20/21` 达到 `100 parsed rows`，当前只剩 `qa1 32k=90/100` 这一条最终长尾还在跑。
- 当前本地只剩 1 个 validate worker 存活；GPU `3` 约 `55.0 GiB`、`99% util`，而 GPU `0/1/2/4/5/6/7` 已空闲。`outputs/eval_p11_500step_validate/p11_500step_validate_score.csv` 也已出现部分产物，但在最后这 10 个样本补齐前仍不能视为 canonical final score。
- 剩余 validate 尾巴的最新 eval-side QUERY_DIAG 仍为 `qa1 32k top1_sim_mean=0.002091`；这继续支持“优化器 bug 修好了，但 router 仍处在 flat-routing regime”这一解释，而不是已经发生 routing recovery。
- 与此同时，远程 `28.59.80.196` 已不再 idle：用户要求的 clean `P8 + selector_temperature=20` 500-step 对照短跑已在 tmux `p8_temp20_500_20260517_105421` 中健康运行到 `BABI step 40/500`，8 卡显存约 `90-95 GiB`、util `94-100%`，首个 QUERY_DIAG `top1_sim_mean=0.020874`，明显高于旧的 `~0.002` floor。
- 本轮 `git status --short` 仍显示 working tree 非干净，且不只状态文件：`.claude/commands/heartbeat.md`、`.gitignore`、`docs/`、`scripts/...`、`locomo` 等改动仍在；heartbeat 需要继续把 clean-tree / push 视为独立 WARNING，而不是默认仓库已干净。
- 当前自动闭环优先级：
  1. 盯 `outputs/eval_p11_500step_validate/` 从当前 `21/21 materialized, 20/21 complete` 继续推进到 `21/21 × 100 parsed rows`
  2. 一旦 validate eval 收齐，立刻产出评分表，并与 `temp20 final=35.24` / `step500=33.81` / `step4500=27.43` / `P11 final=26.33` / `P8=59.14` 对比
  3. 并行盯住 clean `P8 + selector_temperature=20` 短跑到 `step 500/500`，一旦 final `mem_space_adapter.pt` 落盘，就立刻接上同口径 21-cell BABILong eval
  4. 若 validate 最终分数仍明显弱于 `temp20 final`，则更应把 `selector_temperature=20` 继续视为当前 8B 主线里的默认 cheap lever，并按 researcher 排序转去 `memory gate = 0` eval-only 与 `num_slots=64 + temp20`
  5. 1B 线上继续优先做 **L2 eval-only disable** 或 **L2 reset fix 的短跑**
- 当前工作负载状态：本机 H20 正在收尾 validate final eval 的最后一条 `qa1 32k` 长尾（仅 GPU `3` 忙）；远程 `28.59.80.196` 则已被 clean `P8 + temp20` 500-step 短跑占满，形成用户要求的并行双前线。 
- 之后这条 clean `P8 + temp20` 500-step 远程短跑已在 `2026-05-17 11:20 CST` 干净完成：`logs/p8_temp20_500_20260517_105421.log` 给出 final `mem_space_adapter.pt` 与 `Training complete: steps=500 babilong=411 pg19=89 non-finite=0`。由于远程节点随即空闲，heartbeat 已在 `2026-05-17 23:42 CST` 自动拉起 canonical 21-cell final eval：tmux=`p8_temp20_final_eval_20260517_2342`，结果目录 `outputs/eval_p8_temp20_final_20260517_2342/`。
- 对这条 `p8_temp20_final_eval_20260517_2342` 的纠偏复查（`2026-05-18 00:43 CST`）表明：旧 heartbeat 之所以看到“顶层目录 `0 CSV`”，只是因为结果实际写在 `p8_temp20_final_{qa1,qa2,qa5}_{short,long}/` 子目录里；此时 tmux / eval worker 已全部退出，远程 8×H20 也已回到 `0 MiB / 0% util`，6 路日志均以 `Evaluation complete!` 收尾。
- 按 `babilong.metrics.compare_answers` 的 canonical 口径重算后，clean `P8 + selector_temperature=20` final 结果为：`qa1=65.00`、`qa2=31.57`、`qa5=67.29`，overall **54.62**，short avg **62.08**，long avg **44.67**；这比 `P11 temp20 final=35.24` 高 **+19.38pp**，比 `P11 final=26.33` 高 **+28.29pp**，但仍比旧 `P8=59.14` 低 **-4.52pp**。
- original B200 / cluster-1 路线也已纠偏：`.144` 并不是“没有可用环境”，而是**不能用 `torch-base`，必须改用 project `.venv`**。`/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/.venv/bin/python` 现已确认 `torch 2.10.0+cu128`、arch list 包含 `sm_100`，`torch.zeros(..., device="cuda")` 成功。
- heartbeat 已据此在 `.144` 上启动 `phase1b_v5_coldstart_alpha_origb200_20260518_004122`：tmux=`v5_coldstart_alpha_20260518_004122`，到 `2026-05-18 00:43 CST` 已推进到 `BABI step 250/5000`，8 卡显存约 `27.7–30.8 GiB`、util 非零，说明这条 original-B200 run 已真实进入训练循环。
- 但 v5 的早期机制信号还不乐观：`QUERY_DIAG top1_sim_mean` 在 `step 25/49/73/99/124/149/175/196/219/241` 仍落在 `0.002106–0.002228`，因此下一个真正的决策点是它能否在 `step 500` 前后摆脱旧的 flat-routing floor。
- 本地 `exp_b_train` 这一支则被确认只是 stale tmux wrapper：它的真实命令仍然指向缺失的 `scripts/exp_b_train.sh` 并只剩 `sleep 99999`；本轮 heartbeat 已将其 kill，当前本地 8×H20 已重新回到干净空闲态。

---

## 背景

- H-series v1 (Llama-3-8B base + NIAH 训练) 在 BABILong 上 0%
- Root cause (researcher 确认): base model 不会 follow instructions + 训练任务 (NIAH) 和 eval (BABILong) 分布完全不同
- Solution: 跟随 ARMT 两阶段配方 (PG19 LM pretrain → BABILong 任务微调)，换 Llama-3.2-1B (和 ARMT 论文一致)

---

## 训练矩阵

| 变体 | 架构 | Base 策略 | Memory | 节点 | Phase 1 log |
|---|---|---|---|---|---|
| **A** | cross-attn slots (write L8, read L10/12/14) | Frozen base | 64 slots | b200-1 (本地) | `logs/h_v2_phase1_A_b2001.log` |
| **B** | Joint attention (early H-series) | **Full FT** | 128 slots (scaled up) | b200-4 | `logs/h_v2_phase1_B_b2004.log` |
| **C** | ARMT associative memory (reference) | Frozen + memory tokens | 32 tokens, d_mem=64 | b200-2（HMT 完成后已切入） | `logs/h_v2_phase1_C_b2002.log` |
| **D** | cross-attn slots + LoRA | Frozen base + LoRA(r=8, α=32) | 64 slots | b200-3 (等 MemoryLLM 完) | `logs/h_v2_phase1_D_b2003.log` |

**Baselines 对照** (已跑/在跑)：
- ARMT full training: b200-5/6/7/8（已完成，train_loss≈2.70, eval_loss≈2.78）
- RMT full training: 被 SIGHUP 杀掉，**待重启**（优先考虑 replacement B200 空闲节点）
- HMT full training: b200-2（已完成；final test PPL≈11.405，节点已切给 H-v2 C Phase 1）
- MemoryLLM eval: ✅ 完成（b200-3 已释放并转给 H-v2 D）
- Beacon eval: ✅ 完成 (BABILong avg 60.9%)

---

## 2026-05-12 17:08 资源调度决议

### 当前资源图
- stable b200-1..4：全部占用（A/B/D + 新切入的 C Phase 1）
- replacement b200-5..8：全部空闲（32× L20A 可立即使用）
- h20-1..4：全部可达且空闲，但 MemoryLLM 仍需先在 h20-1 完成 verify 修复

### 本轮 heartbeat 强制执行计划
1. **立即在 b200-5 启动 H-v2 C Phase 1**，不再等待 b200-2 上 HMT 完成
2. **b200-6 已用于 RMT 重启**；保留 b200-7..8 作为 follow-up 训练池，优先给 C 的后续阶段、成熟 baseline 扩展、或 researcher/coder 产出的高置信新训练任务
3. **H20 侧先修后跑**：一旦 h20-1 上 MemoryLLM verify 通过，立刻在 h20-1 拉起 `1k/2k/4k/8k/16k/32k` full wave；h20-2..4 保留给后续成熟 eval/repro 扩展
4. **heartbeat 后续检查顺序**：优先确认 `b200-5` 的 C Phase 1 是否稳定输出 step/loss；其次确认 h20-1 verify 是否通过并触发 full wave

## 2026-05-12 18:09 H20-only pivot

- 用户最新澄清：**暂时不在的是 replacement B200 `b200-5..8`，稳定 B200 `b200-1..4` 仍在**。我上一轮把范围误写成了全部 B200 offline。
- 原 heartbeat cron 已暂停，避免继续按错误前提自动执行；若恢复 heartbeat，应恢复对 `b200-1..4` 的正常认知，并只把 `b200-5..8` 视为暂不可用。
- h20-1 上的 MemoryLLM verify 已补齐三处兼容性修复：
  1. `scripts/eval_baseline_babilong.py` 兼容 chat template 返回 `BatchEncoding`
  2. `scripts/eval_baseline_babilong.py` 为 MemoryLLM 补 `generation_config`
  3. `scripts/launch_h20_memoryllm.sh` 默认改用 `${WORKDIR}/.venv/bin/python`
- 之后已在 h20-1 触发 `1k/2k/4k/8k/16k/32k` 六路 MemoryLLM wave（`2026-05-12 18:08 CST`，`scripts/launch_h20_memoryllm.sh`）；当前这些任务已快速结束，日志显示结果文件已存在而 skip。
- 当前主执行面仍偏向 H20 结果核查，但这不等于 `b200-1..4` 不存在；它们仍可在后续 heartbeat/调度中纳入考虑。

## 2026-05-12 H20 剩余 eval 执行计划

### 当前已确认状态
- `Llama-3.2-1B-Instruct`：BABILong 6/6 length 已完成，可直接作为 vanilla reference。
- `Beacon-Qwen2-7B`：BABILong 6/6 length 已完成，当前是最强已发布 baseline。
- `MemoryLLM-8B-chat`：h20-1 结果目录已齐全覆盖 `qa1..qa10 × {1k,2k,4k,8k,16k,32k}` 共 60 个 csv；heartbeat 已重新刷新本地 `status/babilong_baselines_h20.json` / `status/babilong_results.json`，当前完整聚合后 AVG 约维持在 **31-37%**。
- `M+`：`scripts/eval_baseline_babilong.py` 已支持 `--baseline mplus`；h20-1 的 six-wave `1k/2k/4k/8k/16k/32k` 全量 eval 早已完成，`MPlus-8B` 结果目录也已齐全覆盖 **60/60 csv**，但那批 full-wave 分数仍应视为被 runtime bug 污染的无效 0%。后续修复已先移除 runtime crash（device mismatch / CUDA tensor→numpy），本轮 heartbeat 进一步确认 **H20 线上 evaluator 曾经实际跑的是 stale copy**，现已同步为本地带 RoPE `inv_freq` 重建补丁的版本。同步后，`MPlus-8B-smoke-fix4-20260513` 的 `qa1_1k` 单样本输出已不再是 `!!!!!!!!!!!!!!!!!!!!`，而是重复的 `Incomplete...` / `INT...` 片段（`correct=0`）。因此当前 blocker 已从 **punctuation-only collapse** 收敛为 **post-sync semantic corruption / generate-path mismatch**；full wave 仍不应直接重跑。基于本轮人工代码审计，最可疑的剩余差异是 `scripts/eval_baseline_babilong.py` 里 MPlus `_generate()` 仍手工构造 attention mask 并强制 `use_cache=False`，与 `MemoryLLM-source/README.md` 的原始 `model.generate(input_ids=..., max_new_tokens=20)` 用法不一致。
- `ARMT / RMT`：repo 内已有官方 BABILong finetune/eval 入口（`third_party/associative-recurrent-memory-transformer/run_finetuning_babilong_rmt.py` + `third_party/associative-recurrent-memory-transformer/scripts/babilong/*`），但前提是对应 checkpoint 已可读且 layout 对得上。本轮 heartbeat 已通过 SSH+rsync 将 `legacy/`、`babilong/`、`data/armt_pg19_real_tokenized_full/` 推到 H20 shared mount；其中 RMT 还额外补了 H20-side shim `src/memory/rmt -> ../../legacy/memory/rmt`，并在 `h20-2/3/4` 拉起 `rmt_eval_ppl_h20_2`、`rmt_eval_nih_h20_3`、`rmt_eval_mem_h20_4`。这 3 个 retry3 当前都已结束，且一致死在 `legacy/scripts/eval_rmt.py` 的 **legacy/v10 loader mismatch**：evaluator 期待 `memory_embeddings/extractor.*`，实际 checkpoint `rmt_memory.pt` 是 v10 key（`l0.memory`, `recon_head.proj.*`）。H20 上直接运行 `legacy/scripts/debug_eval_rmt_v10.py --skip_model` 已确认该 checkpoint 的 memory keys 本身是 **valid v10**。本轮进一步人工审计表明：repo 内现成的 **v10-aware loader** 在 `legacy/scripts/eval_needle_haystack_v10.py` / `legacy/scripts/eval_needle_haystack_extended.py`，但尚未确认有现成的 **BABILong v10 eval 入口**；因此下一步应基于这条 v10 loader 路径做 BABILong wrapper/adaptation，而不是继续重试 `legacy/scripts/eval_rmt.py`。ARMT checkpoint 本身仍未在 H20 shared mount 上确认 ready，因此本轮仍未直接启动 ARMT BABILong eval。
- `HMT`：当前 repo 未发现直接的 BABILong eval 入口；survey 也标记为 `BABILong: NOT reported`，因此应视为“需要适配”，不是 H20 上立刻可跑的现成 eval。
- replacement B200 当前节点映射：`b200-5=28.89.18.252`, `b200-6=28.89.20.82`, `b200-7=28.89.20.27`, `b200-8=28.89.18.19`；密码文件仍为 `configs/password_b200_ephemeral.txt`。当前更应以较新的 heartbeat 探测为准：`b200-5` / `b200-7` / `b200-8` 现返回 `Permission denied`，`b200-6` 现为 `Connection refused`，因此继续视为“unavailable / not ready for Mixture-of-Memory auto-launch”。历史上该池曾暴露 `share_304376610` 路径与 ARMT `past_key_values` runtime bug，但在认证与连通性重新稳定前，不应继续把它们当作可直接接管 C/RMT 的稳定训练池。

### 执行优先级（给 H20）
1. **P0 — 处理 `M+` 的生成退化问题**
   - 聚合已经完成；`status/babilong_results.json` / `status/babilong_baselines_h20.json` 已纳入 `MPlus-8B`，但当前分数无效。
   - 下一步不是重跑同样的 six-wave，而是先修复 `scripts/eval_baseline_babilong.py` / `MemoryLLM-source/modeling_mplus.py` 路径中的生成语义失真（当前 smoke-fix4 输出为重复 `Incomplete...` / `INT...`，不再是 `!!!!!!!!!!!!!!!!!!!!`）。
   - 优先验证点：让 MPlus `_generate()` 更贴近上游 README 用法（避免额外手工 attention mask，并优先恢复默认 cache generation 语义），再做 `qa1/1k` smoke。
2. **P1 — 跑 ARMT 的 BABILong eval**
   - 前提：ARMT checkpoint 在共享盘上已准备好。
   - 先做 `qa1` smoke / validate-only，确认 checkpoint + tokenizer + data path 全部打通；再扩成全任务全长度。
4. **P3 — 跑 RMT 的 BABILong eval**
   - 前提：RMT restart 产出可评估 checkpoint。
   - 执行方式同 ARMT：先 smoke，后 full sweep。
5. **P4 — HMT 适配**
   - 暂不作为 H20 本轮立即执行项；需要先补一个 BABILong adapter/eval path，再谈 full run。

### 建议实际 run order
1. **先处理 `M+` 的生成退化 bug**（结果已聚合，不再重复同样 six-wave；优先修复 `scripts/eval_baseline_babilong.py` / `MemoryLLM-source/modeling_mplus.py` 路径中当前 `Incomplete...` / `INT...` 重复输出的问题）
2. **优先修复 replacement B200 的 runtime blocker**（路径已切到 `share_304376610`，但本轮 heartbeat 对 b200-5/7/8 现有密码验证失败、b200-6 连接被拒；继续视作 unavailable / not ready）
3. **ARMT qa1 smoke → ARMT full eval**（仅在 checkpoint 路径 + replacement 节点 runtime 都确认可用后执行）
4. **RMT：先补 v10-aware BABILong 入口，再 qa1 smoke → full eval**（当前只确认 NIH/extended evaluator 内已有 v10 loader，不能继续直接复用 `legacy/scripts/eval_rmt.py`）
5. **HMT 暂缓**，先单列 TODO

### 每个方法的完成标准
- 结果目录下应覆盖 `qa1..qa10 × {1k,2k,4k,8k,16k,32k}` 共 **60 个 csv**。
- 每个 csv 应有 **100 行样本**。
- 跑完后必须立刻重跑聚合脚本，更新汇总 JSON；不要再出现 raw 已齐但 summary 仍 partial 的情况。

### 具体落地建议
- **MemoryLLM 聚合刷新**：优先使用 `scripts/aggregate_babilong_baselines.py`；若只想快速生成 baseline summary，也可用 `scripts/score_babilong_baselines.py`。
- **M+ 执行入口**：直接走 `scripts/eval_baseline_babilong.py --baseline mplus`；参数模板尽量照 `scripts/launch_h20_memoryllm.sh` 复用，避免再踩环境/代理/venv 差异。
- **ARMT/RMT 执行入口**：优先沿用 `third_party/associative-recurrent-memory-transformer/run_finetuning_babilong_rmt.py` 和其 `scripts/babilong/*.sh` 逻辑，不要先自行重写 evaluator。
- **HMT**：先补 eval 适配，再安排 H20；本轮只保留为 TODO。

### 本轮明确 TODO
- [x] 刷新 `MemoryLLM-8B-chat` 的聚合 JSON（本地 `status/babilong_baselines_h20.json` 已含 `MemoryLLM-8B-chat`）
- [x] 完成 `M+` BABILong 全量 eval（heartbeat 已完成 60/60 csv 聚合，但结果暴露 runtime bug：当前有效分数为全长度 0%）
- [ ] 修复 replacement B200 的 Mixture-of-Memory runtime blocker（路径已切到 `share_304376610`，`.venv + fla` 已通，但 ARMT 仍在 `language_modeling.py` 的 `past_key_values` 处崩溃）
- [ ] 若 checkpoint ready 且 replacement 环境可用，完成 `ARMT` BABILong eval
- [ ] 若 checkpoint ready，完成 `RMT` BABILong eval
- [ ] 为 `HMT` 补一个可复用的 BABILong eval path
- [ ] 暂不做 `Infini-attention`
- [ ] `MemLong` 延后（依赖更多环境/FAISS，且优先级低于 M+/ARMT/RMT）

---

## Phase 1: PG19 LM Pretrain

**配方**（照抄 ARMT）：
- Dataset: `data/armt_pg19_real_tokenized_full` (已预处理)
- Steps: 50,000
- LR: 1e-5 (memory-only 变体), 2e-5 (full FT 变体 B)
- Scheduler: linear, warmup 5000
- Optimizer: AdamW, wd=0.01, grad clip 1.0
- bf16, 8 GPU, effective batch 64 (per-device 1 × grad_accum 8 × 8 GPU)
- segment_size=512, max_n_segments=2
- `--no_loss_from_first_segment` (只在第2个 segment 上算 loss)

**预计时间**：
- A/D (frozen + memory only): **~3 小时**
- B (full FT): **~4 小时**
- C (ARMT): **~3 小时**

**Phase 1 成功标志**: eval PPL 收敛，train loss < 3.0

---

## Phase 2: BABILong 任务微调

**配方**（照抄 ARMT）：
- Task: qa1 (单任务起步, single-supporting-fact)
- Dataset: BABILong qa1 train split + PG19 noise injection
- Steps: 30,000 (curriculum 2→4→8→16→32 segments, 各 6000 steps)
- LR: 1e-4, linear, warmup 1000
- Loss: **仅在答案 token 上计算** (labels_mask 其他全 -100)
- 使用 chat template 包装 prompt (即使 base model，保持与 eval 格式一致)

**预计时间**：
- 单变体: **10-30 小时** (取决于 curriculum)

**Phase 2 成功标志**: BABILong qa1 eval accuracy > 20% (非零就是里程碑)

### 2026-05-12 当前高置信 follow-up
- **H-v2 D 下一条最值得开的新臂**：把 Phase 2 从 `qa1-only` 改为 `qa1+qa2+qa3` multi-task mix，并保留少量 PG19 noise + instruction-style format wrapper。该建议来自本轮 researcher 高置信结论，优先作为 D 的 follow-up ablation。
- **Baseline 队列**：`M+` 六路跑完后，先聚合 `MPlus-8B`，再推进 `ARMT` BABILong eval；但 ARMT 之前必须先修复 replacement B200 的路径/环境问题。
- **2026-05-13 审计→修复闭环**：上述两点现已落地到代码：`scripts/train_h_v2_babilong.py` 已改为使用 BABILong prompt template 做 answer-only supervision，`scripts/train_h_v2.py` 已去掉 segment 边界的 slot-state `.detach()`；本地 compile 与 dataset sanity 已通过。启动 D 的 multi-task follow-up 时应直接沿用这条已修补路径。

---

## 进度跟踪

### 2026-05-11 里程碑

- [x] PG19 tokenized 数据准备完成 (25552 books, 21GB)
- [x] 4 变体训练脚本准备完成 + smoke 通过 (A/B 跑通, C smoke 需加 `--sample_size 1024` 的 fix)
- [x] ARMT full 训练启动 (b200-5/6/7/8, 4 节点并行)
- [x] Beacon BABILong 完整 eval 完成 (60.9% avg)
- [x] H-v2 A Phase 1 启动 (b200-1)
- [x] H-v2 B Phase 1 启动 (b200-4)
- [x] H-v2 A Phase 1 完成 → Phase 2 启动（2026-05-12 13:31 已在 b200-1 启动 `v2_phase2_A_b2001`）
- [x] H-v2 B Phase 1 完成 → Phase 2 启动（2026-05-12 13:31 已在 b200-4 启动 `v2_phase2_B_b2004`）
- [x] MemoryLLM eval 完成 → b200-3 释放 → H-v2 D Phase 1 启动
- [x] H-v2 D Phase 1 完成 → Phase 2 启动（2026-05-12 13:31 已在 b200-3 启动 `v2_phase2_D_b2003`）
- [x] H-v2 C Phase 1 已在 b200-5 启动（2026-05-12 17:12，tmux `v2_phase1_C_b2005`）；b200-2 上 HMT 继续独立跑完
- [x] ARMT full PG19 在 b200-5/6/7/8 完成（train_loss≈2.699-2.710, eval_loss≈2.778-2.785）
- [x] RMT 重启已在 b200-6 拉起（2026-05-12 17:12，tmux `rmt_v10_l0l1_b2006`）

### Phase 2 里程碑

- [ ] H-v2 A Phase 2 (BABILong qa1) 启动 + 完成（已于 2026-05-12 13:31 启动，待完成）
- [ ] H-v2 B Phase 2 启动 + 完成（已于 2026-05-12 13:31 启动，待完成）
- [ ] H-v2 C Phase 2 启动 + 完成
- [ ] H-v2 D Phase 2 启动 + 完成（已于 2026-05-12 13:31 启动，待完成）
- [ ] 4 变体 BABILong eval → 对比表
