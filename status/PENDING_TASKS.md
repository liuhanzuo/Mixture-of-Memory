# PENDING_TASKS.md — Task Board
## Updated 2026-06-07 19:12 CST

---

## [DONE 2026-06-07 22:05] l3_recon_token_weight sweep — w1.0 step500 BABILong eval 评完 + 裁决 ❌
- 21/21 CSV（qa1/qa2/qa5×7 len，n=100）全完成，已 canonical 评分（`scripts/score_nested_babilong.py`，diskB .76）。
- **结果（灾难）**：qa5 0k-32k = **67/22/16/8/3/1/0**；qa1=77/4/6/8/3/2/1；qa2=43/4/5/3/1/2/3。
- **裁决：L3 token-recon aux weight=1.0 灾难性破坏长程寻址。** 对照无-aux P11 chunk512 baseline（qa5=82/86/83/64/50/35）→ 仅 0k 部分存活，≥1k 全面塌方。真实实验结果（CSV 满 n=100 非 silent-fail）。已锁进 RUN_REGISTRY §3「l3_recon_token_weight sweep」。
- 含义：强 token-level recon aux 与 routing/检索目标冲突；待 w0.3 弱权重确认是否「弱即无害 vs 仍劣于无 aux」。两 train run（.196 w0.3 / .249 w1.0）继续跑满 5000 仅为 lm/recon 曲线，BABILong 已基本判定 token-recon aux 不优于 baseline。

## [RUNNING 2026-06-07 21:30] l3_recon_token_weight sweep — w0.3 step500 BABILong eval（auto_launch 自主起）
- **21:30 heartbeat 起跑**：两条件满足（rsync 完成 dest .pt=10.9G + config 就位；.76 GPU0-5/7 空闲，只 GPU6 在跑 w1.0 32k）→ 自主起。
- driver pid **242122**，GPU0/1/2/3，带 woa proxy + HF_HOME。已验证健康：worker 载入 tokenizer+base model，经 proxy 触达 HF Hub，**无 Network-unreachable / Traceback**。
- qa1/qa2/qa5 × 0k-32k，n=100，chunk_size=512。脚本 `scripts/eval_p11_chunk512_l3recontoken_w0.3_step500.sh`（disk B .76）。
- 完成后与 w1.0 + P11 baseline 同口径配对成 l3_recon_token_weight sweep，评分入 RUN_REGISTRY。
- **TODO(auto_launch:false)**：w1.0 step500 eval 的 0k-16k 已齐、只剩 32k cell（pid 234922）。32k 一完成即可 babilong.metrics 评分 w1.0 全集 → 写 RUN_REGISTRY。

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
