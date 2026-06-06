# PENDING_TASKS.md — Task Board
## Updated 2026-06-07 03:20 CST

---

## [DONE 2026-06-07 03:20] 4-arm chunk512 step500 ablation 评分 + 裁决
- 4 臂全训到 5000、step500 ckpt 同口径 BABILong 评完。**P11 (delta-rule + normalized writeback) = 新最佳臂**，qa5 1k-8k=86/83/64/50 超 top_k16 基线（76/77/54/48）。P10(ST-Gumbel 硬路由) 与 topk8 均劣于基线 → REJECTED。结果锁进 RUN_REGISTRY.md §3 + MEMORY_PROTOCOL_PLAN P10/P11。

## [PENDING] 下一臂：在 P11(delta-rule+normreadout) 底座上叠加 — auto_launch: false（等用户确认方向）
- P11 已确立为新基线配置。建议下一步择一在此底座叠加：
  1. **P11 + chunk1024**（chunk size 仍是最大杠杆，§4 观察1）— 验证 delta-rule 增益能否与更大 chunk 叠加
  2. **P13 surprise-gated write**（写强度∝预测误差，Titans 2501.00663）— 现在 retrieval 已被 P11 抬起，可解锁
  3. **P11 + register slots(P9 num_global_slots)** 组合
- 3 个 H20 节点已空闲（local + .76 全空，.196 收尾 P11 32k eval）→ 可并行铺开。**等用户 go-ahead 起跑。**

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
