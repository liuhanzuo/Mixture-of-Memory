# FORWARD_PLAN_20260515

## 0. heartbeat ↔ plan workflow
- heartbeat 每次必须同步刷新 `PENDING_TASKS.md`、`TRAINER_ACTIVE.md`、`H_V2_PLAN.md`、`TRAINER_ACTIVITY.jsonl`
- GPU 空闲且存在 `auto_launch=true` 的延伸任务时，同一拍直接启动，不允许只记录不执行

## 7. 状态日志
- [2026-05-17 08:54 GMT+8] 本地 `temp20 final BABILong eval` 已全部完成并打分：`21/21` 个 cell 全部达到 `100 parsed rows`；`qa1=36.86`、`qa2=12.86`、`qa5=56.00`，overall **35.24**，short avg **45.42**，long avg **21.67**。
- [2026-05-17 08:54 GMT+8] `temp20 final` 明确优于 flat-routing 对照：vs `step500=33.81` 为 **+1.43pp**，vs `step4500=27.43` 为 **+7.81pp**，vs `P11 final=26.33` 为 **+8.91pp**；但仍显著落后 `P8=59.14`。
- [2026-05-17 08:54 GMT+8] 远程 `step000500` checkpoint eval 也已完成并打分：overall **33.81**，short avg **45.92**，long avg **17.67**；远程 `28.59.80.196` 当前已无 eval worker，仅剩 stale tmux `p11_step4500_eval_20260517_040951`。
- [2026-05-17 08:54 GMT+8] 本地 8×H20 已自动切入新的 `p11_fsdp_500step_validate` 修复验证训练；`logs/p11_fsdp_500step_validate_20260517_0851.log` 已到 `PG19 step 10/500`，8 卡显存约 `56-97 GiB`、util `43-100%`，run 目前健康。
- [2026-05-17 09:38 GMT+8] `p11_fsdp_500step_validate` 已在本地 `8×H20` 上完成：`logs/p11_fsdp_500step_validate_20260517_0851.log` 给出 `step 500/500`、保存 final `mem_space_adapter.pt`，并以 `Training complete: steps=500 babilong=411 pg19=89 non-finite=0` 收尾。
- [2026-05-17 09:49 GMT+8] validate 输出目录现已具备 `adapter_config.json`、`mem_space_adapter_step000250.pt`、`mem_space_adapter.pt`；对应的 final 21-cell eval 也已启动，`outputs/eval_p11_500step_validate/` 当前已有 `12/21` 个 CSV materialize，已知 parsed-row 进度包括 `qa1 0k=100`, `qa1 1k=90`, `qa1 8k=40`, `qa2 0k=100`, `qa2 1k=100`, `qa2 2k=10`, `qa2 8k=40`, `qa5 0k=100`, `qa5 1k=100`, `qa5 2k=100`, `qa5 4k=20`, `qa5 8k=50`。
- [2026-05-17 09:49 GMT+8] 无论训练末段还是 eval 侧，QUERY_DIAG 仍基本停在 `~0.00206-0.00222`；这说明 validate run 已验证“优化器修复 + checkpoint 保存成功”，但 routing 仍未明显脱离 flat regime。远程 `28.59.80.196` 继续保持 `0 MiB / 0% util` 空闲态；`git status --short` 也仍显示 working tree 非干净，且不只状态文件（含 `.claude/commands/heartbeat.md`、`.gitignore`、`docs/`、`scripts/...`、`locomo`）。
- [2026-05-17 10:36 GMT+8] validate final eval 已继续推进到 `21/21` 个 result CSV 全部 materialize，其中 `18/21` 已达到 `100 parsed rows`；当前只剩 `qa1 32k=20`、`qa2 32k=30`、`qa5 32k=40` 三个 `32k` 长尾仍在跑。
- [2026-05-17 10:36 GMT+8] 本地只剩 3 个 long worker 存活在 GPU `3/4/5` 上（约 `40.9-55.0 GiB`、`99% util`），GPU `0/1/2/6/7` 已空闲；最新 eval QUERY_DIAG 仍在 `~0.00208-0.00221`，远程 `28.59.80.196` 继续 idle，且 working tree 仍 dirty beyond status files。
- [2026-05-17 10:58 GMT+8] validate final eval 现已推进到 `21/21` 全 materialize、`20/21` 达到 `100 parsed rows`；当前只剩 `qa1 32k=90` 这一条最终长尾仍在本地 GPU `3` 上运行，其他本地 GPU 已空闲。
- [2026-05-17 10:58 GMT+8] 远程 `28.59.80.196` 已不再空闲：用户要求的 clean `P8 + selector_temperature=20` 500-step ablation 已在 tmux `p8_temp20_500_20260517_105421` 中健康运行到 `BABI step 40/500`，8 卡显存约 `90-95 GiB`、util `94-100%`，首个 QUERY_DIAG `top1_sim_mean=0.020874`，明显高于旧的 `~0.002` floor。

- [2026-05-17 11:09 GMT+8] validate final eval 已收齐并 canonical 评分：`qa1=35.86`、`qa2=13.57`、`qa5=51.71`，overall **33.71**；short avg **45.17**，long avg **18.44**。这与 `P11 step500=33.81` 几乎相等（**−0.10pp**），与 `temp20 final=35.24` 差 **−1.53pp**，距 `P8=59.14` 仍 **−25.43pp**——优化器参数修复**没有**显著改善任务分。
- [2026-05-17 11:09 GMT+8] 远程 `p8_temp20_500_20260517_105421` 已健康推进到 `BABI step 280/500`，`step000200` checkpoint 已落盘；近期 `top1_sim_mean ∈ [0.008, 0.017]` 持续维持；本机 GPU 已全空闲。
- [2026-05-17 11:09 GMT+8] researcher `general-purpose-5` 在 11:07 违规启动了 2-GPU smoke 训练 `phase11_ddp_500step_validate_smoke_20260517_1107`（`total_steps=10`），已 kill PID `1341944/1341951/1341952`，删除 `outputs/`、`logs/` 中相关产物，并向 researcher 下达 read-only 整改指令；其报告 `ops/research_notes/2026-05-17_p8_scaleup_analysis.md` 仍 pending。
- [2026-05-17 23:42 GMT+8] clean `P8 + selector_temperature=20` 500-step run 已在远程 `28.59.80.196` 干净完成（`logs/p8_temp20_500_20260517_105421.log` 以 `Training complete: steps=500 ... non-finite=0` 收尾），heartbeat 已立即把空闲远程 H20 切换到 canonical 21-cell final eval：tmux=`p8_temp20_final_eval_20260517_2342`，结果目录 `outputs/eval_p8_temp20_final_20260517_2342/`。
- [2026-05-17 23:57 GMT+8] 对远程 `p8_temp20_final_eval_20260517_2342` 的二次探针已确认它不是卡在 startup：6 个 worker 仍存活，远程 GPU `0..5` 已占用约 `32/32/36/45/36/38 GiB`；日志显示 `qa1_short` 已完成 `1k 100/100`、`qa2_short` 已进入 `2k`、`qa5_short` 到 `4k 16/100`，long workers 已在 `16k 8–13/100`。当前唯一未满足的是首批 CSV 仍未 materialize，因此下一拍继续盯目录落盘即可。
- [2026-05-17 23:57 GMT+8] original B200 / cluster-1 的 blocker 进一步收敛：当前 direct-root 路径下，只有 `28.89.17.144` 能登录，但它的 `torch-base` 不支持 `sm_100`；`28.89.17.143` 在 `22` 端口 password denied、`36000` 端口 connection refused，`28.89.17.85` 与 `28.89.19.134` 在 `22/36000` 都对当前凭据返回 password denied。所以下一步应优先准备兼容 `sm_100` 的独立 PyTorch 环境，而不是继续重复试同一组凭据。
- [2026-05-17 23:42 GMT+8] 本地 `exp_b_train` 只是一个 stale tmux wrapper：其真实命令仍指向缺失的 `scripts/exp_b_train.sh`，并只剩 `sleep 99999`；本拍已 kill 该 wrapper，当前本地 8×H20 回到 `0 MiB / 0% util` 干净空闲态。
- [2026-05-17 23:39 GMT+8] original B200 / cluster-1 的 root SSH 歧义已解决：`root@28.89.17.144` 可直连，但远端 `/opt/conda/envs/torch-base` 为 `torch 2.8.0 + cu128`、arch list 仅到 `sm_90`；节点报告 `NVIDIA L20A` / compute capability `(10,0)`，即使 `torch.zeros(1, device="cuda")` 也会触发 `CUDA error: no kernel image is available for execution on the device`。因此 v5 launch 当前被环境兼容性阻塞，而不是认证问题。

- [2026-05-18 00:43 GMT+8] 对 `outputs/eval_p8_temp20_final_20260517_2342/` 的纠偏表明：旧 heartbeat 之所以看到“顶层目录 `0 CSV`”，只是因为结果实际写在 `p8_temp20_final_{qa1,qa2,qa5}_{short,long}/` 子目录里；此时 tmux 和 eval worker 已全部退出，远程 8×H20 回到 `0 MiB / 0% util`，6 路日志均以 `Evaluation complete!` 收尾。
- [2026-05-18 00:43 GMT+8] 按 `babilong.metrics.compare_answers` 的 canonical 口径重算后，clean `P8 + selector_temperature=20` final 结果为 `qa1=65.00`、`qa2=31.57`、`qa5=67.29`，overall **54.62**，short avg **62.08**，long avg **44.67**；对比 `P8=59.14` 为 **-4.52pp**，对比 `P11 temp20 final=35.24` 为 **+19.38pp**，对比 `P11 final=26.33` 为 **+28.29pp**。
- [2026-05-18 00:43 GMT+8] original B200 `.144` 的 blocker 也已纠偏：真正结论不是“没有可用环境”，而是“不能用 `torch-base`，必须改用 project `.venv`”。heartbeat 已用 `.venv/bin/python`（`torch 2.10.0+cu128`, `sm_100`, `CUDA_OK`）在 tmux `v5_coldstart_alpha_20260518_004122` 中拉起 `phase1b_v5_coldstart_alpha_origb200_20260518_004122`；到 `00:43` 已推进到 `BABI step 250/5000`，8 卡显存约 `27.7–30.8 GiB`，但早期 `QUERY_DIAG top1_sim_mean` 仍在 `~0.0021`，下一决策点是 `step 500`。

## 8. 当前推进决策树
1. 最高优先级：盯远程 `p8_temp20_final_eval_20260517_2342` 到首批 CSV materialize；随后持续监控直至 `21/21 × 100 parsed rows`，并立刻做 canonical scoring，给出 clean `P8 + temp20` 对照
2. original B200 的下一步不再是反复试密码/用户名，而是二选一：继续探测 cluster-1 其他节点是否已有兼容 `sm_100` 的环境，或单独准备新的 PyTorch 环境后再重启 `phase1b_v5_coldstart_alpha`
3. 已确认: 修复 "FSDP 优化器参数 bug" 不能解释 P11→P8 的 25-pp gap；分析重心继续放在 P8 vs P11 的剩余配方/数据/调度差异，而不是回到旧的 optimizer 假设
4. 用户已倾向于把 L2 视为可疑模块，并对 P8 路线更感兴趣；若本机后续有空闲且 researcher 报告支持，新的便宜训练优先围绕 P8 / L2-off 配方展开
5. clean-tree / push 仍保持独立低优先级 WARNING；在远程 eval 与 B200 环境问题都闭环前，不把当前脏树误判为 "已整理完毕"
