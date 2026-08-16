---
name: heartbeat-preauth-kill-rm
description: 用户预授权 heartbeat 自主执行 kill -9 / rm -rf 类腾卡命令，不等审批
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0dac7a11-5048-4ecf-85c3-ff6b9fab88d3
---

用户 2026-07-12 预授权：autonomous heartbeat 里为腾卡/清理常需 `kill -9 <GPU进程PID>`、`rm -rf <临时结果目录>` 等命令，**可自主执行，不再等待用户审批**。仅限本项目 GPU 运维范围（杀训练/eval 进程、清 partial 结果目录），**不含 git push / 外发 / 删非本人创建的重要数据**。

**Why:** 2026-07-12 一个待审批的危险命令（腾卡的 kill -9 / rm）在用户不在时阻塞了整个会话 **5.5 小时**，期间 .85.73 的 8 张 H20 空转（违反铁律1）。审批机制在无人值守的 heartbeat 场景会卡死会话。

**How to apply:** 腾卡杀进程优先按显式 PID 或 `nvidia-smi --query-compute-apps=pid` 报的 PID（**绝不用 `pkill -f <脚本名>`**——会匹配自身 shell 命令行自杀，已知坑 [[a13b-known-pitfalls]]）；清 partial eval 目录用 `rm -rf ruler_results/.../<cell>` 直接执行。git push/外发仍需走审核流程（不在此授权内）。