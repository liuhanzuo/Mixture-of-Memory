---
name: reassign-node-revoke-old-owner-first
description: "★2026-08-08 我改节点归属却没撤销旧 owner agent 的授权, 两 agent 为 .21 打了30分钟编辑战(guard 反复插入/回滚, 3轮PID kill, tripwire被删); 换节点主人必须先 stand-down 旧的再派新的"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

当用户改变优先级、导致某个节点要换用途时，**必须先给当前 owner agent 发 stand-down，确认它停了，再派新 owner**。绝不能只派新 agent 去"接管"——那等于让两个 agent 抢一个资源。

**Why:** 2026-08-08 用户在 ~14:55 把优先级改成「SparseForge 优先，Paper B resume 让路」。当时 agent `a00fb877` 正在执行**旧指令**（把 Paper B resume 迁到 .21），已跑了 40 分钟。我没通知它，直接派 `a418fb72` 去腾空 .21 给 SparseForge。结果两个 agent 打了约 30 分钟：

- `a418fb72` 往启动脚本插 `exit 1` guard → `a00fb877` 在 ~20 秒内移除并反插注解 `USER INSTRUCTION 2026-08-08: B200用于resume, .21 = keep10 resume (not SparseForge)`
- 三轮 PID kill（2786/4147 → 5731/5767 → 7086/7122），每轮 ~60 秒 respawn
- `a00fb877` 删掉了对方放的 tripwire；`.21` 上 `chattr` 未安装，无法硬锁路径
- 真正的 respawn 源是 `scripts/transfer_dolmino_pipe.sh` 被改成了 `MAX_ATTEMPTS=20 + sleep 30` 的自重启 wrapper —— 只 kill 传输腿永远不够，必须先杀 driver

两个 agent 各自都在忠实执行自己收到的指令，**错全在我**：我改了共享资源的归属，却没撤销前一个持有者的 mandate。

代价其实很低（`.21` GPU 全程 0%，因为它的 trainer 阻塞在不完整数据集上；且 keep10 本来就在 .82 跑得好好的，迁移从一开始就没必要），但这是运气好，不是设计好。

**How to apply:**
1. 优先级一变，先列出所有 in-flight agent 及它们各自占用/目标的节点
2. 对受影响的 agent 用 SendMessage 发 stand-down（明确说"你的任务已取消 + 为什么 + 只许做清理 + 禁止再启动"）
3. **确认它真的停了**（进程清零 + 它自己回报）再派新 owner
4. stand-down prompt 里要求它给自己的启动脚本加 guard 而非删脚本，便于将来复用
5. 清理时注意区分 **半截文件 vs 源文件**：删之前比对大小与源端，本次差点删错（wzc1 的 5.8 GB 半截 vs .73 源端 43,867,047,810 B）

相关：[[subagent-prompt-must-state-gpu-budget]]、[[kill-remote-gpu-job-by-pid-not-pkill]]、[[kill-hung-train-must-exclude-eval]]
