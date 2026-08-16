---
name: kill-remote-gpu-job-by-pid-not-pkill
description: "★远程杀 GPU 训练:pkill -f 常打不中(run 名只在 --output_dir 里)且会自杀 ssh 自己的进程组导致输出被吞;正解=先 ps 拿 PID,再 setsid bash -c 'kill -9 <pids>' </dev/null,然后另开连接验 nvidia-smi 归零"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

远程 kill GPU 训练**不要**用 `pkill -f <run名>`，用 **PID**。

**为什么 `pkill -f` 会失败（2026-08-05 连续踩 3 次）**：
- 训练 run 的名字往往**只出现在 `--output_dir` 里**（如 `paperC_pc1_squad_A1_full32ft`），
  而 torchrun 父进程的 cmdline 与 worker 不同 → `pkill -f 'A1_full32ft'` 打不中父进程，
  8 个 worker 被 torchrun 立刻重启，`nvidia-smi` 显存丝毫不降、PID 一模一样。
- 更坑：`pkill` 会连**当前 ssh 会话自己的进程组**一起杀 → 命令**没有任何输出返回**
  （不是命令失败，是输出被吞），容易误判成"杀成功了"。CODEBUDDY.md 里 A13B 那条
  「pkill 自杀 shell 坑」就是同一个东西。

**正解**：
```bash
# 1. 先拿 PID（连 ppid 一起看，确认父子结构）
ssh ... "ps -eo pid,ppid,etime,cmd | grep -E '<pattern>' | grep -v grep"
# 2. setsid 脱离 + </dev/null，避免杀到自己 / 被 ssh 断连带走
ssh ... 'setsid bash -c "kill -9 <parent_pid> <worker_pids...>; sleep 5;
   for p in \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 \$p; done" </dev/null >/dev/null 2>&1; echo KILL_ISSUED'
# 3. 另开一个连接验证（不要复用同一条）
ssh ... "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader;
         nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader"
```
先杀 torchrun 父进程再兜底扫 compute-apps，否则父进程会补活 worker。

**另一半教训：sharded eval 必须校验 shard 数。**
同一天发现 `scripts/eval_olmo2_probe2_ppl.py --merge` 只校验 `n_tokens>0`，于是
#103 crossing 曲线的 `reheal_step{55000,57500}` 在 shard 0/1/7 因 co-resident 任务吃满显存
（`CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate`）而死后，
**静默 merge 了 5/8** → 2560 windows 的 PPL 看起来完全正常，却与前面 8-shard/4096 的点
**口径不一致**，而那条曲线正是用来 bracket crossing 的。已修（commit d380bbc）：解析
`of{N}` 后缀、要求 N 个齐全、拒绝混合 N、`--allow_partial_merge` 才放行并大声 WARN。
⇒ **任何 sharded eval harness 都要这样做**：merge 前断言 shard 完整，别只看结果"像不像正常值"。

相关：[[cluster-two-disks-not-shared]]、[[heartbeat-preauth-kill-rm]]
