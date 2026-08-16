---
name: kill-hung-train-must-exclude-eval
description: "★kill \"hung train\" 时必须 grep -v eval 排除 EVAL 收尾进程 (2026-08-06 误杀 keep28_scratch merge, 丢 EM 数字, 需重跑 14 min); 判据是命令名不是 GPU util"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**Rule**: kill hung training 时，pgrep 结果里**必须先 grep -v 掉 `eval_paperC_squad_emf1|eval_olmo2|.*eval.*\.py`
之类 EVAL 收尾进程**，只 kill 训练 python (train_olmo2_arch_probe2 / torch.distributed.run)。

**Why**: 2026-08-06 16:24 犯的错。观测到 LOCAL keep28 scratch：
- 训练 log 显示 step 1000/1000（训练确实完成，final.pt 38.4GB 已存）
- 8 个 python train 进程 etime 58 min，util 0%（真的 hung 在 shutdown）
- 但同时 `pgrep -f 'keep28_scratch_refusal25|torch.distributed.run'` 会**同时**匹配到
  wrapper 已经 chain 到的 `eval_paperC_squad_emf1.py --merge --output_name depthsweep_keep28_scratch_refusal25`
  ——因为 `--output_name` 含关键字！

我一把梭 `for p in $(pgrep ...); do kill -9 $p; done` → merge 被误杀 → keep28 scratch 的 EM/F1
**永久丢失**，只能等本轮 chain 全跑完后手动重跑 EVAL (14 min，final.pt 还在)。

**How to apply**:
1. **kill 前必看命令名**，不能只看 "GPU util=0%" 或 "etime 长"：
   ```bash
   pgrep -af '<pat>' | grep -vE 'eval_.*\.py|_emf1\.py|--merge|score_'
   ```
2. 判断 "hung train" 的三条件（须全部满足）：
   - GPU util = 0%（不吃计算）
   - 命令名是 `train_olmo2_arch_probe2` / `torch.distributed.run` / `torchrun`（**不是** eval/score/merge）
   - `final.pt` **已存**（train 主体真的完成了；未存的话是 crash，直接 kill 也无所谓）
3. **正例**（这次真该 kill 的）：
   `python -u scripts/train_olmo2_arch_probe2.py --keep_front_layers 28 --from_scratch ...`
4. **反例**（这次误杀的）：
   `python scripts/eval_paperC_squad_emf1.py --merge --output_name depthsweep_keep28_scratch_refusal25`
   —— pattern 匹配是因为 `--output_name` 里包含 keep28_scratch_refusal25。**pattern 命中 ≠ 该 kill**。
5. 更稳的模式：写 pattern 直接锚定 `train_olmo2_arch_probe2\.py` **且**不锚 output_dir，或者反过来：
   `pgrep -af 'train_olmo2_arch_probe2\.py.*from_scratch'`（直接匹配训练特征参数）。
6. 只 kill 3-5 秒后，如果 wrapper (`bash scripts/run_paperC_depthsweep.sh`) 还活着，让它自己 chain。
   **不要** kill wrapper —— wrapper 死了整条 sweep 挂。上次 wrapper (PID 2475071) 幸存是运气。

**Related**: [[kill-remote-gpu-job-by-pid-not-pkill]]（另一条 kill 教训：pkill -f 打不中远端 run
名），[[heartbeat-preauth-kill-rm]]（heartbeat 有 kill 授权，但授权不等于豁免核对命令名）。
