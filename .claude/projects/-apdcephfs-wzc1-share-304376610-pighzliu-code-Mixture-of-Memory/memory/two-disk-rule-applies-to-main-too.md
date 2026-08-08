---
name: two-disk-rule-applies-to-main-too
description: "★\"文件/数字不存在\"的结论在 wzc1+zwfy6 两盘都搜过前不成立 —— 这条对 MAIN 自己和 subagent 同等适用 (2026-08-06 同一天犯两次)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**Rule**: 任何形如「文件不存在」/「数字追不到 provenance」/「这个实验没结果」的结论，
在 **wzc1 与 zwfy6 两个盘都搜过之前一律不成立**。这条对 **MAIN 自己**和 subagent 完全同等适用。

**Why**: 2026-08-06 同一天，同一根因发作两次：
1. 早上：派 Explore subagent 找 paperA primitive 数字 provenance，它只搜 wzc1 → 误报 2/3
   「no candidate found」。我纠正了它，并写下 memory [[subagent-audit-must-specify-cross-disk]]。
2. **几小时后我自己犯同一个错**：宣布 `tab_replay_latency` 的 931.9/664.4 ms
   「provenance 追不到，有 <2% 漂移」，还据此写了 audit md + 一整段 rebuttal 措辞（own-drift）
   + 建议用户 GPU 重跑三选一。实际 `bench_results/p0_13_quality_latency/latency/latency_proc{0,1,2}.json`
   **在 wzc1 上根本不存在、只在 zwfy6(.82)**，池化后 bit-exactly 复现 tex 全部六个数字（含 p10/p90）。
   我甚至在第一次修正时把原因误写成「我漏搜了子目录」——也是错的，那目录在 wzc1 不可能被搜到。

教训不是"记住有两个盘"（我记住了，还写了 memory），而是**把规则应用到自己身上**：
写 memory 警告 subagent 的那一刻，没意识到同一条约束正是我自己下一个要踩的。

**How to apply**:
1. 说出「X 不存在」之前，先跑两盘：
   ```bash
   ls <path>                                   # wzc1 (本地)
   sshpass -f configs/password_h20_82250.txt ssh -o StrictHostKeyChecking=no \
     -o PreferredAuthentications=password root@28.82.250.82 \
     "ls /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/<path>"
   ```
   （注意：**省略 `-p`**，见 [[ssh-omit-p-flag-port-36000]]）
2. **先问"同一张表/同一次实验的其他数字来自哪个目录"**，去那个目录找 —— 而不是全盘 grep 数值近似。
   数值近似会把你引到**不同 protocol** 的实验上（本例：p0_12 的 pack sha `f7fc7617…` ≠ P0.13 的
   `cae91f9a…`，用它做参照就制造了一个不存在的 150ms「回退」）。
3. 找到只在单盘的证据后，`scp -O` 一份 md5-identical 副本到 wzc1，让证据链在主盘可审计。
4. **代价意识**：这次误判让我写了整套 own-drift rebuttal 措辞、建了 3 选 1 决策项、
   占用用户一次决策，全部白费。"追不到 provenance" 是个**重指控**，门槛要高。

**Related**: [[subagent-audit-must-specify-cross-disk]]（同一事实的 subagent 版）、
[[cluster-two-disks-not-shared]]（物理事实）。
