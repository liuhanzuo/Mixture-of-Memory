---
name: dllm-h20-node
description: QCMem 可调度集群 = 4 节点 32 卡（本机+.21 两台 B200 wzc1 盘 + .73/.82 两台 H20:36000 zwfy6 盘）；★.104 已于 2026-08-05 交还用户绝不碰；死节点清单；原 dllm 节点 29.162.226.120 绝不碰
metadata: 
  node_type: memory
  type: project
  originSessionId: 6c395da6-15fa-436e-b529-3d4585cc5de2
  modified: 2026-08-04T04:38:16.465Z
---

**★ QCMem 可调度集群（2026-08-05 更新）= 4 节点 32 卡**。⚠️ **不是全部共享 wzc1 盘**（旧记录/CLAUDE.md "全在 wzc1" 已被 2026-08-04 实测证否）——分两处物理盘：

- **本机 (local) + `28.89.19.21`（22 端口，8×B200）= 共享 wzc1 盘** `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`（两台免同步）。本机 8×L20A（B200 级 183GB），`.venv/bin/python`（torch2.10+cu128，sm_100）；.21 密码 `configs/password_b200_19021.txt`。⚠️ **本机 `.venv/bin/python` 已坏**（2026-08-03 被 reset 成裸 py3.11 无 torch）→ 本机用 `/opt/conda/envs/torch-base/bin/python`（torch2.13）。
- **`28.85.35.73` + `28.82.250.82`（★均 36000 端口，各 8×H20）= 共享 zwfy6 盘**，规范路径 `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`（2026-08-04 marker 写读 + md5 + git HEAD 实测同盘）。⚠️ **.82 上没有 wzc1 alias**（用 wzc1 路径会 no-such-file）；.73 上 wzc1 路径是指向 zwfy6 的符号链接。两台 git HEAD=2d98c5a（比 B200 陈旧）。密码：.73=`configs/password_h20_853573.txt`，.82=`configs/password_h20_82250.txt`。
- 🚫 **`28.83.24.104` 已于 2026-08-05 交还用户**（用户原话：「.104你不要管了 我在用」）→ **不派任务、不 kill 其上进程、不因看到空闲去补卡、heartbeat 不算 idle 也不报 WARNING**。密码文件 `configs/password_h20_24104.txt` 仍在但不要用它起活。
- ⚠️ **连带影响：zwfy6 侧「合成 16 卡多机 DDP」只剩 `.73+.82` 唯一组合**（原先 .73/.82/.104 任取两台）；wzc1 侧仍是 LOCAL+.21。**跨盘不可合**。
- ⚠️ **两处盘不跨节点可见**：B200 上提交的代码（如 cacheblend @81949b0）H20 看不到，需 cat-over-ssh/`scp -O` 同步到 zwfy6（H20 无 rsync 二进制、**.82 的 sftp subsystem 已坏**普通 scp 会报 `subsystem request failed`）。
- ⚠️ 每台 H20 的 `/opt/conda/envs/torch-base` 是**节点本地**（不随盘共享），缺包 `pip install`（via hy-proxy）。
- **死节点（别再试，均实测拒连/超时）**：B200 `28.89.16.18` `28.89.18.188` `28.88.184.53` `28.89.16.55`；H20 `28.83.53.31` `28.48.7.53` `28.58.245.174` `28.59.80.196` `28.49.x`；H800 `30.203.x`。
- **`29.162.226.120` = 原 dllm 专用节点，2026-07-11 归还但已改密拒连——绝不连**（历史见下）。
- 节点分工见 [[h20-eval-b200-train-split]]；方向2/4 见 [[bottleneck-layer-sweep-monotone]]。

---
## 历史（dllm 项目，2026-07-07~07-10，节点已归还，仅存档）
- 29.162.226.120 曾是 dllm(Confidence-Based-MDLM) 专用、与 agent 双向隔离；项目在 `/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm`（独立 venv `.venv_dllm` torch2.11）。2026-07-10 20:13 起 SSH 密码被拒（节点重启改密码），07-11 以新 IP 归还。[[dllm-autonomous-phase-transition]]
- **命题C 全线收官 + 正向定律（Exp13-32）**：命题C(训练制造dLLM并行)全维度证否；升级为"负结果+正向定律"——**dLLM 可无损并行度 = 任务 token 条件独立度**（copy 零依赖 steps16 仍 1.00 / GSM8K 强依赖最早掉精度，Exp32 n=100）。主贡献=系统性负结果论文，骨架 `dllm/docs/paper_skeleton.md`。findings 记 Exp20-32。
