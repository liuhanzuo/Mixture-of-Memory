---
name: cluster-two-disks-not-shared
description: "★2026-08-04实测:5节点分属两盘(wzc1=LOCAL+.21 / zwfy6=.73+.82+.104),旧文档「全共享wzc1」是错的;.73上/apdcephfs_wzc1是指向zwfy6的symlink,.82上根本不存在;跨盘必须scp -O"
metadata: 
  node_type: memory
  type: project
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**5 个 GPU 节点分属两个物理盘，不是"全部共享 wzc1"** —— 旧 CODEBUDDY.md/CLAUDE.md 写的「5 台全部共享 wzc1 项目盘、互相无需 rsync」是**错的**，已在 2026-08-04 让多个 agent 白跑（Paper B #128 两次卡在 environment check、Paper C #133 差点找错 root）。已改正文档顶部。

- **wzc1 盘** = **LOCAL + .21**：`/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/`（这两台真共享，互相无需 rsync）。
- **zwfy6 盘** = **.73 / .82 / .104**：`/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/`。**是另一份独立 checkout，commit 常落后**（实测 `2d98c5a`，且非 local HEAD 的祖先）。

**两个坑（实测）：**
- **.73**：`/apdcephfs_wzc1` 是**指向 zwfy6 的 symlink** → wzc1 路径字符串"看着能用"，但物理盘不同于 LOCAL/.21，**写进去 LOCAL 看不到**。.73 的 PROJECT_ROOT 要写 zwfy6 路径。
- **.82**：`/apdcephfs_wzc1` **根本不存在**。

**How to apply：**
- 派 GPU agent 前，先让它**自己探测并确认 PROJECT_ROOT**，别在 prompt 里假定 wzc1 路径通用。
- wzc1-only 的新脚本/ckpt 要在 H20 三台跑 → 必须显式 **`scp -O`** 过去（**.82 的 sftp subsystem 已坏，普通 `scp` 报 `subsystem request failed`**），搬完核 md5/sha256。
- 结果产在 zwfy6 → 要进论文/主仓必须 `scp -O` 回 wzc1。
- **合成 16 卡多机 DDP 只能同盘内**：LOCAL+.21，或 .73/.82/.104 任两台；**不可跨盘**。
- 软件差异：三台 H20 `.venv/bin/python` 已坏，且 **LOCAL 的 `.venv` 现也无 torch**（2026-08-04）→ 一律 `/opt/conda/envs/torch-base/bin/python`；**.82 未装 bitsandbytes** → `OPT=bnb8bit` 在 .82 不可用。

相关：[[dllm-h20-node]]、[[h20-paperA-over-paperB-priority]]
