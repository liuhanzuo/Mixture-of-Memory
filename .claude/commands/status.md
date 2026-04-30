---
model: claude-haiku-4-5-20251001
---

---
model: claude-haiku-4-5-20251001
---

# /status — 项目状态快速总览

一键输出当前项目全貌。

## 执行步骤

### 1. 读取状态文件

```
Read: status/TRAINER_ACTIVE.md
Read last 10 lines: UPDATELOG.md
Read last 5 lines: status/TRAINER_REQUESTS.jsonl
Read last 3 lines: status/RESEARCHER_REPORTS.jsonl
Read last 5 lines: status/TRAINER_ACTIVITY.jsonl
Read: configs/remote_experiments.json
```

### 2. 快速 GPU 检查

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

### 3. 输出格式

```
## 项目状态 [YYYY-MM-DD HH:MM GMT+8]

### 本地 GPU
GPU 0: X GB / 97.8 GB  utilization: X%
...

### 活跃训练
- [实验名]: step X/Y, loss Z, 状态
  或：无

### 远程集群
- b200-1 (28.89.17.143): [实验名] — [状态]
- b200-2: ...
- b200-3: ...
- b200-4: ...

### 待审批请求
- [request_id]: [内容]  ← 需要 /approve
  或：无

### 最近动作
- [UPDATELOG 最后几条摘要]

### 研究状态
- 上次 researcher 结论: [摘要]
- 当前优先方向: [方向]

### 建议
- [下一步应该做什么]
```
