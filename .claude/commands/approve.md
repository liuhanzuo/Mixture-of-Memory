---
model: glm-5.1
---

---
model: glm-5.1
---

# /approve — 批准或拒绝 trainer 请求

## 调用方式

```
/approve                    ← 列出所有待审批请求，让用户决定
/approve <request_id>       ← 批准指定请求
/approve <request_id> deny  ← 拒绝指定请求
```

## 执行步骤

### 1. 读取待审批请求

```
Read: status/TRAINER_REQUESTS.jsonl
Read: status/TRAINER_APPROVALS.jsonl
```

找出 request_id 未在 APPROVALS 中出现的条目。

### 2. 无参数时：展示待批请求

格式：
```
## 待审批请求

### req_20260423_001 — config_change [urgency: high]
- **实验**: sparse_memory_v4
- **问题**: OOM at step 0, GPU memory 92/95 GiB
- **建议方案**: 
  A. 减少 max_seq_length 4096→2048
  B. 减少 batch_size 2→1
  C. 两者都减（推荐）
- **影响**: 训练速度约降低 30%

回复 `/approve req_001 A` 选择方案 A
回复 `/approve req_001 deny` 拒绝并说明原因
```

### 3. 有参数时：记录决定

批准时追加到 `status/TRAINER_APPROVALS.jsonl`：

```json
{
  "timestamp": "ISO8601",
  "request_id": "<对应的 request_id>",
  "decision": "approved",
  "chosen_option": "A|B|C|custom",
  "notes": "用户的具体指令",
  "approved_by": "main"
}
```

拒绝时：
```json
{
  "timestamp": "ISO8601",
  "request_id": "<对应的 request_id>",
  "decision": "rejected",
  "reason": "...",
  "alternative_suggestion": "...",
  "approved_by": "main"
}
```

### 4. 批准后自动触发

批准后，追加到 `UPDATELOG.md`，然后**建议**运行 `/trainer` 执行批准的动作（不自动触发）。
