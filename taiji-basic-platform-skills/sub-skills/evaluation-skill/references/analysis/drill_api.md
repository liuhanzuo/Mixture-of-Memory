# 评测下钻分析 API

## get_taiji_eval_drill_dimensions

查询评测任务在指定集合版本上的可下钻维度列表，同时返回该版本下的所有评测集。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_drill_dimensions '{"task_id": 67290, "collection_version_id": 2372}'
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `task_id` | number | ✅ | 评测任务 ID |
| `collection_version_id` | number | ✅ | 集合版本 ID，通过 `get_taiji_eval_collection_version_detail` 获取 |

**返回字段**：`dims`（维度列表，含 id/name/values）、`exercises`（评测集列表，含 id/name）。

---

## get_taiji_eval_drill_metrics

按指定维度下钻查询聚合指标。**支持一次传多个维度**（`dimension_types` 为数组），返回各维度值的交叉聚合分数。

**MCP 工具调用：**
```bash
# 单维度
python3 scripts/connect_mcp.py call get_taiji_eval_drill_metrics '{"task_id": 67290, "collection_version_id": 2372, "dimension_types": ["task_lv1"]}'

# 多维度一次查询
python3 scripts/connect_mcp.py call get_taiji_eval_drill_metrics '{"task_id": 67290, "collection_version_id": 2372, "dimension_types": ["task_lv1", "difficulty", "language"]}'
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `task_id` | number | ✅ | 评测任务 ID |
| `collection_version_id` | number | ✅ | 集合版本 ID |
| `dimension_types` | string[] | ✅ | 维度类型数组，如 `["task_lv1"]` 或 `["task_lv1", "difficulty"]`。维度名来自 `get_taiji_eval_drill_dimensions` 返回的 `dims[].id` |

**返回字段**：`dimensions`（数组，每项含 `key` 维度值、`label` 标签、`count` 数量、`scores` 各 task 分数）、`overall_scores`（总体分数）、`overall_count`（总题数）。

> 🔑 **一次传入所有需要的维度**，不要每个维度单独调一次。单维度空值（如 task_lv1 全为空 `(空)`）不代表需要逐个试其他维度——`drill_dimensions` 已列出可用维度，直接选择有值的维度批量传入。
>
> ❌ **严禁**：`dimension_types=["task_lv1"]` → 发现为空 → `["task_lv2"]` → `["difficulty"]` → `["language"]` 依次调 4 次。
> ✅ **正确**：先调 `drill_dimensions` 看可用维度 → 一次传 `["task_lv1", "difficulty", "language"]`。
