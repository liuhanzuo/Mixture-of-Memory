## get_taiji_eval_task_confidence

查询评测任务在各维度的置信区间（Bootstrap 重采样），判断分数差异是否统计显著。

> ⚠️ **选择规则**：用户只问"评测任务 X 的得分置信区间是多少"时，优先走主链路 `get_taiji_eval_task_detail`，因为任务详情通常已包含置信区间/得分摘要。只有用户明确提到 `bootstrap`、统计显著性、指定 `collection_version_id` 做维度置信区间，或需要自定义 `confidence_level` 时，才调用本工具，且同一问题只调用一次。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_task_confidence '{"task_id": 8828, "collection_version_id": 379}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `task_id` | int | ✅ | 评测任务 ID |
| `collection_version_id` | int | ✅ | 集合版本 ID |
| `score` | string | ❌ | 评分模式，默认 `default`，可选 `acc` |
| `bootstrap` | int | ❌ | Bootstrap 采样次数，默认 1000 |
| `confidence_level` | float | ❌ | 置信水平，默认 0.95 |
| `missing_value_mode` | string | ❌ | 缺失值处理，默认 `ignore` |

**返回：** `AbilityInsight` 列表，含 `weightedScore`、`confidenceIntervalLower`、`confidenceIntervalUpper`。

---

## get_taiji_eval_bench_confidence

查询 Agent 评测任务在指定 Bench 上的置信度。

> ⚠️ **不知道 `collection_version_id` 时**先调 `get_taiji_eval_task_detail(task_id)` 获取，再用完整参数调此工具。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_bench_confidence '{"task_id": 8828, "collection_version_id": 379, "exercise_version_id": 3863}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `task_id` | int | ✅ | 评测任务 ID |
| `collection_version_id` | int | ✅ | 集合版本 ID |
| `exercise_version_id` | int | ❌ | 评测集版本 ID |

**返回：** `AgentBenchConfidenceResponse`，含各 Bench 的置信区间。

---

## get_taiji_eval_performance_trend

查询评测任务的模型性能趋势数据（散点图 + 折线图原始数据，用于可视化分析）。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_performance_trend '{"task_id": 8828, "collection_version_id": 379, "bucket_strategy": "token"}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `task_id` | int | ✅ | 评测任务 ID |
| `collection_version_id` | int | ✅ | 集合版本 ID |
| `exercise_version_id` | int | ❌ | 评测集版本 ID |
| `bucket_strategy` | string | ❌ | 分桶策略，如 `token` |

**返回：** `PerformanceTrendResponse`，含散点图 + 折线图的原始数据点。

---

## get_taiji_eval_task_progress

查询评测任务的整体进度概览（抓取阶段 + 评估阶段的进度和错误率）。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_task_progress '{"task_id": 8828}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `task_id` | int | ✅ | 评测任务 ID |

**返回：** `TaskOverallProgressResponse`，含抓取进度 + 评估进度的整体概览（total / predicted / completed / error_rate）。
