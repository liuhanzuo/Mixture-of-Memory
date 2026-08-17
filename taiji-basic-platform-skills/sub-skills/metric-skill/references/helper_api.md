## query_hunyuan_swanlab_run_columns

查询 SwanLab 实验中某个 run 的列名（keys/指标字段）。

> 💡 **适用场景**：用户问"有哪些 SwanLab 指标/列名/keys"且已知 workspace/project/run 信息时调用。
> ⚠️ **前置条件**：需先调用 `query_hunyuan_train_swanlab_metrics` 获取 `workspace`/`project_name`/`run_id`/`swanlab_api_key`，再用返回的值传入本工具。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_swanlab_run_columns '{"task_id": "basic_train_xxx", "swanlab_api_key": "...", "workspace": "...", "project_name": "...", "run_id": "..."}'
```

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|:---:|------|
| `task_id` | string | ✅ | 任务 ID（`basic_train_` 开头） |
| `swanlab_api_key` | string | ✅ | SwanLab API Key，从 `query_hunyuan_train_swanlab_metrics` 返回获取 |
| `workspace` | string | ✅ | SwanLab 工作空间名，从 `query_hunyuan_train_swanlab_metrics` 返回获取 |
| `project_name` | string | ✅ | SwanLab 项目名，从 `query_hunyuan_train_swanlab_metrics` 返回获取 |
| `run_id` | string | ✅ | SwanLab Run ID，从 `query_hunyuan_train_swanlab_metrics` 返回获取 |
| `page` | integer | ❌ | 页码 |
| `page_size` | integer | ❌ | 每页数量 |
| `column_type` | string | ❌ | 列类型过滤 |

> 🛑 只调用一次，禁止换 page/page_size/column_type 反复重试。本工具**仅用于列名查询**，查指标数据用 `query_hunyuan_train_swanlab_metrics`。
