## verify_hunyuan_swanlab_identity

验证 SwanLab API Key 是否有效，并返回当前用户基本信息。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |

**调用示例：**
```bash
python3 scripts/connect_mcp.py call verify_hunyuan_swanlab_identity '{"swanlab_api_key": "your_key"}'
```

---

## query_hunyuan_swanlab_workspace_list

查询当前用户可访问的所有 SwanLab 空间列表。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ❌ | 空间用户名，不传则为当前登录用户 |

**调用示例：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_swanlab_workspace_list '{"swanlab_api_key": "your_key"}'
```

---

## get_hunyuan_swanlab_workspace_detail

获取指定 SwanLab 空间的详细信息。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ✅ | 空间用户名 |

**调用示例：**
```bash
python3 scripts/connect_mcp.py call get_hunyuan_swanlab_workspace_detail '{"swanlab_api_key": "your_key", "workspace": "X2"}'
```

---

## query_hunyuan_swanlab_project_list

查询指定空间下的 SwanLab 项目列表（分页），支持按创建时间/名称排序、关键词模糊搜索。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ✅ | 空间用户名 |
| `sort` | string | ❌ | 排序方式：`create`（创建时间）/ `name`（名称），默认按更新时间 |
| `search` | string | ❌ | 搜索关键词，模糊匹配项目名 |
| `page` | integer | ❌ | 页码，从 1 开始，默认 1 |
| `page_size` | integer | ❌ | 每页条数，默认 20。**仅接受白名单值：`10, 12, 15, 20, 24, 27, 50, 100`，传入其他值将返回 400 参数错误** |

**调用示例：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_swanlab_project_list '{"swanlab_api_key": "your_key", "workspace": "X2", "page": 1, "page_size": 10}'
```

---

## get_hunyuan_swanlab_project_detail

获取指定 SwanLab 项目的详细信息。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ✅ | 空间用户名 |
| `project_name` | string | ✅ | 项目名称 |

**调用示例：**
```bash
python3 scripts/connect_mcp.py call get_hunyuan_swanlab_project_detail '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project"}'
```

---

## query_hunyuan_swanlab_run_list

查询指定项目下的 SwanLab 实验列表（分页）。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ✅ | 空间用户名 |
| `project_name` | string | ✅ | 项目名称 |
| `page` | integer | ❌ | 页码，从 1 开始，默认 1 |
| `page_size` | integer | ❌ | 每页条数，默认 20。**仅接受白名单值：`10, 12, 15, 20, 24, 27, 50, 100`，传入其他值将返回 400 参数错误** |

**调用示例：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_swanlab_run_list '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project"}'
```

---

## get_hunyuan_swanlab_run_detail

获取指定 SwanLab 实验的详细信息。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ✅ | 空间用户名 |
| `project_name` | string | ✅ | 项目名称 |
| `run_id` | string | ✅ | 实验 ID（experiment_id） |

**调用示例：**
```bash
python3 scripts/connect_mcp.py call get_hunyuan_swanlab_run_detail '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project", "run_id": "abc123"}'
```

---

## get_hunyuan_swanlab_run_profile

获取指定 SwanLab 实验的 profile 配置信息，包含 config（超参数配置）、metadata（元数据）、requirements（依赖包列表）、conda（conda 环境信息）。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ✅ | 空间用户名 |
| `project_name` | string | ✅ | 项目名称 |
| `run_id` | string | ✅ | 实验 ID |

**调用示例：**
```bash
python3 scripts/connect_mcp.py call get_hunyuan_swanlab_run_profile '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project", "run_id": "abc123"}'
```

---

## filter_hunyuan_swanlab_runs

通过条件过滤获取 SwanLab 实验列表。支持两种模式：
1. **简化模式**：通过 `state`/`name`/`group`/`job_type`/`labels`/`config_filters` 参数过滤
2. **高级模式**：直接传入 `filters` 数组（SDK 原生格式）

两种模式互斥，同时传入以 `filters` 为准。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ✅ | 空间用户名 |
| `project_name` | string | ✅ | 项目名称 |
| `state` | string | ❌ | 实验状态：`FINISHED`/`RUNNING`/`CRASHED`/`ABORTED` |
| `name` | string | ❌ | 实验名模糊匹配 |
| `group` | string | ❌ | 实验分组 |
| `job_type` | string | ❌ | 任务类型 |
| `labels` | array[string] | ❌ | 标签列表 |
| `config_filters` | array[string] | ❌ | config 过滤条件，格式 `['key=value', ...]` |
| `filters` | array[object] | ❌ | 高级过滤条件（SDK 原生格式），与简化参数互斥 |
| `page` | integer | ❌ | 页码，从 1 开始，默认 1 |
| `page_size` | integer | ❌ | 每页条数，默认 20，最大 100 |

**调用示例：**
```bash
# 简化模式：查询运行中的实验
python3 scripts/connect_mcp.py call filter_hunyuan_swanlab_runs '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project", "state": "RUNNING"}'

# 高级模式
python3 scripts/connect_mcp.py call filter_hunyuan_swanlab_runs '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project", "filters": [{"key": "state", "type": "STABLE", "op": "EQ", "value": ["FINISHED"]}]}'
```

---

## query_hunyuan_swanlab_run_metrics

查询指定 SwanLab 实验的标量指标数据。支持多种过滤模式。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ✅ | 空间用户名 |
| `project_name` | string | ✅ | 项目名称 |
| `run_id` | string | ✅ | 实验 ID |
| `keys` | array[string] | ✅ | 指标名列表，如 `["loss", "accuracy"]` |
| `step_start` | integer | ❌ | 按 step 过滤起点（闭区间），与 ts_start/ts_end 互斥；step 区间必须用顶层参数 `step_start`，不要写嵌套 `step_range.from` |
| `step_end` | integer | ❌ | 按 step 过滤终点（闭区间），与 ts_start/ts_end 互斥；step 区间必须用顶层参数 `step_end`，不要写嵌套 `step_range.to` |
| `ts_start` | integer | ❌ | 按时间戳过滤起点（UNIX 毫秒），与 step 参数互斥 |
| `ts_end` | integer | ❌ | 按时间戳过滤终点（UNIX 毫秒），与 step 参数互斥 |
| `last` | integer | ❌ | 最近 N 毫秒的数据，与 step/ts 参数互斥 |
| `head` | integer | ❌ | 取前 N 个数据点，与 tail 互斥 |
| `tail` | integer | ❌ | 取后 N 个数据点，与 head 互斥 |
| `sample` | integer | ❌ | 采样数量，最大 1500 |
| `all` | boolean | ❌ | 是否返回全量数据（不受采样限制），默认 false |
| `ignore_timestamp` | boolean | ❌ | 是否去除时间戳字段，默认 false |

**参数互斥规则：**
- `head` 和 `tail` 不能同时指定
- `step_start`/`step_end` 与 `ts_start`/`ts_end` 不能同时使用
- `last` 与 `step_start`/`step_end`/`ts_start`/`ts_end` 互斥

**调用示例：**
```bash
# 查询最近 100 个数据点
python3 scripts/connect_mcp.py call query_hunyuan_swanlab_run_metrics '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project", "run_id": "abc123", "keys": ["loss", "accuracy"], "tail": 100}'

# 查询 step 1000~2000 区间
python3 scripts/connect_mcp.py call query_hunyuan_swanlab_run_metrics '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project", "run_id": "abc123", "keys": ["loss"], "step_start": 1000, "step_end": 2000}'

# 全量数据
python3 scripts/connect_mcp.py call query_hunyuan_swanlab_run_metrics '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project", "run_id": "abc123", "keys": ["loss"], "all": true}'
```

---

## get_hunyuan_swanlab_run_summary

获取指定 SwanLab 实验的标量指标统计摘要（min/max/avg/median/latest）。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ✅ | 空间用户名 |
| `project_name` | string | ✅ | 项目名称 |
| `run_id` | string | ✅ | 实验 ID |
| `keys` | array[string] | ❌ | 指标名列表，不传则返回全部指标的摘要 |

**调用示例：**
```bash
python3 scripts/connect_mcp.py call get_hunyuan_swanlab_run_summary '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project", "run_id": "abc123"}'
```

---

## query_hunyuan_swanlab_run_columns

查询指定 SwanLab 实验下的指标列列表（分页），支持按关键词搜索、按列分类和数据类型过滤。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ✅ | 空间用户名 |
| `project_name` | string | ✅ | 项目名称 |
| `run_id` | string | ✅ | 实验 ID |
| `page` | integer | ❌ | 页码，从 1 开始，默认 1 |
| `page_size` | integer | ❌ | 每页条数，默认 100。**仅接受白名单值：`10, 12, 15, 20, 24, 27, 50, 100`，传入其他值将返回 400 参数错误** |
| `search` | string | ❌ | 模糊搜索关键词 |
| `column_class` | string | ❌ | 列分类过滤：`CUSTOM`/`SYSTEM` |
| `column_type` | string | ❌ | 列数据类型过滤：`FLOAT`/`STRING`/`IMAGE`/`AUDIO`/`VIDEO`/`TEXT`。参数名必须写 `column_type`，不要写成 `type`。 |

**调用示例：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_swanlab_run_columns '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project", "run_id": "abc123", "column_type": "FLOAT"}'
```

---

## query_hunyuan_swanlab_run_logs

获取指定 SwanLab 实验运行时的文本日志。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ✅ | 空间用户名 |
| `project_name` | string | ✅ | 项目名称 |
| `run_id` | string | ✅ | 实验 ID |
| `offset` | integer | ❌ | 分页偏移量，默认 0 |
| `level` | string | ❌ | 日志级别过滤：`DEBUG`/`INFO`/`WARN`/`ERROR`，默认 INFO |
| `ignore_timestamp` | boolean | ❌ | 是否去除时间戳字段，默认 false |

**调用示例：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_swanlab_run_logs '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project", "run_id": "abc123", "level": "ERROR"}'
```

---

## query_hunyuan_swanlab_run_medias

获取指定 SwanLab 实验的媒体数据（图片、音频、视频等），返回预签名 URL。

**参数：**

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `swanlab_api_key` | string | ✅ | SwanLab API Key |
| `workspace` | string | ✅ | 空间用户名 |
| `project_name` | string | ✅ | 项目名称 |
| `run_id` | string | ✅ | 实验 ID |
| `keys` | array[string] | ✅ | 媒体指标名列表；用户只说"图片/图片媒体数据"且未给具体 key 时默认 `["generated_images"]`。 |
| `step` | integer | ❌ | 指定 step，不传则返回最新 |
| `all` | boolean | ❌ | 是否获取全部历史媒体数据，默认 false |

**调用示例：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_swanlab_run_medias '{"swanlab_api_key": "your_key", "workspace": "X2", "project_name": "my_project", "run_id": "abc123", "keys": ["generated_images"]}'
```
