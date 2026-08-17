## list_hunyuan_train_available_metrics

查询指定训练任务支持的所有训练指标及其可用的聚合方式。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call list_hunyuan_train_available_metrics '{"task_id": "<TASK_ID>"}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | string | ✅ | 训练任务 ID，格式如 `basic_train_zhongtaohe_20260306155021_e684d920` |
| instance_id | string | ❌ | 训练实例 ID，不传则自动取最新实例 |

**返回（成功）：**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "task_id": "basic_train_zhongtaohe_20260306155021_e684d920",
    "metric_count": 7,
    "metrics": [
      {
        "key": "step",
        "name": "step",
        "ch_name": "训练进度",
        "monitor_type": "second",
        "aggregation_types": [
          {"name": "Current_period_step_begin", "description": "当前周期的起始step", "example": "当前周期从第几步开始的"},
          {"name": "Current_period_step_end", "description": "当前周期迭代的结束step", "example": "当前周期训练到第几步了"}
        ]
      },
      {
        "key": "loss",
        "name": "lm_loss",
        "ch_name": "lm_loss",
        "monitor_type": "second",
        "aggregation_types": [
          {"name": "Latest_10_step_avg", "description": "最近10个step平均", "example": "最近10步的平均loss"},
          {"name": "Latest_10_step_max", "description": "最近10个step最大", "example": "最近10步的最大loss"}
        ]
      }
    ],
    "summary": "Task basic_train_xxx has 7 available metrics: ..."
  }
}
```

> 💡 **展示引导（必须严格遵守）**：返回结果 `data.metrics` 中每个指标的 `aggregation_types` 已包含 `name`（聚合名）、`description`（中文描述）和 `example`（提问示例），**必须按照表格格式向用户展示**，不得简化为只列出聚合名。

---

## query_hunyuan_train_metric_text

查询训练任务的指标数据，支持聚合模式和原始数据模式。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_metric_text '{"task_id": "<TASK_ID>", "metric": "loss", "latest_n": 10}'
```

**两种查询模式（互斥）：**
| 模式 | 触发条件 | 说明 |
|------|----------|------|
| 聚合模式 | `aggregations` 有值 或两者均为空 | 返回聚合统计值 |
| 原始数据模式 | `latest_n > 0` | 返回最近N条原始数据点，不做聚合 |

> ⚠️ **语义映射与参数名硬约束**：
> - 参数名必须写 `aggregations`（复数），严禁写 `aggregation`/`aggregation_type`。
> - 「总体平均 / 平均值 / 总平均」→ `aggregations="Total_step_avg"`。
> - 「最近 N 步的值/数据」→ `latest_n=N`；「最近 loss 值」默认 `latest_n=100`；「现在 / 当前 / 到多少了」→ `latest_n=1`。
> - 多个指标一起查（如 `loss 和 grad_norm`）→ 一次调用，`metric="loss,grad_norm"`，不要拆成两次调用。
> - `aggregations` 与 `latest_n` 互斥；`aggregations` 的值传字符串，不传数组；不要为同一问题连续用多个窗口或再补聚合查询。

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | string | ✅ | 训练任务 ID |
| metric | string | ✅ | 指标名称，多个用逗号分隔，如 `"loss,grad_norm"` |
| instance_id | string | ❌ | 训练实例 ID，不传则自动取最新实例 |
| aggregations | string | ❌ | 聚合方式，多个用逗号分隔，如 `"Latest_10_step_avg,Total_step_avg"`。与 latest_n 互斥 |
| latest_n | int | ❌ | 返回最近N条原始数据点。大于0时启用原始数据模式，与 aggregations 互斥 |
| start_time | string | ❌ | 查询起始时间，格式 `"yyyy-MM-dd HH:mm:ss"` |
| end_time | string | ❌ | 查询结束时间，格式 `"yyyy-MM-dd HH:mm:ss"` |

**返回（聚合模式）：**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "task_id": "basic_train_xxx",
    "instance_id": "ins-001",
    "metrics": [
      {
        "key": "loss",
        "name": "lm_loss",
        "ch_name": "lm_loss",
        "aggregations": {
          "Latest_10_step_avg": 1.3117,
          "Total_step_avg": 2.456
        },
        "data_points": null,
        "summary": "lm_loss (loss): Latest_10_step_avg=1.3117, Total_step_avg=2.456"
      }
    ],
    "summary": "Metric query results for task xxx, mode: aggregation"
  }
}
```

**返回（原始数据模式）：**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "task_id": "basic_train_xxx",
    "instance_id": "ins-001",
    "metrics": [
      {
        "key": "loss",
        "name": "lm_loss",
        "ch_name": "lm_loss",
        "aggregations": {},
        "data_points": [
          {"time": "2026-03-06 14:00:00", "step": 1050, "value": 1.31},
          {"time": "2026-03-06 14:01:00", "step": 1051, "value": 1.32}
        ],
        "summary": "lm_loss (loss): latest 10 data points, range [1.3100 ~ 1.3500]"
      }
    ],
    "summary": "Metric query results for task xxx, mode: raw data (latest 10)"
  }
}
```

> ⚠️ **指标单位说明（必须遵守）：**
> | 指标 key | 原始单位 | 展示时的单位处理 |
> |----------|----------|------------------|
> | `current_elapsed_time` | **毫秒（ms）** | 展示时标注为「耗时（毫秒）」，或换算为秒（÷1000） |
> | `loss`、`grad_norm` | 无量纲 | 直接展示数值即可 |
> | `step` | 步数 | 直接展示整数 |

---

## query_hunyuan_train_metric_chart

生成训练任务的指标趋势图，返回图片的 HTTPS URL（上传至 COS）和文字摘要。

> ⚠️ **调用时机**：仅当用户表述中**明确包含画图信号词**时才调用。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_metric_chart '{"task_id": "<TASK_ID>", "metric": "loss,grad_norm"}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | string | ✅ | 训练任务 ID |
| metric | string | ✅ | 指标名称，多个用逗号分隔，如 `"loss,grad_norm"` |
| instance_id | string | ❌ | 训练实例 ID，不传则自动取最新实例 |
| latest_n | int | ❌ | 只画最近N个数据点（步），传0则画全部数据 |
| start_time | string | ❌ | 查询起始时间，格式 `"yyyy-MM-dd HH:mm:ss"` |
| end_time | string | ❌ | 查询结束时间，格式 `"yyyy-MM-dd HH:mm:ss"` |

**返回内容：**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "task_id": "basic_train_xxx",
    "instance_id": "ins-001",
    "chart_url": "https://xxx.cos.ap-guangzhou.myqcloud.com/mcp-chart/basic_train_xxx_20260310_204600.jpg",
    "metric_names": ["lm_loss", "grad_norm"],
    "start_time": "2026-03-06 14:00:00",
    "end_time": "2026-03-10 20:46:00",
    "summary": "Chart generated for task basic_train_xxx with 2 metrics."
  }
}
```

> 💡 **展示规则**：从返回 JSON 的 `data.chart_url` 取出图片 URL，拼接为 `![](chart_url)` 后**直接原样输出**。
> **⚠️ 严禁自行画图**：严禁使用 Python（matplotlib/PIL/plotly 等）自行生成图片。

---

## list_hunyuan_train_tf_events_metrics

列出模版化训练任务的可用 tf_events 指标。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call list_hunyuan_train_tf_events_metrics '{"task_id": "finetuning_xxx"}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | string | ✅ | 任务 ID（`finetuning_` 开头的模版化训练任务） |
| instance_id | string | ❌ | 实例 ID，不传则按 running → latest 自动解析 |

---

## query_hunyuan_train_tf_events_text

查询模版化训练任务的 tf_events 指标值。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_tf_events_text '{"task_id": "finetuning_xxx", "metrics": ["actor/ppo_kl", "loss"]}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | string | ✅ | 任务 ID（`finetuning_` 开头） |
| instance_id | string | ❌ | 实例 ID，不传则自动取 running → latest |
| metrics | array[string] | ✅ | 待查询的指标 key 列表，如 `["actor/ppo_kl"]`；即使只有一个指标也必须传数组，严禁传字符串 |
| latest_n | integer | ❌ | 仅取最近 N 个 step |
| step_start | integer | ❌ | step 起点（闭区间） |
| step_end | integer | ❌ | step 终点（闭区间） |

> ⚠️ 不支持 `wsid` 参数；已明确指标名时不要先调用 `list_hunyuan_train_tf_events_metrics`。

---

## query_hunyuan_train_tf_events_chart

生成模版化训练任务的 tf_events 指标趋势图。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_tf_events_chart '{"task_id": "finetuning_xxx", "metrics": ["loss"]}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | string | ✅ | 任务 ID（`finetuning_` 开头） |
| instance_id | string | ❌ | 实例 ID，不传则自动取最新实例 |
| metrics | array[string] | ❌ | 待画图的指标 key 列表，不传时自动选择 loss 相关或前 3 条指标；已知一个指标时也传数组，如 `["train/train_loss"]` |
| latest_n | integer | ❌ | 仅取最近 N 个 step |
| step_start | integer | ❌ | step 起点（闭区间） |
| step_end | integer | ❌ | step 终点（闭区间） |

> ⚠️ 不支持 `wsid` 参数；不要先用字符串调用失败后再重试数组，直接按数组传参。

---

## query_hunyuan_train_swanlab_metrics

查询 SwanLab 训练指标数据，支持多种数据获取模式。

> 💡 **适用场景**：用户提到 SwanLab 自定义上报的指标（如 lm loss / learning-rate 等非平台预定义指标），或用户明确说"SwanLab 指标"。
>
> ⚠️ **实例限制**：本工具**仅支持查询任务最新实例**的 SwanLab 指标，无法指定历史实例。不支持 `instance_id` 参数（后续将支持按实例 ID 查询）。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_swanlab_metrics '{"task_id": "basic_train_xxx", "keys": "loss,accuracy"}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | string | ✅ | 任务 ID（`basic_train_` 开头） |
| keys | string | ✅ | 指标名（逗号分隔，如 `"loss,accuracy"`） |
| sample | integer | ❌ | 采样数量，默认 1500，最大 1500；用户明确"采样 N 个点"时才传 |
| all | boolean | ❌ | 是否返回全量数据（不受采样限制） |
| step_start | integer | ❌ | 按 step 过滤起点（闭区间） |
| step_end | integer | ❌ | 按 step 过滤终点（闭区间） |
| ts_start | integer | ❌ | 按时间戳过滤起点（UNIX 毫秒） |
| ts_end | integer | ❌ | 按时间戳过滤终点（UNIX 毫秒） |
| last | integer | ❌ | 最近 N 毫秒的数据（与 step/ts 范围互斥） |
| head | integer | ❌ | 取前 N 个数据点（与 tail 互斥） |
| tail | integer | ❌ | 取后 N 个数据点（与 head 互斥）；用户未给窗口时默认传 `tail=100` |
| ignore_timestamp | boolean | ❌ | 是否去除时间戳字段 |

> ⚠️ 不支持 `wsid`、`mode` 参数。用户明确说 SwanLab/lm loss/learning-rate 且给出 task_id 时，直接调用本工具，不要先查 task_detail、实例列表或平台预定义指标。
