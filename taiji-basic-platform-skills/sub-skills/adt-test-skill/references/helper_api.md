## get_taiji_eval_task_detail

**来源模块**：`evaluation-skill`

**使用边界**：工程链路评估状态查询专用。评测 `task_id` 为**整数**（区别于 ADT 的字符串 `task_id`）。

查询指定评估任务的详情与评估报告，包括任务状态、模型信息、完成进度、评测集信息以及评估结果（`abilityInsights` insight 得分）等。用户查看评估报告或 insight 得分时应调用此工具。

> 💡 如果需要查看各评测集的逐项得分、完成数量、预测数量等进展信息，请使用 `get_taiji_eval_exercise_results`。
> 💡 对于**服务复用 Provider 任务**（`serviceReuseType: "SERVICE_REUSE_PROVIDER"`），本接口会自动查询并挂载其子任务列表到返回对象的 `consumer_tasks` 字段中。Provider 任务本身不执行评测，真正的评测数据在其子任务（Consumer）上。


**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_task_detail '{"task_id": 8828}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | int | ✅ | 评估任务 ID，例如 `8828` |
| score | string | 否 | 得分计算模式，默认 `default` |


**返回（成功）：**
```json
{
  "task_id": 8828,
  "task": {
    "id": 8828,
    "arenaId": 33,
    "name": "hy3.0_eval_task",
    "desc": "混元3.0评估任务",
    "status": "FINISHED",
    "servingStatus": null,
    "modelSource": "SERVING",
    "modelName": "hunyuan-3.0",
    "creator": "zhangsan",
    "updater": "zhangsan",
    "admin": "zhangsan,lisi",
    "completedNum": 100,
    "totalNum": 100,
    "errMsg": null,
    "createTime": "2026-03-09 15:15:16",
    "updateTime": "2026-03-09 15:35:39",
    "taskType": "HY_ONE_STOP_SERVICE",
    "arenaName": "代码能力评测v1",
    "collectionInfos": [
      {
        "collectionId": 10,
        "collectionVersionId": 379,
        "name": "autocodebench_v2",
        "description": "代码评测集"
      }
    ],
    "abilityInsights": []
  }
}
```

**Provider 任务额外返回字段（仅当任务为 `SERVICE_REUSE_PROVIDER` 时出现，位于顶层返回对象）：**
```json
{
  "consumer_tasks": [
    {
      "id": 102413,
      "name": "a20b_0331_copy",
      "status": "PARSED",
      "serviceReuseType": "SERVICE_REUSE_CONSUMER",
      "reuseProviderTaskId": 102412,
      "completedNum": 1,
      "totalNum": 1
    }
  ],
  "consumer_count": 1,
  "_note": "该任务为服务复用 Provider，以上为子任务（Consumer）列表"
}
```

| 额外字段 | 类型 | 描述 |
|------|------|------|
| consumer_tasks | array | 子任务（Consumer）列表，每个元素为精简版 task 对象 |
| consumer_count | int | 子任务数量 |
| _note | string | 提示信息，说明该任务为 Provider |

**返回（认证失败）：**
```json
{
  "error": "API 请求失败 (HTTP 401): Unauthorized",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

**返回（参数缺失）：**
```json
{
  "error": "task_id 不能为空，请提供评估任务 ID"
}
```

**返回字段说明：**
| 字段 | 类型 | 描述 |
|------|------|------|
| task_id | number | 查询使用的评估任务 ID |
| task | object | 评估任务详细信息（已裁剪大字段） |
| task.id | number | 评估任务主键 ID |
| task.arenaId | number | 所属评测版本 ID |
| task.name | string | 任务名称 |
| task.desc | string | 任务描述 |
| task.status | string | 任务状态（CREATED / RUNNING / FINISHED / FAILED 等） |
| task.servingStatus | string/null | 推理服务状态 |
| task.modelSource | string | 模型来源类型 |
| task.modelName | string | 模型名称 |
| task.creator | string | 创建人 |
| task.updater | string | 最后更新人 |
| task.admin | string | 管理员列表（逗号分隔） |
| task.completedNum | number | 已完成评测样本数 |
| task.totalNum | number | 总评测样本数 |
| task.errMsg | string/null | 错误信息（任务失败时返回） |
| task.createTime | string | 创建时间，格式 `YYYY-MM-DD HH:mm:ss` |
| task.updateTime | string | 最后更新时间 |
| task.taskType | string | 任务类型 |
| task.arenaName | string | 所属评测版本名称 |
| task.collectionInfos | array | 关联的评测集信息列表 |
| task.collectionInfos[].collectionId | number | 评测集 ID |
| task.collectionInfos[].collectionVersionId | number | 评测集版本 ID |
| task.collectionInfos[].name | string | 评测集名称 |
| task.collectionInfos[].description | string | 评测集描述 |
| task.abilityInsights | array | **评估结果（核心字段）**：insight 得分列表，包含各维度的评测得分与分析，用户查询评测结果时应重点关注此字段 |

---

## clone_taiji_eval_task

**来源模块**：`evaluation-skill`

**使用边界**：工程链路评估第 2 步专用。

复制已有评测任务，必传 `source_task_id` 和 `name`，不传的可选参数自动继承源任务值。支持所有模型来源类型。

> ⚠️ **异步操作**：提交后立即返回新任务 ID，任务在后台异步执行。

---

### Step 1：查源任务（前置必做，2 项全查）

调用 clone 前，必须先查源任务详情，确认以下 2 项：

```bash
# 一次查询提取 task_type / model_source
python3 scripts/connect_mcp.py call get_taiji_eval_task_detail '{"task_id": 103546}' 2>&1 | grep -E '"task_type":|"model_source":'
# → "task_type": "HY_ONE_STOP_SERVICE"
# → "model_source": "DISTILLATION_API"
```

| 检查项 | 查什么 | 不通过时 |
|--------|--------|---------|
| **task_type** | 必须是 `HY_ONE_STOP_SERVICE` | 告知"该任务类型为 XXX，不能用于 clone 评估任务。" |
| **model_source** | 决定哪些参数可以覆盖 | 按 modelSource 选可传参数（见下方参数表） |

> ℹ️ **后端不校验源任务 status**：任何状态（RUNNING / FAILED / STOP / PARSED 等）的任务均可 clone，后端不做状态拦截。但部分状态（如 FAILED）的源任务配置可能不完整，导致复制后的任务运行失败，请仔细确认参数。

> 🔴 `model_source` ≠ `task_type`，clone 参数限制只看 `model_source`。
> 🔴 **严禁拿 EXERCISE_VALIDATION 验证任务当源任务来 clone 评估任务**，反之亦然。

拿到 modelSource 后，按下面的参数表选择可传参数。

---

### 参数（按 modelSource 分组）

| 分类 | 参数 | 类型 | 必填 | modelSource | 说明 |
|------|------|------|------|-------------|------|
| 部署参数 | model_ids | array\<int\> | 否 | `MODEL_REPOSITORY` / `MODEL_GROUP` | 模型 ID 列表，不传沿用源任务 |
| | max_context_length | int | 否 | `MODEL_REPOSITORY` / `MODEL_GROUP` | 上下文长度 |
| | replicas | int | 否 | `MODEL_REPOSITORY` / `MODEL_GROUP` | 推理副本数 |
| | host_gpu_num | int | 否 | `MODEL_REPOSITORY` / `MODEL_GROUP` | 单机 GPU 数 |
| | gpu_name | string | 否 | `MODEL_REPOSITORY` / `MODEL_GROUP` | GPU 型号，precheck 会校验应用组资源是否可用 |
| | queue_name | string | 否 | `MODEL_REPOSITORY` / `MODEL_GROUP` | 应用组名，precheck 会校验应用组资源是否可用 |
| | location | string | 否 | `MODEL_REPOSITORY` / `MODEL_GROUP` | 地域，precheck 会校验应用组资源是否可用 |
| | resource_types | array\<string\> | 否 | `MODEL_REPOSITORY` / `MODEL_GROUP` | 资源类型列表：`["private","public","elastic"]` |
| 推理参数 | image | string | 否 | `MODEL_REPOSITORY` | 推理镜像地址 |
| | max_batch_size | int | 否 | `MODEL_REPOSITORY` | 最大批处理大小 |
| | compression_strategy | string | 否 | `MODEL_REPOSITORY` | 压缩策略，🔴 取值必须模型已支持，否则部署失败 |
| | envs | string | 否 | `MODEL_REPOSITORY` | 环境变量，格式 `KEY1=VAL1\nKEY2=VAL2` |
| | reasoning_parser | string | 否 | `MODEL_REPOSITORY` | 推理过程解析器 |
| | tool_parser | string | 否 | `MODEL_REPOSITORY` | 工具调用解析器 |
| 服务复用 | service_reuse_type | string | 否 | `MODEL_REPOSITORY` | 枚举：`NORMAL`/`SERVICE_REUSE_PROVIDER`/`SERVICE_REUSE_CONSUMER`，不传自动继承 |
| | reuse_provider_task_id | int | 否 | `MODEL_REPOSITORY` | `CONSUMER` 时必传，指定被复用源任务 |
| 服务参数 | service_name | string | 否 | `ONE_STOP_SERVICE` / `DISTILLATION_API` | 一站式服务名 |
| | crawl_para_num | int | 否 | `ONE_STOP_SERVICE` / `DISTILLATION_API` | 并发数 |
| 通用参数 | source_task_id | int | ✅ | 所有类型 | 源任务 ID |
| | name | string | ✅ | 所有类型 | 新任务名称 |
| | parameter_configuration | object | 否 | 所有类型 | 模型调用参数（temperature/top_p 等），deep merge 到源任务 |
| | hy_api_protocol | string | 否 | 所有类型 | 调用协议，如 `"openai"` |
| | hy_api_env | string | 否 | 所有类型 | API 环境，如 `"dev"`、`"online"` |
| | collection_version_ids | string | 否 | 所有类型 | 评测集合版本 ID，逗号分隔 |
| | enable_traj | bool | 否 | 所有类型 | 是否开启 Langfuse 轨迹追踪。不传则继承源任务的开关状态 |
| | traj_project_name | string | 否 | 所有类型 | Langfuse 项目名，`enable_traj=true` 时生效。不传则继承源任务的项目名 |

> 🔴 **parameter_configuration 只能放模型调用参数**（temperature / top_p 等），推理/部署参数必须作为顶层字段传。`{"max_batch_size": 128}` 放在 parameter_configuration 里不会生效。
>
> 💡 如果用户想改 `ONE_STOP_SERVICE` 任务的模型或推理配置，这类配置在 service 侧，clone 时不支持直接改，应引导用户先去修改原 service 配置。
>
> **parameter_configuration 示例**——用户说「temperature 改成 0.7，top_p 改成 0.9」：
> ```json
> {"parameter_configuration": {"temperature": 0.7, "top_p": 0.9}}
> ```
> 只需传入要改的字段，未传入的保持源任务原值。
>
> 💡 **enable_traj / traj_project_name**：这两个字段存在源任务 `extra_info` 顶层（不受 `model_source` 类型限制，所有类型均生效）。用户没提到轨迹追踪时不用管，Agent 无需追问，克隆任务会自动继承源任务原有的开关状态和项目名。只有用户明确要求"关闭/开启轨迹追踪"或"换一个 Langfuse 项目"时才传这两个字段。

---

### MCP 调用示例

```bash
# 最简克隆
python3 scripts/connect_mcp.py call clone_taiji_eval_task '{"source_task_id": 8828, "name": "hunyuan-3.0-eval-copy"}'

# 服务复用
python3 scripts/connect_mcp.py call clone_taiji_eval_task '{"source_task_id": 8828, "name": "hunyuan-3.0-eval-consumer", "service_reuse_type": "SERVICE_REUSE_CONSUMER", "reuse_provider_task_id": 5678}'

# 克隆时开启轨迹追踪并指定 Langfuse 项目
python3 scripts/connect_mcp.py call clone_taiji_eval_task '{"source_task_id": 8828, "name": "hunyuan-3.0-eval-traj", "enable_traj": true, "traj_project_name": "hy3.0测试"}'

# 克隆时显式关闭轨迹追踪（即使源任务开着）
python3 scripts/connect_mcp.py call clone_taiji_eval_task '{"source_task_id": 8828, "name": "hunyuan-3.0-eval-no-traj", "enable_traj": false}'
```

---

### 返回

```json
{
  "source_task_id": 8828,
  "warning": null,
  "new_task": {
    "id": 9001,
    "name": "hunyuan-3.0-eval-copy",
    "status": "CREATED",
    "modelSource": "MODEL_REPOSITORY",
    "modelName": "hunyuan-3.0"
  }
}
```

| 关键字段 | 描述 |
|------|------|
| source_task_id | 源任务 ID |
| warning | 被丢弃的参数提示（如 `"以下部署参数不支持，已被丢弃: [model_ids]"`），务必检查 |
| new_task.id | 新任务 ID |
| new_task.status | 新任务状态（`CREATED`） |

---

### Step 2：clone 后确认新任务状态

clone 返回 `new_task.id` 后，用 `get_taiji_eval_task_detail` 确认创建成功。完整输出很长，建议 grep 提取关键字段：

```bash
python3 scripts/connect_mcp.py call get_taiji_eval_task_detail '{"task_id": 9001}' 2>&1 | grep -E '"status":|"id":|"name":|"model_source":'
```
