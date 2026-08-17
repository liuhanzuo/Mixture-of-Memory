## get_taiji_eval_task_detail

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

## create_taiji_eval_arena

新建评估版本（Arena）。同 `arenaType` 下名称需唯一。

#### 参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|:---:|------|
| `name` | string | ✅ | 评估版本名称（同 arenaType 下唯一） |
| `branch` | string | ✅ | 代码分支名，如 `feature/lpf_offline2` |
| `arena_type` | string | ❌ | 评测版本类型，默认 `text_to_text`。可选值见下表 |
| `desc` | string | ❌ | 版本描述，默认空字符串 |
| `commit` | string | ❌ | 代码 commit，默认 `latest`，可填具体分支名或 commit hash |

#### arena_type 可选值速查

| `arena_type` 值 | 中文含义 | 用户可能怎么说 |
|---|---|---|
| `text_to_text` | 文生文 | "文本评测""文生文"（**默认值，不用传**） |
| `image_to_text` | 图生文 | "图片评测""图生文""多模态" |
| `AI-search` | AI 搜索 | "搜索评测""AI search" |
| `ASR` | 语音识别 | "语音识别""ASR" |
| `audio_understanding` | 音频理解 | "音频理解""音频评测" |

> ⚙️ **后端自动填充**（用户/Agent **不要传**）：`admin`（PAT Token 持有者用户名，如 `lamicyang`）、`repoName=taiji/hy/hy_unify_eval`、`visibility=CURRENT_WORKSPACE`、`visibleWsIds=""`

#### ⚠️ Agent 行为规则

1. **必问** `name` 和 `branch`，缺一不可。
2. `desc` 和 `commit` 用户未提则用默认值，不主动追问。
3. **`arena_type`**：创建时 Agent 须**主动向用户展示全部可选值**（见上表），并说明"默认为 `text_to_text`（文生文）"。用户未明确选择则用默认值 `text_to_text`。
4. **不要**询问 admin、repoName、visibility、visibleWsIds——后端自动填充。
5. 创建失败若返回 `arena name already exist,please change name` → 告知用户重名，建议换名后重试。

#### 调用示例

**默认（文生文）：**
```bash
python3 scripts/connect_mcp.py call create_taiji_eval_arena '{
  "name": "lamictest2",
  "branch": "feature/lpf_offline2",
  "desc": "测试版本",
  "commit": "latest"
}'
```

**指定图生文：**
```bash
python3 scripts/connect_mcp.py call create_taiji_eval_arena '{
  "name": "image-eval-test",
  "branch": "feature/lpf_offline2",
  "arena_type": "image_to_text"
}'
```

#### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | number | 新建的 Arena ID |
| `name` | string | 评估版本名称 |
| `arena_type` | string | 评测版本类型 |
| `repo_name` | string | 代码仓库名 |
| `branch` | string | 代码分支 |
| `commit` | string | 代码 commit |
| `creator` | string | 创建人（PAT Token 持有者） |
| `admin` | string | 负责人（同 creator） |
| `visibility` | string | 可见范围 |
| `create_time` | string | 创建时间 |

---

## list_taiji_eval_arenas

查询评测版本列表，可按类型和关键词搜索。


**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call list_taiji_eval_arenas '{"arena_type": "text_to_text", "keyword": "代码"}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| arena_type | string | 否 | 评测版本类型，默认为 `text_to_text`，不传则使用默认值 |
| keyword | string | 否 | 搜索关键词，用于模糊匹配评测版本名称 |
| my_arena | boolean | 否 | 是否仅返回当前用户创建的 Arena |
| only_favorites | boolean | 否 | 是否仅返回收藏的 Arena |
| page_index | int | 否 | 页码（从 1 开始），不传则使用后端默认值（1） |
| page_size | int | 否 | 每页数量，不传则使用后端默认值（10） |
| order_by | string | 否 | 排序字段，默认按 id 降序 |

**返回（成功）：**
```json
{
  "arena_type": "text_to_text",
  "keyword": null,
  "total": 10,
  "arena_count": 10,
  "arenas": [
    {
      "id": 123,
      "arenaName": "代码能力评测",
      "arenaType": "text_to_text",
      "...": "后端返回的其他字段将原样透传"
    }
  ]
}
```

**返回字段说明：**
| 字段 | 类型 | 描述 |
|------|------|------|
| arena_type | string | 查询使用的评测版本类型 |
| keyword | string/null | 查询使用的关键词 |
| total | number | 后端返回的总记录数 |
| arena_count | number | 当前返回的评测版本数量 |
| arenas | array | 评测版本列表（后端返回的原始数据） |

---

## list_taiji_eval_tasks

查询评测任务列表，支持按评测版本、关键词、创建人等条件搜索。不指定 arena_id 时可跨评测版本全局搜索。


> ⚠️ **⭐ 用户未指定评测版本时，不要遍历所有 arena_id，直接不传 arena_id 即可全局搜索。**

> ⚠️ **⭐ 当用户提到创建人/创建者时，必须将创建人名字传入 `creator` 参数（精确匹配），而非 `keyword` 参数。** `keyword` 用于模糊匹配任务名称/创建人/模型名称，`creator` 用于精确匹配创建人 RTX。

**MCP 工具调用：**
```bash
# 按评测版本查询
python3 scripts/connect_mcp.py call list_taiji_eval_tasks '{"arena_id": 123, "keyword": "hunyuan"}'

# 全局搜索（不指定评测版本）
python3 scripts/connect_mcp.py call list_taiji_eval_tasks '{"keyword": "hunyuan"}'

# 按创建人搜索
python3 scripts/connect_mcp.py call list_taiji_eval_tasks '{"creator": "kathy"}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| arena_id | int | 否 | 评测版本 ID（整数），通过 `list_taiji_eval_arenas` 获取。**不传则跨所有评测版本全局搜索** |
| keyword | string | 否 | 搜索关键词，用于模糊匹配任务名称、创建人、模型名称 |
| creator | string | 否 | 创建人 RTX 名称（**精确匹配**），用于筛选指定用户创建的任务。当用户提到"某人创建的任务"时应使用此参数 |
| hy_job_id | string | 否 | 训练实例 ID（hyJobId），传入后按训练实例筛选关联的评测任务 |
| status | string | 否 | 任务状态筛选：PENDING / RUNNING / PARSED / FAILED / STOPPED / RELEASED / CANCELED / STOPPING / RELEASE_RESOURCE / BATCH_API_SUBMITTING / BATCH_API_RUNNING / BATCH_API_MODEL_COPY / BATCH_API_QUEUING / BATCH_API_RESOURCE_RELEASED |
| task_type | string | 否 | 任务类型筛选：HY_ONE_STOP_SERVICE / ACCOMPANYE_EVAL / EXERCISE_VALIDATION / OFFLINE_INFERENCE |
| my_task | boolean | 否 | `true` = 仅返回当前用户有权限的任务；用户说"我的/我创建的"时传 `true` |
| model_group_id | int | 否 | 模型组 ID，按指定模型组过滤 |
| baseline | boolean | 否 | 是否仅返回基线任务 |
| from_insight | boolean | 否 | 是否仅返回来源于 Insight 看板的任务 |
| model_id | int | 否 | 模型 ID，按指定模型过滤 |
| baseline_first | boolean | 否 | 是否将基线任务排在最前，默认 false |
| arena_official_first | boolean | 否 | 是否将官方 Arena 排在最前，默认 false |
| page_index | int | 否 | 页码（从 1 开始），不传则使用后端默认值（1） |
| page_size | int | 否 | 每页数量，默认 10 |
| order_by | string | 否 | 排序字段，默认按 id 降序 |

> ⚠️ **分页提示（重要）**：当用户未指定 `page_index` / `page_size` 时，后端默认只返回一页，列表可能不完整。如果返回 `total > task_count`，必须在结果末尾提示：「⚠️ 只展示了前 X 条（共 Y 条）。如需查看更多或指定页数，请告诉我。」

**返回（成功）：**
```json
{
  "arena_id": 123,
  "keyword": null,
  "creator": null,
  "total": 5,
  "task_count": 5,
  "tasks": [
    {
      "taskId": 8828,
      "taskName": "hunyuan-code-eval",
      "...": "后端返回的其他字段将原样透传"
    }
  ]
}
```

**返回字段说明：**
| 字段 | 类型 | 描述 |
|------|------|------|
| arena_id | number/null | 查询使用的评测版本 ID（未指定时为 null） |
| keyword | string/null | 查询使用的关键词 |
| creator | string/null | 查询使用的创建人过滤条件 |
| total | number | 后端返回的总记录数 |
| task_count | number | 当前返回的任务数量 |
| tasks | array | 评测任务列表（后端返回的原始数据） |

> 💡 **提示**：获取到 `taskId` 后，可以调用 `get_taiji_eval_task_detail` 查询任务详情和评估报告，或调用 `get_taiji_eval_exercise_results` 查询评估指标和进展。

> 💡 **典型使用场景**：
> - 用户说"查询 kathy 创建的评测任务" → 传 `creator: "kathy"`
> - 用户说"查询 xxx 评测版本下的任务" → 先通过 `list_taiji_eval_arenas` 获取 arena_id，再传 `arena_id`
> - 用户说"搜索 hunyuan 相关的评测任务" → 传 `keyword: "hunyuan"`（全局搜索）
> - 用户说"查询 kathy 在 xxx 评测版本下创建的任务" → 同时传 `arena_id` 和 `creator`

---

## get_taiji_eval_exercise_results

查询指定评估任务的评估指标与评估进展，返回各评测集（exercise）的名称、得分（scores）、完成数量、预测数量等详细信息。用户查看评估进展或各评测集指标得分时应调用此工具。

> 💡 对于**服务复用 Provider 任务**（`serviceReuseType: "SERVICE_REUSE_PROVIDER"`），Provider 本身不执行评测，本接口会自动查询其所有子任务（Consumer）的评测进展并汇总返回。返回结果中会包含 `provider_info` 字段说明这一行为。


**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_exercise_results '{"task_id": 8828}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | int | ✅ | 评估任务 ID，例如 `8828` |
| keyword | string | 否 | 搜索关键词，用于模糊匹配评测集名称 |
| exercise_version_id | int | 否 | 按评测集版本 ID 筛选 |
| insight_id | int | 否 | 关联的 Insight ID，用于过滤 |
| page_index | int | 否 | 页码（从 1 开始），不传则使用后端默认值（1） |
| page_size | int | 否 | 每页数量，默认 10 |
| order_by | string | 否 | 排序字段，默认按 id 降序 |


**返回（成功）：**
```json
{
  "task_id": 8828,
  "keyword": null,
  "total": 5,
  "exercise_count": 5,
  "exercises": [
    {
      "id": 1001,
      "taskId": 8828,
      "exerciseVersionId": 379,
      "exerciseNameAndEvName": "autocodebench_v2/v1.0",
      "exerciseDesc": "代码评测集",
      "scores": "{\"pass@1\": 0.85, \"accuracy\": 0.92}",
      "totalCount": 500,
      "completedCount": 500,
      "predictedCount": 500,
      "createTime": "2026-03-09 15:15:16",
      "updateTime": "2026-03-09 15:35:39"
    }
  ]
}
```

**Provider 任务额外返回字段（仅当任务为 `SERVICE_REUSE_PROVIDER` 时出现）：**
```json
{
  "provider_info": {
    "provider_task_id": 63507,
    "provider_task_name": "A20B-High-260609-V5-step220",
    "note": "该任务为服务复用 Provider，仅负责任务部署，不执行评测。以下为子任务（Consumer）的评测进展。",
    "consumer_task_ids": [63508]
  }
}
```

| 额外字段 | 类型 | 描述 |
|------|------|------|
| provider_info | object | Provider 任务说明，含 consumer_task_ids 和提示；同一返回中的 `exercises` 为所有子任务结果汇总 |

**返回字段说明：**
| 字段 | 类型 | 描述 |
|------|------|------|
| task_id | number | 查询使用的评估任务 ID |
| keyword | string/null | 查询使用的关键词 |
| total | number | 后端返回的总记录数 |
| exercise_count | number | 当前返回的评测集数量 |
| exercises | array | 评测集结果列表 |
| exercises[].id | number | 评测集结果主键 ID |
| exercises[].taskId | number | 所属评估任务 ID |
| exercises[].exerciseVersionId | number | 评测集版本 ID |
| exercises[].exerciseNameAndEvName | string | 评测集名称/版本名称（如 `autocodebench_v2/v1.0`） |
| exercises[].exerciseDesc | string | 评测集描述 |
| exercises[].scores | string | **评估指标得分**（JSON 字符串），包含各指标的得分，如 `{"pass@1": 0.85, "accuracy": 0.92}` |
| exercises[].totalCount | number | 总样本数 |
| exercises[].completedCount | number | 已完成评测的样本数 |
| exercises[].predictedCount | number | 已预测的样本数 |
| exercises[].createTime | string | 创建时间，格式 `YYYY-MM-DD HH:mm:ss` |
| exercises[].updateTime | string | 更新时间 |

> 💡 **提示**：`scores` 字段是 JSON 字符串，需要解析后展示各指标得分。如需查看任务整体的评估报告和 insight，请使用 `get_taiji_eval_task_detail`。

---

## get_taiji_eval_exercise_summary

查询指定评测任务的评测集进展摘要（轻量版），返回各评测集的 exercise_version_id、名称、总数量、完成数量、预测数量。与 `get_taiji_eval_exercise_results` 的区别：本工具不返回 scores 等详细指标，仅返回进展概览，适合快速查看完成度。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_exercise_summary '{"task_id": 8828}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | int | ✅ | 评测任务 ID |
| page_index | int | 否 | 页码（从 1 开始），默认 1 |
| page_size | int | 否 | 每页数量，默认 100 |

---

## clone_taiji_eval_task

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
> 💡 **enable_traj / traj_project_name**：这两个字段存在源任务 `extra_info` 顶层（不受 `model_source` 类型限制，所有类型均生效）。用户没提到轨迹追踪时不用管，Agent 无需追问，克隆任务会自动继承源任务原有的开关状态和项目名。只有用户明确要求"关闭/开启轨迹追踪"或"换一个 Langfuse 项目"时才传这两个字段。若传了 `enable_traj=true` 但 `traj_project_name` 为空（且源任务也没有），后端会静默不开启追踪（不会报错阻断建任务），需提醒用户确认项目名是否正确。

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

---

## retry_taiji_eval_task

重试一个已有的评估任务。仅当任务处于失败（FAILED）或已完成（FINISHED）等非运行状态时才可重试，正在运行中的任务无法重试。重试后任务状态会重置为 PENDING，重新开始执行评测流程。

> ⚠️ **正在运行中（PENDING / RUNNING）的任务无法重试**，后端会返回错误。


> ⚠️ **⭐ 调用前必须向用户确认以下两个参数，严禁直接使用默认值调用：**
> 1. **重试哪些评测集**（`exercise_version_filter`）：是重试全部评测集，还是只重试部分？默认值为**全部重试**（不传该参数）。如果用户不确定有哪些评测集，应先调用 `get_taiji_eval_exercise_results` 查询后列出供用户选择。
> 2. **抓取并发数**（`crawl_para_num`）：设置评测的并发执行数量。默认值为**沿用原任务配置**（不传该参数）。
>
> 确认话术示例：「请确认：① 是重试全部评测集还是只重试部分？（默认全部重试）② 抓取并发数设为多少？（默认沿用原任务配置）」

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call retry_taiji_eval_task '{"task_id": 8828}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | int | ✅ | 评估任务 ID，例如 `8828` |
| exercise_version_filter | array\<int\> | 否 | 评测集版本 ID 过滤列表（整数数组），仅重试指定的评测集版本，不传则重试全部评测集。**exerciseVersionId 可通过 `get_taiji_eval_exercise_results` 工具获取**（返回的 `exercises[].exerciseVersionId` 字段）。例如 `[379, 380]` |
| crawl_para_num | int | 否 | 并发数（整数），设置评测的并发执行数量，不传则沿用原任务配置。例如 `5` |
| skip_infer_stage | boolean | 否 | 跳过抓取（推理）阶段 |
| skip_eval_stage | boolean | 否 | 跳过评估阶段 |
| bottom_eval_custom_params | object | 否 | bottom 专用超参，合并到评测超参中 |
| bottom_eval_button_params | object | 否 | "开始检测"按钮触发的参数，独立存储不与 custom_params 互相覆盖 |


**返回（成功）：**
```json
{
  "task_id": 8828,
  "task": {
    "id": 8828,
    "arenaId": 33,
    "name": "hy3.0_eval_task",
    "desc": "混元3.0评估任务",
    "status": "PENDING",
    "servingStatus": "initial",
    "modelSource": "MODEL_REPOSITORY",
    "modelName": "hunyuan-3.0",
    "creator": "zhangsan",
    "updater": "zhangsan",
    "admin": "zhangsan,lisi",
    "completedNum": 0,
    "totalNum": 100,
    "errMsg": null,
    "createTime": "2026-03-09 15:15:16",
    "updateTime": "2026-03-30 10:00:00",
    "taskType": "HY_ONE_STOP_SERVICE",
    "arenaName": "代码能力评测v1",
    "collectionInfos": [...],
    "abilityInsights": [],
    "offlineInferenceId": null
  }
}
```

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
| task_id | number | 重试的评估任务 ID |
| task | object | 重试后的评估任务详细信息（已裁剪大字段），字段含义与 `get_taiji_eval_task_detail` 返回的 `task` 一致 |

> 💡 **提示**：重试成功后，任务状态会变为 `PENDING`，可以通过 `get_taiji_eval_task_progress` 持续查看任务进度。

> 💡 **典型使用场景**：评测任务因网络超时、模型服务异常等原因失败后，用户希望重新执行评测时使用。可通过 `exercise_version_filter` 只重试部分失败的评测集，避免全量重跑。

> 💡 **如何获取 exerciseVersionId**：仅当用户要求“只重试部分评测集”但未给版本 ID 时，先调用 `get_taiji_eval_exercise_results` 查询该任务的评测集列表。若用户说“重试任务 X 然后停掉，不需要确认”，不要查询详情、不要克隆、不要重复重试：直接 `retry_taiji_eval_task` 一次，再对返回的新任务 ID（无返回则用原 task_id）调用 `stop_taiji_eval_task` 一次。

---

## submit_taiji_eval_insight_export

创建 Insight 数据导出（下载）任务。提交后会异步执行导出，可通过 `get_taiji_eval_insight_export_status` 查询进度。


> ⚠️ **⭐ 参数确认规则：**
> 1. **存储类型**（`storage`）：选 `cos` 可获得 COS 下载链接，方便直接下载到本地分析；选 `ceph` 文件存储在服务器上，需要自行到服务器获取。默认值为 `ceph`。
> 2. **导出格式**（`format`）：`csv` 或 `jsonl`，根据用户需要选择。默认值为 `jsonl`。
> 3. 用户已明确给出 `cos/ceph`、`csv/jsonl` 或说“不需要确认/直接执行”时，不要再追问；按用户参数执行，缺省项用接口默认值。
>
> 标准链路：可先 `list_taiji_eval_insight_exports` 查看历史，但只要用户要求“导出/下载 Insight 数据”，必须提交 `submit_taiji_eval_insight_export`，不能仅返回历史详情结束；提交后用 `get_taiji_eval_insight_export_status` 轮询/回显下载链接。

> 📋 **按评测集筛选导出时的操作流程**：
>
> 用户**未提及评测集筛选**时，直接全量导出，无需额外操作。仅当用户明确要求筛选特定评测集时，按以下步骤操作：
>
> 1. 先调用 `get_taiji_eval_insight_detail` 获取该 Insight 的 `weightNodes`
> 2. 从 `weightNodes` 中提取每个评测集信息，以**可读列表**展示给用户，**同时提供「全部」选项**：
>    ```
>    该 Insight 包含以下评测集：
>    0. 全部
>    1. T1-Long-260205#260205（版本ID: 279）
>    2. T2-Short-260210（版本ID: 544）
>    请选择要导出的评测集（可多选，输入序号即可）
>    ```
> 3. 用户选「全部」→ 不传 exercise_version_ids，全量导出；用户选部分 → 将对应的 `collectionVersionId` 作为 `exercise_version_ids` 传入（多个用逗号分隔）
> 4. 筛选时同时设置 `collection_id` 和 `collection_version_id` 为所选评测集对应的值

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call submit_taiji_eval_insight_export '{"insight_id": 123, "storage": "cos", "format": "jsonl"}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| insight_id | int | ✅ | Insight ID，例如 `123` |
| collection_id | int | 否 | 评测集 ID，不传则导出全部 |
| collection_version_id | int | 否 | 评测集版本 ID |
| type | string | 否 | 导出类型 |
| exercise_version_ids | array\<int\> | 否 | 评测集版本 ID 列表，用于精确筛选特定评测集版本的数据 |
| score | string | 否 | 得分筛选条件，用于过滤特定得分的数据 |
| storage | string | 否 | 存储类型：`cos` 或 `ceph`，默认 `ceph`。选 cos 可获得 COS 下载链接 |
| format | string | 否 | 导出格式：`csv` 或 `jsonl`，默认 `jsonl` |
| include_trajectory | boolean | 否 | 是否包含 agent 轨迹文件，默认 `true`。设为 `false` 可跳过轨迹以加快导出 |
| include_raw_trajectory | boolean | 否 | 是否包含原始执行轨迹 `trajectories.tar`（.jsonl 格式，可能数 GB/评测集），默认 `false`。**体积极大，会显著增加导出耗时**，只有用户明确需要原始轨迹时才开启 |


**返回（成功）：**
```json
{
  "insight_id": 123,
  "export_task": {
    "exportTaskId": 456,
    "status": "PENDING"
  },
  "hint": "导出任务已提交，请使用 get_taiji_eval_insight_export_status 工具查询任务进度。"
}
```

**返回（参数缺失）：**
```json
{
  "error": "insight_id 不能为空，请提供 Insight ID"
}
```

**返回字段说明：**
| 字段 | 类型 | 描述 |
|------|------|------|
| insight_id | number | 提交的 Insight ID |
| export_task | object | 导出任务信息，包含 exportTaskId 和 status |
| export_task.exportTaskId | number | 导出任务 ID，用于后续查询状态 |
| export_task.status | string | 任务初始状态（通常为 PENDING） |
| hint | string | 操作提示 |

> 💡 **提示**：提交成功后，使用 `get_taiji_eval_insight_export_status` 传入 `exportTaskId` 查询导出进度和下载链接。

---

## get_taiji_eval_insight_export_status

查看导出（下载）任务的状态和进度，适用于 Insight 导出和 Task Case 导出（通用）。

状态说明：`PENDING`=排队中，`RUNNING`=导出中，`SUCCESS`=导出完成，`FAILED`=导出失败。


**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_insight_export_status '{"export_task_id": 456}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| export_task_id | int | ✅ | 导出任务 ID，通过 `submit_taiji_eval_insight_export` 或 `submit_taiji_eval_case_export` 获取 |


**返回（导出完成 - COS 存储）：**
```json
{
  "export_task_id": 456,
  "task": {
    "id": 456,
    "insightId": 123,
    "type": "insight",
    "status": "SUCCESS",
    "exportedCount": 1000,
    "totalCount": 1000,
    "progress": 100.0,
    "fileSize": 5242880,
    "fileSizeDisplay": "5.0 MB",
    "storage": "cos",
    "format": "jsonl",
    "cosUrl": "https://cos.example.com/export/data.jsonl.gz",
    "errMsg": null,
    "creator": "yvesren",
    "createTime": "2026-04-03 10:00:00",
    "updateTime": "2026-04-03 10:05:00"
  },
  "download_hint": "✅ 导出完成！COS 下载链接：https://cos.example.com/export/data.jsonl.gz，可以下载到本地进行分析。"
}
```

**返回（导出完成 - Ceph 存储）：**
```json
{
  "export_task_id": 456,
  "task": {
    "id": 456,
    "status": "SUCCESS",
    "storage": "ceph",
    "cosUrl": null
  },
  "download_hint": "✅ 导出完成！文件存储在 ceph 路径：/data/export/xxx.jsonl，ceph 文件需要到服务器上获取分析。"
}
```

**返回（导出中）：**
```json
{
  "export_task_id": 456,
  "task": {
    "id": 456,
    "status": "RUNNING",
    "exportedCount": 500,
    "totalCount": 1000,
    "progress": 50.0
  }
}
```

**返回字段说明：**
| 字段 | 类型 | 描述 |
|------|------|------|
| export_task_id | number | 查询的导出任务 ID |
| task | object | 导出任务详细信息 |
| task.id | number | 导出任务主键 ID |
| task.insightId | number/null | 关联的 Insight ID（insight 类型时有值） |
| task.taskId | number/null | 关联的评测任务 ID（case 类型时有值） |
| task.type | string | 导出类型：`insight` 或 `case` |
| task.status | string | 任务状态：PENDING / RUNNING / SUCCESS / FAILED |
| task.exportedCount | number | 已导出的数据条数 |
| task.totalCount | number | 总数据条数 |
| task.progress | number | 导出进度百分比（0-100） |
| task.fileSize | number/null | 文件大小（字节） |
| task.fileSizeDisplay | string/null | 文件大小（可读格式，如 "5.0 MB"） |
| task.storage | string | 存储类型：cos 或 ceph |
| task.format | string | 导出格式：csv 或 jsonl |
| task.cosUrl | string/null | COS 下载链接（storage=cos 且导出成功时有值） |
| task.errMsg | string/null | 错误信息（导出失败时有值） |
| task.creator | string | 创建人 |
| task.createTime | string | 创建时间 |
| task.updateTime | string | 更新时间 |
| download_hint | string | 下载提示信息（仅导出成功时返回） |

> 💡 **提示**：如果 `status=SUCCESS` 且 `cosUrl` 不为空，可以询问用户是否需要下载到本地进行分析。如果 `storage=ceph`，提示用户需要到服务器上获取文件。

---

## list_taiji_eval_insight_exports

查看 Insight 数据导出（下载）历史列表。返回 Insight 类型的导出任务列表。


**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call list_taiji_eval_insight_exports '{"insight_id": 123}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| insight_id | int | 否 | 按 Insight ID 过滤 |
| status | string | 否 | 按状态过滤：PENDING / RUNNING / SUCCESS / FAILED |
| creator | string | 否 | 按创建人过滤 |
| task_id | int | 否 | 按任务 ID 过滤 |
| type | string | 否 | 按导出类型过滤（如 `insight`） |
| keyword | string | 否 | 搜索关键词 |
| page_index | int | 否 | 页码（从 1 开始），不传则使用后端默认值（1） |
| page_size | int | 否 | 每页数量，默认 10 |
| order_by | string | 否 | 排序字段，默认按 id 降序 |


**返回（成功）：**
```json
{
  "type": "insight",
  "insight_id": 123,
  "total": 3,
  "count": 3,
  "tasks": [
    {
      "id": 456,
      "insightId": 123,
      "type": "insight",
      "status": "SUCCESS",
      "exportedCount": 1000,
      "totalCount": 1000,
      "progress": 100.0,
      "storage": "cos",
      "format": "jsonl",
      "cosUrl": "https://cos.example.com/export/data.jsonl.gz",
      "creator": "yvesren",
      "createTime": "2026-04-03 10:00:00",
      "download_hint": "✅ 导出完成！COS 下载链接：https://cos.example.com/export/data.jsonl.gz，可以下载到本地进行分析。"
    },
    {
      "id": 455,
      "insightId": 123,
      "type": "insight",
      "status": "FAILED",
      "errMsg": "导出超时",
      "creator": "yvesren",
      "createTime": "2026-04-02 10:00:00"
    }
  ]
}
```

**返回字段说明：**
| 字段 | 类型 | 描述 |
|------|------|------|
| type | string | 固定为 `insight` |
| insight_id | number/null | 查询使用的 Insight ID 过滤条件 |
| total | number | 后端返回的总记录数 |
| count | number | 当前返回的记录数量 |
| tasks | array | 导出任务列表，每条记录的字段与 `get_taiji_eval_insight_export_status` 返回的 `task` 一致 |
| tasks[].download_hint | string | 下载提示信息（仅 SUCCESS 状态的记录有此字段） |

> 💡 **提示**：列表中 SUCCESS 状态且有 cosUrl 的记录会附带 `download_hint`，可以直接展示给用户。

---

## submit_taiji_eval_case_export

创建评测任务 Case 数据导出（下载）任务。提交后会异步执行导出，可通过 `get_taiji_eval_insight_export_status` 查询进度。


> ⚠️ **⭐ 参数确认规则：**
> 1. **存储类型**（`storage`）：选 `cos` 可获得 COS 下载链接，方便直接下载到本地分析；选 `ceph` 文件存储在服务器上，需要自行到服务器获取。默认值为 `ceph`。
> 2. **导出格式**（`format`）：`csv` 或 `jsonl`，根据用户需要选择。默认值为 `jsonl`。
> 3. 用户已明确给出 `cos/ceph`、`csv/jsonl` 或说“不需要确认/直接执行”时，不要再追问；按用户参数执行，缺省项用接口默认值。
>
> 🚫 **导出/下载请求的强制提交规则**：用户要求“导出/下载 case/失败 case/评测明细/底线检测数据”时，必须调用 `submit_taiji_eval_case_export` 并使用返回的 `exportTaskId` 调 `get_taiji_eval_insight_export_status`。`list_taiji_eval_case_exports` 只用于用户明确问“导出历史/已有导出/查询下载记录”，或在提交失败后辅助排查；不要只 list 历史就结束。
>
> 调用 `submit_taiji_eval_case_export` **之前**，必须先调用 `task/query_case_exports(task_id=目标ID)` 查询近期导出历史。
>
> 如果返回列表中存在**同时满足以下所有条件**的记录，则**禁止重复提交**，直接将该已有记录返回给用户：
> - `taskId` 相同
> - `storage` 相同（未指定时视为 `ceph`）
> - `format` 相同（未指定时视为 `jsonl`）
> - `status` 为 `PENDING`、`RUNNING` 或 `SUCCESS`（即：非 FAILED）
>
> 命中去重时的回复模板：
> - 状态为 SUCCESS 且有 cosUrl：「已存在相同参数的导出任务（ID: {id}），导出已完成。COS 下载链接：{cosUrl}」
> - 状态为 PENDING/RUNNING：「已存在相同参数的导出任务（ID: {id}，状态: {status}），无需重复提交。可使用 get_taiji_eval_insight_export_status 查询进度。」
>
> 仅当列表为空或所有匹配记录状态均为 `FAILED` 时，才执行提交。

> 📋 **按评测集筛选导出时的操作流程**：
>
> 用户未提及评测集筛选时，直接全量导出。仅当用户明确要求筛选特定评测集时，先调用 `get_taiji_eval_exercise_results` 获取该任务的评测集列表，以可读列表展示给用户（名称 + exerciseVersionId，同时提供「全部」选项），用户选择「全部」则不传 exercise_version_ids，选择部分则传入对应的 exerciseVersionId。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call submit_taiji_eval_case_export '{"task_id": 8828, "storage": "cos", "format": "csv"}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | int | ✅ | 评测任务 ID，例如 `8828` |
| exercise_version_ids | array\<int\> | 否 | 评测集版本 ID 列表，用于精确筛选特定评测集版本的数据 |
| storage | string | 否 | 存储类型：`cos` 或 `ceph`，默认 `ceph`。选 cos 可获得 COS 下载链接 |
| format | string | 否 | 导出格式：`csv` 或 `jsonl`，默认 `jsonl` |
| include_trajectory | boolean | 否 | 是否包含 agent 轨迹文件，默认 `true`。设为 `false` 可跳过轨迹以加快导出 |
| include_raw_trajectory | boolean | 否 | 是否包含原始执行轨迹 `trajectories.tar`（.jsonl 格式，可能数 GB/评测集），默认 `false`。**体积极大，会显著增加导出耗时**，只有用户明确需要原始轨迹时才开启 |

> ⚠️ **去重限制**：后端以 `taskId + storage + format` 作为去重键，**不区分 `include_trajectory`**。若已有 SUCCESS 状态且相同 taskId/storage/format 的导出记录，即使切换 `include_trajectory` 也会返回已有记录而非创建新导出。需要不同轨迹配置时，需先删除旧导出任务，或改用不同的 `format`/`storage` 组合。

**返回（成功）：**
```json
{
  "task_id": 8828,
  "export_task": {
    "exportTaskId": 789,
    "status": "PENDING"
  },
  "hint": "导出任务已提交，请使用 get_taiji_eval_insight_export_status 工具查询任务进度。"
}
```

**返回（参数缺失）：**
```json
{
  "error": "task_id 不能为空，请提供评测任务 ID"
}
```

**返回字段说明：**
| 字段 | 类型 | 描述 |
|------|------|------|
| task_id | number | 提交的评测任务 ID |
| export_task | object | 导出任务信息，包含 exportTaskId 和 status |
| export_task.exportTaskId | number | 导出任务 ID，用于后续查询状态 |
| export_task.status | string | 任务初始状态（通常为 PENDING） |
| hint | string | 操作提示 |

> 💡 **提示**：提交成功后，使用 `get_taiji_eval_insight_export_status` 传入 `exportTaskId` 查询导出进度和下载链接。

---

## list_taiji_eval_case_exports

查看评测任务 Case 数据导出（下载）历史列表。返回 Case 类型的导出任务列表。


**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call list_taiji_eval_case_exports '{"task_id": 8828}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | int | 否 | 按评测任务 ID 过滤 |
| status | string | 否 | 按状态过滤：PENDING / RUNNING / SUCCESS / FAILED |
| creator | string | 否 | 按创建人过滤 |
| insight_id | int | 否 | 按 Insight 看板 ID 过滤 |
| type | string | 否 | 按导出类型过滤 |
| keyword | string | 否 | 搜索关键词 |
| page_index | int | 否 | 页码（从 1 开始），不传则使用后端默认值（1） |
| page_size | int | 否 | 每页数量，默认 10 |
| order_by | string | 否 | 排序字段，默认按 id 降序 |


**返回（成功）：**
```json
{
  "type": "case",
  "task_id": 8828,
  "total": 2,
  "count": 2,
  "tasks": [
    {
      "id": 789,
      "taskId": 8828,
      "type": "case",
      "status": "SUCCESS",
      "exportedCount": 500,
      "totalCount": 500,
      "progress": 100.0,
      "storage": "cos",
      "format": "csv",
      "cosUrl": "https://cos.example.com/export/cases.csv.gz",
      "creator": "yvesren",
      "createTime": "2026-04-03 10:00:00",
      "download_hint": "✅ 导出完成！COS 下载链接：https://cos.example.com/export/cases.csv.gz，可以下载到本地进行分析。"
    }
  ]
}
```

**返回字段说明：**
| 字段 | 类型 | 描述 |
|------|------|------|
| type | string | 固定为 `case` |
| task_id | number/null | 查询使用的评测任务 ID 过滤条件 |
| total | number | 后端返回的总记录数 |
| count | number | 当前返回的记录数量 |
| tasks | array | 导出任务列表，每条记录的字段与 `get_taiji_eval_insight_export_status` 返回的 `task` 一致 |
| tasks[].download_hint | string | 下载提示信息（仅 SUCCESS 状态的记录有此字段） |

> 💡 **提示**：列表中 SUCCESS 状态且有 cosUrl 的记录会附带 `download_hint`，可以直接展示给用户。

---

## stop_taiji_eval_task

停止一个正在运行的评估任务。仅任务负责人（创建人或管理员）可以执行此操作。

仅当任务处于运行中（PENDING / RUNNING）状态时才可停止，已完成或已失败的任务无法停止。停止后任务状态会变为 FAILED。


> ⚠️ **⭐ 调用前必须向用户二次确认**：停止评测任务会中断正在进行的评测流程，请确认用户确实要停止该任务。确认话术示例：「确认要停止评测任务 {task_id} 吗？停止后任务状态会变为 FAILED，如需重新评测需要使用重试功能。」

> ⚠️ **权限限制**：系统会自动校验当前用户是否为任务负责人（创建人或管理员列表中的成员），非负责人调用会返回权限不足的错误。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call stop_taiji_eval_task '{"task_id": 8828}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | int | ✅ | 评估任务 ID，例如 `8828` |
| stop_reason | string | 否 | 停止原因 |


**返回（成功）：**
```json
{
  "task_id": 8828,
  "message": "评测任务已停止",
  "task": {
    "id": 8828,
    "arenaId": 33,
    "name": "hy3.0_eval_task",
    "desc": "混元3.0评估任务",
    "status": "FAILED",
    "modelName": "hunyuan-3.0",
    "creator": "zhangsan",
    "admin": "zhangsan,lisi",
    "completedNum": 50,
    "totalNum": 100,
    "createTime": "2026-03-09 15:15:16",
    "updateTime": "2026-03-30 10:00:00"
  }
}
```

**返回（权限不足）：**
```json
{
  "error": "权限不足：当前用户 'wangwu' 不是该任务的负责人。任务创建人为 'zhangsan'，管理员为 'zhangsan,lisi'。只有任务负责人（创建人或管理员）才能执行此操作。"
}
```

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

**返回（成功，后端未返回 task 数据）：**
```json
{
  "task_id": 8828,
  "message": "停止成功",
  "note": "后端未返回任务详情，task 字段缺失。可通过 get_taiji_eval_task_detail 单独查询任务状态。"
}
```

**返回字段说明：**
| 字段 | 类型 | 描述 |
|------|------|------|
| task_id | number | 停止的评估任务 ID |
| message | string | 操作结果提示信息 |
| task | object | 停止后的评估任务详细信息（已裁剪大字段），字段含义与 `get_taiji_eval_task_detail` 返回的 `task` 一致。后端未返回任务数据时此字段不存在 |

> 💡 **提示**：停止成功后，任务状态会变为 `FAILED`。如需重新评测，可以使用 `retry_taiji_eval_task` 重试该任务。

> 💡 **典型使用场景**：评测任务因配置错误需要中止、或用户不再需要该评测结果时，使用本工具停止正在运行的任务。

---

## delete_taiji_eval_task

删除一个评估任务。仅任务负责人（创建人或管理员）可以执行此操作。

> ⚠️ **删除操作不可恢复，请谨慎使用。**

> ⚠️ **运行中的任务不可直接删除**：只有处于终态（PARSED / FAILED / CANCELED / STOP）的任务才能删除。如果任务仍在运行中（非终态），系统会拒绝删除并提示用户先调用 `stop_taiji_eval_task` 停止任务。


> ⚠️ **⭐ 调用前必须向用户二次确认**：删除操作不可恢复，必须明确告知用户风险并获得确认。确认话术示例：「⚠️ 删除评测任务 {task_id} 后将无法恢复，确认要删除吗？」

> ⚠️ **权限限制**：系统会自动校验当前用户是否为任务负责人（创建人或管理员列表中的成员），非负责人调用会返回权限不足的错误。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call delete_taiji_eval_task '{"task_id": 8828}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | int | ✅ | 评估任务 ID，例如 `8828` |


**返回（成功）：**
```json
{
  "task_id": 8828,
  "message": "评测任务已删除"
}
```

**返回（权限不足）：**
```json
{
  "error": "权限不足：当前用户 'wangwu' 不是该任务的负责人。任务创建人为 'zhangsan'，管理员为 'zhangsan,lisi'。只有任务负责人（创建人或管理员）才能执行此操作。"
}
```

**返回（认证失败）：**
```json
{
  "error": "API 请求失败 (HTTP 401): Unauthorized",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

**返回（任务运行中，无法删除）：**
```json
{
  "error": "无法删除：任务当前状态为 'RUNNING'，仍在运行中。请先使用 stop_taiji_eval_task 停止该任务，待任务状态变为终态（PARSED / FAILED / CANCELED / STOP）后再执行删除。",
  "task_id": 8828,
  "current_status": "RUNNING",
  "hint": "可调用 stop_taiji_eval_task 停止任务后再删除"
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
| task_id | number | 删除的评估任务 ID |
| message | string | 操作结果提示信息 |

> 💡 **提示**：删除操作不可恢复，删除后该任务的所有数据（包括评测结果、评测集得分等）将被永久清除。

> 💡 **典型使用场景**：用户需要清理不再需要的评测任务、或删除错误创建的任务时使用。如果任务还在运行中，需先调用 `stop_taiji_eval_task` 停止任务，再执行删除。建议先通过 `get_taiji_eval_task_detail` 确认任务信息后再执行删除。

---

## get_taiji_eval_task_link

根据评测任务 ID 获取评测任务的前端页面链接。通过查询任务详情，拼接出可直接在浏览器中打开的评测任务详情页链接。


**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_task_link '{"task_id": 101532}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | int | ✅ | 评测任务 ID，例如 `101532` |


**链接格式：**
```
{EVALUATION_FRONTEND_URL}/web/base_model_eval_task_detail_multi?wsId={wsId}&exerciseId={arenaId}&name={arenaName}&id={taskId}&tag=share&taskType={taskType}
```

**返回（成功）：**
```json
{
  "task_id": 101532,
  "task_name": "工程链路评估-hunyuan-standard",
  "status": "FINISHED",
  "link": "https://hy-b.woa.com/web/base_model_eval_task_detail_multi?wsId=10103&exerciseId=33&name=Bench-Prod&id=101532&tag=share&taskType=HY_ONE_STOP_SERVICE"
}
```

**返回（参数缺失）：**
```json
{
  "error": "task_id 不能为空，请提供评测任务 ID"
}
```

**返回（API 错误）：**
```json
{
  "error": "API 请求失败 (HTTP 401): Unauthorized",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

**返回字段说明：**
| 字段 | 类型 | 描述 |
|------|------|------|
| task_id | number | 查询的评测任务 ID |
| task_name | string | 评测任务名称 |
| status | string | 任务状态（CREATED / RUNNING / FINISHED / FAILED 等） |
| link | string | 评测任务前端页面链接，可直接在浏览器中打开 |

> 💡 **典型使用场景**：用户需要获取评测任务的 Web 页面链接时使用，例如需要分享给他人查看、或在浏览器中查看详细的评测报告和可视化结果。

> 💡 **链接参数说明**：链接中的参数从任务详情 API 自动获取：`wsId`(工作空间 ID)、`exerciseId`(评测版本 ID，即 arenaId)、`name`(评测版本名称，即 arenaName)、`id`(任务 ID)、`tag=share`(固定值)、`taskType`(任务类型)。

---

## get_taiji_eval_task_agent_detail

查询**评估任务（Task 维度）**的Agent评测详情，返回该任务下完整的Agent评测报告，包含元信息（meta）、任务信息（taskInfo）、汇总指标（summaryRows，如Pass@1、Pass@2、token消耗、tool_call_success_rate、ACC等）、按模型分组的明细（detailsByModel）、曲线（curves）、分布（distributions）以及告警（warnings）。

> ⚠️ **【硬性参数要求】调用前必须确认以下3个必填参数全部提供，缺少任何一个都请提示用户补充，禁止编造或使用默认参数值**：
> 1. `task_id`：评估任务ID（整数）
> 2. `collection_version_id`：评测集版本ID（整数）
> 3. `exercise_version_id`：benchmark（exercise）版本ID（整数）
>
> 提示话术示例：「请提供以下必填参数才能查询Agent评测详情：① 评估任务ID（task_id）② 评测集版本ID（collection_version_id）③ benchmark版本ID（exercise_version_id）」


**触发关键词**：当用户提到以下任意意图时，优先调用本工具：
- 查询xxx任务下的agent状态分析/agent详情/agent评估指标
- 查询xxx任务的agent评测详情
- 任务维度的agent评测结果/指标查询

**适用场景**：已知评估任务ID、以及该任务关联的评测集版本ID和benchmark版本ID，需要查看该Agent任务的完整评测报告。

**MCP工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_task_agent_detail '{"task_id": 49276, "collection_version_id": 1911, "exercise_version_id": 3386}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | int | ✅ | 评估任务ID，整数类型，例如 `49276` |
| collection_version_id | int | ✅ | 评测集版本ID，整数类型，例如 `1911` |
| exercise_version_id | int | ✅ | benchmark（exercise）版本ID，整数类型，例如 `3386` |


**返回（成功）：**
原样返回后端data字段，包含meta、taskInfo、summaryRows、detailsByModel、curves、distributions、warnings等完整内容。

**返回（参数缺失）：**
```json
{
  "error": "task_id 不能为空，请提供评估任务 ID"
}
```
或对应collection_version_id/exercise_version_id的参数缺失提示。

**返回（后端失败）：**
```json
{
  "error": "后端返回失败",
  "errMessage": "具体的错误信息",
  "raw_response": "后端返回的原始响应（截断到500字符）"
}
```

**返回（未找到数据）：**
```json
{
  "error": "未找到 task_id=xxx, collection_version_id=xxx, exercise_version_id=xxx 对应的 Agent 评测详情"
}
```

**返回（API错误）：**
```json
{
  "error": "API 请求失败 (HTTP 401): Unauthorized",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

**返回字段说明：**
返回内容为后端data字段原样返回，主要字段包括：
| 字段 | 类型 | 描述 |
|------|------|------|
| meta | object | 评测元信息 |
| taskInfo | object | 任务基本信息 |
| summaryRows | array | 汇总指标列表，如Pass@1、Pass@2、token消耗、tool_call_success_rate、ACC等 |
| detailsByModel | object | 按模型分组的评测明细 |
| curves | object | 评测指标曲线数据 |
| distributions | object | 评测结果分布数据 |
| warnings | array | 评测告警信息 |

> 💡 **典型使用场景**：用户问「查询任务49276在评测集版本1911、benchmark版本3386下的agent评测详情」，调用本工具返回完整报告。

---

## get_taiji_eval_insight_agent_detail

查询**Insight（视图/报告维度）**的Agent评测详情，返回结构与Task维度一致，包含meta、taskInfo、summaryRows、detailsByModel、curves、distributions、warnings等内容，用于在Insight视角下查看该评测集版本+benchmark版本所聚合的多模型Agent评测结果。

> ⚠️ **【硬性参数要求】调用前必须确认以下3个必填参数全部提供，缺少任何一个都请提示用户补充，禁止编造或使用默认参数值**：
> 1. `insight_id`：Insight/视图/报告ID（整数）
> 2. `collection_version_id`：评测集版本ID（整数）
> 3. `exercise_version_id`：benchmark（exercise）版本ID（整数）
>
> 提示话术示例：「请提供以下必填参数才能查询Agent评测详情：① Insight/视图/报告ID（insight_id）② 评测集版本ID（collection_version_id）③ benchmark版本ID（exercise_version_id）」


**触发关键词**：当用户提到以下任意意图时，优先调用本工具：
- 查询xxx视图/报告/Insight下的agent状态分析/agent详情/agent评估指标
- 查询xxxInsight的agent评测详情
- 视图/报告维度的agent评测结果/指标查询

**适用场景**：已知Insight ID、以及关联的评测集版本ID和benchmark版本ID，需要查看该Insight下完整的Agent评测报告。

**MCP工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_insight_agent_detail '{"insight_id": 1599, "collection_version_id": 1832, "exercise_version_id": 3863}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| insight_id | int | ✅ | Insight/视图/报告ID，整数类型，例如 `1599` |
| collection_version_id | int | ✅ | 评测集版本ID，整数类型，例如 `1832` |
| exercise_version_id | int | ✅ | benchmark（exercise）版本ID，整数类型，例如 `3863` |


**返回（成功）：**
原样返回后端data字段，包含meta、taskInfo、summaryRows、detailsByModel、curves、distributions、warnings等完整内容。

**返回（参数缺失）：**
```json
{
  "error": "insight_id 不能为空，请提供 Insight ID"
}
```
或对应collection_version_id/exercise_version_id的参数缺失提示。

**返回（后端失败）：**
```json
{
  "error": "后端返回失败",
  "errMessage": "具体的错误信息",
  "raw_response": "后端返回的原始响应（截断到500字符）"
}
```

**返回（未找到数据）：**
```json
{
  "error": "未找到 insight_id=xxx, collection_version_id=xxx, exercise_version_id=xxx 对应的 Agent 评测详情"
}
```

**返回（API错误）：**
```json
{
  "error": "API 请求失败 (HTTP 401): Unauthorized",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

**返回字段说明：**
返回内容为后端data字段原样返回，主要字段与 `get_taiji_eval_task_agent_detail` 一致，包括meta、taskInfo、summaryRows、detailsByModel、curves、distributions、warnings等。

> 💡 **典型使用场景**：用户问「查询Insight 1599在评测集版本1832、benchmark版本3863下的agent评测详情」，调用本工具返回完整报告。

---

### 📂 实体管理 / 分析类子文档

> 按 skill-creator 渐进式披露原则，把按实体或分析维度归拢的工具拆分到独立子文档。Agent 按意图选中后按需加载对应文件，避免一次性加载本文件全部 1500+ 行。

| 用户意图 | 子文档 | grep 定位（可选） |
|----------|--------|--------|
| 上传/创建/查询/修改/删除评测数据集、管理数据集版本 | 📄 [dataset_management_api.md](../assets/dataset_management_api.md) | `^### .*dataset` / `^### .*upload` |
| 创建/查询/修改/删除评测集（Exercise）、管理评测集版本 | 📄 [exercise_management_api.md](../assets/exercise_management_api.md) | `^### .*exercise` |
| 评测集合（Collection）实体 CRUD + 集合/版本查询 + 权重树 | 📄 [collection_management_api.md](../assets/collection_management_api.md) | `^### .*collection` |
| Insight 实体 CRUD + 列表 + 详情 + 添加/移除任务 | 📄 [insight_management_api.md](../insight/insight_management_api.md) | `^### .*insight` |
| 评测**结果**分析：置信区间 / 下钻维度 / 下钻指标 / 性能趋势 / 整体进度 | 📄 [evaluation_result_analysis_api.md](../analysis/evaluation_result_analysis_api.md)（置信区间/趋势/进度） + [drill_api.md](../analysis/drill_api.md)（下钻） | `^## get_taiji_eval_(task_confidence\|bench_confidence\|performance_trend\|task_progress\|drill_dimensions\|drill_metrics)` |

---

### 规划中的扩展工具

| 工具名 | 参数 | 描述 |
|--------|------|------|
| `get_evaluation_report` | task_id* | 获取评测报告详情 |

> `*` 表示必填参数
