## get_deploy_inference_detail

根据名称查询单个推理服务的详细信息，包括配置、状态、创建人等完整信息。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| inference_name | string | ✅ | 推理服务名称，用于精确匹配 |
| wsid | int | ✅ | 工作空间 ID，必填，不能为 0 |

**返回（成功）：**
```json
{
  "id": 123,
  "name": "my-inference-service",
  "desc": "推理服务",
  "status": "running",
  "creator": "zhangsan",
  "users": ["zhangsan", "lisi"],
  "create_time": "2026-03-01T10:00:00Z",
  "update_time": "2026-03-15T14:30:00Z"
}
```

**返回字段说明（包含常用业务字段）：**

| 字段 | 类型 | 描述 |
|------|------|------|
| id | number | 推理服务 ID |
| name | string | 推理服务名称 |
| desc | string | 服务描述 |
| status | string | 服务状态 |
| creator | string | 创建人 |
| users | array | 管理员列表 |
| create_time | string | 创建时间 |
| update_time | string | 更新时间 |
| gpu_avg1h | number | GPU 实时利用率（百分比） |
| gpu_avg1d | number | GPU 日均利用率（百分比） |
| gpu_avg7d | number | GPU 周均利用率（百分比） |
| polaris | string | 北极星服务名 |
| mould_names | string | 模型名 |
| serving_monitor_url | string | 智研监控链接 |
| image | string | 镜像名称 |
| replicas | number | 期望实例数 |
| running_replicas | number | 运行实例数 |
| gpu_name | string | GPU 卡型 |
| location | string | 部署地域 |
| queue_name | string | 应用组 |

**返回（名称不存在，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 服务名称不存在",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

**返回（参数缺失）：**
```json
{
  "error": "推理服务名称 (inference_name) 不能为空",
  "hint": "请提供要查询的推理服务名称"
}
```

> 💡 **提示**：获取到服务详情后，可以调用 `update_deploy_inference` 编辑服务信息（需提供 `id` 字段值作为 `inference_id`）。

---

## list_deploy_inferences

查询工作空间下的推理服务列表，支持分页、关键词搜索、创建人过滤、实例状态过滤。

> ⚠️ **wsid 为必填参数**，调用前必须确保用户已提供工作空间 ID。
> ⚠️ **polaris 不支持**：`keyword` 参数仅对服务名/字段进行匹配，**不支持搜索北极星注册地址（polaris 字段）**。用户要求按 polaris 地址筛选/统计时，不要调用本工具，直接说明不支持。
> 🛑 **结果展示规则**：API 返回什么就展示什么，**严禁写 Python 脚本/Bash 二次过滤或重新排序**。`instance_status=has_instance` 返回的是"配置过实例的服务"（可能 `pods=0/N`），若与用户预期不符，告知 API 语义即可，不得绕过。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| wsid | int | ✅ | 工作空间 ID，必填，不能为 0 |
| keyword | string | 否 | 关键词，按推理服务名称等字段模糊搜索 |
| creator | string | 否 | 创建人或最近修改人 RTX，过滤该用户创建或管理的推理服务 |
| only_mine | boolean | 否 | 传 `true` 表示仅查看当前用户负责的推理服务，默认 `false` |
| instance_status | string | 否 | 实例状态筛选，仅支持 `all`/`no_instance`/`has_instance` |
| page | int | 否 | 页码，从 1 开始，默认 1 |
| page_size | int | 否 | 每页数量，默认 20 |

**过滤规则：**
- **查询"我的"推理服务**（仅创建人是当前用户的） → **必须**使用 `only_mine=true`
  - ⚠️ 当用户说"我的服务"、"我创建的"、"我负责的"等表达时，必须传 `only_mine=true`，不得省略！
- **查询某人创建或管理的推理服务** → 使用 `creator=RTX`（会匹配创建人或管理员）
- **按关键词模糊搜索**（匹配名称、描述、模型名称、应用组、ID） → 使用 `keyword`
- `only_mine` 和 `creator` 不要同时使用

> ⚠️ **重要**：当用户意图是查询某个人创建/管理的服务时，**必须使用 `creator` 参数**，**严禁**将人名/RTX 放入 `keyword` 参数进行模糊搜索。`keyword` 仅用于按服务名称、描述等业务关键词搜索。

**💡 通过模型名称搜索：**
`keyword` 参数支持通过**模型名称**搜索关联的推理服务。例如用户说"查找使用了 HunYuan-Large 模型的服务"，直接传 `keyword="HunYuan-Large"` 即可。后端会匹配推理服务关联的模型名称（`mould__name` 字段）。


**返回（成功）：**
```json
{
  "count": 15,
  "results": [
    {
      "id": 123,
      "name": "my-inference-service",
      "desc": "推理服务",
      "status": "running",
      "creator": "zhangsan",
      "create_time": "2026-03-01T10:00:00Z"
    }
  ]
}
```

**返回（wsid 缺失）：**
```json
{
  "error": "wsid（工作空间 ID）不能为空",
  "hint": "请先向用户询问工作空间 ID，可以在太极大模型平台工作台界面找到"
}
```

**返回字段说明：**

| 字段 | 类型 | 描述 |
|------|------|------|
| count | number | 总记录数 |
| results | array | 推理服务列表 |
| results[].id | number | 推理服务 ID（编辑时需要使用） |
| results[].name | string | 推理服务名称 |
| results[].desc | string | 服务描述 |
| results[].status | string | 服务状态 |
| results[].creator | string | 创建人 |
| results[].create_time | string | 创建时间 |

> ⚠️ **列表顺序强制规则**：`results` 数组已按后端默认排序（按 ID 降序，最新创建的排在前面）。展示时**必须严格按照数组原始顺序展示**，严禁重新排序、筛选或截断。
>
> 💡 **调用约束**：只调一次带齐所有已知过滤参数（`only_mine`/`creator`/`keyword`/`instance_status`/`page=1`/`page_size=20`），不重复分页、不写临时脚本、不同参数重查同一列表。
>
> 💡 **提示**：获取到 `id` 后，可以调用 `update_deploy_inference` 编辑推理服务信息，或使用 `name` 调用 `get_deploy_inference_detail` 查看完整详情。

---

## update_deploy_inference

编辑单个推理服务的信息（部分更新），仅支持修改描述和管理员。

> ⚠️ **编辑操作前请先向用户确认变更内容**，确认后再调用此工具。
> ⚠️ `inference_id` 可通过 `get_deploy_inference_detail` 或 `list_deploy_inferences` 获取。
> ⚠️ **可编辑字段限制**：仅支持修改 `desc`（描述）和 `users`（管理员）。其他字段（如名称、状态等）不支持通过此接口修改。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| inference_id | int/string | ✅ | 推理服务 ID |
| desc | string | 否 | 推理服务描述（`desc`/`users` 至少提供其一） |
| users | array[string] | 否 | 推理服务成员用户名列表（`desc`/`users` 至少提供其一） |
| wsid | int | 否 | 工作空间 ID，建议显式传入 |

> 💡 **v2 结构变更**：旧版通过 `data` JSON 字符串包装传参（如 `data='{"desc":"..."}'`），新版直接把 `desc` / `users` 平铺为**顶层参数**，无需再序列化 JSON。


**返回（成功）：**
```json
{
  "id": 123,
  "name": "my-inference-service",
  "desc": "更新后的描述",
  "status": "running",
  "creator": "zhangsan",
  "update_time": "2026-03-20T16:00:00Z"
}
```

**返回（权限不足，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 只有创建人和管理员可以更新",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

**返回（ID 不存在，HTTP 404）：**
```json
{
  "error": "API 请求失败 (HTTP 404): Not found.",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

---

## clone_deploy_inference

克隆一个已有的推理服务，快速创建新服务。根据源服务的配置（模型、镜像、资源等）快速复制出一个新的推理服务，可选择性地覆盖部分配置。

> ⚠️ **克隆操作会创建新服务并占用 GPU 资源，属于高风险操作**，请确认源服务名称和目标服务名称无误后再调用。
>
> 🚫 **⚠️ 禁止直接调用 clone_deploy_inference 部署（最高优先级）**：部署新服务**必须**走 `deploy_from_template.py` 脚本（`## 快速部署` 章节）。**严禁**在未尝试 `deploy_from_template.py` 的情况下直接调用 `clone_deploy_inference`，因为脚本会自动补全模板参数（镜像、端口、环境变量等），手动拼参极易遗漏关键字段导致部署失败或配置错误。只有在脚本明确返回错误且 hint 指引需要手动调用时才可回退到直接调用。
>
> 🚫 **严禁自主决策规则（最高优先级，违反即为严重错误）**：
> 1. **严禁自行选择源服务**：源服务名称必须由用户明确提供，绝不能从服务列表中自行挑选。即使用户有权限的服务列表中只有一个服务，也必须向用户确认。
> 2. **严禁自行选择应用组**：应用组（`app_group_id`）必须由用户明确提供，绝不能从用户有权限的应用组列表中自行挑选。即使用户只有一个应用组，也必须向用户确认。
> 3. **严禁自行决定新服务名称**：新服务名称必须由用户明确提供或确认。
> 4. **严禁自行决定 GPU 卡型和地域**：`gpu_name` 和 `location` 必须由用户明确提供。
> 5. **总原则**：本工具的所有参数值都必须来自用户的明确指示，不得通过查询其他工具后自行组合参数调用。查询结果应展示给用户选择，而非替用户做决定。
>
> ✅ **二次确认规则**：在收集完所有参数后、调用工具前，必须向用户展示完整的克隆配置摘要（源服务、目标服务名、工作空间、应用组、GPU 卡型、地域、模型等）并请求确认，用户明确确认后才能调用工具。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| source_inference_name | string | ✅ | 被克隆的源服务名称 |
| target_inference_name | string | ✅ | 目标服务的名称 |
| wsid | int | 否 | 目标服务所在的工作空间 ID（默认与源服务相同） |
| desc | string | 否 | 目标服务的描述信息 |
| image_name | string | 否 | 目标服务使用的镜像名称（默认与源服务相同） |
| replicas | int | 否 | 目标服务的实例数（默认 1） |
| auto_create_service | boolean | 否 | 是否自动创建同名服务组（默认 false） |
| gpu_name | string | 否 | 目标服务的 GPU 卡型 |
| location | string | 否 | 目标服务的部署地域 |
| app_group_id | string | 否 | 目标服务的应用组 ID |
| model_ids | array[integer/string] | 否 | 新模型 ID 列表，如 `[100527]`。可通过 `search_hunyuan_models_cards` 搜索获取模型 ID |
| model_location | string | 否 | 新模型地域（英文缩写，如 sz、sh、nj） |
| gpu_per_host | int | 否 | 目标服务单机 GPU 卡数（从 `get_deploy_template_detail` 获取） |
| host_count | int | 否 | 目标服务 GPU 机器数（从 `get_deploy_template_detail` 获取） |
| pipeline_parallel_size | int | 否 | 流水线并行大小（INFERENCE_PP_SIZE，从 `get_deploy_template_detail` 获取） |
| tensor_parallel_size | int | 否 | 张量并行大小（INFERENCE_TP_SIZE，从 `get_deploy_template_detail` 获取） |
| framework_type | string | 否 | 推理框架类型（从 `get_deploy_template_detail` 获取） |
| service_scene | string | 否 | 服务场景（从 `get_deploy_template_detail` 获取），如 text_to_text、multimodal、text_to_image、audio |
| start_command | string | 否 | 自定义启动命令（从 `get_deploy_template_detail` 获取） |
| envs | object | 否 | 自定义环境变量键值对，形如 `{"KEY1":"v1","KEY2":"v2"}`（从 `get_deploy_template_detail` 获取） |
| copy_polaris_config | boolean | 否 | 是否复制源服务的北极星配置（`polaris`、`polaris_token`、`polaris_env`、`polaris_weight` 4 个字段），**默认 false**（与前端"不勾选北极星"行为一致）。设为 true 前源服务必须已配置 `polaris_token`，否则接口返回 HTTP 400。**⚠️ 类型严格要求 bool，传字符串 "true"/"false" 会被后端拒绝。** |

> 💡 **v2 参数变更提示**：
> - 参数改名：`from_inference_name→source_inference_name`、`name→target_inference_name`、`queue_name→app_group_id`、`moulds→model_ids`、`mould_location→model_location`、`host_gpu_num→gpu_per_host`、`host_num→host_count`、`pp_size→pipeline_parallel_size`、`tp_size→tensor_parallel_size`。
> - 类型变更：`envs` 由字符串（`"K=V,K2=V2"`）改为**对象**（`{"K":"V"}`）。
> - 语义变更：`app_group_id` 传的是**应用组 ID**（旧 `queue_name` 传的是应用组名称）。

> ⚠️ **参数分组说明**：克隆服务时有两组可选的配置变更，每组内的参数具有关联性：

| 变更类型 | 参数组 | 用户必须先明确的信息 | 缺失信息的补全方式 |
|---------|--------|-------------------|------------------|
| **更换部署应用组** | `app_group_id` + `location` + `gpu_name` | 应用组 ID（`app_group_id`） | 通过 `query_app_group_detail` 和 `query_app_group_gpu_info` 查询后返回给用户选择 |
| **更换部署模型** | `model_ids` + `model_location` | 模型名称或模型 ID（`model_ids`） | 通过 `get_hunyuan_models_card_detail` 查询模型地域信息后返回给用户确认 |


**返回（成功，HTTP 201）：**
```json
{
  "id": 12345,
  "name": "my-new-service",
  "wsid": "workspace-12345",
  "desc": "这是克隆后的新服务",
  "status": 0
}
```

**返回（源服务名称为空）：**
```json
{
  "error": "源服务名称 (source_inference_name) 不能为空",
  "hint": "请提供要克隆的源推理服务名称"
}
```

**返回（目标服务名称为空）：**
```json
{
  "error": "目标服务名称 (target_inference_name) 不能为空",
  "hint": "请提供目标服务的名称"
}
```

**返回（源服务不存在，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 服务[xxx]不存在",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

**返回（自动创建服务组失败，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 自动创建服务组失败: <错误详情>",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

**返回（copy_polaris_config=true 但源服务未配置 polaris_token，HTTP 400）：**
```json
{
  "code": 40001,
  "message": "源服务未配置北极星 token，无法复制北极星配置，请关闭 copy_polaris_config 开关",
  "data": null
}
```

**返回（copy_polaris_config 类型错误，HTTP 400）：**
```json
{
  "code": 40001,
  "message": "copy_polaris_config 参数类型错误，必须为布尔值（true/false）",
  "data": null
}
```

> 💡 **提示**：克隆成功后，可以调用 `get_deploy_inference_detail` 查看新服务的完整详情，或调用 `list_deploy_inferences` 查看服务列表。

### 更换应用组时的信息补全流程

当用户在克隆服务时传入了新的 `app_group_id`（应用组），但未同时提供 `location`（地域）或 `gpu_name`（GPU 卡型）时，**必须**按以下流程补全信息后再调用 `clone_deploy_inference`：

1. **查询应用组地域信息**：调用 `query_app_group_detail`，传入 `business_flag` 参数（值为用户提供的应用组名称），获取返回结果中的 `location` 字段，了解该应用组支持的部署地域。
2. **查询应用组 GPU 卡型配额（按地域）**：调用 `query_app_group_gpu_info`，传入 `app_group_id` 参数，获取返回结果中按 (卡型, 地域) 拆分的配额信息（包括总配额、已使用、空余、排队中等）。
3. **向用户展示可选信息**：将查询到的地域和 GPU 卡型列表（含配额）展示给用户，引导用户选择具体的地域和 GPU 卡型。
4. **用户选择后传入参数**：将用户选择的地域和 GPU 卡型分别作为 `location` 和 `gpu_name` 参数传入 `clone_deploy_inference`。

> ⚠️ **注意**：
> - 如果用户更换了应用组且**同时**传入了 `location` 和 `gpu_name`，则不触发上述查询流程，直接使用用户传入的值。
> - 如果应用组详情查询失败，不阻断主流程，提示用户手动确认地域和卡型信息后继续。

### 更换模型时的跨地域预检查流程

当用户在克隆服务时传入了新的 `model_ids`（模型 ID 列表），**必须**按以下流程进行跨地域预检查后再调用 `clone_deploy_inference`：

1. **查询模型地域信息**：调用 `get_hunyuan_models_card_detail`，传入模型 ID（`model_ids` 列表中的第一个 ID），获取返回结果中的 `location_infos` 字段，了解该模型在各地域的文件分布。
2. **确定部署地域**：
   - 如果用户传入了 `location` 参数 → 部署地域使用用户传入的值。
   - 如果用户未传入 `location` 参数 → 调用 `get_deploy_inference_detail` 查询源服务详情，从返回结果中获取 `location` 字段作为部署地域。
3. **比对模型地域与部署地域**：检查模型的 `location_infos` 中是否有任一地域与部署地域一致（地域统一使用英文缩写，如 sz、sh、nj、bj、gz）。
4. **地域一致**：自动将部署地域作为 `model_location` 参数传入 `clone_deploy_inference`，无需用户确认。
5. **地域不一致**：向用户提示跨地域风险：
   > "⚠️ 当前模型没有部署地域（{部署地域}）同地域的模型文件，跨地域模型克隆可能需要较长时间（预估 2 小时以上）。是否继续跨地域部署？"
   - 用户确认继续 → 将模型实际所在地域作为 `model_location` 参数传入 `clone_deploy_inference` 执行克隆。
   - 用户取消 → 建议用户更换为同地域的模型或调整部署地域。
6. **模型地域查询失败**：不阻断主流程，提示用户"无法获取模型地域信息，请注意可能存在跨地域部署风险"后继续执行。

> ⚠️ **注意**：
> - 如果用户未更换模型（未传入 `model_ids` 参数），则不触发跨地域预检查。
> - 如果同时更换了模型和应用组，应先完成应用组信息补全（确定部署地域），再进行模型跨地域预检查。
> - 模型在多个地域有文件时，`location_infos` 可能包含多个地域，只要其中一个与部署地域一致即视为同地域。

### 字段限制（前置校验）

| 字段 | 最大长度 | 格式要求 | 说明 |
|------|---------|---------|------|
| name（服务名） | **58 字符** | 以字母或数字开头和结尾，中间可包含字母、数字、`.`、`_`、`-` | 全局唯一，正则：`^([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]$` |
| desc（服务描述） | 255 字符 | 无特殊格式要求 | — |

> ✅ 合法示例：`my-service-v1`、`model_deploy.prod`、`HunYuan-7B-SFT`
> ❌ 不合法：`-start-with-dash`、`end-with-dash-`、`中文名称`、`name with spaces`

> ⚠️ **字段映射**：`list_deploy_inferences` 返回 `name`（短名）和 `service`（带 `_AIDE` 后缀）两个字段。clone 时**必须用 `name` 字段的值**作为 `source_inference_name`/`target_inference_name`，用 `service` 会报 HTTP 400。

---

## list_deploy_instances

查询推理服务的实例列表，支持按 IP、版本、状态等条件筛选，可选择性返回实例指标数据。

> ⚠️ **inference_id 为必填参数**，必须通过 `get_deploy_inference_detail(name=...)` 或 `list_deploy_inferences(keyword=...)` 先查到服务拿到 `id` 字段后传入。**严禁**将 URL 中的 `id=` 参数当作 `inference_id` 直接传入——URL 中的 id 不一定是推理服务 ID。如果按名称查不到服务，直接告知用户"找不到该服务"，不要换 keyword 反复搜索。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| inference_id | int/string | ✅ | 推理服务 ID |
| wsid | int | 否 | 工作空间 ID，建议显式传入 |
| page | int | 否 | 页码，从 1 开始，默认 1 |
| page_size | int | 否 | 每页数量，默认 20 |
| enable_metrics | boolean | 否 | 是否返回实例监控指标（如 GPU 利用率等），默认 true |
| ips | array[string] | 否 | 按实例 IP 筛选 |
| host_ips | array[string] | 否 | 按宿主机 IP 筛选 |
| versions | array[string] | 否 | 按版本筛选 |
| instance_ids | array[string] | 否 | 按实例 ID 筛选 |
| statuses | array[string] | 否 | 按状态筛选 |
| namespace | string | 否 | 按 Kubernetes namespace 筛选 |

> 💡 **v2 参数变更提示**：`status_str` 改名为 `statuses`；`ips` / `host_ips` / `versions` / `instance_ids` / `statuses` 全部由**逗号分隔字符串**改为 **`array[string]`**（如 `["29.225.114.146", "29.225.114.147"]`）。


**请求体示例：**
```json
{
  "inference_id": 908049,
  "wsid": 12345,
  "page": 1,
  "page_size": 20,
  "enable_metrics": true,
  "ips": [],
  "host_ips": [],
  "versions": [],
  "instance_ids": [],
  "statuses": [],
  "namespace": ""
}
```

**返回（成功）：**
```json
{
  "count": "1",
  "results": [
    {
      "servingId": "5425650",
      "instanceId": "taiji-serving-custom-5425650-0",
      "namespace": "taiji-hyaide-adt-nj-h20-nanjing-10",
      "ip": "29.225.114.146",
      "hostIp": "29.225.114.146",
      "ctTime": "2026-03-23 17:17:53",
      "upTime": "2026-03-23 17:20:50",
      "version": "v1",
      "versionId": "1428376",
      "status": "Running",
      "statusZh": "运行中",
      "password": "3ESD#r4CZ",
      "instanceIndex": 0,
      "age": "17分",
      "containerInfo": [
        {
          "name": "taiji-serving-container-0",
          "id": "docker://a420ea207e...",
          "image": "mirrors.tencent.com/hunyuan_infer/text2text_infer:H-6.5.1.dev139",
          "status": "Running"
        }
      ],
      "instanceMetricsInfo": {
        "GPU_Util": {
          "keyDataList": [
            {"current": 0, "desc": "平均值"},
            {"current": 0, "desc": "最小值"},
            {"current": 0, "desc": "最大值"}
          ]
        }
      },
      "nodeIpList": ["29.225.114.146"],
      "resourceMetricUrl": "https://zhiyan.woa.com/monitor/..."
    }
  ],
  "flag": true,
  "scene": "test"
}
```

**返回字段说明（展示名称 → 后端字段映射）：**

| 展示名称 | 后端字段 | 类型 | 描述 |
|------|------|------|------|
| 实例ID | instanceId | string | 实例唯一标识 |
| 命名空间 | namespace | string | 实例所属命名空间 |
| 服务镜像 | containerInfo[0].image | string | 取第一个容器（主容器）的镜像 |
| 运行状态 | statusZh | string | 中文状态描述（如"运行中"） |
| GPU 7天平均利用率 | instanceMetricsInfo.GPU_Util.keyDataList[0].current | number | 需 enable_metrics=true，单位百分比(%) |
| 运行时长 | age | string | 如 "17分"、"2小时" |
| 创建时间 | ctTime | string | 格式 YYYY-MM-DD HH:MM:SS |
| 更新时间 | upTime | string | 格式 YYYY-MM-DD HH:MM:SS |

**敏感信息字段（默认不展示，用户追问时才返回）：**

| 展示名称 | 后端字段 | 类型 | 描述 |
|------|------|------|------|
| 容器IP List | nodeIpList | array | IP 列表，多个用逗号拼接展示 |
| Master节点IP | ip | string | Master 节点 IP |
| 宿主机IP | hostIp | string | 宿主机 IP |
| 版本 | version | string | 实例版本 |
| 密码(root用户) | password | string | SSH root 用户密码，敏感信息 |
| ssh_port | ssh_port | string | SSH 端口（如接口未返回则显示 "36001"） |
| GPU指标监控链接 | resourceMetricUrl | string | 智研 GPU 指标监控链接 |

**其他返回字段：**

| 字段 | 类型 | 描述 |
|------|------|------|
| count | number | 实例总数 |
| results | array | 实例列表 |
| results[].servingId | string | 服务 ID |
| results[].versionId | string | 版本 ID |
| results[].ip | string | Master 节点 IP |
| results[].status | string | 英文状态（如 Running） |
| results[].instanceIndex | number | 实例索引 |
| results[].containerInfo | array | 容器信息列表（含镜像、容器ID、状态等） |
| results[].resourceMetricUrl | string | 智研监控链接 |
| flag | boolean | 请求是否成功 |
| scene | string | 场景标识 |

**返回（inference_id 缺失）：**
```json
{
  "error": "推理服务 ID (inference_id) 不能为空",
  "hint": "请先通过查询推理服务详情或列表获取服务 ID"
}
```

**返回（服务不存在，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 服务不存在",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

> 💡 **提示**：查看实例列表前，建议先通过 `get_deploy_inference_detail` 确认服务存在，然后使用返回的 `id` 字段作为 `inference_id` 参数。

---

## get_deploy_instance_logs

查看推理服务实例的日志，支持启动日志（`start_log`）、事件日志（`event_log`）和请求日志（`request_log`）三种类型。

> ⚠️ **instance_id 与 inference_id 均为必填参数**，可通过 `list_deploy_instances` 获取。namespace 为可选参数，传入可提高查询精度。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| instance_id | string | ✅ | 实例 ID，可通过 `list_deploy_instances` 获取 |
| inference_id | int/string | ✅ | 推理服务 ID |
| wsid | int | 否 | 工作空间 ID，建议显式传入 |
| namespace | string | 否 | Kubernetes namespace，传入可提高查询精度 |
| log_type | string | 否 | 日志类型：`start_log`（启动日志，默认）、`event_log`（事件日志）、`request_log`（请求日志） |
| page | int | 否 | 页码，从 1 开始，默认 1 |
| page_size | int | 否 | 每页数量，默认 2000 |
| query | string | 否 | 日志内容关键词搜索 |
| level | string | 否 | 日志级别过滤（如 INFO、WARNING、ERROR） |
| start_time | string | 否 | 开始时间，格式 `YYYY-MM-DD HH:MM:SS`，默认当天 00:00:00 |
| end_time | string | 否 | 结束时间，格式 `YYYY-MM-DD HH:MM:SS`，默认当前时间 |

> 💡 **v2 参数变更提示**：`log_type` 枚举值全部改名 —— `log→start_log`、`event→event_log`、`framework→request_log`；`page_size` 默认值由 `50` 改为 `2000`；`inference_id` 由可选变为**必填**。

**日志类型说明：**

| log_type | 名称 | 说明 |
|----------|------|------|
| start_log | 启动日志 | 实例启动过程中的日志输出 |
| event_log | 事件日志 | Kubernetes 事件信息（调度、拉取镜像、容器状态变化等） |
| request_log | 请求日志 | 推理框架的请求处理日志 |


**返回（成功）：**
```json
{
  "count": 100,
  "results": [
    {
      "timestamp": "2026-03-23 17:17:55",
      "content": "Starting inference server...",
      "level": "INFO"
    }
  ]
}
```

**返回（instance_id 缺失）：**
```json
{
  "error": "实例 ID (instance_id) 不能为空",
  "hint": "请先通过 list_deploy_instances 获取实例 ID"
}
```

**返回（log_type 不合法）：**
```json
{
  "error": "不支持的日志类型: xxx",
  "hint": "log_type 可选值：start_log（启动日志）、event_log（事件日志）、request_log（请求日志）"
}
```

> 💡 **提示**：查看实例日志前，建议先通过 `list_deploy_instances` 获取实例 ID 与 `inference_id`，然后调用本工具查看日志。可以通过 `log_type` 切换不同类型的日志，也可以通过 `query` 和 `level` 进行筛选。可选传入 `namespace` 进一步提高查询精度。
>
> ⚠️ **调用约束**：按用户语义选对 `log_type`（`start_log` / `event_log` / `request_log`），混用拿错数据。根因排查只取一个异常/代表实例的一次日志；不重复取、不用本地 grep 二次处理（除非用户明确要求多实例对比）。
>
> ⚠️ **日志展示强制规则**：日志内容必须使用代码块**原封不动**地展示，**严禁对日志进行任何形式的解读、摘要、分类、过滤或省略**。日志是用户排查问题的原始依据，必须完整、原样返回。

---

## exec_deploy_instance_command

在**推理服务实例（Pod）**内执行一条命令，用于测试服务的现场排查：看进程、看显存、看文件、看环境变量、抓包、重启推理进程等。成功时返回 `stdout` / `stderr` / `exit_code`。

> ⚠️ **别和训练实例 exec 搞混**：本工具只作用于**推理服务**的实例。用户说的是"训练任务/训练实例的 pod 跑命令"时，属于 `instance-skill` 的 `exec_hunyuan_train_instance_command`（参数是 `instance_id` + `pod_name`），必须回顶层切换 skill，不要用本工具。

### 🚦 三个硬前置条件（任一不满足即失败，不要重试、不要换参数试探）

| 条件 | 不满足时 | 怎么提前确认 |
|---|---|---|
| **仅测试服务**：推理服务 `scene` 必须是 `test` | HTTP **403** | `list_deploy_instances` 返回体顶层的 `scene` 字段 |
| **权限**：调用人须是该推理服务的创建人、服务成员（`users`）或超级管理员 | HTTP **403** | `get_deploy_inference_detail` 的 `creator` / `users` |
| **实例 Running**：目标实例必须处于 `Running` | HTTP **400** | `list_deploy_instances` 的 `results[].status` |

### 前置链路（缺参数时才走）

```
get_deploy_inference_detail(inference_name, wsid)   → 取 id 作 inference_id
  → list_deploy_instances(inference_id, wsid)       → 取 results[].instanceId 作 instance_id
                                                      顺带确认 status == "Running"、顶层 scene == "test"
                                                      多容器时容器名取 results[].containerInfo[].name
  → exec_deploy_instance_command(...)
```

> ⚡ **用户已直接给出 `inference_id` + `instance_id` 时，跳过前置链路，一步直接执行**；不要为了"确认一下服务存在/是不是测试服务"补查详情或实例列表——前置条件由后端强校验，失败会明确报 403/400。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| inference_id | int | ✅ | 推理服务 ID，可通过 `get_deploy_inference_detail` / `list_deploy_inferences` 获取 |
| instance_id | string | ✅ | 实例 ID（目标 Pod），可通过 `list_deploy_instances` 获取（`results[].instanceId`） |
| command | string | ✅ | 待执行命令**整句**，如 `nvidia-smi`、`ls -l /data`。不能为空 |
| container | string | 否 | 目标容器名；多容器 Pod 时用于指定，不传则按默认容器规则解析 |
| background | boolean | 否 | 是否后台常驻执行，默认 `false`。语义见下方「background=true」段 |

> ⛔ **参数白名单**：本工具**不接受** `wsid` / `timeout_sec` / `namespace` / `pod_name` / `pod` / `argv` / `shell` 等参数，传了会被后端静默丢弃。

### 🔴 command 不经过 shell（最容易踩的坑）

后端只对 `command` 做**词法分词**后把 argv 直连 exec，**不起 shell**。因此：

- `|`、`>`、`>>`、`&&`、`;`、`$()`、`*` 通配符 **全部退化成普通参数**，不具备 shell 语义
- 需要 shell 语法时**必须显式包一层解释器**：`bash -c "cd /data && ls | head"`（`bash` / `sh` / `python` 等解释器未被拦截）

| 用户想要 | ❌ 错误写法（符号会被当字面量） | ✅ 正确写法 |
|---|---|---|
| 看日志尾部并过滤 | `tail -n 500 /data/logs/x.log \| grep ERROR` | `bash -c "tail -n 500 /data/logs/x.log \| grep ERROR"` |
| 看某个环境变量 | `echo $CUDA_VISIBLE_DEVICES` | `bash -c 'echo $CUDA_VISIBLE_DEVICES'`，或直接 `env` 后自行查看 |
| 进目录再列文件 | `cd /data && ls -l` | `bash -c "cd /data && ls -l"` |
| 输出重定向到文件 | `nvidia-smi > /data/gpu.txt` | `bash -c "nvidia-smi > /data/gpu.txt"` |
| 看 GPU / 进程 / 磁盘 / 文件 | —— 本来就不需要 shell | `nvidia-smi`、`ps aux`、`df -h`、`ls -l /data`、`cat /path/x` |

### ⛔ 高危命令拦截

**只有会造成不可逆破坏的命令要拦下**：删数据、格式化磁盘、关机/重启节点。后端按 `argv[0]` 的文件名拦一批（`rm`、`dd`、`mkfs`、`shred`、`truncate`、`wipefs`、`reboot`、`shutdown`、`halt`、`poweroff`，命中返回 **400**），但等价写法它**拦不住**，判断责任在 Agent。

| 高危类型 | 例子（含后端拦不住的等价写法） |
|---|---|
| 删除 / 清空数据 | `rm -rf /data`、`bash -c "rm -rf /data"`、`find /data -delete`、`bash -c "> /data/model.bin"` |
| 格式化 / 破坏磁盘 | `mkfs`、`wipefs`、`dd of=/dev/...` |
| 关机 / 重启节点 | `reboot`、`shutdown`、`poweroff`、`halt` |

> ✅ **这些不算高危，是本工具的正常用途，照常执行**：`pkill -f 'vllm serve'` / `kill` 推理进程、`background=true` 重新拉起服务、`pip install` 补依赖、改配置文件、`chmod` 单个文件。爆炸半径是单个可重建的测试 Pod，用户本来就有重启推理进程的需求。有副作用的照例把命令原文复述一遍再执行即可。

规则：
1. 命中高危 → 说清后果，**等用户明确确认再执行**。
2. 后端返回 400 拦截 → **原样告知用户并停止**，严禁改写成 `bash -c "..."` 或换等价命令绕过。
3. 命令原文只能来自用户，**严禁自行编造或补** `-r` / `-f` / `-9` / 通配符 `*` / 猜路径。

### ⏱️ 前台 60 秒硬超时 + 400 的双重含义

- 前台（`background=false`，默认）命令有 **60 秒硬超时**，到点被 SIGKILL。
- **命令退出码非 0（含超时）时接口返回 HTTP 400**，`message` 形如 `command terminated with exit code N`；**此时拿不到 stdout / stderr**。
- `exit_code` 字段**只在成功时有意义**（恒为 `0`）。
- ⚠️ 因此"接口调用失败"和"命令本身跑出非 0 结果"在本接口是**同一种返回**。看到 400 且 message 是 `command terminated with exit code N` 时，应如实告诉用户「命令执行了但退出码非 0（可能是命令本身报错，也可能是 60 秒超时）」，**不要**归因成权限问题或服务不存在，**不要**自动重试或换命令。

### 🌙 background=true（后台常驻）

`background=true` 时后端用**固定的 detach 骨架**（`setsid <cmd> </dev/null >>日志 2>&1 &`）把命令拉起：**秒返回**，进程脱离当前会话在 Pod 内常驻，不受 60 秒约束，一直跑到自己退出或 Pod 重建。

- ✅ 适用：重新拉起 vllm、长时间抓包、后台重跑等长任务
- ❌ **只想拿一条命令的输出时严禁开**——开了 `stdout` 只会是骨架回显（形如 `started pid=<pid>`），拿不到真实输出
- 后台任务要看结果：**再发一条前台命令** `tail` 对应日志文件
- 用户不能自定义 detach 方式和重定向目标，骨架由后端写死


**调用示例：**
```json
{
  "inference_id": 908049,
  "instance_id": "8b2027419ea55901019ea5f447a90058",
  "command": "nvidia-smi"
}
```

**返回（成功）：**
```json
{
  "stdout": "Thu Jul 27 10:21:33 2026\n+---------------------...",
  "stderr": "",
  "exit_code": 0
}
```

**返回（正式服务 / 无权限，HTTP 403）：**
```json
{
  "error": "仅测试服务支持命令执行",
  "hint": "本工具仅支持 scene=test 的测试服务；且调用人须是该推理服务的创建人、服务成员或超级管理员"
}
```

**返回（命中黑名单 / 分词失败 / 命令为空 / 实例非 Running / 退出码非 0，HTTP 400）：**
```json
{
  "error": "command terminated with exit code 1",
  "hint": "命令已执行但退出码非 0（命令自身报错或超过 60 秒被 SIGKILL），此时无 stdout/stderr"
}
```

> 🔒 **审计**：每次执行都会落审计（操作人、来源 IP、完整命令、完整输出），可事后追责到人。
>
> ⚠️ **输出展示强制规则**：`stdout` / `stderr` 必须用代码块**原封不动**展示，**严禁解读、摘要、过滤或省略**——这是用户排查问题的原始依据。`env` / `cat` 可能读到 token、密码等凭证，仍原样返回，但提示用户注意不要外传。

---

## stop_deploy_inference_wait

取消推理服务的排队等资源状态，让服务不再继续等待 GPU 等资源分配。**不可逆操作，调用前必须向用户确认。**

> ⚠️ **调用前必须二次确认**：取消排队不可逆，必须明确告知用户后果并等待确认。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| inference_id | int/string | ✅ | 推理服务 ID，可通过 `get_deploy_inference_detail` 或 `list_deploy_inferences` 获取 |
| wsid | int | 否 | 工作空间 ID，建议显式传入 |


**调用示例：**
```json
{
  "inference_id": 1138485,
  "wsid": 12345
}
```

**返回（成功）：**
```json
{
  "code": 0,
  "message": "取消排队成功"
}
```

**返回（inference_id 缺失）：**
```json
{
  "error": "推理服务 ID (inference_id) 不能为空",
  "hint": "请先通过 get_deploy_inference_detail 或 list_deploy_inferences 获取服务 ID"
}
```

**返回（HTTP 400 错误）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 具体错误信息",
  "hint": "请检查推理服务 ID 是否正确，以及当前服务是否处于排队状态"
}
```

> 💡 **提示**：取消排队后，服务不再等待资源分配。如果需要恢复，需要重新触发部署流程。

---

## stop_deploy_inference_change

终止/停止进行中的变更单（如扩缩容、实例重启等），使变更单状态变为 KILL。**不可逆操作，调用前必须向用户确认。**

> ⚠️ **调用前必须二次确认**：终止变更是不可逆的，必须明确告知用户后果并等待确认。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| deployment_id | int/string | ✅ | 变更单 ID，可通过 `create_deploy_inference_change` 返回值获取，或前往太极平台 Web 界面查看 |
| wsid | int | 否 | 工作空间 ID，建议显式传入 |

> 🔗 若刚调用了 `create_deploy_inference_change`，**其返回值中的 `id` 字段就是本工具的 `deployment_id`**，直接传入即可，无需额外查询。


**调用示例：**
```json
{
  "deployment_id": 720600,
  "wsid": 12345
}
```

**返回（成功）：**
```json
{
  "code": 0,
  "message": "变更单已终止",
  "status": "KILL"
}
```

> 💡 **注**：后端有时会返回 200 空响应表示终止成功，MCP 工具会自动将其转换为以上统一成功格式返回。

**返回（deployment_id 缺失或非法）：**
```json
{
  "error": "变更单 ID (deployment_id) 不能为空或小于等于 0",
  "hint": "请先通过 create_deploy_inference_change 返回值获取变更单 ID，或前往太极平台 Web 界面查看"
}
```

**返回（HTTP 错误）：**
```json
{
  "error": "API 请求失败 (HTTP xxx): 具体错误信息",
  "hint": "请检查变更单 ID 是否正确，以及当前变更单是否处于可终止状态"
}
```

> 🛑 **错误处理规则**：调用失败时**立即停止**，把错误原文告知用户。不得重试、换参数、查看源码或写脚本绕过。

> 💡 **提示**：终止后变更单状态变为 KILL，pipeline stages 变为 FAILED。如果需要重新执行变更，需要重新创建变更单。

---

## get_deploy_service_detail

根据名称查询单个服务组的详细信息，包括绑定的推理服务、权重、管理员等完整信息。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| service_name | string | ✅ | 服务组名称，用于精确匹配 |
| wsid | int | ✅ | 工作空间 ID，必填，不能为 0 |


**返回（成功）：**
```json
{
  "id": 456,
  "name": "my-service-group",
  "desc": "服务组描述",
  "creator": "zhangsan",
  "users": ["zhangsan", "lisi"],
  "inferences": [
    {
      "id": 123,
      "name": "my-inference-service",
      "weight": 100
    }
  ],
  "create_time": "2026-03-01T10:00:00Z",
  "update_time": "2026-03-15T14:30:00Z"
}
```

**返回（名称不存在，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 服务组名称不存在",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

> 💡 **提示**：获取到服务组详情后，可以调用 `update_deploy_service` 编辑服务组信息（需提供 `id` 字段值作为 `service_id`）。

---

## list_deploy_services

查询工作空间下的服务组列表，支持分页、关键词搜索和创建人/管理员过滤。

> ⚠️ **wsid 为必填参数**，调用前必须确保用户已提供工作空间 ID。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| wsid | int | ✅ | 工作空间 ID，必填，不能为 0 |
| keyword | string | 否 | 关键词，匹配字段包括：名称、描述、关联的推理服务名称、模型名称 |
| creator | string | 否 | 创建人/管理员 RTX，用于过滤该用户创建或管理的服务组 |
| page | int | 否 | 页码，从 1 开始，默认 1 |
| page_size | int | 否 | 每页数量，默认 20 |

**过滤规则：**
- **查询某人创建或管理的服务组** → 使用 `creator=RTX`（会匹配创建人或管理员）
- **按关键词模糊搜索**（匹配名称、描述、关联的推理服务名称、模型名称） → 使用 `keyword`

> ⚠️ **重要**：当用户意图是查询某个人创建/管理的服务组时，**必须使用 `creator` 参数**，**严禁**将人名/RTX 放入 `keyword` 参数进行模糊搜索。`keyword` 仅用于按服务组名称、描述等业务关键词搜索。

**💡 通过模型名称搜索：**
`keyword` 参数支持通过**模型名称**搜索关联的服务组。例如用户说"查找使用了 HunYuan-Large 模型的服务组"，直接传 `keyword="HunYuan-Large"` 即可。后端会先通过模型名称查找对应的模型（Mould），再反查关联的推理服务（Inference），最后匹配绑定了这些推理服务的服务组。


**返回（成功）：**
```json
{
  "count": 8,
  "results": [
    {
      "id": 456,
      "name": "my-service-group",
      "desc": "服务组描述",
      "creator": "zhangsan",
      "users": ["zhangsan", "lisi"],
      "create_time": "2026-03-01T10:00:00Z"
    }
  ]
}
```

**返回字段说明：**

| 字段 | 类型 | 描述 |
|------|------|------|
| count | number | 总记录数 |
| results | array | 服务组列表 |
| results[].id | number | 服务组 ID（编辑时需要使用） |
| results[].name | string | 服务组名称 |
| results[].desc | string | 服务组描述 |
| results[].creator | string | 创建人 |
| results[].users | array | 管理员列表 |
| results[].create_time | string | 创建时间 |

> ⚠️ **列表顺序强制规则**：`results` 数组已按后端默认排序（按 ID 降序，最新创建的排在前面）。展示时**必须严格按照数组原始顺序展示**，严禁重新排序、筛选或截断。
>
> 💡 **提示**：获取到 `id` 后，可以调用 `update_deploy_service` 编辑服务组信息，或使用 `name` 调用 `get_deploy_service_detail` 查看完整详情。

---

## update_deploy_service

编辑单个服务组的信息（部分更新），支持修改描述和管理员。

> ⚠️ **编辑操作前请先向用户确认变更内容**，确认后再调用此工具。
> ⚠️ `service_id` 可通过 `get_deploy_service_detail` 或 `list_deploy_services` 获取。
> ⚠️ **可编辑字段限制**：仅支持修改 `desc`（描述）和 `users`（管理员）。修改绑定的推理服务需通过 `create_deploy_service_change` 变更接口。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| service_id | int/string | ✅ | 服务组 ID |
| desc | string | 否 | 服务组描述（`desc`/`users` 至少提供其一） |
| users | array[string] | 否 | 服务组成员用户名列表（`desc`/`users` 至少提供其一） |
| wsid | int | 否 | 工作空间 ID，建议显式传入 |

> 💡 **v2 结构变更**：旧版通过 `data` JSON 字符串包装传参（如 `data='{"desc":"..."}'`），新版直接把 `desc` / `users` 平铺为**顶层参数**，无需再序列化 JSON。


**返回（成功）：**
```json
{
  "id": 456,
  "name": "my-service-group",
  "desc": "更新后的描述",
  "users": ["zhangsan", "lisi", "wangwu"],
  "creator": "zhangsan",
  "update_time": "2026-03-20T16:00:00Z"
}
```

**返回（权限不足，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 只有服务组创建人、服务管理员、空间管理员才有权限变更！",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

**返回（ID 不存在，HTTP 404）：**
```json
{
  "error": "API 请求失败 (HTTP 404): Not found.",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

---

## create_deploy_service_chat_completion

通过太极大模型平台的服务组调用大模型进行 OpenAI 兼容的文本对话。该接口通过服务组名称（`model` 参数）路由到对应的推理服务进行推理，返回 OpenAI 兼容格式的对话结果。

> ⚠️ MCP 工具默认使用**非流式模式**（`stream=false`）；如需强制显式关闭/开启流式，可通过 `stream` 参数传入。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| model | string | ✅ | 服务组名称，用于路由到对应的推理服务；不要传 `service_name` |
| messages | array | ✅ | OpenAI 兼容的消息列表，如 `[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "hello"}]` |
| query_id | string | 否 | 请求唯一标识；连通性测试建议显式传稳定短标识，便于排查日志 |
| wsid | int | 否 | 工作空间 ID，可选（v2 起放宽为选填）；已知时建议显式传入 |
| temperature | number | 否 | 采样温度，值越大输出越随机，不传则使用模型默认值 |
| top_p | number | 否 | 核采样参数，不传则使用模型默认值 |
| top_k | int | 否 | Top-K 采样参数，不传则使用模型默认值 |
| max_tokens | int | 否 | 最大输出 token 数，不传则使用模型默认值 |
| repetition_penalty | number | 否 | 重复惩罚系数，不传则使用模型默认值 |
| stream | boolean | ✅ | 是否开启流式输出（v2 新增） |
| chat_template_kwargs | object | 否 | 透传给底层推理框架 `chat_template` 的额外参数，常用于开关**思考模式**等推理行为。例如 `{"reasoning_effort": "high"}`、`{"enable_thinking": true}` 等。**⚠️ 上述示例参数不一定适用于所有模型**：不同模型/框架的 chat_template 支持的 kwargs 名称与取值不同，**调用方需自行确认目标模型是否支持所传参数**；模型不支持时可能被静默忽略或直接报错，由调用方负责。不传时使用底层默认配置。 |

**连通性测试标准调用（最高优先级）：**
```json
{
  "model": "<服务组名>",
  "wsid": 10103,
  "query_id": "probe_<服务组名>",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hello"}
  ]
}
```
连通性测试只需要一次 `create_deploy_service_chat_completion`。如果返回 `finish_reason=stop` 或有效 assistant 内容，即可判定服务组请求成功；如果返回 `ModelRouteError` / 上游非 JSON / 400/500，则原样汇报错误并给出服务组或实例状态排查建议，然后**立刻停止**，不要自动追加 `list_deploy_inferences` / `list_deploy_services` / 详情、实例、日志等工具链，除非用户继续要求排查。

**思考模式（chat_template_kwargs）调用示例：**

当用户希望**开启/关闭模型的思考模式**（如混元/Qwen 系列的深度推理开关）时，通过 `chat_template_kwargs` 透传对应 kwargs。

> ⚠️ **重要**：以下示例中的 `reasoning_effort` 只是**其中一种可能的参数**，**不一定适用于所有模型**。不同模型（混元 / Qwen / DeepSeek / GLM 等）及其对应的 chat_template 支持的 kwargs 名称、取值枚举都不同，**调用方（或使用 Agent 的用户）必须自行保证所传参数与目标模型匹配**。若不确定，请先向用户确认或查阅目标模型的 chat template 文档，不要凭空猜测参数名。

```json
{
  "model": "<服务组名>",
  "wsid": 10103,
  "query_id": "reasoning_high_test",
  "messages": [
    {"role": "user", "content": "证明勾股定理"}
  ],
  "chat_template_kwargs": {
    "reasoning_effort": "high"
  }
}
```

> 💡 **使用注意**：
> - `chat_template_kwargs` 是**纯透传字段**，本接口不做任何 kwargs 合法性校验；具体支持哪些 key 由底层推理框架/模型的 chat_template 决定（常见示例：`reasoning_effort`、`enable_thinking`、`thinking_budget` 等，仅供参考）。
> - **模型适配性由调用方负责**：所传的 kwargs 名称与取值必须与目标模型的 chat template 语义匹配；不同模型（混元 / Qwen / DeepSeek 等）支持的字段完全不同，示例参数并非通用。
> - **值必须严格符合底层框架预期**，例如某些模型的 `reasoning_effort` 接受 `"low" / "medium" / "high"` 字符串枚举，而某些模型可能用 `enable_thinking` 布尔值，请以目标模型文档为准。
> - 如果底层模型不支持所传 kwargs，可能被**静默忽略**或框架**直接报错**，需要按框架文档确认；报错时调用方原样汇报即可，不要自行猜测/修正 kwargs 名。
> - **切勿把 `reasoning_effort` 等字段拍平放到顶层**（如 `"reasoning_effort": "high"` 与 `model` 平级），必须放在 `chat_template_kwargs` 对象内。
> - **Agent 使用规范**：当用户明确要求"开启思考模式"但未指定字段名时，应先向用户确认使用哪个 kwargs（或让用户直接给出 `chat_template_kwargs` 内容），不要凭空猜测。

> 💡 **v2 参数变更提示**：`wsid` 由必填放宽为**选填**；新增 `stream` 布尔参数；新增 `chat_template_kwargs` 对象参数（透传底层 chat_template 额外 kwargs，用于开关思考模式等推理行为）。

**messages 格式说明：**

| 字段 | 类型 | 描述 |
|------|------|------|
| role | string | 消息角色：`system`（系统提示）、`user`（用户消息）、`assistant`（模型回复） |
| content | string | 消息内容 |


**返回（成功）：**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1706428720,
  "model": "HY2.0-406B-A32B-Instruct-FP8-251111-HFTest",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！有什么我可以帮助你的吗？"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

**返回字段说明：**

| 字段 | 类型 | 描述 |
|------|------|------|
| id | string | 对话完成的唯一标识 |
| object | string | 固定为 "chat.completion" |
| model | string | 使用的模型名称 |
| choices | array | 模型回复列表 |
| choices[].message.role | string | 固定为 "assistant" |
| choices[].message.content | string | 模型回复的文本内容 |
| choices[].finish_reason | string | 结束原因（stop/length 等） |
| usage | object | Token 使用统计 |
| usage.prompt_tokens | number | 输入 token 数 |
| usage.completion_tokens | number | 输出 token 数 |
| usage.total_tokens | number | 总 token 数 |

**返回（model 缺失）：**
```json
{
  "error": "服务组名称 (model) 不能为空",
  "hint": "请提供要调用的服务组名称，可通过 list_deploy_services 查询可用的服务组"
}
```

**返回（服务不存在，HTTP 500）：**
```json
{
  "error": "API 请求失败 (HTTP 500): Service matching query does not exist.",
  "hint": "请检查服务组名称是否正确，以及服务是否正常运行"
}
```

> 💡 **提示**：调用前建议先通过 `list_deploy_services` 确认可用的服务组名称。多轮对话时，需要将上一轮的模型回复作为 `assistant` 角色消息加入 `messages` 列表中。

---

## create_deploy_service

创建一个新的服务组，将一个或多个推理服务绑定到服务组中。服务组是太极大模型平台中推理服务的逻辑分组，用于统一管理和路由请求。

> ⚠️ **创建操作会在平台上新建服务组**，调用前必须向用户确认创建内容。
> ⚠️ 服务组名称和描述均为**全局唯一**，不能与已有服务组重复。

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| service_name | string | ✅ | 服务组名称，全局唯一 |
| wsid | int | ✅ | 工作空间 ID，必填，不能为 0 |
| desc | string | 否 | 服务组描述（业务上强烈建议传入） |
| inference_names | array[string] | 二选一 | 绑定的推理服务名称列表，简单创建方式 |
| service_items | array[object] | 二选一 | 高级绑定方式，逐项配置服务项（`name`/`inference_name`/`weight`/`enum`/`batch_size`/`switch_time`） |
| weight | array[object] | 否 | 卡型权重配置（`name` + `weight`）；不传时后端自动生成 |
| enable_context | boolean | 否 | 是否开启上下文能力 |
| enable_fast_reject | boolean | 否 | 是否开启快速拒答 |
| users | array[string] | 否 | 服务组成员用户名列表 |

> 💡 **v2 参数变更提示**：
> - 参数改名：`name→service_name`、`service_names`（逗号分隔字符串）→ `inference_names`（`array[string]`）
> - `desc` 由**必填**放宽为**选填**
> - 新增 `service_items`（高级绑定，与 `inference_names` 二选一）、`weight`（卡型权重，旧版由工具自动构建）、`enable_context`、`enable_fast_reject`、`users`

**service_items 元素结构（高级绑定用）：**

| 字段 | 类型 | 描述 |
|------|------|------|
| name | string | 服务项名称，通常与 `inference_name` 保持一致 |
| inference_name | string | 推理服务名称 |
| weight | int | 该服务项权重，默认 10 |
| enum | string | 服务项枚举值，默认 "2" |
| batch_size | string | 批大小，默认 "1" |
| switch_time | string | 切换时间，默认 "500" |

**调用示例（简单方式）：**
```json
{
  "service_name": "my-service-group",
  "wsid": 10103,
  "desc": "我的服务组描述",
  "inference_names": ["taiji-serving-test-v1"]
}
```

**绑定多个服务的示例：**
```json
{
  "service_name": "multi-service-group",
  "wsid": 10103,
  "desc": "绑定多个服务的服务组",
  "inference_names": ["service-a", "service-b"]
}
```

**高级绑定示例（自定义权重/批大小）：**
```json
{
  "service_name": "advanced-service-group",
  "wsid": 10103,
  "desc": "高级绑定示例",
  "service_items": [
    {"name": "svc-a", "inference_name": "svc-a", "weight": 6, "batch_size": "4"},
    {"name": "svc-b", "inference_name": "svc-b", "weight": 4, "batch_size": "2"}
  ],
  "weight": [
    {"name": "A800", "weight": 10}
  ],
  "enable_context": false,
  "enable_fast_reject": false
}
```


**返回（成功，HTTP 201）：**
```json
{
  "id": 789,
  "name": "taiji-serving-test-v1",
  "desc": "taiji-serving-test-v1",
  "data": [
    {
      "name": "taiji-serving-test-v1",
      "enum": "2",
      "weight": 10,
      "batch_size": "4",
      "switch_time": "500"
    }
  ],
  "wsid": 10103,
  "modifier": "zhangsan",
  "create_time": "2026-04-08 12:00:00",
  "update_time": "2026-04-08 12:00:00"
}
```

**返回（名称重复，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 服务组名称 xxx 已存在，请修改",
  "hint": "请检查服务组名称和描述是否重复，以及是否有空间成员权限"
}
```

**返回（描述重复，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 服务组描述 xxx 已存在，请修改",
  "hint": "请检查服务组名称和描述是否重复，以及是否有空间成员权限"
}
```

**返回（权限不足，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 非空间成员无法创建服务组！！",
  "hint": "请检查服务组名称和描述是否重复，以及是否有空间成员权限"
}
```

**返回（inference_names / service_items 均未提供）：**
```json
{
  "error": "inference_names 和 service_items 至少提供一种绑定方式",
  "hint": "请传入 inference_names（简单方式）或 service_items（高级方式）"
}
```

> 💡 **提示**：创建服务组前，建议先通过 `list_deploy_inferences` 查询可用的推理服务名称。创建成功后，可以通过 `create_deploy_service_chat_completion` 使用该服务组进行大模型对话，或通过 `get_deploy_service_detail` 查看完整详情。

### 字段限制

| 字段 | 最大长度 | 格式要求 | 说明 |
|------|---------|---------|------|
| name | 255 字符 | 无特殊格式要求 | 服务组名称，全局唯一 |
| desc | 255 字符 | 无特殊格式要求 | 服务组描述 |

---

## create_deploy_service_change

创建服务组变更单，修改服务组绑定的推理服务列表或权重配置。**MCP 调用会自动按当前用户跳过审批环节**（后端根据 `request_source=mcp` 自动处理）。

> ⚠️ **变更操作会影响线上服务组路由**，调用前必须向用户二次确认变更内容！
> ⚠️ **权限要求**：仅服务组创建人或管理员可发起变更。权限不足时返回错误。
> 💡 **v2 结构变更**：旧版需要先调用 `init_service_deployment` 获取快照、再拼装 `old_data`/`new_data` 一并提交；新版**后端会自动补齐内部所需快照字段**，客户端只需要传"想变更成什么"即可。

**参数：**

🔴 用户必须提供的参数：

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| wsid | int | ✅ | 工作空间 ID，必填，不能为 0 |
| service_id | int/string | 二选一 | 服务组 ID，可通过 `list_deploy_services` 获取 |
| service_name | string | 二选一 | 服务组名称，`service_id` 未传时可用于定位服务组 |

🟡 变更内容参数（`new_inference_names` / `new_service_items` 至少提供其一）：

| 参数 | 类型 | 说明 |
|------|------|------|
| new_inference_names | array[string] | 新的推理服务名称列表，简单变更方式 |
| new_service_items | array[object] | 高级变更方式，逐项配置服务项（`name`/`inference_name`/`weight`/`enum`/`batch_size`/`switch_time`） |
| new_weight | array[object] | 新的卡型权重配置（`name` + `weight`），不传时后端保持不变或自动生成 |

🟢 变更单元数据参数：

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| deployment_name | string | 否 | 自定义变更单名称，不传则后端自动生成 |
| desc | string | 否 | 变更单描述 |
| approver | array[string] | 否 | 审批人用户名列表，不传时后端按当前请求用户自动补齐 |

**new_service_items 元素结构：**

| 字段 | 类型 | 描述 |
|------|------|------|
| name | string | 服务项名称，通常与 `inference_name` 保持一致 |
| inference_name | string | 推理服务名称 |
| weight | int | 该服务项权重，默认 10 |
| enum | string | 服务项枚举值，默认 "2" |
| batch_size | string | 批大小，默认 "1" |
| switch_time | string | 切换时间，默认 "500" |

### 服务组配置字段速查表（后端自动保留的字段）

以下字段在**服务组的当前配置快照**中，若本次变更未显式修改，后端会自动保留原值：

| 字段 | 类型 | 描述 |
|------|------|------|
| data | array | 当前绑定的推理服务列表（每个服务含 name、weight、batch_size、switch_time、enum） |
| weight | array | 卡型权重配置（GPU 卡型及权重） |
| enable_context | string | 上下文开关 |
| enable_fast_reject | boolean | 快速拒绝开关 |
| weight_type | string | 权重类型 |
| reasoning_parser | string | 推理解析配置 |
| openapi_registered | boolean | OpenAPI 注册状态 |

> 💡 **提示**：如果需要变更 `enable_context`、`enable_fast_reject`、`weight_type` 等字段，请前往太极大模型平台 Web 界面操作；MCP 工具目前仅暴露了推理服务列表和卡型权重的变更能力。


**调用示例（简单方式）：**
```json
{
  "wsid": 10103,
  "service_id": 789,
  "new_inference_names": ["service-a", "service-b"]
}
```

**调用示例（高级方式，自定义权重）：**
```json
{
  "wsid": 10103,
  "service_id": 789,
  "new_service_items": [
    {"name": "service-a", "inference_name": "service-a", "weight": 8},
    {"name": "service-b", "inference_name": "service-b", "weight": 2}
  ],
  "new_weight": [{"name": "H20", "weight": 10}]
}
```

**返回（成功）：**
```json
{
  "id": 12345,
  "name": "service_group_789_change_20260421170000",
  "status": "approved",
  "service_id": 789
}
```

**返回（权限不足，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 只有服务组创建人、服务管理员、空间管理员才有权限变更！",
  "hint": "请检查：1) 是否有服务组变更权限（需为创建人或管理员）；2) 是否存在未终结的发布单；3) 推理服务名称是否正确"
}
```

**返回（存在未终结的发布单，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 存在未终结的发布单，请先完成或取消",
  "hint": "请检查：1) 是否有服务组变更权限（需为创建人或管理员）；2) 是否存在未终结的发布单；3) 推理服务名称是否正确"
}
```

**返回（new_inference_names / new_service_items 均未提供）：**
```json
{
  "error": "new_inference_names 和 new_service_items 至少提供一种变更方式",
  "hint": "请传入 new_inference_names（简单方式）或 new_service_items（高级方式）"
}
```

> 💡 **提示**：创建成功后，变更单已自动审批通过（MCP 调用自动跳过审批）。可以调用 `get_deploy_service_detail` 查看服务组最新状态。

---

## create_deploy_inference_change

创建推理服务变更单（实例扩缩容或实例重启）。**MCP 调用会自动按当前用户跳过审批环节**（后端根据 `request_source=mcp` 自动处理）。

> ⚠️ **变更操作会影响线上服务**，调用前必须向用户二次确认变更内容！
> ⚠️ **权限要求**：仅服务创建人或管理员可发起变更。权限不足时返回错误。
> 💡 **v2 结构变更**：旧版需要先调用 `init_deployment` 拿完整快照、再拼装十几个字段一起提交；新版**后端会自动补齐内部所需快照字段**，客户端只需要传"想变更成什么"即可（`replicas` 或 `instance_ids`）。

### 两个核心场景的参数分类

**场景一：实例扩缩容（`change_type="scale"`）**

🔴 用户必须提供的参数：

| 参数 | 类型 | 描述 |
|------|------|------|
| inference_name | string | 服务名称（与 `inference_id` 二选一） |
| inference_id | int/string | 服务 ID（与 `inference_name` 二选一） |
| replicas | int | 目标副本数 |

🟢 无需手动传入的字段：`change_type` 后端会根据"最外层传了 `replicas`"自动推断为 `"scale"`；`request_source=mcp` 由 MCP 工具自动带上；`deployment_name` 后端自动生成。

**场景二：实例重启（`change_type="restart"`）**

🔴 用户必须提供的参数：

| 参数 | 类型 | 描述 |
|------|------|------|
| inference_name | string | 服务名称（与 `inference_id` 二选一） |
| inference_id | int/string | 服务 ID（与 `inference_name` 二选一） |
| instance_ids | array[string] | 待重启实例 ID 列表，可先通过 `list_deploy_instances` 查看 |

🟢 无需手动传入的字段：`change_type` 后端会根据"最外层传了 `instance_ids`"自动推断为 `"restart"`；`request_source=mcp` 自动带上；`deployment_name` 自动生成。

### 完整参数表

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| wsid | int | ✅ | 工作空间 ID，必填，不能为 0 |
| inference_id | int/string | 二选一 | 推理服务 ID |
| inference_name | string | 二选一 | 推理服务名称 |
| change_type | string | 否 | 变更类型枚举值：`"scale"`（扩缩容） / `"restart"`（实例重启）；不传时后端根据最外层 `replicas` / `instance_ids` **自动推断** |
| replicas | int | 条件必填 | 目标副本数；用于 `scale` 变更 |
| instance_ids | array[string] | 条件必填 | 待重启实例 ID 列表；用于 `restart` 变更 |
| deployment_name | string | 否 | 自定义变更单名称，不传则后端自动生成 |
| desc | string | 否 | 变更单描述 |
| approver | array[string] | 否 | 审批人用户名列表；未传时后端会按当前请求用户自动补齐 |

> 💡 **v2 参数变更提示**：
> - 参数改名：`inference→inference_id`、`new_replicas→replicas`、`restart_instance_ids`（逗号分隔字符串）→ `instance_ids`（`array[string]`）
> - `change_type` 由**整数枚举**（`1`=实例重启，`2`=实例扩缩容）改为**字符串枚举**（`"scale"` / `"restart"`），并且**由必填放宽为选填**（后端可根据 `replicas` / `instance_ids` 自动推断）
> - **移除**：`follow`（关注人列表）、`name`（改名为 `deployment_name`）、`instance_change` / `new_instance_change_type` 等内部字段（后端自动补齐）
> - **回滚初始化**：v2 不再暴露 `init_deployment`；如需回滚请前往太极大模型平台 Web 界面操作

### 推理服务配置字段速查表（后端自动保留的字段）

以下字段在**推理服务的当前配置快照**中，若本次变更未显式修改，后端会自动保留原值：

| 字段 | 类型 | 描述 |
|------|------|------|
| inference_name | string | 服务名称 |
| replicas | int | 当前副本数 |
| image | string | 镜像地址 |
| triton | object | Triton 相关配置 |
| tokenizer | object | Tokenizer 相关配置 |
| hpc | object | HPC 集群相关配置 |
| gpu_name | string | GPU 卡型 |
| location | string | 部署地域 |
| envs | object | 环境变量键值对 |
| start_command | string | 启动命令 |

> 💡 **提示**：如果需要变更镜像、启动命令、环境变量等配置，请前往太极大模型平台 Web 界面操作；MCP 工具目前仅暴露了扩缩容和实例重启两种变更能力。


**扩缩容调用示例：**
```json
{
  "wsid": 10362,
  "inference_name": "test_7b_moe_32k_zesen_v4",
  "replicas": 3
}
```

**实例重启调用示例：**
```json
{
  "wsid": 10362,
  "inference_name": "test_7b_moe_32k_zesen_v4",
  "instance_ids": ["taiji-serving-custom-27498-0"]
}
```

**显式指定变更类型示例：**
```json
{
  "wsid": 10362,
  "inference_id": 500161,
  "change_type": "scale",
  "replicas": 5,
  "deployment_name": "manual-scale-up-2026Q3",
  "desc": "扩容以应对业务高峰"
}
```

**返回（成功）：**
```json
{
  "id": 12345,
  "name": "test_7b_moe_32k_zesen_v4_scale_20260804213119",
  "change_type": "scale",
  "status": "approved",
  "inference_name": "test_7b_moe_32k_zesen_v4"
}
```

> 🔗 返回的 `id` 即为 `deployment_id`，**直接传给 `stop_deploy_inference_change` 即可终止变更单**，不需要再调 `list_deploy_instances` 或其他工具查找。

**返回（参数缺失 — 服务标识为空）：**
```json
{
  "error": "服务名称 (inference_name) 和服务 ID (inference_id) 不能同时为空",
  "hint": "请提供服务名称或服务 ID，可通过 get_deploy_inference_detail 查询获取"
}
```

**返回（未提供 replicas / instance_ids）：**
```json
{
  "error": "扩缩容变更需要提供 replicas，实例重启变更需要提供 instance_ids",
  "hint": "请检查最外层是否传入 replicas（用于 scale）或 instance_ids（用于 restart）"
}
```

**返回（权限不足，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 非服务创建人或服务管理员，无权限发起变更！！",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

**返回（变更单并行度冲突，HTTP 400）：**
```json
{
  "error": "API 请求失败 (HTTP 400): 该服务已有进行中的扩缩容变更单，请先完成或取消",
  "hint": "请检查本地 Token 是否有效，以及是否有权限访问该接口"
}
```

> 🛑 **错误与冲突处理规则（最高优先级）**：调用失败（权限不足、并行度冲突、HTTP 400/500 等）时，**立即停止**，把错误原文告知用户，**不得**：
> - 重复调用本工具重试
> - 换工具名/参数反复尝试
> - 查看 connect_mcp.py 源码寻找绕过方式
> - 写脚本或搜索其他替代方案
>
> 正确做法："创建变更单失败：该服务已有进行中的扩缩容变更单，请先完成或取消后再试。"

> 💡 **提示**：创建成功后，变更单已自动审批通过（MCP 调用自动跳过审批）。可以调用 `get_deploy_inference_detail` 查看服务最新状态，或调用 `list_deploy_instances` 查看实例列表。若需要提前中止一个已下发的变更单，可以调用 `stop_deploy_inference_change`。

---

## list_deploy_templates

查询官方推理模板列表（`template_inference_angelhcf`），对应前端「从模板创建」的 baseline tab。与快速部署流程**独立**，不参与 Step 流程。

> 💡 **v2 参数变更提示**：`wsid` 由**可选变为必填**；新增可选筛选参数 `manufacturer` / `manufacturer_series` / `parameter_size` / `template_type` / `model_id` / `lora_model_id` / `app_group_id` / `env`。旧版通过 `wsid=None` 拉全量的做法**不再支持**，必须显式传入某个空间视角的 `wsid`。
>
> 📌 此处模板特指 **部署模板 / 推理模板 / 在线推理模板**（用于部署在线模型服务），**不是训练模板 / 微调模板**。若用户要的是训练模板，参见 SKILL.md §4.8「模板术语对齐」。

### ⛔ 不支持的参数（严禁幻觉）

本工具**仅支持**下方「支持筛选」表中的参数。下列参数**全部不存在**，传了会被后端**静默丢弃返回全量**，给人"参数生效但没匹配"的假象：

| ❌ 不要传 | ✅ 正确做法 |
|----------|------------|
| `keyword` / `search` / `q` / `name` / `model_name` | 用 `manufacturer` + `manufacturer_series` 等枚举字段精确过滤 |
| ~~`mould_id`~~（旧参数名） | v2 已改名为 `model_id`，仍可作为**筛选**字段传入 |
| 拿到 `model_id` 后先 list 再 get_detail | 已知 `model_id` 应直接调 `get_deploy_template_detail`，不需要先列表 |
| `dsv4` / `deepseek-v4` 等模型代号当作参数值 | 模型代号属厂商系列 → `manufacturer="DeepSeek"`，再按 `parameters` / `gpu_name` 细化 |
| `manufacturer_series="V4"` 这种短前缀 | 后端是**完整串匹配（不支持前缀）**，必须传完整串如 `"V4-Flash"` / `"V4-Pro"`，否则返回 0 条。不知道完整串时先只过滤 `manufacturer` 看返回的真实系列 |

> ⚠️ "查 dsv4 / DeepSeek-V4 模板" 的正确写法是 **`{"manufacturer": "DeepSeek"}`** 拉全量 11 条，再在客户端按 `manufacturer_series` 中包含 `"V4-"` 来筛 V4 系列；**不是** `keyword="dsv4"` 也**不是** `manufacturer_series="V4"`。

### 关于 `wsid` 参数

> 🔧 **`wsid` 不是筛选过滤参数，而是"可见性视角"参数**（语义不同于其它列表类工具）：
> - **不传 wsid**：以默认视角返回所有"跨空间公共模板"
> - **传 wsid**：以该空间视角返回"公共模板 + 该空间私有模板"，**结果数通常 ≥ 不传**
> - 传无权限/不存在的 wsid：会失败（HTTP 错误）
>
> 实测对比（pre 环境）：`{}` → 109 条；`{"wsid": 11331}` → 109 条；`{"wsid": 10103}` → 147 条（多 38 条私有模板）。
>
> 何时传 wsid：① 用户明确要看某空间下可用的全部模板（含私有）② 准备走快速部署，要确认目标空间能复制到模板；何时不传 wsid：纯粹"查有什么官方模板"或多空间通用调研。

### 支持筛选

> 📌 **本工具所有筛选字段都是「完整串匹配 + 大小写不敏感」**：
> - ✅ 大小写无所谓：`"DeepSeek"` / `"deepseek"` / `"DEEPSEEK"` 都返回相同结果
> - ❌ **不支持前缀/子串**：传 `"V4"` 不会命中 `"V4-Flash"` / `"V4-Pro"`，必须传完整串
> - ❌ **不支持模糊**：没有 `keyword` / `search` / `q` 参数（见上一节「⛔ 不支持的参数」）

| 参数 | 说明 |
|------|------|
| `gpu_name` | GPU 卡型。常见取值：`"H20"` / `"P800"` / `"A800"` / `"MLU590"` / `"H800"` |
| `service_scene` | 服务场景。常见取值：`"text_to_text"` / `"multimodal"` / `"audio"` / `"text_to_image"` |
| `framework_type` | 推理框架。常见取值：`"开源vLLM"` / `"开源SGLang"` / `"vLLM_v2.1"` |
| `manufacturer` | 厂商。**中英文混编**，见下方速查表 |
| `manufacturer_series` | 厂商系列。**命名风格不统一**，见下方速查表 |
| `parameters` | 参数量。常见取值：`"13B"` / `"284B"` / `"A13B"` / `"A32B"` / `"A49B"` / `"1T"` |

### ⚠️ manufacturer / manufacturer_series 真实枚举速查表

> 🔴 **极易踩坑**：这两个字段是**后端精确匹配**（不是模糊/前缀匹配），且命名风格混乱：
> - `manufacturer` 是**中英文混编**（`通义千问`、`混元`、`智谱` 是中文；`DeepSeek`、`Google`、`Kimi` 是英文/拼写名）
> - 同厂商的 `manufacturer_series` **命名风格不统一**（如 DeepSeek 既有 `V4-Flash`、`V4-Pro`，又有 `DeepSeek-V3.2`、`OCR`）
> - **不能用** `"V4"` / `"v4"` 这种短前缀，必须传完整串
> - 不确定取值时：**先用 `manufacturer` 拉一遍**（结果通常 < 30 条），从返回的 `manufacturer_series` 字段聚合得到该厂商的真实系列名，再决定下一步

| manufacturer（精确值）| 典型 manufacturer_series 取值 |
|---|---|
| `通义千问` | `Qwen3` / `Qwen3.5` / `Qwen3.6` |
| `混元` | `Bayberry` / `HY1.0` / `HY2.0` / `HY3.0` / `FP8` |
| `智谱` | `GLM-4.7` / `GLM-5` / `GLM-5.1` / `GLM-5.2` / `OCR` |
| `DeepSeek` | `V4-Flash` / `V4-Pro` / `DeepSeek-V3.1` / `DeepSeek-V3.2` / `OCR` |
| `Google` | `Gemma4` |
| `Kimi` | `K2` / `K2.5` / `K2.6` / `K2.7` |
| `小米` | `MiMo-V2.5` / `MiMo-V2-Flash` |
| `PaddlePaddle` | `OCR` |
| `MiniMax` | `MiniMax2.5` / `M2.7` |
| `Infly` | `Infinity-Parser2` |
| `xAI` | `xAI` |
| `Open AI` | `gpt` |
| `百度` | `Qianfan` |

> 📅 该表为快照参考（截至 2026-06）。**最新真实值以 `list_deploy_templates` 不带 manufacturer_series 过滤后的实际返回为准**——若表中某厂商的新系列已经上线，按返回字段原样使用。

### 交互规范（🔴 硬约束）

1. **结果 ≥ 30 条时必须先追问，不可直接展示**：
   - **严禁**截断只展示前 N 条，**严禁**自己用幻觉参数（如 `keyword`）二次过滤
   - 必须输出固定结构：① 总数 + ② 可筛选维度 + 实际命中的取值 + ③ 显式问句

   示例：
   > 共返回 **109 条**官方推理模板。请选择筛选维度（择一或组合）：
   > — **厂商** `manufacturer`：deepseek（11 条）/ qwen / Kimi / GLM / ...
   > — **GPU 卡型** `gpu_name`：H20 / P800 / A800 / MLU590 / ...
   > — **场景** `service_scene`：text_to_text / multimodal / audio
   > — **参数量** `parameters`：13B / 284B / A32B / ...
   >
   > 请告诉我按哪个维度筛选？

2. **结果 < 30 条**：可直接列表呈现关键字段（`id` / `inference_name` / `mould_id` / `gpu_name` / `host_num × host_gpu_num` / `framework_type` / `parameters`）。

### 典型用法

- 查某厂商某卡型是否有模板：`{"manufacturer": "deepseek", "gpu_name": "A800"}`
- 查 DeepSeek 系列全部模板（先看再聚合）：`{"manufacturer": "deepseek"}`
- 查某卡型下所有模板模型：`{"gpu_name": "H20"}` → 从 `mould_id`/`mould_name` 聚合

### 与快速部署衔接

查询结果中选定 `model_id` 后，可进入快速部署流程（从 `## 快速部署` Step 1 传入该 model_id 继续）。

---

## get_deploy_template_detail

获取指定模型的推理部署模板详情，返回创建推理服务所需的全部默认配置（镜像、端口、环境变量等），是「快速部署」流程的核心工具。

> 💡 **v2 参数变更提示**：
> - 参数改名：`mould_id → model_id`、`queue_name → app_group_id`（后者传的是应用组 **ID**，不是应用组名称）
> - `wsid` **由可选变为必填**
> - 新增可选参数 `lora_model_id`、`optimizer_level`
> - 推理框架类型（`framework_type`）由后端自动选择，通常无需用户指定

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| model_id | int/string | ✅ | 模型 ID，可通过 `search_hunyuan_models_cards` 获取 |
| wsid | int | ✅ | 工作空间 ID |
| lora_model_id | int/string | 否 | LoRA 模型 ID |
| inference_type | string | 否 | 推理类型，默认 "inference" |
| service_scene | string | 否 | 服务场景：text_to_text / image_to_text / audio / multimodal |
| gpu_name | string | 否 | GPU 卡型（交叉匹配得出） |
| app_group_id | string | 否 | 应用组 ID（⚠️ 必须来自用户有权限的应用组） |
| framework_type | string | 否 | 推理框架（通常无需指定，后端自动选择） |
| location | string | 否 | 地域缩写（交叉匹配得出） |
| env | string | 否 | 环境类型，默认 formal |
| optimizer_level | string | 否 | 优化等级 |


**返回（成功）：**
```json
{
  "image_name": "taiji_serving_vllm_dsv4_20260525.post2",
  "framework_type": "vLLM_v2.1",
  "env_vars": {},
  "ports": []
}
```

**返回（image_name 为空）：** 表示未匹配到模板，需提示用户提供自己的模板服务。
