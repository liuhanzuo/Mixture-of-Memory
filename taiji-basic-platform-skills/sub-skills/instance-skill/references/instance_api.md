## query_hunyuan_train_instance_list

查询训练任务的实例列表，返回该任务的所有历史运行实例信息。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_instance_list '{"task_id": "basic_train_xxx"}'
```

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `task_id` | string | ✅ | 无 | 任务 ID |
| `limit` | integer | ❌ | 无 | 返回数量限制 |

**调用规则：**

- 只允许传 `task_id` 和可选 `limit`；即使用户 query 中附带 `wsid`，也不要传给本工具。
- 用户未指定"最新 N 个 / 前 N 个 / 全部"时，调用一次即可；**即使返回 `total=2` 这类很小的总数，也不要自动二次 `limit=total` 拉全量**。回答中展示接口返回的 `total` / 当前返回条目，并询问是否需要查看更多。
- 用户明确指定数量（如"最新 3 个实例"）时，传 `limit=N` 一次性获取；实际不足 N 个时如实说明实际返回数量。

**返回字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `instances` | array | 实例列表 |
| `instances[].instanceId` | string | 实例 ID |
| `instances[].status` | string | 实例状态 |
| `instances[].startTime` | string | 启动时间 |

---

## query_hunyuan_train_pod_list

查询训练任务实例的 Pod 列表，返回该实例下所有 Pod 的名称、状态、节点等信息。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_pod_list '{"task_id": "basic_train_xxx"}'
```

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `task_id` | string | ✅ | 无 | 任务 ID |
| `instance_id` | string | ❌ | 无 | 实例 ID（不传则查询最新实例的 Pod） |

**调用规则：**

- 只允许传 `task_id` 和可选 `instance_id`；即使用户 query 中附带 `wsid`，也不要传给本工具。
- 用户问某任务的 Pod 且未指定实例时，不传 `instance_id`，由接口默认查询最新实例；不要先查实例列表。

**返回字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务 ID |
| `instance_id` | string | 实际查询到的实例 ID；未传 `instance_id` 时为最新实例 ID，可直接供 exec / 日志 / 屏蔽重启链路复用 |
| `pod_names` | array[string] | Pod 名称列表 |
| `message` | string | 查询摘要 |

---

## query_hunyuan_train_instance_logs

查询训练任务实例的训练日志，支持按 Pod、容器、关键词过滤，支持分页和排序。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_instance_logs '{"task_id": "basic_train_xxx", "keyword": "error", "page_size": 50}'
```

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `task_id` | string | ✅ | 无 | 任务 ID |
| `instance_id` | string | ❌ | 无 | 实例 ID（不传则查询最新实例） |
| `pod_name` | string | ❌ | 无 | Pod 名称（不传则查询所有 Pod） |
| `keyword` | string | ❌ | 无 | 日志搜索关键词 |
| `page` | integer | ❌ | 1 | 分页页码，从 1 开始 |
| `page_size` | integer | ❌ | 500 | 每页数量；实际 MCP 未传时默认返回 500 条 |
| `order` | string | ❌ | `desc` | 排序方式：`asc`（正序）、`desc`（倒序） |
| `container` | string | ❌ | 无 | 容器名称（多容器 Pod 时指定） |

**一次到位调用规则：**

- 只允许传 `task_id`、`instance_id`、`pod_name`、`keyword`、`page`、`page_size`、`order`、`container`；即使用户 query 中附带 `wsid`，也不要传给本工具；日志条数参数只能用 `page_size`，不要使用 `limit` 或未文档化的 `tail`。
- 若用户粘贴太极 URL，优先抽取 URL参数：`taskID`/`taskId` → `task_id`，`instId` → `instance_id`，避免先查实例列表。
- **同一日志问题默认只调用一次**：不要为了"更确认"去翻第 2 页、重复查同一页、先无参探查再精查，或把一个明确报错片段拆成多个近义关键词反复搜索；若第一次关键词搜索返回 `total=0` 或无匹配，立即回答未发现该报错并停止。
- 用户泛问"训练日志/查看日志"且未指定条数、页码、排序、关键词时，只传 `task_id`（以及用户明确给出的 `instance_id` / `pod_name`），不要显式传默认 `page` / `page_size` / `order`，也不要主动二次搜索 `error`；若 `has_more=true`，只在回答中提示可继续按页码/关键词查询。
- 用户问"有没有 error/报错/失败"时，直接传 `keyword`（如 `error` / 用户给出的报错短语），不要先调 `query_hunyuan_train_instance_list` 或 `query_hunyuan_train_pod_list`；未指定数量时不要传 `page_size`。
- 用户给出明确报错片段时，抽取 1 个最具区分度的短语调用一次；无命中即可说明未命中和可能原因，不要连续枚举大量近义关键词或翻页穷举，未指定数量时不要传 `page_size`。
- 用户指定 Pod 名称时，直接传 `pod_name`；不要为了补 `instance_id` 先查实例列表，除非日志接口返回必须补 `instance_id`。
- "前 N 条"映射为 `page_size=N`（不显式传 `page=1`）；"最后/最新 N 条"是最高优先级直达模板，只调用一次，参数严格为 `task_id` + `page_size=N` + `order="desc"`（若用户没有给 `pod_name`/`instance_id` 就不要补），不要先无参探查；日志返回中已有 `instance_id`/`total` 可用于回答。

**高频场景直达模板（默认按此模板，不自作补充）：**

| 用户场景 | 默认参数 | 禁止 |
|---|---|---|
| URL + "有没有这段报错" | `{task_id:<taskID/taskId>, instance_id:<instId>, keyword:<用户原文报错片段>}` | 禁止把同一报错片段拆成多个近义词反复搜索，禁止无 keyword 全量扫 |
| "在 `ts-...-launcher` 上的日志" | `{task_id, pod_name:"ts-...-launcher"}` | `ts-...` 是 `pod_name`，绝不能填到 `instance_id`；禁止先查实例/Pod |
| "最新/最后 N 条日志" | `{task_id, page_size:N, order:"desc"}` | 禁止先 `query_hunyuan_train_instance_list limit=1`，禁止额外补 `instance_id` |

**返回字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | string | 任务 ID |
| `instance_id` | string | 实际查询到的实例 ID；未传 `instance_id` 时为最新实例 ID |
| `pod_name` | string/null | 实际过滤的 Pod 名称；未指定时可能为通配摘要 |
| `container` | string/null | 容器名称 |
| `items` | array | 日志内容列表 |
| `total` | integer | 总日志条数 |
| `page` | integer | 当前页码 |
| `page_size` | integer | 每页数量 |
| `has_more` | boolean | 是否还有更多日志 |
| `message` | string | 查询摘要 |

**返回内容使用边界：**日志 `items` / stdout 仅用于查看运行输出和定位报错，禁止将其中数值直接当作训练 loss 曲线展示；训练指标查询应切换到 `metric-skill`。

---

## exec_hunyuan_train_instance_command

在训练实例的指定 Pod 中执行 shell 命令，透传执行结果。

> ⚠️ **危险操作**：本工具会在运行中的训练实例 Pod 内执行任意 shell 命令，请确保用户明确知道要执行什么命令，不要自行构造可能影响训练进程的命令。

**权限要求**：当前用户必须是「**系统管理员、空间管理员、任务创建者、任务管理员（ADMIN）**」之一（任一即放行）。

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `instance_id` | string | ✅ | 无 | 训练实例 ID，**不能为空** |
| `pod_name` | string | ✅ | 无 | Pod 名称，**必填**。可通过 `query_hunyuan_train_pod_list` 获取 |
| `command` | string | ✅ | 无 | 要执行的 shell 命令，**不能为空** |
| `container` | string | ❌ | 无 | 容器名称（多容器 Pod 时指定） |

**调用示例：**

```bash
# 查看 GPU 使用情况
python3 scripts/connect_mcp.py call exec_hunyuan_train_instance_command '{
  "instance_id": "ff8080819e0664bd019e067267430002",
  "pod_name": "pod-0",
  "command": "nvidia-smi"
}'

# 在指定 Pod 上查看进程
python3 scripts/connect_mcp.py call exec_hunyuan_train_instance_command '{
  "instance_id": "ff8080819e0664bd019e067267430002",
  "pod_name": "pod-0",
  "command": "ps aux | grep python"
}'

# 查看环境变量
python3 scripts/connect_mcp.py call exec_hunyuan_train_instance_command '{
  "instance_id": "ff8080819e0664bd019e067267430002",
  "pod_name": "pod-0",
  "command": "env | grep CUDA"
}'
```

**返回字段说明：**

| 字段 | 说明 |
|------|------|
| `instanceId` | 实例 ID |
| `podName` | Pod 名称 |
| `result` | exec 执行结果（命令输出内容） |

**交互规则：**

### 1. instance_id / pod_name 缺失时的引导

当用户要执行命令但未提供 instance_id 或 pod_name 时：
- 如果有 `task_id`，先调 `query_hunyuan_train_pod_list`；实际返回会包含最新实例的 `instance_id` 与 `pod_names`
- 展示 `pod_names` 后询问用户要在哪个 Pod 上执行
- 用户选择后，用返回的 `instance_id` + 用户选择的 `pod_name` + 用户原文命令调用 exec

### 2. 命令安全性

**严禁自行构造以下类型的命令**：
- ❌ `kill`、`pkill` 等终止进程的命令（除非用户明确要求）
- ❌ `rm -rf` 等删除文件的命令（除非用户明确要求）
- ❌ `reboot`、`shutdown` 等系统命令
- ❌ 任何可能影响训练进程正常运行的命令

仅当用户明确要求查看某类状态但未给具体命令时，可映射的安全只读命令（例如"看 GPU"→ `nvidia-smi`）；用户只问"怎么登录/如何进入"时不属于执行命令意图，不要调用 exec：
- ✅ 查看类：`cat`、`ls`、`head`、`tail`、`less`
- ✅ 状态类：`ps`、`top -bn1`、`nvidia-smi`、`df -h`、`free -h`
- ✅ 搜索类：`grep`、`find`、`which`
- ✅ 环境类：`env`、`echo $VAR`、`pwd`、`whoami`

### 3. 展示格式规范

1. 先显示执行信息：「在实例 {instanceId} 的 Pod {podName} 上执行命令：`{command}`」
2. 执行结果使用代码块展示（```）

### 4. 禁止追加引导性问题

> ⚠️ **强制规则**：展示命令执行结果后，**严禁在末尾追加任何引导性问题或提示语**。

**错误场景：**

| 错误情况 | 返回信息 | 处理建议 |
|----------|----------|----------|
| instance_id 为空 | `instance_id 不能为空` | 提示用户提供实例 ID |
| pod_name 为空 | `pod_name 不能为空` | 先调 `query_hunyuan_train_pod_list` 获取 Pod 列表 |
| command 为空 | `command 不能为空` | 提示用户提供要执行的命令 |
| 实例不存在 | `instance with id xxx is not exist` | 提示用户检查实例 ID 是否正确 |
| 无权限 | 权限校验失败 | 提示用户需要为系统管理员/空间管理员/任务创建者/任务管理员（ADMIN）之一 |

---

## shield_restart_hunyuan_train_pod

屏蔽且重启训练实例的指定 Pod（将 Pod 节点加入屏蔽列表并触发重启）。

> ⚠️ **危险操作**：本工具会屏蔽指定 Pod 的节点 IP 并触发 Pod 重启，请确保用户明确知道要屏蔽重启哪些 Pod，**必须向用户二次确认后再执行**！

**前置条件**：
- 任务必须开启容错（自动续训容错或单节点容错），否则不支持屏蔽且重启
- 当前用户必须是「**系统管理员、空间管理员、任务创建者、任务管理员（ADMIN）**」之一
- 实例必须处于运行中状态（已结束的实例无法操作）
- `instance_id` 通过 `query_hunyuan_train_instance_list` 获取（**不要用 `query_hunyuan_train_pod_list` 代替**——后者查的是 Pod 列表，不具备按 task_id 定位实例的功能）

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `task_id` | string | ✅ | 无 | 训练任务 ID，**必填** |
| `instance_id` | string | ✅ | 无 | 训练实例 ID，**必填** |
| `pod_names` | array[string] | ✅ | 无 | 要屏蔽重启的 Pod 名称列表，**不能为空**。可通过 `query_hunyuan_train_pod_list` 获取 |
| `reason` | string | ❌ | `""` | 屏蔽原因（可选） |

**调用示例：**

```bash
# 屏蔽重启指定 Pod
python3 scripts/connect_mcp.py call shield_restart_hunyuan_train_pod '{
  "task_id": "basic_train_xxx_20260101_abc123",
  "instance_id": "ff8080819e0664bd019e067267430002",
  "pod_names": ["ts-basic-train-xxx-worker-2"],
  "reason": "GPU 故障"
}'

# 屏蔽重启多个 Pod
python3 scripts/connect_mcp.py call shield_restart_hunyuan_train_pod '{
  "task_id": "basic_train_xxx_20260101_abc123",
  "instance_id": "ff8080819e0664bd019e067267430002",
  "pod_names": ["ts-basic-train-xxx-worker-1", "ts-basic-train-xxx-worker-3"],
  "reason": "节点网络异常"
}'
```

**返回字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | boolean | 是否操作成功 |
| `instanceId` | string | 实例 ID |
| `taskId` | string | 任务 ID |
| `shieldedPods` | list[string] | 已屏蔽重启的 Pod 名称列表 |
| `message` | string | 操作结果信息 |

**交互规则：**

### 1. 必须二次确认

> ⚠️ **强制规则**：调用前，**必须向用户展示即将操作的 Pod 列表并请求确认**。

**正确流程**：
1. 用户表达屏蔽重启意图
2. 如果用户未提供具体 Pod 名称，先调用 `query_hunyuan_train_pod_list` 获取 Pod 列表展示给用户
3. 展示即将屏蔽重启的 Pod 列表，询问用户确认
4. 用户确认后，调用执行

### 2. 展示格式规范

**操作成功时**：
```
✅ 已成功对实例 {instanceId} 的以下 Pod 执行屏蔽且重启：
- {pod_name_1}
- {pod_name_2}

屏蔽原因：{reason}
```

**操作失败时**：
```
❌ 屏蔽重启失败：{error_message}

建议：{处理建议}
```

**错误场景：**

| 错误情况 | 返回信息 | 处理建议 |
|----------|----------|----------|
| task_id 为空 | `task_id 不能为空` | 提示用户提供任务 ID |
| instance_id 为空 | `instance_id 不能为空` | 先调 `query_hunyuan_train_instance_list` 获取实例 |
| pod_names 为空 | `podNames 不能为空` | 先调 `query_hunyuan_train_pod_list` 获取 Pod 列表 |
| Pod 不存在 | `以下 Pod 不存在于实例中：[xxx]` | 根据返回的可用 Pod 列表重新选择 |
| 实例已结束 | `实例已经结束，无法删除pod` | 提示用户该实例已结束 |
| 无权限 | 权限校验失败 | 提示用户需要为系统管理员/空间管理员/任务创建者/任务管理员（ADMIN）之一 |
| 未开启容错 | `任务需要开启容错才能支持屏蔽且重启` | 提示用户需要先在任务配置中开启容错 |

---

## trigger_hunyuan_train_hot_update

在运行中的实例上触发热更新（无需重启任务），支持更新代码、配置文件、镜像、环境变量、Git 仓库、pip 依赖、Ray 配置等。

> ⚠️ **重要操作**：热更新会修改运行中实例的配置和代码，**必须向用户展示变更内容摘要并二次确认后再执行**！

**权限要求**：当前用户必须是「**系统管理员、空间管理员、任务创建者、任务管理员（ADMIN）**」之一（任一即放行）。

**前置条件**：
- 实例必须处于运行中状态
- 任务必须支持热更新操作（operatorTypes 中包含 `hotUpdate`）
- **调用前必须确认 `instance_id`**：用户给定了直接用；未给定则先调 `query_hunyuan_train_instance_list` 获取运行中实例，取最新一条的 `instanceId`

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call trigger_hunyuan_train_hot_update '{
  "instance_id": "ff8080819e0664bd019e067267430002",
  "change_description": "更新custom.toml配置",
  "start_cmd": "echo 7777\nbash run_train.sh",
  "image_name": "mirrors.tencent.com/taiji-ptm-mirrors/tlinux3.2-cuda12.9.1:latest"
}'
```

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `instance_id` | string | ✅ | 无 | 实例 ID，**必填** |
| `change_description` | string | ✅ | 无 | 变更说明（必填，便于历史追溯） |
| `start_cmd` | string | ❌ | 无 | 热更新后的新启动命令 |
| `train_config_file` | array[object] | ❌ | 无 | 训练配置文件列表，每项含 `name`（文件名）和 `content`（文件内容） |
| `start_script` | array[object] | ❌ | 无 | 启动脚本列表，每项含 `name`（文件名）和 `content`（文件内容） |
| `image_name` | string | ❌ | 无 | 镜像名称（完整镜像地址） |
| `env_vars_dict` | object | ❌ | 无 | 环境变量，key-value 形式 |
| `private_env_vars_dict` | object | ❌ | 无 | 私有环境变量，key-value 形式 |
| `git_project_name` | string | ❌ | 无 | 训练代码 Git 项目名称 |
| `git_project_branch` | string | ❌ | 无 | Git 分支 |
| `git_project_commit` | string | ❌ | 无 | Git Commit |
| `git_project_id` | integer | ❌ | 无 | Git 授权项目 ID |
| `is_pull_submodules` | boolean | ❌ | 无 | 是否拉取三方库 |
| `git_repos` | array[object] | ❌ | 无 | 统一 Git 仓库列表（主仓库+从仓库） |
| `enable_install_ray` | boolean | ❌ | 无 | 是否安装 Ray 依赖 |
| `enable_ray_submit_start_cmd` | boolean | ❌ | 无 | 是否使用 Ray 提交启动命令 |
| `ray_config_file` | array[object] | ❌ | 无 | Ray 配置文件列表 |
| `ray_args_config` | object | ❌ | 无 | Ray 参数配置，key-value 形式 |
| `resume_metrics` | boolean | ❌ | 无 | 是否恢复指标 |
| `dependencies` | array[string] | ❌ | 无 | pip 依赖列表，覆盖式替换，最多 50 条（不传则不触发 pip 维度的热更新） |

**调用规则：**

- 只允许传 `instance_id`（必填）及上述文档化的可选字段；即使用户 query 中附带 `wsid`，也不要传给本工具。
- 除 `instance_id` 外至少传 1 个变更字段，否则热更新无实际变更内容。
- 触发热更新前，必须先向用户展示变更内容摘要并请求二次确认。

**返回字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `hotUpdateId` | integer | 热更新记录 ID |
| `instanceId` | string | 实例 ID |
| `status` | string | 热更新状态（PENDING / RUNNING / SUCCESS / FAILED / CANCELLED） |
| `message` | string | 操作结果消息 |

**错误场景：**

| 错误情况 | 返回信息 | 处理建议 |
|----------|----------|----------|
| instance_id 为空 | `instance_id（实例 ID）不能为空` | 提示用户提供实例 ID |
| 实例不存在 | `instance with id xxx is not exist` | 提示用户检查实例 ID 是否正确 |
| 无权限 | 权限校验失败 | 提示用户需要为系统管理员/空间管理员/任务创建者/任务管理员（ADMIN）之一 |
| 无变更内容 | 除 instance_id 外无变更字段 | 提示用户至少指定一个变更项 |

---

## query_hunyuan_train_hot_update_versions

查询某个实例的所有热更新历史记录，包括热更新 ID、版本号、状态、变更说明、创建人、时间等。支持按时间范围过滤。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_hot_update_versions '{
  "instance_id": "ff8080819e0664bd019e067267430002"
}'
```

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `instance_id` | string | ✅ | 无 | 实例 ID，**必填** |
| `start_time` | string | ❌ | 无 | 起始时间（可选），格式 `yyyy-MM-dd HH:mm:ss`，查询 create_time >= 该值 |
| `end_time` | string | ❌ | 无 | 结束时间（可选），格式 `yyyy-MM-dd HH:mm:ss`，查询 create_time <= 该值 |

**调用规则：**

- 只允许传 `instance_id`（必填）及 `start_time`、`end_time`。
- 用户泛问"热更新历史/版本列表"时不传时间范围；用户指定时间段时才传 `start_time` / `end_time`。

**返回字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `instanceId` | string | 实例 ID |
| `versions` | array | 热更新版本列表 |
| `versions[].hotUpdateId` | integer | 热更新记录 ID |
| `versions[].version` | string | 版本号，如 V001 |
| `versions[].status` | string | 热更新状态：PENDING / RUNNING / SUCCESS / FAILED / CANCELLED |
| `versions[].statusMessage` | string | 状态描述信息 |
| `versions[].changeDescription` | string | 变更说明 |
| `versions[].creator` | string | 创建人 |
| `versions[].createdAt` | string | 创建时间 |
| `versions[].updatedAt` | string | 更新时间（完成/失败时间） |
| `total` | integer | 总数 |

---

## query_hunyuan_train_hot_update_status

查询热更新任务的当前执行状态，用于轮询 `trigger_hunyuan_train_hot_update` 的异步执行结果。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_hot_update_status '{
  "hot_update_id": 28202
}'
```

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `hot_update_id` | integer | ✅ | 无 | 热更新记录 ID，**必填**（由 `trigger_hunyuan_train_hot_update` 返回或从版本列表获取） |

**调用规则：**

- **不要主动轮询**，只在用户要求时调用。
- 若用户要求"等待热更新完成"，最多轮询 10 次，间隔 5 秒；若 10 次后仍为 PENDING/RUNNING，告知用户其仍在执行，可稍后手动查询。
- 若状态为 FAILED，将 `status_message` 中的错误原因告知用户，**不要**自动重试或建议重试。

**返回字段说明：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `hotUpdateId` | integer | 热更新记录 ID |
| `status` | string | 热更新状态：PENDING / RUNNING / SUCCESS / FAILED / CANCELLED |
| `statusMessage` | string | 状态描述信息（失败时包含错误原因） |
