## query_hunyuan_train_task_list

查询训练任务列表，支持模版化训练、自定义训练、模型开发三种任务类型。支持丰富的筛选条件。

**MCP 工具调用：**
```bash
# 基础查询
python3 scripts/connect_mcp.py call query_hunyuan_train_task_list '{"wsid": 10103, "task_category": "custom_training", "query_type": "all"}'

# 带筛选条件查询
python3 scripts/connect_mcp.py call query_hunyuan_train_task_list '{"wsid": 10103, "task_category": "custom_training", "query_type": "all", "status": ["TRAINING_RUNNING"], "gpu_type": ["A100"]}'
```

### taskId 格式说明

| taskId 前缀 | 对应任务类型 | 说明 |
|------------|------------|------|
| `finetuning_` | 模版化训练 | 通过 task_category=finetuning 查询 |
| `basic_train_` | 自定义训练 **或** 模型开发 | 通过 task_category 区分（custom_training / model_dev） |

> ⚠️ `basic_train_` 前缀的 taskId 可能是自定义训练也可能是模型开发，两者的 taskId 格式相同。区分方式：task_list 返回的 `task_category` 字段。


### 必填参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| wsid | integer | ✅ | 工作空间 ID（空间 ID），用于筛选特定空间下的任务列表 |
| task_category | string | ✅ | 任务类别：`finetuning`（模版化训练）、`custom_training`（自定义训练）、`model_dev`（模型开发） |
| query_type | string | ✅ | 查询类型：`all`（空间内所有任务）、`my_permission`（我有权限的任务）、`my_created`（我创建的任务） |

### 基础可选参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| keyword | string | ❌ | 模糊搜索关键词（按任务名称/描述/应用组/task_id 搜索） |
| page | integer | ❌ | 分页页码，从 **1** 开始，默认 1 |
| page_size | integer | ❌ | 每页数量，默认 20，最大 100 |

### 高频筛选参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| status | array[string] | ❌ | 状态筛选（支持多选）。可选值：`TRAINING_RESOURCE_WAITING`（等待资源中）、`TRAINING_RUNNING`（运行中）、`RUNNING_SUCCESS`（训练完成）、`RUNNING_FAILED`（运行失败）、`START_FAILED`（启动失败）、`KILLED`（终止）等 |
| creator | array[string] | ❌ | 按创建人 RTX 账号筛选（支持多选） |
| app_group_id | array[string] | ❌ | 按应用组名称筛选（支持多选） |
| order_by | string | ❌ | 排序字段：`created_at`（按创建时间）、`updated_at`（按更新时间） |
| order_type | string | ❌ | 排序方式：`ASC`（升序）、`DESC`（降序，默认） |
| created_at | array[string] | ❌ | 创建时间范围筛选，格式 `["起始时间","结束时间"]`，时间格式 `yyyy-MM-dd HH:mm:ss` |
| updated_at | array[string] | ❌ | 更新时间范围筛选，格式 `["起始时间","结束时间"]`，时间格式 `yyyy-MM-dd HH:mm:ss` |

### 中频筛选参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| train_framework | string | ❌ | 训练框架筛选（仅对 custom_training 生效） |
| model_stages | array[string] | ❌ | 训练阶段筛选（仅对 custom_training 生效，finetuning / model_dev 不支持）。可选值：`Pretrain`、`Midtrain`、`SFT`、`RL` |
| gpu_type | array[string] | ❌ | GPU 卡类型筛选（支持多选） |
| task_tag | array[string] | ❌ | 任务标签筛选（支持多选），可通过 `list_hunyuan_train_task_tag_enums` 获取可选值 |

### 返回（成功）

```json
{
  "items": [
    {
      "task_id": "basic_train_zhongtaohe_20260306155021_e684d920",
      "name": "混元-SFT训练任务",
      "description": "SFT微调训练",
      "task_category": "custom_training",
      "task_category_name": "自定义训练任务",
      "creator": "zhongtaohe",
      "app_group_id": "TaiJi_HYAide_HYapp_EXTRA4DEV",
      "machine_count": 1,
      "gpu_per_machine": 8,
      "gpu_type": "A100",
      "train_framework": "Angel_PTM_V2_Fsdp",
      "train_framework_text": "Angel-PTM2.0-FSDP",
      "status": "SUCCEED",
      "status_text": "结束(成功)",
      "instance_id": "ins-abc-001",
      "created_at": "2026-03-06 15:50:21",
      "updated_at": "2026-03-06 18:30:00"
    }
  ],
  "page": 1,
  "page_size": 20,
  "total": 12,
  "has_more": false
}
```

### 筛选条件智能映射

| 用户表述 | 对应参数 | 传值示例 |
|----------|---------|---------|
| "运行中的任务" | `status` | `["TRAINING_RUNNING"]` |
| "失败的任务" | `status` | `["RUNNING_FAILED", "START_FAILED"]` |
| "已完成的任务" | `status` | `["RUNNING_SUCCESS"]` |
| "等待资源的任务" | `status` | `["TRAINING_RESOURCE_WAITING"]` |
| "已终止的任务" | `status` | `["KILLED"]` |
| "查看xxx创建的任务" | `creator` | `["xxx"]` |
| "混元应用组的任务" | `app_group_id` | `["混元"]` |
| "按创建时间倒序" | `order_by` + `order_type` | `created_at` + `DESC` |
| "最近一周的任务" | `created_at` | `["2026-06-25 00:00:00", "2026-07-02 23:59:59"]` |
| "用A100的任务" | `gpu_type` | `["A100"]` |
| "Angel-PTM2.0框架的任务" | `train_framework` | `Angel_PTM_V2` |
| "带某标签的任务" | `task_tag` | `["标签名"]` |
| "Pretrain阶段的任务" | `model_stages` | `["Pretrain"]` （仅 custom_training 支持） |
| "SFT阶段的任务" | `model_stages` | `["SFT"]` （仅 custom_training 支持） |

> ⚠️ **注意**：运行中（TRAINING_RUNNING）任务的 `items` 中**没有** `status` / `status_text` / `instance_id` 字段；仅已完成/已终止/失败的实例才携带这些字段。finetuning 类型的任务不返回 `machine_count` / `gpu_per_machine` / `gpu_type` / `train_framework` 等资源字段。

### 返回字段使用边界

- 本工具返回的 `businessFlag` / `app_group_id` 仅用于描述或筛选训练任务，禁止据此反推用户拥有的应用组；用户查询应用组时应切换到 `resource-mgmt-skill`。
- 本工具返回的 `wsid` 仅表示任务所属工作空间，禁止据此反推用户拥有的工作空间；用户查询工作空间时应切换到 `workspace-skill`。

### 展示格式规范（必须严格遵守）

任务列表**必须使用 Markdown 表格**展示，禁止使用嵌套列表、缩进子列表等不美观的格式。

推荐表格列：`序号 | 任务名称 | 任务 ID | 描述 | 资源 | 状态 | 创建时间`

| 规则 | 说明 |
|------|------|
| 资源列 | 合并 machine_count、gpu_per_machine、gpu_type，格式："N机M卡 卡类型"（如 "1机8卡 H800"）。finetuning 任务无资源字段则显示 `-` |
| 状态列 | 使用 status_text 字段展示，运行中（无 status_text 字段时）加 ✅ 并显示"运行中"，已终止加 ⏹，已完成显示"结束(成功)" |
| 创建时间 | 只显示日期部分（yyyy-MM-dd），使用 created_at 字段，不显示具体时间 |
| 汇总行 | 表格前显示："以下是 **{wsid} 空间** 的 **{任务类别}** 任务，共 {total} 个（当前显示第 X-Y 条）：" |
| 追加提示 | 表格后**必须先空一行再加 `---` 分割线**，然后追加查询范围提示和引导性操作 |
| 任务 ID | **必须展示完整的 task_id，严禁截断或用 `...` 省略** |

**禁止的展示格式：**
- ❌ 嵌套列表
- ❌ 将字段拆成多行子项展示
- ❌ 省略 task_id、name、description 中的任何一个
- ❌ **严禁截断 task_id**

---

## get_hunyuan_train_task_detail

获取混元训练任务详情，根据 task_id 查询指定训练任务的完整信息。系统根据 task_id 前缀自动识别任务类型。

**MCP 工具调用：**
```bash
# 查询任务当前最新配置
python3 scripts/connect_mcp.py call get_hunyuan_train_task_detail '{"wsid": 10103, "task_id": "basic_train_xxx_20260306_e684d920"}'

# 查询指定实例的快照配置
python3 scripts/connect_mcp.py call get_hunyuan_train_task_detail '{"wsid": 10103, "task_id": "basic_train_xxx_20260306_e684d920", "instance_id": "ins-abc-001"}'
```

### 参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| wsid | integer | ✅ | 工作空间 ID |
| task_id | string | ✅ | 任务 ID |
| instance_id | string | ❌ | 实例 ID（可选，不传则查询最新实例） |

### 返回（成功）

```json
{
  "task_id": "basic_train_zhongtaohe_20260306155021_e684d920",
  "name": "混元-SFT训练任务",
  "description": "SFT微调训练",
  "task_category": "custom_training",
  "task_category_name": "自定义训练任务",
  "creator": "zhongtaohe",
  "app_group_id": "TaiJi_HYAide_HYapp_EXTRA4DEV",
  "machine_count": 1,
  "gpu_per_machine": 8,
  "gpu_type": "A100",
  "train_framework": "Angel_PTM_V2_Fsdp",
  "train_framework_text": "Angel-PTM2.0-FSDP",
  "created_at": "2026-03-06 15:50:21",
  "updated_at": "2026-03-06 18:30:00",
  "config": {
    "git_project_branch": "master",
    "git_project_name": "ptm_v2/AngelPTM",
    "start_cmd": "bash scripts/run_train.sh fsdp ...",
    "image_name": "mirrors.tencent.com/taiji-ptm-mirrors/...",
    "env_vars_dict": null,
    "enable_report_metric": false
  },
  "admin_group": ["zhongtaohe", "user1"],
  "message": "这是一个【自定义训练任务】，任务ID：basic_train_zhongtaohe_20260306155021_e684d920"
}
```

> ⚠️ **注意**：返回是**扁平结构**，不存在嵌套的 `fields` 数组。所有字段在顶层直接访问。`config` 为嵌套对象，包含 Git、镜像、环境变量等配置详情。

---

## get_hunyuan_train_task_config_file

> ⚠️ **没有 `action` 参数**；读取指定配置只传 `file_name`，查看配置列表则不传 `file_name`。严禁传 `action=list/read`。

获取混元训练任务的配置文件。不传 file_name 时返回该任务的配置文件列表；传入 file_name 时返回指定文件的内容。

> ⚠️ **该工具没有 `action` 参数**。要读指定配置时只传 `file_name="custom.toml"`；要看配置列表则不传 `file_name`。严禁传 `action=list/read`。

**MCP 工具调用：**
```bash
# 查看配置文件列表
python3 scripts/connect_mcp.py call get_hunyuan_train_task_config_file '{"wsid": 10103, "task_id": "basic_train_xxx"}'

# 查看指定文件内容
python3 scripts/connect_mcp.py call get_hunyuan_train_task_config_file '{"wsid": 10103, "task_id": "basic_train_xxx", "file_name": "config.yaml"}'
```

### 参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| wsid | integer | ✅ | 工作空间 ID |
| task_id | string | ✅ | 任务 ID |
| file_name | string | ❌ | 配置文件名，不传返回文件列表，传则返回该文件内容 |
| instance_id | string | ❌ | 实例 ID（可选） |

### 返回（文件列表模式）

```json
{
  "task_id": "basic_train_xxx",
  "task_category_name": "自定义训练任务",
  "file_list": [
    {"config_type": "trainConfigFile", "config_type_name": "依赖文件", "file_name": "config.yaml", "file_path": "/data/config/config.yaml"}
  ],
  "message": "配置文件列表"
}
```

### 返回（文件内容模式）

```json
{
  "task_id": "basic_train_xxx",
  "task_category_name": "自定义训练任务",
  "config_type": "trainConfigFile",
  "config_type_name": "依赖文件",
  "file_name": "config.yaml",
  "file_path": "/data/config/config.yaml",
  "file_content": "model:\n  type: gpt\n  ...",
  "message": "自定义训练任务【basic_train_xxx】中配置文件 'config.yaml' 的详情"
}
```

> ⚠️ **注意**：文件内容模式返回的是**扁平结构**，不存在 `file_list` 数组包裹。`file_content` / `file_name` / `file_path` / `config_type` 均为顶层字段。

---

## query_hunyuan_train_checkpoint_list

查询混元训练任务产出的 checkpoint 列表（仅支持模版化训练、自定义训练）。

> ⚠️ **模型开发任务不支持。**

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_checkpoint_list '{"task_id": "basic_train_xxx"}'
```

### 参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | string | ✅ | 任务 ID |
| instance_id | string | ❌ | 实例 ID（可选） |

### 返回（成功）

```json
{
  "task_id": "basic_train_xxx",
  "instance_id": "ins-abc-001",
  "total": 2,
  "message": "这是自定义训练任务【basic_train_xxx】的产出列表，实例ID：ins-abc-001，共 2 个产出。",
  "ckptList": [
    {
      "name": "iter_0010000",
      "path": "/apdcephfs_nj11/share_.../ckpt/iter_0010000",
      "file_size": 4037269258240,
      "released": false,
      "model_id": null,
      "model_name": null,
      "model_path": "/apdcephfs_nj11/share_.../iter_0010000",
      "hf_model_path": "/apdcephfs_nj11/share_.../ckpt/global_step_hf",
      "created_at": "2026-02-01 03:57:05",
      "delete_time": "2026-10-01 03:57:05",
      "export_task": null
    }
  ]
}
```

> ⚠️ **注意**：`released` 为 **boolean** 类型（`true`/`false`），**不是**字符串 "未发布"/"已发布"。`file_size` 为整数（字节数），不是人类可读的字符串。

**展示格式要求：**
- 必须使用 Markdown 表格展示，禁止嵌套列表
- 每条 Checkpoint 必须包含所有字段

---

## update_hunyuan_train_task_permission

分享或移除混元训练任务的管理员权限。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call update_hunyuan_train_task_permission '{"task_id": "basic_train_xxx", "operation": "add", "user_list": ["user1", "user2"]}'
```

### 参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | string | ✅ | 任务 ID |
| operation | string | ✅ | 操作类型：`add`（添加管理员）、`remove`（移除管理员） |
| user_list | array[string] | ✅ | 目标用户 RTX 列表 |

### 返回（成功）

```json
{
  "task_id": "basic_train_xxx",
  "operation": "add",
  "affected_users": ["user1", "user2"],
  "current_admin_group": ["owner", "user1", "user2"],
  "message": "分享成功，user1 已获得自定义训练任务【basic_train_xxx】的管理员权限（可进行运行、停止等操作）"
}
```

---

## start_hunyuan_train_task

启动混元训练平台的训练任务，触发任务执行。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call start_hunyuan_train_task '{"task_id": "basic_train_xxx"}'
```

### 参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | string | ✅ | 任务 ID（finetuning_ 或 basic_train 开头） |

**约束条件：**
- 每个任务同一时间只允许一个实例运行。如果已有运行中实例，调用将返回错误。

### 返回（成功）

```json
{
  "task_id": "basic_train_xxx",
  "message": "任务已成功启动"
}
```

---

## stop_hunyuan_train_task

停止混元训练平台的训练任务，系统会自动停止该任务的最新实例。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call stop_hunyuan_train_task '{"task_id": "basic_train_xxx"}'
```

### 参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | string | ✅ | 任务 ID（finetuning_ 或 basic_train 开头） |

### 返回（成功）

```json
{
  "task_id": "basic_train_xxx",
  "message": "任务已成功停止"
}
```

---

## clone_hunyuan_train_task

克隆混元自定义训练任务并可修改配置（仅支持 basic_train 开头的任务）。支持修改 GPU 资源、启动命令、配置文件、应用组/地域/GPU卡型、基座模型、镜像，并支持选择是否复制原任务的伴生评估配置。

> ⚠️ **仅支持自定义训练和模型开发任务**（taskId 以 `basic_train_` 开头）。模版化训练请用 `clone_hunyuan_train_finetuning_task`。

**典型使用流程：**
1. 用户说"克隆任务 xxx 并修改学习率"
2. 先调用 `get_hunyuan_train_task_config_file` 查看原任务的配置文件列表和内容，了解配置结构
3. 根据用户需求构造 `config_modifications`
4. 调用 `clone_hunyuan_train_task` 执行克隆

**MCP 工具调用：**
```bash
# 简单克隆（不修改任何配置）
python3 scripts/connect_mcp.py call clone_hunyuan_train_task '{"wsid": 10103, "task_id": "basic_train_xxx"}'

# 克隆并修改 GPU 资源
python3 scripts/connect_mcp.py call clone_hunyuan_train_task '{"wsid": 10103, "task_id": "basic_train_xxx", "task_name": "新任务名", "host_num": 2, "host_gpu_num": 8}'

# 克隆并修改配置文件参数
python3 scripts/connect_mcp.py call clone_hunyuan_train_task '{"wsid": 10103, "task_id": "basic_train_xxx", "config_modifications": [{"action": "update", "path": "TRAINING_ARGS", "key": "lr", "value": "0.0001"}]}'

### 支持与不支持修改的字段

- **支持修改**：任务名/描述、GPU 资源、启动命令、配置文件参数（config_modifications）、是否复制伴生评估、指定源实例、应用组/地域/GPU卡型（三参数需同时传入）、基座模型（model_name + model_privacy）。
- **不支持修改**：Git 仓库/分支/commit、环境变量、镜像、训练数据路径、框架类型等。

### 镜像更换规则

用户明确说"换镜像/更换镜像"时传 `image_name`；用户没提时**不得默认更换**。镜像格式 `域名/仓库路径:标签`，不带标签时后端默认用 `latest`。

### 修改应用组的约束流程

传了新的 `app_group_id` 但未同时给 `location`/`gpu_type` 时，必须：① 切 `resource-mgmt-skill` 展示可用的应用组；② 查目标组的 GPU 配额展示给用户选 `location`/`gpu_type`；③ 用户选定后三参数一起传入。

### 伴生评估配置规则

用户说"复制伴生评估配置"时传 `copy_evaluation_config=true`；用户没提则默认不复制。用户未指定源实例 ID 时直接不传，由后端默认选择。

# 克隆并复制原任务的伴生评估配置
python3 scripts/connect_mcp.py call clone_hunyuan_train_task '{"wsid": 10103, "task_id": "basic_train_xxx", "copy_evaluation_config": true}'

# 克隆并从指定源实例复制伴生评估配置
python3 scripts/connect_mcp.py call clone_hunyuan_train_task '{"wsid": 10103, "task_id": "basic_train_xxx", "copy_evaluation_config": true, "copy_evaluation_source_instance_id": "9e2914389ef90081019ef9a4ee850003"}'

# 克隆并切换应用组/地域/GPU卡型（三个参数需同时传入）
python3 scripts/connect_mcp.py call clone_hunyuan_train_task '{"wsid": 10103, "task_id": "basic_train_xxx", "app_group_id": "TaiJi_HYAide_HYapp_NEW", "location": "zw", "gpu_type": "H20"}'

# 克隆并修改基座模型（官方模型）
python3 scripts/connect_mcp.py call clone_hunyuan_train_task '{"wsid": 10103, "task_id": "basic_train_xxx", "model_name": "HY3.0_30B_A3B_Base_15T_Pretrain_2T_Midtrain_200B_256K", "model_privacy": "list_official"}'

# 克隆并修改基座模型（个人模型）
python3 scripts/connect_mcp.py call clone_hunyuan_train_task '{"wsid": 10103, "task_id": "basic_train_xxx", "model_name": "my_custom_model_v2", "model_privacy": "list_user"}'

# 克隆并更换镜像
python3 scripts/connect_mcp.py call clone_hunyuan_train_task '{"wsid": 10103, "task_id": "basic_train_xxx", "image_name": "mirrors.tencent.com/your-repo/your-image:v2.0"}'
```

### 参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| wsid | integer | ✅ | 工作空间 ID |
| task_id | string | ✅ | 要克隆的任务 ID（仅支持 basic_train 开头） |
| task_name | string | ❌ | 新任务名称（不传则自动生成），最长 100 字符 |
| description | string | ❌ | 任务描述（不传则自动生成） |
| host_num | integer | ❌ | GPU 机器数（台），不传则沿用原任务 |
| host_gpu_num | integer | ❌ | 单机 GPU 卡数，不传则沿用原任务 |
| start_cmd | string | ❌ | 启动命令，不传则沿用原任务 |
| copy_evaluation_config | boolean | ❌ | 是否复制原任务的伴生评估配置，默认 `false`；为 `true` 且未指定 `copy_evaluation_source_instance_id` 时，后端默认使用原任务中存在伴生配置的最新实例 |
| copy_evaluation_source_instance_id | string | ❌ | 复制伴生评估配置时选择的源任务实例 ID；仅在 `copy_evaluation_config=true` 时生效；不传则由后端选择原任务中存在伴生配置的最新实例 |
| app_group_id | string | ❌ | 目标应用组 ID，不传则沿用原任务。修改应用组时需同时传入 `location` 和 `gpu_type` |
| location | string | ❌ | 目标地域（如 zw、nj、gy、gz），修改应用组时必传，需与新应用组的可用地域匹配 |
| gpu_type | string | ❌ | 目标 GPU 卡类型（如 H20、L40、L40S），修改应用组时必传，需与新地域的可用卡型匹配 |
| model_name | string | ❌ | 基座模型名称（不传则沿用原任务）。需与 `model_privacy` 配合使用，后端会根据名称查询校验模型是否存在 |
| model_privacy | string | ❌ | 模型分类：`list_official`（官方模型）/ `list_user`（个人模型）/ `list_public`（空间公开模型）。修改基座模型时必填 |
| image_name | string | ❌ | 训练镜像名称（不传则沿用原任务）。传入时后端会校验镜像是否存在，格式为 `域名/仓库路径:标签`，如 `mirrors.tencent.com/repo/image:v2.0` |
| config_modifications | array | ❌ | 配置文件修改列表（直接传数组对象） |

**镜像更换规则：**
- 用户明确说"换镜像 / 更换镜像 / 用新镜像 / 镜像改为"时，传 `image_name`。
- 用户没有提镜像时，不传 `image_name`，沿用原任务镜像，不得默认更换。
- 镜像格式为 `域名/仓库路径:标签`（如 `mirrors.tencent.com/star_formation/hytrain:v2.0`），不带标签时后端默认使用 `latest`。
- 后端会校验镜像是否存在（先查本地缓存，缓存未命中则调用镜像仓库 API 查找 tag），校验失败会返回错误，应如实展示给用户。
- 返回字段 `image_change` 会描述镜像变更情况（如"镜像已从 xxx 更换为 yyy"），应展示给用户。

**伴生评估配置复制规则：**
- 用户明确说"复制伴生评估配置 / 保留伴生评估 / 带上自动评估 / 带上评估配置"时，传 `copy_evaluation_config=true`。
- 用户没有提伴生评估时，不传 `copy_evaluation_config` 或传 `false`，不得默认复制。
- 用户指定源实例 ID 时，传 `copy_evaluation_source_instance_id=<实例ID>`。
- 用户未指定源实例 ID 时，不要为了找"最新有伴生配置的实例"额外查询实例或 evaluation 配置，直接不传 `copy_evaluation_source_instance_id`，由后端按默认规则选择。
- 克隆时复制伴生评估配置仍属于 `clone_hunyuan_train_task` 的参数能力，不需要切到 `evaluation-skill`。
- 若接口返回 warnings 表示原任务无可复制的伴生评估配置，应如实展示 warnings，不自动补调 evaluation 工具。

**config_modifications 参数详细说明：**

数组，每个元素包含：
| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| action | string | ✅ | 操作类型：`add`（新增）/ `update`（更新）/ `delete`（删除） |
| path | string | ✅ | 配置路径，即 section 名称，如 `DATA_ARGS` 或 `DATA_ARGS.sub_section` |
| key | string | ✅ | 要操作的配置项 key |
| value | string | ❌ | 值（add/update 时必填） |

示例：
```json
[
  {"action": "update", "path": "TRAINING_ARGS", "key": "lr", "value": "0.0001"},
  {"action": "add", "path": "DATA_ARGS", "key": "new_param", "value": "new_value"},
  {"action": "delete", "path": "DATA_ARGS", "key": "deprecated_param"}
]
```

### 返回（成功）

```json
{
  "task_id": "basic_train_xxx_20260331_abc123",
  "task_name": "混元-SFT训练_mcp_copy_20260331104400",
  "message": "克隆成功，新任务ID：basic_train_xxx_20260331_abc123，任务名称：混元-SFT训练_mcp_copy_20260331104400",
  "framework_type": "Angel_PTM_V2_Fsdp",
  "resource_change": "GPU资源配置沿用原任务",
  "start_cmd_change": null,
  "config_changes": [],
  "warnings": []
}
```

**展示格式要求：**
1. 显示新任务 ID 和名称
2. 显示配置变更结果（如有）
3. 如果传入了 `copy_evaluation_config=true`，说明已请求复制原任务伴生评估配置；如返回 warnings，必须展示 warnings
4. 引导："新任务已创建。您还可以：启动该任务 / 查看新任务的详情 / 重新克隆一个任务并修改其他参数"

> ⚠️ **克隆任务支持修改的参数范围（严格限制）：**
> - ✅ 任务名称和描述
> - ✅ GPU 资源（机器数、单机卡数）
> - ✅ 启动命令（start_cmd）
> - ✅ 配置文件中的参数（通过 config_modifications 修改 section/key/value）
> - ✅ 是否复制原任务的伴生评估配置（`copy_evaluation_config`）
> - ✅ 指定从哪个源实例复制伴生评估配置（`copy_evaluation_source_instance_id`）
> - ✅ 应用组/地域/GPU卡型（通过 app_group_id、location、gpu_type 修改，三个参数需同时传入）
> - ✅ 基座模型（通过 model_name + model_privacy 指定新模型，后端自动查询校验并回填 modelId、seriesId 等字段）
> - ❌ **不支持**修改：Git 仓库/分支/commit、环境变量、镜像、训练数据路径、框架类型等

### 修改应用组时的约束流程

当用户在克隆任务时传入了新的 `app_group_id`（应用组），但未同时提供 `location`（地域）或 `gpu_type`（GPU 卡型）时，**必须**按以下流程补全信息后再调用 `clone_hunyuan_train_task`。若用户连 `app_group_id` 也未提供，则从 Step 1 开始；若已提供 `app_group_id` 但缺其他参数，则从 Step 2 开始。

**Step 1：查询用户可用应用组列表**
切到 `resource-mgmt-skill`，调用 `query_shared_resources_app_group_list`（传入 `is_usable=true`），获取当前用户可用的应用组列表。将列表展示给用户，**由用户选择目标应用组**。
> ⛔ **严禁编造或跳过此步骤**：必须真实调用工具查询，绝不能猜一个应用组 ID 直接传入。

**Step 2：查询应用组各地域 GPU 卡型及配额**
用户选定 `app_group_id` 后，调用 `query_shared_resources_gpu_info_batch`，传入 `app_group_ids`（数组，只包含目标应用组），获取该应用组在各地域可用的 GPU 卡型及配额信息（总卡/已用/空闲/排队）。

**Step 3：向用户展示地域和卡型选项**
将 Step 2 查询到的地域和 GPU 卡型列表（含配额）展示给用户，**由用户选择具体的 `location` 和 `gpu_type`**。
> ⛔ **严禁编造或跳过此步骤**：必须真实调用工具查询，绝不能根据经验猜测地域缩写或 GPU 卡型名称直接传入。

**Step 4：用户选择后传入参数**
将用户选择的地域和 GPU 卡型分别作为 `location` 和 `gpu_type` 参数，与 `app_group_id` 一起传入 `clone_hunyuan_train_task`。

> ⚠️ **注意**：
> - 修改应用组时 `app_group_id`、`location`、`gpu_type` 三个参数**必须同时传入**，后端会校验应用组权限、地域支持范围和 GPU 卡型可用性。
> - 如果用户已同时提供 `app_group_id`、`location`、`gpu_type` 三个完整参数，则不触发上述查询流程，直接传入即可。
> - 如果查询失败，不阻断主流程，提示用户手动确认后继续。

---

## clone_hunyuan_train_finetuning_task

克隆混元模版化训练任务（仅支持 finetuning_ 开头的任务）。支持修改训练数据集、训练超参数、依赖文件。

> ⚠️ **仅支持模版化训练任务**（taskId 以 `finetuning_` 开头）。自定义训练请用 `clone_hunyuan_train_task`。

**MCP 工具调用：**
```bash
# 简单克隆
python3 scripts/connect_mcp.py call clone_hunyuan_train_finetuning_task '{"wsid": 10103, "task_id": "finetuning_xxx", "new_task_name": "新模版化训练任务"}'

# 克隆并修改超参数
python3 scripts/connect_mcp.py call clone_hunyuan_train_finetuning_task '{"wsid": 10103, "task_id": "finetuning_xxx", "new_task_name": "调参任务", "model_params": {"learning_rate": "0.0001", "epochs": "5"}}'
```

### 参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| wsid | integer | ✅ | 工作空间 ID |
| task_id | string | ✅ | 要克隆的任务 ID（仅支持 finetuning_ 开头） |
| new_task_name | string | ✅ | 新任务名称，最长 100 字符 |
| description | string | ❌ | 新任务描述（可选，不传则自动生成） |
| train_data | array | ❌ | 训练数据集修改：null/空 list 表示沿用原任务；非空则整体覆盖 |
| model_params | object | ❌ | 训练超参数修改：key/value 形式 patch 覆盖，仅覆盖已存在的 key |
| dep_files | array | ❌ | 依赖文件修改：按 name 合并到原 depFiles，同名替换、新名追加 |

**train_data 数组元素：**
| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| id | integer | ❌ | 数据集 ID（CFS 数据集时必填） |
| name | string | ❌ | 数据集名称（仅用于展示） |
| file_name | string | ❌ | 数据集文件名（仅用于展示） |
| file_path | string | ✅ | 数据集文件路径（CFS 路径或对象存储路径） |

**dep_files 数组元素：**
| 字段 | 类型 | 必填 | 描述 |
|------|------|------|------|
| name | string | ✅ | 依赖文件名（用于合并匹配） |
| content | string | ✅ | 依赖文件内容 |

### 返回（成功）

```json
{
  "task_id": "finetuning_xxx_20260331_abc123",
  "task_name": "调参任务",
  "message": "克隆成功，新任务ID：finetuning_xxx_20260331_abc123，任务名称：调参任务"
}
```

---

## query_hunyuan_train_failure_event_list

查询混元训练任务的异常事件列表（仅支持 basic_train 开头的自定义训练任务），支持分页。

> ⚠️ **仅支持自定义训练任务**（taskId 以 `basic_train_` 开头），模型开发和模版化训练任务不支持。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_failure_event_list '{"task_id": "basic_train_xxx_20260131_xxx"}'
```

### 参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | string | ✅ | 任务 ID（仅支持 basic_train 开头的自定义训练任务） |
| instance_id | string | ❌ | 实例 ID（可选） |
| page | integer | ❌ | 页码，从 1 开始，默认 1 |
| page_size | integer | ❌ | 每页数量，默认 20，最大 1000 |

### 返回字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `total` | long | 异常事件总数 |
| `items` | array | 异常事件列表 |
| `page` | integer | 当前页码 |
| `page_size` | integer | 每页数量 |
| `has_more` | boolean | 是否有更多数据 |
| `items[].id` | int | 事件 ID |
| `items[].instance_id` | string | 实例 ID |
| `items[].error_time` | string | 错误时间（yyyy-MM-dd HH:mm:ss） |
| `items[].error_code` | string | 错误代码 |
| `items[].error_msg` | string | 错误信息 |
| `items[].ips` | array | IP 地址列表 |
| `items[].priority` | int | 错误等级（数字越小优先级越高） |
| `items[].pod_name` | string | Pod 名称 |

**展示格式规范：**
1. 先显示汇总信息："任务 {task_id} 的异常事件列表，共 {total} 条"
2. 使用 Markdown 表格展示：`| # | 错误时间 | 错误代码 | 错误信息 | IP | Pod | 优先级 |`
3. 如果无异常事件，显示"🎉 该任务暂无异常事件记录，训练运行正常。"
4. 有异常事件时，表格结束后**必须**输出分析结论

**分析结论规范（有异常事件时必须输出）：**
```
📊 **异常事件分析结论：**
```
分析维度（根据实际数据选择性输出）：

| 分析维度 | 说明 |
|---------|------|
| **故障类型分布** | 统计各 error_code 出现的次数 |
| **时间规律** | 分析异常事件的时间分布 |
| **节点/Pod 集中度** | 分析故障是否集中在某些特定 IP 或 Pod 上 |
| **综合判断** | 给出整体结论 |
| **建议操作** | 根据分析结论给出 1-3 条具体的下一步建议 |

---

## list_hunyuan_train_task_tag_enums

查询混元训练平台指定工作空间下所有可用的任务标签枚举值列表。

> ⚠️ **仅支持自定义训练，模型开发和模版化训练任务不支持。**

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call list_hunyuan_train_task_tag_enums '{"wsid": 10103}'
```

### 参数

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| wsid | integer | ✅ | 工作空间 ID |
| train_template_type | string | ❌ | 训练模版类型，默认 `basic_train` |

### 返回（成功）

```json
["test", "RL", "生产", "评测", "模型转换", "Debug", "baseline", ...]
```

**展示格式规范：**
1. 将所有标签以列表形式**完整展示**
2. 展示后说明："以上是当前空间下所有用户创建的项目标签"
3. 引导后续操作："您可以使用标签筛选任务，例如：'查看标签为 xxx 的任务列表'"

---
