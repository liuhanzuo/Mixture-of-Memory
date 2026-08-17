## create_taiji_eval_exercise

#### 参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `exercise_name` | string | ✅ | 评测集名称（全局唯一） |
| `desc` | string | ❌ | 评测集描述 |
| `admin` | string | ❌ | 管理员用户名，多个用逗号分隔 |
| `exercise_tag` | string | ❌ | 评测集标签（英文，如 `code`/`math`/`pretrain`，非中文） |
| `visibility` | string | ❌ | 可见范围：`CURRENT_WORKSPACE`（默认）/ `ALL_PLATFORM` / `SPECIFIED_WORKSPACES` |
| `visible_ws_ids` | string | ❌ | 可见空间 ID 列表，`visibility=SPECIFIED_WORKSPACES` 时必填 |

#### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `exercise_id` | number | 评测集 ID（后续创建 ExerciseVersion 必用） |
| `exercise_name` | string | 评测集名称 |
| `desc` | string | 描述 |
| `creator` | string | 创建人 |
| `create_time` | string | 创建时间 |

#### 示例

```bash
python3 scripts/connect_mcp.py call create_taiji_eval_exercise '{
  "exercise_name": "我的评测集-2026",
  "desc": "用于测试 skill 上传流程",
  "visibility": "ALL_PLATFORM"
}'
```

---

## create_taiji_eval_exercise_version

#### 参数

> ⚠️ **不要从 `get_taiji_eval_exercise_version_detail` 返回结果直接拼参数！必须先读本表确认必填/选填，参数要求以本表为准。`parameter_configuration` 的值（文生文/多模态/自定义）决定哪些字段必填，见下方模式说明。**

> ⚠️ **parameter_configuration 决定评测模式，不同模式下字段要求不同**：
> - **文生文**：`metric_name`/`data_class_name` 必填；`prompt`/`max_token`/`judge_model_name` **不需要填写**（传了会被忽略，响应中通过 `_warning` 字段提示）
> - **多模态**：`metric_name`/`data_class_name`/`prompt`/`max_token`/`judge_model_name` 均必填非空。prompt 是什么就传什么（可能是 `"-"`、长 JSON、对象等），**不要自作主张跳过**
> - **自定义**：以上字段**均不需要填写**（传了会被忽略并提示），所有评测参数放入 `custom_parameters`（JSON 格式，如 `{"metric_name": "acc", "judge_model_name": "xxx"}`）

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `exercise_id` | number | ✅ | 评测集 ID（由 `create_taiji_eval_exercise` 返回） |
| `exercise_version_name` | string | ✅ | 版本名称，例如 `v1.0` |
| `dataset_id` | number | ✅ | 数据集 ID（由 `create_taiji_eval_dataset` 返回的 `dataset_id`） |
| `dataset_version_id` | number | ✅ | 数据集版本 ID（由 `create_taiji_eval_dataset` 返回的 `dataset_version_id`） |
| `parameter_configuration` | string | ✅ | 评测参数配置类型，如 `文生文`、`多模态` 等 |
| `exercise_version_desc` | string | ❌ | 版本描述 |
| `admin` | string | ❌ | 管理员用户名，多个用逗号分隔 |
| `trial_num` | number | ❌ | 每个样本评测次数，**默认 1，调用前需向用户确认** |
| `sample_limit` | number | ❌ | 评测样本数量上限，**默认 -1（全量评测）** |
| `prompt` | object | 按模式 | Prompt 配置对象。仅多模态必填；文生文/自定义模式不需要（传了会被忽略） |
| `max_token` | number | 按模式 | 最大 token 数。仅多模态必填；文生文/自定义模式不需要（传了会被忽略） |
| `judge_model_name` | string | 按模式 | 裁判模型名称。仅多模态必填；文生文/自定义模式不需要（传了会被忽略） |
| `metric_name` | string | 按模式 | 评测指标名称。文生文/多模态必填；自定义模式不需要。**文生文默认值 `ConsistencyEval`**，用户未指定时直接用默认值并告知，不要追问 |
| `data_class_name` | string | 按模式 | 数据类型类名。文生文/多模态必填；自定义模式不需要。**文生文默认值 `OfficialTemplateData`**，用户未指定时直接用默认值并告知，不要追问 |
| `custom_parameters` | object | ❌ | 自定义参数（JSON）。填写评估的相关参数配置，比如 metric_name/judge_model_name 等，要求 JSON 格式。仅自定义模式使用 |

#### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `exercise_version_id` | number | 评测集版本 ID（创建评测任务时使用） |
| `exercise_id` | number | 所属评测集 ID |
| `exercise_version_name` | string | 版本名称 |
| `dataset_id` | number | 绑定的数据集 ID |
| `dataset_version_id` | number | 绑定的数据集版本 ID |
| `dataset_name_and_dv_name` | string | 数据集+版本组合名称 |
| `parameter_configuration` | string | 评测参数配置类型 |
| `status` | string | 版本状态 |
| `creator` | string | 创建人 |
| `create_time` | string | 创建时间 |

#### 示例

```bash
python3 scripts/connect_mcp.py call create_taiji_eval_exercise_version '{
  "exercise_id": 123,
  "exercise_version_name": "v1.0",
  "dataset_id": 1605,
  "dataset_version_id": 3615,
  "parameter_configuration": "文生文",
  "exercise_version_desc": "绑定测试skill上传的数据集"
}'
```

---

## get_taiji_eval_exercise_version_detail

#### 参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `exercise_version_id` | number | ✅ | 评测集版本 ID |

#### 返回字段

返回字段与 `create_taiji_eval_exercise_version` 基本一致，额外包含：`dv_status`（数据集版本状态）、`blackbox`（是否黑盒）、`update_time`。

#### 示例

```bash
python3 scripts/connect_mcp.py call get_taiji_eval_exercise_version_detail '{
  "exercise_version_id": 456
}'
```

---

### Exercise 完整三步创建流程

```
① exercise/create（创建评测集容器）
   → 得到 exercise_id
          ↓
② exercise/create_version（绑定数据集版本 + 配置评测参数）
   → dataset_version_id 来自 dataset/create 的返回值
   → 得到 exercise_version_id
          ↓
③ 使用 exercise_version_id 创建评测任务
✅ 完成
```

---

### 评测集（Exercise）管理——查询/修改/删除

## list_taiji_eval_exercises

分页查询评测集列表。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call list_taiji_eval_exercises '{"keyword": "评测", "page_index": 1, "page_size": 20}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `keyword` | string | ❌ | 按评测集**名称**模糊匹配（非标签/描述） |
| `exercise_tag` | string | ❌ | 按标签过滤 |
| `my_exercise` | boolean | ❌ | 仅查看我创建的评测集（默认 false） |
| `page_index` | number | ❌ | 页码（1-based，默认 1） |
| `page_size` | number | ❌ | 每页数量（默认 10） |
| `order_by` | string | ❌ | 排序字段（默认 `id` 降序） |

---

## update_taiji_eval_exercise

修改评测集名称、描述或可见范围。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call update_taiji_eval_exercise '{"exercise_id": 123, "exercise_name": "新名称", "exercise_desc": "新描述"}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `exercise_id` | number | ✅ | 评测集 ID |
| `exercise_name` | string | ✅ | 评测集名称（必填） |
| `exercise_desc` | string | ❌ | 新描述 |
| `exercise_tag` | string | ❌ | 评测集标签（可选） |
| `admin` | string | ❌ | 管理员（可选） |
| `visibility` | string | ❌ | 可见范围：`CURRENT_WORKSPACE` / `ALL_PLATFORM` / `SPECIFIED_WORKSPACES` |
| `visible_ws_ids` | string | ❌ | 可见空间 ID 列表（`visibility=SPECIFIED_WORKSPACES` 时必填，多个用逗号分隔） |

---

## delete_taiji_eval_exercise

删除评测集。🔴 **不可逆操作**，执行前必须二次确认。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call delete_taiji_eval_exercise '{"exercise_id": 123}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `exercise_id` | number | ✅ | 评测集 ID |

---

## list_taiji_eval_exercise_versions

查询评测集版本。支持三种查询模式：按评测集过滤、按管理员过滤（批量拉取）、全量拉取。

**MCP 工具调用：**

```bash
# 场景A: 查某个评测集的所有版本（原有）
python3 scripts/connect_mcp.py call list_taiji_eval_exercise_versions '{"exercise_id": 123}'

# 场景B: 拉取用户有管理权限的全部版本（新增 - "下载所有有权限version"的前置步骤）
# ⚠️ 用户说"有权限/我的/管理"时默认走此路径，admin 即权限标识
python3 scripts/connect_mcp.py call list_taiji_eval_exercise_versions '{"admin": "uricornli", "page_size": 50}'

# 场景C: 拉取用户能看到的全部版本（更宽泛的访问权限，含非 admin 的只读可见）
# ⚠️ 仅当用户明确说"能看到/可见/访问权限"时才走此路径
python3 scripts/connect_mcp.py call list_taiji_eval_exercise_versions '{}'
```

> ⚠️ **权限语义消歧规则（Agent 必须遵守）**：
> - 用户说 **"有权限 / 我的 / 我管理的 / 全部"** + 未指定具体评测集 → **默认走场景 B**（`admin` 过滤），因为用户的"权限"通常指管理权限
> - 用户明确说 **"能看到 / 可见 / 有访问权限"** → 走场景 C（不传参，范围更广）
> - 用户指定了具体评测集名称/ID → 走场景 A（`exercise_id` 过滤，原有逻辑）
> - `exercise_id` 与 `admin` 可组合使用，为 **AND 关系**

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `exercise_id` | number | ❌ | 评测集 ID。用于限定查询范围到某个评测集。**不传则返回跨全部评测集的结果** |
| `admin` | string | ❌ | **按管理员过滤（包含匹配）。用户名出现在 admin 字段中即表示对该版本有管理权限。与 `exercise_id` 为 AND 关系** |
| `keyword` | string | ❌ | 关键词搜索（按版本名模糊匹配） |
| `exercise_version_ids` | array\<number\> | ❌ | 版本 ID 列表，精确查询多个 |
| `page_index` | number | ❌ | 页码（1-based，默认 1） |
| `page_size` | number | ❌ | 每页数量（默认 10） |
| `order_by` | string | ❌ | 排序字段（默认 `id` 降序） |

---

## update_taiji_eval_exercise_version

修改评测集版本。**只需传 `exercise_version_id` + 要改的字段**，未传的字段自动保留原值。

**MCP 工具调用：**
```bash
# 只改描述
python3 scripts/connect_mcp.py call update_taiji_eval_exercise_version '{"exercise_version_id": 456, "exercise_version_desc": "新版描述"}'

# 只改裁判模型
python3 scripts/connect_mcp.py call update_taiji_eval_exercise_version '{"exercise_version_id": 456, "judge_model_name": "gpt-oss-120b"}'
```

> 💡 **编辑模式**：未传的字段从原版本补全，不会丢失。只有显式传入的字段才会更新。
> ⚠️ **RELEASED 状态限制**：已发布的版本只能改 `admin`，其他字段变更会报错。需先 `clone_taiji_eval_exercise_version` 复制出新版本再改。

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `exercise_version_id` | number | ✅ | 版本 ID |
| `exercise_version_name` | string | ❌ | 版本名，不传保留原值 |
| `exercise_version_desc` | string | ❌ | 描述，传空字符串可清空 |
| `dataset_id` | number | ❌ | 数据集 ID，不传保留原值 |
| `dataset_version_id` | number | ❌ | 数据集版本 ID，不传保留原值 |
| `parameter_configuration` | string | ❌ | 评测参数配置类型（如 `文生文`），不传保留原值 |
| `admin` | string | ❌ | 管理员 |
| `trial_num` | number | ❌ | 每个样本评测次数 |
| `sample_limit` | number | ❌ | 评测样本数量上限 |
| `prompt` | object | ❌ | 评测 prompt 配置 |
| `max_token` | number | ❌ | 最大 token 数 |
| `judge_model_name` | string | ❌ | 裁判模型名称 |
| `metric_name` | string | ❌ | 评测指标名称 |
| `data_class_name` | string | ❌ | 数据类型类名 |
| `custom_parameters` | object | ❌ | 自定义参数对象 |

---

## delete_taiji_eval_exercise_version

删除评测集版本。🔴 **不可逆操作**，执行前必须二次确认。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call delete_taiji_eval_exercise_version '{"exercise_version_id": 456}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `exercise_version_id` | number | ✅ | 版本 ID |

---

## clone_taiji_eval_exercise_version

复制评测集版本。复制后新版本名称为原名称加 `_copy` 后缀，状态为 `PENDING`。clone 后如需要可调用 `update_taiji_eval_exercise_version` 修改名称、描述等字段。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call clone_taiji_eval_exercise_version '{"exercise_version_id": 456}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `exercise_version_id` | number | ✅ | 要复制的源评测集版本 ID |

**返回字段**：同 `get_taiji_eval_exercise_version_detail`，新版本 `exercise_version_name` 加 `_copy` 后缀，`status` 为 `PENDING`。

---

### Exercise 验证任务

## clone_taiji_eval_exercise_validation

从已有评测任务复制创建 Exercise 验证任务（`EXERCISE_VALIDATION`）。继承源任务的模型配置、推理参数等，替换为指定的 `exercise_version_id`。验证任务挂在单个评测集版本上，不挂评测集合。

与 `clone_taiji_eval_task` 的区别：
- 验证任务 `task_type=EXERCISE_VALIDATION`，挂在单个 exercise_version 上，不使用 collection_version_ids
- 验证任务强制 `service_reuse_type=NORMAL`，不走服务复用
- `branch` 和 `commit` 为必填，用于标记验证的代码版本

### Agent 行为指引

1. **源任务必须是验证任务**：`source_task_id` 指向的源任务必须是 `task_type=EXERCISE_VALIDATION` 的验证任务，不能是普通评测任务。

   如果用户没有指定 source_task_id：
   - 不要猜测，先问用户想基于哪个已有验证任务来创建
   - 可用 `list_taiji_eval_tasks` 帮用户查找，建议筛选条件：
     - `task_type=EXERCISE_VALIDATION`（必须，只查验证任务）
     - `my_task=true`（只看自己的任务）
     - `status=PARSED`（只看成功的任务，排除 FAILED 的）
   - 查到后列出任务 ID、名称、状态、模型名供用户选择

   如果用户已提供 source_task_id：
   - 必须先查源任务的 task_type 校验是否为验证任务：
   ```bash
   python3 scripts/connect_mcp.py call get_taiji_eval_task_detail '{"task_id": 103741}' 2>&1 | grep -o '"task_type": "[^"]*"'
   # 期望输出: "task_type": "EXERCISE_VALIDATION"
   ```
   - 如果不是验证任务类型，告知用户："源任务必须是验证任务（EXERCISE_VALIDATION）类型，当前类型为 XXX，请提供一个验证任务 ID"
   - 如果源任务状态为 FAILED，建议用户选择一个成功的验证任务（status=PARSED）

2. **查源任务 modelSource（前置必做）**：确定 `source_task_id` 后，必须先查源任务的 `model_source`，再根据 modelSource 告知用户哪些参数可以覆盖：
   ```bash
   python3 scripts/connect_mcp.py call get_taiji_eval_task_detail '{"task_id": <source_task_id>}' 2>&1 | grep -o '"model_source": "[^"]*"'
   ```

3. **branch 和 commit 必须由用户提供**：这两个字段标记验证的代码版本，Agent 不能猜测或自动生成。如果用户没提供，必须询问：
   - "请提供要验证的代码分支名（branch）和 commit hash"

4. **exercise_version_id、name、desc 必须由用户提供**：Agent 不能猜测或自动生成。如果用户没提供，必须逐个询问。

5. **创建前二次确认**：调用接口前，必须向用户展示完整参数摘要并确认：
   - 源任务 ID + modelSource + 模型名
   - 目标 exercise_version_id
   - branch / commit
   - name / desc
   - 用户指定的覆盖参数（如有）
   - 确认后再调用接口

#### 必填参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `source_task_id` | int | 源任务 ID，必须是 EXERCISE_VALIDATION 类型的验证任务 |
| `exercise_version_id` | int | 要验证的评测集版本 ID |
| `branch` | string | 代码分支名，存入 extraInfo |
| `commit` | string | commit hash，存入 extraInfo |
| `name` | string | 新验证任务名称 |
| `desc` | string | 任务描述 |

#### 可选参数（按 modelSource 分组，与 clone_taiji_eval_task 一致）

| 分类 | 参数 | 类型 | modelSource | 说明 |
|------|------|------|-------------|------|
| 服务参数 | service_name | string | `ONE_STOP_SERVICE` / `DISTILLATION_API` | 服务名 |
| | crawl_para_num | int | `ONE_STOP_SERVICE` / `DISTILLATION_API` | 并发数 |
| 部署参数 | model_ids | array\<int\> | `MODEL_REPOSITORY` / `MODEL_GROUP` | 模型 ID 列表 |
| | max_context_length | int | `MODEL_REPOSITORY` / `MODEL_GROUP` | 上下文长度 |
| | replicas | int | `MODEL_REPOSITORY` / `MODEL_GROUP` | 推理副本数 |
| | host_gpu_num | int | `MODEL_REPOSITORY` / `MODEL_GROUP` | 单机 GPU 数 |
| | gpu_name | string | `MODEL_REPOSITORY` / `MODEL_GROUP` | GPU 型号 |
| | queue_name | string | `MODEL_REPOSITORY` / `MODEL_GROUP` | 队列名称 |
| | location | string | `MODEL_REPOSITORY` / `MODEL_GROUP` | 地域 |
| | resource_types | array\<string\> | `MODEL_REPOSITORY` / `MODEL_GROUP` | 资源类型(private/public/elastic) |
| 推理参数 | image | string | `MODEL_REPOSITORY` | 镜像地址 |
| | max_batch_size | int | `MODEL_REPOSITORY` | 单实例批大小 |
| | compression_strategy | string | `MODEL_REPOSITORY` | 压缩策略 |
| | envs | string | `MODEL_REPOSITORY` | 环境变量 |
| | reasoning_parser | string | `MODEL_REPOSITORY` | 推理解析器 |
| | tool_parser | string | `MODEL_REPOSITORY` | 工具解析器 |
| 通用覆盖 | parameter_configuration | object | 所有类型 | 模型调用参数，deep merge |
| | hy_api_protocol | string | 所有类型 | 调用协议 |
| | hy_api_env | string | 所有类型 | API 环境 |

> 💡 调用前建议先查源任务的 `model_source`，避免传了不支持的参数。

#### MCP 调用示例

```bash
# 最简创建（仅必填参数）
python3 scripts/connect_mcp.py call clone_taiji_eval_exercise_validation '{
  "source_task_id": 8828,
  "exercise_version_id": 5179,
  "branch": "feat/new-scorer",
  "commit": "a1b2c3d4",
  "name": "MMLU验证任务-v2",
  "desc": "验证MMLU评测集新scorer代码"
}'

# 覆盖上下文长度（MODEL_REPOSITORY）
python3 scripts/connect_mcp.py call clone_taiji_eval_exercise_validation '{
  "source_task_id": 8828,
  "exercise_version_id": 5179,
  "branch": "feat/new-scorer",
  "commit": "a1b2c3d4",
  "name": "MMLU验证任务-v2",
  "desc": "验证MMLU评测集新scorer代码",
  "max_context_length": 4096
}'
```

#### 返回

| 字段 | 描述 |
|------|------|
| `id` | 新验证任务 ID |
| `name` | 任务名称（自动加 `_validation` 后缀或用户指定） |
| `status` | `PENDING` |
| `task_type` | `EXERCISE_VALIDATION` |
| `service_reuse_type` | `NORMAL`（强制） |
| `collection_version_ids` | 空字符串 |

#### 创建后确认

```bash
python3 scripts/connect_mcp.py call get_taiji_eval_task_detail '{"task_id": <新任务ID>}' 2>&1 | grep -E '"status":|"task_type":|"exercise_version_id":|"branch":|"commit":'
```

---

### 下载评测集文件

> 场景：用户要求**下载评测集（Exercise）的数据文件**。评测集版本本身不存储文件，它绑定了一个数据集版本（DatasetVersion），实际文件存储在数据集版本上。
>
> **核心原理**：`get_taiji_eval_exercise_version_detail` 返回中包含 `datasetVersionId` 字段 → 拿到该 ID 后 → 调用 `download_taiji_eval_dataset_version_file`（定义在 [dataset_management_api.md](./dataset_management_api.md)）即可下载。
>

#### 场景区分

| 用户意图 | 入口 | 获取 `dataset_version_id` 的方式 |
|---|---|---|
| **下载数据集版本** | 用户直接给数据集 + 版本名 | `list_taiji_eval_dataset_versions` 按 `version_name` 反查 → 📄 [dataset_management_api.md](./dataset_management_api.md) |
| **下载评测集** | 用户给评测集版本 ID 或名称 | `get_taiji_eval_exercise_version_detail` 返回中的 `data.datasetVersionId`（本节） |

#### 工作流

```
① 收集必填信息
   向用户确认：exercise_version_id（评测集版本 ID）
             （若用户只给了名称，需先调 list_taiji_eval_exercises / list_taiji_eval_exercises_versions 反查）
        ↓
② 查询评测集版本详情，获取绑定的数据集版本 ID
   调用 get_taiji_eval_exercise_version_detail({"exercise_version_id": ...})
   → 从返回 data 中取 datasetVersionId
   → 同时可展示给用户：绑定的数据集名 (datasetNameAndDvName)、评测参数等元信息
        ↓
③ 下载数据集版本文件（跨文档调用 dataset_management_api 的工具）
   调用 download_taiji_eval_dataset_version_file({"dataset_version_id": 上一步取到的 datasetVersionId})
   → base64 解码，写入本地临时文件
        ↓
④ 告知用户下载完成
   输出文件名、大小、本地路径
```

**MCP 工具调用示例：**
```bash
# Step 1: 查评测集版本详情，拿 datasetVersionId
python3 scripts/connect_mcp.py call get_taiji_eval_exercise_version_detail '{"exercise_version_id": 5344}'
# 返回: { "data": { "id": 5344, "datasetVersionId": 5033, "datasetNameAndDvName": "pipeline-e1-dataset@v20260720", ... } }

# Step 2: 用 datasetVersionId 下载数据集文件
python3 scripts/connect_mcp.py call download_taiji_eval_dataset_version_file '{"dataset_version_id": 5033}'
# 返回: { "file_name": "test_data.jsonl", "file_size": 260, "file_content_base64": "..." }
```

> ⚠️ **注意**：
> - 若用户同时要求「下载后修改并重新上传为新评测集版本」，则在步骤③④之间插入修改逻辑，最后调用 `create_taiji_eval_exercise_version` 创建新的评测集版本（绑定到新建的数据集版本）。
> - 评测集下载依赖的是**已有的两个工具**组合（本文件的 `get_taiji_eval_exercise_version_detail` + dataset_management_api 的 `download_taiji_eval_dataset_version_file`），不是独立的新接口。

---

### 裁判模型调优流程

裁判模型调优是一个**验证→分析→调整→再验证**的闭环，每一步都需要向用户确认后再执行，严禁自动跳步。

> 每一步执行前，Agent 必须向用户说明即将做什么、用什么参数，等用户明确同意后才调用接口。

### 流程

#### Step 1：创建验证任务

Agent 必须先问用户：
- 要基于哪个已有评测任务（source_task_id）创建验证任务？
- 要验证哪个评测集版本（exercise_version_id）？
- branch / commit 是什么？

确认后才能调 `clone_taiji_eval_exercise_validation`。

#### Step 2：查看验证任务结果

Agent 必须先问用户：验证任务已完成，是否查看结果？

用户同意后，调 `get_taiji_eval_task_detail` 查看任务状态和评分。可结合 `evaluation_result_analysis_api.md` 和 `drill_api.md` 里的工具做置信区间 / 下钻分析。

#### Step 3：根据结果调整评测集配置

Agent 必须先问用户：
- 根据验证结果，你觉得哪里需要调整？（不要替用户判断）
- 确认要改哪些字段、改成什么值？

用户明确后才能调 `update_taiji_eval_exercise_version`。如果版本已 RELEASED，需先 `clone_taiji_eval_exercise_version` 复制新版本，这一步也要问用户。

#### Step 4：重新验证

Agent 必须先问用户：配置已更新，是否重新创建验证任务对比效果？

用户同意后，回到 Step 1。

### 禁止事项

- 不得自动连续执行（如创建验证任务后直接查结果再直接改配置）
- 不得替用户决定改哪些字段或改成什么值
- 不得在用户未确认的情况下 clone 新版本

---

