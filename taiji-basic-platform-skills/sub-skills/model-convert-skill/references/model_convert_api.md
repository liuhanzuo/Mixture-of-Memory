## list_hunyuan_training_model_convert_options

**用途**：查询指定模型支持的转换类型和导出格式。**建议在创建转换任务前先调用此接口**，确认模型支持的转换类型。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `model_id` | int | ⭕ 与 `model_name` 二选一 | - | 模型 ID（优先使用） |
| `model_name` | str | ⭕ 与 `model_id` 二选一 | - | 模型名称 |

> ⚠️ `model_id` 和 `model_name` 必须至少提供一个。如果同时提供，优先使用 `model_id`。

### 调用示例

```bash
python3 scripts/connect_mcp.py call list_hunyuan_training_model_convert_options '{"model_name": "hunyuan-7b"}'
```

```bash
python3 scripts/connect_mcp.py call list_hunyuan_training_model_convert_options '{"model_id": 123}'
```

### 返回字段说明（`data` 内部结构）

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_id` | int | 模型 ID |
| `model_name` | str | 模型名称 |
| `convert_options` | array | 支持的转换选项列表 |
| `convert_options[].convert_type` | str | 转换类型值（创建任务时传入 `convert_type` 字段），如 `TO_HF`、`TO_MCORE`、`mcore_ptm2_to_hf`、`mcore_angelrl_to_hf`、`DCP_TO_HF` 等 |
| `convert_options[].text` | str | 转换类型显示名称（如 `转HF`） |
| `convert_options[].description` | str | 转换类型功能说明（如 `PTM或非标准HF转HF`） |
| `convert_options[].is_default` | bool | 是否为默认推荐选项 |
| `convert_options[].need_hf_model_path` | bool | 是否需要用户提供 HF 模型路径（`hf_model_path`） |
| `convert_options[].need_vit_config_path` | bool | 是否需要用户提供 ViT 配置目录路径 |
| `convert_options[].export_format_options` | array | 支持的导出格式选项列表（创建任务时传入 `export_format` 字段） |
| `convert_options[].export_format_options[].export_format` | str | 导出格式值（如 `HF`、`Mcore+HF`） |
| `convert_options[].export_format_options[].description` | str | 导出格式功能说明（如 `单独产出HF格式模型卡片。转换后的模型，默认存储在【转换应用组】的产出路径下`） |
| `convert_options[].default_export_format` | str | 默认导出格式值（如 `HF`） |

### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "model_id": 123,
    "model_name": "hunyuan-7b",
    "convert_options": [
      {
        "convert_type": "TO_HF",
        "text": "转HF",
        "description": "PTM或非标准HF转HF",
        "is_default": true,
        "need_hf_model_path": false,
        "need_vit_config_path": false,
        "export_format_options": [
          {
            "export_format": "HF",
            "description": "单独产出HF格式模型卡片。转换后的模型，默认存储在【转换应用组】的产出路径下"
          }
        ],
        "default_export_format": "HF"
      },
      {
        "convert_type": "TO_MCORE",
        "text": "转Mcore",
        "description": "HF转Mcore",
        "is_default": false,
        "need_hf_model_path": true,
        "need_vit_config_path": false,
        "export_format_options": [
          {
            "export_format": "Mcore",
            "description": "单独产出Mcore格式模型卡片"
          },
          {
            "export_format": "Mcore+HF",
            "description": "同时产出Mcore+HF双格式模型卡片"
          }
        ],
        "default_export_format": "Mcore"
      }
    ]
  }
}
```

> 💡 **提示**：
> - 转换类型列表是动态的，不同模型支持的转换类型可能不同。如果用户指定的转换类型不在返回列表中，应提示用户选择支持的类型。
> - 展示给用户时优先使用 `text` 与 `description` 字段（人类可读），把 `convert_type` 作为最终传参值。
> - 若某个 `convert_type` 的 `need_hf_model_path=true`，则创建任务时必须要求用户提供 `hf_model_path` / `hf_template_id` / `hf_template_name`（三选一）。

---

## create_hunyuan_training_model_convert_task

**用途**：创建混元训练平台的模型格式转换任务，支持多种转换类型（如 `mcore_ptm2_to_hf`、`mcore_angelrl_to_hf`、`TO_HF`、`TO_MCORE`、`DCP_TO_HF` 等）。创建成功后任务处于 `INIT` 状态，需要再调用 `start_hunyuan_training_model_convert_task` 启动。

> ⚠️ **必须确认的信息（最高优先级）：**
> 1. **wsid**（工作空间 ID）：必填，不能为 0
> 2. **模型标识**：`model_id` 和 `model_name` 至少提供一个
> 3. **convert_type**（转换类型）：必填，建议先调用 `list_hunyuan_training_model_convert_options` 查询支持的类型
> 4. **资源配置**：`business_flag`、`host_num`、`host_gpu_num`、`gpu_name`、`location` 均为必填
> 5. **导出模型名称**：默认采用自动导出（`export_type = "auto"`），因此 `export_model_name`（导出模型名称）为必填项，需提醒用户提供；`export_model_desc`（导出模型描述）为可选项，不传时后台会自动生成默认描述。**⚠️ 例外：当 `source = "TRAIN_OUTPUT"` 且导出格式为多格式（如 `Mcore+HF`、`DCP+HF`、`PTM+HF`）时，`export_model_name` 和 `export_model_desc` 均不需要用户提供，调用接口时也不需要传这两个参数，后台会自动处理**

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID，不能为 0 |
| `convert_type` | str | ✅ 必填 | - | 转换类型，如 `mcore_ptm2_to_hf`、`mcore_angelrl_to_hf`、`TO_HF`、`TO_MCORE`、`DCP_TO_HF` 等 |
| `business_flag` | str | ✅ 必填 | - | 应用组标识，通过 `query_app_group_list` 获取 |
| `host_num` | int | ✅ 必填 | - | GPU 机器数（通常为 `1`） |
| `host_gpu_num` | float | ✅ 必填 | - | 单台机器 GPU 卡数（后端为 `Float` 类型，支持小数如 `0.5`；常见取值：`1`、`2`、`4`、`8`） |
| `gpu_name` | str | ✅ 必填 | - | GPU 卡型（如 `H20`、`H800`、`A100`、`A800`） |
| `location` | str | ✅ 必填 | - | 地域（使用拼音缩写，如深圳=`sz`、广州=`gz`、上海=`sh`、北京=`bj`、天津=`tj`、重庆=`cq`、成都=`cd`、南京=`nj`、武汉=`wh`、长沙=`cs`） |
| `model_id` | int | ⭕ 与 `model_name` 二选一 | - | 源模型 ID（优先使用） |
| `model_name` | str | ⭕ 与 `model_id` 二选一 | - | 源模型名称 |
| `name` | str | ❌ 可选 | - | 任务名称（不传则自动生成，最长 255 字符） |
| `tokenizer_option` | str | ❌ 可选 | - | Tokenizer 配置来源选项：`PLATFORM_BUILT_IN`（平台内置）/ `CUSTOMIZE`（自定义） |
| `tokenizer` | str | ❌ 可选 | - | 平台内置 tokenizer 名称（`tokenizer_option` 为 `PLATFORM_BUILT_IN` 时使用） |
| `tokenizer_path` | str | ❌ 可选 | - | 自定义 tokenizer 路径（`tokenizer_option` 为 `CUSTOMIZE` 时使用） |
| `hf_model_path` | str | ❌ 可选 | - | HF 模板路径（当直接指定路径而非模板 ID 时使用） |
| `hf_template_id` | long | ❌ 可选 | - | 已注册的 HF 模板 ID（优先级最高，后端为 `Long` 类型） |
| `hf_template_name` | str | ❌ 可选 | - | 已注册的 HF 模板名称（推荐，后端自动解析） |
| `export_type` | str | ❌ 可选 | `"auto"` | 导出方式：`auto`（自动，默认）/ `manual`（手动） |
| `export_model_name` | str | ❌ 可选 | - | 自动导出模型时的模型名称（`export_type = "auto"` 时**必填**；**但 `source = "TRAIN_OUTPUT"` 且导出格式为多格式时不需要传，后台自动处理**） |
| `export_model_desc` | str | ❌ 可选 | - | 自动导出模型时的模型描述 |
| `export_format` | str | ❌ 可选 | - | 导出模型格式，如：`HF` / `Mcore+HF` |
| `debug_mode` | bool | ❌ 可选 | `false` | 是否开启 Debug 模式 |
| `debug_alive_time` | int | ❌ 可选 | `8640000` | Debug 模式保留时长（秒） |
| `env_vars_dict` | object | ❌ 可选 | - | 自定义环境变量字典（**object 类型**，key 为变量名，value 为字符串值） |
| `extra_plat_business` | str | ❌ 可选 | - | 额外挂载的应用组 |
| `quota_type` | str | ❌ 可选 | - | 配额类型（如 `private`/`public`/`discount`/`tidal`/`elastic`/`mixing`） |
| `output_ceph_dir` | str | ❌ 可选 | - | 产出模型的存储目录（Ceph 路径） |
| `source` | str | ❌ 可选 | - | 任务来源标记：`MODEL_CARD`（模型卡片，默认）/ `TRAIN_OUTPUT`（训练产出）。当用户明确指定来源为「模型卡片」时传 `MODEL_CARD`，明确指定为「训练产出」时传 `TRAIN_OUTPUT`；当前会话中前序步骤涉及「发布训练产出为模型卡片」时也应传 `TRAIN_OUTPUT`；其余情况可不传或传 `MODEL_CARD` |

> **参数互斥关系：**
> - **模型标识**（二选一）：`model_id` > `model_name`
> - **HF 配置**（三选一）：`hf_template_id` > `hf_template_name` > `hf_model_path`
> - **Tokenizer**（二选一）：`PLATFORM_BUILT_IN` → 用 `tokenizer`；`CUSTOMIZE` → 用 `tokenizer_path`
> - **导出方式**：默认 `export_type = "auto"`（自动导出），此时 `export_model_name` 必填（需提醒用户提供）；`export_model_desc` 可选，不传时后台自动生成默认描述。**⚠️ 例外：当 `source = "TRAIN_OUTPUT"` 且导出格式为多格式（如 `Mcore+HF`、`DCP+HF`、`PTM+HF`）时，`export_model_name` 和 `export_model_desc` 均不需要传，后台自动处理**
> - **模型来源**：`source` 默认为 `MODEL_CARD`。当用户明确指定来源为「模型卡片」时传 `MODEL_CARD`，明确指定为「训练产出」时传 `TRAIN_OUTPUT`；当前会话中前序步骤涉及「发布训练产出为模型卡片」后再转换时自动设置为 `TRAIN_OUTPUT`

### 调用示例

**最简调用**（仅必填参数，默认自动导出需提供导出模型名称）：

```bash
python3 scripts/connect_mcp.py call create_hunyuan_training_model_convert_task '{
  "wsid": 456,
  "model_name": "hunyuan-7b",
  "convert_type": "mcore_ptm2_to_hf",
  "business_flag": "my_app_group",
  "host_num": 1,
  "host_gpu_num": 1,
  "gpu_name": "H20",
  "location": "sz",
  "export_model_name": "hunyuan-7b-hf"
}'
```

**完整调用**（含可选参数）：

```bash
python3 scripts/connect_mcp.py call create_hunyuan_training_model_convert_task '{
  "wsid": 456,
  "model_id": 123,
  "convert_type": "TO_MCORE",
  "business_flag": "my_app_group",
  "host_num": 1,
  "host_gpu_num": 2,
  "gpu_name": "H800",
  "location": "sz",
  "hf_template_name": "llama-7b-hf",
  "export_type": "auto",
  "export_model_name": "hunyuan-7b-mcore",
  "export_model_desc": "Mcore 格式的 hunyuan-7b",
  "export_format": "Mcore+HF",
  "quota_type": "private",
  "env_vars_dict": {"MY_ENV": "value1", "ANOTHER": "value2"}
}'
```

**发布训练产出后转换**（前序步骤中将训练产出发布为模型卡片后再转换，多格式导出无需 `export_model_name`）：

```bash
python3 scripts/connect_mcp.py call create_hunyuan_training_model_convert_task '{
  "wsid": 456,
  "model_name": "hunyuan-7b-sft-v1",
  "convert_type": "mcore_ptm2_to_hf",
  "business_flag": "my_app_group",
  "host_num": 1,
  "host_gpu_num": 1,
  "gpu_name": "H20",
  "location": "sz",
  "source": "TRAIN_OUTPUT"
}'
```

### 返回字段说明（`data` 内部结构）

创建接口返回一个 **完整的 `OpenModelConvertTaskInfo`**（与 `get_task_detail` 一致），下表仅列创建后用户最关心的字段，完整字段请参阅「get_hunyuan_training_model_convert_task_detail」章节。

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | str | 任务 ID（如 `finetuning_123456`） |
| `task_int_id` | long | 任务数字 ID |
| `name` | str | 任务名称 |
| `model_id` | int | 模型 ID |
| `model_name` | str | 模型名称 |
| `convert_type` | str | 转换类型 |
| `hy_status` | str | 统一任务状态（创建后为 `INIT`） |
| `hy_status_text` | str | 统一状态中文描述（如"已创建"） |
| `wsid` | long | 工作空间 ID |

### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "task_id": "finetuning_123456",
    "task_int_id": 123456,
    "name": "mcore_ptm2_to_hf_hunyuan-7b_20260415160000",
    "model_id": 123,
    "model_name": "hunyuan-7b",
    "convert_type": "mcore_ptm2_to_hf",
    "hy_status": "INIT",
    "hy_status_text": "已创建",
    "export_type": "auto",
    "export_model_name": "hunyuan-7b-hf",
    "wsid": 456
  }
}
```

> 💡 **提示**：创建接口返回体中 **没有 `message` 业务字段**（外层已有 `code`+`message`）。如需向用户展示"任务创建成功"，请基于外层 `code == 0` 自行拼描述。

---

## start_hunyuan_training_model_convert_task

**用途**：启动已创建的模型转换任务。只有状态为 `INIT`（已创建）的任务可以启动，启动后任务进入 `RUNNING` 状态。支持指定重试次数、执行时间、Cron 定时调度等参数。

> ⚠️ **注意事项：**
> - 只有 `INIT` 状态的任务可以启动。如果任务状态不是 `INIT`，调用会失败。
> - 通常在 `create_hunyuan_training_model_convert_task` 创建任务成功后立即调用本工具启动任务。
> - 支持定时执行，通过 `exec_time` 参数指定执行时间；支持周期性调度，通过 `cron_expression` / `stop_cron_expression` 参数配置。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `task_id` | str | ✅ 必填 | - | 要启动的模型转换任务 ID（如 `finetuning_123456`） |
| `retry` | int | ❌ 可选 | - | 失败重试次数 |
| `exec_time` | str | ❌ 可选 | - | 定时执行时间，格式如 `yyyy-MM-dd HH:mm:ss`（空则立即执行） |
| `cron_expression` | str | ❌ 可选 | - | Cron 表达式，用于周期性调度启动 |
| `stop_cron_expression` | str | ❌ 可选 | - | Cron 表达式，用于周期性调度停止 |

### 调用示例

**立即启动**：

```bash
python3 scripts/connect_mcp.py call start_hunyuan_training_model_convert_task '{"task_id": "finetuning_123456"}'
```

**定时启动**：

```bash
python3 scripts/connect_mcp.py call start_hunyuan_training_model_convert_task '{"task_id": "finetuning_123456", "exec_time": "2026-04-16 10:00:00"}'
```

### 返回字段说明（`data` 内部结构）

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | str | 任务名称 |
| `task_id` | str | 任务 ID |
| `instance_id` | str | 实例 ID |
| `wsid` | long | 工作空间 ID |
| `instance_url` | str | 实例详情页 URL，可供用户查看任务执行进度 |

### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "name": "mcore_ptm2_to_hf_hunyuan-7b_20260415160000",
    "task_id": "finetuning_123456",
    "instance_id": "instance_789",
    "wsid": 456,
    "instance_url": "https://taiji.woa.com/workspace/456/model-convert/finetuning_123456/instance/instance_789"
  }
}
```

> 💡 **提示**：启动成功后，建议将 `instance_url` 返回给用户，方便用户在浏览器中查看任务执行进度和日志。

---

## get_hunyuan_training_model_convert_task_detail

**用途**：根据 `task_id` 查询混元训练平台指定模型转换任务的详情，返回任务的完整信息（状态、进度、模型配置、导出信息、创建时间等）。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `task_id` | str | ✅ 必填 | - | 模型转换任务 ID（如 `finetuning_123456`） |

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_training_model_convert_task_detail '{"task_id": "finetuning_123456"}'
```

### 返回字段说明（`data` 内部结构）

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | str | 任务 ID（如 `finetuning_123456`） |
| `task_int_id` | long | 任务数字 ID |
| `creator` | str | 任务创建者 |
| `model_id` | int | 模型 ID |
| `model_name` | str | 模型名称 |
| `name` | str | 任务名称 |
| `convert_type` | str | 转换类型 |
| `tokenizer_option` | str | Tokenizer 选项（`PLATFORM_BUILT_IN` / `CUSTOMIZE`） |
| `tokenizer` | str | 平台内置 tokenizer 名称 |
| `tokenizer_path` | str | 自定义 tokenizer 路径 |
| `hf_model_path` | str | HF 模板路径 |
| `hf_template_id` | long | HF 模板 ID |
| `hf_template_name` | str | HF 模板名称 |
| `resource_config` | object | 资源配置（结构见下方） |
| `status` | str | 任务内部状态（底层处理引擎原始状态，如 `SUBMITTED`、`RUNNING`、`SUCCEEDED` 等） |
| `status_text` | str | 内部状态中文描述 |
| `hy_status` | str | 统一任务状态（`INIT` / `RUNNING` / `SUCCESS` / `FAILED` / `KILLED`）—**推荐展示给用户的状态字段** |
| `hy_status_text` | str | 统一状态中文描述（如"已创建"、"运行中"、"成功"、"失败"、"已停止"） |
| `export_type` | str | 导出方式（`manual` / `auto`） |
| `export_model_name` | str | 导出模型名称 |
| `export_model_desc` | str | 导出模型描述 |
| `export_format` | str | 导出格式（如 `HF`、`Mcore+HF`） |
| `wsid` | long | 工作空间 ID |
| `model_url` | str | 模型详情页 URL |
| `task_url` | str | 任务详情页 URL |

**`resource_config` 字段结构（字段较多，以下仅列常用子集）**：

| 子字段 | 类型 | 说明 |
|------|------|------|
| `business_flag` | str | 应用组标识 |
| `host_num` | int | GPU 机器数 |
| `host_gpu_num` | float | 单台机器 GPU 卡数 |
| `gpu_name` | str | GPU 卡型 |
| `location` | str | 地域拼音缩写 |
| `quota_type` | str | 配额类型 |
| `image_name` | str | 运行环境镜像 |
| `debug_mode` | bool | 是否开启 Debug 模式 |
| `debug_alive_time` | int | Debug 模式保留时长（秒） |
| `env_vars_dict` | object | 自定义环境变量（key-value） |
| `extra_plat_business` | str | 额外挂载的应用组 |

> ⚠️ `resource_config` 完整包含 40+ 字段（包括 RDMA、存储配额、自动续训、KubeRay、弹性任务、错误处理等）。上表只列与模型转换直接相关的子集，其他字段以 API 实际返回为准，向用户展示时只需关注上述关键子集即可。

### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "task_id": "finetuning_123456",
    "task_int_id": 123456,
    "creator": "user_abc",
    "model_id": 123,
    "model_name": "hunyuan-7b",
    "name": "mcore_ptm2_to_hf_hunyuan-7b_20260415160000",
    "convert_type": "mcore_ptm2_to_hf",
    "tokenizer_option": "",
    "tokenizer": "",
    "tokenizer_path": "",
    "hf_model_path": "",
    "hf_template_id": 0,
    "hf_template_name": "",
    "resource_config": {
      "business_flag": "my_app_group",
      "host_num": 1,
      "host_gpu_num": 1,
      "gpu_name": "H20",
      "location": "sz",
      "quota_type": "private",
      "debug_mode": false,
      "env_vars_dict": {}
    },
    "status": "RUNNING",
    "status_text": "运行中",
    "hy_status": "RUNNING",
    "hy_status_text": "运行中",
    "export_type": "manual",
    "export_model_name": "",
    "export_model_desc": "",
    "export_format": "HF",
    "wsid": 456,
    "model_url": "https://taiji.woa.com/workspace/456/model/123",
    "task_url": "https://taiji.woa.com/workspace/456/model-convert/finetuning_123456"
  }
}
```

> 💡 **提示**：
> - `status` / `status_text` 为底层引擎原始状态（内部字段），`hy_status` / `hy_status_text` 为平台统一展示状态（面向用户）。展示时**优先使用 `hy_status_text`**。
> - 展示结果应以清晰的格式展示给用户，重点突出任务状态、转换类型、模型名称、模型链接和任务 URL 等关键信息。

---

## get_hunyuan_training_latest_model_convert_task

**用途**：查询指定模型（`model_id` 或 `model_name`）在指定工作空间下最新的一次模型转换任务详情，用于快速获取最近一次的转换结果或状态。当用户提供模型名称或模型 ID（而非具体的 `task_id`）查询转换状态时，使用本工具。

> ⚠️ **注意事项：**
> - `wsid` 为必填参数，如果用户未提供，需先按 `references/helper_api.md`（`list_user_workspaces`）获取或复用上下文中的 wsid。
> - `model_id` 和 `model_name` 必须至少提供一个。如果同时提供，优先使用 `model_id`。
> - 本工具只返回该模型最新的一条转换任务，如需查看历史任务请使用其他方式。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID，不能为 0 |
| `model_id` | int | ⭕ 与 `model_name` 二选一 | - | 源模型 ID（优先使用） |
| `model_name` | str | ⭕ 与 `model_id` 二选一 | - | 源模型名称 |

### 调用示例

**按模型名称查询**：

```bash
python3 scripts/connect_mcp.py call get_hunyuan_training_latest_model_convert_task '{"wsid": 456, "model_name": "hunyuan-7b"}'
```

**按模型 ID 查询**：

```bash
python3 scripts/connect_mcp.py call get_hunyuan_training_latest_model_convert_task '{"wsid": 456, "model_id": 123}'
```

### 返回字段说明（`data` 内部结构）

返回体结构与 `get_hunyuan_training_model_convert_task_detail` 完全相同（均为 `OpenModelConvertTaskInfo`），详细字段参阅「get_hunyuan_training_model_convert_task_detail」章节。包括：`task_id`、`task_int_id`、`creator`、`model_id`、`model_name`、`convert_type`、`tokenizer_option`、`tokenizer`、`tokenizer_path`、`hf_model_path`、`hf_template_id`、`hf_template_name`、`resource_config`、`status`、`status_text`、`hy_status`、`hy_status_text`、`export_type`、`export_model_name`、`export_model_desc`、`export_format`、`wsid`、`model_url`、`task_url` 等。

### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "task_id": "finetuning_123456",
    "task_int_id": 123456,
    "creator": "user_abc",
    "model_id": 123,
    "model_name": "hunyuan-7b",
    "name": "mcore_ptm2_to_hf_hunyuan-7b_20260415160000",
    "convert_type": "mcore_ptm2_to_hf",
    "status": "SUCCEEDED",
    "status_text": "成功",
    "hy_status": "SUCCESS",
    "hy_status_text": "成功",
    "export_type": "manual",
    "export_model_name": "",
    "export_format": "HF",
    "wsid": 456,
    "model_url": "https://taiji.woa.com/workspace/456/model/123",
    "task_url": "https://taiji.woa.com/workspace/456/model-convert/finetuning_123456"
  }
}
```

> 💡 **路由提示**：当用户说"查看模型 xxx 的转换进度"或"模型 xxx 转换状态"时，应使用本工具（按模型查询最新任务），而非 `get_hunyuan_training_model_convert_task_detail`（按 task_id 查询）。

---

## clone_hunyuan_training_model_convert_task

**用途**：克隆一个已存在的模型转换任务，可以在克隆时修改源模型（`model_id` / `model_name`）以及导出模型的名称与描述。返回新克隆出来的转换任务信息（含新 `task_id`）。克隆成功后新任务处于 `INIT` 状态，需要再调用 `start_hunyuan_training_model_convert_task` 启动。

> **适用场景：**
> - 参考某个转换任务的配置，为另一个模型创建相同配置的转换任务
> - 快速复制任务配置，减少重复填写参数
> - 批量转换多个模型时，先为第一个模型创建任务，后续模型通过克隆方式快速创建

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID，不能为 0 |
| `task_id` | str | ✅ 必填 | - | 被克隆的源转换任务 ID（如 `finetuning_123456`），将复用该任务的所有配置 |
| `model_id` | int | ❌ 可选 | - | 克隆时可覆盖的源模型 ID（不传则复用原任务的模型） |
| `model_name` | str | ❌ 可选 | - | 克隆时可覆盖的源模型名称（不传则复用原任务的模型） |
| `export_model_name` | str | ❌ 可选 | - | 克隆时可覆盖的导出模型名称（不传则复用原任务的导出配置） |
| `export_model_desc` | str | ❌ 可选 | - | 克隆时可覆盖的导出模型描述 |

> ⚠️ **注意事项：**
> - 如果同时提供 `model_id` 和 `model_name`，优先使用 `model_id`。
> - 如果不传 `model_id` 和 `model_name`，新任务将复用原任务的模型配置。
> - 克隆操作会创建一个全新的 `INIT` 状态任务，不会影响原任务。

### 调用示例

**复用原任务模型**（仅克隆配置）：

```bash
python3 scripts/connect_mcp.py call clone_hunyuan_training_model_convert_task '{"task_id": "finetuning_123456", "wsid": 456}'
```

**为新模型克隆**（覆盖模型配置）：

```bash
python3 scripts/connect_mcp.py call clone_hunyuan_training_model_convert_task '{
  "task_id": "finetuning_123456",
  "wsid": 456,
  "model_name": "hunyuan-13b",
  "export_model_name": "hunyuan-13b-hf",
  "export_model_desc": "HF 格式的 hunyuan-13b"
}'
```

### 返回字段说明（`data` 内部结构）

克隆接口返回一个 **完整的 `OpenModelConvertTaskInfo`**（与 `get_task_detail` 一致），下表仅列克隆后最关心的字段，完整字段参阅「get_hunyuan_training_model_convert_task_detail」章节。

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_id` | str | 新任务 ID（如 `finetuning_789012`） |
| `task_int_id` | long | 新任务数字 ID |
| `name` | str | 新任务名称 |
| `model_id` | int | 模型 ID |
| `model_name` | str | 模型名称 |
| `convert_type` | str | 转换类型（继承自原任务） |
| `hy_status` | str | 任务状态（克隆后为 `INIT`） |
| `hy_status_text` | str | 状态中文描述（"已创建"） |
| `wsid` | long | 工作空间 ID |

### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "task_id": "finetuning_789012",
    "task_int_id": 789012,
    "name": "mcore_ptm2_to_hf_hunyuan-13b_20260415170000",
    "model_id": 456,
    "model_name": "hunyuan-13b",
    "convert_type": "mcore_ptm2_to_hf",
    "hy_status": "INIT",
    "hy_status_text": "已创建",
    "wsid": 456
  }
}
```

> 💡 **提示**：克隆接口返回体中 **没有 `message` 业务字段**（外层已有 `code`+`message`）。克隆成功后，建议询问用户是否立即启动新任务，确认后调用 `start_hunyuan_training_model_convert_task` 启动。

---

## list_hunyuan_training_model_convert_output_models

**用途**：查询指定源模型（`model_id` 或 `model_name`）在指定工作空间下所有已产出的转换模型信息，返回转换后模型列表，方便查看历史产出。同一个产出模型如果被多次转换只保留最新记录，按更新时间倒序排列。

> ⚠️ **必须确认的信息（最高优先级）：**
> 1. **wsid**（工作空间 ID）：必填，不能为 0
> 2. **模型标识**：`model_id` 和 `model_name` 至少提供一个

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID，不能为 0 |
| `model_id` | int | ⭕ 与 `model_name` 二选一 | - | 源模型 ID（优先使用） |
| `model_name` | str | ⭕ 与 `model_id` 二选一 | - | 源模型名称（后端自动查询对应 model_id） |

> ⚠️ `model_id` 和 `model_name` 必须至少提供一个。如果同时提供，优先使用 `model_id`。

### 调用示例

**按模型名称查询**：

```bash
python3 scripts/connect_mcp.py call list_hunyuan_training_model_convert_output_models '{"wsid": 456, "model_name": "hunyuan-7b"}'
```

**按模型 ID 查询**：

```bash
python3 scripts/connect_mcp.py call list_hunyuan_training_model_convert_output_models '{"wsid": 456, "model_id": 123}'
```

### 返回字段说明（`data` 内部结构）

| 字段 | 类型 | 说明 |
|------|------|------|
| `src_model_id` | int | 源模型 ID |
| `src_model_name` | str | 源模型名称 |
| `output_models` | array | 产出模型列表 |
| `output_models[].model_id` | int | 产出模型 ID |
| `output_models[].model_name` | str | 产出模型名称 |
| `output_models[].model_desc` | str | 产出模型描述 |
| `output_models[].model_format` | str | 产出模型格式（如 `HF`、`Mcore`、`PTM+HF`、`DCP+HF`、`Mcore+HF` 等） |
| `output_models[].category` | str | 模型格式分类编码（`category` 字段值） |
| `output_models[].convert_type` | str | 转换类型 |
| `output_models[].export_format` | str | 导出格式 |
| `output_models[].location` | str | 地域英文缩写（如 `sh`、`nj`） |
| `output_models[].location_name` | str | 地域中文名称 |
| `output_models[].path` | str | 模型存储路径（主格式路径） |
| `output_models[].hf_model_path` | str | HF 模型路径（仅多格式导出时存在） |
| `output_models[].model_url` | str | 模型详情页链接 |
| `output_models[].convert_task_id` | str | 对应的转换任务 ID |
| `output_models[].wsid` | long | 工作空间 ID |

> ⚠️ 本接口 **不返回 `total` 字段**，产出数量直接以 `output_models.length` 为准。

### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "src_model_id": 123,
    "src_model_name": "hunyuan-7b",
    "output_models": [
      {
        "model_id": 456,
        "model_name": "hunyuan-7b-hf",
        "model_desc": "HF format output",
        "model_format": "HF",
        "category": "LLM",
        "convert_type": "mcore_ptm2_to_hf",
        "export_format": "HF",
        "location": "gz",
        "location_name": "广州",
        "path": "/data/models/hunyuan-7b-hf",
        "model_url": "https://taiji.woa.com/...",
        "convert_task_id": "finetuning_789",
        "wsid": 100
      }
    ]
  }
}
```

> 💡 **适用场景**：
> - 查看某个模型历史上转换产出了哪些格式的模型
> - 确认转换产出模型的存储路径和地域信息
> - 在创建新转换任务前，先检查是否已有相同格式的产出模型
> - 如果用户想查看对应的转换任务详情，可使用返回的 `convert_task_id` 调用 `get_hunyuan_training_model_convert_task_detail`

> ⚠️ **展示规则（强制）**：
> 展示产出模型列表时，**每个产出模型必须附上模型详情页链接**（`model_url` 字段），以 Markdown 超链接形式展示，例如：`[查看详情](https://taiji.woa.com/...)`。如果 `model_url` 为空，则不展示链接。

---

## query_hunyuan_training_hf_template_list

**用途**：分页查询 HF 模板列表，支持按关键词模糊搜索、按来源分类、按模型架构类型筛选、只看全局模板或只看当前用户创建的模板。返回 HF 模板列表（分页），每项包含模板 ID、名称、模型类型、HF 路径等摘要信息。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID |
| `keyword` | str | ❌ 可选 | - | 搜索关键词，用于模糊匹配模板名称/描述 |
| `page` | int | ❌ 可选 | `1` | **页码，从 1 开始，默认 1** |
| `page_size` | int | ❌ 可选 | `20` | 每页数量，默认 20 |
| `source` | str | ❌ 可选 | - | 来源分类过滤（如：官方 / 自定义 等） |
| `model_type` | str | ❌ 可选 | - | 按模型结构类型过滤（合法值可通过 `list_hunyuan_training_hf_model_types` 获取） |
| `is_global` | bool | ❌ 可选 | - | 是否只查询全局模板 |
| `only_mine` | bool | ❌ 可选 | - | 是否只查询当前用户创建的模板 |

### 调用示例

**首页 20 条**：

```bash
python3 scripts/connect_mcp.py call query_hunyuan_training_hf_template_list '{"wsid": 456, "page": 1, "page_size": 20}'
```

**只看我创建的、按关键词过滤**：

```bash
python3 scripts/connect_mcp.py call query_hunyuan_training_hf_template_list '{"wsid": 456, "page": 1, "keyword": "llama", "only_mine": true}'
```

**按模型结构类型过滤**：

```bash
python3 scripts/connect_mcp.py call query_hunyuan_training_hf_template_list '{"wsid": 456, "page": 1, "model_type": "llama"}'
```

### 返回字段说明（分页响应，`data` 内部结构）

> ⚠️ **本工具使用分页响应壳**（`StdOpenAPIPageResponse`），与其他 11 个工具的普通壳不同，`data` 下层多一层 `PageData`。

| 字段 | 类型 | 说明 |
|------|------|------|
| `items` | array | 模板列表（当前页数据） |
| `items[].id` | long | 模板 ID |
| `items[].name` | str | 模板名称 |
| `items[].model_type` | str | 模型结构分类（如 `llama`、`qwen2`、`hy3.0`） |
| `items[].description` | str | 模板描述 |
| `items[].hf_path` | str | HF 文件原始物理存储路径 |
| `items[].status` | str | 模板状态：`active`（正常可用） / `deprecated`（已废弃） / `deleted`（已删除） |
| `items[].file_list` | array | 文件列表（异步上传完成后更新），元素含 `name`（文件名） + `size`（字节数） |
| `items[].is_global` | bool | 是否为全平台公用官方模板 |
| `items[].creator` | str | 创建人 RTX |
| `items[].created_at` | str | 创建时间（ISO-8601 时间字符串） |
| `items[].editable` | bool | 当前用户是否有编辑权限 |
| `page` | int | 当前页码 |
| `page_size` | int | 当前每页大小 |
| `total` | long | 满足条件的总条数 |
| `has_more` | bool | 是否还有下一页（服务端自动计算：`page * page_size < total`） |

### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": 1001,
        "name": "llama-7b-hf",
        "model_type": "llama",
        "description": "Llama 7B HF 官方模板",
        "hf_path": "/data/hf_templates/llama-7b",
        "status": "active",
        "file_list": [
          {"name": "config.json", "size": 1024},
          {"name": "tokenizer.json", "size": 2048576}
        ],
        "is_global": true,
        "creator": "user_abc",
        "created_at": "2026-04-01T10:00:00.000+0000",
        "editable": false
      }
    ],
    "page": 1,
    "page_size": 20,
    "total": 42,
    "has_more": true
  }
}
```

> 💡 **提示**：
> - `page` 从 1 开始。旧接口 `page` 从 0 开始，本工具已升级为从 1 开始，请勿混淆。
> - **模板 ID 字段为 `id`（不是 `template_id`）**，后续调 `get_hunyuan_training_hf_template_detail` 时传入的入参叫 `template_id`，但实际就是本接口返回的 `id` 字段值。
> - `created_at` 字段为 `Date` 类型，序列化为 ISO-8601 时间字符串。

---

## get_hunyuan_training_hf_template_detail

**用途**：根据 `template_id` 查询指定 HF 模板的详情，返回模板名称、描述、模型类型、HF 路径、包含文件列表、创建人、时间等完整信息。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `template_id` | int | ✅ 必填 | - | HF 模板 ID |

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_training_hf_template_detail '{"template_id": 1001}'
```

### 返回字段说明（`data` 内部结构）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | long | 模板 ID |
| `name` | str | 模板名称 |
| `model_type` | str | 模型结构分类（如 `llama`、`qwen2`、`hy3.0`） |
| `description` | str | 模板描述 |
| `hf_path` | str | HF 文件原始物理存储路径 |
| `cos_path` | str | COS 存储路径（异步上传完成后更新） |
| `total_file_size` | long | 文件总大小（字节，异步上传完成后更新） |
| `file_list` | array | 文件列表（异步上传完成后更新） |
| `file_list[].name` | str | 文件名 |
| `file_list[].size` | long | 文件大小（字节） |
| `is_global` | bool | 是否为全平台公用官方模板 |
| `creator` | str | 创建人 RTX |
| `status` | str | 模板状态：`active` / `deprecated` / `deleted` |
| `parent_template_id` | long | 父模板 ID（另存为新模板时使用） |
| `editable` | bool | 当前用户是否有编辑权限 |

> ⚠️ **本接口不返回**：`created_time` / `updated_time` / `wsid`。如需归属信息，请从列表接口（`items[].created_at`、`items[].creator`）获取。

### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": 1001,
    "name": "llama-7b-hf",
    "model_type": "llama",
    "description": "Llama 7B HF 官方模板",
    "hf_path": "/data/hf_templates/llama-7b",
    "cos_path": "cos://taiji-hf-templates/llama-7b/",
    "total_file_size": 13476234880,
    "file_list": [
      {"name": "config.json", "size": 1024},
      {"name": "tokenizer.json", "size": 2048576},
      {"name": "generation_config.json", "size": 256}
    ],
    "is_global": true,
    "creator": "user_abc",
    "status": "active",
    "parent_template_id": null,
    "editable": false
  }
}
```

---

## register_hunyuan_training_hf_template

**用途**：注册（新增）一个 HF 模板。需提供模板名称、模型结构类型、HF 文件路径（该路径需包含 `config.json`、`tokenizer.json`、`generation_config.json` 等必要文件），可选描述与是否设为全局模板。返回新注册模板的 ID 与名称等信息。

> ⚠️ **调用前必做**：
> 1. **先调用 `list_hunyuan_training_hf_model_types` 获取合法的 `model_type` 枚举**，严禁猜测；如果传入了非法的 `model_type`，注册会失败。
> 2. **推荐先调用 `validate_hunyuan_training_hf_template_path` 预检 `hf_path` 下必要文件**，避免注册后才发现路径缺失文件。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID |
| `name` | str | ✅ 必填 | - | 模板名称 |
| `model_type` | str | ✅ 必填 | - | 模型结构类型。**取值必须通过 `list_hunyuan_training_hf_model_types` 获取，严禁猜测** |
| `hf_path` | str | ✅ 必填 | - | HF 文件路径，需包含所需的 HF 配置文件（如 `config.json`、`tokenizer.json`、`generation_config.json`） |
| `description` | str | ❌ 可选 | - | 模板描述 |
| `is_global` | bool | ❌ 可选 | `false` | 是否设为全局模板 |

### 推荐调用流程

```bash
# 步骤 1：查询合法的 model_type 枚举
python3 scripts/connect_mcp.py call list_hunyuan_training_hf_model_types '{}'

# 步骤 2：预检 HF 路径必要文件
python3 scripts/connect_mcp.py call validate_hunyuan_training_hf_template_path '{"hf_path": "/data/my_hf_templates/qwen2-7b"}'

# 步骤 3：注册模板（model_type 使用步骤 1 返回的合法值）
python3 scripts/connect_mcp.py call register_hunyuan_training_hf_template '{
  "wsid": 456,
  "name": "my-qwen2-7b",
  "model_type": "qwen2",
  "hf_path": "/data/my_hf_templates/qwen2-7b",
  "description": "自定义 Qwen2 7B 模板",
  "is_global": false
}'
```

### 返回字段说明（`data` 内部结构）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | long | 新注册模板的 ID |
| `name` | str | 模板名称 |
| `status` | str | 模板状态：`active`（正常可用） / `deprecated`（已废弃） / `deleted`（已删除） |

> ⚠️ **本接口不返回**：`model_type` / `hf_path` / `is_global` / `message`（内层）。如需确认注册后的完整信息，请参阅返回的 `id` 再调 `get_hunyuan_training_hf_template_detail` 获取完整详情。

### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": 2001,
    "name": "my-qwen2-7b",
    "status": "active"
  }
}
```

---

## validate_hunyuan_training_hf_template_path

**用途**：校验指定的 HF 文件路径下必要文件（如 `config.json`、`tokenizer.json`、`generation_config.json` 等）是否完整，用于在注册模板前进行预检。返回是否合法（`valid`）以及缺失文件列表等提示。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `hf_path` | str | ✅ 必填 | - | 待校验的 HF 文件路径 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call validate_hunyuan_training_hf_template_path '{"hf_path": "/data/my_hf_templates/qwen2-7b"}'
```

### 返回字段说明（`data` 内部结构）

| 字段 | 类型 | 说明 |
|------|------|------|
| `valid` | bool | 校验是否通过（必填文件是否完整） |
| `missing_files` | array | 缺失的必填文件列表，校验通过时为空列表 |
| `message` | str | 校验结果描述 |

> ⚠️ **本接口不返回**：`hf_path`（回显参数）、`existing_files`（已存在文件列表）。如需了解已存在的文件，请先注册模板后调 `get_hunyuan_training_hf_template_detail` 查看 `file_list`。

### 返回示例

**校验通过**：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "valid": true,
    "missing_files": [],
    "message": "HF 路径校验通过"
  }
}
```

**校验失败（缺少文件）**：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "valid": false,
    "missing_files": ["tokenizer.json", "generation_config.json"],
    "message": "HF 路径缺少必要文件，请补齐后再注册"
  }
}
```

> 💡 **建议**：注册模板前始终先调此工具做预检，避免注册接口报错。校验失败时，将 `missing_files` 展示给用户，让用户补齐后再注册。

---

## list_hunyuan_training_hf_model_types

**用途**：查询平台支持的 HF 模板模型结构类型枚举值列表，用于在注册 HF 模板 (`register_hunyuan_training_hf_template`) 时的 `model_type` 参数取值参考。**无需入参**。

### 参数

无入参。

### 调用示例

```bash
python3 scripts/connect_mcp.py call list_hunyuan_training_hf_model_types '{}'
```

### 返回字段说明（`data` 内部结构）

| 字段 | 类型 | 说明 |
|------|------|------|
| `types` | array&lt;str&gt; | 模型结构类型枚举值列表（字符串数组，元素直接就是 `model_type` 取值） |

> ⚠️ **本接口返回的是纯字符串数组**（后端 DTO 为 `List<String>`），不包含 `code` / `name` / `description` 等额外元信息。注册模板时，直接将数组元素（字符串）作为 `model_type` 参数传入即可。

### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "types": ["llama", "qwen2", "hy3.0", "mixtral", "deepseek_v3"]
  }
}
```

> 💡 **强制约定**：调用 `register_hunyuan_training_hf_template` 前必须先调本工具，从 `types` 数组中选取作为 `model_type`。**严禁**根据用户口述的名称直接猜测 `model_type` 值。如果无法将用户口述匹配到 `types` 中的某个元素，就把完整列表展示给用户让他确认。
