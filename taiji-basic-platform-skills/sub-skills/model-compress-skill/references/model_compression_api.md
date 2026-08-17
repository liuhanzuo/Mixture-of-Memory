## get_compress_strategy

查询特定模型支持的压缩策略。本工具是创建/复制压缩任务的**前置步骤**，调用后根据策略结果执行 `create_compress_task` 或 `clone_compress_task`。

> ⚠️ **Token 注入**：此工具不需要 `wsid`，但需要有效的 Auth Token。
> ⚠️ **必填参数**：`model_id` 与 `model_name` 至少传一个有效组合（`model_id=-1` 时按 `model_name` 匹配）。
> ⚠️ **不要在此工具后探索其他模块**：这是模型压缩的内部工具链，不要调用 `query_workspace_list` 或 storage/resource 相关工具。


#### 请求体

```json
{
  "model_id": -1,
  "model_name": "Qwen3-4B-Base"
}
```

#### 入参说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `model_id` | integer | ✅ | 模型 ID，`-1` 表示按 `model_name` 匹配 |
| `model_name` | string | ⚠️ (`model_id=-1` 时必填) | 模型名称 |

#### 返回字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 状态码，`0`=成功 |
| `message` | string | 状态消息 |
| `data` | array/object | 策略列表 或 错误信息对象 |

**正常返回时 `data` 结构**（array，按方法分类）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `[].key` | string | 方法类型 key（如 `quantify`=量化, `speculative_sampling`=投机采样） |
| `[].label` | string | 方法类型中文名 |
| `[].value` | array | 该方法下具体策略列表 |
| `[].value[].key` | string | 策略 key（即 `compress_strategy` 可传的值） |
| `[].value[].value` | string | 策略显示名（通常与 key 相同） |
| `[].value[].is_need_data` | boolean | 是否需要校准数据集（`true` 则创建任务时需传 `dataset_ids`） |

#### 返回示例

**正常返回（Qwen3-4B-Base）**：

```json
{
  "code": 0,
  "message": "success",
  "data": [
    {
      "key": "quantify",
      "label": "量化",
      "value": [
        {"key": "W8A8-FP8-static", "value": "W8A8-FP8-static", "is_need_data": true},
        {"key": "W8A8-FP8-dynamic", "value": "W8A8-FP8-dynamic", "is_need_data": false},
        {"key": "W8A8-INT8-PTPC", "value": "W8A8-INT8-PTPC", "is_need_data": false}
      ]
    },
    {
      "key": "speculative_sampling",
      "label": "投机采样",
      "value": [
        {"key": "Eagle3", "value": "Eagle3", "is_need_data": true}
      ]
    }
  ]
}
```

**异常返回（模型不存在）**：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "message": "mould not exist"
  }
}
```

> 💡 **注意**：即使模型不存在，`code` 仍为 `0`，需检查 `data.message` 是否包含错误信息。
> ⚠️ **`is_need_data=true` 的策略必须提供校准数据集**：创建任务时若选择需要数据的策略（如 `W8A8-FP8-static`），必须在 `data_config.trainData` 数组中传入有效的数据集条目（含 `id`/`fileName`/`filePath`/`isSelected=true`）；否则可以不传，走模板默认数据集。

---

## create_compress_task

创建压缩任务。

> ⚠️ **Token 注入**：需要有效的 Auth Token + 正确的 `wsid`。
> ⚠️ **最简形式**：仅需 `wsid` + `task_name` 即可成功创建（内置默认模板：Qwen3-4B-Base / W8A8-FP8-static / 完整资源配置）。


#### 设计原则

采用「真实成功请求作为默认模板」方案：接口内置一份经生产验证的完整压缩任务配置（含基座模型、压缩策略、run.sh 脚本、校准数据、1机8卡资源等所有默认值）。**未传字段一律沿用模板默认**，避免因字段缺失导致太极后端 500。

入参分四类：
1. **模型识别字段**（顶层扁平，覆盖 `basicConfigV2`）
2. **压缩策略字段**（顶层扁平）
3. **资源配置字段**（顶层扁平，覆盖 `resourceConfig`，含基础配置和高级配置）
4. **三大嵌套配置块**（`resource_config`/`data_config`/`compress_config`，深度合并覆盖对应模板块）

#### 请求体（最简）

```json
{
  "wsid": 10314,
  "task_name": "模型压缩11"
}
```

#### 请求体（模型识别）

```json
{
  "wsid": 10362,
  "task_name": "compress-deepseek-671b-w4a8-001",
  "model_id": 52001,
  "model_name": "DeepSeek-V3",
  "model_series_id": 8001,
  "model_scale": "671",
  "model_size": "671B",
  "model_privacy": "list_official",
  "model_struct": "moe",
  "manufacturer": "deepseek",
  "task_scene": "text",
  "compress_strategy": "W4A8-FP8"
}
```

#### 请求体（带资源配置）

```json
{
  "wsid": 10362,
  "task_name": "compress-with-resource",
  "model_name": "Qwen3-8B-Base",
  "compress_strategy": "W8A8-FP8-dynamic",
  "app_group": "TaiJi_HYAide_HYapp_EXTRA",
  "gpu_name": "H20",
  "image_name": "mirrors.tencent.com/taiji/cuda12.3-cudnn9-python3.10-torch2.3-init-llamafactory-compress:20060202",
  "ckpt_save_time": 2592000,
  "output_storage_dir": "/apdcephfs_zwfy/share_303786641",
  "location": "zw",
  "host_num": "1",
  "host_gpu_num": "8",
  "distributed_type": "deepspeed"
}
```

#### 请求体（带高级配置 + 环境变量）

```json
{
  "wsid": 10362,
  "task_name": "compress-with-advanced",
  "model_name": "Qwen3-8B-Base",
  "compress_strategy": "W8A8-FP8-dynamic",
  "enable_fault_tolerance": true,
  "task_queuing_priority": "P2",
  "keep_alive": true,
  "keep_alive_time": 259200,
  "auto_recover": false,
  "env_vars": {
    "compress_task": "1",
    "MANUFACTURER": "qwen",
    "MANUFACTURER_SERIES": "qwen",
    "HUNYUAN_HF_VERSION": "1",
    "max_seqlen": "4096"
  },
  "dynamic_scheduling_config": {
    "dynamic_scheduling_strategy": "speed_first",
    "dynamic_scheduling_resource_queue": {
      "private": {"enable": true},
      "public": {"enable": false}
    }
  }
}
```

#### 请求体（带数据配置）

```json
{
  "wsid": 10362,
  "task_name": "compress-with-data",
  "model_name": "Qwen3-8B-Base",
  "compress_strategy": "W8A8-FP8-static",
  "data_config": {
    "trainData": [
      {
        "id": 52573,
        "name": "compress_default_data",
        "fileName": "PTQ_data.json",
        "filePath": "/cfs_hunyuanaide/52573/PTQ_data.json",
        "isSelected": true,
        "ratio": 1
      }
    ],
    "shuffleJsonl": true,
    "defaultValidationSetting": true
  }
}
```

#### 请求体（带压缩配置：modelParams / alertConfig）

```json
{
  "wsid": 10362,
  "task_name": "compress-with-compress-cfg",
  "model_name": "Qwen3-8B-Base",
  "compress_strategy": "W8A8-FP8-dynamic",
  "compress_config": {
    "modelParams": {
      "position_embedding_type": "rotary_ntk",
      "ntk_alpha": "2500",
      "max_seqlen": "4096"
    },
    "alertConfig": {
      "alert_type": ["rtx"],
      "training_init_or_pending_alert_enable": false,
      "training_init_or_pending_alert_timeout": 5
    }
  }
}
```

#### 入参说明

##### 必填字段

| 字段 | 类型 | 说明 |
|---|---|---|
| `wsid` | integer | 工作空间 ID |
| `task_name` | string | 任务名称，**不能包含空格** |

##### 模型识别字段（可选，不传沿用模板默认 Qwen3-4B-Base）

| 字段 | 类型 | 说明 |
|---|---|---|
| `model_id` | integer | 要压缩的模型 ID，`-1` 表示按 `model_name` 匹配 |
| `model_name` | string | 模型名称（如 `Qwen3-4B-Base`/`Qwen3-8B-Base`/`DeepSeek-V3`） |
| `model_series_id` | integer | 模型系列 ID |
| `model_scale` | string | 模型规模档位（字符串，如 `"671"`/`"25"`/`"42"`） |
| `model_size` | string | 模型参数量展示值（如 `"4B"`/`"8B"`/`"671B"`） |
| `model_privacy` | string | 模型隐私类型（如 `list_official`） |
| `model_struct` | string | 模型结构（`dense`/`moe`） |
| `manufacturer` | string | 模型厂商（`hunyuan`/`deepseek`/`qwen` 等） |
| `task_scene` | string | 任务场景（`text`=文生文，`multimodal`=图生文） |

##### 压缩策略字段（可选，顶层）

| 字段 | 类型 | 说明 |
|---|---|---|
| `compress_strategy` | string | 压缩策略（建议先用 `get_compress_strategy` 查询目标模型支持的实际值，如 `W4A8-FP8`/`W8A8-FP8-static`/`W8A8-FP8-dynamic`/`Eagle3` 等） |
| `compress_strategy_type` | string | 压缩策略类型（如 `quantify`=量化，`speculative_sampling`=投机采样） |
| `data_type` | string | 数据类型（如 `CFS_DATA`） |

##### 资源配置：基础字段（可选，映射到 `resourceConfig.*`）

| 字段 | 底层键 | 类型 | 说明 |
|---|---|---|---|
| `app_group` | `business_flag` | string | **应用组**，如 `TaiJi_HYAide_HYapp_EXTRA` |
| `output_storage_dir` | `container_path` | string | **产出存储目录**，如 `/apdcephfs_zwfy/share_303786641` |
| `ckpt_save_time` | `ckptSaveTime` | integer | **模型保存时长（秒）**，如 `2592000`（30 天） |
| `gpu_name` | `gpu_name` | string | **GPU 卡型号**，如 `H20`/`A100`/`V100` |
| `image_name` | `image_name` | string | **运行环境镜像**，完整镜像地址 |
| `host_num` | `host_num` | string | 主机数（字符串），如 `"1"` |
| `host_gpu_num` | `host_gpu_num` | string | 每机 GPU 数（字符串），如 `"8"` |
| `location` | `location` | string | 地域（如 `sh`=上海，`zw`=中卫，`gz`=广州） |
| `command` | `command` | string | 启动命令（如 `bash`） |
| `args` | `args` | string | 启动参数 |
| `master_port` | `master_port` | string | 主端口（如 `"8005"`） |
| `distributed_type` | `distributed_type` | string | 分布式类型（如 `deepspeed`） |
| `quota_type` | `quota_type` | string | 配额类型（`private`/`public`） |
| `hy_resource_usage` | `hy_resource_usage` | string | 资源用途（如 `experiment`/`production`） |
| `common_resource_type` | `common_resource_type` | string | 通用资源类型 |
| `cpu_task` | `cpu_task` | boolean | 是否 CPU 任务 |
| `enable_rdma` | `enable_rdma` | boolean | 是否启用 RDMA |
| `rdma_in_same_module` | `rdma_in_same_module` | boolean | RDMA 同模块 |

##### 资源配置：高级配置项（可选，映射到 `resourceConfig.*`）

| 字段 | 底层键 | 类型 | 说明 |
|---|---|---|---|
| `auto_recover` | `auto_recover` | boolean | 自动恢复 |
| `recover_type` | `recover_type` | string | 恢复类型（如 `enable_fault_tolerance`） |
| `enable_fault_tolerance` | `enable_fault_tolerance` | boolean | 是否启用容错 |
| `keep_alive` | `keep_alive` | boolean | 是否保持存活 |
| `keep_alive_time` | `keep_alive_time` | integer | 存活时长（秒） |
| `elastic_task` | `elastic_task` | boolean | 是否弹性任务 |
| `elastic_waiting_timeout` | `elastic_waiting_timeout` | integer | 弹性等待超时（秒） |
| `debug_mode` | `debug_mode` | boolean | 调试模式 |
| `debug_alive_time` | `debug_alive_time` | integer | 调试存活时长（秒） |
| `task_queuing_priority` | `task_queuing_priority` | string | 排队优先级（如 `P0`/`P1`/`P2`） |
| `enable_report_metric` | `enable_report_metric` | boolean | 是否上报指标 |
| `enable_runlab_report_metric` | `enable_runlab_report_metric` | boolean | 是否 runlab 上报指标 |
| `swanlab_api_key` | `swanlab_api_key` | string | swanlab API key |
| `swanlab_project_name` | `swanlab_project_name` | string | swanlab 项目名 |
| `trajectory_project_name` | `trajectory_project_name` | string | trajectory 项目名 |
| `enable_kube_ray` | `enable_kube_ray` | boolean | 是否启用 kube-ray |
| `kube_ray_mode` | `kube_ray_mode` | string | kube-ray 模式 |
| `kube_ray_config` | `kube_ray_config` | object | kube-ray 配置 |
| `enable_mixing_offline` | `enable_mixing_offline` | boolean | 是否启用混部离线 |
| `hard_schedule_gpu_num` | `hard_schedule_gpu_num` | integer | 硬调度 GPU 数 |
| `extra_plat_business` | `extra_plat_business` | string | 额外平台业务 |
| `exec_start_in_all_mpi_pods` | `exec_start_in_all_mpi_pods` | boolean | 是否在所有 MPI Pod 中执行启动脚本 |
| `env_vars` | `env_vars_dict` | object | **环境变量字典**（value 必须为 string），如 `{"HUNYUAN_COMPRESSION_STRATEGY":"W8A8-FP8","max_seqlen":"4096"}` |
| `private_env_vars` | `private_env_vars_dict` | object | 私有环境变量字典 |
| `dynamic_scheduling_config` | `dynamic_scheduling_config` | object | 动态调度配置，含 `dynamic_scheduling_strategy` 和 `dynamic_scheduling_resource_queue` |
| `storage_quota_info` | `storage_quota_info` | object | 存储配额信息，含 `cluster_name`/`container_path`/`location`/`is_default` |

##### 三大嵌套配置块（可选，深度合并覆盖对应模板块）

| 字段 | 映射目标 | 类型 | 说明 |
|---|---|---|---|
| `resource_config` | 深度合并到 `resourceConfig` | object | 一次性覆盖多个高级配置项，或覆盖扁平字段未开放的资源字段。**优先级**：`resource_config` 高于同名扁平字段（后覆盖前） |
| `data_config` | 深度合并到 `dataConvertConfig` | object | 数据配置整块，含 `trainData` 数组、`validationData`/`shuffleJsonl`/`defaultValidationSetting` 等 |
| `compress_config` | 深度合并到 `trainConfig` | object | 压缩训练配置，含 `modelParams`（模型参数）/`alertConfig`（告警配置）/`runFile`（脚本文件） |

##### 关键注意事项

- **snake_case 契约**：对外全部使用 snake_case（如 `app_group`/`env_vars`），**不要**用底层驼峰键名（`business_flag`/`env_vars_dict`）
- **时长单位一律为秒**：用户口述"30 天"需换算为 `2592000`；"3 天" → `259200`；"1 小时" → `3600`
- **字符串型数字字段**：`host_num`/`host_gpu_num`/`model_scale`/`model_size` 必须传字符串（`"1"` 而非 `1`）
- **环境变量 dict 内的值全为字符串**：`env_vars.max_seqlen` = `"4096"` 而非 `4096`
- **告警配置位置**：`alertConfig` 必须放在 `compress_config.alertConfig`，**不要**放在 `resource_config` 或顶层
- 🔴 **存储目录 / 地域 / 应用组绑定关系**：当用户指定了 `output_storage_dir`（自定义产出路径），你必须**同时**显式传入匹配的 `app_group` 和 `location`，三者必须一致。ceph 路径的前缀决定了地域归属，地域决定了可用应用组；不传或传错 `app_group` 会导致"应用组和产出存储目录不对应"或"地域不一致"错误。**遇到这类错误时不要多次重试，直接向用户说明路径/地域/应用组三者需要匹配**。
- 🔴 **重试终止规则**：同一个 `create_compress_task` 调用如果连续两次因**同类后端错误**（地域不匹配 / app_group 不匹配 / 模型不可用）失败，**立即停止重试**，将错误摘要反馈给用户。不要尝试自发探索替代模型名、替代地域、替代 cluster 等，这些信息只能由用户提供。
- **多机场景**：`host_num` > 1 时，`env_vars_dict` 会自动传给所有 pod

#### 返回字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 状态码，`0`=成功 |
| `message` | string | 状态消息 |
| `data.id` | integer | 压缩任务**业务主键**（后续 `get_compress_task_detail` 用此 ID 查询） |
| `data.create_time` | string | 创建时间（ISO 格式） |
| `data.update_time` | string | 更新时间 |
| `data.modifier` | string | 最后修改人 |
| `data.status` | string/null | 当前状态（新建时为 `null`） |
| `data.jzStatus` | string/null | 基建状态 |
| `data.name` | string | 任务名称 |
| `data.desc` | string | 任务描述（固定为 `"模型压缩"`） |
| `data.task_id` | string | 平台内部任务 ID |
| `data.taiji_task_id` | string | 太极任务 ID（格式：`finetuning_{user}_{timestamp}_{hash}`） |
| `data.wsid` | integer | 工作空间 ID |
| `data.compress_strategy` | string | 实际使用的压缩策略（可能被后端补全后缀，如 `W8A8-FP8` → `W8A8-FP8-static`） |
| `data.app_group_id` | string/null | 应用组 ID |
| `data.model_name` | string/null | 目标模型名称 |
| `data.userGroup` | null | 用户组 |
| `data.adminGroup` | null/array | 管理员组 |
| `data.location` | string/null | 机房位置 |

#### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": 4426,
    "create_time": "2026-07-03 22:54:11",
    "update_time": "2026-07-03 22:54:11",
    "modifier": "fayizyuan",
    "status": null,
    "jzStatus": null,
    "name": "模型压缩11",
    "desc": "模型压缩",
    "task_id": "732189",
    "taiji_task_id": "finetuning_fayizyuan_20260703225408_3f29bf78",
    "wsid": 10314,
    "compress_strategy": "W8A8-FP8-static",
    "app_group_id": "TaiJi_HYAide_Offline_Inference",
    "model_name": "Qwen3-4B-Base",
    "userGroup": null,
    "adminGroup": null,
    "location": null
  }
}
```

> 💡 **注意**：
> - 创建成功后 `status` 为 `null`（尚未开始执行），任务进入排队等待资源调度。
> - `compress_strategy` 可能被后端补全：创建任务时传入 `W8A8-FP8`，返回时可能变为 `W8A8-FP8-static`（带 `-static` 后缀）。以返回值为准。

---

## clone_compress_task

复制压缩任务。

> ⚠️ **Token 注入**：需要有效的 Auth Token + 正确的 `wsid`。
> ⚠️ **最简形式**：仅需 `id` + `wsid` + `task_name` 即可成功复制（以源任务完整配置为模板，未传字段全部沿用源任务）。
> ⚠️ **鉴权机制**：调用太极创建任务时使用**源任务创建人**的身份进行鉴权，当前调用者无需拥有源任务所在太极工程的 WRITE 权限，只需是目标工作空间成员。
> 🔴 **停止规则**：如果 clone 因"模型在该地域无可用文件 / 地域资源不可用 / 源任务不可访问"等后端错误失败，**立即停止重试**，不要尝试修改 `location`、更换模型名、搜索替代模型等。将错误信息直接反馈给用户。


#### 请求体（最简 — 纯复制）

```json
{
  "id": 4417,
  "wsid": 10314,
  "task_name": "compress_copy_001"
}
```

#### 请求体（复制并替换模型 + 策略）

```json
{
  "id": 4417,
  "wsid": 10362,
  "task_name": "compress_qwen_w4a8",
  "model_id": -1,
  "model_name": "Qwen3-8B-Base",
  "manufacturer": "qwen",
  "task_scene": "text",
  "compress_strategy": "W4A8-AWQ"
}
```

#### 请求体（复制并替换数据集）

```json
{
  "id": 4417,
  "wsid": 10362,
  "task_name": "compress_new_data",
  "train_data": [
    {
      "fileName": "PTQ_data.json",
      "filePath": "/cfs_hunyuanaide/171393/PTQ_data.json",
      "isSelected": true,
      "ratio": 1.0
    }
  ]
}
```

#### 入参说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | integer | ✅ | 源压缩任务业务主键（即 `list_compress_tasks` 或 `create_compress_task` 返回的 `id`） |
| `wsid` | integer | ✅ | 工作空间 ID，新任务归属该工作空间 |
| `task_name` | string | ✅ | 新任务名称，**不能包含空格** |
| `model_id` | integer | ❌ | 替换模型 ID，不传沿用源任务 |
| `model_name` | string | ❌ | 替换模型名称（如 `Qwen3-8B-Base`），不传沿用源任务 |
| `model_series_id` | integer | ❌ | 替换模型系列 ID，不传沿用源任务 |
| `model_scale` | string | ❌ | 替换模型规模档位，不传沿用源任务 |
| `model_size` | string | ❌ | 替换模型参数量展示值，不传沿用源任务 |
| `model_privacy` | string | ❌ | 替换模型隐私类型，不传沿用源任务 |
| `model_struct` | string | ❌ | 替换模型结构，不传沿用源任务 |
| `manufacturer` | string | ❌ | 替换模型厂商，不传沿用源任务 |
| `task_scene` | string | ❌ | 替换任务场景，不传沿用源任务 |
| `compress_strategy` | string | ❌ | 替换压缩策略，不传沿用源任务。建议先用 `get_compress_strategy` 查询目标模型支持的策略 |
| `train_data` | array | ❌ | 替换校准数据集，不传沿用源任务。注意：文生文任务文件名必须为 `PTQ_data.json`，图生文任务文件名必须为 `ptq_data.tar` |

#### 返回字段说明

返回结构与 `create_compress_task` 一致：

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 状态码，`0`=成功 |
| `message` | string | 状态消息 |
| `data.id` | integer | 新压缩任务业务主键 |
| `data.name` | string | 任务名称 |
| `data.task_id` | string | 平台内部任务 ID |
| `data.taiji_task_id` | string | 太极任务 ID |
| `data.wsid` | integer | 工作空间 ID |
| `data.compress_strategy` | string | 压缩策略 |
| `data.model_name` | string | 目标模型名称 |
| `data.status` | string/null | 当前状态（新建时为 `null`） |
| `data.modifier` | string | 创建人（当前调用者，非源任务创建人） |
| `data.create_time` | string | 创建时间 |

#### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "id": 4460,
    "name": "compress_copy_001",
    "task_id": "820105",
    "taiji_task_id": "finetuning_jerrryliang_20260727160000_abc12345",
    "wsid": 10314,
    "compress_strategy": "W8A8-FP8-static",
    "model_name": "Qwen3-4B-Base",
    "status": null,
    "modifier": "binleilei",
    "create_time": "2026-07-27 16:00:00"
  }
}
```

> 💡 **注意**：
> - `modifier` 是当前调用者（谁调的 clone），不是源任务创建人。
> - `taiji_task_id` 中的用户名是源任务创建人（因为用源任务创建人身份调太极）。
> - 复制后的新任务与源任务完全独立，互不影响。
> - `train_data` 数组元素结构参考 `get_compress_task_detail` 返回的 `dataConvertConfig.trainData`。

---

## list_compress_tasks

批量查询压缩任务。

> ⚠️ **Token 注入**：需要有效的 Auth Token + 正确的 `wsid`。
> ⚠️ **分页参数**：`page` 从 `1` 开始，`page_size` 默认 `20`。


#### 请求体

```json
{
  "page": 1,
  "page_size": 5,
  "wsid": 10362,
  "creator": "ziguocheng"
}
```

#### 入参说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `wsid` | integer | ✅ | 工作空间 ID（用于隔离数据范围） |
| `page` | integer | ❌ | 页码，从 `1` 开始，默认 `1` |
| `page_size` | integer | ❌ | 每页数量，默认 `20` |
| `creator` | string | ❌ | 按创建人用户名过滤 |

#### 返回字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 状态码，`0`=成功 |
| `message` | string | 状态消息 |
| `data.items` | array | 任务列表（每项结构与 `create_compress_task` 的 `data` 一致） |
| `data.items[].id` | integer | 业务主键 |
| `data.items[].name` | string | 任务名称 |
| `data.items[].status` | string/null | 当前状态 |
| `data.items[].compress_strategy` | string | 压缩策略 |
| `data.items[].model_name` | string/null | 目标模型名 |
| `data.items[].create_time` | string | 创建时间 |
| `data.items[].task_id` | string | 平台内部任务 ID |
| `data.items[].taiji_task_id` | string | 太极任务 ID |
| `data.page` | integer | 当前页码 |
| `data.page_size` | integer | 每页数量 |
| `data.total` | integer | 总记录数 |
| `data.has_more` | boolean | 是否有更多数据 |

#### 返回示例

**正常返回（有数据）**：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": 4453,
        "create_time": "2026-07-05 11:22:38",
        "update_time": "2026-07-05 11:22:38",
        "modifier": "ziguocheng",
        "status": null,
        "jzStatus": null,
        "name": "compress-deepseek-671b-w4a8",
        "desc": "模型压缩",
        "task_id": "813609",
        "taiji_task_id": "finetuning_ziguocheng_20260705112235_6eb1a4a7",
        "wsid": 10362,
        "compress_strategy": "W4A8-FP8",
        "app_group_id": "TaiJi_HYAide_Offline_Inference",
        "model_name": "Bayberry-2B",
        "userGroup": null,
        "adminGroup": null,
        "location": null
      },
      {
        "id": 4428,
        "create_time": "2026-07-04 16:07:48",
        "name": "compress-deepseek-671b-w4a8-001",
        "compress_strategy": "W4A8-FP8",
        "model_name": "DeepSeek-V3",
        "...": "..."
      }
    ],
    "page": 1,
    "page_size": 5,
    "total": 4,
    "has_more": false
  }
}
```

**空结果返回（wsid 无权限或无任务）**：

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [],
    "page": 1,
    "page_size": 5,
    "total": 0,
    "has_more": false
  }
}
```

> 💡 **注意**：即使 `wsid` 错误或无权访问，接口仍返回 `code=0` + 空 items 列表（不会报错），需通过 `total==0` 判断是否有数据。
> - `creator` 过滤是可选的——不传则返回该 wsid 下所有用户的任务。

---

## get_compress_task_detail

查询压缩任务详情。如果用户意图是**复制/克隆此任务**，获取详情后应调用 `clone_compress_task`。

> ⚠️ **Token 注入**：需要有效的 Auth Token + 对应任务所属 `wsid`。
> ⚠️ **关键参数**：必须使用业务主键 `id` (integer)，不是 `task_id` (string)。


#### 请求体

```json
{
  "id": 4453,
  "wsid": 10362
}
```

#### 入参说明

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `id` | integer | ✅ | 压缩任务业务主键（即 `create` 返回的 `data.id`） |
| `wsid` | integer | ✅ | 工作空间 ID |

#### 返回字段说明（核心字段）

| 字段路径 | 类型 | 说明 |
|---|---|---|
| `code` | integer | 状态码，`0`=成功 |
| `message` | string | 状态消息 |
| `data.type` | string | 固定值 `"compress"` |
| `data.basicConfig` | null | 基础配置（当前为 null） |
| `data.status` | string/null | 任务当前状态（`null`=待运行） |
| `data.jzStatus` | string/null | 基建状态 |
| `data.hyStatus` | string/null | HY 状态 |
| `data.hyStatusText` | string/null | HY 状态文本 |
| `data.compress_strategy` | string | 压缩策略（如 `W4A8-FP8`） |

**trainConfig 子对象**：

| 字段路径 | 类型 | 说明 |
|---|---|---|
| `data.trainConfig.configPath` | string | 任务配置文件路径（CFS 路径） |
| `data.trainConfig.modelParams` | object | 模型参数（position_embedding_type / ntk_alpha / max_seqlen 等） |
| `data.trainConfig.runFile` | array | 运行脚本文件列表（含 name / path / content） |

**runFile 数组元素**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | 文件名（如 `run.sh`, `download.sh`, `download_sd.sh` 等） |
| `path` | string | 文件绝对路径（CFS 路径） |
| `content` | string | **文件完整内容**（脚本源码） |
| `purpose` | string | 用途描述 |
| `mtime` | string | 修改时间 |
| `isSelected` | boolean | 是否选中 |

**resourceConfig 子对象（资源配置）**：

| 字段路径 | 类型 | 说明 |
|---|---|---|
| `data.resourceConfig.host_num` | integer | 机器数（如 `1`） |
| `data.resourceConfig.host_gpu_num` | float | 每机 GPU 数（如 `8.0`） |
| `data.resourceConfig.gpu_name` | string | GPU 型号（如 `H20`） |
| `data.resourceConfig.image_name` | string | 镜像地址 |
| `data.resourceConfig.app_group_id` | string | 应用组 ID |
| `data.resourceConfig.location` | string | 机房位置（如 `sh`） |
| `data.resourceConfig.distributed_type` | string | 分布式类型（如 `deepspeed`） |
| `data.resourceConfig.env_vars_dict` | object | **环境变量字典**（含 `HUNYUAN_COMPRESSION_STRATEGY`/`MANUFACTURER`/`MANUFACTURER_SERIES` 等全部运行时变量） |
| `data.resourceConfig.storage_quota_info` | object | 存储配额信息（used/total/available） |

**basicConfigV2 子对象（任务元信息）**：

| 字段路径 | 类型 | 说明 |
|---|---|---|
| `data.basicConfigV2.id` | integer | 平台任务 ID |
| `data.basicConfigV2.taskID` | string | 太极任务 ID（与 `taiji_task_id` 一致） |
| `data.basicConfigV2.name` | string | 任务名称 |
| `data.basicConfigV2.description` | string | 任务描述 |
| `data.basicConfigV2.creator` | string | 创建人 |
| `data.basicConfigV2.modelName` | string | 模型名称 |
| `data.basicConfigV2.modelSize` | string | 模型规模（如 `2B`） |
| `data.basicConfigV2.manufacturer` | string | 厂商（`hunyuan`/`deepseek`/`qwen`） |
| `data.basicConfigV2.modelPrivacy` | string | 隐私类型 |
| `data.basicConfigV2.scale` | string | 规模档位 |
| `data.basicConfigV2.wsid` | integer | 工作空间 ID |
| `data.basicConfigV2.taskScene` | string | 任务场景 |
| `data.basicConfigV2.taskStage` | string | 阶段（固定 `SFT`） |
| `data.basicConfigV2.operatorTypes` | array | 允许的操作类型列表 |

**dataConvertConfig 子对象（数据配置）**：

| 字段路径 | 类型 | 说明 |
|---|---|---|
| `data.dataConvertConfig.trainData` | array | 训练数据集列表 |
| `data.dataConvertConfig.trainData[].id` | integer | 数据集 ID |
| `data.dataConvertConfig.trainData[].name` | string | 数据集名称 |
| `data.dataConvertConfig.trainData[].fileName` | string | 文件名（如 `PTQ_data.json`） |
| `data.dataConvertConfig.trainData[].filePath` | string | 文件 CFS 路径 |
| `data.dataConvertConfig.validationData` | array | 验证数据集列表（通常为空） |

#### 返回示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "type": "compress",
    "basicConfig": null,
    "trainConfig": {
      "configPath": "/cfs_hy_aide/common/tasks/finetuning_ziguocheng_20260705112235_6eb1a4a7",
      "modelParams": {
        "position_embedding_type": "rotary_ntk",
        "ntk_alpha": "2500",
        "max_seqlen": "4096"
      },
      "runFile": [
        {
          "name": "run.sh",
          "path": "/cfs_hy_aide/common/tasks/finetuning_ziguocheng_20260705112235_6eb1a4a7/run.sh",
          "content": "# Check interactive/non-interactive shell...\n# （完整 run.sh 内容，此处省略数百行）\n",
          "purpose": "",
          "desc": "",
          "type": "",
          "mtime": "2026-07-05 11:22:36",
          "isChanged": false,
          "isSelected": true
        }
      ]
    },
    "depFiles": [
      {
        "name": "download.sh",
        "path": "/cfs_hy_aide/common/tasks/.../download.sh",
        "content": "export no_proxy=...;\ncd ${RUNTIME_SCRIPT_DIR};\nwget ...",
        "purpose": "",
        "isSelected": true
      }
    ],
    "resourceConfig": {
      "host_num": 1,
      "host_gpu_num": 8.0,
      "gpu_name": "H20",
      "image_name": "mirrors.tencent.com/taiji/cuda12.3-cudnn9-python3.10-torch2.3-init-llamafactory-compress:20060202",
      "app_group_id": "TaiJi_HYAide_Offline_Inference",
      "location": "sh",
      "distributed_type": "deepspeed",
      "env_vars_dict": {
        "compress_task": "1",
        "HUNYUAN_COMPRESSION_STRATEGY": "W4A8-FP8",
        "HUNYUAN_MODEL_SCALE": "30",
        "HUNYUAN_MODEL_SCENE_TYPE": "text",
        "MANUFACTURER": "hunyuan",
        "MANUFACTURER_SERIES": "Bayberry",
        "CATEGORY_CHOICES": "2",
        "HUNYUAN_HF_VERSION": "1",
        "position_embedding_type": "rotary_ntk",
        "ntk_alpha": "2500",
        "max_seqlen": "4096"
      },
      "storage_quota_info": {
        "cluster_name": "jp_sh7_cephfs",
        "used_storage_gb": 16230,
        "total_storage_gb": 21000,
        "available_storage_gb": 4770
      }
    },
    "dataConvertConfig": {
      "trainData": [
        {
          "id": 52573,
          "name": "compress_default_data",
          "fileName": "PTQ_data.json",
          "filePath": "/cfs_hunyuanaide/52573/PTQ_data.json",
          "isSelected": true,
          "ratio": 1.0
        }
      ],
      "validationData": []
    },
    "basicConfigV2": {
      "id": 813609,
      "taskID": "finetuning_ziguocheng_20260705112235_6eb1a4a7",
      "name": "compress-deepseek-671b-w4a8",
      "description": "模型压缩",
      "creator": "ziguocheng",
      "modelName": "Bayberry-2B",
      "modelSize": "2B",
      "scale": "30",
      "manufacturer": "hunyuan",
      "modelPrivacy": "list_official",
      "wsid": 10362,
      "taskScene": "text",
      "taskStage": "SFT",
      "operatorTypes": ["copy", "kill", "evaluation", "detail", "start", "delete"]
    },
    "status": null,
    "jzStatus": null,
    "hyStatus": null,
    "hyStatusText": null,
    "compress_strategy": "W4A8-FP8"
  }
}
```

> 💡 **注意**：
> - `detail` 返回的内容非常丰富（含完整的 `run.sh` 脚本源码、环境变量、存储配额等），上例做了适当精简。
> - `runFile[].content` 包含完整的 shell 脚本（可能数千行），是理解压缩执行流程的关键。**展示时建议只展示关键字段摘要而非全文**，除非用户明确要求查看脚本内容。
> - `env_vars_dict` 中的 `MANUFACTURER` 决定了 `run.sh` 内部的分支逻辑（`hunyuan`/`deepseek`/`qwen` 各有独立流程）。
> - `operatorTypes` 列出了当前用户对该任务可执行的操作（`start`/`kill`/`evaluation`/`detail` 等）。
