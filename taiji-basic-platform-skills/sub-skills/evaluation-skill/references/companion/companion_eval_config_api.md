## list_taiji_eval_companion_tasks

按名称查已绑定触发事件的 ID。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task_id` | str | ✅ | 纯数字 jobGroupId，如 `"219261"` |
| `ws_id` | int | ❌ | 默认 10103，无需向用户确认 |

返回示例：

```json
{"triggers": [{"id": 893, "name": "lamictest2", "triggerType": "AUTO", "eventType": "SFT_TRANSFORM_EVAL"}], "total": 1}
```

```bash
python3 scripts/connect_mcp.py call list_taiji_eval_companion_tasks '{"task_id": "219261", "ws_id": 10103}'
```

---

## copy_taiji_eval_companion_trigger

复制已有触发事件创建新触发事件。

> ⛔ **仅支持同一训练任务内（同 job_group_id）的触发事件复制**。
> 新触发事件的 `job_group_id` 强制从源 trigger 继承，**不可覆盖**。
> 若需跨训练任务复制（源和目标 job_group_id 不同），请走 `copy_taiji_eval_companion_resource`（支持通过 `trigger_ids` 参数复制单个或全部触发器）。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `source_trigger_id` | int | ✅ | 上一步查出的 ID |
| `new_name` | str | ✅ | 同 jobGroup 下不可重名 |
| `ws_id` | int | 否 | 默认从源继承 |

可选覆盖（未传继承源值）：`trigger_type` / `event_type` / `strategy_config` / `evaluation_config` / `resource_config`。

`trigger_type` 必须大写：AUTO / MANUAL / DISABLED。
`event_type` 必须大写：EVAL / TRANSFORM_AND_EVAL / SFT / SFT_TRANSFORM_EVAL。

返回：`trigger_id` / `name` / `job_group_id`。

```bash
python3 scripts/connect_mcp.py call copy_taiji_eval_companion_trigger '{
  "source_trigger_id": 893, "new_name": "lamictest3", "ws_id": 10103
}'
```

`job_group_id` 和 `sft_task_config` 从源自动继承，不暴露给用户。⛔ 禁止传入 `job_group_id` 参数——该参数会被后端静默忽略（DTO 无此字段），导致触发事件被创建在源任务下而非目标任务下，接口返回成功但结果错位。

---

## upsert_taiji_eval_companion_config

幂等创建或更新评估配置，所有 model_config 字段有默认值。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `instance_id` | str | ✅ | 训练实例 hyJobId |
| `job_group_id` | str | ✅ | 数值型 jobGroupId |
| `ws_id` | int | 否 | 默认 10103 |
| `root_ceph_path` | str | 首次创建必填 | 已有配置时无需提供 |
| `model_config` | dict | 否 | 只传要改的字段 |

返回：`resource_id` / `action`（created / updated）。

```bash
# 全用默认值
python3 scripts/connect_mcp.py call upsert_taiji_eval_companion_config '{
  "instance_id": "8b0fba199e026eb4019e0643145d00f2", "job_group_id": "219261", "ws_id": 10103
}'

# 只改模型结构
python3 scripts/connect_mcp.py call upsert_taiji_eval_companion_config '{
  "instance_id": "8b0f...", "job_group_id": "219261",
  "model_config": {"model_structure": "dense"}
}'
```

### model_config 映射表（Agent 调用前必须完成转换）

| 用户说的 | 字段 | 传值 | 合法值 |
|---------|------|------|------|
| 预训练/pretrain | `model_stage` | `Pretrain` | Pretrain / Midtrain / SFT / RL |
| 中训/midtrain | `model_stage` | `Midtrain` |
| SFT/微调 | `model_stage` | `SFT` |
| RL/强化学习 | `model_stage` | `RL` |
| 稠密/dense | `model_structure` | `dense` | dense / moe / none |
| MoE/moe | `model_structure` | `moe` |
| 其他/none | `model_structure` | `none` |
| 文本/text | `scene_train` | `text` | 见完整枚举表 |
| 拷贝 | `enable_copy` | `true` | **必须为扁平 key**，不能嵌套在 `copy_strategy` 内（见下方示例） |

### 开启模型拷贝示例

```bash
# ⚠️ enable_copy / copy_target_path / copy_queue_location / copy_queue_name
#   必须作为 model_config 的顶层 key，不能嵌套在 copy_strategy 内
python3 scripts/connect_mcp.py call upsert_taiji_eval_companion_config '{
  "instance_id": "8b0f...",
  "job_group_id": "219261",
  "model_config": {
    "enable_copy": true,
    "copy_target_path": "/apdcephfs_gy6/share_303786641/.../ckpt",
    "copy_queue_location": "gy",
    "copy_queue_name": "TaiJi_HYAide_HYapp_EXTRA"
  }
}'
```

### model_config 完整映射表及合法值
| HF模板 | `hf_template_id` | 整数 | 如 381 / 387 |


### scene_train 合法值

`text` / `text-moe` / `text2sql` / `text-code` / `text-3D` / `image` / `image-dit-1.9` / `image-oteam3.6` / `video` / `music` / `multimodal` / `multimodal-MoE` / `video_to_text` / `audio` / `audio_to_text` / `audio_video_image_to_text` / `roleplay` / `translation` / `function-call` / `deepseek` / `open-source` / `embedding`

多模态场景（multimodal / multimodal-MoE / video_to_text / audio_video_image_to_text）须同时提供 `visual_structure` 和 `vit_input_resolution`。

**Agent 行为**：用户修改 scene_train 为上述多模态值时，**询问用户**"是否需要修改分辨率和视觉结构？如不需要，将使用默认值"。如用户未提供，**自动填入默认值**：

| 字段 | 默认值 |
|------|------|
| `vit_input_resolution` | `448*448` |
| `visual_structure` | `ViT1B_Resampler` |

用户如需自定义，值必须严格匹配合法枚举：

### visual_structure 合法值（区分大小写）

`ViT1B_Resampler` / `ViT2B_Resampler` / `ViT1B_MLP` / `ViT1B-TP_MLP` / `ViT1B-TP_Learnable` / `ViT1B-audio-MLP` / `ViT1B-Siglip-TP_Learnable`

### vit_input_resolution 合法值（区分大小写）

`224*224` / `336*336` / `448*448` / `896*896` / `896_2*896_2` / `1344*1344` / `1792*1792` / `6720*6720` / `Anyres`

---

## bind_taiji_eval_companion_config

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `job_group_id` | str | ✅ | 数值型 jobGroupId |
| `job_group_resource_id` | int | ✅ | 评估配置 ID |
| `trigger_ids` | list[int] | ✅ | 触发事件 ID 列表（全量替换语义：传入的列表即为最终绑定状态） |

```bash
python3 scripts/connect_mcp.py call bind_taiji_eval_companion_config '{
  "job_group_id": "219261", "job_group_resource_id": 212, "trigger_ids": [912]
}'
```

---

## list_taiji_eval_companion_configs

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `job_group_id` | str | ✅ | 数值型 jobGroupId |
| `ws_id` | int | ❌ | 默认 10103 |

```bash
python3 scripts/connect_mcp.py call list_taiji_eval_companion_configs '{"job_group_id": "219261", "ws_id": 10103}'
```

---

### 批量复制伴生配置到新任务

> 用户意图："把训练任务 A 的伴生配置复制到训练任务 B"、"复制训练任务时带上伴生配置"、
> "把任务 A 的 lamictest7 复制到任务 B"。
>
> 这里是**跨训练任务复制**（源和目标 job_group_id 不同），通过 `copy_taiji_eval_companion_resource` 完成。
> 支持两种粒度，由可选参数 `trigger_ids` 控制：
> - 不传 `trigger_ids`：复制源实例下**所有**触发事件（整实例批量复制）
> - 传 `trigger_ids: [904]`：只复制**指定**的触发事件（跨任务复制单个/部分触发器）

### 前提

- 目标训练任务必须**已有实例**（`target_hy_job_id` 由用户提供，Agent 不负责创建实例）
- `target_hy_job_id` 必须与 `source_hy_job_id` 不同，且尚未被其他伴生配置占用（否则后端报错）

### Step 1：查源任务有哪些伴生配置，确认源实例

```
call list_taiji_eval_companion_configs {job_group_id: "<源任务ID>", ws_id: 10103}
→ 返回 items: [{id, hy_job_id, root_ceph_path, model_config: {...}, ...}, ...]
→ 若为空：告知用户"当前任务无可复制的伴生配置"，流程终止
→ 若非空：展示实例列表让用户选择（默认最新的一条，即 items[0]，因为默认按 id 降序返回）
⛔ 即使只有一个实例，也必须展示让用户确认，严禁自动选定
```

### Step 2：查源实例下有哪些触发事件，确认复制范围

```
call list_taiji_eval_companion_tasks {task_id: "<源任务ID>", ws_id: 10103}
→ 返回 triggers: [{id, name, location, triggerType, eventType}, ...]
→ 若为空：告知用户"当前实例无可复制的触发事件"，流程终止
→ 若非空：展示触发事件列表，询问用户复制范围：
   - "复制全部触发事件"（默认）→ Step 3 不传 trigger_ids
   - "只复制部分触发事件" → 让用户选择，记录选中的 trigger id → Step 3 传 trigger_ids
⛔ 即使只有一个触发事件，也必须展示让用户确认，严禁自动选定
```

### Step 3：跨任务复制（一步完成）

```
# 场景A：复制源实例下所有触发事件（不传 trigger_ids）
call copy_taiji_eval_companion_resource {
  "source_job_group_id": "<源任务ID>",
  "source_hy_job_id": "<用户选定的源实例 hy_job_id>",
  "target_job_group_id": "<目标任务ID>",
  "target_hy_job_id": "<用户提供的目标实例 hy_job_id>"
}

# 场景B：只复制指定的触发事件（传 trigger_ids）
call copy_taiji_eval_companion_resource {
  "source_job_group_id": "<源任务ID>",
  "source_hy_job_id": "<用户选定的源实例 hy_job_id>",
  "target_job_group_id": "<目标任务ID>",
  "target_hy_job_id": "<用户提供的目标实例 hy_job_id>",
  "trigger_ids": [<用户选定的 trigger id 列表>]
}

→ 后端一步完成：
   1. 复制指定的触发事件（全部或 trigger_ids 列表中的）到目标任务
   2. 复制 resource（含 model_config）到目标任务
   3. 重建 trigger ↔ resource 绑定关系
   4. 若源开启了模型拷贝（enable_copy=true），拷贝路径自动替换实例 id（避免覆盖原模型）
→ 返回：
  {
    "new_resource_id": 30001,
    "copied_triggers": [{"old_id": 101, "new_id": 501, "name": "xxx-copy"}, ...],
    "copied_trigger_count": 2,
    "relation_count": 2,
    "skipped_relation_count": 0,
    "copy_target_path_rewritten": true
  }
```

**默认行为（无需追问用户）**：
- 模型拷贝状态默认继承源配置（`enable_copy` 不传即沿用源值）
- 拷贝目标路径默认自动派生（不传 `new_copy_target_path`，由后端替换实例 id 生成新路径）
- 复制范围默认全部（不传 `trigger_ids` 即复制源实例下所有触发事件）

**仅在用户主动提及时才传的可选字段**：
- `enable_copy`：用户要求"复制后关闭/开启模型拷贝"时才传
- `new_copy_target_path`：用户指定了具体拷贝路径时才传
- `trigger_ids`：用户明确表示"只复制部分触发事件"时才传（从 Step 2 的 triggers 列表中取 id）

### Step 4：回显

```
call list_taiji_eval_companion_configs {job_group_id: "<目标任务ID>", ws_id: 10103}
→ 展示目标任务下新增的伴生配置（含刚复制的 resource 和 triggers），确认复制成功
```

### ⛔ 严禁

- 严禁在 `target_hy_job_id` 未提供时代替用户假设或猜测实例 ID
- 严禁跳过 Step 1/Step 2 的展示确认环节直接调 Step 3
- 严禁用 `copy_taiji_eval_companion_trigger` 传 `job_group_id` 试图跨任务复制——该参数会被后端静默忽略，导致触发事件被创建在源任务下（接口返回成功但结果错位）。跨任务复制一律走 `copy_taiji_eval_companion_resource`。

## copy_taiji_eval_companion_resource

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `source_job_group_id` | int | ✅ | 源训练任务 ID |
| `source_hy_job_id` | str | ✅ | 源训练实例 ID |
| `target_job_group_id` | int | ✅ | 目标训练任务 ID |
| `target_hy_job_id` | str | ✅ | 目标训练实例 ID，必须与 source 不同 |
| `enable_copy` | bool | ❌ | 是否继承模型拷贝，不传则沿用源状态 |
| `new_copy_target_path` | str | ❌ | 自定义拷贝路径，不传则自动派生 |
| `trigger_ids` | list[int] | ❌ | 指定只复制哪些触发事件，不传=全部复制。id 从 `list_taiji_eval_companion_tasks` 的 `triggers[].id` 获取 |

```bash
# 复制源实例下所有触发事件
python3 scripts/connect_mcp.py call copy_taiji_eval_companion_resource '{
  "source_job_group_id": 219261,
  "source_hy_job_id": "8b0fb87f9d775d8e019d84fe68070214",
  "target_job_group_id": 220100,
  "target_hy_job_id": "955360119dfe48f3019e0158912a0099"
}'

# 只复制指定的触发事件（跨任务复制单个触发器）
python3 scripts/connect_mcp.py call copy_taiji_eval_companion_resource '{
  "source_job_group_id": 219261,
  "source_hy_job_id": "8b0fb87f9d775d8e019d84fe68070214",
  "target_job_group_id": 220100,
  "target_hy_job_id": "955360119dfe48f3019e0158912a0099",
  "trigger_ids": [904]
}'
```

---

## list_taiji_eval_companion_trigger_templates

列出伴生评估触发模板库，支持按触发类型、关键词、标签筛选。

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `trigger_type` | string | ❌ | 触发类型（枚举值） |
| `keyword` | string | ❌ | 关键词，支持名称模糊匹配 |
| `tags` | string | ❌ | 标签筛选 |
| `only_mine` | boolean | ❌ | 是否仅返回当前用户的模板 |
| `page_index` | integer | ❌ | 页码，从 1 开始，默认 1 |
| `page_size` | integer | ❌ | 每页数量，默认 10 |
| `order_by` | string | ❌ | 排序字段，默认按 id 降序 |

**返回：** 模板列表，含 `dataset_name` / `trigger_type` / `event_type` / `strategy_config` / `sft_task_config` 等。

```bash
python3 scripts/connect_mcp.py call list_taiji_eval_companion_trigger_templates '{"page_index": 1, "page_size": 10}'
```

> ⚠️ 本工具返回的是**模板库**，不是已绑定的触发事件。查已绑定触发事件用 `list_taiji_eval_companion_tasks`。

