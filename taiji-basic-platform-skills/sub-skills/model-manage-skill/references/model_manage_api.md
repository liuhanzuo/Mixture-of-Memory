## search_hunyuan_models_cards

> 🛑 **wsid 漏传自查**：每次调用前必须确认已传 `wsid`。用户 query 上下文、页面 URL 中的 `wsId=` 参数、或历史调用中均已提供 wsid，**直接取用**，不要漏掉导致无结果后重试。

**用途**：搜索太极平台上的模型，支持模糊匹配搜索、分页查询和多种过滤条件。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `wsid` | integer | ✅ | 无 | 工作空间 ID，**强制必填，不能为 0 或空值**。如果用户未提供，必须追问："请提供您的工作空间 ID (wsid)" |
| `keyword` | string | ✅ | 无 | 搜索关键词，支持模糊匹配 |
| `page` | integer | ❌ | 1 | 页码，从1开始 |
| `page_size` | integer | ❌ | 12 | 每页数量，范围为1-100 |
| `is_my_model` | boolean | ❌ | false | 是否显示我使用的模型 |
| `is_my_create` | boolean | ❌ | false | 是否显示我创建的模型 |
| `is_show_all` | boolean | ❌ | true | 是否显示所有模型 |

### 使用示例

```bash
# 搜索包含"文本生成"的模型
python3 scripts/connect_mcp.py call search_hunyuan_models_cards '{
  "wsid": 10103,
  "keyword": "文本生成",
  "page": 1,
  "page_size": 10,
  "is_my_model": true
}'

# 查看我使用的模型
python3 scripts/connect_mcp.py call search_hunyuan_models_cards '{
  "wsid": 10103,
  "keyword": "",
  "is_my_model": true,
  "is_show_all": false
}'

# 搜索我创建的模型
python3 scripts/connect_mcp.py call search_hunyuan_models_cards '{
  "wsid": 10103,
  "keyword": "",
  "is_my_create": true
}'
```

### 返回值格式（JSON）

```json
{
  "items": [
    {
      "model_id": 12345,
      "model_name": "hunyuan_sft_v1",
      "desc": "用于文本生成任务的模型",
      "model_stage": "SFT",
      "context_len": "128k",
      "creator": "username",
      "admins": ["admin1", "admin2"],
      "model_tag": ["文本生成", "自然语言处理"],
      "updated_at": "2026-03-27 10:46:04",
      "model_structure": "dense",
      "manufacturer": "hunyuan",
      "manufacturer_series": "hunyuan"
    }
  ],
  "page": 1,
  "page_size": 20,
  "total": 15,
  "has_more": false
}
```

### 返回值字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `items` | List | 模型卡片列表 |
| `items[].model_id` | Integer | 模型卡片 ID |
| `items[].model_name` | String | 模型名称 |
| `items[].desc` | String | 模型描述 |
| `items[].model_stage` | String | 模型阶段（如 `Pretrain` / `SFT` / `DPO`） |
| `items[].context_len` | String | 上下文长度（如 `32k` / `128k`） |
| `items[].creator` | String | 创建人 RTX |
| `items[].admins` | List[String] | 管理员 RTX 列表 |
| `items[].model_tag` | List[String] | 模型标签列表 |
| `items[].updated_at` | String | 更新时间 |
| `items[].model_structure` | String | 模型结构（如 `moe` / `dense`） |
| `items[].manufacturer` | String | 厂商（如 `tencent` / `hunyuan`） |
| `items[].manufacturer_series` | String | 厂商系列（如 `hunyuan` / `hunyuan3.0-Chat`） |
| `page` | Integer | 当前页码 |
| `page_size` | Integer | 每页数量 |
| `total` | Long | 命中总数 |
| `has_more` | Boolean | 是否还有更多数据 |

---

## get_hunyuan_models_card_detail

**用途**：根据模型ID获取太极平台上的模型详细信息，包括模型的基本信息、配置参数、使用说明等。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `wsid` | integer | ✅ | 无 | 工作空间 ID，**强制必填，不能为 0 或空值**。如果用户未提供，必须追问："请提供您的工作空间 ID (wsid)" |
| `model_id` | integer | ✅ | 无 | 模型ID，用于查询模型详细信息 |

### 使用示例

```bash
# 查询模型ID为169755的详细信息
python3 scripts/connect_mcp.py call get_hunyuan_models_card_detail '{
  "wsid": 10103,
  "model_id": 169755
}'
```

### 返回值格式（JSON）

```json
{
  "model_info": {
    "model_id": 169755,
    "model_name": "hunyuan_sft_v1",
    "desc": "模型描述信息",
    "model_structure": "dense",
    "activate_parameters": "7B",
    "total_parameters": "7B",
    "context_len": "4096",
    "model_stage": "SFT",
    "manufacturer": "hunyuan",
    "manufacturer_series": "hunyuan",
    "relate_base_mould": "Hunyuan-7B",
    "scene_type": "text",
    "admins": ["user123", "user456"],
    "admin_members": {
      "users": ["user123", "user456"],
      "user_groups": [8]
    },
    "common_members": {
      "users": ["kevincen"],
      "user_groups": []
    },
    "spaces": "10314,10103",
    "usable_space": ["10314", "10103"]
  },
  "train_info": {
    "task_id": "basic_train_xxx_20260101_xxx",
    "task_name": "模型训练任务",
    "train_framework": "Angel-PTM",
    "model_stage": "SFT",
    "train_platform": "taiji",
    "creator": "user123",
    "created_at": "2026-01-15 10:30:00",
    "task_url": "https://taiji.woa.com/..."
  },
  "location_infos": [
    {
      "id": 176805,
      "location": "sz",
      "ch_location": "深圳",
      "path": "/apdcephfs_sz/share_xxx/hunyuan/model/sft_v1",
      "hf_model_path": "/apdcephfs_sz/share_xxx/hunyuan/model/sft_v1/hf",
      "compression_path": null,
      "queue_name": "TaiJi_HYAide_MODEL_SZ",
      "copy_state": "success",
      "creator": "user123",
      "created_at": "2026-01-15 10:30:00",
      "is_blank_path": false
    }
  ]
}
```

### 返回值字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_info` | Object | 模型基本信息 |
| `model_info.model_id` | Integer | 模型 ID |
| `model_info.model_name` | String | 模型名称 |
| `model_info.desc` | String | 模型描述 |
| `model_info.model_structure` | String | 模型结构（如 `moe` / `dense`）|
| `model_info.activate_parameters` | String | 激活参数量 |
| `model_info.total_parameters` | String | 总参数量 |
| `model_info.context_len` | String | 上下文长度 |
| `model_info.model_stage` | String | 模型阶段（如 `Pretrain` / `SFT` / `DPO`）|
| `model_info.manufacturer` | String | 厂商 |
| `model_info.manufacturer_series` | String | 厂商系列 |
| `model_info.relate_base_mould` | String | 关联基座模型 |
| `model_info.scene_type` | String | 场景类型 |
| `model_info.admins` | List[String] | 管理员 RTX 列表 |
| `model_info.admin_members` | Object | 管理员详情（users / user_groups）|
| `model_info.common_members` | Object | 普通成员详情（users / user_groups）|
| `model_info.spaces` | String | 可展示的空间 ID（逗号分隔字符串）|
| `model_info.usable_space` | List[String] | 可使用的空间 ID 列表 |
| `train_info` | Object | 训练信息 |
| `train_info.task_id` | String | 训练任务 ID |
| `train_info.task_name` | String | 训练任务名称 |
| `train_info.train_framework` | String | 训练框架 |
| `train_info.model_stage` | String | 模型阶段 |
| `train_info.train_platform` | String | 训练平台 |
| `train_info.creator` | String | 创建人 |
| `train_info.created_at` | String | 创建时间 |
| `train_info.task_url` | String | 训练任务链接 |
| `location_infos` | List | 地域分布信息列表 |
| `location_infos[].id` | Integer | 地域记录 ID（MouldLocation 主键，作为预热缓存接口 `regions` 参数）|
| `location_infos[].location` | String | 地域英文缩写（如 `sz` / `sh`）|
| `location_infos[].ch_location` | String | 地域中文名称 |
| `location_infos[].path` | String | 模型路径 |
| `location_infos[].hf_model_path` | String | HF 格式模型路径（可为 null）|
| `location_infos[].compression_path` | String | 压缩模型路径（可为 null）|
| `location_infos[].queue_name` | String | 应用组名称 |
| `location_infos[].copy_state` | String | 拷贝状态（如 `success` / `failed` / `copying`）|
| `location_infos[].creator` | String | 创建人 |
| `location_infos[].created_at` | String | 创建时间 |
| `location_infos[].is_blank_path` | Boolean | 该地域的 path 是否为空 |

---

## list_hunyuan_models_platform_enums

**用途**：查询太极平台的各类枚举值，包括模型场景类型、参数规模、地域映射、厂商系列等基础数据。此工具是其他模型管理工具的基础。当需要了解平台支持的模型类型、参数规模、地域信息、厂商系列等信息时使用此工具。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `enum_types` | list[string] | ❌ | 无 | 需要查询的枚举类型列表。不传或传空列表则返回所有枚举类型 |

### 可用的枚举类型

| 枚举类型 | 说明 |
|----------|------|
| `scene_train_type` | 模型场景训练类型（文生文、文生图、图生文等），包含子场景类型 |
| `model_struct_type` | 模型结构类型（Dense、MoE等） |
| `SCALE_CHOICES_NEW` | 模型参数规模选项（如 7B、13B、70B 等） |
| `SCENE_TEXT_TYPE_CHOICES` | 场景文本类型选项 |
| `CONTEXT_LEN_LIST` | 上下文长度选项（如 4k、8k、32k、128k 等） |
| `MODEL_TYPE` | 模型架构类型（如 llama2、Aries-T/2 等） |
| `MANUFACTURER_CHOICES` | 模型厂商选项（如混元） |
| `LOCATION_MAP` | 地域英文缩写到中文名称的映射（如 sz→深圳、bj→北京） |
| `LOCATION_MAP_CH` | 地域中文名称到英文缩写的映射 |
| `MANUFACTURER_SERIES_CHOICES` | 厂商模型系列选项（包含各系列版本信息） |
| `THOUGHT_TYPE_CHOICES` | 思考模式选项（快思考、慢思考、全思考） |
| `MODEL_STAGE_CHOICES` | 模型阶段选项（一阶段、退火预训练、SFT、GRPO、DPO 等） |
| `model_operation_tags` | 模型运营标签（效果出众、更低延迟、高性价比等） |
| `train_method_enums` | 训练方法枚举（FULL、LoRA、RLHF-DPO、RLHF-PPO、RLHF-RM） |
| `open_source_type` | 开源类型（不开源、司内开源、司外开源） |
| `mould_structure_state` | 模型卡片状态（信息审核中、待发布、已发布、已下架等） |
| `SCENE_TYPE_CHOICES` | 场景类型选项 |
| `CATEGORY_CHOICES` | 模型类别选项（原始模型PTM、转换模型FT、开源模型HF 等） |
| `ENUM_CHOICES` | 模型归属类型（个人模型、公共模型） |
| `STAGE_CHOICES` | 模型训练阶段（Pretrain、Posttrain、Instruct、SFT） |
| `CONVERT_STATE_CHOICES` | 格式转换状态（转换中、转换失败、转换成功） |
| `visual_structure` | 视觉结构选项（ViT1B_Resampler、ViT2B_Resampler 等） |
| `resolution_ratio` | 分辨率选项（224*224、336*336、448*448 等） |
| `compression_strategy_list` | 压缩策略选项（W8A8-FP8、W4A8-AWQ 等） |
| `trans_scene_type` | 场景类型转换映射 |
| `NAS_SERVER_CHOICES` | NAS 存储服务器选项 |
| `MANUFACTURER_SERIES_TYPE` | 厂商系列类型 |
| `LOCATION_PY_MAP` | 地域缩写到拼音的映射 |
| `home_banner` | 首页轮播图和卡片信息 |

### 使用示例

```bash
# 查询所有枚举值
python3 scripts/connect_mcp.py call list_hunyuan_models_platform_enums '{}'

# 查询模型参数规模有哪些选项
python3 scripts/connect_mcp.py call list_hunyuan_models_platform_enums '{
  "enum_types": ["SCALE_CHOICES_NEW"]
}'

# 查询多个枚举类型
python3 scripts/connect_mcp.py call list_hunyuan_models_platform_enums '{
  "enum_types": ["MANUFACTURER_CHOICES", "MANUFACTURER_SERIES_CHOICES", "model_struct_type", "SCALE_CHOICES_NEW", "CONTEXT_LEN_LIST"]
}'

# 查询地域信息
python3 scripts/connect_mcp.py call list_hunyuan_models_platform_enums '{
  "enum_types": ["LOCATION_MAP"]
}'

# 查询思考模式和模型阶段
python3 scripts/connect_mcp.py call list_hunyuan_models_platform_enums '{
  "enum_types": ["THOUGHT_TYPE_CHOICES", "MODEL_STAGE_CHOICES"]
}'
```

### 返回值格式（查询指定枚举类型）

```json
{
  "SCALE_CHOICES_NEW": [
    {"value": "7B", "label": "7B"},
    {"value": "13B", "label": "13B"},
    {"value": "70B", "label": "70B"}
  ]
}
```

### 返回值格式（部分枚举类型未找到）

```json
{
  "data": {"SCALE_CHOICES_NEW": [...]},
  "not_found": ["INVALID_TYPE"],
  "available_enum_types": ["scene_train_type", "SCALE_CHOICES_NEW", ...],
  "hint": "以下枚举类型未找到: INVALID_TYPE，请参考 available_enum_types 中的可用类型"
}
```

### 交互规则

1. 用户说"查看平台支持哪些参数规模" → 传 `enum_types=["SCALE_CHOICES_NEW"]`
2. 用户说"查看所有枚举值" → 不传 `enum_types`（返回全部）
3. 用户在发布自定义训练模型时不确定参数可选值 → 建议先调用本工具查询对应枚举
4. 常用组合查询：发布模型时建议查询 `["MANUFACTURER_CHOICES", "MANUFACTURER_SERIES_CHOICES", "model_struct_type", "SCALE_CHOICES_NEW", "CONTEXT_LEN_LIST", "scene_train_type", "SCENE_TEXT_TYPE_CHOICES"]`
5. 操作成功后引导用户搜索模型或查看模型详情

---

## search_hunyuan_official_models_by_train

**用途**：按「训练类型 / 训练方式 / 训练框架」反查太极平台**支持训练**的**官方模型**。返回的列表与太极平台「微调训练 → 选择基础模型（官方模型）」页面在筛选条件下看到的模型集合在业务语义上一致（含 `sort_list_series` 白名单、wsid 可见范围、`official_model_top_list` 置顶顺序）。

> ⚠️ **wsid 校验是本工具的硬性约束（区别于本 Skill 中的常规规则）**：本工具是**少有的需要在调用前对 wsid 做权限校验**的工具，原因是 `wsid` 直接决定七彩石 `model_show_in_wsids` 的可见范围，错误的 wsid 会让用户看到错误的模型集合。具体校验流程见下方「交互规则」第 4 节。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `wsid` | integer | ✅ | 无 | 工作空间 ID，**强制必填，必须为正整数，不能为 0 或空值**。如果用户未提供，必须先按 `references/helper_api.md`（`list_user_workspaces`）列出候选空间让用户选择；如果用户已提供，仍需按该 helper 校验该 wsid 在用户有权限的列表中后再调用本工具 |
| `train_type` | string | ❌ | null | 训练类型，可选值：`SFT` / `DPO` / `Pretrain`。为空则不参与过滤 |
| `train_method` | string | ❌ | null | 训练方式，可选值：`LORA` / `FullParameter`。为空则不参与过滤 |
| `train_framework` | string | ❌ | null | 训练框架，可选值：`Angel-PTM` / `Megatron`。为空则不参与过滤 |

> 📌 **参数为扁平结构**：`wsid`、`train_type`、`train_method`、`train_framework` 均为顶层字段，不要嵌套到 `params` 对象里。

### 使用示例

```bash
# 查询所有支持训练的官方模型（不加任何条件，返回全量）
python3 scripts/connect_mcp.py call search_hunyuan_official_models_by_train '{
  "wsid": 10103
}'

# 查询哪些官方模型支持 SFT 训练
python3 scripts/connect_mcp.py call search_hunyuan_official_models_by_train '{
  "wsid": 10103,
  "train_type": "SFT"
}'

# 查询哪些官方模型支持 SFT + LORA
python3 scripts/connect_mcp.py call search_hunyuan_official_models_by_train '{
  "wsid": 10103,
  "train_type": "SFT",
  "train_method": "LORA"
}'

# 查询 Angel-PTM 框架下可训练的官方模型
python3 scripts/connect_mcp.py call search_hunyuan_official_models_by_train '{
  "wsid": 10103,
  "train_framework": "Angel-PTM"
}'

# DPO 类型查询
python3 scripts/connect_mcp.py call search_hunyuan_official_models_by_train '{
  "wsid": 10103,
  "train_type": "DPO"
}'
```

### 返回值格式（JSON）

```json
{
  "total": 3,
  "items": [
    {
      "model_id": 12345,
      "model_name": "Hunyuan-7B",
      "desc": "混元 7B 通用对话模型...",
      "train_type": "SFT",
      "train_method": "LORA",
      "train_framework": "Angel-PTM",
      "model_structure": "dense",
      "manufacturer": "hunyuan",
      "manufacturer_series": "hunyuan",
      "context_len": "32k",
      "total_parameters": "7B",
      "activate_parameters": "7B",
      "created_at": "2026-05-20 10:30:00"
    }
  ]
}
```

### 返回值字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `total` | Integer | 命中的官方模型总数 |
| `items` | List | 官方模型列表 |
| `items[].model_id` | Integer | 模型卡片 ID |
| `items[].model_name` | String | 模型名称（系列名） |
| `items[].desc` | String | 模型描述 |
| `items[].train_type` | String | 训练类型（如 `SFT` / `DPO` / `Pretrain`） |
| `items[].train_method` | String | 训练方式（如 `LORA` / `FullParameter`） |
| `items[].train_framework` | String | 训练框架（如 `Angel-PTM` / `Megatron`） |
| `items[].model_structure` | String | 模型结构（如 `moe` / `dense`） |
| `items[].manufacturer` | String | 供应商（如 `tencent` / `hunyuan`） |
| `items[].manufacturer_series` | String | 供应商子系列（如 `hunyuan` / `hunyuan3.0-Chat`） |
| `items[].context_len` | String | 上下文长度（如 `32k` / `128k`） |
| `items[].total_parameters` | String | 总参数量（如 `7B` / `70B`） |
| `items[].activate_parameters` | String | 激活参数量（取该 series 下最新 mould 的值） |
| `items[].created_at` | String | 创建时间 |

### 与其他模型查询工具的区别

| 维度 | `search_hunyuan_models_cards` | `search_hunyuan_official_models_by_train`（本工具） |
|------|----------------------|----------------------------------------------------|
| 数据范围 | 平台**全部**模型卡片（含个人模型、公共模型、官方模型） | 仅**官方模型**（`sort_list_series` 白名单内 + 当前 wsid 可见范围） |
| 检索维度 | 名称模糊匹配 | 训练类型 / 训练方式 / 训练框架 |
| 适用场景 | "找包含 xxx 关键词的模型" | "哪些官方模型支持 xxx 训练" |
| 排序规则 | 按更新时间 | `official_model_top_list` 置顶 + createTime DESC |

### 交互规则

**用户意图识别**（典型表述）：
- "哪些官方模型可以做精调 / 微调"、"哪些模型支持 SFT / DPO / Pretrain"、"哪些模型支持 LORA / 全参"、"Angel-PTM / Megatron 框架下有哪些可训练的官方模型"、"我想用 SFT-LORA 精调，可以选哪些基础模型" → 命中本工具，**而不是** `search_hunyuan_models_cards`。

**wsid 使用规则**：
1. **用户已提供 wsid，或评测/上下文已注入 `wsid 为 xxx`** → 直接把该 wsid 传给本工具，不要再调用 `list_user_workspaces` 做校验，避免重复工具链。
2. **用户未提供 wsid** → 才按 `references/helper_api.md`（`list_user_workspaces`）拿到该用户有权限的工作空间列表，展示给用户选择；用户选定后再调用本工具。
3. **严禁**直接以 `wsid=0`、空值或任意推测值调用本工具。

**训练条件参数选填策略**：
- 用户**只说类型**（如"哪些模型支持 SFT"）→ 仅传 `train_type`
- 用户**说了组合**（如"SFT + LORA"）→ 同时传 `train_type` + `train_method`
- 用户**只说框架**（如"Angel-PTM"）→ 仅传 `train_framework`
- 用户**未明确条件**（如"哪些官方模型可以训练"）→ 三项条件都不传，返回全部官方模型
- ⚠️ 用户给的取值**大小写无关**（服务端做了 `UPPER()` 归一），但建议保留用户原始大小写传入；空白字符串会被视为未提供
- 不要为了确认 SFT/LORA/DPO/Angel-PTM 是否是枚举值而调用 `list_hunyuan_models_platform_enums`；本工具会按条件查询，结果为空再提示无匹配即可。

**结果展示规范**：
- 顶部展示：当前 wsid、命中的训练条件、总数
- 表格列：序号 / 模型名称 / 模型阶段 / 上下文长度 / 激活参数量 / 描述
- 描述过长时截断到 60 字符并以 `…` 收尾
- `items` 为空时，明确提示"未找到匹配的官方模型"，并建议用户检查训练条件取值或确认 wsid 是否符合预期

**Guard/模板候选特例**：
- 用户说有 `Llama-Guard` / `Qwen3Guard` / Guard 模型，想看是否有模板可复用：先 `search_hunyuan_models_cards(keyword="Guard")` 查已有模型卡片，再 `search_hunyuan_official_models_by_train(wsid=<wsid>)` 查全部官方可训练模型；只做这两次查询并汇总候选。
- 不要分别搜索 `Llama-Guard-3-8B`、`Qwen3Guard-Gen-8B`、`Llama`、`Qwen` 多个关键词；不要调用详情、血缘、枚举或 clone，除非用户后续明确指定要查看某个候选或执行克隆。

**常见错误场景**：
- 接口返回 `404 接口不存在` → 提示"model-manager 尚未部署官方模型 MCP 接口，请联系平台运维"
- 接口返回 `403 权限不足` → 重新核对 wsid 与 Token
- `total_count > 0` 但页面看不到 → 该 wsid 在 `model_show_in_wsids` 的可见范围内，可能与用户当前页面登录态不一致；让用户确认是否在同一空间下

---

## release_hunyuan_training_checkpoint_as_model

**用途**：将混元训练平台产出的 checkpoint 发布为模型卡片。需指定运行实例 ID（instance_id）、checkpoint 名称、模型名称、模型描述、工作空间（wsid）等信息，可选指定思考类型、供应商、模型结构、激活/总参数量、上下文长度、场景类型、视觉结构、触发词、已有模型 ID 等。后端异步处理发布流程，返回任务记录 ID（task_record_id）、模型 ID、模型名称、模型路径、导出状态（如 PENDING）等。本接口不涉及模型格式转换。

> ⚠️ **本工具为写入操作**，调用前必须向用户复述要发布的模型名 / checkpoint / 目标 wsid，确认后才执行。
>
> ℹ️ **异步任务查询**：本工具仅创建发布任务并立即返回 `task_record_id` 和初始状态（如 `PENDING` / `PUBLISHING`），后续进度需通过 `get_hunyuan_training_model_release_status` 查询。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `instance_id` | string | ✅ | 无 | 运行实例 ID，必填 |
| `name` | string | ✅ | 无 | 模型名称，必填 |
| `desc` | string | ✅ | 无 | 模型描述，必填，不超过 250 个字符 |
| `checkpoint` | string | ✅ | 无 | checkpoint 名称，必填，如 `"checkpoint-1"` |
| `wsid` | integer | ✅ | 无 | 工作空间 ID，必填，**不能为 0 或空值** |
| `path` | string | ❌ | 无 | checkpoint 路径（Ceph 完整路径） |
| `old_path` | string | ❌ | 无 | checkpoint 源路径（当 action=mv 时的源路径） |
| `thought_type` | string | ❌ | 无 | 模型的思考类型。可选值仅限：`"all"`（全思考）、`"deepThought"`（深度思考）、`"quickThought"`（快思考）；不传即由后端处理 |
| `manufacturer` | string | ❌ | 无 | 供应商（如 `"tencent"`、`"hunyuan"`） |
| `manufacturer_series` | string | ❌ | 无 | 供应商子系列（如 `"hunyuan"`、`"hunyuan3.0-Chat"`） |
| `model_structure` | string | ❌ | 无 | 模型结构。可选值：`"moe"`、`"dense"`、`"MoE"`、`"Dense"` |
| `activate_params` | string | ❌ | 无 | 激活参数量，如 `"3B"`。**moe 模型必填** |
| `total_params` | string | ❌ | 无 | 总参数量，如 `"30B"` |
| `context_len` | string | ❌ | 无 | 上下文长度，如 `"32k"` |
| `category` | string | ❌ | 无 | 模型类别 category，可选；不传时保持后端现有自动推断逻辑 |
| `action` | string | ❌ | `"mv"` | 保存 checkpoint 的 action。可选值：`"mv"`（移动）、`"cp"`（复制）、`"no_action"`（仅创建记录） |
| `scene_train` | string | ❌ | 无 | 场景（如 `"multimodal"`、`"text"` 等） |
| `scene_text_type` | string | ❌ | 无 | 子场景（如 `"multimodal"` 等） |
| `vit_input_resolution` | string | ❌ | 无 | 分辨率（如 `"Anyres"`） |
| `visual_structure` | string | ❌ | 无 | 视觉结构（如 `"ViT1B-Siglip-TP_Learnable"`） |
| `trigger_prompt` | string | ❌ | 无 | 文生图触发词 |
| `model_id` | integer | ❌ | 无 | 已有模型 ID（当更新已有模型卡片时传入） |

> ⚠️ **dense vs moe 模型的 activate_params 规则**：
> - **dense 模型**：全量参数 = 激活参数，用户只需提供 `total_params`，`activate_params` 可不传
> - **moe 模型**：全量参数 ≠ 激活参数，两者都需要用户提供
> - **通用规则**：激活参数不允许大于全量参数，否则会报错

### 推荐调用流程

1. 先按 `references/helper_api.md`（`query_hunyuan_train_checkpoint_list`）调 `query_hunyuan_train_checkpoint_list` 获取训练任务的产出列表。
2. 用户已指定 checkpoint（如 `checkpoint-1`）时，直接在列表中匹配该项；用户未指定时再让用户选择。
3. 从产出列表中获取 `instance_id`、checkpoint 名称和 `path`，映射到本工具的 `instance_id` / `checkpoint` / `path`。
4. `finetuning_*` 模版化训练通常只需产出列表 + 本发布工具；不要额外搜索模型或克隆模型。
5. `basic_train_*` / 自定义训练若缺少模型基础参数，再调用一次 `list_hunyuan_models_platform_enums` 查询枚举值并补齐 `manufacturer` / `manufacturer_series` / `model_structure` / `total_params` / `context_len` / `scene_train` / `scene_text_type`（moe 还需 `activate_params`）。
6. 调用本工具发布模型；若返回"已发布/无权限/参数缺失"等业务错误，直接透传，不要改走 `search_hunyuan_models_cards`、`clone_hunyuan_models_card` 或训练详情工具。

### 使用示例

```bash
# 示例1：基础发布（仅必填参数）
python3 scripts/connect_mcp.py call release_hunyuan_training_checkpoint_as_model '{
  "instance_id": "95a68c749d6c722a019d6ce93cf10040",
  "name": "hunyuan_sft_v1",
  "desc": "混元微调模型 v1",
  "checkpoint": "checkpoint-1",
  "wsid": 10314
}'

# 示例2：指定路径和思考类型
python3 scripts/connect_mcp.py call release_hunyuan_training_checkpoint_as_model '{
  "instance_id": "95a68c749d6c722a019d6ce93cf10040",
  "name": "hunyuan_sft_v1",
  "desc": "混元微调模型 v1",
  "checkpoint": "checkpoint-1",
  "wsid": 10314,
  "path": "/apdcephfs_sh7/share_xxx/ckpt/checkpoint-1",
  "thought_type": "deepThought",
  "action": "mv"
}'

# 示例3：自定义训练发布 - dense 模型（需要额外参数）
python3 scripts/connect_mcp.py call release_hunyuan_training_checkpoint_as_model '{
  "instance_id": "8b1731349a5929fd019a595d7fc7000b",
  "name": "qwen2.5_32b_sft_v1",
  "desc": "基于 qwen2.5-32B 的 SFT 模型",
  "checkpoint": "checkpoint-189",
  "wsid": 10314,
  "path": "/apdcephfs/share_302507459/hunyuan/xiulingwu/ckpt/checkpoint-189",
  "thought_type": "quickThought",
  "action": "mv",
  "manufacturer": "qwen",
  "manufacturer_series": "qwen2.5",
  "model_structure": "dense",
  "total_params": "32B",
  "context_len": "40k",
  "scene_train": "text",
  "scene_text_type": "chat"
}'
# 注意：dense 模型无需传 activate_params

# 示例4：自定义训练发布 - moe 模型（需要额外提供 activate_params）
python3 scripts/connect_mcp.py call release_hunyuan_training_checkpoint_as_model '{
  "instance_id": "abc123def456",
  "name": "hunyuan_moe_v1",
  "desc": "混元 MoE 模型 v1",
  "checkpoint": "checkpoint-500",
  "wsid": 10314,
  "path": "/apdcephfs/share_xxx/ckpt/checkpoint-500",
  "thought_type": "quickThought",
  "action": "mv",
  "manufacturer": "hunyuan",
  "manufacturer_series": "HunYuanMoEV2ForCausalLM",
  "model_structure": "moe",
  "total_params": "389B",
  "activate_params": "52B",
  "context_len": "128k",
  "scene_train": "text",
  "scene_text_type": "chat"
}'
# 注意：moe 模型必须同时提供 total_params 和 activate_params

# 示例5：使用复制模式发布（保留原文件）
python3 scripts/connect_mcp.py call release_hunyuan_training_checkpoint_as_model '{
  "instance_id": "95a68c749d6c722a019d6ce93cf10040",
  "name": "hunyuan_sft_v1",
  "desc": "混元微调模型 v1",
  "checkpoint": "checkpoint-1",
  "wsid": 10314,
  "path": "/apdcephfs_sh7/share_xxx/ckpt/checkpoint-1",
  "action": "cp"
}'

# 示例6：更新已有模型卡片
python3 scripts/connect_mcp.py call release_hunyuan_training_checkpoint_as_model '{
  "instance_id": "95a68c749d6c722a019d6ce93cf10040",
  "name": "hunyuan_sft_v1",
  "desc": "混元微调模型 v1 更新版",
  "checkpoint": "checkpoint-2",
  "wsid": 10314,
  "model_id": 180500
}'
```

### 返回值格式（成功）

```json
{
  "task_record_id": 56789,
  "model_id": 180500,
  "model_name": "hunyuan_sft_v1",
  "model_path": "/apdcephfs_sh7/share_xxx/model/hunyuan_sft_v1",
  "export_status": "PENDING",
  "instance_id": "95a68c749d6c722a019d6ce93cf10040",
  "checkpoint": "checkpoint-1",
  "wsid": 10314
}
```

### 返回值字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_record_id` | Long | 导出任务记录 ID（`tj_hy_fine_tuning_model` 表主键） |
| `model_id` | Long | 发布后的模型卡片 ID（aide 模型 ID），异步发布时可能为 null |
| `model_name` | String | 模型名称 |
| `model_path` | String | 发布后的模型路径 |
| `export_status` | String | 导出状态枚举：`PENDING` / `PROCESSING` / `SUCCESS` / `FAILED` |
| `instance_id` | String | 运行实例 ID（回显） |
| `checkpoint` | String | checkpoint 名称（回显） |
| `wsid` | Long | 工作空间 ID（回显） |

### 交互规则

1. 用户给了训练任务 ID（`basic_train_*` / `finetuning_*`）但未提供 `instance_id/path` → 先按 `references/helper_api.md`（`query_hunyuan_train_checkpoint_list`）调 `query_hunyuan_train_checkpoint_list`；不要先调用 `get_hunyuan_train_task_detail` 或 `get_hunyuan_train_task_config_file`。
2. 用户说"发布模型"但既没有训练任务 ID 也没有 instance_id → 必须追问："请提供训练任务 ID 或运行实例 ID"。
3. 用户未提供 checkpoint → 从产出列表中展示可选 checkpoint；用户已指定 checkpoint 时直接匹配该项。
4. **`thought_type` 为可选**：用户明确指定时才传，取值仅限 `"all"` / `"deepThought"` / `"quickThought"` 三选一；用户未提及则不传，交给后端默认处理，不要擅自猜测或补默认值。
5. **自定义训练额外参数缺失时**：只需调用一次 `list_hunyuan_models_platform_enums` 查询可选值；不要读取训练详情/config 文件来推断。如果 `model_structure` 为 `"moe"`，还必须收集 `activate_params`；如果为 `"dense"`，无需收集 `activate_params`。
6. 发布失败（已发布、无权限、参数错误）后直接透传错误，停止；不要搜索模型、克隆模型或重复 release。
7. 操作成功后引导用户搜索刚发布的模型或查看模型详情；若用户想了解发布进度，用返回的 `task_record_id` 调用 `get_hunyuan_training_model_release_status`。

---

## clone_hunyuan_models_card

> 🛑 **场景区分**：用户给参考卡片 + 新路径 → 用本工具克隆，**不要**因为路径含 `ckpt/iter_` 就试图走 `release_hunyuan_training_checkpoint_as_model` 或 `query_hunyuan_train_checkpoint_list`。
> 🛑 **路径错误处理**：克隆失败（路径不存在、目录为空等）时，**立即停止**并告知用户错误原因。**不得**调用 `list_storage_dir` 等 storage 域工具检查路径、反复重试或写脚本绕过。

**用途**：从已有模型卡片克隆出一个新的模型卡片。克隆操作会复制源模型卡片的所有配置参数（如模型类型、参数规模、地域信息等），仅使用用户提供的新名称、描述和模型路径来创建新卡片。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `source_mould_id` | integer | ✅ | 无 | 源模型卡片 ID，要克隆的模型卡片。可通过 `search_hunyuan_models_cards` 或 `get_hunyuan_models_card_detail` 获取 |
| `name` | string | ✅ | 无 | 新模型名称，克隆后的模型卡片名称 |
| `path` | string | ✅ | 无 | 新模型的 checkpoint 路径 |
| `wsid` | integer | ✅ | 无 | 工作空间 ID，**强制必填，不能为 0 或空值**。如果用户未提供，必须追问："请提供您的工作空间 ID (wsid)" |
| `desc` | string | ❌ | 无 | 新模型描述，可选；如果未填写则自动使用 `name` 作为描述 |
| `hf_model_path` | string | ❌ | `""` | HuggingFace 格式模型路径，可选 |
| `compression_path` | string | ❌ | `""` | 压缩模型路径，可选 |

### 使用场景

- 微调训练完成后，基于基座模型卡片克隆出一个新卡片，指向微调后的 checkpoint 路径
- 复制已有模型卡片的配置，快速创建一个新版本的模型卡片

### 推荐调用流程

1. 源模型是 URL 或明确 `model_id` → 直接 `get_hunyuan_models_card_detail`；源模型只有名称 → 只调用一次 `search_hunyuan_models_cards(keyword=<完整源模型名>, is_show_all=true)` 获取最匹配 `model_id`。
2. 获取源模型详情后，确认新模型的名称和 checkpoint 路径。若用户说"step 240/360"且源卡片名或路径中有 `global_step_xxx` / `iter_xxx`，按用户指定 step 替换生成新名称和路径；不要反复搜索多个候选。
3. 调用本工具克隆模型卡片。
4. 克隆返回 `path目录为空`、`模型路径已关联卡片` 等业务错误时，直接透传错误并建议用户确认路径；不要自动改用 `release_hunyuan_training_checkpoint_as_model`、训练任务查询或再次 clone。

### 使用示例

```bash
# 基于源模型克隆一个新模型卡片
python3 scripts/connect_mcp.py call clone_hunyuan_models_card '{
  "source_mould_id": 169755,
  "name": "hunyuan_sft_v2",
  "path": "/apdcephfs_sh7/share_303786641/hunyuan/model/hunyuan_sft_v2/checkpoint",
  "wsid": 10103
}'

# 克隆模型并指定描述和 HF 路径
python3 scripts/connect_mcp.py call clone_hunyuan_models_card '{
  "source_mould_id": 169755,
  "name": "hunyuan_sft_v2_hf",
  "path": "/apdcephfs_sh7/share_303786641/hunyuan/model/hunyuan_sft_v2/checkpoint",
  "desc": "基于 hunyuan 基座微调的 SFT v2 版本",
  "hf_model_path": "/apdcephfs_sh7/share_303786641/hunyuan/model/hunyuan_sft_v2/hf",
  "wsid": 10103
}'
```

### 返回值格式（成功）

```json
{
  "source_mould_id": 169755,
  "new_model_id": 180601,
  "name": "hunyuan_sft_v2",
  "path": "/apdcephfs_sh7/share_303786641/hunyuan/model/hunyuan_sft_v2/checkpoint",
  "desc": "hunyuan_sft_v2"
}
```

### 返回值字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `source_mould_id` | Integer | 源模型卡片 ID |
| `new_model_id` | Integer | 新克隆出的模型卡片 ID |
| `name` | String | 新模型名称 |
| `path` | String | 新模型路径 |
| `desc` | String | 新模型描述 |

### 返回值格式（失败）

```json
{
  "code": 400,
  "message": "源模型不存在，请检查 source_mould_id",
  "data": null
}
```

### 交互规则

1. 用户说"克隆模型"但未提供源模型 ID 或源模型名称 → 必须追问："请提供要克隆的源模型 ID 或完整模型名称"
2. 用户未提供新模型名称，但给了新路径或 step → 可从路径尾部/step 自动生成可读名称；完全无法推断时再追问。
3. 用户未提供模型路径 → 必须追问："请提供新模型的 checkpoint 路径"
4. 用户未提供 wsid → 必须追问："请提供您的工作空间 ID (wsid)"
5. 如果用户只知道模型名称不知道 ID → 先调用一次 `search_hunyuan_models_cards` 搜索获取 ID，命中后再 `get_hunyuan_models_card_detail`，然后 clone。
6. 操作成功后引导用户查看新模型详情或搜索模型。
7. **路径唯一性/空目录错误处理**：如果调用返回 `模型路径已关联卡片【xxx】` 或 `path目录为空` 类错误：直接向用户报告后端错误，请用户更换新路径、补齐模型文件或先释放已占用的卡片。**不要**自动重试同一路径，也不要改用发布 checkpoint 或数据拷贝工具绕过。

---

## update_hunyuan_models_card_permission

**用途**：增量更新模型卡片的权限信息，支持增加或移除管理员、普通成员、可展示空间和可使用空间。

> ⚠️ **核心语义：增量操作，非全量覆盖**
> 本工具会先自动查询模型当前的完整权限配置，然后根据 `operation` 参数进行增量合并：
> - `operation="add"`（默认）：将传入的用户/用户组/空间**追加**到已有权限中，不影响已有成员
> - `operation="remove"`：将传入的用户/用户组/空间从已有权限中**移除**，不影响其他成员
>
> 例如：用户说"把 kevincen 加到普通成员"，只需传 `common_users=["kevincen"]`，工具会自动保留已有的所有普通成员，仅追加 kevincen。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `model_id` | integer | ✅ | 无 | 模型 ID，可通过 `search_hunyuan_models_cards` 或 `get_hunyuan_models_card_detail` 获取 |
| `staffname` | string | ✅ | 无 | 操作人的 RTX 用户名，**强制必填** |
| `wsid` | integer | ✅ | 无 | 工作空间 ID，**不能为 0 或空值** |
| `operation` | string | ❌ | `"add"` | 操作模式。`"add"`（增加，默认）：将传入的用户/用户组/空间追加到已有权限中；`"remove"`（移除）：将传入的用户/用户组/空间从已有权限中移除 |
| `spaces` | array[string] | ❌ | 无 | 模型卡片可展示的空间 ID 列表，元素为**字符串**，如 `["10314", "10103", "10362"]` |
| `admin_users` | array[string] | ❌ | 无 | 管理员用户 RTX 列表，元素为**字符串**，如 `["zhongtaohe", "kevincen"]` |
| `admin_user_groups` | array[integer] | ❌ | 无 | 管理员用户组 ID 列表，元素为**整数**，如 `[8, 10]` |
| `common_users` | array[string] | ❌ | 无 | 普通成员用户 RTX 列表，元素为**字符串**，如 `["shushuyang", "zesenwu"]` |
| `common_user_groups` | array[integer] | ❌ | 无 | 普通成员用户组 ID 列表，元素为**整数**，如 `[8]` |
| `usable_space` | array[string] | ❌ | 无 | 可使用该模型的空间 ID 列表，元素为**字符串**，如 `["10314", "10103"]`。该空间下的所有成员都可以使用此模型 |

> ⚠️ **参数类型警告**（务必按 array 传，不要传逗号分隔字符串）：
> - `spaces` / `admin_users` / `common_users` / `usable_space` 是 **array[string]**（字符串数组）
> - `admin_user_groups` / `common_user_groups` 是 **array[integer]**（整数数组，用户组 ID 是数字，不要引号）
> - ✅ 正确：`"admin_users": ["alice", "bob"]`、`"common_user_groups": [8, 10]`
> - ❌ 错误：`"admin_users": "alice,bob"`、`"common_user_groups": "8,10"` — 后端 DTO 是 `List<String>`/`List<Integer>`，字符串会被 Jackson 报 `MismatchedInputException`
> - ❌ 错误：`"admin_user_groups": ["8", "10"]` — 用户组 ID 元素类型是 integer，写成字符串会 400
> - 单个元素也必须放在数组里：单个用户传 `["kevincen"]` 而不是 `"kevincen"`

### 使用示例

```bash
# 增加普通成员（不影响已有成员）
python3 scripts/connect_mcp.py call update_hunyuan_models_card_permission '{
  "model_id": 180485,
  "staffname": "shushuyang",
  "operation": "add",
  "common_users": ["kevincen"],
  "wsid": 10103
}'

# 增加管理员并分享到新空间（不影响已有管理员和空间）
python3 scripts/connect_mcp.py call update_hunyuan_models_card_permission '{
  "model_id": 180485,
  "staffname": "shushuyang",
  "operation": "add",
  "admin_users": ["zhongtaohe"],
  "spaces": ["10362"],
  "wsid": 10103
}'

# 移除某个普通成员（不影响其他成员）
python3 scripts/connect_mcp.py call update_hunyuan_models_card_permission '{
  "model_id": 180485,
  "staffname": "shushuyang",
  "operation": "remove",
  "common_users": ["kevincen"],
  "wsid": 10103
}'

# 增加用户组到普通成员（用户组 ID 是整数，不要加引号）
python3 scripts/connect_mcp.py call update_hunyuan_models_card_permission '{
  "model_id": 180485,
  "staffname": "shushuyang",
  "operation": "add",
  "common_user_groups": [8],
  "wsid": 10103
}'

# 一次追加多种权限（管理员 + 空间 + 用户组）
python3 scripts/connect_mcp.py call update_hunyuan_models_card_permission '{
  "model_id": 180485,
  "staffname": "shushuyang",
  "operation": "add",
  "admin_users": ["alice", "bob"],
  "admin_user_groups": [8, 10],
  "spaces": ["10362", "10103"],
  "wsid": 10103
}'
```

### 返回值格式（成功）

假设该模型**变更前**已有权限为：
```json
{
  "admin_users": ["shushuyang"],
  "admin_user_groups": [8],
  "common_users": ["zesenwu"],
  "common_user_groups": [],
  "spaces": ["10314"],
  "usable_space": ["10314"]
}
```

调用 `operation="add"` 追加 `admin_users=["zhongtaohe"]`、`common_users=["kevincen"]`、`spaces=["10103"]`、`usable_space=["10103"]`、`common_user_groups=[8]` 后，返回：

```json
{
  "model_id": 180485,
  "operation": "add",
  "changes": {
    "added_admins": ["zhongtaohe"],
    "added_members": ["kevincen"]
  },
  "final_permissions": {
    "admin_users": ["shushuyang", "zhongtaohe"],
    "admin_user_groups": [8],
    "common_users": ["zesenwu", "kevincen"],
    "common_user_groups": [8],
    "spaces": ["10314", "10103"],
    "usable_space": ["10314", "10103"]
  }
}
```

> 📌 **对照要点**（Agent 必须看懂）：
> - `admin_users` 从 `["shushuyang"]` → `["shushuyang", "zhongtaohe"]`：**shushuyang 被完整保留**，zhongtaohe 是本次追加
> - `common_users` 从 `["zesenwu"]` → `["zesenwu", "kevincen"]`：**zesenwu 被完整保留**，kevincen 是本次追加
> - `spaces` / `usable_space` 从 `["10314"]` → `["10314", "10103"]`：**10314 被完整保留**，10103 是本次追加
> - `admin_user_groups` 未在本次变更中传入，保持原值 `[8]`；`common_user_groups` 传入 `[8]` 后变为 `[8]`
> - **remove 场景同理**：`operation="remove"` 只把入参里给的用户/组从原成员中剔除，未入参的成员一律保留

### 返回值字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_id` | Integer | 模型 ID |
| `operation` | String | 操作模式（add / remove） |
| `changes` | Map | 变更摘要 |
| `final_permissions.admin_users` | List | 变更后的管理员用户列表 |
| `final_permissions.admin_user_groups` | List | 变更后的管理员用户组 ID 列表 |
| `final_permissions.common_users` | List | 变更后的普通成员用户列表 |
| `final_permissions.common_user_groups` | List | 变更后的普通成员用户组 ID 列表 |
| `final_permissions.spaces` | List | 可展示空间 ID 列表 |
| `final_permissions.usable_space` | List | 可使用空间 ID 列表 |

### 交互规则

1. 用户说"设置权限"但未提供模型 ID → 必须追问："请提供模型 ID"
2. 用户未提供 staffname → 必须追问："请提供您的 RTX 用户名"
3. 用户说"把某人加到成员/管理员" → 使用 `operation="add"`，只传需要增加的用户，工具自动保留已有权限
4. 用户说"把某人从成员/管理员中移除" → 使用 `operation="remove"`，只传需要移除的用户
5. 各权限字段均为可选，用户可以只操作部分权限
6. 操作成功后引导用户查看模型详情确认权限更新结果

---

## update_hunyuan_models_card_location

**用途**：为模型卡片新增地域，将模型文件拷贝到新地域的应用组路径下。工具会自动查询模型当前的所有地域信息，将新地域追加进去，并将完整的地域列表提交给后端。


### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `model_id` | integer | ✅ | 无 | 模型 ID，可通过 `search_hunyuan_models_cards` 或 `get_hunyuan_models_card_detail` 获取 |
| `queue_name` | string | ✅ | 无 | 新地域使用的应用组名称（即 queue_name） |
| `wsid` | integer | ✅ | 无 | 工作空间 ID，**不能为 0 或空值** |
| `new_locations` | string | ▲ | 无 | 新增地域英文缩写，多个用英文逗号分隔，如 `"sz,sh"`。**当提供 `cluster_name` 或 `container_path` 时可不传** |
| `path_prefix` | string | ▲ | 无 | 新地域的 ceph 路径前缀（前两级目录），如 `"/apdcephfs_sh3/share_123456"`。**当提供 `cluster_name` 或 `container_path` 时可不传** |
| `cluster_name` | string | ❌ | 无 | ceph 集群名称，如 `"jp_gy6_cephfs"`。提供后会自动从应用组 ceph 信息中匹配 `location` 和 `path_prefix`，无需再传 `new_locations`/`path_prefix` |
| `container_path` | string | ❌ | 无 | 容器挂载路径，如 `"/apdcephfs_gy6/share_303741887"`。提供后会自动从应用组 ceph 信息中匹配 `location` 和 `path_prefix`，无需再传 `new_locations`/`path_prefix` |
| `is_need_copy` | boolean | ❌ | `true` | 是否需要执行 ceph 文件拷贝。`false` 仅添加地域记录不拷贝文件 |

> ℹ️ **两组参数任选一组**：
> - **方式 A（手动指定）**：同时传 `new_locations` + `path_prefix`，适用于已知地域缩写和路径前缀的场景；
> - **方式 B（自动推断，推荐）**：传 `cluster_name` 或 `container_path` 中的任一个，后端会从应用组 ceph 信息中自动匹配地域和路径前缀。

### 路径生成规则

- 取已有地域中第一个有效的 path，将前两级目录替换为用户提供的 `path_prefix`
- 例如原路径：`/apdcephfs_cq11/share_303741887/hunyuan/.../hf`
- 用户提供 `path_prefix=/apdcephfs_sh3/share_123456`
- 新路径：`/apdcephfs_sh3/share_123456/hunyuan/.../hf`
- `hf_model_path` 和 `compression_path` 同理处理

### 推荐调用流程

1. 先调用 `get_hunyuan_models_card_detail` 查看模型当前的地域分布。
2. 若用户要求"重新拷贝/重试"某个已存在但失败的地域（如 sg/sh/gy 的 `copy_state=failed`）：直接从该地域 `location_infos` 复用已有 `queue_name`，并从该地域 `path` 前两级提取 `path_prefix`（如 `/apdcephfs_sgmy/share_303999818`），然后调用本工具，`new_locations` 填目标地域、`is_need_copy=true`；不要再查平台枚举、应用组列表或 ceph locations。
3. 若目标地域尚不存在且用户未给应用组/路径前缀，再按 `references/helper_api.md`（`query_user_app_groups`）调 `query_user_app_groups` 选择应用组，并调 `query_app_group_ceph_locations` 获取 `containerPath` 作为 `path_prefix`。
4. 调用本工具新增或重试地域；若返回"地域已存在"，直接透传并说明该地域记录已存在，必要时请平台侧清理或授权。

### 使用示例

```bash
python3 scripts/connect_mcp.py call update_hunyuan_models_card_location '{
  "model_id": 112187,
  "new_locations": "sh",
  "queue_name": "TaiJi_HYAide_GZZY",
  "path_prefix": "/apdcephfs_sh9/share_303741887",
  "is_need_copy": true,
  "wsid": 10103
}'

# 方式 B：自动推断（传 container_path，无需 new_locations/path_prefix）
python3 scripts/connect_mcp.py call update_hunyuan_models_card_location '{
  "model_id": 112187,
  "queue_name": "TaiJi_HYAide_GZZY",
  "container_path": "/apdcephfs_sh9/share_303741887",
  "is_need_copy": true,
  "wsid": 10103
}'
```

### 返回值格式（成功）

```json
{
  "model_id": 112187,
  "added_locations": [
    {
      "location": "sh",
      "path": "/apdcephfs_sh9/share_303741887/hunyuan/.../hf",
      "hf_model_path": null,
      "queue_name": "TaiJi_HYAide_GZZY",
      "is_need_copy": true
    }
  ],
  "total_locations": ["cq", "sh"]
}
```

### 返回值字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_id` | Integer | 模型 ID |
| `added_locations` | List | 新增的地域列表 |
| `added_locations[].location` | String | 地域英文缩写 |
| `added_locations[].path` | String | 模型路径 |
| `added_locations[].hf_model_path` | String | HF 格式模型路径（可能为 null） |
| `added_locations[].queue_name` | String | 应用组名称 |
| `added_locations[].is_need_copy` | Boolean | 是否需要拷贝文件 |
| `total_locations` | List | 更新后的完整地域列表 |

### 交互规则

1. 用户说"添加地域/重新拷贝地域"但未提供模型 ID 或模型名称 → 必须追问；只给模型名称时先 `search_hunyuan_models_cards` 搜一次拿 ID。
2. 目标地域已在详情中存在且 `copy_state=failed` → 复用该地域已有 `queue_name`，从该地域 `path` 前两级提取 `path_prefix`，可直接调用本工具重试。
3. 只有目标地域不存在或详情缺少 `queue_name/path` 时，才按 `references/helper_api.md`（`query_user_app_groups` / `query_app_group_ceph_locations`）获取应用组列表和 `containerPath`。
4. 操作成功后引导用户查看模型详情确认地域更新结果；失败则透传错误，不要自动切到数据处理工具搬运文件。

---

## create_hunyuan_models_cache

**用途**：将模型文件预热到指定地域的 ADT 高速缓存中，加速后续推理请求的模型加载速度。预热操作会将模型文件从 ceph 存储加载到高速缓存层，使得后续的推理服务可以更快地加载模型。

> ⚠️ **预热是异步操作**：接口返回成功仅表示预热任务已发起，实际完成需要一定时间。
>
> ℹ️ **格式支持由后端判定**：能否预热、以及从哪个字段取 HF 路径（例如 `category=2` 时 HF 路径存放在 `path` 字段，其余情况在 `hf_model_path`），全部由服务端决定。**Agent 不做客户端预判**——用户要求预热就直接调用本工具，让后端返回结果，失败时透传错误即可。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `model_id` | integer | ✅ | 无 | 模型卡片 ID（Mould 表的 id），可通过 `search_hunyuan_models_cards` 搜索获取 |
| `regions` | list[int] | ✅ | 无 | 地域 ID 列表（MouldLocation 表的 id），⚠️ **不是地域英文缩写**，需从 `get_hunyuan_models_card_detail` 返回的 `location_infos` 中获取对应地域的 `id` 字段 |
| `wsid` | integer | ✅ | 无 | 工作空间 ID，**强制必填，不能为 0 或空值**。如果用户未提供，必须追问："请提供您的工作空间 ID (wsid)" |
| `model_ttl` | integer | ❌ | 30 | 缓存有效期（天），范围 1-90 天 |
| `is_force_refresh` | boolean | ❌ | false | 是否强制刷新缓存。设为 true 时即使缓存未过期也会重新预热 |

### 推荐调用流程

> ⚠️ **`regions` 参数是地域记录的数据库 ID（整数），不是地域英文缩写（如 sz、sh）。必须通过以下流程获取正确的 regions 值。**

1. 用户说"预热 xx 模型的 xx 地区"。
2. 用户已给 `model_id` → 直接 `get_hunyuan_models_card_detail`；用户只给模型名 → 先 `search_hunyuan_models_cards` 搜索获取 `model_id`。
3. 只调用一次 `get_hunyuan_models_card_detail` 获取模型详情，从返回的 `location_infos` 中：
   - 根据用户提到的地域名称匹配 `location` / `ch_location` 字段（如 `sg`/"韶关"、`sh`/"上海"）
   - 获取对应记录的 `id` 字段作为 `regions` 参数
   - **不要在客户端预判该地域/该格式能否预热**（`hf_model_path` 是否为空、`category` 取值、路径存在哪个字段等一律不做判断），直接进入下一步
4. 调用本工具 `create_hunyuan_models_cache` 发起预热；不要重复查询详情。是否支持预热、HF 路径从哪个字段读，全部交给后端判定，失败时透传错误即可。

### 使用示例

```bash
# 预热模型到指定地域（单个地域）
python3 scripts/connect_mcp.py call create_hunyuan_models_cache '{
  "model_id": 200911,
  "regions": [176805],
  "wsid": 10103
}'

# 预热模型到多个地域，自定义缓存时间
python3 scripts/connect_mcp.py call create_hunyuan_models_cache '{
  "model_id": 200911,
  "regions": [176805, 176806],
  "model_ttl": 60,
  "wsid": 10103
}'

# 强制刷新缓存（即使未过期也重新预热）
python3 scripts/connect_mcp.py call create_hunyuan_models_cache '{
  "model_id": 200911,
  "regions": [176805],
  "model_ttl": 30,
  "is_force_refresh": true,
  "wsid": 10103
}'
```

### 返回值格式（成功）

```json
{
  "model_id": 200911,
  "regions": [176805],
  "model_ttl": 30,
  "is_force_refresh": false
}
```

### 返回值字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `model_id` | Integer | 模型 ID |
| `regions` | List[Integer] | 预热的地域 ID 列表 |
| `model_ttl` | Integer | 缓存有效期（天） |
| `is_force_refresh` | Boolean | 是否强制刷新 |

### 返回值格式（失败）

```json
{
  "code": 400,
  "message": "wza地域暂不支持预热",
  "data": null
}
```

### 常见错误场景

- `wza地域暂不支持预热` → 该地域不支持预热，请选择其他地域
- `该格式模型暂不支持预热` → 模型格式为纯 PTM/pytorch/DCP/mcore，不包含 HF 格式
- `模型不存在` → model_id 错误，请检查
- 权限不足 → 调用者不是模型的创建人、管理员或共享成员

### 交互规则

1. 用户说"预热模型"但未提供模型 ID → 必须追问："请提供模型名称或 ID"，然后调用 `search_hunyuan_models_cards` 搜索。
2. 用户提供了模型名称但未提供 ID → 先调用一次 `search_hunyuan_models_cards` 搜索获取 ID。
3. 用户未指定地域 → 先调用一次 `get_hunyuan_models_card_detail` 获取模型的地域分布，展示给用户选择。
4. 用户提供了地域名称（如"深圳"）但未提供地域 ID → 调用一次 `get_hunyuan_models_card_detail` 获取 `location_infos`，匹配 `location/ch_location` 字段获取 `id`。
5. 用户未提供 wsid → 必须追问："请提供您的工作空间 ID (wsid)"。
6. **不做客户端拦截**：只要用户提供了 `model_id` + `regions` + `wsid`，就直接调用本工具，不要因为 `hf_model_path` 为空、格式看起来不含 HF、`category` 是某个值等原因自行拒绝或改参数。后端返回"该格式模型暂不支持预热"、"xx地域暂不支持预热"、权限不足等错误时，**直接透传错误并停止**，不要更换参数重复调用。
7. 操作成功后引导用户查看模型详情或搜索其他模型。

---

## get_hunyuan_models_lineage

**用途**：获取混元模型卡片的血缘数据（父子谱系），根据 model_id 与 wsid 追溯该模型的上游/下游模型链路。可通过 parent_level 控制向上追溯的父节点层级（默认 1），通过 parent_simplify 控制父节点详情是否简化返回（默认 true，仅返回精简摘要以降低响应体积）。返回该模型的血缘树/图数据（父模型节点、子模型节点及其基础属性）。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `model_id` | integer | ✅ | 无 | 模型 ID，必填 |
| `wsid` | integer | ✅ | 无 | 工作空间 ID，必填 |
| `parent_level` | integer | ❌ | 1 | 父节点层级，控制向上追溯几层父模型 |
| `parent_simplify` | boolean | ❌ | true | 是否简化父节点详情（true 只返回精简字段，false 返回完整字段） |

### 使用示例

```bash
# 查询模型血缘（默认向上追溯 1 层，简化父节点）
python3 scripts/connect_mcp.py call get_hunyuan_models_lineage '{
  "model_id": 180500,
  "wsid": 10103
}'

# 查询模型血缘，向上追溯 3 层，返回完整父节点信息
python3 scripts/connect_mcp.py call get_hunyuan_models_lineage '{
  "model_id": 180500,
  "wsid": 10103,
  "parent_level": 3,
  "parent_simplify": false
}'
```

### 返回值格式（示例）

```json
{
  "base_model_id": 180500,
  "entities": [
    {
      "model_id": 180500,
      "model_info": {
        "model_name": "Hunyuan-SFT-7B",
        "task_id": "task_abc123",
        "task_url": "https://taiji.woa.com/...",
        "task_platform": "taiji",
        "stage": "SFT",
        "created_at": "2026-06-20 14:30:00"
      },
      "task_info": {
        "id": 1001,
        "task_id": "task_abc123",
        "task_name": "SFT训练-v1",
        "task_url": "https://taiji.woa.com/...",
        "task_desc": "混元7B SFT微调",
        "template_id": 5,
        "template_name": "SFT-Mcore",
        "creator": "shushuyang",
        "created_at": "2026-06-20 14:00:00",
        "updated_at": "2026-06-20 16:00:00",
        "status": "SUCCEEDED"
      },
      "pretrain_data_info": {
        "data_id": 2001,
        "data_name": "pretrain_corpus_v3",
        "storage_path": "/apdcephfs/share_xxx/data/pretrain",
        "shuttle_task_id": 3001,
        "owner_list": ["alice"],
        "created_at": "2026-05-01 10:00:00",
        "shuttle_task_info": {
          "shuttle_task_id": 3001,
          "shuttle_task_name": "数据清洗任务",
          "owner_list": ["alice"],
          "created_at": "2026-04-20 09:00:00",
          "creator": "alice",
          "stage": "completed"
        }
      },
      "posttrain_data_info": null
    },
    {
      "model_id": 169755,
      "model_info": {
        "model_name": "Hunyuan-7B-Base",
        "task_id": null,
        "task_url": null,
        "task_platform": null,
        "stage": "Pretrain",
        "created_at": "2026-03-01 10:00:00"
      },
      "task_info": null,
      "pretrain_data_info": null,
      "posttrain_data_info": null
    }
  ],
  "relations": [
    { "parent_model_id": 169755, "children_model_id": 180500 }
  ]
}
```

### 返回值字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `base_model_id` | Integer | 中心节点模型 ID（即输入的查询模型） |
| `entities` | List | 血缘节点列表 |
| `entities[].model_id` | Integer | 节点模型 ID |
| `entities[].model_info` | Object | 模型基本信息（model_name / task_id / task_url / task_platform / stage / created_at） |
| `entities[].task_info` | Object | 训练任务信息（id / task_id / task_name / task_url / task_desc / template_id / template_name / creator / created_at / updated_at / status） |
| `entities[].pretrain_data_info` | Object | 预训练数据信息（data_id / data_name / storage_path / shuttle_task_id / owner_list / created_at / shuttle_task_info） |
| `entities[].posttrain_data_info` | Object | 后训练数据信息（data_id / data_name / storage_path / shuttle_id / experiment_id / owner_list / created_at / shuttle_info / experiment_info / topic_info） |
| `relations` | List | 血缘关系列表 |
| `relations[].parent_model_id` | Integer | 父模型 ID |
| `relations[].children_model_id` | Integer | 子模型 ID |

> **注意**：当 `parent_simplify=true` 时，父节点的 `task_info`/`pretrain_data_info`/`posttrain_data_info` 为 null（仅返回 `model_info` 精简摘要以降低响应体积）。

### 交互规则

1. 用户说"查看模型血缘/父模型/子模型/模型谱系"且已提供 `model_id` → 直接调用 `get_hunyuan_models_lineage`，不要先 `get_hunyuan_models_card_detail`。
2. 用户未提供 model_id → 必须追问："请提供模型 ID"，或先调用一次 `search_hunyuan_models_cards` 搜索获取 ID。
3. 用户未提供 wsid → 必须追问："请提供您的工作空间 ID (wsid)"。
4. 默认使用 `parent_level=1`、`parent_simplify=true`，除非用户明确要求查看更多层级或完整信息。
5. **必须联动查询关联评测任务（血缘入口专用）**：`get_hunyuan_models_lineage` 返回后，立即以同一 `model_id` 调用 `query_hunyuan_models_lineage_eval_tasks` 并**分页拉全**——从 `page=1, page_size=100`（该接口 `page_size` 上限）开始，根据返回的 `has_more` 字段循环递增 `page` 继续拉取，直到 `has_more=false` 或已累计到 `total` 条为止；然后把血缘节点/关系与合并后的完整评测任务列表**一并**返回给用户。不允许只拉第一页就结束（除非首页 `has_more=false`），也不允许把翻页操作留给用户。若某一页调用失败，透传错误并停止翻页，同时把已拉到的部分与"截断说明"一并汇报。

---

## query_hunyuan_models_lineage_eval_tasks

**用途**：查询指定混元模型关联的评测任务列表（用于血缘视图中展示评测节点），支持按评测任务状态过滤，并可分别控制基线任务优先、竞技场官方任务优先两种排序策略。返回分页结果，包含 items（评测任务摘要：任务 ID、名称、状态、评测数据集、评测指标、创建时间等）以及 page / page_size / total / has_more 分页字段。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `model_id` | integer | ✅ | 无 | 模型 ID，必填 |
| `status` | string | ❌ | 无 | 评测任务状态过滤，如 `PARSED` / `RUNNING` / `SUCCEEDED` / `FAILED` 等 |
| `is_baseline_first` | boolean | ❌ | true | 是否基线任务优先排序 |
| `is_arena_official_first` | boolean | ❌ | true | 是否竞技场官方任务优先排序 |
| `page` | integer | ❌ | 1 | 页码，从 1 开始 |
| `page_size` | integer | ❌ | 100 | 每页数量，最大 100；默认使用最大页大小以减少翻页次数 |

### 使用示例

```bash
# 查询模型关联的所有评测任务
python3 scripts/connect_mcp.py call query_hunyuan_models_lineage_eval_tasks '{
  "model_id": 180500
}'

# 按状态过滤，只查看已成功的评测任务
python3 scripts/connect_mcp.py call query_hunyuan_models_lineage_eval_tasks '{
  "model_id": 180500,
  "status": "SUCCEEDED",
  "page": 1,
  "page_size": 100
}'

# 关闭基线优先和竞技场官方优先排序
python3 scripts/connect_mcp.py call query_hunyuan_models_lineage_eval_tasks '{
  "model_id": 180500,
  "is_baseline_first": false,
  "is_arena_official_first": false
}'
```

### 返回值格式（示例）

```json
{
  "items": [
    {
      "eval_task_id": 10001,
      "arena_id": 5,
      "arena_name": "通用能力竞技场",
      "name": "基线评测-通用能力",
      "desc": "7B模型通用能力基线评测",
      "status": "SUCCEEDED",
      "model_id": 180500,
      "model_name": "Hunyuan-SFT-7B",
      "creator": "shushuyang",
      "wsid": 10103,
      "task_type": "baseline",
      "completed_num": 50,
      "total_num": 50,
      "created_at": "2026-06-25 10:00:00",
      "updated_at": "2026-06-25 12:30:00",
      "collection_infos": [
        {
          "collection_id": 201,
          "collection_version_id": 301,
          "name": "general_benchmark_v2",
          "description": "通用能力评测数据集 v2"
        }
      ],
      "error_rate": {
        "general_benchmark_v2": {
          "error_rate": 0.02,
          "infer_error_rate": 0.01,
          "judge_error_rate": 0.01
        }
      }
    }
  ],
  "page": 1,
  "page_size": 100,
  "total": 5,
  "has_more": false
}
```

### 返回值字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `items` | List | 评测任务列表 |
| `items[].eval_task_id` | Long | 评测任务 ID |
| `items[].arena_id` | Integer | 竞技场 ID |
| `items[].arena_name` | String | 竞技场名称 |
| `items[].name` | String | 评测任务名称 |
| `items[].desc` | String | 评测任务描述 |
| `items[].status` | String | 评测任务状态（PARSED / RUNNING / SUCCEEDED / FAILED）|
| `items[].model_id` | Integer | 关联模型 ID |
| `items[].model_name` | String | 关联模型名称 |
| `items[].creator` | String | 创建人 |
| `items[].wsid` | Integer | 工作空间 ID |
| `items[].task_type` | String | 任务类型 |
| `items[].completed_num` | Integer | 已完成评测数 |
| `items[].total_num` | Integer | 总评测数 |
| `items[].created_at` | String | 创建时间 |
| `items[].updated_at` | String | 更新时间 |
| `items[].collection_infos` | List | 评测数据集信息列表（collection_id / collection_version_id / name / description）|
| `items[].error_rate` | Map | 按数据集维度的错误率详情（error_rate / infer_error_rate / judge_error_rate）|
| `page` | Integer | 当前页码 |
| `page_size` | Integer | 每页数量 |
| `total` | Long | 命中总数 |
| `has_more` | Boolean | 是否还有更多数据 |

### 交互规则

1. 用户说"模型关联了哪些评测任务 / 血缘里的评测节点 / 关联评测"且已提供 `model_id` → 直接调用本工具，不要先调 `get_hunyuan_models_card_detail`、`get_hunyuan_models_lineage` 或 evaluation-skill 的任务列表。
2. 用户未提供 model_id → 必须追问："请提供模型 ID"，或先调用一次 `search_hunyuan_models_cards` 搜索获取 ID。
3. 默认不传 status（返回所有状态的评测任务），除非用户明确要求只看某种状态。
4. 分页参数**统一默认** `page=1`、`page_size=100`（该接口 `page_size` 上限，用大页减少翻页次数）；无论血缘联动还是用户单独调用，都以 100 为默认值，不再使用更小的默认页大小。用户明确要求更小页时再调整。
5. **血缘联动场景（由 `get_hunyuan_models_lineage` 触发）必须分页拉全**：从 `page=1, page_size=100` 开始循环递增 `page` 反复调用本工具，直到返回 `has_more=false` 或累计条数 ≥ `total` 为止，把各页的 `items` 合并去重后一并返回。用户单独调本工具查列表时，同样使用 `page_size=100` 拉首页，但**不自动翻页**（`has_more=true` 时向用户提示可继续翻页，由用户决定是否继续），这是与血缘联动场景的唯一区别。
6. 操作成功后可引导用户查看具体评测任务详情（切换到 evaluation-skill）。

---

## get_hunyuan_training_model_release_status

**用途**：查询由 `release_hunyuan_training_checkpoint_as_model` 发起的**模型发布异步任务**的状态。本接口直接透传底层 `ModelConvertStatusEnum` 状态值，包含 `FAILED` 即视为失败，`SUCCESS` 为最终成功；失败时通过 `error_message` 字段透出错误详情。支持三种查询方式（`task_record_id` / `model_id` / `model_name` 至少传一个），三者可同时传入，后端按 `task_record_id > model_id > model_name` 的优先级取用。

> ⚠️ **职责边界（必须遵守）**：本工具**只**用于查询"模型发布"这一次异步任务的整体状态。虽然发布链路内部**可能包含**一个模型格式转换阶段（因此返回的枚举里会出现 `CONVERT_*` 字样，代表发布链路内部的转换阶段），但**独立的"模型格式转换任务"状态查询建议走 `model-convert-skill`**（HF / Mcore 转换）——它有独立入口，会返回更详细的转换任务信息（进度、日志、产物路径等），本工具透出的 `CONVERT_*` 只是发布链路的粗粒度阶段名，信息量有限。
>
> - ✅ 推荐场景：用户是**通过 `release_hunyuan_training_checkpoint_as_model` 发布**触发的任务，之后追问"发布到哪了 / 发布成功了吗 / 我发布的模型现在什么状态"。
> - 🔄 **优先让位场景**：用户直接问"我那个模型格式转换任务跑到哪了""HF ↔ Mcore 转换进度"——**若环境中存在 `model-convert-skill`，则优先使用它**（提供更详细的转换任务信息）；仅当无法使用 `model-convert-skill` 且用户提供的是发布链路产生的 `task_record_id` 时，才能用本工具看到内部阶段名作为降级参考。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `task_record_id` | integer | ▲ | 无 | 发布任务记录 ID（`release_hunyuan_training_checkpoint_as_model` 返回的 `task_record_id`），**最精确**的查询方式；与 `model_id`、`model_name` 三者至少传一个 |
| `model_id` | integer | ▲ | 无 | 发布后的模型卡片 ID（aide 模型 ID）；与 `task_record_id`、`model_name` 三者至少传一个 |
| `model_name` | string | ▲ | 无 | 模型名称，若同名多条则返回最新一条（按 id 降序）；与 `task_record_id`、`model_id` 三者至少传一个 |

> ⚠️ **至少传一个**：`task_record_id` / `model_id` / `model_name` 三者必须至少提供一个；同时传多个时按 `task_record_id > model_id > model_name` 的优先级取用。

### 状态值说明（`ModelConvertStatusEnum`）

| 状态值 | 类别 | 说明 |
|--------|------|------|
| `PENDING` | 进行中 | 发布任务已创建，尚未开始处理 |
| `PUBLISHING` | 进行中 | 正在发布模型卡片 |
| `PUBLISHED` | 进行中 | 模型卡片已发布，等待后续（可能）的转换步骤 |
| `CREATING_CONVERT_TASK` | 进行中 | 正在创建模型格式转换任务 |
| `CONVERT_RUNNING` | 进行中 | 模型格式转换执行中 |
| `CONVERT_DONE` | 进行中 | 模型格式转换完成，等待导出 |
| `CONVERT_EXPORT_SUCCESS` | 进行中 | 转换产物导出成功，等待收尾 |
| `SUCCESS` | ✅ 最终成功 | 发布及（可选）转换全流程成功 |
| `PUBLISHED_FAILED` | ❌ 失败 | 发布模型卡片阶段失败 |
| `PUBLISHED_CHECK_FAILED` | ❌ 失败 | 发布后置校验失败 |
| `CONVERT_TASK_FAILED` | ❌ 失败 | 创建转换任务失败 |
| `CONVERT_FAILED` | ❌ 失败 | 转换任务执行失败 |
| `CONVERT_EXPORT_FAILED` | ❌ 失败 | 转换产物导出失败 |

### 使用示例

```bash
# 示例1：使用 task_record_id 查询（推荐，最精确）
python3 scripts/connect_mcp.py call get_hunyuan_training_model_release_status '{
  "task_record_id": 56789
}'

# 示例2：使用 model_id 查询
python3 scripts/connect_mcp.py call get_hunyuan_training_model_release_status '{
  "model_id": 180500
}'

# 示例3：使用 model_name 查询（同名多条时返回最新一条）
python3 scripts/connect_mcp.py call get_hunyuan_training_model_release_status '{
  "model_name": "hunyuan_sft_v1"
}'
```

### 返回值格式（成功、进行中）

```json
{
  "task_record_id": 56789,
  "status": "CONVERT_RUNNING",
  "error_message": null,
  "need_convert": true,
  "model_id": 180500,
  "model_name": "hunyuan_sft_v1",
  "model_path": "/apdcephfs_sh7/share_xxx/model/hunyuan_sft_v1",
  "convert_task_id": "convert_20260727_abc123",
  "instance_id": "95a68c749d6c722a019d6ce93cf10040",
  "checkpoint": "checkpoint-1",
  "wsid": 10314
}
```

### 返回值格式（失败）

```json
{
  "task_record_id": 56789,
  "status": "CONVERT_FAILED",
  "error_message": "模型转换失败：输入路径不存在",
  "need_convert": true,
  "model_id": 180500,
  "model_name": "hunyuan_sft_v1",
  "model_path": "/apdcephfs_sh7/share_xxx/model/hunyuan_sft_v1",
  "convert_task_id": "convert_20260727_abc123",
  "instance_id": "95a68c749d6c722a019d6ce93cf10040",
  "checkpoint": "checkpoint-1",
  "wsid": 10314
}
```

### 返回值字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `task_record_id` | Long | 发布任务记录 ID（`tj_hy_fine_tuning_model` 表主键） |
| `status` | String | 模型发布任务当前状态，直接透传底层 `ModelConvertStatusEnum` 名称，取值见「状态值说明」表；含 `FAILED` 即失败，`SUCCESS` 为最终成功。名称里出现的 `CONVERT_*` 仅表示"发布链路内部的转换阶段"，不代表本工具能查独立的模型格式转换任务 |
| `error_message` | String | 错误信息，仅在 `status` 为 `*_FAILED` 时透出，否则为 null |
| `need_convert` | Boolean | 是否需要模型格式转换（false 表示仅发布卡片，无转换步骤） |
| `model_id` | Long | 发布后的模型卡片 ID（aide 模型 ID） |
| `model_name` | String | 模型名称 |
| `model_path` | String | 发布后的模型路径 |
| `convert_task_id` | String | 模型转换任务 ID（仅需要转换时有值，可为 null） |
| `instance_id` | String | 运行实例 ID |
| `checkpoint` | String | checkpoint 名称 |
| `wsid` | Long | 工作空间 ID |

### 推荐调用流程

1. 用户在调用 `release_hunyuan_training_checkpoint_as_model` 后想追踪发布进度 → 用返回的 `task_record_id` 直接调用本工具。
2. 用户丢失 `task_record_id` 但知道 `model_id` 或 `model_name` → 用 `model_id` 或 `model_name` 查询；两者都可用时优先 `model_id`（更精确）。
3. 一次查询后按 `status` 判断：`SUCCESS` 视为完全成功；含 `FAILED` 视为失败，读取 `error_message` 透传给用户；其他状态为进行中，可稍后重试查询。

### 交互规则

1. 用户说"查询发布状态 / 发布进度 / 发布成功了吗 / 模型是否发布完成"等，且已经提供 `task_record_id`、`model_id` 或 `model_name` 中的任一个 → 直接调用本工具，不要先调 `get_hunyuan_models_card_detail` 或 `search_hunyuan_models_cards` 探路。
2. **意图识别边界**：只有当用户是通过 `release_hunyuan_training_checkpoint_as_model` 发布触发的追踪，才优先用本工具。若用户问的是"独立的模型格式转换任务状态"（例如手动发起 HF ↔ Mcore 转换、直接问 convert task 进度），**若环境中存在 `model-convert-skill`，则优先使用它**；仅当环境中不存在 `model-convert-skill` 且用户提供的是发布链路产生的 `task_record_id` / `model_id` / `model_name` 时，才降级用本工具透传内部阶段名。不要基于用户提供的孤立 `convert_task_id` 尝试反查本工具。
3. 用户三者都未提供 → 必须追问："请提供发布任务记录 ID (`task_record_id`)、模型 ID (`model_id`) 或模型名称 (`model_name`) 中的任一个"。
4. 用户同时提供多个查询键 → 直接一次性传给本工具，后端会按 `task_record_id > model_id > model_name` 的优先级取用，不需要自行取舍。
5. `status` 为进行中的中间态（`PENDING` / `PUBLISHING` / `PUBLISHED` / `CREATING_CONVERT_TASK` / `CONVERT_RUNNING` / `CONVERT_DONE` / `CONVERT_EXPORT_SUCCESS`）时，如实告知用户"发布仍在进行中"并展示当前阶段名（`CONVERT_*` 表示发布链路内部的转换阶段，属于本次发布的一部分）；除非用户明确要求"继续轮询"，否则不要在同一轮内反复调用本工具轮询。
6. `status` 含 `FAILED` 时，直接透传 `error_message` 给用户，并停止；不要自动重试发布、切换到其他子 skill 或再调 `release_hunyuan_training_checkpoint_as_model`。
7. `status=SUCCESS` 时，可引导用户使用 `model_id` 调用 `get_hunyuan_models_card_detail` 查看模型详情，或搜索该模型。
8. 本工具是**只读查询**，不做任何写操作；调用前无需向用户复述或确认。
