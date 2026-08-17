## create_hunyuan_data_merge_bins_task

**功能**：把 Ceph 上 `input_path` 目录下的一批 bin/idx 分片文件合并为少量大文件。后端复用太极合并能力：校验源目录存在/为目录/有读权限、校验 `input_path` 地域与计算资源地域一致，随后下发智研 UC 合并任务并落库；同地域直写用户目录，跨地域先写平台路径再自动同步。


> ⚠️ **何时进入本域（强判别）**：用户**手里已经有 bin/idx 分片**、只想合并减少文件数 → 用本文件工具；用户还在 parquet/json/jsonl 阶段要**生成 bin** → 那是转 bin，回 `pretrain_parquet2bin_api.md` / `sft_conversion_api.md`，别混。
> ⚠️ **仅支持 Ceph→Ceph**：输入/输出都在 Ceph（无 HDFS/parquet 分支）。若用户的分片在 HDFS，先引导用 `transfer_tasks_api.md` 搬到 Ceph 再合并。
> ⚠️ 当 `wsid` + `input_path` + `task_config.resource_config`(`resource_id` + `resource_location`) 齐备时，立即通过 MCP 通道直接调用本工具。缺资源时可先列计算资源再建；`input_path` 地域与资源地域不一致时先纠正再提交。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID。不能为 0。 |
| `input_path` | str | ✅ 必填 | - | 待合并的 bin/idx 分片所在 **Ceph 目录**。后端校验其存在、为目录、当前用户有读权限；其地域须与 `resource_location` 一致。 |
| `task_config` | object | ✅ 必填 | - | 任务配置，通过 `resource_config` 承载计算资源，见下方「task_config 结构」。 |
| `output_path` | str | ❌ 可选 | 自动生成 | 合并产物输出目录。不传则后端在 `input_path` 同级生成 `xxx_merged_<时间戳>`；若指定且已存在须为空目录且有写权限。 |
| `merge_bins_count` | int | ❌ 可选 | UC 兜底 | 期望合并后的产物 bin 文件数。不传则由 UC 侧按默认策略决定。 |
| `type` | str | ❌ 可选 | `MERGE_BINS` | 任务类型枚举，手动合并用默认 `MERGE_BINS` 即可，见下方「type 枚举取值」。 |
| `name` | str | ❌ 可选 | `create_by_skill_<YYYYMMDD-HHMM>` | 任务名称。**用户没给名字时，Agent 必须用当前时间生成 `create_by_skill_<YYYYMMDD-HHMM>` 并显式传入**（如 `create_by_skill_20260803-1530`） |
| `owners` | array[str] | ❌ 可选 | 当前用户 | 负责人英文名列表。不传则默认加入创建人。 |

### task_config 结构（嵌套对象）

```
task_config:
  resource_config:
    resource_id:        # str 必填，计算资源 ID（取自 query_hunyuan_data_compute_resources）
    resource_location:  # str 必填，计算资源地域（如 nj/sh/gz），须与 input_path 地域一致
    resource_alias:     # str 可选，计算资源别名
```

### type 枚举取值

| 枚举值 | 含义 |
|--------|------|
| `MERGE_BINS` | 通用 bin 分片合并（**手动合并默认用这个**） |
| `PRETRAIN_CONVERSION` | 由预训练转 BIN 记录触发的合并 |
| `PRETRAIN_SHUTTLE_TASK` | 由预训练融合任务触发的合并 |
| `PRETRAIN_TOPIC_EXPERIMENT` | 由预训练消融实验触发的合并 |

### 任务名兜底规则（`name` 未给时必须自动生成）

- 用户**没有指定任务名称**时，Agent **必须**用**当前时间**生成 `name = create_by_skill_<YYYYMMDD-HHMM>` 并**显式传入** `create_hunyuan_data_merge_bins_task`，如 `create_by_skill_20260803-1530`。
- **格式硬约束**：前缀固定 `create_by_skill_`；时间戳为 `YYYYMMDD-HHMM`（年 4 位 + 月 2 位 + 日 2 位 + `-` + 时 2 位 + 分 2 位，**月/日/时/分不足两位补 0**，24 小时制，本地时间）。
- **必须是真实当前时间**：严禁把 `<YYYYMMDD-HHMM>`、`{YYYYMMDD-HHMM}` 之类**占位符原样提交**，严禁硬编码或复用历史时间戳。
- **用户给了名字就用用户的**：只有用户未提供任何任务名时才兜底生成；用户显式指定时原样使用，不要覆盖、不要追加时间戳。
- **不要为了名字追问用户**：`name` 缺失时静默兜底即可，**严禁**因缺 `name` 而中断流程去问用户。
- **建完回显**：把实际使用的 `name` 一并反馈给用户（与 `output_path` 一起）。
- ⚠️ **与 `output_path` 的区别**：`output_path` 用户没给就**不传**（由后端自动生成 `xxx_merged_<时间戳>`）；`name` 用户没给则由 **Agent 侧兜底生成并传入**。两者兜底方不同，别混。

### 地域一致性硬约束

`input_path` 所在地域**必须**与计算资源 `task_config.resource_config.resource_location` 一致，否则后端直接校验报错。建任务前若用户只给了路径没给资源，或路径地域与资源地域明显不符，**先追问 / 提示纠正**，不要盲目提交。

**缺 `resource_id` / `resource_location` 时**：可先调用 `query_hunyuan_data_compute_resources`（见 `pretrain_parquet2bin_api.md`），**务必传 `entity_type=PRETRAIN_DATA_CONVERSION`**（否则只返回南京资源，非南京地域的 bin 分片将查不到可用资源），取一条与 `input_path` 同地域的 `resource_id` 及其 `location`，再建合并任务；不要臆造资源 id。

### 调用示例

**示例 A（用户显式指定了任务名）**：

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_merge_bins_task '{
  "wsid": 10103,
  "input_path": "/apdcephfs_nj7/share_300377003/leslizhang/bin_shards/",
  "merge_bins_count": 8,
  "task_config": {
    "resource_config": {
      "resource_id": "res-abc123",
      "resource_location": "nj"
    }
  },
  "name": "合并训练bin分片"
}'
```

**示例 B（用户没给任务名 → 按规则用当前时间兜底；下例的 `20260803-1530` 须换成真实当前时间）**：

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_merge_bins_task '{
  "wsid": 10103,
  "input_path": "/apdcephfs_nj7/share_300377003/leslizhang/bin_shards/",
  "merge_bins_count": 8,
  "task_config": {
    "resource_config": {
      "resource_id": "res-abc123",
      "resource_location": "nj"
    }
  },
  "name": "create_by_skill_20260803-1530"
}'
```

### 返回字段说明

返回合并任务详情（`MergeBinsTaskInfo`），关键字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | **合并任务 ID**，后续查询/轮询/重试靠它 |
| `name` | str | 任务名称（入参回显；Agent 未传时为后端按时间生成的名字） |
| `type` | str | 任务类型，见上文「type 枚举取值」 |
| `input_path` | str | 源分片目录。**非 owner/admin 时脱敏为 `***`** |
| `output_path` | str | 合并产物输出目录（若入参未传，这里是自动生成的实际路径）。**非 owner/admin 时脱敏为 `***`** |
| `platform_path` | str | 跨地域时 UC 真实写入的平台路径；同地域为 `null` |
| `merge_bins_count` | int | 期望产物文件数（入参回显） |
| `status` | str | 主合并任务状态，见下方"状态说明" |
| `message` | str | 失败原因（仅 `FAILED` 有值） |
| `uc_task_id` | int | 底层智研 UC 任务 ID |
| `wedata_url` | str | 智研任务页面链接，**原样透传，严禁自拼** |
| `transfer_task_id` | int | 跨地域同步 export 任务 ID；同地域为 `null` |
| `transfer_status` | str | 跨地域同步阶段状态；同地域为 `null` |
| `creator` / `owner_list` | str / array | 创建人 / 负责人列表 |
| `created_at` / `updated_at` / `completed_at` | str | 时间线（规范化 snake_case，`yyyy-MM-dd HH:mm:ss`） |

### 同地域 vs 跨地域

当 `input_path`/`output_path` 地域与计算资源地域一致时，UC 直接把结果写用户目录，`platform_path` 为空；当输出目录与计算资源不在同一地域时，后端先把结果写到平台路径（`platform_path`），主合并任务成功后**自动**创建一个 export 任务把结果同步到用户的 `output_path`，此阶段状态见 `transfer_status` / `transfer_task_id`。

---

## get_hunyuan_data_merge_bins_task

**功能**：按 id 查询单个合并任务的最新状态与详情，用于轮询进度或定位失败原因。


> ⚠️ `task_id` 齐备时立即直接调用；查到 `FAILED` 后**主动透传** `message` 与 `wedata_url`（智研链接）帮用户定位。跨地域任务额外看 `transfer_status`。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `task_id` | int | ✅ 必填 | - | 合并任务 ID（来自 create / query 返回的 `id`）。 |
| `wsid` | int | ❌ 可选 | - | 工作空间 ID。传入则校验该任务归属于此工作空间。 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_merge_bins_task '{
  "task_id": 5001,
  "wsid": 10103
}'
```

### 返回字段说明

返回任务详情，字段与创建工具相同。非任务负责人/管理员时 `input_path` / `output_path` 脱敏为 `***`。

---

## query_hunyuan_data_merge_bins_tasks

**功能**：分页查询工作空间下的 Bin 分片合并任务列表，支持按创建人、关键词、任务类型过滤。


### 状态说明

| 状态值 | Emoji | 说明 | 下一步建议 |
|--------|-------|------|-----------|
| `PENDING` | ⏳ | 待提交 / 排队中 | 稍后再次调用 `get` 轮询 |
| `RUNNING` | 🔄 | 执行中 | 稍后再次调用 `get` 轮询 |
| `SUCCEEDED` | ✅ | **终态**：合并成功 | 向用户展示 `output_path`；跨地域还需确认 `transfer_status` |
| `FAILED` | ❌ | **终态**：合并失败 | 透传 `message` + `wedata_url` 智研链接定位原因 |
| `STOPPED` | ⏹️ | **终态**：已停止 | 原样告知用户 |

> ℹ️ 实际状态枚举以后端返回为准，未知状态原样透传给用户，不要臆造。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID。不能为 0。 |
| `creator` | str | ❌ 可选 | - | 创建人英文名（RTX），按创建人筛选。 |
| `keyword` | str | ❌ 可选 | - | 关键词，按任务名 / 负责人 / 输入路径 / 输出路径模糊搜索。 |
| `type` | str | ❌ 可选 | - | 任务类型筛选，取值见上文「type 枚举取值」。 |
| `page` | int | ❌ 可选 | `1` | 页码，**从 1 开始**。 |
| `page_size` | int | ❌ 可选 | `20` | 每页条数，上限 1000。 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_hunyuan_data_merge_bins_tasks '{
  "wsid": 10103,
  "creator": "v_shguan",
  "page": 1,
  "page_size": 20
}'
```

### 返回字段说明

返回分页结果 `{ items, page, page_size, total, has_more }`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `items` | array | 合并任务数组，每项含 `id`、`name`、`type`、`input_path`、`output_path`、`status` 等 |
| `page` | int | 当前页码 |
| `page_size` | int | 每页条数 |
| `total` | int | 总条数 |
| `has_more` | bool | 是否还有下一页 |

> ⚠️ **展示顺序**：严格按 `items` 数组原始顺序展示（后端已按 id 降序），严禁重排、截断。

---

## retry_hunyuan_data_merge_bins_task

**功能**：重试一个失败的合并任务。后端按任务当前阶段自动分流，无需调用方判断。


> ⚠️ **仅在用户显式要求重试时调用**，不要在查到 FAILED 后自动重试。重试分流由后端决定：
> - 未启动过（`uc_task_id` 为空）→ 重置为 `PENDING` 等下一轮定时器提交；
> - 主合并任务已成功但**跨地域 export 同步阶段失败** → 重置 `transfer_status` 由定时器重建同步任务；
> - 主任务已启动过 → 重跑同一 UC 任务。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `task_id` | int | ✅ 必填 | - | 需要重试的合并任务 ID。 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call retry_hunyuan_data_merge_bins_task '{
  "task_id": 5001
}'
```

### 返回字段说明

返回重试后的任务详情，字段同创建工具（`status` 通常回到 `PENDING`/`RUNNING`）。

---

### 错误处理

> **返回码约定**：工具返回体为 `{"code":..., "message":..., "data":...}`。`code == 0` 表示成功；`code != 0`（如 `40001` / `50001`）为失败，`message` 为原因。以 `code` 为准，不要只看 HTTP 状态码。

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `wsid 不能为空` / `input_path 不能为空` | 缺必填参数（40001） | 补齐后重试 |
| `task_config.resource_config 的 resource_id 与 resource_location 不能为空` | 未传计算资源（40001） | 先 `query_hunyuan_data_compute_resources`（传 `entity_type=PRETRAIN_DATA_CONVERSION`）取资源再建 |
| input_path 地域与计算资源地域不一致 | 数据须就近处理，地域强校验失败 | 确认 `input_path` 与 `resource_location` 同地域，或换同地域资源；查资源时记得传 `entity_type=PRETRAIN_DATA_CONVERSION` 才能列出非南京地域资源 |
| 源目录不存在 / 不是目录 / 无读权限 | `input_path` 校验失败 | 确认路径正确、是 bin 分片目录且有读权限 |
| 输出目录已存在且非空 / 无写权限 | `output_path` 校验失败 | 换空目录、留空让后端自动生成，或找 owner 授权 |
| `API 请求失败 (HTTP 401/403)` | Token 无效/无权限 | 见 `helper_api.md` 配置 Token |
| 只读用户禁止创建 | 当前用户为只读角色 | 联系管理员开通写权限 |
