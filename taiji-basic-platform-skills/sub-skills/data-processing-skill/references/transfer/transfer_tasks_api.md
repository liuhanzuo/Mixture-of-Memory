## create_hunyuan_data_hdfs_to_ceph_transfer

**功能**：创建一个方向固定为 HDFS → Ceph 的数据搬运任务。后端复用太极搬运能力：校验目标 Ceph 目录存在且为空、校验当前用户对该目录的写权限，随后下发智研 UC 搬运任务并落库。


> ⚠️ **方向即工具，不传方向枚举**：搬运方向已由工具本身固定——本工具（HDFS→Ceph）与 `create_hunyuan_data_ceph_to_hdfs_transfer`（Ceph→HDFS）是两个独立工具，**没有也不接受 `type`/方向枚举参数**。根据用户"从哪搬到哪"选对工具即可。
> ⚠️ 当 `wsid` + `source_path`（`hdfs://` 开头）+ `target_path`（Ceph 目录）齐备时，必须立即通过 MCP 通道直接调用本工具。只有缺参数或 `wsid=0` 等非法值时才追问。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID。不能为 0。 |
| `source_path` | str | ✅ 必填 | - | 源 HDFS 路径，**必须以 `hdfs://` 开头**。 |
| `target_path` | str | ✅ 必填 | - | 目标 Ceph 目录路径。后端校验其存在、为空目录且当前用户有写权限。 |
| `name` | str | ❌ 可选 | 自动生成 | 任务名称。不传则后端按时间自动生成。 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_hdfs_to_ceph_transfer '{
  "wsid": 10086,
  "source_path": "hdfs://ss-teg-3-v2/data/tianqiong/xxx/parquet_dir",
  "target_path": "/apdcephfs_jn2/share_302316223/leslizhang/hdfs_dump/",
  "name": "把HDFS训练数据搬到南京ceph"
}'
```

### 返回字段说明

返回搬运任务详情（`TransferTaskInfo`），关键字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | **搬运任务 ID**，后续查询/轮询靠它 |
| `name` | str | 任务名称（可能自动生成） |
| `type` | str | 搬运方向，本工具固定 `HDFS_TO_CEPH` |
| `source_path` | str | 源路径。**非 owner/admin 时脱敏为 `***`** |
| `target_path` | str | 目标路径。**非 owner/admin 时脱敏为 `***`** |
| `status` | str | 状态，见下方"状态说明" |
| `message` | str | 失败原因（仅 `FAILED` 有值，成功为 `null`） |
| `uc_task_id` | int | 底层智研 UC 搬运任务 ID |
| `wedata_url` | str | 智研（WeData）任务页面链接，**原样透传，严禁自拼** |
| `details` | object | 底层 UC 任务原始详情（含各子任务 `state`/`url` 等），排障时可参考 |
| `creator` / `owner_list` | str / array | 创建人 / 负责人列表 |
| `created_at` / `updated_at` / `completed_at` | str | 时间线（规范化 snake_case，`yyyy-MM-dd HH:mm:ss`） |

---

## create_hunyuan_data_ceph_to_hdfs_transfer

**功能**：创建一个方向固定为 Ceph → HDFS 的数据搬运任务。后端复用太极搬运能力：校验源 Ceph 目录存在、为目录、且当前用户有读权限，随后下发智研 UC 搬运任务并落库。


> ⚠️ 当 `wsid` + `source_path`（Ceph 目录）+ `target_path`（`hdfs://` 开头）齐备时，必须立即通过 MCP 通道直接调用本工具。只有缺参数或 `wsid=0` 等非法值时才追问。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID。不能为 0。 |
| `source_path` | str | ✅ 必填 | - | 源 Ceph 目录路径。后端校验其存在、为目录且当前用户有读权限。 |
| `target_path` | str | ✅ 必填 | - | 目标 HDFS 路径，**必须以 `hdfs://` 开头**。 |
| `name` | str | ❌ 可选 | 自动生成 | 任务名称。不传则后端按时间自动生成。 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_ceph_to_hdfs_transfer '{
  "wsid": 10086,
  "source_path": "/apdcephfs_jn2/share_302316223/leslizhang/sft_out/",
  "target_path": "hdfs://ss-teg-3-v2/data/tianqiong/xxx/sft_backup",
  "name": "把南京ceph结果回传HDFS"
}'
```

### 返回字段说明

与 `create_hunyuan_data_hdfs_to_ceph_transfer` 相同（见上文），仅 `type` 固定为 `CEPH_TO_HDFS`。

---

## get_hunyuan_data_transfer_task

**功能**：按 id 查询单个搬运任务的最新状态与详情，用于轮询进度或定位失败原因。


> ⚠️ `task_id` 齐备时立即直接调用；查到 `FAILED` 后**主动透传** `message` 与 `wedata_url`（智研链接）帮用户定位。**严禁自拼 URL**。

### 状态说明

| 状态值 | Emoji | 说明 | 下一步建议 |
|--------|-------|------|-----------|
| `RUNNING` | 🔄 | 执行中 | 稍后再次调用本工具轮询 |
| `SUCCEEDED` | ✅ | **终态**：搬运成功 | 向用户展示目标路径 `target_path` |
| `FAILED` | ❌ | **终态**：搬运失败 | 透传 `message` + `wedata_url` 智研链接定位原因 |

> ℹ️ 实际状态枚举以后端返回为准（可能还有 `PENDING` 等中间态），未知状态原样透传给用户，不要臆造。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `task_id` | int | ✅ 必填 | - | 搬运任务 ID（来自 create / query 返回的 `id`）。 |
| `wsid` | int | ❌ 可选 | - | 工作空间 ID。传入则校验该任务归属于此工作空间。 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_transfer_task '{
  "task_id": 3001,
  "wsid": 10086
}'
```

### 返回字段说明

返回任务详情，字段与创建工具相同，额外含 `message`（失败原因，仅 `FAILED` 有值）。非任务负责人/管理员时 `source_path` / `target_path` 脱敏为 `***`。

---

## query_hunyuan_data_transfer_tasks

**功能**：分页查询工作空间下的数据搬运任务列表，支持按创建人、关键词过滤。


### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID。不能为 0。 |
| `creator` | str | ❌ 可选 | - | 创建人英文名（RTX），按创建人筛选。 |
| `keyword` | str | ❌ 可选 | - | 关键词，按任务名 / 负责人 / 源路径 / 目标路径模糊搜索。 |
| `page` | int | ❌ 可选 | `1` | 页码，**从 1 开始**。 |
| `page_size` | int | ❌ 可选 | `20` | 每页条数，上限 1000。 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_hunyuan_data_transfer_tasks '{
  "wsid": 10086,
  "keyword": "HDFS",
  "page": 1,
  "page_size": 20
}'
```

### 返回字段说明

返回分页结果 `{ items, page, page_size, total, has_more }`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `items` | array | 搬运任务数组，每项含 `id`、`name`、`type`、`source_path`、`target_path`、`status` 等 |
| `page` | int | 当前页码 |
| `page_size` | int | 每页条数 |
| `total` | int | 总条数 |
| `has_more` | bool | 是否还有下一页 |

> ⚠️ **展示顺序**：严格按 `items` 数组原始顺序展示（后端已按 id 降序），严禁重排、截断。

---

### 错误处理

> **返回码约定**：工具返回体为 `{"code":..., "message":..., "data":...}`。`code == 0` 表示成功；`code != 0`（如 `40001` / `50001`）为失败，`message` 为原因。以 `code` 为准，不要只看 HTTP 状态码。

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `wsid 不能为空` / `source_path 不能为空` / `target_path 不能为空` | 缺必填参数（40001） | 补齐后重试 |
| 目标 Ceph 目录不存在 / 非空 / 无写权限 | HDFS→Ceph：目标路径校验失败 | 换空目录或找路径 owner 授权 |
| 源 Ceph 目录不存在 / 不是目录 / 无读权限 | Ceph→HDFS：源路径校验失败 | 确认路径正确且有读权限 |
| `hdfs://` 端路径写反 | HDFS→Ceph 却把 hdfs 填到 target（或反之） | 按方向重新组织 source/target |
| `API 请求失败 (HTTP 401/403)` | Token 无效/无权限 | 见 `helper_api.md` 配置 Token |
| 只读用户禁止创建 | 当前用户为只读角色 | 联系管理员开通写权限 |
