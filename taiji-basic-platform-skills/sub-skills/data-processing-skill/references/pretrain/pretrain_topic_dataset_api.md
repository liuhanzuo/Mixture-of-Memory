## create_hunyuan_data_pretrain_topic

**功能**：创建预训练 Topic（主题）。**写操作**。


**层级模型**：`Workspace(wsid) -> PretrainTopic -> PretrainDataset`。一个工作空间下有多个 Topic（主题），一个 Topic 下挂多个数据集。**创建数据集前必须先有 Topic**（拿到 topic_id）。

> ⚠️ **预训练 vs 后训练工具选择（极易错，务必逐字对照）**：只要用户话里出现「**预训练 / pretrain**」，本文件的 **`*_pretrain_*`** 工具是**唯一正确选择**，**严禁**降级到后训练的 `create_hunyuan_data_topic` / `create_hunyuan_data_topic_dataset` / `query_hunyuan_data_topic_datas`（那几个是**后训练** posttrain 的工具，名字里**没有 `pretrain`**）。
> ⚠️ **创建类工具严格单次调用（防重复建资源，最高优先级）**：本工具是**写操作**，一次任务内对**同一个 Topic 只允许调用一次**。返回 `code=0` 且带回 `id` 即视为**创建成功、立即停止**——**严禁**重复调用创建工具（重复 create 会产生重复资源）。若需确认结果，应改调**只读**的 `query_hunyuan_data_pretrain_topics`，而不是再 create 一次。

### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `wsid` | int | ✅ | 工作空间 ID |
| `name` | str | ✅ | Topic 名称，**全局唯一**，重名报 `Topic名称已存在` |
| `source_id` | int | ❌ | 数据来源 ID |
| `source_name` | str | ❌ | 数据来源名称 |
| `owners` | list[str] | ❌ | 责任人英文名列表，不传默认当前用户 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_pretrain_topic '{
  "wsid": 10103,
  "name": "hy3-general-pretrain",
  "owners": ["your_rtx"]
}'
```

### 返回字段说明

返回 `data`：Topic 详情，含 `id`（后续建数据集用它当 `topic_id`）、`name`、`wsid`、`owners`、`creator`、`create_time` 等。

---

## query_hunyuan_data_pretrain_topics

**功能**：分页查询预训练 Topic 列表（按 name/owner 找 Topic 拿 `topic_id`）。**只读**。


### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `wsid` | int | ✅ | 工作空间 ID |
| `name` | str | ❌ | Topic 名称，模糊匹配 |
| `owner` | str | ❌ | 责任人 / 创建人，模糊匹配 |
| `page` | int | ❌ | 页码，1-based，默认 1 |
| `page_size` | int | ❌ | 每页数量，默认 20，最大 1000 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_hunyuan_data_pretrain_topics '{
  "wsid": 10103,
  "name": "general",
  "page": 1,
  "page_size": 20
}'
```

### 返回字段说明

返回分页结构 `{ items, page, page_size, total, has_more }`。

---

## create_hunyuan_data_pretrain_dataset

**功能**：在某 Topic 下创建预训练数据集。**写操作**。


> ⚠️ **创建数据集前必须先落实 Topic（强约束）**：用户给了明确的 `topic_id` → 直接用；只给 Topic 名称 → 先 `query_hunyuan_data_pretrain_topics` 拿 topic_id；要新建 → 先 `create_hunyuan_data_pretrain_topic` 拿到返回的 `id` 当 `topic_id`。**严禁凭空臆造 topic_id。**
> ⚠️ **未定 Topic 必须先追问、严禁擅自 create**：用户要登记数据集但**未给 `topic_id`、也未明确表示要新建 Topic** 时，只允许先 `query_hunyuan_data_pretrain_topics` 列出候选 Topic，然后**停下来向用户追问**该挂到哪个 Topic（或是否新建）；在拿到用户明确的归属/新建指令前，**严禁**调用本工具，也**严禁**自行 `create_hunyuan_data_pretrain_topic` 凭空造主题。
> ⚠️ **创建类工具严格单次调用（防重复建资源）**：本工具一次任务内对**同一个数据集只允许调用一次**。返回 `code=0` 且带回 `id` 即视为创建成功、立即停止。需确认结果改调只读的 `get_hunyuan_data_pretrain_dataset`，不要再 create。
> 🛑 **权限/参数错误立即停止**：创建失败（权限不足、路径无授权、HTTP 400/500 等）时，**立即停止**并告知错误原因，**不得**换 `type` / `task_type` / `storage_path` 等参数反复重试。

### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `wsid` | int | ✅ | 工作空间 ID |
| `key` | str | ✅ | 数据集全局唯一标识，1-128 位，仅 `字母/数字/-/_` |
| `topic_id` | int | ✅ | 所属 Topic ID（取自 `create_hunyuan_data_pretrain_topic` / `query_hunyuan_data_pretrain_topics`） |
| `storage_path` | str | ✅ | 数据存储路径（CEPH 目录） |
| `stage_list` | list[str] | ✅ | 数据阶段枚举列表，至少 1 个，取值见下方「枚举表」 |
| `type` | str | ✅ | 数据类型枚举，取值见下方「枚举表」 |
| `name` | str | ❌ | 数据集名称，不传默认用 `key` |
| `desc` | str | ❌ | 描述 |
| `category` | str | ❌ | 分类 |
| `app_group` | str | ❌ | 应用组（parquet/HDFS 场景） |
| `location` | str | ❌ | 地域，如 `nj` |
| `hdfs_path` | str | ❌ | HDFS 源路径（parquet 场景） |
| `wedata_app_group` | str | ⚠️ parquet 必填 | wedata 应用组（HDFS parquet → Iceberg 入库通道）；json 模式忽略 |
| `wedata_cluster` | str | ⚠️ parquet 必填 | wedata 集群；json 模式忽略 |
| `task_type` | str | ❌ | 搬运/接入类型，默认 `CEPH`（取值须与后端 `TransferTaskType` 一致） |
| `file_data_type` | str | ❌ | 文件数据类型，默认 `JSON`；`PARQUET`/`JSON`/`ICEBERG` |
| `owners` | list[str] | ❌ | 责任人英文名列表，不传默认当前用户 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_pretrain_dataset '{
  "wsid": 10103,
  "key": "hy3-general-cleaned-v1",
  "name": "hy3 通用清洗数据 v1",
  "topic_id": 8001,
  "stage_list": ["Pretrain"],
  "type": "CLEANED_DATA",
  "file_data_type": "JSON",
  "storage_path": "/apdcephfs_nj7/share_xxx/pretrain/general/cleaned_v1/",
  "location": "nj",
  "owners": ["your_rtx"]
}'
```

### 返回字段说明

返回 `data`：数据集详情，含 `id`、`key`、`name`、`topic_id`、`owning_dataset`、`stage_list`、`type`、`file_data_type`、`storage_path`、`status` 等。

---

## get_hunyuan_data_pretrain_dataset

**功能**：查询单个预训练数据集详情。**只读**。


### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `dataset_id` | int | ✅ | 预训练数据集 ID |
| `wsid` | int | ❌ | 传入则校验归属该工作空间 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_pretrain_dataset '{
  "dataset_id": 9001,
  "wsid": 10103
}'
```

---

## query_hunyuan_data_pretrain_datasets

**功能**：分页查询预训练数据集列表。**只读**。


### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `wsid` | int | ✅ | 工作空间 ID |
| `key` | str | ❌ | 数据集标识，精确匹配 |
| `owner` | str | ❌ | 责任人，模糊匹配 |
| `creator` | str | ❌ | 创建人，模糊匹配 |
| `topic_id` | int | ❌ | 所属 Topic 过滤 |
| `stage` | str | ❌ | 数据阶段枚举过滤 |
| `file_type` | str | ❌ | 文件数据类型过滤（`PARQUET`/`JSON`/`ICEBERG`） |
| `page` | int | ❌ | 页码，1-based，默认 1 |
| `page_size` | int | ❌ | 每页数量，默认 20，最大 1000 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_hunyuan_data_pretrain_datasets '{
  "wsid": 10103,
  "topic_id": 8001,
  "page": 1,
  "page_size": 20
}'
```

---

### 枚举表（严格取值，不可臆造）

### `stage_list` / `stage`（数据阶段，`DataStage`）

| 枚举名 | 中文 | 说明 |
|--------|------|------|
| `STAGE_ONE` | 一阶段 | |
| `ANNEALING` | 退火 | |
| `STAGE_TWO` | 二阶段 | |
| `LONG_CONTEXT` | 长文 | |
| `Pretrain` | 预训练 | 新阶段，含一阶段 |
| `Midtrain` | Midtrain | 新阶段，含退火 + 二阶段 + 长文 |

> `stage_list` 是**数组**：`["Pretrain"]` 或 `["STAGE_TWO", "LONG_CONTEXT"]`。

### `type`（数据类型，`DataType`）

| 枚举名 | 中文 |
|--------|------|
| `HUNYUAN_AGENT` | 混元Agent |
| `HUNYUAN_AGENT_GRAPH_RAG` | 混元Agent-GraphRAG |
| `RAW_BUSINESS_DATA` | 业务原始数据 |
| `CLEANED_DATA` | 清洗后数据 |
| `SAMPLED_DATA` | 抽样后数据 |
| `TRAINING_DATA` | 训练数据 |
| `EVALUATION_DATA` | 评测数据 |
| `MODEL_DATA` | 模型数据 |
| `PROMPT_DATA` | prompt数据 |
| `RAW_ANNOTATION_DATA` | 标注原始数据 |
| `ANNOTATION_RESULT` | 标注结果数据 |
| `INDEX_DATA` | 索引数据 |
| `TEST_DATA` | 测试数据 |
| `RATIO_DATA` | 配比数据 |
| `INTEGRATION_DATA` | 融合数据 |

### `file_data_type` / `file_type`（文件数据类型，`FileDataType`）

| 枚举名 | 说明 |
|--------|------|
| `JSON` | ceph 上的 json（默认） |
| `PARQUET` | parquet（配合 app_group / hdfs_path） |
| `ICEBERG` | iceberg 表 |

---

### 常见错误

| 报错（message） | 原因 | 处理 |
|------|------|------|
| `Topic名称已存在：xxx` | 同名 Topic 已存在 | 用 `query_hunyuan_data_pretrain_topics` 查回 id 复用，不再新建 |
| `Topic不存在：id=xxx` | `topic_id` 无效 | 核对 topic_id，或先建 Topic |
| `数据集不存在：id=xxx` | `dataset_id` 无效 | 核对 dataset_id |
| `key（数据集全局唯一标识）不能为空` | 漏传 key | 补 key（1-128 位字母数字-_） |
| `xxx is an invalid DataStage` / `DataType` | 枚举值拼错 | 对照上方枚举表取正确英文枚举名 |
| `wsid 不能为空` | 漏传 wsid | 补 wsid，或按 `helper_api.md` 查工作空间 |
| 认证失败（401/403） | Token 未配置 / 过期 / 无权限 | 见 `helper_api.md` 的「获取 API Token」段 |
