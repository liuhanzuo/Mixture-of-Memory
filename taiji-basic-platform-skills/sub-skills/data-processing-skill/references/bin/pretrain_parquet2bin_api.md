## query_hunyuan_data_app_groups

**功能**：列出预训练转 bin 可用的应用组（parquet 模式第 1 步）。

**用途场景**：parquet 模式（HDFS 上的 parquet）转 bin 前，先选应用组。json 模式（CEPH json）不需要。

### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `wsid` | int | ✅ | 工作空间 ID |

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_hunyuan_data_app_groups '{"wsid": 10103}'
```

### 单工具 SOP

- 用户只问"预训练转 bin 有哪些应用组可选"时，直接只调一次本工具并回答，不重复调用。
- 返回列表让用户选一个作为 `app_group` 传给 `create_hunyuan_data_pretrain_conversion`（parquet 模式必填）。

---

## list_hunyuan_data_app_group_hdfs_quota

**功能**：列出某应用组下的 HDFS 集群/配额（parquet 模式第 2 步，取 hdfsNn）。

**用途场景**：parquet 模式（HDFS 上的 parquet）转 bin 前，在某应用组下选 HDFS 集群。

### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `app_group_name` | str | ✅ | 应用组名称（取自 `query_hunyuan_data_app_groups`） |

### 调用示例

```bash
python3 scripts/connect_mcp.py call list_hunyuan_data_app_group_hdfs_quota '{"app_group_name": "TaiJi_HYAide_xxx"}'
```

### 单工具 SOP

- 用户只问"某应用组有哪些 HDFS 集群/配额"时，直接只调一次本工具并回答。
- 从返回里取 `hdfsNn` 作为 `hdfs_cluster` 传给 `create_hunyuan_data_pretrain_conversion`（parquet 模式必填）。

---

## query_hunyuan_data_compute_resources

**功能**：列出平台公共计算资源（转 bin 第 3 步，取 `resource_id` + `resource_location`）。

**用途场景**：创建转 bin / 合并 bin 任务前取计算资源。merge_bins 建任务时也复用本工具。

### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `entity_type` | str | ✅ | **务必传 `PRETRAIN_DATA_CONVERSION`**。不传时后端**只返回南京(nj)资源**；传该值才能按预训练转 bin 业务允许的地域集合返回资源（覆盖非南京地域） |
| `location` | str | ❌ | 地域过滤（可选） |
| `resource_group_id` | str | ❌ | 资源组过滤（可选） |
| `page` | int | ❌ | 页码 |
| `page_size` | int | ❌ | 每页条数 |

> ⚠️ 本工具**无 `wsid` 参数**，不要传 `wsid`。

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_hunyuan_data_compute_resources '{"entity_type": "PRETRAIN_DATA_CONVERSION"}'
```

### 单工具 SOP

- 用户只问"平台公共计算资源/预训练转 bin 计算资源"时，直接只调一次本工具并回答。
- 从返回取 `resource_id` + 同行 `location`（地域），传给创建工具；若只查到南京资源，多半是漏传 `entity_type`。

---

## list_hunyuan_data_pretrain_operators

**功能**：列出转 bin 算子候选（转 bin 第 4 步，`operator` 必填）。

### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `wsid` | int | ✅ | 工作空间 ID |

### 调用示例

```bash
python3 scripts/connect_mcp.py call list_hunyuan_data_pretrain_operators '{"wsid": 10103}'
```

### 返回说明

返回算子候选清单（如 `parquet2bin_v2`）。创建任务时 `operator` 字段从中取值，**前端默认选第一个**。

### 单工具 SOP

- 用户已明确指定 `operator`（如 `master_52b16f7a`）时直接使用该值，**不要**再调用本工具。
- 只有用户未给 operator 时才用本工具列出候选，一般建议取第一个。

---

## list_hunyuan_data_pretrain_tokenizer_branches

**功能**：列出 Tokenizer 代码分支（三级级联第 1 级）。

### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `wsid` | int | ✅ | 工作空间 ID |

### 调用示例

```bash
python3 scripts/connect_mcp.py call list_hunyuan_data_pretrain_tokenizer_branches '{"wsid": 10103}'
```

### 单工具 SOP

- 三级级联第 1 级，返回分支列表（前端默认 `master`）。
- 选出的分支作为 `git_based_tokenizer.branch` 传给创建工具。

---

## list_hunyuan_data_pretrain_tokenizer_model_series

**功能**：列出 Tokenizer 模型系列（三级级联第 2 级）。

### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `wsid` | int | ✅ | 工作空间 ID |
| `branch` | str | ✅ | 代码分支（取自第 1 级） |

### 调用示例

```bash
python3 scripts/connect_mcp.py call list_hunyuan_data_pretrain_tokenizer_model_series '{"wsid": 10103, "branch": "master"}'
```

### 单工具 SOP

- 三级级联第 2 级，返回模型系列列表（前端默认 `hy_3`）。
- 选出的模型系列作为 `git_based_tokenizer.model_series` 传给创建工具。

---

## list_hunyuan_data_pretrain_tokenizer_train_stages

**功能**：列出 Tokenizer 训练阶段（三级级联第 3 级）。

### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `wsid` | int | ✅ | 工作空间 ID |
| `branch` | str | ✅ | 代码分支（取自第 1 级） |
| `model_series` | str | ✅ | 模型系列（取自第 2 级） |

### 调用示例

```bash
python3 scripts/connect_mcp.py call list_hunyuan_data_pretrain_tokenizer_train_stages '{"wsid": 10103, "branch": "master", "model_series": "hy_3"}'
```

### 单工具 SOP

- 三级级联第 3 级，返回训练阶段列表（前端默认 `pretrain`）。
- 选出的训练阶段作为 `git_based_tokenizer.train_stage` 传给创建工具。
- 三段 `{branch, model_series, train_stage}` **必须全部齐备**，缺任意一段后端启动转 bin 时报"git tokenizer 选择不完整"。

---

## create_hunyuan_data_pretrain_conversion

**功能**：创建预训练数据转 bin 任务（把预训练数据转换为 bin 文件，输出到 CEPH）。通过 `file_format` 区分两种模式：`parquet`（默认，HDFS 上的 parquet）/ `json`（CEPH 上的 json）。底层按 `file_format` 自动设 `storageType`（parquet→HDFS、json→CEPH），调用方无需关心 storageType。


> ⚠️ 所有必填齐备且已完成必要查询后，立即直接调用本工具，不要把命令行粘给用户。若用户已显式给出 `app_group`、`operator`、`git_based_tokenizer.branch/model_series/train_stage` 等字段，直接复用用户值，**不要**重复调用前置列举工具。
> ⚠️ 本工具同一组参数只调用一次：若返回权限/路径等业务错误，原样转达并停止，**不要**用相同参数重复创建。

### 参数表

| 参数（`*`） | 类型 | 必填 | 默认值 | 说明 |
|------|------|:---:|--------|------|
| `wsid` | int | ✅ | - | 工作空间 ID |
| `file_format` | str | ❌ | `parquet` | 转 bin 模式：`parquet`（HDFS parquet 转 bin）/ `json`（CEPH json 转 bin）。**大小写均可**（parquet/PARQUET/json/JSON 都接受） |
| `input_path` | str | ✅ | - | 输入路径。**parquet**：HDFS 路径（从 `/data` 开头，需是所选 hdfs 集群下的目录）；**json**：CEPH json 目录 |
| `output_path` | str | ✅ | - | CEPH 输出目录，需空目录且**当前用户有写权限**。⚠️ 路径前缀须为后端可识别的地域/容器；**以后端校验为准**——若后端返回「未找到Ceph路径对应的地域」，再按报错列出的「可用」前缀改选即可 |
| `app_group` | str | 🔶 | - | 应用组（来自 `query_hunyuan_data_app_groups`）。**仅 parquet 模式必填**；json 模式不传 |
| `hdfs_cluster` | str | 🔶 | - | HDFS 集群，取 hdfsNn（来自 `list_hunyuan_data_app_group_hdfs_quota`）。**仅 parquet 模式必填**；json 模式不传 |
| `resource_id` | str | ✅ | - | 平台公共计算资源（来自 `query_hunyuan_data_compute_resources`） |
| `resource_location` | str | ❌ | `nj` | 计算资源地域。取计算资源同行「地域」；不传默认 nj |
| `operator` | str | ✅ | - | 转 bin 算子（来自 `list_hunyuan_data_pretrain_operators`，前端默认第一个） |
| `git_based_tokenizer` | obj | ✅ | - | Tokenizer 三级级联 `{branch, model_series, train_stage}`（来自三级列举工具） |
| `seq_len` | int | ❌ | `-1` | 序列长度，如 4096/8192/32768/262144；不填提交 -1（视为未指定） |
| `packing_strategy` | str | ❌ | `CHUNK_LONG_SEQ` | 长文本配置枚举：SKIP_LONG_SEQ / CHUNK_LONG_SEQ / BESTFIT。seq_len 为空/-1 时强制 CHUNK_LONG_SEQ |
| `add_source_id` | bool | ❌ | `false` | 是否添加 SourceID |
| `source_id_offset` | int | ❌ | `150000` | SourceID 启用时的 offset（仅 add_source_id=true 生效） |
| `add_loss_mask` | bool | ❌ | `false` | 是否添加 Loss mask（要求原始数据含 loss mask 字段） |
| `loss_mask_offset` | int | ❌ | `300000` | Loss mask 启用时的 offset（仅 add_loss_mask=true 生效） |
| `partition_num` | int | ❌ | `1024` | 分区数（1~2048），转 bin 并发度，也是最终小 bin 文件数量 |
| `enable_merge_bins` | bool | ❌ | `false` | 是否在转 bin 之后**顺带合并 bin 分片**：把本次产物里的大量碎 bin/idx（数量≈`partition_num`）合并成少量大文件。链路变为「主转 bin → merge_bins → 同步」，merge 成功前不触发同步；merge 失败**不回退主转 bin 状态**，可 retry 重跑 |
| `merge_bins_count` | int | ❌ | 由后端算子兜底 | 合并后的目标 bin 文件个数，**必须 > 0**（传 0/负数报 400）；不传则由后端算子按默认策略决定。**仅 `enable_merge_bins=true` 时生效** |
| `name` | str | ❌ | `create_by_skill_<YYYYMMDD-HHMM>` | 任务名。**用户没给名字时，Agent 必须用当前时间生成 `create_by_skill_<YYYYMMDD-HHMM>` 并显式传入**（如 `create_by_skill_20260803-1530`） |
| `owners` | list[str] | ❌ | 当前用户 | 责任人 RTX 列表 |

> 🔶 = 条件必填（仅 parquet 模式）。

### 两种模式差异（`file_format`）

| 差异点 | parquet 模式 | json 模式 |
|--------|-------------|-----------|
| 输入路径类型 | HDFS parquet（从 `/data` 开头） | CEPH json 目录 |
| `app_group` + `hdfs_cluster` | **必填** | 不需要 |
| 前置步骤 | 需先 ①② 查应用组、HDFS 集群 | 从 ③ 计算资源开始 |
| `file_format` | `parquet`（默认） | `json` |

> 其余字段（operator / git_based_tokenizer 三级 / resource / seq_len / packing_strategy / partition_num / output_path）**完全一致**。用户没说清是 parquet 还是 json 时，先问一句"输入是 HDFS 上的 parquet，还是 CEPH 上的 json？"。

### 任务名兜底规则（`name` 未给时必须自动生成）

- 用户**没有指定任务名称**时，Agent **必须**用**当前时间**生成 `name = create_by_skill_<YYYYMMDD-HHMM>` 并**显式传入**，如 `create_by_skill_20260803-1530`。
- **格式硬约束**：前缀固定 `create_by_skill_`；时间戳为 `YYYYMMDD-HHMM`（年 4 位 + 月 2 位 + 日 2 位 + `-` + 时 2 位 + 分 2 位，**月/日/时/分不足两位补 0**，24 小时制，本地时间）。
- **必须是真实当前时间**：严禁把 `<YYYYMMDD-HHMM>`、`{YYYYMMDD-HHMM}` 之类**占位符原样提交**，严禁硬编码或复用历史时间戳。
- **用户给了名字就用用户的**：只有用户未提供任何任务名时才兜底生成；用户显式指定时原样使用，不要覆盖、不要追加时间戳。
- **不要为了名字追问用户**：`name` 缺失时静默兜底即可，**严禁**因缺 `name` 而中断流程去问用户。
- **建完回显**：把实际使用的 `name` 一并反馈给用户。

### enable_merge_bins（转 bin 顺带合并分片）规则

- **用户手里还是 parquet/json，要转 bin 且嫌产物太碎** → 本工具带 `enable_merge_bins=true`（可选 `merge_bins_count=N`），**一个任务搞定转 bin + 合并**。
- **用户手里已经是 bin/idx 分片、只想合并** → 那是另一条链路，走 `create_hunyuan_data_merge_bins_task`（见 `merge_bins_tasks_api.md`），**不要**在这里空转一次 bin。
- 只有用户明确表达「合并 / 别太碎 / 文件数太多 / 合成几个大文件」时才开启；**没提就不要自作主张传 `true`**（默认 false）。
- `merge_bins_count` 必须 > 0，传 0/负数后端直接报 400；不确定就**不传**，让后端算子按默认策略决定。
- 开启后用 `get_hunyuan_data_pretrain_conversion` 轮询时，除主 `status` 外还要看 `merge_bins_status`（合并阶段状态）、`merge_bins_output_path`（合并产物路径，成功后才回填）；失败时透传 `merge_bins_url`（**严禁自拼 URL**）。**主转 bin 成功 ≠ 全部完成**，合并阶段成功前不会触发跨地域同步；合并失败不回退主状态，可 `retry_hunyuan_data_pretrain_conversion` 重跑。

### 默认值回显

创建工具会兜底 `packing_strategy`(默认 CHUNK_LONG_SEQ) / `add_source_id`(默认 false) / `add_loss_mask`(默认 false) / `partition_num`(默认 1024) / `resource_location`(默认 nj)；`source_id_offset`(启用 SourceID 时默认 150000) / `loss_mask_offset`(启用 Loss mask 时默认 300000)。返回里若有 `applied_defaults` 提示，**必须原样转告用户**。

### 调用示例

**调用示例 A（parquet 转 bin，最小必填）**：

> ℹ️ `name` 用户没给时按规则自动生成 `create_by_skill_<YYYYMMDD-HHMM>`（下例的 `20260803-1530` 须换成**真实当前时间**）。

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_pretrain_conversion '{
  "wsid": 10103,
  "file_format": "parquet",
  "app_group": "TaiJi_HYAide_xxx",
  "hdfs_cluster": "hdfs://nn-xxx",
  "resource_id": "g_teg_hunyuan_yizhanshi_datax.16421",
  "resource_location": "nj",
  "operator": "parquet2bin_v2",
  "git_based_tokenizer": {"branch": "master", "model_series": "hy_3", "train_stage": "pretrain"},
  "input_path": "/data/tianqiong/TEG/.../parquet_dir",
  "output_path": "/apdcephfs_xxx/share_yyy/bin_out",
  "seq_len": 32768,
  "name": "create_by_skill_20260803-1530"
}'
```

**调用示例 B（json 转 bin，无需 app_group / hdfs_cluster）**：

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_pretrain_conversion '{
  "wsid": 10103,
  "file_format": "json",
  "resource_id": "g_teg_hunyuan_yizhanshi_datax.16421",
  "resource_location": "nj",
  "operator": "parquet2bin_v2",
  "git_based_tokenizer": {"branch": "master", "model_series": "hy_3", "train_stage": "pretrain"},
  "input_path": "/apdcephfs_sh7/share_123456/json_dir",
  "output_path": "/apdcephfs_xxx/share_yyy/bin_out",
  "seq_len": 32768,
  "name": "create_by_skill_20260803-1530"
}'
```

**调用示例 C（parquet + 启用 SourceID + Loss mask）**：

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_pretrain_conversion '{
  "wsid": 10103,
  "app_group": "TaiJi_HYAide_xxx",
  "hdfs_cluster": "hdfs://nn-xxx",
  "resource_id": "g_teg_hunyuan_yizhanshi_datax.16421",
  "operator": "parquet2bin_v2",
  "git_based_tokenizer": {"branch": "master", "model_series": "hy_3", "train_stage": "pretrain"},
  "input_path": "/data/.../parquet_dir",
  "output_path": "/apdcephfs_xxx/share_yyy/bin_out",
  "seq_len": 32768,
  "packing_strategy": "CHUNK_LONG_SEQ",
  "add_source_id": true,
  "source_id_offset": 150000,
  "add_loss_mask": true,
  "loss_mask_offset": 300000,
  "partition_num": 1024
}'
```

**调用示例 D（转 bin 顺带合并分片：1024 个碎 bin 合成 8 个大文件）**：

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_pretrain_conversion '{
  "wsid": 10103,
  "file_format": "parquet",
  "app_group": "TaiJi_HYAide_xxx",
  "hdfs_cluster": "hdfs://nn-xxx",
  "resource_id": "g_teg_hunyuan_yizhanshi_datax.16421",
  "resource_location": "nj",
  "operator": "parquet2bin_v2",
  "git_based_tokenizer": {"branch": "master", "model_series": "hy_3", "train_stage": "pretrain"},
  "input_path": "/data/.../parquet_dir",
  "output_path": "/apdcephfs_xxx/share_yyy/bin_out",
  "seq_len": 32768,
  "partition_num": 1024,
  "enable_merge_bins": true,
  "merge_bins_count": 8
}'
```

> 只想合并、**输入已经是 bin/idx 分片**时，不要用上面这个（会白转一次 bin），改用 `create_hunyuan_data_merge_bins_task`（见 `merge_bins_tasks_api.md`）。

### 返回字段说明

创建返回任务详情，含 `id`（任务 ID）、`status`（初始一般为 PENDING）、`applied_defaults` 等。创建后任务进入 PENDING，由后端调度。

---

## get_hunyuan_data_pretrain_conversion

**功能**：查询预训练转 bin 任务状态。启用合并时另看 `merge_bins_status` / `merge_bins_output_path`。

### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `task_id` | int | ✅ | 转 bin 任务 ID |
| `wsid` | int | ✅ | 工作空间 ID |

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_pretrain_conversion '{"task_id": 5001, "wsid": 10103}'
```

### 单工具 SOP

- 查到 `FAILED` → 把 message 报错如实转达；可建议 `retry_hunyuan_data_pretrain_conversion` 重试。
- 启用 `enable_merge_bins` 时，除主 `status` 外还要看 `merge_bins_status` / `merge_bins_output_path` / `merge_bins_url`。

---

## query_hunyuan_data_pretrain_conversions

**功能**：分页查询预训练转 bin 任务列表。

### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `wsid` | int | ✅ | 工作空间 ID |
| `page` | int | ✅ | 页码 |
| `page_size` | int | ✅ | 每页条数 |

> ⚠️ **不要传 `page_index`**，本工具用 `page`。

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_hunyuan_data_pretrain_conversions '{"wsid": 10103, "page": 1, "page_size": 10}'
```

### 单工具 SOP

- 只调用一次。分页查询任务列表，删除（软删）的记录默认已过滤。

---

## retry_hunyuan_data_pretrain_conversion

**功能**：重试失败的预训练转 bin 任务。

**状态前置约束**：**仅 `FAILED` 任务可重试**；`RUNNING` / `PENDING` / 已成功任务会被后端拒（`当前状态不支持重试`）。

### 参数表

| 参数名 | 类型 | 是否必需 | 描述 |
|--------|------|:--------:|------|
| `task_id` | int | ✅ | 转 bin 任务 ID |
| `wsid` | int | ✅ | 工作空间 ID |

### 调用示例

```bash
python3 scripts/connect_mcp.py call retry_hunyuan_data_pretrain_conversion '{"task_id": 5001, "wsid": 10103}'
```

---

## delete_hunyuan_data_pretrain_conversion

**功能**：删除预训练转 bin 任务。**危险操作，调用前必须先向用户二次确认**（仅需 `task_id`）。**严禁**在用户未明确说"确认删除/删掉 task xxx"时自行调用。删除前建议先用 `get_hunyuan_data_pretrain_conversion` 核对任务信息。

**状态前置约束**：**仅终态任务可删**（`FAILED` / `SUCCEEDED` / `STOPPED`）；`RUNNING` / `PENDING` 会被后端拒（`当前状态不支持删除: status=RUNNING`）。

**软删语义**：删除为**软删**（后端打 `deleted_at` 标记，返回 `code:0`）。删除后，列表接口 `query_hunyuan_data_pretrain_conversions` 默认已过滤软删记录（无需传过滤参数）；仅 `get_hunyuan_data_pretrain_conversion`（按 id 单查）仍会返回带 `deleted_at` 的记录，属按主键精确查询的正常行为。

### 参数表

| 参数 | 必填 | 说明 |
|------|:---:|------|
| `task_id` | ✅ | 要删除的任务 ID |

### 调用示例

```bash
python3 scripts/connect_mcp.py call delete_hunyuan_data_pretrain_conversion '{"task_id": 5001}'
```

> 注：本 skill 未暴露 export / cudofs / topic / dataset 的删除工具，这些对象目前无程序化删除入口（如需清理请走 Web 端）。

---

### 常见错误

| 报错 | 原因 | 处理 |
|------|------|------|
| `应用组不能为空` | 没传 app_group | 先 `query_hunyuan_data_app_groups` 选一个 |
| `hdfs集群不能为空` | 没传 hdfs_cluster | 先 `list_hunyuan_data_app_group_hdfs_quota`（入参 `app_group_name`）选一个 |
| `operator 不能为空` | 没传 operator | 先 `list_hunyuan_data_pretrain_operators` 选一个（一般取第一个） |
| `git_based_tokenizer（Tokenizer 三级级联）不完整` | branch/model_series/train_stage 缺段 | 依次用 tokenizer 三级列举工具选齐三段 |
| `git tokenizer 选择不完整` | 后端启动时三级仍不全 | 同上，确保三段都传 |
| `平台账号不在应用组中，请发起申请...` | 平台账号没加入该应用组 | 按提示走 wedata 申请加入应用组并等审批，或换一个已加入的应用组 |
| `计算资源地域不能为空` | 没传 resource_id / resource_location | 先 `query_hunyuan_data_compute_resources`（传 entity_type=PRETRAIN_DATA_CONVERSION）拿 id 和地域；只查到南京资源时多半是漏传 entity_type |
| `packing_strategy 非法` | 传了枚举外的值 | 只能是 SKIP_LONG_SEQ / CHUNK_LONG_SEQ / BESTFIT |
| 输出目录非空 / 无写权限 | output_path 不是空目录或无权限 | 换一个空的、有写权限的 CEPH 路径 |
