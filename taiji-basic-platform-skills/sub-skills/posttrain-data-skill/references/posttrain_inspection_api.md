## create_hunyuan_data_quality_inspection

**用途**：对一条已上传成功的**后训练 Topic 数据版本**触发**旧版**质量检测记录。后端会：
- 该 Topic 数据从未质检过（`inspectionId` 为空）→ 新建 `QualityInspection` 记录；
- 已有质检且 `SUCCEEDED` → 幂等直接返回；
- 已有质检且 `FAILED` → 调 `retry`，复用同一条记录重跑；
- 已有质检且 `PENDING` / `RUNNING` → 后端抛"质量检查未完成，请稍后重试"。

> ⚠️ **与前端 V2 页面列的关键差异（必读）**：太极 Topic 详情页【底线质检】【内容质检】列读的是创建 TopicData 时 `include_baseline` / `include_content` / `include_general_content` 发起的 **V2 AIData 批量任务**。本工具**不会**发起该批量任务，因此：
> - 用户说「注册数据并质检 / 页面上要看到底线质检」→ **不要用本工具**，应走 `create_hunyuan_data_topic_data` 并按前端默认传 `include_baseline=true`（见 `posttrain_topic_data_api.md`）；
> - 已建好的 V2 数据若两列已是「未质检」，前端编辑态也无法补开；本工具也**补不了**这两列，应引导用户**新建一个 version** 并在创建时带上开关；
> - 本工具仍可用于：查询/兼容旧 `inspection_id` 链路、用户明确只要旧 QualityInspection 记录时。

### 触发条件

- 用户提示词**明确包含**以下关键词之一，且**已知 topic_data_id、只要旧质检记录** → 可路由到本工具：
  - 「对已有 topic_data 跑旧版质检记录」「重试 quality-inspection」「查/建 inspection 记录」
- 用户说「注册 / 新建数据版本并做质检 / 页面底线质检」→ ❌ **不走本工具**，走 `create_hunyuan_data_topic_data`（带 `include_*`）。
- 用户说"检查数据"但意图模糊 → 必须追问区分"触发新质检"还是"查已有质检结果"。
- 用户说"转 bin / 分词"→ ❌ 不走本工具，走 `create_hunyuan_data_sft_conversion`。
- 用户说"导出 / 拷贝 / 搬迁 / 同步"→ ❌ 不走本工具，走 `create_hunyuan_data_export_task` 或 `create_hunyuan_data_cudofs_copy_task`。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `topic_data_id` | int | ✅ 必填 | - | Topic 数据版本 ID（`PostTrainTopicData.id`）；**注意不是** `topic_id` 也不是 `dataset_id` |

### 参数前置校验（调用前必做）

1. `topic_data_id` **必须是正整数**；为空 / 0 / 负数 → 直接反馈"topic_data_id 不能为空或非正整数"，不调工具。
2. Topic 数据当前状态必须是 `SUCCEEDED`：
   - 若用户提供的 `topic_data_id` 对应的 Topic 数据还在 `PENDING` / `RUNNING`，后端会抛 `Topic数据未完成上传，不允许进行质检`；
   - 若不确定 Topic 数据状态，建议先调 `get_hunyuan_data_quality_inspection` 辅助查看（通过数据详情里的 `inspectionId` 反推），或直接尝试调用本工具并把后端的错误 message 反馈给用户。
3. **输入文件格式硬约束**（质检算子的隐式假设，违反会产生大量 `format_filter` / `crash_filter` 误报）：
   - Topic 数据的 `sourcePath` 必须是**单个文件**，不能是目录（Topic 数据创建时后端已用 `FileType.FILE` 强校验）；
   - 内容**必须是 JSONL**（每行一个独立 JSON 对象）：后缀 `.json` 也可，但不能是整文档单一 JSON 数组或跨行格式化的 JSON；
   - 路径 / appGroup / location 三者一致，且当前用户对路径**有读权限**。
   - 若用户在创建 Topic 数据阶段就没满足这些约束，质检会"假性成功"但不合格率异常偏高；此时应先引导用户重建一版合规的 Topic 数据，再触发质检。

### 调用示例

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_quality_inspection '{
  "topic_data_id": 3021
}'
```

### 返回字段说明

返回 Markdown 文本；关键字段如下：

| 字段 | 说明 |
|------|------|
| 质检记录 ID (`inspection_id`) | 后端为本次触发生成 / 复用的 `QualityInspection.id`；后续所有查询 / 预览 / 下载都用此 ID |
| Topic 数据状态 | 此时仍是 Topic 数据的 `status`（非质检 status）；正常应为 `SUCCEEDED` |
| 启用质检 (`enableInspection`) | 质检记录（QualityInspection）侧的标记，触发后一般被置为 `true`（与 TopicData 创建入参无关——TopicData 侧已改为 include_baseline/include_content/include_general_content 三开关）|
| 后续操作建议 | 通常提示用户用 `get_hunyuan_data_quality_inspection` 轮询新 `inspection_id` |

### 返回示例

```
# 已触发后训练数据质检 (Topic 数据 ID: 3021)

- **质检记录 ID (inspection_id)**: 88
- **Topic 数据状态**: ✅ SUCCEEDED
- **启用质检**: True
- **Topic ID**: 101
- **数据集 ID**: 7001
- **数据版本**: 2026-04-24-v1
- **源路径 (sourcePath)**: `/apdcephfs_jn2/share_xxx/input/train.jsonl`
- **统一存储路径 (storagePath)**: `/apdcephfs_jn2/share_xxx/unified/posttrain_topic_data_3021/`

### 🔁 后续操作建议
1. 轮询质检进度：调用 `get_hunyuan_data_quality_inspection`，参数 `inspection_id=88`
2. 质检完成（SUCCEEDED）后可预览/下载不合格数据：`preview_hunyuan_data_quality_inspection_data` / `get_hunyuan_data_quality_inspection_download_url`
```

### 典型错误与处理建议

| 触发条件 | 后端 message | 处理建议 |
|------|------|------|
| Topic 数据未 SUCCEEDED | `Topic数据未完成上传，不允许进行质检：id=xxx` | 让用户先等待或排查上传进度（`get_hunyuan_data_quality_inspection` 无法查 Topic 数据本体，需通过其他查询 Topic 数据的工具） |
| 已有质检仍在跑 | `topic数据质量检查未完成，请稍后重试：id=xxx` | 让用户稍等；可直接用已经返回的 `inspection_id` 走 query 轮询 |
| `topic_data_id` 不存在 | `topic数据不存在：id=xxx` | 让用户核对 ID |

---

## get_hunyuan_data_quality_inspection

**用途**：根据 `inspection_id` 查询质检任务的详情与状态。

### 触发条件

- 用户提示词**明确包含**以下关键词之一 → 路由到本工具：
  - 「查质检」「质检状态」「质检进度」「质检结果」「质检跑完了吗」「质检任务详情」
  - 「多少行合格」「多少行不合格」「合格率」「不合格率」
  - 用户给出的是一个明显的**质检 ID**（如说"质检 88"/"inspection 88"/"inspection_id=88"）
- 用户说"查任务 XX 状态"但未说"质检" → ❌ 可能是 `get_hunyuan_data_sft_conversion` / `get_hunyuan_data_export_task` / `get_hunyuan_data_cudofs_copy_task`；此时若 ID 没上下文（不是 `basic_train_` 开头、也没说 `conversion` / `export` / `cudofs`），必须**先追问是哪类任务**再路由。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `inspection_id` | int | ✅ 必填 | - | 质检记录主键 ID（即 `create_hunyuan_data_quality_inspection` 返回的 `inspection_id`，**不是** `topic_data_id`） |

### 参数前置校验

1. `inspection_id` 必须为正整数；否则反馈"inspection_id 不能为空或非正整数"，不调工具。
2. 若用户把 `topic_data_id` 误传为 `inspection_id`，后端会返回"质检记录不存在：id=xxx"；此时应向用户确认 ID 来源。
3. 本接口**不需要 `wsid`**（与 `get_hunyuan_data_cudofs_copy_task` / `create_hunyuan_data_export_task` 等不同），`wsid` 不是本工具必填。

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_quality_inspection '{
  "inspection_id": 88
}'
```

### 返回字段说明（Markdown 渲染自 `QualityInspectionInfo`）

| 字段 | 说明 |
|------|------|
| 状态 (`status`) | `PENDING` / `RUNNING` / `SUCCEEDED` / `FAILED`，分别对应 emoji `⏳` / `🔄` / `✅` / `❌` |
| Topic ID / Topic 数据 ID | 质检归属的 Topic 与 TopicData |
| 链路版本 (`version`) | `v1` = 走 WeData；`v2` = 走 Unity Catalog（UC） |
| 外部任务 ID / 外部任务链接 | 底层 WeData / UC 任务 ID 与详情页 URL，点开可看完整日志 |
| 行数统计 | `validRows` / `invalidRows` / 总计 |
| 大小统计 | `validSize` / `invalidSize`（字节） |
| 合格数据路径 (`validPath`) | 质检合格数据 Ceph 目录 |
| 不合格数据路径 (`invalidPath`) | 质检不合格数据 Ceph 目录 |
| 可下载的不合格数据文件类型 | 基于 `details.existInvalidJson / existInvalidParquet / existInvalidSampledParquet` 聚合出的清单；**直接决定** `get_hunyuan_data_quality_inspection_download_url` 可选的 `file_type` |
| 总耗时 (`costSeconds`) | `endTime - createTime`（秒）；终态才有值 |
| 失败信息 (`message`) | `status=FAILED` 时含失败原因；成功时为空 |

### 返回示例

```
# 后训练质检任务详情 (ID: 88)

- **状态**: ✅ SUCCEEDED
- **Topic ID**: 101
- **Topic 数据 ID**: 3021
- **链路版本**: v2
- **外部任务 ID (WeData/UC)**: 998877
- **外部任务链接**: [https://wedata.xxx/task/998877](https://wedata.xxx/task/998877)
- **行数统计**: 合格 12345 / 不合格 67 / 总计 12412
- **大小统计（字节）**: 合格 10485760 / 不合格 524288
- **合格数据路径**: `/apdcephfs_xxx/inspection/valid`
- **不合格数据路径**: `/apdcephfs_xxx/inspection/invalid`
- **创建人**: leslizhang
- **创建时间**: 2026-04-24 10:00:00
- **更新时间**: 2026-04-24 10:08:00
- **结束时间**: 2026-04-24 10:08:00
- **总耗时（秒）**: 480

### 可下载的不合格数据文件类型
JSON, PARQUET, SAMPLED_PARQUET

> 💡 预览（preview_hunyuan_data_quality_inspection_data）固定使用 SAMPLED_PARQUET；下载（get_hunyuan_data_quality_inspection_download_url）可按上述列表挑选 file_type。

### ✅ 质检已完成
后续可用：preview_hunyuan_data_quality_inspection_data 预览采样不合格数据、get_hunyuan_data_quality_inspection_download_url 获取不合格数据下载说明。
```

### 典型错误

| 触发条件 | 后端 message | 处理建议 |
|------|------|------|
| `inspection_id` 不存在 | `质检记录不存在：id=xxx` | 确认 ID 来源；常见错是把 `topic_data_id` 当 `inspection_id` 传 |

---

## preview_hunyuan_data_quality_inspection_data

**用途**：从质检采样结果中预览一批不合格样本的原始内容。**固定使用 `SAMPLED_PARQUET`** 采样文件，返回的每条是 parquet 里 `raw_data` 列的字符串。

### 触发条件

- 用户提示词**明确包含**以下关键词之一 → 路由到本工具：
  - 「预览不合格」「看看不合格」「抽几条不合格」「不合格数据长啥样」「不合格样本的内容」
  - 「`err_code=xxx` 的样本」（用户已知错误码想过滤）
- 用户说"预览数据"但**没**说"不合格" → ❌ 可能想看训练数据原文（这目前不在本模块范围）；必须先追问"您想预览的是质检不合格样本，还是训练数据本身？"
- 用户说"下载采样 parquet" → ❌ 应走 `get_hunyuan_data_quality_inspection_download_url`（用 `file_type=SAMPLED_PARQUET`）；preview 是把采样结果读出来给用户看内容，不是下载。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `inspection_id` | int | ✅ 必填 | - | 质检记录主键 ID |
| `metric` | str | 否 | `null` | 错误码过滤（对应采样 parquet 的 `err_code` 列，不区分大小写）。参数名必须精确为 `metric`，不得写成 `category` / `filter` / `err_code`。典型取值：`format_filter` / `crash_filter` / `content_error_flag` / `too_long_num_error` / `unicode_error` / `is_language_mix_flag` / `ban_flag` / `hunyuan_flag` 等（可先从 `get_hunyuan_data_quality_inspection` 返回的 metadata 里看到完整列表） |
| `page_index` | int | 否 | `1` | 页码，从 1 开始 |
| `page_size` | int | 否 | `20` | 每页条数，上限 `1000`；实际返回还受后端 `PreviewConfig.limit` 约束 |

> ℹ️ **注意**：`page_index` / `page_size` 会透传到后端，但后端 UC 的 `previewParquet` 实际是按配置固定 limit 取一批采样，**不是严格分页**；用户问"分页展示"时应解释这一点。

### 参数前置校验

1. `inspection_id` 必须正整数；
2. 质检必须已 `SUCCEEDED`——建议先调一次 `get_hunyuan_data_quality_inspection` 确认；未完成会得到后端 `质检未完成，无法预览`；
3. 质检必须有 `SAMPLED_PARQUET`（`details.existInvalidSampledParquet=true`）；**老链路（v1）无 manifest，没采样文件**，会得到 `不合格数据预览文件不存在`——此时应引导用户改用 `get_hunyuan_data_quality_inspection_download_url` 下载 JSON 全量。

### 调用示例

```bash
python3 scripts/connect_mcp.py call preview_hunyuan_data_quality_inspection_data '{
  "inspection_id": 88,
  "metric": "format_filter",
  "page_index": 1,
  "page_size": 20
}'
```

### 返回字段说明

| 字段 | 说明 |
|------|------|
| 命中条数 | 本次返回的样本数 |
| 错误码过滤 (`metric`) | 展示本次过滤使用的值（未指定时特别标注） |
| 不合格样本列表 | 每一条是一行原始数据字符串（来自 parquet 的 `raw_data` 列）；`content` / `role` / `tool_calls` 等 JSON 字段会原样保留 |

### 返回示例

```
# 后训练质检 88 不合格数据预览

- **命中条数**: 3
- **错误码过滤 (metric)**: `format_filter`

### 不合格样本列表（每行为一条原始记录）

**#1**
```
{"role":"user","content":""}
```

**#2**
```
{"role":"user","content":null}
```

**#3**
```
{not valid json at all}
```
```

### 典型错误

| 触发条件 | 后端 message | 处理建议 |
|------|------|------|
| `inspection_id` 不存在 | `质检记录不存在：id=xxx` | 确认 ID 来源；常见错是把 `topic_data_id` 当 `inspection_id` 传 |
| 质检未完成 | `质检未完成，无法预览：id=xxx, status=yyy` | 让用户等待 / 先查状态 |
| 老链路无采样文件 | `不合格数据预览文件不存在：id=xxx` | 建议用户改走 `get_hunyuan_data_quality_inspection_download_url` 下载 `JSON` 全量文件 |
| 质检任务详情丢失 | `质检任务详情不存在：id=xxx` | 属于数据治理问题，让用户找数据管理员 |

---

## get_hunyuan_data_quality_inspection_download_url

**用途**：获取不合格质检数据文件的**下载说明**。

>
> 1. 验证质检已 `SUCCEEDED`、拿到 `invalidPath` 与可用 `file_type` 清单；
> 2. 构造等价下载 URL，不把 token 放进 URL；
> 3. 构造等价 curl 命令（`-OJ`）；
> 4. 展示不合格数据 Ceph 绝对路径（`invalidPath`），方便有 Ceph 权限的用户直接从文件系统取走。
>
> Agent 应把这些内容**如实**反馈给用户，不要擅自改写为"已下载"的措辞。

### 触发条件

- 用户提示词**明确包含**以下任一组合 → 路由到本工具：
  - 「下载不合格数据 / 不合格样本 / 不合规数据」
  - 「下载质检不合格数据」「给我不合格数据的下载命令 / 链接」
  - 「怎么把不合格数据拿到本地」
- 用户说"下载数据"但没提"不合格"——必须追问区分三条下载路径（本工具 / `create_hunyuan_data_export_task` / `create_hunyuan_data_cudofs_copy_task`）。
- 用户说"下载训练数据 / 导出训练数据"（不含"不合格"）→ ❌ 走 `create_hunyuan_data_export_task`，本工具不适用。

### 参数

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `inspection_id` | int | ✅ 必填 | - | 质检记录主键 ID |
| `file_type` | str | 否 | `null` | 下载文件类型，枚举只能是大写 `JSON`（全量 json，老数据兼容）/ `PARQUET`（全量 parquet，新链路）/ `SAMPLED_PARQUET`（采样 parquet，与预览同一份）。参数名必须精确为 `file_type`；不要使用 `file_format`，也不要把 `details.existInvalidParquet` 等字段名反推成 `invalid_parquet`。用户指定格式时按指定值传；用户未指定格式但问"怎么下载/下载链接"时，先显式 query 确认可用类型，默认只调用一次 `file_type="PARQUET"`，不可用再降级到 `JSON` |

### 参数前置校验

1. `inspection_id` 必须正整数；
2. 质检必须已 `SUCCEEDED`——本工具在真正构造下载说明前**会自行做一次 query 探测**，未完成会在响应里明确告知"质检未完成，无法下载"；
3. `file_type`（若传）必须属于 `{JSON, PARQUET, SAMPLED_PARQUET}`；其他值直接反馈"file_type 取值非法"；
4. `file_type`（若传）必须命中 `details.existInvalidXxx`；否则给出友好提示"当前质检记录不存在 XXX 类型的不合格数据文件"，并列出可选项；
5. **权限约束**：底层文件读取由 Ceph 权限控制，用户必须对 `invalidPath` 有读权限。

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_quality_inspection_download_url '{
  "inspection_id": 88,
  "file_type": "PARQUET"
}'
```

### 返回字段说明

| 字段 | 说明 |
|------|------|
| 质检状态 | 若非 `SUCCEEDED`，本工具直接返回错误而非下载说明 |
| 目标 file_type | 本次要下载的类型；未指定时展示"(未指定；后端按 JSON 兼容老数据)" |
| 当前可用 file_type | 基于 `details.existInvalidXxx`，作为用户挑选下载类型的依据 |
| 不合格数据 Ceph 路径 (`invalidPath`) | 让用户在有 Ceph 访问权限的环境下直接从文件系统取走 |
| 下载 URL | 下载链接，含 `fileType` query string |
| 等价 curl 命令 | 带 `-OJ` 选项与认证 Header；用户应把 `<YOUR_API_TOKEN>` 占位符替换为真实 token |

### 返回示例

```
# 后训练质检 88 不合格数据下载说明

- **质检状态**: ✅ SUCCEEDED
- **目标 file_type**: PARQUET
- **当前可用 file_type**: JSON, PARQUET, SAMPLED_PARQUET
- **不合格数据 Ceph 路径 (invalidPath)**: `/apdcephfs_xxx/inspection/invalid`

### 备选：直接从 Ceph 取文件

若你对 `/apdcephfs_xxx/inspection/invalid` 所在的 Ceph 目录有读权限，也可以直接通过
Ceph 客户端 / scp / rsync 等方式取走全部不合格数据（含 JSON / PARQUET / 采样 PARQUET / manifest）。
```

### 典型错误

| 触发条件 | 返回内容 | 处理建议 |
|------|------|------|
| `inspection_id` 不存在 | `质检记录不存在：id=xxx` | 确认 ID 来源；常见错是把 `topic_data_id` 当 `inspection_id` 传 |
| 质检未 SUCCEEDED | `质检未完成，无法下载：inspection_id=xxx, status=yyy` | 让用户先用 `get_hunyuan_data_quality_inspection` 等待变 SUCCEEDED |
| `file_type` 非法 | `file_type 取值非法：xxx` | 让用户在 `{JSON, PARQUET, SAMPLED_PARQUET}` 中选一个 |
| 指定类型文件不存在 | `当前质检记录不存在 XXX 类型的不合格数据文件` | 切换到 `可选 file_type` 中列出的类型 |

---

### 常见问题

### Q1：`inspection_id` 和 `topic_data_id` 是同一个 ID 吗？
不是。`topic_data_id` 是 Topic 数据版本主键；`inspection_id` 是质检记录主键。一条 Topic 数据最多关联一条活跃质检记录，`topic_data_id` → `inspection_id` 的映射由后端维护（体现在 Topic 数据的 `inspectionId` 字段）。

### Q2：质检失败了如何排查？
1. 看 `get_hunyuan_data_quality_inspection` 返回的 `message` 字段（简短原因）；
2. 点开返回里的**外部任务链接**（WeData / UC 任务详情页），看完整执行日志；
3. 常见根因：输入文件不是 JSONL、文件编码异常、路径权限不足；
4. 排除后用 `create_hunyuan_data_quality_inspection` 同一个 `topic_data_id` 再触发一次（后端会自动 retry 已 FAILED 的记录）。

### Q3：`preview_hunyuan_data_quality_inspection_data` 能做真正的分页吗？
否。`page_index` / `page_size` 会透传给后端，但后端 UC 的 `previewParquet` 实际是按配置固定 limit 取一批采样，**不是严格意义上的分页**。`pageSize` 上调可拿到更多条，但不能稳定翻页。需要完整不合格数据请用 `get_hunyuan_data_quality_inspection_download_url` 下载整个文件再自己处理。

### Q4：没有 wsid 能调吗？
**可以**。本模块 4 个工具**都不要求** `wsid` Header。这与 `create_hunyuan_data_export_task` / `create_hunyuan_data_cudofs_copy_task` 等不同。Agent 遇到本模块的调用时，**不要**因为没 `wsid` 而向用户追问。
