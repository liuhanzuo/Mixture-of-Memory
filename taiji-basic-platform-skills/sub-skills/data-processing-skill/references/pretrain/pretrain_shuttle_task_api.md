## get_hunyuan_data_pretrain_shuttle_task

**功能**：按融合任务 id 查询单个预训练融合任务（shuttle-task / 融合 Pipeline）的最新状态、5 个阶段的进度、融合产物存储路径，以及 status / message / uc_task_id / wedata_url / 各时间戳等信息。

> **什么是「预训练融合任务」？** 它是一条把多份预训练数据源经 **去重 → 融合 → 质检 → 转 bin → 同步** 五个阶段串起来的融合 Pipeline，产物是一份可直接用于预训练的融合数据（落在 `storage_path`）。


> ⚠️ 纯只读工具，无副作用，无需二次确认。拿到 `shuttle_task_id` 立即直接调用本工具；**严禁**把命令行粘给用户。
> ⚠️ 本 skill **不暴露**融合任务的创建 / 重试 / 停止 / 删除工具，这些操作请走 Web 端；不要臆造 `create_* / retry_* / stop_* / delete_*` 同名工具。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|:--------:|--------|------|
| `shuttle_task_id` | int | ✅ 必填 | - | 预训练融合任务 ID。为空 / 非正整数时后端返回 `shuttle_task_id 不能为空` |
| `wsid` | int | ❌ 可选 | 不传 | 工作空间 ID。**传入则**后端校验该融合任务是否归属此工作空间（不属于会被拒）；**不传则**跳过归属校验。用户没给就不要传，也不要填 0 |

### 调用示例

**示例 A：最小调用（只给融合任务 id）**

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_pretrain_shuttle_task '{
  "shuttle_task_id": 5001
}'
```

**示例 B：带 wsid 归属校验**

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_pretrain_shuttle_task '{
  "shuttle_task_id": 5001,
  "wsid": 10103
}'
```

### 返回字段说明

返回统一信封 `{code, message, data}`（`code == 0` 成功）；`data` 为 `ShuttleTaskInfo`，字段为 **snake_case**。核心字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 融合任务 ID |
| `name` | str | 任务名称 |
| `status` | str | 任务总状态，见下方「状态说明」 |
| `message` | str | 失败原因（仅 `FAILED` 状态有值） |
| `storage_path` | str | **融合产物存储路径**（最终产出的融合数据落盘位置） |
| `phase_status` | object | **5 个阶段 → 状态** 的映射，见下方「阶段说明」 |
| `phase_details` | object | 5 个阶段 → `{status, detail_url}` 的映射（比 `phase_status` 多一个每阶段的详情链接） |
| `model_version` | str | 模型版本标识 |
| `data_stage` | str | 数据阶段（如 `PRE_TRAIN`） |
| `data_type` | str | 数据类型（如 `GENERAL` / `CODE` / `MATH`） |
| `data_name` / `data_desc` | str | 数据名称 / 描述 |
| `app_group` | str | 应用组 |
| `location` | str | 数据位置 / 地域 |
| `uc_task_id` | int | 底层 UC（wedata）任务 ID |
| `wedata_url` | str | wedata 任务详情页链接，可点进去看底层子任务 |
| `owner_list` | list[str] | 责任人 RTX 列表 |
| `creator` / `updater` | str | 创建人 / 更新人 |
| `create_time` / `update_time` / `end_time` | str | 时间线，格式 `yyyy-MM-dd HH:mm:ss`（GMT+8） |
| `file_type` | str | 文件类型（如 `JSON`） |

> ℹ️ 非 owner/admin 查询时，部分敏感字段（如 `storage_path`）后端可能脱敏；以后端实际返回为准。

### 状态说明（`status`，对应后端 ExtendedStatus）

| 状态值 | Emoji | 说明 |
|--------|-------|------|
| `PENDING` | ⏳ | 待执行 / 调度中 |
| `RUNNING` | 🔄 | 执行中 |
| `SUCCEEDED` | ✅ | **终态**：融合成功，产物在 `storage_path` |
| `FAILED` | ❌ | **终态**：失败，看 `message` + `phase_status` 定位失败阶段 |
| `STOPPING` | ⏸️ | 停止中 |
| `STOPPED` | ⏹️ | **终态**：已停止 |
| `EXPIRED` | 🗑️ | **终态**：执行终止 / 过期 |
| `SKIPPED` | ⏭️ | 跳过 |

### 阶段说明（`phase_status` / `phase_details`）

融合 Pipeline 固定包含 **5 个阶段**（key 为枚举名，value 为该阶段状态串，取值同上方状态说明，另可能为 `SKIPPED`）：

| 阶段 key | 中文 | 说明 |
|----------|------|------|
| `DEDUPLICATION` | 数据去重 | 未开启去重（`task_config.enable_dedup=false`）时为 `SKIPPED` |
| `DATA_INTEGRATION` | 数据融合 | 多源数据融合 |
| `QUALITY_INSPECTION` | 数据质检 | 融合后质检 |
| `DATA_CONVERSION` | 数据转 bin | 配置为跳过转 bin 时为 `SKIPPED` |
| `DATA_EXPORT` | 数据同步 | 融合产物同步 / 导出 |

- `phase_status`：`{ "DEDUPLICATION": "SUCCEEDED", "DATA_INTEGRATION": "RUNNING", ... }`。
- `phase_details`：`{ "DATA_INTEGRATION": { "status": "RUNNING", "detail_url": "https://..." }, ... }`，比 `phase_status` 多了每阶段的 `detail_url`（可点进底层子任务）。
- 判断「卡在哪一步」：找 `phase_status` 里第一个非 `SUCCEEDED` / 非 `SKIPPED` 的阶段即可。

### 返回示例（Markdown 呈现建议）

```
# 预训练融合任务详情 (ID: 5001)

- **任务名称**: hy3-general-融合-0707
- **状态**: 🔄 RUNNING
- **模型版本**: hy_3
- **数据阶段**: PRE_TRAIN
- **数据类型**: GENERAL
- **融合产物路径**: `/apdcephfs_nj7/share_xxx/shuttle_out/5001/`
- **应用组**: TaiJi_HYAide_xxx
- **wedata 链接**: https://wedata.woa.com/.../task/5001
- **创建人**: your_rtx_name
- **创建时间**: 2026-07-07 10:00:00

### 各阶段进度

| 阶段 | 状态 |
|------|------|
| 数据去重（DEDUPLICATION） | ⏭️ SKIPPED |
| 数据融合（DATA_INTEGRATION） | ✅ SUCCEEDED |
| 数据质检（QUALITY_INSPECTION） | 🔄 RUNNING |
| 数据转 bin（DATA_CONVERSION） | ⏳ PENDING |
| 数据同步（DATA_EXPORT） | ⏳ PENDING |

---

> 🔄 融合任务执行中，可稍后再次查询进度。
```

> ⚠️ 表格与后续引导文字之间**必须留一个空行**。

### 单工具 SOP

- 查到 `status=FAILED` → 把 `message` 原样转达用户，并指出是哪个阶段失败（看 `phase_status` 里非 SUCCEEDED 的阶段）；可附上 `wedata_url` 供用户深入排查。**不要**臆测原因、不要自动重试。
- `wsid` 可选，用户没给就只带 `shuttle_task_id` 调用（这点与 `get_hunyuan_data_export_task` / `get_hunyuan_data_pretrain_conversion` 的「wsid 必填」不同）。

---

### 常见错误

| 报错 | 原因 | 处理 |
|------|------|------|
| `shuttle_task_id 不能为空` | 没传 `shuttle_task_id` 或传了非正整数 | 补上正确的融合任务 ID |
| 未找到 ID 为 xxx 的融合任务 | `shuttle_task_id` 不存在 | 核对融合任务 ID |
| 任务不属于该工作空间 / wsid 不匹配 | 传了 `wsid` 但任务不归属该空间 | 去掉 `wsid` 再查，或换成任务实际所属的 `wsid` |
| 认证失败（401/403） | Token 未配置 / 过期 / 无权限 | 见 `helper_api.md` 的「获取 API Token」段 |
