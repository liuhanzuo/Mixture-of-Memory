## create_hunyuan_data_export_task

**功能**：创建一个数据导出任务，把源**文件存储**路径（或太极业务实体对应的数据）拷贝到目标**文件存储**路径。文件存储指 ceph / nitrofs(hifs)，二者同属文件存储（`hifs` 是后端存储类型名，`nitrofs` 是产品名，是同一个东西）。

**支持的 4 种方向（全部同一个工具、同一套参数）**：

| 方向 | 说明 |
|:---|:---|
| ceph → ceph | 历史能力（俗称 ceph2ceph） |
| ceph → nitrofs(hifs) | 升级后新增 |
| nitrofs(hifs) → ceph | 升级后新增 |
| nitrofs(hifs) → nitrofs(hifs) | 升级后新增 |

> ⭐ **调用方无需感知两端的存储类型**：入参只有 `source_path` + `target_path`，**没有**"存储类型 / storage_type / 是否 nitrofs(hifs)"之类的参数，后端按路径自动识别。**严禁**因为一端是 nitrofs(hifs) 就追问用户、拒绝执行或改用其他工具。


> ⚠️ 当 `wsid` + 源（`source_path` 或 `entity_type`+`entity_id`）+ 目标（`target_path`）均已齐备时，立即直接通过 MCP 调用本工具创建任务。

**用户表述归一化**：以下说法都对应本工具，无需追问"到底是导出还是复制"：
- "把数据 yyyy **复制 / 拷贝 / copy** 到另一个路径 xxxx"
- "把数据 yyyy **导出 / export** 到路径 xxxx"
- "把数据 yyyy **搬迁 / 搬到 / 迁移 / 转移 / 同步** 到 xxxx"
- "把 **ceph 上的数据拷到 nitrofs(hifs)**"、"把 **nitrofs(hifs) 的数据拷回 ceph**"、"在两个 **nitrofs(hifs)** 之间拷数据"
- "把 topic 数据版本 123 导出到 /ceph/xxx"
- "因为地域不一致，把输入数据搬到计算资源所在地域"

### 两类调用场景

| 场景 | `data_source` | 必填字段组合 |
|:---|:---|:---|
| **按路径导出**（最常见，用户直接给路径） | `TAIJI_WEB`（默认） | `source_path` + `target_path` |
| **按业务实体导出**（用户给的是 topic 版本 ID、数据集 ID 等） | 非 `TAIJI_WEB`（如 `POSTTRAIN_TOPIC_DATA`、`PRETRAIN_DATASET` 等） | `entity_type` + `entity_id` |

> 两组参数**二选一**，`source_path` / `target_path` 与 `entity_type` / `entity_id` 通常不同时出现。

**必须确认的信息（最高优先级）：**
1. **wsid（工作空间 ID）**：必填，不能为 0 或空值。用户未提供 → 必须追问。
2. **data_source**：决定校验分支，默认 `TAIJI_WEB`（按路径导出场景）。按路径导出且用户未显式指定时，可省略该字段让 MCP/后端默认处理。
3. **source_path / target_path**：`TAIJI_WEB` 场景必填。两端都是**文件存储路径（ceph 或 nitrofs(hifs)，任意组合）**。后端会校验源路径存在、目标为空目录、当前用户读写权限。
4. **entity_id / entity_type**：非 `TAIJI_WEB` 场景必填。

**🚫 本工具【不需要】应用组 / queue_name / 业务组 / 资源队列 / 存储类型 参数（最高优先级）**：
- 本工具全部入参只有 `wsid` / `data_source` / `source_path` / `target_path` / `entity_id` / `entity_type` / `name`，**没有也不接受任何"应用组（appGroup / queue_name / business_flag）"或"存储类型（storage_type / ceph / nitrofs(hifs)）"参数**。
- **严禁向用户索要应用组或存储类型**。文件存储互拷的读写权限由后端**根据源/目标路径自动反查所属应用组与存储类型**完成校验，调用方无需、也无法指定。
- ⚠️ 不要把"模型地域拷贝（`update_hunyuan_models_card_location`，那个确实需要 `queue_name`）"的参数要求**串到本工具**上——两者是完全不同的功能。
- 只要 `wsid` + `source_path` + `target_path` 齐备就**立即调用**，不要因为"缺应用组"而追问用户。
- 若后端返回"没有写权限 appgroup=xxx ..."这类报错，那是**目标路径本身的归属应用组权限问题**（需要找该路径 owner 授权或换路径），**不是要用户在调用时填应用组**，请如实转达报错并给出"找 owner 授权 / 换可写路径"的建议。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID。不能为 0；会作为 HTTP Header `wsid` 传给后端用于数据隔离 |
| `data_source` | str | ❌ 可选 | `"TAIJI_WEB"` | 数据来源枚举名。常见值：`TAIJI_WEB`（按路径）/ `POSTTRAIN_TOPIC_DATA` / `PRETRAIN_DATASET` / `PRETRAIN_SHUTTLE_TASK` / `PRETRAIN_INTEGRATION_DATA` / `DATAX_SDK` / `OPEN_API` 等 |
| `source_path` | str | ⚠️ 条件必填 | `null` | 源文件存储路径（ceph 或 nitrofs(hifs) 均可）。`data_source=TAIJI_WEB` 时必填 |
| `target_path` | str | ⚠️ 条件必填 | `null` | 目标文件存储路径（ceph 或 nitrofs(hifs) 均可，与源端类型可以不同）。`data_source=TAIJI_WEB` 时必填，必须是空目录或不存在 |
| `entity_id` | int | ⚠️ 条件必填 | `null` | 业务实体 ID。`data_source` 非 `TAIJI_WEB` 时必填 |
| `entity_type` | str | ⚠️ 条件必填 | `null` | 业务实体类型枚举名。`data_source` 非 `TAIJI_WEB` 时必填 |
| `name` | str | ❌ 可选 | `null` | 任务名称。不传时后端启动时会基于实体自动生成；用户未明确要求命名时不要主动传 |
| ~~应用组 / queue_name~~ | - | 🚫 **不存在** | - | **本工具没有此参数**。权限由后端按路径自动反查应用组，严禁向用户索要 |
| ~~storage_type / 存储类型（ceph / nitrofs(hifs)）~~ | - | 🚫 **不存在** | - | **本工具没有此参数**。后端按路径自动识别 ceph / nitrofs(hifs)，严禁向用户索要 |

### 调用示例

**示例 A：按路径导出（最常见，用户直接给两个文件存储路径）**

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_export_task '{
  "wsid": 10086,
  "data_source": "TAIJI_WEB",
  "source_path": "/ceph/bucket-a/sft-data/",
  "target_path": "/ceph/bucket-b/sft-data-copy/",
  "name": "把SFT数据拷贝到另一个地域"
}'
```

**示例 A2：ceph → nitrofs(hifs)（与示例 A 唯一区别只是路径，工具/参数完全一致）**

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_export_task '{
  "wsid": 10086,
  "data_source": "TAIJI_WEB",
  "source_path": "/apdcephfs_jn2/share_302316223/leslizhang/sft_data/",
  "target_path": "/apdcephfs_nj33/share_301455001/leslizhang/sft_data_copy/",
  "name": "把SFT数据从ceph拷到nitrofs"
}'
```

> 源 `/apdcephfs_jn2`（id=2 < 30）是 ceph，目标 `/apdcephfs_nj33`（id=33 ≥ 30）是 nitrofs(hifs）；工具与入参不受影响。

**示例 A3：nitrofs(hifs) → ceph（反向同理；两个 nitrofs(hifs) 之间也是同样写法）**

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_export_task '{
  "wsid": 10103,
  "data_source": "TAIJI_WEB",
  "source_path": "/apdcephfs_zw33/share_305375363/share_11111111",
  "target_path": "/apdcephfs_zwfy11/dop-test/test-copy",
  "name": "把nitrofs数据回拷到ceph"
}'
```

> 源 `/apdcephfs_zw33`（id=33 ≥ 30）是 nitrofs(hifs)，目标 `/apdcephfs_zwfy11`（id=11 < 30）是 ceph。两端类型不同**不需要**任何额外参数或额外确认。

**示例 B：按业务实体导出（把某个 topic 数据版本导出到 ceph）**

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_export_task '{
  "wsid": 10086,
  "data_source": "POSTTRAIN_TOPIC_DATA",
  "entity_type": "POSTTRAIN_TOPIC_DATA",
  "entity_id": 123456,
  "target_path": "/ceph/bucket-b/exported/"
}'
```

**示例 C：跨地域中转场景（用户独立调用本工具把数据搬到南京 staging，仅用于排障）**

> 💡 常规跨地域 SFT 转 bin 请走 `sft_bin_auto_transfer_flow.md` 的跨地域 Pipeline（其 Step 1 / Step 3 就是 Agent 直接编排本工具完成"搬到南京 staging / 搬到目标 ceph"）；只有用户明确要"裸跑一次 ceph 拷贝把数据搬到南京"时才单独直接用本工具。

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_export_task '{
  "wsid": 10086,
  "source_path": "/apdcephfs_cq10/.../input-data/",
  "target_path": "/apdcephfs_jn2/.../input-data/",
  "name": "为转bin准备同地域输入数据"
}'
```

### 返回字段说明

返回 Markdown 格式的任务详情（`ExportTaskInfo`），关键字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | **任务 ID**，后续查询与日志都靠它 |
| `name` | str | 任务名称（后端可能自动补齐） |
| `status` | str | 状态，见下方"状态说明" |
| `source` | str | 数据来源（对应入参 `data_source`） |
| `entityType` | str | 业务实体类型（若按实体导出） |
| `entityId` | int | 业务实体 ID（若按实体导出） |
| `sourcePath` | str | 源路径。**非 owner/admin 时会脱敏为 `***`** |
| `storagePath` | str | 目标存储路径。**非 owner/admin 时会脱敏为 `***`** |
| `creator` | str | 创建人 |
| `createTime` / `beginTime` / `endTime` | str | 时间线 |
| `message` | str | 失败时的错误信息（仅 `FAILED` 状态有值） |

> ℹ️ 创建接口后端返回的是 `List<ExportTaskInfo>`（外层 `SingleResponse` 包一层列表），MCP 层已自动取列表首元素并格式化，上层无需感知。

### 返回示例

```
# 数据导出任务详情 (ID: 2001)

- **任务名称**: 把SFT数据拷贝到另一个地域
- **状态**: ⏳ PENDING
- **数据来源**: TAIJI_WEB
- **实体类型**: -
- **实体 ID**: -
- **源路径**: `/ceph/bucket-a/sft-data/`
- **目标路径**: `/ceph/bucket-b/sft-data-copy/`
- **创建人**: your_rtx_name
- **创建时间**: 2026-04-22 16:00:00
- **开始时间**: 2026-04-22 16:00:00
```

> 💡 创建后任务处于 `PENDING`，由 datax 后端定时调度器异步拉起真正的同步任务（v1 走 DOP 拷贝，v2 走智研 CEPHFS_COPY）。需要通过 `get_hunyuan_data_export_task` 轮询真实执行状态。

---

## get_hunyuan_data_export_task

**功能**：根据任务 ID 查询单个数据导出任务的详情与执行状态，用于轮询进度或定位失败原因。


> ⚠️ 当 `task_id` + `wsid` 齐备时，立即通过 MCP 直接调用本工具。查到 `FAILED` 后**主动续跑** `get_hunyuan_data_export_task_log` 取日志。

### 状态说明

| 状态值 | Emoji | 说明 | 下一步建议 |
|--------|-------|------|-----------|
| `PENDING` | ⏳ | 待执行，已创建等待调度器拉起 | 稍后重新调用本工具 |
| `RUNNING` | 🔄 | 执行中 | 可调用 `get_hunyuan_data_export_task_log` 查看实时进度 |
| `SUCCEEDED` | ✅ | **终态**：执行成功 | 向用户展示产出路径 `storagePath` |
| `FAILED` | ❌ | **终态**：执行失败 | **立即调用 `get_hunyuan_data_export_task_log` 查看日志定位原因**，必要时可引导用户通过 datax 后端的 retry 接口重试 |

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `task_id` | int | ✅ 必填 | - | 导出任务 ID（`create_hunyuan_data_export_task` 返回的 `id`） |
| `wsid` | int | ✅ 必填 | `0` | 工作空间 ID；不能为 0。后端会校验任务必须属于此 wsid，否则拒绝 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_export_task '{
  "task_id": 2001,
  "wsid": 10086
}'
```

### 返回字段说明

返回 Markdown 格式的任务详情，字段与 `create_hunyuan_data_export_task` 相同。额外在不同状态下会补充信息：
- **失败态**：展示 `message` 错误详情代码块，并给出"可通过 retry 接口重试"的提示；
- **成功态**：单独展示产出路径；
- **路径脱敏**：若当前账号非 owner/admin，`sourcePath` / `storagePath` 显示为 `***` 并附提示。

### 返回示例

**执行中：**

```
# 数据导出任务详情 (ID: 2001)

- **任务名称**: 把SFT数据拷贝到另一个地域
- **状态**: 🔄 RUNNING
- **数据来源**: TAIJI_WEB
- **实体类型**: -
- **实体 ID**: -
- **源路径**: `/ceph/bucket-a/sft-data/`
- **目标路径**: `/ceph/bucket-b/sft-data-copy/`
- **创建人**: your_rtx_name
- **创建时间**: 2026-04-22 16:00:00
- **开始时间**: 2026-04-22 16:00:10
```

**执行成功：**

```
# 数据导出任务详情 (ID: 2001)

- **任务名称**: 把SFT数据拷贝到另一个地域
- **状态**: ✅ SUCCEEDED
- **数据来源**: TAIJI_WEB
- **源路径**: `/ceph/bucket-a/sft-data/`
- **目标路径**: `/ceph/bucket-b/sft-data-copy/`
- **创建时间**: 2026-04-22 16:00:00
- **开始时间**: 2026-04-22 16:00:10
- **结束时间**: 2026-04-22 16:08:33

### ✅ 任务已完成
产出路径: `/ceph/bucket-b/sft-data-copy/`
```

**执行失败：**

```
# 数据导出任务详情 (ID: 2001)

- **任务名称**: 把SFT数据拷贝到另一个地域
- **状态**: ❌ FAILED
- **数据来源**: TAIJI_WEB
- **源路径**: `/ceph/bucket-a/sft-data/`
- **目标路径**: `/ceph/bucket-b/sft-data-copy/`
- **创建时间**: 2026-04-22 16:00:00
- **开始时间**: 2026-04-22 16:00:10
- **结束时间**: 2026-04-22 16:05:11

### ❌ 错误详情
```
通过智研任务导出失败: taskId=98765, status=FAILED, message=source path access denied
```

> 💡 提示：失败任务可通过 retry 接口重试
```

> 🔗 **场景衔接**：查到 `FAILED` 后，请**立即主动调用** `get_hunyuan_data_export_task_log` 获取底层日志，便于准确定位失败原因并给用户明确建议。

---

## get_hunyuan_data_export_task_log

**功能**：获取数据导出任务首个子任务的执行日志，含 DOP / 智研侧的实时执行详情、智研链接（v2 版本），辅助排查失败原因或确认进度。


> ⚠️ 当 `task_id` 齐备时，立即通过 MCP 直接调用本工具。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `task_id` | int | ✅ 必填 | - | 导出任务 ID |

> ℹ️ 本接口后端侧做的是 `ensureManagePermission` 权限校验，无需额外传 `wsid`。

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_export_task_log '{
  "task_id": 2001
}'
```

### 返回字段说明

返回 Markdown 格式的日志信息，核心字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `version` | str | 导出任务版本：`v1`（底层 DOP 拷贝）/ `v2`（底层智研 CEPHFS_COPY） |
| `instanceOutput` | str | 执行实例的详细输出。`v1` 时是 DOP 任务执行详情（含文件数/百分比、大小/百分比、stage 等）；`v2` 时是智研实例原始输出 |
| `zhiyanUrl` | str | 智研任务详情页链接。**仅 `v2` 版本有值**，`v1` 恒为 null |

### 注意事项

1. **只返回首个子任务日志**：当前后端接口仅返回 `taskDetails` 中第一个子任务的日志，若任务含多个子拷贝任务，其他子任务的日志本接口暂不提供。
2. **任务尚未真正启动时无日志**：若任务还是 `PENDING`（未被调度器拉起），`taskDetails` 为空，返回会是"任务尚未真正启动，或暂无可用日志信息"的占位提示。
3. **建议调用时机**：任务状态进入 `RUNNING` / `SUCCEEDED` / `FAILED` 后再调用，最有价值的是 `FAILED` 排障时。

### 返回示例

**v1（DOP 拷贝，执行中）：**

```
# 导出任务 2001 执行日志

- **版本**: v1

### 执行输出

```
DOP任务执行详情: files=<45.23% / 128>, size=<38.70% / 512.00MiB>, status=RUNNING(执行中), desc=执行中, stage=copy
```
```

**v2（智研 CEPHFS_COPY，失败态）：**

```
# 导出任务 2001 执行日志

- **版本**: v2
- **智研任务链接**: https://zhiyan.oa.com/taskflow/execution/xxxx

### 执行输出

```
[2026-04-22 16:04:55] ERROR: source ceph path /ceph/bucket-a/sft-data/ access denied for user xxxxx
[2026-04-22 16:04:55] job terminated with exit code 1
```
```

**任务尚未启动：**

```
# 导出任务 2001 执行日志

> ⚠️ 任务尚未真正启动，或暂无可用日志信息。
> 请确认任务状态是否已进入 RUNNING / SUCCEEDED / FAILED。
```

---

## create_hunyuan_data_cudofs_copy_task

**功能**：创建一个外租卡（CUDOFS）数据拷贝任务，用于在**外租卡存储与文件存储（ceph / nitrofs(hifs)）之间**、或**两个外租卡存储目录之间**进行数据拷贝。


**底层实现**：对接智研任务 "外租卡拷贝"（智研 `task_id=38475`，`TaskType=CUDOFS_COPY`）。后端 `getTaskValues()` 会自动把 `inputPath` / `outputPath` 分别映射为 `SRC_CUDOFS_PATH` / `DST_CUDOFS_PATH`，并追加固定键 `DIST_STORAGE_TYPE=ceph`（该键是智研脚本侧的**内部标识**，与真实拷贝方向无关）。MCP 侧无需也不要暴露 `taskValues` 参数。

> ⚠️ **路由前提（最高优先级）**：只有已判别为"外租卡存储数据拷贝"时才允许使用本工具。判别维度是"外租卡 vs 文件存储"——源或目标**至少有一个是外租卡路径**（`/apdcephfs_wz*` 或 `/cudofs`）就走 cudofs；两端都是文件存储（ceph / nitrofs(hifs)）则用 `create_hunyuan_data_export_task`。

> ⚠️ 当 `wsid` + `input_path` + `output_path` + 已完成外租卡类型判别**全部满足**时，**必须立即通过 MCP 通道直接调用**本工具。

### 支持的拷贝方向（3 种，均路由到本工具）

| 方向 | 典型路径组合 | 用户表述示例 |
|:---|:---|:---|
| 外租卡 → 文件存储（ceph / nitrofs(hifs)） | `input_path=/apdcephfs_wza2/.../data/` → `output_path=/apdcephfs_jn2/.../backup/` | "把外租卡数据拷贝到 ceph" |
| 文件存储（ceph / nitrofs(hifs)） → 外租卡 | `input_path=/apdcephfs_jn2/.../source/` → `output_path=/apdcephfs_wzb/.../target/` | "把 ceph 数据拷贝到外租卡" |
| 外租卡 → 外租卡 | `input_path=/apdcephfs_wza2/.../x/` → `output_path=/apdcephfs_wzb/.../y/` | "在两个外租卡之间拷贝数据" |

**业务硬约束**：`input_path` 与 `output_path` **至少有一端必须是外租卡存储路径**（命中外租卡识别规则）。若两端都是文件存储路径，请改用 `create_hunyuan_data_export_task`。

**必须确认的信息（最高优先级）：**
1. **wsid（工作空间 ID）**：必填，不能为 0 或空值。用户未提供 → 必须追问。
2. **input_path（源路径）**：必填。可以是外租卡路径也可以是文件存储路径。
3. **output_path（目标路径）**：必填。可以是外租卡路径也可以是文件存储路径。
4. **方向硬约束**：`input_path` 与 `output_path` 至少有一端必须是外租卡。若两端都是文件存储路径 → **应当路由到 `create_hunyuan_data_export_task`**。
5. **owners（责任人）**：可选。若不传，MCP 会用当前 Token 对应的用户自动填充。
6. **不做南京地域强校验**：后端对 `CUDOFS_COPY` 直接跳过 ceph 路径存在性/地域/权限校验。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `wsid` | int | ✅ 必填 | - | 工作空间 ID。不能为 0；会作为 HTTP Header `wsid` 传给后端用于数据隔离 |
| `input_path` | str | ✅ 必填 | - | 源路径。可以是外租卡存储路径（如 `/apdcephfs_wza2/.../`）也可以是 ceph 存储路径 |
| `output_path` | str | ✅ 必填 | - | 目标路径。可以是外租卡存储路径也可以是 ceph 存储路径。**业务硬约束**：与 `input_path` 至少一端是外租卡路径 |
| `name` | str | ❌ 可选 | `null` | 任务名称，便于后续检索与区分 |
| `owners` | list[str] | ❌ 可选 | `null` | 责任人 RTX 列表。不传时 MCP 会自动使用当前 Token 对应的用户 |

> ℹ️ **不需要传的字段**：`taskType`（MCP 层固定为 `CUDOFS_COPY`）、`source`（固定为 `TAIJI_WEB`）、`taskValues`（由 datax 后端自动注入 `SRC_CUDOFS_PATH` / `DST_CUDOFS_PATH` / `DIST_STORAGE_TYPE=ceph`）。

### 调用示例

**示例 A：外租卡 → ceph（最常见的归档场景）**

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_cudofs_copy_task '{
  "wsid": 10103,
  "input_path": "/apdcephfs_wza2/share_303693282/leslizhang/train_data/",
  "output_path": "/apdcephfs_jn2/share_302316223/leslizhang/wz_backup/",
  "name": "外租卡A到南京ceph的归档-001"
}'
```

**示例 B：ceph → 外租卡（从 ceph 把数据推到外租卡）**

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_cudofs_copy_task '{
  "wsid": 10103,
  "input_path": "/apdcephfs_jn2/share_302316223/leslizhang/src_data/",
  "output_path": "/apdcephfs_wzb/share_303693282/leslizhang/restored/",
  "name": "南京ceph到外租卡B的恢复-001"
}'
```

**示例 C：外租卡 → 外租卡（两个外租卡目录之间的同步）**

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_cudofs_copy_task '{
  "wsid": 10103,
  "input_path": "/apdcephfs_wza2/share_303693282/leslizhang/source/",
  "output_path": "/apdcephfs_wzb/share_303693282/leslizhang/mirror/",
  "name": "外租卡A到外租卡B的镜像-001",
  "owners": ["leslizhang"]
}'
```

### 返回字段说明

返回 Markdown 格式的任务详情（`ZhiyanDataTaskInfo`），关键字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | **任务 ID**，后续查询与日志都靠它（对应后端 `ZhiyanDataTask.id`，不同于智研执行记录 ID） |
| `name` | str | 任务名称 |
| `taskStatus` | str | 任务状态，见下方"状态说明" |
| `taskType` | str | 固定为 `CUDOFS_COPY` |
| `inputPath` | str | 源路径。**非 owner/admin 时会脱敏为 `***`** |
| `outputPath` | str | 目标路径。**非 owner/admin 时会脱敏为 `***`** |
| `source` | str | 数据来源，MCP 层固定填 `TAIJI_WEB` |
| `taskId` | int | 智研执行记录 ID（`ZhiyanTaskExecution.id`，与上面的 `id` **不是同一个**）。**创建时为 null**，等后端定时器把任务推到智研后才写入 |
| `zhiyanUrl` | str | 智研任务详情页链接。**创建时为 null**，`taskId` 写入后才生成 |
| `creator` | str | 创建人 |
| `ownerList` | list | 责任人列表 |
| `createTime` / `updateTime` / `endTime` | str | 时间线 |
| `message` | str | 失败时的错误信息（仅 `FAILED` 状态有值） |

> ℹ️ **异步语义**：创建接口在后端只做落库（初始 `taskStatus=PENDING`），真正发起智研拷贝由 datax 后端定时器 `refreshStatus()` 异步驱动。因此创建接口返回的 `taskId` / `zhiyanUrl` **此时几乎总是 null**，调用方需要稍后通过 `get_hunyuan_data_cudofs_copy_task` 轮询才能看到它们被填充。

### 返回示例

**创建成功（初始 PENDING）：**

```
# 外租卡拷贝任务详情 (ID: 62)

- **任务名称**: 外租卡A到南京ceph的归档-001
- **状态**: ⏳ PENDING
- **任务类型**: CUDOFS_COPY
- **源路径**: `/apdcephfs_wza2/share_303693282/leslizhang/train_data/`
- **目标路径**: `/apdcephfs_jn2/share_302316223/leslizhang/wz_backup/`
- **数据来源**: TAIJI_WEB
- **责任人**: leslizhang
- **创建人**: leslizhang
- **创建时间**: 2026-04-24 16:00:00

> ⏳ 任务刚落库，正在等待后台调度器拉起智研拷贝任务；稍后可再次调用 get_hunyuan_data_cudofs_copy_task 查看进展。
```

---

## get_hunyuan_data_cudofs_copy_task

**功能**：根据任务 ID 查询单个外租卡拷贝任务的详情与执行状态，用于轮询进度或定位失败原因。


> ⚠️ 当 `task_id` + `wsid` 齐备时，立即通过 MCP 直接调用本工具。查到 `FAILED` 后**主动续跑** `get_hunyuan_data_cudofs_copy_task_log` 取日志。

### 状态说明

| 状态值 | Emoji | 说明 | 下一步建议 |
|--------|-------|------|-----------|
| `PENDING` | ⏳ | 待执行。任务已落库，等待后端定时器拉起智研任务 | 稍后重新调用本工具；此时**不要**去查日志（会报"智研任务执行记录不存在"） |
| `RUNNING` | 🔄 | 执行中。智研任务已启动，可点击 `zhiyanUrl` 查看实时进度 | 可调用 `get_hunyuan_data_cudofs_copy_task_log` 查看实时输出 |
| `SUCCEEDED` | ✅ | **终态**：执行成功 | 向用户展示产出路径 `outputPath` |
| `FAILED` | ❌ | **终态**：执行失败 | **立即调用 `get_hunyuan_data_cudofs_copy_task_log` 查看日志和智研链接**定位原因 |

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `task_id` | int | ✅ 必填 | - | 外租卡拷贝任务 ID（`create_hunyuan_data_cudofs_copy_task` 返回的 `id`） |
| `wsid` | int | ✅ 必填 | `0` | 工作空间 ID；不能为 0。后端通过 Header `wsid` 做工作空间隔离，必须与创建时一致 |

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_cudofs_copy_task '{
  "task_id": 62,
  "wsid": 10103
}'
```

### 返回字段说明

返回 Markdown 格式的任务详情，字段与 `create_hunyuan_data_cudofs_copy_task` 相同。额外在不同状态下补充信息：
- **PENDING 态**：附加 "任务刚落库，正在等待后台调度器拉起智研拷贝任务" 提示
- **RUNNING 态**：`taskId` / `zhiyanUrl` 已填充，可点进智研页面
- **SUCCEEDED 态**：单独展示产出路径
- **FAILED 态**：展示 `message` 错误详情代码块，并给出智研链接用于深入排查
- **路径脱敏**：若当前账号非 owner/admin，`inputPath` / `outputPath` 显示为 `***` 并附提示

### 返回示例

**执行中（taskId / zhiyanUrl 已填充）：**

```
# 外租卡拷贝任务详情 (ID: 62)

- **任务名称**: 外租卡A到南京ceph的归档-001
- **状态**: 🔄 RUNNING
- **任务类型**: CUDOFS_COPY
- **源路径**: `/apdcephfs_wza2/share_303693282/leslizhang/train_data/`
- **目标路径**: `/apdcephfs_jn2/share_302316223/leslizhang/wz_backup/`
- **数据来源**: TAIJI_WEB
- **责任人**: leslizhang
- **创建人**: leslizhang
- **创建时间**: 2026-04-24 16:00:00
- **更新时间**: 2026-04-24 16:00:20
- **智研执行记录 ID**: 987654
- **智研任务链接**: [https://zhiyan.woa.com/operate/xxx/task/#/task/result/38475?sessionID=xxxxx](https://zhiyan.woa.com/operate/xxx/task/#/task/result/38475?sessionID=xxxxx)
```

**执行成功：**

```
# 外租卡拷贝任务详情 (ID: 62)

- **任务名称**: 外租卡A到南京ceph的归档-001
- **状态**: ✅ SUCCEEDED
- **任务类型**: CUDOFS_COPY
- **源路径**: `/apdcephfs_wza2/share_303693282/leslizhang/train_data/`
- **目标路径**: `/apdcephfs_jn2/share_302316223/leslizhang/wz_backup/`
- **数据来源**: TAIJI_WEB
- **创建时间**: 2026-04-24 16:00:00
- **更新时间**: 2026-04-24 16:08:30
- **结束时间**: 2026-04-24 16:08:30
- **智研执行记录 ID**: 987654
- **智研任务链接**: [https://zhiyan.woa.com/operate/xxx/task/#/task/result/38475?sessionID=xxxxx](https://zhiyan.woa.com/operate/xxx/task/#/task/result/38475?sessionID=xxxxx)

### ✅ 任务已完成
产出路径: `/apdcephfs_jn2/share_302316223/leslizhang/wz_backup/`
```

**执行失败：**

```
# 外租卡拷贝任务详情 (ID: 62)

- **任务名称**: 外租卡A到南京ceph的归档-001
- **状态**: ❌ FAILED
- **任务类型**: CUDOFS_COPY
- **源路径**: `/apdcephfs_wza2/share_303693282/leslizhang/train_data/`
- **目标路径**: `/apdcephfs_jn2/share_302316223/leslizhang/wz_backup/`
- **智研任务链接**: [https://zhiyan.woa.com/operate/xxx/task/#/task/result/38475?sessionID=xxxxx](https://zhiyan.woa.com/operate/xxx/task/#/task/result/38475?sessionID=xxxxx)

### ❌ 错误详情
```
发起智研任务失败：source ceph path /apdcephfs_wza2/.../train_data/ access denied
```

> 💡 进一步排查可查看智研任务页面：https://zhiyan.woa.com/operate/xxx/task/#/task/result/38475?sessionID=xxxxx
```

> 🔗 **场景衔接**：查到 `FAILED` 后，请**立即主动调用** `get_hunyuan_data_cudofs_copy_task_log` 获取底层日志。

---

## get_hunyuan_data_cudofs_copy_task_log

**功能**：获取外租卡拷贝任务的执行日志（智研最新 checkpoint 的首个实例输出 + 智研任务详情页链接），辅助排查失败原因或确认进度。


> ⚠️ 当 `task_id` + `wsid` 齐备时，立即通过 MCP 直接调用本工具。

### 参数表

| 参数名 | 类型 | 是否必需 | 默认值 | 描述 |
|--------|------|----------|--------|------|
| `task_id` | int | ✅ 必填 | - | 外租卡拷贝任务 ID |
| `wsid` | int | ✅ 必填 | `0` | 工作空间 ID；不能为 0 |

> ℹ️ **权限要求**：后端会做 `ensureManagePermission` 校验，**当前用户必须是 admin 或任务 owner**，否则返回权限错误。

### 调用示例

```bash
python3 scripts/connect_mcp.py call get_hunyuan_data_cudofs_copy_task_log '{
  "task_id": 62,
  "wsid": 10103
}'
```

### 返回字段说明

返回 Markdown 格式的日志信息，核心字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `instance_output` | str | 智研任务最新 checkpoint 的第一个实例输出。任务刚提交尚未产生 checkpoint 时可能为 null |
| `zhiyan_url` | str | 智研任务详情页链接，可点进去看完整进度、日志、重跑入口 |

> ⚠️ 外租卡拷贝底层一律走智研任务，因此**不存在** `version=v1/v2` 区分（这是导出任务 `get_hunyuan_data_export_task_log` 独有的概念）。

### 注意事项

1. **前置条件**：本接口依赖任务已真正发起（后端已为记录写入 `taskId`）。若任务仍处于 `PENDING` 状态调用，会得到 "智研任务执行记录不存在：id=null" 错误。**请先用 `get_hunyuan_data_cudofs_copy_task` 确认 `taskStatus` 不是 `PENDING` 再调此接口。**
2. **权限要求**：当前用户必须是 admin 或任务 owner，否则 `ensureManagePermission` 会拒绝。
3. **任务刚提交无日志**：刚进入 `RUNNING` 但尚未产生 checkpoint 时，`instance_output` 可能为 null，此时可以等待几秒后重试。
4. **建议调用时机**：任务状态进入 `RUNNING` / `SUCCEEDED` / `FAILED` 后再调用，最有价值的是 `FAILED` 排障时。

### 返回示例

**RUNNING 状态（有实例输出）：**

```
# 外租卡拷贝任务 62 执行日志

- **智研任务链接**: [https://zhiyan.woa.com/operate/xxx/task/#/task/result/38475?sessionID=xxxxx](https://zhiyan.woa.com/operate/xxx/task/#/task/result/38475?sessionID=xxxxx)

### 执行输出

```
[2026-04-24 16:01:12] INFO  start cudofs copy: src=/apdcephfs_wza2/..., dst=/apdcephfs_jn2/...
[2026-04-24 16:01:20] INFO  copied 128 files, total 512.00MB, progress 45.23%
[2026-04-24 16:01:35] INFO  stage=copy, status=RUNNING
```
```

**FAILED 状态（有错误信息）：**

```
# 外租卡拷贝任务 62 执行日志

- **智研任务链接**: [https://zhiyan.woa.com/operate/xxx/task/#/task/result/38475?sessionID=xxxxx](https://zhiyan.woa.com/operate/xxx/task/#/task/result/38475?sessionID=xxxxx)

### 执行输出

```
[2026-04-24 16:04:55] ERROR: source path /apdcephfs_wza2/.../train_data/ access denied for user xxxxx
[2026-04-24 16:04:55] job terminated with exit code 1
```
```

**任务尚未启动（PENDING）：**

```
# 外租卡拷贝任务 62 执行日志

> ⚠️ 任务尚未真正启动或暂无可用日志信息。
> 请先调用 get_hunyuan_data_cudofs_copy_task 确认 taskStatus 不是 PENDING，再尝试获取日志。
```
