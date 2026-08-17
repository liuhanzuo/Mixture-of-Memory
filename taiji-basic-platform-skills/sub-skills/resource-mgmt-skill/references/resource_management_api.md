## query_shared_resources_app_group_list

**用途**：分页查询当前用户可见/可管理的 GPU 应用组列表。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `keyword` | string | ❌ | — | 关键字模糊匹配（应用组名称/标识/描述） |
| `type` | string | ❌ | — | GpuGroupType 类型过滤（如 `NORMAL` / `HYAIDE` / `RENT`） |
| `tag` | string | ❌ | — | 按 tag 标签过滤 |
| `module` | string | ❌ | — | 按业务模块过滤 |
| `is_admin` | boolean | ❌ | `false` | 只返回当前用户管理的应用组 |
| `is_usable` | boolean | ❌ | `false` | 只返回当前用户可用的应用组 |
| `excludes` | array[string] | ❌ | — | 排除的 `app_group_id` 列表 |
| `order_by` | string | ❌ | `created` | 排序字段 |
| `desc` | boolean | ❌ | `true` | 是否降序 |
| `page` | integer | ❌ | `1` | 页码，1-based |
| `page_size` | integer | ❌ | `10` | 每页数量（上限 3000） |

### 调用示例

```json
{"is_usable": true, "page": 1, "page_size": 20}
```

"我有哪些/可用哪些应用组"场景传`is_usable=true`；"平台上都有哪些应用组"场景可不加过滤。

### 返回

> 返回项里应用组标识字段是**`id`**（不是 `app_group_id`）；`owners` 是分号分隔字符串。

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "items": [
      {
        "id": "TaiJi_HYAide_rentTest_CQ_A100L",
        "type": "jizhi",
        "name": "可读名称",
        "role": "OWNER",
        "note": "说明",
        "created": "2026-05-28T09:13:00.000+00:00",
        "creator": "someone",
        "owners": "userA;userB",
        "tag": "hunyuan",
        "module": null,
        "app_group_type": null
      }
    ],
    "page": 1,
    "page_size": 20,
    "total": 128,
    "has_more": true
  }
}
```

> 后续工具需要的应用组标识，取自返回项的 `id` 字段。

---

## get_shared_resources_app_group_detail

**用途**：查询单个应用组的详细配置（负责人、成员、描述、GPU/CPU 额度配置等）。**在调用 `update_shared_resources_user_capacity` 前必须先调此工具**查现状。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_id` | string | ✅ | — | 应用组标识 |
| `with_budget` | integer | ❌ | `0` | 是否返回 OBS 预算/卡时（`1` = 返回，会显著增加 RT） |

### 调用示例

```json
{"app_group_id": "TaiJi_HYAide_rentTest_CQ_A100L"}
```

用户给了明确 `app_group_id` 时直接查，只传 `app_group_id`；即使 ID 看起来不存在也先调详情拿404。

### 返回关键字段

- `app_group_id` / `readable_name` / `description` / `business_department` / `business_module`
- `owner_list` / `member_list`（RTX 列表）
- `user_gpu_config`：含 `default_gpu_capacity`（默认卡数map）、`user_capacity_list`（按用户覆盖 list）、`dynamic_capacity_list`（动态额度 list）
- `tag` / `module` / `resource_type` / `resource_source` / `is_shared_card` / `avg_gpu_ratio_7d`
- `with_budget=1` 时追加：`obs_budget` / `total_gpu_time_budget`

---

## query_shared_resources_gpu_info_batch

**用途**：批量查询多个应用组的 GPU 资源汇总（总卡/已用/空闲），一次网络请求拿多组数据。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_ids` | array[string] | ✅ | — | 应用组标识列表（建议 ≤50）；单个也必须传数组 |

### 调用示例

```json
{"app_group_ids": ["TaiJi_HYAide_rentTest_CQ_A100L", "TaiJi_HYAide_HYapp"]}
```

"对比 A、B 的 GPU 资源"场景一次传多个`app_group_ids`；"某应用组 GPU卡情况/空卡/排队"场景传单个数组。

### 返回关键字段

> `data` 是**数组**（不是 `data.items`），每元素对应一个应用组；卡型明细在字段 `noraml`（后端原样拼写）下。

- `data[*].id`：应用组标识
- `data[*].noraml[]`：卡型明细数组，每项 `{card_type, total, used, available, waiting, applying, unavailable, quota_type, business_tag, location, is_rent_card, rent_card_manufacturer}`
  - `waiting` 字段即可回答"是否有排队"

---

## query_shared_resources_gpu_resource_info

**用途**：查询单个应用组按**集群+卡型**拆分的 GPU 资源明细，**含被屏蔽节点**。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_id` | string | ✅ | — | 应用组标识 |

### 调用示例

```json
{"app_group_id": "TaiJi_HYAide_rentTest_CQ_A100L"}
```

### 返回关键字段

- `data.resource_list[]`：按集群拆分，每项 `{cluster_name, gpu_items[]}`
- `gpu_items[]`：每项 `{card_type, total_gpu, actual_gpu, using_gpu, available_gpu, unavailable_gpu, not_ready_gpu, zombie_gpu, total_vgpu, quota_type, business_tag, spec[], unavailable_nodes[]}`
- `unavailable_nodes[]`：被屏蔽节点明细，每项 `{ip, reason, card_num, reasons[]}`；`reasons[]` 含 `{current_state, rule_name, created_at, estimated_at, duty_operator, related_tasks}`

---

## query_shared_resources_gpu_rent_info

**用途**：查询单个应用组**含租借**的 GPU 使用情况（自有+租入+租出）。若用户问的是单个应用组配额使用概览，优先用本工具。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_id` | string | ✅ | — | 应用组标识 |

### 调用示例

```json
{"app_group_id": "TaiJi_HYAide_rentTest_CQ_A100L"}
```

### 返回关键字段

>与 `gpu_info_batch` 返回同构：`data` 是数组，租借关系通过 `is_rent_card` 字段区分。

- `data[*].noraml[]`：每项额外含 `{is_rent_card, rent_card_manufacturer, rent_storage_type, rent_storage_region, rent_storage_node, next_recycle_time}`

---

## query_shared_resources_gpu_job_list

**用途**：分页查询应用组下的 GPU 任务实例列表，支持多维度过滤。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_id` | string | ✅ | — | 应用组标识 |
| `status` | array[string] | ❌ | — | 枚举 `running`/`waiting`/`finished`/`pending`；必须传数组 |
| `quota_type` | array[string] | ❌ | — | 枚举 `private`/`public`/`discount`/`elastic`/`mixing` |
| `only_mine` | boolean | ❌ | `false` | 只看我提交的任务 |
| `time_range_column` | string | ❌ | `createTime` | 时间范围作用列：`createTime`或 `endTime` |
| `start_time` | string | ❌ | — | 格式 `yyyy-MM-dd HH:mm:ss` |
| `end_time` | string | ❌ | — | 格式 `yyyy-MM-dd HH:mm:ss` |
| `keyword` | string | ❌ | — | 任务名称/创建人关键字模糊匹配 |
| `task_source` | array[string] | ❌ | — | 枚举 `hunyuan_aide`/`large_model_platform`/`taiji_general_platform` |
| `task_type` | array[string] | ❌ | — | 枚举 `train`/`inf` |
| `creator` | array[string] | ❌ | — | 提交人 RTX 列表过滤 |
| `task_queuing_priority` | array[string] | ❌ | — | 如 `["P1", "P2"]` |
| `card_type_set` | array[string] | ❌ | — | 卡型过滤，如 `["A100L", "H20"]` |
| `location` | array[string] | ❌ | — | 地域过滤，如 `["cq", "sh"]` |
| `order_by` | string | ❌ | — | 排序字段 |
| `desc` | boolean | ❌ | `false` | 是否降序 |
| `page` | integer | ❌ | `1` | 页码，1-based |
| `page_size` | integer | ❌ | `10` | 每页数量 |

### 调用示例

```json
{"app_group_id": "TaiJi_HYAide_rentTest_CQ_A100L", "status": ["running"], "page": 1, "page_size": 10}
```

查排队任务：`"status": ["waiting"]`；查训练任务：`"task_type": ["train"]`。

### 返回关键字段

> 实例标识字段是 **`id`**；写操作（优先级/置顶）入参 `instance_uuid` 的值取自此`id`。

- `data.items[*]`：`{id, jz_instance_id, jz_task_flag, name, creator, task_type, task_source, card_type, quota_type, gpu_num, location, created, business_flag, urls[], labels[]}`
  - `urls[]`：`{label, url}` 页面链接（**严禁自拼**）

---

## query_shared_resources_cpu_job_list

**用途**：分页查询应用组的 CPU 任务列表。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_id` | string | ✅ | — | 应用组标识 |
| `status` | array[string] | ❌ | — | 任务状态过滤 |
| `quota_type` | array[string] | ❌ | — | 资源类型过滤 |
| `only_mine` | boolean | ❌ | `false` | 只看我提交的 |
| `time_range_column` | string | ❌ | `createTime` | `createTime` 或 `endTime` |
| `start_time` | string | ❌ | — | 格式 `yyyy-MM-dd HH:mm:ss` |
| `end_time` | string | ❌ | — | 格式 `yyyy-MM-dd HH:mm:ss` |
| `keyword` | string | ❌ | — | 模糊匹配 |
| `task_source` | array[string] | ❌ | — | 来源平台过滤 |
| `task_type` | array[string] | ❌ | — | 任务类型过滤 |
| `creator` | array[string] | ❌ | — | 提交人过滤 |
| `location` | array[string] | ❌ | — | 地域过滤 |
| `order_by` | string | ❌ | — | 排序字段 |
| `desc` | boolean | ❌ | `false` | 是否降序 |
| `page` | integer | ❌ | `1` | 1-based |
| `page_size` | integer | ❌ | `10` | 每页数量 |

### 返回关键字段

- `data.items[*]`：`{instance_uuid, task_name, creator, status, cpu_cores, host_num, host_cpu_num, memory, cluster, start_time}`

---

## query_shared_resources_user_gpu_util_rank

**用途**：应用组内按用户聚合的 GPU 利用率排行。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_id` | string | ✅ | — | 应用组标识 |
| `status` | array[string] | ❌ | — | 任务状态过滤 |
| `only_mine` | boolean | ❌ | `false` | 只看我的|
| `start_time` | string | ❌ | 最近 7 天 | 格式 `yyyy-MM-dd HH:mm:ss` |
| `end_time` | string | ❌ | 现在 | |
| `page` | integer | ❌ | `1` | |
| `page_size` | integer | ❌ | `10` | |

### 返回关键字段

- `data.items[*]`：`{user, avg_gpu_ratio, gpu_num, total_gpu_time, task_count}`

---

## query_shared_resources_job_gpu_ratio

> 🛑 **数据为空立即停止**：返回 `gpuRatio` 为空时说明实例刚启动尚无采样数据，**立即告知用户"实例刚启动，尚无 GPU 利用率数据"**，不得换 `instance_uuid`/`time_range`/时间范围反复重试。如需查运行中任务，改调 `query_shared_resources_gpu_job_list` 仅一次。

**用途**：批量查询任务实例的 GPU 利用率。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `instance_ids` | array[string] | ✅ | — | 实例 ID 列表（单个也传数组） |
| `app_group_id` | string |❌ | — | 应用组标识（可选） |
| `time_range` | string | ❌ | `latest5minutesAvg` | 枚举：`latest5minutesAvg`/`latest30minutesAvg`/`latest1HourAvg`/`latest24HoursAvg`/`todayAvg`/`thisWeekAvg`/`lastWeekAvg`/`thisMonthAvg`/`lastMonthAvg`/`sinceTaskStartAvg` |

### 调用示例

```json
{"instance_ids": ["i-abc123", "i-def456"], "time_range": "latest1HourAvg"}
```

### 返回关键字段

- `data.items[*]`：`{instance_id, avg_gpu_ratio, mixed_gpu_ratio, sample_count}`
- 某instance 不存在时返回 `avg_gpu_ratio: null`

---

## query_shared_resources_job_mfu

**用途**：批量查询任务实例的 MFU（模型算力利用率）。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `instance_ids` | array[string] | ✅ | — | 实例 ID 列表 |
| `app_group_id` | string | ❌ | — | 应用组标识（可选） |
| `time_range` | string | ❌ | `latest5minutesAvg` | 同`job_gpu_ratio` |

### 返回关键字段

- `data.items[*]`：`{instance_id, avg_mfu, sample_count}`
- 非训练或数据缺失时 `avg_mfu: null`

---

## query_shared_resources_task_resource_detail

**用途**：分页查询任务实例级的资源占用明细变更（流水记录）。

>⚠️ 与 `task_resource_usage` 区分：本工具是资源**变动流水**，问"资源怎么变动"用本工具；问"消耗多少卡时/利用率"用 `task_resource_usage`。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_ids` | array[string] | ✅ | — | 应用组标识列表；单个也必须传数组 |
| `start_time` | string | ❌ | 最近 7 天 | 格式 `yyyy-MM-dd HH:mm:ss` |
| `end_time` | string | ❌ | 现在 | |
| `card_type` | string | ❌ | — | 卡型过滤 |
| `region` | string | ❌ | — | 地域过滤（用`region` 不用 `location`） |
| `resource_category` | string | ❌ | — | 资源类别过滤 |
| `change_type` | integer | ❌ | `0` | 变更类型过滤；`0`=不筛选 |
| `creator` | string | ❌ | — | 提交人 RTX |
| `only_mine` | boolean | ❌ | `false` | |
| `order_by` | string | ❌ | `update_time` | |
| `desc` | boolean | ❌ | `false` | 查最近一次时传 `true` |
| `page` | integer | ❌ | `1` | |
| `page_size` | integer | ❌ | `10` | |

### 返回关键字段

- `data.items[*]`：`{instance_uuid, task_name, card_type, gpu_num, change_type, change_time, creator, location, cluster}`

---

## query_shared_resources_quota_transfer_log

**用途**：分页查询应用组配额级的额度腾挪历史。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_ids` | array[string] | ✅ | — | 单个也必须传数组 |
| `start_time` | string | ❌ | 最近 30 天 | 格式 `yyyy-MM-dd HH:mm:ss` |
| `end_time` | string | ❌ | 现在 | |
| `card_type` | string | ❌ | — | |
| `region` | string | ❌ | — | |
| `change_type` | integer | ❌ | `0` | `0`=不筛选，`1`=新增，`-1`=扣减 |
| `creator` | string | ❌ | — | |
| `only_mine` | boolean | ❌ | `false` | |
| `order_by` | string | ❌ | `update_time` | |
| `desc` | boolean | ❌ | `false` | |
| `page` | integer | ❌ | `1` | |
| `page_size` | integer | ❌ | `10` | |

### 返回关键字段

- `data.items[*]`：`{change_type/transfer_type, card_type, gpu_num, operator/creator, update_time/time, from_app_group_id, to_app_group_id, remark}`

---

## query_shared_resources_task_resource_usage

**用途**：分页查询任务级**卡时/GPU 利用率**消耗统计。

> ⚠️ 与 `task_resource_detail` 区分：本工具是**消耗统计**（卡时/利用率聚合），问"消耗多少卡时"用本工具。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_ids` | array[string] | ✅ | — | 单个也必须传数组 |
| `start_time` | string | ❌ | 前7 天 | 格式 `yyyy-MM-dd`（**纯日期**） |
| `end_time` | string | ❌ | 当天 | 格式 `yyyy-MM-dd` |
| `page` | integer | ❌ | `1` | |
| `page_size` | integer | ❌ | `20` | |

### 调用示例

```json
{"app_group_ids": ["TaiJi_HYAide_HYapp"], "start_time": "2026-07-01", "end_time": "2026-07-06", "page": 1, "page_size": 20}
```

### 返回关键字段

- `data.items[*]`：`{app_group_id, task_flag, readable_name, instance_id, rtx, task_type, gpu_num, total_gpu_time, range_total_gpu_time, avg_gpu_ratio, range_avg_gpu_ratio, elasticity, create_time, running_time}`
- 利用率为按卡时加权平均；`range_*` 为查询时间窗内的聚合。

---

## query_shared_resources_app_group_waiting_queue

**用途**：查询应用组等待队列（排队任务）。内置`status=waiting`，默认按 `taskPriority` 降序。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_id` | string | ✅ | — | |
| `task_queuing_priority` | array[string] | ❌ | — | 枚举 `P1`/`P2`/`P3`；单个也传数组 |
| `only_mine` | boolean | ❌ | `false` | |
| `keyword` | string | ❌ | — | |
| `creator` | array[string] | ❌ | — | |
| `order_by` | string | ❌ | `taskPriority` | |
| `desc` | boolean | ❌ | `true` | |
| `page` | integer | ❌ | `1` | |
| `page_size` | integer | ❌ | `10` | |

### 返回关键字段

- 同 `gpu_job_list` 的任务项结构；写操作 `instance_uuid` 取返回项`id`

---

## query_shared_resources_gpu_cluster_resource

**用途**：查询应用组 GPU 集群物理资源及被屏蔽节点明细。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_id` | string | ✅ | — | |

### 返回关键字段

- 同 `query_shared_resources_gpu_resource_info`，含 `unavailable_nodes[]` 被屏蔽节点明细

---

## query_shared_resources_user_app_groups

**用途**：查询当前用户 OWNER 角色的应用组列表。

> ⚠️ **只返回 `OWNER` 角色的应用组**，MEMBER 级别的权限不在此列表出现。默认返回第一页（`page_size=10`），可通过 `page`/`page_size` 翻页。`keyword` 并非所有接口版本都支持模糊匹配，如果不能精准定位建议翻页后确认。

如果该工具未返回目标应用组，但用户知道 `app_group_id`，应再调 `get_shared_resources_app_group_detail` 确认是否存在 MEMBER 或其他级别的权限。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `keyword` | string | ❌ | — | 关键字模糊匹配（应用组名称/标识） |
| `is_admin` | boolean | ❌ | `false` | 只返回我管理的应用组 |
| `is_usable` | boolean | ❌ | `true` | 只返回我可用的应用组 |
| `page` | integer | ❌ | `1` | 页码，从 1 开始 |
| `page_size` | integer | ❌ | `10` | 每页数量，上限 3000 |

### 返回关键字段

- `data.items[*].id`：应用组标识
- `data.total`：总条数
- `data.has_more`：是否还有更多页

---

## validate_shared_resources_sub_business_bindable

**用途**：父子应用组绑定**前置校验**（纯读）。**在调用 `bind_shared_resources_sub_business` 前必须先调此工具**。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `deliver_business` | string | ✅ | — | 父应用组 |
| `subbusiness_flag_list` | array[string] | ✅ | — | 待校验的子应用组列表（≤20） |
| `location` | string | ✅ | — | 地域|
| `card_type` | string | ✅ | — | 卡型 |
| `retain_quota_to_sub` | boolean | ❌ | `false` | |

### 返回关键字段

- `data.can_bind`：boolean
- `data.reason`：不可绑定的原因
- `data.details[*]`：每个子应用组的校验结果

---

## bind_shared_resources_sub_business

**用途**：**写操作** — 绑定父子应用组（含额度腾挪）。

### 前置条件

必须先调 `validate_shared_resources_sub_business_bindable` 校验通过。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `deliver_business` | string | ✅ | — | 父应用组 |
| `subbusiness_flag_list` | array[string] | ✅ | — | 子应用组列表（≤20） |
| `location` | string | ✅ | — | 地域 |
| `card_type` | string | ✅ | — | 卡型 |
| `retain_quota_to_sub` | boolean | ❌ | `false` | |

>本接口**没有** `reason` / `gpu_num_per_sub` 参数。

### 确认要求

4核心参数（`deliver_business`/`subbusiness_flag_list`/`location`/`card_type`）列给用户复核；严禁跨地域/跨卡型混用；`subbusiness_flag_list.length > 5` 时需额外强调批量影响。

### 返回

`{code: 0, data: {activity_id, bound_subs}}`

---

## unbind_shared_resources_sub_business

**用途**：**写操作** — 解绑父子应用组。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `deliver_business` | string | ✅ | — | 父应用组 |
| `subbusiness_flag_list` | array[string] | ✅ | — | 子应用组列表 |
| `location` | string | ✅ | — | |
| `card_type` | string | ✅ | — | |
| `retain_quota_to_sub` | boolean | ❌ | `false` | 是否保留额度至子应用组 |

> 本接口**没有** `reason` 参数。

### 确认要求

4 参数 + `retain_quota_to_sub` 复核给用户；明确告知 `retain_quota_to_sub` 语义。

---

## update_shared_resources_task_queuing_priority

**用途**：**写操作** — 调整等待任务优先级（P1/P2/P3）。仅对 `status=waiting` 任务生效。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `instance_uuid` | string | ✅ | — | 从`gpu_job_list` 返回的 `id` |
| `app_group_id` | string |✅ | — | |
| `target_priority` | string | ✅ | — | `P1`/`P2`/`P3` |
| `reason` | string | ✅ | — | ≥5 字符，禁 `test`/`123` |

### 确认要求

先确认任务 `status=waiting`；4 参数复核；严禁未经负责人同意调整他人任务。

---

## update_shared_resources_task_top_priority

**用途**：**写操作** — 将等待任务**置顶**。仅对 `status=waiting` 生效。

> 与 `update_queuing` 语义不同：调级用queuing，插队置顶用 top，**不要互相代替**。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `instance_uuid` | string | ✅ | — | |
| `app_group_id` | string | ✅ | — | |
| `reason` | string | ✅ | — | ≥5 字符 |

### 确认要求

3 参数复核给用户。

---

## update_shared_resources_user_capacity

**用途**：**写操作** — 更新用户 GPU 个人额度。**全卡型替换**语义（非增量）。

### 前置条件

**必须先调 `get_shared_resources_app_group_detail`** 拿当前 `user_gpu_config`，避免丢失其他卡型。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_id` | string | ✅ | — | |
| `default_gpu_capacity` | object | ❌ | — | 如 `{"A100L": 4, "H20": 8}` |
| `user_capacity_list` | array[object] | ❌ | — | `[{"username": "alice", "gpu_capacity": {"A100L": 8}}]` |
| `dynamic_capacity_list` | array[object] | ❌ | — | 不修改时从详情原样透传 |

> 本接口**无`reason` 参数**。

### 确认要求

4 参数复核；严禁未经被改用户或负责人同意。首次成功后立即停止。

---

## update_shared_resources_app_group_members

**用途**：**写操作** — 更新应用组成员/管理员列表。**全量覆盖**语义（非增量）。

### 权限

仅系统超管或该应用组管理员可操作；仅 `jizhi` 类型支持。

### 参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `app_group_id` | string | ✅ | |
| `modified_owners` | string | ⚠️ | 管理员列表（分号分隔），全量覆盖 |
| `modified_members` | string | ⚠️ | 成员列表（分号分隔），全量覆盖 |

> 至少传一个。添加/删除需先查详情做read-modify-write。

### 确认要求

先`get_detail` 确认当前用户在`owner_list` 内；算出完整目标名单后复核给用户。

---

## create_shared_resources_activity

**用途**：**写操作** — 创建资源审批单（含任务加白）。会真实提交审批流，创建后仅能走审批撤销。

### `type`枚举

| type | 场景 |
|---|---|
| `create_tdw_group` / `join_tdw_group` | TDW 应用组创建/加入 |
| `create_cpu_suanli_group` / `join_cpu_suanli_group` | CPU 算力应用组 |
| `create_gpu_group` / `join_gpu_group` | GPU 应用组 |
| `apply_cpu` / `adjust_cpu` | CPU 配额 |
| `apply_gpu` / `adjust_gpu` | GPU 配额 |
| `apply_hdfs` / `adjust_hdfs` | HDFS 配额 |
| `apply_ceph` / `adjust_ceph` | CEPH 配额 |
| `apply_deepocean_cpu` / `adjust_deepocean_cpu` | Deepocean CPU |
| `task_whitelist` | 任务加白 |

### 参数（按`type` 分组）

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `type` | string | ✅ | 审批类型 |
| `app_group_name` | string | ⚠️ | 新建类必填，≤64 |
| `app_group_alias` | string | ❌ | 应用组别名 |
| `app_group_id` | string | ⚠️ | 已有应用组场景必填 |
| `note` | string | ❌ | 应用组说明，≤255 |
| `remark` | string | ⚠️ | 申请原因，≤255 |
| `cluster` | string | ❌ | 业务归属/集群 |
| `location` | string | ❌ | 地域 |
| `card_type` | string | ❌ | GPU 卡型 |
| `cpu`/`gpu`/`hdfs`/`ceph` | integer | ❌ | 各资源数量 |
| `owners` | string | ❌ | 责任人（分号分隔） |
| `members` | string | ❌ | 应用组成员（分号分隔） |
| `business_department` | string | ❌ | 业务部门 |
| `business_module` | string | ❌ | 业务模块 |
| `applicant` | string | ❌ | **建议不传**，后端从 Token 解析 |
| `task_name` | string | ⚠️ | 加白必填；训练填 instance UUID，推理填服务名 |
| `task_type` | string | ⚠️ | 加白必填；枚举 `train`/`Infer`/`train_instance`/`train_task`/`train_task_id`/`train_user`/`infer_service`/`infer_service_group`/`infer_user` |
| `expire_time` | string | ⚠️ | 加白必填；格式 `yyyy-MM-dd HH:mm:ss`，须晚于当前 |

### 确认要求

按 `type` 列出核心字段复核；任务加白额外复核 `task_name`/`task_type`/`expire_time`。

### 返回

```json
{"code": 0, "data": {"activity_id": 105660, "approval_url": "https://..."}}
```

`approval_url` **仅 `type="task_whitelist"` 返回**，必须展示给用户。
