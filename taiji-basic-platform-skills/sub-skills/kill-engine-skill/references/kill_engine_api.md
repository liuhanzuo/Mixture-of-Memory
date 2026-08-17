## query_shared_killengine_kill_rules

查询查杀规则列表，支持按应用组、名称、类型、状态、作用域等多条件分页。

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_id` | string | ❌ | `""` | 返回作用于该应用组的规则（scope=all/business_tag，或 app_group_id 命中 values）|
| `rule_name` | string | ❌ | `""` | 名称模糊匹配 |
| `scope_type` | string | ❌ | `""` | `all` / `business_flag` / `business_tag` |
| `status` | string | ❌ | 无 | `enabled`=启用，`disabled`=停用，枚举 `["enabled", "disabled"]` |
| `task_types` | string[] | ❌ | 无 | 任务类型过滤，如 `["train", "inf"]` |
| `training_task_subtypes` | string[] | ❌ | 无 | 训练任务子类型过滤，可选 `experiment` / `debug` / `production` / `debug_mode` |
| `inference_subtypes` | string[] | ❌ | 无 | 推理任务子类型过滤 |
| `resource_types` | string[] | ❌ | 无 | 资源类型过滤，可选 `private` / `public` / `elastic` / `mixing` |
| `locations` | string[] | ❌ | 无 | 地域过滤，如 `["sh", "gz", "qy"]` |
| `page` / `page_size` | integer | ❌ | `1` / `10` | 分页 |

**调用示例：**

```bash
python3 scripts/connect_mcp.py call query_shared_killengine_kill_rules '{
  "app_group_id": "TaiJi_HYAide_HYapp_EXTRA",
  "page": 1,
  "page_size": 10
}'
```

**返回**：Markdown 表格（规则ID / 名称 / 类型 / 状态 / 作用域 / 优先级 / 版本 / 条件摘要 / 创建人 / 更新时间）+ 总数与分页信息。

**参数使用约束（重要）：**
- 本工具不接收 `wsid`；用户 prompt 中的 wsid 仅作上下文，不要透传。
- 按用户明示过滤一次到位：全局规则传 `scope_type="all"`，已启用传 `status="enabled"`，已停用传 `status="disabled"`，推理任务传 `task_types=["inf"]`，训练任务传 `task_types=["train"]`。
- 用户已明确 `scope_type` / `status` / `task_types` / `rule_name` 等过滤条件时，即使 prompt 含 `wsid` 或应用组上下文，也不得调用 `list_user_groups` / `query_shared_resources_app_group_list` 做额外验证；直接使用本工具一次查询。
- 未要求“前 N 条 / 最近 / 第一条 / 翻页”时，不要为了“更全”主动追加 `page_size=50` 或无过滤二次查询；工具默认分页足够回答概览。普通“有哪些规则”可用 `page=1,page_size=20`，但带 `scope_type/status/task_types` 的过滤查询默认不传分页。
- 用户说“GPU 低利用率/低利用率规则”时，用 `rule_name="低利用率"` 做名称模糊匹配。
- 用户要“最新一条/第一条规则的完整详情、条件和动作”时，列表查询只用于取 `rule_id`（用 `page=1,page_size=1`），随后必须调用 `get_shared_killengine_kill_rule`。

---

## get_shared_killengine_kill_rule

查询单条查杀规则的完整定义。

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `rule_id` | string | ✅ | 规则 ID（来自 `query_shared_killengine_kill_rules`） |

**返回**：规则详情（名称/类型/状态/作用域各维度/优先级/版本/条件摘要/告警与查杀动作/描述/操作人），并附 `conditionGroups`/`actionConfig`/`matchConfig` 原始 JSON。

---

## create_shared_killengine_kill_rule

创建一条查杀规则（**写**，需系统管理员或对应应用组管理员权限）。

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `rule_name` | string | ✅ | 无 | 规则名称 |
| `scope_type` | string | ✅ | 无 | `all` / `business_flag` / `business_tag` |
| `scope_config` | object | ✅ | 无 | 作用域配置（见下；scope=flag/tag 时 `values` 必填）|
| `condition_groups` | array | ✅ | 无 | 触发条件（AND/OR，≤2 层，至少一个条件）|
| `action_config` | object | ✅ | 无 | `alarm` + `kill` |
| `match_config` | object | ❌ | 无 | `on_metric_null` 等（临时/清理规则可不传） |
| `description` | string | ❌ | `""` | 规则描述 |
| `priority` | integer | ❌ | `0` | 规则优先级，取值范围 [0, 999] |

**调用示例（最小参数；GPU 利用率低于 10% 持续 2 小时，只告警不查杀）：**

```bash
python3 scripts/connect_mcp.py call create_shared_killengine_kill_rule '{
  "rule_name": "GPU低利用率临时告警规则",
  "scope_type": "business_flag",
  "scope_config": {"values": ["TaiJi_HYAide_HYapp_EXTRA2"]},
  "condition_groups": [
    {
      "logic": "AND",
      "conditions": [
        {"metric": "gpuUtil", "operator": "<", "threshold": 10, "unit": "percent", "duration": 2, "duration_unit": "hour"}
      ]
    }
  ],
  "action_config": {
    "alarm": {"enabled": true},
    "kill": {"enabled": false}
  }
}'
```

**创建参数白名单（重要）：**
- `scope_config`：用户只说“应用组 X”时，只能传 `{"values":["X"]}`，不要默认追加 `task_types`、子类型、资源类型、地域。
- `action_config`：只告警不查杀时，只能传 `{"alarm":{"enabled":true},"kill":{"enabled":false}}`；仅当用户明确指定接收人/渠道/告警间隔/kill 接收人时才补对应字段。
- 临时规则/评测清理规则默认不传 `match_config`、`description`、`priority`、告警间隔、最大次数、接收人类型或渠道。

**返回**：创建成功，含 `ruleId` / `version` / `createdAt`。

---

## update_shared_killengine_kill_rule

修改一条查杀规则（**写**，全量更新，生成新版本 version+1，需写权限）。

> ⚠️ **全量更新**：需提供完整规则定义。**建议先调用 `get_shared_killengine_kill_rule` 取回当前定义，改动后整体提交**，避免遗漏字段。

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `rule_id` | string | ✅ | 规则 ID |
| `rule_name` / `scope_type` / `scope_config` / `condition_groups` / `action_config` / `match_config` | — | ✅ | 同 `create_shared_killengine_kill_rule`（全量） |
| `description` | string | ❌ | 规则描述 |
| `priority` | integer | ❌ | 规则优先级，取值范围 [0, 999] |
| `change_summary` | string | ❌ | 本次变更说明，记录到规则版本历史 |

**返回**：修改成功，含 `ruleId` / 新 `version` / `createdAt`。

---

## set_shared_killengine_kill_rule_status

启用或停用一条查杀规则（**写**，需写权限）。

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `rule_id` | string | ✅ | 规则 ID |
| `status` | string | ✅ | `enabled`=启用，`disabled`=停用，枚举 `["enabled", "disabled"]` |

**返回**：操作成功，含规则 ID / 当前状态 / 版本。

---

## delete_shared_killengine_kill_rule

删除一条查杀规则（**写，不可恢复**；`preset` 预设规则禁止删除；需写权限）。

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `rule_id` | string | ✅ | 规则 ID（preset 规则会被后端拒绝，返回 40301） |

> ⚠️ 删除前务必与用户确认；该操作不可恢复。

---

## query_shared_killengine_kill_records

查询查杀记录（已被查杀引擎处理/查杀的任务实例）。

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `app_group_id` | string | ✅ | 无 | 应用组标识（后端强制必填，缺失返回 40001）|
| `start_at` / `end_at` | string | ❌ | `""` | 格式 `2006-01-02 15:04:05`，按查杀时间过滤 |
| `creator` | string | ❌ | `""` | 任务创建人 |
| `instance_id` | string | ❌ | `""` | 实例 ID |
| `kill_strategy` | string | ❌ | `""` | 按触发查杀的规则 ID 过滤，传入 `rule_id`（该列存储触发本条记录的规则 ID）|
| `page` / `page_size` | integer | ❌ | `1` / `10` | 分页 |

**调用示例：**

```bash
python3 scripts/connect_mcp.py call query_shared_killengine_kill_records '{
  "app_group_id": "TaiJi_HYAide_HYapp_EXTRA",
  "start_at": "2026-03-04 00:00:00",
  "end_at": "2026-06-04 00:00:00",
  "page": 1,
  "page_size": 10
}'
```

**返回**：Markdown 表格（实例UUID / 任务名 / 任务类型 / 创建人 / 卡型 / 卡数 / 地域 / 最大GPU利用率 / 告警次数 / 查杀策略 / 命中规则 / 查杀时间）+ 总数与分页。

**参数使用约束（重要）：**
- 本工具不接收 `wsid`；不要透传用户上下文里的 wsid。
- 用户说“某创建人/谁的任务”时传 `creator`；不要附加 `start_at/end_at/page`，除非用户同时要求“最近/近 X 天/月/前 N 条”。用户要求“最近 N 条”时基于当前时间补动态 `start_at/end_at`，并传 `page=1,page_size=N`。
- 已有明确 `app_group_id`，且用户可通过 `creator` / `kill_strategy` 直接过滤时，不得调用 `list_user_groups` / `query_shared_resources_app_group_list` 做额外验证；直接使用本工具查询。
- 用户要求“先查第一条记录中的 `kill_strategy`，再用它过滤”时，必须执行两次调用：第一次只带 `app_group_id`（及用户明示的时间/创建人条件）并从默认第一页读取第一条；第二次带同一 `app_group_id` 与 `kill_strategy=<第一条记录的规则ID>`。除非用户明确数量/分页，否则这两次都不要附加 `page/page_size`。
