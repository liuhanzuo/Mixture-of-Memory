# 训练任务 Skill 外部依赖工具速查

> 本文件记录本 skill 直调的跨模块依赖工具。外部跳转（非直调）的工具仅保留跳转说明。

## list_user_workspaces

**用途**：查询当前用户有权限的工作空间列表。用户要查训练任务、任务状态、日志、checkpoint 但没有给 `wsid` 时使用。

**入参**：无必填入参，通常传 `{}` 即可。可选传 `platform`（`hunyuan` / `taiji` / `sft`）按平台过滤。

```json
{}
```

**关键出参**：返回顶层是一个对象，重点字段如下：

| 字段 | 类型 | 含义 |
|---|---|---|
| `username` | string | 当前用户名 |
| `is_admin` | boolean | 是否平台级管理员 |
| `workspaces` | array | 工作空间列表 |
| `hy_basic_wsid_map` | object | 混元基础空间 ID 映射（一般忽略） |

`workspaces` 中每个空间对象重点字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `wsid` | string | 工作空间 ID，也就是后续工具需要的 `wsid` |
| `name` | string | 工作空间名称 |
| `desc` | string | 工作空间描述 |
| `workspace_type` | string | 空间类型（`hunyuan` / `sft` / `taiji` / `general`） |
| `is_admin` | boolean | 当前用户是否是该空间管理员 |
| `managers` | string | 管理员列表，分号分隔 |

**本 skill 中的典型用途**：
1. 用户说"帮我看下训练任务"但没给空间
2. 调 `list_user_workspaces({})`
3. 从返回的 `workspaces` 中整理出候选空间
4. 让用户选择某个 `wsid`
5. 再把该值作为后续 `task_list` / `task_detail` / `task_logs` / `task_ckpt_list` 的 `wsid`

**注意事项**：
- 本工具不接收 `wsid`，不要自行拼参数
- `is_admin=true` 只表示平台级可见范围更大，不代表对每个空间内资源都有实际操作权限
- 用户已经明确给了 `wsid` 时，按 `workspace-skill` 规则，不要重复调这个工具

**调用示例**：
```bash
python3 scripts/connect_mcp.py call list_user_workspaces '{}'
```

---

## query_hunyuan_train_instance_logs

**用途**：查询训练任务实例的训练日志，支持按 Pod、容器、关键词过滤，支持分页和排序。**排查"任务为什么失败/被终止"时可直调此工具查看日志中的报错信息**，无需切到 `instance-skill`。

**所属模块**：`instance-skill`（本 skill 直调跨模块依赖）

**入参**：

```json
{"task_id": "basic_train_xxx", "keyword": "error", "page_size": 50}
```

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `task_id` | string | ✅ | 无 | 任务 ID |
| `instance_id` | string | ❌ | 无 | 实例 ID（不传则查最新实例） |
| `pod_name` | string | ❌ | 无 | Pod 名称 |
| `keyword` | string | ❌ | 无 | 日志搜索关键词 |
| `page_size` | integer | ❌ | 500 | 每页数量 |
| `order` | string | ❌ | `desc` | `asc` / `desc` |

**关键出参**：返回日志内容 + `total`（匹配总数）+ `has_more`（是否有更多）。

**本 skill 中的典型用途**：
1. 任务失败诊断流程：`task_detail` → `failure_event_list` → 若需进一步排查报错，直调 `query_hunyuan_train_instance_logs(task_id, keyword="error")`
2. 用户问"为什么被终止/失败"时，查完 failure_events 后可用 `instance_logs` 补充日志上下文

**注意事项**：
- 只传 `task_id`、`instance_id`、`pod_name`、`keyword`、`page`、`page_size`、`order`、`container`
- "最新 N 条日志"：`{task_id, page_size: N, order: "desc"}`
- 同一日志问题只调一次，不要翻页穷举
- 用户贴出太极 URL 时优先抽取 URL 参数中的 `taskID`→`task_id`、`instId`→`instance_id`

**调用示例**：
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_instance_logs '{"task_id": "basic_train_xxx", "keyword": "error"}'
```
