## list_user_workspaces

查询当前用户有权限访问的所有工作空间列表，返回空间 ID、名称、类型、管理员等信息，以及用户是否为平台级管理员。可选按平台过滤。

> **本工具可独立使用**：仅查工作空间列表/wsid 时，只调本工具即可，不要链式调用 `list_user_groups`。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `platform` | string | ❌ | 无（返回所有平台） | 平台标识，可选值：`hunyuan` / `taiji` / `sft`。不传则返回用户在所有平台下有权限的空间 |

### 调用示例

```json
{}
```
不传参返回所有平台空间；`{"platform": "hunyuan"}` 只返回混元平台空间。

### 返回字段

**顶层**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `username` | string | 当前用户名（RTX） |
| `is_admin` | boolean | 是否为平台级管理员（超管或只读管理员） |
| `workspaces` | array | 用户有权限的空间列表 |
| `hy_basic_wsid_map` | object | 混元基础空间 ID 映射（`{平台标识: [wsid, ...] }`），供跨平台联动场景参考 |

**`workspaces[*]`**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `wsid` | string | 空间 ID（后续工具的 `wsid` 参数取值） |
| `name` | string | 空间名称 |
| `desc` | string | 空间描述 |
| `is_admin` | boolean | 当前用户是否为该空间的管理员 |
| `managers` | string | 空间管理员列表（最多 10 人，分号分隔） |
| `workspace_type` | string | 空间类型：`hunyuan` / `sft` / `taiji` / `general` |

---

## list_user_groups

查询指定工作空间下的所有用户组列表，返回用户组 ID、名称、描述、类型、创建者、是否禁用、成员信息、创建/更新时间。支持按名称模糊搜索。

### 参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `wsid` | integer | ✅ | — | 工作空间 ID，**不能为 0 或空值**；调用者需为该空间成员或平台管理员 |
| `keyword` | string | ❌ | 无（返回全部） | 搜索关键词，按用户组名称模糊匹配（后端`LIKE %keyword%`） |

### 调用示例

```json
{"wsid": 10314}
```
按名称模糊搜索：`{"wsid": 10103, "keyword": "管理员"}`

### 返回字段

**顶层**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `wsid` | integer | 回显的工作空间 ID |
| `user_groups` | array | 该空间下的用户组列表 |

**`user_groups[*]`**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `group_id` | integer | 用户组 ID（可直接用于 `update_hunyuan_models_card_permission` 的 `admin_user_groups` / `common_user_groups` 参数） |
| `name` | string | 用户组名称 |
| `description` | string | 用户组描述 |
| `type` | integer | 用户组类型：`1`=空间成员组、`2`=空间管理员组、`3`=自定义组 |
| `creator` | string | 创建人 RTX |
| `is_disabled` | boolean | 是否已禁用 |
| `members.users` | array\<string\> | 成员用户列表（RTX） |
| `members.user_groups` | array\<integer\> | 嵌套的子用户组 ID 列表 |
| `created_at` | string | 创建时间（`yyyy-MM-dd HH:mm:ss`） |
| `updated_at` | string | 更新时间（`yyyy-MM-dd HH:mm:ss`） |

### 返回示例

```json
{
  "wsid": 10103,
  "user_groups": [
    {
      "group_id": 1,
      "name": "空间成员组",
      "description": "空间 10103 的默认成员组",
      "type": 1,
      "creator": "system",
      "is_disabled": false,
      "members": {
        "users": ["shushuyang", "alice", "bob"],
        "user_groups": []
      },
      "created_at": "2026-01-15 10:00:00",
      "updated_at": "2026-07-01 14:30:00"
    },
    {
      "group_id": 15,
      "name": "SFT训练组",
      "description": "负责 SFT 微调训练的团队",
      "type": 3,
      "creator": "shushuyang",
      "is_disabled": false,
      "members": {
        "users": ["alice", "bob"],
        "user_groups": [1]
      },
      "created_at": "2026-03-10 16:00:00",
      "updated_at": "2026-06-28 11:00:00"
    }
  ]
}
```

### 交互规则

1. **仅在以下情况调用本工具**：用户明确要查用户组/成员/权限组/有哪些组。
2. 用户未提供 `wsid` 且当前无 wsid 上下文 → 追问 wsid（不要自动链式调用 `list_user_workspaces` 来获取 wsid 再调本工具）。
3. 用户提到特定用户组名称 → 优先传 `keyword` 做模糊匹配，而不是先拉全量再内存过滤。
4. `group_id` 可直接用于 `update_hunyuan_models_card_permission` 的 `admin_user_groups` / `common_user_groups` 数组入参。
5. 后端不支持按 `type`/`creator` 直接过滤；需要时在首次返回结果中内存过滤。
6. **本工具不是 `list_user_workspaces` 的后续步骤**：用户只问"我的空间/wsid"时不要调本工具。
