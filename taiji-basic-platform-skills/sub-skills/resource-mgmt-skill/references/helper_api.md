# 资源管理 Skill 外部依赖工具速查

> 本文件记录本 skill 直调的跨模块依赖工具。外部跳转（非直调）的工具仅保留跳转说明。

## list_user_workspaces

**用途**：查询当前用户有权限的工作空间列表。用户问资源、配额、应用组但没有明确工作空间上下文时使用。

**入参**：无必填入参。可选传 `platform`（`hunyuan` / `taiji` / `sft`）按平台过滤，一般传 `{}` 即可。

```json
{}
```

**关键出参**：

| 字段 | 类型 | 含义 |
|---|---|---|
| `username` | string | 当前用户名 |
| `is_admin` | boolean | 平台级管理员标识 |
| `workspaces` | array | 工作空间列表 |
| `hy_basic_wsid_map` | object | 混元基础空间 ID 映射（一般忽略） |

`workspaces` 子项重点字段：`wsid`、`name`、`desc`、`workspace_type`、`is_admin`、`managers`。

**本 skill 中的典型用途**：
- 先列出用户可访问的空间，再继续确认资源查询上下文

**注意事项**：
- 本工具不直接给出应用组，需要和本 skill 自己的 `query_shared_resources_app_group_list` 配合使用
- 用户已经提供明确 `wsid` 时，不要重复调用

**调用示例**：
```bash
python3 scripts/connect_mcp.py call list_user_workspaces '{}'
```
