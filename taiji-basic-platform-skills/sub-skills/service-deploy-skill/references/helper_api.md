# 服务部署 Skill 外部依赖工具速查

> 本文件记录本 skill 直调的跨模块依赖工具。外部跳转（非直调）的工具仅保留跳转说明。

## list_user_workspaces

**用途**：查询当前用户有权限的工作空间列表。用户要查模型服务或服务组但没有提供 `wsid` 时使用。

**入参**：无必填入参。可选传 `platform`（`hunyuan` / `taiji` / `sft`）按平台过滤，一般传 `{}` 即可。

```json
{}
```

**关键出参**：

顶层：

- `username`
- `is_admin`
- `workspaces`（数组，用户有权限的空间列表）
- `hy_basic_wsid_map`（一般忽略）

`workspaces` 每一项至少关注：`wsid`、`name`、`desc`、`workspace_type`、`is_admin`、`managers`。

**本 skill 中的典型用途**：
- 缺空间时先列候选空间，再继续查询 `list_deploy_inferences` 或 `list_deploy_services`

**注意事项**：
- 用户已明确提供 `wsid` 时直接复用，不要重复查询

**调用示例**：
```bash
python3 scripts/connect_mcp.py call list_user_workspaces '{}'
```
