# 模型压缩 Skill 外部依赖工具速查

> 本文件记录本 skill 直调的跨模块依赖工具。**仅限以下声明的工具**可直调；未声明的跨模块工具严禁使用。
>
> ⛔ **严禁探索**：不要用 `tool_manual.py` 扫描 storage-mgmt-skill / resource-mgmt-skill / workspace-skill 的工具手册。不要调用 `query_shared_resources_app_group_list`、`query_storage_clusters`、`query_app_group_ceph_locations`、`query_storage_dir_permission` 等资源/存储类工具。

## list_user_workspaces

直调工作空间查询工具（helper）。**仅在缺少 `wsid` 且用户明确要创建/查询压缩任务时使用**；用户已提供 wsid 或其他上下文足够时不要调。

**入参**：无入参。

```json
{}
```

**关键出参**：
- `username`
- `is_admin`
- `wsList`

`wsList` 每项重点读取：`app_id`、`name`、`desc`、`workspaceType`、`is_admin`、`managers`。

**本 skill 中的典型用途**：
- 创建/批量查询/查询详情压缩任务时缺少 `wsid`，先列出工作空间让用户选定，再用选定的 `wsid` 调用 `create_compress_task`/`list_compress_tasks`/`get_compress_task_detail`。

**注意事项**：
- `wsid` 是 `create_compress_task`/`list_compress_tasks`/`get_compress_task_detail` 的强必填，不能用 `0` 或猜测值兜底。
- `get_compress_strategy` 不需要 `wsid`（按模型匹配），无需走此 helper。
- **仅这一个 helper 工具**，不要通过 tool_manual.py 探索其他模块的工具。
- 不要调用任何 storage/resource 类工具（query_storage_*、query_shared_resources_*）。

**调用示例**：
```bash
python3 scripts/connect_mcp.py call list_user_workspaces '{}'
```
