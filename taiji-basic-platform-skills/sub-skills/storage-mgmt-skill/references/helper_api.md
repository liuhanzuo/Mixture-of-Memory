## list_user_workspaces

本 helper 供 `storage-mgmt-skill` 跨模块直调，获取用户有权限的工作空间列表（wsid）。

### 什么时候用

- 用户要查存储治理、冷热分析、扩容申请，但没有给 `wsid`

### 入参

无必填入参，调用时传 `{}` 即可。可选传 `platform`（`hunyuan` / `taiji` / `sft`）按平台过滤。

```json
{}
```

### 关键出参

- `username`: 当前用户名
- `is_admin`: 是否平台级管理员
- `workspaces`: 工作空间列表（数组）

`workspaces` 项常用字段：`wsid`、`name`、`desc`、`workspace_type`、`is_admin`、`managers`。

### 使用边界

- 缺空间上下文时先列可访问工作空间，再让用户确认属于哪个空间
- 这个工具不能直接给出 `app_group_id`，应用组仍需走 `query_shared_resources_app_group_list`
- 用户已给明确 `wsid` 时直接使用该值，不调本工具

### 调用示例

```bash
python3 scripts/connect_mcp.py call list_user_workspaces '{}'
```

---

## query_shared_resources_app_group_list

本 helper 供 `storage-mgmt-skill` 跨模块直调，获取用户可用的应用组列表（app_group_id）。

### 什么时候用

- 用户要查存储集群、目录权限、存储扩容，但不知道 `app_group_id`
- 用户问"我的/我有权限的存储集群、Ceph 磁盘有哪些"，需要先看当前用户可用应用组
- 用户已给出明确 `app_group_id`（如 `TaiJi_...`）时**不要使用本 helper**，直接传给存储治理工具

### 入参

```json
{"is_usable": true, "page": 1, "page_size": 20}
```

### 关键出参

| 字段 | 类型 | 含义 |
|---|---|---|
| `id` | string | 应用组标识，后续就是 `app_group_id` |
| `name` / `note` | string | 应用组可读名称或备注 |
| `tag` | string | 常包含 wsid / 业务标签，可用于按用户给定 wsid 过滤 |
| `ceph_cluster_name_list` | string/list | 该应用组已有 Ceph 文件存储集群，可用于回答"有哪些可用 Ceph 磁盘"或选择一个代表应用组 |

### 使用边界

1. 默认只查第一页并说明"如需更多可继续翻页"；不要主动改大 `page_size`、翻页或重试
2. 若返回中已有 `ceph_cluster_name_list`，可先基于该字段给出候选应用组/集群
3. 不要使用旧工具名 `query_user_app_groups`
4. 不要为了回答"我的存储集群有哪些"自动遍历全部应用组逐个调用 `query_storage_clusters`
5. 拿到的是应用组标识，不是工作空间 ID

### 调用示例

```bash
python3 scripts/connect_mcp.py call query_shared_resources_app_group_list '{"is_usable": true, "page": 1, "page_size": 20}'
```
