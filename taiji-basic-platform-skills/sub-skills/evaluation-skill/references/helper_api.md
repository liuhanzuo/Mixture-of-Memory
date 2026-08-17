# helper_workspace_query

本 helper 供 `evaluation-skill` 使用，说明 `list_user_workspaces` 的入参与关键出参。

### 什么时候用

- 评测任务列表、Insight、评测集合等场景需要 `wsid`，但用户没有提供

### 工具名

`list_user_workspaces`

### 入参

无必填入参。可选传 `platform`（`hunyuan` / `taiji` / `sft`）按平台过滤，一般传 `{}` 即可。

```json
{}
```

### 关键出参

- `username`
- `is_admin`
- `workspaces`（数组）
- `hy_basic_wsid_map`（一般忽略）

`workspaces` 常用字段：`wsid`、`name`、`desc`、`workspace_type`、`is_admin`、`managers`。

### 本 skill 中的典型用途

- 用户未指定空间时，先列空间候选，再继续查评测数据。

### 注意事项

- 本 skill 某些接口文档中写了默认 `10103`，但实际面向业务使用时，最好仍先确认用户空间，避免查错范围

### 调用示例

```bash
python3 scripts/connect_mcp.py call list_user_workspaces '{}'
```
