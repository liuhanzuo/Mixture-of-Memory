# 后训练数据 Skill 外部依赖工具速查

> 本文件记录本 skill 直调的跨模块依赖工具。外部跳转（非直调）的工具仅保留跳转说明。

## list_user_workspaces

**用途**：查询当前用户有权限的工作空间列表。创建 Topic / Dataset / TopicData、质检链路补全前置 TopicData 时缺少 `wsid` 使用。

**入参**：无必填入参。可选传 `platform`（`hunyuan` / `taiji` / `sft`）按平台过滤，一般传 `{}` 即可。

```json
{}
```

**关键出参**：

| 字段 | 类型 | 含义 |
|---|---|---|
| `username` | string | 当前用户名 |
| `is_admin` | boolean | 是否平台级管理员 |
| `workspaces` | array | 工作空间列表 |
| `hy_basic_wsid_map` | object | 混元基础空间 ID 映射（一般忽略） |

`workspaces` 中重点字段：`wsid`、`name`、`desc`、`workspace_type`、`is_admin`、`managers`。

**本 skill 中的典型用途**：
- `create_hunyuan_data_topic` / `create_hunyuan_data_topic_dataset` / `create_hunyuan_data_topic_data` 的 `wsid` 都不能为 0；未知时先用本工具列候选空间

**注意事项**：
- 用户已明确给出 `wsid` 时可直接使用
- 这里查到的是空间，不是 Topic / Dataset 本身的对象 ID

**调用示例**：
```bash
python3 scripts/connect_mcp.py call list_user_workspaces '{}'
```
