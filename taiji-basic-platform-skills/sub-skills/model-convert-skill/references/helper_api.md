# 模型转换 Skill 外部依赖工具速查

> 本文件记录本 skill 直调的跨模块依赖工具。外部跳转（非直调）的工具仅保留跳转说明。

## list_user_workspaces

**用途**：查询当前用户有权限的工作空间列表。创建、查询、克隆模型转换任务 / HF 模板管理缺少 `wsid` 时使用。

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

`workspaces` 子项重点字段：`wsid`、`name`、`desc`、`workspace_type`、`is_admin`、`managers`。

**本 skill 中的典型用途**：
- 创建、查询、克隆模型转换任务时缺少 `wsid`，先列出候选空间再让用户选定后继续

**注意事项**：
- `wsid` 是模型转换的强必填，不能用 `0` 或猜测值兜底
- 用户已经提供明确 `wsid` 时，不要重复调用

**调用示例**：
```bash
python3 scripts/connect_mcp.py call list_user_workspaces '{}'
```
