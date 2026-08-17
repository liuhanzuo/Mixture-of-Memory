# 数据处理 Skill 外部依赖工具速查

> 本文件记录本 skill 直调的跨模块依赖工具。外部跳转（非直调）的工具仅保留跳转说明。

## list_user_workspaces

**用途**：查询当前用户有权限的工作空间列表。用户做数据处理、预训练转 bin、跨地域 SFT 转 bin，但没给 `wsid` 时使用。

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
- Pipeline 需要 `wsid` 且用户未提供时，先查出候选空间再让用户选择。

**注意事项**：
- 用户已经给了明确 `wsid` 时，直接使用该值。
- 只在确实需要 `wsid` 的流程里调用，不要把它当成默认预处理步骤。

**调用示例**：
```bash
python3 scripts/connect_mcp.py call list_user_workspaces '{}'
```

---

### 跨模块跳转说明（非直调）

以下能力不属于本 skill 直调范围，命中时请切换对应子 skill，不要在本 skill 内拼装：
- **后训练数据质检 / 后训练 Topic 数据链路管理**（`create_hunyuan_data_topic` / `create_hunyuan_data_topic_dataset` 等**无 `pretrain` 字样**的工具）→ `posttrain-data-skill`。
