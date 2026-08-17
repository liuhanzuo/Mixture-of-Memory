# 模型管理 Skill 外部依赖工具速查

> 本文件记录本 skill 直调的跨模块依赖工具（工作空间、应用组、训练产出列表）。外部跳转（非直调）的工具仅保留跳转说明。

## list_user_workspaces

**用途**：查询当前用户有权限的工作空间列表。用户没给 `wsid` 时要搜模型、查官方可训练模型、发布模型、改地域时使用；对 `search_hunyuan_official_models_by_train` 做 `wsid` 权限校验时也使用。

**入参**：无必填入参，通常固定传 `{}`。可选参数 `platform`（`hunyuan` / `taiji` / `sft`）用于按平台过滤，本 skill 场景下建议不传，返回全部有权限空间。

```json
{}
```

或者只看混元平台：

```json
{ "platform": "hunyuan" }
```

**关键出参**：

顶层重点字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `username` | string | 当前用户名 |
| `is_admin` | boolean | 是否平台级管理员 |
| `workspaces` | array | 用户可访问的工作空间列表 |
| `hy_basic_wsid_map` | object | 混元基础空间 ID 映射（一般忽略） |

`workspaces` 每一项重点字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `wsid` | string | 工作空间 ID，即后续模型工具的 `wsid` |
| `name` | string | 空间名称 |
| `desc` | string | 空间描述 |
| `workspace_type` | string | 空间类型（`hunyuan` / `sft` / `taiji` / `general`） |
| `is_admin` | boolean | 是否该空间管理员 |
| `managers` | string | 管理员列表（分号分隔） |

**本 skill 中的典型用途**：
- `search_hunyuan_official_models_by_train` 前先查用户有权限的空间
- 用户未给 `wsid` 时，列出候选空间供选择
- 用户已给 `wsid` 时，用 `workspaces` 校验该值是否在授权范围内

**注意事项**：
- 这个工具入参只有可选的 `platform`，不接收 `wsid`
- `search_hunyuan_official_models_by_train` 是例外场景：即使用户给了 `wsid`，也要先用本工具校验
- 其他普通模型查询场景，如果用户已明确给出 `wsid`，可直接复用用户值，不要额外多查

**调用示例**：
```bash
python3 scripts/connect_mcp.py call list_user_workspaces '{}'
```

---

## query_user_app_groups

**用途**：查询当前用户有权限的应用组列表。新增模型地域时用户没给 `queue_name`，或需要先找应用组在哪些 ceph 地域有挂载路径时使用。

**入参**：无强制业务入参，调用时可直接传 `{}`。

```json
{}
```

**关键出参**：返回当前用户有权限的应用组列表。不同调用链可能包装格式不同，但以下字段是本 skill 需要重点读取的：

| 字段 | 类型 | 含义 |
|---|---|---|
| `queue_name` / `business_flag` / `id` | string | 应用组标识，后续常作为 `queue_name` 或 `app_group_id` 使用 |
| `business_readable_name` | string | 应用组可读名称 |
| `wsids` | string | 逗号分隔的空间归属，可能混有 `general` / `hunyuan` |

**典型用途**：让用户从自己有权限的应用组里挑一个作为 `queue_name`。

**调用示例**：
```bash
python3 scripts/connect_mcp.py call query_user_app_groups '{}'
```

---

## query_app_group_ceph_locations

**用途**：查询应用组的 ceph 地域挂载信息。用户选定应用组后，找目标地域对应的 `containerPath` 时使用。

**入参**：

```json
{
  "queue_name": "TaiJi_HYAide_800H20"
}
```

| 参数 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `queue_name` | string | 是 | 应用组名称；通常可直接复用 `query_user_app_groups` 返回的 `queue_name` |

**关键出参**：返回 Markdown 风格的地域与 ceph 信息，重点关注：

| 字段/信息 | 含义 |
|---|---|
| `location` | 地域英文缩写，如 `sh`、`gz` |
| `ch_location` | 地域中文名 |
| `containerPath` | 该地域下可用的 ceph 容器挂载路径 |
| 配额/已用容量信息 | 用于判断该应用组在该地域是否具备可用存储 |

**典型用途**：
- 从返回里找目标地域对应的 `containerPath`
- 将 `containerPath` 的前两级目录作为 `update_hunyuan_models_card_location.path_prefix`

**注意事项**：
- `query_user_app_groups` 用于"先选应用组"，`query_app_group_ceph_locations` 用于"在选定应用组后找地域路径"，两者不要混用
- `query_app_group_ceph_locations` 的入参是 `queue_name`，不是 `wsid`
- `path_prefix` 只取前两级目录，例如 `/apdcephfs_sh3/share_123456`

**调用示例**：
```bash
python3 scripts/connect_mcp.py call query_app_group_ceph_locations '{"queue_name":"TaiJi_HYAide_800H20"}'
```

---

## query_hunyuan_train_checkpoint_list

**用途**：查询训练任务产出列表。用户要把训练任务产出发布成模型，但还没给 `checkpoint`、`instance_id`、`path`，需要从产出列表中挑选具体 checkpoint，或核对某个 checkpoint 是否已发布时使用。

**入参**：

```json
{
  "task_id": "basic_train_xxx"
}
```

| 参数 | 类型 | 必填 | 含义 |
|---|---|---|---|
| `task_id` | string | 是 | 训练任务 ID，支持 `basic_train_*` / `finetuning_*` |

**关键出参**：不同后端版本可能返回 `items` / `checkpoints` / `ckptList`，但本 skill 只需读取以下语义字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `instance_id` / `instanceId` | string | 发布接口的 `instance_id` |
| `name` / `checkpoint` | string | checkpoint 名称，对应发布接口的 `checkpoint` |
| `path` / `ckpt_path` / `model_path` / `hf_model_path` | string | checkpoint 或模型路径，对应发布接口的 `path` |
| `released` | boolean/string | 是否已发布；不要自行猜测 |
| `model_id` | integer | 已发布时关联的模型卡片 ID |
| `create_time` | string | 创建时间 |

**本 skill 中的典型用途**：
1. 先调 `query_hunyuan_train_checkpoint_list` 获取训练任务产出列表。
2. 若用户已指定 checkpoint（如 `checkpoint-1`），直接在列表中匹配该项；若未指定，再让用户选择。
3. 将返回中的 `instance_id`、checkpoint 名称、路径分别映射到 `release_hunyuan_training_checkpoint_as_model` 的 `instance_id`、`checkpoint`、`path`。
4. 若用户明确说"不需要确认/直接执行"，视为已确认写操作，可直接发布；发布失败时透传错误，不要改走搜索或克隆模型。
5. `release_hunyuan_training_checkpoint_as_model` 是异步任务，仅返回 `task_record_id` 和初始状态（`PENDING` / `PUBLISHING`）。若用户随后追问"发布到哪一步了 / 发布成功了吗"，直接用该 `task_record_id`（或已知的 `model_id` / `model_name`）调用 `get_hunyuan_training_model_release_status`，不要走搜索或详情工具探路。该工具**只**查"模型发布"这一次异步任务的状态；若用户问的是独立的"模型格式转换任务"进度（HF ↔ Mcore 等），**若环境中存在 `model-convert-skill`，则优先使用它**（提供更详细的转换任务信息，如进度、日志、产物路径）。

**注意事项**：
- 优先使用 `query_hunyuan_train_checkpoint_list`，不要再调用旧别名 `task_ckpt_list`，避免工具链冗余和命名不一致。
- 产出已发布与否看 `released` 字段；即使已发布，若用户要求验证发布流程，可调用发布工具并透传"已发布/无权限"等后端错误。
- `finetuning_*` 通常只需产出列表 + 发布工具；`basic_train_*` / 自定义训练若缺少模型基础参数，再补 `list_hunyuan_models_platform_enums` 查询枚举值。

**调用示例**：
```bash
python3 scripts/connect_mcp.py call query_hunyuan_train_checkpoint_list '{"task_id":"basic_train_xxx"}'
```
