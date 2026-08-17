## query_hunyuan_data_posttrain_shuttles

**用途**：分页查询后训练数据班车列表。

**必填**：`wsid`  
**常用可选**：`stages`（如 `["SFT"]`）、`thinking_types`（如 `["SLOW_THINKING"]`）、`modality`（`TEXT` / `MULTIMODAL`）、`name`、`page`、`page_size`

对齐前端：上班车弹窗用 TopicData 的 `stage` + `thinking_type` + **`modality`** 过滤班车。Agent 应先 `get_hunyuan_data_topic_data` 拿到这些字段再 query。  
10103/12290 打通空间**强烈建议**显式传 `modality`，否则可能混出跨模态班车，后续 create 会被模态一致性校验拒绝。

```bash
python3 scripts/connect_mcp.py call query_hunyuan_data_posttrain_shuttles '{
  "wsid": 12290,
  "stages": ["GRPO"],
  "thinking_types": ["FAST_THINKING"],
  "modality": "MULTIMODAL",
  "page": 1,
  "page_size": 50
}'
```

返回 `items[].id` 即 `shuttle_id`。

---

## query_hunyuan_data_topic_data_tasks

**用途**：查询某个班车上已上车的 Topic 数据明细（即一个 `TopicDataTask`）。对齐太极前端「后训练数据 → 数据班车详情页」的 Topic 明细列表。**只读**，不会改动任何数据。

一行 = **一条已上车的 TopicData**。

**必填**：`wsid`、`shuttle_id`  
**可选**：`topic_id`（只看某个 Topic 的明细）、`page`（1-based，默认 1）、`page_size`（默认 20，**最大 100**）

默认按 `id` 倒序 —— **最新上车的在前**（与前端一致）。列表按接口返回原始顺序展示，不得重排 / 截断。

```bash
python3 scripts/connect_mcp.py call query_hunyuan_data_topic_data_tasks '{
  "wsid": 12290,
  "shuttle_id": 211,
  "page": 1,
  "page_size": 50
}'
```

### 返回里常用的字段

| 字段 | 含义 |
|------|------|
| `items[].id` | **`topic_data_task_id`**（上车记录 ID，即 create 当时返回的 `id`） |
| `items[].data_id` | **`topic_data_id`**（数据版本 ID，可拿去 `get_hunyuan_data_topic_data` 查详情） |
| `items[].topic_id` / `topic_name` | 所属 Topic |
| `items[].data_name` / `data_key` / `data_version` / `data_desc` | 数据版本标识与描述 |
| `items[].data_source_path` | **该数据版本注册时填写的原始文件路径**（TopicData 的 `source_path`）。用户问「这条数据是从哪个文件来的 / 原始路径是什么」时用它 |
| `items[].storage_path` / `storage_type` | 上车任务侧的落地路径与存储类型（⚠️ **不是**注册时的原始路径，别与 `data_source_path` 混用） |
| `items[].data_check_status` / `data_check_message` | 数据校验状态（`PENDING` / `FAILED` / `SUCCEEDED`）与失败原因 |
| `items[].epochs` / `priority` / `size` / `rows` | 训练轮次、优先级、数据大小与行数 |
| `items[].inspection_result` / `inspection_progress` / `inspection_version` | 质检结果、V2 进度、质检版本 |
| `items[].reason` / `config` | 当时的上车原因与加入方式（`config.type`） |
| `items[].creator` / `create_time` | 谁在什么时候上的车 |
| `total` | 该班车（或该 Topic）下的明细总条数 |

> 时间字段名是 `create_time` / `update_time`（不是 `created_at`）—— 本资源域现网即如此输出。

### 典型用途

1. **回答「这个班车上都有哪些 topic 数据」** → 直接查，逐条展示上表字段并给出 `total`。
2. **找回丢失的 `topic_data_task_id`** → 按 `shuttle_id` 查（知道 Topic 时再带 `topic_id` 收窄），在 `items[]` 里按 `data_id` 认出目标那条，它的 `id` 就是 task id。
3. **上车前 / 后自查** → 上车前确认目标数据是否已在该班车（避免 `50001` 重复上车）；上车后复核结果，不必反复轮询。
4. **查数据来源文件** → 用户问「班车上这些数据分别来自哪个文件 / 原始路径是什么 / 数据是从哪注册进来的」时，展示 `items[].data_source_path`（注册时的原始路径）。**不要**拿 `storage_path` 顶替 —— 那是上车任务侧的落地路径，不是用户当初填的源文件。

> 💡 展示明细时，若用户没特别要求精简，建议把 `data_source_path` 一并列出（它是定位数据来源最直接的信息）。

### 错误分支

| 情况 | 返回 |
|------|------|
| 班车不存在 | `40401`「后训练班车不存在: shuttle_id=…」→ 如实告知 id 有误，可用 `query_hunyuan_data_posttrain_shuttles` 帮用户找正确班车 |
| 班车不属于该 `wsid` | `40301`「当前工作空间无法访问该班车」→ 提示换 `wsid` 或确认班车归属，**不要**换个 wsid 硬试 |
| 缺 `wsid` / `shuttle_id` | `40001` |

> `page_size` 传超过 100 会被后端收敛为 100（明细含质检进度，避免上游扇出过大）。
> 条数多时**照实翻页**，不要把「一页 100 条」当成「总共 100 条」—— 以 `total` / `has_more` 为准。

---

## create_hunyuan_data_topic_data_task

**用途**：**上班车** —— 把已有 TopicData 加入已有后训练班车。

### 必填

| 字段 | 说明 |
|------|------|
| `wsid` | 工作空间 |
| `shuttle_id` | 目标班车 |
| `topic_data_id` | 已就绪的数据版本 |

### 强烈建议 / Agent 默认（对齐前端 Topic 详情「上班车」）

| 字段 | Agent 行为 |
|------|------------|
| `reason` | 用户未给则追问；前端为必填文案 |
| `config.type` | **必须显式传 `APPEND`**（前端默认）；不要省略（省略走后端 legacy，易因同 Topic 已有数据被拒） |
| `config.type=REPLACE` | 仅用户明确要「替换」时；且必须带非空 `replace_ids` |

可选 GRPO/DPO 登记字段见 MCP YAML `config`（训练配置路径、镜像、swanlab 等）；用户未提则不传。

### 前置条件（调用前 Agent 应确认）

1. `get_hunyuan_data_topic_data` → `status=SUCCEEDED`（上传/统计完成）
2. V2（`is_new`/inspection_version=v2）还须 md5 成功（否则后端拒绝）
3. TopicData 的 `stage` / `thinking_type` / **`modality`** 与班车一致（双方非空时后端硬校验跨模态）
4. 同班车同 `topic_data_id` 尚未上过（不可重复）
   → **怎么查**（两条都可用，任选其一即可）：
   - 从数据侧看：`get_hunyuan_data_topic_data` 返回的 **`shuttle_info_list[]`** 就是该 TopicData 已上过的班车（元素形如 `{"id": 157, "name": "..."}`）；
   - 从班车侧看：`query_hunyuan_data_topic_data_tasks(wsid, shuttle_id)` 列出该班车现有明细，看 `items[].data_id` 里有没有目标 `topic_data_id`。
   目标 `shuttle_id` 若已在其中，**不要调用 create**（会返回 `50001 该Topic数据已在当前班车中存在`），直接告知用户「已在该班车，无需重复上车」并列出可选的其他班车。

```bash
python3 scripts/connect_mcp.py call create_hunyuan_data_topic_data_task '{
  "wsid": 12290,
  "shuttle_id": 211,
  "topic_data_id": 2038,
  "reason": "skill_test join shuttle",
  "config": { "type": "APPEND" }
}'
```

成功返回 TopicDataTask 详情，`id` 即 `topic_data_task_id`。向用户回显：已按前端默认 **APPEND** 上班车。

> ⚠️ **`id` 应当场记下并回显给用户**（省一次查询，也让用户立刻拿到凭据）。
> **万一丢了不用慌**：用 `query_hunyuan_data_topic_data_tasks(wsid, shuttle_id)` 按班车反查即可 ——
> 在 `items[]` 里按 `data_id == 本次的 topic_data_id` 认出那条，它的 `id` 就是 `topic_data_task_id`。
> **不要**为了找回 id 去重复调 create（只会得到 `50001 已存在`，且不会返回 id）。

### 选班车规则（多条匹配时）

- 用户已**预授权直接执行**（如「不用确认，直接执行」）→ 可自行选定，优先级：
  1. 排除用户明确点名要避开的班车
  2. 排除 `shuttle_info_list` 里已上过的
  3. 优先 `status=PUBLISH`（正常可用）而非 `QUALITY_INSPECTION` 等中间态
  4. 仍多条 → 取 `id` 最大（最新），并**在回复里说明「我按 X 规则选了班车 N」**
- 用户**未**预授权 → 列出候选（含 `id / name / status`）让用户选，不要擅自决定。

`reason` 是前端必填：用户没给就必须追问，**不要用空值或自己编造理由硬调**。

### 缺 `topic_data_id` 时（必须帮用户查，不要一句「请提供 ID」了事）

用户常说「我那个后训练数据」「我昨天传的数据」而不记得 ID。此时：

**允许并推荐**先调只读的 `query_hunyuan_data_topic_datas`（按 `wsid` + 用户给的线索：`name` 模糊 / `creator` / 创建时间范围）列出候选，
把 `id / name / version / stage / thinking_type / status` 摆给用户确认，再继续创建。

> 与「模棱两可时严禁直接调用任何工具」的关系：那条约束的是**写操作**和**猜 ID 硬调**。
> **只读查询用于消歧是被鼓励的** —— 目的就是让用户不必自己去翻 ID。
> 唯一红线：**绝不能自己从候选里挑一个就去 create**；必须用户确认，或候选只有一条且已明确回显。

线索都没有（用户只说「我的数据」）→ 就按 `wsid` 列最近的若干条给用户挑，仍然**先查再问**，而不是空手追问。

---

### 推荐链路（自然语言「上班车」）

```
① 取 wsid（未知则 references/helper_api.md 的 list_user_workspaces）
② 有 topic_data_id → get_hunyuan_data_topic_data 确认 SUCCEEDED（及 md5）
   无 topic_data_id → 先帮用户查出候选（query_hunyuan_data_topic_datas），不要直接放弃
③ 无 shuttle_id → query_hunyuan_data_posttrain_shuttles(stages=[stage], thinking_types=[thinking_type])
   多条时按「选班车规则」处理
④ create_hunyuan_data_topic_data_task(..., config.type=APPEND, reason=...)
⑤ 汇报 topic_data_task_id（丢了可用 query_hunyuan_data_topic_data_tasks 按 shuttle_id 反查）；
   上车是同步生效的，不要为了「等状态」反复轮询
```

### 模糊追问

> 您是想：  
> 1. **后训练上班车**（把 Topic 数据版本加入数据班车）→ 需要 `topic_data_id` 与目标班车  
> 2. **看某个班车上有哪些 Topic 数据** → 请提供 `shuttle_id`（走 `query_hunyuan_data_topic_data_tasks`）  
> 3. **查预训练融合任务**进度/产物 → 请提供 `shuttle_task_id`（走 data-processing-skill）  
> 4. **新建一个班车** → 当前 Skill 暂不支持，请在太极「后训练数据 → 数据班车」页面创建
