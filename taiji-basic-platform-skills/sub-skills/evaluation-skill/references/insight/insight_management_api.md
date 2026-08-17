## create_taiji_eval_insight

创建一个新的 Insight。创建后默认不包含评测任务，需调用 `add_tasks_to_taiji_eval_insight` 或 `update_taiji_eval_insight` 添加。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call create_taiji_eval_insight '{"name": "GPT-4o 评测对比", "desc": "对比分析", "admin": "zhangsan,lisi"}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `name` | string | ✅ | Insight 名称 |
| `desc` | string | ❌ | 描述 |
| `admin` | string | ❌ | 管理员列表，逗号分隔 |
| `file_type` | string | ❌ | 文件类型 |
| `parent_id` | number | ❌ | 父级 Insight ID，用于创建子级 |
| `visibility` | string | ❌ | 可见范围：`CURRENT_WORKSPACE`（默认）/ `ALL_PLATFORM` / `SPECIFIED_WORKSPACES` |
| `visible_ws_ids` | string | ❌ | 可见空间 ID 列表（`visibility=SPECIFIED_WORKSPACES` 时必填） |

---

## update_taiji_eval_insight

修改 Insight 基本信息（名称、描述、管理员、可见范围、配置等）。⚠️ **仅负责人可操作**。
> ⚠️ **任务管理不在本接口**：增删关联评测任务请用 `add_tasks_to_taiji_eval_insight` / `remove_tasks_from_taiji_eval_insight`；基线任务、别名映射等归入 `conf`（JSON 字符串）字段，不再是独立入参。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call update_taiji_eval_insight '{"insight_id": 1599, "name": "新版对比", "desc": "更新描述"}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `insight_id` | number | ✅ | Insight ID |
| `name` | string | ❌ | 新名称 |
| `desc` | string | ❌ | 新描述 |
| `admin` | string | ❌ | 新管理员列表 |
| `parent_id` | number | ❌ | 父级 ID |
| `visibility` | string | ❌ | 可见范围 |
| `visible_ws_ids` | string | ❌ | 可见空间 ID 列表 |
| `conf` | object | ❌ | Insight 配置对象（含基线任务、别名映射、taskIds 等） |

---

## delete_taiji_eval_insight

删除 Insight。🔴 **不可逆操作**，仅负责人可执行。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call delete_taiji_eval_insight '{"insight_id": 1599}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `insight_id` | number | ✅ | Insight ID |

---

## add_tasks_to_taiji_eval_insight

增量追加评测任务到 Insight（不覆盖已有任务）。内部先查现有 taskIds 再追加。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call add_tasks_to_taiji_eval_insight '{"insight_id": 1599, "task_ids": "100,200"}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `insight_id` | number | ✅ | Insight ID |
| `task_ids` | string | ✅ | 逗号分隔的任务 ID |

---

## remove_tasks_from_taiji_eval_insight

从 Insight 中增量移除评测任务。⚠️ 如果移除的任务恰好是基线任务（`baseLineTaskId`），基线配置会被自动清空，需提示用户重新设置。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call remove_tasks_from_taiji_eval_insight '{"insight_id": 1599, "task_ids": "100"}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `insight_id` | number | ✅ | Insight ID |
| `task_ids` | string | ✅ | 逗号分隔的任务 ID |

---

### Insight 管理——查询

## list_taiji_eval_insights

查询 Insight 列表，可按关键词搜索、筛选自己创建的 Insight。返回 Insight 的 ID、名称、描述、创建人等信息。用户说"查我的/我创建的"时，传 `is_mine: true`。

获取到 Insight ID 后，可以调用 `submit_taiji_eval_insight_export` 导出 Insight 数据，或调用 `get_taiji_eval_task_detail` 查看关联的评测任务详情。


**MCP 工具调用：**
```bash
# 搜索关键词
python3 scripts/connect_mcp.py call list_taiji_eval_insights '{"keyword": "代码能力"}'

# 只看我创建的 Insight
python3 scripts/connect_mcp.py call list_taiji_eval_insights '{"is_mine": true, "page_size": 20}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| keyword | string | 否 | 搜索关键词，支持按 Insight 名称模糊匹配，也支持按 ID 精确匹配（传入数字时） |
| is_mine | boolean | 否 | `true` = 只返回当前用户创建的 Insight；`false` 或不传 = 返回所有可见 Insight（默认 false） |
| parent_id | int | 否 | 父级 Insight ID，用于查询子级 Insight |
| page_index | int | 否 | 页码（从 1 开始），不传则使用后端默认值（1） |
| page_size | int | 否 | 每页数量，默认 10 |
| order_by | string | 否 | 排序字段，默认按 id 降序 |


**返回（成功）：**
```json
{
  "keyword": "代码能力",
  "is_mine": null,
  "total": 3,
  "insight_count": 3,
  "insights": [
    {
      "id": 29,
      "name": "代码能力评估报告-2026Q1",
      "desc": "第一季度代码能力评估",
      "admin": "zhangsan,lisi",
      "creator": "zhangsan",
      "updater": "zhangsan",
      "createTime": "2026-01-15 10:30:00",
      "updateTime": "2026-01-20 15:45:00",
      "fileType": null,
      "parentId": null
    },
    {
      "id": 35,
      "name": "代码能力对比分析",
      "desc": "多模型代码能力横向对比",
      "admin": "lisi",
      "creator": "lisi",
      "updater": "lisi",
      "createTime": "2026-02-10 09:00:00",
      "updateTime": "2026-02-15 14:30:00",
      "fileType": null,
      "parentId": null
    }
  ]
}
```

**返回字段说明：**
| 字段 | 类型 | 描述 |
|------|------|------|
| keyword | string/null | 查询使用的关键词 |
| is_mine | boolean/null | 查询使用的筛选条件 |
| total | number | 后端返回的总记录数 |
| insight_count | number | 当前返回的 Insight 数量 |
| insights | array | Insight 列表 |
| insights[].id | number | Insight 主键 ID，可用于后续导出操作 |
| insights[].name | string | Insight 名称 |
| insights[].desc | string | Insight 描述 |
| insights[].admin | string | 管理员列表（逗号分隔） |
| insights[].creator | string | 创建人 |
| insights[].updater | string | 最后更新人 |
| insights[].createTime | string | 创建时间，格式 `YYYY-MM-DD HH:mm:ss` |
| insights[].updateTime | string | 更新时间 |
| insights[].fileType | string/null | 文件类型 |
| insights[].parentId | number/null | 父级 Insight ID |

> 💡 **提示**：获取到 Insight ID 后，可以调用 `submit_taiji_eval_insight_export` 导出该 Insight 的数据进行下载分析。

> 💡 **典型使用场景**：用户想下载某个 Insight 的数据，但不知道 Insight ID 时，先通过本工具搜索获取 ID，再调用导出工具。

---

## get_taiji_eval_insight_detail

查询指定 Insight 的**详情**，返回 Insight 的基础信息、关联任务列表，以及最关键的**评测维度权重树（`weightNodes`）**——其中每个节点带有 `weight`（权重）、`score`（各任务在该维度的得分，按 taskId 索引）以及错误率（`errorRate`）等指标。当用户想看「指标权重」「指标分数」「各任务在各评测集上的加权得分」「insight detail」时，应调用此工具。

> 💡 与 `list_taiji_eval_insights` 的区别：`list_taiji_eval_insights` 只返回 Insight 的基础元数据（用于查 ID），本工具返回完整的指标权重树和打分明细。


**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_insight_detail '{"insight_id": 318, "score": "default"}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| insight_id | int | ✅ | Insight ID，例如 `318` |
| score | string | 否 | 得分计算模式，默认 `default`（不求交集，计算所有任务的所有题目）。可选值由后端定义（如 `default` / `intersection` 等），未明确时使用 `default` 即可 |
| type | string | 否 | 详情类型，由后端定义 |
| debug | boolean | 否 | 调试模式，默认 `false` |
| missing_value_mode | string | 否 | 缺失值处理模式 |


**字段裁剪说明：**

为控制返回数据大小，工具层会对后端返回的数据进行裁剪：
- `insight.conf.renderConf` 字段已删除（前端渲染配置，对查询无意义）
- `insight.tasks[]` 中每个 task **只保留**以下关键字段，其余字段（如 `yaml`、`storagePath`、`resultPath`、`parameterConfiguration`、`exerciseAdvancedConfig`、`extraInfo`、`abilityInsights`、`modelName`、`modelSource`、`status`、`servingId`、`servingStatus`、`mockConfig`、`consumerTasks` 等）会被丢弃：
  ```
  id, arenaId, name, desc, collectionVersionIds, creator,
  completedNum, totalNum, createTime, updateTime, deleteTime,
  collectionInfos, arenaName, serviceGroupId, taskType,
  reasoningEffort, errorRate, baseline, resourceId, triggerId
  ```
- `insight.weightNodes[]` 不做裁剪，**完整保留权重树**（这是本工具的核心数据）

> ⚠️ 如果后续需要查看被裁剪掉的 task 字段（例如 `modelName`、`status`、`yaml`、`parameterConfiguration` 等），请使用 `get_taiji_eval_task_detail` 按 `task_id` 单独查询。

**返回字段说明（核心）：**
| 字段 | 类型 | 描述 |
|------|------|------|
| insight_id | number | Insight ID（请求参数回显） |
| score_filter | string | 得分计算模式（如 `default`） |
| insight.id | number | Insight 主键 |
| insight.name / desc / admin / creator / updater | string | 基础元数据 |
| insight.conf.taskIds | number[] | 关联的评测任务 ID 列表 |
| insight.conf.baseLineTaskId | number | 基线任务 ID |
| insight.conf.aliasMap | object | `{taskId: 别名}`，展示时优先使用 |
| insight.conf.headerDisplayMode | string | 表头展示模式（如 `taskName`） |
| insight.mode / modeDescription | string | 计算模式及说明 |
| insight.tasks | array | 任务列表（已裁剪，只保留上述关键字段） |
| insight.tasks[].id / arenaId / name / desc | – | 基本信息 |
| insight.tasks[].collectionVersionIds | string | 关联的评测集版本 ID（逗号分隔） |
| insight.tasks[].completedNum / totalNum | number | 评测进度 |
| insight.tasks[].collectionInfos | array | 关联评测集信息（`collectionId` / `collectionVersionId` / `name` / `description`） |
| insight.tasks[].arenaName / taskType / reasoningEffort | – | 所属版本、任务类型、思考强度 |
| insight.tasks[].errorRate | object/null | 任务级错误率 |
| insight.tasks[].baseline | boolean | 是否为基线任务 |
| insight.weightNodes | array | **评测维度权重树（核心字段）**，按评测集（collectionVersion）分组 |
| insight.weightNodes[].collectionVersionId | number | 评测集版本 ID |
| insight.weightNodes[].collectionInfo | object | 评测集信息（id/版本 ID/名称/描述） |
| insight.weightNodes[].weightNode | object | 该评测集对应的权重节点根节点（递归结构） |
| weightNode.node_name | string | 节点名称（顶层为「综合评估」，叶子为「评测集#评测项/指标名」格式） |
| weightNode.weight | number | **节点权重**，同一层级的权重之和通常为 1 |
| weightNode.exercise_id / exercise_version_id | number/null | 评测练习 ID / 版本 ID（叶子节点有值） |
| weightNode.metric_name | string/null | 指标名称，如 `acc` / `pass@1`（叶子节点有值） |
| weightNode.count | number | 该节点参与计算的样本数 |
| weightNode.score | object | **各任务在该节点的得分**，结构为 `{"taskId": 分数}` |
| weightNode.errorRate | object | 各任务的错误率明细：`{"taskId": {"errorRate": x, "inferErrorRate": x, "judgeErrorRate": x}}` |
| weightNode.significance / confidence / pairCi | any/null | 显著性 / 置信区间 / 成对置信区间（可能为空） |
| weightNode.nodes | array/null | 子节点列表（递归结构），叶子节点为 `null` |
| insight.traceInfo / timings | any/null | 调用链 / 计算耗时（调试用，通常为 null） |
| note | string | 字段裁剪说明 |

> 💡 **如何阅读 weightNodes**：
> 1. `weightNodes` 是一个数组，**按评测集分组**（每个元素代表一个 collectionVersion）。
> 2. 每个分组的 `weightNode` 是一棵树：根节点通常叫「综合评估」，下面挂多个叶子节点（具体的评测集#评测项/指标）。
> 3. 叶子节点的 `score["taskId"]` 即「该任务在该评测项上的得分」；非叶子节点的 score 是其子节点按 `weight` 加权后的聚合得分。
> 4. 任务名/别名可在 `insight.conf.aliasMap` 中按 taskId 查询；展示时建议使用 `aliasMap` 优先，回退到 `tasks[].name`。

> 💡 **典型使用场景**：
> - 用户问「318 这个 insight 各任务的指标分数是多少」→ 调用本工具，从 `weightNodes[].weightNode.nodes[]` 中的叶子节点读取 `score`。
> - 用户问「查一下 insight 318 各维度的权重」→ 调用本工具，从 `weightNodes[].weightNode.nodes[].weight` 读取。
> - 用户问「insight detail 318」→ 直接调用本工具。

---

### Insight Case 明细对比

## list_taiji_eval_insight_cases

多任务 Case 级明细对比。传入多个任务 ID，按题目（question_id）维度对比各任务在同一道题上的得分、模型输出、token 消耗等数据。
典型场景：对比多个模型在同一评测集上的 Case 级表现差异、筛选 Badcase、按维度对比特定类别的题目。

> ⚠️ 本工具返回数据量大（单次 100 条约 2MB），不建议在 Agent 上下文直接展示完整结果。推荐通过下方「[可视化分析（脚本）](#可视化分析脚本)」拉取数据到本地文件，再启动可视化服务查看。用户要查明细对比时，Agent 先判断是否需要跑脚本。MCP 调用仅用于结构验证（`page_size=3` 拉少量样本）。

**MCP 工具调用：**
```bash
# 对比两个任务的 Case 明细
python3 scripts/connect_mcp.py call list_taiji_eval_insight_cases '{"task_ids": [103608, 103609]}'

# 按维度和分数范围筛选
python3 scripts/connect_mcp.py call list_taiji_eval_insight_cases '{"task_ids": [103608, 103609], "dimension_filter": {"task_lv1": ["代码能力"]}, "task_score_filters": [{"task_id": 103608, "score_lte": 0.5}]}'
```

**参数：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_ids | array\<int\> | ✅ | 参与对比的任务 ID 列表，1~50 个 |
| exercise_version_ids | array\<int\> | 否 | 按评测集版本筛选 |
| question_ids | array\<int\> | 否 | 按题目 ID 精确筛选 |
| dimension_filter | object | 否 | 维度过滤，key: task_lv1/task_lv2/language/source/difficulty/tag |
| avg_score_filter | object | 否 | 全局平均分过滤，含 operator(EQ/GT/GTE/LT/LTE) 和 value |
| task_score_filters | array | 否 | 按 task 维度分数范围筛选，每项含 task_id/score_gte/score_lte |
| filter_map | array | 否 | Case 级过滤条件列表 |
| page_index | int | 否 | 页码，默认 1 |
| page_size | int | 否 | 每页数量，默认 10 |
| order_by | string | 否 | 排序字段，默认按 id 降序 |

**返回：**
```json
{
  "data": [
    {
      "question_id": 1001,
      "exercise_name": "autocodebench_v2",
      "exercise_version_id": 379,
      "common": {"input": "...", "messages": [...], "ref_answer": "..."},
      "task_case_details": [
        {
          "task_id": 103608,
          "task_name": "Qwen3-8B-上海",
          "avg_score": 0.85,
          "max_score": 0.92,
          "min_score": 0.78,
          "avg_completion_tokens": 512,
          "avg_prompt_tokens": 1024,
          "score": "{\"pass@1\": 0.85}",
          "payload": { ... }
        }
      ]
    }
  ],
  "task_info": {
    "103608": {"task_name": "Qwen3-8B-上海", "model_name": "Qwen3-8B", "date": "2026-07-09"}
  },
  "total": 500
}
```

| 关键字段 | 描述 |
|------|------|
| data[].question_id | 题目 ID |
| data[].common | 公共字段（input/messages/ref_answer） |
| data[].task_case_details | 各任务的 Case 详情（avg_score/tokens/score/payload） |
| task_info | `{task_id → {task_name, model_name, date}}` |
| total | 匹配的题目总数 |

#### 从 Badcase 构造评测集

通过 `avg_score_filter` 筛选得分 0 的 badcase，提取题目数据整理成 JSONL，再走数据集上传流程即可构造新评测集。

> ⚠️ **拉取前务必先评估数据量，与用户确认后再拉取：**
> 1. 先用 `page_size=1` 探查 badcase 总量：`{"task_ids": [...], "avg_score_filter": {"operator": "EQ", "value": 0}, "page_size": 1}`，返回的 `total` 就是 badcase 数量
> 2. 估算文件大小：每条 case 约含 2~20KB payload（模型回答越长越大），`预估大小 ≈ total × 10KB`，告知用户预计多少条、多大文件
> 3. badcase 率异常高（>90%）时提醒用户：可能模型配置有问题（模型选错、参数不对），而非模型本身能力差，建议先检查任务配置
> 4. 确认后再用 `page_size=100` 分页拉取，或用 `fetch_insight_cases.py` 脚本拉取到本地文件

```bash
# Step 1: 先探查 badcase 数量（不要直接拉全量）
python3 scripts/connect_mcp.py call list_taiji_eval_insight_cases '{"task_ids": [103851], "avg_score_filter": {"operator": "EQ", "value": 0}, "page_size": 1}'

# Step 2: 确认数量和大小后，分页拉取样本看数据结构
python3 scripts/connect_mcp.py call list_taiji_eval_insight_cases '{"task_ids": [103851], "avg_score_filter": {"operator": "EQ", "value": 0}, "page_size": 5}'

# Step 3: 大量数据用脚本拉取到本地文件（不要在 Agent 上下文中处理）
python3 scripts/fetch_insight_cases.py --task-ids 103851 --output ./badcases.jsonl
```

**提取字段说明：** 每条 case 返回以下关键字段用于构造数据集：

| 字段 | 用途 |
|------|------|
| `common.messages` | 原始输入题目（构造新数据集的 question，**核心字段**） |
| `payload.doc` | 上下文文档（可保留或丢弃） |
| `payload.responses` | 模型回答（构造评测集时通常丢弃，留作参考可） |
| `payload.gpt_response` | 标准答案（可保留为 ref_answer） |
| `question_id` | 题目 ID（建议保留用于追溯） |
| `exercise_version_id` / `exercise_name` | 来源评测集（建议保留用于分类） |

**构造 JSONL 后上传到平台：** 提取题目字段整理成数据集 JSONL 文件后，通过数据集两步上传流程创建数据集（详见 `dataset_management_api.md`）：

```bash
# 1. 上传 JSONL 文件 → 获取 ceph_path
python3 scripts/connect_mcp.py call upload_taiji_eval_dataset_file '{"file_name": "badcase_dataset.jsonl", "file_content_base64": "<base64编码>"}'
# 返回 file_path（Ceph 路径）

# 2. 用 ceph_path 创建数据集
python3 scripts/connect_mcp.py call create_taiji_eval_dataset '{"dataset_name": "badcase-103851", "dataset_version_name": "v1.0", "dataset_version_ceph_path": "<上一步返回的file_path>"}'
# 返回 dataset_id + dataset_version_id，可用于创建评测集
```

> ⚠️ 上传前需确认：JSONL 文件大小（base64 编码后体积增约 33%），过大时建议分批上传或用 Ceph 直传路径。

### 可视化分析（脚本）

`list_taiji_eval_insight_cases` 返回数据量大（单次 100 条约 2MB），不适宜直接在 Agent 上下文里展示。推荐通过脚本领端到拉取 → 可视化：

**Step 1：拉取数据到本地文件**

```bash
# 按 insight_id 拉取（自动获取关联 task + weight_nodes）
python3 scripts/fetch_insight_cases.py --insight-id 2550 --output ./cases.jsonl

# 按 task_ids 拉取
python3 scripts/fetch_insight_cases.py --task-ids 103674,103673,103671 --output ./cases.jsonl
```

**Step 2：启动可视化服务**

```bash
python3 scripts/serve_insight_compare.py --input ./cases.jsonl --port 8765
# → 浏览器打开 http://localhost:8765
```

> 💡 **更换数据或修改参数后需要重启服务**：服务端在启动时读取 JSONL 并注入到 HTML，运行时不会动态刷新数据。如需切换数据，`Ctrl+C` 停止服务后重新执行上述命令。

可视化页面支持：
- 📊 评测集 avg_score 柱状图 / 雷达图 / 得分分布 / Token 消耗
- 📋 覆盖率矩阵（task × 评测集 热力图）
- 📊 Collection 层级对比表（子维度权重 + 各 task 得分，支持展开/收起/筛选）
- 📋 明细对比表格（差值排序、搜索、点击展开题目内容）
- 左侧 task/评测集 多选筛选 + 分数滑块过滤

---


