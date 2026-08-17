## create_taiji_eval_collection

#### 参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `collection_name` | string | ✅ | 集合名称（全局唯一） |
| `desc` | string | ❌ | 集合描述 |
| `admin` | string | ❌ | 管理员用户名，多个用逗号分隔 |
| `visibility` | string | ❌ | 可见范围：`CURRENT_WORKSPACE`（默认）/ `ALL_PLATFORM` / `SPECIFIED_WORKSPACES` |
| `visible_ws_ids` | string | ❌ | 可见空间 ID 列表（`visibility=SPECIFIED_WORKSPACES` 时必填） |

#### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `collection_id` | number | 集合 ID（后续创建 CollectionVersion 必用） |
| `collection_name` | string | 集合名称 |
| `desc` | string | 描述 |
| `visibility` | string | 可见范围 |
| `creator` | string | 创建人 |
| `create_time` | string | 创建时间 |

#### 调用示例

```bash
python3 scripts/connect_mcp.py call create_taiji_eval_collection '{"collection_name": "my-collection-2026", "desc": "示例集合描述"}'
```

---

## create_taiji_eval_collection_version

创建 CollectionVersion，可传入 `weight_node` 权重树对象挂载 ExerciseVersion。不传 `weight_node` 则创建空版本，后续通过 `update_taiji_eval_collection_weight` 调整。

#### 参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `collection_id` | number | ✅ | 集合 ID（由 `create_taiji_eval_collection` 返回） |
| `version_name` | string | ✅ | 版本名称，例如 `v1.0` |
| `version_desc` | string | ✅ | 版本描述 |
| `weight_node` | object | ❌ | 权重树对象，字段见 [MultiModelEvaluationWeightNode](#multimodelevaluationweightnode-结构)。不传则创建空版本 |
| `admin` | string | ❌ | 管理员用户名，多个用逗号分隔 |

#### 返回字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `collection_version_id` | number | 集合版本 ID（创建评测任务时使用） |
| `collection_id` | number | 所属集合 ID |
| `version_name` | string | 版本名称 |
| `status` | string | 版本状态（初始为 PENDING） |
| `weight_node` | object | 权重树对象（传入时原样回显，未传则为 null） |

#### 示例

```bash
python3 scripts/connect_mcp.py call create_taiji_eval_collection_version '{
  "collection_id": 10,
  "version_name": "v1.0",
  "version_desc": "初版聚合评测",
  "weight_node": {
    "node_name": "root",
    "weight": 1.0,
    "exercise_id": null,
    "exercise_version_id": null,
    "nodes": [
      {"node_name": "ev_101", "weight": 0.5, "exercise_id": 10, "exercise_version_id": 101, "nodes": []},
      {"node_name": "ev_102", "weight": 0.5, "exercise_id": 11, "exercise_version_id": 102, "nodes": []}
    ]
  }
}'
```

---

## get_taiji_eval_collection_version_detail

#### 参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `collection_version_id` | number | ✅ | 集合版本 ID |

#### 返回字段

返回字段与 `create_taiji_eval_collection_version` 的返回字段一致，额外包含：完整 `weight_node` 权重树 JSON 和 `exercise_waiting_list`（备选区 EV 列表）。

#### 调用示例

```bash
python3 scripts/connect_mcp.py call get_taiji_eval_collection_version_detail '{"collection_version_id": 456}'
```

---

### Collection 完整调用流程（含 Exercise 链路）

> 用户想"从零搭建一整套评测"时走这里，而不是分别去读 dataset/exercise 文档各建一次、每次单独追问一个名字。

```
① dataset/create          → dataset_id + dataset_version_id
② exercise/create                    → exercise_id
③ exercise/create_version            → exercise_version_id（可重复多次，创建多个 EV）
④ collection/create                  → collection_id
⑤ collection/create_version          → 传入多个 exercise_version_id，自动均分权重
                                       → collection_version_id
⑥ 使用 collection_version_id 创建评测任务（支持多集聚合评测）
✅ 完成
```

**执行方式**：一次性向用户问齐下面的输入清单，问齐后连续执行①-⑤，不要每步都停下来追问；有可复用的现成 dataset/exercise 时先 `list_taiji_eval_datasets` / `list_taiji_eval_exercises` 确认，不必每次都新建。

**输入清单（哪些名字真正需要问用户）**：

| 名字 | 是否会出现在报告里 | 处理方式 |
|---|---|---|
| `dataset_name` / `dataset_version_name` | ❌ 从不展示 | 用户没有命名诉求时自动生成默认值（如 `{exercise_name}-dataset` / `v{yyyyMMdd}`），并告知已用默认值 |
| `exercise_name` / `exercise_version_name` / `metric_name` | ✅ 组合为 `{exercise_name}/{exercise_version_name}#{metric_name}`，即报告里每行（`weight_node.node_name`）的标签 | 必须向用户确认，每个 Exercise 单独问一次 |
| `collection_name` | ✅ 报告分组/表名（sheet 名） | 必须向用户确认 |
| `collection_version.version_name` | 内部版本标识，不单独展示 | 可用默认值 `v1.0`，用户无异议即用 |

需要多个 Exercise（即多个评测集）时，③按 Exercise 数量循环执行，可复用同一个 dataset，也可分别指定各自的数据。

> ⚠️ **严禁**：把"从零搭建"流程和"已有 `exercise_version_id`/`collection_version_id`，只是要建任务"混淆——后者直接用 `clone_taiji_eval_task`，跳过①-⑤。

---

### 评测集合（Collection）管理——查询/修改/删除

## list_taiji_eval_collections

分页查询集合列表。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call list_taiji_eval_collections '{"keyword": "评测", "page_index": 1, "page_size": 10}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `keyword` | string | ❌ | 按集合**名称**模糊匹配 |
| `my_collection` | boolean | ❌ | 仅查看我创建的集合（默认 false） |
| `page_index` | number | ❌ | 页码（1-based，默认 1） |
| `page_size` | number | ❌ | 每页数量（默认 10） |
| `order_by` | string | ❌ | 排序字段（默认 `id` 降序） |

---

## update_taiji_eval_collection

修改集合名称、描述或可见范围。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call update_taiji_eval_collection '{"collection_id": 10, "collection_name": "新名称", "collection_desc": "新描述"}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `collection_id` | number | ✅ | 集合 ID |
| `collection_name` | string | ❌ | 新名称 |
| `collection_desc` | string | ❌ | 新描述 |
| `admin` | string | ❌ | 管理员（可选） |
| `visibility` | string | ❌ | 可见范围：`CURRENT_WORKSPACE` / `ALL_PLATFORM` / `SPECIFIED_WORKSPACES` |
| `visible_ws_ids` | string | ❌ | 可见空间 ID 列表（`visibility=SPECIFIED_WORKSPACES` 时必填，多个用逗号分隔） |

---

## delete_taiji_eval_collection

删除集合。🔴 **不可逆操作**，执行前必须二次确认。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call delete_taiji_eval_collection '{"collection_id": 10}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `collection_id` | number | ✅ | 集合 ID |

---

## list_taiji_eval_collection_versions

查询集合的所有版本。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call list_taiji_eval_collection_versions '{"collection_id": 10}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `collection_id` | number | ❌ | 集合 ID |
| `keyword` | string | ❌ | 关键词搜索 |
| `collection_ids` | array\<number\> | ❌ | 集合 ID 列表，批量查询 |
| `collection_version_ids` | array\<number\> | ❌ | 版本 ID 列表，精确查询 |
| `page_index` | number | ❌ | 页码（1-based，默认 1） |
| `page_size` | number | ❌ | 每页数量（默认 10） |
| `order_by` | string | ❌ | 排序字段（默认 `id` 降序） |

---

## update_taiji_eval_collection_version

修改集合版本名称和描述。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call update_taiji_eval_collection_version '{"collection_version_id": 50, "version_name": "v2.0", "version_desc": "新版描述"}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `collection_version_id` | number | ✅ | 版本 ID |
| `version_name` | string | ✅ | 新版本名 |
| `version_desc` | string | ✅ | 新描述 |

---

## update_taiji_eval_collection_weight

构造、编辑评测集合版本的权重树（WeightNode）。⚠️ **全量替换**，传入的 `node` 需包含完整树结构，后端直接覆盖旧树。

> ⚠️ **参数名注意**：此接口使用 `cv_id`（**不是** `collection_version_id`）和 `node`（**不是** `weight_node`），与其他接口命名不同，传错参数名会报错。
> 📝 **精度说明**：`weight` 字段后端存储和 API 返回均保留原始精度，不会截断；页面展示时前端会格式化为两位小数，以页面显示为准即可。
>
> ⛔ **状态限制**：已上线（`status=RELEASED`）的集合版本不允许修改权重树，需先下线再改。

#### 参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|:---:|------|
| `cv_id` | number | ✅ | 集合版本 ID |
| `node` | object | ✅ | 权重树对象（**直接传 object，不要转 JSON 字符串**），结构见下方 |
| `exercise_waiting_list` | array | ❌ | 候选区 Exercise 列表（未进入树的 ExerciseVersion），结构见下方 |

#### WeightNode 节点结构

每个节点（无论根、中间、叶子）都是同一结构：

| 字段 | 类型 | 非叶子节点 | 叶子节点 | 说明 |
|------|------|:---:|:---:|------|
| `node_name` | string | ✅ 自定义名 | ✅ `{exercise_name}#{exercise_version_name}/{metric_name}` | 节点名称 |
| `weight` | float | ✅ | ✅ | 权重值（0~1，同级节点权重之和建议为 1.0） |
| `nodes` | array \| null | ✅ `[子节点...]` | `null` | 子节点列表，叶子节点为 null |
| `exercise_id` | long \| null | `null` | ✅ | 评测集 ID |
| `exercise_version_id` | long \| null | `null` | ✅ | 评测集版本 ID |
| `metric_name` | string \| null | `null` | ✅ | 指标名（如 `acc`） |

> 💡 根节点 `weight` 固定为 `1.0`，`exercise_id`/`exercise_version_id`/`metric_name` 均为 null。

#### exerciseWaitingList 结构

`list_taiji_eval_collection_versions` 返回的 `exerciseWaitingList` 中每一项：

```json
{
  "exercise_id": 1746,
  "exercise_version_id": 5157,
  "exercise_name": "uricornli_custom_paratest",
  "exercise_version_name": "uricornli_custom_paratest1",
  "metric_name": "acc"
}
```

把候选项加入树时，用这些字段构造叶子节点。

#### 完整调用示例

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call update_taiji_eval_collection_weight '{
  "cv_id": 50,
  "node": {
    "node_name": "根结点",
    "weight": 1.0,
    "exercise_id": null,
    "exercise_version_id": null,
    "metric_name": null,
    "nodes": [
      {
        "node_name": "一级",
        "weight": 1.0,
        "exercise_id": null,
        "exercise_version_id": null,
        "metric_name": null,
        "nodes": [
          {
            "node_name": "uricornli_custom_paratest#uricornli_custom_paratest1/acc",
            "weight": 0.5,
            "exercise_id": 1746,
            "exercise_version_id": 5157,
            "metric_name": "acc",
            "nodes": null
          },
          {
            "node_name": "floriawnang测试评测集#test111/acc",
            "weight": 0.5,
            "exercise_id": 1745,
            "exercise_version_id": 5165,
            "metric_name": "acc",
            "nodes": null
          }
        ]
      }
    ]
  },
  "exercise_waiting_list": []
}'
```

---

#### 权重树编辑操作指引

> 所有编辑操作的核心流程：**先查 → 内存改 → 全量提交**。`modifyWeightNode` 是全量覆盖，没有增量 API。

##### 标准流程

```
1. list_taiji_eval_collection_versions({collection_id: 816})
   → 取出 weightNode（当前树）+ exerciseWaitingList（候选项）
2. Agent 在内存中按用户意图修改树结构
3. update_taiji_eval_collection_weight({cv_id, node: 新树, exercise_waiting_list: 剩余候选项})
```

##### 常见操作

**① 从候选项加入叶子节点**

从 `exerciseWaitingList` 取一项，构造叶子节点插入目标位置：

```json
// 候选项: {"exercise_id": 1746, "exercise_version_id": 5157, "exercise_name": "xxx", "exercise_version_name": "yyy", "metric_name": "acc"}
// 构造叶子节点:
{
  "node_name": "xxx#yyy/acc",
  "weight": 0.5,
  "exercise_id": 1746,
  "exercise_version_id": 5157,
  "metric_name": "acc",
  "nodes": null
}
```
将该节点追加到目标父节点的 `nodes` 数组，同时从 `exerciseWaitingList` 中移除该候选项。

**② 删除叶子节点**

从父节点的 `nodes` 中移除目标节点。若该叶子有 `exercise_version_id`，需将其放回 `exerciseWaitingList`。

**③ 修改权重**

直接改对应节点的 `weight` 值。⚠️ 同级节点权重之和应保持 1.0。

**④ 新增中间分组节点**

在目标位置插入一个非叶子节点：
```json
{
  "node_name": "分组A",
  "weight": 1.0,
  "exercise_id": null,
  "exercise_version_id": null,
  "metric_name": null,
  "nodes": [...]
}
```

**⑤ 调整树层级（移动节点）**

将节点从原父节点 `nodes` 中移除，追加到新父节点的 `nodes` 中。

##### ⚠️ 注意事项

- **全量覆盖**：`node` 参数必须包含完整树（从根节点开始），后端不做 merge
- **RELEASED 禁改**：已上线版本调用会报错 `已上线的collection不允许修改`
- **根节点 weight 固定 1.0**
- **同级权重和**：建议同级节点 `weight` 之和为 1.0，否则评测分数可能不符合预期
- **叶子节点必须填 `exercise_id` + `exercise_version_id` + `metric_name`**，非叶子节点这三个字段为 null

##### 🔍 边界情况处理（加入评测集时）

当用户要求"把评测集 X 加入权重树"，但 X 不在 `exerciseWaitingList` 中时，Agent 必须先在 `weightNode`（递归遍历所有叶子节点）和 `exerciseWaitingList` 中查找匹配项，按下表决策：

| 子情况 | 判断依据 | Agent 应该怎么做 |
|--------|---------|-----------------|
| **X 已经在权重树里** | 在 `weightNode` 叶子节点中找到匹配的 `exercise_version_id` | 告知用户：「该评测集已在树中（位置：`<根→...→父节点路径>`，权重：`<weight>`），是否需要调整权重或位置？」 |
| **X 没导入到此 collection version** | `exerciseWaitingList` 和 `weightNode` 中都找不到匹配项 | ⛔ 告知用户：「该评测集未导入到此集合版本，当前后端不支持追加导入，需新建一个集合版本时带入该 ExerciseVersion」 |
| **X 根本不存在 / 名称匹配不到** | 查不到任何对应 ExerciseVersion（可通过 `list_taiji_eval_exercise_versions` 搜索确认） | 提示用户：「未找到名称/ID 为 `<X>` 的评测集，请检查名称或 ID 是否正确」 |

> 💡 **匹配优先级**：优先按 `exercise_version_id`（数值）精确匹配；用户只提供名称时，按 `exercise_name` + `exercise_version_name` 模糊匹配，命中多个时需向用户确认具体哪一个。

---

## delete_taiji_eval_collection_version

删除集合版本。🔴 **不可逆操作**，执行前必须二次确认。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call delete_taiji_eval_collection_version '{"collection_version_id": 50}'
```


| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `collection_version_id` | number | ✅ | 版本 ID |

---

### MultiModelEvaluationWeightNode 结构

`node` 参数对应的权重树节点结构，递归定义：

| 字段 | 类型 | 说明 |
|------|------|------|
| `node_name` | string | 节点名称，叶子节点格式：`{exercise_name}/{version_name}#{metric_name}` |
| `weight` | float | 权重值（根节点为 1.0，子节点权重之和应等于父节点权重） |
| `exercise_id` | number | 评测集 ID（仅叶子节点有值） |
| `exercise_version_id` | number | 评测集版本 ID（仅叶子节点有值） |
| `metric_name` | string | 指标名称（仅叶子节点有值） |
| `nodes` | array\<MultiModelEvaluationWeightNode\> | 子节点列表（非叶子节点有值，叶子节点为空数组 `[]`） |

---

## release_taiji_eval_collection_version

发布（上线）或下线评测集合版本。合并到一个工具中，通过 `action` 参数区分操作类型。

#### 参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|:---:|------|
| `collection_version_id` | number | ✅ | 集合版本 ID |
| `action` | string | ✅ | 操作类型：`online`（上线）或 `offline`（下线） |

#### ⚠️ Agent 行为规则（调用前必读）

**调用前必须确认当前 status，然后按状态分支表决策：**
- 若当前上下文中已明确知道 status（如刚执行了 `create_taiji_eval_collection_version` 返回 status=PENDING，或上一步`release` 返回了新 status），可直接使用已知 status，无需额外查询。
- 若 status 未知，则必须先调`get_taiji_eval_collection_version_detail` 查询。

| 当前 status | action=online | action=offline |
|-------------|:------------:|:--------------:|
| **PENDING**（未上线） | ✅ 调 API，正常上线 | ⛔ **反问用户**：该版本尚未发布，是否需要上线？ |
| **RELEASED**（已上线） | ⛔ **反问用户**：该版本已发布，是否需要下线？ | ✅ 调 API，正常下线 |

> 🔄 **"反问" 规则**：当命中 ⛔ 分支时，Agent 必须使用 `AskUserQuestion` 工具反问用户，等待用户确认后再执行。不得跳过反问直接调用 API。

#### 成功返回

```json
{
    "success": true,
    "errMessage": null,
    "data": {
        "id": 3122,
        "collectionId": 816,
        "versionName": "v1",
        "versionDesc": "v1test",
        "status": "RELEASED",
        ...
    }
}
```

#### 调用示例

**上线操作：**
```bash
python3 scripts/connect_mcp.py call release_taiji_eval_collection_version '{"collection_version_id": 3122, "action": "online"}'
```

**下线操作：**
```bash
python3 scripts/connect_mcp.py call release_taiji_eval_collection_version '{"collection_version_id": 3122, "action": "offline"}'
```

#### 典型调用流程

```
1. get_taiji_eval_collection_version_detail({collection_version_id: 3122})
   → status: "PENDING"
2. 用户意图: "上线" → action=online, status=PENDING → ✅ 直接调 publish
3. release_taiji_eval_collection_version({collection_version_id: 3122, action: "online"})
   → 返回 status: "RELEASED"
```

或：

```
1. get_taiji_eval_collection_version_detail({collection_version_id: 3122})
   → status: "RELEASED"
2. 用户意图: "上线" → action=online, status=RELEASED → ⛔ 反问
3. AskUserQuestion: "该版本已是上线状态（RELEASED），是否需要帮你下线？"
4. 用户确认 → 调 publish({action: "offline"})
```

---

