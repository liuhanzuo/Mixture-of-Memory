# validloss — 伴生评估 loss 数据查询

## 📌 本模块流程契约（最高优先级）

> 本段是 SKILL.md 路由进入本文档后的统领规则，覆盖本文档所有章节。

- **何时进入**：用户问"validloss / 伴生 loss / 伴生评估 loss / 验证 loss / 验证集 loss / 主任务 loss / companion validloss"等任一意图，或用户给出一个**纯数字主任务 ID**（如 `199276`、`450968`）并要求查其 validloss / 伴生 loss。
- **核心概念**：validloss 不是主任务自身的训练 loss，而是**主任务在训练过程中自动触发的伴生 SFT 评估任务**上报到 SwanLab 的指标数据。用户给的是**主任务的纯数字 taskId**（即 jobGroupId），需先用 `list_taiji_eval_companion_tasks` 解析出伴生 SFT 任务，再查 SwanLab 指标。
- **关键：taskId 和 wsid 的获取**：
  - 用户可能直接提供**纯数字 taskId**（如 `199276`），也可能提供**任务详情页 URL**
  - URL 示例：`https://hunyuanaide.taiji.woa.com/web/pre_training_inst_list?taskId=450968&taskID=basic_train_doublecxu_20260530075920_49f56698&name=xxx&tab=output&wsId=10103&instId=xxx`
  - 从 URL 中提取：
    - `taskId` 参数值 → 纯数字 ID（如 `450968`），**注意不是 `taskID`（字符串格式）**
    - `wsId` 参数值 → 工作空间 ID（如 `10103`）
  - `list_taiji_eval_companion_tasks` 的 `task_id` 参数是**纯数字 ID**（如 `199276`），即主任务的 jobGroupId，**不是** `basic_train_xxx` 字符串格式
- **关键：指标名格式**：
  - 伴生 validloss 任务的 SwanLab 指标名**不是** `lm loss`，而是带前缀和后缀的格式：`data/lm loss/{序号}.{数据集名} validation`
  - 例如：`data/lm loss/36.SpatialEval_spatialreal validation`
  - **必须先用 `query_hunyuan_swanlab_run_columns` 列出所有可用指标名**，再用完整指标名查数据
- **关键：SwanLab 认证**：
  - Step 1（`list_taiji_eval_companion_tasks`）返回值中包含 `swanlab_api_token`、`swanlab_workspace`、`swanlab_project`、`swanlab_run_id`
  - ⚠️ **字段名注意**：Step 1 返回的字段名是 `swanlab_api_token`（不是 `swanlab_api_key`），但 Step 2/3 的 SwanLab 工具入参名是 `swanlab_api_key`——取 Step 1 的 `swanlab_api_token` 值，赋给 Step 2/3 的 `swanlab_api_key` 参数
  - 这四个参数直接传给 Step 2 和 Step 3 的 SwanLab 工具，**用户无需手动提供 SwanLab API Key**
- **进入后必做**：
  1. 获取主任务的**纯数字 `task_id`** 和 **`ws_id`**。用户可能直接提供纯数字 taskId，也可能提供 URL（从 URL 的 `taskId` 和 `wsId` 参数提取）。如果用户给的是 `basic_train_xxx` 字符串格式（即 URL 中的 `taskID` 参数，不是 `taskId`），需引导用户提供纯数字 ID 或 URL。无 task_id 时直接退回让用户补全，**严禁**猜测主任务 ID。
  2. **Step 1**：调 `list_taiji_eval_companion_tasks(task_id=<纯数字主任务 ID>, ws_id)`。如果用户**已提供 `trigger_name`**（如用户说"查 valid_full_set 的 validloss"），直接带上 `trigger_name` 查伴生任务。如果用户**没提供 `trigger_name`**，不传该参数，返回触发器列表，展示给用户选择后再带 `trigger_name` 查伴生任务。
  3. **Step 2**：对 Step 1 返回的伴生任务，调 `query_hunyuan_swanlab_run_columns(swanlab_api_key=<swanlab_api_token>, workspace=<swanlab_workspace>, project_name=<swanlab_project>, run_id=<swanlab_run_id>)` 列出可用指标名。展示给用户，让用户选择要查哪些指标。
  4. **Step 3**：用用户选择的指标名，调 `query_hunyuan_swanlab_run_metrics(swanlab_api_key=<swanlab_api_token>, workspace=<swanlab_workspace>, project_name=<swanlab_project>, run_id=<swanlab_run_id>, keys=[<用户选择的指标名>], tail=50)` 查具体数据。
  5. 汇总展示：按 step 排序，每条伴生任务一条 lm loss 曲线，标注其 ckpt step。
- **伴生任务数量处理**：
  - 伴生任务 ≤ 5 个 → 全部查询并展示
  - 伴生任务 > 5 个 → 默认查前 5 条，并提示"还有 N 条伴生任务，需要继续查吗？"
- **严禁**：
  - ❌ 拿主任务纯数字 task_id 直接调 SwanLab 工具查 lm loss——主任务的 SwanLab 数据不是 validloss，必须先经 `list_taiji_eval_companion_tasks` 解析伴生任务
  - ❌ 把 `basic_train_xxx` 字符串格式的 task_id 传给 `list_taiji_eval_companion_tasks`——该工具只接受纯数字 jobGroupId
  - ❌ 把 validloss 当作评测得分 / 评测结果（→ `evaluation-skill`）——validloss 是 SwanLab 上报的训练 loss 时序数据，不是评测分数
  - ❌ 把 validloss 当作平台预定义训练指标（→ `metric-skill` 的 `query_training_metrics`）——validloss 走 SwanLab
  - ❌ 在伴生任务列表为空时编造 `items[].task_id`——如实告知"该主任务暂无伴生 SFT 评估任务"
  - ❌ 用 `keys=["lm loss"]` 查伴生任务——伴生 validloss 的指标名是 `data/lm loss/{序号}.{数据集} validation` 格式，必须先调 `query_hunyuan_swanlab_run_columns` 获取完整指标名
  - ❌ 跳过 Step 2 直接猜指标名——必须先列出可用指标让用户选择
- **失败处理**：
  - `list_taiji_eval_companion_tasks` 返回 `total=0` → 告知用户"该主任务暂无伴生 SFT 评估任务，无法查 validloss"，不自动 fallback
  - `query_hunyuan_swanlab_run_columns` 返回空列表 → 告知用户"该伴生任务尚未上报 SwanLab 数据"，跳过该任务
  - `query_hunyuan_swanlab_run_metrics` 返回 `metrics: []` → 告知用户"该指标暂无数据"，继续查其他指标
  - 401/403 → 提示 token/api_key 失效；网络/超时 → 如实告知，不重试、不切换工具

---

## 🔗 工具链（三步串联）

```
用户: "查 taskId 199276 的 validloss"
  或: "查 https://hunyuanaide.taiji.woa.com/web/pre_training_inst_list?taskId=450968&...&wsId=10103 的 validloss"
   │
   ▼
解析输入：
  - 用户提供了纯数字 taskId → 直接使用
  - 用户提供了 URL → 从中提取 taskId（纯数字）和 wsId
  - 用户提供了 basic_train_xxx 字符串 → 引导用户提供纯数字 ID 或 URL
   ▼
Step 1: list_taiji_eval_companion_tasks(task_id="199276", ws_id=10103)
   │     用户没提供 trigger_name → 返回触发器列表
   │     用户提供了 trigger_name → 返回该触发器下的伴生任务
   │     返回：伴生任务列表（含 swanlab_run_id、swanlab_project、swanlab_api_token、swanlab_workspace）
   │     ⚠️ 注意字段名是 swanlab_api_token，不是 swanlab_api_key
   ▼
Step 2: query_hunyuan_swanlab_run_columns(swanlab_api_key, workspace, project_name, run_id)
   │     列出可用指标名（如 data/lm loss/36.SpatialEval_spatialreal validation）
   │     → 展示给用户，让用户选择要查哪些指标
   ▼
Step 3: query_hunyuan_swanlab_run_metrics(swanlab_api_key, workspace, project_name, run_id, keys, tail=50)
   │     查具体指标数据
   │     返回：step + value 对
   ▼
汇总展示: 多条 lm loss 曲线（按 step 对齐）
```

> ⚠️ **三步缺一不可**：查伴生任务 → 列指标名 → 查数据。严禁跳过任何一步。

---

## Step 1 工具：list_taiji_eval_companion_tasks

根据**主任务的纯数字 task_id**（jobGroupId）查询伴生 SFT 评估任务列表。

> 后端链路：调 `/v1/hunyuan/evaluation/companion/task/list`，后端自动查触发器列表 + 事件列表，解析 SFT 节点，提取 SwanLab 参数。

**根据 `trigger_name` 是否传入，行为不同**：
- **未传 `trigger_name`** → 返回触发器列表，让用户选择
- **传了 `trigger_name`** → 只查该触发器下的伴生任务，返回 items 列表

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `task_id` | string | ✅ | 无 | **主任务的纯数字 ID**（jobGroupId，如 `199276`），**不是** `basic_train_xxx` 字符串格式 |
| `ws_id` | int | ✅ | 无 | 工作空间 ID，正整数 |
| `trigger_name` | string | 否 | 空 | 触发器名称（如 `valid_full_set`）。不传时返回触发器列表供选择；传入时只查该触发器下的伴生任务 |
| `page` | int | 否 | `1` | 分页页码 |
| `page_size` | int | 否 | `20` | 每页条数 |

**返回格式（未传 trigger_name，触发器列表）：**

```json
{
  "triggers": [
    {"id": 3569, "name": "update_visual_reasoning", "trigger_type": "AUTO", "event_type": "SFT"},
    {"id": 3527, "name": "valid_full_set", "trigger_type": "AUTO", "event_type": "SFT"},
    ...
  ],
  "total": 9,
  "message": "请选择要查询的触发器名称，带上 trigger_name 参数重新调用本工具"
}
```

**返回格式（传了 trigger_name，伴生任务列表）：**

```json
{
  "items": [
    {
      "task_id": "basic_train_pluspluswu_20260702145836_63957d23",
      "step": "step36000",
      "checkpoint": "/apdcephfs_tj5/.../iter_0036000",
      "trigger_id": 3569,
      "swanlab_run_id": "debug_valid_loss_3569_1024_4k_valid_full_set_ver260629",
      "swanlab_project": "valid_loss_exp_collections",
      "swanlab_api_token": "QG8SxqO6XDeiG9lYVYaYp",
      "swanlab_workspace": "X1"
    }
  ],
  "total": 36,
  "page": 1,
  "page_size": 20,
  "has_more": true
}
```

**返回字段说明（items[]）：**

| 字段 | 说明 |
|------|------|
| `task_id` | 伴生 SFT 任务 ID（`basic_train_` 前缀） |
| `step` | checkpoint 步数（如 `step36000`） |
| `checkpoint` | checkpoint 路径 |
| `trigger_id` | 触发器 ID |
| `swanlab_run_id` | 从 `startCmd` 解析的 SwanLab run_id（用于 Step 2/3 的 `run_id` 参数） |
| `swanlab_project` | SwanLab 项目名（用于 Step 2/3 的 `project_name` 参数） |
| `swanlab_api_token` | SwanLab API Key（**注意字段名是 `swanlab_api_token`**，用于 Step 2/3 的 `swanlab_api_key` 参数，**用户无需手动提供**） |
| `swanlab_workspace` | SwanLab 工作空间名（用于 Step 2/3 的 `workspace` 参数，默认 `X1`） |

---

## Step 2 工具：query_hunyuan_swanlab_run_columns

列出 SwanLab 实验的可用指标列（指标名列表）。

> 本工具属于 SwanLab 模块（非 evaluation 模块），直接调 SwanLab OpenAPI。

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `swanlab_api_key` | string | ✅ | 无 | SwanLab API Key（取 Step 1 返回的 `swanlab_api_token` 字段值，赋给此处的 `swanlab_api_key` 参数） |
| `workspace` | string | ✅ | 无 | SwanLab 工作空间名（从 Step 1 的 `swanlab_workspace` 获取） |
| `project_name` | string | ✅ | 无 | SwanLab 项目名（从 Step 1 的 `swanlab_project` 获取） |
| `run_id` | string | ✅ | 无 | SwanLab 实验 ID（从 Step 1 的 `swanlab_run_id` 获取） |
| `search` | string | 否 | 无 | 模糊搜索关键词 |
| `column_class` | string | 否 | 无 | CUSTOM / SYSTEM |

> 💡 **Agent 行为**：展示指标列时，将 `key` 以 `data/lm loss/` 开头的指标高亮展示，引导用户选择要查哪些数据集的 loss。

---

## Step 3 工具：query_hunyuan_swanlab_run_metrics

查询 SwanLab 实验的标量指标数据。

> 本工具属于 SwanLab 模块（非 evaluation 模块），直接调 SwanLab OpenAPI。

**参数：**

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `swanlab_api_key` | string | ✅ | 无 | SwanLab API Key（取 Step 1 返回的 `swanlab_api_token` 字段值） |
| `workspace` | string | ✅ | 无 | SwanLab 工作空间名 |
| `project_name` | string | ✅ | 无 | SwanLab 项目名 |
| `run_id` | string | ✅ | 无 | SwanLab 实验 ID |
| `keys` | array | ✅ | 无 | 指标名列表（如 `["data/lm loss/36.SpatialEval_spatialreal validation"]`），必须是完整指标名（从 Step 2 获取） |
| `step_start` | int | 否 | 无 | step 区间起点 |
| `step_end` | int | 否 | 无 | step 区间终点 |
| `head` | int | 否 | 无 | 取前 N 个数据点 |
| `tail` | int | 否 | 无 | 取后 N 个数据点 |
| `sample` | int | 否 | 无 | 采样数量 |

> ⚠️ `head` / `tail` / `step_start+step_end` / `sample` 互斥。

---

## 展示规范

### 多任务汇总展示规则

1. **按 step 对齐**：以 step 为横轴，每个伴生任务一条 lm loss 曲线。
2. **标注 ckpt step**：每条曲线标注其对应的 checkpoint step。
3. **数据表格**：以 step 为行、各伴生任务的 lm loss 为列。
4. **任务数 > 5**：默认展示前 5 条 + 提示"还有 N 条伴生任务，需要继续查吗？"。

### 空结果展示

当 `list_taiji_eval_companion_tasks` 返回 `total=0` 时：

```
该主任务 {main_task_id} 暂无伴生 SFT 评估任务，无法查询 validloss。

可能原因：主任务尚未触发伴生评估，或伴生任务未生成。
建议：请在太极平台确认该主任务已配置伴生评估并已触发。
```
