# 工程链路评估编排（第 2 步 + 双任务查询）

> 完整「工程链路评估」= `run_adt_test_pipeline`（第 1 步，本 skill）+ 复制基准评测任务（第 2 步，跨模块 evaluation-skill）。

## 步骤

```text
① run_adt_test_pipeline(model, concurrency)  → 返回 ADT task_id
   ↓ 成功
② clone_taiji_eval_task（evaluation-skill，通过 `scripts/tool_manual.py clone_taiji_eval_task` 确认参数后调用，无需切换 skill）
   { source_task_id=PIPELINE_EVAL_SOURCE_TASK_ID(默认 29645),
     name="工程链路评估-{model}",
     service_name=<被测模型服务组名> }
   ⚠️ 第 ② 步失败不阻断第 ① 步：透传失败原因，保留 ADT task_id
   ↓
③ 用户问"工程链路评估状态"时，同时查两类任务：
   - get_adt_test_task_detail(ADT task_id, 字符串)
   - get_taiji_eval_task_detail(评测 task_id, 整数)
```

## 流转条件

| 从 | 到 | 条件 |
|----|----|------|
| ① | ② | ADT pipeline 成功返回 task_id |
| ② | complete | clone 异步执行，返回新评测 task_id（整数）即可，无需等待完成 |

## 第 2 步参数

> ⚙️ 基准源任务 ID = 可配置项 `PIPELINE_EVAL_SOURCE_TASK_ID`，默认 `29645`，测试环境可整体替换（见 SKILL.md §4.0）。

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `source_task_id` | int | ✅ | = `PIPELINE_EVAL_SOURCE_TASK_ID`，默认 `29645` |
| `name` | string | ✅ | `"工程链路评估-{model}"` |
| `service_name` | string | ✅(默认模板) | 被测模型服务组名。默认模板 `29645` 来源类型 `ONE_STOP_SERVICE`，故用此参数 |
| `model_ids` | — | 条件 | 若替换的模板来源为 `MODEL_REPOSITORY`/`MODEL_GROUP` 则改用此参数，以 evaluation-skill 口径为准 |

> 📌 默认模板 `29645` 自身可能 `STOP`/`serving_status=deleted`，属正常——克隆只复制评测配置，服务换成本次被测模型。
> ⏱️ `29645` 为 Agent 类重型评测（SWE-bench Verified + BrowseComp，`reasoning_effort=high`），耗时较长，非轻量测试。

**调用示例**：
```bash
python3 scripts/connect_mcp.py call clone_taiji_eval_task \
  '{"source_task_id": 29645, "name": "工程链路评估-hunyuan-standard", "service_name": "hunyuan-standard"}'
```

## 双任务查询对照

| 查询对象 | 工具 | task_id 类型 |
|----------|------|:---:|
| ADT 测试任务 | `get_adt_test_task_detail`（本 skill） | **字符串** |
| 评测任务 | `get_taiji_eval_task_detail`（evaluation-skill，通过 `scripts/tool_manual.py get_taiji_eval_task_detail` 确认参数后调用） | **整数** |

> ⚠️ 两类 task_id 类型不同，查询不可混淆。汇报时分别给出 ADT 四项检测结果与评测任务状态。
