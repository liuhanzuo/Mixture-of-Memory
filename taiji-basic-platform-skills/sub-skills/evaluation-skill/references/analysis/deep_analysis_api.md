## trigger_taiji_eval_deep_analysis

在数据库创建一条分析记录（初始状态，summary 为空）。这是闭环的第一步，后续你需要自己完成分析并回写。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call trigger_taiji_eval_deep_analysis '{
  "analysis_type": "INSIGHT",
  "target_id": 505,
  "user_prompt": "分析所有任务的模型表现"
}'
```

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| analysis_type | string | ✅ | `INSIGHT` 或 `EVAL_TASK` |
| target_id | long | ✅ | insightId 或 taskId |
| user_prompt | string | ✅ | 用户的分析需求描述 |

**返回：** `{"analysisId": 1, "status": "RUNNING", "message": "深度分析任务已触发"}`

---

## get_taiji_eval_task_scores

获取指定 task 的各评测集分数结果。**这是分析的核心数据来源。**

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_task_scores '{"task_id": 101677}'
```

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | long | ✅ | 评测任务 ID |

**返回示例：**
```json
{
  "count": 1,
  "scores": [{
    "taskId": 101677,
    "exerciseVersionId": 2631,
    "exerciseNameAndEvName": "dop_test/general",
    "scores": "{\"acc\":100.0,\"bon_acc\":100.0,\"error_rate\":0.0,\"avg_completion_token\":895}",
    "totalCount": 2,
    "completedCount": 2
  }]
}
```

**如何使用返回数据：** `scores` 字段是 JSON 字符串，需要解析提取指标。常见指标有 `acc`、`bon_acc`、`error_rate`、`truncation_rate`、`avg_completion_token`、`avg_prompt_token`。

---

## get_taiji_eval_metric_scores

**⚡ 定位 Badcase 的核心工具。** 获取每道题的 avgScore/passNum/inferStatus，快速筛出分数为 0 的 question_id。

典型用法：
1. 调用 `analysis/query_metric_scores(task_id=X, exercise_version_id=Y)` 获取该评测集所有题目的分数
2. 筛选 `avgScore == 0` 或 `inferStatus != "DONE"` 的记录
3. 这些 `questionId` 就是 badcase，直接写入报告

**不要跳过此步骤直接调 analysis/query_case_detail 盲拉！** metric_scores 是毫秒级轻量查询，能精确定位问题题目。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_metric_scores '{"task_id": 101677}'
```

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | long | ✅ | 评测任务 ID |
| exercise_version_id | long | 否 | 不传则返回所有评测集 |

**返回：** `{"count": N, "metrics": [{taskId, questionId, avgScore, maxScore, minScore, ...}]}`

#### 从高频错题构造评测集

对多个任务调用 `get_taiji_eval_metric_scores`，交叉比对找出在多个任务中都得 0 分的 question_id（高频错题），再用 `list_taiji_eval_insight_cases`（见 `insight_management_api.md`）的 `question_ids` 参数精确拉取这些题目的完整明细，提取题目数据构造新评测集。

> ⚠️ **拉取前务必先评估数据量，与用户确认后再拉取：**
> 1. `get_taiji_eval_metric_scores` 返回的是轻量分数数据（毫秒级，每条约 200 字节），可直接全量拉取做交叉比对，不用担心数据量
> 2. 交叉比对后得到的高频错题数量通常较少（几十到几百条），但用 `list_taiji_eval_insight_cases` 拉 case 明细时每条约 2~20KB，需评估总大小
> 3. 先告知用户：高频错题共多少条、预计文件多大，确认后再拉取 case 明细

```bash
# Step 1: 分别查多个任务的逐题分数（轻量查询，直接全量拉取）
python3 scripts/connect_mcp.py call get_taiji_eval_metric_scores '{"task_id": 103851}'
python3 scripts/connect_mcp.py call get_taiji_eval_metric_scores '{"task_id": 103835}'

# Step 2: 客户端交叉比对 → 找出多个任务都得 0 分的 question_id（高频错题）
#   - 按 exercise_version_id 分组
#   - 同一 exercise_version_id 下，取所有任务 avgScore=0 的 question_id 交集
#   - 这些题目就是"高频错题"

# Step 3: 用 question_ids 精确拉取 case 明细（含原始题目）
python3 scripts/connect_mcp.py call list_taiji_eval_insight_cases '{"task_ids": [103851], "question_ids": [238593100, 238593101], "page_size": 100}'
```

**提取字段说明：** case 明细返回的字段与 badcase 构造相同，核心字段：

| 字段 | 用途 |
|------|------|
| `common.messages` | 原始输入题目（构造新数据集的 question，**核心字段**） |
| `payload.doc` | 上下文文档（可保留或丢弃） |
| `payload.gpt_response` | 标准答案（可保留为 ref_answer） |
| `question_id` | 题目 ID（建议保留用于追溯） |
| `exercise_version_id` / `exercise_name` | 来源评测集（建议保留用于分类） |

**构造 JSONL 后上传到平台：** 提取题目字段整理成数据集 JSONL 文件后，通过数据集两步上传流程创建数据集（详见 `dataset_management_api.md`）：

```bash
# 1. 上传 JSONL 文件 → 获取 ceph_path
python3 scripts/connect_mcp.py call upload_taiji_eval_dataset_file '{"file_name": "high_freq_errors.jsonl", "file_content_base64": "<base64编码>"}'
# 返回 file_path（Ceph 路径）

# 2. 用 ceph_path 创建数据集
python3 scripts/connect_mcp.py call create_taiji_eval_dataset '{"dataset_name": "high-freq-errors", "dataset_version_name": "v1.0", "dataset_version_ceph_path": "<上一步返回的file_path>"}'
# 返回 dataset_id + dataset_version_id，可用于创建评测集
```

> 高频错题 = 多个模型在同一道题上都得 0 分，说明题目本身可能存在难度或歧义，适合作为重点评测样本持续跟踪模型改进情况。

---

## get_taiji_eval_case_detail

获取评测明细（含 payload 大字段，秒级响应）。**典型用法：先用 analysis/query_metric_scores 定位 badcase 的 question_id，再用本接口按 question_id 精确获取该题的完整 payload。**

**MCP 工具调用：**
```bash
# 获取特定 question_id 的明细（推荐用法）
python3 scripts/connect_mcp.py call get_taiji_eval_case_detail '{"task_id": 23658, "exercise_version_id": 1157, "question_id": 240289700}'

# 批量获取（不指定 question_id）
python3 scripts/connect_mcp.py call get_taiji_eval_case_detail '{"task_id": 23658, "exercise_version_id": 1157, "limit": 100}'
```

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_id | long | ✅ | 评测任务 ID |
| exercise_version_id | long | 否 | 不传则返回所有 |
| question_id | long | 否 | 题目 ID，指定后只返回该题明细。**从 analysis/query_metric_scores 的 badcase 中获取** |
| limit | int | 否 | 默认 1000，硬上限 5000 |

**返回：** `{"total": N, "returned": M, "truncated": bool, "cases": [{id, taskId, questionId, payload}]}`

---

## upload_taiji_eval_analysis_file

**上传分析报告获取 HIFS 链接。** 在生成完整 Markdown 报告后调用，获得 `filePath` 作为 `upload_taiji_eval_analysis_result` 的 `report_url`。

> ⚠️ **file_content_base64 特殊字符处理（必读）：** 报告内容在 JSON 传输时，以下字符会导致截断或解析失败，**必须避免或转义**：
> 1. **不要使用 Markdown 表格的 `|` 竖线分隔符** — 改用列表或缩进格式展示对比数据
> 2. **不要使用反引号 `` ` `` 包裹的代码块** — 改用缩进 4 空格表示代码
> 3. **不要在内容中使用未转义的双引号 `"`** — 用 `\"` 转义，或改用单引号 `'`
> 4. **不要使用反斜杠 `\`** — 用 `/` 或省略
> 5. **换行符用 `\n`** — 不要用真实换行
> 6. **内容尽量简洁** — 每条 case 只输出关键字段（question_id、score、instance_id、结论），不要复制完整 payload

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call upload_taiji_eval_analysis_file '{
  "user": "zixinchen",
  "file_content_base64": "# Analysis Report\n\n## Overview\n...",
  "file_name": "insight_505_analysis.md"
}'
```

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| file_name | string | ✅ | 报告文件名（例如 `insight_505_analysis.md`） |
| file_content_base64 | string | ✅ | 文件内容（Markdown 报告全文） |
| user | string | ❌ | 上传用户名，缺省取 X-Auth-Token 用户 |

**返回示例：**
```json
{
  "fileId": 200,
  "filePath": "/taiji-offline-inference-hifs/taiji-eval/analysis/2026-06-28/user/report_xxx.md",
  "fileName": "insight_505_analysis.md",
  "hifsUrl": "/taiji-offline-inference-hifs/taiji-eval/analysis/2026-06-28/user/report_xxx.md"
}
```

**使用方式：** 返回的 `filePath` 直接作为 `upload_taiji_eval_analysis_result` 的 `report_url` 参数。

---

## upload_taiji_eval_analysis_result

**闭环的最后一步：将你的分析结果写回平台。** 必须在完成分析后调用。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call upload_taiji_eval_analysis_result '{
  "analysis_id": 5,
  "analysis_type": "INSIGHT",
  "target_id": 505,
  "user_prompt": "分析所有任务的模型表现",
  "summary": "{\"columns\":[\"指标\",\"模型A\"],\"rows\":[[\"ACC\",\"100%\"]]}",
  "report_url": "/taiji-eval/analysis/2026-06-28/insight_505.md",
  "creator": "username"
}'
```

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| analysis_id | long | ✅ | Step 2 返回的 analysisId，用于精准匹配回写 |
| analysis_type | string | ✅ | `INSIGHT` 或 `EVAL_TASK` |
| target_id | long | ✅ | insightId 或 taskId |
| user_prompt | string | ✅ | 用户的分析需求描述 |
| summary | string | ✅ | JSON 字符串，格式 `{"columns":[...],"rows":[[...]]}` |
| report_url | string | ✅ | 报告 HIFS 地址（可用占位路径） |
| creator | string | ✅ | 创建人用户名 |

**返回：** `{"id": 2, "createTime": "2026-06-28 23:37:41"}`

---

## heartbeat_taiji_eval_analysis

**分析过程中定时调用，刷新 updateTime 防止被后端超时误判为 FAILED。** 仅 RUNNING 状态可调用。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call heartbeat_taiji_eval_analysis '{"analysis_id": 5}'
```

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| analysis_id | long | ✅ | 分析记录 ID（Step 2 返回的 analysisId） |

**返回：** `{"code": 0, "message": "success"}`

---

## list_taiji_eval_analysis_results

查询已有的分析结果。用户问"看看之前的分析"时调用。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call list_taiji_eval_analysis_results '{"analysis_type": "INSIGHT", "target_id": 505}'
```

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| analysis_type | string | ❌ | 分析类型，默认 `INSIGHT` |
| target_id | long | ❌ | 目标 ID，不传则返回该类型下全部记录 |

**返回：** `{"count": N, "results": [{id, analysisType, targetId, summary, reportUrl, creator, createTime}]}`

---

## get_taiji_eval_analysis_detail

查询单条分析结果。

**MCP 工具调用：**
```bash
python3 scripts/connect_mcp.py call get_taiji_eval_analysis_detail '{"analysis_id": 2}'
```

**参数：**

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| analysis_id | long | ✅ | 分析记录 ID |

**返回：** 完整的分析记录，含 summary 表格和 reportUrl。

---

### 完整示例：INSIGHT 分析闭环

以下是一个完整的执行示例，展示 Agent 应该如何处理"分析 Insight 505 的模型表现"：

```
1. 调用 analysis/trigger(analysis_type="INSIGHT", target_id=505, user_prompt="分析 Insight 505 的模型表现")
   → 获得 analysisId=5, status=RUNNING

2. 调用 insight/query_detail(insight_id=505) 获取关联 taskIds，选取 task_id=23658 作为重点分析对象

3. 调用 analysis/query_task_scores(task_id=23658)
   → 获得 64 个评测集的聚合分数，找到表现最差的：
     swe_bench_verified: ACC=73.2%, exerciseVersionId=1157
     arc_agi_v2: ACC=6.88%
     CLBench: ACC=15.88%

4. 对重点评测集调用 analysis/query_metric_scores(task_id=23658, exercise_version_id=1157)
   → 获得 500 个题目的逐题分数，筛选 avgScore=0 的记录：
     question_id=240289900 (astropy__astropy-13398) avgScore=0
     question_id=240294600 (django__django-11477) avgScore=0
     question_id=240303600 (django__django-14034) avgScore=0
     ... 共 132 个 badcase，都有具体 question_id

5. [按需] 对特别值得深挖的 question_id 调用 analysis/query_case_detail 获取完整 payload
   → 查看模型的具体输入/输出/推理过程

6. 分析数据，生成 summary（含 question_id）+ 完整 Markdown 报告:
   summary = {
     "columns": ["评测集", "ACC", "Error Rate", "总题目数", "典型 badcase question_id"],
     "rows": [
       ["swe_bench_verified", "73.2%", "0.4%", "500", "240289900,240294600,240303600"],
       ["arc_agi_v2", "6.88%", "1.2%", "835", "xxx,yyy,zzz"],
       ...
     ]
   }
   report_md = "# Task 23658 Badcase 分析\n\n## Badcase 列表\n| question_id | ... |"

7. 调用 analysis/upload_file(user="username", file_content_base64=report_md, file_name="...")
   → 获得 filePath（HIFS 链接）

8. 调用 analysis/upload_result(
     analysis_id=<Step 2 返回的 analysisId>,
     analysis_type="INSIGHT", target_id=505,
     summary="<summary JSON>",
     report_url="<Step 7 的 filePath>",
     creator="username"
   )
   → 更新 RUNNING 记录为 COMPLETED

9. 向用户展示分析结果表格 + 洞察总结
```

### 关键：analysis/query_metric_scores 是定位 badcase 的正确方式

❌ **错误做法**：跳过 metric_scores，直接调 analysis/query_case_detail 盲拉 500 条再自己过滤
✅ **正确做法**：先调 analysis/query_metric_scores（毫秒级），筛选 avgScore=0 的 question_id，再按需对特定 case 调 analysis/query_case_detail

---

### 性能与限制

| 工具 | 预期耗时 | 限制 |
|------|----------|------|
| `trigger_taiji_eval_deep_analysis` | < 200ms | - |
| `list_taiji_eval_analysis_results` / `get_taiji_eval_analysis_detail` | < 200ms | MySQL 单表 |
| `get_taiji_eval_task_scores` / `get_taiji_eval_metric_scores` | < 500ms | StarRocks 聚合小表 |
| `get_taiji_eval_case_detail` | 1~5s | limit 硬上限 5000 |
| `upload_taiji_eval_analysis_file` | < 3s | 文件最大 100MB |
| `upload_taiji_eval_analysis_result` | < 200ms | 单条 INSERT |
| `heartbeat_taiji_eval_analysis` | < 200ms | 单条 UPDATE |
