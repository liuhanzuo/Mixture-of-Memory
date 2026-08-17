---
name: evaluation-skill
description: 太极平台模型评测子 skill —— 通过 MCP 协议查询评测结果/得分/进展、管理 Insight、复制/重试/停止/删除评测任务、导出下载 Insight & Case 数据、生成评测分析报告、解析 Agent 轨迹；并基于导出数据做下钻分析、错误归因、底线问题检测与 WildClaw Bench 专项分析；同时支持把开源评测集（opencompass/lm-evaluation-harness 等）自带的打分代码改写为 hy_unify_eval 的 Metric 并提交推送到个人分支。当用户提及"评测结果 / 评估报告 / 评测得分 / Insight / case 导出 / 下钻分析 / 错误归因 / 底线问题 / 复读乱码检测 / 深度分析 / 横向对比 / 多模型对比 / 模型优劣 / 逐case分析 / WildClaw / agent 轨迹 / 触发事件 / 伴生评估 / 触发事件复制 / 基于..创建触发事件 / 修改模型参数 / 配置伴生评估 / 复制训练任务 / 批量复制 / 复制到新任务 / 复制伴生配置 / 发布集合版本 / 上线集合版本 / 下线集合版本 / 新建评估版本 / 创建 Arena / 新增开源 metric / 把某评测集改写进 evals / 接入 MMLU HumanEval GSM8K 等开源评测集 / hy_unify_eval 加 metric / langfuse / trace 查询 / 查看调用链 / 拉取 trace 数据"等关键词时，应使用本 skill。边界排除项见 §0。
version: 3.0.0
author: taiji-team
---

# Evaluation Skill（模型评测）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：模型评测全链路。评测结果/评估报告/得分/进展、评测任务 CRUD（复制/重试/停止/删除/链接）、Arena（评测版本）、Insight 管理与 Case 对比、Insight/Case 数据导出下载、Agent 轨迹解析；基于导出数据做下钻分析、错误归因、底线问题检测与 WildClaw 专项分析；伴生评估配置（训练任务自动触发评估）；评测数据管理（Dataset/Exercise/Collection CRUD + 版本 + 发布）；平台深度分析闭环；开源评测集打分代码改写为 hy_unify_eval Metric（open-metric-rewrite）。

**⚠️ 排除**：
- 「质检 / 质检 N / inspection / 底线质检 / 内容质检 / 不合格数据 / 不合格样本」→ 属**后训练数据质检**，走 `posttrain-data-skill`。
- 本 skill 的「底线问题检测」是对**评测结果**做复读/乱码规则检测，与数据质检无关。
- ⛔ **严禁**用 `get_taiji_eval_task_detail` 等评测工具查质检（inspection_id 与评测 task_id 是不同 ID 空间，必然查不到）。

**跨模块边界**：
- 训练任务详情/启停/日志 → `task-skill`
- 应用组/资源使用/卡时配额 → `resource-mgmt-skill`
- 模型搜索/详情/发布 → `model-manage-skill`
- 训练任务 validloss / lm loss 曲线 → `swanlab-skill`（区别于「配置伴生评估」）
- 存储集群 / ceph 地域 / 冷热分析 → `storage-mgmt-skill`

> 缺 `wsid` / 工作空间上下文时可直调 `list_user_workspaces`（通过 `scripts/tool_manual.py list_user_workspaces` 确认参数后调用），无需提示"去加载 workspace-skill"。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **流程文档**：`*_analysis.md`、`open_metric_rewrite_analysis.md`、`export_data_schema.md` 等，§3 命中时先完整阅读再执行；通用规则见本文件 §1。`evaluation_api.md` 仍是大文件，进入前用 §3 的 grep pattern 跳到目标段。
> - **误入场景**：用户转向"任务/资源/模型/工作空间"等非评测意图时，立即退出。
> - **写操作**：创建、更新、删除、停止、复制、绑定、发布等。

---

## 1. 子 skill conventions（模块执行与全局行为）

### 会话初始化与热更新

首次加载本 skill 时，先在**当前 skill 根目录**执行：

```bash
python3 ./hot_reload.py
```

用户明确要求“更新 skill / 拉最新版”时执行：

```bash
python3 ./hot_reload.py --force
```

- 本 skill 被包含在 `taiji-basic-platform-skills/` 整包中时，脚本会自动委托 basic 整包更新；不要单独覆盖当前子目录。
- 本 skill 被单独安装时，脚本只检查、下载并覆盖当前 skill；其版本与其他 skill、basic 整包互不影响。
- 若在本地开发未发布改动，先设置 `TAIJI_NO_HOT_RELOAD=1`，避免自动更新覆盖工作区文件。

### 1.1 凭证与敏感信息

1. `scripts/connect_mcp.py` 自动按以下优先级读取太极 PAT Token：环境变量 `TAIJI_PAT_TOKEN` → `~/.config/taiji/credentials.json`。
2. Token 不属于业务参数；不得在工具 JSON 中传递、猜测、回显，或写入文档/日志/代码/命令示例/最终回复。
3. 不得为验证 Token 调用无关工具或执行探活；直接执行用户真正需要的首个业务调用。
4. 首次业务调用返回 `NO_TOKEN_CONFIGURED`/401/403 或明确凭证失效错误时，才引导用户提供或更新 Token（指引用户打开 https://taiji.woa.com/#/project-list → 工作台 → 右上角企业微信名 → 查看 **ApiToken**）。
5. 收到 Token 后保存：`python3 scripts/connect_mcp.py save-token '<token>'`，保存后只重试原请求一次；仍失败则如实报告并停止。网络错误/5xx/超时/响应解析错误不得归因于 Token。

### 1.2 本模块调用路径

```text
用户 prompt
   ↓
⓪ 先看§0 边界说明——确认请求属于本模块；不属于则立即退出切换
   ↓ 属于本模块
① 查§3快速路由表——命中【流程文档】→ 先完整读该文档再规划；或者命中【工具链路】 → tool_manual.py 批量获取参数细节
   ↓ 未命中
② 语义分析：工具名/参数已确定 → 直接调用；不确定 → scripts/tool_manual.py <tool1> <tool2> ...
   （仅对确定要调用的工具名批量获取，严禁逐个查询）
   ↓
③ ✅ 唯一调用通道：scripts/connect_mcp.py call <tool_name> '<json>'
```

1. **references 分两类，取用方式不同**：
   - **流程文档**（`*_flow.md`/`*_analysis.md`）——§3 命中或语义判定需要时**必须先完整阅读该文档再执行**，不得仅凭 §3 一行字直接调工具；
   - **工具手册**（`*_api.md`，由 `tool_manual.py` 按需提取）——只提供单工具参数规则。
2. 工具名或参数有任何不确定时，先基于 §2 用途列和 query 语义确定目标工具名，然后运行 `tool_manual.py <tool1> <tool2>`——仅对确定要调用的工具名批量获取参数，如果不能精确确认，可以适当多传，但严禁多轮逐个查询。
3. 结果为空、工具返回错误、权限不足、网络失败或超时时，及时停止、如实说明返回信息；不得编造结果、自行暴力搜索、URL、资源状态或成功结论。当工具不支持用户要求的筛选维度或操作时，及时告知用户能力边界，不得通过写脚本、分页遍历客户端过滤等方式绕过。用准确的名称/ID查不到时，直接告知用户找不到，不得换模糊关键词、换工具反复尝试。
4. 页面 URL 只能原样使用工具返回字段；**严禁根据 ID、历史格式或推测拼接 URL**。
5. **MCP 输出截断规则**：当输出过长被截断（`Output too large`）时，**严禁再次调用 MCP 工具**（如缩小 page_size 重查）。正确的处理方式是：用 Read 工具直接读取提示中指定的缓存文件路径，或使用 `grep`/`head`/`tail` 等轻量命令快速定位内容。MCP 调用昂贵（耗时 + Token），`Read` 和 `grep` 几乎零成本。
6. `connect_mcp.py` 是唯一调用通道：严禁改用 `mcporter`、裸 `curl`、`use_mcp_tool` 或自拼 HTTP；报错时仍用 `connect_mcp.py` 修正重试，不切换调用方式。

### 1.3 JSON 输出与结果处理

1. 所有非交互命令默认向 stdout 输出且仅输出一个合法 JSON 值；Agent 直接使用 `json.loads(stdout)` 解析结果。
2. 列表结果保持接口返回的原始顺序；Markdown 表格结束后必须保留一个空行再输出后续内容。
3. 同参数不重调：同一工具同一组参数不得重复调用；返回体过大被落盘时用 Read/grep 处理已落盘文件，不要重新发起同一调用；分页查询按用户指定页/页数调用一次即止。

### 1.4 Agent 行为与跨模块边界

1. 业务参数齐备时，Agent 直接执行工具调用并汇报真实结果；不得把命令交给用户自行运行。
2. 仅在缺少工具必填业务字段、当前模块无法确认用户意图、调用失败，或写操作需确认时追问。
3. 不得猜测、补全或批量枚举用户未提供的资源标识（工作空间、任务、应用组、模型、服务、路径、URL）。
4. **严守用户输入的 id/name**：当用户提供的 id/name 无法满足预期行为时（如搜索无结果），禁止自行搜索替代项重试；直接告诉用户实际情况。
5. **误入立即退出**：进入本模块后，若发现请求核心对象不属于本模块范围，立即切换到正确的子 skill，严禁在本模块内试探性调用。
6. 需要跨模块直调工具时，先用 `scripts/tool_manual.py <工具名>` 确认该工具在本模块 references 中已声明为 helper，再通过 `connect_mcp.py` 调用。

### 1.5 写操作与安全

1. 创建、更新、删除等写操作，必须遵循对应工具手册的前置查询、影响说明和确认要求。
2. 用户未明确目标、范围或影响对象时，不得擅自执行写操作。
3. 写操作成功前不得声称已创建、已更新、已停止或已完成。
4. 工具专属的覆盖语义、read-modify-write、状态限制、二次确认和成功后停止规则保留在对应工具手册，不得被本节覆盖或弱化。

---

## 2. MCP 工具清单表

> 写操作标记：✍️。完整参数/返回/SOP 用 `scripts/tool_manual.py <工具名>` 按需获取（一次可传多个工具名）。

### A. 评测任务管理

| 工具 | 用途 | 写操作 |
|---|---|---|
| `get_taiji_eval_task_detail` | 查任务详情 / 评估报告 / 评估结果 / insight 得分 | |
| `get_taiji_eval_exercise_results` | 查评估指标 / 评估进展 / 各评测集得分 / 完成/预测数量 | |
| `get_taiji_eval_exercise_summary` | 查任务 exercise 级别汇总（含各评测集得分摘要，支持分页） | |
| `list_taiji_eval_tasks` | 分页查询评测任务列表（arena_id/keyword/creator/status/hy_job_id 等多条件筛选） | |
| `clone_taiji_eval_task` | 复制 / 拷贝 / 克隆评测任务 | ✍️ |
| `retry_taiji_eval_task` | 重试 / 重跑评测任务（失败重试） | ✍️ |
| `stop_taiji_eval_task` | 停止 / 终止 / 取消评测任务 | ✍️ |
| `delete_taiji_eval_task` | 删除 / 移除评测任务（**不可逆，需二次确认**） | ✍️ |
| `get_taiji_eval_task_link` | 获取评测任务链接 / 页面链接 / 分享评测 | |
| `get_taiji_eval_task_agent_detail` | 查任务维度 agent 评测详情 / agent 状态分析 | |
| `get_taiji_eval_insight_agent_detail` | 查 Insight/视图/报告维度 agent 评测详情 | |

### B. Arena / 评测版本

| 工具 | 用途 | 写操作 |
|---|---|---|
| `list_taiji_eval_arenas` | 查评测版本 / 评估版本列表 | |
| `create_taiji_eval_arena` | 新建 / 创建评估版本（Arena） | ✍️ |

### C. Insight 管理

| 工具 | 用途 | 写操作 |
|---|---|---|
| `create_taiji_eval_insight` | 创建 / 新建 Insight | ✍️ |
| `update_taiji_eval_insight` | 修改 / 重命名 / 设置管理员（`task_ids` **全量替换**） | ✍️ |
| `delete_taiji_eval_insight` | 删除 Insight（**不可逆，需二次确认**） | ✍️ |
| `add_tasks_to_taiji_eval_insight` | 向 Insight 增量添加任务 | ✍️ |
| `remove_tasks_from_taiji_eval_insight` | 从 Insight 移除任务 | ✍️ |
| `list_taiji_eval_insights` | 查询 Insight 列表（分页 **1-based**，`page_index=0` 报错） | |
| `get_taiji_eval_insight_detail` | 查 Insight 详情 / 指标权重 / 各任务各维度得分（`weightNodes`） | |
| `list_taiji_eval_insight_cases` | 多任务 Case 级明细对比（按 question_id 分组） | |

### D. 导出（Insight 导出 / Case 导出）

| 工具 | 用途 | 写操作 |
|---|---|---|
| `submit_taiji_eval_insight_export` | 创建 Insight 导出任务 / 下载 insight 数据 | ✍️ |
| `list_taiji_eval_insight_exports` | 查看 Insight 导出历史 | |
| `get_taiji_eval_insight_export_status` | 查看导出任务状态 / 下载进度 | |
| `submit_taiji_eval_case_export` | 创建 Task Case 导出任务 / 下载 case / 评测明细 | ✍️ |
| `list_taiji_eval_case_exports` | 查看 Task Case 导出历史 | |

### E. 结果分析 / 深度分析

| 工具 | 用途 | 写操作 |
|---|---|---|
| `get_taiji_eval_task_confidence` | 查置信区间 / 显著性 / Bootstrap 重采样 | |
| `get_taiji_eval_bench_confidence` | 查 Agent 评测 Bench 置信度 | |
| `get_taiji_eval_drill_dimensions` | 查下钻维度列表（topic / difficulty 等） | |
| `get_taiji_eval_drill_metrics` | 按维度下钻查聚合指标 | |
| `get_taiji_eval_performance_trend` | 查性能趋势（散点 + 折线原始数据） | |
| `get_taiji_eval_task_progress` | 查整体进度（抓取 + 评估概览） | |
| `trigger_taiji_eval_deep_analysis` | 创建深度分析记录（后续需 Agent 完成分析并回写） | ✍️ |
| `list_taiji_eval_analysis_results` / `get_taiji_eval_analysis_detail` | 查询深度分析列表 / 详情 | |
| `get_taiji_eval_case_detail` / `get_taiji_eval_task_scores` / `get_taiji_eval_metric_scores` | 深度分析数据采集：题目/任务/指标粒度 | |
| `upload_taiji_eval_analysis_result` / `upload_taiji_eval_analysis_file` | 上传深度分析报告并回写 summary / report_url | ✍️ |

### F. 伴生评估配置

| 工具 | 用途 | 写操作 |
|---|---|---|
| `list_taiji_eval_companion_tasks` | 按名称查已绑定触发事件的 ID（name→ID 解析，不进入配置流程） | |
| `copy_taiji_eval_companion_trigger` | **同任务内**复制触发事件 → 产出 `trigger_id`（⛔ 仅同 job_group_id，不支持跨任务） | ✍️ |
| `upsert_taiji_eval_companion_config` | 幂等创建或更新评估配置 → 产出 `resource_id` | ✍️ |
| `bind_taiji_eval_companion_config` | 绑定触发事件与评估配置（串联①②的 id，不执行则配置不生效） | ✍️ |
| `list_taiji_eval_companion_configs` | 查训练任务下已有伴生评估配置列表及绑定的 trigger（回显 / 排查） | |
| `copy_taiji_eval_companion_resource` | **跨任务**复制伴生配置（触发器+resource+绑定一步完成），支持 `trigger_ids` 控制粒度 | ✍️ |

### G. 数据集 / 评测集 / 集合（Dataset / Exercise / Collection）

| 域 | 工具 | 用途 | 写操作 |
|---|---|---|---|
| Dataset | `upload_taiji_eval_dataset_file` | 上传评测数据集文件（JSONL/CSV → Ceph 路径） | ✍️ |
| Dataset | `create_taiji_eval_dataset` | 创建 / 注册评测数据集 | ✍️ |
| Dataset | `list_taiji_eval_datasets` / `get_taiji_eval_dataset_detail` | 查询数据集列表 / 详情 | |
| Dataset | `update_taiji_eval_dataset` / `delete_taiji_eval_dataset` | 修改 / 删除数据集 | ✍️ |
| Dataset | `list_taiji_eval_dataset_versions` / `update_taiji_eval_dataset_version` / `delete_taiji_eval_dataset_version` / `clone_taiji_eval_dataset_version` | 数据集版本管理（查询/修改/删除/复制） | ✍️ |
| Dataset | `download_taiji_eval_dataset_version_file` / `create_taiji_eval_dataset_version` | 版本导出-修改-重传：下载已有版本文件 / 新建版本 | ✍️ |
| Exercise | `create_taiji_eval_exercise` / `create_taiji_eval_exercise_version` | 创建评测集 / 评测集版本 | ✍️ |
| Exercise | `list_taiji_eval_exercises` / `get_taiji_eval_exercise_version_detail` | 查询评测集列表 / 版本详情 | |
| Exercise | `update_taiji_eval_exercise` / `delete_taiji_eval_exercise` | 修改 / 删除评测集 | ✍️ |
| Exercise | `list_taiji_eval_exercise_versions` / `update_taiji_eval_exercise_version` / `delete_taiji_eval_exercise_version` / `clone_taiji_eval_exercise_version` | 评测集版本管理（查询全部有权限版本/按管理员过滤/修改/删除/复制） | ✍️ |
| Exercise | `clone_taiji_eval_exercise_validation` | 创建验证任务（基于已有任务验证评测集版本） | ✍️ |
| Collection | `create_taiji_eval_collection` / `create_taiji_eval_collection_version` | 创建评测集合 / 集合版本 | ✍️ |
| Collection | `list_taiji_eval_collections` / `list_taiji_eval_collection_versions` | 查询集合列表 / 版本列表 | |
| Collection | `get_taiji_eval_collection_version_detail` | 查询集合版本详情 | |
| Collection | `update_taiji_eval_collection` / `delete_taiji_eval_collection` | 修改 / 删除集合 | ✍️ |
| Collection | `update_taiji_eval_collection_version` / `update_taiji_eval_collection_weight` / `delete_taiji_eval_collection_version` | 集合版本管理 / 调整权重 / 删除版本 | ✍️ |
| Collection | `release_taiji_eval_collection_version` | 发布（上线）/ 下线集合版本 | ✍️ |

### 跨模块速查工具

直达其他模块，通过 `scripts/tool_manual.py <工具名>` 确认参数后调用。

| 跨模块工具 | 用途 |
|---|---|
| `list_user_workspaces` | 直调工作空间查询，缺 `wsid` 时列出可访问空间供用户选定 |

---

## 3. 快速路由表（多工具流程编排）

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。
>
> 仅列多工具串联流程。单工具场景已在 §2 用途列覆盖。命中即按此执行，读对应流程文档。

| 用户意图 | 编排 / 流程文档 |
|---|---|
| 🔥 **评测任务概要查询**（问"状态/结果/进度/得分"） | 标准链路：`get_taiji_eval_task_detail` → `get_taiji_eval_exercise_results` → 可选 `get_taiji_eval_task_progress`。**一次 `tool_manual.py` 批量取全部工具**，禁止逐个查。 |
| 🔥 **Bench 置信度查询**（问"置信度/置信区间/Bootstrap/显著性"） | `get_taiji_eval_bench_confidence(task_id, collection_version_id, exercise_version_id)`——`collection_version_id` 不传会报错（`50001`）。**不知道时**先调 `get_taiji_eval_task_detail(task_id)` 获取；已知则直接调。 |
| 🔥 **Insight 详情+Agent**（问"分析结果/能力分布/多维度"） | 标准链路：`get_taiji_eval_insight_detail` → `get_taiji_eval_insight_agent_detail`。**一次批量取**。 |
| 下钻分析（Topic / Bench / 维度） | 先导出 Insight 数据 → `drilldown_analysis.md` |
| 错误归因（失败样本根因 / LLM 诊断） | 先导出 Insight 数据 → `error_diagnosis_analysis.md` |
| 平台深度分析 / AI 分析 / 多模型横向对比 / 逐case分析 | `deep_analysis_api.md` 闭环：`trigger_taiji_eval_deep_analysis` → `get_taiji_eval_task_scores`/`get_taiji_eval_metric_scores`/`get_taiji_eval_case_detail` → `upload_taiji_eval_analysis_result`/`upload_taiji_eval_analysis_file`；多模型对比走专节（Step A~G） |
| 底线问题检测（复读/乱码/think 标签/tool call 异常） | `bottom_line_analysis.md` |
| WildClaw Bench 专项分析 | `wildclaw_analysis.md` |
| Insight Case 多任务明细对比 | `insight_management_api.md` |
| **伴生评估配置** / 触发事件复制 / 绑定 / 查已有配置 | `companion_eval_config_api.md` |
| **评测任务管理**：详情/列表/Arena/结果/复制/重试/停止/删除/链接/agent | `evaluation_api.md` |
| **数据导出**：Insight/Case 导出、状态/历史 | `evaluation_api.md` |
| **评测结果分析**：置信区间 / Bench 置信度 / 下钻 / 趋势 / 进度 | `evaluation_result_analysis_api.md`（置信区间/趋势/进度） + `drill_api.md`（下钻维度/指标） |
| **从零搭建一整套评测**（Dataset→Exercise→Collection） | `collection_management_api.md`（§Collection 完整调用流程，含输入清单/名字降噪） |
| Dataset 上传/CRUD/版本管理 | `dataset_management_api.md` |
| Exercise 创建/CRUD/版本/验证任务/裁判模型调优 | `exercise_management_api.md` |
| Collection CRUD + 列表/版本 + 权重 + 发布/下线 | `collection_management_api.md` |
| Insight 实体 CRUD + 列表/详情/Case 明细对比 | `insight_management_api.md` |
| **新增开源 Metric**（改写 hy_unify_eval Metric + 注册 + Git 推送） | `open_metric_rewrite_analysis.md`（OpenAPI 对接复用 `exercise_management_api.md`） |
| 导出数据字段 / payload / 指标 / agent 轨迹格式 | `export_data_schema.md`（横切参考） |
| 按 task_id 下载 Langfuse trace 数据 | `langfuse_trace_query_analysis.md` |
| 批量下载全量有权限 Exercise Version | 见 §4「批量下载全量有权限 Exercise Version」 |

> ⚠️ **路由要点**：
> - 「Insight 列表/详情/CRUD」只读 `insight_management_api.md`，**不要**再读 `evaluation_api.md`。
> - 「评测集合列表/版本列表」只读 `collection_management_api.md`，**不要**再回读 `evaluation_api.md`。
> - 「评测结果置信区间/下钻/趋势/进度」只读 `evaluation_result_analysis_api.md` 和 `drill_api.md`。
> - `evaluation_api.md` 仍是 1500+ 行大文件，进入前用上表 grep pattern 跳到目标段。

---

## 4. 模块注意事项

### 4.1 从查询结果拼参数时必须读文档

当把 `get_xxx_detail` 返回值用于 `create_xxx` / `update_xxx` 时，必须：
1. **先读目标工具的完整参数表**，确认每个字段的必填/选填/按模式要求，**不要凭返回值试探**。
2. **不要盲目复制所有字段**：`get_detail` 返回的字段可能不被 `create` 接受。只传参数表中出现的字段。
3. **`parameter_configuration` 决定模式**，模式决定必填字段。先确认模式再一次性传齐，**不要用最小参数集逐渐试探**。
4. **`custom_parameters` 里的字符串化 JSON**（如 `"eval_model_config": "{\"type\": \"hy_openai\"}"`）不要直接拼进 shell 命令会转义失败。先解析内层 JSON 再作为对象放入。
5. **prompt 是什么就传什么**，不要看到 `"-"`、空字符串等就觉得是可选跳过。

### 4.2 伴生评估配置（companion_eval_config_api.md）

- **何时进入**：用户意图含「触发事件 / 伴生评估 / 自动触发评估 / 训练过程中评估 / 给训练任务绑评估 / 按 step 触发评估 / 配置伴生评估 / 绑定自动评估 / 已有伴生评估配置」；或"基于XXX创建/复制/新建触发事件"、"把XXX复制为YYY"、"修改/更新/配置模型参数"、"把训练任务A的伴生配置/触发事件复制到训练任务B"、"复制训练任务时带上伴生配置"等。
- **进入后必做**：
  1. `job_group_id`（训练任务 ID，**必须为数值型 jobGroupId**）和 `instance_id`（训练实例 hyJobId）缺一必追问；`ws_id` 默认 10103，**无需向用户确认**。
  2. **先判断同任务还是跨任务**（源/目标 job_group_id 是否相同）：
     - **同任务内复制**：走 ① `copy_taiji_eval_companion_trigger` → ② `upsert_taiji_eval_companion_config` → ③ `bind_taiji_eval_companion_config` 顺序；③的 `job_group_resource_id`（取自②返回的 `resource_id`）和 `trigger_ids`（取自①返回的 `trigger_id`）**必须取自①②真实返回，严禁猜测或复用历史 id**。
     - **跨任务复制**（源≠目标）：⛔ 一律走 `copy_taiji_eval_companion_resource`，一步完成。**禁止**用 `copy_taiji_eval_companion_trigger` 传 `job_group_id` 试图跨任务——该参数会被后端静默忽略。`trigger_ids` 可选控制粒度（不传=全量，传了=指定）。
  3. `copy_taiji_eval_companion_trigger` 需要 `source_trigger_id`（整数 ID）。用户只给名称时必须先用 `list_taiji_eval_companion_tasks` 查出 ID。严禁用 `list_taiji_eval_companion_trigger_templates`——那是模板库。
  4. `upsert_taiji_eval_companion_config` 的 `model_config` 所有字段有默认值，用户无需提供。只需追问 `root_ceph_path`（首次创建时）和 `visual_structure`/`vit_input_resolution`（多模态时）。
- **严禁**：把「配置伴生评估」与 Infra 2.0「独立评测任务」混用（后者走主链路）；与「validloss / lm loss 曲线查询」混用（走 `swanlab-skill`）；进入链路前调用 task-skill 工具验证 task_id；跨任务复制时用 `copy_taiji_eval_companion_trigger` 传 `job_group_id`。
- **失败处理**：①成功②失败 → 告知 `trigger_id` 已创建、仅 upsert config 失败；②成功③失败 → 告知 `trigger_id`/`resource_id` 均已就绪、仅绑定失败，可单独重试 bind。

### 4.3 评测主链路（evaluation_api.md）

- **何时进入**：用户问"评测结果 / 评测得分 / 评测任务 / Insight / 评测集合 / case 导出 / Agent 轨迹 / 评测分析报告"等任一意图。
- **进入后必做**：
  1. 涉及"导出"时先消歧：是否为评测 Insight/Case 语义（区别于通用 ceph 数据、质检不合格导出）。
  2. 调 `list_taiji_eval_insights` / `list_taiji_eval_tasks` 等列表工具分页是 **1-based**（`page_index=0` 报错）。
  3. **Provider/Consumer 任务**：`SERVICE_REUSE_PROVIDER` 本身不执行评测，查询详情关注返回的 `consumer_tasks`；查询评测集进展关注 `provider_info` 和汇总后的 Consumer 结果。
  4. **空间 ID（wsid）**：`list_taiji_eval_insights` / `get_taiji_eval_insight_detail` / `create_taiji_eval_insight` / `update_taiji_eval_insight` 等需指定 `wsid`（纯数字字符串，默认 `10103`）；用户未明确空间时应主动确认。
  5. 下钻 / 错误归因要在 Insight 数据基础上做时，本文档只负责"产出 Insight 与导出数据"，分析转给对应分析 reference。
- **Insight 管理注意**：
  1. `update_taiji_eval_insight` 的 `task_ids` 是**全量替换**；追加用 `add_tasks_to_taiji_eval_insight`，移除用 `remove_tasks_from_taiji_eval_insight`。
  2. `delete_taiji_eval_insight` 不可逆，调用前必须二次确认。
  3. 创建 Insight 后默认无任务，需 `add_tasks_to_taiji_eval_insight` 添加。
  4. `remove_tasks_from_taiji_eval_insight` 若移除的是基线任务（`baseLineTaskId`），基线配置自动清空，需提示重设。
  5. `update/delete/add_tasks/remove_tasks` 仅 Insight 负责人可执行；无权限时透传服务端错误。
- **严禁**：调 `clone_taiji_eval_task` / `submit_taiji_eval_case_export` / `submit_taiji_eval_insight_export` 不校验 task 归属；猜测 evaluation_id / version_id。
- **导出轨迹参数**：Insight/Case 导出支持 `include_trajectory`（默认 true）和 `include_raw_trajectory`（默认 false）；`include_raw_trajectory=true` 会显著增加导出体积和耗时，只有用户明确需要原始执行轨迹时才开启。
- **失败处理**：导出任务 FAILED → 透传错误原因 + task_id，不主动重试。

### 4.4 新增开源 Metric（open_metric_rewrite_analysis.md）

- **何时进入**：用户意图含"新增开源 metric / 新增评测指标 / 把 xx 评测集打分代码改写/抓取/拉到 evals / hy_unify_eval / 接入 xx（如 MMLU/HumanEval/GSM8K/HellaSwag）到太极评测 / 在 hy_unify_eval 里加一个 metric"；或给出开源仓库（opencompass/lm-evaluation-harness 等）的 `evaluate.py`/`metric.py`/打分脚本要求接入内部评测。
- **能力概要**：把开源评测集打分代码，按 `hy_unify_eval`（**本地代码库路径进入流程前向用户询问，不写死**；目标远程仓库 `git@git.woa.com:taiji/hy/hy_unify_eval.git`）的 `BaseEval`/`BaseData` 规范改写成可被反射加载的 Metric 类，覆盖：选型 → 解读开源打分逻辑（区分 lm-eval-harness/opencompass/单数据集 grader 三种来源，含 loglikelihood→generate_until 适配）→ 改写 Eval 类 → 配套 Data 类 → 注册（两个 `__init__.py`）→ 复用工具 → 对齐校验 → OpenAPI 对接（复用 `exercise_management_api.md`）→ **提交并推送到个人分支**。
- **⚠️ Git 提交安全铁律（最高优先级，不可跳过）**：
  1. 代码改写落库后，进入 Git 提交前**必须先检查当前分支**（`git branch --show-current`）。
  2. **⛔ 严禁在 `master` 分支直接提交或推送**。若在 master，必须先明确告知用户，然后自动创建个人分支（命名 `<git用户名>/add-<metric名>-metric`）并切换。
  3. 推送目标固定为 `git@git.woa.com:taiji/hy/hy_unify_eval.git` 的**当前个人分支**，**⛔ 严禁 `git push origin master`**。
  4. 不代替用户发起 Merge Request，做到推送个人分支为止。
- **严禁**：把"新增开源 Metric"（代码改写 + hy_unify_eval 落库）与"评测主链路"（调 OpenAPI 查/管理评测任务）混用——前者是代码开发，后者是 API 调用；两者仅在 Step 9 OpenAPI 对接处交汇。

### 4.5 批量下载全量有权限 Exercise Version

> **触发条件**：用户意图包含「所有/全部/我有权限的/我管理的/我的」+「评测集版本/exercise version/version 列表」+「下载/导出/拉取/获取清单」，且**未指定具体评测集名称或 ID**。

**权限语义消歧**：

| 用户说法 | 走哪条路径 | 调用方式 |
|---|---|---|
| "有权限 / 我的 / 我管理的 / 全部" | ✅ 默认走此分支（admin 过滤） | `{"admin": "<当前用户>"}` |
| 明确说"能看到 / 可见 / 有访问权限" | ⚠️ 走全量路径（范围更广） | `{}`（不传参） |
| 指定了具体评测集名称/ID | ❌ 不走此分支，走原有路线 | `{"exercise_id": <id>}` |
| 指定具体 version ID | ❌ 走 `get_taiji_eval_exercise_version_detail` | — |

**执行链路**：Step 1 前置批量列出有管理权限的版本（`list_taiji_eval_exercise_versions {"admin": "<当前用户>", "page_size": 50}`，分页翻页直到 `has_more=false`）→ 向用户展示摘要（总数/状态分布/评测集分布）→ Step 2 确认后续方式（A 导出元数据清单 / B 逐个查详情 / C 关联下载评测数据，需先确认已有关联评测任务，否则告知"该版本尚未创建评测任务，无 case 可导出"）→ Step 3 按选择执行。

**分页处理规范**：必须分页（total 可能数十到上千）；翻页终止条件 `has_more=false`；每拉一页向用户汇报进度；total=0 时明确告知"当前用户没有任何管理权限的评测集版本"，不要静默返回空。

**与其他模块关系**：不替代 `list_taiji_eval_exercises`（本分支是 version 维度）；可衔接 `submit_taiji_eval_case_export`（仅限已有关联评测任务的 version）；独立于 Insight 导出（本分支是 version 元数据级别）。

### 4.6 下钻分析（drilldown_analysis.md）

- **何时进入**："下钻 / 评测对比 / bench 对比 / topic 下钻 / 维度下钻 / 评估结果分析"。
- **进入后必做**：必须先有 Insight 导出数据（来源 `evaluation_api.md` 导出工具 + `export_data_schema.md` 字段）；用户没说维度时先列出可用 bench/topic 让其挑；结论必须基于真实字段（accuracy / topic 分布）。
- **严禁**：把"错误归因"与"下钻分析"混为一谈——前者走 `error_diagnosis_analysis.md`，本模块只做指标维度对比。
- **失败处理**：缺数据时告知"先导出 Insight 数据"并指引到 `evaluation_api.md`，不自行换接口。

### 4.7 错误归因（error_diagnosis_analysis.md + call_llm_distill.py）

- **何时进入**："错误归因 / 错误分析 / 失败 case 归因 / topic 错误分析 / 根因分析"。
- **进入后必做**：依赖 Insight 中 `is_correct=false` 样本明细，先确认数据已导出；用户没说 bench/topic 时先列候选；LLM 自动归因输出必须含「错误样本数 / 主要错误类型分布 / 典型 case 引用 / 改进建议」四项，缺一不可。LLM 诊断与聚类摘要通过 `scripts/call_llm_distill.py` 执行。
- **严禁**：把"全量 case 导出"当成归因；本模块只做归因聚合 + 根因分析。
- **失败处理**：样本不足（< 5 条）直接告知"样本太少，归因结果不可信"。

### 4.8 平台深度分析（deep_analysis_api.md）

- **何时进入**："深度分析 / AI分析评测结果 / 自动分析 / 生成分析报告 / 触发分析 / 查看分析结果 / 横向对比 / 多模型对比 / 模型优劣 / 逐case分析"。
- **进入后必做**：新分析必须按闭环流程执行：创建记录 → 获取 Insight/Task 详情 → 拉任务分数/题目指标/case 明细 → 生成 Markdown 报告 → 上传报告 → 回写 summary/result；**严禁只调用 `trigger_taiji_eval_deep_analysis` 就结束**。
- **查询类请求**：查看历史/详情时调用 `list_taiji_eval_analysis_results` / `get_taiji_eval_analysis_detail`。

### 4.9 底线问题检测（bottom_line_analysis.md + detect_floor_anomaly.py）

- **何时进入**："底线分析 / 底线问题 / 下限检测 / 复读 / 乱码 / think 标签异常 / tool call 异常 / markdown 错误 / 模型输出异常"。
- **进入后必做**：依赖 Insight/Case 导出数据；检测前必须与用户确认参数（见文档 §1.2）；检测逻辑必须完全对齐脚本，默认用 `scripts/detect_floor_anomaly.py`，需要 bench 级并发时用 `scripts/detect_floor_anomaly_concurrent.py --workers N`；不得自增/删检测规则；结论必须基于脚本真实统计结果。
- **严禁**：把"底线检测"和"错误归因"混为一谈——底线检测是纯规则检测（不依赖 LLM），错误归因走 `error_diagnosis_analysis.md`。
- **失败处理**：缺数据时告知"先导出评测数据"并指引到 `evaluation_api.md`。

### 4.10 WildClaw 专项分析（wildclaw_analysis.md + analyze_wildclaw.py）

- **何时进入**："分析 wildclaw 结果 / 下载 wildclaw 数据并出报告 / wildclaw 专项 / wildclaw 失败模式"。
- **进入后必做**：完全依据 `wildclaw_analysis.md` 的类目/评分机制/已知失败模式分析，用 `scripts/analyze_wildclaw.py` 跑分与失败模式归因；**不要**走通用 `export_data_schema.md` 的大而全流程。
- **注意**：WildClaw 属 agent 类 benchmark，轨迹文件为 zstd 压缩（`pip install zstandard` 才能解析 trajectory）。

### 4.11 易混淆概念区分

| 易混淆对 | 区分规则 |
|---|---|
| Arena vs 评测集版本 | **Arena 即评测版本**；评测集版本（Exercise version）不是 Arena。查评测版本走 `list_taiji_eval_arenas`，查评测集版本走 `list_taiji_eval_exercise_versions` |
| Insight 导出 vs Case 导出 | `submit_taiji_eval_insight_export` 按 Insight 导出；`submit_taiji_eval_case_export` 按 task 全量导出。用户只说"导出 xxx 任务的数据/case"且未提 Insight → 优先 `submit_taiji_eval_case_export` |
| 伴生评估 vs 独立评测 | 伴生评估是给训练任务配置自动触发评估（companion_eval_config_api）；独立评测是 create/copy/retry evaluation 主链路（evaluation_api）。**严禁混用** |
| 评测任务查询层级 | 任务 → 评测集 → 题目，按信息层级取用：任务详情 `get_taiji_eval_task_detail` → 评测集结果 `get_taiji_eval_exercise_results` → 题目粒度 `list_taiji_eval_insight_cases` / `get_taiji_eval_case_detail` |
| 导出意图消歧 | 导出前先确认是否为评测 Insight/Case 语义（区别于通用 ceph 数据、质检不合格导出） |
| Hydemo 数据集 vs 评测数据集 | 本 skill 的 Dataset 是评测数据集（上传 JSONL/CSV → Ceph）；通用数据/预处理数据集走 data-processing-skill，不在本模块 |

### 4.12 写操作专项约束

- **调用预算**：同一工具对同一对象（同一 task_id / insight_id / exercise_id 等）默认只调用一次，禁止为补字段或换排版重复调用。用户明确要求对多个不同评测任务分别操作时（如"停掉 A、B、C 三个任务"），按每个对象各调一次对应的 stop/clone/delete 工具。
- **clone 失败不换服务名重建**：批量 `clone_taiji_eval_task` 任一步返回服务不存在或参数错误时，立即停止并报告失败原因；禁止改 `service_name`、查 `list_deploy_services` 后重新 clone。

---
