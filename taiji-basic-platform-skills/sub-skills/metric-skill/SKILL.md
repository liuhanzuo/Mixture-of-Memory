---
name: metric-skill
description: 太极平台训练指标查询子 skill —— 通过 MCP 协议查询训练任务的平台预定义训练指标（loss / grad_norm 等）、tf_events 指标（模版化训练）、SwanLab 自定义指标，并可生成指标趋势图。当用户提及"loss / grad_norm / 训练指标 / 指标数据 / 聚合 / 平均 / 最大 / 最近N步 / 画loss曲线 / 趋势图 / 可视化 / chart / plot / SwanLab / tf_events"等关键词时，应使用本 skill。
version: 2.0.0
author: taiji-team
---
# Metric Skill（训练指标）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：训练任务的指标查询与指标趋势图生成，覆盖三类指标来源：

| 指标来源 | 适用任务类型 | 对应工具 |
|----------|-------------|---------|
| 智研监控指标（loss / grad_norm 等） | 自定义训练（`basic_train_` 开头） | `list_hunyuan_train_available_metrics` / `query_hunyuan_train_metric_text` / `query_hunyuan_train_metric_chart` |
| tf_events 指标 | 模版化训练（`finetuning_` 开头） | `list_hunyuan_train_tf_events_metrics` / `query_hunyuan_train_tf_events_text` / `query_hunyuan_train_tf_events_chart` |
| SwanLab 自定义指标 | 自定义训练（`basic_train_` 开头） | `query_hunyuan_train_swanlab_metrics` |

**跨模块边界**：
- 训练任务搜索 / 详情 / 克隆 / 启停 → `task-skill`
- 实例列表 / Pod 列表 / 日志 / 在实例 Pod 上执行命令 → `instance-skill`

> ⚠️ **模型开发任务**不支持训练指标查询。如果用户对模型开发任务查指标，应告知不支持。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **误入场景**：用户转向"训练任务启停、存储集群、查杀规则"等非指标意图时，立即退出。
> - **本模块无写操作工具**。

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

### 智研监控指标（自定义训练 `basic_train_`）

| 工具 | 用途 | 写操作 |
|---|---|---|
| `list_hunyuan_train_available_metrics` | 查询指定训练任务支持的所有训练指标及其可用的聚合方式 | |
| `query_hunyuan_train_metric_text` | 查询训练指标数据，支持聚合模式（avg/max/min 等）和原始数据模式（latest_n） | |
| `query_hunyuan_train_metric_chart` | 生成训练指标趋势图，返回上传至 COS 的图片 HTTPS URL 和文字摘要 | |

### tf_events 指标（模版化训练 `finetuning_`）

| 工具 | 用途 | 写操作 |
|---|---|---|
| `list_hunyuan_train_tf_events_metrics` | 列出模版化训练任务的可用 tf_events 指标 | |
| `query_hunyuan_train_tf_events_text` | 查询 tf_events 指标值，支持 step 范围过滤和取最近 N 个 step | |
| `query_hunyuan_train_tf_events_chart` | 生成 tf_events 指标趋势图 | |

### SwanLab 指标（自定义训练 `basic_train_`）

| 工具 | 用途 | 写操作 |
|---|---|---|
| `query_hunyuan_train_swanlab_metrics` | 查询 SwanLab 训练指标数据（**仅支持任务最新实例**），支持采样、全量、step/时间范围过滤等多种模式 | |

### 跨模块速查工具

直达其他模块，通过 `scripts/tool_manual.py <工具名>` 确认参数后调用。

| 跨模块工具 | 用途 |
|---|---|
| `query_hunyuan_swanlab_run_columns` | 查询 SwanLab Run 列名/指标字段名，需先从 `query_hunyuan_train_swanlab_metrics` 获取凭据 |

---

## 3. 快速路由表（多工具流程编排）

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。
>
> 仅列多工具串联流程。单工具场景已在 §2 用途列覆盖。

> 🛑 **工具参数获取规则**：根据 task_id 前缀确定指标来源（`basic_train_`→智研/SwanLab，`finetuning_`→tf_events），**首次 tool_manual 即把该来源下所有可能用到的工具一次性全部查询**。严禁 chart 返回空后再逐次追 tool_manual(swanlab)→再追 tool_manual(run_columns)。正确做法：`tool_manual.py query_hunyuan_train_metric_chart query_hunyuan_train_metric_text query_hunyuan_train_swanlab_metrics query_hunyuan_swanlab_run_columns list_hunyuan_train_available_metrics`（一次全传）。

| 场景 | 流程 | 流转条件 |
|------|------|----------|
|🔥 **图表/曲线/可视化**（"画loss 图/趋势图/训练曲线"） | 🔴 必须一次`tool_manual.py` 全传：`list_hunyuan_train_available_metrics query_hunyuan_train_metric_chart query_hunyuan_train_swanlab_metrics query_hunyuan_swanlab_run_columns`。执行链：① `query_hunyuan_train_metric_chart` → ② 为空或用户要更多维度时调`query_hunyuan_train_swanlab_metrics` → ③ 需查列名时`query_hunyuan_swanlab_run_columns`。**禁止 chart 失败后才补tool_manual**。（⚠️ 本条为§1.2「结果为空即停止」的例外：chart 返回空时允许继续调SwanLab） |首次调用就已确定全链路工具 |
| 平台预定义指标"平均/最大/统计/聚合"类问题 | ① `list_hunyuan_train_available_metrics`（确认可用聚合名）→② `query_hunyuan_train_metric_text`（用聚合名查询） | 已知 task_id + 指标来源时，聚合类必须先 list 再 text |
| "训练实例的 loss/吞吐/step"等平台指标 | ① `instance-skill` 的 `query_hunyuan_train_instance_list`（取目标实例，最多一次）→② 回本 skill 按用户指定指标调 `query_hunyuan_train_metric_text` | 不查失败事件/日志/SwanLab；不对多个实例逐个查 |
| "最近一次/运行中/特定优先级或机型"等开放式任务发现 | ① `task-skill` 的 `query_hunyuan_train_task_list` **一次**定位候选 →② 回本 skill 只调用用户目标指标工具 | 一次列表无完全匹配时只选**一个**最接近候选继续，禁止对多个候选批量调指标 |

---

## 4. 模块注意事项

### 4.1 指标来源判断与最小调用链

1. **根据 task_id 前缀判断指标来源**：
   - `basic_train_` 开头 → 智研监控指标（`list_hunyuan_train_available_metrics`/`query_hunyuan_train_metric_text`/`query_hunyuan_train_metric_chart`）或 SwanLab 指标（`query_hunyuan_train_swanlab_metrics`）
   - `finetuning_` 开头 → tf_events 指标（`list_hunyuan_train_tf_events_metrics`/`query_hunyuan_train_tf_events_text`/`query_hunyuan_train_tf_events_chart`）
2. 必须传 `task_id`。用户已给明确 task_id 时**直接调用对应工具，不要先去搜索任务、任务详情或实例列表**。聚合方式（avg/max/min/latest_n）按用户语义推断；用户没说默认走原始数据模式取最近若干步。
3. **最小调用链**：已知 `task_id` 且用户已明确指标名/指标来源时，优先只调用目标查询工具；但平台预定义指标的「平均/最大/统计/聚合」类问题必须先 `list_hunyuan_train_available_metrics` 确认可用聚合名，再调用 `query_hunyuan_train_metric_text`。
4. **list 工具触发边界**：只有用户问"支持哪些指标/指标列表/哪个指标组"、指标名不确定，或需要确认聚合方式时才先调 list；图表/曲线、最近N步原始数据、SwanLab、tf_events 已知 metrics 不先 list。

### 4.2 参数与意图映射

1. **参数白名单**：指标工具参数统一通过 `scripts/tool_manual.py <工具名>` 获取（包括 `query_hunyuan_swanlab_run_columns`），**不要把 query 里的 `wsid` 透传给 metric/chart/tf_events/SwanLab 工具**；`wsid` 只用于上游任务检索（如查最近一次任务）所需的 task-list 工具。正确示例：chart 只传 `{"task_id":"...","metric":"loss,grad_norm"}`。
2. **文本 vs 图表判断**：仅当用户表述含「画/图/曲线/趋势图/可视化/chart/plot/graph」信号词时走 chart 工具；否则走 text 工具，返回后追问"需要画图吗？"。
3. **语义到参数的固定映射**：
   - 「最近 N 步的值/数据」→ `latest_n=N`；「最近 loss 值怎么样」未说明 N 时默认取最近 100 步。
   - 「现在/当前/到多少了」→ 只取最新点，`latest_n=1`。
   - 多个指标一起查（如 `loss 和 grad_norm`）→ 一次调用，`metric="loss,grad_norm"`，不要拆成两次 text 调用。
   - 「总体平均/平均值/总平均」→ 先 list，再用可用的总体平均聚合名；`aggregations` 的值传字符串，不传数组，不要先数组失败后再重试字符串。
   - 「实验效果对比」未指定指标时，先分别 list 两个任务确认可用指标，再选择最能代表训练效果的 loss/相近指标查询最近数据；不要额外加入无关指标或聚合查询，除非用户明说。
   - 「曲线/趋势图/异常波动」调用 chart 后直接基于 chart 摘要回答；不要为了分析再补 text 查询。
   - 「输出速率/吞吐/throughput 在哪个指标组」属于平台预定义指标归属问题，只调用 `list_hunyuan_train_available_metrics`，优先从返回的吞吐/throughput 相关指标中匹配，不硬编码单一指标名。
   - `tf_events` 的 `metrics` 必须传 `array[string]`，即 `{"metrics":["train/train_loss"]}` 这类数组形态；即使 query 附带 wsid，也不要传 `wsid`。
   - SwanLab 明确 `keys` 后默认传 `tail=100`；用户给"采样 N 个点"才传 `sample=N`。SwanLab 指标查询前后都不要调用 `get_hunyuan_train_task_detail`/`list_hunyuan_train_available_metrics`，除非用户明确要求任务详情或平台指标。
   - 用户问"有哪些 SwanLab 指标/列名/keys/指标字段"且给了 `task_id` 时，先调用一次 `query_hunyuan_train_swanlab_metrics` 获取后端解析出的 workspace/project/run 线索；如还必须查列名，再只调用一次 `query_hunyuan_swanlab_run_columns`，禁止空 `keys` 重试、禁止对同一 run_columns 换 page/page_size/column_type 反复调用。
4. **目标调用完成即停止**：完成用户目标工具调用后直接基于返回回答；严禁再为了"验证/更全面/补充背景"追加 `get_hunyuan_train_task_detail`、第二个 chart、text、list、SwanLab、资源/GPU 利用率工具或更多 task_list。
5. 输出包含曲线时使用工具返回的 `chart_url`（拼接为 `![](chart_url)`）直接展示，不要把原始 JSON 丢给用户。

### 4.3 开放式任务发现

先用 `task-skill` 的 `query_hunyuan_train_task_list` **一次**定位候选（按用户条件传必要过滤项，不额外翻页），再回到本 skill 只调用用户目标指标工具；不要重复 task_list，不要跨查无关训练类别，不要补调 task_detail、实例、Pod、exec、资源 GPU 利用率或 MFU 工具，除非用户明确要求这些维度。若一次列表内没有完全满足条件的候选，只选择**一个**最接近的候选继续目标指标查询，并在最终答案说明未找到完全匹配项；禁止对多个候选批量调用 SwanLab/metric 工具，也禁止继续无限搜索关键词、标签或更多页。目标指标调用完成后必须停止，不再追加平台指标或验证查询。

### 4.4 严禁与失败处理

- **严禁**对 `finetuning_` 任务使用智研监控指标工具（应使用 tf_events 系列）。
- **严禁**对 `basic_train_` 任务使用 tf_events 工具（应使用智研监控或 SwanLab）。
- **严禁**仅因后端存在名称匹配的聚合方式（如 `Latest_100_step_avg`）就自动走聚合模式，判断只看用户原始表述是否含「平均/最大/统计/聚合」。
- **严禁**为了"更全面"而在目标指标查询之外额外调用 list/text/chart/SwanLab/task_detail/instance_list；多查会降低工具精确率。
- **严禁**使用 Python（matplotlib/PIL/plotly 等）自行生成图片或写入本地文件——图片由后端生成并通过 `chart_url` 返回。
- **严禁**跨模块路由——用户问"任务/实例/评测/模型"时立即跳出本 skill。
- **失败处理**：指标名不在平台预定义列表 → 提示用户该指标可能在 SwanLab，建议改用 `query_hunyuan_train_swanlab_metrics`；401/403 提示 token 失效；网络/超时如实告知，**不重试、不切换工具**。
