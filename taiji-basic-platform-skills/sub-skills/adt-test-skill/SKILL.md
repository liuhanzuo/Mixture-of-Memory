---
name: adt-test-skill
description: 太极平台 ADT 模型基础测试子 skill —— 通过 MCP 协议创建/查询/停止 ADT 模型基础测试任务（乱码/重复/截断/接口四项检测），以及一键「工程链路评估」（创建 ADT 测试 + 复制基准评测任务 + 后台自动轮询 + 完成后企微通知）。当用户提及"ADT 测试 / 模型基础测试 / 基础测试 / 乱码检测 / 重复检测 / 截断检测 / 接口测试 / 工程链路评估 / 链路评估 / 一键测试 / 一键评估 / 复制评测任务 / 克隆评测任务 / 创建ADT / 查询ADT结果 / 停止ADT"等关键词时，应使用本 skill。
version: 1.0.0
author: taiji-team
---
# ADT Test Skill（ADT 模型基础测试）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：ADT 模型基础测试（创建 / 查询 / 停止）与工程链路评估（创建 ADT 测试 + 复制基准评测任务 + 双任务后台轮询/查询 + 企微通知）。ADT 基础测试对**已部署的模型服务组**执行四项基础能力检测：乱码检测（HTEXTGARBLE）、重复检测（HDUPLICATE）、截断检测（HTRUNCATE）、接口测试（HYAPITEST）。

**跨模块边界**：
- 评测任务 / 评测结果 / Insight / Case 导出（区别于 ADT 基础测试）→ `evaluation-skill`
- 模型搜索 / 详情 / 发布 → `model-manage-skill`
- 模型服务组的部署 / 扩缩容（ADT 测试的前提是服务组已部署）→ `service-deploy-skill`
- 训练任务相关 → `task-skill`

> ⚠️ **工程链路评估是 skill 层编排**：其中「复制基准评测任务」并非 ADT 后端能力，而是本 skill 在 Agent 层**组合调用** `evaluation-skill` 的 `clone_taiji_eval_task` 完成。`run_adt_test_pipeline` 本身只负责 ADT 测试 + 后台轮询 + 企微通知。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **ADT 鉴权特例**：ADT 后端使用内置的固定 STAFFTOKEN 认证，当前用户由 `X-Auth-Username` 自动获取。本 skill 4 个 ADT 工具**均无需用户在参数里传 token**；上面的 Token 协议只保证请求能通过鉴权。
> - **误入场景**：用户转向"评测任务、训练指标"等非 ADT 测试意图时，立即退出。
> - **写操作**：创建、停止、工程链路评估等。

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

| 工具 | 用途 | 写操作 |
|---|---|---|
| `create_adt_test_task` | 创建 ADT 模型基础测试任务（测试指定模型服务组的乱码/重复/截断/接口四项基础能力） | ✍️ |
| `get_adt_test_task_detail` | 查询 ADT 测试任务状态与结果（含 `is_terminal` 终态标识、各测试项通过/失败数、报告链接） | |
| `stop_adt_test_task` | 停止一个正在运行的 ADT 模型基础测试任务 | ✍️ |
| `run_adt_test_pipeline` | 工程链路评估（第 1 步）：创建 ADT 测试 + 服务端后台每 5 分钟自动轮询 + 进入终态/超时后企微通知发起人 | ✍️ |
| `clone_taiji_eval_task` *(evaluation-skill)* | 工程链路评估（第 2 步）：复制基准评测任务（源任务 `source_task_id` = 可配置项 `PIPELINE_EVAL_SOURCE_TASK_ID`，默认 `29645`）。**跨模块组合调用**，非本 skill 自有工具 | ✍️ |
| `get_taiji_eval_task_detail` *(evaluation-skill)* | 查询工程链路评估的评测任务状态/结果（评测 `task_id` 为**整数**）。**跨模块组合调用** | |

---

## 3. 快速路由表（多工具流程编排）

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。
>
> 仅列多工具串联流程。单工具场景已在 §2 用途列覆盖。

| 场景 | 流程 | 流转条件 |
|------|------|----------|
| 工程链路评估（一键测试/链路评估） | ① `run_adt_test_pipeline`（第 1 步：ADT 测试+后台轮询+企微通知，属写操作，调用前复述步骤）→② `clone_taiji_eval_task`（第 2 步：复制基准评测任务，参数见 [`references/pipeline_eval_flow.md`](./references/pipeline_eval_flow.md)） | ①成功后记下 ADT `task_id`（字符串）；②成功记下新评测任务 `id`（整数）。**②失败不阻断①，原样透传错误并保留已创建 ADT task_id** |
| 查询工程链路评估状态/结果 | ① `get_adt_test_task_detail`（ADT `task_id` 为**字符串**）→② `get_taiji_eval_task_detail`（评测 `task_id` 为**整数**） | 两类 task_id 类型不同、不可混淆；ADT 侧回显 `status_text`+`is_terminal`+各测试项 `passed/total/failed` |

---

## 4. 模块注意事项

### 4.0 可配置项

工程链路评估「复制基准评测任务」的模板 ID，测试环境可替换：

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `PIPELINE_EVAL_SOURCE_TASK_ID` | `29645` | 基准源任务 ID。测试环境替换为对应 ID 即可。 |

默认模板来源类型 `ONE_STOP_SERVICE`，克隆时传 `service_name`。若替换模板，确认来源类型后调整传参。模板自身状态（`STOP`/已删除）不影响克隆（只复制评测配置）。默认模板为 Agent 类重型评测，调用前提示用户耗时较长。

### 4.1 ADT 状态码字典（status）

| status | 释义 | 终态 |
|:---:|------|:---:|
| 1 | 等待中 | |
| 2 | 运行中 | |
| 3 | 执行成功 | ✅ |
| 4 | 执行失败 | ✅ |
| 5 | 内部错误 | ✅ |
| 6 | 中止中 | |
| 7 | 执行中止 | ✅ |
| 8 | 校验失败 | ✅ |
| 9 | 审核中 | |
| 10 | 审核通过 | ✅ |
| 11 | 审核不通过 | ✅ |

测试项：`HTEXTGARBLE`=乱码检测、`HDUPLICATE`=重复检测、`HTRUNCATE`=截断检测、`HYAPITEST`=接口测试。

### 4.2 通用规则

1. 缺 `model`（模型服务组名）/`task_id` 等必填项 → 立即向用户索要，**严禁**猜测或拼接。
2. `model` 是**已部署的模型服务组名**（如 `hunyuan-standard`），不是模型名/模型 ID；用户给的疑似模型名时先向其确认是否为服务组名。
3. **工程链路评估（一键测试/链路评估）是两步编排**，调用前先向用户复述将执行的步骤再执行：
   - **第 1 步 · 创建 ADT 测试**：调 `run_adt_test_pipeline`（属**写操作**，会启动最长约 6 小时的后台轮询）；成功后记下返回的 ADT `task_id`（**字符串**），并告知用户"完成后会企微通知"。
   - **第 2 步 · 复制基准评测任务**：紧接着跨模块调 `evaluation-skill` 的 `clone_taiji_eval_task`，`source_task_id=PIPELINE_EVAL_SOURCE_TASK_ID`（默认 `29645`）、`name="工程链路评估-{model}"`；默认模板 `29645` 是 `ONE_STOP_SERVICE`，故传 **`service_name=<被测模型服务组>`**（对齐旧版 `serviceName=model`，**不用** `model_ids`）。成功后记下新评测任务 `id`（**整数**）。若替换了默认模板 ID 且其来源类型不同，则按 `evaluation-skill` 口径改用对应传参。
   - **失败隔离**：第 2 步复制评测失败**不得**回滚或阻断第 1 步——原样透传复制失败原因，同时保留并汇报已创建的 ADT `task_id`。
4. **查询工程链路评估状态/结果时，必须同时查两类任务**：用 `get_adt_test_task_detail`（ADT `task_id` 为**字符串**）查 ADT 测试，用 `evaluation-skill` 的 `get_taiji_eval_task_detail`（评测 `task_id` 为**整数**）查评测任务；两类 ID 类型不同、不可混淆。ADT 侧重点回显 `status_text` + `is_terminal` + 各测试项 `passed/total/failed`；`is_terminal=false` 时提示任务仍在进行。
5. 仅查询/停止单个 ADT 任务（非工程链路评估）时，直接用 `get_adt_test_task_detail`/`stop_adt_test_task`，无需触发评测编排。

### 4.3 严禁与失败处理

- **严禁**基于猜测的 `task_id` 调 `stop_adt_test_task`（后端不做 owner 校验，会真把别人任务停掉）；停止/操作他人任务前必须二次确认意图。
- **严禁**跨模块路由——用户问纯"评测结果/训练任务/模型管理"意图（**非**工程链路评估）时回聚合 SKILL.md 重新路由。（**例外**：工程链路评估第 2 步组合调用 `clone_taiji_eval_task`、查询时组合调用 `get_taiji_eval_task_detail`，属本 skill 明确编排，不算跨模块偏离。）
- **失败处理**：`status` 落 4/5/8（执行失败/内部错误/校验失败）等终态时，原样透传 `task_id` + 错误原因，**不主动**重试，等用户在新轮次显式触发；请求出错时把响应里的 trace_id 提供给用户排查。

> ℹ️ **实现说明**：`run_adt_test_pipeline`（ADT 后端能力）本身只做「ADT 测试 + 后台轮询 + 企微通知」，**不含复制评测**。工程链路评估对齐旧 mcp_server 版的端到端行为——「复制基准评测任务（默认源 `PIPELINE_EVAL_SOURCE_TASK_ID`=29645）」由本 skill 在 **Agent 层组合调用** `evaluation-skill` 的 `clone_taiji_eval_task` 补齐，而非 ADT 后端内置步骤。用户若只想做纯评测任务操作（与工程链路评估无关），仍请走 `evaluation-skill`。
