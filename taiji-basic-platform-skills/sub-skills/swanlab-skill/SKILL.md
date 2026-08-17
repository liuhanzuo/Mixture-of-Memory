---
name: swanlab-skill
description: 太极平台 SwanLab 实验管理子 skill —— 通过 MCP 协议与 SwanLab 实验跟踪平台交互，提供身份验证、空间管理、项目管理、实验管理、指标查询、日志查看、媒体数据获取等全功能。当用户提及"SwanLab / swanlab / 实验列表 / 实验详情 / SwanLab 项目 / SwanLab 空间 / 实验日志 / 实验配置 / 实验过滤 / 媒体数据"等关键词时，应使用本 skill。
version: 2.0.0
author: taiji-team
---
# SwanLab Skill（SwanLab 实验管理）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：SwanLab 实验跟踪平台的全功能管理——身份验证、空间管理、项目管理、实验管理、指标查询、日志查看、媒体数据获取。本 skill 直接操作 SwanLab 平台，需要用户提供 **swanlab_api_key**。

**跨模块边界**：
- 平台预定义内置指标（loss / grad_norm 等）、tf_events 指标、**通过 task_id 查 SwanLab 指标**（`query_hunyuan_train_swanlab_metrics`）→ `metric-skill`
- 训练任务搜索 / 详情 / 克隆 / 启停 → `task-skill`
- 实例列表 / Pod 列表 / 日志 / 在实例 Pod 上执行命令 → `instance-skill`

> ⚠️ **与 metric-skill 的区别**：
> - **metric-skill** 中的 `query_hunyuan_train_swanlab_metrics`：通过 **task_id** 查 SwanLab 指标，后端自动解析 workspace/project/run_id，用户无需提供 SwanLab API Key
> - **本 skill**（swanlab-skill）：直接操作 SwanLab 平台，需要用户提供 **swanlab_api_key**，支持完整的空间/项目/实验管理

> 🔗 **validloss / 伴生 loss**：用户给**主任务的纯数字 task_id** 查 validloss 时，走**三步链路**（详见 `references/validloss_companion_flow.md`）：
> 1. `list_taiji_eval_companion_tasks`（evaluation 模块）→ 查伴生任务，返回含 `swanlab_run_id`、`swanlab_project`、`swanlab_api_token`、`swanlab_workspace`
> 2. `query_hunyuan_swanlab_run_columns`（本 skill）→ 列出可用指标名（伴生指标名是 `data/lm loss/{序号}.{数据集} validation` 格式）
> 3. `query_hunyuan_swanlab_run_metrics`（本 skill）→ 用完整指标名查数据
>
> ⚠️ **字段名转换**：Step 1 返回字段名是 `swanlab_api_token`，传给 Step 2/3 时对应参数名是 `swanlab_api_key`（两者是同一个值，只是字段名不同）。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **SwanLab API Key**：本 skill 的所有工具都需要 `swanlab_api_key` 参数（SwanLab 平台 API Key，与太极 PAT Token 不同）。当用户未提供 `swanlab_api_key` 或 `$ENV` 形式的 key 时，先引导用户："需要您的 SwanLab API Key 才能操作。请在 SwanLab 个人设置页获取 API Key 后提供给我。"
> - **流程文档**：`validloss_companion_flow.md`，§3 命中时先完整阅读再执行。
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

| 工具 | 用途 | 写操作 |
|---|---|---|
| `verify_hunyuan_swanlab_identity` | 验证 SwanLab API Key 是否有效，返回用户基本信息 | |
| `query_hunyuan_swanlab_workspace_list` | 查询当前用户可访问的所有空间列表 | |
| `get_hunyuan_swanlab_workspace_detail` | 获取指定空间的详细信息 | |
| `query_hunyuan_swanlab_project_list` | 查询指定空间下的项目列表（分页），支持排序和搜索 | |
| `get_hunyuan_swanlab_project_detail` | 获取指定项目的详细信息 | |
| `query_hunyuan_swanlab_run_list` | 查询指定项目下的实验列表（分页） | |
| `get_hunyuan_swanlab_run_detail` | 获取指定实验的详细信息 | |
| `get_hunyuan_swanlab_run_profile` | 获取实验的 profile 配置（超参数、元数据、依赖包、conda 环境） | |
| `filter_hunyuan_swanlab_runs` | 通过条件过滤实验列表（支持简化模式和高级模式） | |
| `query_hunyuan_swanlab_run_metrics` | 查询实验标量指标数据，支持 step/时间戳范围、采样、全量等模式 | |
| `get_hunyuan_swanlab_run_summary` | 获取实验指标统计摘要（min/max/avg/median/latest） | |
| `query_hunyuan_swanlab_run_columns` | 查询实验的指标列列表，支持按类型和关键词过滤 | |
| `query_hunyuan_swanlab_run_logs` | 获取实验运行时的文本日志，支持偏移分页和级别过滤 | |
| `query_hunyuan_swanlab_run_medias` | 获取实验的媒体数据（图片/音频/视频），返回预签名 URL | |

---

## 3. 快速路由表（多工具流程编排）

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。
>
> 仅列多工具串联流程。单工具场景已在 §2 用途列覆盖。

| 场景 | 流程 | 流转条件 |
|------|------|----------|
| validloss / 伴生 loss / 验证 loss / 主任务 loss | 三步链路：① `list_taiji_eval_companion_tasks`（主任务纯数字 task_id）→② `query_hunyuan_swanlab_run_columns`（列指标名）→③ `query_hunyuan_swanlab_run_metrics`（用完整指标名查数据） | 详见 `references/validloss_companion_flow.md`，三步缺一不可 |

---

## 4. 模块注意事项

### 4.1 通用行为

1. **必须有 `swanlab_api_key` 或 `$ENV` 形式 key**。没有时先引导用户提供，**不要空参调用工具**。
2. **按层级操作**：空间 → 项目 → 实验 → 指标/日志/媒体。如果用户直接给了 workspace + project_name + run_id，可以跳过中间步骤。
3. **分页数据展示**时标注总数和当前页码，引导用户翻页；**默认只调用一次列表工具**，不要主动翻页/重试同一页，也不要主动补 `page`/`page_size`，除非用户明确要求全部、下一页、指定页或每页数量。
4. 指标数据展示遵循表格格式 + 统计摘要的规范。
5. **成功即停**：任一 SwanLab 工具返回 `code=0` 后，禁止用相同工具+相同参数重复调用；不要为了"验证/格式化/解析"再次请求。
6. `wsid` 只是太极上下文备注，**不是 SwanLab 工具参数**；SwanLab 空间用 `workspace` 或 `username`，禁止把 `wsid` 传给 SwanLab 工具，也不要因出现 `wsid` 跨调 workspace-skill / 用户组工具。
7. 对比两个实验 profile 时，对每个 `run_id` **只调用一次** `get_hunyuan_swanlab_run_profile`；两次都成功后在本地解析 JSON 对比，禁止因解析失败重复请求同一 run。

### 4.2 意图到工具强映射

- "空间列表 / 我的 SwanLab 空间" → `query_hunyuan_swanlab_workspace_list`，不要用项目列表代替。
- "项目详情 / 项目信息" → `get_hunyuan_swanlab_project_detail`。
- "实验详情 / run 详情" → `get_hunyuan_swanlab_run_detail`；只有用户要求配置、超参数、环境、profile 时才用 `get_hunyuan_swanlab_run_profile`。
- "图片 / 音频 / 视频 / 媒体数据 / 预签名 URL" → `query_hunyuan_swanlab_run_medias`；图片媒体且用户未指定 key 时默认 `keys=["generated_images"]`；该工具只调用一次，成功返回空列表时直接说明无媒体数据，禁止再调用 columns/detail 兜底。
- "指标列筛选 / FLOAT 列 / IMAGE 列" → `query_hunyuan_swanlab_run_columns`，筛选参数名必须是 `column_type`，不要写成 `type`。
- 纯知识问答（如如何获取 API Key）→ 不调用工具，直接说明。

### 4.3 `page_size` 取值限制

后端 SDK 只接受白名单值 `(10, 12, 15, 20, 24, 27, 50, 100)`，传入其他值返回 HTTP 400。

### 4.4 严禁与失败处理

- **严禁**把"通过 task_id 查 SwanLab 指标"的需求在本 skill 处理——那属于 metric-skill 的 `query_hunyuan_train_swanlab_metrics`。
- **严禁**猜测 workspace / project_name / run_id。
- **严禁**跨模块路由——用户问"平台内置指标 / 任务 / 实例 / 评测 / 模型"时立即跳出本 skill。
- **严禁**工具返回错误后编造搜索结果、页面链接或统计数值。
- **失败处理**：API Key 无效 → 提示用户检查 Key；workspace/project/run 不存在 → 提示用户检查名称；401/403 提示 token 失效；网络/超时如实告知，不重试、不切换工具。
