---
name: service-deploy-skill
description: 太极平台服务部署子 skill —— 通过 MCP 协议管理推理服务（Inference）与服务组（Service Group），覆盖推理服务/服务组查询编辑、实例日志、推理实例（Pod）执行命令、变更任务（扩缩容/重启/终止变更）、取消资源排队、官方推理模板查询、快速模板部署以及大模型对话。当用户提及"推理服务 / 模型服务 / 部署 / 服务组 / 实例日志 / 在推理服务实例上执行命令 / 进推理服务 pod / 扩缩容 / 重启服务 / 终止变更 / 取消排队 / 推理模板 / 快速部署 / 模板部署 / 大模型对话 / openai 对话"等关键词时，应使用本 skill。
version: 3.0.0
author: taiji-team
---

# Service Deploy Skill（服务部署）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：仅处理**推理服务（Inference）**与**服务组（Service Group）**。

**URL 判定**：看到 `instance_new?name=` 或 `instance?id=` → 推理服务，进本模块。看到 `task-inst-list?instId=` → 训练实例，退出到 `instance-skill`。

用户给的名称/ID 无法确定是推理服务还是训练任务时，停下来询问用户。

> 缺 `wsid` / 工作空间上下文时可直调 `list_user_workspaces`（通过 `scripts/tool_manual.py list_user_workspaces` 查看参数），无需提示"去加载 workspace-skill"。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **误入场景**：用户转向"训练任务、训练实例 exec、资源配额、模型发布"等非服务部署意图时，立即退出。
> - **写操作**：创建、编辑、克隆、变更、扩缩容、重启、终止变更、取消排队等。

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
>
> 📊 **版本变更**：共 **19 个工具**（旧版 20 个）。旧的 `init_service_deployment` / `init_deployment` 已被后端合并进 `create_deploy_service_change` / `create_deploy_inference_change`，不再单独暴露。

| 工具 | 用途 | 写操作 |
|---|---|---|
| `get_deploy_inference_detail` | 按名称查询单个推理服务详情（配置、状态、创建人、实例数等） | |
| `list_deploy_inferences` | 查询工作空间下推理服务列表（支持关键词搜索、创建人过滤、实例状态筛选） | |
| `update_deploy_inference` | 编辑推理服务（仅支持改 `desc` 描述和 `users` 管理员） | ✍️ |
| `clone_deploy_inference` | 克隆已有推理服务快速创建新服务（高风险，占用 GPU 资源） | ✍️ |
| `list_deploy_instances` | 查询推理服务的实例列表（按 IP/版本/状态筛选，可返回实例指标） | |
| `get_deploy_instance_logs` | 查看实例日志（启动日志 `start_log` / 事件日志 `event_log` / 请求日志 `request_log`） | |
| `exec_deploy_instance_command` | 在推理服务实例（Pod）内执行一条命令（**仅测试服务 `scene=test`**），返回 `stdout` / `stderr` / `exit_code` | ✍️ |
| `get_deploy_service_detail` | 按名称查询单个服务组详情（绑定的推理服务、权重、管理员等） | |
| `list_deploy_services` | 查询工作空间下服务组列表（支持关键词搜索、创建人过滤） | |
| `update_deploy_service` | 编辑服务组（仅支持改 `desc` 描述和 `users` 管理员） | ✍️ |
| `create_deploy_service` | 创建服务组 / 绑定一个或多个推理服务到服务组（支持简单 `inference_names` 或高级 `service_items`） | ✍️ |
| `create_deploy_service_change` | 创建服务组变更单，修改服务组绑定的推理服务列表（后端自动补齐快照，无需先 init） | ✍️ |
| `create_deploy_inference_change` | 创建推理服务变更单（扩缩容传 `replicas` / 重启传 `instance_ids`，后端自动补齐快照并推断 `change_type`） | ✍️ |
| `stop_deploy_inference_wait` | 取消推理服务排队等资源状态（不可逆，需二次确认） | ✍️ |
| `stop_deploy_inference_change` | 终止/停止进行中的变更单（不可逆，需二次确认） | ✍️ |
| `list_deploy_templates` | 查询官方推理/部署模板列表，支持按 GPU/厂商/系列/场景等筛选 | |
| `get_deploy_template_detail` | 获取单个模型 + 卡型 + 场景的推理模板配置（**新增必填 `wsid`**） | |
| `create_deploy_service_chat_completion` | 通过服务组调用大模型进行 OpenAI 兼容文本对话（支持 `stream` 流式） | |

### 本地脚本（非 MCP 工具，通过 `python3 scripts/deploy_from_template.py` 调用）

|脚本 | 用途 | 写操作 |
|---|---|---|
| `deploy_from_template.py` | 一键部署脚本，Agent 只需执行此脚本即可完成部署。脚本内部自动调用 `get_deploy_template_detail` + `clone_deploy_inference`，**Agent 不要手动分步调这两个 MCP 工具** | ✍️ |

### 跨模块速查工具

直达其他模块，通过 `scripts/tool_manual.py <工具名>` 确认参数后调用。

| 跨模块工具 | 用途 |
|---|---|
| `list_user_workspaces` | 直调工作空间查询，缺 `wsid` 时列出可访问空间供用户选定 |

---

## 3. 快速路由表（多工具流程编排）

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。
>
> 仅列多工具串联流程。单工具场景已在 §2 用途列覆盖。

| 场景 | 流程 | 流转条件 |
|------|------|----------|
| 精确名称详情链路 | `get_deploy_inference_detail(inference_name, wsid)` →（需要实例/GPU 时）`list_deploy_instances(inference_id, wsid, enable_metrics=true)` | 用户给出完整推理服务名并要求详情/实例/GPU/编辑/变更时；严禁用 `list_deploy_inferences(keyword=...)` 代替 |
| 实例执行命令（缺参数时） | `get_deploy_inference_detail(inference_name, wsid)` → `list_deploy_instances(inference_id, wsid)` → `exec_deploy_instance_command(...)` | 用户已给 `inference_id`+`instance_id` 时**一步直接执行**，不要补查；前置条件由后端强校验 |
| 应用组卡型匹配模板 | ① `resource-mgmt-skill` 调 `query_shared_resources_gpu_info_batch(app_group_ids=[app_group_id])` 取真实卡型 → ② `list_deploy_templates(wsid, gpu_name=<卡型>)` | 只调用一次；返回空即说明无匹配模板，不要改 `gpu_name`、翻页或无过滤重查 |
| 模板配置查询 | `list_deploy_templates(wsid, manufacturer=<厂商>)` → 从返回选匹配的 `model_id`/`gpu_name`/`service_scene` → `get_deploy_template_detail(model_id, wsid, gpu_name, service_scene)` | 不要使用不存在的 `keyword` 参数，也不要只停留在列表结果 |
| 可逆编辑闭环（测试 desc） | `get_detail` → `update(desc=测试值)` → `update(desc=原值)` | 用户要求把描述改成临时测试值且"不需要确认"时，完整三步闭环 |
| 快速部署 / 模板部署 | 严格按 `references/quick_deploy_flow.md` 的 SOP：查模板 → 查重名 → 查询/确认骨架服务 → 展示配置摘要 → 用户确认 → 执行 `scripts/deploy_from_template.py` | **严禁**未尝试脚本就手拼 `clone_deploy_inference` |
| 实例日志根因排查 | `get_deploy_inference_detail` → `list_deploy_instances` → `get_deploy_instance_logs` | 只取一个异常/代表实例的目标日志一次；不要对多个实例重复取日志 |
| 终止变更单（只给服务名） | ① 用 `create_deploy_inference_change` 为该服务创建一个可终止的变更单（如基于当前/测试计划设置 `replicas`）→ ② 从返回取得 `deployment_id` → ③ 立即调用 `stop_deploy_inference_change` | 用户只给推理服务名并要求"终止变更单"时；不要停在询问 deployment_id |
| 扩缩容 | ①（按需，不清楚 inference_id 时） `get_deploy_inference_detail(inference_name, wsid)` 取 `inference_id` → ② `create_deploy_inference_change(replicas=目标数, inference_id, wsid)`| 用户要求"把实例数调整为 X"时；如果缺失 inference_id，**必须先调 detail 取 inference_id** |

---

## 4. 模块注意事项

### 4.1 进入判定与最短链路

**何时进入**：用户提及推理服务、模型服务、服务组、推理实例、部署、扩缩容、重启、模板部署、大模型对话等关键词。

**最短链路判定**（进入后立即对照）：
- 连通性测试 / 服务组名 + "请求/测连通性" → 只调一次 `create_deploy_service_chat_completion`，不先查详情/列表/日志。
- 完整推理服务名 + "详情/实例/GPU/编辑/变更" → 第一步必须 `get_deploy_inference_detail`，不用 list keyword 代替。
- 应用组卡型 + "可用哪些模板" → 先切 `resource-mgmt-skill` 取卡型，再回本 skill 调一次 `list_deploy_templates`。
- 推理实例 exec ← `inference_id`+`instance_id` 已给 → 一步直接执行，不补查详情确认测试服务状态。

**非标准 URL**：用户给出非 `https://taiji.woa.com/...` 或 `https://hunyuanaide.taiji.woa.com/...` 链接 → 直接说明不是标准太极链接，禁止解码或调用工具搜索。

### 4.2 参数白名单硬约束

调用任何工具前，必须用 `info <tool>` 确认参数列表。**严禁**自创参数名（如 `search`/`name`/`q`/`filter`）、沿用其他工具的参数名套用。后端会静默丢弃未知参数 → 返回全量结果产生假象 → 必须杜绝。

### 4.3 列表结果 ≥ 30 条 → 追问筛选维度

命中时必须展示可筛选维度 + 已知关键值，追问「需要按哪个维度筛选？」。严禁自行截断或自创参数过滤。

### 4.4 精确名称 ≠ 列表搜索

用户给完整服务名 → 第一步必用 `get_deploy_inference_detail`，不用 list keyword 试探。编辑描述测试后若用临时值，应即时用原 desc 恢复。

### 4.5 搜索能力边界

⛔ 本模块不支持模糊关键词全文搜索。模板按 `manufacturer`/`manufacturer_series`/`gpu_name` 精确过滤；服务搜索仅 `list_deploy_inferences` 支持 `keyword`（不要外溢到其他工具）。

### 4.5a polaris/北极星硬边界（严禁调用工具）

**任何涉及按北极星注册地址（polaris）筛选、统计、过滤的请求，严禁调用任何 MCP 工具。** `list_deploy_inferences` 的 `keyword` 参数不支持搜索 polaris 字段，也不支持按实例状态（有实例/无实例）过滤。直接回答「当前工具不支持按北极星地址搜索推理服务」+ 列出可用的替代过滤方式（`keyword`、`only_mine`、`creator`、`instance_status`），不要尝试拉全量数据用脚本自行过滤，不要分页遍历，不要写 Python 脚本解析 JSON。

### 4.6 模板术语对齐

"模板/推理模板/部署模板"全部同义（在线部署用）。用户只说"模板"未指明训练/部署时 → 追问，严禁直接路由。

### 4.7 实例执行命令（`exec_deploy_instance_command`）

- **何时用**：推理服务实例 Pod 内跑命令。不要路由到 `instance-skill`（训练实例 exec）。
- 前置条件（`scene=test`/权限/`Running`）、高危命令拦截、shell 语法、`background` 语义——二选一错误后果严重，调 `exec_deploy_instance_command` 前务必通过 `info <tool>` 或 `tool_manual.py` 确认。

### 4.8 错误码速查

| HTTP 码 | 含义 | 处理 |
|---------|------|------|
| 400 | 业务错误（名称不存在/校验失败/参数缺失） | 读 `message` 展示 |
| 401 | Token 无效/未携带 | 走 §1.1 |
| 403 | 权限不足 | 提示联系创建人/管理员 |
| 404 | 资源不存在 | 确认 name/id |
| 500 | 后端未捕获异常 | 可能 Token 失效或参数缺失 |

### 4.9 权限与限制

- 变更单权限：仅服务创建人/管理员可发起。
- 变更单并行度：同服务不能同时两个扩缩容变更单；重启无限制。
- MCP 自动跳审批（`request_source=mcp`）。
- 空间黑名单：`create_deploy_inference_change` / `create_deploy_service_change` / `chat_completion` 受限。

### 4.10 实例日志规则

按语义选对 `log_type`（`start_log` / `event_log` / `request_log`）。根因排查只取一次日志、原样展示。

### 4.11 克隆 / 快速部署规则

- 快速部署必须优先走 `deploy_from_template.py`。严禁未尝试脚本就直调 `clone_deploy_inference`；只有脚本明确报错且 hint 指引才可回退。
- clone 严禁自主决策：所有参数（源服务/应用组/新服务名/GPU/地域）必须用户明确提供，收集完展示配置摘要确认后才执行。

### 4.12 写操作确认与变更单规则

- 扩缩容/重启/克隆/服务组变更/取消排队/终止变更 → 二次确认才执行。用户明确说"不需要确认"且参数齐备时视为已授权。
- 创建/编辑前消歧：服务组（`create_deploy_service`） vs 变更任务（`create_deploy_inference_change` / `create_deploy_service_change`）。
- 终止变更单无 `deployment_id` 时：先 create 同名空变更单取 id → 立即 stop；不停在询问阶段。

### 4.13 失败处理

扩缩容失败 → 透传 `deployment_id` + 错误，不自动回滚。401/403 提示 token 失效。网络/超时如实告知。

### 4.14 展示规范

- 列表原序展示（见 §1.3）。
- 推理服务 `name` 最长 58 字符（字母/数字开头结尾，含 `.` `_` `-`），`desc` 255 字符。可编辑字段仅 `desc` 和 `users`。
- 推荐操作链路：查列表 → 查详情 → 编辑/克隆/查实例 → 查实例日志。
