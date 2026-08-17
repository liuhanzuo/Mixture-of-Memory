---
name: kill-engine-skill
description: 太极平台查杀引擎子skill —— 通过MCP协议管理GPU低利用率等异常任务的查杀规则（增删改查/启停）并查询查杀记录。当用户提及"查杀规则 / kill规则 / 查杀引擎 / 低利用率查杀 / 创建·修改·删除·启停规则 / 查杀记录 / 哪些任务被kill"等关键词时，应使用本skill。
version: 1.0.0
author: taiji-team
---
# Kill Engine Skill（查杀引擎）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**范围**：查杀规则（kill rule）的CRUD/启停 + 查杀记录（kill record）查询。

**排除**：
- ⚠️ 「给某任务加白、防止被查杀」→ `resource-mgmt-skill`（`create_shared_resources_activity`），**不在本skill**
- 训练任务 → `task-skill`
- 资源配额 → `resource-mgmt-skill`

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **严守 app_group_id/rule_id**：用户给了直接用，禁止为校验先查列表。
> - **空结果即终态**：返回 `total=0`/`items=[]` 时如实回答"暂无匹配"，严禁自动扩大时间窗、移除过滤条件或跨模块验证。
> - **误入场景**：用户转向"任务加白/模型/评测/资源配额"时立即退出。
> - **写操作权限**：创建/修改/启停/删除需系统管理员或对应应用组管理员权限，权限不足返回 403。
> - **写操作复核**：提交写操作前向用户复核关键参数（规则名/作用域/触发条件/告警与查杀动作/阈值），得到确认后再调用。用户明确说"直接执行/不需要确认"时视为已确认。
> - **写操作危险**：`delete_shared_killengine_kill_rule` **不可恢复**——删除前务必二次确认；`preset` 预设规则后端会拒绝。

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

> 写操作标记：✍️。完整参数/返回/SOP 用`scripts/tool_manual.py <工具名>` 按需获取。

| 工具 | 用途 | 写操作 |
|---|---|---|
| `query_shared_killengine_kill_rules` | 查询查杀规则列表（按 app_group_id/名称/类型/状态/作用域分页）；过滤：`status="enabled"/"disabled"`，`scope_type="all"/"business_flag"/"business_tag"`，`task_types=["train"/"inf"]`，`rule_name` 模糊 | |
| `get_shared_killengine_kill_rule` | 查询单条规则完整定义（条件/动作/匹配配置）；入参 `rule_id` | |
| `query_shared_killengine_kill_records` | 查询查杀记录（哪些任务被kill了）；`app_group_id`后端强制必填，可选`start_at`/`end_at`/`kill_strategy` | |
| `create_shared_killengine_kill_rule` | 创建查杀规则 | ✍️ |
| `update_shared_killengine_kill_rule` | 修改查杀规则（**全量更新**生成新版本，改前必须先`get`取回当前定义） | ✍️ |
| `set_shared_killengine_kill_rule_status` | 启用/停用查杀规则 | ✍️ |
| `delete_shared_killengine_kill_rule` | 删除查杀规则（**不可恢复**；preset 规则禁删） | ✍️ |

---

## 3. 快速路由表（多工具流程编排）

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。

| 场景 | 流程 | 流转条件 |
|------|------|----------|
| 规则详情/完整配置/第一条规则 | ① `query_shared_killengine_kill_rules`（`page=1,page_size=1`取ID）→② `get_shared_killengine_kill_rule(rule_id)` | 列表返回不代替详情接口 |
| 修改规则 | ① `get_shared_killengine_kill_rule`（取回当前定义）→ ② 修改字段 → ③ `update_shared_killengine_kill_rule`（全量提交） | update是全量更新，不取回就改会丢字段 |
| kill_strategy二次过滤 | ① `query_shared_killengine_kill_records`（仅`app_group_id`）→ 取第一条的 `kill_strategy` → ② 再加 `kill_strategy` 过滤 | 第二次不附加额外分页参数 |
| 查某任务是否被查杀（给的是任务名非 task_id） | ① 回顶层走 `task-skill`定位精确`task_id`/`instance_id`/`app_group_id` → ② 回本skill `query_shared_killengine_kill_records` | 严禁枚举全部应用组/规则来"猜" |
| 创建→验证→清理闭环 | ① `create`（取返回 `rule_id`）→ ② 后续启停/更新/删除均复用该 `rule_id` | 严禁创建第二条同类规则 |

---

## 4. 模块注意事项

### 4.0 创建规则必要领域知识

创建/修改查杀规则时涉及以下枚举，**严禁猜测**，不确定先调 `info <tool>` 确认：

- `scope_type`：`all`（全局）/ `business_flag`（按应用组）/ `business_tag`（按标签）
- `condition_groups` 内 `metric`：`gpuUtil` / `gpuUtilMax` / `runDuration` / `gpuNum`；`operator`：`>` `<` `>=` `<=` `=`；`unit`：`percent` / `minute` / `hour` / `card`；GPU 类可带 `duration` + `duration_unit`（`minute`/`hour`）
- `match_config.on_metric_null`：`skip`（默认）/ `skip_and_reset` / `treat_as_zero`
- 写操作需系统管理员或应用组管理员权限，否则 403

> ⚠️ "任务加白防查杀"→ `resource-mgmt-skill`；"管理查杀规则/查杀记录"→ 本模块。

### 4.1 最小参数原则

1. 只传 `kill_engine_api.md` 明确列出的参数；**`wsid` 不是查杀引擎工具入参**，严禁透传。
2. 不要把示例中的 `task_types`/`training_task_subtypes`/`locations`/`resource_types`/`match_config` 等当默认值，除非用户明确要求或工具必填。
3. 严禁猜测枚举类参数（`scope_type`/`metric`/`operator`/`unit` 等必须来自用户明确意图或文档枚举）。

### 4.2 分页与过滤

1. **分页最小化**：不默认使用 `page_size=50`。普通过滤查询不传分页参数；用户说"前N条/最近N条"时才传 `page=1,page_size=N`。
2. **过滤条件一次到位**：`scope_type="all"` / `status="enabled"/"disabled"` / `task_types=["inf"/"train"]` / `rule_name="低利用率"` / `creator=<用户>` 一次传入，不要先无过滤查再过滤验证。
3. **最近记录时间窗**：基于当前日期生成合理`start_at/end_at`，返回空不扩大时间窗。

### 4.3 写操作单次闭环

1. 创建成功后复用返回的 `rule_id` 继续操作，严禁再创建第二条同类规则。
2. 创建临时规则时 `scope_config` 默认只写 `values`，`action_config` 默认只写 `alarm.enabled` 与 `kill.enabled`；不补任务类型/子类型/地域/资源类型/告警间隔/接收人/`match_config`。
3. 路径写错时修正路径重试同一步，不改变业务参数。

### 4.4 错误处理

| 错误 | 含义 | 处理 |
|------|------|------|
| 403 | 权限不足 | 提示需系统管理员或应用组管理员 |
| 401 | Token 失效 | 引导重新配置 |
| 40301 | 尝试删除 preset 规则 | 预设规则禁删，如实告知 |
| 网络/超时 | 服务异常 | 如实告知，不重试写操作 |
