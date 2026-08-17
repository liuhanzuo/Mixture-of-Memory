---
name: workspace-skill
description: 太极平台工作空间查询子skill —— 通过 MCP 协议查询当前用户有权限访问的工作空间（wsid）、空间用户组和应用组信息。当用户提及"我的空间 / 有哪些 wsid / 不知道空间 ID / 空间用户组"等关键词时，应使用本 skill。
version: 1.0.0
author: taiji-team
---
# Workspace Skill（工作空间）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**范围**：工作空间（workspace）查询与空间用户组查询。

**排除**：训练任务→task-skill；应用组/资源→resource-mgmt-skill；模型→model-manage-skill。

**Helper 声明**：本模块的 `list_user_workspaces` 被其他子 skill 作为 helper 直调（获取 wsid），各子 skill 通过 `scripts/tool_manual.py list_user_workspaces` 确认参数后调用。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **严守 wsid**：用户已给明确 wsid 时**严禁**再调 `list_user_workspaces`，直接使用用户给的 wsid。
> - **误入场景**：用户转向"任务/指标/评测/模型"等其他意图时，立即退出。
> - **无跨模块 helper 依赖**：`list_user_workspaces` 本身就是本模块工具，被其他子 skill 作为 helper 直调。
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

> 写操作标记：✍️。完整参数/返回/SOP 用`scripts/tool_manual.py <工具名>` 按需获取。

| 工具 | 用途 | 写操作 |
|---|---|---|
| `list_user_workspaces` | 查询当前用户有权限的工作空间列表（wsid/名称/类型/管理员）；可选 `platform` 过滤（`hunyuan`/`taiji`/`sft`）；询问wsid、不知道wsid时用此工具 | |
| `list_user_groups` | 查询指定工作空间下的用户组列表；必填 `wsid`（整数），可选 `keyword` 模糊搜索；仅在用户明确要查用户组/成员/权限组时调用，查工作空间列表时不要调 | |

---

## 3. 快速路由表

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。

暂无。

---

## 4. 模块注意事项

### 4.1 工具使用时机

1. 用户询问 wsid、不知道 wsid → 调 `list_user_workspaces` 列出候选让用户挑，**严禁猜测** wsid。
2. 用户只给了空间名称、没给 wsid → 先调 `list_user_workspaces`，用名称匹配定位 wsid 后再继续；不要跳过查询直接猜 wsid。
3. 用户已给明确 wsid（如 `10103`）→ **严禁**再调 `list_user_workspaces`，直接用。
4. 用户明确要查用户组/成员/权限组 → 调 `list_user_groups`。**严禁**在查工作空间列表时调 `list_user_groups`，两者不要混用。
5. **严禁**引导用户去网页查 wsid、用其他工具反向试探 wsid、编造或猜测 wsid。
6. 空间信息只有两种合法来源：用户自己传入、`list_user_workspaces` 工具返回。

### 4.2 查询结果复用

1. `list_user_workspaces` 成功返回后，若参数相同必须复用首次返回数据完成排序、筛选、摘要和表格化；**严禁**为确认空列表或补充展示参数相同地重复调用。
2. `list_user_groups` 成功返回后同理——空数组时直接说明"暂无用户组"，不重试不追问。
3. 用户追问满足特定条件的子集时（如"哪些是type=3 的组"），基于首次返回在内存中过滤，不重新调用。

### 4.3 返回字段注意

1. `is_admin = true` 表示**平台级管理员**——能看到全平台空间但不一定对每个空间有读写权限。
2. `list_user_groups` 的`type`：`1`=空间成员组，`2`=空间管理员组（均由后端自动维护），`3`=用户自建组。
3. 本模块工具参数均为简单类型（string/int），不涉及数组参数。

### 4.4 展示规范

1. 展示空间/用户组列表后**不追加**任何引导性问题（"接下来想做什么？"之类）。
2. 列表过长时只展示摘要+总数，不截断也不省略总数说明。
