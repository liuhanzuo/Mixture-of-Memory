---
name: instance-skill
description: 太极平台实例操作子 skill —— 通过 MCP 协议查询训练实例列表、Pod 列表、训练日志，在训练实例的 Pod 中执行 shell 命令（nvidia-smi / ps / df / cat / env 等），以及屏蔽且重启指定 Pod 节点。当用户提及"实例列表 / Pod 列表 / 训练日志 / 在实例上执行 / exec / 在 pod 上跑命令 / 查看 GPU / 查看进程 / 屏蔽重启 / 重启 pod"等关键词时，应使用本 skill。
version: 2.0.0
author: taiji-team
---
# Instance Skill（实例操作）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：训练任务的实例列表、实例 Pod 列表、训练日志查询；在实例 Pod 中执行 shell 命令（nvidia-smi / ps / df / cat / env 等）；屏蔽且重启指定 Pod；热更新（触发 / 版本列表 / 状态轮询）。

**URL 判定**：看到 `instId=` → 训练实例，进本模块。看到 `instance_new?name=` → 推理服务实例，退出到 `service-deploy-skill`。

**⚠️ 排除**：训练任务搜索 / 详情 / 克隆 / 启停 → `task-skill`；训练指标 / loss / grad_norm / SwanLab / tf_events → `metric-skill`；模型服务 / 服务组 → `service-deploy-skill`。

**跨模块边界**：
- 训练任务搜索 / 详情 / 克隆 / 启停 → `task-skill`
- 训练指标 / loss / grad_norm / SwanLab / tf_events → `metric-skill`
- 模型服务 / 服务组 / 实例日志 / 扩缩容 → `service-deploy-skill`

> 本模块工具均不接受 `wsid`；用户 query 中附带的 `wsid` 仅作上下文，不要传给实例/Pod/日志工具。

>🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **误入场景**：用户转向"任务搜索/启停、训练指标、模型服务"等非实例操作意图时，立即退出。
> - **exec 命令**：执行前需确认目标 instance_id + pod_name + 待执行命令，用户确认后执行。
> - **屏蔽重启**：`instance_id` 通过 `query_hunyuan_train_instance_list` 获取；`pod_names` 通过 `query_hunyuan_train_pod_list` 获取。执行前列出受影响 pod 列表，用户确认后执行。

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
| `query_hunyuan_train_instance_list` | 查询训练任务的实例列表；给定 task_id 查实例 → 用此工具 | |
| `query_hunyuan_train_pod_list` | 查询实例的 Pod 列表；查询 Pod 数量/分布/状态 → 用此工具 | |
| `query_hunyuan_train_instance_logs` | 查询训练日志关键词/错误信息；按 Pod/容器/关键词过滤 | |
| `exec_hunyuan_train_instance_command` | 在 Pod 中执行 shell 命令；查网络/节点信息/GPU 状态 → 用此工具（需用户原文命令） | ✍️ |
| `shield_restart_hunyuan_train_pod` | 屏蔽且重启 Pod；Pod 异常/坏节点/需要迁移时用此工具 | ✍️ |
| `trigger_hunyuan_train_hot_update` | 触发热更新（更新代码/配置/镜像/环境变量等，无需重启）；用户要更新运行中任务时用此工具 | ✍️ |
| `query_hunyuan_train_hot_update_versions` | 查询热更新历史版本列表 | |
| `query_hunyuan_train_hot_update_status` | 热更新状态轮询工具（配合 trigger_hunyuan_train_hot_update 使用） | |

---

## 3. 快速路由表（多工具流程编排）

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。
>
> 仅列多工具串联流程。单工具场景已在 §2 用途列覆盖。

| 场景 | 流程 | 流转条件 |
|------|------|----------|
| 用户只要 task_id 想 exec | ① `query_hunyuan_train_pod_list`（取最新实例 `instance_id` 和 `pod_names`）→② 展示 Pod 列表让用户选择 →③ `exec_hunyuan_train_instance_command`（返回的 `instance_id` + 用户选的 `pod_name` + 用户原文命令） | ①返回为空则无法执行 |
| 训练任务热更新（已知 `instance_id`） | `trigger_hunyuan_train_hot_update(instance_id, 变更字段)` | 见 §4.5 规则；至少 1 个变更字段 |

> 本模块其余场景均为单工具直达，无多工具编排。

---

## 4. 模块注意事项

### 4.1 exec 命令规则

1. **必须有**明确的 `instance_id`、`pod_name` 和用户原文给出的 `command`；**严禁**自行编造或拼接命令（含猜测 GPU 个数、路径、文件名）。
2. **登录方法咨询≠执行命令**：用户只问"容器/Pod 怎么登录、如何进入"但没要求执行 shell 命令时，说明太极实例详情页/终端入口的登录方式；不要为验证而调 exec，也不要跨模块扫描任务列表。
3. **命令安全性**：严禁自行构造 `kill`/`pkill`/`rm -rf`/`reboot`/`shutdown` 等影响训练进程的命令，除非用户明确要求。仅当用户明确要求查看某类状态但未给命令时，可映射安全只读命令（如"看 GPU"→`nvidia-smi`；查看/状态/搜索/环境类）。
4. 命令含破坏性操作（`rm -rf`/`kill`/`reboot`/修改环境）→ 执行前向用户复述并确认意图。
5. **权限要求**：当前用户必须是系统管理员、空间管理员、任务创建者、任务管理员（ADMIN）之一。
6. 展示时先显示「在实例 {instanceId} 的 Pod {podName} 上执行命令：`{command}`」，结果用代码块展示；**展示结果后严禁追加"还需要执行其他命令吗"类引导**。
7. exec 返回非 0 退出码 → 透传 stdout/stderr，**不**自动重试或换命令。

### 4.2 日志查询规则

1. **一次到位**：用户已给 `task_id`/`instance_id`/`pod_name`/`keyword`/条数时，直接映射到日志工具，同一问题默认只调用一次。
2. 太极 URL 中 `taskID`/`taskId` 直接作为 `task_id`，`instId` 直接作为 `instance_id`。
3. **训练失败诊断日志预算**：从 task-skill 切来做失败原因诊断时，默认最多查两次日志；优先用用户明确关键词，否则用 `keyword="error"`/`"failed"`（若给了明确 `instance_id` 必须带上）。两次后基于结果回答，除非用户继续指定新关键词。
4. **🔴 日志截断后的处理（禁止重调 MCP）**：日志量过大导致输出截断（`Output too large`）时，MCP 已将完整结果写入本地缓存文件（提示中有路径）。此时**严禁**再次调用 `query_hunyuan_train_instance_logs` 缩小 page_size 重查——这会产生双倍耗时和 Token。正确做法：`Read` 缓存文件或用 `grep`/`head -c`/`python3 -c "import json;..."` 直接解析。对 JSON 格式的日志文件，用 `python3 -c "import json; d=json.load(open('path')); ..."` 比反复 Read 更高效。

### 4.3 参数白名单

调用任何工具前必须以 `info <tool>` 输出为准；禁止传未声明参数、其他工具的参数字段或自定义参数。

### 4.4 屏蔽重启规则

1. 调用 `shield_restart_hunyuan_train_pod` 前**必须向用户展示即将操作的 Pod 列表并二次确认**；缺 `pod_names` 时先调 `query_hunyuan_train_pod_list` 取列表。
2. **前置条件**：任务必须开启容错（自动续训容错或单节点容错）；实例处于运行中状态；当前用户为系统管理员/空间管理员/任务创建者/任务管理员（ADMIN）之一。
3. 展示成功格式：「✅ 已成功对实例 {instanceId} 的以下 Pod 执行屏蔽且重启」+ Pod 列表 + 屏蔽原因；失败时透传错误并给处理建议。

### 4.5 易混淆工具区分

| 易混淆对 | 区分规则 |
|---|---|
| `query_hunyuan_train_instance_list` vs `query_hunyuan_train_pod_list` | **instance=实例维度**（给定 task_id 查任务下所有实例），**pod=Pod维度**（给定 instance_id 查实例内所有 Pod）。查任务下有哪些实例用 instance_list，查实例内有哪些 Pod 用 pod_list。|

### 4.5 热更新规则

1. **触发前二次确认**：调用 `trigger_hunyuan_train_hot_update` 前，必须向用户展示即将变更的内容摘要（instance_id、变更说明、涉及的文件/镜像/环境变量等）并二次确认，用户确认后才执行。
2. **前置条件**：实例处于运行中状态；任务支持热更新操作（operatorTypes 中包含 `hotUpdate`）；当前用户为系统管理员/空间管理员/任务创建者/任务管理员（ADMIN）之一。
3. 除 `instance_id` 外至少传 1 个变更字段，否则热更新无实际变更内容。
4. **状态轮询**：触发成功后若返回 `status` 为 `PENDING`/`RUNNING`，可用 `query_hunyuan_train_hot_update_status` 轮询。**不要主动轮询**，只在用户要求时执行；若用户要求"等待完成"，最多轮询 10 次，间隔 5 秒。10 次后仍非终态则告知仍在执行；FAILED 时透传 `status_message`，**不要**自动重试。
