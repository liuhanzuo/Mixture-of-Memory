---
name: resource-mgmt-skill
description: 太极平台资源管理子 skill。查询应用组列表/详情、GPU 资源与租借、GPU/CPU 任务、GPU 利用率/MFU、任务资源与额度腾挪记录，并支持父子应用组绑定/解绑、排队优先级调整/置顶、用户 GPU 额度、成员/管理员及资源审批单管理。当用户提及应用组、业务组、资源/GPU 配额、GPU 利用率、卡时、排队任务、弹性资源、资源分层、任务置顶、用户额度、成员管理或资源审批时，应使用本 skill。
version: 3.0.0
author: taiji-team
---

# Resource Mgmt Skill（资源管理）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：应用组（业务组）维度的资源查询与管理。查询类含应用组列表/详情、GPU 资源明细/租借、GPU/CPU 任务列表、GPU 利用率/MFU、任务资源明细、额度腾挪记录；写操作类含父子应用组绑定/解绑、任务排队优先级调整/置顶、用户 GPU 额度配置、资源审批单创建、应用组成员/管理员更新。

**⚠️ 排除**：`应用组 + 可选择/有哪些/列举` **且** 上下文含 `预训练/parquet/转bin/tokenizer` → 属 `data-processing-skill`（`query_hunyuan_data_app_groups`）。不含预训练上下文的"应用组列表"仍走本模块。

**跨模块边界**：
- 训练任务详情、启停、日志（"任务资源消耗"理解为任务详情时）→ `task-skill`
- 模型服务 / 服务组 / 实例日志 / 扩缩容 → `service-deploy-skill`
- 模型搜索与发布 → `model-manage-skill`
- 存储集群 / ceph 地域 / 冷热分析 → `storage-mgmt-skill`
- 查杀规则与记录 → `kill-engine-skill`（"任务加白"仍在本模块）

> 缺 `wsid` / 工作空间上下文时可直调 `list_user_workspaces`（通过 `scripts/tool_manual.py list_user_workspaces` 查看参数），无需提示"去加载 workspace-skill"。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **误入场景**：用户转向"训练任务、存储集群、查杀规则"等非资源管理意图时，立即退出。
> - **写操作**：创建、更新、删除、停止、绑定、授权、扩容、调整优先级等。

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
| `query_shared_resources_app_group_list` | 分页查询应用组列表；"平台上有哪些应用组"不加过滤，"我可用哪些"传 `is_usable=true` | |
| `get_shared_resources_app_group_detail` | **查询任意应用组详情**（权限/配置/成员/额度）；知道 `app_group_id` 直接查，不管 OWNER/MEMBER 角色都能返回权限信息 | |
| `query_shared_resources_gpu_info_batch` | 批量查多个应用组 GPU 汇总（总/已用/空闲/排队）；"对比A、B资源"一次传多个 `app_group_ids` | |
| `query_shared_resources_gpu_resource_info` | 单个应用组按集群+卡型 GPU 明细（含被屏蔽节点） | |
| `query_shared_resources_gpu_rent_info` | 单个应用组含租借的 GPU 概览（自有+租入+租出）；"资源使用概览/配额概览 / GPU 卡情况 / 空卡 / 排队"优先用此 | |
| `query_shared_resources_gpu_job_list` | 分页查 GPU 任务列表（多维过滤：状态/卡型/地域/创建人/优先级）；"排队任务"传 `status:["waiting"]` | |
| `query_shared_resources_cpu_job_list` | 分页查 CPU 任务列表 | |
| `query_shared_resources_user_gpu_util_rank` | 应用组内用户 GPU 利用率排行 | |
| `query_shared_resources_job_gpu_ratio` | 批量查任务实例 GPU 利用率（需先拿 `instance_ids`） | |
| `query_shared_resources_job_mfu` | 批量查任务实例 MFU/算力利用率（需先拿 `instance_ids`） | |
| `query_shared_resources_task_resource_detail` | 任务实例级资源**变动流水**（"资源怎么变动"） | |
| `query_shared_resources_quota_transfer_log` | 应用组配额级额度腾挪历史 | |
| `query_shared_resources_task_resource_usage` | 任务级**卡时/利用率消耗**统计（"消耗多少卡时"），`start_time`/`end_time` 用纯日期格式 | |
| `query_shared_resources_app_group_waiting_queue` | 等待队列（排队任务）；写操作 `instance_uuid` 取此返回的 `id` | |
| `query_shared_resources_gpu_cluster_resource` | 集群物理资源/被屏蔽节点/僵尸卡 | |
| `query_shared_resources_user_app_groups` | **我的应用组列表**——只返回 OWNER 角色的应用组；不问"有哪些"、"我没权限"、"查某个app_group"这类场景 | |
| `validate_shared_resources_sub_business_bindable` | 父子应用组绑定前置校验（纯读） | |
| `bind_shared_resources_sub_business` | 绑定父子应用组（含额度腾挪）；必须先 validate校验通过 | ✍️ |
| `unbind_shared_resources_sub_business` | 解绑父子应用组 | ✍️ |
| `update_shared_resources_task_queuing_priority` | 调整等待任务优先级（P1/P2/P3）；仅 `status=waiting` 生效 | ✍️ |
| `update_shared_resources_task_top_priority` | 等待任务置顶；与 queuing 语义不同，不要互相代替 | ✍️ |
| `update_shared_resources_user_capacity` | 更新用户 GPU 个人额度；全卡型替换，必须先查详情现状 | ✍️ |
| `update_shared_resources_app_group_members` | 更新应用组成员/管理员（全量覆盖）；需先查详情做 read-modify-write | ✍️ |
| `create_shared_resources_activity` | 创建资源审批单/任务加白；加白用 `type="task_whitelist"`，成功后展示 `approval_url` | ✍️ |

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
| 正在跑的训练任务 GPU 利用率 | ① `query_shared_resources_gpu_job_list`（`status:["running"]`, `task_type:["train"]`）→② `query_shared_resources_job_gpu_ratio`（`instance_ids` 取①返回的 `id`，加 `time_range`） | ①返回为空则无数据；只问 MFU 时②换 `job_mfu` |
| 最近一次训练的 GPU 利用率（仅给wsid） | ① `query_shared_resources_app_group_list`（拿 `app_group_id`）→ ② `query_shared_resources_task_resource_detail`（`desc:true`, `page_size:1`）→ ③ `query_shared_resources_job_gpu_ratio`（`instance_ids` 取②的 `instance_uuid`） | **全程在本模块完成**，严禁切 `task-skill`；无采样则说明无数据 |
| 给应用组加/删成员或管理员 | ① `get_shared_resources_app_group_detail`（取现有名单）→ ② `update_shared_resources_app_group_members`（read-modify-write 全量覆盖） | 先确认当前用户在 `owner_list` 内 |
| 更新用户 GPU 个人额度 | ① `get_shared_resources_app_group_detail`（取 `user_gpu_config`）→ ② `update_shared_resources_user_capacity`（基于现状修改） | 不查盲改会丢失其他卡型 |
| 父子应用组绑定 | ① `validate_shared_resources_sub_business_bindable`（校验）→ ② `bind_shared_resources_sub_business`（绑定） | ① `can_bind==true` 才继续 |

> ⚠️ 区分两个"任务资源"工具：`task_resource_detail` 是资源**变动流水**，`task_resource_usage` 是**卡时/利用率消耗统计**。问"消耗多少卡时/利用率"用后者，问"资源怎么变动"用前者。

---

## 4. 模块注意事项

### 4.1 应用组查询场景路由

三个查询应用组的工具覆盖不同的权限级别，必须按用户意图选择：

| 用户问 | 正确工具 | 原因 |
|---|---|---|
| "我有哪些/我管理哪些应用组" | `query_shared_resources_user_app_groups` | **只返回 OWNER 角色的应用组**（我管理的），MEMBER 级别不在列表中 |
| "哪些应用组我能用/给我列出可用的" | `query_shared_resources_app_group_list` + `is_usable=true` | 返回当前用户**可用**的应用组（含 OWNER+MEMBER） |
| "平台上都有哪些应用组" | `query_shared_resources_app_group_list`（不加过滤） | 全平台应用组列表 |
| "我有没有 XX 应用组的权限" / 已知 `app_group_id` 查详情 | `get_shared_resources_app_group_detail` | 不管 OWNER/MEMBER，知道 ID 就能查；返回 `role` 字段直接告诉你是什么角色 |

> ⚠️ 不要用 `query_shared_resources_user_app_groups` 回答"我有没有某应用组权限"——MEMBER 级别的应用组不在此列表中，会误判为"无权限"。

### 4.2 应用组标识规则

1. `app_group_id` 参数直接接受应用组字符串名称（如 `TaiJi_HYAide_DopTest`）。
2. 用户没说 `app_group_id` → 先调 `query_shared_resources_app_group_list`（可 `is_usable=true`）；**严禁猜测**应用组标识。
3. 用户给出形如 `TaiJi_...` 的标识时，直接按 `app_group_id` 使用；不要因 `is_usable=true` 查不到就反复列表搜索。

### 4.3 资源数据特性

1. 资源数据具有时效性，结果中标注查询时间。
2. 多应用组汇总优先用 batch 工具（`query_shared_resources_gpu_info_batch`），但必须按应用组分别展示，不能混成单一资源池口径。
3. 返回空时说明可能无权限或标识错误，不默认断言"没有资源"。
4. ⏱️ **时间格式因工具而异**：`gpu_job_list`/`cpu_job_list`/`user_gpu_util_rank`/`task_resource_detail`/`quota_transfer_log` 为 `yyyy-MM-dd HH:mm:ss`；`task_resource_usage` 为纯日期 `yyyy-MM-dd`。
5. 错误处理：`HTTP 401` → Token 失效引导重签；`HTTP 403` → 无权限建议查列表确认归属；`code != 0 + "not exist"` → app_group_id 错误；超时 → 稍后重试。

### 4.4 写操作总则

1. **所有写操作提交前必须把全部核心参数原样列给用户复核**，用户明确确认后才允许调。若用户明确写了「MCP 测试验证用 + 不需要确认」，可跳过人工复核直接执行一次。
2. 用户说"一个排队任务/有个任务"时，只选择列表中的第一个合格任务执行写操作，严禁批量更新多个任务。
3. 具体工具的前置查询、状态限制、覆盖语义、read-modify-write、确认要求以对应工具手册为准（调用前通过 `tool_manual.py` 读取）。
4. 用户未明确写操作目标时不得自行选择对象、批量操作或修改他人资源。
5. 写操作失败时透传 `message`，不要擅自改字段重试。

### 4.5 易混淆工具区分

| 易混淆对| 区分规则 |
|----------|----------|
| `task_resource_detail` vs `task_resource_usage` | 前者是资源**变动流水**（"怎么变动"），后者是**卡时/利用率消耗统计**（"消耗多少"）。`task_resource_usage` 的时间格式是纯日期 `yyyy-MM-dd` |
| `change_shared_resources_task_queuing_priority` vs `upper_shared_resources_task_priority` | 调级用 queuing（含 `target_priority`），置顶用 top。**不要互相代替** |
| `query_shared_resources_gpu_info_batch` vs `query_shared_resources_gpu_rent_info` vs `query_shared_resources_gpu_resource_info` | batch 批量多应用组汇总；rent_info 单应用组概览（自有+租入+租出，问"空卡/排队"用此）；resource_info 单应用组按集群+卡型明细（问"某卡型在某集群有多少"）。不要用 resource_info 替代 rent_info 做概览查询 |
| `query_shared_resources_user_app_groups` vs `get_shared_resources_app_group_detail` | 前者**只返回我 OWNER 的应用组列表**（我的），后者能查**任意已知 ID 的应用组详情**（含 MEMBER 等非 OWNER 权限）。用户问"我有没有某应用组权限"或已知 `app_group_id` 时，直接用 detail 查询，不要依赖 user_app_groups 列表（MEMBER 级别的组不在此列表出现） |
