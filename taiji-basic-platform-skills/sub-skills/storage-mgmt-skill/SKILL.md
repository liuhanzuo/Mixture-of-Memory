---
name: storage-mgmt-skill
description: 太极平台存储治理子skill —— 通过 MCP 协议查询应用组存储集群（含 COS 对象存储）、ceph 地域信息、目录权限、冷热分析目录明细，按 ceph 路径列目录（ll）、读小文件内容、查集群水位，并提交存储扩容/缩容配额申请。当用户提及"存储 / ceph /冷文件 / 冷热分析 / 目录权限 / 存储集群 / 存储配额 / COS / 对象存储 / ll / 列目录 / 读文件内容 / 集群水位 / 剩余空间"等关键词时，应使用本 skill。
version: 1.1.0
author: taiji-team
---
# Storage Management Skill（存储治理）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**范围**：
- **应用组维度（控制面）**：存储集群查询、ceph 地域查询、目录权限查询、冷热分析/目录治理明细、存储配额扩容/缩容申请
- **路径维度（数据面）**：给定完整 ceph 路径列目录（ll）、读小文件内容、查集群水位

**排除**：
- ⚠️ 应用组下HDFS 集群/配额/可用集群 → `data-processing-skill`
- 应用组资源使用/卡时配额 → `resource-mgmt-skill`
- 训练任务 → `task-skill`

**Helper 声明**：缺 `app_group_id` 时可直调 `query_shared_resources_app_group_list`；缺 `wsid` 时可直调 `list_user_workspaces`。调用前分别执行 `python3 scripts/tool_manual.py query_shared_resources_app_group_list` 或 `python3 scripts/tool_manual.py list_user_workspaces` 获取参数说明，无需切换 skill。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **严守 app_group_id**：用户给了 `app_group_id` 直接用，禁止为校验先查列表。
> - **缺必填参数**：告知缺什么并给示例值。
> - **误入场景**：用户转向"卡时配额/模型/任务"等非存储意图时，立即退出。
> - **写操作**：`apply_storage_quota` 调用前**必须**向用户展示完整申请摘要（申请人/应用组/集群/地域/当前配额→目标配额/审批人/申请理由），获得明确确认后才可调用。默认审批人 `joefang`（有固定候选集，无特殊要求时用默认，不要自行编造）。

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

**应用组维度（控制面）**：入参以 `app_group_id` 为主键

| 工具 | 用途 | 写操作 |
|---|---|---|
| `query_storage_clusters` | 查询应用组存储集群列表/冷文件大小/存储概览；默认 `storage_type=filestore`，只有用户明确说 COS/对象存储/桶/bucket 时才传`objectstore` | |
| `query_app_group_ceph_locations` | 查询应用组 ceph 地域（英文+中文逗号串+集群精简明细）；轻量版，"支持哪些地域""加模型地域前选location"用此工具 | |
| `query_storage_dir_permission` | 查询某集群下指定目录的读写/只读用户、配额与子目录权限；入参用 `dir` 不是 `path`/`container_path` | |
| `query_storage_dir_governance_detail` | 治理目录明细：合并目录文件列表与冷热分析；需`container_path`（从 `query_storage_clusters` 获取） | |
| `apply_storage_quota` | 提交文件存储配额扩容/缩容申请单（`task_type=extend_quota`）；仅支持 filestore |✍️ |

**路径维度（数据面）**：入参以完整 ceph `path` 为主键，后端自动反解应用组/集群，**不要传 `app_group_id`/`cluster_name`**

| 工具 | 用途 | 写操作 |
|---|---|---|
| `list_storage_dir` | 按完整 ceph `path` 列目录下文件/子目录元数据（类似 `ll`，最多 2000 条） | |
| `get_storage_file_content` | 按完整 ceph `path` 读取小文件内容（≤1MB，UTF-8 文本） | |
| `query_storage_cluster_free_space` | 按完整 ceph `path` 查所在集群水位：配额/已用/剩余/使用率 | |

### 跨模块速查工具

直达其他模块，通过 `scripts/tool_manual.py <工具名>` 确认参数后调用。

| 跨模块工具 | 用途 |
|---|---|
| `list_user_workspaces` | 直调工作空间查询，缺 `wsid` 时列出可访问空间供用户选定 |
| `query_shared_resources_app_group_list` | 直调查询当前用户有权限的应用组列表 |

---

## 3. 快速路由表（多工具流程编排）

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。

| 场景 | 流程 | 流转条件 |
|------|------|----------|
| 存储概览+冷文件识别（"帮我看看存储情况，有没有需要清理的"） | ① `query_storage_clusters`（取`cluster_name`/`container_path`/`location`）→ ② `query_storage_dir_governance_detail`（选代表集群，`path="/"`) | 代表集群选 `cold_size_gb` 最大且非空的一个；用户未说"遍历全部"则只查一个 |
| 目录权限/配额（"各目录配额用了多少/谁有权限"） | ① `query_storage_clusters`（取 `cluster_name`/`location`）→ ② `query_storage_dir_permission`（选代表集群，`dir="/"` 或用户指定） | 代表集群选 `used_storage_gb` 最大；禁止对第二、三个集群继续调用 |
| 存储扩容申请 | ① `query_storage_clusters`（确认当前配额）→ ② 向用户展示申请摘要→ ③ 确认后`apply_storage_quota` |②必须获得用户明确确认 |

>⚠️ **不要在用户未指定时自动串联**——每个工具独立可调用，仅当用户明确需要且缺参数时再建议先取集群信息。

---

## 4. 模块注意事项

### 4.1 两类工具的分界

- **应用组维度**（控制面）：主键是 `app_group_id`（+ `cluster_name`/`location`/`dir` 等）
- **路径维度**（数据面）：主键是**完整 ceph 路径** `path`（如 `/apdcephfs_jn5/share_305546123/hunyuan`），后端自动反解应用组/集群/地域，**无需也不要传 `app_group_id`/`cluster_name`**
- 读权限由后端按当前 Token 用户校验；无权限报错是正常鉴权结果，不要归因为"路径不存在/工具故障"

### 4.2 默认 filestore

1. 用户只说"存储集群/存储空间/冷文件/冷热分析/目录权限/配额"→ 默认 `storage_type=filestore`，不要同时查`objectstore`。
2. 只有明确说"COS/对象存储/桶/bucket"→ 才用 `storage_type=objectstore`。
3. **COS 能力边界（严禁调工具）**：当前仅支持 `query_storage_clusters(storage_type=objectstore)` 查 COS 概览。以下场景**严禁调用任何 MCP 工具**，直接回答"当前不支持"：
   - COS 桶扩容/缩容 → 不支持（`apply_storage_quota` 仅限 filestore）
   - COS 桶目录权限设置 → 不支持（`query_storage_dir_permission` 仅限 filestore）
   - COS 桶文件列表/目录浏览 → 不支持（`list_storage_dir` 仅限 filestore）
   - COS 桶读文件内容 → 不支持（`get_storage_file_content` 仅限 filestore）

### 4.3 fan-out 控制（严格）

1. **默认不遍历全部集群/应用组**：用户未说"全部/每个/遍历所有"时，只选**唯一一个代表集群**调用下游工具。
2. 代表集群选法：一般优先`used_storage_gb` 最大；冷热治理优先 `cold_size_gb` 最大且非空；目录名明确时直接查该目录。
3. 目录明细默认只查当前层（`path="/"`），不自动下钻子目录。
4. **去重调用**：同一 `app_group_id + cluster_name + dir/path` 的查询只调用一次，结果可用时直接回答，不要重复确认。

### 4.4 路径维度注意

1. 用户给出完整 ceph 路径直接调用 `list_storage_dir`/`get_storage_file_content`/`query_storage_cluster_free_space`，不要再索要`app_group_id`/`cluster_name`，也不要先跑 `query_storage_clusters`。
2. `get_storage_file_content` 仅限小文本（≤1MB）；疑似大文件先用 `list_storage_dir` 看 `size_bytes`。
3. `list_storage_dir` 最多 2000 条；达上限时提醒用户进入子目录再查。

### 4.5 参数易混淆点

1. `query_storage_dir_permission` 的目录参数是 `dir`（不是 `path`/`container_path`）。
2. `query_storage_dir_governance_detail` 需要 `container_path`（从 `query_storage_clusters` 返回获取）。
3. `query_app_group_ceph_locations` 返回的`location`/`ch_location` 是逗号分隔字符串（非数组）。
4. 对象存储 `total_storage_gb` 多为`0`、`cold_size_gb` 多为 `null` 是正常特征。

### 4.6 纯How-to 不调用工具

用户只问"如何/怎么申请/流程/说明"且未要求查询当前资源或提交申请时，基于文档回答即可，不要先调用 MCP。

### 4.7 错误处理

| 错误 | 含义 | 处理 |
|------|------|------|
| HTTP 403 | 权限不足 | 提示确认用户是该应用组成员 |
| `用户对路径无读权限` | 正常鉴权 | 据实转达，不重试不绕道 |
| `file too large: >1MB` | 文件超限 | 提示改用 `list_storage_dir` 看元数据或选更小文件 |
| 冷热分析超时 | 降级 | 工具仍返回目录基础信息，冷热列显示 `—`，不整体报错 |
| `路径不存在` | 路径拼写错误 | 提示核对路径，可先`list_storage_dir` 看上层|
