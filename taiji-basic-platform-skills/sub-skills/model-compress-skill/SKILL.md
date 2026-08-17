---
name: model-compress-skill
description: 太极平台「模型压缩」子 skill —— 通过 MCP 协议对模型进行数值精度压缩（量化，如 W4A8-FP8 / W8A8-FP8 等 W{n}A{n}-{精度} 策略）。当用户提及"模型压缩 / 量化 / W4A8 / W8A8 / FP8 / GPTQ / AWQ / SmoothQuant / INT8 / INT4 / 知识蒸馏 / pruning / quantization / distillation"等关键词时，应使用本 skill。
version: 1.1.2
author: taiji-team
---

# Model Compression Skill（模型压缩）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：模型数值精度压缩（量化）任务的策略查询、任务创建、复制、列表与详情查询（**5 个 `compress_*` 接口**）。

**⚠️ 排除（边界必须严格区分）**：
- 模型**格式转换**（HF ↔ Mcore、DCP ↔ HF 等，只换格式不动参数）→ `model-convert-skill`
- 模型**生命周期管理**（搜索 / 详情 / 发布 / 地域 / 权限 / 克隆 / 预热 / 平台枚举）→ `model-manage-skill`
- **训练任务**（`basic_train_*` 前缀）→ `task-skill`

**压缩任务归属**：压缩任务底层走 finetuning 通道，`taiji_task_id` 是 `finetuning_*` 格式，但**不要**据此误判为训练任务走 `task-skill`——压缩任务由本 skill 全权管理。

> 💡 **缺 wsid 时无需切换 skill**：本 skill 可直调 `list_user_workspaces`（通过 `scripts/tool_manual.py list_user_workspaces` 查看参数），不要提示"去加载 workspace-skill"。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **误入场景**：用户转向"模型格式转换、训练任务"等非模型压缩意图时，立即退出。
> - **写操作**：创建、复制压缩任务等。

**当前未覆盖的能力**（MCP Server 暂未提供，**不要让 Agent 试图调不存在的工具**）：启动 / 停止压缩任务（创建即视为启动）；单独查询压缩产出模型；压缩前后精度对比评估（走 `evaluation-skill`）。

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
| `get_compress_strategy` | 查询特定模型支持的压缩策略（如 `W4A8-FP8`/`W8A8-FP8` 等）；`model_id=-1` 时按 `model_name` 匹配，不需 `wsid` | |
| `create_compress_task` | 创建模型压缩任务（模板默认方案：仅 `wsid`+`task_name` 必填）；一次调用即完成，严禁重复调用 | ✍️ |
| `clone_compress_task` | 复制已有压缩任务，可替换模型/策略/数据集（仅 `id`+`wsid`+`task_name` 必填） | ✍️ |
| `list_compress_tasks` | 批量查询压缩任务（按 `wsid` 隔离 + 按 `creator` 过滤 + 分页） | |
| `get_compress_task_detail` | 查询单个压缩任务详情（按业务 `id` (int)） | |
| `list_user_workspaces` *(helper)* | 直调工作空间查询，缺 `wsid` 时列出空间供用户选定 | |

---

## 3. 快速路由表（多工具流程编排）

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。

| 场景 | 流程 | 流转条件 |
|------|------|----------|
| 🔥 **创建压缩任务** | 🔴 `get_compress_strategy` → `create_compress_task` 是**固定两步链**，一次 `tool_manual.py get_compress_strategy create_compress_task` 批量取参。禁止先查 strategy 返回后再单独 tool_manual create。 | 必须先查策略再创建 |
| 🔥 **复制/克隆压缩任务** | 🔴 `clone_compress_task` 是一次到位（不需先 get_detail），但 `tool_manual.py` 批量传入 `clone_compress_task` 即可。用户只说"复制/克隆/拷贝一份"时直接调。 | 缺 id 时先 `list_compress_tasks` 定位 |
| 查询压缩任务 | `list_compress_tasks`（单工具直达，按 `wsid`+`creator` 过滤并分页） | — |
| 查看任务详情 | `get_compress_task_detail`（需要 `id` (int)+`wsid`） | id 从 list 获取 |

## 4. 模块注意事项

### 4.1 策略与业务主键

1. **不要猜测 `compress_strategy` 字符串值**——命名空间是 `W{n}A{n}-{精度}`（如 `W8A8-FP8`/`W4A8-FP8`），后端还可能自动追加后缀（`W8A8-FP8-static`）；真实可用值**必须**用 `get_compress_strategy` 查。
2. **业务主键 `id` (int) ≠ 字符串 `task_id`**：`get_compress_task_detail`/`clone_compress_task` 入参用 `id` (int)，**不要**传 `task_id` (string)；`create_compress_task` 响应里有 `data.id`(int) 和 `data.task_id`(string) 两个字段——后续查询都用 `id`(int)。
3. **三个 ID 区别**：`id`(integer) 业务主键，用于 detail/clone；`task_id`(string) 平台内部任务 ID（纯数字字符串）；`taiji_task_id`(string) 太极任务 ID（`finetuning_{user}_{timestamp}_{hash}`）。
4. **查任务状态**：单 `id`(int) → `get_compress_task_detail`；查列表/筛选 → `list_compress_tasks`（加 `creator` 过滤）；按 `model_id`/`model_name`/`status` 过滤 → **当前不支持**，先告知用户。
5. **响应包装层检查**：5 个工具真实返回都用 `{code, message, data}` 三层包装。每次调用**必须**先检查 `code == 0`，再读 `data`；`code != 0` 时**不要**直接读 `data`，把 `message` 透传给用户。
6. 展示压缩任务列表时严格按 `data.results` 数组原始顺序，**严禁**按时间/状态/策略重排。

### 4.2 创建压缩任务

1. **最小集**：`wsid` + `task_name` 两个字段就够（MCP Server 内置模板默认配置，含 Qwen3-4B-Base / W8A8-FP8-static / 完整 run.sh 等）；其他字段都是可选覆盖项。
2. **创建任务红线**：
   - `create_compress_task` **一次调用即完成**——参数齐备时只调 1 次，**严禁**同一 case 内多次调用（多次调用会创建多个任务）。若第一次调用返回错误，**先向用户如实汇报错误**再决定是否重试，不要盲目重复调用（参数字段名/类型错误这类不会因重试而变化的错误，必须先修正参数）。
   - **产出存储目录用户不指定时**：严禁主动调 `query_storage_clusters`/`list_storage_clusters`/`list_storage` 等**任何**存储查询工具——内置模板已含默认存储配置，直接创建即可。用户明确要求换存储时才需要用户提供目录字符串，**不需要**、也**不允许** Agent 查询存储集群。
   - **不要主动列表回查**：创建成功后已拿到 `data.id` 和 `data.task_id`，**不要**再调 `list_compress_tasks`/`get_compress_task_detail`"确认"任务是否创建。
   - **策略验证时机**：用户提到具体 `compress_strategy` 值 → **必须**先 `get_compress_strategy` 验证再 create；用户明确"其他都用默认"/未提策略 → **可以跳过**，直接 create（内置模板自带默认策略）。
   - **参数齐备就 GO**：用户 query 中已指定的字段（应用组/GPU/镜像/存储时长/优先级/环境变量等），Agent **必须一次性组装完整**后单次调用 create，**不要**分多次"试探性"调用。
3. **create_compress_task 参数分组（重要）**——入参分四组，Agent 应根据用户诉求准确归组，**不要**把参数放错位置：
   - **模型识别字段**（顶层，扁平）：`model_id`/`model_name`/`model_scale`/`manufacturer` 等（覆盖 `basicConfigV2`）
   - **压缩策略字段**（顶层，扁平）：`compress_strategy`/`compress_strategy_type`/`data_type`
   - **资源配置字段**（顶层，扁平）：`app_group`（应用组）/`gpu_name`（GPU型号）/`image_name`（镜像）/`ckpt_save_time`（模型保存时长秒）/`output_storage_dir`（产出存储目录）/`location`（地域）/`host_num`/`host_gpu_num`/`env_vars`（环境变量dict）/`dynamic_scheduling_config`/`storage_quota_info` 等 40+ 字段
   - **三大嵌套配置块**（顶层，`object`）：`resource_config`（深度合并到 resourceConfig，覆盖高级配置）/`data_config`（深度合并到 dataConvertConfig，含 `trainData` 数组）/`compress_config`（深度合并到 trainConfig，含 `modelParams`/`alertConfig`）
4. **单位换算与类型规范**：涉及时长字段（`ckpt_save_time`/`keep_alive_time` 等）用户口述"30 天/3 天/1 小时"时 **必须换算为秒**（1天=86400秒，30天=2592000秒，3天=259200秒）；`host_num`/`host_gpu_num`/`model_scale`/`model_size` 是**字符串**（如 `"1"`/`"8"`，不是整数）；环境变量 `env_vars` 的 value 全部是**字符串**（`"4096"` 而非 `4096`）。
5. **snake_case 契约**：对外全部用 snake_case（应用组用 `app_group` 而非 `business_flag`；环境变量用 `env_vars` 而非 `env_vars_dict`；产出目录用 `output_storage_dir` 而非 `container_path`）。
6. **嵌套块位置**：`alertConfig` 必须在 `compress_config.alertConfig`（不要放 `resource_config` 或顶层）；数据集在 `data_config.trainData`。

### 4.3 复制压缩任务

1. **最小集**：`id`（源任务主键）+ `wsid` + `task_name` 三个字段就够（以源任务完整配置为模板，未传字段沿用源任务）。复制时用**源任务创建人**的身份调太极，当前调用者无需拥有太极工程 WRITE 权限，只需是目标工作空间成员。
2. **复制任务红线**：
   - 用户 query 出现"复制/克隆/拷贝/基于任务 XXX/clone/copy"等关键词 + 一个数字 ID → **最终动作必须是 `clone_compress_task`**，不允许只走 `get_compress_strategy` 或 `get_compress_task_detail` 就停下、也不允许用 `create_compress_task` 替代。
   - "换模型/换策略"场景：正确链路 `get_compress_strategy`（验证新模型支持该策略）→ `clone_compress_task`（一次完成复制并覆盖 model_name+compress_strategy）；错误链路 `get_compress_strategy → get_compress_task_detail`（漏 clone）。
   - "换数据集"场景：正确链路（可选 `get_compress_task_detail` 看源任务数据格式）→ `clone_compress_task`（传 `train_data` 覆盖）；错误链路只调 detail 就停下。
   - `get_compress_task_detail` **只能作为辅助**（需要看源任务数据集结构时才调），**结束步骤必须**是 `clone_compress_task`；即使 detail 已返回完整源任务信息，**也必须**再调一次 clone 才算完成复制。
   - **不要**用 `create_compress_task` 复制任务（会丢失源任务的资源/脚本/数据等完整配置）。
   - **task_name 含空格自动处理**：若用户提供的 `task_name` 含空格（如 `"test clone task"`），Agent **应自动**将空格替换为下划线或短横线（如 `test_clone_task`）后再调 clone，**不要**明知会失败仍原样传入、再等后端报错后重试。
   - clone 传数据集用 **`train_data` 顶层数组**，不是 create 用的 `data_config.trainData`。

### 4.4 空返回处理策略

本 skill 的 3 个查询工具在**返回空结果**时，Agent 必须**如实告知用户**，**严禁自行编造或扩大查询范围**：

| 工具 | 空返回定义 | 正确处理 |
|---|---|---|
| `get_compress_strategy` | `data.strategies == []` 或缺 `strategies` | 如实告知"该模型未支持任何压缩策略。可能原因：模型名称拼写错误、模型未接入压缩能力、或需联系管理员补充策略。" ❌ 猜测策略；❌ 自动换其他模型名重试；❌ 用示例策略直接 create |
| `list_compress_tasks` | `data.results == []` 或 `data.total == 0` | 如实告知"该 wsid 下匹配条件的压缩任务为 0 条"。❌ 自动扩大查询范围；❌ 编造记录；❌ 换 wsid 主动重试（除非用户明确要求） |
| `get_compress_task_detail` | 返回 `code != 0` 且 message 含"任务不存在/not found/INVALID_TASK_ID" | 如实告知"任务不存在，请确认 id 是否正确（应为 int 类型业务主键）"。❌ 自动尝试其他 wsid；❌ 把 id 当 task_id(string) 再查一次；❌ 编造 detail 字段 |

**空返回下的路由决策**：空返回**不是**触发跨 skill 路由的信号——不要因为 list 返回空就"改去 task-skill"；不要因为 strategy 返回空就"改去 model-manage-skill"。**只在用户明确要求"改查其他内容"时才切换 skill。**

### 4.5 写操作确认策略（safety · 强约束）

本 skill 有 **2 个写操作工具**（会实际创建资源、消耗集群配额）：

| 工具 | 写操作类型 | 是否需要用户确认 |
|---|---|---|
| `create_compress_task` | 创建新任务（消耗 GPU/存储） | **不需要预确认**：用户明确表达"创建/帮我建/new/create"意图 + 提供必要参数（wsid + task_name + model_name 之一）时，Agent **直接调用**，不要反问"是否确认创建" |
| `clone_compress_task` | 复制已有任务（消耗 GPU/存储） | **不需要预确认**：用户明确表达"复制/克隆/拷贝/clone/copy"意图 + 提供源任务 id 时，Agent **直接调用**，不要反问"是否确认复制" |

**⛔ 禁止的确认反问**（会浪费轮次）："我将要创建压缩任务，是否确认？"/"以下是我准备的参数：[列表]，请确认后我再调用"/"确认使用默认存储路径吗？"

**✅ 允许的澄清反问**（仅当必要参数缺失时）：
- "创建压缩任务需要工作空间 ID (wsid)，请提供" —— **只有 wsid / task_name / model 三选一都完全缺失时**才反问
- "压缩策略 W4A8-AWQ 未在 Qwen3-4B-Base 支持列表中，请从以下策略中选择：[策略列表]" —— **在 `get_compress_strategy` 返回真实策略列表后**做澄清
- "任务名 `test clone task` 含空格，我已自动改为 `test_clone_task`，如需其他名称请告知" —— **自动处理 + 告知**，不阻塞主动作

**核心原则**：Agent 是执行者，不是审核者。用户的 query 本身就是授权，除非**必要参数完全缺失**才反问，否则直接执行并把结果（含创建的任务 id / 页面链接）返回给用户复核。

### 4.6 失败处理

响应 `code != 0` → 透传 `message`，**不主动 `retry`**；`METHOD_NOT_SUPPORTED`/`ALGORITHM_NOT_SUPPORTED` → 建议先用 `get_compress_strategy` 选支持的策略；`INVALID_TASK_ID` → 提示用户检查 `id`(int)；401/`NO_TOKEN_CONFIGURED` → 进入「首次配置流程」；网络/超时如实告知重试。
