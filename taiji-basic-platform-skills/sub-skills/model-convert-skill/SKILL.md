---
name: model-convert-skill
description: 太极混元训练平台模型转换子 skill —— 通过 MCP 协议创建/启动/查询/克隆模型格式转换任务（Mcore(PTM2.0)↔HF、Mcore(Angel-RL)↔HF、DCP_TO_HF 等），查询转换选项与转换产出模型，同时提供 HF 模板管理能力（列表/详情/注册/路径校验/模型结构类型枚举）。当用户提及"模型转换 / 格式转换 / 转 HF / 转 Mcore / DCP_TO_HF / 转换任务 / 转换状态 / 转换进度 / 转换产出模型 / HF 模板 / HuggingFace 模板 / 注册模板 / 校验 HF 路径 / 模型结构类型 / finetuning_ 前缀任务"等关键词时，应使用本 skill。
version: 3.0.0
author: taiji-team
---

# Model Convert Skill（模型转换 & HF 模板管理）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：两类意图——

1. **模型格式转换**：创建/启动/查询/克隆转换任务、查询转换选项、查询转换产出模型。模型转换任务统一使用 `finetuning_` 前缀。收到形如 `finetuning_xxx` 的 ID 自动识别为转换任务路由到本 skill。
2. **HF 模板管理**：分页查询 HF 模板列表、按 ID 查模板详情、注册新模板、校验 HF 路径必要文件、查询平台支持的模型结构类型枚举。

**⚠️ 排除**：模型**发布 / 搜索**（走 `model-manage-skill`，本 skill 只做格式转换）；训练任务（`basic_train_*` 自定义训练 / 模型开发，`finetuning_*` 模版化训练 → `task-skill`）；模型服务 / 服务组 / 部署 → `service-deploy-skill`。

> 缺 `wsid` / 工作空间上下文时可直调 `list_user_workspaces`（通过 `scripts/tool_manual.py list_user_workspaces` 查看参数），无需提示"去加载 workspace-skill"。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **误入场景**：用户转向"模型发布/搜索、训练任务"等非模型转换意图时，立即退出。
> - **写操作**：创建、启动、克隆、注册等。

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

### A. 模型转换任务（7 个）

| 工具 | 用途 | 写操作 |
|---|---|---|
| `list_hunyuan_training_model_convert_options` | 查询模型支持的转换类型与导出格式（建议创建任务前先调用） | |
| `create_hunyuan_training_model_convert_task` | 创建模型转换任务（创建后处于 `INIT` 状态） | ✍️ |
| `start_hunyuan_training_model_convert_task` | 启动转换任务（仅 `INIT` 可启动，支持定时执行） | ✍️ |
| `get_hunyuan_training_model_convert_task_detail` | 按 task_id 查询转换任务详情 | |
| `get_hunyuan_training_latest_model_convert_task` | 按模型查询其最新一条转换任务（查进度/状态） | |
| `clone_hunyuan_training_model_convert_task` | 克隆已有转换任务配置创建新任务 | ✍️ |
| `list_hunyuan_training_model_convert_output_models` | 查询某源模型的所有转换产出模型 | |

### B. HF 模板管理（5 个）

| 工具 | 用途 | 写操作 |
|---|---|---|
| `query_hunyuan_training_hf_template_list` | 分页查询 HF 模板列表（支持关键词/来源/模型类型/仅看我的过滤） | |
| `get_hunyuan_training_hf_template_detail` | 按 `template_id` 查询模板详情（含 HF 路径、包含文件列表等） | |
| `register_hunyuan_training_hf_template` | 注册新的 HF 模板（先用 `list_hunyuan_training_hf_model_types` 拿合法 `model_type`） | ✍️ |
| `validate_hunyuan_training_hf_template_path` | 校验 HF 路径必要文件（如 `config.json`、`tokenizer.json` 等），注册前预检 | |
| `list_hunyuan_training_hf_model_types` | 查询平台支持的 HF 模板 `model_type` 枚举，无需入参 | |

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
| 🔴 **校验 HF 路径**（"校验/检查/验证 路径是否完整/规范/有效"） | 单独 `validate_hunyuan_training_hf_template_path`，传入用户给的路径字符串。**任何 ceph 路径的"检查/校验/验证"必须在 model-convert-skill 内用此工具**，严禁切到 storage-skill 做目录遍历或文件枚举。 | 路径必填，`valid=false` 展示 `missing_files` |
| 创建并运行转换任务 | ① `list_hunyuan_training_model_convert_options`（确认 `convert_type`）→ ② `create_hunyuan_training_model_convert_task`（INIT）→ ③ `start_hunyuan_training_model_convert_task` | 创建/启动边界：只有用户给齐创建必填信息且明确要求创建才执行；缺参最多只查转换选项并追问，**禁止**自行查应用组/资源/HF 模板/模型详情凑参数 |
| 查询"转换产出模型及对应任务配置" | ① `list_hunyuan_training_model_convert_output_models` → ② `get_hunyuan_training_model_convert_task_detail`（`output_models[].convert_task_id`） | 不要额外调用转换选项；不要把 `model_id` 拼成 `finetuning_<model_id>` 当 task_id |
| 注册 HF 模板 | ① `list_hunyuan_training_hf_model_types`（拿合法 `model_type`）→ ② `validate_hunyuan_training_hf_template_path`（预检 `hf_path`）→ ③ `register_hunyuan_training_hf_template` | ①返回的枚举选取 `model_type`，严禁猜测；② `valid=false` 则展示缺失文件，补齐后再注册 |
| 克隆转换任务 | `clone_hunyuan_training_model_convert_task`（用户已给 `task_id`+`wsid` 且覆盖参数齐备时直接调，不先查详情） | 克隆后新任务为 `INIT`，除非用户明确说启动，否则不自动 `start` |

> ⚠️ 列表分页：默认只查第一页 `page=1&page_size=20`；`has_more=true` 只提示"还有更多"，不主动翻页，除非用户明确说"全部/下一页/继续"。

---

## 4. 模块注意事项

### 4.1 进入判定与 convert_type

1. 用户必须显式给出 `convert_type`（如 `mcore_ptm2_to_hf` / `mcore_angelrl_to_hf` / `TO_HF` / `DCP_TO_HF` 等），**严禁**根据用户描述自行猜测枚举值；不确定时先调 `list_hunyuan_training_model_convert_options` 列出该模型支持的类型让用户确认。
2. **创建/启动边界**：只有用户已给齐创建必填信息且明确要求创建时，才调用 `create_hunyuan_training_model_convert_task`。缺参最多只查转换选项并追问缺失参数；**禁止**自行查询应用组、资源、HF 模板、模型详情来凑参数，禁止自动创建或启动。
3. 查任务状态：用户给真实 `finetuning_` 前缀 `task_id` → `get_hunyuan_training_model_convert_task_detail`；用户给模型名/ID → `get_hunyuan_training_latest_model_convert_task`。**禁止把 `model_id` 拼成 `finetuning_<...>` 当作 task_id**。
4. 展示产出模型列表时，每个产出模型须附上 `model_url` 链接（为空则不展示）。

### 4.2 HF 模板规则

1. `register_hunyuan_training_hf_template` 需要 `wsid` + `name` + `model_type` + `hf_path` 全部必填；调用前**先执行** `list_hunyuan_training_hf_model_types` 获取合法 `model_type` 枚举，**严禁**猜测；建议同时先跑一次 `validate_hunyuan_training_hf_template_path` 预检 `hf_path` 下必要文件。
2. `query_hunyuan_training_hf_template_list` 分页参数名必须是 `keyword` / `only_mine`，不要自创 `name_keyword` / `only_my` 等。
3. 创建转换任务需要指定 HF 模板时，用户给了 HF 路径可直接用 `hf_model_path`，给了模板名/ID 则用 `hf_template_id` / `hf_template_name`。
4. 注册 HF 模板时，`model_type` 必须从枚举返回中选择；混元/hunyuan 路径或无更具体证据默认选 `HY3.0`；只有明确指定非 HY3/HY4 时才用"其他"，**不要把"其他"当默认兜底**。未明确全局模板时传 `is_global=false`。用户未给描述时，可生成简短通用描述，不编造具体验证场景。

### 4.3 精确参数规则（强制）

- 只传目标工具参数表中声明的字段，禁止把上下文里的 `wsid` 机械附加到所有工具。
- 无 `wsid` 工具：`list_hunyuan_training_model_convert_options`、`get_hunyuan_training_model_convert_task_detail`、`get_hunyuan_training_hf_template_detail`、`validate_hunyuan_training_hf_template_path`、`list_hunyuan_training_hf_model_types`，调用时不要携带 `wsid`。
- 有 `wsid` 工具：创建/克隆/最新转换任务/产出模型/HF 模板列表/注册模板等，按参数表传 `wsid`。

### 4.4 常见编排规则

- 克隆转换任务：用户已给 `task_id` + `wsid` 且覆盖参数齐备时，直接调用 `clone_hunyuan_training_model_convert_task`，不要先查详情。克隆成功后新任务 `INIT`，除非用户明确说"启动"，否则不自动 start。
- URL 上下文优先：若标准太极 URL 中明确包含 `model_id` 与 `wsId`，以 URL 的 `wsId` 作为该模型转换查询上下文，优先级高于全局 wsid；查询最近转换状态时，优先围绕该 `model_id + wsId` 查转换选项或最新任务。
- 查询"转换产出模型"：只调 `list_hunyuan_training_model_convert_output_models`；后续用 `output_models[].convert_task_id` 调 `get_hunyuan_training_model_convert_task_detail`，不要额外调转换选项。

### 4.5 调用前防回归自查

先核对"目标工具是否真的需要 `wsid`"；再核对"是否在做写操作、是否所有必填参数均由用户或可信工具返回提供"；最后核对"列表是否只需首页、URL 中 `wsId` 是否应优先于其它上下文"。任一项不满足时，停止额外工具调用并向用户说明需要的缺失信息。

### 4.6 严禁

- ❌ 把"模型转换"与"模型管理（发布 / 搜索）"混淆——发布走 `model-manage-skill`，本 skill 只做格式转换。
- ❌ 猜测 `convert_type`、`quota_type`、`export_type`、`model_type` 等枚举值。
- ❌ 跨模块路由——用户问"发布模型 / 搜模型 / 训练任务"时立即跳出本 skill。

### 4.7 失败处理

任务 `FAILED` → 透传 `task_id` + 错误原因，**不主动 `retry`**，用户主动重试时再调对应工具；`CONVERT_TYPE_NOT_SUPPORTED` → 建议先 `list_hunyuan_training_model_convert_options` 选支持的类型；HF 路径校验失败 → 展示缺失文件列表，让用户补齐后再注册；401 提示 token 失效；网络/超时如实告知重试。

### 4.8 通用协议说明

- **统一响应壳**：所有工具返回统一遵循 `{"code": 0, "message": "success", "data": <业务数据>}`，分页响应 `data` 内含 `items`/`page`/`page_size`/`total`/`has_more`。`code=0` 为成功。
- **两套状态字段**：返回体中同时存在 `status`/`hy_status` 两套字段。向用户展示时优先使用 `hy_status_text`。
- **参数互斥关系**：模型标识（`model_id` > `model_name`）、HF 配置（`hf_template_id` > `hf_template_name` > `hf_model_path`）、Tokenizer（`PLATFORM_BUILT_IN`→`tokenizer`，`CUSTOMIZE`→`tokenizer_path`）均有优先级。导出默认 `auto`（此时 `export_model_name` 必填，`TRAIN_OUTPUT` + 多格式除外）。
- **任务状态流转**：`INIT → RUNNING → SUCCESS / FAILED / KILLED`。仅 `INIT` 可启动；克隆创建新 `INIT` 任务；RUNNING/SUCCESS/FAILED/KILLED 不能再次启动。

### 4.9 智能推断与上下文复用

- **convert_type 推断**："转为 HF"→`TO_HF` 或平台推荐值；"转为 Mcore"→`TO_MCORE`；"DCP 转 HF"→`DCP_TO_HF`。⚠️ 仍需用户显式确认的枚举（**严禁猜测**）：`convert_type`、`quota_type`、`model_type`——不确定时先调对应枚举查询工具。
- **source 推断**：明确"模型卡片"→`MODEL_CARD`；"训练产出"→`TRAIN_OUTPUT`；其余默认 `MODEL_CARD`。
- **上下文复用**：同一会话已给的 `wsid`、`task_id`、`model_id`、`business_flag`、`hf_template_id`/`hf_template_name` 自动复用。批量转换多模型时建议克隆复用配置。
- **分层参数收集**：第一层确认 `wsid` + 模型标识；第二层确认 `convert_type` + 资源配置；第三层可使默认值（`host_num=1`、`host_gpu_num=1`、`source=MODEL_CARD`）。避免一次性问过多。

### 4.10 错误处理策略

1. 调用前检查必填参数；缺失先收集，不带空值调用。
2. 枚举先查（`convert_type`、`model_type`），严禁猜测。
3. HF 路径先校验（`validate_hunyuan_training_hf_template_path`）。
4. 工具返回含 `error` 字段 → 向用户展示原因。
5. 优雅降级：某步骤失败说明原因并提供替代方案（创建失败→检查参数；启动失败→检查状态；克隆失败→检查原任务；模板注册失败→按缺文件/非法 `model_type` 指导）。
6. **常见错误码**：`NO_TOKEN_CONFIGURED`→引导配置；HTTP 401→Token 失效；HTTP 403→确认权限；`wsid 不能为空`→追问或 helper；`model_id 和 model_name 至少提供一个`→追问；`convert_type 不能为空`→查转换选项；`CONVERT_TYPE_NOT_SUPPORTED`→查支持类型；`任务状态不允许启动`→仅 INIT 可启动；`HF 路径缺少必要文件`→展示 `missing_files`。
