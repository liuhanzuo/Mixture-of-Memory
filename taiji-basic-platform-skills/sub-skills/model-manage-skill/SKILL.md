---
name: model-manage-skill
description: 太极平台模型管理子 skill —— 通过 MCP 协议搜索模型、查看模型详情、按训练条件查官方模型、发布训练产出为模型卡片、查询模型发布任务状态、克隆模型、管理模型地域与权限、预热模型缓存、查询平台枚举值、查询模型血缘。当用户提及"搜索模型 / 模型详情 / 发布模型 / 发布 ckpt / 发布 checkpoint / 发布状态 / 发布进度 / 模型卡片 / 官方模型 / 可训练模型 / 克隆模型 / 模型地域 / 模型权限 / 模型预热 / 平台枚举 / 模型血缘 / 模型谱系 / 父模型 / 子模型 / task_record_id / release_status"等关键词时，应使用本 skill。
version: 3.0.0
author: taiji-team
---

# Model Manage Skill（模型管理）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：模型**搜索 / 详情 / 官方模型查询 / 发布模型卡片 / 发布状态查询 / 克隆模型 / 模型地域管理 / 模型权限管理 / 模型预热 / 平台枚举查询 / 模型血缘**等模型管理意图。

**⚠️ 排除（强制）**：
- 预训练 tokenizer 三级级联**不是模型管理**。如果用户问"tokenizer 的 master 分支下面有哪些模型系列""master 分支的 hy_3 模型系列有哪些训练阶段""代码分支/模型系列/训练阶段"等预训练转 bin tokenizer 级联问题，立即停止模型卡搜索/平台枚举，改按 `data-processing-skill` 契约直接调用 `list_hunyuan_data_pretrain_tokenizer_model_series` 或 `list_hunyuan_data_pretrain_tokenizer_train_stages`，**不要**调用 `search_hunyuan_models_cards`、`search_hunyuan_official_models_by_train`、`list_hunyuan_models_platform_enums` 或 WebFetch。
- 模型**格式转换**（HF / Mcore）→ `model-convert-skill`；训练任务 → `task-skill`。

> 以下查询无需切换 skill：本 skill 可直接补调工作空间、应用组、训练产出列表相关工具（通过 `scripts/tool_manual.py` 确认参数后调用），不要提示去加载 workspace-skill / resource-mgmt-skill / task-skill。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **误入场景**：用户转向"格式转换、训练任务、预训练 tokenizer 级联"等非模型管理意图时，立即退出。
> - **写操作**：发布、克隆、权限、地域、预热等。

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
| `search_hunyuan_models_cards` | 搜索模型。「我有权限的」→ `keyword=""`, `is_my_model=true`, `is_show_all=false`；「我创建的」→ `is_my_create=true`；「所有/关键词搜索」→ `keyword="<词>"`, `is_show_all=true`。wsid 必传，不拆词扩搜 | |
| `get_hunyuan_models_card_detail` | 按 model_id 查询模型详情（基本信息 / 地域分布 / 训练信息） | |
| `search_hunyuan_official_models_by_train` | 按训练类型/方式/框架反查支持训练的**官方模型**。查全部不传额外参数，按训练类型筛选传 `train_type="<类型>"` | |
| `list_hunyuan_models_platform_enums` | 查询平台枚举值（参数规模=`SCALE_CHOICES_NEW`，地域=`LOCATION_MAP`，训练方法=`train_method_enums`） | |
| `release_hunyuan_training_checkpoint_as_model` | 发布训练产出 checkpoint 为模型卡片（自定义训练需额外参数） | ✍️ |
| `clone_hunyuan_models_card` | 从已有模型卡片克隆出新卡片 | ✍️ |
| `update_hunyuan_models_card_permission` | 增量设置模型卡片权限（管理员/成员/可展示·可使用空间） | ✍️ |
| `update_hunyuan_models_card_location` | 为模型卡片新增地域并拷贝文件 | ✍️ |
| `create_hunyuan_models_cache` | 预热模型到指定地域高速缓存（异步；能否预热由后端判定，Agent 不预判） | ✍️ |
| `get_hunyuan_models_lineage` | 获取模型血缘数据（父子谱系），追溯上游/下游模型链路 | |
| `query_hunyuan_models_lineage_eval_tasks` | 查询模型关联的评测任务列表。items 为空就回答无关联评测，不调详情/evaluation-skill | |
| `get_hunyuan_training_model_release_status` | 查询**模型发布**异步任务的状态（配套 release 使用，支持 `task_record_id` / `model_id` / `model_name`） | |

### 跨模块速查工具

直达其他模块，通过 `scripts/tool_manual.py <工具名>` 确认参数后调用。

| 跨模块工具 | 用途 |
|---|---|
| `list_user_workspaces` | 直调工作空间查询，缺 `wsid` 时列出可访问空间；发布模型/搜索官方可训练模型时做权限校验 |
| `query_app_group_ceph_locations` | 直调查询应用组 ceph 地域与挂载路径，选目标地域 `container_path` 用于 `update_hunyuan_models_card_location` |
| `query_hunyuan_train_checkpoint_list` | 直调查询训练任务 checkpoint 产出列表，发布模型前取 `instance_id`/`checkpoint`/`path` |
| `query_user_app_groups` | 直调查询当前用户有权限的应用组列表，新增模型地域缺 `queue_name` 时使用 |

---

## 3. 快速路由表（多工具流程编排）

> 💡 仅列多工具串联流程，单工具场景已在 §2 用途列覆盖。涉及多工具时用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取参数。**命中后按此固定最小链路执行，不自行追加搜索/详情/枚举或跨 skill 调用。**

| 意图类别 | 固定工具链 | 关键参数与禁止项 |
|---|---|---|
| 列出有权限模型并查看参数量/地域等详情 | `search_hunyuan_models_cards` → `get_hunyuan_models_card_detail` | 先查有权限模型首页，再对一个代表模型查详情；配额不在模型搜索/详情中时如实说明，不查应用组配额 |
| 基于已有模型名称克隆创建新模型卡片 | `search_hunyuan_models_cards` → `get_hunyuan_models_card_detail` → `clone_hunyuan_models_card` | 源只有名称时完整搜索一次；clone 报 `path目录为空` 直接透传；禁止 release/训练任务查询 |
| 对指定模型进行地域拷贝/重拷贝（已知 model_id） | `get_hunyuan_models_card_detail` → `update_hunyuan_models_card_location` | 已给 model_id 直接详情；复用失败地域 `queue_name` 和 path 前两级；禁止拆词搜索、训练任务、数据导出 |
| 对指定模型进行地域拷贝/重拷贝（仅知名称） | `search_hunyuan_models_cards` → `get_hunyuan_models_card_detail` → `update_hunyuan_models_card_location` | 完整 search 一次后取 model_id 查详情并 update；禁止追加搜索/lineage |
| 模型预热 | `get_hunyuan_models_card_detail` → `create_hunyuan_models_cache` | 从 `location_infos[].id` 取 `regions`；预热失败只透传，不重复调用 |
| 查找某类模型模板是否可复用 | `search_hunyuan_models_cards` → `search_hunyuan_official_models_by_train` | 严格仅两次调用：先按名称搜索模型卡片，再查官方可训练模型；即使没找到也不继续换关键词搜索、不查详情/血缘/枚举/clone |
| 基于训练 checkpoint 发布模型 | `query_hunyuan_train_checkpoint_list` → `release_hunyuan_training_checkpoint_as_model` | 禁止 `get_hunyuan_train_task_detail` / config / 搜索 / clone |
| 基础训练任务产出发布模型（需补发布枚举） | `query_hunyuan_train_checkpoint_list` → `list_hunyuan_models_platform_enums` → `release_hunyuan_training_checkpoint_as_model` | 只为补发布枚举调一次 enums；禁止任务详情/config |
| 查询模型发布任务状态 | `get_hunyuan_training_model_release_status` | 仅查"发布"这一次异步任务；优先用 `task_record_id`，只调本工具一次，禁止 `search_hunyuan_models_cards` / `get_hunyuan_models_card_detail` 探路 |
| 血缘查询（"血缘/谱系/父模型/子模型"入口） | `get_hunyuan_models_lineage` → `query_hunyuan_models_lineage_eval_tasks`（分页拉全） | 参见 §4.6 |

---

## 4. 模块注意事项

### 4.1 高频执行约束（先读，减少冗余调用）

1. **避免穷举遍历**：列表/搜索类问题先调用一次 `search_hunyuan_models_cards`（默认 `page=1`、建议 `page_size=20`）返回首页结果与 `total/has_more`；结果较多则展示首页并询问是否继续翻页，不自动连续翻页穷举全部结果。
2. **普通模型搜索 ≠ 官方可训练模型**：`search_hunyuan_official_models_by_train` 只用于"官方模型 / 支持训练 / SFT / DPO / LORA / 模板候选"。普通"有权限使用的模型、自己发布的模型、名字包含关键词、某创建人的模型"一律用 `search_hunyuan_models_cards`。
3. **名称找源模型**：用户给源模型名称但无 `model_id` 时，先用完整名称 `search_hunyuan_models_cards(keyword=<完整名称>, is_show_all=true)` 搜一次；拿到最匹配 `model_id` 后再 `get_hunyuan_models_card_detail`，不要改用训练任务、release 或多轮模糊搜索。
4. **写入错误不自动换路**：`clone/release/update_location/cache/permission` 返回业务错误（如 `path目录为空`、`地域已存在`、`checkpoint已发布`、`无权限`、`该格式模型暂不支持预热`）时，直接透传并给建议；不要因错误再搜索、克隆、发布或跨模块尝试绕过。
5. **用户已授权直接执行**：用户明确写了"不需要确认 / 无需向我确认 / 直接按你的计划执行"，视为已确认对应写操作。
6. **无结果也要停**：目标工具已完成一次查询但返回空列表或业务错误，要如实报告"未找到/为空/失败原因"并停止；不要继续换关键词扫描、枚举 ID、查训练任务、查评测任务或切到其他子 skill。
7. **已给 ID 直达**：用户已给 `model_id` 时，血缘、关联评测、预热、权限、地域等任务均直接用该 ID 调目标工具；不要先查模型详情，除非该目标工具明确需要 `location_infos`（预热/地域重拷贝）。
   - **URL 解析**：用户给出 `/model_archives_detail?id=187672&wsId=10362` 这类页面 URL 时，`id=` 就是 `model_id`，`wsId=` 就是 `wsid`。直接提取数值调用对应工具，可以省去从网页标题/路径名中抽取关键词去 search。
8. **同一 query 工具预算**：除非用户明确要求继续翻页或对多个不同 model_id 批量操作，否则同一 query 内 `search_hunyuan_models_cards` 默认只查首页、`get_hunyuan_models_card_detail` 最多 1 次、`search_hunyuan_official_models_by_train` 最多 1 次；`clone/release/update_location/create_cache` 写入工具各最多 1 次。结果较多时先展示已获取部分并询问是否需要更多，不允许自行换关键词穷举继续试。
9. **禁止用详情/文档探路**：固定链路表已给出工具和参数时，不要再通过 `get_hunyuan_models_card_detail`、Grep/Read 文档、训练详情/config 等做探路；直接按链路调用业务工具。

### 4.2 发布模型（写操作）

1. 涉及"发布"语义（`release_hunyuan_training_checkpoint_as_model`）→ **写入操作**，调用前向用户复述要发布的模型名 / checkpoint / 目标 wsid，确认后才执行；用户明确说"无需确认/直接执行"，视为已确认。
2. 发布前优先通过 `scripts/tool_manual.py query_hunyuan_train_checkpoint_list` 确认参数后调用，取 `instance_id`/`checkpoint`/`path`，基础参数为 `instance_id`/`name`/`desc`/`checkpoint`/`wsid`，自定义训练**必须额外收集** `manufacturer`/`manufacturer_series`/`model_structure`/`total_params`/`context_len`/`scene_train`/`scene_text_type`，不确定取值先调 `list_hunyuan_models_platform_enums`。dense 只需 `total_params`，moe 还需 `activate_params`（≤ `total_params`）。
3. **发布后查状态**：`release_hunyuan_training_checkpoint_as_model` 是异步任务，仅返回 `task_record_id` 和初始状态；当用户问"发布到哪了"时，直接用该 ID 调用 `get_hunyuan_training_model_release_status` 一次即可，**不要**改走 `search_hunyuan_models_cards` / `get_hunyuan_models_card_detail` 探路。
4. **职责边界**：本工具只查"模型发布"这一次异步任务的整体状态。

### 4.3 地域拷贝与权限

1. 多地域拷贝（`update_hunyuan_models_card_location`）涉及跨地域数据，调用前确认目标地域用户有权限；若用户要求"重新拷贝/重试"已存在但 `copy_state=failed` 的地域，先 `get_hunyuan_models_card_detail`，复用该地域已有信息和路径。
2. `update_hunyuan_models_card_permission` 为**增量操作**：用户说"加某人"用 `operation=add` 只传需增加项，"移除"用 `operation=remove`，同一操作方向下可把多个权限字段合并到一次调用。

### 4.4 预热

先 `get_hunyuan_models_card_detail` 取 `location_infos[].id` 作为 `regions` 值（整数 ID，非英文缩写）。已拿到详情时不重复查。**不做客户端拦截**：参数齐备直接调用，不因 `hf_model_path` 为空等自行拒绝。后端返回不支持/权限不足时直接透传并停止，不换参数重试。

### 4.5 血缘与评测联动

1. **训练产出/评测血缘最小链路**：如果上游任务产出已返回精确 `model_id`，直接 `get_hunyuan_models_card_detail(model_id)` 一次；用户问"这个模型的评测/得分/评估任务"时立即切 `evaluation-skill` 用 `list_taiji_eval_tasks`，不要再调用 `search_hunyuan_models_cards` 做血缘枚举。
2. **血缘意图专用联动**：当用户明确问"血缘 / 谱系 / 父模型 / 子模型 / 上下游模型"时，必须在一次 `get_hunyuan_models_lineage` 后**紧接着**联动调用 `query_hunyuan_models_lineage_eval_tasks` 把该模型关联的评测任务**分页拉全**（默认 `page=1, page_size=100`，该接口 `page_size` 上限，用大页减少翻页次数；循环递增 `page` 直到返回 `has_more=false` 或已收集到 `total` 条为止），把血缘节点/关系与合并后的完整评测任务列表一并汇报；此联动仅用于血缘入口，不适用于"只问评测/得分"的入口。
3. `query_hunyuan_models_lineage_eval_tasks` 单独调用时同样默认 `page=1, page_size=100` 拉首页，但**不自动翻页**（`has_more=true` 时提示可继续翻页）。

### 4.6 官方模型查询的 wsid 硬校验例外

`search_hunyuan_official_models_by_train` 的 `wsid` 是**硬校验例外**：仅在用户问"官方/可训练/支持 SFT-DPO-LORA/模板候选"时使用；不要把普通"有权限模型/名称搜索"路由到此工具。wsid 规则：用户已提供或上下文已注入 → 直接用；未提供 → 通过 `scripts/tool_manual.py list_user_workspaces` 确认参数后调用，拿到有权限空间列表展示给用户选择；**严禁**以 `wsid=0`、空值或推测值调用。

### 4.7 严禁

- ❌ 把"模型格式转换（HF / Mcore）"当成本 skill 功能（→ `model-convert-skill`）。
- ❌ 用空值/猜测值调用任何工具，或猜测枚举类参数取值（先 `list_hunyuan_models_platform_enums`）。
- ❌ 未做二次确认就执行 `delete_*` 等不可逆写操作。
- ❌ 跨模块路由——用户问"格式转换 / 训练任务 / 资源配额"时立即跳出本 skill。
- ❌ 用户给参考卡片 + 新路径要求"创建新模型卡片"时，因为新路径含 `ckpt/iter_` 就试图走 `release_hunyuan_training_checkpoint_as_model` 或 `query_hunyuan_train_checkpoint_list`——正确做法是 `clone_hunyuan_models_card`。

### 4.8 失败处理

写入操作失败 → 透传错误原因，**不自动重试**；权限类错误（403）告知用户需找空间管理员；预热相关错误（如 `该格式模型暂不支持预热`、`xx地域暂不支持预热`）直接透传，**不要在客户端预判 HF 路径/格式后拒绝调用**；401 提示 token 失效。克隆路径错误（目录为空/不存在）→ 立即停止告知用户，**不得**调用 `list_storage_dir` 等 storage 域工具检查路径、反复重试 clone 或写脚本绕过。

### 4.9 搜索/详情交互补充

- 用户提供关键词 → 直接调用一次 `search_hunyuan_models_cards`（`is_show_all=true`，未指定分页用 `page=1, page_size=20`）；`has_more=true` 只提示可翻页。一次搜索未命中精确对象，不要把长名称拆成多个子词分别搜索，应报告限制并请用户提供 model_id 或更精确名称。搜索是全文模糊匹配，**不支持按创建日期过滤**。
- 用户未提供关键词："我有哪些模型"→`keyword=""`, `is_my_model=true`, `is_show_all=false`；"我创建的"→`is_my_create=true`, `is_show_all=false`；"所有模型"→`keyword=""`, `is_show_all=true`；未明确范围→默认 `keyword=""`, `is_show_all=true`。
- 模型详情：直接提供 ID → `get_hunyuan_models_card_detail`；只给名称 → 先 `search_hunyuan_models_cards` 拿 ID；多个匹配 → 询问用户选择。
- 常见错误场景：搜索无结果 → 返回"未找到匹配的模型"；模型 ID 不存在 → "未找到指定模型"；权限不足 → "权限不足，请检查认证信息"。
