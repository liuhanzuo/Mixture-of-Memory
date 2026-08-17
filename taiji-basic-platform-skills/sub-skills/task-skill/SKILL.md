---
name: task-skill
description: 太极平台训练任务子 skill —— 通过 MCP 协议管理混元训练平台的训练任务（taskId 以 basic_train_ 或 finetuning_ 开头，含模版化训练、自定义训练与模型开发），覆盖任务列表/详情/配置文件查询、分享权限、启动/停止、产出（checkpoint）、克隆任务、异常事件、项目标签等。当用户提及"训练任务 / 我的任务 / 任务状态 / 任务进度 / 启动任务 / 停止任务 / kill 任务 / 任务产出 / checkpoint / 克隆任务 / 自定义训练 / 模版化训练 / 模型开发 / 基模训练"等关键词时，应使用本 skill。
version: 3.0.0
author: taiji-team
---

# Task Skill（训练任务）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：混元训练平台的训练任务，涵盖三类任务——

| taskId 前缀 | 任务类型 | task_category 值 |
|------------|---------|-----------------|
| `finetuning_` | 模版化训练 | `finetuning` |
| `basic_train_` | 自定义训练 | `custom_training` |
| `basic_train_` | 模型开发 | `model_dev` |

覆盖任务列表/详情/配置文件查询、分享权限、启动/停止、产出（checkpoint）、克隆任务、异常事件、项目标签。

**⚠️ 排除**：
- 任务实例列表 / Pod 列表（`query_hunyuan_train_instance_list` / `query_hunyuan_train_pod_list`）→ 走 `instance-skill`
- 训练指标查询（`list_hunyuan_train_available_metrics` / `query_hunyuan_train_metric_text` / `query_hunyuan_train_metric_chart`）→ **不在本 skill**，走 `metric-skill`，**仅限自定义训练任务**
- 模型转换 / SFT 转 bin → `model-convert-skill`
- 应用组 / 资源使用 / 卡时配额 / 任务加白 → `resource-mgmt-skill`
- 模型服务 / 服务组 / 实例日志 / 扩缩容 → `service-deploy-skill`

> 缺 `wsid` / 工作空间上下文时可直调 `list_user_workspaces`（通过 `scripts/tool_manual.py list_user_workspaces` 查看参数），无需提示"去加载 workspace-skill"。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **误入场景**：用户转向"任务实例/Pod/日志、训练指标、模型转换、资源配额"等非训练任务意图时，立即退出。
> - **写操作**：启动、停止、分享权限、克隆等。

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
| `query_hunyuan_train_task_list` | 查询训练任务列表（模版化训练/自定义训练/模型开发，支持状态/创建人/应用组/GPU/标签等筛选） | |
| `get_hunyuan_train_task_detail` | 查询任务详情（可指定 instance_id 查某次实例的快照配置） | |
| `get_hunyuan_train_task_config_file` | 查看任务配置文件列表 / 指定文件内容 | |
| `query_hunyuan_train_checkpoint_list` | 查询任务产出列表（Checkpoint，支持模版化训练、自定义训练） | |
| `update_hunyuan_train_task_permission` | 分享 / 移除训练任务的管理员权限（add / remove） | ✍️ |
| `start_hunyuan_train_task` | 运行（启动）训练任务，创建一个新的训练实例并执行 | ✍️ |
| `stop_hunyuan_train_task` | 停止训练任务，自动停止最新运行中的实例 | ✍️ |
| `clone_hunyuan_train_task` | 克隆自定义训练任务并可选修改 GPU 资源、启动命令、配置文件参数、应用组/地域/GPU卡型、基座模型、镜像；支持复制原任务的伴生评估配置 | ✍️ |
| `clone_hunyuan_train_finetuning_task` | 克隆模版化训练任务并可选修改训练数据、超参数、依赖文件 | ✍️ |
| `query_hunyuan_train_failure_event_list` | 查询任务的异常事件（节点故障/GPU 错误，仅自定义训练） | |
| `list_hunyuan_train_task_tag_enums` | 查询任务的项目标签枚举列表（仅自定义训练） | |

> ⚠️ 以下工具已移出本 skill，请加载 `instance-skill`：任务实例列表 → `query_hunyuan_train_instance_list`；Pod 列表 → `query_hunyuan_train_pod_list`；训练日志 → `query_hunyuan_train_instance_logs`。训练指标工具**不在本 skill**，请加载 `metric-skill`，**仅限自定义训练任务**。

### 能力差异矩阵

| 工具 | 模版化训练 | 自定义训练 | 模型开发 |
|------|:--------:|:--------:|:------:|
| `query_hunyuan_train_task_list` | ✅ | ✅ | ✅ |
| `get_hunyuan_train_task_detail` | ✅ | ✅ | ✅ |
| `get_hunyuan_train_task_config_file` | ✅ | ✅ | ✅ |
| `update_hunyuan_train_task_permission` | ✅ | ✅ | ✅ |
| `start_hunyuan_train_task` | ✅ | ✅ | ✅ |
| `stop_hunyuan_train_task` | ✅ | ✅ | ✅ |
| `query_hunyuan_train_checkpoint_list` | ✅ | ✅ | ❌ |
| `clone_hunyuan_train_task` | ❌ | ✅ | ✅ |
| `clone_hunyuan_train_finetuning_task` | ✅ | ❌ | ❌ |
| `query_hunyuan_train_failure_event_list` | ❌ | ✅ | ❌ |
| `list_hunyuan_train_task_tag_enums` | ❌ | ✅ | ❌ |

### 跨模块速查工具

直达其他模块，通过 `scripts/tool_manual.py <工具名>` 确认参数后调用。

| 跨模块工具 | 用途 |
|---|---|
| `list_user_workspaces` | 直调工作空间查询，缺 `wsid` 时列出可访问空间供用户选定 |
| `query_hunyuan_train_instance_logs` | 查看训练日志 |

---

## 3. 快速路由表（多工具流程编排）

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。
>
> 仅列多工具串联流程。单工具场景已在 §2 用途列覆盖。命中时按最小链路执行，避免冗余调用。

| 场景 | 流程 | 流转条件 |
|------|------|----------|
| 任务失败诊断 | ① `get_hunyuan_train_task_detail` → ② `query_hunyuan_train_failure_event_list` → ③ 可选 `query_hunyuan_train_instance_logs` 查日志补充报错上下文 | 用户想知道任务为什么失败 |
| 任务克隆 | `clone_hunyuan_train_task` → 如需启动则 `start_hunyuan_train_task` | 用户说"复制/克隆"；clone 返回新 task_id 后按需启动 |
| 克隆时保留伴生评估 | `clone_hunyuan_train_task(copy_evaluation_config=true)`，若用户指定了源实例 ID 则加传 `copy_evaluation_source_instance_id` | 用户说"保留评估/伴生评估/自动评估"；不要切到 evaluation-skill |
| 克隆+伴生+启动 | ① `clone_hunyuan_train_task(copy_evaluation_config=true)` → ② `start_hunyuan_train_task` | 用户说"克隆并启动+保留评估"；不先查 detail，写操作无豁免时先确认 |
| 任务配置对比 | ① `get_hunyuan_train_task_detail` ×2 → ② `get_hunyuan_train_task_config_file(file_name="custom.toml")` ×2。**最多 4 次调用**，不追调 failure_event/instance/logs/SwanLab/checkpoint | 用户给两个 task_id 或 URL 做实验/配置对比 |
| 按任务名称对比配置 | ① `query_hunyuan_train_task_list(keyword=<名称>)` → ② `get_hunyuan_train_task_detail` → ③ `get_hunyuan_train_task_config_file(file_name="custom.toml")`，每个名称各走一次，**不先查无 keyword 全量列表** | 用户说"比较 A 和 B 的配置" |
| 修改学习率 | ① `get_hunyuan_train_task_config_file`（**不传 `file_name`**，只看配置文件列表）→ ② `clone_hunyuan_train_task`，`config_modifications` 用 `[{"action":"update","path":"TRAINING_ARGS","key":"lr","value":"<值>"}]` | 用户说"改学习率为 X"；不调 detail，不读文件内容，克隆后不验证 |
| 训练数据换成路径 | `clone_hunyuan_train_task`，`config_modifications` 使用 `[{"action":"update","path":"training","key":"dataset_path","value":"<path>"}]` | 用户说"训练数据换成 X"；不传 task_name/description 避免重名冲突 |
| 启动再停止 | ① `start_hunyuan_train_task(task_id)` → ② `stop_hunyuan_train_task(task_id)` → ③ `get_hunyuan_train_task_detail(wsid, task_id)` 确认终态。**start/stop 不传 wsid** | 用户说"启动起来再停下来/先跑再停"；不要 stop 成功后省略 detail |
| 复制最新任务 | `query_hunyuan_train_task_list(wsid, task_category=<类别>, query_type="my_created", page_size=1, order_by="created_at", order_type="DESC")` → `clone_hunyuan_train_task` | 用户问"复制我名下最新 XX 任务"；第一步不用 query_type="all" |
| checkpoint 查询 | `query_hunyuan_train_checkpoint_list(task_id)` → 可选切 `model-manage-skill`。**不传 wsid** | 查训练产出/ckpt；可选传 instance_id |
| 从配置推断基模 | `get_hunyuan_train_task_detail` → `get_hunyuan_train_task_config_file` → `search_hunyuan_models_cards` | 需要从配置文件追溯基座模型 |
| 按创建人/关键词查列表 | `query_hunyuan_train_task_list` 默认**只查第一页**，展示 total/has_more 引导翻页；**禁止主动追页**。"最近一周"只传 `created_at+order_by+order_type`，不传 page/page_size | 用户说"最近的任务/我创建的/XX 关键词" |

---

## 4. 模块注意事项

### 4.1 进入判定

- **何时进入**：用户问"训练任务 / 我的任务 / 某人的任务 / 任务状态 / 启动 / 停止 / 任务产出 / ckpt / 克隆任务 / 模版化训练 / 自定义训练 / 模型开发"等任一意图，或 taskId 以 `basic_train_` 或 `finetuning_` 开头。
- **泛称"训练任务"未分类 → 先追问**：用户只说"训练任务/在跑的训练任务"但未明确模版化/自定义/模型开发/全部时，必须先追问 task_category，不调任何工具。不要把泛称默认等同"全部"；只有用户明确说"全部"才分别查三类。
- **范围声明**：本 skill 处理 `basic_train_*`（自定义训练 + 模型开发）和 `finetuning_*`（模版化训练）前缀任务。任务实例/Pod/日志相关操作已移至 `instance-skill`。
- raw IP / redirect / base64 / K8s service URL → 不是标准任务链接，不 base64 解码、不猜测 task_id，请用户提供标准链接或 task_id。

### 4.2 通用参数规则

- **wsid 必填工具**：`query_hunyuan_train_task_list`、`get_hunyuan_train_task_detail`、`get_hunyuan_train_task_config_file`、`clone_hunyuan_train_task`、`clone_hunyuan_train_finetuning_task`、`list_hunyuan_train_task_tag_enums`。
- **分页** `page` 从 **1** 开始。
- **训练指标**已移至 `metric-skill`（仅限自定义训练）；**实例/Pod/日志**已移至 `instance-skill`。

### 4.3 查列表的强制三要素

调 `query_hunyuan_train_task_list` **必须传** `task_category` + `wsid` + `query_type`，用户没说时**必须追问**"工作空间 ID + 想查哪类训练任务"。不要把泛称"训练任务"默认等同"全部"。

- **wsid**：用户已提供直接使用；未提供**必须追问**（严禁用 `wsid=0` 或空值调用）。
- **task_category**：未明确**必须追问**；回答"全部"时并行调多次。
- **query_type**：默认 `query_type="all"`；"我的"→`my_created`；"我有权限的"→`my_permission`。

### 4.4 任务名定位规则

1. 用户说"任务状态"但没给 task_id → 先列任务再让用户挑，**严禁**猜测 task_id。
2. 写入类工具调用前向用户复述操作对象 + 影响，得到确认后才执行；若用户明确说"不需要确认 / 直接执行"，可视为已授权。
3. 列表/产出返回的数据**必须按原始顺序展示**，严禁重排/筛选/截断；默认只取第一页。
4. **任务名定位只查一次**：用户给的是任务名/非 `basic_train_*` 字符串时，先按名称和 `wsid` 定位一次；命中后复用返回的 task_id，不要把名称硬当 task_id 调详情。

### 4.5 克隆与伴生评估配置

- 用户说"克隆并保留伴生评估"时，调用 `clone_hunyuan_train_task` 并传 `copy_evaluation_config=true`，不要切到 `evaluation-skill`。
- 详细参数规则（`config_modifications` 格式、镜像更换、应用组修改流程等）见 `clone_hunyuan_train_task` 的 api 文档。

### 4.7 严禁

- ① 把"模型转换 / SFT 转 bin"任务当成训练任务（→ `model-convert-skill`）；② 把"任务实例 / Pod 列表 / 训练日志"在本 skill 处理（→ `instance-skill`）；③ 跨用户调 `stop_hunyuan_train_task`；④ 在没有 wsid 的情况下调用需要 wsid 的工具；⑤ 对 raw IP / redirect / base64 / K8s service URL 做解码或猜测式搜索。

### 4.8 失败处理

写入失败 → 透传错误原因，不自动重试；权限错误（403）告知用户需要任务 owner 授权或调 `update_hunyuan_train_task_permission`；网络/超时如实告知，不重试、不切换工具。

### 4.9 taskId 格式与展示规范

- `finetuning_` 开头为模版化训练；`basic_train_` 开头为自定义训练**或**模型开发，两者 taskId 格式相同，靠 `task_list` 返回的 `task_category` 字段区分（`custom_training` / `model_dev`）。
- 分页 `page` 从 1 开始。
- 任务列表**必须使用 Markdown 表格**展示，禁止嵌套列表/缩进子列表。推荐列：`序号 | 任务名称 | 任务 ID | 描述 | 资源 | 状态 | 创建时间`。资源列合并 "N机M卡 卡类型"（finetuning 无资源字段则显示 `-`）；状态列用 `status_text`（运行中加 ✅、已终止加 ⏹、已完成显示"结束(成功)"）；创建时间只显示日期部分；表格前显示汇总行，表格后先空一行加 `---` 再追加提示；**任务 ID 必须展示完整，严禁截断**。
- 运行中（TRAINING_RUNNING）任务的 `items` 中**没有** `status` / `status_text` / `instance_id` 字段；仅已完成/已终止/失败的实例才携带。finetuning 类型任务不返回 `machine_count` / `gpu_per_machine` / `gpu_type` / `train_framework` 等资源字段。
- `query_hunyuan_train_checkpoint_list` 返回：`released` 为 **boolean**（`true`/`false`），不是字符串；`file_size` 为整数（字节），不是人类可读字符串。
- 每次返回结果后按场景追加引导性问题；Markdown 表格后必须空一行 + 加 `---` 分隔线再写引导文字。引导内容根据任务类型智能裁剪（模型开发任务不要引导"查看训练指标"和"查看产出列表"）。

### 4.6 结果引导规则

每次返回结果后，根据当前场景追加引导性问题，引导用户继续操作。

| 工具 | 推荐引导 |
|------|----------|
| `query_hunyuan_train_task_list` | "查看某任务详情？/ 训练指标？/ 克隆并修改？" |
| `query_hunyuan_train_checkpoint_list` | "查看任务详情？/ 启动新训练？" |
| `get_hunyuan_train_task_config_file`（列表） | "查看某个配置文件内容？请提供文件名。" |
| `get_hunyuan_train_task_config_file`（内容） | "查看其他配置？/ 克隆并修改？" |
| `start_hunyuan_train_task` | "任务已启动。查看实例/训练指标？" |
| `stop_hunyuan_train_task` | "任务已停止。重新启动 / 查看产出？" |
| `clone_hunyuan_train_task` | "新任务已创建。启动 / 查看详情？" |
| `query_hunyuan_train_failure_event_list` | "查看训练日志排查原因？（可直调 `query_hunyuan_train_instance_logs`，见 helper_api.md）" |

> 引导内容根据任务类型裁剪（如模型开发不引导"训练指标"）。

### 4.10 能力边界

- **训完自动导出 mcore**：本模块无法持续监测训练完成并自动触发模型转换。用户问"训完自动导出 mcore"时，说明当前能力边界，不要立即调 model-convert-skill 的转换工具。
- **任务配置读取**：`get_hunyuan_train_task_config_file` 没有 `action` 参数，不要传 `action=list/read`。每个 task_id 只调一次配置列表获取全部配置文件清单；读单个文件时才传 `file_name`。
