---
name: posttrain-data-skill
description: 太极平台后训练数据子 skill —— 通过 MCP 协议管理后训练数据全链路（Topic → 数据集 → 数据版本 TopicData 的创建、单条查询与分页查询、状态轮询）、后训练数据班车查询、查班车上的 Topic 数据明细与「上班车」（把 TopicData 加入已有班车），并对后训练数据做质量检测（触发质检、查询质检结果与状态、预览/下载不合格样本）。当用户提及"质检 / 质检 N / 查质检状态 / 质检跑完了吗 / inspection / inspection_id、底线质检 / 内容质检 / 不合格数据 / 不合格样本、后训练数据质检 / 数据质检、后训练 Topic / TopicData、上班车 / 数据班车 / 后训练班车 / goShuttle、班车上有哪些 topic 数据 / 班车明细、查后训练数据列表"等关键词时，应使用本 skill。边界排除项见 §0。
version: 3.0.0
author: taiji-team
---

# Posttrain Data Skill（后训练数据：链路管理 + 数据质检）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：三类**后训练数据**意图，对应三份 reference 文档：

- **后训练数据质检**（对 TopicData 触发质检、查询质检结果、预览/下载不合格样本）→ `references/posttrain_inspection_api.md`
- **后训练数据链路管理**（Topic → 数据集 → 数据版本 TopicData 的创建与异步状态轮询）→ `references/posttrain_topic_data_api.md`
- **后训练班车 / 上班车**（查询班车、查班车上的 Topic 数据明细、把 TopicData 加入已有班车）→ `references/posttrain_shuttle_api.md`

**⚠️ 排除（不在本 skill）**：
- 预训练 Topic / 预训练数据集（query 含"预训练主题/预训练 topic/预训练数据集/pretrain/登记预训练数据/STAGE_ONE/ANNEALING/STAGE_TWO"，或需要调 `create_hunyuan_data_pretrain_topic` / `create_hunyuan_data_pretrain_dataset` / `query_hunyuan_data_pretrain_*` 等带 `pretrain` 字样的工具）→ `data-processing-skill`（两套 Topic/数据集体系完全独立，**严禁**用后训练工具处理预训练数据资产登记）
- 普通 ceph2ceph 数据导出 / 复制 / 搬迁、外租卡（CUDOFS）拷贝、HDFS parquet 转 bin → `data-processing-skill`
- SFT / 后训练数据转 bin（tokenizer / seq_len / 跨地域 pipeline）→ `data-processing-skill`
- 预训练融合任务（shuttle-task / 融合 Pipeline / phase_status）→ `data-processing-skill`（**不要**与本 skill「上班车」混淆）

> 缺 `wsid` / 工作空间上下文时可直调 `list_user_workspaces`（通过 `scripts/tool_manual.py list_user_workspaces` 查看参数），无需提示"去加载 workspace-skill"。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **误入立即退出（禁止反复试探）**：进入本模块后，若发现请求核心对象不属于本模块范围，**立即**切换到正确的子 skill。常见误入场景：预训练 Topic/数据集 → `data-processing-skill`（不得用后训练工具兜底）；预训练融合任务、数据导出/拷贝、转 bin → `data-processing-skill`；评测 → `evaluation-skill`。
> - **写操作**：创建、注册、上班车等。

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

### A. 后训练数据质检（`references/posttrain_inspection_api.md`，4 个）

| 工具 | 用途 | 写操作 |
|---|---|---|
| `create_hunyuan_data_quality_inspection` | 对一条后训练 TopicData（`topic_data_id`）触发质检 | ✍️ |
| `get_hunyuan_data_quality_inspection` | 查询质检任务详情与状态（合格/不合格行数等） | |
| `preview_hunyuan_data_quality_inspection_data` | 预览质检发现的不合格样本（采样 parquet 原始内容） | |
| `get_hunyuan_data_quality_inspection_download_url` | 获取不合格数据下载说明（URL + curl + Ceph 绝对路径） | |

### B. 后训练数据链路管理（`references/posttrain_topic_data_api.md`，5 个）

| 工具 | 用途 | 写操作 |
|---|---|---|
| `create_hunyuan_data_topic` | 创建后训练 Topic（链路最顶层容器） | ✍️ |
| `create_hunyuan_data_topic_dataset` | 在某 Topic 下创建数据集（承载 stage / thinking_type） | ✍️ |
| `create_hunyuan_data_topic_data` | 在某数据集下注册一个数据版本（TopicData） | ✍️ |
| `get_hunyuan_data_topic_data` | 按 `topic_data_id` 查询单条 TopicData 详情与异步搬运状态 | |
| `query_hunyuan_data_topic_datas` | 按 `wsid` 分页查询 TopicData 列表，支持 name/creator/status/创建时间范围筛选 | |

### C. 后训练班车 / 上班车（`references/posttrain_shuttle_api.md`，3 个）

| 工具 | 用途 | 写操作 |
|---|---|---|
| `query_hunyuan_data_posttrain_shuttles` | 分页查询后训练数据班车（可按 stage / thinking_type 过滤） | |
| `query_hunyuan_data_topic_data_tasks` | 查**某班车上已上车的 Topic 数据明细**（必填 `wsid` + `shuttle_id`；一行 = 一条 TopicData，`items[].id` 即 `topic_data_task_id`，`items[].data_source_path` 是注册时的原始文件路径） | |
| `create_hunyuan_data_topic_data_task` | **上班车**：把已有 TopicData 加入已有班车（**须显式传** `config.type=APPEND`，对齐前端默认） | ✍️ |

### 跨模块速查工具

直达其他模块，通过 `scripts/tool_manual.py <工具名>` 确认参数后调用。

| 跨模块工具 | 用途 |
|---|---|
| `list_user_workspaces` | 直调工作空间查询，缺 `wsid` 时列出可访问空间供用户选定 |

---

## 3. 快速路由表（多工具流程编排）

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。
>
> 仅列多工具串联流程。单工具场景已在 §2 用途列覆盖。**先判别"该读哪份 reference"（最高优先级）：**

- **质检 / 不合格 / 质量检测 / 预览不合格 / 下载不合格 / 查质检状态** → 读 `references/posttrain_inspection_api.md`。
- **建 Topic / 建数据集 / 注册数据版本 / 登记 jsonl / 查单条 TopicData 状态 / 分页查询数据版本、stage、thinking_type、链路依赖** → 读 `references/posttrain_topic_data_api.md`。
- **上班车 / 数据班车 / 后训练班车 / 把 topic_data 上到班车 / 有哪些班车 / 班车上有哪些 topic 数据 / 看班车明细** → 读 `references/posttrain_shuttle_api.md`。
- **预训练融合任务 / shuttle-task / 融合 Pipeline / phase_status** → **不在本 skill**，请加载 `data-processing-skill`。
- **把数据搬 / 拷贝 / 同步到 ceph、外租卡拷贝、parquet 转 bin** → **不在本 skill**，请加载 `data-processing-skill`。
- **把数据转 bin / 分词 / seq_len / tokenizer** → **不在本 skill**，请加载 `data-processing-skill`。

| 场景 | 流程 | 流转条件 |
|------|------|----------|
| 从零到质检的完整链路 | ① 确认是否已有 `topic_data_id`（有则先 `get_hunyuan_data_topic_data` 看 V2 质检进度；无则补全链路）→ ② `create_hunyuan_data_topic` → ③ `create_hunyuan_data_topic_dataset` → ④ `create_hunyuan_data_topic_data`（SFT 默认 `include_baseline=true`）→ ⑤ `get_hunyuan_data_topic_data` 轮询到 SUCCEEDED | 链路严格自上而下；创建时已开开关则不要再调 `create_hunyuan_data_quality_inspection` |
| topic_data 下载链路（三步，⛔ 不许跳步） | ① `get_hunyuan_data_topic_data(topic_data_id, wsid)` 读 `inspectionId` → ② `get_hunyuan_data_quality_inspection(inspection_id)` 确认 `SUCCEEDED` 且看清可用 `file_type` → ③ `get_hunyuan_data_quality_inspection_download_url` | **严禁跳过 ②** 直接从 ① 跳到 ③；不要用 `create_hunyuan_data_quality_inspection` 代替查询已有质检 |
| 上班车（缺 `topic_data_id`） | ① 先 `query_hunyuan_data_topic_datas`（`wsid` + 名称/创建人/时间线索）列候选给用户确认 → ② `get_hunyuan_data_topic_data` 确认 SUCCEEDED → ③ `query_hunyuan_data_posttrain_shuttles` 选班车 → ④ `create_hunyuan_data_topic_data_task`（`config.type=APPEND`） | 缺 ID 先查再问，不要一句"请提供 ID"了事；但不得从候选里自行挑一个就去执行写操作 |
| 上班车（已给 `topic_data_id`+`shuttle_id`） | `get_hunyuan_data_topic_data` 确认 SUCCEEDED（V2 还须 md5）→ `create_hunyuan_data_topic_data_task` | 先查重复上车（`shuttle_info_list[]` 或 `query_hunyuan_data_topic_data_tasks` 的 `items[].data_id`）；已在目标班车则不要 create |

---

## 4. 模块注意事项

### 4.1 不同 ID 不可混用

`topic_id` / `dataset_id` / `topic_data_id` / `inspection_id` / `shuttle_id` / `shuttle_task_id` / `topic_data_task_id` 是**不同 ID**，不可混用。模棱两可时严禁凭猜执行写操作，必须先确认。状态字段取值、参数类型（`topic_data_id` 为正整数等）、时间格式等细节见各 api.md 参数表。

只读查询（query/get 类）用于帮用户消歧是被鼓励的：用户不记得 ID 时，按 `wsid` + 名称/创建人/时间等线索查出候选摆给用户确认，不要只回一句「请提供 ID」。红线：不得从候选里自行挑一个就去执行写操作。

`create_hunyuan_data_topic_data` 的 `(dataset_id, version)` 重复 → 报「topic数据已经存在」；用户已指定 `version` 时必须停止，不能擅自改 `v2_日期` 继续注册。确认创建结果只能用只读工具，不能靠再 create。

### 4.2 后训练数据质检规则

- **质检 ≠ 评测**：用户说「质检 N / inspection N」时，数字是 `inspection_id`，第一步必须调 `get_hunyuan_data_quality_inspection`，严禁先去 `get_taiji_eval_task_detail` 等评测工具。判据：`质检 / inspection` + 数字 → 本模块；`评测 / 评估 / 得分 / Insight / Arena` → evaluation-skill。二者同时出现才追问。
- **ID 语义强约束**：用户说"质检 274 / inspection 274"时，数字按 `inspection_id` 处理，只能走质检三工具；严禁当 `topic_data_id` 调 `get_hunyuan_data_topic_data`。用户明确说 `topic_data 591` 时才走 TopicData 查询。
- **V2 页面列说明**：前端【底线质检】【内容质检】列展示的是创建 TopicData 时 `include_baseline`/`include_content` 发起的 V2 AIData 批量任务。`create_hunyuan_data_quality_inspection` 走旧 V1 协议，不能补开这两列。新注册数据做质检必须走 `create_hunyuan_data_topic_data`（见 4.3）。
- **topic_data 下载链路（三步，不可跳步）**：用户给 `topic_data_id` 问"不合格数据怎么下载"时，必须依次：① `get_hunyuan_data_topic_data` 读 `inspectionId` → ② `get_hunyuan_data_quality_inspection` 确认 `SUCCEEDED` 且看清 `file_type` → ③ `get_hunyuan_data_quality_inspection_download_url`。严禁跳过 ②。
- **下载/预览的调用预算和文件类型**：`download_url` 不真实下载，只返回 URL + curl + Ceph 路径；文件类型只能是 `file_type`（大写 `JSON`/`PARQUET`/`SAMPLED_PARQUET`）。用户只说"预览/看看"时只做 `preview`，不要连带调 `download_url`。
- **只读查询禁止重复**：`get_hunyuan_data_quality_inspection(同一 inspection_id)` 在一轮里只调 1 次；返回拿到后直接作答，禁止为补字段/换排版再调。

### 4.3 后训练数据链路管理规则

- **链路自上而下**：`create_hunyuan_data_topic` → `create_hunyuan_data_topic_dataset` → `create_hunyuan_data_topic_data`。上一步产出 ID 是下一步必填参数。缺 ID 时按需自助补齐。
- **单条查 vs 列表查**：已知 `topic_data_id` → 用 `get_hunyuan_data_topic_data`；只知道 `wsid` → 用 `query_hunyuan_data_topic_datas` 分页查询。列表结果按接口原始顺序展示。
- **⭐ 质检开关必须对齐前端（创建时唯一窗口）**：V2 TopicData 的底线/内容质检**只能在创建时**通过 `include_baseline`/`include_content` 发起，后续无法补开。Agent 必须按前端默认值传参（不得照抄协议 `false`）：
  - SFT（未明确说"不做质检"）：`include_baseline=true`，`include_content=false`
  - GRPO：默认都不开（用户明示要开再改）
  - 用户明示要内容质检 → `include_content=true`（可与底线同时开）
  - **禁止**指望创建后再调 `create_hunyuan_data_quality_inspection` 填满页面【底线质检】【内容质检】列
- **⛔ create 类只调一次**：`create_*` 类每轮只允许成功 1 次。Topic 的 `key`/`name` 全局唯一，重复建不可逆。返回直接用，严禁"再建一次确认下"。只有明确失败（`code!=0` 且原因可修正）才允许改参数后重试一次。
- **轮询节奏**：`get_hunyuan_data_topic_data` 轮询间隔 30 秒~数分钟，不高频连发；超过 5~10 次仍未 SUCCEEDED 则报告进度建议稍后再查。
- **关键字段约束**：`stage`（SFT/GRPO/DPO/REWARD）和 `thinking_type`（FAST_THINKING/SLOW_THINKING/HOLISTIC_THINKING，传枚举名）挂在数据集层，TopicData 自动继承。`source_path` 须为单文件 JSONL（非目录），`(dataset_id, version)` 唯一。
- **参数齐备 → 立即调用**；缺参才追问，不"调一次看报错"。

### 4.4 后训练班车 / 上班车规则

- **与预训练融合强区分**：`融合任务 / shuttle-task / phase_status` → 转 `data-processing-skill`；只有「上班车 / 数据班车 / 后训练班车」才用本模块。
- **查班车列表 ≠ 查班车明细**：`query_hunyuan_data_posttrain_shuttles` 返回班车本身；问「班车上有哪些数据」用 `query_hunyuan_data_topic_data_tasks`（必填 `wsid`+`shuttle_id`）。
- **空结果处理**：`query_hunyuan_data_posttrain_shuttles` 或 `query_hunyuan_data_topic_data_tasks` 返回空列表时，如实告知用户「当前无匹配结果」，不要自行放宽过滤条件重查或编造数据。
- **数据来源路径**：`items[].data_source_path` 是注册时的原始文件路径；不要拿 `items[].storage_path` 顶替。
- **上班车前**：先确认 TopicData `SUCCEEDED`（V2 还须 md5），缺 `shuttle_id` 时用 stage/thinking_type 查班车列表选。
- **APPEND 与重复上车**：传 `config.type=APPEND`（前端默认，不可省）。先查是否已在目标班车（`shuttle_info_list[]` 或 `query_hunyuan_data_topic_data_tasks` 的 `items[].data_id`），已在就别 create（会 50001）。
- **reason 必填**：用户未给则追问。
- **不支持创建班车**：引导去太极页面。
- **查班车同参数只调一次**：拿到返回直接整理作答，不为确认/排版重复调。

### 4.5 常见问题速查

- `stage`/`thinking_type` 挂在数据集层，需要不同值应新建 Dataset。
- `ThinkingType` 枚举传全名（`FAST_THINKING`/`SLOW_THINKING`/`HOLISTIC_THINKING`，不缩写）。
- Topic `key` 不支持修改，建错需删后重建（MCP 暂未封装 DELETE）。
- 已支持 `query_hunyuan_data_topic_datas`（分页）和 `get_hunyuan_data_topic_data`（单条）；暂不支持按名称反查 Topic/Dataset 列表。
