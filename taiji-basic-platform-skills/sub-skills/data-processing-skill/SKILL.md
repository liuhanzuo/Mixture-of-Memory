---
name: data-processing-skill
description: 太极平台数据处理子 skill —— 文件存储（ceph / nitrofs(hifs)）之间的数据导出/复制/搬迁/同步（ceph→ceph、ceph↔nitrofs(hifs)、nitrofs(hifs)→nitrofs(hifs)）、外租卡存储（CUDOFS）数据拷贝、HDFS parquet 预训练数据转 bin、SFT/后训练数据转 bin（含南京地域强校验 + 跨地域自动搬运 Pipeline）、预训练融合任务（shuttle-task / 融合 Pipeline）查询、预训练 Topic / 数据集的创建与查询（预训练数据资产登记）、HDFS ↔ Ceph 双向数据搬运（一端为 hdfs:// 路径）、Bin 分片合并（把 ceph 上已有的一批 bin/idx 分片合并成少量大文件）。当用户提及"数据导出 / 数据拷贝 / 数据搬迁 / 数据同步 / ceph2ceph / 文件存储拷贝 / nitrofs / hifs / ceph 拷到 nitrofs / nitrofs 拷到 ceph / taijifs 路径拷贝 / 外租卡拷贝 / cudofs / wz 拷贝 / 预训练 parquet 转 bin / hdfs 转 bin / SFT 转 bin / 后训练转 bin / tokenizer / 序列长度 / 跨地域转 bin / bin pipeline / 南京地域 / hdfs2ceph / ceph2hdfs / hdfs to ceph / ceph to hdfs / hdfs:// 路径搬运 / 搬运任务 / 搬运进度 / transfer task / 预训练融合任务 / 融合 Pipeline / shuttle task / 融合产物路径 / 预训练 topic / 预训练主题 / 预训练数据集 / 登记预训练数据 / bin 合并 / bin 分片合并 / 合并 bin / merge bin / mergebin / merge_bins / bin 文件太多 / bin 太碎"等关键词时，应使用本 skill。
version: 3.0.0
author: taiji-team
---

# Data Processing Skill（数据处理）

> 📂 脚本路径：`scripts/connect_mcp.py` | `scripts/tool_manual.py`

## 0. 边界说明

**本模块范围**：7 类数据处理意图，分域管理，各对应一份工具手册（`*_api.md`）或流程文档：

| 域 | 覆盖意图 | 工具手册 / 流程文档 | 子目录 |
|---|---|---|---|
| A | 文件存储（ceph / nitrofs(hifs)）互拷、数据导出、CUDOFS 外租卡↔ceph / 外租卡↔外租卡 | `references/transfer/data_processing_api.md` | transfer |
| B | 预训练数据转 bin（HDFS parquet 需应用组+集群 / CEPH json） | `references/bin/pretrain_parquet2bin_api.md` | bin |
| C | SFT / 后训练数据转 bin（两端都在南京的单步） | `references/sft/sft_conversion_api.md` | sft |
| D | 跨地域 SFT 转 bin 自动搬运 Pipeline（任一端不在南京，Agent 编排） | 流程文档 `references/sft/sft_bin_auto_transfer_flow.md` | sft |
| E | HDFS ↔ Ceph 双向搬运（源或目标一端为 `hdfs://` 路径） | `references/transfer/transfer_tasks_api.md` | transfer |
| F | 预训练融合任务（shuttle-task / 融合 Pipeline）查询（只读） | `references/pretrain/pretrain_shuttle_task_api.md` | pretrain |
| G | 预训练 Topic / 数据集创建与查询（预训练数据资产登记，`Workspace→Topic→Dataset` 三级） | `references/pretrain/pretrain_topic_dataset_api.md` | pretrain |
| H | Bin 分片合并（已有 bin/idx 分片 → 少量大文件，仅 Ceph→Ceph） | `references/bin/merge_bins_tasks_api.md` | bin |

> 📂 **references 目录结构**：`transfer/`（数据导出/拷贝/搬运）、`bin/`（转 bin + 合并）、`pretrain/`（预训练 topic/dataset/shuttle）、`sft/`（SFT 转 bin + pipeline）、`helper_api.md`（平铺）。直接按上表完整路径 Read，**不需要 ls**。

**⚠️ 排除**：
- 后训练数据质检 / 后训练 Topic 数据链路管理（`create_hunyuan_data_topic` / `create_hunyuan_data_topic_dataset` 等**无 `pretrain` 字样**的工具）属 `posttrain-data-skill`，不进入本模块，两套体系不要混用。

**跨模块边界**：
- 纯资源 / 应用组（不含预训练上下文）→ `resource-mgmt-skill`
- 训练任务 / 服务 / 模型等其他意图 → 由聚合 skill 路由表选择对应子 skill。

> 💡 **缺 wsid 时无需切换 skill**：本模块可直调 `list_user_workspaces`（通过 `scripts/tool_manual.py list_user_workspaces` 确认参数后调用），不要提示"去加载 workspace-skill"。

> 🔧 **本模块特有行为规则**（以下规则仅对本模块生效，不覆盖 §1 通用规则）：
> - **流程文档**：`sft_bin_auto_transfer_flow.md`（多步 SOP），§3 命中时先完整阅读再执行。
> - **URL 字段**：页面 URL 取工具返回的"智研链接"、`wedata_url`、`zhiyanUrl` 等字段原样展示。
> - **误入场景**：用户转向"任务/指标/评测/模型"等非数据处理意图时，立即退出。

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

| 域 | 工具 | 用途 | 写操作 |
|---|---|---|---|
| A | `create_hunyuan_data_export_task` | 创建**文件存储（ceph / nitrofs(hifs)）之间**的数据导出/复制/搬迁/同步任务（4 种方向 ceph→ceph / ceph→nitrofs / nitrofs→ceph / nitrofs→nitrofs **同一工具同一参数**；也支持按业务实体导出） | ✍️ |
| A | `get_hunyuan_data_export_task` | 查询导出任务状态/进度（传 `task_id` + `wsid`） | |
| A | `get_hunyuan_data_export_task_log` | 查看导出任务日志、失败原因、智研链接 | |
| A | `create_hunyuan_data_cudofs_copy_task` | 创建外租卡存储（CUDOFS）数据拷贝任务（外租卡↔ceph / 外租卡↔外租卡） | ✍️ |
| A | `get_hunyuan_data_cudofs_copy_task` | 查询外租卡拷贝任务状态/进度 | |
| A | `get_hunyuan_data_cudofs_copy_task_log` | 查看外租卡拷贝任务日志、失败原因、智研链接 | |
| B | `query_hunyuan_data_app_groups` | 列出可用应用组（parquet 模式第 1 步） | |
| B | `list_hunyuan_data_app_group_hdfs_quota` | 列出某应用组下 HDFS 集群/配额（parquet 模式第 2 步，取 `hdfsNn`） | |
| B | `query_hunyuan_data_compute_resources` | 列出平台公共计算资源（取 `resource_id`；转 bin / 合并场景**必传 `entity_type=PRETRAIN_DATA_CONVERSION`**，否则只返回南京资源；无 `wsid` 参数） | |
| B | `list_hunyuan_data_pretrain_operators` | 列出转 bin 算子候选（`operator` 必填；用户未给时调用，已给则跳过） | |
| B | `list_hunyuan_data_pretrain_tokenizer_branches` | Tokenizer 三级级联第 1 级（传 `wsid`） | |
| B | `list_hunyuan_data_pretrain_tokenizer_model_series` | 三级级联第 2 级（传 `wsid` + `branch`） | |
| B | `list_hunyuan_data_pretrain_tokenizer_train_stages` | 三级级联第 3 级（传 `wsid` + `branch` + `model_series`） | |
| B | `create_hunyuan_data_pretrain_conversion` | 预训练数据转 bin（`file_format=parquet`：HDFS parquet 需应用组+集群；`file_format=json`：CEPH json 无需。均需 operator + git_based_tokenizer。可带 `enable_merge_bins`+`merge_bins_count` 顺带合并碎 bin；`name` 未给时按当前时间传 `create_by_skill_<YYYYMMDD-HHMM>`） | ✍️ |
| B | `get_hunyuan_data_pretrain_conversion` | 查询预训练转 bin 任务状态（传 `task_id` + `wsid`；启用合并时另看 `merge_bins_status` / `merge_bins_output_path`） | |
| B | `query_hunyuan_data_pretrain_conversions` | 分页查询转 bin 任务列表（传 `wsid` + `page` + `page_size`，**不要传 `page_index`**） | |
| B | `retry_hunyuan_data_pretrain_conversion` | 重试失败的预训练转 bin 任务 | ✍️ |
| B | `delete_hunyuan_data_pretrain_conversion` | 删除预训练转 bin 任务（**危险操作，调用前需用户二次确认**） | ✍️ |
| C | `list_hunyuan_data_sft_tokenizers` | 列出可用的分词器 / tokenizer（**无入参，传 `{}`，不要传 `wsid`**） | |
| C | `create_hunyuan_data_sft_conversion` | 创建 SFT 数据转 bin 任务（南京地域强校验后调用） | ✍️ |
| C | `get_hunyuan_data_sft_conversion` | 查询转 bin 任务状态/进度；**bin 文件 size 在其返回的 `size` 字段** | |
| C | `retry_hunyuan_data_sft_conversion` | 重试失败的转 bin 任务（仅用户显式触发时调用） | ✍️ |
| E | `create_hunyuan_data_hdfs_to_ceph_transfer` | 创建 HDFS → Ceph 搬运任务（方向固定：源 `hdfs://`、目标 Ceph 目录） | ✍️ |
| E | `create_hunyuan_data_ceph_to_hdfs_transfer` | 创建 Ceph → HDFS 搬运任务（方向固定：源 Ceph 目录、目标 `hdfs://`） | ✍️ |
| E | `get_hunyuan_data_transfer_task` | 按 id 查询单个搬运任务状态/进度；失败时透传 `message` + `wedata_url` | |
| E | `query_hunyuan_data_transfer_tasks` | 分页查询工作空间下的搬运任务列表（支持 creator / keyword 过滤） | |
| F | `get_hunyuan_data_pretrain_shuttle_task` | 按 id 查询预训练融合任务（shuttle-task / 融合 Pipeline）状态、5 阶段进度（去重/融合/质检/转bin/同步）与融合产物路径（`storage_path`）。只需 `shuttle_task_id`，`wsid` 可选（传则校验归属） | |
| G | `create_hunyuan_data_pretrain_topic` | 创建预训练 Topic（主题），建数据集第 1 步（传 `wsid` + `name`） | ✍️ |
| G | `query_hunyuan_data_pretrain_topics` | 分页查询预训练 Topic 列表（按 name/owner 找 Topic 拿 `topic_id`） | |
| G | `create_hunyuan_data_pretrain_dataset` | 在某 Topic 下创建预训练数据集（必填 `wsid`+`key`+`topic_id`+`storage_path`+`stage_list`+`type`） | ✍️ |
| G | `get_hunyuan_data_pretrain_dataset` | 查询单个预训练数据集详情（传 `dataset_id`，`wsid` 可选） | |
| G | `query_hunyuan_data_pretrain_datasets` | 分页查询预训练数据集列表（按 `wsid` + 可选 topic_id/stage/file_type 等） | |
| H | `create_hunyuan_data_merge_bins_task` | 创建 Bin 分片合并任务（必填 `wsid`+`input_path`+`task_config.resource_config`；可选 `output_path`/`merge_bins_count`/`type`。`name` 未给时按当前时间传 `create_by_skill_<YYYYMMDD-HHMM>`） | ✍️ |
| H | `get_hunyuan_data_merge_bins_task` | 按 id 查询合并任务状态/进度；失败透传 `message`+`wedata_url`；跨地域看 `transfer_status` | |
| H | `query_hunyuan_data_merge_bins_tasks` | 分页查询工作空间下的合并任务列表（支持 creator / keyword / type 过滤） | |
| H | `retry_hunyuan_data_merge_bins_task` | 重试失败的合并任务（仅用户显式触发时调用） | ✍️ |

### 跨模块速查工具

直达其他模块，通过 `scripts/tool_manual.py <工具名>` 确认参数后调用。

| 跨模块工具 | 用途 |
|---|---|
| `list_user_workspaces` | 直调工作空间查询，缺 `wsid` 时列出可访问空间供用户选定 |

---

## 3. 快速路由表（多工具 Pipeline 编排 + 流程文档索引）

> 💡 涉及多工具的操作，一次用 `scripts/tool_manual.py <工具1> <工具2> ...` 批量获取多个工具说明，避免多次调用。

### 3.1 域判别口诀（先判域，再进流程）

| 用户说法 | 走哪个域 / 文档 |
|---|---|
| 文件存储互拷 / ceph2ceph / 把 A 路径数据放到 B（两端都是文件存储或外租卡） | 域 A `transfer/data_processing_api.md` |
| 外租卡拷贝 / cudofs / wz 拷贝 | 域 A（`create_hunyuan_data_cudofs_copy_task`，见 `transfer/data_processing_api.md`） |
| hdfs to/from ceph / hdfs:// 路径搬运 / transfer task | 域 E `transfer/transfer_tasks_api.md` |
| 预训练 / parquet 转 bin / json 转 bin / 应用组+集群转 bin | 域 B `bin/pretrain_parquet2bin_api.md` |
| SFT / 后训练转 bin / tokenizer / 跨地域转 bin / bin pipeline | 域 C/D（`sft/sft_conversion_api.md` / `sft/sft_bin_auto_transfer_flow.md`） |
| 融合任务 / shuttle / 融合 Pipeline 进度 / 产物路径 | 域 F `pretrain/pretrain_shuttle_task_api.md` |
| 预训练 topic / 预训练数据集 / 登记预训练数据 | 域 G `pretrain/pretrain_topic_dataset_api.md` |
| 已有 bin/idx 分片，合并成少量大文件 / bin 太碎 | 域 H `bin/merge_bins_tasks_api.md` |

> 判别要点：用户手里**已有 bin/idx 分片**只想合并 → 域 H；用户还在 parquet/json/jsonl 要**生成 bin** → 域 B / C。用户"要转 bin **且** 要求产物别太碎/顺便合并" → 不要两步走，直接用 `create_hunyuan_data_pretrain_conversion` 带 `enable_merge_bins=true`（见 §4.4）。

### 3.2 域 B：预训练数据转 bin 推荐调用链（parquet 模式）

```
① query_hunyuan_data_app_groups → 选 app_group
② list_hunyuan_data_app_group_hdfs_quota(app_group_name) → 取 hdfs_cluster（hdfsNn）
③ query_hunyuan_data_compute_resources(entity_type=PRETRAIN_DATA_CONVERSION) → 取 resource_id / resource_location
④ list_hunyuan_data_pretrain_operators → 取 operator（用户已给则跳过）
⑤⑥⑦ Tokenizer 三级级联：
  ⑤ list_hunyuan_data_pretrain_tokenizer_branches(wsid) → 取 branch
  ⑥ list_hunyuan_data_pretrain_tokenizer_model_series(wsid, branch) → 取 model_series
  ⑦ list_hunyuan_data_pretrain_tokenizer_train_stages(wsid, branch, model_series) → 取 train_stage
⑧ create_hunyuan_data_pretrain_conversion(…)
```

**json 模式**：跳过 ①②，直接从 ③ 开始。

### 3.3 域 D：跨地域 SFT 转 bin 自动搬运 Pipeline

命中流程文档 `references/sft/sft_bin_auto_transfer_flow.md`（**无独立封装工具**，Agent 编排 `create_hunyuan_data_export_task` + `create_hunyuan_data_sft_conversion`）：

```
Step 0: Agent 内联路径地域识别 + 生成 <staging_timestamp>（YYYYMMDDHHmmss），整条 pipeline 共用
Step 1: create_hunyuan_data_export_task（源 ceph → <nj_staging_root>/json/<ts>）→ 轮询 get_…_export_task
Step 2: create_hunyuan_data_sft_conversion（南京 /json → /bin，或直接落 target_path）→ 轮询 get_…_sft_conversion
Step 3: create_hunyuan_data_export_task（<nj_staging_root>/bin/<ts> → 目标 ceph）→ 轮询 get_…_export_task
```

> Pipeline 一旦启动，Step 1→2→3 必须由 Agent **自动轮询接续直到终态**，严禁让用户手动催。任一步 FAILED → 透传原始错误 + 输出 🧾 恢复清单 → **直接结束**，严禁自动重试 / 换路径。恢复时基于 `*_task_id` 直接 `get`，**严禁再次 create**（防止搬两遍）。

### 3.4 域 G：建 Topic + 建数据集编排

```
① create_hunyuan_data_pretrain_topic → 拿返回 id 当 topic_id
② create_hunyuan_data_pretrain_dataset(topic_id=<id>, ...)
若①报 Topic名称已存在 → query_hunyuan_data_pretrain_topics 按 name 查回 id 复用
```

---

## 4. 模块注意事项

### 4.1 存储术语别名（强制规则，先读后做）

| 用户说法 | 实际含义 | 处理方式 |
|---|---|---|
| `nitrofs` / `NitroFS` / `nitro` | 太极文件存储的一种（新一代文件存储） | 按**文件存储**处理，走 `create_hunyuan_data_export_task` |
| `hifs` / `HIFS` | **与 nitrofs 是同一个东西**（`hifs` 是后端/接口里的存储类型名，`nitrofs` 是产品名） | 完全等价于 nitrofs，**不要**当成另一套系统，也不要追问"你说的是 hifs 还是 nitrofs" |
| `ceph` / `cephfs` | 太极文件存储的另一种 | 同上，走 `create_hunyuan_data_export_task` |

- **文件存储互拷不需要区分两端是 ceph 还是 nitrofs(hifs)**：4 种方向**都用同一个工具、同一套参数**（`source_path` + `target_path`），**严禁**因为"一端是 nitrofs(hifs)"就去追问用户存储类型或改用别的工具。
- **存储类型识别规则（只用于表述，不影响选工具）**：挂载路径 `/apdcephfs_<location><id>/...` 中 **`<id>` ≥ 30 → nitrofs(hifs)**（如 `/apdcephfs_nj33`、`/apdcephfs_zw33`、`/apdcephfs_gz41`），**`<id>` < 30 → ceph**（如 `/apdcephfs_jn2`、`/apdcephfs_zwfy11`、`/apdcephfs_cq10`）；`/taijifs_*` 前缀一律是 nitrofs(hifs)。算出类型只用于把话说准，**不改变**工具与入参。
- **真正需要靠路径判别的只有两件事**：① `hdfs://` → 走域 E；② 外租卡（`/apdcephfs_wz*`、`/cudofs`，优先于上面的 id 规则）→ 走 `create_hunyuan_data_cudofs_copy_task`。

### 4.2 域判别逻辑与通用行为规则

1. **数据搬运类请求必须先做类型判别（最高优先级）**：意图涉及"拷贝/复制/搬迁/分发/导出/同步"时，先判别是**外租卡拷贝**（`create_hunyuan_data_cudofs_copy_task`）还是**文件存储互拷**（`create_hunyuan_data_export_task`）——二者完全独立、互不替代。判别维度只有一个：**外租卡 vs 文件存储**（不是 ceph vs nitrofs(hifs)）。判别流程：用户明确说"外租卡/cudofs/wz"→ cudofs；路径任一命中 `^/apdcephfs_wz[A-Za-z0-9_]*/` 或 `^/cudofs(/|$)` → cudofs；都不命中 → export；**无法判别时严禁直接调用任何工具**，必须追问"是文件存储（ceph / nitrofs(hifs)）之间的拷贝，还是外租卡存储数据拷贝？"。⚠️ "nitrofs/hifs" **不是**外租卡关键词。
2. **通用返回约定**：工具返回体为 `{"code", "message", "data"}`，`code == 0` 成功，`code != 0` 即业务失败（HTTP 层仍可能是 200），以 `code` 为准，不要只看 HTTP 状态码。
3. **通用参数约定**：列表类型参数（`list[string]`）必须传数组，不传字符串。`wsid` 语义因工具而异（多数必填≠0，少数可选或无），以 `info <tool>` 输出为准。
4. **wsid 语义按工具区分**：多数工具 `wsid` 必填（≠0）；`get_hunyuan_data_pretrain_shuttle_task` / `get_hunyuan_data_pretrain_dataset` 的 `wsid` 可选（传则校验归属），用户没给就不要传、不要填 0；`list_hunyuan_data_sft_tokenizers` 无 `wsid` 参数；`query_hunyuan_data_compute_resources` 无 `wsid` 参数且必须传 `entity_type=PRETRAIN_DATA_CONVERSION`。
5. **不臆造工具**：本模块不暴露融合任务的创建/重试/停止/删除工具，也不暴露 export / cudofs / topic / dataset 的删除工具（预训练转 bin 删除除外）；这些操作请走 Web 端，不要臆造同名工具。
6. **每轮新 prompt 必须重新判断归属**：用户转向"任务 / 指标 / 评测 / 模型"等其他意图时，立即跳出本 skill，由聚合 skill 路由表选择正确子 skill。

### 4.3 A. 数据导出 / 外租卡拷贝

- **参数齐备 → 立即直接调用**：`wsid` + 源/目标路径（或 `task_id`）齐备且类型判别完成时，立即调用对应工具并反馈结果；缺参数（`wsid=0`、外租卡两端都是 ceph 路径等）才追问。
- **ceph 与 nitrofs(hifs) 之间不用再分家**：两端都是文件存储时直接调 `create_hunyuan_data_export_task`（`wsid` + `source_path` + `target_path`），不要追问存储类型、不要索要应用组。用户说"hifs"时按 nitrofs(hifs) 理解。
- **按业务实体导出**（用户给的是 topic 数据版本 ID / 数据集 ID 而非路径）：用 `entity_type` + `entity_id`（topic 数据版本 → `POSTTRAIN_TOPIC_DATA`；预训练数据集 → `PRETRAIN_DATASET`），`data_source` 与 `entity_type` 对齐；与按路径导出二选一。
- **失败时主动续跑下一步**：`get_hunyuan_data_export_task` / `get_hunyuan_data_cudofs_copy_task` 查到 `FAILED` → 主动调用对应日志工具（`get_hunyuan_data_export_task_log` / `get_hunyuan_data_cudofs_copy_task_log`）定位原因并一次性汇报，不要让用户再追问。
- **外租卡任务 PENDING 时不要查日志**（会报"智研任务执行记录不存在"）：先 `get_hunyuan_data_cudofs_copy_task` 确认 `taskStatus` 不是 PENDING。
- **"转 bin + 跨地域"复合需求**：路由到 §3.3 的 Pipeline 编排；只有用户明确说"先帮我把数据从 A 拷到 B"（一次性互拷，不要求转 bin）或"我自己控制流程"时，才单独跑 `create_hunyuan_data_export_task`。

### 4.4 B. 预训练数据转 bin（parquet / json）

- **⭐ 先判别 `file_format`**：用户提到 parquet / hdfs / 应用组 → `file_format=parquet`（默认）；提到 json / ceph 上的 json → `file_format=json`。没说清时**先问一句**"输入是 HDFS 上的 parquet，还是 CEPH 上的 json？"，不要瞎猜。
- **⭐ 调用顺序见 §3.2**。
- **独立查询直达**：用户只是问"转 bin 前有哪些应用组/某应用组有哪些 HDFS 集群/平台公共计算资源/tokenizer 某级候选"时，只调对应工具一次直接回答，不重复调用；不得跳去模型管理、存储治理或资源管理。出现"master 分支 + hy_3 模型系列 + 训练阶段"这类级联字段即按 tokenizer 三级级联处理。
- **必填字段**：`wsid`（≠0）、`resource_id`、`operator`、`git_based_tokenizer`（三段齐备）、`input_path`、`output_path`（CEPH 空目录、有写权限）；**parquet 模式额外必填** `app_group` + `hdfs_cluster`。`file_format` 选填默认 parquet；`seq_len` 选填不填提交 `-1`。
- **两种模式差异只有 3 点**：`file_format`、输入路径类型（HDFS parquet ↔ CEPH json 目录）、parquet 需 `app_group`+`hdfs_cluster` 而 json 不需要。底层按 `file_format` 自动填 `storageType`（parquet→HDFS、json→CEPH），调用方无需关心 storageType。
- **parquet 模式的应用组【必填】**：读 HDFS 数据必须应用组——与 ceph2ceph 拷贝（`create_hunyuan_data_export_task` 不需要应用组）截然不同，不要混淆。
- **新版字段**：旧 `tokenizer` 字符串 / `skip_long_seq` 已废弃，分别由 `git_based_tokenizer` 和 `packing_strategy` 替代；`packing_strategy` 默认 `CHUNK_LONG_SEQ`。
- **默认值回显**：创建工具兜底 `packing_strategy` / `add_source_id` / `add_loss_mask` / `partition_num` / `resource_location` 等；返回里 `applied_defaults` **必须原样转告**用户。
- **任务名兜底（`name` 未给时自动生成）**：用户没指定任务名时，Agent **必须**用**当前时间**生成 `name = create_by_skill_<YYYYMMDD-HHMM>`（如 `create_by_skill_20260803-1530`，月/日/时/分补 0、24 小时制）并**显式传入**创建工具；**严禁**提交 `<YYYYMMDD-HHMM>` 之类占位符或复用历史时间戳。用户给了名字就原样用；`name` 不是必填项，**严禁**为了名字追问用户。建完把实际 `name` 回显。
- **失败时主动查/重试**：FAILED → 如实转达 message，可建议 `retry_hunyuan_data_pretrain_conversion`。
- **转 bin 顺带合并（`enable_merge_bins`）**：用户明确说「产物别太碎 / bin 文件太多 / 顺便合并成 N 个」时，在创建时带 `enable_merge_bins=true`（可选 `merge_bins_count=N`，**必须 > 0**，不确定就不传由后端兜底），**不要**先转完再建独立合并任务。用户没提就别自作主张开启（默认 false）。开启后轮询要同时看 `merge_bins_status` 和 `merge_bins_output_path`（成功后才回填）——**主 `status` 成功 ≠ 全流程结束**，合并成功前不触发跨地域同步；失败透传 `merge_bins_url`，不自拼 URL。若用户**手里已经是 bin/idx 分片**只想合并 → 那是 `create_hunyuan_data_merge_bins_task`（域 H），别在这里空转一次 bin。
- **不重复创建**：同一组参数只调用一次；返回权限/路径等业务错误时原样转达并停止，不要用相同参数重复创建。用户已显式给出的字段（app_group/operator/tokenizer 三段）直接复用，不重复查询。
- **删除是危险操作**：调用 `delete_hunyuan_data_pretrain_conversion` 前**必须先向用户二次确认**（仅需 `task_id`）；**严禁**在用户未明确表达"确认删除"时自行调用。建议删除前先用 `get_hunyuan_data_pretrain_conversion` 核对任务信息再删。

### 4.5 C/D. SFT / 后训练数据转 bin 与跨地域 Pipeline

- **SFT 转 bin 一律在本域处理**：单步工具手册见 `references/sft/sft_conversion_api.md`；跨地域 Pipeline 编排见流程文档 `references/sft/sft_bin_auto_transfer_flow.md`（**无独立封装工具**，由 Agent 编排 `create_hunyuan_data_export_task` + `create_hunyuan_data_sft_conversion`）。
- **南京地域强校验**：单步 `create_hunyuan_data_sft_conversion` 要求源/目标都在南京；任一端不在南京 → 改走跨地域 Pipeline。
- **⭐ 跨地域 Pipeline 自动驱动**：源/目标/tokenizer/seq_len/wsid 已齐备但用户未给 `nj_staging_root` 时，先使用内置 `DEFAULT_NJ_STAGING_ROOT`（`/apdcephfs_nj2/share_303722668/datax-pre/skill/ceph_pipeline_bin_default_address`，见流程文档）继续启动；使用前**必须显式提示**"使用平台默认中转 ceph 地址带宽会比较慢，建议提供您工作空间下的南京 ceph 路径"，取得继续意愿后再用。Pipeline 启动后 Step 1→2→3 自动轮询接续，**严禁**让用户手动催"再查一次"或"现在跑下一步"。
- **`source_path` / `input_path` 必须是单个数据文件**（不能是目录）；用户给目录 → 先索取目录下具体文件路径，未拿到不启动任何 Step。
- **失败即抛出原因并结束**：任一步 FAILED → 透传原始错误（Step 1/3 调 `get_hunyuan_data_export_task_log`；Step 2 取返回 `message`）+ 输出 🧾 恢复清单 → **直接结束**，**严禁**自动重试 / 换路径。仅用户显式要求才 `retry_hunyuan_data_sft_conversion`。
- **bin size 查询**：bin 文件大小在 `get_hunyuan_data_sft_conversion` 返回的 `size` 字段。

### 4.6 E. HDFS ↔ Ceph 双向搬运

- **何时进入本域**：用户要"把 HDFS 数据搬到 Ceph"或"把 Ceph 数据搬到 HDFS"（含导出/拷贝/搬迁/回传等同义表述），且能识别出一端是 `hdfs://` 路径。两端都是 ceph/外租卡 → 回域 A。
- **方向即工具，不传方向枚举**：两个 create 工具**没有也不接受** `type`/方向参数，方向已固定在工具里。源 `hdfs://` → 目标 ceph 用 `create_hunyuan_data_hdfs_to_ceph_transfer`；源 ceph → 目标 `hdfs://` 用 `create_hunyuan_data_ceph_to_hdfs_transfer`。
- **方向判别（强约束）**：两端都不含 `hdfs://` → 不属于本域；方向不明（只给一个路径 / 看不出哪端是 hdfs）→ **严禁直接调用**，先向用户追问完整源/目标路径。
- **参数齐备 → 立即直接调用**：`wsid` + `source_path` + `target_path` 齐备且方向已判别时，立即调用对应工具并反馈结果。`wsid` 缺失可先按 helper 查询工作空间。
- **失败时主动续跑**：`get_hunyuan_data_transfer_task` 查到 `FAILED` → 主动透传 `message`，并把返回的 `wedata_url` 智研链接**原样**给用户定位；**严禁自拼 URL**。

### 4.7 F. 预训练融合任务查询

- 只读场景：拿到 `shuttle_task_id` 即调 `get_hunyuan_data_pretrain_shuttle_task`；`wsid` 可选（传则校验归属，用户没给不要填 0）。
- 查到 `status=FAILED` → 把 `message` 原样转达，并指出失败阶段（`phase_status` 里第一个非 SUCCEEDED / 非 SKIPPED 的阶段）；可附 `wedata_url`。不臆测原因、不自动重试。
- 判别要点：用户提到「融合任务 / 融合 Pipeline / shuttle」或问一条 Pipeline 的多阶段整体进度 → 本域；只提「转 bin」而不提「融合」→ 走域 B / C。

### 4.8 G. 预训练 Topic / 数据集

- **建数据集前先落实 Topic（强约束）**：用户给了 `topic_id` 直接用；只给 Topic 名 → 先 `query_hunyuan_data_pretrain_topics` 查回 id；要新建 → 先 `create_hunyuan_data_pretrain_topic` 拿返回 `id` 当 `topic_id`。**严禁臆造 topic_id。**
- **未定 Topic 必须先追问、严禁擅自 create**：用户要登记数据集但未给 `topic_id`、也未明确表示要新建 Topic 时，只允许先 `query_hunyuan_data_pretrain_topics` 列出候选，然后**停下来向用户追问**挂到哪个 Topic（或是否新建）；拿到明确指令前严禁调用 `create_hunyuan_data_pretrain_dataset` 或自行 `create_hunyuan_data_pretrain_topic`。
- **创建类工具严格单次调用（防重复建资源）**：`create_hunyuan_data_pretrain_topic` / `create_hunyuan_data_pretrain_dataset` 一次任务内对同一个 Topic / 数据集只允许调用一次；返回 `code=0` 且带回 `id` 即视为创建成功、立即停止。需确认结果改调只读的 `get/query`，不要再 create。
- **枚举严格取值**：`stage_list`（STAGE_ONE/ANNEALING/STAGE_TWO/LONG_CONTEXT/Pretrain/Midtrain）、`type`（CLEANED_DATA/TRAINING_DATA/… 见手册枚举表）、`file_data_type`（JSON/PARQUET/ICEBERG）只能取工具手册枚举表里的值；用户说中文按表映射成英文枚举，拿不准先问。
- **创建数据集必填**：`wsid` + `key`（1-128 位字母数字-_）+ `topic_id` + `storage_path`（CEPH 目录）+ `stage_list`（非空数组）+ `type`；缺任一先追问。
- **与后训练体系隔离**：本域是**预训练**；后训练的 `create_hunyuan_data_topic` / `create_hunyuan_data_topic_dataset` 属 `posttrain-data-skill`，别张冠李戴。
- **失败即如实转达**：`Topic名称已存在` → 改用 query 查回 id 复用；枚举报错 → 对照枚举表修正后重试一次。

### 4.9 H. Bin 分片合并

- **何时进入本域（强判别）**：用户**手里已经有 bin/idx 分片**（转 bin 产物太碎、文件数太多），只想把它们合并成少量大文件 → 本域；用户还在 parquet/json/jsonl 阶段、要**生成 bin** → 那是转 bin，回域 B / C，别混。
- **仅 Ceph→Ceph**：输入/输出都在 Ceph，无 HDFS/parquet 分支。分片在 HDFS → 先用域 E 搬到 Ceph 再合并。
- **地域一致性硬约束**：`input_path` 地域必须 == `task_config.resource_config.resource_location`，否则后端报错。缺 `resource_id`/`resource_location` → 先 `query_hunyuan_data_compute_resources`（**务必传 `entity_type=PRETRAIN_DATA_CONVERSION`**，否则只返回南京资源）取资源再建，**严禁臆造资源 id**。
- **参数齐备 → 立即直接调用**：`wsid` + `input_path` + `task_config.resource_config`(`resource_id`+`resource_location`) 齐备时立即调 `create_hunyuan_data_merge_bins_task`。`output_path` 用户没给就**不传**（后端自动在同级生成 `xxx_merged_<时间戳>`），建完回显实际路径，别瞎编。
- **任务名兜底（`name` 未给时自动生成）**：同 §4.4 规则，不赘述。
- **失败时主动透传、留意跨地域**：`get_hunyuan_data_merge_bins_task` 查到 `FAILED` → 透传 `message` + `wedata_url`（**严禁自拼 URL**）；跨地域任务额外看 `transfer_status`（主合并成功后自动建 export 同步）。仅用户显式要求才 `retry_hunyuan_data_merge_bins_task`。
