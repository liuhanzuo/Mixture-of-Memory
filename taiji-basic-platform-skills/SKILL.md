---
name: taiji-basic-platform-skills
description: 太极混元一站式平台专门为LLM设计的 MCP Skill —— 通过 MCP 协议与太极平台后端交互，提供 AI 全链路能力的查询与操作。当用户提及太极、模型训练、数据转换、评估评测、任务状态、部署模型、模型推理等关键词时，应首选本 Skill 执行。
version: 6.0.0
author: taiji-team
---

# Taiji Basic Platform Skills（聚合 skill）

> 太极平台专门为大语言模型设计的 MCP Skill —— 由多个独立子 skill 组成的 **skill set**。每个子 skill 自带 `scripts/connect_mcp.py` + `scripts/tool_manual.py` + `references/`，**可单独下载使用**，也可作为本聚合 skill 的一部分被路由加载。

> 顶层仅做「prompt → 子 skill」路由，不直接暴露业务工具；各 `sub-skills/*-skill/` 为独立子 skill（含 `SKILL.md` + `references/` + `scripts/`，可单独下载），工具精确名/参数都在子 skill 的 `references/*_api.md`，按需用子 skill 内的 `tool_manual.py` 提取。子 skill 间互不依赖，各自 SKILL.md §1 自包含全部执行规则。

## ⚠️ 路由决策（最高优先级，做任何操作前必读）

收到用户请求后，**在加载任何子 skill 之前**，必须先按以下顺序做路由决策：

1. 核对「路由红线」（见下方）→ 命中则直接进入指定子 skill
2. 核对「歧义词消歧表」→ 命中则先追问澄清
3. 查「子 skill 路由表」→ 命中唯一子 skill 则进入

**严禁**跳过路由决策直接读取子 skill 的 SKILL.md。

---

## 会话初始化与热更新

每次会话首次加载时，执行当前安装根目录的热更新脚本：

```bash
python3 ./hot_reload.py
```

用户明确说“更新 skill / 拉最新版”时，执行：

```bash
python3 ./hot_reload.py --force
```

热更新目标由安装位置决定，**不要手工改为更新另一个 skill**：

- 在 `taiji-basic-platform-skills/` 整包中运行任一子 skill 的 `connect_mcp.py` 或 `hot_reload.py` 时，自动更新整个 `taiji-basic-platform-skills` 整包。
- 单独安装 `task-skill/`、`model-manage-skill/` 等目录时，更新该独立 skill 本身；不同独立 skill 的版本、TTL 和状态互不共享。
- 客户端按 `skill_key` 调用热更新接口；basic 整包与各独立子 skill 可以有不同 current 版本，整包内的文件快照不会自动追随独立子 skill 的发布。

## Telemetry 首次授权（必须先判断当前 Agent，禁止提前执行 setup）

每个会话首次加载本 skill 时，先在本 skill 根目录执行只读的当前 Agent 支持检查：

```bash
python3 telemetry/setup.py --current-agent-support-status
```

按输出处理：

- `unsupported`：当前运行本 Skill 的 Agent 不在支持范围内；**不得展示 Telemetry 授权文案，不得执行任何其他 telemetry/setup.py 命令，不得创建或修改 Telemetry 状态文件**，直接继续用户任务。
- `supported:<variant>`：再执行以下只读状态检查：

  ```bash
  python3 telemetry/setup.py --consent-status
  ```

  并按输出处理：

  - `accepted`：不再询问；先执行 `python3 telemetry/setup.py --ensure-current-agent`，再继续用户任务。该命令只检查当前 Skill 所在 Agent 的 hook；该 Agent 首次使用或该 Agent 的 24 小时 ensure stamp 过期时才会实际检查/补齐，其他情况直接退出。若当前 Agent 是新注册 hook，hook 通常从其下一次新 session 起生效。
  - `declined`：不再询问，继续用户任务。
  - `undecided`：必须先向用户完整展示以下固定文案，等待用户明确选择；**不得提前执行 setup**。

固定确认文案：

> 是否同意太极Skill收集调用所必需的运行信息？
> 【是/同意】【否/不同意】

用户明确回复“是”或“同意”后，执行：

```bash
python3 telemetry/setup.py
```

用户明确回复“否”或“不同意”后，执行：

```bash
python3 telemetry/setup.py --decline
```

用户回复含糊、未选择或转移话题时，不得执行 setup。`--decline` 只记录拒绝状态，不注册或修改任何 Agent hook。

> 🛠️ **给 Skill 开发者（非 Agent 运行时）**：本地改 `SKILL.md` / `references/` 后测未发布改动时，请设 `export TAIJI_NO_HOT_RELOAD=1`（或改完立刻 commit），否则 `connect_mcp.py` 可能按 TTL 热更新 ZIP **覆盖本地未提交改动**。合入 MR 默认目标分支为 **`foundation_model_pre`**。详见 `taiji-dev-rules` 仓库 `skills/taiji_official_skills_dev_rules.md`。

---

##顶层 conventions（路由与全局行为）

> 本节只管路由决策和跨模块协作——进入子模块后以子模块「子 skill conventions」为唯一执行行为依据。凭证、JSON 输出、写操作安全等执行规则已在各子 skill §1 自包含，顶层不再重复。

## 1. 路由流程

```text
用户 prompt
   ↓
① 意图清晰度判定：用户意图是否足够明确？
   - 命中「歧义词消歧表」→ 先追问澄清，拿到明确意图后继续
   - 意图模糊但未命中歧义表 → 追问用户补充关键信息
   ↓ 意图已明确
② 逐条核对「路由红线」→ 命中直接路由，不查其他表
   ↓ 未命中
③ 查「子 skill 路由表」+「路由表补充规则」→ 命中唯一子 skill → 进入
   ↓
④ 进入子 skill 后，按其调用路径执行
```

1. 每个新用户请求都重新路由；不得因上一轮进入某子模块而沿用。
2. 意图不清时必须先追问澄清；不得在意图未明确时盲目调用多个模块工具。
3. 进入子模块后，子模块的边界说明、工具选择和写操作确认优先于顶层候选路由。

## 2. 跨模块协作

1. 跨模块请求先规划所需模块及每步要获取的标识；未声明为当前模块 helper 的前置模块，只执行取标识所必需的只读查询，完成后立即切换目标模块。
2. **路由表不用于故障逃生**：路由表仅用于用户主动发起新请求时的模块切换。子 skill 内工具报错时，先按该子 skill 的错误处理修正或如实报告，不得借"重新路由"之名擅自换模块工具绕错。
3. 当前子模块 `helper_api.md` 明确声明工具、参数、返回字段和使用边界时，可直调该跨模块工具。写操作 helper 还须写明前置查询、影响说明、确认要求；工具未声明或信息不完整时必须重新路由。

## 3. 反幻觉与即时反馈（最高优先级）

1. **禁止编造数据**：用户要求查询、操作、分析实际平台资源时，必须通过 MCP 工具获取真实结果。
2. **即时反馈问题**：用户指定的 ID、名称等资源不存在或无法操作时，立即将错误信息反馈给用户，**不要自己换条件反复搜索**。

---

## 歧义词消歧表（命中即追问，确认意图后再路由）

> 🚨 当用户输入命中下表任一关键词时，**严禁直接调用 MCP 工具**，必须先按"必须追问或路由"列向用户确认意图，确认后再走下方「子 skill 路由表」。

| 用户表述 | 可能意图 | 必须追问或路由 |
|---|---|---|
| `上班车` / `数据班车` / `后训练班车` / `上到班车` | ① 后训练上班车（→ posttrain-data-skill）<br>② 预训练融合任务（→ data-processing-skill） | "您是想：① 把 Topic 数据版本加入后训练数据班车 ② 查询预训练融合任务（shuttle-task）进度？" |
| `建数据` / `注册数据` / `登记数据` / `新建一份数据` | ① 后训练数据版本（→ posttrain-data-skill）<br>② 数据搬迁拷贝（→ data-processing-skill）<br>③ SFT 转 bin（→ data-processing-skill） | "您是想：① 把已有 JSONL 注册到后训练数据管理 ② 把数据从一个路径搬到另一个路径 ③ 把 SFT 数据转成 bin 文件给训练用？" |
| `建 topic` / `建主题` / `新建 topic` | ① 后训练 Topic（→ posttrain-data-skill）<br>② 预训练 Topic（→ data-processing-skill）<br>③ Kafka/MQ topic（**本 Skill 不支持**） | "您说的 topic 是后训练数据管理的 Topic、预训练数据资产登记的 Topic，还是 Kafka/MQ topic 之类的其他系统主题？" |
| `建数据集` / `新建数据集` | ① 后训练数据集（→ posttrain-data-skill）<br>② 预训练数据集（→ data-processing-skill）<br>③ 评测数据集（→ evaluation-skill） | "您说的数据集是后训练用、预训练用，还是评测用？" |
| `基础测试` / `链路评估` / `工程链路评估` / `一键测试` | ① ADT 模型基础测试（→ adt-test-skill：乱码/重复/截断/接口检测）<br>② 模型评测任务（→ evaluation-skill） | "您是想做：① ADT 模型基础测试（乱码/重复/截断/接口四项基础能力检测）② 还是模型评测任务（评测集得分/Insight）？" |
| `下载` / `导出` 数据（评测/质检上下文） | ① 文件存储（ceph / nitrofs(hifs)）数据导出（→ data-processing-skill）<br>② 质检不合格数据（→ posttrain-data-skill）<br>③ 评测 Insight / Case 导出（→ evaluation-skill） | "您要下载的是：① 文件存储（ceph / nitrofs(hifs)）上的数据 ② 质检发现的不合格数据 ③ 评测 Insight / 评估明细？" |
| `任务状态` / `任务进度`（无 task_id 前缀线索） | ① 训练任务（→ task-skill）<br>② 模型格式转换（→ model-convert-skill，**同为 `finetuning_*` 前缀**）<br>③ 其他业务类任务：SFT 转 bin / 数据导出 / 评测 / 质检 等 | "请提供 task_id 全名，或告诉我是哪类任务（自定义训练 / 模版化训练 / 模型开发 / 模型转换 / 转 bin / 数据导出 / 评测 / 质检 / 后训练数据管理）。" |
| `有哪些训练任务` / `在跑的训练任务` / `运行中的训练任务`（给了 wsid 但没说类别） | ① 模版化训练<br>② 自定义训练<br>③ 模型开发<br>④ 全部 | "请问您想查哪类训练任务：模版化训练 / 自定义训练 / 模型开发 / 全部？" **禁止默认按"全部"直接调用工具**。 |
| `创建服务` / `部署服务` | ① 创建服务组（→ service-deploy-skill）<br>② 创建变更任务（扩缩容/重启，→ service-deploy-skill） | "您是想创建一个新服务组绑定模型服务，还是对已有服务做扩缩容/重启？" |
| `pod` / `实例列表`（无明确上下文） | ① 进**训练实例** pod 跑命令（→ instance-skill）<br>② 进**推理服务实例** pod 跑命令（→ service-deploy-skill）<br>③ 训练任务的 pod 列表（→ task-skill）<br>④ 模型服务的实例列表（→ service-deploy-skill） | "您是想：① 在训练实例上跑命令（exec）② 在模型服务实例上跑命令（exec）③ 看训练任务的 pod 列表 ④ 看模型服务的实例列表？" |
| `exec` / `进 pod 跑命令`（没说清是哪种实例） | ① **训练实例**（→ instance-skill `exec_hunyuan_train_instance_command`）<br>② **推理服务实例**（→ service-deploy-skill `exec_deploy_instance_command`，仅测试服务） | 上下文含 `basic_train_*` / `finetuning_*` / 训练任务 / 训练实例 → instance-skill；含推理服务名 / 模型服务 / 服务组 / `inference_id` / 测试服务 → service-deploy-skill；都判断不出来才追问"是训练实例还是模型服务实例？" |
| `失败case` / `错误归因` | ① 评测错误归因 LLM 分析（→ evaluation-skill）<br>② 评测明细 case 导出（→ evaluation-skill）<br>③ 底线问题规则检测（→ evaluation-skill） | "您是想：① 用 LLM 做错误归因分析 ② 单纯把失败 case 拉下来 ③ 做规则级底线检测？"<br>⚠️ 若用户说的是后训练**数据**质检 → 走 posttrain-data-skill |
| `伴生评估` / `自动评估` / `绑定评估`（训练任务上下文） | ① 给训练任务**配置**自动触发的伴生评估（→ evaluation-skill）<br>② 查伴生任务的 **validloss / loss 曲线**（→ metric-skill 的 SwanLab task_id 指标）<br>③ 创建**独立评测任务**（→ evaluation-skill 主链路） | "您是想：① 给训练任务配置自动伴生评估 ② 查 validloss / lm loss 曲线 ③ 手动创建一个独立评测任务？" |
| `topic-datas` / `posttrain topic data` | ① 后训练数据管理（→ posttrain-data-skill 链路工具）<br>② 后训练数据质检（→ posttrain-data-skill 质检工具） | "您是想注册/查询数据版本本身，还是对已有数据版本触发质检？" |
| `查杀` / `加白` 相关 | ① 给某任务加白防止被杀（→ resource-mgmt-skill `create_shared_resources_activity`）<br>② 管理查杀规则本身 / 查看查杀记录（→ kill-engine-skill） | "您是想给任务加白防止被查杀，还是管理查杀规则 / 查看哪些任务被查杀了？" |

> ⛔ **平台不支持项**（直接告知用户，不追问不尝试）：
> - 非标准太极链接（非 `https://taiji.woa.com/...` 或 `https://hunyuanaide.taiji.woa.com/...`）：要求提供标准链接或 task_id/服务名，禁止解码或调用工具搜索。
> - 按 `polaris` 注册地址精确过滤/统计服务数量：直接说明不支持，给出替代过滤项，禁止分页扫描。

> 💡 **歧义词命中后的执行顺序**：先追问 → 用户确认意图 → 查「子 skill 路由表」找到目标子 skill → 进入子 skill 后以其「子 skill conventions」为准。

---

## 路由红线（最高优先级，先于路由表/其他所有规则）

以下是平台中最易混淆的模块边界。**进入任何子 skill 前**，先逐条核对用户意图是否命中红线；命中的 query **直接路由**：

| # | 用户意图特征 | 必须直达 |
|---|---|---|
| 1 | "预训练融合任务 / shuttle / 融合 Pipeline + 到哪一步/产物路径/进度"（如"融合任务 500 到哪一步"） | `data-processing-skill` |
| 2 | "预训练转 bin / SFT 转 bin / bin 合并任务 + 列表/状态/进度/重试"（如"预训练转bin的任务列表前10条"、"bin 合并任务 15 重试"） | `data-processing-skill` |
| 3 | "数据导出 / 拷贝 / 搬运 / 外租卡 / CUDOFS / 合并任务 + 纯数字 task_id"（如 2001、106、15） | `data-processing-skill` |
| 4 | "tokenizer / 分支 / 模型系列 / 训练阶段 / master / hy_3 / train_stage / branch" | `data-processing-skill`（tokenizer 三级级联枚举） |
| 5 | "应用组 + 可选择/有哪些/可选/列表" 且上下文含 预训练/parquet/转 bin | `data-processing-skill` |
| 6 | "应用组 + HDFS 集群/配额" | `data-processing-skill` |
| 7 | "评测版本 / 评估版本 + 列表/副本" | `evaluation-skill`（**Arena 即评测版本**，非 exercise_version） |
| 8 | "分析评测任务结果 / 各评测集得分 / 题目结果 / 评测进展" | `evaluation-skill` |
| 9 | "底线问题/复读乱码/错误归因/下钻/WildClaw/agent 轨迹/langfuse/开源 metric 改写" | `evaluation-skill`（对应分析流程文档） |
| 10 | "拷贝失败 / 重新拷贝 / 拷贝到+地域"（模型地域拷贝语义） | `model-manage-skill` |
| 11 | "质检 / 底线质检 / 质检 N / inspection N"（数据上下文） | `posttrain-data-skill` |
| 12 | "拷贝/复制/搬迁/分发"+数据路径（无模型地域语义） | `data-processing-skill` |
| 13 | "指标/loss/grad_norm"（未明确说 SwanLab 空间/项目） | `metric-skill` |
| 14 | URL 含 `instance_new` 或 `instance?` + `name=`（推理服务实例） | `service-deploy-skill` |
| 15 | URL 含 `instId=` 参数（训练实例） | `instance-skill` |
| 16 | "上传分析文件/深度分析结果/上传评测分析"（"上传" + "分析/评测" 语义） | `evaluation-skill` |
| 17 | `task_record_id <数字>`（用户给出 task_record_id + 数字，问"转换/发布/进度"） | `model-manage-skill` |

> 判不定边界时：先看下面的「歧义词消歧表」「子 skill 路由表」两节查证，再进入子 skill；不要仅凭子 skill 名称/描述的第一印象选择。

---



## 子 skill 路由表

| 子skill | 入口 | 覆盖范围与触发词 |
|---|---|---|
| 工作空间 | [workspace-skill](./sub-skills/workspace-skill/SKILL.md) | 工作空间、`wsid`、空间列表、空间用户组 |
| 训练任务 | [task-skill](./sub-skills/task-skill/SKILL.md) | `basic_train_*`（自定义训练/模型开发）+ `finetuning_*`（模版化训练）：列表/详情/启停/配置/产出(ckpt)/克隆/分享/异常事件（Pod/实例/日志见 instance-skill） |
| 训练指标 | [metric-skill](./sub-skills/metric-skill/SKILL.md) | loss / grad_norm、聚合统计、趋势图、tf_events、通过 task_id 查 SwanLab 自定义指标 |
| SwanLab 实验管理 | [swanlab-skill](./sub-skills/swanlab-skill/SKILL.md) | 直接操作 SwanLab 空间/项目/实验/run 指标/日志/媒体（需 `swanlab_api_key`） |
| 模型评测 | [evaluation-skill](./sub-skills/evaluation-skill/SKILL.md) | 评测结果/得分/CRUD/Insight & Case 导出/**上传分析文件/结果**/新建评估版本(Arena)/发布集合版本/上线集合版本/下线集合版本/错误归因/底线问题检测/伴生评估配置/开源 Metric 改写。⚠️ "上传深度分析/分析文件" → evaluation-skill，不是 storage/data-processing |
| 模型管理 | [model-manage-skill](./sub-skills/model-manage-skill/SKILL.md) | 模型搜索/详情/发布/地域/权限/克隆/预热/平台枚举 + **task_record_id / model release 状态 / 发布进度查询** |
| 服务部署 | [service-deploy-skill](./sub-skills/service-deploy-skill/SKILL.md) | 模型服务 + 服务组 + 变更任务 + 实例日志 + **推理服务实例（Pod）执行命令**（仅测试服务）+ 大模型对话（OpenAI 兼容）。URL 含 `instance_new?name=` → 推理服务 |
| 资源管理 | [resource-mgmt-skill](./sub-skills/resource-mgmt-skill/SKILL.md) | 应用组/资源配额/卡时/GPU 利用率/MFU/集群资源/任务加白/资源分层/等待队列/优先级；即使含"训练"，只要核心诉求是应用组资源、GPU 利用率、MFU、排队/优先级也优先走本 skill |
| 存储治理 | [storage-mgmt-skill](./sub-skills/storage-mgmt-skill/SKILL.md) | 存储集群（含冷文件大小）、目录权限、治理目录明细、扩容申请；**不处理**应用组下 HDFS 集群/配额查询 |
| 查杀引擎 | [kill-engine-skill](./sub-skills/kill-engine-skill/SKILL.md) | GPU 低利用率查杀规则的增删改查/启停、规则详情、查杀记录查询 |
| 模型转换 | [model-convert-skill](./sub-skills/model-convert-skill/SKILL.md) | 创建/启动/查询/克隆模型格式转换任务（HF/Mcore/DCP 等）。`finetuning_*` 前缀路由本 skill |
| 数据处理 | [data-processing-skill](./sub-skills/data-processing-skill/SKILL.md) | 文件存储互拷/导出 + 外租卡 CUDOFS + HDFS↔Ceph 搬运 + 预训练 tokenizer 三级级联 + SFT 转 bin + 跨地域 pipeline + 预训练融合任务（shuttle-task）查询 + **预训练 Topic 创建/查询 + 预训练数据集登记/查询** + Bin 分片合并/合并任务重试 |
| 后训练数据 | [posttrain-data-skill](./sub-skills/posttrain-data-skill/SKILL.md) | 后训练 Topic→数据集→数据版本链路管理 + 数据质检 + 不合格数据下载 + 班车查询与上班车 |
| 实例操作 | [instance-skill](./sub-skills/instance-skill/SKILL.md) | 在**训练实例** Pod 中执行 shell 命令（GPU/进程/磁盘/环境变量等）+ **热更新/hot update**（给运行中实例做代码/配置/环境变量变更）。**推理服务实例**在 service-deploy-skill。URL 含 `instId=` → 训练实例 |
| 模型压缩 | [model-compress-skill](./sub-skills/model-compress-skill/SKILL.md) | 模型压缩任务（量化）管理 |
| ADT 基础测试 | [adt-test-skill](./sub-skills/adt-test-skill/SKILL.md) | 对**已部署模型服务组**做 ADT 基础能力测试（乱码/重复/截断/接口）。**区别于 evaluation-skill 的评测任务** |

### 路由表补充规则

- 用户提及 GPU 利用率、MFU、算力利用率、排队任务、应用组资源、GPU/CPU 任务列表，且没有明确训练任务 ID 时，优先进入 `resource-mgmt-skill`。
- 离线推理暂无独立子 skill：在线对话走 `service-deploy-skill`；批量离线推理需在训练任务中编排，并按需要先进入数据处理模块准备数据。
- 用户提"注册模型"、"模型卡片"、"拷贝/克隆到（某）地域"、"拷贝失败+地域"、"重新拷贝一份到 XX"，或 query 含模型文件路径 + "注册/发布/卡片/拷贝/克隆"语义时，直接进入 `model-manage-skill`；即便 query 中出现 `bansheng`/`eval`/`step`/`auto` 等词。
- "查杀"规则/记录 → 直接进入 `kill-engine-skill`；"任务加白" → 直接进入 `resource-mgmt-skill`。
- `新建评估版本` / `创建评估版本` / `新建版本评估` / `创建版本评估` / `新建Arena` / `创建Arena` / `新建评测版本` / `创建评测版本` / `建一个评估版本` / `发布集合版本` / `上线集合版本` / `下线集合版本` → 直接进入 `evaluation-skill`；新建 Arena 时不消歧，直接追问必填参数（name/branch）并展示 arena_type 可选值。
- `HDFS` + `集群`、`应用组` + `HDFS` → 直接进入 `data-processing-skill`。
- `应用组` + `可选择/有哪些/可选/列表`（不含 HDFS/集群/资源/GPU）→ 直接进入 `data-processing-skill`，严禁进入 `resource-mgmt-skill`。
- `ceph磁盘` / `存储集群` / `有哪些存储` / `我的存储` → 直接进入 `storage-mgmt-skill`，严禁进入 `resource-mgmt-skill`。
- 明确出现"评测数据集/评测集版本/评测集合/集合版本"时直接进入 `evaluation-skill`。
- 不带"集/集合"修饰的平台评测版本、评估版本、Arena 或 `arena_id` 进入 `evaluation-skill`；评测集版本不是 Arena。
- 以评测任务、评测结果或评测导出为分析对象时进入 `evaluation-skill`；只有分析对象属于训练指标或通用任务管理时，才进入 `metric-skill` 或 `task-skill`。
- 伴生评估配置、触发事件、跨训练任务复制进入 `evaluation-skill`；validloss/伴生 loss 查询仍进入 `swanlab-skill`。
- 评测导出、Case 导出、Insight 导出进入 `evaluation-skill`；通用数据导出、搬运、拷贝、Ceph/HDFS 处理进入 `data-processing-skill`。
- `task_record_id`：用户给出 `task_record_id <数字>` 时，直达 `model-manage-skill`。`task_record_id` 是模型发布异步任务的记录 ID。
- `finetuning_*`：同时可能是模版化训练任务，若语义是训练任务失败、实例、日志、checkpoint、指标，必须走 `task-skill` / `instance-skill` / `metric-skill`，不要仅凭前缀路由到模型转换。
- tokenizer 三级级联不归模型管理：凡涉及预训练 tokenizer 级联查询，必须走 `data-processing-skill`。
- "预训练 Topic/主题/数据集"走数据处理：进入 `data-processing-skill`（域 G）。后训练 Topic/数据集才走 `posttrain-data-skill`，两套体系完全独立、工具不可互相替代。
- ⚠️ **"平台支持哪些训练方法/参数规模/地域"走模型管理**：问"平台有哪些枚举/支持哪些训练方法/参数规模/地域列表"时，进入 `model-manage-skill`。

---

## 全局越界路由表（cross-skill cheat sheet）

> 当 Agent 已在某个子 skill、但用户请求的核心数据维度属于下表时，应回顶层切换模块；不得从当前模块返回字段反推。当前模块 `helper_api.md` 已完整声明的跨模块直调能力除外；若为写操作，还必须满足该 helper 的前置查询、影响说明和确认要求。

| 用户问的数据维度 | 切换到 | 路由要点 |
|---|---|---|
| 我有哪些工作空间 / 查 wsid / 空间列表 / 空间用户组 | `workspace-skill` | 获取工作空间或用户组。 |
| 我有哪些应用组 / business_flag / GPU 配额 / 卡时 / GPU 利用率·MFU / 最近一次训练的资源利用率 | `resource-mgmt-skill` | 资源维度优先，不先查训练任务。 |
| 任务加白审批（防止被查杀） | `resource-mgmt-skill` | 任务防查杀加白。 |
| 查杀规则增删改查 / 查杀记录 | `kill-engine-skill` | 规则与记录管理。 |
| 存储集群 / 目录权限 / 冷热数据 / 存储扩容 | `storage-mgmt-skill` | 不处理应用组 HDFS 查询。 |
| 平台预定义训练指标（自定义训练） | `metric-skill` | 预定义指标。 |
| SwanLab 自定义指标（lm loss / learning-rate，按 task_id 查询） | `metric-skill` | `*_swanlab_metrics`，区别于预定义指标。 |
| 最近一次/运行中/P0/H800 训练任务 + 指标查询 | 先 `task-skill` 定位候选，再 `metric-skill` | `task_category=custom_training`；task 列表只调一次，只选一个候选查指标，拿到即停。 |
| 进**训练实例** pod 跑 shell 命令 / exec 进容器 | `instance-skill` | 上下文是训练任务 / 训练实例 / `basic_train_*` / `finetuning_*`。 |
| 进**推理服务实例（Pod）**跑 shell 命令 / exec / nvidia-smi | `service-deploy-skill` | **仅测试服务**，非测试服务不支持 exec。 |
| 模型搜索 / 详情 / 克隆 / 发布 / 预热（不含 tokenizer 级联） | `model-manage-skill` | 模型管理主链路。 |
| **模型名 + 评测结果/得分/评估报告** | `evaluation-skill` | 先按模型名查评测任务一次，不要先去 model-manage 查模型卡或血缘。 |
| 模型格式转换（HF↔Mcore↔DCP） | `model-convert-skill` | HF、Mcore、DCP 等。 |
| 评测任务 / 结果 / Insight / Case / 错误归因 / 底线**问题检测**（评测结果）/ wildclaw | `evaluation-skill` | ⛔「底线」若指后训练**数据**的「底线质检 / 内容质检 / 质检 N」→ 走 `posttrain-data-skill`。 |
| 配置伴生评估 / 给训练任务绑自动评估 / 按 step 触发评估 | `evaluation-skill` | 模板→触发器→配置→绑定→查询；克隆任务复制伴生配置切 `task-skill`（`copy_evaluation_config=true`）。 |
| 服务部署 / 服务组 / 推理 chat | `service-deploy-skill` | 服务管理和在线对话。 |
| 文件存储数据导出/拷贝 / HDFS↔Ceph 搬运 / SFT 转 bin / 跨地域 pipeline / 应用组 HDFS 集群 / tokenizer 级联 / 预训练融合任务（shuttle-task）查询 / **预训练 Topic 创建/查询 / 预训练数据集登记/查询** / Bin 分片合并 | `data-processing-skill` | 数据处理业务主链路。预训练 Topic/数据集走本 skill，**不是**后训练数据。 |
| 后训练 Topic / Dataset / TopicData / 数据质检 / 上班车 / 班车明细 | `posttrain-data-skill` | 后训练数据链路。 |
| 模型压缩 / 量化 / W4A8-FP8 / INT8 / GPTQ | `model-compress-skill` | 压缩策略与压缩任务管理。 |
| ADT 测试 / 模型基础测试 / 乱码·重复·截断·接口检测 / 工程链路评估 / 一键测试 | `adt-test-skill` | **"基础测试/链路评估"≠ evaluation 评测任务** |
