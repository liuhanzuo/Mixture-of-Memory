# 太极评测平台 Insight 导出数据 Schema 与指标计算文档

> 本文档供 AI Skill 使用，用于理解太极评测平台 Insight 导出数据的文件格式、字段含义、异常值处理和指标计算方法。
> 文档不依赖测试文件，所有信息均来自后端代码和数据结构定义。

---

## 第一层：基础数据字段

### 1.1 文件层级结构

```
export_{exportTaskId}.tar.gz                          ← 外层归档（多 Task 时）
├── {benchName}__{evId}__task_{taskId}.jsonl.gz       ← 按评测集版本拆分的 case 数据
├── {benchName}__{evId}__task_{taskId}.jsonl.gz
├── trajectory/                                        ← agent 轨迹文件（仅含 agent 类 benchmark 时存在）
│   ├── req-{taskId}_{questionId}_{uuid}.jsonl        ← agent 执行轨迹（逐步 tool call & observation）
│   ├── req-{taskId}_{questionId}_{uuid}-chat.json    ← agent 对话轨迹（完整多轮消息）
│   └── ...
└── ...
```

- **文件命名**：`{exerciseName}__{exerciseVersionId}__task_{taskId}.jsonl.gz`，每个 (taskId, exerciseVersionId) 组合独立一个文件
- **trajectory 子目录**：当 case 的 payload 中包含 `trial_details.trajectory_info` 时，后端会自动将 ceph 上的轨迹文件复制到此目录一起打包
- **压缩**：双重压缩——内层 GZIP（`.jsonl.gz`），外层 tar+GZIP（`.tar.gz`）
- **编码**：UTF-8
- **格式**：JSONL（每行一个 JSON 对象）或 CSV（`payload` 为原始 JSON 字符串）

### 1.2 JSONL 顶层字段（固定 8 字段，顺序不变）

| # | 字段 | 类型 | 必有 | 说明 | 示例 |
|:--|:---|:---|:---|:---|:---|
| 1 | `taskId` | Long | 是 | 评测任务 ID，文件名中的 `{taskId}` 与此一致 | `10444` |
| 2 | `exerciseVersionId` | Long | 是 | 评测集版本 ID | `852` |
| 3 | `questionId` | Long | 是 | 题目 ID，同一题目在不同 Task 中 questionId 相同 | `238805301` |
| 4 | `id` | Long | 是 | StarRocks 行 ID，全局唯一 | `57103060237220` |
| 5 | `productType` | String | 是 | 产品类型（如 `VLM`、`LLM`） | `"VLM"` |
| 6 | `exerciseId` | Long | 是 | 评测集 ID，可能为 `-1` 表示未指定 | `-1` |
| 7 | `date` | String | 是 | 日期分区 `yyyy-MM-dd` | `"2026-03-16"` |
| 8 | `payload` | Object | 是 | **核心负载**，评测用例完整数据 | `{...}` |

> **字段顺序保证**：后端用 `LinkedHashMap` 写入，顺序恒定为上表所列。

### 1.3 payload 核心字段

#### 1.3.1 状态与标识

| 字段 | 类型 | 空值率 | 说明 |
|:---|:---|:---|:---|
| `__infer_status__` | String | 0% | 推理状态：`infer_success` / `infer_failed` |
| `__judge_status__` | String | 0% | 评估状态：`judge_success` / `judge_failed` |
| `model_status` | List\<Boolean\> | 0% | 每轮模型调用是否成功，如 `[true]` |
| `_internal_question_id_` | Long | 0% | 内部题目 ID，含 round 后缀（原始 ID × 100 + round_index） |
| `_internal_question_id_raw_` | Long | 0% | 内部题目原始 ID（不含 round 后缀） |
| `id` | Long | 0% | 数据记录 ID（与顶层 `id` 不同，此为原始数据 ID） |
| `task_id` | null | 100% | payload 内部冗余字段，**始终为 null**，使用顶层 `taskId` |
| `exercise_version_id` | null | 100% | payload 内部冗余字段，**始终为 null**，使用顶层 `exerciseVersionId` |
| `dataset_version_id` | Long | 0% | 数据集版本 ID |

#### 1.3.2 评分体系

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `score` | `List<List<Number>>` | 评分矩阵，维度为 `[round][repeat]`。通常 shape=`1x1`，如 `[[0.85]]` |
| `avg_score` | Number | 平均分。**取值范围 [0, 1]**，为 `score` 矩阵所有元素的均值 |
| `max_score` | Number | 最高分。`score` 矩阵中的最大值 |
| `min_score` | Number | 最低分。**注意：正常 case 中此值通常为 0（而非最小非零分）** |
| `extra_score` | `List<List<Number/null>>` | 额外评分指标，通常为 `[[null]]` |
| `pass_num` | Integer | 通过次数指标。**不是布尔值**，含义因 dataset 而异（见 §2.7） |

#### 1.3.3 Token 统计

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `avg_completion_tokens` | Integer | 平均 completion token 数（模型输出长度） |
| `avg_prompt_tokens` | Integer | 平均 prompt token 数（输入长度） |
| `token_count` | Integer | 总 token 数 = `avg_completion_tokens` + `avg_prompt_tokens` |
| `avg_finish_reason_length` | Integer | **截断标记**：`0` = 正常结束，`1` = 因 max_tokens 截断 |

#### 1.3.4 输入与输出

| 字段 | 类型 | 空值率 | 说明 |
|:---|:---|:---|:---|
| `input` | String | 0% | 原始输入，JSON 字符串格式，包含 messages 数组 |
| `output` | String | 0% | 标准/参考输出 |
| `ref_answer` | String | 0% | 标准答案（与 `output` 可能相同） |
| `prompt` | String/null | ~84% | 提示词模板，大部分为 null |
| `messages` | List\<Object\> | 0% | 完整对话消息列表，包含 system/user 等角色 |
| `responses` | `List<List<String>>` | 0% | 模型原始响应文本，`responses[round][repeat]` |
| `thinking_responses` | `List<List<String>>` | 0% | 模型思考过程（CoT/推理链），空字符串表示无思考过程 |
| `gpt_response` | `List<List<String>>` | 0% | 评判模型（Judge）的评估响应 |
| `model_input` | `List<List<Object>>` | 0% | 模型实际请求参数（见 1.3.6） |

#### 1.3.5 元数据

| 字段 | 类型 | 空值率 | 说明 |
|:---|:---|:---|:---|
| `benchmark` | String | 0% | 评测基准名称，如 `hyeval_v3_0` |
| `dataset` | String | 0% | 数据集名称，即**子评测集类别**（如 `数学`、`科学`、`rubric`、`translation_chrf`、`turing-code`、`ARC-AGI`、`Multi-IF`、`consistency`） |
| `split` | String/null | ~84% | 数据集切分标识 |
| `doc` | Object/null | ~74% | 原始文档/题目元信息 |

#### 1.3.6 model_input 结构

`model_input[round][repeat]` 是一个 dict，包含模型推理请求参数：

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `model` | String | 模型标识/接口名 |
| `messages` | List | 发送给模型的消息列表 |
| `max_tokens` | Integer | 最大生成 token 数 |
| `temperature` | Number | 采样温度 |
| `top_k` | Integer | top-k 采样 |
| `top_p` | Number | top-p（nucleus）采样 |
| `repetition_penalty` | Number | 重复惩罚系数 |
| `stream` | Boolean | 是否流式 |
| `openai_infer` | Boolean | 是否使用 OpenAI 兼容推理 |
| `query_id` | String | 查询 ID |
| `chat_template_kwargs` | Object | 聊天模板参数，如 `{"reasoning_effort": "no_think"}` |

#### 1.3.7 usage 结构

`usage[round][repeat]` 是一个 dict：

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `finish_reason` | String | 结束原因：`stop`（正常结束）/ `length`（因 max_tokens 截断） |
| `stop_reason` | String/null | 停止原因详情 |
| `reasoning_result` | Object/null | 推理结果详情 |
| `result` | Object | Token 使用详情 |
| `result.completion_tokens` | Integer | 本次生成的 token 数 |
| `result.prompt_tokens` | Integer | 本次 prompt 的 token 数 |

#### 1.3.8 内嵌 payload（维度标签）

`payload.payload` 是一个 dict，**不同 dataset 的 key 不完全相同**，数据中存在的字段会自动同步过来，可按需选择进行数据下钻筛选。

**标准下钻维度（可选维度）：**

| 字段 | 类型 | 说明 | 下钻用途 |
|:---|:---|:---|:---|
| `task_lv1` | String | **一级任务维度**（最粗粒度分类，如 `数学推理`、`代码生成`、`阅读理解`） | 按大类分析评测表现 |
| `task_lv2` | String | **二级任务维度**（中粒度分类，如 `初等数学`、`高等数学`） | 按子类别细分 |
| `task_lv3` | String | **三级任务维度**（最细粒度分类，如 `线性代数`、`概率统计`） | 精准定位薄弱知识点 |
| `language` | String | **语言**（如 `zh`、`en`、`ja`、`ko`、`multi`） | 按语言分析多语种表现 |
| `source` | String | **数据来源**（如 `original`、`translated`、`human_annotated`） | 按来源分析数据质量影响 |
| `difficulty` | String | **难度**（如 `easy`、`medium`、`hard`） | 按难度梯度分析能力边界 |

> **字段存在性说明**：以上维度字段并非每个评测集都全部包含，数据中实际存在哪些字段取决于评测集的配置。分析时应先检测哪些维度字段在当前数据中有值，再按存在的维度进行下钻。

**其他常见字段：**

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `data_lv1` | String | 一级数据维度标签（旧版字段，部分评测集使用） |
| `data_lv2` | String | 二级数据维度标签（旧版字段） |
| `dataset_name` | String | 数据集名称 |
| `id` | Any | 原始题目 ID |
| `lang` | String | 语言标签（旧版字段，新版用 `language`） |
| `owner` | String | 数据所属方 |
| `turn` | String/Number | 对话轮次 |
| `year` | String/Number | 年份 |

不同 dataset 的**差异字段**：

| dataset | 额外字段 |
|:---|:---|
| `translation_chrf` | `source_language`, `target_language`, `source_sentence`, `knowledge`, `num_in_context_examples`, `num_tokens_by_tiktokenizer`, `prompt_id`, `example_indices` |
| `科学` | `knowledge`, `topic_v1` |
| `consistency` | `knowledge` |
| `rubric` | `answer_source`, `meta` |
| `Multi-IF` | `answer_source`, `criteria`, `hyeval_id` |
| `ARC-AGI`、`数学` | `answer_source` |
| `turing-code` | `type`（代码题类型） |

#### 1.3.9 time_info 结构

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `start_produce_time` | String | 任务开始生产时间（`yyyy-MM-dd HH:mm:ss`） |
| `infer_receive_time` | String | 推理接收时间 |
| `infer_complete_time` | String | 推理完成时间 |
| `start_judge_time` | String | 开始评判时间 |
| `finish_time` | String | 完成时间 |

#### 1.3.10 黑盒标记

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `blackbox` | Number/null | 黑盒标记，`1` 表示黑盒数据。非 admin 用户导出时 `blackbox=1` 的记录会被过滤 |

#### 1.3.11 Agent 轨迹信息（trial_details）

**仅 agent 类评测 benchmark 包含此字段**。当 payload 中存在 `trial_details` 时，表示该 case 由 agent 执行，包含完整的运行配置、输出结果和轨迹文件路径。

**包含 `trial_details` 的典型 benchmark：**

| Benchmark | agent_name | 说明 |
|:---|:---|:---|
| `swe_bench_verified` / `swe_bench_pro` / `swe_bench_multilingual` | `swe_agent` | SWE-Bench 代码修复 |
| `hyeval_browsecomp` / `hyeval_browsecomp-zh` | `search_agent` | 搜索浏览 |
| `hyeval_finsearchcomp` / `hyeval_seal` / `hyeval_widesearch` | `search_agent` | 金融搜索/综合搜索 |
| `webarena` | `browser_gym_agent` | 网页交互 |
| `terminal_bench` | `terminus2` | 终端操作 |

> **无 agent 的 benchmark**（如 `mmlu`、`gsm8k`、`humaneval` 等）**没有 `trial_details` 字段**。

##### trial_details 顶层结构

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `nexus_server_req_id` | String | Nexus 服务端请求 ID，格式 `{taskId}_{questionId}_{uuid}`，也是轨迹文件名的一部分 |
| `run_input_data` | Object | Agent 运行输入配置（见下方） |
| `run_output_data` | Object | Agent 运行输出结果（见下方） |
| `token_usage` | Object | LLM 调用 token 统计（见下方） |
| `trajectory_info` | Object | **轨迹文件 ceph 路径**（见下方） |

##### trajectory_info —— 轨迹文件映射

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `trajectory_path` | String | Agent 执行轨迹 JSONL 文件的 ceph 绝对路径，每行记录一次 tool call / observation |
| `trajectory_chat_path` | String | Agent 对话轨迹 JSON 文件的 ceph 绝对路径，完整的多轮对话消息列表 |

**路径格式规律：**

```
/apdcephfs_jn2/share_303438049/evaluation/inner_data/hyeval/nexus_agent_trace/
  {yyyyMMdd}_{HH}/
    req-{nexus_server_req_id}.jsonl          ← trajectory_path（执行轨迹）
    req-{nexus_server_req_id}-chat.json      ← trajectory_chat_path（对话轨迹）
```

- 两个文件通过 `nexus_server_req_id` 关联，文件名只差 `-chat` 后缀
- 日期目录 `{yyyyMMdd}_{HH}` 按推理完成的小时粒度分桶
- **导出打包时**，如果 payload 中存在这两个路径，后端会自动将 ceph 文件复制到 `trajectory/` 子目录一起打包

**导出后的 tar.gz 结构（含轨迹文件）：**

```
export_{id}.tar.gz
├── bench_name__evId__task_taskId.jsonl.gz       ← 评测数据
├── bench_name__evId__task_taskId.jsonl.gz
├── trajectory/                                    ← agent 轨迹文件（自动收集）
│   ├── req-18808_240295000_xxx.jsonl             ← 执行轨迹
│   ├── req-18808_240295000_xxx-chat.json         ← 对话轨迹
│   └── ...
```

##### run_input_data —— Agent 运行输入配置

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `agent_name` | String | Agent 名称：`swe_agent` / `search_agent` / `browser_gym_agent` / `terminus2` |
| `agent_config` | Object | Agent 配置：`max_iterations`（最大迭代次数）、`completion_retry_on_tool_error`（工具错误重试次数） |
| `run_id` | String | 运行 ID，通常为 instance_id（如 `django__django-11555`） |
| `task_name` | String | 任务名称 |
| `task_config` | Object | 任务配置，含 `instance`（题目实例）、`eval_timeout_seconds`、`dataset_path` 等 |
| `imports` | List\<String\> | Agent 依赖的 Python 模块列表 |
| `openai_clients.main.config` | Object | LLM 调用配置：`base_url`、`model`、`max_tokens`、`temperature`、`chat_template_kwargs` 等 |
| `runtime_provider_name` | String | 运行时提供方（如 `gongfeng`） |
| `runtime_provider_config` | Object | 运行时配置：`image`（Docker 镜像）、`runtime_image`、`api_base_url`、`request_timeout` |
| `tool_configs` | Object | 工具配置（通常为空 `{}`） |

##### run_output_data —— Agent 运行输出结果

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `success` | Boolean | 运行是否成功 |
| `error` | String/null | 错误信息，成功时为 null |
| `agent_output` | Object | Agent 输出：`exit_status`（如 `submitted`）、`iterations`（实际迭代次数） |
| `task_output` | Object | 任务输出，因 benchmark 而异 |
| `task_output.resolved` | Boolean | （SWE-Bench）是否解决问题 |
| `task_output.score` | Number | （SWE-Bench）得分 |
| `task_output.patch_text` | String | （SWE-Bench）生成的补丁 |
| `task_output.test_output` | String | （SWE-Bench）测试执行输出 |
| `task_output.apply_log` | String | （SWE-Bench）补丁应用日志 |
| `task_output.termination_reason` | String | 终止原因 |
| `processor_results` | Object | 处理器结果，含 `token_counter` |
| `cache_stats` | null | 缓存统计（通常为 null） |

##### token_usage —— LLM 调用 token 统计

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `main.prompt_tokens` | Integer | 总 prompt token 数（所有轮次累计） |
| `main.completion_tokens` | Integer | 总 completion token 数 |
| `main.cached_prompt_tokens` | Integer | 缓存命中的 prompt token 数 |
| `main.non_cached_prompt_tokens` | Integer | 未缓存的 prompt token 数 |
| `main.reasoning_tokens` | Integer | 推理 token 数 |
| `main.llm_call_count` | Integer | LLM 调用总次数（即 agent 执行的轮次/step 数） |

> **关键映射关系**：`token_usage.main.llm_call_count` = `run_output_data.agent_output.iterations`（都是 agent 实际执行的步数）

##### Agent 轨迹相关的额外 payload 字段

agent 类 benchmark 的 payload 还包含一些额外字段（非 `trial_details` 内部）：

| 字段 | 类型 | 出现 benchmark | 说明 |
|:---|:---|:---|:---|
| `control_params` | String(JSON) | swe_bench / webarena 等 | Agent 控制参数：`nexus_timeout`、`nexus_server_call_mode`(async)、`max_failed_retries` |
| `exit_status` | String | swe_bench | Agent 退出状态：`submitted` / `error` / `timeout` |
| `error` | String | swe_bench | 错误信息（成功时为空字符串） |
| `version` | String | swe_bench | Agent 版本（如 `3.0`） |
| `instance_id` | String | swe_bench | 实例 ID（如 `django__django-11555`） |
| `base_commit` | String | swe_bench | 基准 commit SHA |
| `patch` | String | swe_bench | Agent 生成的 diff 补丁 |
| `test_patch` | String | swe_bench | 测试补丁 |
| `image` | String | swe_bench | Docker 镜像地址 |
| `workdir` | String | swe_bench | 容器工作目录（如 `/testbed`） |
| `dataset_type` | String | swe_bench | 数据集类型标识（如 `swe-bench`） |
| `FAIL_TO_PASS` | String(JSON) | swe_bench | 需要通过的失败测试用例列表 |
| `PASS_TO_PASS` | String(JSON) | swe_bench | 需要保持通过的测试用例列表 |

##### Python 工具函数：提取轨迹文件路径

```python
def extract_trajectory_paths(payload):
    """
    从 payload 中提取 agent 轨迹文件的 ceph 路径
    
    返回:
        tuple: (trajectory_path, trajectory_chat_path)，无轨迹时返回 (None, None)
    """
    trial_details = payload.get('trial_details')
    if not isinstance(trial_details, dict):
        return None, None
    trajectory_info = trial_details.get('trajectory_info')
    if not isinstance(trajectory_info, dict):
        return None, None
    return (
        trajectory_info.get('trajectory_path'),
        trajectory_info.get('trajectory_chat_path')
    )


def collect_all_trajectory_paths(jsonl_gz_path):
    """
    从 .jsonl.gz 文件中收集所有 agent 轨迹 ceph 路径
    
    返回:
        list of dict: [{questionId, trajectory_path, trajectory_chat_path, agent_name}, ...]
    """
    import gzip, json
    results = []
    with gzip.open(jsonl_gz_path, 'rt', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            p = record['payload']
            traj_path, chat_path = extract_trajectory_paths(p)
            if traj_path or chat_path:
                td = p.get('trial_details', {})
                results.append({
                    'questionId': record.get('questionId'),
                    'agent_name': td.get('run_input_data', {}).get('agent_name'),
                    'trajectory_path': traj_path,
                    'trajectory_chat_path': chat_path,
                    'iterations': td.get('run_output_data', {}).get('agent_output', {}).get('iterations'),
                    'llm_call_count': td.get('token_usage', {}).get('main', {}).get('llm_call_count'),
                })
    return results
```

#### 1.3.12 Agent 轨迹文件内部格式

导出包中的 `trajectory/` 子目录包含两类文件，均使用 **zstd 压缩**（`.jsonl.zst`）：

| 文件类型 | 命名格式 | 压缩 | 内部格式 | 说明 |
|:---|:---|:---|:---|:---|
| 执行轨迹 | `req-{taskId}_{evId}_{questionId}_{uuid}.jsonl.zst` | zstd | JSONL（每行一个 OpenTelemetry span） | Agent 逐步执行的工具调用与观测结果 |
| 对话轨迹 | `req-{taskId}_{evId}_{questionId}_{uuid}-chat.json` | 无 | JSON（完整消息列表） | Agent 的完整多轮对话历史 |

> **解压依赖**：执行轨迹文件需要 `zstandard` 库（`pip install zstandard`）进行解压。

##### 执行轨迹文件结构（OpenTelemetry Span 格式）

执行轨迹文件采用 **OpenTelemetry span** 格式，每行是一个 span 事件（JSON 对象），记录 Agent 执行的完整决策链。

**span 通用字段：**

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `type` | String | span 事件类型：`START`（开始）/ `UPDATE`（更新/输出）/ `END`（结束） |
| `span_id` | String | span 唯一标识（Base64 编码） |
| `time_unix_nano` | Long | 时间戳（纳秒级 Unix 时间） |
| `parent_span_id` | String | 父 span ID（仅 `START` 类型有，用于构建调用树） |
| `trace_id` | String | 整体 trace ID（同一次 Agent 执行的所有 span 共享） |
| `name` | String | span 名称（仅 `START` 类型有），标识执行动作类型 |
| `attributes` | Object | 属性字典，包含 `inputs`/`outputs` 等关键数据 |
| `events` | List/null | span 事件列表 |
| `status` | Object/null | span 状态 |

**span name 枚举与含义：**

| span name | 类型 | 说明 | 出现频率 |
|:---|:---|:---|:---|
| `swe_task` | task | 整体任务 span（顶层），包含任务输入配置 | 1 次 |
| `swe_agent` | agent | Agent 实例 span，包含用户 prompt（问题描述） | 1 次 |
| `openai_completion` | llm | LLM 推理调用，`attributes.inputs` 含完整 messages，对应 `UPDATE` 含模型响应 | N 次（= llm_call_count） |
| `tool.swe_agent.bash` | tool | 执行 bash 命令（如 `find`/`grep`/`cat`/`cd`/`python` 等） | 多次 |
| `tool.swe_agent.str_replace_editor` | tool | 查看/编辑代码文件（`view`/`create`/`str_replace`/`insert`） | 多次 |
| `tool.swe_agent.review_submit` | tool | 提交最终 patch，Agent 认为修复完成 | 0~2 次 |

> **不同 Agent 的 span name 前缀不同**：`tool.swe_agent.*`（SWE-agent）、`tool.search_agent.*`（搜索 Agent）、`tool.browser_gym_agent.*`（浏览器 Agent）等。

**典型 Agent 执行流程（span 序列）：**

```
1. swe_task START          → 任务初始化，attributes.inputs 含完整任务配置
2. swe_agent START         → Agent 启动，attributes.inputs.agent_input.user_prompt 含问题描述
3. openai_completion START → 第 1 次 LLM 调用，attributes.inputs.messages 含 system+user prompt
4. openai_completion UPDATE → LLM 响应，attributes.outputs 含模型输出和 tool_calls
5. tool.swe_agent.bash START → 执行 bash 命令探索代码
6. tool.swe_agent.bash UPDATE → 命令执行结果
7. openai_completion START → 第 2 次 LLM 调用（含之前的工具结果作为上下文）
8. ... 重复 LLM 调用 → 工具执行 循环 ...
9. tool.swe_agent.review_submit START → Agent 提交 patch
10. swe_agent END          → Agent 执行结束
11. swe_task END           → 任务完成，attributes.outputs 含最终得分和 exit_status
```

**START span 的 `attributes` 关键字段：**

| span name | attributes 关键字段 | 说明 |
|:---|:---|:---|
| `swe_task` | `inputs.task_input` | 任务配置（含 dataset_path、instance 等） |
| `swe_agent` | `inputs.agent_input.user_prompt` | Agent 收到的问题描述（issue 原文） |
| `openai_completion` | `inputs.messages` | 发给 LLM 的完整消息列表（含 system prompt + 历史对话） |
| `openai_completion` | `_client_key` | LLM 客户端标识（通常为 `main`） |
| `tool.*` | `inputs.params` | 工具调用参数（如 `{command: "view", path: "/testbed/src/..."}` ） |
| `tool.*` | `name` | 工具全名（如 `swe_agent/str_replace_editor`） |

**UPDATE span 的 `attributes` 关键字段：**

| span name | attributes 关键字段 | 说明 |
|:---|:---|:---|
| `openai_completion` | `outputs.choices[0].message.content` | LLM 文本响应 |
| `openai_completion` | `outputs.choices[0].message.tool_calls` | LLM 决定调用的工具列表 |
| `openai_completion` | `outputs.usage` | 本次调用的 token 消耗 |
| `tool.*` | `outputs.output` | 工具执行结果（如命令输出、文件内容等） |
| `swe_task`（最终） | `outputs.exit_status` | Agent 退出状态 |
| `swe_task`（最终） | `outputs.score` | 最终得分 |

##### Python 工具函数：解析轨迹文件

```python
import zstandard
import json
import collections

def load_trajectory(zst_path):
    """
    加载 .jsonl.zst 格式的 Agent 执行轨迹文件
    
    参数:
        zst_path: .jsonl.zst 文件路径
    返回:
        list of dict: 每个元素为一个 OpenTelemetry span 事件
    """
    dctx = zstandard.ZstdDecompressor()
    with open(zst_path, 'rb') as fh:
        data = dctx.decompress(fh.read())
    lines = data.decode('utf-8').strip().split('\n')
    return [json.loads(line) for line in lines]


def analyze_trajectory(spans):
    """
    分析 Agent 轨迹的工具使用模式和执行统计
    
    参数:
        spans: load_trajectory() 返回的 span 列表
    返回:
        dict: {
            total_spans: int,            # 总 span 数
            start_spans: int,            # START span 数
            span_names: Counter,         # 各 span name 的出现次数
            llm_calls: int,              # LLM 调用次数（openai_completion 数量）
            tool_calls: int,             # 工具调用次数
            tool_breakdown: Counter,     # 各工具的调用次数
            total_time_seconds: float,   # 总执行时间（秒）
        }
    """
    span_names = collections.Counter()
    tool_breakdown = collections.Counter()
    first_time = None
    last_time = None
    
    for span in spans:
        t = span.get('time_unix_nano', 0)
        if t:
            if first_time is None or t < first_time:
                first_time = t
            if last_time is None or t > last_time:
                last_time = t
        
        if span.get('type') == 'START':
            name = span.get('name', '?')
            span_names[name] += 1
            if name.startswith('tool.'):
                tool_breakdown[name] += 1
    
    llm_calls = span_names.get('openai_completion', 0)
    tool_calls = sum(v for k, v in span_names.items() if k.startswith('tool.'))
    total_time = (last_time - first_time) / 1e9 if first_time and last_time else 0
    
    return {
        'total_spans': len(spans),
        'start_spans': sum(1 for s in spans if s.get('type') == 'START'),
        'span_names': span_names,
        'llm_calls': llm_calls,
        'tool_calls': tool_calls,
        'tool_breakdown': tool_breakdown,
        'total_time_seconds': round(total_time, 1),
    }


def extract_llm_conversations(spans):
    """
    从轨迹中提取 LLM 的输入输出序列（按时间顺序）
    
    返回:
        list of dict: [{
            'step': int,                   # 第几次 LLM 调用
            'input_messages_count': int,   # 输入消息数量
            'output_content': str,         # LLM 输出文本（截取前500字符）
            'tool_calls': list,            # LLM 决定调用的工具
            'tokens': dict,                # token 消耗
        }]
    """
    results = []
    current_step = 0
    pending_llm = None
    
    for span in spans:
        if span.get('type') == 'START' and span.get('name') == 'openai_completion':
            current_step += 1
            attrs = span.get('attributes', {})
            inputs = attrs.get('inputs', {})
            messages = inputs.get('messages', [])
            pending_llm = {
                'step': current_step,
                'input_messages_count': len(messages),
                'output_content': None,
                'tool_calls': [],
                'tokens': {},
            }
        
        elif span.get('type') == 'UPDATE' and pending_llm:
            attrs = span.get('attributes', {})
            outputs = attrs.get('outputs', {})
            
            if isinstance(outputs, dict):
                choices = outputs.get('choices', [])
                if choices:
                    msg = choices[0].get('message', choices[0].get('delta', {}))
                    if isinstance(msg, dict):
                        content = msg.get('content', '')
                        if content:
                            pending_llm['output_content'] = content[:500]
                        tcs = msg.get('tool_calls', [])
                        if tcs:
                            pending_llm['tool_calls'] = [
                                tc.get('function', {}).get('name', '?') for tc in tcs
                            ]
                
                usage = outputs.get('usage', {})
                if usage:
                    pending_llm['tokens'] = usage
            
            results.append(pending_llm)
            pending_llm = None
    
    return results


def batch_analyze_trajectories(trajectory_dir, records):
    """
    批量分析轨迹文件，与评测结果关联
    
    参数:
        trajectory_dir: trajectory/ 目录路径
        records: 评测数据记录列表（从 jsonl.gz 加载）
    返回:
        list of dict: 每条记录包含评测结果 + 轨迹分析
    """
    import os
    
    # 建立 questionId → record 映射
    qid_map = {r['questionId']: r for r in records}
    
    # 建立轨迹文件 → questionId 映射
    traj_files = {}
    for f in os.listdir(trajectory_dir):
        if not f.endswith('.jsonl.zst'):
            continue
        # 文件名: req-{taskId}_{evId}_{questionId}_{uuid}.jsonl.zst
        parts = f.replace('.jsonl.zst', '').split('_')
        if len(parts) >= 4:
            try:
                qid = int(parts[2])
                traj_files[qid] = os.path.join(trajectory_dir, f)
            except ValueError:
                continue
    
    results = []
    for qid, record in qid_map.items():
        p = record['payload']
        entry = {
            'questionId': qid,
            'avg_score': p.get('avg_score', 0),
            'exit_status': p.get('exit_status', 'N/A'),
            'instance_id': p.get('instance_id', ''),
        }
        
        # 从 trial_details 获取 token 信息
        td = p.get('trial_details', {})
        tu = td.get('token_usage', {}).get('main', {})
        entry['completion_tokens'] = tu.get('completion_tokens', 0)
        entry['llm_call_count'] = tu.get('llm_call_count', 0)
        
        # 解析对应的轨迹文件
        if qid in traj_files:
            try:
                spans = load_trajectory(traj_files[qid])
                traj_analysis = analyze_trajectory(spans)
                entry['traj_tool_breakdown'] = dict(traj_analysis['tool_breakdown'])
                entry['traj_total_time'] = traj_analysis['total_time_seconds']
            except Exception:
                entry['traj_tool_breakdown'] = {}
                entry['traj_total_time'] = 0
        
        results.append(entry)
    
    return results
```

##### 评测报告生成最佳实践

###### 通用评测报告（适用所有 benchmark）

**1. 整体概览**
- 总 case 数、成功/失败数、acc（ignore 和 fixed 模式）、得分分布（满分/零分/部分分比例）
- 错误率（推理失败 + 评判失败）、截断率

**2. 标准下钻维度分析**

> **核心下钻维度**：`task_lv1`、`task_lv2`、`task_lv3`、`language`、`source`、`difficulty`
> 数据中存在的字段会自动同步过来，分析时先检测当前数据包含哪些维度，再按存在的维度逐一下钻。

对每个存在的维度，列出：case 数、均分、零分数、零分率，按零分率降序排列。

```python
# 自动检测并按可用维度下钻
DRILL_DOWN_DIMS = ['task_lv1', 'task_lv2', 'task_lv3', 'language', 'source', 'difficulty']

def detect_available_dims(payloads):
    """检测当前数据中实际存在哪些下钻维度"""
    available = []
    for dim in DRILL_DOWN_DIMS:
        has_value = any(
            p.get(dim) is not None and p.get(dim) != ''
            for p in payloads
        )
        if has_value:
            available.append(dim)
    return available

def analyze_by_dim(payloads, dim):
    """按指定维度分组分析"""
    from collections import defaultdict
    groups = defaultdict(list)
    for p in payloads:
        val = p.get(dim, '未知')
        if val is None or val == '':
            val = '未知'
        groups[val].append(p)
    
    results = []
    for val, cases in groups.items():
        total = len(cases)
        ok_scores = [p.get('avg_score', 0) for p in cases if not is_failed_case(p)]
        avg = sum(ok_scores) / len(ok_scores) if ok_scores else 0
        zeros = sum(1 for s in ok_scores if s == 0)
        results.append({
            'value': val, 'total': total, 'avg': avg,
            'zeros': zeros, 'zero_rate': zeros / total if total > 0 else 0
        })
    return sorted(results, key=lambda x: -x['zero_rate'])
```

**3. 交叉分析**
- 选取 2-3 个最有价值的维度做交叉表（如 `task_lv1` × `difficulty`、`task_lv1` × `language`）
- 重点关注零分率异常高的交叉组合

**4. 薄弱环节 TOP N**
- 按 `task_lv2` 或 `task_lv3` 维度，列出零分率最高的 10-15 个细分场景
- 样本数 ≥ 3 的才列入排行，避免小样本偏差

**5. Token 消耗分析**
- 满分 vs 零分 case 的 avg_completion_tokens 对比
- 按 token 区间分段统计零分率（如 0-10k / 10k-20k / 20k-50k / 50k+）
- 识别 token 与得分的相关性趋势

**6. 异常 case 分析**
- 评判失败 case（score 含 -100000）详情
- 零分 case 抽样（5-10 条），展示：题目/标准答案/模型回答/评判结果
- 截断 case（finish_reason=length）统计

**7. 总结与建议**
- 核心发现（按严重程度排序）
- 改进建议（按优先级 P0/P1/P2 排列）

###### Agent 评测报告（额外章节）

当用户需要分析 Agent 类评测数据（如 SWE-bench）时，在通用报告基础上，额外增加以下章节：

**1. 整体概览**
- 各评测集的 Resolve Rate（acc）、case 总数、通过/未通过
- exit_status 分布（submitted / tool_error / max_iterations）

**2. exit_status × 得分交叉分析**
- submitted 的通过率 vs tool_error 的通过率（tool_error 通常通过率极低）
- max_iterations 的 case 分析（Agent 迭代超限）

**3. Token 消耗分析**
- 满分 case vs 零分 case 的 completion_tokens / llm_call_count 对比
- 异常高 token 消耗的 case 识别（可能陷入循环）

**4. 按 Repository/项目分析**
- 各 repo 的 Resolve Rate 排行（找出最弱 repo）
- 不同语言/框架的表现对比

**5. 轨迹深度分析（可选）**
- 工具调用分布：bash vs editor vs submit 的比例
- 满分/零分 case 的工具使用模式差异
- LLM 调用次数与得分的相关性

**6. 异常 case 识别**
- tool_error case 的具体错误原因
- token > 阈值的异常 case
- 评判失败（score=-100000）的 case

---

## 第二层：异常值与边界条件

### 2.1 推理 / 评判失败

| 状态组合 | 含义 | score 表现 | 数据处理建议 |
|:---|:---|:---|:---|
| `infer_success` + `judge_success` | 正常 case | `avg_score` 为有效值 [0, 1] | 正常参与统计 |
| `infer_failed` + `*` | 推理失败 | `score` 中包含 `-100000` 标记值 | **必须过滤**，不参与 acc 计算 |
| `*` + `judge_failed` | 评判失败 | `score` 中包含 `-100000` 标记值 | **必须过滤**，不参与 acc 计算 |

**失败 case 识别方式**（后端实际使用的判断逻辑）：

```sql
-- StarRocks 侧过滤失败 case
WHERE score NOT LIKE '%-100000%'
```

```python
# Python 侧判断
def is_failed_case(payload):
    """判断是否为失败 case"""
    score = payload.get('score', [[]])
    if not score or not score[0]:
        return True
    # 失败 case 的 score 包含 -100000 标记
    for round_scores in score:
        for s in round_scores:
            if s is not None and s <= -100000:
                return True
    return False
```

### 2.2 截断（Truncation）

当模型输出达到 `max_tokens` 上限时被截断：

| 判断依据 | 正常 | 截断 |
|:---|:---|:---|
| `usage[0][0].finish_reason` | `"stop"` | `"length"` |
| `payload.avg_finish_reason_length` | `0` | `1` |
| `avg_completion_tokens` 表现 | 通常远小于 max_tokens | 接近或等于 max_tokens |

截断可能导致**评分偏低**（模型回答不完整），需要在分析时注意区分。

### 2.3 score 矩阵的特殊值

| 情况 | score 值 | avg_score | 说明 |
|:---|:---|:---|:---|
| 满分 | `[[1]]` | `1` 或 `1.0` | 完全正确 |
| 零分 | `[[0]]` | `0` 或 `0` (int) | 完全错误 |
| 部分分 | `[[0.2737]]` | `0.2737` | 如 chrF 等连续评分 |
| 失败标记 | `[[-100000]]` | 负数或极小值 | **异常值，必须过滤** |

**注意**：`avg_score` 类型不固定，可能是 `int`（`0`、`1`）或 `float`（`0.2737`）。

### 2.4 min_score 字段的特殊性

`min_score` 在当前数据中**始终为 0**（100% 的 case）。这是因为：
- 对于 `1x1` 的 score 矩阵（单轮单次），`min_score` = `score[0][0]` 或 0
- 在后端的 `hy_eval_case_metrics` 表中，`min_score >= 0` 被用作过滤失败 case 的条件

**因此 `min_score` 不建议用于分析，请用 `avg_score` 或 `score` 矩阵。**

### 2.5 payload 中的空值字段

| 字段 | 空值率 | 原因 |
|:---|:---|:---|
| `task_id` | 100% | payload 内部冗余，不存储，使用顶层 `taskId` |
| `exercise_version_id` | 100% | payload 内部冗余，不存储，使用顶层 `exerciseVersionId` |
| `prompt` | ~84% | 仅部分 dataset 使用 prompt 模板 |
| `split` | ~84% | 仅部分 dataset 有 split 标记 |
| `doc` | ~74% | 仅部分 dataset 提供原始文档 |

### 2.6 exerciseId = -1

顶层字段 `exerciseId` 可能为 `-1`，表示该 case 未绑定特定评测集。这是一个合法值，不需要过滤。

### 2.7 pass_num 的含义

`pass_num` **不是**简单的 0/1 布尔值，它表示在多次评估中通过的次数：
- 不同 dataset 的 `pass_num` 含义可能不同
- `consistency` 数据集：`pass_num=2` 表示一致性检查中有 2 次通过
- 不建议直接用 `pass_num > 0` 判断是否通过，请使用 `avg_score > 0`

### 2.8 百分制与小数制

**重要**：后端有两种 score 口径：

| 口径 | 说明 | 范围 |
|:---|:---|:---|
| payload 中的 `avg_score` | 原始小数制 | [0, 1] |
| 后端 `ExerciseResult.scores` 中的 `acc` | 百分制，后端展示时 ÷100 | [0, 100] → [0, 1] |

**导出文件中的 `avg_score` 已经是小数制 [0, 1]，无需再除以 100。**

---

## 第三层：指标计算方法

### 3.1 acc（准确率）

**定义**：所有成功 case 的 `avg_score` 的均值。

```python
def calc_acc(cases):
    """
    计算 acc（准确率）
    
    参数:
        cases: list of payload dicts
    返回:
        float: acc 值，范围 [0, 1]
    """
    success_scores = []
    for p in cases:
        # 过滤失败 case
        if is_failed_case(p):
            continue
        score = p.get('avg_score')
        if score is not None and score >= 0:
            success_scores.append(score)
    
    if not success_scores:
        return 0.0
    return sum(success_scores) / len(success_scores)
```

**后端实现对照**（`MultimodalEvaluationInsightServiceImpl`）：
- 过滤条件：`score NOT LIKE '%-100000%'`
- 分子：成功 case 的 `avg_score` 之和
- 分母：成功 case 的数量

#### acc 的 missing_value_mode 变体

| 模式 | 分母 | 说明 |
|:---|:---|:---|
| `ignore`（默认） | 成功 case 数量 | 跳过失败 case，不计入分母 |
| `fixed` | **全部 case 数量**（含失败） | 失败 case 视为 0 分，计入分母 |

```python
def calc_acc_fixed(cases):
    """
    acc（missing_value_mode=fixed）
    失败 case 纳入分母，分数视为 0
    """
    total = len(cases)
    if total == 0:
        return 0.0
    
    score_sum = 0.0
    for p in cases:
        if not is_failed_case(p):
            score = p.get('avg_score', 0)
            if score is not None and score >= 0:
                score_sum += score
        # 失败 case 贡献 0 分但计入分母
    
    return score_sum / total
```

### 3.2 bon_acc（Best-of-N 准确率）

**定义**：使用 `max_score`（而非 `avg_score`）计算的准确率——即每道题取最高一次评分。

```python
def calc_bon_acc(cases):
    """
    计算 bon_acc（Best-of-N 准确率）
    使用 max_score 而非 avg_score
    """
    success_scores = []
    for p in cases:
        if is_failed_case(p):
            continue
        max_score = p.get('max_score')
        if max_score is not None and max_score >= 0:
            success_scores.append(max_score)
    
    if not success_scores:
        return 0.0
    return sum(success_scores) / len(success_scores)
```

### 3.3 avg_completion_token（平均输出长度）

**定义**：所有成功 case 的 `avg_completion_tokens` 均值。

```python
def calc_avg_completion_token(cases):
    """
    计算平均输出 token 长度
    """
    tokens = []
    for p in cases:
        if is_failed_case(p):
            continue
        ct = p.get('avg_completion_tokens')
        if ct is not None:
            tokens.append(ct)
    
    if not tokens:
        return 0.0
    return sum(tokens) / len(tokens)
```

### 3.4 truncation_rate（截断率）

**定义**：因 `max_tokens` 限制被截断的 case 占比。

```python
def calc_truncation_rate(cases):
    """
    计算截断率
    finish_reason='length' 表示被截断
    """
    total = 0
    truncated = 0
    for p in cases:
        if is_failed_case(p):
            continue
        total += 1
        # 方式1：用 avg_finish_reason_length 字段
        if p.get('avg_finish_reason_length', 0) > 0:
            truncated += 1
        # 方式2：用 usage 中的 finish_reason
        # usage = p.get('usage', [[]])
        # if usage and usage[0]:
        #     for u in usage[0]:
        #         if isinstance(u, dict) and u.get('finish_reason') == 'length':
        #             truncated += 1
    
    if total == 0:
        return 0.0
    return truncated / total
```

### 3.5 按 dataset 分组计算（per-benchmark 指标）

每个 `.jsonl.gz` 文件中包含多个 `dataset`（子评测集），实际使用时通常需要**按 dataset 分组**计算指标：

```python
import gzip
import json
from collections import defaultdict

def analyze_by_dataset(jsonl_gz_path):
    """
    按 dataset 分组计算各项指标
    
    返回 dict: {dataset_name: {acc, bon_acc, avg_tokens, truncation_rate, total, success, failed}}
    """
    groups = defaultdict(list)
    
    with gzip.open(jsonl_gz_path, 'rt', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            p = record['payload']
            ds = p.get('dataset', 'unknown')
            groups[ds].append(p)
    
    results = {}
    for ds, cases in groups.items():
        total = len(cases)
        failed = sum(1 for p in cases if is_failed_case(p))
        success = total - failed
        
        results[ds] = {
            'total': total,
            'success': success,
            'failed': failed,
            'acc': calc_acc(cases),
            'bon_acc': calc_bon_acc(cases),
            'avg_completion_token': calc_avg_completion_token(cases),
            'truncation_rate': calc_truncation_rate(cases),
        }
    
    return results
```

### 3.6 跨 Task 对比

一个 tar.gz 中的多个 `task_{taskId}.jsonl.gz` 代表**不同模型在相同题目集上的评测结果**。对比方式：

```python
def compare_tasks(tar_gz_path):
    """
    跨 Task（模型）对比
    相同 questionId 的 case 可以直接配对对比
    """
    import tarfile
    
    task_results = {}
    
    with tarfile.open(tar_gz_path, 'r:gz') as tar:
        for member in tar.getmembers():
            # 提取 taskId
            # 文件名格式: task_{taskId}.jsonl.gz
            task_id = member.name.replace('task_', '').replace('.jsonl.gz', '')
            
            f = tar.extractfile(member)
            task_results[task_id] = analyze_by_dataset_from_fileobj(f)
    
    return task_results
```

### 3.7 指标汇总公式对照表

| 指标名 | 公式 | 分子 | 分母 | 数据字段 |
|:---|:---|:---|:---|:---|
| **acc** | Σ(avg_score) / N_success | 成功 case 的 avg_score 之和 | 成功 case 数 | `payload.avg_score` |
| **acc (fixed)** | Σ(avg_score) / N_total | 成功 case 的 avg_score 之和 | 全部 case 数 | `payload.avg_score` |
| **bon_acc** | Σ(max_score) / N_success | 成功 case 的 max_score 之和 | 成功 case 数 | `payload.max_score` |
| **avg_completion_token** | Σ(tokens) / N_success | 成功 case 的 completion_tokens 之和 | 成功 case 数 | `payload.avg_completion_tokens` |
| **truncation_rate** | N_truncated / N_success | finish_reason=length 的 case 数 | 成功 case 数 | `payload.avg_finish_reason_length` 或 `payload.usage[0][0].finish_reason` |
| **error_rate** | N_failed / N_total | 失败 case 数 | 全部 case 数 | `payload.__infer_status__` + `payload.__judge_status__` |

### 3.8 通用工具函数

```python
import gzip
import json

def is_failed_case(payload):
    """判断是否为失败 case（推理或评判失败）"""
    # 方式1：检查状态字段
    if payload.get('__infer_status__') != 'infer_success':
        return True
    if payload.get('__judge_status__') != 'judge_success':
        return True
    # 方式2：检查 score 中的 -100000 标记（更可靠）
    score = payload.get('score', [[]])
    if not score or not score[0]:
        return True
    for round_scores in score:
        for s in round_scores:
            if s is not None and s <= -100000:
                return True
    return False

def load_jsonl_gz(path):
    """加载 .jsonl.gz 文件，返回 (record_list, payload_list)"""
    records = []
    payloads = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
            payloads.append(record['payload'])
    return records, payloads

def load_tar_gz(tar_path):
    """
    加载 .tar.gz 归档，返回 {filename: (records, payloads)} 和轨迹文件列表
    """
    import tarfile, io
    result = {}
    trajectory_files = {}  # {filename: bytes}
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.name.endswith('.jsonl.gz'):
                f = tar.extractfile(member)
                records, payloads = [], []
                with gzip.open(io.BytesIO(f.read()), 'rt', encoding='utf-8') as gz:
                    for line in gz:
                        record = json.loads(line)
                        records.append(record)
                        payloads.append(record['payload'])
                result[member.name] = (records, payloads)
            elif member.name.startswith('trajectory/'):
                # 轨迹文件：trajectory/req-xxx.jsonl 或 trajectory/req-xxx-chat.json
                f = tar.extractfile(member)
                if f:
                    trajectory_files[member.name] = f.read()
    return result, trajectory_files
```
