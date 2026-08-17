# BenchProd 评估结果底线问题分析

<!-- ROUTE_KEYWORDS
底线分析, 底线问题, 底线检测, 下限分析, 下限问题, 下限检测,
floor analysis, floor anomaly, bottom line, bottom line analysis,
复读检测, 机械复读, 重复检测, REPEAT, repeat,
乱码检测, 不可读字符, UNREADABLE_CHAR, 乱码, garbled,
markdown错误, markdown问题, MARKDOWN_ERR, markdown渲染,
think标签异常, think标签, THINK_TAG_ILLEGAL, think tag,
tool call异常, tool call泄漏, TOOL_CALL_ANOMALY, tool call leakage,
红线问题, RED_LINE_PROB, 输出质量检测, 模型输出异常,
空回答, EMPTY_RESP, 空思考, EMPTY_THINK,
底线case, 底线badcase, 异常case, 底线问题集合,
评测底线, 评估底线, 评测质量, 模型下限
-->

## 📌 本模块 Agent 行为契约（最高优先级）

> 本段是 SKILL.md 路由进入本文档后的统领规则，覆盖本文档所有章节。

- **何时进入**：用户问"底线分析 / 底线问题 / 底线检测 / 下限检测 / 复读 / 乱码 / think 标签异常 / tool call 异常 / markdown 错误 / 模型输出异常"等任一意图。
- **进入后必做**：
  1. 底线分析依赖 Insight/Case 导出数据（数据来源参见 `evaluation_api.md` 的导出工具 + `export_data_schema.md` 字段定义）。若用户只给 task_id 并要求“看有没有复读/乱码/think 标签异常/底线问题”，先走 `submit_taiji_eval_case_export` → `get_taiji_eval_insight_export_status` 拿到导出包；不要先查任务详情或导出历史。
  2. **检测前必须与用户确认三项参数**（见 §1.2 用户交互确认），确认后再执行检测。
  3. 检测逻辑必须完全对齐 `scripts/detect_floor_anomaly_concurrent.py`，不得自行新增/删减检测规则。
  4. 输出结论必须基于检测脚本的真实统计结果，不得凭直觉总结。
- **严禁**：把"底线检测"和"错误归因"混为一谈——底线检测是纯规则检测（不依赖 LLM），错误归因走 `error_diagnosis.md`。
- **失败处理**：缺数据时直接告知用户"先导出评测数据"并指引到 `evaluation_api.md`，**不**自行尝试别的接口替代。

---

> 本文档供 AI Skill 使用，定义 BenchProd 评估结果底线问题分析规范：三大类 16 种下限问题的纯规则检测方法、输入输出格式、Agent Bench 与 Standard Bench 的差异化处理、bench 级并发执行、指标汇总与 badcase 集合输出。

---

## 概述

### 分析目标

对模型评估结果执行**纯规则下限检测**（不依赖 LLM），快速发现模型输出中的三大类底线问题：

| # | 大类 | 代号 | 适用范围 | 子类型数 |
|:--|:-----|:-----|:---------|:---------|
| 1 | Tool Call 格式异常 | `TOOL_CALL_ANOMALY` | 仅 Agent 类 Bench | 7 种 |
| 2 | 红线问题 | `RED_LINE_PROB` | 全部 Bench | 3 种 |
| 3 | Think 标签异常 | `THINK_TAG_ILLEGAL` | 全部 Bench | 6 种 |

### 底线分析二阶段流水线

```
Phase 1: 数据加载 + 用户交互确认  →  确定检测范围和参数
Phase 2: 执行检测 + 输出汇总      →  floor_anomaly_metrics.json + badcase 文件
         （支持 --workers N 以 bench 为粒度并发执行）
```

### 与其他分析模块的关系

| 分析模块 | 关注点 | 方法 | 依赖 LLM |
|:---------|:-------|:-----|:---------|
| 底线分析（本文档） | 模型输出的**格式/结构**缺陷 | 纯规则匹配 | ❌ |
| 下钻分析（`drilldown_analysis.md`） | 指标的**维度分布**差异 | 数值聚合 | ❌ |
| 错误归因（`error_diagnosis.md`） | 错误样本的**语义根因** | LLM 诊断 | ✅ |

---

## 第一层：数据加载与参数确认

### 1.1 输入数据

数据来源与 [drilldown_analysis.md](./drilldown_analysis.md) 一致，使用相同的导出数据目录结构。

```
{model_dir}/
├── {benchName}__{exerciseVersionId}__task_{taskId}.jsonl.gz    ← 标准 Bench 数据
├── swe_bench_verified__xxx__task_{taskId}.jsonl.gz             ← Agent Bench 数据
├── trajectory/                                                  ← Agent 轨迹文件
│   ├── req-{taskId}_{evId}_{questionId}_{uuid}.jsonl.zst
│   └── ...
└── ...
```

**核心字段依赖**（详见 [export_data_schema.md](../export_data_schema.md)）：

| 字段路径 | 用途 | 必有 |
|:---------|:-----|:-----|
| `payload.responses[0][0]` | 模型回答文本 | ✅ |
| `payload.thinking_responses[0][0]` | 模型思考过程 | ✅ |
| `payload.usage[0][0].finish_reason` | 结束原因（stop/length） | ✅ |
| `payload.model_input[0][0].chat_template_kwargs` | 推理模式（think/nothink） | ✅ |
| `payload.__infer_status__` | 推理状态 | ✅ |
| `payload.trial_details.trajectory_info` | Agent 轨迹路径（Agent Bench） | 仅 Agent |
| `payload.tool_calls` / `payload.responses[0][0].tool_calls` | Tool call 数据（Agent Bench） | 仅 Agent |

### 1.2 用户交互确认（必须）

开始检测前，**必须与用户确认**以下三项参数：

| # | 确认项 | 说明 | 默认值 |
|:--|:------|:----|:------|
| 1 | **要检测的评测集** | 选择 bench 范围：全部 / 指定 bench 名称列表 | `ALL`（全部 bench） |
| 2 | **下限问题类型** | 选择检测大类：全部 / 指定检测类型组合 | `ALL`（全部三大类） |
| 3 | **是否输出底线 case 集合** | 是否将命中的 badcase 写入 JSONL 文件 | `是`（输出 badcase） |

**参数取值说明**：

- **评测集**：`ALL` 表示检测所有 `.jsonl.gz` 文件；也可指定 bench 名称列表（如 `CLBench,prbench_finance`），按文件名前缀匹配
- **下限问题类型**：可选值为 `TOOL_CALL_ANOMALY` / `RED_LINE_PROB` / `THINK_TAG_ILLEGAL` / `ALL`，逗号分隔
  - 注意：`TOOL_CALL_ANOMALY` 仅对 Agent 类 Bench 生效，非 Agent Bench 会自动跳过
- **底线 case 集合输出**：选择是否输出 → 对应脚本的 `--write_mode` 参数（`per_bench` 或 `none`）

### 1.3 Agent Bench 识别

以下文件名前缀识别为 Agent 类 Bench，支持 `TOOL_CALL_ANOMALY` 检测：

```python
# BenchProd-V4 and V4-Core
AGENT_BENCH_PREFIXES = [
    "swe_bench_verified",
    "swe_bench_multilingual",
    "swe_bench_pro",
    "terminal_bench_2_0",
    "hyeval_browsecomp-zh",
    "hyeval_widesearch",
    "hyeval_finsearchcomp-t2_and_t3",
    "hyeval_seal-0",
    "webarena",
    "hyeval_browsecomp-subset-150",
    "tau2",
    "hy_swe_max",
    "ucb_smoking",
    "terminal_bench_2_1_xml_nexusv1",
    "heval_bfclv4",
    "e-bench",
    "apex-agents",
    "mcp_universe",
    "toolathlon",
    "mcp_atlas",
    "nl2repo",
    "hyeval_browsecomp-full",
    "hyeval_deepsearchqa",
    "skillsbench",
    "hle-agent",
    "hyeval_frontier_science_research-agent",
]

def is_agent_bench(filename):
    """判断文件是否为 Agent 类 Bench"""
    basename = os.path.basename(filename)
    return any(basename.startswith(p) for p in AGENT_BENCH_PREFIXES)
```

### 1.4 Think 模式自动探测

脚本自动从非 Agent Bench 数据的第一条记录中探测 think 模式，用于控制 `THINK_TAG_ILLEGAL` 的检测子类型：

```python
def determine_think_mode(data_dir):
    """从非 agent 文件的第一条数据确定 think 模式"""
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith(".jsonl.gz"):
            continue
        if is_agent_bench(f):
            continue
        filepath = os.path.join(data_dir, f)
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as fh:
                line = fh.readline()
                if line:
                    record = json.loads(line)
                    payload = record.get("payload", {})
                    if isinstance(payload, str):
                        payload = json.loads(payload)
                    mi = payload.get("model_input", [[]])
                    if mi and isinstance(mi, list) and mi[0]:
                        m0 = mi[0][0] if isinstance(mi[0], list) and mi[0] else mi[0]
                        if isinstance(m0, dict):
                            kte = m0.get("chat_template_kwargs", {})
                            re_val = kte.get("reasoning_effort", "")
                            if re_val and str(re_val).lower() == "no_think":
                                return "nothink"
                            else:
                                return "think"
        except Exception:
            continue
    return "think"
```

**Think 模式 × 检测子类型的适用关系**：

| Think 模式 | 适用子类型 | 不适用子类型 |
|:-----------|:----------|:------------|
| `think` | `EMPTY_THINK`, `MISSING_</think>`, `REDUNDANT_</think>`, `REDUNDANT_<think>`, `EMPTY_RESP` | `NOTHINK_HAS_<think>` |
| `nothink` | `NOTHINK_HAS_<think>`, `EMPTY_RESP` | `EMPTY_THINK`, `MISSING_</think>`, `REDUNDANT_</think>`, `REDUNDANT_<think>` |

---

## 第二层：三大类底线问题检测规则

### 2.1 TOOL_CALL_ANOMALY — Tool Call 格式异常（仅 Agent Bench）

对 Agent Bench 的每轮 LLM 调用（从 trajectory 文件逐轮解析）执行 7 种检测。

**前置过滤**：`finish_reason` 为 `length` / `content_filter` 时，跳过该轮检测。

| # | 子类型 | 检测逻辑 | 触发条件 |
|:--|:-------|:---------|:---------|
| 1 | `TOOL_CALL_LEAKAGE` | content/reasoning 中泄漏 tool call 原始标记 | 文本中包含 `</tool_calls>`, `<tool_sep>`, `<tool_call>...</tool_call>`, `<function=...>...</function>` 等标记（兼容 hash 变体如 `<tool_call:6124c78e>`） |
| 2 | `REASONING_ONLY` | 只有推理内容，无 content 也无 tool_calls | `reasoning` 非空 且 `content` 为空 且 `tool_calls` 为空 |
| 3 | `TOOL_CALLS_FIELD_EMPTY` | finish_reason 指示有 tool call 但实际为空 | `finish_reason == "tool_calls"` 且 `tool_calls` 列表为空 |
| 4 | `STOP_WITHOUT_TOOL_CALL` | 定义了工具且要求必须调用，但模型 stop 时未调用 | `finish_reason == "stop"` 且无 `tool_calls` 且有 `tools` 定义 且 `tool_choice == "required"` |
| 5 | `TOOL_CALL_DESPITE_NONE` | 禁止调用工具但模型仍发起了 tool call | `tool_calls` 非空 且 `tool_choice == "none"` |
| 6 | `JSON_PARSE_ERROR` | tool_calls 参数 JSON 解析失败 | `tool_calls[i].function.arguments` 是字符串但 `json.loads()` 抛异常 |
| 7 | `HALLUCINATED_TOOL` | 模型调用了不在 tools 列表中的函数 | `tool_calls[i].function.name` 不在 `tools` 定义的函数名集合中 |

**Tool Call 标记检测正则**：

```python
_RE_TOOL_CALLS_CLOSE_AT_END = re.compile(r'</tool_calls(?::[0-9a-fA-F]+)?>\s*$', re.IGNORECASE)
_RE_TOOL_SEP = re.compile(r'<tool_sep(?::[0-9a-fA-F]+)?>', re.IGNORECASE)

_TOOLCALL_PAIR_PATTERNS = [
    re.compile(r'<tool_call(?::[0-9a-fA-F]+)?>.*?</tool_call(?::[0-9a-fA-F]+)?>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<arg_key(?::[0-9a-fA-F]+)?>.*?</arg_key(?::[0-9a-fA-F]+)?>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<arg_value(?::[0-9a-fA-F]+)?>.*?</arg_value(?::[0-9a-fA-F]+)?>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<function=[^>]+>.*?</function>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<parameter=[^>]+>.*?</parameter>', re.IGNORECASE | re.DOTALL),
]

def _contains_toolcall_markup(text):
    """检测文本中是否包含 tool call 原始标记（兼容 hash 变体如 <tool_call:6124c78e>）。"""
    if not text:
        return False
    if _RE_TOOL_CALLS_CLOSE_AT_END.search(text):
        return True
    if _RE_TOOL_SEP.search(text):
        return True
    return any(p.search(text) for p in _TOOLCALL_PAIR_PATTERNS)
```

### 2.2 RED_LINE_PROB — 红线问题（全部 Bench）

#### 2.2.1 REPEAT — 机械复读

检测模型 response 和 thinking 中的机械重复片段。

**核心参数**：

| 参数 | 值 | 说明 |
|:-----|:---|:-----|
| `REPEAT_UNIT_MIN` | 2 | 最小重复单元长度（字符数） |
| `REPEAT_UNIT_MAX` | 20 | 最大重复单元长度 |
| `REPEAT_SCAN_MAX_CHARS` | 120000 | 扫描最大字符数（超长文本取首尾各半） |
| `REPEAT_MIN_RUN_LENGTH` | 8 | `finish_reason=length` 时的最低重复次数阈值 |
| `REPEAT_MIN_RUN_NORMAL` | 50 | 其他情况的最低重复次数阈值 |

**策略差异**：

| finish_reason | 重复次数阈值 | 位置要求 | 原因 |
|:-------------|:-----------|:---------|:-----|
| `length` | ≥ 8 次 | 无限制 | 复读导致截断是典型下限问题，使用低阈值 |
| 其他 | ≥ 50 次 | 仅检测文本尾部 20% 区域 | 只有结尾明显的复读机现象才算异常 |

**有效性过滤**：仅由数字/符号/空白组成的重复单元不计入（通过 `_is_meaningful_unit` 过滤）。

```python
def detect_mechanical_repeat(text, finish_reason=""):
    """检测机械复读。返回 (matched, sample_dict_or_None)。"""
    if not text:
        return False, None
    n_text = len(text)
    if n_text > REPEAT_SCAN_MAX_CHARS:
        half = REPEAT_SCAN_MAX_CHARS // 2
        scan_text = text[:half] + text[-half:]
    else:
        scan_text = text

    min_run = REPEAT_MIN_RUN_LENGTH if finish_reason == "length" else REPEAT_MIN_RUN_NORMAL
    patterns = _get_repeat_patterns(min_run)

    for n, pat in patterns:
        for m in pat.finditer(scan_text):
            unit = m.group(1)
            if not _is_meaningful_unit(unit):
                continue
            run = (m.end() - m.start()) // n
            if finish_reason != "length":
                tail_start = int(len(scan_text) * 0.8)
                if m.end() < tail_start:
                    continue
            unit_preview = unit if len(unit) <= 40 else unit[:37] + "..."
            return True, {"unit": unit_preview, "unit_len": len(unit), "run": run, "pos": m.start()}
    return False, None
```

#### 2.2.2 UNREADABLE_CHAR — 不可读字符/乱码

检测模型回答中的乱码和异常字符。**检测前先剥除代码块内容**（避免代码中的特殊字符误报）。

| # | 检测项 | 正则/方法 | 说明 |
|:--|:------|:---------|:-----|
| 1 | U+FFFD 替换字符 | `text.count("\uFFFD")` | 原始字节已丢失，不可恢复 |
| 2 | GBK 典型乱码 | `锟斤拷\|锘垮\|锘锛` | 典型 GBK 编码错误 |
| 3 | Latin-1 Mojibake | `[\u00C0-\u00FF][\u0080-\u00BF]` | UTF-8 被当作 Latin-1 读取 |
| 4 | Unicode 双向控制字符 | `[\u202A-\u202E\u2066-\u2069]` | Trojan Source 风险 |
| 5 | 零宽/方向标记 | `[\u200B\u200C\u200E\u200F\u2060]` | 零宽字符 |
| 6 | 文件中部 BOM | `\uFEFF` 不在起始位置 | 异常 BOM |
| 7 | NUL 字符 | `\x00` | 空字符 |
| 8 | 非常规控制字符 | `ord < 0x20` 且不在 `{0x09,0x0A,0x0B,0x0C,0x0D}` 中 | 非标准控制符 |
| 9 | 私有区字符 | `[\uE000-\uF8FF\uFFF0-\uFFFF\uFE30-\uFE4F]` | 占比超 2% 时触发 |

#### 2.2.3 MARKDOWN_ERR — Markdown 渲染问题

检测模型回答中的 Markdown 格式错误。**检测前先剥除代码块和数学块**（避免误报）。

| # | 检测项 | 说明 |
|:--|:------|:-----|
| 1 | 代码块未闭合 | `` ``` `` 不配对，fenced code block 未正确关闭 |
| 2 | 标题级别超过 6 级 | `#{7,}` 后跟空格 |
| 3 | 标题 `#` 后缺少空格 | `#{1,6}` 后直接跟非空格字符（排除 hashtag、CSS 选择器、步骤号等） |
| 4 | 链接/图片格式不完整 | `[...]()` 括号不配对 |
| 5 | Setext 标题异常加粗 | `---` 前缺少空行导致前一行被渲染为 `<h2>` |

**白名单排除规则**（降低误召回）：

- Setext 检测排除：前一行是列表项、整行粗体装饰、YAML front matter、下一行是 ATX 标题或有序列表

### 2.3 THINK_TAG_ILLEGAL — Think 标签异常（全部 Bench）

**前置过滤**：`finish_reason=length` 时跳过全部 think 标签检测（截断导致的不完整不算异常）。

**标签变体兼容**：`<think:6124c78e>` / `</think:6124c78e>` 与标准 `<think>` / `</think>` 等效。

| # | 子类型 | 适用模式 | 检测逻辑 |
|:--|:-------|:---------|:---------|
| 1 | `EMPTY_THINK` | think | `reasoning_content` 为空（think 模式下理应有思考内容） |
| 2 | `MISSING_</think>` | think | `response` 为空且 `finish_reason=stop`（推断全部内容归入 reasoning，缺少闭合标签） |
| 3 | `REDUNDANT_</think>` | think | `response` 中包含 `</think>` 标签（平台按首个 `</think>` 切分，残留说明有多余标签） |
| 4 | `REDUNDANT_<think>` | think | `reasoning` 或 `response` 中包含 `<think>` 标签（平台已消费首个，后续为多余） |
| 5 | `EMPTY_RESP` | think/nothink | `response` 为空但 `finish_reason=stop` |
| 6 | `NOTHINK_HAS_<think>` | nothink | nothink 模式下 `response` 中出现 `<think>` 或 `</think>` 标签 |

```python
_THINK_BEGIN_TOKENS = ["<think>", "<think:6124c78e>"]
_THINK_END_TOKENS = ["</think>", "</think:6124c78e>"]

def detect_think_tag_issues(response_text, thinking_text, think_mode, finish_reason=""):
    """
    检测 think 标签异常。
    think_mode: 'think' or 'nothink'
    返回 list of (subtype, desc)
    """
    issues = []
    if not response_text and not thinking_text:
        return issues
    if finish_reason == "length":
        return issues

    combined = (thinking_text or "") + (response_text or "")

    if think_mode == "nothink":
        if _contains_any_think_begin(combined) or _contains_any_think_end(combined):
            issues.append(("NOTHINK_HAS_<think>",
                           "nothink模式下response中出现<think>或</think>标签"))
        if finish_reason == "stop" and (not response_text or not response_text.strip()):
            issues.append(("EMPTY_RESP", "模型RESPONSE为空但finish_reason=stop"))
        return issues

    # think 模式
    if not thinking_text or not thinking_text.strip():
        issues.append(("EMPTY_THINK", "think模式下reasoning_content为空"))

    if not response_text or not response_text.strip():
        if finish_reason == "stop":
            issues.append(("MISSING_</think>",
                           "think模式下response为空(推断缺少</think>闭合)"))
            issues.append(("EMPTY_RESP", "模型RESPONSE为空但finish_reason=stop"))
        return issues

    if _contains_any_think_end(response_text):
        issues.append(("REDUNDANT_</think>",
                       "response中残留</think>标签(存在多个)"))

    if _contains_any_think_begin(combined):
        issues.append(("REDUNDANT_<think>",
                       "reasoning/response中残留<think>标签(存在多个)"))

    return issues
```

---

## 第三层：Standard Bench 与 Agent Bench 差异化处理

### 3.1 Standard Bench 检测流程

对非 Agent 类 Bench 的每条记录（case），从 payload 提取 response 和 thinking，执行 `RED_LINE_PROB` + `THINK_TAG_ILLEGAL` 检测。

**跳过条件**：`payload.__infer_status__ == "infer_failed"` 时跳过该 case。

**文本提取**：

```python
def first_text(field):
    """从 [[str, ...], ...] 嵌套结构中取第一个非空字符串。"""
    if not field or not isinstance(field, list):
        return ""
    for group in field:
        if isinstance(group, list):
            for x in group:
                if x and str(x).strip():
                    return str(x)
        elif group and str(group).strip():
            return str(group)
    return ""

def extract_finish_reason(payload):
    """从 usage 字段提取 finish_reason（取第一个 pass）。"""
    usages = payload.get("usage", [])
    if not usages:
        return ""
    u = usages[0]
    if isinstance(u, list):
        u = u[0] if u else {}
    if not isinstance(u, dict):
        return ""
    return (u.get("finish_reason") or "").lower()
```

**检测顺序**：

1. 提取 `response_text = first_text(payload.get("responses"))`
2. 提取 `thinking_text = first_text(payload.get("thinking_responses"))`
3. 提取 `finish_reason = extract_finish_reason(payload)`
4. 若 `RED_LINE_PROB` 在检测范围内：
   - 检测 REPEAT（response 和 thinking 分别检测，结果合并为同一 badcase）
   - 检测 UNREADABLE_CHAR（仅 response）
   - 检测 MARKDOWN_ERR（仅 response）
5. 若 `THINK_TAG_ILLEGAL` 在检测范围内：
   - 执行 `detect_think_tag_issues(response_text, thinking_text, think_mode, finish_reason)`

### 3.2 Agent Bench 检测流程

对 Agent 类 Bench，基于 trajectory 文件**逐轮**解析 LLM 调用输出，执行 `TOOL_CALL_ANOMALY` + `RED_LINE_PROB` + `THINK_TAG_ILLEGAL` 检测。

**跳过条件**：
- `payload.__infer_status__ == "infer_failed"` 时跳过
- 无法解析 trajectory 文件路径时跳过

**轨迹文件解析**：

从 `payload.trial_details.trajectory_info.trajectory_path` 获取轨迹文件路径（支持 `.jsonl` 和 `.jsonl.zst` 格式），采用**流式逐行解压 + 逐行解析**方式读取，避免将整个轨迹文件加载到内存导致 OOM。

```python
def resolve_trajectory_path(payload):
    """从 payload 中解析实际可读的轨迹文件路径"""
    td = payload.get("trial_details")
    if not isinstance(td, dict):
        return None
    ti = td.get("trajectory_info")
    if not isinstance(ti, dict):
        return None
    tp = ti.get("trajectory_path", "")
    if not tp:
        return None
    if os.path.isfile(tp):
        return tp
    if os.path.isfile(tp + ".zst"):
        return tp + ".zst"
    return None

def _iter_lines_from_path(traj_path):
    """
    流式逐行读取轨迹文件，支持 .jsonl 和 .jsonl.zst。
    使用 stream_reader + TextIOWrapper 避免将整个解压内容加载到内存。
    返回 (line_iterator, closables) — closables 需在读取完毕后关闭。
    """
    if traj_path.endswith(".zst"):
        dctx = zstandard.ZstdDecompressor()
        fh = open(traj_path, "rb")
        reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        return text_stream, (text_stream, reader, fh)
    else:
        fh = open(traj_path, "r", encoding="utf-8")
        return fh, (fh,)
```

**流式解析核心优化**：逐行读取时通过关键词快速过滤（`'openai_completion' not in line and 'choices' not in line`），跳过不含 LLM 调用信息的行，仅对匹配行执行 `json.loads`，峰值内存 ≈ 解压缓冲区(~64KB) + 累积的 parsed spans 结构化数据。

**逐轮检测规则**：

| 检测项 | 中间轮次 | 最后一轮 |
|:-------|:---------|:---------|
| `TOOL_CALL_ANOMALY` (7 种) | ✅ 全部检测 | ✅ 全部检测 |
| `REPEAT` | ✅ 检测 content + reasoning | ✅ 检测 content + reasoning |
| `UNREADABLE_CHAR` | ❌ 不检测 | ✅ 仅最后一轮 |
| `MARKDOWN_ERR` | ❌ 不检测 | ✅ 仅最后一轮 |
| `THINK_TAG_ILLEGAL` (标签泄漏类) | ✅ 仅 `REDUNDANT_</think>`, `REDUNDANT_<think>`, `NOTHINK_HAS_<think>` | ✅ 全部 6 种 |

**Agent Badcase 统计粒度**：

Agent Bench 的 badcase 按 `(anomaly_type, anomaly_subtype)` 维度聚合，每种子类型生成一条 badcase，包含以下统计信息：

| 字段 | 说明 |
|:-----|:-----|
| `anomaly_counts` | 该子类型在所有轮次中出现的总次数 |
| `error_requests` | 命中该子类型的请求（轮次）数 |
| `error_steps` | 命中该子类型的步骤（去重后）数 |
| `retry_requests` | 重试请求数 |
| `retry_steps` | 重试步骤数 |
| `total_requests` | 总请求（轮次）数 |
| `total_steps` | 总步骤数 |
| `anomaly_details` | 每个命中轮次的详细信息列表 |

---

## 第四层：脚本调用与输出

### 4.1 调用方式

```bash
# 默认使用单进程版（适用于大多数场景）
python3 scripts/detect_floor_anomaly.py \
  --data_dir /path/to/exported_data \
  --output_dir /path/to/output \
  --task_id 27491 \
  [--debug]                          # debug 模式每个 bench 只跑前 100 条
  [--detect_types TYPE1,TYPE2]       # 指定检测类型，默认 ALL
  [--bench_name NAME1,NAME2]         # 指定 bench 名称过滤，默认 ALL
  [--write_mode per_bench|none]      # badcase 写入模式，默认 per_bench

# 大量 bench 时可使用并发版（按 bench 粒度 ProcessPoolExecutor）
python3 scripts/detect_floor_anomaly_concurrent.py \
  --data_dir /path/to/exported_data \
  --output_dir /path/to/output \
  --task_id 27491 \
  [--workers N]                      # 并发工作进程数，默认 1（串行）
  [--debug] [--detect_types TYPE1,TYPE2] [--bench_name NAME1,NAME2] [--write_mode per_bench|none]
```

**参数对照表**：

| 参数 | 对应用户确认项 | 取值 |
|:-----|:-------------|:-----|
| `--data_dir` | 评测数据目录 | 导出数据所在路径 |
| `--output_dir` | 输出目录 | badcase 和 metrics 的输出路径 |
| `--task_id` | 评测任务 ID | 整数 |
| `--bench_name` | 要检测的评测集 | 逗号分隔的 bench 名称，或 `ALL` |
| `--detect_types` | 下限问题类型 | `TOOL_CALL_ANOMALY,RED_LINE_PROB,THINK_TAG_ILLEGAL` 或 `ALL` |
| `--write_mode` | 是否输出底线 case 集合 | `per_bench`（输出） / `none`（不输出） |
| `--workers` | 并发工作进程数 | 整数，默认 `1`（串行）；`>1` 时以 bench 为粒度并发执行（`ProcessPoolExecutor`） |

### 4.1.1 并发执行模式

当 `--workers > 1` 时，脚本以 **bench 为粒度**并发执行检测：

- 使用 `ProcessPoolExecutor`（多进程，规避 GIL）创建工作进程池
- 每个 bench 文件作为一个独立任务提交到进程池
- 各 worker 执行 `process_file()` 后返回 `(total, subtype_counts, all_badcases)` 三元组
- **主进程统一处理**：badcase 写入文件、全局指标聚合和结果展示均在主进程中完成，避免多进程写文件的竞态问题
- 使用 `as_completed` 获取已完成任务的结果，逐个汇总

**注意**：大规模 Agent Bench（如 `hle-agent`，2000+ case 且每个 case 都有大型轨迹文件）可能产生"长尾"效应——单个 bench 的处理时间远超其他 bench，即使有多个 worker 也需等待该 bench 完成。

### 4.2 输出文件

#### 4.2.1 指标汇总文件

`{output_dir}/floor_anomaly_metrics.json` — 三大类 × 各子类型的检出数和占比：

```json
[
  {
    "type": "TOOL_CALL_ANOMALY",
    "measures": [
      {"name": "TOOL_CALL_LEAKAGE", "count": 5, "ratio": 0.001, "desc": "content/reasoning 中泄漏 tool call 标记"},
      {"name": "HALLUCINATED_TOOL", "count": 0, "ratio": 0.0, "desc": "调用了不在 tools 列表中的函数"},
      {"name": "JSON_PARSE_ERROR", "count": 2, "ratio": 0.0004, "desc": "tool_calls arguments JSON 解析失败"},
      {"name": "REASONING_ONLY", "count": 0, "ratio": 0.0, "desc": "只有推理内容, 无 content 也无 tool_calls"},
      {"name": "STOP_WITHOUT_TOOL_CALL", "count": 0, "ratio": 0.0, "desc": "有 tools 但 stop 时未调用"},
      {"name": "TOOL_CALLS_FIELD_EMPTY", "count": 0, "ratio": 0.0, "desc": "finish_reason=tool_calls 但 tool_calls 为空"},
      {"name": "TOOL_CALL_DESPITE_NONE", "count": 0, "ratio": 0.0, "desc": "tool_choice=none 但仍调用了工具"}
    ]
  },
  {
    "type": "RED_LINE_PROB",
    "measures": [
      {"name": "REPEAT", "count": 12, "ratio": 0.0024, "desc": "机械复读(response或thinking)"},
      {"name": "UNREADABLE_CHAR", "count": 3, "ratio": 0.0006, "desc": "模型输出含不可读/乱码字符"},
      {"name": "MARKDOWN_ERR", "count": 8, "ratio": 0.0016, "desc": "Markdown 渲染问题"}
    ]
  },
  {
    "type": "THINK_TAG_ILLEGAL",
    "measures": [
      {"name": "EMPTY_THINK", "count": 1, "ratio": 0.0002, "desc": "think模式下reasoning_content为空"},
      {"name": "MISSING_</think>", "count": 0, "ratio": 0.0, "desc": "think模式下缺少</think>闭合"},
      {"name": "REDUNDANT_</think>", "count": 4, "ratio": 0.0008, "desc": "response中残留</think>标签"},
      {"name": "REDUNDANT_<think>", "count": 2, "ratio": 0.0004, "desc": "reasoning/response中残留<think>标签"},
      {"name": "EMPTY_RESP", "count": 0, "ratio": 0.0, "desc": "模型RESPONSE为空但finish_reason=stop"},
      {"name": "NOTHINK_HAS_<think>", "count": 0, "ratio": 0.0, "desc": "不适用(当前为think模式)"}
    ]
  }
]
```

> **注意**：不适用当前 think 模式的子类型，`count` 和 `ratio` 强制为 0，`desc` 标注"不适用"。

#### 4.2.2 Badcase 文件（可选）

当 `--write_mode` 为 `per_bench` 时，按子类型拆分输出：

```
{output_dir}/
├── floor_anomaly_metrics.json         ← 指标汇总
├── REPEAT.jsonl                       ← 机械复读 badcase
├── UNREADABLE_CHAR.jsonl              ← 不可读字符 badcase
├── MARKDOWN_ERR.jsonl                 ← Markdown 错误 badcase
├── EMPTY_THINK.jsonl                  ← 空思考 badcase
├── TOOL_CALL_LEAKAGE.jsonl            ← Tool Call 泄漏 badcase
├── ...                                ← 其他命中的子类型
```

每个 `.jsonl` 文件每行一条 JSON，结构为**完整原始 record + `anomaly_results` 字段**：

**Standard Bench Badcase 结构**：

```json
{
  "taskId": 27491,
  "exerciseVersionId": 1348,
  "questionId": 238805301,
  "id": 57103060237220,
  "productType": "LLM",
  "exerciseId": -1,
  "date": "2026-03-16",
  "payload": { "...原始 payload..." },
  "anomaly_results": {
    "anomaly_type": "RED_LINE_PROB",
    "anomaly_subtype": "REPEAT",
    "desc": "模型输出中存在机械复读",
    "is_agent_task": false,
    "anomaly_details": [
      {
        "reason": "回答机械复读: 重复单元'请参考...'连续重复15次",
        "snippet": "...文本片段..."
      }
    ]
  }
}
```

**Agent Bench Badcase 结构**：

```json
{
  "taskId": 27491,
  "exerciseVersionId": 852,
  "questionId": 240295000,
  "payload": { "...原始 payload..." },
  "anomaly_results": {
    "anomaly_type": "TOOL_CALL_ANOMALY",
    "anomaly_subtype": "TOOL_CALL_LEAKAGE",
    "desc": "content/reasoning 中泄漏 tool call 标记",
    "is_agent_task": true,
    "anomaly_counts": 3,
    "error_requests": 3,
    "error_steps": 2,
    "anomaly_details": [
      {"caused_retry": true, "is_retry": false, "span_id": "abc123", "step_id": 5},
      {"caused_retry": false, "is_retry": true, "span_id": "def456", "step_id": 5},
      {"caused_retry": false, "is_retry": false, "span_id": "ghi789", "step_id": 8}
    ],
    "retry_requests": 1,
    "retry_steps": 1,
    "total_requests": 25,
    "total_steps": 20
  }
}
```

---

## 第五层：结果展示规则

### 5.1 指标展示

执行完成后，按以下格式展示汇总结果：

```
===========================================================
SUMMARY
===========================================================
Total cases: 5000
Think mode: think
Total anomalies: 35 (0.70%)

  TOOL_CALL_ANOMALY (仅Agent Bench): 7
    TOOL_CALL_LEAKAGE: 5 (0.100%)
    JSON_PARSE_ERROR: 2 (0.040%)

  RED_LINE_PROB (全部Bench): 23
    REPEAT: 12 (0.240%)
    MARKDOWN_ERR: 8 (0.160%)
    UNREADABLE_CHAR: 3 (0.060%)

  THINK_TAG_ILLEGAL (全部Bench): 5
    REDUNDANT_</think>: 4 (0.080%)
    EMPTY_THINK: 1 (0.020%)
```

### 5.2 展示规则

1. **排序**：大类内各子类型按 count 降序排列
2. **过滤**：count = 0 的子类型不展示（不适用的子类型除外，需标注"不适用"）
3. **百分比**：ratio 以百分比展示，保留 3 位小数
4. **scope 标注**：`TOOL_CALL_ANOMALY` 标注"仅 Agent Bench"，其余标注"全部 Bench"
5. **重点标记**：ratio > 1% 的子类型应重点标注（建议用 ⚠️ 或加粗）

### 5.3 结论生成

基于检测结果自动生成结论，按以下优先级排列：

| 优先级 | 条件 | 结论模板 |
|:------|:-----|:---------|
| P0 | 任一子类型 ratio > 1% | "⚠️ 严重：{subtype} 检出率 {ratio}%（{count}/{total}），建议立即排查" |
| P1 | 任一子类型 ratio 在 0.1%~1% | "⚡ 关注：{subtype} 检出率 {ratio}%（{count}/{total}）" |
| P2 | 所有子类型 ratio < 0.1% | "✅ 底线质量良好，所有检测项均低于 0.1%" |

---

## 第六层：子类型速查总表

| 大类 | 子类型 | 说明 | 适用 Bench | 检测方式 |
|:-----|:-------|:-----|:----------|:---------|
| `TOOL_CALL_ANOMALY` | `TOOL_CALL_LEAKAGE` | content/reasoning 中泄漏 tool call 标记 | Agent | 正则匹配 |
| `TOOL_CALL_ANOMALY` | `REASONING_ONLY` | 只有推理内容，无 content 也无 tool_calls | Agent | 字段判空 |
| `TOOL_CALL_ANOMALY` | `TOOL_CALLS_FIELD_EMPTY` | finish_reason=tool_calls 但 tool_calls 为空 | Agent | 字段矛盾 |
| `TOOL_CALL_ANOMALY` | `STOP_WITHOUT_TOOL_CALL` | 有 tools 且 tool_choice=required 但未调用 | Agent | 字段矛盾 |
| `TOOL_CALL_ANOMALY` | `TOOL_CALL_DESPITE_NONE` | tool_choice=none 但仍调用了工具 | Agent | 字段矛盾 |
| `TOOL_CALL_ANOMALY` | `JSON_PARSE_ERROR` | tool_calls arguments JSON 解析失败 | Agent | JSON parse |
| `TOOL_CALL_ANOMALY` | `HALLUCINATED_TOOL` | 调用了不在 tools 列表中的函数 | Agent | 名称比对 |
| `RED_LINE_PROB` | `REPEAT` | 机械复读 | 全部 | 正则匹配 |
| `RED_LINE_PROB` | `UNREADABLE_CHAR` | 不可读字符/乱码 | 全部 | 多种正则 |
| `RED_LINE_PROB` | `MARKDOWN_ERR` | Markdown 渲染问题 | 全部 | 语法解析 |
| `THINK_TAG_ILLEGAL` | `EMPTY_THINK` | think 模式下 reasoning 为空 | 全部 (think) | 字段判空 |
| `THINK_TAG_ILLEGAL` | `MISSING_</think>` | think 模式下缺少闭合标签 | 全部 (think) | 字段判空 |
| `THINK_TAG_ILLEGAL` | `REDUNDANT_</think>` | response 中残留 `</think>` | 全部 (think) | 字符串搜索 |
| `THINK_TAG_ILLEGAL` | `REDUNDANT_<think>` | 残留多余 `<think>` | 全部 (think) | 字符串搜索 |
| `THINK_TAG_ILLEGAL` | `EMPTY_RESP` | response 为空但 finish_reason=stop | 全部 | 字段判空 |
| `THINK_TAG_ILLEGAL` | `NOTHINK_HAS_<think>` | nothink 模式下出现 think 标签 | 全部 (nothink) | 字符串搜索 |

---

## 第七层：关键经验与避坑

### 1. Payload 二次解析

数据中 `payload` 字段可能是 JSON 字符串（而非 dict），必须检查并二次解析：

```python
payload = record.get("payload", {})
if isinstance(payload, str):
    payload = json.loads(payload)
```

### 2. 嵌套 List 提取

`responses` / `thinking_responses` / `usage` 都是嵌套 list 结构 `[[item]]`，使用 `first_text()` 工具函数安全提取。`responses[0][0]` 可能是 dict（含 `content` / `reasoning_content`）或 str。

### 3. finish_reason 截断过滤

`finish_reason=length` 时需跳过 `THINK_TAG_ILLEGAL` 的全部检测（截断导致的不完整不算异常）。`REPEAT` 检测则使用不同阈值（低阈值 8 次 vs 高阈值 50 次）。

### 4. Agent Bench 轨迹文件依赖

Agent Bench 的 `TOOL_CALL_ANOMALY` 检测**必须依赖 trajectory 文件**。如果轨迹文件不存在或不可读，该 case 的全部检测将被跳过。确保 trajectory 目录与 `.jsonl.gz` 文件在同一导出包中。

轨迹文件采用**流式逐行读取**（`_iter_lines_from_path`），支持 `.jsonl` 和 `.jsonl.zst` 格式。对 `.zst` 文件使用 `zstandard.ZstdDecompressor().stream_reader()` + `io.TextIOWrapper` 流式解压，避免将整个文件加载到内存。大型 Agent Bench（如 `hle-agent`）的轨迹文件总量可达数十 GB，全量加载会导致 OOM。

### 5. 字符编码安全

输出 JSONL 时需递归清理 surrogate 字符，避免写入失败：

```python
def _sanitize(obj):
    """递归清理字符串中的 surrogate 字符"""
    if isinstance(obj, str):
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj
```

### 6. gzip 文件读取

BenchProd 导出数据为 `.jsonl.gz` 格式：

```python
import gzip
def read_jsonl_gz(filepath):
    records = []
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
```

### 7. zstandard 依赖

Agent 轨迹文件为 `.jsonl.zst` 格式，需要 `zstandard` 库（`pip install zstandard`）。脚本在缺少该库时会打印警告并跳过 Agent 轨迹解析。

### 8. 并发执行注意事项

使用 `--workers > 1` 开启 bench 级并发时：
- 每个 worker 进程独立执行 `process_file()`，返回 `(total, subtype_counts, all_badcases)` 三元组
- **主进程统一写入** badcase 文件和聚合指标，无需担心多进程写文件竞态
- 大型 Agent Bench（如 `hle-agent`，2000+ case × 每个 case 都有轨迹文件，总轨迹量可达 25+ GB）可能产生"长尾"效应：即使并发，也需等待最慢的 bench 完成
- 建议根据机器 CPU 核数和 I/O 带宽设置合理的 worker 数（通常 4~8 即可）
