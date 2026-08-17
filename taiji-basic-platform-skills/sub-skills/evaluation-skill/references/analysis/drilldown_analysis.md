# BenchProd 评估结果下钻分析

<!-- ROUTE_KEYWORDS
下钻分析, drilldown, 下钻, bench下钻, topic下钻, 维度下钻,
评估结果分析, 评测结果分析, 评估下钻, 评测下钻,
bench对比, topic对比, 模型对比分析
-->

## 📌 本模块 Agent 行为契约（最高优先级）

> 本段是 SKILL.md 路由进入本文档后的统领规则，覆盖本文档所有章节。

- **何时进入**：用户问"下钻 / 评测对比 / bench 对比 / topic 下钻 / 维度下钻 / 评测结果分析"等任一意图。
- **进入后必做**：
  1. 下钻分析必须先有 Insight 导出数据（数据来源参见 `evaluation_api.md` 的 Insight 导出工具 + `export_data_schema.md` 字段定义）。**禁止**直接基于评测任务列表凭空编造下钻表。
  2. 用户没说要对比的 bench/topic 维度时，先列出当前 Insight 数据中可用的维度，让用户挑选后再下钻。
  3. 输出结论必须基于 Insight 真实字段（accuracy / topic 分布 / 等），不得凭直觉总结。
- **严禁**：把"错误归因"和"下钻分析"混为一谈——错误样本归因走 `error_diagnosis.md`，本文档只做指标维度对比。
- **失败处理**：缺数据时直接告知用户"先导出 Insight 数据"并指引到 `evaluation_api.md`，**不**自行尝试别的接口替代。

---

> 本文档供 AI Skill 使用，定义 BenchProd 生产评测集的下钻分析规范：Topic 分组、Bench 归属、维度下钻字段映射、指标计算方法和核心结论生成规则。

---

## 第一层：数据结构与 Topic 分组

### 1.1 输入数据结构

每个模型的评估结果存储在一个独立目录中（以 taskId 区分），内含按评测集拆分的 `.jsonl.gz` 文件：

```
{model_dir}/
├── {benchName}__{exerciseVersionId}__task_{taskId}.jsonl.gz
├── {benchName}__{exerciseVersionId}__task_{taskId}.jsonl.gz
├── trajectory/
├── README.md
├── log.txt
└── ...
```

- **文件命名**：`{exerciseName}__{exerciseVersionId}__task_{taskId}.jsonl.gz`
- **数据格式**：JSONL（每行一个 JSON 对象），GZIP 压缩，UTF-8 编码
- **每行结构**：固定 8 字段（见 [export_data_schema.md](../export_data_schema.md#12-jsonl-顶层字段固定-8-字段顺序不变)），核心数据在 `payload` 字段

### 1.2 Topic 分组（7 Topic × 37 Bench）

通过文件名中的 `exerciseVersionId`（evId）将评测集归入 7 个 Topic：

| Topic | 显示名 | Bench 列表（evId） |
|:------|:------|:-----------------|
| `complex_IF` | 复杂指令 | HYeval.复杂指令(1348), IFBench(685), InverseIF(681), ToBv3(506) |
| `knowledge_qa` | 知识问答 | HYeval.知识问答(1344), HYQA(959), SimpleQA(957), BrowseComp(1427), HYKR(973), Chinese_SimpleQA(1180), HY_Poem(974) |
| `logic` | 逻辑推理 | HYeval.逻辑推理(1342), ARC-AGI-v1(942)★, ARC-AGI-v2(917)★, BBEH(1075), HeCheng_Logic(1088), Surge_Logic(1089), PRBench_Finance(1073)★, PRBench_Legal(1072)★ |
| `longcontext` | 长文 | HYeval.长文(1341), LongBench_V2(1249), CLBench(884), BrowsCompLong(894) |
| `math` | 数学 | HYeval.数学(1346), IMO_AnswerBench(877), AMO_Bench(878), NAnti_Misc(875), Math_T_v2(1141) |
| `multiturn` | 多轮 | HYeval.多轮(1347)（内部按 source 拆为 6 个子 Bench） |
| `science` | 科学 | HYeval.科学(1343), FSci_Olympiad(924), FSci_Research(962), GPQA-Diamond(933), HLE_Verified(1277), PhyBench(1236), SUPERChem(955) |

> ★ **合并规则**：ARC-AGI-v1 + ARC-AGI-v2 合并展示为 `ARC-AGI`；PRBench_Finance + PRBench_Legal 合并展示为 `PRBench`。

> **不参与分析的评测集**：HLE(971)、代码(1553)、专业领域(1349)、文本理解(1345)、agent 类评测集、安全、人设等。

### 1.3 Bench 描述

| Bench | 描述 |
|:------|:----|
| HYeval.复杂指令 | 指令遵循综合测试 |
| IFBench | 指令遵循的泛化性测试 |
| InverseIF | 抗干扰/抗惯性测试 |
| ToBv3 | 真实业务场景的复杂指令遵循 |
| HYeval.知识问答 | 自建综合知识问答 |
| HYQA | 综合知识问答 |
| SimpleQA | 英文简答 |
| BrowseComp | 基于网页检索的复杂知识推理 |
| HYKR | 知识复杂多跳推理 |
| Chinese_SimpleQA | 中文简答 |
| HY_Poem | 古诗词/歌词背诵 |
| HYeval.逻辑推理 | 综合类推理考察 |
| ARC-AGI | 通用人工智能基准测试 |
| BBEH | 推理能力的全面极限测试 |
| HeCheng_Logic | 规则合成的推理题 |
| Surge_Logic | 纯粹逻辑推理 |
| PRBench | 金融/法律专业领域推理 |
| HYeval.长文 | 自建长文综合能力考察 |
| LongBench_V2 | 开源专业文档长文推理 |
| CLBench | 即时学习与复杂执行 |
| BrowsCompLong | 跨网页检索式问答 |
| HYeval.数学 | 综合数学能力 |
| IMO_AnswerBench | 国际奥数评测集 |
| AMO_Bench | 数学竞赛推理 |
| NAnti_Misc | 数学难题集 |
| Math_T_v2 | 自建数学推理集 |
| HYeval.科学 | 综合科学能力 |
| FSci_Olympiad | 顶尖奥赛解题 |
| FSci_Research | 真实科研任务 |
| GPQA-Diamond | 博士级专家问答 |
| HLE_Verified | 跨学科前沿 |
| PhyBench | 物理数值计算 |
| SUPERChem | 化学专业深度 |

### 1.4 多轮 Bench 子映射（source → Bench）

多轮 Topic 只有一个 evId(1347)，通过 `payload.source` 字段拆分为 6 个子 Bench：

| 子 Bench | source 值 | 描述 |
|:---------|:---------|:----|
| PRBench | `open_benchmark://prbench_finance`, `open_benchmark://prbench_legal` | 专业领域多轮对话 |
| MultiInstruct | `open_benchmark://MultiInstruct` | 多指令跟随 |
| Multi-Challenge | `open_benchmark://multi-challenge` | 多轮上下文记忆 |
| 自建多轮评测集 | `create://挖掘`, `create://topic_text_writing`, `create://mturn_apex_v1`, `create://create` | 自建多轮评测集 |
| LongMemEval | `open_benchmark://longmemeval/oracle` | 长程记忆评测 |
| 综合类多轮测试集 | `open_benchmark://multiIF`, `open_benchmark://SysBench`, `open_benchmark://wildbench-v2`, `open_benchmark://superclue`, `open_benchmark://ImplexConv_opposed` | 综合类多轮测试集 |

---

## 第二层：题目对齐与指标计算

### 2.1 题目对齐 key

跨模型的同一道题通过**对齐 key** 匹配。使用保险方案避免 `_internal_question_id_raw_` 在不同模型间不一致的风险：

```
align_key = MD5( 归一化bench名 + "||" + json.dumps(model_input[0][0].messages, sort_keys=True) )
```

**归一化规则**：
- `HYeval3.0.xxx` / `HYeval3.1_xxx` → `HYeval.xxx`
- `LongBench_V2-all.2` / `LongBench_V2-all.v1` → `LongBench_V2`
- 其他保持原名

**fallback**：当 `model_input` 不可用时，退化为 `MD5( 归一化bench名 + "||qid:" + _internal_question_id_raw_ )`

```python
import hashlib, json, re

def normalize_bench_name(name):
    name = re.sub(r'HYeval3\.\d+', 'HYeval', name)
    name = re.sub(r'LongBench_V2-all[.\w]*', 'LongBench_V2', name)
    return name

def make_align_key(bench_name, payload):
    norm = normalize_bench_name(bench_name)
    mi = payload.get('model_input')
    if mi and isinstance(mi, list) and mi and isinstance(mi[0], list) and mi[0]:
        item = mi[0][0]
        if isinstance(item, dict):
            msgs = item.get('messages')
            if msgs:
                return hashlib.md5((norm + "||" + json.dumps(msgs, ensure_ascii=False, sort_keys=True)).encode('utf-8')).hexdigest()
    # fallback
    qid = payload.get('_internal_question_id_raw_') or payload.get('_internal_question_id_') or ""
    return hashlib.md5((norm + "||qid:" + str(qid)).encode('utf-8')).hexdigest()
```

### 2.2 score 提取

```python
INVALID_SCORE = -9999

def parse_score(raw):
    """从 payload.score 中提取浮点分数。递归解开嵌套 list，<= -9999 视为无效。"""
    v = raw
    for _ in range(4):
        if isinstance(v, list):
            v = v[0] if v else None
        else:
            break
    if v is None:
        return None
    try:
        f = float(v)
        return None if f <= INVALID_SCORE else f
    except (TypeError, ValueError):
        return None
```

### 2.3 指标计算

#### Bench 均分

1. 按 `align_key` 聚合同一题的多 pass（多次评估），取**均值**
2. 所有题目聚合后均值的**简单平均** → 该模型在该 Bench 上的分数

```python
from collections import defaultdict

def compute_bench_metric(records):
    """返回 (avg, n_questions, n_samples)"""
    by_key = defaultdict(list)
    for r in records:
        if r["score"] is None:
            continue
        by_key[r["align_key"]].append(r["score"])
    key_avgs = []
    for v in by_key.values():
        m = sum(v) / len(v)
        key_avgs.append(m)
    if not key_avgs:
        return None, 0, 0
    return sum(key_avgs) / len(key_avgs), len(key_avgs), sum(len(v) for v in by_key.values())
```

#### 逐级等权平均

| 级别 | 计算方式 |
|:----|:--------|
| Bench 均分 | 题目得分的简单平均（按 align_key 聚合后） |
| Topic 均分 | 该 Topic 下各 Bench 均分的**等权平均** |
| 跨 Topic 均分 | 7 个 Topic 均分的**等权平均** |

#### 逐题对比

比较目标模型与对比模型在共同题目（相同 align_key）上的表现：

```python
def gen_compare(records, m1, m2):
    """返回 (improved, regressed, same, n_common)"""
    m1_keys = defaultdict(list)
    m2_keys = defaultdict(list)
    for r in records:
        if r["score"] is None:
            continue
        k = r["align_key"]
        if r["model"] == m1:
            m1_keys[k].append(r["score"])
        elif r["model"] == m2:
            m2_keys[k].append(r["score"])
    common = set(m1_keys.keys()) & set(m2_keys.keys())
    imp = reg = same = 0
    for k in common:
        a = sum(m1_keys[k]) / len(m1_keys[k])
        b = sum(m2_keys[k]) / len(m2_keys[k])
        if a - b > 0.01:
            imp += 1
        elif b - a > 0.01:
            reg += 1
        else:
            same += 1
    return imp, reg, same, len(common)
```

---

## 第三层：维度下钻配置

每个 Topic 有 **Bench 下钻**（必有）+ **1~3 个特定维度下钻**。

### 3.1 复杂指令（complex_IF）

| # | 下钻维度 | 数据来源 | 计分方式 |
|:--|:--------|:--------|:--------|
| 1 | Bench | 4 个 bench 各自均分 | 标准均分 |
| 2 | 约束类型 | 仅 HYeval.复杂指令，`payload.gpt_response[0][0].result_list` | 特殊：满足率 |

**约束类型计分**：

`result_list` 是一个数组，每项包含 `{type, judge, weight}`：
- `type`：约束类型名（如「包含排除」「排版结构」「元素数量」等 ~16 种）
- `judge`：`True`（满足）/ `False`（不满足），字符串类型
- `weight`：权重（**不参与计分**）

每种约束类型的得分 = `judge=True 的数量 / 该类型总数量`（即满足率，不加权）

```python
def compute_constraint_metrics(records):
    """返回 {constraint_type: {model: {rate, n}}}"""
    stats = defaultdict(lambda: defaultdict(lambda: {"true": 0, "total": 0}))
    for r in records:
        if r["bench"] != "HYeval.复杂指令":
            continue
        for ctype, judge in r.get("constraints", []):
            stats[r["model"]][ctype]["total"] += 1
            if judge:
                stats[r["model"]][ctype]["true"] += 1
    result = {}
    all_types = set()
    for m_stats in stats.values():
        all_types.update(m_stats.keys())
    for ctype in sorted(all_types):
        result[ctype] = {}
        for m in stats:
            s = stats[m][ctype]
            result[ctype][m] = {"rate": s["true"] / s["total"] if s["total"] > 0 else None, "n": s["total"]}
    return result
```

**提取约束字段**：

```python
def extract_constraints(payload):
    """从 payload 中提取约束类型判定列表，返回 [(type, judge_bool), ...]"""
    gpt_resp = payload.get('gpt_response')
    if not gpt_resp or not isinstance(gpt_resp, list) or not gpt_resp[0] or not isinstance(gpt_resp[0], list):
        return []
    raw = gpt_resp[0][0]
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except:
            return []
    if not isinstance(raw, dict):
        return []
    result_list = raw.get('result_list', [])
    constraints = []
    for item in result_list:
        if isinstance(item, dict):
            ct = item.get('type')
            cj = item.get('judge')
            if ct and cj is not None:
                constraints.append((ct.strip(), str(cj) == 'True'))
    return constraints
```

### 3.2 知识问答（knowledge_qa）

| # | 下钻维度 | 数据来源 | 说明 |
|:--|:--------|:--------|:----|
| 1 | Bench | 7 个 bench 各自均分 | 标准均分 |
| 2 | 领域 | HYQA + Chinese_SimpleQA 的 `payload.origin_data.primary_category` | 跨 bench 合并映射 |

**领域合并映射**：

| 目标类别 | 原始值来源 |
|:---------|:---------|
| **人文社科** | HYQA「人文社科」、Chinese_SimpleQA「人文与社会科学」「社会」 |
| **生活/艺术** | HYQA「日常生活」「艺术娱乐」、Chinese_SimpleQA「生活、艺术与文化」 |
| **自然科学** | HYQA「自然科学」、Chinese_SimpleQA「自然与自然科学」 |
| **工程技术** | HYQA「工程技术」、Chinese_SimpleQA「工程、技术与应用科学」 |
| **中华文化** | Chinese_SimpleQA「中华文化」 |
| **其他** | HYQA「其他问题」 |

**字段提取**：`payload.origin_data.primary_category`（`origin_data` 可能为 JSON 字符串需解析）

### 3.3 逻辑推理（logic）

| # | 下钻维度 | 数据来源 | 说明 |
|:--|:--------|:--------|:----|
| 1 | Bench | 6 个 bench（ARC-AGI 合并 v1+v2，PRBench 合并 finance+legal） | 标准均分 |
| 2 | 推理类型 | HYeval.逻辑推理 + HeCheng_Logic 的 `payload.task_lv2` | 跨 bench 合并映射为 6 类 |

**推理类型合并映射**：

| 目标类别 | 原始值来源 |
|:---------|:---------|
| **关系推理** | HYeval「关系推理」、HeCheng「人物关系」「因果关系」 |
| **空间推理** | HYeval「空间推理」、HeCheng「空间推理」 |
| **逻辑推理** | HYeval「逻辑推理」「逻辑思维」、HeCheng「逻辑推理」 |
| **符号推理** | HYeval「符号推理」、HeCheng「混合符号推理」「数字符号」「字母符号」 |
| **常识与综合推理** | HYeval「常识与综合推理」「时间推理」 |
| **复杂问题** | HYeval「复杂问题」 |

> 丢弃：HYeval「其他」(3条)

### 3.4 长文（longcontext）

| # | 下钻维度 | 数据来源 | 说明 |
|:--|:--------|:--------|:----|
| 1 | Bench | 4 个 bench 各自均分 | 标准均分 |
| 2 | 任务类型 | HYeval.长文 `task_lv2` + CLBench `details.metadata.context_category` | 跨 bench 合并映射为 6 类 |
| 3 | 文本类型 | LongBench_V2 `details.extra_info.payload.data_lv1` | 映射为中文名 |

**任务类型合并映射**：

| 目标类别 | 原始值来源 |
|:---------|:---------|
| **领域知识推理** | CLBench「Domain Knowledge Reasoning」 |
| **复杂规则/系统理解** | HYeval「复杂规则学习」、CLBench「Rule System Comprehension」 |
| **流程化任务执行** | CLBench「Procedural Task Execution」 |
| **模式发现与归纳** | HYeval「多示例归纳学习」、CLBench「Pattern Discovery & Induction」 |
| **多文档检索** | HYeval「多文档检索」 |
| **多文档推理** | HYeval「多文档推理」 |

**文本类型映射**（LongBench_V2）：

| 英文原值 | 中文展示名 |
|:---------|:---------|
| Single-Document QA | 单文档 |
| Multi-Document QA | 多文档 |
| Long In-context Learning | 长上下文学习 |
| Code Repository Understanding | 代码仓库 |
| Long-dialogue History Understanding | 长对话 |
| Long Structured Data Understanding | 结构化数据理解 |

**字段路径**：`payload.details.extra_info.payload.data_lv1`（三层嵌套，每层可能为 JSON 字符串需解析）

### 3.5 数学（math）

| # | 下钻维度 | 数据来源 | 说明 |
|:--|:--------|:--------|:----|
| 1 | Bench | 5 个 bench 各自均分 | 标准均分 |
| 2 | 子领域 | 仅 IMO_AnswerBench 的 `payload.task_lv1` | 映射为中文 |
| 3 | 语言 | 仅 HYeval.数学的 `payload.language` | 中文 / 英文 |
| 4 | 难度 | 仅 HYeval.数学的 `payload.difficulty` | 超难 / 困难 / 中等 / 简单 |

**子领域映射**（IMO_AnswerBench）：

| 英文原值 | 中文展示名 |
|:---------|:---------|
| Algebra | 代数 |
| Combinatorics | 组合 |
| Geometry | 几何 |
| Number theory | 数论 |

### 3.6 多轮（multiturn）

| # | 下钻维度 | 数据来源 | 说明 |
|:--|:--------|:--------|:----|
| 1 | Bench(source) | `payload.source` 按 §1.4 映射为 6 个子 Bench | 标准均分 |
| 2 | 任务类型 | `payload.task_lv2` 原值 | 丢弃「其他」 |

**保留的任务类型**（6 类）：内容调整、记忆分析、指令遵循、推理计算、自我检查、信息抽取

> 丢弃：「其他」(600条)

### 3.7 科学（science）

| # | 下钻维度 | 数据来源 | 说明 |
|:--|:--------|:--------|:----|
| 1 | Bench | 7 个 bench 各自均分 | 标准均分 |
| 2 | 学科 | 跨 bench 多字段合并 | 映射为 8 类 |

**学科映射**：

| 目标学科 | 来源 |
|:---------|:----|
| **物理** | HYeval `task_lv2`=「物理」、GPQA `subject`=「物理」、FSci_O/R `subject`=「physics」、HLE_V `category`=「Physics」、PhyBench(整体) |
| **化学** | HYeval `task_lv2`=「化学」、GPQA `subject`=「化学」、FSci_O/R `subject`=「chemistry」、HLE_V `category`=「Chemistry」、SUPERChem(整体) |
| **生物/医学** | HYeval `task_lv2`=「生物」、GPQA `subject`=「生物」、FSci_O/R `subject`=「biology」、HLE_V `category`=「Biology/Medicine」 |
| **数学** | HLE_V `category`=「Math」 |
| **计算机/AI** | HLE_V `category`=「Computer Science/AI」 |
| **人文社科** | HLE_V `category`=「Humanities/Social Science」 |
| **工程** | HLE_V `category`=「Engineering」 |
| **其他** | HLE_V `category`=「Other」 |

> 丢弃：HYeval `task_lv2`=「hle」(480条)

---

## 第四层：核心结论生成

### 4.1 结论数据来源

所有结论基于以下计算结果自动生成：

| 数据 | 说明 |
|:----|:----|
| `topic_avgs[model][topic]` | 每个模型在每个 Topic 的均分（百分制） |
| `cross_avg[model]` | 跨 7 Topic 均分 |
| `bench_metrics[bench][model]` | 每个模型在每个 Bench 的均分 |
| `drill_data[dim_value][model]` | 下钻维度值的均分 |
| `compares[comp_model]` | 逐题对比（进步/退步/持平数） |

### 4.2 总览结论

#### 综合排名

按 `cross_avg` 降序排列所有模型，标注目标模型排名。

#### Target vs 每个对比模型

输出：
- 综合差距（pp）
- 优势 Topic：`topic_avgs[TARGET] - topic_avgs[comp] >= 1pp` 的 Topic，按差距降序
- 劣势 Topic：`topic_avgs[TARGET] - topic_avgs[comp] <= -1pp` 的 Topic，按差距升序

### 4.3 Topic 级结论

每个 Topic 输出以下结论：

#### Bench 亮点 / 短板

计算目标模型在每个 Bench 上相对对比模型均值的差距：
```
delta = target_bench_avg - mean(comp1_bench_avg, comp2_bench_avg)
```
- **亮点 Bench** = delta 最大的 Bench
- **短板 Bench** = delta 最小的 Bench

#### 逐题对比摘要

输出：`进步 N 题 / 退步 M 题 / 持平 K 题 / 净 ±X`

#### 下钻维度关键发现

对每个维度下钻：
- **最强维度值** = 目标模型分数最高的维度值
- **最弱维度值** = 目标模型分数最低的维度值
- **最大优势** = 相对对比模型差距最大（正向）的维度值
- **最大劣势** = 相对对比模型差距最大（负向）的维度值

### 4.4 结论文案模板

```python
def gen_topic_conclusion(topic_display, target, comp_models, bench_metrics, drill_data, compares):
    """生成单个 Topic 的核心结论文本"""
    lines = []
    
    # 1) Bench 亮点/短板
    bench_deltas = []
    for b, bm in bench_metrics.items():
        t_avg = bm.get(target, {}).get("avg")
        comp_avgs = [bm.get(c, {}).get("avg") for c in comp_models]
        comp_avgs = [v for v in comp_avgs if v is not None]
        if t_avg is not None and comp_avgs:
            comp_mean = sum(comp_avgs) / len(comp_avgs)
            bench_deltas.append((b, t_avg, comp_mean, t_avg - comp_mean))
    bench_deltas.sort(key=lambda x: -x[3])
    if bench_deltas:
        best = bench_deltas[0]
        worst = bench_deltas[-1]
        lines.append(f"亮点 Bench: {best[0]}（{best[1]:.2f}，vs 对比均值 {best[3]:+.2f}pp）")
        lines.append(f"短板 Bench: {worst[0]}（{worst[1]:.2f}，vs 对比均值 {worst[3]:+.2f}pp）")
    
    # 2) 逐题对比
    for comp, c in compares.items():
        net = c["improved"] - c["regressed"]
        lines.append(f"vs {comp}: {c['common']}道共题，进步{c['improved']} / 退步{c['regressed']} / 净{'+'if net>=0 else ''}{net}")
    
    # 3) 下钻关键发现
    for dim_name, dim_data in drill_data.items():
        if not dim_data:
            continue
        # 找目标模型最强/最弱维度值
        scored = [(dv, dm.get(target, {}).get("avg")) for dv, dm in dim_data.items()]
        scored = [(dv, v) for dv, v in scored if v is not None]
        if scored:
            scored.sort(key=lambda x: -x[1])
            lines.append(f"[{dim_name}] 最强: {scored[0][0]}（{scored[0][1]:.2f}），最弱: {scored[-1][0]}（{scored[-1][1]:.2f}）")
    
    return lines
```
