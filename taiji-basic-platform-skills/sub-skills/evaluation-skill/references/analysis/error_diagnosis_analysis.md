# BenchProd 评估结果错误归因分析

<!-- ROUTE_KEYWORDS
错误归因, error analysis, 错误分析, 根因分析, 错误样本, 失败case, 错误诊断, 错误分类, 错误类型, topic错误分析, bench错误分析, 按topic归因
-->

## 📌 本模块 Agent 行为契约（最高优先级）

> 本段是 SKILL.md 路由进入本文档后的统领规则，覆盖本文档所有章节。

- **何时进入**：用户问"错误归因 / 错误分析 / 失败 case 归因 / topic 错误分析 / 根因分析"等任一意图。
- **进入后必做**：
  1. 错误归因依赖 Insight 中的 `is_correct=false` 样本明细。先确认数据已导出（参见 `evaluation_api.md` 的 `submit_taiji_eval_insight_export`）。
  2. 用户没说要对哪个 bench/topic 做归因 → 先把候选维度列出来让用户选。
  3. LLM 自动归因输出必须包含：错误样本数 / 主要错误类型分布 / 典型 case 引用 / 改进建议——四项缺一不可。
- **严禁**：把"全量 case 导出"当成归因（那是 `evaluation_api.md` 的 case export 工具）；本模块只做归因聚合 + 根因分析。
- **失败处理**：样本不足时（< 5 条）直接告知用户"样本太少，归因结果不可信"，建议补充评测后再分析。

---

> 本文档供 AI Skill 使用，定义 BenchProd 生产评测集按 Topic 进行错误归因分析的完整规范：错误样本筛选、LLM 自动化错误诊断、归因聚合、根因分析与改进建议。

---

## 概述

### 分析目标

针对模型评估结果中的错误样本进行"病理分析"，通过 LLM 自动化错误分析打标，定位模型失败的根因。

### 错误归因四阶段流水线

```
Phase 1: 数据加载与错误样本筛选  →  error_samples.jsonl
Phase 2: LLM 逐条错误诊断       →  error_diagnosis_res.jsonl
Phase 3: 归因聚合将 LLM 输出的细粒度标签聚合为可报告的根因类别    →  error_clusters.jsonl
Phase 4: 根因分析输出典型错误模式 + 改进建议  →  cluster_summaries.json
```

### 两种分析模式

| 模式 | 适用场景 | 筛选逻辑 |
|:-----|:--------|:--------|
| **对比模式** | 存在目标模型 + 对比模型 | 目标模型全做错（avg=0）且对比模型全做对（avg≥0.8）的样本 |
| **单模型模式** | 仅分析单一模型 | 不同 pass 全做错（avg=0）的样本 |

---

## 第一层：数据加载与错误样本筛选

### 1.1 输入数据

数据来源与 [drilldown_analysis.md](./drilldown_analysis.md) 一致，使用相同的导出数据目录结构和字段解析方式。

**核心依赖**：
- `align_key` 生成方式：见 [drilldown_analysis.md §2.1](./drilldown_analysis.md#21-题目对齐-key)
- `score` 解析方式：见 [drilldown_analysis.md §2.2](./drilldown_analysis.md#22-score-提取)
- Topic / Bench 分组：见 [drilldown_analysis.md §1.2](./drilldown_analysis.md#12-topic-分组7-topic--37-bench)

### 1.2 用户交互确认（必须）

开始分析前，**必须与用户确认**以下参数：

| # | 确认项 | 说明 |
|:--|:------|:----|
| 1 | **分析模式** | 对比模式（需指定目标+对比模型）还是单模型模式 |
| 2 | **目标 Topic** | 分析哪个 Topic（7 选 1 或多选） |
| 3 | **模型路径** | 目标模型和对比模型的数据目录路径 |
| 4 | **采样上限** | 每个 Topic 的最大分析样本数（默认 200 条） |
| 5 | **LLM 调用脚本路径** | `call_llm_distill.py` 脚本，用于错误诊断分析 和 聚类摘要生成 |

### 1.3 错误样本筛选逻辑

#### 对比模式

分类：
- **regressed_hard**：目标模型做错 + 前代/基线做对 → 代际退化
- **hard_loss**：目标模型做错 + 竞品做对 → 竞品落败

```python
def filter_error_samples_compare(target_records, compare_records,
                                  threshold_target=0.2, threshold_compare=0.8):
    """
    筛选目标模型全做错、对比模型全做对的样本。
    返回: 符合条件的 align_key 集合
    """
    target_by_key = defaultdict(list)
    compare_by_key = defaultdict(list)
    for r in target_records:
        if r["score"] is not None:
            target_by_key[r["align_key"]].append(r["score"])
    for r in compare_records:
        if r["score"] is not None:
            compare_by_key[r["align_key"]].append(r["score"])
    
    error_keys = set()
    common_keys = set(target_by_key.keys()) & set(compare_by_key.keys())
    for key in common_keys:
        t_avg = sum(target_by_key[key]) / len(target_by_key[key])
        c_avg = sum(compare_by_key[key]) / len(compare_by_key[key])
        if t_avg <= threshold_target and c_avg >= threshold_compare:
            error_keys.add(key)
    return error_keys
```

#### 单模型模式

```python
def filter_error_samples_single(target_records, threshold=0.2):
    """
    筛选所有 pass 全做错的样本。
    返回: 符合条件的 align_key 集合
    """
    by_key = defaultdict(list)
    for r in target_records:
        if r["score"] is not None:
            by_key[r["align_key"]].append(r["score"])
    
    error_keys = set()
    for key, scores in by_key.items():
        avg = sum(scores) / len(scores)
        if avg <= threshold:
            error_keys.add(key)
    return error_keys
```

### 1.4 采样策略

- 每个 Topic 错误样本不超过采样上限 （默认 200 条）
- 按 `bench + task_lv2` 分层采样，保证各 Bench 有代表性
- 中文题目优先（便于人工验证）
- 采样后必须展示样本分布并与用户确认

```python
def stratified_sample(cases, target_n):
    """分层降采样，保留 bench × task_lv2 的分布比例"""
    if len(cases) <= target_n:
        return cases
    groups = defaultdict(list)
    for c in cases:
        key = (c["bench"], c.get("task_lv2", "unknown"))
        groups[key].append(c)
    total = len(cases)
    sampled = []
    for key, group in sorted(groups.items()):
        n = max(1, round(len(group) / total * target_n))
        n = min(n, len(group))
        sampled.extend(random.sample(group, n))
    if len(sampled) > target_n:
        sampled = random.sample(sampled, target_n)
    return sampled
```

### 1.5 错误样本输出字段

每条 `error_samples.jsonl` 必须包含以下字段：

| 字段 | 说明 | 来源 |
|:-----|:----|:----|
| `align_key` | 题目对齐 key | MD5 生成 |
| `bench` | Bench 名称 | 文件名解析 |
| `topic` | Topic 名称 | evId 映射 |
| `task_lv2` | 二级任务类型 | payload 字段 |
| `question` | 题目原文 | `payload.model_input[0][0].messages` 格式化 |
| `ref_answer` | 参考答案 | `payload.ref_answer` |
| `target_response` | 目标模型回答 | `payload.responses[0][0]` |
| `target_thinking` | 目标模型思考 | `payload.thinking_responses[0][0]` |
| `target_score` | 目标模型得分 | `payload.score` 解析 |
| `compare_response` | 对比模型回答（对比模式） | 同上 |
| `compare_thinking` | 对比模型思考 （对比模式）| 同上 |
| `compare_score` | 对比模型得分（对比模式）| 同上 |
| `mode` | 分析模式 | `compare` / `single` |

---

## 第二层：LLM 逐条错误诊断

### 2.1 Topic 错误分类体系

每个 Topic 有预定义的**诊断框架**（诊断阶段）和 **error_content 标签体系**（错误细分类型），执行之前必须与用户确认是否调整或修改预定义的诊断框架和标签体系，LLM 需按照确认后的诊断框架和标签体系对失败case进行错误分析。

#### 2.1.1 知识问答（knowledge_qa）

**诊断框架：C1-C7 七段诊断流水线**

| 阶段 | 名称 | 检查内容 |
|:-----|:----|:---------|
| C1 | 题意解析 | 问题目标、任务类型、对象与约束条件 |
| C2 | 知识定位与调用 | 关键事实、规则、概念的检索和调出 |
| C3 | 知识锚定 | 知识应用到具体对象、实体、语境 |
| C4 | 推理论证 | 推理链合法性、最早断裂点 |
| C5 | 边界控制 | 结论限定、知识边界识别 |
| C6 | 答案构建 | 最终答案完整性、清晰性 |
| C7 | 过程-答案一致性 | 答案是否忠实反映思维链 |


**错误分类体系（10 个聚合根因）**

| # | 聚合根因 | 匹配 error_content | 关键词 |
|:--|:---------|:-------------------|:-------|
| 1 | 数值型知识偏差 | 时间偏差, 数值偏差 | 数值, 日期, 年份, 时间, 数字 |
| 2 | 近似实体混淆 | 实体混淆, 名称错误 | 混淆, 张冠李戴, 近似 |
| 3 | 高频知识替代/覆盖 | — | 替代, 覆盖, 高频, 流行 |
| 4 | 传记/人物事实错误 | — | 人物, 传记, 出生, 职位 |
| 5 | 幻觉式编造 | 凭空捏造 | 幻觉, 编造, 杜撰 |
| 6 | 冷门/低频知识缺失 | 知识缺失 | 冷门, 低频, 缺失 |
| 7 | 历史/事件记忆偏差 | 过时信息 | 历史, 过时, 年代 |
| 8 | 推理链路断裂 | 推理错误, 因果倒置 | 推理, 逻辑, 因果 |
| 9 | 过度泛化/边界失控 | 过度泛化, 边界失控 | 泛化, 边界, 笼统 |
| 10 | 属性/关联错配 | 属性错误, 关联错误, 地点错误 | 属性, 关联, 错配 |

**error_content 标签（15 类）**：

```
时间偏差|实体混淆|数值偏差|地点错误|名称错误|因果倒置|属性错误|过时信息|凭空捏造|关联错误|知识缺失|推理错误|过度泛化|边界失控|事实错误
```

#### 2.1.2 逻辑推理（logic）

**诊断框架：L1-L7 七层诊断框架**

| 阶段 | 名称 | 检查内容 |
|:-----|:----|:---------|
| L1 | 规则与约束解析 | 底层规则、初始状态、合法动作空间 |
| L2 | 模式识别与归纳 | 从有限样本中归纳变换规律 |
| L3 | 推理策略与路径规划 | 逆向推理/分类讨论/穷举/矛盾法 |
| L4 | 推理执行 | 单步逻辑转换正确性 |
| L5 | 状态跟踪与记忆一致性 | 中间状态维护 |
| L6 | 答案结论生成 | 推导→最终答案 |
| L7 | 洞察力与建模效率 | 对称性/等价性/捷径识别 |


**错误分类体系（8 个聚合根因）**

| # | 聚合根因 | 匹配 error_content | 关键词 |
|:--|:---------|:-------------------|:-------|
| 1 | 模式归纳失败 | 模式识别错误, 归纳过拟合 | 模式, 归纳, 规律, 拟合 |
| 2 | 题意/条件误读 | 题意误解, 状态误读 | 题意, 误读, 语义绑定 |
| 3 | 状态跟踪失稳 | 状态丢失, 中间结果遗忘, 实体混淆 | 状态, 丢失, 遗忘 |
| 4 | 规则过度泛化 | 规则泛化错误, 规则遗漏 | 泛化错误, 外部常识 |
| 5 | 答案输出偏差 | 答案转录错误, 格式不符, 过程答案冲突 | 转录, 格式, 自相矛盾 |
| 6 | 逻辑执行谬误 | 单步谬误, 计算错误, 符号替换错误 | 谬误, 计算, 符号替换 |
| 7 | 搜索策略缺陷 | 搜索空间不足, 路径盲目, 策略选择错误 | 搜索, 策略, 穷举 |
| 8 | 幻觉/知识编造 | 知识编造 | 编造, 幻觉, 杜撰 |

**error_content 标签（23 类）**：

```
规则遗漏|规则泛化错误|状态误读|模式识别错误|归纳过拟合|策略选择错误|搜索空间不足|路径盲目|推导跳步|单步谬误|符号替换错误|因果倒置|计算错误|实体混淆|状态丢失|中间结果遗忘|答案转录错误|过程答案冲突|格式不符|题意误解|知识编造|过早放弃|其他
```

#### 2.1.3 数学（math）

**诊断框架：C1-C6 六段诊断流水线**

| 阶段 | 名称 | 检查内容 |
|:-----|:----|:---------|
| C1 | 题意理解与条件提取 | 目标、已知量、未知量、约束条件 |
| C2 | 知识与定理调用 | 定理、引理、经典结论、定义 |
| C3 | 解题策略与方法选择 | 方程建立、构造法、反证法、归纳法 |
| C4 | 推理演算与逻辑链 | 每步推导合法性 |
| C5 | 计算执行 | 代数化简、组合计数、模运算 |
| C6 | 最终答案提取 | 答案准确性、完整性、格式 |


**错误分类体系（两层 L1-L2）**

| L1 阶段 | L2 聚类 |
|:--------|:--------|
| 知识与定理调用 | 定理/公式缺失、定理记忆或使用错误、知识编造 |
| 解题策略与方法选择 | 策略方向选择错误、分类讨论框架缺陷、条件/约束遗漏 |
| 推理演算与逻辑链 | 逻辑跳步/推导断裂、充要条件与不等式方向错误、代数变形错误、边界/分支遗漏 |
| 题意理解与条件提取 | 求解对象/目标识别错误、关键条件遗漏 |
| 最终答案提取 | 答案转录/格式错误 |
| 计算执行 | 代数与数值计算错误 |

**error_content 标签（25 类）**：

```
条件遗漏|条件误读|求解对象错误|定理缺失|定理记忆错误|定理适用条件不满足|知识编造|策略方向错误|分类讨论遗漏|归纳法设置错误|构造不当|非法跳步|充要条件混淆|代数变形错误|不等式方向错误|等号条件忽略|边界情形遗漏|代数计算错误|组合计数错误|模运算错误|符号正负错误|答案转录错误|答案格式错误|过程答案矛盾|多解遗漏
```

#### 2.1.4 科学（science）

**诊断框架：六段诊断流水线**

| 阶段 | 名称 | 检查内容 |
|:-----|:----|:--------|
| C1 | 题意理解 | 是否正确理解问题的科学本质和求解目标 |
| C2 | 知识调用 | 是否调出正确的科学概念、定律、常数 |
| C3 | 形式化建模 | 是否选择了正确的物理/化学模型和方程 |
| C4 | 推理演算 | 科学推理和数学推导是否正确 |
| C5 | 计算执行 | 数值计算是否准确 |
| C6 | 答案提取 | 最终答案（含单位、有效数字）是否正确 |


**错误分类体系（8 个聚合根因）**

| 聚合根因 | 匹配 error_content |
|:---------|:-------------------|
| 题意理解偏差 | 条件遗漏 |
| 专业知识缺失/编造 | 概念缺失, 知识编造 |
| 科学概念混淆 | 概念混淆 |
| 建模方法错误 | 公式误用, 适用条件违反, 守恒关系错误 |
| 机制/路径选择错误 | 机制路径错误 |
| 推理链路错误 | 推导跳步, 因果倒置 |
| 计算执行错误 | 代数错误, 数值代入错误, 常数错误, 单位错误, 数量级错误 |
| 答案生成错误 | 答案转录错误, 过程答案冲突 |

**error_content 标签（17 类）**：

```
条件遗漏|概念缺失|概念混淆|公式误用|适用条件违反|守恒关系错误|机制路径错误|推导跳步|因果倒置|代数错误|数值代入错误|单位错误|常数错误|数量级错误|知识编造|答案转录错误|过程答案冲突
```

#### 2.1.5 复杂指令（complex_IF）

**诊断框架：指令遵循专项诊断**

| 阶段 | 名称 | 检查内容 |
|:-----|:----|:--------|
| C1 | 约束识别 | 是否完整识别出指令中的所有约束条件 |
| C2 | 约束理解 | 是否正确理解每个约束的语义和要求 |
| C3 | 约束执行 | 是否在生成过程中逐一执行所有约束 |
| C4 | 约束校验 | 生成结果是否通过所有约束的校验 |
| C5 | 输出质量 | 在满足约束的前提下内容质量是否达标 |


**错误分类体系**：LLM 动态语义聚类

与其他 Topic 不同，复杂指令采用**动态聚类**：先逐条诊断，再由 LLM 对所有 root_cause 做语义聚类，自动生成根因类别。

**error_content 标签（18 类）**：

```
约束遗漏|约束误解|格式违规|内容缺失|内容多余|长度违规|数量违规|语言违规|角色脱离|模板偏离|重复遗漏|部分执行|过度执行|逻辑错误|知识错误|质量不足|理解偏差|其他
```


#### 2.1.6 长文（longcontext）

**诊断框架：S1-S7 七阶段诊断流水线**

| 阶段 | 名称 | 检查内容 |
|:-----|:----|:--------|
| S1 | 任务理解 | 是否正确理解问题目标、所需证据类型和范围 |
| S2 | 证据检索 | 是否从长文中找到了正确且充分的证据 |
| S3 | 状态与结构追踪 | 实体属性、时间线、共指链、因果链是否稳定维护 |
| S4 | 干扰抵抗 | 是否抵御了高显著性噪声和误导性线索 |
| S5 | 推理整合 | 是否正确连接了来自长文不同部分的证据 |
| S6 | 答案构建 | 最终答案是否完整、准确、可用 |
| S7 | 过程-答案一致性 | 最终答案与思考链是否一致 |

**错误分类体系（7 个聚合根因）**

| 聚合根因 | 匹配 error_content |
|:---------|:-------------------|
| 任务理解偏差 | 题意误读, 任务类型混淆, 证据范围错位, 外部知识替代 |
| 证据检索缺失 | 关键证据遗漏, 多跳证据不全, 无关证据使用, 位置偏差 |
| 状态与结构追踪失稳 | 实体混淆, 共指错误, 时间线错乱, 因果链断裂 |
| 干扰抵抗失败 | 近因偏差, 词面相似误导, 背景观点当事实 |
| 推理整合错误 | 推理跳步, 错误连接, 时序当因果, 局部过度泛化 |
| 答案构建偏差 | 结论缺失, 关键限定丢失, 格式不符 |
| 过程-答案不一致 | 条件丢失, 过程答案冲突, 不确定性丢失 |

**error_content 标签（25 类）**：

```
题意误读|任务类型混淆|证据范围错位|外部知识替代|关键证据遗漏|多跳证据不全|无关证据使用|位置偏差|实体混淆|共指错误|时间线错乱|因果链断裂|近因偏差|词面相似误导|背景观点当事实|推理跳步|错误连接|时序当因果|局部过度泛化|结论缺失|关键限定丢失|格式不符|条件丢失|过程答案冲突|不确定性丢失
```
#### 2.1.7 多轮（multiturn）

**诊断框架：M1-M7 七层诊断框架**

| 阶段 | 名称 | 检查内容 |
|:-----|:----|:--------|
| M1 | 上下文理解与记忆 | 是否遗忘早期内容、混淆不同轮次信息 |
| M2 | 指令遵循与约束解析 | 是否遗漏显式约束、违反隐式规则 |
| M3 | 内容生成质量 | 内容是否空洞/重复/逻辑混乱/不准确 |
| M4 | 多轮一致性与连贯性 | 是否前后矛盾、立场反复、风格突变 |
| M5 | 推理与计算 | 逻辑推理、数学计算是否有错 |
| M6 | 自我纠错与反思 | 是否能根据反馈调整 |
| M7 | 格式与输出规范 | 格式是否符合要求 |


**错误分类体系（8 个聚合根因）**

| # | 聚合根因 | 匹配 error_content |
|:--|:---------|:-------------------|
| 1 | 上下文遗忘/误读 | 上下文遗忘, 历史信息混淆, 意图追踪断裂 |
| 2 | 指令违背/约束遗漏 | 约束遗漏, 隐式规则违背, 修正要求忽略, 累积约束丢失 |
| 3 | 多轮一致性断裂 | 前后矛盾, 立场反复, 风格突变 |
| 4 | 内容质量不足 | 内容空洞, 信息不准确, 逻辑混乱, 表达不清 |
| 5 | 推理/计算错误 | 推理错误, 计算错误, 因果偏差, 中间状态丢失 |
| 6 | 格式/输出偏差 | 格式不符, 输出结构混乱 |
| 7 | 知识编造/幻觉 | 知识编造, 过度解读 |
| 8 | 题意误解/信息扭曲 | 题意误解, 纠错失败, 过度修改, 反馈无响应 |

**error_content 标签（28 类）**：

```
上下文遗忘|历史信息混淆|意图追踪断裂|约束遗漏|隐式规则违背|修正要求忽略|累积约束丢失|内容空洞|信息不准确|逻辑混乱|表达不清|前后矛盾|立场反复|风格突变|推理错误|计算错误|因果偏差|中间状态丢失|纠错失败|过度修改|反馈无响应|格式不符|输出结构混乱|知识编造|过度解读|题意误解|其他
```

### 2.2 错误诊断 Prompt 模板

每个 Topic 的 prompt 由三部分组成：**System 角色 + 诊断框架 + 输出 JSON Schema**。需要与用户确认最终拼接后的prompt。

> ⚠️ **Prompt 中的 JSON 示例**：`{` 和 `}` 在 Python `.format()` 中需转义为 `{{` 和 `}}`。

#### 通用 Prompt 结构

```python
ERROR_DIAGNOSIS_TEMPLATE = """
{system_prompt}

=== 输入 ===
题目: {question}
参考答案: {ref_answer}
目标模型回答（得分: {target_score}）: {target_response}
目标模型思考过程: {target_thinking}
{compare_section}

=== 诊断框架 ===
{diagnosis_framework}

=== 分析方法 ===
请按诊断框架逐阶段检查模型的求解过程，找到**首个致命错误（first fatal error）**所在的阶段。
不需要每个阶段都展开详述——只对存在问题的阶段进行分析。

=== 输出要求 ===
请直接输出以下 JSON，不要包含代码块标记：

{{{{
  "first_fatal_error_stage": "{stage_enum}|无",
  "first_fatal_error_desc": "一句话描述首个致命错误及其原因",
  "error_content": "从以下选择最匹配的一个: {error_content_tags}",
  "error_chain_summary": "完整错误链因果路径",
  "root_cause": "1-2句话精准描述根本原因",
  "overall_result": "完全错误|部分错误|接近正确",
  "comp_diff": "对比模型做对而目标模型做错的关键差异点（无对比模型时填'无'）",
  "is_ref_answer_wrong": false
}}}}
"""
```

#### 各 Topic System Prompt

| Topic | System Prompt |
|:------|:-------------|
| knowledge_qa | 你是一位知识问答错误归因专家。请对以下评测失败的 case 进行系统性错误诊断。 |
| logic | 你是一位LLM逻辑推理评测专家，擅长各类逻辑推理、规则推理、模式识别和策略规划的深度错误分析。 |
| math | 你是一位数学评测错误诊断专家。请对以下评测失败的 case 进行系统性错误诊断。 |
| science | 你是一位科学评测错误诊断专家。请对以下评测失败的 case 进行系统性错误诊断。 |
| complex_IF | 你是一名大语言模型指令遵循能力评估专家。请对以下指令遵循任务中模型回答错误的 case 进行深度诊断。 |
| longcontext | 你是一位长文理解能力评测专家。请对以下评测失败的 case 进行系统性错误诊断。 |
| multiturn | 你是一位LLM多轮对话评测专家。请对以下评测失败的 case 进行系统性错误诊断。 |

### 2.3 文本截断参数

执行前需要与用户确认截断参数是否调整。

| Topic | thinking 上限 | response 上限 | question 上限 | ref_answer 上限 |
|:------|:-------------|:-------------|:-------------|:---------------|
| knowledge_qa | 6000 | 4000 | 不截断 | 不截断 |
| logic | 6000 | 4000 | 不截断 | 不截断 |
| math | 6000 | 4000 | 不截断 | 不截断 |
| science | 6000 | 4000 | 不截断 | 不截断 |
| complex_IF | 6000 | 4000 | 不截断 | 不截断 |
| longcontext | 50000 | 50000 | 50000 | 10000 |
| multiturn | 6000 | 4000 | 50000 | 不截断 |

### 2.4 LLM 调用方式

执行前，需要与用户确认LLM的调用方式，默认调用脚本 call_llm_distill.py 通过蒸馏平台调用外部LLM，需要自行配置蒸馏平台的APP_ID, APP_KEY, MODEL_NAME 。

```bash
PYTHONIOENCODING=utf-8 python3 scripts/call_llm_distill.py \
  --input_file error_diagnosis_prompts.jsonl \
  --output_file error_diagnosis_res.jsonl \
  --prompt_key prompt \
  --num_jobs 5 \
  --reasoning_effort high \
  --tqdm
```

---

## 第三层：归因聚合

### 3.1 聚合方法

采用**两阶段匹配**策略，先 `error_content` 精确匹配，再 `root_cause` 关键词兜底：

```python
def classify_case(parsed, cluster_rules):
    """
    两阶段分类：
    1. error_content 精确匹配（优先级高）
    2. root_cause + first_fatal_error_desc + error_chain_summary 关键词匹配
    """
    content = parsed.get("error_content", "")
    
    # 特殊类别：非模型错误
    if parsed.get("is_ref_answer_wrong") == True:
        return "参考答案/评测问题"
    if parsed.get("first_fatal_error_stage") == "无":
        return "参考答案/评测问题"
    
    # Phase 1: error_content 精确匹配
    for cluster_name, rule in cluster_rules.items():
        if content in rule.get("contents", []):
            return cluster_name
    
    # Phase 2: 关键词匹配
    combined = " ".join([
        parsed.get("root_cause", ""),
        parsed.get("first_fatal_error_desc", ""),
        parsed.get("error_chain_summary", ""),
    ])
    for cluster_name, rule in cluster_rules.items():
        keywords = rule.get("keywords", [])
        if keywords and any(kw in combined for kw in keywords):
            return cluster_name
    
    return "其他"
```

> ⚠️ **匹配顺序很重要**：`error_content` 精确匹配必须优先于关键词匹配，否则宽泛关键词会吞掉本应归入其他类别的 case。

### 3.2 各 Topic 聚合根因体系（CLUSTER_RULES）

各 Topic 聚合根因体系（聚合方式） 见第二层 2.1.1 ~ 2.1.7 的错误分类体系定义。

---

## 第四层：根因分析（LLM 聚类摘要）

### 4.1 聚类摘要生成

对每个聚合类别（case 数 ≥ 3），将该类别下所有 case 的 `root_cause`, `first_fatal_error_desc`, `error_chain_summary` 文本汇总，送入 LLM 生成结构化摘要。

**聚类摘要 Prompt 模板**：

```python
CLUSTER_SUMMARY_TEMPLATE = """
你是一位{topic_expert}评测分析专家。以下是同一类型「{cluster_name}」下的 {n_cases} 个错误 case 的根因描述。

类别说明: {cluster_desc}

=== 各 case 的根因描述 ===
{case_descriptions}

=== 请直接输出 JSON ===
{{{{
  "root_cause": "2-4句话精炼总结该类错误的共性根因（本质机制、触发条件、为何模型容易犯此错），300-500字",
  "patterns": ["典型错误模式1: ...", "典型错误模式2: ...", "典型错误模式3: ..."],
  "fix_suggestion": "1-2句话的修复建议"
}}}}
"""
```

其中 `{case_descriptions}` 格式为：
```
[0] root_cause: "..."  error_desc: "..."  error_chain: "..."
[1] root_cause: "..."  error_desc: "..."  error_chain: "..."
```

每个聚类最多 30 条描述文本，超过时随机采样。

**各 Topic expert 角色**：

| Topic | topic_expert |
|:------|:------------|
| knowledge_qa | 知识问答 |
| logic | 逻辑推理 |
| math | 数学 |
| science | 科学 |
| complex_IF | 指令遵循 |
| longcontext | 长文理解 |
| multiturn | 多轮对话 |

### 4.2 LLM 调用

```bash
PYTHONIOENCODING=utf-8 python3 scripts/call_llm_distill.py \
  --input_file cluster_summary_prompts.jsonl \
  --output_file cluster_summary_res.jsonl \
  --prompt_key prompt \
  --num_jobs 5 \
  --reasoning_effort medium \
  --tqdm
```

### 4.3 输出格式

**`cluster_summaries.json`** 结构（以`知识问答`为例）：

```json
{
  "topic": "knowledge_qa",
  "mode": "compare",
  "target_model": "model_A",
  "compare_model": "model_B",
  "total_error_cases": 200,
  "real_model_errors": 185,
  "clusters": {
    "近似实体混淆": {
      "n_cases": 42,
      "pct": "22.7%",
      "error_contents": {"实体混淆": 30, "名称错误": 12},
      "bench_dist": {"HYQA": 20, "Chinese_SimpleQA": 15, "HYeval.知识问答": 7},
      "root_cause": "LLM 生成的共性根因描述...",
      "patterns": ["典型模式1: ...", "典型模式2: ..."],
      "fix_suggestion": "修复建议...",
      "representative_cases": [
        {"align_key": "xxx", "question_preview": "...", "root_cause": "..."}
      ]
    }
  }
}
```

### 4.4 结果展示规则

1. **排序**：聚合根因类别按 case 数从大到小排序
2. **噪声类置末**：「参考答案/评测问题」固定放在最后
3. **过滤**：不展示 case 数 < 3 的聚类
4. **展示结构**（每个聚类）：
   - 类别名 + case 数 + 占比%
   - 错误子类标签（error_content 分布）
   - 共性根因（LLM 生成）
   - 典型错误模式（LLM 生成，编号列表）
   - 修复建议（LLM 生成）
   - 3 个典型代表 case

---


## 第五层：关键经验与避坑

### 1. Payload 二次解析
数据中 `payload` 字段可能是 JSON 字符串（而非 dict），必须检查并二次解析。

### 2. 嵌套 List 提取
`responses`/`thinking_responses`/`usage` 都是嵌套 list 结构 `[[item]]`，提取时需按 `[0][0]` 索引。`responses[0][0]` 可能是 dict（含 `content`/`reasoning_content`）或 str。

### 3. LLM 返回解析
LLM 的实际回答在 `server_response` 字段（JSON 字符串），需要 robust 解析：

```python
def robust_parse_json(text):
    """容错 JSON 解析"""
    import re
    try:
        return json.loads(text)
    except:
        pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    cleaned = text.replace('\u201c', '"').replace('\u201d', '"')
    cleaned = cleaned.replace('\u2018', "'").replace('\u2019', "'")
    m = re.search(r'\{[\s\S]*\}', cleaned)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return None
```

### 4. 聚合根因 ≠ error_content
**绝对不能**直接用 `error_content` 作为顶层分类展示。必须通过 CLUSTER_RULES 做语义聚合，`error_content` 仅作为每个聚合类别下的错误子类标签。

### 5. Prompt 中花括号转义
prompt 模板中的 JSON 输出示例必须将 `{` 和 `}` 转义为 `{{` 和 `}}`，否则 Python `.format()` 会报 `KeyError`。

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
