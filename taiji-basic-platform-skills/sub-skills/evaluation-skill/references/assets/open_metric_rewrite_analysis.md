# 新增开源 Metric 改写进 hy_unify_eval

把开源评测集自带的打分代码（主要来自 GitHub 上的 opencompass / lm-evaluation-harness 等评测框架，以及 MMLU/HumanEval/GSM8K/BBH/MATH/AGIEval 等数据集仓库）按太极内部 `hy_unify_eval` 框架的 `BaseEval`/`BaseData` 规范改写成可被一站式评测系统反射加载的 Metric，改完后自动提交到个人分支并推送到远程仓库。

**何时进入**：用户说"新增一个开源 metric / 评测"、"把 xx 评测集的打分代码改写/抓取/拉到 evals / hy_unify_eval"、"接入 xx（MMLU/HumanEval/GSM8K/...）到太极评测"、"在 hy_unify_eval 里加一个 metric"，或给出一个开源仓库的 `evaluate.py`/`metric.py`/打分脚本要求接入内部评测时。

⚠️ **代码库路径不写死——进入流程前必须先向用户询问 hy_unify_eval 代码库的本地克隆路径**（不同用户克隆位置不同，不要假设默认路径）。目标远程仓库为 `git@git.woa.com:taiji/hy/hy_unify_eval.git`；拿到本地路径后用 `git -C <本地路径> remote -v` 确认 origin 指向该地址，确保操作的是正确仓库。下文所有 `<本地路径>` 均指此处询问得到的路径。

## 前置：数据流与两个核心类

```
原始数据 → DataClass.infer_preprocess() → 模型推理 → DataClass.infer_postprocess() → EvalClass.get_sample_score() → get_final_score() 汇总
```

| 类 | 作用 | 核心方法 |
|---|---|---|
| `BaseData` | 数据预处理/后处理 | `infer_preprocess()`, `infer_postprocess()` |
| `BaseEval` | 评测打分 | `get_sample_score()`（必须）, `get_final_score()`（默认已实现，按需重写） |

`responses` 与 `score` 均为 `[trial][turn]` 二维数组（第一维试验次数，第二维对话轮次）。

`hy_unify_eval` 本身不做服务化（无 FastAPI/entrypoint），它是一个被上游一站式评测系统 `import` 的 SDK；Metric 通过在两个 `__init__.py` 中显式注册后被反射加载，再经评测集版本（exercise_version）的 `metric_name`/`data_class_name` 暴露给上层服务（见本 skill `exercise_management_api.md`）。

### BaseEval 接口（`sdk/eval_class/base_eval.py`）

```python
ERROR_SCORE = -100000   # 错误分数（model_status=False / finish_reason=length 且策略为 as_error）
ZERO_SCORE = 0          # 零分（finish_reason=length 且策略为 as_zero）
```

`BaseEval` 通过 `__init_subclass__` 自动包装子类的 `get_sample_score`，注入预处理逻辑。子类**无需**也**不应**手动处理 model_status / finish_reason：

1. 调用子类方法前：检查 `sample["model_status"][trial]`，若为 `False` 则该 trial 标记为 `ERROR_SCORE`。
2. 检查 `sample["usage"][trial][turn]["finish_reason"] == "length"`，按 `judge_kwargs["response_postprocessing_policy"]` 处理：`"as_error"`→`ERROR_SCORE`；`"as_zero"`→`ZERO_SCORE`；`"as_is"`（默认）→跳过。
3. 调用子类方法后：用预处理状态覆盖有问题的 trial 分数。

```python
@classmethod
def get_sample_score(cls, sample, judge_kwargs):
    """
    Args:
        sample: 待评测样本，含 messages / responses[trial][turn] / ref_answer（字段名随数据集，
                 如 gsm8k 用 answer，bbh 用 target）/ model_status / usage
        judge_kwargs: 可含 eval_model_config（Judge 模型配置）、response_postprocessing_policy
    Returns:
        sample，必须填 score（二维），可选填 gpt_response、extra_score（均二维）
    """
```

**约束**：score 必须与 responses 维度一致；解析失败/异常一律返回负分 `-100000`，不要抛异常、不要静默给 0；负分样本会被 `get_final_score` 严格过滤。

```python
@classmethod
def get_final_score(cls, records, pass_threshold=1.0):
    """
    Returns: acc / bon_acc / error_rate / truncation_rate / avg_completion_token /
             avg_prompt_token / pass@k(n>1) / trial_detail(n>1)
    """
```

**严格过滤逻辑**：若某题在任一 trial 出现错误（分数 < 0 或数据为空），该题所有 trial 都不参与 `acc`/`bon_acc`/`pass@k` 计算（仍计入 `error_rate`/token 统计）。**何时重写**：仅需自定义汇总口径（如按子任务拆分准确率、删 error_rate）时才重写，调用 `super().get_final_score(records)` 再加工（先例见下文「改写先例与改造点对照」的 AI2ARC）。

### BaseData 接口（`sdk/data_class/base_data.py`）

```python
class BaseData:
    @classmethod
    def infer_preprocess(cls, sample, infer_kwargs): ...   # 构造 messages/prompt + inference_param
    @classmethod
    def infer_postprocess(cls, sample, infer_kwargs): ...  # 默认直接返回 sample
```

`infer_preprocess` 默认行为：合并 `inference_params`；`fix_` 前缀参数高优先级覆盖；历史兼容映射（`max_token`→`max_tokens`、`topk`→`top_k`、`topp`→`top_p`）；`non_infer_kargs`（`exercise_name`/`dataset_path`/`trial_num`/`metric_name` 等）不进入 `inference_param`；按 `modelStage` 分流 Pretrain（续写）/ 对话两种模式。子类重写时应先 `super().infer_preprocess()` 拿基础 `inference_param` 再覆盖 `messages`，不要从零拼。

`infer_postprocess` 产出的 `sample` 必须含 `responses`（`[trial][turn]` 二维），供 `get_sample_score` 消费。多数场景无需重写。

**OfficialTemplateData**：继承 `BaseData` 默认行为，支持 SFT/Pretrain 两种模式，数据无需特殊 prompt 拼接时直接 `class XxxData(OfficialTemplateData): pass`。

**生成参数映射**（开源 `generation_kwargs`/`gen_kwargs` → hy_unify_eval `inference_param`）：

| 开源参数 | hy_unify_eval `inference_param` |
|---|---|
| `until` | `stop` |
| `temperature` | `temperature` |
| `do_sample: false` | `temperature: 0` |
| `max_gen_toks`/`max_new_tokens` | `max_tokens` |
| `top_p`/`top_k` | `top_p`/`top_k`（已有历史兼容映射） |

优先在任务配置的 `inference_params` 设（由调度层注入），不在 Data 类硬编码；仅当某数据集必须固定生成参数时才在 `infer_preprocess` 里 `res["inference_param"].update({...})`。字段名以 `sdk/client/` 实际约定为准。

## 工作流

### Step 1 选型与来源确认

- 确认目标开源 metric 的来源框架/仓库。已知主要来源：**lm-evaluation-harness**（`https://github.com/EleutherAI/lm-evaluation-harness`）、**opencompass**（`https://github.com/open-compass/opencompass`）、单数据集仓库自带 grader（MMLU/HumanEval/GSM8K/BBH/MATH 等）。
- 确认 commit/版本、目标 metric 在仓库里的具体路径；License（GPL/AGPL 类需谨慎，改写而非照搬可降低风险，但仍需登记来源）；输入数据格式与 `ref_answer` 字段格式。

### Step 2 定位与解读开源打分逻辑（按来源框架识别）

**A. lm-evaluation-harness 式**：打分逻辑在 `lm_eval/tasks/<dataset>/<task>.yaml` 的 `metric_list`，引用 `lm_eval/api/metric.py` 注册的 metric 函数；关键方法 `lm_eval/api/task.py` 的 `process_results(doc, results)`；可能有 `filter` 机制（如 `extract_answer`）。提取要点：metric 函数 + 聚合器 + 答案提取 filter。

**B. opencompass 式**：评估器在 `opencompass/opencompass/evaluator/`，基类 `BaseEvaluator`，内置 `AccEvaluator`/`ECEEvaluator`/`BleuEvaluator`/`CorrelationEvaluator`，核心 `score(predictions, references)` 方法；数据集配置通过 `eval = dict(evaluator=AccEvaluator())` 或 `ICL_EVALUATORS` 引用。提取要点：evaluator 的 `score` 方法 + 答案提取/后处理。

**C. 单数据集仓库自带 grader**：如 MATH 的 `math_equivalence.py`、HumanEval 的 `execution.py`。独立函数式，最易提取。

**通用提炼**：提炼"纯打分内核"——输入（模型输出+标准答案）、输出（分数/对错）、核心比较逻辑；剥离 IO/网络/数据加载/CLI；记录原版边界处理（空答案/解析失败/多答案），改写时映射为负分。

> 以上框架结构基于通用认知，接入前以目标仓库实际代码为准（先 ls/读确认 metric 文件确切位置）。

**若 metric 函数源码读不全**（路径变动/404/仓库重构——实测 lm-eval-harness 的 `metric.py` 在 main 分支即 404）：以 task config 的 metric 名 + 参数（如 `ignore_case`/`ignore_punctuation`/`aggregation`）为权威，normalize 等用标准做法先实现，对齐校验（Step 8）时再按偏差细调，不必死等读全源码。

**输出类型适配（loglikelihood → generate_until）**：hy_unify_eval 只支持 generate_until（`responses[trial][turn]` 为生成文本），**不支持 loglikelihood 多选**。若开源原版是 `multiple_choice`/`loglikelihood`（如 HellaSwag/MMLU/WinoGrande 原版），必须适配为 generate_until：让模型在 prompt（题干 + A/B/C/D 选项）后生成选项字母，Eval 类提取字母与 gold 比较。参考下文「改写先例与改造点对照」的 AI2ARC 先例（字母提取 + `original_responses`）。**注意：适配后评测方式改变，分数不可对齐开源原版**——验收口径见 Step 8。

### Step 3 选定分类目录并新建文件

`hy_unify_eval` 的 eval 与 data 目录**不对称**，务必成对创建。已核实的映射（`text_to_text/` 下，仍建议新建前 `ls` 复核）：

| 类别（实际承载任务） | Eval 目录 | Data 目录 | 对称性 |
|---|---|---|---|
| 数理推理 + 开放域 QA | `P1_eval/<数据集>/` | `P1_eval_data/<数据集>/` | 对称（`_eval`↔`_eval_data`） |
| 代码 | `T3_code_eval/<数据集>/` | `code_eval_data/<数据集>/` | **不对称**（不是 `T3_code_eval_data`） |
| 通用模板 | `official_template/` | `official_template/` | 对称（同名） |
| evalscope | `evalscope/` | `evalscope/` | 对称（同名） |
| E1 | `E1_eval/<数据集>/` | `E1_data/<数据集>/` | 对称（`_eval`↔`_data`） |
| T1 | `T1_eval/<数据集>/` | `T1_data/<数据集>/` | 对称（`_eval`↔`_data`） |
| T2_logic | `T2_logic_eval/<数据集>/` | `T2_logic_data/<数据集>/` | 对称（`_eval`↔`_data`） |
| ICL | `ICL_eval/<数据集>/` | `ICL_eval_data/<数据集>/` | 对称（`_eval`↔`_eval_data`） |
| R1 | `R1_eval/<数据集>/` | 无独立 data 目录 | 仅 eval 侧 |
| Agent | `nexus/<数据集>/` | 无独立 data 目录 | 仅 eval 侧 |
| yuanbao | `yuanbao/<数据集>/` | 无独立 data 目录 | 仅 eval 侧 |

**规律与例外**：多数对称（`Xxx_eval`↔`Xxx_eval_data`/`Xxx_data`）；例外 1：代码类 eval=`T3_code_eval`，data=`code_eval_data`（前缀都变了）；例外 2：`R1_eval`/`nexus`/`yuanbao` 无独立 data 目录。**不要假设，新建前一律 `ls sdk/data_class/text_to_text/` 复核。**

> 分类名不严格对应任务类型——`P1_eval` 名义"数理推理"，实际也放开放域 QA（hotpotqa/popqa/triviaqa 等）。**选目录以"同类先例在哪个目录"为准**，而非看分类名字面意思。

命名约定：eval 文件用 `metric_eval.py` 或 `<数据集>_eval.py`；data 文件用 `<数据集>_data.py`。类名 PascalCase，如 `ZeroShotMMLUEval`/`ZeroShotMMLUData`。

### Step 4 改写 Eval 类（核心）

写一个继承 `BaseEval` 的类，实现 `get_sample_score`。必须遵守的约束（违反则注册了也跑不通或指标失真）：

1. 签名：`@classmethod def get_sample_score(cls, sample, judge_kwargs):`
2. 遍历二维 `responses[trial][turn]`，产出二维 `score[trial][turn]`。
3. 错误用负分（`ERROR_SCORE = -100000`），**不要抛异常**；基类靠负分识别错误样本并严格过滤。
4. 字段名沿用开源原版（如 gsm8k 的 `answer="cot####ans"`、bbh 的 `target`），降低对齐成本。
5. 返回 `sample`，填 `sample["score"]`；可选填 `sample["gpt_response"]`、`sample["extra_score"]`。
6. 默认 `get_final_score` 已实现，多数场景不重写；仅需自定义汇总时才重写。

完整先例代码见下文「改写先例与改造点对照」。

### Step 5 改写配套 Data 类

继承 `BaseData`，实现 `infer_preprocess`（构造 `messages`+`inference_param`），通常 `infer_postprocess` 用默认即可。多数场景可继承 `OfficialTemplateData` 或用 `BaseData` 默认行为，仅需自定义 prompt 拼接时重写。若所属分类无独立 data 目录（如 `R1_eval`/`nexus`/`yuanbao`），复用 `official_template` 或确认是否需要 data 类。

### Step 6 注册（两个 `__init__.py`，缺一不可）

一站式评测系统从 `sdk/eval_class/__init__.py` 和 `sdk/data_class/__init__.py` 反射加载类。**漏注册 = 不可用**。

```python
# sdk/eval_class/__init__.py（以 gsm8k 行为锚，紧邻加一行）
from sdk.eval_class.text_to_text.P1_eval.gsm8k.gsm8k_eval import ZeroShotGSM8KEval
from sdk.eval_class.text_to_text.P1_eval.<新数据集>.<新文件> import <NewEval>
# __all__ 列表同样紧邻加一行
__all__ = [..., "ZeroShotGSM8KEval", "<NewEval>", ...]
```

```python
# sdk/data_class/__init__.py（同理）
from sdk.data_class.text_to_text.P1_eval_data.gsm8k.gsm8k_data import ZeroShotGSM8KData
from sdk.data_class.text_to_text.P1_eval_data.<新数据集>.<新文件> import <NewData>
__all__ = [..., "ZeroShotGSM8KData", "<NewData>", ...]
```

注意 import 路径里 eval 侧分类目录（如 `P1_eval`）与 data 侧（如 `P1_eval_data`）目录名、文件名都可能不同，照抄先例的实际路径，不要想当然。代码类的路径是 `T3_code_eval.<数据集>.<文件>` ↔ `code_eval_data.<数据集>.<文件>`，两侧前缀都不同。

### Step 7 复用已有工具（避免重复造轮子）

- 数学答案验证：`sdk/eval_class/text_to_text/P1_eval/stem_utils/`（`parse_ground_truth`/`extract_answer`/`math_equal`，本身即从 Hendrycks' MATH 开源代码移植）、`sdk/thirdparty/patch_math_verify.py`。
- 代码执行：`sandbox_server`/`UnifiedSandboxProcessor`。
- LLM-as-Judge：`sdk/eval_class/call_judge_model_common.py`（配合 `sdk/client/`）。
- 通用工具：`sdk/utils/`（`extract_answer`/`extract_json_from_text`/`decorators`）。

### Step 8 对齐校验（验收）

- **验收口径（默认）**：在原评测集上用改写后的 Metric 跑一遍，分数与开源版本对齐，差异 ≤ ±0.5%。若需求方另指定口径（如"跑通即可"），以需求方为准。
- **评测方式改变时**：若原版 output_type 与 hy_unify_eval 范式不同（如 loglikelihood 多选 → generate_until 生成字母），分数不可对齐开源原版。此时验收口径改为：①打分逻辑正确性；②生成式 acc 在合理范围；③无负分异常堆积。不硬求 ±0.5% 对齐。
- 不一致时重点查：答案解析逻辑、等价判断宽松度（`math_equal`）、负分/空答案处理、字段映射。

### Step 9 OpenAPI 对接（让上游能用）

Metric 在 `hy_unify_eval` 注册后，用**本 skill**创建/更新 exercise_version，配置 `metric_name`（新 Eval 类名）+ `data_class_name`（新 Data 类名），类名须与 `__init__.py` 注册的完全一致。具体调用见本 skill 的 `exercise_management_api.md`（`create_taiji_eval_exercise_version`/`update_taiji_eval_exercise_version`）。若需纳入对比集合，把该 exercise_version 挂到 `collection_version` 的 `weight_node`（见 `collection_management_api.md`）。

### Step 10 提交并推送到远程仓库（⚠️ 严禁在 master 分支操作）

Metric 代码改写完成（Step 4-6 落库）后，必须提交并推送到远程 `git@git.woa.com:taiji/hy/hy_unify_eval.git`。**这是强制安全规则，任何情况下不可跳过或绕过**：

1. **检查当前分支**：`git -C <本地路径> branch --show-current`。
2. **⛔ 若当前在 `master` 分支**：**严禁**在 master 上直接提交或推送。必须先告知用户"当前在 master 分支，禁止直接操作，将为你创建个人分支"，然后自动创建并切换到个人分支：
   - 分支命名：`<git config user.name>/add-<metric名小写>-metric`（如 `lamicyang/add-triviaqa-metric`）；若 `user.name` 取不到或非常规（如含空格/大写不规范），改用 `git config user.email` 的用户名部分。
   - 命令：`git -C <本地路径> checkout -b <分支名>`。
   - 创建后向用户确认新分支名，再继续。
3. **若当前已在非 master 的个人分支**：直接在当前分支继续，无需新建。
4. **提交**：`git add` 新增/修改的文件（新建的 Eval/Data 类文件 + 两个 `__init__.py`），`git commit -m "<描述性信息，如 feat: add <Metric名> metric from <来源框架>>"`。
5. **推送**：`git push origin <当前分支名>`（**首次推送新分支需要** `git push -u origin <分支名>`）。**⛔ 严禁 `git push origin master`，严禁在任何情况下推送到 master。**
6. 推送成功后告知用户分支名和远程地址，建议用户后续在 git.woa.com 发起 Merge Request 合入 master（**不代替用户发 MR，只做到推送个人分支为止**）。

**失败处理**：推送失败（如权限/冲突）时，如实告知错误信息，不静默重试、不强推（`--force`）。冲突时提示用户手动处理，不擅自 `git reset`/`git rebase` 覆盖用户历史。

---

## 改写先例与改造点对照

下面给出由简到繁的先例，并总结"开源脚本式打分 → BaseEval 类式"的标准改造点。

### 先例 1：GSM8K（纯函数式打分，最简范式）

**Eval 类** `sdk/eval_class/text_to_text/P1_eval/gsm8k/gsm8k_eval.py`：

```python
from sdk.eval_class.base_eval import BaseEval
from sdk.eval_class.text_to_text.P1_eval.stem_utils.parser import *
from sdk.eval_class.text_to_text.P1_eval.stem_utils.grader import *

class ZeroShotGSM8KEval(BaseEval):
    @classmethod
    def get_sample_score(cls, sample, judge_kwargs):
        data_name = "gsm8k"
        sample['score'] = []
        for trial_num in range(len(sample['responses'])):
            trial_score = []
            gt_cot, gt_ans = parse_ground_truth(sample, data_name)
            for response in sample['responses'][trial_num]:
                pred = extract_answer(response, data_name)
                if math_equal(pred, gt_ans, timeout=True):
                    trial_score.append(1)
                else:
                    trial_score.append(0)
            sample['score'].append(trial_score)
        return sample
```

**Data 类** `sdk/data_class/text_to_text/P1_eval_data/gsm8k/gsm8k_data.py`：

```python
from sdk.data_class.base_data import BaseData

class ZeroShotGSM8KData(BaseData):
    def __init__(self, infer_kwargs):
        pass

    @classmethod
    def infer_preprocess(cls, sample, infer_kwargs):
        res = super().infer_preprocess(sample, infer_kwargs)
        system_prompt = ("A conversation between User and Assistant. The user asks a question, "
                         "and the Assistant solves it.\nThe assistant first thinks about the "
                         "reasoning process in the mind and then provides the user with the answer.\n")
        question = sample["question"]
        res["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        return res

    @classmethod
    def infer_postprocess(cls, sample, infer_kwargs):
        return sample
```

**改写要点**：开源 GSM8K 的打分逻辑（提取数字、数值等价比较）已被下沉到 `stem_utils/`（`grader.py` 注明 "copied from Hendrycks' MATH release"）。改写时直接复用 `parse_ground_truth`/`extract_answer`/`math_equal`，不重复实现。Data 类调用 `super().infer_preprocess()` 拿基础 `inference_param` 再覆盖 `messages`，不要从零拼。

### 先例 2：SimpleQAEval（自定义包含判断）

```python
from sdk.eval_class.base_eval import BaseEval

class SimpleQAEval(BaseEval):
    @classmethod
    def get_sample_score(cls, sample, judge_kwargs):
        ref_answer = sample.get("ref_answer", "").lower()
        sample["score"] = []
        sample["gpt_response"] = []
        sample["extra_score"] = []
        for trial_responses in sample["responses"]:
            trial_scores = []
            for response in trial_responses:
                score = 1 if ref_answer in response.lower() else 0
                trial_scores.append(score)
            sample["score"].append(trial_scores)
            sample["gpt_response"].append([""] * len(trial_scores))
            sample["extra_score"].append([None] * len(trial_scores))
        return sample
```

要点：显式填全 `score`/`gpt_response`/`extra_score` 三个二维数组；单轮单次场景下 `responses = [["回答"]]`，对应 `score = [[1]]`。

### 先例 3：LLMJudgeEval（调用 Judge 模型）

```python
from sdk.eval_class.base_eval import BaseEval
from sdk.client import Client

class LLMJudgeEval(BaseEval):
    @classmethod
    def get_sample_score(cls, sample, judge_kwargs):
        client = Client(config=judge_kwargs.get("eval_model_config", {}))
        question = sample["messages"][-1]["content"]
        ref_answer = sample.get("ref_answer", "")
        sample["score"] = []
        sample["gpt_response"] = []
        sample["extra_score"] = []
        for trial_responses in sample["responses"]:
            trial_scores = []
            trial_judge_responses = []
            for response in trial_responses:
                judge_prompt = f"判断以下回答是否正确。\n问题：{question}\n参考答案：{ref_answer}\n模型回答：{response}\n请回答 \"正确\" 或 \"错误\"。"
                try:
                    judge_result = client.chat([{"role": "user", "content": judge_prompt}])
                    judge_answer = judge_result.choices[0].message.content
                    score = 1 if "正确" in judge_answer else 0
                except Exception as e:
                    judge_answer = f"Error: {e}"
                    score = -100000   # 错误标记
                trial_scores.append(score)
                trial_judge_responses.append(judge_answer)
            sample["score"].append(trial_scores)
            sample["gpt_response"].append(trial_judge_responses)
            sample["extra_score"].append([None] * len(trial_scores))
        return sample
```

要点：异常捕获后给 `-100000`（错误标记），而非 0；Judge 模型响应写入 `gpt_response` 便于回溯。生产场景优先用 `sdk/eval_class/call_judge_model_common.py` 而非自己拼 prompt。

### 先例 4：TriviaQA（多 gold / 别名列表 any-match，来自 lm-evaluation-harness 实测）

gold 是别名列表（`answer.aliases`），预测归一化后匹配任一 alias 即计 1。区别于先例 1-3 的单一 gold。

```python
import string
from sdk.eval_class.base_eval import BaseEval

ERROR_SCORE = -100000

def _normalize(text, ignore_case=True, ignore_punctuation=True):
    """对齐 lm-eval exact_match 的归一化: ignore_case + ignore_punctuation + 去空白。"""
    if not isinstance(text, str):
        text = str(text)
    if ignore_case:
        text = text.lower()
    if ignore_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())

class TriviaQAEval(BaseEval):
    @classmethod
    def get_sample_score(cls, sample, judge_kwargs):
        # gold 是别名列表（任一匹配即对），字段名沿用开源 answer.aliases
        answer = sample.get("answer", {})
        aliases = answer.get("aliases", []) if isinstance(answer, dict) else []
        if isinstance(aliases, str):
            aliases = [aliases]
        gold_norm = [_normalize(a) for a in aliases]

        sample["score"] = []
        sample["gpt_response"] = []
        sample["extra_score"] = []
        for trial_responses in sample["responses"]:
            trial_scores = []
            for response in trial_responses:
                if response is None or (isinstance(response, str) and response.strip() == ""):
                    trial_scores.append(ERROR_SCORE)   # 错误样本负分
                    continue
                pred_norm = _normalize(" ".join(str(response).strip().split()))
                hit = 1 if (pred_norm and any(pred_norm == g for g in gold_norm)) else 0
                trial_scores.append(hit)
            sample["score"].append(trial_scores)
            sample["gpt_response"].append([""] * len(trial_scores))
            sample["extra_score"].append([None] * len(trial_scores))
        return sample
```

要点：
- **gold 为列表时用 `any(pred == g for g in gold_list)` 任一匹配**（非单一 gold 的相等判断）。
- 空响应/None 用负分 `ERROR_SCORE`，不抛异常。
- normalize（lower + 去标点 + 去空白）对齐 lm-eval `exact_match` 的 `ignore_case`/`ignore_punctuation` 参数；若 metric 源码读不全，以 task config 参数为准先用标准 normalize，对齐校验时细调。

### 先例 5：AI2ARC（多选题 generate_until + 字母提取 + get_final_score 重写）

多选题适配为 generate_until：模型生成选项字母，Eval 类提取字母与 `answerKey` 比较。重写 `get_final_score` 按 `subset_name` 拆分准确率。来源 `sdk/eval_class/text_to_text/P1_eval/ai2_arc/metric_eval.py`。

```python
from sdk.eval_class.base_eval import BaseEval
import numpy as np

class AI2ARCEval(BaseEval):
    @classmethod
    def get_sample_score(cls, sample, judge_kwargs):
        sample['score'] = []
        sample['gpt_response'] = []
        sample['extra_score'] = []
        target = sample.get("answerKey", "").strip().upper()   # gold 字母 A/B/C/D
        original_responses = sample.get('original_responses', [])  # 保留原始生成便于回溯
        for trial_num, trial_responses in enumerate(sample.get('responses', [[]])):
            trial_original = original_responses[trial_num] if trial_num < len(original_responses) else []
            for idx, response in enumerate(trial_responses):
                predicted = response.strip().upper()   # 字母提取
                is_correct = (predicted == target)
                # 填 score (1/0)、gpt_response (Original/Extracted/Target/Match)、extra_score
        return sample

    @classmethod
    def get_final_score(cls, records):
        res = super().get_final_score(records)   # 复用基类 acc/bon/pass@k
        # 按 subset_name 拆分准确率（如 ARC-Easy / ARC-Challenge）
        subset_stats = {}
        for record in records:
            subset = record.get('subset_name', 'unknown')
            # ... 统计每个 subset 的 correct/total
            res[f'subset_acc_{subset}'] = round(correct / total * 100, 2)
        if 'error_rate' in res:
            del res['error_rate']   # 多选题不需要 error_rate
        return res
```

要点：
- **多选题字母提取**：模型生成字母，`response.strip().upper()` 提取，与 `answerKey` 比较。这是 loglikelihood 多选任务适配为 generate_until 的标准范式（HellaSwag/MMLU 同理）。
- **original_responses**：保留原始生成便于回溯，`gpt_response` 记录 Original/Extracted/Target/Match 对照。
- **get_final_score 重写**：调用 `super().get_final_score()` 拿基础指标，再按 `subset_name` 拆分，可删不需要的指标（如 `error_rate`）。

### 开源脚本 → BaseEval 标准改造点对照

| 开源脚本式 | BaseEval 式 |
|---|---|
| `def evaluate(pred, gold): return ...` 函数式入口 | `@classmethod get_sample_score(cls, sample, judge_kwargs)`，从 `sample["responses"][trial][turn]` 取 pred |
| 单条打分 `score = evaluate(resp, ans)` | 套两层循环：`for trial in responses: for turn in trial: ...`，产出二维 `score[trial][turn]` |
| 抛异常 / 返回 None 表示失败 | 返回负分 `-100000`，不抛异常 |
| 自带 IO、数据加载、CLI | 全部剥离，只留打分内核；数据加载由 Data 类负责 |
| 自定义字段名 | 沿用开源原版字段名（`answer`/`target`/`ref_answer` 等） |
| 单一 gold 答案 | gold 可能是列表/别名集，用 `any(...)` 任一匹配（如 triviaqa 的 `answer.aliases`） |
| 独立数学/比较工具 | 优先复用 `stem_utils/`、`patch_math_verify.py`、`sandbox_server`、`call_judge_model_common.py` |
| 自行汇总准确率 | 默认用 `get_final_score`（严格过滤 + acc/bon/pass@k）够用；需按子任务拆分/删指标时重写，调用 `super()` 再扩展（如 ai2_arc 按 subset） |
| loglikelihood 多选 | 适配为 generate_until（模型生成字母 + 字母提取匹配），参考 ai2_arc；分数不可对齐开源原版 |

### 其它已接入的开源评测集（可作参考）

`P1_eval/`：gsm8k、math、aime、minerva、olympiadbench、mgsm、docmath（共用 `stem_utils/`）；hotpotqa、popqa（开放域 QA，也在此目录）。
`T3_code_eval/`：humaneval_plus_pretrain、mbpp_plus_pretrain、bigcodebench_pretrain、livecodebench_pretrain（用 sandbox 执行）。
其它：bbh、ai2_arc_pretrain、mmmlu、longbench_v2 等。

接入新 Metric 前建议先 `ls` 同类目录，找一个最接近的先例对照改写。
