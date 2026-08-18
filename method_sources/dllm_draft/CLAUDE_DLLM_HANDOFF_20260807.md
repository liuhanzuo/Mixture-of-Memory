# dLLM / Scaffold-Coder 交接给 Claude

更新时间：2026-08-07 11:46（Asia/Shanghai）

## 0. 接管原则

1. **先读 `DLLM_RESULTS_20260807.md` 的第 0 节 retraction。**
2. 所有质量数字必须来自 EvalPlus 官方 grader；不要再使用旧
   `sandbox_exec` 结果。
3. 效率比较使用 `cumulative_model_tokens`，不要只比较 NFE。
4. 不要终止未知进程。LOCAL 当前有 Claude 自己启动的实验；`.104` 使用
   `ops/queue.tsv` / `ops/state/active_run.json` 管理注册任务。
5. Codex 对话心跳已停止，避免和 Claude 同时推进。

---

## 1. 当前工作树和分支

LOCAL 仓库：

```text
/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
HEAD = 3cef855
branch = main
```

`.104` 仓库：

```text
/apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104
HEAD = 3cef855
```

GitHub：

```text
repo   = THUtigerhfj/structural-dLLM
branch = lhz/elastic-scaffold
```

GitHub 分支是经过整理的公开分支；LOCAL `main` 含完整实验历史。LOCAL
当前 clean。

核心文档：

```text
DLLM_RESULTS_20260807.md          # 最新、最可信的论文级结果和撤回说明
ELASTIC_SCAFFOLD_EXPERIMENT_TODO.md
ELASTIC_SCAFFOLD_PROPOSAL.md
SEMANTIC_PRESERVATION_GATE.md
STAGE1_RESULTS.md
TODO.md
```

---

## 2. 已完成的工程推进

### 2.1 Scaffold runtime

已实现并测试：

- typed meta tokens / typed holes；
- mutable tree runtime；
- deterministic template expansion；
- line slots 和 rule-emitted indentation；
- clause constraints；
- token/line `[expand]`、`[delete]`；
- capacity limits 和 pressure instrumentation；
- partial failure cost；
- C1 leaf remask、C2 subtree backtrack、C3 structural deferral；
- 统一 EvalPlus generation sidecar；
- cumulative model tokens、NFE、canvas length、termination reason；
- 8-GPU sharded generation 和 FSDP/LoRA SFT。

### 2.2 训练与 matched controls

主要结果：

```text
Dream-Coder Instruct 512-NFE:
  HE+   70.7%   (LOCAL 最新正确协议)
  MBPP+ 68.0%

Stage-1 Scaffold:
  HE+   17.7–18.3%（不同 harness/run）
  MBPP+ 32.0–35.4%

Schedule-only:
  HE+  3.66%
  MBPP+ 11.38%

Plain matched SFT:
  HE+  21.95%
  MBPP+ 24.34%
```

结论：

- depth-banded schedule 明显有害；
- meta-token Scaffold 在 MBPP 上强于 matched plain，但明显弱于
  Dream-Coder Instruct；
- 旧 Base 初始化 + 五轮 SFT 有严重语义遗忘。

### 2.3 Teacher-KL / structural adaptation 探索

做过：

- semantic-preserving LoRA；
- LoRA scale calibration；
- compact PEFT trainable token rows；
- topology-only rows；
- frozen teacher lexical KL；
- leaf-only elastic pilot；
- `[STMT]` prior、statement length、expand prior calibration。

可信结论：

- teacher-KL 能保住 vanilla 能力（16-task vanilla 通常 15/16 parse）；
- `[STMT]` bonus 能把深度递归失败从 16/16 降为 0/16；
- 正确函数签名 + stmt=4 可到 8/16 parse；
- 增加固定 statement slots 会恶化；
- 64-step leaf-only training 后 `[expand]` 仍 0 次；
- expand bonus 1/2/4 也仍 0 次。

因此当前 `[expand]` 长度控制路线已止损。若重开，应该考虑显式 length
token/head，而不是继续盲目增加相同训练步数。

---

## 3. 论文上目前真正存活的 finding

详见 `DLLM_RESULTS_20260807.md` §2–§3b。

### 3.1 NFE 饱和曲线跨 benchmark 复现

Dream-Coder Instruct：

```text
HE+:
NFE 64/128/256/512/1024 = .018/.140/.555/.707/.707

MBPP+:
NFE 64/128/256/512/1024 = .029/.233/.545/.680/.680
```

两个 benchmark 都在 512 饱和，512→1024 精确 0 收益。

### 3.2 Scaffold capacity ladder

Full HumanEval+：

```text
Tiny/Small/Medium/Large HE+ = .000/.043/.177/.177
```

Full MBPP+：

```text
Tiny/Small/Medium/Large MBPP+ = .000/.156/.354/.325
```

Medium 是当前稳定 operating point。Large@512 有大量
`model_call_budget` 截断。

### 3.3 最强、可写的结果：token-cost Pareto 分段

不要说 “Scaffold 整体优于 Dream-Coder”。正确表述：

> Structural runtime 和 flat masked diffusion 占据 quality–compute Pareto
> 曲线的不同区段：低成本段属于 Scaffold，高质量段属于 flat diffusion。

关键数字：

```text
HE+ Scaffold Medium:
  13.8k model tokens -> .177

HE+ vanilla NFE32:
  21.9k model tokens -> .000

MBPP+ Scaffold Medium:
  7.1k model tokens -> .354

MBPP+ vanilla NFE32:
  19.3k model tokens -> .008
```

Vanilla 在 ≤22k token 区间几乎完全不能工作；但高质量上限
`.707/.680` 完全属于 vanilla。

---

## 4. 已撤回、绝对不要再引用的结论

### 4.1 旧 verifier refinement 数字

旧 verifier 没有比较函数返回值，只检查是否抛异常。空程序也可能被判
pass。相关 16 个 run 已重命名：

```text
runs/*.BROKEN_VERIFIER
```

不要引用。

### 4.2 “capacity-invariant rescue” / “small+refine dominates”

旧 `restart` policy 丢弃 Scaffold draft，从 prompt 重新跑 Dream-Coder。
因此 grid 实际测的是 Dream，不是结构修订。两条 headline finding 已撤回。

真正有意义的 refinement 必须实现：

```text
typed subtree collapse
→ 保留其余 Scaffold tree
→ 局部 re-diffuse
```

也就是 `DLLM_RESULTS_20260807.md` 中的 Arm C。该 arm 尚未实现。

---

## 5. 当前正在运行的任务

### 5.1 LOCAL：Claude 外部任务，严禁另一个 agent 终止

```text
scripts/_run_scaffold_large_2xbudget.sh
run: runs/scaffold_large_heplus_budget1024
parent Claude PID tree starts at 4018139
```

交接时进度：

```text
151 / 164 tasks
resolved = 139
model_call_budget = 12
remaining = 13
```

仍有 rank 0/1/4/5 的极长任务在跑。即使 budget 从 512 加到 1024，
仍有长尾触顶，初步说明 Large 的问题不只是 512 太小。让它自然完成；
脚本随后会 merge、EvalPlus，并继续 MBPP+ Large@1024。

### 5.2 `.104`：Dream-Coder Base 官方协议 smoke，当前 NEEDS_DEBUG

注册 ID：

```text
DREAM-CODER-BASE-PROTOCOL-HE16-8GPU-001
```

生成本身完成：

```text
16 tasks
generation errors = 0
parseable = 4/16
```

EvalPlus 失败原因不是模型，而是 16-task samples 配完整 164-task
`HUMANEVAL_OVERRIDE_PATH`，触发：

```text
AssertionError: Missing problems in samples
```

产物：

```text
.104/outputs/dream_base_protocol_he16/solutions.jsonl
.104/outputs/dream_base_protocol_he16/metrics.jsonl
```

下一步若要继续：构造 matched 16-task override 文件，或 smoke 只做 parse
gate。注意 parse 只有 4/16，未达到原定 8/16，因此官方协议修正尚不能说
复现成功。

---

## 6. 最近新增的 Base 协议修正

Commit：

```text
3cef855 align Dream-Coder Base continuation protocol
```

`scripts/generate_evalplus_dream.py` 新增：

```text
--add-bos-token
--base-continuation
```

行为：

```text
input = BOS + raw benchmark function prefix
solution = prompt + generated continuation
```

这比旧 `--no-chat` 更接近官方 lm-eval。官方参考：

```text
vendor/Dream-Coder/base/eval_code_base.sh
vendor/Dream-Coder/base/lm_eval/models/diffllm.py
```

旧 Base 数字 `HE+=.079 / MBPP+=.159` 仍只能写作“协议未对齐”，不能写
“复现失败”。

---

## 7. 推荐 Claude 接管后的优先顺序

### P0-A：收尾 LOCAL Large@1024 falsifiability

1. 等当前 HE+ 自然结束；
2. 检查 merge 和 EvalPlus；
3. 让脚本继续 MBPP+；
4. 比较 Large@512 与 @1024：
   - pass@1；
   - model_call_budget hits；
   - cumulative model tokens；
   - wall-clock。
5. 更新 `DLLM_RESULTS_20260807.md`。

这是当前最直接、已运行中的论文 falsifiability test。

### P0-B：冻结可写 claim 和图表

围绕唯一存活主张：

```text
token-cost-matched Pareto segmentation
```

生成：

- HE+/MBPP+ 两张 Pareto 图；
- NFE saturation 曲线；
- capacity ladder 表；
- termination/failure-tail 图；
- 明确标注高质量区仍由 vanilla 占据。

### P1：Arm C typed subtree collapse

如果继续方法创新，这是最值得投入的方向。不要再做 restart refinement。
需要真正保存/重建 Scaffold runtime tree，并只 collapse verifier 定位的
subtree。

### P2：Base 官方协议

Base 不是主线阻塞项。若要清除：

1. matched smoke dataset；
2. 先确认 prompt+continuation extraction；
3. 再 full HE+/MBPP+；
4. 或安装 vendor lm-eval 缺失依赖并完全照官方脚本。

---

## 8. 环境与访问

LOCAL：

```text
root = /apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
python = /opt/conda/envs/dllm-env/bin/python
GPU = 8×L20A
```

`.104`：

```text
host = 28.83.24.104:36000
root = /apdcephfs_zwfy6/share_304376610/pighzliu_code/dllm_draft_104
password file =
/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/configs/password_h20_24104.txt
python = .venv_dream/bin/python
GPU = 8×H20
```

SSH：

```bash
sshpass -f "$PASSWORD_FILE" ssh -p 36000 \
  -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=ops/ssh/known_hosts \
  root@28.83.24.104
```

不要提交密码或 `SERVER_ACCESS.md`。

---

## 9. 最短接管检查清单

```text
[ ] 阅读 DLLM_RESULTS_20260807.md §0 retractions
[ ] 检查 LOCAL Large@1024 进程，不要重复启动
[ ] 检查 .104 active_run/queue，归档 Base smoke NEEDS_DEBUG
[ ] 不引用 *.BROKEN_VERIFIER
[ ] 效率一律用 cumulative model tokens
[ ] 收尾 Large@1024 后更新最新结果文档
[ ] 论文主张限定为低成本/高质量 Pareto 分段
```
