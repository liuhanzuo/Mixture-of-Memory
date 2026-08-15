---
name: hand-composed-demo-strings-must-be-executed
description: "★手写的「最小复现」例子必须真跑一遍再入库: B11 两次把没有展示所声称机制的 demo string 写进 STATUS.json (row1 在 canonical 和 no-trunc 下都是 False, 死于 uniqueness 而非 truncation); 单变量对照要在函数级也成立, 不只在实验级"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**Rule**：任何「最小复现例子」/「demo string」/「反例」在写进记录或对外 issue 之前，
**必须真的执行一遍，并同时跑上「关掉被指控那一行」的对照**。两边输出不同，才算隔离出了那个机制。

**Why**（2026-08-15，B11 的 babilong scorer 审计，**同一个方向上第二次**）：

STATUS.json 里当作「line 27 first-period truncation 有害」证据的 row 1 是：

```
raw    = 'Choices: A. In the kitchen B. In the garden. The answer is kitchen.'
target = 'kitchen'
```

实测（`compare_answers(target, raw, question, task_labels)`，注意 `task_labels` 是必填参数）：

| | canonical | 关掉 line 27 | 结论 |
|---|---|---|---|
| row 1 | **False** | **False** | 两边都 False，**没有隔离 line 27** |
| `'The answer is A. kitchen'` | False | True | truncation 真的杀掉了正确答案 |
| `'John moved several times. He is in the kitchen'` | False | True | 同上 |
| `'kitchen is wrong. the answer is garden'` | **True** | False | truncation 还能**制造**正确答案（假阳性）|

row 1 两边都 False，但**死因不同**：canonical 截成 `'choices: a'`（一个 label 都不剩）；
no-trunc 留下 `kitchen` 和 `garden` 两个 label → 死于 `metrics.py:56` 的
**uniqueness 要求** `len(labels_in_output)==1`。**两个不同机制，都不是 line 27。**
拿它去上游开 issue，维护者一执行就打回。

B11 之前已经栽过一次（`dead_code_recheck_20260814.CORRECTION_to_K1`），**两次都是 line 27
悄悄替真正的机制干了活**。这不是手滑，是「合成例子时脑内模拟了一条 happy path，
没模拟其余的 filter」。

**同批的第二个教训 —— 「这条路径不可达」≠「这一行不可达」**：
断言说 `metrics.py:31` 的 `split('Question')` **unreachable**，因为 line 25 先 lowercase。
「`'Question' in s.lower()` 恒为 False」是对的（可对整个 Unicode 穷举证明，且 `str.lower`
逐码位幂等）；但 `sys.settrace` 实测执行行号 = `[25, 27, 29, 30, 31, 32]` —— **31 每次都执行**。
它是 **guaranteed no-op**，不是 dead code。措辞错了会被维护者一句「不，它会跑」关掉。

**How to apply**：
1. demo string 入库前跑双臂：`canonical` vs `去掉被指控那一步`。**输出相同 = 这个例子无效**，
   换例子，别改结论。
2. 报「某行不可达」前用 `sys.settrace` 看真实执行行号；区分
   **不可达（never executed）** / **恒不触发（executed, guard always false）** / **无副作用**。
3. 单变量原则要落到**函数级**：被测函数里每一个 filter 都是一个潜在变量
   （lowercase / first-period / `<context>` / uniqueness / labels_in_question 减法）。
   只想到自己关心的那个，就会把别的 filter 的效果记到它头上。
4. 声称「和上游一致所以行号可引用」时，**下载上游 HEAD 做 byte-diff**，不要只比 md5 说法。
   本次实测 `raw.githubusercontent.com/booydar/babilong/main/babilong/metrics.py`
   与 `third_party/babilong-pkg/` 副本 md5 均为 `0a5ecc52ade4e337d35b8f9c97c38310`，diff 为空 → 行号可引用。
5. 净效应符号依赖数据构成时，**必须分层报**：本例 overall `+0.16 pp`，
   但 LIST 格式 `-8.86 pp` vs 非 LIST `+0.25 pp` —— **符号翻转**。只报 overall 等于藏掉结论。

**Related**：[[fix-the-class-not-the-instance]]（同型：只修自己想到的那一个实例）、
[[a-range-is-not-a-measurement-until-it-clears-its-floor]]、
[[one-sample-is-not-a-trend-or-state]]、
[[read-the-trainer-docstring-before-designing-a-control]]（先读被测对象的真实签名/语义）。
