---
name: a-green-checker-covers-only-what-it-targets
description: "paperC 的 check_prose_vs_evidence 报 91/91 OK 0 mismatch, 而同一天我在正文里查出三个 prose-vs-evidence 缺陷(14/15、4.6×、nine flips); 它 min_decimals=3 只看 ≥3 位小数的 floor/chance 字面量, 整数计数在它的目标空间之外"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**「checker 全绿」只说明*它所瞄准的那些*断言没问题。报数字之前先读它的 target 定义，否则绿色会被当成覆盖。**

**Why:** 2026-08-16 我在 paperC 正文里查出**三个** prose-vs-evidence 缺陷：

1. abstract/intro 仍写已被撤回的分母 `14/15`（正文 §5/§6 已是 `15/17`）；
2. `03b_nulls.tex:12` 的 `4.6×` 实际是 `0.532164/0.125914 = 4.2264`；
3. `tab_integrity` 写 "all **nine** flips"，证据 JSON 三种独立数法都是 **8**。

而同一份 `code/check_prose_vs_evidence.py` 在**每一次**都报
`n_checked=91 n_ok=91 n_mismatch=0 n_uncovered=0`，`verdict: PASS`。它没坏 —— 读它自己的 schema 就明白：

- `what`: "prose/.tex statements of **each construct null**, checked against the full-precision value"
- `min_decimals: 3` ← **只看 ≥3 位小数的字面量**
- site 规则要求那一行**点名 construct** 且带 floor/null marker
- `excluded_label_prefixes` 还主动排除 p 值 / CI / seed

所以 **`14/15`、`nine`、`4.6` 这类整数或 1 位小数的计数，从来不在它的目标空间里**。
`n_uncovered=0` 指的是「我列出的 targets 都覆盖到了」，**不是**「正文所有数字都覆盖到了」。

**How to apply:**
- 引用某个 checker 的绿色结论前，**先打印它的 `what` / target 规则 / 阈值**，用一句话写清它检的是什么类。
  报告里要写「X 类断言 91/91 通过」，**不能写「prose 与 evidence 一致」**。
- **计数类断言（分母、flip 数、cell 数、n/N 比）需要单独的 gate**，因为它们通常是整数、位数少、
  且往往跨文件（正文一处、证据一处、appendix 再一处）。我这次给分母加了
  `gate_designated_denominator.py` 的 CHECK 6（扫 prose 里被撤回的比值）；flip 计数目前仍无 gate。
- **「三个缺陷全都躲过同一个 checker」是关于 checker 的信息，不是关于运气的。** 出现这种模式，
  立刻去读它的 scope，而不是继续人工找第四个。
- 同族：[[selftest-over-invented-inputs-proves-nothing-about-the-pipeline]]（selftest 用编造输入 → 通过也无意义）、
  [[a-pipe-makes-a-failing-command-report-success]] 与 [[capture-rc-before-any-command-substitution]]
  （rc 本身就是伪造的）。三者共同的形状：**我的验证工具报告成功 ≠ 产物本身正确。**
