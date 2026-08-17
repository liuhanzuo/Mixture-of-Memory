---
name: wrong-interpreter-reads-as-a-content-regression
description: "我用缺 PyMuPDF 的 .venv 跑 paperC gate 得 rc=2, 就把它归因给 commit ac59854 说是内容回归; 换 conda 后同一 gate PASS(26/26 页 sha 全对) —— ModuleNotFoundError 被我读成了 FAIL"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**一个 gate 失败时，先确认它是**因为被测对象错了**还是**因为解释器/环境缺件**。
`ModuleNotFoundError` 不是内容缺陷，而 gate 常常把它打印成 `FAIL:` 。**换解释器复跑再归因。**

**Why:** 2026-08-17 我跑 paperC 的 10 个 gate，用 `.venv/bin/python`：

| gate | `.venv` | conda | 真相 |
|---|---|---|---|
| `gate_build_record_matches_pdf` | **rc=2 FAIL** | **rc=0 PASS** | 内容完全正确：pdf_bytes/sha256/**pages 26==26** 三项全 OK |
| `gate2_crossfamily_nulls` | rc=1 (`No module named numpy`) | rc=2 (缺 argparse positionals) | 真·pre-existing：从未接线 |

`.venv` **没有 torch / numpy / PyMuPDF**（CLAUDE.md 早已记「LOCAL 的 .venv 现也已无 torch」，
我读过这条还是踩了）。gate 自己把缺件写成：

> `FAIL: cannot measure pdf_pages on this host (ModuleNotFoundError("No module named 'fitz'"));
>  the build record's pdf_pages=26 is therefore UNVERIFIED`

**它的措辞是诚实的**（"UNVERIFIED"，不是 "MISMATCH"），是我把 UNVERIFIED 读成了 FAILED。

**最贵的一步在归因**：我此前据这个 rc=2 断言「它在 `ac59854^` 是 rc=0、在 `ac59854` 是 rc=2，
所以是那个 commit 引入的回归」。**父提交对照本身是对的方法，但两次都跑在缺件的解释器上**，
于是我拿一个恒定的环境故障去证明一个不存在的代码变更。**同一个坏源跑两次不产生对照。**

**How to apply:**
- gate/checker 失败先跑一行分诊：`<py> -c "import fitz,numpy"`。缺件 → 换解释器复跑，
  **然后才**谈内容。paperC 的 gate 要用 `/opt/conda/envs/torch-base/bin/python`，不是 `.venv`。
- **区分三种非零 rc**：(a) 被测对象真错、(b) 环境缺件、(c) 工具没接线（缺参数）。
  三者都打印 FAIL，处置完全不同 —— 只有 (a) 该改论文。
- **"pre-existing" 这个判断必须在同一环境下做**。我说 gate2 pre-existing 是对的（两个环境都非零，
  且换环境后错因才显出真面目），说 build_record 是回归是错的 —— 差别就在有没有换源复跑。
- 同族：[[absence-on-path-is-not-absence-on-disk]]（这条是它的解释器版：**在一个解释器里缺失
  不等于在主机上缺失**）、[[repo-checkers-are-writers-not-probes]]（先存缺陷必须在干净树复跑）、
  [[a-pipe-makes-a-failing-command-report-success]]（rc 读错的另一套外衣）。
  共同点：**我把工具的一句话当成了被测对象的属性。**
