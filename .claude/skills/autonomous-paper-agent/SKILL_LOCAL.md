# SKILL_LOCAL.md — 本项目版覆盖层

上游 `SKILL.md` **原样保留、不改**（provenance）。凡与本项目环境冲突之处，**以本文件为准**。
操作入口是 `/paper`（`.claude/commands/paper.md`）。

## 1. 三条上游断言在本项目是错的 / 需要改写

| 上游 | 本项目实测 |
|---|---|
| `README.md:74-95` 装到 `.claude/skills/` + `.claude/agents/` 即可被发现 | `.claude/agents/` **从不存在、机制未验证**。已验证可用的入口是 **`.claude/commands/*.md`**（`/paper` 已注册成功）。payload 放 `.claude/skills/autonomous-paper-agent/` 但**按路径读取**，不依赖自动发现。 |
| `SKILL.md:38-47` 优先用 `paper-reviewer` 等自定义 agent | 用 **`Workflow`**（`workflows/review_round.js`）。它已验证可用，且 `agent(…, {schema})` 强制 JSON 校验 —— reviewer 返回不合 schema 会重试，不会静默变成「少一篇审稿」。 |
| `SKILL.md:503` "the LaTeX package compiles" 是普通 gate | **可真实强制**（见 §2），已在 paperC 上跑通。但 `:503` 的 "PDF visually inspected" **在本机永远做不到**（无 pdftoppm/gs/mutool/PyMuPDF），`build_record.json` 里恒为 `false` 并附原因。 |

## 2. LaTeX：工具链在盘上，只是不在 PATH

`.texlive/2026/bin/x86_64-linux/` = **142 个二进制**（pdflatex/lualatex/xelatex/latexmk/bibtex/biber）。
实测 paperA rc=0/29 页；**paperC rc=0 / 16 页 / 293609 B / 0 undefined ref / 0 undefined cite / 0 overfull / 0 error**。

⚠️ 两份历史记载都说「没有 latexmk」——`paperA/V6_ITERATION_NOTES.md:82` 和 2026-08-14 某 subagent 报告。
**两者都只反映 `$PATH`，不反映磁盘。** 与 paperC E1（"没有 python 有 pyarrow"，真相是 `venv_union9` 一直有）同一类错误。
**判据：「工具不可用」需要枚举位置，不是问一次 shell。**

## 3. 脚本分工（为什么不直接用上游/既有的）

| 脚本 | 状态 | 说明 |
|---|---|---|
| `scripts/paper_build.py` | **新写** | 唯一编译入口。prepend `.texlive`，发现式识别 sty/bib，统计 undefined ref/cite + overfull + 缺失 `\input` + 未解析 `\cite`；**gate 失败 exit 1**。页数从 TeX log 读（PDF 用 `/ObjStm` 压缩，grep `/Type /Page` 会得 0 —— 第一版就栽在这，已修）。 |
| `scripts/check_numbers.py` | **新写** | 补 paperC 自陈的 E3（无 tex↔evidence 绑定）。每个数字要么直配、要么是**正确四舍五入**、要么是两个证据数之差；否则进 `unmatched_needs_human`。正确舍入判定能抓到 `paperA/audit_20260806` 记录的真实 bug（磁盘 99.187 被写成 99.20，正解 99.19）。 |
| `scripts/freeze_round.py` | **新写** | 合并两者优点：**依赖闭包**（取自 `scripts/freeze_paper_version.py`）+ **`round_NN/` 和 hashed MANIFEST**（取自上游 `make_review_snapshot.py`）。venue style **发现而非假设** —— `freeze_paper_version.py:32-33` 硬编码 `acl.sty`，指向 paperC 会 FileNotFoundError。并**强制排除** `review_rounds/`/`review_history/`/`tcodex_out/`/`SCORE_HISTORY`/`WRITER_NOTES`，否则盲审直接失效。 |
| `scripts/aggregate_reviews.py` `select_best_round.py` `make_review_snapshot.py` | **上游原样** | 未改一字节（`cmp` 已验证）。`select_best_round.py` 的 `round_(\d+)$` 正则是我们用 `round_NN/` 命名的原因。 |
| `workflows/review_round.js` | **新写** | 6 审并行 → meta → 逐 issue 对抗式验收。<4/6 返回即 **abort**（部分 panel 的 median 不可解释）。 |

## 4. 既有资产必须复用，不要另造

- `review_prompts/{STRICT,NORMAL}_REVIEW_TEMPLATE.md` —— 10 步 evidence audit + strict/normal 双校准，
  **本项目最值钱的自制资产**，已内联进 `review_round.js` 的 reviewer prompt。
- `REVIEW_PROTOCOL.md` —— 3 strict + 3 normal + 独立性 + **保留 outlier 不平均掉**。
- `paperA/paperB/review_history/SCORE_HISTORY.md` —— ARR 1-5 历史趋势 + **calibration warning**
  （不同 prompt 代际不可纵向比）。新一轮追加时必须标代际。
- `paperB/scripts/generate_appendix_tables.py:175-212` (`write_integrity`) —— tex 数字从 `summary.json` 现算，
  不手输。`check_numbers.py` 是它的通用化版本。
- **legacy `vN_*` 快照原样不动**。新轮次走 `paper<X>/review_rounds/round_NN/`，两套并存。

## 5. 已知未做 / 诚实边界

1. **PDF 视觉检查做不到**（无渲染工具）⇒ integrity gate 该项恒 OPEN。
2. `check_numbers.py` 的「差值派生」是启发式：负测里 `12.3456` 曾被两个无关证据数之差偶然命中。
   因此 `derived_as_difference` **单独计数**，不与 `direct_match` 混同。
3. **venue 核实必须联网**，且分两套家族（`CODEBUDDY.md:147-151`）。脚本不做，由 `/paper` §5 人工/agent 走。
4. 6 审 + meta + N 个 verifier 是**真花 token** 的。默认 0 GPU，但不是 0 成本。
