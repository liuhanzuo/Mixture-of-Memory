---
name: venue-verify-must-use-openreview-2026
description: "★2026-08-06 AUDIT0 教训: 2026 会议论文的 S2/DBLP 有回填滞后, 靠 S2 一路会把 ICLR'26/ICML'26 Poster 全误判为 preprint (2411.15558/2605.02105/2602.11137/2602.14486/2506.11389/2510.10071/2601.20009 七篇同错); venue 核实必须多通路交叉"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**Rule**: 核 venue 时**必须多通路交叉**，尤其对 2610-2611 前缀（2026 会议论文）：

1. **S2 `paper/arXiv:<id>`（配额 A）**
2. **S2 `paper/search/match?query=<title>`（配额 B，独立于 A）** —— 429 时 A 死了 B 还可用
3. **arXiv abs 的 `citation_journal_title` / `jref` / COMMENT** —— 作者自述
4. **DBLP `search/publ/api` 标题检索 + 若无果补作者集检索** —— camera-ready 常改标题
5. **OpenReview API2 `notes/search?query=<title>`** —— 查 `venueid` + invitations

> ⚠️⚠️ **只有 `notes/search` 这一个 endpoint 可用；`notes?id=` / `notes?forum=` 会返 403
> `ChallengeRequiredError`（MAIN 2026-08-15 实测，两个 endpoint 同一次会话内一个 403 一个 200）。**
> **不要因为 `notes?id=` 403 就断定「OpenReview 挂了、camera-ready 无法核实」** —— 2026-08-15
> 一个 agent 正是这么判的，于是整轮 **0 次 camera-ready 核查**，四个 venue 只能记 second-hand；
> 同一天另一个 agent 用 `notes/search` **成功核出** `2605.07271 = venueid=ICML.cc/2026/Conference`
> + `Camera_Ready_Revision`（我复核确认：`notes?id=` → 403，`notes/search` → 200，返回完整
> `venueid` / `venue` / `invitations` / `pdate`）。
> **判据：「某个 endpoint 403」≠「该权威源不可用」。换 endpoint 再试，403 的那个可能只是需要鉴权。**
> 一行自查：
> `curl -s -o /dev/null -w '%{http_code}' "https://api2.openreview.net/notes/search?query=<title>&limit=3"`

**Why**: 2026-08-06 AUDIT0 独立核实 49 篇 venue，发现 **7 篇 2026 会议论文** S2/DBLP
全部滞后返回 `arXiv.org` / corr-only，OpenReview 才是权威来源：

| arXiv | S2 显示 | OpenReview 显示 |
|---|---|---|
| 2411.15558 | arXiv.org | **`venueid=ICLR.cc/2026/Conference`, Camera_Ready_Revision** = ICLR 2026 Poster |
| 2605.02105 SAM | arXiv.org | **ICML 2026 Submission19173 Camera_Ready** |
| 2602.11137 wd plasticity | arXiv.org | **ICML 2026 Submission26903 Camera_Ready** |
| 2602.14486 Aristotelian | arXiv.org | **ICML 2026 Submission15852 Camera_Ready** |
| 2506.11389 CGLS | arXiv.org | **ICML 2026** |
| 2510.10071 ADEPT | arXiv.org | **ICLR 2026 Poster** |
| 2601.20009 LinguaMap | arXiv.org | **ICLR 2026 Poster** |

**MAIN 2026-08-05 就因单走 S2 而把 2411.15558 判成 preprint**，然后据此判「Paper C 构造被
一篇 preprint 全占」，一错传两错（venue 错 + 后来核实构造也读错 —— 它训继承末层不是随机新层）。
详见 [[paperc-pc1-scooped-eval-invalid]]。

**How to apply**:
1. **看到 S2 返回 `venue='arXiv.org'` 或 `publicationVenue=None`，2026 前缀（26xx）一律不下 preprint 结论**，
   立刻查 OpenReview
2. OpenReview 判定规则：
   - **`venueid=<Conf>.cc/<Year>/Conference` + 存在 `Camera_Ready_Revision` invitation**
     → **peer-reviewed 主会/主 track**
   - `venueid` 里含 `Submission` / `Withdrawn_Submission` / `Rejected_Submission` / `TMLR/Rejected`
     → **不算录用**
   - `venueid=<Conf>.cc/<Year>/Workshop/<Name>` → **workshop（除非有 DBLP `conf/` key，
     BlackboxNLP 是特例）**
3. 主 track vs Findings 只能靠 **DBLP `dblp.org/rec/<key>.xml` 的 `booktitle`**（`ACL (1)` = main，
   `ACL Findings` = Findings；S2 的 `venue` 字段不区分）
4. arXiv `journal_ref` 通常滞后半年，**camera-ready invitation 存在即代表录用生效**
5. 429 是限流不是"无 venue"，重试到 200 再下结论；`paper/search/match` 是**独立配额**

**Related**: [[paperc-pc1-scooped-eval-invalid]]（第一次被坑的具体案例）、
[[two-disk-rule-applies-to-main-too]]（"不存在"重指控门槛要高的同型教训）、
[[tcodex-exec-no-dash-c-flag]]（另一处方法学坑）。
