---
name: venue-verify-acl-family-needs-anthology
description: ★venue 核实分两套：ICLR/NeurIPS/ICML 系用 OpenReview venueid；ACL/NAACL/EMNLP(含 Findings) 必须用 ACL Anthology + DBLP，OpenReview 查不到会误判成 preprint
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**venue 核实规则必须按会议家族分流，不能一律 OpenReview。**

- **OpenReview 系**（ICLR / NeurIPS / ICML / 及其 workshop）：用 `venueid` + `Camera_Ready_Revision`。S2 和 DBLP 对这些**滞后**，会把已接收论文报成 arXiv preprint。这是 [[venue-verify-must-use-openreview-2026]] 记的教训。
- **ACL 系**（ACL / NAACL / EMNLP / EACL / AACL，**含 Findings**）：**权威记录是 aclanthology.org，DBLP 索引及时。** 这些会议**不以最终标题走 OpenReview**，所以拿标题查 OpenReview 会返回噪声命中，导致误判为 preprint。

**Why**：2026-08-07 我把「只有 OpenReview venueid 才权威」写进了一个 workflow 的 hard rule，sweep agent 照做，把 `arXiv:2410.15225` 报成 "UNVERIFIED / treat as preprint"（OpenReview 前几名命中是 "Autonomous Car Chasing" 之类）。deep-read agent 用 DBLP 一查即得：**Findings of NAACL 2025, pp.1943-1957, DOI 10.18653/v1/2025.findings-naacl.103**。MAIN 独立用 DBLP API 复核确认。**agent 的执行是对的，是我的规则错了**——把 OpenReview 教训过度泛化成了「OpenReview or bust」，代价是在该 angle 最重要的那篇 skeptical 论文上丢掉了同行评审资格认定。

**How to apply**：
- 判 venue 前先看会议家族。ACL 系 → Anthology + DBLP；OpenReview 系 → OpenReview venueid。两套都查最稳。
- **ACL 系论文的 PDF 正文标题可能与 Anthology metadata 不一致**。同一篇 2410.15225 的 PDF body 印 "Chasing Random – Investigating the 'Gains' achieved through Instruction Selection Strategies at Scale"，而 arXiv metadata 与 Anthology 都给 "Chasing Random: Instruction Selection Strategies Fail to Generalize"。**引用必须用 Anthology 形式**，否则审稿人搜不到。
- **对任何同行评审论文，都要 diff arXiv 版 vs camera-ready。** 同一篇的 camera-ready §A.4 有一句 arXiv v1 完全没有的让步（"gains from selection are definitely more pronounced at our largest budget"），而那句恰是对我们最不利的证据。**审稿人读 camera-ready。**
- 与 [[prior-work-differentiate-dont-abandon]] 配合：venue 核实用于精确定位（判 concurrent 需准确日期），不用于劝退。
