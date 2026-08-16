---
name: reviewer-observation-right-attribution-may-be-mine
description: "四个 reviewer 说「artifact 缺少它声称发布的证据」完全正确, 但根因是我 freeze 时只手传了 2 个 --evidence (盘上 24 个); 归因错了就会去改论文而真正该改的是打包"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**Reviewer 的观察为真 ≠ 缺陷在他们指的那个对象上。收到指控先分「是论文的问题还是我的工具链的问题」，分错了就会修错东西。**

**Why:** 2026-08-16 paperC round_04，六个 codex reviewer 里四个（X1/X2/X5/X6）独立报同一条重指控：「frozen artifact 不含它反复声称要发布的 evidence records，所以 headline 实证无法独立核验」。X5（repro lens）把它列为**首要**扣分理由，明确写「主因不是 presentation」。

我去数了一下：`paperC/evidence/` 盘上 **24 个** artifact，`round_04/submission/` 只打了 **2 个**。根因是我调 freeze 时手写了 `--evidence` 列表：

```
python freeze_round.py paperC --round 4 \
  --evidence paperC/gate/build_record.json \
  --evidence paperC/evidence/claim_evidence_map.tsv
```

脚本忠实照做。**这是我的打包 bug，不是论文缺陷。** 如果我照 review 去「改论文」，会白改一通而 artifact 依旧不完整；正确动作是改打包接口 + 在完整 artifact 上重打分。

**同轮对照**：另一条同样四人指控（MMLU-Pro null 不合法）我复核后确认是**论文的真缺陷**。所以不是「reviewer 说的都得打折」，而是**每条都要独立定位根因**。

**How to apply:**
- 收到 review 后，对每条指控先答一个问题：**「如果这条为真，该改的文件在哪个目录？」** 落在 `code/` / 打包脚本 / harness 里的，就不要去改 `sections/`。
- **「默认不带 + 手动枚举」的接口一定会漏。** 改成「默认全带 + 显式排除」，并加断言。需要人手数 24 个文件名的参数就是设计缺陷。
- `missing_dependencies: []` 报空**不代表没漏** —— 它只查 manuscript 的 `\input` 链，不查正文点名引用的 evidence 标签（`\textsf{E-CAL}` 这类）。**要为「正文引用的标签是否都打进 snapshot」单独加检查并让它退出非 0。**
- 归因为打包的 issue，**其 review 分数需要在完整 artifact 上重打**，不能拿来驱动论文改写；但**不许自己去改 reviewer 的分数或结论** —— 只标注归因，裁定交给独立 meta-reviewer。
- 已发生的 snapshot hash 是**已发生事实的 provenance**，重新冻结要另建目录，**绝不覆盖** —— 覆盖了 round_04 的可审计性就没了。

见 [[agent-output-must-be-persisted-to-the-consumers-file]]、[[fix-the-class-not-the-instance]]、[[repo-checkers-are-writers-not-probes]]。
