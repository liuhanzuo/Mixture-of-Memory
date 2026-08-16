---
name: repo-checkers-are-writers-not-probes
description: "paperC 的 check_prose_vs_evidence.py / validate_tex_static.py 会写 evidence/*.json; 对着 symlink 的 evidence 跑过之后, 它自己的输出就不再是关于干净树的证据"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**`paperC/code/check_prose_vs_evidence.py` 和 `validate_tex_static.py` 是 WRITER，不是 read-only probe** —— 它们把结果 JSON 写进 `paperC/evidence/`。

**Why:** 2026-08-16 一个 agent 把 **live** `paperC/evidence/` symlink 进了自己的 scratch build，于是这两个 checker 覆盖了真实树里的
`evidence/tex_static_validation.json` 和 `evidence/prose_vs_evidence_check.json`（它用 `git checkout --` 还原了）。
更要紧的后果：它在那个被污染的环境里看到「1 个 pre-existing mismatch（`README.md:23` 的 `0.2845` vs 存储的 `0.268908`）」，
并把它当作**先存回归**写进交付物。我在干净树里复跑得到 `n_checked=81 n_ok=81 n_mismatch=0`，rc=0 —— **不存在这个回归**。
`check_prose_vs_evidence.py:67` 明确把 `0.2845` 认定为正确值（README.md:23 的 MMLU content null），
`README.md:214` 把 `0.2689` / `0.2845` 并列为两个不同 floor，而 `0.2845` 只在 README、零个 `sections/*.tex`。

**How to apply:**
- 跑这两个 checker 前**先备份它的输出 JSON，跑完立刻还原，并确认 `git status paperC/evidence/` 为空** —— 除非你就是要更新它们（对真实树跑时是正当的）。
- **绝不把 scratch build 的 `evidence/` 指向 live 目录**；给它自己的拷贝。
- 干净树基线是 `n_checked=81 n_ok=81 n_mismatch=0` / rc=0。**看到 mismatch 就是自己引入的，要修，不能记成「先存问题」。**
- 一般化：**一个 writer-checker 一旦在被污染的环境里跑过，它自己的输出就不再是关于干净树的证据。** 报「先存缺陷」前先在干净树复跑一次。

见 [[reporting-a-gap-is-not-closing-it]]、[[agent-output-must-be-persisted-to-the-consumers-file]]。
