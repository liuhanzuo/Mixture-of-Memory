# Sources

- `../../../status/PAPERF_ACCNORM_VERIFIED.md`
- `../../../status/PAPERF_BS_LADDER_VERDICT.md`
- `../../../evidence_evalfragility_code/`
- `../../../status/ICC_DESIGN_EFFECT.md`

`status/PAPERF_ACCNORM_REDO.md` 已被归档为 superseded，不得作为证据。

## Analysis code (moved in on 2026-08-14)

- `code/analyze_b04_5rung.py` — OLMo-2-7B ladder analyzer. Emits
  `evidence/B04_6rung_bs16_analysis.json`. Cited by `DIRECTION_A_VERDICT.md`.
- `code/analyze_b04_qwen_6rung.py` — Qwen3-8B cross-family analyzer, identical margin
  definition. Emits `evidence/B04_Qwen_6rung_bs16_analysis.json`. Cited by
  `DIRECTION_A_QWEN_VERDICT.md`.

**Provenance of the move.** Until 2026-08-14 B04 occupied **two** sibling directories:
these two scripts lived alone in `../B04-eval-fragility/` (no `STATUS.json`, no
`PROPOSAL.md`), while every document lived here. That split meant a reader landing on
the code could not find the direction's verdict, and a reader landing on the verdict
had to know to look one directory over for the code. The scripts were `git mv`d here
(history preserved); `../B04-eval-fragility/README.md` is left as a pointer stub.

Both were verified to be the **only** copies before the move: searched the whole wzc1
`pighzliu_code/` tree (repo + parent) and, over ssh to `.73`, the whole zwfy6
`/apdcephfs_zwfy6/share_304376610/pighzliu_code/` tree — zwfy6 has no `B04-*`
directory at all. Per CLAUDE.md ("被活提案 SOURCES.md 或 code/ 引用 → 禁删") they were
moved, never deleted.

Downstream references repointed in the same commit: `DIRECTION_A_VERDICT.md:96`,
`DIRECTION_A_QWEN_VERDICT.md:145`, `status/SESSION_HANDOFF.md:27`,
`status/scout_21/lane4_cheapest_killer.md:179`,
`status/scout_21/LAUNCH_RANKING.md:{368,369,370,604}`,
`scripts/_run_b04_qwen_xfamily_21.sh:213`, and the cross-session memory note
`direction-a-eval-fragility-established.md`. Historical `.claude/projects/**/*.jsonl`
transcripts and `.claude/file-history/` snapshots were deliberately **not** edited —
they are archived provenance of what was true at the time.

