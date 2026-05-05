# CODE_CLEANUP_SUGGESTIONS.md
生成时间: 2026-05-05 (初始模板，等待 CI 首次运行后自动填充)

## A. 建议删除（高置信度死代码）
| 文件路径 | 理由 | confidence |
|---------|------|-----------|
| (待 CI 分析) | — | — |

## B. 建议移入 legacy/ 文件夹
| 文件路径 | 理由 | confidence |
|---------|------|-----------|
| (待 CI 分析) | — | — |

## C. 文件内部简化建议
| 文件路径 | 建议 | confidence |
|---------|------|-----------|
| (待 CI 分析) | — | — |

## D. 活跃文件摘要（供 agent 参考）
| 文件路径 | 近期改动次数 | 当前状态 |
|---------|-----------|--------|
| (待 CI 分析) | — | — |

## 说明
- confidence: high → heartbeat/coder 可自主执行删除或移入 `legacy/`（执行前必须 `grep` 验证无 import）
- confidence: medium → 写入 `PENDING_TASKS.md` 为 `[PENDING]` 任务，auto_launch=false，等用户确认
- confidence: low → 仅供参考，不做任何操作
- **绝对规则：移动/删除前必须确认文件未被任何 import 或 `__init__.py` 引用**
- 本文档由 `.github/workflows/ci_cleanup_suggestions.yml` 每周一自动生成（也可手动触发 `workflow_dispatch`）
- 触发条件：每周 Monday 02:00 UTC，或 main 分支中 `src/` 或 `scripts/` 有改动时
