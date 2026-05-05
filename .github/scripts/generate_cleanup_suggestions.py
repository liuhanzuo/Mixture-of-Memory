#!/usr/bin/env python3
"""
CI script: LLM-driven code cleanup analysis.

Reads:
  - Full .py file tree (src/, scripts/, tests/)
  - CLAUDE.md abandoned-direction keywords
  - Recent git log (60 days)
  - Previous CODE_CLEANUP_SUGGESTIONS.md (for continuity)

Outputs: CODE_CLEANUP_SUGGESTIONS.md with three sections:
  A. Candidates for deletion (high confidence dead code)
  B. Candidates for legacy/ folder (obsolete but historically valuable)
  C. In-file simplification suggestions

Agent autonomy rules embedded in the doc:
  - confidence: high  → heartbeat/coder can delete/move autonomously
  - confidence: medium → write to PENDING_TASKS.md, await user confirmation
  - confidence: low   → reference only, no action

Called by .github/workflows/ci_cleanup_suggestions.yml
Requires: ANTHROPIC_API_KEY environment variable.
"""

import subprocess
import datetime
import pathlib
import os
import sys

try:
    import anthropic
except ImportError:
    print("anthropic package not found, skipping cleanup suggestion generation")
    sys.exit(0)


ROOT = pathlib.Path(".")
SKIP_DIRS = {"__pycache__", ".git", "third_party", ".venv", "venv", "build", "dist"}


def get_file_tree():
    """Collect all .py files, skipping irrelevant dirs."""
    result = []
    for p in sorted(ROOT.rglob("*.py")):
        rel = p.relative_to(ROOT)
        parts = set(rel.parts)
        if parts & SKIP_DIRS:
            continue
        result.append(str(rel))
    return result


def get_git_log_summary():
    """Recent 60-day commit history showing touched files."""
    try:
        out = subprocess.check_output(
            ["git", "log", "--since=60 days ago", "--name-only",
             "--format=COMMIT: %s (%cd)", "--date=short", "--diff-filter=ACMD"],
            stderr=subprocess.DEVNULL
        ).decode()
        return out[:6000]
    except subprocess.CalledProcessError:
        return "(git log unavailable)"


def read_claude_md_abandoned():
    """Extract lines from CLAUDE.md that mention abandoned directions."""
    try:
        content = pathlib.Path("CLAUDE.md").read_text(encoding="utf-8")
        keywords = ["放弃", "FAILED", "dead end", "已放弃", "已完成", "不再", "ceiling"]
        lines = [l.strip() for l in content.split("\n")
                 if any(kw in l for kw in keywords) and l.strip()]
        return "\n".join(lines[:40])
    except Exception:
        return ""


def read_existing_suggestions():
    """Read previous suggestions for continuity."""
    p = pathlib.Path("CODE_CLEANUP_SUGGESTIONS.md")
    if p.exists():
        content = p.read_text(encoding="utf-8")
        # Only send a truncated version to avoid context overflow
        return content[:3000]
    return "(no prior suggestions)"


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set, writing placeholder")
        write_placeholder()
        sys.exit(0)

    file_tree = get_file_tree()
    git_log = get_git_log_summary()
    abandoned = read_claude_md_abandoned()
    prior = read_existing_suggestions()
    today = datetime.date.today().isoformat()

    file_list_str = "\n".join(file_tree)
    # Cap to avoid token overflow
    if len(file_list_str) > 8000:
        file_list_str = file_list_str[:8000] + "\n... (truncated)"

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2500,
        messages=[{
            "role": "user",
            "content": f"""你是 Mixture-of-Memory 项目的代码审查专家。
本项目研究方向：固定大小 memory buffer 压缩长上下文（研究项目，不是产品）。

已放弃的方向（从 CLAUDE.md 中提取）：
{abandoned}

===== 当前 Python 文件列表 =====
{file_list_str}

===== 近 60 天 git 提交历史 =====
{git_log}

===== 上次建议（供参考，避免重复） =====
{prior}

请生成《代码清理建议》文档。输出纯 Markdown，不要输出其他内容。
文档需包含以下内容：

# CODE_CLEANUP_SUGGESTIONS.md
生成时间: {today}

## A. 建议删除（高置信度死代码）
<!-- confidence: high = 文件对应已明确放弃的方向，且近 60 天内无 git 活动，几乎确定不再使用 -->
| 文件路径 | 理由 | confidence |
|---------|------|-----------|
| ... | ... | high/medium |

## B. 建议移入 legacy/ 文件夹
<!-- confidence: medium = 历史上有参考价值（如 RMT v3-v5 的实现思路），但现在不活跃 -->
<!-- 移动后文件仍可访问，适合 "可能有参考价值但不需要在主目录" 的情况 -->
| 文件路径 | 理由 | confidence |
|---------|------|-----------|
| ... | ... | medium |

## C. 文件内部简化建议
<!-- 仅列出特别明显的冗余（如有多个相同功能的函数、大段注释掉的代码等） -->
| 文件路径 | 建议 | confidence |
|---------|------|-----------|
| ... | ... | low/medium |

## D. 活跃文件摘要（供 agent 参考）
<!-- 列出近 60 天最活跃的 5-10 个文件，帮助 heartbeat 了解当前工作焦点 -->
| 文件路径 | 近期改动次数 | 当前状态 |
|---------|-----------|--------|
| ... | ... | 活跃/观察中 |

## 说明
- confidence: high → heartbeat/coder 可自主执行删除或移入 legacy/（执行前必须 grep 验证无 import）
- confidence: medium → 写入 PENDING_TASKS.md 为 [PENDING] 任务，auto_launch=false，等用户确认
- confidence: low → 仅供参考，不做任何操作
- **绝对规则：移动/删除前必须确认文件未被任何 import 或 __init__.py 引用**
- 本文档由 CI 自动生成，不代表最终决策，agent 有权降级 confidence 级别
"""
        }]
    )

    output = resp.content[0].text.strip()
    pathlib.Path("CODE_CLEANUP_SUGGESTIONS.md").write_text(output, encoding="utf-8")
    print(f"Written CODE_CLEANUP_SUGGESTIONS.md ({len(output)} chars)")


def write_placeholder():
    today = datetime.date.today().isoformat()
    placeholder = f"""# CODE_CLEANUP_SUGGESTIONS.md
生成时间: {today} (CI 无 API key，跳过分析)

## A. 建议删除（高置信度死代码）
| 文件路径 | 理由 | confidence |
|---------|------|-----------|
| (待 CI 分析) | - | - |

## B. 建议移入 legacy/ 文件夹
| 文件路径 | 理由 | confidence |
|---------|------|-----------|
| (待 CI 分析) | - | - |

## C. 文件内部简化建议
| 文件路径 | 建议 | confidence |
|---------|------|-----------|
| (待 CI 分析) | - | - |

## D. 活跃文件摘要
| 文件路径 | 近期改动次数 | 当前状态 |
|---------|-----------|--------|
| (待 CI 分析) | - | - |

## 说明
- confidence: high → heartbeat/coder 可自主执行删除或移入 legacy/（执行前必须 grep 验证无 import）
- confidence: medium → 写入 PENDING_TASKS.md 为 [PENDING] 任务，auto_launch=false，等用户确认
- confidence: low → 仅供参考，不做任何操作
- **绝对规则：移动/删除前必须确认文件未被任何 import 或 __init__.py 引用**
"""
    pathlib.Path("CODE_CLEANUP_SUGGESTIONS.md").write_text(placeholder, encoding="utf-8")
    print("Written placeholder CODE_CLEANUP_SUGGESTIONS.md")


if __name__ == "__main__":
    main()
