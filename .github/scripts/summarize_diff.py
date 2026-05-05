#!/usr/bin/env python3
"""
CI script: Generate a Chinese-language summary of the latest git commit's diff
and append it to UPDATELOG.md.

Called by .github/workflows/ci_code_summary.yml after every push to main.
Requires: ANTHROPIC_API_KEY environment variable.
"""

import subprocess
import datetime
import os
import sys

try:
    import anthropic
except ImportError:
    print("anthropic package not found, skipping summary generation")
    sys.exit(0)


def run(cmd):
    return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set, skipping")
        sys.exit(0)

    try:
        diff_stat = run(["git", "diff", "HEAD~1", "HEAD", "--stat", "--diff-filter=ACMRD"])
    except subprocess.CalledProcessError:
        diff_stat = "(initial commit or no prior commit)"

    diff_stat = diff_stat[:8000]  # cap to avoid token overflow

    commit_msg = run(["git", "log", "-1", "--format=%s"])
    commit_hash = run(["git", "rev-parse", "--short", "HEAD"])
    author = run(["git", "log", "-1", "--format=%an"])
    today = datetime.date.today().isoformat()

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": f"""你是 Mixture-of-Memory 项目的代码审查助手。
请用中文总结这次 git commit 的改动，输出格式如下（3-5行，精炼，不要输出其他内容）：

## [{today} {commit_hash}] {commit_msg}
- 主要改动：...（说明改了什么文件/功能）
- 影响模块：...（影响哪个 memory 模块或训练流程）
- 备注：...（有无特殊注意事项，如 breaking change）

git diff --stat:
{diff_stat}
"""
        }]
    )

    summary = resp.content[0].text.strip()

    updatelog_path = "UPDATELOG.md"
    with open(updatelog_path, "a", encoding="utf-8") as f:
        f.write(f"\n{summary}\n")

    print(f"Appended summary for {commit_hash} to {updatelog_path}")


if __name__ == "__main__":
    main()
