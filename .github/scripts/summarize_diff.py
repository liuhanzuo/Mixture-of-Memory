#!/usr/bin/env python3
"""
CI script: Generate a Chinese-language summary of the latest git commit's diff
and append it to UPDATELOG.md.

Called by .github/workflows/ci_code_summary.yml after every push to main.
Requires: ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) environment variable.

Supports Tencent Claude gateway via ANTHROPIC_BASE_URL env var.
GitHub Secrets to configure:
  - ANTHROPIC_API_KEY: Tencent token (e.g. c92b...8cB)
  - ANTHROPIC_BASE_URL: https://copilot.code.woa.com/server/chat/codebuddy-gateway/codebuddy-code
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


def build_client():
    """Build Anthropic client, supporting Tencent gateway via ANTHROPIC_BASE_URL."""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        return None
    base_url = os.environ.get("ANTHROPIC_BASE_URL")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
        # Tencent gateway requires x-api-key custom header
        kwargs["default_headers"] = {"x-api-key": api_key}
    return anthropic.Anthropic(**kwargs)


def get_model(tier="haiku"):
    """Get model name, respecting Tencent gateway aliases."""
    if tier == "haiku":
        return os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", "claude-3-5-haiku-20241022")
    return os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-3-5-sonnet-20241022")


def main():
    client = build_client()
    if not client:
        print("No API key (ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN) set, skipping")
        sys.exit(0)

    try:
        diff_stat = run(["git", "diff", "HEAD~1", "HEAD", "--stat", "--diff-filter=ACMRD"])
    except subprocess.CalledProcessError:
        diff_stat = "(initial commit or no prior commit)"

    diff_stat = diff_stat[:8000]  # cap to avoid token overflow

    commit_msg = run(["git", "log", "-1", "--format=%s"])
    commit_hash = run(["git", "rev-parse", "--short", "HEAD"])
    today = datetime.date.today().isoformat()

    resp = client.messages.create(
        model=get_model("haiku"),
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
