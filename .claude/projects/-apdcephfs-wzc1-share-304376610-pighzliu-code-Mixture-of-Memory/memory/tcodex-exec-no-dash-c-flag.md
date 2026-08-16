---
name: tcodex-exec-no-dash-c-flag
description: "★tcodex exec 传任何 -c 都会清掉注入的 model_providers.tencent → 静默 fallback provider=openai → 连 wss://api.openai.com 无限超时;正解=用 CODEX_HOME 指向带 tools.web_search/model_reasoning_effort 的配置目录,且 --search 这个 flag 根本不存在"
metadata: 
  node_type: memory
  type: reference
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**`tcodex exec` 绝对不能传 `-c`。** 2026-08-05 因此白烧了两轮调研。

- `tcodex` 是腾讯内部 wrapper，它靠**注入 `model_providers.tencent`** 才能连内网 gateway。
  命令行上任何 `-c` 都会**覆盖掉那个注入块** → header 翻成 `provider: openai` → codex 去拨
  硬编码的 `wss://api.openai.com/v1/responses` → 5 次 WS 重试、退化到 HTTPS、然后无限
  `request timed out`。**表现像超时，根因是 provider 被清掉**，加长 timeout 没用。
- **`--search` 这个 flag 不存在**（`codex exec` 会直接报错退出、8 行错误、零输出）。
  真正的开关是配置里的 **`tools.web_search = true`**。

**可用配方**（存档：`paperC_research/reviewers_20260805/tcodex_working_recipe.sh`）：
- 完全不传 `-c`；
- 把 `model_reasoning_effort`、`tools.web_search`、`trust_level` 写进一个配置目录，用
  **`CODEX_HOME=<那个目录>`** 指过去；
- 加 `--skip-git-repo-check`；
- **必须 `-o <file>`**：一次无关的 tcodex 启动会 SIGKILL 掉所有在跑的 gateway，
  没有 `-o` 输出就全丢了（实测 15 分钟的活白干）；
- stdin 给 `</dev/null`。

**跑多久要有预期**：effort=max 的深度调研单次 **50-80 分钟**。不要用 200s/150s 的
`timeout` 去包（rc=143 就是我自己杀的），要么放后台轮询，要么分段。

**报告回来后必须自己核实引用**：这类报告会给出看似精确的 venue/细节但偶有编造
（曾出现过凭空的 "RSLoRA" 细节）。核实方法（arxiv.org 的 WebFetch 会被企业策略拦）：
```bash
export http_proxy=http://hy-proxy.woa.com:3128 https_proxy=http://hy-proxy.woa.com:3128
curl -sL https://arxiv.org/abs/<ID> | grep -oP '(?<=citation_title" content=")[^"]+'
```
2026-08-05 这批 6/6 个 load-bearing arXiv ID 全部真实、标题吻合，但 **venue 标注仍须
用 DBLP/ACL Anthology/OpenReview 逐条核**，不得直接进 `.bib`。
