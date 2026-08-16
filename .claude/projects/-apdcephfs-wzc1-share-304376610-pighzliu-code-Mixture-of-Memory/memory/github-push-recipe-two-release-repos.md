---
name: github-push-recipe-two-release-repos
description: "★GitHub push 唯一可行路径:每仓 .git/config 的 core.sshCommand = ssh -F /dev/null -i configs/github_deploy_key ... ProxyCommand configs/gh_proxy_connect.py -p 443;全局 ssh 默认 port 36000 会打死 github;直接 git push 别加 GIT_SSH_COMMAND"
metadata: 
  node_type: memory
  type: project
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**两个 release 仓推 GitHub 的唯一可行方式**（2026-08-05 实测踩坑后确认）：

每个仓在**自己的 `.git/config`** 里已配好 `core.sshCommand`，**直接 `git push origin main` 就行，不要在命令行覆盖 `GIT_SSH_COMMAND`**：

```
core.sshCommand = ssh -F /dev/null \
  -i <REPO_ROOT>/configs/github_deploy_key \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  -o ProxyCommand='python3 <REPO_ROOT>/configs/gh_proxy_connect.py %h %p' \
  -p 443
```
其中 `<REPO_ROOT>` = `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory`。

**关键事实（每条都踩过）：**
- 代理脚本在 **`configs/gh_proxy_connect.py`**，密钥是 **`configs/github_deploy_key`** —— **不是** `scripts/gh_proxy_connect.py`，**不是** `configs/comem_deploy_key`（后者虽存在但不是 push 用的那把）。
- **`-F /dev/null` 必须有**：全局/系统 ssh 默认把 **`port 36000`**（H20 集群用）套到所有 host，会让 `ssh.github.com` 连接超时（报 `connect to host ssh.github.com port 36000: Connection timed out`）。`-p 443` + `-F /dev/null` 才能绕开。
- **SSH 22/443 直连都被墙**；star-proxy 只走 HTTPS，而 HTTPS push 会被拒（`Invalid username or token`，`~/.git-credentials` 里那份对 GitHub 无效，全局 config 是面向内网 `git.woa.com` 的）→ **只有上面这个 ProxyCommand 路径能推**。
- 新克隆/新仓（如 `perplexity-heals-knowledge-lags` 当时就缺）**必须先 `git config --local core.sshCommand "..."`**，否则同样掉进 port 36000 坑。

**推之前**：按 CODEBUDDY.md 走 review-subagent（APPROVED/REJECTED）；author 必须 `LiuHanzuo <lhz24@mails.tsinghua.edu.cn>`；禁 AI trailer；禁 `git add -A/.`；禁 `--force`。
**COMem 若被 reject 为 non-fast-forward**：远端可能有 CI/他人 commit（如 `e272745` 改 eval `--n` 默认值 ruler 50/babilong 100）→ `git fetch` + `git merge origin/main`，**merge 后必须核对远端那侧的改动没被冲掉**，并在 merged tree 上重跑 `python -m comem.selftest` + 各 driver `--help`。

相关：[[cluster-two-disks-not-shared]]
