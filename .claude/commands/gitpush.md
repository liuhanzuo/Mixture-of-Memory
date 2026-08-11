---
model: opus
---

# /gitpush — 安全 git push 到 GitHub（含 subagent 审核）

**用途**：将本地 commits 安全地推送到 GitHub，推送前必须派专属 subagent 审核代码变更。

---

## 必须遵守的执行顺序

### Step 1: Pre-push 安全检查

```bash
# 确认无敏感文件
git diff --name-only HEAD | grep -E "\.(pt|bin|pth|ckpt)$|password" && echo "❌ SENSITIVE FILES — ABORT" || echo "✅ OK"

# 查看待推送 commits
git log --oneline origin/main..HEAD 2>/dev/null || git log --oneline -5
```

### Step 2: 派 subagent 审核（必须，不可跳过）

**在推送前，必须派一个 general-purpose subagent 执行代码审核。** subagent 的 prompt 模板：

```
你是 Mixture-of-Memory 项目的代码审核员。
工作目录：/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/

请审核以下 commits 准备推送到 GitHub main 分支：
[插入 git log --oneline 输出]

请执行：
1. git diff origin/main..HEAD --stat（查看改动范围）
2. git diff origin/main..HEAD（查看具体改动，重点关注 src/ configs/ scripts/）
3. 检查项：
   - 有无 *.pt / *.bin / password.txt 等敏感文件被意外包含？
   - 有无破坏性改动（删除关键配置、重命名接口、破坏现有实验）？
   - 有无明显代码错误（语法错误、import 错误、明显逻辑 bug）？
   - commit message 是否清晰准确？
4. 输出审核结论：
   - ✅ APPROVED（无问题，可以推送）
   - ⚠️ APPROVED with notes（有小问题但不阻塞推送，列出建议）
   - ❌ REJECTED（有严重问题，列出必须修复的内容）
```

### Step 3: 根据审核结论决定行动

- **APPROVED / APPROVED with notes** → 继续 Step 4 推送
- **REJECTED** → 停止推送，将问题写入 PENDING_TASKS.md，等修复后再调用 /gitpush

### Step 4: 确定推送目标分支

- 有实验进展（ratio 改善、bug 修复完成、新功能可用）→ 推送到 `main`
- 无明显进展但代码值得保存 → 推送到 `archive/<exp_name>-$(date +%Y%m%d)`

### Step 5: 通过 star-proxy 推送

```bash
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
git push origin main   # 或 archive/... 分支
```

### Step 6: 如果推送因 PAT workflow 权限失败

若错误为 "refusing to allow a Personal Access Token to create or update workflow without `workflow` scope"：
- 记录失败到 UPDATELOG.md
- 写入 PENDING_TASKS.md 提示用户更新 PAT scope（Settings → Developer Settings → PAT → `workflow` scope）
- 不要重试

---

## 禁止规则
- 禁止跳过 subagent 审核直接推送
- 禁止 `git push --force` 或 `git push -f`
- 禁止推送包含 `*.pt`, `*.bin`, `configs/password.txt` 的 commit
- 禁止 `git push origin main` 在 REJECTED 审核结论后执行
