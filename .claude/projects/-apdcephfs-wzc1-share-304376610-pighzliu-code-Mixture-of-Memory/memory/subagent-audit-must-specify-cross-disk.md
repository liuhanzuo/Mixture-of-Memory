---
name: subagent-audit-must-specify-cross-disk
description: "派 provenance/文件搜索类 subagent 必须在 prompt 里显式声明 wzc1+zwfy6 两盘都要搜, 否则会漏 (2026-08-06 又中招)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: b405362c-2a9f-405a-978e-56afc9e4b104
---

**Rule**: 派 subagent 做「找 X 文件在磁盘的位置」/「audit provenance」/「locate script output」类任务时,
**prompt 必须显式列出 wzc1 + zwfy6 两个物理盘的搜索路径**, 不能只让它搜"当前工作目录"。

**Why**: 2026-08-06 04:00 派 Explore subagent 定位 paperA 3 组 primitive 数字的 disk 来源,
它只搜 wzc1 (工作目录默认盘), 报告 2/3 "no candidate found in {searched paths}".
MAIN 独立核实发现 P0.13/P0.17 实验跑在 .82 (zwfy6), 磁盘全部存在且值精确匹配 tex,
subagent 白报"未找到"。这是 [[cluster-two-disks-not-shared]] 坑的**第 N 次**复现,
但换个形式 (不是 shell 直接踩, 而是 subagent 的 mental model 不知两盘存在)。

**How to apply**: 派 audit/搜索类 subagent 时, prompt 里加一段:

```
## 搜索范围 (必读, 跨 2 盘)

当前工作目录 (wzc1 盘) 只覆盖 LOCAL + .21 两台机器。**如果找不到, 必须去 zwfy6 盘搜**:
- .82 (28.82.250.82:36000, 密码 configs/password_h20_82250.txt): /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/
- .73 (28.85.35.73:36000, 密码 configs/password_h20_853573.txt): 同上路径

远程 grep 用法:
  sshpass -f <密码文件> ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -p 36000 root@<IP> \
    "cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory && grep -rln '<pattern>' bench_results/"

任何 "no candidate found" 结论前, 必须报告"已在 wzc1 + zwfy6 两盘都搜过"。
```

**Cross-refs**: 与 [[cluster-two-disks-not-shared]] 是同一个物理事实的两个表现:
前者是"main agent 自己踩两盘"的教训, 本条是"派 subagent 时必须显式教它"的推论。
