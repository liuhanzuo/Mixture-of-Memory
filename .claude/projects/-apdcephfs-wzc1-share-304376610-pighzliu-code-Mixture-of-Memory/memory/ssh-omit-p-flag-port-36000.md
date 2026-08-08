---
name: ssh-omit-p-flag-port-36000
description: 集群 ssh 一律省略 -p（全局 ssh_config 已设 Port 36000）；写 -p 22 会 Permission denied 且极像密码过期
metadata:
  type: reference
---

本机 `/etc/ssh/ssh_config` 有全局 `Host * / Port 36000`，所以**所有节点 ssh 一律省略 `-p`**。

**坑**：某些节点的 22 端口上另有一个 sshd，host key 与 36000 相同但账户不同 →
写 `-p 22` 会得到 `Permission denied`，**表现得极像密码过期或凭据失效**。

**判定口径**：`ssh -G <host> | grep '^port'` 看真实生效端口。

**史实**（2026-08-06，节点 `.252`=`28.89.19.252`，**该节点已退役，被 `.21`=`28.89.19.21` 替代**）：
我因为命令里带了 `-p 22` 而误判「B200 密码轮换 / 凭据失效」，白等 13 小时。
根因是端口不是凭据。**"auth 被拒" ≠ "凭据失效"，先查端口。**

当前 B200 节点是 `.21` = `28.89.19.21`，密码 `configs/password_b200_19021.txt`
（8× L20A cc10.0 183GB，driver 580.105.08，真 wzc1 盘，conda torch 2.13.0）。
见 [[cluster-two-disks-not-shared]]。
