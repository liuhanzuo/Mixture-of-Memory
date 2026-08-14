#!/usr/bin/env python3
"""sshpass 的 pexpect 回退：`sshpass` 不在时用这个连远程节点。

为什么存在
----------
2026-08-13 22:2x 的节点重启把 `sshpass` 从 /usr/bin 和 /opt/conda/bin **全部抹掉**，
而项目里几乎每个 launch / eval 脚本都靠它连远程。当时唯一能用的是 `pexpect`（4.9.0 尚在），
于是临时写了这个。后来 `yum install -y sshpass` 装回来了，所以这是**回退**不是主路径。

用法
----
    python tools/ssh_pexpect_fallback.py <IP> <密码文件> '<远程命令>' [超时秒]

例：
    python tools/ssh_pexpect_fallback.py 28.85.35.73 configs/password_h20_853573.txt 'nvidia-smi -L' 60

注意
----
* **一律省略 `-p`**：全局 ssh_config 已设 `Port 36000`；写 `-p 22` 会 Permission denied。
* 密码文件只 `rstrip("\n")`，**不要** `tr -d` —— 末尾逗号是密码的一部分。
* 先试 `which sshpass`；有就用标准配方，别用这个（这个每次连接都要起一个 pexpect 子进程，慢且不便于并发）。

放在 wzc1 而不是 /tmp
--------------------
2026-08-14 用户指令：需要保留的东西放 wzc1 或 diskB。/tmp 已被重启证明会清空 ——
这个文件的全部意义就是「重启之后还能连上机器」，放在 /tmp 等于它在最需要它的时刻消失。
"""
import sys, pexpect
host, pwfile, cmd = sys.argv[1], sys.argv[2], sys.argv[3]
pw = open(pwfile).read().rstrip("\n")
c = pexpect.spawn("/usr/bin/ssh", ["-o","StrictHostKeyChecking=no","-o","ConnectTimeout=15",
     "-o","PreferredAuthentications=password","-o","PubkeyAuthentication=no",
     "root@"+host, cmd], timeout=int(sys.argv[4]) if len(sys.argv)>4 else 90, encoding="utf-8")
i = c.expect([r"[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT])
if i==0:
    c.sendline(pw); c.expect([pexpect.EOF, pexpect.TIMEOUT])
print(c.before or "", flush=True)
c.close()
print("EXITSTATUS=%s" % c.exitstatus)
