#!/usr/bin/env python3
"""SCP the model.py to all remote nodes."""
import pexpect

PASSWORD = "4fS6h9nHdbICfm6,"
SSH_OPTS = "-o PubkeyAuthentication=no -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
LOCAL = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/src/memory/sparse_memory/model.py"
REMOTE = "/root/Mixture-of-Memory/src/memory/sparse_memory/model.py"

for ip in ["28.89.17.144", "28.89.17.85", "28.89.19.134"]:
    print(f"=== {ip} ===")
    child = pexpect.spawn(f"scp {SSH_OPTS} {LOCAL} root@{ip}:{REMOTE}", timeout=30, encoding="utf-8")
    child.expect("[Pp]assword:", timeout=15)
    child.sendline(PASSWORD)
    child.expect(pexpect.EOF, timeout=30)
    print(f"  {child.before.strip()[-100:]}")
    child.close()
    
    # Verify
    child = pexpect.spawn(f"ssh {SSH_OPTS} root@{ip}", timeout=30, encoding="utf-8")
    child.expect("[Pp]assword:", timeout=15)
    child.sendline(PASSWORD)
    child.expect("#", timeout=15)
    child.sendline("grep -c 'use_l1' /root/Mixture-of-Memory/src/memory/sparse_memory/model.py")
    child.expect("#", timeout=15)
    print(f"  use_l1 count: {child.before.strip()}")
    child.sendline("exit")
    child.close()

print("Done.")
