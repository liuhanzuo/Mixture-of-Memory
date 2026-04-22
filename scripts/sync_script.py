#!/usr/bin/env python3
"""SCP the updated train_sparse_memory.py to all remote nodes."""
import pexpect

PASSWORD = "4fS6h9nHdbICfm6,"
SSH_OPTS = "-o PubkeyAuthentication=no -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
LOCAL_SCRIPT = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/scripts/train_sparse_memory.py"
REMOTE_DIR = "/root/Mixture-of-Memory/scripts/"

NODES = ["28.89.17.144", "28.89.17.85", "28.89.19.134"]

for ip in NODES:
    print(f"\n=== Syncing to {ip} ===")
    scp_cmd = f"scp {SSH_OPTS} {LOCAL_SCRIPT} root@{ip}:{REMOTE_DIR}train_sparse_memory.py"
    child = pexpect.spawn(scp_cmd, timeout=30, encoding="utf-8")
    child.expect("[Pp]assword:", timeout=15)
    child.sendline(PASSWORD)
    child.expect(pexpect.EOF, timeout=30)
    print(f"  Exit: {child.exitstatus}, output: {child.before.strip()[-100:]}")
    child.close()
    
    # Verify
    ssh_cmd = f"ssh {SSH_OPTS} root@{ip}"
    child = pexpect.spawn(ssh_cmd, timeout=30, encoding="utf-8")
    child.expect("[Pp]assword:", timeout=15)
    child.sendline(PASSWORD)
    child.expect("#", timeout=15)
    child.sendline("grep -c 'use_l1\\|num_mem_tokens\\|recon_loss' /root/Mixture-of-Memory/scripts/train_sparse_memory.py")
    child.expect("#", timeout=15)
    print(f"  Verified matches: {child.before.strip()}")
    child.sendline("exit")
    child.close()

print("\nDone.")
