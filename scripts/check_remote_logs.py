#!/usr/bin/env python3
"""Check training logs on remote nodes."""
import pexpect

PASSWORD = "4fS6h9nHdbICfm6,"
SSH_OPTS = "-o PubkeyAuthentication=no -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"

NODES = {
    "28.89.17.144": ("b200-2", "outputs/sparse_l01_seg512_mem32"),
    "28.89.17.85": ("b200-3", "outputs/sparse_l01_mem128_recon05"),
    "28.89.19.134": ("b200-4", "outputs/sparse_l01_mem64_bptt4"),
}

for ip, (name, odir) in NODES.items():
    print(f"\n=== {ip} ({name}) ===")
    child = pexpect.spawn(f"ssh {SSH_OPTS} root@{ip}", timeout=30, encoding="utf-8")
    child.expect("[Pp]assword:", timeout=15)
    child.sendline(PASSWORD)
    child.expect("#", timeout=15)
    child.sendline(f"tail -80 /root/Mixture-of-Memory/{odir}/train.log 2>/dev/null || echo NO_LOG")
    child.expect("#", timeout=15)
    print(child.before)
    child.sendline("exit")
    child.close()
