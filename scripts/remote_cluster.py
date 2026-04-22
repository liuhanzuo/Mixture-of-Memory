#!/usr/bin/env python3
"""Remote cluster helper - run commands on B200/L20A nodes."""
import pexpect, time, sys, json

HOSTS = ["28.89.17.143", "28.89.17.144", "28.89.17.85", "28.89.19.134"]
PWD = "4fS6h9nHdbICfm6,"
NAMES = ["b200-1", "b200-2", "b200-3", "b200-4"]

def ssh_run(host, cmd, timeout=120):
    """Run command on remote host, return stdout."""
    child = pexpect.spawn(
        f'ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password root@{host}',
        timeout=timeout, encoding='utf-8', codec_errors='replace'
    )
    child.expect('assword:')
    child.sendline(PWD)
    time.sleep(4)
    child.sendline(cmd)
    time.sleep(3)
    child.sendline('echo __DONE__')
    child.expect('__DONE__', timeout=timeout-10)
    output = child.before
    child.sendline('exit')
    child.close()
    return output

def ssh_run_bg(host, cmd, timeout=5):
    """Run command in background on remote host (non-blocking), return (child, pid)."""
    child = pexpect.spawn(
        f'ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password root@{host}',
        timeout=timeout, encoding='utf-8', codec_errors='replace'
    )
    child.expect('assword:')
    child.sendline(PWD)
    time.sleep(4)
    child.sendline(f'nohup {cmd} > /tmp/remote_cmd.log 2>&1 & echo __BG_PID__ $! __BG_PID__')
    time.sleep(2)
    child.expect('__BG_PID__', timeout=10)
    # Read the PID
    rest = child.before
    child.sendline('exit')
    child.close()
    return rest

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <node|all> <command>")
        print(f"  Nodes: {', '.join(NAMES)} or 'all'")
        print(f"  --bg: run in background (append before command)")
        sys.exit(1)
    
    target = sys.argv[1]
    cmd = ' '.join(sys.argv[2:])
    
    if target == 'all':
        for name, host in zip(NAMES, HOSTS):
            print(f"\n=== {name} ({host}) ===")
            try:
                out = ssh_run(host, cmd)
                print(out)
            except Exception as e:
                print(f"  FAILED: {e}")
    else:
        idx = NAMES.index(target) if target in NAMES else HOSTS.index(target)
        host = HOSTS[idx]
        print(f"=== {NAMES[idx]} ({host}) ===")
        out = ssh_run(host, cmd)
        print(out)
