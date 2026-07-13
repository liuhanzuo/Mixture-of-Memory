#!/usr/bin/env python3
"""SSH ProxyCommand: tunnel to <host> <port> through the star HTTP proxy via CONNECT.
Usage (as ssh ProxyCommand): gh_proxy_connect.py %h %p
"""
import sys, socket, select, os
PROXY = ("star-proxy.oa.com", 3128)
th, tp = sys.argv[1], int(sys.argv[2])
s = socket.create_connection(PROXY, timeout=15)
s.sendall(("CONNECT %s:%d HTTP/1.0\r\nHost: %s:%d\r\n\r\n" % (th, tp, th, tp)).encode())
buf = b""
while b"\r\n\r\n" not in buf:
    ch = s.recv(1)
    if not ch:
        sys.stderr.write("proxy closed during CONNECT\n"); sys.exit(1)
    buf += ch
if b" 200 " not in buf.split(b"\r\n", 1)[0]:
    sys.stderr.write("proxy CONNECT failed: " + buf.split(b"\r\n")[0].decode("latin1") + "\n"); sys.exit(1)
# keep the socket BLOCKING: select tells us a fd is readable; recv/read won't block,
# and the blocking sendall/os.write to the peer drains fine (ssh + github both read).
# (A non-blocking socket makes sendall raise BlockingIOError once the send buffer fills
#  during a large push -> broken pipe. Blocking + select is the correct relay pattern.)
fin, fout = sys.stdin.fileno(), sys.stdout.fileno()
while True:
    r, _, _ = select.select([fin, s], [], [])
    if fin in r:
        d = os.read(fin, 65536)
        if not d: break
        s.sendall(d)
    if s in r:
        d = s.recv(65536)
        if not d: break
        os.write(fout, d)
