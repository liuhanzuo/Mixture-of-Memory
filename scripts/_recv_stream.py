#!/usr/bin/env python3
"""Listen on a TCP port, copy the received byte stream to stdout. Used to pull a
tar stream over the fast internal network (bypasses the slow public gateway)."""
import socket, sys
port = int(sys.argv[1])
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", port))
s.listen(1)
s.settimeout(180)
conn, addr = s.accept()
sys.stderr.write("recv: connection from %s\n" % (addr,))
sys.stderr.flush()
out = sys.stdout.buffer
total = 0
while True:
    chunk = conn.recv(4 << 20)
    if not chunk:
        break
    out.write(chunk)
    total += len(chunk)
out.flush()
conn.close()
sys.stderr.write("recv: done, %d bytes\n" % total)
