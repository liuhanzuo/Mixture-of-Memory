#!/usr/bin/env python3
"""Read stdin bytes, send them to <host> <port> over TCP. Feed a tar stream in.
Used to push a tar stream over the fast internal network."""
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(120)
s.connect((host, port))
inp = sys.stdin.buffer
total = 0
while True:
    chunk = inp.read(4 << 20)
    if not chunk:
        break
    s.sendall(chunk)
    total += len(chunk)
s.shutdown(socket.SHUT_WR)
s.close()
sys.stderr.write("send: done, %d bytes\n" % total)
