"""Shared JSON-line transport; wire schemas and controller checks remain at callers."""

from __future__ import annotations

import json
import socket
import time


class OrinConnection:
    """Own both socket and its file wrapper so completed GUIs release the stream."""

    def __init__(self, host, port, timeout=30, connect_timeout=10):
        self.socket = socket.create_connection((host, port), timeout=connect_timeout)
        self.socket.settimeout(timeout)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.incoming = self.socket.makefile("rb")

    def exchange(self, request):
        begin = time.perf_counter()
        self.socket.sendall((json.dumps(request, allow_nan=False) + "\n").encode())
        line = self.incoming.readline(32769)
        elapsed_ms = (time.perf_counter() - begin) * 1000
        if len(line) > 32768 or not line.endswith(b"\n"):
            raise RuntimeError("Invalid or disconnected Orin JSON-line stream")
        return json.loads(line), elapsed_ms

    def close(self):
        try:
            self.incoming.close()
        finally:
            self.socket.close()
