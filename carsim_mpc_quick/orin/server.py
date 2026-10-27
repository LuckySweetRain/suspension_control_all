"""Single-client, bounded-frame TCP controller service; no GPU required."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time

from controller import Controller


def serve(host, port, data):
    controller = Controller(data)
    with open(data, "rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(1)
        print(
            json.dumps({"event": "ready", "host": host, "port": port, "sha256": digest}), flush=True
        )
        while True:
            connection, peer = listener.accept()
            controller.reset()
            last_seq, last_t = -1, None
            print(json.dumps({"event": "connect", "peer": peer}), flush=True)
            with connection:
                connection.settimeout(30)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    with connection.makefile("rb") as incoming:
                        while True:
                            line = incoming.readline(32769)
                            if not line:
                                break
                            if len(line) > 32768 or not line.endswith(b"\n"):
                                raise ValueError("Oversized or incomplete frame")
                            request = json.loads(line)
                            if request.get("version") != 1 or request.get("seq") != last_seq + 1:
                                raise ValueError("Protocol version/sequence mismatch")
                            import math

                            t = float(request["time"])
                            if not math.isfinite(t):
                                raise ValueError("Nonfinite simulation time")
                            if (
                                last_t is not None
                                and abs(t - last_t - controller.c["control_dt"]) > 1e-7
                            ):
                                raise ValueError("Control sample time mismatch")
                            start = time.perf_counter()
                            forces, diagnostics, fallback = controller.step(request["exports"])
                            response = {
                                "version": 1,
                                "seq": request["seq"],
                                "forces": forces,
                                "diagnostics": diagnostics,
                                "fallback": fallback,
                                "compute_ms": 1000 * (time.perf_counter() - start),
                                "sha256": digest,
                            }
                            connection.sendall(
                                (json.dumps(response, allow_nan=False) + "\n").encode()
                            )
                            last_seq, last_t = request["seq"], t
                except (ValueError, KeyError, TypeError, OSError) as error:
                    print(json.dumps({"event": "disconnect", "reason": str(error)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.50.2")
    parser.add_argument("--port", type=int, default=55001)
    parser.add_argument("--data", default="quick_mpc_data.mat")
    args = parser.parse_args()
    serve(args.host, args.port, args.data)
