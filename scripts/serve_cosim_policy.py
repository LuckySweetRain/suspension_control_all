"""Orin CPU policy inference endpoint with explicit schema/hash and bounded messages.

Receives already-normalized observations. Actuator projection stays in the
shared experiment adapter; this server never controls hardware directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--bind", default="192.168.50.2")
    parser.add_argument("--port", type=int, default=55111)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    path = args.bundle / "policy.ts"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["policy_sha256"] or manifest["schema"] != "cosim-ppo-v1":
        raise ValueError("Policy identity/schema mismatch")
    model = torch.jit.load(str(path), map_location="cpu").eval()
    with torch.inference_mode():
        for _ in range(30):
            model(torch.zeros((1, 104)))
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((args.bind, args.port))
            listener.listen(1)
            print(json.dumps(dict(ready=True, policy_sha256=digest, port=args.port)), flush=True)
            while True:
                connection, _ = listener.accept()
                with connection, connection.makefile("rb") as incoming:
                    connection.settimeout(300)
                    previous = -1
                    try:
                        while True:
                            line = incoming.readline(16385)
                            if not line:
                                break
                            if len(line) > 16384 or not line.endswith(b"\n"):
                                raise ValueError("Invalid inference message")
                            request = json.loads(line)
                            if request.get("op") == "status":
                                reply = dict(
                                    ready=True, schema="cosim-ppo-v1", policy_sha256=digest
                                )
                            else:
                                seq = request["seq"]
                                if (
                                    request.get("version") != 1
                                    or not isinstance(seq, int)
                                    or seq != previous + 1
                                ):
                                    raise ValueError("Inference sequence mismatch")
                                if request.get("policy_sha256") != digest:
                                    raise ValueError("Requested policy differs from loaded weights")
                                obs = np.asarray(request["observation"], dtype=np.float32)
                                if obs.shape != (104,) or not np.isfinite(obs).all():
                                    raise ValueError("Invalid normalized observation")
                                start = time.perf_counter_ns()
                                output = model(torch.from_numpy(obs[None])).numpy()[0]
                                elapsed = (time.perf_counter_ns() - start) / 1e6
                                if not np.isfinite(output).all():
                                    raise ValueError("Nonfinite policy output")
                                reply = dict(
                                    seq=seq,
                                    raw_axis_N=output.tolist(),
                                    compute_ms=elapsed,
                                    policy_sha256=digest,
                                    device="cpu",
                                )
                                previous = seq
                            connection.sendall((json.dumps(reply, allow_nan=False) + "\n").encode())
                    except (ValueError, KeyError, OSError, RuntimeError) as error:
                        print(json.dumps(dict(error=str(error), last_seq=previous)), flush=True)


if __name__ == "__main__":
    main()
