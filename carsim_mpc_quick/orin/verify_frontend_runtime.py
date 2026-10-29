"""Check frontend resource paths, lazy imports and JSON-line framing on loopback."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
from pathlib import Path

from runtime.paths import CARSIM_LIBRARY, ROOT, resolve_resource
from runtime.protocol import OrinConnection


def check_frame(payload, exception=None, fragmented=False):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    address = listener.getsockname()
    errors = []

    def peer():
        try:
            with listener, listener.accept()[0] as connection, connection.makefile("rb") as stream:
                request = json.loads(stream.readline())
                assert request == dict(seq=7, time=0.07, label="frame")
                if fragmented:
                    for offset in range(0, len(payload), 3):
                        connection.sendall(payload[offset : offset + 3])
                else:
                    connection.sendall(payload)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=peer, daemon=True)
    thread.start()
    client = OrinConnection(address[0], address[1], timeout=2)
    try:
        if exception is None:
            response, duration = client.exchange(dict(seq=7, time=0.07, label="frame"))
            assert response == dict(seq=7, forces=[0.0] * 8)
            assert duration >= 0
        else:
            try:
                client.exchange(dict(seq=7, time=0.07, label="frame"))
            except exception:
                pass
            else:
                raise AssertionError("Invalid protocol frame was accepted")
    finally:
        client.close()
        thread.join(3)
    assert client.incoming.closed and client.socket.fileno() == -1
    assert not thread.is_alive() and not errors, errors


def verify():
    assert ROOT == Path(__file__).resolve().parents[2]
    assert resolve_resource("results") == ROOT / "results"
    # Merely importing the infrastructure must not load the CARLA ABI or any DLL.
    from runtime.carsim import CarSim
    from runtime.sensors import image_at, pack_vision, spawn_camera

    assert all(callable(item) for item in [CarSim, image_at, pack_vision, spawn_camera])
    assert "carla" not in sys.modules
    windows_dll_rejected = False
    if os.name != "nt" and CARSIM_LIBRARY.suffix.lower() == ".dll":
        try:
            CarSim(ROOT / "results/unused_platform_guard")
        except RuntimeError as error:
            assert "Linux cannot load a Windows CarSim DLL" in str(error)
            windows_dll_rejected = True
        else:
            raise AssertionError("Linux attempted to load a Windows solver DLL")
    valid = (json.dumps(dict(seq=7, forces=[0.0] * 8)) + "\n").encode()
    check_frame(valid, fragmented=True)
    check_frame(valid[:-1], exception=RuntimeError)
    check_frame(b"{invalid}\n", exception=ValueError)
    check_frame(b" " * 32768 + b"{}\n", exception=RuntimeError)
    result = dict(
        passed=True,
        repository_root_correct=True,
        lazy_carla_import=True,
        resources_relative_to_repository=True,
        linux_windows_dll_guard=windows_dll_rejected if os.name != "nt" else "not_applicable",
        fragmented_reply_accepted=True,
        truncated_reply_rejected=True,
        invalid_json_rejected=True,
        oversized_reply_rejected=True,
        socket_and_file_wrapper_closed=True,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
