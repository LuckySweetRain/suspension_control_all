"""Bounded local RPC for a WSL coordinator and a Windows VehicleSim process.

Terrain requests are handled while waiting for an integration reply, avoiding
the deadlock of a synchronous solver calling back into a blocked coordinator.
"""

from __future__ import annotations

import json
import math
import socket
import warnings
from pathlib import Path

from .paths import ROOT, SETTINGS

SERVICE = "carsim-windows-rpc"
MAX_BYTES = 128 * 1024


def send_frame(connection, frame):
    raw = (json.dumps(frame, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("Solver RPC frame too large")
    connection.sendall(raw)


def read_frame(incoming):
    raw = incoming.readline(MAX_BYTES + 1)
    if not raw or len(raw) > MAX_BYTES or not raw.endswith(b"\n"):
        raise RuntimeError("Disconnected or invalid solver RPC frame")
    frame = json.loads(raw)
    if not isinstance(frame, dict):
        raise ValueError("Solver RPC requires a JSON object")
    return frame


def connect_bridge(timeout=60):
    connection = socket.create_connection(
        (SETTINGS["carsim_bridge_host"], int(SETTINGS["carsim_bridge_port"])), timeout=timeout
    )
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return connection


def bridge_control(operation, timeout=2):
    with connect_bridge(timeout) as connection, connection.makefile("rb") as incoming:
        send_frame(connection, dict(version=1, seq=0, op=operation))
        response = read_frame(incoming)
        if response.get("service") != SERVICE or not response.get("ok"):
            raise RuntimeError("Unexpected CarSim bridge endpoint: " + str(response))
        return response


class RemoteCarSim:
    """The same initial/step/t/close interface, with DLL ownership on Windows."""

    def __init__(self, output, terrain=None, contact=False, driving=False, stop_time=None):
        relative = Path(output).resolve().relative_to(ROOT / "results")
        if terrain is not None and not terrain.origin_calibrated:
            raise ValueError("Calibrate the CarSim origin before remote terrain initialization")
        self.terrain = terrain
        self.connection = connect_bridge()
        self.incoming = self.connection.makefile("rb")
        self.seq = 0
        self.closed = False
        self.initial = []
        self.contact, self.driving = contact, driving
        try:
            reply = self.exchange(
                "init",
                output=relative.as_posix(),
                terrain=terrain is not None,
                contact=contact,
                driving=driving,
                stop_time=stop_time,
            )
            self.n_import, self.n_export = reply["n_import"], reply["n_export"]
            self.owner_pid = reply["dll_owner_pid"]
            self.server_carla_client_loaded = reply["carla_client_loaded"]
            self.initial = reply["values"]
            self.validate_state(self.initial)
        except Exception:
            self.close_socket()
            raise

    def terrain_request(self, request):
        if self.terrain is None:
            raise ValueError("Terrain callback without a terrain provider")
        points = request["points"]
        if not isinstance(points, list) or not 1 <= len(points) <= 5:
            raise ValueError("Invalid terrain batch size")
        heights = []
        for point in points:
            if len(point) != 2 or not all(math.isfinite(v) for v in point):
                raise ValueError("Invalid terrain coordinates")
            heights.append(self.terrain.height(*point))
        return dict(event="terrain_heights", id=request["id"], heights=heights)

    def exchange(self, operation, **payload):
        send_frame(self.connection, dict(version=1, seq=self.seq, op=operation, **payload))
        while True:
            reply = read_frame(self.incoming)
            if reply.get("event") == "terrain_heights":
                try:
                    answer = self.terrain_request(reply)
                except Exception as error:
                    answer = dict(event="terrain_heights", id=reply.get("id"), error=str(error))
                send_frame(self.connection, answer)
                continue
            if reply.get("version") != 1 or reply.get("seq") != self.seq:
                raise RuntimeError("Solver RPC clock/sequence mismatch")
            if not reply.get("ok"):
                raise RuntimeError("Windows CarSim: " + reply.get("error", "unknown failure"))
            self.seq += 1
            if "time" in reply:
                self.t = reply["time"]
            if self.terrain is not None:
                self.terrain.calls = reply.get("terrain_calls", self.terrain.calls)
            return reply

    def validate_state(self, values):
        if len(values) != self.n_export or not all(math.isfinite(v) for v in values):
            raise RuntimeError("Invalid Windows solver state vector")

    def step(self, forces, contact_force=0, driving_inputs=None):
        previous_time = self.t
        reply = self.exchange(
            "step", forces=list(forces), contact_force=contact_force, driving_inputs=driving_inputs
        )
        self.values = reply["values"]
        if abs(self.t - previous_time - 0.01) > 1e-7:
            raise RuntimeError("Windows solver did not advance exactly 10 ms")
        self.validate_state(self.values)
        return self.values

    def close_socket(self):
        if not self.closed:
            self.closed = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.incoming.close()
            self.connection.close()

    def close(self):
        if not self.closed:
            try:
                self.exchange("close")
            except (OSError, RuntimeError, ValueError) as error:
                warnings.warn("Closing disconnected Windows solver: " + str(error), stacklevel=2)
            finally:
                self.close_socket()
