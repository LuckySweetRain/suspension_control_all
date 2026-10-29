"""Windows-only CarSim DLL owner; no CARLA, camera, model or plotting imports."""

from __future__ import annotations

import ctypes as ct
import math
import os
import socket
import sys

from runtime.carsim import CarSim
from runtime.paths import ROOT, SETTINGS
from runtime.solver_rpc import SERVICE, read_frame, send_frame


class CallbackTerrain:
    """Cache 5 cm ground samples; request only missing heights from WSL."""

    def __init__(self, connection, incoming):
        self.connection, self.incoming = connection, incoming
        self.cache, self.errors = {}, []
        self.calls, self.request_id = 0, 0
        self.origin_calibrated = True
        self.origin_cs = (0.0, 0.0)
        pointer = ct.POINTER(ct.c_double)
        signature = ct.CFUNCTYPE(
            ct.c_int, ct.c_double, ct.c_double, ct.c_double, *([pointer] * 5), ct.c_void_p
        )
        self.callback = signature(self.query)

    def query(self, x, y, z, out_z, out_dx, out_dy, out_mu, out_rr, unused):
        self.calls += 1
        out_z[0], out_dx[0], out_dy[0], out_mu[0], out_rr[0] = 0, 0, 0, 0.85, 0.01
        try:
            keys = [
                (round(px / 0.05), round(py / 0.05))
                for px, py in [(x, y), (x + 0.1, y), (x - 0.1, y), (x, y + 0.1), (x, y - 0.1)]
            ]
            missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
            if missing:
                send_frame(
                    self.connection,
                    dict(
                        event="terrain_heights",
                        id=self.request_id,
                        points=[[kx * 0.05, ky * 0.05] for kx, ky in missing],
                    ),
                )
                answer = read_frame(self.incoming)
                if answer.get("event") != "terrain_heights" or answer.get("id") != self.request_id:
                    raise ValueError("Mismatched terrain callback")
                if "error" in answer:
                    raise RuntimeError(answer["error"])
                heights = answer["heights"]
                if len(heights) != len(missing) or not all(math.isfinite(h) for h in heights):
                    raise ValueError("Invalid terrain heights")
                self.cache.update(zip(missing, heights))
                self.request_id += 1
            values = [self.cache[key] for key in keys]
            out_z[0], out_dx[0], out_dy[0] = (
                values[0],
                (values[1] - values[2]) / 0.2,
                (values[3] - values[4]) / 0.2,
            )
            return 1
        except Exception as error:
            self.errors.append(str(error))
            return 0

    def install(self, dll):
        dll.vs_road_install_terrain_function.argtypes = [type(self.callback), ct.c_void_p]
        dll.vs_road_install_terrain_function(self.callback, None)

    def check(self):
        if self.errors:
            raise RuntimeError(self.errors[-1])


def serve_connection(connection):
    solver, terrain = None, None
    seq = 0
    with connection, connection.makefile("rb") as incoming:
        connection.settimeout(120)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while True:
                request = read_frame(incoming)
                if request.get("version") != 1 or request.get("seq") != seq:
                    raise ValueError("Invalid solver RPC sequence/version")
                operation = request["op"]
                reply = dict(
                    version=1,
                    seq=seq,
                    ok=True,
                    service=SERVICE,
                    dll_owner_pid=os.getpid(),
                    carla_client_loaded="carla" in sys.modules,
                )
                if operation in ("status", "shutdown"):
                    send_frame(connection, dict(reply, dll_owner_pid=os.getpid()))
                    return operation != "shutdown"
                if operation == "init":
                    if solver is not None:
                        raise ValueError("Only one DLL owner per session")
                    output = (ROOT / "results" / request["output"]).resolve()
                    output.relative_to((ROOT / "results").resolve())
                    output.mkdir(parents=True, exist_ok=True)
                    terrain = CallbackTerrain(connection, incoming) if request["terrain"] else None
                    solver = CarSim(
                        output,
                        terrain=terrain,
                        contact=bool(request["contact"]),
                        driving=bool(request["driving"]),
                        stop_time=request["stop_time"],
                    )
                    reply.update(
                        values=solver.initial, n_import=solver.n_import, n_export=solver.n_export
                    )
                elif operation == "step" and solver is not None:
                    reply["values"] = solver.step(
                        request["forces"], request["contact_force"], request["driving_inputs"]
                    )
                elif operation == "close":
                    send_frame(connection, reply)
                    # A scene terrain installs a process-global DLL callback.
                    # Exit after that run so no future initialization can reach
                    # a callback referring to a closed WSL socket.
                    return terrain is None
                else:
                    raise ValueError("Unknown solver operation or missing init")
                reply.update(time=solver.t, terrain_calls=terrain.calls if terrain else 0)
                send_frame(connection, reply)
                seq += 1
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
            try:
                send_frame(
                    connection,
                    dict(version=1, seq=seq, ok=False, service=SERVICE, error=str(error)),
                )
            except OSError:
                pass
            print("Solver session ended:", error, flush=True)
            # Initialization errors may leave process-global DLL state behind.
            return solver is not None and terrain is None
        finally:
            if solver is not None:
                solver.close()


def serve():
    if os.name != "nt":
        raise RuntimeError("Run the CarSim DLL bridge in Windows")
    host, port = SETTINGS["carsim_bridge_host"], int(SETTINGS["carsim_bridge_port"])
    if host != "127.0.0.1":
        raise ValueError("The local solver bridge must bind to loopback")
    with socket.socket() as listener:
        listener.bind((host, port))
        listener.listen(2)
        listener.settimeout(300)
        print("Windows DLL bridge ready", host, port, flush=True)
        while True:
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                return
            if not serve_connection(connection):
                return


if __name__ == "__main__":
    serve()
