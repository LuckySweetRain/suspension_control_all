"""Bounded CARLA 0.9.11 / CarSim 2020 / Orin integration checks.

Run with conda run -n carla0911 python ... (legacy CARLA ABI requires Python 3.7).
The triple mode owns CarSim in Python; CARLA vehicle physics is disabled.
It demonstrates pose/camera synchronization, NOT CARLA terrain-driven tyre contact
or image-derived road preview. Road inputs remain the validated CarSim dataset.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import math
import os
import queue
import re
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CARLA_ROOT = Path("E:/Desktop/workspace/CARLA_0.9.11/WindowsNoEditor")
DATABASE = Path("E:/Desktop/workspace/CarSim_database/carsim_active_suspension")


class CarSim:
    """One DLL owner; keep original 8 imports / 77 exports and append 4 pose exports."""

    def __init__(self, output):
        text = (ROOT / "results/orin/runtime_run.par").read_text(encoding="utf-8")
        text = re.sub(r"(?m)^PORTS_EXP\s+1,77\s*$", "PORTS_EXP 1,81", text)
        text = re.sub(
            r"(?m)^END\s*$", "EXPORT Xcg_TM\nEXPORT Ycg_TM\nEXPORT Yaw\nEXPORT Roll\nEND", text
        )
        par = output / "runtime.par"
        par.write_text(text, encoding="utf-8")
        sim = output / "runtime.sim"
        prefix = output / "LastRun"
        dll = "E:/Carsim2020.0_Prog/Programs/solvers/carsim_64.dll"
        sim.write_text(
            f"SIMFILE\nINPUT {par}\nECHO {prefix}_echo.par\nLOGFILE {prefix}_log.txt\n"
            f"FINAL {prefix}_end.par\nERDFILE {prefix}.vs\nPROGDIR E:/Carsim2020.0_Prog/\n"
            f"DATADIR {DATABASE}/\nVEHICLE_CODE i_i\nEXT_MODEL_STEP 0.001\n"
            f"PORTS_IMP 1,8\nPORTS_EXP 1,81\nDLLFILE {dll}\nEND\n",
            encoding="utf-8",
        )
        self.dll = ct.CDLL(dll)
        self.dll.vs_set_opt_error_dialog(ct.c_int(0))
        self.dll.vs_get_error_message.restype = ct.c_char_p
        self.dll.vs_read_configuration.argtypes = [
            ct.c_char_p,
            ct.POINTER(ct.c_int32),
            ct.POINTER(ct.c_int64),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]
        self.dll.vs_integrate_io.argtypes = [
            ct.c_double,
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
        ]
        self.dll.vs_copy_export_vars.argtypes = [ct.POINTER(ct.c_double)]
        self.dll.vs_terminate_run.argtypes = [ct.c_double]
        ni, ne = ct.c_int32(), ct.c_int64()
        start, stop, dt = ct.c_double(), ct.c_double(), ct.c_double()
        old = os.getcwd()
        try:
            os.chdir(str(DATABASE))
            self.dll.vs_read_configuration(
                str(sim).encode(),
                ct.byref(ni),
                ct.byref(ne),
                ct.byref(start),
                ct.byref(stop),
                ct.byref(dt),
            )
        finally:
            os.chdir(old)
        self.t = start.value
        self.check()
        if (ni.value, ne.value) != (8, 81) or abs(dt.value - 0.001) > 1e-10:
            self.close()
            raise RuntimeError(f"Unexpected CarSim I/O or step: {(ni.value, ne.value, dt.value)}")
        self.values = (ct.c_double * 81)()
        self.dll.vs_copy_export_vars(self.values)
        self.t = start.value
        self.initial = list(self.values)

    def check(self):
        if self.dll.vs_error_occurred():
            raise RuntimeError(self.dll.vs_get_error_message().decode(errors="replace"))

    def step(self, forces):
        inputs = (ct.c_double * 8)(*forces)
        for _ in range(10):
            self.t += 0.001
            status = self.dll.vs_integrate_io(self.t, inputs, self.values)
            self.check()
            if status:
                raise RuntimeError(f"CarSim integration stopped: {status} at {self.t}")
        if not all(math.isfinite(v) for v in self.values):
            raise RuntimeError("Nonfinite CarSim state")
        return list(self.values)

    def close(self):
        self.dll.vs_terminate_run(self.t)


def image_at(images, frame):
    deadline = time.monotonic() + 15
    while True:
        image = images.get(timeout=max(0.01, deadline - time.monotonic()))
        if image.frame == frame:
            return image
        if image.frame > frame or time.monotonic() > deadline:
            raise RuntimeError("Camera/frame synchronization failed")


def run(args):
    egg = CARLA_ROOT / "PythonAPI/carla/dist/carla-0.9.11-py3.7-win-amd64.egg"
    sys.path.insert(0, str(egg))
    import carla

    output = ROOT / "results/carla_stack" / (args.mode + "_" + time.strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30)
    world = client.get_world()
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    actors, solver, connection = [], None, None
    viewer = None
    trace = []
    report = {
        "mode": args.mode,
        "client_version": client.get_client_version(),
        "server_version": client.get_server_version(),
        "map": world.get_map().name,
        "passed": False,
    }
    try:
        if report["client_version"] != report["server_version"]:
            raise RuntimeError("CARLA client/server version mismatch")
        if list(world.get_actors().filter("vehicle.*")):
            raise RuntimeError("World already has vehicles; do not alter another simulation")
        world.apply_settings(settings)
        bp = world.get_blueprint_library().find("vehicle.tesla.model3")
        spawn = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(bp, spawn)
        actors.append(vehicle)
        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "640")
        camera_bp.set_attribute("image_size_y", "360")
        camera_bp.set_attribute("sensor_tick", "0.0")
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-12)),
            attach_to=vehicle,
        )
        actors.append(camera)
        images = queue.Queue()
        camera.listen(images.put)
        first = image_at(images, world.tick())
        first.save_to_disk(str(output / "first.png"))
        if args.visualize:
            from live_view import LiveView

            viewer = LiveView()
            viewer.show(first)
        start_position = vehicle.get_location()
        ticks, control_steps, max_rtt, max_violation = 0, 0, 0.0, 0.0
        report["carsim_python_methods"] = [s for s in dir(vehicle) if "carsim" in s.lower()]
        if args.mode == "plugin":
            source = (
                CARLA_ROOT
                / "Engine/Plugins/Marketplace/CarSim/ThirdParty/VehicleSim/Data/Runs/ue4simfile.sim"
            )
            text = source.read_text()
            text = re.sub(
                r"(?m)^SET_MACRO \$\(OUTPUT_FILE_PREFIX\)\$.*$",
                "SET_MACRO $(OUTPUT_FILE_PREFIX)$ " + str(output / "PluginRun").replace("\\", "/"),
                text,
            )
            plugin_sim = output / "plugin.sim"
            plugin_sim.write_text(text)
            vehicle.enable_carsim(str(plugin_sim))
            vehicle.use_carsim_road(False)
        if args.mode == "triple":
            vehicle.set_simulate_physics(False)
            solver = CarSim(output)
            connection = socket.create_connection((args.orin_host, 55001), timeout=5)
            connection.settimeout(2)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            incoming = connection.makefile("rb")
            digest = hashlib.sha256(
                (ROOT / "results/orin/quick_mpc_data.mat").read_bytes()
            ).hexdigest()
            values = solver.initial
            for seq in range(round(args.seconds / 0.01)):
                request = {"version": 1, "seq": seq, "time": seq * 0.01, "exports": values[:77]}
                begin = time.perf_counter()
                connection.sendall((json.dumps(request, allow_nan=False) + "\n").encode())
                line = incoming.readline(32769)
                if len(line) > 32768 or not line.endswith(b"\n"):
                    raise RuntimeError("Invalid Orin frame")
                reply = json.loads(line)
                elapsed = (time.perf_counter() - begin) * 1000
                max_rtt = max(max_rtt, elapsed)
                if reply["version"] != 1 or reply["seq"] != seq or reply["sha256"] != digest:
                    raise RuntimeError("Orin sequence/data mismatch")
                forces = reply["forces"]
                if (
                    reply["fallback"]
                    or len(forces) != 8
                    or not all(math.isfinite(f) and abs(f) <= 5000 for f in forces)
                ):
                    raise RuntimeError("Orin control entered fallback or returned invalid forces")
                max_violation = max(max_violation, reply["diagnostics"][3])
                values = solver.step(forces)
                trace.append(
                    {
                        "time": solver.t,
                        "exports": values,
                        "forces": forces,
                        "diagnostics": reply["diagnostics"],
                        "rtt_ms": elapsed,
                    }
                )
                control_steps += 1
                if (seq + 1) % 5 == 0:
                    # Explicit world-frame convention: reflect CarSim Y/yaw, align initial pose.
                    yaw0 = math.radians(spawn.rotation.yaw)
                    dx, dy = values[77] - solver.initial[77], -(values[78] - solver.initial[78])
                    location = carla.Location(
                        x=spawn.location.x + math.cos(yaw0) * dx - math.sin(yaw0) * dy,
                        y=spawn.location.y + math.sin(yaw0) * dx + math.cos(yaw0) * dy,
                        z=spawn.location.z + values[38] - solver.initial[38],
                    )
                    rotation = carla.Rotation(
                        pitch=-values[41],
                        yaw=spawn.rotation.yaw - values[79] + solver.initial[79],
                        roll=-values[80],
                    )
                    vehicle.set_transform(carla.Transform(location, rotation))
                    last = image_at(images, world.tick())
                    ticks += 1
                    if viewer is not None:
                        viewer.show(
                            last, solver.t, values, reply["diagnostics"], elapsed, control_steps
                        )
            report["road_source"] = "CarSim dataset; camera preview not yet fed to Orin"
        else:
            vehicle.apply_control(carla.VehicleControl(throttle=0.3))
            for _ in range(round(args.seconds / 0.05)):
                last = image_at(images, world.tick())
                ticks += 1
                if viewer is not None:
                    viewer.show(last)
        last.save_to_disk(str(output / "last.png"))
        displacement = vehicle.get_location().distance(start_position)
        if displacement < 0.1:
            raise RuntimeError("Vehicle did not move")
        report.update(
            passed=True,
            camera_frames=ticks,
            control_steps=control_steps,
            displacement_m=displacement,
            max_orin_rtt_ms=max_rtt,
            max_normalized_constraint_violation=max_violation,
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if type(exc).__name__ == "ViewerClosed":
            report["cancelled_by_user"] = True
        if viewer is not None and not viewer.closed:
            viewer.finish("运行停止：" + report["error"])
        raise
    finally:
        if connection is not None:
            connection.close()
        if solver is not None:
            solver.close()
        for actor in reversed(actors):
            if actor.type_id.startswith("sensor."):
                actor.stop()
            actor.destroy()
        world.apply_settings(original)
        if trace:
            (output / "trace.json").write_text(json.dumps(trace, allow_nan=False))
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print("Artifacts:", output)
    if viewer is not None:
        viewer.finish(f"仿真完成 · {args.seconds:g} 秒 · 可保留窗口查看曲线，关闭窗口退出")
    return viewer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scene", "plugin", "triple"], default="scene")
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--orin-host", default="192.168.50.2")
    parser.add_argument("--visualize", action="store_true", help="显示实时相机、控制力与加速度曲线")
    arguments = parser.parse_args()
    if not math.isfinite(arguments.seconds) or arguments.seconds < 0.05 or arguments.seconds > 30:
        parser.error("seconds must be between 0.05 and 30")
    live = run(arguments)
    if live is not None:
        live.wait()
