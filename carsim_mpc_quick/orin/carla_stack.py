"""Bounded CARLA 0.9.11 / CarSim 2020 / Orin integration checks.

Run with conda run -n carla0911 python ... (legacy CARLA ABI requires Python 3.7).
The triple mode owns CarSim in Python; CARLA vehicle physics is disabled.
It demonstrates pose/camera synchronization, NOT CARLA terrain-driven tyre contact
or image-derived road preview. Road inputs remain the validated CarSim dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time

from runtime.carsim import CarSim
from runtime.paths import CARLA_ROOT, ROOT, load_carla
from runtime.protocol import OrinConnection
from runtime.sensors import image_at, spawn_camera


def run(args):
    carla = load_carla()

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
        camera, images = spawn_camera(
            world,
            carla,
            "rgb",
            carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-12)),
            640,
            360,
            attach_to=vehicle,
        )
        actors.append(camera)
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
            connection = OrinConnection(args.orin_host, 55001, timeout=2, connect_timeout=5)
            digest = hashlib.sha256(
                (ROOT / "results/orin/quick_mpc_data.mat").read_bytes()
            ).hexdigest()
            values = solver.initial
            for seq in range(round(args.seconds / 0.01)):
                request = {"version": 1, "seq": seq, "time": seq * 0.01, "exports": values[:77]}
                reply, elapsed = connection.exchange(request)
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
