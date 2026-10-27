"""Two independent CarSim processes, one CARLA clock, MPC vs passive damping."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import multiprocessing as mp
import queue
import socket
import sys
import time

import numpy as np
from carla_stack import CARLA_ROOT, ROOT, image_at
from carla_terrain import CarlaTerrain
from comparison_worker import worker
from rough_road import PROFILES, generate_world, specification, verify_mesh
from run_environment_loop import pack_vision


def receive(pipe, timeout=60):
    if not pipe.poll(timeout):
        raise TimeoutError("CarSim comparison worker timed out")
    response = pipe.recv()
    if "error" in response:
        raise RuntimeError(response["error"] + "\n" + response.get("traceback", ""))
    return response


def passive_forces(values, gain, limit):
    corners = np.clip(gain * np.asarray(values[60:64]), -limit, limit)
    return np.tile(corners, 2).tolist()


def save_pair(images, path):
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1280, 392), "#101827")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 18)
    for index, picture in enumerate(images):
        rgb = Image.frombytes(
            "RGBA", (picture.width, picture.height), bytes(picture.raw_data), "raw", "BGRA"
        ).convert("RGB")
        canvas.paste(rgb.resize((640, 360)), (index * 640, 32))
        draw.text(
            (index * 640 + 12, 10),
            "左：Orin MPC 主动悬架" if index == 0 else "右：被动悬架（主动力为零）",
            fill="#22d3ee" if index == 0 else "#fb923c",
            font=font,
        )
    canvas.save(str(path))


def run(args):
    sys.path.insert(0, str(CARLA_ROOT / "PythonAPI/carla/dist/carla-0.9.11-py3.7-win-amd64.egg"))
    import carla

    output = ROOT / "results/carla_comparison" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    report = dict(
        passed=False,
        profile=args.road_profile,
        clock="one CARLA tick owner",
        integration_dt_s=0.001,
        control_dt_s=0.01,
        camera_dt_s=0.05,
        dynamics="two process-isolated CarSim solvers",
        camera="fixed-height follow, no motion amplification",
        rgb_camera=dict(
            width=640, height=360, fov_deg=65, height_m=3, pitch_deg=-20, postprocess_effects=True
        ),
    )
    actors, pipes, processes, terrains = [], [], [], []
    incoming, connection, viewer, world, original = None, None, None, None, None
    trace = []
    start = time.perf_counter()
    try:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(60)
        world = generate_world(client, carla, args.road_profile, output, parallel=True)
        original = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
        world.apply_settings(settings)
        report["mesh_checks"] = [
            verify_mesh(world, carla, args.road_profile, output),
            verify_mesh(
                world,
                carla,
                args.road_profile,
                output,
                center_y=10,
                filename="passive_road_mesh_check.json",
            ),
        ]
        (output / "road_specification.json").write_text(
            json.dumps(specification(args.road_profile), indent=2)
        )
        vehicles, spawns, cameras, queues = [], [], [], {}
        for index in range(2):
            spawn = world.get_map().get_waypoint_xodr(index + 1, -1, 10).transform
            spawn.location.z += 0.3
            spawns.append(spawn)
            vehicle = world.spawn_actor(
                world.get_blueprint_library().find("vehicle.tesla.model3"), spawn
            )
            vehicle.set_simulate_physics(False)
            actors.append(vehicle)
            vehicles.append(vehicle)
            terrain = CarlaTerrain(
                world,
                carla,
                spawn,
                procedural_road=True,
                road_bounds=(0, 160, index * 8, index * 8 + 4),
            )
            terrains.append(terrain)
            bp = world.get_blueprint_library().find("sensor.camera.rgb")
            for key, value in dict(
                image_size_x="640",
                image_size_y="360",
                fov="65",
                sensor_tick="0",
                enable_postprocess_effects="true",
            ).items():
                bp.set_attribute(key, value)
            camera = world.spawn_actor(
                bp,
                carla.Transform(
                    carla.Location(x=4, y=spawn.location.y, z=3), carla.Rotation(pitch=-20)
                ),
            )
            actors.append(camera)
            cameras.append(camera)
            queues[index] = queue.Queue()
            camera.listen(queues[index].put)
        for name, kind in [("depth", "depth"), ("semantic", "semantic_segmentation")]:
            bp = world.get_blueprint_library().find("sensor.camera." + kind)
            for key, value in dict(
                image_size_x="320",
                image_size_y="180",
                fov="90",
                sensor_tick="0",
                lens_k="0",
                lens_kcube="0",
            ).items():
                bp.set_attribute(key, value)
            camera = world.spawn_actor(
                bp,
                carla.Transform(carla.Location(x=1.8, z=1.3), carla.Rotation(pitch=-35)),
                attach_to=vehicles[0],
            )
            actors.append(camera)
            queues[name] = queue.Queue()
            camera.listen(queues[name].put)
        report["tracks"] = [
            dict(
                role=role,
                road_id=index + 1,
                spawn_world=[
                    spawns[index].location.x,
                    spawns[index].location.y,
                    spawns[index].location.z,
                ],
                terrain_bounds_world=list(terrains[index].road_bounds),
            )
            for index, role in enumerate(["active", "passive"])
        ]
        frame = world.tick()
        images = {key: image_at(q, frame) for key, q in queues.items()}
        save_pair([images[0], images[1]], output / "first_pair.png")
        context = mp.get_context("spawn")
        initial = []
        for index, role in enumerate(["active", "passive"]):
            parent, child = context.Pipe()
            spawn = spawns[index]
            process = context.Process(
                target=worker,
                args=(
                    child,
                    str(output / role),
                    [spawn.location.x, spawn.location.y, spawn.location.z, spawn.rotation.yaw],
                    terrains[index].road_bounds,
                    role,
                ),
            )
            process.start()
            child.close()
            pipes.append(parent)
            processes.append(process)
        for index, pipe in enumerate(pipes):
            ready = receive(pipe)
            initial.append(ready)
            terrains[index].origin_cs = tuple(ready["origin_cs"])
            terrains[index].origin_calibrated = True
            terrains[index].z_origin = ready["z_origin"]
        difference = float(np.max(np.abs(np.asarray(initial[0]["values"]) - initial[1]["values"])))
        report["initial_state_max_difference"] = difference
        if difference > 1e-8:
            raise RuntimeError("Initial states differ between comparison vehicles")
        hashes = [
            hashlib.sha256((output / role / "runtime.par").read_bytes()).hexdigest()
            for role in ["active", "passive"]
        ]
        report["vehicle_parameter_hashes"] = hashes
        if hashes[0] != hashes[1]:
            raise RuntimeError("CarSim vehicle parameter files differ")
        values = [ready["values"] for ready in initial]
        if not args.no_visualize:
            from comparison_view import ComparisonView

            viewer = ComparisonView()
            viewer.show([images[0], images[1]], 0, values, [0, 0], [0, 0], warmup=True)
        connection = socket.create_connection((args.orin_host, 55002), timeout=10)
        connection.settimeout(30)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        incoming = connection.makefile("rb")
        digest = hashlib.sha256((ROOT / "results/orin/quick_mpc_data.mat").read_bytes()).hexdigest()
        vision = pack_vision(images["depth"], images["semantic"], terrains[0], 0)
        vision["initial_road_heights_m"] = initial[0]["initial_road_heights_m"]
        (output / "first_vision_packet.json").write_text(json.dumps(vision))
        sum_squares = np.zeros(2)
        for seq in range(round(args.seconds / 0.01)):
            request = dict(
                version=2,
                seq=seq,
                time=seq * 0.01,
                exports=values[0][:77],
                vision=vision,
                contact=dict(active=False),
            )
            connection.sendall((json.dumps(request, allow_nan=False) + "\n").encode())
            line = incoming.readline(32769)
            if len(line) > 32768 or not line.endswith(b"\n"):
                raise RuntimeError("Orin comparison stream disconnected")
            reply = json.loads(line)
            if (
                reply["seq"] != seq
                or reply["version"] != 2
                or reply["sha256"] != digest
                or reply["fallback"]
            ):
                raise RuntimeError("Rejected MPC comparison reply")
            baseline = reply["passive_baseline_config"]
            gain, limit = baseline["damper_gain_N_per_mmps"], baseline["corner_force_limit_N"]
            if not (
                math.isfinite(gain) and gain > 0 and math.isfinite(limit) and 0 < limit <= 5000
            ):
                raise ValueError("Invalid passive baseline parameters")
            controls = [reply["forces"], passive_forces(values[1], gain, limit)]
            for pipe, forces in zip(pipes, controls):
                pipe.send(dict(command="step", seq=seq, forces=forces))
            responses = [receive(pipe) for pipe in pipes]
            if any(
                row["seq"] != seq or abs(row["time"] - (seq + 1) * 0.01) > 1e-8 for row in responses
            ):
                raise RuntimeError("Independent solver clocks diverged")
            values = [row["values"] for row in responses]
            sum_squares += np.square([row[40] * 9.80665 for row in values])
            rms = np.sqrt(sum_squares / (seq + 1))
            trace.append(
                dict(
                    time=responses[0]["time"],
                    active=values[0],
                    passive=values[1],
                    active_forces=controls[0],
                    passive_forces=controls[1],
                    reply=reply,
                )
            )
            vision = None
            if (seq + 1) % 5 == 0:
                for index, (vehicle, terrain, state) in enumerate(zip(vehicles, terrains, values)):
                    position = terrain.to_world(
                        state[77],
                        state[78],
                        spawns[index].location.z
                        - terrain.z_origin
                        + state[38]
                        - initial[index]["values"][38],
                    )
                    vehicle.set_transform(
                        carla.Transform(
                            position,
                            carla.Rotation(
                                pitch=-state[41],
                                yaw=spawns[index].rotation.yaw - state[79],
                                roll=-state[80],
                            ),
                        )
                    )
                    cameras[index].set_transform(
                        carla.Transform(
                            carla.Location(x=position.x - 6, y=position.y, z=3),
                            carla.Rotation(pitch=-20),
                        )
                    )
                frame = world.tick()
                images = {key: image_at(q, frame) for key, q in queues.items()}
                vision = pack_vision(
                    images["depth"], images["semantic"], terrains[0], (seq + 1) * 0.01
                )
                trace[-1]["camera_frame"] = frame
                if viewer is not None:
                    viewer.show(
                        [images[0], images[1]],
                        (seq + 1) * 0.01,
                        values,
                        reply["diagnostics"][:2],
                        rms,
                        warmup=reply["preview_warmup"],
                    )
        save_pair([images[0], images[1]], output / "last_pair.png")
        report.update(
            passed=True,
            steps=len(trace),
            camera_updates=len(trace) // 5,
            acceleration_rms_mps2=dict(active=float(rms[0]), passive=float(rms[1])),
            acceleration_rms_relative_change_percent=float((rms[0] / rms[1] - 1) * 100)
            if rms[1] > 1e-8
            else None,
            passive_baseline_config=baseline,
            max_speed_difference_kph=max(
                abs(row["active"][76] - row["passive"][76]) for row in trace
            ),
            rms_definition="unweighted body vertical acceleration over all control samples",
            rms_window_seconds=[0, args.seconds],
            peak_acceleration_mps2={
                role: max(abs(row[role][40]) * 9.80665 for row in trace)
                for role in ["active", "passive"]
            },
        )
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        if incoming is not None:
            incoming.close()
        if connection is not None:
            connection.close()
        for pipe, process in zip(pipes, processes):
            if process.is_alive():
                try:
                    pipe.send(dict(command="stop"))
                    report.setdefault("workers", []).append(receive(pipe, timeout=15))
                except (OSError, EOFError, RuntimeError, TimeoutError) as error:
                    report.setdefault("cleanup_errors", []).append(str(error))
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
                report["passed"] = False
                report.setdefault("cleanup_errors", []).append("Worker did not terminate normally")
            if process.exitcode != 0:
                report["passed"] = False
                report.setdefault("cleanup_errors", []).append(
                    "Worker exit code: " + str(process.exitcode)
                )
            pipe.close()
        if report.get("cleanup_errors"):
            report["passed"] = False
        for actor in reversed(actors):
            if actor.type_id.startswith("sensor."):
                actor.stop()
            actor.destroy()
        if world is not None and original is not None:
            world.apply_settings(original)
        report.update(completed_steps=len(trace), wall_seconds=time.perf_counter() - start)
        (output / "trace.json").write_text(json.dumps(trace, allow_nan=False))
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print("Artifacts:", output, flush=True)
    if not report["passed"]:
        raise RuntimeError("Comparison cleanup failed; see " + str(output / "report.json"))
    if viewer is not None:
        viewer.finish("对照完成 · 左 MPC / 右被动 · 指标为本次未加权加速度 RMS")
    return viewer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--road-profile", choices=PROFILES, default="combined")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--orin-host", default="192.168.50.2")
    parser.add_argument("--no-visualize", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0.05 <= args.seconds <= 13:
        parser.error("seconds must be between 0.05 and 13")
    live = run(args)
    gc.collect()
    if live is not None:
        live.wait()
