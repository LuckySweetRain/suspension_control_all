"""CARLA terrain + Orin depth-image preview + explicit CarSim contact-force loop."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import queue
import socket
import sys
import time
import zlib

import numpy as np
from carla_contact import FrontalContact
from carla_stack import CARLA_ROOT, ROOT, CarSim, image_at
from carla_terrain import CarlaTerrain


def timing_summary(trace):
    """Wall-clock measurements; RPC includes remote processing, not pure network RTT."""
    summary = {}
    for key in ("rpc_ms", "solver_ms", "elapsed_ms", "cycle_ms", "orin_compute_ms"):
        samples = np.asarray([row[key] for row in trace if key in row], dtype=float)
        if samples.size:
            summary[key] = dict(
                count=int(samples.size),
                p50=float(np.percentile(samples, 50)),
                p95=float(np.percentile(samples, 95)),
                p99=float(np.percentile(samples, 99)),
                max=float(np.max(samples)),
                over_10ms_fraction=float(np.mean(samples > 10)),
            )
    return summary


def pack_vision(depth, semantic, terrain, sim_time):
    if depth.frame != semantic.frame:
        raise RuntimeError("Depth/semantic camera frames are not paired")
    cs, sn = terrain.cs, terrain.sn
    p = terrain.spawn.location
    world_to_cs = np.array(
        [
            [cs, sn, 0, -cs * p.x - sn * p.y + terrain.origin_cs[0]],
            [sn, -cs, 0, -sn * p.x + cs * p.y + terrain.origin_cs[1]],
            [0, 0, 1, -terrain.z_origin],
            [0, 0, 0, 1],
        ]
    )
    return dict(
        frame=depth.frame,
        sim_time=sim_time,
        width=depth.width,
        height=depth.height,
        fov=depth.fov,
        world_to_carsim=world_to_cs.tolist(),
        camera_to_carsim=(world_to_cs @ np.array(depth.transform.get_matrix())).tolist(),
        depth=base64.b64encode(zlib.compress(bytes(depth.raw_data), 1)).decode(),
        semantic=base64.b64encode(zlib.compress(bytes(semantic.raw_data), 1)).decode(),
    )


def run(args):
    sys.path.insert(0, str(CARLA_ROOT / "PythonAPI/carla/dist/carla-0.9.11-py3.7-win-amd64.egg"))
    import carla

    output = ROOT / "results/carla_environment" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30)
    world = client.get_world()
    if list(world.get_actors().filter("vehicle.*")):
        raise RuntimeError("Another vehicle simulation is running")
    if args.spawn is None:
        if world.get_map().name.split("/")[-1] != "Town03":
            raise RuntimeError(
                "Default validation route requires Town03; specify --spawn for another map"
            )
        args.spawn = 159
    if args.spawn >= len(world.get_map().get_spawn_points()):
        raise ValueError("Spawn index is outside this map")
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    actors, solver, connection, viewer, contact = [], None, None, None, None
    incoming = None
    wall_start = time.perf_counter()
    trace, vision_audit = [], []
    report = dict(
        passed=False,
        road_source="CARLA scene terrain callback",
        preview_source="Orin depth image reconstruction",
        semantic_mask="CARLA ground truth, not a trained RGB model",
        contact_model="frontal compliant geometry contact, not PhysX",
        spawn_index=args.spawn,
        requested_sim_seconds=args.seconds,
        execution_mode="offline_synchronous",
        integration_dt_s=0.001,
        control_dt_s=0.01,
        camera_dt_s=0.05,
        map=world.get_map().name,
    )
    try:
        world.apply_settings(settings)
        spawn = world.get_map().get_spawn_points()[args.spawn]
        terrain = CarlaTerrain(world, carla, spawn)
        vehicle = world.spawn_actor(
            world.get_blueprint_library().find("vehicle.tesla.model3"), spawn
        )
        actors.append(vehicle)
        vehicle.set_simulate_physics(False)
        queues = {}
        for name, kind, transform, width, height in [
            (
                "rgb",
                "rgb",
                carla.Transform(carla.Location(x=-6, z=3), carla.Rotation(pitch=-12)),
                640,
                360,
            ),
            (
                "depth",
                "depth",
                carla.Transform(carla.Location(x=1.8, z=1.3), carla.Rotation(pitch=-35)),
                320,
                180,
            ),
            (
                "semantic",
                "semantic_segmentation",
                carla.Transform(carla.Location(x=1.8, z=1.3), carla.Rotation(pitch=-35)),
                320,
                180,
            ),
        ]:
            bp = world.get_blueprint_library().find("sensor.camera." + kind)
            for key, value in dict(
                image_size_x=str(width),
                image_size_y=str(height),
                sensor_tick="0.0",
                fov="90",
                lens_k="0",
                lens_kcube="0",
            ).items():
                bp.set_attribute(key, value)
            sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
            actors.append(sensor)
            queues[name] = queue.Queue()
            sensor.listen(queues[name].put)
        if args.obstacle_at is not None:
            p = terrain.to_world(args.obstacle_at, 0, 0.05)
            barrier = world.spawn_actor(
                world.get_blueprint_library().find("static.prop.streetbarrier"),
                carla.Transform(p, carla.Rotation(yaw=spawn.rotation.yaw + 90)),
            )
            actors.append(barrier)
        frame = world.tick()
        images = {k: image_at(q, frame) for k, q in queues.items()}
        images["rgb"].save_to_disk(str(output / "first.png"))
        images["depth"].save_to_disk(str(output / "first_depth_raw.png"))
        images["semantic"].save_to_disk(str(output / "first_semantic_raw.png"))
        if args.visualize:
            from live_view import LiveView

            viewer = LiveView("CARLA 地形 · Orin 深度图预瞄（语义真值掩码）· 碰撞为正向接触原型")
            viewer.root.title("CARLA 环境反馈闭环 — 地形与深度预瞄")
            viewer.show(images["rgb"])
        # Calibrate the CarSim CG/path-origin offset before terrain initialization.
        # Changing that offset afterwards would move a slope beneath settled tyres.
        preflight = output / "origin_calibration"
        preflight.mkdir()
        reference = CarSim(preflight)
        terrain.origin_cs = (reference.initial[77], reference.initial[78])
        terrain.origin_calibrated = True
        reference.close()
        solver = CarSim(output, terrain=terrain, contact=True)
        contact = FrontalContact(world, carla, terrain, vehicle)
        connection = socket.create_connection((args.orin_host, 55002), timeout=10)
        connection.settimeout(30)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        incoming = connection.makefile("rb")
        digest = hashlib.sha256((ROOT / "results/orin/quick_mpc_data.mat").read_bytes()).hexdigest()
        values = solver.initial
        vision = pack_vision(images["depth"], images["semantic"], terrain, 0)
        # One-time equilibrium calibration, not future road preview. The dynamics
        # contact surface defines where the relative CarSim state was initialized.
        initial_heights = [terrain.height(values[i], values[i + 4]) for i in range(68, 72)]
        initial_road = [float(np.mean(initial_heights[::2])), float(np.mean(initial_heights[1::2]))]
        vision["initial_road_heights_m"] = initial_road
        report["initial_road_heights_m"] = initial_road
        (output / "first_vision_packet.json").write_text(json.dumps(vision))
        for seq in range(round(args.seconds / 0.01)):
            cycle_start = time.perf_counter()
            fx, collision = contact.force(values, seq * 0.01)
            request = dict(
                version=2,
                seq=seq,
                time=seq * 0.01,
                exports=values[:77],
                contact=collision,
                vision=vision,
            )
            start = time.perf_counter()
            connection.sendall((json.dumps(request, allow_nan=False) + "\n").encode())
            line = incoming.readline(32769)
            rpc_ms = (time.perf_counter() - start) * 1000
            if len(line) > 32768 or not line.endswith(b"\n"):
                raise RuntimeError("Orin environment stream disconnected")
            reply = json.loads(line)
            if (
                reply["version"] != 2
                or reply["seq"] != seq
                or reply["sha256"] != digest
                or reply["preview_source"] != "depth_image"
                or reply["fallback"]
            ):
                report["rejected_reply"] = reply
                raise RuntimeError("Rejected environment control reply")
            forces = reply["forces"]
            if len(forces) != 8 or not all(math.isfinite(f) and abs(f) <= 5000 for f in forces):
                raise RuntimeError("Invalid actuator force")
            solver_start = time.perf_counter()
            values = solver.step(forces, fx)
            solver_ms = (time.perf_counter() - solver_start) * 1000
            elapsed = (time.perf_counter() - start) * 1000
            trace.append(
                dict(
                    time=solver.t,
                    values=values,
                    forces=forces,
                    contact=collision,
                    reply=reply,
                    elapsed_ms=elapsed,
                    rpc_ms=rpc_ms,
                    solver_ms=solver_ms,
                    orin_compute_ms=reply["compute_ms"],
                )
            )
            vision = None
            if (seq + 1) % 5 == 0:
                position = terrain.to_world(
                    values[77],
                    values[78],
                    spawn.location.z - terrain.z_origin + values[38] - solver.initial[38],
                )
                vehicle.set_transform(
                    carla.Transform(
                        position,
                        carla.Rotation(
                            pitch=-values[41], yaw=spawn.rotation.yaw - values[79], roll=-values[80]
                        ),
                    )
                )
                frame = world.tick()
                images = {k: image_at(q, frame) for k, q in queues.items()}
                vision = pack_vision(images["depth"], images["semantic"], terrain, solver.t)
                vision_audit.append(
                    dict(
                        frame=frame,
                        sim_time=solver.t,
                        depth_sha256=hashlib.sha256(bytes(images["depth"].raw_data)).hexdigest(),
                    )
                )
                if viewer is not None:
                    viewer.show(
                        images["rgb"], solver.t, values, reply["diagnostics"], rpc_ms, seq + 1
                    )
            trace[-1]["cycle_ms"] = (time.perf_counter() - cycle_start) * 1000
        images["rgb"].save_to_disk(str(output / "last.png"))
        report.update(
            passed=True,
            steps=len(trace),
            terrain_callback_calls=terrain.calls,
            terrain_rays=terrain.rays,
            terrain_height_min=min(terrain.cache.values()),
            terrain_height_max=max(terrain.cache.values()),
            image_updates=len(vision_audit),
            contact_events=contact.events,
            max_contact_force_N=contact.max_force,
            final_speed_kph=values[76],
            max_normalized_qp_violation=max(t["reply"]["diagnostics"][3] for t in trace),
            negative_saturation_fraction=np.mean(
                np.array([t["reply"]["diagnostics"][:2] for t in trace]) <= -1999, axis=0
            ).tolist(),
            positive_saturation_fraction=np.mean(
                np.array([t["reply"]["diagnostics"][:2] for t in trace]) >= 1999, axis=0
            ).tolist(),
        )
    except Exception as error:
        report["error"] = str(error)
        if "images" in locals():
            images["rgb"].save_to_disk(str(output / "failure.png"))
        raise
    finally:
        if incoming is not None:
            incoming.close()
        if connection is not None:
            connection.close()
        if solver is not None:
            solver.close()
        for actor in reversed(actors):
            if actor.type_id.startswith("sensor."):
                actor.stop()
            actor.destroy()
        world.apply_settings(original)
        report.update(
            completed_steps=len(trace),
            completed_sim_seconds=trace[-1]["time"] if trace else 0,
            wall_seconds=time.perf_counter() - wall_start,
            timing=timing_summary(trace),
            rejected_contact=contact.rejected_contact if contact is not None else None,
        )
        (output / "trace.json").write_text(json.dumps(trace, allow_nan=False))
        (output / "vision_frames.json").write_text(json.dumps(vision_audit))
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print("Artifacts:", output)
    if viewer is not None:
        viewer.finish("离线同步闭环完成；碰撞为正向柔顺接触原型，尚未保证 10 ms 实时")
        viewer.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument(
        "--spawn", type=int, default=None, help="默认使用 Town03 已验证的 159 号起点"
    )
    parser.add_argument("--orin-host", default="192.168.50.2")
    parser.add_argument("--obstacle-at", type=float)
    parser.add_argument("--visualize", action="store_true")
    arguments = parser.parse_args()
    if not math.isfinite(arguments.seconds) or not 0.05 <= arguments.seconds <= 30:
        parser.error("seconds must be between 0.05 and 30")
    if arguments.spawn is not None and arguments.spawn < 0:
        parser.error("spawn must be nonnegative")
    if arguments.obstacle_at is not None and (
        not math.isfinite(arguments.obstacle_at) or arguments.obstacle_at <= 3
    ):
        parser.error("obstacle-at must be finite and greater than 3 m")
    run(arguments)
