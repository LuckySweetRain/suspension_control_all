"""CARLA terrain + Orin depth-image preview + explicit CarSim contact-force loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time

import numpy as np

from carla_contact import FrontalContact
from carla_terrain import CarlaTerrain
from rough_road import PROFILES, generate_world, road_bounds, specification, verify_mesh
from runtime.carsim import CarSim
from runtime.paths import ROOT, load_carla
from runtime.protocol import OrinConnection
from runtime.sensors import image_at, pack_vision, spawn_camera, spawn_driving_lidar


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


def run(args):
    carla = load_carla()
    planner_name = getattr(args, "planner", None)
    curvature = getattr(args, "road_curvature", 0.0)

    output = ROOT / "results/carla_environment" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60)
    world = client.get_world()
    if list(world.get_actors().filter("vehicle.*")):
        raise RuntimeError("Another vehicle simulation is running")
    procedural = args.road_profile != "town"
    mesh_check = None
    if procedural:
        try:
            world = generate_world(client, carla, args.road_profile, output, curvature=curvature)
            mesh_check = verify_mesh(world, carla, args.road_profile, output, curvature=curvature)
        except Exception as error:
            (output / "report.json").write_text(
                json.dumps(
                    dict(
                        passed=False,
                        stage="procedural_world_setup",
                        road_profile=args.road_profile,
                        error=str(error),
                    ),
                    indent=2,
                )
            )
            print("Artifacts:", output, flush=True)
            raise
        (output / "road_specification.json").write_text(
            json.dumps(specification(args.road_profile), indent=2)
        )
    elif getattr(args, "town", None):
        if world.get_map().name.split("/")[-1] != args.town:
            world = client.load_world(args.town)
    elif world.get_map().name.split("/")[-1] == "OpenDriveMap":
        world = client.load_world("Town03")
    if args.spawn is None and not procedural:
        if world.get_map().name.split("/")[-1] != "Town03":
            raise RuntimeError(
                "Default validation route requires Town03; specify --spawn for another map"
            )
        args.spawn = 159
    if not procedural and args.spawn >= len(world.get_map().get_spawn_points()):
        raise ValueError("Spawn index is outside this map")
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    actors, solver, connection, viewer, contact = [], None, None, None, None
    planner_connection, driving_observation = None, None
    plans = []
    wall_start = time.perf_counter()
    trace, vision_audit = [], []
    report = dict(
        passed=False,
        road_source="CARLA scene terrain callback",
        preview_source="Orin depth image reconstruction",
        semantic_mask="CARLA ground truth, not a trained RGB model",
        contact_model="frontal compliant geometry contact, not PhysX",
        spawn_index=None if procedural else args.spawn,
        requested_sim_seconds=args.seconds,
        execution_mode="offline_synchronous",
        integration_dt_s=0.001,
        control_dt_s=0.01,
        camera_dt_s=0.05,
        map=world.get_map().name,
        road_profile=args.road_profile,
        road_curvature_per_m=curvature,
        mesh_check=mesh_check,
        driving_planner=planner_name,
        cruise_target_kph=getattr(args, "cruise_kph", None),
    )
    if procedural:
        report["semantic_mask"] = "depth within known test-road XY footprint; no road semantic tag"
        report["spawn_station_m"] = 10.0
    captures = []
    try:
        world.apply_settings(settings)
        if procedural:
            spawn = world.get_map().get_waypoint_xodr(1, -1, 10).transform
            spawn.location.z += 0.3
        else:
            spawn = world.get_map().get_spawn_points()[args.spawn]
        terrain = CarlaTerrain(
            world, carla, spawn, procedural_road=procedural, road_bounds=road_bounds(curvature)
        )
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
            sensor, queues[name] = spawn_camera(
                world,
                carla,
                kind,
                transform,
                width,
                height,
                attach_to=vehicle,
                lens_k=0,
                lens_kcube=0,
            )
            actors.append(sensor)
        if planner_name:
            from driving_observation import CAMERA_VIEWS, DrivingObservation

            driving_observation = DrivingObservation(planner_name)
            camera_specs = (
                [(name, yaw, 1.3, 2.3, 640, 360) for name, yaw in zip(CAMERA_VIEWS, (0, -60, 60))]
                if planner_name == "qwen"
                else [("driving_rgb", 0, -1.5, 2.0, 900, 256)]
                if planner_name == "tcp"
                else [("driving_rgb", 0, 1.3, 2.3, 400, 300)]
            )
            for name, yaw, x, z, width, height in camera_specs:
                sensor, queues[name] = spawn_camera(
                    world,
                    carla,
                    "rgb",
                    carla.Transform(carla.Location(x=x, z=z), carla.Rotation(yaw=yaw)),
                    width,
                    height,
                    fov=100,
                    attach_to=vehicle,
                    lens_k=0,
                    lens_kcube=0,
                )
                actors.append(sensor)
            if planner_name == "transfuser":
                sensor, queues["driving_lidar"] = spawn_driving_lidar(world, carla, vehicle)
                actors.append(sensor)
                report["driving_lidar"] = dict(
                    channels=64, points_per_second=600000, rotation_hz=20, yaw_deg=-90, z_m=2.5
                )
            planner_connection = OrinConnection(args.orin_host, 55003, timeout=180)
            status, _ = planner_connection.exchange({"op": "status"})
            expected_identity = {
                "qwen": "qwen-drive-1.0-sft",
                "tcp": "tcp-third-party-reproduction",
                "transfuser": "transfuser-cvpr2021-official",
            }[planner_name]
            if status.get("model") != expected_identity or not status.get("ready"):
                raise RuntimeError("Selected Orin model is not ready: " + str(status))
            report["driving_scope"] = (
                "lane following with an explicit next-junction navigation hint; not a driving benchmark"
            )
            report["driving_control_location"] = "Orin: shared PID in MPC endpoint"
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

            mask_description = "已知道路范围的深度筛选" if procedural else "语义真值掩码"
            viewer = LiveView(
                "CARLA 地形 · Orin 深度预瞄（" + mask_description + "）· 离线同步仿真",
                heading=(planner_name.upper() + "（Orin）· PID 跟踪 · MPC 悬架")
                if planner_name
                else None,
            )
            viewer.root.title("CARLA 环境反馈闭环 — 地形与深度预瞄")
            if procedural:
                viewer.root.title("CARLA 起伏路面闭环 — " + args.road_profile)
            if planner_name:
                viewer.root.title(planner_name.upper() + " · Orin PID/MPC · " + args.road_profile)
            viewer.show(images["rgb"])
        # Calibrate the CarSim CG/path-origin offset before terrain initialization.
        # Changing that offset afterwards would move a slope beneath settled tyres.
        preflight = output / "origin_calibration"
        preflight.mkdir()
        reference = CarSim(preflight)
        terrain.origin_cs = (reference.initial[77], reference.initial[78])
        terrain.origin_calibrated = True
        reference.close()
        solver = CarSim(
            output,
            terrain=terrain,
            contact=True,
            driving=bool(planner_name),
            stop_time=args.seconds + 1,
        )
        contact = FrontalContact(
            world, carla, terrain, vehicle, unknown_policy=getattr(args, "unknown_contact", "stop")
        )
        report["unknown_contact_policy"] = contact.unknown_policy
        if contact.unknown_policy == "record":
            report["contact_model"] = (
                "tagged frontal geometry only; unclassified hits recorded without force"
            )
        connection = OrinConnection(args.orin_host, 55002)
        digest = hashlib.sha256((ROOT / "results/orin/quick_mpc_data.mat").read_bytes()).hexdigest()
        values = solver.initial
        if driving_observation is not None:
            driving_observation.observe_state(values, 0)
            driving_observation.observe_images(images, 0)
        vision = pack_vision(images["depth"], images["semantic"], terrain, 0)
        if planner_name:
            vision["driving_preview_mode"] = True
            report["driving_preview"] = "four-tyre spatial depth sampling along commanded curvature"
        # One-time equilibrium calibration, not future road preview. The dynamics
        # contact surface defines where the relative CarSim state was initialized.
        initial_heights = [terrain.height(values[i], values[i + 4]) for i in range(68, 72)]
        initial_road = [float(np.mean(initial_heights[::2])), float(np.mean(initial_heights[1::2]))]
        vision["initial_road_heights_m"] = initial_road
        report["initial_road_heights_m"] = initial_road
        (output / "first_vision_packet.json").write_text(json.dumps(vision))
        navigation_waypoint = None
        for seq in range(round(args.seconds / 0.01)):
            cycle_start = time.perf_counter()
            plan = None
            if driving_observation is not None and driving_observation.ready and seq % 20 == 0:
                nav_command = {"straight": 0, "left": 1, "right": 2}[
                    getattr(args, "nav_command", "straight")
                ]
                target_xy = None
                if planner_name != "qwen":
                    from driving import Pose

                    # Advance on connected lanes; nearest-road matching at an
                    # intersection can jump to a crossing or opposing lane.
                    if navigation_waypoint is None:
                        navigation_waypoint = world.get_map().get_waypoint(spawn.location)
                    location = vehicle.get_location()
                    for _ in range(30):
                        forward = navigation_waypoint.transform.get_forward_vector()
                        base = navigation_waypoint.transform.location
                        ahead = (location.x - base.x) * forward.x + (
                            location.y - base.y
                        ) * forward.y
                        if ahead < 2:
                            break
                        connected = navigation_waypoint.next(2.0)
                        if not connected:
                            raise RuntimeError("Connected navigation route ended")
                        previous_yaw = navigation_waypoint.transform.rotation.yaw
                        navigation_waypoint = min(
                            connected,
                            key=lambda candidate, yaw=previous_yaw: abs(
                                (candidate.transform.rotation.yaw - yaw + 180) % 360 - 180
                            ),
                        )
                    waypoint = navigation_waypoint
                    upcoming = waypoint.next(12.0 + max(0.0, min(2.0, ahead)))
                    if not upcoming:
                        raise RuntimeError("Navigation route ended; no future lane waypoint")
                    desired_turn = {0: 0, 1: -90, 2: 90}[nav_command]

                    def turn_score(candidate, waypoint=waypoint, desired_turn=desired_turn):
                        delta = (
                            candidate.transform.rotation.yaw - waypoint.transform.rotation.yaw + 180
                        ) % 360 - 180
                        return abs(delta - desired_turn)

                    point = min(upcoming, key=turn_score).transform.location
                    dx, dy = point.x - spawn.location.x, point.y - spawn.location.y
                    target = (
                        terrain.cs * dx + terrain.sn * dy + terrain.origin_cs[0],
                        terrain.sn * dx - terrain.cs * dy + terrain.origin_cs[1],
                    )
                    target_xy = Pose(values[77], values[78], math.radians(values[79])).local_point(
                        *target
                    )
                packet = driving_observation.request(
                    len(plans), nav_command=nav_command, target_xy_m=target_xy
                )
                if not plans:
                    (output / "first_driving_request.json").write_text(json.dumps(packet))
                if viewer is None:
                    plan, planner_rpc_ms = planner_connection.exchange(packet)
                else:
                    plan, planner_rpc_ms = viewer.wait_for_planner(
                        lambda packet=packet: planner_connection.exchange(packet)
                    )
                if plan.get("version") != 3 or plan.get("seq") != packet["seq"]:
                    raise RuntimeError("Rejected planner reply: " + str(plan))
                expected_model = {
                    "qwen": "qwen-drive-1.0-sft",
                    "tcp": "tcp-third-party-reproduction",
                    "transfuser": "transfuser-cvpr2021-official",
                }[planner_name]
                if (
                    plan.get("model") != expected_model
                    or plan["captured_at"] != packet["captured_at"]
                ):
                    raise RuntimeError("Planner identity or capture timestamp mismatch")
                plans.append(dict(plan, rpc_ms=planner_rpc_ms))
            fx, collision = contact.force(values, seq * 0.01)
            request = dict(
                version=2,
                seq=seq,
                time=seq * 0.01,
                exports=values[:77],
                contact=collision,
                vision=vision,
            )
            if driving_observation is not None:
                request["driving"] = driving_observation.driving_state(values, plan)
                if getattr(args, "cruise_kph", None) is not None:
                    request["driving"]["cruise_speed_mps"] = args.cruise_kph / 3.6
            start = time.perf_counter()
            reply, rpc_ms = connection.exchange(request)
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
            driving_inputs = None
            if driving_observation is not None:
                driving_inputs = reply["driving"]["carsim_inputs"]
            values = solver.step(forces, fx, driving_inputs=driving_inputs)
            if driving_observation is not None:
                driving_observation.observe_state(values, solver.t)
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
                if driving_observation is not None:
                    driving_observation.observe_images(images, solver.t)
                    lane = world.get_map().get_waypoint(position)
                    right = lane.transform.get_right_vector()
                    lateral = (position.x - lane.transform.location.x) * right.x + (
                        position.y - lane.transform.location.y
                    ) * right.y
                    heading_error = (
                        spawn.rotation.yaw - values[79] - lane.transform.rotation.yaw + 180
                    ) % 360 - 180
                    trace[-1]["lane_tracking"] = dict(
                        lateral_m=lateral, heading_error_deg=heading_error, station_m=lane.s
                    )
                vision = pack_vision(images["depth"], images["semantic"], terrain, solver.t)
                if planner_name:
                    vision["driving_preview_mode"] = True
                if procedural:
                    for kind, station in [("wave", 31.25), ("bump", 62), ("pothole", 87)]:
                        if args.road_profile not in (kind, "combined") or kind in captures:
                            continue
                        current_station = (
                            world.get_map().get_waypoint(position).s if curvature else position.x
                        )
                        if current_station >= station - 4:
                            sample_folder = output / ("approach_" + kind)
                            sample_folder.mkdir()
                            sample = dict(vision, initial_road_heights_m=initial_road)
                            (sample_folder / "first_vision_packet.json").write_text(
                                json.dumps(sample)
                            )
                            images["rgb"].save_to_disk(str(sample_folder / "first.png"))
                            captures.append(kind)
                vision_audit.append(
                    dict(
                        frame=frame,
                        sim_time=solver.t,
                        depth_sha256=hashlib.sha256(bytes(images["depth"].raw_data)).hexdigest(),
                    )
                )
                if viewer is not None:
                    viewer.show(
                        images["rgb"],
                        solver.t,
                        values,
                        reply["diagnostics"],
                        rpc_ms,
                        seq + 1,
                        road_preview=reply["image_preview_w"][:4] if procedural else None,
                        driving=reply.get("driving"),
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
            distance_travelled_m=sum(
                math.hypot(b[77] - a[77], b[78] - a[78])
                for a, b in zip(
                    [solver.initial] + [row["values"] for row in trace[:-1]],
                    [row["values"] for row in trace],
                )
            ),
            feature_captures=captures,
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
        if connection is not None:
            connection.close()
        if planner_connection is not None:
            planner_connection.close()
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
            unclassified_contact_count=contact.unclassified_count if contact is not None else 0,
            unclassified_contact_samples=contact.unclassified_samples
            if contact is not None
            else [],
        )
        (output / "trace.json").write_text(json.dumps(trace, allow_nan=False))
        (output / "vision_frames.json").write_text(json.dumps(vision_audit))
        (output / "report.json").write_text(json.dumps(report, indent=2))
        if planner_name:
            (output / "driving_plans.json").write_text(json.dumps(plans, indent=2))
            tracking = [row["lane_tracking"] for row in trace if "lane_tracking" in row]
            report["driving_validation"] = dict(
                scope="integration_smoke; not a validated autonomous driving benchmark",
                model_plan_count=len(plans),
                max_lateral_error_m=max((abs(row["lateral_m"]) for row in tracking), default=None),
                max_heading_error_deg=max(
                    (abs(row["heading_error_deg"]) for row in tracking), default=None
                ),
            )
            (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print("Artifacts:", output, flush=True)
    if viewer is not None:
        viewer.finish("离线同步闭环完成；碰撞为正向柔顺接触原型，尚未保证 10 ms 实时")
    # Release solver, sensor and CARLA client references before holding the
    # completed GUI open. Old clients must not retain episodes across map loads.
    return viewer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument(
        "--cruise-kph",
        type=float,
        help="Explicit longitudinal cruise target; model retains steering and stop requests",
    )
    parser.add_argument(
        "--spawn", type=int, default=None, help="默认使用 Town03 已验证的 159 号起点"
    )
    parser.add_argument("--orin-host", default="192.168.50.2")
    parser.add_argument(
        "--town", help="Explicit CARLA city map, e.g. Town05; specify a spawn index"
    )
    parser.add_argument("--obstacle-at", type=float)
    parser.add_argument("--unknown-contact", choices=("stop", "record"), default="stop")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--road-profile", choices=("town",) + PROFILES, default="town")
    parser.add_argument(
        "--nav-command",
        choices=("straight", "left", "right"),
        default="straight",
        help="Next junction route hint; model still predicts the trajectory",
    )
    parser.add_argument(
        "--road-curvature",
        type=float,
        default=0.0,
        help="OpenDRIVE curvature [1/m], positive left; +/-0.01 maximum",
    )
    parser.add_argument(
        "--planner",
        choices=("qwen", "tcp", "transfuser"),
        help="Orin real-weight driving planner for CARLA town or procedural roads",
    )
    arguments = parser.parse_args()
    if arguments.cruise_kph is not None and (
        not arguments.planner
        or not math.isfinite(arguments.cruise_kph)
        or not 0 < arguments.cruise_kph <= 43.2
    ):
        parser.error("cruise-kph requires a driving planner and must be within (0,43.2]")
    if not math.isfinite(arguments.road_curvature) or abs(arguments.road_curvature) > 0.01:
        parser.error("road-curvature must be finite and within +/-0.01 per metre")
    if arguments.road_curvature and (not arguments.planner or arguments.road_profile == "town"):
        parser.error(
            "Curved test roads require an Orin driving planner and a procedural road profile"
        )
    if not math.isfinite(arguments.seconds) or not 0.05 <= arguments.seconds <= 120:
        parser.error("seconds must be between 0.05 and 120")
    if arguments.spawn is not None and arguments.spawn < 0:
        parser.error("spawn must be nonnegative")
    if arguments.obstacle_at is not None and (
        not math.isfinite(arguments.obstacle_at) or arguments.obstacle_at <= 3
    ):
        parser.error("obstacle-at must be finite and greater than 3 m")
    if arguments.road_profile != "town" and arguments.obstacle_at is not None:
        parser.error(
            "Procedural test roads use an isolated road-only world; obstacle-at requires town mode"
        )
    if arguments.road_profile != "town" and arguments.seconds > 13:
        parser.error("Generated road runs are limited to 13 s at the current 35 km/h setting")
    live = run(arguments)
    if live is not None:
        import gc

        gc.collect()
        live.wait()
