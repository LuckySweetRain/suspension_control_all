"""Separate protocol-v2 endpoint: image preview and collision state on Orin."""

from __future__ import annotations

import hashlib
import json
import socket
import time

from controller import Controller
from depth_preview import DepthPreview
from driving import DrivingSession


def serve():
    controller = Controller("quick_mpc_data.mat")
    with open("quick_mpc_data.mat", "rb") as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("0.0.0.0", 55002))
        listener.listen(1)
        print("CARLA environment endpoint ready on 55002", flush=True)
        while True:
            connection, _ = listener.accept()
            controller.reset()
            vision = DepthPreview()
            driving = DrivingSession()
            seq = 0
            with connection:
                # Offline synchronous model inference can pause the simulator
                # during first-use CUDA compilation without advancing its clock.
                connection.settimeout(300)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    with connection.makefile("rb") as incoming:
                        while True:
                            line = incoming.readline(2 * 1024 * 1024 + 1)
                            if not line:
                                break
                            if len(line) > 2 * 1024 * 1024 or not line.endswith(b"\n"):
                                raise ValueError("Invalid v2 frame")
                            request = json.loads(line)
                            now = float(request["time"])
                            if (
                                request["version"] != 2
                                or request["seq"] != seq
                                or abs(now - seq * 0.01) > 1e-7
                            ):
                                raise ValueError("Protocol clock/sequence mismatch")
                            begin = time.perf_counter()
                            if request.get("vision") is not None:
                                vision.observe(request["vision"])
                            driving_control, driving_path = None, None
                            if request.get("driving") is not None:
                                driving_control = driving.step(request["driving"], now)
                                driving_path = dict(
                                    pose=request["driving"]["pose"],
                                    steer_rad=driving_control["steer_rad"],
                                    lateral_speed_mps=request["driving"].get(
                                        "lateral_speed_mps", 0.0
                                    ),
                                )
                            preview = vision.preview(
                                request["exports"], controller.c, now, driving_path
                            )
                            contact = bool(request.get("contact", {}).get("active", False))
                            preview_warmup = min(vision.coverage_fractions) < 1.0
                            force, diag, fallback = controller.step(
                                request["exports"],
                                preview,
                                contact or preview_warmup,
                                lateral_speed_mps=(request.get("driving") or {}).get(
                                    "lateral_speed_mps", 0.0
                                ),
                            )
                            reply = dict(
                                version=2,
                                seq=seq,
                                sha256=digest,
                                forces=force,
                                diagnostics=diag,
                                fallback=fallback,
                                preview_source="depth_image",
                                road_mask_mode=vision.mask_mode,
                                passive_baseline_config=dict(
                                    damper_gain_N_per_mmps=float(
                                        controller.c["carsim_damper_gain"]
                                    ),
                                    corner_force_limit_N=float(controller.c["corner_force_limit"]),
                                ),
                                camera_frame=vision.frame,
                                valid_road_pixels=vision.valid_points,
                                image_preview_w=preview[0].tolist(),
                                image_reference_plane=vision.reference_plane.tolist(),
                                image_state_correction=preview[3].tolist(),
                                initial_road_heights_m=vision.initial_road.tolist(),
                                image_axle_coverage=vision.coverage_fractions,
                                preview_warmup=preview_warmup,
                                contact_supervisor=contact,
                                suspension_supervisor=controller.motion_status(),
                                compute_ms=(time.perf_counter() - begin) * 1000,
                            )
                            if driving_control is not None:
                                reply["driving"] = driving_control
                                reply["compute_ms"] = (time.perf_counter() - begin) * 1000
                            connection.sendall((json.dumps(reply, allow_nan=False) + "\n").encode())
                            seq += 1
                except (ValueError, KeyError, TypeError, OSError) as error:
                    print(f"Environment stream rejected: {error}", flush=True)


if __name__ == "__main__":
    serve()
