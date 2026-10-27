"""Orin Qwen-Drive SFT trajectory endpoint, separate from the 100 Hz MPC process.

Protocol v3 is newline-delimited JSON. Three views contain real frames at
capture offsets [-1.5, -1, -.5, 0] s; state history has 16 samples at 10 Hz.
Only simulation time defines plan age. Wall-clock inference time is reported.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import socket
import time
from pathlib import Path

from driving import Pose, Trajectory

MAX_PACKET_BYTES = 24 * 1024 * 1024


class QwenPlanner:
    def __init__(self, model_path):
        import torch
        from qwen_drive import QwenDriveForPlanning

        if not torch.cuda.is_available():
            raise RuntimeError("Qwen planner requires CUDA on Orin")
        self.path = model_path.resolve()
        self.identity = "qwen-drive-1.0-sft"
        planner = self.path / "planner-sft"
        if not (planner / "config.json").is_file():
            raise FileNotFoundError("Released planner-sft weights are required")
        self.config_sha256 = hashlib.sha256((planner / "config.json").read_bytes()).hexdigest()
        torch.set_num_threads(4)
        self.model = QwenDriveForPlanning.from_pretrained(
            str(self.path),
            planner=str(planner),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
            local_files_only=True,
        ).eval()

    def infer(self, request):
        import numpy as np
        import torch
        from PIL import Image
        from qwen_drive import InferenceMode
        from qwen_drive.scene import CAMERA_VIEWS, CameraFrame, DrivingScene

        if request["version"] != 3:
            raise ValueError("Expected planning protocol version 3")
        origin = Pose(*request["origin"])
        captured = float(request["captured_at"])
        if not math.isfinite(captured) or captured < 0:
            raise ValueError("Invalid simulation capture time")
        history = request["history"]

        def array(key, shape):
            value = np.asarray(history[key], dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid history " + key)
            return value

        poses = array("poses", (16, 3))
        velocities = array("velocities", (16, 2))
        accelerations = array("accelerations", (16, 2))
        offsets = array("offsets_s", (16,))
        if not np.allclose(offsets, np.arange(-15, 1) / 10, atol=1e-6):
            raise ValueError("History must be sampled at 10 Hz, ending at capture time")
        if not np.allclose(poses[-1], 0, atol=1e-6):
            raise ValueError("History must be expressed in the capture ego frame")
        nav = request["nav_command"]
        if nav not in (0, 1, 2):
            raise ValueError("Navigation command must be straight=0, left=1, right=2")
        # Qwen's four-way ego command is distinct from its three-way nav command.
        command = np.asarray(request["driving_command"], dtype=float)
        if (
            command.shape != (4,)
            or not np.all((command == 0) | (command == 1))
            or command.sum() != 1
        ):
            raise ValueError("Expected a four-way one-hot driving command")
        views = {}
        for name in CAMERA_VIEWS:
            frames = request["views"][name]
            if len(frames) != 4:
                raise ValueError("Each view requires four actual history frames")
            views[name] = []
            for frame, offset in zip(frames, (-1.5, -1.0, -0.5, 0.0)):
                if abs(float(frame["offset_s"]) - offset) > 1e-6:
                    raise ValueError("Camera history timestamp mismatch")
                raw = base64.b64decode(frame["jpeg"], validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.width * image.height > 4096 * 2160:
                        raise ValueError("Image exceeds sensor integration limit")
                    views[name].append(CameraFrame(image.convert("RGB")))
        scene = DrivingScene(
            views=views,
            history=poses,
            history_velocity=velocities,
            history_acceleration=accelerations,
            ego_velocity=velocities[-1],
            ego_acceleration=accelerations[-1],
            driving_command=command,
            nav_command=nav,
            token="carsim-" + str(request["seq"]),
        )
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            result = self.model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=1)
        torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - begin) * 1000
        trajectory = np.asarray(result.trajectories)
        if trajectory.shape != (1, 50, 3) or not np.isfinite(trajectory).all():
            raise RuntimeError("Model did not produce 50 finite x/y/heading waypoints")
        hz = float(self.model.config.trajectory_hz)
        points = tuple(
            (float(i + 1) / hz, float(p[0]), float(p[1])) for i, p in enumerate(trajectory[0])
        )
        plan = Trajectory(
            "qwen-drive-1.0-sft", str(self.path), request["seq"], captured, origin, points
        )
        return {
            "version": 3,
            "seq": plan.sequence,
            "model": plan.model,
            "checkpoint": plan.checkpoint,
            "planner_config_sha256": self.config_sha256,
            "captured_at": captured,
            "origin": request["origin"],
            "points": plan.points,
            "headings_rad": trajectory[0, :, 2].tolist(),
            "inference_ms": inference_ms,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }


def serve(model_path, bind, port):
    planner = QwenPlanner(model_path)
    serve_planner(planner, bind, port)


def _send_reply(connection, reply):
    try:
        connection.sendall((json.dumps(reply, allow_nan=False) + "\n").encode())
    except OSError:
        return False
    return True


def serve_planner(planner, bind, port):
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((bind, port))
        listener.listen(1)
        print("Real-weight driving planner ready on " + str(port), flush=True)
        while True:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(180)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with connection.makefile("rb") as incoming:
                    previous_seq = -1
                    while True:
                        try:
                            line = incoming.readline(MAX_PACKET_BYTES + 1)
                        except OSError:
                            break
                        if not line:
                            break
                        try:
                            if len(line) > MAX_PACKET_BYTES or not line.endswith(b"\n"):
                                raise ValueError("Planning packet exceeds limit")
                            request = json.loads(line)
                            if request.get("op") == "status":
                                if not _send_reply(
                                    connection,
                                    {"version": 3, "ready": True, "model": planner.identity},
                                ):
                                    break
                                continue
                            if (
                                not isinstance(request["seq"], int)
                                or request["seq"] <= previous_seq
                            ):
                                raise ValueError("Planning sequence must strictly increase")
                            reply = planner.infer(request)
                            previous_seq = request["seq"]
                        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as error:
                            _send_reply(connection, {"error": str(error)})
                            break
                        if not _send_reply(connection, reply):
                            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bind", default="192.168.50.2")
    parser.add_argument("--port", type=int, default=55003)
    args = parser.parse_args()
    serve(args.model, args.bind, args.port)
