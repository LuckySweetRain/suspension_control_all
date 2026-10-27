"""Build timestamped Qwen requests from real CARLA images and CarSim CG history."""

from __future__ import annotations

import base64
import io
import math
from collections import deque

import numpy as np
from PIL import Image

from driving import Pose

CAMERA_VIEWS = ("<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>")


class DrivingObservation:
    def __init__(self, backend="qwen"):
        self.backend = backend
        self.states = deque(maxlen=151)
        self.frames = deque(maxlen=31)

    def observe_state(self, values, sim_time):
        pose = Pose(values[77], values[78], math.radians(values[79]))
        speed = values[76] / 3.6
        lateral = values[81] / 3.6 if len(values) > 81 else 0.0
        velocity = (
            speed * math.cos(pose.yaw) - lateral * math.sin(pose.yaw),
            speed * math.sin(pose.yaw) + lateral * math.cos(pose.yaw),
        )
        acceleration = (0.0, 0.0)
        if self.states:
            previous = self.states[-1]
            dt = sim_time - previous[0]
            if abs(dt - 0.01) > 1e-6:
                raise ValueError("Driving state must be sampled at 100 Hz")
            acceleration = tuple((v - p) / dt for v, p in zip(velocity, previous[2]))
        self.states.append((sim_time, pose, velocity, acceleration))

    def observe_images(self, images, sim_time):
        if self.frames and abs(sim_time - self.frames[-1][0] - 0.05) > 1e-6:
            raise ValueError("Driving camera images must be sampled at 20 Hz")
        # CARLA's raw image object is immutable per frame and retained until pruning.
        data = tuple(images[name] for name in CAMERA_VIEWS) if self.backend == "qwen" else images
        self.frames.append((sim_time, data))

    @property
    def ready(self):
        if self.backend != "qwen":
            return bool(self.states and self.frames)
        return len(self.states) == 151 and len(self.frames) == 31

    def request(self, sequence, nav_command=0, target_xy_m=None):
        if nav_command not in (0, 1, 2):
            raise ValueError("Navigation command must be straight=0, left=1, right=2")
        if not self.ready:
            raise RuntimeError("Need 1.5 seconds of actual image and state history")
        now, origin, _, _ = self.states[-1]
        if abs(self.frames[-1][0] - now) > 1e-6:
            raise ValueError("Latest camera and state capture times differ")
        if self.backend != "qwen":
            images = self.frames[-1][1]
            raw = images["driving_rgb"]
            stream = io.BytesIO()
            pixels = np.frombuffer(raw.raw_data, dtype=np.uint8).reshape(raw.height, raw.width, 4)
            Image.fromarray(pixels[:, :, 2::-1].copy()).save(stream, format="JPEG", quality=95)
            packet = {
                "version": 3,
                "seq": sequence,
                "captured_at": now,
                "origin": [origin.x, origin.y, origin.yaw],
                "speed_mps": math.hypot(*self.states[-1][2]),
                "target_xy_m": target_xy_m,
                "route_command": {0: 4, 1: 1, 2: 2}[nav_command],
                "rgb_jpeg": base64.b64encode(stream.getvalue()).decode("ascii"),
            }
            if self.backend == "transfuser":
                lidar = images["driving_lidar"]
                if lidar.frame != raw.frame:
                    raise ValueError("LiDAR/camera frame mismatch")
                packet.update(
                    lidar_f32=base64.b64encode(bytes(lidar.raw_data)).decode("ascii"),
                    lidar_preset="cvpr2021_yaw_minus90_z2.5",
                )
            return packet
        rows = list(self.states)[::10]
        poses, velocities, accelerations = [], [], []
        c, s = math.cos(origin.yaw), math.sin(origin.yaw)
        for _, pose, velocity, acceleration in rows:
            x, y = origin.local_point(pose.x, pose.y)
            yaw = (pose.yaw - origin.yaw + math.pi) % (2 * math.pi) - math.pi
            poses.append([x, y, yaw])
            velocities.append(
                [c * velocity[0] + s * velocity[1], -s * velocity[0] + c * velocity[1]]
            )
            accelerations.append(
                [
                    c * acceleration[0] + s * acceleration[1],
                    -s * acceleration[0] + c * acceleration[1],
                ]
            )
        views = {name: [] for name in CAMERA_VIEWS}
        for captured, images in list(self.frames)[::10]:
            for name, raw in zip(CAMERA_VIEWS, images):
                pixels = np.frombuffer(raw.raw_data, dtype=np.uint8).reshape(
                    raw.height, raw.width, 4
                )
                stream = io.BytesIO()
                Image.fromarray(pixels[:, :, 2::-1].copy()).save(stream, format="JPEG", quality=90)
                views[name].append(
                    {
                        "offset_s": round(captured - now, 6),
                        "jpeg": base64.b64encode(stream.getvalue()).decode("ascii"),
                    }
                )
        # Verified against the released straight, left-turn and right-turn demos.
        command = [0, 0, 0, 0]
        command[{0: 1, 1: 0, 2: 2}[nav_command]] = 1
        return {
            "version": 3,
            "seq": sequence,
            "captured_at": now,
            "origin": [origin.x, origin.y, origin.yaw],
            "views": views,
            "history": {
                "poses": poses,
                "velocities": velocities,
                "accelerations": accelerations,
                "offsets_s": [round(row[0] - now, 6) for row in rows],
            },
            "nav_command": nav_command,
            "driving_command": command,
        }

    @staticmethod
    def driving_state(values, plan=None):
        return {
            "pose": [values[77], values[78], math.radians(values[79])],
            "speed_mps": values[76] / 3.6,
            "lateral_speed_mps": values[81] / 3.6 if len(values) > 81 else 0.0,
            "plan": plan,
        }
