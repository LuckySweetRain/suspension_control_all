"""Real TransFuser CVPR2021 / TCP trajectory branches with shared protocol v3.

Official model code stays in a pinned external checkout. ImageNet initialization
is disabled only during construction; a strict full-checkpoint load is mandatory.
The TCP checkpoint currently available is an explicitly labelled reproduction.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import io
import math
import sys
import time
from pathlib import Path
from unittest.mock import patch

from driving import Pose, Trajectory
from qwen_planner_server import serve_planner


class _CheckpointMetadata:
    """Inert Lightning callback metadata; no Lightning code executes at load time."""


class LegacyPlanner:
    def __init__(self, backend, source, checkpoint):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Driving models must run on Orin CUDA")
        self.backend = backend
        self.checkpoint = checkpoint.resolve()
        self.digest = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        sys.path.insert(0, str(source.resolve()))
        package = "TCP" if backend == "tcp" else "transfuser"
        module = importlib.import_module(package + ".model")
        config = importlib.import_module(package + ".config").GlobalConfig()
        torch.set_num_threads(4)
        if backend == "tcp":
            resnet = importlib.import_module("TCP.resnet").resnet34
            with patch.object(module, "resnet34", lambda **kw: resnet(pretrained=False)):
                self.model = module.TCP(config)
            callback = "pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint"
            with torch.serialization.safe_globals([(_CheckpointMetadata, callback)]):
                weights = torch.load(str(self.checkpoint), map_location="cpu", weights_only=True)
            weights = weights["state_dict"]
            weights = {k.removeprefix("model."): v for k, v in weights.items()}
            self.identity = "tcp-third-party-reproduction"
        else:
            original = module.models.resnet34
            with patch.object(module.models, "resnet34", lambda **kw: original(weights=None)):
                self.model = module.TransFuser(config, "cuda")
            weights = torch.load(str(self.checkpoint), map_location="cpu", weights_only=True)
            self.identity = "transfuser-cvpr2021-official"
        self.model.load_state_dict(weights, strict=True)
        self.model = self.model.cuda().eval()
        self.config = config
        self.tcp_policy = None
        self.transfuser_points = None
        self.transfuser_prediction_time = None

    def reset_model_controllers(self):
        for name, prefix in (("turn_controller", "turn"), ("speed_controller", "speed")):
            previous = getattr(self.model, name)
            setattr(
                self.model,
                name,
                type(previous)(
                    K_P=getattr(self.config, prefix + "_KP"),
                    K_I=getattr(self.config, prefix + "_KI"),
                    K_D=getattr(self.config, prefix + "_KD"),
                    n=getattr(self.config, prefix + "_n"),
                ),
            )

    def infer(self, request):
        import numpy as np
        import torch
        from PIL import Image
        from torchvision.transforms import functional as tf

        if request["version"] != 3:
            raise ValueError("Expected planning protocol v3")
        origin = Pose(*request["origin"])
        captured = float(request["captured_at"])
        speed = float(request["speed_mps"])
        target = np.asarray(request["target_xy_m"], dtype=float)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("Expected finite forward/left target point in metres")
        if not math.isfinite(captured) or captured < 0 or not math.isfinite(speed) or speed < 0:
            raise ValueError("Invalid capture time or forward speed")
        with Image.open(io.BytesIO(base64.b64decode(request["rgb_jpeg"], validate=True))) as image:
            image = image.convert("RGB")
        # Both released agents use (right, negative-forward) waypoint coordinates.
        goal = torch.tensor([[-target[1], -target[0]]], device="cuda", dtype=torch.float32)
        velocity = torch.tensor([speed], device="cuda", dtype=torch.float32)
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        tcp_control = None
        tcp_action = None
        transfuser_action = None
        network_inferred = True
        with torch.inference_mode():
            if self.backend == "tcp":
                if image.size != (900, 256):
                    raise ValueError("TCP requires its native 900x256 front camera")
                rgb = tf.normalize(
                    tf.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                )
                command = request["route_command"]
                if command not in range(1, 7):
                    raise ValueError("TCP route command must be CARLA RoadOption 1..6")
                one_hot = torch.zeros((1, 6), device="cuda")
                one_hot[0, command - 1] = 1
                state = torch.cat([velocity.view(1, 1) / 12, goal, one_hot], dim=1)
                predictions = self.model(rgb.unsqueeze(0).cuda(), state, goal)
                points = predictions["pred_wp"]
                if request["seq"] == 0:
                    from tcp_policy import TCPOriginalPolicy

                    self.tcp_policy = TCPOriginalPolicy()
                    for name, prefix in (
                        ("turn_controller", "turn"),
                        ("speed_controller", "speed"),
                    ):
                        previous = getattr(self.model, name)
                        setattr(
                            self.model,
                            name,
                            type(previous)(
                                K_P=getattr(self.config, prefix + "_KP"),
                                K_I=getattr(self.config, prefix + "_KI"),
                                K_D=getattr(self.config, prefix + "_KD"),
                                n=getattr(self.config, prefix + "_n"),
                            ),
                        )
                if self.tcp_policy is None:
                    raise ValueError("TCP episode must start at sequence zero")
                steer, throttle, brake, _ = self.model.process_action(
                    predictions, command, velocity, goal
                )
                tcp_control = dict(
                    steer_right_positive=float(steer),
                    throttle=float(throttle),
                    brake=float(brake),
                    applied=False,
                )
                if not all(math.isfinite(v) for v in (steer, throttle, brake)):
                    raise RuntimeError("Nonfinite TCP control head")
                tcp_action = self.tcp_policy.step(
                    self.model,
                    predictions,
                    tcp_control,
                    velocity,
                    goal,
                    warmup=request["seq"] < self.config.seq_len,
                )
            else:
                from transfuser.data import lidar_to_histogram_features, scale_and_crop_image

                if image.size != (400, 300):
                    raise ValueError("TransFuser CVPR2021 requires a native 400x300 camera")
                raw = base64.b64decode(request["lidar_f32"], validate=True)
                cloud = np.frombuffer(raw, dtype="<f4").reshape(-1, 4)[:, :3].copy()
                if not len(cloud) or not np.isfinite(cloud).all():
                    raise ValueError("Need a finite real LiDAR frame")
                if request["lidar_preset"] != "cvpr2021_yaw_minus90_z2.5":
                    raise ValueError("Unexpected LiDAR extrinsics")
                if request["seq"] == 0:
                    self.reset_model_controllers()
                    self.transfuser_points = None
                network_inferred = request["seq"] % 2 == 0 or request["seq"] <= 4
                cloud[:, 1] *= -1
                bev = torch.from_numpy(lidar_to_histogram_features(cloud)).unsqueeze(0).cuda()
                # Preserve the released agent's raw 0..255 float preprocessing.
                rgb = torch.from_numpy(scale_and_crop_image(image)).unsqueeze(0).float().cuda()
                if network_inferred:
                    self.transfuser_points = self.model([rgb], [bev], goal, velocity)
                    self.transfuser_prediction_time = captured
                if self.transfuser_points is None:
                    raise ValueError("TransFuser episode must start at sequence zero")
                points = self.transfuser_points
                if request["seq"] < self.config.seq_len:
                    steer, throttle, brake, desired_speed = 0.0, 0.0, 0.0, 0.0
                else:
                    steer, throttle, brake, metadata = self.model.control_pid(
                        points.clone(), velocity
                    )
                    if brake < 0.05:
                        brake = 0.0
                    if throttle > brake:
                        brake = 0.0
                    desired_speed = float(metadata["desired_speed"])
                transfuser_action = dict(
                    policy="transfuser_cvpr2021_agent",
                    steer_right_positive=float(steer),
                    throttle=float(throttle),
                    brake=float(brake),
                    desired_speed_mps=desired_speed,
                )
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - begin) * 1000
        waypoints = points.detach().float().cpu().numpy()
        if waypoints.shape != (1, 4, 2) or not np.isfinite(waypoints).all():
            raise RuntimeError("Expected four finite learned waypoints")
        samples = tuple(
            (0.5 * (i + 1), float(-p[1]), float(-p[0])) for i, p in enumerate(waypoints[0])
        )
        plan = Trajectory(
            self.identity, str(self.checkpoint), request["seq"], captured, origin, samples
        )
        return {
            "version": 3,
            "seq": plan.sequence,
            "model": plan.model,
            "checkpoint": plan.checkpoint,
            "checkpoint_sha256": self.digest,
            "captured_at": captured,
            "origin": request["origin"],
            "points": plan.points,
            "inference_ms": elapsed,
            "tcp_control_head": tcp_control,
            "tcp_control": tcp_action,
            "transfuser_control": transfuser_action,
            "network_inferred": network_inferred,
            "waypoint_age_s": captured - self.transfuser_prediction_time
            if self.backend == "transfuser"
            else 0.0,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("tcp", "transfuser"), required=True)
    parser.add_argument("--source", type=Path, required=True, help="Parent of TCP/ or transfuser/")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bind", default="192.168.50.2")
    parser.add_argument("--port", type=int, default=55003)
    args = parser.parse_args()
    serve_planner(LegacyPlanner(args.backend, args.source, args.checkpoint), args.bind, args.port)
