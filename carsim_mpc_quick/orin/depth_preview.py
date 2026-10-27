"""Orin-side geometric preview from timestamped CARLA depth/semantic camera images.

City roads use simulator semantic truth. Isolated procedural test roads use a known XY
footprint and unlabelled-mesh depth. Neither mode uses a trained RGB segmenter. Only
initial contact elevations calibrate the state frame; future heights come from images.
"""

from __future__ import annotations

import base64
import zlib

import numpy as np


class DepthPreview:
    def __init__(self):
        self.frame = None
        self.observed_time = None
        self.points = np.empty((0, 3))
        self.valid_points = 0
        self.reference_plane = None
        self.initial_road = None
        self.previous_reference = None
        self.profile_cache = None
        self.coverage_fractions = [0.0, 0.0]
        self.mask_mode = None
        self.spatial_tree = None

    def observe(self, packet):
        mode = packet.get("road_mask_mode", "semantic_truth")
        if mode not in ("semantic_truth", "opendrive_test_road"):
            raise ValueError("Unsupported road mask mode")
        if self.mask_mode is not None and mode != self.mask_mode:
            raise ValueError("Road mask mode changed within the run")
        self.mask_mode = mode
        if self.initial_road is None:
            calibration = np.asarray(packet.get("initial_road_heights_m"), dtype=float)
            if calibration.shape != (2,) or not np.all(np.isfinite(calibration)):
                raise ValueError("Initial axle road-height calibration is required")
            self.initial_road = calibration
        width, height = int(packet["width"]), int(packet["height"])
        if not (32 <= width <= 640 and 32 <= height <= 480):
            raise ValueError("Unsupported camera dimensions")
        arrays = []
        for name in ("depth", "semantic"):
            decoder = zlib.decompressobj()
            raw = decoder.decompress(
                base64.b64decode(packet[name], validate=True), width * height * 4 + 1
            )
            if len(raw) != width * height * 4 or not decoder.eof:
                raise ValueError("Invalid compressed camera frame")
            arrays.append(np.frombuffer(raw, np.uint8).reshape(height, width, 4))
        depth, semantic = arrays
        channels = depth[:, :, :3].astype(float)
        d = (channels[:, :, 2] + 256 * channels[:, :, 1] + 65536 * channels[:, :, 0]) * (
            1000 / 16777215
        )
        vv, uu = np.indices(d.shape)
        focal = width / (2 * np.tan(np.deg2rad(float(packet["fov"])) / 2))
        labels = [0, 6, 7] if mode == "opendrive_test_road" else [6, 7]
        mask = np.isin(semantic[:, :, 2], labels) & (d > 0.2) & (d < 20)
        mask &= (vv % 2 == 0) & (uu % 2 == 0)
        local = np.column_stack(
            (
                d[mask],
                (uu[mask] + 0.5 - width / 2) * d[mask] / focal,
                -(vv[mask] + 0.5 - height / 2) * d[mask] / focal,
                np.ones(np.count_nonzero(mask)),
            )
        )
        transform = np.asarray(packet["camera_to_carsim"], dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("Invalid camera calibration")
        cloud = (local @ transform.T)[:, :3]
        if mode == "opendrive_test_road":
            # Runtime procedural meshes have no road semantic tag in 0.9.11.
            # This isolated world contains only the road and ego vehicle; use
            # its known XY footprint, never its authored elevation, to mask depth.
            roi = np.asarray(packet.get("road_roi_carsim"), dtype=float)
            if (
                roi.shape != (4,)
                or not np.all(np.isfinite(roi))
                or roi[0] >= roi[1]
                or roi[2] >= roi[3]
            ):
                raise ValueError("Invalid procedural road footprint")
            cloud = cloud[
                (cloud[:, 0] >= roi[0])
                & (cloud[:, 0] <= roi[1])
                & (cloud[:, 1] >= roi[2])
                & (cloud[:, 1] <= roi[3])
            ]
        if len(cloud) < 50 or not np.all(np.isfinite(cloud)):
            raise ValueError("Insufficient valid image road points")
        if self.reference_plane is None:
            near = cloud[
                (np.abs(cloud[:, 1] - transform[1, 3]) < 1.2) & (cloud[:, 0] < transform[0, 3] + 8)
            ]
            if len(near) < 50:
                raise ValueError("Insufficient near-field image points for road-frame calibration")
            design = np.column_stack((np.ones(len(near)), near[:, :2]))
            self.reference_plane = np.linalg.lstsq(design, near[:, 2], rcond=None)[0]
            residual = near[:, 2] - design @ self.reference_plane
            keep = np.abs(residual - np.median(residual)) < 0.03
            if np.count_nonzero(keep) >= 50:
                self.reference_plane = np.linalg.lstsq(design[keep], near[keep, 2], rcond=None)[0]
        self.points = np.concatenate((self.points, cloud))
        # Retain a bounded local history to cover the road now occluded beneath the car.
        center = float(transform[0, 3])
        if packet.get("driving_preview_mode"):
            self.points = self.points[
                np.linalg.norm(self.points[:, :2] - transform[:2, 3], axis=1) < 24
            ]
        else:
            self.points = self.points[
                (self.points[:, 0] > center - 8)
                & (self.points[:, 0] < center + 20)
                & (np.abs(self.points[:, 1] - transform[1, 3]) < 8)
            ]
        # Keep spatial coverage, not the last N pixels. Repeated near-field
        # pixels otherwise evict a road patch before the rear axle reaches it.
        # Retain the newest observation in each 5 cm x 10 cm ground cell.
        cells = np.floor(self.points[:, :2] / [0.05, 0.1]).astype(np.int64)
        _, latest = np.unique(cells[::-1], axis=0, return_index=True)
        self.points = self.points[len(self.points) - 1 - latest]
        self.frame, self.observed_time = int(packet["frame"]), float(packet["sim_time"])
        self.valid_points = len(cloud)
        self.profile_cache = None
        self.spatial_tree = None

    def preview(self, y, cfg, now, driving_path=None):
        if self.observed_time is None or not -1e-6 <= now - self.observed_time <= 0.15:
            raise ValueError("Missing or stale camera preview; no static-road fallback")
        y = np.asarray(y, dtype=float)[-39:]
        if driving_path is not None:
            return self._driving_preview(y, cfg, now, driving_path)
        mask = np.abs(self.points[:, 1] - np.mean(y[34:38])) < 1.2
        points = self.points[mask]
        if len(points) < 50:
            raise ValueError("Insufficient road-image coverage along the current track")
        if self.profile_cache is not None and np.array_equal(mask, self.profile_cache[0]):
            _, stations, heights = self.profile_cache
        else:
            bins = np.round(points[:, 0] / 0.05).astype(int)
            residual_height = points[:, 2] - (
                self.reference_plane[0] + points[:, :2] @ self.reference_plane[1:]
            )
            # Sort once, rather than scanning the historical cloud for every bin.
            order = np.argsort(bins)
            sorted_bins = bins[order]
            boundaries = np.flatnonzero(np.diff(sorted_bins)) + 1
            heights = np.array(
                [np.median(group) for group in np.split(residual_height[order], boundaries)]
            )
            stations = sorted_bins[np.r_[0, boundaries]] * 0.05
            # Reuse only when the actual selected points are identical. New frames
            # invalidate this cache; lateral track changes are checked every step.
            self.profile_cache = (mask, stations, heights)
        v = max(abs(y[38]) / 3.6, cfg["preview"]["minimum_speed_mps"])
        n, dt = int(cfg["horizon"]), cfg["control_dt"]
        distances = np.arange(n + 1) * v * dt
        xf, xr = np.mean(y[[30, 32]]), np.mean(y[[31, 33]])
        zf, zr = (
            np.interp(xf + distances, stations, heights),
            np.interp(xr + distances, stations, heights),
        )
        w = np.column_stack((zf[:-1], np.diff(zf) / dt, zr[:-1], np.diff(zr) / dt)).ravel()
        # Extrapolated/occluded near field is explicitly assigned larger uncertainty.
        observed_f = (xf + distances[:-1] >= stations[0]) & (xf + distances[:-1] <= stations[-1])
        observed_r = (xr + distances[:-1] >= stations[0]) & (xr + distances[:-1] <= stations[-1])
        self.coverage_fractions = [float(np.mean(observed_f)), float(np.mean(observed_r))]
        hb_f, hb_r = np.where(observed_f, 0.015, 0.030), np.where(observed_r, 0.015, 0.030)
        bound = np.column_stack((hb_f, np.full(n, 0.03 * v), hb_r, np.full(n, 0.03 * v))).ravel()
        if not np.all(np.isfinite(w)):
            raise ValueError("Nonfinite image preview")
        return (
            w,
            bound,
            [float(distances[-1]), float(distances[-1]), min(self.coverage_fractions)],
            self._state_offset(y, cfg, now),
        )

    def _state_offset(self, y, cfg, now, velocity_xy=None):
        xf, xr = np.mean(y[[30, 32]]), np.mean(y[[31, 33]])
        yf, yr = np.mean(y[[34, 36]]), np.mean(y[[35, 37]])
        plane_at_axles = (
            self.reference_plane[0] + np.array([[xf, yf], [xr, yr]]) @ self.reference_plane[1:]
        )
        axle_shift = plane_at_axles - self.initial_road
        # State displacements start at each axle's initialized equilibrium, while
        # the camera road is absolute elevation minus an affine reference plane.
        # Transform BOTH axle heights, body heave and pitch into that same frame.
        # This preserves suspension travel and physical tyre deflection exactly.
        axle_velocity = np.full(2, self.reference_plane[1] * y[38] / 3.6)
        if velocity_xy is not None:
            axle_velocity[:] = self.reference_plane[1:] @ velocity_xy
        if self.previous_reference is not None:
            previous_time, previous_plane = self.previous_reference
            if now > previous_time:
                axle_velocity = (plane_at_axles - previous_plane) / (now - previous_time)
        self.previous_reference = (now, plane_at_axles)
        a, b = cfg["vehicle"]["a"], cfg["vehicle"]["b"]
        dz = (b * axle_shift[0] + a * axle_shift[1]) / (a + b)
        pitch = (axle_shift[0] - axle_shift[1]) / (a + b)
        velocity = (b * axle_velocity[0] + a * axle_velocity[1]) / (a + b)
        pitch_rate = (axle_velocity[0] - axle_velocity[1]) / (a + b)
        offset = np.array(
            [
                dz,
                velocity,
                pitch,
                pitch_rate,
                axle_shift[0],
                axle_velocity[0],
                axle_shift[1],
                axle_velocity[1],
            ]
        )
        return offset

    def _driving_preview(self, y, cfg, now, path):
        """Sample the measured 2D road cloud along a short commanded-curvature arc.

        The four actual tyre positions anchor the rollout. Neither map elevation
        nor generated-road formulas enter this path; missing coverage disables
        active control through the existing warmup supervisor.
        """
        from scipy.spatial import cKDTree

        pose = np.asarray(path["pose"], dtype=float)
        steer = float(path["steer_rad"])
        lateral_speed = float(path.get("lateral_speed_mps", 0.0))
        if (
            pose.shape != (3,)
            or not np.isfinite(pose).all()
            or not np.isfinite([steer, lateral_speed]).all()
        ):
            raise ValueError("Invalid driving preview pose or steering")
        n, dt = int(cfg["horizon"]), cfg["control_dt"]
        speed = max(abs(y[38]) / 3.6, cfg["preview"]["minimum_speed_mps"])
        distances = np.arange(n + 1) * speed * dt
        curvature = np.tan(steer) / (cfg["vehicle"]["a"] + cfg["vehicle"]["b"])
        yaw = pose[2]
        angles = distances * curvature
        if abs(curvature) < 1e-8:
            shift = distances[:, None] * np.array([np.cos(yaw), np.sin(yaw)])
        else:
            shift = (
                np.column_stack(
                    (np.sin(yaw + angles) - np.sin(yaw), np.cos(yaw) - np.cos(yaw + angles))
                )
                / curvature
            )
        if abs(curvature) < 1e-8:
            shift += (np.arange(n + 1) * dt * lateral_speed)[:, None] * np.array(
                [-np.sin(yaw), np.cos(yaw)]
            )
        else:
            shift += (
                lateral_speed
                / (speed * curvature)
                * np.column_stack(
                    (np.cos(yaw + angles) - np.cos(yaw), np.sin(yaw + angles) - np.sin(yaw))
                )
            )
        wheels = np.column_stack((y[30:34], y[34:38]))
        delta = wheels - pose[:2]
        c, s = np.cos(angles), np.sin(angles)
        rotation = np.stack((np.column_stack((c, -s)), np.column_stack((s, c))), axis=1)
        targets = pose[:2] + shift[:, None, :] + np.einsum("tij,wj->twi", rotation, delta)
        if self.spatial_tree is None:
            self.spatial_tree = cKDTree(self.points[:, :2])
        distance, indices = self.spatial_tree.query(targets.reshape(-1, 2), k=8)
        residuals = self.points[:, 2] - (
            self.reference_plane[0] + self.points[:, :2] @ self.reference_plane[1:]
        )
        height, observed = [], []
        for d, index in zip(distance, indices):
            valid = d <= 0.25
            observed.append(np.count_nonzero(valid) >= 3 and d[0] <= 0.15)
            height.append(float(np.median(residuals[index[valid] if np.any(valid) else index])))
        height = np.asarray(height).reshape(n + 1, 4)
        observed = np.asarray(observed).reshape(n + 1, 4)
        front, rear = np.mean(height[:, [0, 2]], axis=1), np.mean(height[:, [1, 3]], axis=1)
        interval_observed = observed[:-1] & observed[1:]
        seen_front, seen_rear = (
            np.all(interval_observed[:, [0, 2]], axis=1),
            np.all(interval_observed[:, [1, 3]], axis=1),
        )
        self.coverage_fractions = [float(np.mean(seen_front)), float(np.mean(seen_rear))]
        w = np.column_stack(
            (front[:-1], np.diff(front) / dt, rear[:-1], np.diff(rear) / dt)
        ).ravel()
        bound = np.column_stack(
            (
                np.where(seen_front, 0.015, 0.030),
                np.full(n, 0.03 * speed),
                np.where(seen_rear, 0.015, 0.030),
                np.full(n, 0.03 * speed),
            )
        ).ravel()
        if not np.isfinite(w).all():
            raise ValueError("Nonfinite curved-path image preview")
        return (
            w,
            bound,
            [float(distances[-1]), float(distances[-1]), min(self.coverage_fractions)],
            self._state_offset(
                y,
                cfg,
                now,
                np.array([np.cos(yaw), np.sin(yaw)]) * (y[38] / 3.6)
                + np.array([-np.sin(yaw), np.cos(yaw)]) * lateral_speed,
            ),
        )
