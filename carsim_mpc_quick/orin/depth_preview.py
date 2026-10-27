"""Orin-side geometric preview from timestamped CARLA depth/semantic camera images.

The semantic road mask is simulator ground truth, not a trained RGB segmentation model.
No CarSim road table or CARLA ray-cast heights enter this perception module.
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
        self.initial_cg = None
        self.profile_cache = None

    def observe(self, packet):
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
        mask = np.isin(semantic[:, :, 2], [6, 7]) & (d > 0.2) & (d < 20)
        mask &= (vv % 2 == 0) & (uu % 2 == 0)
        local = np.column_stack(
            (
                d[mask],
                (uu[mask] - width / 2) * d[mask] / focal,
                -(vv[mask] - height / 2) * d[mask] / focal,
                np.ones(np.count_nonzero(mask)),
            )
        )
        transform = np.asarray(packet["camera_to_carsim"], dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("Invalid camera calibration")
        cloud = (local @ transform.T)[:, :3]
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
        self.points = self.points[
            (self.points[:, 0] > center - 8) & (self.points[:, 0] < center + 20)
        ][-80000:]
        self.frame, self.observed_time = int(packet["frame"]), float(packet["sim_time"])
        self.valid_points = len(cloud)
        self.profile_cache = None

    def preview(self, y, cfg, now):
        if self.observed_time is None or not -1e-6 <= now - self.observed_time <= 0.15:
            raise ValueError("Missing or stale camera preview; no static-road fallback")
        y = np.asarray(y, dtype=float)[-39:]
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
        observed = (xf + distances[:-1] >= stations[0]) & (xf + distances[:-1] <= stations[-1])
        hb = np.where(observed, 0.015, 0.030)
        bound = np.column_stack((hb, np.full(n, 0.03 * v), hb, np.full(n, 0.03 * v))).ravel()
        if not np.all(np.isfinite(w)):
            raise ValueError("Nonfinite image preview")
        cg = (xf - cfg["vehicle"]["a"] + xr + cfg["vehicle"]["b"]) / 2
        if self.initial_cg is None:
            self.initial_cg = cg
        dz = self.reference_plane[1] * (cg - self.initial_cg)
        velocity = self.reference_plane[1] * y[38] / 3.6
        offset = np.array([dz, velocity, 0, 0, dz, velocity, dz, velocity])
        return (
            w,
            bound,
            [float(distances[-1]), float(distances[-1]), float(np.mean(observed))],
            offset,
        )
