"""Continuous Chrono terrain shared by tire queries and native road geometry.

The longitudinal coordinate is prescribed by speed, not an extra mechanical
DOF. This analytic ChTerrain adds no collision forces: the existing vertical
tire constitutive law remains responsible for normal support.
"""

from __future__ import annotations

import math

import numpy as np
import pychrono as ch
import pychrono.vehicle as veh

from .roads import FourWheelRoad


class RoadTerrain(veh.ChTerrain):
    """A z-up height field with the original left/right wheel tracks.

    Station zero lies at world x = front_distance_m. Between the innermost
    left/right wheel tracks, a cubic smoothstep interpolates laterally. Outside
    those tracks the same-side height is constant, so unequal axle widths do
    not change the four original road inputs. The interpolated strip is an
    explicit geometry assumption, not a measured two-dimensional road surface.

    Only the source profile is retained; there is no reference to the plant or
    to an Irrlicht renderer. Chrono 10 enables SWIG directors for ChTerrain.
    """

    def __init__(self, road: FourWheelRoad, config: dict):
        super().__init__()
        vehicle = config["vehicle"]
        self.road = road
        self.front_distance = float(vehicle["front_distance_m"])
        self.rear_distance = float(vehicle["rear_distance_m"])
        front_track = float(vehicle["front_half_track_m"])
        rear_track = float(vehicle["rear_half_track_m"])
        self.half_track = min(front_track, rear_track)
        if (
            not np.all(
                np.isfinite([self.front_distance, self.rear_distance, front_track, rear_track])
            )
            or min(self.front_distance, self.rear_distance, front_track, rear_track) <= 0
        ):
            raise ValueError("Terrain axle distances and half tracks must be finite and positive")
        if not np.isclose(
            road.wheelbase, self.front_distance + self.rear_distance, rtol=0, atol=1e-12
        ):
            raise ValueError("Terrain and source road must use the same wheelbase")
        self.wheel_y = (front_track, -front_track, rear_track, -rear_track)
        self._profile_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def _remember(self, x: float, heights: np.ndarray, slopes: np.ndarray) -> None:
        # Bound storage independently of episode length and render mesh size.
        if len(self._profile_cache) >= 16 and x not in self._profile_cache:
            self._profile_cache.pop(next(iter(self._profile_cache)))
        self._profile_cache[x] = heights, slopes

    def _tracks_at_x(self, x: float) -> tuple[np.ndarray, np.ndarray]:
        if not math.isfinite(x):
            raise ValueError("Terrain query x must be finite")
        if x not in self._profile_cache:
            station = x - self.front_distance
            heights, slopes = self.road.sample_tracks(station)
            self._remember(x, heights, slopes)
        return self._profile_cache[x]

    def _surface(self, loc) -> tuple[float, float, float]:
        """Return height, dz/dx, dz/dy, in SI units at a world point."""
        heights, slopes = self._tracks_at_x(float(loc.x))
        y = float(loc.y)
        if not math.isfinite(y):
            raise ValueError("Terrain query y must be finite")
        if y >= self.half_track:
            return float(heights[0]), float(slopes[0]), 0.0
        if y <= -self.half_track:
            return float(heights[1]), float(slopes[1]), 0.0
        u = (y + self.half_track) / (2 * self.half_track)
        weight = u * u * (3 - 2 * u)
        weight_gradient = 3 * u * (1 - u) / self.half_track
        height = heights[1] + weight * (heights[0] - heights[1])
        longitudinal_slope = slopes[1] + weight * (slopes[0] - slopes[1])
        lateral_slope = weight_gradient * (heights[0] - heights[1])
        return float(height), float(longitudinal_slope), float(lateral_slope)

    def GetHeight(self, loc) -> float:
        return self._surface(loc)[0]

    def GetPoint(self, loc):
        return ch.ChVector3d(loc.x, loc.y, self.GetHeight(loc))

    def GetNormal(self, loc):
        _, slope_x, slope_y = self._surface(loc)
        scale = math.sqrt(1 + slope_x * slope_x + slope_y * slope_y)
        return ch.ChVector3d(-slope_x / scale, -slope_y / scale, 1 / scale)

    def GetCoefficientFriction(self, loc) -> float:
        """Nominal Chrono terrain property; unused by the vertical tire law."""
        return 0.8

    def sample_wheels(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """Query four virtual wheel paths; return heights (m), speeds (m/s).

        One vectorized source evaluation primes the two axle cross sections.
        Every wheel height then passes through GetHeight, the same interface
        used to build the visual surface. Priming also preserves the original
        arithmetic at the four wheel stations without x-offset roundoff.
        """
        if not math.isfinite(time):
            raise ValueError("Terrain wheel sample time must be finite")
        heights, velocities = self.road.sample(time)
        advance = self.road.speed * float(time)
        front_x = self.front_distance + advance
        rear_x = -self.rear_distance + advance
        self._remember(front_x, heights[:2], velocities[:2] / self.road.speed)
        self._remember(rear_x, heights[2:], velocities[2:] / self.road.speed)
        result_height, result_velocity = np.empty(4), np.empty(4)
        for index, (x, y) in enumerate(
            zip((front_x, front_x, rear_x, rear_x), self.wheel_y, strict=True)
        ):
            loc = ch.ChVector3d(x, y, 0)
            result_height[index] = self.GetHeight(loc)
            result_velocity[index] = self.road.speed * self._surface(loc)[1]
        return result_height, result_velocity

    def clear_cache(self) -> None:
        self._profile_cache.clear()
