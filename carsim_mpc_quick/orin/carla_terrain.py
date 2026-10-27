"""VehicleSim terrain callback backed exclusively by CARLA scene ray intersections."""

from __future__ import annotations

import ctypes as ct
import math


class CarlaTerrain:
    def __init__(self, world, carla, spawn, procedural_road=False, road_bounds=(0, 160, 0, 4)):
        self.world, self.carla, self.spawn = world, carla, spawn
        self.procedural_road = procedural_road
        self.road_bounds = road_bounds
        self.angle = math.radians(spawn.rotation.yaw)
        self.cs, self.sn = math.cos(self.angle), math.sin(self.angle)
        self.cache = {}
        self.calls, self.rays = 0, 0
        self.errors = []
        self.origin_cs = (0.0, 0.0)
        self.origin_calibrated = False
        self.z_origin = self.height_world(spawn.location.x, spawn.location.y)
        real_pointer = ct.POINTER(ct.c_double)
        signature = ct.CFUNCTYPE(
            ct.c_int, ct.c_double, ct.c_double, ct.c_double, *([real_pointer] * 5), ct.c_void_p
        )
        self.callback = signature(self.query)

    def to_world(self, x, y, z=0):
        x, y = x - self.origin_cs[0], y - self.origin_cs[1]
        return self.carla.Location(
            x=self.spawn.location.x + self.cs * x + self.sn * y,
            y=self.spawn.location.y + self.sn * x - self.cs * y,
            z=z + self.z_origin,
        )

    def height_world(self, x, y):
        xmin, xmax, ymin, ymax = self.road_bounds
        if self.procedural_road and not (xmin <= x <= xmax and ymin <= y <= ymax):
            raise RuntimeError("Terrain query outside the generated test-road footprint")
        hits = self.world.cast_ray(
            self.carla.Location(x=x, y=y, z=self.spawn.location.z + 8),
            self.carla.Location(x=x, y=y, z=self.spawn.location.z - 8),
        )
        self.rays += 1
        # Ignore vehicles/roofs: terrain contact uses road, sidewalk and ground surfaces.
        ground = [
            h.location.z
            for h in hits
            if str(h.label) in ("Roads", "RoadLines", "Sidewalks", "Ground", "Terrain")
            or (self.procedural_road and str(h.label).upper() == "NONE")
        ]
        if not ground:
            raise RuntimeError(f"CARLA terrain has no ground hit at {x:.3f},{y:.3f}")
        return min(ground, key=lambda z: abs(z - self.spawn.location.z))

    def height(self, x, y):
        key = (round(x / 0.05), round(y / 0.05))
        if key not in self.cache:
            p = self.to_world(key[0] * 0.05, key[1] * 0.05)
            self.cache[key] = self.height_world(p.x, p.y) - self.z_origin
        return self.cache[key]

    def query(self, x, y, z, out_z, out_dx, out_dy, out_mu, out_rr, unused):
        self.calls += 1
        out_z[0], out_dx[0], out_dy[0], out_mu[0], out_rr[0] = 0, 0, 0, 0.85, 0.01
        try:
            h = self.height(x, y)
            out_z[0] = h
            out_dx[0] = (self.height(x + 0.1, y) - self.height(x - 0.1, y)) / 0.2
            out_dy[0] = (self.height(x, y + 0.1) - self.height(x, y - 0.1)) / 0.2
            return 1
        except Exception as error:
            self.errors.append(str(error))
            return 0

    def install(self, dll):
        dll.vs_road_install_terrain_function.argtypes = [type(self.callback), ct.c_void_p]
        dll.vs_road_install_terrain_function(self.callback, None)

    def check(self):
        if self.errors:
            raise RuntimeError(self.errors[-1])
