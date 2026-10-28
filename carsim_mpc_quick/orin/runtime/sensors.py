"""CARLA camera frame pairing and depth transport; no simulation clock ownership."""

from __future__ import annotations

import base64
import queue
import time
import zlib

import numpy as np


def spawn_camera(
    world, carla, kind, transform, width, height, fov=90, attach_to=None, **attributes
):
    """Return a camera and its queue; ownership and world ticks stay at the caller."""
    blueprint = world.get_blueprint_library().find("sensor.camera." + kind)
    settings = dict(image_size_x=width, image_size_y=height, sensor_tick="0.0", fov=fov)
    settings.update(attributes)
    for key, value in settings.items():
        blueprint.set_attribute(key, str(value))
    sensor = (
        world.spawn_actor(blueprint, transform)
        if attach_to is None
        else world.spawn_actor(blueprint, transform, attach_to=attach_to)
    )
    frames = queue.Queue()
    sensor.listen(frames.put)
    return sensor, frames


def image_at(images, frame):
    deadline = time.monotonic() + 15
    while True:
        image = images.get(timeout=max(0.01, deadline - time.monotonic()))
        if image.frame == frame:
            return image
        if image.frame > frame or time.monotonic() > deadline:
            raise RuntimeError("Camera/frame synchronization failed")


def spawn_driving_lidar(world, carla, vehicle):
    """Original CVPR2021 leaderboard LiDAR settings (10 Hz, half-scan at 20 FPS)."""
    blueprint = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    for key, value in dict(
        channels=64,
        points_per_second=600000,
        rotation_frequency=10,
        range=85,
        upper_fov=10,
        lower_fov=-30,
        sensor_tick=0.0,
    ).items():
        blueprint.set_attribute(key, str(value))
    sensor = world.spawn_actor(
        blueprint,
        carla.Transform(carla.Location(x=1.3, z=2.5), carla.Rotation(yaw=-90)),
        attach_to=vehicle,
    )
    frames = queue.Queue()
    sensor.listen(frames.put)
    return sensor, frames


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
    packet = dict(
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
    if terrain.procedural_road:
        xmin, xmax, ymin, ymax = terrain.road_bounds
        corners = (
            np.array(
                [[xmin, ymin, 0, 1], [xmax, ymin, 0, 1], [xmin, ymax, 0, 1], [xmax, ymax, 0, 1]]
            )
            @ world_to_cs.T
        )
        packet["road_mask_mode"] = "opendrive_test_road"
        packet["road_roi_carsim"] = [
            float(corners[:, 0].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].min()),
            float(corners[:, 1].max()),
        ]
    return packet
