"""Inspect native CarSim activation, UDP listeners and straight validation routes.

UDP availability is a prerequisite observation, not proof of a VS Connect handshake.
Run under conda environment carla0911. Only the probe's own actor is destroyed.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time

from carla_stack import CARLA_ROOT, ROOT


def udp_endpoints():
    command = (
        "$owner=(Get-NetTCPConnection -State Listen -LocalPort 2000 | "
        "Select-Object -First 1).OwningProcess; "
        "@(Get-NetUDPEndpoint -OwningProcess $owner -ErrorAction SilentlyContinue | "
        "Select-Object LocalAddress,LocalPort) | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout) if result.stdout.strip() else []


def straight_routes(world):
    road_map = world.get_map()
    routes = []
    for index, spawn in enumerate(road_map.get_spawn_points()):
        waypoint = road_map.get_waypoint(spawn.location)
        yaw = math.radians(spawn.rotation.yaw)
        distance = 0
        for _ in range(40):
            choices = waypoint.next(5)
            if not choices:
                break
            waypoint = min(
                choices,
                key=lambda w: abs(
                    (w.transform.rotation.yaw - spawn.rotation.yaw + 180) % 360 - 180
                ),
            )
            p = waypoint.transform.location
            dx, dy = p.x - spawn.location.x, p.y - spawn.location.y
            along = math.cos(yaw) * dx + math.sin(yaw) * dy
            lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
            angle = abs((waypoint.transform.rotation.yaw - spawn.rotation.yaw + 180) % 360 - 180)
            if abs(lateral) > 0.5 or angle > 2 or along <= distance:
                break
            distance = along
        routes.append(dict(spawn=index, straight_distance_m=distance))
    return sorted(routes, key=lambda row: row["straight_distance_m"], reverse=True)[:12]


def main():
    sys.path.insert(0, str(CARLA_ROOT / "PythonAPI/carla/dist/carla-0.9.11-py3.7-win-amd64.egg"))
    import carla

    output = ROOT / "results/carla_native_probe" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30)
    world = client.get_world()
    if list(world.get_actors().filter("vehicle.*")):
        raise RuntimeError("World already has vehicles")
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    report = dict(map=world.get_map().name, udp_before=udp_endpoints())
    actor = None
    try:
        report["straight_routes"] = straight_routes(world)
        world.apply_settings(settings)
        actor = world.spawn_actor(
            world.get_blueprint_library().find("vehicle.tesla.model3"),
            world.get_map().get_spawn_points()[0],
        )
        world.tick()
        source = (
            CARLA_ROOT
            / "Engine/Plugins/Marketplace/CarSim/ThirdParty/VehicleSim/Data/Runs/ue4simfile.sim"
        )
        text = re.sub(
            r"(?m)^SET_MACRO \$\(OUTPUT_FILE_PREFIX\)\$.*$",
            "SET_MACRO $(OUTPUT_FILE_PREFIX)$ " + (output / "PluginRun").as_posix(),
            source.read_text(),
        )
        simfile = output / "plugin.sim"
        simfile.write_text(text)
        actor.enable_carsim(str(simfile))
        actor.use_carsim_road(False)
        start = actor.get_location()
        for _ in range(10):
            actor.apply_control(carla.VehicleControl(throttle=0.3))
            world.tick()
        report["native_displacement_m"] = start.distance(actor.get_location())
        report["udp_after_enable"] = udp_endpoints()
        report["vs_connect_default_port"] = 4367
        report["vs_connect_handshake_verified"] = False
        report["suspension_variables_verified"] = False
    finally:
        if actor is not None:
            actor.destroy()
        world.apply_settings(original)
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print("Artifacts:", output)


if __name__ == "__main__":
    main()
