"""Process-isolated CarSim instance for a lockstep active/passive comparison."""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path

from carla_terrain import CarlaTerrain
from runtime.carsim import CarSim
from runtime.paths import load_carla


def worker(connection, folder, spawn_data, bounds, role):
    carla = load_carla()

    output = Path(folder)
    output.mkdir(parents=True)
    solver = None
    rows = []
    try:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(30)
        world = client.get_world()
        spawn = carla.Transform(carla.Location(*spawn_data[:3]), carla.Rotation(yaw=spawn_data[3]))
        terrain = CarlaTerrain(world, carla, spawn, procedural_road=True, road_bounds=bounds)
        calibration = output / "origin_calibration"
        calibration.mkdir()
        reference = CarSim(calibration)
        terrain.origin_cs = tuple(reference.initial[77:79])
        terrain.origin_calibrated = True
        reference.close()
        solver = CarSim(output, terrain=terrain)
        heights = [terrain.height(solver.initial[i], solver.initial[i + 4]) for i in range(68, 72)]
        initial = dict(
            values=solver.initial,
            initial_road_heights_m=[(heights[0] + heights[2]) / 2, (heights[1] + heights[3]) / 2],
            origin_cs=list(terrain.origin_cs),
            z_origin=terrain.z_origin,
            role=role,
        )
        (output / "initial.json").write_text(json.dumps(initial, indent=2))
        connection.send(dict(ready=True, **initial))
        sequence = 0
        while True:
            command = connection.recv()
            if command["command"] == "stop":
                break
            if command["command"] != "step" or command["seq"] != sequence:
                raise ValueError("Worker sequence mismatch")
            forces = command["forces"]
            if len(forces) != 8 or not all(math.isfinite(f) and abs(f) <= 5000 for f in forces):
                raise ValueError("Invalid comparison actuator forces")
            values = solver.step(forces)
            row = dict(seq=sequence, time=solver.t, values=values, forces=forces)
            rows.append(row)
            connection.send(row)
            sequence += 1
        connection.send(
            dict(
                stopped=True,
                steps=len(rows),
                terrain_calls=terrain.calls,
                terrain_rays=terrain.rays,
            )
        )
    except Exception as error:
        try:
            connection.send(dict(error=str(error), traceback=traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if solver is not None:
            solver.close()
        (output / "trace.json").write_text(json.dumps(rows, allow_nan=False))
        connection.close()
