"""Record sequential real TCP/CarSim/Orin runs on several installed CARLA towns."""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from runtime.paths import ROOT, load_carla


def choose_start(client, town):
    """Prefer long, initially straight lanes with a signal-controlled intersection."""
    if list(client.get_world().get_actors().filter("vehicle.*")):
        raise RuntimeError("A vehicle run is still active; refusing to replace its map")
    world = client.load_world(town)
    world.wait_for_tick(20)
    road_map = world.get_map()
    lights = [a.get_location() for a in world.get_actors().filter("traffic.traffic_light")]
    candidates, fallback = [], []
    for index, transform in enumerate(road_map.get_spawn_points()):
        waypoint = road_map.get_waypoint(transform.location)
        if waypoint.is_junction or waypoint.lane_width < 3.2:
            continue
        initial_yaw = waypoint.transform.rotation.yaw
        first_turn, first_junction = 600, None
        points, junctions = [], set()
        for station in range(0, 601, 2):
            location = waypoint.transform.location
            points.append([station, location.x, location.y])
            if waypoint.is_junction:
                if any(math.hypot(p.x - location.x, p.y - location.y) < 55 for p in lights):
                    junctions.add(waypoint.junction_id)
                    if first_junction is None:
                        first_junction = station
            delta = abs((waypoint.transform.rotation.yaw - initial_yaw + 180) % 360 - 180)
            if delta > 25 and first_turn == 600:
                first_turn = station
            upcoming = waypoint.next(2)
            if not upcoming:
                break
            previous = waypoint.transform.rotation.yaw
            waypoint = min(
                upcoming,
                key=lambda p, yaw=previous: abs((p.transform.rotation.yaw - yaw + 180) % 360 - 180),
            )
        fallback.append(
            dict(
                spawn=index,
                score=first_turn,
                first_turn_m=first_turn,
                first_signal_junction_m=first_junction,
                signal_junctions=len(junctions),
                route=points,
                signal_route_available=False,
            )
        )
        if first_junction is None or first_junction < 45:
            continue
        score = first_turn + min(3, len(junctions)) * 20 - abs(first_junction - 110) * 0.1
        candidates.append(
            dict(
                spawn=index,
                score=score,
                first_turn_m=first_turn,
                first_signal_junction_m=first_junction,
                signal_junctions=len(junctions),
                route=points,
                signal_route_available=True,
            )
        )
    if not candidates:
        if fallback:
            return max(fallback, key=lambda c: c["score"])
        raise RuntimeError("No drivable approach found in " + town)
    return max(candidates, key=lambda c: c["score"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--towns", nargs="+", default=["Town01", "Town02"])
    parser.add_argument("--seconds", type=float, default=90)
    args = parser.parse_args()
    if not 45 <= args.seconds <= 120:
        parser.error("seconds must be within [45,120]")
    if any(t not in ("Town01", "Town02", "Town03", "Town04", "Town05") for t in args.towns):
        parser.error("Only installed standard towns are supported")
    output = ROOT / "results/tcp_town_suite" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    manifest = dict(seconds=args.seconds, runs=[])
    carla = load_carla()
    for town in args.towns:
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(90)
        selection = choose_start(client, town)
        del client
        gc.collect()
        (output / (town + "_route.json")).write_text(json.dumps(selection, indent=2))
        entry = dict(
            town=town, spawn=selection["spawn"], started=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        manifest["runs"].append(entry)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("Starting", town, "spawn", selection["spawn"], flush=True)
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_environment_loop.py")),
            "--planner",
            "tcp",
            "--town",
            town,
            "--spawn",
            str(selection["spawn"]),
            "--road-profile",
            "town",
            "--unknown-contact",
            "record",
            "--seconds",
            str(args.seconds),
            "--record",
            "--visualize",
            "--close-on-complete",
        ]
        log_path = output / (town + "_run.log")
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        entry["returncode"] = result.returncode
        for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
            if line.startswith("Artifacts:"):
                folder = Path(line.partition(":")[2].strip())
                entry["artifacts"] = str(folder)
                entry["report"] = json.loads((folder / "report.json").read_text())
                break
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(
            "Finished",
            town,
            "exit",
            result.returncode,
            "artifacts",
            entry.get("artifacts"),
            flush=True,
        )
    print("Suite:", output, flush=True)


if __name__ == "__main__":
    main()
