"""One co-simulation launcher for Windows and WSL, without copying solver code."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "carsim_mpc_quick/orin"


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", type=Path)
    command.add_argument("--backend", choices=("auto", "windows-interop"), default="auto")
    command.add_argument("--server-only", action="store_true", help=argparse.SUPPRESS)
    command.add_argument("--check-only", action="store_true")
    command.add_argument("--dry-run", action="store_true", help="Print commands; do not launch")
    command.add_argument("--seconds", type=float, default=10)
    command.add_argument("--spawn", type=int, default=159)
    command.add_argument("--town", default="")
    command.add_argument("--planner", choices=("qwen", "tcp", "transfuser"))
    command.add_argument("--orin-host")
    command.add_argument(
        "--road-profile", choices=("town", "wave", "bump", "pothole", "combined"), default="town"
    )
    command.add_argument("--road-curvature", type=float, default=0)
    command.add_argument("--nav-command", choices=("straight", "left", "right"), default="straight")
    command.add_argument("--unknown-contact", choices=("stop", "record"), default="stop")
    command.add_argument("--no-visualize", action="store_true")
    command.add_argument("--record", action="store_true")
    command.add_argument("--compare", action="store_true")
    command.add_argument("--close-on-complete", action="store_true")
    return command


def endpoint_ready(host, port):
    try:
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except OSError:
        return False


def wsl_command(args):
    """Only control crosses WSL interop; sensor/solver loops stay on Windows."""
    bridge = shutil.which("powershell.exe")
    if bridge is None or shutil.which("wslpath") is None:
        raise RuntimeError("This backend needs WSL Windows interop (powershell.exe and wslpath)")

    def windows_path(path):
        return subprocess.check_output(["wslpath", "-w", str(path.resolve())], text=True).strip()

    command = [bridge, "-NoProfile", "-File", windows_path(SOURCE / "start_environment_loop.ps1")]
    command += ["-Seconds", str(args.seconds), "-SpawnIndex", str(args.spawn)]
    command += ["-RoadProfile", args.road_profile, "-RoadCurvature", str(args.road_curvature)]
    command += ["-NavCommand", args.nav_command, "-UnknownContact", args.unknown_contact]
    for name, value in [
        ("Planner", args.planner),
        ("Town", args.town),
        ("OrinHost", args.orin_host),
    ]:
        if value:
            command += ["-" + name, value]
    if args.config:
        command += ["-Config", windows_path(args.config)]
    for name, enabled in [
        ("CheckOnly", args.check_only),
        ("NoVisualize", args.no_visualize),
        ("Record", args.record),
        ("Compare", args.compare),
        ("CloseOnComplete", args.close_on_complete),
    ]:
        if enabled:
            command.append("-" + name)
    return command


def windows_command(args, paths):
    name = "run_comparison.py" if args.compare else "run_environment_loop.py"
    road = "combined" if args.compare and args.road_profile == "town" else args.road_profile
    command = [sys.executable, str(SOURCE / name), "--seconds", str(args.seconds)]
    command += [
        "--orin-host",
        args.orin_host or paths.SETTINGS["orin_host"],
        "--road-profile",
        road,
    ]
    if args.compare:
        if args.no_visualize:
            command.append("--no-visualize")
        return command
    command += ["--spawn", str(args.spawn), "--unknown-contact", args.unknown_contact]
    if not args.no_visualize:
        command.append("--visualize")
    if args.planner:
        command += ["--planner", args.planner, "--nav-command", args.nav_command]
    if args.town:
        command += ["--town", args.town]
    if args.road_curvature:
        command += ["--road-curvature", str(args.road_curvature)]
    if args.record:
        command.append("--record")
    if args.close_on_complete:
        command.append("--close-on-complete")
    return command


def ensure_carla(paths):
    if endpoint_ready(paths.SETTINGS["carla_host"], 2000):
        return
    executable = paths.CARLA_ROOT / "CarlaUE4.exe"
    subprocess.Popen(
        [
            str(executable),
            "-quality-level=Low",
            "-ResX=800",
            "-ResY=600",
            "-windowed",
            "-nosound",
            "-RenderOffScreen",
        ],
        cwd=str(paths.CARLA_ROOT),
        creationflags=0x08000000,
    )
    for _ in range(30):
        if endpoint_ready(paths.SETTINGS["carla_host"], 2000):
            return
        time.sleep(1)
    raise RuntimeError("CARLA did not start on port 2000")


def run_wsl(args, paths):
    if args.compare:
        raise ValueError(
            "WSL split mode currently supports single-car runs; use --backend windows-interop for comparison"
        )
    command = windows_command(args, paths)
    command[0] = str(paths.LINUX_FRONTEND_PYTHON)
    if args.dry_run:
        print(json.dumps(dict(backend="wsl-coordinator-windows-solver", command=command), indent=2))
        return 0
    if sys.version_info[:2] != (3, 7):
        executable = paths.LINUX_FRONTEND_PYTHON
        if not executable.is_file() or executable.resolve() == Path(sys.executable).resolve():
            raise RuntimeError("Configure a Python 3.7 Linux client in linux_frontend_python")
        environment = os.environ.copy()
        environment["PATH"] = str(executable.parent) + os.pathsep + environment["PATH"]
        environment["CONDA_PREFIX"] = str(executable.parent.parent)
        return subprocess.call(
            [str(executable), str(Path(__file__)), *sys.argv[1:]], env=environment
        )
    import numpy
    from PIL import Image

    carla = paths.load_carla()
    print(
        "WSL coordinator:",
        sys.executable,
        "NumPy",
        numpy.__version__,
        "CARLA",
        carla.__file__,
        flush=True,
    )
    Image.new("RGB", (8, 8))
    if not args.no_visualize:
        import tkinter

        window = tkinter.Tk()
        window.withdraw()
        window.update()
        window.destroy()
    if args.check_only:
        print("Native Linux Python and CARLA client OK; Windows DLL will use RPC", flush=True)
        return 0
    from runtime.solver_rpc import bridge_control

    bootstrap = wsl_command(args)
    # Bootstrap only external simulator processes, never the Windows main loop.
    server_bootstrap = bootstrap + ["-ServerOnly"]
    if "-NoVisualize" not in server_bootstrap:
        server_bootstrap.append("-NoVisualize")
    subprocess.run(server_bootstrap, cwd=ROOT, check=True)
    owned, process, log = False, None, None
    try:
        try:
            bridge_control("status")
        except (OSError, RuntimeError):
            if endpoint_ready(
                paths.SETTINGS["carsim_bridge_host"], int(paths.SETTINGS["carsim_bridge_port"])
            ):
                raise RuntimeError(
                    "CarSim bridge port is busy; refusing to replace its owner"
                ) from None
            folder = ROOT / "results/wsl_bridge"
            folder.mkdir(exist_ok=True, parents=True)
            log = (folder / (time.strftime("%Y%m%d_%H%M%S") + ".log")).open("wb")
            process = subprocess.Popen(
                bootstrap + ["-SolverBridge"], cwd=ROOT, stdout=log, stderr=log
            )
            owned = True
            for _ in range(30):
                try:
                    bridge_control("status")
                    break
                except (OSError, RuntimeError):
                    if process.poll() is not None:
                        raise RuntimeError(
                            "Windows solver bridge exited; inspect results/wsl_bridge"
                        ) from None
                    time.sleep(1)
            else:
                raise RuntimeError("Windows solver bridge failed to start")
        environment = os.environ.copy()
        environment["COSIM_SOLVER_BACKEND"] = "windows-rpc"
        return subprocess.call(command, cwd=ROOT, env=environment)
    finally:
        if owned:
            try:
                bridge_control("shutdown", timeout=5)
            except OSError:
                pass  # Terrain runs close their DLL-owning process themselves.
            except RuntimeError as error:
                print("Bridge shutdown:", error, file=sys.stderr)
            if process is not None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
        if log is not None:
            log.close()


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if not math.isfinite(args.seconds) or not 0.05 <= args.seconds <= 120 or args.spawn < 0:
        cli.error("seconds must be within [0.05,120]; spawn must be nonnegative")
    if args.record and (args.planner not in ("tcp", "transfuser") or args.compare):
        cli.error("recording requires a TCP/TransFuser single-car run")
    if args.compare and (args.planner or args.town or args.close_on_complete):
        cli.error("comparison does not support planner, town or close-on-complete")
    if not math.isfinite(args.road_curvature) or abs(args.road_curvature) > 0.01:
        cli.error("road-curvature must be within [-0.01,0.01]")
    if args.road_curvature and (not args.planner or args.road_profile == "town" or args.compare):
        cli.error("curvature requires a single-car planner on a generated road")
    if (args.compare or args.road_profile != "town") and args.seconds > 13:
        cli.error("generated roads support at most 13 seconds")
    if args.config is None and os.environ.get("COSIM_CONFIG"):
        args.config = Path(os.environ["COSIM_CONFIG"])
    if args.config:
        os.environ["COSIM_CONFIG"] = str(args.config.resolve())
    if os.name != "nt" and args.backend == "windows-interop":
        command = wsl_command(args)
        if args.dry_run:
            print(json.dumps(dict(backend="wsl-windows-interop", command=command), indent=2))
            return 0
        return subprocess.call(command, cwd=ROOT)

    sys.path.insert(0, str(SOURCE))
    from runtime import paths

    if os.name != "nt":
        return run_wsl(args, paths)

    command = windows_command(args, paths)
    if args.dry_run:
        print(json.dumps(dict(backend="windows", command=command), indent=2))
        return 0
    # Must be launched inside the pinned Windows CARLA Conda environment.
    import numpy

    print("NumPy import OK", numpy.__version__, flush=True)
    if not args.no_visualize:
        import tkinter

        from PIL import Image, ImageTk

        root = tkinter.Tk()
        root.withdraw()
        image = ImageTk.PhotoImage(Image.new("RGB", (8, 8)), master=root)
        root.update()
        root.destroy()
        del image
        print("Pillow and Tk rendering OK", flush=True)
    if args.check_only:
        resources = {
            "carla_executable": paths.CARLA_ROOT / "CarlaUE4.exe",
            "carla_python_egg": paths.CARLA_EGG,
            "carsim_library": paths.CARSIM_LIBRARY,
            "carsim_database": paths.DATABASE,
            "carsim_template": paths.CARSIM_TEMPLATE,
        }
        missing = [key for key, path in resources.items() if not path.exists()]
        print(json.dumps({key: str(path) for key, path in resources.items()}, indent=2))
        if missing:
            raise FileNotFoundError("Missing co-simulation resources: " + ", ".join(missing))
        print("Resource paths OK; license and live control have not been tested", flush=True)
        return 0
    if args.server_only:
        ensure_carla(paths)
        return 0
    host = args.orin_host or paths.SETTINGS["orin_host"]
    for port in [55002] + ([55003] if args.planner else []):
        if not endpoint_ready(host, port):
            raise RuntimeError(f"Orin service unavailable: {host}:{port}")
    ensure_carla(paths)
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
