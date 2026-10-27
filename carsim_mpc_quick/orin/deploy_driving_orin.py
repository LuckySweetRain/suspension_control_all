"""Install/switch the project's driving service without changing the MPC environment.

Requires the existing validated Orin model runtime and local pinned source/weight
cache. Password is prompted and never written to a service or project file.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import shlex
import socket
import subprocess
import time
from pathlib import Path

import paramiko

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
REMOTE = "/home/nvidia/suspension_control/driving"


def deploy(planner, host, cache, password):
    manifest = json.loads((SOURCE / "driving_models.json").read_text(encoding="utf-8"))
    weights = {
        "tcp": cache / "tcp_reproduction.ckpt",
        "transfuser": cache / "weights/transfuser/best_model.pth",
    }
    for key, repository in (("tcp", "TCP"), ("transfuser", "transfuser")):
        revision = subprocess.check_output(
            ["git", "-C", str(cache / repository), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != manifest["models"][key]["revision"]:
            raise ValueError("Unexpected source revision for " + key)
    for key, path in weights.items():
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest["models"][key]["checkpoint_sha256"]
        ):
            raise ValueError("Checkpoint hash mismatch: " + str(path))
    client = paramiko.SSHClient()
    client.load_host_keys(str(ROOT / "results/orin/ssh_known_hosts"))
    client.connect(host, username="nvidia", password=password, timeout=10)

    def run(command):
        _, out, err = client.exec_command(command, timeout=120)
        result, errors = out.read().decode(), err.read().decode()
        if out.channel.recv_exit_status():
            raise RuntimeError(result + errors)
        return result

    try:
        run(
            "mkdir -p "
            + REMOTE
            + "/vendor/TCP "
            + REMOTE
            + "/vendor/transfuser "
            + REMOTE
            + "/weights "
            + REMOTE
            + "/cache /home/nvidia/.config/systemd/user"
        )
        with client.open_sftp() as sftp:
            for name in (
                "driving.py",
                "qwen_planner_server.py",
                "legacy_planner_server.py",
                "driving_models.json",
            ):
                sftp.put(str(SOURCE / name), REMOTE + "/" + name)
            for package, folder, names in (
                ("TCP", "TCP/TCP", ("model.py", "config.py", "resnet.py")),
                ("transfuser", "transfuser/transfuser", ("model.py", "config.py", "data.py")),
            ):
                for name in names:
                    sftp.put(str(cache / folder / name), REMOTE + "/vendor/" + package + "/" + name)
                license_path = cache / folder.split("/")[0] / "LICENSE"
                sftp.put(str(license_path), REMOTE + "/vendor/" + package + "/LICENSE")
            for key, path in weights.items():
                remote_path = REMOTE + "/weights/" + key + ".pth"
                # Repeated switches reuse a verified immutable checkpoint.
                code = (
                    "import hashlib,pathlib; p=pathlib.Path("
                    + repr(remote_path)
                    + "); print(hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else 'missing')"
                )
                existing = run("python3 -c " + shlex.quote(code)).strip()
                if existing != manifest["models"][key]["checkpoint_sha256"]:
                    sftp.put(str(path), remote_path)
            executable = manifest["runtime_python"]
            if planner == "qwen":
                command = [
                    executable,
                    "-u",
                    REMOTE + "/qwen_planner_server.py",
                    "--model",
                    manifest["qwen_weights"],
                ]
            else:
                command = [
                    executable,
                    "-u",
                    REMOTE + "/legacy_planner_server.py",
                    "--backend",
                    planner,
                    "--source",
                    REMOTE + "/vendor",
                    "--checkpoint",
                    REMOTE + "/weights/" + planner + ".pth",
                ]
            unit = """[Unit]
Description=Suspension project driving planner on Orin
ConditionPathExists={executable}

[Service]
WorkingDirectory={remote}
Environment=PYTHONPATH={remote}/vendor:{qwen_code}
Environment=PYTHONDONTWRITEBYTECODE=1
Environment=HF_HUB_OFFLINE=1
Environment=HF_HOME={remote}/cache/huggingface
Environment=TRITON_CACHE_DIR={remote}/cache/triton
Environment=TORCHINDUCTOR_CACHE_DIR={remote}/cache/inductor
Environment=CUDA_CACHE_PATH={remote}/cache/cuda
Environment=OMP_NUM_THREADS=4
Environment=OPENBLAS_NUM_THREADS=4
ExecStart={command}
Restart=no
TimeoutStopSec=20

[Install]
WantedBy=default.target
""".format(
                executable=executable,
                remote=REMOTE,
                qwen_code=manifest["qwen_code"],
                command=shlex.join(command),
            )
            with sftp.open(
                "/home/nvidia/.config/systemd/user/suspension-driving.service", "w"
            ) as stream:
                stream.write(unit)
        run(
            "systemctl --user daemon-reload && systemctl --user enable suspension-driving.service && systemctl --user restart suspension-driving.service"
        )
        print(run("systemctl --user is-active suspension-driving.service"), end="")
        identity = manifest["models"][planner]["identity"]
        for _ in range(120):
            try:
                with socket.create_connection((host, 55003), timeout=2) as connection:
                    connection.sendall(b'{"op":"status"}\n')
                    with connection.makefile("rb") as incoming:
                        reply = json.loads(incoming.readline(4096))
                if reply.get("ready") and reply.get("model") == identity:
                    print("Weights loaded; selected model:", identity)
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError(
                "Planner did not become ready; inspect suspension-driving.service logs"
            )
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", choices=("qwen", "tcp", "transfuser"), required=True)
    parser.add_argument("--host", default="192.168.50.2")
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "results/autodrive_bootstrap_20261027/sources"
    )
    args = parser.parse_args()
    deploy(args.planner, args.host, args.cache, getpass.getpass("Orin SSH password: "))
