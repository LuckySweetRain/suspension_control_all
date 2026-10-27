"""Deploy to the user's Orin via SSH. Password is prompted, never saved.

Generate the deployment MAT/fixture with prepare_orin_data.m first.
Run from any working directory: python carsim_mpc_quick/orin/deploy_orin.py
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
from pathlib import Path

import paramiko


def deploy(host="192.168.1.41"):
    source = Path(__file__).resolve().parent
    results = source.parents[1] / "results" / "orin"
    mat = results / "quick_mpc_data.mat"
    fixture = results / "parity_fixture.mat"
    if not mat.is_file() or not fixture.is_file():
        raise FileNotFoundError("Run prepare_orin_data.m first")
    digest = hashlib.sha256(mat.read_bytes()).hexdigest()
    (results / "quick_mpc_data.sha256").write_text(digest + "\n", encoding="ascii")
    client = paramiko.SSHClient()
    known = results / "ssh_known_hosts"
    if known.is_file():
        client.load_host_keys(str(known))
    # Trust on initial setup; later connections verify the saved device host key.
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host, username="nvidia", password=getpass.getpass("Orin SSH password: "), timeout=8
    )
    client.save_host_keys(str(known))
    remote = "/home/nvidia/suspension_control/orin"

    def run(command):
        _, out, err = client.exec_command(command, timeout=60)
        output, errors = out.read().decode(), err.read().decode()
        print(output, end="")
        if errors:
            print(errors, end="")
        status = out.channel.recv_exit_status()
        if status:
            raise RuntimeError(f"Remote command failed ({status}): {command}")

    try:
        run(f"mkdir -p {remote} /home/nvidia/.config/systemd/user")
        with client.open_sftp() as sftp:
            for filename in (
                "controller.py",
                "server.py",
                "verify_controller.py",
                "requirements.txt",
                "native_solver.py",
                "qp_native.cpp",
                "depth_preview.py",
                "environment_server.py",
            ):
                sftp.put(str(source / filename), f"{remote}/{filename}")
            sftp.put(str(mat), f"{remote}/quick_mpc_data.mat")
            sftp.put(str(fixture), f"{remote}/parity_fixture.mat")
            unit = f"""[Unit]
Description=Active suspension CarSim preview MPC on Orin
After=network.target

[Service]
Type=simple
WorkingDirectory={remote}
Environment=OPENBLAS_NUM_THREADS=1
Environment=OMP_NUM_THREADS=1
ExecStart=/usr/bin/python3 -u {remote}/server.py --host 0.0.0.0
Restart=on-failure
RestartSec=2
NoNewPrivileges=true

[Install]
WantedBy=default.target
"""
            with sftp.open(
                "/home/nvidia/.config/systemd/user/suspension-mpc.service", "w"
            ) as stream:
                stream.write(unit)
            environment_unit = unit.replace(
                "Active suspension CarSim preview MPC on Orin",
                "CARLA terrain and depth-image suspension controller",
            )
            environment_unit = environment_unit.replace(
                f"{remote}/server.py --host 0.0.0.0", f"{remote}/environment_server.py"
            )
            environment_unit = environment_unit.replace(
                "Restart=on-failure",
                f"StandardOutput=append:{remote}/environment.log\nStandardError=append:{remote}/environment.log\nRestart=on-failure",
            )
            with sftp.open(
                "/home/nvidia/.config/systemd/user/suspension-environment.service", "w"
            ) as stream:
                stream.write(environment_unit)
        run(
            f"cd {remote} && g++ -std=c++17 -O3 -Wall -Wextra -Werror -fPIC -shared qp_native.cpp -o qp_native.so"
        )
        run(f"cd {remote} && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 verify_controller.py")
        run(
            "systemctl --user daemon-reload && systemctl --user enable --now suspension-mpc.service"
        )
        run("systemctl --user restart suspension-mpc.service")
        run(
            "systemctl --user enable --now suspension-environment.service && systemctl --user restart suspension-environment.service"
        )
        run("systemctl --user --no-pager status suspension-mpc.service")
        print(f"Deployed MAT sha256={digest}")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.41")
    args = parser.parse_args()
    deploy(args.host)
