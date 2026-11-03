"""Wait for an identified training process, then evaluate and deploy final weights.

Run with the management Python (Paramiko and psutil). Credentials are read from
stdin/getpass and held only in process memory, never saved in the experiment.
This pipeline does not claim real-time or performance superiority merely by finishing.
"""

from __future__ import annotations

import argparse
import getpass
import json
import shlex
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import paramiko
import psutil

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conda", default="E:/anaconda3/Scripts/conda.exe")
    parser.add_argument("--carla-python", default="E:/anaconda3/envs/carla0911/python.exe")
    parser.add_argument("--host", default="192.168.50.2")
    parser.add_argument("--planner-port", type=int, default=55013)
    parser.add_argument("--policy-port", type=int, default=55112)
    parser.add_argument("--password-stdin", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.training_run = args.training_run.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    password = (
        sys.stdin.readline().rstrip("\r\n")
        if args.password_stdin
        else getpass.getpass("Orin SSH password: ")
    )
    if not password:
        raise ValueError("Missing SSH authentication")
    state = dict(stage="waiting_for_training", completed=False, started_at=time.time())
    save(
        args.output / "plan.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )

    def stage(name):
        state.update(stage=name, updated_at=time.time())
        save(args.output / "status.json", state)
        print(name, flush=True)

    def run(name, script, *arguments):
        stage(name)
        command = [
            args.conda,
            "run",
            "--no-capture-output",
            "-n",
            "suspension-control",
            "python",
            str(ROOT / "scripts" / script),
            *[str(x) for x in arguments],
        ]
        save(args.output / (name + "_command.json"), command)
        with (args.output / (name + ".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state["child_pid"] = process.pid
            save(args.output / "status.json", state)
            code = process.wait()
        if code:
            raise RuntimeError(f"Stage {name} exited {code}; inspect its log")

    connection = None
    try:
        stage("waiting_for_training")
        try:
            training = psutil.Process(args.training_pid)
            creation = training.create_time()
            if args.training_run.name not in " ".join(training.cmdline()):
                raise ValueError("PID does not identify the requested training run")
            while training.is_running() and training.create_time() == creation:
                if list(args.training_run.glob("episode_*/failure.json")):
                    raise RuntimeError("Training failure detected; evaluation not started")
                time.sleep(5)
        except psutil.NoSuchProcess:
            pass
        checkpoint = args.training_run / "final.pt"
        if not checkpoint.is_file():
            raise RuntimeError("Training process ended without final.pt")
        run("audit_training", "audit_cosim_training.py", args.training_run)
        audit = json.loads((args.training_run / "audit.json").read_text())
        if not audit["run_complete"]:
            raise RuntimeError("Full training audit did not pass")
        base = [
            "--carla-python",
            args.carla_python,
            "--episodes",
            "1",
            "--seconds",
            "10",
            "--road-profile",
            "combined",
        ]
        context = ["--planner", "qwen", "--planner-port", args.planner_port, "--qwen-context-only"]
        runs = {}

        def evaluate(name, mode, extra):
            folder = args.output / name
            run(name, "train_cosim_ppo.py", *base, "--mode", mode, "--output", folder, *extra)
            run(name + "_metrics", "summarize_cosim_ppo.py", folder)
            metrics = json.loads((folder / "metrics.json").read_text())["episodes"]
            if len(metrics) != 1 or not metrics[0]["completed"] or metrics[0]["steps"] != 1000:
                raise RuntimeError(f"Incomplete physical evaluation: {name}")
            runs[name] = metrics[0]
            return folder

        evaluate("passive", "passive", [])
        local = evaluate("local_qwen", "evaluate", ["--checkpoint", checkpoint, *context])
        evaluate("local_no_qwen", "evaluate", ["--checkpoint", checkpoint])
        bundle = args.output / "policy_bundle"
        run(
            "export",
            "export_cosim_policy.py",
            "--checkpoint",
            checkpoint,
            "--transitions",
            local / "episode_0000/transitions.jsonl",
            "--output",
            bundle,
        )
        stage("deploy_final_policy")
        connection = paramiko.SSHClient()
        connection.load_system_host_keys()
        connection.load_host_keys(str(ROOT / "results/orin/ssh_known_hosts"))
        connection.connect(args.host, username="nvidia", password=password, timeout=15)
        password = None
        remote = "/home/nvidia/suspension_control/ppo_candidates/" + args.output.name
        unit = "suspension-ppo-" + args.output.name.replace("_", "-")

        def ssh(command):
            _, out, err = connection.exec_command(command, timeout=120)
            text, errors = out.read().decode(), err.read().decode()
            code = out.channel.recv_exit_status()
            if code:
                raise RuntimeError(f"Remote command failed ({code}): {errors}")
            return text + errors

        ssh("mkdir -p " + shlex.quote(remote))
        with connection.open_sftp() as sftp:
            for source in bundle.iterdir():
                if source.is_file():
                    sftp.put(str(source), remote + "/" + source.name)
            for name in ("serve_cosim_policy.py", "benchmark_cosim_policy.py"):
                sftp.put(str(ROOT / "scripts" / name), remote + "/" + name)
        executable = "/mnt/nvme/qwen-drive/environments/qwen-drive-py312/bin/python"
        benchmark = ssh(
            " ".join(
                shlex.quote(x) for x in [executable, remote + "/benchmark_cosim_policy.py", remote]
            )
        )
        (args.output / "orin_benchmark.log").write_text(benchmark, encoding="utf-8")
        with connection.open_sftp() as sftp:
            sftp.get(remote + "/benchmark.json", str(args.output / "orin_benchmark.json"))
        command = [
            "systemd-run",
            "--user",
            "--unit=" + unit,
            "--property=WorkingDirectory=" + remote,
            "--property=StandardOutput=append:" + remote + "/server.log",
            "--property=StandardError=append:" + remote + "/server.log",
            executable,
            "-u",
            remote + "/serve_cosim_policy.py",
            remote,
            "--port",
            str(args.policy_port),
        ]
        start = ssh(" ".join(shlex.quote(x) for x in command))
        (args.output / "orin_service.log").write_text(start, encoding="utf-8")
        manifest = json.loads((bundle / "manifest.json").read_text())
        deadline = time.monotonic() + 90
        while True:
            try:
                with socket.create_connection((args.host, args.policy_port), timeout=3) as client:
                    client.settimeout(5)
                    client.sendall(b'{"op":"status"}\n')
                    with client.makefile("rb") as incoming:
                        reply = json.loads(incoming.readline(16385))
                    if reply.get("policy_sha256") != manifest["policy_sha256"]:
                        raise ValueError("Wrong policy is listening on evaluation port")
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError("Orin inference endpoint did not become ready") from None
                time.sleep(1)
        evaluate(
            "orin_qwen",
            "evaluate",
            [
                "--checkpoint",
                checkpoint,
                *context,
                "--inference-host",
                args.host,
                "--inference-port",
                args.policy_port,
                "--policy-bundle",
                bundle,
            ],
        )
        if runs["orin_qwen"].get("orin_inference_samples") != 1000:
            raise RuntimeError("Evaluation did not use Orin for every action")
        if not runs["local_qwen"]["qwen_plans"] or not runs["orin_qwen"]["qwen_plans"]:
            raise RuntimeError("Missing real Qwen predictions")
        if any(abs(m["travelled_m"] - runs["passive"]["travelled_m"]) > 0.5 for m in runs.values()):
            raise RuntimeError("Distance mismatch invalidates comfort comparison")
        baseline = runs["passive"]["vertical_accel_rms_mps2"]
        for metrics in runs.values():
            metrics["vertical_rms_change_vs_passive_percent"] = 100 * (
                metrics["vertical_accel_rms_mps2"] / baseline - 1
            )
        save(
            args.output / "comparison.json",
            dict(
                policy_sha256=manifest["policy_sha256"],
                checkpoint=str(checkpoint),
                remote=remote,
                unit=unit,
                evaluations=runs,
                scope="one deterministic held-out combined road; offline synchronous; not hard real-time",
                vertical_rms_lower_in_this_run=all(
                    runs[name]["vertical_rms_change_vs_passive_percent"] < 0
                    for name in ("local_qwen", "orin_qwen")
                ),
            ),
        )
        state["completed"] = True
        stage("completed_synchronous_acceptance")
    except BaseException:
        state.update(error=traceback.format_exc(), completed=False)
        stage("failed")
        raise
    finally:
        if connection:
            connection.close()


if __name__ == "__main__":
    main()
