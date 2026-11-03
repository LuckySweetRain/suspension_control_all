"""Train project PPO from real CARLA/CarSim transitions via a localhost endpoint.

The legacy CARLA client runs in its own Python 3.7 process. This learner uses
Python 3.10+ and CPU; no simulated surrogate transitions are substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from rl.ppo import PPOAgent, PPOConfig, finish_ppo_trajectory  # noqa: E402


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


class RemotePolicy:
    """Fail closed on disconnection or identity mismatch; no local-policy fallback."""

    def __init__(self, host, port, manifest):
        self.digest = manifest["policy_sha256"]
        self.socket = socket.create_connection((host, port), timeout=10)
        self.socket.settimeout(30)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.incoming = self.socket.makefile("rb")
        self.seq = 0
        self.last_timing = None
        status = self.exchange(dict(op="status"))
        if status.get("policy_sha256") != self.digest or status.get("schema") != "cosim-ppo-v1":
            self.close()
            raise ValueError("Orin policy identity/schema mismatch")

    def exchange(self, request):
        self.socket.sendall((json.dumps(request, allow_nan=False) + "\n").encode())
        line = self.incoming.readline(16385)
        if len(line) > 16384 or not line.endswith(b"\n"):
            raise RuntimeError("Disconnected or invalid Orin policy response")
        return json.loads(line)

    def action(self, obs):
        start = time.perf_counter_ns()
        reply = self.exchange(
            dict(version=1, seq=self.seq, policy_sha256=self.digest, observation=obs.tolist())
        )
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if reply.get("seq") != self.seq or reply.get("policy_sha256") != self.digest:
            raise ValueError("Orin inference identity/sequence mismatch")
        action = np.asarray(reply["raw_axis_N"], dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all() or np.max(abs(action)) > 2000.01:
            raise ValueError("Invalid Orin action")
        self.last_timing = dict(
            rpc_ms=elapsed, compute_ms=reply["compute_ms"], device=reply["device"]
        )
        self.seq += 1
        return action

    def close(self):
        self.incoming.close()
        self.socket.close()


class Episode:
    """Explicit native units, deterministic actuator projection and causal context."""

    def __init__(self, agent, stream, mode="train", remote=None):
        self.agent, self.stream, self.mode = agent, stream, mode
        self.remote = remote
        self.origin = None
        self.previous = np.zeros(2)
        self.plan = None
        self.pending = None
        self.seq = 0
        self.complete = False
        self.buffer = {key: [] for key in ("obs", "act", "logp", "value", "rew", "done")}

    def observation(self, request):
        y = np.asarray(request["exports"], dtype=float)
        if y.size < 77 or not np.isfinite(y).all():
            raise ValueError("Invalid native CarSim exports")
        if self.origin is None:
            self.origin = y.copy()
        if request.get("plan") is not None:
            plan = request["plan"]
            if plan["model"] != "qwen-drive-1.0-sft":
                raise ValueError("Only genuine Qwen plans may populate Qwen context")
            self.plan = plan
        context = np.zeros(5)
        if self.plan is not None:
            age = request["time"] - self.plan["captured_at"]
            points = np.asarray(self.plan["points"], dtype=float)
            if age < -1e-6:
                raise ValueError("Future Qwen context")
            context[2] = min(age / 5, 2)
            future = points[points[:, 0] > age]
            if len(future) >= 2 and age <= 1:
                delta = future[1] - future[0]
                speed = np.linalg.norm(delta[1:]) / delta[0]
                context[:] = [speed / 20, np.arctan2(delta[2], delta[1]), age / 5, 1, 0]
        # Last context channel is reserved, zero: no invented Qwen comfort advice.
        base = np.concatenate(
            (
                [
                    y[39] / 3.6,
                    y[40] * 9.80665 / 5,
                    np.deg2rad(y[41] - self.origin[41]) / 0.1,
                    np.deg2rad(y[42]),
                ],
                (y[44:48] - self.origin[44:48]) / 80,
                y[48:52] / 1000,
                (y[52:56] - self.origin[52:56]) / np.maximum(abs(self.origin[52:56]), 1),
                self.previous / self.agent.cfg.act_limit,
                [y[76] / 72],
                context,
            )
        )
        preview = np.asarray(request["preview"], dtype=float).reshape(20, 4)
        obs = np.concatenate((base, (preview / [0.05, 0.5, 0.05, 0.5]).ravel()))
        if obs.shape != (104,) or not np.isfinite(obs).all():
            raise ValueError("Observation schema mismatch")
        return np.clip(obs, -20, 20).astype(np.float32)

    def handle(self, message):
        if message.get("version") != 1 or message["seq"] != self.seq:
            raise ValueError("Training protocol sequence/version mismatch")
        if message["op"] == "act":
            if self.pending is not None or self.complete:
                raise ValueError("Action without completed transition")
            if abs(message["time"] - self.seq * 0.01) > 1e-6:
                raise ValueError("Control clock mismatch")
            obs = self.observation(message)
            if self.mode == "passive":
                raw, logp, value = np.zeros(2), 0.0, 0.0
            elif self.mode == "evaluate":
                raw = self.remote.action(obs) if self.remote else self.agent.select_action(obs)
                logp, value = 0.0, self.agent.estimate_value(obs)
            else:
                raw, logp, value = self.agent.act_for_training(obs)
            executed = np.clip(raw, self.previous - 150, self.previous + 150)
            if message["supervisor"]:
                executed = np.zeros(2)
            y = np.asarray(message["exports"])
            corners = np.clip(3 * y[60:64] + 0.5 * executed[[0, 1, 0, 1]], -5000, 5000)
            self.pending = dict(
                request=message,
                obs=obs,
                raw=raw,
                logp=logp,
                value=value,
                executed=executed,
                corner_forces=corners,
                previous=self.previous.copy(),
            )
            return dict(
                op="action",
                seq=self.seq,
                forces=np.tile(corners, 2).tolist(),
                raw_axis_N=raw.tolist(),
                commanded_axis_N=executed.tolist(),
            )
        if message["op"] != "transition" or self.pending is None:
            raise ValueError("Transition without action")
        if abs(message["time"] - (self.seq + 1) * 0.01) > 1e-6:
            raise ValueError("Transition clock mismatch")
        p = self.pending
        y = np.asarray(message["exports"], dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Invalid next state; abort without fabricating a transition")
        travel = (y[44:48] - self.origin[44:48]) / 80
        loads = (y[52:56] - self.origin[52:56]) / np.maximum(abs(self.origin[52:56]), 1)
        terms = dict(
            accel=(y[40] * 9.80665 / 2) ** 2,
            pitch=0.2 * y[43] ** 2,
            travel=0.2 * np.mean(travel**2),
            tire=0.1 * np.mean(loads**2),
            effort=0.01 * np.mean((p["executed"] / 2000) ** 2),
            slew=0.01 * np.mean(((p["executed"] - p["previous"]) / 150) ** 2),
            violation=10 * np.mean(np.maximum(abs(travel) - 1, 0) ** 2),
        )
        reward = -float(sum(terms.values()))
        self.previous = p["executed"]
        for key, item in (
            ("obs", p["obs"]),
            ("act", p["raw"]),
            ("logp", p["logp"]),
            ("value", p["value"]),
            ("rew", reward),
            ("done", bool(message["terminated"])),
        ):
            self.buffer[key].append(item)
        self.stream.write(
            json.dumps(
                dict(
                    seq=self.seq,
                    time=message["time"],
                    reward=reward,
                    reward_terms=terms,
                    raw_action=p["raw"].tolist(),
                    executed_action=self.previous.tolist(),
                    corner_forces=p["corner_forces"].tolist(),
                    supervisor=p["request"]["supervisor"],
                    inference=self.remote.last_timing if self.remote else None,
                    exports=y.tolist(),
                    observation=p["obs"].tolist(),
                    logp=p["logp"],
                    value=p["value"],
                    terminated=message["terminated"],
                    truncated=message["truncated"],
                    plan=p["request"].get("plan"),
                ),
                allow_nan=False,
            )
            + "\n"
        )
        self.stream.flush()
        if message["terminated"] or message["truncated"]:
            # Bootstrap uses the final physical state and last available measured preview.
            final_request = dict(
                p["request"], exports=message["exports"], time=message["time"], plan=None
            )
            bootstrap = (
                0
                if message["terminated"]
                else self.agent.estimate_value(self.observation(final_request))
            )
            ret, adv = finish_ppo_trajectory(
                self.buffer["rew"],
                self.buffer["value"],
                self.buffer["done"],
                self.agent.cfg.gamma,
                self.agent.cfg.lam,
                bootstrap,
            )
            self.buffer.update(ret=ret, adv=adv)
            self.complete = True
        self.pending = None
        self.seq += 1
        return dict(op="recorded", seq=message["seq"], reward=reward)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--carla-python", required=True)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--port", type=int, default=55110)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--planner", choices=("qwen",))
    parser.add_argument("--planner-port", type=int, default=55003)
    parser.add_argument("--qwen-context-only", action="store_true")
    parser.add_argument("--async-qwen", action="store_true")
    parser.add_argument("--mode", choices=("train", "evaluate", "passive"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--inference-host")
    parser.add_argument("--inference-port", type=int, default=55111)
    parser.add_argument("--policy-bundle", type=Path)
    parser.add_argument(
        "--road-profile", choices=("wave", "bump", "pothole", "combined"), default="wave"
    )
    parser.add_argument(
        "--road-cycle",
        nargs="+",
        choices=("wave", "bump", "pothole", "combined"),
        help="Cycle training roads by episode; reserve separate roads for evaluation",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "evaluate" and args.checkpoint is None:
        parser.error("Deterministic evaluation requires --checkpoint")
    if args.qwen_context_only and args.planner != "qwen":
        parser.error("qwen-context-only requires --planner qwen")
    if args.async_qwen and not (args.planner == "qwen" and args.qwen_context_only):
        parser.error("async-qwen requires --planner qwen --qwen-context-only")
    if args.inference_host and (args.mode != "evaluate" or args.policy_bundle is None):
        parser.error("Remote inference requires --mode evaluate and --policy-bundle")
    if args.episodes < 1 or not 0.05 <= args.seconds <= 13:
        parser.error("Require episodes >= 1 and seconds between 0.05 and 13")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = replace(
        PPOConfig.from_project_config(load_config(ROOT / "configs/ppo.yaml")),
        obs_dim_base=24,
        preview_steps=20,
        preview_token_dim=4,
        act_limit=2000,
        transformer_dropout=0,
        train_epochs=4,
        minibatch_size=128,
        transformer_layers=1,
        hidden_sizes=[64, 64],
        preview_embed_dim=32,
        transformer_heads=2,
        transformer_ff_dim=64,
    )
    agent = PPOAgent(config, torch.device("cpu"))
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["cfg"] != config.__dict__:
            raise ValueError(
                "Checkpoint architecture/configuration differs from the current schema"
            )
        agent.load(args.checkpoint)
    remote_manifest = None
    if args.inference_host:
        remote_manifest = json.loads((args.policy_bundle / "manifest.json").read_text())
        if (
            remote_manifest["checkpoint_sha256"]
            != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        ):
            raise ValueError("Inference bundle and evaluation checkpoint differ")
    write_json(
        args.output / "manifest.json",
        dict(
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            ppo=config.__dict__,
            git=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            device="cpu",
            observation_schema="cosim-ppo-v1",
            scope="real CARLA terrain, CarSim dynamics, Orin measured depth preview",
            timing="offline_synchronous; not real-time validation",
            reward_sampling="10 ms endpoint samples; not integrated 1 ms reward",
            qwen_context="actual trajectory only; no learned comfort advice",
        ),
    )
    (args.output / "working_tree.patch").write_bytes(
        subprocess.check_output(["git", "diff"], cwd=ROOT)
    )
    source_dir = args.output / "source_snapshot"
    source_dir.mkdir()
    sources = [
        Path(__file__).resolve(),
        ROOT / "carsim_mpc_quick/orin/run_environment_loop.py",
        ROOT / "rl/ppo.py",
        ROOT / "rl/networks.py",
    ]
    hashes = {}
    for source in sources:
        data = source.read_bytes()
        (source_dir / source.name).write_bytes(data)
        hashes[str(source.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    write_json(source_dir / "sha256.json", hashes)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", args.port))
        listener.listen(1)
        listener.settimeout(1)
        for index in range(args.episodes):
            folder = args.output / f"episode_{index:04d}"
            folder.mkdir()
            command = [
                args.carla_python,
                str(ROOT / "carsim_mpc_quick/orin/run_environment_loop.py"),
                "--seconds",
                str(args.seconds),
                "--road-profile",
                args.road_cycle[index % len(args.road_cycle)]
                if args.road_cycle
                else args.road_profile,
                "--rl-port",
                str(args.port),
            ]
            if args.planner:
                command += ["--planner", args.planner, "--planner-port", str(args.planner_port)]
            if args.qwen_context_only:
                command += ["--qwen-context-only"]
            if args.async_qwen:
                command += ["--async-qwen"]
            write_json(folder / "command.json", command)
            with (
                (folder / "simulator.log").open("w", encoding="utf-8") as log,
                (folder / "transitions.jsonl").open("w", encoding="utf-8") as stream,
            ):
                remote = (
                    RemotePolicy(args.inference_host, args.inference_port, remote_manifest)
                    if args.inference_host
                    else None
                )
                child_env = os.environ.copy()
                prefix = Path(args.carla_python).resolve().parent
                # Python 3.7 NumPy loads native DLLs from PATH. Never inherit the
                # learner environment's DLL search priority into the CARLA child.
                child_env["PATH"] = (
                    os.pathsep.join(
                        str(prefix / entry) for entry in ("", "Library/bin", "DLLs", "Scripts")
                    )
                    + os.pathsep
                    + child_env.get("PATH", "")
                )
                child_env["CONDA_PREFIX"] = str(prefix)
                child_env.pop("PYTHONHOME", None)
                child_env.pop("PYTHONPATH", None)
                process = subprocess.Popen(
                    command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT
                )
                episode = Episode(agent, stream, mode=args.mode, remote=remote)
                try:
                    deadline = time.monotonic() + 180
                    while True:
                        try:
                            connection, _ = listener.accept()
                            break
                        except TimeoutError:
                            if process.poll() is not None or time.monotonic() > deadline:
                                raise RuntimeError(
                                    "Simulator did not connect; inspect simulator.log"
                                ) from None
                    with connection, connection.makefile("rb") as incoming:
                        connection.settimeout(300)
                        while not episode.complete:
                            line = incoming.readline(1024 * 1024 + 1)
                            if len(line) > 1024 * 1024 or not line.endswith(b"\n"):
                                raise RuntimeError("Invalid/disconnected training stream")
                            answer = episode.handle(json.loads(line))
                            connection.sendall(
                                (json.dumps(answer, allow_nan=False) + "\n").encode()
                            )
                    code = process.wait(timeout=120)
                    if code != 0 or not episode.complete:
                        raise RuntimeError(f"Simulator exit {code}; no PPO update")
                    losses = None
                    if args.mode == "train":
                        losses = agent.train_trajectory(episode.buffer)
                        if not np.isfinite(losses).all() or not all(
                            bool(torch.isfinite(p).all())
                            for model in (agent.actor, agent.value)
                            for p in model.parameters()
                        ):
                            raise RuntimeError("Nonfinite PPO update; checkpoint not accepted")
                        agent.save(folder / "checkpoint.pt")
                    summary = dict(
                        episode=index,
                        steps=episode.seq,
                        reward_sum=sum(episode.buffer["rew"]),
                        losses=losses,
                    )
                    write_json(folder / "summary.json", summary)
                    print(json.dumps(summary), flush=True)
                except BaseException:
                    write_json(
                        folder / "failure.json",
                        dict(
                            error=traceback.format_exc(),
                            simulator_pid=process.pid,
                            simulator_exit=process.poll(),
                        ),
                    )
                    if process.poll() is None:
                        # Only this invocation's child is owned here; never stop other simulators.
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.terminate()
                            process.wait(timeout=15)
                    raise
                finally:
                    if remote:
                        remote.close()
    if args.mode == "train":
        agent.save(args.output / "final.pt")


if __name__ == "__main__":
    main()
