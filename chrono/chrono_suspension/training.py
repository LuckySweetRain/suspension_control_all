"""SB3 training with process-local Chrono systems and resumable checkpoints.

All actions are normalized wheel forces, ordered FL, FR, RL, RR.  Workers
never create a renderer; the learner owns the neural network and checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "chrono"
ALGORITHMS = ("ppo", "sac")
OBSERVATION_MODES = ("privileged", "sensors")
PPO_STEPS_PER_WORKER = 256


def write_json(path: Path, payload: Any) -> None:
    """Replace a JSON manifest atomically; never expose half-written metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.writing-{uuid.uuid4().hex}")
    staging.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(staging, path)


def limit_worker_threads() -> None:
    """Set before spawning workers so BLAS/OpenMP imports inherit the limits."""
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"


def _build_worker(
    config: dict[str, Any], observation_mode: str, seed: int, rank: int, monitor_path: str | None
) -> Any:
    # This function executes in the child, never pickling a Chrono system.
    limit_worker_threads()
    import gymnasium as gym
    import torch
    from stable_baselines3.common.monitor import Monitor

    from chrono_suspension.env import ChronoSuspensionEnv

    torch.set_num_threads(1)

    class EpisodeSeedWrapper(gym.Wrapper):
        """Disjoint deterministic streams with an explicit seed at every reset."""

        def __init__(self, env: gym.Env) -> None:
            super().__init__(env)
            self.episode_index = 0

        def reset(self, *, seed: int | None = None, options: dict | None = None) -> Any:
            # Pairing injects (run seed, worker rank, episode) without overlap
            # for the supported < 1e6 episodes/worker and < 32 worker limits.
            episode_seed = 100_000_000 + run_seed * 32_000_000 + rank * 1_000_000
            episode_seed += self.episode_index
            if self.episode_index >= 1_000_000:
                raise RuntimeError("Training seed stream exhausted; start another run seed.")
            self.episode_index += 1
            return self.env.reset(seed=episode_seed, options=options)

        def get_seed_state(self) -> dict[str, int]:
            return {"run_seed": run_seed, "rank": rank, "next_episode_index": self.episode_index}

        def set_seed_state(self, state: dict[str, int]) -> None:
            if state["run_seed"] != run_seed or state["rank"] != rank:
                raise ValueError("Checkpoint worker seed does not match this worker.")
            self.episode_index = int(state["next_episode_index"])

    run_seed = int(seed)
    env = ChronoSuspensionEnv(config=config, observation_mode=observation_mode, render_mode=None)
    return EpisodeSeedWrapper(Monitor(env, filename=monitor_path))


def make_vector_env(
    config: dict[str, Any],
    observation_mode: str = "privileged",
    seed: int = 42,
    n_envs: int = 4,
    monitor_dir: Path | None = None,
    normalization_path: Path | None = None,
) -> Any:
    """Create independent spawn workers and an observation normalizer.

    One environment uses DummyVecEnv to avoid needless IPC; two or more use
    SubprocVecEnv(start_method='spawn'), including on Linux for consistent rules.
    """
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(f"observation_mode must be one of {OBSERVATION_MODES}")
    if not 1 <= n_envs <= 32 or seed < 0:
        raise ValueError("n_envs must be 1..32 and seed must be nonnegative")
    limit_worker_threads()
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    factories = []
    for rank in range(n_envs):
        monitor_path = None
        if monitor_dir is not None:
            monitor_dir.mkdir(parents=True, exist_ok=True)
            # New shards preserve monitoring from previous resume/chunk sessions.
            monitor_path = str(monitor_dir / f"worker_{rank}_{uuid.uuid4().hex[:8]}")
        factories.append(partial(_build_worker, config, observation_mode, seed, rank, monitor_path))
    vector = (
        DummyVecEnv(factories) if n_envs == 1 else SubprocVecEnv(factories, start_method="spawn")
    )
    try:
        if normalization_path is not None:
            normalized = VecNormalize.load(str(normalization_path), vector)
            normalized.training = True
            normalized.norm_reward = False
            return normalized
        return VecNormalize(vector, norm_obs=True, norm_reward=False, clip_obs=10.0)
    except BaseException:
        vector.close()
        raise


def select_device(algorithm: str, requested: str = "auto") -> str:
    """PPO defaults to CPU; SAC selects CUDA only in the learner process."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    import torch

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if requested != "auto":
        return requested
    return "cuda" if algorithm == "sac" and torch.cuda.is_available() else "cpu"


def create_model(algorithm: str, vector: Any, seed: int, device: str, log_dir: Path) -> Any:
    """Fixed initial hyperparameters; the held-out test set never tunes them."""
    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.logger import configure

    torch.set_num_threads(1)
    shared = {"seed": seed, "device": device, "verbose": 0, "learning_rate": 3e-4, "gamma": 0.99}
    if algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            vector,
            n_steps=PPO_STEPS_PER_WORKER,
            batch_size=256,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
            **shared,
        )
    elif algorithm == "sac":
        model = SAC(
            "MlpPolicy",
            vector,
            buffer_size=300_000,
            batch_size=256,
            learning_starts=1024,
            train_freq=1,
            gradient_steps=-1,
            policy_kwargs={"net_arch": [128, 128]},
            **shared,
        )
    else:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    model.set_logger(configure(str(log_dir), ["csv", "tensorboard"]))
    return model


def _versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "python_full": sys.version,
        "platform": platform.platform(),
    }
    for name in ("numpy", "torch", "gymnasium", "stable-baselines3", "pychrono"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not available as Python distribution metadata"
    if versions["pychrono"] == "not available as Python distribution metadata":
        # Official conda PyChrono builds do not necessarily publish Python
        # distribution metadata. Read installed records, never package caches.
        for record_path in sorted((Path(sys.prefix) / "conda-meta").glob("pychrono-*.json")):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if record.get("name") != "pychrono":
                continue
            versions["pychrono"] = str(record.get("version", "unknown"))
            versions["pychrono_build"] = str(record.get("build", "unknown"))
            channel = str(record.get("channel", "")).strip("/")
            # Export only known public identifiers, not URLs, credential
            # tokens, private mirror paths or usernames from conda records.
            public_channel = "custom channel (redacted)"
            for identifier in ("projectchrono/label/release", "projectchrono", "conda-forge"):
                if channel in (identifier, f"https://conda.anaconda.org/{identifier}"):
                    public_channel = identifier
                    break
            versions["pychrono_channel"] = public_channel
            versions["pychrono_metadata_source"] = "installed conda-meta record"
            break
    return versions


def save_checkpoint(
    model: Any,
    vector: Any,
    config: dict[str, Any],
    run_dir: Path,
    algorithm: str,
    observation_mode: str,
    seed: int,
    complete_update: bool,
) -> Path:
    """Publish model, normalization, configuration and SAC replay as one directory."""
    parent = run_dir / "checkpoints"
    parent.mkdir(parents=True, exist_ok=True)
    suffix = uuid.uuid4().hex[:8]
    destination = parent / f"step_{model.num_timesteps:010d}_{suffix}"
    staging = parent / f".saving-{suffix}"
    staging.mkdir()
    model.save(str(staging / "model.zip"))
    vector.save(str(staging / "vecnormalize.pkl"))
    replay_reference: dict[str, str] = {}
    if algorithm == "sac":
        # Two atomically replaced banks retain the previous complete replay
        # while publishing a new one, without multiplying 300k-entry buffers
        # across all historical evaluation checkpoints.
        complete_pointer = run_dir / "latest_complete.json"
        prior_slot = 1
        if complete_pointer.exists():
            prior_slot = int(
                json.loads(complete_pointer.read_text(encoding="utf-8")).get("replay_slot", 1)
            )
        slot_name = f"slot_{1 - prior_slot}" if complete_update else "partial"
        replay_dir = run_dir / "replay"
        replay_dir.mkdir(exist_ok=True)
        model.save_replay_buffer(str(staging / "replay_buffer.pkl"))
        os.replace(staging / "replay_buffer.pkl", replay_dir / f"{slot_name}.pkl")
        write_json(replay_dir / f"{slot_name}.json", {"checkpoint_id": destination.name})
        replay_reference = {
            "replay_file": f"../../replay/{slot_name}.pkl",
            "replay_manifest": f"../../replay/{slot_name}.json",
        }
    serialized = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    metadata = {
        "algorithm": algorithm,
        "observation_mode": observation_mode,
        "seed": seed,
        "num_timesteps": int(model.num_timesteps),
        "n_envs": int(vector.num_envs),
        "device": str(model.device),
        "complete_update": complete_update,
        "config_sha256": hashlib.sha256(serialized).hexdigest(),
        "source_sha256_at_save": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "versions": _versions(),
        "worker_seed_states": vector.env_method("get_seed_state"),
        "created_unix_s": time.time(),
        "resume_scope": "policy/optimizer, normalizer, replay and next episode seeds; not bitwise",
        **replay_reference,
    }
    write_json(staging / "config.json", config)
    write_json(staging / "metadata.json", metadata)
    os.replace(staging, destination)
    write_json(run_dir / "latest.json", {"checkpoint": str(destination)})
    if complete_update:
        pointer = {"checkpoint": str(destination)}
        if algorithm == "sac":
            pointer["replay_slot"] = 1 - prior_slot
        write_json(run_dir / "latest_complete.json", pointer)
    return destination


def load_checkpoint(
    checkpoint: Path,
    run_dir: Path,
    requested_device: str = "auto",
    config: dict[str, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Resume into freshly initialized physics; never silently drop SAC replay."""
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.logger import configure

    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    saved_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config is not None and config != saved_config:
        raise ValueError("Resume configuration differs from checkpoint; start a separate run.")
    algorithm = metadata["algorithm"]
    if algorithm not in ALGORITHMS:
        raise ValueError("Checkpoint has an unsupported algorithm.")
    replay_path = checkpoint / metadata.get("replay_file", "replay_buffer.pkl")
    if algorithm == "sac":
        if not replay_path.is_file():
            raise FileNotFoundError(
                "SAC replay buffer missing: this checkpoint is evaluation-only."
            )
        if "replay_manifest" in metadata:
            replay_manifest = json.loads(
                (checkpoint / metadata["replay_manifest"]).read_text(encoding="utf-8")
            )
            if replay_manifest["checkpoint_id"] != checkpoint.name:
                raise ValueError(
                    "Historical checkpoint is evaluation-only; use latest_complete.json to resume SAC."
                )
    vector = make_vector_env(
        saved_config,
        metadata["observation_mode"],
        metadata["seed"],
        metadata["n_envs"],
        run_dir / "monitor",
        checkpoint / "vecnormalize.pkl",
    )
    try:
        for rank, state in enumerate(metadata["worker_seed_states"]):
            vector.env_method("set_seed_state", state, indices=rank)
        cls = PPO if algorithm == "ppo" else SAC
        model = cls.load(
            str(checkpoint / "model.zip"),
            env=vector,
            device=select_device(algorithm, requested_device),
            force_reset=True,
        )
        if algorithm == "sac":
            model.load_replay_buffer(str(replay_path))
        log_dir = run_dir / "logs" / f"resume_{model.num_timesteps}_{uuid.uuid4().hex[:8]}"
        model.set_logger(configure(str(log_dir), ["csv", "tensorboard"]))
        return model, vector, saved_config, metadata
    except BaseException:
        vector.close()
        raise


def load_policy(checkpoint: Path, env: Any, device: str = "cpu") -> Any:
    """Return ``policy(raw_observation) -> normalized four-wheel action``.

    The caller owns/reset/steps/closes ``env``.  The VecEnv here only binds the
    saved normalizer to the observation/action spaces; it is never stepped.
    Evaluation does not require a replay buffer.  Only load trusted local SB3
    checkpoints because model and normalization serialization use pickle.
    """
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata["algorithm"] not in ALGORITHMS:
        raise ValueError("Checkpoint has an unsupported algorithm.")
    cls = PPO if metadata["algorithm"] == "ppo" else SAC
    model = cls.load(str(checkpoint / "model.zip"), device=device)
    normalization = VecNormalize.load(
        str(checkpoint / "vecnormalize.pkl"), DummyVecEnv([lambda: env])
    )
    normalization.training = False
    normalization.norm_reward = False

    def predict(raw_observation: np.ndarray) -> np.ndarray:
        normalized = normalization.normalize_obs(np.asarray(raw_observation, dtype=np.float32))
        action, _ = model.predict(normalized, deterministic=True)
        return np.asarray(action, dtype=np.float32).reshape(4)

    return predict


def _training_callback(deadline: float | None, csv_path: Path) -> Any:
    from stable_baselines3.common.callbacks import BaseCallback

    class HealthCallback(BaseCallback):
        """Stop cooperatively on budget and record physical, unnormalized signals."""

        def __init__(self) -> None:
            super().__init__()
            self.budget_exhausted = False
            self.started = time.monotonic()
            self.stream: Any = None
            self.writer: Any = None

        def _on_training_start(self) -> None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            present = csv_path.exists() and csv_path.stat().st_size > 0
            self.stream = csv_path.open("a", newline="", encoding="utf-8")
            self.writer = csv.writer(self.stream)
            if not present:
                self.writer.writerow(
                    ["timesteps", "elapsed_s", "body_acc_mps2", "roll_rad", "pitch_rad", "reward"]
                )

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            if self.n_calls % 100 == 0 and infos:
                metrics = [info.get("metrics", {}) for info in infos]
                values = [
                    float(np.mean([m.get(name, 0.0) for m in metrics]))
                    for name in ("body_acc_mps2", "roll_rad", "pitch_rad")
                ]
                reward = float(np.mean(self.locals.get("rewards", [0.0])))
                self.writer.writerow(
                    [self.num_timesteps, time.monotonic() - self.started, *values, reward]
                )
                self.stream.flush()
                for name, value in zip(
                    ("body_acc_mps2", "roll_rad", "pitch_rad"), values, strict=True
                ):
                    self.logger.record(f"physics/{name}", value)
            self.budget_exhausted = deadline is not None and time.monotonic() >= deadline
            return not self.budget_exhausted

        def _on_training_end(self) -> None:
            if self.stream is not None:
                self.stream.close()

    return HealthCallback()


def training_quantum(n_envs: int) -> int:
    """A common number of aggregate transitions for exact PPO/SAC comparisons."""
    return PPO_STEPS_PER_WORKER * n_envs


def train_chunk(
    config: dict[str, Any],
    run_dir: Path,
    algorithm: str = "ppo",
    observation_mode: str = "privileged",
    seed: int = 42,
    n_envs: int = 4,
    target_timesteps: int = 65_536,
    checkpoint: Path | None = None,
    device: str = "auto",
    deadline: float | None = None,
) -> dict[str, Any]:
    """Train until an absolute aggregate step target, then checkpoint.

    Targets must be aligned to the PPO rollout quantum.  This avoids hidden
    PPO overshoot when the twelve-group experiment compares identical budgets.
    A deadline can interrupt a partial rollout; it is explicitly not marked as
    a completed training update and never used as a fair comparison checkpoint.
    """
    if algorithm not in ALGORITHMS or observation_mode not in OBSERVATION_MODES:
        raise ValueError("Unsupported algorithm or observation mode.")
    quantum = training_quantum(n_envs)
    if target_timesteps <= 0 or target_timesteps % quantum:
        raise ValueError(f"target_timesteps must be a positive multiple of {quantum}")
    if deadline is not None and time.monotonic() >= deadline:
        return {
            "completed": False,
            "reason": "budget_exhausted",
            "checkpoint": None,
            "elapsed_s": 0.0,
        }
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config)
    if checkpoint is None:
        vector = make_vector_env(config, observation_mode, seed, n_envs, run_dir / "monitor")
        try:
            model = create_model(
                algorithm,
                vector,
                seed,
                select_device(algorithm, device),
                run_dir / "logs" / "initial",
            )
        except BaseException:
            vector.close()
            raise
    else:
        model, vector, _, metadata = load_checkpoint(checkpoint, run_dir, device, config)
        expected = (algorithm, observation_mode, seed, n_envs)
        actual = tuple(metadata[key] for key in ("algorithm", "observation_mode", "seed", "n_envs"))
        if expected != actual:
            vector.close()
            raise ValueError(f"Resume run identity mismatch: {actual} != {expected}")
        if not metadata["complete_update"]:
            vector.close()
            raise ValueError("Resume from a completed chunk to preserve common rollout boundaries.")
    previous_steps = int(model.num_timesteps)
    callback = None
    try:
        if previous_steps > target_timesteps:
            raise ValueError("Absolute target is smaller than the checkpoint's existing steps.")
        if previous_steps == target_timesteps:
            return {
                "completed": True,
                "checkpoint": str(checkpoint),
                "num_timesteps": previous_steps,
                "elapsed_s": time.monotonic() - started,
                "new_timesteps": 0,
            }
        callback = _training_callback(deadline, run_dir / "physical_training.csv")
        model.learn(
            total_timesteps=target_timesteps - previous_steps,
            reset_num_timesteps=False,
            callback=callback,
            progress_bar=False,
        )
        completed = not callback.budget_exhausted and model.num_timesteps == target_timesteps
        saved = save_checkpoint(
            model, vector, config, run_dir, algorithm, observation_mode, seed, completed
        )
        result = {
            "completed": completed,
            "checkpoint": str(saved),
            "num_timesteps": int(model.num_timesteps),
            "new_timesteps": int(model.num_timesteps) - previous_steps,
            "elapsed_s": time.monotonic() - started,
            "reason": "target_reached" if completed else "budget_exhausted",
        }
        write_json(run_dir / "last_chunk.json", result)
        return result
    finally:
        try:
            if callback is not None and callback.stream is not None:
                callback.stream.close()
            model.logger.close()
        finally:
            vector.close()


def benchmark_workers(
    config: dict[str, Any],
    output_dir: Path,
    counts: tuple[int, ...] = (1, 2, 4, 8),
    vector_steps: int = 200,
    seed: int = 9001,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Measure headless stepping separately from startup and learner throughput."""
    if vector_steps <= 0 or not counts:
        raise ValueError("Provide worker counts and positive vector_steps.")
    rows: list[dict[str, Any]] = []
    for count in counts:
        if deadline is not None and time.monotonic() >= deadline:
            break
        vector = None
        try:
            started = time.monotonic()
            vector = make_vector_env(config, seed=seed, n_envs=count)
            observations = vector.reset()
            startup = time.monotonic() - started
            action = np.zeros((count, 4), dtype=np.float32)
            for _ in range(10):
                vector.step(action)
            started = time.monotonic()
            steps = 0
            for _ in range(vector_steps):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                observations, rewards, _, infos = vector.step(action)
                if not np.isfinite(observations).all() or not np.isfinite(rewards).all():
                    raise FloatingPointError("Nonfinite observation/reward in benchmark.")
                if len(infos) != count:
                    raise RuntimeError("Wrong worker result count.")
                steps += 1
            elapsed = time.monotonic() - started
            rows.append(
                {
                    "workers": count,
                    "startup_s": startup,
                    "elapsed_s": elapsed,
                    "transitions": steps * count,
                    "transitions_per_s": steps * count / elapsed if elapsed else 0.0,
                    "simulation_s_per_wall_s": steps
                    * count
                    * config["physics"]["control_dt"]
                    / elapsed,
                    "status": "ok" if steps == vector_steps else "budget_exhausted",
                }
            )
        except Exception as exc:
            rows.append(
                {"workers": count, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            )
        finally:
            if vector is not None:
                vector.close()
    good = [row for row in rows if row["status"] == "ok"]
    if not good:
        write_json(output_dir / "benchmark.json", {"results": rows, "recommended_workers": None})
        raise RuntimeError(
            "All worker benchmarks failed or exhausted budget; inspect benchmark.json."
        )
    fastest = max(row["transitions_per_s"] for row in good)
    # Prefer fewer processes if within 5% of peak rather than spend CPU for noise.
    selected = min(row["workers"] for row in good if row["transitions_per_s"] >= 0.95 * fastest)
    result = {
        "results": rows,
        "recommended_workers": selected,
        "measurement": "physics stepping only; no learner or rendering",
    }
    write_json(output_dir / "benchmark.json", result)
    return result


def align_steps(requested: int, n_envs: int, *, upward: bool = False) -> int:
    quantum = training_quantum(n_envs)
    operation = math.ceil if upward else math.floor
    return int(operation(requested / quantum)) * quantum
