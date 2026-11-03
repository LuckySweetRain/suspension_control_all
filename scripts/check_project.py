"""Run bounded numerical and training regressions without pytest or existing checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import deep_update, half_car_dimensions, load_config
from controllers.mpc import MPCController
from controllers.prior import make_prior_controller, prepare_policy_action
from envs import HalfCarEnv
from experiments.evaluation import (
    PolicySafetyController,
    ResidualController,
    evaluate_all,
    make_controller,
    rollout,
)
from rl.ppo import PPOAgent, PPOConfig, finish_ppo_trajectory
from rl.replay_buffer import ReplayBuffer
from roads.road_profiles import RoadProfileFactory, TableRoadProfile
from scripts.evaluate_all import resolve_checkpoints
from scripts.generate_road_dataset import generate_dataset
from scripts.train_rl import (
    AGENTS,
    collect_expert_buffer,
    evaluate_agent,
    set_seed,
    train_off_policy,
    train_ppo,
)
from scripts.validate_training_config import validate_config, validate_config_data
from visualization.spectra import one_sided_psd, sampling_frequency


def expect_error(error_type, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"{function.__name__} did not reject invalid input")


def check_configs(config: dict, out: Path) -> dict:
    reports = []
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        report = validate_config(path)
        assert report["passed"], report
        reports.append(report)
    (out / "configs.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    override = {"items": [1], "nested": {"values": [2]}}
    merged = deep_update({"nested": {"keep": True}}, override)
    merged["items"].append(3)
    merged["nested"]["values"].append(4)
    assert override == {"items": [1], "nested": {"values": [2]}}
    (out / "a.yaml").write_text("inherits: b.yaml\n", encoding="utf-8")
    (out / "b.yaml").write_text("inherits: a.yaml\n", encoding="utf-8")
    expect_error(ValueError, load_config, out / "a.yaml")
    (out / "invalid.yaml").write_text("- not-a-mapping\n", encoding="utf-8")
    assert not validate_config(out / "invalid.yaml")["passed"]
    for patch in (
        {"dt": float("nan")},
        {"control_dt": 0.0001},
        {"preview": {"steps": 1.5}},
        {"observation": {"history": {"enabled": True, "steps": 2}}},
        {"scenario_sampling": {"mode": "weighted", "weights": [-1.0]}},
        {"scenario_sampling": {"mode": "weighted", "weights": [float("nan")]}},
        {"preview_error": {"enabled": True}},
    ):
        assert not validate_config_data(deep_update(config, patch), check_roads=False)["passed"]
    for action_in_obs in (False, True):
        cfg = deep_update(config, {"action_in_observation": action_in_obs})
        env = HalfCarEnv(cfg)
        obs, _ = env.reset(seed=42)
        assert env.base_obs_dim == half_car_dimensions(cfg)[2]
        for config_cls, _ in AGENTS.values():
            assert config_cls.from_project_config(cfg).obs_dim == obs.size
        env.close()
    for value in ("sac=", "unknown=latest", "td3", "=latest"):
        expect_error(ValueError, resolve_checkpoints, value)
    return {"configurations": len(reports), "dimension_modes": 2}


def check_environment(config: dict, out: Path) -> dict:
    env = HalfCarEnv(config)
    try:
        check_env(env, skip_render_check=True)
        np.random.seed(731)
        global_state = np.random.get_state()
        obs, info = env.reset(seed=42)
        np.testing.assert_array_equal(np.random.get_state()[1], global_state[1])
        assert np.random.get_state()[2:] == global_state[2:]
        initial_safety = deepcopy(info["safety"])
        env.last_safety = {"unsafe": True, "violations": ["previous_episode"]}
        _, reset_info = env.reset(seed=42)
        assert reset_info["safety"] == initial_safety
        for action in ([1.0], [1.0, 2.0, 3.0], [float("nan"), 0], [float("inf"), 0]):
            expect_error(ValueError, env.step, action)
        assert env.step_count == 0
        info["road_preview"][:] = 999
        assert not np.any(env._info(0.0)["road_preview"] == 999)
        for name in ("PASSIVE", "PID", "SPDF", "MPC"):
            controller = make_controller(name, env, config)
            run = rollout(env, controller, use_preview=True)
            assert len(run["data"]) == env.max_steps
            assert np.all(np.isfinite(run["data"].select_dtypes(include=[np.number])))
    finally:
        env.close()
    limited = HalfCarEnv(deep_update(config, {"force_limit": 123.0}))
    try:
        limited.reset()
        _, _, _, _, info = limited.step([5000, -5000])
        np.testing.assert_array_equal(info["action"], [123, -123])
        assert info["derived"]["Uaf"] == 123 and info["derived"]["Uar"] == -123
    finally:
        limited.close()
    return {"gymnasium_checker": True, "classical_rollouts": 4}


def scalar_preview(road, t: float, speed: float, wheelbase: float, steps: int, dt: float):
    return np.asarray([road.pair(t + k * dt, speed, wheelbase)[::2] for k in range(1, steps + 1)])


def check_roads(config: dict, out: Path) -> dict:
    scenarios = load_config(ROOT / "configs/default.yaml")["scenarios"]
    times = np.linspace(0.0, 2.0, 601)
    for scenario in scenarios:
        road = RoadProfileFactory.create(scenario, duration=2.0, dt=0.001)
        np.testing.assert_allclose(
            road.values(times), [road.value(float(t)) for t in times], rtol=0, atol=1e-14
        )
        for t in (0.0, 0.59, 1.5, 7.0):
            np.testing.assert_allclose(
                road.preview(t, 20.0, 2.7, 450, 0.002),
                scalar_preview(road, t, 20.0, 2.7, 450, 0.002),
                rtol=0,
                atol=1e-14,
            )
        assert road.preview(0.0, 20.0, 2.7, 0, 0.01).shape == (0, 2)
    expect_error(ValueError, TableRoadProfile, np.array([0.0, 1.0]), np.array([0.0, np.nan]))
    expect_error(ValueError, RoadProfileFactory.create, {"type": "unknown"}, 1.0, 0.001)
    manifest = generate_dataset(out / "road_dataset", duration=0.1, dt=0.01)
    report = validate_config(Path(manifest["config"]))
    assert report["passed"], report
    road = RoadProfileFactory.create(scenarios[0], duration=2.0, dt=0.001)
    timings = {}
    for label, function in (
        ("scalar", lambda: scalar_preview(road, 0.1, 20.0, 2.7, 450, 0.002)),
        ("vectorized", lambda: road.preview(0.1, 20.0, 2.7, 450, 0.002)),
    ):
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(50):
                function()
            samples.append((time.perf_counter() - start) / 50)
        timings[label] = float(np.median(samples))
    return {"road_types_checked": len(scenarios), "seconds_per_450_step_preview": timings}


def check_mpc(config: dict, out: Path) -> dict:
    env = HalfCarEnv(config)
    try:
        env.reset()
        for preview_dt in (0.002, 0.007, 0.02):
            cfg = deep_update(config, {"preview": {"dt": preview_dt}, "mpc": {"horizon": 3}})
            controller = MPCController(env.model, cfg)
            sample_times = np.arange(1, 21) * preview_dt
            info = {
                "road": np.array([1, 2, 3, 4]),
                "road_preview": np.column_stack([1 + 2 * sample_times, 3 + 4 * sample_times]),
            }
            predicted = controller._preview_heights(info)
            t = np.arange(4) * config["control_dt"]
            np.testing.assert_allclose(
                predicted, np.column_stack([1 + 2 * t, 3 + 4 * t]), rtol=0, atol=1e-14
            )
        short_cfg = deep_update(config, {"mpc": {"horizon": 1, "replan_interval": 2}})
        controller = MPCController(env.model, short_cfg)
        obs, info = env.reset()
        for _ in range(3):
            obs, _, _, _, info = env.step(controller.compute_action(obs, info))
        assert np.all(np.isfinite(obs))
    finally:
        env.close()
    return {"preview_intervals_checked": 3, "single_step_horizon": True}


class FixedPolicy:
    def reset(self):
        pass

    def compute_action(self, obs, info):
        return np.array([1000.0, -800.0], dtype=np.float32)


def check_policy_pipeline(config: dict, out: Path) -> dict:
    for residual in (False, True):
        cfg = deep_update(
            config,
            {
                "residual_control": {
                    "enabled": residual,
                    "prior": "spdf",
                    "scale": 0.2,
                    "shield": {"enabled": True, "max_residual_fraction": 0.1},
                },
                "policy_action_parameterization": {"enabled": True, "max_delta_fraction": 0.04},
                "policy_improvement_gate": {"enabled": True, "max_scale": 0.3},
                "policy_safety": {"enabled": True, "max_delta_fraction": 0.01},
            },
        )
        training_env, eval_env = HalfCarEnv(cfg), HalfCarEnv(cfg)
        try:
            obs, info = training_env.reset()
            eval_obs, eval_info = eval_env.reset()
            prior = make_prior_controller("spdf", training_env, cfg) if residual else None
            controller = FixedPolicy()
            if residual:
                controller = ResidualController(
                    make_prior_controller("spdf", eval_env, cfg),
                    controller,
                    eval_env,
                    cfg["residual_control"],
                )
            wrapper = PolicySafetyController(
                controller,
                eval_env,
                cfg["policy_safety"],
                {} if residual else cfg["policy_action_parameterization"],
                {} if residual else cfg["policy_improvement_gate"],
            )
            wrapper.reset()
            if prior is not None:
                prior.reset()
            for _ in range(5):
                prepared = prepare_policy_action(
                    FixedPolicy().compute_action(obs, info), training_env, obs, info, cfg, prior
                )
                deployed = wrapper.compute_action(eval_obs, eval_info)
                np.testing.assert_allclose(prepared.action, deployed, rtol=0, atol=1e-5)
                obs, reward, _, _, info = training_env.step(prepared.step_action)
                eval_obs, eval_reward, _, _, eval_info = eval_env.step(deployed)
                np.testing.assert_allclose(obs, eval_obs, rtol=0, atol=1e-6)
                assert abs(reward - eval_reward) < 1e-6
        finally:
            training_env.close()
            eval_env.close()
    returns, advantages = finish_ppo_trajectory([1.0], [2.0], [False], 0.9, 0.95, last_value=3.0)
    np.testing.assert_allclose(returns, [3.7])
    np.testing.assert_allclose(advantages, [1.7])
    returns, _ = finish_ppo_trajectory([1.0], [2.0], [True], 0.9, 0.95, last_value=3.0)
    np.testing.assert_allclose(returns, [1.0])
    return {"action_modes_checked": 2, "termination_bootstrap": True}


def check_spectra(config: dict, out: Path) -> dict:
    for count in (128, 129):
        times = np.arange(count) * 0.01
        values = np.sin(2 * np.pi * 7 * np.arange(count) / count)
        freq, power = one_sided_psd(times, values)
        np.testing.assert_allclose(freq[np.argmax(power)], 700.0 / count)
        np.testing.assert_allclose(np.sum(power) * (freq[1] - freq[0]), np.var(values))
    expect_error(ValueError, sampling_frequency, np.array([0.0, 0.1, 0.11]))
    expect_error(ValueError, one_sided_psd, np.array([0.0]), np.array([1.0]))
    return {"frequency_peak": True, "variance_integral": True}


def check_training(config: dict, out: Path) -> dict:
    checkpoints = {}
    for algorithm, (config_cls, agent_cls) in AGENTS.items():
        set_seed(42)
        run_dir = out / algorithm
        run_dir.mkdir()
        cfg = config_cls.from_project_config(config)
        agent = agent_cls(cfg, torch.device("cpu"))
        if algorithm == "ppo":
            history, _ = train_ppo(agent, config, run_dir, 1)
        else:
            history, _ = train_off_policy(agent, cfg, config, algorithm, run_dir, 1)
        assert np.isfinite(history[0]["critic_loss"])
        assert np.isfinite(history[0]["actor_loss"])
        checkpoint = run_dir / "checkpoints/final.pt"
        agent.save(checkpoint)
        checkpoints[algorithm] = str(checkpoint)
        env = HalfCarEnv(config)
        obs, _ = env.reset()
        before = agent.select_action(obs)
        loaded = agent_cls(cfg, torch.device("cpu"))
        loaded.load(checkpoint)
        np.testing.assert_array_equal(loaded.select_action(obs), before)
        direct = evaluate_agent(loaded, config)["mean_return"]
        deployed = rollout(
            env,
            make_controller(algorithm, env, config, checkpoint=str(checkpoint)),
            use_preview=True,
        )["return"]
        np.testing.assert_allclose(direct, deployed, rtol=1e-5, atol=1e-5)
        env.close()
    metrics, _ = evaluate_all(config, checkpoints, out / "evaluation")
    assert len(metrics) == 8
    assert np.all(np.isfinite(metrics.select_dtypes(include=[np.number])))
    one_step = deep_update(config, {"episode_seconds": config["control_dt"]})
    agent = PPOAgent(PPOConfig.from_project_config(one_step), torch.device("cpu"))
    history, _ = train_ppo(agent, one_step, out / "ppo_single_step", 1)
    assert np.isfinite(history[0]["actor_loss"]) and np.isfinite(history[0]["critic_loss"])
    expert_cfg = deep_update(
        config, {"expert": {"enabled": True, "controllers": ["pid"], "episodes_per_scenario": 1}}
    )
    buffer = collect_expert_buffer(expert_cfg, PPOConfig.from_project_config(config), 64)
    assert len(buffer) == 10 and not buffer.done[: len(buffer)].any()
    config_cls, agent_cls = AGENTS["td3"]
    batch = buffer.sample_indices([0, 1, 2, 3], torch.device("cpu"))
    losses = []
    for penalty in (0.0, 2.0):
        set_seed(17)
        cfg = replace(config_cls.from_project_config(config), action_l2_weight=penalty)
        agent = agent_cls(cfg, torch.device("cpu"))
        losses.append(agent._update_actor(batch, expert_batch=batch, bc_weight=1.0))
    assert losses[1] > losses[0], "TD3 action regularization must remain active with BC"
    empty = ReplayBuffer(22, 2, 2)
    expect_error(ValueError, empty.sample, 1, torch.device("cpu"))
    return {"algorithms": list(AGENTS), "checkpoint_roundtrips": 4, "evaluation_rows": len(metrics)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/project_checks"))
    args = parser.parse_args()
    out = args.out / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = load_config(ROOT / "configs/smoke.yaml")
    checks = {}
    for function in (
        check_configs,
        check_environment,
        check_roads,
        check_mpc,
        check_policy_pipeline,
        check_spectra,
        check_training,
    ):
        try:
            checks[function.__name__] = {"passed": True, **function(config, out)}
        except Exception as exc:
            checks[function.__name__] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        print(
            f"{function.__name__}: {'PASS' if checks[function.__name__]['passed'] else checks[function.__name__]['error']}"
        )
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {
            name: version(name) for name in ("numpy", "torch", "scipy", "gymnasium", "ruff")
        },
        "cuda_available": torch.cuda.is_available(),
        "checks": checks,
        "passed": all(item["passed"] for item in checks.values()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {out / 'report.json'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
