from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from controllers import MPCController, PIDController, SPDFController
from controllers.rl_transformer import RLTransformerController
from controllers.prior import (
    parameterize_policy_action,
    policy_improvement_gate,
    residual_gate,
    shield_policy_action,
    shield_residual_action,
)
from envs import HalfCarEnv, MPCParameterPolicyEnv, MuJoCoFullCarEnv, MuJoCoHalfCarEnv, MuJoCoVehicleEnv
from experiments.evaluation import evaluate_all, make_controller, make_env
from scripts.benchmark_vector_env import benchmark_vector_env
from scripts.build_mpc_parameter_evidence_table import build_mpc_parameter_evidence_table
from scripts.build_si_rppo_strategy_evidence_table import build_si_rppo_strategy_evidence_table
from scripts.build_thesis_claim_matrix import build_markdown as build_thesis_claim_markdown
from scripts.build_thesis_chapter_draft import build_thesis_chapter_draft
from scripts.build_thesis_chapter_latex import build_thesis_chapter_latex
from scripts.build_thesis_completion_audit import build_completion_audit
from scripts.build_thesis_latex_snippets import LatexSnippet, build_thesis_latex_snippets
from scripts.build_thesis_objective_evidence import build_thesis_objective_evidence
from scripts.build_thesis_standalone_preview import build_thesis_standalone_preview
from scripts.check_thesis_claim_boundaries import check_thesis_claim_boundaries
from scripts.check_thesis_readiness import build_thesis_readiness_report
from scripts.export_thesis_package import export_thesis_package
from scripts.stage_thesis_template_import import stage_thesis_template_import
from scripts.verify_thesis_package import verify_thesis_package
from scripts.build_delta_ppo_evidence_table import build_delta_ppo_evidence_table
from scripts.run_si_rppo_preview_robustness import build_pairwise_delta_summary, build_preview_case_summary
from scripts.export_mujoco_env_spec import export_env_spec
from scripts.evaluate_all import resolve_checkpoints
from scripts.generate_road_dataset import generate_dataset
from scripts.import_road_directory import import_road_directory
from scripts.preflight_mujoco_training import preflight_mujoco_training
from scripts.check_gym_registration import check_registration
from scripts.collect_env_statistics import collect_env_statistics
from scripts.collect_expert_dataset import collect_expert_dataset
from scripts.run_mujoco_benchmark import run_benchmark
from scripts.run_mujoco_robustness_matrix import run_robustness_matrix
from scripts.run_mpc_parameter_learning_benchmark import run_mpc_parameter_learning_benchmark
from scripts.run_mpc_parameter_seed_sweep import run_mpc_parameter_seed_sweep
from scripts.run_projection_seed_sweep import run_projection_seed_sweep
from scripts.run_si_rppo_ablation import build_claim_report, run_si_rppo_ablation
from scripts.summarize_benchmark import build_report
from scripts.summarize_mpc_parameter_schedule import summarize_mpc_parameter_schedule
from scripts.train_rl import ScenarioSampler
from scripts.train_rl import pretrain_ppo_from_imitation
from scripts.train_mpc_parameter_neural_policy import train_mpc_parameter_neural_policy
from scripts.train_mpc_parameter_policy import train_mpc_parameter_policy
from scripts.tune_mpc_parameters import tune_mpc_parameters
from scripts.validate_training_config import validate_config
from scripts.validate_mujoco_env import validate_environment
from models import HalfCarModel, HalfCarParams
from rl.ddpg import DDPGConfig
from rl.mpc_parameter_policy import MPCParameterGaussianPolicy
from rl.networks import TransformerActor, TransformerGaussianActor, TransformerValue
from rl.ppo import PPOConfig, finish_ppo_trajectory
from rl.ppo import PPOAgent
from rl.replay_buffer import MixedReplaySampler, ReplayBuffer
from rl.sac import SACConfig
from rl.td3 import TD3Config
from roads.road_profiles import RoadProfileFactory


def cfg():
    c = load_config(ROOT / "configs" / "default.yaml")
    c["episode_seconds"] = 1.0
    c["mpc"]["max_iter"] = 2
    return c


def test_half_car_zero_equilibrium():
    model = HalfCarModel(HalfCarParams.from_seed(42))
    x = np.zeros(8)
    road = np.zeros(4)
    action = np.zeros(2)
    dx = model.derivative(x, action, road)
    assert np.allclose(dx, 0.0)


def test_half_car_parameters_match_spdf_constant_sim():
    p = HalfCarParams()
    assert p.mb == 1200.0
    assert p.Ip == 600.0
    assert p.mwf == 100.0
    assert p.mwr == 100.0
    assert p.kf1 == 15000.0
    assert p.kr1 == 15000.0
    assert p.knf1 == 1000.0
    assert p.knr1 == 1000.0
    assert p.be == 1500.0
    assert p.bc == 1200.0
    assert p.kf2 == 200000.0
    assert p.kr2 == 200000.0
    assert p.bf2 == 1500.0
    assert p.br2 == 2000.0
    assert p.a == 1.2
    assert p.b == 1.5
    assert p.fmax == 5000.0


def test_vehicle_param_overrides_are_exported():
    c = cfg()
    c["vehicle_params"] = {
        "mb": 1500.0,
        "Ip": 2500.0,
        "kf1": 28000.0,
        "kr1": 28000.0,
        "be": 1800.0,
        "bc": 1800.0,
        "a": 1.35,
        "b": 1.35,
    }
    env = HalfCarEnv(c, scenario=c["scenarios"][0], use_preview=False)
    assert env.params.mb == 1500.0
    assert env.params.mb_real == 1500.0
    assert env.params.Ip_real == 2500.0
    assert env.params.kf1_real == 28000.0
    assert env.params.kr1_real == 28000.0
    assert env.params.be_real == 1800.0
    assert env.params.bc_real == 1800.0
    assert env.params.a_real == 1.35
    assert env.params.b_real == 1.35


def test_road_preview_shape():
    c = cfg()
    scenario = c["scenarios"][0]
    road = RoadProfileFactory.create(scenario, duration=1.0, dt=c["dt"])
    preview = road.preview(0.0, scenario["speed"], c["wheelbase"], c["preview"]["steps"], c["control_dt"])
    assert preview.shape == (10, 2)
    assert np.all(np.isfinite(preview))


def test_extended_road_profiles_smoke():
    c = cfg()
    scenarios = [
        {"type": "sine", "speed": 10.0, "amplitude": 0.02, "wavelength": 3.0},
        {"type": "step", "height": 0.03, "start_time": 0.2},
        {"type": "pothole", "speed": 10.0, "depth": 0.04, "length": 0.5},
        {"type": "table", "times": [0.0, 0.5, 1.0], "heights": [0.0, 0.02, -0.01]},
        {
            "type": "composite",
            "speed": 10.0,
            "components": [
                {"type": "sine", "amplitude": 0.01, "wavelength": 2.0},
                {"type": "bump", "height": 0.02, "length": 0.3},
            ],
        },
    ]
    for scenario in scenarios:
        road = RoadProfileFactory.create(scenario, duration=1.0, dt=c["dt"])
        preview = road.preview(0.1, scenario.get("speed", 10.0), c["wheelbase"], 5, c["control_dt"])
        assert preview.shape == (5, 2)
        assert np.all(np.isfinite(preview))


def test_csv_road_profile_smoke(tmp_path):
    road_file = tmp_path / "road.csv"
    road_file.write_text("0.0,0.0\n0.5,0.02\n1.0,-0.01\n", encoding="utf-8")
    c = cfg()
    road = RoadProfileFactory.create({"type": "csv", "path": str(road_file)}, duration=1.0, dt=c["dt"])
    preview = road.preview(0.1, 10.0, c["wheelbase"], 5, c["control_dt"])
    assert preview.shape == (5, 2)
    assert np.all(np.isfinite(preview))


def test_pid_spdf_outputs_without_preview():
    c = cfg()
    env = HalfCarEnv(c, scenario=c["scenarios"][0], use_preview=False)
    obs, info = env.reset()
    for controller in (
        PIDController(env.params, env.control_dt, env.force_limit),
        SPDFController(env.params, env.control_dt),
    ):
        action = controller.compute_action(obs, info)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))
        assert np.max(np.abs(action)) <= env.force_limit


def test_mpc_uses_preview_shape():
    c = cfg()
    env = HalfCarEnv(c, scenario=c["scenarios"][0], use_preview=True)
    obs, info = env.reset()
    assert info["road_preview"].shape == (10, 2)
    action = MPCController(env.model, c).compute_action(obs, info)
    assert action.shape == (2,)
    assert np.all(np.isfinite(action))


def test_transformer_actor_output_shape():
    c = cfg()
    td3_cfg = TD3Config.from_project_config(c)
    actor = TransformerActor(td3_cfg)
    obs = torch.zeros(4, td3_cfg.obs_dim)
    action = actor(obs)
    assert action.shape == (4, 2)


def test_transformer_gaussian_actor_and_value_shapes():
    c = cfg()
    sac_cfg = SACConfig.from_project_config(c)
    actor = TransformerGaussianActor(sac_cfg)
    value = TransformerValue(PPOConfig.from_project_config(c))
    obs = torch.zeros(4, sac_cfg.obs_dim)
    action, logp = actor.sample(obs)
    assert action.shape == (4, 2)
    assert logp.shape == (4, 1)
    assert value(obs).shape == (4, 1)


def test_rl_controller_algorithms_smoke():
    c = cfg()
    env = HalfCarEnv(c, scenario=c["scenarios"][0], use_preview=True)
    obs, info = env.reset()
    for algorithm in ("td3", "ddpg", "sac", "ppo"):
        controller = RLTransformerController(c, algorithm=algorithm)
        action = controller.compute_action(obs, info)
        assert action.shape == (2,)
        assert np.all(np.isfinite(action))
        assert np.max(np.abs(action)) <= env.force_limit


def test_ddpg_config_matches_td3_observation_space():
    c = cfg()
    td3_cfg = TD3Config.from_project_config(c)
    ddpg_cfg = DDPGConfig.from_project_config(c)
    assert ddpg_cfg.obs_dim == td3_cfg.obs_dim
    assert ddpg_cfg.act_dim == td3_cfg.act_dim


def test_ppo_advantage_finish_shapes():
    returns, advantages = finish_ppo_trajectory(
        rewards=[1.0, 1.0],
        values=[0.5, 0.25],
        dones=[False, True],
        gamma=0.99,
        lam=0.95,
    )
    assert len(returns) == 2
    assert len(advantages) == 2
    assert np.all(np.isfinite(returns))
    assert np.all(np.isfinite(advantages))


def test_ppo_unsafe_penalty_accepts_trajectory_flags():
    c = cfg()
    c["rl"]["ppo"] = {
        "train_epochs": 1,
        "minibatch_size": 2,
        "projection_penalty_weight": 0.1,
        "unsafe_penalty_weight": 0.5,
        "action_delta_penalty_weight": 0.25,
        "feasibility_advantage_weight": 1.0,
        "feasibility_advantage_min": 0.1,
    }
    ppo_cfg = PPOConfig.from_project_config(c)
    agent = PPOAgent(ppo_cfg, torch.device("cpu"))
    trajectory = {
        "obs": np.zeros((2, ppo_cfg.obs_dim), dtype=np.float32),
        "act": np.zeros((2, ppo_cfg.act_dim), dtype=np.float32),
        "logp": [0.0, 0.0],
        "ret": [0.0, -1.0],
        "adv": [0.5, -0.5],
        "projection_error": [0.0, 0.2],
        "unsafe": [0.0, 1.0],
        "action_delta": [0.0, 0.1],
    }
    value_loss, actor_loss = agent.train_trajectory(trajectory)
    assert np.isfinite(value_loss)
    assert np.isfinite(actor_loss)
    assert ppo_cfg.unsafe_penalty_weight == 0.5
    assert ppo_cfg.action_delta_penalty_weight == 0.25
    assert ppo_cfg.feasibility_advantage_weight == 1.0


def test_ppo_adaptive_constraint_weights_update_from_trajectory():
    c = cfg()
    c["rl"]["ppo"] = {
        "train_epochs": 1,
        "minibatch_size": 2,
        "projection_penalty_weight": 0.1,
        "unsafe_penalty_weight": 0.2,
        "action_delta_penalty_weight": 0.3,
        "adaptive_constraint_weights": True,
        "constraint_weight_lr": 1.0,
        "projection_error_target": 0.1,
        "unsafe_fraction_target": 0.0,
        "action_delta_target": 0.05,
        "max_constraint_weight": 2.0,
        "feasibility_advantage_weight": 1.0,
        "feasibility_advantage_min": 0.1,
    }
    ppo_cfg = PPOConfig.from_project_config(c)
    agent = PPOAgent(ppo_cfg, torch.device("cpu"))
    trajectory = {
        "obs": np.zeros((2, ppo_cfg.obs_dim), dtype=np.float32),
        "act": np.zeros((2, ppo_cfg.act_dim), dtype=np.float32),
        "logp": [0.0, 0.0],
        "ret": [0.0, -1.0],
        "adv": [0.5, -0.5],
        "projection_error": [0.2, 0.2],
        "unsafe": [0.0, 1.0],
        "action_delta": [0.1, 0.1],
    }
    agent.train_trajectory(trajectory)
    weights = agent.constraint_weight_summary()
    assert weights["projection_penalty_weight"] > 0.1
    assert weights["unsafe_penalty_weight"] > 0.2
    assert weights["action_delta_penalty_weight"] > 0.3


def test_mujoco_half_car_env_render_smoke():
    c = cfg()
    c["episode_seconds"] = 0.05
    env = MuJoCoHalfCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, _ = env.reset()
    obs, reward, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    frame = env.render()
    env.close()
    assert obs.shape == (env.obs_dim,)
    assert np.isfinite(reward)
    assert not terminated
    assert frame.shape == (90, 160, 3)
    assert frame.dtype == np.uint8


def test_mujoco_vehicle_env_dynamics_smoke():
    c = cfg()
    c["episode_seconds"] = 0.05
    c["mujoco"] = {"settle_seconds": 0.3}
    env = MuJoCoVehicleEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset()
    assert obs.shape == (env.obs_dim,)
    assert info["engine"] == "mujoco"
    assert np.max(np.abs(obs[:12])) < 1e-6
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    frame = env.render()
    env.close()
    assert obs.shape == (env.obs_dim,)
    assert np.isfinite(reward)
    assert not terminated
    assert "derived" in info
    assert frame.shape == (90, 160, 3)
    assert frame.dtype == np.uint8


def test_mujoco_full_car_env_dynamics_smoke():
    c = load_config(ROOT / "configs" / "mujoco_full_car.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.3
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset()
    assert obs.shape == (env.obs_dim,)
    assert info["engine"] == "mujoco_full_car"
    assert info["road_corners"].shape == (8,)
    assert np.max(np.abs(obs[:14])) < 1e-6
    obs, reward, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    frame = env.render()
    env.close()
    assert obs.shape == (env.obs_dim,)
    assert np.isfinite(reward)
    assert not terminated
    assert "ddroll" in info["derived"]
    assert frame.shape == (90, 160, 3)
    assert frame.dtype == np.uint8


def test_gymnasium_registration_full_car_smoke():
    pytest = __import__("pytest")
    gym = pytest.importorskip("gymnasium")
    report = check_registration(config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml", smoke=False)
    assert report["gymnasium_available"]
    env = gym.make(
        "ActiveSuspensionMuJoCoFullCar-v0",
        config_path=str(ROOT / "configs" / "mujoco_full_car_corner.yaml"),
        scenario_index=0,
        width=160,
        height=90,
    )
    obs, info = env.reset(seed=11)
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    env.close()
    assert obs.shape == env.observation_space.shape
    assert info["engine"] == "mujoco_full_car"
    assert np.isfinite(reward)
    assert not terminated


def test_mujoco_full_car_corner_action_interface():
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.3
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset()
    sac_cfg = SACConfig.from_project_config(c)
    assert env.action_space.shape == (4,)
    assert env.preview_token_dim == 4
    assert info["road_preview"].shape == (env.preview_steps, 4)
    assert obs.shape == (sac_cfg.obs_dim,)
    assert sac_cfg.act_dim == 4
    action = np.asarray([100.0, -100.0, 50.0, -50.0], dtype=np.float32)
    obs, reward, terminated, _, info = env.step(action)
    env.close()
    assert obs.shape == (sac_cfg.obs_dim,)
    assert np.allclose(info["command_corner_action"], action)
    assert info["corner_action"].shape == (4,)
    assert np.isfinite(reward)
    assert not terminated
    assert "reward_components" in info
    assert "action_metrics" in info
    assert "action_delta_rms" in info["action_metrics"]


def test_mujoco_full_car_actuator_dynamics_lag_command():
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.05
    c["settle_seconds"] = 0.0
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    c["actuator"] = {"enabled": True, "time_constant": 0.05, "rate_limit": 2000.0}
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    env.reset(seed=3)
    command = np.asarray([1000.0, -1000.0, 500.0, -500.0], dtype=np.float32)
    _, _, _, _, info = env.step(command)
    env.close()
    actual = info["corner_action"]
    assert np.allclose(info["command_corner_action"], command)
    assert np.max(np.abs(actual)) < np.max(np.abs(command))
    assert info["actuator"]["enabled"]


def test_mujoco_full_car_configurable_safety_limits_terminate():
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.05
    c["settle_seconds"] = 0.0
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    c["safety_limits"] = {
        "max_suspension_travel": 0.25,
        "max_pitch": -1.0,
        "max_roll": 0.45,
        "max_wheel_displacement": 1.0,
        "terminate_on_unsafe": True,
        "unsafe_penalty": 10.0,
    }
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    env.reset(seed=5)
    _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
    env.close()
    assert terminated
    assert info["unsafe"]
    assert "pitch" in info["safety"]["violations"]
    assert info["safety"]["limits"]["terminate_on_unsafe"]


def test_mujoco_full_car_domain_randomization_changes_reset_params():
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    _, info1 = env.reset(seed=123)
    _, info2 = env.reset()
    env.close()
    dr1 = info1["domain_randomization"]
    dr2 = info2["domain_randomization"]
    assert dr1["enabled"]
    changed = any(
        not np.isclose(dr1[name], dr2[name])
        for name in ("speed", "road_scale", "mb", "Ip", "roll_inertia")
    )
    assert changed


def test_mujoco_full_car_preview_error_model_changes_preview():
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    c["preview_error"] = {
        "enabled": True,
        "delay_steps": 0,
        "height_noise_std": 0.0,
        "bias_std": 0.01,
        "dropout_prob": 0.0,
        "scale_error_std": 0.0,
    }
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset(seed=7)
    env.close()
    preview = info["road_preview"]
    clean = info["road_preview_clean"]
    assert obs.shape == (env.obs_dim,)
    assert preview.shape == clean.shape == (env.preview_steps, 4)
    assert info["preview_error"]["enabled"]
    assert not np.allclose(preview, clean)


def test_export_mujoco_env_spec_contains_reproducibility_metadata(tmp_path):
    spec = export_env_spec(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_path=tmp_path / "env_spec.json",
    )
    meta = spec["environment_metadata"]
    assert (tmp_path / "env_spec.json").is_file()
    assert meta["engine"] == "mujoco_full_car"
    assert meta["act_dim"] == 4
    assert meta["preview_token_dim"] == 4
    assert meta["full_car"]["action_mode"] == "corner"
    assert "mb_real" in meta["vehicle_params"]
    assert meta["actuator"]["enabled"]
    assert meta["preview_error"]["enabled"]


def test_mujoco_full_car_evaluation_smoke(tmp_path):
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["scenarios"] = [c["scenarios"][0]]
    c["evaluation"] = {"controllers": ["PASSIVE"]}
    metrics, trajectories = evaluate_all(c, checkpoints=None, result_dir=tmp_path)
    manifest = json.loads((tmp_path / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "evaluation_manifest.json").is_file()
    assert len(metrics) == 1
    assert "RollAccRMS_radps2" in metrics.columns
    assert "MaxSusp_fl_m" in metrics.columns
    assert "PASSIVE_class_b_correlated" in trajectories
    assert manifest["environment_metadata"][0]["engine"] == "mujoco_full_car"


def test_road_dataset_and_benchmark_report_smoke(tmp_path):
    dataset_dir = tmp_path / "dataset"
    manifest = generate_dataset(dataset_dir, duration=0.2, dt=0.01, base_config=ROOT / "configs" / "mujoco_full_car_corner.yaml")
    assert len(manifest["roads"]) >= 3
    assert (dataset_dir / "road_dataset_manifest.json").is_file()
    config = load_config(dataset_dir / "mujoco_full_car_dataset.yaml")
    config["episode_seconds"] = 0.05
    config["mujoco"]["settle_seconds"] = 0.1
    config["scenarios"] = config["scenarios"][:1]
    eval_dir = tmp_path / "eval"
    evaluate_all(config, checkpoints=None, result_dir=eval_dir)
    report = build_report(eval_dir)
    text = report.read_text(encoding="utf-8")
    assert "MuJoCo Suspension Benchmark Report" in text
    assert "Mean Metrics By Controller" in text


def test_import_road_directory_smoke(tmp_path):
    road_dir = tmp_path / "source_roads"
    road_dir.mkdir()
    (road_dir / "road_a.csv").write_text("time,height\n0.0,0.0\n0.1,0.01\n0.2,0.0\n", encoding="utf-8")
    (road_dir / "road_b.csv").write_text("time,height\n0.0,0.0\n0.1,-0.02\n0.2,0.0\n", encoding="utf-8")
    out_dir = tmp_path / "imported"
    manifest = import_road_directory(
        road_dir=road_dir,
        out_dir=out_dir,
        base_config=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        speed=12.0,
    )
    config_path = Path(manifest["config"])
    assert config_path.is_file()
    assert (out_dir / "imported_road_manifest.json").is_file()
    assert len(manifest["roads"]) == 2
    config = load_config(config_path)
    assert len(config["scenarios"]) == 2
    assert config["scenarios"][0]["type"] == "csv"
    assert config["scenarios"][0]["speed"] == 12.0


def test_mujoco_benchmark_runner_passive_smoke(tmp_path):
    out_dir = tmp_path / "benchmark"
    manifest = run_benchmark(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_dir=out_dir,
        algorithms=[],
        episodes=0,
        generate_roads=True,
        road_dataset_dir=tmp_path / "roads",
        road_duration=0.2,
        road_dt=0.01,
    )
    assert (out_dir / "benchmark_manifest.json").is_file()
    assert Path(manifest["benchmark_report"]).is_file()
    assert (out_dir / "evaluation" / "metrics.csv").is_file()
    assert manifest["metric_rows"] >= 1


def test_validate_mujoco_env_smoke(tmp_path):
    report = validate_environment(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_path=tmp_path / "validation.json",
        max_steps=3,
        action_mode="passive",
        render=False,
    )
    assert report["passed"]
    assert (tmp_path / "validation.json").is_file()
    assert report["total_steps"] >= 3
    assert report["steps_per_second"] > 0.0


def test_validate_training_config_static_checks(tmp_path):
    report = validate_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    assert report["passed"]
    assert report["expected_act_dim"] == 4
    assert report["expected_preview_token_dim"] == 4
    bad = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    bad["rl"]["act_dim"] = 2
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(__import__("yaml").safe_dump(bad, sort_keys=False), encoding="utf-8")
    bad_report = validate_config(bad_path, check_roads=False)
    assert not bad_report["passed"]
    assert any("rl.act_dim" in error for error in bad_report["errors"])


def test_vector_env_benchmark_smoke(tmp_path):
    report = benchmark_vector_env(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_path=tmp_path / "vector.json",
        num_envs=2,
        steps=2,
        action_mode="passive",
    )
    assert report["passed"]
    assert report["total_transitions"] == 4
    assert report["transitions_per_second"] > 0.0
    assert (tmp_path / "vector.json").is_file()


def test_collect_env_statistics_smoke(tmp_path):
    report = collect_env_statistics(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_path=tmp_path / "stats.json",
        episodes=1,
        max_steps=3,
        action_mode="passive",
    )
    assert report["total_steps"] == 3
    assert report["observation"]["count"] == 3
    assert len(report["observation"]["mean"]) == 54
    assert report["reward"]["count"] == 3
    assert "action_metrics" in report
    assert "reward_components" in report
    assert "saturation_ratio" in report["action_metrics"]
    assert (tmp_path / "stats.json").is_file()


def test_mujoco_full_car_history_observation_config():
    c = load_config(ROOT / "configs" / "mujoco_full_car_history.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset(seed=9)
    env.close()
    assert env.history_steps == 5
    assert env.obs_base_dim == 124
    assert obs.shape == (env.obs_dim,)
    assert info["engine"] == "mujoco_full_car"
    assert info["base_obs"].shape == (124,)
    assert SACConfig.from_project_config(c).obs_dim == env.obs_dim


def test_reduced_full_car_preview_controller_smoke():
    c = load_config(ROOT / "configs" / "mujoco_full_car_residual.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset(seed=10)
    controller = make_controller("FULL_CAR_MPC_LITE", env, c)
    action = controller.compute_action(obs, info)
    env.close()
    assert action.shape == env.action_space.shape
    assert np.all(np.isfinite(action))
    assert np.max(np.abs(action)) <= env.force_limit


def test_residual_gate_and_deviation_reward_smoke():
    c = load_config(ROOT / "configs" / "mujoco_full_car_residual.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    c["reward"]["deviation"] = 1.0
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True, width=160, height=90)
    obs, info = env.reset(seed=12)
    cfg = c["residual_control"]
    gate_nominal = residual_gate(info, cfg)
    noisy_info = dict(info)
    noisy_info["preview_error"] = {
        "delay_steps": 3,
        "height_noise_std": 0.02,
        "bias_std": 0.01,
        "dropout_prob": 0.2,
        "scale_error_std": 0.2,
    }
    gate_noisy = residual_gate(noisy_info, cfg)
    near_limit_info = dict(info)
    near_limit_info["safety"] = dict(info["safety"])
    near_limit_info["safety"]["max_abs_suspension_travel"] = 0.99 * near_limit_info["safety"]["limits"]["max_suspension_travel"]
    shielded = shield_residual_action(np.full(env.action_space.shape, env.force_limit, dtype=np.float32), env, near_limit_info, cfg)
    policy_limited = shield_policy_action(
        np.full(env.action_space.shape, env.force_limit, dtype=np.float32),
        env,
        info,
        {"enabled": False, "max_action_fraction": 0.75, "max_delta_fraction": 0.08},
    )
    policy_blocked = shield_policy_action(
        np.full(env.action_space.shape, env.force_limit, dtype=np.float32),
        env,
        near_limit_info,
        {"enabled": True, "max_action_fraction": 0.75, "hard_margin": 0.05, "soft_margin": 0.18},
    )
    prior = np.zeros(env.action_space.shape, dtype=np.float32)
    action = np.full(env.action_space.shape, 100.0, dtype=np.float32)
    _, _, _, _, no_prior_info = env.step(action)
    assert not no_prior_info["has_prior_action"]
    assert no_prior_info["action_metrics"]["deviation_rms"] == 0.0
    assert no_prior_info["reward_components"]["deviation"] == 0.0
    _, reward, _, _, info = env.step({"action": action, "prior_action": prior})
    env.close()
    assert gate_noisy < gate_nominal
    assert np.max(np.abs(shielded)) < 1e-3
    assert np.max(np.abs(policy_limited - env.last_action)) <= 0.08 * env.force_limit + 1e-5
    assert np.max(np.abs(policy_blocked)) < 1e-3
    assert info["has_prior_action"]
    assert np.allclose(info["prior_action"], prior)
    assert info["action_metrics"]["deviation_rms"] > 0.0
    assert info["reward_components"]["deviation"] > 0.0
    assert np.isfinite(reward)


def test_delta_policy_action_parameterization_limits_step_change():
    c = load_config(ROOT / "configs" / "mujoco_full_car_residual.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True)
    _, info = env.reset(seed=15)
    env.last_action = np.asarray([100.0, -100.0, 50.0, -50.0], dtype=np.float32)
    raw = np.ones(env.action_space.shape, dtype=np.float32) * env.force_limit
    parameterized = parameterize_policy_action(
        raw,
        env,
        info,
        {
            "enabled": True,
            "mode": "delta",
            "max_delta_fraction": 0.05,
            "max_action_fraction": 0.75,
        },
    )
    last_action = np.asarray(env.last_action, dtype=np.float32)
    env.close()
    assert parameterized.shape == env.action_space.shape
    assert np.max(np.abs(parameterized - last_action)) <= 0.05 * env.force_limit + 1e-5
    assert np.max(np.abs(parameterized)) <= 0.75 * env.force_limit + 1e-5


def test_policy_improvement_gate_scales_passive_deviation():
    c = load_config(ROOT / "configs" / "mujoco_full_car_residual.yaml")
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    env = MuJoCoFullCarEnv(c, scenario=c["scenarios"][0], use_preview=True)
    _, info = env.reset(seed=16)
    action = np.ones(env.action_space.shape, dtype=np.float32) * 1000.0
    quiet_info = dict(info)
    quiet_info["road_preview_clean"] = np.zeros_like(info["road_preview_clean"])
    gated_quiet, quiet_scale = policy_improvement_gate(
        action,
        env,
        quiet_info,
        {"enabled": True, "preview_rms_threshold": 0.1, "preview_rms_softness": 0.1, "accel_threshold": 100.0},
    )
    rough_info = dict(info)
    rough_info["road_preview_clean"] = np.ones_like(info["road_preview_clean"]) * env.road_scale
    gated_rough, rough_scale = policy_improvement_gate(
        action,
        env,
        rough_info,
        {"enabled": True, "preview_rms_threshold": 0.1, "preview_rms_softness": 0.1, "accel_threshold": 100.0},
    )
    env.close()
    assert quiet_scale < rough_scale
    assert np.max(np.abs(gated_quiet)) < np.max(np.abs(gated_rough))


def test_collect_expert_dataset_and_ppo_bc_smoke(tmp_path):
    cfg_path = ROOT / "configs" / "mujoco_full_car_residual.yaml"
    out_path = tmp_path / "expert.npz"
    manifest = collect_expert_dataset(
        config_path=cfg_path,
        out_path=out_path,
        expert="FULL_CAR_MPC_LITE",
        residual_prior="FULL_CAR_MPC_LITE",
        episodes=1,
        max_steps=3,
        seed=21,
    )
    assert out_path.is_file()
    assert Path(str(out_path).replace(".npz", ".manifest.json")).is_file()
    assert manifest["transitions"] == 3
    assert manifest["raw_transitions"] == 3
    assert not manifest["skip_unsafe"]
    assert manifest["skipped_unsafe"] >= 0
    assert 0.0 <= manifest["raw_unsafe_fraction"] <= 1.0
    data = np.load(out_path, allow_pickle=True)
    assert data["obs"].shape[0] == 3
    assert data["act"].shape == data["residual_act"].shape
    assert np.max(np.abs(data["residual_act"])) < 1e-5

    c = load_config(cfg_path)
    c["episode_seconds"] = 0.05
    c["mujoco"]["settle_seconds"] = 0.2
    c["domain_randomization"]["enabled"] = False
    c["imitation"] = {
        "enabled": True,
        "dataset": str(out_path),
        "epochs": 1,
        "batch_size": 2,
        "residual_targets": True,
    }
    agent = PPOAgent(PPOConfig.from_project_config(c), torch.device("cpu"))
    losses = pretrain_ppo_from_imitation(agent, c, tmp_path)
    assert len(losses) == 1
    assert np.isfinite(losses[0])
    assert agent.bc_anchor_obs is None
    assert (tmp_path / "imitation_pretrain.json").is_file()

    c["imitation"]["anchor_enabled"] = True
    c["imitation"]["anchor_max_samples"] = 2
    anchored = PPOAgent(PPOConfig.from_project_config(c), torch.device("cpu"))
    pretrain_ppo_from_imitation(anchored, c, tmp_path)
    assert anchored.bc_anchor_obs is not None
    assert anchored.bc_anchor_obs.shape[0] == 2


def test_si_rppo_ablation_dry_run_writes_variants(tmp_path):
    manifest = run_si_rppo_ablation(
        base_config_path=ROOT / "configs" / "mujoco_full_car_safe_ppo.yaml",
        out_dir=tmp_path / "ablation",
        episodes=1,
        expert_episodes=1,
        expert_max_steps=2,
        expert_controller="PASSIVE",
        skip_unsafe_expert=True,
        train_scenario_limit=1,
        eval_scenario_limit=1,
        episode_seconds=0.05,
        mujoco_settle_seconds=0.2,
        baseline_algorithms=["td3", "sac"],
        dry_run=True,
    )
    assert manifest["dry_run"]
    assert Path(manifest["materialized_base_config"]).is_file()
    materialized = load_config(Path(manifest["materialized_base_config"]))
    assert materialized["episode_seconds"] == 0.05
    assert materialized["mujoco"]["settle_seconds"] == 0.2
    assert len(materialized["scenarios"]) == 1
    assert manifest["expert_manifest"]["planned"]
    assert manifest["expert_controller"] == "PASSIVE"
    assert manifest["skip_unsafe_expert"]
    assert manifest["expert_manifest"]["expert"] == "PASSIVE"
    assert manifest["expert_manifest"]["skip_unsafe"]
    assert set(manifest["variants"]) == {
        "ppo_scratch",
        "bc_ppo",
        "residual_bc_ppo",
        "safe_residual_bc_ppo",
    }
    for report in manifest["variants"].values():
        config_path = Path(report["config"])
        assert config_path.is_file()
        assert report["planned_train_command"]
    residual_cfg = load_config(Path(manifest["variants"]["residual_bc_ppo"]["config"]))
    safe_cfg = load_config(Path(manifest["variants"]["safe_residual_bc_ppo"]["config"]))
    bc_cfg = load_config(Path(manifest["variants"]["bc_ppo"]["config"]))
    scratch_cfg = load_config(Path(manifest["variants"]["ppo_scratch"]["config"]))
    assert scratch_cfg["rl"]["ppo"]["projection_penalty_weight"] == 0.0
    assert scratch_cfg["rl"]["ppo"]["unsafe_penalty_weight"] == 0.0
    assert scratch_cfg["rl"]["ppo"]["action_delta_penalty_weight"] == 0.0
    assert not scratch_cfg["rl"]["ppo"]["adaptive_constraint_weights"]
    assert scratch_cfg["rl"]["ppo"]["feasibility_advantage_weight"] == 0.0
    assert not scratch_cfg["policy_action_parameterization"]["enabled"]
    assert not scratch_cfg["policy_improvement_gate"]["enabled"]
    assert bc_cfg["rl"]["ppo"]["projection_penalty_weight"] > 0.0
    assert bc_cfg["rl"]["ppo"]["unsafe_penalty_weight"] > 0.0
    assert bc_cfg["rl"]["ppo"]["action_delta_penalty_weight"] > 0.0
    assert bc_cfg["rl"]["ppo"]["adaptive_constraint_weights"]
    assert "feasibility_advantage_weight" in bc_cfg["rl"]["ppo"]
    assert bc_cfg["policy_action_parameterization"]["enabled"]
    assert bc_cfg["policy_improvement_gate"]["enabled"]
    assert not residual_cfg["imitation"]["anchor_enabled"]
    assert not residual_cfg["policy_action_parameterization"]["enabled"]
    assert not residual_cfg["policy_improvement_gate"]["enabled"]
    assert not residual_cfg["residual_control"]["shield"]["enabled"]
    assert not safe_cfg["imitation"]["anchor_enabled"]
    assert not safe_cfg["policy_action_parameterization"]["enabled"]
    assert not safe_cfg["policy_improvement_gate"]["enabled"]
    assert safe_cfg["residual_control"]["shield"]["enabled"]
    assert bc_cfg["imitation"]["anchor_enabled"]
    assert set(manifest["off_policy_baselines"]) == {"td3_baseline", "sac_baseline"}
    for report in manifest["off_policy_baselines"].values():
        config_path = Path(report["config"])
        assert config_path.is_file()
        assert report["planned_train_command"]
        baseline_cfg = load_config(config_path)
        assert not baseline_cfg["policy_action_parameterization"]["enabled"]
        assert not baseline_cfg["policy_improvement_gate"]["enabled"]
    assert Path(manifest["combined_metrics"]).is_file()
    assert manifest["claim_report"]["status"] == "missing_data"
    assert manifest["claim_report"]["core_status"] == "missing_data"
    assert Path(manifest["claim_report"]["json"]).is_file()
    assert Path(manifest["claim_report"]["markdown"]).is_file()


def test_si_rppo_claim_report_detects_supported_ablation(tmp_path):
    rows = [
        {
            "Variant": "ppo_scratch",
            "Controller": "PPO",
            "Scenario": "road_a",
            "EpisodeReturn": -100.0,
            "UnsafeSteps": 2,
            "ActuatorSaturationRatio": 0.2,
            "ActionDeltaRMS_N": 50.0,
            "BodyAccRMS_mps2": 1.2,
            "PitchAccRMS_radps2": 0.3,
            "RollAccRMS_radps2": 0.4,
            "ActionDeviationRMS_N": 20.0,
        },
        {
            "Variant": "bc_ppo",
            "Controller": "PPO",
            "Scenario": "road_a",
            "EpisodeReturn": -80.0,
            "UnsafeSteps": 1,
            "ActuatorSaturationRatio": 0.1,
            "ActionDeltaRMS_N": 40.0,
            "BodyAccRMS_mps2": 1.0,
            "PitchAccRMS_radps2": 0.25,
            "RollAccRMS_radps2": 0.35,
            "ActionDeviationRMS_N": 18.0,
        },
        {
            "Variant": "residual_bc_ppo",
            "Controller": "PPO",
            "Scenario": "road_a",
            "EpisodeReturn": -70.0,
            "UnsafeSteps": 1,
            "ActuatorSaturationRatio": 0.1,
            "ActionDeltaRMS_N": 42.0,
            "BodyAccRMS_mps2": 0.9,
            "PitchAccRMS_radps2": 0.22,
            "RollAccRMS_radps2": 0.32,
            "ActionDeviationRMS_N": 16.0,
        },
        {
            "Variant": "safe_residual_bc_ppo",
            "Controller": "PPO",
            "Scenario": "road_a",
            "EpisodeReturn": -68.0,
            "UnsafeSteps": 0,
            "ActuatorSaturationRatio": 0.02,
            "ActionDeltaRMS_N": 30.0,
            "BodyAccRMS_mps2": 0.92,
            "PitchAccRMS_radps2": 0.23,
            "RollAccRMS_radps2": 0.33,
            "ActionDeviationRMS_N": 8.0,
        },
        {
            "Variant": "td3_baseline",
            "Controller": "TD3",
            "Scenario": "road_a",
            "EpisodeReturn": -90.0,
            "UnsafeSteps": 2,
            "ActuatorSaturationRatio": 0.12,
            "ActionDeltaRMS_N": 60.0,
            "BodyAccRMS_mps2": 1.1,
            "PitchAccRMS_radps2": 0.3,
            "RollAccRMS_radps2": 0.4,
            "ActionDeviationRMS_N": 0.0,
        },
        {
            "Variant": "sac_baseline",
            "Controller": "SAC",
            "Scenario": "road_a",
            "EpisodeReturn": -88.0,
            "UnsafeSteps": 2,
            "ActuatorSaturationRatio": 0.1,
            "ActionDeltaRMS_N": 55.0,
            "BodyAccRMS_mps2": 1.05,
            "PitchAccRMS_radps2": 0.28,
            "RollAccRMS_radps2": 0.38,
            "ActionDeviationRMS_N": 0.0,
        },
    ]
    report = build_claim_report(pd.DataFrame(rows), tmp_path)
    assert report["status"] == "ready"
    statuses = {item["name"]: item["status"] for item in report["comparisons"]}
    assert statuses["projection_aware_imitation"] == "supported"
    assert statuses["imitation_initialization"] == "supported"
    assert statuses["residual_prior_structure"] == "supported"
    assert statuses["safe_residual_gate"] == "supported"
    assert statuses["projection_aware_ppo_vs_td3"] == "supported"
    assert statuses["projection_aware_ppo_vs_sac"] == "supported"
    assert statuses["safe_residual_ppo_vs_td3"] == "supported"
    assert statuses["safe_residual_ppo_vs_sac"] == "supported"
    assert Path(report["json_path"]).is_file()
    assert Path(report["markdown_path"]).is_file()


def test_projection_seed_sweep_dry_run(tmp_path):
    summary = run_projection_seed_sweep(
        base_config_path=ROOT / "configs" / "mujoco_full_car_safe_ppo.yaml",
        out_dir=tmp_path / "seed_sweep",
        seeds=[42, 43],
        episodes=1,
        expert_episodes=1,
        expert_max_steps=2,
        train_scenario_limit=1,
        eval_scenario_limit=1,
        episode_seconds=0.05,
        mujoco_settle_seconds=0.2,
        dry_run=True,
    )
    assert summary["dry_run"]
    assert summary["seed_count"] == 2
    assert summary["supported_seeds"] == 0
    assert Path(summary["json_path"]).is_file()
    assert Path(summary["csv_path"]).is_file()
    assert Path(summary["markdown_path"]).is_file()
    for seed in (42, 43):
        assert (tmp_path / "seed_sweep" / f"seed_{seed}" / "seed_config.yaml").is_file()
        assert (tmp_path / "seed_sweep" / f"seed_{seed}" / "ablation" / "si_rppo_ablation_manifest.json").is_file()


def test_delta_ppo_evidence_table_builder(tmp_path):
    seed_dir = tmp_path / "seed_42"
    seed_dir.mkdir()
    (seed_dir / "projection_seed_sweep_summary.json").write_text(
        json.dumps(
            {
                "seeds": [
                    {
                        "seed": 42,
                        "core_status": "supported",
                        "delta_EpisodeReturn": 1.0,
                        "delta_UnsafeSteps": -1.0,
                        "delta_BodyAccRMS_mps2": 0.1,
                        "delta_PitchAccRMS_radps2": -0.2,
                        "delta_RollAccRMS_radps2": -0.3,
                        "delta_ActionDeltaRMS_N": -4.0,
                        "delta_ActuatorTrackingRMS_N": -5.0,
                        "delta_PolicyProjectionError": -0.6,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    pd.DataFrame(
        [
            {"Variant": "ppo_scratch", "Controller": "PPO", "EpisodeReturn": -2.0, "UnsafeSteps": 1},
            {"Variant": "bc_ppo", "Controller": "PPO", "EpisodeReturn": -1.0, "UnsafeSteps": 0},
        ]
    ).to_csv(matrix_dir / "combined_metrics.csv", index=False)
    summary = build_delta_ppo_evidence_table([seed_dir], matrix_dir, tmp_path / "evidence")
    assert summary["supported_seeds"] == 1
    assert Path(summary["seed_csv"]).is_file()
    assert Path(summary["matrix_csv"]).is_file()
    assert Path(summary["markdown"]).is_file()


def test_preflight_mujoco_training_smoke(tmp_path):
    manifest = preflight_mujoco_training(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_dir=tmp_path / "preflight",
        validation_steps=2,
        vector_envs=1,
        vector_steps=2,
        statistics_episodes=1,
        statistics_steps=2,
        action_mode="passive",
    )
    assert manifest["passed"]
    assert Path(manifest["manifest"]).is_file()
    for artifact in manifest["artifacts"].values():
        assert Path(artifact).is_file()


def test_tune_mpc_parameters_smoke(tmp_path):
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.03
    c["mujoco"]["settle_seconds"] = 0.05
    c["scenarios"] = [c["scenarios"][0]]
    cfg_path = tmp_path / "mpc_tune.yaml"
    cfg_path.write_text(__import__("yaml").safe_dump(c, sort_keys=False), encoding="utf-8")
    manifest = tune_mpc_parameters(
        config_path=cfg_path,
        out_dir=tmp_path / "tune",
        trials=2,
        scenario_limit=1,
        scale_low=0.9,
        scale_high=1.1,
    )
    assert Path(manifest["manifest"]).is_file()
    assert Path(manifest["trials_csv"]).is_file()
    assert Path(manifest["best_config"]).is_file()
    assert np.isfinite(manifest["best_score"])
    assert "k_preview" in manifest["best_gains"]


def test_mpc_parameter_policy_env_and_cem_smoke(tmp_path):
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.03
    c["mujoco"]["settle_seconds"] = 0.05
    multi_scenario = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    limited_env = MPCParameterPolicyEnv(config=multi_scenario, scenario_limit=1)
    limited_names = [limited_env.reset(seed=i)[1]["active_scenario"] for i in range(3)]
    limited_env.close()
    assert limited_names == [multi_scenario["scenarios"][0]["name"]] * 3

    c["scenarios"] = [c["scenarios"][0]]
    env = MPCParameterPolicyEnv(config=c, scenario_limit=1, scale_low=0.9, scale_high=1.1)
    obs, info = env.reset(seed=1)
    obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    env.close()
    assert obs.shape == env.observation_space.shape
    assert obs.shape[0] == 17
    assert terminated and not truncated
    assert np.isfinite(reward)
    assert "k_preview" in info["last_gains"]
    assert info["active_scenario"] == c["scenarios"][0]["name"]
    assert len(info["scenario_features"]) == 7

    cfg_path = tmp_path / "mpc_param_policy.yaml"
    cfg_path.write_text(__import__("yaml").safe_dump(c, sort_keys=False), encoding="utf-8")
    manifest = train_mpc_parameter_policy(
        config_path=cfg_path,
        out_dir=tmp_path / "mpc_policy",
        iterations=1,
        population=2,
        scenario_limit=1,
        scale_low=0.9,
        scale_high=1.1,
    )
    assert Path(manifest["manifest"]).is_file()
    assert Path(manifest["history_csv"]).is_file()
    assert Path(manifest["best_config"]).is_file()
    assert manifest["method"] == "cem_parameter_policy"
    history = pd.read_csv(manifest["history_csv"])
    assert "scenario" in history.columns


def test_mpc_parameter_neural_policy_smoke(tmp_path):
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.03
    c["mujoco"]["settle_seconds"] = 0.05
    c["scenarios"] = c["scenarios"][:2]
    env = MPCParameterPolicyEnv(config=c, scenario_limit=1, scale_low=0.9, scale_high=1.1)
    obs, _ = env.reset(seed=7)
    policy = MPCParameterGaussianPolicy(obs_dim=obs.shape[0], act_dim=env.action_space.shape[0], hidden_sizes=[16])
    action, log_prob, entropy = policy.sample(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    deterministic_action, _, _ = policy.sample(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=True)
    env.close()
    assert action.shape[-1] == env.action_space.shape[0]
    assert torch.max(torch.abs(deterministic_action)).item() < 1e-6
    assert log_prob.shape == (1, 1)
    assert entropy.shape == (1, 1)

    cfg_path = tmp_path / "mpc_param_neural_policy.yaml"
    cfg_path.write_text(__import__("yaml").safe_dump(c, sort_keys=False), encoding="utf-8")
    manifest = train_mpc_parameter_neural_policy(
        config_path=cfg_path,
        out_dir=tmp_path / "mpc_neural_policy",
        episodes=2,
        batch_size=1,
        hidden_sizes=[16],
        anchor_coef=0.2,
        anchor_final_coef=0.05,
        elite_bc_coef=0.1,
        scenario_limit=1,
        scale_low=0.9,
        scale_high=1.1,
        baseline_penalty_weight=1.0,
        unsafe_penalty_weight=2.0,
        suspension_penalty_weight=1.0,
        actuator_penalty_weight=0.1,
        eval_scenario_offset=1,
        eval_scenario_limit=1,
        deployment_gate=True,
        deployment_min_train_improvement=0.0,
        device="cpu",
    )
    assert Path(manifest["manifest"]).is_file()
    assert Path(manifest["history_csv"]).is_file()
    assert Path(manifest["checkpoint"]).is_file()
    assert Path(manifest["best_config"]).is_file()
    assert Path(manifest["final_policy_actions_csv"]).is_file()
    assert manifest["method"] == "neural_reinforce_mpc_parameter_policy"
    assert manifest["constrained_training"]
    assert manifest["anchor_coef"] == 0.2
    assert manifest["anchor_final_coef"] == 0.05
    assert manifest["anchor_schedule"] == "linear"
    assert manifest["elite_bc_coef"] == 0.1
    assert manifest["deployment_gate_enabled"]
    assert "deployment_gate_active" in manifest
    assert "train_policy_delta_vs_fixed" in manifest
    assert "raw_final_policy_score" in manifest
    assert manifest["eval_scenarios"] == [c["scenarios"][1]["name"]]
    assert np.isfinite(manifest["best_score"])
    assert np.isfinite(manifest["final_policy_score"])
    history = pd.read_csv(manifest["history_csv"])
    assert {
        "episode",
        "scenario",
        "loss",
        "score",
        "raw_reward",
        "constraint_penalty",
        "anchor_loss",
        "anchor_coef",
        "elite_bc_loss",
        "elite_bc_coef",
    }.issubset(history.columns)
    assert np.isclose(history["anchor_coef"].iloc[0], 0.2)
    assert np.isclose(history["anchor_coef"].iloc[-1], 0.05)
    final_actions = pd.read_csv(manifest["final_policy_actions_csv"])
    assert {"scenario", "score", "action_k_preview", "k_preview"}.issubset(final_actions.columns)
    assert final_actions["scenario"].tolist() == [c["scenarios"][1]["name"]]

    control_env = make_env(c, scenario=c["scenarios"][1], use_preview=True)
    try:
        controller = make_controller("BP_NMPS", control_env, c, checkpoint=manifest["checkpoint"])
        obs, info = control_env.reset(options={"scenario": c["scenarios"][1]})
        action = controller.compute_action(obs, info)
        assert action.shape == control_env.action_space.shape
        assert "k_preview" in controller.last_gains
    finally:
        control_env.close()

    eval_dir = tmp_path / "bp_nmps_eval"
    c["evaluation"] = {"controllers": ["BP_NMPS"]}
    metrics, _ = evaluate_all(c, checkpoints={"bp_nmps": manifest["checkpoint"]}, result_dir=eval_dir)
    assert (eval_dir / "metrics.csv").is_file()
    assert (eval_dir / "evaluation_manifest.json").is_file()
    assert set(metrics["Controller"]) == {"BP_NMPS"}
    trajectory = pd.read_csv(eval_dir / f"BP_NMPS_{c['scenarios'][0]['name']}.csv")
    assert {"mpc_param_action_0", "mpc_param_gain_k_preview"}.issubset(trajectory.columns)
    schedule = summarize_mpc_parameter_schedule(eval_dir)
    assert Path(schedule["summary_csv"]).is_file()
    assert Path(schedule["report"]).is_file()
    assert Path(schedule["figure"]).is_file()
    assert Path(schedule["delta_pct_figure"]).is_file()
    assert "k_preview" in schedule["gain_columns"]
    assert "delta_pct_k_preview" in schedule["delta_pct_columns"]


def test_evaluate_all_cli_resolves_bp_nmps_checkpoint(tmp_path):
    checkpoint = tmp_path / "mpc_parameter_neural_policy.pt"
    checkpoint.write_bytes(b"placeholder")
    resolved = resolve_checkpoints(f"bp_nmps={checkpoint}")
    assert resolved == {"bp_nmps": str(checkpoint)}


def test_mpc_parameter_learning_benchmark_smoke(tmp_path):
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.03
    c["mujoco"]["settle_seconds"] = 0.05
    c["scenarios"] = c["scenarios"][:2]
    cfg_path = tmp_path / "mpc_param_benchmark.yaml"
    cfg_path.write_text(__import__("yaml").safe_dump(c, sort_keys=False), encoding="utf-8")
    manifest = run_mpc_parameter_learning_benchmark(
        config_path=cfg_path,
        out_dir=tmp_path / "mpc_param_benchmark",
        scenario_limit=1,
        tuning_trials=1,
        cem_iterations=1,
        cem_population=1,
        neural_episodes=1,
        neural_batch_size=1,
        neural_anchor_coef=0.2,
        neural_anchor_final_coef=0.05,
        neural_elite_bc_coef=0.1,
        neural_deployment_gate=True,
        neural_deployment_min_train_improvement=0.0,
        scale_low=0.9,
        scale_high=1.1,
        neural_baseline_penalty_weight=1.0,
        neural_unsafe_penalty_weight=2.0,
        neural_suspension_penalty_weight=1.0,
        neural_actuator_penalty_weight=0.1,
        eval_scenario_offset=1,
        eval_scenario_limit=1,
        device="cpu",
    )
    assert Path(manifest["manifest"]).is_file()
    assert Path(manifest["comparison_csv"]).is_file()
    assert Path(manifest["report"]).is_file()
    assert manifest["eval_scenarios"] == [c["scenarios"][1]["name"]]
    assert manifest["neural_constraint_weights"]["baseline_penalty_weight"] == 1.0
    assert manifest["neural_constraint_weights"]["anchor_coef"] == 0.2
    assert manifest["neural_constraint_weights"]["anchor_final_coef"] == 0.05
    assert manifest["neural_constraint_weights"]["elite_bc_coef"] == 0.1
    assert manifest["neural_constraint_weights"]["deployment_gate"]
    rows = pd.read_csv(manifest["comparison_csv"])
    assert set(rows["variant"]) == {
        "fixed_mpc_lite",
        "random_gain_search",
        "cem_parameter_policy",
        "neural_parameter_policy",
    }
    assert rows["best_score"].notna().all()
    assert {"score_delta_vs_fixed", "BodyAccRMS_mps2", "UnsafeSteps", "deployment_gate_active"}.issubset(rows.columns)
    neural_row = rows[rows["variant"] == "neural_parameter_policy"].iloc[0]
    assert np.isfinite(neural_row["policy_score"])
    assert np.isfinite(neural_row["policy_delta_vs_fixed"])


def test_mpc_parameter_seed_sweep_smoke(tmp_path):
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.03
    c["mujoco"]["settle_seconds"] = 0.05
    c["scenarios"] = c["scenarios"][:2]
    cfg_path = tmp_path / "mpc_param_seed_sweep.yaml"
    cfg_path.write_text(__import__("yaml").safe_dump(c, sort_keys=False), encoding="utf-8")
    summary = run_mpc_parameter_seed_sweep(
        config_path=cfg_path,
        out_dir=tmp_path / "mpc_param_seed_sweep",
        seeds=[1],
        scenario_limit=1,
        tuning_trials=1,
        cem_iterations=1,
        cem_population=1,
        neural_episodes=1,
        neural_batch_size=1,
        scale_low=0.9,
        scale_high=1.1,
        eval_scenario_offset=1,
        eval_scenario_limit=1,
        min_support_rate=0.0,
        device="cpu",
    )
    assert Path(summary["json_path"]).is_file()
    assert Path(summary["csv_path"]).is_file()
    assert Path(summary["markdown_path"]).is_file()
    assert Path(summary["latex_table"]).is_file()
    assert Path(summary["delta_figure"]).is_file()
    assert summary["seed_count"] == 1
    assert summary["train_scenarios"] == [c["scenarios"][0]["name"]]
    assert summary["eval_scenarios"] == [c["scenarios"][1]["name"]]
    assert "support_rate" in summary
    assert summary["status"] in {"supported", "needs_more_evidence"}
    assert summary["noninferiority_status"] in {"supported", "needs_more_evidence"}
    assert summary["improvement_status"] in {"supported", "needs_more_evidence"}
    assert summary["improvement_status"] == "needs_more_evidence"
    assert "mean_policy_improvement" in summary["improvement_failed_gates"]
    assert "claim_thresholds" in summary
    assert "noninferiority_failed_gates" in summary
    assert "improvement_failed_gates" in summary
    rows = pd.read_csv(summary["csv_path"])
    assert {"seed", "fixed_score", "neural_policy_score", "neural_policy_supported"}.issubset(rows.columns)
    latex = Path(summary["latex_table"]).read_text(encoding="utf-8")
    assert "BP-NMPS score" in latex
    assert "Score delta" in latex
    assert "Gate" in latex


def test_build_mpc_parameter_evidence_table(tmp_path):
    summary_dir = tmp_path / "seed_sweep"
    summary_dir.mkdir()
    summary = {
        "seed_count": 2,
        "support_rate": 1.0,
        "mean_policy_delta_vs_fixed": -0.2,
        "mean_unsafe_delta_vs_fixed": 0.0,
        "noninferiority_status": "supported",
        "improvement_status": "supported",
        "train_scenarios": ["train_a"],
        "eval_scenarios": ["eval_b"],
        "seeds": [
            {
                "fixed_score": 1.0,
                "neural_raw_policy_score": 0.8,
                "neural_policy_score": 0.8,
                "deployment_gate_active": False,
            },
            {
                "fixed_score": 1.0,
                "neural_raw_policy_score": 1.2,
                "neural_policy_score": 1.0,
                "deployment_gate_active": True,
            },
        ],
    }
    (summary_dir / "mpc_parameter_seed_sweep_summary.json").write_text(
        __import__("json").dumps(summary),
        encoding="utf-8",
    )
    summary["improvement_status"] = "needs_more_evidence"
    (summary_dir / "mpc_parameter_seed_sweep_summary.json").write_text(
        __import__("json").dumps(summary),
        encoding="utf-8",
    )
    manifest = build_mpc_parameter_evidence_table([summary_dir], tmp_path / "evidence", labels=["gate_probe"])
    assert Path(manifest["csv"]).is_file()
    assert Path(manifest["markdown"]).is_file()
    assert Path(manifest["latex"]).is_file()
    rows = pd.read_csv(manifest["csv"])
    assert rows.loc[0, "label"] == "gate_probe"
    assert np.isclose(rows.loc[0, "gate_rate"], 0.5)
    assert np.isclose(rows.loc[0, "mean_raw_policy_delta_vs_fixed"], 0.0)
    latex = Path(manifest["latex"]).read_text(encoding="utf-8")
    assert "BP-NMPS held-out evidence" in latex
    assert "gate\\_probe" in latex
    assert "needs\\_more\\_evidence" in latex


def test_thesis_claim_matrix_markdown():
    bp_nmps = {
        "seed_count": 5,
        "support_rate": 1.0,
        "mean_policy_delta_vs_fixed": -0.0013,
        "mean_unsafe_delta_vs_fixed": 0.0,
        "noninferiority_status": "supported",
        "improvement_status": "supported",
        "seeds": [{"deployment_gate_active": False}, {"deployment_gate_active": True}],
    }
    si_rppo = {
        "status": "ready",
        "core_status": "supported",
        "comparisons": [
            {
                "name": "projection_aware_imitation",
                "status": "supported",
                "metrics": {
                    "EpisodeReturn": {"delta": 10.0},
                    "UnsafeSteps": {"delta": -2.0},
                    "ActionDeltaRMS_N": {"delta": -3.0},
                    "PolicyProjectionError": {"delta": -0.4},
                },
            }
        ],
    }
    delta_ppo = {"seed_count": 3, "supported_seeds": 3}
    markdown = build_thesis_claim_markdown(bp_nmps, si_rppo, delta_ppo)
    assert "Thesis Claim Matrix" in markdown
    assert "BP-NMPS MPC parameter learning | supported" in markdown
    assert "gate rate 0.500" in markdown
    assert "BP-NMPS schedule evidence" in markdown
    assert "BP-NMPS preview-robustness evidence" in markdown
    assert "BP-NMPS preview-robustness LaTeX table" in markdown
    assert "BP-NMPS preview-robustness Chinese summary" in markdown
    assert "Safe-teacher gated SI-RPPO policy network | supported" in markdown
    assert "Delta PPO repeated-seed support: `3/3`" in markdown
    assert "SI-RPPO gate evidence" in markdown
    assert "SI-RPPO gate LaTeX table" in markdown
    assert "SI-RPPO method summary" in markdown
    assert "SI-RPPO innovation framing" in markdown
    assert "SI-RPPO preview-robustness evidence" in markdown
    assert "SI-RPPO preview-robustness LaTeX table" in markdown
    assert "SI-RPPO strategy-network evidence" in markdown
    assert "SI-RPPO strategy-network LaTeX table" in markdown
    assert "Thesis method/experiment draft" in markdown
    assert "Thesis chapter draft" in markdown
    assert "Thesis chapter LaTeX draft" in markdown
    assert "Thesis standalone LaTeX preview" in markdown
    assert "Thesis standalone preview PDF" in markdown
    assert "Direct road-preview control plan" in markdown
    assert "Thesis workflow index" in markdown
    assert "Thesis evidence package" in markdown
    assert "Thesis LaTeX snippets" in markdown
    assert "Thesis LaTeX integration guide" in markdown
    assert "Thesis defense Q&A" in markdown
    assert "Thesis final submission checklist" in markdown
    assert "Thesis readiness report" in markdown
    assert "Thesis export package" in markdown
    assert "Thesis completion audit" in markdown


def test_build_thesis_latex_snippets(tmp_path):
    source = tmp_path / "table.tex"
    source.write_text(
        (
            "\\begin{table}[t]\n"
            "\\caption{Demo}\n"
            "\\label{tab:demo}\n"
            "\\begin{tabular}{lr}\n"
            "Name & Value \\\\\n"
            "A & 1 \\\\\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "snippets.tex"
    built = build_thesis_latex_snippets(
        snippets=(
            LatexSnippet(
                title="Demo table",
                path=source,
                note="Demo note.",
            ),
        ),
        out_path=out_path,
    )
    text = built.read_text(encoding="utf-8")
    assert "Required packages" in text
    assert "\\usepackage{booktabs,graphicx}" in text
    assert "\\resizebox{\\textwidth}{!}{%" in text
    assert "Demo table" in text
    assert "\\label{tab:demo}" in text
    assert "Source:" in text


def test_si_rppo_preview_summary_groups_cases():
    rows = pd.DataFrame(
        [
            {"PreviewCase": "clean", "Variant": "bc_ppo", "Controller": "PPO", "EpisodeReturn": 1.0, "UnsafeSteps": 0.0},
            {"PreviewCase": "clean", "Variant": "bc_ppo", "Controller": "PPO", "EpisodeReturn": 3.0, "UnsafeSteps": 2.0},
            {"PreviewCase": "strong", "Variant": "ppo_scratch", "Controller": "PPO", "EpisodeReturn": -1.0, "UnsafeSteps": 4.0},
        ]
    )
    summary = build_preview_case_summary(rows)
    clean = summary[(summary["PreviewCase"] == "clean") & (summary["Variant"] == "bc_ppo")].iloc[0]
    assert clean["EpisodeReturn"] == 2.0
    assert clean["UnsafeSteps"] == 1.0


def test_si_rppo_preview_pairwise_delta_summary():
    summary = pd.DataFrame(
        [
            {"PreviewCase": "clean", "Variant": "bc_ppo", "Controller": "PPO", "EpisodeReturn": 3.0, "UnsafeSteps": 0.0},
            {"PreviewCase": "clean", "Variant": "ppo_scratch", "Controller": "PPO", "EpisodeReturn": 1.0, "UnsafeSteps": 2.0},
            {"PreviewCase": "strong", "Variant": "bc_ppo", "Controller": "PPO", "EpisodeReturn": 2.0, "UnsafeSteps": 0.0},
            {"PreviewCase": "strong", "Variant": "ppo_scratch", "Controller": "PPO", "EpisodeReturn": -1.0, "UnsafeSteps": 3.0},
        ]
    )
    pairwise = build_pairwise_delta_summary(summary)
    row = pairwise[pairwise["Comparison"] == "SI-RPPO vs PPO scratch"].iloc[0]
    assert row["EpisodeReturnDelta"] == 2.5
    assert row["UnsafeStepsDelta"] == -2.5


def test_build_si_rppo_strategy_evidence_table(tmp_path):
    claim = {
        "comparisons": [
            {
                "name": "projection_aware_imitation",
                "status": "supported",
                "metrics": {
                    "EpisodeReturn": {"delta": 10.0},
                    "UnsafeSteps": {"delta": -2.0},
                    "ActionDeltaRMS_N": {"delta": -3.0},
                    "PolicyProjectionError": {"delta": -0.4},
                },
            },
            {
                "name": "imitation_initialization",
                "status": "supported",
                "metrics": {
                    "EpisodeReturn": {"delta": 10.0},
                    "ActionDeltaRMS_N": {"delta": -3.0},
                },
            },
            {
                "name": "residual_prior_structure",
                "status": "weak_or_contradicted",
                "metrics": {
                    "EpisodeReturn": {"delta": -1.0},
                    "UnsafeSteps": {"delta": 1.0},
                },
            },
        ]
    }
    claim_path = tmp_path / "claim.json"
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    gate_path = tmp_path / "gate.csv"
    pd.DataFrame(
        [
            {"PolicyImprovementGate": 0.0, "GateActiveRatioGt0p1": 0.0, "GateMax": 0.0},
            {"PolicyImprovementGate": 0.8, "GateActiveRatioGt0p1": 1.0, "GateMax": 1.0},
        ]
    ).to_csv(gate_path, index=False)
    pairwise_path = tmp_path / "pairwise.csv"
    pd.DataFrame(
        [
            {
                "Comparison": "SI-RPPO vs PPO scratch",
                "EpisodeReturnDelta": 5.0,
                "UnsafeStepsDelta": -2.0,
                "ActionDeltaRMS_NDelta": -10.0,
                "PolicyProjectionErrorDelta": -0.3,
            }
        ]
    ).to_csv(pairwise_path, index=False)
    manifest = build_si_rppo_strategy_evidence_table(claim_path, gate_path, pairwise_path, tmp_path / "out")
    assert Path(manifest["csv"]).is_file()
    assert Path(manifest["latex"]).is_file()
    text = Path(manifest["markdown"]).read_text(encoding="utf-8")
    assert "Projection-aware imitation PPO" in text
    assert "Safe-teacher improvement gate" in text
    assert "SI-RPPO vs PPO scratch" in text


def test_build_thesis_completion_audit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("controllers").mkdir()
    Path("tests").mkdir()
    Path("tests/test_core.py").write_text("ok", encoding="utf-8")
    Path("docs").mkdir()
    Path("docs/thesis_claim_matrix.md").write_text("ok", encoding="utf-8")
    Path("docs/thesis_chapter_draft_cn.tex").write_text("ok", encoding="utf-8")
    Path("docs/thesis_latex_snippets.tex").write_text("ok", encoding="utf-8")
    Path("docs/thesis_latex_integration_guide.md").write_text("ok", encoding="utf-8")
    Path("docs/thesis_final_submission_checklist.md").write_text("ok", encoding="utf-8")
    Path("docs/thesis_readiness_report.md").write_text("ok", encoding="utf-8")
    Path("docs/thesis_standalone_preview.tex").write_text("ok", encoding="utf-8")
    Path("docs/latex_preview").mkdir()
    Path("docs/latex_preview/thesis_standalone_preview.pdf").write_bytes(b"%PDF")
    bp_dir = Path("results/mpc_parameter_seed_sweep_formal_gate")
    bp_dir.mkdir(parents=True)
    (bp_dir / "mpc_parameter_seed_sweep_summary.json").write_text(
        json.dumps(
            {
                "noninferiority_status": "supported",
                "improvement_status": "supported",
                "support_rate": 1.0,
                "mean_policy_delta_vs_fixed": -0.1,
                "mean_unsafe_delta_vs_fixed": 0.0,
            }
        ),
        encoding="utf-8",
    )
    for path in [
        "results/bp_nmps_schedule_evidence/bp_nmps_schedule_evidence.md",
        "results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.md",
        "results/bp_nmps_evidence_table/bp_nmps_evidence_table.md",
        "results/si_rppo_gate_evidence/si_rppo_gate_evidence.md",
        "results/si_rppo_preview_robustness/si_rppo_preview_robustness.md",
        "results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.md",
    ]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok", encoding="utf-8")
    si_dir = Path("results/si_rppo_e20_improvement_gate_fair_baselines")
    si_dir.mkdir(parents=True)
    (si_dir / "si_rppo_claim_report.json").write_text(
        json.dumps({"status": "ready", "core_status": "supported"}),
        encoding="utf-8",
    )
    (Path("results/si_rppo_strategy_evidence") / "si_rppo_strategy_evidence_table_manifest.json").write_text(
        json.dumps({"row_count": 7, "markdown": "results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.md"}),
        encoding="utf-8",
    )
    audit = build_completion_audit(out_path=Path("docs/audit.md"))
    assert audit["overall_status"] == "evidence_ready_with_remaining_thesis_work"
    text = Path(audit["out"]).read_text(encoding="utf-8")
    assert "BP-NMPS" in text
    assert "SI-RPPO" in text
    assert "Thesis integration artifacts ready: `True`" in text
    assert "\\usepackage{booktabs,graphicx}" in text
    assert "thesis_final_submission_checklist.md" in text
    assert "check_thesis_readiness.py" in text


def test_build_thesis_objective_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for path in [
        "controllers/__init__.py",
        "tests/test_core.py",
        "docs/thesis_claim_matrix.md",
        "docs/thesis_chapter_draft_cn.tex",
        "docs/thesis_latex_snippets.tex",
        "docs/si_rppo_ccfa_innovation_framing.md",
        "results/bp_nmps_schedule_evidence/bp_nmps_schedule_evidence.md",
        "results/bp_nmps_preview_robustness/bp_nmps_preview_robustness.md",
        "results/bp_nmps_evidence_table/bp_nmps_evidence_table.md",
        "results/si_rppo_gate_evidence/si_rppo_gate_evidence.md",
        "results/si_rppo_preview_robustness/si_rppo_preview_robustness.md",
        "results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table.md",
    ]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok", encoding="utf-8")
    bp_path = Path("results/mpc_parameter_seed_sweep_formal_gate/mpc_parameter_seed_sweep_summary.json")
    bp_path.parent.mkdir(parents=True)
    bp_path.write_text(
        json.dumps(
            {
                "noninferiority_status": "supported",
                "improvement_status": "supported",
                "support_rate": 1.0,
                "mean_policy_delta_vs_fixed": -0.001,
                "mean_unsafe_delta_vs_fixed": 0.0,
            }
        ),
        encoding="utf-8",
    )
    si_path = Path("results/si_rppo_e20_improvement_gate_fair_baselines/si_rppo_claim_report.json")
    si_path.parent.mkdir(parents=True)
    si_path.write_text(json.dumps({"status": "ready", "core_status": "supported"}), encoding="utf-8")
    strategy_path = Path("results/si_rppo_strategy_evidence/si_rppo_strategy_evidence_table_manifest.json")
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(json.dumps({"row_count": 7}), encoding="utf-8")
    report = build_thesis_objective_evidence(out_dir=Path("docs"))
    assert report["status"] == "objective_evidence_ready"
    assert report["ready_count"] == 3
    assert Path(report["json"]).is_file()
    assert Path(report["markdown"]).is_file()
    text = Path(report["markdown"]).read_text(encoding="utf-8")
    assert "SPDF方法，已经完成" in text
    assert "BP-NMPS is a conservative RL-assisted MPC parameter scheduler" in text
    assert "SI-RPPO can be framed as a deployability-oriented policy-network contribution" in text


def test_check_thesis_claim_boundaries_allows_guarded_mentions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    safe_text = (
        "本文以已感知路面不平度预瞄为输入。\n"
        "本文不将道路视觉重建或 PreScan 路面轮廓提取作为核心研究内容。\n"
        "Boundary: Claim conservative adaptive MPC parameter scheduling, not full MPC replacement.\n"
        "Do Not Say\n"
        "- SI-RPPO is universally superior to SAC.\n"
        "Chrono/VLA belongs in future work, not as the main thesis contribution.\n"
        "BP-NMPS and SI-RPPO should be stated with careful SAC tradeoff wording.\n"
    )
    for path in [
        "docs/thesis_chapter_draft_cn.tex",
        "docs/thesis_chapter_draft_cn.md",
        "docs/thesis_defense_qa.md",
        "docs/si_rppo_ccfa_innovation_framing.md",
        "docs/thesis_final_submission_checklist.md",
    ]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(safe_text, encoding="utf-8")
    report = check_thesis_claim_boundaries(out_dir=Path("docs"))
    assert report["status"] == "claim_boundaries_pass"
    assert report["violation_count"] == 0
    assert report["guarded_mention_count"] >= 2


def test_check_thesis_claim_boundaries_flags_unguarded_overclaim(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bad = Path("bad.md")
    bad.write_text(
        "已感知路面不平度 PreScan BP-NMPS SI-RPPO SAC\n"
        "SI-RPPO is universally superior to SAC.\n",
        encoding="utf-8",
    )
    report = check_thesis_claim_boundaries(out_dir=Path("docs"), targets=(bad,))
    assert report["status"] == "claim_boundaries_fail"
    assert report["violation_count"] == 1


def test_build_thesis_readiness_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("docs/latex_preview").mkdir(parents=True)
    for path in [
        "docs/thesis_chapter_draft_cn.tex",
        "docs/thesis_latex_integration_guide.md",
        "docs/thesis_defense_qa.md",
        "docs/si_rppo_ccfa_innovation_framing.md",
        "docs/thesis_workflow_index.md",
        "docs/thesis_claim_matrix.md",
        "docs/thesis_completion_audit.md",
        "docs/thesis_objective_evidence.md",
        "docs/thesis_objective_evidence.json",
        "docs/thesis_claim_boundary_audit.md",
    ]:
        Path(path).write_text("ok", encoding="utf-8")
    Path("docs/thesis_claim_boundary_audit.json").write_text(
        json.dumps({"status": "claim_boundaries_pass"}),
        encoding="utf-8",
    )
    Path("docs/latex_preview/thesis_standalone_preview.pdf").write_bytes(b"%PDF")
    Path("docs/thesis_standalone_preview.tex").write_text("\\usepackage{graphicx}\n", encoding="utf-8")
    Path("docs/thesis_latex_snippets.tex").write_text(
        "\\usepackage{booktabs,graphicx}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        + "\n".join(f"\\label{{{label}}}" for label in [
            "tab:bp_nmps_seed_sweep",
            "tab:bp_nmps_evidence",
            "tab:bp_nmps_preview_robustness",
            "tab:si_rppo_gate_evidence",
            "tab:si_rppo_preview_robustness",
            "tab:si_rppo_strategy_evidence",
        ]),
        encoding="utf-8",
    )
    Path("docs/thesis_final_submission_checklist.md").write_text(
        "already perceived road unevenness\n"
        "Do not claim that PreScan\n"
        "Do not claim that Chrono/VLA\n"
        "Do not state that SI-RPPO is universally superior to SAC\n",
        encoding="utf-8",
    )
    Path("docs/thesis_export_package").mkdir()
    Path("docs/thesis_export_package/manifest.json").write_text("{}", encoding="utf-8")
    Path("docs/thesis_export_package/README.md").write_text("ok", encoding="utf-8")
    Path("docs/thesis_export_package.zip").write_bytes(b"PK")
    Path("docs/thesis_export_package.zip.sha256").write_text("hash  thesis_export_package.zip\n", encoding="utf-8")
    report = build_thesis_readiness_report(out_dir=Path("docs"))
    assert report["status"] == "ready_for_template_migration"
    assert report["failed"] == 0
    assert Path(report["json"]).is_file()
    assert Path(report["markdown"]).is_file()


def test_export_thesis_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for path in [
        "docs/thesis_chapter_draft_cn.tex",
        "docs/thesis_latex_snippets.tex",
        "docs/thesis_standalone_preview.tex",
        "docs/thesis_latex_integration_guide.md",
        "docs/thesis_workflow_index.md",
        "docs/thesis_final_submission_checklist.md",
        "docs/thesis_defense_qa.md",
        "docs/si_rppo_ccfa_innovation_framing.md",
        "docs/thesis_evidence_package.md",
        "docs/thesis_claim_matrix.md",
        "docs/thesis_completion_audit.md",
        "docs/thesis_objective_evidence.md",
        "docs/thesis_objective_evidence.json",
        "docs/thesis_claim_boundary_audit.md",
        "docs/thesis_claim_boundary_audit.json",
        "docs/thesis_readiness_report.md",
        "docs/thesis_readiness_report.json",
    ]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok", encoding="utf-8")
    pdf = Path("docs/latex_preview/thesis_standalone_preview.pdf")
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF")
    manifest = export_thesis_package(out_dir=Path("export"))
    assert manifest["status"] == "exported"
    assert manifest["copied_count"] == 18
    assert Path(manifest["manifest"]).is_file()
    assert Path(manifest["readme"]).is_file()
    assert Path(manifest["archive"]).is_file()
    assert Path(manifest["archive_sha256_file"]).is_file()
    assert manifest["archive_size_bytes"] > 0
    assert len(manifest["archive_sha256"]) == 64
    assert "SHA-256" in Path(manifest["readme"]).read_text(encoding="utf-8")
    assert Path("export/latex/thesis_chapter_draft_cn.tex").is_file()
    assert Path("export/evidence/thesis_objective_evidence.json").is_file()
    assert Path("export/evidence/thesis_claim_boundary_audit.json").is_file()
    assert Path("export/evidence/thesis_readiness_report.json").is_file()


def test_verify_thesis_package(tmp_path):
    archive = tmp_path / "thesis_export_package.zip"
    archive.write_bytes(b"zip-bytes")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    sidecar = tmp_path / "thesis_export_package.zip.sha256"
    sidecar.write_text(f"{digest}  thesis_export_package.zip\n", encoding="utf-8")
    result = verify_thesis_package(archive_path=archive, sha256_path=sidecar)
    assert result["status"] == "verified"
    assert result["expected_sha256"] == digest
    assert result["actual_sha256"] == digest
    assert result["size_bytes"] == len(b"zip-bytes")


def test_verify_thesis_package_detects_mismatch(tmp_path):
    archive = tmp_path / "thesis_export_package.zip"
    archive.write_bytes(b"zip-bytes")
    sidecar = tmp_path / "thesis_export_package.zip.sha256"
    sidecar.write_text(f"{'0' * 64}  thesis_export_package.zip\n", encoding="utf-8")
    result = verify_thesis_package(archive_path=archive, sha256_path=sidecar)
    assert result["status"] == "failed"
    assert result["actual_sha256"] != result["expected_sha256"]


def test_stage_thesis_template_import(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for path in [
        "docs/thesis_export_package/latex/thesis_chapter_draft_cn.tex",
        "docs/thesis_export_package/latex/thesis_latex_snippets.tex",
        "docs/thesis_export_package/guides/thesis_latex_integration_guide.md",
        "docs/thesis_export_package/guides/thesis_workflow_index.md",
        "docs/thesis_export_package/guides/thesis_final_submission_checklist.md",
        "docs/thesis_export_package/guides/thesis_defense_qa.md",
        "docs/thesis_export_package/evidence/thesis_readiness_report.md",
    ]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok", encoding="utf-8")
    pdf = Path("docs/thesis_export_package/latex/thesis_standalone_preview.pdf")
    pdf.write_bytes(b"%PDF")
    template = Path("school_template")
    template.mkdir()
    (template / "main.tex").write_text("\\documentclass{ctexbook}\n", encoding="utf-8")
    manifest = stage_thesis_template_import(template)
    assert manifest["status"] == "staged"
    assert manifest["copied_count"] == 8
    assert Path(manifest["manifest"]).is_file()
    assert Path(manifest["instructions"]).is_file()
    assert (template / "main.tex").read_text(encoding="utf-8") == "\\documentclass{ctexbook}\n"
    assert (template / "thesis_import" / "latex" / "thesis_chapter_draft_cn.tex").is_file()


def test_build_thesis_chapter_draft(tmp_path):
    bp = {
        "seed_count": 5,
        "support_rate": 1.0,
        "mean_policy_delta_vs_fixed": -0.001,
        "mean_unsafe_delta_vs_fixed": 0.0,
        "noninferiority_status": "supported",
        "improvement_status": "supported",
    }
    bp_path = tmp_path / "bp.json"
    bp_path.write_text(json.dumps(bp), encoding="utf-8")
    strategy_path = tmp_path / "strategy.csv"
    pd.DataFrame(
        [
            {
                "Component": "Projection-aware imitation PPO",
                "Evidence": "BC-PPO vs PPO scratch",
                "MainResult": "return 10",
                "Status": "supported",
            },
            {
                "Component": "Residual prior branch",
                "Evidence": "residual BC-PPO vs direct BC-PPO",
                "MainResult": "return -1",
                "Status": "weak_or_contradicted",
            }
        ]
    ).to_csv(strategy_path, index=False)
    pairwise_path = tmp_path / "pairwise.csv"
    pd.DataFrame(
        [
            {"Comparison": "SI-RPPO vs PPO scratch", "EpisodeReturnDelta": 1.0, "UnsafeStepsDelta": -1.0, "ActionDeltaRMS_NDelta": -2.0, "PolicyProjectionErrorDelta": -0.1},
            {"Comparison": "SI-RPPO vs SAC", "EpisodeReturnDelta": -1.0, "ActionDeltaRMS_NDelta": -2.0, "PolicyProjectionErrorDelta": -0.1},
            {"Comparison": "SI-RPPO vs TD3", "EpisodeReturnDelta": 2.0, "UnsafeStepsDelta": -3.0, "ActionDeltaRMS_NDelta": -4.0},
        ]
    ).to_csv(pairwise_path, index=False)
    out = tmp_path / "chapter.md"
    build_thesis_chapter_draft(
        out_path=out,
        bp_summary_path=bp_path,
        si_strategy_csv_path=strategy_path,
        si_preview_pairwise_path=pairwise_path,
    )
    text = out.read_text(encoding="utf-8")
    assert "BP-NMPS" in text
    assert "SI-RPPO" in text
    assert "tab:si_rppo_strategy_evidence" in text


def test_build_thesis_chapter_latex(tmp_path):
    bp_path = tmp_path / "bp.json"
    bp_path.write_text(
        json.dumps(
            {
                "seed_count": 5,
                "support_rate": 1.0,
                "mean_policy_delta_vs_fixed": -0.001,
                "mean_unsafe_delta_vs_fixed": 0.0,
            }
        ),
        encoding="utf-8",
    )
    strategy_path = tmp_path / "strategy.csv"
    pd.DataFrame(
        [
            {
                "Component": "Projection-aware imitation PPO",
                "Evidence": "BC-PPO vs PPO scratch",
                "MainResult": "return 10",
                "Status": "supported",
            },
            {
                "Component": "Residual prior branch",
                "Evidence": "residual BC-PPO vs direct BC-PPO",
                "MainResult": "return -1",
                "Status": "weak_or_contradicted",
            }
        ]
    ).to_csv(strategy_path, index=False)
    pairwise_path = tmp_path / "pairwise.csv"
    pd.DataFrame(
        [
            {"Comparison": "SI-RPPO vs PPO scratch", "EpisodeReturnDelta": 1.0, "UnsafeStepsDelta": -1.0, "ActionDeltaRMS_NDelta": -2.0, "PolicyProjectionErrorDelta": -0.1},
            {"Comparison": "SI-RPPO vs SAC", "EpisodeReturnDelta": -1.0, "ActionDeltaRMS_NDelta": -2.0, "PolicyProjectionErrorDelta": -0.1},
            {"Comparison": "SI-RPPO vs TD3", "EpisodeReturnDelta": 2.0, "UnsafeStepsDelta": -3.0, "ActionDeltaRMS_NDelta": -4.0},
        ]
    ).to_csv(pairwise_path, index=False)
    out = tmp_path / "chapter.tex"
    build_thesis_chapter_latex(
        out_path=out,
        bp_summary_path=bp_path,
        si_strategy_csv_path=strategy_path,
        si_preview_pairwise_path=pairwise_path,
    )
    text = out.read_text(encoding="utf-8")
    assert "\\chapter" in text
    assert "\\ref{tab:bp_nmps_seed_sweep}" in text
    assert "\\ref{tab:si_rppo_strategy_evidence}" in text
    assert "weak\\_or\\_contradicted" in text


def test_build_thesis_standalone_preview(tmp_path):
    chapter = tmp_path / "chapter.tex"
    chapter.write_text("\\chapter{Demo}\\label{chap:demo}\n", encoding="utf-8")
    snippets = tmp_path / "snippets.tex"
    snippets.write_text("\\begin{table}\\caption{T}\\label{tab:t}\\end{table}\n", encoding="utf-8")
    out = tmp_path / "preview.tex"
    build_thesis_standalone_preview(out_path=out, chapter_path=chapter, snippets_path=snippets)
    text = out.read_text(encoding="utf-8")
    assert "\\documentclass" in text
    assert "\\usepackage{graphicx}" in text
    assert "基于路面预瞄的主动悬架控制方法与实验预览" in text
    assert "\\input{" in text
    assert "ctexbook" in text


def test_mpc_parameter_heldout_split_requires_scenarios(tmp_path):
    c = load_config(ROOT / "configs" / "mujoco_full_car_corner.yaml")
    c["episode_seconds"] = 0.03
    c["mujoco"]["settle_seconds"] = 0.05
    c["scenarios"] = c["scenarios"][:1]
    cfg_path = tmp_path / "mpc_param_bad_split.yaml"
    cfg_path.write_text(__import__("yaml").safe_dump(c, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="selected no scenarios"):
        run_mpc_parameter_learning_benchmark(
            config_path=cfg_path,
            out_dir=tmp_path / "bad_split",
            scenario_limit=1,
            tuning_trials=1,
            cem_iterations=1,
            cem_population=1,
            neural_episodes=1,
            neural_batch_size=1,
            eval_scenario_offset=1,
            eval_scenario_limit=1,
            device="cpu",
        )


def test_mujoco_robustness_matrix_smoke(tmp_path):
    matrix = run_robustness_matrix(
        config_path=ROOT / "configs" / "mujoco_full_car_corner.yaml",
        out_dir=tmp_path / "matrix",
        max_steps=2,
        scenario_limit=1,
        max_unsafe_fraction=1.0,
    )
    assert matrix["passed"]
    assert len(matrix["cases"]) == 4
    assert Path(matrix["matrix_report"]).is_file()


def test_scenario_sampler_weighted_and_curriculum():
    scenarios = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
    weighted = ScenarioSampler(
        scenarios,
        {"seed": 1, "scenario_sampling": {"mode": "weighted", "weights": {"a": 0.0, "b": 1.0, "c": 0.0}}},
    )
    assert [weighted.select(i)["name"] for i in range(5)] == ["b"] * 5
    curriculum = ScenarioSampler(
        scenarios,
        {
            "seed": 1,
            "scenario_sampling": {
                "mode": "cycle",
                "curriculum": [
                    {"until_episode": 2, "mode": "cycle", "scenarios": ["a"]},
                    {"until_episode": 4, "mode": "cycle", "scenarios": ["b"]},
                ],
            },
        },
    )
    assert [curriculum.select(i)["name"] for i in range(5)] == ["a", "a", "b", "b", "b"]


def test_controller_smoke_1s():
    c = cfg()
    for scenario in c["scenarios"]:
        for controller_cls, use_preview in (
            (PIDController, False),
            (SPDFController, False),
            (MPCController, True),
        ):
            env = HalfCarEnv(c, scenario=scenario, use_preview=use_preview)
            obs, info = env.reset()
            if controller_cls is PIDController:
                controller = PIDController(env.params, env.control_dt, env.force_limit)
            elif controller_cls is SPDFController:
                controller = SPDFController(env.params, env.control_dt)
            else:
                controller = MPCController(env.model, c)
            done = False
            while not done:
                action = controller.compute_action(obs if use_preview else obs[: env.base_obs_dim], info)
                obs, _, terminated, truncated, info = env.step(action)
                assert np.all(np.isfinite(obs))
                done = terminated or truncated


def test_mixed_replay_sampler_decays_expert_ratio():
    online = ReplayBuffer(obs_dim=3, act_dim=2, capacity=20)
    expert = ReplayBuffer(obs_dim=3, act_dim=2, capacity=20)
    for i in range(10):
        obs = np.ones(3, dtype=np.float32) * i
        act = np.ones(2, dtype=np.float32)
        online.store(obs, act, 0.0, obs, False)
        expert.store(obs, act, 1.0, obs, False)
    sampler = MixedReplaySampler(online, expert, 0.5, 0.1, 10)
    sampler.set_episode(0)
    assert np.isclose(sampler.expert_ratio, 0.5)
    sampler.set_episode(10)
    assert np.isclose(sampler.expert_ratio, 0.1)
    batch = sampler.sample(8, torch.device("cpu"))
    assert batch["obs"].shape == (8, 3)
    assert batch["act"].shape == (8, 2)
