from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from controllers import MPCController, PIDController, RLTransformerController, SPDFController
from controllers.prior import (
    adapt_action_to_env,
    parameterize_policy_action,
    policy_improvement_gate,
    shield_policy_action,
)
from envs import HalfCarEnv


class PassiveController:
    def __init__(self, act_dim: int):
        self.act_dim = int(act_dim)

    def reset(self):
        pass

    def compute_action(self, obs, info):
        return np.zeros(self.act_dim, dtype=np.float32)


class PolicySafetyController:
    def __init__(self, controller, env, cfg: dict, action_param_cfg: dict, gate_cfg: dict):
        self.controller = controller
        self.env = env
        self.cfg = dict(cfg)
        self.action_param_cfg = dict(action_param_cfg)
        self.gate_cfg = dict(gate_cfg)
        self.last_projection_error = 0.0
        self.last_projection_delta_rms = 0.0
        self.last_improvement_gate = 1.0

    def reset(self):
        self.controller.reset()
        self.last_projection_error = 0.0
        self.last_projection_delta_rms = 0.0
        self.last_improvement_gate = 1.0

    def compute_action(self, obs, info):
        raw = self.controller.compute_action(obs, info)
        parameterized = parameterize_policy_action(raw, self.env, info, self.action_param_cfg)
        parameterized, self.last_improvement_gate = policy_improvement_gate(
            parameterized,
            self.env,
            info,
            self.gate_cfg,
        )
        projected = shield_policy_action(parameterized, self.env, info, self.cfg)
        raw_action = adapt_action_to_env(parameterized, self.env).astype(np.float64)
        projected_action = adapt_action_to_env(projected, self.env).astype(np.float64)
        delta = projected_action - raw_action
        force_limit = max(float(getattr(self.env, "force_limit", 1.0)), 1e-6)
        self.last_projection_delta_rms = float(np.sqrt(np.mean(np.square(delta))))
        self.last_projection_error = float(self.last_projection_delta_rms / force_limit)
        return projected


def make_controller(name: str, env: HalfCarEnv, config: dict, checkpoint: str | None):
    key = name.lower()
    if key == "passive":
        return PassiveController(env.action_space.shape[0])
    if key == "pid":
        return PIDController(env.params, env.control_dt, env.force_limit)
    if key == "spdf":
        return SPDFController(env.params, env.control_dt, config=config)
    if key == "mpc":
        return MPCController(env.model, config)
    if key in {"ppo", "si-rppo", "sirppo"}:
        if not checkpoint:
            raise ValueError(f"{name} requires a checkpoint path")
        base = RLTransformerController(config, algorithm="ppo", checkpoint=checkpoint)
        safety_cfg = dict(config.get("policy_safety", {}))
        action_param_cfg = dict(config.get("policy_action_parameterization", {}))
        gate_cfg = dict(config.get("policy_improvement_gate", {}))
        if safety_cfg.get("enabled", False) or action_param_cfg.get("enabled", False) or gate_cfg.get("enabled", False):
            return PolicySafetyController(base, env, safety_cfg, action_param_cfg, gate_cfg)
        return base
    raise ValueError(f"Unknown controller: {name}")


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def peak(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if values.size else 0.0


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rollout(env: HalfCarEnv, controller, use_preview: bool) -> tuple[list[dict], dict]:
    obs, info = env.reset(options={"scenario": env.scenario})
    controller.reset()
    rows: list[dict] = []
    done = False
    while not done:
        obs_for_controller = obs if use_preview else obs[: env.base_obs_dim]
        action = controller.compute_action(obs_for_controller, info)
        obs, reward, terminated, truncated, info = env.step(action)
        d = info["derived"]
        state = info["state"]
        road = info["road"]
        executed = np.asarray(info["action"], dtype=np.float64).reshape(-1)
        row = {
            "time": float(info["time"]),
            "zb": float(state[0]),
            "dzb": float(state[1]),
            "theta": float(state[2]),
            "dtheta": float(state[3]),
            "zwf": float(state[4]),
            "zwr": float(state[6]),
            "zdf": float(road[0]),
            "zdr": float(road[2]),
            "ddzb": float(d["ddzb"]),
            "ddtheta": float(d["ddtheta"]),
            "delta_yf": float(d["delta_yf"]),
            "delta_yr": float(d["delta_yr"]),
            "Fpf": float(d["Fpf"]),
            "Fpr": float(d["Fpr"]),
            "Uaf": float(d.get("Uaf", executed[0])),
            "Uar": float(d.get("Uar", executed[1] if executed.size > 1 else 0.0)),
            "reward": float(reward),
            "unsafe": int(info["unsafe"]),
        }
        if hasattr(controller, "last_projection_error"):
            row["policy_projection_error"] = float(controller.last_projection_error)
            row["policy_projection_delta_rms"] = float(controller.last_projection_delta_rms)
        if hasattr(controller, "last_improvement_gate"):
            row["policy_improvement_gate"] = float(controller.last_improvement_gate)
        if hasattr(controller, "last_reference"):
            for ref_name, value in dict(controller.last_reference).items():
                if np.isscalar(value):
                    row[f"reference_{ref_name}"] = float(value)
        rows.append(row)
        done = bool(terminated or truncated)
    return rows, dict(info.get("reward_scales", {}))


def compute_metrics(rows: list[dict], settle_seconds: float) -> dict:
    view = [row for row in rows if row["time"] >= settle_seconds] or rows

    def arr(name: str) -> np.ndarray:
        return np.asarray([row[name] for row in view], dtype=np.float64)

    all_time = np.asarray([row["time"] for row in rows], dtype=np.float64)
    dt = float(np.mean(np.diff(all_time))) if all_time.size > 1 else 0.01
    uaf = np.asarray([row["Uaf"] for row in rows], dtype=np.float64)
    uar = np.asarray([row["Uar"] for row in rows], dtype=np.float64)
    dyf = np.asarray([row["delta_yf"] for row in rows], dtype=np.float64)
    dyr = np.asarray([row["delta_yr"] for row in rows], dtype=np.float64)
    power = uaf * np.diff(dyf, prepend=dyf[0]) / max(dt, 1e-9)
    power += uar * np.diff(dyr, prepend=dyr[0]) / max(dt, 1e-9)

    metrics = {
        "BodyAccRMS_mps2": rms(arr("ddzb")),
        "BodyAccPeak_mps2": peak(arr("ddzb")),
        "PitchAccRMS_radps2": rms(arr("ddtheta")),
        "PitchAccPeak_radps2": peak(arr("ddtheta")),
        "MaxPitch_rad": peak(arr("theta")),
        "MaxFrontSusp_m": peak(arr("delta_yf")),
        "MaxRearSusp_m": peak(arr("delta_yr")),
        "FrontTireLoadRMS_N": rms(arr("Fpf")),
        "RearTireLoadRMS_N": rms(arr("Fpr")),
        "MaxUaf_N": peak(arr("Uaf")),
        "MaxUar_N": peak(arr("Uar")),
        "ControlAbsEnergy_J": float(np.sum(np.abs(power)) * dt),
        "UnsafeSteps": int(np.sum(arr("unsafe"))),
        "EpisodeReturn": float(sum(row["reward"] for row in rows)),
    }
    for name, label in (
        ("policy_projection_error", "PolicyProjectionError"),
        ("policy_projection_delta_rms", "PolicyProjectionDeltaRMS_N"),
        ("policy_improvement_gate", "PolicyImprovementGate"),
    ):
        if name in view[0]:
            metrics[label] = float(np.mean([row.get(name, 0.0) for row in view]))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--controllers", default="PASSIVE,PID,SPDF,MPC")
    parser.add_argument("--checkpoint", default=None, help="PPO/SI-RPPO checkpoint path")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_config(config_path)
    controllers = [item.strip() for item in args.controllers.split(",") if item.strip()]
    out_dir = Path(args.out) if args.out else ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_report_eval")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    manifest = {
        "config": str(config_path),
        "controllers": controllers,
        "scenarios": [scenario.get("name", "") for scenario in config["scenarios"]],
        "checkpoint": args.checkpoint,
        "reward_scales": None,
    }
    with (out_dir / "evaluation_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    for scenario in config["scenarios"]:
        for controller_name in controllers:
            controller_config = deepcopy(config)
            use_preview = controller_name.lower() in set(config["preview"]["enabled_for"]) or controller_name.lower() in {
                "mpc",
                "ppo",
                "si-rppo",
                "sirppo",
            }
            env = HalfCarEnv(controller_config, scenario=scenario, use_preview=use_preview)
            controller = make_controller(controller_name, env, controller_config, args.checkpoint)
            rows, reward_scales = rollout(env, controller, use_preview)
            if manifest["reward_scales"] is None:
                manifest["reward_scales"] = reward_scales
            key = f"{controller_name}_{scenario['name']}".replace("/", "_")
            write_csv(out_dir / f"{key}.csv", rows)
            metrics = compute_metrics(rows, float(config.get("settle_seconds", 1.0)))
            metric_rows.append(
                {
                    "Controller": controller_name,
                    "Scenario": scenario["name"],
                    "ScenarioLabel": scenario.get("label", scenario["name"]),
                    "PreviewUsed": int(use_preview),
                    **metrics,
                }
            )
            print(
                f"{controller_name:8s} {scenario['name']:12s} "
                f"return={metrics['EpisodeReturn']:.3f} "
                f"body_rms={metrics['BodyAccRMS_mps2']:.4g} "
                f"pitch_rms={metrics['PitchAccRMS_radps2']:.4g} "
                f"unsafe={metrics['UnsafeSteps']}"
            )

    write_csv(out_dir / "metrics.csv", metric_rows)
    with (out_dir / "evaluation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation to {out_dir}")


if __name__ == "__main__":
    main()
