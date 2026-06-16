from __future__ import annotations

import argparse
import json
import sys
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.mpc_parameter_env import MPCParameterPolicyEnv
from rl.mpc_parameter_policy import MPCParameterGaussianPolicy
from config import load_config
from scripts.tune_mpc_parameters import _evaluate_gain_config


def _parse_hidden_sizes(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, str):
        return [int(v.strip()) for v in value.split(",") if v.strip()]
    return [int(v) for v in value]


def _constraint_value(metrics: dict | None, key: str, default: float = 0.0) -> float:
    if not metrics:
        return default
    values = [float(row[key]) for row in metrics.values() if key in row and row[key] is not None]
    return float(np.mean(values)) if values else default


def _relative_excess(value: float, reference: float, eps: float = 1e-6) -> float:
    return max(0.0, float(value) - float(reference)) / max(abs(float(reference)), eps)


def _sched_controller(config: dict) -> str:
    """A2': 从 config 读调度的控制器名（默认整车 MPC-lite，半车走 mpc_parameter_scheduling）"""
    return config.get("mpc_parameter_scheduling", {}).get("controller_name", "FULL_CAR_MPC_LITE")


def _sched_section(config: dict) -> str:
    return config.get("mpc_parameter_scheduling", {}).get("gain_section", "reduced_full_car_mpc")


def _baseline_by_scenario(config: dict, scenario_limit: int | None) -> dict[str, dict]:
    scenarios = list(config.get("scenarios", []))
    if scenario_limit is not None:
        scenarios = scenarios[:scenario_limit]
    baselines: dict[str, dict] = {}
    for scenario in scenarios:
        trial_config = dict(config)
        trial_config["scenarios"] = [scenario]
        score, metrics = _evaluate_gain_config(
            trial_config,
            controller_name=_sched_controller(config),
            scenario_limit=1,
            score_weights=config.get("mpc_parameter_tuning", {}).get("score_weights"),
        )
        baselines[str(scenario.get("name", ""))] = {
            "score": float(score),
            "metrics": metrics,
            "body_acc": _constraint_value(metrics, "BodyAccRMS_mps2"),
            "unsafe": _constraint_value(metrics, "UnsafeSteps"),
            "suspension": _constraint_value(metrics, "MaxFrontSusp_m") + _constraint_value(metrics, "MaxRearSusp_m"),
            "actuator": _constraint_value(metrics, "ControlAbsEnergy_J"),
        }
    return baselines


def _shaped_reward(
    raw_reward: float,
    score: float,
    metrics: dict | None,
    scenario: str,
    baselines: dict[str, dict],
    baseline_penalty_weight: float,
    unsafe_penalty_weight: float,
    suspension_penalty_weight: float,
    actuator_penalty_weight: float,
) -> tuple[float, dict[str, float]]:
    baseline = baselines.get(scenario, {})
    penalty_terms = {
        "score_penalty": float(baseline_penalty_weight) * _relative_excess(score, baseline.get("score", score)),
        "unsafe_penalty": float(unsafe_penalty_weight)
        * _relative_excess(_constraint_value(metrics, "UnsafeSteps"), baseline.get("unsafe", 0.0), eps=1.0),
        "suspension_penalty": float(suspension_penalty_weight)
        * _relative_excess(
            _constraint_value(metrics, "MaxFrontSusp_m") + _constraint_value(metrics, "MaxRearSusp_m"),
            baseline.get("suspension", 0.0),
        ),
        "actuator_penalty": float(actuator_penalty_weight)
        * _relative_excess(_constraint_value(metrics, "ControlAbsEnergy_J"), baseline.get("actuator", 0.0)),
    }
    total_penalty = sum(penalty_terms.values())
    return float(raw_reward) - total_penalty, {**penalty_terms, "constraint_penalty": float(total_penalty)}


def _evaluate_deterministic_policy(
    env: MPCParameterPolicyEnv,
    policy: MPCParameterGaussianPolicy,
    device: torch.device,
    seed: int,
) -> tuple[float, dict, list[dict]]:
    rows = []
    scenario_metrics = {}
    for index, scenario in enumerate(env.scenarios or [None]):
        options = {"scenario": scenario} if scenario is not None else None
        obs, info = env.reset(seed=seed + index, options=options)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, _, _ = policy.sample(obs_t, deterministic=True)
        action = action_t.detach().cpu().numpy()[0]
        _, reward, terminated, truncated, step_info = env.step(action)
        scenario_name = step_info.get("active_scenario", info.get("active_scenario", ""))
        score = -float(reward)
        scenario_metrics.update(step_info["last_metrics"] or {})
        rows.append(
            {
                "scenario": scenario_name,
                "score": score,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                **{f"action_{key}": float(value) for key, value in zip(step_info["gain_keys"], action)},
                **{key: float(step_info["last_gains"][key]) for key in step_info["gain_keys"]},
            }
        )
    mean_score = float(np.mean([row["score"] for row in rows])) if rows else float("inf")
    return mean_score, scenario_metrics, rows


def _fixed_policy_rows(config: dict, scenario_metrics: dict, base_gains: dict, gain_keys: list[str]) -> list[dict]:
    rows = []
    for scenario in config.get("scenarios", []):
        scenario_name = str(scenario.get("name", ""))
        metrics = scenario_metrics.get(scenario_name, {})
        score = float(metrics.get("score", 0.0))
        rows.append(
            {
                "scenario": scenario_name,
                "score": score,
                "reward": -score,
                "terminated": True,
                "truncated": False,
                "deployment_gate_active": True,
                **{f"action_{key}": 0.0 for key in gain_keys},
                **{key: float(base_gains[key]) for key in gain_keys},
            }
        )
    return rows


def _eval_config_with_scenarios(config: dict, offset: int | None, limit: int | None) -> tuple[dict, list[str]]:
    scenarios = list(config.get("scenarios", []))
    start = int(offset or 0)
    if start < 0:
        raise ValueError("eval_scenario_offset must be non-negative")
    selected = scenarios[start:]
    if limit is not None:
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError(
            f"eval scenario split selected no scenarios: offset={offset}, limit={limit}, total={len(scenarios)}"
        )
    eval_config = copy.deepcopy(config)
    eval_config["scenarios"] = selected
    return eval_config, [str(s.get("name", "")) for s in selected]


def _scheduled_anchor_coef(
    initial_coef: float,
    final_coef: float | None,
    update: int,
    total_updates: int,
) -> float:
    if final_coef is None:
        return float(initial_coef)
    if total_updates <= 1:
        return float(final_coef)
    progress = min(1.0, max(0.0, float(update) / float(total_updates - 1)))
    return float(initial_coef) + progress * (float(final_coef) - float(initial_coef))


def train_mpc_parameter_neural_policy(
    config_path: Path,
    out_dir: Path,
    episodes: int = 16,
    batch_size: int = 4,
    hidden_sizes: list[int] | tuple[int, ...] = (64, 64),
    actor_lr: float = 3e-3,
    entropy_coef: float = 1e-3,
    anchor_coef: float = 0.1,
    anchor_final_coef: float | None = None,
    elite_bc_coef: float = 0.0,
    seed: int = 42,
    scenario_limit: int | None = None,
    scale_low: float = 0.5,
    scale_high: float = 1.5,
    gradient_clip: float = 1.0,
    baseline_penalty_weight: float = 0.0,
    unsafe_penalty_weight: float = 0.0,
    suspension_penalty_weight: float = 0.0,
    actuator_penalty_weight: float = 0.0,
    eval_scenario_offset: int | None = None,
    eval_scenario_limit: int | None = None,
    deployment_gate: bool = False,
    deployment_min_train_improvement: float = 0.0,
    device: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    full_config = load_config(config_path)
    env = MPCParameterPolicyEnv(
        config=full_config,
        scenario_limit=scenario_limit,
        scale_low=scale_low,
        scale_high=scale_high,
    )
    train_gate_config = copy.deepcopy(env.config)
    policy = MPCParameterGaussianPolicy(
        obs_dim=int(env.observation_space.shape[0]),
        act_dim=int(env.action_space.shape[0]),
        hidden_sizes=_parse_hidden_sizes(hidden_sizes),
    ).to(torch_device)
    optimizer = optim.Adam(policy.parameters(), lr=actor_lr)
    constraint_weights = {
        "baseline_penalty_weight": float(baseline_penalty_weight),
        "unsafe_penalty_weight": float(unsafe_penalty_weight),
        "suspension_penalty_weight": float(suspension_penalty_weight),
        "actuator_penalty_weight": float(actuator_penalty_weight),
    }
    constrained_training = any(value > 0.0 for value in constraint_weights.values())
    scenario_baselines = _baseline_by_scenario(env.config, scenario_limit) if constrained_training else {}
    rows = []
    baseline = 0.0
    best = {"reward": -float("inf"), "score": float("inf"), "action": None, "gains": None, "metrics": None}
    total_updates = int(np.ceil(max(1, episodes) / max(1, batch_size)))

    try:
        episode = 0
        update = 0
        while episode < episodes:
            log_probs = []
            entropies = []
            anchor_losses = []
            mus = []
            sampled_actions = []
            rewards = []
            batch_rows = []
            for _ in range(min(batch_size, episodes - episode)):
                obs, reset_info = env.reset(seed=seed + episode)
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=torch_device).unsqueeze(0)
                action_t, log_prob_t, entropy_t = policy.sample(obs_t, deterministic=False)
                mu_t, _ = policy(obs_t)
                anchor_loss_t = mu_t.pow(2).mean()
                action = action_t.detach().cpu().numpy()[0]
                _, raw_reward, terminated, truncated, info = env.step(action)
                score = -float(raw_reward)
                scenario_name = info.get("active_scenario", reset_info.get("active_scenario", ""))
                reward, penalty_info = _shaped_reward(
                    raw_reward=float(raw_reward),
                    score=score,
                    metrics=info["last_metrics"],
                    scenario=scenario_name,
                    baselines=scenario_baselines,
                    baseline_penalty_weight=baseline_penalty_weight,
                    unsafe_penalty_weight=unsafe_penalty_weight,
                    suspension_penalty_weight=suspension_penalty_weight,
                    actuator_penalty_weight=actuator_penalty_weight,
                )
                rewards.append(float(reward))
                log_probs.append(log_prob_t.squeeze())
                entropies.append(entropy_t.squeeze())
                anchor_losses.append(anchor_loss_t.squeeze())
                mus.append(mu_t.squeeze(0))
                sampled_actions.append(action_t.detach().squeeze(0))
                row = {
                    "episode": episode,
                    "update": update,
                    "scenario": scenario_name,
                    "reward": float(reward),
                    "raw_reward": float(raw_reward),
                    "score": score,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "anchor_loss": float(anchor_loss_t.detach().cpu().item()),
                    **penalty_info,
                    **{f"action_{key}": float(value) for key, value in zip(info["gain_keys"], action)},
                    **{key: float(info["last_gains"][key]) for key in info["gain_keys"]},
                }
                batch_rows.append(row)
                if reward > best["reward"]:
                    best = {
                        "reward": float(reward),
                        "raw_reward": float(raw_reward),
                        "score": score,
                        "action": [float(v) for v in action],
                        "gains": info["last_gains"],
                        "metrics": info["last_metrics"],
                    }
                episode += 1

            reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=torch_device)
            if len(rewards) > 1:
                advantages = (reward_t - reward_t.mean()) / (reward_t.std(unbiased=False) + 1e-6)
            else:
                advantages = reward_t - baseline
            log_prob_t = torch.stack(log_probs)
            entropy_t = torch.stack(entropies)
            anchor_loss = torch.stack(anchor_losses).mean()
            policy_loss = -(log_prob_t * advantages.detach()).mean()
            entropy_loss = -float(entropy_coef) * entropy_t.mean()
            mu_t = torch.stack(mus)
            sampled_action_t = torch.stack(sampled_actions)
            elite_mask = advantages.detach() > 0.0
            if bool(elite_mask.any()):
                elite_bc_loss = (mu_t[elite_mask] - sampled_action_t[elite_mask]).pow(2).mean()
            else:
                elite_bc_loss = torch.zeros((), dtype=torch.float32, device=torch_device)
            current_anchor_coef = _scheduled_anchor_coef(anchor_coef, anchor_final_coef, update, total_updates)
            loss = policy_loss + entropy_loss + current_anchor_coef * anchor_loss + float(elite_bc_coef) * elite_bc_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(gradient_clip))
            optimizer.step()
            baseline = 0.9 * baseline + 0.1 * float(np.mean(rewards))
            for row in batch_rows:
                row["loss"] = float(loss.detach().cpu().item())
                row["policy_loss"] = float(policy_loss.detach().cpu().item())
                row["entropy_loss"] = float(entropy_loss.detach().cpu().item())
                row["batch_anchor_loss"] = float(anchor_loss.detach().cpu().item())
                row["elite_bc_loss"] = float(elite_bc_loss.detach().cpu().item())
                row["elite_bc_coef"] = float(elite_bc_coef)
                row["anchor_coef"] = float(current_anchor_coef)
                row["baseline"] = float(baseline)
                rows.append(row)
            update += 1
    finally:
        env.close()

    score_weights = full_config.get("mpc_parameter_tuning", {}).get("score_weights")
    train_gate_env = MPCParameterPolicyEnv(
        config=train_gate_config,
        scenario_limit=scenario_limit,
        scale_low=scale_low,
        scale_high=scale_high,
    )
    try:
        train_policy_score, train_policy_metrics, train_policy_rows = _evaluate_deterministic_policy(
            train_gate_env,
            policy,
            torch_device,
            seed=seed + episodes + 5_000,
        )
    finally:
        train_gate_env.close()
    train_fixed_score, train_fixed_metrics = _evaluate_gain_config(
        train_gate_config,
        controller_name=_sched_controller(train_gate_config),
        scenario_limit=scenario_limit,
        score_weights=score_weights,
    )
    train_policy_delta = float(train_policy_score) - float(train_fixed_score)
    deployment_gate_active = bool(deployment_gate and train_policy_delta > -float(deployment_min_train_improvement))

    eval_config, eval_scenario_names = _eval_config_with_scenarios(full_config, eval_scenario_offset, eval_scenario_limit)
    eval_env = MPCParameterPolicyEnv(
        config=eval_config,
        scenario_limit=eval_scenario_limit,
        scale_low=scale_low,
        scale_high=scale_high,
    )
    try:
        final_policy_score, final_policy_metrics, final_policy_rows = _evaluate_deterministic_policy(
            eval_env,
            policy,
            torch_device,
            seed=seed + episodes + 10_000,
        )
    finally:
        eval_env.close()
    raw_final_policy_score = float(final_policy_score)
    raw_final_policy_metrics = final_policy_metrics
    raw_final_policy_rows = final_policy_rows
    if deployment_gate_active:
        final_policy_score, final_policy_metrics = _evaluate_gain_config(
            eval_config,
            controller_name=_sched_controller(eval_config),
            scenario_limit=eval_scenario_limit,
            score_weights=score_weights,
        )
        final_policy_rows = _fixed_policy_rows(eval_config, final_policy_metrics, env.base_gains, list(env._info()["gain_keys"]))
    else:
        for row in final_policy_rows:
            row["deployment_gate_active"] = False

    history_path = out_dir / "mpc_parameter_neural_policy_history.csv"
    pd.DataFrame(rows).to_csv(history_path, index=False)
    final_policy_path = out_dir / "mpc_parameter_neural_policy_final_actions.csv"
    pd.DataFrame(final_policy_rows).to_csv(final_policy_path, index=False)
    checkpoint_path = out_dir / "mpc_parameter_neural_policy.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "obs_dim": int(env.observation_space.shape[0]),
            "act_dim": int(env.action_space.shape[0]),
            "hidden_sizes": _parse_hidden_sizes(hidden_sizes),
            "gain_keys": list(env._info()["gain_keys"]),
            "scale_low": float(scale_low),
            "scale_high": float(scale_high),
            "deployment_gate_active": bool(deployment_gate_active),
            "deployment_gate": {
                "enabled": bool(deployment_gate),
                "min_train_improvement": float(deployment_min_train_improvement),
                "train_policy_score": float(train_policy_score),
                "train_fixed_score": float(train_fixed_score),
                "train_policy_delta_vs_fixed": float(train_policy_delta),
            },
        },
        checkpoint_path,
    )
    best_config = env.config.copy()
    best_config[_sched_section(env.config)] = best["gains"]
    best_config_path = out_dir / "best_mpc_parameter_neural_policy_config.yaml"
    with best_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_config, f, sort_keys=False, allow_unicode=True)
    manifest = {
        "config": str(config_path.resolve()),
        "episodes": int(episodes),
        "batch_size": int(batch_size),
        "hidden_sizes": _parse_hidden_sizes(hidden_sizes),
        "actor_lr": float(actor_lr),
        "entropy_coef": float(entropy_coef),
        "anchor_coef": float(anchor_coef),
        "anchor_final_coef": None if anchor_final_coef is None else float(anchor_final_coef),
        "anchor_schedule": "constant" if anchor_final_coef is None else "linear",
        "elite_bc_coef": float(elite_bc_coef),
        "deployment_gate_enabled": bool(deployment_gate),
        "deployment_min_train_improvement": float(deployment_min_train_improvement),
        "deployment_gate_active": bool(deployment_gate_active),
        "train_policy_score": float(train_policy_score),
        "train_fixed_score": float(train_fixed_score),
        "train_policy_delta_vs_fixed": float(train_policy_delta),
        "train_policy_metrics": train_policy_metrics,
        "train_fixed_metrics": train_fixed_metrics,
        "train_policy_rows": train_policy_rows,
        **constraint_weights,
        "constrained_training": bool(constrained_training),
        "scenario_baselines": scenario_baselines,
        "scenario_limit": scenario_limit,
        "eval_scenario_offset": eval_scenario_offset,
        "eval_scenario_limit": eval_scenario_limit,
        "eval_scenarios": eval_scenario_names,
        "scale_low": float(scale_low),
        "scale_high": float(scale_high),
        "best_reward": float(best["reward"]),
        "best_raw_reward": float(best.get("raw_reward", best["reward"])),
        "best_score": float(best["score"]),
        "best_action": best["action"],
        "best_gains": best["gains"],
        "best_metrics": best["metrics"],
        "raw_final_policy_score": float(raw_final_policy_score),
        "raw_final_policy_metrics": raw_final_policy_metrics,
        "raw_final_policy_rows": raw_final_policy_rows,
        "final_policy_score": float(final_policy_score),
        "final_policy_metrics": final_policy_metrics,
        "final_policy_actions_csv": str(final_policy_path),
        "history_csv": str(history_path),
        "checkpoint": str(checkpoint_path),
        "best_config": str(best_config_path),
        "method": "neural_reinforce_mpc_parameter_policy",
    }
    manifest_path = out_dir / "mpc_parameter_neural_policy_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    manifest["manifest"] = str(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mujoco_full_car_corner.yaml")
    parser.add_argument("--out", default="results/mpc_parameter_neural_policy")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-sizes", default="64,64")
    parser.add_argument("--actor-lr", type=float, default=3e-3)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--anchor-coef", type=float, default=0.1)
    parser.add_argument("--anchor-final-coef", type=float, default=None)
    parser.add_argument("--elite-bc-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--scale-low", type=float, default=0.5)
    parser.add_argument("--scale-high", type=float, default=1.5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--baseline-penalty-weight", type=float, default=0.0)
    parser.add_argument("--unsafe-penalty-weight", type=float, default=0.0)
    parser.add_argument("--suspension-penalty-weight", type=float, default=0.0)
    parser.add_argument("--actuator-penalty-weight", type=float, default=0.0)
    parser.add_argument("--eval-scenario-offset", type=int, default=None)
    parser.add_argument("--eval-scenario-limit", type=int, default=None)
    parser.add_argument("--deployment-gate", action="store_true")
    parser.add_argument("--deployment-min-train-improvement", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    manifest = train_mpc_parameter_neural_policy(
        config_path=config_path,
        out_dir=out_dir,
        episodes=args.episodes,
        batch_size=args.batch_size,
        hidden_sizes=_parse_hidden_sizes(args.hidden_sizes),
        actor_lr=args.actor_lr,
        entropy_coef=args.entropy_coef,
        anchor_coef=args.anchor_coef,
        anchor_final_coef=args.anchor_final_coef,
        elite_bc_coef=args.elite_bc_coef,
        seed=args.seed,
        scenario_limit=args.scenario_limit,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
        gradient_clip=args.gradient_clip,
        baseline_penalty_weight=args.baseline_penalty_weight,
        unsafe_penalty_weight=args.unsafe_penalty_weight,
        suspension_penalty_weight=args.suspension_penalty_weight,
        actuator_penalty_weight=args.actuator_penalty_weight,
        eval_scenario_offset=args.eval_scenario_offset,
        eval_scenario_limit=args.eval_scenario_limit,
        deployment_gate=args.deployment_gate,
        deployment_min_train_improvement=args.deployment_min_train_improvement,
        device=args.device,
    )
    print(f"Best score {manifest['best_score']:.6g}; checkpoint: {manifest['checkpoint']}")


if __name__ == "__main__":
    main()
