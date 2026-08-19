from __future__ import annotations

import argparse
import json
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from experiments.evaluation import compute_metrics, make_controller, make_env, rollout


CLASSICAL_CONTROLLERS = ("PASSIVE", "PID", "SPDF", "MPC")


def _resolve_path(value: str | None) -> Path | None:
    if not value:
        return None
    if str(value).lower() in {"none", "null", "skip"}:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _validate_checkpoint_config(path: Path | None, target_config: dict, label: str) -> Path | None:
    if path is None:
        return None
    run_config_path = path.parents[1] / "config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"{label} checkpoint has no sibling run config: {run_config_path}")
    try:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} run config is not valid JSON: {run_config_path}") from exc
    if not _checkpoint_config_matches(run_config, target_config):
        raise ValueError(
            f"{label} checkpoint config does not match the requested evaluation config: {path}"
        )
    return path


def _effective_obs_dim_base(config: dict) -> int:
    act_dim = int(config.get("rl", {}).get("act_dim", 2))
    history_cfg = dict(config.get("observation", {}).get("history", {}))
    history_steps = int(history_cfg.get("steps", 0)) if history_cfg.get("enabled", False) else 0
    action_obs_dim = act_dim if config.get("action_in_observation", False) else 0
    expected = 14 + action_obs_dim + history_steps * (14 + 2 * act_dim)
    return int(config.get("rl", {}).get("obs_dim_base", expected))


def _checkpoint_config_matches(run_config: dict, target_config: dict) -> bool:
    checks = [
        int(run_config.get("preview", {}).get("steps", -1)) == int(target_config.get("preview", {}).get("steps", -2)),
        float(run_config.get("preview", {}).get("dt", -1.0)) == float(target_config.get("preview", {}).get("dt", -2.0)),
        bool(run_config.get("action_in_observation", False)) == bool(target_config.get("action_in_observation", False)),
        _effective_obs_dim_base(run_config) == _effective_obs_dim_base(target_config),
        bool(run_config.get("imitation", {}).get("enabled", False))
        == bool(target_config.get("imitation", {}).get("enabled", False)),
        bool(run_config.get("policy_improvement_gate", {}).get("enabled", False))
        == bool(target_config.get("policy_improvement_gate", {}).get("enabled", False)),
        bool(run_config.get("policy_safety", {}).get("enabled", False))
        == bool(target_config.get("policy_safety", {}).get("enabled", False)),
        bool(run_config.get("policy_action_parameterization", {}).get("enabled", False))
        == bool(target_config.get("policy_action_parameterization", {}).get("enabled", False)),
    ]
    if bool(target_config.get("imitation", {}).get("enabled", False)):
        checks.extend(
            [
                str(run_config.get("imitation", {}).get("dataset", ""))
                == str(target_config.get("imitation", {}).get("dataset", "")),
                bool(run_config.get("imitation", {}).get("residual_targets", False))
                == bool(target_config.get("imitation", {}).get("residual_targets", False)),
            ]
        )
    return all(checks)


def _candidate_score(path: Path) -> tuple[float, float]:
    summary_path = path.parent / "best_summary.json"
    if path.name == "best.pt" and summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            return float(summary.get("mean_return", -float("inf"))), path.stat().st_mtime
        except json.JSONDecodeError:
            pass
    return -float("inf"), path.stat().st_mtime


def _latest_checkpoint_for_config(config_name: str, target_config: dict) -> Path | None:
    config_name = str((ROOT / config_name).resolve())
    candidates: list[Path] = []
    for manifest_path in (ROOT / "results").glob("*_ppo/training_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(manifest.get("config", "")) != config_name:
            continue
        run_config_path = manifest_path.parent / "config.json"
        if not run_config_path.is_file():
            continue
        try:
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not _checkpoint_config_matches(run_config, target_config):
            continue
        ckpt_dir = manifest_path.parent / "checkpoints"
        for name in ("final.pt", "best.pt", "episode_400.pt", "episode_200.pt", "episode_150.pt", "episode_50.pt"):
            path = ckpt_dir / name
            if path.is_file():
                candidates.append(path)
                break
    return max(candidates, key=_candidate_score) if candidates else None


def _controller_config(base_config: dict, controller_name: str) -> dict:
    cfg = deepcopy(base_config)
    return cfg


def _run_one(
    config: dict,
    scenario: dict,
    controller_name: str,
    label: str,
    checkpoint: Path | None,
    out_dir: Path,
) -> dict:
    is_rl = controller_name.lower() == "ppo"
    use_preview = is_rl or controller_name.lower() in set(config["preview"]["enabled_for"])
    cfg = _controller_config(config, controller_name)
    env = make_env(cfg, scenario=scenario, use_preview=use_preview)
    controller = make_controller(
        controller_name,
        env,
        cfg,
        checkpoint=str(checkpoint) if checkpoint else None,
        algorithm="ppo" if is_rl else None,
    )
    run = rollout(env, controller, use_preview=use_preview)
    if hasattr(env, "close"):
        env.close()

    df = run["data"]
    key = f"{label}_{scenario['name']}"
    df.to_csv(out_dir / f"{key}.csv", index=False)
    metrics = compute_metrics(df, float(config.get("settle_seconds", 1.0)))
    return {
        "Controller": label,
        "ControllerImpl": controller_name,
        "Scenario": scenario["name"],
        "ScenarioLabel": scenario.get("label", scenario["name"]),
        "PreviewUsed": bool(use_preview),
        "PreviewSteps": int(config["preview"]["steps"]) if use_preview else 0,
        "PreviewDt_s": float(config["preview"].get("dt", config["control_dt"])) if use_preview else 0.0,
        "Engine": str(config.get("environment", {}).get("engine", "python")),
        **metrics,
    }


def evaluate_ch6(
    base_config_path: Path,
    ppo_config_path: Path,
    si_config_path: Path,
    ppo_checkpoint: Path | None,
    si_checkpoint: Path | None,
    out_dir: Path,
    update_comparison: bool,
) -> pd.DataFrame:
    base_config = load_config(base_config_path)
    ppo_config = load_config(ppo_config_path)
    si_config = load_config(si_config_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    manifest: dict = {
        "base_config": str(base_config_path.resolve()),
        "ppo_config": str(ppo_config_path.resolve()),
        "si_config": str(si_config_path.resolve()),
        "ppo_checkpoint": str(ppo_checkpoint) if ppo_checkpoint else None,
        "si_checkpoint": str(si_checkpoint) if si_checkpoint else None,
        "controllers": list(CLASSICAL_CONTROLLERS),
        "scenarios": [s.get("name", "") for s in base_config["scenarios"]],
    }

    for scenario in base_config["scenarios"]:
        for controller_name in CLASSICAL_CONTROLLERS:
            rows.append(_run_one(base_config, scenario, controller_name, controller_name, None, out_dir))
        if ppo_checkpoint is not None:
            rows.append(_run_one(ppo_config, scenario, "ppo", "PPO", ppo_checkpoint, out_dir))
        if si_checkpoint is not None:
            rows.append(_run_one(si_config, scenario, "ppo", "SI-RPPO", si_checkpoint, out_dir))

    metrics = pd.DataFrame(rows)
    metrics_path = out_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    manifest["metric_columns"] = list(metrics.columns)
    manifest["trajectory_files"] = sorted(p.name for p in out_dir.glob("*.csv") if p.name != "metrics.csv")
    with (out_dir / "evaluation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    if update_comparison:
        shutil.copyfile(metrics_path, ROOT / "results" / "comprehensive_comparison.csv")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/half_car_safe_ppo.yaml")
    parser.add_argument("--ppo-config", dest="ppo_config", default="configs/ppo.yaml")
    parser.add_argument("--scratch-config", dest="ppo_config", help=argparse.SUPPRESS)
    parser.add_argument("--si-config", default="configs/half_car_safe_ppo.yaml")
    parser.add_argument("--ppo-checkpoint", dest="ppo_checkpoint", default="auto")
    parser.add_argument("--scratch-checkpoint", dest="ppo_checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--si-checkpoint", default="auto")
    parser.add_argument("--out", default=None)
    parser.add_argument("--update-comparison", action="store_true")
    args = parser.parse_args()

    ppo_config = load_config(ROOT / args.ppo_config)
    si_config = load_config(ROOT / args.si_config)
    ppo_checkpoint = (
        _latest_checkpoint_for_config(args.ppo_config, ppo_config)
        if args.ppo_checkpoint == "auto"
        else _resolve_path(args.ppo_checkpoint)
    )
    si_checkpoint = (
        _latest_checkpoint_for_config(args.si_config, si_config)
        if args.si_checkpoint == "auto"
        else _resolve_path(args.si_checkpoint)
    )
    ppo_checkpoint = _validate_checkpoint_config(ppo_checkpoint, ppo_config, "PPO")
    si_checkpoint = _validate_checkpoint_config(si_checkpoint, si_config, "SI-RPPO")
    out_dir = Path(args.out) if args.out else ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_ch6_unified")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    metrics = evaluate_ch6(
        ROOT / args.base_config,
        ROOT / args.ppo_config,
        ROOT / args.si_config,
        ppo_checkpoint,
        si_checkpoint,
        out_dir,
        bool(args.update_comparison),
    )
    print(metrics)
    print(f"Saved unified Chapter 6 evaluation to {out_dir}")
    if ppo_checkpoint is None:
        print("PPO checkpoint was not found; train it before final report regeneration.")
    if si_checkpoint is None:
        print("SI-RPPO checkpoint was not found; train it before final report regeneration.")


if __name__ == "__main__":
    main()
