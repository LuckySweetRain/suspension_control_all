"""Audit completed physical rollouts, PPO behavior probabilities and saved weights.

An in-progress run is reported as incomplete, never as failed merely for lacking
its final checkpoint. No simulator is started and no policy is updated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl.ppo import PPOAgent, PPOConfig  # noqa: E402


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def audit(run):
    manifest = json.loads((run / "manifest.json").read_text())
    cfg = PPOConfig(**manifest["ppo"])
    torch.set_num_threads(2)
    torch.manual_seed(manifest["args"]["seed"])
    behavior = PPOAgent(cfg, torch.device("cpu"))
    behavior.actor.eval()
    results = []
    expected_episodes = manifest["args"]["episodes"]
    for index in range(expected_episodes):
        folder = run / f"episode_{index:04d}"
        if (folder / "failure.json").exists():
            results.append(dict(episode=index, passed=False, error="failure.json exists"))
            break
        if not (folder / "summary.json").exists():
            break
        result = dict(episode=index, passed=False)
        try:
            summary = json.loads((folder / "summary.json").read_text())
            rows = [
                json.loads(line) for line in (folder / "transitions.jsonl").read_text().splitlines()
            ]
            require(
                len(rows) == summary["steps"] == round(manifest["args"]["seconds"] * 100),
                "Incomplete control-step count",
            )
            previous = np.zeros(2)
            for step, row in enumerate(rows):
                require(row["seq"] == step, "Nonconsecutive action sequence")
                require(abs(row["time"] - (step + 1) * 0.01) < 1e-6, "Control clock mismatch")
                require(len(row["observation"]) == 104, "Observation dimension mismatch")
                require(len(row["exports"]) >= 81, "Missing physical state")
                require(
                    np.isfinite(row["observation"] + row["exports"] + row["raw_action"]).all(),
                    "Nonfinite trajectory",
                )
                require(
                    abs(sum(row["reward_terms"].values()) + row["reward"]) < 1e-8,
                    "Reward components do not sum to reward",
                )
                raw = np.asarray(row["raw_action"])
                executed = np.asarray(row["executed_action"])
                expected = (
                    np.zeros(2)
                    if row["supervisor"]
                    else np.clip(raw, previous - 150, previous + 150)
                )
                require(np.allclose(executed, expected, atol=1e-5), "Actuator projection mismatch")
                require(np.max(abs(raw)) <= cfg.act_limit + 1e-4, "Policy action exceeds bound")
                require(np.max(np.abs(row["corner_forces"])) <= 5000, "Corner force exceeds bound")
                require(not row["terminated"], "Failure termination counted as successful episode")
                require(row["truncated"] == (step == len(rows) - 1), "Time-limit boundary mismatch")
                previous = executed
            require(
                abs(sum(row["reward"] for row in rows) - summary["reward_sum"]) < 1e-6,
                "Summary return mismatch",
            )
            obs = torch.tensor(
                np.asarray([row["observation"] for row in rows]), dtype=torch.float32
            )
            act = torch.tensor(np.asarray([row["raw_action"] for row in rows]), dtype=torch.float32)
            with torch.inference_mode():
                recomputed, _ = behavior.log_prob(obs, act)
            error = float(np.max(abs(recomputed.numpy().ravel() - [row["logp"] for row in rows])))
            require(error < 0.002, f"Stored logp does not match behavior actor: {error}")
            checkpoint = folder / "checkpoint.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            require(saved["cfg"] == cfg.__dict__, "Checkpoint configuration drift")
            require(
                all(
                    bool(torch.isfinite(t).all())
                    for key in ("actor", "value")
                    for t in saved[key].values()
                ),
                "Nonfinite checkpoint",
            )
            changed = any(
                not torch.equal(t, saved["actor"][name])
                for name, t in behavior.actor.state_dict().items()
            )
            require(changed, "No actor update between successive checkpoints")
            log = (folder / "simulator.log").read_text(encoding="utf-8", errors="replace")
            artifact_lines = [line for line in log.splitlines() if line.startswith("Artifacts: ")]
            require(bool(artifact_lines), "Missing simulator report path")
            simulation = Path(artifact_lines[-1].removeprefix("Artifacts: ").strip())
            report = json.loads((simulation / "report.json").read_text())
            require(
                report["passed"] and report["completed_steps"] == len(rows),
                "Simulator did not complete",
            )
            require(
                report["terrain_callback_calls"] > 0 and report["image_updates"] > 0,
                "Missing actual CARLA terrain or image updates",
            )
            physical = json.loads((simulation / "trace.json").read_text())
            require(len(physical) == len(rows), "Simulator/learner trajectory lengths differ")
            for row, actual in zip(rows, physical, strict=True):
                require(
                    np.allclose(row["exports"], actual["values"], rtol=0, atol=1e-9),
                    "Learner next state differs from actual solver output",
                )
                require(
                    np.allclose(
                        np.tile(row["corner_forces"], 2), actual["forces"], rtol=0, atol=1e-6
                    ),
                    "Recorded action differs from forces sent to solver",
                )
            result.update(
                passed=True,
                steps=len(rows),
                logp_max_error=error,
                actor_changed=True,
                simulator_report=str(simulation / "report.json"),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
            behavior.load(checkpoint)
            behavior.actor.eval()
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            result["error"] = str(error)
        results.append(result)
        if not result["passed"]:
            break
    complete = len(results) == expected_episodes and all(row["passed"] for row in results)
    final_matches = False
    if complete and (run / "final.pt").exists():
        final = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
        final_matches = all(
            torch.equal(t, final["actor"][name]) for name, t in behavior.actor.state_dict().items()
        )
    output = dict(
        run=str(run),
        expected_episodes=expected_episodes,
        audited=results,
        run_complete=complete and final_matches,
        final_actor_matches=final_matches,
        scope="physical rollout completion and PPO data integrity; not performance superiority",
    )
    (run / "audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = audit(args.run)
    print(json.dumps(report, indent=2))
    if any(not row["passed"] for row in report["audited"]):
        raise SystemExit(1)
