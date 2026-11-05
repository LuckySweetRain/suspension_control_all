"""Verify owned rollout files before compacting successful training artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch


def inspect_episode(folder, agent):
    rows = [json.loads(s) for s in (folder / "transitions.jsonl").read_text().splitlines()]
    log = (folder / "simulator.log").read_text(encoding="utf-8")
    simulation = Path(
        [s[11:].strip() for s in log.splitlines() if s.startswith("Artifacts: ")][-1]
    ).resolve()
    report = json.loads((simulation / "report.json").read_text())
    physical = json.loads((simulation / "trace.json").read_text())
    if not report["passed"] or len(rows) != len(physical) or report["completed_steps"] != len(rows):
        raise ValueError("Incomplete physical rollout; retain artifacts")
    for i, (r, p) in enumerate(zip(rows, physical, strict=True)):
        if r["seq"] != i or r["terminated"] or r["truncated"] != (i == len(rows) - 1):
            raise ValueError("Invalid rollout boundary")
        if not np.isfinite(r["exports"]).all() or not np.isfinite(r["reward"]):
            raise ValueError("Nonfinite rollout")
        if not np.allclose(r["exports"], p["values"], rtol=0, atol=1e-9):
            raise ValueError("Native state mismatch")
        if not np.allclose(np.tile(r["corner_forces"], 2), p["forces"], rtol=0, atol=1e-6):
            raise ValueError("Native force mismatch")
    obs = torch.tensor(np.asarray([r["observation"] for r in rows]), dtype=torch.float32)
    act = torch.tensor(np.asarray([r["raw_action"] for r in rows]), dtype=torch.float32)
    with torch.no_grad():
        logp, _ = agent.log_prob(obs, act)
    error = float(abs(logp.cpu().numpy().ravel() - [r["logp"] for r in rows]).max())
    if error > 0.002:
        raise ValueError("Behavior policy likelihood mismatch")
    raw = np.asarray([r["raw_action"] for r in rows])
    executed = np.asarray([r["executed_action"] for r in rows])
    y = np.asarray([r["exports"] for r in rows])
    active = np.asarray([not r["supervisor"] for r in rows])
    # Native constraints remain visible even with zero constraint reward weights.
    mapping = [r.get("state_mapping_diagnostics") or {} for r in rows]
    metrics = dict(
        acceleration_rms_mps2=float(np.sqrt(np.mean((y[:, 40] * 9.80665) ** 2))),
        force_rms_N=float(np.sqrt(np.mean(executed**2))),
        min_native_tire_load_N=float(y[:, 52:56].min()),
        mapping_outside_fraction=float(np.mean([r.get("outside_bounds", 0) / 4 for r in mapping])),
        mapping_saturation_fraction=float(
            np.mean([r.get("numerical_saturations", 0) / 4 for r in mapping])
        ),
        projection_fraction=float(np.any(abs(raw[active] - executed[active]) > 1e-3, axis=1).mean())
        if active.any()
        else 0.0,
        **{
            "reward/" + k: float(np.mean([r["reward_terms"][k] for r in rows]))
            for k in rows[0]["reward_terms"]
        },
    )
    proof = dict(
        passed=True,
        steps=len(rows),
        behavior_logp_max_error=error,
        native_report=report,
        simulation=str(simulation),
        transitions_sha256=hashlib.sha256((folder / "transitions.jsonl").read_bytes()).hexdigest(),
        scope="online per-step native state/force and behavior likelihood verification before optional pruning",
    )
    if all("physical_constraints" in r for r in rows):
        constraints = [r["physical_constraints"] for r in rows]
        metrics.update(
            max_relative_travel_m=max(r["max_relative_travel_m"] for r in constraints),
            travel_violation_fraction=float(
                np.mean([r["travel_violation_fraction"] for r in constraints])
            ),
            low_load_fraction=float(np.mean([r["low_load_fraction"] for r in constraints])),
        )
    return metrics, proof


def prune_episode(folder, run, proof, root):
    folder, run, root = folder.resolve(), run.resolve(), root.resolve()
    simulation = Path(proof["simulation"]).resolve()
    # Resolve and verify both complete deletion targets before deleting either.
    if (
        folder.parent != run
        or not folder.name.startswith("episode_")
        or (folder / "failure.json").exists()
    ):
        raise ValueError("Refuse to prune an unowned/failed episode")
    if simulation.parent != root / "results/carla_environment":
        raise ValueError("Simulator artifact path escaped expected directory")
    if proof["native_report"].get("artifact_owner") != str(folder) or not proof["passed"]:
        raise ValueError("Missing simulator ownership/verification")
    shutil.rmtree(simulation)
    shutil.rmtree(folder)
