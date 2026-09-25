"""Audit reference-controller signs and local linear stability; no pytest.

From the repository root, using the configured chrono-rl Conda environment:
    python chrono/scripts/check_reference_controllers.py
    python chrono/scripts/check_reference_controllers.py --output \
        results/chrono/comparison/controller_audit_20260925.json

This does not run or tune on any road. It checks every predeclared candidate
against the same static D-Class tangent. The augmented calculation includes
the first-order actuator lag and integer control-period command delay using
exact zero-order-hold discretization of the linear continuous model. It is an
independent approximation of the actual Chrono integrator: finite-angle
geometry, force/force-rate/power limits, unilateral tires and stop tables are
not included in the eigenvalue calculation. Stability here is local only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import expm

PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY = PROJECT.parent
sys.path.insert(0, str(PROJECT))

from chrono_suspension.comparison import candidate_controllers, selected_config  # noqa: E402
from chrono_suspension.config import load_config  # noqa: E402
from chrono_suspension.controllers import linear_matrices, lqr_gain  # noqa: E402


def feedback_matrix(config: dict, baseline: str, jacobian: np.ndarray) -> np.ndarray:
    """Return K for force = -K [q, qdot], in FL/FR/RL/RR order."""
    if baseline == "lqr":
        return lqr_gain(config)
    gain = np.zeros((4, 14))
    if baseline == "skyhook":
        damping = config["controllers"]["skyhook"]["damping_nspm"]
        assert np.isfinite(damping) and damping > 0
        gain[:, 7:10] = damping * jacobian[:, :3]
        # Positive sprung heave velocity must command negative separating force.
        heave_velocity = np.zeros(14)
        heave_velocity[7] = 1.0
        np.testing.assert_allclose(-gain @ heave_velocity, -damping * np.ones(4))
    elif baseline != "passive":
        raise ValueError(f"Unknown baseline {baseline!r}")
    return gain


def sampled_closed_loop(
    a: np.ndarray, b: np.ndarray, gain: np.ndarray, config: dict
) -> tuple[np.ndarray, int]:
    """Sampled state is [q, qdot, actual_force, queued_commands_oldest_first]."""
    actuator = config["actuator"]
    period = config["physics"]["control_dt"]
    tau = actuator["time_constant_s"]
    delay_samples = int(round(actuator["delay_s"] / period))
    if not np.isclose(delay_samples * period, actuator["delay_s"], rtol=0, atol=1e-12):
        raise ValueError("This audit requires command delay to be an integer control period")
    continuous_a = np.block([[a, b], [np.zeros((4, 14)), -np.eye(4) / tau]])
    continuous_b = np.vstack([np.zeros((14, 4)), np.eye(4) / tau])
    transition = expm(np.block([[continuous_a, continuous_b], [np.zeros((4, 22))]]) * period)
    discrete_a, discrete_b = transition[:18, :18], transition[:18, 18:]
    augmented_gain = np.hstack([gain, np.zeros((4, 4))])
    if delay_samples == 0:
        return discrete_a - discrete_b @ augmented_gain, delay_samples
    closed = np.zeros((18 + 4 * delay_samples, 18 + 4 * delay_samples))
    closed[:18, :18] = discrete_a
    closed[:18, 18:22] = discrete_b
    if delay_samples > 1:
        closed[18:-4, 22:] = np.eye(4 * (delay_samples - 1))
    closed[-4:, :18] = -augmented_gain
    return closed, delay_samples


def encode_roots(roots: np.ndarray) -> list[dict]:
    return [
        {"real": float(value.real), "imaginary": float(value.imag)}
        for value in sorted(roots, key=lambda value: (value.real, value.imag))
    ]


def audit_controllers(config: dict) -> dict:
    original = deepcopy(config)
    plant = {key: value for key, value in config.items() if key != "controllers"}
    results = []
    for candidate in candidate_controllers():
        selected = selected_config(config, candidate)
        assert {key: value for key, value in selected.items() if key != "controllers"} == plant
        a, b, jac = linear_matrices(selected)
        gain = feedback_matrix(selected, candidate["baseline"], jac)
        assert gain.shape == (4, 14) and np.isfinite(gain).all()
        # Virtual work: positive force separates the chassis and wheel, so its
        # generalized force is J.T F and power is F.T J qdot near equilibrium.
        v = config["vehicle"]
        mass = np.array([v["sprung_mass_kg"], *v["inertia_kgm2"][:2], *v["unsprung_mass_kg"]])
        np.testing.assert_allclose(b[7:] * mass[:, None], jac.T, rtol=0, atol=1e-12)
        assert np.all(b[7, :] > 0)
        np.testing.assert_allclose(np.diag(b[10:, :]), -1 / mass[3:], rtol=0, atol=1e-12)
        roots = np.linalg.eigvals(a - b @ gain)
        largest_real = float(np.max(roots.real))
        assert largest_real < 0, candidate["id"]
        augmented, delay_samples = sampled_closed_loop(a, b, gain, selected)
        discrete_roots = np.linalg.eigvals(augmented)
        spectral_radius = float(np.max(np.abs(discrete_roots)))
        assert spectral_radius < 1, candidate["id"]
        results.append(
            {
                "candidate": candidate,
                "mechanical_max_real_eigenvalue_per_s": largest_real,
                "mechanical_eigenvalues_per_s": encode_roots(roots),
                "lag_delay_sampled_spectral_radius": spectral_radius,
                "lag_delay_sampled_eigenvalues": encode_roots(discrete_roots),
                "sampled_state_dimension": int(augmented.shape[0]),
                "delay_control_samples": delay_samples,
                "feedback_gain": gain.tolist(),
            }
        )
    assert config == original
    return {
        "passed": True,
        "candidate_count": len(results),
        "checks": [
            "all candidates preserve identical non-controller configuration",
            "Skyhook damping is finite/positive and opposes sprung heave velocity",
            "positive actuator force raises the body and lowers its unsprung mass",
            "input matrix agrees with suspension virtual work and mechanical power sign",
            "every unsaturated mechanical A-BK is Hurwitz",
            "every lag/delay augmented sampled linear model is Schur stable",
            "controller selection leaves the input configuration unmodified",
        ],
        "physics_dt_s": config["physics"]["dt"],
        "control_dt_s": config["physics"]["control_dt"],
        "actuator_time_constant_s": config["actuator"]["time_constant_s"],
        "actuator_delay_s": config["actuator"]["delay_s"],
        "derivation": {
            "mechanical": "x_dot = A*x+B*f; x=[q(7),qdot(7)]; commanded u=-K*x.",
            "lag": "f_dot=(u_delayed-f)/tau; augmented continuous state [x,f].",
            "discretization": "Exact matrix exponential at control_dt, with zero-order-hold input.",
            "delay": "Append a queue of commands; apply oldest and append -K*x each sample.",
            "criteria": "max(real(eig(A-BK)))<0 and max(abs(eig(sampled_augmented)))<1.",
        },
        "limitations": [
            "No road outcomes enter this check and no controller gains are tuned here.",
            "These are local linear checks, not a nonlinear/global stability proof.",
            "Exact continuous-model discretization differs from Chrono's physics integrator.",
            "Force/rate/power saturation, contact loss and nonlinear stops are excluded.",
            "Ideal full-state feedback is assumed, without sensor noise or state estimation.",
        ],
        "candidates": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT / "configs/carsim_dclass.yaml"))
    parser.add_argument("--output")
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        Path(args.output)
        if args.output
        else REPOSITORY / "results/chrono/comparison" / f"controller_audit_{timestamp}.json"
    ).resolve()
    if (REPOSITORY / "results").resolve() not in output.parents:
        raise ValueError("Generated controller checks must be under repository results/")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    config = load_config(args.config)
    report = audit_controllers(config)
    report["created_at_utc"] = timestamp
    report["resolved_config"] = config
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_root = PROJECT / "chrono_suspension"
    report["source_sha256"] = {
        name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        for name in ("controllers.py", "config.py", "comparison.py", "constitutive.py")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PASS: {report['candidate_count']} candidates; all 7 checks passed")
    for candidate in report["candidates"]:
        print(
            f"{candidate['candidate']['id']}: "
            f"max_real={candidate['mechanical_max_real_eigenvalue_per_s']:.6f}/s; "
            f"sampled_radius={candidate['lag_delay_sampled_spectral_radius']:.6f}"
        )
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
