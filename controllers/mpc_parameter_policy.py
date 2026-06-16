from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from envs.mpc_parameter_env import MPCParameterPolicyEnv
from rl.mpc_parameter_policy import MPCParameterGaussianPolicy

from .base import Controller
from .reduced_full_car import ReducedFullCarPreviewController


class NeuralMPCParameterController(Controller):
    """Reduced full-car MPC-lite with gains scheduled by a neural parameter policy."""

    name = "BP_NMPS"

    def __init__(self, env, config: dict, checkpoint: str | Path | None = None, device: str | None = None):
        ckpt_path = checkpoint or config.get("mpc_parameter_policy", {}).get("checkpoint")
        if not ckpt_path:
            raise ValueError("NeuralMPCParameterController requires an MPC parameter policy checkpoint.")
        self.env = env
        self.base_config = copy.deepcopy(config)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            data = torch.load(Path(ckpt_path), map_location=self.device, weights_only=True)
        except TypeError:  # pragma: no cover - older torch compatibility
            data = torch.load(Path(ckpt_path), map_location=self.device)
        self.policy = MPCParameterGaussianPolicy(
            obs_dim=int(data["obs_dim"]),
            act_dim=int(data["act_dim"]),
            hidden_sizes=list(data.get("hidden_sizes", [64, 64])),
        ).to(self.device)
        self.policy.load_state_dict(data["model_state_dict"])
        self.policy.eval()
        self.helper = MPCParameterPolicyEnv(
            config=self.base_config,
            scale_low=float(data.get("scale_low", 0.5)),
            scale_high=float(data.get("scale_high", 1.5)),
        )
        self.deployment_gate_active = bool(data.get("deployment_gate_active", False))
        self.deployment_gate = dict(data.get("deployment_gate", {}))
        self.current_scenario_name: str | None = None
        self.last_policy_action = np.zeros(self.helper.action_space.shape, dtype=np.float32)
        self.last_gains = dict(self.helper.base_gains)
        self.inner = ReducedFullCarPreviewController(env, self.base_config)

    def reset(self):
        self.current_scenario_name = None
        self.inner.reset()

    def _scenario(self):
        return getattr(self.env, "scenario", None)

    def _scenario_name(self) -> str:
        scenario = self._scenario()
        return str(scenario.get("name", "")) if isinstance(scenario, dict) else ""

    def _update_inner_controller(self):
        scenario = self._scenario()
        scenario_name = self._scenario_name()
        if scenario_name == self.current_scenario_name:
            return
        if self.deployment_gate_active:
            cfg = copy.deepcopy(self.base_config)
            cfg["reduced_full_car_mpc"] = {**self.helper.base_gains}
            self.inner = ReducedFullCarPreviewController(self.env, cfg)
            self.current_scenario_name = scenario_name
            self.last_policy_action = np.zeros(self.helper.action_space.shape, dtype=np.float32)
            self.last_gains = dict(self.helper.base_gains)
            return
        self.helper.active_scenario = scenario if isinstance(scenario, dict) else None
        obs = torch.as_tensor(self.helper._obs(), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.policy.sample(obs, deterministic=True)
        action_np = action.cpu().numpy()[0].astype(np.float32)
        gains = self.helper.action_to_gains(action_np)
        cfg = copy.deepcopy(self.base_config)
        cfg["reduced_full_car_mpc"] = {**self.helper.base_gains, **gains}
        self.inner = ReducedFullCarPreviewController(self.env, cfg)
        self.current_scenario_name = scenario_name
        self.last_policy_action = action_np
        self.last_gains = gains

    def compute_action(self, obs, info) -> np.ndarray:
        self._update_inner_controller()
        return self.inner.compute_action(obs, info)
