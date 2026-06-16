from __future__ import annotations

import math

import torch
import torch.nn as nn

from .networks import mlp


class MPCParameterGaussianPolicy(nn.Module):
    """Gaussian policy for normalized MPC-lite gain scaling actions."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list[int] | tuple[int, ...] = (64, 64),
        log_std_init: float = -0.3,
        zero_init_mean: bool = True,
    ):
        super().__init__()
        hidden = list(hidden_sizes)
        self.trunk = mlp(obs_dim, hidden, hidden[-1] if hidden else obs_dim)
        last_dim = hidden[-1] if hidden else obs_dim
        self.mu = nn.Linear(last_dim, act_dim)
        if zero_init_mean:
            nn.init.zeros_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(obs)
        mu = torch.tanh(self.mu(x))
        log_std = torch.clamp(self.log_std, -5.0, 1.0).expand_as(mu)
        return mu, log_std

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        mu, log_std = self(obs)
        if deterministic:
            action = mu
            entropy = torch.zeros((obs.shape[0], 1), dtype=obs.dtype, device=obs.device)
            return action, None, entropy

        std = log_std.exp()
        noise = torch.randn_like(std)
        raw = mu + std * noise
        action = torch.clamp(raw, -1.0, 1.0)
        log_prob = -0.5 * (((raw - mu) / (std + 1e-8)).pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
        entropy = (0.5 + 0.5 * math.log(2.0 * math.pi) + log_std).sum(dim=-1, keepdim=True)
        return action, log_prob.sum(dim=-1, keepdim=True), entropy
