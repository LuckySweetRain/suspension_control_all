"""Reward direction, bridge serialization, PPO update and KL rejection checks (no pytest)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from rl.cosim_reward import reward_terms, validate_reward  # noqa: E402
from rl.ppo import PPOAgent, PPOConfig, finish_ppo_trajectory  # noqa: E402
from scripts.train_cosim_ppo import Episode  # noqa: E402


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261104)
    project = load_config(ROOT / "configs/cosim_ppo_literature.yaml")
    reward = project["cosim_reward"]
    validate_reward(reward)
    origin = np.zeros(81)
    origin[52:56] = 3000
    zero = np.zeros(2)

    def terms(y, raw=zero, executed=zero, supervisor=False):
        return reward_terms(y, origin, raw, executed, zero, supervisor, reward)

    assert sum(terms(origin).values()) == 0
    acceleration = origin.copy()
    acceleration[40] = 2 / 9.80665
    doubled = acceleration.copy()
    doubled[40] *= 2
    assert np.isclose(terms(doubled)["accel"], 4 * terms(acceleration)["accel"])
    unsafe = origin.copy()
    unsafe[44:48] = 90
    unsafe[52:56] = 0
    assert terms(unsafe)["violation"] > 0 and terms(unsafe)["unload"] > 0
    assert terms(origin, raw=np.ones(2) * 1000)["projection"] > 0
    assert terms(origin, raw=np.ones(2) * 1000, supervisor=True)["projection"] == 0
    cfg = PPOConfig.from_project_config(project)
    cfg.obs_dim_base = 24
    cfg.preview_steps = 20
    cfg.preview_token_dim = 4
    cfg.transformer_dropout = 0
    agent = PPOAgent(cfg, torch.device("cpu"))
    # Exercise the actual bridge's normalized observation, projection and reward serialization.
    stream = io.StringIO()
    episode = Episode(agent, stream, reward_config=reward)
    episode.handle(
        dict(
            version=1,
            op="act",
            seq=0,
            time=0,
            exports=origin.tolist(),
            preview=np.zeros((20, 4)).tolist(),
            supervisor=False,
        )
    )
    episode.handle(
        dict(
            version=1,
            op="transition",
            seq=0,
            time=0.01,
            exports=acceleration.tolist(),
            terminated=False,
            truncated=True,
        )
    )
    record = json.loads(stream.getvalue())
    assert np.isclose(record["reward"], -sum(record["reward_terms"].values()))
    trajectory = {k: [] for k in ("obs", "act", "logp", "value", "rew", "done")}
    for _ in range(128):
        obs = (
            np.random.default_rng(len(trajectory["obs"]))
            .normal(0, 0.2, cfg.obs_dim)
            .astype(np.float32)
        )
        action, logp, value = agent.act_for_training(obs)
        for k, v in zip(trajectory, (obs, action, logp, value, -0.05, False), strict=True):
            trajectory[k].append(v)
    trajectory["ret"], trajectory["adv"] = finish_ppo_trajectory(
        trajectory["rew"], trajectory["value"], trajectory["done"], cfg.gamma, cfg.lam
    )
    before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    losses = agent.train_trajectory(trajectory)
    assert np.isfinite(losses).all() and agent.last_train_stats["optimizer_steps"] > 0
    assert any(not torch.equal(before[k], v) for k, v in agent.actor.state_dict().items())
    corrupted = dict(trajectory, logp=(np.asarray(trajectory["logp"]) + 10).tolist())
    before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    agent.train_trajectory(corrupted)
    assert (
        agent.last_train_stats["kl_early_stop"] and agent.last_train_stats["optimizer_steps"] == 0
    )
    assert all(torch.equal(before[k], v) for k, v in agent.actor.state_dict().items())
    print(
        "PASS: equilibrium, quadratic acceleration, constraints, projection exemption, bridge reward, finite PPO update, KL rejection"
    )


if __name__ == "__main__":
    main()
