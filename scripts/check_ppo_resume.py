"""Check CPU optimizer and RNG continuation against uninterrupted PPO updates."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
from rl.ppo import PPOAgent, PPOConfig, finish_ppo_trajectory  # noqa: E402


def main():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    np.random.seed(123)
    cfg = replace(
        PPOConfig.from_project_config(load_config(ROOT / "configs/cosim_ppo_literature.yaml")),
        transformer_dropout=0,
        train_epochs=2,
        minibatch_size=32,
    )
    agent = PPOAgent(cfg, torch.device("cpu"))

    def update(model):
        batch = {k: [] for k in ("obs", "act", "logp", "value", "rew", "done")}
        for _ in range(64):
            obs = np.random.normal(0, 0.2, cfg.obs_dim).astype(np.float32)
            act, logp, value = model.act_for_training(obs)
            for k, v in zip(
                batch, (obs, act, logp, value, -float(np.mean(obs**2)), False), strict=True
            ):
                batch[k].append(v)
        batch["ret"], batch["adv"] = finish_ppo_trajectory(
            batch["rew"], batch["value"], batch["done"], cfg.gamma, cfg.lam
        )
        model.train_trajectory(batch)
        return batch

    update(agent)
    output = ROOT / "results/ppo_resume_checks" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    agent.save(output / "checkpoint.pt")
    expected = update(agent)
    restored = PPOAgent(cfg, torch.device("cpu"))
    restored.load(output / "checkpoint.pt", resume_training=True)
    actual = update(restored)
    assert np.array_equal(expected["act"], actual["act"])
    assert np.array_equal(expected["obs"], actual["obs"])
    for name in ("actor", "value"):
        assert all(
            torch.equal(tensor, getattr(restored, name).state_dict()[key])
            for key, tensor in getattr(agent, name).state_dict().items()
        )
    (output / "report.json").write_text(
        json.dumps(
            dict(passed=True, scope="CPU optimizer and RNG exact next-rollout/update parity")
        ),
        encoding="utf-8",
    )
    print("PASS:", output / "report.json")


if __name__ == "__main__":
    main()
