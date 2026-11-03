"""Export deterministic raw axis-force inference; actuator projection remains external."""

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


class Policy(torch.nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, observation):
        mean, _ = self.actor(observation)
        return self.actor.act_limit * torch.tanh(mean)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    agent = PPOAgent(PPOConfig(**checkpoint["cfg"]), torch.device("cpu"))
    agent.load(args.checkpoint)
    model = Policy(agent.actor).eval()
    rows = [json.loads(line) for line in args.transitions.read_text().splitlines()]
    observations = np.asarray([row["observation"] for row in rows], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 104:
        raise ValueError("Expected cosim-ppo-v1 observations")
    with torch.inference_mode():
        inputs = [torch.from_numpy(row[None]) for row in observations]
        expected = np.concatenate([model(item).numpy() for item in inputs])
        traced = torch.jit.trace(model, inputs[0], check_inputs=[(inputs[-1],)])
        traced.save(str(args.output / "policy.ts"))
        restored = torch.jit.load(str(args.output / "policy.ts")).eval()
        actual = np.concatenate([restored(item).numpy() for item in inputs])
    error = float(np.max(abs(actual - expected)))
    if error > 0.01:
        raise RuntimeError(f"Export force mismatch: {error} N")
    np.save(args.output / "observations.npy", observations)
    np.save(args.output / "expected_forces.npy", expected)
    manifest = dict(
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256((args.output / "policy.ts").read_bytes()).hexdigest(),
        schema="cosim-ppo-v1",
        input_shape=[1, 104],
        output_shape=[1, 2],
        output="raw front/rear active axis forces in N; projection and damping mapping external",
        samples=len(observations),
        max_export_error_N=error,
        torch_version=torch.__version__,
        cfg=checkpoint["cfg"],
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
