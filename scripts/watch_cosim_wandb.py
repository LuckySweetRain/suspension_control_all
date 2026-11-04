"""Mirror scalar histories to W&B independently of the simulation/learner process."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def events(root, offset):
    manifest = read_json(root / "manifest.json")
    if manifest is None:
        return []
    steps = round(manifest["args"]["seconds"] * 100)
    history = root / "history.jsonl"
    if history.exists():
        records = []
        for line in history.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break  # A concurrent append may not have finished yet.
    else:
        records = [read_json(p) for p in sorted(root.glob("episode_*/summary.json"))]
    output = []
    for row in records:
        if row is None:
            continue
        step = offset + (row["episode"] + 1) * steps
        metrics = {"train/return": row["reward_sum"], "train/episode_steps": row["steps"]}
        if row.get("losses") is not None:
            metrics.update({"loss/value": row["losses"][0], "loss/actor": row["losses"][1]})
        metrics.update({"ppo/" + k: v for k, v in (row.get("ppo_diagnostics") or {}).items()})
        metrics.update({"train/" + k: v for k, v in row.get("metrics", {}).items()})
        output.append((step, str(root) + ":train:" + str(row["episode"]), metrics))
    for row in read_json(root / "evaluation_history.json", []):
        label = row["label"]
        completed = int(label.split("_")[1]) if label.startswith("after_") else 0
        prefix = "baseline/" if label == "passive" else "eval/"
        metrics = {
            prefix + k: v for k, v in row.items() if isinstance(v, (float, int)) and k != "episode"
        }
        output.append((offset + completed * steps, str(root) + ":eval:" + label, metrics))
    return sorted(output, key=lambda item: (item[0], item[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--step-offset", action="append", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", default="suspension-control")
    parser.add_argument("--entity")
    parser.add_argument("--name", default="PPO literature 300k")
    parser.add_argument("--mode", choices=("online", "offline"), default="online")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if len(args.run_dir) != len(args.step_offset):
        parser.error("Each run requires its matching cumulative step offset")
    import wandb

    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "sync_state.json"
    state = read_json(state_path, {})
    first = read_json(args.run_dir[0] / "manifest.json", {})
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        id=state.get("run_id"),
        resume="allow",
        mode=args.mode,
        dir=str(args.output),
        config={
            "ppo": first.get("ppo"),
            "reward": first.get("reward_config"),
            "observation": first.get("observation_schema"),
            "planned_steps": first.get("args", {}).get("episodes", 0)
            * round(first.get("args", {}).get("seconds", 0) * 100),
            "source": "verified local scalar summaries",
        },
        settings=wandb.Settings(
            base_url="https://api.wandb.ai",
            disable_git=True,
            disable_code=True,
            console="off",
            x_disable_stats=True,
        ),
    )
    run.define_metric("training_steps")
    for prefix in ("train/*", "loss/*", "ppo/*", "eval/*", "baseline/*"):
        run.define_metric(prefix, step_metric="training_steps")
    seen = set(state.get("uploaded", []))
    state.update(run_id=run.id, url=run.url, pid=os.getpid())
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run.id, "url": run.url}), flush=True)
    try:
        while True:
            pending = []
            for root, offset in zip(args.run_dir, args.step_offset, strict=True):
                pending.extend(events(root, offset))
            for step, key, metrics in sorted(pending, key=lambda item: (item[0], item[1])):
                if key not in seen:
                    run.log({"training_steps": step, **metrics})
                    seen.add(key)
            state.update(uploaded=sorted(seen), heartbeat=time.time())
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            status = read_json(args.run_dir[-1] / "status.json", {})
            if (
                args.once
                or status.get("stage") == "completed"
                or status.get("stage", "").startswith("failed")
            ):
                run.summary["training_status"] = status.get("stage", "import_only")
                break
            time.sleep(5)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
