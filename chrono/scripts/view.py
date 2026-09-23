"""View a live controller or replay a recorded trajectory in a native window."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chrono_suspension.config import load_config  # noqa: E402
from chrono_suspension.roads import ROAD_KINDS  # noqa: E402
from chrono_suspension.visualization import NativeViewer, result_path, validate_frame  # noqa: E402


def _json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_frame(stream: Any, frame: dict[str, Any]) -> None:
    validate_frame(frame)
    stream.write(json.dumps(frame, default=_json_value, allow_nan=False) + "\n")


def read_trajectory(path: Path) -> tuple[dict | None, list[dict]]:
    """Read versioned JSONL or plain export_frame() lines without re-simulation."""
    metadata = None
    frames: list[dict] = []
    last_time = -math.inf
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Trajectory line {line_number} must be an object")
            if item.get("kind") == "metadata":
                if frames or metadata is not None:
                    raise ValueError("Trajectory metadata must appear once before all frames")
                if item.get("schema") != "chrono-suspension-frame-v1":
                    raise ValueError("Unsupported trajectory schema")
                metadata = item
                continue
            validate_frame(item)
            timestamp = float(item["time_s"])
            if timestamp < last_time:
                raise ValueError(f"Time decreases at trajectory line {line_number}")
            last_time = timestamp
            frames.append(item)
    if not frames:
        raise ValueError("Trajectory has no frames")
    return metadata, frames


def _view_options(cfg: dict, args: argparse.Namespace) -> dict:
    cfg["visualization"] = {
        **cfg.get("visualization", {}),
        "speed": args.speed,
        "camera": args.camera,
        "cutaway": not args.body_shell,
        "paused": args.paused,
    }
    return cfg


def _save_image(viewer: NativeViewer, path: str | None) -> None:
    if path:
        print(f"Screenshot: {viewer.screenshot(path)}")


def replay(args: argparse.Namespace) -> None:
    metadata, frames = read_trajectory(Path(args.replay))
    source_cfg = metadata.get("config") if metadata else None
    cfg = load_config(args.config if args.config else source_cfg)
    viewer = NativeViewer(None, _view_options(cfg, args))
    index = 0
    frame_count = 0
    image_saved = False
    playback_time = float(frames[0]["time_s"])
    previous_wall = time.perf_counter()
    try:
        while True:
            frame_start = time.perf_counter()
            elapsed = min(frame_start - previous_wall, 0.25)
            previous_wall = frame_start
            if not viewer.paused:
                playback_time += elapsed * viewer.speed
                while index + 1 < len(frames) and frames[index + 1]["time_s"] <= playback_time:
                    index += 1
            if not viewer.render(frames[index]):
                break
            frame_count += 1
            last_frame = index == len(frames) - 1
            if last_frame and args.screenshot and not image_saved:
                _save_image(viewer, args.screenshot)
                image_saved = True
            if args.max_frames and frame_count >= args.max_frames or last_frame and not args.hold:
                if not image_saved:
                    _save_image(viewer, args.screenshot)
                break
            time.sleep(max(0, 1.0 / args.fps - (time.perf_counter() - frame_start)))
    finally:
        viewer.close()


def live(args: argparse.Namespace) -> None:
    from chrono_suspension.env import ChronoSuspensionEnv

    observation_mode = "privileged"
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).resolve()
        metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
        source_cfg = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        observation_mode = metadata["observation_mode"]
        cfg = load_config(args.config if args.config else source_cfg)
    else:
        cfg = load_config(args.config)
    if args.duration is not None:
        cfg["physics"]["episode_seconds"] = args.duration
    cfg = load_config(cfg)
    _view_options(cfg, args)
    env = ChronoSuspensionEnv(config=cfg, observation_mode=observation_mode, render_mode=None)
    viewer = None
    record = None
    try:
        options = {"scenario": {"kind": args.scenario}} if args.scenario else None
        observation, reset_info = env.reset(seed=args.seed, options=options)
        predict = None
        if args.checkpoint:
            from chrono_suspension.training import load_policy

            predict = load_policy(checkpoint, env, device="cpu")
        if args.record:
            path = result_path(args.record)
            record = path.open("x", encoding="utf-8")
            record.write(
                json.dumps(
                    {
                        "kind": "metadata",
                        "schema": "chrono-suspension-frame-v1",
                        "config": cfg,
                        "seed": args.seed,
                        "scenario": options,
                        "resolved_road": reset_info["road_metadata"],
                        "controller": args.checkpoint or args.controller,
                        "coordinates": "x_forward_y_left_z_up; quaternion_wxyz; SI",
                    },
                    default=_json_value,
                    allow_nan=False,
                )
                + "\n"
            )
            print(f"Recording: {path}")
        frame = env.export_frame()
        if record:
            _write_frame(record, frame)
        if not args.no_window:
            viewer = NativeViewer(env.model, cfg)
        dt = float(cfg["physics"]["control_dt"])
        elapsed_budget = 0.0
        previous_wall = time.perf_counter()
        frame_count = 0
        ended = False
        image_saved = False
        while True:
            frame_start = time.perf_counter()
            elapsed = min(frame_start - previous_wall, 0.25)
            previous_wall = frame_start
            if viewer is None:
                elapsed_budget = dt
            elif not viewer.paused and not ended:
                elapsed_budget += elapsed * viewer.speed
            while elapsed_budget >= dt and not ended:
                action = predict(observation) if predict else env.baseline_action(args.controller)
                observation, _, terminated, truncated, _ = env.step(action)
                ended = terminated or truncated
                frame = env.export_frame()
                if record:
                    _write_frame(record, frame)
                elapsed_budget -= dt
            if viewer is not None and not viewer.render(frame):
                break
            frame_count += 1
            if ended and viewer is not None and args.screenshot and not image_saved:
                _save_image(viewer, args.screenshot)
                image_saved = True
            limit_reached = bool(args.max_frames and frame_count >= args.max_frames)
            if limit_reached or ended and (not args.hold or viewer is None):
                if viewer is not None and not image_saved:
                    _save_image(viewer, args.screenshot)
                break
            if viewer is not None:
                time.sleep(max(0, 1.0 / args.fps - (time.perf_counter() - frame_start)))
    finally:
        if record is not None:
            record.close()
        if viewer is not None:
            viewer.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", help="Optional simulation YAML; defaults to saved config on replay"
    )
    parser.add_argument("--controller", choices=("passive", "skyhook", "lqr"), default="skyhook")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--checkpoint", help="SB3 checkpoint directory with normalization statistics")
    mode.add_argument("--replay", help="JSONL trajectory; does not advance dynamics")
    parser.add_argument("--scenario", choices=ROAD_KINDS, help="Road kind override")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, help="Episode duration in seconds (live only)")
    parser.add_argument("--record", help="Write a new JSONL trajectory within results/")
    parser.add_argument("--screenshot", help="Save final rendered frame within results/")
    parser.add_argument(
        "--no-window", action="store_true", help="Record a live run without graphics"
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed, 0.05 to 4.0")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--camera", choices=("iso", "side", "front", "top"), default="iso")
    parser.add_argument(
        "--body-shell", action="store_true", help="Start with vehicle shell visible"
    )
    parser.add_argument("--paused", action="store_true")
    parser.add_argument(
        "--hold", action="store_true", help="Keep final state visible until window closes"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Limit rendering frames for smoke checks"
    )
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0 or not 0.05 <= args.speed <= 4:
        parser.error("fps must be positive and finite; speed must be in [0.05, 4.0]")
    if args.max_frames < 0 or args.seed < 0:
        parser.error("max-frames and seed must be nonnegative")
    if args.no_window and (args.replay or args.screenshot or not args.record):
        parser.error("--no-window requires --record, and cannot use --replay or --screenshot")
    if args.replay and (args.record or args.scenario or args.duration is not None):
        parser.error(
            "Replay uses recorded states; record, scenario and duration apply to live runs"
        )
    if args.screenshot and Path(args.screenshot).exists():
        parser.error("Screenshot already exists; select a new output path")
    if args.record and Path(args.record).exists():
        parser.error("Recording already exists; select a new output path")
    if args.replay:
        replay(args)
    else:
        live(args)


if __name__ == "__main__":
    main()
