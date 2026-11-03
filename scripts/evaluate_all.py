"""多控制器评估入口：读取配置和显式权重，输出指标、轨迹及图表。

从仓库根目录运行；历史实验复现应指定权重路径，避免 latest 随新训练变化。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from experiments.evaluation import evaluate_all
from visualization.plots import plot_all


def _latest_checkpoint_candidates(algorithm: str) -> list[Path]:
    """按修改时间倒序查找根层训练权重；归档目录不参与自动选择。

    保留此历史兼容规则；整理结果时不能移动候选或改写其修改时间。
    """
    return sorted(
        (ROOT / "results").glob(f"*_{algorithm}/checkpoints/final.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def resolve_checkpoint(value: str | None, algorithm: str = "td3") -> str | None:
    if not value:
        return None
    if "<run>" in value:
        raise ValueError(
            "Checkpoint path still contains the placeholder '<run>'. "
            "Use a real run directory, for example "
            "results\\20260517_180625_td3\\checkpoints\\final.pt."
        )
    if value.lower() == "latest":
        candidates = _latest_checkpoint_candidates(algorithm)
        if not candidates:
            pattern = f"results\\*_{algorithm}\\checkpoints\\final.pt"
            raise FileNotFoundError(f"No {algorithm.upper()} checkpoint found under {pattern}")
        return str(candidates[0])
    checkpoint = Path(value)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return str(checkpoint)


def resolve_checkpoints(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    resolved = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(
                "Use --checkpoints as algorithm=path pairs, for example td3=latest,sac=latest"
            )
        algorithm, checkpoint = item.split("=", 1)
        algorithm = algorithm.strip().lower()
        if algorithm not in {"td3", "ddpg", "sac", "ppo"}:
            raise ValueError(
                f"Unsupported checkpoint algorithm: {algorithm!r}; choose td3, ddpg, sac, or ppo"
            )
        if algorithm in resolved:
            raise ValueError(f"Duplicate checkpoint algorithm: {algorithm}")
        if not checkpoint.strip():
            raise ValueError(f"Checkpoint path cannot be empty for {algorithm}")
        resolved[algorithm] = resolve_checkpoint(checkpoint.strip(), algorithm=algorithm)
    return resolved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="Comma-separated algorithm=checkpoint pairs, e.g. td3=latest,sac=latest,ppo=latest",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    checkpoints = resolve_checkpoints(args.checkpoints)
    result_dir = (
        Path(args.out)
        if args.out
        else ROOT / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_eval")
    )
    metrics, _ = evaluate_all(config, checkpoints, result_dir)
    print(metrics)
    plot_all(result_dir)
    print(f"Saved evaluation to {result_dir}")


if __name__ == "__main__":
    main()
