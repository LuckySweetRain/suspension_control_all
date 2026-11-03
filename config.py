"""主研究工程的配置入口：合并 YAML 继承并校验半车观测与动作维度。

父配置相对当前 YAML 解析；算法覆盖返回副本，不修改调用方的基线配置。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict, override: dict) -> dict:
    """递归合并字典并复制可变值；列表整体替换，避免实验之间共享配置状态。"""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 并从父层到子层合并；循环继承或非字典根节点抛出 ValueError。"""
    current = Path(path).resolve()
    paths: list[Path] = []
    layers: list[dict[str, Any]] = []
    while True:
        if current in paths:
            chain = " -> ".join(str(item) for item in [*paths, current])
            raise ValueError(f"Config inheritance cycle: {chain}")
        paths.append(current)
        with current.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must contain a YAML mapping: {current}")
        parent_name = data.pop("inherits", None)
        layers.append(data)
        if parent_name is None:
            break
        if not isinstance(parent_name, str) or not parent_name.strip():
            raise ValueError(f"inherits must be a non-empty file path: {current}")
        current = (current.parent / parent_name).resolve()
    result: dict[str, Any] = {}
    for layer in reversed(layers):
        result = deep_update(result, layer)
    return result


def half_car_dimensions(config: dict[str, Any]) -> tuple[int, int, int]:
    """返回动作、预瞄 token、基础观测维度；仅支持 privileged 半车观测。

    动作和 token 均为前后轴两维，基础观测为 14 维，可追加两维上一时刻动作。
    不支持的历史或噪声设置直接报错，避免配置与实际网络输入不一致。
    """
    observation = config.get("observation", {})
    if str(observation.get("mode", "privileged")).lower() != "privileged":
        raise ValueError("HalfCarEnv only implements observation.mode=privileged")
    if observation.get("history", {}).get("enabled", False):
        raise ValueError("HalfCarEnv does not implement observation.history")
    if float(observation.get("state_noise_std", 0.0)) != 0.0:
        raise ValueError("HalfCarEnv does not implement observation.state_noise_std")
    base_dim = 14 + (2 if config.get("action_in_observation", False) else 0)
    rl = config.get("rl", {})
    for key, expected in (("act_dim", 2), ("preview_token_dim", 2), ("obs_dim_base", base_dim)):
        if key in rl and rl[key] != expected:
            raise ValueError(f"rl.{key} must be {expected} for HalfCarEnv")
    return 2, 2, base_dim


def apply_algorithm_overrides(config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    result = deepcopy(config)
    overrides = result.get("rl", {}).get("algorithm_overrides", {}).get(algorithm.lower(), {})
    if overrides:
        result["rl"] = deep_update(result["rl"], overrides)
    return result


def as_namespace(data: dict[str, Any]):
    class Namespace:
        pass

    ns = Namespace()
    for key, value in data.items():
        if isinstance(value, dict):
            value = as_namespace(value)
        setattr(ns, key, value)
    return ns
