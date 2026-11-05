"""Compare physical validation metrics across a 2x2 state-mapping experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root):
    rows = []
    for arm in ("plain_keep", "mapped_keep", "plain_no_state", "mapped_no_state"):
        run = root / arm
        status = (
            json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
        )
        path = run / "evaluation_history.json"
        evaluations = json.loads(path.read_text()) if path.exists() else []
        passive = next((e for e in evaluations if e["label"] == "passive"), None)
        trained = [e for e in evaluations if e["label"].startswith("after_")]
        latest = trained[-1] if trained else None
        row = dict(arm=arm, status=status.get("stage", "pending"), latest_validation=latest)
        if latest and passive:
            row["rms_reduction_vs_passive_pct"] = 100 * (
                1 - latest["vertical_accel_rms_mps2"] / passive["vertical_accel_rms_mps2"]
            )
        rows.append(row)
    report = dict(
        complete=all(r["status"] == "completed" for r in rows),
        arms=rows,
        interpretation="Compare physical metrics at equal steps, not different reward sums; no hard-safety claim",
    )
    (root / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = [
        "# 状态映射四组实验",
        "",
        "按最新完成的验证点展示；不同训练步数不可直接归因比较。",
        "状态映射不保证物理约束。奖励定义不同，不比较 return 高低。",
        "",
        "| 组别 | 状态 | 验证点 | 加速度 RMS | 相对被动改善% | 行程超限比例 | 最小轮胎载荷 N |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        e = r["latest_validation"] or {}
        text.append(
            f"| {r['arm']} | {r['status']} | {e.get('label', '尚无')} | "
            f"{e.get('vertical_accel_rms_mps2', '')} | {r.get('rms_reduction_vs_passive_pct', '')} | "
            f"{e.get('relative_travel_over_80mm_fraction', '')} | {e.get('min_native_tire_load_N', '')} |"
        )
    (root / "comparison.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
