"""Verify native TCP provenance and index each recorded town, including failures."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from evaluate_signal_recording import evaluate


def review(suite, evaluate_signals=False):
    import matplotlib

    matplotlib.use("Agg")
    from plot_carsim_states import configure_fonts, export

    configure_fonts()
    suite = suite.resolve()
    manifest = json.loads((suite / "manifest.json").read_text())
    summaries, cards = [], []
    for entry in manifest["runs"]:
        if "artifacts" not in entry:
            cards.append(
                "<section><h2>"
                + html.escape(entry["town"])
                + "</h2><p>待完成或设置失败；查看运行日志。</p></section>"
            )
            continue
        folder = Path(entry["artifacts"])
        export(folder)
        report = json.loads((folder / "report.json").read_text())
        assert report["driving_policy"] == "tcp_official_agent"
        trace = json.loads((folder / "trace.json").read_text())
        plans = json.loads((folder / "driving_plans.json").read_text())
        plan_map = {p["seq"]: p for p in plans}
        for step in trace:
            actual = step["reply"]["driving"]
            native = plan_map[actual["plan_seq"]]["tcp_control"]
            assert actual["control_source"] == "tcp_official_agent"
            assert all(actual[k] == native[k] for k in ("throttle", "brake"))
            assert actual["native_steer_right_positive"] == native["steer_right_positive"]
        rows = list(csv.DictReader((folder / "video_telemetry.csv").open(encoding="utf-8-sig")))
        encoder = shutil.which("ffmpeg")
        probe = Path(encoder).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        stream = json.loads(
            subprocess.check_output(
                [
                    str(probe),
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=duration,nb_frames,r_frame_rate",
                    "-of",
                    "json",
                    str(folder / "tcp_controls.mp4"),
                ],
                universal_newlines=True,
            )
        )["streams"][0]
        assert int(stream["nb_frames"]) == len(rows) == report["recording"]["frames"]
        subprocess.run(
            [
                encoder,
                "-v",
                "error",
                "-i",
                str(folder / "tcp_controls.mp4"),
                "-f",
                "null",
                "NUL" if os.name == "nt" else "/dev/null",
            ],
            check=True,
        )
        distance = sum(
            math.hypot(b["values"][77] - a["values"][77], b["values"][78] - a["values"][78])
            for a, b in zip(trace, trace[1:])
        )
        red_ids = sorted(
            {
                int(r["associated_light_id"])
                for r in rows
                if r["associated_light_id"] and r["associated_light_state"] == "Red"
            }
        )
        assessments = []
        if evaluate_signals:
            audit = json.loads((folder / "traffic_audit.json").read_text())
            for light_id in red_ids:
                first = next(
                    r
                    for r in rows
                    if r["associated_light_id"] == str(light_id)
                    and r["associated_light_state"] == "Red"
                )
                now = float(first["sim_time_s"])
                junction = (
                    int(first["junction_id"])
                    if first["junction_id"]
                    else next(
                        (
                            e["junction_id"]
                            for e in audit["events"]
                            if e["event"] == "enter" and now <= e["time_s"] <= now + 60
                        ),
                        -1,
                    )
                )
                assessments.append(evaluate(folder, light_id, junction))
        summary = dict(
            town=entry["town"],
            spawn=entry["spawn"],
            folder=str(folder),
            integration_completed=report["passed"],
            video_seconds=float(stream["duration"]),
            distance_m=distance,
            native_actuator_ticks_verified=len(trace),
            model_plans=len(plans),
            contact_events=report.get("contact_events", []),
            error=report.get("error"),
            red_signal_ids=red_ids,
            signal_assessments=assessments,
            driving_compliance="not certified; inspect video and traffic traces",
        )
        summaries.append(summary)
        link = os.path.relpath(str(folder), str(suite)).replace("\\", "/")
        signal_links = " · ".join(
            '<a href="'
            + link
            + "/signal_"
            + str(a["target_light_id"])
            + '/signal_validation.json">灯 '
            + str(a["target_light_id"])
            + " 评估</a>"
            for a in assessments
        )
        notes = html.escape(str(summary["contact_events"] or summary["error"] or "未报告接触事件"))
        state = "运行结束" if report["passed"] else "运行中止"
        cards.append(
            f'<section><h2>{entry["town"]} · {summary["video_seconds"]:.1f} s · {distance:.0f} m · {state}</h2><video controls preload="metadata" src="{link}/tcp_controls.mp4"></video><p>起点 {entry["spawn"]}；已核对 {len(trace)} 次原版执行输出。{notes}</p><p><a href="{link}/video_telemetry.csv">控制量 CSV</a> · <a href="{link}/traffic_audit.json">信号记录</a> · <a href="{link}/report.json">运行报告</a> {signal_links}</p></section>'
        )
        cards.append(
            f'<p><a href="{link}/carsim_states.png">车辆状态曲线 PNG</a> · <a href="{link}/carsim_states.pdf">PDF</a> · <a href="{link}/carsim_states.csv">100 Hz 状态 CSV</a></p><img style="width:100%" src="{link}/carsim_states.png" alt="CarSim车辆状态曲线">'
        )
    (suite / "native_review.json").write_text(json.dumps(summaries, indent=2))
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>TCP 原始策略：其余城市地图重跑</title><style>body{background:#101827;color:#f1f5f9;font:17px sans-serif;max-width:1200px;margin:25px auto;padding:20px;line-height:1.7}video{width:100%;max-width:1100px}a{color:#67e8f9}section{border-top:1px solid #475569;margin-top:30px}</style><h1>TCP 原始策略：其余城市地图重跑</h1><p>原版双分支融合、无额外横纵向 PID、无速度覆盖、无信号真值控制；垂向仍为 MPC。运行结束仅说明接口完成，不表示驾驶或红灯合规。自动灯色时序评估不能代替画线停止线与录像核查。</p>'
        + "".join(cards)
        + "</html>"
    )
    (suite / "index.html").write_text(page, encoding="utf-8")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--evaluate-signals", action="store_true")
    args = parser.parse_args()
    print(json.dumps(review(args.suite, args.evaluate_signals), indent=2))
