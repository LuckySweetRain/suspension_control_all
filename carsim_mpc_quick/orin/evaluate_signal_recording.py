"""Evaluate observed red wait / green departure without feeding truth to TCP.

This checks temporal behavior and junction clearance, not painted stop-line
compliance or CARLA Leaderboard infractions. The original video is preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path


def evaluate(folder, light_id, junction_id):
    folder = folder.resolve()
    evaluation_output = folder / ("signal_" + str(light_id))
    evaluation_output.mkdir(exist_ok=True)
    audit = json.loads((folder / "traffic_audit.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    rows = list(csv.DictReader((folder / "video_telemetry.csv").open(encoding="utf-8-sig")))
    video_time_origin = float(rows[0]["sim_time_s"])
    samples = {round(s["time_s"], 5): s for s in audit["samples"]}
    observed = []
    for row in rows:
        now = float(row["sim_time_s"])
        sample = samples[round(now, 5)]
        signal = next((s for s in sample["nearby_signals"] if s["id"] == light_id), None)
        if signal is not None:
            observed.append(
                dict(
                    time=now,
                    signal=signal["state"],
                    frozen=signal["frozen"],
                    speed=abs(float(row["speed_kph"])),
                    throttle=float(row["applied_throttle"]),
                    brake=float(row["applied_brake"]),
                    position=sample["world_pose"][:2],
                )
            )
    approach = next(
        (
            float(r["sim_time_s"])
            for r in rows
            if r["associated_light_id"] == str(light_id) and r["associated_light_state"] == "Red"
        ),
        None,
    )
    stopped, block = [], []
    for item in observed:
        if (
            approach is not None
            and item["time"] >= approach
            and item["signal"] == "Red"
            and item["speed"] < 0.5
            and item["brake"] > 0.5
        ):
            block.append(item)
        else:
            if block and block[-1]["time"] - block[0]["time"] >= 1:
                stopped.append(block)
            block = []
    if block and block[-1]["time"] - block[0]["time"] >= 1:
        stopped.append(block)
    stop = stopped[0] if stopped else []
    green = next(
        (p for p in observed if stop and p["time"] > stop[-1]["time"] and p["signal"] == "Green"),
        None,
    )
    if green:
        preceding_stops = [b for b in stopped if b[-1]["time"] < green["time"]]
        stop = preceding_stops[-1]
    waiting = [
        p for p in observed if stop and green and stop[0]["time"] <= p["time"] < green["time"]
    ]
    stable_until_green = bool(
        waiting
        and max(p["speed"] for p in waiting) < 0.5
        and green["time"] - stop[-1]["time"] <= 0.11
    )
    departure = next(
        (
            p
            for p in observed
            if green and p["time"] >= green["time"] and p["speed"] > 1 and p["throttle"] > 0.01
        ),
        None,
    )
    exit_event = next(
        (
            e
            for e in audit["events"]
            if departure
            and e["event"] == "exit"
            and e["junction_id"] == junction_id
            and e["time_s"] > departure["time"]
        ),
        None,
    )
    position_after = next(
        (p for p in observed if departure and p["time"] > departure["time"] + 2), None
    )
    moved = (
        math.hypot(*(a - b for a, b in zip(departure["position"], position_after["position"])))
        if departure and position_after
        else None
    )
    trace = json.loads((folder / "trace.json").read_text())
    plans = json.loads((folder / "driving_plans.json").read_text())
    plan_map = {p["seq"]: p for p in plans}
    for row in trace:
        actual = row["reply"]["driving"]
        native = plan_map[actual["plan_seq"]]["tcp_control"]
        assert actual["control_source"] == "tcp_official_agent"
        assert actual["throttle"] == native["throttle"] and actual["brake"] == native["brake"]
        assert actual["native_steer_right_positive"] == native["steer_right_positive"]
    contacts = [
        r
        for r in trace
        if stop
        and exit_event
        and stop[0]["time"] <= r["time"] <= exit_event["time_s"]
        and r["reply"].get("contact_supervisor")
    ]
    result = dict(
        target_light_id=light_id,
        target_junction_id=junction_id,
        integration_completed=report["passed"],
        recorded_sim_seconds=float(rows[-1]["sim_time_s"]),
        observed_red_stop_green_departure=bool(
            stable_until_green and departure and exit_event and moved and moved > 2 and not contacts
        ),
        red_association_time_s=approach,
        sustained_red_stop_start_s=stop[0]["time"] if stop else None,
        first_strong_brake_red_stop_s=stopped[0][0]["time"] if stopped else None,
        strong_brake_red_stop_intervals_s=[[b[0]["time"], b[-1]["time"]] for b in stopped],
        max_red_speed_after_first_stop_kph=max(
            (
                p["speed"]
                for p in observed
                if stopped
                and green
                and stopped[0][0]["time"] <= p["time"] < green["time"]
                and p["signal"] == "Red"
            ),
            default=None,
        ),
        sustained_red_stop_end_s=stop[-1]["time"] if stop else None,
        stopped_until_green=stable_until_green,
        red_wait_max_speed_kph=max(p["speed"] for p in waiting) if waiting else None,
        red_wait_drift_m=math.hypot(
            *(a - b for a, b in zip(stop[0]["position"], green["position"]))
        )
        if stop and green
        else None,
        green_start_observed_s=green["time"] if green else None,
        departure_above_1kph_s=departure["time"] if departure else None,
        departure_delay_s=departure["time"] - green["time"] if departure else None,
        junction_exit_s=exit_event["time_s"] if exit_event else None,
        displacement_2s_after_departure_m=moved,
        contact_supervisor_ticks_during_wait_and_clearance=len(contacts),
        target_signal_ever_frozen=any(p["frozen"] for p in observed),
        verified_native_actuator_ticks=len(trace),
        stop_line_compliance="not certified: trigger/junction geometry is not the painted stop line",
        signal_state_used_for_control=False,
        scope="single recorded approach; not an official traffic infraction score or general reliability claim",
        stop_detection="abs speed <0.5 km/h and brake >0.5 for at least 1 s; conservative strong-brake detection",
    )
    (evaluation_output / "signal_validation.json").write_text(json.dumps(result, indent=2))
    ffmpeg = shutil.which("ffmpeg")
    for name, sample in [
        ("red_wait", stop[0] if stop else None),
        ("green_release", green),
        ("departure", departure),
    ]:
        if sample is not None:
            subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    str(max(0, sample["time"] - video_time_origin)),
                    "-i",
                    str(folder / "tcp_controls.mp4"),
                    "-frames:v",
                    "1",
                    str(evaluation_output / (name + ".png")),
                ],
                check=True,
            )
    if stop and green:
        start = max(0, stop[0]["time"] - 2)
        end = (
            min(float(rows[-1]["sim_time_s"]), exit_event["time_s"] + 3)
            if exit_event
            else float(rows[-1]["sim_time_s"])
        )
        result["excerpt_source_start_s"] = start
        result["excerpt_source_end_s"] = end
        (evaluation_output / "signal_validation.json").write_text(json.dumps(result, indent=2))

        def stamp(t):
            total = max(0, int(round(t * 100)))
            return f"{total // 360000}:{total // 6000 % 60:02d}:{total // 100 % 60:02d}.{total % 100:02d}"

        subtitles = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1280",
            "PlayResY: 860",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Signal,Microsoft YaHei,20,&H00FFFFFF,&H00FFFFFF,&H00101827,&H00101827,0,0,0,0,100,100,0,0,3,2,0,9,20,20,20,1",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        for i, item in enumerate(observed):
            until = min(
                item["time"] + 0.05, observed[i + 1]["time"] if i + 1 < len(observed) else end, end
            )
            begin = max(start, item["time"])
            if until <= begin:
                continue
            label, color = {
                "Red": ("红灯", "0000FF"),
                "Green": ("绿灯", "00FF00"),
                "Yellow": ("黄灯", "00FFFF"),
            }.get(item["signal"], (item["signal"], "FFFFFF"))
            subtitles.append(
                f"Dialogue: 0,{stamp(begin - start)},{stamp(until - start)},Signal,,0,0,0,,{{\\c&H{color}&}}评估灯 {light_id}：{label}"
            )
        (evaluation_output / "signal_overlay.ass").write_text(
            "\n".join(subtitles), encoding="utf-8-sig"
        )
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-ss",
                str(max(0, start - video_time_origin)),
                "-i",
                str(folder / "tcp_controls.mp4"),
                "-t",
                str(end - start),
                "-vf",
                "subtitles=signal_overlay.ass",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-movflags",
                "+faststart",
                str(evaluation_output / "red_green_passage.mp4"),
            ],
            check=True,
            cwd=str(evaluation_output),
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--light-id", type=int, required=True)
    parser.add_argument("--junction-id", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.folder, args.light_id, args.junction_id), indent=2))
