"""Record actual CARLA frames with source-labelled controls at simulation-time 20 FPS."""

from __future__ import annotations

import csv
import math
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont

from runtime.paths import FONT_PATH


class ControlRecording:
    def __init__(self, output, unknown_contact="stop", town="", planner="tcp"):
        self.town = town
        self.planner = planner
        self.name = planner.upper()
        self.contact_note = "未分类碰撞仅记录" if unknown_contact == "record" else "未分类碰撞中止"
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg must be available on PATH to record")
        font = str(FONT_PATH)
        self.font = ImageFont.truetype(font, 21)
        self.small = ImageFont.truetype(font, 17)
        self.title = ImageFont.truetype(font, 28)
        self.path = output / (planner + "_controls.mp4")
        self.log = (output / "video_encoder.log").open("wb")
        self.table = (output / "video_telemetry.csv").open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.writer(self.table)
        self.writer.writerow(
            [
                "sim_time_s",
                "control_time_s",
                "speed_kph",
                planner + "_desired_speed_kph",
                "applied_steering_wheel_deg_left_positive",
                "applied_throttle",
                "applied_brake",
                planner + "_steer_right_positive",
                planner + "_throttle",
                planner + "_brake",
                planner + "_trajectory_speed_kph",
                "plan_seq",
                "inference_ms",
                "mpc_front_N",
                "mpc_rear_N",
                "junction_id",
                "associated_light_id",
                "associated_light_state",
            ]
        )
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                "1280x860",
                "-framerate",
                "20",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self.output, self.frames, self.last = output, 0, None

    def write(self, image, now, values, reply, plan, traffic=None):
        canvas = Image.new("RGB", (1280, 860), "#101827")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (18, 14),
            self.name + "（Orin）· " + self.town + " · 原始策略 · MPC 主动悬架",
            font=self.title,
            fill="#f1f5f9",
        )
        rgb = Image.frombytes(
            "RGBA", (image.width, image.height), bytes(image.raw_data), "raw", "BGRA"
        ).convert("RGB")
        canvas.paste(rgb.resize((960, 540)), (16, 64))
        d = reply["driving"]
        native = (
            (
                plan.get("tcp_control_head")
                if self.planner == "tcp"
                else plan.get("transfuser_control")
            )
            if plan
            else None
        )
        trajectory_speed = None
        if plan:
            a, b = plan["points"][:2]
            trajectory_speed = math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0]) * 3.6
        draw.text((996, 70), self.name + " 控制输出", font=self.font, fill="#67e8f9")
        draw.text(
            (996, 102),
            "原版融合前的控制分支" if self.planner == "tcp" else "模型自带控制器输出",
            font=self.small,
            fill="#fbbf24",
        )
        labels = [("转向（右+）", "steer_right_positive"), ("油门", "throttle"), ("制动", "brake")]
        for index, (label, key) in enumerate(labels):
            text = "--" if native is None else f"{native[key]:+.3f}"
            draw.text((996, 143 + 38 * index), label + "  " + text, font=self.font, fill="white")
        draw.text((996, 268), self.name + " 轨迹（捕获帧）", font=self.font, fill="#67e8f9")
        panel = Image.new("RGB", (270, 270), "#182337")
        pen = ImageDraw.Draw(panel)
        pen.line((135, 250, 135, 5), fill="#475569")
        pen.text((142, 7), "前", font=self.small, fill="white")
        if plan:
            points = [(135, 250)] + [(135 - p[2] * 12, 250 - p[1] * 12) for p in plan["points"]]
            pen.line(points, fill="#22d3ee", width=3)
            for x, y in points[1:]:
                pen.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#fbbf24")
        canvas.paste(panel, (994, 305))
        if traffic:
            draw.text(
                (996, 579),
                f"已通过路口 {traffic['completed_junctions']}",
                font=self.small,
                fill="#a3e635",
            )
            draw.text(
                (996, 602),
                "关联灯：" + str(traffic["associated_light_state"] or "未关联"),
                font=self.small,
                fill="#fbbf24",
            )
        rows = [
            f"仿真时间 {now:6.2f} s    实际车速 {values[76]:5.1f} km/h    {self.name} 轨迹期望 {d['target_speed_mps'] * 3.6:5.1f} km/h",
            f"{self.name} 原始策略执行：方向盘（左+）{d['carsim_inputs'][0]:7.2f}°    油门 {d['throttle']:.3f}    制动 {d['brake']:.3f}",
            self.name
            + " 轨迹隐含速度："
            + ("--" if trajectory_speed is None else f"{trajectory_speed:.1f} km/h")
            + (
                f"    规划帧 {plan['seq']}    模型推理 {plan['inference_ms']:.1f} ms"
                if plan
                else ""
            ),
            f"MPC 主动力：前轴 {reply['diagnostics'][0]:8.1f} N    后轴 {reply['diagnostics'][1]:8.1f} N    垂向加速度 {values[40] * 9.80665:.3f} m/s²",
        ]
        for i, line in enumerate(rows):
            draw.text((18, 620 + i * 43), line, font=self.font, fill="white")
        draw.text(
            (18, 808),
            "离线同步 · 按仿真时间 20 FPS 编码 · " + self.contact_note + "（非完整碰撞验证）",
            font=self.small,
            fill="#fbbf24",
        )
        self.process.stdin.write(canvas.tobytes())
        self.writer.writerow(
            [
                now,
                now - 0.01,
                values[76],
                d["target_speed_mps"] * 3.6,
                d["carsim_inputs"][0],
                d["throttle"],
                d["brake"],
                *(
                    [native[k] for k in ("steer_right_positive", "throttle", "brake")]
                    if native
                    else [None] * 3
                ),
                trajectory_speed,
                plan["seq"] if plan else None,
                plan["inference_ms"] if plan else None,
                *reply["diagnostics"][:2],
                *(
                    [
                        traffic[k]
                        for k in ("junction_id", "associated_light_id", "associated_light_state")
                    ]
                    if traffic
                    else [None] * 3
                ),
            ]
        )
        self.frames += 1
        self.last = canvas
        if self.frames % 20 == 0:
            self.table.flush()
        if self.frames == 1:
            canvas.save(self.output / "video_first.png")

    def close(self):
        try:
            self.process.stdin.close()
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise
            if self.last is not None:
                self.last.save(self.output / "video_last.png")
            if code:
                raise RuntimeError("Video encoder failed; inspect video_encoder.log")
        finally:
            self.table.close()
            self.log.close()
