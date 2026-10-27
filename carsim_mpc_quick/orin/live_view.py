"""Live CARLA camera and suspension telemetry window (legacy Python 3.7 frontend)."""

from __future__ import annotations

import threading
import time
import tkinter as tk
from collections import deque

from PIL import Image, ImageTk


class ViewerClosed(RuntimeError):
    """The user closed the live run window."""


class LiveView:
    def __init__(self, footnote=None, heading=None):
        self.root = tk.Tk()
        self.root.title("CARLA + CarSim + Orin — 实时联合仿真")
        self.root.configure(bg="#101827")
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.StringVar(value="正在初始化三端仿真…")
        self.metrics = tk.StringVar(value="等待控制数据")
        tk.Label(
            self.root,
            text=heading or "CARLA 场景 · CarSim 动力学 · Orin 悬架控制",
            bg="#101827",
            fg="#f1f5f9",
            font=("Microsoft YaHei UI", 16, "bold"),
        ).pack(pady=(12, 5))
        tk.Label(
            self.root,
            textvariable=self.status,
            bg="#101827",
            fg="#67e8f9",
            font=("Microsoft YaHei UI", 10),
        ).pack()
        self.camera = tk.Label(self.root, bg="#101827")
        self.camera.pack(padx=12, pady=8)
        tk.Label(
            self.root,
            textvariable=self.metrics,
            bg="#101827",
            fg="#e2e8f0",
            font=("Consolas", 11),
            justify="left",
        ).pack(padx=12, pady=5)
        self.plot = tk.Canvas(self.root, width=900, height=155, bg="#182337", highlightthickness=0)
        self.plot.pack(padx=12, pady=6)
        tk.Label(
            self.root,
            text=footnote
            or "当前路面与预瞄来自 CarSim；相机尚未用于感知控制。关闭窗口可停止本次运行。",
            bg="#101827",
            fg="#94a3b8",
            font=("Microsoft YaHei UI", 9),
        ).pack(pady=(2, 10))
        self.history = deque(maxlen=240)
        self.photo = None
        self.root.update_idletasks()
        self.root.lift()

    def close(self):
        self.closed = True
        self.root.destroy()

    def show(
        self,
        image,
        sim_time=0,
        values=None,
        diagnostics=None,
        rtt=0,
        steps=0,
        road_preview=None,
        driving=None,
    ):
        if self.closed:
            raise ViewerClosed("用户关闭可视化窗口，已停止本次仿真")
        rgb = Image.frombytes(
            "RGBA", (image.width, image.height), bytes(image.raw_data), "raw", "BGRA"
        ).convert("RGB")
        self.photo = ImageTk.PhotoImage(rgb.resize((900, 506)))
        self.camera.configure(image=self.photo)
        self.status.set("运行中 · CARLA 相机画面 · 同步仿真时间")
        if values is not None and diagnostics is not None:
            front, rear = diagnostics[:2]
            acceleration = values[40] * 9.80665
            text = (
                f"仿真时间 {sim_time:6.2f} s     车速 {values[76]:6.2f} km/h     控制次数 {steps}\n"
                f"前轴主动力 {front:8.1f} N     后轴主动力 {rear:8.1f} N     Orin 请求 {rtt:6.2f} ms\n"
                f"车身垂向加速度 {acceleration:7.3f} m/s²     俯仰角 {values[41]:7.3f} deg"
            )
            if road_preview is not None:
                text += f"\n前/后轴预瞄路高 {road_preview[0] * 1000:6.1f} / {road_preview[2] * 1000:6.1f} mm（相对初始平面）"
            if driving is not None:
                text += (
                    f"\nPID 目标车速 {driving['target_speed_mps'] * 3.6:5.1f} km/h"
                    f"  方向盘 {driving['carsim_inputs'][0]:6.1f}°"
                    f"  油门/制动 {driving['throttle']:.2f}/{driving['brake']:.2f}"
                )
            self.metrics.set(text)
            self.history.append(
                (
                    sim_time,
                    front,
                    rear,
                    acceleration,
                    road_preview[0] if road_preview is not None else None,
                    road_preview[2] if road_preview is not None else None,
                )
            )
            self.draw_plot()
        self.root.update_idletasks()
        self.root.update()
        if self.closed:
            raise ViewerClosed("用户关闭可视化窗口，已停止本次仿真")

    def wait_for_planner(self, function):
        """Keep the window responsive while an offline inference pauses the clock."""
        result, errors = [], []

        def work():
            try:
                result.append(function())
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        self.status.set("Orin 模型推理中 · 仿真时钟暂停 · 可关闭窗口停止")
        while worker.is_alive():
            if self.closed:
                raise ViewerClosed("用户关闭可视化窗口，已停止本次仿真")
            self.root.update()
            time.sleep(0.02)
        if self.closed:
            raise ViewerClosed("用户关闭可视化窗口，已停止本次仿真")
        if errors:
            raise errors[0]
        return result[0]

    def draw_plot(self):
        self.plot.delete("all")
        items = list(self.history)
        has_road = any(item[4] is not None for item in items)
        road_scale = max(
            0.005,
            max(
                (abs(item[index]) for item in items for index in (4, 5) if item[index] is not None),
                default=0,
            ),
        )
        panels = [
            (12, 440, "主动轴力 N  · 前轴青色 / 后轴橙色"),
            (465, 888, "车身垂向加速度 m/s²"),
        ]
        if has_road:
            panels = [
                (12, 300, "主动轴力 N · 青前/橙后"),
                (320, 595, "车身垂向加速度 m/s²"),
                (615, 888, f"预瞄路高 ±{road_scale * 1000:.0f} mm · 青前/橙后"),
            ]
        for left, right, title in panels:
            self.plot.create_text(left, 12, text=title, anchor="w", fill="#cbd5e1")
            self.plot.create_line(left, 85, right, 85, fill="#475569")
        if len(items) < 2:
            return
        span = max(items[-1][0] - items[0][0], 0.05)
        acc_scale = max(1.0, max(abs(item[3]) for item in items))
        series = [
            (1, "#22d3ee", 12, 440, 2100),
            (2, "#fb923c", 12, 440, 2100),
            (3, "#a3e635", 465, 888, acc_scale),
        ]
        if has_road:
            series = [
                (1, "#22d3ee", 12, 300, 2100),
                (2, "#fb923c", 12, 300, 2100),
                (3, "#a3e635", 320, 595, acc_scale),
                (4, "#22d3ee", 615, 888, road_scale),
                (5, "#fb923c", 615, 888, road_scale),
            ]
        for index, color, left, right, scale in series:
            points = []
            for item in items:
                if item[index] is None:
                    continue
                points.extend(
                    (
                        left + (right - left) * (item[0] - items[0][0]) / span,
                        85 - 52 * item[index] / scale,
                    )
                )
            self.plot.create_line(*points, fill=color, width=2)

    def finish(self, message):
        if not self.closed:
            self.status.set(message)
            self.root.update()

    def wait(self):
        if not self.closed:
            self.root.mainloop()
