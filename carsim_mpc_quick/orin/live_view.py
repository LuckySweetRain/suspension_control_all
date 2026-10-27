"""Live CARLA camera and suspension telemetry window (legacy Python 3.7 frontend)."""

from __future__ import annotations

import tkinter as tk
from collections import deque

from PIL import Image, ImageTk


class ViewerClosed(RuntimeError):
    """The user closed the live run window."""


class LiveView:
    def __init__(self, footnote=None):
        self.root = tk.Tk()
        self.root.title("CARLA + CarSim + Orin — 实时联合仿真")
        self.root.configure(bg="#101827")
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.StringVar(value="正在初始化三端仿真…")
        self.metrics = tk.StringVar(value="等待控制数据")
        tk.Label(
            self.root,
            text="CARLA 场景 · CarSim 动力学 · Orin 悬架控制",
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

    def show(self, image, sim_time=0, values=None, diagnostics=None, rtt=0, steps=0):
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
            self.metrics.set(
                f"仿真时间 {sim_time:6.2f} s     车速 {values[76]:6.2f} km/h     控制次数 {steps}\n"
                f"前轴力 {front:8.1f} N     后轴力 {rear:8.1f} N     Orin 往返 {rtt:6.2f} ms\n"
                f"车身垂向加速度 {acceleration:7.3f} m/s²     俯仰角 {values[41]:7.3f} deg"
            )
            self.history.append((sim_time, front, rear, acceleration))
            self.draw_plot()
        self.root.update_idletasks()
        self.root.update()
        if self.closed:
            raise ViewerClosed("用户关闭可视化窗口，已停止本次仿真")

    def draw_plot(self):
        self.plot.delete("all")
        items = list(self.history)
        for left, right, title in [
            (12, 440, "主动轴力 N  · 前轴青色 / 后轴橙色"),
            (465, 888, "车身垂向加速度 m/s²"),
        ]:
            self.plot.create_text(left, 12, text=title, anchor="w", fill="#cbd5e1")
            self.plot.create_line(left, 85, right, 85, fill="#475569")
        if len(items) < 2:
            return
        span = max(items[-1][0] - items[0][0], 0.05)
        acc_scale = max(1.0, max(abs(item[3]) for item in items))
        for index, color, left, right, scale in [
            (1, "#22d3ee", 12, 440, 2100),
            (2, "#fb923c", 12, 440, 2100),
            (3, "#a3e635", 465, 888, acc_scale),
        ]:
            points = []
            for item in items:
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
