"""Synchronized two-camera comparison with actual measured suspension telemetry."""

from __future__ import annotations

import tkinter as tk
from collections import deque

from PIL import Image, ImageTk


class ComparisonView:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("主动与被动悬架同步对照 — 左 MPC / 右被动")
        self.root.configure(bg="#101827")
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.StringVar(value="初始化两个独立 CarSim 求解器…")
        tk.Label(
            self.root,
            textvariable=self.status,
            font=("Microsoft YaHei UI", 13),
            bg="#101827",
            fg="#e2e8f0",
        ).pack(pady=8)
        columns = tk.Frame(self.root, bg="#101827")
        columns.pack()
        self.cameras, self.metrics = [], []
        for index, (title, color) in enumerate(
            [("左：Orin MPC 主动悬架", "#22d3ee"), ("右：被动悬架（主动力为零）", "#fb923c")]
        ):
            frame = tk.Frame(columns, bg="#101827")
            frame.grid(row=0, column=index, padx=5)
            tk.Label(
                frame, text=title, font=("Microsoft YaHei UI", 13, "bold"), bg="#101827", fg=color
            ).pack(pady=4)
            camera = tk.Label(frame, bg="#101827")
            camera.pack()
            metric = tk.StringVar(value="等待数据")
            tk.Label(
                frame,
                textvariable=metric,
                font=("Consolas", 11),
                bg="#101827",
                fg="#e2e8f0",
                justify="left",
            ).pack(pady=5)
            self.cameras.append(camera)
            self.metrics.append(metric)
        self.plot = tk.Canvas(self.root, width=1200, height=175, bg="#182337", highlightthickness=0)
        self.plot.pack(padx=10, pady=5)
        self.summary = tk.StringVar(
            value="同参数、相同路面副本、同一仿真时钟；固定高度跟拍，无位移放大"
        )
        tk.Label(
            self.root,
            textvariable=self.summary,
            bg="#101827",
            fg="#cbd5e1",
            font=("Microsoft YaHei UI", 11),
        ).pack(pady=4)
        tk.Label(
            self.root,
            text="离线同步比较 · 左侧深度预瞄由 Orin 处理 · 右侧保留相同弹簧与阻尼 · 关闭窗口可停止",
            bg="#101827",
            fg="#94a3b8",
            font=("Microsoft YaHei UI", 10),
        ).pack(pady=5)
        self.photos = [None, None]
        self.history = deque(maxlen=260)
        self.root.update_idletasks()
        self.root.lift()

    def close(self):
        self.closed = True
        self.root.destroy()

    def show(self, images, now, states, active_force, rms, warmup=False):
        if self.closed:
            raise RuntimeError("Comparison window closed by user")
        for index, (image, values) in enumerate(zip(images, states)):
            picture = Image.frombytes(
                "RGBA", (image.width, image.height), bytes(image.raw_data), "raw", "BGRA"
            ).convert("RGB")
            self.photos[index] = ImageTk.PhotoImage(picture.resize((600, 338)))
            self.cameras[index].configure(image=self.photos[index])
            force = active_force if index == 0 else [0.0, 0.0]
            self.metrics[index].set(
                f"车速 {values[76]:5.2f} km/h   垂向加速度 {values[40] * 9.80665:+6.3f} m/s²\n"
                f"前/后轴主动力 {force[0]:+7.1f} / {force[1]:+7.1f} N\n"
                f"加速度 RMS {rms[index]:6.3f} m/s²   俯仰 {values[41]:+6.3f} deg"
                f"\n前/后轴轮跳 {(values[44] + values[46]) / 2:+6.2f} / {(values[45] + values[47]) / 2:+6.2f} mm（压缩为正）"
            )
        self.status.set(
            f"同步仿真时间 {now:.2f} s"
            + (" · 深度覆盖建立中，两侧均使用被动悬架" if warmup else "")
        )
        a, p = states
        self.history.append(
            (now, a[40] * 9.80665, p[40] * 9.80665, a[41], p[41], active_force[0], active_force[1])
        )
        self.draw()
        change = (rms[0] / rms[1] - 1) * 100 if rms[1] > 1e-8 else 0
        self.summary.set(
            f"车身垂向加速度 RMS：MPC {rms[0]:.3f} / 被动 {rms[1]:.3f} m/s²"
            f"   MPC 相对变化 {change:+.1f}%（负值表示降低）"
        )
        self.root.update_idletasks()
        self.root.update()

    def draw(self):
        self.plot.delete("all")
        rows = list(self.history)
        specs = [
            (12, 390, "垂向加速度 m/s² · 青=MPC / 橙=被动", [(1, "#22d3ee"), (2, "#fb923c")]),
            (412, 790, "俯仰角 deg · 青=MPC / 橙=被动", [(3, "#22d3ee"), (4, "#fb923c")]),
            (812, 1188, "主动轴力 N · 青=前轴 / 橙=后轴", [(5, "#22d3ee"), (6, "#fb923c")]),
        ]
        for left, right, title, series in specs:
            self.plot.create_text(left, 12, text=title, anchor="w", fill="#cbd5e1")
            self.plot.create_line(left, 95, right, 95, fill="#475569")
            if len(rows) < 2:
                continue
            scale = max(1e-3, max(abs(row[i]) for row in rows for i, _ in series))
            span = max(rows[-1][0] - rows[0][0], 0.05)
            for index, color in series:
                points = []
                for row in rows:
                    points.extend(
                        (
                            left + (right - left) * (row[0] - rows[0][0]) / span,
                            95 - 60 * row[index] / scale,
                        )
                    )
                self.plot.create_line(*points, fill=color, width=1.5)

    def finish(self, message):
        if not self.closed:
            self.status.set(message)
            self.root.update()

    def wait(self):
        if not self.closed:
            self.root.mainloop()
