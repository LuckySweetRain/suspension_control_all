"""CarSim 100 Hz state log; explicit native export units, no solver modification."""

from __future__ import annotations

import csv

FIELDS = (
    [
        "sim_time_s",
        "vx_kph",
        "vy_kph",
        "zcg_m",
        "vz_mps",
        "az_mps2",
        "pitch_deg",
        "pitch_rate_deg_s",
        "pitch_accel_rad_s2",
        "roll_deg",
        "yaw_deg",
        "x_m",
        "y_m",
    ]
    + ["jounce_" + w + "_mm" for w in ("L1", "L2", "R1", "R2")]
    + ["load_" + w + "_N" for w in ("L1", "L2", "R1", "R2")]
    + ["mpc_front_N", "mpc_rear_N", "steering_wheel_deg", "throttle", "brake"]
)


def state_row(now, values, reply):
    driving = reply.get("driving", {})
    return [
        now,
        values[76],
        values[81] if len(values) > 81 else float("nan"),
        values[38],
        values[39] / 3.6,
        values[40] * 9.80665,
        values[41],
        values[42],
        values[43],
        values[80],
        values[79],
        values[77],
        values[78],
        *values[44:48],
        *values[52:56],
        *reply["diagnostics"][:2],
        driving.get("carsim_inputs", [float("nan")])[0],
        driving.get("throttle", float("nan")),
        driving.get("brake", float("nan")),
    ]


class StateTelemetry:
    def __init__(self, output, filename="carsim_states.csv"):
        self.file = (output / filename).open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.writer(self.file)
        self.writer.writerow(FIELDS)
        self.count = 0

    def write(self, now, values, reply):
        self.writer.writerow(state_row(now, values, reply))
        self.count += 1
        if self.count % 20 == 0:
            self.file.flush()

    def close(self):
        self.file.close()
