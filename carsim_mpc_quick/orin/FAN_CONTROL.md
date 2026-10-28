# Orin 风扇温控（2026-10-28）

这台 GEACX1 载板的实际风扇接口由 `geacx1-fancontrol.service` 管理：
`/sys/class/rb_gpio/fan_1-power/value` 与 `fan_2-power/value`。
目前验证的是两路风扇供电开关，不是可读转速的多档 PWM 接口。

| 条件 | 行为 |
| --- | --- |
| CPU / TJ 最高温度达到 60°C | 两路供电开启 |
| 温度降至 50°C 以下且已开启至少 30 s | 两路供电关闭 |
| 50–60°C 之间 | 保持原状态，避免反复启停 |
| 必需温度传感器异常 | 两路供电开启 |

服务已启用开机自启，每 2 s 采样一次；可选 GPU 传感器不可用时使用 CPU、TJ。
配置为 `/etc/geacx1-fancontrol.json`；修改前备份为
`/etc/geacx1-fancontrol.json.suspension-backup-20261028`。
脚本和系统服务保持原实现，仅修改温度阈值与最短开启时间。

通过 `sudo cat /run/geacx1-fancontrol/status.json` 查看状态。
`gpio=1` 只能证明供电输出读回开启，当前测速接口返回 0，不能据此宣称实际 RPM。
配置校验与回差策略检查记录在 `results/orin_fan_20261028/configuration_check.txt`。
