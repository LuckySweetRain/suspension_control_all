# PPO 状态区间映射与移除约束奖励的对照实验

## 原笔记的适用边界

参考用户 Obsidian 的《带有约束的主动悬架强化学习算法》方法一。
定义 T(x)=atanh(2(x-l)/(h-l)-1)，逆变换为 l+(h-l)(1+tanh(z))/2。
有限 z 的逆变换落在开区间 (l,h)，端点对应无穷，不是有限 z 可达的闭区间。
但当真实状态由动力学推进、策略只控制作用力时，重新编码观测不保证下一物理状态仍在区间内。
例如 x=0.9、x_next=x+u、u=0.2，当前 atanh(x) 有限，下一状态仍越过 1。
若 z=T(x)，正确的变换动力学是 z_next=T(A*T_inverse(z)+B*u+E*d)，仅在下一状态仍处于定义域时成立。
atanh 的导数在边界增大，因此不能笼统声称它消除梯度爆炸或保证更快收敛。

状态不变性需要额外控制条件；可参考 [Ames 等，Control Barrier Function Based Quadratic Programs for Safety Critical Systems](https://arxiv.org/abs/1609.06408)。本次不声称实现了 CBF、MPC 安全过滤器或状态硬约束保证。

## 实现

- `rl/bounded_state.py`：输入单位为 m；对四轮相对行程的 [-0.08,0.08] 做逆 tanh 观测变换。
- epsilon=1e-4，只为网络输入数值稳定将归一化值裁剪到 [-1+epsilon,1-epsilon]。
- 真实动力学状态、奖励计算、物理约束指标不裁剪。单独记录测量越界和数值饱和。
- 四轮带噪行程变换后进入原有 8 帧历史；其他输入、网络、动作和优化参数保持一致。
- PPO 原动作已使用 tanh；本次是固定状态编码，不改动作概率分布或动作 logp 的 Jacobian。
- 使用 measured 方案固定观测，不同时引入速度估计，避免混淆变量。

## 四组配对初筛

| 实验 | 行程观测 | 状态相关奖励 |
|---|---|---|
| mapped_no_state（用户目标组，先运行） | atanh | 去掉 |
| plain_keep | 原归一化 | 保留 |
| mapped_keep | atanh | 保留 |
| plain_no_state | 原归一化 | 去掉 |

去掉意味着 travel、tire、violation、unload 四项权重置零；accel、pitch、effort、slew、projection 保持不变。
其中 travel/tire 也包括连续性能成本，实验报告应明确这一点，而不是说仅删除二值越限罚项。
不同奖励定义下的 return 不直接比较高低；主要比较同路面的确定性加速度 RMS、俯仰 RMS、行程和载荷、动作平滑性。

每组从头训练 30 回合×1000 步，种子 20261105；共享噪声种子保持配置原值 20261104。
wave/bump/pothole 顺序循环，每 10 回合在 combined 验证；单一配对种子、固定验证集，不是泛化或安全证明。
初筛完成后再决定是否增加训练预算和随机种子。新结果不与旧中断任务拼接。

```powershell
conda run --no-capture-output -n suspension-control python scripts/run_sensor_ablation.py --suite state-mapping --episodes 30 --eval-every 10 --seed 20261105 --output results/state_mapping_20261105
```

共享 CARLA 顺序执行，独立 W&B run，compact 工件策略，不自动重试失败路线。
失败会保留现场并停止队列；旧 `sensor_ablation_20261104` A 组完成 40 回合后 CARLA 地图生成超时，B 尚未开始，不能记作 30 万步完成。

## 验证

`check_bounded_state.py` 检查区间内往返、单调性、边界有限性、无原始状态修改、无状态保证的动力学反例、四组观测/奖励配对以及原生越限指标汇总。
`check_cosim_observation.py` 检查隐藏状态隔离和历史输入。`check_project.py` 及 Ruff 检查通过。
真实 CarSim/CARLA/Orin 短测：3 回合×300 步，完成 PPO 更新及状态/力/logp 审计；只有 wave 在该时长内触达显著路面，bump/pothole 尚未到达对应障碍，因此仅作流程短测。

汇总修正：旧 `summarize_cosim_ppo.py` 用 104 维观测索引反推初始行程，不能用于 100 维历史观测或 atanh 状态。本次日志增加独立原生约束记录；新版优先采用它。缺少原生记录的历史 100 维运行标记约束指标不可恢复，不假造零违规。旧 A 组先前导出的行程指标需重新审查，不能作为已验证约束证据。
原生指标仍是 10 ms 采样，不能证明所有 1 ms 子步峰值均满足约束。

本地 Coding Plan 辅助审查发生 backend_timeout，未获得建议；实现和结论不依赖该调用。
