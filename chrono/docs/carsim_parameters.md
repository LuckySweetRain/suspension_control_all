# CarSim D-Class Sedan 参数迁移

本配置读取用户本机CarSim 2020.0数据库的D-Class Sedan参数，换算后接入严格七自由度Chrono模型。
本次没有运行或修改CarSim，也没有完成两套求解器在同一道路和控制输入下的响应对比。
默认 `sedan_reference.yaml` 保留；新参数通过独立配置显式选择。

## 来源与可信度

用户提供路径的实际目录是 `E:\Desktop\workspace\CarSim_database\python_entry`。
`simfile.sim` 的INPUT宏展开后指向：

```text
E:\Desktop\workspace\CarSim_database\carsim_active_suspension\Results\Run_db4bb818-8634-450e-8fee-ebd655e9283b\Run_all.par
```

它在2026年8月18日展开，车型为 `D-Class, Sedan`，试验为 `Bounce Sine Sweep`。
入口目录的 `OUTPUT_FILE_PREFIX_echo.par` 第11–13行明确记录2026年3月9日初始化失败，
因此仅作单位、字段含义与静载旁证，不作为成功仿真结果。`simple_controller.py` 是路径跟踪
控制器，不包含车辆悬架参数。

`results/chrono/carsim/source_audit.json` 保存三个源文件的路径、SHA256、时间、行号、单位、
转换公式及限制。YAML是独立的参数快照，运行时不需要CarSim安装或许可；修改源数据库不会
自动同步这些配置。

## 已应用参数

四角顺序统一为FL、FR、RL、RR。下表行号属于上述 `Run_all.par`。

| 项目 | Chrono SI值 | 来源/换算 |
|---|---|---|
| 簧载质量 | 1370 kg | `M_SU`，3450 |
| 簧载质心惯量Ixx/Iyy/Izz | 671.3 / 1972.8 / 2315.3 kg·m² | 3451–3453 |
| 质心到前/后轴距离 | 1.110 / 1.756 m | `LX_CG_SU=1110`、`LX_AXLE=2866` mm，3448–3449 |
| 前/后轮距 | 1.550 / 1.550 m | `L_TRACK`，739、2087 |
| 前单轮簧下质量 | 66.95 kg | 12.48+32.47 kg悬架，914–915；另加22 kg轮胎，1324 |
| 后单轮簧下质量 | 54.6 kg | 32.6+0 kg悬架，2258–2259；另加22 kg轮胎，2663 |
| 整车质量 | 1613.1 kg | 簧载＋四角簧下质量 |
| 轮胎自由半径 | 0.334 m | `R0`，1321；不能用0.325m滚动半径RRE代替 |
| 轮胎垂向刚度 | 268000 N/m | 268 N/mm，1322 |
| 轮胎力上限 | 每角100000 N | `FZ_MAX`，1323 |
| 前轮端弹簧刚度 | 64929.2332608 N/m | 153 N/mm×1000×0.65144²，1109、1115 |
| 后轮端弹簧刚度 | 52087.138 N/m | 82 N/mm×1000×0.797²，2449、2455 |
| 前/后附加侧倾刚度 | 22001.579333 / 19709.748153 N·m/rad | 384/344 N·m/deg×180/π，1228、2568 |
| 重力、积分步长 | 9.80665 m/s²、0.0005 s | CarSim常量G、输入TSTEP |

该轮胎选项的垂向本构为弹簧，未定义垂向阻尼，故设置为零。计算静载为每轮前
4772.405538N、后3137.148019N；平路轮心高度为前0.316192517m、后0.322294224m。

## 阻尼、限位与侧倾耦合

运动比 `r=元件压缩量/轮端压缩量`。用虚功转换：`k_wheel=k_element*r²`、
`v_wheel=v_element/r`、`F_wheel=r*F_element`。源mm和mm/s先转为米和m/s。

已转换 `Big Car` 阻尼表全部13个节点，按前后运动比分别生成四角轮端表，替代原常数阻尼。
零速切线前4837.864439、后7241.382600 N·s/m仅用于分析及LQR线性化，不代表全速域阻尼。
原 `FD_TABLE SPLINE` 的完整边界算法未得到确认，当前采用保留全部节点的分段线性插值。

前后压缩/回弹限位表原本是LINEAR，现已转换并成对替代旧对称二次止挡。所有力表均采用
端部斜率线性外推，过行程力不会钳在最后节点。

| 轮端开始接触限位，压缩为正 | 前轴 | 后轴 |
|---|---:|---:|
| 压缩限位 | +90.5686 mm | +86.5747 mm |
| 回弹限位 | −59.8674 mm | −61.4806 mm |

物理回调使用等效压缩 `j=-(length-rest_length)` 和压缩速度 `-GetVelocity()`。
有限转角时，这个TSDA伸缩不完全等于CarSim轮心在车身Z轴上的投影。弹簧±20N加载/卸载
偏置及2mm滞回记忆未实现，对应轮端滞回约前±13.03N、后±15.94N。

附加侧倾刚度转为同轴交叉弹簧 `k_cross=K_roll/track²`，其势能为
`0.5*k_cross*(extension_left-extension_right)²`，两侧施加相反恢复力；同相跳动不产生附加力。
该项不增加机械自由度，LQR静态刚度矩阵也包含它。另一侧长度从当前刚体几何计算，避开
TSDA长度缓存的更新顺序影响。源附加侧倾阻尼为零。

## 几何与控制边界

`H_CG_SU=0.520m` 是相对设计车身原点的高度。结合设计轮心高度（前0.340m、后0.325m）
及静载半径，推得平路质心高度约0.504360158m、俯仰约0.007362698rad。本配置采用该高度，
将七自由度静平衡参考俯仰重置为零，并使用质心平面的等效安装点
`suspension_mount_height_m=0`。这不是完整悬架硬点导入或完整姿态复原。

CarSim输入15937–15944行的 `IMP_FD_* Replace` 会替换减振器力，且按FL、RL、FR、RR
重复声明为8通道。入口 `test2DS.py` 的外部控制循环被注释，其控制函数输出又采用
FL、FR、RL、RR，启用时存在轮序错配风险；本次没有修改该外部代码。

Chrono采用“源车辆被动机械特性＋四角主动执行器”，动作继续为FL、FR、RL、RR顺序的
归一化目标轮端力。执行器±3500N、20ms一阶响应、10ms时延、速率及功率设置仍是研究假设。
传感器、奖励、七类道路及默认15m/s车速保留，没有复制CarSim的正弦扫频试验。

90mm仍是设计/奖励阈值；行程提前终止为其2倍，另有轮心、安装端点及姿态保护。它与上述
不对称物理限位接触位置不同。转向、纵横向运动、轮转、轮胎剪切、气动力及悬架顺从/运动学
映射不在本七自由度模型内，因此不能声称与CarSim整车动态完全等价。

## 使用

在仓库根目录、`chrono-rl`环境运行：

```powershell
python chrono/scripts/view.py --config chrono/configs/carsim_dclass.yaml --controller skyhook --scenario composite --camera road --hold
python chrono/scripts/check_model.py --config chrono/configs/carsim_dclass.yaml
python chrono/scripts/check_carsim.py --with-vector
python chrono/scripts/train.py --config chrono/configs/carsim_dclass_seven_roads.yaml --algorithm ppo --observation-mode sensors --workers 2 --steps 65536 --name carsim_dclass_ppo
python chrono/scripts/evaluate.py --config chrono/configs/carsim_dclass.yaml --baseline passive --road-suite seven --split validation --name carsim_dclass_passive_validation
```

七路面YAML通过相对 `inherits: carsim_dclass.yaml` 复用机械参数，加载器检查循环，检查点
保存展开配置。0.5ms物理步长使默认特权观测从58维变成98维（时延命令队列增加），传感器
观测仍为130维。应新建此车型的训练，不能直接用旧Sedan权重和归一化统计恢复。

核对的本机官方文档：`E:\Carsim2020.0_Prog\Help\Screens\Vehicle.pdf`第17–18页、
`Tire_Models.pdf`第8–9及18–19页、`Suspension_Systems.pdf`第5、9–10、29–30页，
以及 `Help\Manuals\VehicleSim_Solvers.pdf`第46页。运行证据见 [验收记录](validation.md)。
