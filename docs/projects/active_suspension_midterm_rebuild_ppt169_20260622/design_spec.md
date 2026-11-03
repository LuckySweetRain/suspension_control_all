# active_suspension_midterm_rebuild - Design Spec

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | active_suspension_midterm_rebuild |
| **Canvas Format** | PPT 16:9 (1280x720) |
| **Page Count** | 16 |
| **Design Style** | `pyramid` mode + `swiss-minimal` visual style |
| **Target Audience** | 研究生中期答辩评审教师 |
| **Use Case** | 学术中期答辩：汇报研究进展、已完成成果、实验结果、问题与后续计划 |
| **Content Strategy** | balanced default；在不引入外部事实的前提下，将报告内容重组为答辩叙事 |
| **Created Date** | 2026-06-22 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280x720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 56px, top 48px, bottom 36px |
| **Content Area** | 1168x600 |

---

## III. Visual Theme

### Theme Style

- **Mode**: `pyramid`，每页标题尽量使用结论式表达，数据页给出比较和含义。
- **Visual style**: `swiss-minimal`，白底、严谨网格、少装饰、以线条和留白组织信息。
- **Theme**: Light theme.
- **Tone**: 学术、工程、克制、可信。

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#FFFFFF` | 页面背景 |
| **Secondary bg** | `#F3F6FA` | 信息区底色、浅色分区 |
| **Primary** | `#031D40` | 主标题、主线条、页脚 |
| **Accent** | `#B91C1C` | 关键风险、突出结论 |
| **Secondary accent** | `#4472C4` | 数据高亮、流程节点 |
| **Body text** | `#1D2430` | 正文 |
| **Secondary text** | `#4B5563` | 解释文本 |
| **Tertiary text** | `#7A869A` | 页码、脚注 |
| **Border/divider** | `#D9E2EF` | 细分隔线 |
| **Grid** | `#E8EEF7` | 背景辅助网格 |
| **Light blue** | `#EAF2FF` | 图标/指标浅底 |
| **Pale red** | `#FFF1F2` | 风险/问题浅底 |
| **Success** | `#0F766E` | 已完成与提升 |
| **Warning** | `#B91C1C` | 问题与风险 |

### Gradient Scheme

仅用于封面和转场页的极浅背景：

```xml
<linearGradient id="bgBlueWash" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%" stop-color="#EAF2FF" stop-opacity="0.75"/>
  <stop offset="100%" stop-color="#FFFFFF" stop-opacity="1"/>
</linearGradient>
```

---

## IV. Typography System

### Font Plan

**Typography direction**: 中文答辩投影优先，沿用开题参考 PPT 的微软雅黑/黑体系统，拉开标题和正文权重。

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | `SimHei` | `Arial` | `sans-serif` |
| **Body** | `Microsoft YaHei` | `Calibri` | `sans-serif` |
| **Emphasis** | `SimHei` | `Arial` | `sans-serif` |
| **Code** | -- | `Consolas`, `"Courier New"` | `monospace` |

**Per-role font stacks**

- Title: `SimHei, Arial, sans-serif`
- Body: `"Microsoft YaHei", Calibri, Arial, sans-serif`
- Emphasis: `SimHei, Arial, sans-serif`
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

**Baseline**: Body font size = 20px.

| Purpose | Size |
| ------- | ---- |
| Cover title | 48-60px |
| Page title | 32-38px |
| Subtitle / takeaway | 24-28px |
| Body content | 18-22px |
| Annotation / caption | 13-16px |
| Footer / source | 11-13px |

Formula rendering policy: `text-only`，简单变量与公式保留为可编辑文本。

---

## V. Layout Principles

### Page Structure

- **Header area**: 48-110px，左侧页标题 + 细横线，右侧页码或章节标签。
- **Content area**: 120-650px，采用 12 栏网格，图表和论点按信息权重非对称排布。
- **Footer area**: 664-700px，放置来源/阶段标签/页码。

### Layout Pattern Library

- 封面与阶段结论页使用大标题 + 原生几何/轻网格。
- 数据页使用图表/指标卡 + 一句结论，不做满屏表格。
- 方法页使用左侧机制说明 + 右侧结构图或流程图。
- 图像页尽量让图片占主导，旁边只保留可读的结论和解释。

### Spacing Specification

| Element | Current Project |
| ------- | --------------- |
| Safe margin from canvas edge | 56px |
| Content block gap | 24-36px |
| Icon-text gap | 10-12px |
| Card gap | 22-28px |
| Card padding | 20-26px |
| Card border radius | 6px |

---

## VI. Icon Usage Specification

### Source

- **Built-in icon library**: `tabler-outline`
- **Stroke width**: 2
- **Usage method**: `<use data-icon="tabler-outline/name" .../>`

### Recommended Icon List

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| 研究进度 | `tabler-outline/check` | P03 |
| 技术路线 | `tabler-outline/route` | P04 |
| 半车模型 | `tabler-outline/car` | P05 |
| SPDF 控制 | `tabler-outline/activity` | P06-P07 |
| 安全门控 | `tabler-outline/shield-check` | P09-P10 |
| Transformer / RL | `tabler-outline/brain` | P09 |
| 仿真平台 | `tabler-outline/cpu` | P11 |
| 数据结果 | `tabler-outline/chart-line` | P12-P13 |
| 计划安排 | `tabler-outline/calendar-time` | P16 |

---

## VII. Visualization Reference List

Catalog read: 71 templates.

| Page | Template | Path | Summary-quote (verbatim) | Usage |
| ---- | -------- | ---- | ------------------------ | ----- |
| P03 | kpi_cards | `templates/charts/kpi_cards.svg` | "Pick for 4-8 standalone numeric metrics shown as overview cards (2x2 or 1x4) — exec summary opener, dashboard headline, quarterly recap, results-at-a-glance. Skip if metrics have target baselines (use bullet_chart) or single hero number (use gauge_chart)." | 中期进展总览：成果、平台、算法、实验 |
| P04 | process_flow | `templates/charts/process_flow.svg` | "Pick for 3-8 sequential steps connected by simple arrows — approval workflows, customer onboarding, request handling, lifecycle stages. Skip if cyclical (use circular_stages) or stages produce named outputs (use pipeline_with_stages)." | 背景问题到两条技术路线 |
| P16 | timeline | `templates/charts/timeline.svg` | "Pick for 3-8 milestone events on a horizontal time axis (no duration). Skip for tasks with start/end ranges (use gantt_chart) or vertical layout (use roadmap_vertical)." | 后续研究计划 |

**Runners-up considered**

- `agenda_list` | rejected for P02: 提纲页需要简洁索引，直接自由排版更贴近参考答辩风格。
- `comparison_table` | rejected for P12: 全方法结果过多，页面采用重点指标 + 原图组合，比完整比较表更适合投影。
- `roadmap_vertical` | rejected for P16: 时间轴仅 4 个连续阶段，横向 timeline 更符合 16:9 答辩页。

---

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Acquire Via | Status | Reference | text_policy | page_role |
| -------- | ---------- | ----- | ------- | ---- | -------------- | ----------- | ------ | --------- | ----------- | --------- |
| research_framework_flow.png | 2647x3462 | 0.76 | 研究框架图，用于技术路线页 | Diagram | `#10 Centered image with radial callouts pointing outward` + `#21 Rounded rectangle crop` | user | Existing | 课题总体研究框架 |  | local |
| spdf_controller_framework.png | 1920x1080 | 1.78 | SPDF 控制结构图 | Diagram | `#3 Right-third image + left text body` + `#21 Rounded rectangle crop` | user | Existing | SPDF 控制器结构 |  | local |
| sirppo_policy_network.png | 3200x1800 | 1.78 | SI-RPPO 策略网络和安全执行链路 | Diagram | `#44 Background image + native network/architecture diagram` | user | Existing | SI-RPPO 策略网络与门控链路 |  | local |
| midterm_class_b_response.png | 2400x1400 | 1.71 | Class B 响应曲线 | Data figure | `#5 Top-band image + bottom multi-column text` + `#21 Rounded rectangle crop` | user | Existing | Class B 随机路面响应 |  | local |
| midterm_class_c_response.png | 2400x1400 | 1.71 | Class C 响应曲线备用 | Data figure | `#47 Small multiples — 3–6 same-kind images in an evenly spaced row` | user | Existing | Class C 随机路面响应 |  | local |
| midterm_bump_response.png | 2400x1400 | 1.71 | 凸包路面响应备用 | Data figure | `#47 Small multiples — 3–6 same-kind images in an evenly spaced row` | user | Existing | 凸包路面响应 |  | local |
| midterm_control_input_class_b.png | 2000x1400 | 1.43 | 控制输入响应备用 | Data figure | `#19 Image floating in whitespace with thin frame and caption` | user | Existing | Class B 控制输入 |  | local |
| midterm_psd_summary_class_b.png | 2400x800 | 3.00 | PSD 摘要 | Data figure | `#5 Top-band image + bottom multi-column text` | user | Existing | Class B 功率谱密度摘要 |  | local |
| controller_original_preview.png | 1600x900 | 1.78 | 控制器开发参考图备用 | Diagram | `#58 Image fragment as decorative corner element` | user | Existing | 控制器原始预览 |  | local |

Image-as-canvas coverage: P09 使用 `#44 Background image + native network/architecture diagram`，满足 4 个以上图像页时至少一页采用 image-as-canvas 的要求。

---

## IX. Content Outline

### Part 1: Opening and Context

#### Slide 01 - 封面

- **Cover impact**: “模型控制的确定性”与“预瞄强化学习的泛化”并行推进；构图采用白底工程网格 + 双主线轨迹。
- **Layout**: 左侧大标题，右侧两条蓝色技术轨迹汇入主动悬架对象。
- **Title**: 基于道路预瞄信息的主动悬架控制策略研究
- **Subtitle**: 中期答辩
- **Info**: 2026 年 6 月；研究方向：主动悬架控制、道路预瞄、强化学习

#### Slide 02 - 本次汇报围绕“进展、结果、问题、计划”展开

- **Layout**: 四段式提纲，右侧细线时间感。
- **Core message**: 答辩重点是说明已完成两条技术主线，并交代后续如何补齐鲁棒性、预瞄误差和实时验证。
- **Content**: 研究进展；模型与指标；SPDF 方法和结果；SI-RPPO 方法和结果；当前问题；后续计划。

#### Slide 03 - 中期阶段已形成“模型控制 + 安全强化学习”的完整闭环

- **Layout**: 2x2 KPI 卡 + 底部一句阶段结论。
- **Visualization**: kpi_cards.
- **Core message**: 当前进展已经覆盖建模、方法、平台和实验四个关键环节。
- **Content**: SPDF 已发表 IJRNC/JCR Q1；统一半车仿真平台已搭建；SI-RPPO 已完成训练和消融；后续聚焦鲁棒性、能耗折中与非理想预瞄。

#### Slide 04 - 课题矛盾从“被动折中”推进到“两条互补控制主线”

- **Layout**: 问题链路 → SPDF → SI-RPPO 的流程图，配研究框架缩略图。
- **Visualization**: process_flow.
- **Core message**: SPDF 解决可解释和快速收敛，SI-RPPO 进一步利用预瞄信息和安全约束。
- **Content**: 被动悬架难以多路况兼顾；MPC 可用预瞄但在线优化敏感；RL 表达能力强但安全探索风险高；本课题用 SPDF 和 SI-RPPO 分别回答确定性控制与数据驱动泛化问题。

#### Slide 05 - 统一半车模型把舒适性、安全性和执行器需求放到同一评价框架

- **Layout**: 左侧模型/接口说明，右侧指标矩阵。
- **Core message**: 所有方法都在同一半车模型和同一指标体系下比较，结果具有横向可比性。
- **Content**: 状态包括车身垂向、俯仰、前后非簧载质量；指标覆盖车身加速度、俯仰加速度、悬架动行程、轮胎动载荷、控制力峰值和能耗；道路预瞄以未来前后轮高程序列输入策略网络。

### Part 2: SPDF

#### Slide 06 - SPDF 通过周期滞后滑模面获得实用指定时间收敛

- **Layout**: 左侧三步机制，右侧 SPDF 控制框架图。
- **Core message**: SPDF 的核心价值是将收敛时间和抖振抑制纳入可解释控制律设计。
- **Content**: 转化垂向/俯仰误差子系统；构造周期滞后滑模面；设计连续滑模控制律；Lyapunov 分析证明误差进入可调小邻域。

#### Slide 07 - SPDF 在随机和凸包路面下显著压低车身与俯仰振动

- **Layout**: 上方关键曲线图，下方三组数字对比。
- **Core message**: SPDF 在三类路况下均给出最强舒适性改善，但需要关注执行器和行程约束。
- **Content**: B 级随机路面 RMS(zb_ddot) 从 0.215 降至 0.00186；C 级随机路面降至 0.00195；凸包路面降至 0.000947；代价是控制力与悬架行程需要整定。

#### Slide 08 - SPDF 的强先验也暴露了进一步利用道路预瞄的必要性

- **Layout**: 左侧 SPDF 优势/边界，右侧转向 SI-RPPO 的桥接逻辑。
- **Core message**: 模型控制提供稳定先验，但预瞄信息、模型不确定性和在线泛化需要学习型控制补充。
- **Content**: SPDF 依赖模型和参数上界；对真实感知误差和参数失配敏感；道路预瞄没有被充分利用；因此以 SPDF 轨迹作为教师，训练安全强化学习策略。

### Part 3: SI-RPPO

#### Slide 09 - SI-RPPO 让 PPO 在安全教师门控下直接输出主动控制力

- **Layout**: SI-RPPO 策略网络图为主画面，叠加三个可编辑注释卡。
- **Core message**: SI-RPPO 将示教初始化、预瞄编码和动作可行性修正合成一个部署可用的在线 PPO 策略。
- **Content**: Transformer 编码道路预瞄；SPDF 轨迹行为克隆初始化；部署阶段安全回退为被动零主动力；动作屏蔽层约束幅值、变化率和安全裕度。

#### Slide 10 - 安全门控把“何时主动用力”变成状态相关决策

- **Layout**: 中央门控公式逻辑图 + 三类状态卡片。
- **Core message**: 低激励或安全裕度不足时保持保守，高激励且裕度充足时允许 PPO 主动修正。
- **Content**: 门控系数 g_t 趋近 0 时退化为被动安全教师；g_t 增大时策略获得主动控制自由度；约束修正量进入 PPO 损失，抑制不可行动作和高频控制力。

#### Slide 11 - 统一仿真平台支撑六类控制方法和七种路况横向评估

- **Layout**: 平台模块图 + 奖励归一化条形说明。
- **Core message**: 平台层已把模型、路面、控制器、训练和评估脚本统一起来，支撑 SI-RPPO 的可复现实验。
- **Content**: 路况包括 ISO B/C、正弦、凸包、阶跃、凹坑、复合；控制器包括 PASSIVE、PID、SPDF、MPC、PPO scratch、SI-RPPO；奖励权重归一化为车身 0.45、俯仰 0.20、悬架 0.25、轮胎 0.10。

#### Slide 12 - 横向结果显示：SPDF 最强减振，SI-RPPO 综合回报最优

- **Layout**: 左侧重点指标表，右侧舒适性/能耗权衡图或指标柱状图。
- **Core message**: SI-RPPO 相比 PPO scratch 显著提升回报和舒适性，而 SPDF 仍是强减振上界。
- **Content**: 平均回报 SI-RPPO 为 -578，优于 PPO scratch 的 -762；车身加速度 RMS 从 0.274 降至 0.203；俯仰加速度 RMS 从 1.192 降至 0.659；控制能耗从 461 J 增至 1585 J，体现能耗--舒适性折中。

#### Slide 13 - 消融实验说明：示教初始化和安全门控带来主要收益

- **Layout**: PPO scratch vs SI-RPPO 对比卡 + 训练回报摘要。
- **Core message**: SI-RPPO 的收益来自 SPDF 先验、门控回退和约束反馈的组合。
- **Content**: 预训练模型全场景均值回报约 -941，PPO 微调后达到 -578；平均门控系数约 0.353；低激励接近被动教师，高激励允许主动介入。

#### Slide 14 - 阶段结论：两条主线已经打通，但目标从“能控”转向“鲁棒可用”

- **Layout**: 三条结论横向排列 + 底部转入问题。
- **Core message**: 中期成果证明方法有效，后半程重点是鲁棒性、非理想预瞄和实时验证。
- **Content**: SPDF 完成理论与发表成果；SI-RPPO 完成平台训练与横向验证；接下来需要让算法面对参数失配、预瞄噪声和硬件实时约束。

### Part 4: Problems and Plan

#### Slide 15 - 当前剩余问题集中在模型鲁棒性、能耗折中和预瞄误差

- **Layout**: 四个问题块，右侧用红色细线标出风险链。
- **Core message**: 现阶段问题不是方法不可行，而是要把仿真有效性推进到复杂不确定条件。
- **Content**: SPDF 依赖准确模型和参数上界；MPC 权重与专家数据质量需改进；SI-RPPO 仍以更高能耗换取舒适性；理想预瞄尚未覆盖噪声、时延、截断和缺失；硬件在环仍在准备。

#### Slide 16 - 后续工作按“鲁棒性补强 → 预瞄误差 → 实时验证 → 论文汇总”推进

- **Closing impact**: 最后留下明确路线：把已验证的控制效果推进到非理想预瞄和实时验证。
- **Layout**: 横向时间轴 + 底部致谢。
- **Visualization**: timeline.
- **Content**: 2026.06-08 补充 SPDF 鲁棒性、MPC 整定、SI-RPPO 消融；2026.08-10 构建非理想预瞄模型；2026.10-12 开展硬件在环或实时验证；2026.12-2027.01 汇总论文和学位论文。

---

## X. Speaker Notes Requirements

- **Filename**: `notes/total.md` 先按 `# 01_cover` 等标题组织，再由 `total_md_split.py` 拆分。
- **Content**: 每页 60-100 字，先讲结论，再讲图表或数据含义，最后过渡到下一页。
- **Duration**: 约 12-15 分钟。
- **Notes style**: 正式、清晰、答辩口吻。

---

## XI. Technical Constraints Reminder

1. SVG viewBox must be `0 0 1280 720`.
2. Text wrapping uses `<tspan>`; `<foreignObject>` is forbidden.
3. Use `fill-opacity` / `stroke-opacity`; `rgba()` is forbidden.
4. Forbidden: `<style>`, `class`, `textPath`, `animate*`, `script`, `<g opacity>`.
5. HTML named entities are forbidden except XML-reserved escapes such as `&amp;`.
6. Use only colors, fonts, icons, and images listed in `spec_lock.md`.
