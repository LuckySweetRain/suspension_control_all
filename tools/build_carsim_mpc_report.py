"""Rebuild the historical CarSim Word report into results/carsim_report/."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "carsim_mpc_quick"
OUT = ROOT / "results" / "carsim_report" / "CarSim_QP-MPC联合仿真技术报告.docx"
ASSET_DIR = OUT.parent / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)
OUT.parent.mkdir(parents=True, exist_ok=True)

# Resolved design system: standard_business_brief + memo_masthead.
INK = "172B4D"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
PALE_BLUE = "E8F1F8"
LIGHT_GRAY = "F2F4F7"
MID_GRAY = "667085"
GREEN = "287D3C"
GOLD = "8A6500"
RED = "9B1C1C"
WHITE = "FFFFFF"
TABLE_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120


def cn_font() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for p in candidates:
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            return font_manager.FontProperties(fname=str(p)).get_name()
    return "Microsoft YaHei"


CN_FONT = cn_font()
plt.rcParams["font.family"] = CN_FONT
plt.rcParams["axes.unicode_minus"] = False


def set_cell_shading(cell, fill: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths_dxa)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.first_child_found_in("w:tblInd")
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(TABLE_INDENT_DXA))
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for w in widths_dxa:
        gc = OxmlElement("w:gridCol")
        gc.set(qn("w:w"), str(w))
        grid.append(gc)
    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.first_child_found_in("w:tcW")
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(widths_dxa[idx]))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_run_font(
    run, size=None, bold=None, color=None, italic=None, name="Calibri", east_asia="Microsoft YaHei"
):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def add_field(paragraph, instruction):
    run = paragraph.add_run()
    fld_char = OxmlElement("w:fldChar")
    fld_char.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char, instr, sep, text, end])


def set_paragraph_border_bottom(paragraph, color=BLUE, size=10, space=5):
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is None:
        p_bdr = OxmlElement("w:pBdr")
        p_pr.append(p_bdr)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(size))
    bottom.set(qn("w:space"), str(space))
    bottom.set(qn("w:color"), color)
    p_bdr.append(bottom)


def add_para(
    doc,
    text="",
    *,
    bold=False,
    italic=False,
    color=None,
    size=None,
    align=None,
    before=0,
    after=6,
    keep=False,
    style=None,
):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.10
    p.paragraph_format.keep_together = keep
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold, italic=italic, color=color)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.167
    r = p.add_run(text)
    set_run_font(r)
    return p


def add_number(doc, text):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.167
    r = p.add_run(text)
    set_run_font(r)
    return p


def add_callout(doc, label, text, fill=PALE_BLUE, accent=BLUE):
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [TABLE_WIDTH_DXA])
    cell = table.cell(0, 0)
    set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.10
    r = p.add_run(label + "  ")
    set_run_font(r, bold=True, color=accent)
    r = p.add_run(text)
    set_run_font(r, color=INK)
    add_para(doc, "", after=2)


def add_table(doc, headers, rows, widths_dxa, numeric_cols=()):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for i, h in enumerate(headers):
        set_cell_shading(hdr.cells[i], LIGHT_GRAY)
        p = hdr.cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(str(h))
        set_run_font(r, bold=True, size=9.5, color=INK)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            p = cells[i].paragraphs[0]
            p.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if i in numeric_cols else WD_ALIGN_PARAGRAPH.LEFT
            )
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            r = p.add_run(str(value))
            set_run_font(r, size=9.2)
    set_table_geometry(table, widths_dxa)
    add_para(doc, "", after=2)
    return table


def keep_with_next(paragraph):
    paragraph.paragraph_format.keep_with_next = True


def add_caption(doc, text):
    p = add_para(
        doc,
        text,
        color=MID_GRAY,
        size=9,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        before=2,
        after=8,
        keep=True,
    )
    p.paragraph_format.keep_with_next = False
    return p


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def make_architecture_chart():
    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.1)
    ax.axis("off")
    boxes = [
        (0.25, 2.55, 2.0, 0.9, "CarSim 道路与车辆\n高保真整车对象", "#E8F1F8"),
        (2.85, 2.55, 2.0, 0.9, "39 路状态/轮端信号\n单位换算与状态重构", "#F2F4F7"),
        (5.45, 2.55, 2.0, 0.9, "道路空间表 + 轮位\n15 步前/后轮预瞄", "#FFF4D6"),
        (8.05, 2.55, 2.0, 0.9, "半车 QP-MPC\nADMM 在线求解", "#E9F5EC"),
        (7.35, 0.65, 2.7, 0.9, "前/后轴力 × 0.10\n左右各分配 1/2", "#E8F1F8"),
        (3.45, 0.65, 2.7, 0.9, "CarSim 基线阻尼力\n3 × CmpRD", "#F2F4F7"),
        (0.25, 0.65, 2.5, 0.9, "四角总输入力\n饱和到 ±5000 N", "#FDECEC"),
    ]
    for x, y, w, h, txt, fill in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=fill, edgecolor="#365B7D", linewidth=1.25)
        )
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=10.3, color="#172B4D")
    arrow = dict(arrowstyle="->", color="#365B7D", lw=1.5)
    ax.annotate("", (2.85, 3.0), (2.25, 3.0), arrowprops=arrow)
    ax.annotate("", (5.45, 3.0), (4.85, 3.0), arrowprops=arrow)
    ax.annotate("", (8.05, 3.0), (7.45, 3.0), arrowprops=arrow)
    ax.annotate("", (8.7, 1.55), (9.0, 2.55), arrowprops=arrow)
    ax.annotate("", (6.15, 1.10), (7.35, 1.10), arrowprops=arrow)
    ax.annotate("", (2.75, 1.10), (3.45, 1.10), arrowprops=arrow)
    ax.annotate(
        "", (1.50, 1.55), (1.35, 2.55), arrowprops=dict(arrowstyle="->", color="#9B1C1C", lw=1.5)
    )
    ax.text(
        5.25,
        0.18,
        "Simulink 在 0.5 ms CarSim 步长下运行；MPC 每 10 ms 更新一次",
        ha="center",
        fontsize=9.5,
        color="#667085",
    )
    path = ASSET_DIR / "carsim_mpc_architecture.png"
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_result_chart(metrics):
    labels = ["车身加速度 RMS", "俯仰加速度 RMS", "前悬架峰值行程", "后悬架峰值行程", "角点力峰值"]
    keys = [
        "body_acc_rms_mps2",
        "pitch_acc_rms_radps2",
        "max_front_jounce_m",
        "max_rear_jounce_m",
        "max_corner_force_N",
    ]
    base = [float(metrics[k]["Baseline"]) for k in keys]
    active = [float(metrics[k]["PreviewMPC"]) for k in keys]
    active_norm = [100 * a / b for a, b in zip(active, base, strict=True)]
    y = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    ax.barh([v + 0.18 for v in y], [100] * 5, height=0.34, label="基线", color="#B8C4D2")
    ax.barh([v - 0.18 for v in y], active_norm, height=0.34, label="预瞄 QP-MPC", color="#2E74B5")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(92, 101.5)
    ax.set_xlabel("相对基线（基线 = 100%，越低越好）")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    for i, val in enumerate(active_norm):
        ax.text(
            val - 0.1,
            i - 0.18,
            f"{100 - val:.2f}% 改善",
            ha="right",
            va="center",
            color="white" if val < 99.3 else "#172B4D",
            fontsize=8.7,
        )
    fig.tight_layout()
    path = ASSET_DIR / "metrics_normalized.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_sweep_chart(rows):
    s = [float(r["ResidualScale"]) for r in rows]
    imp = [float(r["BodyAccImprovementPercent"]) for r in rows]
    pitch = [float(r["PitchAccRMS_radps2"]) for r in rows]
    rear = [1000 * float(r["MaxRearJounce_m"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].plot(s, imp, marker="o", color="#2E74B5", lw=2)
    axes[0].axhline(0, color="#98A2B3", lw=1)
    axes[0].set(xlabel="残差比例", ylabel="车身加速度改善（%）", title="比例过大将恶化舒适性")
    axes[0].grid(alpha=0.25)
    axes[1].plot(s, pitch, marker="s", label="俯仰加速度 RMS", color="#8A6500", lw=2)
    ax2 = axes[1].twinx()
    ax2.plot(s, rear, marker="o", label="后悬架峰值行程", color="#287D3C", lw=2)
    axes[1].set(xlabel="残差比例", ylabel="俯仰加速度 (rad/s²)", title="姿态与行程的折中")
    ax2.set_ylabel("后悬架峰值行程 (mm)")
    axes[1].grid(alpha=0.25)
    lines = axes[1].lines + ax2.lines
    axes[1].legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    path = ASSET_DIR / "scale_sweep.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def configure_styles(doc):
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string("202124")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10
    for name, size, color, before, after in [
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ]:
        s = styles[name]
        s.font.name = "Calibri"
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = RGBColor.from_string(color)
        s._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        s.paragraph_format.space_before = Pt(before)
        s.paragraph_format.space_after = Pt(after)
        s.paragraph_format.keep_with_next = True
    for name in ("List Bullet", "List Number"):
        s = styles[name]
        s.font.name = "Calibri"
        s.font.size = Pt(11)
        s._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")


def setup_page(doc):
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(1)
    sec.bottom_margin = Inches(1)
    sec.left_margin = Inches(1)
    sec.right_margin = Inches(1)
    sec.header_distance = Inches(0.492)
    sec.footer_distance = Inches(0.492)
    header = sec.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run("CarSim 主动悬架联合仿真  |  QP-MPC 技术报告")
    set_run_font(r, size=8.5, color=MID_GRAY, bold=True)
    set_paragraph_border_bottom(p, color="D6DEE8", size=6, space=3)
    footer = sec.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run("内部技术资料  ·  2026-07-21  ·  ")
    set_run_font(r, size=8, color=MID_GRAY)
    add_field(p, "PAGE")


def h(doc, level, text):
    p = doc.add_heading(text, level=level)
    keep_with_next(p)
    return p


def build():
    metrics_rows = load_csv(RESULTS / "metrics.csv")
    metrics = {r["Metric"]: r for r in metrics_rows}
    sweep = load_csv(RESULTS / "scale_sweep.csv")
    arch = make_architecture_chart()
    chart = make_result_chart(metrics)
    sweep_chart = make_sweep_chart(sweep)

    doc = Document()
    configure_styles(doc)
    setup_page(doc)
    core = doc.core_properties
    core.title = "基于 CarSim 的半车道路预瞄 QP-MPC 联合仿真技术报告"
    core.subject = "主动悬架 MPC 控制器设计、CarSim 联合仿真及结果分析"
    core.author = "悬架控制项目组"
    core.keywords = "CarSim; Simulink; QP-MPC; 道路预瞄; 主动悬架"

    # Memo masthead title block.
    add_para(doc, "技术报告 / QUICK VALIDATION BASELINE", bold=True, color=BLUE, size=10, after=6)
    p = add_para(
        doc,
        "基于 CarSim 的半车道路预瞄 QP-MPC 联合仿真",
        bold=True,
        color=INK,
        size=23,
        after=4,
        keep=True,
    )
    p.paragraph_format.line_spacing = 1.0
    add_para(
        doc,
        "控制器设计、联合仿真架构、量化结果与动画视频导出",
        color=MID_GRAY,
        size=13,
        after=14,
        keep=True,
    )
    meta = [
        ("方案定位", "快速可运行版：半车 QP-MPC + CarSim 高保真整车"),
        ("仿真平台", "CarSim 2020 / MATLAB R2026a / Simulink"),
        ("控制对象", "前、后轴作动力；左右轮等分，不主动控制侧倾"),
        ("报告日期", "2026 年 7 月 21 日"),
    ]
    for label, value in meta:
        p = add_para(doc, "", after=2)
        r = p.add_run(f"{label}：")
        set_run_font(r, bold=True, color=INK, size=10.5)
        r = p.add_run(value)
        set_run_font(r, color="344054", size=10.5)
    rule = add_para(doc, "", after=12)
    set_paragraph_border_bottom(rule, color=BLUE, size=12, space=4)
    add_callout(
        doc,
        "结论",
        "已完成可运行的严格凸 QP-MPC 联合仿真基线：控制器在线求解有限时域二次规划，并将前后轴力左右等分输入 CarSim。当前默认比例 0.10，在既定 Bounce Sine Sweep 工况下，俯仰加速度 RMS 改善 2.07%，后悬架峰值行程改善 5.50%；车身加速度改善仅 0.19%，因此结果应定位为接口与算法闭环验证，而非最终性能定论。",
    )

    h(doc, 1, "1. 方案定位与技术边界")
    add_para(
        doc,
        "本方案用于尽快打通“道路预瞄—在线 QP—Simulink—CarSim 整车模型—结果评估”的完整链路。CarSim 负责高保真车辆与道路响应，MPC 内部使用四自由度半车线性预测模型，在每个控制周期内优化前、后轴作动力。控制力作为残差叠加到已能稳定工作的 CarSim 阻尼基线上，降低模型失配导致闭环恶化的风险。",
    )
    add_bullet(
        doc,
        "属于严格的有限时域凸 QP-MPC：含预测矩阵、性能代价、输入与输入变化率约束，以及在线迭代求解。",
    )
    add_bullet(doc, "不是完整四角整车 MPC：预测模型不含侧倾自由度，前后轴力分别左右等分。")
    add_bullet(
        doc, "当前结果只覆盖一个车辆、一个道路工况和一次参数组合，不能据此宣称显著优于基线。"
    )

    h(doc, 1, "2. MPC 控制器设计")
    h(doc, 2, "2.1 半车预测模型")
    add_para(
        doc,
        "预测对象为 4-DOF 半车模型，包括车身升沉、车身俯仰、前簧下质量和后簧下质量。状态、控制量与道路扰动定义为：",
    )
    add_callout(
        doc,
        "状态与输入",
        "x = [z_b, ż_b, θ, θ̇, z_wf, ż_wf, z_wr, ż_wr]ᵀ；u = [F_f, F_r]ᵀ；w = [z_rf, ż_rf, z_rr, ż_rr]ᵀ。",
        fill="F6F8FA",
        accent=DARK_BLUE,
    )
    add_para(
        doc,
        "连续模型在静平衡点附近线性化，并用矩阵指数进行精确零阶保持离散化：x(k+1)=A_d x(k)+B_d u(k)+E_d w(k)。预测输出包含车身垂向加速度、俯仰角加速度、前/后悬架动行程以及前/后轮胎挠度。",
    )

    h(doc, 2, "2.2 预测域与道路预瞄")
    add_table(
        doc,
        ["参数", "设定", "说明"],
        [
            ("控制周期", "0.010 s", "MPC 每 10 ms 更新一次"),
            ("预测步数", "15", "总预瞄时间 0.150 s"),
            ("CarSim 积分步长", "0.0005 s", "每个控制周期包含 20 个整车积分步"),
            ("仿真时长", "20 s", "与当前 Run 配置一致"),
        ],
        [2100, 1800, 5460],
    )
    add_para(
        doc,
        "道路预瞄不是简单使用当前轮端位移，而是读取 CarSim 的 ROAD_DZ 空间表，根据前、后轮当前 X 坐标和车速，将未来 15 步的空间位置映射为前后轮道路高度及高度变化率。这样可在路面激励到达车轮前进入优化问题。",
    )

    h(doc, 2, "2.3 有限时域 QP")
    add_para(
        doc,
        "将预测域内的状态递推堆叠为 X=Sₓx₀+SᵤU+S𝓌W，并对输出与控制量进行凝聚，得到只以控制序列 U 为决策变量的凸二次规划。核心目标可写为：",
    )
    add_callout(
        doc,
        "优化目标",
        "min  Σ (‖y(k)‖²_Q + ‖u(k)‖²_R + ‖Δu(k)‖²_S)，其中 y 同时惩罚车身/俯仰加速度、悬架行程和轮胎挠度。",
        fill="F6F8FA",
        accent=DARK_BLUE,
    )
    add_para(
        doc,
        "控制力按 ±5000 N 的轴力尺度归一化，以改善数值条件。约束同时施加在整个预测域，并在输出前再执行一次显式投影，确保实际下发的首个控制量满足限幅与速率限制。",
    )
    add_table(
        doc,
        ["约束/权重", "当前值", "工程含义"],
        [
            ("前/后轴力", "±5000 N", "预测序列硬约束"),
            ("轴力变化率", "±1.0×10⁵ N/s", "相当于每 10 ms 最大变化 1000 N"),
            ("悬架行程参考边界", "±0.080 m", "用于模型内行程尺度和约束构造"),
            ("输出权重", "[1.00, 0.35, 0.25, 0.25, 0.08, 0.08]", "重点抑制升沉和俯仰"),
            ("残差比例", "0.10", "MPC 轴力按 10% 叠加至基线"),
        ],
        [2500, 2100, 4760],
    )

    h(doc, 2, "2.4 在线求解与降级保护")
    add_para(
        doc,
        "由于当前 MATLAB 环境未安装 Optimization Toolbox，项目采用内置的 OSQP 风格 ADMM 迭代器，而不是 quadprog。求解器使用热启动，最大 400 次迭代，绝对/相对容差均为 2×10⁻⁴。若 CarSim 输出维度异常、出现非有限值或求解器不可用，控制器返回安全的零残差命令；随后总角点力仍经过 ±5000 N 饱和。",
    )

    h(doc, 1, "3. CarSim 联合仿真设计")
    doc.add_picture(str(arch), width=Inches(6.35))
    add_caption(doc, "图 1  CarSim—Simulink—QP-MPC 联合仿真闭环")
    h(doc, 2, "3.1 输入力分配")
    add_callout(
        doc,
        "四角力映射",
        "F_L1 = F_base,L1 + 0.10·F_f/2；F_R1 = F_base,R1 + 0.10·F_f/2；F_L2 = F_base,L2 + 0.10·F_r/2；F_R2 = F_base,R2 + 0.10·F_r/2。",
        fill="F6F8FA",
        accent=DARK_BLUE,
    )
    add_para(
        doc,
        "基线阻尼律为 3×CmpRD（CmpRD 单位 mm/s），MPC 输出只作为附加主动控制分量。前轴命令等分给左前和右前，后轴命令等分给左后和右后。这种映射可快速验证纵向姿态控制，但对左右不对称道路和侧倾控制能力有限。",
    )

    h(doc, 2, "3.2 I/O 适配与数据处理")
    add_para(
        doc,
        "当前 CarSim 的 Run_all.par 同时启用了 I/O 数据集与 Additional Data 命令，导致 4 个导入力重复为 8 路、39 个导出量重复为 77 路。适配层向两组导入口发送相同四角力，并从第 39 至 77 路读取第二组完整输出。该处理属于现有数据库的兼容措施；若后续在 CarSim GUI 中清理重复配置，应同步修改 simfile、S-Function 端口宽度和运行脚本。",
    )
    add_table(
        doc,
        ["信号组", "用途", "关键处理"],
        [
            ("车身信号", "Zcg、Vz、Az、Pitch、角速度/角加速度", "转换为半车状态与性能指标"),
            ("轮端信号", "四轮 jounce、jounce rate、轮胎力", "前后轮取左右平均，构造降阶状态"),
            ("道路定位", "四轮 X/Y 与车辆 Vx", "查表生成 0.15 s 空间预瞄"),
            ("控制输入", "IMP_FD_L1/L2/R1/R2", "总力限幅后发送至 CarSim"),
        ],
        [2100, 3200, 4060],
    )

    h(doc, 2, "3.3 可复现实验流程")
    for text in [
        "进入 carsim_mpc_quick，加入 src 与 tests 路径，运行 validate_quick_mpc_offline() 检查 QP 数值与约束。",
        "首次运行或 CarSim 接口变化后执行 build_carsim_mpc_model()，生成 Simulink 联合仿真模型和预测数据。",
        "运行 run_carsim_mpc_demo()；脚本依次执行 residual scale=0 的基线和 scale=0.10 的预瞄 MPC 工况。",
        "检查 results/carsim_mpc_quick 下的 metrics.csv、scale_sweep.csv、carsim_mpc_run.mat 以及 LastRun 动画数据。",
    ]:
        add_number(doc, text)

    h(doc, 1, "4. 仿真结果与分析")
    h(doc, 2, "4.1 离线 QP 检查")
    add_para(
        doc,
        "离线测试得到首步前轴力 −1000 N、后轴力 617.08 N；33 次 ADMM 迭代收敛，最大约束残差约 1.51×10⁻⁵。联合仿真中显式下发轴力峰值为 5000 N，单步轴力增量峰值为 1000 N，分别到达但未超过设定上限。",
    )

    h(doc, 2, "4.2 基线与预瞄 MPC 对比")
    table_rows = []
    label_map = {
        "body_acc_rms_mps2": ("车身加速度 RMS", "m/s²"),
        "pitch_acc_rms_radps2": ("俯仰加速度 RMS", "rad/s²"),
        "max_front_jounce_m": ("前悬架峰值行程", "m"),
        "max_rear_jounce_m": ("后悬架峰值行程", "m"),
        "max_corner_force_N": ("角点力峰值", "N"),
        "max_qp_constraint_violation": ("最大 QP 约束残差", "—"),
        "qp_converged_fraction": ("QP 收敛比例", "—"),
    }
    for row in metrics_rows:
        label, unit = label_map[row["Metric"]]
        b = float(row["Baseline"])
        a = float(row["PreviewMPC"])
        imp = float(row["ImprovementPercent"])
        if "fraction" in row["Metric"]:
            bv = f"{b * 100:.2f}%"
            av = f"{a * 100:.2f}%"
        elif unit == "N":
            bv = f"{b:.1f}"
            av = f"{a:.1f}"
        elif unit == "m":
            bv = f"{b:.5f}"
            av = f"{a:.5f}"
        else:
            bv = f"{b:.4f}"
            av = f"{a:.4f}"
        table_rows.append((label, unit, bv, av, f"{imp:+.2f}%"))
    add_table(
        doc,
        ["指标", "单位", "基线", "预瞄 QP-MPC", "改善"],
        table_rows,
        [2750, 900, 1600, 2400, 1710],
        numeric_cols=(1, 2, 3, 4),
    )
    doc.add_picture(str(chart), width=Inches(6.3))
    add_caption(doc, "图 2  主要性能指标相对基线对比（低值指标）")
    add_callout(
        doc,
        "结果解读",
        "预瞄控制对后悬架峰值行程和俯仰加速度的改善最明确；车身加速度 RMS 仅改善约 0.19%，接近小幅变化。说明当前半车预测模型与 CarSim 整车之间仍存在显著模型失配，控制器更适合作为联合仿真与在线 QP 的基线。",
        fill="FFF8E8",
        accent=GOLD,
    )

    h(doc, 2, "4.3 残差比例扫描")
    doc.add_picture(str(sweep_chart), width=Inches(6.35))
    add_caption(doc, "图 3  残差比例扫描及性能折中")
    add_para(
        doc,
        "比例扫描表明控制强度并非越大越好：scale=0.05 时车身加速度改善约 0.34%，但对俯仰和后悬架行程的改善弱于 0.10；scale=0.20 和 0.40 虽继续降低俯仰加速度，却开始恶化车身加速度，0.40 时恶化约 16.97%。因此默认选择 0.10 作为当前工况的折中值。更换道路、车速或车辆参数后必须重新整定。",
    )

    h(doc, 1, "5. CarSim 动画与视频导出")
    add_callout(
        doc,
        "可以导出",
        "CarSim 2020 的 VS Visualizer 在 Windows 下可以将全部或部分动画直接导出为 AVI；AVI 不含音频。当前 LastRun.vs / LastRun.vsb 已包含本次仿真的回放数据。",
        fill="E9F5EC",
        accent=GREEN,
    )
    for text in [
        "在 CarSim Run Control 页面打开 Video，或在 VS Visualizer 中打开 LastRun.vs。",
        "调整相机角度、缩放和动画窗口尺寸；窗口尺寸即导出视频分辨率。",
        "选择 File → Export Video，或在动画区域右键选择 Export Video。",
        "指定 AVI 文件路径；勾选 Set Video Time 可设置起止时间，不勾选则导出完整动画。",
        "确认后在 Video Compression 对话框选择系统已安装的编码器并开始导出。",
    ]:
        add_number(doc, text)
    add_para(
        doc,
        "若需要论文、答辩或网页常用的 MP4，建议先由 CarSim 导出无损或高质量 AVI，再用 ffmpeg 转码为 H.264 MP4。若导出菜单不可用，应先确认运行环境为 Windows，并检查显卡/OpenGL、编码器安装与 Visualizer 版本。",
        color=INK,
    )
    add_para(
        doc,
        "本节依据本机 CarSim 2020 官方手册《VS Visualizer Reference Manual》（2019-12），第 26–27 页“Export Video / The Process of Making an AVI Video File”。",
        size=9,
        color=MID_GRAY,
        after=8,
    )

    h(doc, 1, "6. 当前局限与下一步建议")
    add_table(
        doc,
        ["优先级", "建议", "目的"],
        [
            (
                "P0",
                "清理 CarSim 重复 I/O 配置，并固定 39 入参/输出映射",
                "降低接口脆弱性，避免错误读取信号",
            ),
            (
                "P0",
                "用 CarSim 扫频/随机路面数据辨识半车降阶模型",
                "缩小预测模型与高保真对象的动态失配",
            ),
            ("P1", "增加 Kalman 滤波或移动窗状态估计", "减少由差分和左右平均造成的状态噪声"),
            ("P1", "覆盖随机路、凸块、正弦扫频及多车速工况", "形成论文所需的统计性对比"),
            (
                "P2",
                "升级为四角整车 MPC，显式加入侧倾与四轮独立力",
                "获得横向不对称工况和侧倾控制能力",
            ),
            ("P2", "增加软约束与故障降级逻辑", "提高约束冲突时的可行性和工程鲁棒性"),
        ],
        [1100, 5000, 3260],
    )
    add_para(
        doc,
        "正式论文对比建议至少报告：车身垂向加速度 RMS/PSD、俯仰与侧倾角加速度、悬架动行程峰值、轮胎动载荷 RMS、控制能耗与约束触发比例；同时给出被动/基线阻尼、无预瞄 MPC、预瞄 MPC 三组消融。",
    )

    h(doc, 1, "附录 A 关键文件与结果")
    add_table(
        doc,
        ["类型", "位置/文件", "说明"],
        [
            (
                "控制器配置",
                "carsim_mpc_quick/quick_mpc_config.m",
                "车辆参数、预测域、约束、权重与 ADMM 参数",
            ),
            ("MPC 核心", "carsim_mpc_quick/src/preview_qp_solve.m", "凝聚 QP 与在线求解"),
            (
                "Simulink 接口",
                "carsim_mpc_quick/src/half_car_preview_qp_sfun.m",
                "CarSim 输出到四角输入力的实时适配",
            ),
            (
                "联合仿真入口",
                "carsim_mpc_quick/run_carsim_mpc_demo.m",
                "基线/主动工况运行与指标输出",
            ),
            ("量化结果", "results/carsim_mpc_quick/metrics.csv", "最终对比指标"),
            ("比例扫描", "results/carsim_mpc_quick/scale_sweep.csv", "0–0.40 残差比例敏感性"),
            (
                "动画数据",
                "results/carsim_mpc_quick/LastRun.vs, LastRun.vsb",
                "供 VS Visualizer 回放与导出 AVI",
            ),
        ],
        [1500, 4440, 3420],
    )

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
