# 文档辅助工具

此目录服务于研究材料制作，独立于半车训练与评估入口。

| 工具 | 输入与输出 |
| --- | --- |
| `build_carsim_mpc_report.py` | 读取 `results/carsim_mpc_quick/` 中的历史 CSV，生成 Word 报告与图表到 `results/carsim_report/` |
| `create_metric_formula_slide.ps1` | 使用本机 PowerPoint COM 接口生成指标公式幻灯片；输出位置由脚本中的路径变量指定 |

报告脚本从原 `tmp_docx_qa/` 迁入，保留历史报告叙述。更换实验数据后须重新核对文字结论；
保留在 `docs/` 的定稿不会被该脚本覆盖。运行需要 Conda 环境中的 Matplotlib 和额外的
`python-docx`；后者属于文档工具依赖，不是控制实验的运行依赖。

公式幻灯片脚本依赖 Windows 和已安装的 PowerPoint。两项工具都不属于
`scripts/check_project.py` 的控制流程检查范围。
