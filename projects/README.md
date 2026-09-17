# 研究材料导航

| 目录 | 保留内容 |
| --- | --- |
| `active_suspension_midterm_ppt_master/` | 原版中期答辩 PPT、论文 PDF/TeX、开题参考 PPT、图表素材及制作分析记录 |
| `active_suspension_midterm_rebuild_ppt169_20260622/` | 重排版答辩的 SVG 源稿、最终 SVG、讲稿、设计规范、素材和最终 PPT |
| `active_suspension_diagrams/` | 从临时目录迁入的 SI-RPPO Transformer 与预瞄编码器独立 SVG 图稿 |

最终演示文稿：

- [原版中期答辩](active_suspension_midterm_ppt_master/exports/active_suspension_midterm_defense_20260622_214542.pptx)
- [重排版中期答辩](active_suspension_midterm_rebuild_ppt169_20260622/exports/active_suspension_midterm_rebuild_20260622_230218.pptx)

## 编辑与清理约定

`sources/`、`images/`、`icons/`、`notes/`、`analysis/` 和设计说明保留制作依据。
`svg_output/` 是可编辑源稿；`svg_final/` 包含完成嵌入处理的图形，两者不是重复备份。
两个答辩项目各自引用的相同 PNG 仍分别保存，方便独立编辑。

`ppt_render_png/`、`validation/`、`confirm_ui/` 和已确认与源稿相同的 `backup/`
属于可再生成或重复的中间产物，已清理并加入忽略规则。重新导出时保留完整源稿与最终
`exports/`，不要以临时截图替代原始素材。
