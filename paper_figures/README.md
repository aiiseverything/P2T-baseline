# 论文图表与可编辑 Heatmap

本目录保存 2026-09-20 已上传并通过云端校验的 53 个图表包文件，另附本页 GitHub 导航。详细清单、复现命令和 PPT 编辑说明见 [README_先读.md](README_先读.md)。

- [可编辑 PowerPoint](fig6_credit_case_heatmaps_editable.pptx)：一页完整四案例图和四页单案例放大图；文字、token 色块、图例均为原生对象。
- [五页 PDF 预览](fig6_credit_case_heatmaps_editable_preview.pdf)。已在 LibreOffice Impress 打开、导出并逐页检查；未声称使用 Microsoft PowerPoint 实测。
- [图表索引](FIGURE_INDEX.csv)：reward、raw RM reward、length、KL、entropy、allocator ESS / 权重标准差、Qwen-Base 消融和四案例 heatmap。
- [数据与文件 SHA256 / MD5 清单](UPLOAD_MANIFEST.json)；[八张 PNG 的逐文件复现验证](REPRODUCTION_VALIDATION.json)。
- [Google Drive 同步目录](https://drive.google.com/drive/folders/12a-jhmrgIYsUiyxrb_JoewxHHbQlLXz_)。

安装 `requirements_figures.txt` 后，可运行 `reproduce_training_figures.py`、`reproduce_case_study.py` 和 `make_editable_heatmap.py`；它们读取随附 CSV / JSON，不需要模型或训练服务器。两个 `reproduce_*` 脚本默认输出到本目录的 `reproduced/`。原始脚本与来源路径同时保留，便于核对旧环境。

图中的 Random / Random-direction 是已完成的随机方向消融，不能改称 Shuffle；真正的 Shuffle 与 Norm-product 定义见 [消融说明](../docs/credit-controls-2026-09-20.md)。
