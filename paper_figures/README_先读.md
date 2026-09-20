# 论文图表与可编辑 Case Study

本目录于2026-09-20补齐，来自 `runs/paper-2x2-rl-20260919`。原目录31个顶层文件完整保留；新增可编辑PPT、随附数据的重画脚本、字体和说明。

## 图表索引

| 文件前缀 | 内容 | 主要格式 |
| --- | --- | --- |
| fig1_reward_2x2 | 四组GRPO / VPO λ4训练reward（含soft length penalty） | PNG / PDF |
| fig1b_raw_reward_2x2 | 原始RM分数 | PNG / PDF |
| fig2_length_2x2 | 回答token长度 | PNG / PDF |
| fig3_kl_2x2 | 相对初始化模型的KL（k3估计） | PNG / PDF |
| fig4_entropy_2x2 | 回答token entropy | PNG / PDF |
| fig5_allocator_2x2 | credit ESS与权重标准差 | PNG / PDF |
| fig6_credit_case_heatmaps_paper | 四案例论文版token-credit heatmap | PNG / PDF / SVG / JSON / CSV |
| fig7_qwen_base_ablations | λ2/4/8与Random-direction消融的reward、length、ESS | PNG / PDF |
| fig6_credit_case_heatmaps_editable | 四案例的可编辑PowerPoint | PPTX |

`rl_metrics_2x2.csv` 包含8条主运行、2000个rollout；`rl_metrics_qwen_base_ablation.csv` 包含5条消融运行、1250个rollout。保留原始点和25-rollout trailing mean的原绘图口径。

Base实验为Base→SFT→RL，Instruct实验为直接RL。KL、长度与其他曲线均使用这些对应运行。不同RM家族的绝对reward尺度不能直接横向比较。

## 在PowerPoint里编辑heatmap

打开 `fig6_credit_case_heatmaps_editable.pptx`：

- 第1页：完整四案例论文图。
- 第2–5页：每个案例的放大版本，便于分别修改和排版。
- 标题、prompt、响应正文、advantage/rank、图例文字都是原生文本框。每个token的背景是独立矩形；不是截图。
- 每个案例组成一个 `CASE | ...` 组。双击进入组编辑，或右键选择“组合→取消组合”；也可在“选择窗格”中按token位置、ID和权重寻找色块。
- 正文按整行保留连续文字，避免人为插入subword间隔。文字与色块是独立对象；改动文字、字体或换行后，需要同时调整对应色块。
- PPT使用Arial、Times New Roman、Courier New；图例中的下标使用原生文字格式。原静态图使用的开放字体随附在 `fonts/`，供脚本复现。
- 每页备注保留该页的完整prompt、完整response及逐token数据；原始数据另见同名paper JSON和 `*_tokens.csv`。

已使用LibreOffice Impress实际打开并导出5页PDF，逐页检查排版；预览见 `fig6_credit_case_heatmaps_editable_preview.pdf` / `.png`。

76个token的原始权重保持不变，四个案例共用固定 `[0.25,4]` 区间与平方根映射。颜色表示credit weight，不是独立的token reward；正负学习信号来自 `A × w_t`。案例为人工挑选的定性示例。

## 下载后在任意机器重画

在本目录运行（Python3.10及以上）：

```bash
python -m pip install -r requirements_figures.txt
python reproduce_training_figures.py
python reproduce_case_study.py
python make_editable_heatmap.py
```

前两个命令默认输出到 `reproduced/`，也可加 `--out 指定目录`。第三个命令重新生成可编辑PPT。它们读取随附CSV/JSON，不需要下载模型或访问原服务器；已经验证重画的七组训练PNG和四案例heatmap PNG与原文件一致。

`make_paper_figs.py`、`find_credit_cases.py`、`render_case_heatmaps.py`、`render_credit_case_study_original.py` 是保留的原绘图/选例代码；原训练文件路径仅作为来源记录。需要直接从原始训练dump重新选例时才使用这些原环境脚本。

`fig6_credit_case_heatmaps.png` 是较早的十案例预览，配套原代码一并保留。论文与可编辑PPT采用改进后的四案例paper版本，其完整色阶说明见 `fig6_credit_case_heatmaps_paper.md`。

文件完整性与此次补充内容见 `UPLOAD_MANIFEST.json`。此目录的更新独立于上一级旧ZIP；新增PPT和展开文件请从本目录下载。
