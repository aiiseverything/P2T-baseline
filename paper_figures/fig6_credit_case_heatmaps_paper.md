# VPO token-credit case study

论文版本：`fig6_credit_case_heatmaps_paper.pdf`。同名 PNG 用于预览，SVG 用于矢量编辑。原来的十案例 PNG 和绘图脚本保留，便于比较。

## 视觉设计

- 双栏宽度 7.2 英寸；左侧两个正 advantage 案例，右侧两个负 advantage 案例。
- 正文使用 Liberation Serif，标题和注释使用 Lato，字符串示例使用 Liberation Mono。PDF 嵌入 TrueType 字体；SVG 将字形转为路径，避免换机器后字体替换。
- 白底、暖橙色连续色阶。最低端 `#fcf8f3`，最高端 `#ce7853`；深色文字在最深底色上的对比度约 4.75:1。
- 全图共用固定色阶：`s(w) = sqrt((w - 0.25) / 3.75)`，范围覆盖 λ = 4 对应的完整 `[0.25, 4]` 权重区间。图例刻度仍显示原始权重。
- 保留真实 subword token 边界；按完整文字行绘制前景，避免出现 `Enc ourage` 之类的人为空格。仅规范化换行和空白，并在截断的句中片段旁明确显示省略号。

原图将低于 1 的权重都映射到最低色阶，掩盖了这一段的差异。新图使用完整区间，并用平方根映射提高中低权重的可见度。所有案例共用同一映射，不做逐行归一化，也不修改 token 权重。

## Suggested paper caption

**Token-level credit allocation by VPO.** Selected response excerpts from the canonical Qwen3-14B-Base run with λ = 4. Warm-orange intensity encodes the saved credit weight w_t using the shared square-root mapping s(w_t) = √((w_t − 0.25) / 3.75); the colorbar reports the original weights. For positive-advantage responses (a, b), larger weights amplify positive token advantages; for negative-advantage responses (c, d), they amplify negative token advantages. The signed token advantage is A w_t, and w_t = 1 corresponds to uniform credit. The examples illustrate a terminology correction, concrete guidance, a misleading category, and an invalid character-deduplication output. Prompts are abridged, response text and subword weights are preserved, and rank is measured within the eight responses to the same prompt. These are manually selected qualitative examples.

## 案例与原始数据

所有案例取自 `runs/rl-fp32-is-canonical-20260917/lam4/train`，选择范围为已有 `credit_case_candidates.json` 中的候选。response 行号和 token 下标均从 0 开始；右端下标不包含在节选内。

| Panel | Rollout / row | Token interval | Saved advantage A | Peak token | w_t | A w_t |
| --- | --- | --- | ---: | --- | ---: | ---: |
| (a) | 1 / 48 | [0, 24) | +1.472865 | ` accurately` | 4.0 | +5.891459 |
| (b) | 2 / 18 | [238, 257) | +1.344076 | ` accountable` | 4.0 | +5.376305 |
| (c) | 3 / 31 | [129, 143) | −1.936108 | ` water` | 4.0 | −7.744431 |
| (d) | 1 / 45 | [40, 59) | −2.161689 | `bv` | 4.0 | −8.646756 |

正负标签来自保存的 response advantage，不把 response 排名直接当作 token 正确性的标注。颜色显示分配系数 w_t，而不是独立测量的 token reward。(d) 的 reference 按首次出现顺序保留输入字符去重得到；高权重 `bv` 位于错误输出中，不据此断言这个 token 是全部错误的唯一原因。

完整 prompt、完整 response、76 个原始 token 的 ID / 文本 / 权重 / signed advantage，以及输入文件的 SHA-256 均记录在同名 JSON；逐 token 表另存为 `fig6_credit_case_heatmaps_paper_tokens.csv`。原始 credit dump 的权重存储精度为 float16；图中使用其实际保存值。

## Reproduce

从项目根目录执行：

```bash
OMP_NUM_THREADS=1 /root/miniconda3/envs/sml/bin/python scripts/render_credit_case_study.py
```

当前环境已经有依赖，不需要下载模型或额外安装。`--gamma` 控制幂映射，默认 0.5；`--dpi` 默认为 360。所有生成文件和 Matplotlib 缓存均位于当前项目目录。

LaTeX 插图示例（将 PDF 放到论文的 `figures` 目录）：

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig6_credit_case_heatmaps_paper.pdf}
  \caption{Token-level credit allocation by VPO. ...}
  \label{fig:vpo-credit-cases}
\end{figure*}
```
