#!/usr/bin/env python3
"""Render the paper's curated VPO token-credit excerpts from saved rollouts.

No weights, tokens, or advantages are inferred from the previous PNG. The
shared square-root color scale covers the full [1/lambda, lambda] band.
Token backgrounds follow measured glyph positions, without inserting spaces
inside words. PNG, vector PDF/SVG, and an auditable JSON/CSV accompany the plot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs/paper-2x2-rl-20260919"
DEFAULT_TRAIN = ROOT / "runs/rl-fp32-is-canonical-20260917/lam4/train"
STEM = "fig6_credit_case_heatmaps_paper"

# Exact substrings of the saved responses, not edited or reconstructed prose.
CASES = [
    dict(
        key="terminology", rollout=1, row=48, kind="positive",
        title="Refining an imprecise term",
        prompt_summary="What are the lasting legacies of the Greek Empire?",
        excerpt="The Greek Empire, or more accurately, Ancient Greece, has left a lasting legacy in various aspects of our world today.",
    ),
    dict(
        key="accountability", rollout=2, row=18, kind="positive",
        title="Giving concrete guidance",
        prompt_summary="How can a company promote transparency and accountability?",
        excerpt="Encourage leaders to model transparency and accountability, and hold them accountable for their actions and decisions.",
    ),
    dict(
        key="rainforest", rollout=3, row=31, kind="negative",
        title="Using a misleading category",
        prompt_summary="What physical factors contribute to the Amazon Rainforest's distinct wildlife?",
        excerpt="Tap water: The Amazon Rainforest has a high level of precipitation,",
        suffix=" …",
    ),
    dict(
        key="deduplication", rollout=1, row=45, kind="negative",
        title="Producing an invalid string transformation",
        prompt_summary="Remove duplicate characters from QFAWjYIbRIJWvLvKLmdNsW.",
        excerpt="here is the output without duplicate characters:\n\nQFAWjYIKmsNbvmdw",
        prefix="… ",
        monospace=True,
        reference="QFAWjYIbRJvLKmdNs",
    ),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(train, tokenizer_path):
    import numpy as np
    import torch
    from tokenizers import Tokenizer

    manifest_path = train / "profile_manifest.json"
    config = json.loads(manifest_path.read_text())["config"]
    band_top = float(config["credit_lambda"])
    group_size = int(config["group_size"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cache, cases, sources = {}, [], {str(manifest_path): sha256(manifest_path)}
    for spec in CASES:
        roll, row = spec["rollout"], spec["row"]
        if roll not in cache:
            paths = {name: train / f"rollout-{roll}-{name}.{ext}" for name, ext in
                     [("tokens", "json"), ("prompts", "json"), ("rewards", "json"), ("credit", "pt")]}
            cache[roll] = {name: (torch.load(path, weights_only=True, map_location="cpu")
                                 if name == "credit" else json.loads(path.read_text()))
                           for name, path in paths.items()}
            sources.update({str(path): sha256(path) for path in paths.values()})
        data = cache[roll]
        ids = data["tokens"][row]
        pieces = [tokenizer.decode([tid], skip_special_tokens=False) for tid in ids]
        full_text = tokenizer.decode(ids, skip_special_tokens=False)
        if "".join(pieces) != full_text:
            raise ValueError("Token decoding does not preserve this response's characters")
        wanted = spec["excerpt"]
        if full_text.count(wanted) != 1:
            raise ValueError(f"Expected a unique verbatim excerpt: {spec['key']}")
        start, end = full_text.index(wanted), full_text.index(wanted) + len(wanted)
        offsets = np.cumsum([0] + [len(piece) for piece in pieces]).tolist()
        lo = next(i for i in range(len(ids)) if offsets[i + 1] > start)
        hi = next((i for i in range(lo, len(ids)) if offsets[i] >= end), len(ids))
        if full_text[offsets[lo]:start].strip() or full_text[end:offsets[hi]].strip():
            raise ValueError("Excerpt would cut through a non-whitespace token fragment")
        ws = data["credit"]["w"][row, :len(ids)].float().numpy()
        if not np.isfinite(ws).all() or not (ws >= 1 / band_top - 1e-3).all() or not (ws <= band_top + 1e-3).all():
            raise ValueError("Saved weights are outside the declared band")
        peak = int(ws.argmax())
        if not lo <= peak < hi:
            raise ValueError("Selected excerpt must contain the response's peak weight")
        reward = data["rewards"][row]
        advantage = float(reward["advantage"])
        if (advantage > 0) != (spec["kind"] == "positive"):
            raise ValueError("Case label and saved advantage disagree")
        group_start = row - row % group_size
        group_rewards = data["rewards"][group_start:group_start + group_size]
        rank = 1 + sum(float(r["reward"]) > float(reward["reward"]) for r in group_rewards)
        if rank != (1 if advantage > 0 else group_size):
            raise ValueError("Curated cases must be the best/worst in their prompt group")
        prompts = data["prompts"]
        if isinstance(prompts, dict):
            prompts = prompts["prompts"]
        records = [dict(position=i, token_id=ids[i], text=pieces[i],
                        weight=float(ws[i]), signed_advantage=advantage * float(ws[i]))
                   for i in range(lo, hi)]
        cases.append(dict(**spec, advantage=advantage, reward=reward["reward"],
                          group_rank=rank, group_size=group_size,
                          prompt=prompts[row // group_size], full_response=full_text,
                          token_start=lo, token_end_exclusive=hi, peak_position=peak,
                          peak_weight=float(ws[peak]), tokens=records,
                          saved_weight_dtype=str(data["credit"]["w"].dtype)))
    sources[str(tokenizer_path)] = sha256(tokenizer_path)
    return cases, band_top, sources


def display_spans(case):
    """Map saved tokens into displayed text; normalize whitespace only."""
    text, spans = case.get("prefix", ""), []
    for token in case["tokens"]:
        piece = re.sub(r"[^\S\n]+", " ", token["text"])
        piece = re.sub(r"\n+", "\n", piece)
        start = len(text)
        text += piece
        spans.append((start, len(text), token))
    text += case.get("suffix", "")
    return text, spans


def wrap_ranges(text, width, measure):
    """Balance line lengths at whitespace; never split words or subwords."""
    from functools import lru_cache

    lines, paragraph_start = [], 0
    for paragraph in text.split("\n"):
        words = list(re.finditer(r"\S+", paragraph))
        if words:
            @lru_cache(None)
            def span_width(first, end):
                return measure(paragraph[words[first].start():words[end - 1].end()])

            count, first = 1, 0
            for end in range(1, len(words) + 1):
                if span_width(first, end) > width:
                    count += 1
                    first = end - 1
            target = span_width(0, len(words)) / count

            @lru_cache(None)
            def solve(first, remaining):
                if remaining == 0:
                    return (0, ()) if first == len(words) else (float("inf"), ())
                best = (float("inf"), ())
                for end in range(first + 1, len(words) - remaining + 2):
                    length = span_width(first, end)
                    if length > width:
                        break
                    cost, ends = solve(end, remaining - 1)
                    candidate = ((length - target) ** 2 + cost, (end,) + ends)
                    if candidate[0] < best[0]:
                        best = candidate
                return best

            cost, ends = solve(0, count)
            if cost == float("inf"):
                raise ValueError("An unbreakable word exceeds the available line width")
            first = 0
            for end in ends:
                lines.append((paragraph_start + words[first].start(),
                              paragraph_start + words[end - 1].end()))
                first = end
        paragraph_start += len(paragraph) + 1
    if any(measure(text[a:b]) > width for a, b in lines):
        raise ValueError("An unbreakable word exceeds the available line width")
    return lines


def render(cases, band_top, output, gamma, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
    from matplotlib.font_manager import FontProperties, findfont
    from matplotlib.patches import Rectangle
    import numpy as np

    plt.rcParams.update({
        "font.family": "Liberation Serif", "font.size": 10.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "mathtext.fontset": "stix", "axes.unicode_minus": True,
    })
    # Dark text remains readable even on the deepest warm-orange fill.
    palette = ["#fcf8f3", "#f7e8d9", "#f1d1b7", "#e8b08b", "#dc936b", "#ce7853"]
    ramp = LinearSegmentedColormap.from_list("warm_terracotta", palette, N=1024)
    norm = PowerNorm(gamma=gamma, vmin=1 / band_top, vmax=band_top, clip=True)
    ink, muted, accent, rule = "#29231f", "#72665e", "#9a4c30", "#e8ddd3"
    sans = "Lato"
    font_paths = {family: findfont(FontProperties(family=family), fallback_to_default=False)
                  for family in [sans, "Liberation Serif", "Liberation Mono"]}

    # All layout values are printer points; the PDF is a two-column-width figure.
    width, left, right, line_height = 518.4, 13.0, 505.4, 16.0
    gutter = 26.0
    column_width = (right - left - gutter) / 2
    columns = [left, left + column_width + gutter]
    fig = plt.figure(figsize=(width / 72, 6), dpi=144, facecolor="white")
    renderer = fig.canvas.get_renderer()

    def measure(text, family, size, weight="normal"):
        prop = FontProperties(family=family, size=size, weight=weight)
        return renderer.get_text_width_height_descent(text, prop, ismath=False)[0] * 72 / fig.dpi

    layouts, row_rules = [], []
    grouped = [[c for c in cases if c["kind"] == group] for group in ["positive", "negative"]]
    y = 88.0
    for row_index in range(2):
        row_layouts = []
        for column, column_cases in enumerate(grouped):
            case = column_cases[row_index]
            family = "Liberation Mono" if case.get("monospace") else "Liberation Serif"
            size = 9.2 if case.get("monospace") else 10.8
            text, spans = display_spans(case)
            lines = wrap_ranges(text, column_width - 2,
                                lambda s: measure(s, family, size))
            prompt = "Prompt: " + case["prompt_summary"]
            prompt_lines = wrap_ranges(prompt, column_width, lambda s: measure(s, sans, 7.6))
            body_offset = 38.0 + (len(prompt_lines) - 1) * 10.0 + 18.0
            case_height = body_offset + (len(lines) - 1) * line_height + 8 + (17 if case.get("reference") else 0)
            row_layouts.append(dict(case=case, x=columns[column], y=y, height=case_height,
                                   body_offset=body_offset, family=family, size=size,
                                   prompt=prompt, prompt_lines=prompt_lines,
                                   text=text, spans=spans, lines=lines))
        common_body_offset = max(layout["body_offset"] for layout in row_layouts)
        for layout in row_layouts:
            layout["height"] += common_body_offset - layout["body_offset"]
            layout["body_offset"] = common_body_offset
        layouts.extend(row_layouts)
        y += max(layout["height"] for layout in row_layouts)
        if row_index == 0:
            row_rules.append(y + 8)
            y += 18.0
    height = y + 33.0
    fig.set_size_inches(width / 72, height / 72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")

    def label(x, y, text, size, family=sans, color=ink, weight="normal", ha="left", math=False):
        return ax.text(x, y, text, fontsize=size, family=family, color=color,
                       fontweight=weight, ha=ha, va="baseline", parse_math=math)

    label(left, 19, "Token-level credit allocation", 14.5, "Liberation Serif", weight="bold")
    label(left, 34, "VPO  ·  Qwen3-14B-Base  ·  λ = 4", 8.0, color=muted)

    bar_x, bar_y, bar_w, bar_h = 352.0, 21.0, 151.0, 7.0
    label(bar_x, 14, r"Credit weight $w_t$", 7.6, color=muted, math=True)
    ax.imshow(np.linspace(0, 1, 1024).reshape(1, -1), cmap=ramp, aspect="auto",
              extent=(bar_x, bar_x + bar_w, bar_y + bar_h, bar_y),
              interpolation="nearest", zorder=1)
    for tick in [1 / band_top, 0.5, 1.0, 2.0, band_top]:
        x = bar_x + float(norm(tick)) * bar_w
        ax.plot([x, x], [bar_y + bar_h, bar_y + bar_h + 2.5], lw=.5, color=muted)
        label(x, bar_y + bar_h + 11, f"{tick:g}", 7.1, color=muted, ha="center")
    label(bar_x + bar_w, 49, "Shared square-root scale" if gamma == .5 else f"Shared power scale (γ = {gamma:g})",
          6.8, color=muted, ha="right")
    ax.plot([left, right], [55, 55], lw=.65, color=rule)

    for column, (heading, explanation) in enumerate([
        ("Positive advantage", "Deeper orange: stronger reinforcement"),
        ("Negative advantage", "Deeper orange: stronger penalty"),
    ]):
        label(columns[column], 71, heading, 10.0, color=accent, weight="bold")
        label(columns[column], 83, explanation, 7.6, color=muted)
    for rule_y in row_rules:
        for x in columns:
            ax.plot([x, x + column_width], [rule_y, rule_y], color=rule, lw=.55)
    ax.plot([(left + right) / 2] * 2, [64, height - 34], color=rule, lw=.55)

    token_boxes, peak_boxes = [], []
    for layout in layouts:
        case, top, x = layout["case"], layout["y"], layout["x"]
        index = cases.index(case)
        label(x, top + 12, f"({chr(97 + index)})  {case['title']}", 8.55, weight="bold")
        label(x, top + 25, f"A = {case['advantage']:+.2f}    ·    rank {case['group_rank']}/{case['group_size']}",
              7.5, color=muted)
        for prompt_no, (a, b) in enumerate(layout["prompt_lines"]):
            label(x, top + 38 + prompt_no * 10, layout["prompt"][a:b], 7.6, color=muted)
        text, family, size = layout["text"], layout["family"], layout["size"]
        for line_no, (a, b) in enumerate(layout["lines"]):
            baseline = top + layout["body_offset"] + line_no * line_height
            # One complete text line avoids artificial gaps at subword boundaries.
            for start, end, token in layout["spans"]:
                lo, hi = max(start, a), min(end, b)
                if lo >= hi:
                    continue
                while lo < hi and text[lo].isspace():
                    lo += 1
                while hi > lo and text[hi - 1].isspace():
                    hi -= 1
                if lo == hi:
                    continue
                x0 = x + measure(text[a:lo], family, size)
                x1 = x + measure(text[a:hi], family, size)
                fill = ramp(norm(token["weight"]))
                ax.add_patch(Rectangle((x0 - .28, baseline - size * .83), x1 - x0 + .56,
                                       size * 1.12, facecolor=fill, edgecolor="white",
                                       linewidth=.22, zorder=1))
                box = dict(case=case["key"], position=token["position"],
                           x=x0, y=baseline - size * .83, width=x1 - x0, height=size * 1.12,
                           right_boundary=x + column_width)
                token_boxes.append(box)
                if token["position"] == case["peak_position"]:
                    peak_boxes.append(box)
            label(x, baseline, text[a:b], size, family)
        if case.get("reference"):
            reference_y = top + layout["body_offset"] + len(layout["lines"]) * line_height + 1
            label(x, reference_y, "Reference: " + case["reference"], 7.1,
                  "Liberation Mono", color=muted)
    label(left, height - 12, r"Color encodes $w_t$; the signed token advantage is $A\,w_t$.  Uniform credit: $w_t = 1$.",
          7.3, color=muted, math=True)

    # Check actual rendered bounds before export, including all metadata.
    fig.canvas.draw()
    bounds = fig.bbox
    for artist in ax.texts:
        extent = artist.get_window_extent(fig.canvas.get_renderer())
        if extent.x0 < bounds.x0 - .5 or extent.x1 > bounds.x1 + .5 or extent.y0 < bounds.y0 - .5 or extent.y1 > bounds.y1 + .5:
            raise ValueError(f"Text outside figure: {artist.get_text()!r}")
    if len(peak_boxes) != len(cases):
        raise ValueError("A peak token is missing or wrapped across lines")
    if any(b["x"] + b["width"] > b["right_boundary"] + .5 for b in token_boxes):
        raise ValueError("Token highlight extends beyond its line")

    def luminance(color):
        rgb = np.array(to_rgb(color))
        rgb = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        return float(rgb @ [.2126, .7152, .0722])

    contrast = (luminance(palette[-1]) + .05) / (luminance(ink) + .05)
    if contrast < 4.5:
        raise ValueError("Text contrast on the darkest highlight is too low")
    for extension in ["png", "pdf", "svg"]:
        path = output / f"{STEM}.{extension}"
        fig.savefig(path, dpi=dpi, facecolor="white", transparent=False)
        print(path)
    plt.close(fig)
    return dict(figure_width_inches=width / 72, figure_height_inches=height / 72,
                png_dpi=dpi, fonts=font_paths, palette=palette, gamma=gamma,
                color_mapping=f"((w - {1 / band_top:g}) / {band_top - 1 / band_top:g}) ** {gamma:g}",
                weight_min=1 / band_top, weight_max=band_top,
                normalization="one fixed scale for every case; no row-wise normalization",
                darkest_fill_text_contrast=contrast,
                token_boxes=token_boxes, peak_boxes=peak_boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/Qwen3-14B-Base/tokenizer.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gamma", type=float, default=.5)
    parser.add_argument("--dpi", type=int, default=360)
    args = parser.parse_args()
    if not 0 < args.gamma <= 1 or args.dpi < 72:
        parser.error("Use 0 < gamma <= 1 and dpi >= 72")
    output = args.out.resolve()
    approved = [ROOT.resolve(), Path("/data/VPO-RM").resolve()]
    if not any(output.is_relative_to(root) for root in approved):
        parser.error("Output must stay inside an approved VPO-RM project root")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    cases, band_top, sources = load_cases(args.train, args.tokenizer)
    rendering = render(cases, band_top, output, args.gamma, args.dpi)
    artifact = dict(training_directory=str(args.train), source_sha256=sources,
                    renderer_sha256=sha256(Path(__file__)), rendering=rendering, cases=cases,
                    selection="Four manually selected illustrative excerpts from the existing candidate pool; not a random sample.",
                    interpretation="Color encodes positive credit weights, not token reward labels. The signed learning signal is the saved response advantage multiplied by each weight.")
    (output / f"{STEM}.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
    with (output / f"{STEM}_tokens.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "rollout", "row", "advantage", "position", "token_id", "text", "weight", "signed_advantage"])
        writer.writeheader()
        for case in cases:
            for token in case["tokens"]:
                writer.writerow(dict(case=case["key"], rollout=case["rollout"], row=case["row"],
                                     advantage=case["advantage"], **token))
    print(f"Validated {len(cases)} excerpts and {sum(len(c['tokens']) for c in cases)} original tokens.")


if __name__ == "__main__":
    main()
