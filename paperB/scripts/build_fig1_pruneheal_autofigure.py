#!/usr/bin/env python3
"""Build Paper B Figure 1 as an editable proxy-validity SVG and export PDF/PNG.

The source values are frozen existing results. No experiment or model inference is
performed here. The resulting figure is subsequently passed through the existing
AutoFigure-Edit optimization stage for an auditable edit/reconstruction run.
"""
from __future__ import annotations

import html
from pathlib import Path

import cairosvg
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paperB" / "figures"
OUT_SVG = FIG_DIR / "fig1_pruneheal_autofigure.svg"
OUT_PDF = FIG_DIR / "fig1_pruneheal_autofigure.pdf"
OUT_PNG = FIG_DIR / "fig1_pruneheal_autofigure.png"

WIDTH = 1800
HEIGHT = 910


def esc(s: str) -> str:
    return html.escape(s)


def text(x, y, value, cls="body", anchor="start", fill=None, weight=None, size=None, rotate=None):
    a = [f'x="{x}"', f'y="{y}"', f'class="{cls}"', f'text-anchor="{anchor}"']
    if fill:
        a.append(f'fill="{fill}"')
    if weight:
        a.append(f'font-weight="{weight}"')
    if size:
        a.append(f'font-size="{size}"')
    if rotate is not None:
        a.append(f'transform="rotate({rotate} {x} {y})"')
    return f"<text {' '.join(a)}>{esc(value)}</text>"


def multiline(x, y, lines, cls="body", anchor="start", line_height=27,
              fill=None, weight=None, size=None):
    a = [f'x="{x}"', f'y="{y}"', f'class="{cls}"', f'text-anchor="{anchor}"']
    if fill:
        a.append(f'fill="{fill}"')
    if weight:
        a.append(f'font-weight="{weight}"')
    if size:
        a.append(f'font-size="{size}"')
    spans = []
    for i, line_text in enumerate(lines):
        spans.append(
            f'<tspan x="{x}" dy="{0 if i == 0 else line_height}">'
            f'{esc(line_text)}</tspan>'
        )
    return f"<text {' '.join(a)}>{''.join(spans)}</text>"


def line(x1, y1, x2, y2, stroke="#667085", width=2, dash=None):
    a = [
        f'x1="{x1}"', f'y1="{y1}"', f'x2="{x2}"', f'y2="{y2}"',
        f'stroke="{stroke}"', f'stroke-width="{width}"',
    ]
    if dash:
        a.append(f'stroke-dasharray="{dash}"')
    return f"<line {' '.join(a)}/>"


def rect(x, y, w, h, fill="#fff", stroke="#D0D5DD", sw=2, rx=10):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>'
    )


def circle(x, y, r, fill, stroke="#FFFFFF", sw=2):
    return (
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>'
    )


def polyline(points, stroke, width=4, dash=None):
    a = [
        f'points="{points}"', 'fill="none"', f'stroke="{stroke}"',
        f'stroke-width="{width}"', 'stroke-linejoin="round"',
        'stroke-linecap="round"',
    ]
    if dash:
        a.append(f'stroke-dasharray="{dash}"')
    return f"<polyline {' '.join(a)}/>"


def build_svg() -> str:
    p = []
    p.append(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}"
 viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
<title id="title">Perplexity improvement alone does not imply target recovery</title>
<desc id="desc">Two-panel proxy-validity figure. Panel A shows the single keep14 observed trajectory at 128k, 153.5k, and 200k steps. Panel B shows endpoint and null operating points, including a random-init content-score floor, a coupled ShortGPT construction, and a 25k-only full32 reference.</desc>
<style>
  .title {{ font-family: "Liberation Sans", "Arial", sans-serif; font-size: 38px; font-weight: 700; fill: #182230; }}
  .subtitle {{ font-family: "Liberation Sans", "Arial", sans-serif; font-size: 26px; fill: #475467; }}
  .panel {{ font-family: "Liberation Sans", "Arial", sans-serif; font-size: 30px; font-weight: 700; fill: #182230; }}
  .body {{ font-family: "Liberation Sans", "Arial", sans-serif; font-size: 27px; fill: #344054; }}
  .small {{ font-family: "Liberation Sans", "Arial", sans-serif; font-size: 25px; fill: #475467; }}
  .tiny {{ font-family: "Liberation Sans", "Arial", sans-serif; font-size: 25px; fill: #667085; }}
  .axis {{ font-family: "Liberation Sans", "Arial", sans-serif; font-size: 25px; fill: #667085; }}
  .axisStrong {{ font-family: "Liberation Sans", "Arial", sans-serif; font-size: 25px; font-weight: 700; fill: #344054; }}
  .value {{ font-family: "Liberation Sans", "Arial", sans-serif; font-size: 25px; font-weight: 700; fill: #182230; }}
  .chip {{ font-family: "Liberation Sans", "Arial", sans-serif; font-size: 25px; font-weight: 700; }}
</style>
<rect width="1800" height="910" fill="#FFFFFF"/>'''
    )
    p.append(text(60, 57, "PPL improvement alone does not imply target recovery", "title"))
    p.append(
        text(
            60, 94,
            "Observed OLMo-2-7B paths · one run per construction · same-source Dolmino PPL",
            "subtitle",
        )
    )

    p.append(rect(45, 120, 770, 700, "#FBFCFE", "#D0D5DD", 2, 14))
    p.append(rect(840, 120, 915, 700, "#FBFCFE", "#D0D5DD", 2, 14))
    p.append(text(72, 166, "A", "panel", fill="#175CD3"))
    p.append(text(113, 166, "Literal keep14 path", "panel"))
    p.append(text(867, 166, "B", "panel", fill="#B54708"))
    p.append(text(908, 166, "Endpoints and null points", "panel"))

    # Panel A: literal keep14 late trajectory.
    x0, x1 = 150, 700
    y0, y1 = 265, 610
    p.append(line(x0, y1, x1, y1, "#667085", 2))
    p.append(line(x0, y0, x0, y1, "#667085", 2))
    p.append(line(x1, y0, x1, y1, "#98A2B3", 2))
    for val in [10.5, 10.6, 10.7, 10.8, 10.9]:
        yy = y1 - (val - 10.5) / .4 * (y1 - y0)
        p.append(line(x0 - 7, yy, x0, yy, "#667085", 1.5))
        p.append(line(x0, yy, x1, yy, "#EAECF0", 1))
        p.append(text(x0 - 13, yy + 5, f"{val:.1f}", "axis", "end"))
    for val in [.25, .27, .29, .31, .33]:
        yy = y1 - (val - .25) / .08 * (y1 - y0)
        p.append(line(x1, yy, x1 + 7, yy, "#667085", 1.5))
        p.append(text(x1 + 13, yy + 5, f"{val:.2f}", "axis", "start"))
    p.append(text(83, 438, "in-domain PPL ↓", "axisStrong", "middle", rotate=-90))
    p.append(text(770, 438, "MMLU letter ↑", "axisStrong", "middle", rotate=90))
    p.append(
        text(
            425, 682,
            "CPT step · nominal token presentations",
            "axisStrong", "middle",
        )
    )

    steps = [128000, 153500, 200000]
    xs = [x0 + (s - 128000) / (200000 - 128000) * (x1 - x0) for s in steps]
    ppl = [10.826, 10.693, 10.561]
    mmlu = [.3012, .3124, .3191]
    yp = [y1 - (v - 10.5) / .4 * (y1 - y0) for v in ppl]
    ym = [y1 - (v - .25) / .08 * (y1 - y0) for v in mmlu]
    for xx, lab in zip(xs, ["128k · 33.6B", "153.5k · 40.2B", "200k · 52.4B"]):
        p.append(line(xx, y1, xx, y1 + 7, "#667085", 1.5))
        p.append(text(xx, y1 + 34, lab, "axis", "middle"))
    p.append(polyline(" ".join(f"{x},{y}" for x, y in zip(xs, yp)), "#175CD3", 5))
    p.append(
        polyline(
            " ".join(f"{x},{y}" for x, y in zip(xs, ym)),
            "#D92D20", 5, "10 7",
        )
    )
    for xx, yy, val in zip(xs, yp, ppl):
        p.append(circle(xx, yy, 8, "#175CD3"))
        p.append(text(xx, yy - 14, f"{val:.3f}", "value", "middle", fill="#175CD3"))
    for xx, yy, val in zip(xs, ym, mmlu):
        p.append(
            f'<rect x="{xx - 7}" y="{yy - 7}" width="14" height="14" '
            'rx="2" fill="#D92D20" stroke="#FFFFFF" stroke-width="2"/>'
        )
        p.append(text(xx, yy + 27, f"{val:.3f}", "value", "middle", fill="#B42318"))
    p.append(line(x0, y1, x1, y1, "#98A2B3", 1.5, "6 6"))
    p.append(text(x1 - 4, y1 - 9, "chance .250", "tiny", "end"))
    p.append(rect(105, 194, 250, 48, "#EFF8FF", "#B2DDFF", 1.5, 24))
    p.append(text(230, 226, "PPL 10.826 → 10.561", "chip", "middle", fill="#175CD3"))
    p.append(rect(375, 194, 235, 48, "#FEF3F2", "#FECDCA", 1.5, 24))
    p.append(text(492, 226, "MMLU .301 → .319", "chip", "middle", fill="#B42318"))
    p.append(rect(630, 194, 145, 48, "#F2F4F7", "#D0D5DD", 1.5, 24))
    p.append(text(702, 226, "base .605", "chip", "middle", fill="#475467"))
    p.append(
        multiline(
            82, 744,
            [
                "PPL improves, but MMLU remains 28.6 points below base.",
            ],
            "body", line_height=34, fill="#344054", weight=700,
        )
    )

    # Panel B: endpoint/null operating points. Construction details stay in the
    # caption/table so all figure text remains comfortably printable.
    names = ["base", "full32", "keep14", "ShortGPT", "frozen", "random"]
    p.append(
        text(
            875, 210,
            "base/full32: 32L · others: 16L · full32: 25k; trained endpoints: 200k",
            "small",
        )
    )

    py, ph = 270, 165
    p.append(text(887, 248, "in-domain PPL ↓", "axisStrong"))
    p.append(line(1000, py + ph, 1695, py + ph, "#98A2B3", 1.5))
    values_ppl = [7.398, 7.670, 10.561, 9.780, 12.797, 11.498]
    colors = ["#344054", "#667085", "#175CD3", "#EAAA08", "#7F56D9", "#12B76A"]
    bx = [1025 + i * 112 for i in range(6)]
    for xx, val, color, name in zip(bx, values_ppl, colors, names):
        hh = (val - 7.0) / 6.2 * ph
        p.append(rect(xx, py + ph - hh, 52, hh, color, color, 0, 4))
        p.append(text(xx + 26, py + ph - hh - 9, f"{val:.3f}",
                      "tiny", "middle", weight=700, fill=color))
        p.append(text(xx + 26, py + ph + 29, name, "tiny", "middle"))
    p.append(text(980, py + ph + 5, "7", "axis", "end"))
    p.append(text(980, py + 5, "13", "axis", "end"))

    my, mh = 535, 145
    p.append(text(887, 512, "MMLU answer-letter accuracy ↑", "axisStrong"))
    p.append(line(1000, my + mh, 1695, my + mh, "#98A2B3", 1.5))
    values_letter = [.605, .588, .319, .474, .262, .247]
    for idx, (xx, val, color, name) in enumerate(
            zip(bx, values_letter, colors, names)):
        hh = max(2, (val - .24) / .38 * mh)
        p.append(rect(xx, my + mh - hh, 52, hh, color, color, 0, 4))
        value_y = my + mh - hh - (22 if idx == 5 else 8)
        p.append(text(xx + 26, value_y, f"{val:.3f}",
                      "tiny", "middle", weight=700, fill=color))
        p.append(text(xx + 26, my + mh + 29, name, "tiny", "middle"))
    chance_y = my + mh - (.25 - .24) / .38 * mh
    p.append(line(1000, chance_y, 1695, chance_y, "#98A2B3", 1.3, "5 5"))
    p.append(text(992, chance_y - 7, "chance .250", "tiny", "end"))

    p.append(rect(875, 735, 285, 70, "#FEF3F2", "#FDA29B", 1.5, 9))
    p.append(
        multiline(
            1017, 763, ["Random: letter .247", "content .360"],
            "small", "middle", 29, fill="#B42318", weight=700,
        )
    )
    p.append(rect(1175, 735, 285, 70, "#FFFAEB", "#FEC84B", 1.5, 9))
    p.append(
        multiline(
            1317, 763,
            ["ShortGPT: stronger", "coupled construction"],
            "small", "middle", 29, fill="#B54708", weight=700,
        )
    )
    p.append(rect(1475, 735, 250, 70, "#F2F4F7", "#D0D5DD", 1.5, 9))
    p.append(
        multiline(
            1600, 763, ["full32: 25k only", "no 200k control"],
            "small", "middle", 29, fill="#475467", weight=700,
        )
    )

    p.append(rect(45, 840, 1710, 52, "#F8FAFC", "#D0D5DD", 1.5, 9))
    p.append(
        text(
            70, 875,
            "Supported: PPL improvement alone does not imply target recovery on these observed paths.",
            "body", weight=700, fill="#344054",
        )
    )
    p.append("</svg>")
    return "\n".join(p)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    etree.fromstring(svg.encode())
    OUT_SVG.write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(OUT_PDF))
    cairosvg.svg2png(
        bytestring=svg.encode(),
        write_to=str(OUT_PNG),
        output_width=3600,
        output_height=1840,
    )
    print(OUT_SVG)
    print(OUT_PDF)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
