#!/usr/bin/env python3
"""Generate Paper A Figure 1 as editable SVG plus vector PDF and high-res PNG.

The composition is a manual academic redraw informed by the AutoFigure-Edit
stage-1 import artifact under ``autofigure_run_A/``.  All visible elements are
native SVG vectors/text; no raster image is embedded in the final figure.
"""

from __future__ import annotations

import html
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg
from PIL import Image


WIDTH = 1600
HEIGHT = 900

NAVY = "#17324D"
INK = "#243746"
MUTED = "#5E6E7E"
LINE = "#9AABB8"
LIGHT_LINE = "#D5DEE5"
PAPER = "#FFFFFF"
PANEL = "#F7F9FB"
WRITE = "#4C78A8"
WRITE_LIGHT = "#EAF2F8"
SELECT = "#2A9D8F"
SELECT_LIGHT = "#E7F5F3"
QUERY = "#D97757"
QUERY_LIGHT = "#FCEFEA"
GOLD = "#C9952E"
GOLD_LIGHT = "#FBF4DF"
GRAY = "#7B8794"
GRAY_LIGHT = "#F1F3F5"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def attrs(**kwargs: object) -> str:
    out: list[str] = []
    for key, value in kwargs.items():
        if value is None:
            continue
        key = key.rstrip("_").replace("_", "-")
        out.append(f'{key}="{esc(value)}"')
    return " ".join(out)


class SVG:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, value: str) -> None:
        self.parts.append(value)

    def group_start(self, id_: str, **kwargs: object) -> None:
        extra = attrs(id=id_, **kwargs)
        self.add(f"<g {extra}>")

    def group_end(self) -> None:
        self.add("</g>")

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = PAPER,
        stroke: str = NAVY,
        sw: float = 2,
        rx: float = 10,
        dash: str | None = None,
        id_: str | None = None,
        opacity: float | None = None,
    ) -> None:
        self.add(
            "<rect "
            + attrs(
                id=id_,
                x=x,
                y=y,
                width=w,
                height=h,
                rx=rx,
                fill=fill,
                stroke=stroke,
                stroke_width=sw,
                stroke_dasharray=dash,
                opacity=opacity,
            )
            + "/>"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = NAVY,
        sw: float = 2.2,
        dash: str | None = None,
        marker: str | None = None,
        opacity: float | None = None,
    ) -> None:
        self.add(
            "<line "
            + attrs(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                stroke=stroke,
                stroke_width=sw,
                stroke_dasharray=dash,
                marker_end=f"url(#{marker})" if marker else None,
                opacity=opacity,
            )
            + "/>"
        )

    def path(
        self,
        d: str,
        *,
        fill: str = "none",
        stroke: str = NAVY,
        sw: float = 2.2,
        dash: str | None = None,
        marker: str | None = None,
        opacity: float | None = None,
    ) -> None:
        self.add(
            "<path "
            + attrs(
                d=d,
                fill=fill,
                stroke=stroke,
                stroke_width=sw,
                stroke_dasharray=dash,
                marker_end=f"url(#{marker})" if marker else None,
                opacity=opacity,
            )
            + "/>"
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        size: float = 24,
        weight: int = 400,
        fill: str = INK,
        anchor: str = "start",
        italic: bool = False,
        letter: float | None = None,
        cls: str | None = None,
        id_: str | None = None,
    ) -> None:
        self.add(
            "<text "
            + attrs(
                id=id_,
                x=x,
                y=y,
                fill=fill,
                font_size=size,
                font_weight=weight,
                text_anchor=anchor,
                font_style="italic" if italic else None,
                letter_spacing=letter,
                class_=cls,
            )
            + f">{esc(value)}</text>"
        )

    def multiline(
        self,
        x: float,
        y: float,
        lines: list[str] | tuple[str, ...],
        *,
        size: float = 23,
        weight: int = 400,
        fill: str = INK,
        anchor: str = "middle",
        line_height: float = 1.18,
        italic: bool = False,
        id_: str | None = None,
    ) -> None:
        self.add(
            "<text "
            + attrs(
                id=id_,
                x=x,
                y=y,
                fill=fill,
                font_size=size,
                font_weight=weight,
                text_anchor=anchor,
                font_style="italic" if italic else None,
            )
            + ">"
        )
        for idx, line in enumerate(lines):
            dy = 0 if idx == 0 else size * line_height
            self.add(
                f'<tspan x="{x}" dy="{dy:.1f}">{esc(line)}</tspan>'
            )
        self.add("</text>")

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        *,
        fill: str = PAPER,
        stroke: str = NAVY,
        sw: float = 2,
    ) -> None:
        self.add(
            "<circle "
            + attrs(
                cx=cx,
                cy=cy,
                r=r,
                fill=fill,
                stroke=stroke,
                stroke_width=sw,
            )
            + "/>"
        )

    def chip(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        *,
        fill: str = PAPER,
        stroke: str = NAVY,
        text_fill: str = INK,
        size: float = 22,
        weight: int = 500,
        dash: str | None = None,
        rx: float = 8,
    ) -> None:
        self.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.8, rx=rx, dash=dash)
        self.text(
            x + w / 2,
            y + h / 2 + size * 0.34,
            label,
            size=size,
            weight=weight,
            fill=text_fill,
            anchor="middle",
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = NAVY,
        sw: float = 2.2,
        dash: str | None = None,
    ) -> None:
        marker = {
            NAVY: "arrow-navy",
            WRITE: "arrow-write",
            SELECT: "arrow-select",
            QUERY: "arrow-query",
            GRAY: "arrow-gray",
        }.get(color, "arrow-navy")
        self.line(x1, y1, x2, y2, stroke=color, sw=sw, dash=dash, marker=marker)

    def render(self) -> str:
        body = "\n".join(self.parts)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  Paper A Figure 1: CoMem Write–Select–Read.
  Manual editable-vector reconstruction after AutoFigure-Edit stage-1 import.
  No raster image is embedded in this SVG.
-->
<svg xmlns="http://www.w3.org/2000/svg"
     width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}"
     role="img" aria-labelledby="figure-title figure-desc"
     shape-rendering="geometricPrecision">
  <title id="figure-title">CoMem Write–Select–Read architecture</title>
  <desc id="figure-desc">Context chunks are written once to a persistent split-depth store, a query selects a bounded top-k subset, and the selected cached residuals resume only the upper transformer layers. The figure compares raw-text j equals zero replay with cached h sub j continuation and includes an overlap-Write context variant.</desc>
  <defs>
    <style>
      text {{
        font-family: "Liberation Sans", "DejaVu Sans", Arial, Helvetica, sans-serif;
      }}
    </style>
    <marker id="arrow-navy" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L9,4.5 L0,9 z" fill="{NAVY}"/>
    </marker>
    <marker id="arrow-write" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L9,4.5 L0,9 z" fill="{WRITE}"/>
    </marker>
    <marker id="arrow-select" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L9,4.5 L0,9 z" fill="{SELECT}"/>
    </marker>
    <marker id="arrow-query" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L9,4.5 L0,9 z" fill="{QUERY}"/>
    </marker>
    <marker id="arrow-gray" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L9,4.5 L0,9 z" fill="{GRAY}"/>
    </marker>
  </defs>
{body}
</svg>
"""


def draw_stage_label(
    s: SVG,
    *,
    number: str,
    y: float,
    name: str,
    sub: str,
    color: str,
) -> None:
    s.circle(45, y, 22, fill=PAPER, stroke=color, sw=2.6)
    s.text(45, y + 8, number, size=25, weight=700, fill=color, anchor="middle")
    s.text(78, y + 8, name, size=30, weight=700, fill=NAVY)
    s.text(25, y + 43, sub, size=21, weight=500, fill=MUTED)


def build_svg() -> str:
    s = SVG()
    s.rect(0, 0, WIDTH, HEIGHT, fill=PAPER, stroke="none", sw=0, rx=0)

    # Header.
    s.group_start("header")
    s.text(
        WIDTH / 2,
        35,
        "CoMem: persistent Write – bounded Select – depth-reused Read",
        size=30,
        weight=700,
        fill=NAVY,
        anchor="middle",
    )
    s.text(
        WIDTH / 2,
        63,
        "Write each context chunk once; every query touches only a fixed top-k pack.",
        size=21,
        weight=500,
        fill=MUTED,
        anchor="middle",
    )
    s.group_end()

    # Stage background bands and separators.
    s.rect(10, 78, 1580, 267, fill="#FBFCFD", stroke=LIGHT_LINE, sw=1.2, rx=14)
    s.rect(10, 354, 1580, 166, fill="#FCFDFD", stroke=LIGHT_LINE, sw=1.2, rx=14)
    s.rect(10, 529, 1580, 306, fill="#FBFCFD", stroke=LIGHT_LINE, sw=1.2, rx=14)
    s.line(160, 87, 160, 827, stroke=LIGHT_LINE, sw=1.6)

    # ------------------------------------------------------------------ WRITE
    s.group_start("stage-write")
    draw_stage_label(s, number="1", y=115, name="WRITE", sub="once / context", color=WRITE)
    s.text(
        180,
        104,
        "Independent chunk writes (c = 512; chunk-local positions)",
        size=23,
        weight=700,
        fill=NAVY,
    )

    chunk_xs = [195, 355, 675]
    chunk_labels = ["chunk x₁", "chunk x₂", "chunk x_N"]
    for x, label in zip(chunk_xs, chunk_labels):
        s.chip(x, 121, 135, 43, label, fill=PAPER, stroke=WRITE, size=22, weight=600)
        s.arrow(x + 67.5, 165, x + 67.5, 184, color=WRITE)
        s.rect(x, 187, 135, 51, fill=WRITE_LIGHT, stroke=WRITE, sw=2, rx=9)
        s.multiline(
            x + 67.5,
            209,
            ["lower layers", "[0 : j)"],
            size=20,
            weight=650,
            fill=NAVY,
            line_height=1.08,
        )
        s.arrow(x + 67.5, 239, x + 67.5, 268, color=WRITE)

    s.text(590, 151, "⋯", size=32, weight=700, fill=MUTED, anchor="middle")
    s.text(590, 217, "⋯", size=32, weight=700, fill=MUTED, anchor="middle")

    s.rect(175, 268, 650, 61, fill=WRITE_LIGHT, stroke=WRITE, sw=2, rx=11)
    s.multiline(
        190,
        290,
        ["persistent", "store"],
        size=19,
        weight=700,
        fill=NAVY,
        anchor="start",
        line_height=1.05,
    )
    s.chip(300, 281, 120, 35, "h_j(x₁)", fill=PAPER, stroke=WRITE, size=20)
    s.chip(435, 281, 120, 35, "h_j(x₂)", fill=PAPER, stroke=WRITE, size=20)
    s.text(585, 307, "⋯", size=28, weight=700, fill=MUTED, anchor="middle")
    s.chip(625, 281, 140, 35, "h_j(x_N)", fill=PAPER, stroke=WRITE, size=20)
    s.text(790, 307, "reused", size=18, weight=700, fill=WRITE, anchor="middle")

    # Overlap-Write inset.
    s.group_start("overlap-write-variant")
    s.rect(846, 91, 420, 239, fill=PAPER, stroke=QUERY, sw=1.8, rx=12, dash="7 5")
    s.text(
        862,
        119,
        "Overlap-Write context variant",
        size=23,
        weight=700,
        fill=QUERY,
    )
    s.text(
        862,
        143,
        "prepend document-left context; cache only the target chunk",
        size=18.5,
        weight=500,
        fill=MUTED,
    )
    s.rect(865, 160, 176, 49, fill=PAPER, stroke=QUERY, sw=1.8, rx=8)
    s.rect(865, 160, 61, 49, fill=QUERY_LIGHT, stroke=QUERY, sw=1.2, rx=8)
    s.text(895, 190, "left w", size=18, weight=650, fill=QUERY, anchor="middle")
    s.text(983, 190, "chunk x_t", size=20, weight=650, fill=NAVY, anchor="middle")
    s.arrow(1047, 184, 1074, 184, color=QUERY)
    s.rect(1081, 160, 161, 49, fill=WRITE_LIGHT, stroke=WRITE, sw=1.8, rx=8)
    s.multiline(
        1161.5,
        181,
        ["lower [0 : j)", "over w + c tokens"],
        size=17.5,
        weight=650,
        fill=NAVY,
        line_height=1.08,
    )
    s.arrow(1161, 211, 1161, 232, color=WRITE)
    s.rect(1042, 235, 200, 47, fill=PANEL, stroke=LINE, sw=1.5, rx=8)
    s.rect(1042, 235, 67, 47, fill=GRAY_LIGHT, stroke=LINE, sw=1.1, rx=8)
    s.line(1051, 243, 1100, 273, stroke=QUERY, sw=2.0)
    s.line(1100, 243, 1051, 273, stroke=QUERY, sw=2.0)
    s.text(1075, 302, "discard prefix", size=17.5, weight=650, fill=QUERY, anchor="middle")
    s.rect(1109, 235, 133, 47, fill=SELECT_LIGHT, stroke=SELECT, sw=1.5, rx=8)
    s.text(1176, 265, "cache h_j(x_t)", size=18, weight=700, fill=SELECT, anchor="middle")
    s.text(
        1056,
        319,
        "Longer one-time Write; identical store bytes and Read pack.",
        size=17.5,
        weight=600,
        fill=MUTED,
        anchor="middle",
    )
    s.group_end()

    # Split-depth inset.
    s.group_start("split-depth-inset")
    s.rect(1281, 91, 294, 239, fill=PAPER, stroke=NAVY, sw=1.8, rx=12)
    s.text(1428, 119, "Split j controls reuse", size=23, weight=700, fill=NAVY, anchor="middle")
    s.text(1301, 146, "L", size=19, weight=700, fill=MUTED, anchor="middle")
    s.rect(1320, 132, 230, 63, fill=SELECT_LIGHT, stroke=SELECT, sw=1.8, rx=8)
    s.multiline(
        1435,
        157,
        ["upper layers [j : L)", "run per query"],
        size=19,
        weight=650,
        fill=NAVY,
        line_height=1.08,
    )
    s.line(1301, 195, 1558, 195, stroke=NAVY, sw=2.3)
    s.chip(1288, 182, 32, 27, "j", fill=NAVY, stroke=NAVY, text_fill=PAPER, size=18, weight=700, rx=13)
    s.rect(1320, 203, 230, 58, fill=WRITE_LIGHT, stroke=WRITE, sw=1.8, rx=8)
    s.multiline(
        1435,
        226,
        ["lower layers [0 : j)", "prepaid once"],
        size=19,
        weight=650,
        fill=NAVY,
        line_height=1.08,
    )
    s.text(1301, 263, "0", size=19, weight=700, fill=MUTED, anchor="middle")
    s.text(1298, 290, "j = 0", size=18, weight=700, fill=QUERY)
    s.text(1354, 290, "raw IDs → replay [0 : L)", size=18, weight=500, fill=INK)
    s.text(1298, 316, "j > 0", size=18, weight=700, fill=WRITE)
    s.text(1354, 316, "cached h_j → resume [j : L)", size=18, weight=500, fill=INK)
    s.group_end()
    s.group_end()

    # ----------------------------------------------------------------- SELECT
    s.group_start("stage-select")
    draw_stage_label(s, number="2", y=391, name="SELECT", sub="per query", color=SELECT)

    s.chip(184, 394, 132, 62, "query q", fill=QUERY_LIGHT, stroke=QUERY, text_fill=QUERY, size=24, weight=700)
    s.arrow(322, 425, 359, 425, color=QUERY)

    s.rect(367, 381, 270, 91, fill=SELECT_LIGHT, stroke=SELECT, sw=2.1, rx=11)
    s.multiline(
        502,
        410,
        ["query-conditioned selector", "BM25 flagship; selector-agnostic"],
        size=21,
        weight=650,
        fill=NAVY,
        line_height=1.18,
    )
    s.arrow(644, 425, 680, 425, color=SELECT)

    s.rect(688, 375, 515, 104, fill=PAPER, stroke=LINE, sw=1.8, rx=11)
    s.text(704, 400, "rank keys across N stored chunks", size=20, weight=650, fill=MUTED)
    store_chip_xs = [707, 773, 839, 905, 971, 1037, 1103]
    for idx, x in enumerate(store_chip_xs, start=1):
        chosen = idx in (1, 4, 7)
        s.chip(
            x,
            416,
            55,
            43,
            f"s{idx}",
            fill=SELECT_LIGHT if chosen else GRAY_LIGHT,
            stroke=SELECT if chosen else LINE,
            text_fill=SELECT if chosen else MUTED,
            size=20,
            weight=700 if chosen else 500,
        )
    s.arrow(1210, 425, 1243, 425, color=SELECT)
    s.text(1227, 405, "top-k", size=18, weight=700, fill=SELECT, anchor="middle")

    s.rect(1252, 375, 314, 104, fill=SELECT_LIGHT, stroke=SELECT, sw=2.1, rx=11)
    s.text(1409, 401, "bounded result (k fixed)", size=21, weight=700, fill=NAVY, anchor="middle")
    s.chip(1274, 416, 67, 42, "s₁", fill=PAPER, stroke=SELECT, text_fill=SELECT, size=20, weight=700)
    s.chip(1351, 416, 67, 42, "s₂", fill=PAPER, stroke=SELECT, text_fill=SELECT, size=20, weight=700)
    s.text(1440, 447, "⋯", size=27, weight=700, fill=MUTED, anchor="middle")
    s.chip(1471, 416, 72, 42, "s_k", fill=PAPER, stroke=SELECT, text_fill=SELECT, size=20, weight=700)
    s.text(
        1030,
        505,
        "Read length = sink + k · c + query  —  independent of stored-context length N",
        size=22,
        weight=700,
        fill=SELECT,
        anchor="middle",
    )
    s.group_end()

    # -------------------------------------------------------------------- READ
    s.group_start("stage-read")
    draw_stage_label(s, number="3", y=567, name="READ", sub="per query", color=GOLD)
    s.text(
        182,
        559,
        "Same selected chunk IDs and causal pack; only the split endpoint changes",
        size=22,
        weight=700,
        fill=NAVY,
    )

    # Selected-set handoff from Select.
    s.path(
        "M1409 480 L1409 543 L803 543",
        stroke=SELECT,
        sw=2.0,
        dash="6 5",
        marker="arrow-select",
    )
    s.text(1120, 536, "selected IDs in document order", size=18.5, weight=650, fill=SELECT, anchor="middle")

    # Lane labels.
    s.chip(
        183,
        596,
        156,
        48,
        "j = 0 control",
        fill=QUERY_LIGHT,
        stroke=QUERY,
        text_fill=QUERY,
        size=21,
        weight=700,
    )
    s.text(183, 670, "raw-text replay", size=20, weight=650, fill=MUTED)

    s.chip(
        183,
        713,
        156,
        48,
        "j > 0 CoMem",
        fill=WRITE_LIGHT,
        stroke=WRITE,
        text_fill=WRITE,
        size=21,
        weight=700,
    )
    s.text(183, 787, "cached continuation", size=20, weight=650, fill=MUTED)

    # Raw-text j=0 lane.
    s.rect(360, 582, 445, 78, fill=PAPER, stroke=QUERY, sw=1.9, rx=10)
    s.text(582.5, 610, "packed raw token IDs / text", size=20, weight=700, fill=QUERY, anchor="middle")
    s.text(
        582.5,
        642,
        "[ BOS  |  x(s₁)  |  x(s₂)  |  ···  |  x(s_k)  |  q ]",
        size=21,
        weight=600,
        fill=INK,
        anchor="middle",
    )
    s.arrow(813, 621, 847, 621, color=QUERY)
    s.rect(855, 582, 385, 78, fill=GRAY_LIGHT, stroke=GRAY, sw=2, rx=10)
    s.multiline(
        1047.5,
        611,
        ["full decoder [0 : L)", "recompute all layers every query"],
        size=21,
        weight=700,
        fill=NAVY,
        line_height=1.17,
    )

    # Cached h_j continuation lane.
    s.rect(360, 698, 445, 89, fill=WRITE_LIGHT, stroke=WRITE, sw=2.1, rx=10)
    s.text(582.5, 726, "packed residual states", size=20, weight=700, fill=WRITE, anchor="middle")
    s.text(
        582.5,
        756,
        "[ h_j(BOS) | h_j(s₁) | ··· | h_j(s_k) | h_j(q) ]",
        size=20.5,
        weight=650,
        fill=INK,
        anchor="middle",
    )
    s.text(
        582.5,
        779,
        "query q traverses lower [0 : j) once",
        size=17.5,
        weight=600,
        fill=MUTED,
        anchor="middle",
    )

    # Ghost lower band plus active upper continuation.
    s.rect(855, 707, 135, 70, fill=PAPER, stroke=WRITE, sw=1.8, rx=9, dash="7 5", opacity=0.72)
    s.multiline(
        922.5,
        733,
        ["lower [0 : j)", "already prepaid"],
        size=17.5,
        weight=650,
        fill=WRITE,
        line_height=1.1,
    )
    s.path(
        "M813 742 C850 742, 982 692, 1012 708",
        stroke=SELECT,
        sw=2.4,
        marker="arrow-select",
    )
    s.text(922, 688, "direct continuation at h_j", size=18, weight=700, fill=SELECT, anchor="middle")
    s.rect(1017, 698, 223, 89, fill=SELECT_LIGHT, stroke=SELECT, sw=2.2, rx=10)
    s.multiline(
        1128.5,
        728,
        ["upper layers [j : L)", "full causal cross-pack attention"],
        size=19.5,
        weight=700,
        fill=NAVY,
        line_height=1.18,
    )

    # Shared output.
    s.rect(1324, 626, 232, 101, fill=GOLD_LIGHT, stroke=GOLD, sw=2.1, rx=12)
    s.multiline(
        1440,
        665,
        ["logits / answer", "same output interface"],
        size=23,
        weight=700,
        fill=NAVY,
        line_height=1.18,
    )
    s.path("M1247 621 L1285 621 L1285 651 L1316 651", stroke=GRAY, sw=2.2, marker="arrow-gray")
    s.path("M1247 742 L1285 742 L1285 701 L1316 701", stroke=SELECT, sw=2.4, marker="arrow-select")

    s.rect(360, 800, 880, 25, fill=SELECT_LIGHT, stroke=SELECT, sw=1.4, rx=12)
    s.text(
        800,
        819,
        "CoMem skips repeated lower-layer document compute; only the bounded top-k pack enters Read.",
        size=18.5,
        weight=700,
        fill=SELECT,
        anchor="middle",
    )
    s.group_end()

    # Keep the footer clear; color semantics are named directly in the panels.
    return s.render()


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    svg_path = out_dir / "fig1_comem_autofigure.svg"
    pdf_path = out_dir / "fig1_comem_autofigure.pdf"
    png_path = out_dir / "fig1_comem_autofigure.png"

    svg = build_svg()
    ET.fromstring(svg)
    if "<image" in svg:
        raise RuntimeError("Final SVG unexpectedly embeds a raster image")

    svg_path.write_text(svg, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(pdf_path))
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(png_path),
        output_width=WIDTH * 2,
        output_height=HEIGHT * 2,
    )

    with Image.open(png_path) as image:
        if image.size != (WIDTH * 2, HEIGHT * 2):
            raise RuntimeError(f"Unexpected PNG size: {image.size}")

    print(f"Wrote editable SVG: {svg_path}")
    print(f"Wrote vector PDF:  {pdf_path}")
    print(f"Wrote PNG preview: {png_path} ({WIDTH * 2}x{HEIGHT * 2})")


if __name__ == "__main__":
    main()
