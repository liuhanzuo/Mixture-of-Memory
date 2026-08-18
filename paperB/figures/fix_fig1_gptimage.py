#!/usr/bin/env python3
"""Correct the exact scientific claims in the GPT Image 2 Figure 1.

The source image has a strong layout but its outcome boxes overstate the result:
the inherited model retains 19.5% of above-chance MMLU signal at 200k steps,
whereas the fully random-init control is at approximately zero recovery.  This script masks the
generated box contents and redraws exact, reproducible text.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
SRC = HERE / "ChatGPT Image 2026年7月27日 17_12_39.png"
OUT = HERE / "fig1_concept_gptimage_corrected.png"
OUT_PDF = HERE / "fig1_concept_gptimage_corrected.pdf"

FONT_REG = "/usr/share/fonts/dejavu/DejaVuSansCondensed.ttf"
FONT_BOLD = "/usr/share/fonts/dejavu/DejaVuSansCondensed-Bold.ttf"


def draw_lines(draw, x, y, title, body, fill=(17, 20, 24), body_size=21,
               line_step=30):
    title_font = ImageFont.truetype(FONT_BOLD, 24)
    body_font = ImageFont.truetype(FONT_REG, body_size)
    draw.text((x, y), title, font=title_font, fill=fill)
    yy = y + 38
    for line in body:
        draw.text((x, yy), line, font=body_font, fill=fill)
        yy += line_step


im = Image.open(SRC).convert("RGB")
draw = ImageDraw.Draw(im)

# Preserve the generated borders while replacing the icons and misleading text.
green = (234, 245, 234)
rose = (254, 236, 236)
draw.rounded_rectangle((1219, 383, 1484, 531), radius=14, fill=green)
draw.rounded_rectangle((1219, 652, 1484, 799), radius=14, fill=rose)

draw_lines(
    draw,
    1240,
    397,
    "Perplexity recovers",
    ["PPL 10.561 vs. 7.398", "1.428× tax @ 200k"],
)
draw_lines(
    draw,
    1240,
    666,
    "MMLU lags",
    ["Train all: 19.4%", "ShortGPT-16: 63.0%", "Random init: ≈0%"],
    body_size=18,
    line_step=24,
)

# Replace the generated cross-model depth labels with OLMo same-model results.
white = (255, 255, 255)
draw.rectangle((0, 135, 320, 300), fill=white)
draw.rectangle((0, 395, 305, 485), fill=white)
draw.rectangle((0, 750, 295, 850), fill=white)
depth_font = ImageFont.truetype(FONT_REG, 20)
draw.text((55, 180), "Next-token sat95\n~1.000L (OLMo)", font=depth_font,
          fill=(112, 43, 151))
draw.text((55, 415), "OLMo answer-letter readout\n0.562–0.594L", font=depth_font,
          fill=(214, 118, 0))
draw.text((55, 770), "Semantic sat95\n~0.073L (OLMo)", font=depth_font,
          fill=(31, 91, 166))
draw.rectangle((880, 950, 1536, 1024), fill=white)

im.save(OUT, dpi=(300, 300), optimize=True)
im.save(OUT_PDF, "PDF", resolution=300.0)
print(f"wrote {OUT} and {OUT_PDF}")
