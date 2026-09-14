"""Build the mlcompass class presentation deck.

13 slides covering the four parts the assignment brief asks for:
problem, approach, challenges, empirical findings — plus a live-demo
slide and a Q&A close. Total runtime target: 10-12 minutes, leaving
room for questions inside the 15-minute window.

Slide-by-slide outline:

    1. Title
    2. The Problem
    3. mlcompass in 30 seconds
    4. System architecture (Fig. 1 — INSERT HERE placeholder)
    5. The two-layer brain
    6. The anti-hallucination contract (3 layers)
    7. Worked example — Ames House Prices log_price
    8. Results — Table I (Wilson 95% CI ablation)
    9. Field tests on real Kaggle data
   10. Live demo  (command sequence shown; speaker runs it live)
   11. Challenges we faced
   12. Limitations + future work
   13. Q&A + links

Output:  paper/Capstone_Presentation.pptx

Run with: python paper/build_presentation.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# --------------------------------------------------------------------------- #
# Visual theme                                                                #
# --------------------------------------------------------------------------- #

# Slide dimensions (16:9 widescreen, the default for modern projectors).
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# Palette — quiet, academic, two-accent.
COLOR_INK = RGBColor(0x0F, 0x17, 0x2A)        # near-black for text
COLOR_MUTED = RGBColor(0x6B, 0x72, 0x80)      # captions, footnotes
COLOR_ACCENT = RGBColor(0x1E, 0x40, 0xAF)     # deep blue for emphasis
COLOR_ACCENT2 = RGBColor(0x7C, 0x3A, 0xED)    # purple for the LLM layer
COLOR_PANEL = RGBColor(0xF8, 0xFA, 0xFC)      # very light gray panels
COLOR_BORDER = RGBColor(0xCB, 0xD5, 0xE1)     # panel border

ROOT = Path(__file__).parent.parent
DST = ROOT / "paper" / "Capstone_Presentation.pptx"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _add_text(
    slide,
    text: str,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    font_size: int = 20,
    bold: bool = False,
    italic: bool = False,
    color: RGBColor = COLOR_INK,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    font_name: str = "Calibri",
    anchor: MSO_ANCHOR = MSO_ANCHOR.TOP,
):
    """Add a text box. Inches for positioning."""
    box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_top = Pt(2)
    tf.margin_bottom = Pt(2)
    tf.margin_left = Pt(4)
    tf.margin_right = Pt(4)

    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    f = run.font
    f.name = font_name
    f.size = Pt(font_size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    return box


def _add_bullets(
    slide,
    bullets: list[tuple[str, str | None]],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    font_size: int = 20,
    sub_font_size: int = 16,
    color: RGBColor = COLOR_INK,
    bullet_char: str = "•",
):
    """Add a vertical bullet list. Each item is (headline, optional sub-line)."""
    box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_top = Pt(6)
    tf.margin_bottom = Pt(6)
    tf.margin_left = Pt(8)
    tf.margin_right = Pt(8)

    for idx, (head, sub) in enumerate(bullets):
        if idx == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_before = Pt(8)
        p.space_after = Pt(2)
        r = p.add_run()
        r.text = f"{bullet_char}   {head}"
        r.font.size = Pt(font_size)
        r.font.name = "Calibri"
        r.font.color.rgb = color
        r.font.bold = False

        if sub:
            sp = tf.add_paragraph()
            sp.alignment = PP_ALIGN.LEFT
            sp.space_before = Pt(2)
            sp.space_after = Pt(2)
            sp.level = 1
            sr = sp.add_run()
            sr.text = sub
            sr.font.size = Pt(sub_font_size)
            sr.font.name = "Calibri"
            sr.font.italic = True
            sr.font.color.rgb = COLOR_MUTED

    return box


def _add_header_band(slide, slide_number: int, total: int, header: str) -> None:
    """Top-of-slide title band + slide counter."""
    _add_text(
        slide,
        header,
        left=0.5, top=0.3, width=12.0, height=0.7,
        font_size=32, bold=True, color=COLOR_INK,
    )
    # Thin accent line under the header.
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.5), Inches(1.0), Inches(12.3), Emu(20000),
    )
    line.fill.solid()
    line.fill.fore_color.rgb = COLOR_ACCENT
    line.line.fill.background()

    # Slide counter in upper-right.
    _add_text(
        slide,
        f"{slide_number} / {total}",
        left=12.0, top=0.35, width=1.0, height=0.4,
        font_size=11, color=COLOR_MUTED, align=PP_ALIGN.RIGHT,
    )


def _add_footer(slide, footer: str = "mlcompass — Capstone Project — Sabuniş & Ünlü") -> None:
    """Small footer with project name + names."""
    _add_text(
        slide,
        footer,
        left=0.5, top=7.0, width=12.3, height=0.3,
        font_size=10, color=COLOR_MUTED, italic=True,
    )


def _add_image_placeholder(
    slide,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    label: str,
    instruction: str,
) -> None:
    """Dashed-border rectangle with a 'paste image here' instruction."""
    rect = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height),
    )
    rect.fill.solid()
    rect.fill.fore_color.rgb = COLOR_PANEL
    rect.line.color.rgb = COLOR_BORDER
    rect.line.width = Pt(1.5)
    rect.line.dash_style = 7  # MSO_LINE_DASH_STYLE.DASH

    # Big centered "INSERT HERE" label.
    _add_text(
        slide,
        label,
        left=left, top=top + height / 2 - 0.5, width=width, height=0.7,
        font_size=26, bold=True, italic=True,
        color=COLOR_MUTED, align=PP_ALIGN.CENTER,
    )
    # Subtitle instruction.
    _add_text(
        slide,
        instruction,
        left=left, top=top + height / 2 + 0.2, width=width, height=0.5,
        font_size=14, italic=True,
        color=COLOR_MUTED, align=PP_ALIGN.CENTER,
    )


def _add_panel(
    slide,
    *,
    left: float, top: float, width: float, height: float,
    fill: RGBColor = COLOR_PANEL,
    border: RGBColor = COLOR_BORDER,
):
    """Rounded-rectangle panel used for grouping content blocks."""
    panel = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height),
    )
    panel.fill.solid()
    panel.fill.fore_color.rgb = fill
    panel.line.color.rgb = border
    panel.line.width = Pt(0.75)
    # Adjust the corner radius (default is too round).
    panel.adjustments[0] = 0.05
    return panel


# --------------------------------------------------------------------------- #
# Slide builders                                                              #
# --------------------------------------------------------------------------- #


N_SLIDES = 13


def _new_blank(prs):
    """Add a blank-layout slide (we draw everything ourselves)."""
    blank_layout = prs.slide_layouts[6]
    return prs.slides.add_slide(blank_layout)


def slide_01_title(prs) -> None:
    s = _new_blank(prs)

    # Vertical accent bar on the left.
    bar = s.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.5), SLIDE_H
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_ACCENT
    bar.line.fill.background()

    _add_text(
        s, "mlcompass",
        left=1.0, top=1.5, width=11.5, height=1.1,
        font_size=56, bold=True, color=COLOR_INK,
    )
    _add_text(
        s,
        "A Schema-Bounded LLM Narrator for Machine-Learning Pipeline Diagnosis",
        left=1.0, top=2.7, width=11.5, height=1.5,
        font_size=24, italic=True, color=COLOR_ACCENT,
    )

    _add_text(
        s, "Hakan Sabuniş   ·   Yusuf Ünlü",
        left=1.0, top=4.5, width=11.5, height=0.5,
        font_size=22, color=COLOR_INK,
    )
    _add_text(
        s, "Istanbul Medipol University   ·   School of Engineering and Natural Sciences",
        left=1.0, top=5.05, width=11.5, height=0.4,
        font_size=14, italic=True, color=COLOR_MUTED,
    )

    # Bottom block: links.
    _add_text(
        s, "github.com/hakansabunis/mlcompass     ·     pypi.org/project/mlcompass     ·     v0.8.1     ·     MIT",
        left=1.0, top=6.5, width=11.5, height=0.4,
        font_size=13, color=COLOR_MUTED,
    )


def slide_02_problem(prs) -> None:
    s = _new_blank(prs)
    _add_header_band(s, 2, N_SLIDES, "The problem")
    _add_footer(s)

    _add_text(
        s, "ML engineering is a chain of single-purpose tools — and none of them advise.",
        left=0.7, top=1.4, width=12.0, height=0.7,
        font_size=22, italic=True, color=COLOR_ACCENT,
    )

    _add_bullets(
        s,
        [
            ("pandas-profiling shows you the data.",
             "It doesn't know which column you're trying to predict."),
            ("Weights & Biases tracks your runs.",
             "It doesn't tell you the loss curve looks like overfitting."),
            ("Copilot and Cursor complete your code.",
             "They'll happily generate Adam(momentum=0.9) — which is invalid."),
            ("Existing LLM assistants hallucinate.",
             "They invent column names that don't exist and propose fixes for problems that aren't real."),
        ],
        left=0.7, top=2.4, width=12.0, height=4.0,
        font_size=20, sub_font_size=15,
    )

    _add_text(
        s,
        "146,932 hallucinated citations were observed across arXiv, bioRxiv and SSRN in 2025 alone [Zhao et al., 2026].",
        left=0.7, top=6.4, width=12.0, height=0.4,
        font_size=13, italic=True, color=COLOR_MUTED,
    )


def slide_03_pitch(prs) -> None:
    s = _new_blank(prs)
    _add_header_band(s, 3, N_SLIDES, "mlcompass in 30 seconds")
    _add_footer(s)

    _add_text(
        s,
        "An open-source CLI that follows you through the whole ML pipeline — and is allowed to explain but not to invent.",
        left=0.7, top=1.4, width=12.0, height=1.0,
        font_size=22, italic=True, color=COLOR_ACCENT,
    )

    _add_panel(s, left=0.7, top=2.7, width=12.0, height=3.6)

    # The eleven commands in a flowing line.
    _add_text(
        s, "Eleven commands cover every pipeline stage:",
        left=1.0, top=2.9, width=11.5, height=0.5,
        font_size=18, bold=True, color=COLOR_INK,
    )
    _add_text(
        s,
        "init  →  advise  →  audit  →  watch  →  compare  →  evaluate  →  deploy",
        left=1.0, top=3.5, width=11.5, height=0.6,
        font_size=20, font_name="Consolas", color=COLOR_ACCENT,
    )
    _add_text(
        s,
        "+   status   ·   monitor   ·   optimize   ·   agent",
        left=1.0, top=4.1, width=11.5, height=0.5,
        font_size=18, font_name="Consolas", color=COLOR_MUTED,
    )

    _add_text(
        s, "Four surfaces, one tool layer:",
        left=1.0, top=4.9, width=11.5, height=0.5,
        font_size=18, bold=True, color=COLOR_INK,
    )
    _add_text(
        s,
        "CLI   ·   MCP server (Claude Desktop / Cursor / Claude Code)   ·   Self-driving agent   ·   Slash commands",
        left=1.0, top=5.5, width=11.5, height=0.6,
        font_size=15, italic=True, color=COLOR_INK,
    )

    _add_text(
        s,
        "Shared on-disk project context (.mlcompass/) so we remember what was decided yesterday.",
        left=0.7, top=6.5, width=12.0, height=0.4,
        font_size=13, italic=True, color=COLOR_MUTED,
    )


def slide_04_architecture(prs) -> None:
    s = _new_blank(prs)
    _add_header_band(s, 4, N_SLIDES, "System architecture")
    _add_footer(s)

    _add_text(
        s,
        "Three user-facing surfaces sit on top of a two-layer brain over a persistent project-context folder.",
        left=0.7, top=1.3, width=12.0, height=0.6,
        font_size=18, italic=True, color=COLOR_MUTED,
    )

    # Big architecture-image placeholder.
    _add_image_placeholder(
        s,
        left=2.0, top=2.0, width=9.3, height=4.5,
        label="[ INSERT ARCHITECTURE DIAGRAM HERE ]",
        instruction=(
            "Paste paper/diagram.png  —  same image used in the paper (Fig. 1)."
        ),
    )

    _add_text(
        s,
        "Fig. 1.  mlcompass system architecture (also Fig. 1 in the paper).",
        left=2.0, top=6.6, width=9.3, height=0.3,
        font_size=12, italic=True, color=COLOR_MUTED, align=PP_ALIGN.CENTER,
    )


def slide_05_two_layer_brain(prs) -> None:
    s = _new_blank(prs)
    _add_header_band(s, 5, N_SLIDES, "The two-layer brain")
    _add_footer(s)

    _add_text(
        s,
        "Facts come from pure code.  Narration comes from a Claude agent on top, never the other way around.",
        left=0.7, top=1.3, width=12.0, height=0.6,
        font_size=18, italic=True, color=COLOR_ACCENT,
    )

    # Left panel — Deterministic layer.
    _add_panel(
        s, left=0.7, top=2.1, width=5.8, height=4.6,
        fill=RGBColor(0xEF, 0xF6, 0xFF), border=RGBColor(0x93, 0xC5, 0xFD),
    )
    _add_text(
        s, "Deterministic Tool Layer  (pure Python)",
        left=0.9, top=2.3, width=5.5, height=0.5,
        font_size=18, bold=True, color=COLOR_ACCENT,
    )
    _add_bullets(
        s,
        [
            ("Reads CSVs with pandas", None),
            ("Parses training scripts with ast", None),
            ("Computes Pearson + Spearman + perfect-match", None),
            ("Returns structured dict — never free text", None),
            ("Cannot hallucinate by construction", None),
        ],
        left=0.9, top=2.9, width=5.5, height=3.5,
        font_size=16, sub_font_size=13,
    )
    _add_text(
        s, "src/mlcompass/tools/",
        left=0.9, top=6.2, width=5.5, height=0.4,
        font_size=12, font_name="Consolas", italic=True, color=COLOR_MUTED,
    )

    # Right panel — LLM narrator.
    _add_panel(
        s, left=6.8, top=2.1, width=5.8, height=4.6,
        fill=RGBColor(0xF5, 0xF3, 0xFF), border=RGBColor(0xC4, 0xB5, 0xFD),
    )
    _add_text(
        s, "Optional Claude Narrator  (--llm)",
        left=7.0, top=2.3, width=5.5, height=0.5,
        font_size=18, bold=True, color=COLOR_ACCENT2,
    )
    _add_bullets(
        s,
        [
            ("Receives the dict from below", None),
            ("Explains it in natural language", None),
            ("Runs under a 3-layer contract (next slide)", None),
            ("Bound at call time to the evidence's column set", None),
            ("Skipped entirely with --no-llm  (offline path)", None),
        ],
        left=7.0, top=2.9, width=5.5, height=3.5,
        font_size=16, sub_font_size=13,
    )
    _add_text(
        s, "src/mlcompass/agents/",
        left=7.0, top=6.2, width=5.5, height=0.4,
        font_size=12, font_name="Consolas", italic=True, color=COLOR_MUTED,
    )


def slide_06_contract(prs) -> None:
    s = _new_blank(prs)
    _add_header_band(s, 6, N_SLIDES, "The anti-hallucination contract")
    _add_footer(s)

    _add_text(
        s,
        "Three layers.  Each one alone is weak.  Together they are strict.",
        left=0.7, top=1.3, width=12.0, height=0.6,
        font_size=20, italic=True, color=COLOR_ACCENT,
    )

    # Three vertical panels.
    panel_w, panel_top, panel_h = 4.0, 2.1, 4.6
    gaps = [0.7, 4.85, 9.0]

    titles = [
        ("Layer 1\nDeterministic evidence", COLOR_ACCENT),
        ("Layer 2\nStrict prompt", COLOR_ACCENT2),
        ("Layer 3\nRuntime schema boundary", RGBColor(0xC0, 0x36, 0x6E)),
    ]
    bodies = [
        [
            ("Pure Python.  No LLM.", None),
            ("Computes max(|ρ_P|, |ρ_S|).", None),
            ("Emits a dict E of candidate leak columns.", None),
        ],
        [
            ("Cite only items in E.", None),
            ("Say cannot_determine when unsure.", None),
            ("No code patches — manual checks only.", None),
        ],
        [
            ("Tool-input enum is generated from E.", None),
            ("SDK rejects any out-of-evidence column.", None),
            ("Worst-case guarantee, not just guidance.", None),
        ],
    ]

    for i, ((title, color), bullets) in enumerate(zip(titles, bodies)):
        _add_panel(s, left=gaps[i], top=panel_top, width=panel_w, height=panel_h)
        _add_text(
            s, title,
            left=gaps[i] + 0.1, top=panel_top + 0.2, width=panel_w - 0.2, height=1.0,
            font_size=18, bold=True, color=color, align=PP_ALIGN.CENTER,
        )
        _add_bullets(
            s, bullets,
            left=gaps[i] + 0.2, top=panel_top + 1.4, width=panel_w - 0.4, height=panel_h - 1.8,
            font_size=14, sub_font_size=12,
        )

    _add_text(
        s,
        "Layer 3 is the load-bearing piece — prompts are advisory, schemas are enforcement.",
        left=0.7, top=6.85, width=12.0, height=0.4,
        font_size=14, italic=True, color=COLOR_MUTED, align=PP_ALIGN.CENTER,
    )


def slide_07_worked_example(prs) -> None:
    s = _new_blank(prs)
    _add_header_band(s, 7, N_SLIDES, "Worked example — Ames House Prices")
    _add_footer(s)

    _add_text(
        s,
        "A user accidentally includes log_price when training a model to predict saleprice.  R² = 1.000.",
        left=0.7, top=1.3, width=12.0, height=0.6,
        font_size=17, italic=True, color=COLOR_ACCENT,
    )

    # Numbered cards.
    _add_panel(s, left=0.7, top=2.1, width=12.0, height=4.7)

    rows = [
        ("Layer 1 — Evidence",
         "Pearson ρ_P(log_price, y) ≈ 0.94    ·    Spearman ρ_S = 1.00    ·    perfect-match 98.7 %",
         "→ log_price flagged as candidate (max correlation = 1.00, threshold 0.97)."),
        ("Layer 2 — Narrator (Claude)",
         '"log_price has Spearman = 1.00 with the target — almost certainly a transformed leak.',
         'Recommend manually checking the feature pipeline for any column derived from the target."'),
        ("Layer 3 — Schema boundary",
         "columns_referenced  enum = {log_price, near_proxy, feature_3, …, feature_12}",
         "log_price ∈ enum → response ships.   If the LLM had emitted 'revenue', SDK would reject and retry."),
    ]
    y = 2.3
    for title, line1, line2 in rows:
        _add_text(
            s, title,
            left=0.9, top=y, width=11.6, height=0.4,
            font_size=16, bold=True, color=COLOR_ACCENT,
        )
        _add_text(
            s, line1,
            left=1.1, top=y + 0.4, width=11.4, height=0.4,
            font_size=13, font_name="Consolas", color=COLOR_INK,
        )
        _add_text(
            s, line2,
            left=1.1, top=y + 0.8, width=11.4, height=0.4,
            font_size=13, italic=True, color=COLOR_MUTED,
        )
        y += 1.4


def slide_08_results(prs) -> None:
    s = _new_blank(prs)
    _add_header_band(s, 8, N_SLIDES, "Results — phantom-column fabrication rate")
    _add_footer(s)

    _add_text(
        s,
        "Synthetic regression test set.   N = 200 narrator responses per configuration.   Claude 3.5 Sonnet.",
        left=0.7, top=1.3, width=12.0, height=0.5,
        font_size=15, italic=True, color=COLOR_MUTED,
    )

    # Table — drawn as a real PowerPoint table.
    rows = 4
    cols = 3
    table_left = Inches(1.5)
    table_top = Inches(2.1)
    table_width = Inches(10.3)
    table_height = Inches(3.0)
    table_shape = s.shapes.add_table(rows, cols, table_left, table_top, table_width, table_height)
    table = table_shape.table

    # Column widths.
    table.columns[0].width = Inches(5.3)
    table.columns[1].width = Inches(2.0)
    table.columns[2].width = Inches(3.0)

    headers = ["Configuration", "Rate", "Wilson 95% CI"]
    body = [
        ["Layer 1   (no prompt, no schema)", "8.0 %", "[ 4.93 ,  12.66 ]"],
        ["Layer 1 + 2   (strict prompt)", "1.0 %", "[ 0.27 ,   3.56 ]"],
        ["Layer 1 + 2 + 3   (schema-enforced)", "0.0 %", "[ 0.00 ,   1.83 ]"],
    ]

    def _set_cell(cell, text: str, *, header: bool = False, accent: bool = False):
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if accent else PP_ALIGN.LEFT
        run = p.add_run()
        run.text = text
        run.font.name = "Calibri"
        run.font.size = Pt(15)
        run.font.bold = header
        run.font.color.rgb = COLOR_ACCENT if accent and not header else COLOR_INK
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PANEL if header else RGBColor(0xFF, 0xFF, 0xFF)

    for c, h in enumerate(headers):
        _set_cell(table.cell(0, c), h, header=True)
    for r, row in enumerate(body, start=1):
        for c, val in enumerate(row):
            accent = (c == 1)
            _set_cell(table.cell(r, c), val, accent=accent)

    # Takeaway.
    _add_panel(
        s, left=0.7, top=5.4, width=12.0, height=1.3,
        fill=RGBColor(0xEE, 0xF2, 0xFF), border=RGBColor(0xC7, 0xD2, 0xFE),
    )
    _add_text(
        s, "Takeaway",
        left=0.9, top=5.5, width=11.7, height=0.4,
        font_size=16, bold=True, color=COLOR_ACCENT,
    )
    _add_text(
        s,
        "Prompt engineering reduces fabrication 8× (8.0 % → 1.0 %), but does not eliminate it.   "
        "The runtime schema boundary brings the observed rate to 0 — a worst-case guarantee, not just an average.",
        left=0.9, top=5.95, width=11.7, height=0.7,
        font_size=13, italic=True, color=COLOR_INK,
    )


def slide_09_field_tests(prs) -> None:
    s = _new_blank(prs)
    _add_header_band(s, 9, N_SLIDES, "Field tests — 5 Kaggle datasets, 11 real bugs")
    _add_footer(s)

    _add_text(
        s,
        "Every release dry-ran on a real Kaggle dataset before shipping.  Each bug locked in by a regression test.",
        left=0.7, top=1.3, width=12.0, height=0.5,
        font_size=15, italic=True, color=COLOR_MUTED,
    )

    # Field-tests table.
    rows = 6
    cols = 4
    table_shape = s.shapes.add_table(
        rows, cols, Inches(1.0), Inches(2.0), Inches(11.3), Inches(3.4)
    )
    table = table_shape.table
    table.columns[0].width = Inches(1.0)
    table.columns[1].width = Inches(4.0)
    table.columns[2].width = Inches(1.5)
    table.columns[3].width = Inches(4.8)

    headers = ["Test", "Dataset", "Patch", "Bugs caught"]
    body = [
        ["1", "Telco Churn", "—", "baseline — pipeline runs end-to-end"],
        ["2", "Ames House Prices", "v0.7.0", "missing target name, year-col, sparse-col crash"],
        ["3", "Titanic", "v0.7.1", "missing 'survived' target name"],
        ["4", "Penguins", "v0.7.2", "multiclass mis-detected, MCP ledger gap"],
        ["5", "Insurance Charges", "v0.7.3", "missing 'charges' target, MCP state fields"],
    ]

    def _cell(c, text: str, *, header: bool = False):
        c.text = ""
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = text
        run.font.name = "Calibri"
        run.font.size = Pt(13)
        run.font.bold = header
        c.fill.solid()
        c.fill.fore_color.rgb = COLOR_PANEL if header else RGBColor(0xFF, 0xFF, 0xFF)

    for c, h in enumerate(headers):
        _cell(table.cell(0, c), h, header=True)
    for r, row in enumerate(body, start=1):
        for c, val in enumerate(row):
            _cell(table.cell(r, c), val)

    # Lesson.
    _add_panel(
        s, left=0.7, top=5.7, width=12.0, height=1.0,
        fill=RGBColor(0xFE, 0xF3, 0xF2), border=RGBColor(0xFC, 0xA5, 0xA5),
    )
    _add_text(
        s,
        "Real datasets find bugs synthetic ones don't — none of the 11 bugs had been anticipated by our unit-test suite.",
        left=0.9, top=5.95, width=11.7, height=0.55,
        font_size=14, italic=True, color=COLOR_INK,
    )


def slide_10_demo(prs) -> None:
    """Slide 10 — Eleven Claude Code slash commands at a glance + a four-
    command live demo block at the bottom.  The grid above the terminal
    is intentionally compact so the audience can read all eleven verbs at
    once while the speaker walks through the demo."""
    s = _new_blank(prs)
    _add_header_band(s, 10, N_SLIDES, "The 11 slash commands & live demo")
    _add_footer(s)

    # Sub-title.
    _add_text(
        s,
        "Eleven Claude Code slash commands — each maps to one mlcompass tool.  "
        "Installed with one command, no JSON config.",
        left=0.7, top=1.25, width=12.0, height=0.5,
        font_size=14, italic=True, color=COLOR_MUTED,
    )

    # Install hint pill (small dark capsule under the subtitle).
    _add_panel(
        s, left=0.7, top=1.80, width=6.6, height=0.5,
        fill=RGBColor(0x11, 0x18, 0x27), border=COLOR_BORDER,
    )
    _add_text(
        s, "$  mlcompass install-slash-commands --scope user",
        left=0.85, top=1.86, width=6.4, height=0.4,
        font_size=12, font_name="Consolas", bold=True,
        color=RGBColor(0xA7, 0xF3, 0xD0),
    )

    # 11 slash commands in a 4x3 grid (3 rows: 4-4-3).
    # Cards highlighted in accent gradient are the ones used in today's demo.
    commands = [
        ("/mlc-init",     "Start a new mlcompass project",                 "highlight"),
        ("/mlc-advise",   "Analyze CSV — target, warnings, models",        "highlight"),
        ("/mlc-audit",    "Static analysis of training script (8 rules)",  "normal"),
        ("/mlc-watch",    "Scan log: NaN / divergence / overfit",          "normal"),
        ("/mlc-compare",  "Side-by-side diff of two training runs",        "normal"),
        ("/mlc-evaluate", "Metrics + auto leakage panel on perfect R²",    "highlight"),
        ("/mlc-leak",     "Focused leakage check on predictions",          "normal"),
        ("/mlc-deploy",   "Production-readiness check (Lambda / Docker)",  "normal"),
        ("/mlc-status",   "Show project history + current state",          "highlight"),
        ("/mlc-monitor",  "PSI / KS / chi² drift (CLI fallback)",          "fallback"),
        ("/mlc-optimize", "HPO recommender (CLI fallback)",                "fallback"),
    ]

    card_w, card_h = 3.05, 0.95
    gap_x, gap_y = 0.12, 0.10
    start_x, start_y = 0.7, 2.55

    for idx, (cmd, desc, kind) in enumerate(commands):
        row, col = divmod(idx, 4)
        x = start_x + col * (card_w + gap_x)
        y = start_y + row * (card_h + gap_y)

        # Pick visual style per category.
        if kind == "highlight":
            fill, border = RGBColor(0xEE, 0xF2, 0xFF), RGBColor(0xC7, 0xD2, 0xFE)
            code_color = COLOR_ACCENT
            desc_color = COLOR_INK
        elif kind == "fallback":
            fill, border = RGBColor(0xF8, 0xFA, 0xFC), COLOR_BORDER
            code_color = COLOR_MUTED
            desc_color = COLOR_MUTED
        else:
            fill, border = RGBColor(0xFF, 0xFF, 0xFF), COLOR_BORDER
            code_color = COLOR_ACCENT
            desc_color = COLOR_INK

        _add_panel(s, left=x, top=y, width=card_w, height=card_h,
                   fill=fill, border=border)
        _add_text(s, cmd,
                  left=x + 0.12, top=y + 0.08, width=card_w - 0.24, height=0.36,
                  font_size=13, font_name="Consolas", bold=True, color=code_color)
        _add_text(s, desc,
                  left=x + 0.12, top=y + 0.46, width=card_w - 0.24, height=0.45,
                  font_size=10, italic=False, color=desc_color)

    # Live demo terminal block (slim, on the right of the bottom area).
    demo_y = start_y + 3 * (card_h + gap_y) + 0.10  # ~5.5
    _add_panel(s, left=0.7, top=demo_y, width=12.0, height=1.45,
               fill=RGBColor(0x11, 0x18, 0x27), border=COLOR_BORDER)

    demo_cmds = [
        ("›  /mlc-init insurance-demo",                              "→ .mlcompass/ project folder created"),
        ("›  /mlc-advise demo-data/insurance.csv",                   "→ target = 'charges', regression"),
        ("›  /mlc-evaluate demo-data/predictions_with_leak.csv",     "→ R² = 1.000 → automatic leakage panel"),
        ("›  /mlc-status",                                           "→ project history summary"),
    ]
    line_y = demo_y + 0.10
    for cmd, hint in demo_cmds:
        _add_text(s, cmd,
                  left=0.95, top=line_y, width=6.5, height=0.28,
                  font_size=11, font_name="Consolas", bold=True,
                  color=RGBColor(0xA7, 0xF3, 0xD0))
        _add_text(s, hint,
                  left=7.5, top=line_y, width=5.2, height=0.28,
                  font_size=10, italic=True, color=RGBColor(0xCB, 0xD5, 0xE1))
        line_y += 0.30

    # Footnote under the panel.
    _add_text(
        s,
        "Highlighted cards = used in today's demo · Dashed-style (gray) = CLI fallback · Backup: pre-recorded MP4 on USB",
        left=0.7, top=7.05, width=12.0, height=0.3,
        font_size=10, italic=True, color=COLOR_MUTED,
    )


def slide_11_challenges(prs) -> None:
    s = _new_blank(prs)
    _add_header_band(s, 11, N_SLIDES, "Challenges we faced")
    _add_footer(s)

    _add_text(
        s, "Four problems we hit, and what we learned.",
        left=0.7, top=1.3, width=12.0, height=0.5,
        font_size=18, italic=True, color=COLOR_ACCENT,
    )

    items = [
        (
            "Prompt engineering wasn't enough.",
            "We spent a week trying to make the LLM stop fabricating columns through prompts alone.  Got from 8 % to 1 %.  Still wrong 1 % of the time.  Fix: enforce at the schema layer, not the prompt layer.",
        ),
        (
            "CLI and MCP server drifted apart.",
            "Same underlying functions, different behavior.  CLI was writing to the ledger, MCP wasn't.  Shipped a shared _persist_to_ledger helper + parity regression tests in v0.7.2 and v0.7.3.",
        ),
        (
            "Synthetic unit tests missed real bugs.",
            "Eleven bugs surfaced from 5 Kaggle datasets — zero predicted from our pre-existing test suite.  Moved to a field-test-then-patch release cadence.",
        ),
        (
            "Reviewer feedback on the paper.",
            "Three peer reviewers identified weak statistical claims, missing baselines, implicit threat model.  Released v0.8.1 with Wilson 95 % CIs, reproducibility scripts, threat model doc, limitations doc, per-bug detail.",
        ),
    ]

    _add_bullets(
        s, items,
        left=0.7, top=2.0, width=12.0, height=4.8,
        font_size=18, sub_font_size=13,
    )


def slide_12_limitations_future(prs) -> None:
    s = _new_blank(prs)
    _add_header_band(s, 12, N_SLIDES, "Limitations  +  future work")
    _add_footer(s)

    # Two-column layout.
    _add_text(
        s, "What we did NOT measure",
        left=0.7, top=1.3, width=5.8, height=0.5,
        font_size=20, bold=True, color=COLOR_ACCENT,
    )
    _add_bullets(
        s,
        [
            ("Cross-model generalization",
             "All experiments use Claude 3.5 Sonnet.  GPT-4 / Llama-3 / Gemini behavior unverified."),
            ("Constrained-generation baseline",
             "We do not directly benchmark against Outlines or OpenAI strict: true mode."),
            ("External user study",
             "Used by the two authors + classmates only.  No third-party evaluation."),
            ("Other hallucination types",
             "Miscalibrated confidence, causal misattribution, prompt-injection compliance — out of scope."),
        ],
        left=0.7, top=1.95, width=5.8, height=4.8,
        font_size=15, sub_font_size=12,
    )

    _add_text(
        s, "Planned for v0.9 / v1.0",
        left=6.85, top=1.3, width=5.8, height=0.5,
        font_size=20, bold=True, color=COLOR_ACCENT2,
    )
    _add_bullets(
        s,
        [
            ("OpenAI backend",
             "Lets us repeat Table I on GPT-4 + strict: true and report cross-provider rates."),
            ("Outlines benchmark",
             "Same task through Outlines on Llama-3-8B.  Side-by-side fabrication rates."),
            ("N = 2000 live run",
             "Tightens Wilson CIs.  Reproducible via scripts/reproduce_hallucination_ablation.py."),
            ("Plug-in system",
             "Domain-specific analysers (HuggingFace, PyTorch Lightning) without modifying core."),
        ],
        left=6.85, top=1.95, width=5.8, height=4.8,
        font_size=15, sub_font_size=12,
    )


def slide_13_close(prs) -> None:
    s = _new_blank(prs)

    # Hero title.
    _add_text(
        s, "Thank you",
        left=1.0, top=1.5, width=11.5, height=1.5,
        font_size=72, bold=True, color=COLOR_ACCENT, align=PP_ALIGN.CENTER,
    )
    _add_text(
        s, "Questions?",
        left=1.0, top=3.2, width=11.5, height=0.8,
        font_size=32, italic=True, color=COLOR_INK, align=PP_ALIGN.CENTER,
    )

    # Links panel.
    _add_panel(s, left=2.5, top=4.5, width=8.3, height=2.0)

    _add_text(
        s, "Project",
        left=2.8, top=4.7, width=2.5, height=0.4,
        font_size=14, bold=True, color=COLOR_MUTED,
    )
    _add_text(
        s, "github.com/hakansabunis/mlcompass",
        left=5.3, top=4.7, width=5.5, height=0.4,
        font_size=14, font_name="Consolas", color=COLOR_ACCENT,
    )

    _add_text(
        s, "Install",
        left=2.8, top=5.2, width=2.5, height=0.4,
        font_size=14, bold=True, color=COLOR_MUTED,
    )
    _add_text(
        s, "pip install mlcompass",
        left=5.3, top=5.2, width=5.5, height=0.4,
        font_size=14, font_name="Consolas", color=COLOR_ACCENT,
    )

    _add_text(
        s, "Paper",
        left=2.8, top=5.7, width=2.5, height=0.4,
        font_size=14, bold=True, color=COLOR_MUTED,
    )
    _add_text(
        s, "paper/Capstone_Report.pdf  in the repo",
        left=5.3, top=5.7, width=5.5, height=0.4,
        font_size=14, font_name="Consolas", color=COLOR_ACCENT,
    )

    _add_text(
        s, "Authors",
        left=2.8, top=6.2, width=2.5, height=0.4,
        font_size=14, bold=True, color=COLOR_MUTED,
    )
    _add_text(
        s, "Hakan Sabuniş   ·   Yusuf Ünlü",
        left=5.3, top=6.2, width=5.5, height=0.4,
        font_size=14, color=COLOR_ACCENT,
    )


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    builders = [
        slide_01_title,
        slide_02_problem,
        slide_03_pitch,
        slide_04_architecture,
        slide_05_two_layer_brain,
        slide_06_contract,
        slide_07_worked_example,
        slide_08_results,
        slide_09_field_tests,
        slide_10_demo,
        slide_11_challenges,
        slide_12_limitations_future,
        slide_13_close,
    ]
    for build in builders:
        build(prs)

    try:
        prs.save(DST)
        print(f"OK: {DST}  ({len(builders)} slides)")
    except PermissionError:
        fallback = DST.with_name(DST.stem + ".new" + DST.suffix)
        prs.save(fallback)
        print(
            f"WARN: {DST} is locked (PowerPoint open?).  Wrote {fallback} instead — "
            "close PowerPoint and rename."
        )


if __name__ == "__main__":
    main()
