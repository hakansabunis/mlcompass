"""Render Capstone_Report.md as an IEEE conference-style two-column DOCX.

Why this exists. The course brief asks for a 2–6 page academic report
formatted with a standard conference template (ACL or IEEE). pandoc's
default reference.docx produces a single-column report look that is
nothing like a conference paper. We therefore generate the DOCX
manually with python-docx so we can:

* set a US-letter two-column body
* use Times New Roman 10pt body / 9pt references
* render Roman-numeral section headings centred and uppercase
* render A. B. C. sub-section headings flush-left and italic
* keep the title block (title + authors + affiliations) in a single
  one-column section across the top, then switch to two columns

The script is idempotent: re-run it after editing Capstone_Report.md.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).parent
SRC = ROOT / "Capstone_Report.md"
DST = ROOT / "Capstone_Report.docx"


# --------------------------------------------------------------------------- #
# Low-level XML helpers                                                       #
# --------------------------------------------------------------------------- #


def set_two_columns(section, num_cols: int = 2, sep: bool = False) -> None:
    """Configure a section for `num_cols` body columns (IEEE-style)."""
    sectPr = section._sectPr
    # Remove any existing cols element.
    for cols in sectPr.findall(qn("w:cols")):
        sectPr.remove(cols)
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(num_cols))
    cols.set(qn("w:space"), "360")  # 0.25" between columns
    if sep:
        cols.set(qn("w:sep"), "1")
    sectPr.append(cols)


def set_cell_shading(cell, hex_color: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


# --------------------------------------------------------------------------- #
# Section formatting                                                          #
# --------------------------------------------------------------------------- #


def configure_styles(doc) -> None:
    """Body = Times New Roman 10pt, single-spaced, justified.

    IEEE conference papers don't use the Heading styles in their
    canonical look (headings are centred Roman numerals). We still wire
    the styles so manual heading paragraphs can inherit something sane.
    """
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.first_line_indent = Inches(0.2)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def add_title_block(doc, title: str, authors: list[str], affiliations: list[str], emails: list[str]) -> None:
    """One-column title block at the top of the paper."""
    # Title
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(title)
    run.font.name = "Times New Roman"
    run.font.size = Pt(20)
    run.font.bold = True

    # Authors
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run(", ".join(authors))
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)

    # Affiliations
    for aff in affiliations:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Inches(0)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(aff)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)

    # Emails (italic)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(", ".join(emails))
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    run.font.italic = True


def add_section_heading(doc, text: str, level: int = 1) -> None:
    """IEEE-style section/subsection heading.

    Level 1 → centred, small caps, Roman numeral prefix already in `text`
              (e.g., "I. Introduction").
    Level 2 → flush-left, italic, letter prefix already in `text`
              (e.g., "A. System Architecture").
    Level 3 → flush-left, italic, numbered prefix already in `text`
              (e.g., "1) User-facing surfaces").
    """
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0)
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text.upper())
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run.font.bold = True
    elif level == 2:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run.font.italic = True
        run.font.bold = True
    else:  # level 3+
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(text)
        run.font.name = "Times New Roman"
        run.font.size = Pt(10)
        run.font.italic = True


def add_abstract(doc, abstract_text: str) -> None:
    """Bold 'Abstract—' lead-in followed by the abstract body."""
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    head = p.add_run("Abstract—")
    head.font.name = "Times New Roman"
    head.font.size = Pt(10)
    head.font.bold = True
    head.font.italic = True
    add_inline_runs_to_paragraph(p, abstract_text, base_size=10)


def add_keywords(doc, keywords: str) -> None:
    """IEEE-style 'Keywords:' line, italic."""
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(0)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    head = p.add_run("Index Terms—")
    head.font.name = "Times New Roman"
    head.font.size = Pt(10)
    head.font.bold = True
    head.font.italic = True
    body = p.add_run(keywords)
    body.font.name = "Times New Roman"
    body.font.size = Pt(10)
    body.font.italic = True


def add_body_paragraph(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(0.2)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    add_inline_runs_to_paragraph(p, text, base_size=10)


def add_reference(doc, text: str) -> None:
    """IEEE-style bibliography entry: hanging indent, 9pt."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(-0.2)
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.space_after = Pt(2)
    add_inline_runs_to_paragraph(p, text, base_size=9)


def _set_table_borders_black(table) -> None:
    """Apply IEEE-style minimal borders: single black lines top, bottom,
    header-row underline, no left/right borders, no vertical inside lines.

    This mimics the booktabs / three-line-table look standard in conference
    papers (top rule, midrule under header row, bottom rule).
    """
    tbl = table._tbl
    tblPr = tbl.find(qn("w:tblPr"))
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr")
        tbl.insert(0, tblPr)

    # Drop any existing borders from the table style.
    for borders in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(borders)

    tblBorders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "8")  # 1 pt
        b.set(qn("w:color"), "000000")
        tblBorders.append(b)
    for edge in ("left", "right", "insideV"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "nil")
        tblBorders.append(b)
    # Inside-horizontal: only the rule between header and body (handled below).
    b = OxmlElement("w:insideH")
    b.set(qn("w:val"), "nil")
    tblBorders.append(b)
    tblPr.append(tblBorders)


def _set_cell_bottom_border(cell, color: str = "000000", sz: str = "6") -> None:
    """Add a bottom border to a single cell — used for the midrule under
    the header row."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn("w:tcBorders"))
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)
    # Drop existing bottom border if any.
    for b in tcBorders.findall(qn("w:bottom")):
        tcBorders.remove(b)
    b = OxmlElement("w:bottom")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), sz)
    b.set(qn("w:color"), color)
    tcBorders.append(b)


def add_table_ieee(doc, caption: str, rows: list[list[str]]) -> None:
    """IEEE-style booktabs three-line table with a small-caps caption above.

    No coloured shading — IEEE conference papers are printed grayscale. The
    table has one rule above the header, one rule beneath the header, and
    one rule beneath the last row. No vertical lines, no row shading.
    """
    if not rows:
        return
    # Caption above the table (small-caps style, centred).
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(caption.upper())
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    run.font.bold = False

    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = 1  # centred

    # Use the "Table Grid" stock style so python-docx doesn't paint coloured
    # shading from a fancier theme style, then we replace its borders below.
    try:
        table.style = "Table Grid"
    except KeyError:
        pass

    for r, row in enumerate(rows):
        for c, cell_text in enumerate(row):
            cell = table.rows[r].cells[c]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER if c > 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.space_after = Pt(0)
            add_inline_runs_to_paragraph(p, cell_text.strip(), base_size=9)
            for run in p.runs:
                if r == 0:
                    run.font.bold = True
                run.font.size = Pt(9)

    # Three-line-table border treatment: top rule, midrule under header,
    # bottom rule. No verticals, no row shading.
    _set_table_borders_black(table)
    # Midrule beneath the header row (single border on every header cell's bottom).
    for c in range(len(rows[0])):
        _set_cell_bottom_border(table.rows[0].cells[c], color="000000", sz="6")

    # Small space after the table.
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(3)
    sp.paragraph_format.first_line_indent = Inches(0)


def add_figure_placeholder(doc, label: str = "Fig. 1") -> None:
    """A centred, dashed-border placeholder where the user inserts the
    architecture diagram in Word.

    Renders as a single-cell table so Word's border-painting machinery
    handles the dashed border. The user finds it, deletes the table, and
    Insert → Picture in its place.
    """
    table = doc.add_table(rows=1, cols=1)
    table.alignment = 1  # centred
    cell = table.rows[0].cells[0]

    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(f"[ INSERT {label.upper()} HERE ]")
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)

    p2 = cell.add_paragraph()
    p2.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.first_line_indent = Inches(0)
    p2.paragraph_format.space_before = Pt(0)
    p2.paragraph_format.space_after = Pt(12)
    run = p2.add_run(
        "Delete this box. Insert → Picture → architecture_diagram.png."
    )
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x6B, 0x72, 0x80)

    # Dashed border around the placeholder cell.
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "dashed")
        b.set(qn("w:sz"), "12")
        b.set(qn("w:color"), "6B7280")
        tcBorders.append(b)
    tcPr.append(tcBorders)

    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(3)
    sp.paragraph_format.first_line_indent = Inches(0)


# --------------------------------------------------------------------------- #
# Inline run rendering (handles **bold**, *italic*, `code`, [link](url))     #
# --------------------------------------------------------------------------- #


INLINE_CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
MATH_RE = re.compile(r"\$([^$]+)\$")


def add_inline_runs_to_paragraph(paragraph, text: str, base_size: int = 10) -> None:
    """Render Markdown inline formatting (bold, italic, code, $math$, links)."""
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text)

    # Build the token list.
    tokens: list[tuple[int, int, str, str]] = []
    for m in INLINE_CODE_RE.finditer(text):
        tokens.append((m.start(), m.end(), "code", m.group(1)))
    for m in BOLD_RE.finditer(text):
        tokens.append((m.start(), m.end(), "bold", m.group(1)))
    for m in ITALIC_RE.finditer(text):
        if any(c[2] == "code" and c[0] <= m.start() < c[1] for c in tokens):
            continue
        tokens.append((m.start(), m.end(), "italic", m.group(1)))
    for m in LINK_RE.finditer(text):
        tokens.append((m.start(), m.end(), "link", m.group(1)))
    for m in MATH_RE.finditer(text):
        # Render LaTeX math inline as italic text — Word users will still
        # recognize ``R^2`` etc. and we don't want to depend on Word's
        # math equation editor.
        tokens.append((m.start(), m.end(), "math", m.group(1)))
    tokens.sort()

    # Drop overlapping tokens, keeping the earliest.
    deduped: list[tuple[int, int, str, str]] = []
    last_end = -1
    for tok in tokens:
        if tok[0] >= last_end:
            deduped.append(tok)
            last_end = tok[1]

    pos = 0
    for start, end, kind, payload in deduped:
        if start > pos:
            run = paragraph.add_run(text[pos:start])
            run.font.name = "Times New Roman"
            run.font.size = Pt(base_size)
        run = paragraph.add_run(payload)
        run.font.name = "Times New Roman"
        run.font.size = Pt(base_size)
        if kind == "code":
            run.font.name = "Consolas"
            run.font.size = Pt(base_size - 1)
        elif kind == "bold":
            run.font.bold = True
        elif kind == "italic":
            run.font.italic = True
        elif kind == "link":
            run.font.color.rgb = RGBColor(0x00, 0x4F, 0xA8)
        elif kind == "math":
            run.font.italic = True
        pos = end
    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        run.font.name = "Times New Roman"
        run.font.size = Pt(base_size)


# --------------------------------------------------------------------------- #
# Markdown parsing                                                            #
# --------------------------------------------------------------------------- #


def parse_table_block(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        line = lines[i].strip()
        cells_raw = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells_raw):
            i += 1
            continue
        rows.append(cells_raw)
        i += 1
    return rows, i


# --------------------------------------------------------------------------- #
# Top-level render                                                            #
# --------------------------------------------------------------------------- #


def render_body(doc, md: str) -> None:
    """Walk the Markdown and emit body paragraphs, headings, tables.

    Markdown conventions for this report:
        ``# Title``                              → handled by add_title_block
        ``## Abstract``                          → handled by add_abstract
        ``## I. Introduction``                   → Level-1 section heading
        ``### A. Subsection``                    → Level-2 section heading
        ``**TABLE I. CAPTION**`` followed by ``|`` table → IEEE-style table
        ``[1]`` opener at start of paragraph     → bibliography entry
    """
    lines = md.split("\n")
    i = 0
    pending_table_caption: str | None = None
    in_references = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Horizontal rule = section divider, skip.
        if re.fullmatch(r"-{3,}|_{3,}|\*{3,}", stripped):
            i += 1
            continue

        # Heading.
        m = re.match(r"^(#+)\s+(.*)$", stripped) if stripped.startswith("#") else None
        if m is not None:
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            heading_text_lower = heading_text.lower()

            if level == 1:
                # The first H1 is the paper title — already rendered.
                i += 1
                continue

            if heading_text_lower == "abstract":
                # Read the paragraph that follows as the abstract.
                i += 1
                # Skip blank lines.
                while i < len(lines) and not lines[i].strip():
                    i += 1
                buf: list[str] = []
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("#"):
                    buf.append(lines[i].strip())
                    i += 1
                add_abstract(doc, " ".join(buf))
                continue

            if heading_text_lower == "references":
                in_references = True
                add_section_heading(doc, "References", level=1)
                i += 1
                continue

            # Normal section / subsection heading.
            add_section_heading(doc, heading_text, level=level - 1)
            i += 1
            continue

        # Table caption: "**TABLE I. ABC**"
        cap_match = re.match(r"^\*\*(TABLE [^*]+)\*\*$", stripped)
        if cap_match:
            pending_table_caption = cap_match.group(1)
            i += 1
            continue

        # Table.
        if stripped.startswith("|"):
            rows, i = parse_table_block(lines, i)
            if rows:
                add_table_ieee(doc, pending_table_caption or "TABLE", rows)
                pending_table_caption = None
            continue

        # Code block — skip in body, render inline code via backticks.
        if stripped.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
            i += 1
            continue

        # Bullet list — IEEE rarely uses bullets but we keep them
        # when the markdown explicitly authored them.
        if re.match(r"^[-*]\s+", stripped):
            text = re.sub(r"^[-*]\s+", "", stripped)
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_inline_runs_to_paragraph(p, text, base_size=10)
            i += 1
            continue

        # Numbered list — same treatment.
        if re.match(r"^\d+\.\s+", stripped):
            text = re.sub(r"^\d+\.\s+", "", stripped)
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_inline_runs_to_paragraph(p, text, base_size=10)
            i += 1
            continue

        # Figure placeholder: "[INSERT FIGURE N HERE]" or
        # "[INSERT FIGURE N HERE]" on its own line → render the dashed-border box.
        fig_match = re.match(r"^\[INSERT\s+(FIGURE\s+\d+)\s+HERE\]$", stripped, re.IGNORECASE)
        if fig_match:
            add_figure_placeholder(doc, label="Fig. 1")
            i += 1
            continue

        # Default: collect a paragraph (wrapped lines joined).
        buf = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not (
            lines[i].lstrip().startswith(("#", "-", "*", ">", "|", "```"))
            or re.match(r"^\d+\.\s+", lines[i].strip())
            or re.match(r"^\*\*TABLE", lines[i].strip())
        ):
            buf.append(lines[i].strip())
            i += 1
        paragraph_text = " ".join(buf)

        if in_references and re.match(r"^\[\d+\]", paragraph_text):
            add_reference(doc, paragraph_text)
        else:
            add_body_paragraph(doc, paragraph_text)


def extract_title_block(md: str) -> tuple[str, list[str], list[str], list[str], str]:
    """Pull the title, authors, affiliations, emails, and remainder from the
    top of the Markdown source. The title block in our markdown lives in the
    first H1 paragraph block:

        # <title>

        **Author A, Author B**
        Affiliation A, Affiliation B
        *email1, email2*

        ## Abstract
        ...

    Returns (title, authors, affiliations, emails, rest_of_markdown_starting_at_abstract).
    """
    lines = md.split("\n")
    title = ""
    authors: list[str] = []
    affiliations: list[str] = []
    emails: list[str] = []

    i = 0
    # Find the first H1.
    while i < len(lines) and not lines[i].lstrip().startswith("# "):
        i += 1
    if i < len(lines):
        title = lines[i].lstrip("# ").strip()
        i += 1
    # Skip blank lines.
    while i < len(lines) and not lines[i].strip():
        i += 1
    # Authors line: bold.
    if i < len(lines) and lines[i].strip().startswith("**"):
        names = lines[i].strip().strip("*").strip()
        authors = [n.strip() for n in names.split(",")]
        i += 1
    # Affiliations: any non-blank, non-italic, non-H2 lines.
    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("##"):
        l = lines[i].strip()
        if l.startswith("*") and l.endswith("*"):
            emails.append(l.strip("*").strip())
        else:
            affiliations.append(l)
        i += 1

    rest_lines = lines[i:]
    return title, authors, affiliations, emails, "\n".join(rest_lines)


def main() -> None:
    md = SRC.read_text(encoding="utf-8")

    doc = Document()

    # US Letter, IEEE-style margins.
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.75)
    sec.bottom_margin = Inches(1.0)
    sec.left_margin = Inches(0.75)
    sec.right_margin = Inches(0.75)

    configure_styles(doc)

    # 1) Parse out the title block.
    title, authors, affiliations, emails, rest_md = extract_title_block(md)

    # 2) Render the title block in a one-column section across the top.
    set_two_columns(doc.sections[0], num_cols=1)
    add_title_block(doc, title, authors, affiliations, emails)

    # 3) Add a continuous section break, then switch to two columns.
    new_section = doc.add_section(WD_SECTION.CONTINUOUS)
    new_section.top_margin = sec.top_margin
    new_section.bottom_margin = sec.bottom_margin
    new_section.left_margin = sec.left_margin
    new_section.right_margin = sec.right_margin
    set_two_columns(new_section, num_cols=2, sep=False)

    # 4) Render the body (abstract, sections, references) into the two-column area.
    render_body(doc, rest_md)

    # 5) Save, falling back to *.new.docx if Word has the canonical name open.
    try:
        doc.save(DST)
        print(f"OK: {DST}")
    except PermissionError:
        fallback = DST.with_name(DST.stem + ".new" + DST.suffix)
        doc.save(fallback)
        print(
            f"WARN: {DST} is locked. Wrote {fallback} instead — "
            "close Word and rename."
        )


if __name__ == "__main__":
    main()
