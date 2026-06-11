"""Render INISTA_Paper.md as an IEEE conference-style two-column DOCX.

Reuses the proven low-level helpers from build_capstone_docx.py (two-column
section XML, IEEE booktabs tables, inline Markdown runs) and adds what the
INISTA paper needs on top:

* three-author title block with footnote symbols (\\*, †) and a separate
  affiliation + e-mail line
* an "Index Terms:" line rendered IEEE-style
* blockquote propositions (`> **Proposition 1 ...**`) as indented blocks
* the REAL architecture figure embedded (architecture.png) with its caption,
  instead of an insert-here placeholder

Idempotent: re-run after editing INISTA_Paper.md.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

import build_capstone_docx as base

ROOT = Path(__file__).parent
SRC = ROOT / "INISTA_Paper.md"
DST = ROOT / "INISTA_Paper.docx"
FIG = ROOT / "contract_boundary.png"


def _unescape(s: str) -> str:
    return s.replace("\\*", "*")


def extract_title_block(md: str) -> tuple[str, str, str, str, str]:
    """Pull (title, authors_line, affiliation_line, emails_line, rest_md).

    INISTA_Paper.md layout::

        # <title>

        **Hakan Sabuniş\\*, Yusuf Ünlü\\*, Selim Akyokuş†**
        *\\*Dept. ..., †Dept. ..., Istanbul Medipol University, ...*
        *email1, email2, email3*

        ## Abstract
    """
    lines = md.split("\n")
    i = 0
    while i < len(lines) and not lines[i].lstrip().startswith("# "):
        i += 1
    title = lines[i].lstrip("# ").strip() if i < len(lines) else ""
    i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1

    authors = ""
    if i < len(lines) and lines[i].strip().startswith("**"):
        authors = _unescape(lines[i].strip().strip("*").strip())
        i += 1

    italics: list[str] = []
    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("##"):
        ln = lines[i].strip()
        if ln.startswith("*") and ln.endswith("*"):
            italics.append(_unescape(ln.strip("*").strip()))
        i += 1
    affiliation = italics[0] if italics else ""
    emails = italics[-1] if len(italics) > 1 else ""

    return title, authors, affiliation, emails, "\n".join(lines[i:])


def add_proposition(doc, text: str) -> None:
    """Indented proposition/definition block (the `> ...` quotes)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.left_indent = Inches(0.12)
    p.paragraph_format.right_indent = Inches(0.05)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    base.add_inline_runs_to_paragraph(p, text, base_size=10)


def add_figure(doc, caption: str) -> None:
    """Embed the architecture figure at column width with a 9pt caption."""
    if FIG.exists():
        doc.add_picture(str(FIG), width=Inches(3.3))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[-1].paragraph_format.space_before = Pt(6)
    else:
        base.add_figure_placeholder(doc, label="Fig. 1")
    if caption:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Inches(0)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(6)
        base.add_inline_runs_to_paragraph(p, caption, base_size=9)
        for run in p.runs:
            run.font.size = Pt(9)


def render_body(doc, md: str) -> None:
    lines = md.split("\n")
    i = 0
    pending_table_caption: str | None = None
    pending_figure = False
    in_references = False

    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if re.fullmatch(r"-{3,}|_{3,}|\*{3,}", stripped):
            i += 1
            continue

        # Headings.
        if stripped.startswith("#"):
            m = re.match(r"^(#+)\s+(.*)$", stripped)
            level = len(m.group(1))
            heading = m.group(2).strip()
            low = heading.lower()
            if level == 1:
                i += 1
                continue
            if low == "abstract":
                i += 1
                while i < len(lines) and not lines[i].strip():
                    i += 1
                buf: list[str] = []
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("#"):
                    if lines[i].strip().startswith("**Index Terms"):
                        break
                    buf.append(lines[i].strip())
                    i += 1
                base.add_abstract(doc, " ".join(buf))
                continue
            if low == "references":
                in_references = True
                base.add_section_heading(doc, "References", level=1)
                i += 1
                continue
            base.add_section_heading(doc, heading, level=level - 1)
            i += 1
            continue

        # Index Terms line.
        m = re.match(r"^\*\*Index Terms:?\*\*:?\s*(.*)$", stripped)
        if m:
            base.add_keywords(doc, m.group(1).strip())
            i += 1
            continue

        # Table caption.
        cap = re.match(r"^\*\*(TABLE [^*]+)\*\*$", stripped)
        if cap:
            pending_table_caption = cap.group(1)
            i += 1
            continue

        # Table.
        if stripped.startswith("|"):
            rows, i = base.parse_table_block(lines, i)
            if rows:
                base.add_table_ieee(doc, pending_table_caption or "TABLE", rows)
                pending_table_caption = None
            continue

        # Figure marker (bold-wrapped in this paper) + following caption line.
        if re.match(r"^\*{0,2}\[INSERT\s+FIGURE\s+\d+\s+HERE\]\*{0,2}$", stripped, re.IGNORECASE):
            pending_figure = True
            i += 1
            continue
        if pending_figure and stripped.startswith("*Fig."):
            add_figure(doc, stripped.strip("*").strip())
            pending_figure = False
            i += 1
            continue
        if pending_figure:
            add_figure(doc, "")
            pending_figure = False
            # fall through to render this line normally

        # Blockquote propositions: join consecutive `> ` lines.
        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            add_proposition(doc, " ".join(b for b in buf if b))
            continue

        # Numbered list items (contributions, limitations) — single-line each.
        # Rendered as plain paragraphs with a literal "N." prefix: Word's
        # List Number style continues numbering ACROSS lists, which made the
        # limitations list start at 4 after a 3-item contributions list.
        mnum = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if mnum:
            p = doc.add_paragraph()
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.first_line_indent = Inches(-0.15)
            p.paragraph_format.space_after = Pt(2)
            numrun = p.add_run(f"{mnum.group(1)}. ")
            numrun.font.name = "Times New Roman"
            numrun.font.size = Pt(10)
            base.add_inline_runs_to_paragraph(p, mnum.group(2), base_size=10)
            i += 1
            continue

        if re.match(r"^[-*]\s+", stripped):
            text = re.sub(r"^[-*]\s+", "", stripped)
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            base.add_inline_runs_to_paragraph(p, text, base_size=10)
            i += 1
            continue

        # Default paragraph (join wrapped lines).
        buf = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not (
            lines[i].lstrip().startswith(("#", "-", "*", ">", "|", "```"))
            or re.match(r"^\d+\.\s+", lines[i].strip())
        ):
            buf.append(lines[i].strip())
            i += 1
        text = " ".join(buf)
        if in_references and re.match(r"^\[\d+\]", text):
            base.add_reference(doc, text)
        else:
            base.add_body_paragraph(doc, text)


def main() -> None:
    md = SRC.read_text(encoding="utf-8")
    doc = Document()

    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.75)
    sec.bottom_margin = Inches(1.0)
    sec.left_margin = Inches(0.75)
    sec.right_margin = Inches(0.75)

    base.configure_styles(doc)

    title, authors, affiliation, emails, rest = extract_title_block(md)
    base.set_two_columns(doc.sections[0], num_cols=1)
    base.add_title_block(
        doc,
        title,
        [authors] if authors else [],
        [affiliation] if affiliation else [],
        [emails] if emails else [],
    )

    new_section = doc.add_section(WD_SECTION.CONTINUOUS)
    new_section.top_margin = sec.top_margin
    new_section.bottom_margin = sec.bottom_margin
    new_section.left_margin = sec.left_margin
    new_section.right_margin = sec.right_margin
    base.set_two_columns(new_section, num_cols=2, sep=False)

    render_body(doc, rest)

    try:
        doc.save(DST)
        print(f"OK: {DST}")
    except PermissionError:
        fallback = DST.with_name(DST.stem + ".new" + DST.suffix)
        doc.save(fallback)
        print(f"WARN: {DST} is locked. Wrote {fallback} instead — close Word and rename.")


if __name__ == "__main__":
    main()
