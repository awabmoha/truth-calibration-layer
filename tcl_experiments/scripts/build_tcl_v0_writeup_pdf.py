"""Build the standalone TCL-v0 research writeup PDF from Markdown."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MD = ROOT / "TCL-v0-research-writeup.md"
OUTPUT_PDF = ROOT / "TCL-v0-research-writeup.pdf"

BLUE = colors.HexColor("#2E74B5")
DARK_BLUE = colors.HexColor("#1F4D78")
INK = colors.HexColor("#0B2545")
MUTED = colors.HexColor("#667085")
TABLE_FILL = colors.HexColor("#F2F4F7")
BORDER = colors.HexColor("#A6B4C2")


def inline_markup(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    return text


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "TCLTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "body": ParagraphStyle(
            "TCLBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=13.2,
            textColor=INK,
            spaceAfter=7,
        ),
        "h1": ParagraphStyle(
            "TCLHeading1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=BLUE,
            spaceBefore=14,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "TCLHeading2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "TCLBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=13,
            textColor=INK,
            leftIndent=0.34 * inch,
            firstLineIndent=-0.18 * inch,
            spaceAfter=5,
        ),
        "code": ParagraphStyle(
            "TCLCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=9,
            leading=11,
            textColor=INK,
            leftIndent=0.12 * inch,
            rightIndent=0.12 * inch,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "TCLTable",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.6,
            leading=9,
            textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "TCLTableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.6,
            leading=9,
            textColor=INK,
        ),
    }
    return styles


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) >= 2 and all(set(cell) <= {"-", ":"} for cell in rows[1]):
        rows = [rows[0], *rows[2:]]
    return rows


def table_widths(col_count: int) -> list[float]:
    if col_count == 7:
        return [1.25 * inch, 0.82 * inch, 0.82 * inch, 0.72 * inch, 0.72 * inch, 0.96 * inch, 0.96 * inch]
    if col_count == 5:
        return [1.55 * inch, 1.32 * inch, 1.32 * inch, 0.98 * inch, 0.98 * inch]
    return [(6.5 * inch) / col_count] * col_count


def add_table(story: list, rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> None:
    if not rows:
        return
    col_count = max(len(row) for row in rows)
    normalized = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        normalized.append([Paragraph(inline_markup(row[i]) if i < len(row) else "", style) for i in range(col_count)])
    table = Table(normalized, colWidths=table_widths(col_count), repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.45, BORDER),
                ("BACKGROUND", (0, 0), (-1, 0), TABLE_FILL),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 8))


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(inch, 0.55 * inch, "Truth Calibration Layer (TCL) - TCL-v0 Research Note")
    canvas.drawRightString(7.5 * inch, 0.55 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_story() -> list:
    styles = make_styles()
    story: list = []
    lines = SOURCE_MD.read_text(encoding="utf-8").splitlines()
    in_code = False
    code_lines: list[str] = []
    table_lines: list[str] = []

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            add_table(story, parse_table(table_lines), styles)
            table_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_table()
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(line)
            continue

        flush_table()

        if not stripped:
            continue
        if stripped.startswith("# "):
            story.append(Paragraph(inline_markup(stripped[2:]), styles["title"]))
            story.append(Spacer(1, 2))
        elif stripped.startswith("## "):
            story.append(Paragraph(inline_markup(stripped[3:]), styles["h1"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(inline_markup(stripped[4:]), styles["h2"]))
        elif stripped.startswith("- "):
            story.append(Paragraph(f"&bull; {inline_markup(stripped[2:])}", styles["bullet"]))
        else:
            story.append(Paragraph(inline_markup(stripped), styles["body"]))

    flush_table()
    return story


def main() -> None:
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=0.85 * inch,
        title="TCL-v0 Research Writeup",
        author="Awab Mohamed",
    )
    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
