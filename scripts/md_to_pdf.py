"""
md_to_pdf.py
------------
Render a Markdown document into a branded ClauseLens PDF (navy/gold, serif
headings) using reportlab. Handles the subset of Markdown used in the project
docs: headings, paragraphs, bullet lists, tables, fenced code blocks,
blockquotes, horizontal rules, and inline **bold** / `code`.

Usage:
    python scripts/md_to_pdf.py docs/CASE_STUDY.md docs/ClauseLens_Case_Study.pdf
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, ListFlowable, ListItem, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

NAVY = colors.HexColor("#0B1F35")
TEAL = colors.HexColor("#185A63")
GOLD = colors.HexColor("#C79A3C")
INK = colors.HexColor("#1f2933")
MUTED = colors.HexColor("#5b6b7d")
LIGHT = colors.HexColor("#f2f5f8")


def _styles():
    ss = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=ss["Title"], fontName="Times-Bold",
                                 fontSize=26, textColor=NAVY, spaceAfter=2, leading=30),
        "subtitle": ParagraphStyle("st", parent=ss["Normal"], fontName="Times-Italic",
                                    fontSize=13, textColor=GOLD, spaceAfter=14),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold",
                             fontSize=15, textColor=NAVY, spaceBefore=16, spaceAfter=5),
        "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName="Times-Bold",
                             fontSize=12.5, textColor=TEAL, spaceBefore=10, spaceAfter=3),
        "h4": ParagraphStyle("h4", parent=ss["Heading4"], fontName="Helvetica-Bold",
                             fontSize=10.5, textColor=NAVY, spaceBefore=8, spaceAfter=2),
        "body": ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica",
                               fontSize=9.7, textColor=INK, leading=14.5, spaceAfter=6, alignment=TA_LEFT),
        "bullet": ParagraphStyle("bl", parent=ss["Normal"], fontName="Helvetica",
                                 fontSize=9.7, textColor=INK, leading=14),
        "code": ParagraphStyle("c", parent=ss["Normal"], fontName="Courier",
                               fontSize=8.3, textColor=colors.HexColor("#233"), leading=11,
                               backColor=LIGHT, borderPadding=6, spaceAfter=8, leftIndent=4),
        "quote": ParagraphStyle("q", parent=ss["Normal"], fontName="Helvetica-Oblique",
                                fontSize=10, textColor=colors.HexColor("#2c3e50"), leading=15,
                                leftIndent=12, borderPadding=2, spaceAfter=6),
        "cell": ParagraphStyle("cell", parent=ss["Normal"], fontName="Helvetica",
                               fontSize=8.7, textColor=INK, leading=11.5),
        "cellh": ParagraphStyle("cellh", parent=ss["Normal"], fontName="Helvetica-Bold",
                                fontSize=8.7, textColor=colors.white, leading=11.5),
    }


_TRANSLIT = {
    "—": " - ", "–": "-", "→": "->", "←": "<-", "↓": "|", "↑": "|",
    "│": "|", "├": "|", "└": "|", "┌": "|", "┐": "|", "┘": "|", "─": "-", "•": "*",
    "×": "x", "✓": "[ok]", "≈": "~", "≥": ">=", "≤": "<=", "…": "...",
    "’": "'", "‘": "'", "“": '"', "”": '"', "§": "Sec. ", "⚡": "", "™": "(tm)",
    "🔵": "", "🟢": "", "🟡": "", "⚪": "", "⚖️": "", "⚠️": "!", "⚠": "!",
}


def _normalize(text: str) -> str:
    for k, v in _TRANSLIT.items():
        text = text.replace(k, v)
    # Drop any remaining non-Latin-1 characters the base PDF fonts can't render.
    return "".join(c if ord(c) < 256 else "" for c in text)


def _inline(text: str) -> str:
    text = _normalize(text)
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="8.6">\1</font>', text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    return text


def parse(md: str, S: dict) -> list:
    lines = md.splitlines()
    flow: list = []
    i, n = 0, len(lines)
    bullets: list = []

    def flush_bullets():
        nonlocal bullets
        if bullets:
            flow.append(ListFlowable(
                [ListItem(Paragraph(_inline(b), S["bullet"]), leftIndent=12,
                          value="•", bulletColor=GOLD) for b in bullets],
                bulletType="bullet", start="•"))
            flow.append(Spacer(1, 5))
            bullets = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):  # code block
            flush_bullets()
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(html.escape(_normalize(lines[i])) or "&nbsp;")
                i += 1
            flow.append(Paragraph("<br/>".join(buf) or "&nbsp;", S["code"]))
            i += 1
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:  # table
            flush_bullets()
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[\s:|-]+$", lines[i].strip().strip("|")):
                    rows.append(cells)
                i += 1
            if rows:
                flow.append(_table(rows, S))
                flow.append(Spacer(1, 8))
            continue

        if not stripped:
            flush_bullets()
            i += 1
            continue

        if stripped.startswith("# "):
            flush_bullets(); flow.append(Paragraph(_inline(stripped[2:]), S["title"]))
        elif stripped.startswith("## "):
            flush_bullets()
            flow.append(Paragraph(_inline(stripped[3:]), S["h2"]))
            flow.append(HRFlowable(width="100%", thickness=1.4, color=GOLD, spaceAfter=6))
        elif stripped.startswith("### "):
            flush_bullets(); flow.append(Paragraph(_inline(stripped[4:]), S["h3"]))
        elif stripped.startswith("#### "):
            flush_bullets(); flow.append(Paragraph(_inline(stripped[5:]), S["h4"]))
        elif stripped.startswith("## AI-Powered"):
            pass
        elif stripped.startswith("> "):
            flush_bullets(); flow.append(Paragraph(_inline(stripped[2:]), S["quote"]))
        elif stripped in ("---", "***", "___"):
            flush_bullets(); flow.append(Spacer(1, 4))
        elif stripped.startswith(("- ", "* ")):
            bullets.append(stripped[2:])
        elif re.match(r"^\d+\.\s", stripped):
            flush_bullets(); flow.append(Paragraph(_inline(stripped), S["body"]))
        else:
            # subtitle heuristic: the italic tagline line right after title
            flush_bullets(); flow.append(Paragraph(_inline(stripped), S["body"]))
        i += 1

    flush_bullets()
    return flow


def _table(rows: list, S: dict) -> Table:
    header, *body = rows
    ncols = len(header)
    data = [[Paragraph(_inline(c), S["cellh"]) for c in header]]
    for r in body:
        r = (r + [""] * ncols)[:ncols]
        data.append([Paragraph(_inline(c), S["cell"]) for c in r])
    avail = A4[0] - 40 * mm
    t = Table(data, colWidths=[avail / ncols] * ncols, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d5dde5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _decorations(canvas, doc):
    canvas.saveState()
    w, h = A4
    # top gold rule
    canvas.setStrokeColor(GOLD)
    canvas.setLineWidth(2)
    canvas.line(20 * mm, h - 12 * mm, w - 20 * mm, h - 12 * mm)
    # footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, 10 * mm, "ClauseLens — AI-Powered Legal Contract Analyzer")
    canvas.drawRightString(w - 20 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main(src: str, out: str) -> None:
    md = Path(src).read_text(encoding="utf-8")
    S = _styles()
    doc = SimpleDocTemplate(out, pagesize=A4, topMargin=20 * mm, bottomMargin=18 * mm,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            title="ClauseLens — Case Study", author="Podugu Bala Veera Venkata Sunil")
    doc.build(parse(md, S), onFirstPage=_decorations, onLaterPages=_decorations)
    print(f"wrote {out}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "docs/CASE_STUDY.md"
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/ClauseLens_Case_Study.pdf"
    main(src, out)
