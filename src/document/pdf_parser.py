"""
pdf_parser.py
-------------
Extracts text from PDF contracts, page by page, using pdfplumber.

Design notes:
- We keep page boundaries (list of {page_num, text}) rather than one big
  string, because later phases (risk findings, "jump to page") need to
  cite a page number as evidence.
- pdfplumber handles most digitally-generated PDFs well. Scanned/image-only
  PDFs will come back with little or no text — we flag that case rather
  than silently returning an empty document, since it's one of the most
  common real-world failure modes for contract PDFs.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


@dataclass
class PageText:
    page_num: int  # 1-indexed, matches what a human would call "page 8"
    text: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text.strip())


@dataclass
class ParsedDocument:
    source_path: str
    pages: list[PageText]
    likely_scanned: bool  # True if almost no extractable text was found

    @property
    def full_text(self) -> str:
        """Join all pages into one string, with a page-marker between
        pages so downstream code can still recover page numbers if needed."""
        return "\n".join(f"[PAGE {p.page_num}]\n{p.text}" for p in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def extract_text_from_pdf(path: str | Path) -> ParsedDocument:
    """Extract per-page text from a PDF contract.

    Raises FileNotFoundError if the path doesn't exist, and ValueError
    if the file can't be opened as a PDF at all (e.g. corrupted upload).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    pages: list[PageText] = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                pages.append(PageText(page_num=i, text=raw))
    except Exception as e:  # pdfplumber can raise several underlying exceptions
        raise ValueError(f"Could not parse PDF '{path.name}': {e}") from e

    total_chars = sum(p.char_count for p in pages)
    # Heuristic: a normal contract page has hundreds of characters of text.
    # If the average across the doc is very low, it's almost certainly a
    # scanned image PDF with no text layer (would need OCR, out of scope
    # for Phase 1).
    avg_chars_per_page = total_chars / max(len(pages), 1)
    likely_scanned = avg_chars_per_page < 40

    return ParsedDocument(source_path=str(path), pages=pages, likely_scanned=likely_scanned)
