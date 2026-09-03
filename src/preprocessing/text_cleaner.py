"""
text_cleaner.py
----------------
Normalizes raw extracted text before section/clause detection.

Kept deliberately conservative in Phase 1: we remove obvious noise
(repeated whitespace, common header/footer junk, page markers used
internally) but we do NOT rewrite or reflow sentences, since Phase 2's
ML classifier will be trained on text that looks like real contract
prose, not a paraphrased version of it.
"""

import re

# Patterns commonly seen as running headers/footers in contract exports.
_NOISE_PATTERNS = [
    r"^\s*Page \d+ of \d+\s*$",
    r"^\s*CONFIDENTIAL\s*$",
    r"^\s*DRAFT\s*-\s*SUBJECT TO CHANGE\s*$",
    r"^\s*-{2,}\s*$",  # stray horizontal-rule lines from PDF extraction
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)

_PAGE_MARKER_RE = re.compile(r"^\[PAGE \d+\]$")


def clean_text(raw_text: str) -> str:
    """Clean a full contract text blob (may span multiple pages)."""
    lines = raw_text.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _NOISE_RE.match(stripped):
            continue
        # Collapse internal repeated whitespace but keep the line itself
        stripped = re.sub(r"[ \t]{2,}", " ", stripped)
        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def strip_page_markers(text: str) -> str:
    """Remove the [PAGE N] markers added by pdf_parser.full_text, if a
    caller wants plain text without them (e.g. for display)."""
    lines = [l for l in text.splitlines() if not _PAGE_MARKER_RE.match(l.strip())]
    return "\n".join(lines)


def normalize_whitespace(text: str) -> str:
    """Collapse 3+ blank lines into a single blank line for readability."""
    return re.sub(r"\n{3,}", "\n\n", text)
