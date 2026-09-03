"""
Tests for Phase 1: parsers, text cleaning, and rule-based clause extraction.

Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.text_cleaner import clean_text, strip_page_markers, normalize_whitespace
from src.extraction.clause_extractor import segment_into_sections, classify_section, extract_clauses
from src.document.docx_parser import extract_text_from_docx

SAMPLE_DOCX = Path(__file__).resolve().parent.parent / "sample_contracts" / "sample_services_agreement.docx"


# ---------- text_cleaner ----------

def test_clean_text_removes_page_footers():
    raw = "Some clause text\nPage 3 of 12\nMore clause text"
    cleaned = clean_text(raw)
    assert "Page 3 of 12" not in cleaned
    assert "Some clause text" in cleaned
    assert "More clause text" in cleaned


def test_clean_text_collapses_internal_whitespace():
    raw = "This   has     extra   spaces"
    cleaned = clean_text(raw)
    assert "  " not in cleaned


def test_strip_page_markers():
    raw = "[PAGE 1]\nHello\n[PAGE 2]\nWorld"
    stripped = strip_page_markers(raw)
    assert "[PAGE" not in stripped
    assert "Hello" in stripped and "World" in stripped


def test_normalize_whitespace_collapses_blank_lines():
    raw = "A\n\n\n\n\nB"
    assert normalize_whitespace(raw) == "A\n\nB"


# ---------- clause_extractor: segmentation ----------

def test_segment_into_sections_detects_numbered_headings():
    text = "1. CONFIDENTIALITY\nSome confidential text.\n2. PAYMENT\nInvoice terms here."
    sections = segment_into_sections(text)
    assert len(sections) == 2
    assert sections[0].number == "1"
    assert sections[0].heading == "CONFIDENTIALITY"
    assert sections[1].heading == "PAYMENT"


def test_segment_into_sections_fallback_when_no_headings():
    text = "Just a blob of contract text with no numbered sections at all."
    sections = segment_into_sections(text)
    assert len(sections) == 1
    assert sections[0].heading == "(untitled document)"


# ---------- clause_extractor: classification ----------

def test_classify_section_matches_confidentiality():
    from src.extraction.clause_extractor import Section
    s = Section(
        number="1",
        heading="CONFIDENTIALITY",
        body="The receiving party shall keep all confidential information strictly confidential.",
        start_line=0,
    )
    clause_type, confidence, matched = classify_section(s)
    assert clause_type == "CONFIDENTIALITY"
    assert confidence > 0.15
    assert "confidential" in matched


def test_classify_section_no_match_returns_none():
    from src.extraction.clause_extractor import Section
    s = Section(number="99", heading="MISCELLANEOUS", body="The sky is blue today.", start_line=0)
    clause_type, confidence, matched = classify_section(s)
    assert clause_type is None
    assert matched == []


def test_extract_clauses_end_to_end_on_sample_text():
    text = (
        "1. LIABILITY\n"
        "Supplier shall be liable for all losses and damages, including consequential damages.\n"
        "2. TERMINATION\n"
        "Either party may terminate this Agreement upon 30 days written notice.\n"
    )
    # Pin to the keyword baseline: this test validates the Phase 1 matcher's
    # taxonomy (LIABILITY/TERMINATION), which is independent of the ML model.
    clauses = extract_clauses(text, use_ml=False)
    types_found = {c.clause_type for c in clauses}
    assert "LIABILITY" in types_found
    assert "TERMINATION" in types_found


# ---------- docx_parser (requires the generated sample file) ----------

def test_extract_text_from_docx_sample_contract():
    assert SAMPLE_DOCX.exists(), "Sample contract not found — regenerate it before running tests."
    parsed = extract_text_from_docx(SAMPLE_DOCX)
    assert parsed.paragraph_count > 5
    assert "CONFIDENTIALITY" in parsed.full_text.upper()


def test_full_pipeline_on_sample_contract():
    """Integration test: parse the sample DOCX, clean it, extract clauses,
    and confirm we detect the clause types the sample was designed to contain."""
    parsed = extract_text_from_docx(SAMPLE_DOCX)
    cleaned = normalize_whitespace(clean_text(parsed.full_text))
    clauses = extract_clauses(cleaned, use_ml=False)  # keyword-baseline taxonomy
    detected_types = {c.clause_type for c in clauses if c.clause_type}

    expected = {"CONFIDENTIALITY", "PAYMENT", "INDEMNIFICATION", "LIABILITY", "TERMINATION", "GOVERNING_LAW"}
    # Require most (not necessarily all — keyword matching is a baseline) expected types to be found.
    overlap = expected & detected_types
    assert len(overlap) >= 4, f"Only matched {overlap}, expected most of {expected}"
