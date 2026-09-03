"""
Tests for Phase 3: rule-based entity extraction.

Covers the high-value fields (parties, dates, governing law, money, notice,
term) across the date/phrasing variants contracts actually use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction.entity_extractor import extract_entities, extract_parties

PREAMBLE = (
    'This Agreement is entered into between Acme Corporation ("Supplier") and '
    'Globex LLC ("Customer") as of 15 January 2026.'
)


def test_extract_parties_from_between_clause():
    parties = extract_parties(PREAMBLE)
    assert "Acme Corporation" in parties
    assert "Globex LLC" in parties


def test_effective_date_day_first():
    e = extract_entities(PREAMBLE)
    assert e.effective_date == "15 January 2026"


def test_effective_date_month_first_with_cue():
    text = "This Agreement is effective as of January 1, 2020 and continues thereafter."
    e = extract_entities(text)
    assert e.effective_date == "January 1, 2020"


def test_governing_law():
    text = "This Agreement shall be governed by the laws of the State of California."
    e = extract_entities(text)
    assert e.governing_law == "California"


def test_monetary_amounts():
    text = "The total fee shall be $1,250,000.00 payable in USD 50,000 installments."
    e = extract_entities(text)
    assert any("1,250,000" in m for m in e.monetary_amounts)
    assert any("50,000" in m for m in e.monetary_amounts)


def test_notice_period():
    text = "Either party may terminate upon 60 days' prior written notice."
    e = extract_entities(text)
    assert "60 days" in e.notice_periods


def test_term_extraction():
    text = "The term of this Agreement shall be 3 years from the Effective Date."
    e = extract_entities(text)
    assert e.term == "3 years"


def test_empty_text_returns_empty_entities():
    e = extract_entities("")
    assert e.parties == []
    assert e.effective_date is None
    assert e.governing_law is None


def test_dates_are_deduplicated():
    text = "Signed January 1, 2020. Confirmed January 1, 2020. Renewed March 5, 2021."
    e = extract_entities(text)
    # Two distinct dates, not three (the duplicate Jan 1 is collapsed).
    assert len(e.dates) == 2
