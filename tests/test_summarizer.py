"""
Tests for Phase 5: extractive summarization + fact-based contract overview.
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.summarizer import (
    split_sentences,
    summarize_text,
    summarize_contract,
)


def test_split_sentences_basic():
    text = "This is one sentence. Here is another one! And a third here?"
    sents = split_sentences(text)
    assert len(sents) == 3


def test_summarize_text_returns_short_text_unchanged():
    text = "The parties agree to the terms herein described in full."
    # Only one sentence -> returned as-is.
    assert summarize_text(text, max_sentences=3).strip() == text.strip()


def test_summarize_text_reduces_length():
    long_text = " ".join(
        f"Clause number {i} discusses obligations of the supplier regarding delivery and payment."
        for i in range(10)
    )
    summary = summarize_text(long_text, max_sentences=3)
    assert len(summary) < len(long_text)
    assert len(split_sentences(summary)) <= 3


def test_summarize_text_preserves_original_order():
    text = (
        "Alpha term defines the scope of work. "
        "Beta term defines the payment schedule and invoicing. "
        "Gamma term defines confidentiality obligations of both parties. "
        "Delta term defines the governing law and jurisdiction for disputes."
    )
    summary = summarize_text(text, max_sentences=2)
    # Whatever two sentences are chosen, they must stay in document order.
    idxs = [text.index(s[:15]) for s in split_sentences(summary)]
    assert idxs == sorted(idxs)


# ---- fact-based overview ----


@dataclass
class FakeEntities:
    parties: list = field(default_factory=list)
    effective_date: str | None = None
    governing_law: str | None = None
    term: str | None = None
    monetary_amounts: list = field(default_factory=list)
    notice_periods: list = field(default_factory=list)


@dataclass
class FakeClause:
    clause_type: str | None


@dataclass
class FakeFinding:
    title: str
    severity: str


@dataclass
class FakeRisk:
    level: str
    score: int
    findings: list


def test_contract_overview_includes_parties_and_law():
    ent = FakeEntities(
        parties=["Acme Corp", "Globex LLC"],
        effective_date="January 1, 2026",
        governing_law="Delaware",
        term="2 years",
    )
    clauses = [FakeClause("GOVERNING_LAW"), FakeClause("CAP_ON_LIABILITY")]
    risk = FakeRisk("Medium", 40, [FakeFinding("High-risk clause: X", "high")])
    text = summarize_contract(clauses, ent, risk)
    assert "Acme Corp" in text and "Globex LLC" in text
    assert "Delaware" in text
    assert "January 1, 2026" in text
    assert "2 years" in text
    assert "Medium" in text


def test_contract_overview_handles_empty_inputs():
    ent = FakeEntities()
    text = summarize_contract([], ent, None)
    assert "Not enough" in text
