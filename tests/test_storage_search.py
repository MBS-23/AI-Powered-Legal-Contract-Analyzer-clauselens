"""
Tests for Phase 6: SQLite persistence + TF-IDF similarity search.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.storage import db
from src.search.semantic_search import SemanticSearchIndex


@dataclass
class FakeClause:
    section_number: str
    heading: str
    clause_type: str | None
    confidence: float
    method: str
    text: str


def _sample_clauses():
    return [
        FakeClause("1", "GOVERNING LAW", "GOVERNING_LAW", 0.9, "ml",
                   "This Agreement is governed by the laws of Delaware."),
        FakeClause("2", "LIABILITY", "CAP_ON_LIABILITY", 0.8, "ml",
                   "Total liability is capped at the fees paid in the prior 12 months."),
        FakeClause("3", "TERMINATION", "TERMINATION_FOR_CONVENIENCE", 0.7, "ml",
                   "Either party may terminate upon 30 days written notice."),
    ]


# ---------------- storage ----------------


def test_save_and_get_contract(tmp_path):
    dbfile = tmp_path / "t.db"
    cid = db.save_contract(
        "nda.docx", _sample_clauses(),
        risk_score=40, risk_level="Medium", summary="A test contract.",
        entities={"parties": ["A", "B"]}, full_text="full text here",
        db_path=dbfile,
    )
    assert isinstance(cid, int)
    got = db.get_contract(cid, db_path=dbfile)
    assert got["filename"] == "nda.docx"
    assert got["risk_level"] == "Medium"
    assert got["entities"] == {"parties": ["A", "B"]}
    assert len(got["clauses"]) == 3


def test_list_and_count(tmp_path):
    dbfile = tmp_path / "t.db"
    assert db.count_contracts(db_path=dbfile) == 0
    db.save_contract("a.docx", _sample_clauses(), db_path=dbfile)
    db.save_contract("b.docx", _sample_clauses(), db_path=dbfile)
    assert db.count_contracts(db_path=dbfile) == 2
    listing = db.list_contracts(db_path=dbfile)
    assert len(listing) == 2
    # newest first
    assert listing[0]["filename"] == "b.docx"


def test_delete_cascades_clauses(tmp_path):
    dbfile = tmp_path / "t.db"
    cid = db.save_contract("a.docx", _sample_clauses(), db_path=dbfile)
    assert len(db.get_all_clauses(db_path=dbfile)) == 3
    db.delete_contract(cid, db_path=dbfile)
    assert db.count_contracts(db_path=dbfile) == 0
    assert db.get_all_clauses(db_path=dbfile) == []


def test_get_all_clauses_includes_filename(tmp_path):
    dbfile = tmp_path / "t.db"
    db.save_contract("mycontract.docx", _sample_clauses(), db_path=dbfile)
    rows = db.get_all_clauses(db_path=dbfile)
    assert all(r["filename"] == "mycontract.docx" for r in rows)


# ---------------- search ----------------


def test_search_finds_relevant_clause(tmp_path):
    dbfile = tmp_path / "t.db"
    db.save_contract("a.docx", _sample_clauses(), db_path=dbfile)
    index = SemanticSearchIndex.from_clauses(db.get_all_clauses(db_path=dbfile))
    assert not index.is_empty
    results = index.search("limitation of liability cap", top_k=3)
    assert len(results) >= 1
    # The liability clause should rank first.
    assert results[0].clause_type == "CAP_ON_LIABILITY"
    assert results[0].score > 0


def test_search_ranked_descending(tmp_path):
    dbfile = tmp_path / "t.db"
    db.save_contract("a.docx", _sample_clauses(), db_path=dbfile)
    index = SemanticSearchIndex.from_clauses(db.get_all_clauses(db_path=dbfile))
    results = index.search("termination notice", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_index_returns_no_results():
    index = SemanticSearchIndex.from_clauses([])
    assert index.is_empty
    assert index.search("anything") == []


def test_unrelated_query_filtered_out(tmp_path):
    dbfile = tmp_path / "t.db"
    db.save_contract("a.docx", _sample_clauses(), db_path=dbfile)
    index = SemanticSearchIndex.from_clauses(db.get_all_clauses(db_path=dbfile))
    # A query with no lexical overlap should return nothing above min_score.
    results = index.search("photosynthesis chlorophyll biology", top_k=3)
    assert results == []
