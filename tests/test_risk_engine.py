"""
Tests for Phase 4: the risk analysis engine.

Uses lightweight stand-in clause objects (anything with .clause_type/.heading/
.text works, since the engine is decoupled from the classifier).
"""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.risk_engine import analyze_risk


@dataclass
class FakeClause:
    clause_type: str | None
    heading: str = ""
    text: str = ""


def test_present_high_risk_clause_is_flagged():
    clauses = [FakeClause("UNCAPPED_LIABILITY", "Liability", "Liability shall be unlimited.")]
    report = analyze_risk(clauses, "Liability shall be unlimited.")
    titles = [f.title for f in report.findings]
    assert any("Uncapped Liability" in t for t in titles)
    assert report.counts["high"] >= 1


def test_missing_liability_cap_flagged_when_absent():
    # A contract with only a governing-law clause and no cap on liability.
    clauses = [FakeClause("GOVERNING_LAW")]
    text = "This Agreement is governed by the laws of Delaware."
    report = analyze_risk(clauses, text)
    assert any("Missing clause: Liability cap" in f.title for f in report.findings)


def test_cap_on_liability_present_not_flagged_missing():
    clauses = [FakeClause("CAP_ON_LIABILITY")]
    report = analyze_risk(clauses, "Total liability is capped at fees paid.")
    assert not any("Missing clause: Liability cap" in f.title for f in report.findings)


def test_red_flag_language_detected():
    text = "The Provider may terminate in its sole discretion and provides the software as is."
    report = analyze_risk([FakeClause(None)], text)
    titles = [f.title for f in report.findings]
    assert any("sole discretion" in t.lower() for t in titles)
    assert any("warranty disclaimer" in t.lower() for t in titles)


def test_confidentiality_detected_via_keyword_scan():
    # ML has no CONFIDENTIALITY label, but the keyword scan should still find it
    # in the text, so it must NOT be reported as missing.
    clauses = [FakeClause("GOVERNING_LAW")]
    text = (
        "The receiving party shall keep all confidential information and "
        "proprietary information secret. Governed by the laws of Delaware."
    )
    report = analyze_risk(clauses, text)
    assert not any("Missing clause: Confidentiality" in f.title for f in report.findings)


def test_score_and_level_bounds():
    clauses = [FakeClause(None)]
    report = analyze_risk(clauses, "Nothing risky here at all.")
    assert 0 <= report.score <= 100
    assert report.level in {"Low", "Medium", "High"}


def test_findings_sorted_high_severity_first():
    text = "Provider may act in its sole discretion; liability shall be unlimited; any and all claims."
    clauses = [FakeClause("UNCAPPED_LIABILITY", text=text)]
    report = analyze_risk(clauses, text)
    severities = [f.severity for f in report.findings]
    # No 'high' should appear after a 'low' in the ordered list.
    order = {"high": 0, "medium": 1, "low": 2}
    assert severities == sorted(severities, key=lambda s: order[s])
