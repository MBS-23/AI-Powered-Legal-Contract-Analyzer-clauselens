"""
Tests for src/core.py — the end-to-end orchestration used by the app.

Runs the whole pipeline (parse -> clean -> clauses -> entities -> risk ->
summaries) on the sample contract and checks the AnalysisResult is coherent
and JSON-exportable.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import analyze_file, analyze_text
from src.report import build_html_report

SAMPLE = Path(__file__).resolve().parent.parent / "sample_contracts" / "sample_services_agreement.docx"


def test_analyze_file_end_to_end():
    result = analyze_file(SAMPLE)
    assert result.filename.endswith(".docx")
    assert len(result.clauses) > 3
    assert result.overview  # non-empty plain-English overview
    assert result.risk.level in {"Low", "Medium", "High"}
    assert 0 <= result.risk.score <= 100
    # Sample contract names two parties in its preamble.
    assert len(result.entities.parties) >= 2


def test_analysis_result_is_json_exportable():
    result = analyze_file(SAMPLE)
    payload = json.dumps(result.to_export_dict())
    reloaded = json.loads(payload)
    assert reloaded["filename"] == result.filename
    assert "risk" in reloaded and "clauses" in reloaded


def test_html_report_builds():
    result = analyze_file(SAMPLE)
    html = build_html_report(result)
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "Risk Assessment" in html
    assert result.entities.parties[0] in html


def test_analyze_text_keyword_only():
    text = (
        "1. CONFIDENTIALITY\nThe parties shall keep information confidential.\n"
        "2. GOVERNING LAW\nGoverned by the laws of the State of New York.\n"
    )
    result = analyze_text(text, filename="mini.txt", use_ml=False)
    types = {c.clause_type for c in result.clauses if c.clause_type}
    assert "CONFIDENTIALITY" in types
    assert result.entities.governing_law == "New York"
