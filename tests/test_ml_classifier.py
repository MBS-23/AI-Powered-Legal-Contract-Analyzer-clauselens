"""
Tests for Phase 2: the trained ML clause classifier and its integration into
extract_clauses().

These tests skip cleanly if the model artifact hasn't been trained yet
(models/clause_classifier.joblib absent), so a fresh checkout still has a
green suite before `python training/train_classifier.py` is run.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extraction import ml_classifier
from src.extraction.clause_extractor import extract_clauses

pytestmark = pytest.mark.skipif(
    not ml_classifier.is_available(),
    reason="Trained model not found — run `python training/train_classifier.py` first.",
)


# Representative clause snippets with an unambiguous CUAD category each.
GOVERNING_LAW = (
    "This Agreement shall be governed by and construed in accordance with the "
    "laws of the State of Delaware, without regard to its conflict of laws principles."
)
INSURANCE = (
    "During the Term, each party shall maintain commercial general liability "
    "insurance with coverage of not less than $2,000,000 per occurrence and "
    "furnish a certificate of insurance upon request."
)


def test_model_loads_and_reports_metrics():
    metrics = ml_classifier.model_metrics()
    assert metrics is not None
    # The trained model should comfortably beat the most-frequent baseline.
    assert metrics["test_accuracy"] > metrics["baseline_accuracy"]
    assert metrics["test_accuracy"] > 0.5


def test_predict_governing_law():
    pred = ml_classifier.predict(GOVERNING_LAW)
    assert pred.clause_type == "GOVERNING_LAW"
    assert 0.0 < pred.confidence <= 1.0


def test_predict_insurance():
    pred = ml_classifier.predict(INSURANCE)
    assert pred.clause_type == "INSURANCE"


def test_predict_empty_text_returns_none():
    pred = ml_classifier.predict("   ")
    assert pred.clause_type is None
    assert pred.confidence == 0.0


def test_extract_clauses_uses_ml_method():
    text = "1. GOVERNING LAW\n" + GOVERNING_LAW + "\n2. INSURANCE\n" + INSURANCE
    clauses = extract_clauses(text, use_ml=True)
    methods = {c.method for c in clauses if c.clause_type}
    # At least one section should have been labelled by the ML model.
    assert "ml" in methods
    ml_types = {c.clause_type for c in clauses if c.method == "ml"}
    assert "GOVERNING_LAW" in ml_types
