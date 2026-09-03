"""
ml_classifier.py
----------------
Loads the trained scikit-learn clause classifier (models/clause_classifier.joblib,
produced by training/train_classifier.py) and exposes a small prediction API
that drops in behind the same interface the Phase 1 keyword matcher used.

Why a thin wrapper:
  - The model is loaded once and cached (lru_cache), so the Streamlit app pays
    the ~13 MB load cost a single time per process.
  - If the model file is absent (e.g. a fresh checkout before training), every
    function degrades gracefully to "unavailable" so the app can fall back to
    the keyword baseline instead of crashing.
  - We surface the most influential TF-IDF terms per prediction, which gives
    the UI a cheap, honest "why did it say this?" explanation without a heavy
    explainability library.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

# models/ lives at the project root, two levels up from this file.
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "clause_classifier.joblib"


@dataclass
class MLPrediction:
    clause_type: str | None  # None when below the confidence threshold
    confidence: float
    top_terms: list[str]  # most influential tokens for the predicted class


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the model artifact, or return None if it isn't present.

    Returns the dict saved by train_classifier.py:
        {pipeline, labels, threshold, metrics, sklearn_version, ...}
    """
    if not MODEL_PATH.exists():
        return None
    import joblib  # local import so the module imports even without joblib

    try:
        return joblib.load(MODEL_PATH)
    except Exception:  # corrupt/incompatible artifact -> behave as unavailable
        return None


def is_available() -> bool:
    """True if a usable trained model is on disk."""
    return load_model() is not None


def model_metrics() -> dict | None:
    """Return the saved evaluation metrics (accuracy, macro-F1, ...) or None."""
    artifact = load_model()
    return artifact["metrics"] if artifact else None


def _top_terms_for_class(artifact, class_label: str, text: str, k: int = 5) -> list[str]:
    """Return up to k tokens from `text` that most push toward `class_label`.

    Works for the linear models we train (LogisticRegression exposes coef_;
    a CalibratedClassifierCV over LinearSVC does not, so we simply skip term
    attribution there and return an empty list).
    """
    pipeline = artifact["pipeline"]
    try:
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        coefs = getattr(clf, "coef_", None)
        classes = getattr(clf, "classes_", None)
        if coefs is None or classes is None:
            return []
        class_idx = list(classes).index(class_label)
        vec = tfidf.transform([text])
        feature_names = tfidf.get_feature_names_out()
        # Contribution of each present feature = tfidf weight * class coef.
        present = vec.nonzero()[1]
        contrib = [(feature_names[j], vec[0, j] * coefs[class_idx, j]) for j in present]
        contrib.sort(key=lambda t: t[1], reverse=True)
        return [term for term, weight in contrib[:k] if weight > 0]
    except Exception:
        return []


def predict(text: str) -> MLPrediction:
    """Predict the clause type for a single block of text.

    Returns MLPrediction(None, confidence, []) when either the model is
    unavailable or the top probability is below the trained threshold.
    """
    artifact = load_model()
    if artifact is None or not text or not text.strip():
        return MLPrediction(None, 0.0, [])

    pipeline = artifact["pipeline"]
    threshold = float(artifact.get("threshold", 0.15))

    proba = pipeline.predict_proba([text])[0]
    best_idx = int(np.argmax(proba))
    confidence = float(proba[best_idx])
    predicted = pipeline.classes_[best_idx]

    if confidence < threshold:
        return MLPrediction(None, confidence, [])

    top_terms = _top_terms_for_class(artifact, predicted, text)
    return MLPrediction(clause_type=str(predicted), confidence=confidence, top_terms=top_terms)
