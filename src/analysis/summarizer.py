"""
summarizer.py
-------------
Two kinds of summary, both fully local (no LLM, no network, no torch):

  1. Extractive summary (summarize_text / summarize_clause)
     Ranks the sentences of a passage by TF-IDF centrality — how similar each
     sentence is to the passage as a whole — and returns the top few in their
     original order. This never invents text, so it can't hallucinate a term
     the contract doesn't contain, which matters a lot for legal review.

  2. Fact-based overview (summarize_contract)
     A plain-English paragraph assembled from the *structured* outputs of the
     rest of the pipeline (parties, dates, governing law, detected clauses,
     risk headline). Because every sentence is built from an extracted fact,
     it's both readable and verifiable against the source.

The extractive method uses only scikit-learn + numpy, which are already
dependencies of the classifier, so this phase adds no new packages.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Sentence boundary: end punctuation followed by whitespace and a capital /
# digit / quote. Good enough for contract prose without an NLP dependency.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, dropping empties and trivially short bits."""
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text.replace("\n", " "))
    # Drop tiny fragments (stray numbers, headings) but keep real short sentences.
    return [s.strip() for s in parts if len(s.strip()) > 10]


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Extractive summary: the `max_sentences` most central sentences, kept in
    their original order.

    Centrality = cosine similarity between each sentence's TF-IDF vector and
    the mean (document) vector. The most representative sentences score highest.
    Falls back to returning the text as-is when it's already short.
    """
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences) if sentences else text.strip()

    try:
        vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # e.g. every token is a stop word -> nothing to vectorize.
        return " ".join(sentences[:max_sentences])

    # Document centroid, then cosine similarity of each sentence to it.
    centroid = np.asarray(matrix.mean(axis=0)).ravel()
    norms = np.linalg.norm(matrix.toarray(), axis=1)
    centroid_norm = np.linalg.norm(centroid) or 1.0
    scores = (matrix @ centroid) / (norms * centroid_norm + 1e-9)
    scores = np.asarray(scores).ravel()

    top_idx = sorted(np.argsort(scores)[-max_sentences:])
    return " ".join(sentences[i] for i in top_idx)


def summarize_clause(clause_text: str, max_sentences: int = 2) -> str:
    """A short 1-2 sentence gist of a single clause."""
    return summarize_text(clause_text, max_sentences=max_sentences)


def _join_list(items: list[str], max_items: int = 3) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    shown = items[:max_items]
    if len(items) > max_items:
        return ", ".join(shown) + f" (+{len(items) - max_items} more)"
    if len(shown) == 1:
        return shown[0]
    return ", ".join(shown[:-1]) + " and " + shown[-1]


def summarize_contract(clauses, entities, risk_report) -> str:
    """Assemble a plain-English overview from extracted facts.

    `entities` is a ContractEntities; `risk_report` is a RiskReport; `clauses`
    is a list of ClauseMatch. Any of them may be sparse — each sentence is
    only emitted when the underlying fact exists.
    """
    lines: list[str] = []

    parties = getattr(entities, "parties", []) or []
    if len(parties) >= 2:
        lines.append(f"This agreement is between {_join_list(parties, 4)}.")
    elif len(parties) == 1:
        lines.append(f"This agreement involves {parties[0]}.")

    if getattr(entities, "effective_date", None):
        term = getattr(entities, "term", None)
        if term:
            lines.append(f"It takes effect on {entities.effective_date} for a term of {term}.")
        else:
            lines.append(f"It takes effect on {entities.effective_date}.")

    if getattr(entities, "governing_law", None):
        lines.append(f"It is governed by the laws of {entities.governing_law}.")

    detected = sorted({c.clause_type.replace("_", " ").title()
                       for c in clauses if getattr(c, "clause_type", None)})
    if detected:
        lines.append(f"Key clauses identified: {_join_list(detected, 6)}.")

    money = getattr(entities, "monetary_amounts", []) or []
    if money:
        lines.append(f"Monetary figures referenced include {_join_list(money, 3)}.")

    notices = getattr(entities, "notice_periods", []) or []
    if notices:
        lines.append(f"Notice periods referenced: {_join_list(notices, 3)}.")

    if risk_report is not None:
        level = getattr(risk_report, "level", None)
        n = len(getattr(risk_report, "findings", []) or [])
        if level:
            lines.append(
                f"Overall risk is assessed as {level} "
                f"({risk_report.score}/100) across {n} finding(s)."
            )
            highs = [f.title for f in risk_report.findings if f.severity == "high"]
            if highs:
                lines.append(f"Highest-priority items: {_join_list(highs, 3)}.")

    if not lines:
        return "Not enough structured information was extracted to summarize this document."
    return " ".join(lines)
