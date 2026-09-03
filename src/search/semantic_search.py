"""
semantic_search.py
------------------
Similarity search across a corpus of clauses, so a reviewer can ask "show me
all the indemnification-style clauses" or paste a clause and find the closest
matches across every stored contract.

Implementation note (honest about the method):
  This uses TF-IDF vectors + cosine similarity, NOT transformer embeddings.
  Sentence-transformer embeddings need torch, which has no Python 3.14 wheel
  and is too heavy for the target deployment. TF-IDF with word 1-2 grams plus
  sublinear term-frequency scaling captures a lot of the useful signal for
  legal boilerplate (which is highly formulaic and repetitive), and it indexes
  and queries in milliseconds with no model download. The interface below is
  deliberately small, so a future embedding backend could replace the internals
  without touching callers.

Typical use:
    index = SemanticSearchIndex.from_clauses(get_all_clauses())
    results = index.search("limitation of liability", top_k=5)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SearchResult:
    score: float               # cosine similarity, 0.0-1.0
    text: str
    heading: str
    clause_type: str | None
    filename: str
    contract_id: int | None
    clause_id: int | None


class SemanticSearchIndex:
    """A TF-IDF cosine-similarity index over a list of clause dicts."""

    def __init__(self, clauses: list[dict]):
        # Keep only clauses with usable text.
        self._clauses = [c for c in (clauses or []) if (c.get("text") or "").strip()]
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        if self._clauses:
            corpus = [self._doc_text(c) for c in self._clauses]
            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                stop_words="english",
                strip_accents="unicode",
            )
            self._matrix = self._vectorizer.fit_transform(corpus)

    @staticmethod
    def _doc_text(clause: dict) -> str:
        """Index the heading + body together; headings carry strong signal."""
        heading = clause.get("heading") or ""
        text = clause.get("text") or ""
        return f"{heading} {text}".strip()

    @classmethod
    def from_clauses(cls, clauses: list[dict]) -> "SemanticSearchIndex":
        return cls(clauses)

    @property
    def size(self) -> int:
        return len(self._clauses)

    @property
    def is_empty(self) -> bool:
        return self._matrix is None or self.size == 0

    def search(self, query: str, top_k: int = 5, min_score: float = 0.01) -> list[SearchResult]:
        """Return the top_k most similar clauses to `query`, best first.

        Results below `min_score` are dropped so an unrelated query returns
        nothing rather than noise.
        """
        if self.is_empty or not query or not query.strip():
            return []

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix).ravel()

        # top_k highest, then filter + sort descending.
        k = min(top_k, len(sims))
        top_idx = np.argsort(sims)[-k:][::-1]

        results: list[SearchResult] = []
        for i in top_idx:
            score = float(sims[i])
            if score < min_score:
                continue
            c = self._clauses[i]
            results.append(
                SearchResult(
                    score=score,
                    text=c.get("text", ""),
                    heading=c.get("heading", ""),
                    clause_type=c.get("clause_type"),
                    filename=c.get("filename", ""),
                    contract_id=c.get("contract_id"),
                    clause_id=c.get("id"),
                )
            )
        return results
