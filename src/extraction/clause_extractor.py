"""
clause_extractor.py
--------------------
Phase 1 clause extraction: rule-based, not ML.

Two steps:
  1. segment_into_sections() splits the contract into sections using
     numbered/lettered heading patterns ("8. LIABILITY", "Section 8:",
     "ARTICLE VIII", etc).
  2. classify_section() scores each section's body text against a
     keyword dictionary per clause type and returns the best match with
     a confidence score.

This is intentionally the "dumb but honest" baseline described in the
project plan — it exists so:
  (a) the app is demoable end-to-end before any model is trained, and
  (b) Phase 2 has something concrete to beat. When you train the TF-IDF/
      SVM and Legal-BERT classifiers, report their accuracy AGAINST this
      keyword baseline — that comparison is a good portfolio talking
      point ("naive keyword matching got 58% clause-type accuracy on my
      hand-labeled eval set; the SVM got 84%; Legal-BERT got 91%").

The keyword list is deliberately short and editable — expand it as you
look at real contracts and see what your baseline misses.
"""

import re
from dataclasses import dataclass

from src.extraction import ml_classifier

# Matches heading lines such as:
#   "8. LIABILITY"
#   "8.1 Limitation of Liability"
#   "Section 8: Liability"
#   "SECTION 8 - LIABILITY"
#   "ARTICLE VIII. LIABILITY"
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?P<num>\d+(?:\.\d+)*)[\.\)]?\s+(?P<title1>[A-Z][A-Za-z0-9 ,&/\-]{2,80})"
    r"|"
    r"(?:SECTION|ARTICLE)\s+(?P<numword>[\dIVXLC]+)\s*[:\.\-]?\s*(?P<title2>[A-Za-z0-9 ,&/\-]{2,80})"
    r")\s*$"
)

CLAUSE_KEYWORDS: dict[str, list[str]] = {
    "CONFIDENTIALITY": ["confidential", "non-disclosure", "proprietary information"],
    "INDEMNIFICATION": ["indemnify", "indemnification", "hold harmless"],
    "TERMINATION": ["terminate", "termination", "notice of termination"],
    "GOVERNING_LAW": ["governing law", "governed by the laws", "jurisdiction"],
    "LIABILITY": ["liable", "liability", "limitation of liability", "consequential damages"],
    "PAYMENT": ["invoice", "payment terms", "shall pay", "fees due"],
    "NON_COMPETE": ["non-compete", "non-solicitation", "restrictive covenant"],
    "INTELLECTUAL_PROPERTY": ["intellectual property", "copyright", "trademark", "work product"],
    "DISPUTE_RESOLUTION": ["arbitration", "mediation", "dispute resolution"],
    "WARRANTY": ["warrant", "warranty", "as is", "merchantability"],
    "ASSIGNMENT": ["assign this agreement", "assignment", "successors and assigns"],
    "FORCE_MAJEURE": ["force majeure", "acts of god", "beyond its reasonable control"],
    "DATA_PROTECTION": ["personal data", "data protection", "gdpr", "data processing"],
    "AUDIT": ["right to audit", "audit rights", "inspect the records"],
    "INSURANCE": ["insurance coverage", "certificate of insurance", "maintain insurance"],
}


@dataclass
class Section:
    number: str  # e.g. "8" or "8.1"; empty string if no heading matched
    heading: str
    body: str
    start_line: int


@dataclass
class ClauseMatch:
    section_number: str
    heading: str
    text: str
    clause_type: str | None  # None if nothing matched confidently
    confidence: float  # 0.0-1.0 (ML probability, or keyword-density score)
    matched_keywords: list[str]
    method: str = "keyword"  # "ml" (trained classifier) or "keyword" (baseline)


def segment_into_sections(text: str) -> list[Section]:
    """Split contract text into sections based on numbered/lettered headings.

    Falls back to treating the whole document as one section if no
    headings are detected (common with poorly-formatted contracts) —
    the caller can still run clause classification on that single block,
    it just won't have per-clause granularity.
    """
    lines = text.splitlines()
    sections: list[Section] = []
    current_heading = ""
    current_number = ""
    current_body: list[str] = []
    current_start = 0
    any_heading_matched = False

    def flush():
        if current_body or current_heading:
            sections.append(
                Section(
                    number=current_number,
                    heading=current_heading,
                    body="\n".join(current_body).strip(),
                    start_line=current_start,
                )
            )

    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            any_heading_matched = True
            flush()
            current_number = m.group("num") or m.group("numword") or ""
            current_heading = (m.group("title1") or m.group("title2") or "").strip()
            current_body = []
            current_start = i
        else:
            current_body.append(line)
    flush()

    if not any_heading_matched:
        # No headings detected anywhere in the doc — treat it as one
        # untitled section rather than returning fragments with no label.
        return [Section(number="", heading="(untitled document)", body=text.strip(), start_line=0)]

    return sections


def classify_section(section: Section) -> tuple[str | None, float, list[str]]:
    """Score a section's heading+body against the keyword dictionary.

    Confidence is a simple normalized keyword-hit score, NOT a calibrated
    probability — it's a placeholder Phase 2's trained classifier will
    replace. Treat any confidence below ~0.15 as "no confident match."
    """
    haystack = f"{section.heading} {section.body}".lower()
    best_type = None
    best_score = 0.0
    best_matches: list[str] = []

    for clause_type, keywords in CLAUSE_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in haystack]
        if not matches:
            continue
        # Score: fraction of this clause's keyword list that appears,
        # with a small bonus if a keyword appears in the heading itself
        # (headings are a much stronger signal than body mentions).
        score = len(matches) / len(keywords)
        if any(kw in section.heading.lower() for kw in matches):
            score += 0.3
        if score > best_score:
            best_score = score
            best_type = clause_type
            best_matches = matches

    confidence = min(best_score, 1.0)
    if confidence < 0.15:
        return None, confidence, []
    return best_type, confidence, best_matches


def extract_clauses(text: str, use_ml: bool = True) -> list[ClauseMatch]:
    """Segment the contract into sections and classify each one.

    Classification strategy:
      - If ``use_ml`` and the trained model is available, each section is
        labelled by the scikit-learn classifier (41 CUAD clause categories,
        calibrated confidence). Sections the model isn't confident about fall
        back to the Phase 1 keyword baseline, so a section is only left
        unlabelled when *both* methods decline.
      - If the model is missing (fresh checkout, or ``use_ml=False``), the
        keyword baseline is used throughout.

    The return type is unchanged from Phase 1 (``list[ClauseMatch]``) — only
    the ``method`` field and the source of ``clause_type``/``confidence``
    differ — so callers like the dashboard need no changes.
    """
    ml_ready = use_ml and ml_classifier.is_available()

    sections = segment_into_sections(text)
    results: list[ClauseMatch] = []
    for s in sections:
        clause_type: str | None = None
        confidence = 0.0
        matched: list[str] = []
        method = "keyword"

        if ml_ready:
            pred = ml_classifier.predict(f"{s.heading}\n{s.body}".strip())
            if pred.clause_type is not None:
                clause_type = pred.clause_type
                confidence = pred.confidence
                matched = pred.top_terms
                method = "ml"

        # Fall back to the keyword baseline when ML is unavailable or unsure.
        if clause_type is None:
            kw_type, kw_conf, kw_matched = classify_section(s)
            clause_type, confidence, matched = kw_type, kw_conf, kw_matched
            method = "keyword"

        results.append(
            ClauseMatch(
                section_number=s.number,
                heading=s.heading,
                text=s.body,
                clause_type=clause_type,
                confidence=confidence,
                matched_keywords=matched,
                method=method,
            )
        )
    return results
