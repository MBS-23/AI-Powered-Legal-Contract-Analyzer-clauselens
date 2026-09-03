"""
core.py
-------
The single orchestration entry point that ties every stage of the pipeline
together, so both the Streamlit app and any tests/scripts run *exactly* the
same analysis path.

    raw file  ->  parse (pdf/docx)  ->  clean text
              ->  clause extraction (ML + keyword fallback)
              ->  entity extraction
              ->  risk analysis
              ->  summaries (fact-based overview + extractive)

Keeping this here (rather than in app.py) means the UI stays a thin
presentation layer and the business logic is importable and testable.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from src.document.pdf_parser import extract_text_from_pdf
from src.document.docx_parser import extract_text_from_docx
from src.preprocessing.text_cleaner import clean_text, strip_page_markers, normalize_whitespace
from src.extraction.clause_extractor import extract_clauses, ClauseMatch
from src.extraction.entity_extractor import extract_entities, ContractEntities
from src.analysis.risk_engine import analyze_risk, RiskReport
from src.analysis.summarizer import summarize_contract, summarize_text
from src.security import clean_text_input


@dataclass
class AnalysisResult:
    filename: str
    meta: dict
    clean_text: str
    clauses: list[ClauseMatch]
    entities: ContractEntities
    risk: RiskReport
    overview: str            # fact-based plain-English summary
    key_points: str          # extractive summary of the body
    used_ml: bool = False

    @property
    def detected_clauses(self) -> list[ClauseMatch]:
        return [c for c in self.clauses if c.clause_type]

    def to_export_dict(self) -> dict:
        """A JSON-serializable snapshot of the whole analysis."""
        return {
            "filename": self.filename,
            "meta": self.meta,
            "overview": self.overview,
            "key_points": self.key_points,
            "entities": self.entities.to_dict(),
            "risk": {
                "score": self.risk.score,
                "level": self.risk.level,
                "counts": self.risk.counts,
                "findings": [
                    {
                        "title": f.title,
                        "severity": f.severity,
                        "category": f.category,
                        "detail": f.detail,
                        "recommendation": f.recommendation,
                        "evidence": f.evidence,
                    }
                    for f in self.risk.findings
                ],
            },
            "clauses": [
                {
                    "section_number": c.section_number,
                    "heading": c.heading,
                    "clause_type": c.clause_type,
                    "confidence": c.confidence,
                    "method": c.method,
                    "matched_keywords": c.matched_keywords,
                    "text": c.text,
                }
                for c in self.clauses
            ],
        }


def _parse_file(path: str | Path) -> tuple[str, dict]:
    """Parse a PDF/DOCX file into (raw_text, meta)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        parsed = extract_text_from_pdf(path)
        meta = {
            "file_type": "PDF",
            "page_count": parsed.page_count,
            "likely_scanned": parsed.likely_scanned,
        }
        return parsed.full_text, meta
    if suffix == ".docx":
        parsed = extract_text_from_docx(path)
        meta = {
            "file_type": "DOCX",
            "paragraph_count": parsed.paragraph_count,
            "likely_scanned": False,
        }
        return parsed.full_text, meta
    raise ValueError(f"Unsupported file type: {suffix}. Upload a .pdf or .docx file.")


def analyze_text(text: str, filename: str = "contract", meta: dict | None = None,
                 use_ml: bool = True) -> AnalysisResult:
    """Run the full analysis pipeline on already-extracted text."""
    # Security boundary: normalize + bound untrusted document text (strips
    # control chars, caps length) before any processing or storage.
    text = clean_text_input(text)
    cleaned = normalize_whitespace(clean_text(strip_page_markers(text)))

    clauses = extract_clauses(cleaned, use_ml=use_ml)
    entities = extract_entities(cleaned)
    risk = analyze_risk(clauses, cleaned)
    overview = summarize_contract(clauses, entities, risk)
    key_points = summarize_text(cleaned, max_sentences=4)
    used_ml = any(c.method == "ml" for c in clauses)

    return AnalysisResult(
        filename=filename,
        meta=meta or {},
        clean_text=cleaned,
        clauses=clauses,
        entities=entities,
        risk=risk,
        overview=overview,
        key_points=key_points,
        used_ml=used_ml,
    )


def analyze_file(path: str | Path, use_ml: bool = True) -> AnalysisResult:
    """Parse a file from disk and run the full pipeline."""
    raw_text, meta = _parse_file(path)
    return analyze_text(raw_text, filename=Path(path).name, meta=meta, use_ml=use_ml)


def analyze_upload(uploaded_file, use_ml: bool = True) -> AnalysisResult:
    """Analyze a Streamlit UploadedFile by writing it to a temp file first."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        raw_text, meta = _parse_file(tmp_path)
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass
    return analyze_text(raw_text, filename=uploaded_file.name, meta=meta, use_ml=use_ml)
