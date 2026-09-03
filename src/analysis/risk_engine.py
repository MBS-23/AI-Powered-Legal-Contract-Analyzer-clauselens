"""
risk_engine.py
--------------
Turns the structured output of the extraction pipeline into a prioritized
risk report — the "what should a lawyer look at first?" layer.

It combines three independent signals, which is how a human reviewer actually
works:

  1. Present high-risk clauses  — some clause *types* are inherently risky to
     the reviewing party (uncapped liability, perpetual licenses, most-favored-
     nation, auto-renewal traps, broad IP assignment...). If the classifier
     found one, we surface it with a severity and a recommendation.

  2. Missing protective clauses — some clauses you *want* to see; their absence
     is itself a risk (no liability cap, no governing law, no termination
     right, no confidentiality). We check the detected clause-type set against
     an expected checklist.

  3. Red-flag language         — specific phrasings ("sole discretion",
     "irrevocable and perpetual", "any and all", "as is", "waives any right")
     that are risky regardless of which section they sit in. Found by regex
     over the full text, each with the evidence snippet.

The three feed a single 0-100 risk score and a Low/Medium/High level, so the
dashboard can lead with a headline and then drill into the ranked findings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from src.extraction.clause_extractor import CLAUSE_KEYWORDS

# Severity -> points contributed toward the overall risk score.
_SEVERITY_POINTS = {"high": 22, "medium": 11, "low": 4}
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------------------
# 1. Present high-risk clause types (keyed to the CUAD classifier taxonomy,
#    with keyword-baseline aliases in parentheses handled via the alias map).
# ---------------------------------------------------------------------------
_HIGH_RISK_CLAUSES: dict[str, tuple[str, str, str]] = {
    # clause_type: (severity, why, recommendation)
    "UNCAPPED_LIABILITY": (
        "high",
        "Liability is not capped — exposure could exceed the contract value.",
        "Negotiate a liability cap (e.g. fees paid in the prior 12 months).",
    ),
    "LIQUIDATED_DAMAGES": (
        "high",
        "Pre-agreed damages can be triggered automatically on breach.",
        "Confirm the amount is a reasonable estimate, not a penalty.",
    ),
    "IRREVOCABLE_OR_PERPETUAL_LICENSE": (
        "high",
        "A perpetual/irrevocable license cannot be walked back later.",
        "Confirm scope and consider a term limit or revocation trigger.",
    ),
    "IP_OWNERSHIP_ASSIGNMENT": (
        "high",
        "IP ownership is being assigned — you may lose rights to work product.",
        "Confirm which party keeps ownership of pre-existing and new IP.",
    ),
    "MOST_FAVORED_NATION": (
        "medium",
        "Most-favored-nation pricing constrains future deals with others.",
        "Scope the MFN narrowly and time-box it.",
    ),
    "NON_COMPETE": (
        "medium",
        "A non-compete can restrict future business activity.",
        "Check duration, geography, and scope for enforceability.",
    ),
    "MINIMUM_COMMITMENT": (
        "medium",
        "A minimum spend/volume commitment creates fixed exposure.",
        "Model the worst case and align it to realistic demand.",
    ),
    "ANTI_ASSIGNMENT": (
        "medium",
        "Assignment is restricted — this can block M&A or restructuring.",
        "Carve out assignment to affiliates and change-of-control.",
    ),
    "CHANGE_OF_CONTROL": (
        "medium",
        "A change-of-control clause may let the counterparty exit or re-price.",
        "Understand what a sale/merger triggers.",
    ),
    "COVENANT_NOT_TO_SUE": (
        "medium",
        "You may be waiving the right to bring certain claims.",
        "Confirm which claims are being given up.",
    ),
    "TERMINATION_FOR_CONVENIENCE": (
        "low",
        "The counterparty can terminate for convenience — revenue is less certain.",
        "Check the notice period and any wind-down/kill fees.",
    ),
    "AUDIT_RIGHTS": (
        "low",
        "The counterparty can audit your records.",
        "Bound audit frequency, scope, and who bears the cost.",
    ),
}

# Keyword-baseline labels mapped onto the risk taxonomy above.
_CLAUSE_ALIASES = {
    "LIABILITY": "UNCAPPED_LIABILITY",
    "INTELLECTUAL_PROPERTY": "IP_OWNERSHIP_ASSIGNMENT",
    "NON_COMPETE": "NON_COMPETE",
    "AUDIT": "AUDIT_RIGHTS",
    "TERMINATION": "TERMINATION_FOR_CONVENIENCE",
}


# ---------------------------------------------------------------------------
# 2. Expected protective clauses — absence is a risk. Value = (severity, why).
#    Membership is tested against BOTH taxonomies (ML + keyword aliases).
# ---------------------------------------------------------------------------
_EXPECTED_CLAUSES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "Liability cap": (
        "high",
        "No cap-on-liability clause was detected — liability may be unlimited.",
        ("CAP_ON_LIABILITY",),
    ),
    "Governing law": (
        "medium",
        "No governing-law clause was detected — the venue for disputes is unclear.",
        ("GOVERNING_LAW",),
    ),
    "Termination": (
        "medium",
        "No termination clause was detected — exit rights are unclear.",
        ("TERMINATION_FOR_CONVENIENCE", "TERMINATION", "NOTICE_PERIOD_TO_TERMINATE_RENEWAL"),
    ),
    "Confidentiality": (
        "medium",
        "No confidentiality clause was detected — information sharing is unprotected.",
        ("CONFIDENTIALITY",),
    ),
    "Insurance": (
        "low",
        "No insurance clause was detected.",
        ("INSURANCE",),
    ),
}


# ---------------------------------------------------------------------------
# 3. Red-flag language patterns. (pattern, title, severity, recommendation)
# ---------------------------------------------------------------------------
_RED_FLAGS: list[tuple[re.Pattern, str, str, str]] = [
    (
        re.compile(r"\bsole\s+discretion\b", re.IGNORECASE),
        "Unilateral 'sole discretion'",
        "medium",
        "One party decides unilaterally — check what it applies to.",
    ),
    (
        re.compile(r"\b(?:un)?limited\s+liability\b|\bwithout\s+limitation\s+of\s+liability\b|\bunlimited\s+liability\b", re.IGNORECASE),
        "Unlimited liability language",
        "high",
        "Confirm whether liability is genuinely uncapped.",
    ),
    (
        re.compile(r"\birrevocab\w*\b.{0,40}\bperpetu\w*|\bperpetu\w*\b.{0,40}\birrevocab\w*", re.IGNORECASE | re.DOTALL),
        "Irrevocable and perpetual grant",
        "high",
        "A grant that is both irrevocable and perpetual cannot be undone.",
    ),
    (
        re.compile(r"\bany\s+and\s+all\b", re.IGNORECASE),
        "Broad 'any and all' language",
        "low",
        "Broad catch-all wording — confirm the intended scope.",
    ),
    (
        re.compile(r"\bas\s+is\b|\bno\s+warrant(?:y|ies)\b|\bdisclaim\w*\s+all\s+warrant", re.IGNORECASE),
        "Warranty disclaimer",
        "medium",
        "Goods/services provided 'as is' with no warranty — assess quality risk.",
    ),
    (
        re.compile(r"\bautomatically\s+renew\w*\b|\bauto[- ]?renew\w*\b", re.IGNORECASE),
        "Automatic renewal",
        "medium",
        "Auto-renewal can lock you in — note the opt-out notice window.",
    ),
    (
        re.compile(r"\bwaive[sd]?\b.{0,30}\bright", re.IGNORECASE | re.DOTALL),
        "Waiver of rights",
        "medium",
        "A right is being waived — confirm which one and the consequences.",
    ),
    (
        re.compile(r"\bindemnif\w+\b.{0,40}\bany\s+and\s+all\b", re.IGNORECASE | re.DOTALL),
        "Broad indemnification",
        "high",
        "Indemnity for 'any and all' claims is very broad — narrow it.",
    ),
]


@dataclass
class RiskFinding:
    title: str
    severity: str  # "high" | "medium" | "low"
    category: str  # "present_clause" | "missing_clause" | "language"
    detail: str
    recommendation: str
    evidence: str | None = None  # snippet, when the signal came from the text
    clause_type: str | None = None


@dataclass
class RiskReport:
    score: int  # 0-100, higher = riskier
    level: str  # "Low" | "Medium" | "High"
    findings: list[RiskFinding] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)  # severity -> count

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _canonical_clause(clause_type: str | None) -> str | None:
    if clause_type is None:
        return None
    return _CLAUSE_ALIASES.get(clause_type, clause_type)


def _snippet(text: str, match: re.Match, width: int = 70) -> str:
    start = max(match.start() - width, 0)
    end = min(match.end() + width, len(text))
    snip = text[start:end].replace("\n", " ").strip()
    return f"...{snip}..."


def analyze_risk(clauses, full_text: str) -> RiskReport:
    """Produce a ranked RiskReport from detected clauses + the raw text.

    `clauses` is a list of ClauseMatch (from extract_clauses); we only read
    their `.clause_type` and `.heading`/`.text`, so this stays decoupled from
    whichever classifier produced them.
    """
    findings: list[RiskFinding] = []

    detected_types: set[str] = set()
    for c in clauses:
        if getattr(c, "clause_type", None):
            detected_types.add(c.clause_type)
            detected_types.add(_canonical_clause(c.clause_type))

    # Augment with a keyword scan of the full text. The ML taxonomy (41 CUAD
    # categories) has no label for some clauses the keyword baseline covers
    # (e.g. CONFIDENTIALITY, PAYMENT), so a purely ML-based "is it present?"
    # check would flag those as missing even when they're clearly there.
    haystack = (full_text or "").lower()
    for clause_type, keywords in CLAUSE_KEYWORDS.items():
        if any(kw in haystack for kw in keywords):
            detected_types.add(clause_type)

    # ---- 1. Present high-risk clauses ----
    seen_present: set[str] = set()
    for c in clauses:
        canon = _canonical_clause(getattr(c, "clause_type", None))
        if canon in _HIGH_RISK_CLAUSES and canon not in seen_present:
            seen_present.add(canon)
            severity, why, rec = _HIGH_RISK_CLAUSES[canon]
            evidence = (c.text or c.heading or "")[:200] or None
            findings.append(
                RiskFinding(
                    title=f"High-risk clause: {canon.replace('_', ' ').title()}",
                    severity=severity,
                    category="present_clause",
                    detail=why,
                    recommendation=rec,
                    evidence=f"...{evidence}..." if evidence else None,
                    clause_type=canon,
                )
            )

    # ---- 2. Missing protective clauses ----
    for label, (severity, why, expected_types) in _EXPECTED_CLAUSES.items():
        if not any(t in detected_types for t in expected_types):
            findings.append(
                RiskFinding(
                    title=f"Missing clause: {label}",
                    severity=severity,
                    category="missing_clause",
                    detail=why,
                    recommendation=f"Add a {label.lower()} clause or confirm its omission is intentional.",
                    evidence=None,
                )
            )

    # ---- 3. Red-flag language ----
    seen_flags: set[str] = set()
    for pattern, title, severity, rec in _RED_FLAGS:
        m = pattern.search(full_text or "")
        if m and title not in seen_flags:
            seen_flags.add(title)
            findings.append(
                RiskFinding(
                    title=title,
                    severity=severity,
                    category="language",
                    detail="Risk-associated language detected in the contract text.",
                    recommendation=rec,
                    evidence=_snippet(full_text, m),
                )
            )

    # ---- Score + level ----
    raw = sum(_SEVERITY_POINTS[f.severity] for f in findings)
    score = min(raw, 100)
    if score >= 55:
        level = "High"
    elif score >= 25:
        level = "Medium"
    else:
        level = "Low"

    counts = {sev: sum(1 for f in findings if f.severity == sev) for sev in ("high", "medium", "low")}

    findings.sort(key=lambda f: _SEVERITY_ORDER[f.severity])
    return RiskReport(score=score, level=level, findings=findings, counts=counts)
