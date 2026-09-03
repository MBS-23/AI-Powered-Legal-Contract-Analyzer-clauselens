"""
entity_extractor.py
-------------------
Rule-based extraction of the key facts a reviewer looks for first in any
contract: who the parties are, the important dates, the money, the governing
law, notice periods, and the term.

Why rules, not a NER model:
  - The target deployment (Streamlit Cloud, Python 3.14, no torch) can't run
    a transformer NER model, and spaCy wheels aren't guaranteed on 3.14.
  - For the well-structured, high-value fields below, carefully written regex
    over normalized contract text is precise, fast, explainable, and has zero
    heavy dependencies. Each finding carries the exact snippet it came from,
    so the UI can show evidence rather than an opaque label.

Everything here operates on the already-cleaned contract text produced by
src.preprocessing.text_cleaner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Date patterns
# ---------------------------------------------------------------------------
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
# "January 1, 2020" / "Jan 1 2020"
_DATE_LONG = rf"(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}"
# "15 January 2026" / "15th January, 2026" (day-first, common outside the US)
_DATE_DAY_FIRST = rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?,?\s+\d{{4}}"
# "1st day of January, 2020" / "1 day of January 2020"
_DATE_DAY_OF = rf"\d{{1,2}}(?:st|nd|rd|th)?\s+day\s+of\s+(?:{_MONTHS})\.?,?\s+\d{{4}}"
# "01/02/2020" or "2020-01-02"
_DATE_NUMERIC = r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})"
# Order matters: try "day of" before day-first before month-first.
_ANY_DATE = rf"{_DATE_DAY_OF}|{_DATE_DAY_FIRST}|{_DATE_LONG}|{_DATE_NUMERIC}"
_DATE_RE = re.compile(rf"(?:{_ANY_DATE})", re.IGNORECASE)

# Effective / agreement date, captured with its cue phrase for confidence.
_EFFECTIVE_DATE_RE = re.compile(
    rf"(?:effective\s+as\s+of|effective\s+date(?:\s+of)?|dated(?:\s+as\s+of)?|"
    rf"entered\s+into[^.\n]*?\bas\s+of|entered\s+into\s+on|made\s+(?:as\s+of|on)|"
    rf"\bas\s+of)\s+"
    rf"(?P<date>{_ANY_DATE})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Money, governing law, notice, term
# ---------------------------------------------------------------------------
_MONEY_RE = re.compile(
    r"(?:US\$|USD|\$)\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s?(?:million|billion|thousand))?"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d{2})?\s?(?:dollars|USD)",
    re.IGNORECASE,
)

# The cue phrase is case-insensitive (scoped (?i:...)) so "Governed"/"governed"
# both match; the captured jurisdiction stays case-sensitive because places are
# proper nouns (Delaware, New York), which avoids grabbing lowercase filler.
_GOVERNING_LAW_RE = re.compile(
    r"(?i:govern(?:ed|ing)\s+(?:by|in accordance with)?[^.\n]*?"
    r"laws?\s+of\s+(?:the\s+)?"
    r"(?:State\s+of\s+|Commonwealth\s+of\s+|Province\s+of\s+)?)"
    r"(?P<place>[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})",
)

_NOTICE_RE = re.compile(
    r"(?P<days>\d{1,3}|thirty|sixty|ninety|forty-five|fifteen|ten)\s*(?:\(\d+\)\s*)?"
    r"(?P<unit>days?|months?|years?)['’]?\s+(?:prior\s+)?(?:written\s+)?notice",
    re.IGNORECASE,
)

_TERM_RE = re.compile(
    r"(?:term\s+of\s+(?:this\s+agreement\s+)?(?:shall\s+be\s+|is\s+|will\s+be\s+)?"
    r"|for\s+(?:an?\s+)?(?:initial\s+)?(?:period|term)\s+of\s+)"
    r"(?P<num>\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+"
    r"(?P<unit>days?|months?|years?)",
    re.IGNORECASE,
)

# Preamble parties: "(by and) between X ... and Y ..."
_PARTIES_RE = re.compile(
    r"(?:by\s+and\s+)?between\s+(?P<a>[A-Z].+?)\s+and\s+(?P<b>[A-Z].+?)(?:\.|,\s+(?:each|collectively)|\n|\()",
    re.IGNORECASE | re.DOTALL,
)
# Defined party labels: ("Company"), ("the Supplier"), ("Licensee")
_DEFINED_PARTY_RE = re.compile(r'[\(\["“]\s*(?:the\s+)?"?([A-Z][A-Za-z ]{2,40}?)"?\s*[\)\]"”]')


@dataclass
class ContractEntities:
    parties: list[str] = field(default_factory=list)
    effective_date: str | None = None
    dates: list[str] = field(default_factory=list)
    governing_law: str | None = None
    monetary_amounts: list[str] = field(default_factory=list)
    notice_periods: list[str] = field(default_factory=list)
    term: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _dedupe_keep_order(items: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        key = it.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(it.strip())
        if limit and len(out) >= limit:
            break
    return out


def _clean_party(raw: str) -> str:
    """Trim a captured party string down to the entity name.

    Cuts at the first defined-term parenthesis or role phrase so we keep
    'Acme Corporation' rather than the whole run-on preamble sentence.
    """
    party = raw.strip().strip(",").strip()
    party = re.split(r"\s*[\(\[“]", party)[0]
    party = re.split(r",?\s+a\s+[A-Za-z]+\s+(?:corporation|company|limited)", party, flags=re.IGNORECASE)[0]
    # Keep it to a sane length; preamble captures can run long.
    return party.strip().strip(",").strip()[:120]


def extract_parties(text: str) -> list[str]:
    parties: list[str] = []
    m = _PARTIES_RE.search(text)
    if m:
        for grp in ("a", "b"):
            cleaned = _clean_party(m.group(grp))
            if cleaned and len(cleaned) > 2:
                parties.append(cleaned)
    if not parties:
        # Fallback: defined party labels near the top of the document.
        head = text[:2000]
        labels = [lbl for lbl in _DEFINED_PARTY_RE.findall(head) if lbl.lower() not in {"agreement", "effective date"}]
        parties = labels
    return _dedupe_keep_order(parties, limit=6)


def extract_entities(text: str) -> ContractEntities:
    """Extract all supported entity types from cleaned contract text."""
    if not text:
        return ContractEntities()

    # Effective date (with cue) first; fall back to the first date overall.
    eff_match = _EFFECTIVE_DATE_RE.search(text)
    effective_date = eff_match.group("date").strip() if eff_match else None

    all_dates = _dedupe_keep_order([m.group(0) for m in _DATE_RE.finditer(text)], limit=25)
    if effective_date is None and all_dates:
        effective_date = all_dates[0]

    gl_match = _GOVERNING_LAW_RE.search(text)
    governing_law = gl_match.group("place").strip() if gl_match else None

    money = _dedupe_keep_order([m.group(0).strip() for m in _MONEY_RE.finditer(text)], limit=25)

    notices = _dedupe_keep_order(
        [f"{m.group('days')} {m.group('unit')}".strip() for m in _NOTICE_RE.finditer(text)],
        limit=12,
    )

    term_match = _TERM_RE.search(text)
    term = f"{term_match.group('num')} {term_match.group('unit')}" if term_match else None

    return ContractEntities(
        parties=extract_parties(text),
        effective_date=effective_date,
        dates=all_dates,
        governing_law=governing_law,
        monetary_amounts=money,
        notice_periods=notices,
        term=term,
    )
