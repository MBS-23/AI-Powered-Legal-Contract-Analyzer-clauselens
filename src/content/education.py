"""
education.py
------------
Static educational content for the Learn section: flashcards (concept Q&A)
and short articles about legal contracts and the clause types this tool
detects. All content is authored/curated here (no external calls), so it is
trusted and safe to render — but the UI still escapes it defensively.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Flashcard:
    front: str
    back: str
    category: str


@dataclass(frozen=True)
class Article:
    title: str
    icon: str
    summary: str
    body: str          # markdown
    read_minutes: int


FLASHCARDS: list[Flashcard] = [
    Flashcard(
        "What is an indemnification clause?",
        "A promise by one party to cover the losses, damages, or legal costs of "
        "the other party if certain events occur (e.g. a third-party lawsuit). It "
        "shifts risk from one party to another.",
        "Risk allocation",
    ),
    Flashcard(
        "What does 'limitation of liability' (cap on liability) do?",
        "It sets a ceiling on how much one party can be required to pay the other "
        "for losses under the contract — often capped at the fees paid in the "
        "prior 12 months. Without it, exposure can be unlimited.",
        "Risk allocation",
    ),
    Flashcard(
        "What is a 'governing law' clause?",
        "It specifies which jurisdiction's laws will be used to interpret and "
        "enforce the contract, and often where disputes must be heard. It brings "
        "predictability to how the agreement is read.",
        "Boilerplate",
    ),
    Flashcard(
        "What is 'force majeure'?",
        "A clause excusing a party from performing its obligations when "
        "extraordinary events beyond its control (natural disasters, war, "
        "pandemics) make performance impossible.",
        "Boilerplate",
    ),
    Flashcard(
        "What is a confidentiality (NDA) clause?",
        "It obliges the parties to keep specified information secret and to use "
        "it only for the purposes of the agreement, protecting trade secrets and "
        "sensitive data.",
        "Protection",
    ),
    Flashcard(
        "What is 'termination for convenience'?",
        "A right for a party to end the contract without cause, usually with "
        "advance notice. It creates flexibility for one side but revenue "
        "uncertainty for the other.",
        "Term & exit",
    ),
    Flashcard(
        "What is an anti-assignment clause?",
        "It restricts a party from transferring its rights or obligations under "
        "the contract to someone else without consent — which can complicate "
        "mergers, acquisitions, or restructuring.",
        "Change of control",
    ),
    Flashcard(
        "What is a 'most favored nation' (MFN) clause?",
        "A promise that one party will receive terms at least as good as those "
        "given to any other counterparty. It constrains future deals and can be "
        "hard to administer.",
        "Commercial",
    ),
    Flashcard(
        "What does 'IP ownership assignment' mean?",
        "A transfer of ownership of intellectual property (e.g. work product, "
        "inventions) from one party to another. Watch carefully: you may be "
        "giving away rights to what you create.",
        "Intellectual property",
    ),
    Flashcard(
        "What is 'liquidated damages'?",
        "A pre-agreed, fixed sum payable on a specific breach, set in advance so "
        "the parties don't have to prove actual loss. If it's unreasonably high, "
        "courts may treat it as an unenforceable penalty.",
        "Remedies",
    ),
    Flashcard(
        "What are 'audit rights'?",
        "A right allowing one party to inspect the other's records to verify "
        "compliance (e.g. royalties, usage, security). Well-drafted versions "
        "bound the frequency, scope, and cost.",
        "Compliance",
    ),
    Flashcard(
        "Why does a *missing* clause matter?",
        "Absence is itself a risk. No liability cap can mean unlimited exposure; "
        "no termination clause can mean unclear exit rights. Good review checks "
        "for what *should* be there, not just what is.",
        "Review method",
    ),
]


ARTICLES: list[Article] = [
    Article(
        "What is a Legal Contract Analyzer?",
        "🏛️",
        "How AI-assisted tooling speeds up contract review without replacing lawyers.",
        """
A **legal contract analyzer** is software that reads an agreement and pulls out
the information a reviewer needs first — the parties, the dates, the money, and
the clauses that carry risk — so a human can focus on judgement instead of
hunting through pages of boilerplate.

**What it typically does**

- **Data extraction** — parties, effective dates, payment terms, obligations.
- **Clause classification** — labelling each section (indemnity, governing law, termination, …).
- **Risk detection** — flagging unusual clauses, missing protections, and risky language.
- **Search** — finding similar clauses across a whole library of contracts.

**Why it helps**

- **Speed** — routine first-pass review that once took hours takes minutes.
- **Consistency** — the same checklist is applied to every document.
- **Coverage** — nothing gets skipped because a reviewer was tired on page 40.

An analyzer is an *assistant*, not a substitute for legal advice: it surfaces
what to look at, and a qualified professional decides what to do about it.
""",
        3,
    ),
    Article(
        "Anatomy of a Contract: The Clauses That Matter",
        "📑",
        "A tour of the high-value clauses and why reviewers zero in on them.",
        """
Most commercial contracts share a common skeleton. Knowing it makes review far
faster.

**The commercial core**
- **Payment & fees** — how much, when, and what happens on late payment.
- **Term & renewal** — how long it lasts and whether it auto-renews.
- **Termination** — how each side can exit, and with how much notice.

**The risk-allocation layer**
- **Limitation of liability** — the single most negotiated clause; it caps exposure.
- **Indemnification** — who covers whom when things go wrong.
- **Warranties** — promises about quality; watch for "as is" disclaimers.

**The protection layer**
- **Confidentiality** — keeps sensitive information private.
- **Intellectual property** — who owns what is created or licensed.
- **Insurance** — required coverage as a backstop.

**The boilerplate that isn't boring**
- **Governing law** — whose rules apply and where disputes are heard.
- **Assignment** — whether the deal can be transferred (matters in M&A).
- **Force majeure** — relief when the extraordinary happens.

A good analyzer tags each of these and tells you not only what's present, but
what's conspicuously **absent**.
""",
        4,
    ),
    Article(
        "Reading Risk: Red Flags in Contract Language",
        "⚠️",
        "The words and structures that should make a reviewer slow down.",
        """
Risk often hides in a handful of phrases. A few worth pausing on:

- **"Sole discretion"** — one party decides unilaterally. Fine for trivia,
  dangerous for anything that affects price, scope, or termination.
- **"Irrevocable and perpetual"** — a grant that can never be undone. Confirm the
  scope is genuinely intended to be permanent.
- **"Any and all"** — a catch-all that can sweep in far more than expected,
  especially around indemnities.
- **"As is" / "no warranty"** — you accept quality risk with no promise of fitness.
- **"Automatically renew"** — convenient, but a missed opt-out window locks you in.
- **Unlimited or uncapped liability** — exposure can exceed the entire contract value.

Equally important is the **missing** protection: no liability cap, no
confidentiality clause, no clear exit right. A disciplined review scores both
what the contract says and what it fails to say — which is exactly how the Risk
tab in this app works.
""",
        4,
    ),
    Article(
        "How This Tool Works — and Its Limits",
        "🧠",
        "The pipeline behind the dashboard, in plain language, and where to be careful.",
        """
This analyzer runs entirely on an **open-source, local pipeline** — no external
AI service, and nothing you upload leaves the machine unless you save it.

**The pipeline**
1. **Ingest** the PDF/DOCX and clean the text.
2. **Classify** each section with a machine-learning model trained on the public
   **CUAD** dataset (41 clause types), backed up by a keyword baseline.
3. **Extract** entities (parties, dates, money, governing law) with precise rules.
4. **Score risk** from risky clauses present, protections missing, and red-flag language.
5. **Summarize** in plain English, built only from extracted facts.

**Where to be careful**
- The model is trained on commercial contracts; unusual document types may match poorly.
- Scanned image PDFs have no text to read (there's no OCR step) — they're flagged, not processed.
- The output is **informational only**. It is not legal advice and does not
  replace review by a qualified attorney.

Used well, it turns a blank page of legalese into a structured, searchable,
risk-ranked starting point.
""",
        3,
    ),
]


def flashcard_categories() -> list[str]:
    return sorted({c.category for c in FLASHCARDS})
