# ClauseLens

## AI-Powered Legal Contract Analyzer

*A project by Podugu Bala Veera Venkata Sunil*

---

ClauseLens is an AI-assisted platform that helps with the first-pass review of legal contracts. Upload a PDF or Word agreement and it reads the document, identifies each clause, extracts the key terms, scores the risk, and presents everything through a clean, interactive interface — turning what is normally hours of careful reading into a structured review that takes minutes.

This document explains what the project does, how it is built, and — most importantly — **why each technology was chosen**. It is written for a general professional audience, so it stays at the level of design decisions rather than internal implementation detail.

---

## The problem it addresses

Reviewing a commercial contract is slow, repetitive, and easy to get wrong. A reviewer has to read pages of dense legal language to find the handful of clauses that actually carry risk, pull out the commercial facts (who, when, how much, under whose law), and notice not only the risky terms that are present but the protective terms that are missing. Doing this consistently across many documents is exactly where fatigue causes mistakes.

ClauseLens removes that mechanical first pass so a person can spend their time on judgement.

---

## What it does

- **Reads PDF and Word contracts** and extracts clean text.
- **Identifies clauses** — classifies each section into one of 41 clause types (indemnity, limitation of liability, governing law, termination, and more).
- **Extracts key terms** — parties, effective date, term, governing law, monetary amounts, notice periods.
- **Scores risk (0–100)** — from risky clauses present, protective clauses missing, and red-flag language, with each finding shown alongside the text it came from.
- **Writes a plain-English summary** built only from the extracted facts.
- **Makes contracts searchable** — find similar clauses across a saved library.
- **Generates a professional report** — a branded document you can download and share.
- **Teaches the concepts** — a knowledge center with flashcards and articles.

---

## How it works

```
Upload contract  ->  Read & clean text  ->  Identify clauses  ->  Extract key terms
     ->  Score risk  ->  Summarize  ->  Report, dashboard & search
```

Each stage is a focused, independently tested component, so the system is transparent and maintainable rather than a black box.

---

## The technology — and why

| Technology | Why it was chosen |
|---|---|
| **Python** | The natural language for an NLP and machine-learning project, and it keeps the model and the app in one codebase. |
| **Streamlit** | Turns Python directly into an interactive, data-rich web application with no separate front-end to build, and it deploys for free. Ideal for a focused, single-user review tool. |
| **scikit-learn (TF-IDF + Logistic Regression)** | A lightweight, explainable model that runs on CPU in milliseconds, is a few megabytes on disk, and can be measured and trusted. |
| **CUAD dataset** | The standard, expert-annotated public dataset for contract clause understanding (41 clause categories) — real, legally grounded training data. |
| **Rule-based extraction & risk scoring** | Contract entities and risk signals follow regular, well-understood patterns; precise rules are more accurate and fully auditable here than a general model. |
| **Plotly & SQLite** | Interactive, publication-quality charts and a zero-configuration local database for the contract library. |

### The key engineering decision: classical ML, not a large language model

The most important choice in this project was to **not** use a large language model for analysis. Classical machine learning gave me a model that is:

- **Private** — contract text never leaves the application during analysis.
- **Instant and free** — millisecond inference on an ordinary CPU, no per-request cost.
- **Explainable and deterministic** — the same contract always produces the same result, and every finding cites its evidence.
- **Safe** — it cannot invent a clause that isn't in the document.

A larger model might score a few points higher on accuracy, but at roughly a hundred times the size and cost, with new failure modes. For a private, deployable, decision-support tool, the classical approach was the *engineering win* — not a compromise. Choosing the right-sized model for the constraints is itself the skill on display.

---

## Results

On the standard public test set, the clause classifier reaches:

| Metric | Value |
|---|---|
| Accuracy | 77.1% across 41 clause types |
| Balanced accuracy (macro-F1) | 0.70 |
| Improvement over a naive baseline | ~4.8x |

A full contract is analyzed in seconds on an ordinary machine. These numbers reflect a lightweight model built for deployment; ClauseLens is a first-pass aid, not a substitute for professional judgement.

---

## Design and experience

ClauseLens is designed to feel like a professional legal-technology product, not a demo:

- A **deep-navy, teal and gold** visual identity — the colors of law and enterprise software.
- A refined interface with serif display headings, considered spacing, and subtle, purposeful motion.
- A complete **light and dark theme** system with a single toggle, so both modes are intentionally designed variants of the same brand.
- Clear, risk-focused visualizations and a polished, downloadable report.

The application was also built with **security and privacy in mind** — input is validated and handled defensively, and nothing is sent to any third-party service during analysis.

---

## Honest limitations

- Accuracy is around 77% on an imbalanced, 41-category task; unusual or ambiguous contracts may classify less well.
- It does not read scanned image-only PDFs (these are detected and flagged).
- The risk score is a transparent, rule-based indicator, not a legal judgement.

**ClauseLens is a decision-support tool. It does not provide legal advice and does not replace a qualified attorney.** Every result should be reviewed by a professional.

---

## What this project demonstrates

- End-to-end delivery: document processing, a trained ML model, rule-based analysis, an interactive UI, storage, search, and reporting.
- **Engineering judgement** — matching the technology to real constraints (privacy, cost, explainability, deployability) rather than reaching for the largest available model.
- Attention to product quality: design, testing, and an honest account of what the system can and cannot do.

---

## About the developer

**Podugu Bala Veera Venkata Sunil**

Interested in AI, machine learning, and applied natural language processing.

- GitHub: *add your profile link*
- LinkedIn: *add your profile link*

---

*Built with Python, scikit-learn, and Streamlit. ClauseLens is an AI-assisted decision-support tool for contract review and does not constitute legal advice.*
