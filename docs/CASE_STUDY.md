# ClauseLens — Case Study & LinkedIn Post

This document is both a portfolio case study and a ready-to-share writeup. The first section is a **copy-paste-ready LinkedIn post**; the rest is a fuller case study you can keep with the project.

---

## 📋 Ready-to-post LinkedIn version

> Copy everything in this block, add 1–2 screenshots (the dashboard and a report), and post.

---

⚖️ I built **ClauseLens** — an AI-assisted legal contract analyzer that turns hours of contract review into a structured, minutes-long first pass.

Reviewing a contract is slow and repetitive: you read pages of boilerplate to find the few clauses that actually carry risk, pull out the who / when / how-much, and try to notice not just the risky terms that *are* there, but the protections that are *missing*. ClauseLens does that first pass for you.

**What it does 👇**
📄 Reads PDF & Word contracts and extracts clean text
🏷️ Identifies **41 clause types** with a trained ML model (76% accuracy vs a 16% baseline — a 4.7× lift)
🔍 Extracts parties, dates, money & governing law
⚠️ Scores legal risk 0–100 — flagging risky clauses *and* missing protections, each finding citing its evidence
📝 Writes a plain-English summary and a shareable, branded report
🔎 Makes every clause semantically searchable

**The interesting engineering decision:** I deliberately used **classical ML (TF-IDF + Logistic Regression on the CUAD dataset)** instead of a large language model. Why?
✅ Runs on CPU in milliseconds, deploys free, cold-starts instantly
✅ 100% private — no contract text ever leaves the process
✅ Deterministic & explainable — no hallucinated clauses, which is non-negotiable for a legal tool
✅ Immune to prompt-injection because there's no prompt in the loop

Sometimes the right answer isn't the biggest model — it's the one you can measure, trust, deploy, and defend. A transformer might score a few points higher at 100× the size and cost; for a private, deployable review tool, that trade wasn't worth it.

Built end-to-end with **Python, scikit-learn, Streamlit, and SQLite**, hardened against the OWASP Web & LLM Top 10, and covered by 73 automated tests.

⚠️ It's a decision-support tool, not a replacement for a qualified attorney.

Would you trust a classical ML model over an LLM for a task like this? Curious what others think. 👇

#MachineLearning #LegalTech #Python #DataScience #NLP #AI #scikitlearn #Streamlit #SoftwareEngineering #ContractManagement

---
---

## 📖 Full case study

### The problem
Contract review is a bottleneck in sales, procurement, and due diligence. It's slow, repetitive, and fatigue-prone — and the mistakes happen exactly where attention runs out, deep in the boilerplate. I wanted to remove the mechanical first pass so a human can spend their time on judgement.

### The solution
**ClauseLens** is an AI-assisted contract analyzer. Upload a PDF or Word contract and it:
- reads and cleans the document,
- classifies each section into one of 41 clause types,
- extracts the commercial facts (parties, dates, money, governing law, term, notice),
- scores risk 0–100 from risky clauses present, protective clauses missing, and red-flag language,
- writes a plain-English summary,
- and makes every clause searchable across a saved library.

It ships with a portfolio dashboard, a branded exportable report, and a "Learn" section of flashcards and articles on contract concepts.

### Key results
| Metric | Result |
|---|---|
| Clause-classification accuracy | **76.0%** (macro-F1 0.66) across 41 types |
| Baseline (majority/keyword) | 16.2% → **~4.7× improvement** |
| Training / test data | 10,901 / 1,368 CUAD clauses |
| Speed | Full contract analyzed in **seconds on CPU**, no network calls |
| Tests | **73 automated tests**, incl. a security suite |

### The engineering story: classical ML over an LLM
The defining decision was to **not** use a large language model at analysis time. Classical ML (TF-IDF + Logistic Regression) gave me a model that is a few MB, runs on CPU in milliseconds, deploys free, is fully explainable, and — critically for a legal tool — can never hallucinate a clause. It reaches 76% accuracy, more than enough for first-pass triage. A transformer would likely edge that higher, but at 100× the footprint and cost and with new failure modes (hallucination, prompt-injection, latency, price). For a private, deployable tool, the classical approach was the *engineering* win, not a compromise.

### Built with
`Python` · `scikit-learn` (TF-IDF + Logistic Regression) · `CUAD` dataset · `pdfplumber` / `python-docx` · rule-based entity extraction & risk scoring · `SQLite` · `Streamlit` · `Plotly`.

### Security
Engineered defensively from the start: output escaping (no XSS), parameterized SQL, magic-byte upload validation, input hardening — mapped to the OWASP Web & LLM Top 10 and covered by tests. With no LLM in the loop, there is no prompt-injection surface at all.

### What I learned
- Match the model to the constraints — deployability, privacy, explainability and cost are real requirements, not afterthoughts.
- In regulated domains, **determinism and traceability** can matter more than a few points of accuracy.
- A clean, layered architecture makes an "AI project" testable, honest, and shippable.

### Try it / see the code
🔗 GitHub: *[add your repository link]*
🔗 Live demo: *[add your Streamlit Cloud link]*

---

*ClauseLens provides automated analysis to accelerate review. It is a decision-support tool and does not replace advice from a qualified attorney.*
