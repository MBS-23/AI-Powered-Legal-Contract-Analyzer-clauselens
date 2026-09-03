# ⚖️ ClauseLens — Legal Contract Analyzer

AI-assisted contract review — upload a PDF or DOCX and get clause
classification, key-entity extraction, a prioritized risk report, plain-English
summaries, a searchable contract library, and downloadable JSON/HTML reports.

Built on an **open-source, fully-local stack** (scikit-learn + Streamlit) — no
GPU, no paid API, no data leaves the machine. Designed to deploy to
**Streamlit Community Cloud** in a few clicks.

> ⚠️ This tool provides automated, informational analysis only. It is **not
> legal advice** and does not replace review by a qualified attorney.

---

## 📚 Documentation

- **[Full project documentation](docs/PROJECT_DOCUMENTATION.md)** — the complete from-scratch explanation: problem, architecture, every tech-stack decision and *why*, the ML model in depth, design, security, and results.
- **[Case study & LinkedIn post](docs/CASE_STUDY.md)** — a shareable writeup and a ready-to-post version.
- **[Security posture](SECURITY.md)** — threat model and OWASP Web/LLM Top-10 mapping.

## ✨ Features

| Capability | How it works |
|---|---|
| **Document ingestion** | PDF via `pdfplumber`, DOCX via `python-docx` (with a `docx2txt` fallback); scanned-PDF detection |
| **Clause classification** | scikit-learn **TF-IDF + Logistic Regression** trained on the **CUAD** dataset (41 clause types), with a keyword baseline fallback |
| **Entity extraction** | Rule-based parties, effective date, term, governing law, monetary amounts, notice periods |
| **Risk analysis** | High-risk clauses + missing protective clauses + red-flag language → a 0–100 risk score with ranked, actionable findings |
| **Summaries** | A fact-based plain-English overview + an extractive (TF-IDF centrality) key-points summary |
| **Contract library** | SQLite persistence of analyzed contracts and their clauses |
| **Semantic search** | TF-IDF cosine similarity search across every stored clause |
| **Portfolio dashboard** | Library-wide analytics — risk distribution, most common clauses, KPIs |
| **Learn section** | Interactive flip-card flashcards + articles on contract concepts |
| **Export** | Download a full analysis as JSON or a printable HTML report |
| **Security-hardened** | Escaped output (no XSS), parameterized SQL, validated uploads — see [SECURITY.md](SECURITY.md) |

## 📊 Model performance

The clause classifier is trained on 10,901 CUAD clause spans and evaluated on
1,368 held-out spans (split **by contract** to prevent leakage):

| Metric | Value |
|---|---|
| Test accuracy | **76.0%** |
| Macro-F1 | **0.66** |
| Weighted-F1 | **0.76** |
| Most-frequent baseline accuracy | 16.2% |
| Clause types | 41 |

Live metrics are shown in the app's **Model & About** page and stored in
`models/metrics.json`.

---

## 🚀 Quick start (local)

```bash
# 1. create + activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. install runtime dependencies
pip install -r requirements.txt

# 3. run the app (the trained model is committed, so it works immediately)
streamlit run app.py
```

Open http://localhost:8501 and upload
`sample_contracts/sample_services_agreement.docx` to see it end-to-end.

## 🧪 Tests

```bash
pytest -q
```

73 tests cover parsing, cleaning, clause classification, entity extraction,
the risk engine, summarization, storage, search, the end-to-end pipeline, and
**security** (XSS/HTML-injection escaping, SQL-injection resistance, and file-upload validation).

## 🛡️ Security

The app is built defensively — see **[SECURITY.md](SECURITY.md)** for the full
threat model and OWASP Web/LLM Top-10 mapping. Highlights:

- **No XSS / HTML injection** — all document-derived text is HTML-escaped before render
- **No SQL injection** — every query is parameterized
- **Upload validation** — size, extension, and magic-byte checks before parsing
- **No LLM attack surface** — classification is scikit-learn, not a prompt-driven model
- **Local-only** — no outbound network calls at runtime; nothing leaves the machine

---

## 🔁 Reproducing the model (optional)

The trained model is already committed. To rebuild it from scratch:

```bash
pip install -r requirements.txt -r requirements-train.txt

# downloads CUAD (~18 MB) and writes leak-free train/val/test splits
python training/prepare_cuad_dataset.py

# trains, evaluates vs baseline, and saves models/clause_classifier.joblib
python training/train_classifier.py
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repository to GitHub (the committed
   `models/clause_classifier.joblib` is required at runtime).
2. Go to **share.streamlit.io** → **New app**, pick the repo/branch, and set
   **Main file path** to `app.py`.
3. Under **Advanced settings**, choose **Python 3.13** (torch/transformers are
   intentionally not used, so the lean `requirements.txt` installs quickly).
4. Deploy. Streamlit installs `requirements.txt`, loads the model, and serves
   the app.

The SQLite contract library lives on the app's ephemeral disk — fine for a
demo/portfolio deployment. For durable multi-user storage, swap the
`src/storage/db.py` backend for a hosted database (the module is the only place
that would change).

---

## 🧩 Architecture

```
raw file ─▶ src/document ─▶ src/preprocessing ─▶ src/extraction ─▶ src/analysis ─▶ app.py
           (pdf/docx)       (clean text)         (clauses + ML,     (risk +         (Streamlit UI)
                                                   entities)          summaries)
                                                         │
                                          src/storage (SQLite) ◀─┴─▶ src/search (TF-IDF)
```

```
legal-contract-analyzer/
├── app.py                          # Streamlit dashboard (presentation only)
├── src/
│   ├── core.py                     # end-to-end orchestration (AnalysisResult)
│   ├── report.py                   # HTML report generator
│   ├── document/                   # pdf_parser.py, docx_parser.py
│   ├── preprocessing/              # text_cleaner.py
│   ├── extraction/                 # clause_extractor.py, ml_classifier.py, entity_extractor.py
│   ├── analysis/                   # risk_engine.py, summarizer.py
│   ├── storage/                    # db.py (SQLite)
│   └── search/                     # semantic_search.py (TF-IDF)
├── training/                       # prepare_cuad_dataset.py, train_classifier.py
├── models/                         # clause_classifier.joblib, metrics.json  (committed)
├── tests/                          # 59 tests
├── sample_contracts/
├── requirements.txt                # app runtime deps
└── requirements-train.txt          # extra deps for dataset prep + training
```

---

## ⚙️ Tech stack

- **UI:** Streamlit + Plotly
- **ML:** scikit-learn (TF-IDF + Logistic Regression), joblib
- **Parsing:** pdfplumber, python-docx, docx2txt
- **Storage/search:** SQLite (stdlib), TF-IDF cosine similarity
- **Data:** [CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset)

## 📄 License & data

CUAD is released by The Atticus Project under CC BY 4.0. This project is for
educational and demonstration purposes.
