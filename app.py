"""
app.py — Legal Contract Analyzer dashboard.

A single-file Streamlit presentation layer over the analysis pipeline in
src/. All business logic lives in src/core.py (orchestration), so this file
only handles layout, state, and rendering.

Features
  - Analyze Contract : upload a PDF/DOCX, get clauses, entities, a risk report,
                       plain-English summaries, and downloadable JSON/HTML.
  - Contract Library : browse/persist analyzed contracts (SQLite).
  - Semantic Search  : find similar clauses across the whole library.
  - Model & About    : classifier metrics, supported clause types, disclaimer.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core import analyze_upload, AnalysisResult
from src.report import build_html_report
from src.extraction import ml_classifier
from src.extraction.clause_extractor import CLAUSE_KEYWORDS
from src.storage import db
from src.search.semantic_search import SemanticSearchIndex

st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      .metric-card { background: var(--secondary-background-color); border-radius: 10px;
                     padding: 0.9rem 1.1rem; border: 1px solid rgba(128,128,128,0.15); }
      .finding-card { border: 1px solid rgba(128,128,128,0.18); border-left: 5px solid #ccc;
                      border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.6rem; }
      .sev-tag { color:#fff; font-size:0.7rem; font-weight:700; padding:0.1rem 0.5rem;
                 border-radius:4px; margin-right:0.5rem; }
      .evidence { background: rgba(128,128,128,0.08); border-radius:5px; padding:0.4rem 0.6rem;
                  font-size:0.8rem; font-family: ui-monospace, monospace; margin-top:0.4rem; }
      .app-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

_SEV_COLORS = {"high": "#c0392b", "medium": "#d68910", "low": "#7d8a99"}
_LEVEL_COLORS = {"High": "#c0392b", "Medium": "#d68910", "Low": "#1e8449"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _search_index(cache_key: int) -> SemanticSearchIndex:
    """Build (and cache) the clause search index. `cache_key` is the contract
    count, so the index rebuilds automatically when the library changes."""
    return SemanticSearchIndex.from_clauses(db.get_all_clauses())


def _risk_gauge(score: int, level: str) -> go.Figure:
    color = _LEVEL_COLORS.get(level, "#333")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 25], "color": "rgba(30,132,73,0.15)"},
                    {"range": [25, 55], "color": "rgba(214,137,16,0.15)"},
                    {"range": [55, 100], "color": "rgba(192,57,43,0.15)"},
                ],
            },
            title={"text": f"{level} risk"},
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _severity_tag(sev: str) -> str:
    return f"<span class='sev-tag' style='background:{_SEV_COLORS.get(sev, '#333')}'>{sev.upper()}</span>"


# ---------------------------------------------------------------------------
# Page: Analyze
# ---------------------------------------------------------------------------
def page_analyze(use_ml: bool) -> None:
    st.header("📄 Analyze a Contract")
    st.caption(
        "Upload a PDF or DOCX contract. Nothing is stored unless you choose "
        "**Save to library**. Automated analysis — not legal advice."
    )

    uploaded = st.file_uploader("Upload contract", type=["pdf", "docx"], key="uploader")
    if uploaded is None:
        st.info("Upload a contract to begin. Try `sample_contracts/sample_services_agreement.docx`.")
        return

    # Re-analyze only when the file changes.
    sig = (uploaded.name, uploaded.size, use_ml)
    if st.session_state.get("analysis_sig") != sig:
        with st.spinner("Analyzing contract…"):
            try:
                st.session_state["analysis"] = analyze_upload(uploaded, use_ml=use_ml)
                st.session_state["analysis_sig"] = sig
                st.session_state["saved_id"] = None
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't process this file: {e}")
                return

    result: AnalysisResult = st.session_state["analysis"]

    if result.meta.get("likely_scanned"):
        st.warning(
            "This PDF looks like a scanned image with little extractable text. "
            "There's no OCR step, so results will be sparse."
        )

    # ---- headline metrics ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File type", result.meta.get("file_type", "—"))
    c2.metric("Sections", len(result.clauses))
    c3.metric("Clauses classified", len(result.detected_clauses))
    c4.metric("Risk", f"{result.risk.level} · {result.risk.score}/100")

    tabs = st.tabs(
        ["📋 Overview", "⚠️ Risk", "📑 Clauses", "🏷️ Entities", "📄 Full Text", "⬇️ Export & Save"]
    )

    # Overview
    with tabs[0]:
        st.subheader("Plain-English overview")
        st.write(result.overview)
        st.subheader("Key points")
        st.write(result.key_points)
        if result.detected_clauses:
            df = pd.DataFrame(
                [{"Clause Type": c.clause_type} for c in result.detected_clauses]
            )
            counts = df["Clause Type"].value_counts().reset_index()
            counts.columns = ["Clause Type", "Count"]
            fig = px.bar(counts, x="Clause Type", y="Count", title="Detected clause types")
            fig.update_layout(xaxis_tickangle=-30, height=380)
            st.plotly_chart(fig, use_container_width=True)

    # Risk
    with tabs[1]:
        gcol, scol = st.columns([1, 2])
        with gcol:
            st.plotly_chart(_risk_gauge(result.risk.score, result.risk.level), use_container_width=True)
            counts = result.risk.counts
            st.caption(
                f"🔴 {counts.get('high',0)} high · 🟠 {counts.get('medium',0)} medium · "
                f"⚪ {counts.get('low',0)} low"
            )
        with scol:
            if not result.risk.findings:
                st.success("No risk findings were raised for this contract.")
            for f in result.risk.findings:
                evidence = f"<div class='evidence'>{f.evidence}</div>" if f.evidence else ""
                st.markdown(
                    f"<div class='finding-card' style='border-left-color:{_SEV_COLORS.get(f.severity)}'>"
                    f"{_severity_tag(f.severity)}<strong>{f.title}</strong>"
                    f"<div style='margin:.3rem 0'>{f.detail}</div>"
                    f"<div style='font-size:.88rem'><em>Recommendation:</em> {f.recommendation}</div>"
                    f"{evidence}</div>",
                    unsafe_allow_html=True,
                )

    # Clauses
    with tabs[2]:
        st.caption("Every detected section in document order. Expand to read the text.")
        for c in result.clauses:
            bits = []
            if c.section_number:
                bits.append(f"§{c.section_number}")
            bits.append(c.heading or "(no heading)")
            if c.clause_type:
                bits.append(f"→ {c.clause_type} ({c.confidence:.0%}, {c.method})")
            else:
                bits.append("→ unclassified")
            with st.expander(" ".join(bits)):
                st.write(c.text or "_(empty section body)_")
                if c.matched_keywords:
                    st.caption("Signals: " + ", ".join(c.matched_keywords))

    # Entities
    with tabs[3]:
        e = result.entities
        left, right = st.columns(2)
        with left:
            st.markdown("**Parties**")
            st.write(", ".join(e.parties) if e.parties else "—")
            st.markdown("**Effective date**")
            st.write(e.effective_date or "—")
            st.markdown("**Term**")
            st.write(e.term or "—")
            st.markdown("**Governing law**")
            st.write(e.governing_law or "—")
        with right:
            st.markdown("**Monetary amounts**")
            st.write(", ".join(e.monetary_amounts) if e.monetary_amounts else "—")
            st.markdown("**Notice periods**")
            st.write(", ".join(e.notice_periods) if e.notice_periods else "—")
            st.markdown("**Dates mentioned**")
            st.write(", ".join(e.dates) if e.dates else "—")

    # Full text
    with tabs[4]:
        st.text_area("Cleaned extracted text", result.clean_text, height=460)

    # Export & save
    with tabs[5]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "⬇️ Download JSON",
                data=json.dumps(result.to_export_dict(), indent=2),
                file_name=f"{result.filename}_analysis.json",
                mime="application/json",
                use_container_width=True,
            )
            st.download_button(
                "⬇️ Download HTML report",
                data=build_html_report(result),
                file_name=f"{result.filename}_report.html",
                mime="text/html",
                use_container_width=True,
            )
        with col_b:
            if st.session_state.get("saved_id"):
                st.success(f"Saved to library (contract #{st.session_state['saved_id']}).")
            if st.button("💾 Save to library", use_container_width=True, type="primary"):
                cid = db.save_contract(
                    result.filename,
                    result.clauses,
                    risk_score=result.risk.score,
                    risk_level=result.risk.level,
                    summary=result.overview,
                    entities=result.entities.to_dict(),
                    full_text=result.clean_text,
                )
                st.session_state["saved_id"] = cid
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Library
# ---------------------------------------------------------------------------
def page_library() -> None:
    st.header("📚 Contract Library")
    contracts = db.list_contracts()
    if not contracts:
        st.info("No contracts saved yet. Analyze a contract and click **Save to library**.")
        return

    st.caption(f"{len(contracts)} contract(s) stored.")
    df = pd.DataFrame(contracts)
    df_display = df[["id", "filename", "upload_date", "num_clauses", "risk_level", "risk_score"]]
    st.dataframe(df_display, use_container_width=True, hide_index=True)

    ids = [c["id"] for c in contracts]
    selected = st.selectbox(
        "View a contract",
        ids,
        format_func=lambda i: next(c["filename"] for c in contracts if c["id"] == i),
    )
    contract = db.get_contract(selected)
    if not contract:
        return

    top = st.columns([3, 1])
    with top[0]:
        st.subheader(contract["filename"])
        if contract.get("summary"):
            st.write(contract["summary"])
    with top[1]:
        st.metric("Risk", f"{contract.get('risk_level','—')} · {contract.get('risk_score','—')}")
        if st.button("🗑️ Delete", use_container_width=True):
            db.delete_contract(selected)
            st.rerun()

    st.markdown("**Clauses**")
    clause_df = pd.DataFrame(contract["clauses"])
    if not clause_df.empty:
        show = clause_df[["section_number", "heading", "clause_type", "confidence", "method"]]
        st.dataframe(show, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Semantic Search
# ---------------------------------------------------------------------------
def page_search() -> None:
    st.header("🔎 Semantic Clause Search")
    st.caption(
        "Search for similar clauses across every contract in your library "
        "using TF-IDF cosine similarity."
    )
    count = db.count_contracts()
    if count == 0:
        st.info("Your library is empty. Analyze and save some contracts first.")
        return

    index = _search_index(count)
    if index.is_empty:
        st.info("No clauses available to search yet.")
        return

    st.caption(f"Index covers {index.size} clauses across {count} contract(s).")
    query = st.text_input("Search clauses", placeholder="e.g. limitation of liability, termination for convenience…")
    top_k = st.slider("Results to show", 1, 20, 5)

    examples = ["limitation of liability", "confidential information", "governing law jurisdiction", "termination notice"]
    ex_cols = st.columns(len(examples))
    for col, ex in zip(ex_cols, examples):
        if col.button(ex, use_container_width=True):
            query = ex

    if not query:
        return

    results = index.search(query, top_k=top_k)
    if not results:
        st.warning("No matching clauses found. Try different wording.")
        return

    for r in results:
        with st.container():
            st.markdown(
                f"**{r.heading or '(no heading)'}** · `{r.clause_type or 'unclassified'}` · "
                f"_{r.filename}_ · similarity **{r.score:.2f}**"
            )
            snippet = r.text if len(r.text) < 400 else r.text[:400] + "…"
            st.write(snippet)
            st.divider()


# ---------------------------------------------------------------------------
# Page: Model & About
# ---------------------------------------------------------------------------
def page_about() -> None:
    st.header("ℹ️ Model & About")

    metrics = ml_classifier.model_metrics()
    if metrics:
        st.subheader("Clause classifier performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
        m2.metric("Macro-F1", f"{metrics['test_macro_f1']:.2f}")
        m3.metric("Classes", metrics["n_classes"])
        m4.metric("Baseline acc.", f"{metrics['baseline_accuracy']:.1%}")
        st.caption(
            f"Model: {metrics.get('selected_model','?')} · trained on {metrics.get('n_train','?')} "
            f"CUAD clause spans · evaluated on {metrics.get('n_test','?')} held-out spans. "
            f"The lift over the most-frequent baseline is "
            f"+{metrics['test_accuracy'] - metrics['baseline_accuracy']:.0%} accuracy."
        )
        with st.expander("Per-class F1 scores"):
            pcf = pd.DataFrame(
                sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1], reverse=True),
                columns=["Clause type", "F1"],
            )
            st.dataframe(pcf, use_container_width=True, hide_index=True)
    else:
        st.warning(
            "The trained model isn't loaded. The app falls back to the keyword "
            "baseline. Run `python training/train_classifier.py` to enable ML."
        )

    st.subheader("How it works")
    st.markdown(
        """
        1. **Ingestion** — PDF (`pdfplumber`) / DOCX (`python-docx`) → clean text
        2. **Clause extraction** — heading segmentation + a scikit-learn TF-IDF
           classifier (41 CUAD clause types), with a keyword baseline fallback
        3. **Entity extraction** — rule-based parties, dates, money, governing law, notice, term
        4. **Risk analysis** — high-risk clauses + missing protective clauses + red-flag language → 0–100 score
        5. **Summaries** — a fact-based overview and an extractive key-points summary
        6. **Library & search** — SQLite persistence + TF-IDF similarity search
        """
    )

    st.subheader("Keyword-baseline clause types")
    st.write(", ".join(sorted(CLAUSE_KEYWORDS.keys())))

    st.info(
        "⚖️ This tool provides automated, informational analysis only. It is not "
        "legal advice and does not replace review by a qualified attorney."
    )


# ---------------------------------------------------------------------------
# Sidebar / router
# ---------------------------------------------------------------------------
def main() -> None:
    with st.sidebar:
        st.markdown("<div class='app-title'>⚖️ Legal Contract Analyzer</div>", unsafe_allow_html=True)
        st.caption("AI-assisted contract review")
        st.divider()

        page = st.radio(
            "Navigate",
            ["📄 Analyze Contract", "📚 Contract Library", "🔎 Semantic Search", "ℹ️ Model & About"],
            label_visibility="collapsed",
        )

        st.divider()
        ml_ok = ml_classifier.is_available()
        use_ml = st.toggle("Use ML classifier", value=ml_ok, disabled=not ml_ok,
                           help="Falls back to the keyword baseline if off or unavailable.")
        if ml_ok:
            st.caption("🟢 ML model loaded")
        else:
            st.caption("🟡 Keyword baseline (no model)")
        st.caption(f"📚 {db.count_contracts()} contracts in library")

    if page.startswith("📄"):
        page_analyze(use_ml)
    elif page.startswith("📚"):
        page_library()
    elif page.startswith("🔎"):
        page_search()
    else:
        page_about()


if __name__ == "__main__":
    main()
