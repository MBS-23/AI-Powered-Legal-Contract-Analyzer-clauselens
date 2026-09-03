"""
app.py — Legal Contract Analyzer dashboard.

A professional multi-page Streamlit UI over the analysis pipeline in src/.
All business logic lives in src/ (core.py orchestrates); this file is the
presentation layer: theming, layout, charts, and state.

Security notes (see SECURITY.md):
  - Every dynamic value placed inside an `unsafe_allow_html` block is escaped
    with security.safe(); raw contract text is never rendered as HTML.
  - Uploads pass security.validate_upload() (size/extension/magic bytes) before
    any parser touches them.

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
from src.security import validate_upload, safe, clean_display, clean_query, MAX_UPLOAD_BYTES
from src.content.education import FLASHCARDS, ARTICLES, flashcard_categories

st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
NAVY = "#0f2a4a"
NAVY_2 = "#1c4a7e"
GOLD = "#c9a227"
TEAL = "#2a9d8f"
SLATE = "#5b6b7d"
SEV_COLORS = {"high": "#c0392b", "medium": "#d68910", "low": "#7d8a99"}
LEVEL_COLORS = {"High": "#c0392b", "Medium": "#d68910", "Low": "#1e8449"}
CATEGORICAL = [NAVY_2, GOLD, TEAL, "#8e44ad", "#c0392b", "#16a085", "#d68910", "#2c3e50"]

# ---------------------------------------------------------------------------
# Theme / CSS  (plain string — NOT an f-string — because of the CSS braces)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700;800&display=swap');

:root {
  --navy:#0f2a4a; --navy-2:#1c4a7e; --gold:#c9a227; --gold-2:#e6c65c;
  --bg:#eef2f7; --card:#ffffff; --ink:#1f2933; --muted:#5b6b7d;
}

/* Base */
html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
[data-testid="stAppViewContainer"] { background:
   radial-gradient(1200px 500px at 100% -10%, rgba(28,74,126,.06), transparent),
   var(--bg); }
.block-container { padding-top:1.4rem; padding-bottom:3rem; max-width:1250px;
   animation: fadeInUp .5s ease both; }
h1,h2,h3 { font-family:'Playfair Display',Georgia,serif; color:var(--navy);
   letter-spacing:.2px; }
a { color:var(--navy-2); }

@keyframes fadeInUp { from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;} }
@keyframes shimmer { 0%{background-position:-400px 0;} 100%{background-position:400px 0;} }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(192,57,43,.45);} 70%{box-shadow:0 0 0 12px rgba(192,57,43,0);} 100%{box-shadow:0 0 0 0 rgba(192,57,43,0);} }
@keyframes floatIn { from{opacity:0; transform:translateY(20px) scale(.98);} to{opacity:1; transform:none;} }

/* Hero */
.hero { position:relative; overflow:hidden; border-radius:18px; padding:1.7rem 2rem;
  background:linear-gradient(125deg, var(--navy) 0%, var(--navy-2) 100%);
  color:#fff; box-shadow:0 14px 40px rgba(15,42,74,.28); margin-bottom:1.3rem;
  animation:floatIn .6s ease both; }
.hero:after { content:""; position:absolute; right:-60px; top:-60px; width:220px; height:220px;
  border-radius:50%; background:radial-gradient(circle, rgba(201,162,39,.35), transparent 70%); }
.hero:before { content:""; position:absolute; left:-40px; bottom:-80px; width:200px; height:200px;
  border-radius:50%; background:radial-gradient(circle, rgba(255,255,255,.06), transparent 70%); }
.hero h1 { color:#fff; margin:0 0 .35rem 0; font-size:2rem; }
.hero p { color:rgba(255,255,255,.85); margin:0; font-size:1rem; max-width:760px; }
.hero .accent { width:64px; height:4px; border-radius:3px; background:var(--gold); margin:.7rem 0; }
.hero .eyebrow { text-transform:uppercase; letter-spacing:3px; font-size:.72rem;
  color:var(--gold-2); font-weight:700; }

/* Cards */
.lca-card { background:var(--card); border:1px solid rgba(15,42,74,.08); border-radius:14px;
  padding:1.1rem 1.2rem; box-shadow:0 6px 18px rgba(15,42,74,.06);
  transition:transform .2s ease, box-shadow .2s ease; animation:floatIn .5s ease both; }
.lca-card:hover { transform:translateY(-3px); box-shadow:0 14px 30px rgba(15,42,74,.12); }
.feature-icon { font-size:1.7rem; }

/* Metrics -> cards */
[data-testid="stMetric"] { background:var(--card); border:1px solid rgba(15,42,74,.08);
  border-radius:14px; padding:1rem 1.1rem; box-shadow:0 6px 18px rgba(15,42,74,.06);
  transition:transform .2s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-3px); }
[data-testid="stMetricValue"] { color:var(--navy); font-weight:700; }

/* Findings */
.finding-card { background:var(--card); border:1px solid rgba(15,42,74,.08); border-left:5px solid #ccc;
  border-radius:12px; padding:.8rem 1.05rem; margin-bottom:.7rem; box-shadow:0 4px 14px rgba(15,42,74,.05);
  transition:transform .18s ease; animation:floatIn .45s ease both; }
.finding-card:hover { transform:translateX(3px); }
.sev-tag { color:#fff; font-size:.68rem; font-weight:800; letter-spacing:.5px; padding:.12rem .5rem;
  border-radius:5px; margin-right:.5rem; vertical-align:middle; }
.sev-high-pulse { animation:pulse 2s infinite; }
.evidence { background:rgba(15,42,74,.05); border-radius:6px; padding:.45rem .65rem; font-size:.8rem;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:#42505f; margin-top:.45rem;
  border-left:3px solid var(--gold); overflow-wrap:anywhere; }

/* Pills / badges */
.pill { display:inline-block; padding:.28rem .7rem; border-radius:999px; font-weight:700;
  font-size:.8rem; color:#fff; }
.tag { display:inline-block; padding:.16rem .55rem; border-radius:7px; font-size:.74rem;
  background:rgba(28,74,126,.1); color:var(--navy-2); margin:.12rem; font-weight:600; }

/* Sidebar */
section[data-testid="stSidebar"] { background:linear-gradient(185deg, var(--navy) 0%, #0a1f38 100%);
  border-right:1px solid rgba(201,162,39,.25); }
section[data-testid="stSidebar"] * { color:#dfe7f1; }
section[data-testid="stSidebar"] .app-title { font-family:'Playfair Display',serif; color:#fff;
  font-size:1.2rem; font-weight:800; }
section[data-testid="stSidebar"] .app-sub { color:var(--gold-2); font-size:.75rem;
  letter-spacing:2px; text-transform:uppercase; }
section[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.12); }
/* nav radio as menu */
section[data-testid="stSidebar"] [role="radiogroup"] label { padding:.5rem .7rem; border-radius:10px;
  margin:.12rem 0; transition:background .2s ease, transform .15s ease, border-left-color .2s;
  border-left:3px solid transparent; cursor:pointer; }
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(255,255,255,.07);
  transform:translateX(3px); }
section[data-testid="stSidebar"] [role="radiogroup"] label p { font-weight:600; font-size:.95rem; }
.side-stat { background:rgba(255,255,255,.06); border:1px solid rgba(201,162,39,.2);
  border-radius:10px; padding:.5rem .7rem; font-size:.82rem; margin-top:.4rem; }

/* Buttons */
.stButton>button, .stDownloadButton>button { border-radius:10px; font-weight:600;
  border:1px solid rgba(15,42,74,.15); transition:all .18s ease; }
.stButton>button:hover, .stDownloadButton>button:hover { transform:translateY(-2px);
  border-color:var(--gold); box-shadow:0 6px 16px rgba(15,42,74,.14); }
.stButton>button[kind="primary"] { background:var(--navy); border-color:var(--navy); }
.stButton>button[kind="primary"]:hover { background:var(--navy-2); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:.3rem; }
.stTabs [data-baseweb="tab"] { border-radius:10px 10px 0 0; padding:.35rem .8rem; }
.stTabs [aria-selected="true"] { background:rgba(28,74,126,.1); color:var(--navy); }

/* Flashcards */
.flip-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(255px,1fr)); gap:1rem; }
.flip-card { perspective:1200px; height:200px; }
.flip-inner { position:relative; width:100%; height:100%; transition:transform .7s cubic-bezier(.4,.2,.2,1);
  transform-style:preserve-3d; }
.flip-card:hover .flip-inner { transform:rotateY(180deg); }
.flip-front, .flip-back { position:absolute; inset:0; backface-visibility:hidden;
  border-radius:16px; padding:1.1rem; display:flex; flex-direction:column; justify-content:center;
  box-shadow:0 8px 22px rgba(15,42,74,.14); }
.flip-front { background:linear-gradient(135deg,var(--navy),var(--navy-2)); color:#fff; }
.flip-front .q { font-family:'Playfair Display',serif; font-size:1.05rem; font-weight:700; }
.flip-front .cat { position:absolute; top:.7rem; right:.8rem; font-size:.65rem; color:var(--gold-2);
  text-transform:uppercase; letter-spacing:1.5px; font-weight:700; }
.flip-front .hint { position:absolute; bottom:.7rem; left:1.1rem; font-size:.7rem; color:rgba(255,255,255,.6);}
.flip-back { background:var(--card); color:var(--ink); transform:rotateY(180deg);
  border:1px solid var(--gold); font-size:.9rem; line-height:1.45; overflow:auto; }

/* Article cards */
.article { background:var(--card); border:1px solid rgba(15,42,74,.08); border-radius:14px;
  padding:1.1rem 1.2rem; box-shadow:0 6px 18px rgba(15,42,74,.06); height:100%;
  transition:transform .2s ease, box-shadow .2s ease; animation:floatIn .5s ease both; }
.article:hover { transform:translateY(-3px); box-shadow:0 14px 30px rgba(15,42,74,.12); }
.article .ico { font-size:1.8rem; }
.article h4 { font-family:'Playfair Display',serif; color:var(--navy); margin:.4rem 0 .3rem; }
.article .meta { color:var(--muted); font-size:.78rem; }

/* Scrollbar */
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-thumb { background:rgba(15,42,74,.28); border-radius:6px; }
::-webkit-scrollbar-thumb:hover { background:rgba(15,42,74,.45); }

/* Security badge grid */
.sec-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:.8rem; }
.sec-item { background:var(--card); border:1px solid rgba(15,42,74,.08); border-left:4px solid var(--teal, #2a9d8f);
  border-radius:12px; padding:.8rem 1rem; box-shadow:0 4px 14px rgba(15,42,74,.05);
  transition:transform .18s ease; animation:floatIn .45s ease both; }
.sec-item:hover { transform:translateY(-3px); }
.sec-item .h { font-weight:700; color:var(--navy); font-size:.92rem; }
.sec-item .d { color:var(--muted); font-size:.82rem; margin-top:.2rem; }
.sec-check { color:#1e8449; font-weight:800; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small render helpers
# ---------------------------------------------------------------------------
def hero(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{safe(icon)} &nbsp;Legal Contract Intelligence</div>
          <h1>{safe(title)}</h1>
          <div class="accent"></div>
          <p>{safe(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _theme_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#1f2933"),
        title_font=dict(family="Playfair Display, serif", color=NAVY, size=16),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


def risk_gauge(score: int, level: str) -> go.Figure:
    color = LEVEL_COLORS.get(level, NAVY)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 30, "color": NAVY}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": SLATE},
                "bar": {"color": color, "thickness": 0.3},
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(30,132,73,.18)"},
                    {"range": [25, 55], "color": "rgba(214,137,16,.18)"},
                    {"range": [55, 100], "color": "rgba(192,57,43,.18)"},
                ],
                "threshold": {"line": {"color": color, "width": 4}, "value": score},
            },
            title={"text": f"<b>{level} risk</b>"},
        )
    )
    return _theme_fig(fig, height=250)


def clause_bar(detected) -> go.Figure | None:
    if not detected:
        return None
    counts = pd.Series([c.clause_type for c in detected]).value_counts().reset_index()
    counts.columns = ["Clause Type", "Count"]
    fig = px.bar(counts, x="Clause Type", y="Count", title="Detected clause types",
                 color_discrete_sequence=[NAVY_2])
    fig.update_layout(xaxis_tickangle=-30)
    return _theme_fig(fig, 360)


def severity_donut(counts: dict) -> go.Figure | None:
    data = {k.title(): v for k, v in counts.items() if v}
    if not data:
        return None
    fig = px.pie(values=list(data.values()), names=list(data.keys()), hole=0.55,
                 title="Risk findings by severity",
                 color=list(data.keys()),
                 color_discrete_map={"High": SEV_COLORS["high"], "Medium": SEV_COLORS["medium"], "Low": SEV_COLORS["low"]})
    fig.update_traces(textinfo="value")
    return _theme_fig(fig, 300)


def method_donut(clauses) -> go.Figure | None:
    labels = [("ML model" if c.method == "ml" else "Keyword") for c in clauses if c.clause_type]
    if not labels:
        return None
    s = pd.Series(labels).value_counts()
    fig = px.pie(values=s.values, names=s.index, hole=0.55, title="Classifier used",
                 color_discrete_sequence=[NAVY_2, GOLD])
    return _theme_fig(fig, 300)


# ---------------------------------------------------------------------------
# Cached search index
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _search_index(cache_key: int) -> SemanticSearchIndex:
    return SemanticSearchIndex.from_clauses(db.get_all_clauses())


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    hero("🏛️", "Legal Contract Analyzer",
         "AI-assisted contract review that reads, classifies, and risk-scores your "
         "agreements — clause extraction, entity detection, risk analysis, and "
         "semantic search, running entirely on a private, open-source pipeline.")

    contracts = db.list_contracts()
    if not contracts:
        st.markdown("#### Get started")
        cols = st.columns(3)
        feats = [
            ("📄", "Analyze a contract", "Upload a PDF or DOCX and get clauses, entities, a risk report and summaries."),
            ("🔎", "Search your library", "Find similar clauses across every stored contract with semantic search."),
            ("🎓", "Learn the concepts", "Flashcards and articles on the clauses that matter and how to read risk."),
        ]
        for col, (ic, t, d) in zip(cols, feats):
            col.markdown(
                f"<div class='lca-card'><div class='feature-icon'>{safe(ic)}</div>"
                f"<h3 style='margin:.3rem 0'>{safe(t)}</h3>"
                f"<div style='color:var(--muted)'>{safe(d)}</div></div>",
                unsafe_allow_html=True,
            )
        st.info("Open **Analyze Contract** in the sidebar and upload the sample in "
                "`sample_contracts/` to see the full pipeline in action.")
        return

    # ---- Library analytics ----
    df = pd.DataFrame(contracts)
    total = len(df)
    avg_risk = int(df["risk_score"].dropna().mean()) if df["risk_score"].notna().any() else 0
    high = int((df["risk_level"] == "High").sum())
    total_clauses = int(df["num_clauses"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Contracts analyzed", total)
    c2.metric("Avg. risk score", f"{avg_risk}/100")
    c3.metric("High-risk contracts", high)
    c4.metric("Clauses indexed", total_clauses)

    left, right = st.columns(2)
    with left:
        lvl = df["risk_level"].value_counts()
        if not lvl.empty:
            fig = px.pie(values=lvl.values, names=lvl.index, hole=0.55,
                         title="Portfolio risk distribution",
                         color=lvl.index,
                         color_discrete_map=LEVEL_COLORS)
            st.plotly_chart(_theme_fig(fig, 320), use_container_width=True)
    with right:
        all_clauses = db.get_all_clauses()
        types = [c["clause_type"] for c in all_clauses if c.get("clause_type")]
        if types:
            s = pd.Series(types).value_counts().head(10).reset_index()
            s.columns = ["Clause Type", "Count"]
            fig = px.bar(s, x="Count", y="Clause Type", orientation="h",
                         title="Most common clauses across library",
                         color_discrete_sequence=[GOLD])
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(_theme_fig(fig, 320), use_container_width=True)

    st.markdown("#### Recent contracts")
    st.dataframe(
        df[["id", "filename", "upload_date", "num_clauses", "risk_level", "risk_score"]].head(8),
        use_container_width=True, hide_index=True,
    )


# ---------------------------------------------------------------------------
# Page: Analyze
# ---------------------------------------------------------------------------
def page_analyze(use_ml: bool) -> None:
    hero("📄", "Analyze a Contract",
         "Upload a PDF or DOCX. Nothing is stored unless you choose Save to library. "
         "Automated analysis — not legal advice.")

    uploaded = st.file_uploader(
        f"Upload contract (PDF or DOCX, max {MAX_UPLOAD_BYTES // (1024*1024)} MB)",
        type=["pdf", "docx"], key="uploader",
    )
    if uploaded is None:
        st.info("Upload a contract to begin. Try `sample_contracts/sample_services_agreement.docx`.")
        return

    # ---- security gate: validate before parsing ----
    data = uploaded.getvalue()
    check = validate_upload(uploaded.name, data)
    if not check.ok:
        st.error(f"🛡️ Upload rejected: {check.reason}")
        return

    sig = (uploaded.name, uploaded.size, use_ml)
    if st.session_state.get("analysis_sig") != sig:
        with st.spinner("Analyzing contract…"):
            try:
                st.session_state["analysis"] = analyze_upload(uploaded, use_ml=use_ml)
                st.session_state["analysis_sig"] = sig
                st.session_state["saved_id"] = None
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't process this file: {safe(e)}")
                return

    result: AnalysisResult = st.session_state["analysis"]

    if result.meta.get("likely_scanned"):
        st.warning("This PDF looks like a scanned image with little extractable text. "
                   "There's no OCR step, so results will be sparse.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File type", result.meta.get("file_type", "—"))
    c2.metric("Sections", len(result.clauses))
    c3.metric("Clauses classified", len(result.detected_clauses))
    c4.metric("Risk", f"{result.risk.level} · {result.risk.score}/100")

    tabs = st.tabs(["📋 Overview", "⚠️ Risk", "📑 Clauses", "🏷️ Entities", "📄 Full Text", "⬇️ Export & Save"])

    with tabs[0]:
        st.markdown("#### Plain-English overview")
        st.write(result.overview)
        st.markdown("#### Key points")
        st.write(result.key_points)
        col1, col2 = st.columns([3, 2])
        fig = clause_bar(result.detected_clauses)
        if fig:
            col1.plotly_chart(fig, use_container_width=True)
        mfig = method_donut(result.clauses)
        if mfig:
            col2.plotly_chart(mfig, use_container_width=True)

    with tabs[1]:
        gcol, scol = st.columns([1, 2])
        with gcol:
            st.plotly_chart(risk_gauge(result.risk.score, result.risk.level), use_container_width=True)
            dfig = severity_donut(result.risk.counts)
            if dfig:
                st.plotly_chart(dfig, use_container_width=True)
        with scol:
            if not result.risk.findings:
                st.success("No risk findings were raised for this contract.")
            for f in result.risk.findings:
                pulse = " sev-high-pulse" if f.severity == "high" else ""
                evidence = f"<div class='evidence'>{clean_display(f.evidence, 260)}</div>" if f.evidence else ""
                st.markdown(
                    f"<div class='finding-card{pulse}' style='border-left-color:{SEV_COLORS.get(f.severity)}'>"
                    f"<span class='sev-tag' style='background:{SEV_COLORS.get(f.severity)}'>{safe(f.severity.upper())}</span>"
                    f"<strong>{safe(f.title)}</strong>"
                    f"<div style='margin:.3rem 0; color:var(--ink)'>{safe(f.detail)}</div>"
                    f"<div style='font-size:.88rem; color:var(--muted)'><em>Recommendation:</em> {safe(f.recommendation)}</div>"
                    f"{evidence}</div>",
                    unsafe_allow_html=True,
                )

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

    with tabs[3]:
        e = result.entities
        left, right = st.columns(2)
        with left:
            st.markdown("**Parties**"); st.write(", ".join(e.parties) if e.parties else "—")
            st.markdown("**Effective date**"); st.write(e.effective_date or "—")
            st.markdown("**Term**"); st.write(e.term or "—")
            st.markdown("**Governing law**"); st.write(e.governing_law or "—")
        with right:
            st.markdown("**Monetary amounts**"); st.write(", ".join(e.monetary_amounts) if e.monetary_amounts else "—")
            st.markdown("**Notice periods**"); st.write(", ".join(e.notice_periods) if e.notice_periods else "—")
            st.markdown("**Dates mentioned**"); st.write(", ".join(e.dates) if e.dates else "—")

    with tabs[4]:
        st.text_area("Cleaned extracted text", result.clean_text, height=460)

    with tabs[5]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("⬇️ Download JSON", data=json.dumps(result.to_export_dict(), indent=2),
                               file_name=f"{result.filename}_analysis.json", mime="application/json",
                               use_container_width=True)
            st.download_button("⬇️ Download HTML report", data=build_html_report(result),
                               file_name=f"{result.filename}_report.html", mime="text/html",
                               use_container_width=True)
        with col_b:
            if st.session_state.get("saved_id"):
                st.success(f"Saved to library (contract #{st.session_state['saved_id']}).")
            if st.button("💾 Save to library", use_container_width=True, type="primary"):
                cid = db.save_contract(
                    result.filename, result.clauses,
                    risk_score=result.risk.score, risk_level=result.risk.level,
                    summary=result.overview, entities=result.entities.to_dict(),
                    full_text=result.clean_text,
                )
                st.session_state["saved_id"] = cid
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Library
# ---------------------------------------------------------------------------
def page_library() -> None:
    hero("📚", "Contract Library", "Every contract you've saved, with its clauses and risk profile.")
    contracts = db.list_contracts()
    if not contracts:
        st.info("No contracts saved yet. Analyze a contract and click **Save to library**.")
        return

    df = pd.DataFrame(contracts)
    st.dataframe(df[["id", "filename", "upload_date", "num_clauses", "risk_level", "risk_score"]],
                 use_container_width=True, hide_index=True)

    ids = [c["id"] for c in contracts]
    selected = st.selectbox("View a contract", ids,
                            format_func=lambda i: next(c["filename"] for c in contracts if c["id"] == i))
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
        st.dataframe(clause_df[["section_number", "heading", "clause_type", "confidence", "method"]],
                     use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Search
# ---------------------------------------------------------------------------
def page_search() -> None:
    hero("🔎", "Semantic Clause Search",
         "Find similar clauses across every contract in your library using TF-IDF cosine similarity.")
    count = db.count_contracts()
    if count == 0:
        st.info("Your library is empty. Analyze and save some contracts first.")
        return

    index = _search_index(count)
    if index.is_empty:
        st.info("No clauses available to search yet.")
        return

    st.caption(f"Index covers {index.size} clauses across {count} contract(s).")
    raw_query = st.text_input("Search clauses",
                              placeholder="e.g. limitation of liability, termination for convenience…")
    top_k = st.slider("Results to show", 1, 20, 5)

    examples = ["limitation of liability", "confidential information", "governing law jurisdiction", "termination notice"]
    ex_cols = st.columns(len(examples))
    for col, ex in zip(ex_cols, examples):
        if col.button(ex, use_container_width=True):
            raw_query = ex

    query = clean_query(raw_query)
    if not query:
        return

    for r in index.search(query, top_k=top_k):
        st.markdown(
            f"<div class='lca-card' style='margin-bottom:.7rem'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<div><strong>{safe(r.heading or '(no heading)')}</strong> "
            f"<span class='tag'>{safe(r.clause_type or 'unclassified')}</span></div>"
            f"<span class='pill' style='background:{NAVY_2}'>sim {r.score:.2f}</span></div>"
            f"<div style='color:var(--muted); font-size:.8rem; margin:.25rem 0'>📄 {safe(r.filename)}</div>"
            f"<div style='color:var(--ink)'>{clean_display(r.text, 400)}</div></div>",
            unsafe_allow_html=True,
        )
    if not index.search(query, top_k=top_k):
        st.warning("No matching clauses found. Try different wording.")


# ---------------------------------------------------------------------------
# Page: Learn (flashcards + articles)
# ---------------------------------------------------------------------------
def page_learn() -> None:
    hero("🎓", "Learn — Contracts 101",
         "Interactive flashcards and short articles on the clauses that matter and how to read risk.")

    tab_cards, tab_articles = st.tabs(["🃏 Flashcards", "📰 Articles"])

    with tab_cards:
        cats = ["All"] + flashcard_categories()
        chosen = st.selectbox("Filter by topic", cats)
        cards = [c for c in FLASHCARDS if chosen == "All" or c.category == chosen]
        st.caption(f"{len(cards)} card(s). Hover a card to reveal the answer.")
        html = "<div class='flip-grid'>"
        for c in cards:
            html += (
                "<div class='flip-card'><div class='flip-inner'>"
                f"<div class='flip-front'><div class='cat'>{safe(c.category)}</div>"
                f"<div class='q'>{safe(c.front)}</div>"
                "<div class='hint'>hover to flip ↻</div></div>"
                f"<div class='flip-back'>{safe(c.back)}</div>"
                "</div></div>"
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

    with tab_articles:
        cols = st.columns(2)
        for i, art in enumerate(ARTICLES):
            with cols[i % 2]:
                st.markdown(
                    f"<div class='article'><div class='ico'>{safe(art.icon)}</div>"
                    f"<h4>{safe(art.title)}</h4>"
                    f"<div class='meta'>⏱ {art.read_minutes} min read</div>"
                    f"<p style='color:var(--ink); margin-top:.5rem'>{safe(art.summary)}</p></div>",
                    unsafe_allow_html=True,
                )
                with st.expander("Read article"):
                    st.markdown(art.body)


# ---------------------------------------------------------------------------
# Page: Security & About
# ---------------------------------------------------------------------------
def page_security_about() -> None:
    hero("🛡️", "Security & About",
         "The classifier's performance, how the pipeline works, and the app's security posture.")

    metrics = ml_classifier.model_metrics()
    if metrics:
        st.markdown("#### Clause classifier performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
        m2.metric("Macro-F1", f"{metrics['test_macro_f1']:.2f}")
        m3.metric("Classes", metrics["n_classes"])
        m4.metric("Baseline acc.", f"{metrics['baseline_accuracy']:.1%}")
        with st.expander("Per-class F1 scores"):
            pcf = pd.DataFrame(sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1], reverse=True),
                               columns=["Clause type", "F1"])
            st.dataframe(pcf, use_container_width=True, hide_index=True)

    st.markdown("#### 🛡️ Security posture")
    st.caption("Defensive controls mapped to OWASP Web Top 10 and OWASP LLM/AI Top 10.")
    controls = [
        ("XSS / HTML injection", "All contract-derived text is HTML-escaped before rendering; no raw HTML from documents."),
        ("SQL injection", "Every database query is parameterized (no string-built SQL)."),
        ("Malicious file upload", "Uploads validated by size, extension, and magic bytes before parsing."),
        ("Prompt injection / LLM", "No LLM in the inference path — the model returns a label only, so there is no prompt surface."),
        ("Injection (input)", "Untrusted text is Unicode-normalized, control-char-stripped, and length-capped."),
        ("DoS via large input", "Upload size and processed-text length are hard-capped."),
        ("Data exposure", "Runs locally; nothing is sent to third-party services. Contracts persist only if you save them."),
        ("CSRF", "Streamlit's built-in XSRF protection is enabled."),
    ]
    grid = "<div class='sec-grid'>"
    for title, desc in controls:
        grid += (f"<div class='sec-item'><div class='h'><span class='sec-check'>✔</span> {safe(title)}</div>"
                 f"<div class='d'>{safe(desc)}</div></div>")
    grid += "</div>"
    st.markdown(grid, unsafe_allow_html=True)

    st.markdown("#### How it works")
    st.markdown(
        """
        1. **Ingestion** — PDF (`pdfplumber`) / DOCX (`python-docx`) → clean text
        2. **Clause extraction** — heading segmentation + a scikit-learn TF-IDF classifier
           (41 CUAD clause types), with a keyword baseline fallback
        3. **Entity extraction** — rule-based parties, dates, money, governing law, notice, term
        4. **Risk analysis** — high-risk clauses + missing protective clauses + red-flag language → 0–100 score
        5. **Summaries** — a fact-based overview and an extractive key-points summary
        6. **Library & search** — SQLite persistence + TF-IDF similarity search
        """
    )
    st.markdown("#### Detected clause types (keyword baseline)")
    st.markdown("".join(f"<span class='tag'>{safe(k)}</span>" for k in sorted(CLAUSE_KEYWORDS)),
                unsafe_allow_html=True)

    st.info("⚖️ This tool provides automated, informational analysis only. It is not legal "
            "advice and does not replace review by a qualified attorney.")


# ---------------------------------------------------------------------------
# Sidebar / router
# ---------------------------------------------------------------------------
PAGES = {
    "🏛️ Dashboard": "dashboard",
    "📄 Analyze Contract": "analyze",
    "📚 Contract Library": "library",
    "🔎 Semantic Search": "search",
    "🎓 Learn": "learn",
    "🛡️ Security & About": "about",
}


def main() -> None:
    with st.sidebar:
        st.markdown("<div class='app-title'>⚖️ Legal Contract Analyzer</div>", unsafe_allow_html=True)
        st.markdown("<div class='app-sub'>AI-assisted review</div>", unsafe_allow_html=True)
        st.divider()

        page_label = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")

        st.divider()
        ml_ok = ml_classifier.is_available()
        use_ml = st.toggle("Use ML classifier", value=ml_ok, disabled=not ml_ok,
                           help="Falls back to the keyword baseline if off or unavailable.")
        status = "🟢 ML model loaded" if ml_ok else "🟡 Keyword baseline"
        st.markdown(f"<div class='side-stat'>{status}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='side-stat'>📚 {db.count_contracts()} contracts in library</div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='side-stat'>🛡️ Hardened · local-only</div>", unsafe_allow_html=True)

    page = PAGES[page_label]
    if page == "dashboard":
        page_dashboard()
    elif page == "analyze":
        page_analyze(use_ml)
    elif page == "library":
        page_library()
    elif page == "search":
        page_search()
    elif page == "learn":
        page_learn()
    else:
        page_security_about()


if __name__ == "__main__":
    main()
