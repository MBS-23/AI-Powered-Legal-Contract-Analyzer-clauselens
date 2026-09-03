"""
app.py — Legal Contract Analyzer dashboard.

Professional multi-page Streamlit UI over the analysis pipeline in src/.
Business logic lives in src/ (core.py orchestrates); this file is the
presentation layer: branding, theming, layout, charts, and state.

(Security hardening is applied throughout the code — output escaping, upload
validation, parameterized SQL — but kept behind the scenes.)

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
    page_title="Lexalytics · Legal Contract Analyzer",
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

# ---------------------------------------------------------------------------
# Brand mark (inline SVG — crisp scales-of-justice emblem)
# ---------------------------------------------------------------------------
LOGO_SVG = """
<svg viewBox="0 0 64 64" width="46" height="46" xmlns="http://www.w3.org/2000/svg" aria-label="logo">
  <defs>
    <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#1c4a7e"/><stop offset="1" stop-color="#0f2a4a"/>
    </linearGradient>
  </defs>
  <circle cx="32" cy="32" r="30" fill="url(#lg)" stroke="#c9a227" stroke-width="2.5"/>
  <g stroke="#e6c65c" stroke-width="2" fill="none" stroke-linecap="round">
    <line x1="32" y1="17" x2="32" y2="47"/>
    <line x1="17" y1="23" x2="47" y2="23"/>
    <circle cx="32" cy="16" r="2.6" fill="#e6c65c" stroke="none"/>
    <line x1="17" y1="23" x2="12" y2="35"/><line x1="17" y1="23" x2="22" y2="35"/>
    <path d="M11 35 a6 4.5 0 0 0 12 0" fill="#e6c65c" fill-opacity="0.22"/>
    <line x1="47" y1="23" x2="42" y2="35"/><line x1="47" y1="23" x2="52" y2="35"/>
    <path d="M41 35 a6 4.5 0 0 0 12 0" fill="#e6c65c" fill-opacity="0.22"/>
    <line x1="25" y1="47" x2="39" y2="47"/>
    <line x1="27" y1="50" x2="37" y2="50"/>
  </g>
</svg>
"""

# ---------------------------------------------------------------------------
# Theme / CSS (plain string — NOT an f-string; CSS uses braces)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700;800;900&display=swap');

:root {
  --navy:#0f2a4a; --navy-2:#1c4a7e; --gold:#c9a227; --gold-2:#e6c65c;
  --bg:#eaf0f6; --card:#ffffff; --ink:#1c2733; --muted:#5b6b7d;
  --ring:rgba(15,42,74,.08);
}

html, body, [class*="css"] { font-family:'Inter',system-ui,sans-serif; }
[data-testid="stAppViewContainer"] { background:
   radial-gradient(1100px 480px at 100% -8%, rgba(28,74,126,.08), transparent),
   radial-gradient(900px 400px at -10% 0%, rgba(201,162,39,.06), transparent),
   var(--bg); }
[data-testid="stHeader"] { background:transparent; }
.block-container { padding-top:2.7rem; padding-bottom:3.5rem; max-width:1220px;
   animation:fadeInUp .5s ease both; }
.block-container p { line-height:1.65; color:var(--ink); }
.block-container li { line-height:1.6; }
h1,h2,h3,h4 { font-family:'Playfair Display',Georgia,serif; color:var(--navy); letter-spacing:.2px; }

@keyframes fadeInUp { from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;} }
@keyframes floatIn { from{opacity:0; transform:translateY(18px) scale(.985);} to{opacity:1; transform:none;} }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(192,57,43,.4);} 70%{box-shadow:0 0 0 12px rgba(192,57,43,0);} 100%{box-shadow:0 0 0 0 rgba(192,57,43,0);} }

/* ---- Hero (fully rounded, never clipped) ---- */
.hero { position:relative; overflow:hidden; border-radius:20px; padding:2.1rem 2.3rem;
  background:linear-gradient(120deg, #0c223d 0%, var(--navy) 45%, var(--navy-2) 100%);
  color:#fff; box-shadow:0 18px 46px rgba(15,42,74,.30); margin:.2rem 0 1.4rem;
  animation:floatIn .6s ease both; }
.hero:after { content:""; position:absolute; right:-70px; top:-70px; width:250px; height:250px;
  border-radius:50%; background:radial-gradient(circle, rgba(201,162,39,.30), transparent 70%); }
.hero:before { content:""; position:absolute; left:-50px; bottom:-90px; width:220px; height:220px;
  border-radius:50%; background:radial-gradient(circle, rgba(255,255,255,.05), transparent 70%); }
.hero .eyebrow { text-transform:uppercase; letter-spacing:3px; font-size:.72rem; color:var(--gold-2); font-weight:700; }
.hero h1 { color:#fff; margin:.35rem 0 .2rem; font-size:2.15rem; line-height:1.12; font-weight:800; }
.hero .accent { width:70px; height:4px; border-radius:3px; background:var(--gold); margin:.7rem 0 .8rem; }
.hero p { color:rgba(255,255,255,.9); margin:0; font-size:1.02rem; max-width:820px; }
.hero .badges { margin-top:1rem; display:flex; gap:.5rem; flex-wrap:wrap; }
.hero .badge { background:rgba(255,255,255,.1); border:1px solid rgba(230,198,92,.35); color:#fff;
  padding:.3rem .75rem; border-radius:999px; font-size:.78rem; font-weight:600; }

/* ---- Cards ---- */
.lca-card { background:var(--card); border:1px solid var(--ring); border-radius:16px;
  padding:1.2rem 1.3rem; box-shadow:0 8px 22px rgba(15,42,74,.07);
  transition:transform .2s ease, box-shadow .2s ease; animation:floatIn .5s ease both; height:100%; }
.lca-card:hover { transform:translateY(-4px); box-shadow:0 18px 36px rgba(15,42,74,.14); }
.lca-card .ic { font-size:1.9rem; }
.lca-card h3 { margin:.5rem 0 .35rem; font-size:1.2rem; }

/* ---- Stat tiles ---- */
.stat { background:var(--card); border:1px solid var(--ring); border-radius:16px; padding:1.1rem 1.2rem;
  box-shadow:0 8px 22px rgba(15,42,74,.07); position:relative; overflow:hidden; height:100%;
  transition:transform .2s ease; animation:floatIn .5s ease both; }
.stat:hover { transform:translateY(-4px); }
.stat .v { font-family:'Playfair Display',serif; font-size:2rem; font-weight:800; color:var(--navy); line-height:1; }
.stat .l { color:var(--muted); font-size:.85rem; margin-top:.35rem; font-weight:500; }
.stat .ic { position:absolute; right:.9rem; top:.7rem; font-size:1.3rem; opacity:.55; }
.stat.gold { border-top:3px solid var(--gold); }

/* ---- Metrics -> cards ---- */
[data-testid="stMetric"] { background:var(--card); border:1px solid var(--ring);
  border-radius:16px; padding:1rem 1.15rem; box-shadow:0 8px 22px rgba(15,42,74,.07); transition:transform .2s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-3px); }
[data-testid="stMetricValue"] { color:var(--navy); font-weight:700; }

/* ---- Findings ---- */
.finding-card { background:var(--card); border:1px solid var(--ring); border-left:5px solid #ccc;
  border-radius:13px; padding:.85rem 1.1rem; margin-bottom:.75rem; box-shadow:0 5px 16px rgba(15,42,74,.05);
  transition:transform .18s ease; animation:floatIn .45s ease both; }
.finding-card:hover { transform:translateX(3px); }
.sev-tag { color:#fff; font-size:.68rem; font-weight:800; letter-spacing:.5px; padding:.12rem .5rem; border-radius:5px; margin-right:.5rem; }
.sev-high-pulse { animation:pulse 2.2s infinite; }
.evidence { background:rgba(15,42,74,.05); border-radius:7px; padding:.45rem .65rem; font-size:.8rem;
  font-family:ui-monospace,Menlo,monospace; color:#42505f; margin-top:.45rem; border-left:3px solid var(--gold); overflow-wrap:anywhere; }

/* ---- Pills / tags ---- */
.pill { display:inline-block; padding:.28rem .7rem; border-radius:999px; font-weight:700; font-size:.8rem; color:#fff; }
.tag { display:inline-block; padding:.2rem .6rem; border-radius:8px; font-size:.75rem;
  background:rgba(28,74,126,.1); color:var(--navy-2); margin:.14rem; font-weight:600; }

/* ---- Workflow steps ---- */
.step { background:var(--card); border:1px solid var(--ring); border-radius:16px; padding:1.15rem 1.2rem; height:100%;
  box-shadow:0 8px 22px rgba(15,42,74,.06); animation:floatIn .5s ease both; transition:transform .2s ease; }
.step:hover { transform:translateY(-4px); }
.step .num { display:inline-flex; align-items:center; justify-content:center; width:34px; height:34px; border-radius:50%;
  background:linear-gradient(135deg,var(--navy),var(--navy-2)); color:var(--gold-2); font-weight:800; font-family:'Playfair Display',serif; }
.step h4 { margin:.6rem 0 .3rem; }
.step p { color:var(--muted); font-size:.9rem; margin:0; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { background:linear-gradient(190deg,#123457 0%, var(--navy) 55%, #0a1d34 100%);
  border-right:1px solid rgba(201,162,39,.28); }
section[data-testid="stSidebar"] * { color:#dbe6f2; }
.brand { display:flex; align-items:center; gap:.7rem; padding:.2rem 0 .1rem; }
.brand .name { font-family:'Playfair Display',serif; color:#fff; font-size:1.12rem; font-weight:800; line-height:1.05; }
.brand .name small { display:block; color:var(--gold-2); font-size:.62rem; letter-spacing:2.5px; text-transform:uppercase; font-family:'Inter',sans-serif; font-weight:700; margin-top:2px; }
section[data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.12); margin:.7rem 0; }
section[data-testid="stSidebar"] [role="radiogroup"] { gap:.15rem; }
section[data-testid="stSidebar"] [role="radiogroup"] label { padding:.55rem .8rem; border-radius:11px; margin:.1rem 0;
  transition:background .2s, transform .15s, border-color .2s; border-left:3px solid transparent; cursor:pointer; }
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(255,255,255,.08); transform:translateX(3px); }
section[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:rgba(201,162,39,.16); border-left-color:var(--gold); }
section[data-testid="stSidebar"] [role="radiogroup"] label p { font-weight:600; font-size:.95rem; }
.side-stat { background:rgba(255,255,255,.06); border:1px solid rgba(201,162,39,.22); border-radius:11px;
  padding:.5rem .75rem; font-size:.82rem; margin-top:.45rem; }
.side-foot { color:rgba(219,230,242,.55); font-size:.7rem; margin-top:1rem; line-height:1.5; }

/* ---- Buttons ---- */
.stButton>button, .stDownloadButton>button { border-radius:11px; font-weight:600; border:1px solid var(--ring); transition:all .18s ease; }
.stButton>button:hover, .stDownloadButton>button:hover { transform:translateY(-2px); border-color:var(--gold); box-shadow:0 8px 18px rgba(15,42,74,.14); }
.stButton>button[kind="primary"] { background:var(--navy); border-color:var(--navy); }
.stButton>button[kind="primary"]:hover { background:var(--navy-2); }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap:.3rem; }
.stTabs [data-baseweb="tab"] { border-radius:11px 11px 0 0; padding:.4rem .9rem; }
.stTabs [aria-selected="true"] { background:rgba(28,74,126,.1); color:var(--navy); }

/* ---- Flashcards ---- */
.flip-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(258px,1fr)); gap:1.1rem; }
.flip-card { perspective:1300px; height:210px; }
.flip-inner { position:relative; width:100%; height:100%; transition:transform .75s cubic-bezier(.4,.2,.2,1); transform-style:preserve-3d; }
.flip-card:hover .flip-inner { transform:rotateY(180deg); }
.flip-front, .flip-back { position:absolute; inset:0; backface-visibility:hidden; border-radius:17px; padding:1.15rem;
  display:flex; flex-direction:column; justify-content:center; box-shadow:0 10px 26px rgba(15,42,74,.16); }
.flip-front { background:linear-gradient(140deg,var(--navy),var(--navy-2)); color:#fff; }
.flip-front .q { font-family:'Playfair Display',serif; font-size:1.08rem; font-weight:700; }
.flip-front .cat { position:absolute; top:.75rem; right:.85rem; font-size:.62rem; color:var(--gold-2); text-transform:uppercase; letter-spacing:1.5px; font-weight:800; }
.flip-front .hint { position:absolute; bottom:.75rem; left:1.15rem; font-size:.7rem; color:rgba(255,255,255,.6); }
.flip-back { background:var(--card); color:var(--ink); transform:rotateY(180deg); border:1px solid var(--gold); font-size:.9rem; line-height:1.48; overflow:auto; }

/* ---- Article reader ---- */
.reader-head { background:linear-gradient(120deg,var(--navy),var(--navy-2)); color:#fff; border-radius:16px;
  padding:1.6rem 1.9rem; position:relative; overflow:hidden; box-shadow:0 12px 30px rgba(15,42,74,.22);
  margin-bottom:1.3rem; animation:floatIn .5s ease both; }
.reader-head:after { content:""; position:absolute; right:-40px; top:-40px; width:160px; height:160px; border-radius:50%;
  background:radial-gradient(circle,rgba(201,162,39,.3),transparent 70%); }
.reader-head .ico { font-size:2rem; }
.reader-head h2 { color:#fff; margin:.35rem 0 .3rem; }
.reader-head .meta { color:var(--gold-2); font-size:.82rem; font-weight:600; }
/* Article body: elegant single-column typography on the page */
.article-wrap { max-width:820px; }
.article-wrap p, .article-wrap li { font-size:1rem; line-height:1.75; color:#26333f; }
.article-wrap h4 { margin-top:1.4rem; color:var(--navy); }
.article-wrap p:first-of-type:first-letter { font-family:'Playfair Display',serif; font-size:2.6rem;
  font-weight:800; color:var(--navy-2); float:left; line-height:.8; margin:.15rem .5rem 0 0; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-thumb { background:rgba(15,42,74,.28); border-radius:6px; }
::-webkit-scrollbar-thumb:hover { background:rgba(15,42,74,.45); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def hero(eyebrow: str, title: str, subtitle: str, badges: list[str] | None = None) -> None:
    badge_html = ""
    if badges:
        badge_html = "<div class='badges'>" + "".join(
            f"<span class='badge'>{safe(b)}</span>" for b in badges
        ) + "</div>"
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{safe(eyebrow)}</div>
          <h1>{safe(title)}</h1>
          <div class="accent"></div>
          <p>{safe(subtitle)}</p>
          {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_tile(value, label: str, icon: str = "", gold: bool = False) -> str:
    cls = "stat gold" if gold else "stat"
    return (f"<div class='{cls}'><div class='ic'>{safe(icon)}</div>"
            f"<div class='v'>{safe(value)}</div><div class='l'>{safe(label)}</div></div>")


def _theme_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#1c2733"),
        title_font=dict(family="Playfair Display, serif", color=NAVY, size=16),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


def risk_gauge(score: int, level: str) -> go.Figure:
    color = LEVEL_COLORS.get(level, NAVY)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "/100", "font": {"size": 30, "color": NAVY}},
        gauge={"axis": {"range": [0, 100], "tickcolor": SLATE},
               "bar": {"color": color, "thickness": 0.3}, "borderwidth": 0,
               "steps": [{"range": [0, 25], "color": "rgba(30,132,73,.18)"},
                         {"range": [25, 55], "color": "rgba(214,137,16,.18)"},
                         {"range": [55, 100], "color": "rgba(192,57,43,.18)"}],
               "threshold": {"line": {"color": color, "width": 4}, "value": score}},
        title={"text": f"<b>{level} risk</b>"}))
    return _theme_fig(fig, 250)


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
                 title="Findings by severity", color=list(data.keys()),
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


@st.cache_resource(show_spinner=False)
def _search_index(cache_key: int) -> SemanticSearchIndex:
    return SemanticSearchIndex.from_clauses(db.get_all_clauses())


def _md_to_html(md: str) -> str:
    """Convert the small markdown subset used by articles into HTML, escaping
    first so the output is safe to render. Supports bold-only subheads,
    bullet lists, and paragraphs."""
    import re

    def inline(t: str) -> str:
        t = safe(t.strip())
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)

    out: list[str] = []
    para: list[str] = []
    bullets: list[str] = []

    def flush_para():
        if para:
            out.append(f"<p>{' '.join(inline(x) for x in para)}</p>")
            para.clear()

    def flush_bullets():
        if bullets:
            out.append("<ul>" + "".join(f"<li>{inline(b)}</li>" for b in bullets) + "</ul>")
            bullets.clear()

    for raw in md.strip().splitlines():
        line = raw.strip()
        if not line:
            flush_para(); flush_bullets(); continue
        if line.startswith("- "):
            flush_para(); bullets.append(line[2:]); continue
        m = re.fullmatch(r"\*\*(.+?)\*\*:?", line)
        if m:  # a bold-only line acts as a subheading
            flush_para(); flush_bullets()
            out.append(f"<h4>{inline('**' + m.group(1) + '**')}</h4>")
            continue
        flush_bullets(); para.append(line)
    flush_para(); flush_bullets()
    return "".join(out)


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    metrics = ml_classifier.model_metrics()
    hero(
        "Contract Intelligence Platform",
        "Review contracts in minutes, not hours",
        "Lexalytics reads your agreements the way a diligence team would — surfacing the "
        "clauses, obligations, dates and dollar figures that matter, scoring risk, and making "
        "every contract searchable. Faster first-pass review, consistent coverage, better decisions.",
        badges=["⚡ Seconds per contract", "🎯 41 clause types", "🔒 100% private & local"],
    )

    contracts = db.list_contracts()

    if not contracts:
        # Impact stats
        acc = f"{metrics['test_accuracy']:.0%}" if metrics else "76%"
        ncls = metrics["n_classes"] if metrics else 41
        st.markdown("<div style='height:.2rem'></div>", unsafe_allow_html=True)
        cols = st.columns(4)
        cols[0].markdown(stat_tile(acc, "Clause-classifier accuracy", "🎯", gold=True), unsafe_allow_html=True)
        cols[1].markdown(stat_tile(ncls, "Clause categories detected", "🏷️"), unsafe_allow_html=True)
        cols[2].markdown(stat_tile("6", "Signals per contract", "🧭"), unsafe_allow_html=True)
        cols[3].markdown(stat_tile("0", "Bytes sent to the cloud", "🔒"), unsafe_allow_html=True)

        st.markdown("### From upload to insight in three steps")
        steps = [
            ("1", "📄 Upload", "Drop in a PDF or Word contract. Text is extracted and cleaned automatically."),
            ("2", "🤖 Analyze", "Clauses are classified, key terms extracted, and risks scored — instantly."),
            ("3", "📊 Decide", "Read a plain-English summary, review ranked risks, and export a report."),
        ]
        scols = st.columns(3)
        for col, (n, t, d) in zip(scols, steps):
            col.markdown(
                f"<div class='step'><span class='num'>{n}</span><h4>{safe(t)}</h4><p>{safe(d)}</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("### What it solves")
        wcols = st.columns(3)
        solves = [
            ("⚖️", "Cut review time", "Turn hours of manual reading into a minutes-long, structured first pass."),
            ("🛑", "Catch hidden risk", "Flag uncapped liability, one-sided rights, auto-renewals — and missing protections."),
            ("🔎", "Never lose a clause", "Search every contract you've reviewed to find precedents and compare terms."),
        ]
        for col, (ic, t, d) in zip(wcols, solves):
            col.markdown(
                f"<div class='lca-card'><div class='ic'>{safe(ic)}</div><h3>{safe(t)}</h3>"
                f"<div style='color:var(--muted)'>{safe(d)}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.info("▶ Open **Analyze Contract** in the sidebar and try the sample in `sample_contracts/`.")
        return

    # ---- Populated: portfolio analytics ----
    df = pd.DataFrame(contracts)
    avg_risk = int(df["risk_score"].dropna().mean()) if df["risk_score"].notna().any() else 0
    high = int((df["risk_level"] == "High").sum())
    total_clauses = int(df["num_clauses"].sum())
    cols = st.columns(4)
    cols[0].markdown(stat_tile(len(df), "Contracts analyzed", "📚", gold=True), unsafe_allow_html=True)
    cols[1].markdown(stat_tile(f"{avg_risk}/100", "Average risk score", "📈"), unsafe_allow_html=True)
    cols[2].markdown(stat_tile(high, "High-risk contracts", "🛑"), unsafe_allow_html=True)
    cols[3].markdown(stat_tile(total_clauses, "Clauses indexed", "🏷️"), unsafe_allow_html=True)

    st.markdown("### Portfolio insights")
    left, right = st.columns(2)
    with left:
        lvl = df["risk_level"].value_counts()
        if not lvl.empty:
            fig = px.pie(values=lvl.values, names=lvl.index, hole=0.55,
                         title="Risk distribution", color=lvl.index, color_discrete_map=LEVEL_COLORS)
            st.plotly_chart(_theme_fig(fig, 330), use_container_width=True)
    with right:
        types = [c["clause_type"] for c in db.get_all_clauses() if c.get("clause_type")]
        if types:
            s = pd.Series(types).value_counts().head(10).reset_index()
            s.columns = ["Clause Type", "Count"]
            fig = px.bar(s, x="Count", y="Clause Type", orientation="h",
                         title="Most common clauses across library", color_discrete_sequence=[GOLD])
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(_theme_fig(fig, 330), use_container_width=True)

    st.markdown("### Recent contracts")
    st.dataframe(df[["id", "filename", "upload_date", "num_clauses", "risk_level", "risk_score"]].head(8),
                 use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Analyze
# ---------------------------------------------------------------------------
def page_analyze(use_ml: bool) -> None:
    hero("Document Analysis", "Analyze a Contract",
         "Upload a PDF or Word document to extract clauses, key terms and risks. "
         "Files are processed locally and are never stored unless you save them.")

    uploaded = st.file_uploader(
        f"Drop a contract here — PDF or DOCX, up to {MAX_UPLOAD_BYTES // (1024*1024)} MB",
        type=["pdf", "docx"], key="uploader")
    if uploaded is None:
        st.info("Upload a contract to begin, or try `sample_contracts/sample_services_agreement.docx`.")
        return

    data = uploaded.getvalue()
    check = validate_upload(uploaded.name, data)
    if not check.ok:
        st.error(f"Upload rejected: {check.reason}")
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
        st.warning("This PDF looks like a scanned image with little extractable text, so results may be sparse.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File type", result.meta.get("file_type", "—"))
    c2.metric("Sections", len(result.clauses))
    c3.metric("Clauses classified", len(result.detected_clauses))
    c4.metric("Risk", f"{result.risk.level} · {result.risk.score}/100")

    tabs = st.tabs(["📋 Overview", "⚠️ Risk", "📑 Clauses", "🏷️ Key Terms", "📄 Full Text", "⬇️ Export & Save"])

    with tabs[0]:
        st.markdown("#### Executive summary")
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
                    f"{evidence}</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.caption("Every detected section in document order. Expand to read the text.")
        for c in result.clauses:
            bits = []
            if c.section_number:
                bits.append(f"§{c.section_number}")
            bits.append(c.heading or "(no heading)")
            bits.append(f"→ {c.clause_type} ({c.confidence:.0%}, {c.method})" if c.clause_type else "→ unclassified")
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
                               file_name=f"{result.filename}_analysis.json", mime="application/json", use_container_width=True)
            st.download_button("⬇️ Download report (HTML)", data=build_html_report(result),
                               file_name=f"{result.filename}_report.html", mime="text/html", use_container_width=True)
        with col_b:
            if st.session_state.get("saved_id"):
                st.success(f"Saved to library (contract #{st.session_state['saved_id']}).")
            if st.button("💾 Save to library", use_container_width=True, type="primary"):
                cid = db.save_contract(result.filename, result.clauses, risk_score=result.risk.score,
                                       risk_level=result.risk.level, summary=result.overview,
                                       entities=result.entities.to_dict(), full_text=result.clean_text)
                st.session_state["saved_id"] = cid
                st.rerun()


# ---------------------------------------------------------------------------
# Page: Library
# ---------------------------------------------------------------------------
def page_library() -> None:
    hero("Your Repository", "Contract Library",
         "Every contract you've saved, with its clauses, key terms and risk profile — all searchable.")
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
    hero("Knowledge Retrieval", "Semantic Clause Search",
         "Search similar clauses across every contract in your library — find precedents, compare terms, reuse language.")
    count = db.count_contracts()
    if count == 0:
        st.info("Your library is empty. Analyze and save some contracts first.")
        return

    index = _search_index(count)
    if index.is_empty:
        st.info("No clauses available to search yet.")
        return

    st.caption(f"Searching {index.size} clauses across {count} contract(s).")
    raw_query = st.text_input("Search clauses", placeholder="e.g. limitation of liability, termination for convenience…")
    top_k = st.slider("Results to show", 1, 20, 5)

    examples = ["limitation of liability", "confidential information", "governing law jurisdiction", "termination notice"]
    for col, ex in zip(st.columns(len(examples)), examples):
        if col.button(ex, use_container_width=True):
            raw_query = ex

    query = clean_query(raw_query)
    if not query:
        return

    results = index.search(query, top_k=top_k)
    if not results:
        st.warning("No matching clauses found. Try different wording.")
        return
    for r in results:
        st.markdown(
            f"<div class='lca-card' style='margin-bottom:.8rem'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;gap:.5rem'>"
            f"<div><strong>{safe(r.heading or '(no heading)')}</strong> "
            f"<span class='tag'>{safe(r.clause_type or 'unclassified')}</span></div>"
            f"<span class='pill' style='background:{NAVY_2}'>match {r.score:.0%}</span></div>"
            f"<div style='color:var(--muted); font-size:.8rem; margin:.3rem 0'>📄 {safe(r.filename)}</div>"
            f"<div style='color:var(--ink)'>{clean_display(r.text, 400)}</div></div>",
            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Learn
# ---------------------------------------------------------------------------
def page_learn() -> None:
    hero("Knowledge Center", "Contracts 101",
         "Build fluency in contract language — interactive flashcards and in-depth articles on the "
         "clauses that matter and how to read risk.")

    mode = st.radio("section", ["🃏 Flashcards", "📰 Articles"], horizontal=True, label_visibility="collapsed")

    if mode.endswith("Flashcards"):
        cats = ["All"] + flashcard_categories()
        chosen = st.selectbox("Filter by topic", cats)
        cards = [c for c in FLASHCARDS if chosen == "All" or c.category == chosen]
        st.caption(f"{len(cards)} card(s) · hover a card to reveal the answer.")
        html = "<div class='flip-grid'>"
        for c in cards:
            html += ("<div class='flip-card'><div class='flip-inner'>"
                     f"<div class='flip-front'><div class='cat'>{safe(c.category)}</div>"
                     f"<div class='q'>{safe(c.front)}</div><div class='hint'>hover to flip ↻</div></div>"
                     f"<div class='flip-back'>{safe(c.back)}</div></div></div>")
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
        return

    # Articles: single-column reader (no ragged grid, no wasted space)
    titles = [a.title for a in ARTICLES]
    choice = st.selectbox("Choose an article", titles)
    art = next(a for a in ARTICLES if a.title == choice)
    st.markdown(
        f"<div class='reader-head'><div class='ico'>{safe(art.icon)}</div>"
        f"<h2>{safe(art.title)}</h2><div class='meta'>⏱ {art.read_minutes} min read · {safe(art.summary)}</div></div>"
        f"<div class='article-wrap'>{_md_to_html(art.body)}</div>",
        unsafe_allow_html=True)
    st.caption("Browse more articles from the selector above.")


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def page_about() -> None:
    hero("About the Platform", "How it works",
         "Lexalytics runs an open, private analysis pipeline — from document parsing to clause "
         "classification, risk scoring and search — with no external services.")

    metrics = ml_classifier.model_metrics()
    if metrics:
        st.markdown("#### Model performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
        m2.metric("Macro-F1", f"{metrics['test_macro_f1']:.2f}")
        m3.metric("Clause types", metrics["n_classes"])
        m4.metric("Training examples", f"{metrics.get('n_train','—'):,}" if isinstance(metrics.get('n_train'), int) else "—")
        with st.expander("Per-clause accuracy (F1)"):
            pcf = pd.DataFrame(sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1], reverse=True),
                               columns=["Clause type", "F1"])
            st.dataframe(pcf, use_container_width=True, hide_index=True)

    st.markdown("#### The pipeline")
    pcols = st.columns(3)
    stages = [
        ("📥", "Ingest & clean", "PDF and Word documents are parsed and normalized into clean text."),
        ("🤖", "Classify & extract", "A machine-learning model labels 41 clause types; rules pull parties, dates, money and law."),
        ("📊", "Score & summarize", "Risk is scored from clauses and language; plain-English summaries are generated."),
    ]
    for col, (ic, t, d) in zip(pcols, stages):
        col.markdown(f"<div class='step'><div style='font-size:1.6rem'>{safe(ic)}</div><h4>{safe(t)}</h4><p>{safe(d)}</p></div>",
                     unsafe_allow_html=True)

    st.markdown("#### Clause types recognized")
    st.markdown("".join(f"<span class='tag'>{safe(k.replace('_',' ').title())}</span>" for k in sorted(CLAUSE_KEYWORDS)),
                unsafe_allow_html=True)

    st.info("⚖️ Lexalytics provides automated, informational analysis to accelerate review. "
            "It is not legal advice and does not replace a qualified attorney.")


# ---------------------------------------------------------------------------
# Sidebar / router
# ---------------------------------------------------------------------------
PAGES = {
    "🏛️ Dashboard": "dashboard",
    "📄 Analyze Contract": "analyze",
    "📚 Contract Library": "library",
    "🔎 Semantic Search": "search",
    "🎓 Learn": "learn",
    "ℹ️ About": "about",
}


def main() -> None:
    with st.sidebar:
        st.markdown(
            f"<div class='brand'>{LOGO_SVG}"
            f"<div class='name'>Lexalytics<small>Contract Intelligence</small></div></div>",
            unsafe_allow_html=True)
        st.divider()
        page_label = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()

        ml_ok = ml_classifier.is_available()
        use_ml = st.toggle("Use ML classifier", value=ml_ok, disabled=not ml_ok,
                           help="Falls back to the keyword baseline if off or unavailable.")
        st.markdown(f"<div class='side-stat'>{'🟢 AI model active' if ml_ok else '🟡 Keyword mode'}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='side-stat'>📚 {db.count_contracts()} contracts saved</div>", unsafe_allow_html=True)
        st.markdown("<div class='side-foot'>Runs privately on your machine.<br>Informational analysis — not legal advice.</div>",
                    unsafe_allow_html=True)

    page = PAGES[page_label]
    {"dashboard": page_dashboard, "analyze": lambda: page_analyze(use_ml), "library": page_library,
     "search": page_search, "learn": page_learn, "about": page_about}[page]()


if __name__ == "__main__":
    main()
