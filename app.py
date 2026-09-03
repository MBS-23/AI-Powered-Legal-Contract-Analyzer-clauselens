"""
app.py — ClauseLens · Legal Contract Analyzer.

Professional multi-page Streamlit UI over the analysis pipeline in src/.
Business logic lives in src/ (core.py orchestrates); this file is the
presentation layer: branding, theming, layout, charts, and state.
"""

from __future__ import annotations

import json
import re
import time

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
    page_title="ClauseLens · Contract Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme system — two fully-designed palettes sharing one ClauseLens brand.
# Every token below is emitted as a CSS variable AND read by Plotly charts,
# so the whole UI (HTML + charts + report) switches consistently.
# ---------------------------------------------------------------------------
THEMES = {
    "dark": {
        "bg": "#071525", "bg2": "#0B1F35", "card": "#10283A", "card2": "#143344",
        "teal": "#185A63", "gold": "#D6B35A", "gold2": "#E8CA78",
        "ink": "#F5F3ED", "muted": "#A8B7C4", "ring": "rgba(214,179,90,.16)", "line": "rgba(168,183,196,.14)",
        "success": "#5FBF8F", "warning": "#D9A441", "danger": "#C96868",
        "shadow": "rgba(0,0,0,.42)",
        "hero_from": "#071627", "hero_mid": "#0c2a3a", "hero_to": "#185A63",
        "hero_text": "#ffffff", "hero_p": "rgba(245,243,237,.82)", "hero_badge": "rgba(18,58,70,.55)",
        "hero_grid": "rgba(168,183,196,.06)", "hero_glow": "rgba(214,179,90,.28)",
        "sidebar_from": "#0a2230", "sidebar_to": "#061320", "sidebar_text": "#cddbe4",
        "panel_from": "#143344", "panel_to": "#185A63", "input_bg": "#10283A",
        "stat_grad": "#0b1d2e", "on_risk": "#0a1622",
    },
    "light": {
        "bg": "#F5F3ED", "bg2": "#ECEAE3", "card": "#FFFFFF", "card2": "#FAF9F5",
        "teal": "#185A63", "gold": "#B89232", "gold2": "#C79A3C",
        "ink": "#102235", "muted": "#526574", "ring": "rgba(11,31,53,.10)", "line": "rgba(11,31,53,.08)",
        "success": "#27845A", "warning": "#A87516", "danger": "#B64D4D",
        "shadow": "rgba(11,31,53,.10)",
        "hero_from": "#FBFAF6", "hero_mid": "#eef3f1", "hero_to": "#dde9e7",
        "hero_text": "#0B1F35", "hero_p": "#3a4a5a", "hero_badge": "rgba(11,31,53,.05)",
        "hero_grid": "rgba(11,31,53,.05)", "hero_glow": "rgba(184,146,50,.22)",
        "sidebar_from": "#FFFFFF", "sidebar_to": "#ECEAE3", "sidebar_text": "#28384a",
        "panel_from": "#123A46", "panel_to": "#185A63", "input_bg": "#FFFFFF",
        "stat_grad": "#FAF9F5", "on_risk": "#ffffff",
    },
}


def _theme_root(theme: str) -> str:
    """Emit a :root{} block of CSS variables for the given theme."""
    p = THEMES[theme]
    return ":root{" + "".join(f"--{k.replace('_','-')}:{v};" for k, v in p.items()) + "}"


def PAL() -> dict:
    return THEMES[st.session_state.get("theme", "dark")]


# Semantic risk colours as CSS vars for inline HTML (theme-aware automatically).
SEV_VAR = {"high": "var(--danger)", "medium": "var(--warning)", "low": "var(--muted)"}
LEVEL_VAR = {"High": "var(--danger)", "Medium": "var(--warning)", "Low": "var(--success)"}


def _level_colors() -> dict:
    p = PAL()
    return {"High": p["danger"], "Medium": p["warning"], "Low": p["success"]}


def _sev_colors() -> dict:
    p = PAL()
    return {"high": p["danger"], "medium": p["warning"], "low": p["muted"]}

# ---------------------------------------------------------------------------
# Icon set (clean line icons — no emoji)
# ---------------------------------------------------------------------------
ICONS = {
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "search": '<circle cx="11" cy="11" r="7"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1.4"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15 14"/>',
    "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><line x1="12" y1="9" x2="12" y2="14"/><line x1="12" y1="17.5" x2="12.01" y2="17.5"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/>',
    "chart": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11.5 14.5 15.5 10"/>',
    "compass": '<circle cx="12" cy="12" r="9"/><polygon points="16 8 10.5 10.5 8 16 13.5 13.5 16 8"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
}


def icon(name: str, size: int = 20, color: str = "var(--gold2)", sw: float = 1.9) -> str:
    # Colour applied via `style` (not the stroke attribute) so CSS vars resolve
    # and the icon follows the active theme.
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align:middle;stroke:{color}">{ICONS.get(name, "")}</svg>')


# Refined app-icon style emblem: rounded square, gold balance-scale glyph.
LOGO_SVG = """
<svg viewBox="0 0 48 48" width="42" height="42" xmlns="http://www.w3.org/2000/svg" aria-label="ClauseLens">
  <defs>
    <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#20507f"/><stop offset="1" stop-color="#0e2540"/>
    </linearGradient>
  </defs>
  <rect x="2" y="2" width="44" height="44" rx="12" fill="url(#lg)" stroke="#c9a227" stroke-width="1.5"/>
  <g stroke="#e6c65c" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
    <line x1="24" y1="13" x2="24" y2="35"/>
    <circle cx="24" cy="12" r="2" fill="#e6c65c" stroke="none"/>
    <line x1="12" y1="17" x2="36" y2="17"/>
    <line x1="12" y1="17" x2="8" y2="26"/><line x1="12" y1="17" x2="16" y2="26"/>
    <path d="M7 26 a5 3.4 0 0 0 10 0" fill="#e6c65c" fill-opacity="0.2"/>
    <line x1="36" y1="17" x2="32" y2="26"/><line x1="36" y1="17" x2="40" y2="26"/>
    <path d="M31 26 a5 3.4 0 0 0 10 0" fill="#e6c65c" fill-opacity="0.2"/>
    <line x1="18" y1="35" x2="30" y2="35"/>
  </g>
</svg>
"""

# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800;900&display=swap');

/* :root tokens (dark defaults) are overridden per-theme by _theme_root() in main(). */
:root {
  --bg:#071525; --bg2:#0B1F35; --card:#10283A; --card2:#143344; --teal:#185A63;
  --gold:#D6B35A; --gold2:#E8CA78; --ink:#F5F3ED; --muted:#A8B7C4;
  --ring:rgba(214,179,90,.16); --line:rgba(168,183,196,.14);
  --success:#5FBF8F; --warning:#D9A441; --danger:#C96868;
  --hero-from:#071627; --hero-mid:#0c2a3a; --hero-to:#185A63; --hero-text:#ffffff;
  --hero-p:rgba(245,243,237,.82); --hero-badge:rgba(18,58,70,.55); --hero-grid:rgba(168,183,196,.06);
  --hero-glow:rgba(214,179,90,.28); --sidebar-from:#0a2230; --sidebar-to:#061320; --sidebar-text:#cddbe4;
  --panel-from:#143344; --panel-to:#185A63; --input-bg:#10283A; --stat-grad:#0b1d2e; --on-risk:#0a1622;
}
html, body, [class*="css"] { font-family:'Manrope',system-ui,sans-serif; color:var(--ink); }
[data-testid="stAppViewContainer"], .stApp { background:
   radial-gradient(1200px 520px at 100% -10%, var(--hero-glow), transparent),
   radial-gradient(900px 420px at -10% 0%, var(--hero-glow), transparent),
   var(--bg); }
[data-testid="stHeader"] { background:transparent; }
/* smooth cross-theme transition */
.stApp, section[data-testid="stSidebar"], .hero, .lca-card, .stat, .step, [data-testid="stMetric"],
.finding-card, .flip-front, .flip-back, .reader-head, [data-testid="stFileUploaderDropzone"],
.stButton>button, input, textarea, .tag, .side-stat, [data-testid="stExpander"] {
  transition: background-color .28s ease, color .28s ease, border-color .28s ease, box-shadow .28s ease; }
.block-container { padding-top:2.6rem; padding-bottom:3.5rem; max-width:1240px; animation:fadeInUp .5s ease both; }
.block-container p, .block-container li { line-height:1.65; color:var(--ink); }
h1,h2,h3,h4 { font-family:'Playfair Display',Georgia,serif; color:var(--ink); letter-spacing:.2px; }
h2 { font-size:1.35rem; margin:1.9rem 0 .7rem; }
hr { border-color:var(--line); }
a { color:var(--gold2); }

@keyframes fadeInUp { from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;} }
@keyframes floatIn { from{opacity:0; transform:translateY(18px) scale(.985);} to{opacity:1; transform:none;} }
@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(201,104,104,.5);} 70%{box-shadow:0 0 0 12px rgba(201,104,104,0);} 100%{box-shadow:0 0 0 0 rgba(201,104,104,0);} }
@keyframes sheen { 0%{background-position:-380px 0;} 100%{background-position:380px 0;} }
/* Staggered reveal of cards laid out in columns */
[data-testid="stHorizontalBlock"] [data-testid="column"] { animation:floatIn .5s ease both; }
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(2){ animation-delay:.06s; }
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(3){ animation-delay:.12s; }
[data-testid="stHorizontalBlock"] [data-testid="column"]:nth-child(4){ animation-delay:.18s; }

/* ---- Hero ---- */
.hero { position:relative; overflow:hidden; border-radius:22px; padding:2.2rem 2.4rem;
  background:linear-gradient(125deg,var(--hero-from) 0%, var(--hero-mid) 55%, var(--hero-to) 140%);
  border:1px solid var(--ring); box-shadow:0 22px 55px var(--shadow);
  margin:.2rem 0 1.5rem; animation:floatIn .6s ease both; }
.hero:before { content:""; position:absolute; inset:0; opacity:.5;
  background-image:linear-gradient(var(--hero-grid) 1px,transparent 1px),linear-gradient(90deg,var(--hero-grid) 1px,transparent 1px);
  background-size:34px 34px; mask-image:radial-gradient(600px 300px at 85% 0%, #000, transparent 75%); }
.hero:after { content:""; position:absolute; right:-70px; top:-70px; width:270px; height:270px; border-radius:50%;
  background:radial-gradient(circle, var(--hero-glow), transparent 70%); }
.hero > * { position:relative; z-index:1; }
.hero .eyebrow { text-transform:uppercase; letter-spacing:3.5px; font-size:.7rem; color:var(--gold2); font-weight:800; }
.hero h1 { color:var(--hero-text); margin:.4rem 0 .2rem; font-size:2.25rem; line-height:1.1; font-weight:800; }
.hero h1 em { font-style:italic; color:var(--gold2); }
.hero .accent { width:72px; height:4px; border-radius:3px; background:var(--gold); margin:.75rem 0 .85rem; }
.hero p { color:var(--hero-p); margin:0; font-size:1.03rem; max-width:840px; }
.hero .badges { margin-top:1.1rem; display:flex; gap:.55rem; flex-wrap:wrap; }
.hero .badge { display:inline-flex; align-items:center; gap:.45rem; background:var(--hero-badge);
  border:1px solid var(--ring); color:var(--hero-text); padding:.36rem .85rem; border-radius:999px; font-size:.78rem; font-weight:600;
  backdrop-filter:blur(4px); transition:transform .2s ease, border-color .2s ease; }
.hero .badge:hover { transform:translateY(-2px); border-color:var(--gold); }

/* ---- Cards ---- */
.lca-card { background:var(--card); border:1px solid var(--ring); border-radius:18px; padding:1.3rem 1.35rem;
  box-shadow:0 10px 26px rgba(0,0,0,.28); transition:transform .22s ease, box-shadow .22s ease, border-color .22s ease; height:100%; }
.lca-card:hover { transform:translateY(-5px); box-shadow:0 22px 44px rgba(0,0,0,.4); border-color:rgba(214,179,90,.4); }
.lca-card .ic { width:46px; height:46px; border-radius:13px; display:flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg, rgba(24,90,99,.5), rgba(214,179,90,.16)); border:1px solid var(--ring); margin-bottom:.5rem; }
.lca-card h3 { margin:.5rem 0 .35rem; font-size:1.18rem; color:var(--ink); }

/* ---- Stat tiles ---- */
.stat { background:linear-gradient(180deg, var(--card), var(--stat-grad)); border:1px solid var(--ring); border-radius:18px;
  padding:1.15rem 1.25rem; box-shadow:0 10px 26px rgba(0,0,0,.28); position:relative; overflow:hidden;
  min-height:118px; display:flex; flex-direction:column; justify-content:center; transition:transform .22s ease, border-color .22s ease; }
.stat:hover { transform:translateY(-5px); border-color:rgba(214,179,90,.4); }
.stat .v { font-family:'Playfair Display',serif; font-size:2.1rem; font-weight:800; color:var(--ink); line-height:1; }
.stat .v.sm { font-size:1.5rem; letter-spacing:.3px; }
.stat .l { color:var(--muted); font-size:.84rem; margin-top:.45rem; font-weight:500; }
.stat .ic { position:absolute; right:1rem; top:1rem; opacity:.9; }
.stat.gold { border-top:2px solid var(--gold); }

/* ---- Streamlit metric widgets ---- */
[data-testid="stMetric"] { background:var(--card); border:1px solid var(--ring); border-radius:16px; padding:1rem 1.15rem;
  box-shadow:0 10px 26px rgba(0,0,0,.26); transition:transform .2s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-3px); }
[data-testid="stMetricValue"] { color:var(--ink); font-weight:800; }
[data-testid="stMetricLabel"] { color:var(--muted); }

/* ---- Findings ---- */
.finding-card { background:var(--card); border:1px solid var(--ring); border-left:5px solid #555; border-radius:14px;
  padding:.9rem 1.1rem; margin-bottom:.75rem; box-shadow:0 8px 20px rgba(0,0,0,.25); transition:transform .18s ease; animation:floatIn .45s ease both; }
.finding-card:hover { transform:translateX(4px); }
.finding-card strong { color:var(--ink); }
.sev-tag { color:#fff; font-size:.66rem; font-weight:800; letter-spacing:.5px; padding:.14rem .55rem; border-radius:6px; margin-right:.5rem; }
.sev-high-pulse { animation:pulse 2.2s infinite; }
.evidence { background:var(--card2); border-radius:8px; padding:.5rem .7rem; font-size:.8rem;
  font-family:ui-monospace,Menlo,monospace; color:var(--muted); margin-top:.5rem; border-left:3px solid var(--gold); overflow-wrap:anywhere; }

/* ---- Pills / tags ---- */
.pill { display:inline-block; padding:.3rem .75rem; border-radius:999px; font-weight:800; font-size:.78rem; color:#0a1622; }
.tag { display:inline-block; padding:.22rem .62rem; border-radius:8px; font-size:.74rem; background:rgba(24,90,99,.4);
  color:var(--gold2); margin:.14rem; font-weight:600; border:1px solid var(--ring); }
.risk-box { text-align:center; }
.risk-box .lbl { font-size:.68rem; letter-spacing:1.6px; color:var(--muted); text-transform:uppercase; margin-bottom:.4rem; }
.risk-box .val { display:inline-block; padding:.55rem 1.15rem; border-radius:13px; color:#fff; font-weight:800; font-size:1.05rem; box-shadow:0 8px 20px var(--shadow); }

/* ---- Workflow steps ---- */
.step { background:var(--card); border:1px solid var(--ring); border-radius:18px; padding:1.2rem 1.25rem; height:100%;
  box-shadow:0 10px 26px rgba(0,0,0,.26); transition:transform .22s ease, border-color .22s ease; }
.step:hover { transform:translateY(-5px); border-color:rgba(214,179,90,.4); }
.step .num { display:inline-flex; align-items:center; justify-content:center; width:36px; height:36px; border-radius:50%;
  background:linear-gradient(135deg,var(--teal),var(--card2)); color:var(--gold2); font-weight:800; font-family:'Playfair Display',serif; border:1px solid var(--ring); }
.step h4 { margin:.65rem 0 .3rem; color:var(--ink); } .step p { color:var(--muted); font-size:.9rem; margin:0; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { background:linear-gradient(195deg,var(--sidebar-from) 0%, var(--sidebar-to) 100%); border-right:1px solid var(--ring); }
section[data-testid="stSidebar"] * { color:var(--sidebar-text); }
.brand { display:flex; align-items:center; gap:.7rem; padding:.2rem 0 .1rem; }
.brand .name { font-family:'Playfair Display',serif; color:var(--ink); font-size:1.22rem; font-weight:800; line-height:1.05; }
.brand .name small { display:block; color:var(--gold2); font-size:.6rem; letter-spacing:2.6px; text-transform:uppercase; font-family:'Manrope',sans-serif; font-weight:700; margin-top:3px; }
section[data-testid="stSidebar"] hr { border-color:rgba(143,161,174,.16); margin:.7rem 0; }
section[data-testid="stSidebar"] .stButton>button { justify-content:flex-start !important; text-align:left !important;
  border:none; background:transparent; color:var(--sidebar-text); font-weight:600; border-left:3px solid transparent; border-radius:10px;
  padding:.58rem .95rem; box-shadow:none; white-space:nowrap; overflow:hidden; gap:.7rem !important; transition:all .18s ease; }
section[data-testid="stSidebar"] .stButton>button > * { justify-content:flex-start !important; }
section[data-testid="stSidebar"] .stButton>button p,
section[data-testid="stSidebar"] .stButton>button div,
section[data-testid="stSidebar"] .stButton>button span { white-space:nowrap; margin:0; text-align:left !important; }
section[data-testid="stSidebar"] .stButton>button:hover { background:rgba(24,90,99,.28); transform:translateX(3px); color:var(--ink); }
section[data-testid="stSidebar"] .stButton>button[kind="primary"] { background:linear-gradient(90deg,rgba(214,179,90,.22),rgba(24,90,99,.18)); color:var(--ink); border-left-color:var(--gold); }
.side-stat { display:flex; align-items:center; gap:.5rem; background:rgba(18,58,70,.4); border:1px solid rgba(214,179,90,.2);
  border-radius:11px; padding:.55rem .75rem; font-size:.82rem; margin-top:.5rem; }

/* ---- Buttons ---- */
.stButton>button, .stDownloadButton>button { border-radius:12px; font-weight:700; background:var(--card2); color:var(--ink);
  border:1px solid var(--ring); transition:all .18s ease; }
.stButton>button:hover, .stDownloadButton>button:hover { transform:translateY(-2px); border-color:var(--gold);
  box-shadow:0 8px 22px rgba(214,179,90,.22); color:#fff; }
.stButton>button[kind="primary"] { background:linear-gradient(135deg,var(--gold),#c79f45); border:none; color:#0a1622; }
.stButton>button[kind="primary"]:hover { box-shadow:0 10px 26px rgba(214,179,90,.4); filter:brightness(1.05); }

/* ---- Inputs / widgets ---- */
[data-testid="stTextInput"] input, [data-baseweb="select"] > div, [data-testid="stNumberInput"] input,
textarea, [data-baseweb="input"] { background:var(--card) !important; border-color:var(--line) !important; color:var(--ink) !important; }
[data-testid="stTextInput"] input:focus { border-color:var(--gold) !important; box-shadow:0 0 0 2px rgba(214,179,90,.25) !important; }
[data-testid="stWidgetLabel"] p, label p { color:var(--muted) !important; }
/* keep radio / checkbox / slider text legible in both themes */
.stRadio label p, .stCheckbox label p, [data-baseweb="radio"] div, .stSlider label,
[data-testid="stMarkdownContainer"] p { color:var(--ink); }
[data-baseweb="popover"] li, [data-baseweb="menu"] li { color:var(--ink); }
/* Native headings/captions follow the theme (config textColor is fixed-dark, so
   without this they stay off-white and vanish on the light background). */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5,
[data-testid="stHeading"], [data-testid="stHeadingWithActionElements"] *,
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 { color:var(--ink) !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color:var(--muted) !important; }
/* File-uploader browse button */
[data-testid="stFileUploaderDropzone"] button {
  background:var(--card2) !important; color:var(--ink) !important; border:1px solid var(--ring) !important; }
[data-testid="stFileUploaderDropzone"] button:hover { border-color:var(--gold) !important; }

/* ---- File uploader — premium drop zone ---- */
[data-testid="stFileUploaderDropzone"] { background:linear-gradient(180deg,var(--card),var(--card2));
  border:2px dashed rgba(214,179,90,.4); border-radius:16px; padding:2rem 1.5rem; transition:all .25s ease; }
[data-testid="stFileUploaderDropzone"]:hover { border-color:var(--gold); background:linear-gradient(180deg,var(--card2),var(--card));
  box-shadow:0 0 0 4px rgba(214,179,90,.12), 0 14px 34px var(--shadow); transform:translateY(-2px); }
[data-testid="stFileUploaderDropzone"] * { color:var(--ink) !important; }
[data-testid="stFileUploaderDropzone"] small { color:var(--muted) !important; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] { gap:.3rem; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius:11px 11px 0 0; padding:.45rem 1rem; color:var(--muted); }
.stTabs [aria-selected="true"] { background:rgba(24,90,99,.35); color:var(--gold2); }

/* ---- Expander / status / dataframe ---- */
[data-testid="stExpander"] { background:var(--card); border:1px solid var(--ring); border-radius:14px; }
[data-testid="stExpander"] summary:hover { color:var(--gold2); }
[data-testid="stDataFrame"] { border:1px solid var(--ring); border-radius:12px; }

/* ---- Themed data tables (used instead of st.dataframe so they follow theme) ---- */
.table-wrap { max-height:440px; overflow:auto; border:1px solid var(--ring); border-radius:13px; box-shadow:0 8px 22px var(--shadow); }
table.lca-table { width:100%; border-collapse:collapse; font-size:.86rem; }
table.lca-table th { background:var(--card2); color:var(--gold2); text-align:left; padding:.55rem .8rem;
  border-bottom:1px solid var(--ring); font-weight:700; position:sticky; top:0; white-space:nowrap; }
table.lca-table td { padding:.5rem .8rem; border-bottom:1px solid var(--line); color:var(--ink); vertical-align:top; }
table.lca-table tr:last-child td { border-bottom:none; }
table.lca-table tbody tr:hover td { background:var(--card2); }

/* ---- Flashcards ---- */
.flip-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(258px,1fr)); gap:1.1rem; }
.flip-card { perspective:1300px; height:212px; }
.flip-inner { position:relative; width:100%; height:100%; transition:transform .75s cubic-bezier(.4,.2,.2,1); transform-style:preserve-3d; }
.flip-card:hover .flip-inner { transform:rotateY(180deg); }
.flip-front, .flip-back { position:absolute; inset:0; backface-visibility:hidden; border-radius:17px; padding:1.2rem;
  display:flex; flex-direction:column; justify-content:center; box-shadow:0 12px 30px rgba(0,0,0,.4); }
.flip-front { background:linear-gradient(140deg,var(--panel-from),var(--panel-to)); color:#fff; border:1px solid var(--ring); }
.flip-front .q { font-family:'Playfair Display',serif; font-size:1.08rem; font-weight:700; }
.flip-front .cat { position:absolute; top:.8rem; right:.9rem; font-size:.62rem; color:var(--gold2); text-transform:uppercase; letter-spacing:1.5px; font-weight:800; }
.flip-front .hint { position:absolute; bottom:.8rem; left:1.2rem; font-size:.7rem; color:rgba(245,243,237,.6); }
.flip-back { background:var(--card); color:var(--ink); transform:rotateY(180deg); border:1px solid var(--gold); font-size:.9rem; line-height:1.5; overflow:auto; }

/* ---- Article reader ---- */
.reader-head { background:linear-gradient(120deg,var(--panel-from),var(--panel-to)); color:#fff; border-radius:18px; padding:1.7rem 2rem;
  position:relative; overflow:hidden; box-shadow:0 16px 36px rgba(0,0,0,.4); border:1px solid var(--ring); margin-bottom:1.4rem; animation:floatIn .5s ease both; }
.reader-head:after { content:""; position:absolute; right:-40px; top:-40px; width:170px; height:170px; border-radius:50%;
  background:radial-gradient(circle,rgba(214,179,90,.3),transparent 70%); }
.reader-head h2 { color:#fff; margin:.35rem 0 .3rem; } .reader-head .meta { color:var(--gold2); font-size:.82rem; font-weight:600; }
.article-wrap { max-width:820px; }
.article-wrap p, .article-wrap li { font-size:1.02rem; line-height:1.8; color:var(--ink); }
.article-wrap h4 { margin-top:1.5rem; color:var(--gold2); }
.article-wrap strong { color:var(--ink); }
.article-wrap p:first-of-type:first-letter { font-family:'Playfair Display',serif; font-size:2.8rem; font-weight:800; color:var(--gold); float:left; line-height:.8; margin:.15rem .6rem 0 0; }

::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-thumb { background:rgba(214,179,90,.3); border-radius:6px; }
::-webkit-scrollbar-thumb:hover { background:rgba(214,179,90,.5); }

@media (prefers-reduced-motion: reduce) {
  *, .hero, .lca-card, .stat, .step, [data-testid="column"] { animation:none !important; transition:none !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def hero(eyebrow: str, title: str, subtitle: str, badges: list[tuple[str, str]] | None = None) -> None:
    badge_html = ""
    if badges:
        badge_html = "<div class='badges'>" + "".join(
            f"<span class='badge'>{icon(ic, 14, 'var(--gold2)')} {safe(txt)}</span>" for ic, txt in badges
        ) + "</div>"
    st.markdown(
        f"<div class='hero'><div class='eyebrow'>{safe(eyebrow)}</div>"
        f"<h1>{safe(title)}</h1><div class='accent'></div><p>{safe(subtitle)}</p>{badge_html}</div>",
        unsafe_allow_html=True)


def render_table(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Render a DataFrame as a theme-aware HTML table (st.dataframe can't follow
    our runtime light/dark theme). Values are escaped."""
    cols = columns or list(df.columns)
    head = "".join(f"<th>{safe(c.replace('_',' ').title())}</th>" for c in cols)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{safe(row[c])}</td>" for c in cols)
        rows.append(f"<tr>{cells}</tr>")
    st.markdown(
        f"<div class='table-wrap'><table class='lca-table'><thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>",
        unsafe_allow_html=True)


def stat_tile(value, label: str, icon_name: str, gold: bool = False, small: bool = False) -> str:
    cls = "stat gold" if gold else "stat"
    col = "var(--gold)" if gold else "var(--gold2)"
    vcls = "v sm" if small else "v"
    return (f"<div class='{cls}'><div class='ic'>{icon(icon_name, 22, col)}</div>"
            f"<div class='{vcls}'>{safe(value)}</div><div class='l'>{safe(label)}</div></div>")


def _theme_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    p = PAL()
    dark = st.session_state.get("theme", "dark") == "dark"
    grid = "rgba(168,183,196,.14)" if dark else "rgba(11,31,53,.09)"
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=48, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Manrope, sans-serif", color=p["ink"]),
                      title_font=dict(family="Playfair Display, serif", color=p["gold2"], size=16),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=dict(color=p["ink"])))
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, color=p["muted"])
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, color=p["muted"])
    return fig


def risk_gauge(score: int, level: str) -> go.Figure:
    p = PAL()
    dark = st.session_state.get("theme", "dark") == "dark"
    color = _level_colors().get(level, p["gold"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number={"suffix": "/100", "font": {"size": 32, "color": p["ink"]}},
        gauge={"axis": {"range": [0, 100], "tickcolor": p["muted"], "tickfont": {"color": p["muted"]}},
               "bar": {"color": color, "thickness": 0.32}, "borderwidth": 0,
               "bgcolor": "rgba(255,255,255,.04)" if dark else "rgba(11,31,53,.04)",
               "steps": [{"range": [0, 25], "color": "rgba(95,191,143,.22)"}, {"range": [25, 55], "color": "rgba(217,164,65,.22)"},
                         {"range": [55, 100], "color": "rgba(201,104,104,.22)"}],
               "threshold": {"line": {"color": color, "width": 4}, "value": score}},
        title={"text": f"<b style='color:{p['ink']}'>{level} risk</b>"}))
    return _theme_fig(fig, 250)


def clause_bar(detected) -> go.Figure | None:
    if not detected:
        return None
    counts = pd.Series([c.clause_type for c in detected]).value_counts().reset_index()
    counts.columns = ["Clause Type", "Count"]
    fig = px.bar(counts, x="Clause Type", y="Count", title="Clause types found", color_discrete_sequence=[PAL()["teal"]])
    fig.update_layout(xaxis_tickangle=-30)
    return _theme_fig(fig, 360)


def severity_donut(counts: dict) -> go.Figure | None:
    data = {k.title(): v for k, v in counts.items() if v}
    if not data:
        return None
    fig = px.pie(values=list(data.values()), names=list(data.keys()), hole=0.55, title="Findings by severity",
                 color=list(data.keys()), color_discrete_map={"High": _sev_colors()["high"], "Medium": _sev_colors()["medium"], "Low": _sev_colors()["low"]})
    fig.update_traces(textinfo="value")
    return _theme_fig(fig, 300)


def method_donut(clauses) -> go.Figure | None:
    labels = [("Smart model" if c.method == "ml" else "Keyword") for c in clauses if c.clause_type]
    if not labels:
        return None
    s = pd.Series(labels).value_counts()
    fig = px.pie(values=s.values, names=s.index, hole=0.55, title="How clauses were identified",
                 color_discrete_sequence=[PAL()["teal"], PAL()["gold"]])
    return _theme_fig(fig, 300)


@st.cache_resource(show_spinner=False)
def _search_index(contract_count: int, clause_count: int) -> SemanticSearchIndex:
    # Keyed on (contracts, clauses) so the index rebuilds whenever the library
    # changes — adding OR removing contracts, even if the count nets out.
    return SemanticSearchIndex.from_clauses(db.get_all_clauses())


def _md_to_html(md: str) -> str:
    """Convert the small markdown subset used by articles into safe HTML."""
    def inline(t: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe(t.strip()))

    out, para, bullets = [], [], []

    def flush_para():
        if para:
            out.append(f"<p>{' '.join(inline(x) for x in para)}</p>"); para.clear()

    def flush_bullets():
        if bullets:
            out.append("<ul>" + "".join(f"<li>{inline(b)}</li>" for b in bullets) + "</ul>"); bullets.clear()

    for raw in md.strip().splitlines():
        line = raw.strip()
        if not line:
            flush_para(); flush_bullets(); continue
        if line.startswith("- "):
            flush_para(); bullets.append(line[2:]); continue
        m = re.fullmatch(r"\*\*(.+?)\*\*:?", line)
        if m:
            flush_para(); flush_bullets(); out.append(f"<h4>{inline('**' + m.group(1) + '**')}</h4>"); continue
        flush_bullets(); para.append(line)
    flush_para(); flush_bullets()
    return "".join(out)


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    metrics = ml_classifier.model_metrics()
    hero("Contract Intelligence",
         "Review contracts in minutes, not hours",
         "ClauseLens reads an agreement the way an experienced review team would — pulling out the "
         "clauses, obligations, dates and figures that matter, scoring the risk, and making every "
         "contract searchable. Faster first-pass review, consistent coverage, and clearer decisions.",
         badges=[("zap", "Minutes, not hours"), ("layers", "41 clause types"), ("search", "Fully searchable")])

    contracts = db.list_contracts()

    if not contracts:
        acc = f"{metrics['test_accuracy']:.0%}" if metrics else "76%"
        ncls = metrics["n_classes"] if metrics else 41
        st.markdown("<div style='height:.2rem'></div>", unsafe_allow_html=True)
        cols = st.columns(4)
        cols[0].markdown(stat_tile(acc, "Clause recognition accuracy", "target", gold=True), unsafe_allow_html=True)
        cols[1].markdown(stat_tile(ncls, "Clause types recognized", "layers"), unsafe_allow_html=True)
        cols[2].markdown(stat_tile("Seconds", "Average analysis time", "clock", small=True), unsafe_allow_html=True)
        cols[3].markdown(stat_tile("0–100", "Risk scored on every contract", "shield", small=True), unsafe_allow_html=True)

        st.markdown("### From document to decision in three steps")
        steps = [("1", "Upload", "Add a PDF or Word contract. The text is read and organized for you."),
                 ("2", "Analyze", "Clauses are identified, key terms extracted, and risks scored — in seconds."),
                 ("3", "Decide", "Read a clear summary, review ranked risks, and share a polished report.")]
        for col, (n, t, d) in zip(st.columns(3), steps):
            col.markdown(f"<div class='step'><span class='num'>{n}</span><h4>{safe(t)}</h4><p>{safe(d)}</p></div>", unsafe_allow_html=True)

        st.markdown("### Why review teams choose ClauseLens")
        solves = [("clock", "Faster first-pass review", "Hours of careful reading become a structured review that takes minutes — without losing rigour."),
                  ("shield", "Risk surfaced, not missed", "One-sided terms, uncapped liability and auto-renewals are flagged — along with the protections that are absent."),
                  ("search", "Institutional memory", "Every reviewed contract stays searchable, so precedents and standard language are always a query away.")]
        for col, (ic, t, d) in zip(st.columns(3), solves):
            col.markdown(f"<div class='lca-card'><div class='ic'>{icon(ic, 22, 'var(--gold2)')}</div><h3>{safe(t)}</h3>"
                         f"<div style='color:var(--muted)'>{safe(d)}</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.info("Head to **Analyze Contract** and upload one of your agreements — watch ClauseLens reveal the clauses, key terms and risks in seconds.")
        return

    df = pd.DataFrame(contracts)
    avg_risk = int(df["risk_score"].dropna().mean()) if df["risk_score"].notna().any() else 0
    high = int((df["risk_level"] == "High").sum())
    total_clauses = int(df["num_clauses"].sum())
    cols = st.columns(4)
    cols[0].markdown(stat_tile(len(df), "Contracts reviewed", "file", gold=True), unsafe_allow_html=True)
    cols[1].markdown(stat_tile(f"{avg_risk}/100", "Average risk score", "chart"), unsafe_allow_html=True)
    cols[2].markdown(stat_tile(high, "High-risk contracts", "alert"), unsafe_allow_html=True)
    cols[3].markdown(stat_tile(total_clauses, "Clauses on file", "layers"), unsafe_allow_html=True)

    st.markdown("### Portfolio insights")
    left, right = st.columns(2)
    with left:
        lvl = df["risk_level"].value_counts()
        if not lvl.empty:
            fig = px.pie(values=lvl.values, names=lvl.index, hole=0.55, title="Risk across your contracts",
                         color=lvl.index, color_discrete_map=_level_colors())
            st.plotly_chart(_theme_fig(fig, 330), use_container_width=True)
    with right:
        types = [c["clause_type"] for c in db.get_all_clauses() if c.get("clause_type")]
        if types:
            s = pd.Series(types).value_counts().head(10).reset_index()
            s.columns = ["Clause Type", "Count"]
            fig = px.bar(s, x="Count", y="Clause Type", orientation="h", title="Most common clauses", color_discrete_sequence=[PAL()["gold"]])
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(_theme_fig(fig, 330), use_container_width=True)

    st.markdown("### Recent contracts")
    render_table(df.head(8), ["id", "filename", "upload_date", "num_clauses", "risk_level", "risk_score"])


# ---------------------------------------------------------------------------
# Page: Analyze
# ---------------------------------------------------------------------------
def page_analyze(use_ml: bool) -> None:
    hero("Document Review", "Analyze a Contract",
         "Upload a PDF or Word document to extract its clauses, key terms and risks. "
         "A contract is only added to your library when you choose to save it.")

    uploaded = st.file_uploader(f"Drop a contract here — PDF or Word, up to {MAX_UPLOAD_BYTES // (1024*1024)} MB",
                                type=["pdf", "docx"], key="uploader")
    if uploaded is None:
        st.info("Drop in one of your agreements above — ClauseLens reads it end to end and shows you the "
                "clauses, key terms, and exactly where the risk sits.")
        return

    data = uploaded.getvalue()
    check = validate_upload(uploaded.name, data)
    if not check.ok:
        st.error(f"That file couldn't be accepted: {check.reason}")
        return

    sig = (uploaded.name, uploaded.size, use_ml)
    if st.session_state.get("analysis_sig") != sig:
        try:
            with st.status("Analyzing contract…", expanded=True) as status:
                stages = ["Reading document", "Extracting text", "Identifying clauses",
                          "Extracting key terms", "Assessing risk"]
                for s in stages:
                    st.markdown(f"<span style='color:#8FA1AE'>›</span> {s}…", unsafe_allow_html=True)
                    time.sleep(0.16)
                st.session_state["analysis"] = analyze_upload(uploaded, use_ml=use_ml)
                st.markdown("<span style='color:#5FBF8F'>✓</span> Report ready", unsafe_allow_html=True)
                status.update(label="Analysis complete", state="complete", expanded=False)
            st.session_state["analysis_sig"] = sig
            st.session_state["saved_id"] = None
        except Exception as e:  # noqa: BLE001
            st.error(f"Sorry — this file couldn't be processed: {safe(e)}")
            return

    result: AnalysisResult = st.session_state["analysis"]
    if result.meta.get("likely_scanned"):
        st.warning("This looks like a scanned image with little readable text, so results may be limited.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Format", result.meta.get("file_type", "—"))
    c2.metric("Sections", len(result.clauses))
    c3.metric("Clauses identified", len(result.detected_clauses))
    # Level as the value (short — never truncates); score in the label.
    c4.metric(f"Risk · {result.risk.score}/100", result.risk.level)

    tabs = st.tabs(["Summary", "Risk", "Clauses", "Key Terms", "Full Text", "Export & Save"])

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
                    f"<div class='finding-card{pulse}' style='border-left-color:{SEV_VAR.get(f.severity)}'>"
                    f"<span class='sev-tag' style='background:{SEV_VAR.get(f.severity)}'>{safe(f.severity.upper())}</span>"
                    f"<strong>{safe(f.title)}</strong>"
                    f"<div style='margin:.3rem 0; color:var(--ink)'>{safe(f.detail)}</div>"
                    f"<div style='font-size:.88rem; color:var(--muted)'><em>Recommendation:</em> {safe(f.recommendation)}</div>"
                    f"{evidence}</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.caption("Every section in document order. Expand to read the text.")
        for c in result.clauses:
            bits = []
            if c.section_number:
                bits.append(f"§{c.section_number}")
            bits.append(c.heading or "(no heading)")
            bits.append(f"— {c.clause_type} ({c.confidence:.0%})" if c.clause_type else "— unclassified")
            with st.expander(" ".join(bits)):
                st.write(c.text or "_(empty section)_")
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
        st.text_area("Extracted text", result.clean_text, height=460)

    with tabs[5]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("Download data (JSON)", data=json.dumps(result.to_export_dict(), indent=2),
                               file_name=f"{result.filename}_analysis.json", mime="application/json", use_container_width=True)
            st.download_button("Download report (HTML)", data=build_html_report(result, theme=st.session_state.get("theme", "dark")),
                               file_name=f"{result.filename}_report.html", mime="text/html", use_container_width=True)
        with col_b:
            if st.session_state.get("saved_id"):
                st.success(f"Saved to your library (contract #{st.session_state['saved_id']}).")
            if st.button("Save to library", use_container_width=True, type="primary"):
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
        st.info("No contracts saved yet. Analyze a contract and choose **Save to library**.")
        return

    df = pd.DataFrame(contracts)
    render_table(df, ["id", "filename", "upload_date", "num_clauses", "risk_level", "risk_score"])

    ids = [c["id"] for c in contracts]
    selected = st.selectbox("Open a contract", ids, format_func=lambda i: next(c["filename"] for c in contracts if c["id"] == i))
    contract = db.get_contract(selected)
    if not contract:
        return

    top = st.columns([3, 1])
    with top[0]:
        st.subheader(contract["filename"])
        if contract.get("summary"):
            st.write(contract["summary"])
    with top[1]:
        lvl = contract.get("risk_level", "—")
        sc = contract.get("risk_score", "—")
        color = LEVEL_VAR.get(lvl, "var(--teal)")
        st.markdown(f"<div class='risk-box'><div class='lbl'>Overall risk</div>"
                    f"<div class='val' style='background:{color}'>{safe(lvl)} · {safe(sc)}/100</div></div>",
                    unsafe_allow_html=True)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        if st.button("Delete contract", use_container_width=True):
            db.delete_contract(selected)
            st.rerun()

    st.markdown("**Clauses**")
    clause_df = pd.DataFrame(contract["clauses"])
    if not clause_df.empty:
        clause_df = clause_df.copy()
        if "confidence" in clause_df:
            clause_df["confidence"] = clause_df["confidence"].map(lambda x: f"{float(x):.0%}" if pd.notna(x) else "—")
        render_table(clause_df, ["section_number", "heading", "clause_type", "confidence", "method"])


# ---------------------------------------------------------------------------
# Page: Search
# ---------------------------------------------------------------------------
def page_search() -> None:
    hero("Knowledge Retrieval", "Find Similar Clauses",
         "Search across every contract in your library to find precedents, compare terms and reuse language.")
    count = db.count_contracts()
    if count == 0:
        st.info("Your library is empty. Analyze and save some contracts first.")
        return
    index = _search_index(count, db.count_clauses())
    if index.is_empty:
        st.info("No clauses available to search yet.")
        return

    st.caption(f"Searching {index.size} clauses across {count} contract(s).")
    raw_query = st.text_input("Search", placeholder="e.g. limitation of liability, termination for convenience…", label_visibility="collapsed")
    top_k = st.slider("Results to show", 1, 20, 5)
    examples = ["limitation of liability", "confidential information", "governing law", "termination notice"]
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
            f"<div><strong>{safe(r.heading or '(no heading)')}</strong> <span class='tag'>{safe(r.clause_type or 'unclassified')}</span></div>"
            f"<span class='pill' style='background:var(--gold)'>match {r.score:.0%}</span></div>"
            f"<div style='color:var(--muted); font-size:.8rem; margin:.3rem 0; display:flex; align-items:center; gap:.35rem'>"
            f"{icon('file', 14, 'var(--muted)')} {safe(r.filename)}</div>"
            f"<div style='color:var(--ink)'>{clean_display(r.text, 400)}</div></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Learn
# ---------------------------------------------------------------------------
def page_learn() -> None:
    hero("Knowledge Center", "Contracts 101",
         "Build fluency in contract language — quick flashcards and in-depth articles on the clauses that "
         "matter and how to read risk.")
    mode = st.radio("section", ["Flashcards", "Articles"], horizontal=True, label_visibility="collapsed")

    if mode == "Flashcards":
        cats = ["All"] + flashcard_categories()
        chosen = st.selectbox("Filter by topic", cats)
        cards = [c for c in FLASHCARDS if chosen == "All" or c.category == chosen]
        st.caption(f"{len(cards)} card(s) · hover a card to reveal the answer.")
        html = "<div class='flip-grid'>"
        for c in cards:
            html += ("<div class='flip-card'><div class='flip-inner'>"
                     f"<div class='flip-front'><div class='cat'>{safe(c.category)}</div>"
                     f"<div class='q'>{safe(c.front)}</div><div class='hint'>hover to flip</div></div>"
                     f"<div class='flip-back'>{safe(c.back)}</div></div></div>")
        st.markdown(html + "</div>", unsafe_allow_html=True)
        return

    choice = st.selectbox("Choose an article", [a.title for a in ARTICLES])
    art = next(a for a in ARTICLES if a.title == choice)
    st.markdown(f"<div class='reader-head'><h2>{safe(art.title)}</h2>"
                f"<div class='meta'>{art.read_minutes} min read · {safe(art.summary)}</div></div>"
                f"<div class='article-wrap'>{_md_to_html(art.body)}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def page_about() -> None:
    hero("About ClauseLens", "How it works",
         "ClauseLens analyzes each contract end to end — reading the document, identifying its clauses, "
         "extracting key terms, scoring risk, and making everything searchable.")
    metrics = ml_classifier.model_metrics()
    if metrics:
        st.markdown("#### How accurate is it?")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Recognition accuracy", f"{metrics['test_accuracy']:.1%}")
        m2.metric("Balanced accuracy (F1)", f"{metrics['test_macro_f1']:.2f}")
        m3.metric("Clause types", metrics["n_classes"])
        m4.metric("Examples learned from", f"{metrics.get('n_train','—'):,}" if isinstance(metrics.get('n_train'), int) else "—")
        with st.expander("Accuracy by clause type"):
            pcf = pd.DataFrame(sorted(metrics["per_class_f1"].items(), key=lambda kv: kv[1], reverse=True),
                               columns=["Clause type", "Score"])
            render_table(pcf, ["Clause type", "Score"])

    st.markdown("#### The process")
    stages = [("file", "Read & organize", "PDF and Word documents are read and organized into clean, structured text."),
              ("cpu", "Identify & extract", "Each clause is recognized and the key terms — parties, dates, money, governing law — are pulled out."),
              ("chart", "Score & summarize", "Risk is scored from the clauses and their language, and a clear summary is written.")]
    for col, (ic, t, d) in zip(st.columns(3), stages):
        col.markdown(f"<div class='step'><div class='ic' style='width:44px;height:44px;border-radius:12px;display:flex;"
                     f"align-items:center;justify-content:center;background:linear-gradient(135deg,rgba(28,74,126,.12),rgba(201,162,39,.14))'>"
                     f"{icon(ic, 22, 'var(--teal)')}</div><h4>{safe(t)}</h4><p>{safe(d)}</p></div>", unsafe_allow_html=True)

    st.markdown("#### Clause types recognized")
    st.markdown("".join(f"<span class='tag'>{safe(k.replace('_',' ').title())}</span>" for k in sorted(CLAUSE_KEYWORDS)),
                unsafe_allow_html=True)
    st.info("ClauseLens provides automated analysis to speed up review. It is a decision-support "
            "tool and does not replace advice from a qualified attorney.")


# ---------------------------------------------------------------------------
# Sidebar / router
# ---------------------------------------------------------------------------
NAV = [
    ("Dashboard", "dashboard", ":material/dashboard:"),
    ("Analyze Contract", "analyze", ":material/upload_file:"),
    ("Contract Library", "library", ":material/folder_open:"),
    ("Find Clauses", "search", ":material/search:"),
    ("Learn", "learn", ":material/menu_book:"),
    ("About", "about", ":material/info:"),
]
ROUTES = {"dashboard": page_dashboard, "library": page_library, "search": page_search,
          "learn": page_learn, "about": page_about}


def main() -> None:
    if "nav" not in st.session_state:
        st.session_state.nav = "dashboard"
    # Theme: default from URL query param (persists across sessions), else dark.
    if "theme" not in st.session_state:
        qp = st.query_params.get("theme")
        st.session_state.theme = qp if qp in ("dark", "light") else "dark"

    # Inject the active theme's CSS variables (overrides the dark defaults).
    st.markdown(f"<style>{_theme_root(st.session_state.theme)}</style>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"<div class='brand'>{LOGO_SVG}<div class='name'>ClauseLens<small>Contract Analyzer</small></div></div>",
                    unsafe_allow_html=True)
        st.divider()
        for label, key, ic in NAV:
            active = st.session_state.nav == key
            if st.button(label, key=f"nav_{key}", icon=ic, use_container_width=True,
                         type="primary" if active else "secondary"):
                if not active:
                    st.session_state.nav = key
                    st.rerun()
        st.divider()
        ml_ok = ml_classifier.is_available()
        use_ml = st.toggle("Smart clause detection", value=ml_ok, disabled=not ml_ok,
                           help="Uses the trained model to recognize clauses; falls back to keywords if turned off.")
        st.markdown(f"<div class='side-stat'>{icon('check', 15, 'var(--gold2)')} {'Model ready' if ml_ok else 'Keyword mode'}</div>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='side-stat'>{icon('file', 15, 'var(--gold2)')} {db.count_contracts()} contracts saved</div>",
                    unsafe_allow_html=True)

        # ---- Theme toggle ----
        st.divider()
        st.markdown("<div style='font-size:.72rem;letter-spacing:2px;text-transform:uppercase;color:var(--muted);"
                    "font-weight:700;margin-bottom:.3rem'>Appearance</div>", unsafe_allow_html=True)
        cur = st.session_state.theme
        choice = st.radio("Theme", ["🌙  Dark", "☀️  Light"], index=0 if cur == "dark" else 1,
                          horizontal=True, label_visibility="collapsed", key="theme_radio")
        new_theme = "dark" if choice.startswith("🌙") else "light"
        if new_theme != cur:
            st.session_state.theme = new_theme
            st.query_params["theme"] = new_theme
            st.rerun()

    nav = st.session_state.nav
    if nav == "analyze":
        page_analyze(use_ml)
    else:
        ROUTES[nav]()


if __name__ == "__main__":
    main()
