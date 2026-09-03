"""
report.py
---------
Renders an AnalysisResult into a polished, standalone HTML report that a user
can download, share, or print to PDF. Everything (styles, logo) is inlined, so
the file opens correctly offline. All contract-derived values are HTML-escaped.
"""

from __future__ import annotations

import html
from datetime import datetime

NAVY = "#0f2a4a"
NAVY_2 = "#1c4a7e"
GOLD = "#c9a227"
_SEVERITY_COLORS = {"high": "#c0392b", "medium": "#d68910", "low": "#7d8a99"}
_LEVEL_COLORS = {"High": "#c0392b", "Medium": "#d68910", "Low": "#1e8449"}

# Inline brand emblem (matches the app sidebar logo).
_LOGO = """
<svg viewBox="0 0 48 48" width="40" height="40" style="vertical-align:middle">
  <defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#20507f"/><stop offset="1" stop-color="#0e2540"/></linearGradient></defs>
  <rect x="2" y="2" width="44" height="44" rx="12" fill="url(#rg)" stroke="#c9a227" stroke-width="1.5"/>
  <g stroke="#e6c65c" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round">
    <line x1="24" y1="13" x2="24" y2="35"/><circle cx="24" cy="12" r="2" fill="#e6c65c" stroke="none"/>
    <line x1="12" y1="17" x2="36" y2="17"/>
    <line x1="12" y1="17" x2="8" y2="26"/><line x1="12" y1="17" x2="16" y2="26"/>
    <path d="M7 26 a5 3.4 0 0 0 10 0" fill="#e6c65c" fill-opacity="0.2"/>
    <line x1="36" y1="17" x2="32" y2="26"/><line x1="36" y1="17" x2="40" y2="26"/>
    <path d="M31 26 a5 3.4 0 0 0 10 0" fill="#e6c65c" fill-opacity="0.2"/>
    <line x1="18" y1="35" x2="30" y2="35"/></g>
</svg>
"""


def _esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def _entities_rows(entities: dict) -> str:
    labels = {"parties": "Parties", "effective_date": "Effective date", "term": "Term",
              "governing_law": "Governing law", "notice_periods": "Notice periods",
              "monetary_amounts": "Monetary amounts", "dates": "Dates mentioned"}
    rows = []
    for key, label in labels.items():
        val = entities.get(key)
        if not val:
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        rows.append(f"<tr><th>{_esc(label)}</th><td>{_esc(val)}</td></tr>")
    return "\n".join(rows) or "<tr><td colspan='2' style='color:#8794a3'>No key terms extracted.</td></tr>"


def build_html_report(result) -> str:
    r = result
    risk = r.risk
    level_color = _LEVEL_COLORS.get(risk.level, NAVY)
    generated = datetime.now().strftime("%B %d, %Y · %H:%M")

    # Findings
    findings_html = []
    for f in risk.findings:
        color = _SEVERITY_COLORS.get(f.severity, "#333")
        evidence = f"<div class='evidence'>{_esc(f.evidence)}</div>" if f.evidence else ""
        findings_html.append(
            f"<div class='finding' style='border-left-color:{color}'>"
            f"<span class='sev' style='background:{color}'>{_esc(f.severity.upper())}</span>"
            f"<span class='ftitle'>{_esc(f.title)}</span>"
            f"<p>{_esc(f.detail)}</p>"
            f"<p class='rec'><strong>Recommendation:</strong> {_esc(f.recommendation)}</p>{evidence}</div>")
    findings_block = "\n".join(findings_html) or "<p class='muted'>No risk findings were raised.</p>"

    # Clause rows
    clause_rows = []
    for c in r.clauses:
        if not c.clause_type:
            continue
        conf = f"{c.confidence:.0%}"
        clause_rows.append(
            f"<tr><td>{_esc(c.section_number)}</td><td>{_esc(c.heading)}</td>"
            f"<td><span class='chip'>{_esc(c.clause_type.replace('_',' ').title())}</span></td>"
            f"<td style='text-align:right'>{conf}</td></tr>")
    clause_table = "\n".join(clause_rows) or "<tr><td colspan='4' class='muted'>No clauses classified.</td></tr>"

    counts = risk.counts

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Contract Analysis — {_esc(r.filename)}</title>
<style>
  :root {{ --navy:{NAVY}; --navy2:{NAVY_2}; --gold:{GOLD}; --ink:#1f2933; --muted:#7d8a99; --ring:#e4e9f0; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif; color:var(--ink);
         max-width:900px; margin:0 auto; padding:0 0 3rem; line-height:1.6; background:#f4f7fb; }}
  .sheet {{ background:#fff; }}
  header.hero {{ background:linear-gradient(120deg,#0c223d,var(--navy) 45%,var(--navy2));
    color:#fff; padding:2rem 2.4rem; position:relative; overflow:hidden; }}
  header.hero:after {{ content:""; position:absolute; right:-60px; top:-60px; width:200px; height:200px;
    border-radius:50%; background:radial-gradient(circle,rgba(201,162,39,.28),transparent 70%); }}
  .brandline {{ display:flex; align-items:center; gap:.7rem; }}
  .brandline .bn {{ font-weight:800; letter-spacing:.5px; font-size:1.05rem; }}
  .brandline .bn small {{ display:block; color:var(--gold); font-size:.62rem; letter-spacing:2.5px;
    text-transform:uppercase; font-weight:700; }}
  header .eyebrow {{ text-transform:uppercase; letter-spacing:3px; font-size:.68rem; color:#e6c65c; font-weight:700; margin-top:1.3rem; }}
  header h1 {{ font-size:1.7rem; margin:.2rem 0 .1rem; font-weight:800; }}
  header .meta {{ color:rgba(255,255,255,.8); font-size:.85rem; }}
  main {{ padding:1.6rem 2.4rem; }}
  h2 {{ font-size:1.15rem; color:var(--navy); margin:1.8rem 0 .6rem; padding-bottom:.35rem; border-bottom:2px solid #eef2f7; }}
  h2:first-child {{ margin-top:.4rem; }}
  .summary {{ background:#f7f9fc; border:1px solid var(--ring); border-left:4px solid var(--gold); border-radius:10px; padding:1rem 1.2rem; }}
  .riskrow {{ display:flex; gap:1rem; align-items:center; flex-wrap:wrap; }}
  .score {{ display:flex; flex-direction:column; align-items:center; justify-content:center; width:120px; height:120px;
    border-radius:50%; color:#fff; background:{level_color}; box-shadow:0 8px 20px rgba(15,42,74,.18); }}
  .score .n {{ font-size:2rem; font-weight:800; line-height:1; }} .score .u {{ font-size:.75rem; opacity:.85; }}
  .score .lv {{ font-size:.8rem; font-weight:700; margin-top:.2rem; letter-spacing:.5px; }}
  .sevchips {{ display:flex; gap:.5rem; flex-wrap:wrap; }}
  .sevchip {{ padding:.3rem .8rem; border-radius:999px; color:#fff; font-size:.8rem; font-weight:700; }}
  .finding {{ border:1px solid var(--ring); border-left:5px solid #ccc; border-radius:10px; padding:.8rem 1rem; margin:.6rem 0; background:#fff; }}
  .finding .sev {{ color:#fff; font-size:.66rem; font-weight:800; padding:.12rem .5rem; border-radius:5px; margin-right:.5rem; letter-spacing:.5px; }}
  .finding .ftitle {{ font-weight:700; color:var(--navy); }}
  .finding p {{ margin:.35rem 0; font-size:.92rem; }} .finding .rec {{ color:#42505f; font-size:.88rem; }}
  .evidence {{ background:#f7f9fc; border-left:3px solid var(--gold); border-radius:5px; padding:.4rem .6rem;
    font-size:.8rem; color:#52606d; margin-top:.4rem; font-family:ui-monospace,Menlo,monospace; overflow-wrap:anywhere; }}
  table {{ border-collapse:collapse; width:100%; margin-top:.5rem; font-size:.9rem; }}
  th, td {{ border:1px solid var(--ring); padding:.5rem .65rem; text-align:left; vertical-align:top; }}
  thead th {{ background:var(--navy); color:#fff; font-weight:600; }}
  .kv th {{ background:#f7f9fc; width:170px; color:var(--navy); }}
  .chip {{ background:rgba(28,74,126,.1); color:var(--navy2); padding:.12rem .5rem; border-radius:6px; font-size:.78rem; font-weight:600; }}
  .muted {{ color:var(--muted); }}
  footer {{ padding:1.2rem 2.4rem 0; color:var(--muted); font-size:.8rem; border-top:1px solid var(--ring); margin-top:1.5rem; }}
  @media print {{ body {{ background:#fff; }} .sheet {{ box-shadow:none; }} }}
</style></head>
<body><div class="sheet">
  <header class="hero">
    <div class="brandline">{_LOGO}<div class="bn">ClauseLens<small>Contract Analyzer</small></div></div>
    <div class="eyebrow">Contract Analysis Report</div>
    <h1>{_esc(r.filename)}</h1>
    <div class="meta">Generated {generated} · {len(r.detected_clauses)} clauses identified</div>
  </header>
  <main>
    <h2>Executive summary</h2>
    <div class="summary">{_esc(r.overview)}</div>

    <h2>Risk assessment</h2>
    <div class="riskrow">
      <div class="score"><div class="n">{risk.score}</div><div class="u">/ 100</div><div class="lv">{_esc(risk.level).upper()}</div></div>
      <div class="sevchips">
        <span class="sevchip" style="background:{_SEVERITY_COLORS['high']}">{counts.get('high',0)} High</span>
        <span class="sevchip" style="background:{_SEVERITY_COLORS['medium']}">{counts.get('medium',0)} Medium</span>
        <span class="sevchip" style="background:{_SEVERITY_COLORS['low']}">{counts.get('low',0)} Low</span>
      </div>
    </div>
    <div style="margin-top:1rem">{findings_block}</div>

    <h2>Key terms</h2>
    <table class="kv">{_entities_rows(r.entities.to_dict())}</table>

    <h2>Detected clauses</h2>
    <table>
      <thead><tr><th>#</th><th>Heading</th><th>Clause type</th><th style="text-align:right">Confidence</th></tr></thead>
      <tbody>{clause_table}</tbody>
    </table>

    <h2>Key points</h2>
    <p>{_esc(r.key_points)}</p>
  </main>
  <footer>
    Generated by <strong>ClauseLens · Contract Analyzer</strong>. This automated analysis is for
    informational purposes only and is not legal advice.
  </footer>
</div></body></html>"""
